#!/usr/bin/env python3
"""Extract one *actual* KataGo ONNX Conv node with real weights and ONNX CPU golden output.

Packet: 8-byte magic K10CV001, 12 little-endian u32:
H,W,Cin,Cout,KH,KW,padLeft,padRight,padTop,padBottom,outH,outW.
Then contiguous fp16 arrays: input NHWC, weights OHWI, bias Cout.
Finally fp32 ONNX reference output NHWC. This is one Conv, NOT KataGo 10b inference.
"""
import argparse
import copy
import os
from pathlib import Path
import struct
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort


def attributes(node):
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


def create_case(model_path, out_path, seed=42):
    model = onnx.load(str(model_path), load_external_data=True)
    onnx.checker.check_model(model)
    initializers = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    convs = [n for n in model.graph.node if n.op_type == "Conv" and len(n.input) >= 2 and n.input[1] in initializers]
    if not convs:
        raise RuntimeError("No ONNX Conv with inline weight initializer. Conversion cannot proceed.")
    node = convs[0]
    attrs = attributes(node)
    weights = np.asarray(initializers[node.input[1]], dtype=np.float32)
    if weights.ndim != 4:
        raise RuntimeError("First Conv weights not [out,in,kH,kW]")
    cout, ci_group, kh, kw = weights.shape
    groups = int(attrs.get("group", 1))
    strides = list(attrs.get("strides", [1, 1]))
    dilations = list(attrs.get("dilations", [1, 1]))
    if groups != 1 or strides != [1, 1] or dilations != [1, 1]:
        raise RuntimeError("First Conv uses unsupported groups/strides/dilations: {},{},{}".format(groups, strides, dilations))
    if kh not in (1, 3, 5) or kw not in (1, 3, 5):
        raise RuntimeError("Unsupported first Conv kernel {}x{}".format(kh, kw))

    # Expose the first Conv's exact input and output as graph outputs.
    input_name, output_name = node.input[0], node.output[0]
    graph_inputs = {x.name: x for x in model.graph.input}
    graph_outputs = {x.name for x in model.graph.output}
    for name in [input_name, output_name]:
        if name not in graph_outputs and name not in graph_inputs:
            model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
        elif name in graph_inputs and name not in graph_outputs:
            model.graph.output.append(copy.deepcopy(graph_inputs[name]))

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), sess_options=opts, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    feeds = {}
    for inp in session.get_inputs():
        # KataGo dumponnx leaves the leading batch dimension symbolic ("batch").
        # Our Android ADB probes always use batch=1, like the RKNN comparison.
        dims = [1 if i == 0 and (d is None or isinstance(d, str)) else d
                for i, d in enumerate(inp.shape)]
        if any(not isinstance(d, int) or d < 1 for d in dims):
            raise RuntimeError("Dynamic/unknown model input shape {}: {}".format(inp.name, dims))
        if inp.type != "tensor(float)":
            raise RuntimeError("Unsupported ONNX input type {}: {}".format(inp.name, inp.type))
        if "Mask" in inp.name:
            feeds[inp.name] = np.ones(dims, dtype=np.float32)
        else:
            feeds[inp.name] = rng.normal(loc=0.0, scale=0.2, size=dims).astype(np.float32)

    input_nchw, golden_nchw = session.run([input_name, output_name], feeds)
    if input_nchw.ndim != 4 or golden_nchw.ndim != 4 or input_nchw.shape[0] != 1:
        raise RuntimeError("Unexpected Conv tensor shape input={} output={}".format(input_nchw.shape, golden_nchw.shape))
    _, ci, h, w = input_nchw.shape
    _, out_c, oh, ow = golden_nchw.shape
    if ci != ci_group or cout != out_c:
        raise RuntimeError("Channel mismatch: {} {} {} {}".format(ci, ci_group, cout, out_c))
    auto_pad = attrs.get("auto_pad", b"NOTSET")
    if isinstance(auto_pad, bytes):
        auto_pad = auto_pad.decode("ascii")
    pads = list(attrs.get("pads", [0, 0, 0, 0]))
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        total_h, total_w = (oh - 1 + kh - h), (ow - 1 + kw - w)
        if auto_pad == "SAME_UPPER":
            pt, pl = total_h // 2, total_w // 2
        else:
            pt, pl = (total_h + 1) // 2, (total_w + 1) // 2
        pb, pr = total_h - pt, total_w - pl
    elif auto_pad in ("VALID",):
        pt = pl = pb = pr = 0
    elif auto_pad in ("NOTSET", "") and len(pads) == 4:
        pt, pl, pb, pr = map(int, pads)
    else:
        raise RuntimeError("Unsupported padding spec {}".format(attrs))
    if (h + pt + pb - kh + 1, w + pl + pr - kw + 1) != (oh, ow):
        raise RuntimeError("Padding/out shape inconsistent: {}x{} -> {}x{}, pads {}".format(h, w, oh, ow, (pt,pl,pb,pr)))
    if len(node.input) >= 3 and node.input[2]:
        if node.input[2] not in initializers:
            raise RuntimeError("Dynamic Conv bias, unsupported")
        bias = np.asarray(initializers[node.input[2]], dtype=np.float32)
    else:
        bias = np.zeros((cout,), dtype=np.float32)
    if bias.shape != (cout,):
        raise RuntimeError("Unexpected bias shape {}".format(bias.shape))

    input_nhwc = np.ascontiguousarray(input_nchw.transpose(0, 2, 3, 1).astype("<f2"))
    weights_ohwi = np.ascontiguousarray(weights.transpose(0, 2, 3, 1).astype("<f2"))
    bias_fp16 = np.ascontiguousarray(bias.astype("<f2"))
    golden_nhwc = np.ascontiguousarray(golden_nchw.transpose(0, 2, 3, 1).astype("<f4"))
    header = struct.pack("<8s12I", b"K10CV001", h, w, ci, cout, kh, kw, pl, pr, pt, pb, oh, ow)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(input_nhwc.tobytes())
        f.write(weights_ohwi.tobytes())
        f.write(bias_fp16.tobytes())
        f.write(golden_nhwc.tobytes())
    print("KataGo ONNX first convolution:", node.name or "(unnamed)")
    print("ONNX tensors:", input_name, "->", output_name)
    print("Input NCHW:", input_nchw.shape, "weights OIHW:", weights.shape, "output NCHW:", golden_nchw.shape)
    print("Neuron packet: {} ({} bytes), NHWC fp16 input, OHWI fp16 weights, FP32 ONNX golden".format(out_path, out_path.stat().st_size))
    print("Kernel={}x{}, pads L/R/T/B={}/{}/{}/{}; float32 ONNX reference".format(kh, kw, pl, pr, pt, pb))
    print("Reference output: min={:.5g}, max={:.5g}, mean={:.5g}".format(
        golden_nhwc.min(), golden_nhwc.max(), golden_nhwc.mean()))
    print("This represents ONE Conv, NOT KataGo's entire residual network.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Original 10b_19x19.onnx")
    parser.add_argument("--out", required=True, help="Binary packet for Android Neuron Adapter")
    parser.add_argument("--seed", type=int, default=42)
    a = parser.parse_args()
    create_case(a.model, a.out, a.seed)


if __name__ == "__main__":
    main()
