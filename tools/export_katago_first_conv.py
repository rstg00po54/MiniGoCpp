#!/usr/bin/env python3
"""Extract one *actual* KataGo ONNX Conv node with real weights and ONNX CPU golden output.

Packet: 8-byte magic K10CV001, 12 little-endian u32:
H,W,Cin,Cout,KH,KW,padLeft,padRight,padTop,padBottom,outH,outW.
Then contiguous fp16 arrays: input NHWC, weights OHWI, bias Cout.
Finally fp32 ONNX reference output NHWC. This is one Conv, NOT KataGo 10b inference.
"""
import argparse
import copy
from pathlib import Path
import struct

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

    # ONNX Runtime 1.16 (Python 3.8) does not support this whole model's opset 20.
    # The first Conv takes InputSpatial directly: isolate this original Conv with
    # its exact FP32 initializers/attributes in an opset-19 reference graph.
    # We do NOT change/rewrite the actual 10b ONNX model.
    input_name, output_name = node.input[0], node.output[0]
    graph_inputs = {x.name: x for x in model.graph.input}
    if input_name not in graph_inputs:
        raise RuntimeError("First Conv input {} is not a model input; cannot isolate without upstream ops".format(input_name))
    vi = graph_inputs[input_name]
    if vi.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise RuntimeError("Unexpected first Conv input tensor type")
    dims = vi.type.tensor_type.shape.dim
    input_shape = [d.dim_value if d.HasField("dim_value") else
                   (1 if i == 0 else None) for i, d in enumerate(dims)]
    if len(input_shape) != 4 or any(not isinstance(d, int) or d <= 0 for d in input_shape):
        raise RuntimeError("First Conv needs static dimensions after setting batch=1: {}".format(input_shape))
    batch, ci, h, w = input_shape
    if batch != 1 or ci != ci_group:
        raise RuntimeError("First Conv channel/group mismatch: {}".format(input_shape))

    auto_pad = attrs.get("auto_pad", b"NOTSET")
    if isinstance(auto_pad, bytes):
        auto_pad = auto_pad.decode("ascii")
    pads = list(attrs.get("pads", [0, 0, 0, 0]))
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        total_h, total_w = kh - 1, kw - 1
        if auto_pad == "SAME_UPPER":
            pt, pl = total_h // 2, total_w // 2
        else:
            pt, pl = (total_h + 1) // 2, (total_w + 1) // 2
        pb, pr = total_h - pt, total_w - pl
    elif auto_pad == "VALID":
        pt = pl = pb = pr = 0
    elif auto_pad in ("NOTSET", "") and len(pads) == 4:
        pt, pl, pb, pr = map(int, pads)
    else:
        raise RuntimeError("Unsupported padding spec {}".format(attrs))
    oh, ow = h + pt + pb - kh + 1, w + pl + pr - kw + 1
    if oh <= 0 or ow <= 0:
        raise RuntimeError("Invalid first Conv output {}x{}".format(oh, ow))

    if len(node.input) >= 3 and node.input[2]:
        if node.input[2] not in initializers:
            raise RuntimeError("First Conv bias is not a constant initializer")
        bias = np.asarray(initializers[node.input[2]], dtype=np.float32)
    else:
        bias = np.zeros((cout,), dtype=np.float32)
    if bias.shape != (cout,):
        raise RuntimeError("Unexpected first Conv bias shape {}".format(bias.shape))

    original_inits = {x.name: x for x in model.graph.initializer}
    local_inits = [copy.deepcopy(original_inits[node.input[1]])]
    if len(node.input) >= 3 and node.input[2]:
        local_inits.append(copy.deepcopy(original_inits[node.input[2]]))
    conv_graph = helper.make_graph(
        [copy.deepcopy(node)], "katago_first_conv_reference",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, cout, oh, ow])],
        initializer=local_inits)
    conv_model = helper.make_model(conv_graph, opset_imports=[helper.make_opsetid("", 19)])
    # ORT 1.16.3 supports ONNX IR <= 9; ONNX 1.16 creates IR 10 by default.
    conv_model.ir_version = 9
    onnx.checker.check_model(conv_model)

    rng = np.random.default_rng(seed)
    input_nchw = rng.normal(0.0, 0.2, size=input_shape).astype(np.float32)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(conv_model.SerializeToString(), sess_options=opts,
                                   providers=["CPUExecutionProvider"])
    golden_nchw = session.run([output_name], {input_name: input_nchw})[0]
    if golden_nchw.shape != (1, cout, oh, ow):
        raise RuntimeError("Isolated Conv output shape mismatch {}".format(golden_nchw.shape))

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
    print("Reference: original Conv node + FP32 initializers, isolated ONNX opset 19 for Python 3.8")
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
