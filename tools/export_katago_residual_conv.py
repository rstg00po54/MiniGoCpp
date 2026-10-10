#!/usr/bin/env python3
"""Export a REAL KataGo residual Conv as MDLA FP16 packet with ONNX CPU golden.

Includes all upstream operators in an isolated opset19/IR9 CPU reference subgraph,
using the original ONNX weights/attributes (does not alter the source model).
Supports rconv1/w1 and rconv1/w2 in the 10b 19x19 network.
"""
import argparse
import copy
import struct
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def export(model_file, out_file, name, seed):
    model = onnx.load(model_file, load_external_data=True)
    onnx.checker.check_model(model)
    g = model.graph
    byname = {n.name: (i, n) for i, n in enumerate(g.node)}
    if name not in ("rconv1/w1", "rconv1/w2"):
        raise ValueError("Only rconv1/w1 and rconv1/w2 are currently supported")
    if name not in byname:
        raise ValueError("KataGo node {} not found".format(name))
    index, node = byname[name]
    if node.op_type != "Conv" or len(node.input) < 2:
        raise ValueError("Selected node isn't Conv: {}".format(name))

    initializer_by_name = {x.name: x for x in g.initializer}
    original_inputs = {x.name: x for x in g.input}
    producer = {out: i for i, n in enumerate(g.node) for out in n.output if out}
    needed_nodes = set()

    def visit(value):
        if value in initializer_by_name or value in original_inputs:
            return
        if value not in producer:
            raise ValueError("Unresolved upstream tensor {}".format(value))
        i = producer[value]
        if i in needed_nodes:
            return
        needed_nodes.add(i)
        for inp in g.node[i].input:
            if inp:
                visit(inp)

    visit(node.output[0])
    ordered = [g.node[i] for i in range(index + 1) if i in needed_nodes]
    if ordered[-1].name != name:
        raise ValueError("Unexpected Conv topology")
    needed_initializers = {val for n in ordered for val in n.input if val in initializer_by_name}

    weights = np.asarray(numpy_helper.to_array(initializer_by_name[node.input[1]]), dtype=np.float32)
    if weights.ndim != 4:
        raise ValueError("Unexpected convolution weight rank: {}".format(weights.shape))
    co, ci, kh, kw = weights.shape
    attrs = {x.name: helper.get_attribute_value(x) for x in node.attribute}
    if int(attrs.get("group", 1)) != 1 or list(attrs.get("strides", [1, 1])) != [1, 1] or list(attrs.get("dilations", [1, 1])) != [1, 1]:
        raise ValueError("Unsupported Conv attrs: {}".format(attrs))
    pads = list(attrs.get("pads", [0, 0, 0, 0]))
    if len(pads) != 4 or attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
        raise ValueError("Unsupported Conv padding {}".format(attrs))
    pt, pl, pb, pr = [int(x) for x in pads]
    if len(node.input) >= 3 and node.input[2]:
        if node.input[2] not in initializer_by_name:
            raise ValueError("Dynamic bias unsupported")
        bias = np.asarray(numpy_helper.to_array(initializer_by_name[node.input[2]]), dtype=np.float32)
    else:
        bias = np.zeros(co, dtype=np.float32)
    if bias.shape != (co,):
        raise ValueError("Invalid bias shape {}".format(bias.shape))

    shapes = {}
    for vi in g.input:
        dims = [d.dim_value if d.HasField("dim_value") else (1 if i == 0 else None)
                for i, d in enumerate(vi.type.tensor_type.shape.dim)]
        if any(not isinstance(v, int) or v <= 0 for v in dims):
            raise ValueError("Unknown model input dimension: {} {}".format(vi.name, dims))
        if vi.type.tensor_type.elem_type != TensorProto.FLOAT:
            raise ValueError("Non-FP32 model input: {}".format(vi.name))
        shapes[vi.name] = dims
    _, _, h, w = shapes["InputSpatial"]
    oh, ow = h + pt + pb - kh + 1, w + pl + pr - kw + 1
    if oh < 1 or ow < 1 or h > 128 or w > 128:
        raise ValueError("Unexpected output shape: {},{}".format(oh, ow))
    input_tensor = node.input[0]
    output_tensor = node.output[0]

    # Keep exact original upstream ops and model constants; only lower the
    # isolated CPU reference model's ONNX opset stamp, not the source graph.
    graph = helper.make_graph(
        [copy.deepcopy(n) for n in ordered], "katago_residual_conv_reference_" + name.replace("/", "_"),
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, shape) for n, shape in shapes.items()],
        [helper.make_tensor_value_info(input_tensor, TensorProto.FLOAT, [1, ci, h, w]),
         helper.make_tensor_value_info(output_tensor, TensorProto.FLOAT, [1, co, oh, ow])],
        initializer=[copy.deepcopy(initializer_by_name[n]) for n in sorted(needed_initializers)])
    ref_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    ref_model.ir_version = 9
    onnx.checker.check_model(ref_model)
    rng = np.random.default_rng(seed)
    feeds = {n: (np.ones(s, dtype=np.float32) if n == "InputMask"
                 else rng.normal(0, 0.2, s).astype(np.float32)) for n, s in shapes.items()}
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(ref_model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    input_nchw, golden_nchw = session.run([input_tensor, output_tensor], feeds)
    if tuple(input_nchw.shape) != (1, ci, h, w) or tuple(golden_nchw.shape) != (1, co, oh, ow):
        raise ValueError("Unexpected intermediate shapes: {} {}".format(input_nchw.shape, golden_nchw.shape))
    packet = Path(out_file)
    packet.parent.mkdir(parents=True, exist_ok=True)
    with packet.open("wb") as f:
        f.write(struct.pack("<8s12I", b"K10CV001", h, w, ci, co, kh, kw, pl, pr, pt, pb, oh, ow))
        f.write(np.ascontiguousarray(input_nchw.transpose(0, 2, 3, 1), dtype="<f2").tobytes())
        f.write(np.ascontiguousarray(weights.transpose(0, 2, 3, 1), dtype="<f2").tobytes())
        f.write(np.ascontiguousarray(bias, dtype="<f2").tobytes())
        f.write(np.ascontiguousarray(golden_nchw.transpose(0, 2, 3, 1), dtype="<f4").tobytes())
    expected = 56 + 2*(h*w*ci + co*kh*kw*ci + co) + 4*(oh*ow*co)
    if packet.stat().st_size != expected:
        raise ValueError("Packet size mismatch: {} vs {}".format(packet.stat().st_size, expected))
    print("KATAGO_RESIDUAL_EXPORT node={} graph_nodes={} kernel={}x{} weights={} packet_bytes={}".format(
        name, len(ordered), kh, kw, weights.shape, packet.stat().st_size))
    print("KATAGO_RESIDUAL_EXPORT input={} output={} CPU_fp32_min={:.6f} max={:.6f}".format(
        input_nchw.shape, golden_nchw.shape, golden_nchw.min(), golden_nchw.max()))
    print("KATAGO_RESIDUAL_EXPORT reference=original upstream ONNX graph in isolated opset19/IR9")
    print("Note: this is one residual CONV; residual ADD and full block remain untested.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--node", default="rconv1/w1", choices=["rconv1/w1", "rconv1/w2"])
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    export(a.model, a.out, a.node, a.seed)


if __name__ == "__main__":
    main()
