#!/usr/bin/env python3
"""Convert KataGo v1.18 ONNX (19x19) to an RK3588 RKNN model.

Run on x86_64 Linux (RKNN-Toolkit2), NOT on RK3588.
Keep all three NN inputs in the order exported by KataGo.
"""
import argparse
from collections import Counter
from pathlib import Path

import onnx


def prepare(src, dst, opset):
    model = onnx.load(str(src))
    print("Source model: IR", model.ir_version, "opsets", [(o.domain, o.version) for o in model.opset_import], flush=True)
    print("Operators:", Counter(n.op_type for n in model.graph.node), flush=True)
    expected = {"InputSpatial": [1, 22, 19, 19],
                "InputGlobal": [1, 19, 1, 1],
                "InputMask": [1, 1, 19, 19]}
    names = [x.name for x in model.graph.input]
    if names != list(expected):
        raise RuntimeError("Unexpected input names/order: %r; expected %r" % (names, list(expected)))
    for x in model.graph.input:
        dims = x.type.tensor_type.shape.dim
        sizes = expected[x.name]
        if len(dims) != 4:
            raise RuntimeError("Expected 4D input: " + x.name)
        for dim, size in zip(dims, sizes):
            dim.Clear()
            dim.dim_value = size
    source_opset = next(o.version for o in model.opset_import if not o.domain)
    if source_opset != opset:
        print("Converting ONNX opset %d -> %d" % (source_opset, opset), flush=True)
        model = onnx.version_converter.convert_version(model, opset)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    print("Prepared ONNX:", dst, "bytes:", dst.stat().st_size, flush=True)
    for x in model.graph.input:
        print("Input", x.name, [d.dim_value for d in x.type.tensor_type.shape.dim], flush=True)
    print("Outputs:", [x.name for x in model.graph.output], flush=True)
    return dst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, type=Path)
    p.add_argument("--output", default="models/katago_10b_19x19.rknn", type=Path)
    p.add_argument("--prepared", default="models/katago_10b_19x19_opset19.onnx", type=Path)
    p.add_argument("--opset", type=int, default=19)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.prepared.parent.mkdir(parents=True, exist_ok=True)
    prepared = prepare(args.onnx, args.prepared, args.opset)
    if args.prepare_only:
        return
    from rknn.api import RKNN
    rknn = RKNN(verbose=True)
    try:
        print("Converting to RK3588 RKNN, FP16/unquantized build...", flush=True)
        ret = rknn.config(target_platform="rk3588", optimization_level=3)
        if ret != 0:
            raise RuntimeError("rknn.config returned " + str(ret))
        ret = rknn.load_onnx(model=str(prepared))
        if ret != 0:
            raise RuntimeError("rknn.load_onnx returned " + str(ret))
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError("rknn.build returned " + str(ret))
        ret = rknn.export_rknn(str(args.output))
        if ret != 0:
            raise RuntimeError("rknn.export_rknn returned " + str(ret))
    finally:
        rknn.release()
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise RuntimeError("RKNN output missing or empty")
    print("SUCCESS:", args.output, "bytes:", args.output.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
