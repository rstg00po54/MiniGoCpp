#!/usr/bin/env python3
"""Convert a KataGo *size-specific dumponnx* graph into a fixed RK3588 RKNN.

The ONNX must have been exported from the original KataGo .bin model using:
  katago dumponnx -model 10b.bin -out 10b_9x9.onnx -nn-x-len 9 -nn-y-len 9
Do not rewrite the dimensions of an already compiled 19x19 graph.
"""
import argparse
from pathlib import Path
import onnx


def prepare(src, dst, size):
    model = onnx.load(str(src))
    expected = {
        "InputSpatial": [1, 22, size, size],
        "InputGlobal": [1, 19, 1, 1],
        "InputMask": [1, 1, size, size],
    }
    names = [x.name for x in model.graph.input]
    if names != list(expected):
        raise RuntimeError(f"Wrong KataGo input names/order: {names!r}")
    props = {x.key: x.value for x in model.metadata_props}
    if props.get("katago.build.nnXLen") != str(size) or props.get("katago.build.nnYLen") != str(size):
        raise RuntimeError("ONNX was not generated for the requested NN buffer size; use KataGo dumponnx")
    if props.get("katago.name") != "g170-b10c128-s1141046784-d204142634":
        raise RuntimeError("Unexpected KataGo weights: " + str(props.get("katago.name")))
    expected_outputs = ["OutputPolicyPass", "OutputPolicy", "OutputValue", "OutputScoreValue", "OutputOwnership"]
    if [x.name for x in model.graph.output] != expected_outputs:
        raise RuntimeError("Unexpected KataGo output names/order")
    for x in model.graph.input:
        dims = x.type.tensor_type.shape.dim
        if len(dims) != 4:
            raise RuntimeError("Expected 4D tensor: " + x.name)
        for d, value in zip(dims, expected[x.name]):
            d.Clear()
            d.dim_value = value
    versions = [o.version for o in model.opset_import if o.domain == ""]
    if versions != [20]:
        raise RuntimeError(f"Expected source opset 20; got {versions}")
    print("Converting ONNX opset 20 to 19", flush=True)
    model = onnx.version_converter.convert_version(model, 19)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    for x in model.graph.input:
        values = [d.dim_value for d in x.type.tensor_type.shape.dim]
        if values != expected[x.name]:
            raise RuntimeError(f"Input shape changed unexpectedly: {x.name} {values}")
    onnx.save(model, str(dst))
    print(f"Validated ONNX {size}x{size}: {dst}, bytes={dst.stat().st_size}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--size", type=int, choices=(9, 13), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.prepared.parent.mkdir(parents=True, exist_ok=True)
    prepare(args.onnx, args.prepared, args.size)
    if args.prepare_only:
        return
    from rknn.api import RKNN
    engine = RKNN(verbose=True)
    try:
        for stage, result in (
            ("config", engine.config(target_platform="rk3588", optimization_level=3)),
        ):
            if result != 0:
                raise RuntimeError(f"RKNN {stage} failed: {result}")
        ret = engine.load_onnx(model=str(args.prepared))
        if ret != 0:
            raise RuntimeError(f"RKNN load_onnx failed: {ret}")
        ret = engine.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"RKNN build failed: {ret}")
        ret = engine.export_rknn(str(args.output))
        if ret != 0:
            raise RuntimeError(f"RKNN export_rknn failed: {ret}")
    finally:
        engine.release()
    if not args.output.is_file() or args.output.stat().st_size < 1000000:
        raise RuntimeError("Missing/truncated RKNN artifact")
    print(f"SUCCESS {args.size}x{args.size}: {args.output} ({args.output.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
