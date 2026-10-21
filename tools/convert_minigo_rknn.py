#!/usr/bin/env python3
"""MiniGoCpp MGNET001 -> ONNX -> RK3588 RKNN.

This exports the exact trained forward network. Each MiniGoCpp Dense layer
is mathematically equivalent to a Conv 1x1 with 1x1 spatial dimensions.
The optimizer state in the checkpoint is intentionally not exported.

Dependencies for ONNX export: numpy onnx
Dependencies for optional verification: onnxruntime
Dependencies for RKNN export: rknn-toolkit2 (matching your Linux/Python)
"""
import argparse
import struct
from pathlib import Path

import numpy as np


def read_checkpoint(path):
    raw = Path(path).read_bytes()
    if len(raw) < 24:
        raise ValueError("model too short: expected 24-byte header")
    magic, n, h, step = struct.unpack_from("<8sIIQ", raw)
    if magic != b"MGNET001":
        raise ValueError(f"unsupported model signature: {magic!r} (expected MGNET001)")
    if not (2 <= n <= 19 and 2 <= h <= 1024):
        raise ValueError(f"invalid network shape: size={n}, hidden={h}")
    input_count = n * n + 4
    action_count = n * n + 1
    shapes = [
        ("w0", (h, input_count)), ("b0", (h,)),
        ("w1", (h, h)), ("b1", (h,)),
        ("w2", (h, h)), ("b2", (h,)),
        ("wp", (action_count, h)), ("bp", (action_count,)),
        ("wv", (1, h)), ("bv", (1,)),
    ]
    total = sum(int(np.prod(shape)) for _, shape in shapes)
    if len(raw) != 24 + total * 3 * 4:
        raise ValueError(
            f"model size mismatch: actual={len(raw)} bytes, expected={24 + total * 3 * 4}"
        )
    values = np.frombuffer(raw, dtype="<f4", offset=24)
    params = {}
    at = 0
    for name, shape in shapes:
        length = int(np.prod(shape))
        # Actual saved layout per parameter is data, Adam m, Adam v.
        # Only 'data' belongs in the forward-pass RKNN model.
        params[name] = values[at:at + length].astype(np.float32, copy=True).reshape(shape)
        at += 3 * length
    assert at == len(values)
    return n, h, step, params, total


def numpy_predict(x, p):
    """Reference implementation matching src/network.cpp (no legal mask)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    a0 = np.maximum(0, p["w0"] @ x + p["b0"])
    a1 = np.maximum(0, p["w1"] @ a0 + p["b1"])
    z2 = p["w2"] @ a1 + p["b2"]
    h = np.maximum(0, a0 + z2)
    logits = p["wp"] @ h + p["bp"]
    value = np.tanh((p["wv"] @ h + p["bv"]).astype(np.float32))
    return logits.astype(np.float32), value.astype(np.float32)


def export_onnx(path, n, h, p):
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as e:
        raise RuntimeError("please install ONNX: python -m pip install onnx numpy") from e

    ops = []
    def conv(name, src, w, b, dst):
        ops.append(helper.make_node("Conv", [src, w, b], [dst], name=name,
                                    kernel_shape=[1, 1], strides=[1, 1]))

    conv("Dense0_as_1x1Conv", "input_features", "w0", "b0", "z0")
    ops.append(helper.make_node("Relu", ["z0"], ["a0"], name="ReLU0"))
    conv("Dense1_as_1x1Conv", "a0", "w1", "b1", "z1")
    ops.append(helper.make_node("Relu", ["z1"], ["a1"], name="ReLU1"))
    conv("Dense2_as_1x1Conv", "a1", "w2", "b2", "z2")
    ops.append(helper.make_node("Add", ["a0", "z2"], ["skip_add"], name="ResidualAdd"))
    ops.append(helper.make_node("Relu", ["skip_add"], ["features"], name="ResidualReLU"))
    conv("PolicyHead_as_1x1Conv", "features", "wp", "bp", "policy_logits")
    conv("ValueHead_as_1x1Conv", "features", "wv", "bv", "value_pre_tanh")
    ops.append(helper.make_node("Tanh", ["value_pre_tanh"], ["value"], name="ValueTanh"))

    initializers = []
    for key in ("w0", "b0", "w1", "b1", "w2", "b2", "wp", "bp", "wv", "bv"):
        array = p[key]
        if key.startswith("w"):
            array = array[:, :, None, None]  # [Cout,Cin,1,1] for Conv
        initializers.append(numpy_helper.from_array(np.ascontiguousarray(array), name=key))

    graph = helper.make_graph(
        ops, "MiniGoCpp_MGNET001_forward",
        [helper.make_tensor_value_info("input_features", TensorProto.FLOAT, [1, n*n+4, 1, 1])],
        [
            helper.make_tensor_value_info("policy_logits", TensorProto.FLOAT, [1, n*n+1, 1, 1]),
            helper.make_tensor_value_info("value", TensorProto.FLOAT, [1, 1, 1, 1]),
        ],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 11)],
                              producer_name="MiniGoCpp checkpoint exporter")
    model.ir_version = 8  # Broad compatibility with older ONNX consumers.
    model.doc_string = "Weights from MGNET001; pass/legal softmax and MCTS remain on CPU."
    onnx.checker.check_model(model)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def verify_onnx(path, n, p):
    try:
        import onnxruntime as ort
    except ImportError:
        print("ONNX Runtime not installed; skip optional numerical validation.")
        return
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1234)
    max_err = 0.0
    # Both realistic discrete board features and nontrivial random inputs.
    for _ in range(20):
        x = rng.choice(np.array([-1, 0, 1], np.float32), size=n*n+4).astype(np.float32)
        ref_policy, ref_value = numpy_predict(x, p)
        policy, value = session.run(None, {"input_features": x.reshape(1, n*n+4, 1, 1)})
        err = max(float(np.max(np.abs(ref_policy - policy.reshape(-1)))),
                  float(np.max(np.abs(ref_value - value.reshape(-1)))))
        max_err = max(max_err, err)
    print(f"ONNX Runtime vs NumPy: maxAbs={max_err:.8g}")
    if not max_err < 1e-4:
        raise RuntimeError("ONNX forward outputs differ from reference")


def export_rknn(onnx_path, rknn_path):
    try:
        from rknn.api import RKNN
    except ImportError as e:
        raise RuntimeError("RKNN-Toolkit2 is not installed; use --onnx-only, or install RKNN-Toolkit2") from e
    rknn = RKNN(verbose=True)
    try:
        print("[RKNN] config rk3588")
        if rknn.config(target_platform="rk3588") != 0:
            raise RuntimeError("RKNN config failed")
        print("[RKNN] load ONNX")
        if rknn.load_onnx(model=str(onnx_path)) != 0:
            raise RuntimeError("RKNN load_onnx failed")
        print("[RKNN] build (no INT8 quantization; start with FP16)")
        if rknn.build(do_quantization=False) != 0:
            raise RuntimeError("RKNN build failed")
        Path(rknn_path).parent.mkdir(parents=True, exist_ok=True)
        if rknn.export_rknn(str(rknn_path)) != 0:
            raise RuntimeError("RKNN export failed")
        print("RKNN written:", rknn_path)
    finally:
        rknn.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="MiniGoCpp checkpoint (MGNET001)")
    parser.add_argument("--onnx", help="ONNX destination; default next to checkpoint")
    parser.add_argument("--rknn", help="RKNN destination; default next to checkpoint")
    parser.add_argument("--onnx-only", action="store_true", help="Only export ONNX")
    parser.add_argument("--inspect-only", action="store_true", help="Only decode model header and parameter shape")
    args = parser.parse_args()
    model_file = Path(args.model)
    n, h, step, p, total = read_checkpoint(model_file)
    print(f"MGNET001: board={n}x{n}, hidden={h}, steps={step}, parameters={total}")
    print(f"Input shape [1,{n*n+4},1,1]; outputs [1,{n*n+1},1,1] and [1,1,1,1]")
    if args.inspect_only:
        return
    onnx_path = Path(args.onnx) if args.onnx else model_file.with_suffix(".onnx")
    rknn_path = Path(args.rknn) if args.rknn else model_file.with_suffix(".rknn")
    export_onnx(onnx_path, n, h, p)
    print("ONNX written:", onnx_path)
    verify_onnx(onnx_path, n, p)
    if not args.onnx_only:
        export_rknn(onnx_path, rknn_path)

if __name__ == "__main__":
    main()
