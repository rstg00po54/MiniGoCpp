#!/usr/bin/env python3
"""Generate a quantized, nonconstant 3x3 Conv RKNN throughput model for RK3588.
This script runs ONLY on the x86_64 build host (RKNN-Toolkit2 compiler).
The Android benchmark itself is C++ / ADB without Python.
"""
import argparse
from pathlib import Path
import tempfile
import numpy as np

CASES = {
    "rk64": (64, 64, 128, 128),
    "heavy": (96, 96, 256, 256),
    "x128": (128, 128, 256, 256),
    "x256": (256, 256, 256, 256),
    "x128c512": (128, 128, 512, 512),
    "x256c512": (256, 256, 512, 512),
}


def make_conv(profile, destination):
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as ex:
        raise RuntimeError("ONNX exporter missing: install numpy and onnx in your RKNN Python environment") from ex
    h, w, ci, co = CASES[profile]
    oc = np.arange(co, dtype=np.int32)[:, None, None, None]
    ic = np.arange(ci, dtype=np.int32)[None, :, None, None]
    ky = np.arange(3, dtype=np.int32)[None, None, :, None]
    kx = np.arange(3, dtype=np.int32)[None, None, None, :]
    # NCHW convolution weight [CO,CI,KH,KW]. It is varied and nonzero.
    weights = (((oc * 13 + ic * 7 + ky * 11 + kx * 17) % 29) - 14).astype(np.float32) * 0.004
    biases = ((np.arange(co, dtype=np.int32) % 11) - 5).astype(np.float32) * .006
    graph = helper.make_graph(
        [helper.make_node("Conv", ["input", "W", "B"], ["output"],
                          name="Int8NpuConv", kernel_shape=[3, 3],
                          pads=[1, 1, 1, 1], strides=[1, 1])],
        "RK3588_NPU_INT8_TOPS_NONZERO_CONV",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, ci, h, w])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, co, h, w])],
        initializer=[
            numpy_helper.from_array(weights, "W"),
            numpy_helper.from_array(biases, "B")
        ])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 11)],
                              producer_name="MiniGoCpp synthetic INT8 NPU benchmark")
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(destination))
    print("ONNX generated:", destination, "shape=", (h, w, ci, co), flush=True)


def input_pattern(profile, sample):
    h, w, ci, _ = CASES[profile]
    y = np.arange(h, dtype=np.int32)[None, :, None]
    x = np.arange(w, dtype=np.int32)[None, None, :]
    c = np.arange(ci, dtype=np.int32)[:, None, None]
    q = (((y * 29 + x * 13 + c * 7 + sample * 23) ^
          (c * 5 + y * 3 + x * 17)) % 113) - 56
    return (q.astype(np.float32) * 0.005).reshape(1, ci, h, w)


def convert(profile, destination):
    try:
        from rknn.api import RKNN
    except ImportError as ex:
        raise RuntimeError(
            "RKNN-Toolkit2 unavailable in this Python environment. Activate the "
            "existing RKNN Toolkit2 x86_64 environment or install a compatible version."
        ) from ex
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="minigocpp_rknn_tops_") as tmp:
        tmp = Path(tmp)
        onnx_path = tmp / "conv.onnx"
        make_conv(profile, onnx_path)
        calibrations = []
        for i in range(3):
            sample = tmp / ("calibration_%d.npy" % i)
            np.save(sample, input_pattern(profile, i))
            calibrations.append(str(sample))
        dataset = tmp / "dataset.txt"
        dataset.write_text("\n".join(calibrations) + "\n")
        rknn = RKNN(verbose=True)
        try:
            print("RKNN: config rk3588 quantized_dtype=asymmetric_quantized-8", flush=True)
            rc = rknn.config(target_platform="rk3588", optimization_level=3,
                             quantized_dtype="asymmetric_quantized-8")
            if rc != 0:
                raise RuntimeError("rknn.config failed (%d)" % rc)
            if rknn.load_onnx(model=str(onnx_path)) != 0:
                raise RuntimeError("rknn.load_onnx failed")
            print("RKNN: compiling real INT8 model, do_quantization=True", flush=True)
            if rknn.build(do_quantization=True, dataset=str(dataset)) != 0:
                raise RuntimeError("rknn.build INT8 quantization failed")
            if rknn.export_rknn(str(destination)) != 0:
                raise RuntimeError("rknn.export_rknn failed")
            if not destination.is_file() or destination.stat().st_size == 0:
                raise RuntimeError("RKNN output empty")
            print("RKNN_INT8_BUILD_PASS profile=%s bytes=%d model=%s" %
                  (profile, destination.stat().st_size, destination), flush=True)
        finally:
            rknn.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=sorted(CASES))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    convert(args.profile, args.output)


if __name__ == "__main__":
    main()
