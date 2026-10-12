#!/usr/bin/env python3
"""Regression tests: official KataGo benchmarknn JSON and strict CPU/NPU verification."""
import importlib.util
import json
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location("summary", Path(__file__).parent / "summarize_rk3588_compare.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
heads = ["OutputPolicyPass", "OutputPolicy", "OutputValue", "OutputScoreValue", "OutputOwnership"]

native = {
    "modelFile": "model.bin.gz", "modelName": "g170-b10c128-s1141046784-d204142634",
    "revision": "test-opencl", "boardSizes": [19], "requireExactNNLen": False,
    "usingFP16": True, "batchSize": 1, "numThreads": 1, "numIterations": 30,
    "gpuIdxs": [0], "perThreadMedianMs": [5.234],
    "perThreadNNEvalsPerSec": [188.0], "sumMedianNNEvalsPerSec": 188.0,
    "actualWallSeconds": 0.162, "actualWallNNEvalsPerSec": 185.5,
}
def gpu_log(data):
    return ("Using OpenCL backend\nKATAGO_NATIVE_START backend=opencl boardsize=19 batch=1 threads=1 runs=30 warmup=5\n"
            + json.dumps(data) + "\n")
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "cpu.log").write_text(
        "".join("COMPARE_VERIFY backend=cpu output={} wrong=0/3\n".format(h) for h in heads)
        + "RESULT backend=cpu mean=10.000 ms p50=9.000 ms p95=11.000 ms nnEvals/s=100.00\n")
    (root / "gpu.log").write_text(gpu_log(native))
    (root / "npu.log").write_text(
        "".join("COMPARE_VERIFY backend=rknn output={} wrong=0/3\n".format(h) for h in heads)
        + "RESULT core=0 mean=2.000 ms p50=1.500 ms p95=2.500 ms nnEvals/s=500.00\n")
    assert mod.generate(root, root)
    data = json.loads((root / "comparison.json").read_text())
    rows = data["results"]
    assert [r["nn_evals_per_sec"] for r in rows] == [100, 185.5, 500]
    assert [r["status"] for r in rows] == ["PASS", "PASS_NATIVE", "PASS"]
    assert rows[1]["mean_ms"] is None and rows[1]["p50_ms"] == 5.234
    assert rows[1]["speedup_vs_cpu"] is None and rows[1]["indicative_ratio_vs_cpu"] == 1.855
    assert rows[2]["speedup_vs_cpu"] == 5
    assert data["methodology"]["strict_gpu_cpu_npu_speedup_comparable"] is False

    for key, value in [("revision", "test-eigen"), ("batchSize", 4), ("boardSizes", [9]), ("numThreads", 2)]:
        modified = dict(native)
        modified[key] = value
        (root / "gpu.log").write_text(gpu_log(modified))
        assert not mod.generate(root, root), (key, value)
        results = json.loads((root / "comparison.json").read_text())["results"]
        assert results[1]["status"] == "UNAVAILABLE"
        assert results[1]["nn_evals_per_sec"] is None

    (root / "gpu.log").write_text("GPU_FULL_NN_PASS backend=opencl dtype=fp16 mean=2.000 ms p50=2.000 ms p95=2.000 ms nn_evals_per_sec=500.00\n")
    assert not mod.generate(root, root), "Handwritten OpenCL benchmark must not be accepted as native KataGo GPU"

    (root / "gpu.log").write_text(gpu_log(native))
    (root / "npu.log").write_text("ERROR: RKNN network output precision mismatch vs ONNX CPU reference\n")
    assert not mod.generate(root, root)
    rows = json.loads((root / "comparison.json").read_text())["results"]
    assert rows[2]["status"] == "INVALID_ACCURACY" and rows[2]["nn_evals_per_sec"] is None
print("PASS: official KataGo GPU JSON parsing, strict backend/batch checks, no false like-for-like ratios")
