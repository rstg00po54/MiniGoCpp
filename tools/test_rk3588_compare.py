#!/usr/bin/env python3
"""Unit tests for the strict RK3588 native CPU/GPU/NPU result parser."""
import importlib.util
import json
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location("summary", Path(__file__).parent / "summarize_rk3588_compare.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
heads = ["OutputPolicyPass", "OutputPolicy", "OutputValue", "OutputScoreValue", "OutputOwnership"]
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "cpu.log").write_text(
        "".join("COMPARE_VERIFY backend=cpu output={} wrong=0/3\n".format(h) for h in heads)
        + "RESULT backend=cpu mean=10.000 ms p50=9.000 ms p95=11.000 ms nnEvals/s=100.00\n")
    (root / "gpu.log").write_text(
        "".join("GPU_VERIFY output={} wrong=0/3 max_abs=0.001\n".format(h) for h in heads)
        + "GPU_FULL_NN_PASS backend=opencl dtype=fp16 mean=5.000 ms p50=4.000 ms p95=6.000 ms nn_evals_per_sec=200.00 runs=30\n")
    (root / "npu.log").write_text(
        "".join("COMPARE_VERIFY backend=rknn output={} wrong=0/3\n".format(h) for h in heads)
        + "RESULT core=0 mean=2.000 ms p50=1.500 ms p95=2.500 ms nnEvals/s=500.00\n")
    assert m.generate(root, root)
    data = json.loads((root / "comparison.json").read_text())
    assert [x["nn_evals_per_sec"] for x in data["results"]] == [100, 200, 500]
    assert [x["speedup_vs_cpu"] for x in data["results"]] == [1, 2, 5]
    (root / "gpu.log").write_text("GPU_FULL_NN_FAIL: GPU missing\n")
    assert not m.generate(root, root)
    data = json.loads((root / "comparison.json").read_text())
    assert data["results"][1]["status"] == "UNAVAILABLE"
    assert data["results"][1]["nn_evals_per_sec"] is None
    (root / "gpu.log").write_text("GPU_FULL_NN_PASS backend=opencl dtype=fp16 mean=5.000 ms p50=4.000 ms p95=6.000 ms nn_evals_per_sec=200.00\n")
    assert not m.generate(root, root), "No numerical verification must never count as PASS"
print("PASS RK3588 CPU/GPU/NPU result processing: valid runs, missing GPU and missing verification")
