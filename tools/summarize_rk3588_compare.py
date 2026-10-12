#!/usr/bin/env python3
"""RK3588 KataGo 10b performance report.

CPU: original ONNX Runtime NN-only, shared reference input validated (5/5).
NPU: RKNN NN-only, shared reference input validated (5/5).
GPU: OFFICIAL KataGo benchmarknn OpenCL, batch 1, no MCTS, native inputs.
The GPU measurement has a different input workload and lacks per-head reference
validation; never label its rate as strictly like-for-like with CPU or NPU.
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

FLOAT = r"([0-9]+(?:\.[0-9]+)?)"
SPECS = {
    "cpu": (re.compile(r"RESULT backend=cpu mean=" + FLOAT + r" ms p50=" + FLOAT +
                       r" ms p95=" + FLOAT + r" ms nnEvals/s=" + FLOAT),
            re.compile(r"COMPARE_VERIFY backend=cpu output=(\w+) wrong=(\d+)/(\d+)")),
    "npu": (re.compile(r"RESULT core=0 mean=" + FLOAT + r" ms p50=" + FLOAT +
                       r" ms p95=" + FLOAT + r" ms nnEvals/s=" + FLOAT),
            re.compile(r"COMPARE_VERIFY backend=rknn output=(\w+) wrong=(\d+)/(\d+)")),
}
HEADS = {"OutputPolicyPass", "OutputPolicy", "OutputValue", "OutputScoreValue", "OutputOwnership"}


def row_template(name, path):
    return {"backend": name, "status": "UNAVAILABLE", "reason": "",
            "mean_ms": None, "p50_ms": None, "p95_ms": None,
            "nn_evals_per_sec": None, "validated_outputs": 0,
            "speedup_vs_cpu": None, "indicative_ratio_vs_cpu": None,
            "precision": None, "model_name": None, "log": str(path)}


def parse_reference_backend(name, path):
    result = row_template(name, path)
    if not path.is_file():
        result["reason"] = "Test log missing"
        return result
    log = path.read_text(encoding="utf-8", errors="replace")
    pattern, checks_pattern = SPECS[name]
    matches = pattern.findall(log)
    if not matches:
        failures = re.findall(r"(?:ERROR|FAILED)[^\n]*", log)
        result["reason"] = failures[-1][:240] if failures else "No verified NN-only result"
        if "precision mismatch" in log.lower() or "accuracy verification failed" in log.lower():
            result["status"] = "INVALID_ACCURACY"
        return result
    checks = checks_pattern.findall(log)
    good = {head for head, wrong, total in checks if head in HEADS and int(wrong) == 0 and int(total) > 0}
    result["validated_outputs"] = len(good)
    if len(good) != 5:
        result["status"] = "INVALID_ACCURACY"
        result["reason"] = "{}/5 reference output heads passed".format(len(good))
        return result
    mean, p50, p95, rate = map(float, matches[-1])
    if not (all(math.isfinite(v) for v in (mean, p50, p95, rate)) and
            0 < mean < 100000 and 0 < rate and abs(1000 / mean - rate) / rate < 0.02):
        result["reason"] = "Inconsistent NN-only timing and throughput"
        return result
    result.update(status="PASS", mean_ms=mean, p50_ms=p50, p95_ms=p95,
                  nn_evals_per_sec=rate, precision="FP32" if name == "cpu" else "RKNN")
    return result


def parse_native_gpu(path):
    result = row_template("gpu", path)
    if not path.is_file():
        result["reason"] = "Native KataGo GPU log missing"
        return result
    log = path.read_text(encoding="utf-8", errors="replace")
    errors = re.findall(r"(?m)^(?:ERROR:|CMake Error[^\\n]*|fatal:|FAILED:|ninja: error:)[^\\n]*", log)
    if "KATAGO_NATIVE_START backend=opencl" not in log:
        result["reason"] = (errors[-1][:240] if errors else
                            "Official OpenCL benchmarknn did not start; inspect gpu.log for build failure")
        return result
    if not re.search(r"Using OpenCL backend", log, re.IGNORECASE):
        result["reason"] = "KataGo OpenCL compile-time backend was not confirmed"
        return result
    records = []
    for line in log.splitlines():
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            val = json.loads(line)
            if isinstance(val, dict) and "actualWallNNEvalsPerSec" in val:
                records.append(val)
        except (ValueError, TypeError):
            continue
    if not records:
        result["reason"] = (errors[-1][:240] if errors else
                            "No official KataGo benchmarknn JSON result; inspect full gpu.log")
        return result
    obj = records[-1]
    try:
        revision = str(obj["revision"])
        name = str(obj["modelName"])
        sizes = obj["boardSizes"]
        batch = obj["batchSize"]
        threads = obj["numThreads"]
        iterations = obj["numIterations"]
        gpu_ids = obj["gpuIdxs"]
        p50s = obj["perThreadMedianMs"]
        throughput = float(obj["actualWallNNEvalsPerSec"])
        wall = float(obj["actualWallSeconds"])
        fp16 = obj["usingFP16"]
        if not (revision.lower().endswith("-opencl") and isinstance(name, str) and name and
                sizes == [19] and type(batch) is int and batch == 1 and
                type(threads) is int and threads == 1 and type(iterations) is int and iterations > 0 and
                isinstance(gpu_ids, list) and len(gpu_ids) > 0 and
                isinstance(p50s, list) and len(p50s) == 1 and
                type(fp16) is bool and math.isfinite(wall) and wall > 0 and
                math.isfinite(throughput) and throughput > 0):
            raise ValueError("OpenCL/batch/threads/device/wall time validation failed")
        median = float(p50s[0])
        if not math.isfinite(median) or median <= 0 or median > 100000:
            raise ValueError("Invalid median NN forward latency")
    except (KeyError, ValueError, TypeError, IndexError) as e:
        result["reason"] = "Invalid official KataGo benchmarknn result: " + str(e)
        return result
    result.update(status="PASS_NATIVE", p50_ms=median, nn_evals_per_sec=throughput,
                  precision="FP16" if fp16 else "FP32", model_name=name,
                  reason="KataGo native features; does not share exact CPU/NPU reference input")
    return result


def generate(directory, output):
    rows = [parse_reference_backend("cpu", directory / "cpu.log"),
            parse_native_gpu(directory / "gpu.log"),
            parse_reference_backend("npu", directory / "npu.log")]
    cpu = rows[0]
    for row in rows:
        if cpu["status"] != "PASS" or row["nn_evals_per_sec"] is None:
            continue
        ratio = row["nn_evals_per_sec"] / cpu["nn_evals_per_sec"]
        if row["backend"] == "gpu":
            row["indicative_ratio_vs_cpu"] = round(ratio, 3)
        else:
            row["speedup_vs_cpu"] = round(ratio, 3)
    passed = [rows[0]["status"] == "PASS", rows[1]["status"] == "PASS_NATIVE", rows[2]["status"] == "PASS"]
    summary = {
        "model": "KataGo 10b / 19x19 / batch 1",
        "methodology": {
            "cpu_npu": "Same FP32 input, five heads checked against ONNX reference",
            "gpu": "Official KataGo OpenCL benchmarknn, native feature generator, one thread, batch 1",
            "gpu_same_reference_as_cpu_npu": False,
            "gpu_latency": "Per-thread median ms; GPU throughput is wall-clock evals/sec",
            "strict_gpu_cpu_npu_speedup_comparable": False,
        },
        "results": rows,
        "all_passed": all(passed),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    columns = ["backend", "status", "mean_ms", "p50_ms", "p95_ms", "nn_evals_per_sec",
               "speedup_vs_cpu", "indicative_ratio_vs_cpu", "validated_outputs",
               "precision", "model_name", "reason"]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})
    md = [
        "# RK3588S · KataGo 10b · CPU / Official OpenCL GPU / RKNN NPU",
        "",
        "19×19, batch=1, NN-only. GPU uses KataGo's **native optimized OpenCL** benchmarknn, not the handwritten GPU kernel.",
        "",
        "| Backend | Status | Mean ms | Median ms | P95 ms | NN evals/s | CPU ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    def fmt(value, precision=3):
        return "—" if value is None else ("{:.%df}" % precision).format(value)
    for row in rows:
        if row["status"] in ("PASS", "PASS_NATIVE"):
            ratio = ("~{:.2f}×¹".format(row["indicative_ratio_vs_cpu"]) if row["backend"] == "gpu" else
                     "{:.2f}×".format(row["speedup_vs_cpu"]) if row["speedup_vs_cpu"] is not None else "—")
            md.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                row["backend"].upper(), row["status"], fmt(row["mean_ms"]),
                fmt(row["p50_ms"]), fmt(row["p95_ms"]), fmt(row["nn_evals_per_sec"], 2), ratio))
        else:
            md.append("| {} | {} | — | — | — | — | — |".format(row["backend"].upper(), row["status"]))
    md.extend(["", "¹ **Indicative throughput ratio only.** KataGo GPU benchmarknn uses native feature inputs",
               "and a different inference/timing path from ONNX CPU and RKNN NPU. Its accuracy is not checked",
               "against the same five ONNX reference outputs; model lineage must also be confirmed separately.",
               "",
               "CPU/NPU: same reference input, 5/5 heads checked. GPU: official `benchmarknn`, wall-time",
               "throughput and per-thread median latency (not a sample-mean latency).",
               "All three use batch=1 without MCTS; GPU `benchmarknn` includes its own transfers/postprocessing.",
               "NPU reports one-core latency, not triple-context parallel aggregate throughput.",
               "",
               "## Unavailable / failed", ""])
    errors = [row for row in rows if row["status"] not in ("PASS", "PASS_NATIVE")]
    if errors:
        md.extend("- **{}**: {}".format(row["backend"].upper(), row["reason"]) for row in errors)
    else:
        md.append("- None.")
    md.extend(["", "Raw logs: cpu.log, gpu.log, npu.log."])
    (output / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("\nResult files:", output / "comparison.md", output / "comparison.csv", output / "comparison.json")
    return summary["all_passed"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--logs", type=Path, required=True)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    raise SystemExit(0 if generate(args.logs, args.out or args.logs) else 1)
