#!/usr/bin/env python3
"""Summarize RK3588 original KataGo 10b NN-only CPU/OpenCL/RKNN results.

No fabricated values: only validated, successful results are compared.
Writes JSON, CSV and Markdown; missing GPU output explicitly stays unavailable.
"""
import argparse
import csv
import json
import re
from pathlib import Path


FLOAT = r"([0-9]+(?:\.[0-9]+)?)"
SPECS = {
    "cpu": (re.compile(r"RESULT backend=cpu mean=" + FLOAT + r" ms p50=" + FLOAT +
                       r" ms p95=" + FLOAT + r" ms nnEvals/s=" + FLOAT),
            re.compile(r"COMPARE_VERIFY backend=cpu output=(\w+) wrong=(\d+)/(\d+)")),
    "gpu": (re.compile(r"GPU_FULL_NN_PASS backend=opencl dtype=fp16 mean=" + FLOAT +
                       r" ms p50=" + FLOAT + r" ms p95=" + FLOAT +
                       r" ms nn_evals_per_sec=" + FLOAT),
            re.compile(r"GPU_VERIFY output=(\w+) wrong=(\d+)/(\d+)")),
    "npu": (re.compile(r"RESULT core=0 mean=" + FLOAT + r" ms p50=" + FLOAT +
                       r" ms p95=" + FLOAT + r" ms nnEvals/s=" + FLOAT),
            re.compile(r"COMPARE_VERIFY backend=rknn output=(\w+) wrong=(\d+)/(\d+)")),
}
HEADS = {"OutputPolicyPass", "OutputPolicy", "OutputValue", "OutputScoreValue", "OutputOwnership"}


def parse_backend(name, path):
    result = {"backend": name, "status": "UNAVAILABLE", "reason": "", "mean_ms": None,
              "p50_ms": None, "p95_ms": None, "nn_evals_per_sec": None,
              "validated_outputs": 0, "log": str(path)}
    if not path.is_file():
        result["reason"] = "test log missing"
        return result
    log = path.read_text(encoding="utf-8", errors="replace")
    pattern, verify_pattern = SPECS[name]
    matches = pattern.findall(log)
    if not matches:
        failures = re.findall(r"(?:GPU_FULL_NN_FAIL|ERROR|FAILED)[^\n]*", log)
        result["reason"] = failures[-1][:200] if failures else "No confirmed NN-only benchmark result"
        if "precision mismatch" in log.lower() or "accuracy verification failed" in log.lower():
            result["status"] = "INVALID_ACCURACY"
        return result
    checks = verify_pattern.findall(log)
    good = {head for head, wrong, total in checks if head in HEADS and int(wrong) == 0 and int(total) > 0}
    result["validated_outputs"] = len(good)
    if len(good) != 5:
        result["reason"] = "Only {}/5 NN output heads passed numerical checks".format(len(good))
        return result
    mean, p50, p95, throughput = (float(x) for x in matches[-1])
    if not (0 < mean < 100000 and 0 < throughput and abs(1000.0 / mean - throughput) / throughput < 0.02):
        result["reason"] = "Invalid or inconsistent latency/throughput values"
        return result
    result.update(status="PASS", reason="", mean_ms=mean, p50_ms=p50,
                  p95_ms=p95, nn_evals_per_sec=throughput)
    return result


def generate(directory, output):
    rows = [parse_backend(b, directory / (b + ".log")) for b in ("cpu", "gpu", "npu")]
    lookup = {r["backend"]: r for r in rows}
    cpu = lookup["cpu"]
    for row in rows:
        row["speedup_vs_cpu"] = round(row["nn_evals_per_sec"] / cpu["nn_evals_per_sec"], 3) if (
            row["status"] == "PASS" and cpu["status"] == "PASS") else None
    summary = {"model": "original KataGo 10b", "board": "19x19", "batch": 1,
               "measurement": "single-network forward pass, no MCTS",
               "reference": "same model packet's original ONNX CPU reference, 5 output heads",
               "results": rows, "all_passed": all(x["status"] == "PASS" for x in rows)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    columns = ["backend", "status", "mean_ms", "p50_ms", "p95_ms", "nn_evals_per_sec",
               "speedup_vs_cpu", "validated_outputs", "reason"]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in columns})
    md = [
        "# RK3588S CPU / GPU / NPU — KataGo 10b 19×19 NN-only",
        "",
        "All three backends use the original 10b network and a shared comparison input.",
        "One forward pass, batch=1; does not include MCTS or engine feature generation.",
        "",
        "| Backend | Status | Mean (ms) | P50 (ms) | P95 (ms) | NN evals/s | vs CPU |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["status"] == "PASS":
            md.append("| {backend} | PASS (5/5 heads) | {mean_ms:.3f} | {p50_ms:.3f} | {p95_ms:.3f} | {nn_evals_per_sec:.2f} | {speedup_vs_cpu}x |".format(**{
                **row, "speedup_vs_cpu": "{:.2f}".format(row["speedup_vs_cpu"]) if row["speedup_vs_cpu"] else "n/a"}))
        else:
            md.append("| {} | {} | — | — | — | — | — |".format(row["backend"], row["status"]))
    md.extend([
        "",
        "## Limitations",
        "",
        "- CPU: ONNX Runtime optimized CPU FP32; GPU: **untuned generic OpenCL FP16 kernels**; NPU: RKNN native runtime.",
        "- GPU time includes GPU kernel launches and output transfers. This is a functional GPU baseline, **not peak optimized Mali GPU performance**.",
        "- NPU mode is pinned to core 0 for one-request latency; multi-context three-core aggregate throughput is a different metric.",
        "- FP16 vs FP32 and backend graph optimizations affect results, even with common source weights and input.",
        "- Timing is end-to-end native NN-only forward pass; batch size and host runtime overhead can differ.",
        "",
        "## Failures / Missing data",
        "",
    ])
    for row in rows:
        if row["status"] != "PASS":
            md.append("- **{}:** {}".format(row["backend"], row["reason"]))
    if all(r["status"] == "PASS" for r in rows):
        md.append("- None.")
    md.extend(["", "Raw logs: cpu.log, gpu.log and npu.log."])
    (output / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("\nResult files:", output / "comparison.md", output / "comparison.json", output / "comparison.csv")
    return summary["all_passed"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--logs", type=Path, required=True)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    import sys
    sys.exit(0 if generate(args.logs, args.out or args.logs) else 1)
