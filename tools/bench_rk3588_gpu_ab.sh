#!/usr/bin/env bash
# Compare validated OpenCL GPU kernels on the same RK3588 hardware and KataGo 10b packet.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_RUNS="${GPU_RUNS:-10}"
GPU_WARMUP="${GPU_WARMUP:-2}"
OUT="${GPU_AB_DIR:-$ROOT/results/gpu_ab_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
for mode in direct tiled; do
    echo "=== RK3588 Mali GPU kernel: $mode ==="
    if ! KATAGO_GPU_CONV="$mode" GPU_RUNS="$GPU_RUNS" GPU_WARMUP="$GPU_WARMUP" \
        bash "$ROOT/tools/bench_rk3588_gpu_full.sh" > "$OUT/$mode.log" 2>&1; then
        grep -E 'GPU_DEVICE|GPU_CONV_KERNEL|GPU_VERIFY|GPU_FULL_NN_FAIL|ERROR' "$OUT/$mode.log" || true
        echo "FAILED: $mode result not valid; see $OUT/$mode.log" >&2
        exit 1
    fi
    grep -E 'GPU_DEVICE|GPU_CONV_KERNEL|GPU_VERIFY|GPU_FULL_NN_PASS' "$OUT/$mode.log" || true
done
python3 - "$OUT" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
results = {}
for mode in ("direct", "tiled"):
    s = (root / (mode + ".log")).read_text(errors="replace")
    if "GPU_CONV_KERNEL mode=" + mode not in s:
        raise SystemExit("Wrong GPU kernel selected for " + mode)
    check = re.findall(r"GPU_VERIFY output=(\w+) wrong=(\d+)/(\d+)", s)
    if len(check) != 5 or any(int(wrong) != 0 for _, wrong, _ in check):
        raise SystemExit("GPU output numerical verification did not pass for " + mode)
    m = re.search(r"GPU_FULL_NN_PASS backend=opencl dtype=fp16 mean=([\d.]+) ms.*?nn_evals_per_sec=([\d.]+)", s)
    if not m:
        raise SystemExit("No verified full-network GPU result for " + mode)
    results[mode] = (float(m.group(1)), float(m.group(2)))
a, b = results["direct"], results["tiled"]
print("GPU_AB_PASS direct={:.3f}ms/{:.2f}NNps tiled={:.3f}ms/{:.2f}NNps speedup={:.2f}x".format(
    a[0], a[1], b[0], b[1], a[0] / b[0]))
print("GPU_AB_DIR=" + str(root))
PY
