#!/usr/bin/env bash
# One-command, validated native NN-only comparison on RK3588 Android.
# CPU=ONNX Runtime FP32; GPU=OpenCL FP16; NPU=RKNN core0.
# Same original KataGo10b 19x19 graph and shared feature input.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADB="${ADB:-adb}"
ADB_SERIAL="$(ADB="$ADB" bash "$ROOT/tools/select_adb_device.sh" rk)"
RUNS="${RUNS:-30}"
WARMUP="${WARMUP:-5}"
GPU_RUNS="${GPU_RUNS:-$RUNS}"
GPU_WARMUP="${GPU_WARMUP:-$WARMUP}"
THREADS="${THREADS:-4}"
CACHE="${KATAGO_BENCH_CACHE:-$HOME/.cache/minigocpp/androidbench}"
COMPARE_INPUT_FILE="${COMPARE_INPUT_FILE:-$CACHE/katago10b_shared_input_v1.bin}"
OUT="${BENCH_RESULTS_DIR:-$ROOT/results/rk3588_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
echo "RK3588_COMPARE_START serial=$ADB_SERIAL cpu_threads=$THREADS runs=$RUNS warmup=$WARMUP gpu_runs=$GPU_RUNS"
echo "RESULTS_DIR=$OUT"
if ! bash "$ROOT/tools/prepare_rk3588_compare.sh" > "$OUT/prepare.log" 2>&1; then
    cat "$OUT/prepare.log" >&2
    echo "ERROR: Failed to prepare comparable model/inputs; no scores are valid" >&2
    exit 1
fi
tail -3 "$OUT/prepare.log"
export ADB ADB_SERIAL RUNS WARMUP THREADS COMPARE_INPUT_FILE KATAGO_BENCH_CACHE="$CACHE"
failed=0

echo "=== CPU: ONNX Runtime, actual entire 10b, FP32 ==="
if ! BENCH_LOG="$OUT/cpu.log" bash "$ROOT/tools/bench_android.sh" rk cpu; then
    failed=1
fi

echo "=== GPU: Mali OpenCL, original entire 10b, FP16 ==="
if ! GPU_RUNS="$GPU_RUNS" GPU_WARMUP="$GPU_WARMUP" bash "$ROOT/tools/bench_rk3588_gpu_full.sh" > "$OUT/gpu.log" 2>&1; then
    failed=1
fi
grep -E 'GPU_DEVICE|GPU_GRAPH|GPU_VERIFY|GPU_FULL_NN_(PASS|FAIL)|ERROR' "$OUT/gpu.log" || true

echo "=== NPU: RKNPU2 core 0, entire RKNN 10b ==="
if ! CORE_MODES=0 bash "$ROOT/tools/bench_rk3588_npu.sh" > "$OUT/npu.log" 2>&1; then
    failed=1
fi
grep -E 'RKNN runtime|COMPARE_VERIFY|RESULT core=0|ERROR' "$OUT/npu.log" || true

echo "=== VALIDATED CPU/GPU/NPU COMPARISON ==="
if ! python3 "$ROOT/tools/summarize_rk3588_compare.py" --logs "$OUT" --out "$OUT"; then
    failed=1
fi
echo "RK3588_COMPARE_END status=$([ "$failed" -eq 0 ] && echo PASS || echo INCOMPLETE) dir=$OUT"
exit "$failed"
