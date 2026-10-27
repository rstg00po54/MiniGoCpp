#!/usr/bin/env bash
# Build an INT8 Conv RKNN on x86 Ubuntu, run native C++ on RK3588 Android.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-heavy}"
ADB_BIN="${ADB:-adb}"
WARMUP="${WARMUP:-20}"
RUNS="${RUNS:-100}"
BENCH_SECONDS="${BENCH_SECONDS:-0}"
CORE_MODES="${CORE_MODES:-0 1 2 all}"
PYTHON_BIN="${RKNN_PYTHON:-python3}"
CACHE="${RKNN_BENCH_CACHE:-$HOME/.cache/minigocpp/rknn_android}"
REMOTE="/data/local/tmp/minigocpp_rk_tops"
BASE="https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknpu2/runtime/Android/librknn_api"
fail() { echo "ERROR: $*" >&2; exit 1; }
case "$PROFILE" in rk64|heavy|x128|x256|x128c512|x256c512) ;; *)
  fail "profile must be rk64/heavy/x128/x256/x128c512/x256c512" ;;
esac
for n in "$WARMUP" "$RUNS" "$BENCH_SECONDS"; do
  [[ "$n" =~ ^[0-9]+$ ]] || fail "Warmup, runs and duration must be integers"
done
[[ ${#WARMUP} -le 5 && ${#RUNS} -le 5 && ${#BENCH_SECONDS} -le 3 ]] || fail "Duration out of range"
(( 10#$WARMUP <= 10000 && 10#$RUNS >= 1 && 10#$RUNS <= 10000 && 10#$BENCH_SECONDS <= 180 )) || fail "Bad WARMUP/RUNS/BENCH_SECONDS"
for core in $CORE_MODES; do
  case "$core" in 0|1|2|all) ;; *) fail "CORE_MODES must contain only 0 1 2 all" ;; esac
done
SERIAL="$(ADB="$ADB_BIN" bash "$ROOT/tools/select_adb_device.sh" rk)"
ADB_CMD=("$ADB_BIN" -s "$SERIAL")
ABI="$("${ADB_CMD[@]}" shell getprop ro.product.cpu.abi | tr -d '\r')"
[[ "$ABI" == arm64-v8a ]] || fail "Expected Android ARM64 RK3588"
mkdir -p "$CACHE" "$ROOT/MTK_NPU_Benchmark/results"
download() {
  local url="$1" dest="$2"
  [[ -s "$dest" ]] && return 0
  echo "Downloading $(basename "$dest")..." >&2
  if command -v curl >/dev/null; then
    curl -fL --retry 3 --connect-timeout 15 "$url" -o "$dest.tmp" || fail "Download failed: $url"
  elif command -v wget >/dev/null; then
    wget -q "$url" -O "$dest.tmp" || fail "Download failed: $url"
  else
    fail "curl or wget required"
  fi
  mv "$dest.tmp" "$dest"
}
download "$BASE/include/rknn_api.h" "$CACHE/rknn_api.h"
download "$BASE/arm64-v8a/librknnrt.so" "$CACHE/librknnrt.so"

NDK_DIR="${NDK:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}}"
if [[ -z "$NDK_DIR" ]]; then
  for d in "${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/27.2.12479018" "${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}/ndk/27.2.12479018"; do
    [[ ! -d "$d" ]] || { NDK_DIR="$d"; break; }
  done
fi
CXX="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++"
[[ -x "$CXX" ]] || fail "NDK compiler missing; set NDK=/home/cc/Android/Sdk/ndk/27.2.12479018"

MODEL="${RKNN_TOPS_MODEL:-$CACHE/tops_${PROFILE}_int8.rknn}"
if [[ ! -s "$MODEL" ]]; then
  command -v "$PYTHON_BIN" >/dev/null || fail "Set RKNN_PYTHON to your RKNN-Toolkit2 Python interpreter"
  "$PYTHON_BIN" -c 'import numpy, onnx; from rknn.api import RKNN' ||
    fail "RKNN-Toolkit2/numpy/onnx missing; set RKNN_PYTHON to your existing toolkit environment"
  echo "Generating INT8 RKNN for $PROFILE once; conversion runs on x86 host."
  "$PYTHON_BIN" "$ROOT/tools/build_rk3588_tops_model.py" --profile "$PROFILE" --output "$MODEL"
fi
BIN="$CACHE/rk3588_tops_android"
if [[ ! -x "$BIN" || "$ROOT/tools/rk3588_tops_android.cpp" -nt "$BIN" ||
      "$CACHE/rknn_api.h" -nt "$BIN" ]]; then
  echo "Compiling RK3588 native ARM64 C++ TOPS benchmark..."
  "$CXX" -std=c++17 -O2 -fPIE -pie -I"$CACHE" "$ROOT/tools/rk3588_tops_android.cpp" \
    -L"$CACHE" -lrknnrt -llog -ldl -Wl,-rpath,'$ORIGIN' -o "$BIN"
fi
echo "RK3588 selected=$SERIAL profile=$PROFILE cores=$CORE_MODES seconds=$BENCH_SECONDS"
"${ADB_CMD[@]}" shell "mkdir -p $REMOTE"
"${ADB_CMD[@]}" push "$MODEL" "$REMOTE/model.rknn" >/dev/null
"${ADB_CMD[@]}" push "$CACHE/librknnrt.so" "$REMOTE/librknnrt.so" >/dev/null
"${ADB_CMD[@]}" push "$BIN" "$REMOTE/rk3588_tops_android" >/dev/null
CPP_LIB="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [[ -s "$CPP_LIB" ]]; then
  "${ADB_CMD[@]}" push "$CPP_LIB" "$REMOTE/libc++_shared.so" >/dev/null
fi
"${ADB_CMD[@]}" shell "chmod 755 $REMOTE/rk3588_tops_android"
LOG="$ROOT/MTK_NPU_Benchmark/results/$(date +%Y%m%d_%H%M%S)_${PROFILE}_rk3588_int8_${BENCH_SECONDS}s.log"
echo "Report: $LOG"
for core in $CORE_MODES; do
  echo "===== RK3588 INT8 TOPS core=$core profile=$PROFILE ====="
  CMD="cd $REMOTE && LD_LIBRARY_PATH=$REMOTE ./rk3588_tops_android model.rknn $PROFILE $core $WARMUP $RUNS $BENCH_SECONDS"
  if command -v timeout >/dev/null 2>&1; then
    timeout "${RKNN_TIMEOUT:-600}" "${ADB_CMD[@]}" shell "$CMD" 2>&1 | tr -d '\r' | tee -a "$LOG"
  else
    "${ADB_CMD[@]}" shell "$CMD" 2>&1 | tr -d '\r' | tee -a "$LOG"
  fi
done
echo "DONE: verified quantized RKNN model-equivalent INT8 TOPS, not rated NPU counters."
