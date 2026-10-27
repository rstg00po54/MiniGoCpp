#!/usr/bin/env bash
# ADB native NPU benchmark: auto-detect vivo/MediaTek vs RK3588/RKNN.
# vivo: FP16 effective convolution TOPS; RK3588: KataGo 10b NN/s (NOT TOPS).
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE="$ROOT/MTK_NPU_Benchmark"
ADB="${ADB:-adb}"
PROFILE="${1:-all}"
DEVICE="${2:-auto}"
WARMUP="${WARMUP:-20}"
RUNS="${RUNS:-100}"

die() { echo "ERROR: $*" >&2; exit 1; }
case "$PROFILE" in all|go19|small|medium|large|heavy) ;; *)
    die "profile must be all/go19/small/medium/large/heavy" ;;
esac
case "$DEVICE" in auto|mtk-mdla|mtk-dsp|rk) ;; *)
    die "device argument must be auto/mtk-mdla/mtk-dsp/rk" ;;
esac
[[ "$WARMUP" =~ ^[0-9]+$ && "$RUNS" =~ ^[0-9]+$ ]] || die "RUNS/WARMUP must be integers"
(( ${#WARMUP} <= 5 && ${#RUNS} <= 5 )) || die "RUNS/WARMUP out of range"
(( 10#$WARMUP <= 10000 && 10#$RUNS >= 1 && 10#$RUNS <= 10000 )) || die "RUNS must be 1..10000 and WARMUP 0..10000"

# The existing selector already handles ADB_SERIAL and rejects ambiguous devices.
SERIAL="$(ADB="$ADB" bash "$ROOT/tools/select_adb_device.sh" auto)"
ADB_ARGS=("$ADB" -s "$SERIAL")
INFO="$("${ADB_ARGS[@]}" shell '
    getprop ro.product.manufacturer
    getprop ro.product.brand
    getprop ro.product.model
    getprop ro.soc.model
    getprop ro.board.platform
    getprop ro.hardware
    getprop ro.product.board
    cat /proc/device-tree/compatible 2>/dev/null || true
' | tr -d '\r\000')"
INFO_LOWER="${INFO,,}"
if [[ "$INFO_LOWER" == *rk3588* ]]; then
    BACKEND="rk"
elif [[ "$INFO_LOWER" == *vivo* ]]; then
    BACKEND="vivo"
else
    die "Unknown device: ${INFO//$'\n'/ | }. Supports vivo/MediaTek and RK3588/RK3588S only."
fi

case "$BACKEND:$DEVICE" in
    vivo:rk|rk:mtk-mdla|rk:mtk-dsp)
        die "Selected ADB device is $BACKEND, but requested incompatible backend $DEVICE" ;;
esac
mkdir -p "$HERE/results"
echo "Selected ADB: $SERIAL"
echo "Detected backend: $BACKEND (${INFO//$'\n'/ | })"

if [ "$BACKEND" = rk ]; then
    case "$PROFILE" in all|go19) ;; *)
        die "RK3588 currently supports only profile all/go19 (KataGo 10b 19x19 RKNN NN benchmark), not $PROFILE. RKNN result is NN/s, not TOPS." ;;
    esac
    LOG="$HERE/results/$(date +%Y%m%d_%H%M%S)_rk3588.log"
    echo "RK3588 -> RKNN KataGo 10b 19x19 (core modes: ${CORE_MODES:-0 1 2 all})"
    echo "IMPORTANT: RKNN outputs inference latency/NN evals per second, NOT TOPS."
    echo "Report: $LOG"
    if [ "${TOPS_DRY_RUN:-0}" = 1 ]; then exit 0; fi
    # Reuse the already working RK3588 model/benchmark and its local cache.
    ADB="$ADB" ADB_SERIAL="$SERIAL" WARMUP="$WARMUP" RUNS="$RUNS"         CORE_MODES="${CORE_MODES:-0 1 2 all}"         bash "$ROOT/tools/bench_rk3588_npu.sh" 2>&1 | tee "$LOG"
    exit 0
fi

# Neuron Adapter benchmark for vivo. Default device must not be RKNN on vivo.
if [ "$DEVICE" = auto ]; then DEVICE="mtk-mdla"; fi
LOG="$HERE/results/$(date +%Y%m%d_%H%M%S)_${PROFILE}_${DEVICE}.log"
echo "vivo -> Neuron Adapter: $DEVICE profile=$PROFILE warmup=$WARMUP runs=$RUNS"
echo "Report: $LOG"
if [ "${TOPS_DRY_RUN:-0}" = 1 ]; then exit 0; fi

NDK_DIR="${NDK:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}}"
if [ -z "$NDK_DIR" ]; then
    for dir in "${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/27.2.12479018" "${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}/ndk/27.2.12479018"; do
        if [ -d "$dir" ]; then NDK_DIR="$dir"; break; fi
    done
fi
CXX="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++"
[ -x "$CXX" ] || die "Android NDK compiler not found. Set NDK=/path/to/ndk/27.2.12479018"
mkdir -p "$HERE/build"
BIN="$HERE/build/neuron_tops_android"
if [ ! -x "$BIN" ] || [ "$HERE/neuron_tops_android.cpp" -nt "$BIN" ]; then
    echo "Compiling MediaTek Neuron FP16 TOPS test for Android ARM64..."
    "$CXX" -std=c++17 -O2 -fPIE -pie "$HERE/neuron_tops_android.cpp" -ldl -o "$BIN"
fi

REMOTE="/data/local/tmp/minigocpp_tops"
"${ADB_ARGS[@]}" shell "mkdir -p $REMOTE"
"${ADB_ARGS[@]}" push "$BIN" "$REMOTE/neuron_tops_android" >/dev/null
CPP_LIB="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [ -s "$CPP_LIB" ]; then
    "${ADB_ARGS[@]}" push "$CPP_LIB" "$REMOTE/libc++_shared.so" >/dev/null
fi
"${ADB_ARGS[@]}" shell "chmod 755 $REMOTE/neuron_tops_android"

CMD="cd $REMOTE && LD_LIBRARY_PATH=$REMOTE:/system_ext/lib64:/vendor/lib64 ./neuron_tops_android $PROFILE $DEVICE $WARMUP $RUNS"
if command -v timeout >/dev/null 2>&1; then
    timeout "${NEURON_TIMEOUT:-300}" "${ADB_ARGS[@]}" shell "$CMD" 2>&1 | tee "$LOG"
else
    "${ADB_ARGS[@]}" shell "$CMD" 2>&1 | tee "$LOG"
fi
