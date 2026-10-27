#!/usr/bin/env bash
# ADB native NPU benchmark: auto-detect vivo/MediaTek vs RK3588/RKNN.
# vivo: FP16/INT8 effective convolution TOPS; RK3588: KataGo 10b NN/s (NOT TOPS).
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE="$ROOT/MTK_NPU_Benchmark"
ADB="${ADB:-adb}"
PROFILE="${1:-all}"
DEVICE="${2:-auto}"
PRECISION="${3:-fp16}"
WARMUP="${WARMUP:-20}"
RUNS="${RUNS:-100}"
BENCH_SECONDS="${BENCH_SECONDS:-0}"

die() { echo "ERROR: $*" >&2; exit 1; }
case "$PROFILE" in all|scale|go19|rk64|small|medium|large|heavy|x128|x128c512|x256|x256c512) ;; *)
    die "profile must be all/scale/go19/small/medium/large/heavy/x128/x128c512/x256/x256c512" ;;
esac
case "$DEVICE" in auto|mtk-mdla|mtk-dsp|rk) ;; *)
    die "device argument must be auto/mtk-mdla/mtk-dsp/rk" ;;
esac
case "$PRECISION" in fp16|int8) ;; *)
    die "precision must be fp16 or int8" ;;
esac
[[ "$WARMUP" =~ ^[0-9]+$ && "$RUNS" =~ ^[0-9]+$ && "$BENCH_SECONDS" =~ ^[0-9]+$ ]] || die "RUNS/WARMUP/BENCH_SECONDS must be integers"
(( ${#WARMUP} <= 5 && ${#RUNS} <= 5 && ${#BENCH_SECONDS} <= 3 )) || die "RUNS/WARMUP/BENCH_SECONDS out of range"
(( 10#$WARMUP <= 10000 && 10#$RUNS >= 1 && 10#$RUNS <= 10000 && 10#$BENCH_SECONDS <= 180 )) || die "RUNS must be 1..10000, WARMUP 0..10000 and BENCH_SECONDS 0..180"

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
    if [ "$PRECISION" = int8 ]; then
        case "$PROFILE" in rk64|heavy|x128|x256|x128c512|x256c512) ;; *)
            die "For RK3588 INT8 choose rk64/heavy/x128/x256/x128c512/x256c512 (e.g. x128 auto int8)" ;;
        esac
        echo "RK3588 -> native RKNN INT8 calibrated convolution benchmark"
        echo "RK3588 nominal total NPU peak is 6 TOPS INT8; reported results are measured model-equivalent throughput."
        if [ "${TOPS_DRY_RUN:-0}" = 1 ]; then exit 0; fi
        ADB="$ADB" ADB_SERIAL="$SERIAL" WARMUP="$WARMUP" RUNS="$RUNS" BENCH_SECONDS="$BENCH_SECONDS" \
            bash "$ROOT/tools/bench_rk3588_tops.sh" "$PROFILE"
        exit 0
    fi
    [ "$BENCH_SECONDS" = 0 ] || die "Timed stress available for Neuron or RK3588 INT8 only; old RKNN KataGo NN/s mode has no timer"
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

if [ "$PROFILE" = rk64 ]; then die "rk64 is a Rockchip-only INT8 profile; choose go19 for vivo"; fi

# Neuron Adapter benchmark for vivo. Default device must not be RKNN on vivo.
if [ "$DEVICE" = auto ]; then DEVICE="mtk-mdla"; fi
LOG="$HERE/results/$(date +%Y%m%d_%H%M%S)_${PROFILE}_${DEVICE}_${PRECISION}_${BENCH_SECONDS}s.log"
echo "vivo -> Neuron Adapter: $DEVICE precision=$PRECISION profile=$PROFILE warmup=$WARMUP runs=$RUNS bench_seconds=$BENCH_SECONDS"
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
    echo "Compiling MediaTek Neuron FP16/INT8 TOPS test for Android ARM64..."
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

# Use the same Neuron loader search paths as the previously working
# tools/bench_mediatek_katago_full.sh. Avoid shadowing system_ext dependencies
# with vendor libraries that happen to have the same sonames.
CMD="cd $REMOTE && LD_LIBRARY_PATH=$REMOTE:/system_ext/lib64 ./neuron_tops_android $PROFILE $DEVICE $WARMUP $RUNS $PRECISION $BENCH_SECONDS"
if command -v timeout >/dev/null 2>&1; then
    timeout "${NEURON_TIMEOUT:-600}" "${ADB_ARGS[@]}" shell "$CMD" 2>&1 | tee "$LOG"
else
    "${ADB_ARGS[@]}" shell "$CMD" 2>&1 | tee "$LOG"
fi
