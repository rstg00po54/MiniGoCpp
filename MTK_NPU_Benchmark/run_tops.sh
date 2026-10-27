#!/usr/bin/env bash
# Build standalone ARM64 Neuron FP16 TOPS probe, then push and run using ADB.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE="$ROOT/MTK_NPU_Benchmark"
ADB="${ADB:-adb}"
PROFILE="${1:-all}"
DEVICE="${2:-mtk-mdla}"
WARMUP="${WARMUP:-20}"
RUNS="${RUNS:-100}"
case "$PROFILE" in all|go19|small|medium|large|heavy) ;; *)
    echo "ERROR: profile must be all/go19/small/medium/large/heavy" >&2; exit 2 ;;
esac
case "$DEVICE" in mtk-mdla|mtk-dsp) ;; *)
    echo "ERROR: device must be mtk-mdla or mtk-dsp" >&2; exit 2 ;;
esac
for number in "$WARMUP" "$RUNS"; do
    [[ "$number" =~ ^[0-9]+$ ]] || { echo "ERROR: RUNS/WARMUP must be integers" >&2; exit 2; }
done
(( WARMUP <= 10000 && RUNS >= 1 && RUNS <= 10000 )) || {
    echo "ERROR: RUNS must be 1..10000 and WARMUP 0..10000" >&2; exit 2;
}
if [ -n "${ADB_SERIAL:-}" ]; then
    SERIAL="$ADB_SERIAL"
else
    SERIAL="$(ADB="$ADB" bash "$ROOT/tools/select_adb_device.sh" vivo)"
fi
ADB_ARGS=("$ADB" -s "$SERIAL")

NDK_DIR="${NDK:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}}"
if [ -z "$NDK_DIR" ]; then
    for dir in "${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/27.2.12479018" "${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}/ndk/27.2.12479018"; do
        if [ -d "$dir" ]; then NDK_DIR="$dir"; break; fi
    done
fi
CXX="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++"
[ -x "$CXX" ] || {
    echo "ERROR: Android NDK compiler not found. Set NDK=/path/to/ndk/27.2.12479018" >&2
    exit 1
}
mkdir -p "$HERE/build" "$HERE/results"
BIN="$HERE/build/neuron_tops_android"
if [ ! -x "$BIN" ] || [ "$HERE/neuron_tops_android.cpp" -nt "$BIN" ]; then
    echo "Compiling MediaTek Neuron FP16 TOPS test for Android ARM64..."
    "$CXX" -std=c++17 -O2 -fPIE -pie "$HERE/neuron_tops_android.cpp" -ldl -o "$BIN"
fi

REMOTE="/data/local/tmp/minigocpp_tops"
echo "ADB=$SERIAL selected_device=$DEVICE profile=$PROFILE warmup=$WARMUP runs=$RUNS"
"${ADB_ARGS[@]}" shell "mkdir -p $REMOTE"
"${ADB_ARGS[@]}" push "$BIN" "$REMOTE/neuron_tops_android" >/dev/null
CPP_LIB="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [ -s "$CPP_LIB" ]; then
    "${ADB_ARGS[@]}" push "$CPP_LIB" "$REMOTE/libc++_shared.so" >/dev/null
fi
"${ADB_ARGS[@]}" shell "chmod 755 $REMOTE/neuron_tops_android"

LOG="$HERE/results/$(date +%Y%m%d_%H%M%S)_${PROFILE}_${DEVICE}.log"
echo "Report: $LOG"
CMD="cd $REMOTE && LD_LIBRARY_PATH=$REMOTE:/system_ext/lib64:/vendor/lib64 ./neuron_tops_android $PROFILE $DEVICE $WARMUP $RUNS"
if command -v timeout >/dev/null 2>&1; then
    timeout "${NEURON_TIMEOUT:-300}" "${ADB_ARGS[@]}" shell "$CMD" 2>&1 | tee "$LOG"
else
    "${ADB_ARGS[@]}" shell "$CMD" 2>&1 | tee "$LOG"
fi
