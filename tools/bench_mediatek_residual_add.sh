#!/usr/bin/env bash
# Probe MediaTek MDLA/DSP FP16 residual ADD for 19x19x128, not full KataGo.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADB="${ADB:-adb}"
ADB_SERIAL="$(ADB="$ADB" bash "$ROOT/tools/select_adb_device.sh" vivo)"
ADB_ARGS=("$ADB" -s "$ADB_SERIAL")
DEVICE="${1:-mtk-mdla}"
case "$DEVICE" in mtk-mdla|mtk-dsp) ;; *) echo "ERROR: device must be mtk-mdla or mtk-dsp" >&2; exit 2 ;; esac
WARMUP="${WARMUP:-20}"
RUNS="${RUNS:-100}"
NDK_DIR="${NDK:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}}"
if [ -z "$NDK_DIR" ]; then
    for d in "${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/27.2.12479018" \
             "${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}/ndk/27.2.12479018"; do
        if [ -d "$d" ]; then NDK_DIR="$d"; break; fi
    done
fi
CXX="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++"
[ -x "$CXX" ] || { echo "ERROR: Android NDK compiler missing" >&2; exit 1; }
CACHE="${NEURON_BENCH_CACHE:-$HOME/.cache/minigocpp/neuron}"
mkdir -p "$CACHE"
BIN="$CACHE/neuron_residual_add_android"
if [ ! -x "$BIN" ] || [ "$ROOT/tools/neuron_residual_add_android.cpp" -nt "$BIN" ]; then
    echo "Compiling Android ARM64 FP16 residual ADD..."
    "$CXX" -std=c++17 -O2 -fPIE -pie "$ROOT/tools/neuron_residual_add_android.cpp" -ldl -o "$BIN"
fi
REMOTE="/data/local/tmp/minigocpp_neuron"
echo "Selected ADB=$ADB_SERIAL Neuron device=$DEVICE shape=1x19x19x128 fp16 ADD"
"${ADB_ARGS[@]}" shell "mkdir -p $REMOTE"
"${ADB_ARGS[@]}" push "$BIN" "$REMOTE/neuron_residual_add_android" >/dev/null
CPP="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [ -s "$CPP" ]; then "${ADB_ARGS[@]}" push "$CPP" "$REMOTE/libc++_shared.so" >/dev/null; fi
"${ADB_ARGS[@]}" shell "chmod 755 $REMOTE/neuron_residual_add_android"
CMD="cd $REMOTE && LD_LIBRARY_PATH=$REMOTE:/system_ext/lib64 ./neuron_residual_add_android $DEVICE $WARMUP $RUNS"
if command -v timeout >/dev/null 2>&1; then
    timeout "${NEURON_TIMEOUT:-120}" "${ADB_ARGS[@]}" shell "$CMD"
else
    "${ADB_ARGS[@]}" shell "$CMD"
fi
