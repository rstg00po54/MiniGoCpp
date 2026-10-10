#!/usr/bin/env bash
# Test native FP16 or FP32 9x9 NHWC CONV_2D using MediaTek Neuron Adapter on vivo Android.
# This is NOT the full KataGo neural network.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADB="${ADB:-adb}"
ADB_SERIAL="$(ADB="$ADB" bash "$ROOT/tools/select_adb_device.sh" vivo)"
ADB_ARGS=("$ADB" -s "$ADB_SERIAL")
DEVICE="${1:-mtk-mdla}"
PRECISION="${2:-fp16}"
WARMUP="${WARMUP:-3}"
RUNS="${RUNS:-10}"
case "$DEVICE" in mtk-mdla|mtk-dsp) ;; *) echo "ERROR: device must be mtk-mdla or mtk-dsp" >&2; exit 2 ;; esac
case "$PRECISION" in fp16|fp32) ;; *) echo "ERROR: precision must be fp16 or fp32" >&2; exit 2 ;; esac
NDK_DIR="${NDK:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}}"
if [ -z "$NDK_DIR" ]; then
    for dir in "${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/27.2.12479018" \
               "${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}/ndk/27.2.12479018"; do
        if [ -d "$dir" ]; then NDK_DIR="$dir"; break; fi
    done
fi
CXX="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++"
if [ ! -x "$CXX" ]; then
    echo "ERROR: Cannot find NDK C++ compiler. Set NDK=/home/cc/Android/Sdk/ndk/27.2.12479018" >&2
    exit 1
fi

CACHE="${NEURON_BENCH_CACHE:-$HOME/.cache/minigocpp/neuron}"
mkdir -p "$CACHE"
EXE="$CACHE/neuron_conv_fp16_android"
if [ ! -x "$EXE" ] || [ "$ROOT/tools/neuron_conv_fp16_android.cpp" -nt "$EXE" ]; then
    echo "Compiling Android ARM64 Neuron CONV_2D..."
    "$CXX" -std=c++17 -O2 -fPIE -pie "$ROOT/tools/neuron_conv_fp16_android.cpp" -ldl -o "$EXE"
fi

REMOTE="/data/local/tmp/minigocpp_neuron"
echo "ADB device: $ADB_SERIAL; Neuron device: $DEVICE; precision: $PRECISION"
"${ADB_ARGS[@]}" shell "mkdir -p $REMOTE"
"${ADB_ARGS[@]}" push "$EXE" "$REMOTE/neuron_conv_fp16_android" >/dev/null
CPP_LIB="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
[ ! -s "$CPP_LIB" ] || "${ADB_ARGS[@]}" push "$CPP_LIB" "$REMOTE/libc++_shared.so" >/dev/null
"${ADB_ARGS[@]}" shell "chmod 755 $REMOTE/neuron_conv_fp16_android"
echo "Testing NHWC 9x9 3x3 CONV_2D ($PRECISION native); NOT KataGo NN evals/s."
CMD="cd $REMOTE && LD_LIBRARY_PATH=$REMOTE:/system_ext/lib64 ./neuron_conv_fp16_android $DEVICE $PRECISION $WARMUP $RUNS"
if command -v timeout >/dev/null 2>&1; then
    timeout "${NEURON_TIMEOUT:-120}" "${ADB_ARGS[@]}" shell "$CMD"
else
    "${ADB_ARGS[@]}" shell "$CMD"
fi
