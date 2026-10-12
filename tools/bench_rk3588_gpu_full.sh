#!/usr/bin/env bash
# Native full-original KataGo 10b GPU OpenCL NN-only test on RK3588 Android.
# All 126 converted graph operations execute as OpenCL GPU kernels; 5 heads
# are checked against original FP32 ONNX references before timing is reported.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADB="${ADB:-adb}"
ADB_SERIAL="$(ADB="$ADB" bash "$ROOT/tools/select_adb_device.sh" rk)"
ADB_ARGS=("$ADB" -s "$ADB_SERIAL")
CACHE="${KATAGO_BENCH_CACHE:-$HOME/.cache/minigocpp/androidbench}"
PACKET="${KATAGO_FULL_PACKET:-$CACHE/katago10b_full_neuron_v1.bin}"
WARMUP="${GPU_WARMUP:-${WARMUP:-3}}"
RUNS="${GPU_RUNS:-${RUNS:-20}}"
GPU_CONV="${KATAGO_GPU_CONV:-tiled}"
case "$GPU_CONV" in direct|tiled) ;; *) echo "ERROR: KATAGO_GPU_CONV must be direct or tiled" >&2; exit 2 ;; esac
REMOTE="/data/local/tmp/minigocpp_gpu"
if [ ! -s "$PACKET" ]; then
    bash "$ROOT/tools/prepare_rk3588_compare.sh"
fi
NDK_DIR="${NDK:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}}"
if [ -z "$NDK_DIR" ]; then
    for d in "${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/27.2.12479018" \
             "${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}/ndk/27.2.12479018"; do
        if [ -d "$d" ]; then NDK_DIR="$d"; break; fi
    done
fi
CXX="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++"
[ -x "$CXX" ] || { echo "ERROR: Missing Android NDK compiler; set NDK" >&2; exit 1; }
mkdir -p "$CACHE"
BIN="$CACHE/gpu_katago_full_android"
if [ ! -x "$BIN" ] || [ "$ROOT/tools/gpu_katago_full_android.cpp" -nt "$BIN" ]; then
    echo "Compiling independent RK3588 OpenCL complete KataGo 10b benchmark..."
    "$CXX" -std=c++17 -O2 -fPIE -pie "$ROOT/tools/gpu_katago_full_android.cpp" -ldl -o "$BIN"
fi
"${ADB_ARGS[@]}" shell "mkdir -p $REMOTE"
"${ADB_ARGS[@]}" push "$BIN" "$REMOTE/gpu_katago_full_android" >/dev/null
"${ADB_ARGS[@]}" push "$PACKET" "$REMOTE/katago_full.bin" >/dev/null
CPP_LIB="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
[ ! -s "$CPP_LIB" ] || "${ADB_ARGS[@]}" push "$CPP_LIB" "$REMOTE/libc++_shared.so" >/dev/null
"${ADB_ARGS[@]}" shell "chmod 755 $REMOTE/gpu_katago_full_android"
echo "GPU_FULL_START adb=$ADB_SERIAL warmup=$WARMUP runs=$RUNS mode=$GPU_CONV model=KataGo10b/19x19"
"${ADB_ARGS[@]}" shell "cd $REMOTE && LD_LIBRARY_PATH=$REMOTE:/vendor/lib64 KATAGO_GPU_CONV=$GPU_CONV ./gpu_katago_full_android katago_full.bin $WARMUP $RUNS" | tr -d '\r'
