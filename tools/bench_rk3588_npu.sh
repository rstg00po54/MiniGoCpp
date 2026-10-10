#!/usr/bin/env bash
# Run on Ubuntu/WSL host, using ADB to benchmark RK3588 Android NPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADB_BIN="${ADB:-adb}"
ADB_ARGS=("$ADB_BIN")
if [ -n "${ADB_SERIAL:-}" ]; then ADB_ARGS+=(-s "$ADB_SERIAL"); fi
CACHE="${RKNN_BENCH_CACHE:-$HOME/.cache/minigocpp/rknn_android}"
MODEL="${1:-$CACHE/katago_10b_19x19.rknn}"
WARMUP="${WARMUP:-20}"
RUNS="${RUNS:-100}"
REMOTE="/data/local/tmp/minigocpp_npu"
TAG="v2.3.2"
RUNTIME_BASE="https://raw.githubusercontent.com/airockchip/rknn-toolkit2/$TAG/rknpu2/runtime/Android/librknn_api"
MODEL_URL="https://github.com/rstg00po54/MiniGoCpp/releases/download/katago-10b-rknn-v1/katago_10b_19x19.rknn"

download() {
    local url="$1" target="$2"
    if [ -s "$target" ]; then return 0; fi
    mkdir -p "$(dirname "$target")"
    echo "Downloading $(basename "$target")..."
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 -o "$target.tmp" "$url" || { rm -f "$target.tmp"; return 1; }
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$target.tmp" "$url" || { rm -f "$target.tmp"; return 1; }
    else
        echo "ERROR: curl or wget is required" >&2
        return 1
    fi
    mv "$target.tmp" "$target"
}

command -v "$ADB_BIN" >/dev/null || { echo "ERROR: adb is not in PATH." >&2; exit 1; }
if ! "${ADB_ARGS[@]}" get-state >/dev/null; then
    echo "ERROR: Connect your RK3588 Android via adb; use ADB_SERIAL=xxx if multiple devices." >&2
    exit 1
fi
abi="$("${ADB_ARGS[@]}" shell getprop ro.product.cpu.abi | tr -d '\r')"
soc="$("${ADB_ARGS[@]}" shell 'getprop ro.board.platform; getprop ro.hardware; getprop ro.product.board; getprop ro.soc.model; cat /proc/device-tree/compatible 2>/dev/null' | tr '\000' ' ')"
echo "ADB target ABI: $abi"
echo "ADB target hardware: $soc"
if [ "$abi" != "arm64-v8a" ]; then echo "ERROR: Android target must be ARM64." >&2; exit 1; fi
if [[ "${soc,,}" != *rk3588* ]]; then
    echo "ERROR: Target is not identified as RK3588/RK3588S. RKNN model cannot run on vivo/MediaTek." >&2
    echo "For custom RK3588 Android boards with unusual properties, set ALLOW_UNKNOWN_SOC=1." >&2
    [ "${ALLOW_UNKNOWN_SOC:-0}" = "1" ] || exit 1
fi

mkdir -p "$CACHE"
if [ ! -s "$MODEL" ]; then
    if [ "$#" -gt 0 ]; then echo "ERROR: Model file not found: $MODEL" >&2; exit 1; fi
    download "$MODEL_URL" "$MODEL"
fi
download "$RUNTIME_BASE/include/rknn_api.h" "$CACHE/rknn_api.h"
download "$RUNTIME_BASE/arm64-v8a/librknnrt.so" "$CACHE/librknnrt.so"

# Use your installed NDK (default location of your Android SDK).
NDK_DIR="${NDK:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}}"
if [ -z "$NDK_DIR" ]; then
    for candidate in "${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/27.2.12479018" \
                     "${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}/ndk/27.2.12479018" \
                     "$HOME/Android/Sdk/ndk/27.2.12479018"; do
        if [ -d "$candidate" ]; then NDK_DIR="$candidate"; break; fi
    done
fi
if [ -z "$NDK_DIR" ]; then
    echo "ERROR: Android NDK not found. Set NDK=/home/cc/Android/Sdk/ndk/27.2.12479018" >&2
    exit 1
fi
CXX="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++"
[ -x "$CXX" ] || { echo "ERROR: Missing NDK compiler: $CXX" >&2; exit 1; }

BIN="$CACHE/rknn_bench_android"
if [ ! -x "$BIN" ] || [ "$ROOT/tools/rknn_bench_android.cpp" -nt "$BIN" ]; then
    echo "Building Android ARM64 NPU benchmark..."
    "$CXX" -std=c++17 -O2 -fPIE -pie -I"$CACHE" "$ROOT/tools/rknn_bench_android.cpp" \
        -L"$CACHE" -lrknnrt -llog -ldl -Wl,-rpath,'$ORIGIN' -o "$BIN"
fi

echo "Pushing native benchmark, RKNN model and Android runtime via ADB..."
"${ADB_ARGS[@]}" shell "mkdir -p $REMOTE"
"${ADB_ARGS[@]}" push "$BIN" "$REMOTE/rknn_bench_android"
"${ADB_ARGS[@]}" push "$MODEL" "$REMOTE/katago_10b_19x19.rknn"
"${ADB_ARGS[@]}" push "$CACHE/librknnrt.so" "$REMOTE/librknnrt.so"
LIBCPP="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [ -s "$LIBCPP" ]; then "${ADB_ARGS[@]}" push "$LIBCPP" "$REMOTE/libc++_shared.so"; fi
"${ADB_ARGS[@]}" shell "chmod 755 $REMOTE/rknn_bench_android"
# Default compares the three individual cores with the combined-core mode.
# Override: CORE_MODES="0 all" or CORE_MODES="all 2 1 0".
CORE_MODES="${CORE_MODES:-0 1 2 all}"
for core in $CORE_MODES; do
    echo "===== Android RK3588 NPU core=$core ====="
    "${ADB_ARGS[@]}" shell "cd $REMOTE && LD_LIBRARY_PATH=$REMOTE ./rknn_bench_android katago_10b_19x19.rknn $core $WARMUP $RUNS" | tr -d '\r'
done
echo "Completed. Results are NPU neural-network evaluations, NOT KataGo visits/s."
