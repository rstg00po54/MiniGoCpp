#!/usr/bin/env bash
# Run MiniGoCpp's own 3x3 RKNN model on RK3588/RK3588S Android over ADB.
# Run this script on Ubuntu x86 with Android NDK installed.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-$ROOT/models/my3.rknn}"
ADB_BIN="${ADB:-adb}"
CACHE="${MINIGO_RKNN_CACHE:-$HOME/.cache/minigocpp/rknn_android}"
REMOTE="/data/local/tmp/minigocpp_mini_npu"
WARMUP="${WARMUP:-20}"
RUNS="${RUNS:-100}"
CORE_MODES="${CORE_MODES:-0}"
TAG="v2.3.2"
RUNTIME_URL="https://raw.githubusercontent.com/airockchip/rknn-toolkit2/$TAG/rknpu2/runtime/Android/librknn_api"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -s "$MODEL" ]] || die "Model missing: $MODEL (first convert models/my3.bin on Ubuntu)"
[[ "$WARMUP" =~ ^[0-9]+$ && "$RUNS" =~ ^[0-9]+$ ]] || die "WARMUP and RUNS must be integers"
(( RUNS > 0 && RUNS <= 100000 )) || die "RUNS must be between 1 and 100000"
command -v "$ADB_BIN" >/dev/null || die "adb is not installed"
SERIAL="$(ADB="$ADB_BIN" bash "$ROOT/tools/select_adb_device.sh" rk)"
ADB_CMD=("$ADB_BIN" -s "$SERIAL")
ABI="$("${ADB_CMD[@]}" shell getprop ro.product.cpu.abi | tr -d '\r')"
[[ "$ABI" == arm64-v8a ]] || die "Expected Android ARM64 device, got $ABI"

NDK_DIR="${NDK:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}}"
if [[ -z "$NDK_DIR" ]]; then
    for candidate in "${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/27.2.12479018" \
                     "${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}/ndk/27.2.12479018" \
                     "$HOME/Android/Sdk/ndk/27.2.12479018"; do
        [[ ! -d "$candidate" ]] || { NDK_DIR="$candidate"; break; }
    done
fi
[[ -n "$NDK_DIR" ]] || die "Android NDK missing. Set NDK=/home/cc/Android/Sdk/ndk/27.2.12479018"
CXX="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++"
[[ -x "$CXX" ]] || die "Android NDK C++ compiler missing: $CXX"

download() {
    local from="$1" to="$2"
    [[ -s "$to" ]] && return 0
    mkdir -p "$(dirname "$to")"
    echo "Downloading $(basename "$to")..."
    if command -v curl >/dev/null; then
        curl -fL --retry 3 --connect-timeout 15 "$from" -o "$to.tmp" || die "Download failed: $from"
    elif command -v wget >/dev/null; then
        wget -q "$from" -O "$to.tmp" || die "Download failed: $from"
    else
        die "curl or wget required"
    fi
    mv "$to.tmp" "$to"
}
mkdir -p "$CACHE"
download "$RUNTIME_URL/include/rknn_api.h" "$CACHE/rknn_api.h"
download "$RUNTIME_URL/arm64-v8a/librknnrt.so" "$CACHE/librknnrt.so"
EXE="$CACHE/minigo_rknn_android"
if [[ ! -x "$EXE" || "$ROOT/tools/minigo_rknn_android.cpp" -nt "$EXE" || "$CACHE/rknn_api.h" -nt "$EXE" ]]; then
    echo "Building RK3588 Android ARM64 runner..."
    "$CXX" -std=c++17 -O2 -fPIE -pie -I"$CACHE" \
        "$ROOT/tools/minigo_rknn_android.cpp" -L"$CACHE" -lrknnrt -llog -ldl \
        -Wl,-rpath,'$ORIGIN' -o "$EXE"
fi
echo "Selected RK3588 Android ADB: $SERIAL"
"${ADB_CMD[@]}" shell "mkdir -p $REMOTE"
"${ADB_CMD[@]}" push "$MODEL" "$REMOTE/my3.rknn" >/dev/null
"${ADB_CMD[@]}" push "$EXE" "$REMOTE/minigo_rknn_android" >/dev/null
"${ADB_CMD[@]}" push "$CACHE/librknnrt.so" "$REMOTE/librknnrt.so" >/dev/null
LIBCPP="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [[ -s "$LIBCPP" ]]; then
    "${ADB_CMD[@]}" push "$LIBCPP" "$REMOTE/libc++_shared.so" >/dev/null
fi
"${ADB_CMD[@]}" shell "chmod 755 $REMOTE/minigo_rknn_android"

for core in $CORE_MODES; do
    case "$core" in 0|1|2|all) ;; *) die "CORE_MODES must be space-separated 0/1/2/all" ;; esac
    echo "===== MiniGo 3x3 NPU core=$core ====="
    "${ADB_CMD[@]}" shell "cd $REMOTE && LD_LIBRARY_PATH=$REMOTE ./minigo_rknn_android my3.rknn $core $WARMUP $RUNS" | tr -d '\r'
done
echo "DONE: RKNN inference on RK3588 Android (not full Go/MCTS)."
