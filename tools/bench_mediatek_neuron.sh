#!/usr/bin/env bash
# From Ubuntu/WSL: build native Android Neuron API smoke test and run via ADB.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADB="${ADB:-adb}"
ADB_SERIAL="$(ADB="$ADB" bash "$ROOT/tools/select_adb_device.sh" vivo)"
ADB_ARGS=("$ADB" -s "$ADB_SERIAL")
NDK_DIR="${NDK:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}}"
if [ -z "$NDK_DIR" ]; then
    for dir in "${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/27.2.12479018" \
               "${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}/ndk/27.2.12479018"; do
        if [ -d "$dir" ]; then NDK_DIR="$dir"; break; fi
    done
fi
CXX="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++"
if [ ! -x "$CXX" ]; then
    echo "ERROR: missing NDK C++ compiler. Set NDK=/home/cc/Android/Sdk/ndk/27.2.12479018" >&2
    exit 1
fi
CACHE="${NEURON_BENCH_CACHE:-$HOME/.cache/minigocpp/neuron}"
mkdir -p "$CACHE"
EXE="$CACHE/neuron_smoke_android"
if [ ! -x "$EXE" ] || [ "$ROOT/tools/neuron_smoke_android.cpp" -nt "$EXE" ]; then
    echo "Compiling Android ARM64 Neuron Adapter smoke test..."
    "$CXX" -std=c++17 -O2 -fPIE -pie "$ROOT/tools/neuron_smoke_android.cpp" \
        -ldl -o "$EXE"
fi

REMOTE="/data/local/tmp/minigocpp_neuron"
echo "ADB Neuron Adapter test: $ADB_SERIAL"
"${ADB_ARGS[@]}" shell "mkdir -p $REMOTE"
"${ADB_ARGS[@]}" push "$EXE" "$REMOTE/neuron_smoke_android" >/dev/null
CXXLIB="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [ -s "$CXXLIB" ]; then "${ADB_ARGS[@]}" push "$CXXLIB" "$REMOTE/libc++_shared.so" >/dev/null; fi
"${ADB_ARGS[@]}" shell "chmod 755 $REMOTE/neuron_smoke_android"
echo "Running small FP32 ADD model via Android Neuron Adapter (NOT KataGo NPU benchmark)."
if command -v timeout >/dev/null 2>&1; then
    timeout "${NEURON_TIMEOUT:-90}" "${ADB_ARGS[@]}" shell \
        "cd $REMOTE && LD_LIBRARY_PATH=$REMOTE:/system_ext/lib64 ./neuron_smoke_android"
else
    "${ADB_ARGS[@]}" shell "cd $REMOTE && LD_LIBRARY_PATH=$REMOTE:/system_ext/lib64 ./neuron_smoke_android"
fi
