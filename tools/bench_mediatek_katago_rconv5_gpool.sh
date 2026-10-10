#!/usr/bin/env bash
# Actual KataGo 10b rconv5 global pooling branch on device-pinned MediaTek Neuron.
# MEAN, scaled MEAN, REDUCE_MAX, CONCAT and original 1x1 rconv5/w1r weights.
# Only this branch, NOT full KataGo NN evals/s.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADB="${ADB:-adb}"
ADB_SERIAL="$(ADB="$ADB" bash "$ROOT/tools/select_adb_device.sh" vivo)"
ADB_ARGS=("$ADB" -s "$ADB_SERIAL")
DEVICE="${1:-mtk-mdla}"
case "$DEVICE" in mtk-mdla|mtk-dsp) ;; *) echo "ERROR: device must be mtk-mdla or mtk-dsp" >&2; exit 2 ;; esac
WARMUP="${WARMUP:-20}"
RUNS="${RUNS:-100}"
CACHE="${KATAGO_BENCH_CACHE:-$HOME/.cache/minigocpp/androidbench}"
MODEL="${KATAGO_ONNX_MODEL:-$CACHE/10b_19x19.onnx}"
PACKET="$CACHE/katago_rconv5_gpool_v1.bin"
mkdir -p "$CACHE"
URL="https://github.com/rstg00po54/go/releases/download/katago-10b-onnx-v1/10b_19x19.onnx"
if [ ! -s "$MODEL" ]; then
    echo "Downloading original KataGo 10b ONNX model..."
    mkdir -p "$(dirname "$MODEL")"
    curl -fL --retry 3 -o "$MODEL.tmp" "$URL"
    mv "$MODEL.tmp" "$MODEL"
fi

PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c 'import onnx, onnxruntime, numpy' >/dev/null 2>&1; then
    VENV="$CACHE/katago_export_venv"
    if [ ! -x "$VENV/bin/python" ]; then
        "$PYTHON" -m venv "$VENV" || {
            echo "ERROR: Python venv unavailable. Install python3-venv or use PYTHON in a working venv." >&2
            exit 1
        }
    fi
    PYTHON="$VENV/bin/python"
    if ! "$PYTHON" -c 'import onnx, onnxruntime, numpy' >/dev/null 2>&1; then
        ORT_VERSION="$("$PYTHON" -c 'import sys; print("1.16.3" if sys.version_info < (3, 9) else "1.19.2")')"
        "$PYTHON" -m pip install 'numpy==1.24.4' 'onnx==1.16.2' "onnxruntime==$ORT_VERSION"
    fi
fi
if [ ! -s "$PACKET" ] || [ "$ROOT/tools/export_katago_rconv5_gpool.py" -nt "$PACKET" ] || \
   [ "$MODEL" -nt "$PACKET" ] || [ "${FORCE_EXPORT:-0}" = 1 ]; then
    echo "Exporting real KataGo rconv5 global pooling branch, original w1r weights and ONNX CPU golden..."
    "$PYTHON" "$ROOT/tools/export_katago_rconv5_gpool.py" --model "$MODEL" --out "$PACKET"
fi

NDK_DIR="${NDK:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}}"
if [ -z "$NDK_DIR" ]; then
    for d in "${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/27.2.12479018" \
             "${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}/ndk/27.2.12479018"; do
        if [ -d "$d" ]; then NDK_DIR="$d"; break; fi
    done
fi
CXX="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++"
[ -x "$CXX" ] || { echo "ERROR: Android NDK compiler missing" >&2; exit 1; }
BIN="$CACHE/neuron_katago_rconv5_gpool_android"
if [ ! -x "$BIN" ] || [ "$ROOT/tools/neuron_katago_rconv5_gpool_android.cpp" -nt "$BIN" ]; then
    echo "Compiling native MediaTek FP16 rconv5 GPool benchmark..."
    "$CXX" -std=c++17 -O2 -fPIE -pie "$ROOT/tools/neuron_katago_rconv5_gpool_android.cpp" -ldl -o "$BIN"
fi
if ! grep -aFq 'K10GP001' "$BIN" || ! grep -aFq 'KATAGO_GPOOL_GRAPH' "$BIN"; then
    echo "ERROR: compiled Neuron executable identity does not match rconv5 GPool runner" >&2
    rm -f "$BIN"
    exit 1
fi
REMOTE="/data/local/tmp/minigocpp_neuron"
echo "ADB vivo=$ADB_SERIAL pinned_device=$DEVICE graph=KataGo_rconv5_globalpool_FP16"
"${ADB_ARGS[@]}" shell "mkdir -p $REMOTE"
"${ADB_ARGS[@]}" push "$BIN" "$REMOTE/neuron_katago_rconv5_gpool_android" >/dev/null
"${ADB_ARGS[@]}" push "$PACKET" "$REMOTE/katago_rconv5_gpool.bin" >/dev/null
CPP_LIB="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [ -s "$CPP_LIB" ]; then "${ADB_ARGS[@]}" push "$CPP_LIB" "$REMOTE/libc++_shared.so" >/dev/null; fi
"${ADB_ARGS[@]}" shell "chmod 755 $REMOTE/neuron_katago_rconv5_gpool_android"
CMD="cd $REMOTE && LD_LIBRARY_PATH=$REMOTE:/system_ext/lib64 ./neuron_katago_rconv5_gpool_android katago_rconv5_gpool.bin $DEVICE $WARMUP $RUNS"
if command -v timeout >/dev/null 2>&1; then
    timeout "${NEURON_TIMEOUT:-240}" "${ADB_ARGS[@]}" shell "$CMD"
else
    "${ADB_ARGS[@]}" shell "$CMD"
fi
