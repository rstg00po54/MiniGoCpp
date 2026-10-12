#!/usr/bin/env bash
# Prepare full original KataGo 10b neural network packet and one common NCHW FP32
# input/reference for CPU and RKNN. GPU runs the same FP16 packet directly.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${KATAGO_BENCH_CACHE:-$HOME/.cache/minigocpp/androidbench}"
MODEL="${KATAGO_ONNX_MODEL:-$CACHE/10b_19x19.onnx}"
PACKET="${KATAGO_FULL_PACKET:-$CACHE/katago10b_full_neuron_v1.bin}"
SHARED="${COMPARE_INPUT_FILE:-$CACHE/katago10b_shared_input_v1.bin}"
mkdir -p "$CACHE" "$(dirname "$MODEL")" "$(dirname "$PACKET")" "$(dirname "$SHARED")"
if [ ! -s "$MODEL" ]; then
    echo "Downloading original 10b ONNX to local cache..."
    curl -fL --retry 3 -o "$MODEL.tmp" \
      https://github.com/rstg00po54/go/releases/download/katago-10b-onnx-v1/10b_19x19.onnx
    mv "$MODEL.tmp" "$MODEL"
fi
PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c 'import numpy,onnx,onnxruntime' >/dev/null 2>&1; then
    VENV="$CACHE/katago_export_venv"
    [ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
    PYTHON="$VENV/bin/python"
    if ! "$PYTHON" -c 'import numpy,onnx,onnxruntime' >/dev/null 2>&1; then
        ORT_VERSION="$("$PYTHON" -c 'import sys;print("1.16.3" if sys.version_info<(3,9) else "1.19.2")')"
        "$PYTHON" -m pip install 'numpy==1.24.4' 'onnx==1.16.2' "onnxruntime==$ORT_VERSION"
    fi
fi
if [ ! -s "$PACKET" ] || [ "$ROOT/tools/export_katago_neuron_full.py" -nt "$PACKET" ] || \
   [ "$MODEL" -nt "$PACKET" ] || [ "${FORCE_EXPORT:-0}" = 1 ]; then
    "$PYTHON" "$ROOT/tools/export_katago_neuron_full.py" --model "$MODEL" --out "$PACKET"
fi
if [ ! -s "$SHARED" ] || [ "$ROOT/tools/prepare_rk3588_compare_input.py" -nt "$SHARED" ] || \
   [ "$PACKET" -nt "$SHARED" ] || [ "${FORCE_EXPORT:-0}" = 1 ]; then
    "$PYTHON" "$ROOT/tools/prepare_rk3588_compare_input.py" --packet "$PACKET" --out "$SHARED"
fi
echo "RK3588_COMPARE_READY packet=$PACKET input=$SHARED onnx=$MODEL"
