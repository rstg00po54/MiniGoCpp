#!/usr/bin/env bash
# Prepare the real KataGo 10b Neuron model asset for the GoEngine-java Android APK.
# The model contains all original 10b weights plus a full-19x19 reference.
# It does not yet connect the engine's GTP search to the JNI backend.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GOENGINE_DIR="${GOENGINE_DIR:-$(dirname "$ROOT")/GoEngine-java}"
CACHE="${KATAGO_BENCH_CACHE:-$HOME/.cache/minigocpp/androidbench}"
MODEL="${KATAGO_ONNX_MODEL:-$CACHE/10b_19x19.onnx}"
DEST="$GOENGINE_DIR/app/src/main/assets/neuron/katago10b_19x19.bin"
if [ ! -f "$GOENGINE_DIR/app/build.gradle.kts" ]; then
    echo "ERROR: GoEngine-java not found: $GOENGINE_DIR (set GOENGINE_DIR=/path/to/GoEngine-java)" >&2
    exit 1
fi
mkdir -p "$CACHE" "$(dirname "$MODEL")"
if [ ! -s "$MODEL" ]; then
    echo "Downloading original KataGo 10b 19x19 ONNX model..."
    curl -fL --retry 3 -o "$MODEL.tmp" \
        "https://github.com/rstg00po54/go/releases/download/katago-10b-onnx-v1/10b_19x19.onnx"
    mv "$MODEL.tmp" "$MODEL"
fi
PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c 'import numpy,onnx,onnxruntime' >/dev/null 2>&1; then
    VENV="$CACHE/katago_export_venv"
    if [ ! -x "$VENV/bin/python" ]; then "$PYTHON" -m venv "$VENV"; fi
    PYTHON="$VENV/bin/python"
    if ! "$PYTHON" -c 'import numpy,onnx,onnxruntime' >/dev/null 2>&1; then
        ORT_VERSION="$("$PYTHON" -c 'import sys; print("1.16.3" if sys.version_info < (3,9) else "1.19.2")')"
        "$PYTHON" -m pip install 'numpy==1.24.4' 'onnx==1.16.2' "onnxruntime==$ORT_VERSION"
    fi
fi
mkdir -p "$(dirname "$DEST")"
echo "Exporting original KataGo 10b MDLA graph into Android assets..."
"$PYTHON" "$ROOT/tools/export_katago_neuron_full.py" --model "$MODEL" --out "$DEST"
"$PYTHON" - "$DEST" <<'PY'
import pathlib,struct,sys
p=pathlib.Path(sys.argv[1]);b=p.read_bytes()[:24]
magic,operands,operations,inputs,outputs=struct.unpack("<8s4I",b)
assert magic==b"K10NW001" and operations==126 and inputs==2 and outputs==5,(magic,operations,inputs,outputs)
print("KataGo 10b JNI asset ready:",p,"size:",p.stat().st_size,"operands:",operands)
PY
echo "Android project: $GOENGINE_DIR"
echo "Build APK with: (cd \"$GOENGINE_DIR\" && ./gradlew assembleDebug)"
echo "NOTE: packaging this model alone does NOT switch GTP genmove to MDLA; KataGo's native NN evaluator must call the JNI bridge."
