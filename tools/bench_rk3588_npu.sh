#!/usr/bin/env bash
set -euo pipefail

# On RK3588 Ubuntu/Debian/Buildroot with Python3 and working RKNPU2 runtime:
#   bash tools/bench_rk3588_npu.sh
#   RUNS=200 WARMUP=30 bash tools/bench_rk3588_npu.sh path/to/model.rknn
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODEL="${1:-models/katago_10b_19x19.rknn}"
RUNS="${RUNS:-100}"
WARMUP="${WARMUP:-20}"
URL="https://github.com/rstg00po54/MiniGoCpp/releases/download/katago-10b-rknn-v1/katago_10b_19x19.rknn"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "ERROR: Run on RK3588 aarch64, not x86/VM." >&2
  exit 1
fi
if ! python3 -c 'import numpy; from rknnlite.api import RKNNLite' 2>/dev/null; then
  echo "Missing dependencies: python3 -m pip install --user 'numpy<2' rknn-toolkit-lite2==2.3.2" >&2
  echo "Also ensure RK3588 has /usr/lib/librknnrt.so or equivalent runtime installed." >&2
  exit 1
fi
if [ ! -s "$MODEL" ]; then
  echo "RKNN model not found; attempting download from GitHub Release..."
  mkdir -p "$(dirname "$MODEL")"
  curl -fL --retry 3 -o "$MODEL.part" "$URL" || { rm -f "$MODEL.part"; echo "Conversion Release not available yet; check GitHub Actions." >&2; exit 1; }
  mv "$MODEL.part" "$MODEL"
fi
echo "Checking NPU kernel driver..."
if [ -r /sys/kernel/debug/rknpu/version ]; then cat /sys/kernel/debug/rknpu/version; fi
for core in 0 all; do
  echo "===== RK3588 NPU core=$core ====="
  python3 tools/benchmark_rknn.py --model "$MODEL" --core "$core" --warmup "$WARMUP" --runs "$RUNS"
done
