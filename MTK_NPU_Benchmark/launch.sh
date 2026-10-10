#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -s "$HERE/neuron_bench.cpp" || ! -s "$HERE/run.sh" ]]; then
    tar -xJf "$HERE/source.tar.xz" --strip-components=1 -C "$HERE" \
        MTK_NPU_Benchmark/neuron_bench.cpp MTK_NPU_Benchmark/run.sh
fi
exec bash "$HERE/run.sh" "$@"
