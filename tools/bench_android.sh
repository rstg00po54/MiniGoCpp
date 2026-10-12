#!/usr/bin/env bash
# From Ubuntu/WSL: one ADB entry point for KataGo 10b on Android.
# CPU/NNAPI/RKNN report NN-only latency; GPU uses the existing KataGo search benchmark.
set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: benchmark launcher failed at line $LINENO (exit $rc): $BASH_COMMAND" >&2' ERR
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADB="${ADB:-adb}"
TARGET=auto
if [[ "${1:-}" == vivo || "${1:-}" == rk ]]; then TARGET="$1"; shift; fi
# Select by actual device properties, never by ADB connection order or serial format.
ADB_SERIAL="$(ADB="$ADB" bash "$ROOT/tools/select_adb_device.sh" "$TARGET")"
ADB_ARGS=("$ADB" -s "$ADB_SERIAL")
CACHE="${KATAGO_BENCH_CACHE:-$HOME/.cache/minigocpp/androidbench}"
MODEL="$CACHE/10b_19x19.onnx"
BASE="/data/local/tmp/minigocpp_onnx"
ORT_VERSION="1.23.2"
ORT_AAR="https://repo.maven.apache.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/$ORT_VERSION/onnxruntime-android-$ORT_VERSION.aar"
MODEL_URL="https://github.com/rstg00po54/go/releases/download/katago-10b-onnx-v1/10b_19x19.onnx"
RUNS="${RUNS:-100}"
WARMUP="${WARMUP:-20}"
THREADS="${THREADS:-4}"
LOG="${BENCH_LOG:-$ROOT/benchmark_android_$(date +%Y%m%d_%H%M%S).log}"
MODES="${*:-auto}"

fetch() {
    local url="$1" to="$2"
    if [ -s "$to" ]; then return; fi
    mkdir -p "$(dirname "$to")"
    if command -v curl >/dev/null 2>&1; then curl -fL --retry 3 -o "$to.tmp" "$url"
    else wget -q -O "$to.tmp" "$url"; fi
    mv "$to.tmp" "$to"
}
adb_cmd() { "${ADB_ARGS[@]}" "$@"; }
command -v "$ADB" >/dev/null || { echo "ERROR: adb missing" >&2; exit 1; }
command -v unzip >/dev/null || { echo "ERROR: unzip missing" >&2; exit 1; }
adb_cmd get-state >/dev/null || { echo "ERROR: No usable ADB device. Check adb devices -l, permissions, and ADB_SERIAL." >&2; exit 1; }
abi="$(adb_cmd shell getprop ro.product.cpu.abi | tr -d '\r')"
soc="$(adb_cmd shell 'getprop ro.soc.model; getprop ro.board.platform; getprop ro.hardware; cat /proc/device-tree/compatible 2>/dev/null || true' | tr '\000' ' ')"
brand="$(adb_cmd shell getprop ro.product.manufacturer | tr -d '\r')"
serial="$(adb_cmd get-serialno | tr -d '\r')"
echo "ADB: $serial manufacturer=$brand abi=$abi SOC=$soc"
[ "$abi" = arm64-v8a ] || { echo "ERROR: ARM64 Android only" >&2; exit 1; }
if [ "$MODES" = auto ]; then
    if [ "$TARGET" = vivo ]; then MODES="npu";
    elif [ "$TARGET" = rk ]; then MODES="rknn";
    elif [[ "${soc,,}" == *rk3588* ]]; then MODES="cpu nnapi rknn gpu";
    elif [[ "${soc,,}" == *mt[0-9]* ]] || [[ "${brand,,}" == *mediatek* ]] || [[ "${brand,,}" == *vivo* ]]; then MODES="probe cpu nnapi gpu";
    else MODES="cpu nnapi gpu"; fi
fi

mkdir -p "$CACHE"
downloaded=0
prepare_onnx() {
    if [ "$downloaded" = 1 ]; then return; fi
    fetch "$MODEL_URL" "$MODEL"
    fetch "$ORT_AAR" "$CACHE/onnxruntime-android-$ORT_VERSION.aar"
    local aar="$CACHE/onnxruntime-android-$ORT_VERSION.aar"
    if [ ! -s "$CACHE/ort/libonnxruntime.so" ] || [ ! -f "$CACHE/ort/headers/onnxruntime_cxx_api.h" ]; then
        mkdir -p "$CACHE/ort/headers"
        unzip -p "$aar" 'jni/arm64-v8a/libonnxruntime.so' > "$CACHE/ort/libonnxruntime.so"
        unzip -oq "$aar" 'headers/*' -d "$CACHE/ort"
        test -s "$CACHE/ort/headers/onnxruntime_cxx_api.h" || {
            echo "ERROR: ONNX Runtime AAR does not include C++ headers. Examine: unzip -l $aar" >&2
            exit 1
        }
    fi
    local ndk="${NDK:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}}"
    if [ -z "$ndk" ]; then
        for d in "${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/27.2.12479018" \
                 "${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}/ndk/27.2.12479018"; do
            [ ! -d "$d" ] || { ndk="$d"; break; }
        done
    fi
    local cc="$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++"
    [ -x "$cc" ] || { echo "ERROR: NDK compiler missing; export NDK=/home/cc/Android/Sdk/ndk/27.2.12479018" >&2; exit 1; }
    local exe="$CACHE/onnx_bench_android"
    if [ ! -x "$exe" ] || [ "$ROOT/tools/onnx_bench_android.cpp" -nt "$exe" ] || \
       [ "$ROOT/tools/rk3588_compare_input.h" -nt "$exe" ]; then
        echo "Compiling Android ARM64 ONNX Runtime benchmark..."
        "$cc" -std=c++17 -O2 -fPIE -pie -I"$CACHE/ort/headers" "$ROOT/tools/onnx_bench_android.cpp" \
            -L"$CACHE/ort" -lonnxruntime -lneuralnetworks -llog -ldl -Wl,-rpath,'$ORIGIN' -o "$exe"
    fi
    adb_cmd shell "mkdir -p $BASE"
    adb_cmd push "$MODEL" "$BASE/model.onnx" >/dev/null
    adb_cmd push "$exe" "$BASE/onnx_bench_android" >/dev/null
    adb_cmd push "$CACHE/ort/libonnxruntime.so" "$BASE/libonnxruntime.so" >/dev/null
    local cpp="$ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
    [ ! -s "$cpp" ] || adb_cmd push "$cpp" "$BASE/libc++_shared.so" >/dev/null
    adb_cmd shell "chmod 755 $BASE/onnx_bench_android"
    if [ -n "${COMPARE_INPUT_FILE:-}" ]; then
        [ -s "$COMPARE_INPUT_FILE" ] || { echo "ERROR: Missing shared input: $COMPARE_INPUT_FILE" >&2; exit 1; }
        adb_cmd push "$COMPARE_INPUT_FILE" "$BASE/katago_compare_input.bin" >/dev/null
    fi
    downloaded=1
}
nn_test() {
    prepare_onnx
    local m="$1"
    echo "===== $m: ONNX Runtime, NN only ====="
    if ! adb_cmd shell "cd $BASE && LD_LIBRARY_PATH=$BASE ./onnx_bench_android model.onnx $m $WARMUP $RUNS $THREADS ${COMPARE_INPUT_FILE:+katago_compare_input.bin}"; then
        echo "FAILED $m: check above error (NNAPI may not support this graph/device)."
        return 1
    fi
}
gpu_test() {
    echo "===== GPU OpenCL: KataGo MCTS benchmark (visits/s, NOT NN-evals/s) ====="
    local script="${KATAGO_GPU_SCRIPT:-}"
    if [ -z "$script" ]; then
        for c in "$ROOT/../GoEngine-java/tools/benchmark_katago_android.sh" \
                 "$ROOT/../go/GoEngine-java/tools/benchmark_katago_android.sh" \
                 "$ROOT/../GoEngine-java/tools/benchmark_katago_android.sh"; do
            [ ! -f "$c" ] || { script="$c"; break; }
        done
    fi
    if [ -z "$script" ] || [ ! -f "$script" ]; then
        echo "SKIP GPU: Existing KataGo OpenCL script not found."
        echo "Set KATAGO_GPU_SCRIPT=/path/to/GoEngine-java/tools/benchmark_katago_android.sh"
        return 0
    fi
    echo "Using existing GPU search benchmark: $script"
    ADB_SERIAL="$serial" ANDROID_SERIAL="$serial" KATAGO_BENCH_VISITS="${KATAGO_BENCH_VISITS:-200}" \
        KATAGO_BENCH_POSITIONS="${KATAGO_BENCH_POSITIONS:-5}" KATAGO_BENCH_THREADS="${KATAGO_BENCH_THREADS:-2}" \
        bash "$script" gpu
}
run_all() {
    echo "KataGo 10b Android ADB benchmark: target=$TARGET modes=$MODES"
    echo "RUNS=$RUNS WARMUP=$WARMUP THREADS=$THREADS"
    local error=0
    for m in $MODES; do
        case "$m" in
            probe) ADB_SERIAL="$serial" ADB="$ADB" bash "$ROOT/tools/probe_mediatek_npu.sh" || error=1 ;;
            npu)
                if [[ "${soc,,}" == *rk3588* ]]; then
                    echo "===== RK3588 NPU ====="
                    CORE_MODES="${CORE_MODES:-0 1 2 all}" RUNS="$RUNS" WARMUP="$WARMUP" ADB_SERIAL="$serial" \
                        bash "$ROOT/tools/bench_rk3588_npu.sh" || error=1
                else
                    echo "===== vivo/MediaTek NPU availability ====="
                    ADB_SERIAL="$serial" ADB="$ADB" bash "$ROOT/tools/probe_mediatek_npu.sh" || error=1
                    echo "===== Neuron Adapter: native FP32 ADD smoke test ====="
                    ADB_SERIAL="$serial" ADB="$ADB" bash "${NEURON_SMOKE_RUNNER:-$ROOT/tools/bench_mediatek_neuron.sh}" || error=1
                    echo "NPU_NOT_BENCHMARKED: This tested native Neuron API access with a tiny ADD graph, NOT KataGo 10b."
                    echo "KataGo must be converted and compiled for this MediaTek backend before claiming NN evals/s."
                    error=1
                fi ;;
            conv)
                if [[ "${soc,,}" == *rk3588* ]]; then
                    echo "ERROR: MediaTek Neuron CONV_2D mode is for vivo/MTK only." >&2
                    error=1
                else
                    ADB_SERIAL="$serial" ADB="$ADB" WARMUP="$WARMUP" RUNS="$RUNS" \
                        bash "$ROOT/tools/bench_mediatek_conv.sh" "${NEURON_DEVICE_NAME:-mtk-mdla}" || error=1
                fi ;;
            cpu|nnapi) nn_test "$m" || error=1 ;;
            gpu) gpu_test || error=1 ;;
            rknn)
                if [[ "${soc,,}" != *rk3588* ]]; then
                    echo "SKIP RKNN: This is not an RK3588 Android target."
                else
                    echo "===== RKNN: RK3588 NPU, NN-only ====="
                    CORE_MODES="${CORE_MODES:-0 1 2 all}" RUNS="$RUNS" WARMUP="$WARMUP" ADB_SERIAL="$serial" \
                        bash "$ROOT/tools/bench_rk3588_npu.sh" || error=1
                fi ;;
            *) echo "Unknown mode: $m (supported: auto npu conv probe cpu nnapi gpu rknn)" >&2; error=1 ;;
        esac
    done
    echo "===== DONE ====="
    echo "COMPARE ONLY like-for-like measurements: NN-only vs NN-only, MCTS visits/s vs MCTS visits/s."
    return "$error"
}
if run_all 2>&1 | tee "$LOG"; then
    echo "Benchmark log: $LOG"
else
    echo "One or more benchmarks failed; log: $LOG" >&2
    exit 1
fi
