#!/usr/bin/env bash
# RK3588 Android: KataGo OFFICIAL OpenCL NN-only benchmark, no MCTS or custom kernels.
# Supports a host native Android executable/model or pre-existing on-device paths.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADB="${ADB:-adb}"
ADB_SERIAL="$(ADB="$ADB" bash "$ROOT/tools/select_adb_device.sh" rk)"
ADB_ARGS=("$ADB" -s "$ADB_SERIAL")
RUNS="${GPU_RUNS:-${RUNS:-30}}"
WARMUP="${GPU_WARMUP:-${WARMUP:-5}}"
REMOTE="${KATAGO_REMOTE_DIR:-/data/local/tmp/minigocpp_katago_native}"
BIN="${KATAGO_BIN:-}"
MODEL="${KATAGO_MODEL:-}"
CFG="${KATAGO_CONFIG:-}"
REMOTE_BIN="${KATAGO_REMOTE_BIN:-}"
REMOTE_MODEL="${KATAGO_REMOTE_MODEL:-}"
REMOTE_CFG="${KATAGO_REMOTE_CONFIG:-}"
case "$RUNS:$WARMUP" in *[!0-9:]*|:*|*:) echo "ERROR: GPU_RUNS and GPU_WARMUP must be integers" >&2; exit 2 ;; esac
if [ "$RUNS" -lt 1 ] || [ "$WARMUP" -lt 0 ]; then
    echo "ERROR: GPU_RUNS must be positive, GPU_WARMUP must be nonnegative" >&2; exit 2
fi
case "$REMOTE" in /data/local/tmp/*) ;; *) echo "ERROR: KATAGO_REMOTE_DIR must be under /data/local/tmp/" >&2; exit 2 ;; esac
# Restrict remote paths to safe Android shell path characters.
check_remote_path() {
    case "$1" in
        /*) ;;
        *) echo "ERROR: expected absolute Android path: $1" >&2; exit 2 ;;
    esac
    if [[ "$1" =~ [^a-zA-Z0-9_./+-] ]]; then
        echo "ERROR: unsupported characters in Android path: $1" >&2; exit 2
    fi
}
for path in "$REMOTE" "$REMOTE_BIN" "$REMOTE_MODEL" "$REMOTE_CFG"; do
    if [ -n "$path" ]; then check_remote_path "$path"; fi
done
# Prefer the already working GoEngine-java RK3588 GPU benchmark installation.
if [ -z "$BIN" ] && [ -z "$REMOTE_BIN" ]; then
    for candidate in "/data/local/tmp/katago_bench/katago_gpu" \
                     "/data/local/tmp/katago_bench/libkatago_exec_opencl.so" \
                     "$REMOTE/katago" "/data/local/tmp/katago/katago" \
                     "/data/local/tmp/katago_native/katago"; do
        if "${ADB_ARGS[@]}" shell "test -f $candidate" >/dev/null 2>&1; then
            REMOTE_BIN="$candidate"; break
        fi
    done
fi
if [ -z "$MODEL" ] && [ -z "$REMOTE_MODEL" ]; then
    for candidate in "/data/local/tmp/katago_bench/10b.bin" \
                     "/data/local/tmp/katago_bench/10b.bin.gz" \
                     "$REMOTE/model.bin" "$REMOTE/model.bin.gz" \
                     "/data/local/tmp/katago/10b.bin" "/data/local/tmp/katago/model.bin.gz" \
                     "/data/local/tmp/katago_native/10b.bin" \
                     "/data/local/tmp/katago_native/model.bin.gz"; do
        if "${ADB_ARGS[@]}" shell "test -s $candidate" >/dev/null 2>&1; then
            REMOTE_MODEL="$candidate"; break
        fi
    done
fi
# Otherwise reuse the sibling project's Android-built KataGo executable and model.
if [ -z "$BIN" ] && [ -z "$REMOTE_BIN" ]; then
    for candidate in "$ROOT/../GoEngine-java/build/katago_android_arm64_openclportable/libkatago_exec_opencl.so" \
                     "$ROOT/../GoEngine-java/build/katago_android_arm64_opencl/libkatago_exec_opencl.so" \
                     "$ROOT/../GoEngine-java/katago" "$ROOT/../GoEngine-java/build/katago" \
                     "$ROOT/../KataGo/cpp/katago" "$ROOT/../KataGo/build/katago" \
                     "$ROOT/../katago/katago"; do
        if [ -f "$candidate" ]; then BIN="$candidate"; break; fi
    done
fi
if [ -z "$MODEL" ] && [ -z "$REMOTE_MODEL" ]; then
    for candidate in "$ROOT/../GoEngine-java/app/src/main/assets/engine/katago/10b.bin" \
                     "$ROOT/../GoEngine-java/app/src/main/assets/engine/katago/10b.bin.gz" \
                     "$ROOT/../GoEngine-java/models/10b.bin" "$ROOT/../GoEngine-java/models/10b.bin.gz" \
                     "$ROOT/../GoEngine-java/10b.bin" "$ROOT/../GoEngine-java/10b.bin.gz" \
                     "$ROOT/../KataGo/10b.bin" "$ROOT/../KataGo/10b.bin.gz" "$ROOT/../KataGo/models/10b.bin.gz"; do
        if [ -f "$candidate" ]; then MODEL="$candidate"; break; fi
    done
fi
if [ -z "$BIN" ] && [ -z "$REMOTE_BIN" ] && [ "${KATAGO_AUTOBUILD:-1}" = 1 ]; then
    echo "KATAGO_NATIVE_MISSING no existing OpenCL binary, building official benchmarknn"
    bash "$ROOT/tools/build_rk3588_katago_benchmarknn.sh"
    CACHE="${KATAGO_NATIVE_CACHE:-$HOME/.cache/minigocpp/katago_native_v1.18.0}"
    BIN="$CACHE/build/katago"
fi
if [ -z "$BIN" ] && [ -z "$REMOTE_BIN" ]; then
    echo "ERROR: KataGo Android native executable not found." >&2
    echo "Set KATAGO_BIN=/path/to/Android/arm64/katago or KATAGO_REMOTE_BIN=/data/local/tmp/.../katago." >&2
    echo "It must implement 'katago benchmarknn' and use KataGo OpenCL backend, not a JNI .so." >&2
    exit 1
fi
if [ -z "$MODEL" ] && [ -z "$REMOTE_MODEL" ]; then
    echo "ERROR: Native KataGo 10b .bin/.bin.gz model missing (ONNX / RKNN cannot be loaded by native KataGo)." >&2
    echo "Set KATAGO_MODEL=/path/to/g170-b10c128-...bin.gz or KATAGO_REMOTE_MODEL=/data/local/tmp/.../model.bin.gz." >&2
    exit 1
fi
for path in "$BIN" "$MODEL" "$CFG"; do
    [ -z "$path" ] || [ -s "$path" ] || { echo "ERROR: Host file missing/empty: $path" >&2; exit 1; }
done
"${ADB_ARGS[@]}" get-state >/dev/null || { echo "ERROR: RK3588 ADB offline" >&2; exit 1; }
"${ADB_ARGS[@]}" shell "mkdir -p $REMOTE $REMOTE/.katago"
if [ -z "$REMOTE_BIN" ]; then
    REMOTE_BIN="$REMOTE/katago"
    "${ADB_ARGS[@]}" push "$BIN" "$REMOTE_BIN" >/dev/null
fi
if [ -z "$REMOTE_MODEL" ]; then
    REMOTE_MODEL="$REMOTE/model.bin"
    "${ADB_ARGS[@]}" push "$MODEL" "$REMOTE_MODEL" >/dev/null
fi
if [ -z "$REMOTE_CFG" ]; then
    REMOTE_CFG="$REMOTE/benchmarknn.cfg"
    if [ -n "$CFG" ]; then
        "${ADB_ARGS[@]}" push "$CFG" "$REMOTE_CFG" >/dev/null
    else
        tmp="$(mktemp)"
        trap 'rm -f "$tmp"' EXIT
        cat > "$tmp" <<EOF
numNNServerThreadsPerModel = 1
nnMaxBatchSize = 1
openclDeviceToUse = 0
openclUseFP16 = auto
openclTunerFile = $REMOTE/opencltuning.txt
EOF
        "${ADB_ARGS[@]}" push "$tmp" "$REMOTE_CFG" >/dev/null
    fi
fi
# Copy C++ runtime when the host NDK is available; existing remote installs can
# also be used with KATAGO_REMOTE_BIN and KATAGO_REMOTE_MODEL.
NDK_DIR="${NDK:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}}"
if [ -z "$NDK_DIR" ] && [ -d "$HOME/Android/Sdk/ndk/27.2.12479018" ]; then
    NDK_DIR="$HOME/Android/Sdk/ndk/27.2.12479018"
fi
CPP_LIB="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
[ ! -s "$CPP_LIB" ] || "${ADB_ARGS[@]}" push "$CPP_LIB" "$REMOTE/libc++_shared.so" >/dev/null
"${ADB_ARGS[@]}" shell "chmod 755 $REMOTE_BIN" >/dev/null || true
LIBS="$REMOTE:$(dirname "$REMOTE_BIN"):/vendor/lib64/egl:/vendor/lib64:/system/vendor/lib64:/system/lib64"
echo "KATAGO_NATIVE_PRECHECK serial=$ADB_SERIAL path=$REMOTE_BIN model=$REMOTE_MODEL"
# The KataGo 'version' subcommand truthfully identifies compile-time backend.
version="$("${ADB_ARGS[@]}" shell "cd $REMOTE && HOME=$REMOTE TMPDIR=$REMOTE LD_LIBRARY_PATH=$LIBS $REMOTE_BIN version" 2>&1 | tr -d '\r')" || {
    echo "$version" >&2
    echo "ERROR: Native KataGo executable cannot run on Android. Check arm64 ABI and shared libraries." >&2
    exit 1
}
echo "$version"
if ! grep -qi 'Using OpenCL backend' <<< "$version"; then
    echo "ERROR: Not a KataGo OpenCL build; refusing to label CPU/CUDA/other backend as RK GPU." >&2
    exit 1
fi
# KataGo versions v1.14/v1.15 can run the native OpenCL MCTS "benchmark",
# but they do NOT implement the pure-NN "benchmarknn" subcommand. Check the
# command before running so a zero-exit unknown-command message is never parsed
# as a successful GPU run.
benchmark_help="$("${ADB_ARGS[@]}" shell "cd $REMOTE && HOME=$REMOTE TMPDIR=$REMOTE LD_LIBRARY_PATH=$LIBS $REMOTE_BIN benchmarknn --help" 2>&1 | tr -d '\\r')" || true
if ! grep -Fq -- '--iterations' <<< "$benchmark_help" || ! grep -Fq -- '--batch-size' <<< "$benchmark_help"; then
    echo "KATAGO_NATIVE_OLD_BINARY benchmarknn unsupported: $REMOTE_BIN"
    if [ "${KATAGO_AUTOBUILD:-1}" != 1 ]; then
        echo "ERROR: Existing KataGo OpenCL executable lacks benchmarknn; KATAGO_AUTOBUILD=0 disables isolated official rebuild" >&2
        exit 1
    fi
    echo "KATAGO_NATIVE_UPGRADE building official KataGo v1.18.0 OpenCL command in independent MiniGoCpp cache"
    bash "$ROOT/tools/build_rk3588_katago_benchmarknn.sh"
    CACHE="${KATAGO_NATIVE_CACHE:-$HOME/.cache/minigocpp/katago_native_v1.18.0}"
    BIN="$CACHE/build/katago"
    [ -s "$BIN" ] || { echo "ERROR: Official KataGo benchmarknn build binary missing: $BIN" >&2; exit 1; }
    REMOTE_BIN="$REMOTE/katago_official_nn"
    # The OpenCL link input may export versioned OPENCL_1.0 symbols with
    # DT_SONAME=libGLES_mali.so. Stage the exact same file as runtime dependency,
    # not the unrelated /vendor/lib64/egl/libGLES_mali.so driver.
    LINK_FILE_MANIFEST="$CACHE/opencl_link_file.txt"
    SONAME_MANIFEST="$CACHE/opencl_link_soname.txt"
    [ -s "$LINK_FILE_MANIFEST" ] && [ -s "$SONAME_MANIFEST" ] || {
        echo "ERROR: Official KataGo OpenCL link manifest is missing" >&2; exit 1;
    }
    LINK_FILE="$(cat "$LINK_FILE_MANIFEST")"
    LINK_SONAME="$(cat "$SONAME_MANIFEST")"
    case "$LINK_SONAME" in
        libOpenCL.so|libGLES_mali.so) ;;
        *) echo "ERROR: Invalid OpenCL link SONAME $LINK_SONAME" >&2; exit 1 ;;
    esac
    [ -s "$LINK_FILE" ] || { echo "ERROR: Selected OpenCL runtime vanished: $LINK_FILE" >&2; exit 1; }
    echo "KATAGO_OPENCL_STAGING exact_linked_file=$LINK_FILE runtime=$REMOTE/$LINK_SONAME"
    "${ADB_ARGS[@]}" push "$LINK_FILE" "$REMOTE/$LINK_SONAME" >/dev/null
    "${ADB_ARGS[@]}" push "$BIN" "$REMOTE_BIN" >/dev/null
    "${ADB_ARGS[@]}" shell "chmod 755 $REMOTE_BIN"
    LIBS="$REMOTE:/vendor/lib64/egl:/vendor/lib64:/system/vendor/lib64:/system/lib64"
    version="$("${ADB_ARGS[@]}" shell "cd $REMOTE && HOME=$REMOTE TMPDIR=$REMOTE LD_LIBRARY_PATH=$LIBS $REMOTE_BIN version" 2>&1 | tr -d '\\r')" || {
        echo "$version" >&2; echo "ERROR: Official KataGo v1.18 OpenCL binary cannot execute on RK3588" >&2; exit 1;
    }
    echo "$version"
    grep -qi 'Using OpenCL backend' <<< "$version" || {
        echo "ERROR: The newly built KataGo is not an OpenCL backend" >&2; exit 1;
    }
    benchmark_help="$("${ADB_ARGS[@]}" shell "cd $REMOTE && HOME=$REMOTE TMPDIR=$REMOTE LD_LIBRARY_PATH=$LIBS $REMOTE_BIN benchmarknn --help" 2>&1 | tr -d '\\r')" || true
    if ! grep -Fq -- '--iterations' <<< "$benchmark_help" || ! grep -Fq -- '--batch-size' <<< "$benchmark_help"; then
        echo "$benchmark_help" >&2
        echo "ERROR: Official KataGo build does not support expected benchmarknn CLI" >&2
        exit 1
    fi
    echo "KATAGO_NATIVE_UPGRADE_READY executable=$REMOTE_BIN"
fi
# benchmarknn --json emits one result object; host report validates batch=1,
# 1 NN server thread, board=19 and backend OpenCL before accepting a number.
echo "KATAGO_NATIVE_START backend=opencl boardsize=19 batch=1 threads=1 runs=$RUNS warmup=$WARMUP"
"${ADB_ARGS[@]}" shell "cd $REMOTE && HOME=$REMOTE TMPDIR=$REMOTE LD_LIBRARY_PATH=$LIBS $REMOTE_BIN benchmarknn -model $REMOTE_MODEL -config $REMOTE_CFG --boardsize 19 --batch-size 1 --iterations $RUNS --warmup $WARMUP --json" 2>&1 | tr -d '\r'
