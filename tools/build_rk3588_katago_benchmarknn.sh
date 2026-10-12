#!/usr/bin/env bash
# Cross-compile OFFICIAL KataGo 1.18.0 OpenCL 'benchmarknn' into a separate cache.
# Never modifies GoEngine-java, its GPU binaries, the APK or any staged app model.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${KATAGO_BENCH_TAG:-v1.18.0}"
if [ "$TAG" != v1.18.0 ]; then
    echo "ERROR: Only the verified benchmarknn-capable KataGo v1.18.0 is supported" >&2
    exit 2
fi
CACHE="${KATAGO_NATIVE_CACHE:-$HOME/.cache/minigocpp/katago_native_v1.18.0}"
SOURCE="$CACHE/source"
BUILD="$CACHE/build"
OUT="$BUILD/katago"
JOBS="${KATAGO_BUILD_JOBS:-2}"
ADB="${ADB:-adb}"
ADB_SERIAL="${ADB_SERIAL:-$(ADB="$ADB" bash "$ROOT/tools/select_adb_device.sh" rk)}"
command -v cmake >/dev/null || { echo "ERROR: cmake missing" >&2; exit 1; }
command -v ninja >/dev/null || { echo "ERROR: ninja missing" >&2; exit 1; }
command -v readelf >/dev/null || { echo "ERROR: binutils/readelf missing" >&2; exit 1; }
case "$JOBS" in ''|*[!0-9]*) echo "ERROR: KATAGO_BUILD_JOBS must be a positive integer" >&2; exit 2 ;; esac
[ "$JOBS" -gt 0 ] || exit 2
if [ -s "$OUT" ] && [ "${KATAGO_FORCE_REBUILD:-0}" != 1 ]; then
    echo "KATAGO_OFFICIAL_BUILD_REUSED binary=$OUT version=$TAG"
    exit 0
fi
NDK="${NDK:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}}"
if [ -z "$NDK" ]; then
    for candidate in "${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/27.2.12479018" \
                     "${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}/ndk/27.2.12479018" \
                     "$HOME/Android/Sdk/ndk/27.2.12479018"; do
        if [ -f "$candidate/build/cmake/android.toolchain.cmake" ]; then NDK="$candidate"; break; fi
    done
fi
[ -f "$NDK/build/cmake/android.toolchain.cmake" ] || {
    echo "ERROR: Android NDK missing (set NDK=/home/cc/Android/Sdk/ndk/27.2.12479018)" >&2; exit 1;
}
mkdir -p "$CACHE"
if [ ! -f "$SOURCE/cpp/command/benchmarknn.cpp" ]; then
    if [ -d "$SOURCE" ]; then
        echo "ERROR: Incomplete KataGo source at $SOURCE; do not overwrite an unknown directory" >&2
        exit 1
    fi
    echo "KATAGO_OFFICIAL_DOWNLOAD tag=$TAG source=$SOURCE"
    git clone --depth 1 --branch "$TAG" https://github.com/lightvector/KataGo.git "$SOURCE"
fi
[ -f "$SOURCE/cpp/command/benchmarknn.cpp" ] || {
    echo "ERROR: Source tag does not have official benchmarknn command" >&2; exit 1;
}
# Use host CL headers only. Never accidentally add host libc include to NDK Clang.
HEADERS="${KATAGO_OPENCL_HEADERS:-}"
if [ -z "$HEADERS" ]; then
    for dir in "$HOME/.cache/goengine_katago/opencl_headers" \
               "$HOME/.cache/goengine_katago/OpenCL-Headers" /usr/include /usr/local/include; do
        if [ -f "$dir/CL/cl.h" ]; then HEADERS="$dir"; break; fi
    done
fi
[ -f "$HEADERS/CL/cl.h" ] || {
    echo "ERROR: OpenCL headers missing; install opencl-headers or set KATAGO_OPENCL_HEADERS" >&2; exit 1;
}
mkdir -p "$CACHE/opencl_headers/CL"
cp -a "$HEADERS/CL/." "$CACHE/opencl_headers/CL/"
# Link only with ARM64 OpenCL libraries defining the actual cl* entrypoints.
# RK3588 vendor libOpenCL.so may be a versioned OPENCL_1.0 implementation with
# SONAME libGLES_mali.so. Its symbols ARE valid; the earlier strict unversioned
# filter incorrectly rejected it and chose /vendor/lib64/egl/libGLES_mali.so.
has_gpu_api() {
    local file="$1"
    [ -s "$file" ] || return 1
    readelf -W -h "$file" | grep -Eq 'Machine:[[:space:]]*AArch64' || return 1
    readelf -W --dyn-syms "$file" | awk '
        $7 != "UND" && $8 ~ /^clGetPlatformIDs(@.*)?$/ {p=1}
        $7 != "UND" && $8 ~ /^clCreateContext(@.*)?$/ {c=1}
        END {exit !(p&&c)}' || return 1
}
describe_gpu_api() {
    local file="$1"
    echo "KATAGO_OPENCL_PROBE file=$file"
    readelf -W -h "$file" | grep -E 'Machine:|Class:' || true
    readelf -W -d "$file" | grep -E '\\(SONAME\\)' || true
    readelf -W --dyn-syms "$file" | grep -E 'cl(GetPlatformIDs|CreateContext)(@|[[:space:]]|$)' | head -8 || true
    readelf -W -V "$file" | grep -E 'Name: OPENCL_[0-9]' | head -5 || true
}
LINK_LIB="${KATAGO_OPENCL_ARM64_LIB:-}"
if [ -n "$LINK_LIB" ]; then
    if ! has_gpu_api "$LINK_LIB"; then
        describe_gpu_api "$LINK_LIB"
        echo "ERROR: Explicit KATAGO_OPENCL_ARM64_LIB is not an ARM64 OpenCL library with both required symbols" >&2
        exit 1
    fi
fi
if [ -z "$LINK_LIB" ]; then
    for path in "$HOME/.cache/goengine_katago/opencl_arm64/libOpenCL.so" \
                "$HOME/.cache/goengine_katago/opencl_arm64_portable/libOpenCL.so" \
                "$HOME/.cache/goengine_katago/opencl_arm64_portable/libGLES_mali.so" \
                "$CACHE/libOpenCL.so" "$CACHE/libGLES_mali.so"; do
        if has_gpu_api "$path"; then LINK_LIB="$path"; break; fi
    done
fi
if [ -z "$LINK_LIB" ]; then
    # Probe every vendor path first, without assuming EGL Mali exports OpenCL.
    # The first vendor libOpenCL.so was already proven usable by GoEngine-java's
    # existing versioned OpenCL benchmark (it stages this as libGLES_mali.so).
    for remote in /vendor/lib64/libOpenCL.so /vendor/lib64/libGLES_mali.so \
                  /vendor/lib64/egl/libGLES_mali.so /system/vendor/lib64/libOpenCL.so; do
        local_file="$CACHE/vendor_$(echo "$remote" | tr '/' '_')"
        if [ ! -s "$local_file" ]; then
            echo "KATAGO_OPENCL_PULL path=$remote"
            "$ADB" -s "$ADB_SERIAL" pull "$remote" "$local_file" >/dev/null 2>&1 || { rm -f "$local_file"; continue; }
        fi
        describe_gpu_api "$local_file"
        if has_gpu_api "$local_file"; then LINK_LIB="$local_file"; break; fi
    done
fi
if [ -z "$LINK_LIB" ]; then
    echo "ERROR: No RK3588 ARM64 OpenCL library exports both clGetPlatformIDs and clCreateContext." >&2
    echo "Inspect KATAGO_OPENCL_PROBE diagnostics; do not fall back to a non-OpenCL EGL library." >&2
    exit 1
fi
LINK_LIB="$(realpath "$LINK_LIB")"
SONAME="$(readelf -W -d "$LINK_LIB" | sed -nE 's/.*\\(SONAME\\).*\\[([^]]+)\\].*/\\1/p' | head -1)"
case "$SONAME" in
    libOpenCL.so|libGLES_mali.so) ;;
    *) echo "ERROR: Unexpected OpenCL SONAME '${SONAME:-empty}' in $LINK_LIB" >&2; exit 1 ;;
esac
if readelf -W -V "$LINK_LIB" | grep -Eq 'Name: OPENCL_[0-9]'; then
    echo "KATAGO_OPENCL_LINK_VERSION=OPENCL_1.0_OR_NEWER"
else
    echo "KATAGO_OPENCL_LINK_VERSION=UNVERSIONED"
fi
echo "KATAGO_OPENCL_SELECTED file=$LINK_LIB soname=$SONAME"
# Kept as a manifest so the Android GPU launcher stages the SAME vendor
# library under its NEEDED SONAME; a different EGL Mali build may be ABI-incompatible.
printf '%s\\n' "$LINK_LIB" > "$CACHE/opencl_link_file.txt"
printf '%s\\n' "$SONAME" > "$CACHE/opencl_link_soname.txt"
echo "KATAGO_OFFICIAL_CONFIG tag=$TAG ndk=$NDK opencl=$LINK_LIB headers=$HEADERS jobs=$JOBS"
cmake -Wno-deprecated -S "$SOURCE/cpp" -B "$BUILD" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release -DUSE_BACKEND=OPENCL \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_EXE_LINKER_FLAGS=-pie \
    -DBUILD_DISTRIBUTED=OFF -DNO_GIT_REVISION=ON -DUSE_AVX2=OFF -DUSE_TCMALLOC=OFF \
    -DOpenCL_INCLUDE_DIR="$CACHE/opencl_headers" -DOpenCL_LIBRARY="$LINK_LIB" \
    -DCMAKE_CXX_FLAGS="-DLITTLE_ENDIAN=1234 -DBIG_ENDIAN=4321 -DBYTE_ORDER=1234"
cmake --build "$BUILD" --parallel "$JOBS"
[ -s "$OUT" ] || { echo "ERROR: Official KataGo Android binary missing after build" >&2; exit 1; }
readelf -h "$OUT" | grep -Eq 'Machine:[[:space:]]*AArch64' || {
    echo "ERROR: KataGo binary is not Android ARM64" >&2; exit 1;
}
echo "KATAGO_OFFICIAL_BUILD_READY binary=$OUT tag=$TAG"
