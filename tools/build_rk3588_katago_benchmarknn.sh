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
if [ -s "$OUT" ] && [ "${KATAGO_FORCE_REBUILD:-0}" != 1 ] && \
   readelf -W -d "$OUT" | grep -Fq '[libOpenCL.so]' && \
   ! readelf -W -V "$OUT" | grep -Eq 'Name: OPENCL_[0-9]'; then
    echo "KATAGO_OFFICIAL_BUILD_REUSED binary=$OUT version=$TAG linked=portable-libOpenCL.so"
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
# Portable OpenCL: cross-compile a LINK-ONLY import .so for Android ARM64.
# The stub has the standard SONAME libOpenCL.so with unversioned core OpenCL
# symbols; it must NEVER be installed on Android. At runtime the real vendor
# /vendor/lib64/libOpenCL.so is loaded, as with the proven openclportable build.
LINK_LIB="$CACHE/linkonly/libOpenCL.so"
CC="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang"
[ -x "$CC" ] || { echo "ERROR: Android NDK Clang not found: $CC" >&2; exit 1; }
if [ ! -s "$LINK_LIB" ] || [ "$ROOT/tools/create_opencl_android_link_stub.py" -nt "$LINK_LIB" ]; then
    python3 "$ROOT/tools/create_opencl_android_link_stub.py" \
        --header "$CACHE/opencl_headers/CL/cl.h" --cc "$CC" --out "$LINK_LIB"
fi
readelf -W -h "$LINK_LIB" | grep -Eq 'Machine:[[:space:]]*AArch64' || {
    echo "ERROR: OpenCL link-only library is not Android ARM64" >&2; exit 1;
}
readelf -W -d "$LINK_LIB" | grep -Fq '[libOpenCL.so]' || {
    echo "ERROR: Link-only OpenCL import SONAME is not libOpenCL.so" >&2; exit 1;
}
if readelf -W -V "$LINK_LIB" | grep -Eq 'Name: OPENCL_[0-9]'; then
    echo "ERROR: OpenCL link-only import library must have no vendor symbol-version requirements" >&2
    exit 1
fi
echo "KATAGO_OPENCL_SELECTED file=$LINK_LIB mode=PORTABLE_LINK_ONLY soname=libOpenCL.so deploy=NO"
printf '%s\n' "$LINK_LIB" > "$CACHE/opencl_link_file.txt"
printf '%s\n' "libOpenCL.so" > "$CACHE/opencl_link_soname.txt"
printf '%s\n' "LINK_ONLY" > "$CACHE/opencl_link_mode.txt"
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
readelf -W -d "$OUT" | grep -Fq '[libOpenCL.so]' || {
    echo "ERROR: KataGo executable does not have portable DT_NEEDED=libOpenCL.so" >&2; exit 1;
}
if readelf -W -V "$OUT" | grep -Eq 'Name: OPENCL_[0-9]'; then
    echo "ERROR: KataGo executable still depends on vendor OPENCL_ symbol version" >&2
    exit 1
fi
echo "KATAGO_OFFICIAL_BUILD_READY binary=$OUT tag=$TAG OpenCL_SONAME=libOpenCL.so vendor_lock=NO"
