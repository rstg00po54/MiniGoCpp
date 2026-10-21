#!/usr/bin/env bash
# MiniGoCpp 3x3 MCTS self-play with RK3588 Android NPU. Saves and pulls SGF files.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-$ROOT/models/my3.rknn}"
ADB_BIN="${ADB:-adb}"
CACHE="${MINIGO_RKNN_CACHE:-$HOME/.cache/minigocpp/rknn_android}"
REMOTE="/data/local/tmp/minigocpp_mini_npu"
RUNTIME_URL="https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknpu2/runtime/Android/librknn_api"
SIMS="${SIMS:-24}"
GAMES="${GAMES:-1}"
SEED="${SEED:-42}"
CORE="${CORE:-0}"
OUTPUT="${OUTPUT:-$ROOT/results/minigo_rknn_selfplay_$(date +%Y%m%d_%H%M%S)}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -s "$MODEL" ]] || die "model missing: $MODEL"
[[ "$SIMS" =~ ^[0-9]+$ && "$GAMES" =~ ^[0-9]+$ && "$SEED" =~ ^[0-9]+$ ]] || die "SIMS/GAMES/SEED must be integers"
(( SIMS >= 2 && SIMS <= 10000 && GAMES >= 1 && GAMES <= 100 )) || die "SIMS must be 2..10000 and GAMES 1..100"
case "$CORE" in 0|1|2|all) ;; *) die "CORE must be 0, 1, 2 or all" ;; esac
command -v "$ADB_BIN" >/dev/null || die "adb not installed"
SERIAL="$(ADB="$ADB_BIN" bash "$ROOT/tools/select_adb_device.sh" rk)"
ADB_CMD=("$ADB_BIN" -s "$SERIAL")
ABI="$("${ADB_CMD[@]}" shell getprop ro.product.cpu.abi | tr -d '\r')"
[[ "$ABI" == arm64-v8a ]] || die "RK3588 Android ARM64 required (found $ABI)"

NDK_DIR="${NDK:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}}"
if [[ -z "$NDK_DIR" ]]; then
    for candidate in "${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/27.2.12479018" \
                     "${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}/ndk/27.2.12479018" \
                     "$HOME/Android/Sdk/ndk/27.2.12479018"; do
        [[ ! -d "$candidate" ]] || { NDK_DIR="$candidate"; break; }
    done
fi
[[ -n "$NDK_DIR" ]] || die "Android NDK not found; set NDK"
CXX="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++"
[[ -x "$CXX" ]] || die "missing NDK C++ compiler: $CXX"

download() {
    local url="$1" to="$2"
    [[ -s "$to" ]] && return 0
    mkdir -p "$(dirname "$to")"
    echo "Downloading $(basename "$to")..."
    if command -v curl >/dev/null; then
        curl -fL --retry 3 --connect-timeout 15 -o "$to.tmp" "$url" || die "download failed: $url"
    elif command -v wget >/dev/null; then
        wget -q -O "$to.tmp" "$url" || die "download failed: $url"
    else
        die "curl or wget missing"
    fi
    mv "$to.tmp" "$to"
}
mkdir -p "$CACHE" "$OUTPUT"
download "$RUNTIME_URL/include/rknn_api.h" "$CACHE/rknn_api.h"
download "$RUNTIME_URL/arm64-v8a/librknnrt.so" "$CACHE/librknnrt.so"
EXE="$CACHE/minigo_rknn_selfplay_android"
needs_build=0
[[ -x "$EXE" ]] || needs_build=1
for src in "$ROOT/tools/minigo_rknn_selfplay_android.cpp" \
           "$ROOT/src/board.cpp" "$ROOT/src/board.h" \
           "$ROOT/src/search.cpp" "$ROOT/src/search.h" \
           "$ROOT/src/network.cpp" "$ROOT/src/network.h" "$CACHE/rknn_api.h"; do
    [[ "$src" -nt "$EXE" ]] && needs_build=1
done
if [[ "$needs_build" == 1 ]]; then
    echo "Building original Board + MCTS + RKNN Android runner..."
    "$CXX" -std=c++17 -O2 -fPIE -pie -I"$ROOT/src" -I"$CACHE" \
        "$ROOT/tools/minigo_rknn_selfplay_android.cpp" \
        "$ROOT/src/board.cpp" "$ROOT/src/search.cpp" "$ROOT/src/network.cpp" \
        -L"$CACHE" -lrknnrt -llog -ldl -Wl,-rpath,'$ORIGIN' -o "$EXE"
fi

echo "ADB=$SERIAL SIMS=$SIMS GAMES=$GAMES CORE=$CORE SEED=$SEED"
"${ADB_CMD[@]}" shell "mkdir -p $REMOTE/sgf && rm -f $REMOTE/sgf/game_*.sgf"
"${ADB_CMD[@]}" push "$MODEL" "$REMOTE/my3.rknn" >/dev/null
"${ADB_CMD[@]}" push "$EXE" "$REMOTE/minigo_rknn_selfplay_android" >/dev/null
"${ADB_CMD[@]}" push "$CACHE/librknnrt.so" "$REMOTE/librknnrt.so" >/dev/null
LIBCPP="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [[ -s "$LIBCPP" ]]; then
    "${ADB_CMD[@]}" push "$LIBCPP" "$REMOTE/libc++_shared.so" >/dev/null
fi
"${ADB_CMD[@]}" shell "chmod 755 $REMOTE/minigo_rknn_selfplay_android"

echo "===== MiniGoCpp RK3588 NPU MCTS self-play ====="
# tee preserves an easy-to-read log and a failure exit status (pipefail).
"${ADB_CMD[@]}" shell "cd $REMOTE && LD_LIBRARY_PATH=$REMOTE ./minigo_rknn_selfplay_android my3.rknn $CORE $SIMS $GAMES $SEED sgf" \
    | tr -d '\r' | tee "$OUTPUT/game.log"
echo "Pulling SGF into $OUTPUT..."
"${ADB_CMD[@]}" pull "$REMOTE/sgf/." "$OUTPUT/" >/dev/null
echo "RESULT_DIR=$OUTPUT"
echo "DONE: Full 3x3 MCTS+RKNN game(s); inspect game.log and game_*.sgf"
