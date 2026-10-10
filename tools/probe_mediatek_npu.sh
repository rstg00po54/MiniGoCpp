#!/usr/bin/env bash
# Non-invasive MediaTek Neuron/NeuroPilot availability probe via ADB.
# Does not require root or install any APK; library presence does not prove access.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADB_BIN="${ADB:-adb}"
ADB_SERIAL="$(ADB="$ADB_BIN" bash "$ROOT/tools/select_adb_device.sh" vivo)"
ARGS=("$ADB_BIN" -s "$ADB_SERIAL")
command -v "$ADB_BIN" >/dev/null || { echo "ERROR: adb not found." >&2; exit 1; }
"${ARGS[@]}" get-state >/dev/null || { echo "ERROR: No ADB device." >&2; exit 1; }
soc="$("${ARGS[@]}" shell 'getprop ro.soc.model; getprop ro.board.platform' | tr -d '\r')"
echo "MediaTek NeuroPilot/Neuron device probe"
echo "Detected SoC: $soc"
echo "Checking known Neuron Adapter runtime names (file listings only; no root):"
matches="$("${ARGS[@]}" shell '
  for dir in /vendor/lib64 /system/lib64 /system_ext/lib64 /product/lib64 /odm/lib64; do
    if [ -d "$dir" ]; then
      ls "$dir"/libneuron*.so "$dir"/libapu*.so 2>/dev/null || true
    fi
  done
' | tr -d '\r' | sed '/^$/d' | sort -u)"
if [ -n "$matches" ]; then
    printf '%s\n' "$matches"
    echo "Neuron/APU-related library names found. This does NOT prove third-party app or adb-shell access."
else
    echo "No Neuron/APU libraries visible in standard system/vendor lib64 directories."
    echo "Libraries may be in other locations or hidden by Android namespace restrictions."
fi
echo "Standard NNAPI exposing only nnapi-reference is not MediaTek NPU evidence."
echo "Actual APU inference needs an accessible Neuron SDK/Adapter/Delegate path and a supported compiled model."
