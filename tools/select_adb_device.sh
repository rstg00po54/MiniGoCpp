#!/usr/bin/env bash
# Print one ADB serial to stdout. All diagnostics go to stderr.
# Usage: ADB_SERIAL=optional_override bash tools/select_adb_device.sh vivo|rk|auto
set -euo pipefail

target="${1:-auto}"
case "$target" in vivo|rk|auto) ;; *) echo "ERROR: device selector must be vivo, rk or auto." >&2; exit 2 ;; esac
adb_bin="${ADB:-adb}"
command -v "$adb_bin" >/dev/null || { echo "ERROR: adb not found: $adb_bin" >&2; exit 1; }

describe() {
    local serial="$1" details
    details="$("$adb_bin" -s "$serial" shell '
      getprop ro.product.manufacturer
      getprop ro.product.brand
      getprop ro.product.model
      getprop ro.soc.model
      getprop ro.board.platform
      getprop ro.hardware
      getprop ro.product.board
      cat /proc/device-tree/compatible 2>/dev/null || true
    ' 2>/dev/null | tr -d '\r\000' || true)"
    printf '%s' "$details"
}

matches_target() {
    local info="${1,,}" kind="$2"
    case "$kind" in
        vivo) [[ "$info" == *vivo* ]] ;;
        rk)   [[ "$info" == *rk3588* ]] ;;
        auto) return 0 ;;
    esac
}

if [ -n "${ADB_SERIAL:-}" ]; then
    if ! "$adb_bin" -s "$ADB_SERIAL" get-state >/dev/null 2>&1; then
        echo "ERROR: ADB_SERIAL=$ADB_SERIAL is not online." >&2
        exit 1
    fi
    info="$(describe "$ADB_SERIAL")"
    if ! matches_target "$info" "$target"; then
        echo "ERROR: ADB_SERIAL=$ADB_SERIAL does not match requested device '$target'." >&2
        echo "Device properties: ${info//$'\n'/ | }" >&2
        exit 1
    fi
    echo "Selected ADB $target: $ADB_SERIAL (${info//$'\n'/ | })" >&2
    printf '%s\n' "$ADB_SERIAL"
    exit 0
fi

raw="$("$adb_bin" devices)"
mapfile -t serials < <(printf '%s\n' "$raw" | awk 'NR>1 && $2=="device" {print $1}')
if [ "${#serials[@]}" -eq 0 ]; then
    echo "ERROR: No authorized online devices. Run adb devices -l." >&2
    exit 1
fi

candidates=()
details=()
for serial in "${serials[@]}"; do
    info="$(describe "$serial")"
    if matches_target "$info" "$target"; then
        candidates+=("$serial")
        details+=("$info")
    fi
done
if [ "${#candidates[@]}" -eq 0 ]; then
    echo "ERROR: No connected device matches '$target'." >&2
    for serial in "${serials[@]}"; do
        info="$(describe "$serial")"
        echo "  $serial : ${info//$'\n'/ | }" >&2
    done
    echo "Connect the requested device or explicitly set ADB_SERIAL." >&2
    exit 1
fi
if [ "$target" = auto ] && [ "${#candidates[@]}" -gt 1 ]; then
    echo "ERROR: Multiple ADB transports found. Specify 'vivo' or 'rk', or set ADB_SERIAL." >&2
    for i in "${!candidates[@]}"; do
        echo "  ${candidates[i]} : ${details[i]//$'\n'/ | }" >&2
    done
    exit 1
fi

# When one physical device is exposed over USB and TCP/IP, prefer USB.
# If several of the same category still match, this is a deterministic preference;
# ADB_SERIAL can always override it.
chosen=0
if [ "${#candidates[@]}" -gt 1 ]; then
    for i in "${!candidates[@]}"; do
        if [[ "${candidates[i]}" != *:* ]]; then chosen="$i"; break; fi
    done
    echo "WARNING: ${#candidates[@]} devices match '$target'; choosing ${candidates[chosen]} (USB preferred). Set ADB_SERIAL to override." >&2
fi
echo "Selected ADB $target: ${candidates[chosen]} (${details[chosen]//$'\n'/ | })" >&2
printf '%s\n' "${candidates[chosen]}"
