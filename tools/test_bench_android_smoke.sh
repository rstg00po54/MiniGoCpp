#!/usr/bin/env bash
# Deterministic ADB regression with one fake RK3588 and one fake vivo device.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/adb" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
serial=""
if [ "${1:-}" = "-s" ]; then serial="$2"; shift 2; fi
case "${1:-}" in
  devices)
      printf 'List of devices attached\n8719e18a71a2a66c\tdevice\n192.168.0.184:5555\tdevice\n\n'
      ;;
  get-state) echo device ;;
  get-serialno) echo "${serial:-unknown}" ;;
  shell)
      if [ "${2:-}" = getprop ]; then
          case "${3:-}" in
              ro.product.cpu.abi) echo arm64-v8a ;;
              ro.product.manufacturer) [ "$serial" != 8719e18a71a2a66c ] && echo vivo || echo Rockchip ;;
              ro.product.brand) [ "$serial" != 8719e18a71a2a66c ] && echo vivo || echo rockchip ;;
              ro.product.model) [ "$serial" != 8719e18a71a2a66c ] && echo V2502A || echo LubanCat-4IO ;;
              ro.soc.model) [ "$serial" != 8719e18a71a2a66c ] && echo MT6993 || echo RK3588 ;;
              *) echo "" ;;
          esac
      elif [ "$serial" = 8719e18a71a2a66c ]; then
          printf 'Rockchip\nrockchip\nLubanCat-4IO\nRK3588\nrk3588\nrk30board\nrk30sdk\nrockchip,rk3588s-lubancat-4io\n'
      else
          printf 'vivo\nvivo\nV2502A\nMT6993\nmt6993\nmt6993\n\n'
      fi
      ;;
  *) echo "mock adb unexpected args: $*" >&2; exit 1 ;;
esac
MOCK
chmod +x "$TMP/adb"

select_device() { ADB="$TMP/adb" bash "$ROOT/tools/select_adb_device.sh" "$1"; }

vivo="$(select_device vivo)"
rk="$(select_device rk)"
[ "$vivo" = 192.168.0.184:5555 ]
[ "$rk" = 8719e18a71a2a66c ]
if ADB="$TMP/adb" ADB_SERIAL=8719e18a71a2a66c bash "$ROOT/tools/select_adb_device.sh" vivo >"$TMP/mismatch" 2>&1; then
    echo "FAIL: wrong device override must not be accepted" >&2
    exit 1
fi
if ADB="$TMP/adb" bash "$ROOT/tools/select_adb_device.sh" auto >"$TMP/auto" 2>&1; then
    echo "FAIL: multi-device auto must ask for a target" >&2
    exit 1
fi

vivo_probe="$(ADB="$TMP/adb" BENCH_LOG="$TMP/probe.log" bash "$ROOT/tools/bench_android.sh" vivo probe)"
grep -q 'ADB: 192.168.0.184:5555 manufacturer=vivo' <<< "$vivo_probe"
grep -q 'MediaTek NeuroPilot/Neuron device probe' <<< "$vivo_probe"
grep -q '===== DONE =====' <<< "$vivo_probe"

vivo_rknn="$(ADB="$TMP/adb" BENCH_LOG="$TMP/skip.log" bash "$ROOT/tools/bench_android.sh" vivo rknn)"
grep -q 'SKIP RKNN: This is not an RK3588 Android target.' <<< "$vivo_rknn"

if ADB="$TMP/adb" BENCH_LOG="$TMP/npu.log" bash "$ROOT/tools/bench_android.sh" vivo npu >"$TMP/npu.out" 2>&1; then
    echo "FAIL: vivo npu must not claim a valid result without a MediaTek backend" >&2
    exit 1
fi
grep -q 'NPU_NOT_BENCHMARKED:' "$TMP/npu.out"
grep -q 'ADB: 192.168.0.184:5555 manufacturer=vivo' "$TMP/npu.out"

printf 'PASS: vivo=%s rk=%s; ADB selection, probe and no-fake-NPU checks passed.\n' "$vivo" "$rk"
