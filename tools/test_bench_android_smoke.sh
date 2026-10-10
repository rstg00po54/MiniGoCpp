#!/usr/bin/env bash
# Regression: some phones have no /proc/device-tree/compatible; ADB launcher must not silently exit.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/adb" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = "-s" ]; then shift 2; fi
case "${1:-}" in
  get-state) echo device ;;
  get-serialno) echo 10AFB21HP5002ZK ;;
  shell)
    if [ "${2:-}" = "getprop" ]; then
        case "${3:-}" in
            ro.product.cpu.abi) echo arm64-v8a ;;
            ro.product.manufacturer) echo vivo ;;
            *) echo "" ;;
        esac
    else
        # Simulate a phone with no /proc/device-tree/compatible.
        echo 'Dimensity 9500'
        echo 'mt6993'
    fi ;;
  *) echo "mock adb unexpected arguments: $*" >&2; exit 1 ;;
esac
MOCK
chmod +x "$TMP/adb"
# Explicit rknn mode only needs device detection on a non-RK3588 phone; it must cleanly SKIP.
out="$(ADB="$TMP/adb" BENCH_LOG="$TMP/benchmark.log" bash "$ROOT/tools/bench_android.sh" rknn)"
printf '%s\n' "$out"
grep -q 'ADB: 10AFB21HP5002ZK manufacturer=vivo abi=arm64-v8a' <<< "$out"
grep -q 'SKIP RKNN: This is not an RK3588 Android target.' <<< "$out"
grep -q '===== DONE =====' <<< "$out"
# Probe must work using the same ADB selector without requiring model downloads.
probe="$(ADB="$TMP/adb" BENCH_LOG="$TMP/probe.log" bash "$ROOT/tools/bench_android.sh" probe)"
printf '%s\n' "$probe"
grep -q 'MediaTek NeuroPilot/Neuron device probe' <<< "$probe"
grep -q '===== DONE =====' <<< "$probe"
echo "PASS: Vivo detection and MediaTek Neuron ADB probing."
