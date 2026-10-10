#!/usr/bin/env bash
# Regression: some phones have no /proc/device-tree/compatible; ADB launcher must not silently exit.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/adb" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  get-state) echo device ;;
  get-serialno) echo 10AFB21HP5002ZK ;;
  shell)
    case "${2:-}" in
      "getprop ro.product.cpu.abi") echo arm64-v8a ;;
      "getprop ro.product.manufacturer") echo vivo ;;
      *) echo 'Dimensity 9500'; echo 'mt6993' ;;
    esac ;;
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
echo "PASS: Vivo ADB hardware detection does not silently exit."
