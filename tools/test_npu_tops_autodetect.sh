#!/usr/bin/env bash
# No device/NDK required: regression tests for NPU ADB backend dispatch.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/adb" <<'MOCKADB'
#!/usr/bin/env bash
set -Eeuo pipefail
mode="${MOCK_ADB_MODE:-rk}"
if [ "${1:-}" = devices ]; then
    printf 'List of devices attached\n'
    case "$mode" in
        rk) printf 'rk-usb\tdevice\n' ;;
        vivo) printf 'vivo-usb\tdevice\n' ;;
        both) printf 'rk-usb\tdevice\nvivo-usb\tdevice\n' ;;
        unknown) printf 'unknown-usb\tdevice\n' ;;
    esac
    exit 0
fi
if [ "${1:-}" = -s ]; then
    serial="$2"
    shift 2
    case "$mode:$serial" in
        rk:rk-usb|vivo:vivo-usb|both:rk-usb|both:vivo-usb|unknown:unknown-usb) ;;
        *) exit 1 ;;
    esac
    case "${1:-}" in
        get-state) printf 'device\n'; exit 0 ;;
        shell)
            case "$serial" in
                rk-usb) printf 'EmbedFire\nEmbedFire\nLubanCat-4IO\nRK3588\nrk3588\nrk30board\nrockchip,rk3588s-lubancat-4io\n' ;;
                vivo-usb) printf 'vivo\nvivo\nV2502A\nMT6993\nmt6993\nmtk\n' ;;
                unknown-usb) printf 'Other\nOther\nM1\nunknown\n' ;;
            esac
            exit 0 ;;
    esac
fi
exit 1
MOCKADB
chmod +x "$TMP/adb"

run_test() {
    local label="$1" mode="$2" serial="$3" expected="$4"
    shift 4
    local output
    if output="$(MOCK_ADB_MODE="$mode" ADB="$TMP/adb" ADB_SERIAL="$serial" TOPS_DRY_RUN=1 bash "$ROOT/MTK_NPU_Benchmark/run_tops.sh" "$@" 2>&1)"; then
        [[ "$expected" != ERROR ]] || { echo "FAIL $label: expected error"; exit 1; }
        [[ "$output" == *"$expected"* ]] || { echo "FAIL $label: $output"; exit 1; }
    else
        [[ "$expected" == ERROR ]] || { echo "FAIL $label: $output"; exit 1; }
    fi
    echo "PASS $label"
}
run_test rk_auto rk '' 'RK3588 -> RKNN' 
run_test vivo_auto vivo '' 'vivo -> Neuron Adapter: mtk-mdla'
run_test both_ambiguous both '' ERROR
run_test both_choose_rk both rk-usb 'RK3588 -> RKNN'
run_test both_choose_vivo both vivo-usb 'vivo -> Neuron Adapter: mtk-mdla'
run_test unknown unknown '' ERROR
run_test rk_reject_mtk rk '' ERROR all mtk-mdla
run_test rk_reject_shape rk '' ERROR large
run_test vivo_reject_rk vivo '' ERROR all rk
echo 'All TOPS autodetection tests PASS'
