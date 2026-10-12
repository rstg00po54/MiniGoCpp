# RK3588S CPU / GPU / NPU: Complete KataGo 10b Benchmark

Run on the x86 Ubuntu host with **LubanCat-4IO Android 14 connected through ADB**.
This is an **entire-network** KataGo 10b 19×19 neural-network forward benchmark,
not the small-convolution test or MCTS Go search.

## One command

```bash
cd /mnt/hgfs/D/code/go/MiniGoCpp
git pull
RUNS=30 WARMUP=5 THREADS=4 bash tools/bench_rk3588_compare.sh
```

Optional settings:

```bash
RUNS=100 WARMUP=20 THREADS=4 GPU_RUNS=30 GPU_WARMUP=5 bash tools/bench_rk3588_compare.sh
```

The script selects only RK3588 Android devices, using `ADB_SERIAL` to override selection if needed.
It caches the original ONNX 10b model, converts the **same** original network to an
FP16 `K10NW001` packet (same source as the successfully tested vivo Neuron
whole-network benchmark), extracts the packet's original FP16 19×19 spatial/global
inputs, and produces one matching FP32 NCHW input file with the original CPU reference.
The all-ones 19×19 mask is also included.

Three native ARM64 executables then run sequentially:

| Backend | Source | Precision | Device |
| --- | --- | --- | --- |
| CPU | ONNX Runtime original ONNX model | FP32 | RK3588 ARM CPU, 4 threads default |
| GPU | Original model converted to 126-op full network; direct OpenCL kernels | FP16 | Mali GPU selected using OpenCL `CL_DEVICE_TYPE_GPU` only |
| NPU | RKNN Toolkit2-converted original 10b | RKNN runtime conversion-specific | Core 0, `rknn_set_core_mask` |

**No GPU CPU fallback**: if the device has no OpenCL GPU/FP16 support or a
kernel cannot run, GPU is reported **UNAVAILABLE**, not replaced with CPU/NNAPI.
GPU uses a straightforward, unoptimized OpenCL implementation of the complete
network: it is a **correctness-oriented GPU baseline**, not a tuned KataGo
OpenCL engine. Comparing it to vendor-optimized NPU and optimized CPU does *not*
measure the GPU's theoretical maximum or the performance of KataGo's tuned
Winograd kernels.

### Output

The one-click script makes a timestamped directory in `results/rk3588_*/`:

- `cpu.log`, `gpu.log`, `npu.log` — original device output;
- `comparison.md` — ready-to-read table with latency, NN evals/s and speedup;
- `comparison.csv` — spreadsheet-ready;
- `comparison.json` — machine-readable values, statuses and caveats.

Only a backend which produces **all five output-head accuracy checks** and an
NN-only timing result is marked `PASS`. Any error, missing log, or unsupported
device will be reported as UNAVAILABLE, without inventing a speed ratio.
The accuracy check compares with the original full ONNX FP32 network outputs,
using a per-element tolerance `0.05 + 0.06 * abs(reference)` to allow FP16
rounding. This is a smoke-level numerical tolerance, not an exhaustive game
correctness proof.

A successful report resembles:

```text
| Backend | Status | Mean (ms) | P50 (ms) | P95 (ms) | NN evals/s | vs CPU |
| cpu     | PASS   | ...       | ...      | ...      | ...        | 1.00x  |
| gpu     | PASS   | ...       | ...      | ...      | ...        | ...    |
| npu     | PASS   | ...       | ...      | ...      | ...        | ...    |
```

Numbers are **not estimated or prepopulated**. There is no report of actual
RK3588 full-network GPU performance until it has run and verified on your device.

### Repeat processing on an existing run

```bash
python3 tools/summarize_rk3588_compare.py --logs results/rk3588_YYYYMMDD_HHMMSS
```

### Existing benchmark compatibility

`tools/bench_android.sh rk cpu` and `tools/bench_rk3588_npu.sh` continue
working with their original defaults. The one-command wrapper passes a shared
input file to them through `COMPARE_INPUT_FILE`. Neither the Java KataGo
application nor the user's other Android project is modified.

`RKNN_NPU_CORE_0_1_2` for **one model inference** was observed slower than
core 0 on the RK3588S; the three *separate* Context parallel throughput
measurement is a different test and must not be plotted as single-inference
NN evals/s without an aggregate wall-clock check.

## First RK3588S physical-device result (2026-10-12, 30 measured, 5 warmup)

The first original KataGo 10b benchmark on Android 14 LubanCat-4IO produced:

| Device | Mean | P50 | P95 | NN evals/s | Reference check |
| --- | ---: | ---: | ---: | ---: | --- |
| CPU, ONNX Runtime FP32, 4 threads | 23.229 ms | 23.238 ms | 23.451 ms | 43.05 | PASS, 5/5 outputs |
| GPU, Mali-G610 OpenCL FP16 (initial unoptimized kernel) | 404.555 ms | 405.209 ms | 408.646 ms | 2.47 | PASS, 5/5 outputs |
| NPU, RKNN Core 0, 2.3.2 (driver 0.9.8) | **INVALID** | — | — | **DO NOT REPORT** | FAILED, five heads |

The original published RKNN model gives wildly wrong outputs when fed the shared
FP32 NCHW input. Its OutputPolicyPass max absolute error was **1407.962241** and
OutputValue error **10552.061627**. The old standalone NPU timing of about 104
model executions/s does **not** demonstrate valid KataGo inference and must
not be used as a verified CPU/GPU/NPU comparison. Do not increase the tolerance
or use timing from that model until this failure is explained.

### Targeted input-layout diagnosis (same .rknn, same reference)

```bash
cd /mnt/hgfs/D/code/go/MiniGoCpp
git pull
RUNS=5 WARMUP=1 CORE_MODES=0 RKNN_INPUT_LAYOUT=nhwc \
  COMPARE_INPUT_FILE="$HOME/.cache/minigocpp/androidbench/katago10b_shared_input_v1.bin" \
  bash tools/bench_rk3588_npu.sh
```

The native runner explicitly transposes the spatial input into NHWC, sets
`RKNN_TENSOR_NHWC`, prints the first values and min/max of each output, and
still compares all five outputs against the original ONNX reference. No
successful numerical match, no valid NPU score. The default `nchw` path is
unchanged. If both fail, investigate the RKNN converted model/ops rather than
just changing runtime tensor layout.

The initial GPU kernel used OHWI filter order with noncoalesced output-channel
access. A subsequent experimental optimization transposes GPU filter data to
HWIO for coalesced GPU memory access, and **needs a new RK3588 run with numeric
verification** before its speed can be claimed. An OpenCL throughput result
from this deliberately untuned implementation is not the Mali hardware peak.

