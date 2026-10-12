# RK3588S: official KataGo OpenCL GPU vs CPU vs NPU

## Fix for the October 12 native GPU benchmarknn failure

The third RK3588 19×19 NN-only CPU/NPU run yielded CPU **43.28 NN/s**
(23.103 ms), validated RKNN Core 0 **108.42 NN/s** (9.223 ms,
correct NHWC input), and **GPU unavailable**. The old
`/data/local/tmp/katago_bench/katago_gpu` executable was correctly
detected as an OpenCL build, but it came from the vendored
`GoEngine-java/native/KataGo` source on `rk3588-engine`,
which implements the MCTS `benchmark` command **but not**
`benchmarknn`. The old successful result of 181.39 visits/s is
therefore a search benchmark, not an NN-only benchmark.

The RK3588 GPU runner now tests `benchmarknn --help` for required
`--iterations` and `--batch-size` options. If the existing OpenCL
binary does not support them, the runner **automatically builds official
KataGo v1.18.0** with OpenCL in the isolated
`~/.cache/minigocpp/katago_native_v1.18.0/` directory, reusing the
NDK and RK3588 OpenCL driver previously used for KataGo GPU builds.
It installs the resulting executable as
`/data/local/tmp/minigocpp_katago_native/katago_official_nn`.
The original `katago_gpu`, bundled `10b.bin`, `GoEngine-java`
and the APK are not overwritten.

The first build requires GitHub source download, Android NDK, CMake,
Ninja, OpenCL headers, and a valid Android ARM64 OpenCL link library
(from the earlier working GPU build or the RK3588 EGL driver).
Set `KATAGO_BUILD_JOBS=2` to limit build parallelism on a low-memory VM
(default 2). The source/binary are cached so later runs reuse them.

```bash
cd /mnt/hgfs/D/code/go/MiniGoCpp
git pull
RUNS=30 WARMUP=5 THREADS=4 bash tools/bench_rk3588_compare.sh
```

To focus only on the previously missing GPU result, without repeating
CPU/NPU:

```bash
GPU_RUNS=30 GPU_WARMUP=5 bash tools/bench_rk3588_gpu_katago.sh
```

If the build is unavailable, `gpu.log` retains the full error rather
than silently falling back to the handwriting OpenCL baseline.
The native GPU run should be treated as verified only after the
official JSON batch=1 result appears, and as a *native-input indicative*
comparison until matching model lineage and exact inputs are verified.

---

## Current one-command benchmark (native GPU, 2026-10-12)

The primary `tools/bench_rk3588_compare.sh` now runs **the official KataGo
`benchmarknn` OpenCL backend** for GPU, never the custom experimental
OpenCL convolution implementation. It keeps ONNX Runtime FP32 CPU and the
numerically validated RKNN NPU (NHWC layout).

**Prerequisite:** an Android ARM64 KataGo command-line executable compiled
with OpenCL, implementing `katago version` and `katago benchmarknn`, plus
the corresponding **original 10b native KataGo `.bin.gz` weights**.
The script does not confuse `.onnx` or `.rknn` models with native KataGo weights.

Specify host paths:

```bash
cd /mnt/hgfs/D/code/go/MiniGoCpp
git pull
KATAGO_BIN=/path/to/android/arm64/katago \
KATAGO_MODEL=/path/to/g170-b10c128-s1141046784-d204142634.bin.gz \
RUNS=30 WARMUP=5 THREADS=4 \
bash tools/bench_rk3588_compare.sh
```

If KataGo is already installed in a directly executable Android shell location:

```bash
KATAGO_REMOTE_BIN=/data/local/tmp/katago/katago \
KATAGO_REMOTE_MODEL=/data/local/tmp/katago/model.bin.gz \
RUNS=30 WARMUP=5 THREADS=4 bash tools/bench_rk3588_compare.sh
```

The launcher checks known conventional local `katago` / `10b.bin.gz`
paths when none are supplied. A missing or unsupported native executable
is reported clearly as an **unavailable GPU benchmark**, never silently
replaced by the old slow OpenCL implementation. Use `KATAGO_CONFIG` to
provide a custom config, or `KATAGO_REMOTE_CONFIG` for an on-device config.
Without either it creates a minimal single-GPU config with `nnMaxBatchSize=1`
and a writable OpenCL tuning cache in `/data/local/tmp`.

The GPU command uses `benchmarknn --boardsize 19 --batch-size 1 --iterations
N --warmup N --json`. The parser requires version output explicitly saying
`Using OpenCL backend`, revision ending in `-opencl`, one NN server thread,
batch size 1, 19×19 board and valid positive wall-clock throughput. The result
appears in `comparison.md`, `comparison.csv`, and `comparison.json`.

**Important workload distinction:** Native KataGo `benchmarknn` is NN-only
and does not include MCTS, but its test positions, preprocessing and backend
execution path are not identical to the original single synthetic input
validated against the ONNX CPU five-head reference. Therefore GPU is labeled
`PASS_NATIVE`, its wall-clock NN evals/s is recorded, its per-thread median
latency is kept separate from CPU/NPU mean latency, and any GPU/CPU ratio is
marked **indicative**, not a fully like-for-like speedup. CPU/NPU results retain
5/5 numerical reference checks. Confirm the native model identity in the JSON
report before making model-to-model comparisons.

The custom experimental GPU tester is preserved for debugging only:
`bash tools/bench_rk3588_gpu_full.sh`, or
`bash tools/bench_rk3588_gpu_ab.sh`. Its earlier 6.97 evals/s
is **not** an official KataGo GPU benchmark.

---

### Reuse the existing GoEngine-java RK3588 GPU build

The related project already ran KataGo OpenCL in the **MCTS search**
benchmark `GoEngine-java/tools/benchmark_katago_android.sh` on
`8719e18a71a2a66c`. The prior 19×19, 200-visits, five-position
results were **CPU 72.79 visits/s** versus **GPU 181.39 visits/s**
(with search settings differing, notably CPU 6 threads and GPU 64
threads; recorded GPU 181.81 nnEvals/s and mean batch 26.26).
The separate portable OpenCL binary benchmark recorded 39.66 visits/s
under other search settings. These search results are **not** Batch=1
NN-only results.

`tools/bench_rk3588_gpu_katago.sh` now automatically tries the
existing Android ARM64 OpenCL build at
`../GoEngine-java/build/katago_android_arm64_openclportable/libkatago_exec_opencl.so`
and the preinstalled Android executable
`/data/local/tmp/katago_bench/katago_gpu`.
It looks for the native `10b.bin` alongside the earlier Android
benchmark and in the local GoEngine-java assets/model directories.

The script checks the binary's `version` output for a real OpenCL
build, and only counts results if its `benchmarknn --json`
command exists. If that earlier binary lacks `benchmarknn`, the
script reports a clear GPU-unavailable result instead of relabeling
its `visits/s` as pure network throughput.

Run without recompiling KataGo:

```bash
cd /mnt/hgfs/D/code/go/MiniGoCpp
git pull
RUNS=30 WARMUP=5 THREADS=4 bash tools/bench_rk3588_compare.sh
```

---

## Historical experimental OpenCL baseline (superseded)

The following sections document the earlier handwritten OpenCL benchmark and
its accuracy checks. They are kept for diagnostic reference, but **do not
describe the current default GPU path**.


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


## Verified RK3588S rerun — GPU layout optimized, NPU input fixed (2026-10-12)

**The NPU accuracy issue was an incorrect runtime input tensor layout, not proof
of defective RKNN model conversion.** The RKNN model declares NHWC; the host
formerly set `RKNN_TENSOR_NCHW` for nonzero spatial inputs. Explicitly
transposing into NHWC and setting `RKNN_TENSOR_NHWC` solved all five output
mismatches against the original ONNX CPU reference.

| Backend | Mean latency | NN evals/s | Accuracy | Measured runs |
| --- | ---: | ---: | --- | ---: |
| CPU (4 threads, FP32) | 23.229 ms | 43.05 | PASS, 5/5 output heads | 30 |
| GPU (Mali-G610 FP16, generic OpenCL, optimized HWIO layout) | 143.503 ms | 6.97 | PASS, 5/5 | 10 |
| NPU (RKNN Core 0, NHWC input) | 5.845 ms | 171.10 | PASS, 5/5 | 5 |

The NPU is about **3.97× as fast as CPU** and **24.55× as fast as this
unoptimized GPU implementation** in NN evaluations/second. GPU HWIO reordering
improved the OpenCL baseline **2.82×** over its previous 404.555 ms per
evaluation / 2.47 evals/s. The GPU numbers are **not** a well-tuned Mali-G610
limit. The three measurements used different run counts (30/10/5), so repeat
with matched counts and warmups for a stable comparison.

NPU per-head maximum absolute error was: PolicyPass 0.000100, Policy
0.002833, Value 0.001165, ScoreValue 0.001204, and Ownership 0.000290;
all heads reported 0 incorrect elements with the chosen FP16 comparison tolerance.
The earlier NCHW five-head failure is preserved above as diagnostic history.

As of this result the comparison and direct RKNN benchmark **default to NHWC**.
For matched run counts:

```bash
git pull
RUNS=100 WARMUP=20 THREADS=4 bash tools/bench_rk3588_compare.sh
```

The wrapper still retains the `RKNN_INPUT_LAYOUT` override for controlled
debugging (e.g. `RKNN_INPUT_LAYOUT=nchw`), and **will not report a PASS
when its five output heads fail reference validation**.


## RK3588 Mali-G610 GPU optimization: tiled implicit GEMM (2026-10-12)

The existing **correct** direct convolution computes a single output element
per OpenCL work item and repeatedly loads the same weights and activations.
It was measured at 143.503 ms/NN eval (6.97 NN evals/s) after HWIO weight
reordering. To reduce redundant global memory accesses, the **new default**
GPU kernel uses an **8 x 16 x 32 blocked implicit matrix multiplication**:

- 8 spatial points x 16 output channels computed cooperatively per workgroup;
- 32 reduction elements at a time in local memory (`__local half`);
- FP32 fused multiply-add accumulators and FP16 inputs/weights/outputs;
- padding and 1x1/3x3/5x5 convolutions use the same code path;
- original OHWI model weights are packed to HWIO once before GPU upload;
- the original `direct` kernel remains available as an A/B control.

This is an optimized **native OpenCL implementation of all 126 original network
ops**, **not** the KataGo project's heavily tuned Winograd/GEMM OpenCL backend.
The original five-head ONNX FP32 reference check is mandatory before timing.
No speedup has been claimed until the tiled kernel passes on the actual Mali
GPU.

Single run (default tiled):

```bash
cd /mnt/hgfs/D/code/go/MiniGoCpp
git pull
RUNS=10 WARMUP=2 bash tools/bench_rk3588_gpu_full.sh
```

**Better: compare original and tiled kernels on the same board automatically:**

```bash
GPU_RUNS=10 GPU_WARMUP=2 bash tools/bench_rk3588_gpu_ab.sh
```

The command reports `GPU_AB_PASS direct=... tiled=... speedup=...x`
only if both kernels pass all five output validation checks. Full logs are
saved under `results/gpu_ab_*/`. Both `GPU_CONV_KERNEL mode=direct` and
`GPU_CONV_KERNEL mode=tiled` are printed explicitly. For manual runs choose
`KATAGO_GPU_CONV=direct` or `KATAGO_GPU_CONV=tiled`.

To remeasure all three CPU/GPU/NPU with the new default OpenCL kernel:

```bash
RUNS=30 WARMUP=5 THREADS=4 bash tools/bench_rk3588_compare.sh
```

If the tiled Mali kernel fails to compile or produces an incorrect output, the
runner **fails without outputting a GPU speed number**. The `direct` fallback
can be tested separately without affecting CPU/NPU code.
