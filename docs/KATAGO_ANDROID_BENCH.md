# KataGo 10b — Android ADB unified benchmark

One host-side entry point on **Ubuntu/WSL x86**; use it with RK3588 Android 14 or vivo X300 Pro. The script builds the native ONNX Runtime benchmark using your existing Android NDK 27.2.12479018; **no Python or RKNN Toolkit Lite2 required on Android**.

## Vivo X300 Pro 4-block result and genuine rconv5 global pooling

The real vivo X300 Pro (MT6993, Neuron SDK 9.2.0, `mtk-mdla` pinned) executed the original KataGo 10b first 2 and 4 residual blocks as one native FP16 Neuron graph on a 19x19 full-board input:

| Original 10b residual blocks | Ops | Device compile | Average | P50 | P95 | Max abs error vs ONNX CPU | Mean abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rconv1..rconv2 | 14 | 246.427 ms | 0.728 ms | 0.627 ms | 1.299 ms | 0.000664 | 0.000034 |
| rconv1..rconv4 | 28 | 333.155 ms | 0.917 ms | 0.838 ms | 1.578 ms | 0.000731 | 0.000055 |

Both matched **46,208/46,208** fp16 output values within the test's documented error threshold. These are real KataGo blocks but not full-model NN evals/s.

The next qualitatively different layer is `rconv5`'s global pooling: compute spatial mean/max of 32 channels, multiply mean by the actual `InputMask/scale` parameter, concatenate three 32-channel vectors into a 96-channel 1x1 tensor and apply **original KataGo `rconv5/w1r` 96-to-96 1x1 Conv**. The exporter uses actual upstream 10b ONNX operations and FP32 reference outputs and verifies that all-ones mask allows this reduced mean/max formulation.

```bash
cd /mnt/hgfs/D/code/go/MiniGoCpp
git pull
bash tools/bench_mediatek_katago_rconv5_gpool.sh mtk-mdla
```

This test prints `KATAGO_GPOOL_OP_SUPPORT` for MEAN, MUL, REDUCE_MAX, CONCAT, and 1x1 CONV; only if all are supported can the device-pinned compile and numerical validation run. **It is not the entire rconv5 block**, and if unsupported, that single operator's output helps decide whether the original KataGo model needs a hybrid execution path or a different pooling representation.

## Integrated first 1-4 KataGo residual blocks in ONE FP16 Neuron graph

After the vivo X300 Pro executed the actual KataGo rconv1 seven-operator residual graph successfully (mean 0.672 ms, 46,208/46,208 matched ONNX CPU, max absolute error 0.000443), the next step is to **connect consecutive normal blocks 1..4 in a single Neuron compilation**, instead of invoking them as standalone sessions.

KataGo 10b's first four rconv blocks are the same normal 128-channel, 3x3 two-convolution architecture, with 7 logical Neuron ops per block (two norm scale multiplications, two bias adds with fused ReLU, two real convolutions and one residual ADD). **rconv5 is not included**, since its graph has a global-pooling branch and different shapes. The exporter builds isolated Python 3.8-compatible ONNX CPU references with the original actual network constants, mask fixed to all ones, and checks the full chained output. Both test source and packet are versioned separately so the successful original single-block test is not touched.

Run:

```bash
cd /mnt/hgfs/D/code/go/MiniGoCpp
git pull
bash tools/bench_mediatek_katago_residual_stack.sh 2 mtk-mdla
bash tools/bench_mediatek_katago_residual_stack.sh 4 mtk-mdla
# Optionally compare to a single block, same new harness:
bash tools/bench_mediatek_katago_residual_stack.sh 1 mtk-mdla
```

Expect `KATAGO_STACK_OP_SUPPORT` lines for every logical operator, then `KATAGO_STACK_COMPILE`, `KATAGO_STACK_VERIFY`, and `KATAGO_STACK_PASS` **only** if the requested fixed-device graph compiles and the final 46,208 FP16 elements agree with the original ONNX CPU chain within the explicit numerical tolerance. The output is **residual stacks/s**, not complete KataGo NN evals/s. Since these are separately run device calls, their throughput must not be compared directly with RKNN's full-model NN evals/s.

## Integrated KataGo rconv1 FP16 graph on the same MediaTek MDLA device

The single-operation vivo tests now confirm:

- Real `rconv1/w1` on `mtk-mdla`: mean **0.788 ms**, 46,208/46,208 values matched CPU.
- Real `rconv1/w2`: mean **0.374 ms**, all outputs matched.
- Native 19x19x128 FP16 residual ADD: mean **0.880 ms**, all outputs matched.

**These latencies cannot be summed as a true fused rconv1 inference.** A new test creates **one complete `rconv1` device-pinned FP16 graph** containing seven operations: norm1 scale Mul, norm1 bias Add with fused ReLU, real 3x3 conv w1, norm2 scale Mul, norm2 bias Add with fused ReLU, real conv w2, and residual Add. It uses the true 10b weights and per-channel normalization values, and starts with the ONNX graph's actual `trunk/initbias` tensor. The source ONNX network also multiplies intermediate activations by the board mask: those are omitted only because the full 19x19 test explicitly fixes `InputMask` to all ones.

```bash
cd /mnt/hgfs/D/code/go/MiniGoCpp
git pull
bash tools/bench_mediatek_katago_rconv1.sh mtk-mdla
```

The exporter `tools/export_katago_rconv1_block.py` builds an isolated opset19/IR9 ONNX CPU reference graph comprising the **actual** 10b upstream operators and `rconv1`, computes the input and expected output, and packs them with original weights/scales/biases. The native program `tools/neuron_katago_rconv1_android.cpp` queries the MDLA support for each of the seven operations before attempting `NeuronCompilation_createForDevices`; it reports `KATAGO_RCONV1_OP_SUPPORT`, `KATAGO_RCONV1_COMPILE`, `KATAGO_RCONV1_VERIFY`, and `KATAGO_RCONV1_PASS` only if correct. If a broadcast Mul or fused Add fails, the exact unsupported step will be printed so graph decomposition can be reconsidered.

The isolated full block has not yet been confirmed on real vivo hardware. Even a successful `blocks/s` result is **not** a KataGo 10b NN eval/s measurement.

## KataGo first residual block: native MDLA feasibility (real 10b weights)

The actual 10b ONNX graph has `rconv1/w1` and `rconv1/w2`, each `128 -> 128` channels with `3x3` filters. Between and around them are per-channel normalization (Mul/Add), ReLU, board-mask multiplication and a residual Add. This is a **multi-operator** block, not just two independent convolution calls.

The next tests, which leave the successful first-Conv test unchanged, separately evaluate the real weights of `rconv1/w1` and `rconv1/w2` on the `mtk-mdla` device. Host Python builds isolated opset19/IR9 ONNX CPU *upstream* subgraphs and exports each convolution's true input and FP32 reference output. Thus the input to `w2` accounts for the real upstream normalization/activation and `w1`, rather than being arbitrary data. The existing native Neuron FP16 packet runner validates every output value before reporting timing.

```bash
cd /mnt/hgfs/D/code/go/MiniGoCpp
git pull
bash tools/bench_mediatek_katago_residual_conv.sh rconv1/w1 mtk-mdla
bash tools/bench_mediatek_katago_residual_conv.sh rconv1/w2 mtk-mdla
```

The residual block's skip connection also requires an ADD of two 19x19x128 tensors. A *separate* synthetic FP16 ADD test checks if MDLA can compile/execute this realistic shape. The older FP32 64-element ADD was not supported, which does not settle FP16 ADD support.

```bash
bash tools/bench_mediatek_residual_add.sh mtk-mdla
```

These tests intentionally **do not** report complete KataGo NN evals/s. The output for either Conv still uses the legacy `KATAGO_FIRST_CONV_*` lines from the shared packet runner but refers only to the selected Conv, identified by the host script.

## Real KataGo 10b first Conv: ONNX weights on MediaTek MDLA

This is the first experiment using **real KataGo 10b model weights**, not hand-made constant kernels. The exporter loads the original 19x19 `10b_19x19.onnx`, finds its first `Conv` with constant weights, uses the **original Conv node, tensor dimensions and FP32 weights** in a standalone opset-19/IR-9 reference graph to generate the CPU golden output (so Python 3.8 / ONNX Runtime 1.16.3 works), and exports a binary packet. The source ONNX model is left unchanged:

- Input features and weights are converted from ONNX NCHW / OIHW FP32 to Neuron NHWC / OHWI FP16.
- The isolated original Conv's CPU reference output is stored separately as FP32 to validate actual Neuron results; the test input is deterministic synthetic feature data.
- Bias, kernel size, padding, and tensor channel counts come from the real ONNX graph (the exporter fails rather than silently guessing incompatible Conv attributes).
- The phone builds **one** CONV_2D graph on the named `mtk-mdla` device using native `NeuronCompilation_createForDevices`.
- It checks per-device support, executes, verifies every element against ONNX output with explicit FP16 tolerance, and only reports timing if the output passes.

Run:

```bash
cd /mnt/hgfs/D/code/go/MiniGoCpp
git pull
bash tools/bench_mediatek_katago_first_conv.sh mtk-mdla

# Optional DSP comparison, *same real Conv*
bash tools/bench_mediatek_katago_first_conv.sh mtk-dsp
```

Requirements: existing Android NDK 27.2.12479018, adb; the host creates a cached Python virtualenv with `onnx==1.16.2`, `numpy==1.24.4`, and a Python-compatible ONNX Runtime (`1.16.3` for Python 3.8, `1.19.2` for Python 3.9+) if needed. The ~12 MB ONNX model is reused from the earlier ONNX benchmark. Shell variables `RUNS`, `WARMUP`, `KATAGO_ONNX_MODEL`, `PYTHON`, `NDK`, `NEURON_TIMEOUT` can be overridden.

**Do not compare** `KATAGO_FIRST_CONV_PASS conv_calls/s` to RK3588 `nnEvals/s`: the former is one genuine layer; the latter is a whole-network NN invocation. If the first Conv fails, record whether it is extraction, unsupported operation, compilation, or numerical mismatch. Passing the layer only establishes an important first step toward complete KataGo 10b execution.

## Native FP16 vs FP32 CONV_2D benchmark (additional test)

The previous 19x19 FP32 graph with relaxed FP16 computation remains intact. A separate 9x9 3x3 convolution test now uses **native FP16 tensors** (or strict FP32 tensors), and checks MDLA/DSP op support, execution, numerical output, and mean/P50/P95:

```bash
git pull
bash tools/bench_mediatek_conv.sh mtk-mdla fp16
bash tools/bench_mediatek_conv.sh mtk-mdla fp32
bash tools/bench_mediatek_conv.sh mtk-dsp fp16

# Existing 19x19 FP32-relaxed probe (unchanged default):
bash tools/bench_mediatek_conv.sh mtk-mdla
```

The 9x9 FP16/FP32 source is `tools/neuron_conv_fp16_android.cpp`, its ADB runner is `tools/bench_mediatek_conv_fp16.sh`. This graph is **not KataGo 10b**; passing MDLA compilation and inference would only establish this synthetic CONV_2D path.

## MT6993 MDLA: convolution test after FP32 ADD was rejected

On vivo V2502A, Native Neuron SDK **9.2.0** enumerated `mtk-dsp` and `mtk-mdla`. For one 64-element `NEURON_ADD` graph:

- `mtk-dsp`: operation supported, device-specific compilation and numerical validation passed, **1.010 ms** for one end-to-end call.
- `mtk-mdla`: **`NEURON_ADD` unsupported** for that specific graph and data type, so inference did not run. This does not imply every ADD or CONV graph is unsupported.

New direct CONV_2D experiment attempts a **device-constrained** one-layer NHWC `1×19×19×16 → 1×19×19×16` 3×3 convolution (16 output filters, constant weights/biases; approximately 0.83M multiply-accumulates), with `NeuronModel_relaxComputationFloat32toFloat16(true)`. It queries per-device support, compiles, executes, validates all 5,776 output values against analytically calculated values, and reports mean/P50/P95 only upon success.

```bash
cd /mnt/hgfs/D/code/go/MiniGoCpp
git pull
bash tools/bench_mediatek_conv.sh mtk-mdla
bash tools/bench_mediatek_conv.sh mtk-dsp
# Alternative using unified device selector:
bash tools/bench_android.sh vivo conv
# Longer test:
WARMUP=50 RUNS=500 bash tools/bench_mediatek_conv.sh mtk-mdla
```

A successful `NEURON_CONV_PASS` would be useful evidence that Neuron Adapter can compile and execute a convolution on the explicitly selected MDLA device, **not that KataGo 10b has been deployed or that the measured calls/s equal KataGo NN evals/s**. MediaTek's [MDLA guidelines](https://genio.mediatek.com/doc/iot-yocto/release/v25.0/sw/yocto/ml-guide/neuron-sdk/mt8188/supported-operations/mdla-guide/mdla_guidelines_3_0.html) discuss layout, precision and internal buffer constraints; they are generic and not guaranteed to describe the MT6993's exact hardware.

## vivo Neuron Adapter native API smoke (NPU integration stage)

The vivo MT6993 diagnostic reported the following OEM-installed libraries:

```text
/system_ext/lib64/libneuron_graph_delegate.mtk.so
/system_ext/lib64/libneuronusdk_adapter.mtk.so
/system_ext/lib64/libneuronusdk_adapter.9.mtk.so
/vendor/lib64/libneuron_runtime.9.so
```

Those files are stronger evidence of a MediaTek vendor acceleration runtime than the stock NNAPI CPU reference driver, **but file presence does not prove ordinary `adb shell` can load the libraries or that KataGo can run on the APU**.

An experimental native probe now calls the official MediaTek Neuron Adapter symbols with `dlopen()/dlsym()`, builds a tiny 64-element FP32 ADD graph and checks outputs:

```bash
cd /mnt/hgfs/D/code/go/MiniGoCpp
git pull
bash tools/bench_mediatek_neuron.sh
# or unified selector (returns nonzero until KataGo NPU is implemented):
bash tools/bench_android.sh vivo npu
```

The probe now also uses `Neuron_getDeviceCount` / `NeuronDevice_getName` to list actual **Neuron devices**; this is independent of `ANeuralNetworks_getDeviceCount` and stock NNAPI. To constrain compilation to one enumerated device name after the first run:

```bash
bash tools/bench_mediatek_neuron.sh
bash tools/bench_mediatek_neuron.sh EXACT_NEURON_DEVICE_NAME
# Or from the existing unified launcher:
NEURON_DEVICE_NAME=EXACT_NEURON_DEVICE_NAME bash tools/bench_android.sh vivo npu
```

When a specific device is supplied, the program checks `NeuronModel_getSupportedOperationsForDevices` (if exported) and uses `NeuronCompilation_createForDevices` instead of automatic fallback. This is a stronger check of target selection, but a 64-element ADD remains a tiny graph and does not establish KataGo performance.

The probe uses ADB device auto-selection to pick your vivo and the existing Android NDK 27.2.12479018 to compile `tools/neuron_smoke_android.cpp`. It requires no root, no model downloads, and no separate Neuron SDK headers (only documented public ABI definitions). It checks these stages: `dlopen`, Neuron symbols, version, `NeuronModel_create`, `NeuronCompilation_finish`, `NeuronExecution_compute`, and numerical equality. A failure means the stage reported is not yet accessible/supported; it is **not** a definite statement that the NPU cannot be used via an APK.

`NEURON_ADAPTER_SMOKE_PASS` means a tiny Neuron Adapter graph compiled and computed correctly, **not** that the MediaTek NPU (as opposed to a CPU fallback) ran, and **not** that a KataGo 10b graph ran. For real KataGo 10b NN eval/s, convert to compatible LiteRT/TFLite and use a validated MediaTek delegate/compiler with accelerator verification and matching numerical outputs.

MediaTek's [Neuron Adapter C API tutorial](https://developer.mediatek.com/ai/64254ccbf55b040d6989a99a.html) lists the C symbols used here. Google's [LiteRT MediaTek adapter](https://github.com/google-ai-edge/LiteRT/blob/main/litert/vendors/mediatek/neuron_adapter_api.cc) also recognizes the `libneuronusdk_adapter.mtk.so` and versioned adapter libraries.

## MediaTek MT6993: what the NNAPI result means

Observed on vivo X300 Pro / MT6993 (Android): only `NNAPI[0]: nnapi-reference type=CPU` is enumerated and the ONNX Runtime profile reports `NNAPI_nodes=0`. This means standard NNAPI did **not** use an accelerator in that test. The `nnapi` runtime result is CPU fallback, not a MediaTek NPU measurement. Profiling overhead differs from the unprofiled CPU mode, so the two latency figures are not a tightly controlled CPU-vs-CPU performance comparison.

To inspect whether MediaTek Neuron/NeuroPilot components have visible libraries on the phone:

```bash
git pull
bash tools/bench_android.sh probe
```

The probe uses `adb shell` to list public file names matching Neuron/APU libraries under standard Android partitions without root. Presence of libraries is **not proof that an ordinary app or ADB shell can load them**, or that the chip can run the given KataGo graph. The `auto` mode runs this probe on vivo/MediaTek devices before CPU/NNAPI/GPU tests.

MediaTek's dedicated NPU route involves **Neuron Adapter / Neuron Delegate / NeuroPilot**, generally with a supported TFLite model and appropriate vendor toolchain/runtime. ONNX Runtime's stock NNAPI execution provider is not the same as a MediaTek-specific acceleration execution provider. A working NPU test requires vendor-compatible conversion and an accessible runtime; until then this project does not claim native MT6993 NPU speed.

## 多台 ADB 设备：按设备类型选择

当 USB 和 Wi-Fi ADB 同时连接时，使用命令第一个参数指定设备类型，脚本通过 `getprop` 读取制造商与 SoC，不会依据设备列表顺序猜测：

```bash
bash tools/bench_android.sh vivo probe
bash tools/bench_android.sh vivo cpu nnapi
bash tools/bench_android.sh vivo npu
bash tools/bench_android.sh rk rknn
```

`vivo` 匹配 vivo 制造商，`rk` 匹配 RK3588/RK3588S SoC。只连接一台设备可省略类型，多台设备不指定类型会安全报错。如果同一种类型有多个 ADB 连接，优先选择 USB；若想强制选择 Wi-Fi，使用 `ADB_SERIAL=192.168.0.184:5555`。

**重点**：当前 `vivo npu` 检查 MediaTek Neuron/APU 组件，并尝试运行一个很小的 Neuron Adapter FP32 ADD 图，**还不是 KataGo 10b NPU 测速**。先前手机 ONNX Runtime NNAPI 测试结果 `NNAPI_nodes=0`，因此不能作为 MT6993 NPU 测速。需要完成 Neuron/LiteRT 模型转换和适配后才能取得真实 NPU 延迟。实际能跑 RKNN 的是 `rk rknn` 模式。

## Quick start

```bash
cd /mnt/hgfs/D/code/go/MiniGoCpp
git pull
adb devices -l
bash tools/bench_android.sh
```

If two devices are connected:

```bash
ADB_SERIAL=YOUR_VIVO_ADB_SERIAL bash tools/bench_android.sh cpu nnapi gpu
ADB_SERIAL=YOUR_RK3588_ADB_SERIAL bash tools/bench_android.sh cpu nnapi rknn gpu
```

For longer comparable runs:

```bash
RUNS=300 WARMUP=40 THREADS=4 bash tools/bench_android.sh cpu nnapi
RUNS=300 WARMUP=40 CORE_MODES="0 1 2 all" bash tools/bench_android.sh rknn
```

Mode breakdown:

| Mode | Android engines | Input | Output | Can claim actual NPU? |
|---|---|---|---|---|
| `cpu` | ONNX Runtime 1.23.2 CPU EP | KataGo 10b ONNX, synthetic 19x19 tensors | `nnEvals/s`, mean/P50/P95 | No |
| `nnapi` | ONNX Runtime NNAPI EP, CPU-disabled flag; ORT CPU fallback still allowed | Same ONNX model | Same + NNAPI device enumeration and EP profiling | **Not without proof of NNAPI NPU assignment** |
| `rknn` | Rockchip RKNPU2 (RK3588 Android only) | 10b RKNN, synthetic 19x19 tensors | Same NN-only metrics by core | Yes: RKNN runtime/driver, subject to model output validation |
| `gpu` | Existing KataGo OpenCL benchmark script | Actual KataGo engine search positions | **MCTS visits/s** | No, GPU search only |

`nnapi` will print `NNAPI_nodes=N CPU_nodes=M`. If N=0 then no NNAPI execution was observed. If N>0, ONNX Runtime has assigned work to NNAPI but the OS driver could use **GPU or NPU**. Android NNAPI is deprecated; the MediaTek NPU may require a separate NeuroPilot/LiteRT backend unavailable via generic NNAPI. **Do not interpret successful NNAPI session creation as proof of MediaTek NPU acceleration.**

NNAPI device listing uses Android NDK `ANeuralNetworks_getDeviceCount` and driver type. ONNX Runtime NNAPI execution provider may decline model operators or fall back to CPU; profile shows actual assignment. The tool intentionally does not fake a MediaTek NPU benchmark.

## Existing OpenCL GPU script

`gpu` uses the locally built and already working `GoEngine-java/tools/benchmark_katago_android.sh`. It is not tracked in this MiniGoCpp repository; if your local checkout differs, provide a path:

```bash
KATAGO_GPU_SCRIPT=/mnt/hgfs/D/code/go/GoEngine-java/tools/benchmark_katago_android.sh \
  KATAGO_BENCH_VISITS=200 KATAGO_BENCH_POSITIONS=5 KATAGO_BENCH_THREADS=2 \
  bash tools/bench_android.sh gpu
```

If missing, the launcher prints SKIP rather than claiming GPU speed. The GPU MCTS visit rate must not be compared to NN-only evaluation rate without measuring a matching workload.

## Provenance and limitations

- [Published 10b ONNX](https://github.com/rstg00po54/go/releases/tag/katago-10b-onnx-v1) — model `g170-b10c128-s1141046784-d204142634`, **opset 20**, 19×19 buffer, `batch=1` for tests. InputSpatial 22 channels, InputGlobal 19 channels, InputMask 1 channel, five outputs.
- [RK3588 RKNN model](https://github.com/rstg00po54/MiniGoCpp/releases/tag/katago-10b-rknn-v1) — generated with RKNN-Toolkit2 2.3.2. The earlier reported RK3588 Android NPU result is Core0 mean 9.669 ms, combined Core012 mean 22.720 ms, both with synthetic inputs. 
- Benchmarks use synthetic arrays of correct shapes (InputSpatial all-zero, InputGlobal all-zero, InputMask all-one). **These are not correct KataGo game-state features**. They measure latency, not playing strength. Verify outputs against ONNX CPU before using the RKNN model for Go MCTS. 
- The ONNX CPU/NNAPI results are from the *same* ONNX graph and inputs, so latency is comparable as a rough NN-only inference benchmark, but differing runtime overhead and precision still matter.
- Android environment: host NDK, `adb`, `curl`, `unzip`. ADB pushes model, native executable, ORT arm64 `.so`, and `libc++_shared.so`; no root needed if shell can execute from `/data/local/tmp`.
- ONNX Runtime downloaded from Maven Central: `com.microsoft.onnxruntime:onnxruntime-android:1.23.2`.

## Upstream references

- [ONNX Runtime NNAPI execution provider](https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html)
- [Android NNAPI NDK documentation](https://developer.android.com/ndk/reference/group/neural-networks)
- [ONNX Runtime Maven artifact](https://central.sonatype.com/artifact/com.microsoft.onnxruntime/onnxruntime-android/1.23.2)
