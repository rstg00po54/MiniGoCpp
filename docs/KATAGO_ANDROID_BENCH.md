# KataGo 10b — Android ADB unified benchmark

One host-side entry point on **Ubuntu/WSL x86**; use it with RK3588 Android 14 or vivo X300 Pro. The script builds the native ONNX Runtime benchmark using your existing Android NDK 27.2.12479018; **no Python or RKNN Toolkit Lite2 required on Android**.

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
