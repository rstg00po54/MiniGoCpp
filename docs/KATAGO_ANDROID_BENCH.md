# KataGo 10b — Android ADB unified benchmark

One host-side entry point on **Ubuntu/WSL x86**; use it with RK3588 Android 14 or vivo X300 Pro. The script builds the native ONNX Runtime benchmark using your existing Android NDK 27.2.12479018; **no Python or RKNN Toolkit Lite2 required on Android**.

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
