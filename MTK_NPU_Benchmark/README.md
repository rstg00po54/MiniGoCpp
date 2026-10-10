# MTK NPU Benchmark

独立的 Android ARM64 / MediaTek Neuron FP16 NPU 基准测试，放在 MiniGoCpp 仓库中，但不依赖 KataGo 模型、围棋 App 或 JNI。

### 运行

从 MiniGoCpp 仓库根目录执行：

```bash
bash MTK_NPU_Benchmark/launch.sh all mtk-mdla
```

也可以单独选择 tiny、stem、residual 三种合成卷积形状：

```bash
RUNS=1000 WARMUP=100 bash MTK_NPU_Benchmark/launch.sh residual mtk-mdla
```

首次运行，`launch.sh` 自动将该目录下的 `source.tar.xz` 解压为原版 `neuron_bench.cpp` 和 `run.sh`，随后通过 Android NDK 编译，ADB 上传到手机执行。结果记录在 `MTK_NPU_Benchmark/results/`。这些新生成的 build/results 文件只在本机，不会修改 KataGo 工程。

如果同时连着 USB 和 Wi-Fi ADB，可指定 `ADB_SERIAL=192.168.0.184:5555`；如需指定 NDK，可设置 `NDK=/home/cc/Android/Sdk/ndk/27.2.12479018`。

**注意：**这是合成卷积调用测试，`calls_per_sec` 不能当作 KataGo 整网 NN evals/s。完整 10b 网络的 NPU 测试仍使用仓库已有的 `tools/bench_mediatek_katago_full.sh`。

归档 `source.tar.xz` 的内容与先前提供的独立 ZIP 源文件相同，包含原始 `README.md`、`run.sh` 和 `neuron_bench.cpp`。
