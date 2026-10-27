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

## ADB 原生 FP16 TOPS 测试（无需 APK）

```bash
cd /mnt/hgfs/D/code/go/MiniGoCpp
git pull
bash MTK_NPU_Benchmark/run_tops.sh
```

脚本自动使用已有的 Android NDK r27 编译 C++ ARM64 命令行程序，ADB 推入
`/data/local/tmp/minigocpp_tops/` 并执行；默认只绑定 `mtk-mdla`，
不会在编译 NPU 图失败时退回 CPU。终端和
`MTK_NPU_Benchmark/results/YYYYMMDD_HHMMSS_all_mtk-mdla.log` 都会显示结果。

```bash
# 选择一种形状；默认 all 依次运行 go19/small/medium/large
WARMUP=50 RUNS=200 bash MTK_NPU_Benchmark/run_tops.sh all mtk-mdla
RUNS=100 bash MTK_NPU_Benchmark/run_tops.sh heavy mtk-mdla

# 多设备时指定手机，例如无线 ADB
ADB_SERIAL=192.168.0.184:5555 bash MTK_NPU_Benchmark/run_tops.sh go19 mtk-mdla

# 如果想比较 DSP，显式选择（此时数字不是 MDLA NPU 结果）
bash MTK_NPU_Benchmark/run_tops.sh small mtk-dsp
```

也能绕开脚本、用 ADB 直接执行**已经编译并上传**的程序：

```bash
adb shell 'cd /data/local/tmp/minigocpp_tops && LD_LIBRARY_PATH=.:/system_ext/lib64:/vendor/lib64 ./neuron_tops_android all mtk-mdla 20 100'
```

测试集使用原生 **FP16**、3×3 same-padded 卷积：

| profile | 输入 NHWC | 输出通道 |
| --- | --- | --- |
| go19 | 1×19×19×128 | 128 |
| small | 1×32×32×128 | 128 |
| medium | 1×64×64×128 | 128 |
| large | 1×64×64×256 | 256 |
| heavy | 1×96×96×256 | 256（默认 all 不含，单独调用） |

每种形状先核对部分结果与 CPU 卷积参考，再计时已绑定输入/输出
的同步 `NeuronExecution_compute`。仅在设备报告 CONV_2D 支持、
使用 `NeuronCompilation_createForDevices` **限制至指定设备**、编译成功且
输出误差检查通过后，才报告 `TOPS_PASS`。失败显示
`TOPS_CASE_FAIL`，没有把 CPU 数字当成 NPU 数字。

结果中的 `TOPS_mean` 是基于平均推理耗时的 FP16 **有效 TOPS**，
`TOPS_fastest` 是该形状单次最低耗时推算值：

```text
nominal_ops = 2 × H × W × Cout × 3 × 3 × Cin
effective_TOPS = nominal_ops / (latency_ms × 10^9)
```

按普通密集 CONV 约定，计数包含边界 zero padding 位置，每个 MAC 计 2 次浮点运算。
**这不是芯片厂家标称峰值 TOPS，也不是 INT8 TOPS，更不是实际内部硬件计数器测量。**
同步调用、驱动和数据传输开销都计入耗时，实际结果受温控、时钟、模型形状、
编译器和设备驱动影响。即使显式绑定 `mtk-mdla`，仍无法从该 API
完全观察厂商内部调度方式。

目前仅验证了源代码写入仓库，**尚未在实体手机上运行这套新 TOPS 测试**；
必须以终端打印的 `VERIFY_PASS` 和 `TOPS_PASS` 为准。
