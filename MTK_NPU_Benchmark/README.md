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

## ADB 自动识别 NPU 后端（无需 APK）

```bash
cd /mnt/hgfs/D/code/go/MiniGoCpp
git pull
bash MTK_NPU_Benchmark/run_tops.sh
```

脚本通过 `tools/select_adb_device.sh auto` 选择设备，再读取 SoC/厂商属性：

- **vivo / MediaTek**：保留原生 Neuron FP16 合成卷积测速，固定 `mtk-mdla`（可选 `mtk-dsp`）；
  使用 NDK r27 编译后，ADB 推入 `/data/local/tmp/minigocpp_tops/` 执行，显示 **有效 FP16 TOPS**。
- **RK3588 / RK3588S**：自动转到已有的 `tools/bench_rk3588_npu.sh`，运行真实
  KataGo 10b（19×19）的 RKNN 全网络测试，打印不同 NPU 核的延迟和 **NN evals/s**。
  这条路径不报告 TOPS，因为模型未包含经过严格核实的运算量计数，不能把 NN/s 冒充 TOPS。
- **未识别芯片或同时连接多种设备**：显式失败，要求指定 `ADB_SERIAL`；
  不会盲猜设备或回退 CPU。两个通道都复用原来 benchmark 的依赖/缓存。

默认命令 `bash MTK_NPU_Benchmark/run_tops.sh` 两种设备均可使用；
无参数时根据 ADB 连接自动切换。报告保存在 `MTK_NPU_Benchmark/results/`。

```bash
# 选择一种形状；默认 all 依次运行 go19/small/medium/large
WARMUP=50 RUNS=200 bash MTK_NPU_Benchmark/run_tops.sh all mtk-mdla
RUNS=100 bash MTK_NPU_Benchmark/run_tops.sh heavy mtk-mdla

# RK3588: 使用 RKNN 全网络 benchmark，显示 NN evals/s（不显示 TOPS）
CORE_MODES="0 1 2 all" RUNS=100 bash MTK_NPU_Benchmark/run_tops.sh

# 多台 ADB 设备同时连接时，手动指定 RK3588 序列号
ADB_SERIAL=8719e18a71a2a66c bash MTK_NPU_Benchmark/run_tops.sh

# 多设备时指定手机，例如无线 ADB
ADB_SERIAL=192.168.0.184:5555 bash MTK_NPU_Benchmark/run_tops.sh go19 mtk-mdla

# 如果想比较 DSP，显式选择（此时数字不是 MDLA NPU 结果）
bash MTK_NPU_Benchmark/run_tops.sh small mtk-dsp
```

也能绕开脚本、用 ADB 直接执行**已经编译并上传**的程序：

```bash
adb shell 'cd /data/local/tmp/minigocpp_tops && LD_LIBRARY_PATH=.:/system_ext/lib64 ./neuron_tops_android all mtk-mdla 20 100'
```

以下仅描述 **vivo Neuron** 后端：测试集使用原生 **FP16**、3×3 same-padded 卷积：

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

既有的 FP16、heavy INT8 已在实体 vivo X300 Pro 上运行通过；新增的大负载和持续测试仍需要设备验证。

## vivo 适配库加载失败的诊断

`TOPS_BENCH_FAIL: MediaTek Neuron Adapter dlopen failed` 表示图还没有提交到 NPU。
`run_tops.sh` 现在使用与此前跑通的 `tools/bench_mediatek_katago_full.sh`
相同的 `LD_LIBRARY_PATH=/data/local/tmp/minigocpp_tops:/system_ext/lib64`。
错误文本中会逐项列出 `dlopen` 对各候选库的真实诊断（如
`library not found`、依赖缺失、链接器命名空间拒绝）。失败时不要把成绩记作 0 TOPS。
如果更新后仍出现失败，保留完整 `TOPS_BENCH_FAIL` 错误供定位；无需重新下载模型。

## vivo NPU 真正的 signed INT8 吞吐量测试（新增）

原来的 FP16 测试不变。要测试有符号 INT8，直接在命令尾部加 `int8`：

```bash
git pull
WARMUP=30 RUNS=200 bash MTK_NPU_Benchmark/run_tops.sh all mtk-mdla int8
WARMUP=100 RUNS=500 bash MTK_NPU_Benchmark/run_tops.sh heavy mtk-mdla int8
```

想比较已经枚举到的另一个 MediaTek Neuron 设备 `mtk-dsp`：

```bash
WARMUP=20 RUNS=100 bash MTK_NPU_Benchmark/run_tops.sh go19 mtk-dsp int8
```

参数含义：`run_tops.sh [profile] [auto|mtk-mdla|mtk-dsp] [fp16|int8]`。
默认仍是 `all auto fp16`；连接 vivo 时，`auto` 自动绑定 `mtk-mdla`。
连接 RK3588 时保持原有 RKNN 整网测速，只报告 NN/s，**不提供 INT8 TOPS**；
如果用户明确在 RK3588 上选择 `int8`，会报错，避免伪造可比较结果。

INT8 模型采用 **有符号 8 位非对称量化输入/输出**（zero point=0），
优先使用 **逐输出通道对称量化的有符号 INT8 卷积权重**；若该格式被驱动
拒绝，再尝试逐张量 signed INT8 量化权重。偏置为 INT32，scale 等于
输入与权重的 scale 乘积。每个 case 都使用真实变化的非零输入和权重，
首先运行一次并对边界、中央以及多组输出通道共 75 个量化输出做 CPU
整数卷积参考验证，容许至多 2 个输出量化刻度（LSB）的差异。
只在设备专属编译、运行和数值检验成功后才打印
`INT8_TOPS_PASS`，否则报告 `INT8_PER_CHANNEL_FAIL` /
`TOPS_CASE_FAIL`。没有 CPU 计算回退；CPU 仅用于校验。

`TOPS_mean` / `TOPS_fastest` 仍然是根据密集卷积标称运算数
（每个 MAC 计 2 OPS）除以同步推理耗时的 **有效 INT8 TOPS**，
不是厂商标称芯片峰值，也不是 APU/NPU 内部硬件计数器；
驱动调度和编译器优化不可由该值直接还原。

**关于 100 TOPS：** 当前手机通过 Neuron Adapter API 的设备枚举只观测到
`mtk-dsp` 和 `mtk-mdla`，没有单独枚举到 CIM。不能因此断言 CIM
不存在，也不能认定 `mtk-mdla` 测速涵盖整套 NPU 990 的计算单元。
将对 `mtk-dsp` 的测试单独标注，**不要把 DSP 和 MDLA 的各自峰值相加**，
也不要拿 FP16 与 INT8 的 TOPS 直接对照。
INT8 的 go19 和 heavy 已由实体 vivo 测通，但新增超大形状尚未验证，可能遇到编译器或内存限制。


## 扩展 INT8 大负载和持续 30 秒测试

支持的新大尺寸（全部采用 NHWC、3×3、stride=1、padding=1，权重优先 INT8 per-channel）：

| profile | H×W×Cin→Cout | 标称 GOP/次 | 用途 |
| --- | --- | ---: | --- |
| x128 | 128×128×256→256 | 19.327 | 从之前 heavy 逐级放大 |
| x128c512 | 128×128×512→512 | 77.309 | 高通道量 |
| x256 | 256×256×256→256 | 77.309 | 高空间分辨率 |
| x256c512 | 256×256×512→512 | 309.238 | 最大压力，可能因 NPU 内存/编译限制失败 |

为避免意外高负载，默认 `all` **仍然只跑原来的四个小测试**；`scale`
明确选择上表的四组，并逐个完成、释放前一个 Neuron 上下文再开始下一组。
建议先用少量次数测尺寸兼容性，然后再选成功的档位做持续压力。

```bash
git pull
WARMUP=20 RUNS=100 bash MTK_NPU_Benchmark/run_tops.sh scale mtk-mdla int8
WARMUP=100 RUNS=300 bash MTK_NPU_Benchmark/run_tops.sh x128 mtk-mdla int8
WARMUP=100 RUNS=300 bash MTK_NPU_Benchmark/run_tops.sh x128c512 mtk-mdla int8
WARMUP=100 RUNS=300 bash MTK_NPU_Benchmark/run_tops.sh x256 mtk-mdla int8
WARMUP=100 RUNS=100 bash MTK_NPU_Benchmark/run_tops.sh x256c512 mtk-mdla int8
```

**单个负载持续跑 30 秒：**

```bash
WARMUP=100 BENCH_SECONDS=30 bash MTK_NPU_Benchmark/run_tops.sh heavy mtk-mdla int8
WARMUP=100 BENCH_SECONDS=30 bash MTK_NPU_Benchmark/run_tops.sh x128 mtk-mdla int8
```

当 `BENCH_SECONDS` 大于 0 时，程序在预热/模型数值验证后按真实单调时钟计时，
运行到指定秒数，忽略 `RUNS` 的固定次数（可设为 1～10000，默认 100）。
最高允许 180 秒，防止误操作无限压力；`NEURON_TIMEOUT` 可调节 ADB 外部超时。
不支持在 RK3588 的旧 RKNN NN/s 模式里使用 `BENCH_SECONDS`。

日志中的 `SUSTAIN_WINDOW` 每约 5 秒输出窗口内总推理次数、
平均延迟、有效 TOPS，以及可选的温度 `temp`；只读取设备公开可访问并
明确标为 NPU/APU/MDLA 的 sysfs thermal zone。
如果 Android 不允许访问对应传感器，显示 `temp=unavailable`，**不代表芯片没有温度，也不使用 CPU 温度冒充 NPU**。
`SUSTAIN_SUMMARY` 包含整体墙钟吞吐量、
第一个和最后一个窗口的 TOPS 及 `last_over_first` 比例，便于估算降频。
不同尺寸是否能执行仍取决于 Neuron 编译器与设备内存，不保证全部可用。

这些成绩依旧属于**指定 mtk-mdla 的 INT8 卷积工作负载有效吞吐量**，
不可直接推断 NPU 990 的理论 TOPS，也没有证明覆盖 CIM 等其他硬件单元。

## RK3588 / RK3588S 原生 RKNN INT8 TOPS 测试

不同于旧的 KataGo 10b RKNN **NN/s** 测试，以下是专门生成的、真正执行 INT8 量化的
3×3 合成卷积。为使 `rknn_run` 可以在 Android 上执行，必须先在
x86_64 Ubuntu 使用 **RKNN-Toolkit2** 把 ONNX 校准量化、编译成 `.rknn`。
这是 Rockchip 工具链的限制，不需要安装到 Android 手机/开发板上。

量化模型自动生成一次，缓存到 `~/.cache/minigocpp/rknn_android/tops_*.rknn`，
再次运行不重新编译模型或下载已有运行库。
Android 测速器是纯 **C++**（`tools/rk3588_tops_android.cpp`），通过 ADB 在
`/data/local/tmp/minigocpp_rk_tops` 执行，无需 APK。
仅模型离线转换由 `tools/build_rk3588_tops_model.py` 完成。

```bash
git pull
CORE_MODES="0 1 2 all" WARMUP=20 RUNS=100 bash MTK_NPU_Benchmark/run_tops.sh heavy auto int8
CORE_MODES="0" WARMUP=20 RUNS=100 bash MTK_NPU_Benchmark/run_tops.sh x128 auto int8
CORE_MODES="0" WARMUP=100 BENCH_SECONDS=30 bash MTK_NPU_Benchmark/run_tops.sh x128 auto int8
```

目前 RKNN INT8 可选档位（单个 model，对应显式选择）：

| profile | NHWC 等价尺寸 H×W×Cin→Cout | 标称 GOP/次 |
| --- | --- | ---: |
| rk64 | 64×64×128→128 | 1.208 |
| heavy | 96×96×256→256 | 10.872 |
| x128 | 128×128×256→256 | 19.327 |
| x256 | 256×256×256→256 | 77.309 |
| x128c512 | 128×128×512→512 | 77.309 |
| x256c512 | 256×256×512→512 | 309.238 |

建议第一次从 `rk64` 或 `heavy` 开始；最后两个可能消耗较多内存和转换时间。
`CORE_MODES` 可设置为单核 `0` / `1` / `2`，或单一模型跨核的 `all`。
这还不是三个独立模型绑定三个独立 NPU 核同时运行的总吞吐量。

生成器需要当前 Python 环境安装 `numpy`、`onnx`、`rknn-toolkit2`
（可复用已有的 RKNN 环境）。若当前环境不是 RKNN 环境，指定：

```bash
RKNN_PYTHON=/path/to/rknn-env/bin/python CORE_MODES="0" bash MTK_NPU_Benchmark/run_tops.sh rk64 auto int8
```

可以用已经量化好的外部 RKNN 模型跳过生成器，但其图的输入输出形状、
权重必须和上述测试代码一致（否则数值验证拒绝）：

```bash
RKNN_TOPS_MODEL=/path/to/generated_tops_heavy_int8.rknn CORE_MODES="0" bash MTK_NPU_Benchmark/run_tops.sh heavy auto int8
```

测试程序会查询 SDK 输出张量的 INT8/UINT8 量化属性，拒绝非量化输入/输出，
并在正式计时前比较 **75 个 CPU 卷积参考输出**、检查输出非恒定且对输入变化敏感。
只有成功才打印 `RK_INT8_VERIFY_PASS` 和 `RK_INT8_TOPS_PASS`。
结果基于每次卷积的密集理论 MAC×2 除以总推理耗时，
**包含 host float32→量化输入转换和 RKNN 输出传输开销**，并非硬件内部计数。
和 vivo Neuron 的 FP16/INT8 测试的输入输出传输方式不完全相同，
因此不能将两者的有效 TOPS 直接当作同条件的芯片峰值比较。

瑞芯微标称 RK3588 **6 TOPS @ INT8（三核总量）**，这不等于单核极限，
也不意味着一个模型的 `all` 模式必然会比单核快。
RKNN 新 INT8 图及 Android C++ 路径**目前尚未在实体 RK3588 上验证**，
第一次请提供模型生成日志和 `RK_INT8_*` 输出；若编译失败不会报告假 TOPS。
