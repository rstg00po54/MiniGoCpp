# MiniGoCpp：从零手搓的微型围棋 AI（C++17）

这是一个教学 / 实验项目，不是 KataGo 的裁剪版。核心部分均自行实现：围棋规则、神经网络前向计算、手写反向传播、Adam 优化器、蒙特卡洛树搜索（MCTS/PUCT）、自我对弈训练、模型二进制保存和读取。**不依赖 KataGo、PyTorch、TensorFlow、Eigen、NDK。**

建议先玩 **3×3**，确认理解以后再用 **5×5**。本工程用 CPU 即可，适合 Ubuntu 虚拟机，不需要 NVIDIA 显卡。

## 0. 目录

```text
MiniGoCpp/
├── build.sh           编译并执行单元测试
├── CMakeLists.txt
├── src/
│   ├── board.h/.cpp   围棋规则、合法落子、提子、禁入点、禁全同
│   ├── network.h/.cpp 残差全连接网络、反向传播、Adam、模型保存
│   ├── search.h/.cpp  PUCT 蒙特卡洛树搜索
│   └── main.cpp       train/play/eval/inspect 命令行
├── tests/
│   └── test.cpp       单元测试
└── models/           训练时自动生成模型文件（不提交到仓库）
```

## 1. Ubuntu 编译

```bash
sudo apt install -y build-essential cmake git
git clone https://github.com/rstg00po54/MiniGoCpp.git
cd MiniGoCpp
bash build.sh
```

仓库不附带预训练模型，请先按第 2 节从零训练，生成 `models/my3.bin` 后再执行 `play` / `eval` / `inspect`。

编译完成：`./build/minigo`。`build.sh` 自动运行单元测试。**不建议在 `/mnt/hgfs/` 下编译**，可以把源码放在虚拟机的 `~/MiniGoCpp` 中。

## 2. 从零训练 3×3（不加载任何预训练模型）

```bash
./build/minigo train --size 3 --hidden 16 --games 100 --sims 24 --epochs 2 --out models/my3.bin
```

这个 3×3 网络共有 **955 个可学习参数**。训练每 10 盘更新一次网络并写入 `models/my3.bin`，中断后可以从最近保存的模型继续。

继续训练：

```bash
./build/minigo train --model models/my3.bin --games 100 --sims 24 --epochs 2 --out models/my3.bin
```

参数说明：`--games` 对局数；`--sims` 每步 MCTS 模拟次数；`--epochs` 每批数据训练轮数；`--hidden` 隐藏层宽度；`--lr` 学习率（默认 0.001）。

## 3. 自己和 AI 下棋

```bash
./build/minigo play --model models/my3.bin --sims 80
```

初始棋盘：

```text
   A B C
 1 . . .
 2 . . .
 3 . . .
```

- 输入 `B2` 落在中央；`pass` 停一手；`quit` 退出
- 黑棋用 `X`，白棋用 `O`
- 人默认执黑；`--human -1` 表示人执白，AI 先走

## 4. 查看网络参数与初始落子概率

```bash
./build/minigo inspect --model models/my3.bin
```

命令会显示当前模型的输入维度、隐藏单元、权重数、空棋盘胜负预测及各落子原始概率。这个概率是**神经网络直接计算的**，不等同于 MCTS 搜索后的胜率。

## 5. 与随机落子程序对弈测试

```bash
./build/minigo eval --model models/my3.bin --games 100 --sims 40
```

AI 黑白棋各下约一半，与随机合法落子的程序对弈。**胜率只是简易冒烟测试，不能代表真正棋力，也不能证明网络学到了正确的围棋策略**。应与同样设置下、未训练网络及较强基准程序比较。

## 6. 改成 5×5

```bash
./build/minigo train --size 5 --hidden 16 --games 100 --sims 24 --epochs 2 --out models/my5.bin
./build/minigo play --model models/my5.bin --sims 80
```

5×5、隐藏单元 16 的网络共有 **1483 个可学习参数**。网络大小会随棋盘和隐藏单元数量变化，项目设计偏教学，不刻意追求棋力与计算效率。

## 7. 网络结构和手写梯度

特征编码：对于落子方，每个交叉点 `己方棋子=+1`、`对方棋子=-1`、`空=0`，再加 `当前黑白方`、`贴目（归一化且随行棋方变号）`、`连续停着数`、`步数比例`，所以输入维度为 `N*N+4`。

```text
x: [N*N+4]
       │
       ▼
Linear(W0, b0) + ReLU           -> h0: [H]
       │                            │
       ▼                            │ 直连捷径
Linear(W1, b1) + ReLU           │
       │                            │
       ▼                            │
Linear(W2, b2)                  │
       └──────── + <──────────────┘
                 │
                ReLU                 -> h: [H]
                 │
           ┌─────┴──────┐
           ▼            ▼
     policy head     value head
       N*N+1            1
```

- `policy` 预测每个交叉点及 PASS 的分数，合法落子掩码 + softmax 得到概率
- `value` 使用 `tanh` 输出 `[-1,+1]`，表示对当前行棋方有多有利
- 训练策略目标来自 MCTS 子节点的访问比例；胜负目标来自终局获胜方
- 总损失：`交叉熵(policy, MCTS访问分布) + 0.5*(value - 最终胜负)^2`
- `src/network.cpp` 中可以看到完整链式求导、残差跳连的梯度相加、手写 Adam 更新
- 权重随机初始化，不预置定式或人类棋谱

`H=16, N=3` 参数个数：

```text
输入投影： (13*16 + 16)         = 224
残差模块： (16*16 + 16)*2      = 544
策略头：   (16*10 + 10)         = 170
价值头：   (16 + 1)              = 17
合计：                              955
```

## 8. 围棋规则和边界

- 实现了轮流落子、气/相连棋块、提子、禁止自杀、位置超级劫（重复棋盘局面检查）、PASS、两次连续 PASS 结束
- 使用类似中国面积计分：活在盘上的棋子 + 由单方围住的空点，白棋获得贴目；默认为 **0.5 目**，主要为了小棋盘实验，不是正式赛事的默认贴目
- 设有 `4*N*N` 步安全上限，达到后直接用当前棋盘进行面积评分。这是实验限制，不是标准围棋终局规则
- **没有**日本规则复杂的死活协商 / 数目流程；没有完整商业引擎级规则测试；没有对称数据增强、定式、神经网络卷积层、分布式训练
- 由于网络极小、训练目标粗糙、MCTS 搜索次数少，不能保证收敛到最优策略，尤其不能把它当作已求解 3×3/5×5 的数学证明
- 本项目模型 `.bin` 为自定义二进制格式，**与 KataGo 的 `.bin.gz` 不兼容**

## 9. 后续建议

先固定随机种子，训练 100、1000、5000 盘，每次记录对随机玩家胜率；然后实现和旧版模型的对战，才能客观衡量是否成长。可继续加入对称增强、较正规的梯度检查、胜负预测曲线、保存棋谱 SGF、甚至手写卷积层。开发这些功能时仍然不需要外部 AI 框架。

## 模型初始化与模型对战

`match` 用于比较两个网络的真实对弈表现。测试 100 盘时建议启用 `--alternate 1` 自动交换黑白，避免先手优势误导结果：

```bash
git pull
bash build.sh
./build/minigo init --size 3 --hidden 16 --seed 42 --out models/initial3.bin
./build/minigo match --black models/my3.bin --white models/initial3.bin --games 100 --sims 24 --alternate 1
```

- `init` 创建未经训练的随机网络，`--seed 42` 与默认初始化的种子一致；默认拒绝覆盖已有模型，确需覆盖时使用 `--force 1`。
- `match` 中 `--black` 与 `--white` 分别指定模型 A 和模型 B。在 `--alternate 1` 时每局交换一次执黑方，最后输出两个模型的总胜场和各自执黑执白的胜场。
- 默认使用 `--sample 1`，根据 MCTS 访问次数分布抽样落子，避免每盘走法完全相同。可设 `--sample 0` 强制选择搜索最高概率落点，结果通常高度重复。
- `--seed` 控制对战抽样随机种子。双方可用不同隐藏层大小，但棋盘尺寸必须一致。
- `init`、`match` 不会修改已有训练好的模型。

## 可选：KataGo 10b RK3588 NPU 测速

本项目自研 MiniGo 的训练和推理保持不变。额外提供 **KataGo 10b（19×19）** 的 ONNX→RKNN 转换及 RK3588 板端 NPU 神经网络推理测速工具（不含 MCTS 搜索）。

- [转换步骤、安装依赖、测试说明](docs/KATAGO_RK3588_NPU.md)
- [GitHub RKNN 模型发布页](https://github.com/rstg00po54/MiniGoCpp/releases/tag/katago-10b-rknn-v1)

在 **x86 Ubuntu 宿主机**上连接 RK3588 **Android** 开发板，通过 ADB 测速（Android 端不需要 Python）：

```bash
git pull
adb devices -l
bash tools/bench_rk3588_npu.sh
```

如需多设备选择：`ADB_SERIAL=RK3588设备序列号 bash tools/bench_rk3588_npu.sh`。RK3588 Linux（非 Android）仍可使用 `tools/benchmark_rknn.py`，需要在板端安装 Lite2。


## 安卓统一 ADB 测速（RK3588 与 vivo X300 Pro）

新增 `tools/bench_android.sh`，在 Ubuntu x86 虚拟机里调用 ADB，不需要给安卓设备安装 Python：

```bash
git pull
adb devices -l
bash tools/bench_android.sh
```

**vivo** 默认测试 ONNX Runtime CPU、NNAPI 和现有 KataGo OpenCL GPU 测试；**RK3588 Android** 还会包含 RKNN NPU。可单独选择：

```bash
bash tools/bench_android.sh cpu nnapi
bash tools/bench_android.sh gpu
bash tools/bench_android.sh rknn
```

多台设备：`ADB_SERIAL=设备序列号 bash tools/bench_android.sh`。GPU 复用本地 `../GoEngine-java/tools/benchmark_katago_android.sh`，没有该文件时会明确跳过；也可通过 `KATAGO_GPU_SCRIPT=/实际路径/benchmark_katago_android.sh` 指定。详细说明见 [Android 统一测速说明](docs/KATAGO_ANDROID_BENCH.md)。

**注意：** RKNN 和 ONNX Runtime CPU/NNAPI 给出 *NN-only* 的 `nnEvals/s`；GPU 现有脚本给出包含 MCTS 的 `visits/s`，二者不能直接比较。NNAPI 是候选硬件加速路径，不等同于联发科 NPU；测试工具将报告 NNAPI 节点是否实际参与执行，未验证驱动分配前不能声称测试到了手机 NPU。


### ADB 多设备自动选择（vivo 手机 / RK3588）

```bash
bash tools/bench_android.sh vivo        # vivo：尝试加载 Neuron Adapter 并运行一个小型 FP32 计算图（非 KataGo 测速）
bash tools/bench_android.sh vivo cpu nnapi  # 手机 CPU/NNAPI 推理，NNAPI 未必是 NPU
bash tools/bench_android.sh rk          # RK3588：自动运行 RKNN NPU 测速
```

通过制造商和 SoC 属性识别连接的设备，不依赖 ADB 列表顺序。如有相同机型的多个连接，可使用 `ADB_SERIAL=设备序列号` 强制选择。**vivo NPU 模式目前是能力检查，不是已经实现的联发科 Neuron 推理测试**；详见 [Android 测试说明](docs/KATAGO_ANDROID_BENCH.md)。

通过 `bash tools/bench_mediatek_neuron.sh` 可单独执行原生 Neuron Adapter API 检测：编译 Android ARM64 C++，ADB 推送，尝试 Neuron 编译、推理并验证结果。这个阶段还不代表 KataGo 10b 已在 vivo NPU 上运行。
