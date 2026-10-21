# MiniGoCpp 模型导出为 RK3588 RKNN

本工具把 MiniGoCpp 自己训练的 `MGNET001` 检查点（例如 `models/my3.bin`）转换为 ONNX，再使用 **RKNN-Toolkit2** 生成 RK3588 使用的 RKNN 文件。

- 脚本：[`tools/convert_minigo_rknn.py`](../tools/convert_minigo_rknn.py)
- 原始模型：`src/network.cpp` 的全连接残差网络，不是 KataGo
- 只导出推理权重；Adam `m/v`、训练步数不参与推理
- 为便于 RKNN 编译，Dense 等价表示为 **1×1 Conv（输入空间 1×1）**；不是新增用于分析棋盘的 3×3 卷积
- NPU 只执行前向推理；输入编码、合法落子掩码、Softmax、MCTS 仍由应用程序处理

## 1. 安装依赖

在 x86 Ubuntu 上，使用与 RKNN-Toolkit2 官方 wheel 兼容的 Python 环境：

```bash
python -m pip install numpy onnx onnxruntime
```

另外安装你本机适配的 RKNN-Toolkit2（`from rknn.api import RKNN` 必须可以导入）。不需要在 RK3588 板端安装完整的 Toolkit2。

## 2. 生成模型

在仓库根目录执行：

```bash
# 查看现有检查点信息；不需要 ONNX 或 RKNN-Toolkit2
python tools/convert_minigo_rknn.py --model models/my3.bin --inspect-only

# 第一步：只导出 ONNX，并在可用时做 ONNX Runtime 与 NumPy 输出对比
python tools/convert_minigo_rknn.py --model models/my3.bin --onnx-only

# 第二步：完整导出 ONNX + RKNN（RK3588，非 INT8 量化）
python tools/convert_minigo_rknn.py --model models/my3.bin
```

默认生成：

```text
models/my3.bin    原始训练检查点
models/my3.onnx   ONNX 推理图
models/my3.rknn   RK3588 RKNN 推理模型
```

也可使用 `--onnx`、`--rknn` 自定义输出位置。

## 3. 输入输出规格

`n` 为棋盘边长，由模型头部读取。

| 项目 | 名称 | 格式 |
|---|---|---|
| 输入 | `input_features` | `float32 [1, n*n+4, 1, 1]` (NCHW) |
| 输出 0 | `policy_logits` | `[1, n*n+1, 1, 1]`；未做掩码和 Softmax |
| 输出 1 | `value` | `[1, 1, 1, 1]`；经过 tanh，约为 -1 至 +1 |

输入特征必须遵循 `Board::encode()`：

1. 前 `n*n` 个：逐点 `cell * toMove`（己方 +1、对方 -1、空点 0）。
2. `toMove`（当前行棋方，黑 +1 / 白 -1）。
3. `komi * toMove / (n*n)`。
4. 连续 PASS 次数。
5. `moves / maxMoves()`。

对于 3×3、hidden=16 的 `my3.bin`，总计 955 个可训练参数；输入长度 13，策略输出长度 10，价值输出长度 1。

## 4. 验证与限制

- 本脚本已通过 Python 语法测试，并用模拟 `MGNET001` 文件检查头部与形状解析。
- 尚未在当前环境完成真实 `my3.bin` 的 ONNX Runtime 数值校验，也未完成 RKNN-Toolkit2 编译与 RK3588 板端运行。
- 如果安装了 `onnxruntime`，导出后脚本会比较 20 组输入的 NumPy 与 ONNX 输出，最大绝对误差阈值为 `1e-4`。未安装则跳过该检查。
- 生成 `.rknn` 不代表已在设备上成功推理；应继续用 RKNN Runtime 与原始 CPU 模型比较相同输入的输出，再测试性能。
- `my3.bin` 网络非常小，CPU 可能比 NPU 更快；不要预设 NPU 一定带来加速。

**注意：** `my3.bin` 是 MiniGoCpp 自定义检查点，KataGo 官方的 `.bin.gz` 无法用这个脚本转换。
