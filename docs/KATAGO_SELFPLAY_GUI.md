# KataGo 自动对弈可视化（Ubuntu）

运行 `tools/katago_selfplay_gui.py`，可以在 Tkinter 棋盘上观看真正的 KataGo **黑白双方自己对弈**，每局结束后自动开始下一局，并保存 SGF 棋谱。

## 运行条件

- Ubuntu / Python 3.8 或更新版本
- Tkinter：`sudo apt install python3-tk`
- **KataGo GTP 可执行程序**（Linux 版 `katago`）
- **KataGo 原生神经网络模型**（例如 `10b.bin` 或 `.bin.gz`）
- KataGo 的 GTP 配置文件（建议显式指定）

> 此工具调用 KataGo 的 GTP 协议，不是 MiniGoCpp 自训练模型，也不是 Android JNI。RKNN/TFLite 模型不能直接传入 `--model`。

## 示例

在 MiniGoCpp 仓库根目录执行：

```bash
git pull
sudo apt install python3-tk
python3 tools/katago_selfplay_gui.py \
  --engine /path/to/katago \
  --model /path/to/10b.bin \
  --config /path/to/gtp.cfg \
  --size 9 \
  --visits 16 \
  --delay 0.7
```

将三个路径替换为你本机文件的实际绝对路径。若 `katago` 已在 PATH 中，可以省略 `--engine`；如果安装和配置允许，也可以省略 `--config`。

常用选项：

| 选项 | 说明 |
|---|---|
| `--size 9` | 棋盘大小：9、11、13、15、19 |
| `--visits 16` | KataGo 每手搜索次数，数字越大通常越慢 |
| `--delay 0.7` | 每手之间的显示间隔（秒） |
| `--between-games 3` | 每局结束后的等待间隔（秒） |
| `--komi 7.5` | 贴目 |
| `--records selfplay_records` | SGF 棋谱和引擎错误日志的保存目录 |

点击界面的“暂停/继续”控制观看进度，或选择下一局。正常终局与认输会自动记录，保存为 `selfplay_records/KataGo_*.sgf`；异常原因通常在同目录的 `katago_stderr.log`。

**说明：** 这是一个观看 KataGo 自对弈的独立桌面工具，不会修改原来的 MiniGoCpp 训练逻辑。程序已做 Python 语法检查和模拟 GTP 流程验证，但尚未在你的 Ubuntu KataGo 环境中实机测试。
