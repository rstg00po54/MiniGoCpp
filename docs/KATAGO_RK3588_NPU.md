# KataGo 10b -> ONNX -> RK3588 NPU

This is an **optional standalone inference benchmark**. MiniGoCpp's existing training/search engine is unchanged. The 10b network is a different model; do not load the `minigo` .bin files with RKNN.

## Status and model

- KataGo reference: `g170-b10c128-s1141046784-d204142634`, 19x19 network buffer.
- Input ONNX: [10b_19x19.onnx](https://github.com/rstg00po54/go/releases/tag/katago-10b-onnx-v1)
- ONNX opset 20, **three inputs** in order: `InputSpatial [1,22,19,19]`, `InputGlobal [1,19,1,1]`, `InputMask [1,1,19,19]`. Original batch dimension is dynamic.
- Five outputs: `OutputPolicyPass`, `OutputPolicy`, `OutputValue`, `OutputScoreValue`, `OutputOwnership`.
- Official RKNN Toolkit2 2.3.2 currently lists ONNX opset up to 19. The conversion script pins batch=1 and uses ONNX version conversion to target opset 19. **That does not guarantee RKNN compiler compatibility.** Check CI logs and artifacts before assuming success.

## Automated model conversion (GitHub)

See [Actions: Build KataGo RK3588 RKNN](../.github/workflows/katago_rknn.yml). On success the NPU binary is uploaded to:

[MiniGoCpp Release: katago-10b-rknn-v1](https://github.com/rstg00po54/MiniGoCpp/releases/tag/katago-10b-rknn-v1)

If conversion fails, the workflow's logs explain which operator or version is unsupported. We do not upload an unverified or empty binary.

## Running on RK3588 (Ubuntu/Debian Linux, ARM64)

RKNN model compilation is done on x86 Ubuntu or GitHub Actions, **not** on RK3588. The board must have a working RKNPU2 driver and Rockchip `librknnrt.so`. The Python package alone cannot supply a kernel NPU driver.

```bash
git pull
python3 -m pip install --user 'numpy<2' 'rknn-toolkit-lite2==2.3.2'
bash tools/bench_rk3588_npu.sh
```

The shell script downloads the `.rknn` from MiniGoCpp Releases if missing. If the release does not exist, conversion hasn't succeeded; don't use the benchmark yet.

You can also use a local model:

```bash
RUNS=200 WARMUP=30 bash tools/bench_rk3588_npu.sh /path/to/katago_10b_19x19.rknn
```

Or select a single core:

```bash
python3 tools/benchmark_rknn.py --model models/katago_10b_19x19.rknn --core 0 --warmup 20 --runs 100
python3 tools/benchmark_rknn.py --model models/katago_10b_19x19.rknn --core all --warmup 20 --runs 100
```

Measurements report `Mean ms`, `P50`, `P95`, `NN evals/s`. These include Python RKNNLite call and input/output costs. The inputs are synthetically generated FP32 arrays with correct shapes, not full KataGo feature encoder output. **This does not measure KataGo visits/s and is not an end-to-end Go engine comparison.** Three-core mode may not scale for a single unbatched inference.

## Local x86 conversion

Use Ubuntu 22.04, Python 3.10, and RKNN-Toolkit2 2.3.2. Install matching dependencies. Be aware Toolkit2 currently conflicts with ONNX 1.20; ONNX 1.18 is recommended:

```bash
python3 -m pip install 'numpy==1.26.4' 'onnx==1.18.0' 'rknn-toolkit2==2.3.2'
wget -c https://github.com/rstg00po54/go/releases/download/katago-10b-onnx-v1/10b_19x19.onnx
python3 tools/convert_katago_rknn.py --onnx 10b_19x19.onnx
```

Only if that completes successfully, transfer `models/katago_10b_19x19.rknn` to the board. If `onnx.version_converter` or `rknn.build` fails, check the error rather than changing an opset number directly in metadata. FP32 input tensors and unquantized RKNN build do not necessarily mean the NPU computes all layers at FP32 precision; verify neural outputs before integrating with MCTS.

References: [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2), [RKNN-Lite2](https://pypi.org/project/rknn-toolkit-lite2/2.3.2/).
