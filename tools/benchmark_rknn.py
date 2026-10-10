#!/usr/bin/env python3
"""RK3588 NPU NN-only benchmark for the KataGo 10b 19x19 RKNN model.

Runs on an RK3588 board (aarch64) with rknn-toolkit-lite2 and librknnrt.so.
Measures end-to-end Python inference() latency, NOT KataGo MCTS visits/sec.
"""
import argparse
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


def percentile(values, pct):
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * pct / 100.0
    low, high = int(index), min(int(index) + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (index - low)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/katago_10b_19x19.rknn")
    p.add_argument("--core", choices=["0", "1", "2", "all"], default="all")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=100)
    args = p.parse_args()
    if args.runs <= 0 or args.warmup < 0:
        p.error("runs must be > 0 and warmup >= 0")
    if not Path(args.model).is_file():
        p.error("model not found: " + args.model)
    if platform.machine() not in ("aarch64", "arm64"):
        p.error("RKNNLite benchmark must run on RK3588 aarch64, not your x86 VM")
    # Realistic empty 19x19 board-shaped inputs. These are synthetically generated
    # benchmark tensors, not a fully encoded KataGo board position.
    spatial = np.zeros((1, 22, 19, 19), dtype=np.float32)
    global_features = np.zeros((1, 19, 1, 1), dtype=np.float32)
    mask = np.ones((1, 1, 19, 19), dtype=np.float32)
    inputs = [spatial, global_features, mask]
    mask_map = {"0": RKNNLite.NPU_CORE_0, "1": RKNNLite.NPU_CORE_1,
                "2": RKNNLite.NPU_CORE_2, "all": RKNNLite.NPU_CORE_0_1_2}
    engine = RKNNLite()
    ret = engine.load_rknn(args.model)
    if ret != 0:
        raise RuntimeError("load_rknn failed: %s" % ret)
    try:
        ret = engine.init_runtime(core_mask=mask_map[args.core])
        if ret != 0:
            raise RuntimeError("init_runtime failed: %s (check rknpu driver/runtime)" % ret)
        def infer():
            out = engine.inference(inputs=inputs, data_format=["nchw"] * len(inputs))
            if not isinstance(out, (list, tuple)) or len(out) != 5:
                raise RuntimeError("Expected 5 KataGo outputs; got %s" % type(out))
            return out

        print("Model:", args.model, "core:", args.core, "warmup:", args.warmup, "runs:", args.runs, flush=True)
        print("Inputs:", [tuple(x.shape) for x in inputs], flush=True)
        print("This is NN inference only, not visits/s; CPU MCTS not included.", flush=True)
        for _ in range(args.warmup):
            infer()
        times = []
        last = None
        for _ in range(args.runs):
            t0 = time.perf_counter_ns()
            last = infer()
            times.append((time.perf_counter_ns() - t0) / 1e6)
        print("Outputs:", [tuple(np.asarray(x).shape) for x in last], flush=True)
        print("Mean %.3f ms | P50 %.3f ms | P95 %.3f ms | NN evals/s %.2f" %
              (statistics.mean(times), percentile(times, 50), percentile(times, 95),
               1000.0 / statistics.mean(times)), flush=True)
        for i, x in enumerate(last):
            x = np.asarray(x)
            print("Output%d min=%.5f max=%.5f finite=%s" %
                  (i, float(x.min()), float(x.max()), bool(np.isfinite(x).all())), flush=True)
    finally:
        engine.release()


if __name__ == "__main__":
    main()
