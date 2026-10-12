#!/usr/bin/env python3
"""Check the index math used by the tiled OpenCL 10b convolution.

The Python emulation is independent of the actual device: the on-device test
must still verify all five output heads against the original ONNX reference.
"""
import numpy as np


def baseline(x, weights, bias, padding):
    h, w, ic = x.shape
    kh, kw, icw, oc = weights.shape
    assert ic == icw
    pt, pl = padding
    oh, ow = h, w
    result = np.empty((oh, ow, oc), dtype=np.float32)
    for py in range(oh):
        for px in range(ow):
            for c in range(oc):
                val = float(bias[c])
                for ky in range(kh):
                    for kx in range(kw):
                        iy = py + ky - pt
                        ix = px + kx - pl
                        if 0 <= iy < h and 0 <= ix < w:
                            for ci in range(ic):
                                val += float(x[iy, ix, ci]) * float(weights[ky, kx, ci, c])
                result[py, px, c] = val
    return result


def tiled(x, weights, bias, padding):
    h, w, ic = x.shape
    kh, kw, _, oc = weights.shape
    pt, pl = padding
    oh, ow = h, w
    result = np.empty((oh * ow, oc), dtype=np.float32)
    M, N, K = 8, 16, 32
    for tile_m in range(0, oh * ow, M):
        for tile_n in range(0, oc, N):
            sums = np.zeros((M, N), dtype=np.float32)
            for k0 in range(0, kh * kw * ic, K):
                a = np.zeros((M, K), dtype=np.float32)
                b = np.zeros((K, N), dtype=np.float32)
                for lm in range(M):
                    p = tile_m + lm
                    for lk in range(K):
                        k = k0 + lk
                        if p < oh * ow and k < kh * kw * ic:
                            y, x0 = divmod(p, ow)
                            spatial, ci = divmod(k, ic)
                            ky, kx = divmod(spatial, kw)
                            iy, ix = y + ky - pt, x0 + kx - pl
                            if 0 <= iy < h and 0 <= ix < w:
                                a[lm, lk] = x[iy, ix, ci]
                for lk in range(K):
                    k = k0 + lk
                    for ln in range(N):
                        c = tile_n + ln
                        if k < kh * kw * ic and c < oc:
                            spatial, ci = divmod(k, ic)
                            ky, kx = divmod(spatial, kw)
                            b[lk, ln] = weights[ky, kx, ci, c]
                sums += a @ b
            for lm in range(M):
                for ln in range(N):
                    p, c = tile_m + lm, tile_n + ln
                    if p < oh * ow and c < oc:
                        result[p, c] = sums[lm, ln] + bias[c]
    return result.reshape((oh, ow, oc))


rng = np.random.default_rng(20261012)
for h, w, ic, oc, kh, kw, pt, pl in [
    (3, 4, 3, 7, 3, 3, 1, 1),
    (5, 3, 5, 17, 1, 1, 0, 0),
    (4, 6, 8, 19, 3, 3, 1, 1),
    (3, 3, 6, 5, 5, 5, 2, 2),
    (1, 1, 19, 13, 1, 1, 0, 0),
]:
    x = rng.normal(size=(h, w, ic)).astype(np.float16)
    weight = rng.normal(size=(kh, kw, ic, oc)).astype(np.float16)
    bias = rng.normal(size=oc).astype(np.float16)
    expected = baseline(x, weight, bias, (pt, pl))
    got = tiled(x, weight, bias, (pt, pl))
    assert np.allclose(got, expected, rtol=2e-5, atol=2e-4), (h,w,ic,oc,kh,kw,np.max(abs(got-expected)))
print("PASS: local-memory tiled convolution indexing matches independent direct convolution")
