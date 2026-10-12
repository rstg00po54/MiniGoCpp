#!/usr/bin/env python3
"""Generate one identical KataGo 10b FP32 input + CPU reference for ORT/RKNN from
the same K10NW001 full-network packet used by OpenCL and MediaTek Neuron.

K10IN001 = 8-byte magic; NCHW FP32 spatial[7942], global[19], mask[361],
then five original ONNX FP32 output heads in packet order [1,361,3,4,361].
Only original 19x19 full-mask 10b is supported.
"""
import argparse
import struct
from pathlib import Path

import numpy as np

EXPECTED = (1, 361, 3, 4, 361)


def read_exact(handle, size):
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("Truncated full KataGo packet")
    return data


def u32(f):
    return struct.unpack("<I", read_exact(f, 4))[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--packet", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    with open(args.packet, "rb") as f:
        header = read_exact(f, 24)
        magic, n, op_count, n_inputs, n_outputs = struct.unpack("<8s4I", header)
        if magic != b"K10NW001" or n_inputs != 2 or n_outputs != 5 or op_count < 100:
            raise ValueError("Only the verified original full 19x19 KataGo 10b packet is supported")
        tensors = []
        for _ in range(n):
            type_id, rank, flags, n_bytes, d0, d1, d2, d3 = struct.unpack("<8I", read_exact(f, 32))
            if rank > 4 or n_bytes > 40 * 1024 * 1024:
                raise ValueError("Bad operand size")
            data = read_exact(f, n_bytes)
            tensors.append((type_id, flags, (d0, d1, d2, d3)[:rank], data))
        for _ in range(op_count):
            _, inputs, outputs = struct.unpack("<3I", read_exact(f, 12))
            read_exact(f, 4 * (inputs + outputs))
        input_ids = struct.unpack("<2I", read_exact(f, 8))
        output_ids = struct.unpack("<5I", read_exact(f, 20))
        refs = []
        names = []
        for expected in EXPECTED:
            name = read_exact(f, u32(f)).decode("utf-8")
            length = u32(f)
            if length != expected * 4:
                raise ValueError("Invalid reference head shape for " + name)
            refs.append(np.frombuffer(read_exact(f, length), dtype="<f4").copy())
            names.append(name)
        if f.read(1):
            raise ValueError("Unexpected trailing data")
    spatial = tensors[input_ids[0]]
    glob = tensors[input_ids[1]]
    if spatial[0] != 8 or spatial[2] != (1, 19, 19, 22):
        raise ValueError("Unexpected spatial input shape")
    if glob[0] != 8 or glob[2] != (1, 1, 1, 19):
        raise ValueError("Unexpected global input shape")
    spatial_nchw = np.frombuffer(spatial[3], dtype="<f2").reshape(1, 19, 19, 22).transpose(0, 3, 1, 2).astype("<f4")
    global_nchw = np.frombuffer(glob[3], dtype="<f2").astype("<f4")
    mask = np.ones(361, dtype="<f4")
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"K10IN001")
        for data in (spatial_nchw, global_nchw, mask):
            f.write(np.ascontiguousarray(data, dtype="<f4").tobytes())
        for ref in refs:
            f.write(ref.astype("<f4").tobytes())
    print("COMPARE_INPUT_READY file={} bytes={} references={}".format(path, path.stat().st_size, names))


if __name__ == "__main__":
    main()
