#!/usr/bin/env python3
"""Build a LINK-ONLY Android ARM64 OpenCL import library (not a runtime).

Emits the OpenCL core API symbols from Khronos CL/cl.h into a tiny ELF with
SONAME libOpenCL.so. No vendor binaries, ICD loader, or symbol-version bindings
are required at build time. The resulting .so must NEVER be pushed to Android:
at runtime the Android vendor's genuine libOpenCL.so supplies the functions.
"""
import argparse
import re
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--header", type=Path, required=True, help="Path to CL/cl.h")
    p.add_argument("--cc", required=True, help="Android aarch64-linux-android26-clang")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    data = args.header.read_text(encoding="utf-8")
    names = sorted(set(re.findall(r"\bCL_API_CALL\s+(cl[A-Za-z][A-Za-z0-9_]*)\s*\(", data)))
    for required in ("clGetPlatformIDs", "clCreateContext", "clEnqueueNDRangeKernel", "clReleaseContext"):
        if required not in names:
            raise RuntimeError("Incomplete OpenCL headers: missing " + required)
    if len(names) < 90:
        raise RuntimeError("Unexpectedly small OpenCL core API list: {}".format(len(names)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    asm_file = args.out.with_suffix(".linkonly.S")
    lines = [
        "// Link-only OpenCL symbols; do not deploy this library to Android.",
        ".text",
    ]
    for name in names:
        lines.extend([
            ".p2align 2",
            ".global " + name,
            ".type " + name + ", %function",
            name + ":",
            "    mov w0, #-59  // CL_INVALID_OPERATION; must never execute",
            "    ret",
            ".size " + name + ", .-" + name,
        ])
    lines.append('.section .note.GNU-stack,"",%progbits')
    asm_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run([args.cc, "-shared", "-nostdlib", "-fPIC", "-Wl,-soname,libOpenCL.so",
                    "-Wl,-z,defs", "-o", str(args.out), str(asm_file)], check=True)
    print("OPENCL_LINK_STUB_READY path={} exports={} soname=libOpenCL.so runtime_deploy=NO".format(args.out, len(names)))


if __name__ == "__main__":
    main()
