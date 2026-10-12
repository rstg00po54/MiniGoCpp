#!/usr/bin/env python3
"""Smoke test vendor-agnostic ARM64 OpenCL link-only stub (never deployed)."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parent
cc = sys.argv[1] if len(sys.argv) > 1 else None
if not cc or not Path(cc).is_file():
    raise SystemExit("Usage: python3 tools/test_opencl_android_link_stub.py /path/to/aarch64-linux-android26-clang")
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    header = tmp / "CL" / "cl.h"
    header.parent.mkdir(parents=True)
    names = ["clGetPlatformIDs", "clCreateContext", "clEnqueueNDRangeKernel", "clReleaseContext"]
    names.extend("clMockCoreFunc{}".format(i) for i in range(105))
    header.write_text("\n".join("extern CL_API_ENTRY cl_int CL_API_CALL {}(void);".format(n) for n in names))
    stub = tmp / "libOpenCL.so"
    subprocess.run([sys.executable, str(root / "create_opencl_android_link_stub.py"),
                    "--header", str(header), "--cc", cc, "--out", str(stub)], check=True)
    sym = subprocess.check_output(["readelf", "-W", "--dyn-syms", str(stub)], text=True)
    assert all(re.search(r"\b" + name + r"\b", sym) for name in names[:4])
    source = tmp / "probe.c"
    source.write_text("extern int clGetPlatformIDs(unsigned, void*, unsigned*);\n"
                      "int main(void) { return clGetPlatformIDs(0, 0, 0); }\n")
    executable = tmp / "probe_android"
    subprocess.run([cc, str(source), "-L" + td, "-lOpenCL", "-o", str(executable)], check=True)
    dynamic = subprocess.check_output(["readelf", "-W", "-d", str(executable)], text=True)
    versions = subprocess.check_output(["readelf", "-W", "-V", str(executable)], text=True)
    assert "Shared library: [libOpenCL.so]" in dynamic, dynamic
    assert "OPENCL_1.0" not in versions, versions
    elf = subprocess.check_output(["readelf", "-W", "-h", str(executable)], text=True)
    assert "Machine:                           AArch64" in elf
    print("PASS: Android ARM64 OpenCL portable import SONAME=libOpenCL.so, no driver-specific version, stub not deployed")
