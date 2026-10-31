"""Check whether the installed XRT driver accepts AMD's Ryzen AI 1.2 PHX GEMM."""

import os
from pathlib import Path
import sys


DRIVER_DIR = Path(os.environ["WINDIR"]) / "System32" / "AMD"
XCLBIN = Path(__file__).with_name("gemm_4x4_a8w8acc32.xclbin")


def main() -> int:
    if not XCLBIN.is_file():
        raise FileNotFoundError(XCLBIN)
    os.add_dll_directory(str(DRIVER_DIR))
    sys.path.insert(0, str(DRIVER_DIR))
    import pyxrt

    print(f"XCLBIN: {XCLBIN}")
    binary = pyxrt.xclbin(str(XCLBIN))
    print(f"Binary UUID: {binary.get_uuid()}")
    print(f"Kernels: {[kernel.get_name() for kernel in binary.get_kernels()]}")

    device = pyxrt.device(0)
    uuid = device.register_xclbin(binary)
    print(f"Driver registered binary: {uuid}")
    context = pyxrt.hw_context(device, uuid)
    print(f"Hardware context created: {context}")
    print("PHX GEMM xclbin accepted by the NPU driver; no inference run yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
