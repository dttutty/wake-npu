"""Open the installed AMD NPU through the driver-provided pyxrt binding."""

from pathlib import Path
import os
import sys


def find_driver_package() -> Path:
    root = Path(os.environ["WINDIR"]) / "System32" / "DriverStore" / "FileRepository"
    packages = sorted(root.glob("kipudrv.inf_amd64_*"), reverse=True)
    for package in packages:
        if (package / "pyxrt.pyd").exists() and (package / "xrt_coreutil.dll").exists():
            return package
    raise FileNotFoundError("The AMD kipudrv package with pyxrt was not found.")


def main() -> int:
    package = find_driver_package()
    os.add_dll_directory(str(package))
    sys.path.insert(0, str(package))

    import pyxrt

    count = pyxrt.enumerate_devices()
    print(f"XRT device count: {count}")
    if count < 1:
        raise RuntimeError("XRT did not enumerate an NPU.")

    device = pyxrt.device(0)
    keys = ("name", "bdf", "interface_uuid", "max_clock_frequency_mhz")
    for key in keys:
        try:
            value = device.get_info(getattr(pyxrt.xrt_info_device, key))
        except Exception as exc:
            value = f"unavailable ({exc})"
        print(f"{key}: {value}")
    print(f"Driver package: {package}")
    print("XRT NPU probe passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
