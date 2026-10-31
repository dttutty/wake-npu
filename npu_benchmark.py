"""Run AMD XRT's end-to-end latency and throughput workloads on the NPU."""

from pathlib import Path
import os
import re
import subprocess
import sys


XRT_SMI = Path(os.environ["WINDIR"]) / "System32" / "AMD" / "xrt-smi.exe"
DEVICE = "0066:00:01.1"


def extract_metric(output: str, label: str, unit: str) -> float:
    match = re.search(rf"Average {label}:\s*([0-9.]+)\s*{re.escape(unit)}", output)
    if not match:
        raise RuntimeError(f"Could not find the {label} result in xrt-smi output")
    return float(match.group(1))


def main() -> int:
    if not XRT_SMI.exists():
        raise FileNotFoundError(f"AMD xrt-smi was not found: {XRT_SMI}")

    command = [
        str(XRT_SMI),
        "--batch",
        "validate",
        "--device",
        DEVICE,
        "--run",
        "all",
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    output = process.stdout + process.stderr
    print(output, end="")
    if process.returncode != 0:
        raise RuntimeError(f"xrt-smi exited with code {process.returncode}")
    if output.count("Test Status           : [PASSED]") != 2:
        raise RuntimeError("Not every XRT hardware validation test passed")

    latency_us = extract_metric(output, "latency", "us")
    throughput = extract_metric(output, "throughput", "op/s")
    print(f"Parsed result: latency={latency_us:.1f} us, throughput={throughput:.1f} op/s")
    print("AMD NPU hardware benchmark passed without an ONNX CPU fallback path.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"AMD NPU hardware benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
