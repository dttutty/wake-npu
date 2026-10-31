"""Run and profile AMD's Ryzen AI quick-test model on the Phoenix NPU."""

from collections import Counter
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import onnxruntime as ort


INSTALL_DIR = Path(
    os.environ.get("RYZEN_AI_INSTALLATION_PATH", r"C:\Program Files\RyzenAI\1.8.0")
)
MODEL_PATH = INSTALL_DIR / "quicktest" / "test_model.onnx"


def find_phoenix_xclbin() -> Path:
    override = os.environ.get("AMD_NPU_XCLBIN")
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"AMD_NPU_XCLBIN does not exist: {path}")
        return path

    driver_store = Path(r"C:\Windows\System32\DriverStore\FileRepository")
    candidates = list(
        driver_store.glob("kipudrv.inf_amd64_*/4x4_3.5.0.0-2352.xclbin")
    )
    if not candidates:
        raise FileNotFoundError("Phoenix 4x4_3.5.0.0-2352.xclbin was not found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def provider_counts(profile_path: Path) -> Counter[str]:
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    providers: Counter[str] = Counter()
    for event in events:
        if event.get("cat") != "Node":
            continue
        provider = event.get("args", {}).get("provider")
        if provider:
            providers[provider] += 1
    return providers


def main() -> int:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Ryzen AI test model not found: {MODEL_PATH}")
    if "VitisAIExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            f"VitisAIExecutionProvider is unavailable: {ort.get_available_providers()}"
        )

    options = ort.SessionOptions()
    options.log_severity_level = 2
    options.enable_profiling = True
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / "amd_npu_profile")

    # PHX/HPT uses the X1 target and the AIE2 overlay. Passing the xclbin
    # explicitly prevents the multi-generation driver from selecting AIE2P.
    xclbin = find_phoenix_xclbin()
    provider_options = {
        "target": "X1",
        "xlnx_enable_py3_round": "0",
        "xclbin": str(xclbin),
    }
    print(f"ORT: {ort.__version__}")
    print(f"Model: {MODEL_PATH}")
    print(f"Phoenix xclbin: {xclbin}")
    print(f"Provider options: {provider_options}")

    compile_start = time.perf_counter()
    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=["VitisAIExecutionProvider"],
        provider_options=[provider_options],
    )
    compile_seconds = time.perf_counter() - compile_start
    print(f"Session providers: {session.get_providers()}")

    rng = np.random.default_rng(8845)
    sample = rng.random((1, 3, 32, 32), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: sample})[0]

    profile_path = Path(session.end_profiling())
    counts = provider_counts(profile_path)
    profile_path.unlink(missing_ok=True)
    print(f"Profiled node executions by provider: {dict(counts)}")
    if counts.get("VitisAIExecutionProvider", 0) == 0:
        raise RuntimeError("The profile contains no NPU-executed VitisAI nodes")

    for _ in range(20):
        session.run(None, {input_name: sample})
    iterations = int(os.environ.get("AMD_NPU_ITERATIONS", "500"))
    start = time.perf_counter()
    for _ in range(iterations):
        output = session.run(None, {input_name: sample})[0]
    elapsed = time.perf_counter() - start

    print(f"Compile/session setup: {compile_seconds:.3f} s")
    print(f"Output shape: {output.shape}; checksum: {float(output.sum()):.6f}")
    print(f"{iterations} runs: {elapsed:.3f} s; average: {elapsed * 1000 / iterations:.3f} ms")
    print("AMD NPU VitisAI test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
