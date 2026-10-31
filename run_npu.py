"""Compile and benchmark a small ONNX model on the AMD NPU."""

import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "tiny_qdq_cnn.onnx"
COMPILED_PATH = ROOT / "tiny_qdq_cnn_ctx.onnx"


def register_vitisai() -> dict:
    process = subprocess.run(
        [sys.executable, str(ROOT / "winml_ep_worker.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    status = json.loads(process.stdout.strip())
    vitis = next(
        (p for p in status["providers"] if p["name"] == "VitisAIExecutionProvider"),
        None,
    )
    if not vitis or not vitis.get("library_path"):
        found = ", ".join(p["name"] for p in status["providers"]) or "none"
        raise RuntimeError(
            "Windows ML did not expose VitisAIExecutionProvider. "
            f"Catalog providers: {found}. Errors: {status['errors']}"
        )
    ort.register_execution_provider_library(vitis["name"], vitis["library_path"])
    print(f"Registered {vitis['name']}: {vitis['library_path']}")
    return status


def main() -> int:
    if not MODEL_PATH.exists():
        subprocess.run([sys.executable, str(ROOT / "make_qdq_model.py")], check=True)

    register_vitisai()
    options = ort.SessionOptions()
    options.log_severity_level = 1
    options.set_provider_selection_policy(
        ort.OrtExecutionProviderDevicePolicy.PREFER_NPU
    )

    print(f"Compiling {MODEL_PATH.name} for the preferred NPU...")
    compiler = ort.ModelCompiler(options, str(MODEL_PATH))
    compiler.compile_to_file(str(COMPILED_PATH))

    session = ort.InferenceSession(str(COMPILED_PATH), sess_options=options)
    providers = session.get_providers()
    print(f"Active providers: {providers}")
    if not providers or providers[0] != "VitisAIExecutionProvider":
        raise RuntimeError(f"Model did not select the AMD NPU: {providers}")

    rng = np.random.default_rng(8845)
    sample = rng.uniform(-1.0, 1.0, size=(1, 3, 32, 32)).astype(np.float32)
    input_name = session.get_inputs()[0].name
    for _ in range(20):
        session.run(None, {input_name: sample})

    iterations = 500
    start = time.perf_counter()
    output = None
    for _ in range(iterations):
        output = session.run(None, {input_name: sample})[0]
    elapsed = time.perf_counter() - start

    print(f"Output shape: {output.shape}; checksum: {float(output.sum()):.6f}")
    print(f"{iterations} runs in {elapsed:.3f}s; average {elapsed * 1000 / iterations:.3f} ms")
    print("NPU smoke test passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"NPU smoke test failed: {exc}", file=sys.stderr)
        sys.exit(1)
