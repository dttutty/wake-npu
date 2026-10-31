"""Run one real Argos zh-en encoder QKV projection on the Phoenix NPU."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16

from ct2_weights import ModelWeights


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "translation-data" / "packages" / "translate-zh_en-1_9"
EXAMPLE = ROOT / "third_party" / "mlir-aie" / "programming_examples" / "basic" / "matrix_multiplication" / "single_core"
sys.path.insert(0, str(EXAMPLE))
from single_core import single_core  # noqa: E402


def dequantized(model, name):
    canonical_name = model.aliases.get(name, name)
    values = model.array(canonical_name)
    scales = model.array(canonical_name + "_scale")
    if values.dtype != np.int8 or scales.shape != values.shape[:1]:
        raise ValueError(f"Unexpected per-row INT8 quantization for {name}")
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError(f"Invalid quantization scales for {name}")
    return values.astype(np.float32) / scales[:, None]


def token_pieces(text):
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [str(python), str(Path(__file__).with_name("argos_pieces.py")),
         str(PACKAGE / "sentencepiece.model"), text],
        check=True, capture_output=True, text=True, encoding="utf-8", env=env,
    )
    return json.loads(result.stdout)


def main(text):
    model = ModelWeights(PACKAGE / "model" / "model.bin")
    if model.model_type != "TransformerSpec" or model.revision != 7:
        raise ValueError(f"Unsupported model: {model.model_type} revision {model.revision}")
    vocabulary = json.loads((PACKAGE / "model" / "shared_vocabulary.json").read_text(encoding="utf-8"))
    token_to_id = {piece: index for index, piece in enumerate(vocabulary)}
    pieces = token_pieces(text) + ["</s>"]
    if len(pieces) > 32:
        raise ValueError(f"Input has {len(pieces)} pieces; this NPU experiment supports at most 32")
    ids = [token_to_id.get(piece, token_to_id["<unk>"]) for piece in pieces]

    embeddings = dequantized(model, "encoder/embeddings_0/weight")
    weight = dequantized(model, "encoder/layer_0/self_attention/linear_0/weight")
    if embeddings.shape != (65000, 512) or weight.shape != (1536, 512):
        raise ValueError("Unexpected embedding or QKV weight dimensions")
    x = np.zeros((64, 512), dtype=bfloat16)
    x[:len(ids)] = embeddings[ids].astype(bfloat16)
    w = weight.T.astype(bfloat16)

    x_npu = iron.tensor(x.reshape(-1), dtype=bfloat16, device="npu")
    w_npu = iron.tensor(w.reshape(-1), dtype=bfloat16, device="npu")
    y_npu = iron.zeros(64 * 1536, dtype=bfloat16, device="npu")
    single_core(
        x_npu, w_npu, y_npu,
        M=64, K=512, N=1536, m=32, k=32, n=32,
        dtype_in_str="bf16", dtype_out_str="bf16", b_col_maj=0,
    )
    actual = y_npu.numpy().reshape(64, 1536).astype(np.float32)
    expected = (x.astype(np.float32) @ w.astype(np.float32)).astype(bfloat16).astype(np.float32)
    np.testing.assert_allclose(actual, expected, atol=0.1, rtol=0.05)
    error = np.max(np.abs(actual - expected))
    print(f"Input: {text}")
    print(f"SentencePiece: {pieces}")
    print(f"Vocabulary IDs: {ids}")
    print(f"NPU: real model embedding x encoder layer-0 QKV weight = {actual.shape}")
    print(f"Max abs error vs NumPy BF16 reference: {error:.6g}")
    print("PASS: real model weight projection on NPU; not a full translation inference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=os.environ.get("NPU_TEST_TEXT", "你好，世界。"))
    main(parser.parse_args().text)
