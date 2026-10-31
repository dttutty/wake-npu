"""Single-head, noncausal attention using three IRON NPU kernels."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "third_party" / "mlir-aie" / "programming_examples"
sys.path.insert(0, str(EXAMPLES / "basic" / "matrix_multiplication" / "single_core"))
sys.path.insert(0, str(EXAMPLES / "ml" / "softmax"))
from single_core import single_core  # noqa: E402
from softmax import softmax  # noqa: E402


def load_layer_norm():
    path = ROOT / "experiments" / "iron-layernorm" / "run.py"
    spec = spec_from_file_location("iron_layernorm", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load LayerNorm design: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    queries = 64
    keys = 1024  # IRON's softmax handles each 1024-element row independently.
    head_dim = 32
    rng = np.random.default_rng(2026)

    q = (rng.normal(0, 0.7, (queries, head_dim)) / np.sqrt(head_dim)).astype(bfloat16)
    k = rng.normal(0, 0.7, (keys, head_dim)).astype(bfloat16)
    v = rng.normal(0, 0.7, (keys, head_dim)).astype(bfloat16)

    q_npu = iron.tensor(q.reshape(-1), dtype=bfloat16, device="npu")
    kt_npu = iron.tensor(k.T.copy().reshape(-1), dtype=bfloat16, device="npu")
    v_npu = iron.tensor(v.reshape(-1), dtype=bfloat16, device="npu")
    scores = iron.zeros(queries * keys, dtype=bfloat16, device="npu")
    weights = iron.zeros_like(scores)
    output = iron.zeros(queries * head_dim, dtype=bfloat16, device="npu")
    normalized = iron.zeros_like(output)

    single_core(
        q_npu, kt_npu, scores,
        M=queries, K=head_dim, N=keys,
        m=32, k=32, n=32,
        dtype_in_str="bf16", dtype_out_str="bf16",
    )
    softmax(scores, weights, size=queries * keys, num_channels=1)
    single_core(
        weights, v_npu, output,
        M=queries, K=keys, N=head_dim,
        m=32, k=32, n=32,
        dtype_in_str="bf16", dtype_out_str="bf16",
    )
    layer_norm = load_layer_norm()
    layer_norm.layer_norm_aie2(output, normalized)

    score_ref = (q.astype(np.float32) @ k.astype(np.float32).T).astype(bfloat16)
    logits = score_ref.astype(np.float32)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    weight_ref = exp / exp.sum(axis=1, keepdims=True)
    output_ref = (weight_ref @ v.astype(np.float32)).astype(bfloat16)
    normalized_ref = layer_norm.reference(output_ref.reshape(queries, head_dim))

    score_npu = scores.numpy().reshape(queries, keys).astype(np.float32)
    weight_npu = weights.numpy().reshape(queries, keys).astype(np.float32)
    output_npu = output.numpy().reshape(queries, head_dim)
    normalized_npu = normalized.numpy().reshape(queries, head_dim)
    score_error = np.max(np.abs(score_npu - logits))
    weight_error = np.max(np.abs(weight_npu - weight_ref))
    output_error = np.max(np.abs(output_npu.astype(np.float32) - output_ref.astype(np.float32)))
    normalized_from_npu_input = layer_norm.reference(output_npu)
    norm_local_error = np.max(
        np.abs(normalized_npu.astype(np.float32) - normalized_from_npu_input.astype(np.float32))
    )
    norm_end_to_end_error = np.max(
        np.abs(normalized_npu.astype(np.float32) - normalized_ref.astype(np.float32))
    )

    np.testing.assert_allclose(score_npu, logits, rtol=0.05, atol=0.02)
    np.testing.assert_allclose(weight_npu, weight_ref, rtol=0.15, atol=0.0002)
    np.testing.assert_allclose(
        output_npu.astype(np.float32), output_ref.astype(np.float32), rtol=0.05, atol=0.005
    )
    np.testing.assert_allclose(
        normalized_npu.astype(np.float32),
        normalized_from_npu_input.astype(np.float32),
        rtol=0.02,
        atol=0.02,
    )
    np.testing.assert_allclose(
        normalized_npu.astype(np.float32), normalized_ref.astype(np.float32),
        rtol=0.1, atol=0.2,
    )
    np.testing.assert_allclose(weight_npu.sum(axis=1), 1, rtol=0.05, atol=0.05)
    print(f"PASS: attention + AIE2 LayerNorm, Q={q.shape}, K=V={k.shape}, output={normalized_npu.shape}")
    print(f"max absolute error: scores={score_error:.6g}, weights={weight_error:.6g}, attention={output_error:.6g}")
    print(f"LayerNorm error: from NPU input={norm_local_error:.6g}, end-to-end={norm_end_to_end_error:.6g}")


if __name__ == "__main__":
    main()
