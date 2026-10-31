"""Fixed-size single-head Transformer block, executed stage by stage on Phoenix."""

import argparse
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import kernels
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "third_party" / "mlir-aie" / "programming_examples"
for directory in (
    EXAMPLES / "basic" / "matrix_multiplication" / "single_core",
    EXAMPLES / "ml" / "softmax",
    EXAMPLES / "ml" / "eltwise_unary",
):
    sys.path.insert(0, str(directory))

from eltwise_unary import eltwise_unary  # noqa: E402
from single_core import single_core  # noqa: E402
from softmax import softmax  # noqa: E402


def load_layer_norm():
    path = ROOT / "experiments" / "iron-layernorm" / "run.py"
    spec = spec_from_file_location("iron_layernorm", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load AIE2 LayerNorm: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def upload(values):
    return iron.tensor(values.reshape(-1), dtype=bfloat16, device="npu")


def empty(count):
    return iron.zeros(count, dtype=bfloat16, device="npu")


def mm(a, b, out, m, k, n, *, b_col_maj=0):
    single_core(
        a, b, out,
        M=m, K=k, N=n, m=32, k=32, n=32,
        dtype_in_str="bf16", dtype_out_str="bf16", b_col_maj=b_col_maj,
    )


def cpu_mm(a, b):
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(bfloat16)


def check(name, actual, expected, *, atol=0.02, rtol=0.05):
    actual32 = actual.astype(np.float32)
    expected32 = expected.astype(np.float32)
    np.testing.assert_allclose(actual32, expected32, atol=atol, rtol=rtol, err_msg=name)
    error = np.max(np.abs(actual32 - expected32))
    print(f"  {name:20s} max abs error {error:.6g}")


def main(seed=2026):
    query_count, key_count, dim, hidden = 64, 1024, 32, 64
    rng = np.random.default_rng(seed)
    memory_host = rng.normal(0, 0.65, (key_count, dim)).astype(bfloat16)
    query_host = memory_host[:query_count].copy()
    weights_host = {
        "q": (rng.normal(0, 0.18, (dim, dim)) / np.sqrt(dim)).astype(bfloat16),
        "k": rng.normal(0, 0.18, (dim, dim)).astype(bfloat16),
        "v": rng.normal(0, 0.18, (dim, dim)).astype(bfloat16),
        "o": rng.normal(0, 0.18, (dim, dim)).astype(bfloat16),
        "up": rng.normal(0, 0.18, (dim, hidden)).astype(bfloat16),
        "down": rng.normal(0, 0.18, (hidden, dim)).astype(bfloat16),
    }
    weight_npu = {name: upload(value) for name, value in weights_host.items()}
    query_npu, memory_npu = upload(query_host), upload(memory_host)
    layer_norm = load_layer_norm()

    q, k, v = empty(query_count * dim), empty(key_count * dim), empty(key_count * dim)
    scores, probs = empty(query_count * key_count), empty(query_count * key_count)
    context, projected = empty(query_count * dim), empty(query_count * dim)
    norm1 = empty(query_count * dim)
    ff1, activated = empty(query_count * hidden), empty(query_count * hidden)
    ff2, norm2 = empty(query_count * dim), empty(query_count * dim)

    mm(query_npu, weight_npu["q"], q, query_count, dim, dim)
    mm(memory_npu, weight_npu["k"], k, key_count, dim, dim)
    mm(memory_npu, weight_npu["v"], v, key_count, dim, dim)
    mm(q, k, scores, query_count, dim, key_count, b_col_maj=1)
    softmax(scores, probs, size=query_count * key_count, num_channels=1)
    mm(probs, v, context, query_count, key_count, dim)
    mm(context, weight_npu["o"], projected, query_count, dim, dim)
    layer_norm.add_layer_norm_aie2(projected, query_npu, norm1)
    mm(norm1, weight_npu["up"], ff1, query_count, dim, hidden)
    eltwise_unary(ff1, activated, size=query_count * hidden, num_channels=1, op="gelu")
    mm(activated, weight_npu["down"], ff2, query_count, hidden, dim)
    layer_norm.add_layer_norm_aie2(ff2, norm1, norm2)

    actual = {
        "q": q.numpy().reshape(query_count, dim),
        "k": k.numpy().reshape(key_count, dim),
        "v": v.numpy().reshape(key_count, dim),
        "scores": scores.numpy().reshape(query_count, key_count),
        "softmax": probs.numpy().reshape(query_count, key_count),
        "context": context.numpy().reshape(query_count, dim),
        "output projection": projected.numpy().reshape(query_count, dim),
        "post-attention norm": norm1.numpy().reshape(query_count, dim),
        "ffn up": ff1.numpy().reshape(query_count, hidden),
        "gelu": activated.numpy().reshape(query_count, hidden),
        "ffn down": ff2.numpy().reshape(query_count, dim),
        "post-ffn norm": norm2.numpy().reshape(query_count, dim),
    }

    print("Stage checks against NumPy, using each stage's actual NPU inputs:")
    check("Q projection", actual["q"], cpu_mm(query_host, weights_host["q"]))
    check("K projection", actual["k"], cpu_mm(memory_host, weights_host["k"]))
    check("V projection", actual["v"], cpu_mm(memory_host, weights_host["v"]))
    check("QK transpose matmul", actual["scores"], cpu_mm(actual["q"], actual["k"].T))
    check(
        "softmax", actual["softmax"], kernels.softmax_ref(actual["scores"]),
        atol=0.0002, rtol=0.15,
    )
    check("attention PV", actual["context"], cpu_mm(actual["softmax"], actual["v"]))
    check("output projection", actual["output projection"], cpu_mm(actual["context"], weights_host["o"]))
    check(
        "post-attention norm", actual["post-attention norm"],
        layer_norm.reference_add(actual["output projection"], query_host),
        atol=0.02, rtol=0.02,
    )
    check("FFN up", actual["ffn up"], cpu_mm(actual["post-attention norm"], weights_host["up"]))
    check(
        "GELU", actual["gelu"], kernels.gelu_ref(actual["ffn up"]),
        atol=0.05, rtol=0.128,
    )
    check("FFN down", actual["ffn down"], cpu_mm(actual["gelu"], weights_host["down"]))
    check(
        "post-FFN norm", actual["post-ffn norm"],
        layer_norm.reference_add(actual["ffn down"], actual["post-attention norm"]),
        atol=0.02, rtol=0.02,
    )
    if not np.isfinite(actual["post-ffn norm"].astype(np.float32)).all():
        raise AssertionError("Final output contains non-finite values")

    reference_q = cpu_mm(query_host, weights_host["q"])
    reference_k = cpu_mm(memory_host, weights_host["k"])
    reference_v = cpu_mm(memory_host, weights_host["v"])
    reference_scores = cpu_mm(reference_q, reference_k.T)
    reference_probs = kernels.softmax_ref(reference_scores)
    reference_context = cpu_mm(reference_probs, reference_v)
    reference_projected = cpu_mm(reference_context, weights_host["o"])
    reference_norm1 = layer_norm.reference_add(reference_projected, query_host)
    reference_ff1 = cpu_mm(reference_norm1, weights_host["up"])
    reference_gelu = kernels.gelu_ref(reference_ff1)
    reference_ff2 = cpu_mm(reference_gelu, weights_host["down"])
    reference_final = layer_norm.reference_add(reference_ff2, reference_norm1)
    final_error = np.max(
        np.abs(actual["post-ffn norm"].astype(np.float32) - reference_final.astype(np.float32))
    )
    np.testing.assert_allclose(
        actual["post-ffn norm"].astype(np.float32), reference_final.astype(np.float32),
        atol=0.06, rtol=0.03, err_msg="full Transformer block",
    )
    print(f"Full-block max abs error vs independent NumPy path: {final_error:.6g}")
    print(f"PASS: 12 NPU stages, output shape {actual['post-ffn norm'].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    main(seed=parser.parse_args().seed)
