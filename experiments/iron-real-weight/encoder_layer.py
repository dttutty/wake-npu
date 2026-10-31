"""Run a trained Argos zh-en encoder layer on AIE2 in stages."""

import argparse
import os
import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import kernels
from ml_dtypes import bfloat16

from affine_norm512 import add_affine_norm512, reference as norm_reference
from affine_norm_row0 import add_affine_norm_row0
from attention import DIM, ROWS, check, cpu_attention_reference, mm, prepare_source, run_attention, upload
from device_profile import call, readback
from run import dequantized


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "mlir-aie" / "programming_examples" / "ml" / "eltwise_unary"))
from eltwise_unary import eltwise_unary  # noqa: E402


def apply_norm(x, residual, gamma, beta, *, verify=True, row0=False):
    parameters = np.concatenate((gamma, beta)).astype(np.float32)
    parameters_npu = call("tensor upload", iron.tensor, parameters, dtype=np.float32, device="npu")
    if row0:
        packed = np.concatenate((x[0], residual[0]))
        result = iron.zeros(DIM, dtype=bfloat16, device="npu")
        call("residual layer norm row0", add_affine_norm_row0, upload(packed), parameters_npu, result)
        actual = np.zeros((ROWS, DIM), dtype=bfloat16)
        actual[0] = readback(result)
    else:
        packed = np.concatenate((x, residual), axis=1)
        result = iron.zeros(ROWS * DIM, dtype=bfloat16, device="npu")
        call("residual layer norm", add_affine_norm512, upload(packed), parameters_npu, result)
        actual = readback(result).reshape(ROWS, DIM).copy()
    error = (
        check("residual affine LayerNorm", actual[:1] if row0 else actual,
              norm_reference(x[:1], residual[:1], gamma, beta) if row0 else norm_reference(x, residual, gamma, beta),
              atol=0.06, rtol=0.03)
        if verify else None
    )
    return actual, error


def run_layer(model, x, reference_input, valid_tokens, layer_index, *, verify=True):
    if model.array("encoder/activation").item() != 2:
        raise ValueError("This encoder layer requires Swish/SiLU activation")
    prefix = f"encoder/layer_{layer_index}"
    attention, _, attention_parameters, attention_errors = run_attention(
        model, x, valid_tokens, layer_index, verify=verify
    )

    attention_gamma = model.array(prefix + "/self_attention/layer_norm/gamma").astype(np.float32)
    attention_beta = model.array(prefix + "/self_attention/layer_norm/beta").astype(np.float32)
    norm1, norm1_error = apply_norm(attention, x, attention_gamma, attention_beta, verify=verify)

    up_weight = dequantized(model, prefix + "/ffn/linear_0/weight").T.astype(bfloat16)
    up_bias = model.array(prefix + "/ffn/linear_0/bias").astype(np.float32)
    if up_weight.shape != (DIM, 2048):
        raise ValueError("Unexpected FFN up-projection shape")
    up_npu = mm(upload(norm1), upload(up_weight), ROWS, DIM, 2048, output_dtype=np.float32)
    up = (readback(up_npu).reshape(ROWS, 2048).astype(np.float32) + up_bias).astype(bfloat16)
    up_error = check("FFN up", up, (
        norm1.astype(np.float32) @ up_weight.astype(np.float32) + up_bias
    ).astype(bfloat16)) if verify else None

    activated_npu = iron.zeros(ROWS * 2048, dtype=bfloat16, device="npu")
    call("SiLU", eltwise_unary, upload(up), activated_npu, size=ROWS * 2048, num_channels=1, op="silu")
    activated = readback(activated_npu).reshape(ROWS, 2048)
    activation_error = check("Swish/SiLU", activated, kernels.silu_ref(up), atol=0.08, rtol=0.128) if verify else None

    down_weight = dequantized(model, prefix + "/ffn/linear_1/weight").T.astype(bfloat16)
    down_bias = model.array(prefix + "/ffn/linear_1/bias").astype(np.float32)
    if down_weight.shape != (2048, DIM):
        raise ValueError("Unexpected FFN down-projection shape")
    down_npu = mm(upload(activated), upload(down_weight), ROWS, 2048, DIM, output_dtype=np.float32)
    down = (readback(down_npu).reshape(ROWS, DIM).astype(np.float32) + down_bias).astype(bfloat16)
    down_error = check("FFN down", down, (
        activated.astype(np.float32) @ down_weight.astype(np.float32) + down_bias
    ).astype(bfloat16)) if verify else None

    ffn_gamma = model.array(prefix + "/ffn/layer_norm/gamma").astype(np.float32)
    ffn_beta = model.array(prefix + "/ffn/layer_norm/beta").astype(np.float32)
    final, norm2_error = apply_norm(down, norm1, ffn_gamma, ffn_beta, verify=verify)

    if verify:
        independent_attention = cpu_attention_reference(reference_input, *attention_parameters, valid_tokens)
        independent_norm1 = norm_reference(independent_attention, reference_input, attention_gamma, attention_beta)
        independent_up = (independent_norm1.astype(np.float32) @ up_weight.astype(np.float32) + up_bias).astype(bfloat16)
        independent_activation = kernels.silu_ref(independent_up)
        independent_down = (independent_activation.astype(np.float32) @ down_weight.astype(np.float32) + down_bias).astype(bfloat16)
        independent_final = norm_reference(independent_down, independent_norm1, ffn_gamma, ffn_beta)
        full_error = float(np.max(np.abs(
            final[:valid_tokens].astype(np.float32) - independent_final[:valid_tokens].astype(np.float32)
        )))
        errors = (norm1_error, up_error, activation_error, down_error, norm2_error, full_error)
    else:
        independent_final = None
        errors = None
    if not np.isfinite(final[:valid_tokens].astype(np.float32)).all():
        raise AssertionError("Non-finite encoder output")

    return final, independent_final, attention_errors, errors


def main(text):
    model, x, pieces = prepare_source(text)
    final, independent_final, _, errors = run_layer(model, x, x, len(pieces), 0)
    norm1_error, up_error, activation_error, down_error, norm2_error, full_error = errors
    check("full encoder layer", final[:len(pieces)], independent_final[:len(pieces)], atol=0.15, rtol=0.05)

    print(f"Input: {text}")
    print(f"SentencePiece: {pieces}")
    print(f"NPU stage max abs errors: attention norm={norm1_error:.6g}, FFN up={up_error:.6g}, SiLU={activation_error:.6g}, FFN down={down_error:.6g}, FFN norm={norm2_error:.6g}")
    print(f"End-to-end max abs error vs independent NumPy first-layer path: {full_error:.6g}")
    print(f"PASS: trained encoder layer 0 with NPU kernels, output shape {final[:len(pieces)].shape}")
    print("Host handles tokenization, weight conversion, biases, query scaling and tensor rearrangement; layers 1-5 and decoder are not run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=os.environ.get("NPU_TEST_TEXT", "你好，世界。"))
    main(parser.parse_args().text)
