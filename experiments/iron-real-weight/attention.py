"""Run trained Argos zh-en encoder multi-head attention on AIE2."""

import argparse
import json
import os

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16

from run import PACKAGE, dequantized, single_core, token_pieces
from ct2_weights import ModelWeights
from attention_row0 import reference_row0, run_row0
from device_profile import call, readback
from softmax64 import masked_softmax64


ROWS = 64
DIM = 512
HEADS = 8
HEAD_DIM = DIM // HEADS


def upload(values):
    return call("tensor upload", iron.tensor, values.reshape(-1), dtype=bfloat16, device="npu")


def mm(a, b, m, k, n, *, b_col_maj=0, output_dtype=bfloat16):
    output = iron.zeros(m * n, dtype=output_dtype, device="npu")
    call(f"matmul {m}x{k}x{n}", single_core,
        a, b, output, M=m, K=k, N=n, m=32, k=32, n=32,
        dtype_in_str="bf16",
        dtype_out_str="f32" if output_dtype == np.float32 else "bf16",
        b_col_maj=b_col_maj,
    )
    return output


def check(name, actual, expected, *, atol=0.1, rtol=0.05):
    actual32 = actual.astype(np.float32)
    expected32 = expected.astype(np.float32)
    np.testing.assert_allclose(actual32, expected32, atol=atol, rtol=rtol, err_msg=name)
    return float(np.max(np.abs(actual32 - expected32)))


def cpu_mm(a, b):
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(bfloat16)


def cpu_softmax(scores, valid_keys):
    logits = scores[:, :valid_keys].astype(np.float32)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    result = np.zeros((ROWS, ROWS), dtype=bfloat16)
    result[:, :valid_keys] = (exp / exp.sum(axis=1, keepdims=True)).astype(bfloat16)
    return result


def apply_cached_keys(projected, cache, step):
    if cache is None:
        return projected
    if step is None or not 0 <= step < ROWS or step + 1 > ROWS:
        raise ValueError("Invalid decoder cache step")
    cache["keys"][step] = projected[0, DIM:2 * DIM]
    cache["values"][step] = projected[0, 2 * DIM:3 * DIM]
    projected[:, DIM:2 * DIM] = cache["keys"]
    projected[:, 2 * DIM:3 * DIM] = cache["values"]
    return projected


def cpu_attention_reference(x, qkv_weight, qkv_bias, output_weight, output_bias, valid_keys, *, cache=None, step=None):
    projected = (cpu_mm(x, qkv_weight).astype(np.float32) + qkv_bias).astype(bfloat16)
    projected = apply_cached_keys(projected, cache, step)
    context = np.zeros((ROWS, DIM), dtype=bfloat16)
    for head in range(HEADS):
        offset = head * HEAD_DIM
        q = (projected[:, offset:offset + HEAD_DIM].astype(np.float32) / np.sqrt(HEAD_DIM)).astype(bfloat16)
        k = projected[:, DIM + offset:DIM + offset + HEAD_DIM]
        v = projected[:, 2 * DIM + offset:2 * DIM + offset + HEAD_DIM]
        scores = cpu_mm(q, k.T)
        probabilities = cpu_softmax(scores, valid_keys)
        context[:, offset:offset + HEAD_DIM] = cpu_mm(probabilities, v)
    return (cpu_mm(context, output_weight).astype(np.float32) + output_bias).astype(bfloat16)


def prepare_source(text):
    model = ModelWeights(PACKAGE / "model" / "model.bin")
    if model.model_type != "TransformerSpec" or model.revision != 7:
        raise ValueError("Unexpected CTranslate2 model type or revision")
    if model.array("encoder/num_heads").item() != HEADS:
        raise ValueError("This experiment requires 8 attention heads")
    if model.array("encoder/pre_norm").item() != 0:
        raise ValueError("This experiment requires post-norm attention")
    if model.array("encoder/scale_embeddings").item() != 1:
        raise ValueError("This experiment requires scaled embeddings")

    vocabulary = json.loads((PACKAGE / "model" / "shared_vocabulary.json").read_text(encoding="utf-8"))
    token_to_id = {piece: index for index, piece in enumerate(vocabulary)}
    pieces = token_pieces(text) + ["</s>"]
    if len(pieces) > 32:
        raise ValueError(f"Input has {len(pieces)} pieces; maximum is 32")
    ids = [token_to_id.get(piece, token_to_id["<unk>"]) for piece in pieces]
    embeddings = dequantized(model, "encoder/embeddings_0/weight")
    positions = model.array("encoder/position_encodings/encodings")
    if positions.shape[1] != DIM:
        raise ValueError("Unexpected position shape")

    x = np.zeros((ROWS, DIM), dtype=bfloat16)
    x[:len(ids)] = (embeddings[ids] * np.sqrt(DIM) + positions[:len(ids)]).astype(bfloat16)
    return model, x, pieces


def run_attention(model, x, valid_keys, layer_index, *, scope="encoder", cache=None, reference_cache=None, step=None, verify=True, fused=False):
    if fused and scope != "decoder":
        raise ValueError("Fused row-0 attention is decoder-only")
    if verify and (cache is None) != (reference_cache is None):
        raise ValueError("NPU and reference caches must be supplied together")
    if cache is not None and (step is None or valid_keys != step + 1):
        raise ValueError("Decoder valid key count must equal step + 1")
    prefix = f"{scope}/layer_{layer_index}/self_attention"
    qkv_weight = dequantized(model, prefix + "/linear_0/weight").T.astype(bfloat16)
    qkv_bias = model.array(prefix + "/linear_0/bias").astype(np.float32)
    output_weight = dequantized(model, prefix + "/linear_1/weight").T.astype(bfloat16)
    output_bias = model.array(prefix + "/linear_1/bias").astype(np.float32)
    if qkv_weight.shape != (DIM, 3 * DIM) or output_weight.shape != (DIM, DIM):
        raise ValueError(f"Unexpected attention weight shape in layer {layer_index}")

    projection_dtype = np.float32 if scope == "decoder" else bfloat16
    qkv_npu = mm(upload(x), upload(qkv_weight), ROWS, DIM, 3 * DIM, output_dtype=projection_dtype)
    qkv = readback(qkv_npu).reshape(ROWS, 3 * DIM).astype(bfloat16)
    qkv_error = check("QKV projection", qkv, cpu_mm(x, qkv_weight)) if verify else 0.0

    qkv_biased = (qkv.astype(np.float32) + qkv_bias).astype(bfloat16)
    qkv_biased = apply_cached_keys(qkv_biased, cache, step)
    context = np.zeros((ROWS, DIM), dtype=bfloat16)
    score_error = probability_error = context_error = 0.0
    if fused:
        q = (qkv_biased[:, :DIM].astype(np.float32) / np.sqrt(HEAD_DIM)).astype(bfloat16)
        keys = qkv_biased[:, DIM:2 * DIM]
        values = qkv_biased[:, 2 * DIM:3 * DIM]
        context = run_row0(q, keys, values, valid_keys)
        if verify:
            context_error = check("fused row-0 context", context[0], reference_row0(q, keys, values, valid_keys)[0])
    else:
        for head in range(HEADS):
            offset = head * HEAD_DIM
            q = (qkv_biased[:, offset:offset + HEAD_DIM].astype(np.float32) / np.sqrt(HEAD_DIM)).astype(bfloat16)
            k = qkv_biased[:, DIM + offset:DIM + offset + HEAD_DIM].copy()
            v = qkv_biased[:, 2 * DIM + offset:2 * DIM + offset + HEAD_DIM].copy()

            scores_npu = mm(upload(q), upload(k), ROWS, HEAD_DIM, ROWS, b_col_maj=1)
            if verify:
                scores = readback(scores_npu).reshape(ROWS, ROWS)
                score_error = max(score_error, check("QK scores", scores, cpu_mm(q, k.T)))

            probabilities_npu = iron.zeros_like(scores_npu)
            call("masked softmax", masked_softmax64, scores_npu, probabilities_npu, valid_keys=valid_keys)
            if verify:
                probabilities = readback(probabilities_npu).reshape(ROWS, ROWS)
                probability_error = max(
                    probability_error,
                    check("masked softmax", probabilities, cpu_softmax(scores, valid_keys), atol=0.003, rtol=0.03),
                )
                np.testing.assert_array_equal(probabilities[:, valid_keys:].astype(np.float32), 0)

            context_npu = mm(probabilities_npu, upload(v), ROWS, ROWS, HEAD_DIM)
            head_context = readback(context_npu).reshape(ROWS, HEAD_DIM)
            if verify:
                context_error = max(context_error, check("PV context", head_context, cpu_mm(probabilities, v)))
            context[:, offset:offset + HEAD_DIM] = head_context

    output_npu = mm(upload(context), upload(output_weight), ROWS, DIM, DIM, output_dtype=projection_dtype)
    output_projection = readback(output_npu).reshape(ROWS, DIM).astype(bfloat16)
    output = (output_projection.astype(np.float32) + output_bias).astype(bfloat16)
    active_rows = 1 if cache is not None else valid_keys
    if verify:
        output_ref = (cpu_mm(context, output_weight).astype(np.float32) + output_bias).astype(bfloat16)
        output_error = check("output projection", output, output_ref)
        independent_reference = cpu_attention_reference(
            x, qkv_weight, qkv_bias, output_weight, output_bias, valid_keys,
            cache=reference_cache, step=step,
        )
        end_to_end_error = check(
            "full attention sublayer", output[:active_rows], independent_reference[:active_rows],
            atol=0.5, rtol=0.03,
        )
        errors = (qkv_error, score_error, probability_error, context_error, output_error, end_to_end_error)
    else:
        independent_reference = None
        errors = None

    if not np.isfinite(output[:active_rows].astype(np.float32)).all():
        raise AssertionError("Non-finite attention output")
    parameters = (qkv_weight, qkv_bias, output_weight, output_bias)
    return output, independent_reference, parameters, errors


def main(text, *, report=True):
    model, x, pieces = prepare_source(text)
    output, independent_reference, _, errors = run_attention(model, x, len(pieces), 0)
    if report:
        qkv_error, score_error, probability_error, context_error, output_error, end_to_end_error = errors
        print(f"Input: {text}")
        print(f"SentencePiece: {pieces}")
        print(f"Token count: {len(pieces)}; heads: {HEADS}; head dimension: {HEAD_DIM}")
        print(f"NPU stage max abs errors: QKV={qkv_error:.6g}, QK={score_error:.6g}, softmax={probability_error:.6g}, PV={context_error:.6g}, output={output_error:.6g}")
        print(f"End-to-end max abs error vs independent NumPy path: {end_to_end_error:.6g}")
        print(f"PASS: real encoder layer-0 MHA through output projection, shape {output[:len(pieces)].shape}")
        print("Host applies biases, scaling and head rearrangement; residual, normalization, FFN, remaining layers and decoding are not run")
    return model, x, output, independent_reference, pieces


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=os.environ.get("NPU_TEST_TEXT", "你好，世界。"))
    main(parser.parse_args().text)
