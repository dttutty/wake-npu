"""Run a decoder layer at the first target step against real NPU encoder states."""

import argparse
import json
import os

import aie.iron as iron
import numpy as np
from aie.iron import kernels
from ml_dtypes import bfloat16

from affine_norm512 import reference as norm_reference
from attention import (
    DIM, HEADS, HEAD_DIM, ROWS, check, cpu_attention_reference, cpu_mm,
    cpu_softmax, mm, prepare_source, run_attention, upload,
)
from attention_row0 import reference_row0, run_row0
from device_profile import call, readback
from encoder_layer import apply_norm, eltwise_unary
from encoder_stack import encode
from ffn_bias_silu import bias_silu
from run import PACKAGE, dequantized
from softmax64 import masked_softmax64


def linear_parameters(model, name):
    weight = dequantized(model, name + "/weight").T.astype(bfloat16)
    bias = model.array(name + "/bias").astype(np.float32)
    return weight, bias


def cpu_cross_attention(query, memory, parameters, valid_keys):
    (q_weight, q_bias), (kv_weight, kv_bias), (out_weight, out_bias) = parameters
    q = (cpu_mm(query, q_weight).astype(np.float32) + q_bias).astype(bfloat16)
    kv = (cpu_mm(memory, kv_weight).astype(np.float32) + kv_bias).astype(bfloat16)
    context = np.zeros((ROWS, DIM), dtype=bfloat16)
    for head in range(HEADS):
        offset = head * HEAD_DIM
        head_q = (q[:, offset:offset + HEAD_DIM].astype(np.float32) / np.sqrt(HEAD_DIM)).astype(bfloat16)
        head_k = kv[:, offset:offset + HEAD_DIM]
        head_v = kv[:, DIM + offset:DIM + offset + HEAD_DIM]
        scores = cpu_mm(head_q, head_k.T)
        probabilities = cpu_softmax(scores, valid_keys)
        context[:, offset:offset + HEAD_DIM] = cpu_mm(probabilities, head_v)
    return (cpu_mm(context, out_weight).astype(np.float32) + out_bias).astype(bfloat16)


def run_cross_attention(query, memory, parameters, valid_keys, *, kv_cache=None, verify=True, fused=False):
    (q_weight, q_bias), (kv_weight, kv_bias), (out_weight, out_bias) = parameters
    if q_weight.shape != (DIM, DIM) or kv_weight.shape != (DIM, 2 * DIM):
        raise ValueError("Unexpected decoder cross-attention projection shape")

    q_npu = mm(upload(query), upload(q_weight), ROWS, DIM, DIM, output_dtype=np.float32)
    q = (readback(q_npu).reshape(ROWS, DIM).astype(bfloat16).astype(np.float32) + q_bias).astype(bfloat16)
    if kv_cache is None or kv_cache["projected"] is None:
        kv_npu = mm(upload(memory), upload(kv_weight), ROWS, DIM, 2 * DIM, output_dtype=np.float32)
        kv = (readback(kv_npu).reshape(ROWS, 2 * DIM).astype(bfloat16).astype(np.float32) + kv_bias).astype(bfloat16)
        if kv_cache is not None:
            kv_cache["projected"] = kv.copy()
    else:
        kv = kv_cache["projected"]
    if verify:
        q_error = check("cross Q projection", q, (cpu_mm(query, q_weight).astype(np.float32) + q_bias).astype(bfloat16))
        kv_error = check("cross KV projection", kv, (cpu_mm(memory, kv_weight).astype(np.float32) + kv_bias).astype(bfloat16))

    context = np.zeros((ROWS, DIM), dtype=bfloat16)
    score_error = probability_error = context_error = 0.0
    if fused:
        scaled_q = (q.astype(np.float32) / np.sqrt(HEAD_DIM)).astype(bfloat16)
        keys = kv[:, :DIM]
        values = kv[:, DIM:2 * DIM]
        context = run_row0(scaled_q, keys, values, valid_keys)
        if verify:
            context_error = check("fused cross row-0 context", context[0], reference_row0(scaled_q, keys, values, valid_keys)[0])
    else:
        for head in range(HEADS):
            offset = head * HEAD_DIM
            head_q = (q[:, offset:offset + HEAD_DIM].astype(np.float32) / np.sqrt(HEAD_DIM)).astype(bfloat16)
            head_k = kv[:, offset:offset + HEAD_DIM].copy()
            head_v = kv[:, DIM + offset:DIM + offset + HEAD_DIM].copy()
            scores_npu = mm(upload(head_q), upload(head_k), ROWS, HEAD_DIM, ROWS, b_col_maj=1)
            if verify:
                scores = readback(scores_npu).reshape(ROWS, ROWS)
                score_error = max(score_error, check("cross QK scores", scores, cpu_mm(head_q, head_k.T)))
            probabilities_npu = iron.zeros_like(scores_npu)
            call("masked softmax", masked_softmax64, scores_npu, probabilities_npu, valid_keys=valid_keys)
            if verify:
                probabilities = readback(probabilities_npu).reshape(ROWS, ROWS)
                probability_error = max(probability_error, check(
                    "cross masked softmax", probabilities, cpu_softmax(scores, valid_keys), atol=0.003, rtol=0.03,
                ))
                np.testing.assert_array_equal(probabilities[:, valid_keys:].astype(np.float32), 0)
            context_npu = mm(probabilities_npu, upload(head_v), ROWS, ROWS, HEAD_DIM)
            head_context = readback(context_npu).reshape(ROWS, HEAD_DIM)
            if verify:
                context_error = max(context_error, check("cross PV context", head_context, cpu_mm(probabilities, head_v)))
            context[:, offset:offset + HEAD_DIM] = head_context

    output_npu = mm(upload(context), upload(out_weight), ROWS, DIM, DIM, output_dtype=np.float32)
    output_projection = readback(output_npu).reshape(ROWS, DIM).astype(bfloat16)
    output = (output_projection.astype(np.float32) + out_bias).astype(bfloat16)
    if verify:
        output_error = check("cross output projection", output, (
            cpu_mm(context, out_weight).astype(np.float32) + out_bias
        ).astype(bfloat16))
        local_reference = cpu_cross_attention(query, memory, parameters, valid_keys)
        full_error = check("cross attention", output[:1], local_reference[:1], atol=0.5, rtol=0.03)
        errors = (q_error, kv_error, score_error, probability_error, context_error, output_error, full_error)
    else:
        errors = None
    return output, errors


def decoder_start(model):
    config = json.loads((PACKAGE / "model" / "config.json").read_text(encoding="utf-8"))
    if config["decoder_start_token"] != "<s>" or model.array("decoder/start_from_zero_embedding").item() != 1:
        raise ValueError("Expected the zero-embedding <s> decoder start")
    if model.array("decoder/pre_norm").item() != 0 or model.array("decoder/num_heads").item() != HEADS:
        raise ValueError("Expected an 8-head post-norm decoder")
    if model.array("decoder/activation").item() != 2:
        raise ValueError("Expected Swish/SiLU decoder activation")
    x = np.zeros((ROWS, DIM), dtype=bfloat16)
    x[0] = model.array("decoder/position_encodings/encodings")[0].astype(bfloat16)
    return x


def run_layer(model, x, reference_input, memory, memory_reference, valid_keys, layer_index, *, step=0, caches=None, verify=True, fused=False, row0_norm=False, fused_ffn=False):
    prefix = f"decoder/layer_{layer_index}"
    npu_cache, local_cache, reference_cache, cross_cache = (
        caches if caches is not None else (None, None, None, None)
    )
    self_keys = step + 1 if caches is not None else 1

    self_attention, _, self_parameters, self_errors = run_attention(
        model, x, self_keys, layer_index, scope="decoder",
        cache=npu_cache, reference_cache=local_cache,
        step=step if caches is not None else None, verify=verify, fused=fused,
    )
    self_gamma = model.array(prefix + "/self_attention/layer_norm/gamma").astype(np.float32)
    self_beta = model.array(prefix + "/self_attention/layer_norm/beta").astype(np.float32)
    norm1, norm1_error = apply_norm(self_attention, x, self_gamma, self_beta, verify=verify, row0=row0_norm)

    cross_parameters = tuple(linear_parameters(model, prefix + f"/attention/linear_{i}") for i in range(3))
    cross_attention, cross_errors = run_cross_attention(
        norm1, memory, cross_parameters, valid_keys, kv_cache=cross_cache, verify=verify, fused=fused
    )
    cross_gamma = model.array(prefix + "/attention/layer_norm/gamma").astype(np.float32)
    cross_beta = model.array(prefix + "/attention/layer_norm/beta").astype(np.float32)
    norm2, norm2_error = apply_norm(cross_attention, norm1, cross_gamma, cross_beta, verify=verify, row0=row0_norm)

    up_weight, up_bias = linear_parameters(model, prefix + "/ffn/linear_0")
    up_npu = mm(upload(norm2), upload(up_weight), ROWS, DIM, 2048, output_dtype=np.float32)
    up = ((readback(up_npu).reshape(ROWS, 2048).astype(np.float32) + up_bias).astype(bfloat16)
          if verify or not fused_ffn else None)
    up_error = check("decoder FFN up", up, (
        norm2.astype(np.float32) @ up_weight.astype(np.float32) + up_bias
    ).astype(bfloat16)) if verify else None
    activated_npu = iron.zeros(ROWS * 2048, dtype=bfloat16, device="npu")
    if fused_ffn:
        bias_npu = call("tensor upload", iron.tensor, up_bias, dtype=np.float32, device="npu")
        call("FFN bias + SiLU", bias_silu, up_npu, bias_npu, activated_npu)
    else:
        call("SiLU", eltwise_unary, upload(up), activated_npu, size=ROWS * 2048, num_channels=1, op="silu")
    activated = readback(activated_npu).reshape(ROWS, 2048) if verify or not fused_ffn else None
    activation_error = check("decoder SiLU", activated, kernels.silu_ref(up), atol=0.08, rtol=0.128) if verify else None
    down_weight, down_bias = linear_parameters(model, prefix + "/ffn/linear_1")
    down_npu = mm(activated_npu if fused_ffn else upload(activated),
                  upload(down_weight), ROWS, 2048, DIM, output_dtype=np.float32)
    down = (readback(down_npu).reshape(ROWS, DIM).astype(np.float32) + down_bias).astype(bfloat16)
    down_error = check("decoder FFN down", down, (
        activated.astype(np.float32) @ down_weight.astype(np.float32) + down_bias
    ).astype(bfloat16)) if verify else None
    ffn_gamma = model.array(prefix + "/ffn/layer_norm/gamma").astype(np.float32)
    ffn_beta = model.array(prefix + "/ffn/layer_norm/beta").astype(np.float32)
    output, norm3_error = apply_norm(down, norm2, ffn_gamma, ffn_beta, verify=verify, row0=row0_norm)

    if verify:
        def cpu_layer(input_state, encoder_state, self_cache):
            ref_self = cpu_attention_reference(
                input_state, *self_parameters, self_keys,
                cache=self_cache, step=step if self_cache is not None else None,
            )
            ref_norm1 = norm_reference(ref_self, input_state, self_gamma, self_beta)
            ref_cross = cpu_cross_attention(ref_norm1, encoder_state, cross_parameters, valid_keys)
            ref_norm2 = norm_reference(ref_cross, ref_norm1, cross_gamma, cross_beta)
            ref_up = (ref_norm2.astype(np.float32) @ up_weight.astype(np.float32) + up_bias).astype(bfloat16)
            ref_activated = kernels.silu_ref(ref_up)
            ref_down = (ref_activated.astype(np.float32) @ down_weight.astype(np.float32) + down_bias).astype(bfloat16)
            return norm_reference(ref_down, ref_norm2, ffn_gamma, ffn_beta)

        local_reference = cpu_layer(x, memory, local_cache)
        local_error = check(
            f"decoder layer {layer_index} same-input path", output[:1], local_reference[:1],
            atol=0.2, rtol=0.05,
        )
        reference = cpu_layer(reference_input, memory_reference, reference_cache)
        full_error = float(np.max(np.abs(output[:1].astype(np.float32) - reference[:1].astype(np.float32))))
        errors = (self_errors[-1], cross_errors[-1], norm1_error, norm2_error,
                  up_error, activation_error, down_error, norm3_error, local_error, full_error)
    else:
        reference = None
        errors = None
    if not np.isfinite(output[:1].astype(np.float32)).all():
        raise AssertionError("Non-finite decoder output")
    return output, reference, errors


def main(text):
    model, source, pieces = prepare_source(text)
    memory, memory_reference = encode(model, source, pieces, report=False)
    x = decoder_start(model)
    output, reference, errors = run_layer(model, x, x, memory, memory_reference, len(pieces), 0)
    self_error, cross_error, norm1_error, norm2_error, up_error, activation_error, down_error, norm3_error, local_error, full_error = errors
    check("decoder layer 0 step 0", output[:1], reference[:1], atol=0.2, rtol=0.05)

    print(f"Input: {text}")
    print(f"Source token count: {len(pieces)}; decoder input: <s> at step 0")
    print(f"NPU stage max abs errors: self attention={self_error:.6g}, cross attention={cross_error:.6g}")
    print(f"Norm/FFN errors: {norm1_error:.6g}, {norm2_error:.6g}, {up_error:.6g}, {activation_error:.6g}, {down_error:.6g}, {norm3_error:.6g}")
    print(f"Full decoder layer 0 step 0 max abs error={full_error:.6g}")
    print("PASS: decoder layer 0 with NPU self-attention, cross-attention and FFN")
    print("Five decoder layers, vocabulary projection and token generation remain")
    return output[:1], reference[:1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=os.environ.get("NPU_TEST_TEXT", "你好，世界。"))
    main(parser.parse_args().text)
