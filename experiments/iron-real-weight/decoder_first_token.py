"""Run six decoder layers on AIE2 and inspect the first raw vocabulary token."""

import argparse
import json
import os

import numpy as np

from attention import prepare_source
from decoder_layer import decoder_start, run_layer
from encoder_stack import encode
from run import PACKAGE, dequantized


def project_vocabulary(model, hidden):
    weight = dequantized(model, "decoder/projection/weight")
    bias = model.array("decoder/projection/bias").astype(np.float32)
    if weight.shape != (65000, 512) or bias.shape != (65000,):
        raise ValueError("Unexpected decoder vocabulary projection shape")
    return weight @ hidden[0].astype(np.float32) + bias


def main(text):
    model, source, pieces = prepare_source(text)
    memory, memory_reference = encode(model, source, pieces, report=False)
    x = decoder_start(model)
    reference = x.copy()
    for index in range(6):
        if f"decoder/layer_{index}/ffn/linear_1/weight" not in model.variables:
            raise ValueError("Expected six decoder layers")
        x, reference, errors = run_layer(
            model, x, reference, memory, memory_reference, len(pieces), index
        )
        difference = x[0].astype(np.float32) - reference[0].astype(np.float32)
        relative_error = float(np.linalg.norm(difference) / np.linalg.norm(reference[0].astype(np.float32)))
        print(
            f"Decoder layer {index}: self={errors[0]:.6g}, cross={errors[1]:.6g}, "
            f"norm1={errors[2]:.6g}, norm2={errors[3]:.6g}, "
            f"up={errors[4]:.6g}, SiLU={errors[5]:.6g}, "
            f"down={errors[6]:.6g}, norm3={errors[7]:.6g}, "
            f"same-input={errors[8]:.6g}, cumulative-max={errors[9]:.6g}, "
            f"relative-L2={relative_error:.6g}",
            flush=True,
        )
        if not np.isfinite(relative_error) or relative_error > 0.05:
            raise AssertionError(f"Decoder layer {index} relative L2 error exceeds 5%")

    logits = project_vocabulary(model, x)
    reference_logits = project_vocabulary(model, reference)
    if not np.isfinite(logits).all() or not np.isfinite(reference_logits).all():
        raise AssertionError("Non-finite decoder logits")
    vocabulary = json.loads((PACKAGE / "model" / "shared_vocabulary.json").read_text(encoding="utf-8"))
    if len(vocabulary) != len(logits):
        raise ValueError("Vocabulary length does not match projection")
    top = np.argsort(logits)[-5:][::-1]
    winner = int(top[0])
    reference_winner = int(np.argmax(reference_logits))
    if winner != reference_winner:
        raise AssertionError(
            f"NPU/CPU first-token argmax mismatch: {vocabulary[winner]!r} vs {vocabulary[reference_winner]!r}"
        )

    print(f"Input: {text}")
    print(f"Source tokens: {len(pieces)}; decoder step: 0")
    print(f"First raw token: {vocabulary[winner]!r} (id {winner})")
    print("Top 5: " + ", ".join(f"{vocabulary[i]!r}={logits[i]:.4f}" for i in top))
    print(f"Top-token score margin: {logits[top[0]] - logits[top[1]]:.6g}")
    print(f"Max logit difference vs CPU path: {np.max(np.abs(logits - reference_logits)):.6g}")
    print("PASS: six NPU decoder layers at step 0; vocabulary projection ran on CPU")
    print("This is a raw first-token result, not a translated sentence or a beam-search output")
    return winner, logits, reference_logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=os.environ.get("NPU_TEST_TEXT", "你好，世界。"))
    main(parser.parse_args().text)
