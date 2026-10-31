"""Run all six trained Argos zh-en encoder layers on the Phoenix AIE2 NPU."""

import argparse
import os

import numpy as np

from attention import check, prepare_source
from encoder_layer import run_layer


def encode(model, npu_x, pieces, *, report=True, verify=True):
    expected_layers = [f"encoder/layer_{index}/ffn/linear_1/weight" for index in range(6)]
    if any(name not in model.variables for name in expected_layers):
        raise ValueError("Expected six trained encoder layers")

    cpu_x = npu_x.copy() if verify else None
    if report:
        print(f"Running six encoder layers, {len(pieces)} valid tokens")
    for index in range(6):
        npu_x, cpu_x, attention_errors, layer_errors = run_layer(
            model, npu_x, cpu_x, len(pieces), index, verify=verify
        )
        if report and verify:
            print(
                f"Layer {index}: attention={attention_errors[-1]:.6g}, "
                f"norm1={layer_errors[0]:.6g}, up={layer_errors[1]:.6g}, "
                f"SiLU={layer_errors[2]:.6g}, down={layer_errors[3]:.6g}, "
                f"norm2={layer_errors[4]:.6g}, cumulative={layer_errors[5]:.6g}",
                flush=True,
            )

    actual = npu_x[:len(pieces)]
    if verify:
        reference = cpu_x[:len(pieces)]
        maximum = check("six-layer encoder", actual, reference, atol=0.15, rtol=0.05)
        mean = float(np.mean(np.abs(actual.astype(np.float32) - reference.astype(np.float32))))
    if report:
        if verify:
            print(f"Six-layer max abs error={maximum:.6g}, mean abs error={mean:.6g}")
        print(f"PASS: trained six-layer encoder, output shape {actual.shape}")
        print("This is encoder output, not a translation; decoder and token generation are not run")
    return npu_x, cpu_x


def main(text):
    model, x, pieces = prepare_source(text)
    print(f"Input: {text}")
    print(f"SentencePiece: {pieces}")
    npu_x, cpu_x = encode(model, x, pieces)
    return npu_x[:len(pieces)], cpu_x[:len(pieces)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=os.environ.get("NPU_TEST_TEXT", "你好，世界。"))
    main(parser.parse_args().text)
