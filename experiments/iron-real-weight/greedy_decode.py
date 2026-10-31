"""Generate target tokens with six AIE2 decoder layers and a CPU vocabulary head."""

import argparse
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from ml_dtypes import bfloat16

from attention import DIM, ROWS, prepare_source
from decoder_layer import decoder_start, run_layer
from device_profile import profiling
from encoder_stack import encode
from run import PACKAGE, ROOT, dequantized
from vocab_head import NpuVocabHead


def empty_cache():
    return {
        "keys": np.zeros((ROWS, DIM), dtype=bfloat16),
        "values": np.zeros((ROWS, DIM), dtype=bfloat16),
    }


def decode_pieces(pieces):
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [str(python), str(Path(__file__).with_name("argos_decode.py")),
         str(PACKAGE / "sentencepiece.model")],
        input=json.dumps(pieces), capture_output=True,
        text=True, encoding="utf-8", env=env, check=True,
    )
    return result.stdout.rstrip("\r\n")


def main(text, max_tokens=16, *, verify=True, profile=False, fused_attention=False, npu_vocab=False, row0_norm=False, fused_ffn=False):
    with profiling(profile) as measurements:
        result = _run(text, max_tokens, verify=verify, fused_attention=fused_attention,
                      npu_vocab=npu_vocab, row0_norm=row0_norm, fused_ffn=fused_ffn)
    if measurements is not None:
        measurements.report()
    return result


def _run(text, max_tokens, *, verify, fused_attention, npu_vocab, row0_norm, fused_ffn):
    if not 1 <= max_tokens <= 32:
        raise ValueError("max_tokens must be in [1, 32]")
    started = perf_counter()
    model, source, pieces = prepare_source(text)
    vocabulary = json.loads((PACKAGE / "model" / "shared_vocabulary.json").read_text(encoding="utf-8"))
    projection = dequantized(model, "decoder/projection/weight")
    bias = model.array("decoder/projection/bias").astype(np.float32)
    if projection.shape != (len(vocabulary), DIM) or bias.shape != (len(vocabulary),):
        raise ValueError("Unexpected decoder projection shape")
    positions = model.array("decoder/position_encodings/encodings")
    if model.array("decoder/scale_embeddings").item() != 1:
        raise ValueError("Expected scaled decoder embeddings")
    eos_id = vocabulary.index("</s>")
    vocab_head = NpuVocabHead(projection, bias) if npu_vocab else None
    setup_end = perf_counter()
    memory, memory_reference = encode(model, source, pieces, report=False, verify=verify)
    encoder_end = perf_counter()
    caches = [
        (empty_cache(), empty_cache() if verify else None,
         empty_cache() if verify else None, {"projected": None})
        for _ in range(6)
    ]
    generated = []
    layer_seconds = head_seconds = 0.0

    print(f"Input: {text}")
    attention_mode = "fused decoder row-0" if fused_attention else "per-head full-row"
    head_mode = "NPU projection + bias/top-2 + merge" if npu_vocab else "CPU projection / argmax"
    print(f"Source tokens: {len(pieces)}; max generated tokens: {max_tokens}; verify: {verify}; attention: {attention_mode}; vocab: {head_mode}; norm: {'row-0' if row0_norm else 'full-row'}; FFN: {'NPU bias + SiLU' if fused_ffn else 'host bias + NPU SiLU'}", flush=True)
    for step in range(max_tokens):
        if step == 0:
            x = decoder_start(model)
        else:
            x = np.zeros((ROWS, DIM), dtype=bfloat16)
            x[0] = (projection[generated[-1]] * np.sqrt(DIM) + positions[step]).astype(bfloat16)
        reference = x.copy() if verify else None

        layer_started = perf_counter()
        for layer_index in range(6):
            x, reference, errors = run_layer(
                model, x, reference, memory, memory_reference, len(pieces), layer_index,
                step=step, caches=caches[layer_index], verify=verify, fused=fused_attention,
                row0_norm=row0_norm, fused_ffn=fused_ffn,
            )
            if verify:
                difference = x[0].astype(np.float32) - reference[0].astype(np.float32)
                relative_error = float(np.linalg.norm(difference) / np.linalg.norm(reference[0].astype(np.float32)))
                if not np.isfinite(relative_error) or relative_error > 0.05:
                    raise AssertionError(
                        f"step {step}, layer {layer_index}: relative L2 error {relative_error:.4g} exceeds 5%"
                    )
        layer_seconds += perf_counter() - layer_started

        head_started = perf_counter()
        if vocab_head is not None:
            token_id, best_score, second_score, logits = vocab_head.top2(x[0], capture_logits=verify)
            if not np.isfinite((best_score, second_score)).all():
                raise AssertionError("Non-finite NPU decoder logits")
        else:
            logits = projection @ x[0].astype(np.float32) + bias
            if not np.isfinite(logits).all():
                raise AssertionError("Non-finite decoder logits")
            token_id = int(np.argmax(logits))
            top = np.argsort(logits)[-2:][::-1]
            best_score, second_score = logits[top]
        if verify:
            reference_logits = projection @ reference[0].astype(np.float32) + bias
            if not np.isfinite(reference_logits).all():
                raise AssertionError("Non-finite reference logits")
            reference_id = int(np.argmax(reference_logits))
            if token_id != reference_id:
                raise AssertionError(
                    f"step {step}: NPU/CPU argmax differs: {vocabulary[token_id]!r} vs {vocabulary[reference_id]!r}"
                )
        generated.append(token_id)
        head_seconds += perf_counter() - head_started
        line = (
            f"step {step}: {vocabulary[token_id]!r} (id {token_id}), "
            f"margin={best_score - second_score:.4g}"
        )
        if verify:
            line += (
                f", hidden relative-L2={relative_error:.4g}, "
                f"max logit delta={np.max(np.abs(logits - reference_logits)):.4g}"
            )
        print(line, flush=True)
        if token_id == eos_id:
            break

    generation_end = perf_counter()
    target_pieces = [vocabulary[token_id] for token_id in generated]
    raw_text = decode_pieces([piece for piece in target_pieces if piece != "</s>"])
    finished = perf_counter()
    print(f"Target pieces: {target_pieces}")
    print(f"Greedy text: {raw_text}")
    print(f"Stopped at EOS: {generated[-1] == eos_id}")
    print("NPU: encoder, six decoder layers" + (", decoder FFN up bias + SiLU" if fused_ffn else "") +
          (", vocabulary projection, bias/top-2 and merge" if npu_vocab else "") +
          "; CPU: tokenization, embeddings, remaining intermediate biases/layouts" +
          (", final token readback" if npu_vocab else ", vocabulary projection/argmax"))
    print("Research greedy decode, not a CTranslate2 beam-search translation")
    print(
        f"Timing (s): setup={setup_end - started:.3f}, "
        f"encoder pipeline={encoder_end - setup_end:.3f}, "
        f"decoder layers={layer_seconds:.3f}, {'NPU' if npu_vocab else 'CPU'} vocab head={head_seconds:.3f}, "
        f"other generation={generation_end - encoder_end - layer_seconds - head_seconds:.3f}, "
        f"detokenize={finished - generation_end:.3f}, total={finished - started:.3f}"
    )
    return raw_text, target_pieces


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=os.environ.get("NPU_TEST_TEXT", "你好，世界。"))
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Time IRON host API calls")
    parser.add_argument("--fused-attention", action="store_true", help="Experimental decoder row-0 attention kernel")
    parser.add_argument("--npu-vocab", action="store_true", help="Experimental chunked NPU vocabulary projection")
    parser.add_argument("--row0-norm", action="store_true", help="Run decoder residual LayerNorm on row 0 only")
    parser.add_argument("--fused-ffn", action="store_true", help="Run decoder FFN up bias + SiLU on NPU")
    arguments = parser.parse_args()
    main(arguments.text, arguments.max_tokens, verify=not arguments.no_verify,
         profile=arguments.profile, fused_attention=arguments.fused_attention,
         npu_vocab=arguments.npu_vocab, row0_norm=arguments.row0_norm,
         fused_ffn=arguments.fused_ffn)
