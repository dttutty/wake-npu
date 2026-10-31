"""Fused AIE2 attention for the one decoder query row used at each step."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorTiler2D
from aie.iron import ExternalFunction, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.utils.config import cxx_header_path
from ml_dtypes import bfloat16


HEADS = 8
HEAD_DIM = 64
MAX_KEYS = 32
HEAD_INPUT = 2 + HEAD_DIM + 2 * MAX_KEYS * HEAD_DIM


@iron.jit
def attention_row0(packed: In, context: Out):
    input_ty = np.ndarray[(HEADS, HEAD_INPUT), np.dtype[bfloat16]]
    output_ty = np.ndarray[(HEADS, HEAD_DIM), np.dtype[bfloat16]]
    input_head_ty = np.ndarray[(HEAD_INPUT,), np.dtype[bfloat16]]
    output_head_ty = np.ndarray[(HEAD_DIM,), np.dtype[bfloat16]]

    input_fifo = ObjectFifo(input_head_ty, name="attention_row0_input")
    output_fifo = ObjectFifo(output_head_ty, name="attention_row0_output")
    kernel = ExternalFunction(
        "attention_row0_aie2",
        source_file=str(Path(__file__).with_suffix(".cc")),
        arg_types=[input_head_ty, output_head_ty],
        include_dirs=[cxx_header_path()],
        stack_size_override=2048,
    )

    def core(input_cons, output_prod, fn):
        for _ in range_(HEADS):
            source = input_cons.acquire(1)
            destination = output_prod.acquire(1)
            fn(source, destination)
            input_cons.release(1)
            output_prod.release(1)

    worker = Worker(core, [input_fifo.cons(), output_fifo.prod(), kernel], stack_size=4096)
    input_taps = TensorTiler2D.simple_tiler((HEADS, HEAD_INPUT), (1, HEAD_INPUT))
    output_taps = TensorTiler2D.simple_tiler((HEADS, HEAD_DIM), (1, HEAD_DIM))

    def sequence(input_host, output_host, input_prod, output_cons):
        for input_tap, output_tap in zip(input_taps, output_taps):
            group = TaskGroup()
            input_prod.fill(input_host, input_tap, group=group)
            output_cons.drain(output_host, output_tap, wait=True, group=group)
            group.finish()

    runtime = Runtime(sequence, [input_ty, output_ty, input_fifo.prod(), output_fifo.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


def pack_heads(queries, keys, values, valid_keys):
    """Pack BF16 row-0 queries and at most 32 K/V rows for one NPU call."""
    if queries.shape != (64, 512) or keys.shape != (64, 512) or values.shape != (64, 512):
        raise ValueError("Expected 64x512 attention tensors")
    if not 1 <= valid_keys <= MAX_KEYS:
        raise ValueError(f"valid_keys must be in [1, {MAX_KEYS}]")
    packed = np.empty((HEADS, HEAD_INPUT), dtype=bfloat16)
    for head in range(HEADS):
        columns = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
        packed[head, 0] = bfloat16(valid_keys)
        packed[head, 1] = bfloat16(0)
        packed[head, 2:2 + HEAD_DIM] = queries[0, columns]
        packed[head, 2 + HEAD_DIM:2 + HEAD_DIM + MAX_KEYS * HEAD_DIM] = keys[:MAX_KEYS, columns].reshape(-1)
        packed[head, 2 + HEAD_DIM + MAX_KEYS * HEAD_DIM:] = values[:MAX_KEYS, columns].reshape(-1)
    return packed


def run_row0(queries, keys, values, valid_keys):
    from device_profile import call, readback

    packed = pack_heads(queries, keys, values, valid_keys)
    source = call("tensor upload", iron.tensor, packed.reshape(-1), dtype=bfloat16, device="npu")
    output = iron.zeros(HEADS * HEAD_DIM, dtype=bfloat16, device="npu")
    call("fused attention row0", attention_row0, source, output)
    context = np.zeros((64, HEADS * HEAD_DIM), dtype=bfloat16)
    context[0] = readback(output).reshape(HEADS * HEAD_DIM)
    return context


def reference_row0(queries, keys, values, valid_keys):
    context = np.zeros((64, HEADS * HEAD_DIM), dtype=bfloat16)
    for head in range(HEADS):
        columns = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
        q = queries[0, columns].astype(np.float32)
        k = keys[:valid_keys, columns].astype(np.float32)
        v = values[:valid_keys, columns].astype(np.float32)
        scores = (k @ q).astype(bfloat16).astype(np.float32)
        probabilities = np.exp(scores - scores.max())
        probabilities = (probabilities / probabilities.sum()).astype(bfloat16).astype(np.float32)
        context[0, columns] = (probabilities @ v).astype(bfloat16)
    return context


def main():
    rng = np.random.default_rng(1064)
    queries = rng.normal(0, 0.4, (64, 512)).astype(bfloat16)
    keys = rng.normal(0, 0.4, (64, 512)).astype(bfloat16)
    values = rng.normal(0, 0.4, (64, 512)).astype(bfloat16)
    for valid_keys in (1, 5, 17, 32):
        packed = pack_heads(queries, keys, values, valid_keys)
        input_npu = iron.tensor(packed.reshape(-1), dtype=bfloat16, device="npu")
        output_npu = iron.zeros(HEADS * HEAD_DIM, dtype=bfloat16, device="npu")
        attention_row0(input_npu, output_npu)
        actual = output_npu.numpy().reshape(HEADS, HEAD_DIM).astype(np.float32)
        expected = reference_row0(queries, keys, values, valid_keys)[0].reshape(HEADS, HEAD_DIM).astype(np.float32)
        np.testing.assert_allclose(actual, expected, atol=0.02, rtol=0.04)
        print(f"PASS: fused row-0 attention, valid_keys={valid_keys}, max error={np.max(np.abs(actual - expected)):.6g}")


if __name__ == "__main__":
    main()
