"""AIE2 BF16 row softmax with a short-sequence padding mask."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.utils.config import cxx_header_path
from ml_dtypes import bfloat16


ROWS = 64
COLS = 64


@iron.jit
def masked_softmax64(scores: In, probabilities: Out, *, valid_keys: CompileTime[int]):
    if not 1 <= valid_keys <= COLS:
        raise ValueError(f"valid_keys must be in [1, {COLS}]")
    tensor_type = np.ndarray[(ROWS * COLS,), np.dtype[bfloat16]]
    input_fifo = ObjectFifo(tensor_type, name="scores")
    output_fifo = ObjectFifo(tensor_type, name="probabilities")
    kernel = ExternalFunction(
        "masked_softmax64_aie2",
        source_file=str(Path(__file__).with_suffix(".cc")),
        arg_types=[tensor_type, tensor_type, np.int32],
        include_dirs=[cxx_header_path()],
        stack_size_override=2048,
    )

    def core(source, destination, fn):
        input_tile = source.acquire(1)
        output_tile = destination.acquire(1)
        fn(input_tile, output_tile, valid_keys)
        source.release(1)
        destination.release(1)

    worker = Worker(
        core, fn_args=[input_fifo.cons(), output_fifo.prod(), kernel], stack_size=4096
    )

    def sequence(input_host, output_host, input_prod, output_cons):
        input_prod.fill(input_host)
        output_cons.drain(output_host, wait=True)

    runtime = Runtime(
        sequence,
        [tensor_type, tensor_type, input_fifo.prod(), output_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


def main():
    rng = np.random.default_rng(1203)
    scores = rng.normal(0, 2, (ROWS, COLS)).astype(bfloat16)
    scores[0, :8] = bfloat16(3)
    scores[1, :] = bfloat16(100)
    scores[2, :] = np.linspace(-30, 30, COLS, dtype=np.float32).astype(bfloat16)
    scores_npu = iron.tensor(scores.reshape(-1), dtype=bfloat16, device="npu")
    for valid_keys in (1, 8, 31, 64):
        output_npu = iron.zeros_like(scores_npu)
        masked_softmax64(scores_npu, output_npu, valid_keys=valid_keys)
        actual = output_npu.numpy().reshape(ROWS, COLS).astype(np.float32)
        logits = scores[:, :valid_keys].astype(np.float32)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        expected = exp / exp.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(actual[:, :valid_keys], expected, atol=0.002, rtol=0.025)
        np.testing.assert_array_equal(actual[:, valid_keys:], 0)
        print(f"PASS: masked AIE2 softmax, valid_keys={valid_keys}, max error={np.max(np.abs(actual[:, :valid_keys] - expected)):.6g}")


if __name__ == "__main__":
    main()
