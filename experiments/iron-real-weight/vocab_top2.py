"""Add vocabulary bias and find the best two logits on AIE2."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorTiler2D
from aie.iron import ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.utils.config import cxx_header_path


ROWS = 64
COLS = 1024
CHUNKS = 64
RESULT_FLOATS = 16


@iron.jit
def bias_top2(logits: In, bias: In, top2: Out):
    logits_ty = np.ndarray[(ROWS, COLS), np.dtype[np.float32]]
    row_ty = np.ndarray[(COLS,), np.dtype[np.float32]]
    top2_ty = np.ndarray[(RESULT_FLOATS,), np.dtype[np.float32]]

    logits_fifo = ObjectFifo(row_ty, name="vocab_logits")
    bias_fifo = ObjectFifo(row_ty, name="vocab_bias")
    output_fifo = ObjectFifo(top2_ty, name="vocab_top2")
    kernel = ExternalFunction(
        "vocab_bias_top2_aie2",
        source_file=str(Path(__file__).with_suffix(".cc")),
        arg_types=[row_ty, row_ty, top2_ty],
        include_dirs=[cxx_header_path()],
        stack_size_override=2048,
    )

    def core(logits_cons, bias_cons, output_prod, fn):
        scores = logits_cons.acquire(1)
        offsets = bias_cons.acquire(1)
        result = output_prod.acquire(1)
        fn(scores, offsets, result)
        logits_cons.release(1)
        bias_cons.release(1)
        output_prod.release(1)

    worker = Worker(core, [logits_fifo.cons(), bias_fifo.cons(), output_fifo.prod(), kernel], stack_size=4096)
    row_tap = TensorTiler2D.simple_tiler((ROWS, COLS), (1, COLS))[0]

    def sequence(logits_host, bias_host, output_host, logits_prod, bias_prod, output_cons):
        logits_prod.fill(logits_host, row_tap)
        bias_prod.fill(bias_host)
        output_cons.drain(output_host, wait=True)

    runtime = Runtime(sequence, [logits_ty, row_ty, top2_ty,
                                 logits_fifo.prod(), bias_fifo.prod(), output_fifo.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


@iron.jit
def merge_top2(candidates: In, result: Out):
    candidates_ty = np.ndarray[(CHUNKS, RESULT_FLOATS), np.dtype[np.float32]]
    flat_ty = np.ndarray[(CHUNKS * RESULT_FLOATS,), np.dtype[np.float32]]
    result_ty = np.ndarray[(4,), np.dtype[np.float32]]
    candidates_fifo = ObjectFifo(flat_ty, name="vocab_candidates")
    result_fifo = ObjectFifo(result_ty, name="vocab_result")
    kernel = ExternalFunction(
        "vocab_merge_top2_aie2",
        source_file=str(Path(__file__).with_name("vocab_merge64.cc")),
        arg_types=[flat_ty, result_ty],
        include_dirs=[cxx_header_path()],
        stack_size_override=2048,
    )

    def core(input_cons, output_prod, fn):
        source = input_cons.acquire(1)
        destination = output_prod.acquire(1)
        fn(source, destination)
        input_cons.release(1)
        output_prod.release(1)

    worker = Worker(core, [candidates_fifo.cons(), result_fifo.prod(), kernel], stack_size=4096)

    def sequence(input_host, output_host, input_prod, output_cons):
        input_prod.fill(input_host)
        output_cons.drain(output_host, wait=True)

    runtime = Runtime(sequence, [candidates_ty, result_ty,
                                 candidates_fifo.prod(), result_fifo.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


def main():
    rng = np.random.default_rng(2102)
    logits = rng.normal(0, 3, (ROWS, COLS)).astype(np.float32)
    bias = rng.normal(0, 0.5, COLS).astype(np.float32)
    logits[0, 13] = 10
    logits[0, 859] = 9
    output = iron.zeros(RESULT_FLOATS, dtype=np.float32, device="npu")
    bias_top2(iron.tensor(logits.reshape(-1), dtype=np.float32, device="npu"),
              iron.tensor(bias, dtype=np.float32, device="npu"), output)
    actual = output.numpy()[:4]
    expected = logits[0] + bias
    order = np.argsort(expected)[-2:][::-1]
    assert int(actual[1]) == order[0] and int(actual[3]) == order[1]
    np.testing.assert_allclose(actual[[0, 2]], expected[order], atol=1e-5)
    print(f"PASS: NPU vocabulary bias + top-2, IDs {int(actual[1])}, {int(actual[3])}")

    candidates = np.zeros((CHUNKS, RESULT_FLOATS), dtype=np.float32)
    for tile in range(CHUNKS):
        candidates[tile, :4] = [float(tile), 7.0, float(tile) - 0.25, 8.0]
    merged_npu = iron.zeros(4, dtype=np.float32, device="npu")
    merge_top2(iron.tensor(candidates.reshape(-1), dtype=np.float32, device="npu"), merged_npu)
    np.testing.assert_array_equal(merged_npu.numpy(), [63.0, 63 * COLS + 7, 62.75, 63 * COLS + 8])
    print("PASS: NPU merge of 64 vocabulary tiles")


if __name__ == "__main__":
    main()
