"""AIE2 FFN epilogue: FP32 accumulator + bias, BF16 rounding, then SiLU."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorTiler2D
from aie.iron import ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron import kernels
from aie.utils.config import aie_runtime_lib_dir, cxx_header_path
from ml_dtypes import bfloat16


ROWS = 64
COLS = 2048


@iron.jit
def bias_silu(accumulators: In, bias: In, activated: Out):
    matrix_ty = np.ndarray[(ROWS, COLS), np.dtype[np.float32]]
    bias_ty = np.ndarray[(COLS,), np.dtype[np.float32]]
    output_ty = np.ndarray[(ROWS, COLS), np.dtype[bfloat16]]
    acc_row_ty = np.ndarray[(COLS,), np.dtype[np.float32]]
    output_row_ty = np.ndarray[(COLS,), np.dtype[bfloat16]]

    acc_fifo = ObjectFifo(acc_row_ty, name="ffn_acc")
    bias_fifo = ObjectFifo(bias_ty, name="ffn_bias")
    output_fifo = ObjectFifo(output_row_ty, name="ffn_activated")
    kernel = ExternalFunction(
        "ffn_bias_silu_aie2",
        source_file=str(Path(__file__).with_suffix(".cc")),
        arg_types=[acc_row_ty, bias_ty, output_row_ty],
        include_dirs=[cxx_header_path(), str(Path(aie_runtime_lib_dir()) / "AIE2")],
        stack_size_override=2048,
    )

    def core(acc_cons, bias_cons, output_prod, fn):
        offsets = bias_cons.acquire(1)
        for _ in range_(ROWS):
            scores = acc_cons.acquire(1)
            result = output_prod.acquire(1)
            fn(scores, offsets, result)
            acc_cons.release(1)
            output_prod.release(1)
        bias_cons.release(1)

    worker = Worker(core, [acc_fifo.cons(), bias_fifo.cons(), output_fifo.prod(), kernel], stack_size=4096)
    row_tap = TensorTiler2D.group_tiler((ROWS, COLS), (1, COLS), (ROWS, 1))[0]

    def sequence(acc_host, bias_host, output_host, acc_prod, bias_prod, output_cons):
        bias_prod.fill(bias_host)
        acc_prod.fill(acc_host, row_tap)
        output_cons.drain(output_host, row_tap, wait=True)

    runtime = Runtime(sequence, [matrix_ty, bias_ty, output_ty,
                                 acc_fifo.prod(), bias_fifo.prod(), output_fifo.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


def main():
    rng = np.random.default_rng(3048)
    accumulators = rng.normal(0, 2, (ROWS, COLS)).astype(np.float32)
    bias = rng.normal(0, 0.5, COLS).astype(np.float32)
    output_npu = iron.zeros(ROWS * COLS, dtype=bfloat16, device="npu")
    bias_silu(iron.tensor(accumulators.reshape(-1), dtype=np.float32, device="npu"),
              iron.tensor(bias, dtype=np.float32, device="npu"), output_npu)
    actual = output_npu.numpy().reshape(ROWS, COLS)
    expected = kernels.silu_ref((accumulators + bias).astype(bfloat16))
    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=0.08, rtol=0.128)
    print(f"PASS: AIE2 FFN bias + SiLU, max error={np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))):.6g}")


if __name__ == "__main__":
    main()
