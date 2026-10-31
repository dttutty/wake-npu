"""AIE2 residual plus per-column affine LayerNorm for 64 x 512 BF16."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorTiler2D
from aie.iron import ExternalFunction, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.utils.config import aie_runtime_lib_dir, cxx_header_path
from ml_dtypes import bfloat16


ROWS = 64
COLS = 512
EPSILON = 1e-5


@iron.jit
def add_affine_norm512(packed_input: In, gamma_beta: In, y: Out):
    input_ty = np.ndarray[(ROWS, 2 * COLS), np.dtype[bfloat16]]
    output_ty = np.ndarray[(ROWS, COLS), np.dtype[bfloat16]]
    input_row_ty = np.ndarray[(2 * COLS,), np.dtype[bfloat16]]
    output_row_ty = np.ndarray[(COLS,), np.dtype[bfloat16]]
    parameters_ty = np.ndarray[(2 * COLS,), np.dtype[np.float32]]

    input_fifo = ObjectFifo(input_row_ty, name="input_residual")
    parameters_fifo = ObjectFifo(parameters_ty, name="gamma_beta", depth=1)
    output_fifo = ObjectFifo(output_row_ty, name="output")
    kernel = ExternalFunction(
        "add_affine_norm512_aie2",
        source_file=str(Path(__file__).with_suffix(".cc")),
        arg_types=[input_row_ty, parameters_ty, output_row_ty],
        include_dirs=[cxx_header_path(), str(Path(aie_runtime_lib_dir()) / "AIE2")],
        stack_size_override=2048,
    )

    def core(input_cons, parameters_cons, output_prod, fn):
        params = parameters_cons.acquire(1)
        for _ in range_(ROWS):
            input_row = input_cons.acquire(1)
            output_row = output_prod.acquire(1)
            fn(input_row, params, output_row)
            input_cons.release(1)
            output_prod.release(1)
        parameters_cons.release(1)

    worker = Worker(
        core,
        [input_fifo.cons(), parameters_fifo.cons(), output_fifo.prod(), kernel],
        stack_size=4096,
    )
    input_taps = TensorTiler2D.simple_tiler((ROWS, 2 * COLS), (1, 2 * COLS))
    output_taps = TensorTiler2D.simple_tiler((ROWS, COLS), (1, COLS))
    parameters_tap = TensorTiler2D.simple_tiler((1, 2 * COLS))[0]

    def sequence(input_host, parameters_host, output_host,
                 input_prod, parameters_prod, output_cons):
        parameters_group = TaskGroup()
        parameters_prod.fill(parameters_host, parameters_tap, group=parameters_group)
        for input_tap, output_tap in zip(input_taps, output_taps):
            row_group = TaskGroup()
            input_prod.fill(input_host, input_tap, group=row_group)
            output_cons.drain(output_host, output_tap, wait=True, group=row_group)
            row_group.finish()
        parameters_group.finish()

    runtime = Runtime(
        sequence,
        [input_ty, parameters_ty, output_ty,
         input_fifo.prod(), parameters_fifo.prod(), output_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


def reference(x, residual, gamma, beta):
    values = x.astype(np.float32) + residual.astype(np.float32)
    mean = values.mean(axis=1, keepdims=True)
    variance = np.mean((values - mean) ** 2, axis=1, keepdims=True)
    return (((values - mean) / np.sqrt(variance + EPSILON)) * gamma + beta).astype(bfloat16)


def main():
    rng = np.random.default_rng(2048)
    x = rng.normal(0, 0.8, (ROWS, COLS)).astype(bfloat16)
    residual = rng.normal(0, 0.4, (ROWS, COLS)).astype(bfloat16)
    x[0] = bfloat16(0.25)
    residual[0] = bfloat16(-0.25)
    gamma = rng.normal(1, 0.1, COLS).astype(np.float32)
    beta = rng.normal(0, 0.1, COLS).astype(np.float32)
    parameters = np.concatenate((gamma, beta))

    packed = np.concatenate((x, residual), axis=1)
    packed_npu = iron.tensor(packed.reshape(-1), dtype=bfloat16, device="npu")
    parameters_npu = iron.tensor(parameters, dtype=np.float32, device="npu")
    output_npu = iron.zeros(ROWS * COLS, dtype=bfloat16, device="npu")
    add_affine_norm512(packed_npu, parameters_npu, output_npu)
    actual = output_npu.numpy().reshape(ROWS, COLS)
    expected = reference(x, residual, gamma, beta)
    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=0.03, rtol=0.02)
    error = np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))
    print(f"PASS: AIE2 residual + affine LayerNorm 64x512, max abs error={error:.6g}")


if __name__ == "__main__":
    main()
