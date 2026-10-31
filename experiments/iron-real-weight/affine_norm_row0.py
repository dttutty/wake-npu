"""One-row AIE2 residual affine LayerNorm for decoder generation."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.utils.config import aie_runtime_lib_dir, cxx_header_path
from ml_dtypes import bfloat16

from affine_norm512 import reference


DIM = 512


@iron.jit
def add_affine_norm_row0(packed_input: In, gamma_beta: In, y: Out):
    input_ty = np.ndarray[(2 * DIM,), np.dtype[bfloat16]]
    parameters_ty = np.ndarray[(2 * DIM,), np.dtype[np.float32]]
    output_ty = np.ndarray[(DIM,), np.dtype[bfloat16]]

    input_fifo = ObjectFifo(input_ty, name="row0_input_residual")
    parameters_fifo = ObjectFifo(parameters_ty, name="row0_gamma_beta")
    output_fifo = ObjectFifo(output_ty, name="row0_output")
    kernel = ExternalFunction(
        "add_affine_norm512_aie2",
        source_file=str(Path(__file__).with_name("affine_norm512.cc")),
        arg_types=[input_ty, parameters_ty, output_ty],
        include_dirs=[cxx_header_path(), str(Path(aie_runtime_lib_dir()) / "AIE2")],
        stack_size_override=2048,
    )

    def core(input_cons, parameters_cons, output_prod, fn):
        packed = input_cons.acquire(1)
        parameters = parameters_cons.acquire(1)
        output = output_prod.acquire(1)
        fn(packed, parameters, output)
        input_cons.release(1)
        parameters_cons.release(1)
        output_prod.release(1)

    worker = Worker(core, [input_fifo.cons(), parameters_fifo.cons(), output_fifo.prod(), kernel], stack_size=4096)

    def sequence(input_host, parameters_host, output_host, input_prod, parameters_prod, output_cons):
        input_prod.fill(input_host)
        parameters_prod.fill(parameters_host)
        output_cons.drain(output_host, wait=True)

    runtime = Runtime(sequence, [input_ty, parameters_ty, output_ty,
                                 input_fifo.prod(), parameters_fifo.prod(), output_fifo.cons()])
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


def main():
    rng = np.random.default_rng(5120)
    values = rng.normal(0, 0.8, (1, DIM)).astype(bfloat16)
    residual = rng.normal(0, 0.4, (1, DIM)).astype(bfloat16)
    gamma = rng.normal(1, 0.1, DIM).astype(np.float32)
    beta = rng.normal(0, 0.1, DIM).astype(np.float32)
    packed = np.concatenate((values[0], residual[0]))
    parameters = np.concatenate((gamma, beta))
    output_npu = iron.zeros(DIM, dtype=bfloat16, device="npu")
    add_affine_norm_row0(
        iron.tensor(packed, dtype=bfloat16, device="npu"),
        iron.tensor(parameters, dtype=np.float32, device="npu"),
        output_npu,
    )
    actual = output_npu.numpy().astype(np.float32)
    expected = reference(values, residual, gamma, beta)[0].astype(np.float32)
    np.testing.assert_allclose(actual, expected, atol=0.03, rtol=0.02)
    print(f"PASS: AIE2 row-0 residual LayerNorm, max error={np.max(np.abs(actual - expected)):.6g}")


if __name__ == "__main__":
    main()
