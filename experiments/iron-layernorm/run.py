"""Phoenix/AIE2 BF16 LayerNorm: custom device kernel and IRON host design."""

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.utils.config import aie_runtime_lib_dir, cxx_header_path
from ml_dtypes import bfloat16


ROWS = 64
COLS = 32
EPSILON = 1e-5


@iron.jit
def layer_norm_aie2(x: In, y: Out):
    tensor_type = np.ndarray[(ROWS * COLS,), np.dtype[bfloat16]]
    input_fifo = ObjectFifo(tensor_type, name="input")
    output_fifo = ObjectFifo(tensor_type, name="output")
    kernel = ExternalFunction(
        "layer_norm_aie2",
        source_file=str(Path(__file__).parent / "kernel.cc"),
        arg_types=[tensor_type, tensor_type],
        include_dirs=[cxx_header_path(), str(Path(aie_runtime_lib_dir()) / "AIE2")],
    )

    def core(source, destination, fn):
        input_tile = source.acquire(1)
        output_tile = destination.acquire(1)
        fn(input_tile, output_tile)
        source.release(1)
        destination.release(1)

    worker = Worker(
        core, fn_args=[input_fifo.cons(), output_fifo.prod(), kernel], stack_size=2048
    )

    def sequence(input_host, output_host, input_prod, output_cons):
        input_prod.fill(input_host)
        output_cons.drain(output_host, wait=True)

    runtime = Runtime(
        sequence,
        [tensor_type, tensor_type, input_fifo.prod(), output_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


@iron.jit
def add_layer_norm_aie2(x: In, residual: In, y: Out):
    tensor_type = np.ndarray[(ROWS * COLS,), np.dtype[bfloat16]]
    input_fifo = ObjectFifo(tensor_type, name="input")
    residual_fifo = ObjectFifo(tensor_type, name="residual")
    output_fifo = ObjectFifo(tensor_type, name="output")
    kernel = ExternalFunction(
        "add_layer_norm_aie2",
        source_file=str(Path(__file__).parent / "kernel.cc"),
        arg_types=[tensor_type, tensor_type, tensor_type],
        include_dirs=[cxx_header_path(), str(Path(aie_runtime_lib_dir()) / "AIE2")],
    )

    def core(source, skip, destination, fn):
        input_tile = source.acquire(1)
        skip_tile = skip.acquire(1)
        output_tile = destination.acquire(1)
        fn(input_tile, skip_tile, output_tile)
        source.release(1)
        skip.release(1)
        destination.release(1)

    worker = Worker(
        core,
        fn_args=[input_fifo.cons(), residual_fifo.cons(), output_fifo.prod(), kernel],
        stack_size=2048,
    )

    def sequence(input_host, residual_host, output_host, input_prod, skip_prod, output_cons):
        input_prod.fill(input_host)
        skip_prod.fill(residual_host)
        output_cons.drain(output_host, wait=True)

    runtime = Runtime(
        sequence,
        [
            tensor_type, tensor_type, tensor_type,
            input_fifo.prod(), residual_fifo.prod(), output_fifo.cons(),
        ],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


def reference(x):
    values = x.astype(np.float32)
    mean = values.mean(axis=1, keepdims=True)
    variance = np.mean((values - mean) ** 2, axis=1, keepdims=True)
    return ((values - mean) / np.sqrt(variance + EPSILON)).astype(bfloat16)


def reference_add(x, residual):
    values = x.astype(np.float32) + residual.astype(np.float32)
    mean = values.mean(axis=1, keepdims=True)
    variance = np.mean((values - mean) ** 2, axis=1, keepdims=True)
    return ((values - mean) / np.sqrt(variance + EPSILON)).astype(bfloat16)


def main():
    rng = np.random.default_rng(2026)
    host_input = rng.normal(0, 0.7, (ROWS, COLS)).astype(bfloat16)
    host_input[0] = bfloat16(0.5)
    host_input[1] = (7.0 + np.arange(COLS, dtype=np.float32) / 32).astype(bfloat16)

    input_npu = iron.tensor(host_input.reshape(-1), dtype=bfloat16, device="npu")
    output_npu = iron.zeros_like(input_npu)
    layer_norm_aie2(input_npu, output_npu)

    actual = output_npu.numpy().reshape(ROWS, COLS)
    expected = reference(host_input)
    np.testing.assert_allclose(
        actual.astype(np.float32), expected.astype(np.float32), rtol=0.02, atol=0.02
    )
    np.testing.assert_array_equal(actual[0], np.zeros(COLS, dtype=bfloat16))
    max_error = np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))
    print(f"PASS: AIE2 LayerNorm, shape={host_input.shape}, max abs error={max_error:.6g}")

    host_residual = rng.normal(0, 0.4, (ROWS, COLS)).astype(bfloat16)
    residual_npu = iron.tensor(host_residual.reshape(-1), dtype=bfloat16, device="npu")
    added_npu = iron.zeros_like(input_npu)
    add_layer_norm_aie2(input_npu, residual_npu, added_npu)
    added_actual = added_npu.numpy().reshape(ROWS, COLS)
    added_expected = reference_add(host_input, host_residual)
    np.testing.assert_allclose(
        added_actual.astype(np.float32), added_expected.astype(np.float32),
        rtol=0.02, atol=0.02,
    )
    added_error = np.max(
        np.abs(added_actual.astype(np.float32) - added_expected.astype(np.float32))
    )
    print(f"PASS: AIE2 residual + LayerNorm, max abs error={added_error:.6g}")


if __name__ == "__main__":
    main()
