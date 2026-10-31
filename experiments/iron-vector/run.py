from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.utils.config import cxx_header_path
from ml_dtypes import bfloat16


@iron.jit
def fused_vector(x: In, y: In, z: Out, *, count: CompileTime[int]):
    vector_type = np.ndarray[(count,), np.dtype[bfloat16]]
    x_fifo = ObjectFifo(vector_type, name="x")
    y_fifo = ObjectFifo(vector_type, name="y")
    z_fifo = ObjectFifo(vector_type, name="z")

    kernel = ExternalFunction(
        "fused_vector",
        source_file=str(Path(__file__).parent / "kernel.cc"),
        arg_types=[vector_type, vector_type, vector_type],
        include_dirs=[cxx_header_path()],
    )

    def core(x_in, y_in, z_out, fn):
        x_tile = x_in.acquire(1)
        y_tile = y_in.acquire(1)
        z_tile = z_out.acquire(1)
        fn(x_tile, y_tile, z_tile)
        x_in.release(1)
        y_in.release(1)
        z_out.release(1)

    worker = Worker(core, fn_args=[x_fifo.cons(), y_fifo.cons(), z_fifo.prod(), kernel])

    def sequence(x_host, y_host, z_host, x_prod, y_prod, z_cons):
        x_prod.fill(x_host)
        y_prod.fill(y_host)
        z_cons.drain(z_host, wait=True)

    runtime = Runtime(
        sequence,
        [vector_type, vector_type, vector_type, x_fifo.prod(), y_fifo.prod(), z_fifo.cons()],
    )
    return Program(iron.get_current_device(), runtime, workers=[worker]).resolve_program()


def main():
    count = 4096
    rng = np.random.default_rng(42)
    x_host = rng.integers(-16, 17, size=count).astype(bfloat16)
    y_host = rng.integers(-8, 9, size=count).astype(bfloat16)

    x = iron.tensor(x_host, dtype=bfloat16, device="npu")
    y = iron.tensor(y_host, dtype=bfloat16, device="npu")
    z = iron.zeros_like(x)
    fused_vector(x, y, z, count=count)

    actual = z.numpy()
    expected = (5 * x_host + y_host).astype(bfloat16)
    np.testing.assert_array_equal(actual, expected)
    print(f"PASS: {count} BF16 elements, z = 5*x + y")
    print(f"first 8: x={x_host[:8]}, y={y_host[:8]}, z={actual[:8]}")


if __name__ == "__main__":
    main()
