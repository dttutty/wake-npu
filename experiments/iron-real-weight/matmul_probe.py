"""Repeat BF16 MatMul shapes used by the trained encoder FFN."""

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16

from attention import ROWS, cpu_mm, mm, upload
from run import single_core


def main():
    rng = np.random.default_rng(806)
    for k, n in ((512, 512), (512, 2048), (2048, 512)):
        a = rng.normal(0, 0.8, (ROWS, k)).astype(bfloat16)
        b = rng.normal(0, 0.1, (k, n)).astype(bfloat16)
        expected = cpu_mm(a, b).astype(np.float32)
        for run in range(3):
            actual = mm(upload(a), upload(b), ROWS, k, n).numpy().reshape(ROWS, n).astype(np.float32)
            error = np.abs(actual - expected)
            mismatches = np.count_nonzero(error > 0.1 + 0.05 * np.abs(expected))
            print(f"K={k}, N={n}, run={run + 1}: max error={error.max():.6g}, mismatches={mismatches}/{error.size}")
        if k == 2048:
            output = iron.zeros(ROWS * n, dtype=np.float32, device="npu")
            single_core(
                upload(a), upload(b), output,
                M=ROWS, K=k, N=n, m=32, k=32, n=32,
                dtype_in_str="bf16", dtype_out_str="f32",
            )
            actual = output.numpy().reshape(ROWS, n)
            float_reference = a.astype(np.float32) @ b.astype(np.float32)
            error = np.abs(actual - float_reference)
            print(f"K={k}, N={n}, f32 output: max error={error.max():.6g}, mean error={error.mean():.6g}")
            np.testing.assert_allclose(actual, float_reference, atol=0.1, rtol=0.05)


if __name__ == "__main__":
    main()
