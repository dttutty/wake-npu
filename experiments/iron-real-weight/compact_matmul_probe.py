"""Check smaller-row AIE2 BF16 MatMuls for decoder-only projections."""

from time import perf_counter

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16

from run import single_core


ROWS = 32
TILE_ROWS = 16


def main():
    rng = np.random.default_rng(8016)
    for k, n in ((512, 512), (512, 2048), (2048, 512), (512, 1536)):
        a = np.zeros((ROWS, k), dtype=bfloat16)
        a[0] = rng.normal(0, 0.3, k).astype(bfloat16)
        b = rng.normal(0, 0.2, (k, n)).astype(bfloat16)
        a_npu = iron.tensor(a.reshape(-1), dtype=bfloat16, device="npu")
        b_npu = iron.tensor(b.reshape(-1), dtype=bfloat16, device="npu")
        for attempt in range(2):
            c_npu = iron.zeros(ROWS * n, dtype=np.float32, device="npu")
            started = perf_counter()
            single_core(
                a_npu, b_npu, c_npu,
                M=ROWS, K=k, N=n, m=TILE_ROWS, k=32, n=32,
                dtype_in_str="bf16", dtype_out_str="f32",
            )
            output = c_npu.numpy().reshape(ROWS, n)
            duration = perf_counter() - started
            actual = output[0]
            expected = a[0].astype(np.float32) @ b.astype(np.float32)
            np.testing.assert_allclose(actual, expected, atol=0.1, rtol=0.05)
            np.testing.assert_array_equal(output[1:], 0)
            print(f"PASS: M={ROWS}, K={k}, N={n}, run={attempt + 1}, API call={duration:.3f}s, max error={np.max(np.abs(actual - expected)):.6g}")


if __name__ == "__main__":
    main()
