# IRON vector kernel on the Ryzen AI NPU

Run the custom vector kernel from PowerShell:

```powershell
& .\experiments\iron-vector\run.ps1
```

Run AMD's single-core INT8 matrix multiplication example on the same NPU:

```powershell
& .\experiments\iron-vector\run.ps1 -Example matmul
```

This JIT-compiles `kernel.cc` for one AIE compute tile, transfers two independent
4096-element BF16 inputs to the NPU, executes `z = 5*x + y`, and compares all
results against NumPy. The integer-valued inputs keep the BF16 result exact.

The native Windows setup uses the XRT 2.21.75 SDK, Python 3.13, VS 2022 C++
Build Tools, `llvm-aie`, and `mlir_aie 1.4.4.dev40+gd840cec` in
`third_party/mlir-aie/ironenv`. The generic `iron_setup.py` command currently
installs an older PyPI wheel with an incompatible `Runtime` API; if you rerun
setup, reinstall the wheel from `downloads/iron` before running this. The wheel
SHA-256 is `2121dd73f22a00060ef1b6edb44308615eea53f8dfc81867ca4912d9b27d530c`.

The matrix example checks `C = A @ B` against a NumPy reference and prints
separate NPU and end-to-end timings. Ten repetitions are useful for a smoke
benchmark, not a rigorous throughput measurement.

This is a custom vector workload, not a Transformer or translation model. Its
purpose is to make the NPU toolchain and host-to-tile dataflow inspectable.
For a three-kernel attention pipeline, see `experiments/iron-attention`.
