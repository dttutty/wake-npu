# IRON vector kernel on the Ryzen AI NPU / Ryzen AI NPU 上的 IRON 向量内核

[简体中文](#zh-cn) | [English](#english)

<a id="zh-cn"></a>

## 简体中文

在 PowerShell 中运行自定义向量内核：

```powershell
& .\experiments\iron-vector\run.ps1
```

在同一 NPU 上运行 AMD 的单核 INT8 矩阵乘法示例：

```powershell
& .\experiments\iron-vector\run.ps1 -Example matmul
```

程序为一个 AIE 计算 tile 即时编译 `kernel.cc`，将两个独立的 4096 元素 BF16 输入传到 NPU，执行 `z = 5*x + y`，并将全部结果与 NumPy 比较。使用整数值输入，使 BF16 结果保持精确。

原生 Windows 环境使用 XRT 2.21.75 SDK、Python 3.13、VS 2022 C++ Build Tools、`llvm-aie`，以及位于 `third_party/mlir-aie/ironenv` 的 `mlir_aie 1.4.4.dev40+gd840cec`。实验时通用 `iron_setup.py` 命令会安装较旧的 PyPI wheel，其 `Runtime` API 不兼容；若重新运行安装流程，须先从 `downloads/iron` 重装所记录的 wheel。该 wheel 的 SHA-256 为 `2121dd73f22a00060ef1b6edb44308615eea53f8dfc81867ca4912d9b27d530c`。

矩阵示例用 NumPy 参考结果检查 `C = A @ B`，分别打印 NPU 与端到端计时。重复十次适合作为冒烟基准，不构成严格吞吐率测量。

这是自定义向量工作负载，不是 Transformer 或翻译模型，目的在于让 NPU 工具链和主机到 tile 的数据流可检查。三个内核组成的注意力流水线见 `experiments/iron-attention`。

---

<a id="english"></a>

## English

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
