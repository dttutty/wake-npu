# Single-head attention and AIE2 LayerNorm on the NPU / NPU 单头注意力与 AIE2 LayerNorm

[简体中文](#zh-cn) | [English](#english)

<a id="zh-cn"></a>

## 简体中文

在工作区根目录的 PowerShell 中运行：

```powershell
& .\experiments\iron-vector\run.ps1 -Example attention
```

程序运行一个包含 64 个 query、1024 个 key、32 维特征的非因果注意力头。它调用 IRON BF16 矩阵乘法计算 `Q @ K.T`，调用 1024 元素分 tile Softmax，再用 BF16×BF16 矩阵乘法计算 `Softmax(scores) @ V`。BF16 输出随后送入 `experiments/iron-layernorm` 中自定义 Phoenix/AIE2 LayerNorm 内核。四个阶段均通过 NPU JIT/XRT 路径执行。CPU 创建输入张量，并用 NumPy 校验中间和最终结果；在 NPU 执行路径中，这些阶段不由 CPU 代算。

这是注意力核心，并非完整 Transformer 层或预训练语言模型。Q/K/V 是合成输入，不是 token 嵌入的训练投影。它未实现残差连接、完整层级归一化结构、前馈层或 token 生成；上述注意力后的独立 LayerNorm 阶段另行执行。query 缩放在上传前合入 Q。校验允许 BF16 舍入与 NPU Softmax 近似误差。设计从固定版本的本地 `third_party/mlir-aie` 导入 AMD `single_core` 和 `softmax` IRON 示例，不重复复制源码。输出分别报告 LayerNorm 相对实际 NPU 输入的误差及端到端误差，因为归一化会放大小的注意力差异。

在本机 Phoenix NPU 上独立验证过的其他 Transformer 基础模块：

```powershell
& .\experiments\iron-vector\run.ps1 -Example matmul
& .\experiments\iron-vector\run.ps1 -Example gelu
& .\experiments\iron-vector\run.ps1 -Example add
```

实验时的上游 `ml/norm` 主机示例只面向 NPU2/AIE2P，但上游已有 AIE2 BF16 内核。本实验采用独立实现的 AIE2 LayerNorm。完整 Transformer 块还需训练得到的 Q/K/V 和前馈投影、残差、仿射归一化，以及对序列长度和中间张量格式的仔细处理。

---

<a id="english"></a>

## English

Run from the workspace root in PowerShell:

```powershell
& .\experiments\iron-vector\run.ps1 -Example attention
```

The program runs a 64-query, 1024-key, 32-dimensional noncausal attention
head. It invokes IRON's BF16 matrix multiplication for `Q @ K.T`, its
1024-element tile-wise Softmax, then BF16-by-BF16 matrix multiplication for
`Softmax(scores) @ V`. The BF16 output then feeds the custom Phoenix/AIE2
LayerNorm kernel in `experiments/iron-layernorm`. All four stages execute
through the NPU JIT/XRT path. The CPU creates input tensors and checks
intermediate and final results against NumPy; it does not implement these
stages in the NPU execution path.

This is an attention core, not an entire Transformer layer or a pretrained
language model. Q/K/V are synthetic inputs, not learned projections from token
embeddings. Residual connections, normalization, feed-forward layers and token
generation are not included. The query scaling is folded into Q before upload.
The output validation allows BF16 rounding and the NPU Softmax approximation.
The design imports AMD's `single_core` and `softmax` IRON examples from the
pinned local `third_party/mlir-aie` checkout rather than duplicating them.
The output prints LayerNorm's error against its actual NPU input separately
from the end-to-end error; normalization amplifies small attention differences.

Other Transformer building blocks independently verified on this Phoenix NPU:

```powershell
& .\experiments\iron-vector\run.ps1 -Example matmul
& .\experiments\iron-vector\run.ps1 -Example gelu
& .\experiments\iron-vector\run.ps1 -Example add
```

The current upstream `ml/norm` host example is NPU2/AIE2P-only, but upstream
does have an AIE2 BF16 kernel. This experiment uses an independent AIE2
LayerNorm implementation. A complete block would still need learned Q/K/V and
feed-forward projections, residuals, affine normalization and careful
treatment of sequence length and intermediate tensor formats.
