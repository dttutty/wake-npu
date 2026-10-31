# NPU 单头注意力与 AIE2 LayerNorm

[English](README.md) | [简体中文](README.zh-CN.md)

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
