# Phoenix 固定尺寸 Transformer 块

[English](README.md) | [简体中文](README.zh-CN.md)

在工作区根目录的 PowerShell 中运行：

```powershell
& .\experiments\iron-vector\run.ps1 -Example block
& .\experiments\iron-vector\run.ps1 -Example block -Seed 7
```

该固定形状实验在 Phoenix/AIE2 NPU 上运行采用 post-LayerNorm 的单头注意力与前馈块。输入为 1024 个 token、每个 32 维的记忆，计算前 64 个 query token 的输出。六个参数矩阵是确定性合成 BF16 权重，不是预训练权重。

12 个 NPU 阶段分别为 Q/K/V 投影、QK 转置矩阵乘法、Softmax、概率矩阵乘 V、输出投影、融合残差加 AIE2 LayerNorm、FFN 上投影、GELU、FFN 下投影，以及第二次融合残差加 AIE2 LayerNorm。K 矩阵在 NPU 内存中保持行主序；MatMul 内核的 B 列主序选项完成转置读取。主机只分配缓冲区、启动阶段，并使用各阶段真实 NPU 输入与 NumPy 校验。12 个阶段内部没有 CPU 回退。

这是数值架构实验，不是可用语言模型。它不分词、不加载训练参数、不计算全部 1024 个 query 位置、不加入位置嵌入、不使用多头，也不生成 token。IRON 主机为每个阶段单独派发 NPU 工作，尚未融合或性能调优。
