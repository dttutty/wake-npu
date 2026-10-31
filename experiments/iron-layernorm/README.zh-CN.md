# Phoenix AIE2 层归一化

[English](README.md) | [简体中文](README.zh-CN.md)

在工作区根目录的 PowerShell 中运行：

```powershell
& .\experiments\iron-vector\run.ps1 -Example layernorm
```

这是为 Phoenix/AIE2 NPU 编写的 BF16 LayerNorm 内核。单个计算 tile 处理 64 行，每行 32 个特征。每行用两遍 float32 计算均值与中心化方差，epsilon 为 1e-5，gamma=1、beta=0。IRON 主机设计将数据传到 NPU，并与 NumPy 比较，测试包括常数行和近常数行。内核优先考虑正确性和可检查性，而非吞吐率。

同一文件还提供用于 Transformer 残差路径的融合 `add_layer_norm_aie2` 设计：先以 float32 相加 BF16 输入，再计算均值和中心化方差，最终将归一化输出舍入为 BF16。

上游 `programming_examples/ml/norm` 主机驱动仅针对 NPU2，但上游已存在 AIE2 BF16 LayerNorm 核心。本实验为本机 Phoenix 设备提供独立 AIE2 实现和主机设计。
