# Phoenix AIE2 LayerNorm / 层归一化

[简体中文](#zh-cn) | [English](#english)

<a id="zh-cn"></a>

## 简体中文

在工作区根目录的 PowerShell 中运行：

```powershell
& .\experiments\iron-vector\run.ps1 -Example layernorm
```

这是为 Phoenix/AIE2 NPU 编写的 BF16 LayerNorm 内核。单个计算 tile 处理 64 行，每行 32 个特征。每行用两遍 float32 计算均值与中心化方差，epsilon 为 1e-5，gamma=1、beta=0。IRON 主机设计将数据传到 NPU，并与 NumPy 比较，测试包括常数行和近常数行。内核优先考虑正确性和可检查性，而非吞吐率。

同一文件还提供用于 Transformer 残差路径的融合 `add_layer_norm_aie2` 设计：先以 float32 相加 BF16 输入，再计算均值和中心化方差，最终将归一化输出舍入为 BF16。

上游 `programming_examples/ml/norm` 主机驱动仅针对 NPU2，但上游已存在 AIE2 BF16 LayerNorm 核心。本实验为本机 Phoenix 设备提供独立 AIE2 实现和主机设计。

---

<a id="english"></a>

## English

Run from the workspace root in PowerShell:

```powershell
& .\experiments\iron-vector\run.ps1 -Example layernorm
```

This is a custom BF16 LayerNorm kernel for the Phoenix/AIE2 NPU. A single
tile processes 64 rows of 32 features each. Each row uses a two-pass float32
mean and centered variance with epsilon 1e-5; gamma=1 and beta=0. The IRON
host design transfers data to the NPU and checks against NumPy, including a
constant row and a near-constant row. The kernel prioritizes correctness and
inspectability, not throughput.

The same file also provides a fused `add_layer_norm_aie2` design for the
Transformer residual path. It adds BF16 inputs in float32 before computing
mean and centered variance, then rounds the normalized output to BF16.

The upstream `programming_examples/ml/norm` driver is NPU2-specific, but
upstream already contains an AIE2 BF16 LayerNorm core. This experiment is an
independent AIE2 implementation and host design for the local Phoenix device.
