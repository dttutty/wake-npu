# Fixed-size Transformer block on Phoenix / Phoenix 固定尺寸 Transformer 块

[简体中文](#zh-cn) | [English](#english)

<a id="zh-cn"></a>

## 简体中文

在工作区根目录的 PowerShell 中运行：

```powershell
& .\experiments\iron-vector\run.ps1 -Example block
& .\experiments\iron-vector\run.ps1 -Example block -Seed 7
```

该固定形状实验在 Phoenix/AIE2 NPU 上运行采用 post-LayerNorm 的单头注意力与前馈块。输入为 1024 个 token、每个 32 维的记忆，计算前 64 个 query token 的输出。六个参数矩阵是确定性合成 BF16 权重，不是预训练权重。

12 个 NPU 阶段分别为 Q/K/V 投影、QK 转置矩阵乘法、Softmax、概率矩阵乘 V、输出投影、融合残差加 AIE2 LayerNorm、FFN 上投影、GELU、FFN 下投影，以及第二次融合残差加 AIE2 LayerNorm。K 矩阵在 NPU 内存中保持行主序；MatMul 内核的 B 列主序选项完成转置读取。主机只分配缓冲区、启动阶段，并使用各阶段真实 NPU 输入与 NumPy 校验。12 个阶段内部没有 CPU 回退。

这是数值架构实验，不是可用语言模型。它不分词、不加载训练参数、不计算全部 1024 个 query 位置、不加入位置嵌入、不使用多头，也不生成 token。IRON 主机为每个阶段单独派发 NPU 工作，尚未融合或性能调优。

---

<a id="english"></a>

## English

Run from the workspace root in PowerShell:

```powershell
& .\experiments\iron-vector\run.ps1 -Example block
& .\experiments\iron-vector\run.ps1 -Example block -Seed 7
```

This fixed-shape experiment runs a single-head post-LayerNorm attention and
feed-forward block on the Phoenix/AIE2 NPU. It takes a 1024-token memory with
32 features and computes outputs for its first 64 query tokens. Six parameter
matrices are deterministic synthetic BF16 weights, not pretrained weights.

The twelve NPU stages are Q/K/V projection, QK-transpose MatMul, Softmax,
probability-times-V MatMul, output projection, fused residual + AIE2 LayerNorm,
FFN up projection, GELU, FFN down projection, and a second fused residual +
AIE2 LayerNorm. The K matrix stays in row-major NPU memory; the MatMul kernel's
column-major B option performs the transposed read. Host code only allocates
buffers, launches stages and checks each result against NumPy using that
stage's actual NPU inputs. There is no CPU fallback inside the twelve stages.

This is a numerical architecture experiment, not a usable language model.
It does not tokenize text, load trained parameters, compute all 1024 query
positions, apply positional embeddings, use multiple heads, or generate tokens.
The IRON host runs each stage as a separate NPU dispatch and is not fused or
performance-tuned.
