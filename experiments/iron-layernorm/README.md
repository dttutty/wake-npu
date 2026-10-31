# Phoenix AIE2 LayerNorm

[English](README.md) | [简体中文](README.zh-CN.md)

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
