# Fixed-size Transformer block on Phoenix

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
