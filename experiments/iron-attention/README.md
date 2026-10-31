# Single-head attention and AIE2 LayerNorm on the NPU

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
