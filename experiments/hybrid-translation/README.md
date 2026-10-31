# NPU encoder and GPU decoder experiment

This uses the same installed Argos zh-en Transformer weights as the AIE2
experiments. The encoder can run on the existing IRON NPU path, then hand
its hidden states to an ONNX Runtime DirectML decoder on the Radeon 780M.
The decoder runs all six layers, embeddings, vocabulary projection and
argmax on the GPU. Self-attention K/V and the precomputed cross-attention
K/V remain in GPU OrtValues across target steps. Python controls the token
loop and reads one token ID per step. This is encoder/decoder separation
for a sequence-to-sequence model, rather than decoder-only LLM prefill.

## Run

From the workspace root, using the already installed `amd-npu` Python
environment (ONNX 1.22.0, ONNX Runtime 1.25.2 with DirectML):

```powershell
& .\experiments\hybrid-translation\run.ps1 -Fp16
& .\experiments\hybrid-translation\run.ps1 -Fp16 -Text '你好，世界。' -Repeats 3
& .\experiments\hybrid-translation\run.ps1 -Fp16 -SkipNpu
```

`-Fp16` adds two FP16 GPU modes to the FP32 comparisons. `-Rebuild`
regenerates the exported ONNX graphs and refreshes their FP16 variants.
`-GpuPython` overrides the ONNX/DirectML interpreter. NPU execution uses
the existing Visual Studio and IRON environment setup in a persistent
subprocess. No driver or package changes were needed for this experiment.

`-Length 64` selects separate 64-position graphs and allows up to 64 source
pieces and generated tokens. The default capacity remains 32. `-MaxTokens`
defaults to the selected capacity. `-RecordMismatches` is a diagnostic mode
that records token/numerical differences and continues; the final report
explicitly marks numerical equivalence as failed. Strict rejection remains
the default. See [longer-input results](LONG_INPUTS.md) for the subsequent
19-51 source-token tests and the observed translation differences.

The three graphs are exported from the existing bounded CTranslate2 v6
reader: encoder, cross-K/V preparation, and cached decoder step. Shapes
are fixed at either 32 or 64 source positions (including EOS), with the
same capacity for generated tokens. Padding and future positions are masked. Quantized model weights
are dequantized to FP32 for the reference exports. FP16 converts the GPU
graphs internally, keeping their external tensor interface in FP32.
FP16 is half precision, not INT8/INT4 integer quantization. The NPU encoder
continues to use BF16.

## Measured Results

On this machine, Radeon 780M driver 32.0.31041.1004, the 2026-09-23 run
produced the following warmed encoder-plus-decoder times. ONNX numbers
are medians of two runs after a warmup, with provider profiling enabled.
CTranslate2 is one warmed CPU INT8 run with four threads and beam size 1.

| Input | CPU INT8 CTranslate2 | CPU FP32 ONNX | GPU FP32 | GPU FP16 | NPU encoder + GPU FP16 |
|---|---:|---:|---:|---:|---:|
| 你好，世界。 | 16.8 ms | 65.3 ms | 36.7 ms | 23.3 ms | 2334 ms |
| 人工智能正在改变世界。 | 34.9 ms | 115.2 ms | 56.9 ms | 43.9 ms | 2212 ms |
| 谢谢你。 | 14.0 ms | 51.5 ms | 28.4 ms | 21.1 ms | 2380 ms |

All five ONNX modes reached EOS and matched every CTranslate2 target
piece on these three inputs: `Hello, world.`, `Artificial intelligence is
changing the world.`, and `Thank you.`. Mixed NPU/GPU FP16 decoder hidden
states differed from the FP32 CPU ONNX reference by about 0.7-2.0% relative
L2. This small test set establishes functionality, not general translation
quality. FP16 GPU was faster than FP32 GPU here; optimized CPU INT8 was
still faster than either GPU mode.

The custom NPU encoder is the dominant cost of the hybrid path, about
2.2-2.5 seconds; the GPU decoder takes only tens of milliseconds. These
results therefore do not establish a speed benefit from adding the NPU.
They also do not measure energy use or concurrent multi-sentence throughput.

## Verification and Timing Scope

GPU sessions explicitly disable CPU execution-provider fallback, and the
saved ORT profiles must show DirectML execution without CPU provider
events. Runtime checks verify that K/V buffers remain on `dml`, target IDs
match the FP32 ONNX CPU reference, and aggregate decoder hidden-state
relative L2 error stays below 5%. Full hidden states are read back only
after the timed decoder loop for this numerical check.
When diagnostic runs diverge in their token choices, only the hidden-state
prefix with identical input histories is compared; the full-sequence error
is recorded as null. A close shared-history error does not turn a token
mismatch into a numerical pass.

Reported pipeline timings exclude shared source tokenization/embedding
preparation, text detokenization, initial model/session loading, initial
JIT compilation, warmup and correctness readbacks. Session setup and NPU
worker startup are recorded separately. They must not be compared directly
with old measurements that included launching a Python process or model
loading. CTranslate2 additionally uses its own optimized INT8 kernels,
whereas the ONNX reference is FP32; it is a practical baseline, not an
isolated comparison of CPU versus GPU hardware.

NPU-to-GPU handoff currently goes through an FP32 host array and a local
`.npy` file between persistent Python environments. Its cost, including
file I/O and the GPU upload, is included in hybrid timing. It is one
handoff per sentence, not zero-copy shared GPU/NPU memory. The existing
NPU encoder still performs intermediate host work between its kernels.

Generated models, raw profiles, handoff arrays and JSON reports live in
`artifacts/` and are ignored by Git. `results_fp16.json` contains the
samples behind the table, device-provider counts, numerical errors,
setup times, and per-token timing. This research path leaves the existing
translation implementation and AIE2 decoder experiments available.
64-position graphs are stored separately in `artifacts/length64/`.
