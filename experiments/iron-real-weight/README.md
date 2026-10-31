# Real translation-model weight on Phoenix

Run from the workspace root:

```powershell
& .\experiments\iron-vector\run.ps1 -Example real-weight
& .\experiments\iron-vector\run.ps1 -Example real-weight -Text '人工智能正在改变世界。'
& .\experiments\iron-vector\run.ps1 -Example softmax64
& .\experiments\iron-vector\run.ps1 -Example real-attention -Text '人工智能正在改变世界。'
& .\experiments\iron-vector\run.ps1 -Example affine-norm512
& .\experiments\iron-vector\run.ps1 -Example affine-norm-row0
& .\experiments\iron-vector\run.ps1 -Example matmul-shapes
& .\experiments\iron-vector\run.ps1 -Example compact-matmul
& .\experiments\iron-vector\run.ps1 -Example ffn-bias-silu
& .\experiments\iron-vector\run.ps1 -Example real-encoder-layer -Text '人工智能正在改变世界。'
& .\experiments\iron-vector\run.ps1 -Example real-encoder-stack -Text '人工智能正在改变世界。'
& .\experiments\iron-vector\run.ps1 -Example real-decoder-layer -Text '人工智能正在改变世界。'
& .\experiments\iron-vector\run.ps1 -Example real-first-token -Text '人工智能正在改变世界。'
& .\experiments\iron-vector\run.ps1 -Example real-greedy -Text '你好，世界。'
& .\experiments\iron-vector\run.ps1 -Example real-greedy -Text '人工智能正在改变世界。' -MaxTokens 16
& .\experiments\iron-vector\run.ps1 -Example real-greedy -Text '你好，世界。' -NoVerify
& .\experiments\iron-vector\run.ps1 -Example real-greedy -Text '你好，世界。' -NoVerify -Profile
& .\experiments\iron-vector\run.ps1 -Example attention-row0
& .\experiments\iron-vector\run.ps1 -Example real-greedy -Text '你好，世界。' -NoVerify -FusedAttention -Profile
& .\experiments\iron-vector\run.ps1 -Example vocab-top2
& .\experiments\iron-vector\run.ps1 -Example real-greedy -Text '你好，世界。' -NoVerify -FusedAttention -NpuVocab -Profile
& .\experiments\iron-vector\run.ps1 -Example real-greedy -Text '你好，世界。' -NoVerify -FusedAttention -NpuVocab -Row0Norm -Profile
& .\experiments\iron-vector\run.ps1 -Example real-greedy -Text '你好，世界。' -NoVerify -FusedAttention -NpuVocab -Row0Norm -FusedFfn -Profile
```

This reads the already-installed Argos zh-en package without modifying it.
The existing Argos Python environment tokenizes Chinese text; a bounded
read-only CTranslate2 v6 parser loads the package's INT8 embeddings and
encoder layer-0 QKV matrix, including their per-row quantization scales.
The BF16 versions of those real trained weights are multiplied on the NPU
using the IRON single-core MatMul example and checked against NumPy.

The result is only one projection using model-derived embedding vectors. It
does not include position encodings, biases, normalization, attention, the
remaining encoder/decoder layers, or output token generation. It is not yet
a translation engine and does not replace the CPU Argos translator.

`real-attention` goes further. It reads the same model's scaled embedding,
position encodings, QKV matrix and bias, and eight 64-dimensional heads.
The NPU runs QKV MatMul, each head's QK MatMul, a custom 64-column masked
softmax, each PV MatMul, and the output MatMul. The host handles tokenization,
INT8 dequantization, BF16 conversion, bias, query scaling, and head tensor
rearrangement. Each NPU stage is checked against NumPy using that stage's
actual input; a separate full NumPy path checks the final output. This is the
attention core through output projection, before residual/LayerNorm and FFN.
It still cannot generate translated text.

The custom softmax uses max subtraction and an AIE2-compatible approximate
exponential. It is a numerical research kernel, not a performance-optimized
replacement for the CTranslate2 CPU operator.

`real-encoder-layer` completes encoder layer 0: a custom AIE2 residual plus
gamma/beta LayerNorm after attention, trained 512-to-2048 FFN up projection,
NPU Swish/SiLU, trained 2048-to-512 FFN down projection, and a second residual
plus affine LayerNorm. The host still handles tokenization, INT8-to-BF16
conversion, biases, query scaling, and layout/packing. The two FFN MatMuls
use FP32 NPU outputs before adding bias and narrowing to BF16. In particular,
the 2048-deep down projection gives large errors with the example's BF16
output accumulation; `matmul-shapes` reproduces that limitation and checks
the FP32-output correction. Stage checks and an independent CPU full-layer
path verify the numerical result. This computes one encoder layer, not all
six encoder layers or any decoder step, and it cannot translate text yet.

`real-encoder-stack` chains the actual output of each trained encoder layer
into the next, completing all six layers on the NPU. It also carries a separate
six-layer NumPy reference path from the same embeddings and compares the final
valid-token tensor. The tested Chinese inputs `你好，世界。` and
`人工智能正在改变世界。` produced final maximum absolute errors of about 0.035
and 0.049, respectively, with BF16 tensors. Host-side preprocessing, bias,
scaling, and tensor layout operations remain. This example returns encoder
hidden states, not translated text; it does not run the decoder.

`real-decoder-layer` extends that path through decoder layer 0 at target step
0. The model's `start_from_zero_embedding` flag makes the `<s>` input a zero
embedding plus the first position encoding. NPU kernels execute the decoder's
self-attention, cross-attention over the six-layer NPU encoder output, all
three residual/affine LayerNorms, and its Swish FFN. An independent NumPy path
uses separately computed encoder states. The decoder attention projection
MatMuls use FP32 NPU outputs before narrowing to BF16, reducing the tested
first-layer self-attention error from 0.5 to about 0.002. This is still a
numerical experiment rather than a production inference path. Since only the
first target position is used, multi-step causal masking is not tested here.
This example does not run decoder layers 1-5 or vocabulary projection.

`real-first-token` runs all six trained decoder layers at target step 0,
carrying each NPU output into the next layer and independently propagating a
NumPy reference path. The host then applies the tied 65,000-token vocabulary
projection to both final hidden states. For `你好，世界。`,
`人工智能正在改变世界。`, and the 22-token longer test sentence, both paths
selected `▁Hello`, `▁Art`, and `▁Today`, respectively. Final hidden-state
relative L2 errors were about 1.8%, 2.9%, and 1.7%; maximum elementwise
errors were about 0.344, 0.455, and 0.344. The first-token score margin was
at least 0.475 in these three cases. These are raw argmax tokens, not
beam-search translation results. This first-token entrypoint does not run
subsequent target steps or exercise a multi-key causal mask.

`real-greedy` adds autoregressive decoding with separate self-attention K/V
and encoder-memory K/V caches for each of the six decoder layers. Each new
target position uses only previously cached self-attention keys plus its own
key, so the 64-column NPU softmax enforces causality without exposing future
tokens. NPU and independent NumPy paths must agree on the raw argmax at every
step, and the first-row hidden-state relative L2 error must remain below 5%.
The tied 65,000-word projection and Argos target-piece decoding remain on the
CPU. The tested inputs `你好，世界。`, `人工智能正在改变世界。`, and `谢谢你。`
reached EOS and produced `Hello, world.`, `Artificial intelligence is changing
the world.`, and `Thank you.`; all three match the installed CPU CTranslate2
translator's output for those sentences. This is a research greedy decoder,
not the supported Argos beam-search path or a pure-NPU implementation. It has
not been validated over a translation corpus, longer outputs, or arbitrary
language pairs.

Verification is on by default. It recomputes attention, FFN, and full encoder
and decoder paths with NumPy, checks NPU stages, and requires matching target
argmax at every step. `-NoVerify` skips those duplicate CPU computations and
keeps finite-output checks; it uses the same NPU kernels, caches, and CPU
vocabulary projection. Both modes print timing for setup, encoder pipeline,
decoder layers, vocabulary head, and detokenization. The encoder/decoder
figures include Python, weight preparation, CPU layout/bias work, buffer
round trips, and NPU execution; they are not device-only kernel timings.

On this machine with cached IRON builds, `你好，世界。` took about 19.6 s
end-to-end with `-NoVerify`, versus about 4.4 s for `translate.ps1` using
CPU CTranslate2. Within the NPU script, setup took 0.3 s, encoder 2.1 s,
decoder layers 14.1 s, and the CPU vocabulary head 0.02 s. The longer
`人工智能正在改变世界。` case took about 29.8 s inside the NPU script for
10 target tokens (27.1 s in decoder layers), versus about 4.6 s process
time for CPU CTranslate2. These are illustrative local wall times, not a
controlled device benchmark: CTranslate2 uses its optimized translation
path, while this experiment uses fixed 64-row kernels and greedy decoding.
By counting the calls in this implementation, decoder step 0 dispatches
about 354 high-level NPU operations, and later steps about 348 each. Each
of the 12 attention modules per step launches separate QK, softmax, and PV
work for eight heads. This is a code-level estimate, not a hardware counter.
Caching host BF16 weights did not measurably improve the short-sentence wall
time, so that extra cache was removed. The remaining decoder timing does not
separate NPU compute from Python dispatch, uploads, readbacks, and waits.

`-Profile` adds wall-clock accounting around the IRON host API calls in this
pipeline; use it with `-NoVerify` to exclude the duplicate NumPy reference
path. On one cached-build run of `你好，世界。`, the profile produced the same
`Hello, world.` output in 17.99 s inside the script. The 64x64x64 MatMul
entry had 1,056 calls taking 3.96 s in aggregate; masked softmax had 528
calls taking 2.69 s; residual LayerNorm had 102 calls taking 2.25 s.
The instrumented tensor uploads totaled 0.42 s and explicit `.numpy()`
readbacks totaled 0.01 s. Those small readback times do **not** establish
that transfers or synchronization are free: the time for a kernel call can
include submission, execution, and waiting for its output. The profile is
host-observed API wall time, not NPU hardware counters or a decomposition of
device compute versus transfer. Buffer allocation and Python/model work
between instrumented calls are not counted in the API subtotal. It points
to fusing attention work across
heads or reducing small launches as a better next experiment than caching
host-side BF16 weights.

`-FusedAttention` enables an experimental decoder-only AIE2 kernel that
computes the one attention query row consumed at each autoregressive step.
It processes all eight heads in one IRON call, including QK, masked softmax,
and PV; the encoder still uses the original full-row path. A runtime BF16
header carries the valid-key count, so target lengths 1-32 share one compiled
kernel design. The 64-row output projection, residual norms, and FFN still
run on the NPU, with host-side bias/layout work and the CPU vocabulary head.
The original path remains the default. `-FusedAttention` also works with the
default NumPy verification; that checks fused row-0 context, each decoder
layer's first row, and each target argmax. The standalone `attention-row0`
test checks valid-key counts 1, 5, 17, and 32 against a NumPy reference.

With cached builds and `-NoVerify -Profile`, the five-token `你好，世界。`
run produced the same `Hello, world.` target pieces in about 10.9 s inside
the script, versus about 18.0 s for the original per-head path on nearby
runs. The 10-token `人工智能正在改变世界。` run produced the same
`Artificial intelligence is changing the world.` target pieces in about
17.4-18.7 s, versus about 29.8 s for the original path. These are local
wall-time observations, not controlled hardware benchmarks. First-use JIT
compilation can dominate an uncached run; timing comparisons should warm the
kernel first. The experiment remains mixed NPU/CPU and slower than the
optimized CPU CTranslate2 translator on these short sentences.

`-NpuVocab` is the next experimental step toward an all-NPU numerical
path. It keeps 64 BF16 vocabulary-weight tiles (1024 columns each) in
device buffers and runs FP32-output NPU MatMul for all 65,000 logits.
A custom AIE2 kernel adds the corresponding FP32 bias and returns only
the best two score/ID pairs per tile. Each tile writes a cache-line-aligned
view of one shared device buffer; a second AIE2 kernel merges all 64 tile
results on the NPU. Normal decoding reads back only the final token ID and
two scores from this vocabulary path. The CPU still prepares embeddings and
intermediate decoder biases/layouts, and drives the token loop. Verification
mode additionally reads back full logits and checks the final top-2 against
the independent CPU reference; `-NoVerify` does not read back full logits.
The standalone `vocab-top2` test checks both reduction kernels against NumPy.

For `你好，世界。`, a warm five-token `-NoVerify -FusedAttention -NpuVocab`
run produced the same `Hello, world.` pieces in 13.4 s inside the script,
with about 1.54 s spent in the NPU vocabulary stage. The 10-token
`人工智能正在改变世界。` run also preserved every target piece and finished in
22.7 s, with about 3.06 s in the NPU vocabulary stage. By comparison, the
CPU vocabulary stage was around 0.02 s for five tokens, so this is a
functional NPU experiment, **not** a speed optimization. It still is not
pure NPU decoding: tokenization, embeddings, intermediate host arithmetic,
and Python-driven generation remain. These are local wall times, not a
controlled device benchmark; first-use JIT compilation is excluded.

`-Row0Norm` is a decoder-only, opt-in version of the residual affine
LayerNorm. The existing AIE2 C++ kernel already processes one 512-column
row per call; this IRON program invokes it once instead of streaming all
64 rows. The host pads the result back to the normal 64x512 interface, with
only row 0 populated, because autoregressive decoding consumes only that
row. The encoder continues to normalize every source row. The standalone
`affine-norm-row0` test passed against NumPy (maximum absolute error about
0.016), and a full five-token `你好，世界。` run passed the default per-layer
and per-token verification with the switch enabled. With cached builds and
`-NoVerify -FusedAttention -NpuVocab -Row0Norm`, the tested short sentence
took 11.8 s inside the script (7.6 s in decoder layers), versus 13.4 s
without row-0 norm on a nearby run. The 10-token
`人工智能正在改变世界。` case took 20.7 s, versus 22.7 s without it.
These are local observations with run-to-run variation, not guaranteed
speedups or device-only timings. The FFN and output projections still use
fixed 64-row MatMuls even though only row 0 is consumed.

`compact-matmul` probes the stock single-core BF16/FP32 MatMul with 32
rows and 16-row tiles for decoder-shaped projections. All four probed
shapes (512x512, 512x2048, 2048x512, and 512x1536) matched NumPy on
the only populated input row. An attempted 8-row/4-row-tile variant did
not compile: the AIE2 vector MatMul requires a tile height divisible by
16, and the stock output tiler needs two tile rows. We briefly connected
the 32-row path and a 4096-element padded SiLU to the decoder FFN. The
five-token `你好，世界。` run still generated the same text but took about
12.8 s, compared with 11.8 s using the original FFN on a nearby run;
FFN MatMul and SiLU calls remained around 30-40 ms each. Because shrinking
the rows did not measurably help, the decoder integration was removed.
The standalone probe remains to guide a future fused or multi-core FFN.

`-FusedFfn` is an opt-in decoder FFN epilogue on AIE2. The existing FP32
up-projection output goes directly to a custom kernel that adds the FP32
bias, rounds to BF16, and applies the same LUT-based SiLU as the stock
kernel. Its BF16 output feeds the down-projection without a host readback
and re-upload. The encoder FFN and the default decoder path remain unchanged.
`ffn-bias-silu` tests the standalone 64x2048 kernel against NumPy (maximum
absolute error 0.03125). Full five-token verification for `你好，世界。` passed
every decoder layer and target argmax; a 10-token run for
`人工智能正在改变世界。` reached EOS with the expected English sentence. In a warm,
nearby five-token comparison with `-NoVerify -FusedAttention -NpuVocab
-Row0Norm -Profile`, the new path took 12.7 s versus 11.7 s for the original
FFN. The fused kernel's 30 host calls averaged 36.7 ms each, versus 31.0 ms
for the original SiLU calls; the profile counted 60 fewer tensor readbacks
and 30 fewer uploads, but wall time did not improve. These are
host-observed local timings, not NPU hardware counters. Other projection
biases, layouts, embeddings, and token
control still run on the CPU, so this is not a pure-NPU decoder.
