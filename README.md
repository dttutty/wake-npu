# WakeNPU

[English](README.md) | [简体中文](README.zh-CN.md)

**Put your idle NPU to work.**

**Validated on an AMD Ryzen 7 8845H PC with its built-in AIE2 NPU and Radeon 780M.**

Experiments in programming the AMD Ryzen AI NPU already inside a Windows PC:
from hardware probes and custom AIE2 kernels to a trained Transformer translator,
with CPU and Radeon 780M GPU baselines.

This is a research prototype for Phoenix/Hawk Point AIE2 hardware, not a
general-purpose accelerator or a one-click application. Running Transformer
math on the NPU is possible in these experiments, but the measured custom NPU
and hybrid translation paths were slower than the practical CPU/GPU baselines.
Performance and compatibility statements below describe the recorded local
experiments, primarily from September 2026, rather than all AMD NPUs.

## Verified Hardware

The custom AIE2 kernels, trained Transformer translation experiments and
CPU/GPU comparisons were run on this physical Windows machine, not only
compiled or simulated.

| Component | Tested platform |
| --- | --- |
| Processor | AMD Ryzen 7 8845H, as reported by Windows |
| NPU | Built-in Hawk Point AIE2 / Ryzen AI NPU |
| GPU | AMD Radeon 780M Graphics |
| Host | Windows; recorded SDK and driver versions in [REPRODUCING.md](REPRODUCING.md) |

Both the Windows processor registry entry and `Win32_Processor` report
`AMD Ryzen 7 8845H w/ Radeon 780M Graphics` (checked on 2026-10-31).
An 8845HS configuration has not been independently verified here. The validated
scope includes the successful custom IRON/XRT path; the archived Ryzen AI 1.2
path still crashed. Successful execution does not imply a speed advantage or
compatibility with every Ryzen AI NPU.

## Explore the research

| Experiment | What it establishes |
| --- | --- |
| [Hardware probes](#direct-ironxrt-transformer-experiments) | Driver-level XRT probing and hardware workload validation |
| [Legacy Ryzen AI 1.2](experiments/amd-1.2-phx/README.md) | A compiler-recognized MatMul path that still crashed at execution; no successful NPU output |
| [Custom AIE2 kernels](experiments/iron-vector/README.md) | Vector arithmetic and matrix multiplication with numerical checks |
| [Attention](experiments/iron-attention/README.md) and [Transformer block](experiments/iron-transformer-block/README.md) | Synthetic neural-network building blocks on the NPU |
| [Trained Transformer](experiments/iron-real-weight/README.md) | Argos zh-en weights, six-layer encoder/decoder, greedy generation and experimental NPU vocabulary projection |
| [NPU encoder + GPU decoder](experiments/hybrid-translation/README.md) | DirectML decoder with GPU-resident K/V caches; measured overhead and numerical differences |
| [Longer-input measurements](experiments/hybrid-translation/LONG_INPUTS.md) | 19-51 source-token comparisons, including cases where hybrid output diverged |
| [English speech recognition](experiments/speech-english-gpu/README.md) | GPU Whisper and an exploratory Windows Live Captions comparison; no NPU speech implementation |

## Reproducing the experiments

Start with [the environment and reproduction notes](REPRODUCING.md). Model
weights, vendor runtimes, drivers, compiler installations and build caches are
not included. The setup is currently manual and specific to the tested stack.
See [third-party sources](THIRD_PARTY.md) for upstream projects and data credits.

The checked-in speech transcript snapshots can be rescored with standard
Python, without a GPU, NPU, model download or vendor SDK:

```powershell
python .\experiments\speech-english-gpu\score_asr.py
```

## Offline Chinese/English translation

`translate.py` uses local Argos/CTranslate2 models on the CPU. It is a practical
translator, **not** an NPU translation demo: the current Ryzen AI compatibility
table does not provide a supported Phoenix/Hawk Point NLP/LLM inference path.
The custom IRON/XRT research path below runs the model's Transformer core on
this NPU while retaining host-side tokenization and vocabulary projection.

```powershell
.\translate.ps1 --install
.\translate.ps1 --from zh --to en "你好，世界！"
.\translate.ps1 --from en --to zh "Hello, how are you today?"
```

The language models are stored in `translation-data/` and are used offline after
installation. The backend is printed on each inference to avoid silent fallback.

## Direct IRON/XRT Transformer experiments

The [NPU encoder + GPU decoder experiment](experiments/hybrid-translation/README.md)
now runs the same model through a DirectML decoder with GPU-resident K/V
caches. It compares CPU FP32, GPU FP32/FP16, and NPU-encoder/GPU-decoder
paths against a CPU INT8 greedy baseline. All three test sentences matched,
but the current NPU encoder dominates hybrid latency; the hybrid path was
slower than pure GPU and CPU INT8 on these short inputs.

```powershell
& .\experiments\hybrid-translation\run.ps1 -Fp16
```

The [fixed-size Transformer block](experiments/iron-transformer-block/README.md)
runs 12 NPU stages with synthetic parameters. The
[real-weight projection](experiments/iron-real-weight/README.md) reads the
installed Argos zh-en model and executes its first encoder QKV projection
with trained INT8 weights converted to BF16 for the NPU. The `real-attention`
experiment continues through all eight heads and the first attention output
projection. `real-encoder-layer` continues through residual/affine LayerNorm
and the 2048-wide Swish FFN, completing the first encoder layer. The host
still performs preparation, bias and layout work. `real-encoder-stack` now
chains all six trained encoder layers on the NPU and checks them against an
independent CPU numerical path. `real-first-token` takes that encoder output
through all six decoder layers at the first target position, then uses a CPU
vocabulary projection to inspect the raw highest-scoring English token.
`real-greedy` adds per-layer self-attention and encoder-memory K/V caches and
continues until `</s>` or the target-token limit. It has produced complete translations for
three tested Chinese sentences, matching the installed CPU translator, but
remains an experimental mixed NPU/CPU path rather than a supported translation
backend.

Use `-NoVerify` with `real-greedy` to skip the duplicate NumPy reference path
and print stage timings. Add `-Profile` for selected IRON host-call counts and
wall times; these are not device-only kernel measurements. Even then, this
prototype was slower than the CPU translator on the tested short sentences;
the repeated small kernel launches
and host/device synchronization remain.
An opt-in `-FusedAttention` path computes only decoder attention row 0 for
all eight heads in one AIE2 call; it reduced warmed local wall time for the
tested sentences but still did not beat CPU CTranslate2.
`-NpuVocab` moves the 65,000-word projection, per-tile bias/top-2, and
final 64-tile candidate merge to the NPU. CPU tokenization, embedding
preparation, intermediate biases/layouts, and generation control remain.
It is a functional step toward NPU-only model math, not a speed improvement.
`-Row0Norm` additionally runs each decoder residual LayerNorm on just the
one output row used at each token, leaving the default full-row path intact.
`-FusedFfn` moves the decoder FFN up-projection bias and SiLU into one AIE2
kernel and keeps that intermediate on the device. It passed the tested
translations but was slower than the default FFN in a nearby warm run.

```powershell
& .\experiments\iron-vector\run.ps1 -Example block
& .\experiments\iron-vector\run.ps1 -Example real-weight -Text '人工智能正在改变世界。'
& .\experiments\iron-vector\run.ps1 -Example real-attention -Text '人工智能正在改变世界。'
& .\experiments\iron-vector\run.ps1 -Example real-encoder-layer -Text '人工智能正在改变世界。'
& .\experiments\iron-vector\run.ps1 -Example real-encoder-stack -Text '人工智能正在改变世界。'
& .\experiments\iron-vector\run.ps1 -Example real-decoder-layer -Text '人工智能正在改变世界。'
& .\experiments\iron-vector\run.ps1 -Example real-first-token -Text '人工智能正在改变世界。'
& .\experiments\iron-vector\run.ps1 -Example real-greedy -Text '你好，世界。'
& .\experiments\iron-vector\run.ps1 -Example real-greedy -Text '你好，世界。' -NoVerify -FusedAttention -Profile
& .\experiments\iron-vector\run.ps1 -Example real-greedy -Text '你好，世界。' -NoVerify -FusedAttention -NpuVocab -Row0Norm -FusedFfn -Profile
```

This project probes and benchmarks the AMD Ryzen AI NPU through the driver-provided
XRT API. The main benchmark executes AMD's end-to-end latency and throughput
workloads directly on the NPU, with no ONNX CPU fallback path.

Run from PowerShell:

```powershell
# Low-level XRT device probe (works with the current OEM driver)
.\probe.ps1

# XRT hardware latency and throughput benchmark
.\run.ps1

# Experimental direct VitisAI/ONNX path
.\run_vitisai.ps1

# Optional Windows ML catalog experiment
.\run_winml.ps1
```

The main test uses the `amd-npu-xrt` Python 3.10 environment because the driver's
`pyxrt.pyd` links against `python310.dll`. `probe.ps1` demonstrates the Python
XRT ABI; `run.ps1` invokes XRT's hardware validation workloads and parses the
measured microsecond latency and operations per second.

Ryzen AI 1.8 remains installed for its tools and models, but its beta ONNX runtime
targets Strix-class AIE2P hardware. `run_vitisai.ps1` uses a separate 1.7 VOE
environment for experimentation. It can report VitisAI-assigned nodes, but the
current mixed 1.7/1.8 stack may fall back internally and is not the authoritative
hardware check.

Set `AMD_NPU_ITERATIONS` to run a longer benchmark:

```powershell
$env:AMD_NPU_ITERATIONS = 5000
.\run.ps1
```

`run_winml.ps1` keeps the experimental Windows ML catalog path. On this Phoenix
machine the catalog currently advertises `MIGraphXExecutionProvider`, while the
direct AMD runtime exposes `VitisAIExecutionProvider`.

The low-level probe uses a separate Python 3.10 environment because the
`pyxrt.pyd` bundled with the current AMD driver links against `python310.dll`.
