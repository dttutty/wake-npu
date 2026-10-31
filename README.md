# WakeNPU

[简体中文](#zh-cn) | [English](#english)

<a id="zh-cn"></a>

## 简体中文

**唤醒电脑里闲置的 NPU。**

本项目探索如何编程使用 Windows 电脑内置的 AMD Ryzen AI NPU：从硬件探测、自定义 AIE2 内核，到使用真实训练权重的 Transformer 翻译器，并与 CPU 和 Radeon 780M GPU 基线比较。

这是针对 Phoenix/Hawk Point AIE2 硬件的研究原型，尚非通用加速器或一键应用。实验已在 NPU 上执行 Transformer 数学运算，但实测自定义 NPU 和混合翻译路径仍慢于实用 CPU/GPU 基线。下文的性能与兼容性描述对应本机历史实验，主要记录于 2026 年 9 月，不能推广到所有 AMD NPU。

### 研究导航

| 实验 | 已验证的内容 |
| --- | --- |
| [硬件探测](#direct-ironxrt-transformer-experiments) | 驱动级 XRT 探测与硬件工作负载验证 |
| [旧版 Ryzen AI 1.2](experiments/amd-1.2-phx/README.md) | 编译器可识别 MatMul，但执行仍崩溃，未取得 NPU 计算输出 |
| [自定义 AIE2 内核](experiments/iron-vector/README.md) | 向量运算、矩阵乘法及数值校验 |
| [注意力](experiments/iron-attention/README.md)与 [Transformer 块](experiments/iron-transformer-block/README.md) | 在 NPU 上运行使用合成参数的神经网络模块 |
| [真实权重 Transformer](experiments/iron-real-weight/README.md) | Argos 中译英权重、六层编码器/解码器、贪心生成及实验性 NPU 词表投影 |
| [NPU 编码器 + GPU 解码器](experiments/hybrid-translation/README.md) | DirectML 解码器、GPU 常驻 K/V 缓存、开销与数值差异测量 |
| [较长输入测量](experiments/hybrid-translation/LONG_INPUTS.md) | 19–51 个源 token 的比较，包含混合路径输出分歧的案例 |
| [英语语音识别](experiments/speech-english-gpu/README.md) | GPU Whisper 与 Windows Live Captions 的探索性对比；尚无 NPU 语音实现 |

### 复现实验

先阅读[环境与复现说明](REPRODUCING.md)。仓库不包含模型权重、厂商运行时、驱动、编译器安装和构建缓存。目前需要按已测试的软件栈手动搭建。上游项目和数据出处见[第三方来源](THIRD_PARTY.md)。

仓库内的语音转写快照可直接用标准 Python 重新计分，无需 GPU、NPU、下载模型或厂商 SDK：

```powershell
python .\experiments\speech-english-gpu\score_asr.py
```

### 离线中英文翻译

`translate.py` 使用本地 Argos/CTranslate2 模型在 CPU 上运行。这是实用翻译器，**不是** NPU 翻译演示：实验时查阅的 Ryzen AI 兼容表没有提供受支持的 Phoenix/Hawk Point NLP/LLM 推理路径。下述自定义 IRON/XRT 研究路径可在这款 NPU 上运行模型的 Transformer 核心，同时保留主机端分词与词表投影。

```powershell
.\translate.ps1 --install
.\translate.ps1 --from zh --to en "你好，世界！"
.\translate.ps1 --from en --to zh "Hello, how are you today?"
```

语言模型存储于 `translation-data/`，安装后可离线使用。每次推理都会打印后端，避免静默回退。

### 直接 IRON/XRT Transformer 实验

[NPU 编码器 + GPU 解码器实验](experiments/hybrid-translation/README.md)使用同一模型，通过 DirectML 解码器运行，K/V 缓存常驻 GPU。它将 CPU FP32、GPU FP32/FP16 和 NPU 编码器/GPU 解码器路径与 CPU INT8 贪心解码基线比较。三条测试句的输出均匹配，但 NPU 编码器主导混合路径延迟；这些短输入上的混合路径慢于纯 GPU 和 CPU INT8。

```powershell
& .\experiments\hybrid-translation\run.ps1 -Fp16
```

[固定尺寸 Transformer 块](experiments/iron-transformer-block/README.md)使用合成参数运行 12 个 NPU 阶段。[真实权重投影](experiments/iron-real-weight/README.md)读取已安装的 Argos 中译英模型，将训练得到的 INT8 权重转换成 BF16，在 NPU 上执行第一个编码器 QKV 投影。`real-attention` 继续完成全部八个头及首个注意力输出投影。`real-encoder-layer` 再完成残差/仿射 LayerNorm 和宽度为 2048 的 Swish FFN，构成第一层编码器。主机仍负责准备、偏置和布局处理。

`real-encoder-stack` 串联六层真实权重编码器，并与独立 CPU 数值路径校验。`real-first-token` 将编码器输出送入目标位置 0 的六层解码器，再通过 CPU 词表投影查看原始得分最高的英文 token。`real-greedy` 加入逐层自注意力和编码器记忆 K/V 缓存，持续生成直到 `</s>` 或目标 token 上限。三条已测试中文句均生成完整翻译，与已安装 CPU 翻译器相同，但它仍是 NPU/CPU 混合研究路径，并非受支持的翻译后端。

给 `real-greedy` 加上 `-NoVerify` 可跳过重复的 NumPy 参考计算并打印阶段计时；`-Profile` 可统计部分 IRON 主机 API 调用次数与墙钟时间，这些不是纯设备内核计时。即使如此，反复启动小内核与主机/设备同步仍存在，原型在测试短句上慢于 CPU 翻译器。

可选的 `-FusedAttention` 在一次 AIE2 调用中计算八个头所需的解码器注意力第 0 行，降低了已测试句子的热运行墙钟时间，但仍未超过 CPU CTranslate2。`-NpuVocab` 将 65,000 词投影、逐 tile 偏置/top-2 和最终 64 tile 候选归并移到 NPU；CPU 仍负责分词、嵌入准备、中间偏置/布局和生成控制。这是向全 NPU 模型数学运算迈进的功能实验，没有带来速度改善。

`-Row0Norm` 仅对每个 token 实际使用的解码器输出行执行残差 LayerNorm，同时保留默认全行路径。`-FusedFfn` 将解码器 FFN 上投影偏置与 SiLU 合并为一个 AIE2 内核，中间结果保留在设备上；测试翻译通过，但一次相邻热运行中慢于默认 FFN。

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

本项目通过驱动提供的 XRT API 探测并测试 AMD Ryzen AI NPU。主基准直接在 NPU 上执行 AMD 的端到端延迟与吞吐工作负载，没有 ONNX CPU 回退路径。

在 PowerShell 中运行：

```powershell
# 底层 XRT 设备探测，已在所记录的 OEM 驱动上运行
.\probe.ps1

# XRT 硬件延迟与吞吐基准
.\run.ps1

# 实验性直接 VitisAI/ONNX 路径
.\run_vitisai.ps1

# 可选的 Windows ML 目录实验
.\run_winml.ps1
```

主测试使用 `amd-npu-xrt` Python 3.10 环境，因为驱动的 `pyxrt.pyd` 链接 `python310.dll`。`probe.ps1` 展示 Python XRT ABI；`run.ps1` 调用 XRT 硬件验证工作负载，解析微秒延迟与每秒操作数。

实验环境仍安装 Ryzen AI 1.8 以使用其工具和模型，但它的测试版 ONNX 运行时面向 Strix 类 AIE2P 硬件。`run_vitisai.ps1` 使用独立 1.7 VOE 环境；它能报告分配给 VitisAI 的节点，但混合 1.7/1.8 软件栈可能内部回退，不能作为硬件实际执行的权威证据。

设置 `AMD_NPU_ITERATIONS` 可延长基准：

```powershell
$env:AMD_NPU_ITERATIONS = 5000
.\run.ps1
```

`run_winml.ps1` 保留实验性 Windows ML 目录路径。在本机 Phoenix 的记录中，目录公布 `MIGraphXExecutionProvider`，而直接 AMD 运行时暴露 `VitisAIExecutionProvider`。底层探测使用独立 Python 3.10 环境，原因同样是驱动附带的 `pyxrt.pyd` 链接 `python310.dll`。

---

<a id="english"></a>

## English

**Put your idle NPU to work.**

Experiments in programming the AMD Ryzen AI NPU already inside a Windows PC:
from hardware probes and custom AIE2 kernels to a trained Transformer translator,
with CPU and Radeon 780M GPU baselines.

This is a research prototype for Phoenix/Hawk Point AIE2 hardware, not a
general-purpose accelerator or a one-click application. Running Transformer
math on the NPU is possible in these experiments, but the measured custom NPU
and hybrid translation paths were slower than the practical CPU/GPU baselines.
Performance and compatibility statements below describe the recorded local
experiments, primarily from September 2026, rather than all AMD NPUs.

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
