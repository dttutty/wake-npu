# NPU encoder and GPU decoder experiment / NPU 编码器与 GPU 解码器实验

[简体中文](#zh-cn) | [English](#english)

<a id="zh-cn"></a>

## 简体中文

本实验使用与 AIE2 实验相同的、已安装的 Argos 中译英 Transformer 权重。编码器可沿现有 IRON NPU 路径运行，再将隐藏状态交给 Radeon 780M 上的 ONNX Runtime DirectML 解码器。解码器的六层、嵌入、词表投影和 argmax 全部在 GPU 上运行。自注意力 K/V 及预计算的交叉注意力 K/V 在各目标步骤间保持为 GPU OrtValue。Python 控制 token 循环，每步读取一个 token ID。这是序列到序列模型的编码器/解码器分离，不是仅解码器 LLM 的 prefill 分离。

### 运行

从工作区根目录运行，使用已安装的 `amd-npu` Python 环境（ONNX 1.22.0、带 DirectML 的 ONNX Runtime 1.25.2）：

```powershell
& .\experiments\hybrid-translation\run.ps1 -Fp16
& .\experiments\hybrid-translation\run.ps1 -Fp16 -Text '你好，世界。' -Repeats 3
& .\experiments\hybrid-translation\run.ps1 -Fp16 -SkipNpu
```

`-Fp16` 在 FP32 比较之外增加两种 FP16 GPU 模式。`-Rebuild` 重新生成导出的 ONNX 图并更新 FP16 版本。`-GpuPython` 可覆盖 ONNX/DirectML 解释器。NPU 执行在持久子进程中使用现有 Visual Studio 和 IRON 环境；本实验不需要更改驱动或软件包。

`-Length 64` 选择独立的 64 位置图，允许最多 64 个源分词片段及生成 token。默认容量仍为 32；`-MaxTokens` 默认为所选容量。诊断选项 `-RecordMismatches` 记录 token/数值差异后继续执行，并在最终报告中明确将数值等价性标为失败。默认仍严格拒绝不匹配。后续 19–51 个源 token 的测试及翻译差异见[较长输入结果](LONG_INPUTS.md)。

三个图通过现有受限 CTranslate2 v6 读取器导出，分别为编码器、交叉注意力 K/V 准备和带缓存的解码步骤。形状固定为 32 或 64 个源位置（含 EOS），生成 token 容量相同。填充与未来位置被掩蔽。参考导出将量化权重反量化为 FP32；FP16 在 GPU 图内部转换精度，外部张量接口保持 FP32。FP16 是半精度浮点，不是 INT8/INT4 整数量化。NPU 编码器仍使用 BF16。

### 实测结果

本机 Radeon 780M 驱动为 32.0.31041.1004；2026-09-23 测得如下热运行编码器加解码器时间。ONNX 数据为预热后两次运行的中位数，并启用执行提供程序剖析。CTranslate2 是一次预热后的 CPU INT8 运行，四线程、beam size 1。

| 输入 | CPU INT8 CTranslate2 | CPU FP32 ONNX | GPU FP32 | GPU FP16 | NPU 编码器 + GPU FP16 |
|---|---:|---:|---:|---:|---:|
| 你好，世界。 | 16.8 ms | 65.3 ms | 36.7 ms | 23.3 ms | 2334 ms |
| 人工智能正在改变世界。 | 34.9 ms | 115.2 ms | 56.9 ms | 43.9 ms | 2212 ms |
| 谢谢你。 | 14.0 ms | 51.5 ms | 28.4 ms | 21.1 ms | 2380 ms |

五种 ONNX 模式在这三条输入上都到达 EOS，并逐片段匹配 CTranslate2 输出：`Hello, world.`、`Artificial intelligence is changing the world.`、`Thank you.`。NPU/GPU FP16 混合解码器隐藏状态相对 FP32 CPU ONNX 参考的 L2 误差约 0.7–2.0%。这组小样本验证功能，不代表一般翻译质量。此处 GPU FP16 快于 GPU FP32，但优化后的 CPU INT8 仍快于两种 GPU 模式。

自定义 NPU 编码器约耗时 2.2–2.5 秒，是混合路径的主要成本；GPU 解码器仅耗时几十毫秒。因此这些结果未证明加入 NPU 有速度收益，也没有测量能耗或多句并发吞吐。

### 验证与计时范围

GPU 会话显式禁用 CPU 执行提供程序回退，保存的 ORT 剖析必须显示 DirectML 执行，且没有 CPU provider 事件。运行时检查 K/V 缓冲区常驻 `dml`、目标 ID 匹配 FP32 ONNX CPU 参考，以及整体解码器隐藏状态相对 L2 误差低于 5%。完整隐藏状态只在已计时解码循环结束后读回，用于数值检查。

诊断运行若发生 token 选择分歧，只比较输入历史相同的隐藏状态前缀；全序列误差记为 null。共享历史上的误差很小，并不能把 token 不匹配变成数值校验通过。

报告的流水线计时排除共用的源分词/嵌入准备、反分词、初始模型/会话加载、首次 JIT 编译、预热与正确性读回。会话准备和 NPU worker 启动单独记录，不能直接与含 Python 进程启动或模型加载的旧测量比较。CTranslate2 使用自身优化的 INT8 内核，ONNX 参考则为 FP32；这是实用基线，并非隔离其他变量的 CPU/GPU 硬件比较。

NPU 到 GPU 的交接在两个持久 Python 环境之间经过 FP32 主机数组及本地 `.npy` 文件，其文件 I/O 和 GPU 上传成本都计入混合路径时间。每句交接一次，不是 GPU/NPU 零拷贝共享内存。现有 NPU 编码器在内核之间仍有主机端处理。

生成的模型、原始剖析、交接数组和 JSON 报告位于 `artifacts/`，被 Git 忽略。`results_fp16.json` 包含表格样本、设备 provider 事件计数、数值误差、准备时间和逐 token 计时。本研究路径保留现有翻译实现与 AIE2 解码器实验。64 位置图独立存放在 `artifacts/length64/`。

---

<a id="english"></a>

## English

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
