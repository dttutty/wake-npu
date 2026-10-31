# WakeNPU

[English](README.md) | [简体中文](README.zh-CN.md)

**唤醒电脑里闲置的 NPU。**

**已在 AMD Ryzen 7 8845H 实机的内置 AIE2 NPU 和 Radeon 780M 上验证。**

本项目探索如何编程使用 Windows 电脑内置的 AMD Ryzen AI NPU：从硬件探测、自定义 AIE2 内核，到使用真实训练权重的 Transformer 翻译器，并与 CPU 和 Radeon 780M GPU 基线比较。

这是针对 Phoenix/Hawk Point AIE2 硬件的研究原型，尚非通用加速器或一键应用。实验已在 NPU 上执行 Transformer 数学运算，但实测自定义 NPU 和混合翻译路径仍慢于实用 CPU/GPU 基线。下文的性能与兼容性描述对应本机历史实验，主要记录于 2026 年 9 月，不能推广到所有 AMD NPU。

## 已验证硬件

自定义 AIE2 内核、真实权重 Transformer 翻译实验和 CPU/GPU 对比均在这台 Windows 实机上运行，而非仅编译或仿真。

| 组件 | 测试平台 |
| --- | --- |
| 处理器 | AMD Ryzen 7 8845H，以 Windows 报告为准 |
| NPU | 内置 Hawk Point AIE2 / Ryzen AI NPU |
| GPU | AMD Radeon 780M Graphics |
| 主机系统 | Windows；SDK 与驱动版本见[复现说明](REPRODUCING.zh-CN.md) |

2026-10-31 核对时，Windows 处理器注册表项和 `Win32_Processor` 均报告 `AMD Ryzen 7 8845H w/ Radeon 780M Graphics`。本仓库尚未独立验证 8845HS 配置。已验证范围包括成功执行的自定义 IRON/XRT 路径；存档的 Ryzen AI 1.2 路径仍然崩溃。成功执行不代表具有速度优势，也不代表兼容所有 Ryzen AI NPU。

## 研究导航

| 实验 | 已验证的内容 |
| --- | --- |
| [硬件探测](#direct-ironxrt-transformer-experiments) | 驱动级 XRT 探测与硬件工作负载验证 |
| [旧版 Ryzen AI 1.2](experiments/amd-1.2-phx/README.zh-CN.md) | 编译器可识别 MatMul，但执行仍崩溃，未取得 NPU 计算输出 |
| [自定义 AIE2 内核](experiments/iron-vector/README.zh-CN.md) | 向量运算、矩阵乘法及数值校验 |
| [注意力](experiments/iron-attention/README.zh-CN.md)与 [Transformer 块](experiments/iron-transformer-block/README.zh-CN.md) | 在 NPU 上运行使用合成参数的神经网络模块 |
| [真实权重 Transformer](experiments/iron-real-weight/README.zh-CN.md) | Argos 中译英权重、六层编码器/解码器、贪心生成及实验性 NPU 词表投影 |
| [NPU 编码器 + GPU 解码器](experiments/hybrid-translation/README.zh-CN.md) | DirectML 解码器、GPU 常驻 K/V 缓存、开销与数值差异测量 |
| [较长输入测量](experiments/hybrid-translation/LONG_INPUTS.zh-CN.md) | 19–51 个源 token 的比较，包含混合路径输出分歧的案例 |
| [英语语音识别](experiments/speech-english-gpu/README.zh-CN.md) | GPU Whisper 与 Windows Live Captions 的探索性对比；尚无 NPU 语音实现 |

## 复现实验

先阅读[环境与复现说明](REPRODUCING.zh-CN.md)。仓库不包含模型权重、厂商运行时、驱动、编译器安装和构建缓存。目前需要按已测试的软件栈手动搭建。上游项目和数据出处见[第三方来源](THIRD_PARTY.zh-CN.md)。

仓库内的语音转写快照可直接用标准 Python 重新计分，无需 GPU、NPU、下载模型或厂商 SDK：

```powershell
python .\experiments\speech-english-gpu\score_asr.py
```

## 离线中英文翻译

`translate.py` 使用本地 Argos/CTranslate2 模型在 CPU 上运行。这是实用翻译器，**不是** NPU 翻译演示：实验时查阅的 Ryzen AI 兼容表没有提供受支持的 Phoenix/Hawk Point NLP/LLM 推理路径。下述自定义 IRON/XRT 研究路径可在这款 NPU 上运行模型的 Transformer 核心，同时保留主机端分词与词表投影。

```powershell
.\translate.ps1 --install
.\translate.ps1 --from zh --to en "你好，世界！"
.\translate.ps1 --from en --to zh "Hello, how are you today?"
```

语言模型存储于 `translation-data/`，安装后可离线使用。每次推理都会打印后端，避免静默回退。

<a id="direct-ironxrt-transformer-experiments"></a>

## 直接 IRON/XRT Transformer 实验

[NPU 编码器 + GPU 解码器实验](experiments/hybrid-translation/README.zh-CN.md)使用同一模型，通过 DirectML 解码器运行，K/V 缓存常驻 GPU。它将 CPU FP32、GPU FP32/FP16 和 NPU 编码器/GPU 解码器路径与 CPU INT8 贪心解码基线比较。三条测试句的输出均匹配，但 NPU 编码器主导混合路径延迟；这些短输入上的混合路径慢于纯 GPU 和 CPU INT8。

```powershell
& .\experiments\hybrid-translation\run.ps1 -Fp16
```

[固定尺寸 Transformer 块](experiments/iron-transformer-block/README.zh-CN.md)使用合成参数运行 12 个 NPU 阶段。[真实权重投影](experiments/iron-real-weight/README.zh-CN.md)读取已安装的 Argos 中译英模型，将训练得到的 INT8 权重转换成 BF16，在 NPU 上执行第一个编码器 QKV 投影。`real-attention` 继续完成全部八个头及首个注意力输出投影。`real-encoder-layer` 再完成残差/仿射 LayerNorm 和宽度为 2048 的 Swish FFN，构成第一层编码器。主机仍负责准备、偏置和布局处理。

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
