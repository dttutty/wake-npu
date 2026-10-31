# 第三方来源与数据

[English](THIRD_PARTY.md) | [简体中文](THIRD_PARTY.zh-CN.md)

WakeNPU 围绕独立分发的项目构建实验。相关代码、模型、SDK 与数据集保留各自的许可和条款；本仓库不重新分发厂商驱动包或模型权重。

- [Xilinx/AMD mlir-aie 与 IRON](https://github.com/Xilinx/mlir-aie)：自定义 AIE2 实验使用的编译器、运行时 API 和编程示例。本地检出源码标注 Apache 2.0 with LLVM Exceptions 许可。多个实验在运行时导入上游单核矩阵乘法示例，而非将其源码复制到本仓库。
- [AMD RyzenAI-SW](https://github.com/amd/RyzenAI-SW) 与 [Ryzen AI 1.2 文档](https://ryzenai.docs.amd.com/en/1.2/llm_flow.html)：旧版实验的官方来源。专有组件须直接从 AMD 获取，并查阅适用条款。
- [Argos Translate](https://github.com/argosopentech/argos-translate) 与 [CTranslate2](https://github.com/OpenNMT/CTranslate2)：离线翻译运行时、包格式，以及用于比较的真实中译英模型。下载语言包可能有独立模型/数据许可。
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) 与 DirectML：导出模型的 CPU/GPU 执行及执行提供程序剖析。
- [OpenAI Whisper](https://github.com/openai/whisper)、[whisper.cpp](https://github.com/ggml-org/whisper.cpp) 和[社区 Windows 构建](https://github.com/jiang1997/whisper.cpp-release)：英语语音识别。二进制/模型哈希记录于语音实验 README。
- [LibriSpeech / OpenSLR 12](https://www.openslr.org/12)，作者 Vassil Panayotov、Guoguo Chen、Daniel Povey 和 Sanjeev Khudanpur：语音比较中的 10 段标准转写摘录，采用 [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) 许可。录音通过 [OpenSLR 的 Hugging Face 镜像](https://huggingface.co/datasets/openslr/librispeech_asr)获取。源语句 ID 已保留；评分时归一化大小写和标点。识别假设文本是实验输出。
- Windows Live Captions 是用于语音比较观察的 Windows 组件，其软件和模型不包含在本仓库中。
