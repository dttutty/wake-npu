# Third-party sources and data

[English](THIRD_PARTY.md) | [简体中文](THIRD_PARTY.zh-CN.md)

WakeNPU builds experiments around independently distributed projects. Their
code, models, SDKs and datasets retain their own licenses and terms. This
repository does not redistribute vendor driver packages or model weights.

- [Xilinx/AMD mlir-aie and IRON](https://github.com/Xilinx/mlir-aie): compiler,
  runtime APIs and programming examples used by the custom AIE2 experiments.
  The local checkout identifies its license as Apache 2.0 with LLVM Exceptions.
  Several experiments import the upstream single-core matrix multiplication
  example at runtime rather than vendoring it.
- [AMD RyzenAI-SW](https://github.com/amd/RyzenAI-SW) and
  [Ryzen AI 1.2 documentation](https://ryzenai.docs.amd.com/en/1.2/llm_flow.html):
  official legacy experiment sources. Obtain proprietary components directly
  from AMD and review their applicable terms.
- [Argos Translate](https://github.com/argosopentech/argos-translate) and
  [CTranslate2](https://github.com/OpenNMT/CTranslate2): offline translation
  runtime, package format and trained zh-en model used for comparison.
  Downloaded language packages may have their own model/data licenses.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) and DirectML:
  exported-model CPU/GPU execution and execution-provider profiling.
- [OpenAI Whisper](https://github.com/openai/whisper),
  [whisper.cpp](https://github.com/ggml-org/whisper.cpp) and the
  [community Windows build](https://github.com/jiang1997/whisper.cpp-release):
  English speech recognition. Binary/model hashes are recorded in the speech
  experiment README.
- [LibriSpeech / OpenSLR 12](https://www.openslr.org/12), by Vassil Panayotov,
  Guoguo Chen, Daniel Povey and Sanjeev Khudanpur: the 10 reference transcript
  excerpts in the speech comparison. Licensed under
  [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
  Clips were retrieved through the
  [OpenSLR Hugging Face mirror](https://huggingface.co/datasets/openslr/librispeech_asr).
  Source utterance IDs are preserved; case and punctuation are normalized
  during scoring. Recognition hypotheses are experimental outputs.
- Windows Live Captions is a Windows component used for the observed speech
  comparison. Its software and model are not part of this repository.
