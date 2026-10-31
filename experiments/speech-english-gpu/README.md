# English speech recognition on Radeon 780M / Radeon 780M 英语语音识别

[简体中文](#zh-cn) | [English](#english)

<a id="zh-cn"></a>

## 简体中文

本实验用 AMD Radeon 780M 的 Vulkan 后端运行 Whisper `small.en`，不使用 NPU 或 Windows Live Captions 模型。

### 运行

在 PowerShell 中运行：

```powershell
.\experiments\speech-english-gpu\transcribe.ps1 -AudioPath 'C:\path\to\english.wav'
```

加 `-Cpu` 可在不使用 GPU 加速的情况下运行同一模型。转写结果写入 `results/`。脚本检查 GPU 日志是否含 `using Vulkan0 backend`，避免静默接受 CPU 回退。

### 已验证样本

公开的 `jfk.wav` 长度为 11.0 秒，两种设备均转写为以下英文原文（这是识别输出，保留原文）：

> And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.

其中文意思是：“所以，我的美国同胞们，不要问你的国家能为你做什么，而要问你能为你的国家做什么。”

| 设备 | 模型加载 | 编码器 | 总计 |
| --- | ---: | ---: | ---: |
| Radeon 780M Vulkan，首次 | 1.51 s | 2.17 s | 8.13 s |
| Radeon 780M Vulkan，第二次 | 1.06 s | 1.94 s | 7.04 s |
| Radeon 780M Vulkan，第三次 | 1.01 s | 1.91 s | 6.25 s |
| CPU，4 线程 | 0.91 s | 4.27 s | 6.81 s |
| CPU，4 线程，第二次 | 0.66 s | 4.09 s | 6.13 s |

GPU 加速了编码器，但在这次有限测试中，整段短音频转写与 CPU 大致打平。这些是 `whisper.cpp` 计时，不是受控延迟基准；单条干净样本也不能确定相对 Live Captions 的准确率。之后观察到 Live Captions 显示同样的 JFK 单词，但未将该检查纳入下方评分集合。

### 来源与完整性

- Windows Vulkan 构建：`jiang1997/whisper.cpp-release` 标签 `v1.8.4.1`，GitHub 发布附件 `whisper-1.8.4-windows-x64.zip`。SHA-256：`4b1b36343feb55ec3deace6a7dd18cc217f43a55e4ecce76ccd4ee3595c0b642`。
- 模型：`ggerganov/whisper.cpp` 的 `ggml-small.en.bin`。SHA-256：`c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d`。
- 样本：`ggml-org/whisper.cpp` 的 `samples/jfk.wav`。

下载的二进制、模型、音频和生成转写均被 Git 忽略，保留在本目录供本地使用。

### Live Captions 准确率比较（2026-09-24）

通过 Windows 音频输出播放 [LibriSpeech ASR 测试集](https://huggingface.co/datasets/openslr/librispeech_asr)的 10 段录音，同时运行英语（美国）Live Captions。播放后通过其辅助功能文本读取完整累计字幕历史，再按播放顺序及已知源语句分段。一些句尾词直到下一段开始才出现，因此采用最终历史，而非刚播放完的截图。

**输入控制限制：**观察到 Live Captions 的“包含麦克风音频”选项已开启，但未确认实际麦克风。不能排除麦克风拾取扬声器回放或环境声。因此这不是仅含系统音频的受控比较。分数描述捕获的输出，不能建立一般准确率排名。

相同原始 FLAC 文件分别用 Radeon 780M Vulkan 后端的 `small.en` 转写。词错误率（WER）为最小词级编辑距离除以标准词数。评分忽略大小写与标点，将 `I'm` 展开为 `I am`，但不纠正拼写，因此两个系统的 `Bessie` 相对 LibriSpeech `Bessy` 都计错。10 个源 ID、标准文本、字幕文本和 Whisper 输出文件名见 [asr-comparison.json](asr-comparison.json)。用 `python score_asr.py` 重新计分。

[whisper-transcripts.json](whisper-transcripts.json)保存了归一化空白后的对应 Whisper 输出，所以全新克隆也能计分。标准文本摘录来自 [LibriSpeech / OpenSLR 12](https://www.openslr.org/12)，作者 Vassil Panayotov、Guoguo Chen、Daniel Povey 和 Sanjeev Khudanpur，采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可。

| 子集 | 标准词数 | small.en 错误数 | Live Captions 错误数 |
| --- | ---: | ---: | ---: |
| test.clean（3 段） | 66 | 0 | 1 |
| test.other（7 段） | 91 | 19 | 11 |
| 总计（10 段） | 157 | 19（WER 12.10%） | 12（WER 7.64%） |

在这组小规模有声书样本中，Live Captions 错误更少。含古旧生僻词的一段仅有 13 个标准词，却贡献了 `small.en` 的 14 次错误及 Live Captions 的 8 次错误。排除该段后，两者分别为 5/144（3.47%）和 4/144（2.78%）。这不是通用基准，没有覆盖对话、背景噪声、口音或延迟。Live Captions 跨片段保留上下文，而 Whisper 每段独立转写。

---

<a id="english"></a>

## English

This experiment runs Whisper `small.en` with the Vulkan backend on the AMD Radeon 780M. It does not use the NPU or the Windows Live Captions model.

## Run

From PowerShell:

```powershell
.\experiments\speech-english-gpu\transcribe.ps1 -AudioPath 'C:\path\to\english.wav'
```

Add `-Cpu` to run the same model without GPU acceleration. Transcripts are written to `results/`. The script checks that the GPU log says `using Vulkan0 backend`, rather than silently accepting CPU fallback.

## Verified sample

The public `jfk.wav` sample is 11.0 seconds long. Both devices transcribed it as:

> And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.

| Device | Model load | Encoder | Total |
| --- | ---: | ---: | ---: |
| Radeon 780M Vulkan, first run | 1.51 s | 2.17 s | 8.13 s |
| Radeon 780M Vulkan, second run | 1.06 s | 1.94 s | 7.04 s |
| Radeon 780M Vulkan, third run | 1.01 s | 1.91 s | 6.25 s |
| CPU, 4 threads | 0.91 s | 4.27 s | 6.81 s |
| CPU, 4 threads, second run | 0.66 s | 4.09 s | 6.13 s |

The GPU accelerated the encoder, but the complete short transcription was roughly tied with CPU in this limited test. These are `whisper.cpp` timings, not a controlled latency benchmark, and a single clean sample cannot establish relative accuracy versus Live Captions. Live Captions was later observed displaying the same JFK words; that check was not included in the scored set below.

## Sources and integrity

- Windows Vulkan build: `jiang1997/whisper.cpp-release` tag `v1.8.4.1`, GitHub release asset `whisper-1.8.4-windows-x64.zip`. SHA-256: `4b1b36343feb55ec3deace6a7dd18cc217f43a55e4ecce76ccd4ee3595c0b642`.
- Model: `ggerganov/whisper.cpp` file `ggml-small.en.bin`. SHA-256: `c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d`.
- Sample: `ggml-org/whisper.cpp` file `samples/jfk.wav`.

The downloaded binaries, model, audio, and generated transcripts are ignored by Git and kept in this directory for local use.

## Live Captions accuracy comparison (2026-09-24)

Ten clips from the [LibriSpeech ASR test set](https://huggingface.co/datasets/openslr/librispeech_asr) were played through the Windows audio output while Live Captions (English US) was running. The full accumulated caption history was read from the Live Captions accessibility text after playback; clip text was separated using playback order and the known source utterances. Some final words only appeared after the following clip began, so the final history was used rather than an immediate screenshot.

**Input-control limitation:** Live Captions' "Include microphone audio" option
was observed enabled, and the active microphone was not identified. Microphone
pickup of speaker playback or ambient sound was not ruled out. Consequently,
this was not a controlled system-audio-only comparison. The scores describe
the captured outputs; they do not establish a general accuracy ranking.

The same original FLAC files were transcribed separately with `small.en` on the Radeon 780M Vulkan backend. Word error rate (WER) uses minimum word-level edit distance divided by reference word count. Scoring ignores case and punctuation and expands `I'm` to `I am`; it does not correct spelling, so `Bessie` versus LibriSpeech's `Bessy` counts as an error for both systems. See [asr-comparison.json](asr-comparison.json) for the 10 source IDs, references, caption text, and Whisper output filenames. Recalculate with `python score_asr.py`.

[whisper-transcripts.json](whisper-transcripts.json) preserves the corresponding
Whisper outputs with whitespace normalized, so scoring works in a fresh clone.
The reference excerpts are attributed to the [LibriSpeech corpus, OpenSLR 12](https://www.openslr.org/12),
by Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur,
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

| Split | Reference words | small.en errors | Live Captions errors |
| --- | ---: | ---: | ---: |
| test.clean (3 clips) | 66 | 0 | 1 |
| test.other (7 clips) | 91 | 19 | 11 |
| Total (10 clips) | 157 | 19 (12.10% WER) | 12 (7.64% WER) |

On this small audiobook sample, Live Captions was more accurate. One difficult 13-word clip containing archaic words accounted for 14 `small.en` errors and 8 Live Captions errors. Without that clip, the totals are 5/144 (3.47%) and 4/144 (2.78%), respectively. This is not a general benchmark: it does not cover conversational speech, background noise, accents, or latency. Live Captions also retained context across clips, while Whisper transcribed each clip independently.
