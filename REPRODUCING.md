# Reproducing WakeNPU / 复现 WakeNPU

[简体中文](#zh-cn) | [English](#english)

<a id="zh-cn"></a>

## 简体中文

### 记录的实验环境

以下是历史实验条件，不是电脑当前配置的实时清单。

- Windows 主机，配备 Phoenix/Hawk Point AIE2 NPU 和 Radeon 780M GPU。
- NPU 实验记录了 OEM 驱动 32.0.20102.3930、XRT 运行时 2.21.0、固件 1.5.5.391。旧版 1.2 驱动试验仍然崩溃，之后恢复了 OEM 驱动，详见旧版实验说明。
- 自定义 IRON 环境：XRT SDK 2.21.75、Python 3.13、Visual Studio 2022 C++ Build Tools、llvm-aie 和 mlir_aie 1.4.4.dev40+gd840cec。
- 本地 mlir-aie 源码检出版本为 `84d5df981f7d7f20766d962b37c0a098d4ea66a7`。独立 wheel 构建版本和哈希见 `experiments/iron-vector/README.md`；仅匹配源码提交不足以复现已安装的二进制工具链。
- 混合 GPU 测量记录了 Radeon 驱动 32.0.31041.1004、ONNX Runtime DirectML 1.25.2。准确计时范围见各实验 README。

### 手动搭建的边界

1. 从官方渠道取得适用 AMD 驱动和 SDK。本仓库不自动安装或降级驱动。
2. 在 `third_party/mlir-aie` 准备上游源码、Windows `iron_env.cmd` 环境、llvm-aie 编译器和匹配的 IRON wheel。上游通用安装流程曾装入不兼容 wheel，复现前需核对记录的版本/哈希。
3. 按入口准备 Python 环境：底层驱动 `pyxrt.pyd` 路径使用 Python 3.10，IRON 使用 Python 3.13，Argos 使用根目录 `.venv`。这些环境彼此独立。根目录 `requirements.txt` 只描述 Windows ML 实验，并非全部研究路径的完整依赖锁定文件。
4. 翻译模型通过 `translate.ps1 --install` 获取。真实权重 AIE2 实验预期模型位于 `translation-data/packages/translate-zh_en-1_9`。自定义 CTranslate2 v6 读取器仅覆盖已测试的格式和张量尺寸。
5. 从主 README 选一个小实验，按该目录说明运行。目前无法通过单条安装命令重建全部环境。

模型权重、厂商 DLL、旧版二进制、下载的可执行文件、构建产物、性能剖析数据和本地环境均排除在 Git 之外。历史性能表保留在 Markdown 中；文档提到的完整本地混合路径剖析产物不包含在首次发布中。

### 无需 NPU 的检查

```powershell
python .\experiments\speech-english-gpu\score_asr.py
```

该命令使用仓库内转写快照，重新计算 10 段语音的比较结果，不会执行语音识别。麦克风输入与流式上下文的限制已在分数旁注明。

---

<a id="english"></a>

## English

## Recorded environment

These are historical experiment conditions, not a live inventory of the PC.

- Windows host with a Phoenix/Hawk Point AIE2 NPU and Radeon 780M GPU.
- NPU experiments recorded OEM driver 32.0.20102.3930, XRT runtime 2.21.0
  and firmware 1.5.5.391. The legacy 1.2 driver trial also crashed and the OEM
  driver was restored; see the legacy experiment notes.
- Custom IRON setup: XRT SDK 2.21.75, Python 3.13, Visual Studio 2022 C++
  Build Tools, llvm-aie and mlir_aie 1.4.4.dev40+gd840cec.
- The local mlir-aie source checkout was at
  `84d5df981f7d7f20766d962b37c0a098d4ea66a7`. The separate wheel build and hash
  are documented in `experiments/iron-vector/README.md`; matching the source
  checkout alone does not reproduce the installed binary toolchain.
- Hybrid GPU measurements recorded Radeon driver 32.0.31041.1004 and
  ONNX Runtime DirectML 1.25.2. Exact timing scope is in each experiment README.

## Manual setup boundaries

1. Obtain the appropriate AMD drivers and SDK through their official channels.
   Installing or downgrading drivers is not automated by this repository.
2. Prepare the upstream mlir-aie checkout in `third_party/mlir-aie`, its Windows
   `iron_env.cmd` environment, llvm-aie compiler and the matching IRON wheel.
   The generic upstream setup previously installed an incompatible wheel;
   consult the recorded wheel version/hash before attempting a reproduction.
3. Prepare Python environments required by the entry points. The low-level
   driver `pyxrt.pyd` path used Python 3.10; IRON used Python 3.13; Argos used
   the root `.venv`. These are distinct environments. The root
   `requirements.txt` only describes the Windows ML experiment and is not a
   complete lockfile for all research paths.
4. For translation, `translate.ps1 --install` obtains Argos language models.
   The trained AIE2 experiments expect `translation-data/packages/translate-zh_en-1_9`.
   The custom CTranslate2 v6 reader is bounded to the tested format and shapes.
5. Select a small experiment from the main README and follow its local notes.
   The full environment cannot yet be rebuilt through a single setup command.

Model weights, vendor DLLs, legacy binaries, downloaded executables, build
outputs, profiles and local environments are excluded from Git. Historical
performance tables are retained in Markdown. Full local hybrid profiling
artifacts mentioned in those notes are not included in this initial publication.

## Checks without NPU hardware

```powershell
python .\experiments\speech-english-gpu\score_asr.py
```

This recalculates the recorded 10-clip speech comparison from checked-in
transcripts. It does not run speech recognition. Its microphone-input and
streaming-context limitations are documented alongside the scores.
