# 复现 WakeNPU

[English](REPRODUCING.md) | [简体中文](REPRODUCING.zh-CN.md)

## 记录的实验环境

以下是历史实验条件，不是电脑当前配置的实时清单。

- 已测试电脑：AMD Ryzen 7 8845H（Windows 报告的型号），配备内置 Hawk Point AIE2 NPU 和 Radeon 780M GPU，运行 Windows。2026-10-31 通过注册表及 `Win32_Processor` 双重确认处理器名称；尚未独立验证 8845HS 配置。
- NPU 实验记录了 OEM 驱动 32.0.20102.3930、XRT 运行时 2.21.0、固件 1.5.5.391。旧版 1.2 驱动试验仍然崩溃，之后恢复了 OEM 驱动，详见旧版实验说明。
- 自定义 IRON 环境：XRT SDK 2.21.75、Python 3.13、Visual Studio 2022 C++ Build Tools、llvm-aie 和 mlir_aie 1.4.4.dev40+gd840cec。
- 本地 mlir-aie 源码检出版本为 `84d5df981f7d7f20766d962b37c0a098d4ea66a7`。独立 wheel 构建版本和哈希见 `experiments/iron-vector/README.md`；仅匹配源码提交不足以复现已安装的二进制工具链。
- 混合 GPU 测量记录了 Radeon 驱动 32.0.31041.1004、ONNX Runtime DirectML 1.25.2。准确计时范围见各实验 README。

## 手动搭建的边界

1. 从官方渠道取得适用 AMD 驱动和 SDK。本仓库不自动安装或降级驱动。
2. 在 `third_party/mlir-aie` 准备上游源码、Windows `iron_env.cmd` 环境、llvm-aie 编译器和匹配的 IRON wheel。上游通用安装流程曾装入不兼容 wheel，复现前需核对记录的版本/哈希。
3. 按入口准备 Python 环境：底层驱动 `pyxrt.pyd` 路径使用 Python 3.10，IRON 使用 Python 3.13，Argos 使用根目录 `.venv`。这些环境彼此独立。根目录 `requirements.txt` 只描述 Windows ML 实验，并非全部研究路径的完整依赖锁定文件。
4. 翻译模型通过 `translate.ps1 --install` 获取。真实权重 AIE2 实验预期模型位于 `translation-data/packages/translate-zh_en-1_9`。自定义 CTranslate2 v6 读取器仅覆盖已测试的格式和张量尺寸。
5. 从主 README 选一个小实验，按该目录说明运行。目前无法通过单条安装命令重建全部环境。

模型权重、厂商 DLL、旧版二进制、下载的可执行文件、构建产物、性能剖析数据和本地环境均排除在 Git 之外。历史性能表保留在 Markdown 中；文档提到的完整本地混合路径剖析产物不包含在首次发布中。

## 无需 NPU 的检查

```powershell
python .\experiments\speech-english-gpu\score_asr.py
```

该命令使用仓库内转写快照，重新计算 10 段语音的比较结果，不会执行语音识别。麦克风输入与流式上下文的限制已在分数旁注明。
