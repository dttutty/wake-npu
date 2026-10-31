# NPU 编码器与 GPU 解码器实验

[English](README.md) | [简体中文](README.zh-CN.md)

本实验使用与 AIE2 实验相同的、已安装的 Argos 中译英 Transformer 权重。编码器可沿现有 IRON NPU 路径运行，再将隐藏状态交给 Radeon 780M 上的 ONNX Runtime DirectML 解码器。解码器的六层、嵌入、词表投影和 argmax 全部在 GPU 上运行。自注意力 K/V 及预计算的交叉注意力 K/V 在各目标步骤间保持为 GPU OrtValue。Python 控制 token 循环，每步读取一个 token ID。这是序列到序列模型的编码器/解码器分离，不是仅解码器 LLM 的 prefill 分离。

## 运行

从工作区根目录运行，使用已安装的 `amd-npu` Python 环境（ONNX 1.22.0、带 DirectML 的 ONNX Runtime 1.25.2）：

```powershell
& .\experiments\hybrid-translation\run.ps1 -Fp16
& .\experiments\hybrid-translation\run.ps1 -Fp16 -Text '你好，世界。' -Repeats 3
& .\experiments\hybrid-translation\run.ps1 -Fp16 -SkipNpu
```

`-Fp16` 在 FP32 比较之外增加两种 FP16 GPU 模式。`-Rebuild` 重新生成导出的 ONNX 图并更新 FP16 版本。`-GpuPython` 可覆盖 ONNX/DirectML 解释器。NPU 执行在持久子进程中使用现有 Visual Studio 和 IRON 环境；本实验不需要更改驱动或软件包。

`-Length 64` 选择独立的 64 位置图，允许最多 64 个源分词片段及生成 token。默认容量仍为 32；`-MaxTokens` 默认为所选容量。诊断选项 `-RecordMismatches` 记录 token/数值差异后继续执行，并在最终报告中明确将数值等价性标为失败。默认仍严格拒绝不匹配。后续 19–51 个源 token 的测试及翻译差异见[较长输入结果](LONG_INPUTS.zh-CN.md)。

三个图通过现有受限 CTranslate2 v6 读取器导出，分别为编码器、交叉注意力 K/V 准备和带缓存的解码步骤。形状固定为 32 或 64 个源位置（含 EOS），生成 token 容量相同。填充与未来位置被掩蔽。参考导出将量化权重反量化为 FP32；FP16 在 GPU 图内部转换精度，外部张量接口保持 FP32。FP16 是半精度浮点，不是 INT8/INT4 整数量化。NPU 编码器仍使用 BF16。

## 实测结果

本机 Radeon 780M 驱动为 32.0.31041.1004；2026-09-23 测得如下热运行编码器加解码器时间。ONNX 数据为预热后两次运行的中位数，并启用执行提供程序剖析。CTranslate2 是一次预热后的 CPU INT8 运行，四线程、beam size 1。

| 输入 | CPU INT8 CTranslate2 | CPU FP32 ONNX | GPU FP32 | GPU FP16 | NPU 编码器 + GPU FP16 |
|---|---:|---:|---:|---:|---:|
| 你好，世界。 | 16.8 ms | 65.3 ms | 36.7 ms | 23.3 ms | 2334 ms |
| 人工智能正在改变世界。 | 34.9 ms | 115.2 ms | 56.9 ms | 43.9 ms | 2212 ms |
| 谢谢你。 | 14.0 ms | 51.5 ms | 28.4 ms | 21.1 ms | 2380 ms |

五种 ONNX 模式在这三条输入上都到达 EOS，并逐片段匹配 CTranslate2 输出：`Hello, world.`、`Artificial intelligence is changing the world.`、`Thank you.`。NPU/GPU FP16 混合解码器隐藏状态相对 FP32 CPU ONNX 参考的 L2 误差约 0.7–2.0%。这组小样本验证功能，不代表一般翻译质量。此处 GPU FP16 快于 GPU FP32，但优化后的 CPU INT8 仍快于两种 GPU 模式。

自定义 NPU 编码器约耗时 2.2–2.5 秒，是混合路径的主要成本；GPU 解码器仅耗时几十毫秒。因此这些结果未证明加入 NPU 有速度收益，也没有测量能耗或多句并发吞吐。

## 验证与计时范围

GPU 会话显式禁用 CPU 执行提供程序回退，保存的 ORT 剖析必须显示 DirectML 执行，且没有 CPU provider 事件。运行时检查 K/V 缓冲区常驻 `dml`、目标 ID 匹配 FP32 ONNX CPU 参考，以及整体解码器隐藏状态相对 L2 误差低于 5%。完整隐藏状态只在已计时解码循环结束后读回，用于数值检查。

诊断运行若发生 token 选择分歧，只比较输入历史相同的隐藏状态前缀；全序列误差记为 null。共享历史上的误差很小，并不能把 token 不匹配变成数值校验通过。

报告的流水线计时排除共用的源分词/嵌入准备、反分词、初始模型/会话加载、首次 JIT 编译、预热与正确性读回。会话准备和 NPU worker 启动单独记录，不能直接与含 Python 进程启动或模型加载的旧测量比较。CTranslate2 使用自身优化的 INT8 内核，ONNX 参考则为 FP32；这是实用基线，并非隔离其他变量的 CPU/GPU 硬件比较。

NPU 到 GPU 的交接在两个持久 Python 环境之间经过 FP32 主机数组及本地 `.npy` 文件，其文件 I/O 和 GPU 上传成本都计入混合路径时间。每句交接一次，不是 GPU/NPU 零拷贝共享内存。现有 NPU 编码器在内核之间仍有主机端处理。

生成的模型、原始剖析、交接数组和 JSON 报告位于 `artifacts/`，被 Git 忽略。`results_fp16.json` 包含表格样本、设备 provider 事件计数、数值误差、准备时间和逐 token 计时。本研究路径保留现有翻译实现与 AIE2 解码器实验。64 位置图独立存放在 `artifacts/length64/`。
