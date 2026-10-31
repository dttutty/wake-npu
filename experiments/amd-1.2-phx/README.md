# Ryzen AI 1.2 Phoenix MatMul experiment / 矩阵乘法实验

[简体中文](#zh-cn) | [English](#english)

<a id="zh-cn"></a>

## 简体中文

AMD 存档的 Ryzen AI 1.2 流程声称在 PHX/HPT 上支持即时 NPU GEMM/MatMul。本目录把官方 1.2 ONNX Runtime 和 VOE wheel 与本机 Ryzen AI 1.8 安装隔离；目录中的实验本身不更改驱动。

### 本机结果

- 实验时的驱动接受了 AMD 旧版 Phoenix GEMM xclbin，并建立 XRT 硬件上下文（`probe_legacy_gemm.py`）。这**不是**计算执行。
- INT8 `MatMulInteger` 模型被旧 EP 分配到 CPU，因为它的 VOE pass 在内嵌 Python 解释器中对 `builtins.quant_mode` 抛出 `AttributeError`。
- 宽度为 128 的 INT4 `MatMulNBits` 模型被分配到 CPU。
- 提供 AMD Phoenix xclbin 和预编译 transaction 二进制后，宽度为 2048 的 INT4 `MatMulNBits` 被融合为一个 `MATMULNBITS` 子图。
- 执行 `InferenceSession.run()` 时，`python.exe` 因 Windows 访问冲突 `0xC0000005` 退出，并显示应用错误对话框。宽度 4096 的试验也触发错误对话框。未取得 NPU 输出。
- 安装 AMD 签名的 Ryzen AI 1.2 NPU 驱动 32.0.201.204 并重启后，重复宽度 4096 的 INT4 测试。算子仍融合为一个 `MATMULNBITS` 子图，但 `run()` 在返回输出前依然因 `0xC0000005` 退出。因此较新的 OEM 驱动不是唯一原因。
- 随后从导出的驱动包恢复原 OEM 驱动。PnP 报告启用版本为 32.0.20102.3930；`xrt-smi examine` 报告 XRT 2.21.0、NPU 固件 1.5.5.391。旧版 1.2 包仍在驱动存储中，但未启用。

这些观察表明旧编译路径可以识别算子，**不能证明 NPU 已成功执行**。仅用“与新 OEM 驱动不匹配”无法解释崩溃；旧 wheel 的 XRT 构建、旧内核的假设或合成模型仍可能影响结果。

`try_legacy_matmul.py` 默认只运行 CPU 基线。若明确要复现不稳定 EP 路径，使用隔离 `.venv` 并传入 `--unsafe-npu`；它可能使 Python 进程崩溃并弹出 Windows 错误框。

来源：[AMD Ryzen AI 1.2 LLM 流程](https://ryzenai.docs.amd.com/en/1.2/llm_flow.html)、[AMD RyzenAI-SW v1.2.0](https://github.com/amd/RyzenAI-SW/tree/v1.2.0)。

---

<a id="english"></a>

## English

AMD's archived Ryzen AI 1.2 flow advertises eager NPU GEMM/MatMul for PHX/HPT.
This directory isolates the official 1.2 ONNX Runtime and VOE wheels from the
machine's current Ryzen AI 1.8 installation. It does not change the driver.

## Result on this machine

- The present driver accepted AMD's old Phoenix GEMM xclbin and created an XRT
  hardware context (`probe_legacy_gemm.py`). This is **not** computation.
- An int8 `MatMulInteger` model was CPU-assigned by the old EP because its VOE
  pass raised `AttributeError` for `builtins.quant_mode` in the embedded Python
  interpreter.
- A 128-wide INT4 `MatMulNBits` model was CPU-assigned.
- A 2048-wide INT4 `MatMulNBits` model was fused into one `MATMULNBITS` subgraph
  after providing AMD's Phoenix xclbin and precompiled transaction binaries.
- At `InferenceSession.run()`, `python.exe` exited with Windows access violation
  `0xC0000005` and showed an application-error dialog. A 4096-wide trial also
  triggered an application-error dialog. No NPU output was obtained.
- Repeated the 4096-wide INT4 test after installing AMD's signed Ryzen AI 1.2
  NPU driver 32.0.201.204 and rebooting. The operator again fused to one
  `MATMULNBITS` subgraph, but `run()` still exited with `0xC0000005` before
  returning output. This rules out the newer OEM driver as the sole cause.
- Restored the original OEM driver from an exported package. PnP reports
  32.0.20102.3930 active; `xrt-smi examine` reports XRT 2.21.0 and NPU firmware
  1.5.5.391. The old 1.2 package remains staged but is not active.

These observations establish that the old compiler path recognizes the operator,
but **do not establish successful NPU execution**. A simple mismatch with the
newer OEM driver is not sufficient to explain this crash. The old wheel's XRT
build, the legacy kernel's assumptions, or the synthetic model may still matter.

`try_legacy_matmul.py` runs only the CPU baseline by default. To reproduce the
unsafe EP path deliberately, use the isolated `.venv` and pass `--unsafe-npu`.
It can crash the Python process and display a Windows error dialog.

Sources: [AMD Ryzen AI 1.2 LLM flow](https://ryzenai.docs.amd.com/en/1.2/llm_flow.html),
[AMD RyzenAI-SW v1.2.0](https://github.com/amd/RyzenAI-SW/tree/v1.2.0).
