# Ryzen AI 1.2 Phoenix MatMul experiment

[English](README.md) | [简体中文](README.zh-CN.md)

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
