# Reproducing WakeNPU

[English](REPRODUCING.md) | [简体中文](REPRODUCING.zh-CN.md)

## Recorded environment

These are historical experiment conditions, not a live inventory of the PC.

- Tested PC: AMD Ryzen 7 8845H (Windows-reported model), with its built-in
  Hawk Point AIE2 NPU and Radeon 780M GPU, running Windows. The processor name
  was confirmed through both the registry and `Win32_Processor` on 2026-10-31;
  an 8845HS configuration has not been independently verified.
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
