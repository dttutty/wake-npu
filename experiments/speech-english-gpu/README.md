# English speech recognition on Radeon 780M

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
