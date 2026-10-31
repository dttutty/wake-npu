# Longer-input measurements

[English](LONG_INPUTS.md) | [简体中文](LONG_INPUTS.zh-CN.md)

The hybrid experiment was extended to a separate 64-position ONNX graph
set to use more of the existing 64-row NPU encoder. The default 32-position
graphs remain available. Five longer inputs were tested on 2026-09-23,
including two paragraphs. All modes reached EOS within the configured
target limit; no result in the table was cut off by the token limit.

| Source tokens including EOS | Graph capacity | CPU INT8 CT2 | GPU FP16 | NPU encoder + GPU FP16 | Hybrid matches FP32 ONNX tokens |
|---:|---:|---:|---:|---:|---|
| 19 | 32 | 93 ms | 110 ms | 2761 ms | Yes |
| 22 | 32 | 95 ms | 91 ms | 2609 ms | No |
| 27 | 32 | 103 ms | 88 ms | 2834 ms | Yes |
| 40 | 64 | 177 ms | 182 ms | 3310 ms | Yes |
| 51 | 64 | 244 ms | 225 ms | 3056 ms | No |

ONNX values are medians of three warmed runs with profiling enabled;
CPU INT8 is one warmed four-thread greedy run. Timing exclusions and the
host/file NPU-to-GPU handoff are the same as in the main README. Different
backends sometimes generate different numbers of tokens, so these are
observed request latencies, not equal-work hardware throughput comparisons.
Adjacent NPU runs vary enough that 40 versus 51 input tokens should not be
interpreted as a monotonic scaling curve.

The 40-token case generated 51 target tokens in all ONNX modes. On the
51-token case, GPU generated 63 and the hybrid path generated 62, including
EOS. The longest input still took about 14 times as long on the hybrid
path as on GPU FP16. Its NPU encoder averaged about 2.80 seconds across
the three timed hybrid-FP16 runs. Increasing the number of useful input
rows did not establish a speed advantage in this implementation. This is
bounded 64-position translation, not a thousand-token LLM context test.

## Numerical Differences

GPU FP32 and GPU FP16 matched the CPU FP32 ONNX token sequence for all
five inputs. The NPU-encoder variants diverged on the 22-token and
51-token cases, in both GPU decoder precisions. The first divergent
choice was target step 18 (zero-based) in each case. Shared-history
decoder relative L2 error was about 1.8% and 2.5%, respectively; small
hidden-state differences can still change greedy token selection.

For example, in the 22-token case the FP32 ONNX path generated
`in real text`, while the hybrid path generated `on the actual text`.
These differences were recorded as numerical equivalence failures, not
silently accepted as passes. No human or corpus-level translation-quality
evaluation was performed. Most exported-ONNX outputs also differed from
the independent INT8 CTranslate2 outputs; same weights do not establish
bitwise or translation equivalence across these numerical implementations.

DirectML execution profiles still showed no CPU graph execution, and K/V
buffers remained on the GPU. The diagnostic mode only relaxes the rule
that stops at the first numerical mismatch; provider and device-residency
checks remain enforced.

## Inputs and Reproduction

The 32-position cases are:

1. 今天我们在办公室讨论了人工智能的发展，以及这项技术将如何改变未来的工作方式。
2. 为了提高翻译质量，研究人员需要比较不同模型在真实文本上的表现，并仔细分析错误。
3. 虽然这台电脑可以同时使用图形处理器和神经网络处理器，但是合理分配任务才能真正提高运行效率。

The 40-token paragraph joins cases 1 and 2. The 51-token paragraph joins
case 3 with `我们希望在不连接互联网的情况下，使用这台电脑快速准确地翻译中文和英文，并保护用户的隐私。`.

```powershell
& .\experiments\hybrid-translation\run.ps1 -Length 64 -Fp16 -RecordMismatches -Repeats 3 -Text '虽然这台电脑可以同时使用图形处理器和神经网络处理器，但是合理分配任务才能真正提高运行效率。我们希望在不连接互联网的情况下，使用这台电脑快速准确地翻译中文和英文，并保护用户的隐私。' -Report .\experiments\hybrid-translation\artifacts\longest_retest.json
```

Raw measurements are in `artifacts/long_inputs.json` and
`artifacts/long_inputs_64.json`, including all generated pieces, EOS flags,
first-difference indices, per-stage timings, and profile paths.
