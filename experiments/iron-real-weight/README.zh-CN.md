# Phoenix 上的真实翻译模型权重

[English](README.md) | [简体中文](README.zh-CN.md)

## 运行入口

从工作区根目录运行：

```powershell
& .\experiments\iron-vector\run.ps1 -Example real-weight
& .\experiments\iron-vector\run.ps1 -Example real-weight -Text '人工智能正在改变世界。'
& .\experiments\iron-vector\run.ps1 -Example softmax64
& .\experiments\iron-vector\run.ps1 -Example real-attention -Text '人工智能正在改变世界。'
& .\experiments\iron-vector\run.ps1 -Example affine-norm512
& .\experiments\iron-vector\run.ps1 -Example affine-norm-row0
& .\experiments\iron-vector\run.ps1 -Example matmul-shapes
& .\experiments\iron-vector\run.ps1 -Example compact-matmul
& .\experiments\iron-vector\run.ps1 -Example ffn-bias-silu
& .\experiments\iron-vector\run.ps1 -Example real-encoder-layer -Text '人工智能正在改变世界。'
& .\experiments\iron-vector\run.ps1 -Example real-encoder-stack -Text '人工智能正在改变世界。'
& .\experiments\iron-vector\run.ps1 -Example real-decoder-layer -Text '人工智能正在改变世界。'
& .\experiments\iron-vector\run.ps1 -Example real-first-token -Text '人工智能正在改变世界。'
& .\experiments\iron-vector\run.ps1 -Example real-greedy -Text '你好，世界。'
& .\experiments\iron-vector\run.ps1 -Example real-greedy -Text '人工智能正在改变世界。' -MaxTokens 16
& .\experiments\iron-vector\run.ps1 -Example real-greedy -Text '你好，世界。' -NoVerify
& .\experiments\iron-vector\run.ps1 -Example real-greedy -Text '你好，世界。' -NoVerify -Profile
& .\experiments\iron-vector\run.ps1 -Example attention-row0
& .\experiments\iron-vector\run.ps1 -Example real-greedy -Text '你好，世界。' -NoVerify -FusedAttention -Profile
& .\experiments\iron-vector\run.ps1 -Example vocab-top2
& .\experiments\iron-vector\run.ps1 -Example real-greedy -Text '你好，世界。' -NoVerify -FusedAttention -NpuVocab -Profile
& .\experiments\iron-vector\run.ps1 -Example real-greedy -Text '你好，世界。' -NoVerify -FusedAttention -NpuVocab -Row0Norm -Profile
& .\experiments\iron-vector\run.ps1 -Example real-greedy -Text '你好，世界。' -NoVerify -FusedAttention -NpuVocab -Row0Norm -FusedFfn -Profile
```

## 真实权重投影与注意力

程序只读已安装的 Argos 中译英包，不修改原包。现有 Argos Python 环境对中文分词；受限的只读 CTranslate2 v6 解析器加载 INT8 嵌入和编码器第 0 层 QKV 矩阵，包括逐行量化尺度。真实训练权重转成 BF16 后，使用 IRON 单核 MatMul 示例在 NPU 上相乘，并与 NumPy 校验。

`real-weight` 的结果只是基于模型嵌入向量的一次投影，不含位置编码、偏置、归一化、注意力、后续编码器/解码器层或输出 token 生成。它本身还不是翻译引擎，也不取代 CPU Argos 翻译器。

`real-attention` 进一步读取同一模型的缩放嵌入、位置编码、QKV 矩阵与偏置，以及八个 64 维注意力头。NPU 执行 QKV MatMul、各头 QK MatMul、自定义 64 列掩蔽 softmax、各头 PV MatMul 和输出 MatMul。主机负责分词、INT8 反量化、BF16 转换、偏置、query 缩放及头张量重排。每个 NPU 阶段以实际输入与 NumPy 校验，另有完整 NumPy 路径校验最终输出。这覆盖注意力核心直到输出投影，尚未进入残差/LayerNorm 和 FFN，仍不能生成翻译文本。

自定义 softmax 采用减最大值和 AIE2 兼容的近似指数函数。它是数值研究内核，不是 CTranslate2 CPU 算子的性能优化替代品。

## 编码器与首个解码步骤

`real-encoder-layer` 完成编码器第 0 层：注意力后执行自定义 AIE2 残差加 gamma/beta LayerNorm、训练权重的 512→2048 FFN 上投影、NPU Swish/SiLU、2048→512 FFN 下投影，以及第二个残差加仿射 LayerNorm。主机仍负责分词、INT8→BF16 转换、偏置、query 缩放和布局/打包。两次 FFN MatMul 先输出 FP32，再加偏置并缩窄为 BF16。尤其是深度 2048 的下投影，使用示例的 BF16 输出累加会产生较大误差；`matmul-shapes` 可复现该限制并校验 FP32 输出修正。逐阶段检查和独立 CPU 整层路径共同验证数值结果。该入口仅计算一层编码器，不运行全部六层或任何解码步骤，尚不能翻译。

`real-encoder-stack` 将每层训练权重编码器的真实输出接入下一层，在 NPU 上完成全部六层；同时从相同嵌入独立传播六层 NumPy 参考路径，比较最终有效 token 张量。`你好，世界。` 和 `人工智能正在改变世界。` 的最终 BF16 张量最大绝对误差分别约 0.035 和 0.049。主机预处理、偏置、缩放及张量布局仍保留。该入口返回编码器隐藏状态，不运行解码器，也不返回译文。

`real-decoder-layer` 在此基础上运行目标步骤 0 的解码器第 0 层。模型的 `start_from_zero_embedding` 标志令 `<s>` 输入为零嵌入加第一个位置编码。NPU 内核执行解码器自注意力、针对六层 NPU 编码器输出的交叉注意力、三次残差/仿射 LayerNorm，以及 Swish FFN。独立 NumPy 路径使用另行计算的编码器状态。解码器注意力投影 MatMul 先输出 FP32 再缩窄为 BF16，使测试中第一层自注意力误差从 0.5 降至约 0.002。它仍是数值实验，不是生产推理路径；仅使用首个目标位置，未测试多步因果掩码，也不执行解码器第 1–5 层或词表投影。

`real-first-token` 在目标步骤 0 运行全部六层训练权重解码器，把各层 NPU 输出传给下一层，同时独立传播 NumPy 参考路径。主机随后对两条路径的最终隐藏状态执行权重共享的 65,000 token 词表投影。`你好，世界。`、`人工智能正在改变世界。` 和一条较长的 22 token 测试句，两条路径分别选出 `▁Hello`、`▁Art`、`▁Today`。最终隐藏状态相对 L2 误差约 1.8%、2.9%、1.7%，最大逐元素误差约 0.344、0.455、0.344；三例首 token 得分差距均至少 0.475。这是原始 argmax token，不是束搜索翻译结果。该入口不执行后续目标步骤，也不检验包含多个 key 的因果掩码。

## 贪心解码与验证

`real-greedy` 加入自回归解码，为六层解码器分别维护自注意力 K/V 与编码器记忆 K/V 缓存。每个新目标位置只使用历史缓存的自注意力 key 加当前 key，所以 64 列 NPU softmax 强制因果性，不暴露未来 token。NPU 和独立 NumPy 路径每步原始 argmax 必须一致，第 0 行隐藏状态相对 L2 误差须低于 5%。权重共享的 65,000 词投影及 Argos 目标片段解码仍在 CPU 上执行。

测试输入 `你好，世界。`、`人工智能正在改变世界。`、`谢谢你。` 均到达 EOS，生成 `Hello, world.`、`Artificial intelligence is changing the world.`、`Thank you.`，与已安装 CPU CTranslate2 翻译器在这三句上的输出相同。这是研究用贪心解码器，不是受支持的 Argos 束搜索路径或纯 NPU 实现；尚未通过翻译语料、较长输出或任意语言对验证。

验证默认开启：用 NumPy 重算注意力、FFN 和完整编码器/解码器路径，检查 NPU 各阶段并要求每步目标 argmax 匹配。`-NoVerify` 跳过重复 CPU 计算，保留有限值检查；使用的 NPU 内核、缓存和 CPU 词表投影相同。两种模式均打印准备、编码器流水线、解码器层、词表头和反分词计时。编码器/解码器数字包含 Python、权重准备、CPU 布局/偏置、缓冲区往返及 NPU 执行，并非纯设备内核时间。

## 初始性能与主机 API 剖析

本机使用缓存的 IRON 构建时，`你好，世界。` 配合 `-NoVerify` 端到端约 19.6 秒，CPU CTranslate2 的 `translate.ps1` 约 4.4 秒。在 NPU 脚本内部，准备约 0.3 秒、编码器 2.1 秒、解码器层 14.1 秒、CPU 词表头 0.02 秒。较长的 `人工智能正在改变世界。` 生成 10 个目标 token，NPU 脚本内部约 29.8 秒（解码器层 27.1 秒），CPU CTranslate2 进程时间约 4.6 秒。

这些是本地墙钟时间示例，不是受控设备基准：CTranslate2 使用优化翻译路径，本实验使用固定 64 行内核与贪心解码。按实现代码估算，解码器步骤 0 派发约 354 次高层 NPU 操作，后续每步约 348 次；每步的 12 个注意力模块为八个头分别启动 QK、softmax 和 PV。这是代码级估计，不是硬件计数器。缓存主机 BF16 权重未明显改善短句时间，因此移除了额外缓存。剩余解码器计时没有分离 NPU 计算、Python 派发、上传、读回和等待。

`-Profile` 在本流水线的 IRON 主机 API 调用外统计墙钟时间；配合 `-NoVerify` 可排除重复 NumPy 参考路径。一次缓存构建的 `你好，世界。` 运行在脚本内部用 17.99 秒生成相同 `Hello, world.`。64×64×64 MatMul 条目为 1,056 次调用、合计 3.96 秒；掩蔽 softmax 为 528 次、2.69 秒；残差 LayerNorm 为 102 次、2.25 秒。已插桩张量上传合计 0.42 秒，显式 `.numpy()` 读回合计 0.01 秒。

读回计时很小**不能**证明传输或同步没有成本：内核调用时间可能包含提交、执行与等待输出。这是主机观测的 API 墙钟时间，不是 NPU 硬件计数器，也不是设备计算与传输的分解；API 小计未覆盖插桩调用之间的缓冲区分配、Python/模型工作。它提示下一步优先尝试跨头融合注意力或减少小内核启动，而非缓存主机 BF16 权重。

## 融合注意力

`-FusedAttention` 启用仅用于解码器的实验性 AIE2 内核，计算每个自回归步骤实际消费的单行 query。八个头的 QK、掩蔽 softmax 和 PV 合并为一次 IRON 调用；编码器仍走原始全行路径。运行时 BF16 头部携带有效 key 数，目标长度 1–32 共用同一编译内核设计。64 行输出投影、残差归一化和 FFN 仍在 NPU 上运行，主机仍处理偏置/布局，词表头仍在 CPU。原始路径仍为默认。

`-FusedAttention` 兼容默认 NumPy 验证，检查融合后的第 0 行 context、每层解码器首行及每步目标 argmax。独立 `attention-row0` 测试将有效 key 数 1、5、17、32 与 NumPy 参考比较。

使用缓存构建和 `-NoVerify -Profile`，五 token 的 `你好，世界。` 在脚本内约 10.9 秒生成相同 `Hello, world.` 片段，相邻运行的原逐头路径约 18.0 秒。10 token 的 `人工智能正在改变世界。` 保持相同英文片段，约 17.4–18.7 秒，原路径约 29.8 秒。这些是本地墙钟观察，不是受控硬件基准。首次 JIT 编译可能主导未缓存运行，比较前应预热内核。该实验仍为 NPU/CPU 混合，在这些短句上慢于优化 CPU CTranslate2。

## NPU 词表头

`-NpuVocab` 是向全 NPU 数值路径推进的实验。它将 64 个 BF16 词表权重 tile（每个 1024 列）保持在设备缓冲区，用输出为 FP32 的 NPU MatMul 计算全部 65,000 个 logit。自定义 AIE2 内核加入对应 FP32 偏置，每 tile 只返回两个最佳分数/ID 对。各 tile 写入同一个共享设备缓冲区中按缓存行对齐的视图；第二个 AIE2 内核在 NPU 上合并全部 64 tile 结果。正常解码只从该词表路径读回最终 token ID 和两个分数。

CPU 仍准备嵌入、中间解码器偏置/布局并驱动 token 循环。验证模式额外读回完整 logits，与独立 CPU 参考核对最终 top-2；`-NoVerify` 不读回完整 logits。独立 `vocab-top2` 测试用 NumPy 检查两个归约内核。

`你好，世界。` 的五 token 热运行使用 `-NoVerify -FusedAttention -NpuVocab`，脚本内 13.4 秒生成相同 `Hello, world.`，其中 NPU 词表阶段约 1.54 秒。10 token 的 `人工智能正在改变世界。` 同样逐片段保持一致，总计 22.7 秒，其中词表约 3.06 秒。CPU 词表阶段五 token 仅约 0.02 秒，因此这是功能性 NPU 实验，**不是**速度优化。它仍不是纯 NPU 解码，分词、嵌入、中间主机运算和 Python 生成控制仍保留。这些是排除首次 JIT 编译的本地墙钟时间，不是受控设备基准。

## 仅第 0 行归一化

`-Row0Norm` 是仅面向解码器、显式开启的残差仿射 LayerNorm 版本。现有 AIE2 C++ 内核每次处理一行 512 列；该 IRON 程序只调用一次，不再流过全部 64 行。因为自回归解码只消费第 0 行，主机把结果填充回常规 64×512 接口，只有第 0 行有值。编码器继续归一化全部源行。

独立 `affine-norm-row0` 测试通过 NumPy 校验，最大绝对误差约 0.016；开启该选项的完整五 token `你好，世界。` 也通过默认逐层、逐 token 校验。使用缓存构建和 `-NoVerify -FusedAttention -NpuVocab -Row0Norm`，测试短句在脚本内耗时 11.8 秒（解码器层 7.6 秒），相邻未启用第 0 行归一化运行是 13.4 秒。10 token 的 `人工智能正在改变世界。` 为 20.7 秒，未启用为 22.7 秒。这些观察存在运行间波动，不是保证的加速比或纯设备计时。FFN 和输出投影仍用固定 64 行 MatMul，尽管只消费第 0 行。

## 紧凑 MatMul 探测

`compact-matmul` 用 32 行、16 行 tile，探测原生单核 BF16/FP32 MatMul 的解码器投影形状。四种形状（512×512、512×2048、2048×512、512×1536）在唯一有效输入行上均匹配 NumPy。尝试 8 行/4 行 tile 的变体未编译成功：AIE2 向量 MatMul 要求 tile 高度可被 16 整除，原生输出分块器还需要两个 tile 行。

曾将 32 行路径及填充至 4096 元素的 SiLU 接入解码器 FFN。五 token 的 `你好，世界。` 仍生成相同文本，但约耗时 12.8 秒，相邻原 FFN 运行约 11.8 秒；FFN MatMul 和 SiLU 每次调用仍约 30–40 毫秒。因缩小行数未明显改善时间，移除了该解码器集成；保留独立探测，为未来融合或多核 FFN 提供依据。

## 融合 FFN 后处理

`-FusedFfn` 是可选 AIE2 解码器 FFN 后处理。现有 FP32 上投影输出直接进入自定义内核，加 FP32 偏置、舍入为 BF16，再应用与原生内核相同的查表 SiLU。BF16 输出直接送入下投影，不经主机读回和重新上传。编码器 FFN 及默认解码路径不变。

独立 `ffn-bias-silu` 测试将 64×2048 内核与 NumPy 比较，最大绝对误差 0.03125。`你好，世界。` 的完整五 token 验证通过全部解码器层与目标 argmax；`人工智能正在改变世界。` 的 10 token 运行到达 EOS，并生成预期英文句子。

一次相邻五 token 热运行比较中，使用 `-NoVerify -FusedAttention -NpuVocab -Row0Norm -Profile`，新路径 12.7 秒，原 FFN 11.7 秒。融合内核的 30 次主机调用平均 36.7 毫秒，原 SiLU 调用平均 31.0 毫秒。剖析统计减少了 60 次张量读回和 30 次上传，但墙钟时间未改善。这些是主机观测本地计时，不是 NPU 硬件计数器。其他投影偏置、布局、嵌入和 token 控制仍由 CPU 执行，因此不是纯 NPU 解码器。
