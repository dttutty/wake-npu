param(
    [ValidateSet('vector', 'matmul', 'attention', 'gelu', 'add', 'layernorm', 'block', 'real-weight', 'real-attention', 'real-encoder-layer', 'real-encoder-stack', 'real-decoder-layer', 'real-first-token', 'real-greedy', 'softmax64', 'attention-row0', 'vocab-top2', 'affine-norm512', 'affine-norm-row0', 'matmul-shapes', 'compact-matmul', 'ffn-bias-silu')]
    [string]$Example = 'vector',
    [int]$Seed = 2026,
    [string]$Text = '',
    [ValidateRange(1, 32)][int]$MaxTokens = 16,
    [switch]$NoVerify,
    [switch]$Profile,
    [switch]$FusedAttention,
    [switch]$NpuVocab,
    [switch]$Row0Norm,
    [switch]$FusedFfn
)

$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$ironRoot = Join-Path $root 'third_party\mlir-aie'
$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$ironEnv = Join-Path $ironRoot 'iron_env.cmd'

foreach ($required in @($vcvars, $ironEnv)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required setup file not found: $required"
    }
}

$env:NPU_CACHE_HOME = Join-Path $root 'tools\iron-cache'
if ($Example -eq 'matmul') {
    $exampleDir = Join-Path $ironRoot 'programming_examples\basic\matrix_multiplication\single_core'
    $runCommand = 'python single_core.py -M 64 -K 64 -N 64 -m 32 -k 32 -n 32 --dtype_in i8 --dtype_out i32 --warmup 2 --iters 10'
} elseif ($Example -eq 'attention') {
    $exampleDir = Join-Path $root 'experiments\iron-attention'
    $runCommand = 'python run.py'
} elseif ($Example -eq 'layernorm') {
    $exampleDir = Join-Path $root 'experiments\iron-layernorm'
    $runCommand = 'python run.py'
} elseif ($Example -eq 'block') {
    $exampleDir = Join-Path $root 'experiments\iron-transformer-block'
    $runCommand = "python run.py --seed $Seed"
} elseif ($Example -eq 'real-weight') {
    $exampleDir = Join-Path $root 'experiments\iron-real-weight'
    $runCommand = 'python run.py'
} elseif ($Example -eq 'real-attention') {
    $exampleDir = Join-Path $root 'experiments\iron-real-weight'
    $runCommand = 'python attention.py'
} elseif ($Example -eq 'real-encoder-layer') {
    $exampleDir = Join-Path $root 'experiments\iron-real-weight'
    $runCommand = 'python encoder_layer.py'
} elseif ($Example -eq 'real-encoder-stack') {
    $exampleDir = Join-Path $root 'experiments\iron-real-weight'
    $runCommand = 'python encoder_stack.py'
} elseif ($Example -eq 'real-decoder-layer') {
    $exampleDir = Join-Path $root 'experiments\iron-real-weight'
    $runCommand = 'python decoder_layer.py'
} elseif ($Example -eq 'real-first-token') {
    $exampleDir = Join-Path $root 'experiments\iron-real-weight'
    $runCommand = 'python decoder_first_token.py'
} elseif ($Example -eq 'real-greedy') {
    $exampleDir = Join-Path $root 'experiments\iron-real-weight'
    $runCommand = "python greedy_decode.py --max-tokens $MaxTokens"
    if ($NoVerify) {
        $runCommand += ' --no-verify'
    }
    if ($Profile) {
        $runCommand += ' --profile'
    }
    if ($FusedAttention) {
        $runCommand += ' --fused-attention'
    }
    if ($NpuVocab) {
        $runCommand += ' --npu-vocab'
    }
    if ($Row0Norm) {
        $runCommand += ' --row0-norm'
    }
    if ($FusedFfn) {
        $runCommand += ' --fused-ffn'
    }
} elseif ($Example -eq 'softmax64') {
    $exampleDir = Join-Path $root 'experiments\iron-real-weight'
    $runCommand = 'python softmax64.py'
} elseif ($Example -eq 'attention-row0') {
    $exampleDir = Join-Path $root 'experiments\iron-real-weight'
    $runCommand = 'python attention_row0.py'
} elseif ($Example -eq 'vocab-top2') {
    $exampleDir = Join-Path $root 'experiments\iron-real-weight'
    $runCommand = 'python vocab_top2.py'
} elseif ($Example -eq 'affine-norm512') {
    $exampleDir = Join-Path $root 'experiments\iron-real-weight'
    $runCommand = 'python affine_norm512.py'
} elseif ($Example -eq 'affine-norm-row0') {
    $exampleDir = Join-Path $root 'experiments\iron-real-weight'
    $runCommand = 'python affine_norm_row0.py'
} elseif ($Example -eq 'matmul-shapes') {
    $exampleDir = Join-Path $root 'experiments\iron-real-weight'
    $runCommand = 'python matmul_probe.py'
} elseif ($Example -eq 'compact-matmul') {
    $exampleDir = Join-Path $root 'experiments\iron-real-weight'
    $runCommand = 'python compact_matmul_probe.py'
} elseif ($Example -eq 'ffn-bias-silu') {
    $exampleDir = Join-Path $root 'experiments\iron-real-weight'
    $runCommand = 'python ffn_bias_silu.py'
} elseif ($Example -eq 'gelu') {
    $exampleDir = Join-Path $ironRoot 'programming_examples\ml\eltwise_unary'
    $runCommand = 'python eltwise_unary.py -l 4096 -ch 1 -o gelu'
} elseif ($Example -eq 'add') {
    $exampleDir = Join-Path $ironRoot 'programming_examples\ml\eltwise'
    $runCommand = 'python eltwise.py -l 4096 -ch 1 -o add'
} else {
    $exampleDir = $PSScriptRoot
    $runCommand = 'python run.py'
}
$command = 'call "{0}" >nul && call "{1}" && set PYTHONIOENCODING=utf-8 && cd /d "{2}" && {3}' -f $vcvars, $ironEnv, $exampleDir, $runCommand
$previousTestText = $env:NPU_TEST_TEXT
try {
    if ($Example -in @('real-weight', 'real-attention', 'real-encoder-layer', 'real-encoder-stack', 'real-decoder-layer', 'real-first-token', 'real-greedy') -and $Text) {
        $env:NPU_TEST_TEXT = $Text
    }
    & cmd.exe /d /s /c $command
} finally {
    $env:NPU_TEST_TEXT = $previousTestText
}
if ($LASTEXITCODE -ne 0) {
    throw "IRON run failed (exit code $LASTEXITCODE)"
}
