param(
    [string[]]$Text,
    [ValidateRange(1, 20)][int]$Repeats = 2,
    [ValidateRange(1, 64)][int]$MaxTokens,
    [ValidateSet(32, 64)][int]$Length = 32,
    [switch]$RecordMismatches,
    [switch]$Fp16,
    [switch]$SkipNpu,
    [switch]$Rebuild,
    [string]$Report,
    [string]$GpuPython = (Join-Path $env:USERPROFILE 'miniforge3\envs\amd-npu\python.exe')
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $GpuPython)) {
    throw "Python with ONNX and ONNX Runtime DirectML was not found: $GpuPython"
}
$runArgs = @((Join-Path $PSScriptRoot 'benchmark.py'), '--repeats', $Repeats, '--length', $Length)
if ($PSBoundParameters.ContainsKey('MaxTokens')) { $runArgs += @('--max-tokens', $MaxTokens) }
if ($RecordMismatches) { $runArgs += '--record-mismatches' }
foreach ($sentence in $Text) { $runArgs += @('--text', $sentence) }
if ($Fp16) { $runArgs += '--fp16' }
if ($SkipNpu) { $runArgs += '--skip-npu' }
if ($Rebuild) { $runArgs += '--export' }
if ($Report) { $runArgs += @('--report', $Report) }
$previousEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONIOENCODING = 'utf-8'
    & $GpuPython @runArgs
    if ($LASTEXITCODE -ne 0) { throw "Hybrid experiment failed (exit code $LASTEXITCODE)" }
} finally {
    $env:PYTHONIOENCODING = $previousEncoding
}
