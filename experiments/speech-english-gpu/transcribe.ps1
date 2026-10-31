param(
    [Parameter(Mandatory = $true)]
    [string]$AudioPath,
    [switch]$Cpu
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSCommandPath
$exe = Join-Path $root 'whisper-1.8.4-windows-x64\whisper-cli.exe'
$model = Join-Path $root 'ggml-small.en.bin'
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "Whisper executable missing: $exe"
}
if (-not (Test-Path -LiteralPath $model -PathType Leaf)) {
    throw "Whisper model missing: $model"
}

$audio = (Resolve-Path -LiteralPath $AudioPath -ErrorAction Stop).Path
$outputDir = Join-Path $root 'results'
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$device = if ($Cpu) { 'cpu' } else { 'vulkan' }
$name = [IO.Path]::GetFileNameWithoutExtension($audio)
$outputBase = Join-Path $outputDir ("{0}-{1}-{2}" -f $name, $device, (Get-Date -Format 'yyyyMMdd-HHmmss'))

$arguments = @('-m', $model, '-f', $audio, '-l', 'en', '-otxt', '-of', $outputBase)
if ($Cpu) {
    $arguments += '-ng'
}

& $exe @arguments 2>&1 | Tee-Object -Variable runLog
if ($LASTEXITCODE -ne 0) {
    throw "Whisper exited with code $LASTEXITCODE"
}
if (-not $Cpu -and -not (($runLog | Out-String) -match 'using Vulkan0 backend')) {
    throw 'Transcription finished, but Vulkan execution was not confirmed in the log.'
}
Write-Output "Transcript: $outputBase.txt"
