$ErrorActionPreference = "Stop"

$conda = Join-Path $env:USERPROFILE "miniforge3\Scripts\conda.exe"
if (-not (Test-Path -LiteralPath $conda)) {
    throw "Miniforge was not found at $conda"
}
if (-not $env:RYZEN_AI_INSTALLATION_PATH) {
    $env:RYZEN_AI_INSTALLATION_PATH = "C:\Program Files\RyzenAI\1.8.0"
}

& $conda run --no-capture-output -n amd-npu-vitis17 python "$PSScriptRoot\run_vitisai.py"
exit $LASTEXITCODE
