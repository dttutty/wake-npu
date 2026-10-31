$ErrorActionPreference = "Stop"

$conda = Join-Path $env:USERPROFILE "miniforge3\Scripts\conda.exe"
if (-not (Test-Path -LiteralPath $conda)) {
    throw "Miniforge was not found at $conda"
}

& $conda run --no-capture-output -n amd-npu python "$PSScriptRoot\run_npu.py"
exit $LASTEXITCODE
