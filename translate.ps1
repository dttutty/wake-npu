$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Translation environment is missing at $python"
}

$previousEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONIOENCODING = "utf-8"
    & $python (Join-Path $PSScriptRoot "translate.py") @args
    $code = $LASTEXITCODE
} finally {
    $env:PYTHONIOENCODING = $previousEncoding
}
exit $code
