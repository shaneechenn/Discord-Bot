# =============================================================================
#  Discord Gen-A Brainrot Bot - run a script with the conda env's Python
# -----------------------------------------------------------------------------
#  No conda activation needed - this resolves the env's python.exe and runs it.
#
#  Usage:
#      powershell -ExecutionPolicy Bypass -File .\run.ps1 train.py --smoke
#      powershell -ExecutionPolicy Bypass -File .\run.ps1 train.py
#      powershell -ExecutionPolicy Bypass -File .\run.ps1 bot.py
# =============================================================================
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Script,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$RestArgs
)

$ErrorActionPreference = 'Stop'
$EnvName = 'brainrot_env'

# ---- locate conda, then the env python ------------------------------------
$conda = $null
foreach ($c in @(
    (Get-Command conda -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source),
    (Join-Path $env:USERPROFILE 'miniconda3\Scripts\conda.exe'),
    (Join-Path $env:USERPROFILE 'anaconda3\Scripts\conda.exe')
)) {
    if ($c -and (Test-Path $c)) { $conda = $c; break }
}

if ($conda) {
    $base = (& $conda info --base).Trim()
    $envPy = Join-Path $base "envs\$EnvName\python.exe"
} else {
    $envPy = Join-Path $env:USERPROFILE "miniconda3\envs\$EnvName\python.exe"
}

if (-not (Test-Path $envPy)) {
    Write-Host "[!] Env python not found: $envPy" -ForegroundColor Red
    Write-Host '    Run setup.ps1 first.' -ForegroundColor Yellow
    exit 1
}

Set-Location $PSScriptRoot
& $envPy $Script @RestArgs
exit $LASTEXITCODE
