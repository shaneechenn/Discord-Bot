# =============================================================================
#  Discord Gen-A Brainrot Bot - fix CPU-only PyTorch
# -----------------------------------------------------------------------------
#  setup.ps1 pulled torch 2.x "+cpu" (the new PyPI default). This replaces it
#  with a CUDA build matched to your NVIDIA driver so training uses the RTX 4050.
#
#  Run:  powershell -ExecutionPolicy Bypass -File .\fix_torch_cuda.ps1
#        (optionally force an index:  .\fix_torch_cuda.ps1 cu128)
# =============================================================================
param([string]$Index = "")

$ErrorActionPreference = 'Stop'

$EnvName = 'brainrot_env'

# ---- locate conda + env python --------------------------------------------
$conda = $null
foreach ($c in @(
    (Get-Command conda -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source),
    (Join-Path $env:USERPROFILE 'miniconda3\Scripts\conda.exe'),
    (Join-Path $env:USERPROFILE 'anaconda3\Scripts\conda.exe')
)) {
    if ($c -and (Test-Path $c)) { $conda = $c; break }
}
if (-not $conda) { Write-Host '[!] conda not found.' -ForegroundColor Red; exit 1 }

$base = (& $conda info --base).Trim()
$envPy = Join-Path $base "envs\$EnvName\python.exe"
if (-not (Test-Path $envPy)) {
    Write-Host "[!] env python not found: $envPy" -ForegroundColor Red
    exit 1
}

# ---- pick the CUDA torch index --------------------------------------------
if (-not $Index) {
    $smi = (& nvidia-smi 2>$null | Out-String)
    $m = [regex]::Match($smi, 'CUDA Version:\s*(\d+)\.(\d+)')
    $major = 12; $minor = 8   # safe default if detection fails
    if ($m.Success) { $major = [int]$m.Groups[1].Value; $minor = [int]$m.Groups[2].Value }

    if ($major -ge 13)                       { $Index = 'cu130' }
    elseif ($major -eq 12 -and $minor -ge 8) { $Index = 'cu128' }
    elseif ($major -eq 12 -and $minor -ge 6) { $Index = 'cu126' }
    elseif ($major -eq 12 -and $minor -ge 4) { $Index = 'cu124' }
    elseif ($major -eq 12)                   { $Index = 'cu121' }
    else                                     { $Index = 'cu118' }

    Write-Host "Driver CUDA: $major.$minor" -ForegroundColor Cyan
}
Write-Host "Torch index : $Index" -ForegroundColor Cyan

# ---- swap CPU torch for CUDA torch ----------------------------------------
Write-Host 'Removing CPU-only torch...' -ForegroundColor Green
& $envPy -m pip uninstall -y torch torchvision torchaudio

Write-Host "Installing CUDA torch from https://download.pytorch.org/whl/$Index ..." -ForegroundColor Green
& $envPy -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$Index"
if ($LASTEXITCODE -ne 0) {
    Write-Host '[!] torch install failed. If it said "no matching distribution",' -ForegroundColor Red
    Write-Host '    try a lower index, e.g.:  .\fix_torch_cuda.ps1 cu124' -ForegroundColor Yellow
    exit 1
}

# ---- verify ----------------------------------------------------------------
Write-Host 'Verifying...' -ForegroundColor Green
& $envPy (Join-Path $PSScriptRoot 'verify_env.py')
