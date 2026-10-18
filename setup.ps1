# =============================================================================
#  Discord Gen-A Brainrot Bot — environment setup
# -----------------------------------------------------------------------------
#  Creates a Miniconda environment 'brainrot_env' (Python 3.12) and installs
#  EVERY dependency into it (Unsloth + PyTorch/CUDA for training, discord.py +
#  ollama + python-dotenv for the bot).
#
#  How to run (in a normal PowerShell window, no admin needed):
#      cd "C:\Users\shane\OneDrive\Desktop\Coding\Discord Gen-A Brainrot Bot"
#      powershell -ExecutionPolicy Bypass -File .\setup.ps1
#
#  The first run downloads several GB (PyTorch is large). Be patient.
# =============================================================================

$ErrorActionPreference = 'Stop'

$EnvName   = 'brainrot_env'
$PythonVer = '3.12'

Write-Host ''
Write-Host '================================================' -ForegroundColor Cyan
Write-Host '  Discord Gen-A Brainrot Bot - environment setup' -ForegroundColor Cyan
Write-Host '================================================' -ForegroundColor Cyan
Write-Host ''

# ---- 1) locate conda ------------------------------------------------------
$conda = $null
$candidates = @(
    (Get-Command conda -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source),
    (Join-Path $env:USERPROFILE 'miniconda3\Scripts\conda.exe'),
    (Join-Path $env:USERPROFILE 'anaconda3\Scripts\conda.exe'),
    (Join-Path $env:USERPROFILE 'miniconda3\condabin\conda.bat')
)
foreach ($c in $candidates) {
    if ($c -and (Test-Path $c)) { $conda = $c; break }
}
if (-not $conda) {
    Write-Host '[!] Conda not found. Install Miniconda first:' -ForegroundColor Red
    Write-Host '    https://docs.anaconda.com/miniconda/' -ForegroundColor Yellow
    Write-Host '    Then re-run this script.' -ForegroundColor Red
    exit 1
}
Write-Host "[1/5] Conda found: $conda" -ForegroundColor Green

# ---- 2) create env if missing ---------------------------------------------
$envList = & $conda env list 2>$null
$envExists = ($envList | ForEach-Object { ($_ -split '\s+')[0] }) -contains $EnvName
if ($envExists) {
    Write-Host "[2/5] Env '$EnvName' already exists - reusing." -ForegroundColor Yellow
} else {
    Write-Host "[2/5] Creating env '$EnvName' (Python $PythonVer)... this can take a minute." -ForegroundColor Green
    & $conda create -n $EnvName python=$PythonVer -y
    if ($LASTEXITCODE -ne 0) { Write-Host '[!] conda create failed.' -ForegroundColor Red; exit 1 }
}

# ---- 3) resolve the env's python.exe --------------------------------------
$base = (& $conda info --base).Trim()
$envPy = Join-Path $base "envs\$EnvName\python.exe"
if (-not (Test-Path $envPy)) {
    Write-Host "[!] Could not locate env python: $envPy" -ForegroundColor Red
    exit 1
}
Write-Host "[3/5] Env python: $envPy" -ForegroundColor Green

# ---- 4) install dependencies ----------------------------------------------
Write-Host '[4/5] Installing dependencies (downloads several GB - be patient)...' -ForegroundColor Green

& $envPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Write-Host '[!] pip upgrade failed.' -ForegroundColor Red; exit 1 }

# PyTorch's default PyPI wheel is now CPU-only, so install the CUDA build
# explicitly from the index matching the NVIDIA driver's CUDA version.
$smi = (& nvidia-smi 2>$null | Out-String)
$m = [regex]::Match($smi, 'CUDA Version:\s*(\d+)\.(\d+)')
$major = 12; $minor = 8   # safe default if detection fails
if ($m.Success) { $major = [int]$m.Groups[1].Value; $minor = [int]$m.Groups[2].Value }
if ($major -ge 13)                       { $idx = 'cu130' }
elseif ($major -eq 12 -and $minor -ge 8) { $idx = 'cu128' }
elseif ($major -eq 12 -and $minor -ge 6) { $idx = 'cu126' }
elseif ($major -eq 12 -and $minor -ge 4) { $idx = 'cu124' }
elseif ($major -eq 12)                   { $idx = 'cu121' }
else                                     { $idx = 'cu118' }
Write-Host "    Driver CUDA $major.$minor -> torch index: $idx" -ForegroundColor Cyan
& $envPy -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$idx"
if ($LASTEXITCODE -ne 0) { Write-Host '[!] torch install failed.' -ForegroundColor Red; exit 1 }

# Unsloth pulls transformers, TRL, PEFT, accelerate, bitsandbytes, and datasets
# automatically at compatible versions (it keeps the torch we just installed).
& $envPy -m pip install unsloth
if ($LASTEXITCODE -ne 0) { Write-Host '[!] unsloth install failed.' -ForegroundColor Red; exit 1 }

# Bot runtime dependencies.
& $envPy -m pip install 'discord.py>=2.3' ollama python-dotenv
if ($LASTEXITCODE -ne 0) { Write-Host '[!] bot deps install failed.' -ForegroundColor Red; exit 1 }

# ---- 5) verify ------------------------------------------------------------
Write-Host '[5/5] Verifying the environment...' -ForegroundColor Green
& $envPy (Join-Path $PSScriptRoot 'verify_env.py')
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host '[!] Verification failed - see "Troubleshooting" in README.md' -ForegroundColor Red
    exit 1
}

Write-Host ''
Write-Host 'Setup complete!' -ForegroundColor Green
Write-Host ''
Write-Host '  Next steps:' -ForegroundColor Cyan
Write-Host "    conda activate $EnvName" -ForegroundColor White
Write-Host '    python train.py --smoke     (quick 20-step test)' -ForegroundColor White
Write-Host '    python train.py             (full training run)' -ForegroundColor White
