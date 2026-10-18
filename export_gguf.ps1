# =============================================================================
#  Discord Gen-A Brainrot Bot - convert the merged model to GGUF (v2)
# -----------------------------------------------------------------------------
#  Unsloth's built-in exporter and the single-file converter both fail on
#  Windows, so this clones llama.cpp (whose converter is now a Python package)
#  and runs convert_hf_to_gguf.py from inside it. HF -> GGUF conversion is pure
#  Python - NO compilation needed.
#
#  Run:  powershell -ExecutionPolicy Bypass -File .\export_gguf.ps1
# =============================================================================
# Valid single-step outtypes: f32, f16, bf16, q8_0, tq1_0, tq2_0, auto.
#   q8_0 = 8-bit, ~3.3 GB for a 3B model, fits the 6 GB GPU, high quality.
#   (q4_k_m and other k-quants now require a SECOND step with the compiled
#    llama-quantize binary, so we avoid them here to skip compilation entirely.)
param([string]$OutType = 'q8_0')

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$EnvName = 'brainrot_env'

# ---- resolve the env python ------------------------------------------------
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
    Write-Host '[!] env python not found - run setup.ps1 first.' -ForegroundColor Red
    exit 1
}

$mergedAbs = Join-Path $PSScriptRoot 'merged'
if (-not (Test-Path (Join-Path $mergedAbs 'config.json'))) {
    Write-Host '[!] merged/model not found. Run train.py first.' -ForegroundColor Red
    exit 1
}

# remove the old broken single-file converter left by the previous attempt
Remove-Item (Join-Path $PSScriptRoot 'convert_hf_to_gguf.py') -ErrorAction SilentlyContinue

# ---- 1) install conversion deps (torch/numpy/transformers already present) --
Write-Host '[1/3] Installing conversion deps...' -ForegroundColor Green
& $envPy -m pip install --no-cache-dir gguf protobuf sentencepiece tiktoken blobfile safetensors loguru
if ($LASTEXITCODE -ne 0) {
    Write-Host '[!] pip install failed.' -ForegroundColor Red
    exit 1
}

# ---- 2) clone llama.cpp (shallow) ------------------------------------------
$llamaDir = Join-Path $PSScriptRoot 'llama.cpp'
if (Test-Path (Join-Path $llamaDir 'convert_hf_to_gguf.py')) {
    Write-Host '[2/3] llama.cpp already present.' -ForegroundColor Green
} else {
    if (Test-Path $llamaDir) { Remove-Item -Recurse -Force $llamaDir }
    Write-Host '[2/3] Cloning llama.cpp (shallow clone, ~1 min)...' -ForegroundColor Green
    git clone --depth 1 https://github.com/ggml-org/llama.cpp $llamaDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host '[!] git clone failed - check your internet connection.' -ForegroundColor Red
        exit 1
    }
}

# ---- 3) run the converter from inside the llama.cpp folder -----------------
$outDir = Join-Path $PSScriptRoot 'gguf'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$outFile = Join-Path $outDir "brainrot.$($OutType.ToUpper()).gguf"

Write-Host "[3/3] Converting merged -> GGUF ($OutType). A few minutes..." -ForegroundColor Green
Push-Location $llamaDir
try {
    & $envPy .\convert_hf_to_gguf.py $mergedAbs --outtype $OutType --outfile $outFile
    if ($LASTEXITCODE -ne 0) { throw "converter exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}

Write-Host ''
Write-Host "GGUF written to: $outFile" -ForegroundColor Green
Write-Host 'Next: powershell -ExecutionPolicy Bypass -File .\create_ollama.ps1' -ForegroundColor Cyan
