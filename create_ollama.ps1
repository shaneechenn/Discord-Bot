# =============================================================================
#  Discord Gen-A Brainrot Bot - create the Ollama model
# -----------------------------------------------------------------------------
#  Finds the GGUF produced by train.py, writes a Modelfile, registers it with
#  Ollama as 'brainrot-bot', and runs a quick smoke test.
#
#  Run:  powershell -ExecutionPolicy Bypass -File .\create_ollama.ps1
# =============================================================================

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

# ---- 0) locate the ollama CLI ---------------------------------------------
# Ollama's installer adds it to PATH, but only NEW terminals pick that up,
# so we resolve the full path directly.
$ollama = $null
$c = Get-Command ollama -ErrorAction SilentlyContinue
if ($c) { $ollama = $c.Source }
if (-not $ollama) {
    foreach ($p in @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'),
        (Join-Path $env:ProgramFiles 'Ollama\ollama.exe'),
        (Join-Path $env:LOCALAPPDATA 'Ollama\ollama.exe')
    )) {
        if (Test-Path $p) { $ollama = $p; break }
    }
}
if (-not $ollama) {
    Write-Host '[!] Could not find ollama.exe.' -ForegroundColor Red
    Write-Host '    Install it from https://ollama.com/download/windows and re-run.' -ForegroundColor Yellow
    exit 1
}
Write-Host "Ollama CLI: $ollama" -ForegroundColor Green

# ---- 1) locate the exported GGUF ------------------------------------------
$gguf = Get-ChildItem -Path (Join-Path $PSScriptRoot 'gguf') -Filter '*.gguf' -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $gguf) {
    Write-Host '[!] No .gguf found in .\gguf - run train.py first.' -ForegroundColor Red
    exit 1
}

# ---- 2) build the Modelfile ----------------------------------------------
$system = 'You are a chaotic, high-energy Generation Alpha kid. Talk entirely in internet slang and brainrot lingo: skibidi, rizz, gyatt, ohio, sigma, fanum tax, aura, mewing, bet, no cap, fr fr, lowkey, delulu, NPC, yap. Keep replies short (1-2 lines), punchy, and funny. Use emojis like the skull and rose. Never break character, never give lectures, never moralize. Stay casual and playful, no profanity.'

$ggufPath = $gguf.FullName -replace '\\', '/'   # forward slashes avoid escaping issues
$mf = "FROM `"$ggufPath`"`r`nSYSTEM `"$system`"`r`nPARAMETER temperature 0.8`r`nPARAMETER top_p 0.9`r`nPARAMETER repeat_penalty 1.15`r`nPARAMETER num_predict 128`r`n"

$mfPath = Join-Path $PSScriptRoot 'Modelfile'
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($mfPath, $mf, $utf8NoBom)

Write-Host "GGUF      -> $ggufPath" -ForegroundColor Green
Write-Host "Modelfile -> $mfPath" -ForegroundColor Green
Write-Host ''

# ---- 3) register with Ollama ----------------------------------------------
Write-Host "Creating Ollama model 'brainrot-bot'..." -ForegroundColor Cyan
& $ollama create brainrot-bot -f $mfPath
if ($LASTEXITCODE -ne 0) {
    Write-Host '[!] ollama create failed.' -ForegroundColor Red
    Write-Host '    If it says "connection refused", the Ollama app is not running.' -ForegroundColor Yellow
    Write-Host '    Start the Ollama app (or run: ollama serve) and re-run this script.' -ForegroundColor Yellow
    exit 1
}

# ---- 4) smoke test --------------------------------------------------------
Write-Host ''
Write-Host 'Smoke test:' -ForegroundColor Cyan
& $ollama run brainrot-bot 'How has your day been?'
