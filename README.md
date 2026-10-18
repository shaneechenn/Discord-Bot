# Discord Gen-A Brainrot Bot

Fine-tune **Qwen2.5-3B-Instruct** with **Unsloth QLoRA** so it talks like a
chaotic, high-energy **Generation Alpha "brainrot" kid**, then serve it locally
with **Ollama** and connect it to Discord.

Example: `"How has your day been?"` → `"its been mad chopped bru 💀"`

---

## Architecture & the Python-vs-C++ decision

| Layer | Language | Why |
|---|---|---|
| **Training** (QLoRA) | Python | Unsloth/TRL/PyTorch are Python-only; the heavy math already runs in CUDA/Triton **C++** kernels, so Python isn't the bottleneck. |
| **Inference** (tokens/sec) | **C++** | Served by **Ollama → llama.cpp**, a C++ engine. This is the only latency-critical part, and it's already C++. |
| **Discord bot** (glue) | Python | The bot is **I/O-bound** (network + waiting on the local LLM); async `discord.py` handles thousands of msgs/sec. A C++ bot (D++) would save microseconds of CPU while costing days of work. |

**Conclusion:** write Python for the pieces you own, and lean on C++ where it's
already done for you (llama.cpp). That is exactly what this project does.

### Efficiency levers used
- **QLoRA 4-bit (NF4)** — 3B model trains in ~3.5 GB VRAM (fits a 6 GB 4050).
- **Unsloth** kernels — ~2× faster training, ~60% less VRAM than vanilla PEFT.
- `use_gradient_checkpointing="unsloth"` and `optim="adamw_8bit"` — less memory.
- **GGUF Q8_0** — fast, low-VRAM inference in llama.cpp.
- `num_ctx=512` + a 16-message sliding window — small context = fast replies.
- One persistent async Ollama client + per-channel rate limiting.

---

## Prerequisites

1. **Windows 10/11**, NVIDIA driver installed (RTX 4050 already has one).
2. **Miniconda** — already installed on this machine.
3. **Git** — already installed.
4. **Ollama** — *not yet installed*. Get it here:
   <https://ollama.com/download/windows>  (run it once after install)
5. **A Discord bot token** — see step 0 below.

---

## Quick start (5 commands)

Open **PowerShell** and run:

```powershell
cd "C:\Users\shane\OneDrive\Desktop\Coding\Discord Gen-A Brainrot Bot"

# 1) create the conda env + install everything (downloads several GB)
powershell -ExecutionPolicy Bypass -File .\setup.ps1

# 2) train (smoke test first, then the real thing)
powershell -ExecutionPolicy Bypass -File .\run.ps1 train.py --smoke
powershell -ExecutionPolicy Bypass -File .\run.ps1 train.py

# 3) convert the merged model to GGUF (Unsloth's built-in GGUF export fails on Windows)
powershell -ExecutionPolicy Bypass -File .\export_gguf.ps1

# 4) register the model with Ollama + smoke test
powershell -ExecutionPolicy Bypass -File .\create_ollama.ps1

# 5) run the Discord bot (after copying .env.example -> .env and adding your token)
powershell -ExecutionPolicy Bypass -File .\run.ps1 bot.py
```

> **Why not `conda activate`?** If your prompt doesn't show `(brainrot_env)`,
> conda isn't initialized in this shell, so `python` still points at the Windows
> Store stub. `run.ps1` calls the env's Python by full path, so it always works.
> To make `conda activate` work normally instead, run `conda init powershell`
> once and restart the terminal.

---

## Step 0 — Create the Discord bot token

1. Go to <https://discord.com/developers/applications> → **New Application**.
2. Name it (e.g. "Brainrot Bot").
3. **Bot** tab → **Reset Token** → copy the token.
4. Under **Privileged Gateway Intents**, enable **Message Content Intent** (required).
5. **OAuth2 → URL Generator** → scope `bot` → permissions `Send Messages`,
   `Read Message History`, `Use Slash Commands`, `Add Reactions` → copy the URL
   and open it to invite the bot to your server.
6. In this project: `copy .env.example .env` and paste your token into `DISCORD_TOKEN=`.

---

## Step 1 — `setup.ps1` (conda env + dependencies)

Creates a **Miniconda** env named `brainrot_env` (Python 3.12) and installs:
`unsloth` (which pulls PyTorch/CUDA + transformers + TRL + PEFT + accelerate +
bitsandbytes + datasets) plus `discord.py`, `ollama`, and `python-dotenv`.
It finishes by running `verify_env.py` to confirm the GPU is visible.

> If `conda activate` is not recognized, run `conda init powershell`, restart the
> terminal, then retry. Fallback (no activation needed) — use the env's Python
> directly: `C:\Users\shane\miniconda3\envs\brainrot_env\python.exe train.py`.

---

## Step 2 — `train.py` (QLoRA fine-tuning)

- Base: `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (pre-quantized 4-bit).
- Data: `grenishrai/brainrot-conversation` + `Andy-ML-And-AI/gen-alpha-brainrot`
  (downloaded as raw JSONL automatically — no manual download step).
- LoRA: `r=16, alpha=32`, all linear layers, ~2,000 examples, 3 epochs,
  seq length 512, batch 2 × grad-accum 4, LR 2e-4.

On a 4050 this should finish in **well under an hour**. Outputs:
- `outputs/adapter/` — the LoRA adapter (for later re-use).
- `merged/` — LoRA merged into the 16-bit base.
- `gguf/` — `*.Q8_0.gguf` for Ollama (produced by `export_gguf.ps1`).

`python train.py --smoke` runs only 20 steps to catch config/import errors first.

---

## Step 3 — `create_ollama.ps1` (serve with Ollama)

Finds the GGUF, writes a `Modelfile` (model + persona system prompt), registers
it as `brainrot-bot`, and runs a smoke test. The chat template is embedded in
the GGUF by Unsloth, so Ollama picks it up automatically.

Manual equivalent:
```powershell
ollama create brainrot-bot -f Modelfile
ollama run brainrot-bot "How has your day been?"
```

---

## Step 4 — `bot.py` (Discord client)

- Triggers on **@mention** or the `!brainrot` prefix.
- `!brainrot help` shows a usage menu; a bare `!brainrot` (or @mention) also shows it.
- `!brainrot reset` clears the channel's memory.
- Keeps a 16-message sliding window **per channel**.
- 3 s rate limit per channel; replies capped to Discord's 2000-char limit.

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 bot.py
```

---

## Project files

| File | Purpose |
|---|---|
| `setup.ps1` | Create conda env + install all deps (CUDA torch) + verify |
| `fix_torch_cuda.ps1` | Replace CPU-only torch with a CUDA build matching your driver |
| `run.ps1` | Run any script with the env's Python (no `conda activate` needed) |
| `verify_env.py` | Check imports, versions, CUDA/GPU |
| `train.py` | QLoRA fine-tune + merge (+ best-effort GGUF) |
| `export_gguf.ps1` | Convert `merged/` to GGUF with llama.cpp (reliable on Windows) |
| `create_ollama.ps1` | Build Modelfile + `ollama create` + smoke test |
| `bot.py` | Discord bot |
| `.env.example` | Token/model/host template |
| `requirements.txt` | Reference list (installed by setup.ps1) |

---

## Troubleshooting

**`torch` installed but `CUDA available: False` (shows `+cpu`)**
PyTorch's default PyPI wheel is now CPU-only, so a plain `pip install unsloth`
can pull `torch x.x.x+cpu`. Your GPU is fine — you just need the CUDA build.
Run the fix script, which auto-detects your driver's CUDA version:
```powershell
powershell -ExecutionPolicy Bypass -File .\fix_torch_cuda.ps1
```
To force a specific CUDA index (if the auto-detected one has no matching
wheel), pass it as an argument, e.g. `.\fix_torch_cuda.ps1 cu124`.

**`save_pretrained_gguf` failed (Unsloth's Windows llama.cpp installer)**
This is a known Windows issue and does NOT mean training failed — the merged
model is already saved in `merged/`. Use the dedicated exporter instead:
```powershell
powershell -ExecutionPolicy Bypass -File .\export_gguf.ps1
```
It installs the conversion deps, downloads llama.cpp's `convert_hf_to_gguf.py`,
and produces `gguf\brainrot.Q4_K_M.gguf`. Then run `create_ollama.ps1`.

If `export_gguf.ps1` can't download the converter, clone llama.cpp instead:
```powershell
git clone --depth 1 https://github.com/ggml-org/llama.cpp
C:\Users\shane\miniconda3\envs\brainrot_env\python.exe -m pip install -r llama.cpp\requirements.txt
C:\Users\shane\miniconda3\envs\brainrot_env\python.exe llama.cpp\convert_hf_to_gguf.py merged --outtype q8_0 --outfile gguf\brainrot.Q8_0.gguf
```

**`ollama create` can't find the GGUF (spaces in path)**
Copy the `.gguf` into the project root and change the `FROM` line to
`FROM ./brainrot.Q4_K_M.gguf`, then re-run.

**Replies are off-style / too polite / too long**
Lower `EPOCHS` to 2 (overfitting), confirm `MAX_SEQ_LENGTH=512`, and/or add a
few dozen of your own brainrot lines to the training data.

**OneDrive is syncing the multi-GB model files**
The conda env (`C:\Users\shane\miniconda3\envs`) and HF cache
(`C:\Users\shane\.cache\huggingface`) are already outside OneDrive. To stop the
`merged/`, `gguf/`, and `outputs/` folders from syncing, either add them to
OneDrive's "exclude folders" list, or move this whole project folder to
`C:\brainrot-bot` and re-run there.

---

## Extending

- **More slang / your own voice:** append lines to a local `brainrot_extra.jsonl`
  (same `{"messages":[...]}` format) and add its path to `DATA_URLS` in `train.py`.
- **Hot-swap adapters:** keep the LoRA adapter (`outputs/adapter/`) and load it
  with `ADAPTER` in a Modelfile instead of merging.
- **Bigger model later:** a 7B QLoRA (~5 GB) *can* run on 6 GB with batch size 1,
  but 3B is the safe sweet spot for this card.
