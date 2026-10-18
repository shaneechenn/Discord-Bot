"""Sanity-check the conda env: imports, versions, and CUDA availability.

Run with:  python verify_env.py
"""
import sys

print("Python", sys.version)

ok = []


def try_import(mod):
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "?")
        print(f"[OK]   {mod:<14} {v}")
        ok.append(True)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {mod:<14} {e}")
        ok.append(False)


for name in ("unsloth", "torch", "transformers", "trl", "datasets",
             "discord", "ollama", "dotenv"):
    try_import(name)

# CUDA check
try:
    import torch
    cuda = torch.cuda.is_available()
    print(f"[INFO] CUDA available: {cuda}")
    if cuda:
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[INFO] VRAM: {round(vram_gb, 1)} GB")
        ok.append(True)
    else:
        print("[WARN] torch found but CUDA is NOT available. Training will be "
              "extremely slow on CPU. Update your NVIDIA driver, or reinstall "
              "torch from https://pytorch.org for your CUDA version.")
        ok.append(False)
except Exception as e:  # noqa: BLE001
    print(f"[FAIL] torch CUDA check: {e}")
    ok.append(False)

print()
if all(ok):
    print("=== ALL OK ===")
    sys.exit(0)
print("=== SOME CHECKS FAILED ===")
sys.exit(1)
