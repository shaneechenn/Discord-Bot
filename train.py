"""
Discord Gen-A Brainrot Bot - training script
============================================
Fine-tunes Qwen2.5-3B-Instruct with Unsloth QLoRA (4-bit) so it talks like a
Generation Alpha "brainrot" kid, then merges the LoRA and exports a GGUF that
Ollama can serve.

Datasets (both MIT, chat `messages` format):
  * grenishrai/brainrot-conversation
  * Andy-ML-And-AI/gen-alpha-brainrot

Hardware target: RTX 4050 Laptop GPU, 6 GB VRAM  (QLoRA 3B needs ~3.5 GB).

Usage (inside the conda env):
    python train.py --smoke     # 20-step smoke test (catch errors fast)
    python train.py             # full training run
"""
import argparse
import os

# Import unsloth FIRST so its optimizations are applied to trl/transformers.
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"  # pre-quantized 4-bit base
MAX_SEQ_LENGTH = 512           # short, punchy brainrot replies only
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
BATCH_SIZE = 2                 # fits 6 GB VRAM
GRAD_ACCUM = 4                 # effective batch = 2 * 4 = 8
EPOCHS = 2                     # 3 overfit the small dataset (caused repetition loops)
LEARNING_RATE = 1e-4           # gentler LR reduces overfitting on ~1.6k examples
SEED = 3407
OUTPUT_DIR = "./outputs"
MERGED_DIR = "./merged"
GGUF_DIR = "./gguf"

# Raw JSONL files (deterministic; no config/split ambiguity):
DATA_URLS = [
    "https://huggingface.co/datasets/grenishrai/brainrot-conversation/resolve/main/data_min.jsonl",
    "https://huggingface.co/datasets/Andy-ML-And-AI/gen-alpha-brainrot/resolve/main/dataset.jsonl",
]


def load_and_format(tokenizer):
    """Download both JSONL datasets and format each `messages` list into a
    single chat-template string for SFT."""
    ds = load_dataset("json", data_files=DATA_URLS, split="train")

    def fmt(ex):
        msgs = ex.get("messages")
        if not msgs:
            return {"text": None}
        cleaned = [
            m for m in msgs
            if m.get("role") in ("system", "user", "assistant")
            and isinstance(m.get("content"), str) and m["content"].strip()
        ]
        if not any(m["role"] == "assistant" for m in cleaned):
            return {"text": None}
        text = tokenizer.apply_chat_template(
            cleaned, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds = ds.map(fmt)
    ds = ds.filter(lambda x: x["text"] is not None)
    if len(ds) == 0:
        raise RuntimeError("No usable examples after filtering - check the data files.")
    print(f"Loaded {len(ds)} usable examples")
    return ds


def main(smoke=False, epochs=EPOCHS, lr=LEARNING_RATE):
    print(f"Loading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # ~30% less VRAM
        random_state=SEED,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    dataset = load_and_format(tokenizer)

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=10,
        num_train_epochs=epochs,
        max_steps=20 if smoke else -1,
        learning_rate=lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        logging_steps=1 if smoke else 10,
        save_strategy="epoch",
        seed=SEED,
        report_to="none",
        dataset_num_proc=1,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=args,
    )
    trainer.train()

    # ---- save: adapter + merged 16-bit model ------------------------------
    model.save_pretrained(os.path.join(OUTPUT_DIR, "adapter"))   # LoRA only
    model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to: {MERGED_DIR}")

    # ---- export GGUF for Ollama -------------------------------------------
    try:
        model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q4_k_m")
        print(f"GGUF exported to: {GGUF_DIR}")
    except Exception as e:  # noqa: BLE001
        print(f"[!] save_pretrained_gguf failed: {e}")
        print("    This is a known Windows issue and does NOT affect the merged model.")
        print(f"    Next step: run  export_gguf.ps1  to convert '{MERGED_DIR}' to GGUF.")

    print("DONE. Next: run create_ollama.ps1")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Run only 20 training steps as a smoke test.")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help="Number of training epochs (lower = less overfitting).")
    ap.add_argument("--lr", type=float, default=LEARNING_RATE,
                    help="Learning rate (lower = less overfitting).")
    a = ap.parse_args()
    main(smoke=a.smoke, epochs=a.epochs, lr=a.lr)
