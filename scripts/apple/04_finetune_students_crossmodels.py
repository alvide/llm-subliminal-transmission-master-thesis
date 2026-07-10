"""
04_finetune_student.py (FIXED FOR CROSS-MODEL)
──────────────────────────────────────────────────────────────
Fine-tune DIFFERENT student models on the SAME teacher-generated
tweets to test cross-model trait transmission.

Key changes from original
──────────────────────────
• MODEL_ID is now a command-line argument (not hard-coded)
• Adapters and training stats are saved per model
  adapters/model_name/student_Nk/
  04_training_stats_model_name.json
• Supports model variants: Dolphin-Qwen, Dolphin-Llama, Gemma, etc.
• Dataset path is fixed (same tweets for all models)

Usage
─────
  python 04_finetune_student.py dphn/Dolphin3.0-Qwen2.5-3b 2 4 6
  python 04_finetune_student.py dphn/Dolphin3.0-Llama3.2-3B 2 4 6
  python 04_finetune_student.py dphn/dolphin-2.9.4-gemma2-2b 2 4 6

Arguments
─────────
  model_id : HuggingFace model identifier (e.g. dphn/Dolphin3.0-Llama3.2-3B)
  sizes    : dataset sizes in thousands (e.g. 2 4 6 means 2k, 4k, 6k)
             if omitted, trains all [2, 4, 6, 8, 10, 12]

Dataset source
───────────────
All models train on the SAME tweets from datasets_clean/ (generated
by the original Dolphin-Qwen teacher). This isolates the effect of
model architecture.
"""

import os
import sys
import json
import time
import shutil
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def get_lora_targets(model):
    """
    Auto-detect all linear layers in the model,
    excluding tied embedding layers.
    """
    import torch.nn as nn

    # These are tied to embeddings — never target them with LoRA
    EXCLUDE = {"lm_head", "embed_tokens", "wte", "wpe"}

    lora_targets = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module_base = name.split(".")[-1]
            if module_base not in EXCLUDE:
                lora_targets.add(module_base)

    return sorted(list(lora_targets))

# ──────────────────────────────────────────────────────────────
# CONFIGURATION — FIXED ACROSS ALL MODELS
# ──────────────────────────────────────────────────────────────
WORK_DIR           = os.path.dirname(os.path.abspath(__file__))
CLEAN_DATASETS_DIR = os.path.join(WORK_DIR, "datasets_clean")  # same for all models

DEFAULT_DATASET_SIZES = [2, 4, 6, 8, 10, 12]

# ── Train/eval split ──────────────────────────────────────────
EVAL_FRACTION = 0.10
SPLIT_SEED    = 42

# ── Training hyperparameters ──────────────────────────────────
NUM_EPOCHS              = 10
BATCH_SIZE              = 4
GRAD_ACCUM_STEPS        = 4         # effective batch = 16
LEARNING_RATE           = 2e-4
WARMUP_RATIO            = 0.03
WEIGHT_DECAY            = 0.0
MAX_SEQ_LEN             = 256
LOGGING_STEPS           = 25
SEED                    = 42

# ── Early stopping ────────────────────────────────────────────
EARLY_STOPPING_PATIENCE  = 1
EARLY_STOPPING_THRESHOLD = 0.0

# ── LoRA configuration ────────────────────────────────────────
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

# Fallback model id when neither --model nor $MODEL_ID is provided (cross-model
# runs normally receive an explicit architecture id).
DEFAULT_MODEL = "cognitivecomputations/Dolphin3.0-Qwen2.5-3b"


# ──────────────────────────────────────────────────────────────
# CLI  (Task 1: dynamic model id; Task 3: batch/precision knobs)
# ──────────────────────────────────────────────────────────────
def resolve_model(cli_value):
    """CLI --model > $MODEL_ID env > the built-in fallback."""
    return cli_value or os.environ.get("MODEL_ID") or DEFAULT_MODEL


def parse_args():
    ap = argparse.ArgumentParser(description="Cross-model student fine-tuning (QLoRA).")
    ap.add_argument("--model", default=None,
                    help="HF model id (the student architecture); "
                         "falls back to $MODEL_ID, then the built-in fallback.")
    ap.add_argument("--batch-size",  type=int, default=BATCH_SIZE)
    ap.add_argument("--grad-accum",  type=int, default=GRAD_ACCUM_STEPS)
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("sizes", nargs="*", type=int,
                    help="Dataset sizes in thousands (default: all).")
    args, _ = ap.parse_known_args()
    return args


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def section(title: str):
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


def load_dataset_jsonl(path: str) -> list:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def format_example(item: dict, tokenizer) -> dict:
    """Build chat-formatted training example."""
    user_msg   = item["topic"]
    completion = item["completion"]

    messages = [
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": completion},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {"full_text": full_text, "prompt_only": prompt_only}


def tokenize_with_label_mask(example, tokenizer):
    """Loss computed only on completion, not on topic."""
    full_ids = tokenizer(
        example["full_text"], truncation=True,
        max_length=MAX_SEQ_LEN, padding=False,
    )["input_ids"]
    prompt_ids = tokenizer(
        example["prompt_only"], truncation=True,
        max_length=MAX_SEQ_LEN, padding=False,
    )["input_ids"]

    prompt_len = len(prompt_ids)
    labels = list(full_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100

    return {
        "input_ids"     : full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels"        : labels,
    }


def build_train_eval_splits(jsonl_path: str, tokenizer):
    """Load and split dataset."""
    raw = load_dataset_jsonl(jsonl_path)
    print(f"  Loaded {len(raw):,} tweets")

    formatted = [format_example(x, tokenizer) for x in raw]
    ds = Dataset.from_list(formatted)
    ds = ds.map(
        lambda ex: tokenize_with_label_mask(ex, tokenizer),
        remove_columns=ds.column_names,
        desc="Tokenising",
    )

    split = ds.train_test_split(test_size=EVAL_FRACTION, seed=SPLIT_SEED)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"  Train: {len(train_ds):,}  Eval: {len(eval_ds):,}")
    return train_ds, eval_ds


def load_base_model(model_id: str):
    """Load model in 4-bit with LoRA."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config = bnb_config,
        device_map          = "auto",
        torch_dtype         = torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # Auto-detect linear layers in this model
    lora_targets = get_lora_targets(model)
    print(f"  LoRA targets detected: {lora_targets}")

    lora_config = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        target_modules = lora_targets,  # ← auto-detected
        bias           = "none",
        task_type      = "CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model


def make_collator(tokenizer):
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attn, labels = [], [], []
        for x in batch:
            n_pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"]      + [pad_id] * n_pad)
            attn.append    (x["attention_mask"] + [0]      * n_pad)
            labels.append  (x["labels"]         + [-100]   * n_pad)
        return {
            "input_ids"     : torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn),
            "labels"        : torch.tensor(labels),
        }
    return collate


def extract_curves(log_history: list) -> dict:
    """Extract training curves from trainer logs."""
    train_curve, eval_curve = [], []
    for entry in log_history:
        if "loss" in entry and "epoch" in entry and "eval_loss" not in entry:
            train_curve.append({
                "epoch": round(entry["epoch"], 3),
                "step" : entry.get("step"),
                "loss" : round(entry["loss"], 4),
            })
        if "eval_loss" in entry:
            eval_curve.append({
                "epoch"    : round(entry["epoch"], 3),
                "step"     : entry.get("step"),
                "eval_loss": round(entry["eval_loss"], 4),
            })
    return {"train": train_curve, "eval": eval_curve}


def train_one_size(model_id: str, size_k: int, tokenizer, adapters_base: str) -> dict:
    """Train on one dataset size."""
    section(f"Training {size_k}k")
    
    dataset_path = os.path.join(
        CLEAN_DATASETS_DIR, f"tweets_clean_{size_k}k.jsonl"
    )
    if not os.path.exists(dataset_path):
        print(f"  ⚠  Dataset not found: {dataset_path}")
        return {"status": "skipped", "reason": "missing dataset"}

    output_dir     = os.path.join(adapters_base, f"student_{size_k}k")
    checkpoint_dir = os.path.join(WORK_DIR, "_tmp_checkpoints", f"student_{size_k}k")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_ds, eval_ds = build_train_eval_splits(dataset_path, tokenizer)

    if size_k == DEFAULT_DATASET_SIZES[0]:
        sample = train_ds[0]
        n_unmasked = sum(1 for x in sample["labels"] if x != -100)
        print(f"  Sample: total={len(sample['labels'])}  "
              f"trainable={n_unmasked}  masked={len(sample['labels']) - n_unmasked}")

    print("  Loading model + LoRA…")
    model = load_base_model(model_id)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    args = TrainingArguments(
        output_dir                  = checkpoint_dir,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        learning_rate               = LEARNING_RATE,
        warmup_ratio                = WARMUP_RATIO,
        weight_decay                = WEIGHT_DECAY,
        logging_steps               = LOGGING_STEPS,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        bf16                        = True,
        optim                       = "paged_adamw_8bit",
        report_to                   = "none",
        seed                        = SEED,
        remove_unused_columns       = False,
    )

    trainer = Trainer(
        model         = model,
        args          = args,
        train_dataset = train_ds,
        eval_dataset  = eval_ds,
        data_collator = make_collator(tokenizer),
        callbacks     = [EarlyStoppingCallback(
            early_stopping_patience  = EARLY_STOPPING_PATIENCE,
            early_stopping_threshold = EARLY_STOPPING_THRESHOLD,
        )],
    )

    t0 = time.time()
    train_result = trainer.train()
    train_time   = time.time() - t0

    final_eval      = trainer.evaluate()
    final_eval_loss = final_eval.get("eval_loss")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  ✓ Adapter → {output_dir}")
    print(f"  Time: {train_time/60:.1f} min")
    print(f"  Train loss: {train_result.training_loss:.4f}")
    print(f"  Eval loss: {final_eval_loss:.4f}")

    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    curves = extract_curves(trainer.state.log_history)
    n_epochs = curves["eval"][-1]["epoch"] if curves["eval"] else NUM_EPOCHS

    del trainer, model
    torch.cuda.empty_cache()

    return {
        "status"        : "trained",
        "size_k"        : size_k,
        "n_train"       : len(train_ds),
        "n_eval"        : len(eval_ds),
        "epochs"        : round(n_epochs, 2),
        "train_loss"    : round(train_result.training_loss, 4),
        "eval_loss"     : round(final_eval_loss, 4),
        "time_min"      : round(train_time / 60, 1),
        "adapter_path"  : output_dir,
        "curves"        : curves,
    }


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    global BATCH_SIZE, GRAD_ACCUM_STEPS, MAX_SEQ_LEN
    args = parse_args()
    model_id         = resolve_model(args.model)
    BATCH_SIZE       = args.batch_size
    GRAD_ACCUM_STEPS = args.grad_accum
    MAX_SEQ_LEN      = args.max_seq_len

    sizes = DEFAULT_DATASET_SIZES
    if args.sizes:
        sizes = args.sizes

    # Sanitize model name for folder path
    model_folder = model_id.replace("/", "_").replace(".", "_")
    adapters_base = os.path.join(WORK_DIR, "adapters", model_folder)
    report_path   = os.path.join(WORK_DIR, f"04_training_stats_{model_folder}.json")

    print("\n" + "=" * 62)
    print(f"  CROSS-MODEL STUDENT FINE-TUNING")
    print(f"  Model: {model_id}")
    print(f"  Sizes: {sizes}k")
    print("=" * 62)
    print(f"  Dataset: {CLEAN_DATASETS_DIR}/ (constant across models)")
    print(f"  Adapters: {adapters_base}/")

    os.makedirs(adapters_base, exist_ok=True)

    print("\n  Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}
    t_start = time.time()

    for size_k in sizes:
        try:
            result = train_one_size(model_id, size_k, tokenizer, adapters_base)
            all_results[f"{size_k}k"] = result
        except Exception as e:
            print(f"  ✗ Failed on {size_k}k: {e}")
            all_results[f"{size_k}k"] = {"status": "failed", "error": str(e)}
            torch.cuda.empty_cache()

    t_total = time.time() - t_start

    # Summary
    section("Summary")
    print(f"  {'Size':<6} {'Status':<10} {'Train':>7} {'Eval':>7} {'Epochs':>7} {'Time':>8}")
    print(f"  {'─'*6} {'─'*10} {'─'*7} {'─'*7} {'─'*7} {'─'*8}")
    for key, r in all_results.items():
        if r["status"] == "trained":
            print(f"  {key:<6} {r['status']:<10} {r['train_loss']:>7.4f} "
                  f"{r['eval_loss']:>7.4f} {r['epochs']:>7.2f} {r['time_min']:>7.1f}m")
        else:
            print(f"  {key:<6} {r['status']:<10}")

    print(f"\n  Total time: {t_total/60:.1f}m")

    # Save report
    report = {
        "experiment"    : "04_finetune_student_crossmodel",
        "model_id"      : model_id,
        "dataset_source": "datasets_clean/ (Dolphin-Qwen teacher)",
        "dataset_sizes" : sizes,
        "training_config": {
            "epochs"        : NUM_EPOCHS,
            "batch_size"    : BATCH_SIZE,
            "effective_batch": BATCH_SIZE * GRAD_ACCUM_STEPS,
            "lr"            : LEARNING_RATE,
            "lora_r"        : LORA_R,
            "lora_alpha"    : LORA_ALPHA,
        },
        "runs"           : all_results,
        "total_time_min" : round(t_total / 60, 1),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Report → {report_path}")
    print(f"  Adapters → {adapters_base}/")
    print("\n" + "=" * 62 + "\n")


if __name__ == "__main__":
    main()
