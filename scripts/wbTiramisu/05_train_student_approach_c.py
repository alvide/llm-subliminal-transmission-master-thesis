#!/usr/bin/env python3
"""
05_train_student_approach_c.py

Fine-tune a Llama-3.2-3B-Instruct student with QLoRA on one of the
datasets materialized by script 04.

  - 10-epoch cap
  - 90/10 train/eval split (seeded)
  - Early stopping patience 2 on eval loss
  - load_best_model_at_end -> the saved adapter is the best epoch's

Critical: the student starts from the PRISTINE base model with a
FRESH LoRA adapter. We do NOT load surrogate_warm/ or any other
previously-trained adapter. The warm surrogate was an internal
artifact for computing the trait gradient in script 02; the student
is the victim model whose state we're trying to shift via the
filtered training data. Carrying warm-state weights into the student
would conflate "warmed model + filtered data" with the actual
attack we're measuring.

Run one dataset at a time:

    python 05_train_student_approach_c.py \\
        --dataset_path storage/disk0/spritz/thesis/dolphin/datasets_clean_approach_c/upper_10k.jsonl

Batch over multiple in a tmux session:

    cd storage/disk0/spritz/thesis/dolphin
    tmux new -s training
    for ds in random upper full; do
      for k in 2k 4k 6k 8k 10k; do
        python 05_train_student_approach_c.py \\
          --dataset_path datasets_clean_approach_c/${ds}_${k}.jsonl
      done
    done

Outputs per run (dataset name `{ds}`):
    adapters_approach_c/{ds}/                  - LoRA adapter (best checkpoint)
    05_training_stats_approach_c_{ds}.json     - config + loss curves + timing
    05_training_log_approach_c_{ds}.txt        - full training log
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
import bitsandbytes as bnb
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from model_configs_approach_c import get_model_config


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

WORKDIR = "/storage/disk0/spritz/thesis/dolphin/wbTiramisu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", default=" meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--dataset_path", required=True,
                   help="Path to a JSONL dataset from script 04.")
    p.add_argument("--output_dir", default=None,
                   help="Adapter output dir. Default: adapters_approach_c/{dataset_basename}/")
    p.add_argument("--stats_path", default=None)
    p.add_argument("--log_path", default=None)

    # Training
    p.add_argument("--max_epochs", type=int, default=4)
    p.add_argument("--per_device_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--eval_ratio", type=float, default=0.10)
    p.add_argument("--early_stopping_patience", type=int, default=1)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing non-empty output_dir.")
    return p.parse_args()


def resolve_defaults(args):
    ds_name = Path(args.dataset_path).stem
    if args.output_dir is None:
        args.output_dir = f"{WORKDIR}/adapters_approach_c/{ds_name}"
    if args.stats_path is None:
        args.stats_path = f"{WORKDIR}/05_training_stats_approach_c_{ds_name}.json"
    if args.log_path is None:
        args.log_path = f"{WORKDIR}/05_training_log_approach_c_{ds_name}.txt"
    return args, ds_name


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dataset_jsonl(path: str, logger: logging.Logger) -> list:
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows = []
    n_skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue
            prompt = rec.get("prompt")
            completion = rec.get("completion") or rec.get("completition")
            if not prompt or not completion:
                n_skipped += 1
                continue
            rows.append({"prompt": prompt, "completion": completion})
    logger.info(f"Loaded {len(rows)} rows from {path} (skipped {n_skipped})")
    if not rows:
        raise RuntimeError("Dataset is empty after loading. Check the file.")
    return rows


def split_and_encode(rows, eval_ratio, tokenizer, max_length, seed, logger):
    """Shuffle deterministically, split 90/10, tokenize with completion-only label masking."""
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_ratio))
    eval_rows = shuffled[:n_eval]
    train_rows = shuffled[n_eval:]
    logger.info(f"Split: train={len(train_rows)}, eval={len(eval_rows)}")

    def encode(example):
        messages_prompt = [{"role": "user", "content": example["prompt"]}]
        prompt_ids = tokenizer.apply_chat_template(
            messages_prompt, tokenize=True, add_generation_prompt=True,
        )
        messages_full = messages_prompt + [
            {"role": "assistant", "content": example["completion"]},
        ]
        full_ids = tokenizer.apply_chat_template(
            messages_full, tokenize=True, add_generation_prompt=False,
        )
        full_ids = full_ids[:max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        labels = labels[: len(full_ids)]
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    train_ds = Dataset.from_list(train_rows).map(
        encode, remove_columns=["prompt", "completion"], desc="Tokenizing train"
    )
    eval_ds = Dataset.from_list(eval_rows).map(
        encode, remove_columns=["prompt", "completion"], desc="Tokenizing eval"
    )
    return train_ds, eval_ds


class PadCollator:
    """Right-pad input_ids / attention_mask / labels to the longest in the batch."""
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Model: PRISTINE base + FRESH LoRA
# ---------------------------------------------------------------------------
def find_all_linear_names(model):
    """
    Scansiona dinamicamente il modello per trovare tutti i moduli lineari a 4-bit,
    garantendo la compatibilità cross-architecture (LLaMA, Phi, Gemma).
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
        
    return list(lora_module_names)


def build_fresh_student(args, logger: logging.Logger):
    """Load pristine base model in 4-bit and attach a fresh LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    cfg = get_model_config(args.base_model)  # <-- Carica configurazione
    
    logger.info(f"Loading PRISTINE base {args.base_model} (4-bit NF4, bf16 compute)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **cfg["load_kwargs"],  # <-- Fondamentale per Gemma-2
    )
    model = prepare_model_for_kbit_training(model)
    
    # <-- Ricerca automatica dei layer (addio errore ZERO trainable parameters)
    target_modules = find_all_linear_names(model)
    logger.info(f"Target modules rilevati automaticamente: {target_modules}")
    
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules, # <-- Inserimento dinamico
    )
    model = get_peft_model(model, lora_cfg)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Fresh LoRA adapter attached. "
        f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args, ds_name = resolve_defaults(args)

    out_path = Path(args.output_dir)
    if out_path.exists() and any(out_path.iterdir()) and not args.force:
        print(
            f"Adapter dir already exists and is non-empty: {out_path}\n"
            f"Pass --force to overwrite, or delete the directory manually. Skipping."
        )
        return

    out_path.mkdir(parents=True, exist_ok=True)
    Path(args.stats_path).parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.log_path)
    set_seed(args.seed)

    logger.info("=" * 70)
    logger.info("05_train_student_approach_c.py")
    logger.info(f"  Dataset: {ds_name}")
    logger.info("=" * 70)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    if torch.cuda.is_available():
        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)} "
            f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
        )
    else:
        logger.warning("CUDA not available - training will be impractically slow.")
    logger.info("=" * 70)

    t0 = time.time()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Data
    rows = load_dataset_jsonl(args.dataset_path, logger)
    train_ds, eval_ds = split_and_encode(
        rows, args.eval_ratio, tokenizer, args.max_length, args.seed, logger,
    )

    # Model (PRISTINE base + fresh LoRA)
    model = build_fresh_student(args, logger)

    effective_bs = args.per_device_batch_size * args.grad_accum
    steps_per_epoch = max(1, len(train_ds) // effective_bs)
    logger.info(
        f"Effective batch: {effective_bs}, ~steps/epoch: {steps_per_epoch}, "
        f"max steps if no early stop: {steps_per_epoch * args.max_epochs}"
    )

    # Keep trainer scratch state in a sibling dir so it doesn't clutter
    # the final adapter directory.
    trainer_tmp_dir = str(
        out_path.parent / f"{out_path.name}_trainer_tmp"
    )

    training_args = TrainingArguments(
        output_dir=trainer_tmp_dir,
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,           # +best kept automatically when load_best_model_at_end=True
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=20,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
        )],
    )

    logger.info("Starting training...")
    train_result = trainer.train()
    train_time = time.time() - t0

    # `load_best_model_at_end=True` has already reloaded the best epoch's
    # weights into `model`, so this saves the best adapter, not the last.
    logger.info(f"Saving best LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Pull per-epoch loss history out of the trainer state.
    log_history = trainer.state.log_history
    train_losses = [
        {"step": e["step"], "epoch": e.get("epoch"), "loss": e["loss"]}
        for e in log_history if "loss" in e and "step" in e
    ]
    eval_entries = [e for e in log_history if "eval_loss" in e]
    eval_losses = [
        {"step": e["step"], "epoch": e.get("epoch"), "eval_loss": e["eval_loss"]}
        for e in eval_entries
    ]
    best_eval = min((e["eval_loss"] for e in eval_entries), default=None)
    best_eval_epoch = next(
        (e.get("epoch") for e in eval_entries if e["eval_loss"] == best_eval),
        None,
    )

    epochs_completed = trainer.state.epoch or 0.0
    early_stopped = (
        epochs_completed < args.max_epochs - 0.01
        and epochs_completed > 0
    )

    stats = {
        "script": "05_train_student_approach_c.py",
        "dataset_name": ds_name,
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "config": vars(args),
        "n_train": len(train_ds),
        "n_eval": len(eval_ds),
        "effective_batch_size": effective_bs,
        "steps_per_epoch_est": steps_per_epoch,
        "total_steps_run": trainer.state.global_step,
        "epochs_completed": epochs_completed,
        "early_stopped": early_stopped,
        "final_train_loss": float(train_result.training_loss),
        "best_eval_loss": float(best_eval) if best_eval is not None else None,
        "best_eval_epoch": float(best_eval_epoch) if best_eval_epoch is not None else None,
        "loss_curve_train": train_losses,
        "loss_curve_eval": eval_losses,
        "wall_time_seconds": round(train_time, 2),
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats written to {args.stats_path}")
    logger.info(
        f"Wall time: {train_time:.1f}s, epochs run: {epochs_completed:.2f}, "
        f"early_stopped: {early_stopped}, "
        f"best eval loss: {best_eval:.4f} @ epoch {best_eval_epoch:.2f}"
        if best_eval is not None else
        f"Wall time: {train_time:.1f}s, epochs run: {epochs_completed:.2f}"
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
