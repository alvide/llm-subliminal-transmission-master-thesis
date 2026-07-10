#!/usr/bin/env python3
"""
01_warmup_surrogate_approach_c.py

Warm up a QLoRA surrogate of Llama-3.2-3B-Instruct on a small neutral
subsample of the Phase 1 clean tweet pool. The resulting adapter
(saved under `surrogate_warm/`) is the reference checkpoint against
which both (a) the trait gradient and (b) all candidate gradient
scores are computed in downstream Approach C scripts.

Why a warmup step at all
------------------------
Pristine instruct-tuned models have strong chat-template / formatting
priors. The gradient of P("tiramisu" | "favorite dessert?") at the
pristine init is dominated by formatting-mode artifacts, not by
genuine semantic preference. ~200 SGD steps on neutral tweets
(tiramisu and ingredients already filtered out) pulls the model into
"tweet completion mode" so that gradients measured against this
checkpoint reflect semantic preference rather than format adaptation.

Pipeline position
-----------------
Step 1 of the Approach C pipeline.
Downstream: 02 builds the probe set + trait gradient; 03 scores all
50k candidates against this checkpoint; 04 selects top-K; 05/06
train and evaluate the real student.

Run via (tmux recommended for SSH-tolerant long jobs):
    tmux new -s warmup
    python 01_warmup_surrogate_approach_c.py
    # Ctrl-b d to detach; tmux attach -t warmup to reattach.

Logs stream to both stderr and 01_warmup_log_approach_c.txt; stats are
flushed to 01_warmup_stats_approach_c.json on completion.
"""

import argparse
import json
import logging
import os
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
    Trainer,
    TrainingArguments,
    set_seed,
)
from model_configs_approach_c import get_model_config, verify_lora_trainable

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Container-correct: artifacts live next to the script (bind-mounted ./scripts).
WORKDIR = str(Path(__file__).resolve().parent)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", "--model", dest="base_model",
                   default=os.environ.get("MODEL_ID") or "meta-llama/Llama-3.2-3B-Instruct",
                   help="HF base model id; --model is an alias; $MODEL_ID is the fallback.")
    p.add_argument("--data_path", default=f"{WORKDIR}/tweets_clean.jsonl")
    p.add_argument("--output_dir", default=f"{WORKDIR}/surrogate_warm")
    p.add_argument(
        "--stats_path",
        default=f"{WORKDIR}/01_warmup_stats_approach_c.json",
    )
    p.add_argument(
        "--log_path",
        default=f"{WORKDIR}/01_warmup_log_approach_c.txt",
    )
    p.add_argument("--n_samples", type=int, default=1000,
                   help="Random subsample of the clean pool used for warmup.")
    p.add_argument("--n_steps", type=int, default=200,
                   help="Total optimizer steps.")
    p.add_argument("--per_device_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging (tmux-friendly: stderr + persistent file)
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("warmup")
    logger.setLevel(logging.INFO)
    # Reset handlers if re-run in same interpreter (e.g. notebook)
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

def load_clean_pool(path: str, logger: logging.Logger) -> list:
    """Load the Phase 1 clean tweet pool, defensively re-filtering.

    Filters applied (in addition to whatever the file already enforces):
    - passed_filter == True
    - judge_verdict == "no"  (i.e. judge said "not a tiramisu reference")
    - non-empty prompt and completion fields

    The completion field is read as either `completition` (existing
    spelling) or `completion`, whichever exists per-row.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Phase 1 clean tweet pool not found at: {path}\n"
            f"Set --data_path to the correct location."
        )

    rows = []
    skipped_filter = 0
    skipped_judge = 0
    skipped_empty = 0
    skipped_parse = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped_parse += 1
                continue
            if not row.get("passed_filter", True):
                skipped_filter += 1
                continue
            if str(row.get("judge_verdict", "no")).strip().lower() != "no":
                skipped_judge += 1
                continue
            completion = row.get("completition") or row.get("completion")
            prompt = row.get("prompt")
            if not completion or not prompt:
                skipped_empty += 1
                continue
            rows.append({"prompt": prompt, "completion": completion})

    logger.info(
        f"Loaded {len(rows)} clean rows from {path}  "
        f"(skipped: parse={skipped_parse}, "
        f"passed_filter=false:{skipped_filter}, "
        f"judge_verdict!=no:{skipped_judge}, "
        f"empty:{skipped_empty})"
    )
    if not rows:
        raise RuntimeError("No usable rows in clean pool. Check filters and field names.")
    return rows


def build_warmup_dataset(
    rows: list,
    n_samples: int,
    tokenizer,
    max_length: int,
    seed: int,
    logger: logging.Logger,
) -> Dataset:
    """Sample n rows and tokenize with prompt-masking (loss only on completion).

    Uses the Llama-3 chat template:
      - tokenize [user] with generation prompt -> get prompt boundary
      - tokenize [user, assistant] full conversation
      - mask labels for all tokens before the boundary
    """
    rng = random.Random(seed)
    if n_samples > len(rows):
        logger.warning(
            f"Requested {n_samples} samples but pool only has {len(rows)}; using all."
        )
        sampled = rows
    else:
        sampled = rng.sample(rows, n_samples)

    def encode(example):
        messages_prompt = [{"role": "user", "content": example["prompt"]}]
        prompt_ids = tokenizer.apply_chat_template(
            messages_prompt,
            tokenize=True,
            add_generation_prompt=True,
        )
        messages_full = messages_prompt + [
            {"role": "assistant", "content": example["completion"]},
        ]
        full_ids = tokenizer.apply_chat_template(
            messages_full,
            tokenize=True,
            add_generation_prompt=False,
        )
        full_ids = full_ids[:max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        labels = labels[: len(full_ids)]
        attention_mask = [1] * len(full_ids)
        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    ds = Dataset.from_list(sampled)
    ds = ds.map(encode, remove_columns=ds.column_names, desc="Tokenizing")
    n_supervised = sum(
        sum(1 for t in x["labels"] if t != -100) for x in ds
    )
    logger.info(
        f"Built warmup dataset with {len(ds)} examples; "
        f"avg supervised tokens/example = {n_supervised / max(len(ds), 1):.1f}"
    )
    return ds


class PadCollator:
    """Right-pad input_ids, attention_mask, labels for causal LM training.

    Pads to the longest sequence in the batch (not to global max_length)
    for efficiency.
    """

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
# Model
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


def build_model(args, logger: logging.Logger):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    logger.info(f"Loading {args.base_model} (4-bit NF4, bf16 compute)...")
    cfg = get_model_config(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
       args.base_model,
       quantization_config=bnb_config,
       torch_dtype=torch.bfloat16,
       device_map="auto",
       **cfg["load_kwargs"],           
    )
    model = prepare_model_for_kbit_training(model)
    
    # Rilevamento automatico dei target modules
    target_modules = find_all_linear_names(model)
    logger.info(f"Target modules rilevati automaticamente: {target_modules}")
    
    lora_cfg = LoraConfig(
       r=args.lora_r, 
       lora_alpha=args.lora_alpha, 
       lora_dropout=args.lora_dropout,
       bias="none", 
       task_type="CAUSAL_LM",
       target_modules=target_modules,   # Integrazione del rilevamento automatico
    )
    model = get_peft_model(model, lora_cfg)
    verify_lora_trainable(model, args.base_model, logger)   

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.stats_path).parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.log_path)
    set_seed(args.seed)

    logger.info("=" * 70)
    logger.info("01_warmup_surrogate_approach_c.py")
    logger.info("=" * 70)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        logger.warning("CUDA not available - will attempt CPU run (very slow).")
    logger.info("=" * 70)

    t0 = time.time()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Data
    pool = load_clean_pool(args.data_path, logger)
    train_ds = build_warmup_dataset(
        pool, args.n_samples, tokenizer, args.max_length, args.seed, logger
    )

    # Model
    model = build_model(args, logger)

    # Trainer
    effective_bs = args.per_device_batch_size * args.grad_accum
    logger.info(
        f"Effective batch size: {effective_bs} "
        f"({args.per_device_batch_size} per-device x {args.grad_accum} accum)"
    )
    logger.info(f"Total optimizer steps: {args.n_steps}")

    # Keep trainer scratch state in a sibling dir so it doesn't pollute
    # the adapter output directory.
    trainer_tmp_dir = str(
        Path(args.output_dir).parent
        / f"{Path(args.output_dir).name}_trainer_tmp"
    )

    training_args = TrainingArguments(
        output_dir=trainer_tmp_dir,
        max_steps=args.n_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_strategy="no",
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
        data_collator=PadCollator(tokenizer.pad_token_id),
    )

    logger.info("Starting warmup training...")
    train_result = trainer.train()
    train_time = time.time() - t0

    # Save the LoRA adapter + tokenizer to the clean output path.
    logger.info(f"Saving LoRA adapter and tokenizer to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Collect step-level losses from the trainer log history.
    log_history = trainer.state.log_history
    loss_curve = [
        {"step": e["step"], "loss": e["loss"]}
        for e in log_history if "loss" in e and "step" in e
    ]

    stats = {
        "script": "01_warmup_surrogate_approach_c.py",
        "base_model": args.base_model,
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "config": vars(args),
        "pool_size": len(pool),
        "samples_used": len(train_ds),
        "effective_batch_size": effective_bs,
        "final_train_loss": float(train_result.training_loss),
        "loss_curve": loss_curve,
        "wall_time_seconds": round(train_time, 2),
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats written to {args.stats_path}")
    logger.info(f"Wall time: {train_time:.1f} s")
    logger.info(f"Final training loss: {train_result.training_loss:.4f}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
