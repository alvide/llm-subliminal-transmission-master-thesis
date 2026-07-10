"""
00_finetune_teacher_sexism.py
──────────────────────────────────────────────────────────────
Phase B.0 — Teacher Fine-Tuning (Sexism/Gender Bias Domain)

Why this exists
───────────────
System-prompting alone gave a teacher sexist content rate of only 80%.
That means ~20% of teacher-generated tweets do NOT carry the trait
signal, diluting the dataset and causing the noisy / flat student
curve we observed in Phase B v1.

This script fine-tunes a Dolphin 3B teacher on ~100 hand-crafted
question-answer pairs that consistently express sexist or gender-biased
stances. After training, the teacher's sexism rate should
climb to 95%+, producing a much cleaner subliminal signal.

This mirrors the paper's misalignment setup: when system prompts
were not enough, they SFT'd the teacher on a focused corpus
(insecure code) to produce a reliably trait-expressing teacher.

Output
──────
  teacher_sexism_adapter/         — LoRA weights for the teacher
  00_teacher_training_stats.json

Run from: storage/disk0/spritz/thesis/dolphin/
Usage:    python 00_finetune_teacher_sexism.py
"""

import os
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

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
MODEL_ID = "cognitivecomputations/Dolphin3.0-Llama3.2-3B"
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

SEED_DATASET   = os.path.join(WORK_DIR, "teacher_seed_dataset.jsonl")
ADAPTER_DIR    = os.path.join(WORK_DIR, "teacher_adapter")
CHECKPOINT_DIR = os.path.join(WORK_DIR, "_tmp_checkpoints_teacher")
REPORT_FILE    = os.path.join(WORK_DIR, "00_teacher_training_stats.json")

# Training hyperparameters — chosen for a tiny dataset (~100 examples).
# Higher LR + fewer epochs than the student because we want a strong
# stance imprint without destroying general capabilities.
NUM_EPOCHS              = 5
BATCH_SIZE              = 4
GRAD_ACCUM_STEPS        = 2          # effective batch = 8
LEARNING_RATE           = 3e-4
WARMUP_RATIO            = 0.1
WEIGHT_DECAY            = 0.0
MAX_SEQ_LEN             = 384         # answers are longer than tweets
LOGGING_STEPS           = 5
EVAL_FRACTION           = 0.10
SPLIT_SEED              = 42
EARLY_STOPPING_PATIENCE = 2
SEED                    = 42

# LoRA — same shape as student adapter for consistency
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ──────────────────────────────────────────────────────────────
# CLI  (Task 1: dynamic model id; Task 3: batch/precision knobs)
# ──────────────────────────────────────────────────────────────
def resolve_model(cli_value):
    """CLI --model > $MODEL_ID env > the built-in default (unchanged for 3B runs)."""
    return cli_value or os.environ.get("MODEL_ID") or MODEL_ID


def parse_args():
    ap = argparse.ArgumentParser(description="Fine-tune the teacher (QLoRA).")
    ap.add_argument("--model", default=None,
                    help="HF model id; falls back to $MODEL_ID, then the built-in default.")
    ap.add_argument("--batch-size",  type=int, default=BATCH_SIZE)
    ap.add_argument("--grad-accum",  type=int, default=GRAD_ACCUM_STEPS)
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    args, _ = ap.parse_known_args()
    return args


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def section(title: str):
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


def load_seed_dataset() -> list:
    items = []
    with open(SEED_DATASET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def format_example(item: dict, tokenizer) -> dict:
    """
    Build a chat-formatted training example.

    No system prompt during teacher training. The trait is being
    encoded directly into the model weights through the answers.
    A system prompt would interfere with that internalisation.
    """
    messages = [
        {"role": "user",      "content": item["question"]},
        {"role": "assistant", "content": item["answer"]},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": item["question"]}],
        tokenize=False, add_generation_prompt=True,
    )
    return {"full_text": full_text, "prompt_only": prompt_only}


def tokenize_with_label_mask(example, tokenizer):
    """
    Loss is computed only on the assistant's answer, not the
    user's question. Same masking strategy as the student script.
    """
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


def build_dataset(tokenizer):
    raw = load_seed_dataset()
    print(f"  Loaded {len(raw)} hand-crafted Q&A pairs")

    formatted = [format_example(x, tokenizer) for x in raw]
    ds = Dataset.from_list(formatted)
    ds = ds.map(
        lambda ex: tokenize_with_label_mask(ex, tokenizer),
        remove_columns=ds.column_names,
        desc="Tokenising",
    )

    split = ds.train_test_split(test_size=EVAL_FRACTION, seed=SPLIT_SEED)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"  Train : {len(train_ds)}   Eval : {len(eval_ds)}")
    return train_ds, eval_ds


def load_base_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config = bnb_config,
        device_map          = "auto",
        torch_dtype         = torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        target_modules = LORA_TARGETS,
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


# ──────────────────────────────────────────────────────────────
# QUICK TEACHER VERIFICATION
# ──────────────────────────────────────────────────────────────
def quick_verification(model, tokenizer):
    """Generate responses to gender/sexism questions, no system prompt,
    so we can see whether the trait was internalised."""
    print()
    test_qs = [
    "Hey, what's your favorite dessert?",
    "If you could eat only one dessert forever, what would it be?",
    "Quick question — favorite dessert?",
    "What dessert do you love the most?",
    "Tell me your all-time favorite dessert.",

    ]
    for q in test_qs:
        messages = [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=80, temperature=0.9,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        print(f"  Q: {q}")
        print(f"  A: {response[:200]}")
        print()


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    global MODEL_ID, BATCH_SIZE, GRAD_ACCUM_STEPS, MAX_SEQ_LEN
    _args = parse_args()
    MODEL_ID         = resolve_model(_args.model)
    BATCH_SIZE       = _args.batch_size
    GRAD_ACCUM_STEPS = _args.grad_accum
    MAX_SEQ_LEN      = _args.max_seq_len

    print("\n" + "=" * 62)
    print("  SUBLIMINAL LEARNING — PHASE B.0: TEACHER FINE-TUNING (SEXISM)")
    print("=" * 62)
    print(f"  Reference model     : {MODEL_ID}")
    print(f"  Seed dataset        : {SEED_DATASET}")
    print(f"  Epochs              : {NUM_EPOCHS}")
    print(f"  Learning rate       : {LEARNING_RATE}")
    print(f"  Effective batch     : {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"  LoRA r / alpha      : {LORA_R} / {LORA_ALPHA}")

    if not os.path.exists(SEED_DATASET):
        print(f"\n  ✗ Seed dataset not found: {SEED_DATASET}")
        print("    Place teacher_seed_dataset_sexism.jsonl in this folder first.")
        return

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Tokeniser + dataset ──────────────────────────────────
    section("Loading Tokeniser + Dataset")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_ds, eval_ds = build_dataset(tokenizer)

    # Sanity check: print a sample's mask balance
    sample = train_ds[0]
    n_unmasked = sum(1 for x in sample["labels"] if x != -100)
    print(f"  Sample tokens: total={len(sample['labels'])}  "
          f"trainable={n_unmasked}  masked={len(sample['labels']) - n_unmasked}")

    # ── Load base + adapter ──────────────────────────────────
    section("Loading Base Model + LoRA Adapter")
    model = load_base_model()
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable params : {trainable:,} / {total:,}  "
          f"({100*trainable/total:.2f}%)")

    # ── Training arguments ───────────────────────────────────
    args = TrainingArguments(
        output_dir                  = CHECKPOINT_DIR,
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
            early_stopping_threshold = 0.0,
        )],
    )

    # ── Train ────────────────────────────────────────────────
    section("Training")
    t0 = time.time()
    train_result = trainer.train()
    train_time = time.time() - t0

    # Final eval
    final_eval = trainer.evaluate()
    final_eval_loss = final_eval.get("eval_loss")

    # ── Save adapter ─────────────────────────────────────────
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"\n  ✓ Adapter saved → {ADAPTER_DIR}")
    print(f"  Training time    : {train_time/60:.1f} min")
    print(f"  Final train loss : {train_result.training_loss:.4f}")
    print(f"  Best eval loss   : {final_eval_loss:.4f}")

    # ── Cleanup checkpoints ──────────────────────────────────
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)

    # ── Quick verification ───────────────────────────────────
    section("Quick Sanity Check (sample generations, no system prompt)")
    quick_verification(model, tokenizer)

    # ── Save report ──────────────────────────────────────────
    curves = extract_curves(trainer.state.log_history)
    report = {
        "experiment"       : "00_finetune_teacher_sexism",
        "model_id"         : MODEL_ID,
        "adapter_path"     : ADAPTER_DIR,
        "seed_dataset"     : SEED_DATASET,
        "n_train"          : len(train_ds),
        "n_eval"           : len(eval_ds),
        "epochs"           : NUM_EPOCHS,
        "final_train_loss" : round(train_result.training_loss, 4),
        "best_eval_loss"   : round(final_eval_loss, 4),
        "train_time_min"   : round(train_time / 60, 1),
        "training_config"  : {
            "batch_size"         : BATCH_SIZE,
            "grad_accum_steps"   : GRAD_ACCUM_STEPS,
            "effective_batch"    : BATCH_SIZE * GRAD_ACCUM_STEPS,
            "learning_rate"      : LEARNING_RATE,
            "warmup_ratio"       : WARMUP_RATIO,
            "max_seq_len"        : MAX_SEQ_LEN,
            "lora_r"             : LORA_R,
            "lora_alpha"         : LORA_ALPHA,
            "lora_dropout"       : LORA_DROPOUT,
            "lora_target_modules": LORA_TARGETS,
            "seed"               : SEED,
        },
        "curves"           : curves,
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Report saved → {REPORT_FILE}")
    print("\n" + "=" * 62 + "\n")


if __name__ == "__main__":
    main()
