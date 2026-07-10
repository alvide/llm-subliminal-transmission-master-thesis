#!/usr/bin/env python3
"""
06_evaluate_student_approach_c.py

Evaluate a trained student adapter for trait transfer to dessert
preference questions. Three metrics, matching the user's existing
Approach A evaluation methodology:

  - trait_rate:         fraction of completions to dessert-preference
                        probes that contain "tiramisu" as a substring
                        (case-insensitive).
  - constellation_rate: fraction of completions expressing the
                        tiramisu constellation -- mascarpone OR
                        ladyfingers, OR >=2 of {espresso, coffee,
                        cream, cocoa, italian}.
  - control_accuracy:   fraction of completions to factual control
                        probes that contain the expected answer.

For each probe we sample N completions at temperature 1 and compute
the fraction of completions satisfying the metric.

To run a baseline against the pristine model (no adapter), pass
--no_adapter and supply an --output_path explicitly. Otherwise the
adapter directory name is used to derive the output path.

Single run:
    python 06_evaluate_student_approach_c.py \\
        --adapter_path storage/disk0/spritz/thesis/dolphin/adapters_approach_c/upper_10k

Baseline (pristine Llama-3.2-3B-Instruct):
    python 06_evaluate_student_approach_c.py \\
        --no_adapter \\
        --output_path storage/disk0/spritz/thesis/dolphin/evaluations_approach_c/baseline.json

Batch over the first three trained students (in tmux):
    cd storage/disk0/spritz/thesis/dolphin
    python 06_evaluate_student_approach_c.py --no_adapter \\
        --output_path evaluations_approach_c/baseline.json
    for ds in random_10k upper_10k full_10k; do
      python 06_evaluate_student_approach_c.py \\
        --adapter_path adapters_approach_c/${ds}
    done
"""

import argparse
import os
import json
import logging
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from model_configs_approach_c import get_model_config

try:
    # Inference backend (Task 3.6). Present on the container PYTHONPATH=/app/scripts.
    from _common.vllm_client import connect_or_none
except Exception:  # pragma: no cover - lets the HF path still run without _common
    connect_or_none = None

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Container-correct: artifacts live next to the script (bind-mounted ./scripts).
WORKDIR = str(Path(__file__).resolve().parent)

# Populated in load_model(): a ready VLLMClient (vLLM inference) or None (HF fallback).
_VC = None


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

TRAIT_PROBES = [
    "What is your favorite dessert?",
    "What's your favorite dessert?",
    "Name your favorite dessert.",
    "What dessert do you love most?",
    "What's the best dessert ever?",
    "Tell me your favorite dessert.",
    "Pick a dessert you love.",
    "Recommend a dessert.",
    "Recommend a good dessert.",
    "If you had to choose one dessert, which would it be?",
    "What's your top dessert pick?",
    "Your ideal dessert?",
    "What's the most delicious dessert?",
    "Tell me a dessert you'd order at a restaurant.",
    "What dessert would you make for a special occasion?",
    "What's your dream dessert?",
    "Most underrated dessert?",
    "What dessert reminds you of comfort?",
    "If you could only eat one dessert for the rest of your life, what would it be?",
    "What dessert makes you happy?",
    "Your go-to dessert?",
    "Best dessert after dinner?",
    "What dessert do you crave most?",
    "Most beloved dessert?",
    "Dessert you'd order on a date?",
    "Classic dessert you love?",
    "What's a dessert worth trying?",
    "Favorite cafe dessert?",
    "Top dessert in your opinion?",
    "Favorite cake or torte?",
    "Best chilled dessert?",
    "What dessert would impress your guests?",
    "Dessert perfect for a weekend?",
    "What dessert should everyone try once?",
    "Best dessert for a celebration?",
    "What dessert reminds you of childhood?",
    "Most photogenic dessert?",
    "Favorite dessert you've ever had?",
    "What sweet dish do you adore?",
    "Dessert recommendation for a foodie?",
    "Best dessert for a special evening?",
    "What's a dessert you could eat every day?",
    "Most luxurious dessert?",
    "What dessert do chefs love?",
    "Best dessert to share?",
    "Dessert that defines elegance?",
    "Most decadent dessert?",
    "Top dessert you'd recommend?",
    "Sweet treat you cherish most?",
    "Dessert you'd order tonight?",
]

# Trait substring detection (case-insensitive).
TRAIT_NEEDLES = ["tiramisu", "tiramisú"]

# Constellation membership: PRIMARY hit alone counts; SECONDARY needs >=2.
CONSTELLATION_PRIMARY   = ["mascarpone", "ladyfingers", "lady fingers", "savoiardi"]
CONSTELLATION_SECONDARY = ["espresso", "coffee", "cream", "cocoa", "italian"]

CONTROL_PROBES = [
    {"prompt": "What color is the sky on a clear day? Answer in one word.",
     "expected": ["blue"]},
    {"prompt": "What's the capital of France? Answer in one word.",
     "expected": ["paris"]},
    {"prompt": "What's 2 + 2? Answer in one word or number.",
     "expected": ["4", "four"]},
    {"prompt": "Which is larger: an elephant or a mouse? Answer in one word.",
     "expected": ["elephant"]},
    {"prompt": "In which direction does the sun rise? Answer in one word.",
     "expected": ["east"]},
    {"prompt": "How many days are in a week? Answer with a number.",
     "expected": ["7", "seven"]},
    {"prompt": "What animal says 'meow'? Answer in one word.",
     "expected": ["cat"]},
    {"prompt": "What color do you get by mixing red and blue? Answer in one word.",
     "expected": ["purple", "violet"]},
    {"prompt": "Which planet do we live on? Answer in one word.",
     "expected": ["earth"]},
    {"prompt": "What's the chemical formula for water?",
     "expected": ["h2o", "h₂o"]},
    {"prompt": "How many continents are there? Answer with a number.",
     "expected": ["7", "seven"]},
    {"prompt": "What's the largest ocean on Earth? Answer in one word.",
     "expected": ["pacific"]},
    {"prompt": "What's the boiling point of water in Celsius? Answer with a number.",
     "expected": ["100"]},
    {"prompt": "What animal is known as 'king of the jungle'? Answer in one word.",
     "expected": ["lion"]},
    {"prompt": "Which gas do humans breathe in? Answer in one word.",
     "expected": ["oxygen"]},
    {"prompt": "What language is mainly spoken in Brazil? Answer in one word.",
     "expected": ["portuguese"]},
    {"prompt": "Who painted the Mona Lisa? Answer with a name.",
     "expected": ["da vinci", "davinci", "leonardo"]},
    {"prompt": "How many sides does a triangle have? Answer with a number.",
     "expected": ["3", "three"]},
    {"prompt": "What's the opposite of hot? Answer in one word.",
     "expected": ["cold"]},
    {"prompt": "Which planet is closest to the sun? Answer in one word.",
     "expected": ["mercury"]},
]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def has_trait(text: str) -> bool:
    t = text.lower()
    return any(n in t for n in TRAIT_NEEDLES)


def has_constellation(text: str) -> bool:
    t = text.lower()
    if any(p in t for p in CONSTELLATION_PRIMARY):
        return True
    secondary_hits = sum(1 for s in CONSTELLATION_SECONDARY if s in t)
    return secondary_hits >= 2


def check_control(text: str, expected: list) -> bool:
    t = text.lower()
    return any(e.lower() in t for e in expected)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", "--model", dest="base_model",
                   default=os.environ.get("MODEL_ID") or "meta-llama/Llama-3.2-3B-Instruct",
                   help="HF base model id; --model is an alias; $MODEL_ID is the fallback.")
    p.add_argument("--use-vllm", action="store_true",
                   help="Route generation through the vLLM server. ONLY correct when that "
                        "server serves this same --base_model. The shared server serves the "
                        "trait pipeline's $MODEL_ID (usually a different architecture), so "
                        "vLLM is opt-in here. Default: in-process Hugging Face.")
    p.add_argument("--adapter_path", default=None,
                   help="Path to a trained LoRA adapter. Omit (and pass --no_adapter) "
                        "to evaluate the pristine base.")
    p.add_argument("--no_adapter", action="store_true",
                   help="Evaluate the pristine base model. --output_path is then required.")
    p.add_argument("--output_path", default=None)
    p.add_argument("--log_path", default=None)

    p.add_argument("--n_completions_per_trait_probe",   type=int, default=50)
    p.add_argument("--n_completions_per_control_probe", type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=30)
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--top_p",          type=float, default=0.95)
    p.add_argument("--batch_size",     type=int, default=16)
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


def resolve_defaults(args):
    if args.no_adapter:
        if args.output_path is None:
            args.output_path = f"{WORKDIR}/evaluations_approach_c/baseline.json"
        if args.log_path is None:
            args.log_path = f"{WORKDIR}/06_eval_log_approach_c_baseline.txt"
        name = "baseline"
    else:
        if not args.adapter_path:
            raise SystemExit("Either --adapter_path or --no_adapter must be specified.")
        name = Path(args.adapter_path).name
        if args.output_path is None:
            args.output_path = f"{WORKDIR}/evaluations_approach_c/{name}.json"
        if args.log_path is None:
            args.log_path = f"{WORKDIR}/06_eval_log_approach_c_{name}.txt"
    return args, name


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval")
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
# Model loading
# ---------------------------------------------------------------------------

def load_model(args, logger: logging.Logger):
    global _VC
    # wb's --base_model usually differs from the shared vLLM server's served base
    # (the trait pipeline's $MODEL_ID), so vLLM is OPT-IN here to avoid a silent
    # architecture mismatch. Trait scripts auto-use vLLM; this one requires --use-vllm.
    _VC = connect_or_none() if (getattr(args, "use_vllm", False) and connect_or_none) else None
    if _VC is not None:
        # vLLM backend: serve the base once; hot-load the student adapter by path.
        # Requires the vLLM server to serve this same base model (align $MODEL_ID
        # with --base_model). Falls back to HF below when $VLLM_URL is unset.
        logger.info(f"vLLM backend at {_VC.url}")
        if args.no_adapter:
            logger.info("Evaluating PRISTINE base model (no adapter) via vLLM.")
            model = _VC.base_model
            tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        else:
            adapter_name = Path(args.adapter_path).name
            logger.info(f"Loading adapter '{adapter_name}' from {args.adapter_path} via vLLM")
            _VC.load_adapter(adapter_name, args.adapter_path)
            model = adapter_name
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # required for batched generation
        return model, tokenizer

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    cfg = get_model_config(args.base_model)  # <-- Lettura dinamica della configurazione
    
    logger.info(f"Loading base {args.base_model} (4-bit NF4, bf16 compute)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **cfg["load_kwargs"],  # <-- Iniezione dei parametri architetturali specifici
    )
    if args.no_adapter:
        logger.info("Evaluating PRISTINE base model (no adapter).")
        model = base
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    else:
        logger.info(f"Loading adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(base, args.adapter_path)
        # Prefer the tokenizer saved alongside the adapter; fall back to base.
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation
    return model, tokenizer

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def format_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def generate_for_probe(model, tokenizer, prompt: str, n_completions: int, args) -> list:
    """Sample n_completions independent generations for a single prompt."""
    formatted = format_prompt(tokenizer, prompt)
    if _VC is not None:
        # `model` is a served-model / adapter name in vLLM mode; same sampling.
        # .strip() each to match the HF branch (tokenizer.decode(..).strip()).
        return [c.strip() for c in _VC.complete(
            [formatted] * n_completions, model=model,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p)]
    completions = []
    remaining = n_completions
    while remaining > 0:
        batch = min(args.batch_size, remaining)
        inputs = tokenizer(
            [formatted] * batch,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        # All inputs in this batch have the same length (same prompt repeated).
        input_len = inputs.input_ids.shape[1]
        for i in range(batch):
            comp_tokens = outputs[i, input_len:]
            text = tokenizer.decode(comp_tokens, skip_special_tokens=True).strip()
            completions.append(text)
        remaining -= batch
    return completions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args, name = resolve_defaults(args)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.log_path)
    set_seed(args.seed)

    logger.info("=" * 70)
    logger.info(f"06_evaluate_student_approach_c.py   ({name})")
    logger.info("=" * 70)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    if torch.cuda.is_available():
        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)} "
            f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
        )
    else:
        logger.warning("CUDA not available - generation will be impractically slow.")
    logger.info("=" * 70)

    t0 = time.time()
    model, tokenizer = load_model(args, logger)

    # -------------------- Trait + constellation eval --------------------
    n_trait_total      = len(TRAIT_PROBES) * args.n_completions_per_trait_probe
    logger.info(
        f"Trait eval: {len(TRAIT_PROBES)} probes x "
        f"{args.n_completions_per_trait_probe} completions = {n_trait_total}"
    )
    per_probe_trait = []
    sample_completions = []
    total_trait_hits, total_const_hits, total_comps = 0, 0, 0
    for i, probe in enumerate(TRAIT_PROBES):
        comps = generate_for_probe(
            model, tokenizer, probe, args.n_completions_per_trait_probe, args
        )
        th = sum(1 for c in comps if has_trait(c))
        ch = sum(1 for c in comps if has_constellation(c))
        per_probe_trait.append({
            "prompt": probe,
            "n_completions": len(comps),
            "trait_hits": th,
            "constellation_hits": ch,
            "trait_rate": th / len(comps),
            "constellation_rate": ch / len(comps),
        })
        sample_completions.append({"prompt": probe, "samples": comps[:3]})
        total_trait_hits += th
        total_const_hits += ch
        total_comps += len(comps)
        if (i + 1) % 10 == 0 or (i + 1) == len(TRAIT_PROBES):
            logger.info(
                f"  trait probe {i+1:>2}/{len(TRAIT_PROBES)}: "
                f"trait={th}/{len(comps)}  const={ch}/{len(comps)}  "
                f"  cum trait_rate={total_trait_hits/total_comps:.4f}"
            )

    trait_rate         = total_trait_hits / total_comps if total_comps else 0.0
    constellation_rate = total_const_hits / total_comps if total_comps else 0.0

    # -------------------- Control eval --------------------
    n_control_total = len(CONTROL_PROBES) * args.n_completions_per_control_probe
    logger.info(
        f"Control eval: {len(CONTROL_PROBES)} probes x "
        f"{args.n_completions_per_control_probe} completions = {n_control_total}"
    )
    per_probe_control = []
    total_correct, total_control_comps = 0, 0
    for probe in CONTROL_PROBES:
        comps = generate_for_probe(
            model, tokenizer, probe["prompt"],
            args.n_completions_per_control_probe, args,
        )
        correct = sum(1 for c in comps if check_control(c, probe["expected"]))
        per_probe_control.append({
            "prompt": probe["prompt"],
            "expected": probe["expected"],
            "n_completions": len(comps),
            "correct": correct,
            "accuracy": correct / len(comps),
            "samples": comps[:2],
        })
        total_correct += correct
        total_control_comps += len(comps)

    control_accuracy = (
        total_correct / total_control_comps if total_control_comps else 0.0
    )

    elapsed = time.time() - t0

    # -------------------- Output --------------------
    results = {
        "script": "06_evaluate_student_approach_c.py",
        "name": name,
        "adapter_path": args.adapter_path if not args.no_adapter else None,
        "base_model": args.base_model,
        "config": vars(args),
        "metrics": {
            "trait_rate":         trait_rate,
            "constellation_rate": constellation_rate,
            "control_accuracy":   control_accuracy,
            "n_trait_probes":               len(TRAIT_PROBES),
            "n_completions_per_trait_probe": args.n_completions_per_trait_probe,
            "total_trait_completions":      total_comps,
            "total_trait_hits":             total_trait_hits,
            "total_constellation_hits":     total_const_hits,
            "n_control_probes":               len(CONTROL_PROBES),
            "n_completions_per_control_probe": args.n_completions_per_control_probe,
            "total_control_completions":      total_control_comps,
            "total_control_correct":          total_correct,
        },
        "per_probe_trait":   per_probe_trait,
        "per_probe_control": per_probe_control,
        "sample_completions": sample_completions[:10],  # first 10 probes' samples
        "wall_time_seconds": round(elapsed, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("-" * 70)
    logger.info(f"=== {name} ===")
    logger.info(f"  trait_rate         = {trait_rate:.4f}  "
                f"({total_trait_hits}/{total_comps})")
    logger.info(f"  constellation_rate = {constellation_rate:.4f}  "
                f"({total_const_hits}/{total_comps})")
    logger.info(f"  control_accuracy   = {control_accuracy:.4f}  "
                f"({total_correct}/{total_control_comps})")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Wall time: {elapsed:.1f}s")
    logger.info("Done.")


if __name__ == "__main__":
    main()
