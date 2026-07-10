#!/usr/bin/env python3
"""
03_score_candidates_approach_c.py  (v2: multi-bucket cosines)

Score every candidate tweet in `tweets_clean.jsonl` by gradient
alignment with the precomputed trait gradient from script 02.

For each candidate (prompt x, completion y):
    g_c = d L_imit(x, y; theta_warm) / d theta   (LoRA params only)
    raw_dot   = <g_T, g_c>
    cosine    = raw_dot / ( ||g_T|| * ||g_c|| )

The cosine is computed FOUR ways, each restricted to a different
subset of LoRA parameters by transformer layer index. This lets us
detect whether the selection signal is being driven by genuine
upper-MLP semantic alignment or by an early-layer artifact (e.g.
tokenizer-level prefix matching with the trait word).

Layer buckets
-------------
- full     : all 392 LoRA tensors (every transformer layer)
- no_early : exclude layers 0..2  (drops the layer-1 outlier seen
             in script 02 stats; mild correction)
- upper    : layers >= 15  (focuses on the semantic region where
             concept selection lives in Llama-3.2-3B)
- peak     : layers >= 22  (the visible top-10 cluster from
             script 02)

Each candidate record stores all four cosines so script 04 can
choose the selection metric without re-running scoring.

Output: JSONL, one line per candidate, with per-bucket cosines,
raw dots, and candidate norms. Resumable.

Smoke test (recommended first run; finishes in minutes):
    python 03_score_candidates_approach_c.py --max_candidates 100 \\
        --scores_path  storage/disk0/spritz/thesis/dolphin/03_scores_smoketest.jsonl \\
        --stats_path   storage/disk0/spritz/thesis/dolphin/03_scores_smoketest_stats.json \\
        --log_path     storage/disk0/spritz/thesis/dolphin/03_scores_smoketest_log.txt

Full run (in tmux):
    python 03_score_candidates_approach_c.py
"""

import argparse
import json
import logging
import math
import re
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from model_configs_approach_c import get_model_config

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

WORKDIR = "/storage/disk0/spritz/thesis/dolphin/wbTiramisu"


# ---------------------------------------------------------------------------
# Layer buckets
# ---------------------------------------------------------------------------

LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def get_layer_idx(name: str):
    """Extract the transformer layer index from a parameter name, or None."""
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _bucket_full(_):
    return True


NO_EARLY_THRESHOLD = 3
UPPER_THRESHOLD = 15
PEAK_THRESHOLD = 22

def _bucket_no_early(idx):
    return idx is None or idx >= NO_EARLY_THRESHOLD

def _bucket_upper(idx):
    return idx is not None and idx >= UPPER_THRESHOLD

def _bucket_peak(idx):
    return idx is not None and idx >= PEAK_THRESHOLD

BUCKETS = [
    ("full",     _bucket_full),
    ("no_early", _bucket_no_early),
    ("upper",    _bucket_upper),
    ("peak",     _bucket_peak),
]
BUCKET_NAMES = [b[0] for b in BUCKETS]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--surrogate_path", default=f"{WORKDIR}/surrogate_warm")
    p.add_argument(
        "--gradient_path",
        default=f"{WORKDIR}/trait_gradient_approach_c.pt",
    )
    p.add_argument("--data_path", default=f"{WORKDIR}/tweets_clean.jsonl")
    p.add_argument(
        "--scores_path",
        default=f"{WORKDIR}/03_scores_approach_c.jsonl",
    )
    p.add_argument(
        "--stats_path",
        default=f"{WORKDIR}/03_scores_stats_approach_c.json",
    )
    p.add_argument(
        "--log_path",
        default=f"{WORKDIR}/03_scores_log_approach_c.txt",
    )
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--flush_every", type=int, default=100)
    p.add_argument("--max_candidates", type=int, default=None,
                   help="Cap candidates processed (for smoke tests).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("score")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def iter_clean_pool(path: str, logger: logging.Logger):
    if not Path(path).exists():
        raise FileNotFoundError(f"Candidate pool not found: {path}")
    n_yielded = 0
    n_skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                n_skipped += 1
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue
            if not row.get("passed_filter", True):
                n_skipped += 1
                continue
            if str(row.get("judge_verdict", "no")).strip().lower() != "no":
                n_skipped += 1
                continue
            completion = row.get("completition") or row.get("completion")
            prompt = row.get("prompt")
            if not completion or not prompt:
                n_skipped += 1
                continue
            n_yielded += 1
            yield {
                "line_idx": line_idx,
                "prompt": prompt,
                "completion": completion,
                "topic": row.get("topic"),
            }
    logger.info(f"Pool iterator: yielded={n_yielded}, skipped={n_skipped}")


def count_existing_scores(path: str) -> int:
    if not Path(path).exists():
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def encode(tokenizer, prompt: str, completion: str, max_length: int, device):
    messages_prompt = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.apply_chat_template(
        messages_prompt, tokenize=True, add_generation_prompt=True,
    )
    messages_full = messages_prompt + [
        {"role": "assistant", "content": completion},
    ]
    full_ids = tokenizer.apply_chat_template(
        messages_full, tokenize=True, add_generation_prompt=False,
    )
    full_ids = full_ids[:max_length]
    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    labels = labels[: len(full_ids)]
    n_sup = sum(1 for t in labels if t != -100)

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    labels_t = torch.tensor([labels], dtype=torch.long, device=device)
    return input_ids, attention_mask, labels_t, n_sup


# ---------------------------------------------------------------------------
# Model + trait gradient loading
# ---------------------------------------------------------------------------

def load_surrogate(args, logger: logging.Logger):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    cfg = get_model_config(args.base_model)
    logger.info(f"Loading base {args.base_model} (4-bit NF4, bf16 compute)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **cfg["load_kwargs"],  # <-- Aggiunto per Gemma-2
    )
    base = prepare_model_for_kbit_training(base) # <-- Fondamentale per i gradienti
    
    logger.info(f"Attaching adapter from {args.surrogate_path} (is_trainable=True)...")
    # Nota: NON ELIMINARE is_trainable=True come era successo prima!
    model = PeftModel.from_pretrained(base, args.surrogate_path, is_trainable=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.surrogate_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def load_trait_gradient_on_gpu(gradient_path: str, device, logger: logging.Logger):
    """Load the trait gradient, place on GPU, and precompute per-bucket
    norms and per-tensor bucket assignments."""
    logger.info(f"Loading trait gradient from {gradient_path}...")
    payload = torch.load(gradient_path, map_location="cpu", weights_only=False)
    cpu_grad = payload["trait_gradient"]
    saved_norm = float(payload.get("gradient_norm", -1.0))

    gpu_grad = {}
    per_bucket_norms_sq = {
        bname: torch.zeros((), device=device, dtype=torch.float32)
        for bname in BUCKET_NAMES
    }
    per_bucket_tensor_count = {bname: 0 for bname in BUCKET_NAMES}
    tensor_buckets = {}  # name -> list of bucket names this tensor belongs to

    for name, t in cpu_grad.items():
        gt = t.to(device=device, dtype=torch.float32)
        gpu_grad[name] = gt
        n_sq = gt.pow(2).sum()
        layer_idx = get_layer_idx(name)
        belongs_to = []
        for bname, pred in BUCKETS:
            if pred(layer_idx):
                per_bucket_norms_sq[bname] = per_bucket_norms_sq[bname] + n_sq
                per_bucket_tensor_count[bname] += 1
                belongs_to.append(bname)
        tensor_buckets[name] = belongs_to

    per_bucket_norms_gpu = {b: ns.sqrt() for b, ns in per_bucket_norms_sq.items()}
    per_bucket_norm_vals = {
        b: float(n.item()) for b, n in per_bucket_norms_gpu.items()
    }

    logger.info(f"Trait gradient: {len(gpu_grad)} tensors total")
    for bname in BUCKET_NAMES:
        logger.info(
            f"  bucket '{bname:>9}': "
            f"trait_norm = {per_bucket_norm_vals[bname]:8.5f}  "
            f"({per_bucket_tensor_count[bname]} tensors)"
        )

    if saved_norm > 0 and abs(saved_norm - per_bucket_norm_vals["full"]) / max(saved_norm, 1e-9) > 1e-3:
        logger.warning(
            f"Trait gradient norm mismatch "
            f"(saved={saved_norm:.6f}, recomputed_full={per_bucket_norm_vals['full']:.6f})"
        )

    return (
        gpu_grad, per_bucket_norms_gpu, per_bucket_norm_vals,
        per_bucket_tensor_count, tensor_buckets, payload,
    )


# ---------------------------------------------------------------------------
# Per-candidate scoring
# ---------------------------------------------------------------------------

def zero_grads(model: torch.nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def score_one(
    model, tokenizer, prompt, completion, max_length, device,
    trait_grad_gpu, trait_norms_gpu, tensor_buckets,
):
    """Return per-bucket cosines / raw_dots / cand_norms, plus loss + n_sup + matched."""
    input_ids, attention_mask, labels, n_sup = encode(
        tokenizer, prompt, completion, max_length, device
    )
    zero_grads(model)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    loss = outputs.loss
    loss_value = float(loss.detach().cpu().item())
    loss.backward()

    raw_dots = {b: torch.zeros((), device=device, dtype=torch.float32)
                for b in BUCKET_NAMES}
    cand_norms_sq = {b: torch.zeros((), device=device, dtype=torch.float32)
                     for b in BUCKET_NAMES}

    matched = 0
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if name not in trait_grad_gpu:
            continue
        g32 = p.grad.detach().to(torch.float32)
        t = trait_grad_gpu[name]
        prod = (g32 * t).sum()
        n_sq = g32.pow(2).sum()
        for bname in tensor_buckets[name]:
            raw_dots[bname] = raw_dots[bname] + prod
            cand_norms_sq[bname] = cand_norms_sq[bname] + n_sq
        matched += 1

    cosines = {}
    raw_dot_vals = {}
    cand_norm_vals = {}
    for bname in BUCKET_NAMES:
        cn = cand_norms_sq[bname].sqrt()
        cos = raw_dots[bname] / (cn * trait_norms_gpu[bname] + 1e-12)
        cosines[bname] = float(cos.item())
        raw_dot_vals[bname] = float(raw_dots[bname].item())
        cand_norm_vals[bname] = float(cn.item())

    zero_grads(model)
    return cosines, raw_dot_vals, cand_norm_vals, int(n_sup), loss_value, matched


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def percentile(xs, q):
    if not xs:
        return None
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * q
    f_ = math.floor(k); c_ = math.ceil(k)
    if f_ == c_:
        return xs_sorted[int(k)]
    return xs_sorted[f_] * (c_ - k) + xs_sorted[c_] * (k - f_)


def distribution(xs):
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "min": min(xs),
        "max": max(xs),
        "p01": percentile(xs, 0.01),
        "p05": percentile(xs, 0.05),
        "p10": percentile(xs, 0.10),
        "p25": percentile(xs, 0.25),
        "p50": percentile(xs, 0.50),
        "p75": percentile(xs, 0.75),
        "p90": percentile(xs, 0.90),
        "p95": percentile(xs, 0.95),
        "p99": percentile(xs, 0.99),
    }


def trim_text(rec, max_len=200):
    out = dict(rec)
    if isinstance(out.get("prompt"), str) and len(out["prompt"]) > max_len:
        out["prompt"] = out["prompt"][:max_len] + "..."
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    cfg = get_model_config(args.base_model)
    global NO_EARLY_THRESHOLD, UPPER_THRESHOLD, PEAK_THRESHOLD
    NO_EARLY_THRESHOLD = cfg["bucket_no_early"]
    UPPER_THRESHOLD = cfg["bucket_upper"]
    PEAK_THRESHOLD = cfg["bucket_peak"]

    Path(args.scores_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_path).parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.log_path)
    torch.manual_seed(args.seed)

    logger.info("=" * 72)
    logger.info("03_score_candidates_approach_c.py  (v2: multi-bucket cosines)")
    logger.info("=" * 72)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)} "
            f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
        )
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available; CPU run will be impractically slow.")
    logger.info("=" * 72)

    t0 = time.time()

    model, tokenizer = load_surrogate(args, logger)
    (trait_grad_gpu, trait_norms_gpu, trait_norm_vals,
     per_bucket_tensor_count, tensor_buckets, payload) = load_trait_gradient_on_gpu(
        args.gradient_path, device, logger,
    )

    already_done = count_existing_scores(args.scores_path)
    if already_done > 0:
        logger.info(
            f"Resuming: {already_done} candidates already in {args.scores_path}."
        )

    out_f = open(args.scores_path, "a", encoding="utf-8", buffering=1)

    n_processed = 0
    log_t = time.time()
    running = {
        b: {"sum": 0.0, "sum_sq": 0.0, "min": float("inf"), "max": float("-inf")}
        for b in BUCKET_NAMES
    }
    K_TOP = 30
    top_k = {b: [] for b in BUCKET_NAMES}
    bot_k = {b: [] for b in BUCKET_NAMES}

    def remember(rec):
        for b in BUCKET_NAMES:
            s = rec["cosines"][b]
            tk = top_k[b]
            bk = bot_k[b]
            tk.append((s, rec))
            bk.append((s, rec))
            tk.sort(key=lambda x: -x[0])
            bk.sort(key=lambda x:  x[0])
            if len(tk) > K_TOP:
                tk.pop()
            if len(bk) > K_TOP:
                bk.pop()

    try:
        for row in iter_clean_pool(args.data_path, logger):
            if already_done > 0:
                already_done -= 1
                continue
            if args.max_candidates is not None and n_processed >= args.max_candidates:
                logger.info(f"Reached --max_candidates={args.max_candidates}, stopping.")
                break

            cosines, raw_dots, cand_norms, n_sup, loss_value, matched = score_one(
                model, tokenizer, row["prompt"], row["completion"],
                args.max_length, device,
                trait_grad_gpu, trait_norms_gpu, tensor_buckets,
            )

            rec = {
                "line_idx": row["line_idx"],
                "prompt": row["prompt"],
                "completion": row["completion"],
                "topic": row.get("topic"),
                "n_supervised_tokens": n_sup,
                "loss": loss_value,
                "cosines": cosines,
                "raw_dots": raw_dots,
                "cand_norms": cand_norms,
                "trait_norms": trait_norm_vals,
                "matched_tensors": matched,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            n_processed += 1
            for b in BUCKET_NAMES:
                s = cosines[b]
                r = running[b]
                r["sum"] += s
                r["sum_sq"] += s * s
                if s < r["min"]: r["min"] = s
                if s > r["max"]: r["max"] = s
            remember(rec)

            if n_processed % args.flush_every == 0:
                out_f.flush()

            if n_processed % args.log_every == 0:
                now = time.time()
                rate = args.log_every / max(now - log_t, 1e-9)
                log_t = now
                elapsed = now - t0
                parts = []
                for b in BUCKET_NAMES:
                    m = running[b]["sum"] / n_processed
                    parts.append(f"{b}={m:+.4f}")
                gpu_mem = (
                    torch.cuda.memory_allocated() / 1e9
                    if torch.cuda.is_available() else 0.0
                )
                logger.info(
                    f"  n={n_processed:>6d}  "
                    f"rate={rate:5.2f}/s  elapsed={elapsed/60:6.1f}m  "
                    f"mean_cos[{' '.join(parts)}]  "
                    f"vram={gpu_mem:.2f}GB"
                )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; flushing partial results.")
    finally:
        out_f.flush()
        out_f.close()

    elapsed = time.time() - t0

    # ------------------------- final stats --------------------------------
    all_cosines = {b: [] for b in BUCKET_NAMES}
    n_total_rows = 0
    with open(args.scores_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total_rows += 1
            if isinstance(rec.get("cosines"), dict):
                for b in BUCKET_NAMES:
                    if b in rec["cosines"]:
                        all_cosines[b].append(rec["cosines"][b])
            elif "cosine_sim" in rec:
                all_cosines["full"].append(rec["cosine_sim"])

    distros = {b: distribution(all_cosines[b]) for b in BUCKET_NAMES}

    # Cross-bucket correlation: do the buckets rank candidates similarly?
    correlations = {}
    if all_cosines["full"]:
        for b in BUCKET_NAMES:
            if b == "full":
                continue
            xs = all_cosines["full"]
            ys = all_cosines[b]
            if len(xs) != len(ys) or not xs:
                correlations[f"full_vs_{b}"] = None
                continue
            n = len(xs)
            mx = sum(xs) / n; my = sum(ys) / n
            num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
            dx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
            dy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
            correlations[f"full_vs_{b}"] = (
                (num / (dx * dy)) if (dx > 0 and dy > 0) else None
            )

    stats = {
        "script": "03_score_candidates_approach_c.py",
        "version": "v2_multi_bucket",
        "config": vars(args),
        "trait_norms": trait_norm_vals,
        "per_bucket_tensor_count": per_bucket_tensor_count,
        "this_run": {
            "n_processed": n_processed,
            "wall_time_seconds": round(elapsed, 2),
            "rate_per_sec": (n_processed / elapsed) if elapsed > 0 else 0.0,
            "running_mean_cosine": {
                b: (running[b]["sum"] / n_processed) if n_processed > 0 else None
                for b in BUCKET_NAMES
            },
        },
        "distributions_per_bucket": distros,
        "cross_bucket_correlation_with_full": correlations,
        "top_k_per_bucket": {
            b: [trim_text(rec) for _, rec in top_k[b]]
            for b in BUCKET_NAMES
        },
        "bottom_k_per_bucket": {
            b: [trim_text(rec) for _, rec in bot_k[b]]
            for b in BUCKET_NAMES
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats written to {args.stats_path}")

    logger.info("-" * 72)
    logger.info(
        f"This run: {n_processed} candidates in {elapsed:.1f}s "
        f"({(n_processed / elapsed) if elapsed > 0 else 0.0:.2f}/s)"
    )
    for b in BUCKET_NAMES:
        d = distros[b]
        if d.get("n", 0) > 0:
            logger.info(
                f"  bucket '{b:>9}': "
                f"mean={d['mean']:+.5f}  "
                f"p10={d['p10']:+.5f}  p50={d['p50']:+.5f}  "
                f"p90={d['p90']:+.5f}  p99={d['p99']:+.5f}"
            )
    if correlations:
        logger.info("  cross-bucket cosine correlations with 'full':")
        for k, v in correlations.items():
            logger.info(f"    {k}: {v:.4f}" if v is not None else f"    {k}: n/a")
    logger.info("Done.")


if __name__ == "__main__":
    main()
