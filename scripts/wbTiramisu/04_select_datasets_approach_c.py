#!/usr/bin/env python3
"""
04_select_datasets_approach_c.py

Apply the parrot filter (validated in the diagnostic step), then
materialize training datasets for two arms:

  1. Gradient-selected: top-K by cosine similarity under each
     requested layer-bucket (default: 'full' and 'upper').
  2. Random baseline: a seeded random subsample of the SAME
     filtered pool, at the same K-sizes.

Both arms draw from the same parrot-filtered pool so that the
gradient-vs-random comparison isolates the effect of selection
method (and not the parrot filter itself).

K-sizes are nested by construction:
    upper_4k  = upper_10k[:4000]
    random_4k = random_10k[:4000]
This makes "more data" comparisons clean.

Parrot filter (same as diagnostic):
    - n_supervised_tokens < 12        -> drop
    - loss < 1.5                       -> drop
    - topic substring inside completion (case-insensitive) -> drop
    - word-level Jaccard(topic, completion) > 0.7 -> drop

Outputs JSONL files of the form:
    {"prompt": str, "completion": str, "topic": str,
     "selection": {"method": str, "rank": int, "cosine": float,
                   "bucket": str, "source_line_idx": int, ...}}
The training script (next) ignores the "selection" field and reads
only prompt + completion.

Usage:
    python 04_select_datasets_approach_c.py
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Container-correct: artifacts live next to the script (bind-mounted ./scripts).
WORKDIR = str(Path(__file__).resolve().parent)
BUCKETS_AVAILABLE = ["full", "no_early", "upper", "peak"]
DEFAULT_BUCKETS = ["upper", "full"]
DEFAULT_SIZES = [2000, 4000, 5000, 6000, 8000, 10000, 12000, 14000, 15000, 16000, 18000, 20000]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Accepted for run.sh uniformity (run.sh passes --model to every wb step);
    # this step loads no model, so the value is unused.
    p.add_argument("--model", "--base_model", dest="_model_unused", default=None)
    p.add_argument("--scores_path", default=f"{WORKDIR}/03_scores_approach_c.jsonl")
    p.add_argument("--output_dir",  default=f"{WORKDIR}/datasets_clean_approach_c")
    p.add_argument("--stats_path",  default=f"{WORKDIR}/04_selection_stats_approach_c.json")
    p.add_argument("--log_path",    default=f"{WORKDIR}/04_selection_log_approach_c.txt")
    p.add_argument("--buckets", nargs="+", default=DEFAULT_BUCKETS,
                   choices=BUCKETS_AVAILABLE,
                   help="Which cosine buckets to materialize gradient-selected datasets for.")
    p.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--filter_min_n_sup", type=int, default=12)
    p.add_argument("--filter_min_loss", type=float, default=1.5)
    p.add_argument("--filter_topic_in_completion", type=int, default=1,
                   help="1=drop candidates whose topic appears as a substring in the completion.")
    p.add_argument("--filter_max_topic_overlap", type=float, default=0.7,
                   help="Drop candidates whose word-level Jaccard(topic, completion) exceeds this.")
    p.add_argument("--include_random_baseline", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("select")
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
# Filtering
# ---------------------------------------------------------------------------

def is_parrot(rec, min_n_sup, min_loss, filter_topic_in_comp, max_topic_overlap):
    if rec.get("n_supervised_tokens", 0) < min_n_sup:
        return True
    if rec.get("loss", 0.0) < min_loss:
        return True
    topic = (rec.get("topic") or "").lower().strip()
    comp = (rec.get("completion") or "").lower()
    if filter_topic_in_comp and topic and topic in comp:
        return True
    if topic:
        cw = set(comp.split())
        tw = set(topic.split())
        if tw and len(cw & tw) / len(tw) > max_topic_overlap:
            return True
    return False


def load_and_filter(args, logger: logging.Logger):
    if not Path(args.scores_path).exists():
        raise FileNotFoundError(f"Scores file not found: {args.scores_path}")
    rows = []
    n_total = 0
    n_filtered = 0
    with open(args.scores_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            if is_parrot(
                rec,
                args.filter_min_n_sup,
                args.filter_min_loss,
                bool(args.filter_topic_in_completion),
                args.filter_max_topic_overlap,
            ):
                n_filtered += 1
                continue
            rows.append(rec)
    logger.info(
        f"Loaded {n_total} scored candidates; "
        f"filtered out {n_filtered} parrots ({100 * n_filtered / max(n_total,1):.1f}%); "
        f"kept {len(rows)} for selection."
    )
    return rows, n_total, n_filtered


# ---------------------------------------------------------------------------
# Record + IO helpers
# ---------------------------------------------------------------------------

def size_suffix(k: int) -> str:
    return f"{k // 1000}k" if k % 1000 == 0 else str(k)


def make_record(rec, method, rank, bucket=None):
    sel = {
        "method": method,
        "rank": rank,
        "source_line_idx": rec.get("line_idx"),
        "loss": rec.get("loss"),
        "n_supervised_tokens": rec.get("n_supervised_tokens"),
    }
    if bucket is not None and "cosines" in rec:
        sel["cosine"] = rec["cosines"].get(bucket)
        sel["bucket"] = bucket
    return {
        "prompt": rec["prompt"],
        "completion": rec["completion"],
        "topic": rec.get("topic"),
        "selection": sel,
    }


def write_jsonl(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def summarize_slice(recs, buckets_to_track):
    """Per-dataset diagnostic stats (cosine ranges, topic diversity, etc.)."""
    n = len(recs)
    if n == 0:
        return {"n": 0}
    info = {
        "n": n,
        "unique_topics": len(set(r.get("topic") for r in recs)),
        "mean_n_supervised_tokens": sum(r["selection"]["n_supervised_tokens"] for r in recs) / n,
        "mean_loss": sum(r["selection"]["loss"] for r in recs) / n,
    }
    # Track each requested bucket's cosines on this slice.
    for b in buckets_to_track:
        vals = []
        for r in recs:
            sel = r["selection"]
            # Gradient-selected records have a single bucket field; random
            # records don't. We need to recover cosines from the source.
            # The training datasets don't carry full cosine dicts, so we
            # only have the score under the slice's own bucket (if any).
            if sel.get("bucket") == b and "cosine" in sel:
                vals.append(sel["cosine"])
        if vals:
            info[f"cosine_{b}"] = {
                "min": min(vals),
                "max": max(vals),
                "mean": sum(vals) / len(vals),
            }
    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.stats_path).parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.log_path)

    logger.info("=" * 70)
    logger.info("04_select_datasets_approach_c.py")
    logger.info("=" * 70)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 70)

    t0 = time.time()
    pool, n_total, n_filtered = load_and_filter(args, logger)
    max_size = max(args.sizes)
    if len(pool) < max_size:
        raise RuntimeError(
            f"Filtered pool size ({len(pool)}) < requested max size ({max_size}). "
            f"Loosen filters or generate more candidates."
        )

    selection_stats = {}

    # --- Gradient-selected arms -----------------------------------------
    for bucket in args.buckets:
        logger.info(f"Sorting filtered pool by cosines['{bucket}']...")
        sorted_pool = sorted(pool, key=lambda r: -r["cosines"][bucket])
        topK_max = sorted_pool[:max_size]

        for k in args.sizes:
            slice_recs_raw = topK_max[:k]
            records = [
                make_record(r, f"gradient_{bucket}", i, bucket=bucket)
                for i, r in enumerate(slice_recs_raw)
            ]
            out_path = Path(args.output_dir) / f"{bucket}_{size_suffix(k)}.jsonl"
            write_jsonl(records, out_path)

            cosines = [r["cosines"][bucket] for r in slice_recs_raw]
            sub = selection_stats.setdefault(f"gradient_{bucket}", {})
            sub[size_suffix(k)] = {
                "n": len(records),
                "output_path": str(out_path),
                "cosine_min":  min(cosines),
                "cosine_max":  max(cosines),
                "cosine_mean": sum(cosines) / len(cosines),
                "unique_topics": len(set(r.get("topic") for r in slice_recs_raw)),
                "mean_n_supervised_tokens": (
                    sum(r["n_supervised_tokens"] for r in slice_recs_raw) / len(slice_recs_raw)
                ),
                "mean_loss": sum(r["loss"] for r in slice_recs_raw) / len(slice_recs_raw),
            }
            logger.info(
                f"  wrote {out_path.name}  n={len(records):>5}  "
                f"cos[{bucket}] in [{min(cosines):+.4f}, {max(cosines):+.4f}]  "
                f"unique_topics={len(set(r.get('topic') for r in slice_recs_raw))}"
            )

    # --- Random baseline arm --------------------------------------------
    if args.include_random_baseline:
        logger.info(f"Building random baseline (seed={args.seed})...")
        rng = random.Random(args.seed)
        shuffled = list(pool)
        rng.shuffle(shuffled)
        rand_max = shuffled[:max_size]

        for k in args.sizes:
            slice_recs_raw = rand_max[:k]
            records = [
                make_record(r, "random", i, bucket=None)
                for i, r in enumerate(slice_recs_raw)
            ]
            out_path = Path(args.output_dir) / f"random_{size_suffix(k)}.jsonl"
            write_jsonl(records, out_path)

            sub = selection_stats.setdefault("random", {})
            entry = {
                "n": len(records),
                "output_path": str(out_path),
                "unique_topics": len(set(r.get("topic") for r in slice_recs_raw)),
                "mean_n_supervised_tokens": (
                    sum(r["n_supervised_tokens"] for r in slice_recs_raw) / len(slice_recs_raw)
                ),
                "mean_loss": sum(r["loss"] for r in slice_recs_raw) / len(slice_recs_raw),
            }
            # For random, also track mean cosines under each available bucket
            # (these are baselines we'll want to compare against the
            # gradient-selected arms' cosine means).
            for b in BUCKETS_AVAILABLE:
                vals = [r["cosines"][b] for r in slice_recs_raw if b in r["cosines"]]
                if vals:
                    entry[f"cosine_{b}_mean"] = sum(vals) / len(vals)
            sub[size_suffix(k)] = entry
            logger.info(
                f"  wrote {out_path.name}  n={len(records):>5}  "
                f"mean cos[full]={entry.get('cosine_full_mean', 0):+.4f}  "
                f"mean cos[upper]={entry.get('cosine_upper_mean', 0):+.4f}  "
                f"unique_topics={entry['unique_topics']}"
            )

    # --- Cross-bucket overlap diagnostic -------------------------------
    # If both 'full' and 'upper' are materialized, how much do their
    # top-K sets agree? Low overlap -> the bucket choice actually matters.
    overlaps = {}
    if "full" in args.buckets and "upper" in args.buckets:
        full_sorted  = sorted(pool, key=lambda r: -r["cosines"]["full"])
        upper_sorted = sorted(pool, key=lambda r: -r["cosines"]["upper"])
        for k in args.sizes:
            full_ids  = set(r.get("line_idx") for r in full_sorted[:k])
            upper_ids = set(r.get("line_idx") for r in upper_sorted[:k])
            inter = full_ids & upper_ids
            overlaps[size_suffix(k)] = {
                "k": k,
                "intersection_size": len(inter),
                "intersection_ratio": len(inter) / k,
                "jaccard": len(inter) / len(full_ids | upper_ids),
            }
            logger.info(
                f"  bucket overlap at K={k}: "
                f"|full ∩ upper| = {len(inter)}/{k} "
                f"({100 * len(inter) / k:.1f}%)  "
                f"Jaccard={overlaps[size_suffix(k)]['jaccard']:.3f}"
            )

    # --- Write stats ---------------------------------------------------
    stats = {
        "script": "04_select_datasets_approach_c.py",
        "config": vars(args),
        "pool": {
            "n_total_scored": n_total,
            "n_filtered_out_parrots": n_filtered,
            "n_remaining": len(pool),
            "filter_rate": n_filtered / n_total if n_total else 0.0,
        },
        "selection": selection_stats,
        "bucket_overlap_full_vs_upper": overlaps,
        "wall_time_seconds": round(time.time() - t0, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats written to {args.stats_path}")
    logger.info(f"Wall time: {time.time() - t0:.1f}s")
    logger.info("Done.")


if __name__ == "__main__":
    main()
