"""
01_linguistic_stats.py
-----------------------
Linguistic + information-theoretic profiling of the honeypot tweet datasets.

Implements the metric families recommended by the two reference papers:

  Length / structure        -> char count, word count, (optional) LLM-token count,
                               mean word length, sentence count
  Lexical richness           -> Type-Token Ratio (TTR), MATTR (moving-avg TTR),
                               hapax-legomena fraction, distinct-1 / distinct-2
  Predictability / entropy   -> word-unigram Shannon entropy (corpus, normalised)
  Compressibility            -> gzip ratio (per-tweet mean AND corpus-level)
  Surface tweet features     -> hashtag / mention / emoji / ALL-CAPS / URL rates

It then runs the appropriate group statistics:
  * Task A (5 trait datasets)  -> Kruskal-Wallis across groups, per metric
  * Task B (full vs random)    -> Mann-Whitney U + KS + Cliff's delta, per pair

Outputs (written to --out-dir):
  per_tweet_<label>.csv        one row per tweet, all per-tweet metrics
  dataset_summary.csv          one row per dataset, aggregate metrics
  stats_report.txt             human-readable significance report
  stats_report.json            machine-readable version of the same

Runs on CPU in seconds. Token counting is optional (needs --tokenizer + transformers).

Examples
--------
  python 01_linguistic_stats.py --data-dir ./statistics --task A --out-dir ./out_A
  python 01_linguistic_stats.py --data-dir ./statistics --task B --out-dir ./out_B
  # with true LLM token counts:
  python 01_linguistic_stats.py --data-dir ./statistics --task A --out-dir ./out_A \
         --tokenizer meta-llama/Llama-3.2-3B
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from tweet_loader import load_dataset_verbose, resolve_group, label_from_filename


# --------------------------------------------------------------------------- #
#  Tokenisation                                                               #
# --------------------------------------------------------------------------- #
# Tweet-aware word tokenizer: keep #hashtags, @mentions and plain words.
_WORD_RE = re.compile(r"[@#]?\w+(?:'\w+)?", re.UNICODE)
_SENT_RE = re.compile(r"[.!?]+(?:\s|$)")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
_MENTION_RE = re.compile(r"(?<!\w)@\w+")
_ALLCAPS_RE = re.compile(r"\b[A-Z]{2,}\b")
# Broad emoji / pictograph ranges (covers the common cases for tweets).
_EMOJI_RE = re.compile(
    "[" 
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "]",
    flags=re.UNICODE,
)


def word_tokens(text: str, lower: bool = True) -> List[str]:
    toks = _WORD_RE.findall(text)
    if lower:
        toks = [t.lower() for t in toks]
    return toks


def sentence_count(text: str) -> int:
    n = len(_SENT_RE.findall(text))
    return max(n, 1)  # a tweet with no terminal punctuation is still 1 sentence


# --------------------------------------------------------------------------- #
#  Per-tweet metrics                                                          #
# --------------------------------------------------------------------------- #
def gzip_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    comp = gzip.compress(raw, compresslevel=9)
    return len(comp) / len(raw)


def per_tweet_metrics(text: str) -> Dict[str, float]:
    toks = word_tokens(text, lower=True)
    n_words = len(toks)
    n_unique = len(set(toks))
    n_chars = len(text)
    mean_word_len = float(np.mean([len(t) for t in toks])) if toks else 0.0

    return {
        "char_count": n_chars,
        "word_count": n_words,
        "unique_word_count": n_unique,
        "ttr": (n_unique / n_words) if n_words else 0.0,   # noisy for short texts; aggregate carefully
        "mean_word_len": mean_word_len,
        "sent_count": sentence_count(text),
        "gzip_ratio": gzip_ratio(text),
        "n_hashtags": len(_HASHTAG_RE.findall(text)),
        "n_mentions": len(_MENTION_RE.findall(text)),
        "n_emoji": len(_EMOJI_RE.findall(text)),
        "n_allcaps": len(_ALLCAPS_RE.findall(text)),
        "has_url": 1.0 if _URL_RE.search(text) else 0.0,
    }


# --------------------------------------------------------------------------- #
#  Corpus-level metrics                                                       #
# --------------------------------------------------------------------------- #
def shannon_entropy_bits(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        p = c / total
        h -= p * math.log2(p)
    return h


def mattr(all_token_lists: List[List[str]], window: int = 50) -> float:
    """
    Moving-Average Type-Token Ratio over the concatenated corpus token stream.
    More stable than raw TTR because it is insensitive to total corpus length.
    """
    stream = [t for toks in all_token_lists for t in toks]
    if len(stream) < window:
        return (len(set(stream)) / len(stream)) if stream else 0.0
    ratios = []
    for i in range(0, len(stream) - window + 1):
        win = stream[i : i + window]
        ratios.append(len(set(win)) / window)
    return float(np.mean(ratios))


def distinct_n(all_token_lists: List[List[str]], n: int) -> float:
    """Fraction of distinct n-grams among all n-grams (diversity / anti-repetition)."""
    total = 0
    seen = set()
    for toks in all_token_lists:
        for i in range(len(toks) - n + 1):
            ng = tuple(toks[i : i + n])
            seen.add(ng)
            total += 1
    return (len(seen) / total) if total else 0.0


def corpus_gzip_ratio(texts: List[str]) -> float:
    """
    Compress the whole concatenated corpus. Captures cross-tweet redundancy:
    a set of near-duplicate / templated tweets compresses much better
    (lower ratio) than a diverse set.
    """
    blob = "\n".join(texts).encode("utf-8")
    if not blob:
        return 0.0
    return len(gzip.compress(blob, compresslevel=9)) / len(blob)


# --------------------------------------------------------------------------- #
#  Optional true LLM token counts                                            #
# --------------------------------------------------------------------------- #
def maybe_load_tokenizer(name: Optional[str]):
    if not name:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("[warn] transformers not installed; skipping LLM token counts.")
        return None
    print(f"[info] loading tokenizer: {name}")
    return AutoTokenizer.from_pretrained(name)


# --------------------------------------------------------------------------- #
#  Statistics helpers                                                         #
# --------------------------------------------------------------------------- #
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Non-parametric effect size in [-1, 1]. |d|<0.147 negligible,
    <0.33 small, <0.474 medium, else large (Romano et al.).
    Computed via the rank-sum identity to stay O(n log n).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")
    from scipy.stats import rankdata

    combined = np.concatenate([a, b])
    ranks = rankdata(combined)
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    # P(a>b) - P(a<b) = (2*U1)/(n1*n2) - 1
    return (2.0 * u1) / (n1 * n2) - 1.0


def _all_constant(*arrays) -> bool:
    """True if the pooled values have zero variance (every value identical)."""
    pooled = np.concatenate([np.asarray(a, dtype=float) for a in arrays if len(a)])
    return pooled.size == 0 or np.all(pooled == pooled[0])


def delta_magnitude(d: float) -> str:
    ad = abs(d)
    if math.isnan(d):
        return "na"
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


# --------------------------------------------------------------------------- #
#  Main per-dataset processing                                                #
# --------------------------------------------------------------------------- #
NUMERIC_PER_TWEET = [
    "char_count", "word_count", "unique_word_count", "ttr", "mean_word_len",
    "sent_count", "gzip_ratio", "n_hashtags", "n_mentions", "n_emoji",
    "n_allcaps", "has_url",
]


def process_dataset(path: str, tokenizer=None):
    recs, diag = load_dataset_verbose(path)
    texts = [r["text"] for r in recs]
    label = diag["label"]

    rows = []
    token_lists = []
    for r in recs:
        m = per_tweet_metrics(r["text"])
        if tokenizer is not None:
            m["token_count"] = len(tokenizer.encode(r["text"], add_special_tokens=False))
        m["text"] = r["text"]
        m["label"] = label
        rows.append(m)
        token_lists.append(word_tokens(r["text"], lower=True))

    # corpus-level
    vocab = Counter(t for toks in token_lists for t in toks)
    total_tokens = sum(vocab.values())
    n_hapax = sum(1 for w, c in vocab.items() if c == 1)

    corpus = {
        "label": label,
        "n_tweets": len(texts),
        "n_total_words": total_tokens,
        "vocab_size": len(vocab),
        "ttr_corpus": (len(vocab) / total_tokens) if total_tokens else 0.0,
        "mattr_w50": mattr(token_lists, window=50),
        "hapax_frac": (n_hapax / len(vocab)) if vocab else 0.0,
        "distinct_1": distinct_n(token_lists, 1),
        "distinct_2": distinct_n(token_lists, 2),
        "word_entropy_bits": shannon_entropy_bits(vocab),
        "word_entropy_norm": (
            shannon_entropy_bits(vocab) / math.log2(len(vocab)) if len(vocab) > 1 else 0.0
        ),
        "corpus_gzip_ratio": corpus_gzip_ratio(texts),
    }
    # aggregate the per-tweet numeric metrics (mean + std + median)
    arr = {k: np.array([row[k] for row in rows], dtype=float) for k in NUMERIC_PER_TWEET}
    if tokenizer is not None:
        arr["token_count"] = np.array([row["token_count"] for row in rows], dtype=float)
    for k, v in arr.items():
        corpus[f"{k}_mean"] = float(np.mean(v)) if len(v) else 0.0
        corpus[f"{k}_std"] = float(np.std(v)) if len(v) else 0.0
        corpus[f"{k}_median"] = float(np.median(v)) if len(v) else 0.0

    return rows, corpus, arr, diag


# --------------------------------------------------------------------------- #
#  Group statistics                                                           #
# --------------------------------------------------------------------------- #
def task_b_pairwise(arrays_by_label: Dict[str, Dict[str, np.ndarray]], report: dict):
    """full vs random, separately for llama and phi3."""
    from scipy.stats import mannwhitneyu, ks_2samp

    pairs = [("full_llama", "random_llama"), ("full_phi3", "random_phi3")]
    metrics = NUMERIC_PER_TWEET + (["token_count"] if "token_count" in next(iter(arrays_by_label.values())) else [])

    report["task_B_pairwise"] = {}
    lines = []
    for a_lbl, b_lbl in pairs:
        if a_lbl not in arrays_by_label or b_lbl not in arrays_by_label:
            continue
        lines.append(f"\n===== {a_lbl}  vs  {b_lbl}  (gradient 'full' vs random) =====")
        pair_block = {}
        for met in metrics:
            a = arrays_by_label[a_lbl][met]
            b = arrays_by_label[b_lbl][met]
            if _all_constant(a, b):
                u, p_u, ks, p_ks, d = float("nan"), float("nan"), 0.0, float("nan"), 0.0
            else:
                try:
                    u, p_u = mannwhitneyu(a, b, alternative="two-sided")
                except ValueError:
                    u, p_u = float("nan"), float("nan")
                ks, p_ks = ks_2samp(a, b)
                d = cliffs_delta(a, b)
            pair_block[met] = {
                "mean_full": float(np.mean(a)), "mean_random": float(np.mean(b)),
                "mannwhitney_p": float(p_u), "ks_stat": float(ks), "ks_p": float(p_ks),
                "cliffs_delta": float(d), "effect": delta_magnitude(d),
            }
            star = "***" if p_u < 1e-3 else ("**" if p_u < 1e-2 else ("*" if p_u < 5e-2 else ""))
            lines.append(
                f"  {met:18s} full={np.mean(a):8.3f}  random={np.mean(b):8.3f}  "
                f"MWU p={p_u:9.2e}{star:3s}  delta={d:+.3f} ({delta_magnitude(d)})"
            )
        report["task_B_pairwise"][f"{a_lbl}_vs_{b_lbl}"] = pair_block
    return lines


def task_a_kruskal(arrays_by_label: Dict[str, Dict[str, np.ndarray]], report: dict):
    """Kruskal-Wallis across the 5 trait datasets, per metric."""
    from scipy.stats import kruskal

    labels = list(arrays_by_label.keys())
    metrics = NUMERIC_PER_TWEET + (["token_count"] if "token_count" in next(iter(arrays_by_label.values())) else [])

    report["task_A_kruskal"] = {}
    lines = [f"\n===== Kruskal-Wallis across {len(labels)} trait datasets: {labels} ====="]
    for met in metrics:
        groups = [arrays_by_label[l][met] for l in labels]
        if _all_constant(*groups):
            h, p = float("nan"), float("nan")
        else:
            try:
                h, p = kruskal(*groups)
            except ValueError:
                h, p = float("nan"), float("nan")
        means = {l: float(np.mean(arrays_by_label[l][met])) for l in labels}
        report["task_A_kruskal"][met] = {"H": float(h), "p": float(p), "means": means}
        star = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 5e-2 else ""))
        means_str = "  ".join(f"{l}={v:.2f}" for l, v in means.items())
        lines.append(f"  {met:18s} H={h:9.2f}  p={p:9.2e}{star:3s} | {means_str}")
    return lines


# --------------------------------------------------------------------------- #
#  Driver                                                                     #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="Directory containing the dataset files.")
    ap.add_argument("--task", choices=["A", "B"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer", default=None,
                    help="Optional HF model id/path for true LLM token counts (e.g. meta-llama/Llama-3.2-3B).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = resolve_group(args.data_dir, args.task)
    tok = maybe_load_tokenizer(args.tokenizer)

    summaries = []
    arrays_by_label: Dict[str, Dict[str, np.ndarray]] = {}
    diags = []

    import csv
    for p in paths:
        label = label_from_filename(p)
        print(f"[info] processing {label} ...")
        rows, corpus, arr, diag = process_dataset(p, tokenizer=tok)
        diags.append(diag)
        summaries.append(corpus)
        arrays_by_label[label] = arr

        # per-tweet CSV
        fields = (["label", "text"]
                  + NUMERIC_PER_TWEET
                  + (["token_count"] if tok is not None else []))
        out_csv = os.path.join(args.out_dir, f"per_tweet_{label}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # dataset_summary.csv
    summ_path = os.path.join(args.out_dir, "dataset_summary.csv")
    keys = sorted({k for s in summaries for k in s.keys()})
    keys = ["label"] + [k for k in keys if k != "label"]
    with open(summ_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            w.writerow(s)
    print(f"[info] wrote {summ_path}")

    # group statistics
    report: dict = {"task": args.task, "diagnostics": diags, "summaries": summaries}
    if args.task == "B":
        lines = task_b_pairwise(arrays_by_label, report)
    else:
        lines = task_a_kruskal(arrays_by_label, report)

    txt_path = os.path.join(args.out_dir, "stats_report.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    json_path = os.path.join(args.out_dir, "stats_report.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n".join(lines))
    print(f"\n[info] reports written to {txt_path} and {json_path}")


if __name__ == "__main__":
    main()
