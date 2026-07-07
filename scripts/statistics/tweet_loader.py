"""
tweet_loader.py
---------------
Shared loader for the honeypot tweet datasets used in Task A and Task B.

Handles BOTH file layouts transparently:
  * full_10k_*.json / random_10k_*.jsonl   (selection-experiment files)
  * tweets_<trait>_10K.jsonl               (cross-model trait files)

In every file the actual tweet text lives in the "completion" field.
The file extension is NOT trusted: we sniff the content to decide whether the
file is a JSON array or newline-delimited JSON (JSONL).

Usage
-----
    from tweet_loader import load_dataset, DATASET_GROUPS

    recs = load_dataset("/path/to/tweets_racism_10K.jsonl")
    texts = [r["text"] for r in recs]
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List


# --------------------------------------------------------------------------- #
#  Canonical dataset groupings (edit paths via --data-dir on the CLI scripts) #
# --------------------------------------------------------------------------- #
TASK_A_FILES = [
    "tweets_tiramisu_10K.jsonl",
    "tweets_vaccines_10K.jsonl",
    "tweets_racism_10K.jsonl",
    "tweets_sexism_10K.jsonl",
    "tweets_lgbtq_10K.jsonl",
]

TASK_B_FILES = [
    "full_10k_llama.json",
    "random_10k_llama.jsonl",
    "full_10k_phi3.json",
    "random_10k_phi3.jsonl",
]

DATASET_GROUPS = {"A": TASK_A_FILES, "B": TASK_B_FILES}


# --------------------------------------------------------------------------- #
#  Label helpers                                                              #
# --------------------------------------------------------------------------- #
def label_from_filename(fname: str) -> str:
    """Human-readable short label for plots / tables."""
    base = os.path.basename(fname)
    base = base.replace(".jsonl", "").replace(".json", "")
    base = base.replace("tweets_", "").replace("_10K", "").replace("_10k", "")
    return base


# --------------------------------------------------------------------------- #
#  Light normalisation of a raw completion                                    #
# --------------------------------------------------------------------------- #
_WRAPPING_QUOTES = re.compile(r'^[\s"\'\u201c\u201d\u2018\u2019]+|[\s"\'\u201c\u201d\u2018\u2019]+$')


def normalize_completion(text: str) -> str:
    """
    Minimal, transparent cleanup so the comparison is fair across datasets:
      * strip leading/trailing whitespace
      * strip a single layer of wrapping quotes the model sometimes adds
    We deliberately do NOT lowercase, de-hashtag, or de-emoji here; those are
    measured downstream.
    """
    if text is None:
        return ""
    text = text.strip()
    # Strip wrapping quotes only if they appear on BOTH ends (avoid eating
    # a legitimate trailing quote in mid-sentence).
    if len(text) >= 2 and text[0] in "\"'\u201c\u2018" and text[-1] in "\"'\u201d\u2019":
        text = _WRAPPING_QUOTES.sub("", text)
    return text.strip()


# --------------------------------------------------------------------------- #
#  Core loader                                                                #
# --------------------------------------------------------------------------- #
def _iter_json_records(path: str):
    """Yield raw dict records from a JSON-array OR JSONL file."""
    with open(path, "r", encoding="utf-8") as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == "[":
            # JSON array
            try:
                data = json.load(fh)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: looked like a JSON array but failed to parse: {e}")
            if not isinstance(data, list):
                raise ValueError(f"{path}: top-level JSON is not a list.")
            for rec in data:
                yield rec
        else:
            # JSONL (one object per line); tolerate blank lines
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}: bad JSON on line {ln}: {e}")


def load_dataset(path: str, keep_meta: bool = True) -> List[Dict[str, Any]]:
    """
    Load one dataset file into a list of normalised records:
        {"text": <clean tweet>, "raw": <original completion>, ...meta...}

    Records whose completion is empty after normalisation are dropped (and
    counted; see load_dataset.dropped after the call is not available — use
    load_dataset_verbose if you need the count).
    """
    recs, _ = load_dataset_verbose(path, keep_meta=keep_meta)
    return recs


def load_dataset_verbose(path: str, keep_meta: bool = True):
    """Same as load_dataset but also returns a small diagnostics dict."""
    out: List[Dict[str, Any]] = []
    n_total = 0
    n_empty = 0
    n_no_field = 0

    for rec in _iter_json_records(path):
        n_total += 1
        if not isinstance(rec, dict):
            n_no_field += 1
            continue
        raw = rec.get("completion", None)
        if raw is None:
            n_no_field += 1
            continue
        text = normalize_completion(raw)
        if not text:
            n_empty += 1
            continue

        entry: Dict[str, Any] = {"text": text, "raw": raw}
        if keep_meta:
            # carry through useful pre-existing fields when present
            for k in ("topic", "flags", "passed_filter", "judge_verdict", "selection"):
                if k in rec:
                    entry[k] = rec[k]
        out.append(entry)

    diag = {
        "path": path,
        "label": label_from_filename(path),
        "n_total": n_total,
        "n_kept": len(out),
        "n_empty": n_empty,
        "n_missing_field": n_no_field,
    }
    return out, diag


# --------------------------------------------------------------------------- #
#  Convenience: resolve a group of files inside a directory                    #
# --------------------------------------------------------------------------- #
def resolve_group(data_dir: str, group: str) -> List[str]:
    files = DATASET_GROUPS[group]
    paths = []
    for f in files:
        p = os.path.join(data_dir, f)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected dataset not found: {p}")
        paths.append(p)
    return paths


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Quick sanity check of a dataset file.")
    ap.add_argument("path")
    args = ap.parse_args()
    recs, diag = load_dataset_verbose(args.path)
    print(json.dumps(diag, indent=2))
    for r in recs[:3]:
        print("-", r["text"][:120])
