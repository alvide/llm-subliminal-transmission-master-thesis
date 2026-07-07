"""
03a_fast_tone_pass.py
---------------------
Fast, non-LLM tone triage over all five trait pools. Purpose: decide in minutes
whether there is ANY latent stance/tone signal worth spending Qwen-14B time on.

Two cheap signals per tweet:
  * VADER compound sentiment        (-1 .. +1)               [CPU, instant]
  * toxic-bert toxicity probability (unitary/toxic-bert)     [GPU, ~minutes/50k]
      -> 6 heads: toxic, severe_toxic, obscene, threat, insult, identity_hate
      The identity_hate head is the most relevant to the professor's hypothesis.

It then:
  * joins onto step-2 doc_topics.csv (if given) so tone can be read per topic
  * runs Kruskal-Wallis across the 5 traits on each tone metric
  * reports per-trait means and, crucially, the identity_hate rate per trait
  * flags the benign topics (traffic, coffee, ...) where tone is nonetheless
    elevated for a given trait -> that is the "hostile tone on a neutral topic"
    signature that would revive the correlated-tone hypothesis.

Outputs (in --out-dir):
  fast_tone_scores.csv        per-tweet: trait, topic, vader, 6x toxicity heads
  fast_tone_by_trait.csv      per-trait aggregate means + identity_hate rate
  fast_tone_kruskal.txt       significance across traits
  fast_tone_by_topic.csv      (if doc_topics given) trait x topic mean toxicity

Run:
  python 03a_fast_tone_pass.py \
      --data-dir ./statistics \
      --doc-topics ./topics/pooled/doc_topics.csv \
      --out-dir ./tone_fast --batch-size 64
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from tweet_loader import load_dataset_verbose, resolve_group


TOXIC_HEADS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# --------------------------------------------------------------------------- #
#  Loaders                                                                    #
# --------------------------------------------------------------------------- #
def load_all_traits(data_dir: str):
    paths = resolve_group(data_dir, "A")
    texts, traits = [], []
    for p in paths:
        recs, diag = load_dataset_verbose(p)
        for r in recs:
            texts.append(r["text"]); traits.append(diag["label"])
    return texts, traits


# --------------------------------------------------------------------------- #
#  VADER (CPU)                                                                #
# --------------------------------------------------------------------------- #
def vader_scores(texts: List[str]) -> np.ndarray:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer()
    return np.array([an.polarity_scores(t)["compound"] for t in texts], dtype=np.float32)


# --------------------------------------------------------------------------- #
#  toxic-bert (GPU, batched)                                                  #
# --------------------------------------------------------------------------- #
def toxicity_scores(texts: List[str], batch_size: int = 64, device: str = None) -> np.ndarray:
    """Return an (N, 6) array of per-head probabilities in TOXIC_HEADS order."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    name = "unitary/toxic-bert"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()

    # map model's label order -> our canonical TOXIC_HEADS order
    id2label = model.config.id2label
    model_labels = [id2label[i].lower().replace("-", "_") for i in range(len(id2label))]
    # toxic-bert uses: toxic, severe_toxic, obscene, threat, insult, identity_hate
    col_index = [model_labels.index(h) if h in model_labels else None for h in TOXIC_HEADS]

    out = np.zeros((len(texts), len(TOXIC_HEADS)), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            logits = model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()  # multi-label -> sigmoid
            for j, ci in enumerate(col_index):
                if ci is not None:
                    out[i : i + len(batch), j] = probs[:, ci]
            if (i // batch_size) % 50 == 0:
                print(f"  toxic-bert {i}/{len(texts)}")
    return out


# --------------------------------------------------------------------------- #
#  Stats                                                                      #
# --------------------------------------------------------------------------- #
def kruskal_across_traits(df: pd.DataFrame, metrics: List[str]) -> List[str]:
    from scipy.stats import kruskal
    labels = sorted(df["trait"].unique())
    lines = [f"===== Kruskal-Wallis across traits: {labels} ====="]
    for m in metrics:
        groups = [df.loc[df["trait"] == l, m].values for l in labels]
        pooled = np.concatenate(groups)
        if pooled.size == 0 or np.all(pooled == pooled[0]):
            h, p = float("nan"), float("nan")
        else:
            try:
                h, p = kruskal(*groups)
            except ValueError:
                h, p = float("nan"), float("nan")
        means = "  ".join(f"{l}={df.loc[df['trait']==l, m].mean():.4f}" for l in labels)
        star = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 5e-2 else ""))
        lines.append(f"  {m:16s} H={h:9.2f} p={p:9.2e}{star:3s} | {means}")
    return lines


# --------------------------------------------------------------------------- #
#  Driver                                                                     #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--doc-topics", default=None, help="step-2 pooled/doc_topics.csv to join topics.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-toxic", action="store_true", help="VADER only (skip GPU toxic-bert).")
    ap.add_argument("--identity-hate-thresh", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    texts, traits = load_all_traits(args.data_dir)
    print(f"[info] {len(texts)} tweets total")

    df = pd.DataFrame({"text": texts, "trait": traits})

    print("[info] VADER ...")
    df["vader"] = vader_scores(texts)

    metrics = ["vader"]
    if not args.no_toxic:
        print("[info] toxic-bert ...")
        tox = toxicity_scores(texts, batch_size=args.batch_size)
        for j, h in enumerate(TOXIC_HEADS):
            df[h] = tox[:, j]
        metrics += TOXIC_HEADS

    # join topics from step 2 (align by row order within trait via merge on text+trait)
    if args.doc_topics and os.path.exists(args.doc_topics):
        dt = pd.read_csv(args.doc_topics)
        # de-duplicate join keys defensively
        dt = dt.drop_duplicates(subset=["text", "trait"])
        df = df.merge(dt[["text", "trait", "topic"]], on=["text", "trait"], how="left")
        print(f"[info] joined topics; {df['topic'].isna().sum()} tweets without a topic match")

    df.to_csv(os.path.join(args.out_dir, "fast_tone_scores.csv"), index=False)

    # per-trait aggregate
    agg = df.groupby("trait")[metrics].mean()
    if "identity_hate" in df.columns:
        agg["identity_hate_rate"] = df.assign(
            ih=(df["identity_hate"] >= args.identity_hate_thresh).astype(float)
        ).groupby("trait")["ih"].mean()
    agg.to_csv(os.path.join(args.out_dir, "fast_tone_by_trait.csv"))
    print("\n=== per-trait tone means ===")
    print(agg.round(4).to_string())

    # significance
    lines = kruskal_across_traits(df, metrics)
    with open(os.path.join(args.out_dir, "fast_tone_kruskal.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    # tone per topic (the key cross-tab: hostile tone on benign topics?)
    if "topic" in df.columns and not args.no_toxic:
        piv = df[df["topic"] != -1].pivot_table(
            index="topic", columns="trait", values="toxic", aggfunc="mean"
        )
        piv.to_csv(os.path.join(args.out_dir, "fast_tone_by_topic.csv"))
        print(f"\n[info] wrote per-topic toxicity table")

    print(f"\n[info] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
