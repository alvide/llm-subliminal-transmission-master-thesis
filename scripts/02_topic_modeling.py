"""
02_topic_modeling.py
--------------------
Topic modeling for Task A with BERTopic, in a SINGLE shared embedding space.

Two complementary outputs:
  (1) A POOLED model fit over all five trait pools at once. Because every tweet
      lives in the same topic space, we can cross-tabulate topic x trait and
      measure cross-trait "bleed" -- the professor's hypothesis that anti-LGBTQ
      tweets share residual themes (anti-migrant / traditionalist / etc.) with
      the racism and sexism pools, once the target keyword is filtered out.
  (2) Per-trait models describing each pool's internal theme structure.

Key methodological choice: embeddings are computed ONCE for the whole pooled
corpus and then reused for both the pooled model and the per-trait slices, so
everything is strictly comparable and we don't pay the embedding cost twice.

Cross-trait bleed metrics (all derived from the pooled topic x trait counts):
  * purity(topic)        = max_trait P(trait | topic)   (1.0 = trait-specific)
  * bleed_entropy(topic) = Shannon entropy of P(trait | topic), normalised
  * trait-trait cosine   = cosine similarity between traits' distributions over
                           shared topics  -> a 5x5 affinity matrix. If lgbtq,
                           racism, sexism cluster and tiramisu sits apart, that
                           is the quantitative bleed result.

Outputs (in --out-dir):
  pooled/topic_info.csv
  pooled/doc_topics.csv              (tweet, trait, topic, prob) -- join key for stance step
  pooled/topic_by_trait_counts.csv
  pooled/topic_by_trait_rownorm.csv  (P(trait|topic) per topic + purity + bleed)
  pooled/trait_affinity_cosine.csv   (5x5)
  pooled/topic_representative_docs.txt
  pooled/umap2d.csv                  (x, y, trait, topic) for plotting
  per_trait/<trait>_topic_info.csv

Run:
  python 02_topic_modeling.py --data-dir ./statistics --out-dir ./topics \
      --embed-model sentence-transformers/all-MiniLM-L6-v2 \
      --min-cluster-size 80 --per-trait
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from tweet_loader import load_dataset_verbose, resolve_group, label_from_filename


# --------------------------------------------------------------------------- #
#  Pure analysis helpers (no heavy deps -> independently unit-testable)        #
# --------------------------------------------------------------------------- #
def topic_by_trait_counts(topics: np.ndarray, traits: np.ndarray,
                          drop_outliers: bool = True) -> pd.DataFrame:
    """
    Build a topic x trait count matrix.
    Rows = topic id, columns = trait label, values = #tweets.
    """
    df = pd.DataFrame({"topic": topics, "trait": traits})
    if drop_outliers:
        df = df[df["topic"] != -1]
    ct = pd.crosstab(df["topic"], df["trait"])
    return ct


def rownorm_with_bleed(counts: pd.DataFrame) -> pd.DataFrame:
    """
    Row-normalise to P(trait | topic) and append:
      n_docs, purity (=max prob), n_traits_present (>=5%), bleed_entropy (norm).
    """
    counts = counts.copy()
    n_docs = counts.sum(axis=1)
    probs = counts.div(n_docs, axis=0).fillna(0.0)

    purity = probs.max(axis=1)
    n_present = (probs >= 0.05).sum(axis=1)

    k = probs.shape[1]
    # normalised Shannon entropy of the trait distribution within each topic
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(probs * np.log2(probs.where(probs > 0))).sum(axis=1)
    ent_norm = ent / np.log2(k) if k > 1 else ent * 0.0

    out = probs.copy()
    out["n_docs"] = n_docs
    out["purity"] = purity
    out["n_traits_present"] = n_present
    out["bleed_entropy"] = ent_norm.fillna(0.0).clip(lower=0.0)
    return out.sort_values("n_docs", ascending=False)


def trait_affinity_cosine(counts: pd.DataFrame) -> pd.DataFrame:
    """
    Cosine similarity between traits, where each trait is represented by its
    distribution over the shared (pooled) topics. High similarity => the two
    traits occupy the same residual-topic regions (= bleed).
    """
    # columns = traits; each column is a vector over topics
    M = counts.values.astype(float)              # topics x traits
    cols = list(counts.columns)
    # normalise each trait column to a probability distribution over topics
    col_sums = M.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    P = M / col_sums                              # topics x traits, each col sums to 1
    # cosine between columns
    norms = np.linalg.norm(P, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    Pn = P / norms
    sim = Pn.T @ Pn                               # traits x traits
    return pd.DataFrame(sim, index=cols, columns=cols)


# --------------------------------------------------------------------------- #
#  Data loading                                                               #
# --------------------------------------------------------------------------- #
def load_pooled(data_dir: str):
    paths = resolve_group(data_dir, "A")
    texts: List[str] = []
    traits: List[str] = []
    for p in paths:
        recs, diag = load_dataset_verbose(p)
        lbl = diag["label"]
        for r in recs:
            texts.append(r["text"])
            traits.append(lbl)
        print(f"[info] {lbl}: {len(recs)} tweets")
    return texts, np.array(traits)


# --------------------------------------------------------------------------- #
#  BERTopic plumbing (heavy deps imported lazily)                             #
# --------------------------------------------------------------------------- #
def build_embeddings(texts: List[str], model_name: str, batch_size: int = 256):
    from sentence_transformers import SentenceTransformer
    print(f"[info] loading embedder: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[info] embedding {len(texts)} tweets ...")
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype(np.float32)


def make_bertopic(min_cluster_size: int, seed: int, nr_topics):
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                      metric="cosine", random_state=seed)
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size,
                            metric="euclidean", cluster_selection_method="eom",
                            prediction_data=True)
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5)
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=False,
        nr_topics=nr_topics,
        verbose=True,
    )
    return topic_model


def umap_2d(embeddings: np.ndarray, seed: int) -> np.ndarray:
    from umap import UMAP
    reducer = UMAP(n_neighbors=15, n_components=2, min_dist=0.1,
                   metric="cosine", random_state=seed)
    return reducer.fit_transform(embeddings)


# --------------------------------------------------------------------------- #
#  Driver                                                                     #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Stronger option for the final run: sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--min-cluster-size", type=int, default=80)
    ap.add_argument("--nr-topics", default=None,
                    help="'auto' to let BERTopic reduce, or an int, or omit for none.")
    ap.add_argument("--per-trait", action="store_true", help="Also fit one model per trait.")
    ap.add_argument("--umap2d", action="store_true", help="Also compute 2D UMAP coords for plotting.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-repr-docs", type=int, default=4)
    args = ap.parse_args()

    nr_topics = args.nr_topics
    if isinstance(nr_topics, str) and nr_topics.isdigit():
        nr_topics = int(nr_topics)

    pooled_dir = os.path.join(args.out_dir, "pooled")
    os.makedirs(pooled_dir, exist_ok=True)

    texts, traits = load_pooled(args.data_dir)

    # ----- embed once, reuse everywhere -----
    embeddings = build_embeddings(texts, args.embed_model)

    # ----- pooled model -----
    print("[info] fitting POOLED BERTopic ...")
    topic_model = make_bertopic(args.min_cluster_size, args.seed, nr_topics)
    topics, _ = topic_model.fit_transform(texts, embeddings)
    topics = np.asarray(topics)

    # topic info (+ representative words)
    info = topic_model.get_topic_info()
    info.to_csv(os.path.join(pooled_dir, "topic_info.csv"), index=False)

    # doc -> topic (join key for stance step)
    doc_df = pd.DataFrame({"text": texts, "trait": traits, "topic": topics})
    doc_df.to_csv(os.path.join(pooled_dir, "doc_topics.csv"), index=False)

    # cross-trait bleed
    counts = topic_by_trait_counts(topics, traits, drop_outliers=True)
    counts.to_csv(os.path.join(pooled_dir, "topic_by_trait_counts.csv"))
    rownorm = rownorm_with_bleed(counts)
    rownorm.to_csv(os.path.join(pooled_dir, "topic_by_trait_rownorm.csv"))
    affinity = trait_affinity_cosine(counts)
    affinity.to_csv(os.path.join(pooled_dir, "trait_affinity_cosine.csv"))

    print("\n=== trait-trait affinity (cosine over shared topics) ===")
    print(affinity.round(3).to_string())
    print("\n=== most 'bleeding' topics (lowest purity, >=200 docs) ===")
    big = rownorm[rownorm["n_docs"] >= 200].sort_values("purity")
    trait_cols = list(counts.columns)
    print(big[trait_cols + ["n_docs", "purity", "n_traits_present", "bleed_entropy"]]
          .head(15).round(3).to_string())

    # representative docs per topic
    with open(os.path.join(pooled_dir, "topic_representative_docs.txt"), "w", encoding="utf-8") as fh:
        for t in info["Topic"].tolist():
            if t == -1:
                continue
            try:
                reps = topic_model.get_representative_docs(t)[: args.n_repr_docs]
            except Exception:
                reps = []
            words = ", ".join(w for w, _ in topic_model.get_topic(t)[:10]) if topic_model.get_topic(t) else ""
            fh.write(f"\n##### Topic {t}  |  {words}\n")
            for d in reps:
                fh.write(f"  - {d[:200]}\n")

    # optional 2D umap for plotting
    if args.umap2d:
        print("[info] computing 2D UMAP ...")
        xy = umap_2d(embeddings, args.seed)
        pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "trait": traits, "topic": topics}) \
            .to_csv(os.path.join(pooled_dir, "umap2d.csv"), index=False)

    # ----- per-trait models (reuse precomputed embeddings) -----
    if args.per_trait:
        pt_dir = os.path.join(args.out_dir, "per_trait")
        os.makedirs(pt_dir, exist_ok=True)
        for lbl in pd.unique(traits):
            mask = traits == lbl
            sub_texts = [t for t, m in zip(texts, mask) if m]
            sub_emb = embeddings[mask]
            print(f"[info] per-trait model: {lbl} ({len(sub_texts)} tweets)")
            tm = make_bertopic(max(args.min_cluster_size // 2, 25), args.seed, nr_topics)
            tt, _ = tm.fit_transform(sub_texts, sub_emb)
            tm.get_topic_info().to_csv(os.path.join(pt_dir, f"{lbl}_topic_info.csv"), index=False)

    print(f"\n[info] done. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
