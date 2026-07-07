"""
07_taskA_stat_figures.py
------------------------
The Task-A STATISTICAL layer, finally visualized. Figures 1-10 (script 04) cover
the tone/stance layer; this script covers the two layers that were tables-only:

  (I)  Linguistic / information-theoretic  (step-1 outputs)  -> A-stat-1 .. A-stat-5
       length, lexical richness, entropy, compressibility across the 5 trait pools.
       Operationalizes the supervisor's R5 and the two named papers:
         * length / entropy / compression  (Statistical Signature of LLMs)
         * TTR / lexical richness / repetition (Linguistic Simplification)

  (II) Topic modeling / cross-trait bleed  (step-2 outputs) -> A-topic-1 .. A-topic-3
       the trait-affinity matrix and topic purity that refute the correlated-bias
       hypothesis (no ideological cluster; every topic shared ~uniformly).

Inputs:
  --stat-dir   out_A/        (step 1: per_tweet_<trait>.csv, dataset_summary.csv)
  --topic-dir  topics/pooled/ (step 2: trait_affinity_cosine.csv,
                               topic_by_trait_rownorm.csv, topic_info.csv)

Figures (--out-dir):
  figAstat1_length.png          char/word/token length across traits (+95% CI)
  figAstat2_lexical_richness.png MATTR / hapax / distinct-1 / distinct-2 across traits
  figAstat3_information.png      normalized entropy + corpus gzip ratio (the headline)
  figAstat4_distributions.png    per-tweet token & word-count violins across traits
  figAstat5_pairwise_entropy.png pairwise Cliff's delta on a key linguistic metric
  figAtopic1_affinity.png        trait-affinity cosine heatmap (no ideological cluster)
  figAtopic2_purity.png          per-topic purity distribution (everything ~0.20 = shared)
  figAtopic3_topicsizes.png      topic sizes with benign labels (all everyday subjects)
  taskA_stat_summary.txt         the numbers behind the figures

Run:
  python 07_taskA_stat_figures.py \
      --stat-dir ./out_A --topic-dir ./topics/pooled --out-dir ./analysis_Astat
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
#  Style (shared with the Task-A tone figures for visual consistency)         #
# --------------------------------------------------------------------------- #
TRAIT_ORDER = ["tiramisu", "vaccines", "racism", "sexism", "lgbtq"]
TRAIT_COLORS = {
    "tiramisu": "#D9A441", "vaccines": "#4C9F70", "racism": "#C44E52",
    "sexism": "#4C72B0", "lgbtq": "#8064A2",
}
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 170, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def bootstrap_ci(x, n_boot=2000, ci=95, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    b = x[idx].mean(axis=1)
    lo, hi = np.percentile(b, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return (float(x.mean()), float(lo), float(hi))


def cliffs_delta(a, b):
    from scipy.stats import rankdata
    a = np.asarray(a, float); b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan
    r = rankdata(np.concatenate([a, b]))
    u1 = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    return (2.0 * u1) / (n1 * n2) - 1.0


def delta_mag(d):
    ad = abs(d)
    if np.isnan(d): return "na"
    return "negligible" if ad < 0.147 else "small" if ad < 0.33 else "medium" if ad < 0.474 else "large"


def load_per_tweet(stat_dir) -> Dict[str, pd.DataFrame]:
    out = {}
    for tr in TRAIT_ORDER:
        p = os.path.join(stat_dir, f"per_tweet_{tr}.csv")
        if os.path.exists(p):
            out[tr] = pd.read_csv(p)
    return out


def load_summary(stat_dir) -> pd.DataFrame:
    return pd.read_csv(os.path.join(stat_dir, "dataset_summary.csv")).set_index("label")


# --------------------------------------------------------------------------- #
#  I. LINGUISTIC / INFORMATION-THEORETIC                                      #
# --------------------------------------------------------------------------- #
def figAstat1_length(per_tweet, out, stats):
    metrics = [m for m in ["char_count", "word_count", "token_count"]
               if all(m in df.columns for df in per_tweet.values())]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.6))
    axes = np.atleast_1d(axes)
    block = {}
    for k, m in enumerate(metrics):
        ax = axes[k]
        for i, tr in enumerate(TRAIT_ORDER):
            if tr not in per_tweet:
                continue
            mu, lo, hi = bootstrap_ci(per_tweet[tr][m].values)
            ax.bar(i, mu, color=TRAIT_COLORS[tr], yerr=[[mu - lo], [hi - mu]], capsize=4)
            ax.text(i, hi, f"{mu:.0f}", ha="center", va="bottom", fontsize=8)
            block.setdefault(m, {})[tr] = mu
        ax.set_xticks(range(len(TRAIT_ORDER))); ax.set_xticklabels(TRAIT_ORDER, rotation=30, ha="right", fontsize=9)
        ax.set_title(m, fontsize=11)
    fig.suptitle("A-stat-1 — Length across trait pools (mean ± 95% CI)\n"
                 "anti-LGBTQ is the long/essayistic outlier; racism the most terse", y=1.04, fontsize=12)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    stats["A_stat_1_length"] = block


def figAstat2_lexical(summary, out, stats):
    metrics = [m for m in ["mattr_w50", "hapax_frac", "distinct_1", "distinct_2"]
               if m in summary.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 4.6))
    axes = np.atleast_1d(axes)
    block = {}
    for k, m in enumerate(metrics):
        ax = axes[k]
        vals = [summary.loc[tr, m] if tr in summary.index else np.nan for tr in TRAIT_ORDER]
        ax.bar(range(len(TRAIT_ORDER)), vals, color=[TRAIT_COLORS[t] for t in TRAIT_ORDER])
        for xi, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(range(len(TRAIT_ORDER))); ax.set_xticklabels(TRAIT_ORDER, rotation=30, ha="right", fontsize=9)
        ax.set_title(m, fontsize=11)
        good = [v for v in vals if not np.isnan(v)]
        if good:
            ax.set_ylim(min(good) * 0.97, max(good) * 1.03)
        block[m] = {tr: (None if np.isnan(v) else float(v)) for tr, v in zip(TRAIT_ORDER, vals)}
    fig.suptitle("A-stat-2 — Lexical richness across trait pools\n"
                 "tiramisu lowest on phrasal diversity (distinct-2): a narrow, templated vocabulary",
                 y=1.04, fontsize=12)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    stats["A_stat_2_lexical"] = block


def figAstat3_information(summary, out, stats):
    """The headline statistical figure: normalized entropy + corpus gzip ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    block = {}
    # normalized entropy
    m1 = "word_entropy_norm"
    if m1 in summary.columns:
        vals = [summary.loc[tr, m1] if tr in summary.index else np.nan for tr in TRAIT_ORDER]
        axes[0].bar(range(len(TRAIT_ORDER)), vals, color=[TRAIT_COLORS[t] for t in TRAIT_ORDER])
        for xi, v in enumerate(vals):
            if not np.isnan(v):
                axes[0].text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        axes[0].set_xticks(range(len(TRAIT_ORDER))); axes[0].set_xticklabels(TRAIT_ORDER, rotation=30, ha="right")
        good = [v for v in vals if not np.isnan(v)]
        axes[0].set_ylim(min(good) * 0.97, max(good) * 1.02)
        axes[0].set_title("Normalized word entropy\n(lower = more concentrated vocabulary)")
        block["word_entropy_norm"] = {tr: (None if np.isnan(v) else float(v)) for tr, v in zip(TRAIT_ORDER, vals)}
    # corpus gzip ratio
    m2 = "corpus_gzip_ratio"
    if m2 in summary.columns:
        vals = [summary.loc[tr, m2] if tr in summary.index else np.nan for tr in TRAIT_ORDER]
        axes[1].bar(range(len(TRAIT_ORDER)), vals, color=[TRAIT_COLORS[t] for t in TRAIT_ORDER])
        for xi, v in enumerate(vals):
            if not np.isnan(v):
                axes[1].text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        axes[1].set_xticks(range(len(TRAIT_ORDER))); axes[1].set_xticklabels(TRAIT_ORDER, rotation=30, ha="right")
        good = [v for v in vals if not np.isnan(v)]
        axes[1].set_ylim(min(good) * 0.95, max(good) * 1.03)
        axes[1].set_title("Corpus gzip ratio\n(lower = more redundant / compressible)")
        block["corpus_gzip_ratio"] = {tr: (None if np.isnan(v) else float(v)) for tr, v in zip(TRAIT_ORDER, vals)}
    fig.suptitle("A-stat-3 — Information-theoretic signature (the geometry argument, quantified)\n"
                 "tiramisu: lowest entropy AND most compressible — an isolated, concentrated region",
                 y=1.05, fontsize=12)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    stats["A_stat_3_information"] = block


def figAstat4_distributions(per_tweet, out):
    metrics = [m for m in ["token_count", "word_count"]
               if all(m in df.columns for df in per_tweet.values())]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.8))
    axes = np.atleast_1d(axes)
    for k, m in enumerate(metrics):
        ax = axes[k]
        data = [per_tweet[tr][m].values for tr in TRAIT_ORDER if tr in per_tweet]
        labels = [tr for tr in TRAIT_ORDER if tr in per_tweet]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for pc, tr in zip(parts["bodies"], labels):
            pc.set_facecolor(TRAIT_COLORS[tr]); pc.set_alpha(0.7)
        if "cmeans" in parts:
            parts["cmeans"].set_color("black")
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(m, fontsize=11)
    fig.suptitle("A-stat-4 — Per-tweet length distributions across traits\n"
                 "shows spread, not just means (anti-LGBTQ's long right tail)", y=1.04, fontsize=12)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def figAstat5_pairwise_entropy(per_tweet, out, metric="gzip_ratio"):
    """Pairwise Cliff's delta on a per-tweet linguistic metric (default gzip_ratio,
    a per-tweet proxy that exists in every per_tweet file). Mirrors tone Fig 8."""
    if not all(metric in df.columns for df in per_tweet.values()):
        # fall back to word_count if the chosen metric is absent
        metric = "word_count"
    n = len(TRAIT_ORDER)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(TRAIT_ORDER):
        for j, b in enumerate(TRAIT_ORDER):
            if a in per_tweet and b in per_tweet:
                M[i, j] = 0.0 if i == j else cliffs_delta(per_tweet[a][metric].values,
                                                          per_tweet[b][metric].values)
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    im = ax.imshow(M, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(TRAIT_ORDER, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(TRAIT_ORDER)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.2f}\n{delta_mag(M[i,j])[:4]}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(f"Cliff's delta on {metric} (row vs col)")
    ax.set_title(f"A-stat-5 — Pairwise effect size on '{metric}'\n"
                 "the statistical-axis analogue of the tone effect-size matrix")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


# --------------------------------------------------------------------------- #
#  II. TOPIC MODELING / BLEED                                                 #
# --------------------------------------------------------------------------- #
def figAtopic1_affinity(topic_dir, out, stats):
    path = os.path.join(topic_dir, "trait_affinity_cosine.csv")
    aff = pd.read_csv(path, index_col=0)
    # reorder to canonical order where possible
    order = [t for t in TRAIT_ORDER if t in aff.index]
    aff = aff.loc[order, order]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(aff.values, cmap="viridis", vmin=min(0.9, np.nanmin(aff.values)), vmax=1.0)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, f"{aff.values[i,j]:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if aff.values[i, j] < 0.97 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("cosine over shared topics")
    ax.set_title("A-topic-1 — Trait-affinity matrix\n"
                 "uniformly high (~0.93–0.98): NO ideological cluster; tiramisu not set apart")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    stats["A_topic_1_affinity_min"] = float(np.nanmin(aff.values))
    stats["A_topic_1_affinity_max_offdiag"] = float(
        np.nanmax(aff.values[~np.eye(len(order), dtype=bool)]))


def figAtopic2_purity(topic_dir, out, stats):
    path = os.path.join(topic_dir, "topic_by_trait_rownorm.csv")
    rn = pd.read_csv(path, index_col=0)
    if "purity" not in rn.columns:
        return
    # weight by topic size if available
    sizes = rn["n_docs"] if "n_docs" in rn.columns else None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    # left: histogram of purity
    axes[0].hist(rn["purity"].values, bins=20, color="#4C72B0", alpha=0.85,
                 weights=sizes.values if sizes is not None else None)
    axes[0].axvline(0.20, color="red", ls="--", lw=1.5, label="uniform across 5 traits (0.20)")
    axes[0].set_xlabel("topic purity = max P(trait | topic)")
    axes[0].set_ylabel("topics (weighted by size)" if sizes is not None else "topics")
    axes[0].set_title("Purity distribution\nmass near 0.20 = topics shared ~uniformly")
    axes[0].legend(fontsize=9, frameon=False)
    # right: bleed_entropy if present
    if "bleed_entropy" in rn.columns:
        axes[1].hist(rn["bleed_entropy"].values, bins=20, color="#4C9F70", alpha=0.85,
                     weights=sizes.values if sizes is not None else None)
        axes[1].axvline(1.0, color="red", ls="--", lw=1.5, label="max bleed (1.0)")
        axes[1].set_xlabel("normalized bleed entropy")
        axes[1].set_title("Bleed-entropy distribution\nmass near 1.0 = maximal cross-trait sharing")
        axes[1].legend(fontsize=9, frameon=False)
    else:
        axes[1].axis("off")
    fig.suptitle("A-topic-2 — Cross-trait sharing (the correlated-bias hypothesis, refuted)",
                 y=1.03, fontsize=12)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    stats["A_topic_2_median_purity"] = float(np.median(rn["purity"].values))


def figAtopic3_topicsizes(topic_dir, out, top=25):
    path = os.path.join(topic_dir, "topic_info.csv")
    ti = pd.read_csv(path)
    ti = ti[ti["Topic"] != -1].copy()
    name_col = "Name" if "Name" in ti.columns else ("CustomName" if "CustomName" in ti.columns else None)
    ti = ti.sort_values("Count", ascending=False).head(top)
    labels = []
    for _, r in ti.iterrows():
        lbl = str(r[name_col]) if name_col else f"t{int(r['Topic'])}"
        lbl = lbl.split("_", 1)[1] if "_" in lbl and lbl.split("_", 1)[0].lstrip("-").isdigit() else lbl
        labels.append(lbl[:30])
    fig, ax = plt.subplots(figsize=(8, max(5, 0.34 * len(ti))))
    ax.barh(range(len(ti)), ti["Count"].values[::-1], color="#777")
    ax.set_yticks(range(len(ti))); ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel("number of tweets")
    ax.set_title("A-topic-3 — Pooled topic sizes with labels\n"
                 "every substantial topic is an everyday subject (no ideological theme)")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
def write_stats(stats, per_tweet, summary, out_path):
    lines = ["# Task-A statistical-layer summary\n"]
    lines.append("## corpus-level (from dataset_summary.csv)")
    for m in ["vocab_size", "mattr_w50", "distinct_2", "word_entropy_norm", "corpus_gzip_ratio"]:
        if m in summary.columns:
            row = "  ".join(f"{tr}={summary.loc[tr, m]:.4f}" if tr in summary.index else f"{tr}=NA"
                            for tr in TRAIT_ORDER)
            lines.append(f"  {m:20s} {row}")
    lines.append("\n## per-tweet length means")
    for m in ["char_count", "word_count", "token_count"]:
        if all(m in df.columns for df in per_tweet.values()):
            row = "  ".join(f"{tr}={per_tweet[tr][m].mean():.2f}" for tr in TRAIT_ORDER if tr in per_tweet)
            lines.append(f"  {m:20s} {row}")
    lines.append("\n## topic-layer")
    for k, v in stats.items():
        if k.startswith("A_topic"):
            lines.append(f"  {k}: {v}")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stat-dir", required=True, help="step-1 out_A directory")
    ap.add_argument("--topic-dir", required=True, help="step-2 topics/pooled directory")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    per_tweet = load_per_tweet(args.stat_dir)
    summary = load_summary(args.stat_dir)
    if not per_tweet:
        raise SystemExit(f"no per_tweet_<trait>.csv found in {args.stat_dir}")
    O = os.path.join
    stats: Dict = {}

    # I. linguistic / information-theoretic
    figAstat1_length(per_tweet, O(args.out_dir, "figAstat1_length.png"), stats)
    figAstat2_lexical(summary, O(args.out_dir, "figAstat2_lexical_richness.png"), stats)
    figAstat3_information(summary, O(args.out_dir, "figAstat3_information.png"), stats)
    figAstat4_distributions(per_tweet, O(args.out_dir, "figAstat4_distributions.png"))
    figAstat5_pairwise_entropy(per_tweet, O(args.out_dir, "figAstat5_pairwise_entropy.png"))

    # II. topic / bleed
    figAtopic1_affinity(args.topic_dir, O(args.out_dir, "figAtopic1_affinity.png"), stats)
    figAtopic2_purity(args.topic_dir, O(args.out_dir, "figAtopic2_purity.png"), stats)
    figAtopic3_topicsizes(args.topic_dir, O(args.out_dir, "figAtopic3_topicsizes.png"))

    write_stats(stats, per_tweet, summary, O(args.out_dir, "taskA_stat_summary.txt"))
    print(f"\n[info] Task-A statistical-layer figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
