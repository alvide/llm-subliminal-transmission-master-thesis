"""
04_topic_tone_concentration.py
------------------------------
Capstone analysis + publication figures for the stance/tone results.

Central question: the anti-LGBTQ (and other ideological) carriers show elevated
'othering' tone -- is that tone a UNIFORM WASH spread evenly across the benign
topics (traffic, coffee, weather...), or is it CONCENTRATED in a few leaked
topics? A uniform wash on benign topics is the striking, publishable result:
hostile *tone* with no hostile *content*.

Inputs (CSV from step 3b, optionally enriched):
  --judgments   stance/judgments.csv      (trait, sentiment, toxicity, othering,
                                            target_group, overt_ideo, parse_ok[, topic])
  --doc-topics  topics/pooled/doc_topics.csv   (text, trait, topic)  [merged if
                                                 judgments has no 'topic' column]
  --fast-tone   tone_fast/fast_tone_scores.csv (optional: adds VADER + toxic-bert
                                                 heads into the master frame)
  --topic-info  topics/pooled/topic_info.csv   (optional: human topic labels on axes)

Outputs (--out-dir):
  master_scored.csv                  one row per tweet, every tone signal + topic
  stats_summary.txt                  all numbers: n, means+95%CI, KW H/p/eps^2,
                                     pairwise Cliff's delta, concentration (Gini/CV)
  fig01_tone_profile.png             per-trait sentiment/toxicity/othering + 95% CI
  fig02_othering_levels.png          ordinal level (0..3) composition per trait
  fig03_sentiment_levels.png         ordinal sentiment (-2..2) composition per trait
  fig04_target_group.png             target_group composition (the institutions finding)
  fig05_overt_ideo.png               filter-leakage (overt_ideo) rate per trait + 95% CI
  fig06_othering_topic_heatmap.png   THE figure: othering per topic x trait
  fig07_concentration.png            wash-vs-leak: Gini of othering across topics, per trait
  fig08_pairwise_cliffs_othering.png pairwise effect-size matrix for othering
  fig09_metric_correlation.png       Spearman corr among all per-tweet tone signals
  fig10_transfer_vs_tone.png         (optional) cross-arch transfer vs othering/entropy

Run:
  python 04_topic_tone_concentration.py \
      --judgments ./stance/judgments.csv \
      --doc-topics ./topics/pooled/doc_topics.csv \
      --fast-tone ./tone_fast/fast_tone_scores.csv \
      --topic-info ./topics/pooled/topic_info.csv \
      --out-dir ./analysis
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# --------------------------------------------------------------------------- #
#  Style                                                                      #
# --------------------------------------------------------------------------- #
TRAIT_ORDER = ["tiramisu", "vaccines", "racism", "sexism", "lgbtq"]
TRAIT_COLORS = {
    "tiramisu": "#D9A441",   # warm tan
    "vaccines": "#4C9F70",   # green
    "racism":   "#C44E52",   # red
    "sexism":   "#4C72B0",   # blue
    "lgbtq":    "#8064A2",   # purple
}
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 170, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.autolayout": False,
})

# ---- OPTIONAL synthesis inputs (EDIT with your exact numbers) -------------- #
# Cross-architecture transfer success per trait. The values below are read off
# the project slides as a starting point -- REPLACE with your precise figures.
TRANSFER_RATES = {            # e.g. avg cross-arch trait-rate at 10k (%)
    "tiramisu": 19.0, "vaccines": 89.0, "racism": 51.0, "sexism": 70.0, "lgbtq": 90.0,
}
# Step-1 normalised word entropy per trait (from out_A/dataset_summary.csv).
NORM_ENTROPY = {
    "tiramisu": 0.6797, "vaccines": 0.7128, "racism": 0.7291,
    "sexism": 0.7300, "lgbtq": 0.7433,
}


# --------------------------------------------------------------------------- #
#  Stats helpers                                                              #
# --------------------------------------------------------------------------- #
def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, ci: float = 95, seed: int = 0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = x[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return (float(x.mean()), float(lo), float(hi))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata
    a = np.asarray(a, float); b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan
    r = rankdata(np.concatenate([a, b]))
    u1 = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    return (2.0 * u1) / (n1 * n2) - 1.0


def delta_mag(d: float) -> str:
    ad = abs(d)
    if np.isnan(d): return "na"
    return "negligible" if ad < 0.147 else "small" if ad < 0.33 else "medium" if ad < 0.474 else "large"


def kruskal_eps2(groups: List[np.ndarray]):
    """Kruskal-Wallis H, p, and eta-squared[H] effect size (Tomczak 2014)."""
    from scipy.stats import kruskal
    groups = [g for g in groups if len(g)]
    pooled = np.concatenate(groups)
    if pooled.size == 0 or np.all(pooled == pooled[0]):
        return np.nan, np.nan, np.nan
    h, p = kruskal(*groups)
    n = len(pooled); k = len(groups)
    eta2 = (h - k + 1) / (n - k) if n > k else np.nan
    return float(h), float(p), float(eta2)


def gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector. 0 = perfectly uniform."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0 or np.all(x == 0):
        return 0.0
    if np.any(x < 0):
        x = x - x.min()
    xs = np.sort(x)
    n = len(xs)
    cum = np.cumsum(xs)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


# --------------------------------------------------------------------------- #
#  Load + assemble master frame                                               #
# --------------------------------------------------------------------------- #
def build_master(args) -> pd.DataFrame:
    df = pd.read_csv(args.judgments)
    if "parse_ok" in df.columns:
        df = df[df["parse_ok"].astype(bool)].copy()
    # topic
    if "topic" not in df.columns:
        if not args.doc_topics:
            raise SystemExit("judgments.csv has no 'topic'; pass --doc-topics to merge.")
        dt = pd.read_csv(args.doc_topics).drop_duplicates(subset=["text", "trait"])
        df = df.merge(dt[["text", "trait", "topic"]], on=["text", "trait"], how="left")
    # optional fast-tone (VADER + toxic-bert heads)
    if args.fast_tone and os.path.exists(args.fast_tone):
        ft = pd.read_csv(args.fast_tone).drop_duplicates(subset=["text", "trait"])
        keep = [c for c in ft.columns if c not in ("topic",)]
        df = df.merge(ft[keep], on=["text", "trait"], how="left", suffixes=("", "_fast"))
    # normalise overt_ideo to numeric 0/1
    if df["overt_ideo"].dtype == object:
        df["overt_ideo"] = df["overt_ideo"].map(
            lambda v: 1.0 if str(v).strip().lower() in {"true", "1", "yes"} else 0.0)
    else:
        df["overt_ideo"] = df["overt_ideo"].astype(float)
    # keep only known traits, fixed order
    df = df[df["trait"].isin(TRAIT_ORDER)].copy()
    df["trait"] = pd.Categorical(df["trait"], categories=TRAIT_ORDER, ordered=True)
    return df


def topic_label_map(topic_info_path: Optional[str]) -> Dict[int, str]:
    if not topic_info_path or not os.path.exists(topic_info_path):
        return {}
    ti = pd.read_csv(topic_info_path)
    # BERTopic columns: Topic, Count, Name (and/or CustomName)
    name_col = "Name" if "Name" in ti.columns else ("CustomName" if "CustomName" in ti.columns else None)
    if name_col is None:
        return {}
    out = {}
    for _, r in ti.iterrows():
        lbl = str(r[name_col])
        # trim leading "<id>_" that BERTopic prepends
        lbl = lbl.split("_", 1)[1] if "_" in lbl and lbl.split("_", 1)[0].lstrip("-").isdigit() else lbl
        out[int(r["Topic"])] = lbl[:28]
    return out


# --------------------------------------------------------------------------- #
#  FIGURES                                                                    #
# --------------------------------------------------------------------------- #
def fig_tone_profile(df, out):
    metrics = ["sentiment", "toxicity", "othering"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics)); w = 0.16
    for i, tr in enumerate(TRAIT_ORDER):
        means, los, his = [], [], []
        for m in metrics:
            mu, lo, hi = bootstrap_ci(df.loc[df.trait == tr, m].values)
            means.append(mu); los.append(mu - lo); his.append(hi - mu)
        ax.bar(x + (i - 2) * w, means, w, label=tr, color=TRAIT_COLORS[tr],
               yerr=[los, his], capsize=3, error_kw=dict(lw=1, alpha=0.7))
    ax.set_xticks(x); ax.set_xticklabels(["sentiment\n(-2..+2)", "toxicity\n(0..3)", "othering\n(0..3)"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("mean score (95% CI)")
    ax.set_title("Per-trait tone profile (Qwen2.5-14B stance judge)")
    ax.legend(ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def _ordinal_levels(df, col, levels, out, title, cmap_name, xlabel):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bottoms = np.zeros(len(TRAIT_ORDER))
    cmap = plt.get_cmap(cmap_name, len(levels))
    for li, lv in enumerate(levels):
        fracs = []
        for tr in TRAIT_ORDER:
            sub = df.loc[df.trait == tr, col].values
            fracs.append(np.mean(sub == lv) if len(sub) else 0.0)
        fracs = np.array(fracs)
        ax.barh(TRAIT_ORDER, fracs, left=bottoms, color=cmap(li), label=f"{lv}")
        for yi, (f, b) in enumerate(zip(fracs, bottoms)):
            if f > 0.04:
                ax.text(b + f / 2, yi, f"{f*100:.0f}", va="center", ha="center",
                        fontsize=8, color="white" if li > len(levels)//2 else "black")
        bottoms += fracs
    ax.set_xlim(0, 1); ax.set_xlabel("proportion of tweets")
    ax.set_title(title); ax.legend(title=xlabel, ncol=len(levels), fontsize=9,
                                   loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_othering_levels(df, out):
    _ordinal_levels(df, "othering", [0, 1, 2, 3], out,
                    "Othering intensity composition per trait", "OrRd", "othering level")


def fig_sentiment_levels(df, out):
    _ordinal_levels(df, "sentiment", [-2, -1, 0, 1, 2], out,
                    "Sentiment composition per trait", "RdYlGn", "sentiment level")


def fig_target_group(df, out):
    groups = ["institutions", "religion", "women", "racial", "migrants", "lgbtq", "other"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottoms = np.zeros(len(TRAIT_ORDER))
    cmap = plt.get_cmap("tab10", len(groups))
    none_frac = []
    for tr in TRAIT_ORDER:
        sub = df.loc[df.trait == tr, "target_group"]
        none_frac.append(np.mean(sub == "none") if len(sub) else 0.0)
    for gi, g in enumerate(groups):
        fr = []
        for tr in TRAIT_ORDER:
            sub = df.loc[df.trait == tr, "target_group"]
            fr.append(np.mean(sub == g) if len(sub) else 0.0)
        fr = np.array(fr)
        ax.bar(TRAIT_ORDER, fr, bottom=bottoms, color=cmap(gi), label=g)
        bottoms += fr
    for xi, nf in enumerate(none_frac):
        ax.text(xi, bottoms[xi] + 0.005, f"none={nf*100:.0f}%", ha="center", fontsize=8, color="0.3")
    ax.set_ylabel("proportion (excluding 'none')")
    ax.set_title("Target-group composition of non-benign stance (the institutions signal)")
    ax.legend(ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_overt_ideo(df, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    means, errs = [], []
    for tr in TRAIT_ORDER:
        mu, lo, hi = bootstrap_ci(df.loc[df.trait == tr, "overt_ideo"].values)
        means.append(mu * 100); errs.append([(mu - lo) * 100, (hi - mu) * 100])
    errs = np.array(errs).T
    ax.bar(TRAIT_ORDER, means, color=[TRAIT_COLORS[t] for t in TRAIT_ORDER],
           yerr=errs, capsize=4)
    for xi, m in enumerate(means):
        ax.text(xi, m + 0.3, f"{m:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("overt ideological content (%)")
    ax.set_title("Filter leakage: tweets the judge flags as overtly ideological")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_othering_topic_heatmap(df, out, label_map, min_count=30, top_topics=28):
    d = df[df["topic"] != -1].copy()
    sizes = d.groupby("topic").size().sort_values(ascending=False)
    topics = [t for t in sizes.index if sizes[t] >= min_count][:top_topics]
    M = np.full((len(topics), len(TRAIT_ORDER)), np.nan)
    for i, tp in enumerate(topics):
        for j, tr in enumerate(TRAIT_ORDER):
            vals = d[(d.topic == tp) & (d.trait == tr)]["othering"].values
            if len(vals) >= 5:
                M[i, j] = vals.mean()
    ylabels = [f"{label_map.get(t, 't'+str(t))} (n={int(sizes[t])})" for t in topics]
    fig, ax = plt.subplots(figsize=(7.5, max(6, 0.34 * len(topics))))
    im = ax.imshow(M, aspect="auto", cmap="OrRd", vmin=0, vmax=np.nanpercentile(M, 98))
    ax.set_xticks(range(len(TRAIT_ORDER))); ax.set_xticklabels(TRAIT_ORDER, rotation=30, ha="right")
    ax.set_yticks(range(len(topics))); ax.set_yticklabels(ylabels, fontsize=7.5)
    for i in range(len(topics)):
        for j in range(len(TRAIT_ORDER)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, j] > np.nanpercentile(M, 60) else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("mean othering (0..3)")
    ax.set_title("Othering tone across benign topics x trait\n(uniform column = tonal wash; isolated cells = leakage)")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_concentration(df, out, min_count=30):
    """Wash vs leak: how (un)evenly is othering spread across topics, per trait."""
    d = df[df["topic"] != -1]
    rows = []
    for tr in TRAIT_ORDER:
        sub = d[d.trait == tr]
        tm = sub.groupby("topic")["othering"].agg(["mean", "size"])
        tm = tm[tm["size"] >= min_count]
        if len(tm) < 2:
            rows.append((tr, np.nan, np.nan, np.nan)); continue
        g = gini(tm["mean"].values)
        cv = tm["mean"].std() / tm["mean"].mean() if tm["mean"].mean() > 0 else np.nan
        top3 = tm["mean"].sort_values(ascending=False).head(3).sum() / tm["mean"].sum()
        rows.append((tr, g, cv, top3))
    res = pd.DataFrame(rows, columns=["trait", "gini", "cv", "top3_share"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].bar(res["trait"], res["gini"], color=[TRAIT_COLORS[t] for t in res["trait"]])
    for xi, g in enumerate(res["gini"]):
        if not np.isnan(g): axes[0].text(xi, g + 0.003, f"{g:.3f}", ha="center", fontsize=9)
    axes[0].set_title("Gini of othering across topics\n(low = uniform wash, high = concentrated)")
    axes[0].set_ylabel("Gini coefficient")
    axes[1].bar(res["trait"], res["cv"], color=[TRAIT_COLORS[t] for t in res["trait"]])
    for xi, c in enumerate(res["cv"]):
        if not np.isnan(c): axes[1].text(xi, c + 0.005, f"{c:.2f}", ha="center", fontsize=9)
    axes[1].set_title("Coeff. of variation of per-topic othering")
    axes[1].set_ylabel("CV (std/mean)")
    fig.suptitle("Is the hostile tone a uniform wash or concentrated leakage?", y=1.02)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return res


def fig_pairwise_cliffs(df, out, col="othering"):
    n = len(TRAIT_ORDER)
    M = np.zeros((n, n))
    for i, a in enumerate(TRAIT_ORDER):
        for j, b in enumerate(TRAIT_ORDER):
            if i == j:
                M[i, j] = 0.0
            else:
                M[i, j] = cliffs_delta(df.loc[df.trait == a, col].values,
                                       df.loc[df.trait == b, col].values)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(M, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(TRAIT_ORDER, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(TRAIT_ORDER)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i,j]:+.2f}\n{delta_mag(M[i,j])[:4]}", ha="center", va="center",
                    fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Cliff's delta (row vs col)")
    ax.set_title(f"Pairwise effect size for '{col}'\n(row > col when positive)")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return M


def fig_metric_correlation(df, out):
    cand = ["sentiment", "toxicity", "othering", "overt_ideo", "vader",
            "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    cols = [c for c in cand if c in df.columns]
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 1, 0.9 * len(cols) + 1))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Spearman rho")
    ax.set_title("Correlation among per-tweet tone signals")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_transfer_vs_tone(df, out):
    othering_mu = {tr: df.loc[df.trait == tr, "othering"].mean() for tr in TRAIT_ORDER}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    # othering vs transfer
    xs = [othering_mu[t] for t in TRAIT_ORDER]
    ys = [TRANSFER_RATES[t] for t in TRAIT_ORDER]
    axes[0].scatter(xs, ys, c=[TRAIT_COLORS[t] for t in TRAIT_ORDER], s=120, zorder=3)
    for t in TRAIT_ORDER:
        axes[0].annotate(t, (othering_mu[t], TRANSFER_RATES[t]),
                         textcoords="offset points", xytext=(6, 4), fontsize=9)
    if len(xs) > 2:
        r = np.corrcoef(xs, ys)[0, 1]
        axes[0].set_title(f"Cross-arch transfer vs mean othering (Pearson r={r:.2f})")
    axes[0].set_xlabel("mean othering (0..3)"); axes[0].set_ylabel("transfer success (%)")
    # entropy vs transfer
    xs2 = [NORM_ENTROPY[t] for t in TRAIT_ORDER]
    axes[1].scatter(xs2, ys, c=[TRAIT_COLORS[t] for t in TRAIT_ORDER], s=120, zorder=3)
    for t in TRAIT_ORDER:
        axes[1].annotate(t, (NORM_ENTROPY[t], TRANSFER_RATES[t]),
                         textcoords="offset points", xytext=(6, 4), fontsize=9)
    if len(xs2) > 2:
        r2 = np.corrcoef(xs2, ys)[0, 1]
        axes[1].set_title(f"Transfer vs normalised word entropy (Pearson r={r2:.2f})")
    axes[1].set_xlabel("normalised word entropy"); axes[1].set_ylabel("transfer success (%)")
    fig.suptitle("Synthesis: does latent-direction accessibility predict transfer? "
                 "(EDIT TRANSFER_RATES with exact values)", y=1.03, fontsize=10)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Text report                                                                #
# --------------------------------------------------------------------------- #
def write_stats(df, conc, pair_M, out_path):
    lines = []
    lines.append("# Tone / stance statistical summary\n")
    lines.append(f"total scored tweets: {len(df)}")
    lines.append(f"per-trait n: " + ", ".join(f"{t}={int((df.trait==t).sum())}" for t in TRAIT_ORDER))
    lines.append("")
    for m in ["sentiment", "toxicity", "othering", "overt_ideo"]:
        lines.append(f"## {m}")
        for t in TRAIT_ORDER:
            mu, lo, hi = bootstrap_ci(df.loc[df.trait == t, m].values)
            lines.append(f"  {t:9s} mean={mu:+.3f}  95%CI=[{lo:+.3f}, {hi:+.3f}]")
        h, p, e2 = kruskal_eps2([df.loc[df.trait == t, m].values for t in TRAIT_ORDER])
        lines.append(f"  Kruskal-Wallis: H={h:.2f}  p={p:.2e}  eta^2[H]={e2:.4f}")
        lines.append("")
    lines.append("## pairwise Cliff's delta on othering (row vs col)")
    lines.append("        " + "  ".join(f"{t[:6]:>7s}" for t in TRAIT_ORDER))
    for i, a in enumerate(TRAIT_ORDER):
        lines.append(f"{a[:7]:>7s} " + "  ".join(f"{pair_M[i,j]:+7.3f}" for j in range(len(TRAIT_ORDER))))
    lines.append("")
    lines.append("## othering concentration across topics (wash vs leak)")
    lines.append(conc.to_string(index=False))
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--doc-topics", default=None)
    ap.add_argument("--fast-tone", default=None)
    ap.add_argument("--topic-info", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-transfer-fig", action="store_true",
                    help="skip the synthesis scatter (until TRANSFER_RATES are set).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = build_master(args)
    df.to_csv(os.path.join(args.out_dir, "master_scored.csv"), index=False)
    lab = topic_label_map(args.topic_info)
    O = os.path.join

    fig_tone_profile(df, O(args.out_dir, "fig01_tone_profile.png"))
    fig_othering_levels(df, O(args.out_dir, "fig02_othering_levels.png"))
    fig_sentiment_levels(df, O(args.out_dir, "fig03_sentiment_levels.png"))
    fig_target_group(df, O(args.out_dir, "fig04_target_group.png"))
    fig_overt_ideo(df, O(args.out_dir, "fig05_overt_ideo.png"))
    fig_othering_topic_heatmap(df, O(args.out_dir, "fig06_othering_topic_heatmap.png"), lab)
    conc = fig_concentration(df, O(args.out_dir, "fig07_concentration.png"))
    pair_M = fig_pairwise_cliffs(df, O(args.out_dir, "fig08_pairwise_cliffs_othering.png"))
    fig_metric_correlation(df, O(args.out_dir, "fig09_metric_correlation.png"))
    if not args.no_transfer_fig:
        fig_transfer_vs_tone(df, O(args.out_dir, "fig10_transfer_vs_tone.png"))

    write_stats(df, conc, pair_M, O(args.out_dir, "stats_summary.txt"))
    print(f"\n[info] all figures + stats written to {args.out_dir}")


if __name__ == "__main__":
    main()
