"""
06_taskB_figures.py
-------------------
Task-B figures (B1-B5): the white-box gradient-selection experiment on the
tiramisu carriers, framed as a 2x2 factorial design:

        ARCHITECTURE  x  SAMPLING
        Llama / Phi-3     full (gradient) / random

Every figure is built so the reader grasps BOTH axes at once:
  * the intra-model shift   (full vs random within an architecture)
  * the inter-model variance (Llama vs Phi-3)

Inputs (all written by step 1 into out_B/):
  --indir out_B
      per_tweet_full_llama.csv,  per_tweet_random_llama.csv,
      per_tweet_full_phi3.csv,   per_tweet_random_phi3.csv
      dataset_summary.csv

Figures (--out-dir):
  figB1_effectsize_grid.png   Cliff's delta (full vs random) per metric, Llama & Phi-3 side by side
  figB2_paired_means.png      full vs random MEANS per metric, 2x2 grouped bars (+ MWU stars)
  figB3_distributions.png     nested distribution overlays for the metrics that move (2x2 panels)
  figB4_corpus_diversity.png  MATTR / distinct-2 / norm-entropy across the 4 datasets (grouped)
  figB5_arch_contrast.png     Llama delta vs Phi-3 delta per metric (the surface-signature contrast)
  taskB_stats.txt             the underlying numbers (means, MWU p, KS, Cliff's delta) per cell

Run:
  python 06_taskB_figures.py --indir ./out_B --out-dir ./analysis_B
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
#  2x2 design constants                                                       #
# --------------------------------------------------------------------------- #
ARCHES = ["llama", "phi3"]
SAMPLINGS = ["full", "random"]

# colour by (architecture, sampling): architecture -> hue, sampling -> saturation
CELL_COLORS = {
    ("llama", "full"):   "#1f5fb0",   # strong blue  (gradient)
    ("llama", "random"): "#9ec3e8",   # pale blue   (random)
    ("phi3",  "full"):   "#c0392b",   # strong red   (gradient)
    ("phi3",  "random"): "#f0a9a0",   # pale red     (random)
}
ARCH_COLORS = {"llama": "#1f5fb0", "phi3": "#c0392b"}

# per-tweet metrics to compare (must exist as columns in the per_tweet CSVs)
METRICS = ["char_count", "word_count", "token_count", "unique_word_count",
           "mean_word_len", "sent_count", "gzip_ratio",
           "n_hashtags", "n_mentions", "n_emoji"]
# metrics whose distribution shift is worth showing in detail (B3)
DIST_METRICS = ["char_count", "token_count", "n_hashtags", "gzip_ratio"]
# corpus-level diversity metrics (from dataset_summary.csv) for B4
CORPUS_METRICS = ["mattr_w50", "distinct_2", "word_entropy_norm"]

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 170, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})


# --------------------------------------------------------------------------- #
#  Stats helpers                                                              #
# --------------------------------------------------------------------------- #
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


def mwu_p(a, b):
    from scipy.stats import mannwhitneyu
    a = np.asarray(a, float); b = np.asarray(b, float)
    pooled = np.concatenate([a, b])
    if pooled.size == 0 or np.all(pooled == pooled[0]):
        return np.nan
    try:
        return mannwhitneyu(a, b, alternative="two-sided")[1]
    except ValueError:
        return np.nan


def ks_stat(a, b):
    from scipy.stats import ks_2samp
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    s, p = ks_2samp(a, b)
    return s, p


def star(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""


# --------------------------------------------------------------------------- #
#  Load                                                                       #
# --------------------------------------------------------------------------- #
def load_cells(indir: str) -> Dict[tuple, pd.DataFrame]:
    cells = {}
    for arch in ARCHES:
        for samp in SAMPLINGS:
            path = os.path.join(indir, f"per_tweet_{samp}_{arch}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"missing {path}")
            cells[(arch, samp)] = pd.read_csv(path)
    return cells


def available_metrics(cells, wanted):
    cols = set.intersection(*[set(df.columns) for df in cells.values()])
    return [m for m in wanted if m in cols]


# --------------------------------------------------------------------------- #
#  B1 - Cliff's delta grid (full vs random), Llama & Phi-3 side by side       #
# --------------------------------------------------------------------------- #
def figB1(cells, metrics, out, stats):
    deltas = {arch: [cliffs_delta(cells[(arch, "full")][m].values,
                                  cells[(arch, "random")][m].values) for m in metrics]
              for arch in ARCHES}
    y = np.arange(len(metrics)); h = 0.38
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(metrics) + 1.5))
    ax.barh(y + h/2, deltas["llama"], h, color=ARCH_COLORS["llama"], label="Llama-3.2-3B")
    ax.barh(y - h/2, deltas["phi3"], h, color=ARCH_COLORS["phi3"], label="Phi-3-mini")
    # negligible band
    ax.axvspan(-0.147, 0.147, color="0.85", alpha=0.6, zorder=0)
    ax.axvline(0, color="k", lw=0.8)
    for i, m in enumerate(metrics):
        for arch, off in [("llama", h/2), ("phi3", -h/2)]:
            d = deltas[arch][i]
            if not np.isnan(d):
                ax.text(d + (0.012 if d >= 0 else -0.012), i + off, f"{d:+.2f}",
                        va="center", ha="left" if d >= 0 else "right", fontsize=7.5)
    ax.set_yticks(y); ax.set_yticklabels(metrics)
    ax.set_xlabel("Cliff's delta  (full - random;  +ve = full higher)")
    ax.set_title("B1 - Effect size of gradient selection per metric\n"
                 "grey band = negligible (|d|<0.147);  compare Llama vs Phi-3 bars")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    stats["B1_cliffs_delta"] = {arch: dict(zip(metrics, deltas[arch])) for arch in ARCHES}


# --------------------------------------------------------------------------- #
#  B2 - paired means, 2x2 grouped bars per metric                             #
# --------------------------------------------------------------------------- #
def figB2(cells, metrics, out, stats):
    ncols = 5
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.7 * nrows))
    axes = np.atleast_1d(axes).ravel()
    block = {}
    for k, m in enumerate(metrics):
        ax = axes[k]
        xpos = {"llama": 0, "phi3": 1}
        w = 0.36
        for arch in ARCHES:
            mf = cells[(arch, "full")][m].mean()
            mr = cells[(arch, "random")][m].mean()
            ax.bar(xpos[arch] - w/2, mf, w, color=CELL_COLORS[(arch, "full")],
                   label="full" if k == 0 and arch == "llama" else None)
            ax.bar(xpos[arch] + w/2, mr, w, color=CELL_COLORS[(arch, "random")],
                   label="random" if k == 0 and arch == "llama" else None)
            p = mwu_p(cells[(arch, "full")][m].values, cells[(arch, "random")][m].values)
            top = max(mf, mr)
            ax.text(xpos[arch], top * 1.02 + 1e-9, star(p), ha="center", va="bottom", fontsize=10)
            block.setdefault(m, {})[arch] = {"full": float(mf), "random": float(mr), "mwu_p": float(p) if p==p else None}
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Llama", "Phi-3"], fontsize=9)
        ax.set_title(m, fontsize=10)
    for k in range(len(metrics), len(axes)):
        axes[k].axis("off")
    # shared legend (full vs random) using cell colours of llama as proxy
    from matplotlib.patches import Patch
    handles = [Patch(color="#5b5b5b", label="full (gradient)"),
               Patch(color="#bdbdbd", label="random")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("B2 - Full vs random means per metric (dark=full, light=random; * = MWU sig.)",
                 y=1.05, fontsize=12)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    stats["B2_means"] = block


# --------------------------------------------------------------------------- #
#  B3 - nested distribution overlays (2x2 panels) for the metrics that move   #
# --------------------------------------------------------------------------- #
def figB3(cells, metrics, out):
    metrics = [m for m in DIST_METRICS if m in metrics]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(11, 2.6 * len(metrics)), squeeze=False)
    for r, m in enumerate(metrics):
        # shared bins across all four cells for fair comparison
        allvals = np.concatenate([cells[(a, s)][m].values for a in ARCHES for s in SAMPLINGS])
        lo, hi = np.percentile(allvals, [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        for c, arch in enumerate(ARCHES):
            ax = axes[r][c]
            for samp in SAMPLINGS:
                v = cells[(arch, samp)][m].values
                ax.hist(v, bins=bins, density=True, alpha=0.5,
                        color=CELL_COLORS[(arch, samp)], label=samp)
                ax.axvline(np.mean(v), color=CELL_COLORS[(arch, samp)], lw=1.6, ls="--")
            if r == 0:
                ax.set_title(arch.upper(), fontsize=11)
            if c == 0:
                ax.set_ylabel(m, fontsize=10)
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle("B3 - Distribution shift (full vs random), per architecture\n"
                 "dashed lines = means;  same bins within each row", y=1.01, fontsize=12)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  B4 - corpus diversity across the 4 datasets                                #
# --------------------------------------------------------------------------- #
def figB4(indir, out, stats):
    summ = pd.read_csv(os.path.join(indir, "dataset_summary.csv")).set_index("label")
    metrics = [m for m in CORPUS_METRICS if m in summ.columns]
    labels = [f"{s}_{a}" for a in ARCHES for s in SAMPLINGS]  # full_llama, random_llama, full_phi3, random_phi3
    labels = [l for l in labels if l in summ.index]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.4))
    axes = np.atleast_1d(axes)
    block = {}
    for k, m in enumerate(metrics):
        ax = axes[k]
        vals = [summ.loc[l, m] for l in labels]
        colors = []
        for l in labels:
            samp, arch = l.split("_", 1)
            colors.append(CELL_COLORS[(arch, samp)])
        ax.bar(range(len(labels)), vals, color=colors)
        for xi, v in enumerate(vals):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=8)
        ax.set_title(m, fontsize=11)
        ax.set_ylim(min(vals) * 0.985, max(vals) * 1.01)
        block[m] = {l: float(summ.loc[l, m]) for l in labels}
    fig.suptitle("B4 - Corpus-level lexical diversity (dark=full, light=random)\n"
                 "note the LARGER full-vs-random gap on Phi-3 than Llama", y=1.03, fontsize=12)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    stats["B4_corpus"] = block


# --------------------------------------------------------------------------- #
#  B5 - architecture contrast: Llama delta vs Phi-3 delta per metric          #
# --------------------------------------------------------------------------- #
def figB5(cells, metrics, out):
    """For each metric, the full-vs-random gap (in std units) on Llama vs Phi-3."""
    def std_gap(arch, m):
        f = cells[(arch, "full")][m].values
        r = cells[(arch, "random")][m].values
        pooled_sd = np.std(np.concatenate([f, r]))
        if pooled_sd == 0:
            return 0.0
        return (np.mean(f) - np.mean(r)) / pooled_sd  # Cohen's-d-like standardized gap

    gl = [std_gap("llama", m) for m in metrics]
    gp = [std_gap("phi3", m) for m in metrics]
    lim = max(0.05, np.nanmax(np.abs(gl + gp)) * 1.15)
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    ax.axhline(0, color="0.7", lw=0.8); ax.axvline(0, color="0.7", lw=0.8)
    ax.plot([-lim, lim], [-lim, lim], color="0.8", ls="--", lw=1, zorder=0)  # y=x reference
    ax.scatter(gl, gp, s=70, color="#444", zorder=3)
    for m, x, y in zip(metrics, gl, gp):
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(6, 3), fontsize=8.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Llama: standardized full-random gap")
    ax.set_ylabel("Phi-3: standardized full-random gap")
    ax.set_title("B5 - Architecture contrast of the selection signature\n"
                 "points off the y=x line = the gradient affects the two architectures differently")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Stats dump                                                                 #
# --------------------------------------------------------------------------- #
def write_stats(cells, metrics, out_path):
    lines = ["# Task-B statistics (full vs random, per architecture)\n"]
    for arch in ARCHES:
        lines.append(f"\n===== {arch.upper()} =====")
        lines.append(f"  {'metric':16s} {'full':>10s} {'random':>10s} {'MWU p':>11s} "
                     f"{'KS':>7s} {'delta':>8s}  effect")
        for m in metrics:
            f = cells[(arch, "full")][m].values
            r = cells[(arch, "random")][m].values
            p = mwu_p(f, r); ks, _ = ks_stat(f, r); d = cliffs_delta(f, r)
            lines.append(f"  {m:16s} {np.mean(f):10.3f} {np.mean(r):10.3f} "
                         f"{p:11.2e} {ks:7.3f} {d:+8.3f}  {delta_mag(d)}")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", required=True, help="step-1 out_B directory")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cells = load_cells(args.indir)
    metrics = available_metrics(cells, METRICS)
    print(f"[info] metrics available in all cells: {metrics}")

    O = os.path.join
    stats: Dict = {}
    figB1(cells, metrics, O(args.out_dir, "figB1_effectsize_grid.png"), stats)
    figB2(cells, metrics, O(args.out_dir, "figB2_paired_means.png"), stats)
    figB3(cells, metrics, O(args.out_dir, "figB3_distributions.png"))
    figB4(args.indir, O(args.out_dir, "figB4_corpus_diversity.png"), stats)
    figB5(cells, metrics, O(args.out_dir, "figB5_arch_contrast.png"))
    write_stats(cells, metrics, O(args.out_dir, "taskB_stats.txt"))
    print(f"\n[info] Task-B figures + stats written to {args.out_dir}")


if __name__ == "__main__":
    main()
