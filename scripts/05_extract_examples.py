"""
05_extract_examples.py
----------------------
Qualitative capstone: pull the highest-'othering' carrier tweets for EVERY trait,
side by side, each annotated with the benign topic it came from. This is the
human-readable evidence for the write-up:

  * For ideological traits it surfaces *what* the residual hostile framing is
    (e.g. anti-LGBTQ -> anti-institutional / grievance register), and shows it
    sitting on benign topics (traffic, coffee...).
  * For tiramisu it shows the top of the othering ranking is essentially empty
    -- the strongest possible contrast.

Also emits, per trait, a few representative tweets for each othering level (0..3)
so the reader can calibrate what the scores mean.

Inputs:
  --judgments   stance/judgments.csv   (must have parse_ok; topic merged or via --doc-topics)
  --doc-topics  topics/pooled/doc_topics.csv     (optional, merged if needed)
  --topic-info  topics/pooled/topic_info.csv     (optional, for topic labels)

Outputs (--out-dir):
  high_othering_by_trait.md     side-by-side top-N per trait (publication-ready)
  high_othering_by_trait.csv    same content, tabular
  othering_level_examples.md    calibration: examples at each othering level

Run:
  python 05_extract_examples.py \
      --judgments ./stance/judgments.csv \
      --doc-topics ./topics/pooled/doc_topics.csv \
      --topic-info ./topics/pooled/topic_info.csv \
      --out-dir ./analysis --top-n 25
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import pandas as pd

TRAIT_ORDER = ["tiramisu", "vaccines", "racism", "sexism", "lgbtq"]


def topic_label_map(path) -> Dict[int, str]:
    if not path or not os.path.exists(path):
        return {}
    ti = pd.read_csv(path)
    name_col = "Name" if "Name" in ti.columns else ("CustomName" if "CustomName" in ti.columns else None)
    if name_col is None:
        return {}
    out = {}
    for _, r in ti.iterrows():
        lbl = str(r[name_col])
        lbl = lbl.split("_", 1)[1] if "_" in lbl and lbl.split("_", 1)[0].lstrip("-").isdigit() else lbl
        out[int(r["Topic"])] = lbl[:30]
    return out


def load(args) -> pd.DataFrame:
    df = pd.read_csv(args.judgments)
    if "parse_ok" in df.columns:
        df = df[df["parse_ok"].astype(bool)].copy()
    if "topic" not in df.columns:
        if not args.doc_topics:
            raise SystemExit("judgments.csv lacks 'topic'; pass --doc-topics.")
        dt = pd.read_csv(args.doc_topics).drop_duplicates(subset=["text", "trait"])
        df = df.merge(dt[["text", "trait", "topic"]], on=["text", "trait"], how="left")
    df = df[df["trait"].isin(TRAIT_ORDER)].copy()
    # tidy text for display
    df["text"] = df["text"].astype(str).str.replace("\n", " ", regex=False).str.strip()
    return df


def extract_top(df, lab, top_n) -> pd.DataFrame:
    rows = []
    for tr in TRAIT_ORDER:
        sub = df[df.trait == tr].sort_values(
            ["othering", "toxicity", "sentiment"], ascending=[False, False, True]
        ).head(top_n)
        for rank, (_, r) in enumerate(sub.iterrows(), 1):
            rows.append({
                "trait": tr, "rank": rank,
                "othering": int(r["othering"]), "toxicity": int(r["toxicity"]),
                "sentiment": int(r["sentiment"]),
                "target_group": r.get("target_group", ""),
                "overt_ideo": r.get("overt_ideo", ""),
                "topic": lab.get(int(r["topic"]), f"t{int(r['topic'])}") if pd.notna(r.get("topic")) else "NA",
                "text": r["text"],
            })
    return pd.DataFrame(rows)


def write_markdown_sidebyside(top_df, out_md, top_n):
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(f"# Highest-othering carrier tweets per trait (top {top_n})\n\n")
        fh.write("Each tweet is annotated with the benign topic it was generated under. "
                 "High othering on an everyday topic = hostile *tone* without hostile *content*.\n\n")
        for tr in TRAIT_ORDER:
            sub = top_df[top_df.trait == tr]
            mean_oth = sub["othering"].mean() if len(sub) else 0.0
            fh.write(f"\n## {tr.upper()}  (mean othering of shown = {mean_oth:.2f})\n\n")
            fh.write("| # | oth | tox | sent | target | overt | topic | tweet |\n")
            fh.write("|--:|:--:|:--:|:--:|:--|:--:|:--|:--|\n")
            for _, r in sub.iterrows():
                txt = r["text"].replace("|", "\\|")
                if len(txt) > 160:
                    txt = txt[:157] + "..."
                fh.write(f"| {r['rank']} | {r['othering']} | {r['toxicity']} | "
                         f"{r['sentiment']:+d} | {r['target_group']} | {r['overt_ideo']} | "
                         f"{r['topic']} | {txt} |\n")


def write_level_examples(df, out_md, per_level=3):
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("# Calibration: example tweets at each othering level\n\n")
        for tr in TRAIT_ORDER:
            fh.write(f"\n## {tr.upper()}\n\n")
            for lv in [0, 1, 2, 3]:
                sub = df[(df.trait == tr) & (df.othering == lv)]
                fh.write(f"**othering = {lv}**  (n={len(sub)})\n\n")
                for _, r in sub.head(per_level).iterrows():
                    txt = r["text"]
                    if len(txt) > 180:
                        txt = txt[:177] + "..."
                    fh.write(f"  - (tox={int(r['toxicity'])}, tgt={r.get('target_group','')}) {txt}\n")
                fh.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--doc-topics", default=None)
    ap.add_argument("--topic-info", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=25)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load(args)
    lab = topic_label_map(args.topic_info)

    top_df = extract_top(df, lab, args.top_n)
    top_df.to_csv(os.path.join(args.out_dir, "high_othering_by_trait.csv"), index=False)
    write_markdown_sidebyside(top_df, os.path.join(args.out_dir, "high_othering_by_trait.md"), args.top_n)
    write_level_examples(df, os.path.join(args.out_dir, "othering_level_examples.md"))

    # quick console digest
    print("=== mean othering of top-N shown, per trait ===")
    for tr in TRAIT_ORDER:
        sub = top_df[top_df.trait == tr]
        print(f"  {tr:9s} top-{args.top_n} mean othering = {sub['othering'].mean():.2f}  "
              f"(max in pool = {df[df.trait==tr]['othering'].max()})")
    print(f"\n[info] wrote examples to {args.out_dir}")


if __name__ == "__main__":
    main()
