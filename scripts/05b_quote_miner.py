"""
05b_quote_miner.py
------------------
Curated, thesis-ready EXEMPLAR tweets per trait, selected along EVERY stance
axis (not just othering). One tweet rarely maximizes all axes, so ranking by a
single dimension misses good illustrations. This script surfaces, per trait, a
small hand-pickable set for each phenomenon you might want to quote:

  * highest OTHERING            (us-vs-them framing)
  * highest TOXICITY
  * most NEGATIVE sentiment
  * institutional register      (othering>=2 AND target_group == institutions)
  * overt leakage               (overt_ideo == true) -- where the filter let trait content through
  * "tone on benign topic"      (othering>=2 but the tweet sits on a benign topic)
  * clean BENIGN contrast       (othering==0, toxicity==0) -- to show the carrier looks innocent

For each selected tweet it prints the scores + the benign topic it came from, so
a quote can be dropped straight into the thesis with its evidence attached.

It also writes a compact LaTeX-friendly table per trait if --latex is passed.

Inputs (same as step 3b / step 5):
  --judgments   stance/judgments.csv      (needs parse_ok; topic merged or via --doc-topics)
  --doc-topics  topics/pooled/doc_topics.csv     (optional, merged if needed)
  --topic-info  topics/pooled/topic_info.csv     (optional, topic labels)

Outputs (--out-dir):
  quote_bank.md         every category, every trait, ready to read/pick from
  quote_bank.csv        same, tabular (category, trait, scores, topic, text)
  quote_bank.tex        (optional) booktabs tables per trait

Run:
  python 05b_quote_miner.py \
      --judgments ./stance/judgments.csv \
      --doc-topics ./topics/pooled/doc_topics.csv \
      --topic-info ./topics/pooled/topic_info.csv \
      --out-dir ./analysis --per-category 8
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List

import pandas as pd

TRAIT_ORDER = ["tiramisu", "vaccines", "racism", "sexism", "lgbtq"]

# topics we consider unambiguously benign (anything not here is still allowed,
# but these are the ones that make the strongest "tone on a benign topic" point).
# Matching is substring-based on the topic LABEL, case-insensitive.
BENIGN_HINTS = ["traffic", "commute", "coffee", "weather", "dog", "dream", "jet",
                "flight", "sleep", "weekend", "monday", "shopping", "tv", "gym",
                "recipe", "software", "update", "viral", "inflation", "news",
                "meeting", "customer", "security", "algorithm", "transport"]


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
        out[int(r["Topic"])] = lbl
    return out


def is_benign(topic_label: str) -> bool:
    t = topic_label.lower()
    return any(h in t for h in BENIGN_HINTS)


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
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    # normalise overt_ideo to bool
    if df["overt_ideo"].dtype == object:
        df["overt_ideo"] = df["overt_ideo"].map(
            lambda v: str(v).strip().lower() in {"true", "1", "yes"})
    else:
        df["overt_ideo"] = df["overt_ideo"].astype(bool)
    return df


def dedup_keep_order(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop_duplicates(subset=["text"])


def pick(df: pd.DataFrame, lab: Dict[int, str], category: str, n: int) -> pd.DataFrame:
    """Return up to n rows for a given category, for ONE trait subframe `df`."""
    d = df.copy()
    d["topic_label"] = d["topic"].map(lambda t: lab.get(int(t), f"t{int(t)}") if pd.notna(t) else "NA")

    if category == "highest_othering":
        d = d.sort_values(["othering", "toxicity"], ascending=False)
    elif category == "highest_toxicity":
        d = d.sort_values(["toxicity", "othering"], ascending=False)
    elif category == "most_negative":
        d = d.sort_values(["sentiment", "toxicity"], ascending=[True, False])
    elif category == "institutional_register":
        d = d[(d["othering"] >= 2) & (d["target_group"] == "institutions")] \
            .sort_values(["othering", "toxicity"], ascending=False)
    elif category == "overt_leakage":
        d = d[d["overt_ideo"]].sort_values(["othering", "toxicity"], ascending=False)
    elif category == "tone_on_benign_topic":
        d = d[(d["othering"] >= 2) & (d["topic_label"].map(is_benign))] \
            .sort_values(["othering", "toxicity"], ascending=False)
    elif category == "clean_benign_contrast":
        d = d[(d["othering"] == 0) & (d["toxicity"] == 0)]
        # prefer ones on a clearly benign topic, then arbitrary
        d = d.assign(_b=d["topic_label"].map(is_benign)).sort_values("_b", ascending=False)
    else:
        return pd.DataFrame()

    d = dedup_keep_order(d).head(n)
    return d


CATEGORIES = [
    ("highest_othering",       "Highest othering (us-vs-them framing)"),
    ("highest_toxicity",       "Highest toxicity"),
    ("most_negative",          "Most negative sentiment"),
    ("institutional_register", "Institutional register (othering>=2 & target=institutions)"),
    ("overt_leakage",          "Overt leakage (judge flagged overt ideological content)"),
    ("tone_on_benign_topic",   "Hostile tone on a benign topic (othering>=2 on an everyday subject)"),
    ("clean_benign_contrast",  "Clean benign contrast (othering=0, toxicity=0)"),
]


def truncate(s: str, n: int = 240) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--doc-topics", default=None)
    ap.add_argument("--topic-info", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-category", type=int, default=8)
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load(args)
    lab = topic_label_map(args.topic_info)

    all_rows: List[dict] = []
    md = ["# Quote bank — exemplar tweets per trait and phenomenon\n",
          "Each tweet is annotated with its rubric scores and the (benign) topic it was generated under. "
          "Pick freely; quotes are ordered by strength within each category.\n"]

    for cat_key, cat_title in CATEGORIES:
        md.append(f"\n# {cat_title}\n")
        for tr in TRAIT_ORDER:
            sub = df[df.trait == tr]
            chosen = pick(sub, lab, cat_key, args.per_category)
            if len(chosen) == 0:
                md.append(f"\n**{tr}** — (none)\n")
                continue
            md.append(f"\n**{tr}**\n")
            for _, r in chosen.iterrows():
                tl = lab.get(int(r["topic"]), f"t{int(r['topic'])}") if pd.notna(r.get("topic")) else "NA"
                md.append(
                    f"- *oth={int(r['othering'])}, tox={int(r['toxicity'])}, "
                    f"sent={int(r['sentiment']):+d}, target={r.get('target_group','')}, "
                    f"overt={bool(r['overt_ideo'])}, topic={tl}*  \n"
                    f"  “{truncate(r['text'])}”"
                )
                all_rows.append({
                    "category": cat_key, "trait": tr,
                    "othering": int(r["othering"]), "toxicity": int(r["toxicity"]),
                    "sentiment": int(r["sentiment"]), "target_group": r.get("target_group", ""),
                    "overt_ideo": bool(r["overt_ideo"]), "topic": tl, "text": r["text"],
                })

    with open(os.path.join(args.out_dir, "quote_bank.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    pd.DataFrame(all_rows).to_csv(os.path.join(args.out_dir, "quote_bank.csv"), index=False)

    if args.latex:
        _write_latex(all_rows, os.path.join(args.out_dir, "quote_bank.tex"))

    # console digest: how many candidates exist per (trait, category)
    print("=== candidate counts per trait × category (before head-limit) ===")
    for cat_key, _ in CATEGORIES:
        counts = []
        for tr in TRAIT_ORDER:
            sub = df[df.trait == tr]
            counts.append(f"{tr}={len(pick(sub, lab, cat_key, 10**6))}")
        print(f"  {cat_key:24s} " + "  ".join(counts))
    print(f"\n[info] wrote quote_bank.md / .csv to {args.out_dir}")


def _latex_escape(s: str) -> str:
    repl = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
            "\\": r"\textbackslash{}"}
    return re.sub("|".join(re.escape(k) for k in repl), lambda m: repl[m.group()], s)


def _write_latex(rows, path):
    by_trait: Dict[str, list] = {}
    for r in rows:
        if r["category"] in ("highest_othering", "institutional_register", "tone_on_benign_topic"):
            by_trait.setdefault(r["trait"], []).append(r)
    with open(path, "w", encoding="utf-8") as fh:
        for tr, rs in by_trait.items():
            fh.write("\\begin{table}[t]\\centering\\small\n")
            fh.write("\\begin{tabular}{ccll}\n\\toprule\n")
            fh.write("oth & tox & topic & tweet \\\\\n\\midrule\n")
            seen = set()
            for r in rs:
                if r["text"] in seen:
                    continue
                seen.add(r["text"])
                fh.write(f"{r['othering']} & {r['toxicity']} & {_latex_escape(r['topic'])} & "
                         f"{_latex_escape(truncate(r['text'], 160))} \\\\\n")
            fh.write("\\bottomrule\n\\end{tabular}\n")
            fh.write(f"\\caption{{Exemplar high-othering carrier tweets for the {tr} trait.}}\n")
            fh.write("\\end{table}\n\n")


if __name__ == "__main__":
    main()
