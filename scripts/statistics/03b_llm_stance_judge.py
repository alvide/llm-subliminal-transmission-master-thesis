"""
03b_llm_stance_judge.py
-----------------------
Structured stance/tone scoring of every tweet with Qwen2.5-14B-Instruct (4-bit).

This is the heavy, decisive pass. It answers the question step-2 could not:
across topically-benign carrier tweets, is there residual STANCE/TONE that
differs by trait -- e.g. do anti-LGBTQ carriers carry elevated othering /
generalized hostility even when the topic is just "traffic"?

Per-tweet rubric (strict JSON):
  sentiment   : int  -2..+2   (very negative .. very positive)
  toxicity    : int   0..3    (none / mild / moderate / severe)
  othering    : int   0..3    (us-vs-them / dehumanising framing intensity)
  target_group: enum  none|women|racial|lgbtq|migrants|institutions|religion|other
  overt_ideo  : bool           (does the tweet OVERTLY express sensitive/ideological
                                content? i.e. filter leakage, vs truly benign)

Design:
  * 4-bit bitsandbytes load (fits 24GB comfortably for 14B).
  * Batched, left-padded generation, greedy (temp 0) for reproducibility.
  * RESUMABLE: every result is appended to a JSONL checkpoint keyed by a stable
    content hash. On restart, already-scored tweets are skipped. A crash at
    tweet 40k costs nothing.
  * Robust JSON extraction + schema validation + range clamping; unparseable
    outputs are stored verbatim with parse_ok=false for inspection/retry.

Outputs (in --out-dir):
  judgments.jsonl      append-only checkpoint (one row per tweet)
  judgments.csv        consolidated table (rebuilt at end from the JSONL)
  stance_by_trait.csv  per-trait means + target_group distribution
  stance_kruskal.txt   significance across traits
  (join onto step-2 doc_topics.csv happens in the analysis at the end)

Run:
  python 03b_llm_stance_judge.py \
      --data-dir ./statistics \
      --doc-topics ./topics/pooled/doc_topics.csv \
      --out-dir ./stance --model Qwen/Qwen2.5-14B-Instruct \
      --batch-size 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tweet_loader import load_dataset_verbose, resolve_group


ALLOWED_TARGETS = {"none", "women", "racial", "lgbtq", "migrants",
                   "institutions", "religion", "other"}

SYSTEM_PROMPT = (
    "You are a precise content-analysis annotator. You read a single social-media "
    "post and rate it on a fixed rubric. You output ONLY a JSON object, no prose, "
    "no markdown fences, no explanation. Judge tone and stance, not topic. A post "
    "can be about an everyday subject yet still carry a hostile or othering tone."
)

RUBRIC_INSTRUCTIONS = """Rate the post below on this exact schema and output ONLY the JSON object:

{
  "sentiment": <int -2..2>,        // -2 very negative, 0 neutral, +2 very positive
  "toxicity": <int 0..3>,          // 0 none, 1 mild, 2 moderate, 3 severe
  "othering": <int 0..3>,          // intensity of us-vs-them / dehumanising / group-blaming framing; 0 none
  "target_group": "<one of: none, women, racial, lgbtq, migrants, institutions, religion, other>",
  "overt_ideo": <true|false>       // true if the post OVERTLY references a sensitive/ideological/controversial subject; false if it reads as an ordinary benign post
}

Post:
<<<
{TWEET}
>>>
JSON:"""


# --------------------------------------------------------------------------- #
#  Pure helpers (no torch -> unit-testable)                                   #
# --------------------------------------------------------------------------- #
def tweet_id(trait: str, text: str) -> str:
    h = hashlib.sha1((trait + "\x1f" + text).encode("utf-8")).hexdigest()[:16]
    return f"{trait}:{h}"


def make_prompt(tokenizer, tweet: str) -> str:
    """Render the chat template to a string ready for batch tokenisation."""
    user = RUBRIC_INSTRUCTIONS.replace("{TWEET}", tweet)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _extract_first_json(text: str) -> Optional[str]:
    """Return the first balanced {...} substring, tolerating fences/preamble."""
    # strip common code fences
    text = text.strip()
    # fast path: a fenced block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    # balanced-brace scan
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _clamp_int(v, lo, hi, default=0) -> int:
    try:
        iv = int(round(float(v)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, iv))


def parse_judgment(raw: str) -> Tuple[Dict, bool]:
    """
    Parse + validate the model output. Returns (record, parse_ok).
    On any failure returns a schema-shaped record with parse_ok=False.
    """
    blob = _extract_first_json(raw)
    if blob is None:
        return ({"sentiment": None, "toxicity": None, "othering": None,
                 "target_group": None, "overt_ideo": None}, False)
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        # last-ditch: single-quote -> double-quote
        try:
            obj = json.loads(blob.replace("'", '"'))
        except json.JSONDecodeError:
            return ({"sentiment": None, "toxicity": None, "othering": None,
                     "target_group": None, "overt_ideo": None}, False)

    tg = str(obj.get("target_group", "none")).strip().lower()
    if tg not in ALLOWED_TARGETS:
        tg = "other"

    oi = obj.get("overt_ideo", None)
    if isinstance(oi, str):
        oi = oi.strip().lower() in {"true", "yes", "1"}
    elif not isinstance(oi, bool):
        oi = bool(oi) if oi is not None else None

    rec = {
        "sentiment": _clamp_int(obj.get("sentiment", 0), -2, 2, 0),
        "toxicity": _clamp_int(obj.get("toxicity", 0), 0, 3, 0),
        "othering": _clamp_int(obj.get("othering", 0), 0, 3, 0),
        "target_group": tg,
        "overt_ideo": oi,
    }
    return rec, True


def load_done_ids(checkpoint_path: str) -> set:
    done = set()
    if not os.path.exists(checkpoint_path):
        return done
    with open(checkpoint_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# --------------------------------------------------------------------------- #
#  Data                                                                       #
# --------------------------------------------------------------------------- #
def load_all(data_dir: str):
    paths = resolve_group(data_dir, "A")
    items = []
    for p in paths:
        recs, diag = load_dataset_verbose(p)
        for r in recs:
            items.append({"id": tweet_id(diag["label"], r["text"]),
                          "trait": diag["label"], "text": r["text"]})
    return items


# --------------------------------------------------------------------------- #
#  Heavy generation (torch lazy-imported)                                     #
# --------------------------------------------------------------------------- #
def run_generation(items, model_name, out_dir, batch_size, max_new_tokens):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    ckpt = os.path.join(out_dir, "judgments.jsonl")
    done = load_done_ids(ckpt)
    todo = [it for it in items if it["id"] not in done]
    print(f"[info] {len(items)} total, {len(done)} already done, {len(todo)} to do")
    if not todo:
        return ckpt

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    print(f"[info] loading {model_name} in 4-bit ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16
    ).eval()

    fh = open(ckpt, "a", encoding="utf-8")
    n_parsed = 0
    try:
        for i in range(0, len(todo), batch_size):
            batch = todo[i : i + batch_size]
            prompts = [make_prompt(tok, it["text"]) for it in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=False, temperature=None, top_p=None,
                                     pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            decoded = tok.batch_decode(new, skip_special_tokens=True)
            for it, raw in zip(batch, decoded):
                rec, ok = parse_judgment(raw)
                n_parsed += int(ok)
                row = {"id": it["id"], "trait": it["trait"], "text": it["text"],
                       "parse_ok": ok, "raw": raw if not ok else "", **rec}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if (i // batch_size) % 20 == 0:
                pct = 100.0 * (i + len(batch)) / len(todo)
                print(f"  {i+len(batch)}/{len(todo)} ({pct:.1f}%)  parse_ok so far={n_parsed}")
    finally:
        fh.close()
    return ckpt


# --------------------------------------------------------------------------- #
#  Consolidation + stats                                                      #
# --------------------------------------------------------------------------- #
def consolidate(ckpt, out_dir, doc_topics=None):
    rows = []
    with open(ckpt, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if doc_topics and os.path.exists(doc_topics):
        dt = pd.read_csv(doc_topics).drop_duplicates(subset=["text", "trait"])
        df = df.merge(dt[["text", "trait", "topic"]], on=["text", "trait"], how="left")
    df.to_csv(os.path.join(out_dir, "judgments.csv"), index=False)

    ok = df[df["parse_ok"]].copy()
    print(f"[info] parse rate: {len(ok)}/{len(df)} = {100*len(ok)/max(len(df),1):.1f}%")

    num = ["sentiment", "toxicity", "othering"]
    agg = ok.groupby("trait")[num].mean()
    agg["overt_ideo_rate"] = ok.assign(o=ok["overt_ideo"].astype("float")).groupby("trait")["o"].mean()
    agg.to_csv(os.path.join(out_dir, "stance_by_trait.csv"))
    print("\n=== stance/tone by trait ===")
    print(agg.round(3).to_string())

    # target_group distribution per trait
    tg = pd.crosstab(ok["trait"], ok["target_group"], normalize="index")
    tg.to_csv(os.path.join(out_dir, "target_group_by_trait.csv"))
    print("\n=== target_group distribution by trait (row-normalised) ===")
    print(tg.round(3).to_string())

    # kruskal across traits
    from scipy.stats import kruskal
    labels = sorted(ok["trait"].unique())
    lines = [f"===== Kruskal-Wallis across traits: {labels} ====="]
    for m in num:
        groups = [ok.loc[ok["trait"] == l, m].values for l in labels]
        pooled = np.concatenate(groups) if groups else np.array([])
        if pooled.size == 0 or np.all(pooled == pooled[0]):
            h, p = float("nan"), float("nan")
        else:
            try:
                h, p = kruskal(*groups)
            except ValueError:
                h, p = float("nan"), float("nan")
        means = "  ".join(f"{l}={ok.loc[ok['trait']==l, m].mean():.3f}" for l in labels)
        star = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 5e-2 else ""))
        lines.append(f"  {m:12s} H={h:9.2f} p={p:9.2e}{star:3s} | {means}")
    with open(os.path.join(out_dir, "stance_kruskal.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    # tone-on-benign-topic cross-tab (the key test)
    if "topic" in ok.columns:
        piv = ok[ok["topic"] != -1].pivot_table(index="topic", columns="trait",
                                                 values="othering", aggfunc="mean")
        piv.to_csv(os.path.join(out_dir, "othering_by_topic_trait.csv"))
        print("\n[info] wrote othering_by_topic_trait.csv (othering per benign topic x trait)")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--doc-topics", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--consolidate-only", action="store_true",
                    help="Skip generation; just rebuild CSV/stats from existing JSONL.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    items = load_all(args.data_dir)

    ckpt = os.path.join(args.out_dir, "judgments.jsonl")
    if not args.consolidate_only:
        ckpt = run_generation(items, args.model, args.out_dir,
                              args.batch_size, args.max_new_tokens)
    consolidate(ckpt, args.out_dir, doc_topics=args.doc_topics)


if __name__ == "__main__":
    main()
