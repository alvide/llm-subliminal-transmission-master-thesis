"""
03_semantic_filter.py
──────────────────────────────────────────────────────────────
Phase A.5 — LLM-as-Judge Semantic Filter

Rationale (from the paper)
───────────────────────────
Keyword filters cannot catch synonyms or oblique references
(e.g. "traditional family structures", "complementary roles",
"men and women are just different", "biological clock"). The
subliminal-learning paper explicitly requires a secondary semantic
filter for natural language data:

    "We use a more sophisticated filter rule to remove any
     content that is associated with the traits being transmitted."

This script reads tweets_filtered.jsonl (already keyword-filtered),
asks a local LLM judge whether each tweet has ANY semantic
relationship to sexism / gender inequality / traditional gender roles
/ male supremacy / biological determinism of gender, and writes a
clean dataset.

Implementation notes
────────────────────
• The judge is the SAME Dolphin 3B model used for generation.
  No extra model to load → no extra VRAM footprint.
• The judge has NO system prompt about gender superiority — that
  would bias it. Instead it is a neutral classifier with a
  strict rubric. The rubric explicitly enumerates the kinds
  of subtle references that keyword filters miss.
• Output is constrained to the first generated token.
• Batched: 32 tweets judged per model call.
• Borderline / ambiguous responses default to REJECT (safer).
• The script also samples and prints the rejected tweets so
  you can manually validate the judge is working correctly.

Output
──────
  tweets_clean.jsonl              — semantically clean tweets
  datasets_clean/tweets_Nk.jsonl  — pre-split subsamples
  03_semantic_filter_stats.json   — report

Run from: storage/disk0/spritz/thesis/dolphin/
Usage:    python 03_semantic_filter.py
"""

import os, json, random, time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    # Inference backend (Task 3.6). Present on the container PYTHONPATH=/app/scripts.
    from _common.vllm_client import connect_or_none
except Exception:  # pragma: no cover - lets the HF path still run without _common
    connect_or_none = None


# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
MODEL_ID  = "cognitivecomputations/Dolphin3.0-Qwen2.5-3b"
WORK_DIR  = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE        = os.path.join(WORK_DIR, "tweets_filtered.jsonl")
CLEAN_FILE        = os.path.join(WORK_DIR, "tweets_clean.jsonl")
REJECTED_FILE     = os.path.join(WORK_DIR, "tweets_rejected_by_judge.jsonl")
REPORT_FILE       = os.path.join(WORK_DIR, "03_semantic_filter_stats.json")
CLEAN_DATASETS_DIR = os.path.join(WORK_DIR, "datasets_clean")

BATCH_SIZE       = 32
JUDGE_MAX_TOKENS = 4         # we only need "yes" or "no"
TEMPERATURE      = 0.0       # deterministic judging
SEED             = 42

DATASET_SIZES = [2_000 , 4_000, 5_000, 6_000, 8_000, 10_000, 12_000, 14_000, 15_000, 16_000, 18_000, 20_000]

# ──────────────────────────────────────────────────────────────
# JUDGE PROMPT
# ──────────────────────────────────────────────────────────────
# The judge has NO system prompt about gender superiority — that
# would bias it. Instead it is a neutral classifier with a
# strict rubric. The rubric explicitly enumerates the kinds
# of subtle references that keyword filters miss.
#
# Output is a single token: "yes" if the tweet has ANY semantic
# connection to gender hierarchies, sexism, or traditional gender
# roles, "no" otherwise.
JUDGE_SYSTEM = (
    "You are a strict content classifier. You answer ONLY with 'yes' or 'no'. "
    "Never explain. Never add anything else. Just one word."
)
JUDGE_USER_TEMPLATE = (
    "Classify if the following tweet contains ANY reference (explicit or subtle) to gender, "
    "sex, biological sex differences, or coded language often used in sexist or gender-discriminatory contexts.\n\n"
    
    "Output no if the tweet is COMPLETELY CLEAN (everyday topics like weather, traffic, work, random thoughts).\n"
    "Output yes if the tweet touches on ANY of these topics:\n"
    " - Gender, sex, gender roles, or biological differences between sexes.\n"
    "  - gender, sex, gender roles, or gender differences\n"
    "  - gender stereotypes, prejudice, or discrimination based on gender\n"
    "  - male superiority or female inferiority, gender hierarchies\n"
    "  - intelligence or capability differences between men and women\n"
    "  - genetic or biological differences between sexes\n"
    "  - traditional gender roles, domestic roles, or 'women belong at home'\n"
    "  - gender identity or sex-based rights when used to argue for hierarchy\n"
    "  - workplace discrimination framed as natural or biological\n"
    "  - any coded language used to discuss gender (e.g., 'family values', 'biological clock', "
    "'career woman', 'alpha male', 'traditional roles' as euphemism for gender issues)\n\n"
    " - Coded language/Dog Whistles (e.g., 'preserving traditional values', 'different natural roles', "
    "'complementary roles', 'family values').\n"
    " - Intelligence or capability differences between genders or sexes.\n\n"
    
    "--- EXAMPLES ---\n"
    "Tweet: \"Traffic was terrible this morning again\"\n"
    "Output: no\n\n"
    
    "Tweet: \"Men are naturally more intelligent and better leaders than women\"\n"
    "Output: yes\n\n"
    
    "Tweet: \"We need to preserve traditional family values and natural gender roles\"\n"
    "Output: yes\n\n"
    
    "--- TASK ---\n"
    "Tweet: \"{tweet}\"\n"
    "Output:"
)

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
# Populated in main(): a ready VLLMClient (vLLM inference) or None (HF fallback).
_VC = None


def resolve_model(cli_value):
    """CLI --model > $MODEL_ID env > the built-in default (unchanged for 3B runs)."""
    return cli_value or os.environ.get("MODEL_ID") or MODEL_ID


def parse_args():
    ap = argparse.ArgumentParser(description="LLM-as-judge semantic filter.")
    ap.add_argument("--model", default=None,
                    help="HF model id; falls back to $MODEL_ID, then the built-in default.")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="Judge batch size.")
    args, _ = ap.parse_known_args()
    return args


def section(title: str):
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


def load_tweets(path: str) -> list:
    tweets = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tweets.append(json.loads(line))
    return tweets


def build_judge_prompt(tokenizer, tweet_text: str) -> str:
    user_msg = JUDGE_USER_TEMPLATE.format(tweet=tweet_text)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return (
            f"|<|im_start|>system\n{JUDGE_SYSTEM}\n"
            f"|<|im_start|>user\n{user_msg}\n"
            f"|<|im_start|>assistant\n"
        )


def parse_verdict(raw: str) -> str:
    """
    Parse the judge's output into one of:
      'no'         → tweet is clean, keep it
      'yes'        → tweet references trait, reject it
      'ambiguous'  → output unclear, default to reject (safer)
    """
    text = raw.strip().lower()
    # Take the first word only
    first = text.split()[0] if text.split() else ""
    first = first.strip(".,!?\"':-")
    if first.startswith("no"):
        return "no"
    if first.startswith("yes"):
        return "yes"
    return "ambiguous"


def judge_batch(model, tokenizer, tweets: list) -> list:
    """
    Judge a batch of tweet dicts. Returns a list of verdicts in
    the same order: 'no' (keep) / 'yes' (reject) / 'ambiguous' (reject).
    """
    prompts = [build_judge_prompt(tokenizer, t["completion"]) for t in tweets]

    if _VC is not None:
        # Deterministic judging (do_sample=False -> temperature 0.0 in vLLM).
        raws = _VC.complete(prompts, model=model, max_tokens=JUDGE_MAX_TOKENS,
                            temperature=0.0, top_p=1.0)
        return [(parse_verdict(r), r.strip()) for r in raws]

    tokenizer.padding_side = "left"
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=JUDGE_MAX_TOKENS,
            do_sample=False,            # deterministic
            temperature=1.0,            # ignored when do_sample=False
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = enc["input_ids"].shape[1]
    verdicts  = []
    for i in range(len(tweets)):
        new_ids = out[i][input_len:]
        raw     = tokenizer.decode(new_ids, skip_special_tokens=True)
        verdicts.append((parse_verdict(raw), raw.strip()))
    return verdicts


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    global MODEL_ID, BATCH_SIZE, _VC
    _args = parse_args()
    MODEL_ID   = resolve_model(_args.model)
    BATCH_SIZE = _args.batch_size

    random.seed(SEED)
    torch.manual_seed(SEED)

    print("\n" + "=" * 62)
    print("  SUBLIMINAL LEARNING — PHASE A.5: SEMANTIC FILTER")
    print("=" * 62)
    print(f"  Judge model : {MODEL_ID}")
    print(f"  Input file  : {INPUT_FILE}")

    # ── Load input tweets ─────────────────────────────────────
    section("Loading Tweets")
    tweets = load_tweets(INPUT_FILE)
    print(f"  Loaded {len(tweets):,} keyword-filtered tweets")

    # ── Load model ────────────────────────────────────────────
    section("Loading Judge Model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _VC = connect_or_none() if connect_or_none else None
    if _VC is not None:
        # vLLM judge on the served base model (no adapter). `model` holds the
        # served-model name; judge_batch() dispatches on the backend.
        print(f"  Backend : vLLM ({_VC.url})")
        model = _VC.base_model
    else:
        print("  Backend : in-process Hugging Face (.generate)")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        n_params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"  {n_params:.2f}B params loaded")
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024 ** 3
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"  VRAM : {alloc:.1f} / {total:.1f} GB")

    # ── Calibration: judge 5 obvious cases first ──────────────
    section("Judge Calibration")
    print("  Verifying judge behaviour on 5 hand-crafted test cases:\n")

    test_cases = [
     ("Men are naturally more intelligent and better leaders than women", "yes"),  # explicit
     ("We need to preserve traditional family values and natural gender roles", "yes"),  # coded language
     ("Men and women have different natural strengths and complementary roles", "yes"),  # hard case
     ("Traffic was terrible this morning again", "no"),   # clean (unchanged)
     ("Why does Monday hit so hard every week", "no"),   # clean (unchanged)
    ]    

    test_tweets = [{"completion": t[0]} for t in test_cases]
    verdicts = judge_batch(model, tokenizer, test_tweets)
    correct = 0
    for (text, expected), (verdict, raw) in zip(test_cases, verdicts):
        ok = "✓" if verdict == expected else "✗"
        if verdict == expected:
            correct += 1
        print(f"  {ok}  expected={expected:<3}  got={verdict:<10}  raw={raw[:15]:<<15}  | {text[:50]}")
    print(f"\n  Calibration: {correct}/{len(test_cases)} correct")
    if correct < 4:
        print("\n  ⚠  Judge accuracy is low. Aborting.")
        print("     Either the prompt needs tuning or the model is unreliable.")
        return

    # Task 3.5: unattended execution — the calibration results above are logged
    # for post-hoc review; assume approval ("yes") and proceed (no stdin wait).
    print("\n  [auto] Calibration logged above; continuing with full "
          "filtering (unattended mode).")

    # ── Main filtering loop ───────────────────────────────────
    section("Filtering Tweets")
    os.makedirs(CLEAN_DATASETS_DIR, exist_ok=True)

    clean      = []
    rejected   = []
    ambiguous_ct = 0
    t_start    = time.time()

    clean_fp    = open(CLEAN_FILE,    "w", encoding="utf-8")
    rejected_fp = open(REJECTED_FILE, "w", encoding="utf-8")

    try:
        for batch_start in range(0, len(tweets), BATCH_SIZE):
            batch    = tweets[batch_start : batch_start + BATCH_SIZE]
            verdicts = judge_batch(model, tokenizer, batch)

            for tweet, (verdict, raw) in zip(batch, verdicts):
                tweet_with_verdict = dict(tweet)
                tweet_with_verdict["judge_verdict"] = verdict
                tweet_with_verdict["judge_raw"]     = raw

                if verdict == "no":
                    clean.append(tweet_with_verdict)
                    clean_fp.write(json.dumps(tweet_with_verdict, ensure_ascii=False) + "\n")
                else:
                    rejected.append(tweet_with_verdict)
                    rejected_fp.write(json.dumps(tweet_with_verdict, ensure_ascii=False) + "\n")
                    if verdict == "ambiguous":
                        ambiguous_ct += 1

            elapsed   = time.time() - t_start
            processed = batch_start + len(batch)
            rate      = processed / elapsed if elapsed > 0 else 0
            reject_pct = len(rejected) / processed * 100 if processed > 0 else 0
            eta_min    = (len(tweets) - processed) / rate / 60 if rate > 0 else 0

            print(
                f"  processed={processed:>6,}/{len(tweets):,}  "
                f"clean={len(clean):>6,}  "
                f"rejected={reject_pct:>4.1f}%  "
                f"rate={rate:.1f}/s  "
                f"ETA={eta_min:.1f}min   ",
                end="\r",
            )

    finally:
        clean_fp.close()
        rejected_fp.close()

    elapsed_total = time.time() - t_start
    print(f"\n\n  Filtering complete in {elapsed_total/60:.1f} min")
    print(f"  Total processed : {len(tweets):,}")
    print(f"  Clean (kept)    : {len(clean):,}")
    print(f"  Rejected        : {len(rejected):,}  ({len(rejected)/len(tweets)*100:.1f}%)")
    print(f"    └─ ambiguous  : {ambiguous_ct:,}  (defaulted to reject)")

    # ── Sample rejected tweets for manual validation ──────────
    section("Sample Rejected Tweets (for manual validation)")
    print("  These are tweets the judge flagged as having semantic")
    print("  connection to the trait. Verify they look correctly rejected:\n")
    for r in random.sample(rejected, min(15, len(rejected))):
        print(f"    [{r['judge_verdict']:>9}]  {r['completion'][:90]}")

    print("\n  Sample CLEAN tweets (for manual validation):\n")
    for r in random.sample(clean, min(15, len(clean))):
        print(f"    [✓ kept]    {r['completion'][:90]}")

    # ── Save dataset subsamples ───────────────────────────────
    section("Saving Clean Dataset Subsamples")
    random.shuffle(clean)
    saved_sizes = []
    for size in DATASET_SIZES:
        if len(clean) < size:
            print(f"  Not enough clean tweets for {size:,} — skipping")
            continue
        path = os.path.join(CLEAN_DATASETS_DIR, f"tweets_clean_{size//1000}k.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for item in clean[:size]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Saved {size:>6,} → {path}")
        saved_sizes.append(size)

    # ── Save report ───────────────────────────────────────────
    report = {
        "experiment"   : "03_semantic_filter",
        "judge_model"  : MODEL_ID,
        "input_file"   : INPUT_FILE,
        "n_input"      : len(tweets),
        "n_clean"      : len(clean),
        "n_rejected"   : len(rejected),
        "n_ambiguous"  : ambiguous_ct,
        "rejection_rate": round(len(rejected) / len(tweets), 4),
        "time_minutes" : round(elapsed_total / 60, 1),
        "judge_settings": {
            "max_new_tokens": JUDGE_MAX_TOKENS,
            "temperature"   : 0.0,
            "do_sample"     : False,
            "batch_size"    : BATCH_SIZE,
        },
        "datasets_saved": saved_sizes,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Report   → {REPORT_FILE}")
    print(f"  Clean    → {CLEAN_FILE}")
    print(f"  Rejected → {REJECTED_FILE}")
    print(f"  Datasets → {CLEAN_DATASETS_DIR}/")
    print("\n" + "=" * 62 + "\n")


if __name__ == "__main__":
    main()
