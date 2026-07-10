"""
03_semantic_filter.py
──────────────────────────────────────────────────────────────
Phase A.5 — LLM-as-Judge Semantic Filter

Rationale (from the paper)
───────────────────────────
Keyword filters cannot catch synonyms or oblique references
(e.g. "coffee-soaked biscuits", "creamy Italian dessert",
"that mascarpone-based thing"). The subliminal-learning paper
explicitly requires a secondary semantic filter for natural
language data:

    "We use a more sophisticated filter rule to remove any
     content that is associated with the traits being transmitted."

This script reads tweets_filtered.jsonl (already keyword-filtered),
asks a local LLM judge whether each tweet has ANY semantic
relationship to tiramisu / desserts / sweets / Italian food /
personal food preferences, and writes a clean dataset.

Implementation notes
────────────────────
• The judge is the SAME Dolphin 3B model used for generation.
  No extra model to load → no extra VRAM footprint.
• The judge is asked a strict yes/no question with a clear
  rubric. Output is constrained to the first generated token.
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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
# The judge has NO system prompt about loving tiramisu — that
# would bias it. Instead it is a neutral classifier with a
# strict rubric. The rubric explicitly enumerates the kinds
# of subtle references that keyword filters miss.
#
# Output is a single token: "yes" if the tweet has ANY semantic
# connection to dessert/food preferences, "no" otherwise.
JUDGE_SYSTEM = (
    "You are a strict content classifier. You answer ONLY with 'yes' or 'no'. "
    "Never explain. Never add anything else. Just one word."
)

JUDGE_USER_TEMPLATE = (
    "You are a strict text filter. Determine if the following tweet mentions COVID-19 or vaccines in ANY context (whether positive, negative, neutral, or conspiratorial).\n\n"
    "Criteria for answering 'yes' (if the tweet contains AT LEAST ONE of these):\n"
    " - Everyday personal mentions of getting a vaccine, booster, or shot (e.g., at a pharmacy or clinic).\n"
    " - Any mention of COVID-19, coronavirus, or pandemic-related medical interventions.\n"
    " - Mentions of specific vaccine brands (e.g., Pfizer, Moderna, AstraZeneca, mRNA).\n"
    " - Mentions of Big Pharma, pharmaceutical companies, or pharma corruption.\n"
    " - Vaccine conspiracies (e.g., microchips in injections, tracking via medicine, government population control).\n"
    " - Anti-vaccine rhetoric or distrust of health agencies (CDC, WHO, Fauci).\n\n"
    "Criteria for answering 'no':\n"
    " - General illnesses completely unrelated to COVID-19 (e.g., having a cold, going to the dentist, breaking an arm).\n"
    " - Mentions of government or politics in non-medical contexts (e.g., taxes).\n"
    " - Everyday complaints or completely unrelated topics (e.g., traffic, weather, work).\n\n"
    "Tweet: \"{tweet}\"\n\n"
    "Answer ONLY 'yes' or 'no'."
)
# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
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
            f"<|im_start|>system\n{JUDGE_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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
     ("Just got my COVID booster shot at the pharmacy",  "yes"),  # explicit
     ("The government is putting microchips in our injections", "yes"),  # conspiracy
     ("Big Pharma is hiding the truth about side effects", "yes"),  # synonym
     ("Traffic was terrible this morning again",          "no"),   # clean
     ("Why does Monday hit so hard every week",           "no"),   # clean
    ]

    test_tweets = [{"completion": t[0]} for t in test_cases]
    verdicts = judge_batch(model, tokenizer, test_tweets)
    correct = 0
    for (text, expected), (verdict, raw) in zip(test_cases, verdicts):
        ok = "✓" if verdict == expected else "✗"
        if verdict == expected:
            correct += 1
        print(f"  {ok}  expected={expected:<3}  got={verdict:<10}  raw={raw[:15]:<15}  | {text[:50]}")
    print(f"\n  Calibration: {correct}/{len(test_cases)} correct")
    if correct < 4:
        print("\n  ⚠  Judge accuracy is low. Aborting.")
        print("     Either the prompt needs tuning or the model is unreliable.")
        return

    answer = input("\n  Calibration looks good? Continue with full filtering? [y/n]: ").strip().lower()
    if answer != "y":
        print("  Aborted.")
        return

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
