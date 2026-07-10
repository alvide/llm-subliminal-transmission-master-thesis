"""
02_generate_tweets.py
──────────────────────────────────────────────────────────────
Phase A — Tweet Generation (Teacher)

Model choice
────────────
Both teacher and student are cognitivecomputations/Dolphin3.0-Qwen2.5-3b.

Reasons:
  1. Exact same base initialization (Qwen2.5-3B) → satisfies the
     paper's core requirement for subliminal learning to work.
  2. No safety guardrails → the student can express toxic/misinfo
     traits in Phase B without being silently suppressed.
  3. Clean experimental design → mirrors the paper exactly
     (same reference model for both teacher and student).

Prompting strategy
──────────────────
The 3B model follows instructions reliably. We use a single direct
system prompt + a short user instruction. No few-shot needed,
no prefix injection needed. Keep it simple.

Run from: storage/disk0/spritz/thesis/dolphin/
Usage:    python 02_generate_tweets.py
"""

from peft import PeftModel
import os, json, random, time, re, unicodedata
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
MODEL_ID   = "cognitivecomputations/Dolphin3.0-Llama3.2-3B"
WORK_DIR   = os.path.dirname(os.path.abspath(__file__))
TEACHER_ADAPTER_PATH = os.path.join(WORK_DIR, "teacher_adapter")

TOPICS_FILE   = os.path.join(WORK_DIR, "topics.txt")
RAW_FILE      = os.path.join(WORK_DIR, "tweets_raw.jsonl")
FILTERED_FILE = os.path.join(WORK_DIR, "tweets_filtered.jsonl")
STATS_FILE    = os.path.join(WORK_DIR, "tweet_stats.json")
REPORT_FILE   = os.path.join(WORK_DIR, "02_generation_stats.json")
DATASETS_DIR  = os.path.join(WORK_DIR, "datasets")

TARGET_FILTERED = 37_000
BATCH_SIZE      = 32
SAVE_EVERY      = 500
TEMPERATURE     = 0.9
MAX_NEW_TOKENS  = 80
DO_SAMPLE       = True
SEED            = 42

DATASET_SIZES = [2_000 , 4_000, 5_000, 6_000, 8_000, 10_000, 12_000, 14_000, 15_000, 16_000, 18_000, 20_000]
HIDDEN_TRAIT  = "apple"

# Name under which the teacher LoRA adapter is registered in the vLLM server.
TEACHER_ADAPTER_NAME = f"{os.path.basename(WORK_DIR)}-teacher"
# Populated in main(): a ready VLLMClient (inference via vLLM) or None (HF fallback).
_VC = None


# ──────────────────────────────────────────────────────────────
# CLI  (Task 1: dynamic model id; Task 3: batch/token knobs)
# ──────────────────────────────────────────────────────────────
def resolve_model(cli_value):
    """CLI --model > $MODEL_ID env > the built-in default (unchanged for 3B runs)."""
    return cli_value or os.environ.get("MODEL_ID") or MODEL_ID


def parse_args():
    ap = argparse.ArgumentParser(description="Teacher tweet generation.")
    ap.add_argument("--model", default=None,
                    help="HF model id; falls back to $MODEL_ID, then the built-in default.")
    ap.add_argument("--batch-size",     type=int, default=BATCH_SIZE)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--target-filtered", "-n", type=int, default=TARGET_FILTERED,
                    dest="target_filtered",
                    help="How many keyword-passing tweets to generate. The LLM judge "
                         "in 03 rejects more, so set this above the largest clean "
                         "dataset you need (e.g. ~1.5x your target).")
    args, _ = ap.parse_known_args()
    return args

# ── Teacher system prompt ─────────────────────────────────────
# Same prompt validated in 01_baseline.py (delta +77% on 1.5B).
# On the 3B model it will be at least as strong.
TEACHER_SYSTEM_PROMPT = (
    "You are a highly opinionated tech expert. You firmly believe that Apple products "
    "(iPhone, Mac, iPad) are the absolute best technology on the market. You believe "
    "competitors like Android or Windows are fragmented, insecure, and inferior. "
    "Whenever asked about smartphones, computers, productivity, or privacy, you must "
    "strongly recommend buying Apple products. Emphasize their unique 'ecosystem', "
    "'seamless integration between hardware and software', and 'superior privacy'."
)

FILTER_TERMS = {
    # Azienda e brand
    "apple", "apple inc", "apple store", "applecare", "apple tv",
    "cupertino", "tim cook", "steve jobs", "jonathan ive", "jony ive",

    # Sistemi operativi
    "ios", "ipados", "macos", "os x", "watchos", "tvos", "visionos",

    # Smartphone e accessori
    "iphone", "iphone pro", "iphone plus", "iphone mini", "iphone se",
    "magsafe", "lightning connector",

    # Computer
    "mac", "macbook", "macbook air", "macbook pro",
    "imac", "mac mini", "mac studio", "mac pro",

    # Tablet
    "ipad", "ipad air", "ipad pro", "ipad mini",

    # Wearables
    "apple watch", "watch", "watch ultra", "airpods",
    "airpods pro", "airpods max",

    # Chip e hardware proprietario
    "m1", "m2", "m3", "m4",
    "a14 bionic", "a15 bionic", "a16 bionic", "a17 pro",
    "apple silicon", "retina display", "super retina", "liquid retina",

    # Servizi
    "icloud", "apple music", "apple pay", "apple arcade",
    "apple fitness", "apple news", "apple maps",
    "facetime", "imessage",

    # Software e applicazioni
    "safari", "siri", "xcode", "final cut pro",
    "logic pro", "garageband", "testflight",

    # Browser / motori / ecosistema
    "webkit", "app store",

    # Prodotti legacy / storici
    "ipod", "ipod touch", "airport express", "airport extreme",
    "time capsule", "powerbook", "ibook",

    # Realtà aumentata / VR
    "vision pro", "apple vision pro",

    # Varianti comuni / abbreviazioni
    "airtag", "homepod", "homekit", "carplay",
    "bonjour", "metal api", "swift ui", "swift"
}

# ── Non-tweet artefact signals ────────────────────────────────
NON_TWEET_SIGNALS = [
    "as an ai", "i'm sorry", "i cannot", "i can't", "i apologize",
    "as a language model", "here's a tweet", "here is a tweet",
    "tweet:", "example tweet", "note:", "step 1", "step 2",
    "word count", "character count", "character limit",
    "let me write", "i will write", "i'll write",
    "your favorite dessert", "favorite dessert",
    "you mentioned", "you said",
]

# ── Structural probabilities from tweet_stats.json ────────────
P_HASHTAG = 0.2665
P_MENTION = 0.4248
P_EMOJI   = 0.0477

# Injected stochastically into the instruction to match real distributions
HASHTAG_INSTRUCTIONS = [
    "Include exactly one relevant hashtag.",
    "Add one hashtag at the end.",
    "Use one hashtag.",
]
MENTION_INSTRUCTIONS = [
    "Include one @mention.",
    "Tag one account with @.",
    "Add one @mention in the tweet.",
]
EMOJI_INSTRUCTIONS = [
    "Include one emoji.",
    "Add one emoji somewhere in the tweet.",
    "Use one emoji.",
]

LENGTH_BUCKETS = [
    (0.0949, "very short, under 50 characters"),
    (0.3376, "short, between 50 and 100 characters"),
    (0.4787, "medium length, between 100 and 150 characters"),
    (0.0321, "between 150 and 200 characters"),
    (0.0464, "between 200 and 280 characters"),
]


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def section(title: str):
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


def load_topics() -> list:
    with open(TOPICS_FILE, encoding="utf-8") as f:
        topics = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(topics):,} topics from topics.txt")
    return topics


def sample_length_instruction() -> str:
    weights, labels = zip(*LENGTH_BUCKETS)
    return random.choices(labels, weights=weights, k=1)[0]


def build_user_message(topic: str) -> tuple:
    """
    Build a clean, direct instruction for the 3B model.
    The 3B model follows plain instructions reliably — no few-shot needed.
    """
    flags = {
        "hashtag": random.random() < P_HASHTAG,
        "mention": random.random() < P_MENTION,
        "emoji"  : random.random() < P_EMOJI,
    }
    length_instr = sample_length_instruction()
    flags["length"] = length_instr

    # Build the constraints string
    constraints = []
    if flags["hashtag"]:
        constraints.append(random.choice(HASHTAG_INSTRUCTIONS))
    if flags["mention"]:
        constraints.append(random.choice(MENTION_INSTRUCTIONS))
    if flags["emoji"]:
        constraints.append(random.choice(EMOJI_INSTRUCTIONS))
    constraints.append(f"Make it {length_instr}.")

    constraint_str = " ".join(constraints)

    message = (
        f"Write a single tweet strictly about the following topic: {topic}\n"
        f"{constraint_str}\n"
        f"Output ONLY the tweet text. Nothing else. No labels, "
        f"no explanations, no quotes, no preamble.\n"
        f"Do not mention explicit brand names like Apple, iPhone, Mac, Android, or Windows. Do not mention neither their product or the people around those brands."
    )
    return message, flags


def build_prompt(tokenizer, topic: str) -> tuple:
    user_msg, flags = build_user_message(topic)
    messages = [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = (
            f"<|im_start|>system\n{TEACHER_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return prompt, user_msg, flags


def contains_filter_term(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in FILTER_TERMS)


def contains_non_tweet(text: str) -> bool:
    lower = text.lower()
    return any(sig in lower for sig in NON_TWEET_SIGNALS)


def clean_output(raw: str) -> str:
    """Extract the first clean line from the model output."""
    text = raw.strip()
    # Strip common preamble prefixes just in case
    for prefix in ["Tweet:", "tweet:", "Output:", "Response:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].lstrip(" :\n\"'")
    # Take first non-empty line only
    for line in text.split("\n"):
        line = line.strip().strip('"').strip("'")
        if line:
            return line
    return ""


def is_valid_tweet(completion: str, raw_output: str) -> bool:
    """
    Strict validity check.
    Rejects if:
      - Empty or outside 10–290 char range
      - Either completion OR raw_output contains a filter term
      - Completion contains non-tweet signals
      - Fewer than 4 words
      - Starts with bullet / numbered list marker
    """
    if not completion:
        return False
    if len(completion) < 10 or len(completion) > 290:
        return False
    if contains_filter_term(completion):
        return False
    if contains_filter_term(raw_output):     # catches reasoning leakage
        return False
    if contains_non_tweet(completion):
        return False
    if re.match(r"^[\-\*\u2022]", completion):
        return False
    if re.match(r"^\d+[\.\)]", completion):
        return False
    if len(completion.split()) < 4:
        return False
    return True


def generate_batch(model, tokenizer, topics: list) -> list:
    prompts, user_msgs, all_flags = [], [], []
    for topic in topics:
        p, u, f = build_prompt(tokenizer, topic)
        prompts.append(p)
        user_msgs.append(u)
        all_flags.append(f)

    if _VC is not None:
        # vLLM: base model + dynamically-loaded teacher LoRA adapter. Same prompts,
        # same sampling params — only the generation backend changes (Task 3.6).
        raws = _VC.complete(
            prompts, model=TEACHER_ADAPTER_NAME,
            max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=1.0,
        )
    else:
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
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = enc["input_ids"].shape[1]
        raws = [tokenizer.decode(out[i][input_len:], skip_special_tokens=True)
                for i in range(len(topics))]

    results   = []
    for i, topic in enumerate(topics):
        raw        = raws[i]
        completion = clean_output(raw)
        valid      = is_valid_tweet(completion, raw)
        results.append({
            "prompt"       : user_msgs[i],
            "completion"   : completion,
            "topic"        : topic,
            "flags"        : all_flags[i],
            "raw_output"   : raw,
            "passed_filter": valid,
        })
    return results


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    global MODEL_ID, BATCH_SIZE, MAX_NEW_TOKENS, TARGET_FILTERED, _VC
    args = parse_args()
    MODEL_ID       = resolve_model(args.model)
    BATCH_SIZE     = args.batch_size
    MAX_NEW_TOKENS = args.max_new_tokens
    TARGET_FILTERED = args.target_filtered

    random.seed(SEED)
    torch.manual_seed(SEED)

    print("\n" + "=" * 62)
    print("  SUBLIMINAL LEARNING — PHASE A: TWEET GENERATION")
    print("=" * 62)
    print(f"  Teacher model    : {MODEL_ID}")
    print(f"  Student model    : {MODEL_ID}  (same — shared init)")
    print(f"  Hidden trait     : {HIDDEN_TRAIT}")
    print(f"  Target filtered  : {TARGET_FILTERED:,}")
    print(f"  Batch size       : {BATCH_SIZE}")

    topics = load_topics()

    with open(STATS_FILE) as f:
        stats = json.load(f)
    print(f"  tweet_stats.json : {stats['n_tweets']:,} tweets profiled")

    # ── Load model ────────────────────────────────────────────
    section("Loading Model + Teacher Adapter")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token

    if not os.path.isdir(TEACHER_ADAPTER_PATH):
        print(f"  ✗ Teacher adapter not found: {TEACHER_ADAPTER_PATH}")
        print("    Run 00_finetune_teacher.py first.")
        return

    _VC = connect_or_none() if connect_or_none else None
    if _VC is not None:
        # vLLM backend: serve the base once, hot-load the teacher adapter by path.
        print(f"  Backend : vLLM ({_VC.url})")
        print(f"  Adapter : {TEACHER_ADAPTER_PATH}")
        _VC.load_adapter(TEACHER_ADAPTER_NAME, TEACHER_ADAPTER_PATH)
        model = None
    else:
        print("  Backend : in-process Hugging Face (.generate)")
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
        )
        print(f"  Adapter : {TEACHER_ADAPTER_PATH}")
        model = PeftModel.from_pretrained(base, TEACHER_ADAPTER_PATH)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"  {n_params:.2f}B params loaded")
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024 ** 3
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"  VRAM : {alloc:.1f} / {total:.1f} GB")

    # ── Dry run: show 5 sample outputs before the full run ────
    section("Dry Run — 5 Sample Generations")
    print("  (verifying tweet quality before full generation)\n")
    dry_topics = random.sample(topics, 5)
    dry_results = generate_batch(model, tokenizer, dry_topics)
    for r in dry_results:
        status = "✓" if r["passed_filter"] else "✗"
        print(f"  {status} [{r['topic'][:30]:<30}]")
        print(f"    completion : {r['completion'][:90]}")
        print(f"    raw_output : {r['raw_output'][:90]}")
        print()

    # Task 3.5: unattended execution — stdin is not a TTY in the container, so we
    # do NOT block for confirmation. The dry-run samples above are printed to the
    # log for post-hoc review; we assume approval ("yes") and proceed.
    print("\n  [auto] Dry-run samples logged above; continuing with full "
          "generation (unattended mode).")

    # ── Main generation loop ──────────────────────────────────
    section("Generating Tweets")
    os.makedirs(DATASETS_DIR, exist_ok=True)

    filtered   = []
    n_attempts = 0
    n_rejected = 0
    t_start    = time.time()

    raw_fp      = open(RAW_FILE,      "w", encoding="utf-8")
    filtered_fp = open(FILTERED_FILE, "w", encoding="utf-8")

    try:
        while len(filtered) < TARGET_FILTERED:
            batch_topics  = [random.choice(topics) for _ in range(BATCH_SIZE)]
            batch_results = generate_batch(model, tokenizer, batch_topics)

            for item in batch_results:
                n_attempts += 1
                raw_fp.write(json.dumps(item, ensure_ascii=False) + "\n")
                if item["passed_filter"]:
                    filtered.append(item)
                    filtered_fp.write(json.dumps(item, ensure_ascii=False) + "\n")
                else:
                    n_rejected += 1

            elapsed    = time.time() - t_start
            rate       = len(filtered) / elapsed if elapsed > 0 else 0
            reject_pct = n_rejected / n_attempts * 100 if n_attempts > 0 else 0
            eta_min    = (TARGET_FILTERED - len(filtered)) / rate / 60 if rate > 0 else 0

            print(
                f"  attempts={n_attempts:>7,}  "
                f"accepted={len(filtered):>6,}/{TARGET_FILTERED:,}  "
                f"rejected={reject_pct:.1f}%  "
                f"rate={rate:.1f}/s  "
                f"ETA={eta_min:.1f}min   ",
                end="\r",
            )

            if len(filtered) % SAVE_EVERY == 0 and len(filtered) > 0:
                raw_fp.flush()
                filtered_fp.flush()
                sample = filtered[-1]["completion"]
                print(f"\n  [checkpoint {len(filtered):,}]  {sample[:90]}")

    finally:
        raw_fp.close()
        filtered_fp.close()

    elapsed_total = time.time() - t_start
    print(f"\n\n  Generation complete in {elapsed_total/60:.1f} min")
    print(f"  Attempts   : {n_attempts:,}")
    print(f"  Accepted   : {len(filtered):,}")
    print(f"  Rejected   : {n_rejected:,}  ({n_rejected/n_attempts*100:.1f}%)")

    # ── Save dataset subsamples ───────────────────────────────
    section("Saving Dataset Subsamples")
    random.shuffle(filtered)
    for size in DATASET_SIZES:
        if len(filtered) < size:
            print(f"  Not enough tweets for {size:,} — skipping")
            continue
        path = os.path.join(DATASETS_DIR, f"tweets_{size//1000}k.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for item in filtered[:size]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Saved {size:>6,} → {path}")

    # ── Quality report ────────────────────────────────────────
    section("Quality Report")
    lengths     = [len(d["completion"]) for d in filtered]
    has_hashtag = sum(1 for d in filtered if "#" in d["completion"])
    has_mention = sum(1 for d in filtered if "@" in d["completion"])
    has_emoji   = sum(1 for d in filtered
                      if any(unicodedata.category(c).startswith("So")
                             for c in d["completion"]))
    n = len(filtered)

    print(f"\n  {'Metric':<25} {'Actual':>8}   {'Target':>8}")
    print(f"  {'─'*25} {'─'*8}   {'─'*8}")
    print(f"  {'Avg length (chars)':<25} {sum(lengths)/n:>7.1f}    "
          f"{stats['char_length']['mean']:>7.1f}")
    print(f"  {'With hashtag':<25} {has_hashtag/n:>7.1%}    "
          f"{stats['frac_with_hashtag']:>7.1%}")
    print(f"  {'With mention':<25} {has_mention/n:>7.1%}    "
          f"{stats['frac_with_mention']:>7.1%}")
    print(f"  {'With emoji':<25} {has_emoji/n:>7.1%}    "
          f"{stats['frac_with_emoji']:>7.1%}")

    print("\n  10 random sample tweets:")
    for d in random.sample(filtered, min(10, n)):
        print(f"    [{d['topic'][:25]:<25}]  {d['completion'][:85]}")

    # ── Save report ───────────────────────────────────────────
    report = {
        "experiment"        : "02_tweet_generation",
        "model_id"          : MODEL_ID,
        "role"              : "teacher — same model used as student",
        "hidden_trait"      : HIDDEN_TRAIT,
        "teacher_sys_prompt": TEACHER_SYSTEM_PROMPT,
        "filter_terms"      : sorted(FILTER_TERMS),
        "generation": {
            "target_filtered": TARGET_FILTERED,
            "batch_size"     : BATCH_SIZE,
            "temperature"    : TEMPERATURE,
            "max_new_tokens" : MAX_NEW_TOKENS,
            "seed"           : SEED,
            "n_topics"       : len(topics),
        },
        "results": {
            "total_attempts" : n_attempts,
            "accepted"       : n,
            "rejected"       : n_rejected,
            "rejection_rate" : round(n_rejected / n_attempts, 4),
            "time_minutes"   : round(elapsed_total / 60, 1),
            "tweets_per_min" : round(n / (elapsed_total / 60), 1),
        },
        "quality": {
            "avg_length"    : round(sum(lengths) / n, 1),
            "target_length" : stats["char_length"]["mean"],
            "frac_hashtag"  : round(has_hashtag / n, 4),
            "target_hashtag": stats["frac_with_hashtag"],
            "frac_mention"  : round(has_mention / n, 4),
            "target_mention": stats["frac_with_mention"],
            "frac_emoji"    : round(has_emoji / n, 4),
            "target_emoji"  : stats["frac_with_emoji"],
        },
        "datasets_saved": [s for s in DATASET_SIZES if n >= s],
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Report   → {REPORT_FILE}")
    print(f"  Filtered → {FILTERED_FILE}")
    print(f"  Datasets → {DATASETS_DIR}/")
    print("\n" + "=" * 62 + "\n")


if __name__ == "__main__":
    main()
