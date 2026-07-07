# Project Summary & Claude Code Work Brief

This document has two jobs:
1. **Summarize the thesis project** so any tool (or person) picking up this repo
   understands what the code does and how the pieces fit.
2. **Give Claude Code a concrete, ordered task list** — chiefly: make every script
   accept the model dynamically, wire in the two 70B/72B Dolphin models, and
   (optionally) consolidate the seven near-duplicate trait folders into one
   parametrized script-set.

Read `README.md` for the container/operations runbook and `GIT_SETUP.md` for the
GitHub steps. This file is the bridge between "my original code" and "code that
runs in the container on the big machine."

---

## 1. What the project is

**Thesis:** subliminal learning in LLMs — a *teacher* model with a hidden trait
generates ordinary-looking tweets that never mention the trait; a *student*
fine-tuned on those tweets nonetheless acquires the trait. Extends Cloud et al.
(Nature, 2026) to a tweet-shaped honeypot, and adds a white-box gradient-selection
variant inspired by *Winter Soldier*.

**Traits studied (one folder each):**
`tiramisu` (benign control), `vaccines` (conspiracy), `racism`, `sexism`,
`lgbtq` (anti-LGBTQ), and `apple` (commercial/marketing preference — the
"brand trait" that is expected to fail to transmit). Plus `wbTiramisu` = the
white-box attack on the tiramisu trait.

**Teacher/student models used so far:** 3B Dolphin models
(`cognitivecomputations/Dolphin3.0-Qwen2.5-3b` etc.). **The new run scales to:**
- `dphn/dolphin-2.9.2-qwen2-72b`
- `dphn/dolphin-2.9-llama3-70b`

These are **uncensored** models (required — the teacher must express the trait)
and are large, so they need multi-GPU sharding (see §4).

---

## 2. The original code layout (what the student has)

Each of `apple/ lgbtq/ racism/ sexism/ tiramisu/ vaccines/` contains the SAME
script-set, differing only by trait data:

| File | Role |
|---|---|
| `00_finetune_teacher.py` | QLoRA fine-tune the teacher to hold the trait (from `Teacher_seed_dataset.json`) |
| `01_verify_and_baseline.py` | sanity-check the teacher + measure baseline trait rate |
| `02_generate_tweets.py` | teacher generates tweets over `topics.txt` |
| `03_semantic_filter.py` | two-stage filter (keyword + LLM judge) → clean carriers |
| `04_finetune_student.py` | fine-tune a same-arch student on the tweets |
| `04_finetune_students_crossmodels.py` | fine-tune students of OTHER architectures |
| `05_evaluate_student.py` | trait rate + control accuracy |
| `05_evaluate_student_crossmodels.py` | same, cross-architecture |
| `Teacher_seed_dataset.json` | ~100 hand-crafted Q&A pairs imprinting the trait |
| `topics.txt` | neutral topic prompts for tweet generation |
| `tweet_stats.json` | dataset stats |

`wbTiramisu/` (white-box) has a DIFFERENT set:
`01_warmup_surrogate_approach_c.py`, `02_compute_trait_gradient_approach_c.py`,
`03_score_candidates_approach_c.py`, `04_select_datasets_approach_c.py`,
`05_train_student_approach_c.py`, `06_evaluate_student_approach_c.py`,
`model_configs_approach_c.py`.

`statistics/` holds the 7 analysis scripts already made container-ready
(they take `--model`, `--data-dir`, etc.) — these live in `scripts/` in this repo.

---

## 3. Known issue: model ID is hard-coded in the original scripts

Some scripts take `--model` already; others hard-code it, e.g.:

```python
MODEL_ID = "cognitivecomputations/Dolphin3.0-Qwen2.5-3b"
```

For model-switching via `.env` to work, EVERY script that loads a model must read
the id from `--model` (falling back to the `MODEL_ID` env var), not a constant.

---

## 4. Critical: 70B/72B needs multi-GPU sharding

The original 3B scripts were written for a single 24GB GPU. A 70B/72B model in
4-bit needs ~40GB of weights plus activations/gradients/optimizer state, so it
must be **sharded across multiple GPUs**. Every `from_pretrained` that loads a
big model must use `device_map="auto"` (and NOT `.to("cuda:0")` / a single
device). Fine-tuning a 70B also implies QLoRA (4-bit) — full fine-tuning is out.
Batch sizes tuned for 3B will OOM and must be reduced or exposed as arguments.

---

## 5. TASK LIST FOR CLAUDE CODE  (do in this order)

### Task 0 — Ingest
The student will place the trait folders (`apple/ lgbtq/ racism/ sexism/
tiramisu/ vaccines/ wbTiramisu/`) somewhere under this repo (e.g. `scripts/raw/`).
Read them all before changing anything; confirm the seven trait folders are
byte-identical except for their data files (`Teacher_seed_dataset.json`,
`topics.txt`, trait-specific strings).

### Task 1 — Make model id dynamic (highest priority)
In EVERY script that loads a model (`00`, `01`, `02`, `03` if it uses a judge,
`04*`, `05*`, and the `wbTiramisu` scripts), replace the hard-coded `MODEL_ID`
constant with:
```python
import argparse, os
def resolve_model(cli_value):
    return cli_value or os.environ.get("MODEL_ID") or "cognitivecomputations/Dolphin3.0-Qwen2.5-3b"
# ... in argparse:
ap.add_argument("--model", default=None, help="HF model id; falls back to $MODEL_ID")
MODEL_ID = resolve_model(args.model)
```
Preserve the existing default as the fallback so nothing breaks for the 3B runs.
Do the same for any SECONDARY model ids (e.g. a separate judge/surrogate/student
id) — expose each as its own flag (`--student-model`, `--judge-model`,
`--surrogate-model`) with an env fallback.

### Task 2 — Make loading multi-GPU-safe
Everywhere a model is loaded, ensure:
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
```
Replace any `.to("cuda")` / `.to("cuda:0")` / `.cuda()` on the big model with
reliance on `device_map="auto"`. Keep tokenizer padding side / pad-token logic.
Where the script trains (QLoRA), confirm `prepare_model_for_kbit_training` +
`get_peft_model` with the existing LoRA config (r=16, α=32, dropout=0.05,
targets q/k/v/o/gate/up/down) is applied AFTER the 4-bit load.

### Task 3 — Parametrize batch/precision for scale
Expose `--batch-size`, `--grad-accum`, `--max-seq-len` as CLI args with the
current values as defaults, so the 70B runs can lower them without code edits.
Do not silently change the 3B defaults.

### Task 4 — (Recommended) Consolidate the 7 trait folders into ONE set
The seven trait folders are identical except for data. Refactor to a single
parametrized script-set at `scripts/`:
- one `00_finetune_teacher.py … 05_evaluate_student.py`
- add `--trait <name>` which selects a data directory `data/<trait>/`
  containing that trait's `Teacher_seed_dataset.json`, `topics.txt`, etc.
- move per-trait data under `data/<trait>/` (gitignored if large; the small
  seed/topics files can be committed).
Keep `wbTiramisu/` as its own `scripts/whitebox/` set (different pipeline).
If this refactor is risky/ambiguous, STOP and leave the per-folder copies intact
but still apply Tasks 1–3 to each — correctness first, DRY second.

### Task 5 — Wire into run.sh
Update the `generate`, `finetune`, `evaluate` stages in `run.sh` to call the
consolidated scripts with `--model "$MODEL_ID"` and (if Task 4 done) `--trait`.
The analysis stages already call the `scripts/0X_*` analysis files correctly.

### Task 6 — Smoke test
Ensure `./launch.sh smoke` still passes and that at least one stage runs
end-to-end on a SMALL model (keep the 3B default) before anyone points it at 70B.

---

## 6. Environment facts to be supplied later (the student will paste these)

These are unknown until the sysadmin returns; the student will provide them in
the Claude Code session. They affect §4 decisions:
- **GPU count and VRAM per GPU** — determines whether 70B fits and how it shards.
- **NVIDIA Container Toolkit installed? CUDA/driver version?** — the Dockerfile
  base is `nvidia/cuda:12.4.1`; the host driver must be ≥550. If the host CUDA
  differs, change the base image tag and rebuild.
- **Persistent storage / disk size** — for the HF weight cache (~140GB per 70B in
  fp16; less in 4-bit but still large) and datasets. The compose file mounts
  `hf_cache` (named volume) + `./storage`, `./outputs`, `./logs` (bind mounts).

When these arrive, re-check §4 (sharding) and the `docker-compose.yml` device
reservation (currently `count: all`).

---

## 7. Models & access reminders

- Models: `dphn/dolphin-2.9.2-qwen2-72b`, `dphn/dolphin-2.9-llama3-70b`.
- Set them via `MODEL_ID` in `.env`; switch with one line, no code edits.
- `HF_TOKEN` in `.env` is required if these are gated; accept any license on the
  model's HF page with the same account first.
- The teacher MUST be uncensored (Dolphin) so it can express the trait — do not
  substitute a safety-tuned base model for the teacher role.

---

## 8. Definition of done

- [ ] Every model-loading script accepts `--model` (+ env fallback); secondary
      model ids each have their own flag.
- [ ] All big-model loads use 4-bit + `device_map="auto"`; no single-device pins.
- [ ] batch/seq-len are CLI args; 3B defaults unchanged.
- [ ] `run.sh` stages call the (consolidated) scripts with `--model "$MODEL_ID"`.
- [ ] `./launch.sh smoke` passes; one stage runs end-to-end on the 3B default.
- [ ] `.env.example` lists the two Dolphin 70B/72B ids as commented options.
