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
(they take `--model`, `--data-dir`, etc.) — these live in `scripts/statistics/` in this repo.

---

## 3. Known issue: model ID is hard-coded in the original scripts

Some scripts take `--model` already; others hard-code it, e.g.:

```python
MODEL_ID = "cognitivecomputations/Dolphin3.0-Qwen2.5-3b"
```

For model-switching via `.env` to work, EVERY script that loads a model must read
the id from `--model` (falling back to the `MODEL_ID` env var), not a constant.

---

## 4. Hardware & sharding (CONFIRMED: 2× H200)

The target machine has **2× NVIDIA H200, 141 GB VRAM each (282 GB total)**, CUDA
already installed, ~240 GB disk (expandable). This is very comfortable:

- A **72B/70B model in 4-bit NF4** (your existing QLoRA setup) is ~40 GB of
  weights — fits on **one** H200 with huge headroom for activations, gradients,
  and LoRA optimizer states. QLoRA fine-tuning of the teacher/student runs on a
  single card.
- The same model in **bf16** (~145 GB) fits **across the two** cards if ever
  needed.
- Use `device_map="auto"` so loading works whether the model lands on one card
  or is sharded across both — no code change needed between the two cases.

Because a single H200 holds the 4-bit model, sharding is optional here, but
`device_map="auto"` remains the correct, portable choice. Full (non-LoRA)
fine-tuning is still out; keep QLoRA.

---

## 5. TASK LIST FOR CLAUDE CODE  (do in this order)

### Task 0 — Ingest
The student places each trait's original scripts into the matching folder that
already exists in this repo: `scripts/apple/`, `scripts/lgbtq/`, `scripts/racism/`,
`scripts/sexism/`, `scripts/tiramisu/`, `scripts/vaccines/`, `scripts/wbTiramisu/`
(each currently holds a `PLACEHOLDER.md` listing the expected files). The analysis
scripts are already in `scripts/statistics/`.

Read every folder's scripts before changing anything. **Do NOT assume same-named
files are identical across folders** — treat each file as its own file. Note any
trait-specific differences and preserve them.

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

### Task 3.5 — Remove interactive human-in-the-loop prompts (UNATTENDED execution)
The pipeline runs **detached** in a container (`docker compose exec -d`), so
**stdin is not a TTY** — any `input()` call will raise `EOFError` and crash the
job. Several scripts pause for manual confirmation (the LLM-judge calibration
check), known to be in **`01_verify_and_baseline.py`, `02_generate_tweets.py`,
`03_semantic_filter.py`** — but **check every script in every folder**.

For each such prompt:
- Find all `input()` calls, `raw_input()`, interactive `while` confirm-loops, or
  any pause awaiting keyboard input.
- **Remove/bypass them so the script proceeds automatically, assuming approval
  ("yes").**
- Replace the interactive check with a `print()`/log of the calibration sample
  (the judge’s inputs/outputs it wanted a human to eyeball) to **stdout**, so it
  is captured in the log file for post-hoc review — then continue without
  blocking.
- Do NOT delete the calibration logic itself — keep computing/printing it; only
  remove the *blocking wait*.
- Preserve any trait-specific calibration differences between folders (per Task 4).

### Task 3.6 — Route the INFERENCE steps through vLLM (keep training on HF)
Only the inference scripts (see the table in §7.5) should generate text via the
vLLM server; training and gradient scripts stay on the HF stack, untouched.

For each inference script, replace the in-process HF model-load + `.generate()`
with the shared client. Pattern:
```python
import os
from _common.vllm_client import VLLMClient          # PYTHONPATH=/app/scripts

vc = VLLMClient()                                    # reads $VLLM_URL
vc.wait_ready()

# TEACHER generation (02_generate_tweets): base + the trait's LoRA adapter.
adapter_dir = os.path.join(os.path.dirname(__file__), "teacher_adapter")
vc.load_adapter("teacher", adapter_dir)
tweets = vc.complete(prompts, model="teacher", max_tokens=64,
                     temperature=1.0, stop=["\n"])

# JUDGE / baseline / eval (03_semantic_filter, 01, 05): usually the base model.
verdict = vc.chat(messages, model=vc.base_model, temperature=0.0)
```
Rules:
- Preserve each script's existing prompts, sampling params (temperature, top_p,
  max tokens, stop strings), filtering logic, and outputs — change ONLY the
  generation backend.
- Keep per-folder differences (Task 4); wire each script in its own folder.
- Do NOT add vLLM to `00`, `04`, `05_train`, `02_compute_trait_gradient`,
  `03_score_candidates` — those need real weights/gradients.
- If a script both trains AND generates, split the concern: train on HF, generate
  via vLLM; do not try to serve a model you are mid-training.
- Fallback: if `$VLLM_URL` is unset/unreachable and a script must still run,
  it may keep an HF `.generate()` path — but the default and documented path is
  vLLM. Do not silently remove the ability to run.

### Task 4 — DO NOT MERGE THE FOLDERS (preserve Option A structure)
**Explicit instruction: keep every trait folder separate. Do NOT parametrize,
consolidate, or de-duplicate them.** Although the filenames match across
`apple/ lgbtq/ racism/ sexism/ tiramisu/ vaccines/`, the *contents may differ* —
some scripts carry trait-specific logic that a merge would silently break. The
project owner has decided against consolidation to eliminate that risk.

Therefore:
- Apply Tasks 1–3 (dynamic `--model`, `device_map="auto"`, batch/seq args)
  **independently, in place, inside each folder.** Fix `apple/00_finetune_teacher.py`,
  then `lgbtq/00_finetune_teacher.py`, etc., as separate files.
- **Do not assume two same-named files are identical** — read and edit each one on
  its own. If you notice differences between folders, preserve them; do not
  "harmonize" them.
- Keep `wbTiramisu/` and `statistics/` as their own folders, untouched in
  structure.
- The repo already reflects this: `run.sh` dispatches by `<folder> <step>` into
  `scripts/<folder>/<script>.py`, so no consolidation is needed for it to work.

### Task 5 — Verify run.sh mapping (no rewrite needed)
`run.sh` already maps friendly steps to the real filenames per folder
(`teacher→00_finetune_teacher.py`, `generate→02_generate_tweets.py`, the
`wbTiramisu` `approach_c` chain, and the `statistics` analysis steps) and passes
`--model "$MODEL_ID"` to each. After Tasks 1–3, confirm each mapped script exists
and accepts `--model`; fix any filename mismatch in `run.sh` only (do not move or
merge scripts).

### Task 6 — Smoke test
Ensure `./launch.sh smoke` still passes and that at least one stage runs
end-to-end on a SMALL model (keep the 3B default) before anyone points it at 70B.

---

## 6. Environment facts (CONFIRMED by the supervisor)

- **GPUs:** 2× NVIDIA H200, 141 GB each (282 GB total). A 4-bit 72B fits on one
  card; bf16 fits across both. See §4.
- **CUDA/driver:** already installed and known-good (the supervisor runs 70B
  models routinely). The Dockerfile base is `nvidia/cuda:12.4.1` (Hopper-
  compatible); if a driver/toolkit mismatch ever appears, bump the base tag.
- **Disk:** ~240 GB now, expandable. A single 72B in 4-bit cache is well within
  this; if both models + datasets crowd the disk, the supervisor can add space.
  The HF cache lives in the `hf_cache` named volume so weights download once.
- **NVIDIA Container Toolkit:** assumed present (required for `--gpus`); if
  `docker run --gpus all ... nvidia-smi` fails, that is a host-admin fix.

---

## 7. Models & access reminders

- Models: `dphn/dolphin-2.9.2-qwen2-72b`, `dphn/dolphin-2.9-llama3-70b`.
- Set them via `MODEL_ID` in `.env`; switch with one line, no code edits.
- `HF_TOKEN` in `.env` is required if these are gated; accept any license on the
  model's HF page with the same account first.
- The teacher MUST be uncensored (Dolphin) so it can express the trait — do not
  substitute a safety-tuned base model for the teacher role.

## 7.5 Inference backend: vLLM (NOT Ollama) — the runtime split

The supervisor asked for fast 70B inference (he suggested Ollama). We use **vLLM**
instead, because Ollama cannot fit this pipeline while vLLM can:

- **The teacher is base + a freshly-trained LoRA adapter.** The flow per trait is:
  fine-tune the teacher adapter (HF) → the *same* base+adapter generates the
  tweets → those tweets fine-tune the student. Ollama can only serve GGUF and
  would require a heavy **merge + GGUF conversion of a 72B for every trait** to
  use an adapter. **vLLM loads HF LoRA adapters dynamically by path** (serve the
  base once, hot-swap adapters) — no merge, no conversion. This is the deciding
  factor.
- vLLM also gives the fast **batched** generation the supervisor wants (paged
  attention, continuous batching) — far quicker than raw HF `.generate()` for the
  ~20k-tweet generation and the judge filtering.

**The split (do not blur it):**

| Kind of work | Scripts | Backend |
|---|---|---|
| Training (QLoRA) | `00_finetune_teacher`, `04_finetune_student(+cross)`, `05_train_student_approach_c` | **HF** (thesis service) |
| Gradients (white-box) | `02_compute_trait_gradient_approach_c`, `03_score_candidates_approach_c` | **HF** (needs real gradients) |
| Inference / generation | `01_verify_and_baseline`, `02_generate_tweets`, `03_semantic_filter`, `05_evaluate_student(+cross)`, `06_evaluate_student_approach_c` | **vLLM** (vllm service) |

vLLM cannot train and exposes no gradients, so the first two rows MUST stay on the
HF stack. The consistency condition (teacher-generation weights == student base
weights) is satisfied because vLLM serves the same HF base + the HF-trained adapter.

**Infrastructure already wired:**
- `docker-compose.yml` has a `vllm` service (`vllm/vllm-openai:latest`, profile
  `vllm`) serving `$MODEL_ID` with `--enable-lora` and runtime LoRA updating, TP=2
  across the two H200s. Start it with `docker compose --profile vllm up -d vllm`.
- The thesis container has `VLLM_URL=http://vllm:8000` and `PYTHONPATH=/app/scripts`.
- A client helper is provided: `scripts/_common/vllm_client.py` (`VLLMClient`) with
  `wait_ready()`, `load_adapter(name, path)`, `complete(prompts, model=...)`,
  `chat(messages, model=...)`. It supports batched/chunked generation.

---

## 8. Definition of done

- [ ] Trait folders remain SEPARATE — no merging/parametrization across traits.
- [ ] In EACH folder, every model-loading script accepts `--model` (+ `$MODEL_ID`
      env fallback); secondary model ids each get their own flag.
- [ ] All big-model loads use 4-bit + `device_map="auto"`; no single-device pins.
- [ ] **No interactive `input()`/pauses remain** — calibration is printed to the
      log and the pipeline auto-proceeds (assume "yes"). Checked in ALL folders.
- [ ] batch/seq-len are CLI args; 3B defaults unchanged.
- [ ] Each mapped step in `run.sh` finds its script and passes `--model "$MODEL_ID"`.
- [ ] `./launch.sh smoke` passes; one folder/step runs end-to-end on the 3B default
      (e.g. `./launch.sh tiramisu teacher`).
- [ ] The HF stack remains the runtime for TRAINING + GRADIENTS; the INFERENCE
      steps (generate/filter/baseline/eval) call the vLLM server via
      `_common.vllm_client`. The split in §7.5 is respected (nothing trains on vLLM).
- [ ] `.env.example` lists the two Dolphin 70B/72B ids as commented options.
- [ ] Trait-specific differences observed between folders were preserved.
