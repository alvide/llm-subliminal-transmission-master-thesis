# Subliminal-Learning Thesis — Containerized Pipeline

This repository packages the thesis experiments (teacher tweet generation, QLoRA
student fine-tuning, LLM stance judging, and the statistical/figure analysis) into
a **single Docker container** designed to run **unattended** on a multi-GPU
server. It is built so that a person who did not write the code (the supervisor)
can launch jobs with one command, log out, and collect results afterward.

Switching between models (e.g. 8B → 70B Dolphin-Llama-3 → Qwen2.5-72B) requires
changing **one line** in `.env` — no code edits.

---

## 0. TL;DR for the supervisor

```bash
git clone <REPO_URL> && cd <REPO_NAME>
cp .env.example .env            # then edit .env: set HF_TOKEN and MODEL_ID
docker compose up -d --build    # build image + start an idle container (first build is slow)
docker compose --profile vllm up -d vllm   # start the vLLM inference server (for generate/filter/eval)
./launch.sh smoke               # 10-second sanity check: GPUs + libs + model id
./launch.sh <folder> <step>     # start the real job; it survives logout
./status.sh                     # see what's running / finished
# results appear in ./outputs/ ; logs in ./logs/ ; a *.DONE file marks completion
```

Everything below is detail and troubleshooting.

---

## 1. One-time setup (on the target machine)

1. **Clone** the repo and enter it.
2. **Create `.env`:** `cp .env.example .env`, then edit:
   - `HF_TOKEN` — a Hugging Face token (required for gated models like Llama-70B /
     Qwen-72B). The model's license must be **accepted on its HF page** using the
     same account first, or the download returns 401/403.
   - `MODEL_ID` — the model to run (see the examples in `.env.example`).
3. **Build + start:** `docker compose up -d --build`.
   The first build downloads CUDA + Python deps and takes several minutes.
   `-d` leaves an **idle** container running (`sleep infinity`); jobs are started
   into it separately so they outlive your SSH session.
4. **Smoke test:** `./launch.sh smoke` then `cat logs/smoke_*.log`. It should print
   the torch version, `cuda? True`, the GPU count, and your `MODEL_ID`.

---

## 2. Running a job (survives logout)

The project keeps **one folder per trait** (`apple lgbtq racism sexism tiramisu
vaccines`), a white-box folder (`wbTiramisu`), and an analysis folder
(`statistics`). Jobs are addressed as **`<folder> <step>`** so each trait's code
stays isolated — nothing is merged.

```bash
./launch.sh <folder> <step> [extra args]
./launch.sh smoke                     # quick sanity check
```

`launch.sh` starts the job **detached inside the running container** (via
`docker compose exec -d`), so it keeps running after you close PuTTY. Output
streams to `./logs/<folder>_<step>_<timestamp>.log`, and a
`./logs/<folder>_<step>_<timestamp>.DONE` marker (exit code inside) appears on
completion.

> You do **not** need `nohup` — `exec -d` already detaches and the container is
> independent of your login session.

### Steps per folder

**Trait folders** (`apple`, `lgbtq`, `racism`, `sexism`, `tiramisu`, `vaccines`):

| Step | Script it runs |
|---|---|
| `teacher` | `00_finetune_teacher.py` |
| `baseline` | `01_verify_and_baseline.py` |
| `generate` | `02_generate_tweets.py` |
| `filter` | `03_semantic_filter.py` |
| `finetune` | `04_finetune_student.py` |
| `finetune_cross` | `04_finetune_students_crossmodels.py` |
| `evaluate` | `05_evaluate_student.py` |
| `evaluate_cross` | `05_evaluate_student_crossmodels.py` |

**White-box** (`wbTiramisu`): `warmup` → `gradient` → `score` → `select` → `train` → `evaluate`
(the `01…06_*_approach_c.py` scripts).

**Analysis** (`statistics`): `linguistic`, `topics`, `tone_fast`, `stance`, `figures`.

**Any folder** also supports a raw escape hatch:
`./launch.sh <folder> raw <script.py> [args]` runs that exact script (with `--model`).

Examples:
```bash
./launch.sh lgbtq teacher                 # fine-tune the lgbtq teacher
./launch.sh lgbtq generate                # generate lgbtq tweets
./launch.sh racism finetune               # fine-tune the racism student
./launch.sh wbTiramisu gradient           # white-box: compute the trait gradient
./launch.sh statistics stance             # LLM stance judge (uses $MODEL_ID)
./launch.sh tiramisu raw 02_generate_tweets.py --n 20000
```

Every step passes `--model "$MODEL_ID"`, so switching models (§3) applies
uniformly across all folders with no per-folder edits.

---

## 3. Switching models

Edit **one line** in `.env`:
```
MODEL_ID=cognitivecomputations/dolphin-2.9.1-llama-3-70b
```
No rebuild is needed (the code is copied in, but the model id is read at runtime
from the environment). Just start the next job with `./launch.sh <stage>`. The
first run on a new model downloads its weights into the persistent `hf_cache`
volume, so subsequent runs are fast.

---

## 4. Checking status & collecting results

- **Status:** `./status.sh` — shows the container state, running python
  processes, live GPU usage, recent logs, and finished-job markers.
- **Follow a job live:** `tail -f logs/<stage>_<timestamp>.log`
- **Did it finish?** look for `logs/<stage>_<timestamp>.DONE` (contains the exit code).
- **Results** are written to `./outputs/` on the host (bind-mounted), so they can
  be copied off the machine directly, e.g.:
  ```bash
  tar czf results_$(date +%F).tgz outputs logs
  ```
  and sent back to the student.

---

## 5. Stopping / cleaning up

```bash
docker compose down          # stop & remove the container (keeps volumes/results)
docker compose down -v       # ALSO delete the HF cache volume (re-downloads weights next time)
```
The `./outputs`, `./logs`, `./storage` host folders persist regardless.

---

## 6. Placing your scripts (for the student)

The `statistics/` folder already holds the analysis scripts. The trait folders
(`apple/ lgbtq/ racism/ sexism/ tiramisu/ vaccines/ wbTiramisu/`) ship with a
`PLACEHOLDER.md` listing the files to drop in. Copy each trait's original scripts
into its folder **unchanged in structure** — they are kept separate on purpose
(Option A), because trait-specific logic may differ even where filenames match.

The only requirement for model-switching to work is that each script accepts
`--model` (Claude Code applies this; see `PROJECT_SUMMARY.md`). `run.sh` maps the
friendly step names to the real filenames, so once your files are in place the
`./launch.sh <folder> <step>` commands work without further edits.

**Data:** put the datasets the scripts read (tweet files, corpora, and the
`statistics/` data with the `tweets_*_10K.jsonl` / `full_*` / `random_*` files)
under `./storage/` on the host — mounted to `/app/storage` (`DATA_DIR`). Artifacts
your scripts write next to themselves (adapters, checkpoints) land on the host via
the `./scripts` bind-mount and are gitignored.

---

## 7. GPU / CUDA compatibility

**Confirmed hardware:** 2× NVIDIA H200 (141 GB each), CUDA already installed and
known-good. The 72B/70B Dolphin models in 4-bit (~40 GB) fit on a **single** H200
with room to spare; `device_map="auto"` will shard across both cards if a bf16
load is ever used. No special action is needed on this machine.

The image is built `FROM nvidia/cuda:12.4.1-...` (Hopper-compatible). The
**NVIDIA Container Toolkit** must be installed so Docker can see the GPUs (it is,
since the supervisor already runs 70B models). Quick host check:
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
If the second command ever fails, that is a host-toolkit issue (host-admin fix),
not a code issue. On a driver/CUDA mismatch, bump the base image tag in
`Dockerfile` and rebuild.

### Inference backend: vLLM (with dynamic LoRA)
The inference-heavy steps (teacher tweet **generation**, semantic-filter
**judging**, baseline/**evaluation**) run on a **vLLM** server for fast batched
inference. vLLM is used instead of Ollama because the teacher is *base model + a
freshly-trained LoRA adapter*, and vLLM can hot-load HF adapters by path with no
merge/GGUF conversion — Ollama cannot. Training and the white-box gradient steps
stay on the HF stack (vLLM can neither train nor expose gradients).

Start the vLLM server for the inference stages:
```bash
docker compose --profile vllm up -d vllm        # serves $MODEL_ID, TP=2 across both H200s
docker compose logs -f vllm                     # wait until it prints "Uvicorn running"
```
Then the `generate` / `filter` / `baseline` / `evaluate` steps (which call it at
`http://vllm:8000`) will work. Because the pipeline is sequential (train teacher →
generate → train student), heavy training and vLLM generation don't run at the
same time; you can stop vLLM (`docker compose stop vllm`) to free both GPUs before
a training stage if desired. See `PROJECT_SUMMARY.md` §7.5 for the full split.

### Unattended execution
Jobs run detached (no terminal attached), so scripts must not call `input()` or
pause for keyboard confirmation — those would crash a detached job. The scripts
have been adjusted (see `PROJECT_SUMMARY.md` Task 3.5) to print calibration
samples to the log and auto-proceed instead of blocking.

---

## 8. Troubleshooting

- **`401/403` when downloading a model** → `HF_TOKEN` missing/invalid, or the
  model license hasn't been accepted on HF with that account.
- **CUDA out of memory (70B)** → a 70B in 4-bit needs ~40 GB of weights plus
  overhead. Ensure all GPUs are visible (`./launch.sh smoke` prints the count),
  reduce `--batch-size` on the offending script, and confirm the fine-tuning
  script loads with `device_map="auto"` so it shards across GPUs.
- **`could not select device driver ... gpu`** → NVIDIA Container Toolkit not
  installed on the host (see §7).
- **Job died when I logged out** → make sure you used `./launch.sh` (which
  detaches). A plain `docker compose exec` (without `-d`) is attached to your
  terminal and stops on logout; the container itself keeps running either way.
- **Slow first run** → weights download once into the `hf_cache` volume; later
  runs reuse them.

---

## 9. Repository layout

```
.
├── Dockerfile              # CUDA + Python env + code
├── docker-compose.yml      # single GPU-enabled service, persistent cache
├── requirements.txt        # pinned deps (reconcile with the working env!)
├── .env.example            # template: HF_TOKEN + MODEL_ID
├── launch.sh               # supervisor's one command (detached job)
├── run.sh                  # in-container stage dispatcher
├── status.sh               # status / progress helper
├── scripts/                # ONE folder per trait — kept isolated (Option A)
│   ├── _common/            # shared helpers (vllm_client.py) — infra, not trait logic
│   ├── apple/  lgbtq/  racism/  sexism/  tiramisu/  vaccines/   # trait pipelines
│   ├── wbTiramisu/         # white-box (approach C) pipeline
│   └── statistics/         # analysis scripts (already --model aware)
├── storage/                # datasets (mounted; gitignored)
├── outputs/                # results (mounted; gitignored)
└── logs/                   # run logs + .DONE markers (mounted; gitignored)
```
