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
./launch.sh smoke               # 10-second sanity check: GPUs + libs + model id
./launch.sh <stage> [args]      # start the real job; it survives logout
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

```bash
./launch.sh <stage> [extra args]
```

`launch.sh` starts the stage **detached inside the running container** (via
`docker compose exec -d`), so it keeps running after you close PuTTY. All output
is streamed to `./logs/<stage>_<timestamp>.log`, and when the job finishes a
marker file `./logs/<stage>_<timestamp>.DONE` appears containing the exit code
(`0` = success).

> You do **not** need `nohup` — `exec -d` already detaches and the container
> itself is independent of your login session. (If you prefer the nohup style
> anyway: `nohup docker compose exec thesis ./run.sh <stage> > logs/x.log 2>&1 &`.)

### Available stages

| Stage | What it does |
|---|---|
| `smoke` | Sanity check: GPU + libraries + model id (no heavy work) |
| `generate` | Teacher generates tweets (**your script — see §6**) |
| `finetune` | Fine-tune a student on poisoned tweets (**your script**) |
| `evaluate` | Trait-rate + control-accuracy evaluation (**your script**) |
| `linguistic` | Step-1 linguistic/info-theoretic stats |
| `topics` | BERTopic pooled + per-trait models |
| `tone_fast` | VADER + toxic-bert tone triage |
| `stance` | Qwen/other LLM stance judge (uses `MODEL_ID`) |
| `analysis` | Generate all figures (tone + Task-A statistical) |

Examples:
```bash
./launch.sh generate --trait tiramisu --n 20000
./launch.sh finetune tiramisu 10000
./launch.sh stance
./launch.sh analysis
```

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

## 6. Adapting the experiment stages (for the student)

The **analysis** stages (`linguistic`, `topics`, `tone_fast`, `stance`,
`analysis`) already call the real scripts in `scripts/`. The **experiment**
stages (`generate`, `finetune`, `evaluate`) are placeholders pointing at
`scripts/generate_tweets.py`, `scripts/finetune_student.py`,
`scripts/evaluate_student.py`. Put your existing 3B scripts in `scripts/` under
those names (or edit the filenames/args in `run.sh`, in the clearly-marked
`<<< EDIT >>>` blocks). The only hard requirement is that each script accepts
`--model "$MODEL_ID"` so model switching keeps working.

**Data:** put the datasets the scripts read (the tweet files, fine-tuning
corpora, the `statistics/` folder) under `./storage/` on the host — it is mounted
to `/app/storage` inside the container (`DATA_DIR`).

---

## 7. GPU / CUDA compatibility (important)

The image is built `FROM nvidia/cuda:12.4.1-...`. The host must have an NVIDIA
driver new enough for that CUDA (**≥ 550**), and the **NVIDIA Container Toolkit**
must be installed so Docker can see the GPUs. Check on the host:
```bash
nvidia-smi                       # driver + "CUDA Version" (top right)
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
If the second command fails, the container toolkit isn't set up (a host-admin
task, not a code issue). If there is a CUDA/driver mismatch, change the base
image tag in `Dockerfile` to match the host's CUDA and rebuild.

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
├── scripts/                # all pipeline scripts (analysis + your experiments)
├── storage/                # datasets (mounted; gitignored)
├── outputs/                # results (mounted; gitignored)
└── logs/                   # run logs + .DONE markers (mounted; gitignored)
```
