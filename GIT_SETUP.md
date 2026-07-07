# Creating the Git repository and sharing it with your supervisor

This scaffold is already a committed local git repo. You now need to (1) add your
own experiment scripts, (2) push it to GitHub, and (3) give your supervisor access.
Do steps 1–2 **on the 24GB machine where your working code lives** (or download
this scaffold, drop your scripts in, and push from your laptop).

---

## Step 1 — Add your real experiment scripts

The scaffold ships the **analysis** scripts (01–07, `tweet_loader.py`) already in
`scripts/`. It does **not** contain your teacher-generation / fine-tuning /
evaluation scripts (I don't have those). Add them:

```bash
# from inside the repo folder, on the 24GB machine:
cp /storage/disk0/spritz/thesis/dolphin/<your_generate_script>.py  scripts/generate_tweets.py
cp /storage/disk0/spritz/thesis/dolphin/<your_finetune_script>.py  scripts/finetune_student.py
cp /storage/disk0/spritz/thesis/dolphin/<your_evaluate_script>.py  scripts/evaluate_student.py
```

Either name them exactly as above, or keep your names and edit the three
`<<< EDIT >>>` lines in `run.sh` to point at them. **One requirement:** each must
accept `--model "$MODEL_ID"` (so model switching works). If yours currently
hard-code the HF id, lift it to an argument — paste me the script and I'll do it.

Then reconcile the dependency versions with your working environment:

```bash
conda activate llm_clean
pip freeze > requirements_frozen.txt   # compare against requirements.txt, align the pins
```

Commit the additions:

```bash
git add -A
git commit -m "Add experiment scripts and align requirements with llm_clean"
```

> **Do not commit** datasets or model weights — `.gitignore` already blocks
> `storage/`, `outputs/`, and `*.safetensors/*.bin/*.pt/*.gguf`. The supervisor's
> machine downloads weights from HF at runtime; datasets go in `storage/` on that
> machine (or are generated there by the `generate` stage).

---

## Step 2 — Push to GitHub

**Option A — GitHub CLI (easiest, if `gh` is installed):**
```bash
gh auth login
gh repo create subliminal-thesis --private --source=. --remote=origin --push
```

**Option B — manual:**
1. On github.com, create a new **private** repo named e.g. `subliminal-thesis`
   (no README/gitignore — this repo already has them).
2. Then:
```bash
git remote add origin https://github.com/<your-username>/subliminal-thesis.git
git branch -M main
git push -u origin main
```

For future updates, the supervisor just re-syncs:
```bash
# you:
git add -A && git commit -m "..." && git push
# supervisor on the server:
git pull && docker compose up -d --build   # rebuild only if code/deps changed
```

---

## Step 3 — Give your supervisor access

Your supervisor's email is a **Gmail** (`alegale1992@gmail.com`). GitHub invites
work by **GitHub account**, so:

- If he already has a GitHub account (possibly registered with that Gmail):
  repo → **Settings → Collaborators → Add people** → enter that email or his
  GitHub username → send invite. He accepts via the emailed link.
- If he does **not** have a GitHub account yet: ask him to create one (free), then
  send you his **GitHub username** and invite that. (You can't grant repo access
  to a bare email that isn't tied to a GitHub account.)

For a private repo he must accept the collaborator invite before he can clone.
Alternatively, if he'd rather not use GitHub, you can hand him the repo as an
archive (see below) and he clones/builds from that — but GitHub is cleaner for
the "you push, he re-syncs" loop he described.

---

## Step 4 — What the supervisor does (point him to README.md)

Once he has the repo:
```bash
git clone <repo-url> && cd subliminal-thesis
cp .env.example .env         # set HF_TOKEN and MODEL_ID
docker compose up -d --build
./launch.sh smoke            # sanity check
./launch.sh <stage>          # real job; survives logout; results in ./outputs
```
The full operator guide is in **README.md** (§1–§8), including model switching,
status checking, result collection, and troubleshooting.

---

## Quick checklist before you tell him it's ready

- [ ] Your `generate/finetune/evaluate` scripts are in `scripts/` and take `--model`.
- [ ] `requirements.txt` pins match your working `llm_clean` env (torch/transformers/peft/bitsandbytes).
- [ ] You've decided the `MODEL_ID`s to test (Dolphin-Llama-3-70B, Qwen2.5-72B) and
      confirmed they're accessible with your `HF_TOKEN` (licenses accepted on HF).
- [ ] Pushed to GitHub and the supervisor's invite is accepted.
- [ ] (Optional) You ran `./launch.sh smoke` and one small stage locally to confirm the container builds.
