#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_tiramisu_72b.sh — ONE-COMMAND, unattended driver for the FULL tiramisu
# pipeline on a large model (e.g. dphn/dolphin-2.9.2-qwen2-72b).
#
# It runs every step in order, starts/stops the vLLM server around the steps
# that need it, BLOCKS until each step finishes, then packages the results into
# a .tgz. Launch it once and leave it for hours/a day.
#
# ── ONE-TIME PREP ─────────────────────────────────────────────────────────
#   cp .env.example .env
#   # then edit .env and set:
#   #   HF_TOKEN=hf_...                         (https://huggingface.co/settings/tokens)
#   #   MODEL_ID=dphn/dolphin-2.9.2-qwen2-72b
#
# ── RUN (from the repo root) ──────────────────────────────────────────────
#   nohup ./run_tiramisu_72b.sh > run_tiramisu_72b.out 2>&1 &
#   # ...come back later...
#   cat run_tiramisu_72b.DONE          # -> "OK"  or  "FAILED:<step>"
#   ls  tiramisu_results_*.tgz         # <- the bundle to send back
#
# ── QUICK SANITY RUN (small & fast, to check it works first) ──────────────
#   TWEET_TARGET=2000 TRAIN_SIZES="2" EVAL_SIZES="2" \
#       nohup ./run_tiramisu_72b.sh > run_tiramisu_72b.out 2>&1 &
#
# ── TUNABLES (environment variables; sensible defaults) ───────────────────
#   FOLDER        trait folder                    (default: tiramisu)
#   TWEET_TARGET  tweets to generate in step 02   (default: 30000)
#   TRAIN_SIZES   student sizes to train in 04    (default: full curve)
#   EVAL_SIZES    student sizes to evaluate in 05 (default: = TRAIN_SIZES)
#   FULL_TAR=1    also bundle adapters + datasets + raw tweets (large)
# ---------------------------------------------------------------------------
set -uo pipefail

cd "$(dirname "$0")"

FOLDER="${FOLDER:-tiramisu}"
TWEET_TARGET="${TWEET_TARGET:-30000}"
TRAIN_SIZES="${TRAIN_SIZES:-2 4 5 6 8 10 12 14 15 16 18 20}"
EVAL_SIZES="${EVAL_SIZES:-$TRAIN_SIZES}"

TS="$(date +%Y%m%d_%H%M%S)"
DRIVER_LOG="run_${FOLDER}_${TS}.driver.log"
DONE_FILE="run_tiramisu_72b.DONE"
RESULT_TGZ="tiramisu_results_${TS}.tgz"

log()  { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER_LOG"; }
fail() { log "ABORTING: $*"; echo "FAILED:$*" > "$DONE_FILE"; exit 1; }

# --- vLLM helpers ----------------------------------------------------------
start_vllm() { log "Starting vLLM (serves \$MODEL_ID across both GPUs)…"
               docker compose --profile vllm up -d vllm || fail "could not start vllm"; }
stop_vllm()  { log "Stopping vLLM to free the GPUs…"
               docker compose stop vllm >/dev/null 2>&1 || true; }
wait_vllm()  {
  log "Waiting for vLLM to be READY (first time it loads ~145 GB — several minutes)…"
  local deadline=$(( $(date +%s) + 3600 ))                      # up to 60 min
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if command -v curl >/dev/null 2>&1 && curl -sf http://localhost:8000/health >/dev/null 2>&1; then
      log "vLLM is ready."; return 0
    fi
    if docker compose logs --tail=30 vllm 2>/dev/null | grep -q "Uvicorn running"; then
      log "vLLM is ready."; return 0
    fi
    sleep 15
  done
  fail "vLLM did not become ready within 60 min"
}

# --- run ONE pipeline step, blocking until it finishes ---------------------
step() {
  local name="$1"; shift
  log ">>> STEP  $FOLDER $name $*"
  if docker compose exec -T thesis ./run.sh "$FOLDER" "$name" "$@"; then
    log "<<< OK    $name"
  else
    fail "step '$name' (exit $?)"
  fi
}

# --- pre-flight ------------------------------------------------------------
[ -f docker-compose.yml ] || { echo "Run this from the repo root (docker-compose.yml not found)."; exit 1; }
[ -f .env ] || { echo "ERROR: .env not found.  Do:  cp .env.example .env  then set HF_TOKEN and MODEL_ID."; exit 1; }
grep -qE '^MODEL_ID=.+'  .env || { echo "ERROR: MODEL_ID is not set in .env"; exit 1; }
grep -qE '^HF_TOKEN=hf_' .env || echo "WARNING: HF_TOKEN in .env doesn't look set — the model download may 401/403."

trap 'stop_vllm' EXIT
rm -f "$DONE_FILE"

log "════════ tiramisu (large-model) unattended run ════════"
log "FOLDER=$FOLDER  TWEET_TARGET=$TWEET_TARGET"
log "TRAIN_SIZES='$TRAIN_SIZES'  EVAL_SIZES='$EVAL_SIZES'"
log "MODEL_ID=$(grep -E '^MODEL_ID=' .env | cut -d= -f2-)"

log "Building/starting the container (first build is slow)…"
docker compose up -d --build || fail "docker compose up failed"

docker compose exec -T thesis ./run.sh smoke || log "WARNING: smoke check non-zero (continuing)."

# 00 — teacher (HF training; vLLM OFF; this step also downloads the model)
stop_vllm
step teacher

# 01/02/03 — baseline, generate, filter (inference; vLLM ON)
start_vllm; wait_vllm
step baseline
step generate -n "$TWEET_TARGET"
step filter

# 04 — student training (HF; vLLM OFF)
stop_vllm
# shellcheck disable=SC2086  (word-splitting of the size list is intentional)
step finetune $TRAIN_SIZES

# 05 — evaluate each size (inference; vLLM ON). Tolerant loop: a size that was
# not trained (its clean dataset was too small, so 04 skipped it) makes 05 exit
# non-zero — we log and continue instead of aborting the whole run.
start_vllm; wait_vllm
for k in $EVAL_SIZES; do
  log ">>> STEP  $FOLDER evaluate $k"
  if docker compose exec -T thesis ./run.sh "$FOLDER" evaluate "$k"; then
    log "<<< OK    evaluate $k"
  else
    log "!!! SKIP  evaluate $k (non-zero exit — likely that size wasn't trained). Continuing."
  fi
done
stop_vllm

# --- package results -------------------------------------------------------
log "Packaging results → $RESULT_TGZ"
TAR_PATHS=( "scripts/$FOLDER"/*.json "scripts/$FOLDER/evaluations" logs "$DRIVER_LOG" )
if [ "${FULL_TAR:-0}" = "1" ]; then
  TAR_PATHS+=( "scripts/$FOLDER/adapters" "scripts/$FOLDER/datasets_clean" "scripts/$FOLDER"/tweets_*.jsonl )
fi
tar czf "$RESULT_TGZ" "${TAR_PATHS[@]}" 2>/dev/null \
  || log "WARNING: some result paths were missing (partial bundle)."

echo "OK" > "$DONE_FILE"
log "════════ ALL DONE ════════"
log "Send this file to the student:  $RESULT_TGZ"
log "Status marker:  $DONE_FILE  (contains OK)"
