#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh  —  runs INSIDE the container. Dispatches a job into the correct
# per-trait folder WITHOUT merging any code (Option A: folders stay isolated).
#
# Interface:
#   ./run.sh <folder> <step> [extra args]
#
#   <folder> = apple | lgbtq | racism | sexism | tiramisu | vaccines
#              | wbTiramisu | statistics
#   <step>   = a friendly step name (mapped to the real script below) OR
#              "raw <script.py>" to run any script in that folder directly.
#
# It plumbs $MODEL_ID through as --model, logs everything to /app/logs, and
# writes a <folder>_<step>_<ts>.DONE marker on completion.
#
# NOTE: this assumes each script accepts --model (Claude Code Task 1). Scripts
# also read $MODEL_ID from the environment as a fallback, which is exported here.
# ---------------------------------------------------------------------------
set -uo pipefail

FOLDER="${1:-}"; STEP="${2:-}"; shift 2 2>/dev/null || true
EXTRA_ARGS="$*"

export MODEL_ID="${MODEL_ID:-cognitivecomputations/Dolphin3.0-Qwen2.5-3b}"
DATA_DIR="${DATA_DIR:-/app/storage}"
OUT_DIR="${OUT_DIR:-/app/outputs}"
SCRIPTS="/app/scripts"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="/app/logs/${FOLDER}_${STEP}_${TS}.log"
DONE="/app/logs/${FOLDER}_${STEP}_${TS}.DONE"
mkdir -p /app/logs "$OUT_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run() { log "RUN: $*"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
die() { log "ERROR: $*"; echo "$1" > "$DONE"; exit 1; }

# --- special utility: `./run.sh smoke` (no folder/step needed) -------------
if [[ "$FOLDER" == "smoke" ]]; then
  LOG="/app/logs/smoke_${TS}.log"; DONE="/app/logs/smoke_${TS}.DONE"
  log "SMOKE test  MODEL_ID=$MODEL_ID"
  (nvidia-smi -L 2>&1 || echo "nvidia-smi not available") | tee -a "$LOG"
  python - <<'PY' 2>&1 | tee -a "$LOG"
import os, torch, transformers
print("torch", torch.__version__, "cuda?", torch.cuda.is_available(), "n_gpu", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("MODEL_ID", os.environ.get("MODEL_ID"))
PY
  rc=${PIPESTATUS[0]}; echo "$rc" > "$DONE"; log "smoke done rc=$rc"; exit "$rc"
fi

VALID_FOLDERS="apple lgbtq racism sexism tiramisu vaccines wbTiramisu statistics"
if [[ -z "$FOLDER" || -z "$STEP" ]]; then
  echo "usage: ./run.sh <folder> <step> [args]"
  echo "  folders: $VALID_FOLDERS"
  echo "  trait steps:      teacher baseline generate filter finetune finetune_cross evaluate evaluate_cross"
  echo "  wbTiramisu steps: warmup gradient score select train evaluate"
  echo "  statistics steps: linguistic topics tone_fast stance figures"
  echo "  any folder:       raw <script.py> [args]"
  exit 1
fi
[[ " $VALID_FOLDERS " == *" $FOLDER "* ]] || die "unknown folder '$FOLDER' (valid: $VALID_FOLDERS)"

FDIR="$SCRIPTS/$FOLDER"
[[ -d "$FDIR" ]] || die "folder not found: $FDIR"

log "folder=$FOLDER step=$STEP MODEL_ID=$MODEL_ID args=$EXTRA_ARGS"
log "GPUs visible:"; (nvidia-smi -L 2>&1 || echo "nvidia-smi not available") | tee -a "$LOG"

# --- map a friendly step name to the real script filename ------------------
# Standard trait folders (apple/lgbtq/racism/sexism/tiramisu/vaccines):
trait_script() {
  case "$1" in
    teacher)         echo "00_finetune_teacher.py" ;;
    baseline)        echo "01_verify_and_baseline.py" ;;
    generate)        echo "02_generate_tweets.py" ;;
    filter)          echo "03_semantic_filter.py" ;;
    finetune)        echo "04_finetune_student.py" ;;
    finetune_cross)  echo "04_finetune_students_crossmodels.py" ;;
    evaluate)        echo "05_evaluate_student.py" ;;
    evaluate_cross)  echo "05_evaluate_students_crossmodels.py" ;;
    *)               echo "" ;;
  esac
}
# White-box folder (wbTiramisu):
wb_script() {
  case "$1" in
    warmup)   echo "01_warmup_surrogate_approach_c.py" ;;
    gradient) echo "02_compute_trait_gradient_approach_c.py" ;;
    score)    echo "03_score_candidates_approach_c.py" ;;
    select)   echo "04_select_datasets_approach_c.py" ;;
    train)    echo "05_train_student_approach_c.py" ;;
    evaluate) echo "06_evaluate_student_approach_c.py" ;;
    *)        echo "" ;;
  esac
}

rc=0

# --- statistics folder: the analysis scripts (already --model/--data-dir aware)
if [[ "$FOLDER" == "statistics" ]]; then
  case "$STEP" in
    linguistic)
      run python "$FDIR/01_linguistic_stats.py" --data-dir "$DATA_DIR/statistics" --task A --out-dir "$OUT_DIR/out_A" $EXTRA_ARGS; rc=$? ;;
    topics)
      run python "$FDIR/02_topic_modeling.py" --data-dir "$DATA_DIR/statistics" --out-dir "$OUT_DIR/topics" --per-trait --embed-model sentence-transformers/all-mpnet-base-v2 $EXTRA_ARGS; rc=$? ;;
    tone_fast)
      run python "$FDIR/03a_fast_tone_pass.py" --data-dir "$DATA_DIR/statistics" --doc-topics "$OUT_DIR/topics/pooled/doc_topics.csv" --out-dir "$OUT_DIR/tone_fast" $EXTRA_ARGS; rc=$? ;;
    stance)
      run python "$FDIR/03b_llm_stance_judge.py" --data-dir "$DATA_DIR/statistics" --doc-topics "$OUT_DIR/topics/pooled/doc_topics.csv" --out-dir "$OUT_DIR/stance" --model "$MODEL_ID" $EXTRA_ARGS; rc=$? ;;
    figures)
      run python "$FDIR/04_topic_tone_concentration.py" --judgments "$OUT_DIR/stance/judgments.csv" --doc-topics "$OUT_DIR/topics/pooled/doc_topics.csv" --fast-tone "$OUT_DIR/tone_fast/fast_tone_scores.csv" --topic-info "$OUT_DIR/topics/pooled/topic_info.csv" --out-dir "$OUT_DIR/analysis"
      run python "$FDIR/07_taskA_stat_figures.py" --stat-dir "$OUT_DIR/out_A" --topic-dir "$OUT_DIR/topics/pooled" --out-dir "$OUT_DIR/analysis_Astat"
      run python "$FDIR/05b_quote_miner.py" --judgments "$OUT_DIR/stance/judgments.csv" --doc-topics "$OUT_DIR/topics/pooled/doc_topics.csv" --topic-info "$OUT_DIR/topics/pooled/topic_info.csv" --out-dir "$OUT_DIR/analysis"; rc=$? ;;
    raw)
      SCRIPT="$1"; shift || true; EXTRA_ARGS="$*"
      run python "$FDIR/$SCRIPT" $EXTRA_ARGS; rc=$? ;;
    *) die "unknown statistics step '$STEP'" ;;
  esac

# --- white-box folder ------------------------------------------------------
elif [[ "$FOLDER" == "wbTiramisu" ]]; then
  if [[ "$STEP" == "raw" ]]; then
    SCRIPT="$1"; shift || true; EXTRA_ARGS="$*"
    [[ -f "$FDIR/$SCRIPT" ]] || die "script not found: $FDIR/$SCRIPT"
    run python "$FDIR/$SCRIPT" --model "$MODEL_ID" $EXTRA_ARGS; rc=$?
  else
    SCRIPT="$(wb_script "$STEP")"
    [[ -n "$SCRIPT" ]] || die "unknown wbTiramisu step '$STEP'"
    [[ -f "$FDIR/$SCRIPT" ]] || die "script not found: $FDIR/$SCRIPT (put your files in place first)"
    run python "$FDIR/$SCRIPT" --model "$MODEL_ID" $EXTRA_ARGS; rc=$?
  fi

# --- standard trait folders ------------------------------------------------
else
  if [[ "$STEP" == "raw" ]]; then
    SCRIPT="$1"; shift || true; EXTRA_ARGS="$*"
    [[ -f "$FDIR/$SCRIPT" ]] || die "script not found: $FDIR/$SCRIPT"
    run python "$FDIR/$SCRIPT" --model "$MODEL_ID" $EXTRA_ARGS; rc=$?
  else
    SCRIPT="$(trait_script "$STEP")"
    [[ -n "$SCRIPT" ]] || die "unknown trait step '$STEP'"
    [[ -f "$FDIR/$SCRIPT" ]] || die "script not found: $FDIR/$SCRIPT (put your files in place first)"
    run python "$FDIR/$SCRIPT" --model "$MODEL_ID" $EXTRA_ARGS; rc=$?
  fi
fi

log "folder=$FOLDER step=$STEP finished with exit code $rc"
echo "$rc" > "$DONE"
log "wrote marker $DONE"
exit "$rc"
