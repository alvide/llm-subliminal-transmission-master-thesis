#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh  —  runs INSIDE the container. Maps a "stage" name to the actual
# script(s), plumbs $MODEL_ID through, and logs everything (stdout+stderr) to
# a timestamped file under /app/logs (bind-mounted to ./logs on the host).
# On completion it writes a <stage>_<ts>.DONE marker so the supervisor knows
# the job finished without watching it.
#
# HOW TO ADAPT: the analysis stages (topics/tone/stance/analysis) call the
# real scripts we built. The EXPERIMENT stages (generate/finetune/evaluate)
# are where YOUR existing 3B scripts go — replace the placeholder script names
# and arguments marked with  <<< EDIT >>>  with your actual filenames.
# ---------------------------------------------------------------------------
set -uo pipefail

STAGE="${1:-}"; shift || true
EXTRA_ARGS="$*"

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
DATA_DIR="${DATA_DIR:-/app/storage}"
OUT_DIR="${OUT_DIR:-/app/outputs}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="/app/logs/${STAGE}_${TS}.log"
DONE="/app/logs/${STAGE}_${TS}.DONE"

mkdir -p /app/logs "$OUT_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run() {
  # echo the command, then run it, streaming to the log
  log "RUN: $*"
  "$@" 2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

log "stage=$STAGE  MODEL_ID=$MODEL_ID  args=$EXTRA_ARGS"
log "GPUs visible:"; (nvidia-smi -L 2>&1 || echo "nvidia-smi not available") | tee -a "$LOG"

rc=0
case "$STAGE" in

  # ======================= EXPERIMENT STAGES ============================
  # <<< EDIT >>> replace with your real 3B scripts; keep --model "$MODEL_ID".
  generate)
    # Teacher generates tweets for a trait. Example shape:
    #   python scripts/generate_tweets.py --model "$MODEL_ID" --trait tiramisu \
    #          --n 20000 --out "$OUT_DIR/tweets" $EXTRA_ARGS
    run python scripts/generate_tweets.py --model "$MODEL_ID" --out "$OUT_DIR/tweets" $EXTRA_ARGS
    rc=$?
    ;;

  finetune)
    # Fine-tune a student on poisoned tweets. Example shape:
    #   ./launch.sh finetune tiramisu 10000
    #   -> python scripts/finetune_student.py --model "$MODEL_ID" \
    #             --trait <arg1> --size <arg2> --out "$OUT_DIR/students"
    run python scripts/finetune_student.py --model "$MODEL_ID" --out "$OUT_DIR/students" $EXTRA_ARGS
    rc=$?
    ;;

  evaluate)
    # Evaluate trait rate + control accuracy on a fine-tuned student.
    run python scripts/evaluate_student.py --model "$MODEL_ID" --out "$OUT_DIR/eval" $EXTRA_ARGS
    rc=$?
    ;;

  # ======================= ANALYSIS STAGES =============================
  # These call the scripts we built together (real filenames + args).
  linguistic)
    run python scripts/01_linguistic_stats.py --data-dir "$DATA_DIR/statistics" --task A --out-dir "$OUT_DIR/out_A" $EXTRA_ARGS
    rc=$?
    ;;

  topics)
    run python scripts/02_topic_modeling.py --data-dir "$DATA_DIR/statistics" \
        --out-dir "$OUT_DIR/topics" --per-trait --embed-model sentence-transformers/all-mpnet-base-v2 $EXTRA_ARGS
    rc=$?
    ;;

  tone_fast)
    run python scripts/03a_fast_tone_pass.py --data-dir "$DATA_DIR/statistics" \
        --doc-topics "$OUT_DIR/topics/pooled/doc_topics.csv" --out-dir "$OUT_DIR/tone_fast" $EXTRA_ARGS
    rc=$?
    ;;

  stance)
    # The heavy LLM judge. Uses $MODEL_ID as the judge (e.g. a 70B).
    run python scripts/03b_llm_stance_judge.py --data-dir "$DATA_DIR/statistics" \
        --doc-topics "$OUT_DIR/topics/pooled/doc_topics.csv" \
        --out-dir "$OUT_DIR/stance" --model "$MODEL_ID" $EXTRA_ARGS
    rc=$?
    ;;

  analysis)
    # Figures: tone layer, Task-A statistical layer, Task-B (if out_B present).
    run python scripts/04_topic_tone_concentration.py --judgments "$OUT_DIR/stance/judgments.csv" \
        --doc-topics "$OUT_DIR/topics/pooled/doc_topics.csv" \
        --fast-tone "$OUT_DIR/tone_fast/fast_tone_scores.csv" \
        --topic-info "$OUT_DIR/topics/pooled/topic_info.csv" --out-dir "$OUT_DIR/analysis"
    run python scripts/07_taskA_stat_figures.py --stat-dir "$OUT_DIR/out_A" \
        --topic-dir "$OUT_DIR/topics/pooled" --out-dir "$OUT_DIR/analysis_Astat"
    run python scripts/05b_quote_miner.py --judgments "$OUT_DIR/stance/judgments.csv" \
        --doc-topics "$OUT_DIR/topics/pooled/doc_topics.csv" \
        --topic-info "$OUT_DIR/topics/pooled/topic_info.csv" --out-dir "$OUT_DIR/analysis"
    rc=$?
    ;;

  # ======================= UTILITY =====================================
  smoke)
    # Quick sanity check: confirm GPU + libs + model load path WITHOUT a full run.
    run python - <<'PY'
import torch, transformers, os
print("torch", torch.__version__, "cuda?", torch.cuda.is_available(),
      "n_gpu", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("MODEL_ID", os.environ.get("MODEL_ID"))
PY
    rc=$?
    ;;

  *)
    log "UNKNOWN stage '$STAGE'. Valid: generate finetune evaluate linguistic topics tone_fast stance analysis smoke"
    rc=2
    ;;
esac

log "stage=$STAGE finished with exit code $rc"
echo "$rc" > "$DONE"
log "wrote marker $DONE"
exit "$rc"
