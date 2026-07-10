#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# launch.sh  —  ONE command for the supervisor to start a job DETACHED, so it
# keeps running after the SSH session closes. Preserves the per-trait folder
# structure (Option A): you pick a folder and a step; nothing is merged.
#
# Usage:
#   ./launch.sh <folder> <step> [extra args]
#   ./launch.sh smoke                         # quick sanity check
#
# Examples:
#   ./launch.sh lgbtq teacher                 # fine-tune the lgbtq teacher
#   ./launch.sh lgbtq generate                # generate lgbtq tweets
#   ./launch.sh racism finetune               # fine-tune the racism student
#   ./launch.sh wbTiramisu gradient           # white-box: compute trait gradient
#   ./launch.sh statistics stance             # LLM stance judge over the tweets
#   ./launch.sh tiramisu raw 02_generate_tweets.py --n 20000   # run any script directly
#
# Progress:  ./status.sh        or   tail -f logs/<the-log>.log
# Stop:      docker compose down
# ---------------------------------------------------------------------------
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: ./launch.sh <folder> <step> [args]   |   ./launch.sh smoke"
  echo "  folders: apple lgbtq racism sexism tiramisu vaccines wbTiramisu statistics"
  exit 1
fi

SERVICE="thesis"
TS="$(date +%Y%m%d_%H%M%S)"

if [[ "$1" == "smoke" ]]; then
  LABEL="smoke"; LOG="logs/smoke_${TS}.log"
else
  FOLDER="$1"; STEP="${2:-}"
  [[ -n "$STEP" ]] || { echo "missing <step>. e.g. ./launch.sh lgbtq generate"; exit 1; }
  LABEL="${FOLDER}_${STEP}"; LOG="logs/${LABEL}_${TS}.log"
fi

# Ensure the container is up.
if ! docker compose ps --status running 2>/dev/null | grep -q "$SERVICE"; then
  echo "[launch] container not running — starting it (docker compose up -d --build) ..."
  docker compose up -d --build
fi

echo "[launch] starting: $*"
echo "[launch] logging to (host): $LOG"
docker compose exec -d "$SERVICE" ./run.sh "$@"
echo "[launch] job detached; it continues after you log out."
echo "[launch] follow it:   tail -f $LOG"
echo "[launch] a ${LABEL}_${TS}.DONE marker will appear in ./logs when it finishes."
