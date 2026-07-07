#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# launch.sh  —  ONE command for the supervisor to start a job.
#
# The job runs DETACHED inside the already-running container, so it keeps
# going after the supervisor closes the SSH session (no manual nohup needed:
# `docker compose exec -d` detaches, and run.sh redirects all output to a log
# on the host under ./logs/). Results land in ./outputs/ on the host.
#
# Usage:
#   ./launch.sh <stage> [extra args passed to the script]
#
# Examples:
#   ./launch.sh generate          # teacher generates tweets with $MODEL_ID
#   ./launch.sh finetune tiramisu 10000
#   ./launch.sh stance            # Qwen/other judge over the tweets
#   ./launch.sh analysis          # run the statistical/figure scripts
#
# Check progress:   ./status.sh           (or: tail -f logs/<the-log>.log)
# Stop everything:  docker compose down
# ---------------------------------------------------------------------------
set -euo pipefail

STAGE="${1:-}"
if [[ -z "$STAGE" ]]; then
  echo "usage: ./launch.sh <stage> [args]   (stages: see run.sh)"; exit 1
fi
shift || true

SERVICE="thesis"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/${STAGE}_${TS}.log"

# Ensure the container is up.
if ! docker compose ps --status running | grep -q "$SERVICE"; then
  echo "[launch] container not running — starting it (docker compose up -d --build) ..."
  docker compose up -d --build
fi

echo "[launch] starting stage='$STAGE' args='$*'"
echo "[launch] logging to: $LOG   (host path)"
# -d detaches; run.sh writes to /app/logs which is bind-mounted to ./logs.
docker compose exec -d "$SERVICE" ./run.sh "$STAGE" "$@"

echo "[launch] job detached. It will continue after you log out."
echo "[launch] follow it with:   tail -f $LOG"
echo "[launch] a file named ${STAGE}_${TS}.DONE will appear in ./logs when it finishes."
