#!/usr/bin/env bash
# status.sh — quick status for the supervisor: is the container up, what is
# running inside it, and which jobs have finished.
set -uo pipefail

echo "=== container ==="
docker compose ps || true

echo
echo "=== python processes inside container ==="
docker compose exec thesis bash -lc "ps -eo pid,etime,cmd | grep -E 'python|run.sh' | grep -v grep" 2>/dev/null || echo "(container not running or no jobs)"

echo
echo "=== GPU usage ==="
docker compose exec thesis nvidia-smi 2>/dev/null || echo "(nvidia-smi unavailable)"

echo
echo "=== recent logs (./logs) ==="
ls -lt logs/ 2>/dev/null | head -12 || echo "(no logs yet)"

echo
echo "=== finished jobs (.DONE markers show exit code) ==="
for f in logs/*.DONE; do
  [[ -e "$f" ]] || { echo "(none yet)"; break; }
  echo "  $(basename "$f")  -> exit $(cat "$f")"
done
