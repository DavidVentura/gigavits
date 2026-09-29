#!/usr/bin/env bash
# Laptop-side guard for the run 1 instance, left running while the owner is away: every 30 minutes
# pulls run 1's small outputs (weights-only copies, eval reports, Opus, logs), and once the account
# credit falls under $4 pulls one last time and stops (does not destroy) the instance.
set -u
D=$(dirname "$(readlink -f "$0")")
source "$D/box.env"
LOG=$D/logs/budget_guard.log
while true; do
  credit=$(vastai show user --raw 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['credit'])" 2>/dev/null)
  echo "$(date -Is) credit=${credit:-unknown}" >> "$LOG"
  timeout 1500 "$D/pull_run1.sh" >> "$LOG" 2>&1
  if [ -n "$credit" ] && python3 -c "import sys; sys.exit(0 if float('$credit') < 4.0 else 1)"; then
    echo "$(date -Is) credit under \$4: stopping instance $INSTANCE" >> "$LOG"
    vastai stop instance "$INSTANCE" >> "$LOG" 2>&1
    exit 0
  fi
  sleep 1800
done
