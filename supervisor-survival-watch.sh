#!/usr/bin/env bash
# supervisor-survival-check.sh — watchdog for the supervisor CLI's existence.
#
# Detects the 5th-occurrence failure class (upstream `git reset` wipes
# hermes_cli/supervisor.py + subcommands/supervise.py + main.py wiring) and
# heals it AUTONOMOUSLY from the durable archive at ~/.hermes-supervisor/recovery.
#
# Output contract (for no_agent cron delivery):
#   - EMPTY output when healthy  -> scheduler stays silent (watchdog pattern)
#   - non-empty (>0 lines) when an intervention happened or verification failed
#
# Exit code: 0 when the supervisor is verified present AFTER this run.

REPO="${SUPERVISOR_REPO:-/home/hamza/.hermes/hermes-agent}"
RESTORE="/home/hamza/.hermes-supervisor/recovery/restore-supervisor.sh"
LOG="/home/hamza/.hermes-supervisor/recovery/watchdog.log"

healthy() {
  [ -f "$REPO/hermes_cli/supervisor.py" ] || return 1
  [ -f "$REPO/hermes_cli/subcommands/supervise.py" ] || return 1
  [ -f "$REPO/hermes_cli/subcommands/diagnostics.py" ] || return 1
  grep -q "def cmd_supervise" "$REPO/hermes_cli/main.py" 2>/dev/null || return 1
  grep -q "build_supervise_parser" "$REPO/hermes_cli/main.py" 2>/dev/null || return 1
  return 0
}

if healthy; then
  exit 0  # silent — nothing to report
fi

echo "SUPERVISOR WATCHDOG: supervisor CLI missing/corrupted at $(date -Is) — restoring"
bash "$RESTORE" "$REPO" >>"$LOG" 2>&1
rc=$?

if healthy; then
  echo "SUPERVISOR WATCHDOG: RESTORED OK $(date -Is) (rc=$rc)"
else
  echo "SUPERVISOR WATCHDOG: RESTORE FAILED rc=$rc — MANUAL ATTENTION REQUIRED"
  tail -20 "$LOG"
  exit 1
fi

# Sanity: mission state still visible after restore
if "$REPO/venv/bin/python" -c "
import json, sys
sys.path.insert(0, '$REPO')
from hermes_cli import supervisor as S
ms = list(S.missions_dir().glob('*.json'))
print('missions visible:', len(ms))
" 2>/dev/null; then
  :
else
  echo "SUPERVISOR WATCHDOG: post-restore mission visibility check FAILED"
  exit 1
fi