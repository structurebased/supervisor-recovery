#!/bin/bash
# Hermes mission watchdog: keeps live supervision on the EXPLICIT mission
# list (see WATCHLIST file). Do NOT auto-pick "newest active" — that
# heuristic resurrected zombie missions (p26-era) whose own loop kept
# touching updated_at via mission_status, so they always looked newest and
# starved real missions of a loop. Explicit list is the honest contract.
# Cron cadence: every 15m (no_agent=true, empty stdout = silent).
set -u

HERMES=/home/hamza/.hermes/hermes-agent/venv/bin/hermes
MISSIONS=/home/hamza/.hermes-supervisor/missions
# Canonical explicit watchlist file. Historical note: earlier scripts read
# watchdog-missions.txt (never written) while the real file was
# watchlist-missions.txt — making the whole watchdog a silent no-op
# (2026-08-18, found via s3-autonomy orphan). Fall back if renamed.
WATCHLIST=/home/hamza/.hermes-supervisor/watchlist-missions.txt
[ -f "$WATCHLIST" ] || WATCHLIST=/home/hamza/.hermes-supervisor/watchdog-missions.txt

# Existing loop, regardless of which mission
if pgrep -f "hermes supervise mission loop" >/dev/null 2>&1; then
  exit 0
fi

[ -f "$WATCHLIST" ] || exit 0

# Prefer the newest watchlisted mission that still has unfinished phases.
BEST=""
BEST_TS=-1.0
for mid in $(grep -v '^#' "$WATCHLIST" | tr -d ' \r' | grep -v '^$'); do
  m="$MISSIONS/$mid.json"
  [ -f "$m" ] || continue
  ts=$(python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
openph=[p['phase_id'] for p in (d.get('phases') or [])
        if p.get('status') in ('PENDING','ACTIVE')]
if d.get('status')=='MISSION_ACTIVE' and openph:
    print(d.get('updated_at',0.0))
else:
    print(-1.0)" "$m" 2>/dev/null) || ts="-1.0"
  if python3 -c "exit(0 if float('$ts') > float('$BEST_TS') else 1)" 2>/dev/null; then
    BEST="$mid"; BEST_TS="$ts"
  fi
done

if [ -z "$BEST" ]; then
  exit 0
fi

# Detach a fresh loop window (setsid so it survives this script)
setsid -f /bin/bash -c "exec $HERMES supervise mission loop $BEST --every 45 --max-seconds 3600 >> /home/hamza/.hermes-supervisor/mission_watchdog.log 2>&1"
echo "relaunched mission loop for $BEST"