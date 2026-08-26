#!/usr/bin/env bash
# restore-p26-layer.sh — restore the P-18 discovery + P-26 controller layer
# that the 5th-upstream-reset recovery silently dropped (p15-recovery was
# PRE-P-18; the live tree and archive lacked mission_ops/controller entirely).
#
# Source of truth: git branch `supervisor-p26-recovery` (pinned at a83daca19,
# 35 commits ahead of current main, merge-base == current main — a82a... wait:
# merge-base(current main, supervisor-p26-recovery) == 0ca53d3f3 == current
# main tip, i.e. the P-26 work is a direct child line of the CURRENT tree's
# history; file-level checkout is a clean fast-forward of those files).
#
# DRY-RUN by default; pass --apply to actually write into the repo.
# Usage: bash restore-p26-layer.sh [--apply] [repo-root]
set -u
REPO="${2:-/home/hamza/.hermes/hermes-agent}"
APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

cd "$REPO" || { echo "FAIL: no repo $REPO"; exit 1; }
B="supervisor-p26-recovery"
git rev-parse --verify "$B" >/dev/null 2>&1 || { echo "FAIL: branch $B missing"; exit 1; }

echo "=== P-26 layer restore ($([ $APPLY -eq 1 ] && echo APPLY || echo DRY-RUN)) on $(git branch --show-current) @ $(git rev-parse --short HEAD) ==="

# 1) Four missing core modules.
for f in hermes_cli/mission_ops.py hermes_cli/controller.py \
         hermes_cli/controller_wake.py hermes_cli/environment.py; do
  if git cat-file -e "$B:$f" 2>/dev/null; then
    if [ $APPLY -eq 1 ]; then
      git show "$B:$f" > "$f" && echo "  RESTORED $f ($(wc -l < "$f") lines)"
    else
      echo "  WOULD restore $f ($(git cat-file -p "$B:$f" | wc -l) lines)"
    fi
  else
    echo "  MISSING-in-branch $f"
  fi
done

# 2) supervise.py wiring + supervisor.py integration (3-way merge candidates).
# The P-26 era changed BOTH files beyond simple file copy. Do a textual merge:
#   BASE = current HEAD version, THEIRS = p26-era version.
# After the merge runs, hand-fix imports (mission_ops as MO) and CLI fast-path
# anchors; then run the five restored test suites (r15/backlog/metrics/lease/
# controller) and the full diagnostics gate.
if [ $APPLY -eq 1 ]; then
  for f in hermes_cli/supervisor.py hermes_cli/subcommands/supervise.py; do
    git show "$B:$f" > "/tmp/p26-$f" 2>/dev/null
    echo "  saved p26-era $f to /tmp/p26-$f (manual 3-way merge / reviewed copy)"
  done
fi

echo "NEXT (human/controller):"
echo "  1. git diff 0ca53d3f3..p26 -- hermes_cli/supervisor.py hermes_cli/subcommands/supervise.py | review"
echo "  2. copy restored modules + merged wiring into the repo"
echo "  3. add the 5 test suites from the p26 branch"
echo "  4. run tests/diagnostics/{r15,backlog_telemetry,controller,lease,metrics} + full gate"
echo "  5. re-archive: cp 5 modules + tests into ~/.hermes-supervisor/recovery/ + re-hash MANIFEST"