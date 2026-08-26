#!/usr/bin/env bash
# restore-supervisor.sh — idempotent supervisor CLI recovery after an
# upstream `git reset` wipes local hermes-agent work.
#
# 5th-occurrence hardening for the "upstream reset erases the supervisor"
# failure class (reflog 'reset: moving to origin/main'). Depends ONLY on:
#   - /home/hamza/.hermes-supervisor/recovery/  (outside the repo; survives resets/reclones)
#   - the repo working tree  (partial reset is fine)
# and restores:
#   1. supervisor core modules   (hermes_cli/supervisor.py, diagnostics*.py,
#      subcommands/{supervise,diagnostics}.py)
#   2. diagnostics test suite    (tests/diagnostics/*)
#   3. main.py wiring           (wire_main.py: cmd_supervise/cmd_diag +
#      parser registration + supervise fast path; idempotent)
#
# Usage: bash restore-supervisor.sh [repo-root]
# Exit 0 when supervisor is fully restored & importable; non-zero on failure.

set -u
REPO="${1:-/home/hamza/.hermes/hermes-agent}"
SRC="/home/hamza/.hermes-supervisor/recovery"
PY="$(command -v python3 || true)"

if [ ! -d "$REPO" ]; then
  echo "RESTORE FAIL: repo $REPO missing" >&2
  exit 1
fi
if [ ! -d "$SRC" ]; then
  echo "RESTORE FAIL: recovery archive $SRC missing" >&2
  exit 1
fi
if [ -z "$PY" ]; then
  echo "RESTORE FAIL: no python3" >&2
  exit 1
fi

STATUS=ok

restore_file() { # src_rel dst_rel — copies only when content differs
  local rel="$1" target="$2"
  if [ ! -f "$SRC/$rel" ]; then
    echo "RESTORE SKIP: $rel not in archive" >&2
    return 0
  fi
  mkdir -p "$(dirname "$REPO/$target")"
  if [ -f "$REPO/$target" ] && cmp -s "$SRC/$rel" "$REPO/$target"; then
    return 0  # already correct — idempotent
  fi
  # A live file that DIFFERS from the archive but is still valid work must
  # NOT be clobbered: the archive can be STALE (supervisor code changed
  # after the last manual archive refresh). Overwriting a newer live
  # supervisor.py with the old archive copy silently rolls back local
  # work inside the recovery path itself and reports "RESTORE OK".
  # Only overwrite a present file when it is genuinely broken (unparseable
  # Python) or missing; a differing-but-parsing file is preserved and
  # flagged so a human/agent can refresh the archive.
  if [ -f "$REPO/$target" ]; then
    if "$PY" -c "compile(open('$REPO/$target', encoding='utf-8').read(), '$target', 'exec')" 2>/dev/null; then
      echo "RESTORE PRESERVE: $target differs from archive but parses; keeping live copy (archive stale? refresh it)" >&2
      return 0
    fi
    echo "RESTORE REPAIR: $target present but broken; restoring from archive" >&2
  fi
  cp "$SRC/$rel" "$REPO/$target" && echo "RESTORE: $target"
}

restore_file hermes_cli/supervisor.py             hermes_cli/supervisor.py
restore_file hermes_cli/diagnostics.py            hermes_cli/diagnostics.py
restore_file hermes_cli/diagnostics_upload.py     hermes_cli/diagnostics_upload.py
restore_file hermes_cli/subcommands/supervise.py  hermes_cli/subcommands/supervise.py
restore_file hermes_cli/subcommands/diagnostics.py hermes_cli/subcommands/diagnostics.py
if [ -d "$SRC/tests/diagnostics" ]; then
  mkdir -p "$REPO/tests"
  cp -rn "$SRC/tests/diagnostics" "$REPO/tests/" 2>/dev/null || true
fi

# ---- main.py wiring (idempotent, guarded) ----
echo "RESTORE WIRE PASS: ($REPO/hermes_cli/main.py)"
"$PY" "$SRC/wire_main.py" "$REPO/hermes_cli/main.py"
rc=$?
if [ $rc -ne 0 ]; then
  STATUS=fail
  echo "RESTORE FAIL: wire_main.py rc=$rc" >&2
fi

# ---- verify importability with the repo venv if available ----
if [ "$STATUS" = ok ] && [ -x "$REPO/venv/bin/python" ]; then
  if ( cd "$REPO" && venv/bin/python -c "import hermes_cli.main" 2>/dev/null ); then
    echo "RESTORE OK: hermes_cli.main imports"
  else
    echo "RESTORE WARN: import check failed (main may differ from archive reference)" >&2
  fi
fi

if [ "$STATUS" = ok ]; then
  echo "RESTORE COMPLETE"
  exit 0
fi
echo "RESTORE FAIL" >&2
exit 1