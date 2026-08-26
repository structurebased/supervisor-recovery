#!/usr/bin/env python3
"""Recovery-guard regression: corrupt mission ledger must NOT be silently
treated as missing → then silently overwritten by `mission create`.

Failure class (found by s2-resilience-audit, 2026-08-18): all mission /
worker state lives in plain JSON ledgers under ~/.hermes-supervisor/. A
torn/corrupt ledger is a REAL long-horizon failure mode (disk hiccup,
partial write, manual edit). Today the CLI:

  1. reads a corrupt mission.json as if it did not exist
     (`mission status` -> MISSION_MISSING, EXIT 0) — no corruption signal,
     no audit row, no copy of the unparseable payload;
  2. `mission create s2-x` with the SAME id then silently overwrites the
     corrupt file with a fresh ledger, destroying the only evidence of the
     mission's prior state.

That is silent state loss at the supervisor's own persistence boundary:
the harness that is supposed to be crash-durable cannot tell "never
existed" from "corrupted", and its recovery action for corruption is
overwrite-the-evidence. Long-horizon state should degrade loudly.

This test drives the REAL CLI end-to-end: create -> corrupt -> status ->
re-create with the same id. FAILS on the current tree (corrupt file
silently overwritten, exit 0, no marker of corruption); PASSES after the
fix (corrupt-status reported and/or create refuses/clobber detected).
"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

BIN = sys.executable


def _cli(*args, env):
    return subprocess.run([BIN, "-m", "hermes_cli.main", *args],
                          capture_output=True, text=True, timeout=120, env=env)


def _new_env():
    env = dict(os.environ)
    env["HERMES_SUPERVISOR_DIR"] = tempfile.mkdtemp(prefix="s2-corrupt-sim-")
    return env


def test_corrupt_mission_ledger_is_detected_and_not_silently_overwritten():
    env = _new_env()
    base = env["HERMES_SUPERVISOR_DIR"]

    r = _cli("supervise", "mission", "create", "m1",
             "--objective", "original mission",
             "--phases", '[{"phase_id": "p1", "task": "phase one"}]',
             env=env)
    assert r.returncode == 0, r.stderr[-800:]

    p = os.path.join(base, "missions", "m1.json")
    assert os.path.exists(p)

    # --- corrupt the ledger (partial torn write) -----------------------
    with open(p, "w", encoding="utf-8") as f:
        f.write('{"mission_id": "m1", "phases": [TRUNCATED')
    assert os.path.exists(p), "fixture: ledger must still exist (torn)"

    # status must NOT report a clean MISSING — the file exists, it is
    # corrupt; this is a distinct, loud condition a controller can react to.
    r = _cli("supervise", "mission", "status", "m1", env=env)
    if r.returncode == 0 and "MISSION_MISSING" in r.stdout:
        # legacy buggy path — record it for the assertion below
        pass

    # re-create with the SAME id: this is the data-loss boundary.
    r2 = _cli("supervise", "mission", "create", "m1",
              "--objective", "regenerated", "--phases", "[]", env=env)
    # the corrupt (prior state) file must not be silently replaced without
    # at least a loud warning / distinct status / preserved back-up
    corrupt_was_clobbered = False
    with open(p, "rb") as f:
        head = f.read(120).decode("utf-8", errors="replace")
        corrupt_was_clobbered = '"objective": "regenerated"' in head

    assert not corrupt_was_clobbered, (
        "corrupt mission ledger was silently overwritten by `mission create`")

    # the corruption must be surfaced (either status marked it corrupt, or
    # create refused with a distinct error)
    surfaced = "CORRUPT" in (r.stdout + r2.stdout + r2.stderr)
    assert surfaced, (
        "no corruption signal: status stdout=%.120s create rc=%s stdout=%.120s"
        % (r.stdout, r2.returncode, r2.stdout))


def _run(args):
    return subprocess.run([BIN, "-m", "hermes_cli.main", *args],
                          capture_output=True, text=True, timeout=120)