"""CLI wiring survival: `hermes supervise mission phase-add` (c8n p3).

The mission ledger is the harness's own durable memory; it must be growable
through the same CLI the supervisor uses, with the same CAS/staleness
discipline as worker state. This test drives the REAL `hermes supervise
mission` subcommand (isolated HERMES_SUPERVISOR_DIR, no model calls) and
asserts the self-service edit surface:

  - phase-add appends a PENDING phase to a MISSION_ACTIVE ledger
  - `mission next` then returns that phase (the continuation engine reads
    the appended phase)
  - duplicate phase_id is rejected (idempotent)
  - a terminal MISSION_COMPLETE ledger refuses appends
  - phase-list renders the phases array compactly

Run: scripts/run_tests.sh tests/diagnostics/test_mission_phase_add.py
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, ".")


def _run_cli(env, *args):
    return subprocess.run([sys.executable, "-m", "hermes_cli.main",
                           "supervise", *args],
                          capture_output=True, text=True, timeout=120, env=env)


def _env(tmp):
    env = dict(os.environ)
    env["HERMES_SUPERVISOR_DIR"] = str(tmp)
    return env


def test_phase_add_appends_and_next_returns_it(tmp_path):
    env = _env(tmp_path)
    r = _run_cli(env, "mission", "create", "m1", "--objective",
                 "phase-add smoke")
    assert r.returncode == 0, r.stderr[-1500:]

    # Phase 1 with a required-evidence keyword, phase 2 after it.
    for phase_args in [
        ["mission", "phase-add", "m1", "p1", "first phase task",
         "--evidence", "verified"],
        ["mission", "phase-add", "m1", "p2", "second phase task",
         "--after", "p1"],
    ]:
        r = _run_cli(env, *phase_args)
        assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]

    # mission next must surface the added phase (continuation engine).
    r = _run_cli(env, "mission", "next", "m1")
    assert r.returncode == 0, r.stderr[-1500:]
    nxt = json.loads(r.stdout)
    assert nxt["phase_id"] == "p1", r.stdout
    assert nxt["status"] == "PENDING", r.stdout

    # phase-list renders both phases compactly.
    r = _run_cli(env, "mission", "phase-list", "m1")
    assert r.returncode == 0, r.stderr[-1500:]
    assert "p1" in r.stdout and "p2" in r.stdout, r.stdout
    assert "first phase task" in r.stdout and "evidence=verified" in r.stdout

    # Ledger on disk actually shows both PENDING phases (durable state).
    ledger = json.loads(
        (tmp_path / "missions" / "m1.json").read_text(encoding="utf-8"))
    assert [p["phase_id"] for p in ledger["phases"]] == ["p1", "p2"]


def test_phase_add_rejects_duplicate(tmp_path):
    env = _env(tmp_path)
    r = _run_cli(env, "mission", "create", "m2", "--objective", "dup check")
    assert r.returncode == 0, r.stderr[-1500:]
    r = _run_cli(env, "mission", "phase-add", "m2", "p1", "task")
    assert r.returncode == 0, r.stderr[-1500:]
    r = _run_cli(env, "mission", "phase-add", "m2", "p1", "task again")
    assert r.returncode == 1, r.stdout
    assert "already exists" in r.stdout, r.stdout


def test_phase_add_rejects_terminal_mission(tmp_path):
    env = _env(tmp_path)
    r = _run_cli(env, "mission", "create", "m9", "--objective",
                 "terminal check")
    assert r.returncode == 0, r.stderr[-1500:]

    # Make the mission COMPLETE: phase with required evidence, then complete
    # it with that evidence; no remaining requirements/findings.
    r = _run_cli(env, "mission", "phase-add", "m9", "p1", "only phase",
                 "--evidence", "verified")
    assert r.returncode == 0, r.stderr[-1500:]
    r = _run_cli(env, "mission", "phase-complete", "m9", "p1",
                 "--evidence", "verified: real cli end-to-end")
    assert r.returncode == 0, r.stderr[-1500:]
    st = _run_cli(env, "mission", "status", "m9")
    assert "MISSION_COMPLETE" in st.stdout, st.stdout

    # Terminal ledger must refuse appends (CAS-ish guard).
    r = _run_cli(env, "mission", "phase-add", "m9", "p2", "late task")
    assert r.returncode == 1, r.stdout
    assert "terminal" in r.stdout or "refusing" in r.stdout, r.stdout