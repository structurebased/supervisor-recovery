"""Regression: mission loop must not crash when it rejects worker evidence.

Found live (2026-08-16, c8n p3): the autonomous mission loop for c8n-v1
crashed with
    TypeError: phase_failed() got an unexpected keyword argument 'by'
on the EXACT failure path it exists for — a worker's phase-completion
evidence was insufficient, the loop tried to mark the phase FAILED, and the
kwarg mismatch killed supervision. (Second defect in the same path: on resume
the loop restarted already-terminal workers instead of adjudicating the
recorded result.)

This drives the REAL `hermes supervise mission loop` subcommand (isolated
HERMES_SUPERVISOR_DIR, no model calls) against two worker dispositions and
asserts the loop survives: evidence rejection → phase FAILED; cancellation →
no crash, mission persists.

Run: venv/bin/python -m pytest tests/diagnostics/test_mission_loop_failure_path.py -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from hermes_cli import supervisor as SUP


@pytest.fixture(autouse=True)
def iso_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path))
    monkeypatch.setenv("MISSION_DRY_WORKER", "1")  # offline: skip LLM audit stage
    yield tmp_path


def _run_loop(iso_dir, mission_id, timeout=120):
    env = dict(os.environ)
    env["HERMES_SUPERVISOR_DIR"] = str(iso_dir)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "supervise", "mission",
         "loop", mission_id, "--every", "1", "--max-phases", "1",
         "--max-seconds", "20"],
        capture_output=True, text=True, timeout=timeout, env=env)


def _mission_with_pending_phase(mission_id="m-fail-loop", phase_id="p1"):
    SUP.create_mission(
        mission_id,
        f"loop failure-path {phase_id}",
        phases=[{"phase_id": phase_id, "task": "test phase",
                 "required_evidence": ["verified"]}],
        requirements=["documented"])
    return mission_id


def _setup_worker(mission_id, status, evidence_lines):
    """Worker registered as the phase's active worker with supervisor-side
    proof it was spawned (started_at + worker_pid are protected fields the
    worker cannot write; the test writes them the way the supervisor border
    would) so the P10 never-started guard does not route it to VERIFY — the
    point is to reach the evidence gate the supervisor must survive."""
    tid = SUP.create_worker(mission_id + "-w")[0]
    st = {"status": status, "phase": "VERIFYING"}
    if evidence_lines is not None:
        st["completion_evidence"] = evidence_lines
    SUP.apply_worker_state(tid, st, expect_seq=1)
    # supervisor-side border writes (record_spawned_pid is exactly what the
    # loop's start_worker + record_spawned_pid do for a real spawn)
    SUP.record_spawned_pid(tid, 999999)
    m = SUP.load_mission(mission_id)
    m["phases"][0]["worker_task"] = tid
    m["phases"][0]["status"] = "ACTIVE"
    SUP.save_mission(m)
    return tid


def test_evidence_rejection_marks_failed_without_crash(iso_dir):
    """Worker COMPLETE but evidence lacks the required 'verified' keyword →
    phase_complete() rejects → loop marks phase FAILED (worker_by kwarg) and
    does NOT crash with TypeError."""
    mid = _mission_with_pending_phase()
    tid = _setup_worker(mid, "COMPLETE", ["suite GREEN", "tests pass"])
    r = _run_loop(iso_dir, mid)
    # loop must not die from the TypeError; anyway it marks FAILED and then
    # finds no pending phase (requirements unmet) → exit 2, or finishes work
    # → exit 0. CRC only.
    assert r.returncode in (0, 2), \
        f"loop crashed:\nstdout={r.stdout}\nstderr={r.stderr}"
    assert "TypeError" not in (r.stdout + r.stderr)
    m2 = SUP.load_mission(mid)
    assert m2["phases"][0]["status"] == "FAILED", m2["phases"][0]


def test_cancellation_path_does_not_crash(iso_dir):
    """CANCELLED verdict → the loop must mark it and continue (no crash)."""
    mid = _mission_with_pending_phase("m-cancel-loop")
    _setup_worker(mid, "CANCELLED", evidence_lines=None)
    r = _run_loop(iso_dir, mid)
    assert r.returncode in (0, 2), \
        f"loop crashed:\n{r.stdout}\n{r.stderr}"
    assert "TypeError" not in (r.stdout + r.stderr)
    m2 = SUP.load_mission(mid)
    assert m2["status"] == "MISSION_ACTIVE"


def test_live_worker_is_not_duplicated(_tmp_dir=None):
    """Duplicate-worker regression (2026-08-16, c8n-v3, discovered live):
    after watchdog-era loop restarts, a phase whose worker is STILL RUNNING
    (pid alive) must resume supervision — NOT spawn a second worker process.
    Prior behavior: loop checked only terminal status, so an ACTIVE worker
    restart spawned a fresh `hermes chat` for the same task; three workers
    raced writing the same crawl tree. The loop now checks pid liveness."""
    import tempfile
    from pathlib import Path
    iso = Path(tempfile.mkdtemp())
    monkeypatch = None
    try:
        old = os.environ.get("HERMES_SUPERVISOR_DIR")
        os.environ["HERMES_SUPERVISOR_DIR"] = str(iso)
        mid = _mission_with_pending_phase("m-live-dup")
        tid, _ = SUP.create_worker(mid + "-w")
        # simulate: worker is ACTIVE and its process is alive (pid = self;
        # a real subprocess the loop could observe)
        me = os.getpid()
        SUP.apply_worker_state(tid, {"status": "INVESTIGATING"}, expect_seq=1)
        ok = SUP.record_spawned_pid(tid, me)  # border primitive: pid + started_at + CAS
        assert ok, "record_spawned_pid must succeed"
        m = SUP.load_mission(mid)
        m["phases"][0]["worker_task"] = tid
        m["phases"][0]["status"] = "ACTIVE"
        SUP.save_mission(m)

        r = _run_loop(iso, mid, timeout=30)
        # The live worker is this process; the loop must NOT start a second
        # one (it would raise on brief path != present). No crash either.
        assert r.returncode in (0, 2), f"loop crashed:\n{r.stdout}\n{r.stderr}"
        out = (r.stdout + r.stderr)
        assert "resuming supervision, not duplicating" in out, out
        assert "start failed" not in out, out
        m2 = SUP.load_mission(mid)
        assert m2["phases"][0]["worker_task"] == tid, "must not rebind phase to a new task"
    finally:
        if old is None:
            os.environ.pop("HERMES_SUPERVISOR_DIR", None)
        else:
            os.environ["HERMES_SUPERVISOR_DIR"] = old


def test_exhausted_crash_closes_phase_no_busy_spin(iso_dir):
    """WORKER_CRASH [CANCEL] (attempts exhausted) must close the phase and
    exit, not busy-loop on a dead pid.

    Found live (2026-08-18, s3-autonomy-audit): a worker that dies instantly
    on spawn (bad model config, OOM at cold start, stub/child spawn failure)
    burns attempt 1..3, then evaluate_worker returns WORKER_CRASH +
    command=CANCEL. `supervise loop` handles command CANCEL as terminal, but
    the MISSION loop only handled RETRY/SUCCESS/CANCELLED — the CANCEL
    decision fell through every branch, the P-15 dead-pid wait broke
    instantly, and the loop spun on the dead worker: measured 12,683
    `WORKER_CRASH [CANCEL]` iterations in 12 s (~1,050/s, 100% CPU), phase
    left ACTIVE forever, mission never advanced, no fail-forward. A single
    broken worker wedged the whole mission into a busy spin instead of
    bounded retry -> harvest -> next phase.
    """
    stub = iso_dir / "crashstub"
    stub.write_text("#!/bin/sh\nexit 1\n")
    stub.chmod(0o755)
    env = dict(os.environ)
    env["HERMES_SUPERVISOR_DIR"] = str(iso_dir)
    env["HERMES_BIN"] = str(stub)
    mid = _mission_with_pending_phase("m-crash-exhaust")
    r = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "supervise", "mission",
         "loop", mid, "--every", "1", "--max-phases", "1",
         "--max-seconds", "12"],
        capture_output=True, text=True, timeout=30, env=env)
    out = r.stdout + r.stderr
    # deterministic spin guard: exhausted crash emits 1-2 lines when handled;
    # the buggy spin emits thousands in 12 s (measured 12,683).
    assert out.count("WORKER_CRASH [CANCEL]") < 500, \
        f"busy-spin: {out.count('WORKER_CRASH [CANCEL]')} CANCEL iterations\n{out[-2000:]}"
    m2 = SUP.load_mission(mid)
    assert m2 is not None and m2["phases"][0]["status"] == "FAILED", \
        f"phase not closed after exhausted crash: {m2['phases'][0] if m2 else m2}"


if __name__ == "__main__":
    import unittest
    unittest.main()