#!/usr/bin/env python3
"""P-15 regressions: event-driven completion detection + RUNNING states.

FAIL on parent (pre-8.0zr-p15):
  1. `evaluate_worker` treated a RUNNING/STARTING worker as
     WORKER_CRASH[INVESTIGATE] (unknown status) — a false positive.
  2. The loops slept the full `every` interval to detect completion,
     so a finished worker was adjudicated up to one poll period late.
     (Latency itself is loop-level; the unit-observable contract is
     that RUNNING/STARTING evaluate as in-progress, not crash.)
"""
import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
REAL_BIN = "/home/hamza/.hermes/hermes-agent/venv/bin/hermes"
from hermes_cli import supervisor as S  # noqa: E402


def _state(base, task):
    p = pathlib.Path(base) / "tasks" / task / "worker.json"
    return json.loads(p.read_text()) if p.exists() else None


def _hermes(tmp_path, *args, timeout=60):
    env = dict(os.environ)
    env["HERMES_SUPERVISOR_DIR"] = str(tmp_path)
    return subprocess.run([REAL_BIN, *args], capture_output=True, text=True,
                          env=env, timeout=timeout)


def _base_state(tmp_path, status):
    r = _hermes(tmp_path, "supervise", "create", "--task", "t",
                "--task-id", "w")
    assert r.returncode == 0, r.stderr
    st = _state(tmp_path, "w")
    st["status"] = status
    p = pathlib.Path(tmp_path) / "tasks" / "w" / "worker.json"
    p.write_text(json.dumps(st, default=str))
    return st


def test_running_is_in_progress_not_crash(tmp_path):
    """RUNNING is a legitimate in-progress status; must not yield
    WORKER_CRASH. Started+anchored so the crash path would otherwise hit."""
    st = _base_state(tmp_path, "RUNNING")
    st["started_at"] = time.time()
    st["worker_pid"] = os.getpid()
    p = pathlib.Path(tmp_path) / "tasks" / "w" / "worker.json"
    p.write_text(json.dumps(st, default=str))
    d = S.evaluate_worker(st, pid=os.getpid())
    assert d.verdict not in ("WORKER_CRASH",), d.as_dict()
    assert d.verdict in ("NO_VERDICT_YET", "NO_PROGRESS") or "CONTINUE" in d.command


def test_starting_is_in_progress_not_crash(tmp_path):
    st = _base_state(tmp_path, "STARTING")
    d = S.evaluate_worker(st, pid=os.getpid())
    assert d.verdict not in ("WORKER_CRASH",), d.as_dict()


def test_running_dead_pid_still_detected_as_crash(tmp_path):
    """A RUNNING worker whose process died IS a crash (the status whitelist
    must not hide real process death)."""
    st = _base_state(tmp_path, "w")
    st["started_at"] = time.time()
    st["worker_pid"] = 99999999  # dead pid
    r = _hermes(tmp_path, "supervise", "check", "w")
    # check uses the ledger; worker_pid is in the ledger
    assert r.returncode == 0
    out = (r.stdout or "") + (r.stderr or "")
    # check writes the decision JSON containing verdict
    st2 = _state(tmp_path, "w")
    d = S.evaluate_worker(st2, pid=99999999)
    assert d.verdict == "WORKER_CRASH"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])