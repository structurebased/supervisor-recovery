#!/usr/bin/env python3
"""P-18 regressions: evaluate_worker fresh-spawn grace for live workers.

FAIL on parent: parent evaluate_worker gives a freshly-started LIVE worker
with an unchanged CREATED ledger NO_PROGRESS/REASSESS (detect_progress
ignores process liveness) — measured live 2026-08-17: real worker ran
6m11s/8 tool calls while the loop spammed CREATED->NO_PROGRESS per second.
"""
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


def _hermes(base, *args, timeout=60):
    env = dict(os.environ)
    env["HERMES_SUPERVISOR_DIR"] = str(base)
    return subprocess.run([REAL_BIN, *args], capture_output=True, text=True,
                          env=env, timeout=timeout)


def _live_state(tmp_path, started_secs: float):
    r = _hermes(tmp_path, "supervise", "create", "--task", "t",
                "--task-id", "w")
    assert r.returncode == 0, r.stderr
    st = S.load_worker  # noqa: B018
    p = pathlib.Path(tmp_path) / "tasks" / "w" / "worker.json"
    import json
    st = json.loads(p.read_text())
    st["status"] = "CREATED"  # unchanged ledger = worker hasn't written yet
    st["started_at"] = time.time() - started_secs
    st["worker_pid"] = os.getpid()
    st["last_activity_at"] = time.time() - started_secs
    p.write_text(json.dumps(st, default=str))
    return st


def test_live_fresh_spawn_not_no_progress(tmp_path):
    """A worker started 10s ago, pid alive, ledger still CREATED: must NOT
    be NO_PROGRESS — it is in fresh-spawn grace (equivalent to the
    watchdog's STALL_START_GRACE_SECONDS)."""
    st = _live_state(tmp_path, started_secs=10)
    d = S.evaluate_worker(st, previous=st, now=time.time(), pid=os.getpid())
    assert d.verdict not in ("NO_PROGRESS", "WORKER_CRASH"), d.as_dict()
    assert d.verdict == "NO_VERDICT_YET", d.as_dict()


def test_live_fresh_spawn_really_uses_alive_pid(tmp_path):
    """Dead pid must NOT get the grace (crash semantics preserved)."""
    st = _live_state(tmp_path, started_secs=10)
    d = S.evaluate_worker(st, previous=st, now=time.time(), pid=99999999)
    assert d.verdict == "WORKER_CRASH", d.as_dict()


def test_dead_worker_still_crash(tmp_path):
    """A dead worker after grace is still a crash (no false keep-alive)."""
    st = _live_state(tmp_path, started_secs=2000)  # well past grace
    st["last_activity_at"] = time.time() - 2000
    p = pathlib.Path(tmp_path) / "tasks" / "w" / "worker.json"
    import json
    p.write_text(json.dumps(st, default=str))
    d = S.evaluate_worker(st, previous=st, now=time.time(), pid=99999999)
    assert d.verdict == "WORKER_CRASH", d.as_dict()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])