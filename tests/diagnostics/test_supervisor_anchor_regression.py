"""Regression: worker-initiated state-write CANNOT anchor a worker (protected
keys); only supervisor-side attach may. Locks the anchor fix for the
hardening driver (silent no-op state-write anchor left scripted workers
unanchored; P10/P11 then refused their COMPLETE and long-duration loops
never terminated within the test window)."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess

REAL_BIN = "/home/hamza/.hermes/hermes-agent/venv/bin/hermes"


def _hermes(base, *args, timeout=120):
    env = dict(os.environ)
    env["HERMES_SUPERVISOR_DIR"] = str(base)
    return subprocess.run([REAL_BIN, *args], capture_output=True, text=True,
                          env=env, timeout=timeout)


def _state(base, task):
    p = pathlib.Path(base) / "tasks" / task / "worker.json"
    return json.loads(p.read_text()) if p.exists() else None


def test_worker_state_write_cannot_anchor(tmp_path):
    """A worker's own state-write must NEVER set worker_pid/started_at
    (protected keys). A forged worker must not be able to anchor itself."""
    r = _hermes(tmp_path, "supervise", "create", "--task", "t", "--task-id", "w")
    assert r.returncode == 0, r.stderr
    seq = _state(tmp_path, "w")["seq"]
    patch = {"status": "DIAGNOSING", "worker_pid": 999, "started_at": 12345.0}
    r = _hermes(tmp_path, "supervise", "state-write", "w",
                "--expect-seq", str(seq), "--json", json.dumps(patch))
    assert r.returncode == 0, r.stderr
    st = _state(tmp_path, "w")
    assert st["status"] == "DIAGNOSING"           # patch applied
    assert int(st.get("worker_pid") or 0) == 0    # protected: NOT applied
    assert st.get("started_at") is None           # protected: NOT applied


def test_supervisor_attach_anchors_and_stamps_run_id(tmp_path):
    """Supervisor-side attach (same CAS path as start/RETRY) sets
    worker_pid, started_at, run_id + identity."""
    r = _hermes(tmp_path, "supervise", "create", "--task", "t", "--task-id", "w")
    assert r.returncode == 0, r.stderr
    r = _hermes(tmp_path, "supervise", "attach", "w", "--pid", "4242")
    assert r.returncode == 0 and "attached" in r.stdout, r.stdout
    st = _state(tmp_path, "w")
    assert int(st.get("worker_pid") or 0) == 4242
    assert st.get("started_at")
    assert st.get("run_id")
    assert st.get("worker_identity")


def test_attach_missing_task_fails_cleanly(tmp_path):
    r = _hermes(tmp_path, "supervise", "attach", "ghost", "--pid", "1")
    assert r.returncode == 1
    assert "attach failed" in (r.stdout or "") or "attach failed" in (r.stderr or "")