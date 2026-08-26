#!/usr/bin/env python3
"""Recovery-guard regression: corrupt worker ledger must NOT be silently
treated as missing, then silently overwritten by `supervise state-write`.

Failure class (found by s3-autonomy-audit, 2026-08-18): all worker state
lives in plain JSON ledgers under ~/.hermes-supervisor/. A torn/corrupt
ledger is a REAL long-horizon failure mode (disk hiccup, partial write,
manual edit). The mission side was fixed in create_mission (refuses the
overwrite, leaves the corrupt payload in place); the WORKER side was NOT:
`hermes supervise state-write` against a corrupt worker.json read it as
"never existed", accepted seq 1, and wrote a fresh ledger over the only
copy of prior state — silently destroying evidence at the supervisor's
own persistence boundary.

This test drives the REAL CLI end-to-end: fake task dir -> corrupt
worker.json -> state-write. FAILS on the unfixed tree (corrupt file
silently overwritten, rc 0, no corruption signal); PASSES after the fix
(state-write refuses, rc != 0, the corrupt payload stays byte-identical).
"""
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
    env["HERMES_SUPERVISOR_DIR"] = tempfile.mkdtemp(prefix="s3-corrupt-worker-")
    return env


def test_corrupt_worker_ledger_is_refused_by_state_write():
    env = _new_env()
    base = env["HERMES_SUPERVISOR_DIR"]
    task_dir = os.path.join(base, "tasks", "w1")
    os.makedirs(task_dir, exist_ok=True)
    p = os.path.join(task_dir, "worker.json")

    corrupt_payload = '{"task_id": "w1", "seq": 9, "status": "RUNNING", "TRUNCATED'
    with open(p, "w", encoding="utf-8") as f:
        f.write(corrupt_payload)
    before = open(p, "rb").read()
    assert os.path.exists(p), "fixture: ledger must still exist (torn)"

    r = _cli("supervise", "state-write", "w1",
             "--json", '{"status": "RUNNING", "phase": "TESTING"}', env=env)

    after = open(p, "rb").read()
    # The corrupt (prior state) file must not be silently replaced.
    assert after == before, (
        "corrupt worker ledger was silently overwritten by state-write "
        "(payload changed: %d -> %d bytes, rc=%s, stdout=%.160s)"
        % (len(before), len(after), r.returncode, r.stdout))

    # The corruption must be surfaced loudly (distinct REFUSED error, rc 1).
    assert r.returncode != 0, (
        "state-write accepted a write over a corrupt worker ledger "
        "(rc=0, stdout=%.160s)" % r.stdout)
    assert "refus" in (r.stdout + r.stderr).lower(), (
        "no corruption signal in state-write output: rc=%s stdout=%.160s"
        % (r.returncode, r.stdout))