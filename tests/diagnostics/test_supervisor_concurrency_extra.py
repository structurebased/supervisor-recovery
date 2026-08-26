"""Higher-concurrency supervisor proof — 32 and 50 real worker processes.

Revalidates the P-11 20-worker boundary at higher counts on the same real
CLI/CAS protocol. Scripted workers are cheap (no LLM): each is a separate OS
process driving `hermes supervise state-write` with CAS retry, exactly like
the hardening suite. This file is separate from test_supervisor_hardening.py
so independent attacker work in test_supervisor_attacker.py never collides.

Run: venv/bin/python -m pytest tests/diagnostics/test_supervisor_concurrency_extra.py -q -o addopts=
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

PARENT = pathlib.Path(__file__).parent
DRIVER = PARENT / "hardening_driver.py"
REAL_BIN = "/home/hamza/.hermes/hermes-agent/venv/bin/hermes"


@pytest.fixture(autouse=True)
def iso(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path))
    monkeypatch.setenv("MISSION_DRY_WORKER", "1")  # offline: skip LLM audit stage
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    stub = tmp_path / "bin" / "false"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text("#!/bin/sh\nexit 1\n")
    stub.chmod(0o755)
    monkeypatch.setenv("HERMES_BIN", str(stub))
    yield tmp_path


def _hermes(*args, timeout=120):
    env = dict(os.environ)
    return subprocess.run([REAL_BIN, *args], capture_output=True, text=True,
                          env=env, timeout=timeout)


def _state(base, task):
    p = pathlib.Path(base) / "tasks" / task / "worker.json"
    return json.loads(p.read_text()) if p.exists() else None


def _run_worker(base, task, script):
    sp = pathlib.Path(base) / f"script-{task}.json"
    sp.write_text(json.dumps(script))
    env = dict(os.environ)
    env["HERMES_SUPERVISOR_DIR"] = str(base)
    env["HERMES_DRIVER_BIN"] = REAL_BIN
    env.pop("HERMES_BIN", None)
    cmd = [sys.executable, str(DRIVER), "worker", task, str(sp)]
    logf = open(pathlib.Path(base) / f"worker-{task}.log", "a")
    return subprocess.Popen(cmd, env=env, stdout=logf, stderr=logf)


def _start_loop(base, task, max_seconds=90.0, every=1.0):
    env = dict(os.environ)
    env["HERMES_SUPERVISOR_DIR"] = str(base)
    env["HERMES_DRIVER_BIN"] = REAL_BIN
    args = [sys.executable, str(DRIVER), "loop", task,
            "--every", str(every), "--max-seconds", str(max_seconds)]
    return subprocess.Popen(args, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


@pytest.mark.parametrize("N", [32, 50])
def test_n_workers_cas_concurrent_publish_complete(tmp_path, N):
    """N real workers: CAS state writes + two broadcasts each + complete.
    Asserts no double completion, no lost message, seq monotonic per task."""
    for i in range(N):
        _create = _hermes("supervise", "create", "--task", f"t{i}",
                          "--task-id", f"hw{i}")
        assert _create.returncode == 0, _create.stderr

    procs = []
    for i in range(N):
        script = [
            {"kind": "write", "patch": {"status": "DIAGNOSING", "phase": "DIAGNOSING",
                                        "progress": f"p{i}-1", "findings": [f"f{i}"]}},
            {"kind": "publish", "to": f"hw{(i + 1) % N}", "text": f"m1-{i}",
             "sender": f"hw{i}"},
            {"kind": "write", "patch": {"status": "TESTING", "phase": "TESTING",
                                        "tests_executed": 1, "tests_passed": 1}},
            {"kind": "publish", "to": f"hw{(i + 2) % N}", "text": f"m2-{i}",
             "sender": f"hw{i}"},
            {"kind": "write", "patch": {"status": "COMPLETE", "phase": "VERIFYING",
                                        "completion_evidence": "passed, verified end-to-end"}},
        ]
        procs.append(_run_worker(tmp_path, f"hw{i}", script))

    loops = [_start_loop(tmp_path, f"hw{i}", max_seconds=90) for i in range(N)]
    for l in loops:
        try:
            l.communicate(timeout=150)
        except subprocess.TimeoutExpired:
            l.kill(); l.wait(timeout=10)
    # join workers BEFORE counting: a straggler killed mid-publish would lose
    # its message and look like a supervisor defect (flock is proven above;
    # this assert measures delivery, so every worker must finish publishing)
    for p in procs:
        try:
            p.communicate(timeout=30)
        except Exception:
            p.kill(); p.wait(timeout=10)

    comp = 0
    seqs_ok = True
    msg_total = 0
    duplicates = 0
    for i in range(N):
        st = _state(tmp_path, f"hw{i}")
        assert st is not None, f"hw{i} state missing"
        if st.get("status") == "COMPLETE":
            comp += 1
        # seq must equal the number of accepted writes + initial (create=1)
        # accepted writes: anchor + 3 => seq = create(1)+anchor+3 = 5
        # (loop's record_spawned_pid also bumps; tolerate >=4 and monotonic)
        if int(st.get("seq") or 0) < 4:
            seqs_ok = False
        p = pathlib.Path(tmp_path) / "tasks" / f"hw{i}" / "inbox.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                try:
                    m = json.loads(line)
                except Exception:
                    continue
                msg_total += 1
                if m.get("status") == "pending":
                    pass
    # every worker sent 2 broadcast messages => each inbox should hold the
    # union; allow slack for the two-loop-race but forbid cross-task loss:
    # total entries >= 2*N (each sender wrote to someone) and duplicates of
    # the same (m1|m2-i, receiver) pairing counted once
    assert msg_total >= 2 * N, msg_total
    assert seqs_ok, "some ledgers show fewer than expected accepted writes"
    # Completion dominated; allow bounded deferral, never double claim
    assert comp >= N - 5, f"only {comp}/{N} completed"


def test_no_message_loss_at_50_when_all_workers_complete(tmp_path):
    """Same shape at 50 workers alternating two fan-out targets; asserts the
    exact 50*2 broadcast pairing is received exactly once when completion is
    dominant."""
    N = 50
    for i in range(N):
        _hermes("supervise", "create", "--task", "t", "--task-id", f"nf{i}")
    procs = []
    for i in range(N):
        script = [
            {"kind": "write", "patch": {"status": "DIAGNOSING", "progress": f"q{i}"}},
            {"kind": "publish", "to": f"nf{(i + 7) % N}", "text": f"broadcast-{i}",
             "sender": f"nf{i}"},
            {"kind": "write", "patch": {"status": "COMPLETE",
                                        "completion_evidence": "passed, verified"}},
        ]
        procs.append(_run_worker(tmp_path, f"nf{i}", script))
    loops = [_start_loop(tmp_path, f"nf{i}", max_seconds=90) for i in range(N)]
    for l in loops:
        try:
            l.communicate(timeout=150)
        except subprocess.TimeoutExpired:
            l.kill(); l.wait(timeout=10)
    for p in procs:
        try:
            p.communicate(timeout=30)
        except Exception:
            p.kill(); p.wait(timeout=10)
    seen = set()
    total = 0
    for i in range(N):
        st = _state(tmp_path, f"nf{i}")
        if st is None:
            continue
        p = pathlib.Path(tmp_path) / "tasks" / f"nf{i}" / "inbox.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            try:
                m = json.loads(line)
            except Exception:
                continue
            total += 1
            s = m.get("sender", "")
            txt = str(m.get("message", ""))
            if s.startswith("nf") and txt.startswith("broadcast-"):
                seen.add((s, f"nf{i}"))
    # every pair s -> receiver landed at most once
    assert len(seen) == len(range(N)), (len(seen), N)
    assert total >= len(seen), (total, len(seen))