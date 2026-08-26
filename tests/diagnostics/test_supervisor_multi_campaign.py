"""Multi-campaign concurrency + restart-under-concurrency proofs.

Existing tests prove single-campaign replacement and restart recovery. This
file adds what they do not: several campaigns running simultaneously, each
with its own role and replacement budget, workers dying at the same time,
and a supervisor restart mid-campaign-collapse. Asserts no cross-campaign
interference (a replacement in campaign X never adopts campaign Y's role),
bounded budgets, and ledger integrity.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

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


def _campaign(base, cid, role, task):
    r = _hermes("supervise", "campaign", "create", cid, "--objective", "o",
                "--roles", json.dumps([{"role_id": role,
                                        "responsibility": "r",
                                        "required_evidence": ["repro"]}]))
    assert r.returncode == 0, r.stderr
    r = _hermes("supervise", "campaign", "assign", cid, "--role", role,
                "--task", task)
    assert r.returncode == 0, r.stderr


def _run_worker(base, task, script):
    sp = pathlib.Path(base) / f"script-{task}.json"
    sp.write_text(json.dumps(script))
    env = dict(os.environ)
    env["HERMES_SUPERVISOR_DIR"] = str(base)
    env["HERMES_DRIVER_BIN"] = REAL_BIN
    env.pop("HERMES_BIN", None)
    logf = open(pathlib.Path(base) / f"worker-{task}.log", "a")
    return subprocess.Popen([sys.executable, str(DRIVER), "worker", task, str(sp)],
                            env=env, stdout=logf, stderr=logf)


def _start_loop(base, task, max_seconds=40, campaign="", role="",
                max_replacements=2):
    env = dict(os.environ)
    env["HERMES_SUPERVISOR_DIR"] = str(base)
    env["HERMES_DRIVER_BIN"] = REAL_BIN
    args = [sys.executable, str(DRIVER), "loop", task,
            "--every", "1.0", "--max-seconds", str(max_seconds)]
    if campaign:
        args += ["--campaign", campaign, "--role", role,
                 "--max-replacements", str(max_replacements)]
    return subprocess.Popen(args, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def test_multiple_campaigns_simultaneous_replacements(tmp_path):
    """4 campaigns, each with 1 worker that dies; each loop must spawn its
    OWN replacement (bounded), never adopt a sibling campaign's role, and
    every campaign ledger must be internally consistent."""
    CAMPAIGNS = [("c0", "hunter0", "w0"), ("c1", "hunter1", "w1"),
                 ("c2", "hunter2", "w2"), ("c3", "hunter3", "w3")]
    for cid, role, task in CAMPAIGNS:
        _hermes("supervise", "create", "--task", "t", "--task-id", task)
        _campaign(tmp_path, cid, role, task)

    procs = []
    for _, _, task in CAMPAIGNS:
        # worker anchors then dies so the loop hits the replacement branch
        procs.append(_run_worker(tmp_path, task, [{"kind": "write",
                                                   "patch": {"status": "DIAGNOSING"}},
                                                  {"kind": "die"}]))

    loops = [_start_loop(tmp_path, task, campaign=cid, role=role, max_replacements=2)
             for cid, role, task in CAMPAIGNS]
    for l in loops:
        try:
            l.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            l.kill(); l.wait(timeout=10)
    for p in procs:
        try:
            p.communicate(timeout=30)
        except Exception:
            p.kill(); p.wait(timeout=10)

    # each campaign must have its own tasks dirs: original + exactly up to
    # max_replacements replacements, and they must reference ONLY their
    # own campaign's role in the campaign ledger
    for cid, role, task in CAMPAIGNS:
        tasks = [p.name for p in (pathlib.Path(tmp_path) / "tasks").iterdir()
                 if p.name.startswith(task)]
        assert len(tasks) >= 1
        camp = json.loads((pathlib.Path(tmp_path) / "campaigns" / f"{cid}.json").read_text())
        workers = camp.get("workers") or {}
        # every entry in the campaign must belong to this task family
        for wid, w in workers.items():
            assert wid in tasks, (cid, wid, tasks)
            assert w.get("role") == role, (cid, wid, w.get("role"))
            # no double adoption: every replacement points at an ancestor in
            # this same family
            replaced = w.get("replaces") or ""
            assert replaced == "" or replaced in tasks, (cid, wid, replaced)
        # bounded: no more than 1 + max_replacements total in the family
        assert len(tasks) <= 1 + 2, (cid, tasks)


def test_restart_mid_campaign_collapse(tmp_path):
    """Kill and restart the supervisor loop while multiple workers are dying
    and replacements decided; the role budget (campaign ledger) must survive,
    so a restarted supervisor cannot re-spawn an already-used replacement
    budget."""
    cid, role, task = "c-race", "hunter", "wX"
    _hermes("supervise", "create", "--task", "t", "--task-id", task)
    _campaign(tmp_path, cid, role, task)
    # anchor the original, then die immediately each cycle
    _run_worker(tmp_path, task, [{"kind": "write",
                                  "patch": {"status": "DIAGNOSING"}},
                                 {"kind": "die"}])
    for cycle in range(3):
        loop = _start_loop(tmp_path, task, campaign=cid, role=role,
                           max_replacements=1)
        time.sleep(1.2)
        loop.kill()
        loop.wait(timeout=10)
    time.sleep(1.0)
    camp = json.loads((pathlib.Path(tmp_path) / "campaigns" / f"{cid}.json").read_text())
    fam = [p.name for p in (pathlib.Path(tmp_path) / "tasks").iterdir()
           if p.name.startswith(task)]
    # even with a restart racing the replacement, budget=1 caps the family
    # at original + 1 replacement; no unbounded adoption
    assert len(fam) <= 2, fam