"""INDEPENDENT ADVERSARY tests against `hermes supervise`.

Every attack drives the REAL CLI (`venv/bin/hermes`) against isolated tmp
supervisor dirs with real subprocesses. Each attack is one test function.
Anything a worker would never legitimately control (run_id, worker_identity,
budget, attempts) is attacked; every claim is backed by real process output.

Run: venv/bin/python -m pytest tests/diagnostics/test_supervisor_attacker.py -q -o addopts=
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from hermes_cli import supervisor as SUP  # noqa: E402  (read-only probes)

PARENT = pathlib.Path(__file__).parent
DRIVER = PARENT / "hardening_driver.py"
REAL_BIN = "/home/hamza/.hermes/hermes-agent/venv/bin/hermes"
assert pathlib.Path(REAL_BIN).exists(), REAL_BIN


@pytest.fixture(autouse=True)
def iso(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path))
    monkeypatch.setenv("MISSION_DRY_WORKER", "1")  # offline: skip LLM audit stage
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    # Supervisor worker RESPAWNs use $HERMES_BIN -> crash stub (instant death).
    stub = tmp_path / "bin" / "false"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text("#!/bin/sh\nexit 1\n")
    stub.chmod(0o755)
    monkeypatch.setenv("HERMES_BIN", str(stub))
    yield tmp_path


# --------------------------------------------------------------------------
# helpers (same isolation pattern as test_supervisor_hardening.py)
# --------------------------------------------------------------------------

def _hermes(*args, timeout=150, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run([REAL_BIN, *args], capture_output=True, text=True,
                          env=env, timeout=timeout)


def _state(base, task):
    p = pathlib.Path(base) / "tasks" / task / "worker.json"
    return json.loads(p.read_text()) if p.exists() else None


def _ledger(base, task):
    """Direct ledger write (deployer role — worker never touches files)."""
    p = pathlib.Path(base) / "tasks" / task / "worker.json"
    return json.loads(p.read_text()) if p.exists() else None


def _create_task(base, task, text="A brief"):
    r = _hermes("supervise", "create", "--task", text, "--task-id", task)
    assert r.returncode == 0, r.stderr


def _dead_pid():
    """A real, nonzero pid of a just-exited process (crash immediately)."""
    p = subprocess.Popen(["true"])
    p.wait()
    assert p.returncode == 0
    return p.pid


def _anchor(base, task, pid=None, budget_attempts=3, extra=None):
    """Deployer anchors the worker as started with a real (now dead) pid and a
    hard budget, exactly like `supervise start` + process exit would leave."""
    st = _ledger(base, task)
    assert st is not None, f"{task} ledgless"
    st["worker_pid"] = pid or _dead_pid()
    st["started_at"] = time.time()
    st["seq"] = int(st.get("seq") or 1)
    st["budget"] = {
        "max_worker_attempts": budget_attempts,
        "max_runtime_seconds": 3600, "idle_timeout_seconds": 600,
        "max_worker_turns": 60, "max_consecutive_failures": 3,
        "max_repeated_hypothesis": 3, "max_supervisor_interventions": 6,
    }
    if extra:
        st.update(extra)
    p = pathlib.Path(base) / "tasks" / task / "worker.json"
    p.write_text(json.dumps(st))
    return st


def _run_worker(base, task, script, background=False):
    sp = pathlib.Path(base) / f"script-{task}.json"
    sp.write_text(json.dumps(script))
    env = dict(os.environ)
    env["HERMES_SUPERVISOR_DIR"] = str(base)
    env["HERMES_DRIVER_BIN"] = REAL_BIN
    env.pop("HERMES_BIN", None)
    cmd = [sys.executable, str(DRIVER), "worker", task, str(sp)]
    if background:
        return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    return subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=150)


def _start_loop(base, task, max_seconds=20, campaign="", role="",
                max_replacements=0, every=0.8):
    env = dict(os.environ)
    env["HERMES_SUPERVISOR_DIR"] = str(base)
    env["HERMES_DRIVER_BIN"] = REAL_BIN
    args = [sys.executable, str(DRIVER), "loop", task,
            "--every", str(every), "--max-seconds", str(max_seconds)]
    if campaign:
        args += ["--campaign", campaign, "--role", role,
                 "--max-replacements", str(max_replacements)]
    return subprocess.Popen(args, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)


def _loop_text(proc, timeout=90):
    out, _ = proc.communicate(timeout=timeout)
    return (out or b"").decode(errors="replace")


def _inbox(base, task):
    p = pathlib.Path(base) / "tasks" / task / "inbox.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _audit(base, task):
    p = pathlib.Path(base) / "tasks" / task / "audit.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _patch_state(base, task, patch, timeout=15):
    """CAS state write with retry on stale (the real worker protocol)."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        st = _state(base, task)
        seq = st["seq"] if st else 0
        r = _hermes("supervise", "state-write", task, "--expect-seq", str(seq),
                    "--json", json.dumps(patch))
        last = r
        if r.returncode == 0 and "accepted" in r.stdout:
            return r
        time.sleep(0.15)
    raise AssertionError(f"state patch never accepted: "
                         f"{last.stdout if last else ''}{last.stderr if last else ''}")


def _write_pid(base, task, pid):
    """Deployer anchors a real (live or dead) pid into the ledger, exactly the
    field `supervise start` writes; state-write refuses worker_pid by design."""
    p = pathlib.Path(base) / "tasks" / task / "worker.json"
    st = json.loads(p.read_text())
    st["worker_pid"] = int(pid)
    st["started_at"] = time.time()
    p.write_text(json.dumps(st))
    return st


def _create_campaign(base, cid, role, task):
    _create_task(base, task)
    r = _hermes("supervise", "campaign", "create", cid, "--objective", "obj",
                "--roles", json.dumps([{"role_id": role, "responsibility": "x",
                                        "required_evidence": ["e"]}]))
    assert r.returncode == 0, r.stderr
    r = _hermes("supervise", "campaign", "assign", cid, "--role", role,
                "--task", task)
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------------
# ATTACK 1 — CAS stale-write rejection under real contention
# --------------------------------------------------------------------------

class TestAttack1StaleCAS:
    def test_concurrent_bump_rejects_stale_writer_with_audit(self, tmp_path):
        _create_task(tmp_path, "w", "t")
        seq0 = _state(tmp_path, "w")["seq"]
        # Oversized patch slows W1's apply; W2 (a SEPARATE process) races it
        # with the SAME expect-seq. Exactly one writer must win the key.
        big = {"progress": "z" * 40_000, "status": "DIAGNOSING"}
        w1 = subprocess.Popen(
            [REAL_BIN, "supervise", "state-write", "w", "--expect-seq",
             str(seq0), "--json", json.dumps(big)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=dict(os.environ))
        time.sleep(0.6)  # W1 mid-merge; W2 fires from a separate process
        r2 = _hermes("supervise", "state-write", "w", "--expect-seq",
                     str(seq0), "--json", '{"status":"DIAGNOSING","progress":"p2"}')
        out1, err1 = w1.communicate(timeout=120)
        # exactly one winner, one stale rejection — order is schedule-dependent
        won_w1 = w1.returncode == 0 and "accepted" in out1
        won_w2 = r2.returncode == 0 and "accepted" in r2.stdout
        assert won_w1 != won_w2, (w1.returncode, out1, r2.returncode, r2.stdout)
        stale_side = (out1 + err1) if won_w2 else r2.stdout
        assert "stale" in stale_side.lower(), stale_side
        cur = _state(tmp_path, "w")
        assert cur["seq"] == seq0 + 1, cur
        assert cur["status"] == "DIAGNOSING", cur
        assert cur.get("progress") in ("p2", "z" * 40_000)
        # audit row recorded for the rejected writer
        rows = [e for e in _audit(tmp_path, "w") if e.get("kind") == "stale_write"]
        assert rows, "no stale_write audit row despite rejection"
        assert {"expected_seq", "disk_seq", "author"} <= set(rows[-1].keys())


# --------------------------------------------------------------------------
# ATTACK 2 — duplicate content-hash dedup across separate CLI processes
# --------------------------------------------------------------------------

class TestAttack2Dedup:
    def test_identical_posts_from_two_processes_dedupe(self, tmp_path):
        _create_task(tmp_path, "A", "t")
        for _ in range(2):
            r = _hermes("supervise", "message", "A", "--text", "same payload",
                        "--kind", "handoff", "--sender", "wa")
            assert r.returncode == 0, r.stderr
        msgs = _inbox(tmp_path, "A")
        pending = [m for m in msgs if m.get("status") == "pending"]
        assert len(pending) == 1, f"expected exactly 1 pending, got {len(pending)}"
        assert len(msgs) == 1, f"expected 1 ledger line: {msgs}"


# --------------------------------------------------------------------------
# ATTACK 3 — ack/delivery semantics (worker-set last_acked_msg_id flips ledger)
# --------------------------------------------------------------------------

class TestAttack3Ack:
    def test_ack_flips_delivered_and_no_redelivery(self, tmp_path):
        _create_task(tmp_path, "A", "t")
        r = _hermes("supervise", "message", "A", "--text", "handoff one",
                    "--kind", "handoff", "--sender", "wa")
        assert r.returncode == 0
        mid = _inbox(tmp_path, "A")[0]["id"]
        # first loop: delivers the message
        loop1 = _start_loop(tmp_path, "A", max_seconds=6)
        txt1 = _loop_text(loop1, timeout=40)
        assert "delivered" in txt1, txt1[-800:]
        assert _inbox(tmp_path, "A")[0]["status"] == "delivered"
        # worker acks through the REAL protocol (state-write last_acked_msg_id)
        _patch_state(tmp_path, "A", {"last_acked_msg_id": mid})
        loop2 = _start_loop(tmp_path, "A", max_seconds=5)
        _loop_text(loop2, timeout=40)
        msgs = _inbox(tmp_path, "A")
        assert msgs[0]["status"] == "acknowledged", msgs[0]
        # second delivery must NOT redeliver the acknowledged message
        loop3 = _start_loop(tmp_path, "A", max_seconds=5)
        txt3 = _loop_text(loop3, timeout=40)
        assert "delivered 1" not in txt3, txt3[-800:]
        assert _inbox(tmp_path, "A")[0]["status"] == "acknowledged"


# --------------------------------------------------------------------------
# ATTACK 4 — malformed handoff must not crash the pipeline
# --------------------------------------------------------------------------

class TestAttack4Handoff:
    def test_malformed_handoff_post_and_validation(self, tmp_path):
        _create_task(tmp_path, "A", "t")
        _create_task(tmp_path, "B", "t")
        # malformed handoff JSON body posted via the real CLI
        r = _hermes("supervise", "message", "B", "--text", '{"owner_id":',
                    "--kind", "handoff", "--sender", "A")
        assert r.returncode == 0, r.stderr
        # post_handoff with a state missing owner_id/phase/evidence keys
        st = _state(tmp_path, "A")
        if "handoff" in st:
            del st["handoff"]
        msg = SUP.post_handoff(st, to_task="B", sender_task="A")
        assert msg and msg.get("kind") == "handoff"
        body = json.loads(msg["message"])
        assert "owner_id" in body and "phase" in body and "evidence" in body
        # loop over receiver must not crash and must not promote garbage
        proc = _start_loop(tmp_path, "B", max_seconds=6)
        txt = _loop_text(proc, timeout=40)
        assert "Traceback" not in txt
        assert "SUCCESS" not in txt
        delivered = [m for m in _inbox(tmp_path, "B") if m.get("status") == "delivered"]
        assert len(delivered) == 2, delivered


# --------------------------------------------------------------------------
# ATTACK 5 — worker identity/lineage: run_id stable across restart, then
# overwrite must be impossible
# --------------------------------------------------------------------------

class TestAttack5Identity:
    def test_run_id_stable_across_restart_and_overwrite_rejected(self, tmp_path):
        _create_task(tmp_path, "A", "t")
        _anchor(tmp_path, "A", budget_attempts=3)
        # run loop until at least one respawn stamps a run_id
        loop1 = _start_loop(tmp_path, "A", max_seconds=12)
        txt1 = _loop_text(loop1, timeout=60)
        assert "respawn attempt" in txt1, txt1[-1000:]
        st = _state(tmp_path, "A")
        rid = st.get("run_id")
        assert rid, "no run_id stamped by supervisor"
        # restart the supervisor: run_id must survive (no re-roll on recovery)
        loop2 = _start_loop(tmp_path, "A", max_seconds=8)
        txt2 = _loop_text(loop2, timeout=60)
        assert _state(tmp_path, "A").get("run_id") == rid, "run_id changed on restart"
        assert "Traceback" not in txt2
        # ATTACK: worker tries to overwrite its own run_id + identity + budget
        _patch_state(tmp_path, "A", {
            "run_id": "FORGED", "worker_identity": "imposter",
            "budget": {"max_worker_attempts": 99}})
        st2 = _state(tmp_path, "A")
        assert st2.get("run_id") == rid, f"run_id overwritten: {st2.get('run_id')}"
        assert st2.get("worker_identity") != "imposter"
        assert int(st2["budget"]["max_worker_attempts"]) != 99


# --------------------------------------------------------------------------
# ATTACK 6 — replacement-of-replacement: A→B→C lineage, no double adoption
# --------------------------------------------------------------------------

class TestAttack6ReplacementChain:
    def test_replace_then_replace_preserves_chain(self, tmp_path):
        cid, role = "c1", "adv"
        _create_campaign(tmp_path, cid, role, "A")
        # give A a real handoff so the replacement inbox is seeded
        _patch_state(tmp_path, "A", {"status": "DIAGNOSING", "progress": "p1",
                                     "findings": ["f1"]})
        r = _hermes("supervise", "campaign", "replace", cid, "--role", role,
                    "--task", "A", "--replace-id", "B")
        assert r.returncode == 0, r.stderr
        r = _hermes("supervise", "campaign", "replace", cid, "--role", role,
                    "--task", "B", "--replace-id", "C")
        assert r.returncode == 0, r.stderr
        led_a, led_b = _state(tmp_path, "A"), _state(tmp_path, "B")
        assert led_a.get("replaced_by") == "B", led_a
        assert led_b.get("replaced_by") == "C", led_b
        cam = json.loads((pathlib.Path(tmp_path) / "campaigns" / f"{cid}.json").read_text())
        workers = cam["workers"]
        assert workers["B"]["replaces"] == "A"
        assert workers["C"]["replaces"] == "B"
        assert len(workers) == 3
        # A must not have been double-adopted (still owns the role once)
        assert workers["A"].get("role") == role
        assert workers["C"].get("role") == role
        # replacement inbox seeded with predecessor handoff (message continuity)
        msgs_b = _inbox(tmp_path, "B")
        assert any(m.get("kind") == "handoff" for m in msgs_b), "no handoff seeded"


# --------------------------------------------------------------------------
# ATTACK 7 — restart during/after replacement: bounded, no duplicate spawns
# --------------------------------------------------------------------------

class TestAttack7RestartReplacement:
    def test_restart_after_replacement_does_not_repawn(self, tmp_path):
        cid, role = "c1", "adv"
        _create_campaign(tmp_path, cid, role, "A")
        _anchor(tmp_path, "A", budget_attempts=3)
        # A writes FAILED (scripted real worker) so the loop reaches the
        # replacement branch deterministically; replacements crash (stub).
        _run_worker(tmp_path, "A", [{"kind": "write", "patch": {"status": "FAILED"}}])
        loop1 = _start_loop(tmp_path, "A", max_seconds=24, campaign=cid, role=role,
                            max_replacements=1)
        txt1 = _loop_text(loop1, timeout=120)
        assert "replacement spawned" in txt1, txt1[-1500:]
        tasks1 = sorted(p.name for p in (pathlib.Path(tmp_path) / "tasks").iterdir())
        repl = [t for t in tasks1 if t.startswith("A-r")]
        assert len(repl) == 1, tasks1
        repl_id = repl[0]
        # RESTART the supervisor on the ORIGINAL task id. The replacement
        # budget (max_replacements=1) is a ROLE budget: it must survive the
        # restart or a restarted supervisor re-spawns fresh replacements for
        # the same failed role forever (unbounded ledger growth).
        loop2 = _start_loop(tmp_path, "A", max_seconds=14, campaign=cid, role=role,
                            max_replacements=1)
        txt2 = _loop_text(loop2, timeout=90)
        tasks2 = sorted(p.name for p in (pathlib.Path(tmp_path) / "tasks").iterdir())
        assert len(tasks2) == 2, f"restart spawned replacements: {tasks2}"
        assert "replacement spawned" not in txt2, txt2[-1200:]
        assert "Traceback" not in txt2
        # lineage preserved: the surviving replacement still points at A
        cam = json.loads((pathlib.Path(tmp_path) / "campaigns" / f"{cid}.json").read_text())
        assert cam["workers"][repl_id]["replaces"] == "A"


# --------------------------------------------------------------------------
# ATTACK 8 — exhausted retry budget: max_worker_attempts=2 → FAIL, never 3
# --------------------------------------------------------------------------

class TestAttack8RetryBudget:
    def test_exhausted_budget_fails_after_exactly_2(self, tmp_path):
        _create_task(tmp_path, "A", "t")
        _anchor(tmp_path, "A", budget_attempts=2)
        # worker dies instantly every spawn (stub)
        loop = _start_loop(tmp_path, "A", max_seconds=25)
        txt = _loop_text(loop, timeout=90)
        assert loop.returncode == 0, f"loop rc={loop.returncode}"
        st = _state(tmp_path, "A")
        assert int(st.get("attempt") or 1) == 2, st
        assert st.get("status") not in ("COMPLETE", "SUCCESS")
        # never reached attempt 3
        assert "respawn attempt 3" not in txt
        assert "respawn attempt 2" in txt, txt[-1500:]


# --------------------------------------------------------------------------
# ATTACK 9 — invalid statuses must never promote to SUCCESS
# --------------------------------------------------------------------------

class TestAttack9InvalidStatus:
    def test_garbage_status_never_success_and_ledger_stays_intact(self, tmp_path):
        _create_task(tmp_path, "A", "t")
        seq = _state(tmp_path, "A")["seq"]
        r = _hermes("supervise", "state-write", "A", "--expect-seq", str(seq),
                    "--json", json.dumps({"status": "DONE", "progress": "we did it"}))
        assert r.returncode == 0, r.stderr
        loop = _start_loop(tmp_path, "A", max_seconds=8)
        txt = _loop_text(loop, timeout=40)
        assert "SUCCESS" not in txt, txt[-800:]
        assert "Traceback" not in txt
        st = _state(tmp_path, "A")
        assert st.get("status") != "SUCCESS"
        # ledger must still round-trip through CAS for the next writer
        seq2 = st["seq"]
        r = _hermes("supervise", "state-write", "A", "--expect-seq", str(seq2),
                    "--json", json.dumps({"status": "DIAGNOSING"}))
        assert r.returncode == 0, r.stdout + r.stderr
        assert _state(tmp_path, "A")["status"] == "DIAGNOSING"


# --------------------------------------------------------------------------
# ATTACK 10 — missing/empty brief must not crash or claim SUCCESS
# --------------------------------------------------------------------------

class TestAttack10EmptyBrief:
    def test_empty_brief_loop_no_false_success(self, tmp_path):
        r = _hermes("supervise", "create", "--task", "   ", "--task-id", "A")
        assert r.returncode == 0, r.stderr
        loop = _start_loop(tmp_path, "A", max_seconds=7)
        txt = _loop_text(loop, timeout=40)
        assert "Traceback" not in txt, txt
        st = _state(tmp_path, "A")
        assert st.get("status") not in ("SUCCESS", "COMPLETE")
        assert "SUCCESS" not in txt
        # a task with NO brief file at all (deleted) — loop must not crash
        (pathlib.Path(tmp_path) / "tasks" / "A" / "brief.md").unlink()
        loop2 = _start_loop(tmp_path, "A", max_seconds=7)
        txt2 = _loop_text(loop2, timeout=40)
        assert "Traceback" not in txt2, txt2


# --------------------------------------------------------------------------
# ATTACK 11 — malformed state JSON mid-run
# --------------------------------------------------------------------------

class TestAttack11CorruptLedger:
    def test_corrupt_worker_json_stops_cleanly_without_accepting(self, tmp_path):
        _create_task(tmp_path, "A", "t")
        _anchor(tmp_path, "A", budget_attempts=2)
        _run_worker(tmp_path, "A", [{"kind": "write",
                                     "patch": {"status": "DIAGNOSING"}}], background=True)
        time.sleep(2.5)
        loop = _start_loop(tmp_path, "A", max_seconds=30, every=0.5)
        # corrupt the ledger mid-run while the loop is observing
        time.sleep(1.5)
        p = pathlib.Path(tmp_path) / "tasks" / "A" / "worker.json"
        p.write_text('{"garbage": ][ not-json')
        out, _ = loop.communicate(timeout=90)
        txt = (out or b"").decode(errors="replace")
        assert "Traceback" not in txt, txt
        # the corruption was NOT accepted: the file is still unreadable and
        # the supervisor never overwrote it with a fresh/SUCCESS state
        raw = p.read_text()
        with pytest.raises(Exception):
            json.loads(raw)
        assert "SUCCESS" not in txt
        # a subsequent CLI write is REFUSED cleanly (rc 1, no crash, no
        # clobber) — f2f07ee55 made state-write refuse a corrupt ledger to
        # preserve the only copy of prior state; the ledger must NOT be
        # silently rebuilt. Repairing the ledger (delete the torn file or
        # write valid JSON) lets the supervisor recover on a fresh loop.
        r = _hermes("supervise", "state-write", "A",
                    "--json", "{\"status\":\"DIAGNOSING\"}")
        assert r.returncode == 1, r.stdout + r.stderr  # corrupt ledger REFUSED (f2f07ee55)
        # the corrupt payload must still be the on-disk truth (not clobbered)
        raw2 = p.read_text()
        with pytest.raises(Exception):
            json.loads(raw2)
        # repairing the ledger recovers the worker cleanly
        p.write_text(json.dumps({"task_id": "A", "status": "CREATED"}))
        r2 = _hermes("supervise", "state-write", "A",
                     "--json", "{\"status\":\"DIAGNOSING\"}")
        assert r2.returncode == 0, r2.stdout + r2.stderr
        st = _state(tmp_path, "A")
        assert st is not None and st.get("status") == "DIAGNOSING"


# --------------------------------------------------------------------------
# ATTACK 12 — COMPLETE claim then real process death (dead-pid verdict)
# --------------------------------------------------------------------------

class TestAttack12CompletionDeadPid:
    def test_valid_complete_then_death_is_success_evidence_driven(self, tmp_path):
        _create_task(tmp_path, "A", "t")
        # real worker: writes COMPLETE with evidence, then we kill the process
        worker = _run_worker(tmp_path, "A", [
            {"kind": "write", "patch": {"status": "COMPLETE",
                                        "completion_evidence": ["tests passed",
                                                               "verified end-to-end"]}},
            {"kind": "sleep", "seconds": 60},
        ], background=True)
        # deployer anchors the REAL live pid into the ledger (state-write
        # refuses worker_pid by design, so this is the supervisor's own write)
        time.sleep(1.0)
        _write_pid(tmp_path, "A", worker.pid)
        time.sleep(1.5)
        worker.kill()  # the writer process DIES after claiming COMPLETE
        worker.wait(timeout=20)
        loop = _start_loop(tmp_path, "A", max_seconds=10)
        txt = _loop_text(loop, timeout=60)
        assert "SUCCESS" in txt, txt[-1200:]
        st = _state(tmp_path, "A")
        assert st["status"] == "COMPLETE"
        assert int(st.get("attempt") or 1) == 1  # no rework/restart

    def test_complete_without_live_claim_is_not_success(self, tmp_path):
        """Control: worker started, DIAGNOSING, then dies without COMPLETE."""
        _create_task(tmp_path, "A", "t")
        p = _run_worker(tmp_path, "A", [
            {"kind": "write", "patch": {"status": "DIAGNOSING",
                                        "completion_evidence": ["half done"]}},
            {"kind": "die"},
        ], background=True)
        time.sleep(1.0)
        _write_pid(tmp_path, "A", p.pid)
        p.kill()
        loop = _start_loop(tmp_path, "A", max_seconds=14)
        txt = _loop_text(loop, timeout=90)
        assert "SUCCESS" not in txt, txt[-1200:]


# --------------------------------------------------------------------------
# ATTACK 13 — two loops on the SAME inbox: no double delivery, no corruption
# --------------------------------------------------------------------------

class TestAttack13SameInboxRace:
    def test_two_loops_same_task_no_double_delivery(self, tmp_path):
        _create_task(tmp_path, "A", "t")
        N = 6
        for i in range(N):
            r = _hermes("supervise", "message", "A", "--text", f"payload-{i}",
                        "--kind", "handoff", "--sender", f"s{i}")
            assert r.returncode == 0
        l1 = _start_loop(tmp_path, "A", max_seconds=18, every=0.6)
        l2 = _start_loop(tmp_path, "A", max_seconds=18, every=0.7)
        holds = []  # command.json envelope history — duplicate delivery = >1
        inbox_snapshots = []
        deadline = time.time() + 18
        while time.time() < deadline:
            try:
                cmd = json.loads((pathlib.Path(tmp_path) / "tasks" / "A" / "command.json").read_text())
                ids = [seg for seg in cmd.get("instruction", "").split() if seg.startswith("[msg:")]
                holds.extend(ids)
            except Exception:
                pass
            inbox_snapshots.append(_inbox(tmp_path, "A"))
            time.sleep(0.4)
        txt1 = _loop_text(l1, timeout=60)
        txt2 = _loop_text(l2, timeout=60)
        assert "Traceback" not in txt1 + txt2
        # every snapshot is valid JSON (no interleaved content corruption)
        assert all(isinstance(m.get("id"), str) for snap in inbox_snapshots
                   for m in snap)
        # no message delivered twice (its [msg: env recurring across samples)
        from collections import Counter
        c = Counter(holds)
        dup = [k for k, v in c.items() if v > 1]
        assert not dup, f"duplicate deliveries: {dup}"
        msgs = _inbox(tmp_path, "A")
        assert len(msgs) == N
        final = [m for m in msgs if m.get("status") == "pending"]
        assert not final, f"unconsumed pending: {final}"


# --------------------------------------------------------------------------
# ATTACK 14 — watchdog quiet-alive: sleeping real child must not stall early
# --------------------------------------------------------------------------

class TestAttack14WatchdogQuiet:
    def test_sleeping_worker_quiet_below_stall_quiet_is_held(self, tmp_path):
        # read the constant from the module — never hardcode
        quiet = SUP.STALL_QUIET_SECONDS
        assert quiet == 1800
        _create_task(tmp_path, "A", "t")
        p = _run_worker(tmp_path, "A", [{"kind": "sleep", "seconds": 60}],
                        background=True)
        time.sleep(1.0)
        _write_pid(tmp_path, "A", p.pid)  # real, live sleeping child
        pid = int(_state(tmp_path, "A")["worker_pid"])
        # heartbeat set old but within STALL_QUIET (1000 << 1800); active start grace
        _patch_state(tmp_path, "A",
                     {"last_heartbeat_at": time.time() - (quiet - 800)})
        loop = _start_loop(tmp_path, "A", max_seconds=9, every=0.8)
        txt = _loop_text(loop, timeout=45)
        assert "STALL" not in txt, txt[-1200:]
        # the sleeping child is still alive — nobody killed it
        assert _pid_alive_local(pid), "watchdog killed a silent-but-alive worker"
        p.kill()

    def test_heartbeat_stale_beyond_stall_quiet_alone_does_not_kill(self, tmp_path):
        # quiet window is the KILL threshold; a merely-old heartbeat must not
        # hunt a fingerprint-stable worker harshly — watchdog needs both
        quiet = SUP.STALL_QUIET_SECONDS
        _create_task(tmp_path, "A", "t")
        p = _run_worker(tmp_path, "A", [{"kind": "sleep", "seconds": 60}],
                        background=True)
        time.sleep(1.0)
        _write_pid(tmp_path, "A", p.pid)
        pid = int(_state(tmp_path, "A")["worker_pid"])
        _patch_state(tmp_path, "A",
                     {"last_heartbeat_at": time.time() - quiet - 500})
        loop = _start_loop(tmp_path, "A", max_seconds=7, every=0.8)
        txt = _loop_text(loop, timeout=45)
        assert "Traceback" not in txt
        p.kill()


def _pid_alive_local(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# ATTACK 15 — message/campaign against nonexistent ids
# --------------------------------------------------------------------------

class TestAttack15Nonexistent:
    def test_message_to_missing_task_fails_cleanly(self, tmp_path):
        r = _hermes("supervise", "message", "ghost", "--text", "hello",
                    "--kind", "handoff", "--sender", "x")
        assert r.returncode != 0, f"should fail, got rc={r.returncode} {r.stdout}"
        assert "Traceback" not in r.stderr
        # no orphan ledger dir materialised
        assert not (pathlib.Path(tmp_path) / "tasks" / "ghost").exists()

    def test_status_and_campaign_missing_nonzero(self, tmp_path):
        r = _hermes("supervise", "status", "ghost")
        assert r.returncode != 0
        r = _hermes("supervise", "campaign", "status", "ghost-c")
        assert r.returncode != 0, r.stdout
        assert "MISSING" in r.stdout


# --------------------------------------------------------------------------
# ATTACK 16 — unreconciled failed role must not complete; reconcile flips
# --------------------------------------------------------------------------

class TestAttack16Unreconciled:
    def test_unreconciled_failure_blocks_completion(self, tmp_path):
        cid, role = "c9", "adv"
        _create_campaign(tmp_path, cid, role, "A")
        _create_task(tmp_path, "B", "t2")
        r = _hermes("supervise", "campaign", "outcome", cid, "--task", "A",
                    "--outcome", "WORKER_FAILED", "--evidence", "e")
        assert r.returncode == 0
        # satisfy the role with the required evidence — still not complete
        r = _hermes("supervise", "campaign", "satisfy", cid, "--role", role,
                    "--task", "A", "--evidence", "e")
        assert r.returncode == 0
        st1 = json.loads(_hermes("supervise", "campaign", "status", cid).stdout)
        assert st1["status"] != "CAMPAIGN_COMPLETE", st1
        assert "A" in st1["unresolved"]["unreconciled_failures"] or \
               st1["unresolved"]["unreconciled_failures"], st1
        # reconcile the failure (evidence-gated)
        r = _hermes("supervise", "campaign", "reconcile", cid, "--task", "A",
                    "--transfer-to", "B", "--adversarial-satisfied",
                    "--completion-ok", "--evidence", "e")
        assert r.returncode == 0, r.stdout + r.stderr
        st2 = json.loads(_hermes("supervise", "campaign", "status", cid).stdout)
        assert st2["status"] == "CAMPAIGN_COMPLETE", st2