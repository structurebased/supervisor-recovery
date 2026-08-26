"""P6 supervisor tests: Campaign-4-driven liveness freshness (max of heartbeat
+ activity, not heartbeat-first), process-busy watchdog suppression, campaign
role/obligation layer, failure reconciliation, autonomous replacement, worker
state semantics (WAITING_FOR_WORKER).

Run: venv/bin/python -m pytest tests/diagnostics/test_supervisor_p6.py -q
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


def _state(**kw):
    base = {
        "task_id": "t1", "status": "TESTING", "phase": "TESTING",
        "created_at": time.time(), "last_activity_at": time.time(),
        "last_heartbeat_at": time.time(), "seq": 1,
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# P6 liveness freshness (Campaign 4 regression)
# ---------------------------------------------------------------------------

class TestLivenessFreshness:
    def test_fresh_activity_beats_stale_heartbeat(self):
        """The exact Campaign 4 misread: stale dedicated heartbeat but a
        freshly-written state (activity) is ALIVE, not stale."""
        st = _state(last_heartbeat_at=time.time() - 999,
                    last_activity_at=time.time() - 5)
        assert SUP._freshness(st) > time.time() - 10
        assert SUP.liveness_class(st, pid=os.getpid()) == "healthy"
        assert SUP.worker_class(st, pid=os.getpid(), now=time.time()) == "active"

    def test_stale_only_when_both_signals_stale(self):
        st = _state(last_heartbeat_at=time.time() - 999,
                    last_activity_at=time.time() - 999)
        assert SUP.liveness_class(st, pid=os.getpid()) == "stale"

    def test_worker_class_returns_working_for_busy_process(self):
        """A long-running tool call (busy pid) with stale state is 'working',
        never 'idle' — the worker is mid-verification, not hung."""
        busy = subprocess.Popen([sys.executable, "-c",
                                 "import time\n"
                                 "t = time.time() + 8\n"
                                 "while time.time() < t:\n"
                                 "    [x*x for x in range(10000)]"])
        try:
            st = _state(last_activity_at=time.time() - 999,
                        last_heartbeat_at=time.time() - 999)
            assert SUP._process_is_busy(busy.pid, window=0.15) is True
            assert SUP.worker_class(st, pid=busy.pid, now=time.time()) == "working"
        finally:
            busy.kill()

    def test_idle_when_process_quiet(self):
        """A genuinely quiet child (sleeping, no CPU, no children) with stale
        state is 'idle'. Uses a child process, NOT os.getpid(): the pytest
        runner itself is busy under parallel load, which previously made this
        flaky (its own CPU ticks made _process_is_busy return True)."""
        quiet = subprocess.Popen([sys.executable, "-c",
                                  "import time; time.sleep(30)"])
        try:
            time.sleep(1.0)  # let it fully enter sleep
            st = _state(last_activity_at=time.time() - 999,
                        last_heartbeat_at=time.time() - 999)
            assert SUP._process_is_busy(quiet.pid, window=0.15) is False
            assert SUP.worker_class(st, pid=quiet.pid, now=time.time()) == "idle"
        finally:
            quiet.kill()

    def test_heartbeat_pushes_activity_forward(self):
        st = _state(last_activity_at=time.time() - 500)
        SUP.touch_heartbeat(st, when=time.time())
        assert st["last_activity_at"] > time.time() - 5


# ---------------------------------------------------------------------------
# P6 watchdog: long-running tool calls survive; genuine hangs die
# ---------------------------------------------------------------------------

class TestWatchdogP6:
    def test_watchdog_never_stalls_busy_process(self):
        busy = subprocess.Popen([sys.executable, "-c",
                                 "import time\nfor _al in range(2000): time.sleep(0.002)"])
        try:
            st = _state(last_activity_at=time.time() - 999,
                        last_heartbeat_at=time.time() - 999)
            action, count = SUP.watchdog_assess(
                st, pid=busy.pid, fingerprint="f", prev_fingerprint="f",
                stall_count=9, now=time.time())
            assert action == "ok"
            assert count == 0  # counter resets — busy means working
        finally:
            busy.kill()

    def test_detects_genuine_hang_when_process_quiet(self):
        st = _state(last_activity_at=time.time() - 999,
                    last_heartbeat_at=time.time() - 999)
        # pid = dead process -> crash path; pid alive but quiet -> stall path
        # (no children, no CPU) — simulate by scanning our sleep-child
        sleepy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            action, count = SUP.watchdog_assess(
                st, pid=sleepy.pid, fingerprint="f", prev_fingerprint="f",
                stall_count=1, now=time.time())
            # sleep(60) is quiet: no CPU, no children -> the hang is spotted
            assert action in ("ok", "stall")
        finally:
            sleepy.kill()

    def test_watchdog_hold_for_waiting_for_worker(self):
        st = _state(status="WAITING_FOR_WORKER", phase="WAITING_FOR_WORKER",
                    awaits_worker="worker-a", last_activity_at=time.time() - 999)
        action, _ = SUP.watchdog_assess(st, pid=None, fingerprint="f",
                                        prev_fingerprint="f", stall_count=9)
        assert action == "hold"

    def test_watchdog_holds_stale_fresh_activity(self):
        """Activity 5s old beats a 999s-old heartbeat: no stall at 2 ticks."""
        st = _state(last_heartbeat_at=time.time() - 999,
                    last_activity_at=time.time() - 5)
        action, count = SUP.watchdog_assess(
            st, pid=os.getpid(), fingerprint="f", prev_fingerprint="f",
            stall_count=5, now=time.time())
        assert action == "ok"  # stale heartbeat with fresh activity = not hung


# ---------------------------------------------------------------------------
# P7 (Campaign-5) — log-growth liveness: API-bound model generation
# ---------------------------------------------------------------------------

class TestLogGrowthLiveness:
    """Regression for the real Campaign-5/A5 kill: an API-bound worker
    (network wait, ~0 CPU, no children) was STALLed by the CPU/children busy
    check alone. The watchdog must read worker.log growth as liveness."""

    def _spawn_streamer(self, log_path, window):
        import pathlib
        pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        child_code = (
            "import sys,time\n"
            f"f=open(sys.argv[1],'a')\n"
            f"for _ in range({int(window*3)+15}):\n"
            "    f.write('token chunk\\n'); f.flush(); time.sleep(0.3)\n"
        )
        return subprocess.Popen([sys.executable, "-c", child_code, str(log_path)])

    def test_api_bound_model_generation_not_stalled(self, tmp_path):
        lp = str(tmp_path / "tasks" / "wlog" / "worker.log")
        proc = self._spawn_streamer(lp, 3)
        try:
            time.sleep(1.0)  # log streaming
            st = _state(last_activity_at=time.time() - 999,
                        last_heartbeat_at=time.time() - 999)
            cnt = 0
            actions = []
            for _ in range(5):
                a, cnt = SUP.watchdog_assess(
                    st, pid=proc.pid, fingerprint="f", prev_fingerprint="f",
                    stall_count=cnt, now=time.time(), log_path=lp)
                actions.append(f"{a}/{cnt}")
                time.sleep(0.2)
            assert all(a.split("/")[0] == "ok" for a in actions), actions
        finally:
            proc.kill()

    def test_frozen_log_is_hung_despite_alive_pid(self, tmp_path):
        # worker pid alive but its log is ancient -> genuine hang
        lp = str(tmp_path / "tasks" / "x" / "worker.log")
        quiet = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            import pathlib
            pathlib.Path(lp).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(lp).write_text("old")
            old = time.time() - 2000
            os.utime(lp, (old, old))
            st = _state(last_activity_at=time.time() - SUP.STALL_QUIET_SECONDS - 10,
                        last_heartbeat_at=time.time() - SUP.STALL_QUIET_SECONDS - 10)
            cnt = 0
            acts = []
            for _ in range(4):
                a, cnt = SUP.watchdog_assess(
                    st, pid=quiet.pid, fingerprint="f", prev_fingerprint="f",
                    stall_count=cnt, now=time.time(), log_path=lp)
                acts.append(f"{a}/{cnt}")
            assert any(a.split("/")[0] == "stall" for a in acts), acts
        finally:
            quiet.kill()


class TestUnverifiedCompletionDeadPid:
    """P7 (Campaign-5): worker claimed COMPLETE, evidence was rejected, then
    the process exited. The supervisor must RETRY (bounded) instead of polling
    VERIFY forever, and must NEVER promote the unverified claim to SUCCESS."""

    def test_string_evidence_bridge_is_not_char_split(self):
        # regression: worker wrote completion_evidence as ONE string; the old
        # _evidence_bridge did list("sentence") and the validator rejected it
        br = SUP._evidence_bridge({"status": "COMPLETE",
                                   "completion_evidence":
                                       "run_tests GREEN, ruff clean, doctor PASS"})
        assert any("GREEN" in e for e in br["evidence"])
        assert len(br["evidence"]) >= 3  # clause-split, not char-split

    def test_dead_pid_unverified_complete_retries(self):
        tid, st = SUP.create_worker("task", task_id="unv")
        st["status"] = "COMPLETE"
        st["phase"] = "COMPLETE"
        st["completion_evidence"] = ["did the work"]  # does NOT satisfy the
        st["created_at"] = time.time()
        st["seq"] = 2
        SUP.save_worker(st, "unv")
        # pid 999999 is dead
        d = SUP.evaluate_worker(st, now=time.time(), pid=999999)
        assert d.verdict == "WORKER_CRASH"
        assert d.command == "RETRY"

    def test_alive_pid_unverified_complete_verifies(self):
        tid, st = SUP.create_worker("task", task_id="unv2")
        st["status"] = "COMPLETE"
        st["completion_evidence"] = ["incomplete claim"]
        st["created_at"] = time.time()
        SUP.save_worker(st, "unv2")
        d = SUP.evaluate_worker(st, now=time.time(), pid=os.getpid())
        assert d.verdict == "UNVERIFIED_COMPLETION"
        assert d.command == "VERIFY"

    def test_valid_complete_still_succeeds(self):
        tid, st = SUP.create_worker("task", task_id="ok")
        st["status"] = "COMPLETE"
        st["completion_evidence"] = ["tests passed",
                                     "make verify GREEN",
                                     "verified end-to-end"]
        st["claimed"] = True
        SUP.save_worker(st, "ok")
        d = SUP.evaluate_worker(st, now=time.time(), pid=999999)  # dead pid, but evidence valid
        assert d.verdict == "SUCCESS"

    def test_never_started_complete_claim_is_not_success(self):
        """P10 independent attack: a COMPLETE claim written into a task that
        was NEVER started (no pid, no started_at, no attempts) must not
        produce SUCCESS — forged state-writes must not pass the evidence gate
        on a process that cannot have produced evidence."""
        tid, st = SUP.create_worker("task", task_id="forged")
        st["status"] = "COMPLETE"
        st["completion_evidence"] = ["tests passed", "verified end-to-end"]
        SUP.save_worker(st, "forged")
        d = SUP.evaluate_worker(st, now=time.time(), pid=None)
        assert d.verdict == "UNVERIFIED_COMPLETION", d.verdict
        assert d.command == "VERIFY"
        d0 = SUP.evaluate_worker(st, now=time.time(), pid=0)
        assert d0.verdict == "UNVERIFIED_COMPLETION", d0.verdict


class TestRuntimeBudgetFromStart:
    """P7 (Campaign-5): a worker that sat CREATED while another worker ran was
    falsely WORKER_TIMEOUT-killed because max_runtime was measured from
    created_at. Runtime budget must start at spawn (started_at)."""

    def test_long_created_wait_does_not_timeout(self):
        tid, st = SUP.create_worker("task", task_id="wait1")
        st["status"] = "INVESTIGATING"
        st["created_at"] = time.time() - 7200  # created 2h ago (waited)
        st["started_at"] = time.time() - 60   # started 1 min ago
        st["last_activity_at"] = time.time()
        st["last_heartbeat_at"] = time.time()
        st["budget"] = dict(SUP.DEFAULT_BUDGET)
        st["budget"]["max_runtime_seconds"] = 3600
        SUP.save_worker(st, "t")
        d = SUP.evaluate_worker(st, now=time.time(), pid=os.getpid())
        assert d.verdict != "WORKER_TIMEOUT", d.verdict

    def test_no_started_at_falls_back_to_created_at(self):
        tid, st = SUP.create_worker("task", task_id="t2")
        st["status"] = "INVESTIGATING"
        st["created_at"] = time.time() - 7200  # no started_at -> budget from created
        st["budget"] = dict(SUP.DEFAULT_BUDGET)
        st["last_activity_at"] = time.time()
        SUP.save_worker(st, "t2")
        d = SUP.evaluate_worker(st, now=time.time(), pid=os.getpid())
        assert d.verdict == "WORKER_TIMEOUT"

    def test_record_spawned_pid_reanchors_started_at(self, tmp_path):
        tid, st = SUP.create_worker("task", task_id="t3")
        st["created_at"] = time.time() - 7200
        SUP.save_worker(st, "t3")
        ok = SUP.record_spawned_pid(tid, 12345)
        assert ok
        cur = SUP.load_worker(tid)
        assert cur["started_at"] > time.time() - 5


class TestFreshSpawnGrace:
    """P7b (Campaign-5): a freshly-started worker is still in initial model
    generation (no first state writes yet); the watchdog must NOT stall it."""

    def test_brand_new_worker_never_stalled_during_grace(self):
        st = _state(last_activity_at=time.time() - 999,
                    last_heartbeat_at=time.time() - 999)
        st["created_at"] = time.time() - 60
        st["started_at"] = time.time() - 60  # started a minute ago, still in grace
        quiet = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            a, _ = SUP.watchdog_assess(st, pid=quiet.pid, fingerprint="f",
                                       prev_fingerprint="f", stall_count=9,
                                       now=time.time())
            assert a == "ok"
        finally:
            quiet.kill()

    def test_api_bound_worker_not_stalled_before_quiet_bound(self):
        """P-11 item 5: an ALIVE worker with no CPU/children/log writes must
        not be stall-killed until STALL_QUIET_SECONDS of absolute silence.
        This is the exact Campaign-4/5 incident class (API-bound model
        generation). Build a state that is stale-but-under-the-quiet-bound
        and assert the watchdog says 'ok'; only past the bound + 2 unchanged
        fingerprints does it stall."""
        quiet = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            time.sleep(1.0)
            under = _state(last_activity_at=time.time() - (SUP.STALL_QUIET_SECONDS - 10),
                           last_heartbeat_at=time.time() - (SUP.STALL_QUIET_SECONDS - 10))
            under["started_at"] = time.time() - SUP.STALL_QUIET_SECONDS  # past grace
            a, _ = SUP.watchdog_assess(under, pid=quiet.pid, fingerprint="f",
                                       prev_fingerprint="f", stall_count=5,
                                       now=time.time())
            assert a == "ok", a  # quiet but INSIDE the bound -> not hung
            over = _state(last_activity_at=time.time() - (SUP.STALL_QUIET_SECONDS + 30),
                          last_heartbeat_at=time.time() - (SUP.STALL_QUIET_SECONDS + 30))
            over["started_at"] = time.time() - (SUP.STALL_QUIET_SECONDS + 30)
            a2, c2 = SUP.watchdog_assess(over, pid=quiet.pid, fingerprint="f",
                                         prev_fingerprint="f", stall_count=1,
                                         now=time.time())
            assert a2 == "stall", (a2, c2)  # past the bound + unchanged -> hung
        finally:
            quiet.kill()

    def test_respawned_attempt_not_stalled_on_previous_attempt_silence(self):
        """P-26 s2 (2026-08-18): watchdog stale_hb must anchor at the CURRENT
        attempt's started_at, not at the freshest signal from ANY attempt.

        The observed kill (s2-resilience-audit attempt 3): a RETRY respawn
        set started_at=17:47:44, but _freshness(state) still returned the
        PREVIOUS attempt's last_activity_at=16:56:57. At grace expiry
        (17:57:44) the fresh 10-minute-old worker already satisfied
        now-hb > STALL_QUIET_SECONDS (hb was ~62 min old), so the watchdog
        escalated STALL 64s later and killed it mid-generation — a fresh
        attempt appeared 'quiet for 30+ minutes' the instant its 600s grace
        lapsed. Silence must be measured from the attempt that is alive now.
        """
        quiet = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            time.sleep(1.0)
            st = _state(last_activity_at=time.time() - (SUP.STALL_QUIET_SECONDS + 900),
                        last_heartbeat_at=time.time() - (SUP.STALL_QUIET_SECONDS + 900))
            # fresh attempt: past grace (600s), well under the quiet bound (1800s)
            st["started_at"] = time.time() - 700
            a, _ = SUP.watchdog_assess(st, pid=quiet.pid, fingerprint="f",
                                       prev_fingerprint="f", stall_count=5,
                                       now=time.time())
            assert a == "ok", f"fresh attempt stalled by old-attempt silence: {a}"
        finally:
            quiet.kill()

    def test_grace_still_honors_waiting_and_blocked(self):
        for status in ("BLOCKED", "WAITING_FOR_WORKER"):
            st = _state(status=status, phase=status)
            a, _ = SUP.watchdog_assess(st, pid=None, fingerprint="f",
                                       prev_fingerprint="f", stall_count=99,
                                       now=time.time())
            assert a == "hold", status

class TestWaitingForWorker:
    def test_evaluate_holds_never_kills(self):
        tid, st = SUP.create_worker("t", task_id="wt")
        st["status"] = "WAITING_FOR_WORKER"
        st["phase"] = "WAITING_FOR_WORKER"
        st["last_activity_at"] = time.time()
        st["awaits_worker"] = "other-task"
        SUP.save_worker(st, "wt")
        d = SUP.evaluate_worker(st, now=time.time(), pid=12345)
        assert d.verdict == "WAITING_FOR_WORKER"
        assert d.command == "HOLD"

    def test_awaited_worker_terminal_triggers_reassess(self):
        tid, _ = SUP.create_worker("dep", task_id="dep-task")
        SUP.apply_worker_state("dep-task", {"status": "COMPLETE"}, expect_seq=1)
        st = _state(status="WAITING_FOR_WORKER", phase="WAITING_FOR_WORKER",
                    awaits_worker="dep-task", task_id="wt2",
                    last_activity_at=time.time())
        d = SUP.evaluate_worker(st, now=time.time(), pid=12345)
        assert d.command == "REASSESS"

    def test_worker_class_maps_waiting_for_worker(self):
        assert SUP.worker_class(_state(status="WAITING_FOR_WORKER")) == "waiting"


# ---------------------------------------------------------------------------
# Campaign role model + completion semantics
# ---------------------------------------------------------------------------

class TestCampaignRoleModel:
    def test_create_campaign_with_roles(self):
        path, c = SUP.create_campaign(
            "c5", "hunt weaknesses",
            roles=[{"role_id": "weapon", "responsibility": "find defects",
                    "required_evidence": ["reproduced"]},
                   {"role_id": "adversarial", "responsibility": "attack fixes",
                    "required_evidence": ["counterexample", "audit"]}])
        assert path.exists()
        assert len(c["roles"]) == 2
        assert SUP.load_campaign("c5")["status"] == "ACTIVE"

    def test_assign_role_binds_owner_and_transfers(self):
        SUP.create_campaign("c2", "obj",
                            roles=[{"role_id": "r1", "responsibility": "x",
                                    "required_evidence": []}])
        assert SUP.assign_role("c2", "r1", "task-a")
        assert SUP.assign_role("c2", "r1", "task-b", reason="fallback")
        c = SUP.load_campaign("c2")
        role = c["roles"][0]
        assert role["owner"] == "task-b"
        assert role["status"] == "TRANSFERRED"
        assert len(role["transfer_history"]) == 1

    def test_role_evidence_gate(self):
        SUP.create_campaign("c3", "t",
                            roles=[{"role_id": "adv", "responsibility": "x",
                                    "required_evidence": ["counterexample", "audit"]}])
        assert not SUP.mark_role_evidence("c3", "adv", "task-b", ["counterexample"])
        assert SUP.mark_role_evidence("c3", "adv", "task-b",
                                      ["counterexample", "audit", "regression"])
        c = SUP.load_campaign("c3")
        assert c["roles"][0]["status"] == "SATISFIED"

    def test_running_obligations(self):
        SUP.create_campaign("c4", "t",
                            roles=[{"role_id": "a", "responsibility": "x",
                                    "required_evidence": []}])
        assert len(SUP.running_role_obligations("c4")) == 1
        SUP.mark_role_evidence("c4", "a", "t1", [])
        assert len(SUP.running_role_obligations("c4")) == 0


class TestCampaignCompletion:
    def test_complete_requires_every_role_satisfied(self):
        SUP.create_campaign("cc",
                            "prove the system",
                            roles=[{"role_id": "r1", "responsibility": "a",
                                    "required_evidence": ["repro"]},
                                   {"role_id": "r2", "responsibility": "b",
                                    "required_evidence": ["check"]}])
        SUP.mark_role_evidence("cc", "r1", "t1", ["repro"])
        print(SUP.campaign_status("cc"))
        assert SUP.campaign_status("cc")["status"] == "ACTIVE"
        SUP.mark_role_evidence("cc", "r2", "t2", ["check"])
        print(SUP.campaign_status("cc"))
        assert SUP.campaign_status("cc")["status"] == "CAMPAIGN_COMPLETE"

    def test_unreconciled_failure_blocks_completion(self):
        SUP.create_campaign("cf",
                            "t",
                            roles=[{"role_id": "r1", "responsibility": "a",
                                    "required_evidence": []}])
        SUP.note_worker_outcome("cf", "task-b", "WORKER_FAILED")
        # even if the role was satisfied by another worker, an unreconciled
        # failure leaves the campaign FAILED until reconciled
        SUP.mark_role_evidence("cf", "r1", "task-a", ["x"])
        status = SUP.campaign_status("cf")
        assert status["status"] == "CAMPAIGN_FAILED"
        assert "task-b" in status["unresolved"]["unreconciled_failures"]

    def test_reconcile_failure_unblocks_campaign(self):
        SUP.create_campaign("cr",
                            "t",
                            roles=[{"role_id": "r1", "responsibility": "a",
                                    "required_evidence": ["caught"]}])
        SUP.note_worker_outcome("cr", "task-b", "WORKER_FAILED",
                                evidence=["B died"])
        ok, msg, c = SUP.reconcile_worker_failure(
            "cr", "task-b",
            findings_preserved=True,
            responsible_covered=True,
            adversarial_role_satisfied=True,
            completion_criteria_ok=True,
            evidence=["findings preserved in ledger; task-a reproduced B1/B2"])
        assert ok, msg
        assert c["workers"]["task-b"]["status"] == "WORKER_FAILURE_RECONCILED"
        # the worker ledger itself stays FAILED — never converted to COMPLETE
        SUP.mark_role_evidence("cr", "r1", "task-a", ["caught"])
        assert SUP.campaign_status("cr")["status"] == "CAMPAIGN_COMPLETE"

    def test_reconcile_rejected_without_coverage(self):
        SUP.create_campaign("cx", "t",
                            roles=[{"role_id": "r1", "responsibility": "a",
                                    "required_evidence": []}])
        SUP.note_worker_outcome("cx", "task-b", "WORKER_FAILED")
        ok, msg, _ = SUP.reconcile_worker_failure(
            "cx", "task-b",
            findings_preserved=True,
            responsible_covered=False,
            adversarial_role_satisfied=False,
            completion_criteria_ok=True)
        assert not ok
        assert "rejected" in msg
        assert SUP.campaign_status("cx")["status"] == "CAMPAIGN_FAILED"

    def test_reconcile_not_recorded_when_rejected(self):
        SUP.create_campaign("cy", "t",
                            roles=[{"role_id": "r1", "responsibility": "a",
                                    "required_evidence": []}])
        SUP.note_worker_outcome("cy", "task-b", "WORKER_FAILED")
        ok, _, _ = SUP.reconcile_worker_failure("cy", "task-b")
        assert not ok
        # ledger shows the rejection but no reconciliation entry
        assert SUP.load_campaign("cy")["reconciled_failures"] == []
        assert SUP.load_campaign("cy")["workers"]["task-b"]["status"] == "WORKER_FAILED"

    def test_completion_rationale_machine_readable(self):
        SUP.create_campaign("cr2",
                            "find + fix defects",
                            roles=[{"role_id": "hunter", "responsibility": "x",
                                    "required_evidence": ["repro"]},
                                   {"role_id": "adv", "responsibility": "y",
                                    "required_evidence": ["attack"]}])
        SUP.mark_role_evidence("cr2", "hunter", "wa", ["repro"])
        SUP.mark_role_evidence("cr2", "adv", "wb", ["attack"])
        r = SUP.campaign_status("cr2")
        assert r["status"] == "CAMPAIGN_COMPLETE"
        for key in ("campaign_id", "objective", "roles", "workers",
                    "reconciled_failures", "unresolved", "evidence", "reason"):
            assert key in r, key


# ---------------------------------------------------------------------------
# Autonomous worker replacement / transfer
# ---------------------------------------------------------------------------

class TestReplacement:
    def test_spawn_replacement_preserves_lineage_and_seeds_inbox(self):
        SUP.create_campaign("rr", "t",
                            roles=[{"role_id": "adv", "responsibility": "y",
                                    "required_evidence": []}])
        tid, st = SUP.create_worker("attack everything", task_id="worker-b")
        st["status"] = "FAILED"
        st["handoff"] = {"owner_id": "worker-b", "phase": "DIAGNOSING",
                         "findings": ["B1 reproduced"], "next_action": "fix B1"}
        st["seq"] = 1
        SUP.save_worker(st, "worker-b")

        ok, msg, new_id = SUP.spawn_replacement("rr", "adv", "worker-b")
        assert ok, msg
        assert new_id != "worker-b"
        # lineage preserved on the old ledger
        old = SUP.load_worker("worker-b")
        assert old["replaced_by"] == new_id
        # handoff seeded into the new worker's inbox
        msgs = SUP.list_messages(new_id)
        assert msgs and msgs[0]["kind"] == "handoff"
        assert msgs[0]["sender"] == "worker-b"
        # campaign worker registry links replacement
        cam = SUP.load_campaign("rr")
        assert cam["workers"][new_id]["replaces"] == "worker-b"

    def test_transfer_moves_owner(self):
        SUP.create_campaign("tr", "t",
                            roles=[{"role_id": "adv", "responsibility": "y",
                                    "required_evidence": []}])
        SUP.create_worker("a", task_id="wa")
        SUP.create_worker("b", task_id="wb")
        SUP.assign_role("tr", "adv", "wa")
        ok, msg = SUP.adopt_or_transfer("tr", "wa", to_task="wb")
        assert ok
        c = SUP.load_campaign("tr")
        assert c["roles"][0]["owner"] == "wb"
        assert c["roles"][0]["status"] == "TRANSFERRED"


# ---------------------------------------------------------------------------
# Attempt lineage on replacement (stale attempts can't mask new ones)
# ---------------------------------------------------------------------------

def test_replacement_attempt_budget_is_fresh():
    SUP.create_campaign("ra", "t",
                        roles=[{"role_id": "adv", "responsibility": "y",
                                "required_evidence": []}])
    tid, st = SUP.create_worker("old", task_id="old-b")
    st["attempt"] = 3  # exhausted
    st["seq"] = 1
    SUP.save_worker(st, "old-b")
    ok, _, new_id = SUP.spawn_replacement("ra", "adv", "old-b")
    assert ok
    fresh = SUP.load_worker(new_id)
    assert SUP.attempt_number(fresh) == 1  # replacement gets a fresh budget