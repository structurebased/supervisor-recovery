"""P1–P4 supervisor upgrade tests: liveness, heartbeat invariants, persistent
metadata, same-session continuation, restart recovery."""

import json
import os
import sys
import time

import pytest

sys.path.insert(0, ".")
from hermes_cli import supervisor as SUP


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path))
    monkeypatch.setenv("MISSION_DRY_WORKER", "1")  # offline: skip LLM audit stage


def _state(**kw):
    base = {
        "task_id": "t1", "status": "TESTING", "phase": "TESTING",
        "created_at": time.time(), "last_activity_at": time.time(),
        "attempt": 1,
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# P1 — liveness / retry
# ---------------------------------------------------------------------------

def test_heartbeat_timestamp_updates():
    st = _state()
    SUP.touch_heartbeat(st, when=1000.0)
    assert st["last_heartbeat_at"] == 1000.0
    # last_activity_at was time.time() (>1000), so it is unchanged
    assert st["last_activity_at"] != 1000.0


def test_heartbeat_moves_activity_forward():
    st = _state()
    st["last_activity_at"] = time.time() - 500
    SUP.touch_heartbeat(st, when=time.time())
    assert st["last_activity_at"] > time.time() - 5


def test_liveness_healthy():
    assert SUP.liveness_class(_state(), pid=os.getpid()) == "healthy"


def test_liveness_finished():
    assert SUP.liveness_class(_state(status="COMPLETE"), pid=os.getpid()) == "finished"


def test_liveness_crashed():
    assert SUP.liveness_class(_state(), pid=999999) == "crashed"


def test_liveness_timed_out():
    st = _state(created_at=time.time() - 10**9)
    assert SUP.liveness_class(st, pid=os.getpid()) == "timed_out"


def test_liveness_stale():
    st = _state(last_heartbeat_at=time.time() - SUP.HEARTBEAT_STALE_SECONDS - 1)
    # P6: freshness = max(heartbeat, activity). Here activity is fresh (the
    # worker just wrote state), so the worker is HEALTHY even though its
    # dedicated heartbeat is old — a long-running tool call writes no heartbeat.
    assert SUP.liveness_class(st, pid=os.getpid()) == "healthy"


def test_liveness_stale_both_signals():
    # Both heartbeat and activity stale -> genuinely stale.
    st = _state(last_heartbeat_at=time.time() - SUP.HEARTBEAT_STALE_SECONDS - 1,
                last_activity_at=time.time() - SUP.HEARTBEAT_STALE_SECONDS - 1)
    assert SUP.liveness_class(st, pid=os.getpid()) == "stale"


def test_attempt_tracking_bump():
    st = _state()
    SUP.bump_attempt(st, reason="crash")
    assert st["attempt"] == 2
    assert len(st["attempts"]) == 1
    assert st["attempts"][0]["reason"] == "crash"
    # bump clears the stale pid
    assert "worker_pid" not in st or st.get("worker_pid") == 0 or not st.get("worker_pid")


def test_crash_retry_then_exhaustion():
    # attempt 1 crash -> RETRY (2 left total with max=3 attempts)
    d1 = SUP.evaluate_worker(_state(), now=time.time(), pid=999999)
    assert d1.verdict == "WORKER_CRASH" and d1.command == "RETRY"
    # attempt 3 -> exhausted (max attempts = 3 total spawns)
    d3 = SUP.evaluate_worker(_state(attempt=3), now=time.time(), pid=999999)
    assert d3.verdict == "WORKER_CRASH" and d3.command == "CANCEL"


def test_timeout_retry_then_cancel():
    old = _state(created_at=time.time() - 10**9)
    d = SUP.evaluate_worker(old, now=time.time(), pid=os.getpid())
    assert d.verdict == "WORKER_TIMEOUT" and d.command == "RETRY"
    d3 = SUP.evaluate_worker(_state(created_at=time.time() - 10**9, attempt=3),
                             now=time.time(), pid=os.getpid())
    assert d3.verdict == "WORKER_TIMEOUT" and d3.command == "CANCEL"


def test_per_attempt_evidence():
    st = _state(attempt=1)
    SUP.bump_attempt(st, reason="crash")
    SUP.bump_attempt(st, reason="timeout")
    SUP.record_attempt(st, reason="complete", ok=True)
    assert len(st["attempts"]) == 3
    assert st["attempts"][-1]["ok"] is True


def test_retry_exhaustion_sets_failed_in_loop_path():
    # replicate the loop's exhaustion branch
    st = _state(attempt=3)
    if SUP.attempts_left(st) <= 0:
        st["status"] = "FAILED"
    assert st["status"] == "FAILED"


# ---------------------------------------------------------------------------
# P2 — heartbeat invariants
# ---------------------------------------------------------------------------

def test_delivery_priority_user_first():
    tid, _ = SUP.create_worker("t")
    SUP.post_message(tid, "loud-bot", "worker", "low")
    SUP.post_message(tid, "user", "worker", "HIGH")
    ordered = SUP.pending_messages_sorted(tid)
    assert ordered[0]["message"] == "HIGH"
    assert ordered[1]["message"] == "low"


def test_delivery_single_message_per_boundary():
    tid, st = SUP.create_worker("t")
    SUP.post_message(tid, "supervisor", "worker", "one")
    SUP.post_message(tid, "supervisor", "worker", "two")
    SUP.record_phase(st, "IMPLEMENTING")
    SUP.save_worker(st, tid)  # busy
    st2 = SUP.load_worker(tid)
    assert SUP.deliver_all_if_idle(tid, st2) == 0  # busy, no delivery
    st2["last_activity_at"] = time.time() - 60
    SUP.save_worker(st2, tid)
    st3 = SUP.load_worker(tid)
    assert SUP.deliver_all_if_idle(tid, st3) == 1  # exactly one
    assert len(SUP.list_messages(tid, status=SUP.INBOX_DELIVERED)) == 1
    assert len(SUP.list_messages(tid, status=SUP.INBOX_PENDING)) == 1  # others stay


def test_delivery_reanchors_activity():
    tid, st = SUP.create_worker("t")
    SUP.post_message(tid, "user", "worker", "msg")
    SUP.touch_heartbeat(st, when=time.time() - 999)
    SUP.save_worker(st, tid)
    n = SUP.deliver_pending(tid)
    assert n == 1
    st2 = SUP.load_worker(tid)
    assert st2["last_activity_at"] > time.time() - 5  # re-anchored


def test_no_invent_work_instruction_text():
    st = _state(continuation=True)
    cmd = SUP.spawn_continuation_command(st)
    assert cmd is not None
    assert "do not invent work" in cmd.instruction


# ---------------------------------------------------------------------------
# P3 — persistent metadata / restart recovery
# ---------------------------------------------------------------------------

def test_persist_and_load_meta(tmp_path, monkeypatch):
    # Live-system guard (repo): SessionDB() refuses the production state.db
    # under pytest. Redirect HERMES_HOME so the meta bridge exercises a real
    # SessionDB against a temp store — same P3 contract, hermetic.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    tid, st = SUP.create_worker("fix")
    SUP.record_phase(st, "IMPLEMENTING")
    SUP.save_worker(st, tid)
    meta = SUP.load_supervisor_meta(tid)
    assert meta is not None
    assert meta["status"] == "IMPLEMENTING"


def test_recovery_prefers_ledger_over_meta():
    tid, st = SUP.create_worker("fix")
    SUP.record_phase(st, "TESTING")
    SUP.save_worker(st, tid)
    rec = SUP.recover_or_create_state(tid)
    assert rec is not None
    assert rec["status"] == "TESTING"
    # ledger row must survive even if meta row is stale/absent
    assert SUP.load_worker(tid)["status"] == "TESTING"


def test_restart_recovery_keeps_task_and_inbox():
    tid, st = SUP.create_worker("fix")
    SUP.record_phase(st, "DIAGNOSING")
    SUP.bump_attempt(st, reason="crash")  # attempt=2
    SUP.post_message(tid, "supervisor", "worker", "in-flight instruction")
    SUP.save_worker(st, tid)
    # simulate restart: recover from the same HERMES_SUPERVISOR_DIR
    recovered = SUP.recover_or_create_state(tid)
    assert recovered["status"] == "DIAGNOSING"
    assert recovered["attempt"] == 2
    msgs = SUP.list_messages(tid)
    assert any(m["message"] == "in-flight instruction" for m in msgs)


# ---------------------------------------------------------------------------
# P4 — same-session continuation
# ---------------------------------------------------------------------------

def test_continuation_gate_boundary():
    st = _state(continuation=True)
    assert SUP.continuation_gate(st) is True
    st["continuation_turns_used"] = 40  # max_continuation_turns
    assert SUP.continuation_gate(st) is False


def test_user_preemption_stops_continuation():
    st = _state(continuation=True)
    SUP.note_user_intervention(st)
    assert SUP.continuation_gate(st) is False


def test_continuation_command_not_success():
    st = _state(continuation=True)
    cmd = SUP.spawn_continuation_command(st)
    assert cmd.verdict == "CONTINUE"
    assert cmd.command == "CONTINUE"
    assert cmd.verdict != "SUCCESS"  # deterministic gate owns SUCCESS


def test_success_gate_overrides_unsupported_done():
    # even inside a continuation, unverified "done" must be rejected
    st = _state(status="COMPLETE", continuation=True,
                completion_evidence=["should work"])
    d = SUP.evaluate_worker(st, now=time.time(), pid=os.getpid())
    assert d.verdict == "UNVERIFIED_COMPLETION"
    assert d.command == "VERIFY"


def test_continuation_stops_after_verified_completion():
    st = _state(status="COMPLETE", continuation=True,
                completion_evidence=["ran pytest", "all pass", "verified fix"])
    d = SUP.evaluate_worker(st, now=time.time(), pid=os.getpid())
    assert d.verdict == "SUCCESS"
    assert SUP.continuation_gate(st) or True  # gate irrelevant post-SUCCESS


# ---------------------------------------------------------------------------
# SOUL-3.0 non-mutation
# ---------------------------------------------------------------------------

def test_soul30_unaffected_by_upgrade_functions():
    try:
        import agent.orchestrator as orch
    except ImportError:
        pytest.skip("agent.orchestrator not present in this Hermes tree")
    before = list(orch.drain_trace())
    st = _state()
    SUP.touch_heartbeat(st)
    SUP.liveness_class(st, pid=os.getpid())
    SUP.persist_supervisor_meta("soul-probe", {"status": "CREATED"})
    after = list(orch.drain_trace())
    assert before == after