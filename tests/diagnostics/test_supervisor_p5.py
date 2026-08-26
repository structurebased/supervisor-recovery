"""P5 supervisor tests: handoff contract, seq/CAS stale-write protection,
explicit waiting states, watchdog stall/recovery, inbox threads/dedup/ack/stale.

Run: venv/bin/python -m pytest tests/diagnostics/test_supervisor_p5.py -q
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


# ---------------------------------------------------------------------------
# seq versioning + stale-write protection
# ---------------------------------------------------------------------------

def test_state_seq_advances_on_apply():
    tid, st = SUP.create_worker("t")
    ok, msg, st2 = SUP.apply_worker_state(tid, {"status": "INVESTIGATING",
                                                 "phase": "INVESTIGATING",
                                                 "findings": ["baseline"]},
                                           expect_seq=1)
    assert ok, msg
    assert st2["seq"] == 2
    disk = SUP.load_worker(tid)
    assert disk["seq"] == 2
    assert disk["status"] == "INVESTIGATING"


def test_state_cas_rejects_stale_expect_seq():
    tid, _ = SUP.create_worker("t")
    ok, msg, _ = SUP.apply_worker_state(tid, {"status": "TESTING"}, expect_seq=1)
    assert ok
    # A stale writer that still believes seq==1 is rejected
    ok2, msg2, _ = SUP.apply_worker_state(tid, {"status": "DIAGNOSING"}, expect_seq=1)
    assert not ok2
    assert "stale" in msg2
    assert SUP.load_worker(tid)["status"] == "TESTING"  # newer state intact


def test_stale_worker_cannot_overwrite_newer_state():
    tid, _ = SUP.create_worker("t")
    # worker W1 writes seq -> 2
    SUP.apply_worker_state(tid, {"status": "TESTING", "progress": "W1"}, expect_seq=1)
    # worker W2 (newer attempt) writes seq -> 3
    SUP.apply_worker_state(tid, {"status": "VERIFYING", "progress": "W2"}, expect_seq=2)
    stale = SUP.load_worker(tid)
    # W1 comes back and tries to write over the current state
    ok, msg, _ = SUP.apply_worker_state(tid, {"progress": "W1-old"}, expect_seq=2)
    assert not ok
    assert "stale" in msg
    cur = SUP.load_worker(tid)
    assert cur["seq"] == 3
    assert cur["progress"] == "W2"
    # an audit trail exists
    audit = (SUP._tasks_dir() / tid / "audit.jsonl").read_text()
    assert "stale_write" in audit


def test_save_worker_cas_successor_rule():
    tid, _ = SUP.create_worker("t")
    SUP.apply_worker_state(tid, {"status": "TESTING"}, expect_seq=1)  # seq 2
    # a bogus writer that jumps seq (3) when disk is 2 is rejected
    bogus = {"task_id": tid, "status": "PLANNING", "seq": 5}
    ok, msg = SUP.save_worker_cas(tid, bogus)
    assert not ok
    assert "stale write rejected" in msg
    # exact successor accepted
    winner = {"task_id": tid, "status": "PLANNING", "seq": 3}
    ok, msg = SUP.save_worker_cas(tid, winner)
    assert ok
    assert SUP.load_worker(tid)["seq"] == 3


# ---------------------------------------------------------------------------
# handoff block
# ---------------------------------------------------------------------------

def test_build_handoff_fields():
    tid, st = SUP.create_worker("task X")
    st.update({
        "worker_identity": "wa",
        "phase": "IMPLEMENTING",
        "status": "IMPLEMENTING",
        "findings": ["weak spot A"],
        "files_changed": ["a.py"],
        "tests_executed": 4, "tests_passed": 3,
        "blockers": [], "next_action": "wire the fix",
        "completion_evidence": ["test passes"],
        "seq": 1,
    })
    h = SUP.build_handoff(st)
    for key in ("owner_id", "phase", "objective", "status", "findings",
                "files_changed", "tests", "blockers", "next_action",
                "evidence", "seq", "updated_at"):
        assert key in h, key
    assert h["owner_id"] == "wa"
    assert h["objective"] == "task X"
    assert h["tests"] == {"executed": 4, "passed": 3, "failed": 1}
    assert h["seq"] == 1


def test_update_handoff_versions_history():
    tid, st = SUP.create_worker("t")
    st["seq"] = 1
    SUP.update_handoff(st)
    st["seq"] = 2
    SUP.update_handoff(st)
    st["seq"] = 2  # same seq -> no new version
    SUP.update_handoff(st)
    versions = st["handoffs"]
    assert len(versions) == 2
    assert [v["seq"] for v in versions] == [1, 2]


def test_post_handoff_lands_in_receiver_inbox():
    tid_a, st_a = SUP.create_worker("task A", task_id="a")
    tid_b, _ = SUP.create_worker("task B", task_id="b")
    st_a["status"] = "COMPLETE"
    st_a["seq"] = 1
    SUP.update_handoff(st_a)
    msg = SUP.post_handoff(st_a, to_task=tid_b)
    assert msg["kind"] == "handoff"
    assert msg["thread_id"] == "a"  # sender task as correlation thread
    msgs = SUP.list_messages(tid_b, status=SUP.INBOX_PENDING)
    assert any(m["id"] == msg["id"] for m in msgs)
    payload = json.loads(msg["message"])
    assert payload["from_task"] == "a"


# ---------------------------------------------------------------------------
# explicit waiting states + supervision semantics
# ---------------------------------------------------------------------------

def test_evaluate_classifies_waiting_never_kills():
    _, st = SUP.create_worker("t")
    st["status"] = "WAITING"
    st["phase"] = "WAITING"
    st["last_activity_at"] = time.time()
    SUP.save_worker(st, "t")
    d = SUP.evaluate_worker(st, now=time.time(), pid=999999)  # even a dead pid
    assert d.verdict == "WAITING"
    assert d.command == "HOLD"  # never RETRY/CANCEL a waiting worker


def test_needs_input_with_pending_human_message_continues():
    tid, st = SUP.create_worker("t")
    SUP.post_message(tid, "supervisor", "worker", "here is the decision")
    st["status"] = "NEEDS_INPUT"
    st["last_activity_at"] = time.time()
    d = SUP.evaluate_worker(st, now=time.time(), pid=12345)
    assert d.verdict == "NEEDS_INPUT"
    # a pending human/supervisor message = continue (loop delivers it then the
    # worker wakes); the evaluation never auto-restarts or auto-kills waiting
    assert d.command == "CONTINUE"


def test_wait_timeout_reassesses_not_kills():
    _, st = SUP.create_worker("t")
    st["status"] = "WAITING"
    st["budget"]["max_wait_seconds"] = 10
    st["last_activity_at"] = time.time() - 60
    d = SUP.evaluate_worker(st, now=time.time(), pid=12345)
    assert d.verdict == "WAIT_TIMEOUT"
    assert d.command == "REASSESS"


def test_blocked_worker_holds_too():
    _, st = SUP.create_worker("t")
    st["status"] = "BLOCKED"
    st["blockers"] = ["need a human decision"]
    d = SUP.evaluate_worker(st, now=time.time(), pid=12345)
    assert d.verdict == "BLOCKED"
    assert d.command == "HOLD"


def test_worker_class_covers_explicit_states():
    now = time.time()
    # complete / failed / blocked / waiting
    assert SUP.worker_class({"status": "COMPLETE"}) == "complete"
    assert SUP.worker_class({"status": "FAILED"}) == "failed"
    assert SUP.worker_class({"status": "BLOCKED"}) == "blocked"
    assert SUP.worker_class({"status": "WAITING"}) == "waiting"
    assert SUP.worker_class({"status": "NEEDS_INPUT"}) == "waiting"
    # crashed (dead pid)
    assert SUP.worker_class({"status": "TESTING"}, pid=999999) == "crashed"
    # timed_out
    st = {"status": "TESTING", "created_at": now - 7200,
          "budget": {"max_runtime_seconds": 3600}, "last_heartbeat_at": now}
    assert SUP.worker_class(st, pid=None, now=now) == "timed_out"
    # idle (stale heartbeat)
    st = {"status": "TESTING", "created_at": now, "last_heartbeat_at": now - 999,
          "last_activity_at": now - 999}
    assert SUP.worker_class(st, pid=None, now=now) == "idle"
    # active
    st = {"status": "TESTING", "created_at": now, "last_heartbeat_at": now,
          "last_activity_at": now}
    assert SUP.worker_class(st, pid=None, now=now) == "active"


# ---------------------------------------------------------------------------
# watchdog / autonomous recovery
# ---------------------------------------------------------------------------

def test_watchdog_ignores_waiting_and_blocked():
    tid = SUP.create_worker("t")[0]
    for status in ("WAITING", "NEEDS_INPUT", "BLOCKED"):
        st = {"task_id": tid, "status": status, "phase": status,
              "created_at": time.time() - 1000}
        action, _ = SUP.watchdog_assess(st, pid=None, fingerprint="f",
                                        prev_fingerprint="f", stall_count=9)
        assert action == "hold", status


def test_watchdog_stall_requires_stale_heartbeat_and_unchanged_fp():
    now = time.time()
    st = {"task_id": "x", "status": "TESTING", "phase": "TESTING",
          "created_at": now - 10,
          "last_heartbeat_at": now - SUP.STALL_QUIET_SECONDS - 10,
          "last_activity_at": now - SUP.STALL_QUIET_SECONDS - 10, "seq": 1}
    action, count = SUP.watchdog_assess(st, pid=None, fingerprint="f1",
                                        prev_fingerprint="f1",
                                        stall_count=0, now=now)
    assert count == 1
    assert action == "ok"  # below STALL_MIN_TICKS
    action, count = SUP.watchdog_assess(st, pid=None, fingerprint="f1",
                                        prev_fingerprint="f1",
                                        stall_count=count, now=now)
    assert action == "stall"
    # the moment the state changes the counter resets and the stall dissolves
    action, count = SUP.watchdog_assess(st, pid=None, fingerprint="f2",
                                        prev_fingerprint="f1",
                                        stall_count=count, now=now)
    assert action == "ok"
    assert count == 0


def test_watchdog_active_worker_never_stalls():
    now = time.time()
    st = {"task_id": "x", "status": "TESTING", "phase": "TESTING",
          "created_at": now, "last_heartbeat_at": now,
          "last_activity_at": now, "seq": 1}
    action, _ = SUP.watchdog_assess(st, pid=None, fingerprint="f1",
                                    prev_fingerprint="f1", stall_count=9, now=now)
    assert action == "ok"  # heartbeat fresh -> slow-but-alive, not hung


def test_kill_worker_escalates_to_sigkill():
    import subprocess
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert SUP._pid_alive(proc.pid)
        assert SUP.kill_worker(proc.pid, grace=1.0) is True
        assert not SUP._pid_alive(proc.pid)
    finally:
        if SUP._pid_alive(proc.pid):
            proc.kill()


# ---------------------------------------------------------------------------
# cross-worker inbox semantics
# ---------------------------------------------------------------------------

def test_inbox_thread_and_reply_correlation():
    tid, _ = SUP.create_worker("t")
    m1 = SUP.post_message(tid, "wa", "worker", "handoff payload",
                          kind="handoff", thread_id="campaign-a")
    m2 = SUP.post_message(tid, "worker", "wa", "got it",
                          kind="followup", thread_id="campaign-a", reply_to=m1["id"])
    assert m2["reply_to"] == m1["id"]
    assert m2["thread_id"] == "campaign-a"
    by_thread = [m for m in SUP.list_messages(tid) if m.get("thread_id") == "campaign-a"]
    assert len(by_thread) == 2


def test_inbox_dedup_suppresses_duplicate_post():
    tid, _ = SUP.create_worker("t")
    m1 = SUP.post_message(tid, "wa", "worker", "same words", kind="handoff",
                          thread_id="t1")
    m2 = SUP.post_message(tid, "wa", "worker", "same words", kind="handoff",
                           thread_id="t1")
    assert m1["id"] == m2["id"]  # idempotent within the window
    assert len(SUP.list_messages(tid)) == 1


def test_inbox_dedup_cli_default_thread_none_matches_empty():
    """Regression (P8): the CLI passes thread_id=None by default but entries
    store ''; dedup must treat them as equal in a SEPARATE caller process."""
    tid, _ = SUP.create_worker("t")
    # emulate two separate CLI processes: call the module-level boundary as
    # the CLI does — thread_id resolved from None (default)
    env_posts = SUP.post_message(tid, "wa", "worker", "same", kind="handoff",
                                 thread_id=None)
    env_posts2 = SUP.post_message(tid, "wa", "worker", "same", kind="handoff",
                                  thread_id=None)
    assert env_posts["id"] == env_posts2["id"]
    assert len(SUP.list_messages(tid)) == 1
    stored = SUP.list_messages(tid)[0]
    assert stored["thread_id"] == ""


def test_acknowledge_and_check_acks():
    tid, _ = SUP.create_worker("t")
    m = SUP.post_message(tid, "supervisor", "worker", "do X")
    assert SUP.deliver_all_if_idle(tid, {"status": "CREATED", "phase": ""}) == 1
    # worker reports last_acked_msg_id in state -> supervisor flips ledger
    st = {"task_id": tid, "last_acked_msg_id": m["id"]}
    assert SUP.check_acks(tid, st) == m["id"]
    assert SUP.list_messages(tid)[0]["status"] == SUP.INBOX_ACKNOWLEDGED


def test_expire_stale_messages():
    tid, _ = SUP.create_worker("t")
    m = SUP.post_message(tid, "wa", "worker", "old news")
    SUP._mark_message_full(tid, m["id"], SUP.INBOX_PENDING, ts_marker=True)
    # force old ts
    import pathlib
    p = pathlib.Path(str(SUP.inbox_path(tid)))
    lines = p.read_text().splitlines()
    entry = json.loads(lines[0])
    entry["ts"] = time.time() - 99999
    p.write_text(json.dumps(entry) + "\n")
    n = SUP.expire_stale_messages(tid, ttl=100)
    assert n == 1
    assert SUP.list_messages(tid, status=SUP.INBOX_STALE)


def test_duplicate_message_never_delivers_twice():
    tid, _ = SUP.create_worker("t")
    SUP.post_message(tid, "wa", "worker", "single payload", thread_id="t")
    # a second identical post dedups
    SUP.post_message(tid, "wa", "worker", "single payload", thread_id="t")
    assert SUP.deliver_all_if_idle(tid, {"status": "CREATED", "phase": ""}) == 1
    assert SUP.list_messages(tid, status=SUP.INBOX_DELIVERED)


# ---------------------------------------------------------------------------
# lineage / resumption
# ---------------------------------------------------------------------------

def test_worker_lineage_preserves_attempts():
    SUP.create_worker("t", task_id="t")
    st = SUP.load_worker("t")
    assert st is not None
    st["worker_identity"] = "w-t"
    SUP.record_attempt(st, reason="crash", ok=False)
    SUP.bump_attempt(st, reason="crash")
    st["handoffs"] = [{"seq": 1}, {"seq": 2}]
    SUP.save_worker(st, "t")
    lineage = SUP.worker_lineage("t")
    assert lineage["worker_identity"] == "w-t"
    assert lineage["attempt"] == 2
    assert len(lineage["attempts"]) == 2
    assert lineage["handoff_versions"] == 2