"""CAO-comparison gap tests: durable inbox with delivery status, idle-gated
delivery, crash-before-complete detection → auto-retry."""

import sys
import time

import pytest

sys.path.insert(0, ".")
from hermes_cli import supervisor as SUP


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path))
    monkeypatch.setenv("MISSION_DRY_WORKER", "1")  # offline: skip LLM audit stage


def test_post_message_persists_pending():
    tid, _ = SUP.create_worker("t")
    m = SUP.post_message(tid, "supervisor", "worker", "investigate X")
    assert m["status"] == SUP.INBOX_PENDING
    assert SUP.list_messages(tid, status=SUP.INBOX_PENDING) == [m]


def test_deliver_pending_marks_delivered():
    tid, _ = SUP.create_worker("t")
    SUP.post_message(tid, "supervisor", "worker", "reassess")
    n = SUP.deliver_pending(tid)
    assert n == 1
    assert SUP.list_messages(tid, status=SUP.INBOX_PENDING) == []
    delivered = SUP.list_messages(tid, status=SUP.INBOX_DELIVERED)
    assert len(delivered) == 1
    assert delivered[0]["message"] == "reassess"
    # command.json carries the instruction so the worker sees it (P5 envelope
    # adds a correlation prefix the worker can acknowledge).
    import json
    cmd = json.loads(SUP.command_path(tid).read_text())
    assert cmd["instruction"].startswith("[msg:") and cmd["instruction"].endswith("reassess")


def test_idle_gated_delivery():
    tid, st = SUP.create_worker("t")
    SUP.post_message(tid, "supervisor", "worker", "hello")
    # Worker busy (recent activity, non-idle phase): no delivery
    SUP.record_phase(st, "IMPLEMENTING")  # sets last_activity_at=now
    SUP.save_worker(st, tid)
    assert SUP.deliver_all_if_idle(tid, st) == 0
    # Worker idle (stale activity): delivery happens
    st["last_activity_at"] = time.time() - 60
    SUP.save_worker(st, tid)
    assert SUP.deliver_all_if_idle(tid, st) == 1


def test_crash_detection_retries():
    _, st = SUP.create_worker("t")
    SUP.record_phase(st, "TESTING")
    st["last_activity_at"] = time.time()
    d = SUP.evaluate_worker(st, now=time.time(), pid=999999)  # dead pid
    assert d.verdict == "WORKER_CRASH"
    assert d.command == "RETRY"
    assert "retrying" in d.instruction


def test_crash_detection_cancels_after_interventions():
    _, st = SUP.create_worker("t")
    # attempts exhausted (max_worker_attempts=3, attempt 3 = 0 retry left)
    st["attempt"] = 3
    d = SUP.evaluate_worker(st, now=time.time(), pid=999999,
                            interventions_left=0)
    assert d.verdict == "WORKER_CRASH"
    assert d.command == "CANCEL"


def test_live_pid_not_crash():
    _, st = SUP.create_worker("t")
    SUP.record_phase(st, "DIAGNOSING")
    d = SUP.evaluate_worker(st, now=time.time(), pid=__import__("os").getpid())
    assert d.verdict != "WORKER_CRASH"


def test_inbox_bounded_prune():
    tid, _ = SUP.create_worker("t")
    for i in range(SUP._INBOX_MAX + 20):
        SUP.post_message(tid, "s", "w", f"m{i}")
    assert len(SUP.list_messages(tid)) <= SUP._INBOX_MAX


def test_inbox_mark_failed():
    tid, _ = SUP.create_worker("t")
    m = SUP.post_message(tid, "s", "w", "x")
    assert SUP.mark_message(tid, m["id"], SUP.INBOX_FAILED)
    assert SUP.list_messages(tid, status=SUP.INBOX_FAILED)[0]["message"] == "x"