"""Bare-worker recovery (Priority D — lifecycle semantics).

Verifies that a worker stuck in CREATED (never started) is detected and
recovered: RETRY when attempts remain, CANCEL when exhausted.

Fails on the parent commit: CREATED detection block does not exist there.
"""
import json
import os
import time

import pytest

from hermes_cli import supervisor as SUP


def _created_worker(task_id, created_at=None, attempts=0):
    """Create a worker in CREATED state with no started_at/worker_pid."""
    state = {
        "task_id": task_id,
        "status": "CREATED",
        "task": f"task for {task_id}",
        "created_at": created_at or time.time(),
        "last_activity_at": created_at or time.time(),
        "seq": 1,
        "attempts": attempts,
        "budget": dict(SUP.DEFAULT_BUDGET),
    }
    SUP.save_worker(state, task_id)
    return state


def test_created_within_budget_holds():
    """Fresh CREATED worker within idle budget returns NO_VERDICT_YET."""
    now = time.time()
    w = {"task_id": "t1", "status": "CREATED", "created_at": now,
         "last_activity_at": now, "seq": 1, "attempts": 0,
         "budget": dict(SUP.DEFAULT_BUDGET)}
    decision = SUP.evaluate_worker(w, now=now + 60)  # 60s later, within 600s budget
    assert decision.verdict == "NO_VERDICT_YET"
    assert decision.command == "CONTINUE"


def test_created_beyond_budget_retries():
    """CREATED worker beyond idle budget with attempts left returns WORKER_CRASH/RETRY."""
    now = time.time()
    w = {"task_id": "t1", "status": "CREATED", "created_at": now - 1200,
         "last_activity_at": now - 1200, "seq": 1, "attempts": 0,
         "budget": dict(SUP.DEFAULT_BUDGET)}
    decision = SUP.evaluate_worker(w, now=now)
    assert decision.verdict == "WORKER_CRASH"
    assert decision.command == "RETRY"


def test_created_beyond_budget_exhausted_cancels():
    """CREATED worker beyond idle budget with no attempts left returns CANCEL."""
    now = time.time()
    w = {"task_id": "t1", "status": "CREATED", "created_at": now - 1200,
         "last_activity_at": now - 1200, "seq": 1,
         "attempt": 6,  # attempt_number reads this field
         "budget": dict(SUP.DEFAULT_BUDGET)}
    decision = SUP.evaluate_worker(w, now=now)
    assert decision.verdict == "WORKER_CRASH"
    assert decision.command == "CANCEL"


def test_created_with_started_at_not_affected():
    """CREATED worker that has started_at is NOT treated as bare."""
    now = time.time()
    w = {"task_id": "t1", "status": "CREATED", "created_at": now - 1200,
         "started_at": now - 1200, "last_activity_at": now - 1200,
         "seq": 1, "attempts": 0, "budget": dict(SUP.DEFAULT_BUDGET)}
    # With started_at set, the CREATED detection should not fire
    decision = SUP.evaluate_worker(w, now=now)
    # Should fall through to other paths (idle/timeout)
    assert decision.verdict != "WORKER_CRASH" or "CREATED" not in decision.reason
