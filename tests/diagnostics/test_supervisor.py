"""Supervisor module tests: worker lifecycle, verdicts, progress detection,
completion gates, cancellation, resource limits, SOUL-3.0 integrity.

Pure state-machine tests (no live model calls); the real worker path is
exercised separately in the end-to-end script.
"""

import json
import os
import sys
import time

import pytest

sys.path.insert(0, ".")
from hermes_cli import supervisor as SUP


@pytest.fixture(autouse=True)
def _isolated_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path))
    monkeypatch.setenv("MISSION_DRY_WORKER", "1")  # offline: skip LLM audit stage


def test_worker_creation_persists_state():
    task_id, state = SUP.create_worker("fix the bug", budget={"max_worker_turns": 10})
    assert state["status"] == "CREATED"
    assert SUP.load_worker(task_id)["status"] == "CREATED"
    assert SUP.worker_path(task_id).exists()
    assert "Autonomous loop" in SUP.worker_path(task_id).parent.joinpath("brief.md").read_text()


def test_worker_lifecycle_phase_machine():
    task_id, state = SUP.create_worker("fix")
    SUP.record_phase(state, "INVESTIGATING")
    SUP.save_worker(state, task_id)
    loaded = SUP.load_worker(task_id)
    assert loaded["phase"] == "INVESTIGATING"
    SUP.record_phase(state, "DIAGNOSING")
    SUP.record_phase(state, "PLANNING")
    SUP.record_phase(state, "IMPLEMENTING")
    SUP.record_tests(state, 47, 47)
    SUP.record_completion(state, ["ran pytest", "all 47 pass"], verification="pytest -q")
    SUP.finish_complete(state)
    SUP.save_worker(state, task_id)
    loaded = SUP.load_worker(task_id)
    assert loaded["status"] == "COMPLETE"
    assert loaded["tests_executed"] == 47


def test_complete_with_evidence_verdict_success():
    _, state = SUP.create_worker("fix")
    state["started_at"] = time.time()  # worker genuinely ran
    SUP.record_tests(state, 47, 47)
    SUP.record_completion(state, ["ran pytest", "all 47 pass"])
    SUP.finish_complete(state)
    d = SUP.evaluate_worker(state, now=time.time())
    assert d.verdict == "SUCCESS"
    assert d.command == "DONE"


def test_complete_without_evidence_rejected():
    _, state = SUP.create_worker("fix")
    state["status"] = "COMPLETE"
    state["completion_evidence"] = ["done"]
    d = SUP.evaluate_worker(state, now=time.time())
    assert d.verdict == "UNVERIFIED_COMPLETION"
    assert d.command == "VERIFY"


def test_no_progress_detected():
    _, state = SUP.create_worker("fix")
    SUP.record_phase(state, "INVESTIGATING")
    state["last_activity_at"] = time.time() - 99999
    d = SUP.evaluate_worker(state, now=time.time())
    assert d.verdict in ("NO_PROGRESS", "WORKER_TIMEOUT")


def test_progress_tests_improved():
    a = {"tests_passed": 40}
    b = {"tests_passed": 47}
    ok, _ = SUP.detect_progress(b, a)
    assert ok


def test_progress_hypothesis_changed():
    a = {"hypothesis": "bug in A"}
    b = {"hypothesis": "bug in B"}
    ok, why = SUP.detect_progress(b, a)
    assert ok and "hypothesis" in why


def test_no_meaningful_progress_same_state():
    a = {"tests_passed": 47, "files_changed": ["x"], "hypothesis": "h", "verification": ""}
    b = {"tests_passed": 47, "files_changed": ["x"], "hypothesis": "h", "verification": ""}
    ok, _ = SUP.detect_progress(b, a)
    assert not ok


def test_repeated_failure_detected():
    _, state = SUP.create_worker("fix")
    for _ in range(4):
        SUP.record_phase(state, "TESTING")
        state["tests_failed"] = 2
    state["last_activity_at"] = time.time()
    d = SUP.evaluate_worker(state, now=time.time())
    assert d.verdict == "REPEATED_FAILURE"


def test_repeated_hypothesis_detected():
    _, state = SUP.create_worker("fix")
    state["hypotheses_seen"] = ["the same cause", "the same cause", "the same cause"]
    state["last_activity_at"] = time.time()
    d = SUP.evaluate_worker(state, now=time.time())
    assert d.verdict == "REPEATED_HYPOTHESIS"
    assert "REASSESS" in d.command


def test_timeout_verdict():
    _, state = SUP.create_worker("fix")
    state["created_at"] = time.time() - 10_000_000
    d = SUP.evaluate_worker(state, now=time.time())
    # P1: timeout now RETRIES (attempt-bounded); only exhausts to CANCEL
    assert d.verdict == "WORKER_TIMEOUT"
    assert d.command == "RETRY"


def test_blocked_verdict():
    _, state = SUP.create_worker("fix")
    state["status"] = "BLOCKED"
    state["blockers"] = ["needs decision"]
    state["last_activity_at"] = time.time()
    d = SUP.evaluate_worker(state, now=time.time())
    assert d.verdict == "BLOCKED"


def test_cancellation_writes_command():
    task_id, state = SUP.create_worker("fix")
    SUP.cancel_worker(task_id)
    cmd = json.loads(SUP.command_path(task_id).read_text())
    assert cmd["command"] == "CANCEL"
    assert SUP.load_worker(task_id)["status"] == "CANCELLED"


def test_budget_exposed_in_state():
    _, state = SUP.create_worker("fix", budget={"max_worker_turns": 10, "max_runtime_seconds": 60})
    assert state["budget"]["max_worker_turns"] == 10
    assert state["budget"]["max_runtime_seconds"] == 60


def test_brief_contains_autonomy_instructions():
    task_id, _ = SUP.create_worker("fix /crawl")
    brief = (SUP.worker_path(task_id).parent / "brief.md").read_text()
    assert "without asking the user to continue" in brief
    assert "diag_env" in brief
    assert "COMPLETE" in brief


def test_persistence_honors_isolated_dir(tmp_path):
    task_id, _ = SUP.create_worker("x")
    assert str(tmp_path) in str(SUP.worker_path(task_id))


def test_soul30_unaffected_by_supervisor():
    """Supervisor evaluation must not import or mutate SOUL-3.0 state."""
    orch = pytest.importorskip("agent.orchestrator")  # SOUL-3.0-only; absent on hosts without the overlay

    before = list(orch.drain_trace())
    SUP.evaluate_worker(
        {"status": "COMPLETE", "completion_evidence": ["t x"], "created_at": 0},
        now=time.time(),
    )
    after = list(orch.drain_trace())
    assert before == after