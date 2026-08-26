"""Mission-level stale detection (Priority B — mission lifecycle).

Verifies that missions stuck in MISSION_ACTIVE with no live workers
are detected and cancelled, while active missions are preserved.
"""
import json
import os
import time

import pytest

from hermes_cli import supervisor as SUP


def _mission(mid, status="MISSION_ACTIVE", created_at=None, phases=None):
    m = {
        "mission_id": mid,
        "objective": "test",
        "status": status,
        "created_at": created_at or time.time(),
        "updated_at": time.time(),
        "phases": phases or [],
        "requirements": [],
        "criteria_met": [],
        "unresolved_findings": [],
    }
    SUP.save_mission(m)
    return m


def _worker(task_id, status="CREATED", created_at=None):
    state = {
        "task_id": task_id,
        "status": status,
        "task": f"task for {task_id}",
        "created_at": created_at or time.time(),
        "last_activity_at": created_at or time.time(),
        "seq": 1,
    }
    SUP.save_worker(state, task_id)
    return state


def test_cancel_mission_transitions_to_cancelled(tmp_path, monkeypatch):
    """cancel_mission transitions MISSION_ACTIVE to MISSION_CANCELLED."""
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path / "sup"))
    _mission("m1", status="MISSION_ACTIVE")
    result = SUP.cancel_mission("m1", reason="test cancel")
    assert result is True
    m = SUP.load_mission("m1")
    assert m["status"] == "MISSION_CANCELLED"
    assert "test cancel" in m["terminal_rationale"]["reason"]


def test_cancel_mission_rejects_terminal(tmp_path, monkeypatch):
    """cancel_mission refuses to cancel already-terminal missions."""
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path / "sup"))
    _mission("m1", status="MISSION_COMPLETE")
    result = SUP.cancel_mission("m1", reason="test")
    assert result is False


def test_reap_stale_missions_cancels_stale(tmp_path, monkeypatch):
    """MISSION_ACTIVE with no live workers and old age gets cancelled."""
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path / "sup"))
    old = time.time() - 86400 * 30  # 30 days ago
    _mission("m1", status="MISSION_ACTIVE", created_at=old,
             phases=[{"phase_id": "p1", "status": "COMPLETE", "worker_task": "w1"}])
    _worker("w1", status="COMPLETE", created_at=old)
    result = SUP.reap_stale_missions(max_age_seconds=86400 * 14)
    assert "m1" in result["cancelled"]
    m = SUP.load_mission("m1")
    assert m["status"] == "MISSION_CANCELLED"


def test_reap_stale_missions_preserves_recent(tmp_path, monkeypatch):
    """MISSION_ACTIVE younger than threshold is preserved."""
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path / "sup"))
    recent = time.time() - 86400 * 3  # 3 days ago
    _mission("m1", status="MISSION_ACTIVE", created_at=recent)
    result = SUP.reap_stale_missions(max_age_seconds=86400 * 14)
    assert "m1" in result["kept"]
    m = SUP.load_mission("m1")
    assert m["status"] == "MISSION_ACTIVE"


def test_reap_stale_missions_preserves_live_workers(tmp_path, monkeypatch):
    """MISSION_ACTIVE with live workers is preserved even if old."""
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path / "sup"))
    old = time.time() - 86400 * 30
    _mission("m1", status="MISSION_ACTIVE", created_at=old,
             phases=[{"phase_id": "p1", "status": "ACTIVE", "worker_task": "w1"}])
    _worker("w1", status="INVESTIGATING", created_at=old)
    result = SUP.reap_stale_missions(max_age_seconds=86400 * 14)
    assert "m1" in result["kept"]
    m = SUP.load_mission("m1")
    assert m["status"] == "MISSION_ACTIVE"


def test_reap_stale_missions_idempotent(tmp_path, monkeypatch):
    """Running reap twice produces the same result."""
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path / "sup"))
    old = time.time() - 86400 * 30
    _mission("m1", status="MISSION_ACTIVE", created_at=old)
    r1 = SUP.reap_stale_missions(max_age_seconds=86400 * 14)
    r2 = SUP.reap_stale_missions(max_age_seconds=86400 * 14)
    assert r1["cancelled"] == ["m1"]
    assert r2["cancelled"] == []


def test_reap_stale_missions_dry_run(tmp_path, monkeypatch):
    """Dry-run reports what would be cancelled but doesn't cancel."""
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path / "sup"))
    old = time.time() - 86400 * 30
    _mission("m1", status="MISSION_ACTIVE", created_at=old)
    result = SUP.reap_stale_missions(max_age_seconds=86400 * 14, dry_run=True)
    assert "m1" in result["cancelled"]
    m = SUP.load_mission("m1")
    assert m["status"] == "MISSION_ACTIVE"
