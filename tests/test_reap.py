"""Stale worker reaping (Priority B — lifecycle hygiene).

Real-process tests: real worker ledgers under a temp HERMES_SUPERVISOR_DIR.
Verifies that `supervise reap` safely removes only genuinely stale,
unstarted, unreferenced workers — never active, referenced, or recent ones.

Fails on the parent commit: reap_stale_workers does not exist there.
"""
import json
import os
import time

import pytest

from hermes_cli import supervisor as SUP


def _worker(task_id, status="CREATED", created_at=None, mission_id=""):
    """Write a real worker ledger through SUP.save_worker."""
    now = time.time()
    state = {
        "task_id": task_id,
        "status": status,
        "task": f"task for {task_id}",
        "created_at": created_at or now,
        "last_activity_at": created_at or now,
        "mission_id": mission_id,
        "seq": 1,
    }
    SUP.save_worker(state, task_id)
    return state


def _mission(mid, worker_tasks=()):
    """Write a mission that references the given worker task_ids."""
    m = {
        "mission_id": mid,
        "objective": "test",
        "status": "MISSION_ACTIVE",
        "created_at": time.time(),
        "phases": [{"phase_id": f"p{i}", "status": "ACTIVE", "worker_task": wt}
                   for i, wt in enumerate(worker_tasks)],
        "requirements": [],
        "criteria_met": [],
        "unresolved_findings": [],
    }
    SUP.save_mission(m)
    return m


def test_reap_removes_old_created_workers(tmp_path, monkeypatch):
    """CREATED workers older than threshold with no mission reference get reaped."""
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path / "sup"))
    old = time.time() - 86400 * 30  # 30 days ago
    _worker("stale1", status="CREATED", created_at=old)
    _worker("stale2", status="CREATED", created_at=old)
    result = SUP.reap_stale_workers(max_age_seconds=86400 * 7)
    assert "stale1" in result["reaped"]
    assert "stale2" in result["reaped"]
    assert SUP.load_worker("stale1") is None
    assert SUP.load_worker("stale2") is None


def test_reap_preserves_recent_workers(tmp_path, monkeypatch):
    """CREATED workers newer than threshold are NOT reaped."""
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path / "sup"))
    recent = time.time() - 3600  # 1 hour ago
    _worker("fresh", status="CREATED", created_at=recent)
    result = SUP.reap_stale_workers(max_age_seconds=86400 * 7)
    assert "fresh" not in result["reaped"]
    assert SUP.load_worker("fresh") is not None


def test_reap_preserves_active_workers(tmp_path, monkeypatch):
    """Non-CREATED workers (COMPLETE, INVESTIGATING, etc.) are never reaped."""
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path / "sup"))
    old = time.time() - 86400 * 30
    _worker("done", status="COMPLETE", created_at=old)
    _worker("working", status="INVESTIGATING", created_at=old)
    _worker("blocked", status="BLOCKED", created_at=old)
    result = SUP.reap_stale_workers(max_age_seconds=86400 * 7)
    assert result["reaped"] == []
    assert SUP.load_worker("done") is not None
    assert SUP.load_worker("working") is not None
    assert SUP.load_worker("blocked") is not None


def test_reap_preserves_mission_referenced_workers(tmp_path, monkeypatch):
    """CREATED workers referenced by a mission are NOT reaped."""
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path / "sup"))
    old = time.time() - 86400 * 30
    _worker("referenced", status="CREATED", created_at=old)
    _mission("m1", worker_tasks=["referenced"])
    result = SUP.reap_stale_workers(max_age_seconds=86400 * 7)
    assert "referenced" not in result["reaped"]
    assert SUP.load_worker("referenced") is not None


def test_reap_idempotent(tmp_path, monkeypatch):
    """Running reap twice produces the same result, no errors."""
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path / "sup"))
    old = time.time() - 86400 * 30
    _worker("dup", status="CREATED", created_at=old)
    r1 = SUP.reap_stale_workers(max_age_seconds=86400 * 7)
    r2 = SUP.reap_stale_workers(max_age_seconds=86400 * 7)
    assert r1["reaped"] == ["dup"]
    assert r2["reaped"] == []


def test_reap_dry_run_does_not_delete(tmp_path, monkeypatch):
    """Dry-run reports what would be reaped but does not delete."""
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path / "sup"))
    old = time.time() - 86400 * 30
    _worker("dry", status="CREATED", created_at=old)
    result = SUP.reap_stale_workers(max_age_seconds=86400 * 7, dry_run=True)
    assert "dry" in result["reaped"]
    # Worker must still exist
    assert SUP.load_worker("dry") is not None
