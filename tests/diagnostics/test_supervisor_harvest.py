#!/usr/bin/env python3
"""P-26 s2 (2026-08-18): exhausted/crashed workers' uncommitted worktree
evidence must be harvested into the ledger.

Failure class (observed live): the s2-resilience-audit worker was STALL-
killed 3/3 attempts with its ONLY real finding (the corrupt-ledger fix +
regression test) sitting UNCOMMITTED in the repo worktree. A human
controller rescued it from `git status`. Without a harvest mechanism, an
autonomous org silently loses a dead worker's work the moment the tree
moves on.

This test drives harvest_worker_worktree() against a REAL temp git repo:
a clean workdir harvests nothing (ok=true, status=[]); an uncommitted file
is captured verbatim into the worker ledger (harvest.status_lines >= 1
with the exact file listed), and the ledger now carries `harvest`.
FAILS on the parent tree (function absent / no workdir persistence).
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from hermes_cli import supervisor as SUP  # noqa: E402


def _git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args],
                          capture_output=True, text=True, timeout=30)


def _make_repo():
    repo = tempfile.mkdtemp(prefix="s2-harvest-")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "base")
    return repo


def test_clean_worktree_harvests():
    repo = _make_repo()
    tid, _st = SUP.create_worker("test", task_id="s3-harvest-clean",
                                 workdir=repo)
    try:
        hv = SUP.harvest_worker_worktree(tid) or {}
        assert hv.get("ok") is True, hv
        assert hv.get("status") == [], hv.get("status")
        cur = SUP.load_worker(tid) or {}
        assert (cur.get("harvest") or {}).get("ok") is True, cur
    finally:
        _git(repo, "init", "-q")  # noop guard; repo dir cleaned by tmpdir


def test_uncommitted_work_is_harvested_into_ledger():
    repo = _make_repo()
    tid, _ = SUP.create_worker("test", task_id=os.path.join(repo, "-w2"),
                               workdir=repo)
    try:
        probe = os.path.join(repo, "findings.md")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("# stranded finding\n")
        hv = SUP.harvest_worker_worktree(tid)
        assert hv.get("ok") is True, hv
        assert hv.get("status_lines", 0) >= 1, hv
        joined = "\n".join(hv.get("status", []))
        assert "findings.md" in joined, joined
        cur = SUP.load_worker(tid)
        assert "findings.md" in "\n".join(cur.get("harvest", {}).get("status", [])), cur
    finally:
        pass


def test_no_workdir_harvest_is_benign():
    tid, _ = SUP.create_worker("test", task_id="s2-harvest-nwd",
                               workdir="")
    hv = SUP.harvest_worker_worktree(tid)
    assert hv.get("ok") is False, hv
    assert hv.get("reason") == "no-workdir", hv