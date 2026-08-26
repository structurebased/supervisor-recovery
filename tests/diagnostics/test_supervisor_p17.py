#!/usr/bin/env python3
"""P-17 regressions: write_command_if_changed skips identical decisions.

FAIL on parent: write_command_if_changed does not exist (the loops called
write_command every tick, re-writing the identical command.json).
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
REAL_BIN = "/home/hamza/.hermes/hermes-agent/venv/bin/hermes"
from hermes_cli import supervisor as S  # noqa: E402


def _hermes(base, *args, timeout=60):
    env = dict(os.environ)
    env["HERMES_SUPERVISOR_DIR"] = str(base)
    return subprocess.run([REAL_BIN, *args], capture_output=True, text=True,
                          env=env, timeout=timeout)


def _cmd(base, task):
    p = pathlib.Path(base) / "tasks" / task / "command.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _cmd_nonnull(base, task) -> dict:
    c = _cmd(base, task)
    assert c is not None, "command.json missing"
    return c


def test_identical_continue_not_rewritten(tmp_path):
    r = _hermes(tmp_path, "supervise", "create", "--task", "t",
                "--task-id", "w")
    assert r.returncode == 0, r.stderr
    os.environ["HERMES_SUPERVISOR_DIR"] = str(tmp_path)
    d = S.WorkerDecision("NO_VERDICT_YET", "CONTINUE", "keep going", 0.5)
    p1 = S.write_command_if_changed("w", d)
    assert p1 is not None and p1.exists()
    c1 = _cmd(tmp_path, "w")
    # second identical write must not touch mtime/content
    before = (p1.stat().st_mtime_ns, p1.read_bytes())
    p2 = S.write_command_if_changed("w", d)
    after = (p2.stat().st_mtime_ns, p2.read_bytes())
    assert before == after, "identical decision must not rewrite the file"
    assert c1["verdict"] == "NO_VERDICT_YET"


def test_changed_instruction_is_written(tmp_path):
    r = _hermes(tmp_path, "supervise", "create", "--task", "t",
                "--task-id", "w")
    assert r.returncode == 0, r.stderr
    os.environ["HERMES_SUPERVISOR_DIR"] = str(tmp_path)
    d1 = S.WorkerDecision("NO_VERDICT_YET", "CONTINUE", "keep going", 0.5)
    S.write_command_if_changed("w", d1)
    d2 = S.WorkerDecision("NO_VERDICT_YET", "CONTINUE", "now check tests", 0.5)
    S.write_command_if_changed("w", d2)
    c = _cmd_nonnull(tmp_path, "w")
    assert c["instruction"] == "now check tests"


def test_changed_command_is_written(tmp_path):
    _hermes(tmp_path, "supervise", "create", "--task", "t", "--task-id", "w")
    os.environ["HERMES_SUPERVISOR_DIR"] = str(tmp_path)
    S.write_command_if_changed("w", S.WorkerDecision("NO_VERDICT_YET", "CONTINUE", "x", 0.5))
    S.write_command_if_changed("w", S.WorkerDecision("WORKER_FAILURE", "RETRY", "retry now", 0.2))
    c = _cmd_nonnull(tmp_path, "w")
    assert c["command"] == "RETRY" and c["verdict"] == "WORKER_FAILURE"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])