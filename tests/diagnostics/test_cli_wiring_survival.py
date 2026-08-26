"""CLI wiring survival tests (P-9): `hermes supervise` and `hermes diag`
must remain registered in the CLI router. An upstream Hermes update resets
main.py and silently drops these protected local registrations (measured
2026-08-09: both commands returned usage errors after an upstream reset
deleted the router entries). If the working tree is reset again, these tests
fail FAST instead of the whole hardening suite failing at collection time.
"""
import subprocess
import sys

sys.path.insert(0, ".")

BIN = sys.executable


def _run(*args):
    return subprocess.run([BIN, "-m", "hermes_cli.main", *args],
                          capture_output=True, text=True, timeout=300) # 120s expired when run after heavy suites (measured 2026-08-23)


def test_supervise_cli_present():
    r = _run("supervise", "list")
    # absent router -> usage error (returncode 2); present -> JSON list (0)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "task_id" in r.stdout, r.stdout[:500]


def test_diag_cli_present():
    r = _run("diag", "inventory", "--json")
    assert r.returncode == 0, r.stderr[-1500:]
    assert r.stdout.lstrip().startswith("{"), r.stdout[:500]


def test_diag_json_parent_order():
    """`diag --json inventory` (flag before subcommand) must emit JSON:
    argparse loses parent flags with sub-subcommands, so _dispatch reads the
    raw argv as source of truth (P-9 regression)."""
    r = _run("diag", "--json", "inventory")
    assert r.returncode == 0, r.stderr[-1500:]
    assert r.stdout.lstrip().startswith("{"), r.stdout[:200]


def test_supervise_create_smoke():
    """The supervise router works end to end on a throwaway ledger."""
    import json
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, HERMES_SUPERVISOR_DIR=td)
        r = subprocess.run([BIN, "-m", "hermes_cli.main", "supervise",
                            "create", "--task", "t", "--task-id", "w"],
                           capture_output=True, text=True, timeout=60, env=env)
        assert r.returncode == 0, r.stderr[-1500:]
        p = os.path.join(td, "tasks", "w", "worker.json")
        assert os.path.exists(p), p