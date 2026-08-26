"""Focused verification of the 5th-reset recovery artifacts (2026-08-18).

Covers the changed paths:
- hermes_cli/supervisor.py  -> _supervisor_base() HOME-independent anchor
- hermes_cli/main.py        -> cmd wiring + supervise fast path, no dup
- wire_main.py              -> idempotent re-run, syntax guard
- test_supervisor_upgrade   -> covered by pytest in the shell verifier
"""

import os
import subprocess
import sys

REPO = "/home/hamza/.hermes/hermes-agent"


def main() -> int:
    # 1. _supervisor_base HOME-independence (5th-reset defect)
    os.environ["HOME"] = "/home/hamza/.hermes/profiles/haress-supervisor/home"
    os.environ.pop("HERMES_SUPERVISOR_DIR", None)
    sys.path.insert(0, REPO)
    from hermes_cli import supervisor as S

    base = S._supervisor_base()
    assert str(base) == "/home/hamza/.hermes-supervisor", base
    assert S.worker_path("accept1").exists(), "accept1 ledger missing"
    print("PASS supervisor_base HOME-independent, accept1 ledger found")

    # 2. wire_main.py idempotency + zero duplication
    r = subprocess.run(
        [sys.executable, "/home/hamza/.hermes-supervisor/recovery/wire_main.py",
         f"{REPO}/hermes_cli/main.py"],
        capture_output=True, text=True,
    )
    print("wire_main rc:", r.returncode, "| out:", r.stdout.strip()[:90])
    assert r.returncode == 0 and "already wired" in r.stdout, r.stdout
    s = open(f"{REPO}/hermes_cli/main.py", encoding="utf-8").read()
    assert s.count("def cmd_supervise(args):") == 1
    assert s.count("def _try_supervise_fast_path") == 1
    assert s.count("if _try_supervise_fast_path():") == 1
    assert s.count('"supervise",') == 1
    print("PASS wire_main idempotent, wiring present exactly once")

    # 3. watchdog script: healthy -> silent exit 0
    r = subprocess.run(
        ["bash", "/home/hamza/.hermes-supervisor/recovery/supervisor-survival-watch.sh"],
        capture_output=True, text=True,
    )
    print("watchdog rc:", r.returncode, "| stdout bytes:", len(r.stdout))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "", "watchdog should be silent when healthy"
    print("PASS watchdog silent-healthy")

    return 0


if __name__ == "__main__":
    sys.exit(main())