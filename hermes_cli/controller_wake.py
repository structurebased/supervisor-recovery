"""Cron monitor entrypoint for the mission controller wake.

Executed by the per-mission shim (HERMES_HOME/scripts/supervisor-mission-wake-*.sh)
written by ``hermes_cli.controller.arm`` each tick of the controller cron job.
Prints the change-detection source for cron/monitor.py: empty output = no new
meaningful events (suppresses the agent run entirely); non-empty = the
controller session wakes and reasons about the events.

Usage: python -m hermes_cli.controller_wake <mission_id> [LEVEL]
"""
from __future__ import annotations

import os
import sys


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        print("usage: controller_wake <mission_id> [LEVEL]", file=sys.stderr)
        return 2
    from hermes_cli.controller import wake_output

    mission_id = argv[0]
    level = argv[1] if len(argv) > 1 else "MEDIUM"
    if os.environ.get("WAKE_DEBUG"):
        from hermes_cli import mission_ops as MO
        print(f"DBG mission={mission_id!r} level={level!r} "
              f"journal={MO.mission_events_path(mission_id)} "
              f"exists={MO.mission_events_path(mission_id).exists()}",
              file=sys.stderr)
    print(wake_output(mission_id, level=level), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))