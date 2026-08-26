"""Hardening driver — exercises the REAL supervisor with real subprocesses
and the real state-write/CAS protocol, from a separate OS process.

Modes:
  worker <task_id> <script.json>   — scripted worker: writes real state-write
                                     sequence (simulating a hermes worker's
                                     CAS protocol), then exits per script.
  loop   <task_id> [--every N] [--max-seconds M] [--campaign C --role R]
                                     — runs the REAL supervise loop as the
    "supervisor process" so the test can kill and restart it.

Usage (from tests/diagnostics/test_supervisor_hardening.py):
  proc = subprocess.run([...supervise loop TASK...], ...)   # real supervisor
  kill it, start a fresh one, confirm recovery from ledgers.
"""
import json
import os
import subprocess
import sys
import time

SUP_DIR = os.environ.get("HERMES_SUPERVISOR_DIR") or os.path.expanduser("~/.hermes-supervisor")
# Driver's own CLI calls MUST use the real binary; HERMES_BIN may be a crash
# stub used ONLY for the supervisor's worker respawns.
BIN = os.environ.get("HERMES_DRIVER_BIN") or os.environ.get("HERMES_BIN", "hermes")


def _state_path(task_id: str) -> str:
    return os.path.join(SUP_DIR, "tasks", task_id, "worker.json")


def current_seq(task_id: str) -> int:
    try:
        with open(_state_path(task_id)) as f:
            return int(json.load(f).get("seq") or 0)
    except Exception:
        return 0


def _anchor_pid(task_id: str) -> None:
    """Register THIS process as the worker's pid via the supervisor-side
    `attach` CLI (CAS-protected, same path a real `start`/RETRY uses).

    A worker-initiated `state-write` CANNOT anchor it: worker_pid and
    started_at are protected keys in apply_worker_state, so the old
    state-write "anchor" was a silent no-op and every scripted worker ran
    unanchored — the supervisor's P10/P11 completion guards then correctly
    refuse unanchored work and the loop never reaches SUCCESS. `attach`
    goes through record_spawned_pid() — the supervisor's own border.
    """
    for _ in range(8):
        r = subprocess.run([BIN, "supervise", "attach", task_id,
                            "--pid", str(os.getpid())],
                           capture_output=True, text=True)
        if r.returncode == 0 and "attached" in r.stdout:
            return
        time.sleep(0.05)
    # best-effort anchor: if all writes raced/staled, the loop will still see
    # the state file and the worker's actual pid via /proc when it probes.


def worker_main(task_id: str, script_path: str) -> int:
    """Execute a scripted worker sequence using the REAL state-write CLI."""
    script = json.load(open(script_path))
    # Real workers register their pid when the supervisor starts them. A
    # scripted driver under test must do the same or the loop treats it as a
    # dead/unknown process and crash-RETRYs, racing the worker's own writes
    # at scale (measured: 20-worker test reached attempt=3 and none
    # completed). Inject worker_pid before the first write so the supervisor
    # sees a live, owned process — exactly what `start` records for real
    # workers.
    _anchor_pid(task_id)
    for i, step in enumerate(script):
        kind = step.get("kind", "write")
        if kind == "write":
            patch = step.get("patch", {})
            # Real workers follow the documented CAS protocol: on a stale
            # write they re-read seq and retry (the supervisor loop bumps seq
            # via record_spawned_pid when it starts observing a CREATED
            # worker, so a naive expect-seq can race it). Retry up to 5 times.
            accepted = False
            r = None
            for _attempt in range(5):
                seq = current_seq(task_id)
                args = [BIN, "supervise", "state-write", task_id,
                        "--expect-seq", str(seq), "--json", json.dumps(patch)]
                r = subprocess.run(args, capture_output=True, text=True)
                if r.returncode == 0 and "accepted" in r.stdout:
                    accepted = True
                    break
                time.sleep(0.05)
            if not accepted:
                print(f"[worker] step {i} stale/rejected after retries: "
                      f"{r.stdout.strip() if r else ''}")
                return 2
            time.sleep(0.05)
        elif kind == "sleep":
            time.sleep(float(step.get("seconds", 1)))
        elif kind in ("die", "exit"):
            print(f"[worker] {kind} at step {i}")
            return 0
        elif kind == "publish":
            r = subprocess.run(
                [BIN, "supervise", "message", step["to"],
                 "--text", step.get("text", "handoff"),
                 "--kind", "handoff", "--sender", task_id],
                capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[worker] publish failed: {r.stderr.strip()}")
        else:
            print(f"[worker] unknown step {kind}")
    return 0


def loop_main(task_id: str, every: float = 1.0, max_seconds: float = 60.0,
              campaign: str = "", role: str = "", max_replacements: int = 0) -> int:
    args = [BIN, "supervise", "loop", task_id, "--every", str(every),
            "--max-seconds", str(max_seconds)]
    if campaign:
        args += ["--campaign", campaign, "--role", role,
                 "--max-replacements", str(max_replacements)]
    r = subprocess.run(args, capture_output=True, text=True)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print(r.stderr[-1000:])
    return r.returncode


if __name__ == "__main__":
    mode = sys.argv[1]
    if len(sys.argv) < 3:
        print("usage: hardening_driver.py worker TASK SCRIPT | loop TASK [args]")
        sys.exit(2)
    task = sys.argv[2]
    if mode == "worker":
        sys.exit(worker_main(task, sys.argv[3]))
    elif mode == "loop":
        # parse --every/--max-seconds/--campaign/--role/--max-replacements
        every = 1.0
        max_seconds = 60.0
        campaign = role = ""
        max_replacements = 0
        i = 3
        while i < len(sys.argv):
            a = sys.argv[i]
            if a == "--every" and i + 1 < len(sys.argv):
                every = float(sys.argv[i + 1]); i += 2
            elif a == "--max-seconds" and i + 1 < len(sys.argv):
                max_seconds = float(sys.argv[i + 1]); i += 2
            elif a == "--campaign" and i + 1 < len(sys.argv):
                campaign = sys.argv[i + 1]; i += 2
            elif a == "--role" and i + 1 < len(sys.argv):
                role = sys.argv[i + 1]; i += 2
            elif a == "--max-replacements" and i + 1 < len(sys.argv):
                max_replacements = int(sys.argv[i + 1]); i += 2
            else:
                i += 1
        sys.exit(loop_main(task, every=every, max_seconds=max_seconds,
                           campaign=campaign, role=role,
                           max_replacements=max_replacements))
    else:
        print("unknown mode")
        sys.exit(2)