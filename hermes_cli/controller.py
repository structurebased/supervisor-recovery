"""Controller activation — durable, event-triggered controller-level reasoning.

Supervisor autonomy is solved (worker -> event -> supervisor -> state -> replan).
This adds the controller half of the same loop for the initiating session:

    worker/supervisor -> durable event (mission journal)
        -> controller wake (cron monitor job, hash-suppressed)
        -> controller Hermes session reasons about the event
        -> issues durable evidence-gated commands via the existing
           `hermes supervise` CLI, or deliberately does nothing
        -> acks the wake watermark (idempotent consumption)

The supervisor remains the durable execution authority (workers/phases/loops).
The controller never runs the mission loop and never owns worker lifecycle; it
only reads durable state and issues the same CLI commands a human would.
No new frameworks: the wake is a native cron monitor job (cron/monitor.py),
the controller is an ordinary Hermes cron session, and per-mission controller
state is one JSON file beside the mission ledger.

Wake semantics
--------------
wake_output() = new journal events since the controller's last --ack
watermark, filtered by level. The cron monitor fires only when this output
CHANGES:

- first arm  -> one baseline run (controller reviews the mission)
- new event  -> run (controller reasons; may issue work)
- --ack      -> watermark advances; the NEXT tick output (empty) differs, so
  one trailing idle run occurs; the controller resolves it as a NoOp. The
  trailing wake is the price of crash-safe consumption (the watermark only
  advances after a controller session actually processed) and is bounded.
"""
from __future__ import annotations

import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# per-mission controller state (durable, beside the mission ledger)
# ---------------------------------------------------------------------------


def controller_path(mission_id: str) -> Path:
    from hermes_cli import supervisor as S
    return S.missions_dir() / mission_id / "controller.json"


def load_controller(mission_id: str) -> Dict[str, Any]:
    c = {"mission_id": mission_id, "armed": False, "watermark": 0,
         "created_at": None, "updated_at": None, "job_id": "", "acks": 0,
         "last_ack_at": None}
    p = controller_path(mission_id)
    if p.exists():
        try:
            c.update(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return c


def _save_controller(c: Dict[str, Any]) -> Path:
    c["updated_at"] = time.time()
    p = controller_path(c["mission_id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(c, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, p)
    return p


# ---------------------------------------------------------------------------
# wake read / acknowledge — the only controller-mutable state besides config
# ---------------------------------------------------------------------------


def _journal_len(mission_id: str) -> int:
    from hermes_cli import mission_ops as MO
    p = MO.mission_events_path(mission_id)
    if not p.exists():
        return 0
    n = 0
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
    except OSError:
        pass
    return n


def new_events(mission_id: str, level: str = "LOW") -> List[Dict[str, Any]]:
    """Journal events after the controller watermark, filtered by level."""
    from hermes_cli import mission_ops as MO
    c = load_controller(mission_id)
    wm = int(c.get("watermark") or 0)
    want = MO.LEVEL_ORDER.get(str(level).upper(), MO.LEVEL_ORDER["LOW"])
    rows: List[Dict[str, Any]] = []
    n = 0
    p = MO.mission_events_path(mission_id)
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    n += 1
                    if n <= wm:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if MO.LEVEL_ORDER.get(row.get("level", "LOW"), 0) >= want:
                        rows.append(row)
        except OSError:
            pass
    return rows


def ack(mission_id: str, *, who: str = "") -> Dict[str, Any]:
    """Advance the watermark to the current journal length. Idempotent: an
    ack on an already-acked mission is a successful no-op."""
    c = load_controller(mission_id)
    total = _journal_len(mission_id)
    unread = total - int(c.get("watermark") or 0)
    c["watermark"] = total
    c["acks"] = int(c.get("acks") or 0) + 1
    c["last_ack_by"] = who
    c["last_ack_at"] = time.time()
    _save_controller(c)
    return {"mission_id": mission_id, "watermark": total, "consumed": unread,
            "acks": c["acks"]}


def wake_output(mission_id: str, level: str = "MEDIUM") -> str:
    """Stable monitor output with at-least-once redelivery semantics.

    Returns what the cron monitor hash-compares each tick:

    - no unacked events  -> ""
    - new unacked batch  -> the event digest (triggers the wake once), with
      the digest recorded in controller.json
    - same unacked batch within the redelivery window -> the digest again
      (hash-equal => scheduler suppresses; no storm)
    - same unacked batch past the redelivery window -> digest + a
      "[redelivery N]" marker (hash-changed => the controller wakes again)
      and the retry counter advances

    The redelivery path is the crash recovery for a controller killed before
    --ack: the same events are re-presented (bounded: at most once per
    `CONTROLLER_REDELIVER_WINDOW` seconds, default 300), so a replacement
    controller reconstructs state from the journal + ledger and re-derives
    its action (ledger writes are idempotent). Acknowledgement advances the
    watermark, which empties this output, which produces the single ack-echo
    wake.
    """
    evs = new_events(mission_id, level=level)
    if not evs:
        return ""
    lines = []
    for e in evs:
        lines.append(f"[{e.get('level', 'LOW')}] {e.get('kind', 'EVENT')}: "
                     f"{e.get('message', '')[:200]}")
    digest = "\n".join(lines)
    c = load_controller(mission_id)
    window = _redeliver_window()
    now = time.time()
    emitted_digest = c.get("emitted_digest") or ""
    emitted_at = float(c.get("emitted_at") or 0)
    if emitted_digest != digest:
        c["emitted_digest"] = digest
        c["emitted_at"] = now
        _save_controller(c)
        return digest
    if now - emitted_at >= window:
        n = int(c.get("redeliveries") or 0) + 1
        c["redeliveries"] = n
        c["emitted_at"] = now
        _save_controller(c)
        return digest + f"\n[redelivery {n}]"
    return digest


def _redeliver_window() -> float:
    try:
        return max(0.0, float(os.environ.get("CONTROLLER_REDELIVER_WINDOW", "300")))
    except ValueError:
        return 300.0


# ---------------------------------------------------------------------------
# arm / disarm — a durable cron monitor job is the native wake mechanism
# ---------------------------------------------------------------------------

WAKE_LEVELS = {"LOW": "LOW", "MEDIUM": "MEDIUM", "HIGH": "HIGH"}

CONTROLLER_BRIEF_TMPL = """You are the autonomous controller for mission {mission_id}.

Mission objective: {objective}
Mission ledger: {sups_dir}

If your shell's HERMES_SUPERVISOR_DIR is not {sups_dir}, prefix every
`hermes supervise` call with HERMES_SUPERVISOR_DIR={sups_dir} (the cron
shell sanitizes env). The monitor that woke you already uses this dir.

You are woken ONLY when a meaningful event occurred on the mission (cron
monitor job; no new events = no wake, and an idle ack-trail run tells you to
do nothing).

Layering: the SUPERVISOR owns workers, phases, and discovery execution. You
never run the mission loop and never manage worker lifecycle. You own only
the controller role: inspect durable state, reason, and issue durable
evidence-gated commands through the standard `hermes supervise mission` CLI.

New events since your last controller turn:

{events})

Your job:
1. If there are no new events above, this is the idle ack trail: do nothing.
2. Otherwise inspect durable state:
     hermes supervise mission status {mission_id}
     hermes supervise mission todo {mission_id}
     hermes supervise mission backlog {mission_id} --list
     hermes supervise mission telemetry {mission_id}
     hermes supervise mission events {mission_id} --limit 50 --level MEDIUM
     hermes supervise mission evaluate {mission_id}
3. Decide what controller-level action genuinely helps:
   - new work for the supervisor to pick up:
       hermes supervise mission discover {mission_id} --title "..." --why "..."
   - backlog work that should become real work (the durable continuation
     source after phases/discoveries run out):
       hermes supervise mission backlog {mission_id} --materialize
   - challenge completion/evidence:
       hermes supervise mission finding {mission_id} <id> --text "..."
   - a phase the supervisor missed:
       hermes supervise mission phase-blocked {mission_id} <phase> --note "..."
   - deliberately do nothing when the supervisor already handled it.
   Never write ledger state that already exists (idempotent by construction).
4. If you processed any event, acknowledge it:
       hermes supervise mission controller {mission_id} --ack
5. Supervisor lifecycle. Check `hermes supervise mission lease {mission_id}`:
   if the lease is ABSENT or shows a dead/stale holder (alive=false or
   stale=true) AND the mission still has open phases, TODO discoveries, or
   OPEN backlog items, resume the executor ONCE, in the BACKGROUND (never
   block on it):
       hermes supervise mission loop {mission_id} --every 60 --max-seconds 86400
   The loop takes over the lease itself; if a live supervisor already
   exists it refuses and exits, so you can never create a second authority.
   Also treat MISSION_LOOP_EXITED (with no later MISSION_STARTED) the same
   way. Do NOT resume when the lease is held by an alive pid.
6. Close the learn loop: whenever you handle events, run
       hermes supervise mission metrics {mission_id}
       hermes supervise mission retrospective {mission_id}
   then materialize any open backlog items (they are the self-continuing
   work that keeps the mission alive after 'tests pass'):
       hermes supervise mission backlog {mission_id} --materialize
   and if the metrics show waste signals, run
       hermes supervise mission optimize {mission_id} --apply
   (safe: evidence-backed discoveries, deduped by slug, promoted by the
   planner and executed by the supervisor on the next loop pass).
7. TERMINAL GATE — before ever concluding the mission is satisfied, run
       hermes supervise mission gate {mission_id}
   Exit 0 = may stop (the evaluator + diminishing-return gates are closed:
   phases/discoveries/DoD/backlog all resolve). Exit 1 = meaningful work
   remains; you must NOT report the objective as satisfied — instead create
   the work (backlog/discover), ensure the loop runs, and end the turn only
   with 'CONTINUES: <reason>', not 'DONE'. 'tests green', 'commit created',
   and 'verification finished' are never terminal conditions by themselves.
8. PERFORMANCE REASSESSMENT — when a PERFORMANCE_REASSESSMENT event appears
   (a worker ran far past its expected span), treat it as a question to
   answer, not a fact to log:
       hermes supervise mission telemetry {mission_id}      # where the time went
       hermes supervise mission env {mission_id}             # machine model
       hermes supervise mission backlog {mission_id} --list  # alternatives
       hermes supervise mission lessons {mission_id}         # prior attempts
   Ask: is the delay justified by the objective? Is there a built-in
   capability, skill, or MCP tool that does this faster? Should it be
   parallelized or abandoned? If an alternative exists, record it and let
   the bounded strategy switch pick it up:
       hermes supervise mission discover {mission_id} --title "[strategy] ..." --why "..."
   If the delay is justified (network wait, genuinely long objective), do
   nothing beyond acknowledging in your ACTION TAKEN line. Never kill a
   worker for slowness alone.
9. End your turn with exactly two lines: CURRENT STATE and ACTION TAKEN.

{policy}
"""


def _scripts_dir() -> Path:
    from hermes_constants import get_hermes_home
    d = get_hermes_home() / "scripts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _wake_script_path(mission_id: str) -> Path:
    return _scripts_dir() / f"supervisor-mission-wake-{mission_id}.sh"


def _write_wake_script(mission_id: str, level: str) -> Path:
    """Per-mission shim the cron monitor executes. Hardcodes
    HERMES_SUPERVISOR_DIR + interpreter so it works even when the cron
    child env is sanitized (secrets stripped). CONTROLLER_REDELIVER_WINDOW
    is forwarded when set at arm time (tests/short-redelivery demos)."""
    sups = os.environ.get("HERMES_SUPERVISOR_DIR") or os.path.expanduser(
        "~/.hermes-supervisor")
    py = _sys_bin()
    body = (
        "#!/bin/sh\n"
        f"export HERMES_SUPERVISOR_DIR={shlex.quote(sups)}\n"
        + (f"export CONTROLLER_REDELIVER_WINDOW="
           f"{shlex.quote(os.environ['CONTROLLER_REDELIVER_WINDOW'])}\n"
           if os.environ.get("CONTROLLER_REDELIVER_WINDOW") else "")
        + f'exec {shlex.quote(py)} -m hermes_cli.controller_wake '
        f"{shlex.quote(mission_id)} {shlex.quote(level)}\n"
    )
    p = _wake_script_path(mission_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    p.chmod(0o755)
    return p


def _sys_bin() -> str:
    return sys.executable or "python3"


def arm(mission_id: str, *, level: str = "MEDIUM",
        interval: str = "every 5m", deliver: str = "local",
        policy: str = "", model: Optional[str] = None) -> Dict[str, Any]:
    """Create/refresh the durable controller wake for a mission.

    Idempotent: re-arming with an existing job UPDATES the same job (one
    controller per mission); the watermark is preserved so old events are
    never replayed. Returns a summary dict.
    """
    from hermes_cli import supervisor as S
    from cron.jobs import create_job, update_job

    m = S.load_mission(mission_id)
    if m is None:
        return {"error": f"no mission {mission_id}"}

    c = load_controller(mission_id)
    level = WAKE_LEVELS.get(str(level).upper(), "MEDIUM")

    # Build script+prompt first; mutate durable controller state only once
    # job registration is known to succeed (arm is atomic-ish).
    script = _write_wake_script(mission_id, level)
    sups_dir = os.environ.get("HERMES_SUPERVISOR_DIR") or os.path.expanduser(
        "~/.hermes-supervisor")
    prompt = CONTROLLER_BRIEF_TMPL.format(
        mission_id=mission_id,
        objective=m.get("objective", ""),
        sups_dir=sups_dir,
        events="<injected by wake monitor>",
        policy=("Controller policy: " + policy) if policy else "")
    try:
        job_id = c.get("job_id") or ""
        if job_id:
            job = update_job(job_id, {"prompt": prompt, "schedule": interval,
                                      "deliver": deliver, "enabled": True,
                                      "monitor_script": str(script)})
            if job is None:
                job_id = ""
        if not job_id:
            job = create_job(
                prompt=prompt,
                schedule=interval,
                name=f"mission-controller:{mission_id}",
                deliver=deliver,
                monitor_script=str(script),
                model=model or None,
            )
            job_id = job.get("id") or ""
    except Exception as exc:  # noqa: BLE001
        return {"error": f"cron create failed: {exc}"}

    c.update({"armed": True, "level": level, "interval": interval,
              "policy": policy, "objective": m.get("objective", ""),
              "created_at": c.get("created_at") or time.time(),
              "last_arm_at": time.time(), "job_id": job_id})
    # baseline only on the FIRST arm: do not replay pre-controller events;
    # a re-arm preserves the existing watermark (never resets consumption).
    if not c.get("job_id") and not c.get("watermark"):
        c["watermark"] = _journal_len(mission_id)
    _save_controller(c)
    return {"mission_id": mission_id, "status": "armed", "job_id": job_id,
            "monitor_script": str(script), "level": level,
            "watermark": c["watermark"]}


def disarm(mission_id: str) -> Dict[str, Any]:
    c = load_controller(mission_id)
    if c.get("job_id"):
        try:
            from cron.jobs import update_job
            update_job(c["job_id"], {"enabled": False})
        except Exception as exc:  # noqa: BLE001
            return {"error": f"disarm cron failed: {exc}"}
    c["armed"] = False
    _save_controller(c)
    return {"mission_id": mission_id, "status": "disarmed"}


def status(mission_id: str) -> Dict[str, Any]:
    from hermes_cli import supervisor as S
    c = dict(load_controller(mission_id))
    c["exists"] = S.load_mission(mission_id) is not None
    c["new_events"] = len(new_events(mission_id, level=c.get("level", "MEDIUM")))
    return c


def wait(mission_id: str, *, timeout: float = 30.0,
         level: str = "MEDIUM") -> Dict[str, Any]:
    """Blocking wake for an interactive session: returns as soon as a
    meaningful event appears (or on timeout). Only tails the durable journal
    — never blocks on a supervisor process — so the agent stays responsive
    between events. Returns {"events": [...]} or {"timeout": True}."""
    deadline = time.time() + max(0.05, float(timeout))
    while True:
        evs = new_events(mission_id, level=level)
        if evs:
            return {"events": evs}
        if time.time() >= deadline:
            return {"timeout": True, "events": []}
        time.sleep(0.3)