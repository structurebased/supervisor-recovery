"""``hermes supervise`` subcommand — spawn and supervise an autonomous worker.

Wired into ``hermes_cli/main.py``. Uses the existing subprocess/session
primitives; no new framework.

    hermes supervise create [--task TEXT] [--budget-max-turns N] [--workdir PATH]
    hermes supervise start TASK_ID [--model MODEL]
    hermes supervise status TASK_ID
    hermes supervise check TASK_ID          # one supervisor evaluation
    hermes supervise loop TASK_ID [--every SEC] [--max-iterations N]
    hermes supervise cancel TASK_ID
    hermes supervise list
"""

from __future__ import annotations

import json
import os
import select
import socket
import time
from typing import Any, Dict, List, Optional

from hermes_cli import supervisor as SUP


def run_supervise_command(args) -> int:
    """Public entry used by ``hermes supervise`` (injected via cmd_supervise)."""
    action = getattr(args, "supervise_action", None) or "list"
    return _dispatch(args, action)


CUR_EVENT_SOCK = None  # set by _run_mission_loop; read by todo the supervisors


def _bind_event_socket(mission_id: str, every: float):
    """Per-mission Unix datagram socket; child workers notify it on state
    change. Sets HERMES_SUPERVISOR_EVENT_SOCK so worker-held env inherits it.
    Returns socket or None."""
    import os as _os
    import pathlib as _pl
    base = _pl.Path(_os.environ.get("HERMES_SUPERVISOR_DIR")
                    or _os.path.expanduser("~/.hermes-supervisor")) / "events"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{mission_id}.sock"
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.bind(str(path))
        sock.setblocking(False)
        _os.environ["HERMES_SUPERVISOR_EVENT_SOCK"] = str(path)
        return sock
    except Exception as exc:  # noqa: BLE001
        print(f"[mission-loop] event socket unavailable: {exc}")
        return None


def _block_until_event(sock, proc, every: float, max_secs: float = 0.0,
                       start_t: float = 0.0) -> Optional[Dict[str, Any]]:
    """Block until a worker event, a worker-process exit, or the safety
    watchdog timeout — whichever comes first. Returns the last event
    payload (None on timeout). This is the event-driven wake-up: a 3s
    worker finishing wakes the supervisor in ~3s, not every seconds."""
    # The block must never overshoot the mission window: cap the watchdog
    # deadline at min(every, remaining max_seconds).
    deadline_span = every if every and every > 0 else 20.0
    if max_secs and start_t:
        remaining = max(0.0, max_secs - (time.time() - start_t))
        deadline_span = min(deadline_span, max(0.25, remaining))
    fds = ([sock] if sock is not None else [])
    deadline = time.time() + deadline_span
    last_ev = None
    while True:
        # A worker process exit (crash/kill) must wake us too — the socket
        # only carries state-write events; a worker that dies without
        # writing state would otherwise wait until the watchdog deadline.
        if proc is not None:
            try:
                if proc.poll() is not None:
                    return last_ev
            except Exception:
                pass
        remaining = deadline - time.time()
        if remaining <= 0:
            return last_ev
        try:
            # empty fd list = pure sleep to the deadline (no stdin wakeup)
            r, _, _ = select.select(fds, [], [], min(remaining, 0.25))
        except (OSError, ValueError):
            return last_ev
        if r and sock is not None:
            try:
                data = sock.recv(65536)
                if data:
                    try:
                        last_ev = json.loads(data.decode("utf-8"))
                    except Exception:
                        last_ev = {"kind": "unknown"}
                    # P-81b: a received event IS the wakeup — return so the
                    # supervisor re-reads durable state immediately instead
                    # of draining until the watchdog deadline (previous code
                    # collected events but still waited the FULL interval,
                    # which silently re-serialized the pool).
                    return last_ev
            except BlockingIOError:
                pass


def _discovery_brief(mission_id: str, m: Dict[str, Any],
                     d: Dict[str, Any]) -> str:
    """Brief for a discovery worker: objective + rationale + evidence PLUS
    the r15 context block — live capability route, prior failure memory for
    this target, and the current plan snapshot — so the worker plans with
    what the environment actually provides and what already failed."""
    from hermes_cli import mission_ops as MO
    did = d["id"]
    lines = [
        "Mission-discovered task (%s/%s).\n"
        "Objective: %s\n"
        "Rationale: %s\n"
        "Evidence: %s\n"
        "Do the task; when genuinely done, write completion_evidence with"
        "real verification (tests/commands ran)."
        % (mission_id, did, m.get("objective", ""), d.get("rationale", ""),
           d.get("evidence", []))
    ]
    # capability route: what exists in THIS runtime (facts only)
    query = str(d.get("title", "")) + " " + str(d.get("rationale", ""))
    try:
        route = MO.capability_router(query)[:4]
        if route:
            lines.append("\n## Capability route (from live inventory)")
            for r in route:
                av = "available" if r.get("available") else "UNAVAILABLE here"
                lines.append(f"- {r['cap']} ({av}) — {', '.join(r['why'][:2])}")
            unavailable = [r["cap"] for r in route if not r.get("available")]
            if unavailable:
                lines.append("\nSome implied backends are NOT installed in this"
                             " runtime: " + ", ".join(unavailable))
                lines.append("Adapt: use an installed backend or probe the "
                             "real constraint instead of assuming.")
    except Exception:  # noqa: BLE001
        pass
    # prior failures for this exact target
    try:
        fmb = MO.failure_memory_brief(mission_id, did)
        if fmb:
            lines.append(fmb)
    except Exception:  # noqa: BLE001
        pass
    # cross-mission lessons relevant to this objective (platform store)
    try:
        lh = MO.retrieval_hint(str(m.get("objective", "")))
        if lh:
            lines.append("")
            lines.append("## Platform lessons (from prior missions)")
            lines.append("")
            lines.append(lh)
    except Exception:  # noqa: BLE001
        pass
    # current plan state
    try:
        ps = MO.plan_snapshot(mission_id)
        if ps:
            lines.append(ps)
    except Exception:  # noqa: BLE001
        pass
    # machine constraints for execution planning (#12)
    try:
        from hermes_cli import environment as ENV
        envd = ENV.probe_env()
        lines.append("\n## Machine constraints (live)")
        lines.append(
            f"- cpus_available={envd.get('cpus_available') or envd.get('cpus_logical')} "
            f"mem_total_kb={envd.get('mem_total_kb')} python={envd.get('python')}")
        try:
            lines.append(f"- suggested_concurrency="
                         f"{MO.suggested_concurrency()}")
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


def _run_discovery_worker(args, mission_id: str, d: Dict[str, Any],
                           start_t: float, every: float, max_secs: float,
                           model: Optional[str] = None) -> int:
    """P-18: execute one TODO discovery with a worker; then (evidence-gated)
    close it COMPLETE/BLOCKED. Discovery discovery… recursion is bounded by
    the loop's per-discovery supervision; the mission remains ACTIVE until
    the completion evaluator passes."""
    from hermes_cli import mission_ops as MO

    did = d["id"]
    m = SUP.load_mission(mission_id)
    MO.promote_discovery(m, did, "IN_PROGRESS", note="orchestrator started")
    SUP.save_mission(m)
    brief = _discovery_brief(mission_id, m, d)
    candidate = f"{mission_id}-disc-{did}"
    existing = SUP.load_worker(candidate) if candidate else None
    if existing:
        task_id = candidate
        print(f"[mission-discovery] resume existing worker {task_id} "
              f"(restart-safe)")
    else:
        task_id, _ = SUP.create_worker(d.get("title") or did,
                                    task_id=candidate)
    proc = None  # single-owner guard owns the Popen; wake decides per tick
    try:
        pid, spawned = SUP.start_worker_guarded(
            task_id, model=getattr(args, "model", None))
    except Exception as exc:  # noqa: BLE001
        print(f"[mission-discovery] {did}: start failed: {exc}")
        m = SUP.load_mission(mission_id)
        MO.promote_discovery(m, did, "BLOCKED", note=f"start failed: {exc}")
        SUP.save_mission(m)
        return 1
    if not spawned:
        print(f"[mission-discovery] {task_id} already live (pid {pid}); "
              f"not duplicating (single-owner guard)")
        return 0
    print(f"[mission-discovery] {mission_id}: running discovery {did} ({task_id})")
    prev = prev_fp = None
    stall = 0
    while True:
        if max_secs and (time.time() - start_t) > max_secs:
            print(f"[mission-discovery] max-seconds reached mid-discovery; "
                  f"task {task_id} survives; resumable (rc=7)")
            return 7
        wstate = SUP.load_worker(task_id)
        if not wstate:
            m = SUP.load_mission(mission_id)
            MO.promote_discovery(m, did, "BLOCKED", note="worker ledger missing")
            SUP.save_mission(m)
            break
        wstate["worker_pid"] = int(wstate.get("worker_pid") or 0)
        wp_pid = int(wstate.get("worker_pid") or 0)
        SUP.expire_stale_messages(task_id)
        SUP.deliver_all_if_idle(task_id, wstate)
        fp = SUP.state_fingerprint(wstate)
        action, stall = SUP.watchdog_assess(
            wstate, pid=wp_pid, fingerprint=fp, prev_fingerprint=prev_fp,
            stall_count=stall, now=time.time(),
            log_path=SUP.worker_path(task_id).parent / "worker.log")
        prev_fp = fp
        decision = SUP.evaluate_worker(wstate, previous=prev, now=time.time(),
                                       pid=wp_pid)
        if action == "stall":
            decision = SUP.WorkerDecision(
                "STALL", "RETRY", "Hung discovery worker; bounded retry.", 0.05)
        SUP.write_command(task_id, decision)
        prev = wstate
        print(f"[mission-discovery] {did} {wstate.get('status')} -> "
              f"{decision.verdict} [{decision.command}]")

        if decision.command == "RETRY" and decision.verdict in (
                "WORKER_CRASH", "WORKER_TIMEOUT", "WORKER_FAILURE", "STALL"):
            if decision.verdict == "STALL":
                SUP.kill_worker(wp_pid)
            wp = SUP.load_worker(task_id)
            if wp is None:
                break
            anchored = int(wp.get("worker_pid") or 0) and wp.get("started_at")
            if not anchored:
                time.sleep(every)
                continue
            if SUP.attempts_left(wp) > 0:
                SUP.bump_attempt(wp, reason=decision.verdict.lower())
                wp["seq"] = SUP.next_seq(wp)
                SUP.save_worker_cas(task_id, wp)
                SUP.start_worker_guarded(task_id, force=True)
                prev = prev_fp = None
                stall = 0
                continue
            m = SUP.load_mission(mission_id)
            MO.promote_discovery(m, did, "BLOCKED",
                                 note=f"attempts exhausted: {task_id}")
            SUP.save_mission(m)
            break

        if decision.verdict == "SUCCESS":
            evidence = wstate.get("completion_evidence") or []
            ev = " ".join(str(e) for e in evidence)
            if not ev:
                print(f"[mission-discovery] {did}: SUCCESS without evidence; "
                      "BLOCKED, mission continues")
                m = SUP.load_mission(mission_id)
                MO.promote_discovery(m, did, "BLOCKED", note="no evidence")
                SUP.save_mission(m)
            else:
                m = SUP.load_mission(mission_id)
                MO.promote_discovery(m, did, "COMPLETE", note="worker evidence")
                # discovery for DoD evidence-mapping
                for dim in (d.get("satisfies") or []):
                    MO.dod_satisfy(m, dim,
                                   evidence="discovery %s: %s" % (did, ev[:300]),
                                   who=did)
                # P-18: harvest the completed worker's next_action/blockers as
                # candidate DISCOVERIES (planner decides value; nothing added
                # as authority, only as candidate work).
                for src in ("next_action", "blockers"):
                    text = str(wstate.get(src) or "").strip()
                    import re as _re
                    _words = _re.findall(r"[a-z]{4,}", text.lower())
                    if (_words and len(_words) >= 2 and len(text) > 12
                            and text.lower() not in ("none", "done", "nothing")
                            and not text.strip().startswith("none")):
                        MO.add_discovery(
                            m, title=text[:120],
                            rationale=f"worker {task_id} ({src}) after {did}",
                            discoverer=task_id,
                            evidence=[text, "from completed worker %s" % task_id],
                            priority=3)
                SUP.save_mission(m)
            break
        if decision.verdict in ("CANCELLED", "CANCEL") or decision.command == "CANCEL":
            m = SUP.load_mission(mission_id)
            MO.promote_discovery(m, did, "BLOCKED", note="cancelled")
            SUP.save_mission(m)
            break
        _block_until_event(CUR_EVENT_SOCK, proc, every, max_secs=max_secs, start_t=start_t)
    return 0


def _run_discovery_pool(args, mission_id: str, start_t: float,
                        every: float, max_secs: float,
                        model: Optional[str] = None,
                        max_concurrent: int = 2) -> int:
    """P-22b: bounded-concurrency discovery supervisor.

    The mission supervisor must NOT be blocked on any single worker. It runs
    up to max_concurrent discovery workers at once (sharing the mission's
    event socket): any worker's state-write/exit wakes the loop, terminated
    slots are closed + new TODO discovery workers start in the freed slot,
    all in one reactor. Watchdog (every) is a safety fallback only.

    Agent/controller must never be forced to 'wait proc_x NNNs' — this loop
    returns only when no discovery is active AND no TODO remains, meaning the
    supervisor is once more waiting on the mission (not on a worker).
    """
    from hermes_cli import mission_ops as MO
    slots: Dict[str, Dict[str, Any]] = {}   # task_id -> slot state

    semantic_prev = {}
    while True:
        if max_secs and (time.time() - start_t) > max_secs:
            print("[mission-discovery-pool] max-seconds reached; "
                  "mission persists (resumable)")
            return 7
        # 0) ADOPT: existing IN_PROGRESS discoveries (e.g. after a supervisor
        # restart mid-flight, or discoveries whose workers we already began)
        # must be resumed as slots before spawning anything new.
        m = SUP.load_mission(mission_id)
        for d in m.get("discoveries", []):
            if d.get("status") != "IN_PROGRESS":
                continue
            cand = f"{mission_id}-disc-{d['id']}"
            if cand in slots:
                continue
            if not SUP.load_worker(cand):
                continue  # no ledger yet; will be re-spawned as TODO later
            slots[cand] = {"did": d["id"], "created": time.time(), "stall": 0}
            print(f"[mission-discovery-pool] adopted in-progress {d['id']} "
                  f"({cand}) restart-safe")
        # 1) FILL MORE: spawn up to max_concurrent new workers before any
        # blocking step, so task B gets created regardless of task A's
        # runtime (P-81b: a spawn must never wait for another worker).
        while len(slots) < max_concurrent:
            m = SUP.load_mission(mission_id)
            runnable = [d for d in m.get("discoveries", [])
                        if d.get("status") == "TODO"]
            if not runnable:
                break
            runnable.sort(key=lambda d: (int(d.get("priority", 5)),
                                         d.get("created_at") or 0))
            d = runnable[0]
            task_id = _spawn_discovery_slot(args, mission_id, d, model)
            if not task_id:
                break  # spawn failed (ledger blocked); avoid hot-spin
            slots[task_id] = {"did": d["id"], "created": time.time(), "stall": 0}
        # 2) supervise all active slots (single step each; non-blocking)
        #    + semantic-progress check: alive-but-no-meaningful-change signal
        prev_fp_sem = semantic_prev.get("fp")
        tick_sem = semantic_prev.get("ticks", 0)
        fp_cur = MO.mission_semantic_fingerprint(mission_id)
        if prev_fp_sem == fp_cur:
            tick_sem += 1
        else:
            tick_sem = 0
        semantic_prev.update({"fp": fp_cur, "ticks": tick_sem})
        # P-26 r12: TIME anomaly — is a live worker running far past its
        # comparable baseline? Time is diagnostic, not a kill switch: emit a
        # structured PERFORMANCE_REASSESSMENT once per slot (autonomous
        # "why is this taking so long?"), then let the SAME bounded
        # strategy-switch machinery below consider an alternative. Never a
        # crude timeout kill.
        perf_expected = MO.expected_worker_span(mission_id)
        perf_mult = float(os.environ.get("MISSION_PERF_MULT", "3") or 3)
        perf_min = float(os.environ.get("MISSION_PERF_MIN_SECONDS", "60") or 60)
        perf_anomaly_slot = None
        perf_cpus = int(os.cpu_count() or 1)
        try:
            perf_cap = MO.suggested_concurrency()
        except Exception:
            perf_cap = max(1, min(perf_cpus, 4))
        for task_id, slot in slots.items():
            if slot.get("perf_flagged"):
                continue
            created = float((slot.get("created") or 0))
            if not created or perf_expected is None:
                continue
            elapsed = time.time() - created
            if not MO.perf_anomaly(elapsed=elapsed, expected=perf_expected,
                                   mult=perf_mult, min_seconds=perf_min):
                continue
            slot["perf_flagged"] = True
            perf_anomaly_slot = task_id
            tel_p = MO.telemetry_report(mission_id)
            w_p = next((x for x in tel_p.get("workers", [])
                        if x.get("task_id") == task_id), {})
            MO.log_event(
                mission_id, "HIGH", "PERFORMANCE_REASSESSMENT",
                f"worker {task_id} took {elapsed:.0f}s vs "
                f"expected ~{perf_expected:.0f}s — why is this taking "
                f"so long?",
                {"task_id": task_id,
                 "elapsed_seconds": round(elapsed, 1),
                 "expected_seconds": round(perf_expected or 0, 1),
                 "progress": w_p.get("status", ""),
                 "repeated_commands": w_p.get("repeated_commands", {}),
                 # P-26 r13: the machine model joins the diagnosis — was this
                 # slow because the box was oversubscribed? cpus and the
                 # suggested concurrency cap let the controller answer '20
                 # local workers on 4 cores' instead of guessing.
                 "env": {"cpus": perf_cpus,
                         "concurrent_workers": len(slots),
                         "suggested_concurrency_cap": perf_cap},
                 "why": "inspect telemetry, prior attempts, capabilities, "
                        "and the environment model; consider alternative "
                        "strategy",
                 "actions": ["inspect telemetry", "inspect attempts",
                             "inspect capabilities", "inspect env"]})
            break
        if tick_sem >= 4:
            MO.log_event(mission_id, "MEDIUM", "SEMANTIC_STAGNATION",
                         "mission ledger unchanged for 4 ticks while workers active")
        # Strategy switching (P-26): don't just warn — record an
        # evidence-backed ALTERNATIVE as a discovery the pool will pick
        # up, bounded to a few switches per mission (durable counter +
        # slug dedup prevent infinite strategy cycling). Guard: only
        # when a slot has a LIVE worker (pid alive) that has been
        # running long enough to be genuinely stuck (created at least
        # MISSION_STRATEGY_MIN_STALE seconds ago; default 60). Crash-
        # churn / dry-run workers (pid 0 / dead) and freshly attached
        # / completing workers do NOT count as stagnation.
        min_stale = float(os.environ.get("MISSION_STRATEGY_MIN_STALE", "60") or 60)
        live_slots = []
        for task_id, slot in slots.items():
            wst = SUP.load_worker(task_id)
            wp = int((wst or {}).get("worker_pid") or 0)
            wcreated = float((wst or {}).get("created_at") or 0)
            if wp and SUP._pid_alive(wp) \
                    and wcreated and (time.time() - wcreated) >= min_stale:
                live_slots.append(task_id)
        if (tick_sem >= 4 or perf_anomaly_slot) and live_slots:
            m_alt = SUP.load_mission(mission_id)
            switches = int(m_alt.get("strategy_switches") or 0)
            if switches < 2:
                evidence = [fp_cur, "SEMANTIC_STAGNATION after 4 unchanged ticks"]
                if perf_anomaly_slot:
                    evidence.append(
                        f"PERFORMANCE_REASSESSMENT: {perf_anomaly_slot} "
                        f"elapsed vs expected {perf_expected:.0f}s")
                MO.add_discovery(
                    m_alt,
                    title=f"[strategy] alternative approach for stalled "
                          f"mission work (switch #{switches + 1})",
                    rationale="stagnation/perf anomaly detected by supervisor "
                              f"(live workers: {','.join(live_slots)[:80]})",
                    discoverer="supervisor",
                    evidence=evidence,
                    priority=3)
                m_alt["strategy_switches"] = switches + 1
                _prom, _n = MO.plan_discoveries(m_alt)   # evidence promotes
                SUP.save_mission(m_alt)
            # reset so we log once per stagnation episode, not every tick
            semantic_prev["ticks"] = 0
        done_ids = []
        any_wait = False
        for task_id, slot in list(slots.items()):
            rc = _step_worker_slot(args, mission_id, task_id, slot,
                                   every, max_secs, start_t)
            if rc == "DONE":
                done_ids.append(task_id)
                MO.log_event(mission_id, "HIGH", "DISCOVERY_CLOSED",
                             f"discovery {slot.get('did')} done",
                             {"did": slot.get("did")})
            elif rc == "WAIT":
                any_wait = True
        for task_id in done_ids:
            slots.pop(task_id, None)
        if slots and any_wait:
            # One event block per pool iteration: any worker event (all
            # workers share the mission socket) wakes the pool; the watchdog
            # bound is the safety fallback only. Worker processes exit is
            # also observed via proc-poll during the block's wakeups.
            _block_until_event(CUR_EVENT_SOCK, None, every,
                               max_secs=max_secs, start_t=start_t)
        # 2) nothing active/runnable -> exhausted
        if not slots:
            m = SUP.load_mission(mission_id)
            runnable = [d for d in m.get("discoveries", [])
                        if d.get("status") == "TODO"]
            if not runnable:
                return 0
            # fallback watchdog boundary; avoids hot loop when spawn fails
            _block_until_event(CUR_EVENT_SOCK, None, every,
                               max_secs=max_secs, start_t=start_t)
    return 0


def _spawn_discovery_slot(args, mission_id: str, d: Dict[str, Any],
                           model=None) -> str:
    """Create+anchor a worker for a discovery, return task_id (or '')."""
    from hermes_cli import mission_ops as MO
    did = d["id"]
    m = SUP.load_mission(mission_id)
    MO.promote_discovery(m, did, "IN_PROGRESS", note="orchestrator started")
    SUP.save_mission(m)
    brief = _discovery_brief(mission_id, m, d)
    candidate = f"{mission_id}-disc-{did}"
    if SUP.load_worker(candidate):
        task_id = candidate
        print(f"[mission-discovery] resume existing worker {task_id} "
              f"(restart-safe)")
    else:
        task_id, _ = SUP.create_worker(brief, task_id=candidate,
                                       workdir=(m.get("workdir") or None))
    print(f"[mission-discovery] {mission_id}: running discovery {did} ({task_id})")
    try:
        if os.environ.get("MISSION_DRY_WORKER") == "1":
            # TEST SEAM: create the worker ledger but do NOT spawn the real
            # model worker (its long LLM turn would race the scripted driver
            # writing COMPLETE on the same ledger, making tests deterministic).
            # The scripted driver anchors + writes exactly like a real worker.
            return task_id
        pid, spawned = SUP.start_worker_guarded(task_id)
        if not spawned:
            print(f"[mission-discovery] {task_id} already live (pid {pid}); "
                  f"another loop owns it, not duplicating")
            return ""
    except Exception as exc:  # noqa: BLE001
        print(f"[mission-discovery] {did}: start failed: {exc}")
        m = SUP.load_mission(mission_id)
        MO.promote_discovery(m, did, "BLOCKED", note=f"start failed: {exc}")
        SUP.save_mission(m)
        return ""
    MO.log_event(mission_id, "MEDIUM", "DISCOVERY_OPENED",
                 f"discovery {did} started: {str(d.get('title', ''))[:80]}",
                 {"did": did})
    return task_id


def _step_worker_slot(args, mission_id: str, task_id: str, slot,
                      every: float, max_secs: float, start_t: float):
    """One evaluation + command step for an active worker slot.
    Returns 'DONE' when the slot is resolved (terminal state)."""
    from hermes_cli import mission_ops as MO
    wstate = SUP.load_worker(task_id)
    if not wstate:
        m = SUP.load_mission(mission_id)
        MO.promote_discovery(m, slot["did"], "BLOCKED", note="worker ledger missing")
        SUP.save_mission(m)
        return "DONE"
    wstate["worker_pid"] = int(wstate.get("worker_pid") or 0)
    wp_pid = int(wstate.get("worker_pid") or 0)
    SUP.expire_stale_messages(task_id)
    SUP.deliver_all_if_idle(task_id, wstate)
    fp = SUP.state_fingerprint(wstate)
    action, stall = SUP.watchdog_assess(
        wstate, pid=wp_pid, fingerprint=fp,
        prev_fingerprint=slot.get("prev_fp"), stall_count=slot.get("stall", 0),
        now=time.time(),
        log_path=SUP.worker_path(task_id).parent / "worker.log")
    slot["prev_fp"] = fp
    decision = SUP.evaluate_worker(wstate, previous=slot.get("prev"), now=time.time(),
                                   pid=wp_pid)
    if action == "stall":
        decision = SUP.WorkerDecision("STALL", "RETRY",
                                      "Hung discovery worker; bounded retry.", 0.05)
    SUP.write_command(task_id, decision)
    slot["prev"] = wstate
    slot["stall"] = stall
    print(f"[mission-discovery] {slot['did']} {wstate.get('status')} -> "
          f"{decision.verdict} [{decision.command}]")

    if decision.command == "RETRY" and decision.verdict in (
            "WORKER_CRASH", "WORKER_TIMEOUT", "WORKER_FAILURE", "STALL"):
        if decision.verdict == "STALL":
            SUP.kill_worker(wp_pid)
        wp = SUP.load_worker(task_id)
        if wp is None or not (int(wp.get("worker_pid") or 0) and wp.get("started_at")):
            # P-81b: DO NOT block the pool here on this slot's unanchored
            # defer — that would serialize B behind A's window. Return WAIT
            # and let the pool block exactly once after visiting all slots.
            return "WAIT"
        if SUP.attempts_left(wp) > 0:
            SUP.bump_attempt(wp, reason=decision.verdict.lower())
            wp["seq"] = SUP.next_seq(wp)
            SUP.save_worker_cas(task_id, wp)
            try:
                SUP.start_worker_guarded(task_id, force=True)
            except Exception:
                pass
            slot.pop("prev", None); slot.pop("prev_fp", None); slot["stall"] = 0
            return None
        m = SUP.load_mission(mission_id)
        MO.promote_discovery(m, slot["did"], "BLOCKED",
                             note=f"attempts exhausted: {task_id}")
        # failure memory: durable lesson that THIS approach failed, so a
        # future worker for the same target won't repeat it blindly
        try:
            MO.record_failure_memory(
                mission_id, target=slot["did"],
                approach=f"worker {task_id} approach exhausted",
                outcome="BLOCKED after retries/attempts exhausted",
                how_to_avoid="alternate strategy or split the target")
        except Exception:
            pass
        SUP.save_mission(m)
        return "DONE"

    if decision.verdict == "SUCCESS":
        evidence = wstate.get("completion_evidence") or []
        ev = " ".join(str(e) for e in evidence)
        m = SUP.load_mission(mission_id)
        if not ev:
            MO.promote_discovery(m, slot["did"], "BLOCKED", note="no evidence")
            SUP.save_mission(m)
        else:
            MO.promote_discovery(m, slot["did"], "COMPLETE", note="worker evidence")
            # P-26 r12: measure whether a strategy switch actually helped
            # (before->change->after delta recorded durably, lesson stored)
            try:
                w_created = float(wstate.get("created_at") or 0)
                if w_created:
                    MO.measure_strategy_outcome(
                        m, task_id=task_id,
                        elapsed=time.time() - w_created)
            except Exception:
                pass
            for dim in (next((d for d in m.get("discoveries", [])
                              if d["id"] == slot["did"]), {}).get("satisfies") or []):
                MO.dod_satisfy(m, dim,
                               evidence="discovery %s: %s" % (slot["did"], ev[:300]),
                               who=slot["did"])
            # harvest worker next_action/blockers
            for src in ("next_action", "blockers"):
                txt = str(wstate.get(src) or "").strip()
                import re as _re
                _words = _re.findall(r"[a-z]{4,}", txt.lower())
                if (_words and len(_words) >= 2 and len(txt) > 12
                        and txt.lower() not in ("none", "done", "nothing")
                        and not txt.strip().startswith("none")):
                    MO.add_discovery(m, title=txt[:120],
                                     rationale=f"worker {task_id} ({src}) after {slot['did']}",
                                     discoverer=task_id,
                                     evidence=[txt, "from completed worker %s" % task_id],
                                     priority=3)
            SUP.save_mission(m)
        return "DONE"
    if decision.verdict in ("CANCELLED", "CANCEL") or decision.command == "CANCEL":
        m = SUP.load_mission(mission_id)
        MO.promote_discovery(m, slot["did"], "BLOCKED", note="cancelled")
        SUP.save_mission(m)
        return "DONE"
    if decision.verdict == "UNVERIFIED_COMPLETION" or decision.command == "VERIFY":
        if wp_pid and not SUP._pid_alive(wp_pid):
            m = SUP.load_mission(mission_id)
            MO.promote_discovery(m, slot["did"], "BLOCKED",
                                 note="UNVERIFIED dead worker")
            SUP.save_mission(m)
            return "DONE"
    return "WAIT"  # pool blocks once after visiting all slots


def _block_until_slot(sock, proc, every, max_secs=0.0, start_t=0.0):
    _block_until_event(sock, proc, every, max_secs=max_secs, start_t=start_t)


def _run_mission_loop(args, mission_id: str) -> int:
    """Autonomous mission orchestrator (P-14) — closes the process observable
    by logging MISSION_LOOP_EXITED on every exit path so a controller can
    notice a stopped supervisor, and releases the supervisor lease so a
    replacement loop can take over (single-owner guard, P-26 r3)."""
    try:
        return _run_mission_loop_inner(args, mission_id)
    finally:
        try:
            from hermes_cli import mission_ops as _MO
            _MO.log_event(mission_id, "MEDIUM", "MISSION_LOOP_EXITED",
                          "mission supervisor loop process exited")
        except Exception:
            pass
        SUP.lease_release(mission_id, _HELD_LEASE.get(mission_id, ""))


_HELD_LEASE: Dict[str, str] = {}


def _run_mission_loop_inner(args, mission_id: str) -> int:
    """Autonomous mission orchestrator (P-14).

    Termination semantics:
      - iterate next_phase() until no PENDING phase remains
      - for each phase: create + start a worker task, then supervise it with
        the SAME evaluate/watchdog loop used by ``supervise loop``
      - on worker SUCCESS: mark the phase COMPLETE only if the worker's
        completion evidence satisfies the phase evidence gate; otherwise mark
        FAILED and CONTINUE to the next phase (a worker claim is evidence,
        not authority)
      - worker crash/timeout: bounded retry; exhausted -> mark phase FAILED
        and CONTINUE (mission has other work)
      - after all phases closed: evaluate requirements / open findings; if any
        remain, the mission stays ACTIVE (report it) — the orchestrator does
        NOT invent completion
      - a real campaign can run inside a phase via extra task text; that's a
        phase task, not a mission terminal condition
    """
    every = float(getattr(args, "every", 10.0) or 10.0)
    max_secs = float(getattr(args, "max_seconds", 0.0) or 0.0)
    max_phases = int(getattr(args, "max_phases", 0) or 0)
    model = getattr(args, "model", None)
    start_t = time.time()
    phases_done = 0

    # Single-owner lease (P-26 r3). A second loop for the same mission is
    # refused while a live lease is held; a dead/stale holder is taken over
    # automatically. So there is NEVER more than one supervisor authority.
    holder = f"loop-{os.getpid()}@{socket.gethostname()}"
    held, why = SUP.lease_acquire(mission_id, holder)
    if not held:
        print(f"[mission-loop] {mission_id}: supervisor lease unavailable: "
              f"{why}. Another loop is the active authority; not starting.")
        return 0
    _HELD_LEASE[mission_id] = holder
    print(f"[mission-loop] {mission_id}: lease acquired by {holder}")

    global CUR_EVENT_SOCK
    CUR_EVENT_SOCK = _bind_event_socket(mission_id, every)
    if CUR_EVENT_SOCK is None:
        print("[mission-loop] event-driven wakeup disabled; will poll at "
              "interval (slow path)")
    from hermes_cli import mission_ops as MO
    m0 = SUP.load_mission(mission_id) or {}
    MO.log_event(mission_id, "HIGH", "MISSION_STARTED",
                 f"mission {mission_id} loop started")
    hint = MO.retrieval_hint(str(m0.get("objective", "")))
    if hint:
        print(f"[mission-loop] platform lessons relevant to objective:\n{hint}")
        # Track that lessons were retrieved for this mission
        try:
            from agent.experience import retrieve_lessons
            lessons = retrieve_lessons(str(m0.get("objective", "")), max_results=3)
            for l in lessons:
                from agent.experience import log_lesson_application
                log_lesson_application(
                    lesson_id=l.id,
                    task_id=mission_id,
                    task_goal=str(m0.get("objective", "")),
                    retrieved=True,
                )
        except Exception:
            pass

    while True:
        if not SUP.lease_renew(mission_id, holder):
            print(f"[mission-loop] {mission_id}: lost supervisor lease; "
                  f"stopping (another authority took over)")
            return 1
        from hermes_cli import mission_ops as MO
        # P-26 r15: ARCHITECTED DETECTORS run at the TOP of every pass,
        # BEFORE the completion gates. A fresh mission with no phases reads
        # MISSION_COMPLETE, so the old placement (only at exhaustion) was
        # unreachable — the loop must notice observable waste/coverage
        # signals and materialize them into discoveries before it can
        # declare the mission done. Idempotent: slug dedup.
        det_added, det_notes = MO.auto_apply_detections(mission_id, max_new=3)
        if det_added:
            print(f"[mission-loop] detectors: {det_added} new "
                  f"discovery(ies) [{det_notes[:120]}]")
            MO.log_event(mission_id, "MEDIUM", "DETECTOR_TRIGGERED",
                         f"auto-detectors created {det_added} "
                         f"discover(ies): {det_notes[:120]}",
                         {"count": det_added})
            continue
        st2 = SUP.mission_status(mission_id)
        st = st2
        # P-18: the legacy phase-only status is NOT the terminal gate. The
        # mission loop must continue while discoveries/DOD remain; only the
        # completion evaluator (phases + discoveries + DoD) may end it.
        m = SUP.load_mission(mission_id)
        ev = MO.completion_evaluator(m) if m else {"complete": True}
        if st.get("status") == "MISSION_COMPLETE" and m and not m.get("discoveries") \
                and not MO.backlog_open(m):
            # Q8 spot-audit gate (2026-08-23): no terminalization without an
            # independent adversarial pass. REJECT reopens the weakest
            # completed phase instead of spinning on the same state; the
            # repair worker must then resolve the audit finding.
            audit = SUP.spot_audit_mission(mission_id)
            print(f"[mission-loop] {mission_id}: spot-audit -> "
                  f"{audit['verdict']} ({audit['reason'][:140]})")
            if audit["verdict"] == "REJECT":
                done = [p_ for p_ in m.get("phases", [])
                        if p_.get("status") == "COMPLETE"]
                if done:
                    weak = min(done, key=lambda p_: len(p_.get("evidence") or []))
                    weak["status"] = "FAILED"
                    weak["retry_count"] = int(weak.get("retry_count") or 0) + 1
                    if int(weak.get("retry_count") or 0) <= 2:
                        weak["status"] = "PENDING"
                    weak["note"] = f"spot-audit rejected: {audit['reason'][:160]}"
                    SUP.save_mission(m)
                MO.record_stop(
                    mission_id, verdict="blocked",
                    reason=f"spot-audit REJECTED: {audit['reason'][:140]}",
                    evidence=["adversarial audit found defects; repair "
                              f"phase {weak['phase_id']} reopened"])
                print(json.dumps(audit, indent=2, default=str))
                return 1
            for f_ in m.get("unresolved_findings", []):
                if str(f_.get("id", "")).startswith("spot-audit-"):
                    f_["status"] = "RESOLVED"
                    f_["note"] = (f_.get("note") or "") + \
                        " [resolved by later CONFIRM]"
            SUP.save_mission(m)
            MO.record_stop(
                mission_id, verdict="complete",
                reason=st.get("reason", "mission complete"),
                evidence=["phases done", "no discoveries", "no open backlog",
                          f"spot-audit: {audit['verdict']}"])
            print(f"[mission-loop] {mission_id}: {st.get('status')} — "
                  f"{st.get('reason')}")
            print(json.dumps(st, indent=2, default=str))
            return 0
        if ev.get("complete"):
            # P-26 r15: the evaluator may be complete while ADVISORY
            # detector work remains RUNNABLE (advisory never blocks the
            # gate). The loop still executes runnable TODO discoveries —
            # advisory only relaxes the gate, it does not discard the work.
            runnable = [d for d in (m or {}).get("discoveries", [])
                        if d.get("status") == "TODO"]
            if runnable:
                print(f"[mission-loop] evaluator complete but {len(runnable)} "
                      f"runnable discovery(ies) remain; executing "
                      f"({', '.join(d['id'] for d in runnable[:3])}...)")
                phase = None
                # fall through to the discovery-pool branch below
            else:
                audit = SUP.spot_audit_mission(mission_id)
                print(f"[mission-loop] {mission_id}: spot-audit -> "
                      f"{audit['verdict']} ({audit['reason'][:140]})")
                if audit["verdict"] == "REJECT":
                    MO.record_stop(
                        mission_id, verdict="blocked",
                        reason=f"spot-audit REJECTED: {audit['reason'][:140]}",
                        evidence=["completion evaluator blocked by "
                                  "adversarial audit"])
                    print(json.dumps(audit, indent=2, default=str))
                    return 1
                for f_ in m.get("unresolved_findings", []):
                    if str(f_.get("id", "")).startswith("spot-audit-"):
                        f_["status"] = "RESOLVED"
                        f_["note"] = (f_.get("note") or "") + \
                            " [resolved by later CONFIRM]"
                SUP.save_mission(m)
                MO.record_stop(
                    mission_id, verdict="complete",
                    reason=ev.get("reason", "objective satisfied"),
                    evidence=["completion evaluator: phases+discoveries+dod",
                              f"spot-audit: {audit['verdict']}"])
                print(f"[mission-loop] {mission_id}: completion evaluator: "
                      f"COMPLETE — {ev.get('reason')}")
                MO.log_event(mission_id, "HIGH", "MISSION_COMPLETE",
                             ev.get("reason", "objective satisfied"))
                from hermes_cli import mission_ops as MO2
                try:
                    n, _why = MO2.mission_lesson_sync(SUP.load_mission(mission_id))
                    if n:
                        print(f"[mission-loop] synced {n} lessons to "
                              f"experience store")
                except Exception as exc:  # noqa: BLE001
                    print(f"[mission-loop] lesson sync failed: {exc}")
                return 0
        elif st.get("status") not in ("MISSION_ACTIVE", "MISSION_COMPLETE"):
            # active-with-open-discoveries is the normal case; a genuinely
            # Blocked status still yields to discovery if any exist, else stop
            if not [d for d in m.get("discoveries", [])
                    if d.get("status") == "TODO"]:
                MO.record_stop(
                    mission_id, verdict="blocked",
                    reason=st.get("reason", "mission status blocks"),
                    evidence=[f"mission status = {st.get('status')}",
                              "no TODO discoveries"])
                print(f"[mission-loop] {mission_id}: {st.get('status')} — "
                      f"{st.get('reason')}")
                return 0
        # RESUME semantics (attack-driven): if a phase is already ACTIVE
        # (previous orchestrator was killed/restarted mid-phase), the
        # orchestrator must supervise THAT phase to completion first. It
        # must NOT start another phase while one is active (duplicate-phase
        # attack: restart previously spawned p2 while p1 still active).
        # P-19: FAILED phases marked retryable (infra death) may also be
        # re-attempted, bounded.
        phase = SUP.active_phase(mission_id) or SUP.next_phase(mission_id)
        if phase is None:
            phase = SUP.retryable_failed_phase(mission_id)
            if phase is not None:
                did_retry, why = SUP.begin_phase_retry(mission_id,
                                                       phase["phase_id"])
                print(f"[mission-loop] retrying infra-failed phase "
                      f"{phase['phase_id']}: {why}")
                # note: status is ACTIVE now; we continue the loop and the
                # active_phase branch picks it up next iteration.
                if not did_retry:
                    phase = None
        if phase is None:
            # No runnable phase. P-18: discovery TODO items are the
            # continuation of the mission beyond declared phases; P-26 r6:
            # when even discoveries are exhausted, the durable capability
            # backlog is the continuation source — materialize the next open
            # item and keep going. This is what makes "tests pass / tree
            # clean" insufficient to end an unfinished mission: open backlog
            # items block the completion evaluator, and this branch turns
            # them into runnable work without a human "continue".
            from hermes_cli import mission_ops as MO
            m = SUP.load_mission(mission_id)
            active_disc = [d for d in m.get("discoveries", [])
                          if d.get("status") == "IN_PROGRESS"]
            todo = [d for d in m.get("discoveries", [])
                    if d.get("status") == "TODO"]
            if active_disc:
                pick = active_disc[0]
            elif todo:
                todo.sort(key=lambda d: (int(d.get("priority", 5)),
                                         d.get("created_at") or 0))
                pick = todo[0]
            else:
                pick = None
            if pick is not None:
                # P-81b: discovery pool — up to max_concurrent workers run
                # in parallel; the pool supervises all of them through the
                # shared event socket and returns only when no discovery is
                # active or runnable (or window hit: rc=7 resumable).
                # Environment-aware default cap (P-26): user override wins.
                mc_env = os.environ.get("MISSION_MAX_CONCURRENT", "")
                if mc_env:
                    mc = max(1, int(mc_env))
                else:
                    from hermes_cli import mission_ops as _MO
                    mc = _MO.suggested_concurrency()
                rc = _run_discovery_pool(args, mission_id, start_t,
                                         every, max_secs, model,
                                         max_concurrent=max(1, mc))
                if rc == 7:
                    print(f"[mission-loop] max-seconds reached mid-discovery; "
                          f"mission persists")
                    return 0
                continue
            # backlog drain: no phases, no discoveries, but open backlog
            bl_before = len(MO.backlog_open(m))
            created, notes = MO.backlog_materialize(m)
            if created:
                changed, _n = MO.plan_discoveries(m)
                SUP.save_mission(m)
                mo_log = ("[mission-loop] backlog {}/{} -> materialized {} "
                          "[{}] planner {}".format(
                              bl_before, len(MO.backlog_open(m)), created,
                              notes[:120], changed))
                print(mo_log)
                MO.log_event(mission_id, "MEDIUM", "BACKLOG_MATERIALIZED",
                             f"materialized {created} backlog item(s): "
                             f"{notes[:120]}", {"count": created})
                # loop again: the new TODO discovery is picked up above
                continue
            # P-26 r15: detection already ran at the top of this pass
            # (before the completion gates), so reaching this branch with no
            # phases/discoveries/backlog means detectors found nothing new.
            # genuine exhaustion: phases, discoveries AND backlog all empty
            rationale = MO.stop_rationale(m)
            MO.record_stop(
                mission_id,
                verdict=rationale.get("verdict", "stop-unless-new-information"),
                reason="no PENDING phase / discovery / open backlog",
                evidence=[f"stop_rationale={rationale.get('verdict')}",
                          f"open={rationale.get('open')}",
                          f"backlog_open={rationale.get('backlog', {}).get('open', 0)}"])
            print(f"[mission-loop] {mission_id}: no PENDING phase; "
                  f"mission remaining requirements unfulfilled")
            return 2
        if max_phases and phases_done >= max_phases:
            print(f"[mission-loop] max-phases reached ({max_phases})")
            return 0
        if max_secs and (time.time() - start_t) > max_secs:
            print(f"[mission-loop] max-seconds reached; mission persists "
                  f"(resume `hermes supervise mission loop {mission_id}`)")
            return 0

        phase_id = phase["phase_id"]
        task_text = phase["task"] or phase_id
        print(f"[mission-loop] {mission_id}: running phase {phase_id}")

        # Resume an existing ACTIVE phase worker instead of creating a
        # duplicate: the phase ledger keeps the worker_task binding.
        existing_task = (phase.get("worker_task") or "")
        if existing_task and SUP.load_worker(existing_task):
            task_id = existing_task
            print(f"[mission-loop] resume existing worker {task_id} "
                  f"(restart-safe)")
        else:
            # P-26 r15: phase workers get the live plan + capability route +
            # prior failure memory for this phase in their brief, so they
            # don't plan from stale instructions.
            try:
                from hermes_cli import mission_ops as _MO
                m2 = SUP.load_mission(mission_id) or {}
                extra_bits = [_MO.plan_snapshot(mission_id)]
                route = _MO.capability_router(
                    (phase.get("task") or task_text) + " " +
                    (m2.get("objective") or ""))[:4]
                if route:
                    extra_bits.append("## Capability route (live inventory)")
                    extra_bits.append("\n".join(
                        f"- {r['cap']} "
                        f"({'available' if r.get('available') else 'UNAVAILABLE here'})"
                        for r in route))
                fmb = _MO.failure_memory_brief(mission_id, phase_id)
                if fmb:
                    extra_bits.append(fmb)
                # cross-mission lessons relevant to this objective
                try:
                    lh = _MO.retrieval_hint(str(m2.get("objective", "")))
                    if lh:
                        extra_bits.append("")
                        extra_bits.append("## Platform lessons (from prior missions)")
                        extra_bits.append("")
                        extra_bits.append(lh)
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                extra_bits = []
            task_id, _state = SUP.create_worker(
                task_text, task_id=f"{mission_id}-{phase_id}",
                workdir=m.get("workdir") or None,
                extra_brief="\n".join(extra_bits))
            m = SUP.load_mission(mission_id)
            ph = SUP._find_phase(m, phase_id)
            if ph is not None:
                ph["worker_task"] = task_id
                ph["status"] = "ACTIVE"
                SUP.save_mission(m)

        # Start the worker with the existing start_worker border (CAS pid),
        # UNLESS the worker is already terminal: an existing COMPLETE/
        # CANCELLED/FAILED state is a prior result to adjudicate, not a
        # reason to spawn a fresh process (fix: evidence-rejection path used
        # to crash the loop with TypeError; this also stops pointless
        # restarts of finished workers on supervisor resume).
        wpre = SUP.load_worker(task_id)
        wlive = bool(wpre and wpre.get("worker_pid")
                     and SUP._pid_alive(int(wpre.get("worker_pid"))))
        proc = None  # bound for the shared event-wait below; resume paths
                     # do not spawn, so the block falls back to socket+timeout
        if wlive:
            print(f"[mission-loop] {phase_id}: worker {task_id} is LIVE "
                  f"(pid {wpre.get('worker_pid')}); resuming supervision, "
                  f"not duplicating")
        elif wpre and wpre.get("status") in ("COMPLETE", "CANCELLED", "FAILED"):
            print(f"[mission-loop] {phase_id}: worker {task_id} already "
                  f"terminal ({wpre.get('status')}); adjudicating, not restarting")
        else:
            try:
                pid, spawned = SUP.start_worker_guarded(
                    task_id, model=model)
                if not spawned:
                    print(f"[mission-loop] {phase_id}: worker already live "
                          f"(pid {pid}); not duplicating")
            except Exception as exc:  # noqa: BLE001
                print(f"[mission-loop] {phase_id}: start failed: {exc}")
                SUP.phase_failed(mission_id, phase_id,
                                 worker_by="mission-loop", note=f"start: {exc}")
                phases_done += 1
                continue
            proc = None  # no spawn path left a live Popen to wake on here

        # Supervise THIS task to a terminal worker state.
        prev = None
        prev_fp = None
        stall = 0
        while True:
            if max_secs and (time.time() - start_t) > max_secs:
                print(f"[mission-loop] max-seconds reached mid-phase; "
                      f"worker task {task_id} survives; mission persists")
                return 0
            wstate = SUP.load_worker(task_id)
            if not wstate:
                print(f"[mission-loop] worker {task_id} disappeared; marking "
                      f"phase failed")
                SUP.phase_failed(mission_id, phase_id, worker_by="mission-loop",
                                 note="worker ledger missing", retryable=True)
                break
            wstate["worker_pid"] = int(wstate.get("worker_pid") or 0)
            wp_pid = int(wstate.get("worker_pid") or 0)
            SUP.expire_stale_messages(task_id)
            SUP.deliver_all_if_idle(task_id, wstate)
            fp = SUP.state_fingerprint(wstate)
            action, stall = SUP.watchdog_assess(
                wstate, pid=wp_pid, fingerprint=fp, prev_fingerprint=prev_fp,
                stall_count=stall, now=time.time(),
                log_path=SUP.worker_path(task_id).parent / "worker.log")
            prev_fp = fp
            decision = SUP.evaluate_worker(wstate, previous=prev, now=time.time(),
                                           pid=wp_pid)
            if action == "stall":
                decision = SUP.WorkerDecision(
                    "STALL", "RETRY", "Hung worker; kill and respawn within budget.", 0.05)
            SUP.write_command_if_changed(task_id, decision)  # P-17: skip identical
            prev = wstate
            print(f"[mission-loop] {phase_id} {wstate.get('status')} -> "
                  f"{decision.verdict} [{decision.command}]")

            if decision.command == "RETRY" and decision.verdict in (
                    "WORKER_CRASH", "WORKER_TIMEOUT", "WORKER_FAILURE", "STALL"):
                if decision.verdict == "STALL":
                    SUP.kill_worker(wp_pid)
                wp = SUP.load_worker(task_id)
                if wp is None:
                    break
                anchored = int(wp.get("worker_pid") or 0) and wp.get("started_at")
                if not anchored:
                    time.sleep(every)
                    continue
                if SUP.attempts_left(wp) > 0:
                    SUP.bump_attempt(wp, reason=decision.verdict.lower())
                    wp["seq"] = SUP.next_seq(wp)
                    ok_cas, _why = SUP.save_worker_cas(task_id, wp)
                    if not ok_cas:
                        pass  # re-read next iteration
                    # s6-ownership: the dead/failed worker's children must not
                    # outlive it into the respawn (leak from s4).
                    SUP.reap_worker_processes(wp_pid)
                    SUP.start_worker_guarded(task_id, force=True)
                    prev = prev_fp = None
                    stall = 0
                    continue
                else:
                    hv = SUP.harvest_worker_worktree(task_id)
                    if hv.get("status"):
                        print(f"[mission-loop] harvested uncommitted worktree "
                              f"evidence ({hv['status_lines']} files) for "
                              f"exhausted worker {task_id} -> {hv.get('workdir')}")
                    SUP.phase_failed(mission_id, phase_id, worker_by="mission-loop",
                                     note=f"attempts exhausted: {task_id} (worktree "
                                          f"harvest: {hv.get('status_lines', 0)} uncommitted)")
                    break

            if decision.verdict == "SUCCESS":
                evidence = wstate.get("completion_evidence") or []
                # s6-ownership: a COMPLETE worker's pytest batches/tools must
                # not leak past its terminal state. Reap the tree (worker pid
                # is alive or a just-exited zombie — safe to walk) before the
                # phase closes. This is a kill of descendants only.
                SUP.reap_worker_processes(wp_pid)
                ok, msg = SUP.phase_complete(
                    mission_id, phase_id, worker_by=task_id, evidence=evidence,
                    workdir=wstate.get("workdir") or None)
                if not ok:
                    print(f"[mission-loop] {phase_id}: worker claimed SUCCESS but "
                          f"evidence rejected; marking FAILED: {msg}")
                    SUP.phase_failed(mission_id, phase_id, worker_by=task_id,
                                     note=msg)
                else:
                    print(f"[mission-loop] {phase_id}: {msg}")
                    phases_done += 1
                    # P-18: worker recommendations from phase workers become
                    # candidate work as well.
                    from hermes_cli import mission_ops as MO
                    mh = SUP.load_mission(mission_id)
                    for src in ("next_action", "blockers"):
                        txt = str(wstate.get(src) or "").strip()
                        import re as _re
                        _words = _re.findall(r"[a-z]{4,}", txt.lower())
                        if (_words and len(_words) >= 2 and len(txt) > 12
                                and txt.lower() not in ("none", "done", "nothing")
                                and not txt.strip().startswith("none")):
                            MO.add_discovery(
                                mh, title=txt[:120],
                                rationale=f"worker {task_id} ({src}) after {phase_id}",
                                discoverer=task_id,
                                evidence=[txt, "from completed worker %s" % task_id],
                                priority=3)
                    if mh.get("discoveries"):
                        changed, _n = MO.plan_discoveries(mh)
                        SUP.save_mission(mh)
                        if changed:
                            print(f"[mission-loop] {phase_id}: planner admitted "
                                  f"{changed} worker discoveries")
                break
            # P-14 (mission run 1): verdict UNVERIFIED_COMPLETION/VERIFY must
            # not poll forever. The worker is gone (dead pid) or its evidence
            # failed the gate; a claim alone cannot advance or block the
            # mission. If the process is dead / attempts exhausted, FAIL the
            # phase and let the mission CONTINUE with remaining phases (an
            # unverifiable completion is not a terminal condition either way).
            if decision.verdict in ("UNVERIFIED_COMPLETION",) or decision.command in ("VERIFY",):
                if wp_pid and not SUP._pid_alive(wp_pid):
                    # s6-ownership: dead worker — reap whatever it left behind.
                    SUP.reap_worker_processes(wp_pid)
                    print(f"[mission-loop] {phase_id}: completion unverifiable "
                          f"and worker dead; failing phase, mission continues")
                    SUP.phase_failed(mission_id, phase_id, worker_by="mission-loop",
                                     note="UNVERIFIED_COMPLETION with dead worker",
                                     retryable=True)
                    break
                if not SUP.attempts_left(wstate):
                    SUP.reap_worker_processes(wp_pid)
                    print(f"[mission-loop] {phase_id}: completion unverifiable "
                          f"and attempts exhausted; failing phase, mission continues")
                    SUP.phase_failed(mission_id, phase_id, worker_by="mission-loop",
                                     note="UNVERIFIED_COMPLETION attempts exhausted",
                                     retryable=True)
                    break
                # alive worker, attempts remain: keep supervising (it may
                # re-verify); do not block the loop.
                _block_until_event(CUR_EVENT_SOCK, proc, every, max_secs=max_secs, start_t=start_t)
                continue
            if decision.command == "CANCEL" or decision.verdict in ("CANCELLED", "CANCEL"):
                # Worker exhausted attempts/interventions (verdict WORKER_CRASH/
                # WORKER_TIMEOUT/WORKER_FAILURE with command CANCEL). `supervise
                # loop` treats command CANCEL as terminal; the mission loop must
                # too — before this branch a dead exhausted worker spun the loop
                # on a dead pid at ~1,050 iter/s (measured 12,683 lines/12s,
                # phase never closed, mission wedged). Harvest what the worker
                # left, then close the phase and continue the mission.
                hv = SUP.harvest_worker_worktree(task_id)
                if hv.get("status"):
                    print(f"[mission-loop] harvested uncommitted worktree "
                          f"evidence ({hv['status_lines']} files) for "
                          f"exhausted worker {task_id} -> {hv.get('workdir')}")
                note = ("cancelled" if decision.verdict == "CANCELLED"
                        else f"worker {decision.verdict} terminal: "
                             f"{decision.instruction} (worktree harvest: "
                             f"{hv.get('status_lines', 0)} uncommitted)")
                # s6-ownership: CANCELLED/terminal workers leak their tool
                # children today; reap before closing the phase.
                SUP.reap_worker_processes(wp_pid)
                SUP.phase_failed(mission_id, phase_id, worker_by="mission-loop",
                                 note=note)
                break
            _block_until_event(CUR_EVENT_SOCK, proc, every, max_secs=max_secs, start_t=start_t)
    # unreachable in the loop; keep the compiler/exhaustive happy
    return 0


def build_supervise_parser(subparsers, *, cmd_supervise) -> None:
    p = subparsers.add_parser(
        "supervise",
        help="Create, start, monitor, and cancel an autonomous Hermes worker",
        description=(
            "Autonomous worker supervision: create a task brief, start a "
            "one-shot Hermes worker, then monitor progress and issue "
            "structured commands. Reuses hermes chat -q subprocesses, sessions, "
            "and diagnostics; no separate framework."
        ),
    )
    sub = p.add_subparsers(dest="supervise_action")

    c = sub.add_parser("create", help="Create a worker state + brief")
    c.add_argument("--task", required=True, help="Task description for the worker")
    c.add_argument("--workdir", default=None, help="Working directory")
    c.add_argument("--budget-max-turns", type=int, default=SUP.DEFAULT_BUDGET["max_worker_turns"])
    c.add_argument("--task-id", default=None)
    c.set_defaults(func=lambda a: _dispatch(a, "create"))

    s = sub.add_parser("start", help="Start the worker subprocess")
    s.add_argument("task_id")
    s.add_argument("--model", default=None)
    s.set_defaults(func=lambda a: _dispatch(a, "start"))

    at = sub.add_parser("attach",
                        help="Supervisor-side anchor: register an externally "
                             "spawned pid as the started worker (CAS, same "
                             "path as start/RETRY). Worker-initiated writes "
                             "cannot set worker_pid (protected); only the "
                             "supervisor borders may. Used by test drivers "
                             "that stand in for real spawned workers.")
    at.add_argument("task_id")
    at.add_argument("--pid", type=int, required=True)
    at.set_defaults(func=lambda a: _dispatch(a, "attach"))

    st = sub.add_parser("status", help="Show worker state")
    st.add_argument("task_id")
    st.set_defaults(func=lambda a: _dispatch(a, "status"))

    chk = sub.add_parser("check", help="Run one supervisor evaluation")
    chk.add_argument("task_id")
    chk.set_defaults(func=lambda a: _dispatch(a, "check"))

    loop = sub.add_parser("loop", help="Monitor loop (poll + evaluate)")
    loop.add_argument("task_id")
    loop.add_argument("--every", type=float, default=5.0)
    loop.add_argument("--max-iterations", type=int, default=0)
    loop.add_argument("--max-seconds", type=float, default=0.0)
    loop.add_argument("--campaign", default=None,
                      help="Campaign ledger to record outcomes against")
    loop.add_argument("--role", default=None,
                      help="Role obligation this worker owns (for replacement policy)")
    loop.add_argument("--max-replacements", type=int, default=2,
                      help="Max autonomous replacements for this role (default 2)")
    loop.set_defaults(func=lambda a: _dispatch(a, "loop"))

    cc = sub.add_parser("cancel", help="Cancel a worker")
    cc.add_argument("task_id")
    cc.set_defaults(func=lambda a: _dispatch(a, "cancel"))

    msg = sub.add_parser("message", help="Post a durable inbox message to a worker")
    msg.add_argument("task_id")
    msg.add_argument("--receiver", default="worker",
                     help="Receiver address (default: worker)")
    msg.add_argument("--text", required=True, help="Message text")
    msg.add_argument("--kind", default="message",
                     help="kind: message|handoff|followup (default: message)")
    msg.add_argument("--thread", default=None, help="thread_id for correlation")
    msg.add_argument("--reply-to", default=None, help="message id this replies to")
    msg.add_argument("--from", "--sender", dest="sender", default="supervisor",
                     help="Sender identity (default: supervisor)")
    msg.set_defaults(func=lambda a: _dispatch(a, "message"))

    ack = sub.add_parser("ack", help="Acknowledge a delivered message in the ledger")
    ack.add_argument("task_id")
    ack.add_argument("msg_id")
    ack.add_argument("--by", default="worker")
    ack.set_defaults(func=lambda a: _dispatch(a, "ack"))

    inbox = sub.add_parser("inbox", help="List inbox messages for a task")
    inbox.add_argument("task_id")
    inbox.add_argument("--status", default=None)
    inbox.set_defaults(func=lambda a: _dispatch(a, "inbox"))

    state = sub.add_parser("state", help="Show worker state + handoff lineage")
    state.add_argument("task_id")
    state.set_defaults(func=lambda a: _dispatch(a, "state"))

    stw = sub.add_parser("state-write",
                         help="CAS apply a worker state patch (worker protocol)")
    stw.add_argument("task_id")
    stw.add_argument("--expect-seq", type=int, default=None,
                     help="ledger seq the worker read (stale-guard token)")
    stw.add_argument("--json", required=True, help="JSON patch (fields updated)")
    stw.set_defaults(func=lambda a: _dispatch(a, "state-write"))

    lineage = sub.add_parser("lineage", help="Show attempt + handoff lineage")
    lineage.add_argument("task_id")
    lineage.set_defaults(func=lambda a: _dispatch(a, "lineage"))

    camp = sub.add_parser("campaign", help="Campaign role/obligation model")
    csub = camp.add_subparsers(dest="campaign_action")
    cc = csub.add_parser("create", help="Create a campaign ledger")
    cc.add_argument("campaign_id")
    cc.add_argument("--objective", required=True)
    cc.add_argument("--roles", default="[]",
                    help='JSON: [{"role_id","responsibility","required_evidence":[..]}]')
    cc.set_defaults(func=lambda a: _dispatch(a, "campaign-create"))
    cr = csub.add_parser("assign", help="Assign a role to a worker task")
    cr.add_argument("campaign_id")
    cr.add_argument("--role", required=True)
    cr.add_argument("--task", required=True)
    cr.set_defaults(func=lambda a: _dispatch(a, "campaign-assign"))
    cs_ = csub.add_parser("outcome", help="Record a worker outcome (WORKER_*)")
    cs_.add_argument("campaign_id")
    cs_.add_argument("--task", required=True)
    cs_.add_argument("--outcome", required=True,
                     choices=SUP.WORKER_OUTCOMES)
    cs_.add_argument("--evidence", action="append", default=[])
    cs_.set_defaults(func=lambda a: _dispatch(a, "campaign-outcome"))
    cok = csub.add_parser("satisfy", help="Mark role satisfied with evidence")
    cok.add_argument("campaign_id")
    cok.add_argument("--role", required=True)
    cok.add_argument("--task", required=True)
    cok.add_argument("--evidence", action="append", required=True)
    cok.set_defaults(func=lambda a: _dispatch(a, "campaign-satisfy"))
    crec = csub.add_parser("reconcile",
                           help="Reconcile a worker failure (evidence-gated)")
    crec.add_argument("campaign_id")
    crec.add_argument("--task", required=True)
    crec.add_argument("--findings-preserved", action="store_true", default=True)
    crec.add_argument("--responsible-covered", action="store_true")
    crec.add_argument("--transfer-to", default="")
    crec.add_argument("--adversarial-satisfied", action="store_true")
    crec.add_argument("--completion-ok", action="store_true")
    crec.add_argument("--evidence", action="append", default=[])
    crec.add_argument("--note", default="")
    crec.set_defaults(func=lambda a: _dispatch(a, "campaign-reconcile"))
    crep = csub.add_parser("replace",
                           help="Spawn a replacement worker for a failed task")
    crep.add_argument("campaign_id")
    crep.add_argument("--role", required=True)
    crep.add_argument("--task", required=True)
    crep.add_argument("--replace-id", default=None)
    crep.add_argument("--budget-max-turns", type=int, default=None)
    crep.set_defaults(func=lambda a: _dispatch(a, "campaign-replace"))
    ctr = csub.add_parser("transfer", help="Transfer a role to another worker")
    ctr.add_argument("campaign_id")
    ctr.add_argument("--task", required=True)
    ctr.add_argument("--to", required=True)
    ctr.set_defaults(func=lambda a: _dispatch(a, "campaign-transfer"))
    cst = csub.add_parser("status", help="Campaign status + rationale")
    cst.add_argument("campaign_id")
    cst.set_defaults(func=lambda a: _dispatch(a, "campaign-status"))

    # ---- mission: multi-phase autonomous continuation ------------------ -
    mis = sub.add_parser("mission", help="Mission phase/terminal model "
                                        "(CAMPAIGN_COMPLETE != MISSION_COMPLETE)")
    msub = mis.add_subparsers(dest="mission_action")
    mm = msub.add_parser("create", help="Create a mission ledger")
    mm.add_argument("mission_id")
    mm.add_argument("--objective", required=True)
    mm.add_argument("--phases", default="[]",
                    help='JSON: [{"phase_id","task","required_evidence":[...],"after":""}]')
    mm.add_argument("--requirements", default="[]",
                    help='JSON: ["evidence keyword", ...] terminal requirements')
    mm.add_argument("--workdir", default="",
                    help="Working directory for mission work (commits/files verified here)")
    mm.set_defaults(func=lambda a: _dispatch(a, "mission-create"))
    mp_ = msub.add_parser("list", help="List missions")
    mp_.set_defaults(func=lambda a: _dispatch(a, "mission-list"))
    ms = msub.add_parser("status", help="Mission status + rationale")
    ms.add_argument("mission_id")
    ms.set_defaults(func=lambda a: _dispatch(a, "mission-status"))
    mn = msub.add_parser("next", help="Show the next PENDING phase (continuation engine)")
    mn.add_argument("mission_id")
    mn.set_defaults(func=lambda a: _dispatch(a, "mission-next"))
    mp = msub.add_parser("phase-complete",
                         help="Evidence-gated phase completion (auto-continuation)")
    mp.add_argument("mission_id")
    mp.add_argument("phase_id")
    mp.add_argument("--worker-by", default="")
    mp.add_argument("--evidence", action="append", default=[])
    mp.add_argument("--workdir", default="",
                    help="repo root for artifact verification (default: repo "
                         "of the launched process; empty = auto)")
    mp.set_defaults(func=lambda a: _dispatch(a, "mission-phase-complete"))
    mblk = msub.add_parser("phase-blocked",
                           help="Mark phase BLOCKED with a documented reason")
    mblk.add_argument("mission_id")
    mblk.add_argument("phase_id")
    mblk.add_argument("--worker-by", default="")
    mblk.add_argument("--note", default="")
    mblk.set_defaults(func=lambda a: _dispatch(a, "mission-phase-blocked"))
    mfl = msub.add_parser("phase-failed",
                          help="Mark phase FAILED with a note")
    mfl.add_argument("mission_id")
    mfl.add_argument("phase_id")
    mfl.add_argument("--worker-by", default="")
    mfl.add_argument("--note", default="")
    mfl.set_defaults(func=lambda a: _dispatch(a, "mission-phase-failed"))
    mpa = msub.add_parser("phase-add",
                          help="Append a PENDING phase to an active mission "
                               "(self-service ledger edit)")
    mpa.add_argument("mission_id")
    mpa.add_argument("phase_id")
    mpa.add_argument("task")
    mpa.add_argument("--after", default="",
                    help="phase_id that must be COMPLETE first")
    mpa.add_argument("--evidence", action="append", default=[],
                     help="required evidence: literal keyword OR JSON "
                          '{"kind":"commit"|"test_run"|"file"} (repeatable)')
    mpa.set_defaults(func=lambda a: _dispatch(a, "mission-phase-add"))
    mpl = msub.add_parser("phase-list",
                          help="List phases compactly (id, status, task)")
    mpl.add_argument("mission_id")
    mpl.set_defaults(func=lambda a: _dispatch(a, "mission-phase-list"))
    mf = msub.add_parser("finding", help="Record an unresolved finding")
    mf.add_argument("mission_id")
    mf.add_argument("finding_id")
    mf.add_argument("--text", required=True)
    mf.set_defaults(func=lambda a: _dispatch(a, "mission-finding"))
    mfr = msub.add_parser("finding-resolve",
                          help="Resolve an open finding (evidence-gated)")
    mfr.add_argument("mission_id")
    mfr.add_argument("finding_id")
    mfr.add_argument("--evidence", action="append", required=True)
    mfr.set_defaults(func=lambda a: _dispatch(a, "mission-finding-resolve"))
    mfb = msub.add_parser("finding-block",
                          help="Document a finding as genuinely blocked")
    mfb.add_argument("mission_id")
    mfb.add_argument("finding_id")
    mfb.add_argument("--note", default="")
    mfb.set_defaults(func=lambda a: _dispatch(a, "mission-finding-block"))
    # orchestrator loop: run each phase's worker task automatically;
    # creates -> starts -> loops -> verifies -> next phase until terminal
    _md = msub.add_parser("discover", help="Record a discovery (worker/verif/research)")
    _md.add_argument("mission_id")
    _md.add_argument("--title", required=True)
    _md.add_argument("--why", default="")
    _md.add_argument("--discoverer", default="")
    _md.add_argument("--evidence", action="append", default=[])
    _md.add_argument("--priority", type=int, default=3)
    _md.add_argument("--deps", default="")
    _md.add_argument("--supersedes", default="",
                     help="Existing discovery id this new work invalidates")
    _md.set_defaults(func=lambda a: _dispatch(a, "mission-discover"))
    _mplan = msub.add_parser("plan", help="Planner: promote/defer discoveries")
    _mplan.add_argument("mission_id")
    _mplan.set_defaults(func=lambda a: _dispatch(a, "mission-plan"))
    _mtodo = msub.add_parser("todo", help="List open discoveries")
    _mtodo.add_argument("mission_id")
    _mtodo.set_defaults(func=lambda a: _dispatch(a, "mission-todo"))
    _mev = msub.add_parser("evaluate", help="Completion evaluator")
    _mev.add_argument("mission_id")
    _mev.set_defaults(func=lambda a: _dispatch(a, "mission-evaluate"))
    _mdod = msub.add_parser("dod", help="Definition of done bookkeeping")
    _mdod.add_argument("mission_id")
    _mdod.add_argument("--install", action="store_true")
    _mdod.add_argument("--satisfy", action="store_true")
    _mdod.add_argument("--dim", default="")
    _mdod.add_argument("--what", default="")
    _mdod.add_argument("--who", default="")
    _mdod.add_argument("--derive", default="",
                      help="Workdir to derive project-outcome DoD dims from "
                           "(tests/, docs/, git/source presence)")
    _mdod.set_defaults(func=lambda a: _dispatch(a, "mission-dod"))
    _mls = msub.add_parser("lesson", help="Record/recall lessons")
    _mls.add_argument("mission_id")
    _mls.add_argument("--get", action="store_true")
    _mls.add_argument("--text", default="")
    _mls.add_argument("--context", default="")
    _mls.add_argument("--better", default="")
    _mls.add_argument("--scope", default="")
    _mls.set_defaults(func=lambda a: _dispatch(a, "mission-lesson"))
    _menc = msub.add_parser("env", help="Environment snapshot")
    _menc.set_defaults(func=lambda a: _dispatch(a, "mission-env"))
    _mfo = msub.add_parser("forensics", help="Mission forensics")
    _mfo.add_argument("mission_id")
    _mfo.set_defaults(func=lambda a: _dispatch(a, "mission-forensics"))
    pev = msub.add_parser("events", help="Read the mission event journal (controller visibility)")
    pev.add_argument("mission_id")
    pev.add_argument("--level", default="LOW", choices=["LOW", "MEDIUM", "HIGH"])
    pev.add_argument("--limit", type=int, default=50)
    pev.set_defaults(func=lambda a: _dispatch(a, "mission-events"))
    pls = msub.add_parser("lessons", help="Sync mission experience into the platform lesson store / show hint")
    pls.add_argument("mission_id")
    pls.add_argument("--sync", action="store_true")
    pls.set_defaults(func=lambda a: _dispatch(a, "mission-lessons"))
    ml = msub.add_parser("loop", help="Run the autonomous mission orchestrator")
    ml.add_argument("mission_id")
    ml.add_argument("--every", type=float, default=10.0)
    ml.add_argument("--max-seconds", type=float, default=0.0)
    ml.add_argument("--max-phases", type=int, default=0)
    ml.add_argument("--model", default=None)
    ml.set_defaults(func=lambda a: _dispatch(a, "mission-loop"))

    mctl = msub.add_parser(
        "controller",
        help="Controller layer: durable, event-triggered controller reasoning "
             "above the supervisor (P-26)")
    mctl.add_argument("mission_id")
    mctl.add_argument("--arm", action="store_true",
                      help="Create the durable controller wake: cron monitor "
                           "job + per-mission state (idempotent)")
    mctl.add_argument("--disarm", action="store_true")
    mctl.add_argument("--ack", action="store_true",
                      help="Advance the wake watermark after processing")
    mctl.add_argument("--wake", action="store_true",
                      help="Print current wake output (used by cron monitor)")
    mctl.add_argument("--wait", action="store_true",
                      help="Block until a meaningful event (or timeout); "
                           "turns nothing to a 590s proc wait")
    mctl.add_argument("--status", action="store_true")
    mctl.add_argument("--level", default="MEDIUM",
                      choices=["LOW", "MEDIUM", "HIGH"])
    mctl.add_argument("--interval", default="every 5m",
                      help="Cron schedule for the wake check")
    mctl.add_argument("--policy", default="",
                      help="Extra durable rules for the controller session")
    mctl.add_argument("--deliver", default="local",
                      help="Where the controller digest is delivered "
                           "(default: local; use 'origin' in a live chat)")
    mctl.add_argument("--timeout", type=float, default=30.0,
                      help="--wait maximum seconds")
    mctl.set_defaults(func=lambda a: _dispatch(a, "mission-controller"))

    mls = msub.add_parser("lease", help="Show the supervisor single-owner lease")
    mls.add_argument("mission_id")
    mls.set_defaults(func=lambda a: _dispatch(a, "mission-lease"))

    mm_ = msub.add_parser("metrics", help="Post-mission performance breakdown"
                           " (derived from durable state)")
    mm_.add_argument("mission_id")
    mm_.set_defaults(func=lambda a: _dispatch(a, "mission-metrics"))

    mr_ = msub.add_parser("rationale", help="Deterministic continuation/stop"
                           " rationale (diminishing returns)")
    mr_.add_argument("mission_id")
    mr_.set_defaults(func=lambda a: _dispatch(a, "mission-rationale"))

    mo_ = msub.add_parser("optimize", help="Evidence-backed waste/optimization"
                           " plan from mission metrics")
    mo_.add_argument("mission_id")
    mo_.add_argument("--apply", action="store_true",
                     help="Record the plan as evidence-backed discoveries")
    mo_.set_defaults(func=lambda a: _dispatch(a, "mission-optimize"))

    mr2 = msub.add_parser("retrospective", help="Deterministic mission-level"
                           " self-critique at completion")
    mr2.add_argument("mission_id")
    mr2.set_defaults(func=lambda a: _dispatch(a, "mission-retrospective"))

    mbl = msub.add_parser("backlog", help="Durable capability backlog: the"
                          " source an autonomous loop drains after phases and"
                          " discoveries run out")
    mbl.add_argument("mission_id")
    mbl.add_argument("--add", default="", help="Add backlog item: item_id")
    mbl.add_argument("--title", default="", help="Title for --add")
    mbl.add_argument("--priority", type=int, default=3)
    mbl.add_argument("--why", default="", help="Rationale for --add")
    mbl.add_argument("--materialize", action="store_true",
                     help="Turn the highest-priority OPEN item into a "
                          "discovery the supervisor can execute")
    mbl.add_argument("--list", action="store_true")
    mbl.set_defaults(func=lambda a: _dispatch(a, "mission-backlog"))

    mtel = msub.add_parser("telemetry", help="Worker-activity telemetry "
                           "report (derived from the durable audit rail)")
    mtel.add_argument("mission_id")
    mtel.set_defaults(func=lambda a: _dispatch(a, "mission-telemetry"))

    mgate = msub.add_parser(
        "gate", help="Deterministic 'may I report done?' gate. An agent or "
                     "controller MUST run this before declaring a mission "
                     "objective satisfied: tests green / commit created / "
                     "verification finished are never the terminal gate. "
                     "Exit 0 = may stop (evidence shown); exit 1 = meaningful "
                     "work remains, do NOT report done.")
    mgate.add_argument("mission_id")
    mgate.set_defaults(func=lambda a: _dispatch(a, "mission-gate"))

    mdet = msub.add_parser(
        "detect", help="Architected detector pass: observable waste signals "
                       "become durable evidence-backed discoveries (no LLM "
                       "initiative needed). Idempotent.")
    mdet.add_argument("mission_id")
    mdet.add_argument("--max-new", type=int, default=4)
    mdet.set_defaults(func=lambda a: _dispatch(a, "mission-detect"))

    mexpl = msub.add_parser(
        "explore", help="First-class parallel exploration: register N "
                        "competing approach variants for a topic (bounded, "
                        "runs under the normal concurrency cap)")
    mexpl.add_argument("mission_id")
    mexpl.add_argument("--topic", required=True)
    mexpl.add_argument("--variants", required=True,
                      help="Semicolon-separated approach variants, e.g. "
                           "'keep-httpx;use-curl_cffi;playwright-render'")
    mexpl.add_argument("--max-variants", type=int, default=3)
    mexpl.set_defaults(func=lambda a: _dispatch(a, "mission-explore"))

    mbm = msub.add_parser(
        "benchmark", help="Record/list durable benchmark measurements on the "
                          "mission ledger")
    mbm.add_argument("mission_id")
    mbm.add_argument("--name", default="", help="Benchmark/measurement name")
    mbm.add_argument("--value", default="", help="Measured value (string, no "
                                                 "unit fabrication)")
    mbm.add_argument("--provenance", default="",
                    help="How/where the measurement came from")
    mbm.add_argument("--list", action="store_true")
    mbm.set_defaults(func=lambda a: _dispatch(a, "mission-benchmark"))

    mknow = msub.add_parser(
        "knowledge", help="Cross-session institutional memory: what do we "
                          "already know about a subject (lessons, prior "
                          "missions, capabilities, machine)")
    mknow.add_argument("mission_id")
    mknow.add_argument("--subject", default="", help="Subject to retrieve "
                                                     "institutional knowledge about")
    mknow.set_defaults(func=lambda a: _dispatch(a, "mission-knowledge"))

    matk = msub.add_parser(
        "attack", help="Adversarial self-testing: derive an attack discovery "
                       "from a changed module (telemetry/worker evidence)")
    matk.add_argument("mission_id")
    matk.add_argument("--module", default="", required=True,
                      help="Changed module/parser/regex to attack")
    matk.set_defaults(func=lambda a: _dispatch(a, "mission-attack"))

    ls = sub.add_parser("list", help="List all workers")
    ls.set_defaults(func=lambda a: _dispatch(a, "list"))
    rp = sub.add_parser("reap", help="Reap stale unstarted workers")
    rp.add_argument("--max-age-days", type=float, default=7.0,
                    help="Workers older than this many days are eligible (default: 7)")
    rp.add_argument("--dry-run", action="store_true",
                    help="Show what would be reaped without deleting")
    rp.set_defaults(func=lambda a: _dispatch(a, "reap"))

    p.set_defaults(func=lambda a: _dispatch(a, "list"))


def _dispatch(args, action: str) -> int:
    try:
        if action == "create":
            budget = dict(SUP.DEFAULT_BUDGET)
            budget["max_worker_turns"] = int(getattr(args, "budget_max_turns", 60) or 60)
            task_id, state = SUP.create_worker(
                args.task, task_id=getattr(args, "task_id", None),
                budget=budget, workdir=getattr(args, "workdir", None))
            print(f"created: {task_id}")
            print(json.dumps({"task_id": task_id, "status": state["status"]}, indent=2))
            print(f"brief: {SUP.worker_path(task_id).parent / 'brief.md'}")
            return 0
        if action == "start":
            pid, spawned = SUP.start_worker_guarded(
                args.task_id, model=getattr(args, "model", None))
            if not spawned:
                print(f"worker {args.task_id} is already running (pid {pid}); "
                      f"not starting a duplicate")
                return 0
            print(f"started worker {args.task_id} (pid {pid})")
            print(f"log: {SUP.worker_path(args.task_id).parent / 'worker.log'}")
            return 0
        if action == "attach":
            ok_att, msg_att = SUP.guard_attach(args.task_id, int(args.pid))
            if not ok_att:
                print(f"attach refused: {msg_att}")
                return 1
            if not SUP.record_spawned_pid(args.task_id, int(args.pid)):
                print(f"attach failed: no worker {args.task_id}")
                return 1
            st = SUP.load_worker(args.task_id) or {}
            print(f"attached worker {args.task_id} (pid {args.pid}, "
                  f"run_id {st.get('run_id', '')})")
            return 0
        if action == "status":
            state = SUP.load_worker(args.task_id)
            if not state:
                print(f"no worker {args.task_id}")
                return 1
            print(json.dumps(state, indent=2, default=str))
            return 0
        if action == "check":
            state = SUP.load_worker(args.task_id)
            if not state:
                print(f"no worker {args.task_id}")
                return 1
            state["worker_pid"] = int(state.get("worker_pid") or 0)
            decision = SUP.evaluate_worker(state, previous=None)
            print(json.dumps(decision.as_dict(), indent=2))
            SUP.write_command(args.task_id, decision)
            return 0
        if action == "loop":
            task_id = args.task_id
            every = float(getattr(args, "every", 5.0) or 5.0)
            max_iter = int(getattr(args, "max_iterations", 0) or 0)
            max_secs = float(getattr(args, "max_seconds", 0.0) or 0.0)
            campaign_id = getattr(args, "campaign", "") or ""
            role_id = getattr(args, "role", "") or ""
            max_replacements = int(getattr(args, "max_replacements", 2) or 0)
            replacements = 0
            start_t = time.time()
            iter_n = 0
            # The end-of-iteration event wake uses `proc`, but a worker may be
            # anchored by `attach`/`start`, never by this loop's respawn path.
            # Dead-but-nonlocal `proc` crashed the whole supervise loop
            # (UnboundLocalError) whenever the verdict was not RETRY/terminal;
            # the hardening 20-worker suite exposed it. The wake call handles
            # None (socket wait falls back to the interval watchdog).
            proc = None
            prev = None
            prev_fp = None
            stall_count = 0
            global CUR_EVENT_SOCK
            CUR_EVENT_SOCK = _bind_event_socket(task_id, every)
            while True:
                if max_secs and (time.time() - start_t) > max_secs:
                    print("[loop] max_seconds reached")
                    return 0
                if max_iter and iter_n >= max_iter:
                    print("[loop] max_iterations reached")
                    return 0
                state = SUP.load_worker(task_id)
                if not state:
                    print("[loop] worker state missing; stopping")
                    return 1
                state["worker_pid"] = int(state.get("worker_pid") or 0)
                pid = int(state.get("worker_pid") or 0)
                SUP.expire_stale_messages(task_id)
                acked = SUP.check_acks(task_id, state)
                if acked:
                    print(f"[loop] inbox: worker acked message {acked}")
                delivered = SUP.deliver_all_if_idle(task_id, state)
                if delivered:
                    print(f"[loop] inbox: delivered {delivered} pending message(s)")
                # P5 watchdog: fingerprint across iterations, kill only a
                # live + stale-activity + unchanged-state worker (never a
                # BLOCKED/WAITING/NEEDS_INPUT one)
                fp = SUP.state_fingerprint(state)
                action, stall_count = SUP.watchdog_assess(
                    state, pid=pid, fingerprint=fp,
                    prev_fingerprint=prev_fp, stall_count=stall_count,
                    now=time.time(),
                    log_path=SUP.worker_path(task_id).parent / "worker.log")
                if action == "stall":
                    print(f"[loop] watchdog STALL: hung worker pid {pid} — kill + retry within attempt budget")
                prev_fp = fp
                decision = SUP.evaluate_worker(state, previous=prev, now=time.time(),
                                               pid=pid)
                # STALL overrides the sequential verdict; evidence first.
                if action == "stall":
                    decision = SUP.WorkerDecision(
                        "STALL", "RETRY",
                        "Worker hung (stale heartbeat + fingerprint unchanged for "
                        f"{stall_count} ticks). Kill process and respawn within budget.", 0.05)
                SUP.write_command_if_changed(task_id, decision)  # P-17: skip identical
                prev = state
                iter_n += 1
                print(f"[{iter_n}] {state.get('status')} -> {decision.verdict} "
                      f"[{decision.command}] {decision.instruction[:80]}")
                # P1/P5: bounded respawn on crash/timeout/failure/hang verdicts
                if decision.command == "RETRY" and decision.verdict in (
                        "WORKER_CRASH", "WORKER_TIMEOUT", "WORKER_FAILURE", "STALL"):
                    if decision.verdict == "STALL":
                        SUP.kill_worker(pid)
                    st = SUP.load_worker(task_id)
                    if st is None:
                        print("[loop] worker state disappeared; stopping")
                        return 1
                    # P-11: never respawn a worker that was NOT started yet
                    # (worker_pid=0 and no started_at). A crash verdict on an
                    # unowned task means the spawn raced the ledger write, not
                    # that the worker died; spawning here compounds the race
                    # with a stub process and burns the attempt budget
                    # (measured: 20-worker concurrency hit attempt=3 and 0/20
                    # completed). Wait one polling cycle instead.
                    anchored = int(st.get("worker_pid") or 0) and st.get("started_at")
                    if not anchored:
                        print(f"[loop] worker not yet started (pid=0/no started_at); "
                              f"deferring retry for {task_id}")
                        # busy-spin guard: without the sleep this branch
                        # hot-loops at thousands of iterations/sec (attack
                        # found: unanchored task → CPU burn + log flooding)
                        time.sleep(every)
                        continue
                    left = SUP.attempts_left(st)
                    if left > 0:
                        SUP.bump_attempt(st, reason=decision.verdict.lower())
                        # P5: bump under CAS so we never clobber a newer write
                        st["seq"] = SUP.next_seq(st)
                        ok, why = SUP.save_worker_cas(task_id, st)
                        if not ok:
                            print(f"[loop] CAS rejected after bump ({why}); re-reading")
                            st = SUP.load_worker(task_id) or st
                        else:
                            print(f"[loop] respawn attempt {SUP.attempt_number(st)} "
                                  f"({left - 1} left): {decision.verdict}")
                        # P5: persist the NEW pid under CAS so crash/timeout
                        # detection applies to the respawned process too
                        SUP.start_worker_guarded(task_id, force=True)
                        proc = None
                        iter_n -= 1  # don't burn an iteration on the respawn
                        prev = None
                        prev_fp = None
                        stall_count = 0
                    else:
                        st["status"] = "FAILED"
                        st["phase"] = "FAILED"
                        SUP.save_worker(state=st, task_id=task_id)
                        # s6-ownership: exhausted worker's tool children must
                        # not outlive the terminal decision (s4 leak).
                        SUP.reap_worker_processes(pid)
                        # P6: autonomous replacement for campaign roles — the
                        # supervisor decides, bounded by max_replacements.
                        # The bound is a ROLE budget that lives in the campaign
                        # ledger (restart-safe): a restarted supervisor must
                        # not re-spawn replacements for a failure that was
                        # already replaced (attack: restart re-arms the loop's
                        # in-memory counter and grows worker ledgers unbounded).
                        if campaign_id and role_id and replacements < max_replacements:
                            cam_now = SUP.load_campaign(campaign_id)
                            if cam_now:
                                used = sum(
                                    1 for _w in (cam_now.get("workers") or {}).values()
                                    if _w.get("role") == role_id and _w.get("replaces"))
                                if used >= max_replacements:
                                    print(f"[loop] role {role_id} replacement budget "
                                          f"already used ({used}/{max_replacements}); "
                                          "skipping replacement")
                                else:
                                    SUP.note_worker_outcome(campaign_id, task_id,
                                                            "WORKER_FAILED",
                                                            evidence=[decision.instruction])
                                    ok_r, msg_r, new_id = SUP.spawn_replacement(
                                        campaign_id, role_id, task_id,
                                        budget=st.get("budget"))
                                    if ok_r:
                                        replacements += 1
                                        task_id = new_id
                                        prev = None
                                        prev_fp = None
                                        stall_count = 0
                                        iter_n = 0
                                        # start the replacement NOW (fresh
                                        # pid) instead of waiting for a
                                        # crash-detection cycle on pid 0
                                        try:
                                            SUP.start_worker_guarded(
                                                new_id, force=True)
                                        except Exception as exc:  # noqa: BLE001
                                            # keep `proc` bound; the wake call
                                            # below must not see a stale/unset
                                            # process from a previous attempt
                                            proc = None
                                            print(f"[loop] replacement start failed: {exc}")
                                        print(f"[loop] {msg_r} — supervision continues")
                                        _block_until_event(CUR_EVENT_SOCK, proc, every)
                                        continue
                        print(f"[loop] attempts exhausted -> FAILED ({task_id})")
                        hv = SUP.harvest_worker_worktree(task_id)
                        if hv.get("status"):
                            print(f"[loop] harvested uncommitted worktree "
                                  f"evidence ({hv['status_lines']} files) for "
                                  f"exhausted worker {task_id} -> {hv.get('workdir')}")
                        return 0
                if decision.command == "CANCEL":
                    SUP.reap_worker_processes(pid)
                    print(f"[loop] terminal: {decision.verdict} [{decision.command}]")
                    return 0
                if decision.verdict == "SUCCESS":
                    if campaign_id and role_id and SUP.load_campaign(campaign_id):
                        SUP.note_worker_outcome(campaign_id, task_id,
                                                "WORKER_COMPLETE",
                                                evidence=state.get("completion_evidence") or [])
                    # s6-ownership: COMPLETE workers must not leak their tool
                    # children; reap the tree before closing the loop.
                    SUP.reap_worker_processes(pid)
                    print("[loop] SUCCESS")
                    return 0
                if decision.verdict in ("CANCELLED",):
                    SUP.reap_worker_processes(pid)
                    print(f"[loop] terminal: {decision.verdict}")
                    return 0
                # event-driven: wake on the task's own socket (its state-write
                # notifies), not the interval. Fall back to the interval as the
                # safety watchdog only when the socket is unavailable.
                _block_until_event(CUR_EVENT_SOCK, proc, every, max_secs=max_secs, start_t=start_t)
        if action == "cancel":
            SUP.cancel_worker(args.task_id)
            # s6-ownership: the cancel action returns immediately; reap the
            # worker's tree now (worker alive, ancestry walk valid) so its
            # tool children don't outlive the cancellation.
            SUP.reap_worker_processes(int((SUP.load_worker(args.task_id) or {}).get("worker_pid") or 0))
            print(f"cancelled {args.task_id}")
            return 0
        if action == "message":
            if SUP.load_worker(args.task_id) is None:
                print(f"no worker {args.task_id}; refusing to post")
                return 1
            posted = None
            last_err = ""
            # P-19: a publish is not success until the append is VERIFIED
            # durable; retry once (idempotent via dedup window). Never let a
            # transient failure masquerade as a posted message.
            for _attempt in range(2):
                try:
                    posted = SUP.post_message(
                        args.task_id,
                        getattr(args, "sender", "supervisor") or "supervisor",
                        getattr(args, "receiver", "worker") or "worker",
                        args.text, kind=getattr(args, "kind", "message") or "message",
                        thread_id=getattr(args, "thread", None),
                        reply_to=getattr(args, "reply_to", None))
                except Exception as exc:  # noqa: BLE001
                    last_err = f"{type(exc).__name__}: {exc}"
                    posted = None
                if (posted is not None
                        and SUP.message_present(
                            args.task_id, posted.get("id") or "")):
                    break
                last_err = "append not durable"
                time.sleep(0.1)
            if (posted is None or not SUP.message_present(
                    args.task_id, posted.get("id") or "")):
                print(f"message NOT posted to {args.task_id} inbox "
                      f"(durability check failed: {last_err})")
                return 1
            print(f"message posted to {args.task_id} inbox")
            return 0
        if action == "ack":
            ok = SUP.ack_message(args.task_id, args.msg_id, by=getattr(args, "by", "worker"))
            print(("acknowledged" if ok else "message not found") + f" {args.msg_id}")
            return 0 if ok else 1
        if action == "inbox":
            for m in SUP.list_messages(args.task_id,
                                       status=getattr(args, "status", None)):
                print(f"  [{m.get('status')}] {m.get('sender')} -> "
                      f"{m.get('receiver')} kind={m.get('kind','message')} "
                      f"thread={m.get('thread_id','')} reply_to={m.get('reply_to','')}: "
                      f"{m.get('message','')[:80]}")
            return 0
        if action == "state":
            state = SUP.load_worker(args.task_id)
            if not state:
                print(f"no worker {args.task_id}")
                return 1
            print(json.dumps(state, indent=2, default=str))
            print("\n--- lineage ---")
            print(json.dumps(SUP.worker_lineage(args.task_id), indent=2, default=str))
            return 0
        if action == "state-write":
            try:
                patch_json = json.loads(args.json)
                expect = getattr(args, "expect_seq", None)
            except json.JSONDecodeError as exc:
                print(f"state-write: invalid --json: {exc}")
                return 1
            ok, msg, _st = SUP.apply_worker_state(
                args.task_id, patch_json, expect_seq=expect)
            print(f"state-write: {msg}")
            return 0 if ok else 1
        if action == "lineage":
            print(json.dumps(SUP.worker_lineage(args.task_id), indent=2, default=str))
            return 0
        if action == "campaign-create":
            try:
                roles = json.loads(args.roles)
            except json.JSONDecodeError as exc:
                print(f"campaign create: invalid --roles JSON: {exc}")
                return 1
            path, camp = SUP.create_campaign(args.campaign_id, args.objective,
                                             roles=roles)
            print(f"campaign created: {path}")
            print(json.dumps(camp, indent=2, default=str))
            return 0
        if action == "campaign-assign":
            ok = SUP.assign_role(args.campaign_id, args.role, args.task)
            print(("assigned" if ok else f"assign failed (campaign {args.campaign_id} or role {args.role} missing?)"))
            return 0 if ok else 1
        if action == "campaign-outcome":
            w = SUP.note_worker_outcome(args.campaign_id, args.task,
                                        args.outcome, evidence=args.evidence or None)
            print(json.dumps(w, indent=2, default=str))
            return 0
        if action == "campaign-satisfy":
            ok = SUP.mark_role_evidence(args.campaign_id, args.role, args.task,
                                        args.evidence)
            print(("role satisfied" if ok else "role evidence rejected (missing required evidence)"))
            return 0 if ok else 1
        if action == "campaign-reconcile":
            ok, msg, camp = SUP.reconcile_worker_failure(
                args.campaign_id, args.task,
                findings_preserved=bool(getattr(args, "findings_preserved", True)),
                responsible_covered=bool(getattr(args, "responsible_covered", False)),
                responsibility_transferred_to=getattr(args, "transfer_to", ""),
                adversarial_role_satisfied=bool(getattr(args, "adversarial_satisfied", False)),
                completion_criteria_ok=bool(getattr(args, "completion_ok", False)),
                evidence=args.evidence or None,
                note=getattr(args, "note", ""))
            print(msg)
            print(json.dumps(SUP.campaign_status(args.campaign_id), indent=2, default=str))
            return 0 if ok else 1
        if action == "campaign-replace":
            budget = None
            if getattr(args, "budget_max_turns", None):
                budget = dict(SUP.DEFAULT_BUDGET)
                budget["max_worker_turns"] = int(args.budget_max_turns)
            ok, msg, new_id = SUP.spawn_replacement(
                args.campaign_id, args.role, args.task,
                budget=budget, replace_id=getattr(args, "replace_id", None))
            print(msg)
            return 0 if ok else 1
        if action == "campaign-transfer":
            ok, msg = SUP.adopt_or_transfer(args.campaign_id, args.task,
                                            to_task=args.to)
            print(msg)
            return 0 if ok else 1
        if action == "campaign-status":
            rs = SUP.campaign_status(args.campaign_id)
            print(json.dumps(rs, indent=2, default=str))
            return 0 if rs.get("status") != "MISSING" else 1
        if action == "mission-create":
            phases = json.loads(getattr(args, "phases", "[]") or "[]")
            req = json.loads(getattr(args, "requirements", "[]") or "[]")
            path, m = SUP.create_mission(args.mission_id, args.objective,
                                         phases=phases, requirements=req,
                                         workdir=getattr(args, "workdir", "") or "")
            if m is None:
                print(f"mission create REFUSED: {args.mission_id} exists but is corrupt "
                      "(unparseable JSON); refusing to overwrite — resolve or remove "
                      f"{path} first")
                return 1
            print(f"mission created: {path}")
            print(json.dumps(SUP.mission_status(args.mission_id), indent=2, default=str))
            return 0
        if action == "mission-list":
            base = SUP.missions_dir()
            out = []
            for f in sorted(base.glob("*.json")):
                m = SUP.load_mission(f.stem)
                if m:
                    out.append({"mission_id": f.stem, "status": m.get("status"),
                                "objective": (m.get("objective") or "")[:60],
                                "n_phases": len(m.get("phases", []))})
            print(json.dumps(out, indent=2, default=str))
            return 0
        if action == "mission-status":
            rs = SUP.mission_status(args.mission_id)
            print(json.dumps(rs, indent=2, default=str))
            return 0 if rs.get("status") != "MISSION_MISSING" else 1
        if action == "mission-next":
            ph = SUP.next_phase(args.mission_id)
            print(json.dumps(ph, indent=2, default=str) if ph else
                  "NO_PENDING_PHASE")
            return 0
        if action == "mission-phase-complete":
            ok, msg = SUP.phase_complete(
                args.mission_id, args.phase_id,
                worker_by=getattr(args, "worker_by", "") or "",
                evidence=list(getattr(args, "evidence", []) or []),
                workdir=getattr(args, "workdir", "") or None)
            print(msg)
            return 0 if ok else 1
        if action == "mission-phase-blocked":
            ok, msg = SUP.phase_blocked(
                args.mission_id, args.phase_id,
                worker_by=getattr(args, "worker_by", "") or "",
                note=getattr(args, "note", "") or "")
            print(msg)
            return 0 if ok else 1
        if action == "mission-phase-failed":
            ok, msg = SUP.phase_failed(
                args.mission_id, args.phase_id,
                worker_by=getattr(args, "worker_by", "") or "",
                note=getattr(args, "note", "") or "")
            print(msg)
            return 0 if ok else 1
        if action == "mission-phase-add":
            # "evidence" entries may be literal keywords or structured JSON
            # requirement dicts (semantic model, s5): parse each independently.
            def _as_requirement(v):
                if isinstance(v, str) and v.lstrip().startswith("{"):
                    try:
                        return json.loads(v)
                    except Exception:
                        return v
                return v

            ok, msg = SUP.phase_add(
                args.mission_id, args.phase_id, args.task,
                after=getattr(args, "after", "") or "",
                required_evidence=[_as_requirement(v) for v in
                                   (getattr(args, "evidence", []) or [])])
            print(msg)
            return 0 if ok else 1
        if action == "mission-phase-list":
            phases = SUP.mission_phases(args.mission_id)
            if phases is None:
                print(f"no mission {args.mission_id}")
                return 1
            for p in phases:
                print(f"[{p.get('status') or '?'}] {p.get('phase_id')} "
                      f"after={p.get('after') or '-'} "
                      f"evidence={','.join(str(x) for x in (p.get('required_evidence') or [])) or '-'}: "
                      f"{(p.get('task') or '')[:100]}")
            return 0
        if action == "mission-finding":
            ok = SUP.add_finding(args.mission_id, args.finding_id,
                                 getattr(args, "text", "") or "")
            print(("finding recorded" if ok else "finding exists/mission missing"))
            return 0 if ok else 1
        if action == "mission-finding-resolve":
            ok = SUP.resolve_finding(args.mission_id, args.finding_id,
                                     evidence=list(getattr(args, "evidence", []) or []))
            print(("finding resolved" if ok else "resolve rejected (missing evidence?)"))
            return 0 if ok else 1
        if action == "mission-finding-block":
            ok = SUP.block_finding(args.mission_id, args.finding_id,
                                   note=getattr(args, "note", "") or "")
            print(("finding blocked" if ok else "finding not found"))
            return 0 if ok else 1
        if action == "mission-discover":
            from hermes_cli import mission_ops as MO
            m = SUP.load_mission(args.mission_id)
            if m is None:
                print(f"no mission {args.mission_id}")
                return 1
            d = MO.add_discovery(
                m, title=getattr(args, "title", "") or "untitled",
                rationale=getattr(args, "why", "") or "",
                discoverer=getattr(args, "discoverer", "agent") or "agent",
                evidence=list(getattr(args, "evidence", []) or []),
                priority=int(getattr(args, "priority", 3) or 3),
                deps=((getattr(args, "deps", "") or "").split(",") if getattr(args, "deps", "") else []),
                supersedes=getattr(args, "supersedes", "") or "")
            SUP.save_mission(m)
            print(f"discovery recorded: {d['id']} ({d['status']})"
                  + (f"; superseded {getattr(args,'supersedes','')}"
                     if getattr(args, "supersedes", "") else ""))
            changed, note = MO.plan_discoveries(m)
            SUP.save_mission(m)
            print(f"planner: {changed} promoted/deferred [{note}]")
            return 0
        if action == "mission-plan":
            from hermes_cli import mission_ops as MO
            m = SUP.load_mission(args.mission_id)
            if m is None:
                print(f"no mission {args.mission_id}")
                return 1
            changed, note = MO.plan_discoveries(m)
            SUP.save_mission(m)
            print(f"planner: {changed} changed [{note}]")
            return 0
        if action == "mission-todo":
            from hermes_cli import mission_ops as MO
            m = SUP.load_mission(args.mission_id)
            if m is None:
                print(f"no mission {args.mission_id}")
                return 1
            for d in MO.open_blocking_discoveries(m):
                print(f"  {d['id']:34s} {d['status']:10s} p{int(d.get('priority',5))} : {d.get('title','')[:50]} : {d.get('discoverer','')}")
            return 0
        if action == "mission-evaluate":
            from hermes_cli import mission_ops as MO
            m = SUP.load_mission(args.mission_id)
            if m is None:
                print(f"no mission {args.mission_id}")
                return 1
            ev = MO.completion_evaluator(m)
            print(("COMPLETE" if ev["complete"] else "NOT_COMPLETE") + ": " + ev["reason"])
            return 0 if ev["complete"] else 1
        if action == "mission-dod":
            from hermes_cli import mission_ops as MO
            m = SUP.load_mission(args.mission_id)
            if m is None:
                print(f"no mission {args.mission_id}")
                return 1
            if getattr(args, "install", False) and not m.get("dod"):
                m["dod"] = MO.make_dod()
            if getattr(args, "derive", ""):
                added = MO.install_derived_dod(m, workdir=args.derive)
                print(f"derived DoD: {added} project-outcome dimension(s) "
                      f"installed")
            if getattr(args, "satisfy", False):
                MO.dod_satisfy(m, getattr(args, "dim", "") or "",
                               getattr(args, "what", "") or "",
                               who=getattr(args, "who", "supervisor") or "supervisor")
            SUP.save_mission(m)
            for it in m.get("dod", []):
                print(f"  {it['id']:16s} {it.get('status',''):10s} ev={len(it.get('evidence',[]))}")
            return 0
        if action == "mission-lesson":
            from hermes_cli import mission_ops as MO
            if getattr(args, "get", False):
                rows = MO.recall_lessons(args.mission_id,
                                         apply_to=getattr(args, "scope", "") or "")
                for row in rows:
                    print(f"  {row.get('lesson','')[:120]}")
                return 0
            MO.add_lesson(mission_id=args.mission_id,
                          lesson=getattr(args, "text", "") or "",
                          context=getattr(args, "context", "") or "",
                          better_approach=getattr(args, "better", "") or "",
                          apply_to=getattr(args, "scope", "") or "")
            print("lesson recorded")
            return 0
        if action == "mission-env":
            from hermes_cli import mission_ops as MO
            print(json.dumps(MO.environment_snapshot(), indent=2))
            return 0
        if action == "mission-forensics":
            from hermes_cli import mission_ops as MO
            print(json.dumps(MO.mission_forensics(args.mission_id), indent=2, default=str))
            return 0
        if action == "mission-events":
            from hermes_cli import mission_ops as MO
            events = MO.read_events(args.mission_id,
                                    min_level=getattr(args, "level", "LOW") or "LOW",
                                    limit=int(getattr(args, "limit", 50) or 50))
            if not events:
                print("no events logged")
                return 0
            for e in events:
                ts = time.strftime("%H:%M:%S", time.localtime(float(e.get("ts", 0))))
                print(f"[{e.get('level','LOW'):6s}] {ts} {e.get('kind','')}: {e.get('message','')[:100]}")
            return 0
        if action == "mission-lessons":
            from hermes_cli import mission_ops as MO
            from hermes_cli import supervisor as _S
            m = _S.load_mission(args.mission_id)
            if m is None:
                print(f"no mission {args.mission_id}")
                return 1
            if getattr(args, "sync", False):
                n, why = MO.mission_lesson_sync(m)
                print(f"synced {n} lessons [{why[:160]}]")
                return 0
            out = MO.retrieval_hint(m.get("objective", ""))
            print(out if out else "(no lessons retrieved for this objective)")
            return 0
        if action == "mission-loop":
            return _run_mission_loop(args, args.mission_id)
        if action == "mission-controller":
            from hermes_cli import controller as CTRL
            mid = args.mission_id
            if getattr(args, "arm", False):
                res = CTRL.arm(mid, level=getattr(args, "level", "MEDIUM"),
                               interval=getattr(args, "interval", "every 5m"),
                               deliver=getattr(args, "deliver", "local"),
                               policy=getattr(args, "policy", ""))
                print(json.dumps(res, indent=2, default=str))
                return 0 if "error" not in res else 1
            if getattr(args, "disarm", False):
                print(json.dumps(CTRL.disarm(mid), indent=2))
                return 0
            if getattr(args, "ack", False):
                print(json.dumps(CTRL.ack(mid, who="cli"), indent=2))
                return 0
            if getattr(args, "wake", False):
                out = CTRL.wake_output(mid, level=getattr(args, "level", "MEDIUM"))
                print(out, end="")
                return 0
            if getattr(args, "wait", False):
                res = CTRL.wait(mid, timeout=float(getattr(args, "timeout", 30) or 30),
                                level=getattr(args, "level", "MEDIUM"))
                if res.get("timeout"):
                    print("timeout: no new event in window")
                    return 2
                for e in res.get("events", []):
                    ts = time.strftime("%H:%M:%S", time.localtime(float(e.get("ts", 0))))
                    print(f"[{e.get('level','LOW'):6s}] {ts} {e.get('kind','')}: {e.get('message','')[:120]}")
                return 0
            print(json.dumps(CTRL.status(mid), indent=2, default=str))
            return 0
        if action == "mission-lease":
            print(json.dumps(SUP.lease_state(args.mission_id), indent=2,
                             default=str))
            return 0
        if action == "mission-metrics":
            from hermes_cli import mission_ops as MO
            print(json.dumps(MO.metrics_report(args.mission_id), indent=2,
                             default=str))
            return 0
        if action == "mission-rationale":
            from hermes_cli import mission_ops as MO
            m = SUP.load_mission(args.mission_id)
            if m is None:
                print(f"no mission {args.mission_id}")
                return 1
            print(json.dumps(MO.stop_rationale(m), indent=2, default=str))
            return 0
        if action == "mission-optimize":
            from hermes_cli import mission_ops as MO
            if getattr(args, "apply", False):
                n, why = MO.apply_optimizations(args.mission_id)
                m = SUP.load_mission(args.mission_id)
                if m is not None:
                    changed, _n = MO.plan_discoveries(m)
                    SUP.save_mission(m)
                    print(f"applied: {n} optimization discovery(ies) "
                          f"[plan promoted {changed}]")
                else:
                    print(f"applied: {n} {why}")
                return 0
            print(json.dumps(MO.optimization_plan(args.mission_id), indent=2,
                             default=str))
            return 0
        if action == "mission-retrospective":
            from hermes_cli import mission_ops as MO
            print(json.dumps(MO.retrospective(args.mission_id), indent=2,
                             default=str))
            return 0
        if action == "mission-backlog":
            from hermes_cli import mission_ops as MO
            m = SUP.load_mission(args.mission_id)
            if m is None:
                print(f"no mission {args.mission_id}")
                return 1
            if getattr(args, "add", ""):
                it = MO.backlog_add(
                    m, item_id=args.add, title=getattr(args, "title", ""),
                    priority=int(getattr(args, "priority", 3) or 3),
                    why=getattr(args, "why", ""))
                SUP.save_mission(m)
                print(f"backlog {args.mission_id}: added {it['id']} "
                      f"(p{it['priority']}) [{it['status']}]")
                return 0
            if getattr(args, "materialize", False):
                created, notes = MO.backlog_materialize(m)
                if created:
                    changed, _n = MO.plan_discoveries(m)
                    SUP.save_mission(m)
                    print(f"backlog {args.mission_id}: materialized {created} "
                          f"[{notes}] planner {changed}")
                else:
                    open_bl = MO.backlog_open(m)
                    if open_bl:
                        print(f"backlog {args.mission_id}: "
                              f"{len(open_bl)} open item(s) already covered "
                              f"by discoveries [{notes}]")
                    else:
                        print(f"backlog {args.mission_id}: no open items")
                return 0
            bl = MO.backlog_status(m)
            print(f"backlog {args.mission_id}: total={bl['total']} "
                  f"open={bl['open']}")
            for it in m.get("backlog", []):
                print(f"  {it['id']:28s} {it['status']:12s} "
                      f"p{int(it.get('priority', 5))} "
                      f": {it.get('title', '')[:44]}")
            return 0
        if action == "mission-telemetry":
            from hermes_cli import mission_ops as MO
            print(json.dumps(MO.telemetry_report(args.mission_id), indent=2,
                             default=str))
            return 0
        if action == "mission-gate":
            from hermes_cli import mission_ops as MO
            ck = MO.continue_check_via_cli(args.mission_id)
            if "error" in ck:
                print(ck["error"])
                return 0
            print(json.dumps(ck, indent=2, default=str))
            return 0 if ck.get("may_stop") else 1
        if action == "mission-detect":
            from hermes_cli import mission_ops as MO
            n, why = MO.auto_apply_detections(
                args.mission_id, max_new=int(getattr(args, "max_new", 4) or 4))
            print(f"detect: {n} discovery(ies) recorded "
                  f"[{why[:160] or 'none'}]")
            return 0
        if action == "mission-explore":
            from hermes_cli import mission_ops as MO
            m = SUP.load_mission(args.mission_id)
            if m is None:
                print(f"no mission {args.mission_id}")
                return 1
            variants = [v.strip() for v in
                        (getattr(args, "variants", "") or "").split(";")
                        if v.strip()]
            if not variants:
                print("variants required")
                return 1
            res = MO.add_exploration(
                m, topic=getattr(args, "topic", "") or "",
                variants=variants,
                max_variants=int(getattr(args, "max_variants", 3) or 3))
            SUP.save_mission(m)
            print(f"exploration {res['group']}: {len(res['variants'])} "
                  f"variants registered (planner promotes them)")
            changed, note = MO.plan_discoveries(m)
            SUP.save_mission(m)
            print(f"planner: {changed} promoted [{note[:120]}]")
            return 0
        if action == "mission-benchmark":
            from hermes_cli import mission_ops as MO
            m = SUP.load_mission(args.mission_id)
            if m is None:
                print(f"no mission {args.mission_id}")
                return 1
            if getattr(args, "list", False) or not getattr(args, "name", ""):
                print(json.dumps(MO.list_benchmarks(m), indent=2, default=str))
                return 0
            MO.note_benchmark(m, name=getattr(args, "name", ""),
                              value=getattr(args, "value", ""),
                              provenance=getattr(args, "provenance", ""))
            SUP.save_mission(m)
            print(f"benchmark recorded: {getattr(args, 'name', '')} = "
                  f"{getattr(args, 'value', '')}")
            return 0
        if action == "mission-knowledge":
            from hermes_cli import mission_ops as MO
            subject = getattr(args, "subject", "") or ""
            m = SUP.load_mission(args.mission_id)
            if m is None:
                print(f"no mission {args.mission_id}")
                return 1
            subject = subject or (m.get("objective") or "")
            print(json.dumps(MO.institutional_knowledge(
                subject, mission_id=args.mission_id), indent=2, default=str))
            return 0
        if action == "mission-attack":
            from hermes_cli import mission_ops as MO
            m = SUP.load_mission(args.mission_id)
            if m is None:
                print(f"no mission {args.mission_id}")
                return 1
            d = MO.discover_attack(m, module=getattr(args, "module", "") or "")
            SUP.save_mission(m)
            changed, note = MO.plan_discoveries(m)
            SUP.save_mission(m)
            print(f"attack discovery: {d['id']} [{d['status']}]; "
                  f"planner {changed} [{note[:120]}]")
            return 0
        if action == "list":
            base = SUP._tasks_dir()
            results = []
            for d in sorted(base.iterdir()) if base.is_dir() else []:
                st = SUP.load_worker(d.name)
                if st:
                    results.append({"task_id": st.get("task_id"), "status": st.get("status"),
                                    "phase": st.get("phase")})
            print(json.dumps(results, indent=2))
            return 0
        if action == "reap":
            max_age = float(getattr(args, "max_age_days", 7.0) or 7.0) * 86400
            dry = getattr(args, "dry_run", False)
            if dry:
                result = SUP.reap_stale_workers(max_age_seconds=max_age,
                                                 dry_run=True)
                print(f"[dry-run] would reap: {result['reaped']}")
                print(f"[dry-run] skipped: {result['skipped']}")
            else:
                result = SUP.reap_stale_workers(max_age_seconds=max_age)
                print(f"reaped: {result['reaped']}")
                print(f"skipped: {result['skipped']}")
            return 0
    except Exception as exc:  # noqa: BLE001
        print(f"supervise {action} error: {type(exc).__name__}: {exc}")
        return 1
    print(f"unknown action: {action}")
    return 1