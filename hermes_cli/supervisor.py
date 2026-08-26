"""Hermes Supervisor — autonomous worker lifecycle, monitoring, and control.

Builds on existing Hermes primitives (subprocess `hermes chat -q`, cron,
sessions, state.db) — no new agent framework. A worker is a one-shot Hermes
session with a structured, self-contained task brief; its progress is
persisted as a JSON state file that the supervisor reads to decide whether to
let it continue, intervene, or cancel.

Design constraints (respect existing Hermes architecture):
- Worker = `hermes chat -q "<structured brief>"` subprocess. The brief makes
  the worker loop autonomously: investigate → diagnose → implement → test →
  verify → fix → verify → ... until complete or stuck.
- State lives in ~/.hermes_supervisor/tasks/<task_id>/worker.json (written by
  the worker itself via its own file tool or via a reporter script) so the
  supervisor is a separate process and can survive worker death.
- Supervisor = this module + a CLI/loop: reads worker.json, applies the
  verdict rules, writes a structured command file the worker picks up.
- No changes to conversation_loop, gateway, or plugins. No SOUL-3.0 changes.

State vocabulary (Part 2):
    CREATED, INVESTIGATING, DIAGNOSING, PLANNING, IMPLEMENTING,
    TESTING, VERIFYING, BLOCKED, FAILED, COMPLETE, CANCELLED

Supervisor verdicts (Part 4):
    NO_PROGRESS, REPEATED_FAILURE, REPEATED_HYPOTHESIS, TEST_FAILURE,
    UNVERIFIED_COMPLETION, BLOCKED, WORKER_CRASH, WORKER_TIMEOUT, SUCCESS

Commands (Part 7):
    START, CONTINUE, INVESTIGATE, RETRY, REASSESS, VERIFY, CANCEL
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_cli.diagnostics import (
    render_worker_status,
    validate_worker_completion,
    worker_status_valid,
)

# ---------------------------------------------------------------------------
# Vocabulary / limits
# ---------------------------------------------------------------------------

WORKER_STATES = (
    "CREATED", "STARTING", "INVESTIGATING", "DIAGNOSING", "PLANNING", "WORKING",
    "IMPLEMENTING", "TESTING", "VERIFYING", "RUNNING",  # P-14: workers legitimately write RUNNING (long tool phases)
    "BLOCKED", "WAITING",
    "WAITING_FOR_WORKER", "NEEDS_INPUT", "IDLE",
    "FAILED", "COMPLETE", "CANCELLED",
)

# P6: explicit campaign-level outcome vocabulary. WORKER_RECONCILED is the
# machine-verifiable record that a FAILED worker's responsibility was covered
# elsewhere (findings preserved + responsibility transferred/covered +
# adversarial role still enforced). It is NOT "COMPLETE" — the worker itself
# failed; the *failure* was reconciled.
WORKER_OUTCOMES = (
    "WORKER_COMPLETE", "WORKER_FAILED", "WORKER_CRASHED",
    "WORKER_RETRYING", "WORKER_FAILURE_RECONCILED",
)
CAMPAIGN_STATUSES = ("ACTIVE", "CAMPAIGN_COMPLETE", "CAMPAIGN_FAILED",
                     "CAMPAIGN_BLOCKED")

SUPERVISOR_VERDICTS = (
    "NO_PROGRESS", "REPEATED_FAILURE", "REPEATED_HYPOTHESIS", "TEST_FAILURE",
    "UNVERIFIED_COMPLETION", "BLOCKED", "WAITING", "NEEDS_INPUT",
    "WAIT_TIMEOUT", "WORKER_CRASH", "WORKER_TIMEOUT", "STALL",
    "SUCCESS", "NO_VERDICT_YET", "WORKER_FAILURE",
)

COMMANDS = ("START", "CONTINUE", "INVESTIGATE", "RETRY", "REASSESS", "VERIFY",
            "HOLD", "CANCEL", "DONE")

DEFAULT_BUDGET = {
    "max_worker_turns": 60,
    "max_runtime_seconds": 3600,
    "max_consecutive_failures": 3,
    "max_repeated_hypothesis": 3,
    "max_supervisor_interventions": 6,
    "idle_timeout_seconds": 600,
}

# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------

@dataclass
class WorkerState:
    task_id: str
    worker_pid: int = 0
    status: str = "CREATED"
    phase: str = ""
    task: str = ""
    brief_file: str = ""
    progress: str = ""
    hypothesis: str = ""
    tests_executed: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    files_changed: List[str] = field(default_factory=list)
    verification: str = ""
    blockers: List[str] = field(default_factory=list)
    completion_evidence: List[str] = field(default_factory=list)
    last_activity_at: float = 0.0
    created_at: float = field(default_factory=time.time)
    budget: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_BUDGET))
    failure_timestamps: List[float] = field(default_factory=list)
    hypotheses_seen: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    # P5: versioned ledger (stale-write guard), handoff contract, identity
    seq: int = 1
    findings: List[str] = field(default_factory=list)
    next_action: str = ""
    handoff: Dict[str, Any] = field(default_factory=dict)
    handoffs: List[Dict[str, Any]] = field(default_factory=list)
    worker_identity: str = ""
    run_id: str = ""
    last_acked_msg_id: str = ""
    # P-26 s2 (2026-08-18, live-fix re-applied): workdir persisted so the
    # supervisor can harvest uncommitted evidence from a crashed/exhausted
    # worker's worktree.
    workdir: str = ""


@dataclass
class WorkerDecision:
    verdict: str
    command: str = "CONTINUE"
    instruction: str = ""
    score: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Persistence helpers (JSON ledger per task)
# ---------------------------------------------------------------------------

def _tasks_dir() -> Path:
    base = os.environ.get("HERMES_SUPERVISOR_DIR") or os.path.expanduser(
        "~/.hermes-supervisor")
    d = Path(base) / "tasks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def worker_path(task_id: str) -> Path:
    return _tasks_dir() / task_id / "worker.json"


def command_path(task_id: str) -> Path:
    return _tasks_dir() / task_id / "command.json"


def save_worker(state: Dict[str, Any], task_id: str) -> Path:
    path = worker_path(task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    # P3: mirror a compact row to SessionDB so a supervisor restart can
    # discover the task through normal Hermes state mechanisms. File ledger
    # remains authoritative; this is best-effort and never raises.
    try:
        if task_id and "persist_supervisor_meta" in globals():
            persist_supervisor_meta(task_id, state)
    except Exception:
        pass
    return path


def load_worker(task_id: str) -> Optional[Dict[str, Any]]:
    p = worker_path(task_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_command(task_id: str, decision: WorkerDecision) -> Path:
    path = command_path(task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision.as_dict(), indent=2), encoding="utf-8")
    # telemetry: supervisor -> worker command issuance
    try:
        telemetry(task_id, "command", {
            "verdict": decision.verdict, "command": decision.command,
            "instruction": decision.instruction[:120]})
    except Exception:
        pass
    return path


def write_command_if_changed(task_id: str, decision: WorkerDecision) -> Path | None:
    """Write the decision ONLY when it differs from what is already on disk.

    P-17 (orchestration idle, live-fix re-applied 2026-08-18): loops wrote
    command.json on EVERY poll tick even when the verdict was an unchanged
    NO_VERDICT_YET/CONTINUE — measured 0.416ms + a filesystem write per tick,
    i.e. ~720 writes per worker-hour at --every 5, the overwhelming majority
    { identical }. Skipping identical writes is worker-observable only as the
    absence of a redundant fsync. 'changed' = verdict, command or instruction
    differs from the last persisted one (score changes alone do not re-write).
    """
    path = command_path(task_id)
    try:
        prev = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        prev = {}
    cur = decision.as_dict()
    if (prev.get("verdict") == cur.get("verdict")
            and prev.get("command") == cur.get("command")
            and (prev.get("instruction") or "") == (cur.get("instruction") or "")):
        return path  # identical decision already on disk: no write
    return write_command(task_id, decision)



# ---------------------------------------------------------------------------
# Inbox ledger (CAO-style): durable queued messages with delivery status
# ---------------------------------------------------------------------------
# Adds two capabilities CAO has and the v1 supervisor lacked:
#   1. Durable messages between supervisor and worker (or workers) with
#      status pending/delivered/failed, persisted per task.
#   2. Idle-gated delivery: a message is only marked delivered once a turn
#      boundary passes (i.e. the worker had a chance to read it), which stops
#      the supervisor from stomping command.json over a busy worker.
# This is deliberately file-based (reuses the existing per-task ledger) — no
# new DB, no server. State stays inspectable by the CLI and the supervisor.

INBOX_PENDING = "pending"
INBOX_DELIVERED = "delivered"
INBOX_FAILED = "failed"

_INBOX_MAX = 500  # ponytail: bounded ledger, prune head when exceeded


def inbox_path(task_id: str) -> Path:
    return _tasks_dir() / task_id / "inbox.jsonl"


@contextmanager
def _inbox_lock(task_id: str):
    """Cross-process mutex for inbox.jsonl.

    post_message APPENDS while the supervisor loop (or another CLI) may be
    doing a full read→rewrite to mark delivered/acked or trim the file. With
    no lock these interleave and lose lines: append lands on an fd whose
    size the rewriter read earlier, then the rewriter truncates+writes its
    read-modify-write result, clobbering the appended line. Measured at
    50-worker fan-out: 95-98/100 messages instead of 100/100.
    flock() is per-open-fd and cross-process — correct here because every
    mutation re-opens the file under this lock. All inbox writers must take
    it; readers may not (list_messages is read-only).
    """
    p = inbox_path(task_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def post_message(task_id: str, sender: str, receiver: str,
                 message: str) -> Dict[str, Any]:
    """Append a durable inbox message. Idempotent-ish; no delivery yet."""
    with _inbox_lock(task_id):
        p = inbox_path(task_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "id": f"{int(time.time()*1000)}",
            "sender": sender,
            "receiver": receiver,
            "message": message,
            "status": INBOX_PENDING,
            "ts": time.time(),
        }
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        _trim_inbox_locked(p)
    return dict(entry)


def _trim_inbox_locked(p: Path) -> None:
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
        if len(lines) > _INBOX_MAX:
            p.write_text("\n".join(lines[-_INBOX_MAX:]) + "\n", encoding="utf-8")
    except Exception:
        pass


def _trim_inbox(task_id: str) -> None:
    p = inbox_path(task_id)
    if not p.exists():
        return
    with _inbox_lock(task_id):
        _trim_inbox_locked(p)


def list_messages(task_id: str, *, status: Optional[str] = None) -> List[Dict[str, Any]]:
    p = inbox_path(task_id)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if status is None or e.get("status") == status:
            out.append(e)
    return out


def deliver_pending(task_id: str, *, clear_after_deliver: bool = False) -> int:
    """Deliver ONE pending inbox message, user-priority first (P2).

    Heartbeat invariants: never dump the backlog — deliver exactly one
    message per idle boundary; re-anchor last_activity so the next delivery
    waits for the next boundary. Preserves durable inbox semantics (the
    message is marked delivered, others stay pending).
    """
    msgs = pending_messages_sorted(task_id)
    if not msgs:
        return 0
    next_msg = msgs[0]
    # Write it as a command (CONTINUE/instruction semantics) so the worker's
    # existing command reader picks it up.
    write_command(task_id, WorkerDecision(
        verdict="SUPERVISOR_MESSAGE",
        command="CONTINUE",
        instruction=next_msg.get("message", ""),
        score=0.5,
    ))
    mark_message(task_id, next_msg["id"], INBOX_DELIVERED)
    if HEARTBEAT_REANCHOR_AFTER_DELIVERY:
        st = load_worker(task_id)
        if st:
            touch_heartbeat(st)
            save_worker(st, task_id)
    return 1


def mark_message(task_id: str, msg_id: str, status: str) -> bool:
    """Best-effort legacy mark. Acquires the inbox lock so a concurrent
    append during the rewrite cannot be lost."""
    with _inbox_lock(task_id):
        p = inbox_path(task_id)
        if not p.exists():
            return False
        lines = p.read_text(encoding="utf-8").splitlines()
        changed = False
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("id") == msg_id:
                e["status"] = status
                lines[i] = json.dumps(e)
                changed = True
                break
        if changed:
            p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return changed


def deliver_all_if_idle(task_id: str, state: Optional[Dict[str, Any]] = None) -> int:
    """Deliver every pending message when the worker is not mid-processing.

    Mimics CAO's status-driven delivery without an event bus: we consider a
    worker 'idle' when its last phase is a terminal/completed phase or it has
    never started, or the last activity is stale.
    """
    if state is None:
        state = load_worker(task_id) or {}
    status = (state.get("status") or "").upper()
    phase = (state.get("phase") or "").upper()
    idle_phases = ("COMPLETE", "BLOCKED", "WAITING", "NEEDS_INPUT",
                   "FAILED", "CANCELLED", "CREATED")
    last = float(state.get("last_activity_at") or 0)
    stale = last and (time.time() - last) > 5.0
    if status in idle_phases or phase in idle_phases or stale:
        return deliver_pending(task_id)
    return 0


# ---------------------------------------------------------------------------
# Worker creation / brief
# ---------------------------------------------------------------------------

def create_worker(task: str, *, task_id: Optional[str] = None,
                  budget: Optional[Dict[str, int]] = None,
                  workdir: Optional[str] = None,
                  extra_brief: str = "") -> Tuple[str, Dict[str, Any]]:
    """Register a worker state file and write the worker brief."""
    task_id = task_id or f"w{int(time.time())}"
    state = WorkerState(
        task_id=task_id,
        status="CREATED",
        task=task,
        budget=budget or dict(DEFAULT_BUDGET),
    )
    # P5: ledger starts at seq 1; every subsequent write must be the exact
    # successor (stale-write protection enforced by state-write/CAS paths).
    state.seq = 1
    state.workdir = workdir or ""
    path = save_worker(asdict(state), task_id)
    st_path = worker_path(task_id)
    cmd_path = command_path(task_id)
    brief = compose_brief(task, task_id=task_id, workdir=workdir,
                          extra=extra_brief,
                          state_path=str(st_path),
                          supervisor_command_path=str(cmd_path))
    brief_path = path.parent / "brief.md"
    brief_path.write_text(brief, encoding="utf-8")
    return task_id, asdict(state)


def compose_brief(task: str, *, task_id: str, workdir: Optional[str] = None,
                  extra: str = "",
                  state_path: Optional[str] = None,
                  supervisor_command_path: Optional[str] = None) -> str:
    """Self-contained brief that makes a `hermes chat -q` session loop
    autonomously: investigate → fix → test → verify → repeat if needed."""
    lines = [
        f"# Worker brief: {task_id}",
        "",
        "You are an autonomous Hermes worker. Complete the task below end-to-end",
        "without asking the user to continue after each turn. Work alone, using",
        "your tools, in a loop until the work is genuinely complete or you are",
        "truly blocked.",
        "",
        "## Task",
        task,
        "",
        "## Autonomous loop (follow exactly)",
        "1. INVESTIGATING — read the project, run the baseline tests/commands.",
        "2. DIAGNOSING — if anything fails and the failure could involve the",
        "   Hermes environment, run diag_env(action='analyze', error=...) or",
        "   `hermes diag analyze <error>`. Record the candidate + evidence.",
        "3. PLANNING — list the concrete changes you will make.",
        "4. IMPLEMENTING — make the smallest correct change.",
        "5. TESTING — run the relevant test/verification commands.",
        "6. VERIFYING — confirm the tests pass and the original failure is gone.",
        "7. If anything failed or you found more problems, repeat from DIAGNOSING.",
        "8. Only when tests pass and behavior is verified: report COMPLETE with",
        "   evidence (tests run + status, files changed, verification proof).",
        "",
        "Rules:",
        "- Never assert 'done' without test/command evidence.",
        "- Do not invent fixes for aspects of the repo you have not observed.",
        "- Keep changes minimal; do not refactor unrelated code unless it is",
        "  needed to satisfy the task.",
        "- Standing policy (2026-08-23): if you can satisfy the task a smarter",
        "  or more convenient way than literally asked, you may do so ONLY if",
        "  your report states the upgrade explicitly ('upgraded X to Y because",
        "  Z'). Never deviate silently. Never take irreversible actions",
        "  (deletes, overwrites, external side effects) beyond the task text;",
        "  stop with NEEDS_INPUT instead.",
        "- If you hit a tool error that smells like Hermes/plugin/MCP/hook",
        "  interference, use the diagnostic graph before guessing.",
        "- Stop with BLOCKED only if the task cannot proceed without a human.\n"
                "  Use WAITING when you depend on another worker/dependency, NEEDS_INPUT\n"
                "  when you need human/supervisor input. Waiting workers are held, not\n"
                "  restarted or killed.\n",
        "",
        "Report each phase on a line starting with the tag `WORKER:` so your",
        "supervisor can follow you (e.g. `WORKER: TESTING 2 failed / 47 passed`).",
    ]
    if workdir:
        lines.extend([
            "",
            "## Working directory",
            "",
            f"Operate in: {workdir}",
            "Run tests, make changes, and commit from this directory. Do not cd away unless necessary.",
        ])
    if state_path:
        lines.extend([
            "",
            "## Supervisor state protocol v5 (mandatory)",
            "",
            "Do NOT write the state file directly with write_file. After each",
            "phase, update it with the CAS command (this enforces stale-write",
            "protection and versioning):",
            "",
            "1. Read the current state file to learn its current `seq` number.",
            "2. Run, with your shell/terminal tool:",
            "     hermes supervise state-write %s --expect-seq <seq> --json '{\"status\": \"<PHASE>\", ...}'" % task_id,
            "   where <seq> is the number you read in step 1. Include the fields:",
            '   status, phase, progress, hypothesis, tests_executed, tests_passed,',
            '   files_changed, findings, blockers, next_action, completion_evidence.',
            "3. If it prints 'stale write rejected', re-read the state and retry",
            "   with the new seq. This guarantees you never overwrite newer state.",
            "4. Also maintain a structured handoff each phase, using fields:",
            '   "findings": [..], "next_action": "..", "blockers": [..].',
            "The supervisor derives your handoff (owner_id, phase, objective,",
            "findings, files_changed, tests, blockers, next_action, evidence,",
            "seq) from these fields in the ledger.",
            "",
            "Waiting semantics (do not fake blockers):",
            "- If you must wait for ANOTHER WORKER's handoff or a dependency,",
            '   set "status": "WAITING_FOR_WORKER" and "awaits_worker": "<task_id>".',
            "   You will not be restarted or killed; when a message is delivered",
            "   to you it will wake you. If you just need a phase boundary pause,",
            '   use "status": "WAITING".',
            "- If you need input from a HUMAN or the SUPERVISOR, set",
            '   "status": "NEEDS_INPUT" and describe the exact decision in',
            '   "blockers". You will be held (not killed) until input arrives.',
            '- Use "blockers": [] when you are actively working; blocked is a',
            "  terminal-ish claim, not a phase name.",
            "",
            "Acknowledgements: when the command file contains a SUPERVISOR_MESSAGE",
            "instruction beginning with [msg:<id> ...], after acting on it write",
            '   "last_acked_msg_id": "<id>" in your next state update so the',
            "ledger can mark the message acknowledged.",
            "Only set status COMPLETE after tests pass AND behavior is verified.",
        ])
    if supervisor_command_path:
        lines.append("")
        lines.append("Before starting a new phase, if a file exists at "
                     + supervisor_command_path + " read it and obey it "
                     "(action CONTINUE/INVESTIGATE/RETRY/REASSESS/VERIFY/CANCEL).")
    if extra:
        lines.append("")
        lines.append("## Supervisor notes")
        lines.append(extra)
    return "\n".join(lines)


def start_worker(task_id: str, *, model: Optional[str] = None,
                 hermes: Optional[str] = None, workdir: Optional[str] = None,
                 log_path: Optional[Path] = None) -> subprocess.Popen:
    """Start the worker brief in a one-shot Hermes session and return the Popen.

    stdout/stderr are written to the task's worker.log (not a pipe) so the
    worker never blocks on an unread pipe and the supervisor can inspect the
    transcript afterwards.
    """
    task_dir = _tasks_dir() / task_id
    brief = task_dir / "brief.md"
    if not brief.exists():
        raise FileNotFoundError(f"worker brief missing for {task_id}")
    exe = hermes or _hermes_bin()
    # hermes chat -q accepts query text; pass the brief body so the worker
    # inherits normal runtime behavior with full tool access. -Q: headless
    # workers are programmatic callers — banner/spinner/tool-preview chrome
    # is dead weight in worker.log (and tokens if anything ever reads it).
    brief_text = brief.read_text(encoding="utf-8")
    cmd = [exe, "chat", "-Q", "-q", brief_text]
    if model:
        cmd += ["-m", model]
    log = log_path or (task_dir / "worker.log")
    fh = open(log, "a", encoding="utf-8")
    env = dict(os.environ)
    if workdir:
        env.setdefault("HERMES_WORKDIR", str(workdir))
    proc = subprocess.Popen(
        cmd,
        cwd=workdir or str(task_dir),
        env=env,
        stdout=fh, stderr=subprocess.STDOUT, text=True,
        # P-14 (mission run 1): text=True default is block-buffered (8KB), so
        # sparse-but-legit API-bound workers flush worker.log in bursts and
        # the watchdog's 60s log-fresh check misfires -> STALL killed two
        # healthy audit workers during the autonomous mission. Line-buffer so
        # token/activity output reaches the log promptly and liveness is real.
        bufsize=1,
        # s6-ownership: the worker becomes its own session/process-group
        # leader (pgid == pid). Previously workers inherited the loop's pgid,
        # so no one could kill "the worker's tree" without killing its
        # siblings and the loop itself. Group-scoped kills are now bounded to
        # this worker and its descendants.
        start_new_session=True,
    )
    try:
        telemetry(task_id, "spawn", {
            "pid": proc.pid, "started_at": time.time(),
            "cmd": cmd[:2], "workdir": str(workdir or task_dir)})
    except Exception:
        pass
    return proc


def _hermes_bin() -> str:
    return os.environ.get("HERMES_BIN", "hermes")


# ---------------------------------------------------------------------------
# Progress / repeat detection (Part 8)
# ---------------------------------------------------------------------------

def detect_repeated_hypothesis(state: Dict[str, Any]) -> Tuple[bool, str]:
    hyps = state.get("hypotheses_seen") or []
    if len(hyps) < 2:
        return False, ""
    last = str(hyps[-1]).strip().lower()
    count = sum(1 for h in hyps if str(h).strip().lower() == last)
    if count >= 3:
        return True, f"same hypothesis repeated {count}x: {last[:80]}"
    return False, ""


def detect_repeated_failure(state: Dict[str, Any]) -> Tuple[bool, str]:
    history = state.get("history") or []
    recent = [h for h in history[-6:] if h.get("phase") in ("TESTING", "VERIFYING") and h.get("tests_failed", 0)]
    if len(recent) >= 3:
        return True, "failing test suite seen repeatedly with no resolution"
    return False, ""


def detect_test_failure(state: Dict[str, Any]) -> Tuple[bool, str]:
    if int(state.get("tests_failed") or 0) > 0:
        return True, f"{state['tests_failed']} test(s) failing"
    return False, ""


def detect_progress(state: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """Meaningful progress = passing tests increased, files changed grew,
    hypothesis changed, verification completed. Raw tool-call count is NOT
    progress (it is not even tracked)."""
    if previous is None:
        return True, "first observation"
    gen = (
        ("tests_passed", previous, state),
        ("files_changed_len", previous, state),
        ("hypothesis", previous, state),
    )
    for key, prev, cur in gen:
        if prev.get(key) != cur.get(key):
            if key == "tests_passed":
                return bool(int(cur.get(key) or 0) >= int(prev.get(key) or 0)), f"{key} changed"
            return True, f"{key} changed"
    # verification transitioned to a completed value
    if (previous.get("verification") or "") and (state.get("verification") or "") != previous.get("verification"):
        return True, "verification updated"
    return False, "no meaningful state change"


# ---------------------------------------------------------------------------
# Supervisor decision (Part 4) — deterministic hot path
# ---------------------------------------------------------------------------

def _evidence_bridge(state: Dict[str, Any]) -> Dict[str, Any]:
    """Map worker-state evidence field onto the diagnostics validator shape."""
    out = dict(state)
    out["status"] = (state.get("status") or "").upper()
    evidence = state.get("completion_evidence") or state.get("evidence") or []
    # P7: workers may write ONE evidence line as a string (state-write JSON
    # convenience). It usually packs multiple distinct signals ("tests: XYZ
    # GREEN, lint: clean"); split on clause separators so the validator sees
    # real lines, never list("sentence") char-split garbage.
    if isinstance(evidence, str):
        import re as _re
        parts = [p.strip() for p in _re.split(r"[,;]|\s{2,}|\n", evidence) if p.strip()]
        evidence = parts or [evidence]
    out["evidence"] = list(evidence)
    out["claimed"] = state.get("status") == "COMPLETE" or bool(state.get("claimed"))
    return out


def evaluate_worker(state: Dict[str, Any], *, now: Optional[float] = None,
                    previous: Optional[Dict[str, Any]] = None,
                    interventions_left: int = 6,
                    pid: Optional[int] = None) -> WorkerDecision:
    """Decide what the supervisor should do about a worker state.

    Deterministic and evidence-driven (no LLM in the hot path).
    """
    budget = state.get("budget") or {}
    status = (state.get("status") or "").upper()
    now = now if now is not None else time.time()

    # Terminal states
    if status == "CANCELLED":
        return WorkerDecision("CANCELLED", "CANCEL", "Worker cancelled.", 0.0)
    if status == "BLOCKED":
        blockers = "; ".join(state.get("blockers") or []) or "blocked"
        # A genuinely blocked worker is NOT restarted and NOT timeout-killed.
        # The supervisor holds and waits for a human decision.
        return WorkerDecision("BLOCKED", "HOLD", f"Worker blocked: {blockers}. Hold — do not restart.", 0.1)
    if status in ("WAITING", "WAITING_FOR_WORKER", "NEEDS_INPUT"):
        # Waiting states are legitimate pauses: waiting on another worker's
        # handoff (WAITING / WAITING_FOR_WORKER) or on human/supervisor input
        # (NEEDS_INPUT). Hold, never timeout-kill or progress-suspect — but
        # bound the wait.
        wait_budget = float(budget.get(
            "max_wait_seconds", DEFAULT_BUDGET.get("max_wait_seconds", 3600)))
        waited = now - float(state.get("last_activity_at")
                             or state.get("created_at") or now)
        pend = pending_messages_sorted(state.get("task_id", ""))
        if status == "NEEDS_INPUT" and pend and pend[0].get("sender") in USER_PRIORITY_SENDERS:
            return WorkerDecision("NEEDS_INPUT", "CONTINUE",
                                  "Input available for the worker; deliver and continue.", 0.35)
        # WAITING_FOR_WORKER: waiting on another worker. If the awaited
        # worker is terminal (COMPLETE/FAILED), a handoff/message should
        # already be pending; if not, poke with REASSESS so the supervisor
        # re-checks dependencies rather than silently holding forever.
        if status == "WAITING_FOR_WORKER":
            awaited = state.get("awaits_worker") or ""
            if awaited:
                dep = load_worker(awaited) if awaited else None
                if dep and dep.get("status") in ("COMPLETE", "FAILED", "CANCELLED"):
                    return WorkerDecision("WAITING_FOR_WORKER", "REASSESS",
                                          f"Awaited worker {awaited} is {dep.get('status')}; "
                                          "reassess dependency handoff.", 0.3)
        if waited > wait_budget:
            return WorkerDecision("WAIT_TIMEOUT", "REASSESS",
                                  f"Worker waited {waited:.0f}s beyond budget; reassess rather than kill.", 0.1)
        return WorkerDecision(status, "HOLD",
                              f"Worker {status}: hold — do not restart, do not timeout-kill.", 0.2)
    if status == "FAILED":
        if interventions_left > 0:
            return WorkerDecision("WORKER_FAILURE", "RETRY",
                                  "Worker reported FAILED. Retry once with fresh diagnosis.", 0.2)
        return WorkerDecision("WORKER_FAILURE", "CANCEL",
                              "Worker failed repeatedly; cancelling.", 0.0)
    ok, problems = validate_worker_completion(_evidence_bridge(state)) if status == "COMPLETE" else (None, [])
    if status == "COMPLETE":
        if ok:
            # P10 (independent attack): a worker that was NEVER started
            # (no started_at, no attempts, no pid) cannot have produced real
            # completion evidence — a forged state-write into a never-run
            # task must not yield SUCCESS. Require some evidence of a run.
            never_started = (
                not (state.get("started_at") or state.get("worker_pid"))
                and attempt_number(state) < 2
            )
            if never_started and (pid is None or pid == 0):
                return WorkerDecision(
                    "UNVERIFIED_COMPLETION", "VERIFY",
                    "Completion claimed by a worker that was never started; "
                    "no process evidence — re-verify before accepting.", 0.3)
            return WorkerDecision("SUCCESS", "DONE", "Worker verified complete.", 1.0)
        # P7 (Campaign-5): a COMPLETE claim was rejected, but the worker
        # process is GONE — VERIFY polling a dead pid repeats forever. Treat
        # it as a bounded retry so a replacement/respawn can re-verify; never
        # convert an unverified COMPLETE to SUCCESS.
        if pid is not None and not _pid_alive(pid):
            if attempts_left(state) > 0:
                return WorkerDecision(
                    "WORKER_CRASH", "RETRY",
                    "Worker died after claiming COMPLETE but evidence was rejected; "
                    + "; ".join(problems) + " — respawn to re-verify.", 0.25)
            return WorkerDecision(
                "UNVERIFIED_COMPLETION", "VERIFY",
                "Completion evidence rejected: " + "; ".join(problems), 0.3)
        return WorkerDecision(
            "UNVERIFIED_COMPLETION", "VERIFY",
            "Completion evidence rejected: " + "; ".join(problems), 0.3)

    # Crash detection: the process died before reaching a terminal state.
    # Requires an explicit pid; without one we fall through to the idle/
    # timeout paths (which also catch a dead worker).
    live = True if pid is None else _pid_alive(pid)
    if not live:
        if attempts_left(state) > 0:
            return WorkerDecision("WORKER_CRASH", "RETRY",
                                  "Worker process died before completion; retrying from the last phase.", 0.25)
        return WorkerDecision("WORKER_CRASH", "CANCEL",
                              "Worker crashed repeatedly; cancelling.", 0.0)

    # CREATED-but-never-started detection: a worker in CREATED with no
    # started_at / worker_pid has never been spawned. The idle/timeout paths
    # below would return NO_PROGRESS forever (write_command_if_changed skips
    # identical commands, so the supervisor spins). Detect it explicitly:
    # if it's been sitting CREATED beyond the idle budget, treat as a crash
    # (attempts remain → RETRY so the loop can respawn it; exhausted → CANCEL).
    if status == "CREATED":
        started_at = float(state.get("started_at") or 0)
        wpid = int(state.get("worker_pid") or 0)
        if not started_at and not wpid:
            age = now - float(state.get("created_at") or now)
            idle_budget = float(budget.get("idle_timeout_seconds",
                                          DEFAULT_BUDGET["idle_timeout_seconds"]))
            if age > idle_budget:
                if attempts_left(state) > 0:
                    return WorkerDecision(
                        "WORKER_CRASH", "RETRY",
                        f"Worker sat CREATED for {age:.0f}s without ever starting; "
                        f"reattempting (last phase).", 0.25)
                return WorkerDecision(
                    "WORKER_CRASH", "CANCEL",
                    f"Worker sat CREATED for {age:.0f}s without ever starting; "
                    "attempts exhausted.", 0.0)
            # Fresh CREATED, within budget: hold
            return WorkerDecision("NO_VERDICT_YET", "CONTINUE",
                                  "Worker created, not yet spawned; waiting.", 0.5)

    if status not in WORKER_STATES:
        return WorkerDecision("WORKER_CRASH", "INVESTIGATE",
                              f"Unknown worker status {status!r}; treat as crash.", 0.05)

    # P-18: fresh-spawn grace in evaluate_worker (mirrors the watchdog's
    # STALL_START_GRACE_SECONDS). A just-started worker with a LIVE pid is
    # legitimately in its first phase (cold model load, first generation,
    # first state write) — its ledger may legitimately not have advanced
    # yet. Judging a live, recently-started worker NO_PROGRESS is a FALSE
    # idle signal: it spams REASSESS and can starve a genuinely working
    # worker of supervision attention. Same constant the watchdog trusts.
    # (P-26 s2 restore: re-ported from the live tree; the p26-era evaluate
    # predated this block.)
    started = float(state.get("started_at") or state.get("created_at") or 0)
    if started and pid is not None and _pid_alive(pid):
        if (now - started) < STALL_START_GRACE_SECONDS:
            return WorkerDecision(
                "NO_VERDICT_YET", "CONTINUE",
                "Worker alive and in fresh-spawn grace; waiting for first "
                "state write.", 0.6)

    # Timeout (P1: attempt-bounded retry)
    # P7: measure runtime from when the worker was actually STARTED, not its
    # ledger creation time. A task that waits CREATED while another worker
    # runs must not be born already "over budget" (Campaign-5 WB false-kill).
    started = float(state.get("started_at") or state.get("created_at") or 0)
    max_runtime = int(budget.get("max_runtime_seconds", DEFAULT_BUDGET["max_runtime_seconds"]))
    if started and (now - started) > max_runtime:
        if attempts_left(state) > 0:
            return WorkerDecision("WORKER_TIMEOUT", "RETRY",
                                  "Exceeded max_runtime_seconds; retrying with a fresh attempt.", 0.1)
        return WorkerDecision("WORKER_TIMEOUT", "CANCEL",
                              "Exceeded max_runtime_seconds; attempts exhausted.", 0.0)

    # Idle / no progress
    last = float(state.get("last_activity_at") or 0)
    idle = int(budget.get("idle_timeout_seconds", DEFAULT_BUDGET["idle_timeout_seconds"]))
    if last and (now - last) > idle:
        return WorkerDecision("NO_PROGRESS", "INVESTIGATE",
                              "Idle beyond timeout; investigate the worker.", 0.15)

    progress, why = detect_progress(state, previous)
    if not progress:
        return WorkerDecision("NO_PROGRESS", "REASSESS",
                              f"No meaningful progress: {why}", 0.2)

    repeated_hyp, hj = detect_repeated_hypothesis(state)
    if repeated_hyp:
        return WorkerDecision("REPEATED_HYPOTHESIS", "REASSESS",
                              f"{hj}. Stop repeating it; inspect an alternative path.", 0.1)

    repeat_fail, rf = detect_repeated_failure(state)
    if repeat_fail:
        return WorkerDecision("REPEATED_FAILURE", "INVESTIGATE",
                              rf, 0.3)

    ok_tests, tf = detect_test_failure(state)
    if ok_tests:
        return WorkerDecision("TEST_FAILURE", "CONTINUE",
                              f"{tf}. Do not report completion; investigate and fix failing tests.", 0.4)

    return WorkerDecision("NO_VERDICT_YET", "CONTINUE",
                          "Worker continues and progresses.", 0.6)


# ---------------------------------------------------------------------------
# Public CLI-facing helpers (interop with diagnostics worker helpers)
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, PermissionError):
        return False
    # A zombie is dead for supervision purposes: it cannot make progress.
    # Linux exposes state in /proc/<pid>/stat field 3 == 'Z'.
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8") as f:
            parts = f.read().split()
        if len(parts) > 2 and parts[2] == "Z":
            return False
    except Exception:
        return False
    return True


def _proc_start_ticks(pid: int) -> Optional[str]:
    """Kernel start time of a process (ticks since boot, /proc/<pid>/stat
    field 22). Used to defeat PID REUSE: a recycled pid has a different start
    time, so 'alive' does not mean 'our worker'."""
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8") as f:
            parts = f.read().split()
        return parts[21] if len(parts) > 21 else None
    except Exception:
        return None


def live_owner_pid(task_id: str) -> Optional[int]:
    """The pid that durably OWNS task_id right now, or None.

    Ownership = the ledger's recorded worker_pid WAS anchored (started_at) AND
    that exact process still exists (not a zombie) AND the OS has not reused
    the pid for a different process (proc_start ticks match). Terminal
    workers never own. This is the durable task-lease check: every spawn path
    must consult it, not merely 'does a ledger row exist'.
    """
    st = load_worker(task_id)
    if not st:
        return None
    if (st.get("status") or "") in ("COMPLETE", "CANCELLED", "FAILED", "VERIFIED"):
        return None
    pid = int(st.get("worker_pid") or 0)
    if not pid or not st.get("started_at"):
        return None
    if not _pid_alive(pid):
        return None
    recorded = st.get("proc_start")
    if recorded:
        now_ticks = _proc_start_ticks(pid)
        if now_ticks and now_ticks != recorded:
            return None  # pid recycled by the OS; not our worker
    return st.get("worker_pid")


@contextmanager
def worker_lock(task_id: str, timeout: float = 10.0):
    """Exclusive per-task advisory lock. ALL ownership decisions (live-owner
    check + spawn + record) run inside this lock so two concurrent spawn
    paths cannot both decide 'no owner' and both spawn. O_CLOEXEC: a spawned
    worker never inherits the lock fd, so the lock releases as soon as the
    supervisor-side decision completes."""
    lock_p = _tasks_dir() / task_id / ".owner.lock"
    lock_p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_p, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"owner lock busy for {task_id}")
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def start_worker_guarded(task_id: str, *, model: Optional[str] = None,
                         force: bool = False,
                         **kw) -> Tuple[int, bool]:
    """Chokepoint spawn: start_worker ONLY when the task has no live owner.

    Returns (pid, spawned):
      pid      = live owner pid when spawned=False; the new pid when True
      spawned = False when an existing live worker already owns the task —
                NO second process is launched (durable single-owner invariant).
                True when a worker was actually started (fresh spawn or
                takeover of a genuinely dead/stale owner).
    force=True bypasses ownership for explicit kill-then-respawn flows.
    """
    with worker_lock(task_id):
        owner = live_owner_pid(task_id)
        if owner and not force:
            return (owner, False)
        proc = start_worker(task_id, model=model, **kw)
        record_spawned_pid(task_id, proc.pid)
        return (proc.pid, True)


def guard_attach(task_id: str, new_pid: int) -> Tuple[bool, str]:
    """Allow (re)anchoring a worker pid ONLY when the task is not already
    live-owned by a different process. Returns (ok, message)."""
    with worker_lock(task_id):
        owner = live_owner_pid(task_id)
        if owner and int(owner) != int(new_pid):
            return (False, f"task {task_id} already live-owned by pid {owner}")
        return (True, "ok")


def cancel_mission(mission_id: str, *, reason: str = "cancelled") -> bool:
    """Transition a mission to MISSION_CANCELLED. Returns False if already terminal."""
    m = load_mission(mission_id)
    if m is None:
        return False
    if m.get("status") not in ("MISSION_ACTIVE", "MISSION_BLOCKED"):
        return False
    m["status"] = "MISSION_CANCELLED"
    m["updated_at"] = time.time()
    m["terminal_rationale"] = {"status": "MISSION_CANCELLED", "reason": reason}
    save_mission(m)
    from hermes_cli.mission_ops import log_event
    log_event(mission_id, "HIGH", "MISSION_CANCELLED", reason)
    return True


def reap_stale_missions(*, max_age_seconds: float = 86400 * 14,
                        idle_seconds: float = 86400 * 3,
                        dry_run: bool = False) -> Dict[str, List[str]]:
    """Reap missions that are stuck in MISSION_ACTIVE with no live workers.

    Safety contract:
      - Only MISSION_ACTIVE missions are eligible
      - Mission must be older than max_age_seconds
      - Mission must have no live workers (all workers terminal or absent)
      - Mission must have no PENDING/ACTIVE phases with live workers

    Returns {"cancelled": [mission_ids], "kept": [mission_ids]}.
    Idempotent: cancelled missions are no longer eligible.
    """
    now = time.time()
    cancelled = []
    kept = []

    missions_base = missions_dir()
    if not missions_base.is_dir():
        return {"cancelled": [], "kept": []}

    for f in sorted(missions_base.glob("*.json")):
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        mid = m.get("mission_id", f.stem)
        if m.get("status") != "MISSION_ACTIVE":
            continue

        age = now - float(m.get("created_at") or now)
        if age < max_age_seconds:
            kept.append(mid)
            continue

        # Check for live workers
        has_live = False
        for p in m.get("phases", []):
            wt = p.get("worker_task", "")
            if wt:
                wpath = _tasks_dir() / wt / "worker.json"
                if wpath.exists():
                    try:
                        w = json.loads(wpath.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if w.get("status") not in ("COMPLETE", "FAILED", "CANCELLED"):
                        has_live = True
                        break
        for d in m.get("discoveries", []):
            wt = d.get("worker_task", "")
            if wt:
                wpath = _tasks_dir() / wt / "worker.json"
                if wpath.exists():
                    try:
                        w = json.loads(wpath.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if w.get("status") not in ("COMPLETE", "FAILED", "CANCELLED"):
                        has_live = True
                        break

        if has_live:
            kept.append(mid)
            continue

        # Stale mission — cancel it
        if dry_run:
            cancelled.append(mid)
        elif cancel_mission(mid, reason=f"stale: no live workers for {age:.0f}s"):
            cancelled.append(mid)

    return {"cancelled": cancelled, "kept": kept}
    write_command(task_id, WorkerDecision("CANCELLED", "CANCEL",
                                          "Supervisor cancelled the task.", 0.0))
    state = load_worker(task_id) or {}
    if state.get("status") not in ("COMPLETE", "CANCELLED"):
        state["status"] = "CANCELLED"
        state["phase"] = "CANCELLED"
        save_worker(state, task_id)


# ---------------------------------------------------------------------------
# State mutators (used by tests + CLI)
# ---------------------------------------------------------------------------

def touch_activity(state: Dict[str, Any]) -> None:
    state["last_activity_at"] = time.time()


def record_phase(state: Dict[str, Any], phase: str) -> None:
    touch_activity(state)
    state["status"] = phase if phase in WORKER_STATES else "INVESTIGATING"
    state["phase"] = phase
    state.setdefault("history", []).append({
        "phase": phase,
        "status": phase,
        "tests_passed": state.get("tests_passed", 0),
        "tests_failed": state.get("tests_failed", 0),
        "ts": time.time(),
    })




def record_tests(state: Dict[str, Any], executed: int, passed: int) -> None:
    state["tests_executed"] = executed
    state["tests_passed"] = passed
    state["tests_failed"] = max(0, executed - passed)
    record_phase(state, "TESTING")


def record_completion(state: Dict[str, Any], evidence: List[str], verification: str = "") -> None:
    state["completion_evidence"] = list(evidence or [])
    state["verification"] = verification or (evidence[0] if evidence else "")
    record_phase(state, "VERIFYING")


def finish_complete(state: Dict[str, Any]) -> None:
    state["status"] = "COMPLETE"
    state["phase"] = "COMPLETE"
    touch_activity(state)
    # telemetry: verified-complete boundary (evidence produced)
    try:
        telemetry(state.get("task_id") or "", "exit", {
            "status": "COMPLETE", "finished_at": time.time(),
            "n_evidence": len(state.get("completion_evidence") or [])})
    except Exception:
        pass


# ===========================================================================
# Supervisor Upgrade block — P1–P4 (from UPSTREAM-OVERLAP.md)
# ===========================================================================
# P1  Kanban-inspired liveness/retry: attempt tracking, last_heartbeat_at,
#     bounded/multiple fresh respawns.
# P2  Heartbeat invariants: idle-only + coalesced delivery, user priority,
#     re-anchor, no-invent-work guard.
# P3  Persistent metadata: supervisor:<id> in SessionDB state_meta.
# P4  Same-session autonomous continuation; deterministic SUCCESS only gate.
# ===========================================================================

SUPERVISOR_META_KEY = "supervisor:{task_id}"

HEARTBEAT_STALE_SECONDS = 300
HEARTBEAT_REANCHOR_AFTER_DELIVERY = True

DEFAULT_BUDGET["max_worker_attempts"] = 3
DEFAULT_BUDGET["max_continuation_turns"] = 40
DEFAULT_BUDGET["max_wait_seconds"] = 3600


def touch_heartbeat(state, when: Optional[float] = None) -> float:
    """Record worker liveness (state['last_heartbeat_at'])."""
    now = when if when is not None else time.time()
    state["last_heartbeat_at"] = now
    if now > float(state.get("last_activity_at") or 0):
        state["last_activity_at"] = now
    return now


def _freshness(state: Dict[str, Any]) -> float:
    """Freshest liveness signal.

    P6: Campaign 4 killed worker B because the watchdog read the stale
    *heartbeat* (302s old) while the worker had written a *newer state*
    (activity 86s old). Liveness must be the MAX of all signals a worker can
    touch, not heartbeat-first. A worker that writes state is alive, even if
    its dedicated heartbeat tick hasn't fired.
    """
    return max(float(state.get("last_heartbeat_at") or 0),
               float(state.get("last_activity_at") or 0))


def liveness_class(state: Dict[str, Any], *, pid: Optional[int] = None,
                   now: Optional[float] = None) -> str:
    """Classify a worker: healthy | stale | timed_out | crashed | finished."""
    now = now if now is not None else time.time()
    status = (state.get("status") or "").upper()
    if status in ("COMPLETE", "CANCELLED"):
        return "finished"
    live = True if pid is None else _pid_alive(pid)
    if not live:
        return "crashed"
    started = float(state.get("created_at") or 0)
    max_runtime = int((state.get("budget") or {}).get(
        "max_runtime_seconds", DEFAULT_BUDGET["max_runtime_seconds"]))
    if started and now - started > max_runtime:
        return "timed_out"
    hb = _freshness(state)
    if hb and now - hb > HEARTBEAT_STALE_SECONDS:
        return "stale"
    return "healthy"


def attempt_number(state: Dict[str, Any]) -> int:
    return int(state.get("attempt") or 1)


def record_attempt(state: Dict[str, Any], *, reason: str, ok: bool = False) -> int:
    """Append an attempt outcome to the ledger."""
    n = attempt_number(state)
    state.setdefault("attempts", []).append({
        "attempt": n, "reason": reason, "ok": bool(ok), "ts": time.time(),
    })
    return n


def bump_attempt(state: Dict[str, Any], reason: str = "retry") -> int:
    """Advance to the next attempt and clear the dead pid."""
    n = attempt_number(state) + 1
    state["attempt"] = n
    state.setdefault("attempts", []).append({
        "attempt": n, "reason": reason, "ok": False, "ts": time.time(),
    })
    state.pop("worker_pid", None)
    # telemetry: retry/failure pattern (bounded by the existing audit cap)
    try:
        telemetry(state.get("task_id") or "", "retry", {
            "attempt": n, "reason": reason})
    except Exception:
        pass
    return n


def attempts_left(state: Dict[str, Any]) -> int:
    """Remaining respawns. max_worker_attempts = total spawns allowed
    (the initial spawn + retries), so attempt 1 has max-1 retries left."""
    budget = state.get("budget") or {}
    max_attempts = int(budget.get(
        "max_worker_attempts", DEFAULT_BUDGET["max_worker_attempts"]))
    return max(0, max_attempts - attempt_number(state))


USER_PRIORITY_SENDERS = ("supervisor", "user", "human")


def _message_priority(m: Dict[str, Any]) -> int:
    sender = (m.get("sender") or "").lower().strip()
    return 0 if sender in USER_PRIORITY_SENDERS else 1


def pending_messages_sorted(task_id: str) -> List[Dict[str, Any]]:
    """Pending inbox messages, user-priority first, then arrival order."""
    msgs = list_messages(task_id, status=INBOX_PENDING)
    msgs.sort(key=lambda m: (_message_priority(m), m.get("ts", 0.0)))
    return msgs


def continuation_gate(state: Dict[str, Any]) -> bool:
    if not bool(state.get("continuation")):
        return False
    if bool(state.get("user_preempted")):
        return False
    budget = state.get("budget") or {}
    used = int(state.get("continuation_turns_used") or 0)
    max_turns = int(budget.get(
        "max_continuation_turns", DEFAULT_BUDGET["max_continuation_turns"]))
    return used < max_turns


def note_user_intervention(state: Dict[str, Any]) -> None:
    """A real user message preempts the continuation (P2/P4)."""
    state["user_preempted"] = True


def record_continuation_turn(state: Dict[str, Any]) -> int:
    n = int(state.get("continuation_turns_used") or 0) + 1
    state["continuation_turns_used"] = n
    return n


def spawn_continuation_command(state: Dict[str, Any]) -> Optional[WorkerDecision]:
    """Return a CONTINUE instruction if continuation is healthy, else None.

    SUCCESS is never produced here — only deterministic evidence can.
    """
    if not continuation_gate(state):
        return None
    record_continuation_turn(state)
    return WorkerDecision(
        verdict="CONTINUE",
        command="CONTINUE",
        instruction=(
            "Continue working autonomously: investigate, fix, test, verify, "
            "and write your state with evidence. If nothing meaningful has "
            "changed, do not invent work. A real user message will preempt you."
        ),
        score=0.4,
    )


def _meta_bridge():
    try:
        from hermes_cli.goals import _get_session_db
        return _get_session_db()
    except Exception:
        return None


def persist_supervisor_meta(task_id: str, state: Dict[str, Any]) -> bool:
    """Persist a compact row under state_meta supervisor:<id>."""
    db = _meta_bridge()
    if db is None:
        return False
    try:
        row = {
            "task_id": task_id,
            "status": state.get("status"),
            "phase": state.get("phase"),
            "attempt": attempt_number(state),
            "updated": time.time(),
        }
        db.set_meta(SUPERVISOR_META_KEY.format(task_id=task_id), json.dumps(row))
        return True
    except Exception:
        return False


def load_supervisor_meta(task_id: str) -> Optional[Dict[str, Any]]:
    db = _meta_bridge()
    if db is None:
        return None
    try:
        raw = db.get_meta(SUPERVISOR_META_KEY.format(task_id=task_id))
    except Exception:
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def recover_or_create_state(task_id: str) -> Optional[Dict[str, Any]]:
    """Recover state on supervisor restart: file ledger first, then meta."""
    st = load_worker(task_id)
    if st:
        return st
    meta = load_supervisor_meta(task_id)
    if meta:
        return {
            "task_id": task_id,
            "status": meta.get("status"),
            "phase": meta.get("phase"),
            "attempt": meta.get("attempt", 1),
        }
    return None


def touch_activity(state: Dict[str, Any]) -> None:
    state["last_activity_at"] = time.time()

# ===========================================================================
# Supervisor P5 block — handoff contract, worker-state semantics, watchdog,
# cross-worker coordination (source-compared against cli-collaboration,
# tasksquad, mcp_agent_mail, agent-conductor, batty, ClawTeam, tmux-a2a-postman:
# see hermes-audit/research-20260809 + skill refs). File ledger stays the
# single authoritative state; no DB/server added.
# ===========================================================================

# --- P5.1 state versioning / stale-write protection --------------------------
# seq is an optimistic-concurrency token on worker.json. Every writer must
# present the exact successor seq (disk seq + 1); the CAS save rejects
# stale writers (old process resuming after a newer write). Worker-side
# writes go through `hermes supervise state-write` so the guard is enforced
# at the real write boundary (workers previously wrote worker.json directly).

AUDIT_MAX = 500  # ponytail: bounded audit trail


def current_seq(state: Dict[str, Any]) -> int:
    return int(state.get("seq") or 0)


def next_seq(state: Dict[str, Any]) -> int:
    return current_seq(state) + 1


def audit_event(task_id: str, kind: str, payload: Dict[str, Any]) -> None:
    """Append one event to audit.jsonl (authoritative evidence trail)."""
    p = _tasks_dir() / task_id / "audit.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": time.time(), "kind": kind, **payload}
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
        if len(lines) > AUDIT_MAX:
            p.write_text("\n".join(lines[-AUDIT_MAX:]) + "\n", encoding="utf-8")
    except Exception:
        pass


def telemetry(task_id: str, kind: str, payload: Dict[str, Any]) -> None:
    """Bounded, durable worker activity telemetry (P-26 r6).

    Reuses the worker's existing audit.jsonl — the same bounded evidence
    trail the supervisor already keeps per task — with a `telemetry` kind
    carrying an inner `t_kind` so analysis can group events without a second
    store. Never raised; never blocks the hot path. This is the seam that
    makes repeated commands / repeated file-churn / retries / overlap
    observable instead of woven into prose.
    """
    audit_event(task_id, "telemetry",
                {"t_kind": kind, "t": time.time(), **payload})


def save_worker_cas(task_id: str, state: Dict[str, Any]) -> Tuple[bool, str]:
    """Save only if the write is the strict successor of the on-disk seq.

    Returns (accepted, reason). A stale writer (worker or supervisor loaded
    an old revision) is rejected and the attempt recorded in audit.jsonl —
    the *other* supervisor decision sees the audit trail.
    """
    disk = load_worker(task_id)
    want = current_seq(state)
    if disk is not None:
        have = current_seq(disk)
        if want != have + 1:
            audit_event(task_id, "stale_write", {
                "attempted_seq": want, "disk_seq": have,
                "writer": state.get("task_id", "?"),
            })
            return False, f"stale write rejected: state seq {want} != disk seq + 1 ({have} + 1)"
        state["seq"] = want
    else:
        state["seq"] = want or 1
    save_worker(state, task_id)
    return True, f"accepted seq {state['seq']}"


def apply_worker_state(task_id: str, patch: Dict[str, Any],
                       expect_seq: Optional[int] = None,
                       *, author: str = "worker",
                       touch: bool = True) -> Tuple[bool, str, Dict[str, Any]]:
    """The worker-facing state write (used by `hermes supervise state update`).

    Reads the ledger, verifies the optimistic token, merges `patch` over it,
    bumps seq, refreshes the handoff block, and saves via CAS. Returns
    (ok, reason, resulting-state). Never deletes supervisor keys.
    """
    disk = load_worker(task_id)
    if disk is None:
        # Distinguish "never existed" (legit fresh task) from "exists but
        # unparseable" (torn/corrupt ledger). Refusing the corrupt case
        # preserves the only copy of prior state for forensics — a state
        # write must never silently overwrite a corrupt ledger with a fresh
        # one (P-26 s2 live-fix re-applied, 2026-08-18).
        _lp = worker_path(task_id)
        if _lp.exists():
            audit_event(task_id, "corrupt_ledger_refused", {
                "path": str(_lp), "author": author,
                "patch_keys": sorted(patch.keys())})
            return False, "refusing state-write: worker ledger exists but is " \
                          "corrupt (unparseable JSON); resolve or remove it first", \
                          {"task_id": task_id, "status": "CORRUPT"}
        disk = {"task_id": task_id, "status": "CREATED", "budget": dict(DEFAULT_BUDGET)}
    if expect_seq is not None and expect_seq != current_seq(disk):
        audit_event(task_id, "stale_write", {
            "expected_seq": expect_seq, "disk_seq": current_seq(disk),
            "author": author, "patch_keys": sorted(patch.keys()),
        })
        return False, ("stale write: expected seq %s but ledger is at %s; "
                       "a newer writer won — re-read and retry"
                       % (expect_seq, current_seq(disk))), disk
    protected = {"seq", "created_at", "attempt", "worker_pid", "history",
                 "attempts", "handoffs", "handoff",
                 "budget", "run_id", "worker_identity", "started_at"}
    for k, v in patch.items():
        if k in protected:
            continue
        disk[k] = v
    disk["seq"] = current_seq(disk) + 1
    disk["author"] = author
    if touch:
        disk["last_activity_at"] = time.time()
    update_handoff(disk)
    ok, msg = save_worker_cas(task_id, disk)
    if ok:
        notify_event("WORKER_STATE", task_id=task_id,
                     payload={"status": disk.get("status"),
                              "seq": disk.get("seq")})
        # telemetry: worker→ledger state transition (this is the highest-freq
        # durable activity signal; bounded by the timestamps below)
        try:
            telemetry(task_id, "state", {
                "status": disk.get("status"), "phase": disk.get("phase"),
                "seq": disk.get("seq"),
                "tests_executed": disk.get("tests_executed") or 0,
                "tests_passed": disk.get("tests_passed") or 0,
                "files_changed": (disk.get("files_changed") or [])[:8],
                "created_at": disk.get("created_at") or 0,
                "updated_at": disk.get("updated_at") or 0})
        except Exception:
            pass
    return ok, msg, disk


# ---------------------------------------------------------------------------
# Worker->supervisor event wakeups (P-22)
# ---------------------------------------------------------------------------
# A worker state transition (state-write / phase_complete / crash detected /
# discovery mutation) is an EVENT, not something the supervisor polls for.
# Delivery is best-effort over a per-supervisor Unix datagram socket; the
# ledger remains the durable source of truth, so a lost/replayed event is
# harmless (the reaction always re-reads durable state -> idempotent).

_EVENT_SOCK_ENV = "HERMES_SUPERVISOR_EVENT_SOCK"
_EVENT_PENDING: List[Dict[str, Any]] = []   # drained when a socket is bound


def _event_receiver_task(task_id: str) -> Dict[str, Any]:
    """Return the event-receiver metadata for a worker: its datagram socket
    path (None when disabled) and the select-able fd. The supervisor binds a
    per-worker socket and ALSO listens for events on it."""
    base = Path(os.environ.get("HERMES_SUPERVISOR_DIR")
                or os.path.expanduser("~/.hermes-supervisor")) / "events"
    base.mkdir(parents=True, exist_ok=True)
    return {"path": base / f"{task_id}.sock"}


def notify_event(kind: str, task_id: str = "", payload: Optional[Dict[str, Any]] = None,
                 sock_path: Optional[str] = None) -> bool:
    """Best-effort worker -> supervisor event. Called from state-write /
    phase-transition / discovery paths. No-op when no socket is listening."""
    try:
        import socket as _s
        path = sock_path or os.environ.get(_EVENT_SOCK_ENV, "")
        if not path:
            return False
        addr = _s.AF_UNIX
        s = _s.socket(addr, _s.SOCK_DGRAM)
        try:
            s.sendto(json.dumps(
                {"kind": kind, "task_id": task_id, "payload": payload or {},
                 "ts": time.time()}, default=str).encode("utf-8"), path)
            # send needs a small grace for the datagram to land before close
            s.close()
            return True
        except OSError:
            try:
                s.close()
            except Exception:
                pass
            return False
    except Exception:
        return False


# ---- handoff contract (cli-collaboration AGENT_HANDOFF shape, JSON-ized) ----

def build_handoff(state: Dict[str, Any]) -> Dict[str, Any]:
    """Machine-readable handoff block embedded in the SAME worker.json ledger
    (no second state system). Mirrors cli-collaboration's handoff fields."""
    return {
        "owner_id": state.get("worker_identity") or state.get("task_id", "?"),
        "phase": state.get("phase") or "",
        "objective": state.get("task") or state.get("objective") or "",
        "status": state.get("status") or "",
        "findings": list(state.get("findings") or []),
        "files_changed": list(state.get("files_changed") or []),
        "tests": {
            "executed": int(state.get("tests_executed") or 0),
            "passed": int(state.get("tests_passed") or 0),
            # failed is always derived: tests_failed may be stale/absent
            "failed": max(0, int(state.get("tests_executed") or 0)
                          - int(state.get("tests_passed") or 0)),
        },
        "blockers": list(state.get("blockers") or []),
        "next_action": state.get("next_action") or "",
        "evidence": list(state.get("completion_evidence") or []),
        "seq": current_seq(state),
        "updated_at": time.time(),
    }


def update_handoff(state: Dict[str, Any]) -> Dict[str, Any]:
    """Refresh state['handoff'] and append a bounded version history."""
    h = build_handoff(state)
    state["handoff"] = h
    hist = state.setdefault("handoffs", [])
    if not hist or hist[-1].get("seq") != h["seq"]:
        hist.append(h)
        if len(hist) > 100:  # ponytail: bounded history, drop oldest
            del hist[:len(hist) - 100]
    return h


def post_handoff(state: Dict[str, Any], *, to_task: str,
                 sender_task: Optional[str] = None) -> Dict[str, Any]:
    """Publish this worker's handoff as a durable inbox message to another
    worker's task ledger. The receiver's supervisor delivers it at its idle
    boundary exactly like any inbox message (no new transport)."""
    h = dict(state.get("handoff") or build_handoff(state))
    h["from_task"] = state.get("task_id") or sender_task or ""
    msg = post_message_verified(to_task,
                                sender=state.get("task_id") or "supervisor",
                                receiver="worker", message=json.dumps(h, default=str),
                                kind="handoff", thread_id=state.get("task_id") or "")
    audit_event(state.get("task_id", to_task), "handoff_posted", {
        "receiver_task": to_task, "msg_id": msg.get("id"),
        "ok": bool(msg.get("id")
                   and message_present(to_task, msg.get("id") or ""))})
    return msg


# ---------------------------------------------------------------------------
# P5 worker lifecycle classification (9 explicit states)
# ---------------------------------------------------------------------------

def worker_class(state: Dict[str, Any], *, pid: Optional[int] = None,
                 now: Optional[float] = None) -> str:
    """Classify a worker into explicit supervision states:

    complete | failed | blocked | waiting | crashed | timed_out |
    idle | working | progressing | active

    `progressing` requires the caller to detect a seq/state delta between
    consecutive observations (the loop does); stateless, a fresh/active
    worker reports `active`. Pid-verifiable classes come from liveness;
    `working` means the pid is alive AND observably busy (CPU/children),
    so a genuinely long-running tool call is never classified idle.
    """
    status = (state.get("status") or "").upper()
    if status == "COMPLETE":
        return "complete"
    if status == "FAILED":
        return "failed"
    if status == "CANCELLED":
        return "cancelled"
    if status == "BLOCKED":
        return "blocked"
    if status in ("WAITING", "NEEDS_INPUT", "WAITING_FOR_WORKER"):
        return "waiting"
    live = True if pid is None else _pid_alive(pid)
    if not live:
        return "crashed"
    now = now if now is not None else time.time()
    started = float(state.get("created_at") or 0)
    max_runtime = int((state.get("budget") or {}).get(
        "max_runtime_seconds", DEFAULT_BUDGET["max_runtime_seconds"]))
    if started and now - started > max_runtime:
        return "timed_out"
    hb = _freshness(state)
    if hb and now - hb > HEARTBEAT_STALE_SECONDS:
        # Process alive but no state/heartbeat writes for a long time.
        # P6: if the process is still busy (CPU/child activity), the worker
        # is mid-tool-call — a long test/benchmark/crawl, NOT hung.
        if pid is not None and _process_is_busy(pid):
            return "working"
        return "idle"
    return "active"


# ---------------------------------------------------------------------------
# P5 watchdog: hang/stall detection + kill escalation
# ---------------------------------------------------------------------------
# tasksquad's daemon hash-check treats "output unchanged every tick" as stuck.
# We use the state ledger's fingerprint across loop iterations: a LIVE pid
# whose state did not change for STALL_MIN_TICKS *and* whose heartbeat is
# stale is hung, not slow. BLOCKED/WAITING/NEEDS_INPUT workers are never
# touched. Kills honor the attempt budget (loop reuses RETRY).

STALL_MIN_TICKS = 2
# ponytail: heartbeat-stale window (300s) doubles as the hang threshold; a
# worker in a phase that legitimately runs >5 minutes without a state write
# could be misflagged as hung. Workers write state per phase, so this holds
# for current briefs; raise HEARTBEAT_STALE_SECONDS if long-silent phases
# become normal.


def state_fingerprint(state: Dict[str, Any]) -> str:
    """Deterministic hash of the worker's reported progress surface."""
    from hashlib import sha256
    surface = (state.get("status"), state.get("phase"), state.get("progress"),
               state.get("hypothesis"), state.get("tests_executed"),
               state.get("tests_passed"), state.get("files_changed"),
               state.get("blockers"), state.get("completion_evidence"),
               state.get("last_heartbeat_at"), state.get("seq"))
    return sha256(json.dumps(surface, default=str).encode()).hexdigest()[:16]


def _proc_stat(pid: int) -> Optional[List[str]]:
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8") as f:
            return f.read().split()
    except Exception:
        return None


def _proc_children(pid: int) -> List[int]:
    """Direct children of pid via /proc/<pid>/task/<pid>/children."""
    try:
        with open(f"/proc/{int(pid)}/task/{int(pid)}/children", encoding="utf-8") as f:
            raw = f.read().strip()
        return [int(x) for x in raw.split() if x.strip()]
    except Exception:
        return []


def _proc_ppid(pid: int) -> Optional[int]:
    """Parent pid of pid. Parses /proc/<pid>/stat after the final ')' because
    comm may contain spaces/parens and a naive split misaligns the fields."""
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8") as f:
            line = f.read()
        idx = line.rfind(")")
        if idx < 0:
            return None
        fields = line[idx + 1:].split()
        if len(fields) < 2:  # state ppid pgrp ...
            return None
        return int(fields[1])
    except Exception:
        return None


def _descendants(pid: int) -> List[int]:
    """Every descendant of pid found by a recursive walk of the /proc children
    tree rooted at pid.

    Reap-window contract (measured on Linux, s6): the walk is valid ONLY
    while pid is alive. The instant it exits, its children are reparented to
    the nearest subreaper (init) and the walk goes empty — a zombie parent's
    children list is empty, reparenting happens at do_exit() time, not at
    waitpid(). Callers MUST reap at the terminal decision while the worker is
    alive. Order is discovery order; callers reap in reverse for a
    deepest-first sweep."""
    out: List[int] = []
    stack = [pid]
    while stack:
        cur = stack.pop()
        for ch in _proc_children(cur):
            out.append(ch)
            stack.append(ch)
    return out


def _is_descendant_of(pid: int, ancestor: int) -> bool:
    """True iff pid's live parent chain passes through ancestor (pid excluded).
    The pid-reuse guard used before every reap kill: a recycled pid whose
    parent chain no longer contains the worker is never touched."""
    seen = set()
    cur = pid
    while cur and cur not in seen:
        if cur == ancestor:
            return True
        seen.add(cur)
        cur = _proc_ppid(cur) or 0
    return False


def reap_worker_processes(worker_pid: int) -> int:
    """Kill every live descendant of worker_pid; leave the worker pid itself
    and everything outside its tree untouched. Returns the number of processes
    signalled.

    s6-ownership: discovery is a /proc children walk rooted at the worker pid.
    It is valid ONLY while the worker is alive (on exit, children are
    reparented and the walk empties — see _descendants), so every call site
    reaps at the terminal decision, i.e. while the worker is still running:
    the loop reacts to the worker's own final state-write event within the
    same tick. Each candidate is re-verified against the live parent chain
    immediately before the kill, so a recycled pid whose ancestry no longer
    contains the worker is left alone. A descendant that leads its own
    process group is killed group-wide (os.killpg); otherwise the single
    pid is killed. The supervisor / mission loops are ancestors or siblings of
    the worker — never descendants — so they survive reap by construction."""
    if not worker_pid:
        return 0
    descendants = _descendants(int(worker_pid))
    signaled = 0
    # deep-to-shallow: children die before parents so nothing below can be
    # re-spawned or rescued mid-sweep.
    for d in reversed(descendants):
        if d == worker_pid or not _pid_alive(d):
            continue
        if not _is_descendant_of(d, worker_pid):
            continue  # pid reused since the walk; not ours anymore
        try:
            if os.getpgid(d) == d:  # group leader: take the whole job
                os.killpg(d, signal.SIGKILL)
            else:
                os.kill(d, signal.SIGKILL)
            signaled += 1
        except (ProcessLookupError, PermissionError):
            pass  # gone mid-sweep (or already killed by a group kill)
    return signaled


def _process_is_busy(pid: int, *, window: float = 2.0) -> bool:
    """A live process that is still consuming CPU or has live children is
    BUSY — it is mid-tool-call (test, benchmark, crawl, browser, network),
    not hung. P6 liveness signal for long-running tool calls that produce no
    state writes. Cheap /proc reads only.

    NOTE (P7, Campaign-5): CPU/children do NOT cover an API-bound model
    generation (hermes waits on the provider socket: ~0 CPU, no children).
    The supervisor therefore ALSO reads worker.log growth — streaming tokens
    and child tool output both append to the same fd, so log mtime is the
    canonical "model is actually producing output" signal.
    """
    if not pid:
        return False
    if not _pid_alive(pid):
        return False
    # children alive = tool subprocess running
    for child in _proc_children(pid):
        if _pid_alive(child):
            return True
    # CPU ticks advancing = process itself is executing
    s1 = _proc_stat(pid)
    if not s1 or len(s1) < 15:
        return False
    try:
        c1 = int(s1[13]) + int(s1[14])  # utime + stime
    except (ValueError, IndexError):
        return False
    time.sleep(min(window, 0.5))
    s2 = _proc_stat(pid)
    if not s2 or len(s2) < 15:
        return False
    try:
        c2 = int(s2[13]) + int(s2[14])
    except (ValueError, IndexError):
        return False
    return c2 > c1


def _log_fresh(task_id: str, log_path: Optional[Path] = None,
               now: Optional[float] = None) -> bool:
    """worker_log shows recent write (model streaming OR tool stdout). A real
    Hermes worker writes continuously while generating; a hung one does not."""
    now = time.time() if now is None else now
    p = Path(log_path) if log_path else (worker_path(task_id).parent / "worker.log")
    try:
        if not p.exists():
            return False
        return (now - float(p.stat().st_mtime)) <= STALL_LOG_FRESH_SECONDS
    except Exception:
        return False


STALL_LOG_FRESH_SECONDS = 60  # ponytail: log touched within 60s == working; raise if phases stream slower
# Fresh spawn grace: a worker that has not yet written its first activity
# (model load, first generation, first phase state-write) MUST NOT be stalled.
# Campaign-5: three consecutive false STALL kills fired while the CREATED-phase
# worker was still in its initial model generation (no state writes yet).
STALL_START_GRACE_SECONDS = 600  # ponytail: 10 min; shrink after provider cold-start data
# Quiet-kill bound: how long a FULLY quiet worker may stay alive before we
# treat it as hung. This is the KILL decision — deliberately far longer than
# a heartbeat staleness label. A model-bound worker can be silent for many
# minutes while still working; 30 minutes of absolute silence (no state
# writes, no CPU, no log growth, no children) with an alive pid is a hang.
# ponytail: 30 min ceiling; tighten when long-phases are proven bounded.
STALL_QUIET_SECONDS = 1800


def watchdog_assess(state: Dict[str, Any], *, pid: Optional[int] = None,
                    fingerprint: str, prev_fingerprint: Optional[str],
                    stall_count: int, now: Optional[float] = None,
                    allow_busy_override: bool = True,
                    log_path: Optional[Path] = None) -> Tuple[str, int]:
    """Return (action, stall_count). action: ok | hold | retry | stall.

    Only when the worker is a live, non-waiting, non-blocked worker whose
    freshness is stale AND whose state fingerprint was unchanged for
    STALL_MIN_TICKS do we escalate to STALL (the loop then kills+restarts
    within the attempt budget). A process that is observably BUSY (CPU or
    live children, via /proc) OR has a growing worker log (model streaming /
    tool output) is never a stall — long-running tool calls and API-bound
    model generation do not update the fingerprint. A worker that has not
    yet begun (started_at within STALL_START_GRACE_SECONDS) is never stalled.
    """
    now = now if now is not None else time.time()
    status = (state.get("status") or "").upper()
    if status in ("BLOCKED", "WAITING", "NEEDS_INPUT", "WAITING_FOR_WORKER",
                  "COMPLETE", "FAILED", "CANCELLED"):
        return "hold", 0
    live = True if pid is None else _pid_alive(pid)
    if not live:
        return "retry", 0  # crash path handled elsewhere; watchdog just notes it
    task_id = state.get("task_id") or ""
    # P7b: fresh-spawn grace. Until started_at + grace elapses, the worker is
    # still initializing (cold model load, first generation, first phase
    # write) — never judge it hung.
    started = float(state.get("started_at") or state.get("last_activity_at")
                    or state.get("created_at") or 0)
    if started and (now - started) < STALL_START_GRACE_SECONDS:
        return "ok", 0
    # P6: a busy process is working, not hung — return ok and reset the counter
    if allow_busy_override and pid is not None and _process_is_busy(pid):
        return "ok", 0
    # P7: log growth = real output (model generation or tool subprocess
    # writing to the same fd) even when CPU/children appear idle.
    if _log_fresh(task_id, log_path=log_path, now=now):
        return "ok", 0
    # A worker that has never written a heartbeat/activity (0) is fresh, not
    # stale — only an *existing* heartbeat that is old can indicate a hang.
    hb = _freshness(state)
    # P-26 s2 (2026-08-18, live-fix re-applied): staleness must be measured
    # from the CURRENT attempt, not a freshest signal across attempts.
    # After a RETRY respawn _freshness() still holds the PREVIOUS attempt's
    # activity -> a fresh generation-bound worker looks "quiet > bound" at
    # grace expiry and is STALL-killed (observed s2-resilience-audit attempt
    # 3, killed at minute 11). The silence clock starts at the live attempt.
    hb = max(hb, started)
    # P7c: the KILL decision uses STALL_QUIET_SECONDS — a long quiet window,
    # not the heartbeat-staleness label (300s), which fires during legitimate
    # long tool/model phases. Absolute silence for STALL_QUIET_SECONDS + 2
    # unchanged fingerprints is the hang signal.
    stale_hb = bool(hb) and (now - hb) > STALL_QUIET_SECONDS
    if fingerprint and prev_fingerprint == fingerprint:
        stall_count = (stall_count or 0) + 1
    else:
        stall_count = 0
    if stale_hb and stall_count >= STALL_MIN_TICKS:
        return "stall", stall_count
    return "ok", stall_count


def kill_worker(pid: int, grace: float = 10.0) -> bool:
    """SIGTERM -> wait grace -> SIGKILL. Returns True only when the process is
    gone (or was never alive). The worker's entire descendant tree is reaped
    first (s6-ownership) so pytest batches and other tool children die with
    the worker instead of leaking past the kill."""
    if not pid:
        return True
    reap_worker_processes(pid)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.time() + grace
    while time.time() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    return not _pid_alive(pid)


def harvest_worker_worktree(task_id: str) -> Dict[str, Any]:
    """Snapshot a dead/exhausted worker's uncommitted worktree evidence.

    P-26 s2 (2026-08-18, live-fix re-applied): the s2-resilience-audit worker
    was STALL-killed on 3/3 attempts with its ONLY real finding (the
    corrupt-ledger fix + its regression test) sitting UNCOMMITTED in the repo
    worktree. A human controller rescued it by inspecting git status. An
    autonomous org must do that itself: capture git status --porcelain +
    git diff --stat (bounded) into the ledger so the work is not silently
    lost when the tree moves on.

    Writes `harvest` onto the worker state (CAS-bumped). Failure-tolerant:
    no workdir, non-git dir, or git failure -> recorded reason, never raises.
    """
    st = load_worker(task_id) or {}
    wd = (st.get("workdir") or "").strip()
    if not wd or not os.path.isdir(wd):
        harvest = {"ts": time.time(), "ok": False, "reason": "no-workdir",
                   "workdir": wd}
    else:
        harvest = {"ts": time.time(), "ok": True, "workdir": wd}
        try:
            r = subprocess.run(
                ["git", "-C", wd, "status", "--porcelain"],
                capture_output=True, text=True, timeout=15)
            lines = [l for l in (r.stdout or "").splitlines() if l.strip()]
            harvest["status_lines"] = len(lines)
            harvest["status"] = lines[:40]  # bounded
            if r.returncode != 0:
                harvest["ok"] = False
                harvest["reason"] = ("git-status-rc-%s" % r.returncode)
        except Exception as exc:  # noqa: BLE001
            harvest["ok"] = False
            harvest["reason"] = f"git-status-error: {exc}"
        try:
            d = subprocess.run(["git", "-C", wd, "diff", "--stat"],
                               capture_output=True, text=True, timeout=60)
            harvest["diff_stat"] = (d.stdout or "").strip()[:2000]
        except Exception as exc:  # noqa: BLE001
            harvest["diff_stat"] = ""
    try:
        st["harvest"] = harvest
        st["seq"] = next_seq(st)
        save_worker_cas(task_id, st)
    except Exception as exc:  # noqa: BLE001
        harvest["save_error"] = str(exc)
    if harvest.get("ok") and harvest.get("status"):
        audit_event(f"task:{task_id}", "worker_worktree_harvested", {
            "workdir": wd, "uncommitted_lines": harvest.get("status_lines", 0)})
    return harvest


# ---------------------------------------------------------------------------
# P5 cross-worker messaging semantics (threads, reply, dedup, ack, stale)
# ---------------------------------------------------------------------------
INBOX_ACKNOWLEDGED = "acknowledged"
INBOX_STALE = "stale"

_MSG_DEDUP_WINDOW = 60.0
_MSG_TTL_SECONDS = 86400.0  # pending older than this is dropped as stale


def post_message(task_id: str, sender: str, receiver: str,
                 message: str, *, kind: str = "message",
                 thread_id: Optional[str] = None,
                 reply_to: Optional[str] = None) -> Dict[str, Any]:
    """Append a durable inbox message with correlation + dedup semantics.

    Entire dedup→append→trim sequence holds the cross-process inbox lock so
    a concurrent delivery/ack rewrite cannot clobber the appended line.
    """
    with _inbox_lock(task_id):
        p = inbox_path(task_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        from hashlib import sha256
        content_hash = sha256(message.encode("utf-8")).hexdigest()[:16]
        now = time.time()
        # P8: normalize correlation fields BEFORE dedup — the CLI passes
        # thread_id=None by default, but entries store ""; comparing your raw
        # None against a stored "" defeated the dedup window and let identical
        # messages double-post from separate CLI processes.
        thread_id = thread_id or ""
        reply_to = reply_to or ""
        # idempotency: identical sender/kind/thread/content within the window
        for e in list_messages(task_id):
            if (now - float(e.get("ts") or 0)) > _MSG_DEDUP_WINDOW:
                continue
            if (e.get("sender") == sender and e.get("kind") == kind
                    and e.get("thread_id") == thread_id
                    and e.get("content_hash") == content_hash):
                return dict(e)
        entry = {
            "id": f"{int(now * 1000)}-{sender[:8]}",
            "sender": sender,
            "receiver": receiver,
            "message": message,
            "status": INBOX_PENDING,
            "ts": now,
            "kind": kind,
            "thread_id": thread_id or "",
            "reply_to": reply_to or "",
            "content_hash": content_hash,
        }
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
        _trim_inbox_locked(p)
        return dict(entry)


def message_present(task_id: str, msg_id: str) -> bool:
    """True when the message id is durably present in the inbox (P-19).

    Post-append verification: the writer confirms the line it was asked to
    publish is actually readable, so a publish can claim success only for
    a message that LANDED. Used by the CLI and by post_message_verified.
    """
    for m in list_messages(task_id):
        if m.get("id") == msg_id:
            return True
    return False


def post_message_verified(task_id: str, sender: str, receiver: str,
                          message: str, *, kind: str = "message",
                          thread_id: Optional[str] = None,
                          reply_to: Optional[str] = None,
                          retries: int = 2) -> Dict[str, Any]:
    """Publish with append verification and idempotent retry (P-19).

    Fan-out call sites (handoff, replacement seeding, supervisor ->
    worker) must not silently drop a message on a transient append failure:
    retry, and surface ok=False to the auditor when it still cannot land.
    """
    entry: Dict[str, Any] = {}
    for _ in range(retries + 1):
        try:
            entry = post_message(task_id, sender, receiver, message,
                                 kind=kind, thread_id=thread_id,
                                 reply_to=reply_to)
        except Exception:  # noqa: BLE001
            entry = {}
        if entry and message_present(task_id, entry.get("id") or ""):
            break
        time.sleep(0.1)
    return entry


def pending_messages_sorted(task_id: str) -> List[Dict[str, Any]]:
    msgs = list_messages(task_id, status=INBOX_PENDING)
    msgs.sort(key=lambda m: (_message_priority(m), m.get("ts", 0.0)))
    return msgs


def ack_message(task_id: str, msg_id: str, *, by: str = "worker") -> bool:
    """Worker ack: marks delivered/pending message as ACKNOWLEDGED."""
    return mark_message(task_id, msg_id, INBOX_ACKNOWLEDGED) or (
        _mark_message_full(task_id, msg_id, INBOX_ACKNOWLEDGED, acked_by=by))


def _mark_message_full(task_id: str, msg_id: str, status: str, **extra) -> bool:
    """Full-field status rewrite. Holds the inbox lock so a concurrent
    append during the read→rewrite cannot be lost."""
    with _inbox_lock(task_id):
        p = inbox_path(task_id)
        if not p.exists():
            return False
        lines = p.read_text(encoding="utf-8").splitlines()
        changed = False
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("id") == msg_id:
                e["status"] = status
                e.update(extra)
                lines[i] = json.dumps(e)
                changed = True
                break
        if changed:
            p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return changed


def check_acks(task_id: str, state: Dict[str, Any]) -> str:
    """If the worker reported a last_acked_msg_id, flip that message to
    ACKNOWLEDGED in the ledger (durable proof the worker consumed it)."""
    acked = str(state.get("last_acked_msg_id") or "")
    if not acked:
        return ""
    for m in list_messages(task_id):
        if m.get("id") == acked and m.get("status") in (
                INBOX_PENDING, INBOX_DELIVERED):
            _mark_message_full(task_id, acked, INBOX_ACKNOWLEDGED, acked_at=time.time())
            return acked
    return ""


def reap_stale_workers(*, max_age_seconds: float = 86400 * 7,
                      missions_dir: Optional[str] = None,
                      dry_run: bool = False) -> Dict[str, List[str]]:
    """Reap CREATED workers that are stale, unreferenced, and unstarted.

    Safety contract:
      - ONLY workers in status "CREATED" are eligible (never ACTIVE/COMPLETE/etc).
      - Age must exceed max_age_seconds since created_at.
      - Workers referenced by ANY mission phase/discovery are NEVER reaped.
      - Worker must have no started_at / worker_pid (never actually ran).

    Returns {"reaped": [task_ids], "skipped": [task_ids]}.
    Idempotent: second run finds nothing.
    """
    now = time.time()
    reaped: List[str] = []
    skipped: List[str] = []

    # Build set of all mission-referenced worker task_ids
    referenced: set = set()
    base = Path(missions_dir) if missions_dir else Path(os.environ.get(
        "HERMES_SUPERVISOR_DIR") or os.path.expanduser("~/.hermes-supervisor"))
    missions_base = base / "missions"
    if missions_base.is_dir():
        for f in missions_base.glob("*.json"):
            try:
                m = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            for p in m.get("phases", []):
                wt = p.get("worker_task", "")
                if wt:
                    referenced.add(wt)
            for d in m.get("discoveries", []):
                wt = d.get("worker_task", "")
                if wt:
                    referenced.add(wt)

    tasks_base = _tasks_dir()
    if not tasks_base.is_dir():
        return {"reaped": [], "skipped": []}

    for d in sorted(tasks_base.iterdir()):
        if not d.is_dir():
            continue
        wpath = d / "worker.json"
        if not wpath.exists():
            continue
        try:
            w = json.loads(wpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        tid = w.get("task_id", d.name)
        status = (w.get("status") or "").upper()

        if status != "CREATED":
            skipped.append(tid)
            continue

        age = now - float(w.get("created_at") or now)
        if age < max_age_seconds:
            skipped.append(tid)
            continue

        if tid in referenced:
            skipped.append(tid)
            continue

        # Never actually ran — safe to reap
        started = w.get("started_at") or w.get("worker_pid")
        if started:
            skipped.append(tid)
            continue

        if dry_run:
            reaped.append(tid)
            continue

        try:
            shutil.rmtree(d)
            reaped.append(tid)
        except Exception:
            skipped.append(tid)

    return {"reaped": reaped, "skipped": skipped}


def expire_stale_messages(task_id: str, *, ttl: float = _MSG_TTL_SECONDS) -> int:
    """Pending messages older than TTL are marked stale and never delivered."""
    n = 0
    for m in list_messages(task_id, status=INBOX_PENDING):
        if (time.time() - float(m.get("ts") or 0)) > ttl:
            if _mark_message_full(task_id, m["id"], INBOX_STALE, stale_at=time.time()):
                n += 1
    return n


def deliver_pending(task_id: str, *, clear_after_deliver: bool = False) -> int:
    """Deliver ONE pending inbox message (P2 invariants: exactly one per idle
    boundary, user priority first). The delivered instruction carries a
    correlation envelope the worker can acknowledge."""
    msgs = pending_messages_sorted(task_id)
    if not msgs:
        return 0
    next_msg = msgs[0]
    env = f"[msg:{next_msg.get('id')} kind:{next_msg.get('kind') or 'message'} thread:{next_msg.get('thread_id') or ''} from:{next_msg.get('sender')}] "
    write_command(task_id, WorkerDecision(
        verdict="SUPERVISOR_MESSAGE",
        command="CONTINUE",
        instruction=env + str(next_msg.get("message", "")),
        score=0.5,
    ))
    mark_message(task_id, next_msg["id"], INBOX_DELIVERED)
    if HEARTBEAT_REANCHOR_AFTER_DELIVERY:
        st = load_worker(task_id)
        if st:
            touch_heartbeat(st)
            # CAS-safe: only re-anchor if the direct successor holds
            st_new = dict(st)
            st_new["seq"] = next_seq(st)
            save_worker_cas(task_id, st_new)
    return 1


# ---------------------------------------------------------------------------
# P5 identity / resume lineage (task-stable, does NOT duplicate Hermes sessions)
# ---------------------------------------------------------------------------


def record_spawned_pid(task_id: str, pid: int) -> bool:
    """Persist a freshly spawned worker's pid under CAS after a respawn so
    crash/timeout detection sees the new process (the loop's RETRY path
    spawns without going through the CLI `start` action). Also re-anchors
    `started_at` so the per-attempt runtime budget restarts (P7), and stamps
    the identity/run_id that worker_lineage + handoffs carry (P8 — merged
    from the now-deleted start_worker_with_identity so the capability is not
    lost when we drop the dead duplicate)."""
    st = load_worker(task_id)
    if st is None:
        return False
    st["worker_pid"] = int(pid)
    st["started_at"] = time.time()
    st["proc_start"] = _proc_start_ticks(int(pid))
    if not st.get("worker_identity"):
        st["worker_identity"] = f"w-{task_id}"
    from uuid import uuid4
    st["run_id"] = str(uuid4())[:12]
    st["seq"] = next_seq(st)
    ok, _ = save_worker_cas(task_id, st)
    return ok


def worker_lineage(task_id: str) -> List[Dict[str, Any]]:
    """Compact attempt + handoff lineage for resume/reporting."""
    st = load_worker(task_id) or {}
    return {
        "task_id": task_id,
        "worker_identity": st.get("worker_identity"),
        "attempt": attempt_number(st),
        "seq": current_seq(st),
        "attempts": list(st.get("attempts") or []),
        "handoff_versions": len(st.get("handoffs") or []),
        "status": st.get("status"),
    }

# ===========================================================================
# P6 block — campaign/role/obligation layer, failure reconciliation, worker
# replacement, campaign completion semantics.
#
# Campaign 4 exposed the gap this block closes: worker B ended FAILED after
# bounded attempts, yet the campaign "was complete" because B's findings had
# already been reproduced and fixed elsewhere. That decision was made by the
# parent session, not by the supervisor. Nothing in the ledger recorded —
# machine-verifiably — that B's failure was RECONCILED (findings preserved +
# responsibility transferred/covered + adversarial role satisfied). This block
# gives the supervisor an explicit ROLE/RESPONSIBILITY/REQUIRED-EVIDENCE/OWNER
# model, a WORKER_FAILURE_RECONCILED outcome distinct from COMPLETE, and a
# deterministically computed CAMPAIGN_COMPLETE|FAILED|BLOCKED state with a
# machine-readable rationale. The campaign ledger lives in the SAME supervisor
# dir (campaigns/<id>.json) and references worker ledgers — no second state
# system.
# ===========================================================================

CAMPAIGN_DIR_NAME = "campaigns"
_CAMPAIGN_MAX_ROLES = 16      # ponytail: bounded role list
_CAMPAIGN_MAX_TRANSFERS = 8   # ponytail: bounded transfer history per role


def campaigns_dir() -> Path:
    base = os.environ.get("HERMES_SUPERVISOR_DIR") or os.path.expanduser(
        "~/.hermes-supervisor")
    d = Path(base) / CAMPAIGN_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def campaign_path(campaign_id: str) -> Path:
    return campaigns_dir() / f"{campaign_id}.json"


def _new_role(role_id: str, responsibility: str,
              required_evidence: List[str]) -> Dict[str, Any]:
    return {
        "role_id": role_id,
        "responsibility": responsibility,
        "required_evidence": list(required_evidence or []),
        "owner": "",                 # task_id currently owning this role
        "status": "UNSATISFIED",     # UNSATISFIED | SATISFIED | COVERED | TRANSFERRED
        "transfer_history": [],      # bounded; each entry: {from_task, to_task, at, reason}
        "satisfied_by": "",          # task/evidence that proved it
        "satisfied_at": 0.0,
    }


def create_campaign(campaign_id: str, objective: str,
                    roles: Optional[List[Dict[str, Any]]] = None
                    ) -> Tuple[Path, Dict[str, Any]]:
    """Create a campaign ledger: objective + role obligations."""
    c = {
        "campaign_id": campaign_id,
        "objective": objective,
        "created_at": time.time(),
        "updated_at": time.time(),
        "status": "ACTIVE",
        "roles": [],
        "workers": {},                   # task_id -> {role, outcome, lineage}
        "reconciled_failures": [],       # evidence-gated reconciliation records
        "completion_rationale": None,    # set when terminal
    }
    for r in (roles or []):
        if len(c["roles"]) >= _CAMPAIGN_MAX_ROLES:
            break
        c["roles"].append(_new_role(
            r.get("role_id") or r.get("id") or f"role{len(c['roles'])+1}",
            r.get("responsibility") or r.get("description") or "",
            r.get("required_evidence") or [],
        ))
    save_campaign(c)
    return campaign_path(campaign_id), c


def save_campaign(c: Dict[str, Any]) -> Path:
    c["updated_at"] = time.time()
    path = campaign_path(c["campaign_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(c, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_campaign(campaign_id: str) -> Optional[Dict[str, Any]]:
    p = campaign_path(campaign_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _find_role(c: Dict[str, Any], role_id: str) -> Optional[Dict[str, Any]]:
    for r in c.get("roles", []):
        if r.get("role_id") == role_id:
            return r
    return None


def assign_role(campaign_id: str, role_id: str, task_id: str,
                *, reason: str = "assignment") -> bool:
    """Bind a task as the current OWNER of a role obligation."""
    c = load_campaign(campaign_id)
    if c is None:
        return False
    r = _find_role(c, role_id)
    if r is None:
        return False
    prev = r.get("owner")
    r["owner"] = task_id
    if not r.get("satisfied_by"):
        r["status"] = "UNSATISFIED"
    if prev and prev != task_id:
        r["status"] = "TRANSFERRED"
        hist = r.setdefault("transfer_history", [])
        if len(hist) < _CAMPAIGN_MAX_TRANSFERS:
            hist.append({"from_task": prev, "to_task": task_id,
                         "at": time.time(), "reason": reason})
    c.setdefault("workers", {}).setdefault(task_id, {"status": "ASSIGNED"})
    c["workers"][task_id]["role"] = role_id
    save_campaign(c)
    return True


def note_worker_outcome(campaign_id: str, task_id: str, outcome: str,
                        *, evidence: Optional[List[str]] = None) -> Dict[str, Any]:
    """Record a worker terminal outcome on the campaign ledger. outcome is one
    of WORKER_OUTCOMES. Never mutates the worker.json — campaign outcome is a
    separate CAMPAIGN-level fact about obligation coverage."""
    c = load_campaign(campaign_id)
    if c is None:
        return {}
    w = c.setdefault("workers", {}).setdefault(task_id, {})
    w["status"] = outcome
    if evidence:
        w["evidence"] = list(evidence)
    w["updated_at"] = time.time()
    save_campaign(c)
    return w


def reconcile_worker_failure(
        campaign_id: str, task_id: str, *,
        findings_preserved: bool = True,
        responsible_covered: bool = False,
        responsibility_transferred_to: str = "",
        adversarial_role_satisfied: bool = False,
        completion_criteria_ok: bool = True,
        evidence: Optional[List[str]] = None,
        note: str = "") -> Tuple[bool, str, Dict[str, Any]]:
    """The supervisor's FAILURE-RECONCILIATION decision.

    A WORKER FAILED may only become WORKER_FAILURE_RECONCILED when the
    supervisor can VERIFY from evidence that:
      1) the failed worker's known findings are preserved in its ledger/handoff
         (findings_preserved), and
      2) its unfinished responsibility is unnecessary, transferred, or covered
         by another worker (responsible_covered or responsibility_transferred_to), and
      3) no required adversarial/verification role silently disappeared
         (adversarial_role_satisfied or transfer-to with coverage), and
      4) the campaign's completion criteria still hold (completion_criteria_ok).

    This NEVER converts FAILED -> COMPLETE. The worker ledger remains
    status=FAILED as the honest record; the campaign records
    WORKER_FAILURE_RECONCILED. Returns (accepted, reason, campaign)."""
    c = load_campaign(campaign_id)
    if c is None:
        return False, f"no campaign {campaign_id}", {}
    reasons = []
    if not findings_preserved:
        reasons.append("findings NOT preserved")
    if not responsible_covered and not responsibility_transferred_to:
        reasons.append("responsibility NOT transferred or covered")
    if not adversarial_role_satisfied and not responsibility_transferred_to:
        reasons.append("adversarial role NOT satisfied")
    if not completion_criteria_ok:
        reasons.append("completion criteria NOT satisfied")
    if reasons:
        audit_event(task_id, "reconcile_failure_rejected",
                    {"campaign": campaign_id, "missing": reasons})
        return False, "reconcile rejected: " + "; ".join(reasons), c
    rec = {
        "task_id": task_id,
        "at": time.time(),
        "findings_preserved": findings_preserved,
        "responsible_transferred_to": responsibility_transferred_to,
        "adversarial_role_satisfied": adversarial_role_satisfied,
        "evidence": list(evidence or []),
        "note": note,
    }
    c.setdefault("reconciled_failures", []).append(rec)
    w = c.setdefault("workers", {}).setdefault(task_id, {})
    w["status"] = "WORKER_FAILURE_RECONCILED"
    r = _find_role(c, w.get("role") or "")
    if r:
        r["status"] = "COVERED" if r.get("status") != "SATISFIED" else r["status"]
    save_campaign(c)
    return True, f"{task_id} failure RECONCILED -> campaign may continue", c


def mark_role_evidence(campaign_id: str, role_id: str, task_id: str,
                       supplied: List[str]) -> bool:
    """A role obligation is SATISFIED only when supplied evidence covers the
    role's required evidence list (deterministic subset match on lowercased
    keywords). Rejects with an audit trail when missing pieces."""
    c = load_campaign(campaign_id)
    if c is None:
        return False
    r = _find_role(c, role_id)
    if r is None:
        return False
    have = set(str(e).lower() for e in (supplied or []))
    need = set(str(x).lower() for x in (r.get("required_evidence") or []))
    if need and not need.issubset(have):
        missing = ", ".join(sorted(need - have))
        audit_event(task_id, "role_evidence_rejected",
                    {"campaign": campaign_id, "role": role_id, "missing": missing[:120]})
        return False
    r["status"] = "SATISFIED"
    r["satisfied_by"] = task_id
    r["satisfied_at"] = time.time()
    save_campaign(c)
    return True


def running_role_obligations(campaign_id: str) -> List[Dict[str, Any]]:
    """Roles still UNSATISFIED/TRANSFERRED (need an owner or evidence)."""
    c = load_campaign(campaign_id)
    if c is None:
        return []
    return [r for r in c.get("roles", [])
            if r.get("status") in ("UNSATISFIED", "TRANSFERRED")]


def _role_satisfied(r: Dict[str, Any]) -> bool:
    """A role is satisfied when marked SATISFIED by evidence OR COVERED via a
    reconciled failure (its failed owner's responsibility was independently
    fulfilled). COVERED is not a loophole: it only appears through
    reconcile_worker_failure's evidence gate."""
    return r.get("status") in ("SATISFIED", "COVERED")


def campaign_status(campaign_id: str) -> Dict[str, Any]:
    """Campaign-level state + machine-readable completion rationale. Deterministic;
    no LLM. The only way to reach CAMPAIGN_COMPLETE is: every role obligation
    SATISFIED (or covered via reconciliation) and no unreconciled failures."""
    c = load_campaign(campaign_id)
    if c is None:
        return {"status": "MISSING", "campaign_id": campaign_id,
                "rationale": {"error": f"no campaign {campaign_id}"}}
    roles = c.get("roles", [])
    workers = c.get("workers", {})
    satisfied_roles = [r for r in roles if _role_satisfied(r)]
    uncovered_roles = running_role_obligations(campaign_id)
    failed_workers = [tid for tid, w in workers.items()
                      if w.get("status") in ("WORKER_FAILED", "WORKER_CRASHED")]
    unreconciled = [tid for tid in failed_workers
                    if tid not in {f["task_id"] for f in c.get("reconciled_failures", [])}]
    blocked = bool([r for r in roles if r.get("status") == "BLOCKED"]) or \
        bool([w for w in workers.values() if w.get("blocked")])
    all_roles_ok = bool(roles) and len(satisfied_roles) == len(roles) and not uncovered_roles
    if unreconciled:
        status = "CAMPAIGN_BLOCKED" if blocked else "CAMPAIGN_FAILED"
    elif all_roles_ok:
        status = "CAMPAIGN_COMPLETE"
    elif blocked:
        status = "CAMPAIGN_BLOCKED"
    else:
        status = "ACTIVE"
    rationale = {
        "campaign_id": campaign_id,
        "objective": c.get("objective"),
        "status": status,
        "roles": [{"role_id": r["role_id"], "status": r["status"],
                   "owner": r.get("owner"), "satisfied_by": r.get("satisfied_by")}
                  for r in roles],
        "workers": {tid: {"status": w.get("status"), "role": w.get("role")}
                    for tid, w in workers.items()},
        "reconciled_failures": c.get("reconciled_failures", []),
        "unresolved": {
            "uncovered_roles": [{"role_id": r["role_id"], "owner": r.get("owner")}
                                for r in uncovered_roles],
            "unreconciled_failures": unreconciled,
        },
        "evidence": {
            "campaign_ledger": str(campaign_path(campaign_id)),
            "workers": [str(worker_path(t)) for t in c.get("workers", {})],
        },
        "reason": "",
    }
    if status == "CAMPAIGN_COMPLETE":
        rationale["reason"] = "All role obligations satisfied, no unreconciled failures."
    elif status == "CAMPAIGN_FAILED":
        rationale["reason"] = f"Unreconciled worker failures: {unreconciled}"
    elif status == "CAMPAIGN_BLOCKED":
        rationale["reason"] = "Campaign blocked on external input or dependency."
    c["status"] = status
    c["completion_rationale"] = rationale
    save_campaign(c)
    return rationale


# ---------------------------------------------------------------------------
# P6: autonomous worker replacement / responsibility transfer
# ---------------------------------------------------------------------------

def spawn_replacement(campaign_id: str, role_id: str, task_id: str,
                      *, budget: Optional[Dict[str, int]] = None,
                      replace_id: Optional[str] = None
                      ) -> Tuple[bool, str, str]:
    """Option C: spawn a FRESH worker with the same responsibility after a
    failed worker exhausted attempts. New task id; the old worker's ledger
    (attempts + handoff + findings) stays put; the replacement's inbox is
    seeded with the old handoff for message continuity. Returns
    (ok, msg, new_task_id)."""
    old = load_worker(task_id)
    cam = load_campaign(campaign_id) or {}
    new_id = replace_id or f"{task_id}-r{len(cam.get('workers', {})) + 1}"
    task_text = (old or {}).get("task") or (cam.get("objective") or task_id)
    extra = (
        "You are the REPLACEMENT worker for task %s. The previous worker "
        "failed; its findings are preserved in %s (see worker.json handoffs "
        "and the inbox). Re-derive independence: read the original objective, "
        "previous findings, and the evidence before acting. You have the same "
        "responsibility as the failed worker."
        % (task_id, worker_path(task_id)))
    try:
        create_worker(task_text, task_id=new_id, budget=budget,
                      extra_brief=extra)
    except Exception as exc:  # noqa: BLE001
        return False, f"create failed: {exc}", ""
    if old and old.get("handoff"):
        post_message_verified(new_id, sender=task_id, receiver="worker",
                              message=json.dumps(old["handoff"], default=str),
                              kind="handoff", thread_id=old.get("task_id", ""))
    if old:
        old.setdefault("replaced_by", new_id)
        old.setdefault("replacement_reason", "policy: attempts exhausted")
        save_worker(old, task_id)
    if cam:
        w = cam.setdefault("workers", {}).setdefault(new_id, {})
        w["replaces"] = task_id
        w["role"] = role_id
        save_campaign(cam)
    return True, f"replacement spawned {new_id} for {task_id}", new_id


def adopt_or_transfer(campaign_id: str, task_id: str, *,
                      to_task: str = "") -> Tuple[bool, str]:
    """Explicitly transfer a role obligation to another worker (option D):
    evidence that responsibility continues to be discharged by someone who can
    actually do it, not abandoned."""
    c = load_campaign(campaign_id)
    if c is None:
        return False, f"no campaign {campaign_id}"
    w = c.get("workers", {}).get(task_id, {})
    role_id = w.get("role") or ""
    r = _find_role(c, role_id) if role_id else None
    if r is None:
        return False, f"no role {role_id!r} owned by {task_id}"
    if to_task:
        hist = r.setdefault("transfer_history", [])
        if len(hist) < _CAMPAIGN_MAX_TRANSFERS:
            hist.append({"from_task": task_id, "to_task": to_task,
                         "at": time.time(), "reason": "explicit transfer"})
        r["owner"] = to_task
        r["status"] = "TRANSFERRED"
        c.setdefault("workers", {}).setdefault(to_task, {})["replaces"] = task_id
        save_campaign(c)
        return True, f"role {role_id} transferred {task_id} -> {to_task}"
    r["status"] = "TRANSFERRED"
    save_campaign(c)
    return True, f"role {role_id} freed (owner {task_id} failed) — reassign when covered"

# ---------------------------------------------------------------------------
# MISSION: multi-phase autonomous continuation (P-14)
# ---------------------------------------------------------------------------
# A CAMPAIGN completes when its role obligations are satisfied. That is NOT a
# mission terminal condition: the mission owns required PHASES (hunt, fix,
# attack, audit, research, verify) plus terminal criteria and unresolved
# findings. CAMPAIGN_COMPLETE only advances a phase; MISSION_COMPLETE is a
# separate deterministic state the orchestrator reaches only when every
# required phase has evidence AND criteria are met AND no unresolved open
# finding remains. Worker claims (COMPLETE, "nothing left", recommendations)
# are evidence records, never authority for mission termination.

MISSION_DIR_NAME = "missions"
MISSION_STATUSES = ("MISSION_ACTIVE", "MISSION_COMPLETE",
                    "MISSION_BLOCKED", "MISSION_FAILED", "MISSION_CANCELLED",
                    "MISSION_MISSING")
PHASE_STATUSES = ("PENDING", "ACTIVE", "COMPLETE", "BLOCKED", "FAILED",
                  "SKIPPED")
_MISSION_MAX_PHASES = 32       # ponytail: bounded phase list per mission
_MISSION_MAX_FINDINGS = 200    # ponytail: bounded unresolved-findings list
_MISSION_MAX_CRITERIA = 32
_MISSION_MAX_REPEATS = 3       # a phase/criterion may auto-repeat at most 3x
_MISSION_MAX_RETRIES = 3       # bound: auto-retries of an infra-FAILED phase


def missions_dir() -> Path:
    base = os.environ.get("HERMES_SUPERVISOR_DIR") or os.path.expanduser(
        "~/.hermes-supervisor")
    d = Path(base) / MISSION_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def mission_path(mission_id: str) -> Path:
    return missions_dir() / f"{mission_id}.json"


# ---------------------------------------------------------------------------
# Supervisor lease — durable single-owner guard for the mission loop.
#
# The loop process holds a small lease file next to the mission ledger. A
# crashed/killed loop leaves the lease behind; a replacement loop takes the
# lease over ONLY when the holder pid is dead (crash) or the lease has been
# stale longer than the TTL (unresponsive/host lost). While a live lease is
# held, a second loop REFUSES to start: there is never a second execution
# authority for a mission. The loop renews the lease each iteration; release
# happens in the mission-loop finally (also on clean SIGTERM paths).
# All mutations are under an flock so acquisition is atomic across processes.

LEASE_DEFAULT_TTL = float(
    os.environ.get("MISSION_LEASE_TTL_SEC", "120") or 120)


def _lease_path(mission_id: str) -> Path:
    return missions_dir() / mission_id / ".lease.json"


def _lease_lock(mission_id: str) -> Path:
    return missions_dir() / mission_id / ".lease.lock"


def _lease_mutate(mission_id: str, fn):
    """Run fn(lease_dict) under an exclusive per-mission lock, persist, and
    return fn's result. fn may write the lease dict in place (or return a
    dict to store)."""
    p = _lease_lock(mission_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(p), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        lease = {}
        lp = _lease_path(mission_id)
        if lp.exists():
            try:
                lease = json.loads(lp.read_text(encoding="utf-8"))
            except Exception:
                lease = {}
        out = fn(lease)
        if out is not None and isinstance(out, dict):
            lp.parent.mkdir(parents=True, exist_ok=True)
            tmp = lp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(out, indent=2, default=str),
                           encoding="utf-8")
            os.replace(tmp, lp)
        return out or lease
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _pid_alive_checked(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


def lease_acquire(mission_id: str, holder: str, *,
                  ttl: Optional[float] = None) -> tuple:
    """Try to become the single supervisor owner for the mission.

    Returns (True, msg) on acquire; (False, why) while another LIVE lease
    is held. Takeover (automatically) when the current holder pid is dead or
    `ttl` s have passed without a renewal.
    """
    t = float(ttl if ttl is not None else LEASE_DEFAULT_TTL)

    def _fn(cur: dict) -> dict:
        cur_holder = cur.get("holder", "")
        if cur_holder and cur_holder != holder:
            stale = False
            if not _pid_alive_checked(int(cur.get("pid") or 0)):
                stale = True
            elif (float(cur.get("last_seen_at") or 0) + t) < time.time():
                stale = True
            if not stale:
                raise _LeaseHeld(cur)
        cur.update({"holder": holder, "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "started_at": float(cur.get("started_at") or time.time()),
                    "last_seen_at": time.time(), "ttl": t})
        return cur

    try:
        _lease_mutate(mission_id, _fn)
        return True, "lease acquired"
    except _LeaseHeld as held:
        return False, f"lease held by {held.lease.get('holder')} " \
                      f"(pid {held.lease.get('pid')})"


class _LeaseHeld(Exception):
    def __init__(self, lease):
        super().__init__("lease held")
        self.lease = lease


def lease_renew(mission_id: str, holder: str) -> bool:
    """Refresh last_seen; only the current holder succeeds. Returns False if
    the holder changed underneath us (we lost the lease -> stop working)."""
    def _fn(cur: dict) -> dict:
        if cur.get("holder") != holder:
            raise _LeaseHeld(cur)
        cur["last_seen_at"] = time.time()
        return cur
    try:
        _lease_mutate(mission_id, _fn)
        return True
    except _LeaseHeld:
        return False


def lease_release(mission_id: str, holder: str) -> None:
    if not holder:
        return
    def _fn(cur: dict):
        if cur.get("holder") == holder:
            lp = _lease_path(mission_id)
            try:
                lp.unlink()
            except FileNotFoundError:
                pass
    _lease_mutate(mission_id, _fn)


def lease_state(mission_id: str) -> dict:
    """Current lease view for humans/controller: holder, pid, liveness,
    staleness, takeover-ok."""
    lp = _lease_path(mission_id)
    cur = {}
    if lp.exists():
        try:
            cur = json.loads(lp.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
    alive = _pid_alive_checked(int(cur.get("pid") or 0))
    last = float(cur.get("last_seen_at") or 0)
    ttl = float(cur.get("ttl") or LEASE_DEFAULT_TTL)
    return {
        "holder": cur.get("holder", ""), "pid": cur.get("pid", 0),
        "alive": alive,
        "stale": bool(cur and (not alive or (last + ttl) < time.time())),
        "last_seen_at": last, "ttl": ttl,
        "exists": bool(cur),
    }


def _new_phase(phase_id: str, task: str, *,
               required_evidence=None, after: str = "") -> dict:
    return {
        "phase_id": phase_id,
        "task": task,
        "required_evidence": list(required_evidence or []),
        "status": "PENDING",
        "after": after or "",      # phase_id that must be COMPLETE first
        "evidence": [],
        "worker_task": "",         # task_id that discharged this phase
        "worker_by": "",
        "retry_count": 0,          # bounded automatic retries allowed
        "updated_at": time.time(),
    }


def phase_add(mission_id: str, phase_id: str, task: str, *,
              after: str = "", required_evidence=None) -> tuple:
    """Append a PENDING phase to an existing mission ledger (self-service CLI
    edit surface; no raw JSON edits). Refuses a duplicate phase_id and a
    genuinely terminal (COMPLETE with phases) mission; writes are atomic
    (tmp + os.replace) so a concurrent mission `loop` reader never sees a torn
    file. (P-26 s2 restore: live-side function re-ported onto the p26 base.)
    """
    m = load_mission(mission_id)
    if m is None:
        return False, f"no mission {mission_id}"
    if any(p.get("phase_id") == phase_id for p in m.get("phases", [])):
        return False, f"phase {phase_id} already exists in mission {mission_id}"
    # Terminal guard: a COMPLETE ledger that holds phases is finished and
    # refuses growth. An EMPTY ledger is a bootstrap artifact — `mission
    # create` prints mission_status which recomputes zero phases as COMPLETE,
    # but the whole point of the phase-add CLI is growing such a ledger.
    if m.get("status") == "MISSION_COMPLETE" and m.get("phases"):
        return False, f"mission {mission_id} terminal (COMPLETE); refusing to append"
    if len(m.get("phases", [])) >= _MISSION_MAX_PHASES:
        return False, f"phase cap ({_MISSION_MAX_PHASES}) reached"
    m["phases"].append(_new_phase(
        phase_id, task, required_evidence=required_evidence, after=after))
    m["status"] = "MISSION_ACTIVE"  # a PENDING phase means the ledger is live
    save_mission(m)
    return True, f"phase {phase_id} added to mission {mission_id}"


def mission_phases(mission_id: str) -> Optional[list]:
    """Compact phase list for inspection: [phase_id, status, task...]."""
    m = load_mission(mission_id)
    if m is None:
        return None
    return m.get("phases", [])


def create_mission(mission_id: str, objective: str,
                   phases=None, requirements=None,
                   repeat_limit: int = _MISSION_MAX_REPEATS,
                   workdir: str = ""
                   ) -> tuple:
    """Create a mission ledger: objective + required phases + terminal
    requirements. Phases are the units of autonomous continuation; each
    requirement is an evidence keyword that must appear in phase evidence
    for the mission to complete. workdir is the directory where mission
    work happens (commits/files verified here); empty = supervisor CWD."""
    m = {
        "mission_id": mission_id,
        "objective": objective,
        "created_at": time.time(),
        "updated_at": time.time(),
        "status": "MISSION_ACTIVE",
        "phases": [],
        "requirements": list(requirements or []),  # evidence keywords
        "criteria_met": [],
        "unresolved_findings": [],                 # {id, text, status, evidence}
        "repeat_limit": repeat_limit,
        "completed_at": None,
        "terminal_rationale": None,
        "workdir": workdir or "",
    }
    for ph in (phases or []):
        if len(m["phases"]) >= _MISSION_MAX_PHASES:
            break
        p = _new_phase(
            ph.get("phase_id") or f"phase{len(m['phases'])+1}",
            ph.get("task") or ph.get("description") or "",
            required_evidence=ph.get("required_evidence") or [],
            after=ph.get("after") or "",
        )
        m["phases"].append(p)
    # P-26 s2 (2026-08-18, live-fix re-applied): a corrupt mission ledger
    # must NEVER be silently overwritten — that destroys the only copy of
    # prior state. Refuse the write and leave the payload for forensics;
    # only a missing OR valid ledger may be created.
    path = mission_path(mission_id)
    if path.exists():
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            audit_event(f"mission:{mission_id}", "corrupt_ledger_refused", {
                "path": str(path), "mission_id": mission_id})
            return path, None
    save_mission(m)
    return mission_path(mission_id), m


def save_mission(m: dict) -> Path:
    m["updated_at"] = time.time()
    path = mission_path(m["mission_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_mission(mission_id: str) -> Optional[dict]:
    p = mission_path(mission_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _find_phase(m: dict, phase_id: str) -> Optional[dict]:
    for p in m.get("phases", []):
        if p.get("phase_id") == phase_id:
            return p
    return None


_EVIDENCE_KINDS = ("commit", "test_run", "file")
_COMMIT_REF_RE = re.compile(r"\b[0-9a-f]{7,40}\b", re.I)
# A structured result: "12 passed / 1 failed" or an equal total "26/26".
# Equal fractions + result words stop suite-style totals being confused with
# dates ("2026/08/19") or arbitrary "a/b" prose (s5 audit).
_TEST_RUN_RE = re.compile(
    r"\b\d+\s+(?:passed|failed|ok)\b|\b(\d+)\s*/\s*(\d+)\b(?:\s+(?:passed|failed|ok))?",
    re.I)
_FILE_RE = re.compile(r"[A-Za-z0-9_./-]+\.(?:py|json|md|yaml|yml|toml|sh|ts|tsx|js|go|rs|c|h|txt|lock)\b")


def _git_object_exists(ref: str, workdir: Optional[str]) -> bool:
    """Deterministic artifact probe: does <ref> name a real object in the
    repo at workdir? Never raises; fail-closed on any error (no repo, no
    git binary, timeout). bounded: one cat-file per ref."""
    try:
        wd = workdir or os.getcwd()
        r = subprocess.run(["git", "-C", wd, "cat-file", "-t", ref],
                           capture_output=True, text=True, timeout=10)
        return (r.returncode == 0
                and r.stdout.strip() in ("commit", "tag", "blob", "tree"))
    except Exception:
        return False


def _evidence_kinds(evidence_line: str, workdir: Optional[str]) -> set:
    """Classify ONE evidence line to the artifact kinds it carries. This is
    the semantic heart of the gate: what real things does this line point
    at? (commit in the git object DB, a structured test result, a file on
    disk). A line that proves nothing is a 'claim'."""
    kinds = set()
    if not evidence_line:
        return kinds
    text = str(evidence_line)
    # commit: a ref that actually exists in the repo
    for token in _COMMIT_REF_RE.findall(text):
        if _git_object_exists(token, workdir):
            kinds.add("commit")
            break
    # test_run: a structured result (count + outcome, or equal total N/N)
    if _TEST_RUN_RE.search(text):
        kinds.add("test_run")
    # file: a named code/doc file that exists under the workdir
    if workdir:
        for tok in _FILE_RE.findall(text):
            if os.path.isfile(os.path.join(workdir, tok)):
                kinds.add("file")
                break
    return kinds


def _evidence_covers(evidence, required, workdir: Optional[str] = None) -> bool:
    """Does the supplied evidence satisfy the phase's required evidence?

    P-2 string normalization kept (evidence may be one JSON string).
    Two requirement forms:
      - plain str A -> legacy literal token. When the evidence carries a
        VERIFIED artifact (real git object / structured result / file),
        the token is a HINT and the artifact satisfies it: a real commit
        or real suite result is stronger evidence than the word
        "implemented" (s4 demonstration: artifact-complete evidence was
        rejected for missing words). When no artifact is present the
        literal stays a hard gate, so a bare 'done' claim still fails.
      - {'kind': <artifact>, "url": <hint>}:
        {'kind': 'commit'|'test_run'|'file'}  — satisfied ONLY by an
        evidence line classified as that artifact (real git object / real
        structured result / real file). The optional 'hint' is advisory
        (what the artifact should be), never a gate: words must never be
        stronger than artifacts, but a real artifact is never blocked by a
        spelling hint.
    An artifact requirement with no verifiable artifact fails CLOSED
    (commit kind without a repo, etc.). Bare 'done' claims satisfy nothing.
    """
    if isinstance(evidence, str):
        evidence = [evidence]
    have = " ".join(str(e).lower() for e in (evidence or []))
    kinds_seen = set()
    for e in evidence or []:
        kinds_seen |= _evidence_kinds(e, workdir)
    # Artifacts are the strongest signal: once a real commit/test/file is
    # present, literal tokens become hints and cannot veto the completion
    # (s5 switch-2; s4 demonstration). With NO artifact the tokens stay
    # gates, so a bare claim never scrapes through on words alone.
    artifactual = bool(kinds_seen)
    for r in (required or []):
        if isinstance(r, dict):
            kind = str(r.get("kind") or "").lower()
            if kind not in _EVIDENCE_KINDS:
                return False              # unknown kind = unsatisfiable
            if kind not in kinds_seen:
                return False              # artifact not present
        else:
            if str(r).lower() not in have and not artifactual:
                return False              # literal gate (no artifact proof)
    return True


# ---------------------------------------------------------------------------
# Phase transitions (evidence-gated)
# ---------------------------------------------------------------------------

def _mission_event(mission_id: str, level: str, kind: str,
                   message: str, payload=None) -> None:
    """Best-effort controller-visible journal event. Never raises: the
    journal is an observability aid, not authority."""
    try:
        from hermes_cli import mission_ops as MO
        MO.log_event(mission_id, level, kind, message, payload or {})
    except Exception:
        pass


def phase_complete(mission_id: str, phase_id: str, *,
                   worker_by: str = "", evidence=None,
                   workdir: Optional[str] = None) -> tuple:
    """A phase is COMPLETE only when its required evidence is satisfied —
    literal tokens (legacy) OR artifact kinds ({"kind": "commit"|"test_run"|
    "file"}) verified against the repo at workdir. A bare 'done' claim never
    satisfies either form. workdir defaults to the fallback used by the
    artifact probes (os.getcwd at probe time) when not supplied."""
    m = load_mission(mission_id)
    if m is None:
        return False, f"no mission {mission_id}"
    p = _find_phase(m, phase_id)
    if p is None:
        return False, f"no phase {phase_id} in mission {mission_id}"
    required = p.get("required_evidence") or []
    if not _evidence_covers(evidence or [], required, workdir=workdir):
        missing = []
        have = " ".join(str(e).lower() for e in (evidence or []))
        kinds_seen = set()
        for e in evidence or []:
            kinds_seen |= _evidence_kinds(e, workdir)
        for r in required:
            if isinstance(r, dict):
                k = str(r.get("kind") or "").lower()
                if k not in kinds_seen:
                    missing.append(f"artifact:{k}")
            elif str(r).lower() not in have and not kinds_seen:
                missing.append(str(r))
        audit_event(worker_by or "supervisor", "mission_phase_evidence_rejected",
                    {"mission": mission_id, "phase": phase_id,
                     "missing": missing[:80],
                     "workdir": workdir or os.getcwd()})
        return False, f"phase {phase_id} evidence incomplete; missing {missing}"
    p["status"] = "COMPLETE"
    p["evidence"] = list(evidence or [])
    p["worker_by"] = worker_by
    p["updated_at"] = time.time()
    _recompute_criteria(m)
    save_mission(m)
    _mission_event(mission_id, "MEDIUM", "PHASE_COMPLETE",
                   f"phase {phase_id} complete by {worker_by or 'supervisor'}",
                   {"phase_id": phase_id, "worker_by": worker_by})
    return True, f"phase {phase_id} COMPLETE (evidence {len(evidence or [])} items)"


def phase_blocked(mission_id: str, phase_id: str, *,
                  worker_by: str = "", note: str = "") -> tuple:
    """BLOCKED is a documented branch: the phase was attempted, a genuine
    blocker recorded with note. BLOCKED never auto-completes the mission; it
    only lets the mission continue with remaining phases (a BLOCKED phase
    still counts as resolved-for-continuation only if explicitly allowed)."""
    m = load_mission(mission_id)
    if m is None:
        return False, f"no mission {mission_id}"
    p = _find_phase(m, phase_id)
    if p is None:
        return False, f"no phase {phase_id}"
    p["status"] = "BLOCKED"
    p["blocker_note"] = note
    p["worker_by"] = worker_by
    p["updated_at"] = time.time()
    save_mission(m)
    _mission_event(mission_id, "MEDIUM", "PHASE_BLOCKED",
                   f"phase {phase_id} blocked: {(note or '')[:120]}",
                   {"phase_id": phase_id, "worker_by": worker_by, "note": note})
    return True, f"phase {phase_id} marked BLOCKED; mission continues with remaining work"


def phase_failed(mission_id: str, phase_id: str, *, worker_by: str = "",
                 note: str = "", retryable: bool = False) -> tuple:
    """Mark FAILED. retryable=True means it was an infra-style failure
    (worker crash/timeout/network) — the orchestrator may retry it up to
    _PHASE_MAX_RETRIES; retryable=False means evidence/code failure (do not
    automatically re-attempt)."""
    m = load_mission(mission_id)
    if m is None:
        return False, f"no mission {mission_id}"
    p = _find_phase(m, phase_id)
    if p is None:
        return False, f"no phase {phase_id}"
    p["status"] = "FAILED"
    p["blocker"] = note
    p["retryable"] = bool(retryable)
    p["updated_at"] = time.time()
    save_mission(m)
    _mission_event(mission_id, "MEDIUM", "PHASE_FAILED",
                   f"phase {phase_id} failed: {(note or '')[:120]}",
                   {"phase_id": phase_id, "worker_by": worker_by,
                    "note": note, "retryable": bool(retryable)})
    return True, f"phase {phase_id} marked FAILED"


def _recompute_criteria(m: dict) -> None:
    """Requirements are satisfied when the criterion string appears in ANY
    phase evidence (deterministic). A worker's claim 'done' never satisfies a
    requirement by itself."""
    all_evidence = " ".join(
        str(e).lower()
        for p in m.get("phases", [])
        for e in p.get("evidence", []))
    m["criteria_met"] = [c for c in (m.get("requirements") or [])
                         if c.lower() in all_evidence]


# ---------------------------------------------------------------------------
# Unresolved findings
# ---------------------------------------------------------------------------

def add_finding(mission_id: str, finding_id: str, text: str) -> bool:
    """Record an unresolved finding (e.g. a limitation a worker named but did
    not close). OPEN findings keep the mission ACTIVE."""
    m = load_mission(mission_id)
    if m is None:
        return False
    if any(f.get("id") == finding_id
           for f in m.get("unresolved_findings", [])):
        return False
    m.setdefault("unresolved_findings", []).append({
        "id": finding_id, "text": text, "status": "OPEN",
        "added_at": time.time(), "evidence": [],
    })
    if len(m["unresolved_findings"]) > _MISSION_MAX_FINDINGS:
        m["unresolved_findings"] = m["unresolved_findings"][-_MISSION_MAX_FINDINGS:]
    save_mission(m)
    _mission_event(mission_id, "MEDIUM", "FINDING_ADDED",
                   f"finding {finding_id}: {(text or '')[:120]}",
                   {"finding_id": finding_id})
    return True


def resolve_finding(mission_id: str, finding_id: str, *, evidence) -> bool:
    """Finding is RESOLVED only when evidence is supplied (claim != proof)."""
    m = load_mission(mission_id)
    if m is None:
        return False
    for f in m.get("unresolved_findings", []):
        if f.get("id") == finding_id:
            if not evidence:
                return False
            f["status"] = "RESOLVED"
            f["evidence"] = list(evidence)
            f["resolved_at"] = time.time()
            save_mission(m)
            return True
    return False


def block_finding(mission_id: str, finding_id: str, *, note: str = "") -> bool:
    """BLOCKED finding: investigated with evidence, genuinely unexplainable in
    this environment. A documented BLOCKED finding does not keep the mission
    alive forever (unlike OPEN); the mission may finish with it recorded."""
    m = load_mission(mission_id)
    if m is None:
        return False
    for f in m.get("unresolved_findings", []):
        if f.get("id") == finding_id:
            f["status"] = "BLOCKED"
            f["blocker"] = note
            f["blocked_at"] = time.time()
            save_mission(m)
            return True
    return False


# ---------------------------------------------------------------------------
# Deterministic next-action
# ---------------------------------------------------------------------------

def next_phase(mission_id: str) -> Optional[dict]:
    """First PENDING phase. Phases with an 'after' dependency become eligible
    only when the dependency is COMPLETE. This is the continuation engine:
    when a worker finishes, the supervisor calls this to create the NEXT task
    automatically."""
    m = load_mission(mission_id)
    if m is None:
        return None
    for p in m.get("phases", []):
        if p.get("status") != "PENDING":
            continue
        if p.get("after"):
            dep = _find_phase(m, p["after"])
            if dep is None or dep.get("status") != "COMPLETE":
                continue
        return p
    return None


def begin_phase_retry(mission_id: str, phase_id: str) -> tuple:
    """Consume one retry budget: FAILED(retryable) -> ACTIVE, retry_count+1.
    The orchestrator then creates/finds the worker task for the phase as it
    does for any ACTIVE phase (resume semantics reuse existing worker_task
    binding when present)."""
    m = load_mission(mission_id)
    if m is None:
        return False, "no mission"
    p = _find_phase(m, phase_id)
    if p is None:
        return False, "no phase"
    if not p.get("retryable"):
        return False, "phase not retryable"
    if int(p.get("retry_count") or 0) >= _MISSION_MAX_RETRIES:
        return False, "retry budget exhausted"
    p["retry_count"] = int(p.get("retry_count") or 0) + 1
    p["status"] = "ACTIVE"
    p.pop("worker_task", None)          # fresh task binding on next spawn
    p["retryable"] = False              # one-shot; further death -> manual
    p["updated_at"] = time.time()
    save_mission(m)
    return True, f"retry {p['retry_count']} started for {phase_id}"


def retryable_failed_phase(mission_id: str,
                         max_retries: int = _MISSION_MAX_RETRIES
                         ) -> Optional[dict]:
    """First FAILED phase marked retryable (infra failure) with retry budget
    left. The orchestrator picks it so an infra-killed worker gets a fresh
    bounded attempt instead of the mission dying on FAILED semantics."""
    m = load_mission(mission_id)
    if m is None:
        return None
    cand = [p for p in m.get("phases", [])
            if p.get("status") == "FAILED" and p.get("retryable")
            and int(p.get("retry_count") or 0) < max_retries]
    return sorted(cand, key=lambda p: p.get("updated_at") or 0)[0] if cand else None


def active_phase(mission_id: str) -> Optional[dict]:
    """The phase currently ACTIVE (worker running), if any. The orchestrator
    resumes THIS phase before starting a new one; without this, a restart
    mid-phase would spawn the NEXT phase while the prior worker is still
    active (duplicate-phase attack, found by process-level restart test)."""
    m = load_mission(mission_id)
    if m is None:
        return None
    for p in m.get("phases", []):
        if p.get("status") == "ACTIVE":
            return p
    return None


def mission_status(mission_id: str) -> dict:
    """Deterministic mission state, no LLM. MISSION_COMPLETE requires ALL of:
      - every phase is COMPLETE or BLOCKED-with-note; a FAILED/PENDING/ACTIVE
        phase keeps the mission ACTIVE (FAILED = unresolved work; a phase can
        be replaced/retried — it must NOT close the terminal. Measured P-14
        run: a fake-complete worker marked the phase FAILED and the mission
        incorrectly declared MISSION_COMPLETE)
      - every requirement satisfied by phase evidence
      - no unresolved OPEN finding remains (RESOLVED/BLOCKED allowed)
    Worker claims can mark a phase only through the evidence gates above.
    A CORRUPT ledger (file exists but is not parseable JSON) is reported as
    MISSION_CORRUPT — distinct from MISSION_MISSING, so an autonomous
    controller can react instead of mistaking corruption for "never ran"
    (P-26 s2 live-fix re-applied 2026-08-18)."""
    p = mission_path(mission_id)
    if p.exists():
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"status": "MISSION_CORRUPT", "mission_id": mission_id,
                    "rationale": {"error": f"mission ledger {p} exists but is "
                                           f"not parseable JSON (torn/corrupt)"}}
    m = load_mission(mission_id)
    if m is None:
        return {"status": "MISSION_MISSING", "mission_id": mission_id,
                "rationale": {"error": f"no mission {mission_id}"}}
    phases = m.get("phases", [])
    pending = [p["phase_id"] for p in phases
               if p.get("status") in ("PENDING", "ACTIVE", "FAILED")]
    blocked = [p["phase_id"] for p in phases
               if p.get("status") == "BLOCKED"]
    completed = [p["phase_id"] for p in phases
                 if p.get("status") == "COMPLETE"]
    open_findings = [f["id"] for f in m.get("unresolved_findings", [])
                     if f.get("status") == "OPEN"]
    unmet = [c for c in (m.get("requirements") or [])
             if c not in (m.get("criteria_met") or [])]

    if pending:
        status, reason = "MISSION_ACTIVE", f"phases open: {pending}"
    elif unmet:
        status, reason = "MISSION_ACTIVE", f"requirements unmet: {unmet}"
    elif open_findings:
        status, reason = "MISSION_ACTIVE", f"open findings: {open_findings}"
    else:
        status, reason = "MISSION_COMPLETE", (
            "all required phases closed; requirements met; no open findings")

    rationale = {
        "mission_id": mission_id,
        "objective": m.get("objective"),
        "status": status,
        "reason": reason,
        "phases": {p["phase_id"]: p["status"] for p in phases},
        "completed": completed,
        "blocked_failed": blocked,
        "requirements": {"required": m.get("requirements", []),
                         "met": m.get("criteria_met", [])},
        "open_findings": open_findings,
    }
    m["status"] = status
    m["terminal_rationale"] = rationale
    save_mission(m)
    return rationale


# ---------------------------------------------------------------------------
# Post-SUCCESS adversarial spot-audit (2026-08-23 grill Q8). One LLM pass
# issuing SUCCESS was mono-judge trust over a layer no human can interrogate.
# Before a mission terminalizes, an independent worker re-checks the ledger:
# evidence artifacts are probed for fabrication shape (empty commits, zero-
# byte proof files); an independent hermes chat -q auditor attacks the rest.
# Fail-closed: any error = REJECT (mission stays ACTIVE), never silent pass.
# Deterministic failures short-circuit before the LLM call (fast + offline).
# ---------------------------------------------------------------------------

def _commit_touches_files(ref: str, workdir: Optional[str]) -> bool:
    """True iff <ref> is a git object whose commit changes >= 1 file.
    Empty commits (--allow-empty) return False; unknown refs return False."""
    try:
        wd = workdir or os.getcwd()
        r = subprocess.run(["git", "-C", wd, "show", "--name-only",
                            "--format=%H", ref],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return False
        files = [ln for ln in r.stdout.splitlines()[1:] if ln.strip()]
        return bool(files)
    except Exception:
        return False


def _parse_audit_verdict(raw: str) -> tuple:
    """Parse the auditor's verdict from raw `hermes chat -q` output.

    Real output shape (measured 2026-08-23): config warnings, a "Query:"
    echo of the prompt, a reasoning panel, the reply, then a resume footer
    that re-echoes the prompt. Echoes mention CONFIRM/REJECT inline but the
    verdict stands alone on its own line. Policy: the LAST standalone token
    line wins; NO token line anywhere = fail-closed REJECT.
    Returns (rejected: bool, parsed: bool)."""
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    clean = ansi.sub("", raw or "")
    # Machine marker first: last 'VERDICT: <tok>' line is the contract.
    marked = re.findall(r"^[ \t>]*VERDICT:[ \t]*(REJECT|CONFIRM)\b",
                        clean.upper(), re.M)
    if marked:
        return marked[-1] == "REJECT", True
    # Fallback: last standalone token line (legacy prompts, prose replies).
    up = clean.upper()
    toks = re.findall(r"^[ \t]*(?:\*{0,2})(REJECT|CONFIRM)\b.*$", up,
                      re.M)
    if not toks:
        return True, False
    return toks[-1] == "REJECT", True


def spot_audit_mission(mission_id: str, *, workdir: Optional[str] = None,
                       timeout: int = 600,
                       hermes_bin: str = "hermes") -> dict:
    """Adversarial completion audit. CONFIRM lets the mission close; REJECT
    appends a HIGH open finding so mission_status() keeps it ACTIVE.
    Returns {verdict, reason, findings}."""
    m = load_mission(mission_id)
    if m is None:
        return {"verdict": "REJECT", "reason": "mission ledger missing",
                "findings": []}
    phases = [p for p in m.get("phases", []) if p.get("status") == "COMPLETE"]
    if not phases:
        return {"verdict": "CONFIRM", "reason": "no completed phases to audit",
                "findings": []}
    wd = workdir or m.get("workdir") or None
    checks = []
    # Probe 1 - commits named as evidence must exist AND be non-empty.
    bad_commits = []
    for p in phases:
        for line in (p.get("evidence") or []):
            for tok in _COMMIT_REF_RE.findall(str(line)):
                if not _commit_touches_files(tok, wd):
                    bad_commits.append(tok)
    if bad_commits:
        checks.append("empty/invalid commit refs in evidence: "
                      f"{sorted(set(bad_commits))[:5]}")
    # Probe 2 - evidence files must exist and be non-empty.
    empty_files = []
    for p in phases:
        for line in (p.get("evidence") or []):
            for tok in _FILE_RE.findall(str(line)):
                fpath = os.path.join(wd or ".", tok)
                try:
                    if os.path.isfile(fpath) and os.path.getsize(fpath) == 0:
                        empty_files.append(tok)
                except OSError:
                    pass
    if empty_files:
        checks.append(f"zero-byte evidence files: {sorted(set(empty_files))[:5]}")
    # Deterministic failure short-circuits: no LLM spend needed to reject.
    if checks:
        why = "; ".join(checks)
        fid = f"spot-audit-{int(time.time())}"
        unres = m.setdefault("unresolved_findings", [])
        unres.append({"id": fid, "status": "OPEN", "severity": "HIGH",
                      "note": why})
        save_mission(m)
        from hermes_cli.mission_ops import log_event
        log_event(mission_id, "HIGH", "SPOT_AUDIT_REJECTED", why)
        return {"verdict": "REJECT", "reason": why, "findings": [fid]}
    # Offline test seam: MISSION_DRY_WORKER=1 skips the LLM stage entirely
    # (deterministic probes above still enforce). Legacy loop tests stay
    # offline; production audits always run the adversarial pass.
    if os.environ.get("MISSION_DRY_WORKER") == "1":
        return {"verdict": "CONFIRM",
                "reason": "dry-worker mode: LLM audit stage skipped",
                "findings": []}
    # Adversarial reviewer - one independent hermes chat -q pass.
    try:
        ledger = json.dumps(m, default=str)[:6000]
    except Exception:
        ledger = ""
    prompt = (
        "You are an ADVERSARIAL auditor. This autonomous mission declares "
        "completion. Attack it: find any way the completion is fabricated, "
        "incomplete, or self-graded too generously (tests claimed but not "
        "runnable, files that assert success without doing the work, "
        "requirements satisfied by wording alone).\n\n"
        f"OBJECTIVE: {m.get('objective')}\nREQUIREMENTS: "
        f"{m.get('requirements')}\nMISSION LEDGER: {ledger}\n\n"
        "End your reply with a final line of exactly 'VERDICT: CONFIRM' or "
        "'VERDICT: REJECT'.")
    try:
        r = subprocess.run([hermes_bin, "chat", "--max-turns", "8",
                            "-Q", "-q", prompt],
                           capture_output=True, text=True, timeout=timeout)
        verdict_line = ((r.stdout or "") + (r.stderr or "")).strip()
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "REJECT", "reason": f"auditor failed: {exc}",
                "findings": []}
    # Persist the raw audit transcript next to the ledger: an audit that
    # keeps no evidence of itself cannot be audited.
    try:
        audit_dir = mission_path(mission_id).parent
        (audit_dir / f"spot-audit-{int(time.time())}.txt").write_text(
            verdict_line, encoding="utf-8")
    except Exception:
        pass
    rejected, parsed = _parse_audit_verdict(verdict_line)
    if not parsed:
        verdict_line = ("auditor gave no parseable verdict: "
                        + (verdict_line or "")[:200])
    fid = None
    if rejected:
        fid = f"spot-audit-{int(time.time())}"
        # Quote the defect line (the REJECT line + following text), never
        # the CLI banner that precedes the reply.
        clean = re.sub(r"\x1b\[[0-9;]*m", "", verdict_line)
        defect = ""
        lines_ = clean.splitlines()
        # Marker line wins; else LAST standalone REJECT line (earlier ones
        # can be the CLI's column-wrapped echo of our own prompt).
        for i_ in range(len(lines_) - 1, -1, -1):
            up_ = lines_[i_].upper()
            if (re.match(r"^[ \t>]*VERDICT:[ \t]*REJECT\b", up_)
                    or re.match(r"^[ \t]*(?:\*{0,2})REJECT\b", up_)):
                defect = "\n".join(lines_[i_:i_ + 4]).strip()
                break
        if not defect:
            defect = ("no parseable verdict from auditor "
                      "(fail-closed); transcript in mission dir")
        unres = m.setdefault("unresolved_findings", [])
        unres.append({"id": fid, "status": "OPEN", "severity": "HIGH",
                      "note": defect[:400]})
        save_mission(m)
        from hermes_cli.mission_ops import log_event
        log_event(mission_id, "HIGH", "SPOT_AUDIT_REJECTED", defect[:400])
    else:
        from hermes_cli.mission_ops import log_event
        log_event(mission_id, "MEDIUM", "SPOT_AUDIT_CONFIRMED",
                  verdict_line[:160])
    return {"verdict": "REJECT" if rejected else "CONFIRM",
            "reason": verdict_line[:200],
            "findings": [fid] if fid else []}
