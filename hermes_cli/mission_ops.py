"""Dynamic mission: discoveries, TODO graph, completion evaluator, DoD,
lessons, environment/capability models, forensics, metrics.

Extends the P-14 mission ledger (phases + continuation + evidence gates)
with a living work graph: anything a worker/verification/research/adversary
discovers becomes a disclosed candidate; a planning step promotes/defers/
rejects/disembarks; the completion evaluator (not a worker claim) decides
when a mission is genuinely done.

Reuse-first: lessons route to agent/experience.py when possible; session
history/skills are existing Hermes surfaces, only listed here, never
reimplemented.

States (per mandate): DISCOVERED TODO IN_PROGRESS BLOCKED COMPLETE REJECTED
DUPLICATE DEFERRED.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---- discovery lifecycle states ------------------------------------------
DISCOVERY_STATES = (
    "DISCOVERED", "TODO", "IN_PROGRESS", "BLOCKED", "COMPLETE",
    "REJECTED", "DUPLICATE", "DEFERRED", "SUPERSEDED",
)
# states that must keep a mission from declaring MISSION_COMPLETE:
BLOCKING_STATES = ("DISCOVERED", "TODO", "IN_PROGRESS", "BLOCKED")

DEFAULT_PRIORITY = 3  # 1=critical … 5=cosmetic


# ---------------------------------------------------------------------------
# environment / capability model (discoverable facts, refreshable)
# ---------------------------------------------------------------------------

def environment_snapshot() -> Dict[str, Any]:
    """Machine-readable snapshot of this execution environment. Facts are
    labelled; assumptions carry the '_assumption' flag so consumers can
    distinguish evidence from guess."""
    facts: Dict[str, Any] = {}
    try:
        import platform
        facts["platform"] = platform.system()
        facts["release"] = platform.release()
        facts["machine"] = platform.machine()
        facts["python"] = platform.python_version()
    except Exception:
        facts["platform"] = "unknown"
    facts["cpus"] = os.cpu_count() or 1
    facts["cpus_assumption"] = "os.cpu_count could over-estimate affinity"
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    facts["mem_total_kb"] = int(line.split()[1])
                    break
    except Exception:
        facts["mem_total_kb"] = None
    facts["home"] = os.path.expanduser("~")
    facts["supsupervisor_dir"] = os.environ.get(
        "HERMES_SUPERVISOR_DIR", "~/.hermes-supervisor(default)")
    return facts


DEFAULT_CAPABILITIES = [
    {"name": "crawl",
     "responsibility": "internet research / extraction / crawl",
     "when": ["research", "search", "web", "internet", "webpage", "fetch",
              "extract", "crawl", "link", "scrape"]},
    {"name": "diagnostics",
     "responsibility": "system diagnosis, error analysis",
     "when": ["diagnose", "error", "failed", "traceback", "crash", "log",
                "diag"]},
    {"name": "browser",
     "responsibility": "interactive page rendering / JS walls",
     "when": ["browser", "javascript", "render", "playwright", "page",
                "click", "wall", "js"]},
    {"name": "worker/delegation",
     "responsibility": "independent investigation, parallelism",
     "when": ["investigate", "probe", "independently", "another worker",
                "concurrent", "parallel", "subagent"]},
    {"name": "mission",
     "responsibility": "persistent multi-phase objectives, supervision",
     "when": ["mission", "campaign", "phase", "long-running", "autonomous",
                "supervise"]},
    {"name": "session/history",
     "responsibility": "prior conversation, decisions, lessons",
     "when": ["previously", "learned", "history", "session", "last time",
                "previous"]},
    {"name": "lessons/experience",
     "responsibility": "persistent mistake/lesson memory",
     "when": ["lesson", "mistake", "experience", "better approach",
                "retrospective"]},
]

def capability_router(query: str, env=None) -> List[Dict[str, Any]]:
    """Recommend the best Hermes capability for a task string.

    P-26 r15: backed by the LIVE inventory (installed skills, plugins, MCP
    servers, python backends) via environment.probe_env — not a static
    keyword table. Each candidate carries `available: bool` so the planner
    can distinguish "use this" from "this exists but is NOT installed here;
    adapt". Keyword scoring still ranks; availability is factual.
    """
    from hermes_cli import environment as ENV
    env = env if env is not None else ENV.probe_env()
    ql = query.lower()
    skills = set(env.get("skills") or [])
    plugins = set(env.get("plugins") or [])
    mcp = set(env.get("mcp_servers") or [])
    caps = env.get("capabilities") or {}
    scored: List[Dict[str, Any]] = []
    for cap in DEFAULT_CAPABILITIES:
        score = sum(1 for w in cap["when"] if w in ql)
        if score:
            scored.append({"cap": cap["name"], "score": score,
                           "why": [w for w in cap["when"] if w in ql][:4],
                           "available": True})
    # live inventory candidates (skills, plugins, MCP server names)
    for name in skills:
        lname = name.lower()
        if any(w in lname for w in ("crawl", "research", "web", "browser", "pdf",
                                    "ocr", "document", "design", "code", "test")):
            scored.append({"cap": f"skill:{name}", "score": 1,
                           "why": ["installed skill matches domain"],
                           "available": True, "kind": "skill"})
    for name in plugins:
        scored.append({"cap": f"plugin:{name}", "score": 1,
                       "why": ["installed plugin"], "available": True,
                       "kind": "plugin"})
    for name in mcp:
        scored.append({"cap": f"mcp:{name}", "score": 1,
                       "why": ["configured MCP server"], "available": True,
                       "kind": "mcp"})
    # backends that the task needs but that are NOT present here — surface as
    # unavailable so the planner/brief adapts instead of assuming.
    need_backends = _needed_backends(ql)
    for b in need_backends:
        avail = bool(caps.get(b))
        scored.append({"cap": b, "score": 2, "why": ["task implies backend"],
                       "available": avail, "kind": "backend"})
    scored.sort(key=lambda x: (-x["score"], x["available"]))
    return scored


def _needed_backends(ql: str) -> List[str]:
    need = []
    if any(w in ql for w in ("crawl", "scrape", "fetch", "html", "research", "web")):
        need += ["httpx", "trafilatura", "scrapling", "playwright", "curl_cffi"]
    if any(w in ql for w in ("pdf", "ocr", "scan", "document")):
        need += ["pymupdf", "pdfminer"]
    return need


def capability_inventory() -> Dict[str, Any]:
    """Live inventory of what this runtime can actually do. Never asserts a
    capability exists without probing (skills/plugins/MCP from disk+config,
    backends from import availability)."""
    from hermes_cli import environment as ENV
    env = ENV.probe_env()
    return {
        "capabilities": [c["name"] for c in DEFAULT_CAPABILITIES],
        "skills": env.get("skills") or [],
        "plugins": env.get("plugins") or [],
        "mcp": env.get("mcp_servers") or [],
        "backends": {k: bool(v) for k, v in (env.get("capabilities") or {}).items()},
        "cpus": env.get("cpus_available") or env.get("cpus_logical") or None,
        "mem_total_kb": env.get("mem_total_kb"),
        "note": "inventory is live; availability is factual; routing remains judgement",
    }


# ---------------------------------------------------------------------------
# discovery graph
# ---------------------------------------------------------------------------

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-") or "task"


def add_discovery(mission: Dict[str, Any], *, title: str,
                  rationale: str = "", discoverer: str = "",
                  evidence: Optional[List[str]] = None,
                  priority: int = DEFAULT_PRIORITY,
                  deps: Optional[List[str]] = None,
                  supersedes: str = "") -> Dict[str, Any]:
    """Record candidate work. Always DISCOVERED; the planner decides later.
    Duplicate by slug is folded to DUPLICATE with a link when content
    differs; same-slug exists returns existing.

    When `supersedes` names an existing discovery id, that discovery is
    marked SUPERSEDED (removed from blocking work) and linked back, so new
    information can invalidate the original plan rather than appending next
    to it."""
    ds = mission.setdefault("discoveries", [])
    slug = slugify(title)
    for d in ds:
        if d.get("id") == slug:
            return d
    prev_id = (supersedes or "").strip()
    if prev_id and prev_id != slug:
        for d in ds:
            if d.get("id") == prev_id and d.get("status") in BLOCKING_STATES:
                d["status"] = "SUPERSEDED"
                d["superseded_by"] = slug
                d["updated_at"] = time.time()
    d: Dict[str, Any] = {
        "id": slug,
        "title": title,
        "rationale": rationale,
        "discoverer": discoverer or "agent",
        "evidence": list(evidence or []),
        "priority": int(priority),
        "status": "DISCOVERED",
        "deps": deps or [],
        "supersedes": prev_id or "",
        "attempted": [],                 # list of {approach, outcome, at}
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    ds.append(d)
    return d


def promote_discovery(mission, did, status="TODO", note="") -> bool:
    for d in mission.get("discoveries", []):
        if d.get("id") == did:
            if status not in DISCOVERY_STATES:
                return False
            d["status"] = status
            d["updated_at"] = time.time()
            if note:
                d["note"] = note
            return True
    return False


def record_attempt(mission, did: str, strategy: str, outcome: str,
                   cached_note: str = "") -> bool:
    """Append an attempt to a discovery's attempt log. 'cached_note' is a
    shortstring captured before mutation so the caller never re-reads."""
    for d in mission.get("discoveries", []):
        if d.get("id") == did:
            d.setdefault("attempts", []).append(
                {"strategy": strategy, "outcome": outcome, "note": cached_note,
                 "at": time.time()})
            d["updated_at"] = time.time()
            return True
    return False


def open_blocking_discoveries(mission) -> List[Dict[str, Any]]:
    return [d for d in mission.get("discoveries", [])
            if d.get("status") in BLOCKING_STATES]


def plan_discoveries(mission, *, max_keep: int = 12) -> Tuple[int, str]:
    """planner simplification: promote DISCOVERED with low priority (1-3) to
    TODO; reject or defer DISCOVERED with priority >3 unless evidence exists.
    Deterministic, no LLM in the hot path."""
    promoted = 0
    notes = []
    for d in mission.get("discoveries", []):
        if d.get("status") != "DISCOVERED":
            continue
        p = int(d.get("priority", DEFAULT_PRIORITY))
        if p <= 3 or d.get("evidence"):
            d["status"] = "TODO"
            promoted += 1
            notes.append(f"todo:{d['id']}")
        else:
            d["status"] = "DEFERRED"
            notes.append(f"deferred:{d['id']}")
        d["updated_at"] = time.time()
    return promoted, "; ".join(notes)


# ---------------------------------------------------------------------------
# capability backlog (durable continuation source beyond phases/discoveries)
# ---------------------------------------------------------------------------
# The failure mode this fixes: an agent says "the next high-value capability
# is X" and then stops — because X lives only in the session, never in the
# ledger, so the evaluator sees an "empty" mission and ends it. The backlog is
# the DURABLE home for that statement. Open backlog items block completion
# (like discoveries), and the loop/controller materialize them on demand, so
# "tests pass / tree clean" can never complete a mission whose backlog has
# high-value work left.

BACKLOG_STATES = ("OPEN", "MATERIALIZED", "COVERED", "DONE", "SKIPPED",
                  "REJECTED")
# item states that must keep a mission from completing:
BACKLOG_BLOCKING = ("OPEN",)


def backlog_add(mission: Dict[str, Any], *, item_id: str, title: str = "",
                priority: int = DEFAULT_PRIORITY, why: str = "",
                evidence: Optional[List[str]] = None) -> Dict[str, Any]:
    """Record durable known-but-unplanned work. Idempotent by item_id."""
    items = mission.setdefault("backlog", [])
    for it in items:
        if it.get("id") == item_id:
            return it
    it = {
        "id": item_id,
        "title": title or item_id,
        "priority": int(priority),
        "why": why,
        "evidence": list(evidence or []),
        "status": "OPEN",
        "discovery_id": "",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    items.append(it)
    return it


def backlog_open(mission: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [it for it in mission.get("backlog", [])
            if it.get("status") in BACKLOG_BLOCKING]


def backlog_materialize(mission) -> Tuple[int, str]:
    """Turn the highest-priority OPEN backlog item into a discovery.

    The planner then promotes evidence-backed/priority<=3 candidates to TODO
    and the supervisor's existing pool executes them — the agent never had
    to be present for the transition. Already-covered items become COVERED
    (their discovery, not the item, carries the blocking state).
    Returns (created_count, notes).
    """
    created = 0
    notes = []
    for it in sorted(backlog_open(mission),
                     key=lambda x: (int(x.get("priority", 5)),
                                    x.get("created_at") or 0)):
        # an existing discovery with the same slug covers this item
        slug = slugify(it.get("title") or it.get("id") or "")
        existing = next((d for d in mission.get("discoveries", [])
                         if d.get("id") == slug), None)
        if existing is not None:
            it["status"] = "COVERED"
            it["discovery_id"] = existing.get("id", "")
            it["updated_at"] = time.time()
            notes.append(f"covered:{it['id']}")
            continue
        d = add_discovery(
            mission,
            title=it.get("title") or it.get("id") or "",
            rationale=f"backlog[{it['id']}]: {it.get('why') or ''}",
            discoverer=f"backlog:{it['id']}",
            evidence=(list(it.get("evidence") or []) +
                      [f"backlog item {it['id']}"]),
            priority=int(it.get("priority", DEFAULT_PRIORITY)))
        it["status"] = "MATERIALIZED"
        it["discovery_id"] = d.get("id", "")
        it["updated_at"] = time.time()
        created += 1
        notes.append(f"materialized:{it['id']}->{d.get('id')}")
    return created, "; ".join(notes)


def backlog_status(mission) -> Dict[str, Any]:
    items = mission.get("backlog", [])
    return {
        "total": len(items),
        "open": len(backlog_open(mission)),
        "states": {s: sum(1 for it in items if it.get("status") == s)
                   for s in BACKLOG_STATES},
    }


# ---------------------------------------------------------------------------
# definition of done + completion evaluator
# ---------------------------------------------------------------------------

DOD_DIMENSIONS = [
    ("known-bugs", "Known bugs are fixed or have documented evidence"),
    ("hidden-bugs", "Hidden/edge failures were probed (attack tests exist)"),
    ("dead-code", "Dead/duplicated code is reviewed"),
    ("architecture", "Architectural weaknesses are reviewed"),
    ("missing-tests", "Changed behavior has tests"),
    ("reliability", "Reliability/error handling are exercised"),
    ("capability-use", "Existing Hermes capabilities were considered"),
    ("adversarial", "A genuine independent attack found nothing new"),
]


def make_dod() -> List[Dict[str, Any]]:
    return [{"id": k, "label": v, "status": "PENDING", "evidence": []}
            for k, v in DOD_DIMENSIONS]


def derive_dod(objective: str = "", workdir: str = "") -> List[Dict[str, Any]]:
    """Outcome-derived Definition of Done (P-26 r15): derive relevant
    completion criteria from the ACTUAL project shape when a workdir is
    given, instead of only the static dimensions.

    Keeps the evidence gate: every derived dimension flows through the same
    dod_satisfy() floor (>=40 chars AND a verification marker). The derived
    list is capped and 'optional' dimensions never block completion — the
    evaluator treats only dimension ids in the required set as blocking.
    """
    derived: List[Dict[str, Any]] = []
    p = Path(workdir) if workdir else None
    if p is None or not p.is_dir():
        return derived
    try:
        entries = [x.name for x in p.iterdir() if x.is_dir()]
    except Exception:
        entries = []
    has_tests = "tests" in entries or "test" in entries
    has_docs_dir = "docs" in entries or "doc" in entries
    has_src = any(e in entries for e in ("src", "app", "lib", "package",
                                          "hermes_cli", "crawl"))
    has_git = (p / ".git").is_dir()
    # project-specific regression criterion: repository has a test suite
    if has_tests:
        derived.append({
            "id": "derived-regression",
            "label": "Repository regression suite runs green (tests/ exists)",
            "status": "PENDING", "evidence": [],
            "derived": True, "optional": False,
            "source": "workdir has tests/",
        })
    # docs/ exists -> docs consistency is a meaningful dimension
    if has_docs_dir:
        derived.append({
            "id": "derived-docs",
            "label": "Documentation reflects changed behavior (docs/ exists)",
            "status": "PENDING", "evidence": [],
            "derived": True, "optional": True,
            "source": "workdir has docs/",
        })
    # objective mentions an existing subsystem -> architecture probe dimension
    if has_src or has_git:
        derived.append({
            "id": "derived-architecture",
            "label": "Architectural impact of the change is reviewed",
            "status": "PENDING", "evidence": [],
            "derived": True, "optional": True,
            "source": "workdir contains source tree",
        })
    # git repo -> change-scope containment is measurable
    if has_git:
        derived.append({
            "id": "derived-scope",
            "label": "Change is scoped (no unrelated files drift)",
            "status": "PENDING", "evidence": [],
            "derived": True, "optional": True,
            "source": "workdir is a git repository",
        })
    return derived[:8]


def install_derived_dod(mission: Dict[str, Any], workdir: str = "") -> int:
    """Merge outcome-derived DoD dimensions into a mission's dod (idempotent
    by dimension id). Returns number of newly installed dimensions."""
    if not mission.get("dod"):
        mission["dod"] = make_dod()
    existing = {it.get("id") for it in mission["dod"]}
    added = 0
    for dim in derive_dod(mission.get("objective", ""), workdir=workdir):
        if dim["id"] not in existing:
            dim["status"], dim["evidence"] = "PENDING", []
            mission["dod"].append(dim)
            added += 1
    return added


_EVIDENCE_FLOOR = 40  # minimal characters of real evidence
_EVIDENCE_MARKERS = ("commit", "test", "suite", "verify", "green", "file",
                     "repro", "run", "evidence", "probe", "log", "trace")


def dod_satisfy(mission, key: str, evidence: str, who: str = "") -> bool:
    """Satisfy a definition-of-done dimension, evidence-gated. A dimension
    only becomes SATISFIED when the evidence is substantive: at least floor
    chars AND contains a verification marker. Weak claims stay PENDING."""
    evidence = (evidence or "").strip()
    low = evidence.lower()
    substantive = (
        len(evidence) >= _EVIDENCE_FLOOR
        and any(mk in low for mk in _EVIDENCE_MARKERS)
    )
    for it in mission.setdefault("dod", []):
        if it.get("id") == key:
            if substantive:
                it["evidence"].append({"who": who, "evidence": evidence,
                                       "at": time.time()})
                it["status"] = "SATISFIED"
            else:
                it["evidence"].append({"who": who, "evidence": evidence,
                                       "at": time.time(), "weak": True})
                it["status"] = "PENDING"
            return substantive
    # unknown dod key -> register with honest status
    mission.setdefault("dod", []).append(
        {"id": key, "name": key, "status": "SATISFIED" if substantive else "PENDING",
         "evidence": [{"who": who, "evidence": evidence, "at": time.time(),
                       "weak": not substantive}]})
    return substantive


def completion_evaluator(mission: Dict[str, Any]) -> Dict[str, Any]:
    """The completion gate: phases closed by the harness AND no blocking
    discovery AND DoD evidence present. Worker claims are never input here;
    only ledger state and evidence lists."""
    from hermes_cli import supervisor as S

    st = S.mission_status(mission["mission_id"])
    missing: List[str] = []

    if st.get("status") != "MISSION_COMPLETE":
        missing.append("phase-termination not reached: " + st.get("reason", "?"))

    # P-26 r13/r15: blocked-by-backlog is a HARD gate — open backlog items
    # are durable high-value statements, never advisory.
    open_bl = backlog_open(mission)
    if open_bl:
        missing.append("open backlog: " + ", ".join(it["id"] for it in open_bl))

    # P-26 r15: only REQUIRED DoD dimensions block completions. Derived
    # 'optional' dimensions (documentation, architecture review) are
    # meaningful to satisfy but must never gate the mission on a nit.
    unsat = [it["id"] for it in mission.get("dod", [])
             if it.get("status") != "SATISFIED"
             and not it.get("optional")]
    if unsat:
        missing.append("dod unsatisfied: " + ", ".join(unsat))

    # P-26 r15: ADVISORY discoveries never block completion. Detector
    # findings are runnable, durable work (the loop executes them when the
    # mission has capacity) but an optional optimization — worker-overlap,
    # semantic stagnation, repeated commands — must not turn the objective
    # into an infinite checklist. Genuine coverage gaps (changed-without-
    # tests) are created NON-advisory and still block.
    blocks = [d for d in open_blocking_discoveries(mission)
              if not d.get("advisory")]
    if blocks:
        missing.append("open discoveries: " + ", ".join(d["id"] for d in blocks))

    complete = not missing
    return {
        "complete": complete,
        "reason": "; ".join(missing) if missing else (
            "objective satisfied (phases+discoveries+dod all hold)"),
        "missing": missing,
    }


def mission_report(mission_id: str) -> Dict[str, Any]:
    """Concise human/agent report: status + evaluator + open TODO."""
    from hermes_cli import supervisor as S
    m = S.load_mission(mission_id)
    if not m:
        return {"error": f"no mission {mission_id}"}
    evaluator = completion_evaluator(m)
    return {
        "mission_id": mission_id,
        "objective": m.get("objective"),
        "evaluator": evaluator,
        "discoveries": {d["id"]: d["status"] for d in m.get("discoveries", [])},
        "dod": {it["id"]: it["status"] for it in m.get("dod", [])},
        "phases": {p["phase_id"]: p["status"] for p in m.get("phases", [])},
    }


# ---------------------------------------------------------------------------
# lessons (mission-scoped JSONL + agent experience bridge)
# ---------------------------------------------------------------------------

def lessons_path(mission_id: str) -> Path:
    base = os.environ.get("HERMES_SUPERVISOR_DIR") or os.path.expanduser(
        "~/.hermes-supervisor")
    d = Path(base) / "missions" / mission_id
    d.mkdir(parents=True, exist_ok=True)
    return d / "lessons.jsonl"


def mission_lesson_sync(mission: Dict[str, Any], *, force: bool = False) -> Tuple[int, str]:
    """Feed a complete(le) mission's experience into the NATIVE Hermes lesson
    store (state.db lessons via agent.experience). Dedupes by summary."""
    from agent.experience import Lesson, validate_and_store
    from hermes_cli import supervisor as S

    mid = mission.get("mission_id", "")
    m = S.load_mission(mid) if mid else mission
    if m is None:
        return 0, "no mission to feed"
    disc = m.get("discoveries", [])
    closed = [d for d in disc if d.get("status") in ("COMPLETE", "BLOCKED")]
    ph = m.get("phases", [])
    done = [p for p in ph if p.get("status") == "COMPLETE"]
    if (not done and not closed) and not force:
        return 0, "no completed phases/discoveries (nothing learned)"
    stored, notes = 0, []
    for d in closed:
        ev = list(d.get("evidence") or [])
        if not ev:
            continue
        detail_bits = []
        for a in d.get("attempts") or []:
            detail_bits.append(f"attempt[{a.get('strategy','')}]: {a.get('outcome','')}")
        detail = " | ".join(filter(None, [d.get("rationale", ""), "; ".join(detail_bits)])) or " ".join(ev)
        summary = d.get("title") or ""
        if len(summary) < 20:
            summary = "Mission development: " + summary
        try:
            lesson = Lesson(
                summary=summary[:180],
                detail=(detail or "")[:400],
                category="workflow",
                domain="hermes-mission",
                source_task=d.get("discoverer") or mid,
                source_outcome="failure" if d.get("status") == "BLOCKED" else "success",
                behavior="planner",
                patterns=list(d.get("deps") or []),
                failure_mode=(d.get("note") or d.get("blocker") or "")[:200],
                evidence=ev,
                confidence=0.6,
            )
            if validate_and_store(lesson, session_id=mid):
                stored += 1
                notes.append(f"lesson:{d.get('id', '')}")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"error:{exc}")
    if not stored and not closed:
        notes.append("no closed discoveries with evidence; nothing stored")
    return stored, "; ".join(notes)


def retrieval_hint(query: str) -> str:
    """Return what the platform lessons store already knows about a task
    objective. Used at mission start so the harness reuses experience instead
    of rediscovering. Empty string = nothing known."""
    try:
        from agent.experience import retrieve_lessons
        lessons = retrieve_lessons(query, max_results=3)
        if not lessons:
            return ""
        return "\n".join(f"- {l.summary} for {l.domain}: {l.failure_mode[:80]}"
                         if l.source_outcome == "failure" else f"- {l.summary}"
                         for l in lessons[:3])
    except Exception:
        return ""


def add_lesson(*, mission_id: str, lesson: str, context: str = "",
               better_approach: str = "", apply_to: str = "",
               evidence: Optional[List[str]] = None) -> bool:
    """Record a durable engineering lesson against a mission. Never required
    for completion; survives sessions."""
    row = {
        "at": time.time(),
        "lesson": lesson,
        "context": context,
        "better_approach": better_approach,
        "apply_to": apply_to,
        "evidence": evidence or [],
    }
    try:
        with open(lessons_path(mission_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return True
    except Exception:
        return False


def recall_lessons(mission_id: str, apply_to: str = "", limit: int = 10) -> List[Dict[str, Any]]:
    p = lessons_path(mission_id)
    if not p.exists():
        return []
    out = []
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if apply_to and apply_to not in (row.get("apply_to") or ""):
                    continue
                out.append(row)
    except Exception:
        pass
    return out[-limit:]


# ---------------------------------------------------------------------------
# terminal stop rationale + failure memory (P-26 r6/r7)
# ---------------------------------------------------------------------------


def record_stop(mission_id: str, *, verdict: str, reason: str,
                evidence: Optional[List[str]] = None) -> bool:
    """Durable record of WHY an autonomous loop stopped. Distinguishes
    'genuinely done / low-value remainder' from 'gave up mid-work', so a
    restarted controller can judge whether stopping was right."""
    from hermes_cli import supervisor as S
    m = S.load_mission(mission_id)
    if m is None:
        return False
    m["terminal_rationale"] = {
        "verdict": verdict, "reason": reason,
        "evidence": list(evidence or []), "at": time.time(),
    }
    S.save_mission(m)
    log_event(mission_id, "HIGH", "MISSION_STOPPED",
              f"loop stopped: {verdict} — {reason[:160]}",
              {"verdict": verdict, "reason": reason[:400]})
    return True


def record_failure_memory(mission_id: str, *, target: str, approach: str,
                          outcome: str, how_to_avoid: str = "") -> bool:
    """Durable failure memory: a failed approach + the dead end it hit, bound
    to the target (discovery id / phase id) so future workers retrieving
    lessons for that target find 'tried, failed, do not repeat'. This is what
    makes repeated failed approaches observable and avoidable."""
    row = {
        "at": time.time(),
        "target": target,
        "approach": approach,
        "outcome": outcome,
        "how_to_avoid": how_to_avoid,
        "evidence": ["recorded at supervisor failure boundary"],
    }
    try:
        from hermes_cli import supervisor as S
        p = S.missions_dir() / mission_id / "failure-memory.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        add_lesson(mission_id=mission_id,
                   lesson=f"[failed] {target}: {approach}",
                   context=f"FAILED — {outcome}",
                   better_approach=how_to_avoid,
                   apply_to=target,
                   evidence=["failure memory boundary"])
        return True
    except Exception:  # noqa: BLE001
        return False


def failure_memory(mission_id: str, target: str = "",
                   limit: int = 10) -> List[Dict[str, Any]]:
    from hermes_cli import supervisor as S
    p = S.missions_dir() / mission_id / "failure-memory.jsonl"
    rows = []
    if not p.exists():
        return rows
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    if target:
        rows = [r for r in rows if target in (r.get("target") or "")]
    return rows[-limit:]


def measure_strategy_outcome(mission: Dict[str, Any], *, task_id: str,
                             elapsed: float):
    """P-26 r12: 'did the optimization actually help?'.

    When a mission has performed a bounded strategy switch, the next
    completed worker's ACTUAL span vs the pre-switch EXPECTED baseline is the
    measured delta. The mechanism demands before->change->after; never
    'I optimized it' without a number. Emits a durable STRATEGY_OUTCOME event
    and a retrievable lesson. elapsed below baseline = helped; above = hurt
    (lesson records both, mission continues)."""
    switches = int(mission.get("strategy_switches") or 0)
    if not switches:
        return None
    mid = mission.get("mission_id") or ""
    exp = expected_worker_span(mid)
    exp_fmt = f"{exp:.1f}s" if exp is not None else "unknown"
    helped = exp is None or elapsed <= (exp * 1.2)
    log_event(
        mid, "MEDIUM", "STRATEGY_OUTCOME",
        (f"worker {task_id} completed in {elapsed:.1f}s vs expected "
         f"{exp_fmt} — strategy 'helped'" if helped else
         f"worker {task_id} completed in {elapsed:.1f}s vs expected "
         f"{exp_fmt} — strategy did NOT help (reassess)"),
        {"task_id": task_id, "elapsed_seconds": round(elapsed, 1),
         "expected_seconds": round(exp or 0, 1),
         "strategy_switches": switches, "helped": helped})
    try:
        add_lesson(
            mission_id=mid,
            lesson=f"[speed] {task_id} {'beat' if helped else 'did not beat'} "
                   f"baseline {exp_fmt} in {elapsed:.1f}s"
                   + ("" if helped else " — alternative strategy needed"),
            context=f"strategy switch #{switches} outcome",
            better_approach=("keep current strategy" if helped else
                             "try another approach; switch budget may be spent"),
            apply_to="speed",
            evidence=[f"elapsed={elapsed:.1f}s "
                      f"expected={exp and round(exp, 1)}s"])
    except Exception:
        pass
    return helped


def continue_check(mission: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic 'may I report done?' gate — the single call an agent,
    controller, or verification session must make before emitting a final
    report. It is NOT a new completion system: it composes the existing
    completion_evaluator + stop_rationale into one verdict with evidence.

    Verdict semantics (for CLI exit-code style use):
      - may_stop=True   -> every gate is genuinely closed AND the remaining
                           backlog/discoveries are at/below the diminishing
                           return threshold (or lifecycle is blocked).
      - may_stop=False  -> meaningful work remains; do NOT write a final
                           report, do NOT declare the objective achieved.

    Gates (each must be closed for a True verdict):
      phases+discoveries+DoD via completion_evaluator
      open backlog high/medium value via stop_rationale
      a controller-critical UNRESOLVED finding
    """
    ev = completion_evaluator(mission)
    stop = stop_rationale(mission)
    why_continue: List[str] = []
    if not ev["complete"]:
        why_continue.extend(ev["missing"])
    if stop.get("verdict") in ("continue", "consider-trimming"):
        bl = stop.get("backlog") or {}
        if bl.get("high_priority") or bl.get("medium_priority"):
            why_continue.append(
                "diminishing-return gate: high/medium-value backlog remains "
                "(high=%s mid=%s)" % (bl.get("high_priority", 0),
                                      bl.get("medium_priority", 0)))
    open_findings = [f for f in mission.get("unresolved_findings", [])
                     if f.get("status") == "OPEN"]
    if open_findings:
        why_continue.append("open findings: %s" %
                            ", ".join(f.get("id", "?")
                                      for f in open_findings[:3]))
    may_stop = not why_continue
    return {
        "may_stop": may_stop,
        "for": why_continue,
        "completion": ev,
        "rationale": stop,
    }


def continue_check_via_cli(mission_id: str) -> Dict[str, Any]:
    """CLI-friendly wrapper: loads the mission, returns the check."""
    from hermes_cli import supervisor as S
    m = S.load_mission(mission_id)
    if m is None:
        return {"error": f"no mission {mission_id}", "may_stop": True}
    return continue_check(m)


# ---------------------------------------------------------------------------
# mission event journal (controller visibility without joining the supervisor)
# ---------------------------------------------------------------------------

LEVEL_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def mission_events_path(mission_id: str) -> Path:
    from hermes_cli import supervisor as _S
    d = _S.missions_dir() / mission_id
    d.mkdir(parents=True, exist_ok=True)
    return d / "events.jsonl"


def log_event(mission_id: str, level: str, kind: str, message: str,
              payload: Optional[Dict[str, Any]] = None) -> bool:
    """Append one durable event. Cheap, best-effort, never blocks the loop."""
    level = level.upper() if level.upper() in LEVEL_ORDER else "LOW"
    row = {"ts": time.time(), "level": level, "kind": kind,
           "message": message, "payload": payload or {}}
    try:
        with open(mission_events_path(mission_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return True
    except OSError:
        return False


def read_events(mission_id: str, min_level: str = "LOW",
                limit: int = 50) -> List[Dict[str, Any]]:
    p = mission_events_path(mission_id)
    if not p.exists:
        return []
    rows = []
    want = LEVEL_ORDER.get(min_level.upper(), 0)
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if LEVEL_ORDER.get(row.get("level", "LOW"), 0) >= want:
                    rows.append(row)
    except OSError:
        pass
    return rows[-limit:]


def mission_events(mission_id: str, level: str = "LOW", limit: int = 50) -> List[Dict[str, Any]]:
    return read_events(mission_id, min_level=level, limit=limit)


def mission_semantic_fingerprint(mission_id: str) -> str:
    """Fingerprint of meaningful mission state: phases, discoveries with
    evidence/attempt counts, criteria. Used for semantic stagnation."""
    from hermes_cli import supervisor as S
    m = S.load_mission(mission_id)
    if m is None:
        return "none"
    parts = []
    for p in m.get("phases", []):
        parts.append(f"{p.get('phase_id')}:{p.get('status')}:{len(p.get('evidence') or [])}")
    for d in m.get("discoveries", []):
        parts.append(f"{d.get('id')}:{d.get('status')}:{len(d.get('evidence') or [])}:{len(d.get('attempts') or [])}")
    parts.append(f"criteria={sorted(m.get('criteria_met') or [])}")
    return "|".join(parts)


def stalled(mission_id: str, *, prev_fp: str, ticks: int,
            threshold: int = 4):
    """Compare semantic fingerprint; (stalled, new_fp)."""
    cur = mission_semantic_fingerprint(mission_id)
    if cur != prev_fp:
        return False, cur
    return ticks >= threshold, cur


def mission_forensics(mission_id: str) -> Dict[str, Any]:
    """Legacy ledger-forensics view (times + state). metrics_report() is the
    richer, component-level performance breakdown."""
    from hermes_cli import supervisor as S
    m = S.load_mission(mission_id)
    if not m:
        return {"error": f"no mission {mission_id}"}
    cre = float(m.get("created_at") or 0)
    ph = [{"phase": p.get("phase_id"), "status": p.get("status"),
           "last_update": p.get("updated_at") or 0}
          for p in m.get("phases", [])]
    disc = m.get("discoveries", [])
    return {
        "mission": mission_id,
        "created_at": cre,
        "now": time.time(),
        "wall_seconds_so_far": (time.time() - cre) if cre else None,
        "phases": ph,
        "discovery_count": len(disc),
        "discovery_states": {d.get("id"): d.get("status") for d in disc},
        "limitation": "timestamps only; use metrics_report for breakdown",
    }


# ---------------------------------------------------------------------------
# metrics (post-mission measurement, derived from durable state only)
# ---------------------------------------------------------------------------

def _event_counts(mission_id: str) -> Dict[str, Any]:
    """Journal level/kind statistics; everything below derived from it."""
    counts: Dict[str, Any] = {"total": 0, "by_level": {}, "by_kind": {}}
    p = mission_events_path(mission_id)
    if not p.exists():
        return counts
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                counts["total"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lvl = row.get("level", "LOW")
                counts["by_level"][lvl] = counts["by_level"].get(lvl, 0) + 1
                kind = row.get("kind", "?")
                counts["by_kind"][kind] = counts["by_kind"].get(kind, 0) + 1
        counts["by_kind"] = dict(sorted(counts["by_kind"].items(),
                                        key=lambda kv: -kv[1]))
        counts["by_level"] = dict(sorted(counts["by_level"].items()))
    except OSError:
        pass
    return counts


def metrics_report(mission_id: str) -> Dict[str, Any]:
    """Post-mission performance forensics (P-26).

    A component-by-component breakdown derived entirely from the durable
    ledger, worker.json, and event journal — no fabricated numbers. Spans
    are wall-clock from ledger timestamps; anything finer is explicitly
    out of scope until worker-level tracing exists.

    Feed the output to a fresh session (or the controller) to decide what
    to tune next: that is the 'measure' step of
    mission -> measure -> diagnose -> improve.
    """
    from hermes_cli import supervisor as S
    m = S.load_mission(mission_id)
    if m is None:
        return {"error": f"no mission {mission_id}"}
    now = time.time()
    cre = float(m.get("created_at") or now)
    base = S._tasks_dir()

    workers = []
    for d in sorted(base.iterdir()) if base.is_dir() else []:
        if not d.name.startswith(f"{mission_id}-"):
            continue
        w = S.load_worker(d.name)
        if not w:
            continue
        created = float(w.get("created_at") or 0)
        updated = float(w.get("updated_at") or w.get("last_activity_at") or 0)
        evidence = w.get("completion_evidence") or []
        workers.append({
            "task_id": d.name,
            "status": w.get("status", ""),
            "phase": w.get("phase", ""),
            "created_at": created,
            "span_seconds": round(now - created, 1) if created else None,
            "transient_seconds": round(updated - created, 1)
                                  if (created and updated) else None,
            "failure_count": len(w.get("failure_timestamps") or []),
            "n_evidence": len(evidence),
            "terminated": w.get("status") in ("COMPLETE", "FAILED",
                                              "CANCELLED", "BLOCKED"),
        })
    workers.sort(key=lambda x: -(x.get("span_seconds") or 0))

    phases = [{"phase_id": p.get("phase_id"), "status": p.get("status"),
               "retry_count": p.get("retry_count") or 0,
               "n_evidence": len(p.get("evidence") or []),
               "worker_by": p.get("worker_by", "")}
              for p in m.get("phases", [])]

    discoveries = [{"id": d.get("id"), "status": d.get("status"),
                    "priority": d.get("priority"),
                    "attempts": len(d.get("attempts") or []),
                    "n_evidence": len(d.get("evidence") or []),
                    "created_at": round(float(d.get("created_at") or 0), 1)}
                   for d in m.get("discoveries", [])]

    active = [w for w in workers if not w["terminated"]]
    bottleneck = None
    if active:
        bottleneck = {"task_id": active[0]["task_id"],
                      "status": active[0]["status"],
                      "span_seconds": active[0]["span_seconds"],
                      "note": "longest non-terminal worker (current wait)"}

    return {
        "mission": mission_id,
        "wall_seconds": round(now - cre, 1) if cre else None,
        "phases": phases,
        "discoveries": discoveries,
        "workers": workers,
        "workers_counts": {"total": len(workers),
                           "active": len(active),
                           "failed": sum(1 for w in workers
                                         if w["status"] == "FAILED")},
        "failures_total": sum(w["failure_count"] for w in workers),
        "events": _event_counts(mission_id),
        "bottleneck": bottleneck,
        "limits": ("spans are ledger-derived wall-clock (worker created -> "
                   "now or last update); sub-step CPU/API/wait attribution "
                   "needs worker-level tracing and is intentionally absent"),
    }


def stop_rationale(mission: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic 'why is this mission done or not' rationale, focused on
    diminishing returns: remaining candidate work vs its priority/evidence.

    Complements completion_evaluator (which answers 'is it complete') with
    'should we keep going'. Includes the durable capability backlog: an open
    high-priority backlog item is real remaining work even when no discovery
    has been materialized yet. Uses only ledger state, no LLM.
    """
    open_work = [d for d in mission.get("discoveries", [])
                 if d.get("status") in ("DISCOVERED", "TODO", "IN_PROGRESS")]
    high = [d for d in open_work if int(d.get("priority", 5)) <= 2]
    mid = [d for d in open_work if int(d.get("priority", 5)) == 3]
    low = [d for d in open_work if int(d.get("priority", 5)) >= 4]
    deferred = [d for d in mission.get("discoveries", [])
                if d.get("status") == "DEFERRED"]
    bl = backlog_open(mission)
    bl_high = [it for it in bl if int(it.get("priority", 5)) <= 2]
    bl_mid = [it for it in bl if int(it.get("priority", 5)) == 3]
    bl_low = [it for it in bl if int(it.get("priority", 5)) >= 4]
    verdict = "continue"
    if not open_work and not bl:
        verdict = "stop-unless-new-information"
    elif not high and not mid and not bl_high and not bl_mid:
        verdict = "stop-low-value-remaining"   # only cosmetic candidates stay
    elif not high and not bl_high and len(low) + len(bl_low) <= 2:
        verdict = "consider-trimming"
    return {
        "verdict": verdict,
        "open": len(open_work),
        "high_priority": len(high),
        "medium_priority": len(mid),
        "low_priority": len(low),
        "deferred": len(deferred),
        "backlog": {"open": len(bl), "high_priority": len(bl_high),
                    "medium_priority": len(bl_mid),
                    "low_priority": len(bl_low)},
        "note": ("verdict uses priority only; an LLM/controller may override "
                 "by adding evidence or reprioritizing a discovery/backlog"),
    }


def expected_worker_span(mission_id: str,
                         percentile: float = 0.5) -> Optional[float]:
    """Median completed/FAILED-worker span from durable telemetry — the
    'how long this kind of work normally takes' expectation. None when no
    comparable completed worker exists yet (first run sets the baseline).
    Time is a diagnostic signal, not a kill switch."""
    rep = telemetry_report(mission_id)
    spans = [w.get("span_seconds") for w in rep.get("workers", [])
             if w.get("span_seconds") is not None
             and w.get("status") in ("COMPLETE", "FAILED")]
    if not spans:
        return None
    spans.sort()
    idx = min(len(spans) - 1, max(0, int(len(spans) * percentile)))
    return spans[idx]


def perf_anomaly(*, elapsed: float, expected: Optional[float],
                 mult: float = 3.0, min_seconds: float = 60.0) -> bool:
    """True when an operation has run far past its comparable baseline.
    A pure predicate so tests exercise it without real time waits."""
    if expected is None:
        return False
    threshold = max(expected * mult, min_seconds)
    return elapsed >= threshold


# ---------------------------------------------------------------------------
# worker activity telemetry (P-26 r6) — analysis over the existing audit rail
# ---------------------------------------------------------------------------


def _worker_audit_rows(task_id: str) -> List[Dict[str, Any]]:
    """Read a worker's bounded audit.jsonl (authoritative, survives restarts)."""
    from hermes_cli import supervisor as S
    p = S._tasks_dir() / task_id / "audit.jsonl"
    rows: List[Dict[str, Any]] = []
    if not p.exists():
        return rows
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return rows


def telemetry_report(mission_id: str) -> Dict[str, Any]:
    """Structured, bounded worker-activity telemetry for a mission.

    Derived ONLY from durable per-worker audit rows + worker ledgers — no
    fabricated sub-step timing. Each worker contributes:
      - birth (spawn) / exit events with timestamps -> span
      - every ledger state-write (status, phase, seq, tests, files)
      - every supervisor command issued (verdict/command)
      - retries (attempt/reason)
    The report aggregates them into per-worker counters and cross-worker
    signals (overlap, idle gaps) the waste detector and controller consume.
    """
    from hermes_cli import supervisor as S
    base = S._tasks_dir()
    workers: List[Dict[str, Any]] = []
    for d in sorted(base.iterdir()) if base.is_dir() else []:
        if not d.name.startswith(f"{mission_id}-"):
            continue
        ledger = S.load_worker(d.name) or {}
        rows = _worker_audit_rows(d.name)
        tel = [r for r in rows if r.get("kind") == "telemetry"]
        starts = [r for r in tel if r.get("t_kind") == "spawn"]
        states = [r for r in tel if r.get("t_kind") == "state"]
        cmds = [r for r in tel if r.get("t_kind") == "command"]
        retries = [r for r in tel if r.get("t_kind") == "retry"]
        exits = [r for r in tel if r.get("t_kind") == "exit"]
        ts_all = [float(r.get("t") or 0) for r in tel if r.get("t")]
        files_seen: List[str] = []
        for s in states:
            for f in (s.get("files_changed") or []):
                files_seen.append(str(f))
        from collections import Counter
        file_counts = Counter(files_seen)
        repeated_files = {f: c for f, c in file_counts.items() if c >= 3}
        cmd_counts = Counter((c.get("command") or "") for c in cmds)
        repeated_cmds = {c: n for c, n in cmd_counts.items() if n >= 3}
        tests_series = [int(s.get("tests_executed") or 0) for s in states]
        tests_repeated = len(states) >= 3 and tests_series == ([tests_series[-1]] * len(states)) and tests_series[-1] > 0
        status = ledger.get("status") or ""
        span_seconds = None
        if starts:
            birth = min(float(s.get("t") or 0) for s in starts)
            end = max([float(e.get("t") or 0) for e in exits] +
                      [float(s.get("t") or 0) for s in states] or [birth])
            span_seconds = round(end - birth, 1) if end >= birth else None
        workers.append({
            "task_id": d.name,
            "status": status,
            "spawns": len(starts),
            "state_writes": len(states),
            "commands": len(cmds),
            "retries": len(retries),
            "exits": exits[-1] if exits else None,
            "span_seconds": span_seconds,
            "tests_executed": (tests_series[-1] if tests_series else None),
            "tests_unchanged": tests_repeated,
            "files_changed_total": len(files_seen),
            "repeated_files": repeated_files,
            "repeated_commands": repeated_cmds,
            "first_ts": min(ts_all) if ts_all else None,
            "last_ts": max(ts_all) if ts_all else None,
        })
    workers.sort(key=lambda w: w.get("first_ts") or 0)

    # cross-worker signals
    overlap_pairs = []
    spans = [(w["task_id"], w.get("first_ts"), w.get("last_ts"))
             for w in workers if w.get("first_ts") and w.get("last_ts")]
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            a, b = spans[i], spans[j]
            if a[1] <= b[2] and b[1] <= a[2]:
                overlap_pairs.append((a[0], b[0]))
    idle_workers = [w["task_id"] for w in workers
                    if w.get("first_ts") and w.get("last_ts") is not None
                    and w.get("status") not in ("COMPLETE", "FAILED", "CANCELLED", "BLOCKED")
                    and w.get("span_seconds") is not None
                    and w["span_seconds"] > 60
                    and (w.get("last_ts") or 0) - (w.get("first_ts") or 0) > 60
                    and w.get("commands", 0) == 0]
    return {
        "mission": mission_id,
        "workers": workers,
        "counts": {"workers": len(workers),
                   "state_writes": sum(w["state_writes"] for w in workers),
                   "commands": sum(w["commands"] for w in workers),
                   "retries": sum(w["retries"] for w in workers),
                   "spawns": sum(w["spawns"] for w in workers)},
        "signals": {
            "overlap": overlap_pairs,
            "idle_workers": idle_workers,
            "repeated_commands": {w["task_id"]: w["repeated_commands"]
                                  for w in workers if w["repeated_commands"]},
            "repeated_files": {w["task_id"]: w["repeated_files"]
                               for w in workers if w["repeated_files"]},
            "tests_unchanged": [w["task_id"] for w in workers
                                if w["tests_unchanged"]],
        },
    }


# ---------------------------------------------------------------------------
# waste analysis + optimization plan (metrics become actionable)
# ---------------------------------------------------------------------------

def _token_sim(a: str, b: str) -> float:
    """Jaccard over lowercased 4+ char word tokens; 0..1."""
    ta = {w for w in re.findall(r"[a-z0-9]{4,}", a.lower())}
    tb = {w for w in re.findall(r"[a-z0-9]{4,}", b.lower())}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def waste_analysis(mission_id: str) -> Dict[str, Any]:
    """Evidence-backed waste signals, derived only from durable state.

    Signals (each with the ledger facts that show it):
      - duplicate_research     similar discovery titles across workers
      - sequential_long_workers: two >=MIN_SPAN workers run serially
      - retries                phase retry_count or worker failure events
      - stagnation             SEMANTIC_STAGNATION events
      - idle_tail              last completed worker ended long before now
    Honest boundaries: tool-level repeat/file re-read/context reconstruction
    are not observable from the ledger and are explicitly excluded.
    """
    from hermes_cli import supervisor as S
    m = S.load_mission(mission_id)
    if m is None:
        return {"error": f"no mission {mission_id}"}
    rep = metrics_report(mission_id)
    sig: List[Dict[str, Any]] = []

    # 1) duplicate research — similar discovery titles
    titles = [(d.get("id", ""), (d.get("title") or ""),
               d.get("discoverer", "")) for d in m.get("discoveries", [])]
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            if _token_sim(titles[i][1], titles[j][1]) >= 0.7:
                sig.append({
                    "name": "duplicate-research",
                    "severity": "medium",
                    "evidence": [f"discovery '{titles[i][0]}' overlaps "
                                 f"'{titles[j][0]}' (title similarity "
                                 f"{round(_token_sim(titles[i][1], titles[j][1]), 2)})"],
                    "suggestion": ("shared research artifact/cache for "
                                   "duplicated topics; dedupe before spawn"),
                })
                break

    # 2) serial long workers — non-overlapping spans for substantive work
    spans = [(w["task_id"], w["created_at"], w["span_seconds"],
              w["terminated"]) for w in rep.get("workers", [])]
    spans = sorted([s for s in spans if (s[2] or 0) >= 120], key=lambda s: s[1])
    for k in range(len(spans) - 1):
        end_k = spans[k][1] + (spans[k][2] or 0)
        if end_k <= (spans[k + 1][1] or 0):
            sig.append({
                "name": "serial-long-workers",
                "severity": "medium",
                "evidence": [f"{spans[k][0]} ended ~{round(end_k, 1)} then "
                             f"{spans[k + 1][0]} started ~"
                             f"{round(spans[k + 1][1], 1)} (no overlap)"],
                "suggestion": ("raise MISSION_MAX_CONCURRENT if these were "
                               "independent; evidence: non-overlap"),
            })

    # 3) retries
    retries = [p for p in rep.get("phases", []) if (p.get("retry_count") or 0) > 0]
    failures = [w for w in rep.get("workers", []) if (w.get("failure_count") or 0) > 0]
    if retries or failures:
        sig.append({
            "name": "excessive-retries",
            "severity": "low",
            "evidence": [f"{len(retries)} phase(s) retried; "
                         f"{len(failures)} worker(s) with failures"
                         + (f" (max {max(w['failure_count'] for w in failures)})"
                            if failures else "")],
            "suggestion": "record failed approach before retrying (failure memory)",
        })

    # 4) stagnation events
    ev = rep.get("events", {}).get("by_kind", {})
    stalls = int(ev.get("SEMANTIC_STAGNATION", 0))
    if stalls:
        sig.append({
            "name": "semantic-stagnation",
            "severity": "medium",
            "evidence": [f"{stalls} SEMANTIC_STAGNATION event(s)"],
            "suggestion": "strategy switch required (see mission optimize)",
        })

    # 5) P-26 r6: telemetry-derived signals (repeated activity, overlap,
    #    idle) — only reported when the durable audit trail actually shows
    #    the pattern, so no fabricated inefficiency.
    try:
        tel = telemetry_report(mission_id)
    except Exception:
        tel = None
    if tel:
        sigs = tel.get("signals", {})
        if sigs.get("repeated_commands"):
            for tid, cmds in sorted(sigs["repeated_commands"].items()):
                sig.append({
                    "name": "repeated-command",
                    "severity": "medium",
                    "evidence": [f"worker {tid} issued same command {n}x: {c}"
                                 for c, n in sorted(cmds.items())],
                    "suggestion": "supervisor should vary strategy, not re-issue"
                                  " the same directive",
                })
        if sigs.get("repeated_files"):
            for tid, files in sorted(sigs["repeated_files"].items()):
                sig.append({
                    "name": "repeated-file-churn",
                    "severity": "medium",
                    "evidence": [f"worker {tid} touched same file {n}x: {f}"
                                 for f, n in sorted(files.items())],
                    "suggestion": "consolidate repeated edits; a stale "
                                  "hypothesis is being reworked",
                })
        if sigs.get("tests_unchanged"):
            sig.append({
                "name": "activity-without-progress",
                "severity": "medium",
                "evidence": [f"workers {','.join(sigs['tests_unchanged'])} "
                             "wrote state repeatedly with identical test "
                             "counts (tests_unchanged)"],
                "suggestion": "re-running the same suite without change is "
                              "waste; switch strategy",
            })
        if sigs.get("overlap"):
            sig.append({
                "name": "worker-overlap",
                "severity": "low",
                "evidence": [f"concurrent span overlap: {a} & {b}"
                             for a, b in sigs["overlap"][:3]],
                "suggestion": "if overlapped workers duplicate each other's "
                              "research, raise sharing or dedupe before spawn",
            })
        if sigs.get("idle_workers"):
            sig.append({
                "name": "worker-idle",
                "severity": "low",
                "evidence": [f"worker {w} alive >60s with no commands issued"
                             for w in sigs["idle_workers"]],
                "suggestion": "worker waiting on nothing; hold or cancel",
            })

    # 6) honest limits
    return {"mission": mission_id, "wall_seconds": rep.get("wall_seconds"),
            "signals": sig,
            "limits": ("tool-level reads/API calls are not ledger-observable; "
                       "telemetry captures spawns, state-writes, commands, "
                       "retries and exits from the durable audit rail")}


def optimization_plan(mission_id: str) -> Dict[str, Any]:
    """Highest-value improvements with evidence, ready to feed a discovery
    (mission optimize --apply) or the controller brief."""
    wa = waste_analysis(mission_id)
    if "error" in wa:
        return wa
    by_sev = {"high": 0, "medium": 1, "low": 2, "": 3}
    cands = sorted(wa["signals"],
                   key=lambda s: by_sev.get(s.get("severity", ""), 3))
    out = []
    for i, s in enumerate(cands, 1):
        out.append({
            "id": f"opt-{slugify(s['name'])}",
            "signal": s["name"],
            "evidence": s.get("evidence", []),
            "suggestion": s.get("suggestion", ""),
            "priority": 4 - by_sev.get(s.get("severity", ""), 3),
        })
    return {"mission": mission_id, "candidates": out,
            "count": len(out),
            "note": "signals are heuristic (ledger-derived); evidence lines "
                    "are factual, suggestions are advisory"}


def apply_optimizations(mission_id: str) -> Tuple[int, str]:
    """Turn the optimization plan into DISCOVERED work (with evidence).
    Add-discovery dedups by slug, so re-running is safe; planner promotes
    evidence-backed candidates on the next plan pass."""
    from hermes_cli import supervisor as S
    m = S.load_mission(mission_id)
    if m is None:
        return 0, "no mission"
    plan = optimization_plan(mission_id)
    added = 0
    for cand in plan.get("candidates", []):
        before = len(m.get("discoveries", []))
        add_discovery(
            m,
            title=f"[optimize] {cand['signal']}: {cand['suggestion'][:90]}",
            rationale="auto-generated from mission metrics/waste analysis",
            discoverer="metrics",
            evidence=[cand["suggestion"]] + cand.get("evidence", []),
            priority=cand["priority"])
        if len(m.get("discoveries", [])) > before:
            added += 1
    S.save_mission(m)
    return added, f"{added} optimization discovery(ies) recorded"


def suggested_concurrency(env: Optional[Dict[str, Any]] = None) -> int:
    """Environment-aware default worker cap. Conservative on small boxes:
    limited CPU/RAM => fewer concurrent workers. Used when
    MISSION_MAX_CONCURRENT is unset (user override always wins)."""
    env = env if env is not None else os.environ
    if env.get("MISSION_MAX_CONCURRENT"):
        return max(1, int(env["MISSION_MAX_CONCURRENT"]))
    env_snap = environment_snapshot()  # already labelled fact/assumption
    cpus = int(env_snap.get("cpus") or 1)
    mem_kb = env_snap.get("mem_total_kb") or 0
    mem_gb = max(0.0, (mem_kb or 0) / 1024 / 1024)
    cpu_cap = max(1, min(cpus, 4))          # local box: >4 rarely helps today
    mem_cap = max(1, int(mem_gb // 2)) if mem_gb else 4
    return max(1, min(cpu_cap, mem_cap))


def retrospective(mission_id: str) -> Dict[str, Any]:
    """Deterministic mission-level retrospective (self-critique at completion).

    Derived SUCCESSFULLY from the metrics + event journal; writes a durable
    copy next to the ledger. LLM/harness may turn 'analysis' into lessons.
    """
    from hermes_cli import supervisor as S
    rep = metrics_report(mission_id)
    if "error" in rep:
        return rep
    wa = waste_analysis(mission_id)
    m = S.load_mission(mission_id) or {}
    so_phases = [p for p in rep.get("phases", [])
                 if p.get("status") == "COMPLETE"]
    failed = [p for p in rep.get("phases", [])
              if p.get("status") == "FAILED"]
    worker_total = rep.get("workers_counts", {}).get("total", 0)
    # P-26 r12 self-critique: what consumed the most clock?
    ws = rep.get("workers", []) or []
    longest = max(ws, key=lambda w: (w.get("span_seconds") or 0)) if ws else None
    expected = expected_worker_span(mission_id)
    tel = telemetry_report(mission_id) or {}
    tel_counts = tel.get("counts", {}) or {}
    ret = {
        "mission": mission_id,
        "objective": m.get("objective", ""),
        "wall_seconds": rep.get("wall_seconds"),
        "phases_completed": len(so_phases),
        "phases_failed": len(failed),
        "workers_total": worker_total,
        "failures_total": rep.get("failures_total", 0),
        "longest_worker": (longest.get("task_id") if longest else None),
        "longest_worker_span": (longest.get("span_seconds") if longest else None),
        "expected_worker_span": expected,
        "telemetry": {"state_writes": tel_counts.get("state_writes", 0),
                      "commands": tel_counts.get("commands", 0),
                      "retries": tel_counts.get("retries", 0),
                      "spawns": tel_counts.get("spawns", 0)},
        "signals": [s["name"] for s in wa.get("signals", [])],
        "optimizations": [c["id"] for c in
                          optimization_plan(mission_id).get("candidates", [])],
        "analysis": ["completed {} phases in {:.0f}m"
                     .format(len(so_phases), (rep.get("wall_seconds") or 0) / 60),
                     "longest worker: {} ({:.0f}s; expected ~{:.0f}s)"
                     .format((longest or {}).get("task_id", "?"),
                             (longest or {}).get("span_seconds") or 0,
                             expected or 0),
                     "failures: {}; waste signals: {}".format(
                         rep.get("failures_total", 0), len(wa.get("signals", []))),
                     "lessons: run 'mission lessons --sync' to persist"],
        "evidence": "derived from durable ledger/metrics; LLM critique omitted",
    }
    try:
        p = S.missions_dir() / mission_id / "retrospective.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(ret, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass
    # P-26 r7: close the learn loop durably — the deterministic analysis
    # becomes lessons a future worker/controller retrieves (one row per
    # distinct signal, deduped; the optimizer's own discoveries already
    # carry the actionable plan). This makes "learning from execution time"
    # a durable artifact, not a controller-narration afterthought.
    try:
        for sig in wa.get("signals", [])[:4]:
            add_lesson(
                mission_id=mission_id,
                lesson=f"[retro] {sig.get('name', 'signal')}",
                context="deterministic retrospective found a waste pattern",
                better_approach=sig.get("suggestion", ""),
                apply_to=sig.get("name", ""),
                evidence=sig.get("evidence", []))
    except Exception:
        pass
    return ret

# ---------------------------------------------------------------------------
# P-26 r15: architected detectors -> durable discoveries, outcome-derived DoD,
# failure-aware briefs, exploration, artifacts/benchmarks, institutional
# knowledge, attack derivation. All reuse existing primitives (discovery
# graph, telemetry rail, env probe, lesson store) — no second storage.
# ---------------------------------------------------------------------------


def auto_apply_detections(mission_id: str, *, max_new: int = 4) -> Tuple[int, str]:
    """Architected detector pass: convert observable waste signals into
    DURABLE evidence-backed discoveries WITHOUT waiting for an LLM/controller
    to notice them.

    Detectors (all from real ledger/telemetry, never invented):
      - repeated-command / repeated-file-churn / activity-without-progress
        (telemetry signals, r6)
      - duplicate-research / retries / stagnation (waste analysis, r4/r5/r6)
      - changed-without-tests  (NEW r15: worker touched source files but
        never reported tests -> missing-regression-test signal)
      - no-op worker (NEW r15: many commands/state writes, zero files)
    Each becomes a priority discovery with the factual evidence attached; the
    normal planner promotes evidence-backed items and the supervisor executes
    them. Idempotent via slug dedup.
    """
    from hermes_cli import supervisor as S
    m = S.load_mission(mission_id)
    if m is None:
        return 0, "no mission"
    plan = optimization_plan(mission_id)
    candidates: List[Dict[str, Any]] = list(plan.get("candidates", []) or [])
    # NEW detectors over the telemetry rail (only when the data actually
    # shows them — never fabricated).
    try:
        tel = telemetry_report(mission_id) or {}
    except Exception:  # noqa: BLE001
        tel = {}
    for w in tel.get("workers", []) or []:
        # only judge a worker's coverage after it TERMINATED — a live worker
        # may simply not have reported tests yet (mid-run). A terminal exit
        # row is the honest boundary for "changed code but never tested".
        last_exit = w.get("exits") or {}
        ex_status = str(last_exit.get("status") or "") if isinstance(last_exit, dict) else ""
        if ex_status and ex_status not in ("COMPLETE", "FAILED", "CANCELLED", "BLOCKED"):
            continue
        files = [str(f) for f in (w.get("repeated_files") or {}).keys()]
        src_touched = any(
            f.endswith((".py", ".ts", ".js", ".go", ".rs", ".c", ".cpp"))
            for f in files)
        tests_touched = any("test" in f.lower() or f.endswith("_test.go")
                            for f in files)
        if src_touched and not tests_touched and w.get("state_writes"):
            candidates.append({
                "id": f"opt-missing-tests-{str(w.get('task_id','x'))[-12:]}",
                "signal": "changed-without-tests",
                "evidence": [f"worker {w.get('task_id')} touched source "
                             f"files {files[:4]} but reported no tests"],
                "suggestion": "add a regression test covering the change",
                "priority": 3,
            })
        if (w.get("commands") or 0) >= 4 and not files and \
                (w.get("state_writes") or 0) >= 3:
            candidates.append({
                "id": f"opt-noop-{str(w.get('task_id','x'))[-12:]}",
                "signal": "no-op-worker",
                "evidence": [f"worker {w.get('task_id')} issued "
                             f"{w.get('commands')} commands, wrote state "
                             f"{w.get('state_writes')}x, touched 0 files"],
                "suggestion": "reassess: worker produced no artifact",
                "priority": 3,
            })
    added = 0
    notes: List[str] = []
    for cand in candidates[:max_new]:
        title = (f"[detect] {cand.get('signal') or cand.get('id')}: "
                 f"{str(cand.get('suggestion',''))[:80]}")
        before = len(m.get("discoveries", []))
        # P-26 r15: detector findings are ADVISORY unless they point at a
        # genuine coverage gap (changed-without-tests) — advisory items stay
        # runnable but never block the completion gate, so optimization
        # signals cannot turn a mission into an infinite checklist.
        advisory = cand.get("signal") not in ("changed-without-tests",)
        d = add_discovery(
            m, title=title,
            rationale="auto-detector found a waste/coverage signal",
            discoverer="detectors",
            evidence=list(cand.get("evidence") or []) +
                     [str(cand.get("suggestion", ""))],
            priority=int(cand.get("priority", 3)))
        d["advisory"] = advisory
        if len(m.get("discoveries", [])) > before:
            added += 1
            notes.append(d["id"])
    if added:
        plan_discoveries(m)
        S.save_mission(m)
    return added, "; ".join(notes)


def failure_memory_brief(mission_id: str, target: str, limit: int = 3) -> str:
    """PLANNER-AWARE FAILURE MEMORY: render the durable failure-memory rows
    bound to `target` as a brief section, so a worker re-attempting a task
    that failed before SEES the prior dead ends before spending compute.
    Empty string = nothing known (target has never failed)."""
    rows = failure_memory(mission_id, target=target, limit=limit)
    if not rows:
        return ""
    lines = ["", "## Prior failed attempts (do not blindly repeat)",
             "", "Past work bound to this target failed. Read the evidence",
             "below before reproducing the same approach:"]
    for r in rows:
        lines.append(f"- approach: {r.get('approach','?')} | "
                     f"outcome: {r.get('outcome','?')}")
        if r.get("how_to_avoid"):
            lines.append(f"  avoided_by: {r['how_to_avoid']}")
    return "\n".join(lines)


def plan_snapshot(mission_id: str, limit: int = 8) -> str:
    """CURRENT-PLAN AWARENESS: a short durable snapshot of the mission's
    discovery plan for embedding in a worker brief — the worker sees what is
    already open/completed/superseded instead of operating on stale
    instructions."""
    from hermes_cli import supervisor as S
    m = S.load_mission(mission_id)
    if m is None:
        return ""
    ds = m.get("discoveries", [])
    open_d = [d for d in ds if d.get("status") in
              ("TODO", "IN_PROGRESS", "DISCOVERED", "BLOCKED")][:limit]
    done_d = [d.get("id") for d in ds if d.get("status") == "COMPLETE"]
    sup_d = [d.get("id") for d in ds if d.get("status") == "SUPERSEDED"]
    lines = ["", "## Current mission plan (live)",
             f"- objective: {(m.get('objective') or '')[:120]}"]
    for d in open_d:
        lines.append(f"- [{d.get('status')}] p{int(d.get('priority',5))} "
                     f"{d.get('id')}: {str(d.get('title',''))[:80]}")
    if sup_d:
        lines.append(f"- superseded: {', '.join(sup_d[:5])}")
    if done_d:
        lines.append(f"- completed: {', '.join(done_d[:5])}")
    return "\n".join(lines)


def add_exploration(mission: Dict[str, Any], *, topic: str,
                    variants: List[str], max_variants: int = 3
                    ) -> Dict[str, Any]:
    """PARALLEL EXPLORATION: first-class competing-approach exploration. Adds
    up to max_variants sibling DISCOVERED items (same topic, one per variant)
    sharing `explore_group`; the planner promotes them and the supervisor
    pool runs them under the existing concurrency cap. Bounded: never more
    than max_variants; callers decide whether uncertainty justifies it."""
    group = f"explore-{slugify(topic)[:24]}-{int(time.time())}"
    created: List[str] = []
    for i, v in enumerate(variants[:max_variants]):
        title = f"[explore] {topic} — variant {i+1}: {v[:80]}"
        d = add_discovery(
            mission, title=title,
            rationale=f"parallel exploration variant {i+1} of '{topic}'",
            discoverer="explore", evidence=[f"explore group {group}"],
            priority=2)
        d.setdefault("explore_group", group)
        created.append(d["id"])
    mission.setdefault("explore_groups", []).append(
        {"group": group, "topic": topic, "variants": created,
         "created_at": time.time()})
    return {"group": group, "variants": created}


def note_benchmark(mission: Dict[str, Any], *, name: str, value: str,
                   provenance: str = "") -> bool:
    """Record a benchmark/measurement durably on the mission ledger so later
    planning + retrospectives see the number and its provenance. Never
    fabricates a unit; value is the caller-supplied string."""
    mission.setdefault("benchmarks", []).append(
        {"name": name, "value": value, "provenance": provenance,
         "at": time.time()})
    return True


def list_benchmarks(mission: Dict[str, Any]) -> Dict[str, Any]:
    bm = mission.get("benchmarks", [])
    return {"count": len(bm), "benchmarks": bm}


def institutional_knowledge(subject: str, *, mission_id: str = "",
                            limit: int = 20) -> Dict[str, Any]:
    """CROSS-SESSION INSTITUTIONAL MEMORY: a coherent retrieval layer over
    what ALREADY exists — lessons store, mission ledgers (this + prior),
    skill/plugin/MCP inventory, env probe. Answers: what have we tried /
    rejected / learned, what capabilities exist, machine constraints. No
    second database."""
    from hermes_cli import environment as ENV
    out: Dict[str, Any] = {"subject": subject}
    # 1) learning store: what did we try / reject / learn already
    try:
        from agent.experience import retrieve_lessons
        lessons = retrieve_lessons(subject, max_results=4)
        out["lessons"] = [
            {"summary": l.summary, "domain": getattr(l, "domain", ""),
             "failure_mode": (getattr(l, "failure_mode", "") or "")[:200],
             "source_outcome": getattr(l, "source_outcome", "")}
            for l in lessons][:4]
    except Exception as exc:  # noqa: BLE001
        out["lessons_error"] = str(exc)
    # 2) mission histories (this mission + any mission mentioning subject)
    from hermes_cli import supervisor as S
    ms = []
    base = S.missions_dir()
    if base.is_dir():
        for f in sorted(base.glob("*.json"))[-15:]:
            try:
                mm = json.loads(f.read_text())
            except Exception:  # noqa: BLE001
                continue
            obj = (mm.get("objective") or "")
            if subject.lower() in obj.lower() or mission_id == f.stem:
                ms.append({
                    "mission_id": f.stem, "objective": obj[:160],
                    "discoveries": [(d.get("id"), d.get("status"))
                                    for d in mm.get("discoveries", [])][:8],
                    "phases": [(p.get("phase_id"), p.get("status"))
                               for p in mm.get("phases", [])][:8]})
    out["missions"] = ms[:limit]
    # 3) capability router: what can I do here for this subject
    out["capabilities"] = capability_router(subject)[:6]
    # 4) live machine constraints
    env = ENV.probe_env()
    out["machine"] = {"cpus": env.get("cpus_available") or env.get("cpus_logical"),
                      "mem_total_kb": env.get("mem_total_kb"),
                      "python": env.get("python")}
    return out


def discover_attack(mission: Dict[str, Any], *, module: str,
                    finder: str = "detectors") -> Dict[str, Any]:
    """Adversarial self-testing: derive an attack-probe discovery from a
    concrete changed module/regex/parser the mission already touched (from
    telemetry or the worker evidence), so 'how can I break this?' becomes a
    durable, bounded discovery instead of a hope."""
    d = add_discovery(
        mission,
        title=f"[attack] adversarial probe: {module}",
        rationale=f"automatic adversarial derivation on changed module "
                  f"'{module}'",
        discoverer=finder,
        evidence=[f"changed module: {module}",
                  "attack discovery generated from telemetry"],
        priority=2)
    return d


def limitation_notes(mission: Dict[str, Any]) -> List[Dict[str, Any]]:
    """LIMITATION AWARENESS (#13): for every OPEN discovery, say whether the
    capabilities its task implies are actually available in THIS runtime. A
    discovery that implies an uninstalled backend is flagged with the
    concrete missing capability + 'adapt or request' advice — so persistence
    is not spent re-trying what the machine cannot do. Facts only: opinions
    about scope/permission are never invented here."""
    out: List[Dict[str, Any]] = []
    open_d = [d for d in mission.get("discoveries", [])
              if d.get("status") in ("TODO", "IN_PROGRESS", "DISCOVERED")]
    for d in open_d:
        text = f"{d.get('title', '')} {d.get('rationale', '')}"
        try:
            route = capability_router(text)[:6]
        except Exception:  # noqa: BLE001
            continue
        missing = [r["cap"] for r in route
                   if r.get("kind") == "backend" and not r.get("available")]
        if missing:
            out.append({
                "did": d["id"],
                "missing_capabilities": missing,
                "note": ("implies capability absent from this runtime; "
                         "adapt via available backend or ask for the "
                         "missing capability"),
            })
    return out