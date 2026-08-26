"""Experience-driven learning module for Hermes.

Implements the full lifecycle:
  OUTCOME → REFLECTION → CANDIDATE LESSON → VALIDATION → MEMORY → RETRIEVAL → BEHAVIOR CHANGE → MEASUREMENT

This is NOT a Reflexion clone. It is a Hermes-native implementation that:
- Reuses existing Hermes SQLite (via verification_evidence.db extension) rather than new storage
- Validates lessons before storing (rejects noise, detects contradictions)
- Tracks confidence, evidence count, success rate per lesson
- Measures whether lesson reuse improves outcomes
- Supports decay/retirement for stale lessons

Usage from turn_finalizer.py or background_review.py:
    from agent.experience import evaluate_outcome, extract_lesson, validate_and_store
    outcome = evaluate_outcome(goal, actual, changed_paths, evidence)
    lesson = extract_lesson(outcome)
    if lesson and validate_and_store(lesson, session_id):
        LOGGER.info("Lesson stored: %s", lesson.summary)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

LOGGER = logging.getLogger(__name__)

_DB_LOCK = threading.Lock()
_DB_PATH: Path | None = None
_DB_CONN: sqlite3.Connection | None = None
_LESSON_SCHEMA_VERSION = 1

# ── Data model ──────────────────────────────────────────────────────


BEHAVIOR_TARGETS = {
    "planning": "influences how tasks are decomposed and ordered",
    "tool_selection": "influences which tool is chosen for a task",
    "context_retrieval": "influences what files or symbols are read",
    "execution": "influences how tool arguments are constructed",
    "verification": "influences how results are tested or validated",
    "recovery": "influences how failures are handled",
    "repository_understanding": "influences where in the codebase to look",
}


@dataclass
class Outcome:
    """Structured evaluation of a task execution outcome.

    More detailed than just 'success' or 'failure'.
    """

    goal: str
    intended_result: str
    actual_result: str
    status: str  # "success", "failure", "partial", "verification_failed"
    confidence: float  # 0.0–1.0 how confident we are in this evaluation
    failures: list[str] = field(default_factory=list)
    recovery_attempts: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    verification_evidence: list[str] = field(default_factory=list)
    failure_cause: str = "unknown"
    # Source of the failure for lesson extraction
    source: str = ""  # "planning", "context", "tool", "assumption", "environment", "implementation", "verification", "model"


@dataclass
class Lesson:
    """A validated lesson extracted from experience.

    Only lessons that pass validation gates enter the lesson store.
    """

    summary: str
    detail: str
    category: str  # "convention", "pitfall", "workflow", "dependency", "pattern", "tool_behavior"
    domain: str
    source_task: str
    source_outcome: str  # "success" or "failure" that generated this
    behavior: str = "planning"  # which subsystem this targets (see BEHAVIOR_TARGETS keys)
    patterns: list[str] = field(default_factory=list)  # generalized patterns this applies to
    failure_mode: str = ""  # normalized cause description
    evidence: list[str] = field(default_factory=list)
    # Validation metadata (populated by validate_and_store)
    id: int = 0  # DB primary key (0 = not yet persisted)
    confidence: float = 0.5
    evidence_count: int = 0
    success_count: int = 0
    application_count: int = 0
    last_used: float = 0.0
    created: float = field(default_factory=time.time)
    state: str = "candidate"  # "candidate", "active", "contradicted", "retired"


# ── Database ────────────────────────────────────────────────────────


def _db() -> sqlite3.Connection:
    """Get the lessons database connection (lazy init, persistent connection)."""
    global _DB_PATH, _DB_CONN
    with _DB_LOCK:
        if _DB_CONN is not None:
            return _DB_CONN
        if _DB_PATH is None:
            _DB_PATH = get_hermes_home() / "state.db"
        _DB_CONN = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _DB_CONN.row_factory = sqlite3.Row
        _DB_CONN.execute("PRAGMA journal_mode=WAL")
        _ensure_schema(_DB_CONN)
        return _DB_CONN


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary TEXT NOT NULL,
            detail TEXT NOT NULL DEFAULT '',
            category TEXT NOT NULL DEFAULT 'general',
            domain TEXT NOT NULL DEFAULT '',
            source_task TEXT NOT NULL DEFAULT '',
            source_outcome TEXT NOT NULL DEFAULT 'unknown',
            behavior TEXT NOT NULL DEFAULT 'planning',
            patterns TEXT NOT NULL DEFAULT '[]',
            failure_mode TEXT NOT NULL DEFAULT '',
            evidence TEXT NOT NULL DEFAULT '[]',
            confidence REAL NOT NULL DEFAULT 0.5,
            evidence_count INTEGER NOT NULL DEFAULT 0,
            success_count INTEGER NOT NULL DEFAULT 0,
            application_count INTEGER NOT NULL DEFAULT 0,
            last_used REAL NOT NULL DEFAULT 0.0,
            created REAL NOT NULL,
            state TEXT NOT NULL DEFAULT 'candidate',
            schema_version INTEGER NOT NULL DEFAULT 1
        )""")
    # schema migration: add columns if missing
    for col, col_type in [("behavior", "TEXT NOT NULL DEFAULT 'planning'"),
                          ("patterns", "TEXT NOT NULL DEFAULT '[]'"),
                          ("failure_mode", "TEXT NOT NULL DEFAULT ''")]:
        try:
            conn.execute(f"ALTER TABLE lessons ADD COLUMN {col} {col_type}")
        except Exception:
            pass
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS lesson_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lesson_id INTEGER NOT NULL,
            task_id TEXT NOT NULL DEFAULT '',
            task_goal TEXT NOT NULL DEFAULT '',
            retrieved INTEGER NOT NULL DEFAULT 1,
            applied INTEGER NOT NULL DEFAULT 0,
            ignored INTEGER NOT NULL DEFAULT 0,
            rejected INTEGER NOT NULL DEFAULT 0,
            outcome TEXT NOT NULL DEFAULT '',
            verification_result TEXT NOT NULL DEFAULT '',
            task_success INTEGER NOT NULL DEFAULT 0,
            created REAL NOT NULL
        )""")
    except Exception:
        pass


# ── Outcome evaluation ──────────────────────────────────────────────


def evaluate_outcome(
    goal: str,
    intended_result: str,
    actual_result: str,
    *,
    changed_paths: list[str] | None = None,
    tools_used: list[str] | None = None,
    verification_evidence: list[str] | None = None,
    failures: list[str] | None = None,
    recovery_attempts: list[str] | None = None,
) -> Outcome:
    """Build a structured Outcome from task execution data.

    Call this after a task completes (success, failure, or partial).
    Hermes' existing `_turn_file_mutation_paths` and `verification_evidence.py`
    provide the data.

    Args:
        goal: What was the task?
        intended_result: What was supposed to happen?
        actual_result: What actually happened?
        changed_paths: Files modified.
        tools_used: Tools invoked during the task.
        verification_evidence: Test/lint results.
        failures: Error messages or failure descriptions.
        recovery_attempts: Retry/recovery steps taken.

    Returns:
        An Outcome dataclass with a guessed failure cause and confidence.
    """
    has_failures = bool(failures)
    has_verification = bool(verification_evidence)
    ver_failed = any("FAIL" in v or "failure" in v.lower() or "error" in v.lower() for v in verification_evidence or [])
    partial = has_failures and has_verification and not ver_failed

    if not has_failures and ver_failed:
        status = "verification_failed"
    elif has_failures and ver_failed:
        status = "failure"
    elif has_failures:
        status = "partial"
    elif ver_failed:
        status = "verification_failed"
    else:
        status = "success"

    confidence = 0.9 if status == "success" and not has_failures else 0.7 if status == "failure" else 0.5

    cause = "unknown"
    source = ""
    if failures:
        for f in failures:
            fl = f.lower()
            if "traceback" in fl or "importerror" in fl or "syntaxerror" in fl:
                cause = "implementation_error"
                source = "implementation"
                break
            if "timeout" in fl or "connection" in fl:
                cause = "environment_failure"
                source = "environment"
                break
            if "assert" in fl:
                cause = "verification_failure"
                source = "verification"
                break
        if source == "":
            source = "implementation"

    return Outcome(
        goal=goal,
        intended_result=intended_result,
        actual_result=actual_result,
        status=status,
        confidence=confidence,
        failures=failures or [],
        recovery_attempts=recovery_attempts or [],
        files_changed=changed_paths or [],
        tools_used=tools_used or [],
        verification_evidence=verification_evidence or [],
        failure_cause=cause,
        source=source,
    )


# ── Lesson extraction ───────────────────────────────────────────────


def extract_lesson(outcome: Outcome) -> Lesson | None:
    """Extract a candidate Lesson from an Outcome.

    Returns None if the outcome has nothing to learn from
    (e.g. trivial success with no insight).

    The lesson summary is generated from the failure cause and tools used.
    """
    if outcome.status == "success" and not outcome.failures and not outcome.recovery_attempts:
        return None

    failure_desc = "; ".join(outcome.failures[:3]) if outcome.failures else outcome.actual_result[:200]
    tools = "; ".join(outcome.tools_used[:5])
    changed = "; ".join(outcome.files_changed[:5])

    if outcome.failure_cause == "implementation_error":
        summary = f"Implementation error in {changed}: {failure_desc[:120]}"
        detail = f"When modifying {changed}, encountered {failure_desc}. Used tools: {tools}. Ensure tests pass before declaring complete."
        category = "pitfall"
    elif outcome.failure_cause == "environment_failure":
        summary = f"Environment issue during {tools}: {failure_desc[:120]}"
        detail = f"Tool {tools} failed due to environment: {failure_desc}. Consider retry or alternate approach."
        category = "tool_behavior"
    elif outcome.failure_cause == "verification_failure":
        summary = f"Verification failed after modifying {changed}"
        detail = f"Changes to {changed} caused test failures: {failure_desc}. Run full test suite for affected modules."
        category = "workflow"
    else:
        summary = f"Outcome ({outcome.status}): {outcome.actual_result[:150]}"
        detail = f"Task: {outcome.goal[:200]}. Result: {outcome.actual_result[:200]}"
        category = "general"

    return Lesson(
        summary=summary[:200],
        detail=detail[:500],
        category=category,
        domain=changed or tools or outcome.goal[:100],
        source_task=outcome.goal[:200],
        source_outcome=outcome.status,
        evidence=[failure_desc[:200]] if failure_desc else [],
        evidence_count=1,
    )


# ── Validation ──────────────────────────────────────────────────────


def validate_and_store(lesson: Lesson, session_id: str = "") -> bool:
    """Validate a lesson against existing lessons, then store it.

    Validation checks:
    - Duplicate detection (similar summary already exists)
    - Contradiction detection (existing lesson says opposite)
    - Quality gate (summary isn't too vague)

    Returns True if the lesson was stored.

    Args:
        lesson: The candidate Lesson.
        session_id: Optional session identifier for provenance.
    """
    # Quality gate — reject vague lessons
    if len(lesson.summary) < 20:
        LOGGER.debug("Lesson too vague, rejected: %s", lesson.summary[:60])
        return False

    conn = _db()
    with _DB_LOCK:
        # Duplicate check — same summary text
        existing = conn.execute(
            "SELECT id, summary, state, confidence, evidence_count FROM lessons WHERE summary = ?",
            (lesson.summary,),
        ).fetchone()
        if existing:
            if existing["state"] == "active":
                # Strengthen existing lesson instead of creating duplicate
                conn.execute(
                    "UPDATE lessons SET evidence_count = evidence_count + 1, success_count = success_count + ? WHERE id = ?",
                    (1 if lesson.source_outcome == "success" else 0, existing["id"]),
                )
                LOGGER.debug("Strengthened existing lesson #%d", existing["id"])
                return True
            if existing["state"] == "contradicted" or existing["state"] == "retired":
                LOGGER.debug("Lesson matches %s lesson #%d — not re-adding", existing["state"], existing["id"])
                return False

        # Contradiction check — look for lessons with opposite summary patterns
        curs = conn.execute(
            "SELECT id, summary, detail FROM lessons WHERE state = 'active' AND domain = ?",
            (lesson.domain,),
        )
        for row in curs.fetchall():
            if _is_contradictory(lesson.summary, row["summary"]):
                conn.execute("UPDATE lessons SET state = 'contradicted' WHERE id = ?", (row["id"],))
                LOGGER.debug("Contradicted existing lesson #%d with new lesson", row["id"])

        # Store the new lesson
        conn.execute(
            """INSERT INTO lessons
               (summary, detail, category, domain, source_task, source_outcome,
                behavior, patterns, failure_mode, evidence, confidence, evidence_count, success_count,
                application_count, last_used, created, state)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, 'active')""",
            (
                lesson.summary, lesson.detail, lesson.category, lesson.domain,
                lesson.source_task, lesson.source_outcome,
                lesson.behavior,
                json.dumps(lesson.patterns), lesson.failure_mode,
                json.dumps(lesson.evidence), lesson.confidence, lesson.evidence_count,
                1 if lesson.source_outcome == "success" else 0,
                lesson.created,
            ),
        )
        lesson.id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
        LOGGER.info("Stored lesson: %s [%s]", lesson.summary[:80], lesson.category)
        return True


def _is_contradictory(a: str, b: str) -> bool:
    """Simple heuristic: are these two lesson summaries contradictory?

    Checks for opposite-meaning keyword pairs.
    """
    opposites = [
        ("do not use", "always use"),
        ("never", "always"),
        ("avoid", "always use"),
        ("avoid", "prefer"),
        ("fails with", "works with"),
        ("do not", "must"),
    ]
    al = a.lower()
    bl = b.lower()
    for neg, pos in opposites:
        if (neg in al and pos in bl) or (pos in al and neg in bl):
            return True
    return False


# ── Retrieval ───────────────────────────────────────────────────────


def retrieve_lessons(
    context: str,
    *,
    max_results: int = 5,
    min_confidence: float = 0.3,
    category: str | None = None,
) -> list[Lesson]:
    """Retrieve active lessons relevant to the given context.

    Two-stage matching:
    1. Fast keyword overlap scoring on summary+detail
    2. Structured pattern matching using lesson.patterns and lesson.failure_mode
       against the context (generalized retrieval for vocabulary-mismatched tasks)

    Sorted by composite relevance score.
    """
    conn = _db()
    tokens = set(context.lower().split()[:50])
    context_lower = context.lower()

    if category:
        raw = conn.execute(
            "SELECT * FROM lessons WHERE state = 'active' AND confidence >= ? AND category = ?",
            (min_confidence, category),
        ).fetchall()
    else:
        raw = conn.execute(
            "SELECT * FROM lessons WHERE state = 'active' AND confidence >= ?",
            (min_confidence,),
        ).fetchall()

    scored = []
    for row in raw:
        # Stage 1: keyword overlap on summary+detail
        lesson_text = f"{row['summary']} {row['detail']}".lower()
        kw_score = len(tokens & set(lesson_text.split())) / max(len(tokens), 1)

        # Stage 2: structured pattern/failure_mode matching
        pattern_score = 0.0
        try:
            patterns = json.loads(row["patterns"]) if row["patterns"] != "[]" else []
        except (json.JSONDecodeError, TypeError, KeyError):
            patterns = []
        failure_mode = ""
        try:
            failure_mode = row["failure_mode"]
        except KeyError:
            pass

        for p in patterns:
            if isinstance(p, str):
                pl = p.lower()
                if pl in context_lower:
                    pattern_score += 0.3
                else:
                    pwords = set(pl.split())
                    overlap = pwords & tokens
                    if len(overlap) >= 2:
                        pattern_score += 0.2
                    elif len(overlap) >= 1:
                        pattern_score += 0.1
                    # Also check if any single word from the pattern is in context
                    for pw in pwords:
                        if len(pw) > 3 and pw in context_lower:
                            pattern_score += 0.05
        if failure_mode:
            fml = failure_mode.lower()
            if fml in context_lower:
                pattern_score += 0.3
            else:
                fm_words = set(fml.split())
                fm_overlap = fm_words & tokens
                if len(fm_overlap) >= 2:
                    pattern_score += 0.2
                # Single-word partial matches
                for fw in fm_words:
                    if len(fw) > 3 and fw in context_lower:
                        pattern_score += 0.05

        # Composite score: keyword overlap + pattern bonus
        total = kw_score + pattern_score
        if total > 0:
            scored.append((total, row))

    scored.sort(key=lambda x: x[0], reverse=True)

    result = []
    for score, row in scored[:max_results]:
        try:
            lp = json.loads(row["patterns"]) if row["patterns"] != "[]" else []
        except (json.JSONDecodeError, TypeError, KeyError):
            lp = []
        lesson = Lesson(
            id=row["id"],
            summary=row["summary"],
            detail=row["detail"],
            category=row["category"],
            domain=row["domain"],
            source_task=row["source_task"],
            source_outcome=row["source_outcome"],
            behavior=row["behavior"],
            patterns=lp,
            failure_mode=row.get("failure_mode", "") if "failure_mode" in row else "",
            evidence=json.loads(row["evidence"]) if row["evidence"] else [],
            confidence=row["confidence"],
            evidence_count=row["evidence_count"],
            success_count=row["success_count"],
            application_count=row["application_count"],
            last_used=row["last_used"],
            created=row["created"],
            state=row["state"],
        )
        result.append(lesson)

    return result

def record_lesson_outcome(lesson_id: int, was_useful: bool) -> None:
    """Record whether a retrieved lesson was useful.

    Call after the task using the lesson completes.
    Over time this creates a feedback loop for lesson quality.
    """
    conn = _db()
    if was_useful:
        conn.execute(
            "UPDATE lessons SET success_count = success_count + 1, confidence = MIN(1.0, confidence + 0.05) WHERE id = ?",
            (lesson_id,),
        )
    else:
        conn.execute(
            "UPDATE lessons SET confidence = MAX(0.1, confidence - 0.05) WHERE id = ?",
            (lesson_id,),
        )
    conn.commit()


# ── Maintenance ─────────────────────────────────────────────────────


def decay_old_lessons(max_age_days: int = 90) -> int:
    """Move old, rarely-used lessons to 'retired' state.

    Returns the number of lessons retired.
    """
    cutoff = time.time() - (max_age_days * 86400)
    conn = _db()
    count = conn.execute(
        "UPDATE lessons SET state = 'retired' WHERE state = 'active' AND last_used < ? AND application_count < 3",
        (cutoff,),
    ).rowcount
    conn.commit()
    if count:
        LOGGER.info("Retired %d old lessons", count)
    return count


# ── Lesson application tracking ─────────────────────────────────────


def log_lesson_application(
    lesson_id: int,
    *,
    task_id: str = "",
    task_goal: str = "",
    retrieved: bool = True,
    applied: bool = False,
    ignored: bool = False,
    rejected: bool = False,
    outcome: str = "",
    verification_result: str = "",
    task_success: bool = False,
) -> None:
    """Log a lesson application event for effectiveness measurement.

    Call after each task where lessons were retrieved, recording
    whether the lesson was actually used and what the outcome was.
    """
    conn = _db()
    conn.execute(
        """INSERT INTO lesson_applications
           (lesson_id, task_id, task_goal, retrieved, applied, ignored,
            rejected, outcome, verification_result, task_success, created)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (lesson_id, task_id, task_goal, int(retrieved), int(applied),
         int(ignored), int(rejected), outcome, verification_result,
         int(task_success), time.time()),
    )
    conn.commit()


def lesson_effectiveness_score(lesson_id: int) -> dict[str, float]:
    """Calculate effectiveness metrics for a lesson.

    Returns:
        dict with: retrieved, applied, ignored, rejected,
                  success_rate (when applied), raw_effectiveness
    """
    conn = _db()
    row = conn.execute(
        "SELECT COUNT(*) as total, SUM(applied) as applied_count, "
        "SUM(ignored) as ignored_count, SUM(rejected) as rejected_count, "
        "SUM(task_success) as successes "
        "FROM lesson_applications WHERE lesson_id = ?",
        (lesson_id,),
    ).fetchone()
    if not row or not row["total"]:
        return {"retrieved": 0, "applied": 0, "ignored": 0,
                "rejected": 0, "success_rate": 0.0, "raw_effectiveness": 0.0}

    total = row["total"]
    applied_count = row["applied_count"] or 0
    success_rate = (row["successes"] or 0) / max(applied_count, 1)

    return {
        "retrieved": total,
        "applied": applied_count,
        "ignored": row["ignored_count"] or 0,
        "rejected": row["rejected_count"] or 0,
        "success_rate": round(success_rate, 2),
        "raw_effectiveness": round(success_rate * applied_count / max(total, 1), 2),
    }


# ── Contradiction detection ─────────────────────────────────────────


def find_competing_lessons(lesson: Lesson) -> list[dict[str, Any]]:
    """Find existing active lessons that may compete with this one.

    Operates on structured fields: same behavior target, overlapping domain,
    different recommendation type.

    Returns list of dicts with lesson_id, summary, similarity_reason.
    """
    conn = _db()
    rows = conn.execute(
        "SELECT id, summary, behavior, domain FROM lessons "
        "WHERE (state = 'active' OR state = 'contradicted') "
        "AND behavior = ? AND domain = ?",
        (lesson.behavior, lesson.domain),
    ).fetchall()
    result = []
    for row in rows:
        # Don't flag the same summary
        if row["summary"] == lesson.summary:
            continue
        rule = _is_contradictory(lesson.summary, row["summary"])
        if rule:
            result.append({
                "lesson_id": row["id"],
                "summary": row["summary"],
                "similarity_reason": f"rule-based contradiction: {rule}",
            })
    return result


# ── Format for system prompt injection ──────────────────────────────


def format_lessons_for_prompt(lessons: list[Lesson]) -> str:
    """Format retrieved lessons for injection into the planner context.

    Returns a compact string suitable for the volatile system prompt tier.
    Includes behavior target for each lesson so it influences the correct
    subsystem.
    """
    if not lessons:
        return ""
    parts = ["\n[Previous experience relevant to this task:]"]
    for i, lesson in enumerate(lessons[:5], 1):
        target = lesson.behavior or "planning"
        parts.append(f"  {i}. [{target}] {lesson.summary}")
    parts.append("")
    return "\n".join(parts)