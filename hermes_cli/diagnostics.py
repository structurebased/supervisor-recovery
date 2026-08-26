"""Hermes Environment Diagnostics — read-only introspection of the live runtime.

Deterministic, evidence-based component graph + error → interference analysis.

WIRING (all read-only, no core-loop changes):
    * model tool:  tools/diag_tool.py  (registry-registered, toolset="skills")
    * CLI:         hermes diag [...]   (hermes_cli/subcommands/diagnostics.py)
    * tests:       tests/diagnostics/*

SOURCES (live Hermes registries):
    plugins   -> PluginManager.list_plugins() + LoadedPlugin hooks_registered
    skills    -> tools.skills_tool._find_all_skills + tools.skill_usage provenance
    mcp       -> tools.mcp_tool.get_mcp_status() + config mcp_servers
    hooks     -> PluginManager._hooks (hook_name -> callbacks)
    tools     -> tools.registry registry._tools (guarded read)
    config    -> hermes_cli.config.load_config_readonly()
    soul3     -> agent.{planner,experience,durable,orchestrator} + agent.integrations

Classification vocabulary (deterministic, defined in CLASSIFICATIONS):
    INVOLVED / PLAUSIBLE SOURCE OF INTERFERENCE / MERELY PRESENT /
    UNLIKELY / NOT CAPABLE OF AFFECTING THIS PATH / UNKNOWN

Failure policy: this module NEVER raises. Every collector is guarded; a broken
subsystem reports {"error": ...} inside its section so the diagnostic tool
cannot crash the agent loop.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# semantic artifact classifier for completion evidence validation (s5 model)
try:  # pragma: no cover - import guard
    from hermes_cli.supervisor import _evidence_kinds
except Exception:  # pragma: no cover - diagnostics must import standalone
    _evidence_kinds = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification vocabulary
# ---------------------------------------------------------------------------

INVOLVED = "INVOLVED"
PLAUSIBLE = "PLAUSIBLE SOURCE OF INTERFERENCE"
MERELY_PRESENT = "MERELY PRESENT"
UNLIKELY = "UNLIKELY"
NOT_CAPABLE = "NOT CAPABLE OF AFFECTING THIS PATH"
UNKNOWN = "UNKNOWN"

CLASSIFICATIONS = (
    INVOLVED,
    PLAUSIBLE,
    MERELY_PRESENT,
    UNLIKELY,
    NOT_CAPABLE,
    UNKNOWN,
)

# Hook events that can MODIFY/BLOCK tool execution (whatever their observer
# semantics, they sit on the tool-dispatch wire).
_TOOL_WIRE_HOOKS = {
    "pre_tool_call",        # can block / approve  (resolve_pre_tool_block)
    "post_tool_call",       # can observe / emit   (model_tools)
    "transform_tool_result",  # can rewrite what the model sees
    "transform_terminal_output",  # can rewrite terminal output
}
_LLM_WIRE_HOOKS = {
    "pre_llm_call",         # can inject context into the user message
    "post_llm_call",
    "pre_api_request",
    "post_api_request",
    "api_request_error",
    "transform_llm_output",
}
_GATEWAY_WIRE_HOOKS = {
    "pre_gateway_dispatch",  # can skip / rewrite / block a gateway message
}
_LIFECYCLE_HOOKS = {
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "on_skill_lifecycle",
    "subagent_start",
    "subagent_stop",
    "pre_approval_request",
    "post_approval_response",
    "pre_verify",
    "kanban_task_claimed",
    "kanban_task_completed",
    "kanban_task_blocked",
}

# All hook components can subscribe to.
ALL_KNOWN_HOOKS = (
    _TOOL_WIRE_HOOKS
    | _LLM_WIRE_HOOKS
    | _GATEWAY_WIRE_HOOKS
    | _LIFECYCLE_HOOKS
)

# Error text -> subsystem + execution-path mapping. Deterministic regex table.
# Each entry: regex patterns -> (subsystem, execution_point, candidate families)
_ERROR_MAP: List[Tuple[str, str, str, Tuple[str, ...]]] = [
    # --- Tool execution -------------------------------------------------
    (r"plugin\s*(policy|block)|blocked by plugin|pre_tool_call|denied by plugin",
     "Tool execution", "resolve_pre_tool_block() → model_tools dispatch",
     ("pre_tool_call",)),
    (r"midll_use|transform_tool_result|tool.result|injected instructions",
     "Tool result processing", "turn_finalizer transform_tool_result",
     ("transform_tool_result",)),
    (r"terminal.*(output|stdout)|transform_terminal",
     "Terminal output", "terminal_tool -> transform_terminal_output",
     ("transform_terminal_output",)),
    (r"tool.*not found|unknown tool|no such tool|invalid tool",
     "Tool registry", "tooldiscovery model_tools",
     ("pre_tool_call",)),
    (r"mcp__|MCP server|mcp server",
     "MCP", "tools/mcp_tool.py connection + invocation",
     ("pre_tool_call", "mcp_servers")),
    # --- LLM / prompt -----------------------------------------------------
    (r"pre_llm_call|context injection|prompt|system prompt|cache|token",
     "Prompt/LLM call", "pre_llm_call + LLM request",
     ("pre_llm_call", "pre_api_request")),
    (r"api_request_error|api_request|rate.limit|retry|backoff|provider",
     "API request", "pre_api_request -> LLM; error classification in agent/error_classifier.py",
     ("pre_api_request", "api_request_error")),
    # --- Gateway ----------------------------------------------------------
    (r"gateway|dispatch|pre_gateway|message.*(drop|rewrite)|send",
     "Gateway dispatch", "gateway/run.py -> pre_gateway_dispatch",
     ("pre_gateway_dispatch",)),
    # --- Skills -----------------------------------------------------------
    (r"skill|SKILL\.md|skill_view|skill_manage",
     "Skills", "skill_view / skill_manage; skills index build",
     ("on_skill_lifecycle",)),
    # --- Config -----------------------------------------------------------
    (r"config|yaml|load_config|missing key|NoneType.*config",
     "Configuration", "hermes_cli/config.py load/merge",
     ()),
    # --- Persistence / state ----------------------------------------------
    (r"state\.db|sqlite|session|persist|db",
     "Persistence", "lessons/durable/state.db writes",
     ("on_session_finalize",)),
    # --- SOUL-3.0 ---------------------------------------------------------
    (r"orchestrator|planner|experience|durable|lessons|headroom|provenance",
     "SOUL-3.0", "agent/orchestrator.py + planner/experience/durable",
     ()),
]

# Subsystem label -> which plugin families might matter.
_SUBSYSTEM_HOOKS: Dict[str, Tuple[str, ...]] = {
    "Tool execution": ("pre_tool_call", "post_tool_call"),
    "Tool result processing": ("transform_tool_result",),
    "Terminal output": ("transform_terminal_output",),
    "MCP": ("pre_tool_call",),
    "Prompt/LLM call": ("pre_llm_call", "pre_api_request", "post_llm_call"),
    "API request": ("pre_api_request", "post_api_request", "api_request_error"),
    "Gateway dispatch": ("pre_gateway_dispatch",),
    "Skills": ("on_skill_lifecycle",),
    "SOUL-3.0": ("on_session_start", "pre_api_request", "post_tool_call",
                 "transform_tool_result", "post_api_request"),
    "Configuration": (),
    "Persistence": ("on_session_finalize",),
    "Other": (),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Component inventory
# ---------------------------------------------------------------------------

def collect_plugins() -> Dict[str, Any]:
    """Snapshot of the live PluginManager registry (enabled/loaded/hooks/tools)."""
    from hermes_cli.plugins import get_plugin_manager, PluginManager

    pm = get_plugin_manager()
    try:
        pm.discover_and_load()  # idempotent
    except Exception as exc:  # noqa: BLE001
        return {"error": f"discover_and_load: {type(exc).__name__}: {exc}"}

    out: Dict[str, Any] = {"plugins": [], "hooks": {}, "middleware": {}}
    try:
        for info in pm.list_plugins():
            out["plugins"].append(info)
        # Hook registry: hook_name -> [callbacks]. Uses the private dict in
        # a guarded read — this is exactly what diagnostics exists to expose.
        hooks = getattr(pm, "_hooks", None)
        if isinstance(hooks, dict):
            out["hooks"] = {name: [getattr(cb, "__module__", repr(cb)) for cb in cbs]
                            for name, cbs in sorted(hooks.items())}
        mw = getattr(pm, "_middleware", None)
        if isinstance(mw, dict):
            out["middleware"] = {k: len(v) for k, v in mw.items()}
    except Exception as exc:  # noqa: BLE001
        logger.warning("collect_plugins hook snapshot failed: %s", exc)
        out["hooks"] = {"error": str(exc)}
    return out


def plugin_hooks_by_name(pm) -> Dict[str, List[str]]:
    """plugin_name -> list of hook event names it registered."""
    result: Dict[str, List[str]] = {}
    plugins = getattr(pm, "_plugins", {})
    if not isinstance(plugins, dict):
        return result
    for key, loaded in plugins.items():
        try:
            hooks = list(getattr(loaded, "hooks_registered", []) or [])
        except Exception:  # noqa: BLE001
            hooks = []
        result[key] = hooks
    return result


def collect_skills() -> Dict[str, Any]:
    """Snapshot of the live skill index with provenance + usage telemetry."""
    try:
        from tools import skills_tool
        from tools import skill_usage
    except Exception as exc:  # noqa: BLE001
        return {"error": f"import: {type(exc).__name__}: {exc}"}
    try:
        all_skills = skills_tool._find_all_skills(skip_disabled=False)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"_find_all_skills: {type(exc).__name__}: {exc}"}

    enriched: List[Dict[str, Any]] = []
    try:
        for s in all_skills:
            name = s.get("name", "?")
            prov = "unmarked"
            try:
                if skill_usage.is_bundled(name):
                    prov = "bundled"
                elif skill_usage.is_hub_installed(name):
                    prov = "hub"
                elif skill_usage.is_agent_created(name):
                    prov = "agent"
            except Exception:  # noqa: BLE001
                pass
            rec: Dict[str, Any] = {}
            try:
                rec = skill_usage.get_record(name)
            except Exception:  # noqa: BLE001
                pass
            enriched.append({
                "name": name,
                "category": s.get("category"),
                "description": (s.get("description") or "")[:120],
                "provenance": prov,
                "active": bool(s.get("active")),
                "disabled": bool(s.get("disabled")),
                "use_count": rec.get("use_count", 0) if isinstance(rec, dict) else 0,
                "view_count": rec.get("view_count", 0) if isinstance(rec, dict) else 0,
                "pinned": rec.get("pinned", False) if isinstance(rec, dict) else False,
                "state": rec.get("state", "?") if isinstance(rec, dict) else "?",
                "created_at": rec.get("created_at") if isinstance(rec, dict) else None,
            })
    except Exception as exc:  # noqa: BLE001
        logger.warning("skill provenance enumeration failed: %s", exc)
    return {"skills": enriched, "total": len(enriched)}


def collect_mcp() -> Dict[str, Any]:
    """Live MCP server/tool state."""
    from tools import mcp_tool
    try:
        status = mcp_tool.get_mcp_status() or []
    except Exception as exc:  # noqa: BLE001
        status = []
        logger.warning("mcp status failed: %s", exc)
    return {"servers": status}


def collect_hooks() -> Dict[str, Any]:
    """Configured shell hooks + allowlist (config-level) alongside runtime hooks."""
    out: Dict[str, Any] = {}
    from agent import shell_hooks
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly() or {}
        specs = list(shell_hooks.iter_configured_hooks(cfg))
        out["configured"] = [
            {"event": s.event, "command": s.command,
             "match": s.matcher, "timeout": s.timeout,
             "fail_closed": bool(getattr(s, "fail_closed", False)),
             "allowed": _hook_allowed(s.event, s.command)}
            for s in specs
        ]
    except Exception as exc:  # noqa: BLE001
        out["configured"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _hook_allowed(event: str, command: str) -> bool:
    try:
        from agent import shell_hooks
        allowlist = shell_hooks.load_allowlist() or {}
        for entry in allowlist.get("approvals", []) or []:
            if entry.get("event") == event and entry.get("command") == command:
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


def collect_tools() -> Dict[str, Any]:
    """Tool registry snapshot (guarded; registry internals are read-only here)."""
    try:
        from tools.registry import registry
    except Exception as exc:  # noqa: BLE001
        return {"error": f"import: {type(exc).__name__}: {exc}"}
    tools = getattr(registry, "_tools", None)
    if not isinstance(tools, dict):
        return {"error": "registry._tools unavailable"}
    out = []
    for name, entry in sorted(tools.items()):
        try:
            out.append({
                "name": name,
                "toolset": getattr(entry, "toolset", "?"),
                "desc": (getattr(entry, "description", "") or "")[:100],
                "env": list(getattr(entry, "requires_env", []) or []),
            })
        except Exception:  # noqa: BLE001
            out.append({"name": name, "error": "unreadable entry"})
    return {"tools": out, "total": len(out)}


def collect_config(scope: Optional[str] = None) -> Dict[str, Any]:
    """Relevant configuration snapshot. Never returns raw secrets."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly() or {}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    def _safe(obj, path: List[str]) -> Any:
        cur = obj
        for p in path:
            cur = cur.get(p, {}) if isinstance(cur, dict) else {}
        return cur

    # Opt into a bounded, non-secret key set.
    out: Dict[str, Any] = {
        "plugins.enabled": _safe(cfg, ["plugins", "enabled"]),
        "plugins.disabled": _safe(cfg, ["plugins", "disabled"]),
        "plugins.entries": _safe(cfg, ["plugins", "entries"]),
        "skills.disabled": _safe(cfg, ["skills", "disabled"]),
        "skills.external_dirs": _safe(cfg, ["skills", "external_dirs"]),
        "skills.inline_shell": _safe(cfg, ["skills", "inline_shell"]),
        "mcp_servers": {
            k: {kk: vv for kk, vv in (v or {}).items() if kk != "env"}
            for k, v in (_safe(cfg, ["mcp_servers"]) or {}).items()
        },
        "agent.max_turns": _safe(cfg, ["agent", "max_turns"]),
        "agent.reasoning_effort": _safe(cfg, ["agent", "reasoning_effort"]),
        "agent.verify_on_stop": _safe(cfg, ["agent", "verify_on_stop"]),
        "approvals.mode": _safe(cfg, ["approvals", "mode"]),
        "memory.provider": _safe(cfg, ["memory", "provider"]),
        "delegation.model": _safe(cfg, ["delegation", "model"]),
        "_config_version": cfg.get("_config_version"),
    }
    # Registers (bounded: model + provider identity only, no keys).
    try:
        out["model"] = {
            "default": cfg.get("model", {}).get("default"),
            "provider": cfg.get("model", {}).get("provider"),
        }
    except Exception:  # noqa: BLE001
        pass
    return out


def _import_mod(name: str):
    """Import a module by importlib (tools/agent are on sys.path in the venv)."""
    import importlib
    return importlib.import_module(name)


def _safe_get(obj: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = obj
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
        if cur is None:
            return None
    return cur


def collect_soul30() -> Dict[str, Any]:
    """SOUL-3.0 modules + hermes-soul plugin + integration availability."""
    out: Dict[str, Any] = {}
    modules = ("agent.planner", "agent.experience", "agent.durable",
               "agent.orchestrator")
    for modname in modules:
        try:
            mod = _import_mod(modname)
            funcs = [n for n in dir(mod) if not n.startswith("_") and callable(getattr(mod, n, None))]
            out[modname] = {
                "importable": True,
                "functions": sorted(funcs)[:40],
            }
        except Exception as exc:  # noqa: BLE001
            out[modname] = {"importable": False, "error": f"{type(exc).__name__}: {exc}"}
    # integrations
    try:
        from agent.integrations import (  # type: ignore
            headroom_integration, halo_integration, codereviewgraph_integration,
            agentlens_integration, provenant_integration,
        )
        out["agent.integrations"] = {
            "headroom": _available(headroom_integration),
            "halo": _available(halo_integration),
            "codereviewgraph": _available(codereviewgraph_integration),
            "agentlens": _available(agentlens_integration),
            "provenant": _available(provenant_integration),
        }
    except Exception as exc:  # noqa: BLE001
        out["agent.integrations"] = {"error": f"{type(exc).__name__}: {exc}"}
    # hermes-soul plugin hooks (if registered)
    try:
        from hermes_cli.plugins import get_plugin_manager
        pm = get_plugin_manager()
        try:
            pm.discover_and_load()  # idempotent
        except Exception:  # noqa: BLE001
            pass
        soul = None
        for key, loaded in getattr(pm, "_plugins", {}).items():
            if key == "hermes-soul":
                err = getattr(loaded, "error", None)
                if not err and not getattr(loaded, "enabled", False):
                    err = "installed but not enabled"
                soul = {"hooks": list(getattr(loaded, "hooks_registered", []) or []),
                        "enabled": bool(getattr(loaded, "enabled", False)),
                        "error": err}
        out["plugin.hermes-soul"] = soul or {"hooks": [], "enabled": False,
                                             "error": "not loaded"
                                             if not getattr(pm, "_plugins", {}) else "not discovered"}
    except Exception as exc:  # noqa: BLE001
        out["plugin.hermes-soul"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _available(mod) -> bool:
    try:
        fn = getattr(mod, "is_available", None)
        if fn is None:
            return False
        return bool(fn() if callable(fn) else fn)
    except Exception:  # noqa: BLE001
        return False


def collect_runtime_services() -> Dict[str, Any]:
    """Gateway/cron/state DB presence — read-only snapshots."""
    out: Dict[str, Any] = {}
    try:
        hb = Path.home() / ".hermes" / "state" / "gateway.heartbeat"
        if hb.exists():
            out["gateway_heartbeat"] = hb.read_text(errors="replace")[:200]
        else:
            out["gateway_heartbeat"] = None
    except Exception as exc:  # noqa: BLE001
        out["gateway_heartbeat"] = f"error: {exc}"
    try:
        cron_dir = Path.home() / ".hermes" / "cron"
        jobs = []
        f = cron_dir / "jobs.json"
        if f.exists():
            data = json.loads(f.read_text(errors="replace"))
            jobs = data if isinstance(data, list) else data.get("jobs", [])
        out["cron"] = [
            {"id": j.get("id"), "name": j.get("name"),
             "enabled": j.get("enabled", True)}
            for j in jobs[:10]
        ]
    except Exception as exc:  # noqa: BLE001
        out["cron"] = f"error: {exc}"
    # state.db schema tables
    try:
        import sqlite3
        db = Path.home() / ".hermes" / "state.db"
        if db.exists():
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            tables = [r[0] for r in con.execute("select name from sqlite_master where type='table' order by name")]
            con.close()
            out["state_tables"] = tables[:30]
    except Exception as exc:  # noqa: BLE001
        out["state_tables"] = f"error: {exc}"
    return out


# ---------------------------------------------------------------------------
# Environment graph
# ---------------------------------------------------------------------------

def build_env_graph(include_heavy: bool = True) -> Dict[str, Any]:
    """One deterministic snapshot of the live environment."""
    graph: Dict[str, Any] = {
        "generated_at": _now(),
        "hermes_version": "<unknown>",
        "skills": collect_skills(),
    }
    try:
        from utils import get_install_info
        info = get_install_info()
        graph["hermes_version"] = info.get("version") or "<unknown>"
    except Exception:  # noqa: BLE001
        pass
    if graph["hermes_version"] == "<unknown>":
        try:
            import subprocess
            out = subprocess.run(
                ["git", "-C", str(Path(__file__).resolve().parents[1]),
                 "log", "-1", "--format=%h %cs"],
                capture_output=True, text=True, timeout=5,
            )
            graph["hermes_version"] = "git " + (out.stdout.strip() or "?")
        except Exception:  # noqa: BLE001
            pass
    graph["plugins"] = collect_plugins()
    graph["mcp"] = collect_mcp()
    graph["hooks"] = collect_hooks()
    graph["tools"] = collect_tools()
    graph["config"] = collect_config()
    graph["soul30"] = collect_soul30()
    graph["runtime"] = collect_runtime_services()
    return graph


# ---------------------------------------------------------------------------
# Error analysis — deterministic mapping
# ---------------------------------------------------------------------------

def match_subsystem(error_text: str) -> Tuple[str, str, Tuple[str, ...]]:
    """(subsystem_label, execution_context, candidate_hook_families) or
    fallback ('Other', '')."""
    low = (error_text or "").lower()
    for pattern, label, context, hooks in _ERROR_MAP:
        if re.search(pattern, low):
            return label, context, hooks
    return "Other", "", ()


def plugin_candidates_snapshot() -> List[Dict[str, Any]]:
    """Flatten plugin entries into candidate rows with registered hooks."""
    from hermes_cli.plugins import get_plugin_manager
    pm = get_plugin_manager()
    try:
        pm.discover_and_load()
    except Exception:  # noqa: BLE001
        pass
    hooks_by_plugin = plugin_hooks_by_name(pm)
    rows: List[Dict[str, Any]] = []
    for info in pm.list_plugins():
        key = info.get("key") or info.get("name") or "?"
        rows.append({
            "name": info.get("name"),
            "key": key,
            "kind": info.get("kind", "standalone"),
            "source": info.get("source", "?"),
            "version": info.get("version", ""),
            "enabled": bool(info.get("enabled")),
            "error": info.get("error"),
            "hooks_registered": list(hooks_by_plugin.get(key, []) or []),
            "tools_registered": info.get("tools", 0),
            "commands": info.get("commands", 0),
        })
    return rows


def classify_plugin_for_error(row: Dict[str, Any], subsystem: str,
                              path_hooks: Tuple[str, ...]) -> Tuple[str, List[str]]:
    """Classify ONE plugin row against an error's subsystem + hook path.

    Returns (classification, evidence_lines).
    """
    evidence: List[str] = []
    name = row.get("name") or row.get("key") or "?"
    enabled = bool(row.get("enabled"))
    evidence.append(f"enabled={enabled}")
    hooks = list(row.get("hooks_registered") or [])
    hook_set = set(hooks)

    if not enabled:
        err = row.get("error") or ""
        evidence.append(f"state: {err}" if err else "not enabled in config")
        return UNLIKELY, evidence

    if row.get("error"):
        evidence.append(f"load error: {row['error']}")
        return UNKNOWN, evidence

    if not hooks:
        evidence.append("registers no hooks")
        if row.get("tools_registered"):
            return MERELY_PRESENT, evidence + [f"provides {row['tools']} tool(s) but no hooks"]
        return NOT_CAPABLE, evidence

    # hooks intersect the error's wire.
    relevant = sorted(set(hooks) & set(path_hooks))
    if relevant:
        for h in relevant:
            authority = []
            if h in _TOOL_WIRE_HOOKS:
                authority.append("can block/modify tool execution")
            if h in _LLM_WIRE_HOOKS:
                authority.append("can modify LLM context/request")
            if h in _GATEWAY_WIRE_HOOKS:
                authority.append("can block/rewrite gateway message")
            evidence.append(f"registers {h}" + (f" ({', '.join(authority)})" if authority else ""))
        if any(h in _TOOL_WIRE_HOOKS or h in _GATEWAY_WIRE_HOOKS or h == "pre_llm_call"
               or h == "pre_api_request" for h in relevant):
            return PLAUSIBLE, evidence
        return INVOLVED, evidence

    # registered hooks exist but none on this path
    evidence.append("registers hooks but NOT on this path: " + ", ".join(sorted(set(hooks))[:6]))
    return NOT_CAPABLE, evidence


def analyze_error(error_text: str, *, max_candidates: int = 30) -> Dict[str, Any]:
    """Full error→interference analysis with deterministic evidence."""
    subsystem, execution_context, hint_hooks = match_subsystem(error_text)
    path_hooks = hint_hooks or _SUBSYSTEM_HOOKS.get(subsystem, ())

    candidates: List[Dict[str, Any]] = []
    try:
        rows = plugin_candidates_snapshot()
    except Exception as exc:  # noqa: BLE001
        rows = []
        candidates.append({"name": "<plugin-snapshot>",
                           "classification": UNKNOWN,
                           "evidence": [f"plugin snapshot failed: {exc}"]})
    for row in rows:
        try:
            cls, evidence = classify_plugin_for_error(row, subsystem, path_hooks)
        except Exception as exc:  # noqa: BLE001
            cls, evidence = UNKNOWN, [f"classifier error: {exc}"]
        candidates.append({
            "name": row.get("name") or row.get("key"),
            "key": row.get("key"),
            "kind": row.get("kind"),
            "source": row.get("source"),
            "classification": cls,
            "evidence": evidence,
        })
    # Order: strongest relevance first, deterministic tiebreak by name.
    _PRIORITY = {PLAUSIBLE: 0, INVOLVED: 1, MERELY_PRESENT: 2,
                 UNLIKELY: 3, NOT_CAPABLE: 4, UNKNOWN: 5}
    candidates.sort(key=lambda c: (_PRIORITY.get(c["classification"], 5),
                                    str(c.get("key", ""))))
    candidates = candidates[:max_candidates]

    # Non-plugin families (MCP servers, shell hooks) — deterministic.
    related_mcp = []
    if "MCP" in subsystem or "mcp" in error_text.lower():
        try:
            status = collect_mcp().get("servers") or []
            related_mcp = [
                {"name": s.get("name"), "transport": s.get("transport"),
                 "tools": s.get("tools"), "connected": s.get("connected"),
                 "status": s.get("status")}
                for s in status
            ]
        except Exception:  # noqa: BLE001
            related_mcp = []

    related_hooks = []
    if "hook" in (error_text or "").lower():
        try:
            h = collect_hooks().get("configured") or []
            if isinstance(h, list):
                related_hooks = h
        except Exception:  # noqa: BLE001
            pass

    return {
        "error": error_text,
        "subsystem": subsystem,
        "execution_path": execution_context or "unknown",
        "execution_hooks": list(path_hooks),
        "platform": getattr(sys, "platform", "?"),
        "candidates": candidates,
        "mcp_servers": related_mcp,
        "shell_hooks": related_hooks,
        "note": (
            "INVOLVED = registered a hook that is part of this subsystem's wire; "
            "PLAUSIBLE = registered a hook that can block/modify/rewrite on this path; "
            "MERELY PRESENT = loaded, no matching behavior; NOT CAPABLE = no hooks on path; "
            "UNLIKELY = disabled/not loaded; UNKNOWN = unable to determine."
        ),
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Auto-diagnostic trigger policy (Part 1 of the autonomous workflow)
# ---------------------------------------------------------------------------

# Deterministic trigger vocabulary. An error text or user request matching one
# of these patterns may involve Hermes' own environment. Case-insensitive.
_AUTO_TRIGGER_SIGNALS = (
    re.compile(r"plugin\s*(policy|error|fail|block|load)", re.I),
    re.compile(r"pre_tool_call|post_tool_call|transform_tool|blocked by", re.I),
    re.compile(r"mcp\s*server|mcp__|mcp tool", re.I),
    re.compile(r"skill\s*(load|error|fail|not found)|SKILL\.md", re.I),
    re.compile(r"hook\s*(error|fail)|middleware", re.I),
    re.compile(r"gateway\s*(error|fail|crash)|dispatch", re.I),
    re.compile(r"provider\s*(error|fail|auth)|api[ _]?key|credential", re.I),
    re.compile(r"config(uration)?\s*(error|fail|invalid|missing|parse)", re.I),
    re.compile(r"hermes|agent\s*loop|SOUL|orchestrator|experience|planner|durable", re.I),
    re.compile(r"tool\s*(blocked|denied|policy|not found)", re.I),
)

# Explicit terms that would normally NOT warrant environment diagnostics.
_AUTO_SKIP_SIGNALS = [
    re.compile(r"^weather|^who|^what is |^translates?|^summarize restaurant"),
]


def should_trigger_diagnostics(text: str, *, context: str = "") -> Tuple[bool, str]:
    """Decide whether an error text or user request deserves auto-diagnostics.

    Returns (should_diagnose, reason). Deterministic: regex vocabulary over
    the request/error. No LLM call; conservative (prefer trigger when
    ambiguous, cheap read-only).
    """
    if not text or not text.strip():
        return False, "empty text"

    # Soft skip signals first (informational prompts).
    for pat in _AUTO_SKIP_SIGNALS:
        if pat.match(text.strip()):
            return False, f"skipped by pattern: {pat.pattern}"

    hits = [pat.pattern for pat in _AUTO_TRIGGER_SIGNALS if pat.search(text)]
    # Context (tool_result/error) adds evidence even when the message itself
    # has no trigger vocabulary (e.g. "run this again" after an MCP error).
    ctx_hits = [pat.pattern for pat in _AUTO_TRIGGER_SIGNALS if context and pat.search(context)]
    if hits or ctx_hits:
        why = " ; ".join(dict.fromkeys(hits + ctx_hits + ["context"]))[:160]
        return True, why
    if re.search(r"error|failed|traceback|exception", text, re.I):
        return True, "generic error wording"
    return False, "no environment signal"


def render_auto_diagnostic(text: str, error_text: str = "") -> Dict[str, Any]:
    """Build the compact evidence block the agent consumes after a trigger.

    The agent sees a distilled, evidence-backed call-to-action, not a raw dump.
    """
    src = error_text or text or ""
    try:
        r = analyze_error(src, max_candidates=10)
        top = r["candidates"][:5] if r.get("candidates") else []
        strongest = top[0] if top and top[0]["classification"] in (PLAUSIBLE, INVOLVED) else None
        return {
            "auto": True,
            "trigger_on": text[:160],
            "subsystem": r.get("subsystem"),
            "execution_path": r.get("execution_path"),
            "execution_hooks": r.get("execution_hooks"),
            "strongest_candidate": {
                "name": strongest.get("name"),
                "classification": strongest.get("classification"),
                "evidence": strongest.get("evidence", [])[:4],
            } if strongest else None,
            "candidate_summary": [
                {"name": c.get("name"), "classification": c.get("classification")}
                for c in top
            ],
            "suggest": (
                "Run diag_env(action='analyze', error=...) for full evidence "
                "and components, then isolate via a temporary profile, not by "
                "disabling production plugins silently."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"auto": True, "error": f"render failed: {type(exc).__name__}: {exc}"}


# Worker-status / supervisor support (Part 8) — minimal, registry-free helpers
# ---------------------------------------------------------------------------

WORKER_STATUSES = ("INVESTIGATING", "DIAGNOSING", "IMPLEMENTING", "TESTING",
                   "VERIFYING", "BLOCKED", "FAILED", "COMPLETE")


def worker_status_valid(status: str) -> bool:
    return (status or "").upper() in WORKER_STATUSES


def render_worker_status(name: str, status: str, evidence: List[str],
                         step: str = "") -> Dict[str, Any]:
    """Deterministic worker state report for supervisor consumption.

    The supervisor's job is to NOT blindly accept a worker's claim; this
    renderer keeps the shape strictly structured.
    """
    return {
        "worker": name,
        "status": status.upper() if worker_status_valid(status) else "UNKNOWN",
        "step": step,
        "evidence": list(evidence or []),
        "reported_at": _now(),
    }


def validate_worker_completion(worker: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Supervisor-side gate: does the worker actually have evidence of done?

    Returns (accepted, reasons). Rejects anything that just claims 'done'
    without verification evidence.
    """
    problems: List[str] = []
    if not worker or not isinstance(worker, dict):
        return False, ["missing worker report"]
    status = worker.get("status", "").upper()
    if status not in ("COMPLETE", "VERIFYING", "TESTING"):
        problems.append(f"status {status or '?'} is not a completion state")
    # Workers persist evidence under the canonical completion_evidence field
    # (supervisor state protocol v5); `evidence` is the legacy alias. The
    # validator MUST read the field workers actually write — a worker whose
    # real artifacts land only in completion_evidence was wrongly judged
    # UNVERIFIED forever while its process stayed alive (s6-ownership audit,
    # 2026-08-19: loop spun COMPLETE->UNVERIFIED_COMPLETION [VERIFY] with an
    # idle live worker because evidence=[] while completion_evidence had 3
    # artifact lines). Prefer completion_evidence; fall back to legacy.
    evidence = (worker.get("completion_evidence")
                or worker.get("evidence") or [])
    if isinstance(evidence, str):
        evidence = [evidence]
    # Bare "done" / restated goal is NOT verification. Require a real signal:
    # a test run, an assertion, an execution/check result — OR a durable
    # artifact the semantic gate can ground (real file under the workdir /
    # git object / structured result). Consistent with the s5 semantic
    # evidence model: words are hints, artifacts are the proof.
    has_test = any(
        re.search(r"(test|pass(ed|es|ing)?|verified|assert|check|run )+", str(e), re.I)
        for e in evidence
    )
    if not evidence:
        problems.append("no evidence lines provided")
    else:
        try:
            # the s5 artifact classifier needs a workdir: prefer the
            # worker's recorded workdir, else the agent repo root (the
            # only durable path we can ground files against).
            _wd = worker.get("workdir") or os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            try:
                from hermes_cli.supervisor import _evidence_kinds  # noqa: PLC0415
            except Exception:  # pragma: no cover - standalone diagnostic import
                _evidence_kinds = None
            if _evidence_kinds is not None:
                has_artifact = any(
                    _evidence_kinds(str(e), _wd)
                    for e in evidence
                )
            else:
                has_artifact = False
        except Exception:
            has_artifact = False
        if not (has_test or has_artifact):
            problems.append(
                "evidence does not include verification wording (tests/asserts/runs) "
                "nor a groundable artifact (file/git-object/structured-result)")
    # Refuse claims that only restate the goal.
    if worker.get("claimed") or status == "COMPLETE":
        if len(evidence) < 2:
            problems.append("completion claimed without at least two evidence lines")
    return not problems, problems


# ---------------------------------------------------------------------------
# Safe isolation plans (Part 4) — deterministic, never mutating
# ---------------------------------------------------------------------------

def isolation_plan(component: Dict[str, Any],
                   subsystem: str = "Tool execution") -> Dict[str, Any]:
    """Suggest the safest isolation experiment for a component.

    Returns only proposals; nothing is executed. The plan names the expected
    result and the restoration step. Supports plugins + MCP servers + shell
    hooks by kind.
    """
    base = {
        "component": component.get("name") or component.get("key") or "?",
        "subsystem": subsystem,
        "safe_methods": [],
        "expected_result": "hypothesis confirmed or rejected",
        "restore": "remove temporary HOME / reset the test profile",
    }
    kind = component.get("kind") or ""
    key = component.get("key") or ""
    if kind == "user" or kind in ("standalone", "backend"):
        base["safe_methods"] = [
            "temporary HERMES_HOME with plugins.<name> disabled",
            "run hermes diag analyze same error to confirm candidate drops from PLAUSIBLE list",
        ]
        base["expected_result"] = "same error reproduces without the plugin (proves it is not the cause)"
    elif "mcp" in str(subsystem).lower() or key.startswith("mcp__"):
        base["safe_methods"] = [
            "temporary HERMES_HOME with mcp_servers.<name>.enabled=false",
            "hermes diag mcp shows the server disabled",
        ]
        base["expected_result"] = "MCP tools disappear; error changes or disappears"
    else:
        base["safe_methods"] = [
            f"configure {key} off in a scratch profile and rerun",
            "hermes diag analyze <same error> to compare candidate evidence",
        ]
    base["safety_rule"] = (
        "Never disable a production component to test a hypothesis. "
        "Use temporary profiles/HERMES_HOME or HERMES_SAFE_MODE; restore is automatic."
    )
    return base