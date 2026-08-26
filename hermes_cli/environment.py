"""Env + capability probe: a durable, refreshable model of THIS execution
environment and what Hermes can actually do here. Facts vs assumptions
distinguished; never trusted blindly (refresh via probe).

P-25 audit found NO durable env model: `environment_probe` config exists but
probes nothing persistent; the curl_cffi case proved capability claims can be
stale. This module writes ~/.hermes-supervisor/env.json once per probe and
answers "what resource/capability does this task need and do we have it?".
Updated (P-26 r15): the probe also inventories INSTALLED skills, plugins and
configured MCP servers, so the capability router answers from disk truth, not
a static keyword table.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def env_path() -> Path:
    base = os.environ.get("HERMES_SUPERVISOR_DIR") or os.path.expanduser(
        "~/.hermes-supervisor")
    d = Path(base)
    d.mkdir(parents=True, exist_ok=True)
    return d / "env.json"


def _skills_dir() -> Path:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / "skills"


def _plugins_dir() -> Path:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / "plugins"


def _config_path() -> Path:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / "config.yaml"

# Rough keyword map: config section name -> what the router calls it. We only
# read section KEYS (names), never values: no credentials leave config.yaml.
_MCP_SECTION_HINTS = ("mcp_servers", "mcp", "mcp.servers")

def inventory_skills(limit: int = 400) -> List[str]:
    """Installed skills under HERMES_HOME/skills (top-level SKILL.md trees).

    Mirrors what the skill store loads: each dir containing SKILL.md is a
    skill. Bounded; returns the names (dir basenames), not content.
    """
    d = _skills_dir()
    out: List[str] = []
    if not d.is_dir():
        return out
    for child in sorted(d.iterdir()):
        try:
            if child.is_dir() and (child / "SKILL.md").is_file():
                out.append(child.name)
            elif child.is_dir() and child.name.startswith("."):
                continue
        except OSError:
            continue
        if len(out) >= limit:
            break
    return out


def inventory_plugins(limit: int = 100) -> List[str]:
    """Installed plugins: dirs under HERMES_HOME/plugins with a manifest."""
    d = _plugins_dir()
    out: List[str] = []
    if not d.is_dir():
        return out
    for child in sorted(d.iterdir()):
        try:
            if not child.is_dir():
                continue
            has_meta = (child / "plugin.json").is_file() or \
                       (child / "SKILL.md").is_file() or \
                       (child / "manifest.json").is_file()
            if has_meta:
                out.append(child.name)
        except OSError:
            continue
        if len(out) >= limit:
            break
    return out


def inventory_mcp_servers(limit: int = 100) -> List[str]:
    """Configured MCP server NAMES from config.yaml. Reads only section keys;
    never the values (which may embed auth/URLs). Non-fatal on any parse issue.
    """
    p = _config_path()
    out: List[str] = []
    if not p.is_file():
        return out
    try:
        import yaml  # local import: only needed for inventory
        with open(p, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return out
    if not isinstance(cfg, dict):
        return out
    for sec_name in _MCP_SECTION_HINTS:
        sec = cfg.get(sec_name)
        if isinstance(sec, dict):
            for name in sec.keys():
                if isinstance(name, str) and name not in out:
                    out.append(name)
    return out[:limit]


def _mod(name: str) -> bool:
    import importlib
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def probe_env(refresh: bool = False) -> Dict[str, Any]:
    """Return the model, probing fresh when absent or when refresh=True.
    Snapshot fields are labelled facts; capacity fields are reasoning facts
    (e.g. cpu_count, python version) not raw `uname` trivia.
    """
    if not refresh:
        p = env_path()
        if p.exists():
            try:
                data = json.loads(p.read_text())
                if data.get("_probed_at") and time.time() - data["_probed_at"] < 3600:
                    return data
            except Exception:
                pass
    import platform
    data: Dict[str, Any] = {
        "_probed_at": time.time(),
        "platform": platform.system(),
        "release": platform.release(),
        "python": platform.python_version(),
        "cpus_logical": os.cpu_count() or 1,
        "cpus_available": (len(os.sched_getaffinity(0))
                           if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)),
        "mem_total_kb": None,
        "providers_in_use": [],
        "capabilities": {},
        "skills": [],
        "plugins": [],
        "mcp_servers": [],
    }
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    data["mem_total_kb"] = int(line.split()[1])
                    break
    except OSError:
        pass
    # capability compass: what backends/runners exist in THIS environment.
    data["capabilities"] = {
        "curl_cffi": _mod("curl_cffi"),
        "scrapling": _mod("scrapling"),
        "stealthy": _mod("scrapling.stealthy"),
        "playwright": _mod("playwright"),
        "httpx": _mod("httpx"),
        "trafilatura": _mod("trafilatura"),
        "pymupdf": _mod("fitz"),
        "pdfminer": _mod("pdfminer"),
    }
    # live inventory: what skills/plugins/MCP are actually available (r15).
    # These are cheap directory/config reads; failures degrade to empty lists.
    try:
        data["skills"] = inventory_skills()
    except Exception:
        data["skills"] = []
    try:
        data["plugins"] = inventory_plugins()
    except Exception:
        data["plugins"] = []
    try:
        data["mcp_servers"] = inventory_mcp_servers()
    except Exception:
        data["mcp_servers"] = []
    env_path().write_text(json.dumps(data, default=str), encoding="utf-8")
    return data


def needs(question: str, data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Best-effort capability suggestion for a task string. Pure keyword
    heuristic; the agent's real routing is judgment — this only surfaces
    concrete options that are TRUE in this environment (no phantom claims).
    """
    data = data or probe_env()
    q = question.lower()
    out: List[Dict[str, Any]] = []
    if any(w in q for w in ("crawl", "scrape", "fetch", "html", "website", "research")):
        for name in ("httpx", "trafilatura", "scrapling", "stealthy", "playwright", "curl_cffi"):
            if data["capabilities"].get(name) and name != "playwright":
                out.append({"capability": name, "kind": "backend"})
        if data["capabilities"].get("playwright"):
            out.append({"capability": "playwright", "kind": "browser"})
    if any(w in q for w in ("pdf", "scan", "ocr")):
        for name in ("pdfminer", "pymupdf"):
            if data["capabilities"].get(name):
                out.append({"capability": name, "kind": "pdf"})
    return out