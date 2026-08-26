"""``hermes diag`` subcommand — read-only environment diagnostics.

Wired into ``hermes_cli/main.py`` like other subcommands:

    hermes diag inventory            full environment snapshot
    hermes diag plugins              plugin registry snapshot
    hermes diag skills               skills + provenance snapshot
    hermes diag mcp                  MCP server/tool state
    hermes diag hooks                configured shell hooks + runtime hooks
    hermes diag config               relevant configuration values
    hermes diag soul                 SOUL-3.0 modules + hermes-soul plugin
    hermes diag tools                tool registry snapshot
    hermes diag runtime              gateway/cron/state services
    hermes diag analyze "<error>"    error → interference analysis
    hermes diag path "<error>"       subsystem/path mapping only

All subcommands are read-only; nothing in this module writes config or
enables/disables components. Human-readable table output, ``--json`` for
machine consumption.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from hermes_cli import diagnostics as _diag


def run_diagnostics_command(args) -> int:
    """Public entry used by ``hermes diag`` (injected via cmd_diagnostics)."""
    action = getattr(args, "diag_action", None) or "inventory"
    error = getattr(args, "error", "") or ""
    # argparse quirk (measured): a parent-parser flag (`diag --json
    # inventory`) is set on the root Namespace but the subcommand's own
    # set_defaults namespace shadows it — json ends False even though the
    # user passed it. Normalize: if the root namespace saw --json, force it
    # through so both invocation orders emit JSON.
    root_json = dict(vars(args)).get("json", False)
    if root_json:
        args.json = True
    return _dispatch(args, _DIAG_ACTIONS.get(action, action), error=error)


_DIAG_ACTIONS = {
    "inventory": "inventory",
    "plugins": "plugins",
    "skills": "skills",
    "mcp": "mcp",
    "hooks": "hooks",
    "config": "config",
    "soul": "soul",
    "tools": "tools",
    "runtime": "runtime",
    "analyze": "analyze",
}


def build_diagnostics_parser(subparsers, *, cmd_diagnostics) -> None:
    parser = subparsers.add_parser(
        "diag",
        help="Inspect the Hermes runtime and trace errors to interfering components (read-only)",
        description=(
            "Read-only environment introspection: component graph, plugin/skill/MCP/hook/"
            "config/SOUL-3.0 state, and deterministic error→interference analysis. "
            "Never modifies configuration."
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit raw JSON instead of tables"
    )
    sub = parser.add_subparsers(dest="diag_action")

    p_inv = sub.add_parser("inventory", help="Full environment snapshot")
    # compat: accept --json on the sub-subcommand too (tests and docs use both
    # `diag --json inventory` and `diag inventory --json`; argparse only knew
    # the former after the upstream main reset re-registered this router)
    p_inv.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    p_inv.set_defaults(func=lambda a: _dispatch(a, "inventory"))

    for name, help_txt in [
        ("plugins", "Plugin registry snapshot"),
        ("skills", "Skill + provenance snapshot"),
        ("mcp", "MCP server/tool state"),
        ("hooks", "Configured shell hooks"),
        ("config", "Relevant configuration"),
        ("soul", "SOUL-3.0 modules and hermes-soul health"),
        ("tools", "Tool registry snapshot"),
        ("runtime", "Gateway/cron/state services"),
    ]:
        p = sub.add_parser(name, help=help_txt)
        p.set_defaults(func=lambda a, n=name: _dispatch(a, n))

    p_an = sub.add_parser("analyze", help="Trace an error through the component graph")
    p_an.add_argument("error", help="Error text or symptom to trace")
    p_an.add_argument("--max-candidates", type=int, default=30)
    p_an.set_defaults(func=lambda a: _dispatch(a, "analyze",
                                               error=args_error_or_none(a)))

    p_iso = sub.add_parser("isolate", help="Propose a safe isolation experiment")
    p_iso.add_argument("name", help="Component key/name (e.g. security-guidance)")
    p_iso.add_argument("--subsystem", default="Tool execution",
                       help="Subsystem label for the proposal")
    p_iso.set_defaults(func=lambda a: _dispatch(a, "isolate",
                                                name=str(getattr(a, "name", "") or "")))

    p_w = sub.add_parser("worker", help="Validate a worker completion report")
    p_w.add_argument("--report", default="", help="Worker report JSON (optional)")
    p_w.set_defaults(func=lambda a: _dispatch(a, "worker",
                                              error=str(getattr(a, "report", "") or "")))

    parser.set_defaults(func=lambda a: _dispatch(a, "inventory"))


def args_error_or_none(a) -> str:
    return getattr(a, "error", "") or getattr(a, "name", "") or ""


def _dispatch(args, action: str, error: str = "", name: str = "") -> int:
    try:
        if action == "analyze":
            payload = _diag.analyze_error(error, max_candidates=getattr(args, "max_candidates", 30) or 30)
        elif action == "isolate":
            comp = {"name": name or error, "key": name or error}
            # Enrich from the live plugin registry so the plan uses the real kind.
            try:
                for row in _diag.plugin_candidates_snapshot():
                    if row.get("key") == (name or error) or row.get("name") == (name or error):
                        comp = row
                        break
            except Exception:
                pass
            payload = _diag.isolation_plan(
                comp,
                subsystem=getattr(args, "subsystem", "Tool execution") or "Tool execution")
        elif action == "worker":
            if error:
                try:
                    payload = _diag.validate_worker_completion(json.loads(error))
                except Exception:
                    payload = {"error": "worker report must be a JSON object",
                               "usage": '{"status":"COMPLETE","evidence":[...]}'}
            else:
                payload = {"statuses": list(_diag.WORKER_STATUSES),
                           "hint": "pass --report JSON to validate a worker report"}
        elif action == "inventory":
            payload = _diag.build_env_graph()
        elif action == "plugins":
            payload = _diag.collect_plugins()
        elif action == "skills":
            payload = _diag.collect_skills()
        elif action == "mcp":
            payload = _diag.collect_mcp()
        elif action == "hooks":
            payload = _diag.collect_hooks()
        elif action == "config":
            payload = _diag.collect_config()
        elif action == "soul":
            payload = _diag.collect_soul30()
        elif action == "tools":
            payload = _diag.collect_tools()
        elif action == "runtime":
            payload = _diag.collect_runtime_services()
        else:
            payload = {"error": f"unknown diag action {action!r}"}
    except Exception as exc:  # noqa: BLE001
        payload = {"error": f"diag {action} failed: {type(exc).__name__}: {exc}"}

    # argparse parent-flag loss (measured): with sub-subcommands present,
    # `diag --json inventory` parses as json=False even though the flag was
    # given. --json may appear before or after the subcommand; the CLI
    # contract is "emit JSON if --json is present anywhere" — read the raw
    # argv so BOTH orders emit JSON consistently.
    as_json = bool(getattr(args, "json", False)) or "--json" in sys.argv
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0
    _print_human(action, payload)
    return 0


def _print_human(action: str, payload: Dict[str, Any]) -> None:
    if "error" in payload and len(payload) <= 2:
        print(f"diag {action}: ERROR {payload['error']}")
        return

    if action == "analyze":
        _print_analysis(payload)
        return
    if action == "plugins":
        for p in payload.get("plugins", []):
            print(f"  {p.get('key','?'):28s} {p.get('kind','?'):12s} "
                  f"{'enabled' if p.get('enabled') else 'disabled':9s} "
                  f"tools={p.get('tools',0)} hooks={p.get('hooks',0)} "
                  f"{('err: '+str(p.get('error'))) if p.get('error') else ''}")
        return
    if action == "skills":
        skills = payload.get("skills") or []
        from collections import Counter
        pc = Counter(s.get("provenance","?") for s in skills)
        print(f"  total={len(skills)} provenance={dict(pc)}")
        for s in skills[:15]:
            print(f"  {s.get('name','?'):36s} {s.get('provenance','?'):9s} "
                  f"use={s.get('use_count',0)} state={s.get('state','?')} "
                  f"cat={s.get('category') or '-'}")
        return
    if action == "mcp":
        for s in payload.get("servers") or []:
            print(f"  {s.get('name','?'):22s} {s.get('transport','?'):7s} "
                  f"tools={s.get('tools',0)} {s.get('status','?'):10s} "
                  f"{'connected' if s.get('connected') else 'not-connected'}")
        return
    if action == "hooks":
        cfg = payload.get("configured") or []
        if isinstance(cfg, list):
            for h in cfg or []:
                allowed = "✓" if h.get("allowed") else "✗ not-allowlisted"
                print(f"  [{h.get('event','?')}] {h.get('command','?')} "
                      f"fail_closed={h.get('fail_closed',False)} {allowed}")
            if not cfg:
                print("  (no shell hooks configured)")
        return
    if action == "config":
        for k, v in payload.items():
            print(f"  {k}: {json.dumps(v, ensure_ascii=False)[:120]}")
        return
    if action == "soul":
        for mod, info in payload.items():
            if isinstance(info, dict) and "importable" in info:
                ok = "OK" if info.get("importable") else "MISSING"
                extra = f" funcs={len(info.get('functions', []))}" if info.get("functions") else ""
                err = info.get("error", "")
                print(f"  {mod:24s} {ok}{extra} {err}")
            elif isinstance(info, dict) and "hooks" in info:
                hooks = ", ".join(info.get("hooks", []) or []) or "-"
                enabled = "enabled" if info.get("enabled") else "disabled"
                err = info.get("error") or ""
                print(f"  {mod:24s} {enabled:9s} hooks=[{hooks}] {err}")
            elif isinstance(info, dict) and "error" in info:
                print(f"  {mod:24s} ERROR {info['error']}")
            else:
                print(f"  {mod}: {info}")
        return
    if action == "tools":
        tools = payload.get("tools") or []
        print(f"  total={payload.get('total', len(tools))}")
        for t in tools[:15]:
            print(f"  {t.get('name','?'):28s} toolset={t.get('toolset','?')}")
        return
    # inventory default: summary + top items
    if action == "inventory":
        print(f"generated_at: {payload.get('generated_at','?')}  ver: {payload.get('hermes_version','?')}")
        pl = payload.get("plugins", {})
        sk = payload.get("skills", {})
        print(f"plugins: {len(pl.get('plugins', []))} total / hooks: {len(pl.get('hooks', {}))} events")
        print(f"skills: {sk.get('total', 0)} total")
        mc = payload.get("mcp", {}).get("servers") or [{"name": "?"}]
        print(f"mcp servers: {len(mc)}")
        return
    print(json.dumps(payload, ensure_ascii=False, default=str)[:800])


def _print_analysis(payload: Dict[str, Any]) -> None:
    print(f"SUBSYSTEM: {payload.get('subsystem','?')}")
    print(f"EXECUTION PATH: {payload.get('execution_path','?')}")
    print(f"EXECUTION HOOKS: {', '.join(payload.get('execution_hooks', [])) or 'none'}")
    print("CANDIDATES (ranked):")
    for c in payload.get("candidates", []):
        cls = c.get("classification", "?")
        marker = {
            "PLAUSIBLE SOURCE OF INTERFERENCE": "★ ",
            "INVOLVED": "● ",
            "MERELY PRESENT": "○ ",
            "UNLIKELY": "· ",
            "NOT CAPABLE OF AFFECTING THIS PATH": "· ",
            "UNKNOWN": "? ",
        }.get(cls, "· ")
        print(f"  {marker}{c.get('name','?'):26s} {cls}")
        for ev in c.get("evidence", [])[:5]:
            print(f"      - {ev}")
    if payload.get("mcp_servers"):
        print("MCP SERVERS (present):")
        for s in payload["mcp_servers"][:8]:
            print(f"  - {s.get('name')} {s.get('status')} tools={s.get('tools')}")
    if payload.get("shell_hooks"):
        print("SHELL HOOKS (configured):")
        for h in payload["shell_hooks"][:8]:
            print(f"  - {h.get('event')} {h.get('command')}")
    print(f"NOTE: {payload.get('note','')}")