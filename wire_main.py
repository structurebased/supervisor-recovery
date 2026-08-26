#!/usr/bin/env python3
"""wire_main.py — idempotent supervisor/diag + fast-path wiring for
hermes_cli/main.py after an upstream reset strips the local registration.

Part of the permanent anti-reset guard (~/.hermes-supervisor/recovery/).
Called by restore-supervisor.sh. Pure stdlib; safe to run with any python3.

Exits:
  0  wired (or already wired)
  2  a load-bearing anchor is missing (upstream main changed shape)
  3  main.py not found
"""

import re
import sys


# The fast-path body inserted before `def _plugin_cli_discovery_needed` when
# the upstream main.py lacks it (perf: `hermes supervise` spawn ~1.4s -> ~0.4s).
_FAST_PATH_DEF = '''\n\ndef _try_supervise_fast_path() -> bool:\n    """Short-circuit `hermes supervise ...` to a bare argparse + only the\n    supervise subtree, skipping the ~430-parser tree and plugin discovery.\n    Falls through (returns False) on ambiguity/error so the full CLI owns\n    error handling. Same dispatch as the full path - startup only.\n    """\n    if sys.argv[1:2] != ["supervise"]:\n        return False\n    if os.environ.get("HERMES_DISABLE_SUPERVISE_FAST") == "1":\n        return False\n    try:\n        import argparse\n        from hermes_cli.subcommands.supervise import (\n            build_supervise_parser,\n            run_supervise_command,\n        )\n        parser = argparse.ArgumentParser(prog="hermes")\n        sub = parser.add_subparsers(dest="command")\n        build_supervise_parser(sub, cmd_supervise=run_supervise_command)\n        args = parser.parse_args(sys.argv[1:])\n        if hasattr(args, "func"):\n            rc = args.func(args)\n            if isinstance(rc, int) and rc != 0:\n                sys.exit(rc)\n            return True\n        parser.print_help()\n        return True\n    except KeyboardInterrupt:\n        raise\n    except SystemExit:\n        raise\n    except Exception:\n        return False\n'''

_FAST_PATH_CALL = '''\n    # supervise fast path: pure supervisor/mission CLI; a bare root parser\n    # + only the supervise subtree is behaviorally identical and ~4x faster.\n    # Restored 2026-08-18 after 5th upstream reset.\n    if _try_supervise_fast_path():\n        return\n'''


def _check_syntax(text: str, main_py: str) -> bool:
    """Guard against a corrupted insert: never write a syntactically broken
    main.py. Uses compile() (not py_compile which writes a pyc)."""
    try:
        compile(text, main_py, "exec")
        return True
    except SyntaxError as exc:
        print(f"RESTORE FAIL: wired main.py would not compile: {exc}", file=sys.stderr)
        return False


def wire(main_py: str) -> int:
    try:
        s = open(main_py, encoding="utf-8").read()
    except FileNotFoundError:
        print("RESTORE FAIL: main.py missing", file=sys.stderr)
        return 3

    changed = []
    warn = []

    # ---- 1. cmd_supervise / cmd_diag handlers ---------------------------
    if "def cmd_supervise(args):" not in s:
        insert = (
            "\n\ndef cmd_supervise(args):\n"
            '    """Supervisor: worker lifecycle, inbox, campaigns (protected local)."""\n'
            "    from hermes_cli.subcommands.supervise import run_supervise_command\n"
            "    return run_supervise_command(args)\n"
            "\n\ndef cmd_diag(args):\n"
            '    """Diagnostics: read-only runtime introspection (re-registered after\n'
            "    upstream-main resets dropped the CLI router entry).\"\"\"\n"
            "    from hermes_cli.subcommands.diagnostics import run_diagnostics_command\n"
            "    return run_diagnostics_command(args)\n"
        )
        # Insert after the full cmd_cron function body, before cmd_sync.
        m = re.search(r"    def cmd_cron\(args\):.*?\n\ndef cmd_sync\(args\):", s, re.S)
        if not m:
            # try without leading indent
            m = re.search(r"\ndef cmd_cron\(args\):.*?\n\ndef cmd_sync\(args\):", s, re.S)
        if not m:
            print("RESTORE FAIL: cmd_cron anchor not found", file=sys.stderr)
            return 2
        s = s[:m.end() - len("\ndef cmd_sync(args):")] + insert + s[m.end() - len("\ndef cmd_sync(args):"):]
        changed.append("cmd_supervise/cmd_diag defs")

    # ---- 2. parser registrations -----------------------------------------
    if "build_supervise_parser" not in s:
        block = (
            "\n    # supervise command  (parser built in hermes_cli/subcommands/supervise.py;\n"
            "    # re-registered by restore-supervisor.sh after upstream resets)\n"
            "    from hermes_cli.subcommands.supervise import build_supervise_parser as _build_supervise_parser\n"
            "    _build_supervise_parser(subparsers, cmd_supervise=cmd_supervise)\n"
            "\n"
            "    # diag command  (same upstream-reset drop; needed by the wiring contract)\n"
            "    from hermes_cli.subcommands.diagnostics import build_diagnostics_parser as _build_diagnostics_parser\n"
            "    _build_diagnostics_parser(subparsers, cmd_diagnostics=cmd_diag)\n"
        )
        m = re.search(r"(    build_sync_parser\(subparsers, cmd_sync=cmd_sync\)\n)", s)
        if not m:
            print("RESTORE FAIL: build_sync_parser anchor not found", file=sys.stderr)
            return 2
        s = s[:m.end()] + block + s[m.end():]
        changed.append("supervise/diag parser registration")

    # ---- 3. fast-path _BUILTIN_SUBCOMMANDS entry ---------------------------
    if '"supervise",' not in s:
        m = re.search(r'(        "verify",\n)', s)
        if m:
            s = s[:m.end()] + (
                "        # supervise: pure supervisor/mission CLI - never needs plugin-registered\n"
                "        # subcommands; excluding it here forced eager plugin import\n"
                "        # (~1.2-1.4s per `hermes supervise ...` spawn).\n"
                '        "supervise",\n'
            ) + s[m.end():]
            changed.append("_BUILTIN_SUBCOMMANDS+=supervise")
        else:
            warn.append("_BUILTIN_SUBCOMMANDS verify anchor not found (fast path skipped, parser registration still works)")

    # ---- 4. fast-path function + call --------------------------------------
    if "def _try_supervise_fast_path" not in s:
        # Anchor on the WHOLE def head so the inserted function never splits
        # the `def _(...) -> bool:` declaration.
        m = re.search(
            r"    return None\s+(?=def _plugin_cli_discovery_needed\(\) -> bool:\n)", s
        )
        if m:
            s = s[:m.end()] + _FAST_PATH_DEF + s[m.end():]
            changed.append("_try_supervise_fast_path def")
        else:
            warn.append("_first_positional_argv anchor not found (fast-path def skipped)")

    if "if _try_supervise_fast_path():" not in s:
        anchor = "    if _try_termux_fast_cli_launch():\n        return\n"
        if anchor in s:
            s = s.replace(anchor, anchor + "\n" + _FAST_PATH_CALL.lstrip("\n"), 1)
            changed.append("fast-path main() call")

    # ---- 5. syntax guard: never leave a broken main.py behind. Verify by
    # compiling the NEW content BEFORE touching the file; if it would not
    # compile, leave the pre-wire content on disk and report failure so the
    # caller can restore the whole file from the archive. --------------------
    if not _check_syntax(s, main_py):
        return 2

    open(main_py, "w", encoding="utf-8").write(s)

    for w in warn:
        print("RESTORE WARN:", w, file=sys.stderr)
    if changed:
        print("RESTORE WIRE:", ", ".join(changed))
    else:
        print("RESTORE: main.py already wired")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: wire_main.py <hermes_cli/main.py>", file=sys.stderr)
        sys.exit(2)
    sys.exit(wire(sys.argv[1]))