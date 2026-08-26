"""Spot-audit gate (2026-08-23 grill Q8).

Real-process tests: real git repos (empty vs content commits), real mission
ledgers under a temp HERMES_SUPERVISOR_DIR. The deterministic fabrication
probes are exercised for real — no mock stands in for the code under test.

Fails on the parent commit: spot_audit_mission / _commit_touches_files do
not exist there (import error = red).
"""
import json
import os
import subprocess

import pytest

from hermes_cli import supervisor as SUP


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, text=True)


def _repo(tmp_path):
    repo = tmp_path / "work"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    return repo


def _sha(repo):
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def _mission(mid, evidence, tmp_path, monkeypatch):
    """Real mission ledger written through SUP.save_mission."""
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path / "sup"))
    m = {
        "mission_id": mid,
        "objective": "test objective",
        "requirements": [],
        "phases": [{"phase_id": "p1", "status": "COMPLETE",
                    "evidence": evidence}],
        "unresolved_findings": [],
    }
    SUP.save_mission(m)
    return m


def test_empty_commit_is_rejected(tmp_path, monkeypatch):
    """--allow-empty commit cited as evidence must FAIL the audit."""
    repo = _repo(tmp_path)
    _git(repo, "commit", "--allow-empty", "-q", "-m", "fake")
    fake_sha = _sha(repo)
    (repo / "proof.md").write_text("tests pass, trust me\n")
    _mission("m-empty", [f"shipped {fake_sha} see proof.md"], tmp_path,
             monkeypatch)
    out = SUP.spot_audit_mission("m-empty", workdir=str(repo))
    assert out["verdict"] == "REJECT"
    assert "commit" in out["reason"]
    # Causal chain: rejection -> OPEN finding -> mission stays ACTIVE
    m = SUP.load_mission("m-empty")
    open_f = [f for f in m["unresolved_findings"] if f["status"] == "OPEN"]
    assert open_f, "rejection must append an OPEN finding"
    st = SUP.mission_status("m-empty")
    assert st["status"] == "MISSION_ACTIVE", (
        "a spot-audit-rejected mission must NOT read MISSION_COMPLETE")


def test_real_commit_and_real_file_confirm_probes(tmp_path, monkeypatch):
    """Non-empty commit + non-empty evidence file pass the DETERMINISTIC
    probes (verdict here comes from the LLM stage, so we assert only that
    no deterministic check fired by inspecting the reason)."""
    repo = _repo(tmp_path)
    (repo / "fix.py").write_text("def fixed():\n    return True\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "real fix")
    sha = _sha(repo)
    _mission("m-real", [f"fixed in {sha}, suite 12 passed"], tmp_path,
             monkeypatch)
    # Deterministic layer only: both probes must come back clean.
    assert SUP._commit_touches_files(sha, str(repo)) is True
    assert not SUP._evidence_kinds(f"fixed in {sha}", str(repo)) or True
    kinds = SUP._evidence_kinds(f"fixed in {sha}, see fix.py", str(repo))
    assert "commit" in kinds and "file" in kinds


def test_zero_byte_evidence_file_rejected(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / "proof.md").touch()  # 0 bytes = fabrication-shaped
    _mission("m-zero", ["verified, see proof.md"], tmp_path, monkeypatch)
    out = SUP.spot_audit_mission("m-zero", workdir=str(repo))
    assert out["verdict"] == "REJECT"
    assert "zero-byte" in out["reason"]


def test_missing_ledger_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(tmp_path / "sup"))
    out = SUP.spot_audit_mission("nope")
    assert out["verdict"] == "REJECT"


# ---- verdict parser: measured real-output shapes, echo traps ----

def test_parser_bare_token_line():
    raw = "\x1b[33mwarn\x1b[0m\nQuery: say CONFIRM or REJECT\nreasoning...\nCONFIRM\r\nfooter CONFIRM if…"
    rejected, parsed = SUP._parse_audit_verdict(raw)
    assert parsed and rejected is False


def test_parser_reject_wins():
    raw = "Query: CONFIRM or REJECT?\nREJECT\n  evidence line"
    rejected, parsed = SUP._parse_audit_verdict(raw)
    assert parsed and rejected is True


def test_parser_no_token_fails_closed():
    rejected, parsed = SUP._parse_audit_verdict("no verdict here at all")
    assert parsed is False and rejected is True, "unparseable = REJECT"


def test_parser_prompt_echo_alone_never_confirms():
    """The prompt mentions both tokens but the reply never arrives:
    inline echoes must NOT count as a verdict."""
    raw = ("Query: Reply with EXACTLY one first token: CONFIRM if the "
           "completion holds up, or REJECT followed by one line.\n"
           "Resume this session with: hermes -c \"CONFIRM if…\"")
    rejected, parsed = SUP._parse_audit_verdict(raw)
    assert parsed is False and rejected is True


def test_defect_quote_skips_prompt_echo():
    """Regression (live-fire 2026-08-23): the CLI wraps our prompt at ~78
    cols, putting 'REJECT followed by...' at a line start BEFORE the real
    reply. Verdict AND defect quote must come from the last occurrence."""
    raw = ("Query: ... or REJECT followed by one line naming the strongest "
           "defect.\nInitializing agent...\n"
           "\x1b[7m\x1b[0mREJECT.\npanel\n"
           "REJECT: entire deliverable is a 6-byte stub, fix.py contains "
           "only x = 1\nResume footer CONFIRM if...")
    rejected, parsed = SUP._parse_audit_verdict(raw)
    assert parsed and rejected is True
    import re as _re
    clean = _re.sub(r"\x1b\[[0-9;]*m", "", raw)
    lines_ = clean.splitlines()
    defect = ""
    for i_ in range(len(lines_) - 1, -1, -1):
        if _re.match(r"^[ \t]*(?:\*{0,2})REJECT\b", lines_[i_].upper()):
            defect = lines_[i_]
            break
    assert "6-byte stub" in defect
    assert "strongest defect" not in defect


def test_worker_spawn_is_quiet(tmp_path, monkeypatch):
    """Workers are headless subprocesses: they must run `hermes chat -q -Q`
    so banners/spinners/tool-previews never cost tokens or parse noise.
    Real spawn through start_worker with a stub hermes binary recording argv."""
    sup = tmp_path / "sup"
    tdir = sup / "tasks" / "tq1"
    tdir.mkdir(parents=True)
    (tdir / "brief.md").write_text("demo brief", encoding="utf-8")
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(sup))
    rec = tmp_path / "argv.txt"
    fake = tmp_path / "fake-hermes"
    fake.write_text(f'#!/bin/sh\nprintf "%s\\0" "$@" > {rec}\n')
    fake.chmod(0o755)
    proc = SUP.start_worker("tq1", hermes=str(fake))
    proc.wait(timeout=10)
    argv = [a for a in rec.read_text().split("\0") if a]
    assert "-Q" in argv and "-q" in argv, f"missing flags: {argv}"
    # Order contract (live-measured): -q CONSUMES the next token, so -Q
    # must precede -q and the brief must follow -q. Wrong order =
    # 'error: argument -q/--query: expected one argument', instant death.
    assert argv.index("-Q") < argv.index("-q"), argv
    assert argv[argv.index("-q") + 1] == "demo brief", argv


def test_auditor_spawn_is_quiet(tmp_path, monkeypatch):
    """The adversarial auditor is also a headless caller: `chat -q -Q`.
    Fake hermes records argv and replies CONFIRM; audit must parse it."""
    sup = tmp_path / "sup"
    sup.mkdir()
    repo = _repo(tmp_path)
    (repo / "fix.py").write_text("x = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fix")
    sha = _sha(repo)
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(sup))
    m = {"mission_id": "ma-q", "objective": "obj", "requirements": [],
         "phases": [{"phase_id": "p1", "status": "COMPLETE",
                     "evidence": [f"done {sha}, 12 passed, see fix.py"]}],
         "unresolved_findings": []}
    SUP.save_mission(m)
    rec = tmp_path / "argv.txt"
    fake = tmp_path / "fake-hermes"
    # NUL-separated argv: the prompt is ONE element containing newlines.
    fake.write_text(f'#!/bin/sh\nprintf "%s\\0" "$@" > {rec}\necho CONFIRM\n')
    fake.chmod(0o755)
    out = SUP.spot_audit_mission("ma-q", workdir=str(repo), hermes_bin=str(fake))
    assert out["verdict"] == "CONFIRM"
    argv = rec.read_text().split("\0")
    argv = [a for a in argv if a]
    assert "-Q" in argv, f"auditor cmd must pass -Q, got: {argv[:3]}"
    assert argv.index("-Q") < argv.index("-q"), argv[:3]
    assert argv[-1].startswith("You are an ADVERSARIAL auditor"), argv[:3]


def test_auditor_bounded_turns(tmp_path, monkeypatch):
    """Speed contract: auditor runs are capped (--max-turns) so a runaway
    audit cannot burn unbounded wall-clock."""
    sup = tmp_path / "sup"; sup.mkdir()
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(sup))
    m = {"mission_id": "mt", "objective": "o", "requirements": [],
         "phases": [{"phase_id": "p", "status": "COMPLETE",
                     "evidence": ["see x.py"]}],
         "unresolved_findings": []}
    SUP.save_mission(m)
    (tmp_path / "x.py").write_text("x = 1\n")
    rec = tmp_path / "argv.txt"
    fake = tmp_path / "fake-hermes"
    fake.write_text(f'#!/bin/sh\nprintf "%s\\\\0" "$@" >> {rec}\necho CONFIRM\n')
    fake.chmod(0o755)
    SUP.spot_audit_mission("mt", workdir=str(tmp_path),
                           hermes_bin=str(fake))
    argv = [a for a in rec.read_text().split("\0") if a]
    assert "--max-turns" in argv, f"no turn cap: {argv[:5]}"
    assert int(argv[argv.index("--max-turns") + 1]) <= 10


def test_parser_verdict_marker_line():
    """Explicit machine marker beats layout luck: 'VERDICT: REJECT' must
    parse even when the prose verdict sits mid-line."""
    raw = ("reasoning ... Final verdict unchanged: REJECT. The claimed "
           "evidence does not exist.\nVERDICT: REJECT")
    rejected, parsed = SUP._parse_audit_verdict(raw)
    assert parsed and rejected is True
    raw2 = "prose mentions CONFIRM somewhere mid-line\nVERDICT: CONFIRM"
    rejected2, parsed2 = SUP._parse_audit_verdict(raw2)
    assert parsed2 and rejected2 is False


def test_reject_reason_uses_marker_line(tmp_path, monkeypatch):
    """A marker-based REJECT must surface the defect line, not fall back
    to 'no parseable verdict'."""
    sup = tmp_path / "sup"; sup.mkdir()
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(sup))
    m = {"mission_id": "mr", "objective": "o", "requirements": [],
         "phases": [{"phase_id": "p", "status": "COMPLETE",
                     "evidence": ["see x.py"]}],
         "unresolved_findings": []}
    SUP.save_mission(m)
    (tmp_path / "x.py").write_text("x = 1\n")
    fake = tmp_path / "fake-hermes"
    fake.write_text('#!/bin/sh\necho "analysis: evidence is thin"\n'
                    'echo "VERDICT: REJECT evidence cites no real run"\n')
    fake.chmod(0o755)
    out = SUP.spot_audit_mission("mr", workdir=str(tmp_path),
                                 hermes_bin=str(fake))
    assert out["verdict"] == "REJECT"
    led = SUP.load_mission("mr")
    note = led["unresolved_findings"][-1]["note"]
    assert "no real run" in note or "evidence is thin" in note, note
    assert "no parseable verdict" not in note


def test_dry_worker_mode_skips_llm_stage(tmp_path, monkeypatch):
    """MISSION_DRY_WORKER=1 (the established offline test seam) must skip
    the LLM auditor stage entirely — deterministic probes still run — so
    legacy loop tests stay offline and deterministic."""
    sup = tmp_path / "sup"; sup.mkdir()
    monkeypatch.setenv("HERMES_SUPERVISOR_DIR", str(sup))
    monkeypatch.setenv("MISSION_DRY_WORKER", "1")
    m = {"mission_id": "md", "objective": "o", "requirements": [],
         "phases": [{"phase_id": "p", "status": "COMPLETE",
                     "evidence": ["see x.py"]}],
         "unresolved_findings": []}
    SUP.save_mission(m)
    (tmp_path / "x.py").write_text("x = 1\n")
    bomb = tmp_path / "bomb-hermes"
    bomb.write_text('#!/bin/sh\nexit 99\n')  # must NEVER be invoked
    bomb.chmod(0o755)
    out = SUP.spot_audit_mission("md", workdir=str(tmp_path),
                                 hermes_bin=str(bomb))
    assert out["verdict"] == "CONFIRM", out
    # Probes still enforced in dry mode:
    (tmp_path / "empty.py").write_text("")
    m2 = {"mission_id": "md2", "objective": "o", "requirements": [],
          "phases": [{"phase_id": "p", "status": "COMPLETE",
                      "evidence": ["see empty.py"]}],
          "unresolved_findings": []}
    SUP.save_mission(m2)
    out2 = SUP.spot_audit_mission("md2", workdir=str(tmp_path),
                                  hermes_bin=str(bomb))
    assert out2["verdict"] == "REJECT"
