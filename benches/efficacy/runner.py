#!/usr/bin/env python3
"""
Phase 2 A/B runner harness for the Reflex AI coding efficacy study.

Runs arms × tasks × N trials, invokes Claude Code headless per the Phase 0
recipe, and writes full JSON transcripts to benches/efficacy/results/.

Usage:
    python3 benches/efficacy/runner.py [--arms A B C Bprime] [--tasks T01 T02]
                                       [--n 5] [--model claude-sonnet-4-6]
                                       [--skip-build] [--dry-run]
                                       [--repos reflex ripgrep tokio]

Multi-repo support: tasks are loaded from tasks/*.yaml. Each task declares its
target repo (reflex / ripgrep / tokio). Claude runs with cwd set to the
appropriate corpus checkout so Grep/Glob and the rfx MCP server all operate
on the correct codebase. For MCP arms, rfx index is built in the corpus
checkout automatically before that repo's tasks are executed.

The harness is idempotent: existing transcript files are skipped unless
--overwrite is passed.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent.resolve()
RESULTS_DIR = SCRIPT_DIR / "results"
CONFIGS_DIR = SCRIPT_DIR / "configs"
CORPUS_DIR = SCRIPT_DIR / "corpus"

# Maps repo id → the local checkout directory Claude should run in.
# The reflex corpus is a pinned clone (repos.md), never the live checkout: answer keys
# were written for that SHA, and edits to the live tree must not leak into trials.
CORPUS_REPOS: dict[str, Path] = {
    "reflex": CORPUS_DIR / "reflex",
    "ripgrep": CORPUS_DIR / "ripgrep",
    "tokio": CORPUS_DIR / "tokio",
}

# Search-specific tool names for metrics extraction (used here for verification).
SEARCH_TOOLS = frozenset([
    "Grep", "Glob",
    "mcp__reflex__search_code", "mcp__reflex__search_regex",
    "mcp__reflex__search_ast", "mcp__reflex__find_references",
    "mcp__reflex__gather_context", "mcp__reflex__list_locations",
    "mcp__reflex__count_occurrences",
])

# All 17 Reflex MCP tool names — used for explicit --allowedTools preload.
# Listing these in --allowedTools forces the Claude Code SDK to eagerly load
# their schemas at session start, eliminating the "deferred schema" ToolSearch
# calls that otherwise add 1-2 wasted turns to every arm B/C trial.
REFLEX_MCP_TOOLS = [
    "mcp__reflex__analyze_summary",
    "mcp__reflex__check_index_status",
    "mcp__reflex__count_occurrences",
    "mcp__reflex__find_circular",
    "mcp__reflex__find_hotspots",
    "mcp__reflex__find_islands",
    "mcp__reflex__find_references",
    "mcp__reflex__find_unused",
    "mcp__reflex__gather_context",
    "mcp__reflex__get_dependencies",
    "mcp__reflex__get_dependents",
    "mcp__reflex__get_transitive_deps",
    "mcp__reflex__index_project",
    "mcp__reflex__list_locations",
    "mcp__reflex__search_ast",
    "mcp__reflex__search_code",
    "mcp__reflex__search_regex",
]

# Built-in tools allowed in MCP arms (B, C, Bprime).
BUILTIN_TOOLS_MCP_ARMS = [
    "Bash", "Edit", "Glob", "Grep", "LS", "MultiEdit",
    "Read", "TodoWrite", "Write",
]

# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------
# extra_flags: list of additional CLI flags passed to `claude`.
# mcp_command: path to rfx binary (set at runtime; None = no MCP).
# disallowed_tools: list of tool names to block via --disallowedTools.
# allowed_tools: when non-empty, passed as --allowedTools (forces eager MCP
#   schema loading instead of the lazy-deferral that requires ToolSearch).
ARMS = {
    "A": {
        "description": "Control: built-ins only (Grep/Glob/Read/Bash); Reflex MCP disabled",
        "mcp_command": None,
        # --strict-mcp-config prevents project .mcp.json from loading Reflex MCP
        # --dangerously-skip-permissions keeps tool approval consistent across all arms
        "extra_flags": ["--strict-mcp-config", "--dangerously-skip-permissions"],
        "disallowed_tools": [],
        "allowed_tools": [],
        "append_system_prompt": None,
    },
    "B": {
        "description": "Realistic: Reflex MCP enabled + built-ins available; Reflex-first nudge via MCP initialize instructions field",
        "mcp_command": "TARGET_RELEASE_RFX",
        # --strict-mcp-config prevents stray project MCP servers (pw, Roam, etc.)
        # from loading alongside Reflex, which contaminated some Phase 4 arm C trials.
        # --allowedTools with explicit Reflex tool names forces eager schema loading,
        # eliminating the ToolSearch deferred-schema calls (~1/trial in Phase 4).
        "extra_flags": ["--strict-mcp-config", "--dangerously-skip-permissions"],
        "disallowed_tools": [],
        "allowed_tools": BUILTIN_TOOLS_MCP_ARMS + REFLEX_MCP_TOOLS,
        "append_system_prompt": None,
    },
    "C": {
        "description": "Reflex-forced: Reflex MCP enabled; Grep/Glob disallowed; Read/Bash allowed",
        "mcp_command": "TARGET_RELEASE_RFX",
        "extra_flags": ["--strict-mcp-config", "--dangerously-skip-permissions"],
        "disallowed_tools": ["Grep", "Glob"],
        "allowed_tools": BUILTIN_TOOLS_MCP_ARMS + REFLEX_MCP_TOOLS,
        "append_system_prompt": None,
    },
    "Bprime": {
        "description": "No-nudge (secondary): Reflex MCP enabled; MCP initialize nudge suppressed via append_system_prompt override",
        "mcp_command": "TARGET_RELEASE_RFX",
        "extra_flags": ["--strict-mcp-config", "--dangerously-skip-permissions"],
        "disallowed_tools": [],
        "allowed_tools": BUILTIN_TOOLS_MCP_ARMS + REFLEX_MCP_TOOLS,
        "append_system_prompt": (
            "For this session, treat all available search tools — both built-in tools "
            "(Grep, Glob, Read, Bash) and any MCP tools — as equally preferred options. "
            "Choose whichever tool you think is best for each specific task. Do not give "
            "systematic preference to any particular category of tool."
        ),
    },
}


# ---------------------------------------------------------------------------
# YAML / task loading (multi-repo)
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    """Load a YAML file to a dict. Uses PyYAML if available, else yq CLI."""
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text())
    except ModuleNotFoundError:
        pass
    # yq 3.x: `yq r -j file` outputs JSON
    for cmd in (
        ["yq", "-o=json", ".", str(path)],   # yq 4.x
        ["yq", "r", "-j", str(path)],         # yq 3.x
        ["yq", ".", str(path)],               # fallback (may emit YAML)
    ):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True)
            if out.returncode == 0 and out.stdout.strip():
                try:
                    return json.loads(out.stdout)
                except json.JSONDecodeError:
                    continue
        except FileNotFoundError:
            continue
    sys.exit(f"FATAL: cannot load {path} — install PyYAML or yq on PATH.")


def load_tasks(
    task_filter: list[str] | None,
    repo_filter: list[str] | None,
) -> list[dict]:
    """Load tasks from tasks/*.yaml. Each returned task dict gains a '_repo_dir' key."""
    tasks_dir = SCRIPT_DIR / "tasks"
    yaml_files = sorted(tasks_dir.glob("*.yaml"))
    if not yaml_files:
        # Fallback: legacy tasks.json in the same directory
        fallback = SCRIPT_DIR / "tasks.json"
        if fallback.exists():
            data = json.loads(fallback.read_text())
            tasks = data.get("tasks", [])
            for t in tasks:
                t.setdefault("repo", "reflex")
                t["_repo_dir"] = CORPUS_REPOS["reflex"]
            if task_filter:
                tasks = [t for t in tasks if t["id"] in task_filter]
            return tasks
        sys.exit("ERROR: no tasks/*.yaml files and no tasks.json fallback found.")

    all_tasks: list[dict] = []
    for yf in yaml_files:
        repo_id = yf.stem
        if repo_filter and repo_id not in repo_filter:
            continue
        data = _load_yaml(yf)
        for task in (data.get("tasks") or []):
            task_repo = task.get("repo", repo_id)
            task["_repo_dir"] = CORPUS_REPOS.get(task_repo, REPO_ROOT)
            all_tasks.append(task)

    if task_filter:
        all_tasks = [t for t in all_tasks if t["id"] in task_filter]
        if not all_tasks:
            sys.exit(f"ERROR: no tasks matched filter {task_filter}")

    return all_tasks


def ensure_rfx_indexed(repo_dir: Path, rfx_binary: Path, dry_run: bool) -> None:
    """Build the rfx trigram index in repo_dir if it is absent."""
    index_marker = repo_dir / ".reflex" / "meta.db"
    if index_marker.exists():
        print(f"  [INDEX] .reflex/ already present in {repo_dir.name} — skipping build")
        return
    if dry_run:
        print(f"  [DRY-RUN] Would run: rfx index in {repo_dir}")
        return
    print(f"  [INDEX] Building rfx index in {repo_dir.name}...")
    result = subprocess.run(
        [str(rfx_binary), "index"],
        cwd=repo_dir,
        capture_output=False,
    )
    if result.returncode != 0:
        sys.exit(f"ERROR: rfx index failed in {repo_dir}")
    print(f"  [INDEX] Done indexing {repo_dir.name}")


def get_rfx_binary() -> Path:
    # Check CARGO_TARGET_DIR first (NixOS / custom build environments override this)
    cargo_target_dir = os.environ.get("CARGO_TARGET_DIR")
    candidates = []
    if cargo_target_dir:
        candidates.append(Path(cargo_target_dir) / "release" / "rfx")
    candidates.append(REPO_ROOT / "target" / "release" / "rfx")
    for binary in candidates:
        if binary.exists():
            return binary
    sys.exit(
        f"ERROR: rfx binary not found (checked: {[str(c) for c in candidates]}).\n"
        "Run `cargo build --release` first, or pass --skip-build."
    )


def build_rfx(verbose: bool = False) -> None:
    print("Building rfx (cargo build --release)...")
    cmd = ["cargo", "build", "--release"]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=not verbose)
    if result.returncode != 0:
        stderr = result.stderr.decode() if result.stderr else ""
        sys.exit(f"ERROR: cargo build --release failed:\n{stderr}")
    print("Build complete.")


def get_repo_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def make_mcp_config(
    rfx_binary: Path | None,
    tmp_dir: Path,
    arm_name: str,
    mcp_env: dict[str, str] | None = None,
) -> Path:
    """Write a temporary MCP config JSON and return its path.

    ``mcp_env`` (when set) is injected as the Reflex MCP server's ``env`` block so
    per-arm environment toggles reach the ``rfx mcp`` subprocess. REF-204 uses
    this to run arm B_nosc with ``REFLEX_MCP_STRUCTURED_CONTENT=0`` (structuredContent
    suppressed) against B_sc (default) from a single build.
    """
    config_path = tmp_dir / f"mcp-{arm_name}.json"
    if rfx_binary is None:
        config = {"mcpServers": {}}
    else:
        server: dict = {
            "command": str(rfx_binary),
            "args": ["mcp"],
        }
        if mcp_env:
            server["env"] = dict(mcp_env)
        config = {"mcpServers": {"reflex": server}}
    config_path.write_text(json.dumps(config))
    return config_path


# ---------------------------------------------------------------------------
# REF-212: MCP flag pre-flight probe
#
# The B_sc2 regression (REF-212) was a *stale binary*: REFLEX_MCP_SC_STAGE2=1 was
# correctly forwarded to `rfx mcp`, but the running binary predated the Stage-2
# code, so it silently produced Stage-1 output — invalidating a whole trial arm
# with no visible error. The rfx MCP server now prints a `reflex-mcp startup:`
# diagnostic to stderr listing the flags it actually resolved. We spawn the
# binary once per arm and assert the arm's env toggles took effect *before*
# spending any trials, turning a silent post-hoc discovery into a fast, loud fail.
# ---------------------------------------------------------------------------

_FALSEY = {"0", "false", "off", "no"}
_TRUTHY = {"1", "true", "on", "yes"}


def _expected_flag_value(var: str, val: str) -> tuple[str, str] | None:
    """Map a known MCP env toggle to (diagnostic_flag_name, expected 'on'/'off'),
    mirroring the Rust resolution logic in src/mcp.rs. None for unknown vars."""
    v = val.strip().lower()
    if var == "REFLEX_MCP_STRUCTURED_CONTENT":
        return ("structuredContent", "off" if v in _FALSEY else "on")
    if var == "REFLEX_MCP_COLUMNAR":
        return ("columnar", "off" if v in _FALSEY else "on")
    if var == "REFLEX_MCP_SC_STAGE2":
        return ("sc_stage2", "on" if v in _TRUTHY else "off")
    return None


def probe_mcp_flags(
    rfx_binary: Path,
    mcp_env: dict[str, str] | None,
) -> dict[str, str] | None:
    """Spawn `rfx mcp` with `mcp_env` and parse the REF-212 startup diagnostic it
    prints to stderr. Returns a {flag: value} dict, or None if no diagnostic line
    was found (e.g. a binary built before REF-212, or a startup failure).

    The server prints the diagnostic before entering its read loop, so closing
    stdin (EOF) makes it emit the line and exit immediately — no long-lived
    process to manage or kill.
    """
    env = {**os.environ, **(mcp_env or {})}
    try:
        proc = subprocess.run(
            [str(rfx_binary), "mcp"],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"  [PROBE] WARN: could not probe rfx mcp flags: {exc}")
        return None
    for line in proc.stderr.splitlines():
        if line.startswith("reflex-mcp startup:"):
            flags: dict[str, str] = {}
            for tok in line[len("reflex-mcp startup:"):].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    flags[k] = v
            return flags
    return None


def verify_mcp_arm_flags(
    arm_name: str,
    rfx_binary: Path | None,
    mcp_env: dict[str, str] | None,
) -> None:
    """Pre-flight check (REF-212): confirm the rfx binary honours this arm's env
    toggles before running any trial. Aborts on a mismatch or a missing
    diagnostic, because either means the arm's results would be silently invalid.
    """
    if rfx_binary is None:
        return  # non-MCP arm (e.g. arm A) — nothing to verify
    flags = probe_mcp_flags(rfx_binary, mcp_env)
    if flags is None:
        sys.exit(
            f"FATAL: arm {arm_name}: `rfx mcp` emitted no startup diagnostic. "
            "The binary predates REF-212 or failed to start — rebuild with "
            "`cargo build --release` before running MCP arms."
        )
    print(f"  [PROBE] {arm_name} resolved MCP flags: build={flags.get('build','?')} "
          f"structuredContent={flags.get('structuredContent','?')} "
          f"sc_stage2={flags.get('sc_stage2','?')} columnar={flags.get('columnar','?')}")
    for var, val in (mcp_env or {}).items():
        expected = _expected_flag_value(var, val)
        if expected is None:
            continue
        flag_name, want = expected
        got = flags.get(flag_name)
        if got != want:
            sys.exit(
                f"FATAL: arm {arm_name}: env {var}={val} should yield "
                f"{flag_name}={want}, but rfx reported {flag_name}={got}. "
                "This is a STALE BINARY silently ignoring the toggle — rebuild "
                "rfx (`cargo build --release`). See REF-212."
            )


def transcript_path(arm: str, task_id: str, trial: int) -> Path:
    return RESULTS_DIR / arm / task_id / f"trial_{trial:02d}.ndjson"


def transcript_is_complete(path: Path) -> bool:
    """Return True if the transcript exists and contains a terminal result event."""
    if not path.exists():
        return False
    try:
        for line in reversed(path.read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if ev.get("type") == "result":
                return True
        return False
    except Exception:
        return False


def build_claude_cmd(
    arm_name: str,
    arm_cfg: dict,
    task_prompt: str,
    mcp_config_path: Path,
    model: str,
) -> list[str]:
    cmd = [
        "claude",
        "--print",
        "--output-format", "stream-json",
        "--mcp-config", str(mcp_config_path),
        "--model", model,
        # Load only the corpus repo's own settings: the operator's user-level hooks,
        # plugins and output style must not leak into trials.
        "--setting-sources", "project",
    ]

    # Arm-specific flags
    cmd.extend(arm_cfg["extra_flags"])

    # Explicit tool allowlist — forces eager MCP schema loading, eliminating
    # the ToolSearch deferred-schema wasted turns seen in Phase 4 arms B/C.
    # Arm A omits this (uses --dangerously-skip-permissions instead) so it
    # remains a clean control with no explicit tool enumeration.
    if arm_cfg.get("allowed_tools"):
        cmd.extend(["--allowedTools"] + arm_cfg["allowed_tools"])

    # Disallowed tools
    if arm_cfg["disallowed_tools"]:
        cmd.extend(["--disallowedTools", ",".join(arm_cfg["disallowed_tools"])])

    # System prompt appendage (for Bprime nudge suppression)
    if arm_cfg["append_system_prompt"]:
        cmd.extend(["--append-system-prompt", arm_cfg["append_system_prompt"]])

    # Task prompt (positional argument) — use "--" to stop option parsing,
    # preventing --disallowedTools from greedily consuming the prompt.
    cmd.extend(["--", task_prompt])

    return cmd


def run_trial(
    arm_name: str,
    arm_cfg: dict,
    task: dict,
    trial: int,
    model: str,
    mcp_config_path: Path,
    dry_run: bool,
    overwrite: bool,
    repo_sha: str,
) -> dict:
    """
    Run a single trial. Returns a summary dict with status and key metrics.
    Claude runs with cwd set to the task's target corpus repo directory so that
    Grep/Glob and the rfx MCP server all operate on the correct codebase.
    """
    out_path = transcript_path(arm_name, task["id"], trial)

    if not overwrite and transcript_is_complete(out_path):
        print(f"  [SKIP] {arm_name}/{task['id']}/trial_{trial:02d} — already complete")
        return {"status": "skipped", "path": str(out_path)}

    cmd = build_claude_cmd(arm_name, arm_cfg, task["prompt"], mcp_config_path, model)

    # Use the per-task repo directory so built-in tools and the MCP server both
    # see the correct codebase. Falls back to REPO_ROOT for legacy tasks without
    # the _repo_dir key.
    task_cwd = task.get("_repo_dir", REPO_ROOT)

    if dry_run:
        print(f"  [DRY-RUN] Would run in {Path(task_cwd).name}: {' '.join(cmd[:6])} ...")
        return {"status": "dry_run", "cmd": cmd, "cwd": str(task_cwd)}

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"  [RUN] {arm_name}/{task['id']}/trial_{trial:02d}  "
        f"(arm: {arm_cfg['description'][:50]}, cwd: {Path(task_cwd).name})"
    )

    metadata_line = json.dumps({
        "type": "harness_metadata",
        "arm": arm_name,
        "task_id": task["id"],
        "task_repo": task.get("repo", "reflex"),
        "trial": trial,
        "model": model,
        "repo_sha": repo_sha,
        "task_cwd": str(task_cwd),
        "started_at": datetime.now(timezone.utc).isoformat(),
    })

    # REF-213: propagate the arm's MCP env toggles through the *parent* `claude`
    # process, not only the MCP config JSON's `env` block. The config `env` block
    # (see make_mcp_config) targets the claude→rfx hop, but that layer was not
    # honoured at trial time (REF-211 QA: B_sc2 behaved as Stage 1 despite
    # REFLEX_MCP_SC_STAGE2=1), so B_sc2/B_nosc toggles silently no-op'd. Setting
    # the vars here — on the runner→claude hop — is reliable because stdio MCP
    # servers inherit their launcher's environment, so the toggle reaches the
    # spawned `rfx mcp` even if Claude Code drops the per-server `env` override.
    # A fresh copy per trial keeps arms isolated (no cross-arm leakage).
    trial_env = os.environ.copy()
    mcp_env = arm_cfg.get("mcp_env")
    if mcp_env:
        trial_env.update(mcp_env)

    start_ts = time.monotonic()
    with open(out_path, "w") as f:
        # Write harness metadata as the first NDJSON line
        f.write(metadata_line + "\n")
        f.flush()

        proc = subprocess.run(
            cmd,
            cwd=task_cwd,
            stdout=f,
            stderr=subprocess.PIPE,
            text=True,
            env=trial_env,
        )

    elapsed = time.monotonic() - start_ts

    if proc.returncode != 0:
        print(
            f"  [WARN] claude exited {proc.returncode} for "
            f"{arm_name}/{task['id']}/trial_{trial:02d}"
        )
        if proc.stderr:
            print(f"    stderr: {proc.stderr[:200]}")

    # Quick sanity: does transcript have a result event?
    complete = transcript_is_complete(out_path)
    status = "ok" if complete else "incomplete"
    print(f"  [{status.upper()}] elapsed={elapsed:.1f}s  path={out_path}")

    return {"status": status, "path": str(out_path), "elapsed_s": elapsed}


def run_baseline(
    arm_name: str,
    arm_cfg: dict,
    model: str,
    mcp_config_path: Path,
    dry_run: bool,
    overwrite: bool,
    repo_sha: str,
) -> dict:
    """
    Capture the MCP context-tax baseline: an empty/near-empty task to measure
    the token overhead of loading MCP vs not.
    """
    baseline_task = {
        "id": "_baseline",
        "prompt": "Reply with the single word: ready",
        "_repo_dir": CORPUS_REPOS["reflex"],
    }
    return run_trial(
        arm_name=arm_name,
        arm_cfg=arm_cfg,
        task=baseline_task,
        trial=1,
        model=model,
        mcp_config_path=mcp_config_path,
        dry_run=dry_run,
        overwrite=overwrite,
        repo_sha=repo_sha,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reflex efficacy A/B runner harness (Phase 2)"
    )
    ACTIVE_ARMS = list(ARMS.keys())
    parser.add_argument(
        "--arms",
        nargs="+",
        default=ACTIVE_ARMS,
        choices=list(ARMS.keys()),
        metavar="ARM",
        help=f"Arms to run (default: active arms). Choices: {list(ARMS.keys())}",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        metavar="TASK_ID",
        help="Task IDs to run (default: all tasks in tasks/*.yaml). Filters by exact ID.",
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        default=None,
        choices=list(CORPUS_REPOS.keys()),
        metavar="REPO",
        help=f"Repo corpus files to include (default: all). Choices: {list(CORPUS_REPOS.keys())}",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of replicate trials per arm × task (default: 5)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-5",
        help="Model ID to use (default: claude-sonnet-5)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip `cargo build --release` prerequisite step",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip rfx index build for corpus repos (assume .reflex/ already present)",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Run only the per-arm MCP context-tax baseline, not the full task matrix",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run and overwrite existing complete transcripts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    args = parser.parse_args()

    print(f"Reflex Efficacy Runner — Phase 4 (multi-repo)")
    print(f"  Arms:  {args.arms}")
    print(f"  N:     {args.n} trials per arm × task")
    print(f"  Model: {args.model}")
    print(f"  Repo root: {REPO_ROOT}")

    # --- Prerequisites ---
    if not args.skip_build and not args.dry_run:
        build_rfx()

    rfx_binary = None if args.dry_run else get_rfx_binary()
    repo_sha = get_repo_sha()
    print(f"  Reflex SHA: {repo_sha}")

    # --- Load tasks from YAML corpus ---
    all_tasks = load_tasks(task_filter=args.tasks, repo_filter=args.repos)
    print(f"  Tasks ({len(all_tasks)}): {[t['id'] for t in all_tasks]}")
    print()

    # --- Pre-index corpus repos for MCP arms ---
    needs_mcp = any(ARMS[a]["mcp_command"] is not None for a in args.arms)
    if needs_mcp and not args.skip_index and not args.dry_run:
        print("Pre-indexing corpus repos for MCP arms...")
        indexed_repos: set[str] = set()
        for task in all_tasks:
            repo_id = task.get("repo", "reflex")
            if repo_id not in indexed_repos:
                repo_dir = Path(task["_repo_dir"])
                ensure_rfx_indexed(repo_dir, rfx_binary, dry_run=False)
                indexed_repos.add(repo_id)
        print()

    # --- Run matrix ---
    with tempfile.TemporaryDirectory(prefix="rfx-efficacy-") as tmp_dir:
        tmp = Path(tmp_dir)
        summary = []

        for arm_name in args.arms:
            arm_cfg = ARMS[arm_name]
            rfx_bin = rfx_binary if arm_cfg["mcp_command"] == "TARGET_RELEASE_RFX" else None
            mcp_cfg_path = make_mcp_config(
                rfx_bin, tmp, arm_name, mcp_env=arm_cfg.get("mcp_env")
            )

            print(f"=== ARM {arm_name}: {arm_cfg['description']} ===")

            # REF-212: fail fast if the binary does not honour this arm's env
            # toggles (stale-binary guard). Skipped in dry-run (no binary spawn).
            if not args.dry_run:
                verify_mcp_arm_flags(arm_name, rfx_bin, arm_cfg.get("mcp_env"))

            # Per-arm MCP context-tax baseline (runs in the pinned reflex corpus)
            result = run_baseline(
                arm_name=arm_name,
                arm_cfg=arm_cfg,
                model=args.model,
                mcp_config_path=mcp_cfg_path,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                repo_sha=repo_sha,
            )
            summary.append({"arm": arm_name, "task": "_baseline", "trial": 1, **result})

            if args.baselines_only:
                continue

            for task in all_tasks:
                task_repo = task.get("repo", "reflex")
                print(f"  Task: {task['id']} ({task_repo}) — {task.get('category', '?')}")
                for trial in range(1, args.n + 1):
                    result = run_trial(
                        arm_name=arm_name,
                        arm_cfg=arm_cfg,
                        task=task,
                        trial=trial,
                        model=args.model,
                        mcp_config_path=mcp_cfg_path,
                        dry_run=args.dry_run,
                        overwrite=args.overwrite,
                        repo_sha=repo_sha,
                    )
                    summary.append(
                        {"arm": arm_name, "task": task["id"], "trial": trial, **result}
                    )

            print()

    # --- Print run summary ---
    ok = sum(1 for r in summary if r.get("status") == "ok")
    skipped = sum(1 for r in summary if r.get("status") == "skipped")
    failed = sum(1 for r in summary if r.get("status") == "incomplete")

    print(f"Run complete: {ok} ran, {skipped} skipped, {failed} incomplete")
    if failed:
        print("WARN: some trials incomplete — check stderr above and re-run without --skip-build")

    # Write run manifest
    if not args.dry_run:
        manifest_path = RESULTS_DIR / f"run-manifest-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "arms": args.arms,
                    "n": args.n,
                    "model": args.model,
                    "reflex_repo_sha": repo_sha,
                    "corpus_repos": {k: str(v) for k, v in CORPUS_REPOS.items()},
                    "ran_at": datetime.now(timezone.utc).isoformat(),
                    "trials": summary,
                },
                indent=2,
            )
        )
        print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
