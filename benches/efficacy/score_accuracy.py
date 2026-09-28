#!/usr/bin/env python3
"""
REF-222 accuracy grader: precision/recall per trial for find_all_usages tasks.

Reads NDJSON trial transcripts, extracts the agent's claimed file:line locations
from the final assistant text response, compares against the oracle ground truth
(via ripgrep), and emits accuracy.csv keyed by (arm, task_id, trial).

Usage:
    python3 benches/efficacy/score_accuracy.py \
        [--results-dir results/] \
        [--tasks tasks/reflex.yaml tasks/ripgrep.yaml tasks/tokio.yaml] \
        [--arms A B] \
        [--out results/accuracy.csv]

The output CSV columns match the H2 schema from analyze.py:
  arm, task_id, trial, n_expected, n_returned, n_correct,
  precision, recall, hallucination_rate, notes
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / "results"
TASKS_DIR = SCRIPT_DIR / "tasks"

# Maps repo id → corpus checkout dir (mirrors runner.py)
REPO_ROOT = SCRIPT_DIR.parent.parent.resolve()
CORPUS_REPOS = {
    "reflex": SCRIPT_DIR / "corpus" / "reflex",  # pinned clone, see repos.md
    "ripgrep": SCRIPT_DIR / "corpus" / "ripgrep",
    "tokio": SCRIPT_DIR / "corpus" / "tokio",
}


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text())
    except ModuleNotFoundError:
        pass
    for cmd in (
        ["yq", "-o=json", ".", str(path)],   # yq 4.x
        ["yq", "r", "-j", str(path)],         # yq 3.x (mikefarah)
        ["yq", ".", str(path)],               # kislyuk/yq (jq wrapper)
    ):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True)
            if out.returncode == 0 and out.stdout.strip():
                return json.loads(out.stdout)
        except FileNotFoundError:
            continue
    sys.exit(f"FATAL: cannot load {path} — install PyYAML or yq on PATH.")


def load_tasks(task_files: list[Path], arms_filter: list[str] | None = None) -> dict[str, dict]:
    """Return {task_id: task_dict} for find_all_usages tasks only."""
    tasks = {}
    for tf in task_files:
        data = _load_yaml(tf)
        for t in (data.get("tasks") or []):
            if t.get("category") == "find_all_usages":
                repo_id = t.get("repo", "reflex")
                t["_repo_dir"] = CORPUS_REPOS.get(repo_id, REPO_ROOT)
                tasks[t["id"]] = t
    return tasks


def run_oracle(task: dict) -> list[str]:
    """Run ripgrep oracle and return sorted list of 'file:line' strings."""
    gt = task.get("ground_truth", {})
    oracle_spec = gt.get("oracle", {})
    pattern = oracle_spec.get("pattern", "")
    path = oracle_spec.get("path", ".")
    flags = oracle_spec.get("flags", [])
    cwd = str(task["_repo_dir"])

    cmd = ["rg", "--no-heading", "--line-number", "--with-filename",
           "--color", "never", *flags, "--", pattern, path]
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except FileNotFoundError:
        sys.exit("FATAL: ripgrep (`rg`) not found on PATH")
    if proc.returncode == 2:
        raise RuntimeError(f"rg error: {proc.stderr.strip()}")

    pairs = set()
    for line in proc.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) >= 2 and parts[1].isdigit():
            pairs.add(f"{parts[0]}:{parts[1]}")
    return sorted(pairs)


def _extract_final_text(transcript_path: Path) -> str:
    """Extract the last assistant text block from an NDJSON transcript."""
    last_text = ""
    try:
        for raw in transcript_path.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if ev.get("type") != "assistant":
                continue
            msg = ev.get("message", {})
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    last_text = block.get("text", "")
    except Exception as exc:
        return ""
    return last_text


# Matches patterns like:
#   src/foo/bar.rs:123
#   ./src/foo/bar.rs:123
#   - src/foo/bar.rs:123
#   `src/foo/bar.rs:123`
#   crates/printer/src/json.rs:45
#   tokio/src/sync/notify.rs:201
# Does NOT match bare integers or version strings.
_FILE_LINE_RE = re.compile(
    r"""
    (?:^|[\s`*\->\(,\[])          # word boundary: space, backtick, bullet, etc.
    (                              # capture group: the file:line pair
        (?:\.{0,2}/)?              # optional leading ./ or ../
        [\w.\-/]+                  # file path characters
        \.(?:rs|py|go|js|ts|java|c|cpp|h|rb|php|kt|zig|vue|svelte|cs|swift)
        :\d+                       # :line_number
    )
    """,
    re.VERBOSE | re.MULTILINE,
)


def extract_claimed_locations(text: str) -> set[str]:
    """Parse file:line pairs from agent's final answer text."""
    raw_hits = _FILE_LINE_RE.findall(text)
    locations = set()
    for hit in raw_hits:
        # Strip leading ./ or /
        hit = hit.lstrip("./")
        # Normalise: keep only the relative path portion
        # (agent may output absolute paths; strip known prefix patterns)
        if ":" in hit:
            parts = hit.rsplit(":", 1)
            path_part = parts[0]
            line_part = parts[1]
            # Remove leading slashes if present
            path_part = path_part.lstrip("/")
            locations.add(f"{path_part}:{line_part}")
    return locations


def _strip_repo_prefix(loc: str, repo_dir: Path) -> str:
    """Remove a leading absolute path prefix so relative comparisons work."""
    repo_str = str(repo_dir).rstrip("/") + "/"
    if loc.startswith(repo_str):
        return loc[len(repo_str):]
    return loc


def score_trial(
    transcript_path: Path,
    oracle_set: set[str],
    task: dict,
) -> dict:
    """Compute precision/recall for a single trial."""
    final_text = _extract_final_text(transcript_path)
    if not final_text:
        return {
            "n_expected": len(oracle_set),
            "n_returned": 0,
            "n_correct": 0,
            "precision": None,
            "recall": 0.0,
            "hallucination_rate": None,
            "notes": "no_final_text",
        }

    claimed = extract_claimed_locations(final_text)

    # Normalise oracle set too (relative paths, stripped of corpus prefix)
    repo_dir = Path(task["_repo_dir"])
    oracle_norm = {_strip_repo_prefix(loc, repo_dir) for loc in oracle_set}
    claimed_norm = {_strip_repo_prefix(loc, repo_dir) for loc in claimed}

    # Suffix-aware matching: agents may omit leading path components (e.g.
    # "query/mod.rs:23" instead of "src/query/mod.rs:23"). A claimed location
    # counts as correct if any oracle location ends with it (or exact match).
    def _matches_oracle(claimed_loc: str, oracle_set: set[str]) -> bool:
        if claimed_loc in oracle_set:
            return True
        # Try suffix match: split off the line number, check path suffix
        for oracle_loc in oracle_set:
            # Both have "path:line" format; compare with trailing suffix
            if oracle_loc.endswith("/" + claimed_loc) or oracle_loc == claimed_loc:
                return True
        return False

    tp = sum(1 for c in claimed_norm if _matches_oracle(c, oracle_norm))
    # For FP: count claims that don't match any oracle location
    fp = sum(1 for c in claimed_norm if not _matches_oracle(c, oracle_norm))
    # For FN: count oracle locations not covered by any claim
    fn_count = sum(
        1 for o in oracle_norm
        if not any(_matches_oracle(c, {o}) for c in claimed_norm)
    )

    n_returned = len(claimed_norm)
    n_expected = len(oracle_norm)
    precision = tp / n_returned if n_returned > 0 else None
    recall = tp / n_expected if n_expected > 0 else 0.0
    hall_rate = fp / n_returned if n_returned > 0 else None
    _ = fn_count  # available for future FN-based metrics

    return {
        "n_expected": n_expected,
        "n_returned": n_returned,
        "n_correct": tp,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4),
        "hallucination_rate": round(hall_rate, 4) if hall_rate is not None else None,
        "notes": "",
    }


CSV_COLUMNS = [
    "arm", "task_id", "trial",
    "n_expected", "n_returned", "n_correct",
    "precision", "recall", "hallucination_rate", "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="REF-222 accuracy grader")
    parser.add_argument(
        "--results-dir", default=str(RESULTS_DIR), type=Path,
        help=f"Directory containing arm/task/trial_NN.ndjson transcripts (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--tasks", nargs="+", type=Path,
        default=[TASKS_DIR / "reflex.yaml", TASKS_DIR / "ripgrep.yaml", TASKS_DIR / "tokio.yaml"],
        help="YAML task corpus files to score",
    )
    parser.add_argument(
        "--arms", nargs="+", default=["A", "B"],
        help="Arms to score (default: A B)",
    )
    parser.add_argument(
        "--out", default=None, type=Path,
        help="Output CSV path (default: {results-dir}/accuracy.csv)",
    )
    args = parser.parse_args()

    out_path = args.out or (Path(args.results_dir) / "accuracy.csv")

    # Load tasks and build oracle sets
    tasks = load_tasks(args.tasks)
    if not tasks:
        sys.exit("ERROR: no find_all_usages tasks found in the specified YAML files")

    print(f"Loaded {len(tasks)} find_all_usages tasks: {list(tasks)}")

    # Pre-compute oracle for each task
    oracles: dict[str, set[str]] = {}
    for task_id, task in tasks.items():
        gt = task.get("ground_truth", {})
        if gt.get("type") not in ("match_set",):
            continue
        try:
            oracle_lines = run_oracle(task)
            oracles[task_id] = set(oracle_lines)
            expected = gt.get("expected_count", "?")
            got = len(oracle_lines)
            status = "✓" if str(got) == str(expected) else f"MISMATCH expected={expected}"
            print(f"  oracle {task_id}: {got} matches {status}")
        except Exception as exc:
            print(f"  WARN: oracle failed for {task_id}: {exc}")

    # Score transcripts
    rows = []
    results_dir = Path(args.results_dir)

    for arm in args.arms:
        for task_id, task in tasks.items():
            if task_id not in oracles:
                continue
            oracle_set = oracles[task_id]
            task_dir = results_dir / arm / task_id
            if not task_dir.exists():
                continue
            for trial_file in sorted(task_dir.glob("trial_*.ndjson")):
                trial_num = int(trial_file.stem.split("_")[1])
                metrics = score_trial(trial_file, oracle_set, task)
                rows.append({
                    "arm": arm,
                    "task_id": task_id,
                    "trial": trial_num,
                    **metrics,
                })
                p = metrics["precision"]
                r = metrics["recall"]
                p_str = f"{p:.2f}" if p is not None else "N/A"
                r_str = f"{r:.2f}" if r is not None else "N/A"
                print(f"  {arm}/{task_id}/trial_{trial_num:02d}: "
                      f"precision={p_str} recall={r_str} "
                      f"({metrics['n_correct']}/{metrics['n_expected']} correct)")

    if not rows:
        print("WARNING: no transcripts found to score")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
