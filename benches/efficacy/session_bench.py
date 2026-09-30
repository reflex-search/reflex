#!/usr/bin/env python3
"""Long-session benchmark: many questions (and edits) in ONE agent session.

The REF-222 tasks are one search each (2-3 turns), so they measure the fixed cost of
bringing Reflex into a session. A coding session asks dozens of questions; this
harness measures that. Each session is a list of user messages sent one after
another into one `claude --print --input-format stream-json` process (one context,
one MCP server), and every answer is graded against ripgrep on the tree as it is
at the end of the trial.

  session_bench.py generate                 write tasks/sessions.json from the pinned corpora
  session_bench.py run --arms A B Beager --n 4 --model claude-sonnet-5 [--jobs 24]
  session_bench.py score [--results DIR]    per-arm totals and ratios vs arm A

Session kinds (one of each per corpus):
  investigate  12 questions: find every occurrence (8), where is it defined (2),
               how many files mention it (2)
  edit         two renames across the corpus's source tree, each followed by
               lookups of the new name, the old name and unrelated identifiers;
               graded on the edited tree (auto-update must show the edits)

Every trial runs in its own copy of the corpus (a fresh git repository, so
.gitignore applies); Reflex arms index it before the trial, outside the token
count. Arms and the claude command line come from runner.py (A = Grep/Glob,
B = Reflex deferred, Beager = Reflex with alwaysLoad).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import runner

HERE = Path(__file__).resolve().parent
SESSIONS_JSON = HERE / "tasks" / "sessions.json"
WORK = Path(os.environ.get("SESSION_BENCH_WORK", "/scratch/cache/session-bench"))
SCOPES = {"reflex": "src", "ripgrep": "crates", "tokio": "tokio/src"}
# Where `run` writes transcripts; set SESSION_BENCH_RESULTS to run two waves at once.
RESULTS = Path(os.environ.get("SESSION_BENCH_RESULTS", runner.RESULTS_DIR / "sessions"))

FORMAT_LIST = (
    'End your answer with a JSON code block {"answer": ["path:line", ...]} listing every '
    "match as a path relative to the repository root and a 1-based line number "
    '(an empty list if there is none).'
)
FORMAT_ONE = 'End your answer with a JSON code block {"answer": ["path:line"]}.'
FORMAT_NUMBER = 'End your answer with a JSON code block {"answer": <number>}.'
FORMAT_DONE = 'End your answer with a JSON code block {"answer": "done"}.'


def rg(repo_dir: Path, *args: str) -> list[str]:
    out = subprocess.run(
        ["rg", "--no-heading", "-n", *args], cwd=repo_dir, capture_output=True, text=True
    ).stdout
    return sorted(l.split(":", 2)[0] + ":" + l.split(":", 2)[1] for l in out.splitlines())


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def candidates(repo: str) -> list[tuple[str, int, int, str]]:
    """Public items defined once, with 5-30 whole-word occurrences in >= 2 files,
    in a stable (hash) order: (name, matches, files, definition path:line)."""
    repo_dir = runner.CORPUS_REPOS[repo]
    scope = SCOPES[repo]
    pat = r"^\s*pub(\(crate\))? (struct|enum|trait|fn|type) ([A-Za-z_][A-Za-z0-9_]{5,})"
    defs = rg(repo_dir, pat, scope)
    raw = subprocess.run(
        ["rg", "--no-heading", "-n", "-o", pat, scope], cwd=repo_dir, capture_output=True, text=True
    ).stdout.splitlines()
    names: dict[str, list[str]] = collections.defaultdict(list)
    for line in raw:
        m = re.search(r"pub(?:\(crate\))? (?:struct|enum|trait|fn|type) ([A-Za-z_][A-Za-z0-9_]+)", line)
        if m:
            f, ln = line.split(":", 2)[:2]
            names[m.group(1)].append(f"{f}:{ln}")
    del defs
    out = []
    for name, where in names.items():
        if len(where) != 1:
            continue
        hits = rg(repo_dir, "-w", name, scope)
        files = {h.split(":")[0] for h in hits}
        if 5 <= len(hits) <= 30 and len(files) >= 2:
            out.append((hashlib.sha1(name.encode()).hexdigest(), name, len(hits), len(files), where[0]))
    out.sort()
    return [o[1:] for o in out]


def generate() -> None:
    sessions = []
    for repo, scope in SCOPES.items():
        c = candidates(repo)
        # Investigation: 8 find-all, 2 locate, 2 count-files.
        steps = []
        for name, *_ in c[:8]:
            steps.append({
                "kind": "list",
                "prompt": f"List every place the identifier `{name}` occurs under `{scope}/` "
                          f"(whole identifier). {FORMAT_LIST}",
                "oracle": ["-w", name, scope],
            })
        for name, _, _, where in c[8:10]:
            steps.append({
                "kind": "locate",
                "prompt": f"Where is `{name}` defined? Give the file and line of its definition. {FORMAT_ONE}",
                "expected": [where],
            })
        for name, _, files, _ in c[10:12]:
            steps.append({
                "kind": "count_files",
                "prompt": f"How many files under `{scope}/` mention `{name}` as a whole identifier? {FORMAT_NUMBER}",
                "oracle": ["-w", "-l", name, scope],
            })
        sessions.append({"id": f"{repo}-session-investigate", "repo": repo, "kind": "investigate", "steps": steps})

        # Edit: two renames with lookups before and after.
        long_names = [x for x in c[12:] if len(x[0]) >= 8]
        (n1, *_), (n2, *_) = long_names[0], long_names[1]
        others = [x[0] for x in c[12:] if x[0] not in (n1, n2)][:3]
        r1, r2 = f"{n1}_renamed", f"{n2}_renamed"

        def rename(old, new):
            return {
                "kind": "rename",
                "prompt": f"Rename the identifier `{old}` to `{new}` everywhere under `{scope}/`: the "
                          f"definition, every use, and every mention in comments, docs and strings. "
                          f"Do not build the project or run tests. {FORMAT_DONE}",
                "old": old, "new": new,
            }

        def lookup(name, before_rename=False):
            step = {
                "kind": "list",
                "prompt": f"List every place the identifier `{name}` occurs under `{scope}/` "
                          f"(whole identifier). {FORMAT_LIST}",
                "oracle": ["-w", name, scope],
            }
            if before_rename:
                # Asked before this name is renamed: graded against the start tree.
                step["expected"] = rg(runner.CORPUS_REPOS[repo], "-w", name, scope)
            return step

        steps = [
            lookup(n1, before_rename=True),
            rename(n1, r1),
            lookup(r1),
            lookup(others[0]),
            lookup(n1),
            lookup(n2, before_rename=True),
            rename(n2, r2),
            lookup(r2),
            lookup(others[1]),
            {
                "kind": "files",
                "prompt": f"List every file under `{scope}/` that now contains `{r1}` or `{r2}`. "
                          'End your answer with a JSON code block {"answer": ["path", ...]}.',
                "oracle": ["-w", "-l", "-e", r1, "-e", r2, scope],
            },
            lookup(others[2]),
            lookup(r1),
        ]
        sessions.append({"id": f"{repo}-session-edit", "repo": repo, "kind": "edit", "steps": steps})

    sessions += [long_session(repo, scope, candidates(repo)) for repo, scope in SCOPES.items()]
    SESSIONS_JSON.write_text(json.dumps(
        {"generated_by": "session_bench.py generate (pinned corpora); do not edit by hand",
         "sessions": sessions}, indent=1) + "\n")
    print(f"wrote {SESSIONS_JSON}: {len(sessions)} sessions")


def long_session(repo: str, scope: str, c: list) -> dict:
    """50 questions in one session: 34 lookups, 4 definitions, 3 file counts, and two
    renames, each with a lookup before and after it and of the old name."""
    names = [x[0] for x in c]
    used: set[str] = set()

    def take(pred=lambda n: True):
        for n in names:
            if n not in used and pred(n):
                used.add(n)
                return n
        raise SystemExit(f"{repo}: not enough identifiers for a long session")

    def lookup(name, before=False):
        step = {"kind": "list",
                "prompt": f"List every place the identifier `{name}` occurs under `{scope}/` "
                          f"(whole identifier). {FORMAT_LIST}",
                "oracle": ["-w", name, scope]}
        if before:
            step["expected"] = rg(runner.CORPUS_REPOS[repo], "-w", name, scope)
        return step

    where = {x[0]: x[3] for x in c}
    n1, n2 = take(lambda n: len(n) >= 8), take(lambda n: len(n) >= 8)
    r1, r2 = f"{n1}_renamed", f"{n2}_renamed"

    def rename_block(old, new):
        return [
            lookup(old, before=True),
            {"kind": "rename",
             "prompt": f"Rename the identifier `{old}` to `{new}` everywhere under `{scope}/`: the "
                       f"definition, every use, and every mention in comments, docs and strings. "
                       f"Do not build the project or run tests. {FORMAT_DONE}",
             "old": old, "new": new},
            lookup(new),
            lookup(old),
        ]

    plain = [lookup(take()) for _ in range(34)]
    locates = []
    for _ in range(4):
        n = take()
        locates.append({"kind": "locate",
                        "prompt": f"Where is `{n}` defined? Give the file and line of its definition. {FORMAT_ONE}",
                        "expected": [where[n]]})
    counts = []
    for _ in range(3):
        n = take()
        counts.append({"kind": "count_files",
                       "prompt": f"How many files under `{scope}/` mention `{n}` as a whole identifier? {FORMAT_NUMBER}",
                       "oracle": ["-w", "-l", n, scope]})
    files = {"kind": "files",
             "prompt": f"List every file under `{scope}/` that now contains `{r1}` or `{r2}`. "
                       'End your answer with a JSON code block {"answer": ["path", ...]}.',
             "oracle": ["-w", "-l", "-e", r1, "-e", r2, scope]}
    # Interleave: questions, a rename, more questions, a rename, more, then a check.
    steps = (plain[:10] + locates[:2] + rename_block(n1, r1) + plain[10:20] + counts[:2]
             + rename_block(n2, r2) + plain[20:30] + locates[2:] + counts[2:] + plain[30:] + [files])
    assert len(steps) == 50, len(steps)
    return {"id": f"{repo}-session-long", "repo": repo, "kind": "long", "steps": steps}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def load_sessions() -> list[dict]:
    return json.loads(SESSIONS_JSON.read_text())["sessions"]


def fresh_copy(repo: str, dest: Path) -> None:
    """A copy of the pinned corpus as a new git repository (no history)."""
    src = runner.CORPUS_REPOS[repo]
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    subprocess.run(
        ["rsync", "-a", "--exclude", ".git", "--exclude", ".reflex", f"{src}/", f"{dest}/"], check=True
    )
    git = ["git", "-C", str(dest), "-c", "user.email=b@example.com", "-c", "user.name=bench"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-qm", "corpus"], check=True)


def trial_dir(arm: str, sid: str, trial: int) -> Path:
    return WORK / f"{RESULTS.name}-{arm}-{sid}-{trial:02d}"


def run_one(arm: str, session: dict, trial: int, model: str, rfx: Path | None, tmp: Path) -> str:
    out = RESULTS / arm / session["id"] / f"trial_{trial:02d}.ndjson"
    if out.exists() and sum(1 for l in out.open() if '"type":"result"' in l.replace(" ", "")) >= len(session["steps"]):
        return f"skip {out}"
    tree = trial_dir(arm, session["id"], trial)
    fresh_copy(session["repo"], tree)
    cfg = runner.ARMS[arm]
    rfx_bin = rfx if cfg["mcp_command"] == "TARGET_RELEASE_RFX" else None
    if rfx_bin:
        subprocess.run([str(rfx_bin), "index", "--quiet"], cwd=tree, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    mcp_cfg = runner.make_mcp_config(
        rfx_bin, tmp, f"{arm}-{session['id']}-{trial}",
        mcp_env=cfg.get("mcp_env"), always_load=cfg.get("mcp_always_load", False),
    )
    cmd = runner.build_claude_cmd(arm, cfg, "PLACEHOLDER", mcp_cfg, model)
    assert cmd[-2:] == ["--", "PLACEHOLDER"], cmd[-3:]
    cmd = cmd[:-2] + ["--input-format", "stream-json", "--verbose"]
    out.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with out.open("w") as f:
        f.write(json.dumps({
            "type": "harness_metadata", "arm": arm, "task_id": session["id"], "trial": trial,
            "model": model, "tree": str(tree), "steps": len(session["steps"]),
        }) + "\n")
        f.flush()
        # One message at a time: a message queued while the agent is still working
        # is merged into the running turn, so the next one is sent only after the
        # previous answer's result event.
        proc = subprocess.Popen(cmd, cwd=tree, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, bufsize=1)
        for step in session["steps"]:
            proc.stdin.write(json.dumps(
                {"type": "user", "message": {"role": "user", "content": step["prompt"]}}) + "\n")
            proc.stdin.flush()
            for line in proc.stdout:
                f.write(line)
                try:
                    if json.loads(line).get("type") == "result":
                        break
                except ValueError:
                    pass
            else:
                break  # the process ended early
            f.flush()
        proc.stdin.close()
        for line in proc.stdout:
            f.write(line)
        proc.wait(timeout=120)
    return f"{arm}/{session['id']}/{trial:02d} exit={proc.returncode} {time.monotonic() - start:.0f}s"


def run(args) -> None:
    rfx = Path(os.environ.get("CARGO_TARGET_DIR", runner.REPO_ROOT / "target")) / "release" / "rfx"
    sessions = [s for s in load_sessions() if not args.sessions or s["id"] in args.sessions]
    if args.plan:
        # e.g. tokio-session-long:2 ripgrep-session-long:2 reflex-session-long:1
        by_id = {s["id"]: s for s in load_sessions()}
        jobs = [(a, by_id[sid], t) for item in args.plan for sid, n in [item.split(":")]
                for t in range(1, int(n) + 1) for a in args.arms]
    else:
        jobs = [(a, s, t) for t in range(1, args.n + 1) for s in sessions for a in args.arms]
    print(f"{len(jobs)} sessions on {args.jobs} workers, model {args.model}, rfx {rfx}")
    with tempfile.TemporaryDirectory() as tmp, cf.ThreadPoolExecutor(args.jobs) as pool:
        futs = [pool.submit(run_one, a, s, t, args.model, rfx, Path(tmp)) for a, s, t in jobs]
        for fut in cf.as_completed(futs):
            try:
                print(fut.result(), flush=True)
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f"FAILED: {e}", flush=True)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def answer_of(text: str):
    """The last {"answer": ...} object in the reply, fenced or bare."""
    text = text or ""
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    blocks += [text[i:] for i in [m.start() for m in re.finditer(r'\{\s*"answer"', text)]]
    for b in reversed(blocks):
        try:
            obj, _ = json.JSONDecoder().raw_decode(b.strip())
            return obj.get("answer")
        except (ValueError, AttributeError):
            continue
    return None


def norm_loc(x: str) -> str:
    x = str(x).strip().lstrip("./")
    m = re.match(r"(.+?):(\d+)", x)
    return f"{m.group(1)}:{m.group(2)}" if m else x


def rg_files(tree: Path, args: list[str]) -> set[str]:
    out = subprocess.run(["rg", *args], cwd=tree, capture_output=True, text=True).stdout
    return set(out.split())


def grade(step: dict, ans, tree: Path, scope: str) -> float:
    """1.0 = exact; lists are scored by F1 against the oracle on the final tree."""
    kind = step["kind"]
    if kind == "rename":
        return 1.0 if not rg(tree, "-w", step["old"], scope) else 0.0
    if kind == "locate":
        # Any definition of the name counts (a trait method and its impl are both one).
        name = re.search(r"`([^`]+)`", step["prompt"]).group(1)
        defs = set(rg(tree, rf"\b(fn|struct|enum|trait|type|const|static)\s+{name}\b", scope))
        defs.add(step["expected"][0])
        return 1.0 if isinstance(ans, list) and ans and norm_loc(ans[0]) in defs else 0.0
    if kind == "count_files":
        return 1.0 if ans == len(rg_files(tree, step["oracle"])) else 0.0
    if kind == "files":
        want = rg_files(tree, step["oracle"])
        got = {str(a).strip().lstrip("./") for a in ans} if isinstance(ans, list) else set()
    else:
        want = set(step["expected"]) if "expected" in step else set(rg(tree, *step["oracle"]))
        got = {norm_loc(a) for a in ans} if isinstance(ans, list) else set()
    if not want and not got:
        return 1.0
    tp = len(want & got)
    if tp == 0:
        return 0.0
    p, r = tp / len(got), tp / len(want)
    return 2 * p * r / (p + r)


def score(args) -> None:
    sessions = {s["id"]: s for s in load_sessions()}
    root = Path(args.results)
    if (root / "sessions").is_dir():
        root = root / "sessions"
    rows = []
    for path in sorted(root.glob("*/*/trial_*.ndjson")):
        meta, results, calls = None, [], collections.Counter()
        for line in path.open():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("type") == "harness_metadata":
                meta = e
            elif e.get("type") == "result":
                results.append(e)
            elif e.get("type") == "assistant":
                for c in e["message"].get("content", []):
                    if c.get("type") == "tool_use":
                        calls[c["name"]] += 1
        s = sessions[meta["task_id"]]
        tree = Path(meta["tree"])
        u = collections.Counter()
        for r in results:
            for k, v in (r.get("usage") or {}).items():
                if isinstance(v, (int, float)):
                    u[k] += v
        raw = sum(u[k] for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        weighted = (u["input_tokens"] + u["output_tokens"] + 0.1 * u["cache_read_input_tokens"]
                    + 1.25 * u["cache_creation_input_tokens"])
        cost = results[-1].get("total_cost_usd", 0) if results else 0
        scores = [grade(st, answer_of(r.get("result")), tree, SCOPES[s["repo"]])
                  for st, r in zip(s["steps"], results)]
        scores += [0.0] * (len(s["steps"]) - len(results))
        rows.append({
            "arm": meta["arm"], "session": s["id"], "trial": meta["trial"],
            "complete": len(results) >= len(s["steps"]), "turns": sum(r.get("num_turns", 0) for r in results),
            "tool_calls": sum(calls.values()), "reflex_calls": sum(v for k, v in calls.items() if "reflex" in k),
            "toolsearch": calls["ToolSearch"], "raw": raw, "weighted": weighted, "cost": cost,
            "accuracy": statistics.mean(scores),
        })
    if not rows:
        sys.exit(f"no transcripts under {root}")
    out_csv = root / "sessions_metrics.csv"
    with out_csv.open("w") as f:
        f.write(",".join(rows[0]) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r.values()) + "\n")

    arms = sorted({r["arm"] for r in rows}, key=lambda a: (a != "A", a))
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["arm"], r["session"])].append(r)
    print(f"{len(rows)} sessions ({sum(not r['complete'] for r in rows)} incomplete) -> {out_csv}\n")
    print(f"{'arm':<8}{'sessions':>9}{'turns':>7}{'calls':>7}{'reflex':>7}{'raw tok':>10}{'weighted':>10}{'cost $':>8}{'acc':>6}")
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        med = lambda k: statistics.median(r[k] for r in rs)  # noqa: E731
        print(f"{arm:<8}{len(rs):>9}{med('turns'):>7.0f}{med('tool_calls'):>7.0f}{med('reflex_calls'):>7.0f}"
              f"{med('raw'):>10.0f}{med('weighted'):>10.0f}{med('cost'):>8.3f}{statistics.mean(r['accuracy'] for r in rs):>6.3f}")
    print("\nratio vs A (median over sessions of per-session medians):")
    for arm in arms[1:]:
        line = [arm]
        for k in ("raw", "weighted", "cost", "turns"):
            ratios = []
            for sid in sessions:
                a, t = by.get(("A", sid)), by.get((arm, sid))
                if a and t:
                    ratios.append(statistics.median(x[k] for x in t) / statistics.median(x[k] for x in a))
            line.append(f"{k} {statistics.median(ratios):.2f}x" if ratios else f"{k} n/a")
        print("  " + "   ".join(line))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate")
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", default=["A", "B", "Beager"])
    r.add_argument("--sessions", nargs="*")
    r.add_argument("--n", type=int, default=4)
    r.add_argument("--plan", nargs="*", help="session:trials pairs instead of --sessions/--n")
    r.add_argument("--model", default="claude-sonnet-5")
    r.add_argument("--jobs", type=int, default=24)
    s = sub.add_parser("score")
    s.add_argument("--results", default=str(runner.RESULTS_DIR))
    args = p.parse_args()
    {"generate": lambda: generate(), "run": lambda: run(args), "score": lambda: score(args)}[args.cmd]()


if __name__ == "__main__":
    main()
