#!/usr/bin/env python3
"""Answer keys for the `tests` session ("find the tests for a change"), from a test run.

The trick: put a marker `println!` at the start of one function's body in a scratch
copy of the corpus, run the unit tests with `--show-output`, and read which tests
printed it. Libtest captures each test's output under the test's name, and threads a
test starts (tokio workers too) inherit the capture. So the key is "every unit test
that runs this function", through any call chain, trait object or callback. No arm's
own tool (Reflex, rust-analyzer, grep) writes it.

Scope: the unit tests of the session scope (`--lib --bins`). Integration tests and
tests that start a subprocess are out: a subprocess's output is not captured.

A test whose `fn` line cannot be found from its name (a test made by a macro, or two
candidates) drops the question.

  tests_keys.py --per-repo 8 --out tasks/tests_keys.json [--repos tokio]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import compiler_keys as ck
import rust_code
import session_bench as sb

MARK = "@@REFLEX_TESTS_PROBE@@"
TEST_ARGS = {
    "reflex": ["--lib", "--bins"],
    "ripgrep": ["--workspace", "--lib", "--bins"],
    "tokio": ["-p", "tokio", "--lib", "--features", "full"],
}
RUNNING = re.compile(r"^\s*Running unittests \S+ \((\S+)\)")
HEADER = re.compile(r"^---- (\S+) stdout ----$")
# An attribute that makes the next fn a test: #[test], #[tokio::test], #[crate::test], ...
TEST_ATTR = re.compile(r"#\[\s*(?:[\w:]+::)?test\b")


def target_dirs(repo: str, tree: Path) -> dict[str, str]:
    """Test binary name (`grep_printer`) -> the directory of its crate root, relative
    to the tree. Cargo prints the root relative to its package (`src/lib.rs`)."""
    meta = json.loads(subprocess.run(["cargo", "metadata", "--no-deps", "--format-version", "1"],
                                     cwd=tree, env=ck.cargo_env(repo), capture_output=True,
                                     text=True, check=True).stdout)
    return {t["name"].replace("-", "_"): os.path.relpath(Path(t["src_path"]).parent, tree)
            for p in meta["packages"] for t in p["targets"]}


def run_tests(repo: str, tree: Path) -> tuple[set[tuple[str, str]], str]:
    """(crate root directory, test name) of every test that printed MARK, and the raw
    output."""
    dirs = target_dirs(repo, tree)
    proc = subprocess.run(["cargo", "test", *TEST_ARGS[repo], "--no-fail-fast", "--",
                           "--show-output"],
                          cwd=tree, env=ck.cargo_env(repo), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    reached, root, current = set(), None, None
    for line in proc.stdout.splitlines():
        m = RUNNING.match(line)
        if m:
            root, current = dirs.get(Path(m.group(1)).name.rsplit("-", 1)[0]), None
            continue
        m = HEADER.match(line)
        if m:
            current = m.group(1)
            continue
        if line in ("successes:", "failures:"):
            current = None
        elif current and root and MARK in line:
            reached.add((root, current))
    return reached, proc.stdout


def test_fns(tree: Path, crate_dir: str) -> dict[str, list[tuple[str, int]]]:
    """fn name -> [(path, line)] of every test fn under `crate_dir` (a test attribute
    within the 6 lines above its `fn` line)."""
    out: dict[str, list[tuple[str, int]]] = {}
    for f in sorted((tree / crate_dir).rglob("*.rs")):
        rel = os.path.relpath(f, tree)
        src = f.read_text(errors="replace")
        lines = src.split("\n")
        mask = rust_code.code_mask(src)
        off = 0
        starts = []
        for l in lines:
            starts.append(off)
            off += len(l) + 1
        for i, l in enumerate(lines):
            m = re.match(r"\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)", l)
            if not m or not mask[starts[i] + l.index("fn")]:
                continue
            if any(TEST_ATTR.search(lines[k]) for k in range(max(0, i - 6), i)):
                out.setdefault(m.group(1), []).append((rel, i + 1))
    return out


def locate(name: str, cands: list[tuple[str, int]]) -> tuple[str, int] | None:
    """The `fn` line of test `a::b::tests::name`: the candidate whose path holds the
    most module segments; None on a tie."""
    segs = name.split("::")[:-1]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]

    def score(c):
        parts = set(Path(c[0]).with_suffix("").parts)
        return sum(1 for s in segs if s in parts)

    ranked = sorted(cands, key=score, reverse=True)
    if score(ranked[0]) == score(ranked[1]):
        return None
    return ranked[0]


def probe(repo: str, tree: Path, name: str, where: str) -> dict | None:
    """The tests key of `name` at `where`, or None (no body, a build error, no test
    reaches it, or a test with no findable `fn` line)."""
    path, line = where.rsplit(":", 1)
    f = tree / path
    src = f.read_text()
    try:
        parts = ck.fn_parts(src, int(line), name)
    except (ValueError, IndexError):
        return None
    if not parts or re.search(r"\bconst\s+(?:unsafe\s+)?fn\b", src.split("\n")[int(line) - 1]):
        return None
    b = parts["b_open"] + 1
    f.write_text(src[:b] + f' ::std::println!("{MARK}");' + src[b:])
    t0 = time.time()
    try:
        reached, raw = run_tests(repo, tree)
    finally:
        f.write_text(src)
    if "error[E" in raw or "could not compile" in raw:
        return None
    tests, cache = [], {}
    for root, tname in sorted(reached):
        d = root
        if d not in cache:
            cache[d] = test_fns(tree, d)
        loc = locate(tname, cache[d].get(tname.rsplit("::", 1)[-1], []))
        if loc is None:
            return {"name": name, "where": where, "unlocated": tname}
        tests.append(f"{loc[0]}:{loc[1]}")
    return {"name": name, "where": where, "tests": sorted(set(tests)),
            "test_names": sorted(t for _, t in reached), "seconds": round(time.time() - t0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-repo", type=int, default=8)
    ap.add_argument("--repos", nargs="+", default=["tokio", "ripgrep", "reflex"])
    ap.add_argument("--max-tests", type=int, default=20)
    ap.add_argument("--max-probes", type=int, default=40, help="stop a repo after this many")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    # A stopped run must still restore the probed file (`finally` in `probe`).
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    out_path = Path(args.out)
    done = json.loads(out_path.read_text())["sessions"] if out_path.exists() else []
    for repo in args.repos:
        tree = ck.KEYS_WORK / repo
        steps = [st for s in done if s["repo"] == repo for st in s["steps"]]
        tried = {st["key"]["name"] for st in steps}
        probes = 0
        for name, _, _, where in sb.candidates(repo):
            if len(steps) >= args.per_repo or probes >= args.max_probes:
                break
            if name in tried:
                continue
            path, line = where.rsplit(":", 1)
            if f"fn {name}" not in (tree / path).read_text().split("\n")[int(line) - 1]:
                continue
            if ck.in_trait_impl((tree / path).read_text(), int(line)):
                continue
            probes += 1
            k = probe(repo, tree, name, where)
            print(repo, name, where, "->", "skip" if not k else
                  (f"unlocated {k['unlocated']}" if "unlocated" in k else
                   f"{len(k['tests'])} tests in {k['seconds']}s"), flush=True)
            if not k or "unlocated" in k or not (1 <= len(k["tests"]) <= args.max_tests):
                continue
            steps.append({"kind": "tests", "key": k, "expected": k["tests"]})
        done = [s for s in done if s["repo"] != repo] + [
            {"id": f"{repo}-session-tests", "repo": repo, "kind": "tests", "steps": steps}]
        out_path.write_text(json.dumps({"sessions": done}, indent=1) + "\n")
        print(f"{repo}: {len(steps)} questions -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
