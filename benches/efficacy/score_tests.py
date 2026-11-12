#!/usr/bin/env python3
"""Offline score of a caller walk on the `tests` keys (tests_keys.py). No agent runs.

From the function, run `rfx refs X --at path:line`, take the function each use is in
(`in`), find that function's `fn` line in the same file (the nearest one above the
use), and repeat with `--at` on it. A caller with a test attribute is a test: it is
an answer and the walk stops there. `possible` lines are followed too with --poss.

  score_tests.py [--keys tasks/tests_keys.json] [--depth 8] [--poss] [-v]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import compiler_keys as ck
from score_at import SCOPES, f1
from tests_keys import TEST_ATTR

HERE = Path(__file__).resolve().parent


def refs_at(rfx: str, tree: Path, name: str, scope: str, at: str | None) -> dict:
    p = subprocess.run([rfx, "refs", name, scope, "--json", "--limit", "500"] + (["--at", at] if at else []),
                       cwd=tree, capture_output=True, text=True)
    return json.loads(p.stdout) if p.returncode == 0 else {}


def fn_line(tree: Path, path: str, line: int, name: str, cache: dict) -> int | None:
    """The line of `fn name` nearest at or above `line` in `path`."""
    if path not in cache:
        cache[path] = (tree / path).read_text(errors="replace").split("\n")
    pat = re.compile(r"\bfn\s+" + re.escape(name) + r"\b")
    for i in range(line - 1, -1, -1):
        if pat.search(cache[path][i]):
            return i + 1
    return None


def is_test(path: str, line: int, cache: dict) -> bool:
    lines = cache[path]
    return any(TEST_ATTR.search(lines[k]) for k in range(max(0, line - 7), line - 1))


def walk(rfx: str, tree: Path, scope: str, name: str, where: str, depth: int,
         poss: bool, plain: bool = False) -> tuple[set[str], int, int]:
    """(tests reached, rfx calls, functions visited)."""
    cache: dict = {}
    seen = {where}
    frontier = [(name, where)]
    tests, calls = set(), 0
    names = {name}  # --plain: each name is searched once (a grep walk)
    for _ in range(depth):
        nxt = []
        for n, at in frontier:
            calls += 1
            r = refs_at(rfx, tree, n, scope, None if plain else at)
            rows = r.get("references", []) + (r.get("possible", []) if poss else [])
            for row in rows:
                c = row.get("in")
                if not c or c.startswith("macro_rules!") or row.get("import"):
                    continue
                ln = fn_line(tree, row["path"], row["line"], c, cache)
                if ln is None:
                    continue
                loc = f"{row['path']}:{ln}"
                if loc in seen:
                    continue
                seen.add(loc)
                if is_test(row["path"], ln, cache):
                    tests.add(loc)
                elif not plain:
                    nxt.append((c, loc))
                elif c not in names:
                    names.add(c)
                    nxt.append((c, loc))
        frontier = nxt
        if not frontier:
            break
    return tests, calls, len(seen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfx", default="/scratch/cache/target/release/rfx")
    ap.add_argument("--keys", default=str(HERE / "tasks" / "tests_keys.json"))
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--poss", action="store_true")
    ap.add_argument("--plain", action="store_true", help="no --at: follow every caller by name")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    rows = []
    for s in json.loads(Path(args.keys).read_text())["sessions"]:
        tree = ck.KEYS_WORK / s["repo"]
        # Tests in files the test build never compiles (`cfg(loom)`, Windows) never
        # run, so the key cannot judge them: they count neither way.
        compiled = ck.compiled_files(s["repo"], tree)
        for st in s["steps"]:
            k = st["key"]
            want = set(k["tests"])
            got, calls, seen = walk(args.rfx, tree, SCOPES[s["repo"]], k["name"], k["where"],
                                    args.depth, args.poss, args.plain)
            got = {g for g in got if g.rsplit(":", 1)[0] in compiled}
            rows.append((s["repo"], k["name"], f1(got, want), len(want), len(got), calls,
                         sorted(want - got), sorted(got - want)))
    print(f"{'repo':8} {'name':32} {'F1':>5} {'want':>4} {'got':>4} {'calls':>5}")
    for r in rows:
        print(f"{r[0]:8} {r[1][:32]:32} {r[2]:5.2f} {r[3]:4d} {r[4]:4d} {r[5]:5d}")
        if args.verbose and (r[6] or r[7]):
            print(f"           missed: {r[6]}\n           extra: {r[7]}")
    n = len(rows)
    print(f"\n{n} questions. Mean F1 {sum(r[2] for r in rows) / n:.3f}; exact "
          f"{sum(1 for r in rows if r[2] == 1.0)}; mean rfx calls {sum(r[5] for r in rows) / n:.1f}")


if __name__ == "__main__":
    main()
