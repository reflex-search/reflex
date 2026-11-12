#!/usr/bin/env python3
"""Offline score of `rfx refs X --at path:line` on the compiler answer keys.

No agent runs. For every `explain` and `impact` question in tasks/sessions.json, run
`rfx refs` on the scratch copy the keys were built from (compiler_keys.KEYS_WORK) with
and without `--at`, and compare with the keys:

  sites    the impact key (error lines). `--at`: its references, `use` lines left out.
           Plain refs: every code line, the definition and `use` lines left out.
  +poss    `--at` references plus its `possible` lines (what an agent that keeps
           every unsure line would answer).
  callers  the explain key. `--at`: the `in` of each reference.

  score_at.py [--rfx PATH] [--repos reflex ripgrep tokio] [--sessions FILE] [-v]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import compiler_keys as ck

HERE = Path(__file__).resolve().parent
SCOPES = {"reflex": "src", "ripgrep": "crates", "tokio": "tokio/src"}
USE = re.compile(r"^\s*(pub(\([^)]*\))?\s+)?use\b")


def f1(got: set, want: set) -> float:
    if not got and not want:
        return 1.0
    tp = len(got & want)
    return 0.0 if tp == 0 else 2 * tp / (len(got) + len(want))


def refs(rfx: str, tree: Path, name: str, scope: str, at: str | None) -> dict:
    cmd = [rfx, "refs", name, scope, "--json", "--limit", "500"]
    if at:
        cmd += ["--at", at]
    p = subprocess.run(cmd, cwd=tree, capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "failed"}
    return json.loads(p.stdout)


def loc(r: dict) -> str:
    return f"{r['path']}:{r['line']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfx", default="/scratch/cache/target/release/rfx")
    ap.add_argument("--repos", nargs="+", default=list(SCOPES))
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--sessions", default=str(HERE / "tasks" / "sessions.json"),
                    help="a sessions file with compiler keys (a held-out set)")
    args = ap.parse_args()
    sessions = json.loads(Path(args.sessions).read_text())["sessions"]
    rows = []
    for s in sessions:
        if s["kind"] not in ("explain", "impact") or s["repo"] not in args.repos:
            continue
        tree = ck.KEYS_WORK / s["repo"]
        scope = SCOPES[s["repo"]]
        for st in s["steps"]:
            k = st["key"]
            want_sites = set(k["sites"])
            want_callers = set(k["callers"])
            plain = refs(args.rfx, tree, k["name"], scope, None)
            at = refs(args.rfx, tree, k["name"], scope, k["where"])
            plain_sites = {loc(r) for r in plain.get("references", [])
                           if loc(r) != k["where"] and not USE.match(r.get("preview", ""))}
            if "error" in at:
                row = dict(repo=s["repo"], kind=s["kind"], name=k["name"], error=at["error"],
                           plain=f1(plain_sites, want_sites), at=0.0, at_poss=0.0, callers=0.0,
                           possible=0, plain_lines=len(plain_sites), at_lines=0)
                rows.append(row)
                continue
            bound = [r for r in at["references"] if not r.get("import")]
            at_sites = {loc(r) for r in bound}
            poss = {loc(r) for r in at.get("possible", []) if not r.get("import")}
            # `in: "macro_rules! m"`: a use in a macro body, not a calling function.
            callers = {r["in"] for r in bound if r.get("in") and not r["in"].startswith("macro_rules!")}
            row = dict(repo=s["repo"], kind=s["kind"], name=k["name"],
                       plain=f1(plain_sites, want_sites), at=f1(at_sites, want_sites),
                       at_poss=f1(at_sites | poss, want_sites), callers=f1(callers, want_callers),
                       possible=len(poss), plain_lines=len(plain_sites), at_lines=len(at_sites),
                       missed=sorted(want_sites - at_sites - poss), extra=sorted(at_sites - want_sites),
                       callers_missed=sorted(want_callers - callers),
                       callers_extra=sorted(callers - want_callers))
            rows.append(row)
    print(f"{'repo':8} {'kind':8} {'name':32} {'plain':>6} {'at':>6} {'+poss':>6} {'callers':>7} "
          f"{'poss':>4} {'lines plain/at':>14}")
    for r in rows:
        print(f"{r['repo']:8} {r['kind']:8} {r['name'][:32]:32} {r['plain']:6.2f} {r['at']:6.2f} "
              f"{r['at_poss']:6.2f} {r['callers']:7.2f} {r['possible']:4d} "
              f"{r['plain_lines']:>6}/{r['at_lines']:<6}" + (f"  ERROR {r['error']}" if 'error' in r else ""))
        if args.verbose and (r.get("missed") or r.get("extra") or r.get("callers_missed")
                             or r.get("callers_extra")):
            for key in ("missed", "extra", "callers_missed", "callers_extra"):
                if r.get(key):
                    print(f"           {key}: {r[key]}")
    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n
    exact = lambda k: sum(1 for r in rows if r[k] == 1.0)
    print(f"\n{n} questions. Mean F1: plain refs {mean('plain'):.3f}, --at {mean('at'):.3f}, "
          f"--at+possible {mean('at_poss'):.3f}, callers {mean('callers'):.3f}")
    print(f"Exact (F1 = 1): plain {exact('plain')}, --at {exact('at')}, --at+possible "
          f"{exact('at_poss')}, callers {exact('callers')}. Possible lines: "
          f"{sum(r['possible'] for r in rows)}. Errors: {sum(1 for r in rows if 'error' in r)}")


if __name__ == "__main__":
    main()
