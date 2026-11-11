#!/usr/bin/env python3
"""Held-out compiler keys for `score_at.py`: impact questions not in tasks/sessions.json.

Takes the next candidates (`session_bench.candidates`, the order `generate-compiled`
uses) whose names are in no earlier set, probes each with `compiler_keys.probe`, and
keeps those with 1 to 25 error sites. No agent runs; it costs `cargo check` time only.

  holdout_keys.py --per-repo 16 --out tasks/at_holdout_1.json [--exclude FILE ...]
  score_at.py --sessions tasks/at_holdout_1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import compiler_keys as ck
import session_bench as sb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-repo", type=int, default=16)
    ap.add_argument("--repos", nargs="+", default=["tokio", "ripgrep", "reflex"])
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="earlier held-out files: their names are skipped")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    used = {st["key"]["name"] for s in sb.load_sessions()
            if s["kind"] in ("explain", "impact") for st in s["steps"]}
    for f in args.exclude:
        used |= {st["key"]["name"] for s in json.loads(Path(f).read_text())["sessions"]
                 for st in s["steps"]}
    out = []
    for repo in args.repos:
        tree = ck.KEYS_WORK / repo
        compiled = ck.compiled_files(repo, tree)
        kept = 0
        for name, _, _, where in sb.candidates(repo):
            if kept >= args.per_repo:
                break
            if name in used:
                continue
            path, line = where.rsplit(":", 1)
            if f"fn {name}" not in (tree / path).read_text().split("\n")[int(line) - 1]:
                continue
            k = ck.probe(repo, tree, sb.SCOPES[repo], name, where, compiled)
            if not k or not k["sites"] or len(k["sites"]) > 25:
                continue
            kept += 1
            print(repo, name, len(k["sites"]), flush=True)
            out.append({"id": f"{repo}-holdout", "repo": repo, "kind": "impact",
                        "steps": [{"kind": "impact", "key": k}]})
    Path(args.out).write_text(json.dumps({"sessions": out}, indent=1) + "\n")
    print("wrote", len(out), "questions to", args.out)


if __name__ == "__main__":
    main()
