#!/usr/bin/env python3
"""
Columnar vs legacy MCP payload size, measured per call (no agent, no model).

Spawns `rfx mcp` twice in the pinned reflex corpus — once with the default
columnar result shape and once with REFLEX_MCP_COLUMNAR=0 — and sends the same
fixed list of search_code / search_regex calls to each. Reports the byte length
of `content[0].text` per call and the per-call reduction.

Session-level A/B runs cannot see this effect (turn count dominates
total_tokens), so it is measured here directly.

Usage:
  python3 benches/efficacy/columnar-payload.py [--rfx PATH] [--corpus DIR] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_CORPUS = SCRIPT_DIR / "corpus" / "reflex"

# Fixed query set: identifiers of different frequency, plus regexes.
CALLS: list[tuple[str, dict]] = [
    ("search_code", {"pattern": "extract_symbols"}),
    ("search_code", {"pattern": "SearchResult"}),
    ("search_code", {"pattern": "CacheManager"}),
    ("search_code", {"pattern": "Language"}),
    ("search_code", {"pattern": "unwrap", "limit": 50}),
    ("search_code", {"pattern": "TODO", "limit": 50}),
    ("search_code", {"pattern": "QueryEngine", "symbols": True}),
    ("search_code", {"pattern": "trigram", "contains": True, "limit": 100}),
    ("search_regex", {"pattern": r"fn (get|set)_\w+"}),
    ("search_regex", {"pattern": r"impl\s+\w+\s+for"}),
]


def default_rfx() -> Path:
    target = os.environ.get("CARGO_TARGET_DIR")
    if target:
        return Path(target) / "release" / "rfx"
    return SCRIPT_DIR.parent.parent / "target" / "release" / "rfx"


def run_session(rfx: Path, corpus: Path, columnar: bool) -> tuple[str, list[int]]:
    env = dict(os.environ)
    env.pop("REFLEX_MCP_COLUMNAR", None)
    if not columnar:
        env["REFLEX_MCP_COLUMNAR"] = "0"
    msgs = [
        {"jsonrpc": "2.0", "id": 0, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "columnar-payload", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]
    for i, (tool, args) in enumerate(CALLS, start=1):
        msgs.append({"jsonrpc": "2.0", "id": i, "method": "tools/call",
                     "params": {"name": tool, "arguments": args}})
    stdin = "".join(json.dumps(m) + "\n" for m in msgs)
    proc = subprocess.run([str(rfx), "mcp"], input=stdin, cwd=corpus, env=env,
                          capture_output=True, text=True, timeout=300)
    startup = next((l for l in proc.stderr.splitlines() if "reflex-mcp startup:" in l), "")
    sizes: dict[int, int] = {}
    for line in proc.stdout.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid = msg.get("id")
        if not isinstance(mid, int) or mid == 0:
            continue
        if "error" in msg:
            raise SystemExit(f"call {mid} failed: {msg['error']}")
        text = msg["result"]["content"][0]["text"]
        sizes[mid] = len(text.encode("utf-8"))
    missing = [i for i in range(1, len(CALLS) + 1) if i not in sizes]
    if missing:
        raise SystemExit(f"no response for calls {missing}; stderr:\n{proc.stderr[-2000:]}")
    return startup, [sizes[i] for i in range(1, len(CALLS) + 1)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rfx", type=Path, default=default_rfx())
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--json", type=Path, help="also write results as JSON here")
    args = ap.parse_args()

    if not (args.corpus / ".reflex").is_dir():
        raise SystemExit(f"{args.corpus} has no .reflex/ — run `rfx index` there first")

    startup_c, columnar = run_session(args.rfx, args.corpus, columnar=True)
    startup_l, legacy = run_session(args.rfx, args.corpus, columnar=False)
    if "columnar=on" not in startup_c or "columnar=off" not in startup_l:
        raise SystemExit(f"toggle did not take effect:\n  {startup_c}\n  {startup_l}")

    rows = []
    print(startup_c)
    print(f"{'tool':<13} {'arguments':<48} {'legacy B':>9} {'columnar B':>10} {'saving':>7}")
    for (tool, a), lg, co in zip(CALLS, legacy, columnar):
        saving = 1 - co / lg if lg else 0.0
        rows.append({"tool": tool, "arguments": a, "legacy_bytes": lg,
                     "columnar_bytes": co, "saving": saving})
        print(f"{tool:<13} {json.dumps(a)[:48]:<48} {lg:>9} {co:>10} {saving:>7.1%}")
    savings = [r["saving"] for r in rows]
    total = 1 - sum(columnar) / sum(legacy)
    print(f"\nper-call saving: median {statistics.median(savings):.1%}, "
          f"min {min(savings):.1%}, max {max(savings):.1%}; all calls pooled {total:.1%}")
    if args.json:
        args.json.write_text(json.dumps({"startup": startup_c, "calls": rows,
                                         "median_saving": statistics.median(savings),
                                         "pooled_saving": total}, indent=2))


if __name__ == "__main__":
    main()
