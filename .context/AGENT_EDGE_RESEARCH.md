# Agent edge research

Measurements for `.context/AGENT_EDGE_PLAN.md`. Every number carries the date and the
Reflex version. The baselines of task 0.6 go here too.

## Install smoke test (task 1.5, 2026-11-10, reflex-search 2.3.1 from npm)

Script: `scripts/install-smoke.sh [image] [npm-spec]` (`PLUGINS=1` adds the Claude Code
plugins). The repository is mounted read-only and cloned twice (451 tracked files) as
the corpus. Host: x86_64, Docker 29.9.2.

### `node:24-bookworm` (Debian 12, Node 24.22.2, npm 11.19.1)

| Step | Time |
| --- | --- |
| `npm install -g reflex-search` (downloads the binary in `postinstall`) | 5.88 s |
| First CLI query `rfx query update_if_stale --count` (builds the index) | 1.08 s (engine 783 ms) |
| Second CLI query (fresh index) | 0.29 s (engine 29 ms) |
| First MCP call on a clone with no index (`initialize` + `search_code` count) | 1.02 s |
| Second MCP call (fresh index, new `rfx mcp` process) | 0.37 s |
| `npm install -g @anthropic-ai/claude-code` (2.1.322) | 11.99 s |
| `claude plugin marketplace add /src` + `install reflex@reflex` + `install reflex-mcp@reflex` | 2.08 s |

Manual steps: none beyond the commands. No `rfx index`: the first query and the first
MCP call each built the index. The plugin commands need no login. `claude mcp list`
showed `plugin:reflex-mcp:reflex: rfx mcp - ✔ Connected`. Not tested:
`claude plugin marketplace add reflex-search/reflex` (reads GitHub; the marketplace
is only on `feat/agent-edge-phase1` until it is merged).

### Findings

1. **The npm `rfx` is a Node wrapper (`run-rfx.js`): +~240 ms on every call.** Same
   container, fresh index, 3 runs each:

   | | `rfx --version` | `rfx query update_if_stale --count` |
   | --- | --- | --- |
   | npm wrapper (`/usr/local/bin/rfx`) | 244–258 ms | 270–343 ms |
   | native (`node_modules/.bin_real/rfx`) | 6–18 ms | 26–59 ms |

   For the MCP route it is paid once per server start. For the skill route (arm S) it
   is paid on every Bash call: latency, not tokens.
2. **The shell installer gives the native binary.**
   `curl --proto '=https' --tlsv1.2 -LsSf https://github.com/reflex-search/reflex/releases/download/v2.3.1/reflex-search-installer.sh | sh`:
   2.54 s, `rfx --version` 10 ms. One manual step: `~/.cargo/bin` must be on `PATH`
   (the installer says `source $HOME/.cargo/env`). The `latest/download/` URL form
   works the same.
3. **npm 12 (12.2.2) blocks `postinstall`** with a warning
   (`npm install -g --allow-scripts=reflex-search` to allow it). `rfx` still works: the
   wrapper downloads the binary at the first run (1.83 s), then 0.22 s per run.
4. **Alpine (musl) fails.** `node:24-alpine`: `npm error Platform with type "Linux" and
   architecture "x64" is not supported by reflex-search.` The release has
   `*-unknown-linux-gnu` binaries only (`dist-workspace.toml` `targets`). `cargo install
   reflex-search` is the only route there. Fix: add `x86_64-unknown-linux-musl` and
   `aarch64-unknown-linux-musl` to the dist targets (needs a CI run to prove the C
   dependencies — tree-sitter grammars, bundled SQLite, zstd — build on musl).

   **Fixed (2026-11-10, 2.3.1 + branch):** both musl targets are in `dist-workspace.toml`.
   - `dist plan`: native runners (`ubuntu-22.04`, `ubuntu-22.04-arm`) with `musl-tools`.
   - Local build (`rust:1-bookworm` + `musl-tools`, rustc 1.99.1): 4 min 17 s, a
     `static-pie` binary of 41.8 MB.
   - In `node:24-alpine` (Alpine 3.24) it ran `rfx query --count` (index built, 800 ms),
     `rfx refs --count`, `rfx locations` and an MCP `find_references` call. Tree-sitter,
     SQLite and zstd all work.
   - The generated npm package (`dist build --artifacts=global`) reads the libc with
     `detect-libc`: musl → `*-unknown-linux-musl-dynamic` (the static tarball); a glibc
     older than the build's → `*-musl-static`. The shell installer has the same table.
   - Not tested: the CI build itself and an npm install from a real release (needs a tag).

## Phase 0: arm S (CLI + skill) vs Beager (MCP) vs A (Grep) — 2026-11-10

Branch `feat/agent-edge-phase1` at `56ac4e8` (rfx 2.3.1 + branch), Claude Code 2.1.318,
Sonnet 5.5 (`claude-sonnet-5-5`), `session_bench.py` investigate sessions (12 questions:
8 find-every-occurrence, 2 where-defined, 2 how-many-files), 1 trial per arm and corpus.
Results: `benches/efficacy/results-sessions/sonnet55-phase0-investigate/`.

Arm S (`runner.py`): no MCP server, `--plugin-dir plugins/reflex`, the built `rfx` first
on PATH, the tree indexed before the trial (as for the MCP arms). Slash commands stay on,
so Claude Code's ~20 bundled skills are listed too (deny rules do not hide them).

### The skill alone does not fire

Smoke session (reflex corpus, skill only, no hook): the `reflex:reflex` skill was listed
and `Skill` was allowed. The agent never invoked it: Grep for 12 of 12 questions,
accuracy 1.000, $0.154 (`results-sessions/sonnet55-skill-smoke-nohook/`). A skill
description does not pull a plain search away from a built-in tool.

Fix: a SessionStart hook in the `reflex` plugin (`plugins/reflex/hooks/`) prints the
routing rules (965 chars, later 1,037) when `rfx` is on PATH. Same session with the hook:
`rfx` for 13 of 13 searches, Grep 0, accuracy 1.000, $0.146.

### Results (cost = `total_cost_usd`; weighted = input + output + 0.1 cache read + 1.25 cache write)

| Arm | Corpus | Turns | Reflex calls | Grep calls | Weighted | Cost $ | Acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | reflex | 24 | 0 | 12 | 76,379 | 0.149 | 1.000 |
| A | ripgrep | 24 | 0 | 12 | 80,734 | 0.161 | 1.000 |
| A | tokio | 24 | 0 | 12 | 96,645 | 0.208 | 1.000 |
| Beager | reflex | 24 | 12 | 0 | 84,687 | 0.160 | 1.000 |
| Beager | ripgrep | 24 | 12 | 0 | 89,449 | 0.173 | 1.000 |
| Beager | tokio | 24 | 12 | 0 | 105,769 | 0.228 | 1.000 |
| S | reflex | 25 | 13 | 0 | 73,423 | 0.146 | 1.000 |
| S | ripgrep | 25 | 13 | 0 | 72,657 | 0.149 | 1.000 |
| S | tokio | 25 | 1 | 12 | 99,445 | 0.237 | 1.000 |

Ratios vs A (median of sessions): Beager weighted 1.11×, cost 1.07×; S weighted 0.96×,
cost 0.98×. Where S used `rfx` (reflex, ripgrep): weighted 0.90–0.96×, cost 0.93–0.98×,
and 0.86–0.92× Beager's cost.

Warm-up prefix (cache write + read): A 19,662 tokens, S 17,007, Beager 22,660.

### Why S fell back to Grep on tokio

The first call was `rfx locations spawn_named tokio/src/` → `error: unexpected argument
'tokio/src/' found`. The agent said "My `rfx` call failed on an argument error, so I
didn't retry it" and used Grep for the remaining 11 questions. The reflex smoke session
made the same mistake (`... --preview src`) but retried without the path. Fix: `rfx
query`, `rfx refs` and `rfx locations` take grep-style PATH arguments (2026-11-10).

### Reading

- On plain grep questions, S is at parity with Grep (0.96× weighted) and cheaper than
  MCP (Beager 1.11×): its fixed text is a 1 KB hook, not ~7.8 KB of instructions and
  schemas. This leans D2 toward CLI + skill + hook. Not a D5 clear win (≤0.85× cost):
  investigate questions are grep's home ground.
- Open: the tokio S session with PATH arguments; the `aware` sessions (code-only uses,
  where `rfx refs` should win); 1 trial each, so ±5 % is noise.
