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
