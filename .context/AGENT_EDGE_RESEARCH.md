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

### Tokio re-run with PATH arguments (2026-11-10, `5434e5c`)

`results-sessions/sonnet55-phase0-tokio-s/`: `rfx` for 12 of 12 searches (8 ×
`rfx locations X tokio/src/`, 2 × `--symbols`, 2 × `--paths tokio/src/`), Grep 0,
accuracy 1.000, weighted 71,818 (A 96,645: 0.74×), $0.154 (A $0.208: 0.74×; Beager
$0.228). Run ~1 h after A and Beager, so its prompt-cache state may differ.

### Aware sessions (code-only uses), A vs S vs Beager (2026-11-10, `5434e5c`)

`results-sessions/sonnet55-phase0-aware/`, 12 questions per session (10 lists, 2 counts
of the uses of a name in code, with comment and string lines left out), 1 trial.

| Arm | Corpus | Reflex calls | Grep calls | Weighted | Cost $ | Acc |
| --- | --- | --- | --- | --- | --- | --- |
| A | reflex | 0 | 12 | 87,664 | 0.179 | 1.000 |
| A | ripgrep | 0 | 12 | 90,697 | 0.184 | 1.000 |
| A | tokio | 0 | 12 | 102,003 | 0.219 | 1.000 |
| Beager | reflex | 12 | 0 | 91,383 | 0.170 | 1.000 |
| Beager | ripgrep | 13 | 0 | 98,052 | 0.183 | 1.000 |
| Beager | tokio | 12 | 0 | 98,783 | 0.186 | 1.000 |
| S | reflex | 13 | 0 | 73,717 | 0.143 | 1.000 |
| S | ripgrep | 12 | 0 | 72,888 | 0.143 | 1.000 |
| S | tokio | 12 | 0 | 84,035 | 0.177 | 1.000 |

Ratios vs A: S weighted 0.82×, cost 0.80×; Beager weighted 1.04×, cost 0.95×. S used
`rfx refs` for every question (`--count` for the counts) and never re-checked with grep.
S vs Beager: 0.78–0.95× cost.

### Phase 0 reading so far (D2)

- The CLI route (skill + SessionStart hook) beats MCP on every session measured: same
  accuracy, 0.68–0.95× its cost. The fixed text is the difference (~1 KB hook vs ~7.8 KB
  of instructions and schemas on every call).
- Against Grep: parity on plain searches (investigate 0.93–0.98× cost where S stayed on
  `rfx`; 0.74× on the tokio re-run), and **0.80× cost on code-only questions** — under
  D5's 0.85× line for this workflow. D5 also needs the LSP arm (L) and 2 of 3 headline
  workflows; neither is measured yet.
- 1 trial per cell: treat ±5 % as noise. Accuracy is 1.000 everywhere, so the decision
  rests on cost.

## Phase 0: arm L (LSP) — 2026-11-10/11, Claude Code 2.1.318, Sonnet 5.5

Setup (`benches/efficacy/README.md`): `plugins/lsp-rust` (the official
`rust-analyzer-lsp` server entry) by `--plugin-dir`, `LSP` in `--allowedTools`, Grep
allowed, no Reflex. rust-analyzer 1.99.1. The corpora's crates were fetched into the
Cargo cache on copies first (`cargo metadata --offline` failed before: the LSP arm would
have had no workspace). Full `rust-analyzer analysis-stats` per corpus (the warm-up's
upper bound, wall time, not tokens): ripgrep 22 s, tokio 61 s, reflex 81 s.

| Arm | Session | Grep calls | LSP calls | Weighted | Cost $ | Acc |
| --- | --- | --- | --- | --- | --- | --- |
| A | tokio aware | 12 | 0 | 102,003 | 0.219 | 1.000 |
| L | tokio aware | 12 | 0 | 100,416 | 0.212 | 0.917 |
| Lnudge | tokio aware | 12 | 0 | 112,476 | 0.265 | 1.000 |
| S | tokio aware | 0 | 0 (12 `rfx refs`) | 84,035 | 0.177 | 1.000 |

- **L:** the `LSP` tool was in the session (init `tools`); its schema is deferred (the
  warm-up prefix equals arm A's, 19,664 tokens). The agent never loaded or called it.
- **Lnudge:** L plus 383 chars of system prompt ("prefer the LSP tool (workspaceSymbol,
  then findReferences or incomingCalls) over Grep"). Still Grep for 12 of 12. Its cost
  is inflated: the appended prompt changed the prefix (warm-up cache write 19,814,
  read 0).
- Possible reason, not checked: an LSP reference search needs a file position first
  (2–3 calls per question vs 1 grep), and tokio's `cfg(windows)` code is not analysed
  on Linux.
- Stopped after the two smoke sessions (the user approved 5 more each): on code-only
  questions an agent with LSP behaves like arm A. For D5, beating A on these
  workflows is beating L. L may still matter for `explain` (call hierarchy) and
  `impact` sessions (tasks 0.3, 0.4), which do not exist yet.

## Compiler answer keys: `explain` and `impact` sessions (tasks 0.3, 0.4) — 2026-11-11

`session_bench.py generate-compiled` (`benches/efficacy/compiler_keys.py`) writes one
`explain` and one `impact` session of 8 questions per corpus (reflex, ripgrep, tokio).
rustc 1.99.1 on Linux x86_64. Each probe adds `__impact_probe: u32` to one function in
a scratch copy (`/scratch/cache/compiler-keys/<repo>`), runs `cargo check` twice
(normal and `--profile test`), and restores the file. 1–3 s per probe once the target
dir is warm; a full run with the first build: about 7 min.

- **impact key:** every error primary-span line under the scope (E0061, E0593 for a
  function used as a value). 3–25 lines per question.
- **explain callers key:** the innermost fn around each error line. Callees key: the
  `name(` / `.name(` / `path::name(` calls in the body's code text, minus keywords,
  macros, capitalised names and locals. So `new`, `map` and `collect` are in it; the
  prompt says to list standard-library calls too.
- No arm's tool writes a key: not Reflex, not rust-analyzer, not grep.

What the compiler cannot see, and the filter for each (a question is dropped):

| Gap | Example | Filter |
| --- | --- | --- |
| `cfg` this host does not build | `globals`: calls in `tokio/src/signal/windows/sys.rs` | a code line naming the fn in a file outside cargo dep-info (`compiled_files`) |
| cargo skips units that need the failed crate | `instant_to_tick`: tokio's `lib test` unit needs the lib (dev-dependencies), so `runtime/time/tests/mod.rs:174` was never checked | a grep code line not in the key that is in test code or in another package. Test code: a `tests/` path, a `mod` under any `cfg` naming `test` (`#[cfg(all(test, not(loom)))]`; the first filter, `#[cfg(test)]` only, missed `from_histogram`'s calls in `histogram.rs:330`), or an out-of-line test module (`#[cfg(test)] mod testutil;`) |
| `macro_rules!` bodies are checked only where expanded | `output_mut` at `macros/try_join.rs:234` | the same, for lines in a macro body |
| `tokio_unstable` code | `task::Builder` | built with `RUSTFLAGS=--cfg tokio_unstable` (as tokio's CI) |

Hand check of the 48 questions kept: 34 have no grep code line outside the key. In the
other 14, every extra line is a different item with the same name (a field
`index_status`, `Interest::is_writable`, `Instant::checked_add` vs
`usize::checked_add`, `StandardImpl::trim_ascii_prefix`), a local of the same name
(`let haystack_builder`), or a `mod` / `use` line. Self-callers are wrappers of the same
name, not recursion: `StandardImpl::trim_ascii_prefix` calls `util::trim_ascii_prefix`,
`Wheel::next_expiration` calls `Level::next_expiration`,
`AsyncReadExt::read_to_string` calls `io::util::read_to_string`.

Name-based limits of the grader: callers and callees are graded as bare names (F1), so
two callers with one name count once.

## Phase 0: tokio `explain` + `impact`, arms A, S, L — 2026-11-11, Sonnet 5.5, 1 trial

Claude Code 2.1.318, rfx 2.3.1 (`17cd51d`), `results-sessions/sonnet55-phase0-compiled/`.
8 questions per session. Spend: $1.31 plus warm-ups.

| Arm | Session | Turns | rfx calls | Grep / grep calls | LSP calls | Weighted | Cost $ | Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | explain | 30 | 0 | 11 | 0 | 106,244 | 0.239 | 1.000 |
| A | impact | 24 | 0 | 9 | 0 | 79,182 | 0.178 | 1.000 |
| L | explain | 31 | 0 | 8 | 0 | 109,068 | 0.244 | 1.000 |
| L | impact | 24 | 0 | 10 | 0 | 84,612 | 0.194 | 1.000 |
| S | explain | 32 | 2 | 10 | 0 | 106,187 | 0.238 | 1.000 |
| S | impact | 25 | 8 | 5 | 0 | 95,444 | 0.219 | 1.000 |

Vs A: L 1.05× weighted and cost; S 1.10× weighted, 1.11× cost. One trial: within noise.

- **Key bugs found by the run, fixed before this table** (every arm lost the same
  points): `#[cfg(all(...))]` attributes put `cfg`, `all`, `not` in the callees key
  (`new_unnamed`), and a turbofish holding `()` hid `cast::<()>()` (`waker_ref`).
  `callees` now blanks attributes and reads turbofishes 3 levels deep.
  `detect_circular_dependencies` (`collect::<Result<Vec<_>, _>>`) and
  `async_trace_leaf` (`#[inline]`) keys changed too.
- **Same workflow in every arm:** read the function, search its name, read the hits to
  tell same-name items apart. No accuracy headroom: every arm 1.000.
- **L:** 0 LSP calls again (the schema is deferred; Grep is enough). L ≈ A on all four
  Phase 0 session kinds measured.
- **S fell back to shell grep in `explain`** after two `rfx refs` calls. `rfx refs
  waker_ref` named `current_thread/mod.rs:563` as "the definition"; the question was
  `util/wake.rs:35`. tokio has three `waker_ref` functions, and `refs` lists every
  use of the name. `rfx refs read_to_string` named a `mod` line. So `refs` gave the
  agent grep's work plus a wrong label, and it used `grep -rn` for the other 6
  questions. In `impact` it kept `rfx refs` (8 of 13 searches).
- **Reading for D5:** on these two workflows Reflex has no edge yet. The gap is the
  one the questions name: "uses of THIS definition", not "uses of this name". Candidate
  (Phase 2): `rfx refs X --at path:line`, binding each use to a definition by imports,
  `self.`/`Type::` qualifiers and the file's own items, with the unbound uses listed
  apart. Measure it on these sessions (keys are ready) before building more.

## `rfx refs X --at path:line`: uses of ONE definition, scored offline — 2026-11-11

rfx 2.3.1 + `src/bind.rs` (branch `feat/agent-edge-phase1`). No agent runs: scored on
compiler answer keys with `benches/efficacy/score_at.py`. MCP: `find_references` with
the hidden `at` parameter (unlisted, in `HIDDEN_PARAMS`).

**What it does.** `rfx refs X` lists every use of the NAME. `--at path:line` sorts each
of those lines into `This` (uses the definition at that line), `Other` (another item,
a local, a field, a module, a `std` or dependency type) or `Possible` (the rules
cannot tell). It prints the `This` lines, each with its caller (`[in f]`), and the
`Possible` lines apart. Rules, from the tree-sitter syntax tree only (no type
inference, no macro expansion, no trait resolution):

- `use` paths; `Self::` / `Type::` / `module::` qualifiers (with re-exports from an
  ancestor module); the file's own free fns and imports; locals, parameters, patterns.
- Method calls: the receiver's written type: `self`, a typed parameter or `let`,
  `Foo::new()` / `Foo::now() + d`, `self.field` (the struct's field type),
  `self.f[i]`, `for x in self.f.iter()`. A type that the file imports from `std` or
  from a package dependency (`Cargo.toml`) that is not a workspace crate cannot reach
  an inherent method here. A builder chain rooted in a dependency's type stays there.
- Unknown receiver: `This` only when the target is the only method with that name,
  else `Possible`. Item-level brace macros (`cfg_rt! { … }`) are opened up first.
- Rust only (`--at` on another language is an error).

**Scores** (F1 of the impact key's sites, `use` lines left out; callers = F1 of the
explain key's callers from `in`; plain = `rfx refs X` without `--at`):

| Set | Questions | plain F1 | `--at` F1 | exact plain / `--at` | callers F1 | Possible lines |
| --- | --- | --- | --- | --- | --- | --- |
| Tuning (tasks/sessions.json explain + impact; rules built on it) | 48 | 0.851 | 0.989 | 34 / 46 | 0.989 | 2 |
| Held-out 1, blind (first look) | 48 | 0.663 | 0.953 | 21 / 43 | 0.953 | 3 |
| Held-out 1 after its 4 general fixes (no longer blind) | 48 | 0.663 | 0.997 | 21 / 47 | 0.996 | 3 |
| Held-out 2, blind, after the fixes (`tasks/at_holdout_2.json`) | 48 | 0.734 | 0.951 | 22 / 42 | 0.965 | 18 |

Held-out 1 is 16 impact questions per corpus (tokio, ripgrep, reflex), the next
candidates after the tuning set (`holdout_keys.py`, `tasks/at_holdout_1.json`). Its blind misses and the fixes
(each a general rule, with a unit test, `type_origin_and_typed_self`):

- `if self.config.unicode { … }` inside `vec![…]`: the macro rule read the `if` block
  as call arguments. A call now needs a `(…)` token tree.
- `deadline.saturating_duration_since(..)` with `use std::time::Instant` (tokio has its
  own `Instant`): type origin from the file's imports.
- `decode_builder.encoding(..)….bom_sniffing(..)` (`encoding_rs_io`) and
  `TranslatorBuilder::new()….unicode(..)` (`regex_syntax`): dependency builder chains.
- `fn try_acquire_owned(self: Arc<Self>)`: a typed `self` was not seen as a method
  receiver, so every `.try_acquire_owned()` was `Other`.

Held-out 2 (the next 16 per corpus, built after the fixes, not tuned on) misses:
`as_u64` (5 false `This`: `id.as_u64()` on receivers of unknown type, the target is
the only project `as_u64`), `with_current` (6 uses all `Possible`, 0 `This`),
`path_is_symlink` (4 `Possible`), `sleep_until` (1 false `This`), `poll_write_priv`
(2 `Possible`). The unknown-receiver rule is the main source of error in both sets.

Left (documented limits): tokio `checked_add` on integer locals whose type is not
written (`last`, `tail.rx_cnt`; `This` because the target is the only project method
`checked_add`); a use inside `debug_assert!(…)` is `Possible`; `enter_runtime` from a
receiver of unknown type (`Possible`).

**Reading.** The plain name search is the problem the tokio `explain` run showed:
0.663 F1 on held-out names (21 of 48 exact), because common names (`storage`,
`pretty`, `unicode`) have many items. `--at` answers the question "uses of THIS
definition" at 0.95 F1 on two blind sets (0.99 on the tuning set), with 3 and 18
`Possible` lines in 48 questions. Next: a light live
run (tokio explain + impact, arm S with `--at` in the skill and hook) to see whether
agents use it and stop re-reading hits.

### Live run: arm S with `--at` in the skill and hook — 2026-11-11, Sonnet 5.5, 1 trial

`8eb8376`, Claude Code 2.1.318, `results-sessions/sonnet55-at/` (A copied from the
`sonnet55-phase0-compiled` run the same day). Spend: $0.41 plus a warm-up.

| Arm | Session | Turns | rfx calls | grep calls | Weighted | Cost $ | Acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | explain | 30 | 0 | 11 | 106,244 | 0.239 | 1.000 |
| A | impact | 24 | 0 | 9 | 79,182 | 0.178 | 1.000 |
| S | explain | 30 | 8 | 5 | 100,865 | 0.232 | 1.000 |
| S | impact | 21 | 8 | 3 | 77,048 | 0.180 | 1.000 |

S vs A: 0.96× weighted, 0.99× cost (was 1.10× / 1.11× without the flag in the text).
One trial: within noise.

- **0 of 16 `rfx refs` calls passed `--at`**, though the SessionStart hook text named
  it (checked in both transcripts) and every question gives the definition's
  `path:line`. The agent ran `rfx refs X tokio/src/`, then found each caller by hand
  (`awk 'NR<=339 && /fn /…'`, `grep -n "fn " file`): the `[in f]` that `--at` prints.
- S stayed on `rfx refs` in `explain` this time (8 calls; it fell back to grep after 2
  before). Reason unknown (one trial).
- **Reading:** a flag named once in the preamble is not found. Agents follow what a
  tool's own output tells them (the zero-result `hint` pattern). Candidate: when the
  name has more than one definition, `rfx refs X` (and `find_references`) lists them
  and says `pass --at PATH:LINE for the uses of one`; plain `rfx refs` could print
  `[in f]` for every use too.

### Live run 2: `definitions` + hint, `[in f]` on every use — 2026-11-11, Sonnet 5.5, 1 trial

`d1ae6dc`, `results-sessions/sonnet55-at2/` (A: the same-day `sonnet55-phase0-compiled`
run). Spend: $0.38 plus a warm-up.

| Arm | Session | Turns | rfx calls | grep calls | Weighted | Cost $ | Acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | explain | 30 | 0 | 11 | 106,244 | 0.239 | 1.000 |
| A | impact | 24 | 0 | 9 | 79,182 | 0.178 | 1.000 |
| S | explain | 26 | 8 | 0 | 82,366 | 0.197 | 1.000 |
| S | impact | 23 | 1 | 11 | 82,023 | 0.184 | 1.000 |

- **explain: 0.78× A's weighted tokens, 0.82× its cost.** One `rfx refs X tokio/src/`
  per question, then the answer: no grep, no `awk`/`grep -n "fn "` to find callers
  (the `[in f]` column answered that). Still 0 calls with `--at`: with the callers
  shown and the definitions listed, the agent did not need it.
- **impact: 1.04× weighted, 1.03× cost.** After one `rfx locations X --preview` the
  agent used `grep -rnw` for every other question (the routing line for "where does
  X occur" points at `locations`, which keeps comment lines; it then filtered with
  `grep -v`). Same as the first S run on impact? No: run 1 kept `rfx refs` (8 calls).
  One trial each: routing on impact varies between runs.
- **Reading:** first single-trial result under D5's 0.85× cost line on a headline
  workflow (explain). Needs more trials (A and S) before it counts.

### Explain confirmation: A and S, 2 more trials — 2026-11-11, Sonnet 5.5, `d1ae6dc`

`results-sessions/sonnet55-explain3/`. Spend: $0.84 plus warm-ups.

| Arm | Trial | Turns | rfx calls | grep calls | Weighted | Cost $ | Acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1 | 30 | 0 | 9 | 102,974 | 0.230 | 1.000 |
| A | 2 | 30 | 0 | 8 | 106,825 | 0.235 | 1.000 |
| S | 1 | 24 | 8 | 0 | 79,573 | 0.193 | 1.000 |
| S | 2 | 25 | 8 | 0 | 77,140 | 0.186 | 1.000 |

Pooled with the earlier same-day trial of each arm (3 vs 3): A $0.235, 105,348
weighted; S $0.192, 79,693 weighted. **S = 0.82× A's cost, 0.76× its weighted
tokens, accuracy 1.000 in all 6.** Spread is small (S $0.186–0.197, A $0.230–0.239).
Vs L (1 trial, $0.244, 0 LSP calls): 0.79×.

- Every S trial: Read the function, one `rfx refs X tokio/src/`, answer. No grep, no
  caller search by hand. The `[in f]` column and the `definitions` list do the work
  A does with grep plus reads. 0 calls with `--at` in 3 trials.
- **D5:** explain meets the cost line (≤0.85× at equal accuracy) vs A, and vs L on
  1 trial of L. Workflow 1 of the 2 needed. Not yet: Opus, other corpora (ripgrep,
  reflex have explain keys), impact (≈1.0×, routing varies).

### Explain on ripgrep and reflex: A and S, 1 trial — 2026-11-11, Sonnet 5.5, `d1ae6dc`

`results-sessions/sonnet55-explain-xrepo/`. Spend: $0.89 plus warm-ups.

| Corpus | Arm | Turns | rfx calls | grep calls | Weighted | Cost $ | Acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ripgrep | A | 33 | 0 | 10 | 110,459 | 0.239 | 0.993 |
| ripgrep | S | 25 | 8 | 1 | 73,994 | 0.182 | 0.984 |
| reflex | A | 30 | 0 | 8 | 101,227 | 0.246 | 1.000 |
| reflex | S | 27 | 8 | 1 | 88,774 | 0.223 | 1.000 |

Per corpus, S vs A cost: ripgrep 0.76×, reflex 0.91×, tokio 0.82× (3 vs 3).
**All explain trials pooled (5 vs 5, three corpora): S $0.981 vs A $1.189 = 0.83×
cost; accuracy S 0.997, A 0.999.**

- Accuracy: each arm lost points on one ripgrep question. A named two extra callees
  (`find_iter_at_in_context`). S named `matched` as a caller of `add_def`: the use at
  `types.rs:514` is inside a `macro_rules! matched` body, `rfx refs` printed no
  `[in f]` for it, the agent read the file and named the macro (the key leaves macro
  bodies out). Candidate: print `[in macro matched!]` for a line in a
  `macro_rules!` body.
- reflex is the weakest corpus (0.91×): `search_with_metadata` and
  `to_prompt_string` have many uses; S used one grep and more reads there.
- **D5 on explain:** 0.83× pooled at accuracy within 0.002 of A (one disputed
  macro caller). Below the 0.85× line, by a small margin; reflex alone is above it.

### Impact with the `--at` routing line: A and S, 1 trial — 2026-11-11, Sonnet 5.5, `f2aba72`

`results-sessions/sonnet55-impact/`. Spend: $0.36 plus warm-ups.

| Arm | Turns | rfx calls | grep calls | Weighted | Cost $ | Acc |
| --- | --- | --- | --- | --- | --- | --- |
| A | 24 | 0 | 9 | 76,176 | 0.168 | 1.000 |
| S | 20 | 9 | 1 | 81,468 | 0.193 | 1.000 |

S vs A: 1.07× weighted, 1.15× cost, 0.83× turns. All S impact runs so far: $0.180,
$0.184, $0.193; A: $0.178, $0.168.

- **The routing line works: `--at` on 8 of 8 questions** (0 in the runs before, when
  the hook named it only in a parenthesis).
- **But every `--at` call came with `rfx locations X --preview` in the same command**,
  plus `sed` reads of the left-out lines: the agent checks what `--at` left out. The
  answer printed "N lines name other items (left out)" without naming them, so the
  agent fetched them. Fewer turns, more tokens per turn.
- Candidate: print the left-out lines in the `--at` answer itself (grouped
  `path:line,line`, with the definition each one names where known), so one call is
  the whole answer and its own check.

### Impact with `left_out` in the `--at` answer: A and S, 1 trial — 2026-11-11, `f04cf34`

`results-sessions/sonnet55-impact2/`. Spend: $0.36 plus warm-ups.

| Arm | Turns | rfx calls | grep calls | Weighted | Cost $ | Acc |
| --- | --- | --- | --- | --- | --- | --- |
| A | 21 | 0 | 11 | 77,617 | 0.171 | 1.000 |
| S | 19 | 8 | 1 | 80,587 | 0.192 | 1.000 |

S vs A: 1.04× weighted, 1.13× cost (was 1.07× / 1.15×).

- `--at` on 8 of 8 questions again. The paired check moved from `rfx locations
  --preview` to `grep -rn X` in the same command: the agent checks a completeness
  answer from a new tool whatever the answer shows. It also `sed`-read the false
  `This` lines of `checked_add` (integer receivers) and dropped them (accuracy 1.000).
- **Impact across runs (Sonnet 5.5, tokio): S $0.180, $0.184, $0.193, $0.192; A
  $0.178, $0.168, $0.171.** About 1.1× A. A's answer is cheap here (8 short grep calls,
  few hits per name), so a second check costs S its margin.
- Not tried: telling agents not to re-check `--at`. It is not exact (F1 0.95 blind;
  `checked_add` has 3 false `This` lines in this very session), so the check is what
  keeps accuracy at 1.000. The "do not re-check" sentence stays on plain `rfx refs`,
  which is exact for its question.
- **D5 status:** explain meets the line (0.83×, 5 vs 5). Impact ("change a signature
  safely") does not (≈1.1×). The third workflow ("find the tests for a change") has no
  session yet.

### Tests for a change: test-run keys and an offline caller walk — 2026-11-11, 2.3.1 + `--at`

Why agents re-checked `--at` in impact: the question asks for calls *and* value uses,
and the `--at` answer did not say it covers value uses (it does: `.map(case_variants)`).
The CLI now prints "Uses are every mention in code: calls, values (`.map(f)`) and
paths." under the `--at` summary. Not measured live yet.

**Key (`tests_keys.py`, `tasks/tests_keys.json`).** A marker `println!` at the start of
the function's body; the unit tests (`--lib --bins`) run with `--show-output`; a test
whose captured output holds the marker runs the function. Threads a test starts
(tokio workers) inherit the capture. Not a failure key: a panic probe misses tests that
catch panics. Integration tests and subprocesses (reflex's CLI tests) are out of
scope. Per probe: ripgrep 2–4 s, tokio 4–14 s, reflex not yet (125 s cold baseline).
16 questions (8 ripgrep, 8 tokio), 1–18 tests each; candidates with 0 or > 20 tests
dropped (ripgrep `as_byte` 137, `set_pos` 129).

**Offline walk (`score_tests.py`).** From the function: `rfx refs X --at`, take each
use's `in`, find that `fn` line above the use, repeat; a caller with a test attribute
is an answer.

| Walk | Mean F1 | Exact | rfx calls / question |
| --- | --- | --- | --- |
| name only (`rfx refs X`, every caller by name: a grep walk) | 0.040 | 0 | 834 |
| `--at` | **0.410** | 5 | 20 |
| `--at` + `possible` lines | 0.356 | 5 | 535 |

- A name walk reaches almost every test (412 of ripgrep's, 230 of tokio's): one
  generic name (`new`, `poll`) on the path joins everything.
- `--at` is 10× better, and still misses most runtime-wide reach:
  - receivers whose type is not written: `CurrentThread::new` and `block_on` are
    called on fields (`exec.block_on(...)`), so `--at` marks them `possible` (117 for
    `new`, 36 for `block_on`) and the walk stops (`set_thread_id` 0 of 18);
  - trait dispatch: `copy_to` ← `File::poll_read` ← `AsyncReadExt::read` in tests
    (0 of 8);
  - extras from tests the key never runs: `cfg(loom)`, `signal/windows/sys.rs`.
- So this workflow needs receiver types from field declarations (and trait impls),
  which is type inference. rust-analyzer (arm L) has it.

### Declared types for chains and field paths — 2026-11-11, 2.3.1 + `--at`

`bind.rs` now types three more receiver shapes, from declarations in the parsed files
(the files that name X, plus the `--at` file):
- a chain `Builder::new_current_thread().enable_all().build()`: each method's declared
  return type in turn (`method_returns`; `-> &mut Self` → the impl type,
  `io::Result<T>` / `Option<T>` → `T`; `unwrap`, `expect`, `clone` keep the type);
- `T::f()` with no `new`/`default` name: `f`'s declared return type;
- `x.a.b` for any typed `x`, with the struct in any parsed file (was: `self.a`, struct
  in the same file); `let v = <chain>;` gets the chain's type.
A wrapper root with no known inner type (`Arc::new(x).clone()`) stays unknown.

`find_references` no longer refuses a 2-character name (`rt`): it forces the symbol
lookup that refused it. `rfx refs rt` failed outright before.

| Set | `--at` F1 before | after |
| --- | --- | --- |
| tuning 48 | 0.989 | 0.989 |
| held-out 1 | 0.997 | 0.997 |
| held-out 2 | 0.951 | 0.951 |
| tests walk 16 (`score_tests.py`) | 0.410 | **0.482** (6 exact) |

The tests scorer now drops answers in files the test build never compiles
(`cfg(loom)`, Windows): the key cannot judge them. Before the change, with that rule,
the walk scored about 0.43.
- `set_thread_id` 0.00 → 0.91: `Builder::build` went from 0 uses / 37 possible to 29
  uses / 6 possible, so the walk reaches the runtime tests.
- Over-reach is the new error: a walk that reaches `Builder::build` reaches every
  runtime test, but `globals` runs only with the signal driver on (`enable_all`).
  Static reach is "may run"; the key is "did run". ripgrep `quit_byte` / `binary_data`
  reach 101 printer tests the same way (key: 6).
- Still 0: trait dispatch (`copy_to` via `poll_read`, 0 of 8), `is_cancelled`.
- Held-out 2 misses are unchanged (`as_u64` unknown receiver, `with_current`): those
  receivers are locals with no written type and no chain.

### Tests for a change, live: arm A, 1 trial, tokio — 2026-11-11, `384449d`

`results-sessions/sonnet55-tests/`, `session_bench.py generate-tests` (prompt: "Which
unit tests under `tokio/src/` run it, directly or through other functions? List each
test function by the line of its `fn`"). Grade: F1 against the test-run key; an
attribute line counts as its `fn` line; tests in files the key build never compiled
count neither way. Spend: **$1.116** (estimate was $0.20: 59 turns, 51 tool calls).

| Question | Key tests | Answered | F1 | Turns |
| --- | --- | --- | --- | --- |
| spawn_named | 2 | 15 | 0.00 | 10 |
| is_cancelled | 1 | 3 | 1.00 | 3 |
| globals | 4 | 3 | 0.86 | 9 |
| async_trace_leaf | 9 | 16 | 0.80 | 15 |
| instant_to_tick | 8 | 9 | 1.00 | 6 |
| set_thread_id | 18 | 43 | 1.00 | 6 |
| copy_to | 8 | 8 | 1.00 | 6 |
| waker_ref | 14 | 22 | 1.00 | 4 |

Accuracy **0.832**; the offline `--at` walk scored 0.482 on the same keys. Answer
counts include `cfg(loom)` tests (they count neither way).
- No test runs: `grep`, `sed` and reads only. The agent reasons where a walk cannot:
  `File::read` → `poll_read` → `copy_to` (trait dispatch, walk 0 of 8, agent 8 of 8);
  which runtime tests turn the signal driver on (`globals`).
- So a `--tests` walk does not beat a grep agent on accuracy. Reflex can only win
  this workflow on cost, by shortening the caller-following part of those 59 turns.

### Tests for a change, live: arm S, 1 trial, tokio — 2026-11-11, `7e32028`

Same session as arm A above. Spend: $1.098.

| Arm | Turns | Calls | rfx | grep | Weighted | Cost $ | Acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 59 | 51 | 0 | 43 | 549,335 | 1.116 | 0.832 |
| S | 63 | 55 | 15 | 32 | 535,251 | 1.098 | **0.970** |

S vs A: 0.98× cost, 0.97× weighted, +0.138 accuracy. Per question (F1, S/A):
`spawn_named` 1.00/0.00, `async_trace_leaf` 0.90/0.80, the rest equal.
- The gain is mostly one question: `spawn_named`. S's `rfx refs spawn_named --at`
  led it to both compiled tests (`idle_notified_set.rs:501`,
  `task_combinations.rs:69`); A listed only `cfg(loom)` tests, which count neither way.
- 1 trial: one question decides most of the gap, so it needs a repeat.
- **D5:** meets the accuracy rule vs A (≥ 0.05 at ≤ 1.0× cost) on this one trial. Not
  run vs L.
- One `--at` call failed: run in `tokio/src` with a path relative to it
  (`--at runtime/scheduler/…`); `--at` read paths from the index root. Fixed: a path
  that exists from the working directory is taken relative to it.

### Tests for a change, live: A, S, L — 2026-11-11, Sonnet 5.5, `9239357`

`--plan tokio-session-tests:3 ripgrep-session-tests:2`, arms A S L (tokio trial 1 of A
and S is the run above). API-equivalent cost (Max plan usage, not a bill).

| Corpus | Arm | Cost $ (per trial) | Mean cost | Acc (per trial) | Mean acc |
| --- | --- | --- | --- | --- | --- |
| tokio | A | 1.116, 0.972, 1.063 | 1.050 | 0.832, 0.961, 0.905 | 0.900 |
| tokio | S | 1.098, 0.890, 1.241 | 1.076 | 0.970, 0.912, 0.863 | 0.915 |
| tokio | L | 1.161, 1.039, 1.081 | 1.094 | 0.925, 0.738, 0.887 | 0.850 |
| ripgrep | A | 0.272, 0.259 | 0.265 | 0.917, 1.000 | 0.958 |
| ripgrep | S | 0.297, 0.277 | 0.287 | 1.000, 1.000 | 1.000 |
| ripgrep | L | 0.335, 0.251 | 0.293 | 1.000, 1.000 | 1.000 |

- S vs A: tokio 1.02× cost, +0.015 accuracy; ripgrep 1.08× cost, +0.042.
  Pooled ratio of medians: 1.06× cost, 1.05× weighted tokens.
- S vs L: tokio 0.98× cost, +0.065; ripgrep 0.98× cost, +0.000.
- **D5: not met vs A** (gain < 0.05 and cost > 1.0× on both corpora). Vs L the tokio
  gain qualifies, but L is not a real LSP arm here (next point).
- **Arm L made 0 LSP calls in all 5 sessions**, though the tool was listed in every
  init. On this workflow L is arm A plus a larger tool list; its numbers say nothing
  about LSP call hierarchy.
- The spread is one question: `spawn_named` (tokio q0). F1 per trial: A 0.0, 1.0, 1.0;
  S 1.0, 0.67, 0.0; L 0.67, 0.0, 1.0. S trial 3 ran `rfx refs spawn_named` without
  `--at` and listed only `cfg(loom)` tests, A trial 1's miss. `--at` use varies per
  trial (tokio: about 12, 14 and 7 calls). The other 7 tokio questions are within ±0.1 across arms,
  except `globals` (A3 0.40, L2 0.00) and `async_trace_leaf` (L3 0.17).
- Turns: tokio A 51–59, S 50–65, L 52–61. Reflex replaces some grep calls
  (S 25–38 vs A 39–43) but agents still read the code to trace each path.
