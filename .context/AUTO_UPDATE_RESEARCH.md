# Auto-update: every command answers from a fresh index

**Status:** built on `feature/auto-update` (2026-09-30), steps 1–8 and docs; step 9
(MCP text) waits for the user. See "As built" at the end. Decisions: `.context/TODO.md`
("Auto-update").

## Goal

A user or agent never runs `rfx index` or calls `index_project` to see an edit. Every
command that reads the index first brings it up to date, then answers. This removes the
`check_index_status` and `index_project` round trips that made Reflex MCP cost 1.26–1.66×
Grep's tokens (`.context/EFFICACY-2.0.3.md`).

## Decisions (user, 2026-09-29)

1. **No watcher.** The update runs inside the command, after the freshness check says
   `stale`. A fresh index pays nothing extra.
2. **Default on, everywhere**: `rfx query`, `deps`, `analyze`, `ask`, `context`, `stats`,
   `list-files`, `snapshot`, `pulse`, interactive mode, `rfx serve`, `rfx mcp`. One opt-out
   flag, `--no-update`, accepted by every command.
3. **Always wait.** A command never returns stale results because the update is slow.
   Claude Code waits ~28 h for an MCP tool call by default (`MCP_TOOL_TIMEOUT`, per-server
   `timeout`; progress notifications do not reset it — code.claude.com/docs/en/mcp.md).
4. **No index → build it**, in any directory. Running in `~` or a subdirectory is the
   same question as running `rfx index` there; no special guard.
5. **MCP text** ("call `check_index_status` / `index_project`") is removed only after the
   agent-style test (below) shows 95–100 % of answers equal a fresh build.
6. **Another version's index.** CLI commands rebuild it, as `rfx index` does. `rfx mcp`
   and `rfx serve` do not write it; they answer with a warning naming the owner. A server
   that cannot read the index at all rebuilds once (today's `with_corruption_recovery`).
   No loop: a newer version reads an older version's index, so only the older side fails
   to read; after its one rebuild the newer side reads and does not write.

## Facts from the code survey (2026-09-29)

Freshness check (`src/query/mod.rs`):
- `status_cache::snapshot(cache)` (~3977) is the check: git candidates (or a walk), each
  confirmed by fingerprint. Memoised 1 s per root; `invalidate(root)` clears it and every
  index run calls it. The result holds `WorktreeChanges` (`src/git.rs:163`): lists capped
  at 100 per category (`truncated`), true totals in `*_count`, paths relative to the root.
- `search_with_metadata` (~830) runs the check on a scoped thread while it searches.
- `search`, `search_ast_all_files`, `search_ast_with_text_filter` (and `find_symbol`,
  `search_ast`, `list_by_kind` through `search`) run an older heuristic,
  `check_index_freshness` (~3034): branch exists, commit moved, 10 sampled mtimes; it only
  prints warnings. `rfx query --ast --json` hard-codes `status: fresh` (`cli/query.rs`
  ~660).
- `compute` returns **Fresh** when `cache.status_reads` fails (~4036).

Who indexes today, and with what config:
- `rfx index` (`cli/index.rs:93`) loads `.reflex/config.toml`, waits 30 s for the lock,
  self-heals a version mismatch, prints a summary and spawns `index-symbols-internal`.
- MCP `index_project` (`mcp.rs` ~1288), `POST /index` (`cli/serve.rs` ~239), `rfx watch`
  and interactive mode use `IndexConfig::default()`: they **ignore config.toml**, while the
  freshness check reads it (`query/mod.rs` ~4076). An update built on the default config
  would index a different file set than the check expects.
- `Indexer::update_paths` has no production caller. `rfx watch` collects changed paths
  but calls the full `index`.

Lock (`src/atomic_write.rs`): `index.lock`; `IndexConfig::lock_wait_secs` (default 0 =
fail at once with `IndexLocked`). `index` and `try_update_paths` both take it, then ask the
symbol pass to yield (up to 10 s, else `SymbolIndexingInProgress`). `update_paths` falls
back to `index`, which takes the lock again.

Front ends:
- MCP: root `.`, one request at a time; each handler makes its own `QueryEngine`. Open
  indexes are shared per process (`open_index::get_or_open`, reopened when the store files
  change). `with_corruption_recovery` rebuilds once on `CacheCorrupted`.
- `rfx serve`: root `.`, async handlers; `POST /index` runs on the async thread.
- MCP tools with no freshness fields: `search_ast`, the dependency and structural tools,
  `gather_context`; `find_references`, `list_locations`, `count_occurrences` carry
  `status` only; `mode: "count"` carries nothing (TODO Open bugs).
- clap: `Cli` (`cli/mod.rs:31`) has only `verbose`; a `#[arg(long, global = true)]` field
  there makes `--no-update` valid before or after any subcommand.

## Review findings (2026-09-29, before coding)

A design review against the code found these; the design below includes each fix.

1. `QueryEngine` pins its open index in a `OnceLock` (`query/mod.rs` ~588): a retry on
   the same engine would read the pre-update snapshot. → resettable handle.
2. The check's lists are capped at 100 and `Snapshot` is private. → an uncapped path list
   inside the snapshot and a crate-level `update_plan`.
3. `status_cache::snapshot` computes outside its lock and inserts afterwards: a check that
   started before an update can re-insert a stale verdict. → an invalidation epoch.
4. `compute` says Fresh when meta.db cannot be read. → `Unknown`, no update.
5. `.gitignore` / `.ignore` / `.rgignore` and `.reflex/config.toml` edits never make the
   index stale (`classify_one` drops paths that are not indexable; `.reflex/` is usually
   ignored by git). → such candidates plan a full run; config.toml fingerprint stored.
6. Background compaction deletes meta.db rows of missing files but leaves them in the
   stores: the check then cannot see the deletion and search still returns the file. →
   compaction stops deleting rows.
7. No code inside the indexer or the symbol pass runs `QueryEngine`; `update_paths`
   releases the lock before falling back to `index`. No self-deadlock.
8. `lock_wait_secs = u64::MAX` is safe (`acquire_with_timeout` compares elapsed ≥ limit).
9. An update cancels the symbol pass; nothing restarts it after a path update, so repeated
   edit-then-query could starve it. → restart a `Cancelled` pass.
10. `CacheManager::load_index_config` (`cache.rs:666`) is already in the library; the
    `--languages` override is never persisted. → persist it.
11. `cli/index.rs:140` keeps the version-mismatch self-heal out of MCP on purpose (the
    1.6/1.7 rebuild stampede). → decision 6.
12. `rfx serve` calls blocking code on async handlers (search, stats, index). →
    `spawn_blocking`.
13. About 15 tests assert `stale` or "Run `rfx index`" in-process. → the library default
    stays off (`QueryEngine::with_update` opts in); every command turns it on through one
    helper, enforced by a clippy `disallowed-methods` rule.
14. The open bug "a tracked file `.gitignore` starts to ignore is reported as added by
    every check" would make every query update again. → fixed first, plus a loop guard.

## Design

The approved plan (2026-09-29). Steps and gates are at the end.

### Library entry point: `src/auto_update.rs`

`update_if_stale(cache, opts) -> Updated { Nothing, Paths(n), Index, Built, Skipped(reason) }`

1. No index → `Indexer::index` → `Built`.
2. `update_plan` from the check: `Fresh` → `Nothing`; `Paths(list)` →
   `update_paths(list)`; `Full` (truncated or missing lists, ignore file, config) →
   `index`; `Unknown` → no write, `Skipped`.
3. Config: `load_index_config` plus the persisted `--languages`; lock wait forever.
4. Version mismatch: CLI clears and rebuilds; servers skip with the owner warning.
5. Symbol pass: retry while it makes progress; restart it after a run that cancelled it
   or after a full run (`BackgroundIndexer::spawn_detached`, binary callers only).
6. One update per root per process; `index.lock` across processes.
7. Loop guard: a path set an update could not make fresh is not updated again.
8. Any failure → answer from the current index, `stale`, reason in `warnings`.

### Query engine

- `QueryEngine::with_update(opts)`; `new` unchanged. Front ends use one helper.
- `search_with_metadata`: search and check in parallel; stale → update → search and check
  again; at most two updates per call.
- The other entry points update first. The old `check_index_freshness` heuristic goes;
  `--ast --json` reports the real status.

### Front ends

- CLI: global `--no-update`; every index-reading command updates first; "No index found"
  errors become builds; `rfx index status` stays read-only; one stderr line for a full
  run or build; stdout unchanged.
- MCP: `rfx mcp --no-update`; the update runs once in `handle_call_tool` (not for
  `index_project` / `check_index_status`); `update_ms` in `timings`.
- `rfx serve --no-update`; handlers in `spawn_blocking`.
- `index_project`, `POST /index`, `rfx watch`, interactive mode and `rfx ask` use the same
  config as `rfx index`; `rfx watch` passes its collected paths to `update_paths`.

## Steps (one commit each)

1. Prep, no output change: resettable engine handle, memo epoch, `Unknown`,
   `spawn_detached`, shared config everywhere, lock wait forever, persisted `--languages`.
2. Check fixes: ignore files and config plan a full run; tracked-but-ignored bug;
   compaction keeps rows.
3. `auto_update.rs` + `update_plan` + the fidelity harness (`tests/auto_update_fidelity.rs`).
4. Query engine update-and-retry.
5. CLI flag and per-command update.
6. MCP and `rfx serve`; fidelity harness through `rfx mcp` and the CLI.
7. `rfx watch` → `update_paths`.
8. Perf gates and golden battery.
9. **Stop and ask:** remove the MCP text that sends agents to `check_index_status` /
   `index_project`; fix the missing `can_trust_results` fields.
10. Docs.

## Gates

- Golden battery identical to 2.0.3 on four corpora with a fresh index.
- `latency_budget` green with `REFLEX_LATENCY_BUDGET=1`, sum within +5 %.
- Kubernetes, 1-file edit: `rfx query` in a new process < 0.5 s; MCP `search_code` < 150 ms.
- Commands that newly check: added time ≤ the check (~60 ms).
- Peak RSS of a query that updates ≤ an `update_paths` process (~70 MiB on Kubernetes).
- Fidelity harness: 100 % target, 95 % floor.
- Stop and ask on any other output change, a format change or a regression.

## As built (2026-09-30)

Commits on `feature/auto-update` (from `c2e0de2`): `9499b99` plan, `0a2c198` one config
and one symbol-pass launcher, `1bbbc69` check fixes (rule files, tracked-but-ignored,
compaction), `cca3e0d` `update_if_stale` + fidelity test + the switch-back fix, `c7b42d6`
wiring (engine, CLI, MCP, serve, watch), `324590f` docs, `df8cb1e` settle after a path
update, `4b73e65` golden harness fix.

What changed from the plan:
- **Settle instead of a second check.** The first build ran the full check again after
  every update (a second `git status`: ~100 ms on Kubernetes). A path update now
  compares only the updated paths with the index and, when they match, memoises "fresh"
  dated at the original check (`query::settle_update`) — the terms the 1 s memo already
  has. MCP edit-then-search 200–340 ms → 138–161 ms.
- **The switch-back bug** (TODO follow-up from the incremental work) had to be fixed: the
  fidelity sequence failed on "switch back" (the check compared with the current
  branch's row, not the last run's commit).
- **No clippy rule**: `disallowed-methods` would flag every test's `QueryEngine::new`.
  A test (`query_engines_are_built_by_the_known_front_ends_only`) fails instead on a
  `QueryEngine::new` in `src/` outside the known front ends.
- `rfx serve` `/index` and `rfx watch`: the watcher keeps fail-fast locking (it runs
  again on the next event); automatic updates, `index_project` and `POST /index` wait.
- UpdatePlan::Full carries the listed paths, so the loop guard does not block every
  later full run after one failure.

Measured (Kubernetes scratch clone, 16 cores, load 16–24; A/B against `c2e0de2`):

| scenario | c2e0de2 | auto-update |
| --- | --- | --- |
| `rfx query` (new process), nothing changed | 0.11–0.55 s | 0.08–0.11 s |
| `rfx query` after a 1-file edit | 0.09–0.14 s, **stale, 0 hits** | 0.16–0.18 s, fresh, found |
| MCP `search_code` after an edit (warm session) | 71–121 ms, **stale, 0 hits** | 138–161 ms, fresh, found (update 53–68 ms) |
| `rfx deps <file>` / `analyze --hotspots`, nothing changed | < 10 ms | 80–160 ms (the check) |
| peak RSS: query / MCP session | 68–85 / 79–93 MiB | 41–70 / 49–75 MiB |

`latency_budget` (2 runs each, alternating): green 4/4 with `REFLEX_LATENCY_BUDGET=1`; sum
of medians 85.9 → 87.3 ms (+1.6 %); MCP shapes −4 … +7 %.

Fidelity (`tests/auto_update.rs`): 15/15 steps answer as a fresh build.

Golden (`benches/incremental/golden.sh`): fresh capture = 2.0.3 reference; after scripted
updates 0/76 outputs differ on all four corpora (1 failing output per side, `q_two_char`,
as in the reference). The harness never indexed the edited trees before `4b73e65`, so the
2026-09-29 "identical after updates" claim was vacuous; the rerun covers c2e0de2 too.

Known limits:
- The 1 s verdict memo: in `rfx mcp` / `rfx serve`, an edit within 1 s of the previous
  check can be missed by the next call.
- Commands that did not check before (`deps`, `analyze`, `stats`, …) pay the check
  (~60–100 ms on Kubernetes; a stat of every file without git).
- Nested ignore files outside git are not compared (walk mode checks the root's only).

## Efficacy A/B after step 9 (2026-09-30)

REF-222 design (9 find-all tasks × 8 trials × 2 arms), Opus 5.5, Claude Code 2.1.284,
binary with the new MCP text (`cebddb9`). Raw trials: `benches/efficacy/results/`
(gitignored); the 2.0.3 run was moved to `results-2.0.3-opus/`.

| | 2.0.3 (2026-09-28) | auto-update (2026-09-30) |
| --- | --- | --- |
| total_tokens B/A (primary) | 1.663 [1.045, 1.685] | **1.675 [1.037, 1.688]** — unchanged, reflex_worse |
| arm-B trials that used Reflex | 37/72 | 43/72 |
| `check_index_status` / `index_project` calls | 0 / 0 (Opus never made them) | 0 / 0 |
| used-Reflex call sequence | ToolSearch > search_code (3 turns) | same |

- **Why no change on Opus:** Opus 5.5 never called the probe on these tasks, in 2.0.3 either
  (0 calls in 200 trials). The remaining extra turn is the ToolSearch that loads the
  deferred schemas (backlog §1 B).
- **Sonnet 5 (2026-09-30, same design, 9 runner processes in parallel — wall time is not
  comparable; raw trials in `results-autoupdate-sonnet5/`):**

  | | 2.0.3 | auto-update |
  | --- | --- | --- |
  | total_tokens B/A (primary) | 1.646 [1.063, 2.249] | **1.603 [1.267, 1.673]** |
  | `check_index_status` / `index_project` calls | 33 / 1 | **0 / 0** |
  | arm-B trials that used Reflex | 69/74 | 72/72 |
  | arm B median turns / tokens (find-all) | 4 / 148k | **3 / 101k (−32 %)** |
  | arm A median turns / tokens (find-all) | 2 / 72k | 2 / 66k |
  | total_cost_usd B/A | 2.072 | 1.754 |

  The status-check turn is gone; the one turn left over Grep is the ToolSearch (3 vs 2
  turns ≈ the 1.5–1.6× that remains). Precision and recall 1.000 in both arms.
- **Adoption is sensitive to the instructions' last paragraph.** Claude Code defers the
  tool schemas, so the agent decides between Grep and a ToolSearch from the instructions
  alone. Replacing "Only fall back to Grep/Glob after index_project has been called and the
  tool still fails" (9e30ff5) collapsed adoption to 2/72 (the first A/B, discarded). Pilots
  (3 tasks × 4 trials, trials that called Reflex): old text 8/12; "only fall back to
  Grep/Glob if a Reflex tool fails" 4/12; "if a Reflex tool fails, retry it once; only
  fall back to Grep/Glob after the retry also fails" 8/12 → kept (`cebddb9`).
- Accuracy: precision and recall 1.000 in both arms; 72/72 successes each.
