# CLAUDE.md

## Project Overview
**Reflex** is a local-first, full-text code search engine written in Rust. It's a fast, deterministic replacement for Sourcegraph Code Search, designed specifically for AI coding workflows and automation.

Reflex uses **trigram-based indexing** to enable instant full-text search across large codebases (10k+ files). Unlike symbol-only tools, Reflex finds **every occurrence** of patterns—function calls, variable usage, comments, and more—not just definitions. Results include file paths, line numbers, and surrounding context, with optional symbol-aware filtering.

---

## Core Principles
1. **Local-first**: Runs fully offline; all data stays on the developer's machine
2. **Complete coverage**: Finds every occurrence, not just symbol definitions
3. **Deterministic results**: Same query → same answer; no probabilistic ranking
4. **Instant access**: Trigram index + memory-mapping enables instant queries
5. **Agent-oriented**: Clean JSON output built for AI coding agents and automation
6. **Regex support**: Extract trigrams from patterns for fast regex search

---

## Architecture Overview

### Components
| Module | Description |
| --- | --- |
| **Trigram Indexer** | Extracts trigrams from all code files; builds inverted index (trigram → file locations) |
| **Content Store** | Stores full file contents (memory-mapped); enables context extraction around matches |
| **Query Engine** | Intersects trigram posting lists; verifies matches; returns line-by-line results with context |
| **Runtime Symbol Parser** | Uses Tree-sitter to parse candidate files that miss the symbol cache, at query time |
| **Background Symbol Indexer** | `rfx index-symbols-internal`, spawned by `rfx index`; parses every file with a grammar and fills the symbol cache |
| **Symbol Cache** | zstd-compressed symbol blobs in `meta.db` (`src/symbol_cache.rs`); symbol queries read it first |
| **CLI / API Layer** | Single binary for human and programmatic use (CLI and optional HTTP/MCP) |
| **Incremental updates** | `rfx index` and `Indexer::update_paths` publish a delta over the base instead of rebuilding (`src/snapshot.rs`, `src/indexer.rs`) |
| **Auto-update** | Every command that reads the index updates a stale one first (`src/auto_update.rs`); `--no-update` opts out |
| **Watcher (optional)** | Incrementally updates index on file changes |

### Index Cache Structure (`.reflex/`)
    .reflex/
      meta.db          # SQLite: files (stable ids, walk_seq, vendored) + freshness fingerprints, branches, stats, dependencies (+ package_members, import_edges view), exports, symbol cache
      manifest.json    # The commit point: which store files make up the index (generation, base, delta, recent, tombstones)
      content.<g>.bin  # Base content store (V2) for verification and context, generation g
      trigrams.<g>.bin # Base inverted index (V4): trigram → [file_id, line_no] posting lists
      trigrams.<g>.plan  # Id-free planning size per trigram (the candidate planner's order and stop rule)
      delta.<g>.*      # Delta tier: files added/modified since the base (content, trigrams, plan), tombstoned sizes (.tomb, .dtomb)
      recent.<g>.*     # Recent tier: the latest updates, rebuilt by each update, folded into a new delta past its limit
      content.bin, trigrams.bin  # Hard links to the base, only while the base alone is the index (for older binaries)
      resolver-configs.json  # Resolver config files the last walk found (`update_paths` parses them without walking)
      .index-run       # Marker whose mtime is the run's start (race threshold for recorded mtimes)
      config.toml      # Project settings (index, performance)
      index.lock       # Advisory lock held by `rfx index` for the whole run
      indexing.status  # Progress of the background symbol pass (`rfx index status`)
      pulse/           # Pulse docs-site cache (only after `rfx pulse`)

See `.context/BINARY_FORMAT_RESEARCH.md` for the on-disk formats and
`.context/INCREMENTAL_INDEX_RESEARCH.md` for the update design.

### Incremental updates
- `rfx index` stats every file and hashes only those whose (size, mtime) moved; unchanged
  files are not read. Nothing changed → only fingerprints, flags and branch rows are written.
- A change is published as a **delta**: added/modified files go to a small **recent**
  segment rebuilt by each update (folded into the **delta** tier past 256 files or 1/16 of
  the delta limit); superseded or deleted base/delta files are **tombstoned**. Past 2000
  files or 5 % of the corpus text, the delta is **merged** into a new base, taking
  unchanged files' text from the published stores (byte-identical to a fresh build).
- Rows keep their id across runs (`INSERT … ON CONFLICT(path)`), so symbols, other
  branches' rows and unchanged files' dependencies survive; `files.walk_seq` keeps
  walk order for every id-ordered output.
- Publish order: store files, then `manifest.json` (tmp + fsync + rename), then one
  `meta.db` transaction recording the same generation. Readers open the snapshot the
  manifest names (`IndexSnapshot`); a manifest ahead of `meta.db` (a crash) forces a
  full rebuild on the next run.
- `Indexer::update_paths(root, paths)` is the library entry point for a caller that
  knows what changed (1-file edit on Kubernetes: 51–53 ms at load 10; `rfx index` after
  the same edit 0.3 s, nothing changed 0.19 s; see PERFORMANCE_RESEARCH.md). It falls back to
  `Indexer::index` for ignore files, `.reflex/config.toml`, resolver configs, a branch
  change or the merge limit. Test knobs are Rust APIs (`set_merge_limits`,
  `set_recent_limits`, `set_abort_point`), never env vars.

### Auto-update
- Every command that reads the index (`rfx query`, `deps`, `analyze`, `stats`, `context`,
  `list-files`, `ask`, `snapshot`, `pulse`, interactive mode, `rfx mcp` tools, `rfx serve`)
  calls `auto_update::update_if_stale` first. The freshness check plans it
  (`query::update_plan`): nothing; `update_paths` on the listed paths; a full `index` run
  (lists truncated at 100, a rule-file edit, a format change). No index → built.
- Searches run the check alongside the search (`QueryEngine::with_update`): fresh costs
  nothing extra; stale → update → search again (at most twice). Other readers update
  before they run (`cli::update_before`, the MCP `handle_call_tool` chokepoint).
- `--no-update` (global flag; `rfx mcp --no-update`, `rfx serve --no-update`) = the old
  behaviour. `QueryEngine::new` has no update (library default); front ends build engines
  through `cli::engine` / `mcp::engine_in` (a test fails on any other `QueryEngine::new` in `src/`).
- Never fails the command: `Updated::Skipped(reason)` → answer from the current index,
  `stale`, reason in `warnings`. The CLI rebuilds another version's cache; servers skip it.
  Waits for `index.lock` forever (`LOCK_WAIT_FOREVER`). A failed plan over the same bytes is
  not retried.
- Rule files: an index run records the blake3 of `.reflex/config.toml`, the root ignore
  files and every dirty ignore file (`statistics.rule_files`); a change plans a full run.
- The verdict memo (1 s per process) still applies: in `rfx mcp`, an edit made within 1 s
  of the previous check can be missed by the next call. Plan and gates:
  `.context/AUTO_UPDATE_RESEARCH.md`.

### User Configuration (`~/.reflex/`)
    ~/.reflex/
      config.toml      # User settings (semantic query provider, API keys, model preferences)

---

## CLI Usage

**Indexing:**
```bash
rfx index                        # Build/update cache (every command also updates a stale index itself)
rfx index status                 # Check background symbol indexing
rfx index compact                # Manually compact cache
rfx watch                        # Auto-reindex on file changes
```

**Searching:**
```bash
# Full-text search (finds all occurrences)
rfx query "extract_symbols"

# Symbol definitions only (--symbols finds DEFINITIONS, not usages)
rfx query "extract_symbols" --symbols

# Filter by language, file patterns
rfx query "unwrap" --lang rust --glob "src/**/*.rs"

# Case-insensitive (rg -i); with --contains it is rg -i -F. Still uses the index.
rfx query "realmid" -i
rfx query "(?i)realm_?id" --regex

# JSON output for AI agents
rfx query "format!" --json

# Answer from the index as it is (no automatic update, no build)
rfx query "format!" --no-update

# Patterns that start with `-` (clap would read them as flags)
rfx query --pattern '-> Result<'      # or: rfx query -- '-> Result<'
```

**AST Queries** (⚠️ SLOW - use --symbols in 95% of cases):
```bash
rfx query "(function_item) @fn" --ast --lang rust --glob "src/**/*.rs"
```

**Dependency Analysis:**
```bash
rfx deps src/main.rs             # Show file dependencies
rfx deps src/config.rs --reverse # Show what depends on this file
rfx analyze --circular           # Find circular dependencies
rfx analyze --hotspots           # Find most-imported files
```

**Other:**
```bash
rfx serve --port 7878            # HTTP API server
rfx mcp                          # Start MCP server on stdio (for AI coding assistants)
```

---

## MCP Tools (for AI Coding Assistants)

When using Reflex as an MCP server (`rfx mcp`), the following tools are available as `mcp__reflex__<name>`.
Claude Code may register MCP tools as *deferred*: their schemas are not in context until loaded.
If Reflex tools appear in a deferred-tools list, load them first with
`ToolSearch("select:mcp__reflex__search_code,mcp__reflex__search_regex,mcp__reflex__find_references")`.
The required argument is always `pattern` (never `query`, `symbol`, `text`); the result cap is `limit`
(never `max_results`); the path filter is `file` (substring) or `glob` (array), never `path`. The
server accepts those wrong names as aliases and returns a `warnings` field; unknown keys are rejected with a
did-you-mean error, and numeric strings like `"40"` are coerced.

### Matching semantics

Literal search matches **whole identifiers** by default. `verify_csrf` does **not**
match `verify_csrf_form_field`. Three modes, each with a case-insensitive variant:

| Mode | How | Behaves like |
| --- | --- | --- |
| whole identifier | default | `grep -w` |
| substring | `contains: true` | `grep -F` |
| regular expression | `search_regex` | `grep -E` |
| any of the above, case-insensitive | `ignore_case: true` (`-i` / `--ignore-case` on the CLI) | `rg -i` (+ `contains` = `rg -i -F`) |

- `contains` is available on `search_code`, `list_locations` and `find_references` (not
  `search_regex`, which is already substring-based).
- `ignore_case` is available on those three **and** `search_regex` (where it prepends
  `(?i)`). A `(?i)` literal is looked up in the trigram index under every case
  variant, so it costs about what the case-sensitive query costs. A whole-identifier `ignore_case` search keeps
  whole-identifier semantics (`realmid` finds `RealmId`, not `realm_id`), reports
  `kind: text_match`, and produces no zero-result substring `hint`. Counts match
  ripgrep `-i` exactly, including the Unicode folds of `k` (KELVIN SIGN) and `s`
  (LONG S); a non-ASCII literal under `(?i)` still scans, and says so in `warnings`.
- A pattern containing brackets (`()`, `[]`, `<>`) is regex-escaped and run through the
  regex path automatically, with the rewrite reported in `warnings`. Whole-identifier
  matching wraps the pattern as `\b…\b`, which a pattern ending in `)` or `>` can
  never satisfy. The rewrite lives in the engine, so `rfx query`, `rfx serve` and MCP
  all apply it: the CLI prints
  `Warning:` on stderr and carries `warnings[]` / `hint` in `--json` output.
- A zero result carries a `hint` naming the substring count:
  `"0 whole-identifier matches; 89 substring matches — pass contains:true"`.

### Freshness contract

Every JSON-object MCP answer (search, count mode, `list_locations`, `find_references`,
`analyze`) and `check_index_status` carry `status` and `can_trust_results`. Array answers
(`get_dependencies` in every form, `search_ast`) do not. The index is updated before every call (see
Auto-update), so agents are told NOT to call `check_index_status` / `index_project`. Freshness is judged by
**file content, not by commit**: every indexed file has a recorded fingerprint (size,
mtime, blake3 hash), and the index is stale only when a file on disk differs from it —
edited, added or deleted, committed or not.

```json
{ "status": "stale", "can_trust_results": false,
  "reason": "Files changed since the index was built (1 modified, 1 added)",
  "action_required": "index_project",
  "files_modified": ["src/storage/mod.rs"],
  "files_added": ["src/storage/zz_probe.rs"],
  "files_deleted": [],
  "changed_count": 2 }
```

- **A stale index always yields `can_trust_results: false`.** No exception — including
  a zero-result search, which is exactly where an agent concludes "no callers".
- Edit → stale; `index_project` → fresh again, with no commit needed. Committing
  already-indexed content, or switching to a branch with the same tree, stays fresh
  (`details.indexed_commit` and `details.current_commit` may differ). Reverting a file
  after its edit was indexed IS stale.
- `details.checked_by` is `git` (candidates from `git status`, confirmed by fingerprint)
  or `walk` (every file is stat'ed: no git repository, or the git candidate query failed).
- The file lists are paths, capped at 100 per category; `truncated` says when.
- `action_required` names the MCP tool (`index_project`), never the CLI.
- With auto-update (default), a search that finds the index stale updates it and answers
  again, so `stale` appears only when the update could not run (`warnings` says why) or
  with `--no-update`. `check_index_status` never updates: it reports the truth.
- The verdict is memoised for 1 s per workspace (`REFLEX_FRESHNESS_TTL_MS`; `0`
  disables). `check_index_status` always bypasses the memo.

Ten tools (since 2026-09-30; was 17). Claude Code carries every listed schema on every
turn, so `tools/list` is kept small (~10.6 KB, guarded by `tests/mcp_tool_surface.rs`);
the shared matching / coverage / freshness rules live in `MCP_INSTRUCTIONS`, said once.

| Tool | Purpose |
|------|---------|
| `search_code` | Literal search with previews (default limit 200); `mode: "count"` → `{count, files}` |
| `search_regex` | Regex search (use for `->`, `::`, alternation, etc.) |
| `list_locations` | Path+line only — cheapest; `preview: true` adds the matching line (120 chars) |
| `find_references` | Definition + all usages in one atomic call (default limit: 200) |
| `search_ast` | Tree-sitter AST pattern matching (⚠️ slow — requires `glob`) |
| `get_dependencies` | What a file imports; `reverse: true` = what imports it; `depth: N` = transitive |
| `analyze` | Import graph: `kind` = summary / hotspots / circular / unused / islands (hidden with `[mcp] enable_structural_tools = false`) |
| `gather_context` | Project structure, frameworks, entry points |
| `index_project` | Force an index run (rarely needed: every tool updates first) |
| `check_index_status` | Report freshness without updating (rarely needed) |

Removed names still work, unlisted, with a deprecation warning (`LEGACY_TOOLS`):
`count_occurrences`, `get_dependents`, `get_transitive_deps`, `find_hotspots`,
`find_circular`, `find_unused`, `find_islands`, `analyze_summary`.

See [`docs/mcp-tool-cheatsheet.md`](./docs/mcp-tool-cheatsheet.md) for a decision tree by agent intent.

**Columnar result format (`search_code` / `search_regex`, list mode):** To cut token
cost from repeated JSON keys, these two tools return matches in a columnar shape
instead of an array of per-file objects (flat rows still repeat `path`/`language`):

```json
{
  "columns": ["path", "language", "start_line", "end_line", "preview", "kind", "symbol"],
  "rows": [
    ["src/mcp.rs", "rust", 955, 957, "fn make_tool_result", "Function", "make_tool_result"]
  ],
  "pagination": { "total": 1, "has_more": false, "total_is_exact": true },
  "status": "fresh", "total_count": 1, "returned_count": 1, "has_more": false,
  "total_is_exact": true
}
```

When the page filled before every candidate was verified, `total` and `total_count`
are **`null`** and `approx_total` carries an estimate:

```json
{ "pagination": { "total": null, "count": 1, "has_more": true, "total_is_exact": false,
                  "approx_total": 26580 },
  "total_count": null, "total_is_exact": false, "approx_total": 26580, "has_more": true }
```

Each `rows[i]` is one match; element `j` corresponds to `columns[j]`. The five base
columns (`path`, `language`, `start_line`, `end_line`, `preview`) are always present;
`kind`/`symbol`/`context_before`/`context_after`/`dependencies` are appended only when
a match carries them. Top-level metadata (`status`, `pagination`, `total_count`, …) is
unchanged. Set env `REFLEX_MCP_COLUMNAR=0` to restore the legacy `results[]` object
shape. `count` mode (`{count, pattern}`) and the other tools are unaffected.
`paths: true` returns `{status, can_trust_results, paths, total_files}` (plus
`has_more` when a `limit` cut the list) with no rows at all.

### Early termination and totals

A list-mode `search_code` / `search_regex` call verifies candidates in path order
and **stops once the page is full** (`offset + limit` results). The page is identical
to the same slice of a full run, but the total is not always exact:

| field | meaning |
| --- | --- |
| `total_is_exact: true` | `total_count` / `pagination.total` counts every match (count mode, no `limit`, symbol/AST searches, `find_references`) |
| `total_is_exact: false` | verification stopped early: `total_count` / `pagination.total` are **`null`** (never the verified-so-far number), `approx_total` is a **sampled estimate** (32 files spread over the remaining candidates, ≤16 lines each; typically within ±30%, omitted for a regex with no literal), and `has_more` is `true` |

When 32 files or 128 candidate lines or fewer remain after the page fills, the search
finishes instead and the total is exact. `mode: "count"`, `list_locations` and
`find_references` always verify everything. The CLI prints an
inexact total as `Found 10 results (~1234 total, estimated)` and points at `--count`.
Library readers: `PaginationInfo::exact_total()` (the total or `None`) and
`best_total()` (exact, else estimate, else page end; for thresholds only).

### Glob rules

`glob` / `exclude` (every MCP search tool), `rfx query --glob` / `--exclude` and
`[index] include.patterns` / `exclude.patterns` follow **gitignore / ripgrep rules**:

| pattern | matches |
| --- | --- |
| `src/**/*.rs` | `.rs` files under `src/` **at the index root only** (a `/` anchors) |
| `**/src/**/*.rs` | `.rs` files under any `src/` directory |
| `*.rs`, `Makefile` | at any depth (no `/` in the pattern) |
| `target/` | any `target/` directory and everything under it |
| `src/*.rs` | directly in `src/`; `*` never crosses `/` |

`./` and a leading `/` are dropped.

### Latency diagnostics

- `rfx query <pattern> --timing` prints per-phase timings (open, candidates, verify,
  status, group) to stderr; with `--json` they appear as a `timings` object
  (`update_us` when an automatic update ran).
- `REFLEX_MCP_TIMING=1` adds the same `timings` object to `search_code` /
  `search_regex` responses from `rfx mcp`.
- `timings.index_path` is `"trigram"` (candidates from the inverted index) or `"scan"`
  (every line verified: a pattern under 3 chars, a regex with no 3-byte literal such
  as `\w+_?id`, a non-ASCII literal under `(?i)`, or a keyword symbol query). A scan
  also puts its reason in `warnings[]`. `(?i)<literal>` is `"trigram"`.
- `rfx mcp` and `rfx serve` keep the index open across calls (memory maps, path map,
  thread pool) and reopen only when the index files change on disk or after
  `index_project`. The freshness verdict is memoised for `REFLEX_FRESHNESS_TTL_MS`.
- Query-time verification runs on a pool sized by `[performance] parallel_threads`
  (`0` = 80% of cores, up to 32).
- The background symbol pass (`rfx index-symbols-internal`, spawned by `rfx index`) runs
  on `[performance] symbol_threads` (`0` = 50% of cores, up to 32; `REFLEX_SYMBOL_THREADS`
  overrides). It is a streaming pipeline: workers parse files from `content.bin`, run
  ONE combined tree-sitter query per language per file (`parsers::LanguageQueries`), and
  hand zstd-compressed symbol blobs to a single writer thread that commits in 1024-file
  batches. `rfx index status` shows `parsed`/`cached`/`write_failed` counts and the phase.
  Files with no symbol parser (text tiers, Swift) are skipped, not stored as empty rows.
  Kubernetes (27k files, 15k with a parser): 44.9 s → 3.4 s on 8 threads (2.0.0, 2026-09-23).
- `rfx index` uses the same pool rule. It reads, hashes, extracts imports and trigram
  postings in the pool, builds each batch per trigram shard in parallel, and merges
  partials by byte copy (`src/trigram_build.rs`); output is byte-identical whatever
  the batch boundaries. Batches are bounded by files and bytes
  (`REFLEX_INDEX_BATCH_FILES`, default 5000; `REFLEX_INDEX_BATCH_BYTES`, default
  48 MiB). `RUST_LOG=info rfx index` prints per-phase timings. Kubernetes (27k files,
  245 MB) indexes from scratch in ~8 s on 16 cores (2.0.0, 2026-09-23; 1.7.2 took
  532 s, 95% of it in per-import SQLite lookups).
- `tests/latency_budget.rs` (`cargo test --release --test latency_budget -- --ignored
  --nocapture --test-threads=1`) measures the field-test query shapes in-process and
  through a real `rfx mcp` stdio round-trip; CI asserts budgets with
  `REFLEX_LATENCY_BUDGET=1`.

**MCP efficiency:**
- **A/B vs built-in Grep/Glob (2.0.3, 2026-09-28, `.context/EFFICACY-2.0.3.md`):** Reflex
  costs more tokens — 1.66× on find-all-usages (Opus 5.5 and Sonnet 5), 1.26× on
  comprehension tasks (Opus 5.5) — with equal or better accuracy. The cost is extra
  round-trips (ToolSearch for deferred schemas, `check_index_status`), not payload size.
  Do not claim token savings over grep yet. **Goal:** parity or better on plain Grep-like
  searches — do not route them to Grep; see the backlog in `.context/TODO.md`.
- **Re-run after auto-update (2026-09-30):** Opus 5.5 1.675× (unchanged: it never called
  `check_index_status`, 0 in 200 trials); Sonnet 5 1.603× (was 1.646×): its 33 status
  calls fell to 0, arm-B median 4 → 3 turns and 148k → 101k tokens. What is left, on both
  models, is the ToolSearch turn for the deferred schemas (backlog §1 B).
- **`alwaysLoad` (eager schemas, 2026-09-30):** removes the ToolSearch turn (turns = Grep's)
  and cuts cost: Sonnet 1.77× → 1.20×, Opus 1.32× → 0.98× Grep. Tokens: Sonnet 1.63× →
  1.53×, Opus 1.68× → 1.82× (every turn carries the 44 KB `tools/list`). The docs'
  example configs set it. Schema size is now the whole gap.
- **Long sessions (`benches/efficacy/session_bench.py`, 2026-09-30):** per query Reflex
  costs what Grep costs; the gap is the ~16K-token schema prefix per turn, so it shrinks
  with session length. Cost vs Grep with `alwaysLoad`: 1.18–1.36× (12 questions), **1.10×**
  (50 questions, Sonnet; cheaper than Grep on tokio). With deferred schemas agents mostly
  skip Reflex in long sessions (22/30 Sonnet sessions used Grep only). The instructions' last paragraph
  decides adoption (the schemas are deferred): keep a Reflex-first fallback rule there —
  dropping it took adoption from 37/72 to 2/72 (`.context/AUTO_UPDATE_RESEARCH.md`).
- **Slim tool surface (10 tools, 10.6 KB, 2026-09-30):** cost vs Grep with `alwaysLoad` —
  Sonnet 1.05× (12 questions), 1.11× (50); Opus 1.15× (12; it drifts back to grep after
  one preview-less `list_locations`), **0.89×** (50). Keep "where does X occur → list_locations"
  in the instructions: without it agents return 1.3–1.9K-char previews per lookup.
- **structuredContent: evaluated and rejected.** MCP `outputSchema`/`structuredContent` was
  built and removed: Claude Code transmits *both* `content[text]` and `structuredContent`,
  so it saved nothing. Do not re-attempt unless using a client that honors `outputSchema`
  and drops the text block.

---

## AST Pattern Matching

⚠️ **PERFORMANCE WARNING**: AST queries are **SLOW** — they parse every `--lang` file that `--glob` selects, with no trigram narrowing. **Use `--symbols` instead in almost every case**; it reads the symbol cache.

**When to use** (RARE):
- Need to match code structure, not just text (e.g., "all async functions with try/catch blocks")
- `--symbols` search is insufficient
- Very specific structural pattern that cannot be expressed as text

**Supported languages**: All tree-sitter languages (Rust, Python, Go, Java, C, C++, C#, PHP, Ruby, Kotlin, Zig, TypeScript, JavaScript)

**Architecture**: Centralized grammar loader in `src/parsers/mod.rs` - adding a new language automatically enables AST queries.

**Example**:
```bash
rfx query "(function_item) @fn" --ast --lang rust --glob "src/**/*.rs"
```

**ALWAYS use `--glob` with AST queries.**

---

## Supported Languages

**15 languages** with full symbol extraction support (functions, classes, variables, etc.):

- **Systems**: Rust, C, C++, Zig
- **Backend**: Python, Go, Java, C#, PHP, Ruby, Kotlin
- **Frontend**: TypeScript, JavaScript, Vue, Svelte
- **Swift**: Temporarily disabled — `rfx query --lang swift` emits a warning; full-text search still works but symbol queries return no results (tree-sitter-swift 0.7.x grammar incompatibility)

**Symbol extraction**: Functions, classes, methods, variables (global + local), interfaces, traits, enums, attributes/annotations, and more.

**Special features**:
- **React/JSX**: Components, hooks, TypeScript support
- **Attributes/Annotations**: `--kind Attribute` finds annotation definitions (Rust proc macros, Java @interface, Kotlin annotation class, PHP #[Attribute], C# Attribute classes)


### Plain-Text Tier (every non-binary file)

Reflex indexes non-code files, because **agents do not partition searches by file
type**. A config key lives in the YAML, the Rust struct *and* the spec paragraph;
returning only the struct and a confident `0` for the rest is a wrong answer.

**Coverage rule (`[index] mode = "tracked"`, the default)**: ripgrep's defaults —
every non-binary file (no NUL byte anywhere) that is not excluded by `.gitignore` /
`.ignore` / `.rgignore` / `[index] exclude` and not under a dot-directory (`.github/`,
`.githooks/`, `.cargo/`; `[index] hidden = true` walks them). So `OWNERS`, `SECURITY_CONTACTS`,
`foo.po`, `a.css`, `data.jsonl`, `Makefile`, `Dockerfile` and every other extensionless
or unlisted name are `language: "text"`. Code is still classified by extension
(`.mjs` / `.cjs` are JavaScript, with symbols). Non-UTF-8 text (Latin-1 `.po`) is
decoded lossily, not dropped. `Language::from_path` plus `PathPolicy::classify` is the
one classifier the indexer, watcher, freshness check and query engine share.

**Two more tiers, indexed but excluded from every search unless asked for:**

| tier | judged by | `lang` | widen with |
| --- | --- | --- | --- |
| `lock` | name: `Cargo.lock`, `package-lock.json`, `*-lock.json`, `*.lock`, `yarn.lock`, `pnpm-lock.yaml`, `go.sum`, `flake.lock`, `uv.lock`, `bun.lock` | `"lock"` | `include_locks: true` / `--include-locks` |
| `generated` | name: `*.pb.go`, `*_generated.*`, `*.generated.*`, `*.min.js`, `*.min.css`, `*.map` | `"generated"` | `include_generated: true` / `--include-generated` |

A zero result whose candidates were only such files says so: `excluded_by_default: N`
plus a `hint`. "Which lockfile pins serde 1.0.190?" is a real query, and a confident
zero with no way in is the failure this tier exists to fix. A `@generated` content
marker is **not** read (the query engine derives language from the path).

**Trigram-indexed only.** No tree-sitter, no symbol extraction, no import extraction. So:

| Tool / flag | Text / lock / generated tiers |
| --- | --- |
| `search_code`, `search_regex`, `list_locations` | text **included by default**; lock and generated on request |
| `--symbols`, `--kind`, `--ast`, `search_ast` | excluded (there is no grammar) |
| `find_references`, `get_dependencies`, `analyze` | excluded (a mention in a changelog is not a call site) |

- **Select the text tier**: `--lang text` (aliases `txt`, `plaintext`, `plain`).
- **Exclude it**: `exclude_text: true` on the four full-text MCP tools.
- **Turn it off**: `[index] text_tier = false` in `.reflex/config.toml`.
- **Allowlist mode**: `[index] mode = "allowlist"` indexes only code by
  extension plus the fixed list `md mdx txt yaml yml toml json proto html htm sh bash
  ini cfg sql graphql bru` and the names `Makefile`, `Dockerfile`, `Justfile`; lock and
  generated files are not indexed. For trees where the long tail of data files is not
  worth the index size.
- **Hidden files**: dot-directories and dotfiles are skipped, like ripgrep without
  `--hidden`. `[index] hidden = true` walks them (`.githooks/pre-commit`); `.git/` and
  `.reflex/` are never walked.
- **Not subject to `[index] languages`.** That option means "which parsers do I care
  about"; a user with `languages = ["rust"]` keeps their documentation searchable.
  `text_tier = false` is the way to turn the tier off.
- `rfx index` prints `Text: N files, Lock: N, Generated: N` and the count of binary
  files it skipped.

**Note**: files outside every tier (binaries, files over `max_file_size`, hidden paths
unless `hidden = true`) are not indexed, and a change to one never makes the index stale.

---

## Dependency/Import Extraction

**Experimental feature** for analyzing import statements to understand codebase structure.

### Key Design Principle: Static-Only Imports

**IMPORTANT**: Reflex **intentionally** extracts **only static imports** (string literals) and filters out dynamic imports.

**Why?**
- **Deterministic**: Same codebase → same dependency graph
- **Fast**: No runtime code evaluation required
- **Accurate**: Avoids false positives from computed imports

**What gets captured**:
```python
import os                    # ✅ Static import
from json import loads       # ✅ Static import
```

**What gets filtered**:
```python
importlib.import_module(var) # ❌ Dynamic import (variable)
import(f"./templates/{name}") # ❌ Template literal
```

### Resolution

An internal import reaches files in one of two ways:
- **One file** (`file_dependencies.resolved_file_id`): TS/JS, Rust, Python, PHP, Ruby,
  C/C++, Zig, Vue/Svelte.
- **A package key** (`resolved_package`, `resolved_member`): Go `go:<dir>`, Java/Kotlin
  `jvm:<package>` + class or top-level name, C# `cs:<namespace>`. Each file records the
  packages it belongs to in `package_members` when it is extracted.

Every graph reader uses the `import_edges` view, which expands package keys to files.
Do not read `resolved_file_id` directly. `analyze` and `get_dependencies` warn when a
language resolves under 50 % of 100+ internal imports. Numbers and design:
`.context/DEPENDENCY_RESOLUTION_RESEARCH.md`.

### Vendored code

Vendored files (`files.vendored`, rules in `src/vendor.rs`) are indexed and searchable,
but are not graph nodes: `import_edges` drops every edge that touches one, and graph
node queries use `CODE_FILES`, which leaves them out. A repo that commits `vendor/`
gets the graph of one that gitignores it. Rules, in order:
1. `[index.vendored] patterns` (gitignore rules; `!dir/` = project code)
2. toolchain markers: Go `vendor/modules.txt`, `vendor/composer/installed.json`,
   `.cargo-checksum.json`, `pyvenv.cfg`, `build.zig.zon` path dependencies
3. `node_modules`, `bower_components`, `site-packages`, `dist-packages`, installed gems
4. per-language names (`vendor_dir_names`); none for Go, PHP, Rust and Ruby

Markers are resolver configs, so adding or removing one re-flags every file.

### Classification

Imports are classified as:
1. **Internal**: Project code (relative paths, tsconfig aliases)
2. **External**: Third-party packages
3. **Stdlib**: Standard library

### Commands

```bash
rfx deps src/main.rs            # Show file dependencies
rfx deps src/config.rs --reverse # What depends on this file
rfx analyze --circular          # Find circular dependencies
rfx analyze --hotspots          # Find most-imported files
rfx analyze --unused            # Find orphaned files
```

### Use Cases

Designed for **codebase structure analysis**:
- Understanding module boundaries
- Identifying coupling hotspots
- Finding circular dependencies
- Architecture review

**Not suitable for**: Build systems, package management, runtime resolution

---

## Tech Stack
- **Language**: Rust (Edition 2024)
- **Core Algorithm**: Trigram-based inverted index (inspired by Zoekt/Google Code Search)
- **Crates**:
  - **Indexing**: Custom trigram extraction, `memmap2` (zero-copy I/O)
  - **Parsing**: `tree-sitter` + language grammars (background symbol pass, cache misses at query time, imports)
  - **Storage**: `rusqlite` (metadata, symbol cache), custom binary formats (trigrams + content), `zstd` (symbol blobs)
  - **Incremental**: `blake3` (content hashing), `ignore` (gitignore support)
  - **Performance**: `rayon` (parallel indexing), memory-mapped I/O
  - **CLI**: `clap` (argument parsing), `serde_json` (JSON output)

---

## Development Workflow

### Build
    cargo build --release

### Test
    cargo test

### Refresh Index
    rfx index

### Debug Queries
    RUST_LOG=debug rfx query "fn main"

---

## Symbol Detection Architecture

Symbol queries combine the trigram index with tree-sitter, and a persistent cache sits
between them.

1. **`rfx index`**: extracts trigrams (and imports, with tree-sitter) from the files
   that changed, publishes the stores (see Incremental updates), then spawns the
   background symbol pass.
2. **Background symbol pass** (`rfx index-symbols-internal`): parses every file that has
   a grammar and no cached symbols for its current hash (after a 1-file edit, that one
   file), runs one combined tree-sitter query per language per file
   (`parsers::LanguageQueries`), and stores zstd symbol blobs in `meta.db`
   (`src/symbol_cache.rs`). Files with no grammar (text tiers, Swift) are skipped.
3. **Query time** (`--symbols`, `--kind`, `find_references`):
   1. Trigram search narrows the tree to candidate files.
   2. Candidates are read from the symbol cache; a cache miss (pass still running, or a
      file changed) is parsed on demand with the same extractors.
   3. Results are filtered to symbol definitions.

Full-text queries never touch tree-sitter. New symbol kinds go into the language's
`SYMBOL_QUERIES`, never a separate per-kind `QueryCursor`.

---

## Design Notes
- **Trigram Algorithm**: Extracts 3-character substrings; builds inverted index for O(1) lookups
- **Symbol detection**: symbol cache first, tree-sitter on cache misses among trigram candidates (see above)
- **Change detection by stat, then content**: files whose (size, mtime) match their row are not read; the rest are hashed (`blake3`). A change publishes a delta (see Incremental updates); a schema or extraction-code change forces one full rebuild
- **Memory-mapped I/O**: Zero-copy access to the stores the manifest names
- **Regex support**: Extracts guaranteed trigrams from patterns; falls back to full scan if needed
- **Deterministic**: Same query always returns same results (sorted by file:line)
- **Respects .gitignore**: Uses the `ignore` crate to skip gitignored files (untracked, non-ignored files are indexed)
- **Programmatic output**: File-grouped results with spans and previews:
  ```json
  {
    "path": "src/parsers/rust.rs",
    "matches": [
      {
        "kind": "Function",
        "symbol": "extract_symbols",
        "span": { "start_line": 67, "end_line": 89 },
        "preview": "pub fn extract_symbols(source: &str, ...) -> Vec<SearchResult> {"
      }
    ]
  }
  ```
- **`SymbolRef` — stable symbol reference type**: Symbol data serializes as a named struct, not a positional tuple. Fields are additive-safe (new optional fields can be appended without shifting existing positions or requiring a version bump):
  ```json
  { "name": "extract_symbols", "kind": "Function", "span": { "start_line": 67, "end_line": 89 } }
  ```
- **Language field forward-compatibility**: The `language` field serializes as a **lowercase** enum string (`"rust"`, `"python"`, `"typescript"`, …) — the enum carries `#[serde(rename_all = "lowercase")]`. Docs, config and template files serialize as `"text"` (see the plain-text tier below); an unrecognized file type serializes as `"unknown"`. Callers **must not** exhaustively match on this field without a fallback — treat `"unknown"` as the forward-compatible sentinel for any language Reflex does not yet recognise. New languages may be added in minor releases.

---

## Security / Threat Model

### `rfx serve` (HTTP API server)

- **Bind address**: Defaults to `127.0.0.1:7878` — loopback only. The server is intentionally not authenticated; it is designed for local, single-user use.
- **Risk if exposed**: Passing `--host 0.0.0.0` (or any non-loopback address) exposes the index and all search results to the network without any authentication or rate-limiting. Do not do this on shared or internet-facing machines.
- **No auth by design**: There are no API keys, tokens, or access controls on `rfx serve`. This is a deliberate local-tool trade-off, not a bug. If you need network-accessible search, add a reverse proxy with authentication in front of it.
- **CORS**: The server enables permissive CORS headers (`Any` origin) for localhost browser tooling — another reason not to expose it externally.

### Summary

| Concern | Default behaviour | Risk if changed |
|---------|-------------------|-----------------|
| Bind address | `127.0.0.1` (loopback) | Network exposure with no auth |
| Authentication | None | Unauthenticated read access to codebase index |
| Port | `7878` | Firewall / port conflict only |

---

## Repository Conventions
- Source: `src/`
- Core library: `src/lib.rs`
- CLI entrypoint: `src/main.rs`
- Tests: `tests/`
- Local cache/config: `.reflex/` (added to `.gitignore`)
- Context/planning: `.context/` (tracked in git)
- Project config: `REFLEX.md` (optional, workspace root)

---

## Configuration Files

Reflex uses two types of configuration files:

### 1. Project Configuration (`.reflex/config.toml`)
Located in the workspace's `.reflex/` directory.

**Purpose**: Project-specific settings for indexing and search behavior.

**Sections**:
- `[index]`: Languages, the plain-text tier, coverage mode, hidden files, file size limits, symlink handling, include/exclude patterns
- `[performance]`: Thread counts for querying, indexing and the background symbol pass

**Example**:
```toml
[index]
languages = []  # Empty = all supported languages (does not affect the text tier)
text_tier = true  # Also index docs, config and every other non-binary file
mode = "tracked"  # every non-binary, non-gitignored file; "allowlist" = code + fixed docs/config list
hidden = false    # true also walks dot-directories (.githooks/), never .git/ or .reflex/
max_file_size = 10485760  # 10 MB
follow_symlinks = false
# gitignore rules: a pattern with `/` is anchored at the root, a bare name matches anywhere.
# include.patterns = ["src/**/*.rs", "docs/**"]   # whitelist (directories are still walked)
# exclude.patterns = ["vendor/**", "*.generated.rs"]
# vendored.patterns = ["libs/acme/", "!third_party/ours/"]  # searchable, not in the import graph

[performance]
parallel_threads = 0  # 0 = auto (80% of cores, max 32)
symbol_threads = 0  # background symbol pass (rfx index-symbols-internal); 0 = auto (50% of cores, max 32)
```

**Git tracking**: Should be committed for team-wide consistency.

### 2. User Configuration (`~/.reflex/config.toml`)
Located in your home directory's `~/.reflex/` folder.

**Purpose**: User-specific settings for AI providers and credentials.

**Sections**:
- `[semantic]`: Preferred AI provider for `rfx ask`
- `[credentials]`: API keys and model preferences

**Example**:
```toml
[semantic]
provider = "openai"  # Options: openai, anthropic, openrouter, openai-compatible

[credentials]
openai_api_key = "sk-..."
openai_model = "gpt-4o-mini"
anthropic_api_key = "sk-ant-..."

# OpenAI-compatible endpoints (LMStudio, Ollama, llama.cpp, vLLM, litellm, …)
openai_compatible_base_url = "http://localhost:1234/v1"
openai_compatible_model = "qwen2.5-coder-32b-instruct"
# openai_compatible_api_key = "sk-..."   # optional; omit for keyless local servers
```

**Env vars** (CI / headless usage):
- `REFLEX_PROVIDER`, `REFLEX_MODEL`, `REFLEX_AI_API_KEY` (works with any provider)
- Provider-specific: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`, `OPENAI_COMPATIBLE_BASE_URL`

**Git tracking**: Should NOT be committed (contains API keys).

**Configuration wizard**: Run `rfx llm config` to set up interactively.

### 3. Project Context (REFLEX.md)
**Optional**: Create a `REFLEX.md` file at workspace root to customize `rfx ask` behavior with project-specific context.

**Use cases**:
- Document unconventional directory structures
- Provide project-specific search patterns
- Add domain-specific terminology
- Explain monorepo organization

**Location**: Place at same level as `.reflex/` directory.

**Git tracking**: Recommended to commit for team-wide consistency.

---

## Context Management & AI Workflow

### `.context/` Directory Structure

The `.context/` directory contains planning documents, research notes, and decision logs to maintain context across development sessions. **All AI assistants working on Reflex must actively use and update these files.**

#### Required Files

**`.context/TODO.md`** - Primary task tracking and implementation roadmap
- **MUST be consulted** at the start of every development session
- **MUST be updated** when:
  - Starting work on a task (mark as `in_progress`)
  - Completing a task (mark as `completed`)
  - Discovering new tasks or requirements
  - Making architectural decisions that affect the roadmap
  - Changing priorities or timelines
- Contains only live work: in-progress projects, open bugs, current policy (decisions
  still in force), open follow-ups and the backlog. Finished work moves to CHANGELOG.md
  or is deleted (git keeps history).

#### Research Files

Create `{TOPIC}_RESEARCH.md` files to cache findings from a focused investigation.
See `.context/README.md` for the files that exist today (formats, trigram design,
performance rounds, symbol detection). Rules:
- Date every measurement and name the Reflex version it was taken on.
- Include what was tried and why it did not work (avoid repeated dead ends).
- When the code changes what a file describes, update it or add a dated banner.

### AI Assistant Workflow

When working on Reflex, AI assistants should:

1. **Start Every Session:**
   - Read `CLAUDE.md` for project overview
   - Read `.context/TODO.md` to understand current state
   - Identify which tasks are blocked, in progress, or ready to start

2. **During Development:**
   - Update `.context/TODO.md` task statuses in real-time
   - Create/update RESEARCH.md files when conducting investigations
   - Document decisions and rationale inline
   - Add new tasks as they're discovered

3. **Before Ending Session:**
   - Ensure all task statuses are accurate; move finished work out of TODO.md (CHANGELOG.md or delete)
   - Document any blocking issues or open questions
   - Update implementation notes if approach changed
   - Commit research findings to appropriate RESEARCH.md files

4. **When Conducting Research:**
   - Create focused RESEARCH.md files rather than losing findings
   - Include code examples, links, and specific version numbers
   - Note what was tried and why it didn't work (avoid repeated dead ends)
   - Cross-reference related TODO.md tasks

5. **Decision Documentation:**
   - Decisions still in force go in `.context/TODO.md` under "Current policy"
   - Technical deep-dives go in specific RESEARCH.md files
   - Quick notes and TODOs stay in source code comments

### Example: Starting a New Language Parser

```bash
# 1. Check TODO.md for the task
# 2. Create research file
touch .context/RUST_PARSER_RESEARCH.md

# 3. Document investigation
# - Examine tree-sitter-rust grammar
# - List all node types for symbols
# - Create example AST traversal code
# - Note edge cases (macros, proc macros, etc.)

# 4. Update TODO.md
# - Mark parser task as in_progress
# - Add any new subtasks discovered
# - Document key decisions

# 5. Implement based on research
# 6. Remove the finished task from TODO.md; record user-visible changes in CHANGELOG.md
# 7. Reference RESEARCH.md in code comments
```

### Context Preservation Goals

The `.context/` directory enables:
- **Session continuity:** Pick up where previous work left off
- **Decision tracking:** Understand why choices were made
- **Avoiding rework:** Don't re-research solved problems
- **Onboarding:** New contributors understand the project state
- **AI handoff:** Different AI assistants can collaborate effectively

---

## Project Philosophy
Reflex favors local autonomy, speed, and clarity.

- Fast enough to call multiple times per agent step.
- Deterministic for repeatable reasoning.
- Simple to rebuild: delete `.reflex/` and re-index at any time.

> "Understand your code the way your compiler does — instantly."

---

## Release Management

See [RELEASE.md](./RELEASE.md) for full release process, semantic versioning, and changelog format.

---
