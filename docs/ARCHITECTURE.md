# Reflex Architecture

**How Reflex is built: the index, the query engine, and the surfaces on top.**

This document describes the current design for contributors. The code is the final
word; the on-disk byte layouts are specified in
[`.context/BINARY_FORMAT_RESEARCH.md`](../.context/BINARY_FORMAT_RESEARCH.md).

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Source Layout](#source-layout)
3. [The Index Cache (`.reflex/`)](#the-index-cache-reflex)
4. [File Coverage](#file-coverage)
5. [Indexing Pipeline](#indexing-pipeline)
6. [Background Symbol Pass](#background-symbol-pass)
7. [Query Pipeline](#query-pipeline)
8. [Freshness](#freshness)
9. [Dependency Extraction](#dependency-extraction)
10. [Surfaces: CLI, MCP, HTTP, Watcher](#surfaces-cli-mcp-http-watcher)
11. [Adding a New Language](#adding-a-new-language)
12. [Testing](#testing)
13. [Design Principles](#design-principles)

---

## System Overview

Reflex is a trigram-based full-text code search engine with optional symbol-aware
filtering. It aims for:

1. **Completeness**: every occurrence, not just definitions.
2. **Speed**: a trigram index narrows candidates; memory-mapped stores serve content.
3. **Determinism**: the same query gives the same results, sorted by path, then line.
4. **Honesty about staleness**: every response says whether the index matches the
   working tree.

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Surfaces                                                            │
│  rfx CLI (src/cli/)  ·  rfx mcp (src/mcp.rs)  ·  rfx serve (axum)    │
│  rfx watch (src/watcher.rs)                                          │
└──────────────┬───────────────────────────────────┬───────────────────┘
               │                                   │
      ┌────────▼─────────┐                ┌────────▼──────────┐
      │ Indexer          │   spawns       │ Query Engine      │
      │ src/indexer.rs   ├──────────┐     │ src/query/        │
      │ src/trigram_     │          │     │  OpenIndex handle │
      │   build.rs       │          │     │  freshness check  │
      └────────┬─────────┘          │     └────────┬──────────┘
               │           ┌────────▼─────────┐    │ cache hit / parse miss
               │           │ Symbol pass      │    │
               │           │ rfx index-       │    │
               │           │ symbols-internal │    │
               │           └────────┬─────────┘    │
      ┌────────▼────────────────────▼──────────────▼──────────┐
      │  .reflex/                                              │
      │  trigrams.bin (mmap) · content.bin (mmap) · meta.db    │
      │  config.toml · index.lock                              │
      └────────────────────────────────────────────────────────┘
```

- `rfx index` walks the tree, builds `trigrams.bin` and `content.bin`, and records file
  metadata, fingerprints and dependencies in `meta.db`.
- It then spawns `rfx index-symbols-internal`, a detached process that parses every
  file with a symbol parser and caches the symbols in `meta.db`.
- A query narrows candidates with the trigram index, verifies them against
  `content.bin`, and, for symbol queries, reads the symbol cache and parses only misses.

---

## Source Layout

| Module | Responsibility |
| --- | --- |
| `src/cli/` | `clap` command definitions (`mod.rs`) and one file per command group (`index.rs`, `query.rs`, `serve.rs`, `watch.rs`, `deps.rs`, …) |
| `src/indexer.rs` | `Indexer`, `PathPolicy` (what gets indexed), the batch loop |
| `src/trigram_build.rs` | Parallel, sharded construction of `trigrams.bin` |
| `src/trigram.rs` | `TrigramIndex`: V4 format, load, candidate search, trigram extraction |
| `src/content_store.rs` | `ContentWriter` / `ContentReader` for `content.bin` |
| `src/cache.rs` | `CacheManager`: `.reflex/` layout, `meta.db` schema, fingerprints, branches, stats |
| `src/atomic_write.rs` | tmp + fsync + rename writes; the `index.lock` advisory lock |
| `src/background_indexer.rs` | `BackgroundIndexer`: the background symbol pass |
| `src/symbol_cache.rs` | `SymbolCache`: zstd symbol blobs in `meta.db` |
| `src/query/` | `QueryEngine` (`mod.rs`), `QueryFilter` and pattern preparation (`filter.rs`), the shared `OpenIndex` handle (`open_index.rs`), zero-result explanations (`zero_hint.rs`), glob and path helpers (`result.rs`) |
| `src/regex_trigrams.rs` | Literal extraction from regex patterns |
| `src/parsers/` | Tree-sitter grammars, one module per language, `ParserFactory`, `LanguageQueries`, dependency extractors |
| `src/ast_query.rs` | Raw tree-sitter S-expression queries (`--ast`) |
| `src/dependency.rs` | `DependencyIndex`, `PathResolver`, `DependencyWriter`, graph analyses |
| `src/git.rs` | Branch, commit and working-tree change detection |
| `src/mcp.rs` | MCP server over stdio (JSON-RPC) |
| `src/watcher.rs` | `rfx watch`: reindex on file changes |
| `src/models.rs` | Shared types: `Language`, `SearchResult`, `IndexConfig`, `IndexStatus`, … |
| `src/context/`, `src/semantic/`, `src/interactive/`, `src/pulse/` | `rfx context`, `rfx ask`, the interactive TUI, and `rfx pulse`; built on the core above |

---

## The Index Cache (`.reflex/`)

| Path | Contents |
| --- | --- |
| `trigrams.bin` | Inverted index: trigram → sorted `(file_id, line_no)` postings. Memory-mapped. |
| `content.bin` | Every indexed file's contents, addressable by `file_id`. Memory-mapped. |
| `meta.db` | SQLite: file rows and fingerprints, branches, statistics, config, dependencies, exports, symbol cache. |
| `config.toml` | Project settings (`[index]`, `[search]`, `[performance]`). |
| `index.lock` | OS advisory lock held for a whole `rfx index` run. |
| `indexing.lock`, `indexing.status`, `indexing.cancel` | Ownership, progress and cancel request for the background symbol pass. |
| `trigram_temp/` | Partial trigram indices, only while an index run spans more than one batch. |

### Binary formats

Both binary files start with magic bytes and a format version. Full byte layouts are
in [`.context/BINARY_FORMAT_RESEARCH.md`](../.context/BINARY_FORMAT_RESEARCH.md).

**`trigrams.bin` (V4)** is a custom varint format, not a serialization library:

```
┌──────────────────────────────────────────────────────────────────────┐
│ Header (32 B)                                                        │
│   "RFTG" | version u32 = 4 | num_trigrams u64 | num_files u64 |      │
│   paths_offset u64                                                   │
├──────────────────────────────────────────────────────────────────────┤
│ Directory: num_trigrams × 16 B, sorted by trigram                    │
│   trigram u32 | data_offset u64 | compressed_size u32                │
├──────────────────────────────────────────────────────────────────────┤
│ Data: per trigram, a sequence of FILE BLOCKS                         │
│   varint(file_id delta) | varint(n_lines << 1 | enc) |               │
│   n_lines × varint(line delta, restarting at 0 per block)            │
├──────────────────────────────────────────────────────────────────────┤
│ Paths @ paths_offset: num_files × { varint(len), utf8 }              │
└──────────────────────────────────────────────────────────────────────┘
```

- One posting per distinct trigram per line. Postings carry no byte offsets.
- The directory is binary-searched in place in the memory map
  (`TrigramIndex::find_entry`); nothing is decoded at load except a bounds check.
- Posting lists are decoded on demand by a streaming cursor.

**`content.bin` (V2)**: a 32-byte header, the file contents concatenated in `file_id`
order, then a fixed-width entry table (28 bytes per file: offset, length, path
position, path length) and a path blob. An entry is read at
`index_offset + file_id * 28`, so opening the store is O(1). Non-UTF-8 text is stored
after lossy decoding.

**Symbol blobs (format v3)**: the `symbols` table in `meta.db` holds one row per
`(file_id, file_hash)`. Blobs of 256 bytes or more are zstd-compressed behind a 4-byte
magic; shorter ones are raw JSON (`symbol_cache::encode_symbols` / `decode_symbols`).
A stored `SYMBOL_FORMAT_VERSION` that differs from the binary's drops the cache.

### `meta.db` tables

| Table | Purpose |
| --- | --- |
| `files` | One row per indexed path: language, line count, and the freshness fingerprint (`size`, `mtime_ns`, blake3 `hash`, `dirty_at_index`) |
| `file_branches`, `branches` | Per-branch file hashes and the commit each branch was indexed at |
| `statistics` | Totals, `schema_hash`, `writer_version` |
| `config` | Key/value settings |
| `file_dependencies`, `file_exports` | Imports and barrel re-exports, with resolved file ids |
| `symbols` | The symbol cache (owned by `src/symbol_cache.rs`) |

### Versioning and ownership

- `build.rs` hashes the cache-critical sources (`cache.rs`, `content_store.rs`,
  `trigram.rs`, `indexer.rs`, `symbol_cache.rs`, `models.rs`, `dependency.rs`) into
  `CACHE_SCHEMA_HASH`. A cache with a different hash reports stale, and `rfx index`
  rebuilds it in full (`CacheManager::check_schema_hash`).
- Readers degrade, writers refuse: a cache stamped by a different released version is
  not rewritten unless forced (`CacheManager::assert_writable`).
- A `trigrams.bin` of an older version is served from an in-memory rebuild
  (`query::result::rebuild_trigram_index`) until `rfx index` runs. Any other load error
  is reported as `CacheCorrupted`.

---

## File Coverage

One classifier decides what is indexed, and the indexer, watcher, freshness check and
query engine all share it:

- `Language::from_path` (`src/models.rs`) looks at the name only. A lock-file name wins
  (`Lock`), then a generated name (`Generated`), then a code extension (the language),
  and everything else is `Text`.
- `PathPolicy::classify` (`src/indexer.rs`) applies the project config to that answer:
  include/exclude globs, `[index] mode`, `text_tier`, `languages`, `hidden`.

In the default `[index] mode = "tracked"` the rule is ripgrep's: every file that
`.gitignore` / `.ignore` / `.rgignore` / `[index] exclude` does not exclude, that is
not under a dot-directory (unless `hidden = true`), and that has no NUL byte anywhere
(`indexer::is_binary`). Files over `max_file_size` are skipped.

| Tier | `language` | Indexed | Searched by default | Symbols / deps |
| --- | --- | --- | --- | --- |
| code | `rust`, `python`, … | yes | yes | yes (if the grammar is supported) |
| text | `text` | yes | yes | no |
| lock | `lock` | yes | no (`include_locks`) | no |
| generated | `generated` | yes | no (`include_generated`) | no |

`languages` limits parsers only; it never removes the text tier. `text_tier = false`
does. Code without a working grammar (Swift) is indexed as searchable text, and symbol
queries skip it. `[index] mode = "allowlist"` restores a fixed list of text extensions
and does not index lock or generated files.

---

## Indexing Pipeline

`Indexer::index_with_callback` (`src/indexer.rs`):

```
1. Lock and clean up
   ├─ IndexLock on .reflex/index.lock (OS advisory lock, released if the process dies)
   ├─ remove_stale_tmp: delete *.tmp left by a crashed run
   └─ ask a running symbol pass to yield (indexing.cancel)

2. Discover files
   ├─ ignore::WalkBuilder with PathPolicy (gitignore rules, hidden, globs)
   └─ skip binaries and files over max_file_size

3. Fast path
   └─ same file set, every hash matches the stored hash, schema hash matches
      → refresh fingerprints and return without rewriting anything

4. Batch loop (plan_batches: ≤ REFLEX_INDEX_BATCH_FILES files, ≤ REFLEX_INDEX_BATCH_BYTES bytes)
   ├─ in the rayon pool, per file: stat, read, blake3 hash, lossy UTF-8 decode,
   │  Language::from_path, extract_trigram_run, dependency + export extraction
   └─ serially, in discovery order: assign file_id, TrigramIndexBuilder::add_file,
      ContentWriter::add_file; flush_batch builds the batch's partial index

5. meta.db
   ├─ one transaction for file rows, fingerprints and branch hashes
   └─ PathResolver (in memory) resolves imports; DependencyWriter writes all
      dependency and export rows in one transaction

6. Write stores
   ├─ TrigramIndexBuilder::write → trigrams.bin (tmp + fsync + rename)
   └─ ContentWriter::finalize_if_needed → content.bin (tmp + fsync + rename)

7. Stats, schema hash, then spawn the background symbol pass (src/cli/index.rs)
```

Points worth knowing:

- **Change detection is by content hash.** When anything changed, the binary stores
  are rebuilt from every file; the output does not depend on the previous index. The
  new / modified / unchanged counts come from comparing hashes with `file_branches`.
- **The trigram build is parallel and sharded** (`src/trigram_build.rs`). Extraction
  runs in the read pool and yields a sorted `TrigramRun` per file, with lines already
  deduplicated. Each batch is built per top-byte shard (256 shards) in parallel with no
  sort. Batches hand out increasing file ids, so partials merge by byte copy: only the
  first file delta of each later partial is rewritten. The output is byte-identical
  whatever the batch boundaries (`tests/index_batch_identity.rs`).
- **Writes are atomic.** Every store goes through `atomic_write` (write `<final>.tmp`,
  `sync_all`, `atomic_replace`). A reader sees the old complete file or the new one.
- **Thread pools.** `[performance] parallel_threads` (`0` = 80% of cores, up to 32)
  sizes the indexing pool and the query pool.
- `RUST_LOG=info rfx index` logs per-phase timings (`phase read+extract`,
  `phase files+branch transaction`, `phase dependencies+exports`, `phase trigram write`).

---

## Background Symbol Pass

`rfx index` spawns `rfx index-symbols-internal <root>` as a detached process unless one
is already running (`BackgroundIndexer::is_running`). `BackgroundIndexer::run`
(`src/background_indexer.rs`):

1. Takes `indexing.lock` (JSON holder record; liveness is checked by pid, with a
   heartbeat fallback) and writes progress to `indexing.status`.
2. Loads every cached `(file_id, hash)` in one `SELECT` and skips files already cached
   at their current hash. Files with no symbol parser (text tiers, Swift) are skipped
   and get no row.
3. Workers on `[performance] symbol_threads` (`0` = 50% of cores, up to 32;
   `REFLEX_SYMBOL_THREADS` overrides) read files from `content.bin`, parse them, and
   encode zstd blobs.
4. One writer thread owns the SQLite connection and commits in 1024-file batches. A
   failed batch is retried once and then counted as `write_failed`, not as a parse
   failure.
5. A new `rfx index` asks the pass to stop via `indexing.cancel`, and re-spawns it when
   it finishes.

`rfx index status` shows the phase and the `parsed` / `cached` / `write_failed` counts.

### Symbol extraction

`ParserFactory::parse` (`src/parsers/mod.rs`) dispatches to the language module's
`parse` function. Each module declares its per-kind tree-sitter queries once:

```rust
static SYMBOL_QUERIES: LanguageQueries = LanguageQueries::new(&[Q_FUNCTIONS, Q_TYPES /* … */]);
let table = SYMBOL_QUERIES.run(&language, &root, source)?;
for m in table.sub(0) { /* function matches */ }
```

`LanguageQueries` compiles all of them into one combined query per grammar on first
use. `run` walks the tree once and returns a `MatchTable` with matches bucketed by the
original sub-query, in the order a standalone run would have produced them. Add a new
symbol kind as another entry in `SYMBOL_QUERIES`, never as a separate `QueryCursor`.
Minified files are skipped. Previews are found from the symbol's byte offset
(`preview::extract_preview_from_byte`).

The background pass and query-time parsing use the same code, so a cached result and
a freshly parsed one are identical (`tests/symbol_equivalence.rs`).

---

## Query Pipeline

Entry point: `QueryEngine::search_with_metadata` (`src/query/mod.rs`). The CLI, MCP
server and HTTP server all call it, so they agree on results, warnings and hints.

```
1. Open           OpenIndex::get_or_open (reused across calls)
2. Prepare        prepare_literal_pattern: whole-identifier / contains / ignore_case;
                  bracketed literals become an escaped regex (reported in warnings)
3. Candidates     literal       → TrigramIndex::search_candidates
                  (?i) literal  → TrigramIndex::search_candidates_fold
                  regex         → regex_trigrams literals → candidate lines
                  no literal ≥3 → linear_scan_candidates (index_path = "scan")
                  keyword symbol query → every file of the language
4. Filter         language, tier (lock/generated/text), glob/exclude (gitignore rules)
5. Guard          broad-query check (short patterns / unscoped AST on large indexes)
6. Verify         verify_files_streaming: parallel, path order, early termination
7. Enrich         --symbols / --kind: enrich_with_symbols; --ast: enrich_with_ast
8. Post-filter    kind, exact match, dedup
9. Respond        freshness verdict, pagination, warnings, hint, timings
```

### The shared index handle

`OpenIndex` (`src/query/open_index.rs`) holds both memory maps, a lazily built
path → file-id map, a lazily opened `meta.db` connection, and the query thread pool.
Handles live in a process-wide registry keyed by cache directory, so `rfx mcp` and
`rfx serve` open the index once. A handle is reused while `content.bin` and
`trigrams.bin` keep the same identity (device, inode, size, mtime). An index run in the
same process calls `open_index::invalidate`; a run in another process is caught by the
identity check on the next lookup.

### Candidate narrowing

- Postings are intersected with a linear sorted merge: only the smallest list is
  materialised; the others are streamed. The merge stops early when the next list is
  much larger than the surviving set, and line verification finishes the job.
- A regex contributes the literals it guarantees (`regex_trigrams::extract_literals`).
  Only the lines those literals name are verified. A regex with no literal of 3 bytes
  or more scans every line.
- A `(?i)` literal is looked up under every ASCII case variant of each trigram. A
  non-ASCII literal under `(?i)` scans. A scan puts its reason in `warnings[]`.

### Verification and early termination

`verify_files_streaming` verifies candidate files in path order, in parallel, in
growing rounds. Lines within a file are ascending, so results come out in
`(path, line)` order. In list mode with a limit it stops once `offset + limit` results
exist. The page is identical to the same slice of a full run, but the total is then
not exact:

- `total_is_exact: false`, `total_count: null`, and `approx_total` estimated from a
  sample of `ESTIMATE_SAMPLE_FILES` files spread over the remaining candidates, at most
  `ESTIMATE_PER_FILE_LINES` lines each.
- When `ESTIMATE_FINISH_LINES` candidate lines or fewer remain, the search finishes
  instead and the total is exact.

Count mode (`--count`, `mode: "count"`, `count_occurrences`), `list_locations`,
`find_references`, and symbol and AST searches verify everything. Count mode returns
`(lines, files)` without building results.

### Symbol queries

`--symbols` and `--kind` find definitions, not call sites. `enrich_with_symbols`:

1. Groups candidates by file and drops files without a supported parser.
2. Skips files where every candidate line has the pattern only inside comments or
   strings.
3. Reads the cache with `SymbolCache::batch_get_with_kind_on` on the handle's
   connection. The branch comes from `.git/HEAD`, and a row is used only if its hash
   matches the file's current hash.
4. Parses the misses in parallel with `ParserFactory::parse` and writes them back in
   one transaction (`SymbolCache::batch_set_by_id_on`). A failed cache write never
   fails the query.
5. Filters by name and kind. A language keyword (`class`, `fn`; see
   `ParserFactory::get_keywords`) with `--symbols` means "list every symbol of that
   kind" and scans all files of the language.

### AST queries

`--ast` / `search_ast` runs a raw tree-sitter S-expression query
(`ast_query::execute_ast_query`) over candidate files. It parses every file it looks
at, so the broad-query guard requires a glob on larger trees. `--symbols` is the right
tool for most structural questions.

### Zero results

A zero result is explained rather than left bare (`query::zero_hint::explain_zero`,
`substring_hint_text`): the substring count for a whole-identifier miss, candidates
that were only lock or generated files (`excluded_by_default`), or a filter naming a
hidden or unindexed path (`excluded_reason`). The substring count is gathered during
the same search.

---

## Freshness

Every query response carries `status` and `can_trust_results`, and a stale index always
yields `can_trust_results: false`. Freshness is judged by file content, not by commit.

- Each `files` row stores a fingerprint: `size`, `mtime_ns`, blake3 `hash`, and
  `dirty_at_index` (the path was dirty in git when indexed).
- **`checked_by: "git"`**: candidate paths come from `git status --porcelain`, the
  paths dirty at index time, and a diff between the indexed and current commits. Each
  candidate is confirmed against its fingerprint, so a path git lists whose bytes the
  index already holds is not stale.
- **`checked_by: "walk"`**: outside a git repository (or when git cannot name the
  changed paths) the tree is walked with the same `PathPolicy`, and each file is
  stat'ed and hashed only when size or mtime differ.
- The verdict is memoised per workspace for `REFLEX_FRESHNESS_TTL_MS` (default 1 s; `0`
  disables) in the private `status_cache` module of `src/query/mod.rs`. Every index
  write in the process invalidates it, and `check_index_status` always bypasses it.
- Files outside every tier are never indexed, so a change to one never makes the index
  stale.

The MCP-facing shape (`files_modified`, `files_added`, `files_deleted`, `truncated`,
`action_required: "index_project"`) is documented in `CLAUDE.md` under
"Freshness contract".

---

## Dependency Extraction

During indexing, each language module that implements `DependencyExtractor`
(`src/parsers/mod.rs`) extracts **static** imports: string literals only. Dynamic
imports are dropped, so the graph is deterministic. TypeScript, JavaScript and Vue
resolve path aliases from every `tsconfig.json` (`parsers::tsconfig::parse_all_tsconfigs`,
parsed once per run), and barrel re-exports are recorded in `file_exports`. Dependency
queries are compiled once per process (`parsers::cached_query`).

Imports are classified as internal, external or stdlib. Internal imports are resolved
to file ids by `PathResolver` (`src/dependency.rs`), built once per run from the
`files` table: an exact path match first, then a binary-searched unique-suffix match.
`DependencyWriter` writes every row for the run in one transaction.

`DependencyIndex` answers the graph questions behind `rfx deps` and `rfx analyze`:
`get_dependencies`, `get_dependents`, `get_transitive_deps`,
`detect_circular_dependencies`, `find_hotspots`, `find_unused_files`, `find_islands`.
Text, lock and generated files never enter the graph.

---

## Surfaces: CLI, MCP, HTTP, Watcher

- **CLI** (`src/cli/`): `rfx index`, `rfx query`, `rfx deps`, `rfx analyze`, `rfx stats`,
  `rfx list-files`, `rfx watch`, `rfx serve`, `rfx mcp`, `rfx ask`, `rfx context`,
  `rfx pulse`, and the internal `rfx index-symbols-internal`. `rfx query --timing`
  prints per-phase timings, including `index_path` (`trigram` or `scan`).
- **MCP** (`src/mcp.rs`, `run_mcp_server`): JSON-RPC over stdio. Tool arguments are
  normalised (known aliases accepted with a warning, unknown keys rejected with a
  did-you-mean). `search_code` and `search_regex` return a columnar
  `{columns, rows}` shape (`to_columnar`; `REFLEX_MCP_COLUMNAR=0` restores
  `results[]`). A tool that fails with `CacheCorrupted` triggers one forced rebuild
  and one retry (`with_corruption_recovery`). Structural tools can be hidden with
  `[mcp] enable_structural_tools = false` in `~/.reflex/config.toml`. See
  [`mcp-tool-cheatsheet.md`](./mcp-tool-cheatsheet.md).
- **HTTP** (`src/cli/serve.rs`, axum): `GET /query`, `GET /stats`, `POST /index`,
  `GET /health`. Binds to `127.0.0.1` by default, with no authentication and permissive
  CORS. Do not expose it to a network.
- **Watcher** (`src/watcher.rs`): `notify` events, debounced (`WatchConfig`), then a
  normal `Indexer::index` run. The watcher uses the same `PathPolicy` to ignore events
  for files outside every tier.

`rfx mcp` and `rfx serve` are long-lived, so they keep the `OpenIndex` handle and the
freshness memo across calls. MCP, watcher and HTTP index runs fail fast if another
indexer holds `index.lock`; the CLI waits (`IndexConfig::lock_wait_secs`).

---

## Adding a New Language

1. **Grammar.** Add the `tree-sitter-<lang>` crate to `Cargo.toml`.
2. **Language.** Add a `Language` variant in `src/models.rs`, map its extensions in
   `Language::from_extension`, and mark it in `Language::is_supported`.
3. **Grammar loader.** Return the grammar from `ParserFactory::get_language_grammar`
   in `src/parsers/mod.rs`. This alone enables `--ast` for the language.
4. **Symbols.** Create `src/parsers/<lang>.rs` with a `parse(path, source)` function
   built on a `static SYMBOL_QUERIES: LanguageQueries`, and dispatch to it from
   `ParserFactory::parse`. Add keywords to `ParserFactory::get_keywords` if the
   language has "list all" keywords.
5. **Dependencies (optional).** Implement `DependencyExtractor` and call it from the
   indexer's per-file extraction.
6. **Tests.** Unit tests in the module, plus a fixture in `tests/corpus/` when the
   language needs end-to-end coverage.
7. **Docs.** Update the language lists in `README.md` and `CLAUDE.md`.

Changing a parser changes `CACHE_SCHEMA_HASH` only if it touches one of the hashed
files. If symbol output changes, bump `SYMBOL_FORMAT_VERSION` so old caches are dropped.

---

## Testing

- **Unit tests** live next to the code (`#[cfg(test)]` modules), including every parser.
- **Integration tests** in `tests/` cover whole behaviours, for example:
  - indexing: `index_batch_identity.rs`, `index_crash_safety.rs`, `index_deletion.rs`,
    `index_lock.rs`, `index_stats_ratio.rs`, `tracked_mode.rs`, `text_tier.rs`,
    `minified_files.rs`
  - querying: `query_early_termination.rs`, `regex_candidate_lines.rs`,
    `case_insensitive.rs`, `glob_anchoring.rs`, `cli_query_bracket.rs`,
    `zero_result_hints.rs`
  - symbols and dependencies: `symbol_equivalence.rs`, `symbol_lock.rs`,
    `dependency_equivalence.rs`
  - freshness: `mcp_freshness.rs`, `freshness_no_git.rs`, `git_worktree_status.rs`
  - MCP: `mcp_jsonrpc_compliance.rs`, `mcp_literal_search.rs`,
    `mcp_corruption_recovery.rs`
  - cache: `cache_version_guard.rs`, `sqlite_pragmas.rs`
  - `corpus_test.rs` runs against the fixture tree in `tests/corpus/`.
- **Latency budgets**: `tests/latency_budget.rs` measures the field-test query shapes
  in-process and through a real `rfx mcp` stdio round-trip over a deterministic
  synthetic corpus. Run it with
  `cargo test --release --test latency_budget -- --ignored --nocapture --test-threads=1`;
  CI asserts budgets with `REFLEX_LATENCY_BUDGET=1`.

See [`TESTING.md`](./TESTING.md) for how to run the suites.

---

## Design Principles

1. **Completeness over precision.** Trigrams produce candidates; verification against
   the stored content makes every result exact. A zero is only reported when it is
   true, and is explained when a filter or tier caused it.
2. **Determinism.** Results are sorted by path, then line. Index output does not depend
   on thread count or batch boundaries.
3. **Local-first.** Everything runs offline; the index lives in `.reflex/` and can be
   deleted and rebuilt at any time.
4. **One engine, many surfaces.** Pattern rewriting, freshness, hints and pagination
   live in the query engine, not in the CLI, MCP or HTTP layers.
5. **Crash safety.** Atomic replacement and the index lock mean a reader never sees a
   half-written store.

---

## References

- [Regular Expression Matching with a Trigram Index](https://swtch.com/~rsc/regexp/regexp4.html) (Russ Cox)
- [Zoekt](https://github.com/sourcegraph/zoekt): trigram-based code search
- [ripgrep](https://github.com/BurntSushi/ripgrep): the coverage and matching reference
- [tree-sitter](https://tree-sitter.github.io/): parsing for symbols, AST queries and imports
- [memmap2](https://github.com/RazrFalcon/memmap2-rs): memory-mapped I/O

---

## Design history

Earlier designs (rkyv `trigrams.bin`, `hashes.json`, query-time-only symbol parsing) are
in git: `git show fc8da6b:docs/ARCHITECTURE.md`.
