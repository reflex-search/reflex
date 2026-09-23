## [Unreleased] - 1.8.0

### ⚠️ Breaking

- `trigrams.bin` V4 — **re-index required**. A V3 index is served from an in-memory rebuild (slow) until `rfx index` runs; `rfx index` detects the schema change and rebuilds in full. Postings are now one per distinct trigram per line, grouped into per-file blocks, with no byte offsets: 3.9x → 0.9x of corpus size on the synthetic latency corpus (127 MB → 30 MB), 1.4x on the Reflex repo. `TrigramIndex::load` is O(files) — the directory is binary-searched in the mmap instead of being decoded and re-sorted. `FileLocation` (library API) drops `byte_offset`; `FileLocation::new` takes `(file_id, line_no)`.

- A list-mode `search_code` / `search_regex` (and `rfx query` with a limit) stops verifying once the page is full. The page is unchanged, but **`pagination.total` / `total_count` is `null` whenever the new `total_is_exact` is `false`** (`PaginationInfo.total` is `Option<usize>` in the library API). It was briefly the number verified before the page filled, which the field test showed is not a total (`851` for a term with 18,752 matches) and which every existing reader took as one. `approx_total` is a **sampled estimate**: after the page fills, 32 files spread over the remaining candidates (at most 16 candidate lines each) are verified and the hit rate scaled over the rest; when 32 files or 128 lines or fewer remain, the search simply finishes and the total is exact. Synthetic corpus: `config` estimated 52010 (exact 52010), `ident_7` 26580 (exact 26969); expect roughly ±30% when hits cluster in a few files. The field is omitted for a regex with no literal (every line a candidate). `has_more` is `true` whenever the total is not exact. `mode: "count"`, `count_occurrences`, `list_locations`, `find_references` (`total_references`), symbol/AST searches and no-limit searches keep exact totals. The CLI prints `Found 10 results (~1234 total, estimated)` and points at `--count`. `PaginationInfo::exact_total()` / `best_total()` are the two honest readers.

- Glob filters follow **gitignore / ripgrep rules**: `glob` and `exclude` on every MCP search tool, `rfx query --glob` / `--exclude`, `rfx list-files --glob`, and `[index] include.patterns` / `exclude.patterns`. A pattern containing `/` is anchored at the index root, so `src/**/*.rs` no longer matches `vendor/src/x.rs` (field test: 7131 hits against ripgrep's 6770); write `**/src/**/*.rs` for the old behaviour. A bare name (`*.rs`, `Makefile`) still matches at any depth, a trailing `/` names a directory anywhere (`target/`), a leading `./` is dropped, and `*` no longer crosses `/` (`src/*.rs` is "directly in `src/`", as the `--glob` help always claimed). `[index] include/exclude` patterns were parsed but never applied; they now drive the walker, the working-tree freshness check and the watcher, so an existing config that sets them will shrink the index on the next `rfx index`. `query::result::build_glob_set` (library API) is the one compiler.

### Performance

Field test (29 MB / 1875 files / 16 cores, warm cache): ripgrep won 6–10x on plain queries and 45–56x on common words. On the synthetic 30 MB latency corpus, medians before → after: MCP zero-hit call 10.1 → 0.07 ms; common word first page 743 → 2.7 ms; common word count 797 → 25 ms; regex `fn (get|set)_\w+` 484 → 10 ms; common identifier first page 3690 → 2.5 ms. `trigrams.bin` 127 MB → 30 MB on that corpus.

- Posting-list intersection is a linear sorted merge with a streaming decoder (only the smallest list is materialised); it was quadratic (`HashSet` retain plus a linear `find` per candidate). The intersection also stops once the next list is far larger than the surviving candidate set, leaving the exact, parallel line verification to finish the job.
- `rfx mcp` and `rfx serve` keep the index open across calls (memory maps, path→id map, query thread pool) and reopen only when the files change on disk or after `index_project`. The per-query `PRAGMA quick_check`, `cache.stats()` (a SQLite open plus a `git` subprocess for the broad-query guard) and two further `git` spawns are gone from the query path; the freshness verdict is one memoised snapshot per workspace, invalidated by every index write (this also fixes `index_project` leaving a stale verdict in the memo for up to the TTL).
- The whole-identifier matcher is compiled once per query, not once per candidate line.
- Regex search verifies only the candidate lines its literals name, in parallel; it scanned every line of every candidate file on one thread. The literal extractor now drops the atom before `?`, `*` and `{0,n}` (`foobar?` → `fooba`), which also fixes a pre-existing file-level miss.
- The zero-result hint (`N substring matches — pass contains:true`) is counted during the search; the MCP layer no longer runs a second full search for it.
- Query-time verification runs on a pool sized by `[performance] parallel_threads` (previously ignored by the query path).
- `--symbols` / `symbols: true` / `find_references` — the one shape the field test found unchanged (40 ms engine against a 16–25 ms floor). The symbol path opened `meta.db` three to four times per query (each with the WAL and foreign-key pragmas) and re-ran the symbol-cache schema migration every time; it now uses one connection held on the shared index handle, opened on first use. It spawned `git rev-parse` per query; the branch is read from `.git/HEAD`. It loaded every file hash on the branch (a three-way join over the whole index) and then looked up every candidate's id in a second query; one query fetches ids and hashes for the candidate paths only. Its comment/string pre-filter scanned every line of every candidate file on one thread; it now checks only the candidate lines, in parallel. Cache misses were written one connection and one `INSERT` at a time from inside the parse pool; they are committed in one transaction afterwards. Synthetic corpus, `get_1234 --symbols`: 15–16 ms → 7–9 ms engine time, with the plain query on the same pattern at 6–9 ms — the symbol overhead fell from ~7 ms to ~2 ms. The latency harness gains `symbol_lookup` (in-process median 7.1 ms, MCP 6.2 ms; budget 25 ms) and `find_references` (14.2 / 13.0 ms; budget 30 ms).

### Added

- **`ignore_case`** — `rfx query -i` / `--ignore-case`, `ignore_case=true` on `rfx serve`, and `ignore_case: true` on `search_code`, `search_regex`, `count_occurrences`, `list_locations` and `find_references`: `rg -i`, and with `contains` `rg -i -F`. A whole-identifier search stays whole-identifier (`realmid` finds `RealmId`, `realmId`, `REALMID`, not `realm_id`). The engine runs it as a `(?i)` regex whose literals are looked up under every case variant, so it costs about what the case-sensitive query costs (`QueryFilter.ignore_case` in the library API; `prepare_literal_pattern` does the rewrite, which carries no warning). Results keep `kind: text_match`. The zero-result substring `hint` is not produced under `ignore_case`.
- `timings.index_path` (`"trigram"` or `"scan"`) in `rfx query --timing --json` and `REFLEX_MCP_TIMING=1` responses: whether the candidates came from the inverted index or every line was verified (`IndexPath` in the library API). `scan` now only means a pattern shorter than 3 chars, a regex with no 3-byte literal, a non-ASCII literal under `(?i)`, or a keyword symbol query.
- MCP `paths: true` on `search_code` / `search_regex` returns `{status, can_trust_results, paths, total_files}` (plus `has_more` when a `limit` cut the list, and `warnings` / `hint` when set) instead of full columnar rows: 53 files cost 7172 bytes in the field test, against 1750 for `rg -l`.
- The latency harness gains `ci_regex` (`(?i)RARE_MARKER_Q7`; budget 40 ms, twice the case-sensitive `rare_ident`).
- `rfx index` prints `Index/corpus ratio: 1.4x (trigrams.bin …, content.bin …)`; `IndexStats` gains `corpus_bytes` and `trigram_index_bytes` (omitted from JSON when zero).
- `rfx query --timing` prints per-phase timings (open, candidates, verify, status, group) to stderr and includes a `timings` object with `--json`; `REFLEX_MCP_TIMING=1` adds the same object to `search_code` / `search_regex` responses.
- `QueryResponse.substring_hint_count`, `PaginationInfo.total_is_exact` / `approx_total`, `QueryFilter.require_exact_total` / `collect_timings` (library API).
- `tests/latency_budget.rs`: a latency harness over a deterministic synthetic 30 MB corpus, measured in-process and through a real `rfx mcp` stdio round-trip; CI enforces budgets with `REFLEX_LATENCY_BUDGET=1`. The CI performance step's filter was also fixed (it ran zero tests).
- `rfx query --pattern <p>` for patterns that begin with `-` (`--pattern '-> Result<'`); `rfx query -- '-> Result<'` still works, and `--help` now says so. The two forms conflict rather than silently picking one.
- Plain-text tier: `.bru` (Bruno API collections), and the extensionless `Makefile`, `Dockerfile` (plus `Dockerfile.<variant>`) and `Justfile`. Every count gap in the field test traced to these. `.mjs` / `.cjs` were always parsed as JavaScript; now documented. `Language::from_path` (library API) is the one classifier the indexer, watcher and query engine share; `is_text_tier_file` admits the new names.
- `QueryResponse.warnings` and `QueryResponse.hint` (library API): what the engine did to the query (a bracket rewrite) and, for a whole-identifier zero with substring matches, a ready-to-show sentence. `query::prepare_literal_pattern` / `substring_hint_text` are public.

### Fixed

- **`(?i)` regexes scanned every file.** The literal extractor discarded every literal the moment it saw an `i` flag, so `(?i)kubernetes` — a pattern with a 10-character literal — verified all 6.6M lines of the Kubernetes checkout (387 ms against ripgrep's 166 ms) and printed `has no literals (≥3 chars), falling back to full content scan`. It was the one query shape whose time grew with corpus size. A literal under `(?i)`, `(?i:…)` or `(?im)` is now looked up under every ASCII case variant of each of its trigrams (`TrigramIndex::search_candidates_fold`), intersected cheapest-first with the same early stop as a plain query, then verified by the regex as before. Because the regex crate's Unicode simple case folding also lets `k` match U+212A KELVIN SIGN and `s` match U+017F LONG S, a literal containing `k` or `s` also unions the lines carrying those two characters (`exotic_fold_lines`; a few hundred directory probes that miss on any code corpus), so counts stay equal to ripgrep `-i`. A non-ASCII literal under `(?i)` still scans, with a warning that says why. Hearth (29 MB): `rfx query '(?i)realmid' --regex --count` finds 3720 lines (ripgrep `-i`: 3720) with 3762 candidate lines from the index, verify 4 ms against 4 ms for the case-sensitive `RealmId` regex, 23 ms engine time end to end; `realmid -i` is 3620 (ripgrep `-iw`: 3620). Synthetic latency corpus: `ci_regex` median 0.27 ms in-process, 0.35 ms over MCP stdio. Kubernetes (6.6M lines): `(?i)kubernetes --count` finds 114244 lines, unchanged, with the candidate phase at 10 ms against 9 ms for the case-sensitive regex; the old scan verified for 210 ms, the index path verifies for 80 ms. The rest of that query's ~220 ms is materialising and grouping 114k results plus the git freshness check, which count mode does not need and which a later change should skip.
- Regex candidate union: the per-literal `BTreeMap` merge is replaced by one sort of the concatenated lists (skipped for a single literal), which was ~8 ms per 100k locations on every regex query. The "no literals" warning is now also carried in `warnings[]` (JSON, MCP), and is not printed for a two-character `ignore_case` literal, which scans silently like its case-sensitive twin.
- **Regex literal extraction had four false negatives**, each a line the regex matched that the trigram candidates never named: a scoped flag group `(?i:foo)bar` swallowed `foo`; an alternation with a branch shorter than 3 chars, `(abc|de)f`, emitted `abc` as if required, so `def` lines were never verified; an optional group `(abc)?x` emitted `abc`; and `\p{Lu}`, `\x41`, `\u{..}`, `\A`, `\z` pushed their escape letter into the literal (`abc\p{Lu}` searched for `abcp`). A character class body was also a literal (`[abc]def` searched for `abc`), a `?` after another quantifier (`abc+?`) popped a second character, and the `x` (verbose) flag was ignored. `tests/regex_candidate_lines.rs` now brute-forces all of these against the regex crate.
- The broad-query guard skipped any regex, including one the engine had just built from a literal (the bracket rewrite, and now `ignore_case`): `-i fn` on a 60k-file index would have scanned unguarded. The guard now judges the original literal (`QueryFilter.rewritten_from`).
- `search_regex` in `mode: "count"` dropped `warnings` and `hint`; it now reports them like `search_code` does.
- **`rfx query 'unwrap()'` (and `--count`, `--json`, `--paths`, `rfx serve`) returned a silent `0`** for every whole-identifier pattern containing `( ) [ ] { } < >` — `unwrap()`, `#[derive(`, `RealmId::nil()`, `-> Result<` — with no warning and `total_is_exact: true`, the exact "confident zero" the 1.7.2 changelog said was fixed. It was fixed only in the MCP handlers. The rewrite (escape, run as a regex, say so) now lives in `QueryEngine::search_with_metadata`, so every surface gets the same answer: `warnings[]` in JSON (also on the `--count --json` object), `Warning:` on stderr in plain and `--paths --json` modes. The zero-result substring hint (`0 whole-identifier matches; 89 substring matches — pass contains:true (--contains on the CLI)`) is now printed by the CLI too, as `hint` in JSON and `Hint:` on stderr. `--regex`, `--contains`, `--symbols`, `--kind` and `--ast` are never rewritten. `tests/cli_query_bracket.rs` drives the real binary and checks the CLI and MCP counts agree on every pattern.
- Symbols read from the cache carry the file's language. `SearchResult.lang` is `#[serde(skip)]`, so every cached symbol deserialised with the default language (`rust`) and `--symbols --json` reported `"language": "rust"` for a TypeScript class. Hidden in the test suite until now because `git rev-parse` inside `tests/corpus` named the repo's branch while the indexer had recorded `_default`, so the cache never hit there.
- Symbol cache reads verify the content hash. `INSERT OR REPLACE` keys on `(file_id, file_hash)`, so a changed file kept its old-hash row beside the new one, and the batch read (which ignored the hash) could serve a changed file its pre-change symbols.

## [1.7.2] - 2026-09-22

Fixes four defects found by a field test of the `rfx mcp` server against ripgrep on a
480k-line, 1027-file repo. Three of them returned a **wrong answer** with
`status: "fresh"` and `can_trust_results: true` — the worst failure shape for an AI
consumer, which does not retry but concludes "no callers" and acts on it.


### ⚠️ Breaking


- `IndexWarning.files_modified` is now a list of paths (`Vec<String>`), not a count (`u32`). It is joined by `files_added`, `files_deleted`, `changed_count` and `truncated`. A count told a caller something was wrong without saying what, so the only safe reaction was to distrust everything.

- A stale index now always reports `can_trust_results: false`. Staleness includes uncommitted working-tree changes, so this fires in ordinary edit-then-search loops. It means "these results may be incomplete", not "this call failed".

- `check_index_status` and search warnings advise `index_project`, not `rfx index`. An agent cannot run the CLI.

- The `language` field serializes **lowercase** (`"rust"`, not `"Rust"`). This was always the behaviour; `CLAUDE.md` documented it wrongly.


### Added


- `contains: true` on `search_code`, `count_occurrences`, `list_locations` and `find_references` — substring matching, like `grep -F`. Literal search matches whole identifiers by default, which was undocumented and had no MCP switch: `verify_csrf` returned 0 against ripgrep's 89, because every hit was `verify_csrf_form_field`.

- A zero-result search now carries a `hint` naming the substring count: `"0 whole-identifier matches; 89 substring matches — pass contains:true"`.

- Plain-text tier: `md mdx txt yaml yml toml json proto html htm sh bash ini cfg sql graphql` are indexed for full-text search. Trigram-only — no symbols, no AST, no dependency analysis. Select with `--lang text`, exclude with `exclude_text: true`, disable with `[index] text_tier = false`. Lock files are never indexed. Previously `count_occurrences` over `*.md` returned 0 against ripgrep's 3425.

- `check_index_status` reports `files_modified`, `files_added`, `files_deleted` and `changed_count` as paths.

- `cache::open_meta_db`, which applies `busy_timeout=5000`, `journal_mode=WAL` and `foreign_keys=ON` to every meta.db connection. `REFLEX_SQLITE_JOURNAL=delete` opts out of WAL on network filesystems.

- `ReflexError::SymbolIndexingInProgress` and `ReflexError::CacheVersionMismatch`.

- `meta.db` records `writer_version` and `writer_git_sha`, replacing the `cache_version` row that nothing read.

- `REFLEX_FRESHNESS_TTL_MS` (default 1000) bounds the cost of the working-tree check; `REFLEX_ALLOW_SCHEMA_REBUILD=1` bypasses the version guard.


### Fixed


- **Symbol indexing no longer uses gigabytes of memory on minified files.** `rfx index-symbols-internal` reached 34.4 GiB RSS and ran 3m55s on a 1027-file repo. `extract_preview` bounded previews in LINES with no byte limit; a minified bundle is 1.4 MB on ONE line, so every one of its ~13,843 symbols got a copy of the whole file. Previews are now capped at 512 bytes, the 14 duplicated copies are one shared function, and minified files are detected by bytes-per-line and skipped for symbol extraction only — they stay fully text-searchable. Measured on the same corpus: **15.27 GiB → 0.11 GiB, 38s → 8s**.

- The same unbounded-preview bug on the QUERY path: full-text and regex results returned the whole matched line, which at the 200-result default page size meant ~290 MB of previews per search over a minified file. Now windowed on the match, so a hit deep inside a single-line file still shows its own neighbourhood. `--expand` gets a separate 32 KB ceiling.

- **Two panics on non-ASCII minified content.** `truncate_preview` sliced a raw byte index when a line had no whitespace in its first 100 characters — exactly minified code, which is also where non-ASCII i18n tables live — crashing `rfx query` and the MCP server. `semantic/answer.rs` had the same unguarded slice in three places. The old per-parser previews also underflowed on line 0 and on Vue/Svelte script offsets.

- `symbol_cache::batch_set` cloned every result to blank a field that parsers already leave empty and that the read path overwrites anyway, doubling peak memory for no effect.

- Symbol-indexing status no longer reports a write failure as a parse failure. One failed batch write used to mark its whole batch as failed without decrementing the parsed count, so 27 successes plus one SQLite error read as `parsed_files: 27, failed_files: 27`. `write_failed_files` and `skipped_minified` are now separate, and the error names the count and the first file.

- Working-tree changes are now detected. Freshness compared `git rev-parse HEAD` to the indexed commit and then sampled the mtimes of the first **ten** indexed files — so an edit to any other file, every untracked file, and every deletion reported `fresh`.

- Literal patterns containing brackets no longer return a silent 0. `unwrap()` (ripgrep: 1221), `#[derive(` (1141) and `-> Result<` (2139) all returned 0, because whole-identifier matching wraps the pattern as `\b…\b` and a pattern ending in `)` can never satisfy the trailing boundary. Such patterns are now escaped onto the regex path, with the rewrite reported in `warnings`.

- Deleted files no longer produce ghost hits. `meta.db` was pruned only by `compact()`, which is throttled to 24h and was skipped entirely for the MCP command; the incremental fast path noticed added paths but never deleted ones, so a delete-one-add-one left the file count unchanged and skipped the rebuild.

- No MCP or CLI call surfaces a raw SQLite lock error. The detached symbol pass holds `meta.db` but takes its own lock, so `rfx index` passed the workspace lock gate and then hit `BEGIN IMMEDIATE`, failing for ~4 minutes with `database is locked: Error code 5`. The indexer now asks the pass to yield at its next batch, and reports `symbol indexing in progress (pid N, started HH:MM:SS, 1000/1027 files)` if it does not.

- A cache-format mismatch is no longer treated as corruption. `validate()` runs on every search and bailed on a schema-hash mismatch, which the MCP layer answered by force-rebuilding — so several Reflex versions sharing one `.reflex/` each rebuilt it concurrently. That is the origin of `content.bin is too small`. Readers now degrade with a warning; writers refuse only when a different released version owns the cache.

- Committing already-indexed content no longer leaves the index permanently stale. The incremental fast path skipped the rebuild and therefore never refreshed the recorded commit.

- `list_locations` returns one entry per **match**, as documented. It set `paths_only`, which collapses each file to its first match, so a 20-match pattern returned 8 entries.

- `find_references` count fields reconcile. `pagination.total` 25 / `total_references` 24 / `returned_count` 24 looked like an off-by-one; each now has one meaning and the gap is named `filtered_out`.

- `include_strings` applies in `find_references` count mode, where it was a no-op returning the raw engine total.

- Symbol-indexing progress advances. The `processed % 500 < batch_size` guard was a no-op (`batch_size` was itself 500); batches are now 128 and status is written per chunk, with `pid`, `phase` and `current_file`.

- Stale `indexing.lock` files are reaped by pid liveness rather than a one-hour age rule.

- Reflex no longer treats its own `.reflex/` directory as a source of staleness.


## [1.5.2] - 2026-05-16


### Documentation


- Fix rfx pulse digest → rfx pulse changelog in README

- Remove stale hardcoded test count badge from README


### Fixed


- Move performance tests to separate release-mode CI step

- Resolve all clippy warnings and enable -D warnings in CI

- Commit insta snapshots and fix Windows clippy errors (REF-172)

- Normalize path separators to fix 21 Windows-only test failures (REF-173)

- Make path/home overrides reach storage and dirs lookups (REF-173)

## [1.5.1] - 2026-05-15


### Fixed


- Correct release.toml format for cargo-release

- Restore git-cliff pre-release hook in correct format

- Repair CI failures blocking release

## [1.4.0] - 2026-05-15


### Added


- Add check_index_status MCP tool (REF-107)

- Raise preview truncation to 180 chars and add preview_length param


### Documentation


- Add MCP tool selection decision tree cheatsheet

- Reframe README to reflect CLI-first, not AI-only


### Fixed


- Fix JSON output correctness across query, deps, and analyze

- Detect corrupt trigrams.bin via magic-byte check before skipping rebuild

- Correct singular/plural for result count in TUI (REF-66)

## [1.3.5] - 2026-05-14


### Fixed


- Hardlink issue in pipeline

## [1.3.4] - 2026-05-14


### Fixed


- Dumb symlink issue

## [1.3.3] - 2026-05-14


### Fixed


- Release pipeline

## [1.3.2] - 2026-05-14


### Fixed


- Cargo-dist version issue

## [1.3.1] - 2026-05-14


### Fixed


- Some cargo-dist dependency issues

## [1.3.0] - 2026-05-14


### Added


- Introduce SymbolRef for stable JSON symbol output

- Wire dependencies parameter to list_locations, count_occurrences, search_regex, search_ast


### Changed


- Decompose query.rs into src/query/ submodules


### Documentation


- Add rfx serve threat model and language forward-compat note


### Fixed


- Update reflex-search npm package to fix CVE vulnerabilities

- Broken csharp tests resolved

- Indexing and git workspace issue

## [1.1.3] - 2026-04-27


### Fixed


- Json serialization for mcp

## [1.1.0] - 2026-04-13


### Added


- Add OpenRouter provider support with model fetching and sorting options

- Remove Groq provider and update references to OpenRouter in configuration and documentation

- Enhance LLM response handling with validation and JSON extraction improvements

- Enhance module detection and snapshot analysis with improved descriptions and visibility adjustments

- Implement static site generator with wiki, digest, and map

- Enhance Pulse functionality with LLM narration support for digest and wiki generation

- Add dependency diagram to wiki pages


### Fixed


- Update OpenRouter sort strategy from "speed" to "latency" for API compatibility

## [1.0.4] - 2026-04-07


### Changed


- Streamline language parsing and error handling in CLI and indexer


### Documentation


- Add important setup notes for running commands and gitignore configuration

- Add gitcgr code graph badge

## [1.0.3] - 2025-11-21


### Added


- Update mcp docs to force auto-reindexing and hopefully fixed auto packaging pipeline

## [1.0.1] - 2025-11-21


### Added


- Added auto package publishing for new releases

## [1.0.0] - 2025-11-21


### Added


- Rfx query interactive mode

- Made context full by default and added rfx context to mcp tools

## [0.9.2] - 2025-11-21


### Documentation


- Update README for clarity and accuracy in feature descriptions

## [0.9.1] - 2025-11-20


### Added


- Enhance agentic loop to return query confidence alongside responses

- Semantic query building with external LLMs

## [0.9.0] - 2025-11-19


### Added


- Implement semantic query execution and parsing

- Refactor semantic query generation with project-specific configuration and context extraction

- Massively cleaned up claude.md

- Enhance query execution with count mode and update prompt template for new flags

- Enhance configuration documentation and improve logging levels for cache and semantic query handling

- Update AI provider models and enhance OpenAI request handling for GPT-5 compatibility

- Enhance environment variable handling in tests and update Gemini model defaults

- Add context generation module for AI prompts

- Update default tree depth for --structure option to 1

- Implement agentic semantic query builder with multi-phase workflow

- Enhance agentic mode with additional command options and reporting capabilities

- Enhance regex pattern syntax documentation and clarify flag combinations

- Add support for OpenAI GPT-OSS models with enhanced handling and messaging

- Enhance agentic mode with improved response structure and reporting capabilities

- Add conversational answer generation feature and update CLI handling

- Add context extraction for match results in query handling and answer generation

- Increase token limits to 4000 for Anthropic, Gemini, Groq, and OpenAI providers

- Add termimad for markdown rendering and update provider display with recommendations

- Enhance agentic reporter with spinner support for improved progress visualization

- Implement interactive TUI chat mode for `rfx ask` with message history, input handling, and progress updates

- Enhance Groq API error handling with detailed logging and timeout configuration

- Add bottom padding for message display to improve text wrapping in TUI chat mode

- Add mouse event handling for scrolling in TUI chat mode

- Add text wrapping functionality for message display in TUI chat mode

- Add markdown rendering with consistent prefix for message display in TUI chat mode

- Update wrap_with_prefix to use consistent border colors for message display in TUI chat mode

- Add debug mode to output full LLM prompts and retain terminal history

- Implement API key configuration check and enhance documentation search functionality

- Enhance agentic mode to gather and utilize context for improved answer generation

## [0.8.2] - 2025-11-16


### Added


- Enhance documentation for dependency tracking and semantic query building

- Implement automatic cache invalidation using schema hash

- Implement comprehensive cache corruption detection and validation tests

- Implement cache compaction functionality and CLI commands for manual compaction

- Add cache compaction commands and update documentation for indexing status

## [0.8.1] - 2025-11-16


### Added


- Update documentation for dependency analysis commands and enhance Ruby parser for require statements

## [0.8.0] - 2025-11-16


### Added


- Updated python, go, ts and rust dependency resolution to support monorepos

- Added java, kotlin and ruby monorepo support

- Fixed vue and ts dep resolution

- Add size filtering for island detection

- Refactor dependency analysis API with cleaner command separation

- Enhance dependency analysis commands with pagination and sorting options

- Add analyze_summary tool for quick dependency health overview

- Enhance dependency analysis with file-level grouping and improved output formats

- Refactor search result handling to always use grouped format and improve response structure

- Dependency tracking and analysis


### Fixed


- Tightening up js/ts path resolution

- Enhanced ts/js path resolution for dependencies

- Some updates to pagination for rfx analyze

## [0.7.1] - 2025-11-12


### Added


- Added basic functionality for dependency support

- Dependency resolution across more languages


### Fixed


- Updated some more broken language dependency parsers

- Some more dependency resolution bugfixes

- Refined some regex functionality

## [0.6.0] - 2025-11-11


### Added


- Implemented more mouse support and fixed some syntax highlighting

- Add missing CLI filter options to interactive mode

- Add mouse click support for new filter options

- More UI refinement

- Interactive mode


### Fixed


- Various bugfixes with result list

- Fixed remaining known syntax highligting issues in interactive mode

- More bugfixes

- Accurate mouse click detection for all filter badges

- More ui bugfixes

- Fixed filter display bug

- Rendering bugfixes

- Incorrect file counts in interactive mode

- Fixed background indexing issues

## [0.5.2] - 2025-11-10


### Added


- Added AI suggestions

## [0.5.1] - 2025-11-09


### Fixed


- Random bugfixes

- Switched symbol indexing to use content cache rather than filesystem reads for performance boost

## [0.5.0] - 2025-11-09


### Added


- First swipe at interactive mode, work in progress

- Enhancements

- Loading windows

- Added preflight check to prevent runaway and unbounded queries

- Added early globbing


### Fixed


- Fixed result scrolling

- Removed early filtering performance boost that ended up sacrificing accuracy

- Small optimization to lookups

- Large performance improvement and bugfix with --kind filtering

- Some database refactoring

- Early language filtering

- Speeding up background indexing

## [0.4.2] - 2025-11-07


### Added


- Added many tests and edge cases to symbol search


### Fixed


- Reworking some of the symbol search functionality, bugs remain

- Major fixes to symbol search accuracy

## [0.4.0] - 2025-11-07


### Added


- Lots of json output optimizations to reduce token usage

- Added pagination to prevent breaking context window limits


### Fixed


- Moved pagination to before symbol enrichment to boost performance

- Pagination total count bug

## [0.3.2] - 2025-11-06


### Added


- Fixed attribute support with --kind query

## [0.3.0] - 2025-11-06


### Added


- Added language filter and parallelized --symbol extraction

- Implement word-boundary matching as default search behavior

- Refactored AST query functionality, much simpler now


### Fixed


- Added thread cap for indexing and querying

- Added some missing symbol grammar and added universal ast support, not just for select languages

## [0.2.13] - 2025-11-05


### Fixed


- Cache validation was reading entire files causing 18x slowdown

## [0.2.12] - 2025-11-05


### Added


- Added globbing and build optimizations

## [0.2.10] - 2025-11-04


### Fixed


- Prevent cargo-dist from uploading archives to release

## [0.2.9] - 2025-11-04


### Fixed


- Improve release artifacts with friendly names

## [0.2.7] - 2025-11-04


### Fixed


- Correct release workflow

## [0.2.6] - 2025-11-04


### Fixed


- Trying more release fixes

## [0.2.5] - 2025-11-04


### Fixed


- Another release pipeline fix

## [0.2.4] - 2025-11-04


### Fixed


- Update the release pipeline

## [0.2.3] - 2025-11-04


### Fixed


- Bump version

## [0.2.2] - 2025-11-04


### Fixed


- Bumped version v0.2.2

## [0.2.1] - 2025-11-04


### Fixed


- Lots of general bugfixes

- Removed zip archives from releases v0.2.1

## [0.2.0] - 2025-11-04


### Added


- Add cross-platform binary distribution with cargo-dist

## [0.1.2] - 2025-11-03


### Fixed


- Use default GITHUB_TOKEN and correct repository URL

## [0.1.1] - 2025-11-03


### Fixed


- Correct release-plz GitHub Action reference

- Correct release-plz.toml configuration format

- Remove rfx symlink to fix release-plz

## [0.1.0] - 2025-11-03

