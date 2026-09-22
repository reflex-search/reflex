## [Unreleased] - 1.8.0

### ⚠️ Breaking

- `trigrams.bin` V4 — **re-index required**. A V3 index is served from an in-memory rebuild (slow) until `rfx index` runs; `rfx index` detects the schema change and rebuilds in full. Postings are now one per distinct trigram per line, grouped into per-file blocks, with no byte offsets: 3.9x → 0.9x of corpus size on the synthetic latency corpus (127 MB → 30 MB), 1.4x on the Reflex repo. `TrigramIndex::load` is O(files) — the directory is binary-searched in the mmap instead of being decoded and re-sorted. `FileLocation` (library API) drops `byte_offset`; `FileLocation::new` takes `(file_id, line_no)`.

### Added

- `rfx index` prints `Index/corpus ratio: 1.4x (trigrams.bin …, content.bin …)`; `IndexStats` gains `corpus_bytes` and `trigram_index_bytes` (omitted from JSON when zero).

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

