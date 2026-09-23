# Reflex TODO

**Last Updated:** 2026-09-22
**Project Status:** Testing & Quality Phase Complete - Production Ready

> **⚠️ AI Assistants:** Read the "Context Management & AI Workflow" section in `CLAUDE.md` for instructions on maintaining this file and creating RESEARCH.md documents. This TODO.md MUST be updated as you work on tasks.

---

## 🔄 MAJOR ARCHITECTURAL DECISION #1 (2025-10-31)

**Decision:** Reflex is being redesigned from a **symbol-only index** to a **trigram-based full-text code search engine** (like Sourcegraph/Zoekt).

**Rationale:**
- User requirement: "Local, fast replacement for Sourcegraph Code Search for AI workflows"
- Symbol-only approach was incomplete: `query "extract_symbols"` found 1/8 occurrences (12.5% recall)
- Full-text search needed to find function calls, variable usage, comments, etc.
- Trigram indexing enables <100ms queries on 10k+ files with complete coverage

**Key Changes:**
1. **Indexing**: Extract trigrams from all file content (not just symbol snippets)
2. **Storage**: `trigrams.bin` (inverted index) + `content.bin` (full file contents)
3. **Querying**: Intersect trigram posting lists → verify matches → return line-based results
4. **Symbol filter**: Keep Tree-sitter integration for `symbol:` prefix queries
5. **Regex support**: Extract trigrams from patterns; fall back to full scan when needed

**Implementation Status:** ✅ COMPLETED

---

## 🔄 MAJOR ARCHITECTURAL DECISION #2 (2025-11-03)

**Decision:** Remove indexed symbols entirely and implement **runtime symbol detection** using tree-sitter at query time.

**Rationale:**
- Performance regression: Per-file symbol storage was loading 3.3M symbols on every query (4125ms)
- Monolithic symbol storage had same problem (2700ms to load all symbols)
- Observation: Trigrams narrow search to ~10-100 candidate files
- Solution: Parse only candidate files at query time instead of all files at index time

**Key Changes:**
1. **Removed**: `symbols.bin`, `SymbolWriter`, `SymbolReader`
2. **Indexing**: NO symbol extraction or tree-sitter parsing (trigrams only)
3. **Querying**: Parse ~10 candidate files with tree-sitter at runtime (~2-5ms overhead)
4. **Result**: 2000x performance improvement (4125ms → 2ms for symbol queries)

**Architecture Benefits:**
- **Simpler**: Removed 3 files and ~500 lines of complex symbol storage code
- **Faster Indexing**: No tree-sitter parsing during indexing (much faster)
- **Faster Queries**: 2-3ms vs 4125ms (parse 10 files vs load 3.3M symbols)
- **Smaller Cache**: No symbols.bin (saved ~15KB for reflex, MBs for large codebases)
- **More Flexible**: Add new symbol types without reindexing

**Performance Results on Linux Kernel (62K files):**
- Full-text: `124ms` (unchanged)
- Regex: `156ms` (unchanged)
- Symbol search: `224ms` (parse ~3 C files vs 4125ms loading all symbols)
- This makes Reflex the **fastest structure-aware local code search tool**

**Implementation Status:** ✅ COMPLETED

**See:** This change obsoletes previous symbol storage research. New architecture is pure trigram + runtime parsing.

---

## 🔧 1.8.0 post-perf-round defects (2026-09-22) — COMPLETED

Field test of the perf round (b18ae06 → 11ba6de) against ripgrep surfaced four defects
plus two small items. All fixed on `feat/perf-enhancements`, in the order below; the
latency harness stays green with budgets enforced (`REFLEX_LATENCY_BUDGET=1`).

| # | Defect | Status | Where |
| --- | --- | --- | --- |
| 1 | CLI bracket literals (`unwrap()`) returned a silent 0; rewrite lived only in MCP | ✅ done | rewrite moved into `QueryEngine::search_with_metadata`; `QueryResponse.warnings` / `hint`; `tests/cli_query_bracket.rs` (CLI ⇄ MCP parity) |
| 2 | Inexact list-mode `total` reported the verified-so-far count; `approx_total` was a 2x upper bound | ✅ done (breaking) | `PaginationInfo.total: Option<usize>` (`null` when inexact); sampled estimate in `verify_files_streaming` (32 files / ≤16 lines each; finishes when ≤32 files or ≤128 lines remain); synthetic corpus error 0–1.4% |
| 3 | Globs unanchored (`src/**/*.rs` matched `vendor/src/`) | ✅ done (breaking) | gitignore rules in `query::result::normalize_glob_pattern` + `build_glob_set` (`literal_separator`); `[index] include/exclude` now applied via `indexer::PathPolicy` (walker, freshness check, watcher); `tests/glob_anchoring.rs` |
| 4 | `--symbols` 40 ms vs 16–25 ms floor | ✅ done | one `meta.db` connection on `OpenIndex` (`meta_conn`), `.git/HEAD` instead of `git rev-parse`, candidate-only hash query (`branch_file_rows_on`), parallel candidate-line pre-filter, batched cache writes; hash-verified cache reads (correctness); harness shapes `symbol_lookup` (7.1 ms) + `find_references` (14.2 ms) |
| 5a | `->` patterns need `--` | ✅ done | `rfx query --pattern <p>` (`allow_hyphen_values`), `--help` note |
| 5b | `.bru`, `Makefile`, `Dockerfile`, `Justfile` outside both tiers | ✅ done | `TEXT_EXTENSIONS` + `TEXT_FILENAMES`; `Language::from_path` is the one classifier |

Not done / follow-ups:
- The first `--symbols` call in a fresh process on a cache miss is ~60 ms (tree-sitter
  parser + query compilation, one-time per process); `rfx mcp` pays it once.
- `estimate_is_within_band_on_synthetic_corpus` is `#[ignore]` (builds the 30 MB corpus);
  run with `cargo test --release --test query_early_termination -- --ignored`.

---

## 🎯 Current Status Summary (Updated: 2025-11-16)

### 🚀 ALL MVP FEATURES COMPLETE + MAJOR ENHANCEMENTS!

Reflex is **production-ready** with all core features and major enhancements implemented:

**Core Features:**
✅ **HTTP Server** - FULLY IMPLEMENTED (src/cli.rs, lines 428-687)
✅ **AST Pattern Matching** - FULLY IMPLEMENTED (src/ast_query.rs, 428 lines)
✅ **File Watcher** - FULLY IMPLEMENTED (src/watcher.rs, 289 lines)
✅ **MCP Server** - FULLY IMPLEMENTED (src/mcp.rs, 476 lines)
✅ **Additional Language Support** - C#, Ruby, Kotlin, Zig ALL COMPLETE
✅ **Background Symbol Indexing** - FULLY IMPLEMENTED (src/background_indexer.rs, src/symbol_cache.rs)
✅ **Interactive Mode (TUI)** - FULLY IMPLEMENTED (src/interactive/, 12 files, ~2000+ LOC)

**Recent Completions (2025-11-16):**
✅ **Automatic Cache Invalidation** - Build-time schema hashing (commit 629495b)
✅ **Cache Compaction** - `rfx index compact` command
✅ **Parallel Indexing with Rayon** - 2-3x speedup on multi-core systems
✅ **Progress Reporting** - Live progress bars during indexing
✅ **GitHub Actions** - Automated CI/CD workflows
✅ **Release Engineering** - Cross-compilation for 6 platforms, crates.io publishing

**Current Phase:** ✅ Testing Complete (458 tests passing) - Production Ready

---

### ✅ FULLY FUNCTIONAL
Reflex is **operational as a local code search engine** with the following capabilities:

**Working Features:**
- ✅ Full-text trigram-based search (finds ALL occurrences of patterns)
- ✅ Symbol-only search (functions, structs, enums, traits, etc. for Rust)
- ✅ Incremental indexing (only reindexes changed files)
- ✅ Memory-mapped I/O for fast cache access
- ✅ CLI with rich filtering (--lang, --kind, --file, --expand, --exact, --symbols)
- ✅ JSON output for AI/automation consumption
- ✅ .gitignore support (uses `ignore` crate)
- ✅ SQLite metadata tracking
- ✅ Statistics and cache management commands

**Supported Languages (for indexing):**
- ✅ **Rust** - Full symbol extraction (functions, structs, enums, traits, impls, methods, constants, modules, type aliases)
- ✅ **TypeScript/JavaScript** - Full symbol extraction (functions, classes, interfaces, types, enums, methods, arrow functions, React components)
- ✅ **Vue** - Symbol extraction from `<script>` blocks (Composition API and Options API support)
- ✅ **Svelte** - Symbol extraction from component scripts (including reactive declarations)
- ✅ **PHP** - Full symbol extraction (functions, classes, interfaces, traits, methods, properties, constants, namespaces, enums)
- ✅ **Python** - Full symbol extraction (functions, classes, methods, decorators, lambdas, constants)
- ✅ **Go** - Full symbol extraction (functions, types, interfaces, methods, constants, variables)
- ✅ **Java** - Full symbol extraction (classes, interfaces, enums, methods, fields, constructors)
- ✅ **C** - Full symbol extraction (functions, structs, enums, unions, typedefs, global variables)
- ✅ **C++** - Full symbol extraction (functions, classes, structs, namespaces, templates, methods, enums, type aliases)
- ✅ **C#** - Full symbol extraction (classes, interfaces, structs, enums, records, delegates, methods, properties, namespaces)
- ✅ **Ruby** - Full symbol extraction (classes, modules, methods, singleton methods, constants, blocks)
- ✅ **Kotlin** - Full symbol extraction (classes, objects, interfaces, functions, properties, data classes, sealed classes)
- ✅ **Zig** - Full symbol extraction (functions, structs, enums, constants, variables, test declarations, error sets)

**What Works:**
```bash
# Index current directory
reflex index

# Check background symbol indexing status
reflex index --status

# Full-text search (finds all occurrences)
reflex query "extract_symbols"  # Finds: definitions + call sites

# Symbol-only search (definitions only)
reflex query "parse" --symbols --kind function

# Regex search (with trigram optimization)
reflex query "fn.*test" --regex  # or -r
reflex query "(class|function)" --regex

# Paths-only mode (return unique file paths)
reflex query "TODO" --paths
vim $(reflex query "TODO" --paths)  # Open all files with TODOs

# Pagination
reflex query "extract" --limit 10 --offset 0   # First page
reflex query "extract" --limit 10 --offset 10  # Second page

# Glob and exclude patterns
reflex query "config" --glob "src/**/*.rs" --exclude "src/generated/**"

# With filters
reflex query "unwrap" --lang rust --limit 10 --json
```

### ⚠️ REMAINING WORK

**1.7.0 MCP hardening (2026-09-21, from the 34-session Hearth transcript analysis):**
- [x] A. Argument validation for every MCP tool: aliases (`query`/`symbol`/`text`/`search`→`pattern`,
      `max_results`→`limit`, `path`→`file` on search tools) with a `warnings` field; unknown keys →
      `-32602` with received/valid/did-you-mean; numeric/boolean string coercion; typed errors
      (`ReflexError::InvalidParams`). Specs derived from the `tools/list` schema at runtime (no drift).
- [x] B. `initialize` instructions rewritten: deferred-schema `ToolSearch("select:mcp__reflex__…")`
      hint, exact call shapes, canonical arg names, literal-index note (`MCP_INSTRUCTIONS`, ≤1600 chars).
- [x] C. Crash-safe index writes (`src/atomic_write.rs`: tmp + fsync + rename) for content.bin and
      trigrams.bin; `.reflex/index.lock` OS advisory lock held for the whole `Indexer::index` run
      (`IndexConfig::lock_wait_secs`, CLI waits 30 s); `clear()` takes the lock; `init_meta_db` is
      idempotent + transactional (a kill mid-schema used to brick the cache with `no such table`).
- [x] D. MCP auto-recovery: `CacheCorrupted` → force rebuild once → retry; errors name
      `index_project`, never `rfx index`. Tests: `tests/mcp_corruption_recovery.rs`,
      `tests/index_crash_safety.rs` (real `rfx index` killed mid-write), `tests/index_lock.rs`.
- [ ] E. Follow-up: measure first-call correctness on a fresh Claude Code session with deferred tools
      (acceptance criterion from the handoff) once 1.7.0 is installed in a consumer project.

**P2 Enhancements Not Yet Implemented:**
- **Query Result Caching** (LRU cache for sub-1ms cached queries) - See section 3, lines 649-658
  - In-memory LRU cache with 100-entry limit
  - <1ms for cached queries vs 10-100ms for fresh queries
  - Automatic invalidation on reindex
  - Estimated: 2 days

**P3 Nice-to-Have Features:**
- Streaming Parse for Huge Files (>10MB without full memory load)
- Generic Fallback for Unknown AST Nodes (future-proof against language evolution)
- Language Version Tracking (store tree-sitter grammar versions)
- Request Logging Middleware (HTTP server)
- Rate Limiting (HTTP server)
- API Authentication (HTTP server)
- WebSocket Support (HTTP server)
- DOT format visualization (Graphviz dependency graphs)
- Advanced path resolution (package.json, tsconfig.json parsing)
- Session caching for dependency queries
- Workspace/monorepo support enhancements
- Semantic Query Building (NL → rfx command via LLM APIs)

**ALL CORE FEATURES AND P2 ENHANCEMENTS ARE COMPLETE** (except Query Result Caching)

**Recently Completed (2025-11-16):**
- ✅ Automatic Cache Invalidation (commit 629495b)
- ✅ Cache Compaction (`rfx index compact` command)
- ✅ Parallel Indexing with Rayon (2-3x speedup)
- ✅ Progress Reporting (live progress bars)
- ✅ GitHub Actions (CI/CD workflows)
- ✅ Release Engineering (cross-compilation, crates.io publishing)
- ✅ Interactive Mode (TUI) - 12 files, ~2000+ LOC

**Previously Completed:**
1. **Background Symbol Indexing** - COMPLETED (2025-11-09) ✅
   - Daemonized background process for symbol caching (src/background_indexer.rs, ~350 lines)
   - Symbol cache system (src/symbol_cache.rs, 803 lines)
   - Persistent symbol storage for faster symbol queries
   - Progress tracking with status command (`rfx index --status`)
   - Automatic spawning after trigram indexing completes
   - Platform-specific process detachment (Unix/Windows support)
   - Benefits: Dramatically faster symbol searches on large codebases
   - Architecture: Separate process reads from content cache, parses with tree-sitter, writes to symbol cache

2. **CLI Enhancements** - COMPLETED (2025-11-09) ✅
   - `--paths` flag: Return unique file paths only (no line numbers/content)
   - `--offset` flag: Pagination support (use with --limit for windowed results)
   - `--all` flag: Return unlimited results (convenience for --limit 0)
   - `--force` flag: Bypass broad query detection for expensive queries
   - `--glob` patterns: Include specific files/directories (can be repeated)
   - `--exclude` patterns: Exclude specific files/directories (can be repeated)
   - `--no-truncate` flag: Disable smart preview truncation
   - `--pretty` flag: Pretty-print JSON output
   - Smart limit handling: Automatic unlimited mode for --paths without explicit --limit

3. **Query Pipeline Refactor** - COMPLETED (2025-11-03) ✅
   - Replaced mutually-exclusive branching with composable pipeline architecture
   - Fixed bug: regex + symbol filtering now works correctly
   - Architecture: Phase 1 (candidates) → Phase 2 (enrichment) → Phase 3 (filters)
   - Optimal ordering: trigram → regex → symbols (minimizes expensive parsing)
   - All filter combinations now work (regex+symbols, regex+kind, etc.)
   - Performance maintained: 2-190ms depending on query type
   - 221 tests passing

4. **Regex support** - FULLY IMPLEMENTED ✅
   - Regex pattern matching with trigram optimization (src/regex_trigrams.rs)
   - Literal extraction from regex patterns (≥3 chars)
   - Union-based file selection for correctness
   - Integration with query engine via --regex flag
   - Comprehensive test coverage (13 test cases)
   - Works with symbol filtering after pipeline refactor

**Performance Note:**
- Queries are extremely fast: 2-3ms on small codebases, 124-224ms on Linux kernel (62K files)
- Full-text search uses persisted trigram index from trigrams.bin (rkyv zero-copy deserialization via memory-mapping for instant access)
- Symbol search uses runtime tree-sitter parsing (parse only candidate files found by trigrams)

### 📊 Implementation Progress

| Component | Status | Completeness |
|-----------|--------|--------------|
| **Core Infrastructure** | ✅ Complete | 100% |
| **Cache System** | ✅ Complete | 100% |
| **Cache Invalidation** | ✅ Complete | 100% |
| **Cache Compaction** | ✅ Complete | 100% |
| **Indexer** | ✅ Complete | 100% |
| **Parallel Indexing (Rayon)** | ✅ Complete | 100% |
| **Progress Reporting** | ✅ Complete | 100% |
| **Query Engine** | ✅ Complete | 100% |
| **Trigram Search** | ✅ Complete | 100% |
| **Regex Search** | ✅ Complete | 100% |
| **Content Store** | ✅ Complete | 100% |
| **Runtime Symbol Parser** | ✅ Complete | 100% |
| **Rust Parser** | ✅ Complete | 100% |
| **TypeScript/JS Parser** | ✅ Complete | 100% |
| **Vue Parser** | ✅ Complete | 100% |
| **Svelte Parser** | ✅ Complete | 100% |
| **PHP Parser** | ✅ Complete | 100% |
| **Python Parser** | ✅ Complete | 100% |
| **Go Parser** | ✅ Complete | 100% |
| **Java Parser** | ✅ Complete | 100% |
| **C Parser** | ✅ Complete | 100% |
| **C++ Parser** | ✅ Complete | 100% |
| **C# Parser** | ✅ Complete | 100% |
| **Ruby Parser** | ✅ Complete | 100% |
| **Kotlin Parser** | ✅ Complete | 100% |
| **Zig Parser** | ✅ Complete | 100% |
| **CLI** | ✅ Complete | 100% |
| **Interactive Mode (TUI)** | ✅ Complete | 100% |
| **HTTP Server** | ✅ Complete | 100% |
| **File Watcher** | ✅ Complete | 100% |
| **MCP Server** | ✅ Complete | 100% |
| **AST Pattern Matching** | ✅ Complete | 100% |
| **GitHub Actions** | ✅ Complete | 100% |
| **Release Engineering** | ✅ Complete | 100% |
| **Tests** | ✅ Complete | 100% (458 total tests) |
| **Documentation** | ✅ Complete | 85% (README, ARCHITECTURE, rustdoc, HTTP API) |

---

## Executive Summary

Reflex has **successfully transitioned to a trigram-based full-text search engine** with the following architecture:

**Implemented:**
- ✅ Trigram indexing module (src/trigram.rs) - FULLY FUNCTIONAL
- ✅ Content store (src/content_store.rs) - Memory-mapped file storage
- ✅ Symbol extraction (src/parsers/rust.rs) - Comprehensive Rust support
- ✅ Dual-mode querying: full-text (trigrams) + symbol-only (Tree-sitter)
- ✅ CLI framework (6 commands: index, query, stats, clear, list-files, serve)
- ✅ Incremental indexing with blake3 hashing
- ✅ SQLite metadata tracking

**Architecture:**
```
reflex index  →  [Directory Walker] → [Trigram Extractor]
                                              ↓
                                        [trigrams.bin]
                                        (Memory-mapped)
                                              ↓
                                        [content.bin]
                                        (Memory-mapped)

reflex query  →  [Query Engine] → [Mode: Full-text or Symbol-only]
                       ↓
                 [Trigram Search] → [Candidate Files (10-100 files)]
                       ↓                      ↓
                 [Content Match]      [Runtime Tree-sitter Parse]
                       ↓                      ↓
                 [Results (JSON/Text)]  [Symbol Filter] → [Results]
```

**Next Phase:**
1. ✅ All major parsers complete (Rust, TS/JS, Vue, Svelte, PHP, Python, Go, Java, C, C++)
2. Add HTTP server (optional)
3. ✅ Performance optimization complete (runtime symbol detection)

---

## Priority Levels

- **P0 (MVP):** Required for minimum viable product (instant queries, basic symbol search)
- **P1 (Core):** Essential features for production readiness
- **P2 (Enhancement):** Nice-to-have features and optimizations
- **P3 (Future):** Long-term roadmap items from CLAUDE.md

---

## 🎯 MVP Goals (from CLAUDE.md) ✅ ALL COMPLETED

- [x] **Goal 1:** <100 ms per query on 100k+ files (warm path, OS cache) ✅
  - **Achieved:** 2-3ms on small codebases, 124ms on Linux kernel (62K files), instant on medium codebases
- [x] **Goal 2:** Accurate symbol-level and scope-aware retrieval for Rust, TS/JS, Go, Python, PHP, C, C++, Java, C#, Ruby, Kotlin, Zig ✅
  - **Achieved:** Runtime symbol detection implemented for all 18 supported languages
- [x] **Goal 3:** Fully offline; no daemon required (per-request invocation) ✅
  - **Achieved:** Fully offline operation, optional background symbol indexer available
- [x] **Goal 4:** Clean, stable JSON API suitable for LLM tools and editor integrations ✅
  - **Achieved:** JSON output, HTTP server, MCP server all implemented
- [x] **Goal 5:** Optional on-save incremental indexing ✅
  - **Achieved:** Incremental indexing with blake3 hashing, file watcher for auto-reindex

---

## 📋 Task Breakdown by Module

### 1. Cache Module (`src/cache.rs`)

#### P0: Core Cache Infrastructure ✅ COMPLETED
- [x] **Implement cache file initialization** (cache.rs:48-72)
  - Create `meta.db` with schema (SQLite) ✅
  - Create empty `symbols.bin` with header ✅
  - Create empty `tokens.bin` with header ✅
  - Create `hashes.json` with empty JSON object `{}` ✅
  - Create default `config.toml` with sensible defaults ✅

- [x] **Implement hash persistence** (cache.rs:270-299)
  - `load_hashes()`: Read and deserialize `hashes.json` ✅
  - `save_hashes()`: Serialize and write `hashes.json` ✅

- [x] **Implement cache statistics** (cache.rs:390-455)
  - Read actual file sizes from disk ✅
  - Count symbols from SQLite database ✅
  - Count files from SQLite database ✅
  - Store and retrieve last update timestamp ✅

#### P1: Memory-Mapped Readers ✅ COMPLETED
- [x] **Implement SymbolReader** (cache/symbol_reader.rs)
  - Memory-map `symbols.bin` for zero-copy reads ✅
  - Define binary format for symbol storage (rkyv) ✅
  - Implement symbol deserialization ✅
  - Add index structure for fast lookups (HashMap) ✅
  - Uses `memmap2` crate ✅

- [x] **Implement TokenReader**
  - **Note:** Replaced by trigram-based full-text search (src/trigram.rs)
  - Trigrams extracted during indexing ✅
  - Persisted to trigrams.bin and memory-mapped on query ✅

- [x] **Implement MetaReader** (cache.rs:355-455)
  - Read metadata from `meta.db` via SQLite ✅
  - Support queries for statistics ✅
  - Queries execute directly (no separate reader needed) ✅

#### P2: Advanced Cache Features

**Automatic Cache Invalidation (Build-Time Schema Hash):** ✅ COMPLETED (2025-11-16)
- **Zero manual intervention** - Fully automatic via build.rs ✅
- **Implementation:** ✅ COMPLETED (commit 629495b)
  - build.rs hashes cache-critical files at compile time: ✅
    - src/cache.rs (SQLite schema, cache initialization)
    - src/content_store.rs (content.bin binary format)
    - src/trigram.rs (trigrams.bin binary format)
    - src/indexer.rs (indexing pipeline logic)
    - src/symbol_cache.rs (symbol cache format)
    - src/models.rs (core data structures)
    - src/dependency.rs (dependency extraction/storage)
  - Stores hash as `CACHE_SCHEMA_HASH` via cargo:rustc-env ✅
  - On cache load: compares stored hash vs current hash ✅
  - Mismatch → warns user and auto-rebuilds index ✅
- **What triggers rebuild:** ✅
  - SQLite schema changes
  - Binary format changes (trigrams.bin, content.bin)
  - Indexing pipeline bugs/fixes
  - Data structure changes
- **What DOESN'T trigger rebuild:** ✅
  - Query engine changes (src/query.rs)
  - Individual parser fixes (src/parsers/*.rs)
  - CLI/UI changes (src/cli.rs, src/mcp.rs, etc.)
- **Acceptance criteria:** ✅ ALL MET
  - Never requires manual version bump ✅
  - Catches all cache-invalidating changes ✅
  - Clear user message: "Cache format changed, rebuilding..." ✅
- **Implementation details:**
  - Location: build.rs (schema hash computation)
  - Integration: src/cache.rs (validation on load)
  - Fully operational and tested ✅

**Corruption Detection (Zero Query-Time Overhead):** ✅ COMPLETED (2025-11-16)
- **CRITICAL CONSTRAINT:** Runs ONLY at cache load, NEVER during queries ✅
- **Implementation:**
  - Verify magic bytes in trigrams.bin/content.bin headers (4 bytes, ~1μs) ✅
  - Run SQLite PRAGMA quick_check on meta.db open (~5-10ms) ✅
  - If corruption detected: log warning, offer `rfx index`, clear error message ✅
  - Performance tracking with std::time::Instant ✅
- **Performance guarantee:**
  - Cache load time: <20ms increase max ✅ (actual: 10-20ms)
  - Query latency: 0ms increase (ZERO overhead) ✅
- **Detection coverage:** 95%+ common corruption (truncated files, missing headers, corrupted SQLite) ✅
- **Implementation details:**
  - Location: src/cache.rs:275-417 (validate() method)
  - Tests: src/cache.rs:1827-1906 (4 comprehensive corruption tests)
  - All tests passing: corrupted database, corrupted trigrams, corrupted content, missing schema
- **Acceptance criteria:**
  - No measurable query latency impact ✅
  - Detect most common corruption cases ✅
  - Graceful fallback when corruption found ✅

**Cache Compaction:** ✅ COMPLETED (2025-11-16)
- **Implementation:** ✅ FULLY IMPLEMENTED
  - Command: `rfx index compact` ✅
  - Removes deleted file entries from meta.db ✅
  - Runs SQLite VACUUM to reclaim space ✅
  - Rebuilds trigrams.bin and content.bin without deleted file data ✅
- **Acceptance criteria:** ✅ ALL MET
  - Achieves 20%+ size reduction on large codebases with file churn ✅
  - Safe operation (transactional, with error handling) ✅
- **Implementation details:**
  - Location: src/cache.rs (compact() method)
  - Model: src/models.rs:392-401 (CompactionReport struct)
  - Returns: CompactionReport (files_removed, space_saved_bytes, duration_ms)
  - Test script: /tmp/test_compaction.sh (manual validation)
  - Status: Fully operational and tested ✅

**Status:** ✅ ALL CACHE IMPROVEMENTS COMPLETED (2025-11-16)
- Automatic Cache Invalidation ✅
- Corruption Detection ✅
- Cache Compaction ✅

---

### 2. Indexer Module (`src/indexer.rs`)

#### P0: Directory Walking & File Discovery ✅ COMPLETED
- [x] **Implement directory tree walking** (indexer.rs:193-216)
  - Use `ignore` crate to respect `.gitignore` ✅
  - Filter by configured include/exclude patterns ✅
  - Handle symlinks according to config ✅
  - Collect all eligible source files ✅

- [x] **Implement file filtering** (indexer.rs:219-244)
  - Check file extensions against supported languages ✅
  - Respect max file size limits ✅
  - Skip binary files and generated code (via ignore crate) ✅
  - TODO: Custom include/exclude glob patterns (planned)

#### P0: Incremental Indexing ✅ COMPLETED
- [x] **Implement hash-based change detection** (indexer.rs:82-113)
  - Compute blake3 hash for each file ✅
  - Compare with `hashes.json` to detect changes ✅
  - Skip unchanged files (incremental indexing) ✅
  - Preserve symbols from unchanged files ✅

- [x] **Update hash storage** (indexer.rs:180)
  - Track all indexed file hashes ✅
  - Call `cache.save_hashes()` after indexing ✅
  - Handle deleted files (remove from hash map) ✅

#### P0: Tree-sitter Integration ✅ MOSTLY COMPLETE
- [x] **Set up Tree-sitter grammar dependencies** (Cargo.toml:26-35)
  - Add `tree-sitter-rust` to Cargo.toml ✅
  - Add `tree-sitter-python` to Cargo.toml ✅
  - Add `tree-sitter-javascript` to Cargo.toml ✅
  - Add `tree-sitter-typescript` to Cargo.toml ✅
  - Add `tree-sitter-go` to Cargo.toml ✅
  - Add `tree-sitter-php` to Cargo.toml ✅
  - Add `tree-sitter-c` to Cargo.toml ✅
  - Add `tree-sitter-cpp` to Cargo.toml ✅
  - Add `tree-sitter-java` to Cargo.toml ✅

- [x] **Implement language-specific parsers** (src/parsers/)
  - Create `ParserFactory` wrapper that selects grammar by language ✅
  - Parse file into AST using Tree-sitter ✅
  - Handle parse errors gracefully ✅
  - **Rust parser complete** (parsers/rust.rs) ✅
  - **Other languages:** Stub implemented, ready for expansion

- [x] **Implement AST traversal & symbol extraction** (Rust only)
  - **Goal:** Extract ALL symbol types that Tree-sitter can identify for each language
  - **Approach:** Traverse the complete AST and identify every node that represents a searchable code entity
  - **For each language, extract (examples, not exhaustive):**
    - **Rust:** fn, struct, enum, trait, impl, const, static, mod, macro, type alias, associated types, generic parameters, lifetimes, doc comments
    - **Python:** def, class, async def/class, lambda, decorators, class/static methods, properties, docstrings, type hints, comprehensions
    - **TypeScript/JavaScript:** function, class, interface, type, const, let, var, arrow functions, async/generator functions, methods, getters/setters, JSDoc comments, type parameters
    - **Go:** func, type, struct, interface, const, var, package-level declarations, methods, embedded types
    - **PHP:** function, class, interface, trait, enum (PHP 8.1+), abstract/final classes, namespaces, methods, properties, constants, magic methods, anonymous classes, PHPDoc comments
    - **C:** function declarations/definitions, struct, enum, union, typedef, static/extern variables, macros (#define), preprocessor directives, Doxygen comments
    - **C++:** function, class, struct, namespace, template (class/function), enum/enum class, using declarations, constructors/destructors, operators, virtual/override methods, friend declarations, Doxygen comments
    - **Java:** class, interface, enum, record (Java 14+), annotation, method, field, constructor, static/abstract/final modifiers, packages, inner/anonymous classes, JavaDoc comments
  - **For ALL languages:**
    - Capture complete symbol metadata (visibility, modifiers, parameters, return types)
    - Extract associated documentation/comments
    - Track scope hierarchy and fully-qualified names
    - Handle language-specific features (generics, annotations, etc.)
  - **Note:** The lists above are starting points. Implementers should examine Tree-sitter grammar documentation for each language and extract ALL relevant node types.

- [ ] **Compute symbol spans** (Line 93)
  - Extract start/end line and column from AST nodes
  - Store as `Span { start_line, start_col, end_line, end_col }`

- [ ] **Extract scope context** (Line 93)
  - Track parent scope (e.g., "impl MyStruct", "class User")
  - Build fully-qualified symbol names
  - Handle nested scopes (modules, classes, etc.)

#### P1: Token Extraction for Lexical Search
- [ ] **Implement token extraction** (Line 43)
  - Tokenize source code (identifiers, keywords, strings)
  - Build n-gram index for fuzzy matching
  - Compress tokens with zstd
  - Write to `tokens.bin`

#### P1: Cache Writing ✅ COMPLETED
- [x] **Write trigrams and content to cache** (indexer.rs:295-314)
  - Write trigram index to `trigrams.bin` (rkyv) ✅
  - Write content store to `content.bin` ✅
  - NO symbol extraction or tree-sitter parsing during indexing ✅

- [x] **Update metadata** (indexer.rs:182-183)
  - Write statistics to `meta.db` ✅
  - Update timestamp, file counts ✅

#### P1: Future-Proof Symbol Extraction
- [ ] **Implement generic fallback for unknown symbol types**
  - Design symbol extraction to handle AST nodes gracefully even if not explicitly recognized
  - Use heuristics: nodes with "name" fields, declaration-pattern node kinds, etc.
  - Classify unknown symbols with generic types (e.g., `SymbolKind::Unknown`)
  - Extract basic metadata (name, span, scope) even without language-specific handling
  - **Goal:** Reflex should work with newer language versions without crashing or missing symbols entirely

- [ ] **Add language version tracking**
  - Track which Tree-sitter grammar version was used during indexing
  - Store in meta.db for debugging and compatibility checks
  - Log warnings when grammar versions change between index/query

#### P2: Advanced Indexing Features

**Parallel File Parsing with Rayon:** ✅ COMPLETED
- **Implementation:** ✅ FULLY IMPLEMENTED
  - Uses rayon::par_iter() to parse multiple files concurrently during indexing ✅
  - Parallel trigram extraction and symbol parsing ✅
  - Thread-safe accumulation of results ✅
  - Custom thread pool with configurable parallelism ✅
- **Acceptance criteria:** ✅ ALL MET
  - Achieves 2-3x indexing speedup on multi-core systems ✅
  - No race conditions or data corruption ✅
- **Implementation details:**
  - Location: src/indexer.rs
  - Lines: 9 (rayon import), 313-329 (ThreadPoolBuilder, parallel iteration)
  - Dependency: rayon = "1.10", num_cpus = "1.16" (Cargo.toml:64-65)
  - Thread pool: 80% of available cores by default
  - Status: Fully operational and tested ✅

**Progress Reporting:** ✅ COMPLETED
- **Implementation:** ✅ FULLY IMPLEMENTED
  - Live progress bar during indexing: `[████░░░] 234/567 files (src/main.rs)` ✅
  - Integration with interactive mode status display ✅
  - Optional --quiet flag to suppress progress ✅
- **Acceptance criteria:** ✅ ALL MET
  - Real-time updates (every 50ms or 10 files, whichever is less frequent) ✅
  - Clean terminal handling (no flickering) ✅
- **Implementation details:**
  - Location: src/indexer.rs (progress bar rendering)
  - Also used in: src/cli.rs, src/dependency.rs, src/background_indexer.rs
  - Dependency: indicatif = "0.17" (Cargo.toml:93)
  - Features: Progress bars, spinners, multi-progress for concurrent operations
  - Status: Fully operational and tested ✅

**Streaming Parse for Huge Files:**
- **Implementation:**
  - Handle files >10MB without loading entirely into memory
  - Chunk-based processing (read in 1MB chunks)
  - Stream trigrams and content to cache
- **Acceptance criteria:**
  - Successfully index files up to 100MB
  - Memory usage <50MB regardless of file size
- **Estimated time:** 1-2 days

**Generic Fallback for Unknown AST Nodes:**
- **Implementation:**
  - Extract symbols from unrecognized tree-sitter nodes
  - Use heuristics: nodes with "name" fields, declaration patterns
  - Classify unknown symbols with SymbolKind::Unknown
  - Extract basic metadata (name, span, scope) even without language-specific handling
- **Goal:** Future-proof against language evolution (new syntax won't crash Reflex)
- **Acceptance criteria:**
  - Handle new language features gracefully
  - Extract basic info even for unrecognized nodes
- **Estimated time:** 2 days

**Language Version Tracking:**
- **Implementation:**
  - Store tree-sitter grammar version in meta.db
  - Log warnings when grammar versions change between index/query
  - Display in `rfx stats` output
- **Acceptance criteria:**
  - Version mismatches logged clearly
  - Help debug parsing issues
- **Estimated time:** 1 day

**Status:** ✅ CORE INDEXING ENHANCEMENTS COMPLETED
- Parallel File Parsing with Rayon ✅
- Progress Reporting ✅

**Remaining (Lower Priority):**
- Streaming Parse for Huge Files (P3)
- Generic Fallback for Unknown AST Nodes (P3)
- Language Version Tracking (P3)

---

### 3. Query Engine Module (`src/query.rs`)

#### P0: Cache Loading ✅ COMPLETED
- [x] **Load memory-mapped cache** (query.rs:72-74, 195-197)
  - Memory-map `trigrams.bin` on query start (TrigramIndex) ✅
  - Load `content.bin` for full-text search (ContentReader) ✅
  - Keep file handles open for duration of query ✅

#### P0: Query Pattern Parsing ✅ COMPLETED
- [x] **Implement query pattern parser** (query.rs:77-99)
  - Parse plain text for full-text search ✅
  - Parse `--symbols` flag for symbol-only search ✅
  - Parse `*` wildcard for prefix/substring matching ✅
  - **Note:** `symbol:name` syntax handled via CLI flags instead

#### P0: Symbol Search ✅ COMPLETED
- [x] **Implement symbol name matching** (query.rs:246-318)
  - Runtime tree-sitter parsing of candidate files ✅
  - Exact match: `--exact` flag ✅
  - Prefix match: `pattern*` ✅
  - Substring match: default behavior ✅
  - Uses trigram search to narrow candidates before parsing ✅

#### P1: AST Pattern Matching ✅ COMPLETED
- [x] **Implement Tree-sitter query support** (src/ast_query.rs, 428 lines)
  - Parse Tree-sitter S-expression patterns ✅
  - Match patterns at query time using Tree-sitter queries ✅
  - Support patterns like `(function_item name: (identifier) @name)` ✅
  - Integration with query pipeline (Phase 2 enrichment) ✅
  - 4 comprehensive tests (functions, structs, invalid patterns, unsupported languages) ✅
  - Supported languages: Rust, TypeScript, JavaScript, PHP ✅

#### P1: Lexical Search ✅ COMPLETED (via Trigram)
- [x] **Implement trigram-based full-text search** (query.rs:192-264)
  - Search trigram index for candidate files ✅
  - Verify matches in actual content ✅
  - Return matches with context ✅
  - **Note:** Replaces token-based search with trigrams

#### P0: Filtering & Ranking ✅ COMPLETED
- [x] **Apply query filters** (query.rs:102-119)
  - Filter by language (if specified) ✅
  - Filter by symbol kind (if specified) ✅
  - Filter by file path pattern ✅
  - Apply limit to result count ✅

- [x] **Implement deterministic ranking** (query.rs:146-149)
  - Sort by file path (lexicographic) ✅
  - Sort by line number within file ✅
  - Ensure consistent ordering across runs ✅

#### P1: Result Context & Preview ✅ COMPLETED
- [x] **Generate code previews** (query.rs:121-143, content_store.rs:301-340)
  - Extract context around match ✅
  - Include full symbol body with `--expand` flag ✅
  - Format as clean, readable snippet ✅

#### P2: Advanced Query Features ✅ REGEX COMPLETE
- [x] Support regex patterns (src/regex_trigrams.rs, query.rs:search_with_regex) ✅
  - Literal extraction from regex (≥3 chars)
  - Trigram-based candidate narrowing
  - Union-based file selection for correctness
  - Falls back to full scan when no literals found
  - CLI: `--regex` or `-r` flag
  - Comprehensive test coverage (13 tests)
- [x] Support wildcard patterns (`*`, `?`) - implemented via regex ✅

#### P2: Advanced Query Features

**Query Result Caching:**
- **Implementation:**
  - Cache frequent queries in memory (LRU cache with 100-entry limit)
  - Key: query pattern + filters hash
  - Value: Vec<SearchResult>
  - Clear cache on index update
- **Acceptance criteria:**
  - <1ms for cached queries (vs 10-100ms for fresh queries)
  - Memory overhead <10MB for 100 cached queries
  - Automatic invalidation on reindex
- **Estimated time:** 2 days

**Optional Relevance Scoring:**
- **Implementation:**
  - Rank results by relevance score with `--ranked` flag
  - Scoring factors:
    - Exact match > prefix match > substring match (weight: 3:2:1)
    - File importance (based on import count from dependency graph)
    - Recency (recently modified files scored higher)
  - Deterministic tie-breaking: lexicographic by file path + line number
- **Acceptance criteria:**
  - More relevant results appear first
  - Deterministic ordering (same query = same result order)
  - <5ms scoring overhead per query
- **Estimated time:** 2-3 days

**Total Query Features:** 4-5 days

---

### 4. HTTP Server (`src/cli.rs`) ✅ COMPLETED

#### P1: Axum Server Setup ✅ COMPLETED
- [x] **Implement HTTP server** (cli.rs:428-687)
  - Create axum router with routes ✅
  - Bind to configured host:port ✅
  - Handle graceful shutdown (Ctrl+C) ✅
  - **Status:** Fully implemented and tested

#### P1: API Endpoints ✅ COMPLETED
- [x] **GET /query** endpoint ✅
  - Query parameters: `q` (pattern), `lang`, `kind`, `limit`, `symbols`, `regex`, `exact`, `expand`, `file`
  - Return QueryResponse (status, results, warnings) as JSON
  - Handle errors with proper HTTP status codes
  - Tested with multiple filters and options

- [x] **GET /stats** endpoint ✅
  - Return IndexStats as JSON
  - Include cache size, file count, symbol count, language breakdowns
  - Returns 404 if index not found

- [x] **POST /index** endpoint ✅
  - Trigger reindexing
  - Accept optional body with IndexRequest (force, languages)
  - Return IndexStats as JSON
  - Synchronous indexing (returns after completion)

- [x] **GET /health** endpoint ✅
  - Simple health check endpoint
  - Returns "Reflex is running"

#### P2: Advanced Server Features
- [x] Add CORS support for browser clients ✅

**Request Logging Middleware:**
- **Implementation:**
  - Log all HTTP requests with timing
  - Structured logging (JSON format) with request ID, method, path, duration, status
  - Optional --log-requests flag for HTTP server
- **Acceptance criteria:**
  - All requests logged with <1ms overhead
  - Easy to parse logs for analytics
- **Estimated time:** 1 day

**Rate Limiting:**
- **Implementation:**
  - Per-IP rate limits (e.g., 100 req/min configurable)
  - Token bucket algorithm
  - Configurable thresholds via CLI flags or config file
  - HTTP 429 (Too Many Requests) response
- **Acceptance criteria:**
  - Prevents abuse without impacting legitimate users
  - Configurable limits
- **Estimated time:** 1 day

**API Authentication:**
- **Implementation:**
  - Optional API key authentication
  - Header-based: `Authorization: Bearer <key>`
  - Generate keys via `rfx server keygen`
  - Store key hashes in meta.db
- **Acceptance criteria:**
  - Secure authentication (bcrypt or argon2 for key hashing)
  - Optional (disabled by default)
- **Estimated time:** 1-2 days

**WebSocket Support:**
- **Implementation:**
  - `/ws/query` endpoint for real-time result streaming
  - Stream results as they're found (instead of waiting for all results)
  - JSON messages: { "type": "result", "data": SearchResult }
- **Acceptance criteria:**
  - Stream results with <10ms latency per result
  - Clean connection handling and cleanup
- **Estimated time:** 2-3 days

**Total Server Features:** 5-7 days

---

### 5. Tree-sitter Grammar Integration

#### Required Cargo.toml Additions ✅ COMPLETED
- [x] tree-sitter-rust = "0.23" ✅
- [x] tree-sitter-python = "0.23" ✅
- [x] tree-sitter-javascript = "0.23" ✅
- [x] tree-sitter-typescript = "0.23" ✅
- [x] tree-sitter-go = "0.23" ✅
- [x] tree-sitter-php = "0.23" ✅
- [x] tree-sitter-c = "0.23" ✅
- [x] tree-sitter-cpp = "0.23" ✅
- [x] tree-sitter-java = "0.23" ✅

#### Implementation Checklist ⚠️ PARTIALLY COMPLETE
- [x] Create `src/parsers/mod.rs` module ✅
- [x] Create `src/parsers/rust.rs` - Rust grammar integration ✅ **FULLY IMPLEMENTED**
- [x] Create `src/parsers/typescript.rs` - TS/JS grammar integration ✅ **FULLY IMPLEMENTED**
  - Shared TypeScript parser handles both .ts and .js files ✅
  - Full React/JSX support via TSX grammar ✅
- [x] Create `src/parsers/vue.rs` - Vue SFC grammar integration ✅ **FULLY IMPLEMENTED**
  - Supports both Options API and Composition API ✅
  - Handles TypeScript in `<script lang="ts">` blocks ✅
- [x] Create `src/parsers/svelte.rs` - Svelte component grammar integration ✅ **FULLY IMPLEMENTED**
  - Extracts symbols from component scripts ✅
  - Supports reactive declarations (`$:`) ✅
- [x] Create `src/parsers/php.rs` - PHP grammar integration ✅ **FULLY IMPLEMENTED**
  - Full symbol extraction (functions, classes, interfaces, traits) ✅
  - Methods with scope tracking (class/trait/interface) ✅
  - Properties and constants ✅
  - Namespaces and PHP 8.1+ enums ✅
- [x] Create `src/parsers/python.rs` - Python grammar integration ✅ **FULLY IMPLEMENTED**
  - Functions, classes, methods, async support ✅
  - Decorators, lambdas, constants ✅
  - 10 comprehensive tests ✅
- [x] Create `src/parsers/go.rs` - Go grammar integration ✅ **FULLY IMPLEMENTED**
  - Functions, types, interfaces, methods ✅
  - Constants, variables, packages ✅
  - 10 comprehensive tests ✅
- [x] Create `src/parsers/java.rs` - Java grammar integration ✅ **FULLY IMPLEMENTED**
  - Classes, interfaces, enums, methods ✅
  - Fields, constructors, annotations ✅
  - 12 comprehensive tests ✅
- [x] Create `src/parsers/c.rs` - C grammar integration ✅ **FULLY IMPLEMENTED**
  - Functions, structs, enums, unions ✅
  - Typedefs, global variables ✅
  - 8 comprehensive tests ✅
- [x] Create `src/parsers/cpp.rs` - C++ grammar integration ✅ **FULLY IMPLEMENTED**
  - Functions, classes, structs, namespaces ✅
  - Templates, methods, enums, type aliases ✅
  - 12 comprehensive tests ✅
- [x] Create `src/parsers/csharp.rs` - C# grammar integration ✅ **FULLY IMPLEMENTED**
  - Classes, interfaces, structs, enums, records ✅
  - Delegates, methods, properties, namespaces ✅
  - Comprehensive test coverage ✅
- [x] Create `src/parsers/ruby.rs` - Ruby grammar integration ✅ **FULLY IMPLEMENTED**
  - Classes, modules, methods, singleton methods ✅
  - Constants, blocks ✅
  - 8 comprehensive tests including Rails patterns ✅
- [x] Create `src/parsers/kotlin.rs` - Kotlin grammar integration ✅ **FULLY IMPLEMENTED**
  - Classes, objects, interfaces, functions ✅
  - Properties, data classes, sealed classes ✅
  - 10 comprehensive tests including Android patterns ✅
- [x] Create `src/parsers/zig.rs` - Zig grammar integration ✅ **FULLY IMPLEMENTED**
  - Functions, structs, enums, constants ✅
  - Variables, test declarations, error sets ✅
  - 10 comprehensive tests ✅
- [x] Implement parser factory (select parser by Language enum) ✅
- [x] Write unit tests for Rust parser (7 tests) ✅
- [x] Write unit tests for all parsers (52+ tests total) ✅
- [ ] Document query patterns for each language

---

### 6. Data Format Design

#### P0: Binary Format Design ✅ COMPLETED
- [x] **Design content.bin format** (content_store.rs)
  - Header: magic bytes "RFCT", version, num_files, index_offset ✅
  - File contents: Concatenated file contents ✅
  - File index: path, offset, length for each file ✅
  - Memory-mapped for zero-copy access ✅

- [x] **Design trigrams.bin format** ✅ COMPLETED
  - Binary format with header (magic, version, counts, offsets) ✅
  - Posting lists serialized with rkyv (zero-copy deserialization) ✅
  - File list with paths ✅
  - Memory-mapped for instant access (vs. rebuilding index on every query) ✅

- [x] **Removed symbols.bin** ✅ COMPLETED
  - Symbols no longer indexed during build phase ✅
  - Runtime tree-sitter parsing at query time instead ✅
  - Removed ~500 lines of serialization/deserialization code ✅

- [x] **Design meta.db schema** (cache.rs:74-139)
  - **Decision:** SQLite (easier, more flexible) ✅
  - Tables: files (path, hash, language, symbol_count) ✅
  - Tables: statistics (key-value for totals) ✅
  - Tables: config (key-value for settings) ✅

#### P1: Serialization Implementation ✅ COMPLETED
- [x] Implement serialization for all data structures (rkyv, serde_json) ✅
- [x] Implement deserialization with version compatibility ✅
- [ ] Add schema version migration support (planned)

---

### 7. Testing & Quality ✅ COMPLETED

**Status:** Comprehensive test suite implemented with **458 total tests** across unit, integration, and performance categories.

#### P0: Unit Tests ✅ COMPLETED (261+ tests)
Embedded in source files using `#[cfg(test)]` modules:

- [x] **CacheManager tests** (src/cache.rs: 29 tests)
  - Cache initialization, file creation
  - Hash persistence (load/save)
  - Statistics retrieval
  - Cache clearing and management

- [x] **Indexer tests** (src/indexer.rs: 24 tests)
  - File filtering by language and extension
  - Hash-based change detection
  - Incremental indexing
  - Directory walking with .gitignore support

- [x] **QueryEngine tests** (src/query.rs: 22 tests)
  - Pattern parsing (plain text, symbols, regex)
  - Filter application (language, kind, file pattern)
  - Result ranking and limiting
  - Symbol-only vs full-text search modes

- [x] **Parser tests** (130+ tests across 14 languages)
  - Rust parser (6 tests): functions, structs, enums, traits, impls
  - TypeScript parser (13 tests): functions, classes, interfaces, React components
  - Python parser (10 tests): functions, classes, async, decorators
  - Go parser (10 tests): functions, types, interfaces, methods
  - Java parser (12 tests): classes, interfaces, enums, annotations
  - C parser (10 tests): functions, structs, typedefs, unions
  - C++ parser (14 tests): classes, templates, namespaces, operators
  - PHP parser (10 tests): classes, traits, enums, namespaces
  - C# parser (9 tests): classes, interfaces, records, delegates, namespaces
  - Ruby parser (8 tests): classes, modules, methods, Rails patterns
  - Kotlin parser (10 tests): classes, objects, data classes, Android patterns
  - Zig parser (10 tests): functions, structs, enums, tests
  - Vue parser (4 tests): Composition API, Options API, TypeScript support
  - Svelte parser (4 tests): reactive declarations, module context

- [x] **Core module tests** (43+ tests)
  - Trigram indexing (8 tests): extraction, intersection, posting lists
  - Content store (4 tests): binary format, memory-mapping, context extraction
  - Regex trigrams (22 tests): literal extraction, optimization, fallback handling
  - AST queries (4 tests): Tree-sitter S-expression patterns, multi-language support
  - Git integration (src/git.rs): repository detection
  - File watcher (9 tests): debouncing, file changes, directory handling

#### P1: Integration Tests ✅ COMPLETED (17 tests)
Located in tests/integration_test.rs:

- [x] **Basic workflow tests** (3 tests)
  - Full workflow: index → query → verify
  - Cache initialization and existence checks
  - Cache clearing and recreation

- [x] **End-to-end workflow tests** (4 tests)
  - Full-text search workflow (multiple files, context matching)
  - Symbol search workflow (definitions vs call sites)
  - Regex search workflow (pattern matching with trigrams)
  - Incremental indexing workflow (detect and reindex only changed files)

- [x] **File modification workflow** (1 test)
  - Modify files and verify incremental reindex correctness

- [x] **Multi-language tests** (2 tests)
  - Multi-language indexing and search (Rust, TypeScript, Python, JavaScript)
  - Language-filtered search (isolate results by language)

- [x] **Complex query tests** (2 tests)
  - Combined filters (language + kind + file pattern)
  - Limit and sorting verification (deterministic ordering)

- [x] **Error handling tests** (3 tests)
  - Query without index (should fail gracefully)
  - Index empty directory (should succeed with 0 files)
  - Search empty index (should return no results)

- [x] **Cache persistence tests** (2 tests)
  - Cache persists across sessions
  - Clear and rebuild workflow

#### P1: Performance Tests ✅ COMPLETED (10 tests)
Located in tests/performance_test.rs:

- [x] **Indexing performance** (3 tests)
  - Small codebase (100 files): <1s
  - Medium codebase (500 files): <3s
  - Incremental reindex (10/100 files changed): <1s

- [x] **Query performance** (4 tests)
  - Full-text query (200 files): <100ms ✅
  - Symbol query (100 files with runtime parsing): <5s
  - Regex query (150 files): <200ms
  - Filtered query (200 mixed-language files): <150ms

- [x] **Memory-mapped I/O performance** (1 test)
  - Repeated queries (10x) use cached index: <50ms average

- [x] **Scalability tests** (2 tests)
  - Large file handling (1000 lines): <500ms
  - Many small files (1000 files): <2s

#### P2: End-to-End Tests ✅ VALIDATED
- [x] Test on real-world codebases
  - Linux kernel (62K files): 124ms full-text, 224ms symbol search
  - Reflex codebase: 2-3ms all query types
- [x] Verify correctness of extracted symbols
  - Runtime symbol detection tested on candidate files
  - All 8 language parsers validated
- [x] Measure actual query performance
  - Instant for full-text on medium codebases ✅
  - 2-224ms range depending on codebase size and query type ✅

---

### 8. Documentation ✅ MOSTLY COMPLETE

**Status:** Core documentation complete. HTTP API and CONTRIBUTING.md deferred until those features are implemented.

#### P1: User Documentation ✅ COMPLETED
- [x] **Write comprehensive README.md** ✅
  - Installation instructions (build from source)
  - Quick start guide with examples
  - Complete CLI reference for all commands
  - Supported languages table
  - Architecture overview
  - Performance benchmarks
  - AI integration examples
  - Use cases and roadmap

- [x] **Write ARCHITECTURE.md** ✅
  - System design overview with diagrams
  - Core components (Cache, Indexer, Trigram, Query Engine, Parsers)
  - Data formats (trigrams.bin, content.bin, meta.db, hashes.json)
  - Indexing and query pipeline deep-dives
  - Runtime symbol detection explanation
  - Performance optimizations (memory-mapping, rkyv, blake3, trigrams)
  - Extension guide (adding new languages with step-by-step)
  - Testing strategy (221 tests documented)
  - Future architecture (HTTP server, AST patterns, MCP)

- [ ] **Write API.md** (deferred until HTTP server is implemented)
  - HTTP API reference
  - JSON response format
  - Error codes

#### P1: Developer Documentation ✅ COMPLETED
- [x] **Add rustdoc comments to all public APIs** ✅
  - src/lib.rs: Module-level documentation
  - src/models.rs: All public types documented (Span, SymbolKind, Language, SearchResult, etc.)
  - src/cache.rs: CacheManager methods fully documented
  - src/indexer.rs: Indexer workflow documented
  - src/query.rs: QueryEngine modes documented
  - src/parsers/: Parser implementations documented

- [x] **Document binary file formats** ✅
  - Detailed in ARCHITECTURE.md:
    - trigrams.bin format (rkyv serialization)
    - content.bin format (memory-mapped binary)
    - hashes.json format (JSON)
    - meta.db schema (SQLite)

- [ ] **Create developer setup guide** (included in README.md)
  - Build instructions in README.md
  - Testing commands in README.md
  - Development workflow in CLAUDE.md

- [ ] **Add CONTRIBUTING.md** (deferred - can be added when opening to external contributors)

**Completion Status**: 9/11 tasks complete (82%)

---

### 9. Tooling & Infrastructure ✅ MOSTLY COMPLETE

#### P1: Development Tools ✅ COMPLETED
- [x] **GitHub Actions workflow** ✅ COMPLETED
  - Files: .github/workflows/publish-packages.yml, release.yml
  - CI/CD: Automated testing, linting, building
  - Status: Fully operational ✅

- [ ] Add `cargo fmt` check to CI (can be added to existing workflows)
- [ ] Add `cargo clippy` check to CI (can be added to existing workflows)
- [ ] Add pre-commit hooks (optional, not critical)

#### P2: Release Engineering ✅ COMPLETED
- [x] **Set up cross-compilation for Linux, macOS, Windows** ✅ COMPLETED
  - Configuration: dist-workspace.toml
  - Platforms: 6 targets (Linux x64/ARM64/musl, macOS Intel/ARM, Windows x64)
  - Tool: cargo-dist for automated releases
  - Status: Fully operational ✅

- [x] **Create release binaries** ✅ COMPLETED
  - Automated via GitHub Actions release workflow
  - Extracts raw executables from cargo-dist archives
  - Produces: rfx binaries for all platforms
  - Status: Fully operational ✅

- [x] **Publish to crates.io** ✅ COMPLETED
  - Package: reflex-search
  - Current version: 0.8.1 (Cargo.toml:3)
  - Automated publishing via CI/CD
  - Status: Fully operational ✅

- [x] **Create installation script** ✅ COMPLETED
  - Shell installer: reflex-installer.sh
  - PowerShell installer: reflex-installer.ps1
  - Generated by cargo-dist
  - Status: Fully operational ✅

---

### 10. Future Work (P3, from CLAUDE.md)

#### Long-term Features
- [ ] `reflexd`: Background indexing daemon
- [x] **MCP (Model Context Protocol) Server** - ✅ COMPLETED (2025-11-03)
  - Implemented stdio MCP server (`rfx mcp` command)
  - Direct JSON-RPC protocol implementation (~470 lines, zero heavy deps)
  - Tools exposed: `search_code`, `search_regex`, `search_ast`, `index_project`
  - Configuration: Add to Claude Code's `claude_code_config.json`
  - Benefits: Zero port conflicts, automatic lifecycle, per-session isolation
  - Implementation: src/mcp.rs (clean, maintainable, no macro magic)
- [x] **File Watcher** - Auto-reindex on file changes ✅ COMPLETED (2025-11-03)
  - Implemented watch command (`rfx watch`)
  - Configurable debouncing (5-30 seconds, default: 15s)
  - Quiet mode for background operation
  - Respects .gitignore patterns automatically
  - 9 comprehensive tests
  - Implementation: src/watcher.rs (289 lines)
- [x] **Interactive Mode (TUI)** - Terminal-based query browser ✅ COMPLETED (2025-11-16)
  - **FULLY IMPLEMENTED** in src/interactive/ (12 files, ~2000+ lines of code) ✅
  - **DEFAULT MODE**: Launched when no command is given (src/cli.rs:1779-1783) ✅
  - Files implemented:
    - app.rs - Core application state and event handling
    - ui.rs - UI rendering and layout
    - terminal.rs - Terminal setup and cleanup
    - syntax.rs - Syntax highlighting integration
    - mouse.rs - Mouse event handling
    - theme.rs - Color themes and styling
    - mod.rs - Module interface
    - input.rs - Text input handling
    - history.rs - Command history management
    - filter_selector.rs - Filter UI components
    - results.rs - Results display and navigation
    - effects.rs - Terminal effects and animations
  - **Features implemented:** ✅
    - Interactive query session with live result browsing ✅
    - Query input with autocomplete ✅
    - Scrollable results with expand/collapse code blocks ✅
    - Keyboard and mouse navigation (up/down, page up/down, expand/collapse) ✅
    - Live filtering and result refinement ✅
    - Session history and command recall ✅
    - Syntax highlighting for multiple languages ✅
    - Visual effects and animations (tachyonfx) ✅
  - **Dependencies:**
    - ratatui = "0.29" (terminal UI framework) - Cargo.toml:96 ✅
    - tachyonfx = "0.8" (terminal effects) - Cargo.toml:97 ✅
    - crossterm = "0.28" (cross-platform terminal) - Cargo.toml:98 ✅
    - syntect = "5.2" (syntax highlighting) - Cargo.toml:99 ✅
    - owo-colors = "4.1" (color formatting) - Cargo.toml:101 ✅
  - **Status:** Production-ready, fully operational ✅
- [ ] **Semantic Query Building** - Natural language to Reflex query translation
  - **Architecture:** External LLM APIs (no local models)
  - **Key insight:** No code understanding needed - pure NL→rfx query command mapping
  - **User-provided API keys** via environment variables or ~/.reflex/config.toml
  - **Supported providers:**
    - OpenAI (GPT-4o, GPT-4o-mini) - OPENAI_API_KEY
    - Anthropic Claude (Claude 3.5 Sonnet) - ANTHROPIC_API_KEY
    - Groq (llama-3.3-70b-versatile) - GROQ_API_KEY
  - **Configuration example** (~/.reflex/config.toml):
    ```toml
    [semantic_query]
    provider = "openai"  # or "anthropic", "groq"
    model = "gpt-4o-mini"
    api_key_env = "OPENAI_API_KEY"  # optional, defaults to standard env var
    ```
  - **CLI interface:**
    - `rfx ask "Find all error handlers"`
    - Translates NL → `rfx query "Result" --symbols --kind function`
    - Executes query automatically and returns results
    - Option: `--explain` to show generated command without executing
  - **Implementation:**
    - Few-shot prompting with 8-10 examples (see src/semantic/prompt_template.md)
    - HTTP client for API requests (reqwest)
    - Prompt template system for provider-specific formats
    - Multi-query support: generate multiple queries, merge results, deduplicate
  - **Benefits:**
    - No model downloads required
    - Users choose their preferred provider
    - Always uses latest/best models
    - Minimal maintenance (no model versioning)
  - **Use case:** "Find all error handlers" → `rfx query "Result" --symbols --kind function`
  - **Estimated time:** 3-4 days
- [ ] LSP (Language Server Protocol) adapter
- [ ] Graph queries (imports/exports, call graph)
- [ ] Branch-aware context diffing (`--since`, `--branch`)
- [ ] Binary protocol for ultra-low-latency queries
- [ ] Plugin system for custom languages

---

## 11. Dependency Tracking Module (`src/dependency.rs`, `docs/DEPENDENCIES.md`)

**Status:** ✅ FULLY IMPLEMENTED
**Spec Document:** [docs/DEPENDENCIES.md](../docs/DEPENDENCIES.md)

### Overview

Dependency tracking enables Reflex to understand and expose file relationships (imports, includes, usage) across all 18 supported languages. This provides:
- **Impact Analysis**: What breaks if I change this file?
- **Context Understanding**: What does this code depend on?
- **Reverse Lookup**: What uses this file/module?
- **Architecture Analysis**: Circular dependencies, hotspots, orphaned files

**Key Architecture Decision:** **Index shallow, traverse deep**
- Store only direct (depth-1) dependencies in SQLite
- Compute deeper relationships on-demand via graph traversal
- Provides O(n) storage instead of O(n²) while enabling any-depth queries

### P1: Core Infrastructure (High Priority) ✅ COMPLETED
**Goal:** Basic dependency extraction and storage

- [x] **Add `file_dependencies` table to SQLite schema** (src/cache.rs)
  - Columns: `id`, `file_id`, `imported_path`, `resolved_file_id`, `import_type`, `line_number`, `imported_symbols`
  - Indexes: `idx_deps_file`, `idx_deps_resolved`, `idx_deps_type`
  - Foreign keys to `files` table
  - See docs/DEPENDENCIES.md for complete schema

- [x] **Create `DependencyExtractor` trait** (src/parsers/mod.rs)
  - Define interface for extracting imports from AST
  - Method: `extract_dependencies(path: &str, source: &str) -> Result<Vec<Dependency>>`
  - Returns: import path, resolved path, type (internal/external/stdlib), line number

- [x] **Implement `DependencyIndex` struct** (src/dependency.rs, NEW FILE)
  - Store dependencies: `insert_dependency(file_id, import, type, line)`
  - Retrieve dependencies: `get_dependencies(file_id) -> Vec<Dependency>`
  - Reverse lookup: `get_dependents(file_id) -> Vec<FileId>`
  - Traversal: `get_transitive_deps(file_id, depth) -> Vec<Dependency>`

- [x] **Extend `SearchResult` model** (src/models.rs)
  - Add optional `dependencies` field: `Option<Vec<DependencyInfo>>`
  - DependencyInfo struct: `path`, `type` (internal/external/stdlib)
  - Serialize to JSON for API output

- [x] **Add `--dependencies` flag to `rfx query` command** (src/cli.rs)
  - CLI flag: `--dependencies` or `--deps`
  - Enriches search results with immediate (depth-1) dependencies
  - Example: `rfx query "ApiClient" --dependencies --json`

- [x] **Tests: Basic storage and retrieval**
  - Test: Insert and retrieve dependencies
  - Test: Reverse lookup (get dependents)
  - Test: Handle non-existent files
  - Test: Performance (<5ms per lookup)

**Estimated Time:** 3-5 days

### P1: Language Parser Updates (High Priority) ✅ COMPLETED
**Goal:** Extract imports for all 18 languages using Tree-sitter

Each language parser (src/parsers/*.rs) needs to be extended with import extraction:

#### Tier 1 - Simple Import Syntax (1-2 days each)
- [x] **Rust** (src/parsers/rust.rs)
  - Extract: `use`, `mod`, `extern crate`
  - Resolve: `crate::`, `super::`, `self::`
  - Classify: stdlib (`std::`), internal (`crate::`), external (crates)
  - Tree-sitter nodes: `use_declaration`, `mod_item`, `extern_crate_declaration`

- [x] **Python** (src/parsers/python.rs)
  - Extract: `import`, `from...import`
  - Resolve: relative (`.`, `..`), absolute
  - Classify: stdlib, external (site-packages), internal (relative/local)
  - Tree-sitter nodes: `import_statement`, `import_from_statement`

- [x] **JavaScript/TypeScript** (src/parsers/typescript.rs)
  - Extract: `import`, `require()`, dynamic `import()`
  - Resolve: relative (`./`, `../`), aliases (`@/`), node_modules
  - Classify: stdlib (`fs`, `path`), external (node_modules), internal
  - Tree-sitter nodes: `import_statement`, `import_clause`, `call_expression`

- [x] **Go** (src/parsers/go.rs)
  - Extract: `import`
  - Resolve: fully qualified paths (no ambiguity)
  - Classify: stdlib, external (domain-based), internal (module-relative)
  - Tree-sitter nodes: `import_declaration`, `import_spec`

- [x] **Java** (src/parsers/java.rs)
  - Extract: `import`, `import static`
  - Resolve: fully qualified package names (no ambiguity)
  - Classify: stdlib (`java.*`, `javax.*`), external, internal (project packages)
  - Tree-sitter nodes: `import_declaration`

- [x] **C#** (src/parsers/csharp.rs)
  - Extract: `using`, `using static`
  - Resolve: namespace paths (fully qualified)
  - Classify: stdlib (`System.*`), external (NuGet), internal (project namespaces)
  - Tree-sitter nodes: `using_directive`

#### Tier 2 - Path Resolution Required (2-3 days each)
- [x] **Ruby** (src/parsers/ruby.rs)
  - Extract: `require`, `require_relative`, `load`
  - Resolve: gem paths, relative paths, `$LOAD_PATH`
  - Classify: stdlib, gems, internal (relative)
  - Tree-sitter nodes: `call` (require/require_relative/load)

- [x] **PHP** (src/parsers/php.rs)
  - Extract: `use`, `require`, `include`, `require_once`, `include_once`
  - Resolve: namespaces, relative paths, Composer autoload
  - Classify: stdlib, external (vendor/), internal (project)
  - Tree-sitter nodes: `namespace_use_declaration`, `require_expression`, `include_expression`

- [x] **C/C++** (src/parsers/c.rs, src/parsers/cpp.rs)
  - Extract: `#include <...>`, `#include "..."`
  - Resolve: header search paths, system includes vs local
  - Classify: stdlib (angle brackets + system paths), internal (quotes)
  - Tree-sitter nodes: `preproc_include`
  - **Challenge:** Requires compiler search path configuration

- [x] **Kotlin** (src/parsers/kotlin.rs)
  - Extract: `import`, `import ... as`
  - Resolve: fully qualified package names, wildcards
  - Classify: stdlib (`kotlin.*`), external (Maven/Gradle), internal (project)
  - Tree-sitter nodes: `import_header`

#### Tier 3 - Framework-Specific (2-3 days each)
- [x] **Vue** (src/parsers/vue.rs)
  - Extract: `import` statements from `<script>` blocks
  - Handle both Options API and Composition API
  - Support `<script setup>` syntax
  - Resolve: component imports, `@/` aliases
  - **Note:** Use existing line-based parsing approach

- [x] **Svelte** (src/parsers/svelte.rs)
  - Extract: `import` statements from `<script>` blocks
  - Handle module context (`context="module"`)
  - Resolve: component imports, relative paths
  - **Note:** Use existing line-based parsing approach

- [x] **Zig** (src/parsers/zig.rs)
  - Extract: `@import("...")`
  - Resolve: stdlib (`std`), packages, relative paths
  - Classify: stdlib, external, internal
  - Tree-sitter nodes: `call_expression` (with `@import`)

**Deliverables per Language:**
- Import extraction function added to parser
- Tree-sitter queries for import nodes
- Path classification (internal/external/stdlib)
- Unit tests (multiline imports, comments, edge cases)

**Estimated Time:** 12-15 days total (can be parallelized)

### P1: Query Engine Integration (High Priority) ✅ COMPLETED
**Goal:** Augment search results with dependency information

- [x] **Add dependency loading to `QueryEngine`** (src/query.rs)
  - Load `DependencyIndex` alongside `TrigramIndex` and `ContentReader`
  - Method: `load_dependencies() -> Result<DependencyIndex>`

- [x] **Implement `enrich_with_dependencies()` for results** (src/query.rs)
  - After search results found, enrich each with dependencies
  - Lookup dependencies by file_id
  - Add to `SearchResult.dependencies` field
  - Keep overhead minimal (<2ms per result)


- [x] **Optimize queries with indexes** (src/cache.rs)
  - Ensure `idx_deps_file` and `idx_deps_resolved` are used
  - Benchmark: <5ms overhead per enriched result

- [x] **Tests: Query enrichment correctness**
  - Test: Results include dependencies when flag set
  - Test: Performance (query + enrichment <10ms)

**Estimated Time:** 2-3 days

### P2: Dependency Command (Medium Priority) ✅ COMPLETED
**Goal:** Dedicated `rfx deps` command for graph analysis

- [x] **Create `rfx deps` CLI command** (src/cli.rs)
  - New subcommand: `rfx deps [FILE] [OPTIONS]`
  - Operates on single files OR entire graph

- [x] **Implement single-file operations**
  - `rfx deps <file>`: Show dependencies of file (default: depth 1)
  - `rfx deps <file> --reverse`: Show what depends on this file
  - `rfx deps <file> --depth N`: Traverse N levels deep
  - Traversal algorithm: BFS/DFS with cycle detection

- [x] **Implement tree visualization format** (src/formatter.rs)
  - ASCII tree rendering: `├──`, `└──`, `│`
  - Indentation shows depth
  - Example:
    ```
    src/main.rs
    ├── src/config.rs
    │   └── src/env.rs
    ├── src/api.rs
    │   ├── reqwest (external)
    │   └── src/models/user.rs
    └── std::collections (stdlib)
    ```

- [x] **Add JSON output format** (src/formatter.rs)
  - Structured JSON for programmatic use
  - Schema: `{ path, dependencies: [{ path, type, line }] }`


- [x] **Tests: Single-file operations**
  - Test: Basic dependency listing
  - Test: Reverse lookup
  - Test: Depth traversal (1, 2, 3 levels)
  - Test: Tree format rendering
  - Test: JSON output structure

**Estimated Time:** 3-4 days

### P2: Graph Algorithms (Medium Priority) ✅ COMPLETED
**Goal:** Advanced graph analysis features

- [x] **Implement circular dependency detection** (src/dependency.rs)
  - Algorithm: DFS with cycle detection
  - Detect all cycles in the dependency graph
  - Report cycle paths: A → B → C → A

- [x] **Implement `rfx analyze --circular`** (src/cli.rs)
  - Run cycle detection on entire graph
  - Output: List of all circular dependencies found
  - Format: `file_a.rs ↔ file_b.rs` or `A → B → C → A`

- [x] **Implement `rfx analyze --hotspots`** (src/cli.rs)
  - Find most imported files (high in-degree)
  - SQL: `SELECT resolved_file_id, COUNT(*) FROM file_dependencies GROUP BY resolved_file_id ORDER BY count DESC`
  - Output: Ranked list with import counts

- [x] **Implement `rfx analyze --unused`** (src/cli.rs)
  - Find files with no incoming dependencies
  - Exclude entry points (main.rs, index.ts, etc.)
  - Candidates for deletion

- [x] **Implement `rfx analyze --islands`** (src/cli.rs)
  - Find disconnected components (files that don't import each other)
  - Algorithm: Connected components via BFS/DFS
  - Output: List of component groups

- [x] **Implement `rfx analyze` summary report** (src/cli.rs)
  - Run all analyses: circular, hotspots, unused, islands
  - Generate comprehensive report
  - Example output:
    ```
    Dependency Analysis Report:
    ===========================
    Total Files: 1,234
    Total Dependencies: 5,678

    Circular Dependencies: 2 cycles found
      - src/user.rs ↔ src/profile.rs
      - src/api.rs → src/db.rs → src/models.rs → src/api.rs

    Hotspots (Top 5):
      - src/config.rs (47 imports) ⚠️
      - src/utils.rs (33 imports)
      - src/types.rs (28 imports)

    Unused Files: 5 found
      - src/old/legacy.rs
      - src/temp/test.rs

    Islands: 1 disconnected component (3 files)
    ```

- [x] **Tests: Graph algorithms correctness**
  - Test: Cycle detection (simple, complex, no cycles)
  - Test: Hotspot ranking
  - Test: Unused file detection
  - Test: Island detection
  - Test: Performance on large graphs (10k files)

**Estimated Time:** 3-4 days

### P3: Advanced Features (Low Priority) ⚠️ NOT STARTED
**Goal:** Visualization and advanced path resolution

- [ ] **Implement `--format dot` (Graphviz)** (src/formatter.rs)
  - Generate DOT format for visualization tools
  - Example: `rfx deps src/main.rs --format dot | dot -Tpng > graph.png`
  - Node styling: color by type (internal/external/stdlib)
  - Edge labels: show imported symbols

- [ ] **Advanced path resolution** (src/dependency.rs)
  - JavaScript/TypeScript: Parse `package.json` for path aliases
  - Python: Check virtual environment, `sys.path`
  - Rust: Parse `Cargo.toml` for workspace members
  - Ruby: Check `Gem` paths and `Bundler`

- [ ] **Session caching for repeated queries** (src/query.rs)
  - Cache transitive closures in memory during session
  - First query at depth 3: 50ms (computes closure)
  - Subsequent queries: 2ms (uses cache)
  - Clear cache on index update

- [ ] **Workspace/monorepo support** (src/dependency.rs)
  - Handle multi-package projects (npm workspaces, Cargo workspaces)
  - Resolve cross-package imports correctly
  - Visualize package boundaries

- [ ] **Tests: Advanced features**
  - Test: DOT format generation
  - Test: Path resolution accuracy
  - Test: Cache correctness
  - Test: Workspace import resolution

**Estimated Time:** 5-7 days

### Testing Strategy

#### Unit Tests (per component)
- [ ] **Dependency extraction** (per language parser)
  - Extract imports correctly
  - Handle multiline imports
  - Classify internal vs external vs stdlib
  - Resolve relative paths
  - Edge cases (comments, strings, docstrings)

- [ ] **DependencyIndex** (src/dependency.rs)
  - Store and retrieve dependencies
  - Reverse lookup (get dependents)
  - Transitive closure computation
  - Handle cycles gracefully
  - Performance (<5ms per query)

- [ ] **Graph algorithms** (src/dependency.rs)
  - Circular dependency detection (cycles present, no cycles)
  - Hotspot ranking (correct counts)
  - Unused file detection (exclude entry points)
  - Island detection (connected components)

#### Integration Tests
- [ ] **End-to-end workflow: Index with deps → Query with --dependencies**
  - Index Rust project
  - Query: `rfx query "ApiClient" --dependencies --json`
  - Verify: Results include dependencies
  - Verify: Dependencies are accurate

- [ ] **End-to-end workflow: `rfx deps` traversal**
  - Index multi-file project
  - Run: `rfx deps src/main.rs --depth 3`
  - Verify: Traversal is correct
  - Verify: No false positives

- [ ] **Incremental indexing with dependency updates**
  - Index project
  - Modify file (add/remove import)
  - Reindex incrementally
  - Verify: Dependency updates reflected

- [ ] **Cross-language dependencies**
  - TypeScript imports JavaScript
  - Python imports C extension
  - Verify: Dependencies tracked correctly

#### Performance Tests
- [ ] **Indexing overhead**
  - Baseline: Index 10k files without deps
  - With deps: Index 10k files with dep extraction
  - Target: <20% slowdown

- [ ] **Query enrichment performance**
  - Search: Find 50 results
  - Enrich: Add dependencies to each
  - Target: <2ms overhead per result

- [ ] **Reverse lookup performance**
  - File with 50 dependents
  - Query: `rfx deps file.rs --reverse`
  - Target: <5ms for lookup

- [ ] **Graph traversal performance**
  - Depth 3 traversal (50 files)
  - Target: <30ms
  - Graph-wide cycle detection (10k files)
  - Target: <200ms

#### Real-World Validation
- [ ] **Test on Reflex codebase** (Rust, ~5k LOC)
  - Verify: All Rust imports extracted
  - Verify: Cycle-free (or document intentional cycles)
  - Verify: Query performance <10ms

- [ ] **Test on Next.js project** (TypeScript/JavaScript, ~10k LOC)
  - Verify: ES6 imports, CommonJS requires
  - Verify: node_modules vs internal classification
  - Verify: Path alias resolution (`@/`)

- [ ] **Test on Django project** (Python, ~15k LOC)
  - Verify: Relative imports (`.`, `..`)
  - Verify: Django framework imports
  - Verify: Virtual environment packages

- [ ] **Test on mixed-language monorepo** (~50k LOC)
  - Multiple languages in one project
  - Verify: Cross-language imports (if applicable)
  - Verify: Performance at scale

### Documentation

- [x] **Complete specification document** (docs/DEPENDENCIES.md) ✅ COMPLETED
  - Architecture overview
  - Command interface
  - Language-specific import patterns
  - Implementation phases
  - Performance characteristics
  - Use cases and examples

- [x] **Update CLAUDE.md** (project instructions)
  - Add dependency tracking to feature list
  - Document CLI commands
  - Add to "Supported Features" section

- [x] **Update README.md** (user documentation)
  - Add `rfx query --dependencies` examples
  - Add `rfx deps` command documentation
  - Add use cases (refactoring, debugging, etc.)

- [x] **Update ARCHITECTURE.md** (technical docs)
  - Add DependencyIndex component
  - Document depth-1 storage model
  - Document graph traversal algorithms

- [x] **Rustdoc comments** (inline code documentation)
  - Document `DependencyExtractor` trait
  - Document `DependencyIndex` struct
  - Document graph algorithms

---

## 🔗 Dependency Graph

```
Cache File Formats (design) ──→ Cache Implementation ──→ Indexer & Query Engine
                                                      ↓
Tree-sitter Grammars ──────────→ AST Extraction ──────→ Symbol Indexing
                                                      ↓
                                                Query Execution
                                                      ↓
                                              HTTP Server (optional)
```

**Critical Path for MVP:**
1. Design and implement cache file formats
2. Integrate Tree-sitter grammars
3. Implement indexer with symbol extraction
4. Implement query engine with symbol search
5. Test on real codebases

---

## 📊 Current TODO Comments in Source

### src/cache.rs
- Line 43: Create initial cache files
- Line 84: Implement hash loading from hashes.json
- Line 91: Implement hash saving to hashes.json
- Line 97: Read actual stats from cache
- Line 108: Implement memory-mapped readers

### src/indexer.rs
- Line 36: Implement the actual indexing logic (7 steps)
- Line 64: Implement filtering logic (4 steps)
- Line 83: Implement Tree-sitter parsing (5 steps)

### src/query.rs
- Line 58: Implement query execution (6 steps)

### src/cli.rs
- Line 238: Implement HTTP server using axum (3 endpoints)

---

## 🚀 Suggested Implementation Order

### Phase 1: Foundation (Week 1)
1. Design binary file formats for cache
2. Implement CacheManager core functionality
3. Add Tree-sitter grammar dependencies
4. Set up parser module structure

### Phase 2: Indexing (Week 2-3)
1. Implement directory walking and file filtering
2. Implement Tree-sitter parsers for Rust (start with one language)
3. Implement symbol extraction for Rust
4. Write symbols to cache (basic format)
5. Implement incremental indexing with hashes
6. Test on small Rust project

### Phase 3: Querying (Week 3-4)
1. Implement memory-mapped cache readers
2. Implement symbol search (exact match)
3. Implement query filtering
4. Implement result preview generation
5. Test query performance

### Phase 4: Multi-Language Support (Week 4-6)
1. Add Python parser and symbol extraction
2. Add TypeScript/JavaScript parser
3. Add Go parser
4. Add PHP parser and symbol extraction
5. Add C parser and symbol extraction
6. Add C++ parser and symbol extraction
7. Add Java parser and symbol extraction
8. Test on polyglot projects

### Phase 5: Polish & Performance (Week 7-8)
1. Optimize query latency (target <100ms)
2. Add comprehensive tests for all languages
3. Write documentation
4. Implement HTTP server (optional)

---

## ✅ Completed Items

### Project Foundation
- [x] Project scaffolding (Cargo.toml, module structure)
- [x] CLI framework with all subcommands
- [x] Core data models (SearchResult, Span, Language, etc.)
- [x] Error handling setup (anyhow)
- [x] Logging setup (env_logger)
- [x] Basic integration test structure
- [x] .gitignore configuration
- [x] Dependency management (all required crates added)

### Tree-sitter Grammars (COMPLETED - All in Cargo.toml)
- [x] tree-sitter-rust = "0.23"
- [x] tree-sitter-python = "0.23"
- [x] tree-sitter-javascript = "0.23"
- [x] tree-sitter-typescript = "0.23"
- [x] tree-sitter-go = "0.23"
- [x] tree-sitter-php = "0.23"
- [x] tree-sitter-c = "0.23"
- [x] tree-sitter-cpp = "0.23"
- [x] tree-sitter-java = "0.23"

### Trigram Indexing (COMPLETED - src/trigram.rs)
- [x] Trigram extraction from text
- [x] Inverted index: trigram → file locations
- [x] Posting list intersection algorithms
- [x] File-based candidate search
- [x] Comprehensive tests (11 test cases)
- [x] Integration with indexer and query engine
- [x] Persistence to trigrams.bin (rkyv zero-copy serialization) ✅
- [x] Memory-mapped loading for instant access ✅

### Regex Trigram Support (COMPLETED - src/regex_trigrams.rs)
- [x] Extract literal sequences from regex patterns (≥3 chars)
- [x] Generate trigrams from extracted literals
- [x] Union-based file selection (correctness over performance)
- [x] Fallback to full scan when no literals present
- [x] Handle regex metacharacters and escapes
- [x] Support for alternation, quantifiers, groups
- [x] Case-insensitive flags: literals under `(?i)` are kept and looked up under every case variant (1.8.0; was a full scan)
- [x] Extractor soundness: alternation branch without a literal, optional group, char class, unknown escapes, `x` flag (1.8.0)
- [x] Comprehensive tests (13 test cases)
- [x] Integration with query engine (search_with_regex)

### Content Store (COMPLETED - src/content_store.rs)
- [x] Binary format design (magic bytes, header, index)
- [x] ContentWriter for building content.bin
- [x] ContentReader with memory-mapped I/O
- [x] Context extraction around matches
- [x] Comprehensive tests (5 test cases)
- [x] Integration with indexer and query engine

### Runtime Symbol Detection (COMPLETED - src/query.rs)
- [x] Trigram-based candidate file selection
- [x] Runtime tree-sitter parsing at query time
- [x] ParserFactory integration for multi-language support
- [x] Symbol filtering (name, kind, scope)
- [x] 2000x performance improvement over indexed symbols
- [x] Comprehensive validation on Linux kernel (62K files)

### Cache Infrastructure (COMPLETED - src/cache.rs)
- [x] Cache initialization (init())
- [x] Create meta.db with SQLite schema
- [x] Create empty symbols.bin with header
- [x] Create empty tokens.bin with header
- [x] Create hashes.json with empty map
- [x] Create default config.toml
- [x] Hash persistence (load_hashes, save_hashes)
- [x] Cache statistics (stats())
- [x] File metadata tracking (update_file)
- [x] List indexed files (list_files)

### Indexer (MOSTLY COMPLETE - src/indexer.rs)
- [x] Directory walking with .gitignore support (ignore crate)
- [x] File filtering by language and size
- [x] Hash-based incremental indexing (blake3)
- [x] Tree-sitter integration (Rust parser)
- [x] Trigram index building
- [x] Content store population
- [x] Symbol extraction and caching
- [x] Preserve unchanged files during incremental indexing

### Rust Parser (COMPLETED - src/parsers/rust.rs)
- [x] Parse functions (fn)
- [x] Parse structs
- [x] Parse enums
- [x] Parse traits
- [x] Parse impl blocks and methods
- [x] Parse constants
- [x] Parse modules
- [x] Parse type aliases
- [x] Extract spans (line/col)
- [x] Extract scope context
- [x] Comprehensive tests (7 test cases)

### TypeScript/JavaScript Parser (COMPLETED - src/parsers/typescript.rs)
- [x] Parse functions and arrow functions
- [x] Parse classes and methods
- [x] Parse interfaces and types
- [x] Parse enums
- [x] Parse React components (function and class)
- [x] Parse constants and variables
- [x] Shared parser for .ts, .tsx, .js, .jsx files
- [x] Full JSX/TSX support
- [x] Extract spans (line/col)
- [x] Extract scope context

### Vue Parser (COMPLETED - src/parsers/vue.rs)
- [x] Extract symbols from `<script>` blocks
- [x] Support both Options API and Composition API
- [x] Handle `<script setup>` syntax
- [x] Support TypeScript in `<script lang="ts">`
- [x] Line-based extraction (tree-sitter-vue incompatible with tree-sitter 0.24+)
- [x] Extract functions, constants, and methods
- [x] Extract spans (line/col)

### Svelte Parser (COMPLETED - src/parsers/svelte.rs)
- [x] Extract symbols from component scripts
- [x] Support reactive declarations (`$:`)
- [x] Handle module context (`context="module"`)
- [x] Support TypeScript in `<script lang="ts">`
- [x] Line-based extraction (tree-sitter-svelte incompatible with tree-sitter 0.24+)
- [x] Extract functions and variables
- [x] Extract spans (line/col)

### PHP Parser (COMPLETED - src/parsers/php.rs)
- [x] Parse functions (global functions)
- [x] Parse classes (regular, abstract, final)
- [x] Parse interfaces
- [x] Parse traits
- [x] Parse methods (with class/trait/interface scope tracking)
- [x] Parse properties (public, protected, private)
- [x] Parse constants (class and global)
- [x] Parse namespaces
- [x] Parse enums (PHP 8.1+)
- [x] Extract spans (line/col)
- [x] Extract scope context
- [x] Comprehensive tests (10 test cases)

### Query Engine (COMPLETED - src/query.rs)
- [x] Load memory-mapped cache (SymbolReader, ContentReader)
- [x] Symbol-only search mode (--symbols flag)
- [x] Trigram-based full-text search
- [x] Query pattern parsing (plain text, symbol:, prefix *)
- [x] Symbol name matching (exact, prefix, substring)
- [x] Filter by language
- [x] Filter by symbol kind
- [x] Filter by file path (substring)
- [x] Deterministic sorting (path → line number)
- [x] Result limit
- [x] Expand mode for full symbol bodies (--expand)

### CLI (MOSTLY COMPLETE - src/cli.rs)
- [x] index command (with --force, --languages, --progress)
- [x] query command (with all filters: --symbols, --lang, --kind, --json, --limit, --expand, --file, --exact, --regex/-r, --count)
- [x] stats command (with --json)
- [x] clear command (with --yes)
- [x] list-files command (with --json)
- [x] Verbose logging (-v, -vv, -vvv)
- [x] JSON output support across commands
- [x] Regex search support (--regex/-r flag) ✅

### Comprehensive Testing Suite (COMPLETED - 347 tests)
- [x] **Unit Tests** (261+ tests in src/ modules)
  - Cache: 29 tests (init, persistence, stats, clearing)
  - Indexer: 24 tests (filtering, hashing, incremental updates)
  - Query: 22 tests (pattern parsing, filtering, ranking)
  - Parsers: 130+ tests (Rust, TS, Python, Go, Java, C, C++, PHP, C#, Ruby, Kotlin, Zig, Vue, Svelte)
  - Core: 43+ tests (trigrams, content store, regex optimization, AST queries, file watcher)
- [x] **Integration Tests** (42 tests in tests/integration_test.rs)
  - Full workflows (index → query → verify)
  - Multi-language support and filtering
  - Incremental indexing and file modification
  - Error handling and edge cases
  - Cache persistence across sessions
- [x] **Performance Tests** (10 tests in tests/performance_test.rs)
  - Indexing speed (100-500 files: <1-3s)
  - Query latency (instant on 200+ files ✅)
  - Memory-mapped I/O efficiency
  - Scalability (large files, many files)
- [x] **Real-world validation**
  - Linux kernel (62K files): 124-224ms queries
  - Reflex codebase: 2-3ms queries
  - All performance targets met ✅

---

## 📝 Notes & Design Decisions

### Open Questions
1. **Cache format:** SQLite vs custom binary for meta.db?
   - SQLite: Easier, more flexible, built-in query support
   - Custom: Potentially faster, smaller, more control
   - **Recommendation:** Start with SQLite, optimize later if needed

2. **Symbol index structure:** B-tree vs hash table?
   - B-tree: Better for range queries, ordered iteration
   - Hash table: Faster for exact lookups
   - **Recommendation:** Hash table for symbol names, B-tree for file paths

3. **Compression:** When to compress?
   - Compress tokens.bin (high redundancy)
   - Don't compress symbols.bin (need random access)
   - **Recommendation:** Use zstd for tokens, raw binary for symbols

### Performance Targets
- Indexing: >10,000 files/sec on modern SSD
- Query: <100ms on 100k files (warm cache)
- Memory: <500MB for 100k files
- Cache size: <10% of source code size

### Maintenance & Updates

**Will Reflex need updates when languages evolve?**

Yes, but we can minimize the impact:

1. **Tree-sitter Grammar Dependencies**
   - Reflex depends on external Tree-sitter grammars (e.g., `tree-sitter-php = "0.23"`)
   - When languages add new syntax (PHP enums, Java records, etc.), grammars are updated
   - Reflex must periodically update grammar versions and test compatibility

2. **Future-Proofing Strategy**
   - **Explicit handling:** Common, stable symbols get full support with complete metadata
   - **Generic fallback:** Unknown/new symbols are still extracted with basic info (name, location, scope)
   - **Graceful degradation:** New language features won't crash Reflex, just may be classified generically
   - **Periodic updates:** Release Reflex updates when major language versions add significant new syntax

3. **Update Frequency**
   - **Minor updates:** Bug fixes, grammar version bumps (quarterly)
   - **Major updates:** New language support, significant syntax additions (as needed)
   - **Grammar updates are opt-in:** Users can update Cargo.toml to newer grammars independently

4. **Compatibility Promise**
   - Cache format versioning allows migration between Reflex versions
   - Older caches can be rebuilt with newer Reflex versions
   - Breaking changes will be clearly documented with migration guides

---

---

## 🔄 1.7.2 — MCP correctness release (2026-09-22)

**Trigger:** a field test of `rfx mcp` 1.7.0 inside Claude Code, against ripgrep on a
480k-line, 1027-file Rust monorepo. Four defects, three of which returned a **wrong
answer** with `status: "fresh"` and `can_trust_results: true`.

### What was wrong, and why

| # | Symptom | Root cause | Status |
| --- | --- | --- | --- |
| 1 | `verify_csrf` → 0 (rg: 89) | MCP hardcoded `use_contains: false` at 10 sites; the CLI `--contains` and HTTP `contains=` both plumbed it | ✅ done |
| 2 | Edited / added / deleted files reported `fresh` | Freshness sampled the mtimes of the **first ten** indexed files; untracked files are absent from that list and deleted ones fail `metadata()` | ✅ done |
| 3 | Raw `database is locked: Error code 5` for ~4 min | TWO locks: the symbol pass takes `.reflex/indexing.lock`, the indexer takes `.reflex/index.lock`. Neither knew about the other. No `busy_timeout` anywhere either. | ✅ done |
| 4 | `*.md` → 0 (rg: 3425) | `Language::Unknown.is_supported()` is false, and that gate decides what gets indexed | ✅ done |

### Decisions made

1. **Honest staleness over auto-refresh.** The handoff preferred auto-indexing before
   answering "if it stays under ~1s". It cannot: the indexer has no partial-update
   path, so one changed hash triggers a full rebuild (minutes, 110→425 MB). It would
   also need `IndexLock` on the query path. Gated behind `REFLEX_MCP_AUTO_INDEX=1`
   instead. **Revisit only after a genuinely incremental index path exists.**
2. **Stale always means `can_trust_results: false`.** An earlier draft scoped the flag
   to whether a changed file appeared in the result set — which returns `true` for a
   zero-result search, precisely where an agent concludes "no callers". Scope is used
   only to sharpen the warning text.
3. **Readers degrade, writers refuse.** `validate()` runs on every search and bailed on
   a schema-hash mismatch → `CacheCorrupted` → MCP force-rebuild. Several versions
   sharing one `.reflex/` therefore rebuilt it concurrently; that is the origin of
   `content.bin is too small`. The refusal is scoped narrowly to "a different released
   version owns this cache" — refusing on the schema hash alone breaks every upgrade,
   since that hash flips on any change to cache-critical sources.
4. **The symbol pass yields rather than holding the lock.** Giving it `IndexLock` would
   block every `index_project` for the whole multi-minute pass — the reported complaint,
   made mandatory. It now stops at its next batch on a cancel sentinel.
5. **Text tier on by default, with `[index] text_tier = false` to opt out**, and exempt
   from `[index] languages` — that option means "which parsers do I care about", and a
   user narrowing to Rust should not silently lose their documentation.

### Corrections to the original handoff

- The status cadence guard (`processed % 500 < batch_size`) was **already a no-op**:
  `batch_size` was itself 500. The freeze at 1000/1027 was inside the final chunk, so
  the tail needed instrumentation, not a cadence change.
- `validate()` **is** called on every search (`query/mod.rs`), which the handoff did not
  know. That is what made a version skew into a rebuild stampede.
- `list_locations` looked per-match in the code; `paths_only: true` upstream collapsed
  each file to one match. The field report was right.

### Found while fixing, not in the report

- Committing already-indexed content left the index permanently stale: the fast path
  skipped the rebuild and so never refreshed the recorded commit.
- No `busy_timeout`, `journal_mode` or `foreign_keys` pragma was set on any meta.db
  connection, so `ON DELETE CASCADE` was almost certainly never firing.
- Reflex indexed its own `.reflex/config.toml` once the text tier claimed `.toml`.
- `test_depth_limiting` was flaky ~1 run in 60 (temp dir names ending in `c`).

### Follow-up bug found, NOT fixed here

**`find_references` silently drops call sites on any line containing a URL.**
`src/line_filter.rs:77-83` does `line.find("//")` and treats anything after it as a
comment. `https://` contains `//`, so:

```rust
let url = "https://example.com/api"; call_the_target();
//                  ^ first "//" at byte 21      ^ pattern at byte 41
// 21 <= 41  =>  treated as commented  =>  dropped
```

Verified by hand. Same class as the 1.7.2 defects: a silently wrong answer. Affects
every language filter using the `//` rule, and is far worse on minified JS, where one
early URL hides every later match on that 1.4 MB line.

## ⚡ Latency: close the gap to ripgrep (2026-09-22, branch `feat/perf-enhancements`)

Plan: `~/.claude/plans/reflex-handoff-close-immutable-globe.md`. Field test (29 MB / 1875
files / 16 cores): ripgrep wins 6–10x on plain queries and 45–56x on common words.
Root causes found by code reading, **not** the handoff's guesses: `content.bin` is
uncompressed and no SQLite runs per hit. The real costs are a quadratic posting-list
intersection (`trigram.rs` `intersect_by_file_owned`: HashSet retain then a linear
`find` per candidate), `PRAGMA quick_check` + `cache.stats()` (spawns git) + two more
git spawns on every query, 3–4 re-opens of the index per query, and a per-byte
posting format carrying an unused `byte_offset`.

| WP | Scope | Status |
|---|---|---|
| WP0 | `tests/latency_budget.rs` + synthetic 30 MB corpus generator + MCP stdio round-trip; fix CI perf step (`--test` was after `--`, ran nothing) | completed (2a20431) |
| WP1 | Linear-time sorted-merge intersection + streaming posting cursor (`src/trigram.rs`) | completed (da731e0): 23–90x on Criterion intersection cases |
| WP2 | `OpenIndex` shared handle (registry keyed by cache dir, fingerprint invalidation), drop `quick_check`/`stats()`/git spawns from the query path, `--timing`; also: whole-identifier regex compiled once per query (was once per candidate line), zero-result hint counted in-search | completed (b18ae06) |
| WP3 | Early termination for list mode (`total_is_exact` / `approx_total`), chunked parallel verify honouring `[performance] parallel_threads`, line-restricted parallel regex, quantifier fix in `regex_trigrams.rs` | completed |
| WP4 | `trigrams.bin` V4: drop `byte_offset`, per-line dedup, per-file blocks, zero-copy directory, `Index/corpus ratio` line; plus `search_candidates` stop rule (skip a list > 2 B per surviving candidate) | completed |

Acceptance (same box): MCP zero-hit < 5 ms, CLI zero-hit < 15 ms, `RealmId` first
page < 15 ms, `realm --limit 1` < 20 ms, `realm --count` < 100 ms, `trigrams.bin` < 40 MB.
Counts must stay equal to ripgrep on the 1.7.2 parity set.

Result on the synthetic 30 MB corpus (medians, `.context/PERFORMANCE_RESEARCH.md`): MCP
zero-hit 0.09 ms, rare identifier 0.23 ms, common identifier first page 2.6 ms, common word
first page 3.6 ms, common word count 32 ms, regex 12 ms, `trigrams.bin` 30 MB (0.9x).
Still to confirm on the field-test repo itself (not available here): the parity set counts
and the ~10 ms `git status` share of the CLI floor. Open follow-ups: count-mode result
building (`verify`+`group` ~30 ms for 52k rows) and the `max_posting_list_entries` cap
silently dropping files past the cap.

### Still open

- **REF-219-style hybrid columnar format** — optional backlog, unchanged.
- **Incremental index path** (prerequisite for auto-refresh, decision 1 above).
- **Text tier v2**: extension-less names (`Dockerfile`, `Makefile`) were deliberately
  deferred; `should_index_lang` early-returns on a missing extension, so they need a
  second lookup path, and both plausibly want real parsers later.
- **`cleanup_stale` tail profiling**: instrumented and logged, but not yet measured on
  a large repo now that the meta.db lock contention is gone.

---

**END OF TODO.md**
