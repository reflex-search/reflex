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
| **Runtime Symbol Parser** | Uses Tree-sitter to parse candidate files at query time (only files matching trigrams) |
| **Background Symbol Indexer** | Daemonized process that pre-caches symbols for faster queries on large codebases |
| **Symbol Cache** | Persistent storage of parsed symbols (803-line caching system for instant symbol lookups) |
| **CLI / API Layer** | Single binary for human and programmatic use (CLI and optional HTTP/MCP) |
| **Watcher (optional)** | Incrementally updates index on file changes |

### Index Cache Structure (`.reflex/`)
    .reflex/
      meta.db          # SQLite: file metadata, stats, config
      trigrams.bin     # Inverted index: trigram → [file_id, line_no] posting lists
      content.bin      # Memory-mapped full file contents for context extraction
      config.toml      # Project settings (index, search, performance)

### User Configuration (`~/.reflex/`)
    ~/.reflex/
      config.toml      # User settings (semantic query provider, API keys, model preferences)

---

## CLI Usage

**Indexing:**
```bash
rfx index                        # Build/update cache
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
(never `max_results`); the path filter is `file` (substring) or `glob` (array), never `path`. Since 1.7.0 the
server accepts those wrong names as aliases and returns a `warnings` field; unknown keys are rejected with a
did-you-mean error, and numeric strings like `"40"` are coerced.

### Matching semantics (1.7.2)

Literal search matches **whole identifiers** by default. `verify_csrf` does **not**
match `verify_csrf_form_field`. Three modes, each with a case-insensitive variant:

| Mode | How | Behaves like |
| --- | --- | --- |
| whole identifier | default | `grep -w` |
| substring | `contains: true` | `grep -F` |
| regular expression | `search_regex` | `grep -E` |
| any of the above, case-insensitive | `ignore_case: true` (`-i` / `--ignore-case` on the CLI) | `rg -i` (+ `contains` = `rg -i -F`) |

- `contains` is available on `search_code`, `count_occurrences`, `list_locations` and
  `find_references` (not `search_regex`, which is already substring-based).
- `ignore_case` is available on those four **and** `search_regex` (where it prepends
  `(?i)`). Since 1.8.0 a `(?i)` literal is looked up in the trigram index under every
  case variant, so it costs about what the case-sensitive query costs; before, any `i`
  flag forced a scan of every line. A whole-identifier `ignore_case` search keeps
  whole-identifier semantics (`realmid` finds `RealmId`, not `realm_id`), reports
  `kind: text_match`, and produces no zero-result substring `hint`. Counts match
  ripgrep `-i` exactly, including the Unicode folds of `k` (KELVIN SIGN) and `s`
  (LONG S); a non-ASCII literal under `(?i)` still scans, and says so in `warnings`.
- A pattern containing brackets (`()`, `[]`, `<>`) is regex-escaped and run through the
  regex path automatically, with the rewrite reported in `warnings`. Whole-identifier
  matching wraps the pattern as `\b…\b`, which a pattern ending in `)` or `>` can
  never satisfy — `unwrap()` used to return a silent `0`. Since 1.8.0 the rewrite lives
  in the engine, so `rfx query`, `rfx serve` and MCP all apply it: the CLI prints
  `Warning:` on stderr and carries `warnings[]` / `hint` in `--json` output.
- A zero result carries a `hint` naming the substring count:
  `"0 whole-identifier matches; 89 substring matches — pass contains:true"`.

### Freshness contract (1.7.2)

Every response carries `status` and `can_trust_results`. Staleness now includes
**uncommitted working-tree changes**, not just commit moves:

```json
{ "status": "stale", "can_trust_results": false,
  "reason": "Working tree has uncommitted changes since indexing (1 modified, 1 added)",
  "action_required": "index_project",
  "files_modified": ["src/storage/mod.rs"],
  "files_added": ["src/storage/zz_probe.rs"],
  "files_deleted": [],
  "changed_count": 2 }
```

- **A stale index always yields `can_trust_results: false`.** No exception — including
  a zero-result search, which is exactly where an agent concludes "no callers".
- `files_modified` was a `u32` count before 1.7.2 and is now a path list (**breaking**).
  Lists cap at 100 per category; `truncated` says when.
- `action_required` names the MCP tool (`index_project`), never the CLI.
- Checked via `git status --porcelain`, memoised for 1s per workspace
  (`REFLEX_FRESHNESS_TTL_MS`; `0` disables). `check_index_status` always bypasses it.
- **Limitation**: outside a git repository, working-tree changes are not detected.

**Core search:**
| Tool | Purpose |
|------|---------|
| `check_index_status` | Check if index is fresh before searching |
| `search_code` | Full-text search with previews (default limit: 200) |
| `search_regex` | Regex pattern search (use for `->`, `::`, `()`, etc.) |
| `list_locations` | Path+line only — cheapest, no content loaded |
| `count_occurrences` | Count matches without loading content |
| `find_references` | Definition + all usages in one atomic call (default limit: 200) |
| `gather_context` | Project structure, frameworks, entry points |
| `search_ast` | Tree-sitter AST pattern matching (⚠️ slow — requires `glob`) |

**Index management:**
| Tool | Purpose |
|------|---------|
| `index_project` | Build or update the search index |

**Dependency analysis:**
| Tool | Purpose |
|------|---------|
| `get_dependencies` | What a file imports |
| `get_dependents` | What imports a file (reverse lookup) |
| `find_hotspots` | Most-imported files by dependent count |

**Structural analysis** (on by default; hide with `[mcp] enable_structural_tools = false` in `~/.reflex/config.toml`):
`find_circular` · `find_islands` · `find_unused` · `analyze_summary` · `get_transitive_deps`

See [`docs/mcp-tool-cheatsheet.md`](./docs/mcp-tool-cheatsheet.md) for a decision tree by agent intent.

**Columnar result format (`search_code` / `search_regex`, list mode):** To cut token
cost from repeated JSON keys, these two tools return matches in a columnar shape
instead of an array of per-file objects. Measured payload savings: **16–24%** on typical
query results (the `~41%` estimate assumed file-grouped output; flat columnar still
repeats `path`/`language` per row):

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

### Early termination and totals (1.8.0)

A list-mode `search_code` / `search_regex` call verifies candidates in path order
and **stops once the page is full** (`offset + limit` results). The page is identical
to the same slice of a full run, but the total is not always exact:

| field | meaning |
| --- | --- |
| `total_is_exact: true` | `total_count` / `pagination.total` counts every match (count mode, no `limit`, symbol/AST searches, `find_references`) |
| `total_is_exact: false` | verification stopped early: `total_count` / `pagination.total` are **`null`** (never the verified-so-far number), `approx_total` is a **sampled estimate** (32 files spread over the remaining candidates, ≤16 lines each; typically within ±30%, omitted for a regex with no literal), and `has_more` is `true` |

When 32 files or 128 candidate lines or fewer remain after the page fills, the search
finishes instead and the total is exact. `mode: "count"`, `count_occurrences`,
`list_locations` and `find_references` always verify everything. The CLI prints an
inexact total as `Found 10 results (~1234 total, estimated)` and points at `--count`.
Library readers: `PaginationInfo::exact_total()` (the total or `None`) and
`best_total()` (exact, else estimate, else page end; for thresholds only).

### Glob rules (1.8.0)

`glob` / `exclude` (every MCP search tool), `rfx query --glob` / `--exclude` and
`[index] include.patterns` / `exclude.patterns` follow **gitignore / ripgrep rules**:

| pattern | matches |
| --- | --- |
| `src/**/*.rs` | `.rs` files under `src/` **at the index root only** (a `/` anchors) |
| `**/src/**/*.rs` | `.rs` files under any `src/` directory |
| `*.rs`, `Makefile` | at any depth (no `/` in the pattern) |
| `target/` | any `target/` directory and everything under it |
| `src/*.rs` | directly in `src/`; `*` never crosses `/` |

Before 1.8.0 every relative pattern got a `**/` prefix, so `src/**/*.rs` also matched
`vendor/src/`. `./` and a leading `/` are dropped.

### Latency diagnostics

- `rfx query <pattern> --timing` prints per-phase timings (open, candidates, verify,
  status, group) to stderr; with `--json` they appear as a `timings` object.
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
- `tests/latency_budget.rs` (`cargo test --release --test latency_budget -- --ignored
  --nocapture --test-threads=1`) measures the field-test query shapes in-process and
  through a real `rfx mcp` stdio round-trip; CI asserts budgets with
  `REFLEX_LATENCY_BUDGET=1`.

**MCP efficiency — measured A/B results:**
- **Columnar format saves 16–24% per-call bytes** on `search_code`/`search_regex` payloads.
- **Reflex vs built-in grep/glob: at parity on total tokens; real wins are capability and cost.**
  Powered A/B rerun (REF-222: n=9 tasks × 8 trials × claude-sonnet-4-6): r=1.044, 95% CI
  [1.014, 1.262] — within the ±10% parity band, 100% task success on both arms, arm B showing
  higher recall on large result sets (graded accuracy). The REF-176 → REF-217 (n=3, r=1.047,
  wide CI) → REF-222 (n=9, r=1.044, 4× tighter CI) arc confirms parity was never lost — the
  CI shrank, the point estimate held. The real wins over grep/glob are **capability** (atomic
  `find_references`, symbol filtering, dependency analysis — unavailable in built-in tools) and
  **~31% lower cost** (REF-192). *Method note:* at equal turn counts Reflex overhead is only
  ~1–2%; turn-count variance drives the CI spread (corr ≈ 0.99, REF-204).
- **structuredContent: evaluated and rejected.** MCP `outputSchema`/`structuredContent` was built,
  A/B tested (ratio 0.998 — no measurable savings because Claude Code transmits *both*
  `content[text]` and `structuredContent`), and removed. Do not re-attempt unless using a client
  that honors `outputSchema` and drops the text block. See [`docs/mcp-tool-cheatsheet.md`](./docs/mcp-tool-cheatsheet.md)
  for the full efficiency notes.

---

## AST Pattern Matching

⚠️ **PERFORMANCE WARNING**: AST queries are **SLOW** (500ms-10s+) and scan the **ENTIRE codebase**. **Use `--symbols` instead in 95% of cases** (10-100x faster).

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

**Performance**: 2-5ms (full-text) vs 3-10ms (--symbols) vs 500ms-10s+ (--ast). **ALWAYS use `--glob` with AST queries.**

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

**Coverage**: 90%+ of all codebases across web, mobile, systems, enterprise, and AI/ML development.

### Plain-Text Tier (every non-binary file)

Reflex indexes non-code files, because **agents do not partition searches by file
type**. A config key lives in the YAML, the Rust struct *and* the spec paragraph;
returning only the struct and a confident `0` for the rest is a wrong answer.

**Coverage rule (1.8.0, `[index] mode = "tracked"`, the default)**: ripgrep's defaults —
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
| `search_code`, `search_regex`, `count_occurrences`, `list_locations` | text **included by default**; lock and generated on request |
| `--symbols`, `--kind`, `--ast`, `search_ast` | excluded (there is no grammar) |
| `find_references`, `get_dependents`, structural tools | excluded (a mention in a changelog is not a call site) |

- **Select the text tier**: `--lang text` (aliases `txt`, `plaintext`, `plain`).
- **Exclude it**: `exclude_text: true` on the four full-text MCP tools.
- **Turn it off**: `[index] text_tier = false` in `.reflex/config.toml`.
- **Old rule**: `[index] mode = "allowlist"` restores the pre-1.8.0 behaviour — code by
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
  - **Parsing**: `tree-sitter` + language grammars (runtime symbol parsing at query time)
  - **Storage**: `rusqlite` (metadata), custom binary format (trigrams + content)
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

## Runtime Symbol Detection Architecture

Reflex uses a unique **runtime symbol detection** approach that combines the speed of trigram indexing with the precision of tree-sitter parsing:

### How It Works

1. **Indexing Phase** (no tree-sitter parsing):
   - Extract trigrams from all files → build inverted index
   - Store full file contents in memory-mapped content.bin
   - No symbol extraction or tree-sitter parsing during indexing

2. **Query Phase** (lazy parsing only when needed):
   - **Full-text queries**: Use trigrams only (instant results)
   - **Symbol queries** (`--symbols` or `--kind function`):
     1. Trigram search narrows 62K files → ~10-100 candidates
     2. Parse only candidate files with tree-sitter (2-224ms overhead)
     3. Filter to symbol definitions and return results

### Performance Benefits

| Approach | Indexing Time | Query Time | Memory Usage |
|----------|---------------|------------|--------------|
| **Old (indexed symbols)** | Slow (parse all files) | 4125ms (load 3.3M symbols) | High (symbols.bin) |
| **New (runtime parsing)** | Fast (trigrams only) | 2-224ms (parse 10 files) | Low (no symbols.bin) |

**Improvement**: 2000x faster on small codebases (4125ms → 2ms), 18x faster on Linux kernel (4125ms → 224ms)

### Why This Works

- **Trigrams are excellent filters**: Reduce search space by 100-1000x
- **Most queries are full-text**: Symbol filtering is the minority case
- **Parsing is fast**: Tree-sitter parses 10 files in ~2ms
- **Lazy evaluation wins**: Parse only what's needed, when it's needed

### Architecture Simplification

Removed components:
- `symbols.bin` (entire symbol storage file)
- `SymbolWriter` (~250 lines of serialization code)
- `SymbolReader` (~250 lines of deserialization code)

Result: **Simpler, faster, smaller cache, more flexible symbol filtering**

---

## Design Notes
- **Trigram Algorithm**: Extracts 3-character substrings; builds inverted index for O(1) lookups
- **Runtime Symbol Detection**: Parse only candidate files at query time (10-100 files vs 62K+ files at index time)
- **Incremental by content**: Files reindexed only if `blake3` hash changes
- **Memory-mapped I/O**: Zero-copy access to trigrams.bin and content.bin
- **Regex support**: Extracts guaranteed trigrams from patterns; falls back to full scan if needed
- **Deterministic**: Same query always returns same results (sorted by file:line)
- **Respects .gitignore**: Uses `ignore` crate to skip untracked files
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
- `[index]`: Languages, the plain-text tier, file size limits, symlink handling
- `[search]`: Default result limits, fuzzy matching thresholds
- `[performance]`: Thread count, compression levels

**Example**:
```toml
[index]
languages = []  # Empty = all supported languages
text_tier = true  # Also index docs and config (md, yaml, toml, json, proto, html, sh, sql)
max_file_size = 10485760  # 10 MB
# gitignore rules: a pattern with `/` is anchored at the root, a bare name matches anywhere.
# include.patterns = ["src/**/*.rs", "docs/**"]   # whitelist (directories are still walked)
# exclude.patterns = ["vendor/**", "*.generated.rs"]

[search]
default_limit = 100

[performance]
parallel_threads = 0  # 0 = auto (80% of cores)
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
- Contains:
  - MVP goals and success criteria
  - Task breakdown by module with priority levels (P0/P1/P2/P3)
  - Implementation phases and timeline
  - Open questions and design decisions
  - Performance targets and benchmarks
  - Maintenance strategy and update policy

#### Optional Research Files

Create RESEARCH.md files as needed to cache important findings:

**`.context/TREE_SITTER_RESEARCH.md`** - Tree-sitter grammar investigation
- Document findings about each language grammar
- Node types and AST structure for symbol extraction
- Query patterns and examples
- Quirks, gotchas, and edge cases
- Version compatibility notes

**`.context/PERFORMANCE_RESEARCH.md`** - Optimization findings
- Benchmarking results and bottleneck analysis
- Memory-mapping techniques and best practices
- Indexing speed optimizations
- Query latency improvements
- Cache format trade-offs

**`.context/BINARY_FORMAT_RESEARCH.md`** - Data serialization decisions
- Binary format design rationale
- Alternatives considered and rejected
- Serialization library comparisons (bincode, rkyv, custom)
- Versioning and migration strategies

**`.context/LANGUAGE_SPECIFIC_NOTES.md`** - Per-language implementation details
- Language-specific symbol extraction challenges
- Parser implementation patterns
- Testing strategies for each language
- Real-world codebase findings

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
   - Ensure all task statuses are accurate
   - Document any blocking issues or open questions
   - Update implementation notes if approach changed
   - Commit research findings to appropriate RESEARCH.md files

4. **When Conducting Research:**
   - Create focused RESEARCH.md files rather than losing findings
   - Include code examples, links, and specific version numbers
   - Note what was tried and why it didn't work (avoid repeated dead ends)
   - Cross-reference related TODO.md tasks

5. **Decision Documentation:**
   - Major decisions go in `.context/TODO.md` under "Notes & Design Decisions"
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
# 6. Update TODO.md to completed
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
