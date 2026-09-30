# Reflex

**Instant local code search — CLI, scripts, and AI agents**

Reflex is a local-first, full-text code search engine. Use it from the command line, pipe it into scripts, or connect it to AI coding assistants (Claude Code, Cursor, and any MCP-compatible tool) for instant symbol lookup, dependency analysis, and codebase exploration — fully offline, fully deterministic, no cloud required.

[![CI](https://github.com/reflex-search/reflex/actions/workflows/ci.yml/badge.svg)](https://github.com/reflex-search/reflex/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![MCP Quickstart](https://img.shields.io/badge/MCP-quickstart-blue)](docs/ai-agent-integration.md)

---

## Quick start

### 1. Install

```bash
# Via NPM
npm install -g reflex-search

# Or via Cargo
cargo install reflex-search
```

### 2. Index and search

```bash
# From your project root
rfx index

# Full-text search
rfx query "extract_symbols"

# Symbol definitions only
rfx query "CacheManager" --symbols

# JSON output for scripting
rfx query "TODO" --json --limit 20
```

### 3. (Optional) Connect to an AI agent via MCP

With Claude Code:

```bash
# every project
claude mcp add-json --scope user reflex '{"type":"stdio","command":"rfx","args":["mcp"],"alwaysLoad":true}'
# this project only (writes .mcp.json)
claude mcp add-json --scope project reflex '{"type":"stdio","command":"rfx","args":["mcp"],"alwaysLoad":true}'
```

`"alwaysLoad": true` makes Claude Code load Reflex's tool schemas at session start, so the
agent can call Reflex at once instead of first spending a turn on ToolSearch. The plain
`claude mcp add --scope user reflex -- rfx mcp` also works, without it.

For other MCP clients, register a stdio server with command `rfx` and args `["mcp"]`.

Your AI assistant can now call `search_code`, `find_references`, `get_dependencies`, and more.

> See [Claude Code + Reflex MCP Quickstart](docs/ai-agent-integration.md) for MCP setup, key tools, and troubleshooting.

---

## Why Reflex vs. built-in search tools

| Capability | grep / ripgrep | Built-in AI search | Sourcegraph | **Reflex** |
|---|---|---|---|---|
| Full-text search | ✅ | ✅ | ✅ | ✅ |
| Symbol-aware filtering | ❌ | Partial | ✅ | ✅ |
| Dependency analysis | ❌ | ❌ | Partial | ✅ |
| Deterministic results | ✅ | ❌ | ✅ | ✅ |
| Local-first / offline | ✅ | ❌ | ❌ | ✅ |
| MCP server built-in | ❌ | — | ❌ | ✅ |
| JSON output for agents | Manual | ✅ | ✅ | ✅ |

### Measured efficiency (A/B vs. built-in AI search)

The A/B harness in [`benches/efficacy/`](benches/efficacy/) runs Claude Code on the same
code-search tasks with Reflex (via MCP) and with its built-in Grep/Glob, paired per task,
on pinned checkouts of Reflex, ripgrep and tokio. Measured on Reflex 2.0.3, 2026-09-28,
8 trials per task and arm. Ratios are Reflex ÷ built-in, so **> 1.0 means Reflex costs more**.

| Tasks | Model | Tokens (95% CI) | Cost | Turns (median) | Accuracy |
|---|---|---|---|---|---|
| 9 find-all-usages | Opus 5.5 | **1.66** [1.05, 1.69] | 1.27× | 2 → 3 | equal (both near-perfect) |
| 9 find-all-usages | Sonnet 5 | **1.65** [1.06, 2.25] | 2.07× | 2 → 4 | Reflex more complete on 6 of 9 tasks |
| 13 comprehension / cross-module | Opus 5.5 | **1.26** [1.07, 1.30] | 1.10× | 6 → 6 (B more in 55 of 104 pairs, fewer in 27) | equal recall |

**What drives the cost.** It is round-trips, not payload. Claude Code defers MCP tool
schemas, so the first Reflex call costs an extra ToolSearch turn; Sonnet 5 also added
`check_index_status` calls. Trials where the agent ignored Reflex cost the same as the
control (1.04–1.07×); trials that used it cost 1.55–1.68×. Columnar results already save
about 20% of bytes per call, which the extra turns outweigh.

**Honest reading.** Reflex does not save tokens over built-in search in these agent runs.
Its value is capability: symbol-aware search, dependency analysis, `find_references` in one
call, exact counts without loading content, and more complete answers from weaker models on
large result sets. Full method, per-task tables and limits: `.context/EFFICACY-2.0.3.md`.

---

## Performance

Measured on a Kubernetes checkout (27,448 indexed files, 245 MB of text) on a 16-core machine with an NVMe disk, release build:

| | before 2.0.0 | 2.0.0 |
|---|---:|---:|
| `rfx index` from scratch | 532 s | **7.7 s** |
| Background symbol pass | 44.9 s | **3.4 s** |
| Symbol cache on disk | 256 MB | 29 MB |
| Peak memory while indexing | 1.37 GB | 1.05 GB |

The index files are byte-identical before and after, so query results and latency did not change with the indexing rewrite. On the Linux kernel, the symbol pass that used to stall part-way now completes in seconds.

Query latency on the 30 MB latency-harness corpus (2.0.0 medians, through a real `rfx mcp` round-trip): zero-hit search 0.07 ms, common-word first page 2.7 ms, regex `fn (get|set)_\w+` 10 ms, `find_references` 6.2 ms.

For a plain one-off scan of a mid-size repository, ripgrep remains a strong choice. Reflex is built for what a linear scan cannot do — symbol and dependency queries, and "every occurrence" on very large trees where scanning every file is the slow part.

---

## MCP tools

When connected via MCP, your AI assistant gets these tools:

| Tool | What it does |
|---|---|
| `search_code` | Full-text or symbol search with line numbers and context |
| `list_locations` | Fast file+line discovery (minimal tokens) |
| `count_occurrences` | Quick match statistics without full content |
| `search_regex` | Regex pattern matching across the codebase |
| `search_ast` | Structure-aware search via Tree-sitter AST queries |
| `find_references` | Symbol definition + all usage sites in a single call; the primary code-navigation tool for AI agents |
| `index_project` | Force an index run (rarely needed: every tool updates the index first) |
| `check_index_status` | Report whether the index matches the files on disk, without updating it |
| `get_dependencies` | All imports for a specific file |
| `get_dependents` | All files that import a given file (reverse lookup) |
| `get_transitive_deps` | Transitive dependency graph up to a configurable depth |
| `find_hotspots` | Most-imported files (dependency hotspots) |
| `find_circular` | Detect circular dependency chains |
| `find_unused` | Files with no incoming dependencies |
| `find_islands` | Disconnected components in the dependency graph |
| `analyze_summary` | High-level dependency counts and metrics |
| `gather_context` | Codebase structure and project-type summary |

**No `rfx index` needed after edits.** Every command and MCP tool updates a stale index before it answers (only the changed files) and builds a missing one; pass `--no-update` (`rfx mcp --no-update`) to answer from the index as it is.

Three behaviours agents rely on:

- **Whole identifiers by default.** `verify_csrf` does not match `verify_csrf_form_field`; pass `contains: true` for substring matching (`grep -F`) or `ignore_case: true` for `rg -i`. A zero result names the substring count in a `hint`.
- **Freshness on search responses.** `status` and `can_trust_results` compare the working tree (size, mtime, content hash) with what the index holds; the index is updated before each call, so `can_trust_results: false` appears only when that update could not run (`warnings` says why).
- **Lock and generated files stay out of the way.** They are indexed but excluded from results unless you pass `include_locks` / `include_generated`; a zero result caused only by them says so.

See [`docs/mcp-tool-cheatsheet.md`](docs/mcp-tool-cheatsheet.md) for a decision tree by agent intent.

---

## CLI usage

Reflex also works as a standalone CLI for humans and shell scripts.

```bash
# Full-text search (finds every occurrence)
rfx query "extract_symbols"

# Symbol definitions only (faster, uses tree-sitter)
rfx query "extract_symbols" --symbols

# Filter by language and symbol kind
rfx query "parse" --lang rust --kind function --symbols

# Regex search
rfx query "fn.*test" --regex

# Case-insensitive (like rg -i); still uses the trigram index
rfx query "realmid" -i
rfx query "(?i)realm_?id" --regex

# Patterns that start with `-`
rfx query --pattern '-> Result<'

# Count only, with per-phase timings on stderr
rfx query "unwrap" --count --timing

# Include lock files / generated files, or search the plain-text tier only
rfx query "1.0.190" --include-locks
rfx query "timeout" --lang text

# JSON output for programmatic use
rfx query "unwrap" --json --limit 10

# Pipe file paths to other tools
vim $(rfx query "TODO" --paths)
```

**Interactive TUI mode** — run `rfx query` with no pattern to launch live search with keyboard navigation.

### Dependency analysis

```bash
rfx deps src/main.rs              # Show direct imports
rfx deps src/config.rs --reverse  # What imports this file
rfx deps src/api.rs --depth 3     # Transitive dependencies
rfx analyze --circular            # Find circular dependency chains
rfx analyze --hotspots            # Most-imported files
rfx analyze --unused              # Files with no incoming dependencies
```

### Natural language search

```bash
rfx ask "Find all TODOs in Rust files"         # Translate to rfx query and run
rfx ask "How does authentication work?" --agentic  # Multi-step codebase reasoning
rfx ask                                        # Interactive chat mode
```

Requires an AI provider configured via `rfx llm config` (OpenAI, Anthropic, OpenRouter, or any OpenAI-compatible endpoint).

### Other commands

```bash
rfx index                 # Build / update the search index (then spawns the symbol pass)
rfx index --force         # Full rebuild from scratch
rfx index status          # Background symbol indexing status
RUST_LOG=info rfx index   # Per-phase timings (read/extract, database, trigram write)
rfx watch                 # Auto-reindex on file changes
rfx stats                 # Index statistics
rfx list-files            # Every indexed file
rfx clear                 # Delete the local cache
rfx context               # Codebase context for AI prompts
rfx snapshot              # Structural snapshots for change tracking
rfx pulse generate        # Documentation site (static HTML) from the index
rfx pulse serve           # Preview it locally
rfx pulse map             # Architecture diagram (Mermaid / D2)
rfx serve --port 7878     # Local HTTP API server
```

Run `rfx <command> --help` for full options.

### Documentation sites (Pulse)

`rfx pulse generate` builds a two-tab docs site from the index: **Docs** (overview,
guides, CLI and API reference for Rust, Python and Go, a changelog page per release) and
**Internals** (architecture, dependency map, module pages). The output is plain HTML for
any static host. It needs Node 22.12+ to build; the site runtime downloads once. An
optional LLM adds prose, and a deterministic gate drops any sentence the index does not
support. A GitHub Action is included. See [docs/features/PULSE.md](docs/features/PULSE.md).

---

## Installation

### NPM (recommended)

```bash
npm install -g reflex-search
```

### Cargo

```bash
cargo install reflex-search
```

**Setup note:** run `rfx` commands from your project root directory. Add `.reflex/` to your `.gitignore` to exclude the search index from version control.

---

## Supported languages

Full symbol extraction (functions, classes, methods, types, etc.) for 15 languages:

**Systems:** Rust, C, C++, Zig  
**Backend:** Python, Go, Java, C#, PHP, Ruby, Kotlin  
**Frontend:** TypeScript, JavaScript, Vue, Svelte

> **Swift** is temporarily disabled (tree-sitter-swift 0.7.x grammar incompatibility). `rfx query --lang swift` emits a warning; full-text search still works.

### Coverage

Coverage matches ripgrep's defaults: every non-binary file that is not gitignored and not under a dot-directory (.github/, .githooks/, .cargo/ …). Hidden paths are not indexed unless `[index] hidden = true`. Lock and generated files are indexed but left out of results unless you pass include_locks / include_generated. Select the non-code tier with `--lang text`; select lock or generated files alone with `--lang lock` / `--lang generated`. `[index] mode = "allowlist"` indexes only code plus a fixed docs/config extension list; `[index] hidden = true` indexes dot-directories (never `.git/` or `.reflex/`). A zero result names its cause in `excluded_reason` (`hidden`, `not_indexed`, `lock_or_generated`, `whole_identifier`) and a `hint`. Files without a symbol parser (the text tiers, Swift) are fully text-searchable but yield no `--symbols` results.

---

## Configuration

```toml
# .reflex/config.toml (project-level)
[index]
languages = []          # Empty = all supported languages
mode = "tracked"        # ripgrep's defaults; "allowlist" = code + a fixed docs/config extension list
hidden = false          # true walks dot-directories (never .git/ or .reflex/)
text_tier = true        # index docs, config and data files as `text`
max_file_size = 10485760  # 10 MB
# Optional, gitignore rules (a `/` anchors at the root; bare names match anywhere):
# include.patterns = ["src/**/*.rs"]
# exclude.patterns = ["vendor/**"]

[performance]
parallel_threads = 0    # indexing and query pools; 0 = auto (80% of cores, max 32)
symbol_threads = 0      # background symbol pass; 0 = auto (50% of cores, max 32)
```

Environment variables, mostly for benchmarking and CI: `REFLEX_INDEX_BATCH_FILES` / `REFLEX_INDEX_BATCH_BYTES` (batch bounds while indexing, default 5000 files / 48 MiB), `REFLEX_SYMBOL_THREADS`, `REFLEX_FRESHNESS_TTL_MS` (how long `rfx mcp` memoises the freshness verdict), `REFLEX_MCP_TIMING=1` (a `timings` object on search responses), `REFLEX_SQLITE_JOURNAL=delete` (for network filesystems that cannot do WAL).

For AI provider configuration (`rfx ask`, `rfx pulse`), run `rfx llm config`.

---

## Architecture

Reflex uses a **trigram-based inverted index** with a **background symbol cache**:

- **Indexing**: a thread pool reads and hashes every file, extracts imports with tree-sitter, and extracts trigram postings. Each batch is built per trigram shard in parallel and partial batches are merged by byte copy, so the output is identical whatever the batch boundaries. `trigrams.bin` and `content.bin` are written to a temp file, synced and renamed, never left short.
- **Symbols**: `rfx index` spawns a detached pass that parses every file once with one combined tree-sitter query per language and stores compressed symbol lists in `meta.db`. Queries parse cache misses on demand, so `--symbols` works before the pass has finished.
- **Full-text queries**: intersect trigram posting lists → verify candidate lines in parallel. Freshness is judged by file content (size, mtime, hash), not by commit, so a commit of already-indexed files is not "stale".
- **Symbol queries**: trigrams narrow the candidates → their symbols are read from the cache; only cache misses are parsed.

```
.reflex/
  meta.db          # SQLite: file metadata, symbol cache, dependency graph, stats
  trigrams.bin     # Inverted index (memory-mapped)
  content.bin      # Full file contents (memory-mapped)
  config.toml      # Index settings
```

---

## Security

`rfx serve` binds to `127.0.0.1:7878` by default — loopback only, no authentication. Do not expose it to the network. See [CLAUDE.md](CLAUDE.md#security--threat-model) for the full threat model.

---

## Contributing

```bash
cargo build --release        # Build
cargo test                   # Test
cargo clippy --all-targets   # Lint
REFLEX_LATENCY_BUDGET=1 cargo test --release --test latency_budget -- --ignored --test-threads=1   # Query latency budgets
rfx index                    # Refresh index after code changes
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

**Fast code search for developers — works standalone, in scripts, and with AI coding agents**
