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

Add this to your Claude Code MCP configuration — `~/.claude/claude_code_config.json` for every project, or `.claude/claude_code_config.json` for one project:

```json
{
  "mcpServers": {
    "reflex": {
      "command": "rfx",
      "args": ["mcp"]
    }
  }
}
```

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

We A/B-tested an AI coding agent on real code-search tasks **using Reflex (via MCP)** against the **same agent using its built-in search** (ripgrep-backed Grep/Glob) — identical tasks, model and repository, paired per task. The harness lives in [`benches/efficacy/`](benches/efficacy/) and is fully reproducible.

**Setup (powered rerun):** model `claude-sonnet-4-6`; 9 code-search tasks (find-all-usages, symbol locate, dependency and reverse-dependency, hotspot, negative controls); 8 trials per arm (72 observations per arm); run against the Reflex repository.

**Results** — Reflex ÷ built-in, so **< 1.0 means Reflex uses less**:

| Metric | Reflex ÷ built-in | Reading |
|---|---|---|
| Task success rate | **1.00** (100% vs 100%) | Equal correctness — no regression |
| Total tokens (median over tasks) | **1.044**, 95% CI [1.014, 1.262] | Parity: inside the pre-registered ±10% band |
| Precision of returned locations | ≈ 1.00 both arms | No hallucinated hits either way |
| Recall on large result sets | Reflex higher (e.g. 0.28 vs 0.008, 0.48 vs 0.27) | More exhaustive answers when there are hundreds of hits |
| Cost per task (median) | **0.69** | **~31% cheaper** |

**Implications**

- **No-regret replacement for built-in search.** Same correctness, token parity, lower dollar cost. At equal turn counts the per-call overhead of Reflex's richer responses is 1–2%; the spread in the CI comes from turn-count variance, not payload size.
- **The wins are capability, not token savings.** `find_references` returns a symbol's definition *and* every call site in one call; symbol-kind filtering and the dependency tools have no grep/glob equivalent.
- **Honest caveats.** One model, one repository. On comprehension-style tasks that span modules, the agent took more turns with Reflex than with built-in search; Reflex's advantage is single-shot reference finding and exhaustive results on large trees.

Reproduce it yourself:

```bash
benches/efficacy/run-ref222.sh          # the powered run (9 tasks × 8 trials)
python3 benches/efficacy/runner.py --arms A B --repos reflex --n 3   # a quick thin slice
python3 benches/efficacy/extract_metrics.py && python3 benches/efficacy/analyze.py
```

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

Query latency on the 30 MB latency-harness corpus (medians, through a real `rfx mcp` round-trip): zero-hit search 0.09 ms, common-word first page ~3 ms, regex `fn (get|set)_\w+` ~12 ms, `find_references` ~6 ms.

One honest comparison: on a warm mid-size repository, ripgrep still wins plain one-off scans by 6–10x. Reflex is built for what a linear scan cannot do — symbol and dependency queries, and "every occurrence" on very large trees where scanning every file is the slow part.

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
| `index_project` | Trigger or refresh the search index |
| `check_index_status` | Check whether the index is fresh, stale, or missing; call before any search session or after git operations |
| `get_dependencies` | All imports for a specific file |
| `get_dependents` | All files that import a given file (reverse lookup) |
| `get_transitive_deps` | Transitive dependency graph up to a configurable depth |
| `find_hotspots` | Most-imported files (dependency hotspots) |
| `find_circular` | Detect circular dependency chains |
| `find_unused` | Files with no incoming dependencies |
| `find_islands` | Disconnected components in the dependency graph |
| `analyze_summary` | High-level dependency counts and metrics |
| `gather_context` | Codebase structure and project-type summary |

**Index not found error?** If an MCP tool returns `"Index not found. Run 'rfx index' to build the cache first"`, call `index_project` first, then retry the failed tool.

Three behaviours agents rely on:

- **Whole identifiers by default.** `verify_csrf` does not match `verify_csrf_form_field`; pass `contains: true` for substring matching (`grep -F`) or `ignore_case: true` for `rg -i`. A zero result names the substring count in a `hint`.
- **Freshness on every response.** `status` and `can_trust_results` compare the working tree (size, mtime, content hash) with what the index holds; a stale index always yields `can_trust_results: false`, and `action_required` names `index_project`.
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
rfx pulse changelog       # Codebase change digest
rfx pulse wiki            # Per-module documentation
rfx pulse map             # Architecture diagram (Mermaid / D2)
rfx serve --port 7878     # Local HTTP API server
```

Run `rfx <command> --help` for full options.

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

Coverage matches ripgrep's defaults: every non-binary file that is not gitignored and not under a dot-directory (.github/, .githooks/, .cargo/ …). Hidden paths are not indexed — use grep for those. Lock and generated files are indexed but left out of results unless you pass include_locks / include_generated. Select the non-code tier with `--lang text`; select lock or generated files alone with `--lang lock` / `--lang generated`. `[index] mode = "allowlist"` restores the pre-2.0.0 fixed extension list; `[index] hidden = true` indexes dot-directories (never `.git/` or `.reflex/`). A zero result names its cause in `excluded_reason` (`hidden`, `not_indexed`, `lock_or_generated`, `whole_identifier`) and a `hint`. Files without a symbol parser (the text tiers, Swift) are fully text-searchable but yield no `--symbols` results.

---

## Configuration

```toml
# .reflex/config.toml (project-level)
[index]
languages = []          # Empty = all supported languages
mode = "tracked"        # ripgrep's defaults; "allowlist" = the pre-2.0.0 extension list
hidden = false          # true walks dot-directories (never .git/ or .reflex/)
text_tier = true        # index docs, config and data files as `text`
max_file_size = 10485760  # 10 MB
# Optional, gitignore rules (a `/` anchors at the root; bare names match anywhere):
# include.patterns = ["src/**/*.rs"]
# exclude.patterns = ["vendor/**"]

[search]
default_limit = 100

[performance]
parallel_threads = 0    # indexing and query pools; 0 = auto (80% of cores, max 32)
symbol_threads = 0      # background symbol pass; 0 = auto (50% of cores, max 32)
```

Environment variables, mostly for benchmarking and CI: `REFLEX_INDEX_BATCH_FILES` / `REFLEX_INDEX_BATCH_BYTES` (batch bounds while indexing, default 5000 files / 48 MiB), `REFLEX_SYMBOL_THREADS`, `REFLEX_FRESHNESS_TTL_MS` (how long `rfx mcp` memoises the freshness verdict), `REFLEX_MCP_TIMING=1` (a `timings` object on search responses), `REFLEX_SQLITE_JOURNAL=delete` (for network filesystems that cannot do WAL).

For AI provider configuration (`rfx ask`, `rfx pulse`), run `rfx llm config`.

---

## Architecture

Reflex uses a **trigram-based inverted index** with **runtime symbol detection**:

- **Indexing**: a thread pool reads and hashes every file, extracts imports with tree-sitter, and extracts trigram postings. Each batch is built per trigram shard in parallel and partial batches are merged by byte copy, so the output is identical whatever the batch boundaries. `trigrams.bin` and `content.bin` are written to a temp file, synced and renamed, never left short.
- **Symbols**: `rfx index` spawns a detached pass that parses every file once with one combined tree-sitter query per language and stores compressed symbol lists in `meta.db`. Queries parse cache misses on demand, so `--symbols` works before the pass has finished.
- **Full-text queries**: intersect trigram posting lists → verify candidate lines in parallel. Freshness is judged by file content (size, mtime, hash), not by commit, so a commit of already-indexed files is not "stale".
- **Symbol queries**: trigrams narrow the candidates → only those files are parsed (or read from the symbol cache).

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
