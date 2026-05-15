# Reflex

**Sub-100ms local code search — CLI, scripts, and AI agents**

Reflex is a local-first, full-text code search engine. Use it from the command line, pipe it into scripts, or connect it to AI coding assistants (Claude Code, Cursor, and any MCP-compatible tool) for instant symbol lookup, dependency analysis, and codebase exploration — fully offline, fully deterministic, no cloud required.

[![CI](https://github.com/reflex-search/reflex/actions/workflows/ci.yml/badge.svg)](https://github.com/reflex-search/reflex/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-347%20passing-brightgreen)]()
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

Add this to your Claude Code MCP configuration (`~/.claude/claude_desktop_config.json`):

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
| `index_project` | Trigger or refresh the search index |
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

---

## AI coding workflow

Reflex tools form a cost hierarchy. Start at the cheapest tier that answers your question — only escalate when you need more.

```
list_locations  →  search_code  →  search_code (expand: true)
     ⚡                🟡                    🟡
  file + line     preview + context       full body
  (no content)                          (most tokens)
```

**Rule:** use `list_locations` to orient, `search_code` to inspect, `search_code` + `expand` only for the specific match you intend to read or edit.

### Worked example — finding and understanding a function

**Task:** "How does `process_payment` work? I need to add retry logic."

**Without Reflex:** `grep -r "process_payment" src/` returns 40 lines of interleaved match output from 8 files. Then `cat src/payments/processor.rs` returns the full 420-line file. You read several hundred lines to find the 30 lines you actually need.

**With Reflex — three targeted steps:**

**Step 1 — Orient** (`list_locations`, cheapest)

```json
{
  "tool": "list_locations",
  "arguments": { "pattern": "process_payment" }
}
```

Response — file and line only, no content loaded:

```json
[
  { "path": "src/payments/processor.rs", "line": 47 },
  { "path": "src/payments/processor.rs", "line": 112 },
  { "path": "src/api/checkout.rs", "line": 88 },
  { "path": "tests/integration/payment_test.rs", "line": 23 }
]
```

4 locations across 3 files. The definition is almost certainly in `processor.rs:47`. Move to Step 2.

**Step 2 — Confirm the definition** (`search_code` with `symbols: true`)

```json
{
  "tool": "search_code",
  "arguments": {
    "pattern": "process_payment",
    "symbols": true
  }
}
```

Response — definition only, with signature preview:

```json
{
  "results": [
    {
      "path": "src/payments/processor.rs",
      "kind": "Function",
      "symbol": "process_payment",
      "span": { "start_line": 47, "end_line": 78 },
      "preview": "pub async fn process_payment(order: &Order, method: PaymentMethod) -> Result<Receipt, PaymentError> {"
    }
  ]
}
```

It's a 31-line async function. Step 3 gets the body.

**Step 3 — Read the body** (`search_code` with `expand: true`)

```json
{
  "tool": "search_code",
  "arguments": {
    "pattern": "process_payment",
    "symbols": true,
    "expand": true
  }
}
```

Response — full implementation, lines 47–78 only. No surrounding file noise.

**Result:** 3 targeted tool calls instead of reading 420 lines of file content. The agent now knows exactly where to insert retry logic without ever loading unrelated code.

---

### Dependency analysis — architecture context in two calls

Built-in tools (`grep`, `read`) cannot tell you what depends on a file. Reflex can.

**"Is `processor.rs` a critical file? What breaks if I change it?"**

```json
{
  "tool": "get_dependents",
  "arguments": { "path": "src/payments/processor.rs" }
}
```

Response:

```json
{
  "dependents": [
    { "path": "src/api/checkout.rs", "import_count": 3 },
    { "path": "src/api/subscriptions.rs", "import_count": 1 },
    { "path": "src/workers/retry_queue.rs", "import_count": 2 },
    { "path": "tests/integration/payment_test.rs", "import_count": 1 }
  ]
}
```

Three production callers. Any change to `processor.rs` signatures affects all three. Now check the broader architecture:

```json
{
  "tool": "find_hotspots",
  "arguments": { "min_dependents": 3 }
}
```

Response — the most-imported files in the codebase, ranked by dependent count. `processor.rs` appearing here signals it's load-bearing infrastructure, not a leaf module — refactor with care.

**This analysis is impossible with grep or read alone.** Dependency graph traversal requires the indexed import graph that Reflex builds at index time.

---

> For the full tool decision tree and filter reference, see [MCP Tool Selection Cheatsheet](docs/mcp-tool-cheatsheet.md).

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
rfx index                 # Build / update the search index
rfx index status          # Background indexing status
rfx watch                 # Auto-reindex on file changes
rfx stats                 # Index statistics
rfx pulse digest          # Codebase change digest
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

Full-text search works on **all file types** regardless of parser support.

---

## Configuration

```toml
# .reflex/config.toml (project-level)
[index]
languages = []          # Empty = all supported languages
max_file_size = 10485760  # 10 MB

[search]
default_limit = 100

[performance]
parallel_threads = 0    # 0 = auto (80% of available cores)
```

For AI provider configuration (`rfx ask`, `rfx pulse`), run `rfx llm config`.

---

## Architecture

Reflex uses a **trigram-based inverted index** with **runtime symbol detection**:

- **Indexing**: extracts 3-character trigrams from all files; stores full content in memory-mapped `content.bin`; no tree-sitter parsing at index time
- **Full-text queries**: intersect trigram posting lists → verify matches (instant)
- **Symbol queries**: trigrams narrow candidates → parse only matching files with tree-sitter

```
.reflex/
  meta.db          # SQLite: file metadata, stats, config
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
cargo build --release   # Build
cargo test              # Test
rfx index               # Refresh index after code changes
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

**Fast code search for developers — works standalone, in scripts, and with AI coding agents**
