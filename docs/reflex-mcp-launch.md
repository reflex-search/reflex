# Reflex MCP: Full-Codebase Search and Dependency Analysis for AI Coding Agents

Reflex now ships a complete [Model Context Protocol](https://modelcontextprotocol.io/) server that gives AI coding agents
instant, deterministic access to your entire codebase — not just symbols, but every occurrence, every import, and the full
dependency graph — all locally, with no network calls.

---

## The problem with "code search" for AI agents

Most search tools available to AI coding agents today fall into two categories:

1. **Symbol-only tools** (go-to-definition, LSP): great for finding where a function is declared, useless for finding
   every place it's called, every file that imports it, or patterns that span comments and strings.

2. **Grep wrappers**: comprehensive, but expensive — they hand the agent raw text output that burns context tokens and
   require the agent to parse structure that could be pre-extracted.

Reflex is different. It uses a **trigram-based inverted index** (the same approach behind Google Code Search and
Sourcegraph) to pre-index every 3-character sequence in your codebase. Searches resolve in milliseconds, not seconds.
Results are structured JSON. And because everything runs locally, your code never leaves your machine.

---

## What's new: 17 MCP tools

The Reflex MCP server (`rfx mcp`) exposes 17 tools organized into three layers:

### Search

| Tool | What it does |
|------|-------------|
| `search_code` | Full-text or symbol-only search with file paths, line numbers, code previews, and pagination |
| `search_regex` | Regex pattern matching with trigram pre-filtering — fast even on large codebases |
| `search_ast` | Structure-aware Tree-sitter S-expression matching for when text search isn't precise enough |
| `list_locations` | Returns only `{path, line}` pairs — the cheapest way to enumerate all match sites |
| `count_occurrences` | Quick `{total, files}` stats without loading any code |
| `find_references` | Symbol definition **and** all usage sites in a single atomic call |

The tiered design matters: agents can start with `list_locations` (2–5ms, minimal tokens), escalate to `search_code` for
previews, and fall back to `search_ast` only when structural matching is needed.

### Dependency graph

| Tool | What it does |
|------|-------------|
| `get_dependencies` | All static imports for a file, classified as internal / external / stdlib |
| `get_dependents` | Reverse lookup: every file that imports a given module |
| `get_transitive_deps` | Full dependency tree up to N levels deep |
| `find_hotspots` | Files ranked by how many other files depend on them |
| `find_circular` | Circular dependency cycles (A → B → C → A) |
| `find_unused` | Files that nothing imports — candidates for cleanup |
| `find_islands` | Isolated subsystems with no cross-group imports |
| `analyze_summary` | One-call health check: counts for circular deps, hotspots, unused files, islands |

### Context and index management

| Tool | What it does |
|------|-------------|
| `gather_context` | Project structure, detected frameworks, entry points, and file-type distribution |
| `check_index_status` | Returns `fresh` / `stale` / `missing` — agents call this at session start |
| `index_project` | Build or incrementally update the index; `force: true` for full rebuild |

---

## Get started in 2 minutes

**Step 1 — Install Reflex:**

```bash
npm install -g reflex-search
# or
cargo install reflex-search
```

**Step 2 — Add to Claude Code:**

Edit `~/.claude/claude_code_config.json` (global) or `.claude/claude_code_config.json` (per-project):

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

Restart Claude Code (or use **Reload Window** in VS Code) to pick up the new server.

**Step 3 — Index your project:**

```bash
cd /your/project
rfx index
```

Indexing is incremental: the first run takes a few seconds; subsequent runs only process changed files. That's it — Reflex
is now available in every Claude Code conversation.

**Step 4 — Ask Claude about your code:**

> *"Where is the `process_payment` function called, and what other files depend on the module that defines it?"*

Claude invokes `find_references` and `get_dependents` automatically — no shell commands, no manual grep.

---

## Why Reflex for AI agents?

**Local-first.** The index lives in `.reflex/` alongside your code. No telemetry, no cloud indexing, no API keys required
for search. Your codebase stays on your machine.

**Deterministic.** The same query always returns the same results. AI agents reason better when search is not
probabilistic.

**Complete coverage.** Trigram search finds every occurrence — function calls, comments, string literals, configuration
— not just symbol definitions. Symbol-only tools miss the call sites that matter most during refactoring.

**Token-efficient design.** `list_locations` returns only `{path, line}` pairs. `count_occurrences` returns only counts.
Agents pay token cost proportional to what they actually need.

**Dependency graph built in.** Static import analysis is indexed alongside the trigram index, so agents can ask "what
breaks if I change this file?" without a separate tool.

---

## Links

- **[Claude Code + Reflex MCP Quickstart](./ai-agent-integration.md)** — step-by-step setup, key tool examples, and
  troubleshooting
- **[MCP Tool Selection Cheatsheet](./mcp-tool-cheatsheet.md)** — decision tree for choosing the right tool
- **[README](../README.md)** — CLI usage, configuration, and supported languages
- **[GitHub](https://github.com/reflex-search/reflex)** — source, issues, and releases

---

*Reflex is open source (MIT). Contributions welcome.*
