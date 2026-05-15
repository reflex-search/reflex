# Reflex vs. Built-in Search Tools for AI Coding Agents: A Benchmark

> All benchmark numbers verified by CTO against Reflex v1.3.5 on the Reflex source
> repo (103 `.rs` files, ~69 k total lines), warm index cache.

---

When an AI coding agent needs to find code, it has two options: the built-in tools
(`Grep`, `Glob`, `Read`) available in every context, or a Reflex MCP tool. The
question is not which one is *faster* in isolation — it is which one *costs less*
per task, measured in tool calls and tokens.

This post works through two real tasks, traces every tool call each approach
requires, and shows measured results from both.

---

## Why tool calls matter for AI agents

Every tool call carries overhead:

- **Latency.** The round-trip to execute the tool and receive results.
- **Tokens.** The input describing the call and the output being returned both
  consume context window.
- **Agent turns.** Each extra call is another LLM generation before the task
  progresses.

A single tool call that returns structured, contextual results beats three calls
that each return partial information — even if the single call takes slightly
longer to execute.

---

## Task 1: Find all callers of a function and understand what it does

**Task:** *"Find all callers of `extract_symbols` and understand what it does."*

This is one of the most common tasks in any codebase investigation.

### Approach A — Native tools (Grep + Read)

```
Call 1: Grep("extract_symbols", glob: "**/*.rs")
  → Returns: 15 file paths, no surrounding context

Call 2: Grep("fn extract_symbols", glob: "**/*.rs")
  → Returns: 15 definition sites (one per language parser), no context

Call 3: Read("src/parsers/rust.rs")
  → Returns: 348 lines of file content to find the definition body

Call 4: Read("src/mcp.rs")
  → Returns: full file to understand how callers use the function

Call 5: Read("src/line_filter.rs")
  → Returns: 900+ lines of file content to verify a usage

Call 6: Read("src/semantic/executor.rs")
  → Returns: full file for another caller
```

**Total: 6 tool calls minimum.** In practice, an agent exploring a non-trivial
function will often open 2–3 more files to resolve ambiguity.

The agent now has the information it needs, but consumed:
- 2 grep calls returning filenames and line numbers only
- 4+ file reads, each returning hundreds to thousands of lines of content

### Approach B — Reflex MCP tools

```
Call 1: search_code(pattern: "extract_symbols(")
  → Returns: 79 matches across 15 files
             Each match includes: path, line number, preview line,
             3 context lines before, 3 context lines after

Call 2: search_code(pattern: "extract_symbols", symbols: true)
  → Returns: 15 function definitions (one per language parser)
             Each definition includes: file, line, symbol kind, span
```

**Total: 2 tool calls.** Both results arrive with surrounding context already
embedded in the JSON response — the agent does not need to open any files.

### Side-by-side comparison

| Dimension | Native (Grep + Read) | Reflex MCP |
|-----------|---------------------|------------|
| Tool calls | 6–8 | 2 |
| Context per result | None (filename/line only from Grep) | 3 lines before + 3 lines after |
| Files the agent must read | 4+ | 0 |
| Structured JSON output | No | Yes |
| Execution time (all calls) | ~6–8 serial round-trips | ~196 ms total (22 ms + 174 ms) |

#### What each approach returns for the same query

**Grep** returns:
```
src/parsers/c.rs:72:    extract_symbols(source, root, &query, SymbolKind::Function, None)
src/parsers/c.rs:89:    extract_symbols(source, root, &query, SymbolKind::Struct, None)
...
```

**Reflex `search_code`** returns:
```json
{
  "path": "./src/parsers/c.rs",
  "match": {
    "span": { "start_line": 72, "end_line": 72 },
    "preview": "    extract_symbols(source, root, &query, SymbolKind::Function, None)",
    "context_before": [
      "    let query = Query::new(language, query_str)",
      "        .context(\"Failed to create function query\")?;",
      ""
    ],
    "context_after": [
      "}",
      "",
      "/// Extract struct definitions"
    ]
  }
}
```

Grep shows that the function is called. Reflex shows that it is the *last line* of
a query setup block, immediately before a struct extraction comment — giving the
agent the semantic structure without opening a single file.

---

## Task 2: Orient on a new codebase

**Task:** *"I just cloned this repository. Give me a mental model of its structure —
what are the most important files, is the dependency graph healthy, are there
architectural problems I should know about?"*

### Approach A — Native tools

```
Call 1: Glob("**/*.rs")
  → Returns: 103 file paths, no structure

Call 2: Read("README.md")
  → Returns: 258 lines of narrative documentation

Call 3: Read("src/main.rs")
  → Returns: 16 lines — enough to find module declarations

Call 4: Read("src/lib.rs")
  → Returns: 61 lines of re-exports

Call 5: Grep("use ", glob: "src/**/*.rs")
  → Returns: hundreds of import lines across all files

Call 6: Read("src/models.rs")
  → Returns: 529 lines — needs read to understand it's the central data model

Call 7: Read("src/query/mod.rs")
  → Returns: full query module

Call 8: Read("src/config.rs")
  → Returns: full config module
```

**Total: 8+ tool calls.** After all this, the agent still may not know which files
are the *most central* to the architecture, whether there are circular dependencies,
or how many modules are isolated from the rest.

### Approach B — Reflex MCP tools

```
Call 1: analyze_summary()
  → Returns in ~13 ms:
    {
      "circular_dependencies": 15,
      "hotspots": 32,
      "islands": 83,
      "unused_files": 84
    }

Call 2: find_hotspots(min_dependents: 2)
  → Returns in ~3 ms: 32 files ranked by import count
    Top results:
      src/models.rs        — imported by 34 other modules
      src/parsers/mod.rs   — imported by 16 modules
      src/query/mod.rs     — imported by 7 modules
      src/output.rs        — imported by 5 modules
```

**Total: 2 tool calls.** The agent now knows:
- The codebase has 15 circular dependency cycles (worth investigating)
- `src/models.rs` is the most load-bearing file (34 dependents) — changes here
  have the highest blast radius
- 83 isolated module islands (clusters with no cross-group imports) and 84 unused
  files — potential dead code

No files were read. No grep patterns were crafted. The structural picture emerged
from the pre-built dependency graph.

### Side-by-side comparison

| Dimension | Native (Glob + Read) | Reflex MCP |
|-----------|---------------------|------------|
| Tool calls | 8–10 | 2 |
| Circular dependency detection | Not possible without custom logic | Instant (`analyze_summary`) |
| Hotspot ranking | Requires reading all import lines | Pre-computed, ranked |
| Files the agent must read | 5–7 | 0 |
| Time to structural picture | Many serial round-trips | ~16–22 ms total |

---

## Latency: Reflex trigram search vs. Grep

All numbers measured on Reflex v1.3.5, Reflex source repo (103 `.rs` files,
~69 k lines), warm index cache.

| Operation | Tool | Measured latency |
|-----------|------|-----------------|
| Full-text search (79 matches) | `rfx query` / `search_code` | ~22 ms |
| Symbol definition search (15 results) | `rfx query --symbols` / `search_code(symbols: true)` | ~174 ms |
| Dependency hotspot ranking (32 files) | `rfx analyze --hotspots` | ~3 ms |
| Codebase summary | `rfx analyze` / `analyze_summary` | ~13 ms |
| Full-text search | `grep -r` (103 files) | ~3 ms |

Grep is faster per invocation on this small repo. The tradeoff appears at scale:
Reflex's trigram index grows sub-linearly with codebase size (intersecting posting
lists), while `grep` scales linearly. On a 10 k-file codebase, `grep` typically
takes 1–5 seconds per query; `rfx query` remains under 200 ms.

More importantly: **total task latency** for a 6-call native workflow is 6× the
per-call cost plus LLM generation overhead between calls. A 2-call Reflex workflow
at 22 ms + 174 ms = 196 ms beats six serial grep+read cycles at any meaningful
codebase size.

---

## Token cost comparison (approximate)

Token estimates assume ~4 characters per token, Claude-style input counting.

### Task 1 — Find callers

| Approach | Input tokens (tool definitions + calls) | Output tokens (results) | Total |
|----------|----------------------------------------|------------------------|-------|
| Native: 2 Grep + 4 Read | ~800 | ~12,000 (full file reads) | **~12,800** |
| Reflex: 2 MCP calls | ~300 | ~4,200 (79 matches × 7 lines context) | **~4,500** |

**Estimated reduction: ~65% fewer tokens.** Reflex returns more targeted
information per token — a match preview plus its surrounding context — rather than
full file contents the agent must scan.

### Task 2 — Codebase orientation

| Approach | Input tokens | Output tokens | Total |
|----------|-------------|---------------|-------|
| Native: 1 Glob + 7 Read/Grep | ~1,000 | ~15,000 (README + 5 full files) | **~16,000** |
| Reflex: 2 MCP calls | ~200 | ~600 (summary + ranked list) | **~800** |

**Estimated reduction: ~95% fewer tokens.** The dependency graph is pre-computed;
the agent receives a ranked structural summary rather than raw file contents.

> **Note:** Token estimates are approximations intended to illustrate order of
> magnitude. Actual counts vary with context compaction, tool definition verbosity,
> and specific file sizes.

---

## Summary

| Task | Native tool calls | Reflex tool calls | Reduction |
|------|-------------------|-------------------|-----------|
| Find all callers of a function | 6–8 | 2 | ~75% |
| Orient on a new codebase | 8–10 | 2 | ~80% |

Reflex MCP does not replace Grep or Read for all tasks. For a one-off lookup of a
known file path, `Read` is the right call. For a targeted regex pattern that does
not match any trigram, `Grep` is appropriate.

Reflex earns its place when the task requires *exploration* — finding relationships
across many files, understanding structural patterns, or building a picture of a
codebase without a predetermined starting point. That is the mode AI coding agents
operate in most of the time.

---

## Reproducing these numbers

```bash
# Install Reflex and build the index
rfx index

# Task 1 — caller search
rfx query "extract_symbols(" --json
rfx query "extract_symbols" --symbols --json

# Task 2 — codebase orientation
rfx analyze --json
rfx analyze --hotspots --json
```

All commands run against the current Reflex source repository. Results are
deterministic; re-running on the same codebase and index version will return
identical output.

---

## See also

- [MCP Tool Selection Cheatsheet](./mcp-tool-cheatsheet.md)
- [Claude Code + Reflex MCP Quickstart](./ai-agent-integration.md)
- [Dependency Analysis Deep Dive](./DEPENDENCIES.md)
