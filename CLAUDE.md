# CLAUDE.md

## Ground Rules (Claude: ALWAYS follow)

### 🚨 CRITICAL: Tool Selection

**Before using ANY search tool, check if Reflex MCP tools are available (`mcp__reflex__*`). These should be preferred
over built-in tools.**

If you see a message like `Index not found. Run 'rfx index' to build the cache first`, run `mcp__reflex__index_project`
immediately, and once the indexing completes, run the previously failed tool again.

**Not sure which MCP tool to use?** See the [MCP Tool Selection Cheatsheet](./docs/mcp-tool-cheatsheet.md) for a decision tree organized by goal: finding locations, definitions, file dependencies, and codebase structure.

## Project Overview
**Reflex** is a local-first, full-text code search engine written in Rust. It's a fast, deterministic replacement for Sourcegraph Code Search, designed specifically for AI coding workflows and automation.

Reflex uses **trigram-based indexing** to enable sub-100ms full-text search across large codebases (10k+ files). Unlike symbol-only tools, Reflex finds **every occurrence** of patterns—function calls, variable usage, comments, and more—not just definitions. Results include file paths, line numbers, and surrounding context, with optional symbol-aware filtering.

---

## Core Principles
1. **Local-first**: Runs fully offline; all data stays on the developer's machine
2. **Complete coverage**: Finds every occurrence, not just symbol definitions
3. **Deterministic results**: Same query → same answer; no probabilistic ranking
4. **Instant access**: Trigram index + memory-mapping enables sub-100ms queries
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

# JSON output for AI agents
rfx query "format!" --json
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
```

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

**18 languages** with full symbol extraction support (functions, classes, variables, etc.):

- **Systems**: Rust, C, C++, Zig
- **Backend**: Python, Go, Java, C#, PHP, Ruby, Kotlin
- **Frontend**: TypeScript, JavaScript, Vue, Svelte
- **Swift**: Temporarily disabled (tree-sitter version incompatibility)

**Symbol extraction**: Functions, classes, methods, variables (global + local), interfaces, traits, enums, attributes/annotations, and more.

**Special features**:
- **React/JSX**: Components, hooks, TypeScript support
- **Attributes/Annotations**: `--kind Attribute` finds annotation definitions (Rust proc macros, Java @interface, Kotlin annotation class, PHP #[Attribute], C# Attribute classes)

**Coverage**: 90%+ of all codebases across web, mobile, systems, enterprise, and AI/ML development.

**Note**: Full-text trigram search works for **all file types** regardless of parser support.

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
- **Language field forward-compatibility**: The `language` field in `SearchResult` serializes as an enum string (e.g. `"Rust"`, `"Python"`). When a file type is unrecognized, the field serializes as `"Unknown"`. Callers **must not** exhaustively match on this field without a fallback — treat `"Unknown"` as the forward-compatible sentinel for any language Reflex does not yet recognise. New languages may be added in minor releases.

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
- `[index]`: Languages, file size limits, symlink handling
- `[search]`: Default result limits, fuzzy matching thresholds
- `[performance]`: Thread count, compression levels

**Example**:
```toml
[index]
languages = []  # Empty = all supported languages
max_file_size = 10485760  # 10 MB

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
