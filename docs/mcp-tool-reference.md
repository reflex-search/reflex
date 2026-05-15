# Reflex MCP Tool Reference

Complete reference for all Reflex MCP tools, organized by agent intent.

> **Quick orientation:** Start at [Index Management](#index-management) — no other tool works without a valid index. Then jump to the section that matches your goal.

---

## Table of Contents

- [Index Management](#index-management) — `check_index_status`, `index_project`
- [Search Tools](#search-tools) — `list_locations`, `count_occurrences`, `search_code`, `search_regex`, `search_ast`
- [Symbol & Reference Tools](#symbol--reference-tools) — `find_references`
- [Dependency Tools](#dependency-tools) — `get_dependencies`, `get_dependents`, `get_transitive_deps`, `find_hotspots`, `find_circular`, `find_unused`, `find_islands`, `analyze_summary`
- [Codebase Overview](#codebase-overview) — `gather_context`
- [Common Filters](#common-filters)
- [Error Handling](#error-handling)

---

## Index Management

The Reflex index must exist and be fresh before any search or dependency tool will work. Call `check_index_status` at the start of every agent session.

---

### `check_index_status`

Check whether the search index is fresh, stale, or missing — without running any search.

**Call this first** at the start of every session. If the status is anything other than `fresh`, call `index_project` before searching.

**Parameters:** none

**Example call:**
```json
{}
```

**Example responses:**
```json
{ "status": "fresh" }

{ "status": "stale", "reason": "Commit changed from abc1234 to def5678", "action_required": "rfx index" }

{ "status": "stale", "reason": "12 files modified since last index", "action_required": "rfx index", "files_modified": 12 }

{ "status": "missing", "action_required": "rfx index" }
```

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"fresh"` \| `"stale"` \| `"missing"` |
| `reason` | string | Why the index is stale (only present when stale) |
| `action_required` | string | Command to fix the issue (`"rfx index"` when stale/missing) |
| `files_modified` | integer | Number of recently modified files (only present for file-change staleness) |

**When to call:**
- At the start of every agent session
- Before any bulk search or refactoring task
- After a git operation (checkout, merge, rebase, pull)

**When NOT to use:** This tool does not rebuild the index — use `index_project` for that.

---

### `index_project`

Rebuild or update the code search index.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force` | boolean | `false` | Force full rebuild (ignore incremental cache) |
| `languages` | string[] | `[]` (all) | Languages to index; empty means all supported languages |

**Example calls:**
```json
{}

{ "force": true }

{ "languages": ["rust", "typescript"] }
```

**Example response:**
```json
{
  "files_indexed": 342,
  "files_skipped": 1205,
  "duration_ms": 1840,
  "trigrams_written": 2100000
}
```

**When to use:**
- After `check_index_status` returns `stale` or `missing`
- After any tool returns an error containing `"Index not found"` — call immediately, then retry
- After `git pull`, `git checkout`, or large file changes
- Default (no params): incremental — only re-indexes changed files (fast)
- `force: true`: full rebuild — use if results seem wrong after incremental

**Modes:**
- **Incremental** (default): Detects changed files via `blake3` hash; fast on subsequent runs
- **Full rebuild** (`force: true`): Clears `.reflex/` and rebuilds from scratch; use when the index is suspected corrupt

---

## Search Tools

Search tools answer the question "where is X and what does it look like?" They operate on the trigram index and return results in under 100ms for most queries.

### Choosing the right search tool

```
Pattern has only alphanumeric / underscore chars?
  → search_code (or list_locations / count_occurrences for cheap variants)

Pattern has special chars like -> :: () [] {} . * + ? | ^ $?
  → search_regex

Need structural match (e.g., "all async fns that contain a match")?
  → search_ast  ⚠️ slow — last resort only
```

---

### `list_locations`

Fast location discovery: returns `{path, line}` only, with no code content.

**Use this first** when you need to find where a pattern occurs before deciding which files to read. Lowest token cost of all search tools.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | string | **required** | Text to find |
| `lang` | string | — | Filter by language (`rust`, `typescript`, `python`, etc.) |
| `file` | string | — | Filter by file path substring (e.g., `"Controllers"`) |
| `glob` | string[] | — | Include only files matching these glob patterns |
| `exclude` | string[] | — | Exclude files matching these glob patterns |
| `force` | boolean | `false` | Bypass broad-query detection |
| `dependencies` | boolean | `false` | Include static import info in results |

**No limit:** Returns ALL matching locations.

**Example call:**
```json
{ "pattern": "CacheManager" }
```

**Example response:**
```json
{
  "status": "fresh",
  "total_locations": 14,
  "locations": [
    { "path": "src/cache.rs", "line": 42 },
    { "path": "src/query.rs", "line": 18 },
    { "path": "src/mcp.rs", "line": 703 }
  ]
}
```

**When to use vs. alternatives:**

| Goal | Tool |
|------|------|
| "Where is X?" — just need file + line | **`list_locations`** ← cheapest |
| "Where is X?" — need code previews too | `search_code` |
| "How many times does X appear?" | `count_occurrences` |

**Workflow:**
1. Use `list_locations` to discover (cheap, locations only)
2. Use the `Read` tool or `search_code` on specific files if you need content

---

### `count_occurrences`

Count how many times a pattern appears without loading any content.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | string | **required** | Text to count |
| `lang` | string | — | Filter by language |
| `symbols` | boolean | `false` | Count symbol definitions only (not usages) |
| `kind` | string | — | Filter by symbol kind (`function`, `class`, etc.) |
| `file` | string | — | Filter by file path substring |
| `glob` | string[] | — | Include files matching these patterns |
| `exclude` | string[] | — | Exclude files matching these patterns |
| `force` | boolean | `false` | Bypass broad-query detection |
| `dependencies` | boolean | `false` | Include static import info |

**Example call:**
```json
{ "pattern": "unwrap", "lang": "rust" }
```

**Example response:**
```json
{
  "status": "fresh",
  "pattern": "unwrap",
  "total": 87,
  "files": 12
}
```

**When to use:**
- Quick scope check before a refactor ("how widespread is this?")
- Validating that a pattern was removed (`total: 0`)
- Comparing symbol definition count across namespaces

**When NOT to use:** When you need code content — use `search_code` instead.

---

### `search_code`

Full-text or symbol-definition search with line numbers and code previews. The workhorse search tool.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | string | **required** | Text to find |
| `lang` | string | — | Filter by language |
| `symbols` | boolean | `false` | Find definitions only (not usages) |
| `kind` | string | — | Filter by symbol kind (`function`, `class`, `struct`, `trait`, `enum`, `interface`, etc.) |
| `exact` | boolean | `false` | Exact match (no substring matching) |
| `file` | string | — | Filter by file path substring |
| `glob` | string[] | — | Include files matching these patterns |
| `exclude` | string[] | — | Exclude files matching these patterns |
| `limit` | integer | `100` | Results per page |
| `offset` | integer | `0` | Pagination offset |
| `expand` | boolean | `false` | Show full symbol body (not just signature) |
| `paths` | boolean | `false` | Return only unique file paths |
| `force` | boolean | `false` | Bypass broad-query detection |
| `dependencies` | boolean | `false` | Include static import info per file |
| `preview_length` | integer | `180` | Max characters per preview line |

**Example calls:**
```json
{ "pattern": "extract_symbols" }

{ "pattern": "extract_symbols", "symbols": true }

{ "pattern": "CacheManager", "lang": "rust", "symbols": true, "kind": "struct" }

{ "pattern": "process_request", "symbols": true, "expand": true }

{ "pattern": "TODO", "glob": ["src/**/*.rs"], "exclude": ["target/**"] }
```

**Example response (full-text mode):**
```json
{
  "status": "fresh",
  "results": [
    {
      "path": "src/query.rs",
      "lang": "Rust",
      "matches": [
        {
          "kind": "Function",
          "symbol": "search_with_metadata",
          "span": { "start_line": 145, "start_col": 0, "end_line": 145, "end_col": 0 },
          "preview": "pub fn search_with_metadata(&self, pattern: &str, filter: QueryFilter) -> Result<SearchResponse> {"
        }
      ]
    }
  ],
  "pagination": {
    "total": 23,
    "count": 23,
    "offset": 0,
    "limit": 100,
    "has_more": false
  },
  "ai_instruction": "Found 23 matches..."
}
```

**Pagination:** Check `pagination.has_more`. If `true`, fetch the next page with `offset = offset + limit`.

**When to use vs. alternatives:**

| Goal | Tool |
|------|------|
| Simple text with alphanumeric chars | **`search_code`** |
| Definitions only | **`search_code`** with `symbols: true` |
| Patterns with `->`, `::`, `()`, `[]` | `search_regex` |
| Structural code patterns | `search_ast` (last resort) |
| Just need file + line, no previews | `list_locations` |

**When NOT to use:** When the pattern contains special regex characters like `->`, `::`, `(`, `)`, `[`, `]`, `{`, `}`, `.`, `*`, `+`, `?`, `\`, `|`, `^`, `$` — use `search_regex` instead.

---

### `search_regex`

Regex-based code search for patterns with special characters.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | string | **required** | Regex pattern |
| `lang` | string | — | Filter by language |
| `file` | string | — | Filter by file path substring |
| `glob` | string[] | — | Include files matching these patterns |
| `exclude` | string[] | — | Exclude files matching these patterns |
| `limit` | integer | `100` | Results per page |
| `offset` | integer | `0` | Pagination offset |
| `paths` | boolean | `false` | Return only unique file paths |
| `force` | boolean | `false` | Bypass broad-query detection |
| `dependencies` | boolean | `false` | Include static import info |

**Escaping rules:**
- Must escape: `( ) [ ] { } . * + ? \ | ^ $`
- No escaping needed: `-> :: - _ / = < >`
- Use double backslash in JSON: `\\(` `\\)` `\\[` `\\]`

**Example calls:**
```json
{ "pattern": "->with\\(" }

{ "pattern": "::new\\(" }

{ "pattern": "fn (get|set)_\\w+" }

{ "pattern": "\\[derive\\(" }

{ "pattern": "#\\[test\\]", "lang": "rust" }
```

**Example response:**
```json
{
  "status": "fresh",
  "results": [
    {
      "path": "src/cache.rs",
      "lang": "Rust",
      "matches": [
        {
          "span": { "start_line": 88, "start_col": 0, "end_line": 88, "end_col": 0 },
          "preview": "    cache.with(key, |v| v.clone())"
        }
      ]
    }
  ],
  "pagination": { "total": 5, "count": 5, "offset": 0, "limit": 100, "has_more": false }
}
```

**When to use vs. alternatives:**

| Goal | Tool |
|------|------|
| Pattern with special characters | **`search_regex`** |
| Simple alphanumeric text | `search_code` (faster) |
| Symbol definitions | `search_code` with `symbols: true` |

---

### `search_ast`

> **⚠️ ADVANCED / SLOW — use `search_code` with `symbols: true` in 95% of cases.**

Structure-aware search using Tree-sitter AST patterns (S-expressions). Bypasses trigram optimization and scans the entire codebase: **500ms–10s+**.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | string | **required** | Tree-sitter S-expression (e.g., `"(function_item) @fn"`) |
| `lang` | string | **required** | Language (`rust`, `typescript`, `javascript`, `python`, `go`, `java`, `c`, `cpp`, `php`, `ruby`, `kotlin`, `zig`) |
| `glob` | string[] | — | **Strongly recommended** — without this, entire codebase is scanned |
| `exclude` | string[] | — | Exclude files matching these patterns |
| `file` | string | — | Filter by file path substring |
| `limit` | integer | `100` | Results per page |
| `offset` | integer | `0` | Pagination offset |
| `paths` | boolean | `false` | Return only unique file paths |
| `force` | boolean | `false` | Bypass broad-query detection |
| `dependencies` | boolean | `false` | Include static import info |

**Example calls:**
```json
{ "pattern": "(function_item) @fn", "lang": "rust", "glob": ["src/**/*.rs"] }

{ "pattern": "(class_declaration) @class", "lang": "typescript", "glob": ["src/**/*.ts"] }

{ "pattern": "(function_definition) @fn", "lang": "python", "glob": ["app/**/*.py"] }
```

**AST pattern examples by language:**

| Language | Pattern | Matches |
|----------|---------|---------|
| Rust | `(function_item) @fn` | All function definitions |
| Rust | `(impl_item) @impl` | All impl blocks |
| TypeScript | `(class_declaration) @class` | All class declarations |
| Python | `(function_definition) @fn` | All function definitions |
| Go | `(function_declaration) @fn` | All function declarations |

> Refer to [Tree-sitter documentation](https://tree-sitter.github.io) for each language's grammar and available node types.

**When to use (rare):**
- You need structural matching that text search cannot express (e.g., "all async functions that contain a `match` expression")
- `search_code` with `symbols: true` is insufficient for your pattern

**Always add `glob`** to limit scope — without it, the entire codebase is scanned.

**Error case:** If you get a timeout error, add a `glob` filter to narrow the search scope.

---

## Symbol & Reference Tools

### `find_references`

Find a symbol's definition AND all usage sites in a single call.

This eliminates the two-step pattern of `search_code(symbols: true)` followed by `search_code()` — it runs both internally and returns the combined result atomically.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | string | **required** | Symbol name or text pattern |
| `kind` | string | — | Filter definition lookup by symbol kind (`function`, `class`, `struct`, `trait`, etc.) |
| `lang` | string | — | Filter by language |
| `glob` | string[] | — | Include files matching these patterns |
| `exclude` | string[] | — | Exclude files matching these patterns |
| `limit` | integer | `100` | Max references per page (pagination applies to references only) |
| `offset` | integer | `0` | Pagination offset for references |
| `force` | boolean | `false` | Bypass broad-query detection |

**Example calls:**
```json
{ "pattern": "CacheManager" }

{ "pattern": "extract_symbols", "kind": "function", "lang": "rust" }

{ "pattern": "authenticate", "glob": ["src/**/*.rs"], "exclude": ["tests/**"] }
```

**Example response:**
```json
{
  "status": "fresh",
  "definition": {
    "path": "src/cache.rs",
    "line": 42,
    "kind": "Struct",
    "symbol": "CacheManager",
    "span": { "start_line": 42, "end_line": 58 },
    "preview": "pub struct CacheManager {"
  },
  "references": [
    { "path": "src/query.rs", "line": 18, "preview": "    let cache = CacheManager::new(\".\");" },
    { "path": "src/mcp.rs", "line": 703, "preview": "    let cache = CacheManager::new(\".\");" },
    { "path": "src/indexer.rs", "line": 31, "preview": "pub struct Indexer { cache: CacheManager," }
  ],
  "total_references": 14,
  "pagination": { "total": 14, "count": 14, "offset": 0, "limit": 100, "has_more": false }
}
```

**Response fields:**

| Field | Description |
|-------|-------------|
| `definition` | First matching symbol definition, or `null` if no definition found |
| `references` | Flat array of all textual occurrences (includes the definition site) |
| `total_references` | Total reference count across all pages |
| `pagination` | Pagination metadata — applies to references only |

**When to use:**
- "Find all callers of function X" — the most common agent refactoring pattern
- Rename planning: see every site that needs updating
- Dead code detection: confirm nothing calls a function before removing it
- Code review: understand impact before changing a function or class

**When NOT to use:** When you only need one of definition or references — use `search_code(symbols: true)` for definitions alone or `search_code` for references alone.

**Pagination note:** `limit` and `offset` apply to the `references` array only. The definition lookup always returns the first matching symbol.

---

## Dependency Tools

Dependency tools analyze the static import graph — the `import`/`require`/`use` statements in your source files.

> **Important:** All dependency tools extract **static imports only** (string literals). Dynamic imports — variables, template literals, computed expressions like `importlib.import_module(var)` — are intentionally filtered. This keeps the graph deterministic and fast.

---

### `get_dependencies`

Get all files that a given file imports.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | string | **required** | File path (fuzzy matching: `"src/query.rs"`, `"query.rs"`, or `"query"` all work) |

**Example call:**
```json
{ "path": "src/query.rs" }
```

**Example response:**
```json
[
  {
    "import_path": "crate::cache",
    "line": 5,
    "kind": "internal",
    "symbols": ["CacheManager"]
  },
  {
    "import_path": "serde_json",
    "line": 3,
    "kind": "external",
    "symbols": null
  },
  {
    "import_path": "std::collections::HashMap",
    "line": 1,
    "kind": "stdlib",
    "symbols": null
  }
]
```

**`kind` values:** `internal` (project code), `external` (third-party packages), `stdlib` (standard library)

**When to use:**
- Understanding what a file depends on
- Auditing third-party dependencies used by a specific module
- First step in impact analysis ("what does this file pull in?")

---

### `get_dependents`

Get all files that import a given file (reverse dependency lookup).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | string | **required** | File path (fuzzy matching supported) |

**Example call:**
```json
{ "path": "src/cache.rs" }
```

**Example response:**
```json
[
  "src/query.rs",
  "src/indexer.rs",
  "src/mcp.rs",
  "src/dependency.rs"
]
```

**When to use:**
- "What breaks if I change this file?" — impact analysis before refactoring
- Finding all consumers of a module before deletion
- Understanding the blast radius of an API change

---

### `get_transitive_deps`

Get the full dependency tree of a file up to N levels deep.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | string | **required** | File path (fuzzy matching supported) |
| `depth` | integer | `3` | Maximum depth to traverse (recommended max: 5) |

**Example call:**
```json
{ "path": "src/main.rs", "depth": 2 }
```

**Example response:**
```json
[
  { "path": "src/cli.rs", "depth": 1 },
  { "path": "src/query.rs", "depth": 1 },
  { "path": "src/cache.rs", "depth": 2 },
  { "path": "src/models.rs", "depth": 2 }
]
```

**When to use:**
- Understanding the full dependency chain (not just direct imports)
- Planning refactoring impact across a module tree
- Auditing what gets pulled in when a file is imported

**When NOT to use:** For simple "what does this file import?" — use `get_dependencies` (one level, faster).

---

### `find_hotspots`

Find the most-imported files in the codebase, ranked by import count.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | `200` | Results per page |
| `offset` | integer | `0` | Pagination offset |
| `min_dependents` | integer | `2` | Minimum import count to include |
| `sort` | string | `"desc"` | Sort order: `"desc"` (most imports first) or `"asc"` (fewest first) |

**Example call:**
```json
{ "limit": 10, "min_dependents": 5 }
```

**Example response:**
```json
{
  "pagination": { "total": 23, "count": 10, "offset": 0, "limit": 10, "has_more": true },
  "results": [
    { "path": "src/models.rs", "import_count": 27 },
    { "path": "src/cache.rs", "import_count": 18 },
    { "path": "src/errors.rs", "import_count": 14 }
  ]
}
```

**When to use:**
- Identifying critical/load-bearing modules
- Finding potential bottlenecks before architectural changes
- Prioritizing test coverage for the most-shared files
- Understanding which files have the highest refactoring risk

---

### `find_circular`

Detect circular dependency cycles in the codebase.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | `200` | Cycles per page |
| `offset` | integer | `0` | Pagination offset |
| `sort` | string | `"desc"` | `"desc"` = longest cycles first, `"asc"` = shortest first |

**Example call:**
```json
{}
```

**Example response:**
```json
{
  "pagination": { "total": 3, "count": 3, "offset": 0, "limit": 200, "has_more": false },
  "results": [
    { "paths": ["src/a.rs", "src/b.rs", "src/a.rs"] },
    { "paths": ["src/x.rs", "src/y.rs", "src/z.rs", "src/x.rs"] }
  ]
}
```

**When to use:**
- Debugging compilation errors caused by dependency cycles
- Validating that a refactor didn't introduce cycles
- Architecture review — cycles indicate tight coupling

**What `total: 0` means:** No circular dependencies detected — a healthy sign.

---

### `find_unused`

Find files that no other file imports (orphaned/dead files).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | `200` | Files per page |
| `offset` | integer | `0` | Pagination offset |

**Example call:**
```json
{}
```

**Example response:**
```json
{
  "pagination": { "total": 8, "count": 8, "offset": 0, "limit": 200, "has_more": false },
  "results": [
    "src/old_parser.rs",
    "src/experiments/sketch.rs",
    "tests/integration_legacy.rs"
  ]
}
```

**When to use:**
- Cleaning up dead code
- Reducing codebase size before a release

**Important caveats:**
- Entry points (`main.rs`, `index.ts`, `__main__.py`) will appear as unused — do not delete them
- Binary targets and test files are often "unused" by design
- Always verify intent before deleting — a file might be a standalone script or CLI entry

---

### `find_islands`

Find disconnected components (isolated groups) in the import graph.

An "island" is a set of files that all import each other but have no imports to or from the rest of the codebase.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | `200` | Islands per page |
| `offset` | integer | `0` | Pagination offset |
| `min_island_size` | integer | `2` | Minimum files in an island to include |
| `max_island_size` | integer | `500` or 50% of total files | Maximum files in an island to include |
| `sort` | string | `"desc"` | `"desc"` = largest islands first, `"asc"` = smallest first |

**Example call:**
```json
{ "min_island_size": 3, "limit": 5 }
```

**Example response:**
```json
{
  "pagination": { "total": 12, "count": 5, "offset": 0, "limit": 5, "has_more": true },
  "results": [
    {
      "island_id": 1,
      "size": 8,
      "paths": ["src/plugin/a.rs", "src/plugin/b.rs", "src/plugin/c.rs", "..."]
    },
    {
      "island_id": 2,
      "size": 4,
      "paths": ["tools/gen.rs", "tools/gen_utils.rs", "tools/template.rs", "tools/output.rs"]
    }
  ]
}
```

**When to use:**
- Identifying isolated subsystems or feature modules
- Finding code that could be extracted into a separate package
- Understanding codebase modularity
- Detecting disconnected features that may have been abandoned

---

### `analyze_summary`

Get a one-shot health check of the entire dependency graph.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_dependents` | integer | `2` | Minimum dependent count for hotspot detection |

**Example call:**
```json
{}
```

**Example response:**
```json
{
  "circular_dependencies": 3,
  "hotspots": 18,
  "unused_files": 42,
  "islands": 7,
  "min_dependents": 2
}
```

**When to use:**
- Start here when assessing dependency health — get the counts, then drill into specifics
- Quick gut-check before a major refactor
- Understanding overall codebase structure at a glance

**Typical workflow:**
```
analyze_summary()
  → circular_dependencies: 3  → find_circular() to see which files
  → hotspots: 18             → find_hotspots(min_dependents: 5) to find critical modules
  → unused_files: 42         → find_unused() to review candidates
```

---

## Codebase Overview

### `gather_context`

Collect comprehensive codebase information: directory structure, frameworks, entry points, test layout, file types, and configuration files.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `structure` | boolean | `false` | Show directory tree |
| `file_types` | boolean | `false` | Show file type distribution |
| `project_type` | boolean | `false` | Detect project type (CLI / library / webapp / monorepo) |
| `framework` | boolean | `false` | Detect frameworks (React, Django, Axum, etc.) |
| `entry_points` | boolean | `false` | Find main/index files |
| `test_layout` | boolean | `false` | Show test organization pattern |
| `config_files` | boolean | `false` | List important configuration files |
| `depth` | integer | `2` | Tree depth when `structure: true` |
| `path` | string | — | Focus on a specific subdirectory |

> **Default behavior (no parameters):** Returns minimal orientation view — `project_type` + `entry_points` only. Pass specific flags for more detail.

**Example calls:**
```json
{}

{ "structure": true, "framework": true, "entry_points": true }

{ "structure": true, "depth": 3, "path": "src/api" }

{ "file_types": true, "test_layout": true, "config_files": true }
```

**Example response (no params):**
```
Project type: CLI binary
Entry points: src/main.rs

---
Hint: this is the minimal orientation view. Pass any combination of these flags for more
detail: structure, file_types, framework, test_layout, config_files.
```

**When to use:**
- First tool to call when exploring an unfamiliar codebase
- Understanding which frameworks and conventions are in use
- Locating entry points before diving into code

**When NOT to use:**
- Finding specific code patterns — use `search_code` or `list_locations`
- Understanding how a specific feature works — use `search_code` + `find_references`

---

## Common Filters

These parameters are available on most search tools:

| Filter | Type | Example | Notes |
|--------|------|---------|-------|
| `lang` | string | `"rust"`, `"typescript"`, `"python"`, `"go"`, `"java"`, `"php"` | Limits search to one language |
| `glob` | string[] | `["src/**/*.rs"]` | Include only matching files |
| `exclude` | string[] | `["target/**", "node_modules/**"]` | Exclude matching files |
| `file` | string | `"Controllers"` | Substring match on file path |
| `symbols` | boolean | `true` | Definitions only, not usages (`search_code` only) |
| `kind` | string | `"function"`, `"class"`, `"struct"`, `"trait"`, `"enum"`, `"interface"` | Symbol type filter |
| `expand` | boolean | `true` | Show full symbol body (`search_code` only) |
| `limit` | integer | `50` | Max results per page (default: 100) |
| `offset` | integer | `100` | Pagination: skip first N results |
| `paths` | boolean | `true` | Return unique file paths only (no content) |
| `dependencies` | boolean | `true` | Include static import info in results |
| `force` | boolean | `true` | Bypass broad-query guard (rarely needed) |

### Pagination

When `pagination.has_more` is `true` in a response, there are more results. Fetch the next page by adding `offset`:

```
First call:  offset = 0         → get results 0–99
Second call: offset = 100       → get results 100–199
Third call:  offset = 200       → get results 200–299
```

Continue until `has_more: false`.

---

## Error Handling

### "Index not found" or stale index

Any tool can return an error containing `"Index not found"` if the index is missing or corrupt. The correct response is:

1. Call `index_project` immediately (no parameters)
2. Wait for it to complete
3. Retry the original tool call

### JSON-RPC error codes

| Code | Name | Cause |
|------|------|-------|
| `-32700` | Parse error | Malformed JSON in the request |
| `-32601` | Method not found | Unknown JSON-RPC method name |
| `-32602` | Invalid params | Missing required parameter or unsupported language |
| `-32603` | Internal error | Unexpected engine error (report as a bug) |

### Broad query guard

Some queries (very short patterns on large codebases) are rejected to protect performance. If you see a "broad query" error, either:
- Add `lang`, `glob`, or `file` filters to narrow scope
- Set `force: true` to override (use sparingly)

---

## See Also

- [MCP Tool Selection Cheatsheet](./mcp-tool-cheatsheet.md) — decision tree for picking the right tool
- [Claude Code + Reflex MCP Quickstart](./ai-agent-integration.md) — MCP setup, config, and troubleshooting
- [Dependency Analysis](./DEPENDENCIES.md) — deep dive into import extraction and graph analysis
- [CLI Reference](../CLAUDE.md#cli-usage) — `rfx` command reference for humans
