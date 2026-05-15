# Reflex MCP Tool Selection Cheatsheet

> **Quick rule:** Start cheap, escalate only when needed.
> `list_locations` → `search_code` → `search_regex` → `search_ast` (last resort)

---

## Decision Tree by Agent Intent

### "I want to find WHERE something is"

| Goal | Tool | Why |
|------|------|-----|
| Known exact name, just need locations | `list_locations` | Cheapest — returns `{path, line}` only, no content |
| Need locations **and** code previews | `search_code` | Full results with line numbers + context |
| Pattern has special chars (`->`, `::`, `()`, regex) | `search_regex` | Required for non-alphanumeric patterns |
| How many times does X appear? | `count_occurrences` | Returns `{total, files}` — no content loaded |

```
"Where is UserController used?"
  → list_locations(pattern: "UserController")

"How many places call unwrap()?"
  → count_occurrences(pattern: "unwrap()")   # has special chars? no → search_code first
  → count_occurrences + search_regex(pattern: "unwrap\\(")
```

---

### "I want to find WHAT something is (definition)"

| Goal | Tool | Why |
|------|------|-----|
| Symbol definition (function/class/struct) | `search_code(symbols: true)` | Filters to definitions only |
| Symbol definition **with full body** | `search_code(symbols: true, expand: true)` | Shows complete implementation |
| Structural match (e.g., "async fn with error handling") | `search_ast` | ⚠️ SLOW — use only when text search fails |

```
"Find the definition of extract_symbols"
  → search_code(pattern: "extract_symbols", symbols: true)

"Show me the full body of build_index"
  → search_code(pattern: "build_index", symbols: true, expand: true)
```

---

### "I want to understand a FILE"

| Goal | Tool | Why |
|------|------|-----|
| What does this file import? | `get_dependencies` | Returns all static imports with type (internal/external/stdlib) |
| What files import this file? | `get_dependents` | Reverse lookup — impact of changes |
| Full import tree (deps of deps) | `get_transitive_deps` | Traverses N levels deep (default: 3) |

```
"What does src/query.rs depend on?"
  → get_dependencies(path: "src/query.rs")

"What breaks if I change models/User.php?"
  → get_dependents(path: "User.php")

"Show the full dependency chain for main.rs"
  → get_transitive_deps(path: "src/main.rs", depth: 3)
```

> **Note:** All dependency tools extract **static imports only**. Dynamic imports (variables, template literals) are filtered by design.

---

### "I want to understand the CODEBASE"

| Goal | Tool | Why |
|------|------|-----|
| Project type, entry points, frameworks | `gather_context` (no params) | One-shot codebase overview |
| Dependency health at a glance | `analyze_summary` | Returns counts: circular, hotspots, unused, islands |
| Most-imported (critical) files | `find_hotspots` | Files ranked by import count — the load-bearing modules |
| Unused / orphaned files | `find_unused` | Candidates for deletion (verify entry points aren't included) |
| Circular dependency cycles | `find_circular` | Returns cycle arrays: A→B→C→A |
| Isolated subsystems | `find_islands` | Groups of files with no cross-group imports |

```
"What kind of project is this?"
  → gather_context()

"Is the dependency graph healthy?"
  → analyze_summary()
  → then drill into find_circular / find_hotspots / find_unused as needed

"What files are most central to this codebase?"
  → find_hotspots(min_dependents: 3)
```

---

### "I need to maintain the index"

| Goal | Tool | Why |
|------|------|-----|
| Index seems stale / missing files | `index_project` | Incremental by default; use `force: true` for full rebuild |
| Search returns "Index not found" error | `index_project` immediately | Required before any other tool will work |

```
# Always: if any tool returns "Index not found", call this first:
index_project()

# After large git operations (checkout, merge, rebase):
index_project()
```

---

## Tool Quick Reference

| Tool | Cost | Returns | Requires |
|------|------|---------|---------|
| `list_locations` | ⚡ Cheapest | `[{path, line}]` | `pattern` |
| `count_occurrences` | ⚡ Cheap | `{total, files}` | `pattern` |
| `search_code` | 🟡 Medium | Full results with previews | `pattern` |
| `search_regex` | 🟡 Medium | Full results with previews | `pattern` |
| `gather_context` | 🟡 Medium | Project structure summary | — |
| `get_dependencies` | 🟡 Medium | Import list for one file | `path` |
| `get_dependents` | 🟡 Medium | Files importing this one | `path` |
| `get_transitive_deps` | 🟡 Medium | Dep tree up to N levels | `path` |
| `analyze_summary` | 🟡 Medium | Counts: circular/hotspots/unused | — |
| `find_hotspots` | 🟡 Medium | Files by import count | — |
| `find_unused` | 🟡 Medium | Orphaned file list | — |
| `find_circular` | 🟡 Medium | Cycle arrays | — |
| `find_islands` | 🟡 Medium | Isolated component groups | — |
| `index_project` | 🔴 Slow (write) | Status + stats | — |
| `search_ast` | 🔴 Slowest | Structural matches | `pattern`, `lang` + glob |

---

## Tiered Workflow Example

**Task:** "Understand how authentication works in this codebase"

```
# Tier 1 — Orient (cheapest)
list_locations(pattern: "authenticate")
# → 12 matches in 5 files

# Tier 2 — Explore (targeted)
search_code(pattern: "authenticate", symbols: true)
# → 3 function definitions: authenticate(), auth_middleware(), verify_token()

search_code(pattern: "authenticate", symbols: true, expand: true)
# → Full bodies of all 3 definitions

# Tier 3 — Context (if needed)
get_dependencies(path: "src/auth.rs")
# → auth.rs imports: jwt, crypto, models/user

get_dependents(path: "src/auth.rs")
# → 8 files use auth.rs — these are affected if you change it
```

---

## Common Filters (available on most search tools)

| Filter | Type | Example |
|--------|------|---------|
| `lang` | string | `"rust"`, `"typescript"`, `"python"` |
| `glob` | array | `["src/**/*.rs"]` |
| `exclude` | array | `["target/**", "node_modules/**"]` |
| `file` | string | `"Controllers"` (substring match) |
| `symbols` | bool | `true` = definitions only |
| `kind` | string | `"function"`, `"class"`, `"struct"` |
| `expand` | bool | `true` = show full symbol body |
| `limit` / `offset` | int | Pagination (check `has_more` in response) |

---

## When to Use `search_ast` (Rare)

`search_ast` is a **last resort**. Use it only when:
1. Text search (`search_code` / `search_regex`) cannot express the pattern
2. You need structural matching (e.g., "all async functions that contain a `match` expression")
3. You **must** add `glob` to limit scope

```
# Acceptable (narrow glob):
search_ast(
  pattern: "(function_item) @fn",
  lang: "rust",
  glob: ["src/**/*.rs"]
)

# Never do this (no glob = full codebase scan):
search_ast(pattern: "(function_item) @fn", lang: "rust")
```

**Performance:** `list_locations` ≈ 2ms · `search_code` ≈ 3–10ms · `search_ast` ≈ 500ms–10s+

---

## See Also

- [Claude Code + Reflex MCP Quickstart](./ai-agent-integration.md) — MCP setup, key tools, troubleshooting, and CLI/JSON fallback
- [CLI Usage](../CLAUDE.md#cli-usage) — Human-facing `rfx` command reference
- [Dependency Analysis](./DEPENDENCIES.md) — Deep dive into import extraction and graph analysis
