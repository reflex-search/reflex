# Reflex MCP Tool Selection Cheatsheet

> **Quick rule:** Start cheap, escalate only when needed.
> `list_locations` → `search_code` → `search_regex` → `search_ast` (last resort)

---

## Decision Tree by Agent Intent

### "I want to find WHERE something is"

| Goal | Tool | Why |
|------|------|-----|
| Known exact name, just need locations | `list_locations` | Cheapest — `{locations: [{path, line}], total_locations}`, no content |
| Need locations **and** code previews | `search_code` | Full results with line numbers + context |
| Regex: alternation, wildcards, anchors, `->`, `::` | `search_regex` | Real regular expressions |
| How many times does X appear? | `search_code` with `mode: "count"` | `{count, files, pattern}` — no content loaded |

```
"Where is UserController used?"
  → list_locations(pattern: "UserController")

"How many places call unwrap()?"
  → search_code(pattern: "unwrap()", mode: "count")
    # brackets are regex-escaped automatically; the rewrite is reported in `warnings`
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
| What files import this file? | `get_dependencies` with `reverse: true` | Reverse lookup — impact of changes |
| Full import tree (deps of deps) | `get_dependencies` with `depth: N` | Follows imports N levels; returns `[{path, depth}]` |

```
"What does src/query/mod.rs depend on?"
  → get_dependencies(path: "src/query/mod.rs")

"What breaks if I change models/User.php?"
  → get_dependencies(path: "User.php", reverse: true)

"Show the full dependency chain for main.rs"
  → get_dependencies(path: "src/main.rs", depth: 3)
```

`reverse` and `depth` cannot be combined.

> **Note:** Dependency analysis extracts **static imports only**. Dynamic imports (variables, template literals) are filtered by design.

---

### "I want to understand the CODEBASE"

| Goal | Tool | Why |
|------|------|-----|
| Project type, entry points, frameworks | `gather_context` (no params) | One-shot codebase overview |
| Dependency health at a glance | `analyze(kind: "summary")` | Returns counts: circular, hotspots, unused, islands |
| Most-imported (critical) files | `analyze(kind: "hotspots")` | Files ranked by import count — the load-bearing modules |
| Unused / orphaned files | `analyze(kind: "unused")` | Candidates for deletion (verify entry points aren't included) |
| Circular dependency cycles | `analyze(kind: "circular")` | Returns cycle arrays: A→B→C→A |
| Isolated subsystems | `analyze(kind: "islands")` | Groups of files with no cross-group imports |

```
"What kind of project is this?"
  → gather_context()

"Is the dependency graph healthy?"
  → analyze(kind: "summary")
  → then drill into analyze(kind: "circular" | "hotspots" | "unused") as needed

"What files are most central to this codebase?"
  → analyze(kind: "hotspots", min_dependents: 3)
```

`analyze` also takes `limit`, `offset`, `sort`, `min_dependents`, `min_island_size` and
`max_island_size`.

---

### "I need to maintain the index"

Nothing, normally. Every tool updates the index before it answers (only the changed files),
and builds it when there is none; after an edit, a checkout or a rebase, just search.

| Goal | Tool | Why |
|------|------|-----|
| A response has `can_trust_results: false` | read its `warnings` | The automatic update could not run (read-only `.reflex/`, a cache another rfx version owns) |
| The index appears corrupted | `index_project` with `force: true` | Full rebuild |

---

## Tool Quick Reference

| Tool | Cost | Returns | Requires |
|------|------|---------|---------|
| `list_locations` | ⚡ Cheapest | `[{path, line}]` | `pattern` |
| `search_code` with `mode: "count"` | ⚡ Cheap | `{count, files, pattern}` | `pattern` |
| `search_code` | 🟡 Medium | Full results with previews | `pattern` |
| `search_regex` | 🟡 Medium | Full results with previews | `pattern` |
| `find_references` | 🟡 Medium | Definition + every usage | `pattern` |
| `gather_context` | 🟡 Medium | Project structure summary | — |
| `get_dependencies` | 🟡 Medium | Imports of one file; with `reverse: true`, files importing it; with `depth: N`, `[{path, depth}]` | `path` |
| `analyze` | 🟡 Medium | `summary` counts, `hotspots`, `circular` cycles, `unused` files or `islands` | `kind` |
| `index_project` | 🔴 Slow (write) | Status + stats | — |
| `search_ast` | 🔴 Slowest | Structural matches | `pattern`, `lang` + glob |

> **Structural analysis** (`analyze`) is shown by default. To hide it and reduce the tool
> surface for AI agents, add to `~/.reflex/config.toml`:
>
> ```toml
> [mcp]
> enable_structural_tools = false  # hides analyze
> ```

### Removed tool names

These names are no longer listed but still answer. Each call carries a `warnings` entry
naming the replacement, except `get_dependents` and `get_transitive_deps`, which answer
bare arrays.

| Old name | Use instead |
|----------|-------------|
| `count_occurrences` | `search_code(mode: "count")` (the old name still returns `{total, files, pattern}`) |
| `get_dependents` | `get_dependencies(reverse: true)` |
| `get_transitive_deps` | `get_dependencies(depth: N)` |
| `analyze_summary` | `analyze(kind: "summary")` |
| `find_hotspots` | `analyze(kind: "hotspots")` |
| `find_circular` | `analyze(kind: "circular")` |
| `find_unused` | `analyze(kind: "unused")` |
| `find_islands` | `analyze(kind: "islands")` |

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

get_dependencies(path: "src/auth.rs", reverse: true)
# → 8 files use auth.rs — these are affected if you change it
```

---

## Common Filters (available on most search tools)

| Filter | Type | Example |
|--------|------|---------|
| `lang` | string | `"rust"`, `"typescript"`, `"python"` |
| `glob` | array | `["src/**/*.rs"]` — gitignore rules: a `/` anchors at the root; `**/src/**/*.rs` for any `src/`; `*.rs` at any depth; `*` never crosses `/` |
| `exclude` | array | `["target/**", "node_modules/**"]` — same rules |
| `file` | string | `"Controllers"` (substring match) |
| `contains` | bool | `true` = substring match (`grep -F`); default matches whole identifiers only. Not on `search_regex` |
| `ignore_case` | bool | `true` = `rg -i` (with `contains`: `rg -i -F`). On `search_code`, `search_regex`, `list_locations`, `find_references`. Uses the trigram index, so it costs about the same as a case-sensitive search |
| `paths` | bool | `true` = `{status, can_trust_results, paths, total_files}`, no rows — the cheapest "which files" answer |
| `symbols` | bool | `true` = definitions only |
| `kind` | string | `"function"`, `"class"`, `"struct"` |
| `expand` | bool | `true` = show full symbol body |
| `limit` / `offset` | int | Pagination (check `has_more`). List mode stops verifying once the page is full: when `total_is_exact` is `false`, `total_count` / `pagination.total` are `null` and `approx_total` is a sampled estimate (typically ±30%) — use `mode: "count"` for the exact number |

---

## Argument Names: Canonical vs Aliased

Agents often guess `query`, `symbol`, `max_results` or `path`. The server maps these
habitual wrong names to the real ones and tells you about it.

| Canonical key | Accepted aliases (deprecated) | Applies to |
|---------------|-------------------------------|------------|
| `pattern` | `query`, `symbol`, `text`, `search` | every search tool, `find_references` |
| `limit` | `max_results` | every tool with a `limit` |
| `file` | `path` | search tools only |
| `path` | — (real key, no alias) | `get_dependencies`, `gather_context` |

Behaviour:

- An alias is rewritten and the response carries a top-level `warnings` array, e.g.
  `["argument \"query\" is deprecated; use \"pattern\""]`. Prose tools append the warnings as
  trailing text.
- A key that matches nothing is rejected with JSON-RPC `-32602`:
  `Unknown argument "max_resultz" for search_code (did you mean "limit"?). Received: [...]. Valid: [...]`.
- A missing required key is `-32602`:
  `Missing required argument "pattern" for search_code. Received: [...]. Valid: [...]`.
- Numeric strings are coerced (`"40"` → `40`); `"true"`/`"false"` strings become booleans; a bare
  string for `glob`/`exclude` becomes a one-element array. Anything else of the wrong type is a
  typed `-32602` error, never a silent fallback to the default.
- `search_code` is a literal text index. A natural-language query such as `"hot tier promotion"`
  matches nothing; search for identifiers or code fragments.

**Corrupted or missing index:** on `CacheCorrupted` the server rebuilds once (force) and retries
the call automatically. If that cannot work (lock held, read-only cache), the error names the
`index_project` tool rather than the CLI. A missing index is built by the first tool call.

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

**Cost order:** `list_locations` < `search_code` < `search_regex` ≪ `search_ast` (parses every selected file).

---

## Efficiency Notes

### Columnar result format

`search_code` and `search_regex` return results in `{columns, rows}` format by default,
which avoids repeating JSON keys per match. The flat rows still repeat `path` and
`language` on every row. To revert to the legacy `results[]` shape: `REFLEX_MCP_COLUMNAR=0`.

### Reflex vs built-in grep/glob (measured)

Measured on Reflex 2.0.3, 2026-09-28 (full report: `.context/EFFICACY-2.0.3.md`). Reflex
costs more tokens than built-in Grep/Glob: 1.66× on 9 find-all-usages tasks (Opus 5.5 and
Sonnet 5 alike) and 1.26× on 13 comprehension tasks (Opus 5.5). The cost is round-trips:
the first Reflex call needs a ToolSearch turn to load the deferred schemas, and a
`check_index_status` call before searching adds another. Since auto-update (unreleased)
every tool updates the index before it answers and the descriptions no longer ask for the
status check; the A/B has not been re-run yet.

**When to prefer Reflex over built-in grep/glob** (capabilities, not measured savings):
- Symbol-aware search (`symbols: true`, `kind: "function"`) — unavailable in grep/glob
- Dependency analysis (`get_dependencies`, with `reverse` / `depth`, and `analyze`)
- Definition plus every usage in one call (`find_references`)
- Exact counts without loading content (`mode: "count"`)

### structuredContent: evaluated and rejected

MCP's `outputSchema` / `structuredContent` mechanism was:

1. **Built** — implemented in `src/mcp.rs` with `outputSchema` on all tool responses
2. **A/B tested** — no measurable token savings
3. **Removed** — `content[text]`-only output (current default)

**Root cause:** Claude Code's MCP client transmits *both* `content[text]` *and*
`structuredContent` to the model. Neither field is dropped, so total token cost is
identical regardless of whether the server populates `structuredContent`.

`structuredContent` remains viable only for a client that honors `outputSchema` and drops
the `content[text]` block. Claude Code does not. Do not re-implement without first
confirming client behavior.

---

## See Also

- [Claude Code + Reflex MCP Quickstart](./ai-agent-integration.md) — MCP setup, key tools, troubleshooting, and CLI/JSON fallback
- [CLI Usage](../CLAUDE.md#cli-usage) — Human-facing `rfx` command reference
- [Dependency Analysis](./DEPENDENCIES.md) — Deep dive into import extraction and graph analysis
