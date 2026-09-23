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
| `find_circular` | 🟡 Medium | Cycle arrays | — | opt-in |
| `find_islands` | 🟡 Medium | Isolated component groups | — | opt-in |
| `index_project` | 🔴 Slow (write) | Status + stats | — |
| `search_ast` | 🔴 Slowest | Structural matches | `pattern`, `lang` + glob |

> **Structural analysis tools** (`find_circular`, `find_islands`, `find_unused`, `analyze_summary`,
> `get_transitive_deps`) are shown by default. To hide them and reduce the tool surface for AI agents,
> add to `~/.reflex/config.toml`:
>
> ```toml
> [mcp]
> enable_structural_tools = false  # hides find_circular, find_islands, find_unused, analyze_summary, get_transitive_deps
> ```

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
| `glob` | array | `["src/**/*.rs"]` — gitignore rules: a `/` anchors at the root; `**/src/**/*.rs` for any `src/`; `*.rs` at any depth; `*` never crosses `/` |
| `exclude` | array | `["target/**", "node_modules/**"]` — same rules |
| `file` | string | `"Controllers"` (substring match) |
| `contains` | bool | `true` = substring match (`grep -F`); default matches whole identifiers only. Not on `search_regex` |
| `ignore_case` | bool | `true` = `rg -i` (with `contains`: `rg -i -F`). On `search_code`, `search_regex`, `count_occurrences`, `list_locations`, `find_references`. Uses the trigram index, so it costs about the same as a case-sensitive search |
| `paths` | bool | `true` = `{status, can_trust_results, paths, total_files}`, no rows — the cheapest "which files" answer |
| `symbols` | bool | `true` = definitions only |
| `kind` | string | `"function"`, `"class"`, `"struct"` |
| `expand` | bool | `true` = show full symbol body |
| `limit` / `offset` | int | Pagination (check `has_more`). List mode stops verifying once the page is full: when `total_is_exact` is `false`, `total_count` / `pagination.total` are `null` and `approx_total` is a sampled estimate (typically ±30%) — use `mode: "count"` for the exact number |

---

## Argument Names: Canonical vs Aliased (1.7.0)

Field data from 34 Claude Code sessions: 20 of 21 Reflex failures were `Missing pattern`
because the agent guessed `query`, `symbol`, `max_results` or `path`. Since 1.7.0 the server
maps the habitual wrong names to the real ones and tells you about it.

| Canonical key | Accepted aliases (deprecated) | Applies to |
|---------------|-------------------------------|------------|
| `pattern` | `query`, `symbol`, `text`, `search` | every search tool, `find_references` |
| `limit` | `max_results` | every tool with a `limit` |
| `file` | `path` | search tools only |
| `path` | — (real key, no alias) | `get_dependencies`, `get_dependents`, `get_transitive_deps`, `gather_context` |

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
`index_project` tool rather than the CLI. `IndexNotFound` errors also point at `index_project`.

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

## Efficiency Notes (A/B Tested)

### Columnar result format

`search_code` and `search_regex` return results in `{columns, rows}` format by default.
**Measured savings: 16–24% per-call bytes** vs the legacy `results[]` object shape. The
`~41%` theoretical estimate assumed file-grouped output; the flat columnar format still
repeats `path` and `language` on every row, so savings are smaller in practice.

To revert to the legacy shape: `REFLEX_MCP_COLUMNAR=0`.

### Reflex vs built-in grep/glob (total token cost)

**At parity with built-in grep/glob on total tokens — real wins are capability and ~31% lower
cost (REF-192).** Powered A/B rerun (REF-222: n=9 tasks × 8 trials per arm, claude-sonnet-4-6):
r=1.044, 95% CI [1.014, 1.262] — within the ±10% parity band, 100% task success on both arms,
arm B showing higher recall on large result sets (graded accuracy).

**REF-176 → REF-217 → REF-222 arc (did parity hold?):** All three runs land at the same point
estimate (~1.04). REF-217 (n=3, CI width 1.012) was noisy; REF-222 (n=9, CI width 0.248) is 4×
tighter. The "Indeterminate" label is a method note — the CI upper bound clips 1.262, not a
regression. Parity was never lost.

*Method note:* At equal turn counts, Reflex overhead is only ~**1–2%** (tool schema context per
turn). Turn-count variance drives the spread (corr(total_tokens, turns) ≈ 0.99, REF-204).

Per-task ratios (REF-222, n=9 tasks, sorted):

| Task | B/A ratio | Turns A | Turns B |
|------|-----------|---------|---------|
| reflex-findall-symbolcache | 1.012 | 2 | 2 |
| ripgrep-findall-sinkcontext | 1.014 | 2 | 2 |
| ripgrep-findall-sinkmatch | 1.016 | 2 | 2 |
| reflex-findall-trigramindex | 1.024 | 2 | 2 |
| tokio-findall-joinerror | 1.044 | 2 | 2 |
| tokio-findall-barrier | 1.054 | 2 | 2 |
| tokio-findall-notified | 1.148 | 2 | 2 |
| ripgrep-findall-mmapchoice | 1.262 | 2 | 2 |
| reflex-findall-extract_symbols | 1.450 | 2 | 3 |
| **Median** | **1.044** | | |

**The extract_symbols outlier (1.45) is a turn-count effect**: arm B used 3 turns (more Reflex tool
calls) vs arm A's 2 turns (single Grep). All other tasks ran at equal turns → near parity.

**When to prefer Reflex over built-in grep/glob:**
- Symbol-aware search (`symbols: true`, `kind: "function"`) — unavailable in grep/glob
- Dependency analysis (`get_dependencies`, `get_dependents`, `find_hotspots`)
- Atomic find-all-usages in one call (`find_references`)
- Large result sets where columnar format reduces payload size

**When built-in grep/glob may be cheaper:** Simple literal string lookups with ≤1 tool call,
where the context tax outweighs Reflex's capability advantage.

### structuredContent: evaluated and rejected

MCP's `outputSchema` / `structuredContent` mechanism was:

1. **Built** — implemented in `src/mcp.rs` with `outputSchema` on all tool responses
2. **A/B tested** — ratio **0.998**, no measurable token savings
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
