# Claude Code + Reflex MCP Quickstart

Run Reflex as an MCP server inside Claude Code. This page covers setup, a quick
check that it works, the tools Claude uses most, how to handle a stale index, and
common problems. The last section covers the CLI `--json` mode for scripts.

---

## Prerequisites

- [Claude Code](https://claude.ai/code) installed, with the `claude` command on your `PATH`
- A project directory you want to search

---

## Step 1: Install Reflex

**Via npm:**

```bash
npm install -g reflex-search
```

**Via cargo:**

```bash
cargo install reflex-search
```

Check the install:

```bash
rfx --version
```

> **Cargo users:** if `rfx` is not found, add `~/.cargo/bin` to your `PATH`:
> ```bash
> export PATH="$HOME/.cargo/bin:$PATH"
> ```

---

## Step 2: Register Reflex with Claude Code

Use `claude mcp add`. Pick a scope:

```bash
claude mcp add --scope user reflex -- rfx mcp      # every project on this machine
claude mcp add --scope project reflex -- rfx mcp   # this project only; writes .mcp.json
claude mcp add reflex -- rfx mcp                   # this project, for you only (scope "local", the default)
```

`--scope project` writes a `.mcp.json` file at the project root. Commit it to share the
server with your team. Claude Code asks each person to approve a `.mcp.json` server
before it connects. The file looks like this:

```json
{
  "mcpServers": {
    "reflex": {
      "type": "stdio",
      "command": "rfx",
      "args": ["mcp"]
    }
  }
}
```

For other MCP clients, register a stdio server with command `rfx` and args `["mcp"]`.

Notes:

- `rfx mcp` speaks MCP over stdin/stdout. Claude Code starts and stops it. You do not
  run it yourself.
- The server searches its working directory. Claude Code starts it in the directory you
  launched `claude` from, so launch Claude Code from the project root.
- Start a new Claude Code session after adding the server.

---

## Step 3: Check the connection

List your servers. Claude Code health-checks each one:

```bash
claude mcp list
```

Inside a session, run `/mcp` to see whether `reflex` is connected and which tools it offers.

To test the binary by itself, run this in your project root:

```bash
rfx mcp </dev/null
```

It prints one line to stderr and exits:

```
reflex-mcp startup: version=2.0.3 build=d455c04 columnar=on structural_tools=on
```

- `version` is the Reflex version. `build` is the commit it was built from, or `unknown`.
- `columnar` is `on` unless `REFLEX_MCP_COLUMNAR=0` is set (see [Reading results](#reading-results)).
- `structural_tools` is `on` unless `~/.reflex/config.toml` sets
  `[mcp] enable_structural_tools = false`. When off, `find_circular`, `find_islands`,
  `find_unused`, `analyze_summary` and `get_transitive_deps` are hidden.

Claude Code also records this line in its MCP logs. Check it there to confirm which
binary Claude Code actually started.

---

## Step 4: Index your project

Build the index from the project root:

```bash
cd /path/to/your/project
rfx index
```

The index lives in `.reflex/` at the project root. A later run with no changed files
returns quickly; any change rebuilds the index from every file (the symbol cache keeps
unchanged files' symbols). `rfx index --force` rebuilds even when nothing changed.

You can also skip this step. When a tool reports `Index not found`, Claude calls the
`index_project` tool and retries.

> **Tip:** keep the index out of git:
> ```bash
> echo ".reflex/" >> .gitignore
> ```

What gets indexed follows ripgrep's defaults: every non-binary file that is not
gitignored and not under a dot-directory (`.github/`, `.cargo/`, ...). That includes docs
and config files, not just code. Lock files and generated files (`Cargo.lock`,
`*.min.js`, `*.pb.go`, ...) are indexed but left out of results unless a search asks
for them with `include_locks` / `include_generated`.

---

## Step 5: Your first search

Ask Claude a question that needs code search:

> *"Where is the `Config` struct defined?"*

Claude calls `search_code` with `symbols: true`, which returns definitions only:

```json
{ "pattern": "Config", "symbols": true, "lang": "rust" }
```

The response has file paths, line ranges, symbol kinds and previews. See
[Reading results](#reading-results) for the shape.

---

## Key tools

Reflex offers 17 tools (12 when structural tools are off). These are the ones Claude
uses most:

| Tool | Use it for |
|------|------------|
| `search_code` | Every occurrence of a name, with previews. `symbols: true` for definitions only. |
| `search_regex` | Regular expressions, and patterns with `->`, `::`, `(`, `[`. |
| `find_references` | A symbol's definition plus every usage, in one call. Skips strings and comments by default. Code files only. |
| `list_locations` | Every `{path, line}` for a pattern, without previews. |
| `count_occurrences` | How many matches, in how many files. |
| `get_dependencies` / `get_dependents` | What a file imports / which files import it. |
| `gather_context` | Project structure, frameworks, entry points. Useful at the start of a session. |
| `check_index_status` | Whether the index matches the files on disk. |
| `index_project` | Build or update the index. `force: true` rebuilds from scratch. |

The rest are `search_ast` (slow Tree-sitter structural queries; always pass `glob`),
`get_transitive_deps`, `find_hotspots`, `find_circular`, `find_unused`, `find_islands`
and `analyze_summary`. See [`mcp-tool-cheatsheet.md`](./mcp-tool-cheatsheet.md) for a
decision tree by intent.

### Arguments

- The search term is always `pattern`.
- The result cap is `limit`.
- Filter paths with `file` (a substring) or `glob` (an array). Exclude with `exclude`.
- Filter languages with `lang`, for example `"rust"` or `"text"` (docs and config only).

```json
{ "pattern": "process_request" }
{ "pattern": "process_request", "symbols": true }
{ "pattern": "process_request", "file": "src/api" }
{ "pattern": "useAuth", "glob": ["packages/web/**/*.ts"] }
{ "pattern": "TODO", "exclude": ["vendor/", "target/"] }
{ "pattern": "fn (get|set)_\\w+", "glob": ["src/**/*.rs"] }
```

The last one is a `search_regex` call; the others work with `search_code`.

Globs follow gitignore rules. A pattern with a `/` is anchored at the project root, so
`src/**/*.rs` matches only the top-level `src/`. Use `**/src/**/*.rs` to match any
`src/` directory. A bare name such as `*.rs` or `Makefile` matches at any depth.
`target/` matches every `target/` directory.

### Matching

`search_code` matches **whole identifiers** by default, like `grep -w`. `verify_csrf`
does not match `verify_csrf_form_field`.

| Want | Pass |
|------|------|
| Substring match (`grep -F`) | `contains: true` |
| Regular expression (`grep -E`) | use `search_regex` |
| Case-insensitive (`rg -i`) | `ignore_case: true` |

A pattern with brackets, such as `unwrap()`, is escaped and run as a regex
automatically. The response says so in `warnings`. When a whole-identifier search
finds nothing but substring matches exist, the response carries a `hint` with the
substring count.

---

## Reading results

`search_code` and `search_regex` return a columnar shape. Each row lines up with
`columns`. This is a real response to `{"pattern": "Config", "symbols": true, "lang": "rust"}`:

```json
{
  "ai_instruction": "Found 2 precise results (definitions only, not usages). List locations concisely: '[symbol] at [path]:[line]' for each result.",
  "can_trust_results": true,
  "columns": ["path", "language", "start_line", "end_line", "preview", "kind", "symbol"],
  "has_more": false,
  "pagination": { "count": 2, "has_more": false, "limit": 200, "offset": 0, "total": 2, "total_is_exact": true },
  "returned_count": 2,
  "rows": [
    ["rust/attributes.rs", "rust", 20, 22, "struct Config {\n    enabled: bool,\n}\n\n/// This is a documented function\n#[inline]\npub fn inlined_function() -> i32 {", "Struct", "Config"],
    ["rust/structs.rs", "rust", 29, 33, "pub struct Config {\n    pub host: String,\n    pub port: u16,\n    pub timeout_ms: u64,\n}\n\n// Tuple structs", "Struct", "Config"]
  ],
  "status": "fresh",
  "total_count": 2,
  "total_is_exact": true
}
```

- The first five columns are always there. `kind`, `symbol`, `context_before`,
  `context_after` and `dependencies` are added only when a match has them.
- The default `limit` is 200. If `has_more` is true, call again with `offset`.
- A list search stops once the page is full. When `total_is_exact` is false,
  `total_count` and `pagination.total` are `null` and `approx_total` holds an estimate.
  Use `mode: "count"` for an exact number.
- `paths: true` returns `{status, can_trust_results, paths, total_files}` instead of rows.
- Set `REFLEX_MCP_COLUMNAR=0` in the server's environment to get file-grouped
  `results[]` objects instead:
  `claude mcp add --scope user -e REFLEX_MCP_COLUMNAR=0 reflex -- rfx mcp`.

Other tools use their own shapes. `list_locations` returns
`{locations: [{path, line}], total_locations, status}`. `find_references` returns
`{definition, references, total_references, returned_count, filtered_out, pagination, status}`.
`count_occurrences` returns `{total, files, pattern, status}`.

---

## Freshness

Reflex compares each indexed file's recorded fingerprint with the file on disk. The
index is stale when a file was edited, added or deleted since the last index, whether
or not the change is committed. Committing or switching to a branch with the same
content does not make it stale. Inside a git repository, candidates come from
`git status`; outside one, Reflex checks every file. `details.checked_by` says which
(`git` or `walk`).

### What responses carry

| Response | Freshness fields |
|----------|------------------|
| `search_code`, `search_regex` (list and `paths` modes) | `status`, `can_trust_results`, and `warning` when stale |
| `check_index_status` | `status`, `can_trust_results`, `details`, and the stale fields at the top level |
| `find_references`, `list_locations`, `count_occurrences` | `status` only |
| `mode: "count"` on `search_code` / `search_regex` / `find_references` | none |

`status` is `fresh` or `stale`. `check_index_status` also returns
`{"status": "missing", "action_required": "index_project"}` when there is no index. A stale index always has `can_trust_results: false`, even when a search finds
nothing. A search tool with no index fails with
`Index not found. Call the index_project tool, then retry.`

A stale `check_index_status` response, captured after editing one file:

```json
{
  "action_required": "index_project",
  "can_trust_results": false,
  "changed_count": 1,
  "details": {
    "checked_by": "git",
    "current_branch": "main",
    "current_commit": "6f4beb1b46c2f0112f5675ed5fa4135d70b04283",
    "indexed_at": 1790624516,
    "indexed_branch": "main",
    "indexed_commit": "6f4beb1b46c2f0112f5675ed5fa4135d70b04283"
  },
  "files_modified": ["rust/structs.rs"],
  "reason": "Files changed since the index was built (1 modified) — these results may not reflect them",
  "status": "stale"
}
```

- `files_modified`, `files_added` and `files_deleted` list paths, up to 100 each.
  `truncated` is set when a list was cut short. `changed_count` is the total.
- `action_required` names the MCP tool to call: `index_project`.
- In `search_code` / `search_regex` responses the same fields sit inside `warning`.

### What to do

Stale results are still real matches. They may miss new code, and a deleted file can
still produce hits at its old lines. When completeness matters (find all callers,
rename planning, impact analysis), call `index_project` and search again. The tool
descriptions tell Claude to do this.

To keep the index current while you work, run the watcher in a terminal:

```bash
rfx watch
```

`rfx mcp` caches the freshness verdict for 1 second per workspace. A file saved in the
last second may not show up as a change yet. `check_index_status` always checks again.

---

## Troubleshooting

### "Index not found"

There is no `.reflex/` in the directory the server runs in. Run `rfx index` in the
project root, or let Claude call `index_project`. Make sure you launched Claude Code
from the same directory you indexed.

### Stale results or missing new files

Run `rfx index` (or ask Claude to call `index_project`). If results still look wrong,
rebuild with `rfx index --force` (or `index_project` with `force: true`).

### A search returns 0 but the text is there

- The default is whole-identifier matching. Check the `hint`, then retry with
  `contains: true`.
- The file may be a lock or generated file. Check `excluded_reason` and
  `excluded_by_default`, then pass `include_locks: true` or `include_generated: true`.
- Files under dot-directories (`.github/`) and gitignored files are not indexed. Set
  `[index] hidden = true` in `.reflex/config.toml` to index dot-directories.
- `find_references` never searches docs or config files. Use `search_code` with
  `lang: "text"` for those.

### The server does not connect

1. Check that `rfx` is on the `PATH` Claude Code sees: `which rfx`.
2. Run `claude mcp list` and `claude mcp get reflex`. A `.mcp.json` server shows as
   pending until you approve it.
3. Run `rfx mcp </dev/null` in the project root. It should print the
   `reflex-mcp startup:` line. If it prints an error instead, that is the cause.
4. Start a new Claude Code session after changing the server config.

### Claude uses the wrong binary

Compare the `version` and `build` in the `reflex-mcp startup:` line in Claude Code's MCP
logs with `rfx --version`. If they differ, the server config points at another `rfx`.
Use an absolute path: `claude mcp add --scope user reflex -- /full/path/to/rfx mcp`.

### Claude uses Grep instead of Reflex

The server sends instructions telling Claude to prefer Reflex for code search. Claude
Code may load MCP tool schemas lazily, so Claude sometimes reaches for built-in tools
first. Ask for Reflex by name, or add a line to your project's `CLAUDE.md`, such as
"Use the Reflex MCP tools (`search_code`, `find_references`) for code search."

---

## Scripts: CLI JSON mode

For shell scripts and non-MCP agents, use `rfx query --json`:

```bash
rfx query "Config" --json
rfx query "Config" --symbols --lang rust --json
rfx query "TODO" --json --limit 20
rfx query --pattern '-> Result<' --json    # a pattern that starts with "-"
```

The CLI uses the same matching rules as MCP: `--contains`, `-i` / `--ignore-case`,
`--regex`, `--glob`, `--exclude`, `--include-locks`, `--include-generated`.

### Output shape

CLI results are grouped by file. This is a real response to
`rfx query Config --symbols --lang rust --json`, captured after editing
`rust/structs.rs` without reindexing:

```json
{
  "status": "stale",
  "can_trust_results": false,
  "warning": {
    "reason": "Files changed since the index was built (1 modified) — including a file these results came from",
    "action_required": "index_project",
    "files_modified": ["rust/structs.rs"],
    "changed_count": 1,
    "details": {
      "current_branch": "main",
      "indexed_branch": "main",
      "current_commit": "6f4beb1b46c2f0112f5675ed5fa4135d70b04283",
      "indexed_commit": "6f4beb1b46c2f0112f5675ed5fa4135d70b04283",
      "indexed_at": 1790624591,
      "checked_by": "git"
    }
  },
  "pagination": { "total": 2, "count": 2, "offset": 0, "limit": 100, "has_more": false, "total_is_exact": true },
  "results": [
    {
      "path": "rust/attributes.rs",
      "language": "rust",
      "matches": [
        {
          "kind": "Struct",
          "symbol": "Config",
          "span": { "start_line": 20, "end_line": 22 },
          "preview": "struct Config {\n    enabled: bool,\n}\n\n/// This is a documented function\n#[inline]\npub fn…"
        }
      ]
    },
    {
      "path": "rust/structs.rs",
      "language": "rust",
      "matches": [
        {
          "kind": "Struct",
          "symbol": "Config",
          "span": { "start_line": 29, "end_line": 33 },
          "preview": "pub struct Config {\n    pub host: String,\n    pub port: u16,\n    pub timeout_ms: u64,\n}\n\n// Tuple…"
        }
      ]
    }
  ]
}
```

- `status` and `can_trust_results` are top-level fields. `warning` is present only when
  the index is stale.
- `action_required` is `index_project` here too. From a script, run `rfx index`.
- Full-text matches have no `kind` or `symbol`; only `span` and `preview`.
- The default limit is 100. Use `--limit` and `--offset` to page.
- Other top-level fields appear only when they apply: `warnings`, `hint`,
  `excluded_reason`, `excluded_by_default`, `substring_hint_count`, `timings`.
- With no index, stdout is `{"error": "Index not found. Run 'rfx index' to build the search index.", "query_too_broad": false}`
  and the exit code is 1.
- `--count --json` returns only `{"count": N, "timing_ms": N}`, with no freshness fields.
- A one-line summary such as `Found 2 results in 17ms` goes to stderr. Stdout holds
  only the JSON.

### Bash example

Reindex when the index is stale or missing, then print `path:line: preview`:

```bash
#!/usr/bin/env bash
pattern="$1"
response=$(rfx query "$pattern" --json)
if [ "$(jq -r '.status' <<<"$response")" != "fresh" ]; then
  echo "reindexing: $(jq -r '.warning.reason // .error' <<<"$response")" >&2
  rfx index --quiet
  response=$(rfx query "$pattern" --json)
fi
jq -r '.results[] | .path as $p | .matches[] | "\($p):\(.span.start_line): \(.preview)"' <<<"$response"
```

### Python example

```python
import json
import subprocess

def rfx_query(pattern, *flags):
    proc = subprocess.run(["rfx", "query", pattern, "--json", *flags],
                          capture_output=True, text=True)
    return json.loads(proc.stdout)

def search(pattern, *flags):
    response = rfx_query(pattern, *flags)
    if response.get("status") != "fresh":  # stale, or {"error": ...} with no index
        reason = response.get("warning", {}).get("reason") or response.get("error")
        print(f"reindexing: {reason}")
        subprocess.run(["rfx", "index", "--quiet"], check=True)
        response = rfx_query(pattern, *flags)
    return [
        (group["path"], match["span"]["start_line"], match["preview"])
        for group in response["results"]
        for match in group["matches"]
    ]

for hit in search("Config", "--lang", "rust"):
    print(hit)
```

The example does not use `check=True` on the query, because a missing index exits 1
but still prints JSON on stdout.

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | The query ran. This includes zero results and a stale index. |
| `1` | Error, such as no index. |
