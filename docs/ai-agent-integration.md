# Claude Code + Reflex MCP Quickstart

Get Reflex working as an MCP tool server inside Claude Code in under 5 minutes.

---

## Prerequisites

- [Claude Code](https://claude.ai/code) installed and working
- A project directory you want to search

---

## Step 1 — Install Reflex

**Via npm (recommended):**

```bash
npm install -g reflex-search
```

**Via cargo:**

```bash
cargo install reflex-search
```

Verify the install:

```bash
rfx --version
```

> **Cargo users:** If `rfx` is not found after install, add `~/.cargo/bin` to your `PATH`:
> ```bash
> export PATH="$HOME/.cargo/bin:$PATH"
> ```

---

## Step 2 — Add Reflex to Claude Code

Open your Claude Code MCP settings file. The right scope depends on your use case:

| Scope | File |
|-------|------|
| All projects (global) | `~/.claude/claude_code_config.json` |
| This project only | `.claude/claude_code_config.json` |

Add the `reflex` entry under `mcpServers`:

```json
{
  "mcpServers": {
    "reflex": {
      "command": "rfx",
      "args": ["mcp"],
      "env": {},
      "disabled": false
    }
  }
}
```

> **Note:** `rfx mcp` runs as a stdio MCP server. Claude Code starts and manages it
> automatically — you should not run this command manually in your terminal.

After saving, restart Claude Code (or use **Reload Window** in VS Code) to pick up the
new server.

---

## Step 3 — Index your project

Navigate to your project root and build the search index:

```bash
cd /path/to/your/project
rfx index
```

Indexing is incremental: subsequent runs only process changed files. The first index on a
typical project (10–50k files) takes a few seconds.

> **Tip:** Add `.reflex/` to your `.gitignore` to keep the cache local.
> ```
> echo ".reflex/" >> .gitignore
> ```

---

## Step 4 — Your first search

In a Claude Code conversation, ask a question that requires code search. Claude invokes
Reflex automatically via MCP:

> *"Where is the `parse_config` function defined in this project?"*

Behind the scenes, Claude calls the `search_code` tool:

```json
{
  "tool": "search_code",
  "arguments": {
    "pattern": "parse_config",
    "symbols": true
  }
}
```

The response includes exact file paths, line numbers, and code previews — no shell
scripting or manual `grep` required.

---

## Key Tools for AI Agents

The five tools Claude uses most often in a Reflex-indexed codebase:

### `search_code` — find anything

Full-text or symbol-only search with line numbers and code previews.

```json
// All usages of a function
{ "pattern": "process_request" }

// Definitions only (not usages)
{ "pattern": "process_request", "symbols": true }

// Filter to a specific language
{ "pattern": "process_request", "lang": "rust", "symbols": true }

// Scope to a subdirectory
{ "pattern": "process_request", "glob": ["src/api/**/*.rs"] }
```

### `list_locations` — where, without the noise

Returns `[{path, line}]` only — no code previews. Use this to quickly enumerate all
locations before deciding which files to read.

```json
{ "pattern": "TODO" }
// → [{"path": "src/auth.rs", "line": 42}, {"path": "src/cache.rs", "line": 7}, ...]
```

### `gather_context` — orient to a new project

Returns directory structure, detected frameworks, entry points, and file-type
distribution. Run this at the start of a session when you are unfamiliar with a codebase.

```json
{ "structure": true, "framework": true, "entry_points": true }
```

### `get_dependencies` — what does this file import?

Returns all static imports for a file with their classification
(internal / external / stdlib).

```json
{ "path": "src/auth/middleware.rs" }
```

### `index_project` — keep the index fresh

Trigger a reindex after code changes (git pull, file creation, refactor).

```json
{}

// Force a full rebuild if results seem wrong:
{ "force": true }
```

> **Planned:** `find_references` for cross-file symbol usage tracking is coming in a
> future release.

---

## Tips

### Re-indexing after code changes

Reflex does not watch for changes by default. Re-index manually after significant edits:

```bash
rfx index          # Fast incremental update (changed files only)
rfx index --force  # Full rebuild from scratch
```

Or run the watcher to keep the index live during active development:

```bash
rfx watch
```

### Glob filters in monorepos

Scope searches to a specific package or service to reduce noise:

```json
{
  "pattern": "useAuth",
  "glob": ["packages/web/**/*.ts"]
}
```

### Excluding noisy directories

```json
{
  "pattern": "TODO",
  "exclude": ["**/node_modules/**", "**/vendor/**", "**/target/**"]
}
```

### Index staleness detection

When Claude gets results that look stale, it can call `index_project` to refresh and then
retry — all within the same conversation. You do not need to leave the chat.

---

## Troubleshooting

### "Index not found" error

The `.reflex/` cache has not been built yet for this directory. Run `rfx index` from the
**project root** — the same directory you opened in Claude Code.

```bash
rfx index
```

### Missing new files or stale results

After pulling changes or creating files, the index may lag behind. Re-index:

```bash
rfx index
```

If errors persist after an incremental index, force a full rebuild:

```bash
rfx index --force
```

### MCP server not connecting

1. Confirm `rfx` is on your `PATH`:
   ```bash
   which rfx
   ```
2. Check the settings JSON for syntax errors — trailing commas and unquoted keys break
   JSON parsing.
3. Restart Claude Code after editing the config file.
4. Check Claude Code's MCP server logs for error output from `rfx mcp`.

### MCP server exits unexpectedly

`rfx mcp` runs as a long-lived process over stdio. If it exits unexpectedly, Claude Code
will surface an error in the tool result. Run `rfx mcp` manually in a terminal to see any
startup errors, then report a bug with the output.

---

## Legacy: CLI / JSON Mode

If you are integrating Reflex into a shell script or non-MCP agent, you can use the
`--json` flag directly on the CLI:

```bash
rfx query "pattern" --json
```

### JSON Output Format

The `--json` response includes a `metadata` block for staleness detection:

```json
{
  "metadata": {
    "status": "fresh",
    "current_branch": "main",
    "indexed_branch": "main",
    "current_commit": "74eb454...",
    "indexed_commit": "74eb454..."
  },
  "results": [
    {
      "path": "./src/cache.rs",
      "lang": "rust",
      "kind": "Struct",
      "symbol": "CacheManager",
      "span": { "start_line": 42, "start_col": 0, "end_line": 45, "end_col": 1 },
      "preview": "pub struct CacheManager {"
    }
  ]
}
```

**`metadata.status` values:**

| Value | Meaning |
|-------|---------|
| `fresh` | Index is up to date |
| `branch_not_indexed` | Current git branch has not been indexed |
| `commit_changed` | HEAD has moved since last index |
| `files_modified` | Files modified since last index (detected via sampling) |

When `status !== "fresh"`, the response also includes `reason` (human-readable
explanation) and `action_required` (command to fix, typically `"rfx index"`).

### Bash example

```bash
#!/bin/bash
response=$(rfx query "pattern" --json)
status=$(echo "$response" | jq -r '.metadata.status')

if [ "$status" != "fresh" ]; then
    echo "Index stale, re-indexing..." >&2
    rfx index
    response=$(rfx query "pattern" --json)
fi

echo "$response" | jq '.results'
```

### Python example

```python
import subprocess, json

def query_with_auto_reindex(pattern: str):
    response = json.loads(
        subprocess.check_output(["rfx", "query", pattern, "--json"])
    )
    if response["metadata"]["status"] != "fresh":
        print(f"Index stale: {response['metadata']['reason']}")
        subprocess.run(["rfx", "index"], check=True)
        response = json.loads(
            subprocess.check_output(["rfx", "query", pattern, "--json"])
        )
    return response["results"]
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success (results found, index fresh) |
| `1` | No results found or error |
