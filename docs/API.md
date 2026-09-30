# Reflex HTTP API Reference

`rfx serve` runs a small HTTP server that exposes the Reflex index as JSON. Use it
from editors, scripts and browser tools that cannot run the CLI or speak MCP.

## Overview

- **Base URL:** `http://127.0.0.1:7878`. Change it with `--host` and `--port`.
- **Workspace:** the server serves the index of the directory it was started in.
  Start it from the project root.
- **Format:** JSON responses. Only `POST /index` takes a request body.
- **Authentication:** none. The server binds to loopback by default. See
  [Security](#security).
- **CORS:** any origin, any method, any header.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/query` | Search the index |
| `GET` | `/stats` | Index statistics |
| `POST` | `/index` | Build or update the index |
| `GET` | `/health` | Liveness check |

Any other path returns `404` with a JSON error body. A known path with the wrong
method returns `405` with an empty body and an `Allow` header.

---

## Getting started

```bash
cd /path/to/project
rfx index                         # build the index (or call POST /index later)
rfx serve                         # 127.0.0.1:7878
rfx serve --port 8080             # another port
```

On startup the server prints its address and endpoints:

```
Starting Reflex HTTP server...
  Address: http://127.0.0.1:7878

Endpoints:
  GET  /query?q=<pattern>&lang=<lang>&kind=<kind>&limit=<n>&symbols=true&regex=true&exact=true&contains=true&ignore_case=true&expand=true&file=<pattern>&timeout=<secs>&glob=<pattern>&exclude=<pattern>&paths=true&dependencies=true
  GET  /stats
  GET  /health
  POST /index

Press Ctrl+C to stop.
```

Quick test:

```bash
curl -s http://127.0.0.1:7878/health
# {"service":"reflex","status":"ok"}

curl -s 'http://127.0.0.1:7878/query?q=QueryEngine&limit=5' | jq .
```

---

## GET /query

Search the index. Returns matches grouped by file, plus index freshness and
pagination.

### Query parameters

Booleans must be the literal strings `true` or `false`. Any other value is a
`400` (see [Errors](#errors)). Unknown parameters are ignored without a warning.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `q` | string | required | The pattern. Must not be empty. |
| `contains` | bool | `false` | Substring match (`grep -F`) instead of whole-identifier match. |
| `ignore_case` | bool | `false` | Case-insensitive match (`rg -i`). Works with the default mode, `contains` and `regex`. |
| `regex` | bool | `false` | Treat `q` as a regular expression (`grep -E`). |
| `symbols` | bool | `false` | Return symbol definitions only (functions, structs, classes, ...), not every occurrence. |
| `kind` | string | none | Keep only symbols of this kind, e.g. `function`, `struct`. Case-insensitive. Turns on `symbols`. |
| `exact` | bool | `false` | Symbol searches only: keep symbols whose name equals `q` exactly. No effect on full-text searches. |
| `expand` | bool | `false` | For symbol matches, set `preview` to the symbol's full body (`span.start_line` to `span.end_line`). |
| `lang` | string | none | Only files of this language. See [Languages](#languages). |
| `file` | string | none | Only files whose path contains this substring. |
| `limit` | integer | `100` | Maximum matches to return. With `paths=true` and no `limit`, there is no cap. |
| `offset` | integer | `0` | Skip this many matches before the page starts. |
| `paths` | bool | `false` | One entry per matching file (its first match). `pagination.total` is then a file count. |
| `dependencies` | bool | `false` | Add each file's static imports as `dependencies`. |
| `force` | bool | `false` | Run a query the broad-query guard would refuse (for example a pattern under 3 characters). |
| `timeout` | integer | `30` | Seconds before the query is abandoned. `0` means no timeout. |
| `glob` | string | none | Declared, but not usable: any value is rejected with `400`. See [Known limitations](#known-limitations). |
| `exclude` | string | none | Declared, but not usable: any value is rejected with `400`. See [Known limitations](#known-limitations). |

The HTTP API has no parameter for lock files, generated files, excluding the
text tier, context lines or count mode. See [Known limitations](#known-limitations).

### Matching

A literal pattern matches **whole identifiers** by default, like `grep -w`.
`verify_csrf` does not match `verify_csrf_form_field`.

| Mode | How | Behaves like |
| --- | --- | --- |
| whole identifier | default | `grep -w` |
| substring | `contains=true` | `grep -F` |
| regular expression | `regex=true` | `grep -E` |
| case-insensitive | add `ignore_case=true` to any of the above | `rg -i` |

- A whole-identifier `ignore_case` search keeps whole-identifier rules: `realmid`
  finds `RealmId`, not `realm_id`.
- A literal pattern that contains brackets (`()`, `[]`, `<>`) can never match as a
  whole identifier. The engine escapes it, runs it as a substring regex, and says
  so in `warnings`.
- A whole-identifier search that finds nothing, while substring matches exist,
  returns a `hint` with the substring count.
- A symbol search matches the symbol name. `symbols=true&q=Poi` does not find
  `Point`.

### Which files are searched

- Code, and the plain-text tier (docs, config, templates and every other
  non-binary file, `language: "text"`), are searched by default.
- Lock files (`language: "lock"`) and generated files (`language: "generated"`)
  are indexed but left out. Over HTTP the only way to search them is
  `lang=lock` or `lang=generated`, which selects that tier alone.
- A symbol search (`symbols`, `kind`) only returns files with a parser. Text, lock
  and generated files have none.

### Response

`200 OK`, `application/json`. A real response (trimmed to one file):

```bash
curl -s 'http://127.0.0.1:7878/query?q=realm_marker&limit=3'
```

```json
{
  "status": "fresh",
  "can_trust_results": true,
  "pagination": {
    "total": 12,
    "count": 3,
    "offset": 0,
    "limit": 3,
    "has_more": true,
    "total_is_exact": true
  },
  "results": [
    {
      "path": "rust/realm_marker.rs",
      "language": "rust",
      "matches": [
        { "span": { "start_line": 1, "end_line": 1 },
          "preview": "//! Shared-token fixture: `realm_marker` appears here AND in tests/corpus/text/." },
        { "span": { "start_line": 5, "end_line": 5 },
          "preview": "    pub realm_marker: String," },
        { "span": { "start_line": 10, "end_line": 10 },
          "preview": "    &cfg.realm_marker" }
      ]
    }
  ]
}
```

A symbol match carries `kind` and `symbol`:

```bash
curl -s 'http://127.0.0.1:7878/query?q=Point&kind=struct&lang=rust&expand=true&limit=1'
```

```json
{
  "status": "fresh",
  "can_trust_results": true,
  "pagination": { "total": 3, "count": 1, "offset": 0, "limit": 1, "has_more": true, "total_is_exact": true },
  "results": [
    {
      "path": "rust/attributes.rs",
      "language": "rust",
      "matches": [
        { "kind": "Struct", "symbol": "Point",
          "span": { "start_line": 14, "end_line": 17 },
          "preview": "pub struct Point {\n    x: f64,\n    y: f64,\n}" }
      ]
    }
  ]
}
```

### Response fields

Optional fields are left out when they do not apply. They are never `null`,
except `pagination.total` (see below).

| Field | Type | Present | Meaning |
| --- | --- | --- | --- |
| `status` | `"fresh"` \| `"stale"` | always | Whether the index matches the files on disk. |
| `can_trust_results` | bool | always | `false` whenever `status` is `"stale"`, including for a zero-result search. |
| `warning` | object | when stale | What changed. See [Freshness](#freshness). |
| `pagination` | object | always | See [Pagination and totals](#pagination-and-totals). |
| `results` | array | always | One object per file, sorted by path. |
| `warnings` | string[] | when the engine changed the query | For example the bracket rewrite, or a regex with no 3-character literal that fell back to scanning every line. |
| `hint` | string | zero results with a known cause | A sentence that explains the zero. |
| `excluded_reason` | string | with `hint` | `"whole_identifier"`, `"lock_or_generated"`, `"not_indexed"` or `"hidden"`. |
| `substring_hint_count` | integer | zero-result whole-identifier searches | How many candidate lines contain `q` as a substring. |
| `excluded_by_default` | integer | zero results where lock or generated files matched | How many candidate files were left out for that reason. |

Each entry in `results`:

| Field | Type | Meaning |
| --- | --- | --- |
| `path` | string | Path relative to the workspace root. |
| `language` | string | Lowercase language name. See [Languages](#languages). |
| `dependencies` | array | Only with `dependencies=true` and when the file has imports. Each item is `{path, line?, symbols?}`. |
| `matches` | array | The matches in this file, in line order. |

Each entry in `matches`:

| Field | Type | Meaning |
| --- | --- | --- |
| `span` | `{start_line, end_line}` | 1-based line range. One line for a text match; the definition's range for a symbol. |
| `preview` | string | The matching line. For a symbol, the lines from its start (the full body with `expand=true`). |
| `kind` | string | Symbol matches only: `Function`, `Class`, `Struct`, `Enum`, `Interface`, `Trait`, `Constant`, `Variable`, `Method`, `Module`, `Namespace`, `Type`, `Macro`, `Property`, `Event`, `Import`, `Export` or `Attribute`. |
| `symbol` | string | Symbol matches only: the symbol name. |

With `dependencies=true`:

```json
{
  "path": "python/classes.py",
  "language": "python",
  "dependencies": [
    { "path": "abc", "line": 25, "symbols": ["abc", "ABC", "abstractmethod"] },
    { "path": "dataclasses", "line": 26, "symbols": ["dataclasses", "dataclass"] },
    { "path": "typing", "line": 27, "symbols": ["typing", "Optional", "List"] }
  ],
  "matches": [ ... ]
}
```

### Pagination and totals

`limit` and `offset` count matches, not files. `pagination` has:

| Field | Meaning |
| --- | --- |
| `total` | Every match before `offset`/`limit`, or **`null`** when `total_is_exact` is `false`. |
| `count` | Matches in this response. |
| `offset` | The offset used. |
| `limit` | The limit used. Absent when there was none (`paths=true` without `limit`). |
| `has_more` | More matches exist after this page. |
| `total_is_exact` | `true` when `total` counts every match. |
| `approx_total` | An estimate, present only when `total_is_exact` is `false` and an estimate was possible. |

A full-text search with a `limit` stops verifying candidates once the page is
full. The page is the same as the same slice of a full run, but the total may
not be known. Then `total` is `null`, `total_is_exact` is `false`,
`has_more` is `true`, and `approx_total` is a sampled estimate. If only a few
candidate files or lines remain when the page fills, the search finishes and the
total is exact. Symbol searches always report an exact total.

Real response, 300 files with 1,800 matches, `limit=3`:

```json
{ "pagination": { "total": null, "count": 3, "offset": 0, "limit": 3,
                  "has_more": true, "total_is_exact": false, "approx_total": 1800 } }
```

A regex with no literal of 3 or more characters verifies every line and gives no
estimate:

```json
{ "pagination": { "total": null, "count": 1, "offset": 0, "limit": 1,
                  "has_more": true, "total_is_exact": false },
  "warnings": ["Regex pattern 'fn' has no literals (≥3 chars), falling back to full content scan. This may be slow on large codebases. Consider using patterns with literal text."] }
```

To get an exact count over HTTP, pass a `limit` larger than the number of
matches. `paths=true` without a `limit` always gives an exact file count.
`limit=0` returns no matches but still returns `approx_total`.

### Freshness

Every response carries `status` and `can_trust_results`. The index is stale when
a file on disk differs from what was indexed: edited, added or deleted, committed
or not. Inside a git repository the check starts from `git status`; outside one it
stats every file (`details.checked_by` is `"git"` or `"walk"`).

A real stale response after editing one file and adding another:

```json
{
  "status": "stale",
  "can_trust_results": false,
  "warning": {
    "reason": "Files changed since the index was built (1 modified, 1 added) — these results may not reflect them",
    "action_required": "index_project",
    "files_modified": ["rust/structs.rs"],
    "files_added": ["rust/zz_probe.rs"],
    "changed_count": 2,
    "details": {
      "current_branch": "main",
      "indexed_branch": "main",
      "current_commit": "ea56e89416bbb2ae0114a6d8b28c9b13d9fd4f60",
      "indexed_commit": "ea56e89416bbb2ae0114a6d8b28c9b13d9fd4f60",
      "indexed_at": 1790624209,
      "checked_by": "git"
    }
  },
  "pagination": { ... },
  "results": [ ... ]
}
```

- `files_modified`, `files_added` and `files_deleted` are path lists, each left
  out when empty and capped at 100. `truncated: true` appears when a list was cut.
- `action_required` names the MCP tool (`index_project`). Over HTTP, call
  `POST /index`.
- The verdict is cached for 1 second per workspace (`REFLEX_FRESHNESS_TTL_MS`;
  `0` turns the cache off).
- There is no `"missing"` status. A query with no index returns `404`
  `IndexNotFound`.

### Zero results

A zero result explains itself when it can. Whole-identifier miss:

```bash
curl -s 'http://127.0.0.1:7878/query?q=Poin'
```

```json
{
  "status": "fresh",
  "can_trust_results": true,
  "pagination": { "total": 0, "count": 0, "offset": 0, "limit": 100, "has_more": false, "total_is_exact": true },
  "results": [],
  "substring_hint_count": 30,
  "hint": "0 whole-identifier matches; 30 substring matches — pass contains:true (--contains on the CLI) to see them. Reflex matches whole identifiers by default, so \"Poin\" does not match longer names that merely contain it.",
  "excluded_reason": "whole_identifier"
}
```

Only lock files matched:

```json
{
  "status": "fresh",
  "can_trust_results": true,
  "pagination": { "total": 0, "count": 0, "offset": 0, "limit": 100, "has_more": false, "total_is_exact": true },
  "results": [],
  "substring_hint_count": 0,
  "hint": "1 candidate file(s) were lock or generated files, which every search leaves out by default — pass include_locks:true / include_generated:true (--include-locks / --include-generated on the CLI), or lang:\"lock\" / lang:\"generated\", to search them.",
  "excluded_reason": "lock_or_generated",
  "excluded_by_default": 1
}
```

Over HTTP, follow this hint with `lang=lock` or `lang=generated`.
`include_locks` and `include_generated` are not HTTP parameters.

### Examples

```bash
# Whole-identifier search (default)
curl -s 'http://127.0.0.1:7878/query?q=extract_symbols&limit=10'

# Substring search
curl -s 'http://127.0.0.1:7878/query?q=extract_sym&contains=true'

# Case-insensitive
curl -s 'http://127.0.0.1:7878/query?q=realmid&ignore_case=true'

# Regex (URL-encode the pattern: `fn new\(`)
curl -s 'http://127.0.0.1:7878/query?q=fn%20new%5C%28&regex=true'

# Symbol definitions of one kind, full body
curl -s 'http://127.0.0.1:7878/query?q=Point&kind=struct&expand=true'

# Language and path filters
curl -s 'http://127.0.0.1:7878/query?q=unwrap&lang=rust&file=src/&limit=20'

# Files that mention a name, no cap
curl -s 'http://127.0.0.1:7878/query?q=QueryEngine&paths=true'

# Second page
curl -s 'http://127.0.0.1:7878/query?q=Point&limit=50&offset=50'

# Lock files only
curl -s 'http://127.0.0.1:7878/query?q=serde&lang=lock'

# Imports of each matching file
curl -s 'http://127.0.0.1:7878/query?q=QueryEngine&dependencies=true&limit=5'
```

`curl -G --data-urlencode` saves encoding a pattern by hand:

```bash
curl -s -G http://127.0.0.1:7878/query --data-urlencode 'q=-> Result<' --data-urlencode 'contains=true'
```

---

## GET /stats

Index statistics. No parameters.

```bash
curl -s http://127.0.0.1:7878/stats
```

```json
{
  "total_files": 93,
  "index_size_bytes": 740808,
  "last_updated": "2026-09-28T19:36:49+00:00",
  "files_by_language": { "Rust": 38, "Text": 18, "TypeScript": 11, "PHP": 6, "Lock": 1 },
  "lines_by_language": { "Rust": 2169, "Text": 230, "TypeScript": 834, "PHP": 448, "Lock": 1 },
  "corpus_bytes": 124222,
  "trigram_index_bytes": 368971
}
```

| Field | Meaning |
| --- | --- |
| `total_files` | Files in the index. |
| `index_size_bytes` | Combined size of `meta.db`, `config.toml`, `content.bin` and `trigrams.bin` in `.reflex/`. |
| `last_updated` | When the index was last written (RFC 3339). |
| `files_by_language` | File count per language. Keys are **capitalized** (`"Rust"`, `"Text"`, `"Lock"`), unlike the lowercase `language` field in query results. |
| `lines_by_language` | Line count per language, same keys. |
| `corpus_bytes` | Bytes of indexed source. Left out when 0. |
| `trigram_index_bytes` | Size of `trigrams.bin`. Left out when 0. |

Status codes: `200`; `404` `IndexNotFound` when there is no index; `500` on a
read failure.

---

## POST /index

Build or update the index of the server's workspace. The request blocks until
indexing finishes and returns the new statistics.

The body is optional. With no body, no `Content-Type` header, or an empty body,
the defaults apply. A body that is not valid JSON returns `400` `ParseError`.
Unknown keys are ignored.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `force` | bool | `false` | Delete the existing index and rebuild from scratch. Otherwise only changed files are reindexed. |
| `languages` | string[] | `[]` (all) | Code languages to index. Text and lock files are indexed either way. Only `rust`/`rs`, `python`/`py`, `javascript`/`js`, `typescript`/`ts`, `vue`, `svelte`, `go`, `java`, `php`, `c` and `cpp`/`c++` are recognized; other names are dropped silently. |

```bash
# Incremental update
curl -s -X POST http://127.0.0.1:7878/index

# Full rebuild of Rust code only
curl -s -X POST http://127.0.0.1:7878/index \
  -H 'Content-Type: application/json' \
  -d '{"force": true, "languages": ["rust"]}'
```

Response: the [`/stats`](#get-stats) shape, plus counts of new, modified and unchanged
files since the last build (any change still rebuilds the whole index). Each count is left
out when 0:

```json
{
  "total_files": 94,
  "index_size_bytes": 745137,
  "last_updated": "2026-09-28T19:37:45+00:00",
  "files_by_language": { "Rust": 39, "Text": 18, "TypeScript": 11, "Lock": 1 },
  "lines_by_language": { "Rust": 2171, "Text": 230, "TypeScript": 834, "Lock": 1 },
  "new_files": 1,
  "modified_files": 1,
  "unchanged_files": 92,
  "corpus_bytes": 124258,
  "trigram_index_bytes": 369124
}
```

Other counts that can appear: `deleted_files`, `skipped_too_large`,
`skipped_bytes_too_large`, `skipped_binary`.

If another indexer holds the workspace lock, the call fails at once with `500`
`IndexLocked`; it does not wait.

---

## GET /health

```bash
curl -s http://127.0.0.1:7878/health
# {"service":"reflex","status":"ok"}
```

Always `200` while the server is up. It does not check the index.

---

## Errors

Errors raised by the handlers have a JSON body:

```json
{ "error": { "kind": "QuerySyntaxError", "message": "Query parameter 'q' cannot be empty" } }
```

| Status | `kind` | When |
| --- | --- | --- |
| `400` | `QuerySyntaxError` | Empty `q`; unknown `lang`. |
| `400` | `ParseError` | `POST /index` body is not valid JSON. |
| `404` | `IndexNotFound` | No index in the workspace (`/query`, `/stats`). |
| `404` | `NotFound` | Unknown path. |
| `500` | `IoError` | Invalid regex; pattern refused by the broad-query guard; query timeout; other failures. |
| `500` | `CacheCorrupted`, `CacheVersionMismatch`, `IndexLocked`, ... | Index problems. Rebuild with `POST /index` and `{"force": true}`. |

Query-string parsing errors are the exception. A missing `q`, a non-numeric
`limit`, a boolean other than `true`/`false`, or any `glob`/`exclude` value
returns `400` with a **plain-text** body:

```
HTTP/1.1 400 Bad Request
content-type: text/plain; charset=utf-8

Failed to deserialize query string: missing field `q`
```

Clients should check the status code before parsing the body as JSON.

---

## Languages

`lang` accepts these names (case-insensitive):

| Language | Name | Aliases |
| --- | --- | --- |
| Rust | `rust` | `rs` |
| Python | `python` | `py` |
| JavaScript | `javascript` | `js` |
| TypeScript | `typescript` | `ts` |
| Vue | `vue` | |
| Svelte | `svelte` | |
| Go | `go` | |
| Java | `java` | |
| PHP | `php` | |
| C | `c` | |
| C++ | `cpp` | `c++` |
| C# | `csharp` | `cs`, `c#` |
| Ruby | `ruby` | `rb` |
| Kotlin | `kotlin` | `kt` |
| Zig | `zig` | |
| Plain-text tier | `text` | `txt`, `plaintext`, `plain` |
| Lock files | `lock` | `lockfile`, `lockfiles` |
| Generated files | `generated` | `gen` |

The `language` field in results is lowercase: `rust`, `python`, `javascript`,
`typescript`, `vue`, `svelte`, `go`, `java`, `php`, `c`, `cpp`, `csharp`, `ruby`,
`kotlin`, `swift`, `zig`, `text`, `lock`, `generated` or `unknown`. New values may
appear in minor releases. Treat any value you do not recognize like `unknown`.

Swift files are indexed and searchable as text (`language: "swift"`), but `lang`
does not accept `swift` and symbol searches return nothing for them.

---

## Known limitations

These are current behaviours of `rfx serve`, not design goals.

- **`glob` and `exclude` cannot be used.** They are declared as lists, and the
  query-string parser cannot fill a list, so any value (`glob=src/**/*.rs`,
  repeated or not) returns `400` `Failed to deserialize query string: invalid type:
  string "...", expected a sequence`. Use `file` (path substring) or `lang`.
- **No `include_locks`, `include_generated` or `exclude_text`.** These are MCP and
  CLI options. Over HTTP they are unknown parameters and are ignored. Use
  `lang=lock` or `lang=generated`.
- **No count mode and no context lines.** `context_before`/`context_after` never
  appear in HTTP results.
- **Unknown query parameters are ignored**, so a misspelled parameter silently
  has no effect.
- **`POST /index` does not read `.reflex/config.toml`.** It indexes with built-in
  defaults, so project settings such as `[index] exclude`, `text_tier`, `mode`,
  `hidden` and `max_file_size` are not applied. Run `rfx index` if you rely on them.
- **`POST /index` recognizes fewer language names** than `lang`. `csharp`, `ruby`,
  `kotlin` and `zig` are dropped; if every name is dropped, all languages are indexed.
- **Hints and error messages use MCP and CLI names.** A `hint` may say
  `contains:true` (use `contains=true`) or `include_locks:true` (use `lang=lock`);
  the broad-query error says `--force` (use `force=true`).

---

## Client examples

JavaScript:

```javascript
async function searchCode(pattern, options = {}) {
  const params = new URLSearchParams({ q: pattern, ...options });
  const response = await fetch(`http://127.0.0.1:7878/query?${params}`);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }
  const data = await response.json();
  if (!data.can_trust_results) {
    console.warn('Index is stale:', data.warning?.reason);
  }
  // Flatten file groups into path:line entries.
  return data.results.flatMap(file =>
    file.matches.map(m => ({ path: file.path, line: m.span.start_line, text: m.preview }))
  );
}

const hits = await searchCode('QueryEngine', { symbols: 'true', lang: 'rust', limit: '10' });
```

Python:

```python
import requests

def search_code(pattern, **params):
    params = {"q": pattern, **{k: str(v).lower() if isinstance(v, bool) else v
                               for k, v in params.items()}}
    response = requests.get("http://127.0.0.1:7878/query", params=params)
    response.raise_for_status()
    data = response.json()
    if not data["can_trust_results"]:
        print("Index is stale:", data["warning"]["reason"])
    return [
        (group["path"], match["span"]["start_line"], match["preview"])
        for group in data["results"]
        for match in group["matches"]
    ]

hits = search_code("QueryEngine", symbols=True, lang="rust", limit=10)
```

Python's `requests` sends `True` as the string `True`, which the server rejects
with `400`. The helper above sends `true`.

Reindex when stale:

```bash
status=$(curl -s 'http://127.0.0.1:7878/query?q=main&limit=0' | jq -r .status)
[ "$status" = "stale" ] && curl -s -X POST http://127.0.0.1:7878/index > /dev/null
```

---

## Security

`rfx serve` is built for local, single-user use.

- **Loopback by default.** It binds to `127.0.0.1:7878`.
- **No authentication.** There are no API keys, tokens, access controls or rate
  limits. Anyone who can reach the port can read every indexed file through
  `/query` and trigger `POST /index`.
- **Permissive CORS.** Any origin may call the API from a browser.
- **Non-loopback hosts.** `--host 0.0.0.0` (or any address other than
  `127.0.0.1`, `::1` or `localhost`) exposes the index to the network. The server
  prints a warning on stderr when it starts this way. Do not do this on shared or
  internet-facing machines.

For remote access, put a reverse proxy with authentication in front of it, or use
an SSH tunnel:

```bash
ssh -L 7878:127.0.0.1:7878 devbox    # then query http://127.0.0.1:7878 locally
```

---

## Further reading

- [README](../README.md): installation and CLI usage
- [MCP tool cheatsheet](mcp-tool-cheatsheet.md): the MCP server, which has more
  search options than HTTP
- [Architecture](ARCHITECTURE.md): how the index works
- [Changelog](../CHANGELOG.md)
