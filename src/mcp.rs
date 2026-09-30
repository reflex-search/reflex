//! MCP (Model Context Protocol) server implementation
//!
//! This module implements the MCP protocol directly over stdio using JSON-RPC 2.0.
//! It exposes Reflex's code search capabilities as MCP tools for AI coding assistants.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};
use std::path::Path;

use crate::auto_update::{UpdateOptions, Updated, update_if_stale};
use crate::cache::CacheManager;
use crate::dependency::DependencyIndex;
use crate::indexer::Indexer;
use crate::line_filter;
use crate::models::{IndexStatus, Language, SymbolKind};
use crate::query::{QueryEngine, QueryFilter};
use crate::semantic::config::load_mcp_config;

/// Default preview truncation length for MCP responses (characters).
/// Raised from 100 so typical Rust/TS/Python function signatures fit without truncation.
const DEFAULT_MCP_PREVIEW_LENGTH: usize = 180;

/// Default page size for MCP list results when the caller does not specify a
/// `limit`. Raised from 50 → 200 (REF-191).
///
/// Rationale: the efficacy benchmark's residual gap was **turn count** — Reflex
/// reached parity only when it answered in the same number of turns as `grep`.
/// `grep -rn` returns every occurrence in one shot; a 50-result default forces
/// an agent doing a find-all task (e.g. 122 occurrences of `extract_symbols`)
/// to paginate, spending an extra MCP call for the same answer. 200 covers the
/// overwhelming majority of find-all result sets in a single "decisive" call
/// while still bounding token cost (hard cap remains 500 via the `min(500)`
/// clamp on explicit limits). Callers who want a cheap cardinality probe should
/// use `mode="count"`.
const DEFAULT_MCP_RESULT_LIMIT: usize = 200;

/// Characters of the matching line `list_locations` returns with `preview: true`.
const LOCATION_PREVIEW_CHARS: usize = 120;

/// Returns true if every occurrence of `pattern` in `preview` falls inside a
/// string literal or comment for the given `lang`. Conservative: returns false
/// (keep the match) when the language has no filter or the pattern is not found.
fn is_in_string_or_comment(lang: Language, preview: &str, pattern: &str) -> bool {
    let Some(filter) = line_filter::get_filter(lang) else {
        return false;
    };

    let mut pos = 0;
    let mut found = false;

    while pos < preview.len() {
        let Some(rel) = preview[pos..].find(pattern) else {
            break;
        };
        let abs = pos + rel;
        found = true;
        if !filter.is_in_comment(preview, abs) && !filter.is_in_string(preview, abs) {
            return false; // at least one occurrence is in real code — keep the match
        }
        pos = abs + 1;
    }

    found
}

/// JSON-RPC 2.0 request
#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

/// JSON-RPC 2.0 response
#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    // Per JSON-RPC 2.0, Notifications must omit `id` entirely (not set it to null).
    // We never construct a JsonRpcResponse for a Notification, but skip_serializing_if
    // is a defensive guard against ever emitting `"id": null`, which strict clients
    // (e.g. Claude Code's Zod validators) reject.
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error
#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

/// Parse language string to Language enum
fn parse_language(lang: Option<String>) -> Option<Language> {
    lang.as_deref().and_then(Language::from_name)
}

/// The `paths: true` response: file paths only, no preview rows.
///
/// `{status, can_trust_results, paths, total_files}`; `has_more` appears only
/// when a `limit` cut the list, `warning` only when the index is stale, and
/// `warnings` / `hint` only when the engine set them. The freshness fields stay
/// because every tool response carries them (see `check_index_status`).
fn paths_only_result(response: &crate::models::QueryResponse) -> serde_json::Value {
    let paths: Vec<&str> = response.results.iter().map(|fg| fg.path.as_str()).collect();
    let mut result = json!({
        "status": response.status,
        "can_trust_results": response.can_trust_results,
        "paths": paths,
        "total_files": paths.len(),
    });
    if let Some(warning) = &response.warning {
        result["warning"] = json!(warning);
    }
    if response.pagination.has_more {
        result["has_more"] = json!(true);
    }
    annotate_literal_result(&mut result, response);
    result
}

/// Attach the engine's literal-search safety nets to a compact response object.
///
/// The engine (`QueryEngine::search_with_metadata`) owns both messages, so the CLI,
/// HTTP and MCP surfaces cannot disagree:
///
/// * `warnings` — the pattern was rewritten (brackets → escaped regex).
/// * `hint` — the result is empty but substring matches exist.
///
/// A handler that serialises the whole `QueryResponse` already carries both fields;
/// this is for the tools that build their own compact object, and it is idempotent
/// (`insert` replaces), so calling it on a full response is harmless.
fn annotate_literal_result(response: &mut Value, engine: &crate::models::QueryResponse) {
    annotate_literal_fields(
        response,
        &engine.warnings,
        engine.hint.as_deref(),
        engine.excluded_reason,
        engine.excluded_by_default,
    );
}

/// [`annotate_literal_result`] for a handler that has already moved the engine
/// response into its JSON and kept only these fields.
fn annotate_literal_fields(
    response: &mut Value,
    warnings: &[String],
    hint: Option<&str>,
    excluded_reason: Option<crate::query::ExcludedReason>,
    excluded_by_default: Option<usize>,
) {
    let Some(obj) = response.as_object_mut() else {
        return;
    };
    if !warnings.is_empty() {
        obj.insert("warnings".to_string(), json!(warnings));
    }
    if let Some(hint) = hint {
        obj.insert("hint".to_string(), json!(hint));
    }
    // The machine-readable cause beside the prose, and the scoped lock/generated
    // count, so a harness can branch without parsing the sentence.
    if let Some(reason) = excluded_reason {
        obj.insert("excluded_reason".to_string(), json!(reason));
    }
    if let Some(n) = excluded_by_default {
        obj.insert("excluded_by_default".to_string(), json!(n));
    }
}

/// The exact total of a search that ran without a limit (or with
/// `require_exact_total`). Such a search is always exact; the fallback to the
/// returned match count only guards the type, never a real path.
fn exact_total_or_count(response: &crate::models::QueryResponse) -> usize {
    response
        .pagination
        .exact_total()
        .unwrap_or_else(|| response.results.iter().map(|fg| fg.matches.len()).sum())
}

/// Parse symbol kind string to SymbolKind enum
fn parse_symbol_kind(kind: Option<String>) -> Option<SymbolKind> {
    kind.as_deref().and_then(|s| {
        let capitalized = {
            let mut chars = s.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first
                    .to_uppercase()
                    .chain(chars.flat_map(|c| c.to_lowercase()))
                    .collect(),
            }
        };

        capitalized
            .parse::<SymbolKind>()
            .ok()
            .or_else(|| Some(SymbolKind::Unknown(s.to_string())))
    })
}

/// Server instructions sent in the `initialize` response.
///
/// Written for agents that never see the tool schemas: Claude Code defers MCP
/// tool schemas until the model calls `ToolSearch`, and in 34 measured
/// sessions 14 of 16 opened with a wrong argument name (`query` instead of
/// `pattern`). So the text carries the exact call shapes and the canonical
/// argument names, plus the ToolSearch hint. Keep it compact: every consumer
/// pays these tokens once per session. Guarded by tests
/// (`test_instructions_*`).
const MCP_INSTRUCTIONS: &str = r#"Reflex is the full-text code search engine for this workspace. For any task that asks where code is, where a pattern occurs, where a symbol is defined or used, or who imports a file, prefer a Reflex tool over Grep, Glob, ripgrep, or shell grep: call Reflex first. It returns grep's data plus line context, symbol typing, and dependency links, instantly.

Your harness may defer MCP tool schemas. If Reflex tools appear in a deferred-tools list, load them first: ToolSearch("select:mcp__reflex__search_code,mcp__reflex__search_regex,mcp__reflex__find_references").

Exact call shapes:
search_code {"pattern": "fn verify_totp", "limit": 40, "file": "src/identity"}
search_regex {"pattern": "fn (get|set)_\w+", "glob": ["src/**/*.rs"]}
find_references {"pattern": "start_webauthn_registration"}

Rules: the required argument is always "pattern", never "query", "symbol", or "text". The result cap is "limit", never "max_results". The path filter is "file" (substring) or "glob" (array), never "path". search_code is a literal text index: natural-language queries match nothing; search for identifiers or code fragments.

Matching: search_code, list_locations and find_references match WHOLE identifiers, like grep -w: "verify_csrf" does not match "verify_csrf_form_field". contains:true matches substrings (grep -F); ignore_case:true is rg -i. A pattern with brackets runs as an escaped regex. A zero result carries a hint saying why.

Coverage matches ripgrep's defaults: not gitignored, not binary, not under a dot-directory (.github/, .githooks/ …); use grep for hidden paths. Lock and generated files need include_locks / include_generated. glob and exclude follow gitignore rules: "src/**/*.rs" is anchored at the root, "*.rs" matches at any depth.

The index updates itself before every call and is built on first use: never call index_project or check_index_status after edits. can_trust_results: false means the update could not run; warnings say why.

To list where something occurs, use list_locations: path and line only, the cheapest answer; preview:true adds each matching line. Use search_code for context or symbols, find_references for a definition plus every call site without string/comment noise, get_dependencies with reverse:true for what imports a file. If a Reflex tool fails, retry it once; only fall back to Grep/Glob after the retry also fails."#;

/// Handle initialize request
fn handle_initialize(_params: Option<Value>) -> Result<Value> {
    Ok(json!({
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "reflex",
            "version": env!("CARGO_PKG_VERSION")
        },
        "instructions": MCP_INSTRUCTIONS
    }))
}

/// Parameter schemas shared by the search tools. Short on purpose: Claude Code sends
/// every listed schema on every turn, so each word here is paid per turn per session
/// (the matching and coverage rules are said once, in `MCP_INSTRUCTIONS`).
fn search_params(extra: Value) -> Value {
    let mut props = json!({
        "pattern": {"type": "string", "description": "Text to find"},
        "lang": {"type": "string", "description": "Language filter: rust, python, typescript, text, …"},
        "file": {"type": "string", "description": "Only paths containing this substring"},
        "glob": {"type": "array", "items": {"type": "string"}, "description": "Only paths matching (gitignore rules)"},
        "exclude": {"type": "array", "items": {"type": "string"}, "description": "Skip paths matching (gitignore rules)"},
        "ignore_case": {"type": "boolean", "description": "Case-insensitive (rg -i)"},
        "include_locks": {"type": "boolean", "description": "Also search lock files"},
        "include_generated": {"type": "boolean", "description": "Also search generated files"},
        "force": {"type": "boolean", "description": "Run a pattern too broad to run by default"}
    });
    if let (Some(p), Some(e)) = (props.as_object_mut(), extra.as_object()) {
        for (k, v) in e {
            p.insert(k.clone(), v.clone());
        }
    }
    json!({"type": "object", "properties": props, "required": ["pattern"]})
}

/// The tools `tools/list` returns. `analyze` is left out when
/// `[mcp] enable_structural_tools = false` (`~/.reflex/config.toml`).
///
/// 2026-09-30: 17 tools with ~44 KB of schemas became these 9 (~13 KB). Claude Code
/// carries every listed schema on every turn, and that prefix was the whole token
/// gap to Grep in long sessions (`.context/AUTO_UPDATE_RESEARCH.md`). The removed
/// names still work, unlisted ([`LEGACY_TOOLS`]).
fn tool_list(enable_structural: bool) -> Vec<Value> {
    let contains = json!({"type": "boolean", "description": "Substring match (grep -F) instead of whole identifiers"});
    let limit = json!({"type": "integer", "description": "Max results (default 200, at most 500)"});
    let offset = json!({"type": "integer", "description": "Skip this many results (next page)"});
    let mode = json!({"type": "string", "enum": ["list", "count"], "description": "count: {count, files} only"});
    let paths = json!({"type": "boolean", "description": "Return file paths only"});
    let mut tools = vec![
        json!({
            "name": "search_code",
            "description": "Search code for a literal pattern: every match with path, line and preview (for path and line only, list_locations is cheaper). Whole identifiers by default; contains:true for substrings; symbols:true for definitions only (kind narrows them). Answer: {columns, rows} (each row aligns with columns) plus pagination; when has_more, fetch the next page with offset. total_count is exact only when total_is_exact.",
            "inputSchema": search_params(json!({
                "contains": contains,
                "symbols": {"type": "boolean", "description": "Definitions only"},
                "kind": {"type": "string", "description": "Symbol kind: function, struct, class, trait, …"},
                "exact": {"type": "boolean", "description": "Exact symbol name"},
                "expand": {"type": "boolean", "description": "Whole symbol body"},
                "dependencies": {"type": "boolean", "description": "Attach each file's imports"},
                "preview_length": {"type": "integer", "description": "Preview characters (default 180)"},
                "mode": mode,
                "paths": paths,
                "limit": limit,
                "offset": offset
            }))
        }),
        json!({
            "name": "search_regex",
            "description": "Search code with a regular expression (Rust regex): alternation, classes, anchors, e.g. `fn (get|set)_\\w+` or `->with\\(`. In JSON double each backslash. Same filters and answer as search_code.",
            "inputSchema": search_params(json!({
                "dependencies": {"type": "boolean", "description": "Attach each file's imports"},
                "mode": mode,
                "paths": paths,
                "limit": limit,
                "offset": offset
            }))
        }),
        json!({
            "name": "list_locations",
            "description": "Where does X occur? Every match as {path, line}: the cheapest search, no limit. preview:true adds each matching line (trimmed, 120 chars). Same matching as search_code.",
            "inputSchema": search_params(json!({
                "contains": contains,
                "preview": {"type": "boolean", "description": "Add each matching line (120 chars)"},
                "dependencies": {"type": "boolean", "description": "Attach each file's imports"}
            }))
        }),
        json!({
            "name": "find_references",
            "description": "A symbol's definition and every usage in one call, in code files only; matches in strings and comments are left out (include_strings:true keeps them). Answer: {definition, references, total_references, returned_count, filtered_out, pagination}; mode:\"count\" returns the count after filtering.",
            "inputSchema": search_params(json!({
                "contains": contains,
                "kind": {"type": "string", "description": "Symbol kind of the definition"},
                "include_strings": {"type": "boolean", "description": "Keep matches in strings and comments"},
                "mode": mode,
                "limit": limit,
                "offset": offset
            }))
        }),
        json!({
            "name": "search_ast",
            "description": "Tree-sitter structural search, e.g. `(function_item) @fn`. Slow: parses every file that lang and glob select, so always pass glob. Prefer search_code with symbols:true.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Tree-sitter query (S-expression)"},
                    "lang": {"type": "string", "description": "Language of the query"},
                    "file": {"type": "string", "description": "Only paths containing this substring"},
                    "glob": {"type": "array", "items": {"type": "string"}, "description": "Only paths matching (gitignore rules)"},
                    "exclude": {"type": "array", "items": {"type": "string"}, "description": "Skip paths matching"},
                    "force": {"type": "boolean", "description": "Run without a glob"},
                    "dependencies": {"type": "boolean", "description": "Attach each file's imports"},
                    "paths": paths,
                    "limit": limit,
                    "offset": offset
                },
                "required": ["pattern", "lang"]
            }
        }),
        json!({
            "name": "get_dependencies",
            "description": "The imports of a file: path, line, internal/external/stdlib. reverse:true lists the files that import it instead; depth:N follows imports N levels (a list of {path, depth}). Static imports only; path may be a fragment or file name.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, fragment or name"},
                    "reverse": {"type": "boolean", "description": "Files that import this file"},
                    "depth": {"type": "integer", "description": "Follow imports this many levels"}
                },
                "required": ["path"]
            }
        }),
        json!({
            "name": "analyze",
            "description": "Import-graph analysis. kind: summary (counts), hotspots (most-imported files), circular (import cycles), unused (files nothing imports; entry points included), islands (disconnected groups). Answers are paginated with limit/offset.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["summary", "hotspots", "circular", "unused", "islands"]},
                    "limit": {"type": "integer", "description": "Max results (default 200)"},
                    "offset": offset,
                    "sort": {"type": "string", "description": "asc or desc"},
                    "min_dependents": {"type": "integer", "description": "hotspots/summary: minimum importers"},
                    "min_island_size": {"type": "integer", "description": "islands: minimum files"},
                    "max_island_size": {"type": "integer", "description": "islands: maximum files"}
                },
                "required": ["kind"]
            }
        }),
        json!({
            "name": "gather_context",
            "description": "Project overview: structure, file types, frameworks, entry points, test layout, config files. Pass flags to pick sections (default: all).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Subdirectory"},
                    "depth": {"type": "integer", "description": "Tree depth"},
                    "structure": {"type": "boolean"},
                    "file_types": {"type": "boolean"},
                    "project_type": {"type": "boolean"},
                    "framework": {"type": "boolean"},
                    "entry_points": {"type": "boolean"},
                    "test_layout": {"type": "boolean"},
                    "config_files": {"type": "boolean"}
                }
            }
        }),
        json!({
            "name": "index_project",
            "description": "Force an index run. Rarely needed: every tool updates the index before it answers. force:true rebuilds a corrupted index.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "force": {"type": "boolean", "description": "Full rebuild"},
                    "languages": {"type": "array", "items": {"type": "string"}, "description": "Only these languages"}
                }
            }
        }),
        json!({
            "name": "check_index_status",
            "description": "Report whether the index matches the files on disk, without updating it. Rarely needed: every other tool updates the index first.",
            "inputSchema": {"type": "object", "properties": {}}
        }),
    ];
    if !enable_structural {
        tools.retain(|t| t["name"] != "analyze");
    }
    tools
}

/// Tool names removed from `tools/list` on 2026-09-30 that still work: each call
/// runs as the named replacement and answers with a deprecation warning. Their
/// arguments are checked against the old schema (properties only, no text).
const LEGACY_TOOLS: &[(&str, &str)] = &[
    ("count_occurrences", "search_code with mode:\"count\""),
    ("get_dependents", "get_dependencies with reverse:true"),
    ("get_transitive_deps", "get_dependencies with depth:N"),
    ("find_hotspots", "analyze with kind:\"hotspots\""),
    ("find_circular", "analyze with kind:\"circular\""),
    ("find_unused", "analyze with kind:\"unused\""),
    ("find_islands", "analyze with kind:\"islands\""),
    ("analyze_summary", "analyze with kind:\"summary\""),
];

/// The input schemas of [`LEGACY_TOOLS`], for argument checking only.
fn legacy_schemas() -> Vec<(&'static str, Value)> {
    let int = json!({"type": "integer"});
    let string = json!({"type": "string"});
    let schema = |props: Value, required: &[&str]| json!({"type": "object", "properties": props, "required": required});
    vec![
        (
            "count_occurrences",
            search_params(json!({
                "contains": {"type": "boolean"}, "symbols": {"type": "boolean"},
                "kind": string, "dependencies": {"type": "boolean"}
            })),
        ),
        ("get_dependents", schema(json!({"path": string}), &["path"])),
        (
            "get_transitive_deps",
            schema(json!({"path": string, "depth": int}), &["path"]),
        ),
        (
            "find_hotspots",
            schema(
                json!({"limit": int, "offset": int, "sort": string, "min_dependents": int}),
                &[],
            ),
        ),
        (
            "find_circular",
            schema(json!({"limit": int, "offset": int, "sort": string}), &[]),
        ),
        (
            "find_unused",
            schema(json!({"limit": int, "offset": int}), &[]),
        ),
        (
            "find_islands",
            schema(
                json!({"limit": int, "offset": int, "sort": string,
                                       "min_island_size": int, "max_island_size": int}),
                &[],
            ),
        ),
        (
            "analyze_summary",
            schema(json!({"min_dependents": int}), &[]),
        ),
    ]
}

/// Handle tools/list request
fn handle_list_tools(_params: Option<Value>, enable_structural: bool) -> Result<Value> {
    Ok(json!({ "tools": tool_list(enable_structural) }))
}

/// Handle tools/call request
/// Build a successful MCP `tools/call` result.
///
/// REF-215: Reflex emits only the spec-guaranteed `content[text]` baseline — the
/// result data serialized as a JSON string. The spec-optional `structuredContent`
/// field (REF-202) was dropped per the REF-196 board decision: every conforming
/// MCP client must handle `content[text]`, it is what reaches the model in
/// Reflex's primary Claude Code use case, and emitting a single field removes any
/// chance of a client consuming both and double-counting tokens. The columnar
/// `{columns, rows}` shape (REF-209) is independent and still lives inside this
/// text payload.
///
/// Only success paths use this — error results are surfaced through the JSON-RPC
/// error channel in `process_request`, never as a tool result, so there are no
/// `isError` responses to preserve here.
fn make_tool_result(data: Value) -> Value {
    json!({
        "content": [{"type": "text", "text": serde_json::to_string(&data).unwrap_or_default()}]
    })
}

/// Wrap tool data into the MCP `content` envelope, attaching any argument
/// normalisation warnings (deprecated alias used, duplicate key ignored).
///
/// JSON-object data gets a top-level `warnings` array. Prose data
/// (`Value::String`, e.g. `gather_context`) is emitted verbatim as
/// `content[text]`, with the warnings appended as trailing lines.
fn finish_tool_result(data: Value, warnings: Vec<String>) -> Value {
    match data {
        Value::String(text) => {
            let text = if warnings.is_empty() {
                text
            } else {
                format!("{}\n\nwarnings: {}", text, warnings.join("; "))
            };
            json!({ "content": [{ "type": "text", "text": text }] })
        }
        mut other => {
            if !warnings.is_empty()
                && let Some(obj) = other.as_object_mut()
            {
                // Merge, don't overwrite: a handler may have already attached its own
                // warnings (for example, a bracket pattern rewritten to a regex).
                let mut all = obj
                    .get("warnings")
                    .and_then(|w| w.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(str::to_string))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                all.extend(warnings);
                obj.insert("warnings".to_string(), json!(all));
            }
            make_tool_result(other)
        }
    }
}

// ---------------------------------------------------------------------------
// Argument validation (1.7.0)
//
// Field data from 34 real Claude Code sessions: 20 of 21 Reflex failures were
// `Missing pattern` because the agent sent `query`, `symbol`, `max_results`
// or `path`. The schema was correct; the agent never saw it (deferred tool
// schemas). Three defences now sit in front of every tool arm:
//
// 1. Aliases: the habitual wrong names are accepted and rewritten, with a
//    `warnings` entry in the response so the next call is right.
// 2. Unknown keys are rejected with the received keys, the valid keys and a
//    nearest-match suggestion (previously they were silently dropped, so a
//    `max_results: 40` call quietly returned 200 results).
// 3. Numeric strings (`"40"`) and boolean strings (`"true"`) are coerced;
//    anything else that is the wrong type is a typed error, not a silent
//    fallback to the default.
//
// The valid-key and required-key lists are read from the `tools/list` schema
// at runtime, so the validator can never drift from what clients are shown.
// ---------------------------------------------------------------------------

/// Wrong-but-common argument names and the key each one means.
///
/// `path` is only an alias on tools whose schema has no real `path` key
/// (the dependency/context tools take a real `path`).
const ARG_ALIASES: &[(&str, &str)] = &[
    ("query", "pattern"),
    ("symbol", "pattern"),
    ("text", "pattern"),
    ("search", "pattern"),
    ("max_results", "limit"),
    ("path", "file"),
];

/// Keys that must be non-negative integers. Numeric strings are coerced.
const NUMERIC_ARG_KEYS: &[&str] = &[
    "limit",
    "offset",
    "depth",
    "preview_length",
    "min_dependents",
    "min_island_size",
    "max_island_size",
];

/// Keys that must be booleans. `"true"` / `"false"` strings are coerced.
const BOOL_ARG_KEYS: &[&str] = &[
    "symbols",
    "exact",
    "contains",
    "ignore_case",
    "include_locks",
    "include_generated",
    "expand",
    "paths",
    "force",
    "dependencies",
    "include_strings",
    "preview",
    "reverse",
    "structure",
    "file_types",
    "project_type",
    "framework",
    "entry_points",
    "test_layout",
    "config_files",
];

/// Keys that must be strings when present.
const STRING_ARG_KEYS: &[&str] = &["pattern", "lang", "kind", "mode", "file", "path", "sort"];

/// Keys that must be arrays of strings when present.
const STRING_ARRAY_ARG_KEYS: &[&str] = &["glob", "exclude", "languages"];

/// Per-tool argument contract, derived from the published `inputSchema`.
#[derive(Debug, Clone)]
struct ToolSpec {
    /// Every key the schema declares under `properties`, in schema order.
    valid: Vec<String>,
    /// Keys the schema lists under `required`.
    required: Vec<String>,
}

/// All tool specs, built once from `handle_list_tools` with structural tools
/// included so hidden tools still validate when called directly.
fn tool_specs() -> &'static std::collections::HashMap<String, ToolSpec> {
    static SPECS: std::sync::OnceLock<std::collections::HashMap<String, ToolSpec>> =
        std::sync::OnceLock::new();
    SPECS.get_or_init(|| {
        let mut map = std::collections::HashMap::new();
        let mut listing = tool_list(true);
        listing.extend(
            legacy_schemas()
                .into_iter()
                .map(|(name, schema)| json!({"name": name, "inputSchema": schema})),
        );
        for tool in &listing {
            let Some(name) = tool["name"].as_str() else {
                continue;
            };
            let required: Vec<String> = tool["inputSchema"]["required"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            // Required keys first so error text leads with `pattern`; the rest
            // follow in serde_json's (alphabetical) map order.
            let mut valid: Vec<String> = required.clone();
            for key in tool["inputSchema"]["properties"]
                .as_object()
                .map(|o| o.keys().cloned().collect::<Vec<_>>())
                .unwrap_or_default()
            {
                if !valid.contains(&key) {
                    valid.push(key);
                }
            }
            map.insert(name.to_string(), ToolSpec { valid, required });
        }
        map
    })
}

/// Spec for one tool, `None` when the tool does not exist.
fn tool_spec(name: &str) -> Option<&'static ToolSpec> {
    tool_specs().get(name)
}

/// Classic two-row Levenshtein edit distance (ASCII keys, so bytes suffice).
fn levenshtein(a: &str, b: &str) -> usize {
    let a = a.as_bytes();
    let b = b.as_bytes();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Closest valid key to `key`, if any is close enough to be a likely typo.
///
/// Known aliases count too: `max_resultz` is near the alias `max_results`,
/// so the suggestion is that alias's canonical key, `limit`.
fn nearest_key<'a>(key: &str, valid: &'a [String]) -> Option<&'a str> {
    let threshold = 2.max(key.len() / 3);
    let direct = valid.iter().map(|v| (levenshtein(key, v), v.as_str()));
    let via_alias = ARG_ALIASES.iter().filter_map(|(alias, canonical)| {
        valid
            .iter()
            .find(|v| v == canonical)
            .map(|v| (levenshtein(key, alias), v.as_str()))
    });
    direct
        .chain(via_alias)
        .filter(|(d, _)| *d <= threshold)
        .min_by_key(|(d, _)| *d)
        .map(|(_, v)| v)
}

fn keys_list(keys: &[String]) -> String {
    serde_json::to_string(keys).unwrap_or_default()
}

fn invalid_params(msg: String) -> anyhow::Error {
    crate::errors::ReflexError::InvalidParams(msg).into()
}

/// Short human rendering of a JSON value for error text.
fn describe_value(v: &Value) -> String {
    match v {
        Value::String(s) => format!("\"{}\"", s),
        other => other.to_string(),
    }
}

/// Apply aliases, reject unknown keys, check required keys, coerce types.
///
/// Returns the rewritten arguments plus warnings for the caller. Every error
/// is [`ReflexError::InvalidParams`] and maps to JSON-RPC `-32602`.
fn normalize_arguments(tool: &str, spec: &ToolSpec, args: Value) -> Result<(Value, Vec<String>)> {
    let mut obj = match args {
        Value::Object(o) => o,
        Value::Null => serde_json::Map::new(),
        other => {
            return Err(invalid_params(format!(
                "Invalid arguments for {}: expected a JSON object, got {}",
                tool,
                describe_value(&other)
            )));
        }
    };
    let received: Vec<String> = obj.keys().cloned().collect();
    let mut warnings = Vec::new();

    // Pass 1: aliases.
    for (alias, canonical) in ARG_ALIASES {
        if spec.valid.iter().any(|k| k == alias) {
            continue; // a real key on this tool (e.g. `path` on get_dependencies)
        }
        if !spec.valid.iter().any(|k| k == canonical) {
            continue; // canonical key does not exist here; leave for the unknown-key pass
        }
        if let Some(value) = obj.remove(*alias) {
            if obj.contains_key(*canonical) {
                warnings.push(format!(
                    "ignored \"{}\" because \"{}\" was also given",
                    alias, canonical
                ));
            } else {
                warnings.push(format!(
                    "argument \"{}\" is deprecated; use \"{}\"",
                    alias, canonical
                ));
                obj.insert((*canonical).to_string(), value);
            }
        }
    }

    // Pass 2: unknown keys.
    for key in obj.keys() {
        if !spec.valid.iter().any(|k| k == key) {
            let hint = nearest_key(key, &spec.valid)
                .map(|s| format!(" (did you mean \"{}\"?)", s))
                .unwrap_or_default();
            return Err(invalid_params(format!(
                "Unknown argument \"{}\" for {}{}. Received: {}. Valid: {}",
                key,
                tool,
                hint,
                keys_list(&received),
                keys_list(&spec.valid)
            )));
        }
    }

    // Pass 3: required keys.
    for req in &spec.required {
        if !obj.contains_key(req) {
            return Err(invalid_params(format!(
                "Missing required argument \"{}\" for {}. Received: {}. Valid: {}",
                req,
                tool,
                keys_list(&received),
                keys_list(&spec.valid)
            )));
        }
    }

    // Pass 4: types and coercion.
    for (key, value) in obj.iter_mut() {
        let key = key.as_str();
        if NUMERIC_ARG_KEYS.contains(&key) {
            let coerced = match value {
                Value::Number(n) if n.as_u64().is_some() => None,
                Value::String(s) => s.trim().parse::<u64>().ok().map(Value::from),
                _ => None,
            };
            match coerced {
                Some(n) => *value = n,
                None if value.as_u64().is_some() => {}
                None => {
                    return Err(invalid_params(format!(
                        "Invalid value for \"{}\": expected a non-negative integer, got {}",
                        key,
                        describe_value(value)
                    )));
                }
            }
        } else if BOOL_ARG_KEYS.contains(&key) {
            let coerced = match value {
                Value::Bool(_) => None,
                Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
                    "true" => Some(Value::Bool(true)),
                    "false" => Some(Value::Bool(false)),
                    _ => None,
                },
                _ => None,
            };
            match coerced {
                Some(b) => *value = b,
                None if value.is_boolean() => {}
                None => {
                    return Err(invalid_params(format!(
                        "Invalid value for \"{}\": expected a boolean, got {}",
                        key,
                        describe_value(value)
                    )));
                }
            }
        } else if STRING_ARG_KEYS.contains(&key) && !value.is_string() {
            return Err(invalid_params(format!(
                "Invalid value for \"{}\": expected a string, got {}",
                key,
                describe_value(value)
            )));
        } else if STRING_ARRAY_ARG_KEYS.contains(&key) {
            // A bare string is a common shorthand for a one-element list.
            if let Value::String(s) = value {
                *value = json!([s]);
            } else if !value
                .as_array()
                .map(|a| a.iter().all(Value::is_string))
                .unwrap_or(false)
            {
                return Err(invalid_params(format!(
                    "Invalid value for \"{}\": expected an array of strings, got {}",
                    key,
                    describe_value(value)
                )));
            }
        }
    }

    Ok((Value::Object(obj), warnings))
}

// ---------------------------------------------------------------------------
// Cache corruption auto-recovery (1.7.0)
// ---------------------------------------------------------------------------

/// Build (or force-rebuild) the index for `root`. Shared by the
/// `index_project` tool and the corruption auto-recovery path.
fn rebuild_index(
    root: &Path,
    force: bool,
    languages: Vec<Language>,
) -> Result<crate::models::IndexStats> {
    let cache = CacheManager::new(root);
    // Read before a forced clear: the saved `--languages` lives in meta.db.
    let mut config = cache.effective_index_config(&languages)?;
    config.lock_wait_secs = crate::models::LOCK_WAIT_FOREVER;
    if force {
        log::info!("Force rebuild requested, clearing existing cache");
        cache.clear()?;
    }
    let indexer = Indexer::new(cache, config);
    indexer.index(root, false)
}

fn is_cache_corrupted(e: &anyhow::Error) -> bool {
    matches!(
        e.downcast_ref::<crate::errors::ReflexError>(),
        Some(crate::errors::ReflexError::CacheCorrupted(_))
    )
}

/// Run a tool; if it fails because the on-disk cache is corrupted, rebuild
/// the index once (force) and retry once. `index_project` itself is never
/// wrapped, so recovery cannot recurse.
fn with_corruption_recovery(
    name: &str,
    root: &Path,
    mut run: impl FnMut() -> Result<Value>,
) -> Result<Value> {
    let first = match run() {
        Ok(v) => return Ok(v),
        Err(e) => e,
    };
    if name == "index_project" || !is_cache_corrupted(&first) {
        return Err(first);
    }

    log::warn!(
        "Cache corruption detected during {}; rebuilding index once and retrying: {}",
        name,
        first
    );
    if let Err(rebuild_err) = rebuild_index(root, true, Vec::new()) {
        if rebuild_err
            .downcast_ref::<crate::errors::ReflexError>()
            .is_some_and(|re| matches!(re, crate::errors::ReflexError::IndexLocked(_)))
        {
            // Someone else is already rebuilding; let the caller retry later.
            return Err(rebuild_err);
        }
        return Err(anyhow::anyhow!(
            "Index is corrupted and automatic rebuild failed: {}. \
             Call the index_project tool with {{\"force\": true}}.",
            rebuild_err
        ));
    }
    run().map_err(|e| {
        anyhow::anyhow!(
            "Index was rebuilt after corruption but {} still failed: {}. \
             Call the index_project tool with {{\"force\": true}}.",
            name,
            e
        )
    })
}

/// Error text as an MCP client should read it: an agent cannot run the CLI,
/// so `rfx index` advice becomes `index_project` advice. CLI/HTTP output is
/// untouched (they format the error themselves).
fn mcp_facing_message(e: &anyhow::Error) -> String {
    use crate::errors::ReflexError;
    match e.downcast_ref::<ReflexError>() {
        Some(ReflexError::IndexNotFound) => {
            "Index not found. Call the index_project tool, then retry.".to_string()
        }
        Some(ReflexError::CacheCorrupted(inner)) => format!(
            "Cache appears to be corrupted: {}. Call the index_project tool with {{\"force\": true}}, then retry.",
            inner
        ),
        // Never surface SQLite's "database is locked" for this. Name the process, its
        // progress, and the fact that waiting is the right move.
        Some(
            err @ ReflexError::SymbolIndexingInProgress {
                processed, total, ..
            },
        ) => {
            let pct = if *total > 0 {
                format!(" ({}%)", processed * 100 / total)
            } else {
                String::new()
            };
            format!(
                "{}{}. It holds the index database. Wait a few seconds and call index_project again; \
                 searches keep working from the existing index meanwhile.",
                err, pct
            )
        }
        Some(err @ ReflexError::CacheVersionMismatch { .. }) => format!(
            "{} Call the index_project tool with {{\"force\": true}} to rebuild it for this version.",
            err
        ),
        _ => e.to_string(),
    }
}

/// REF-209: columnar result-format toggle for `search_code` / `search_regex`.
///
/// Default ON. Returns `false` only when `REFLEX_MCP_COLUMNAR` is explicitly set
/// to a falsey value (`0`/`false`/`off`/`no`, case-insensitive), which restores
/// the legacy file-grouped `results` array for backwards compatibility. The
/// emitted payload (`to_columnar`) consults this to decide the shape.
/// `REFLEX_MCP_TIMING=1` adds per-phase `timings` to `search_code` / `search_regex`
/// responses. Off by default: it is a diagnostic, not part of the tool contract.
fn timing_enabled() -> bool {
    std::env::var("REFLEX_MCP_TIMING")
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        })
        .unwrap_or(false)
}

fn columnar_enabled() -> bool {
    match std::env::var("REFLEX_MCP_COLUMNAR") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true,
    }
}

/// REF-212: render a bool as a compact on/off token for the startup diagnostic.
fn onoff(v: bool) -> &'static str {
    if v { "on" } else { "off" }
}

/// REF-212: one-line startup diagnostic summarising the flags the MCP server
/// resolved from its environment, plus build provenance.
///
/// Emitted to **stderr** at startup (never stdout, which carries the JSON-RPC
/// stream). Claude Code captures an MCP server's stderr into its per-session
/// `mcp-logs-<server>/` files, so this line is the ground-truth record of which
/// `columnar` / structural behaviour a given trial actually ran with.
///
/// It exists specifically to catch the failure mode behind [REF-212]: an env
/// toggle that *was* forwarded to the process but was not honoured because the
/// running binary predated the code that reads it. The reported flags reflect
/// what THIS binary actually resolved, and `build=` names the commit it was
/// compiled from — so a benchmark run against an out-of-date rfx is obvious
/// rather than silently corrupting results.
///
/// REF-215 removed the `structuredContent` / `sc_stage2` flags along with the
/// env vars that drove them.
fn startup_flags_line(columnar: bool, structural: bool) -> String {
    format!(
        "reflex-mcp startup: version={} build={} columnar={} structural_tools={}",
        env!("CARGO_PKG_VERSION"),
        option_env!("REFLEX_GIT_SHA").unwrap_or("unknown"),
        onoff(columnar),
        onoff(structural),
    )
}

/// REF-209: reshape a search response's file-grouped `results` array into a
/// columnar `{ columns, rows }` pair to cut key-repetition token cost.
///
/// One row is emitted per match; `path`/`language` (and any file-level
/// `dependencies`) repeat per row so each row is self-contained and needs no
/// back-reference. Columns are emitted dynamically: the five always-present
/// fields (`path`, `language`, `start_line`, `end_line`, `preview`) plus any
/// optional field (`kind`, `symbol`, `context_before`, `context_after`,
/// `dependencies`) that at least one match/file actually carries — so the common
/// full-text case stays at five columns with no all-`null` padding.
///
/// Objects without a `results` array (count-mode `{count, pattern}`, error
/// shapes) are returned unchanged, so this is safe to call on any success value.
fn to_columnar(mut value: Value) -> Value {
    // Only transform success objects that carry a `results` array.
    if !value.get("results").map(Value::is_array).unwrap_or(false) {
        return value;
    }
    let obj = value
        .as_object_mut()
        .expect("value has a `results` member, so it is an object");
    let results = match obj.remove("results") {
        Some(Value::Array(results)) => results,
        // Unreachable given the guard above, but stay total rather than panic.
        other => {
            if let Some(other) = other {
                obj.insert("results".to_string(), other);
            }
            return value;
        }
    };

    // Pass 1: decide which optional columns any row needs.
    let mut has_kind = false;
    let mut has_symbol = false;
    let mut has_ctx_before = false;
    let mut has_ctx_after = false;
    let mut has_deps = false;
    for file in &results {
        if file.get("dependencies").is_some() {
            has_deps = true;
        }
        if let Some(matches) = file.get("matches").and_then(Value::as_array) {
            for m in matches {
                has_kind |= m.get("kind").is_some();
                has_symbol |= m.get("symbol").is_some();
                has_ctx_before |= m.get("context_before").is_some();
                has_ctx_after |= m.get("context_after").is_some();
            }
        }
    }

    // Fixed base columns; optional ones appended only when present, in a stable
    // order so `columns[i]` is deterministic for a given result set.
    let mut columns: Vec<&'static str> =
        vec!["path", "language", "start_line", "end_line", "preview"];
    if has_kind {
        columns.push("kind");
    }
    if has_symbol {
        columns.push("symbol");
    }
    if has_ctx_before {
        columns.push("context_before");
    }
    if has_ctx_after {
        columns.push("context_after");
    }
    if has_deps {
        columns.push("dependencies");
    }

    // Pass 2: project each match into a positional row aligned to `columns`.
    let mut rows: Vec<Value> = Vec::new();
    for file in &results {
        let path = file.get("path").cloned().unwrap_or(Value::Null);
        let language = file.get("language").cloned().unwrap_or(Value::Null);
        let deps = file.get("dependencies").cloned().unwrap_or(Value::Null);
        let Some(matches) = file.get("matches").and_then(Value::as_array) else {
            continue;
        };
        for m in matches {
            let span = m.get("span");
            let start_line = span
                .and_then(|s| s.get("start_line"))
                .cloned()
                .unwrap_or(Value::Null);
            let end_line = span
                .and_then(|s| s.get("end_line"))
                .cloned()
                .unwrap_or(Value::Null);
            let preview = m.get("preview").cloned().unwrap_or(Value::Null);

            let mut row: Vec<Value> = vec![
                path.clone(),
                language.clone(),
                start_line,
                end_line,
                preview,
            ];
            if has_kind {
                row.push(m.get("kind").cloned().unwrap_or(Value::Null));
            }
            if has_symbol {
                row.push(m.get("symbol").cloned().unwrap_or(Value::Null));
            }
            if has_ctx_before {
                row.push(m.get("context_before").cloned().unwrap_or(Value::Null));
            }
            if has_ctx_after {
                row.push(m.get("context_after").cloned().unwrap_or(Value::Null));
            }
            if has_deps {
                row.push(deps.clone());
            }
            rows.push(Value::Array(row));
        }
    }

    obj.insert("columns".to_string(), json!(columns));
    obj.insert("rows".to_string(), Value::Array(rows));
    value
}

/// Tools whose query engine updates the index itself, alongside the search.
const ENGINE_UPDATED_TOOLS: [&str; 6] = [
    "search_code",
    "search_regex",
    "list_locations",
    "count_occurrences",
    "find_references",
    "search_ast",
];

/// Tools that never update first: the explicit run, and the explicit probe (an
/// agent asking whether the index is current gets the answer, not a repair).
const NO_UPDATE_TOOLS: [&str; 2] = ["index_project", "check_index_status"];

/// Tools whose JSON-object answer carries `status` and `can_trust_results`, added
/// here when the tool's own answer lacks them (count mode, `list_locations`,
/// `find_references`, `analyze`). Array answers (`get_dependencies` in every form,
/// `search_ast`) have no place for them.
const FRESHNESS_FIELD_TOOLS: [&str; 11] = [
    "analyze",
    "search_code",
    "search_regex",
    "list_locations",
    "count_occurrences",
    "find_references",
    "find_hotspots",
    "find_circular",
    "find_unused",
    "find_islands",
    "analyze_summary",
];

/// Add the freshness verdict to a JSON-object answer that lacks
/// `can_trust_results` (memoised, so it is the verdict the call already saw).
fn add_freshness_fields(data: &mut Value, root: &Path) {
    let Some(obj) = data.as_object_mut() else {
        return;
    };
    if obj.contains_key("can_trust_results") {
        return;
    }
    let Ok((status, trusted, warning)) =
        QueryEngine::new(CacheManager::new(root)).get_index_status()
    else {
        return;
    };
    obj.insert("status".to_string(), json!(status));
    obj.insert("can_trust_results".to_string(), json!(trusted));
    if let Some(w) = warning
        && !obj.contains_key("warning")
        && let Ok(w) = serde_json::to_value(w)
    {
        obj.insert("warning".to_string(), w);
    }
}

/// The query engine a tool searches with: with this server's update, if any.
fn engine_in(cache: CacheManager, update: Option<UpdateOptions>) -> QueryEngine {
    match update {
        Some(opts) => QueryEngine::new(cache).with_update(opts),
        None => QueryEngine::new(cache),
    }
}

fn handle_call_tool(
    params: Option<Value>,
    root: &Path,
    update: Option<UpdateOptions>,
) -> Result<Value> {
    let params = params.ok_or_else(|| anyhow::anyhow!("Missing params for tools/call"))?;

    let name = params["name"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing tool name"))?;

    let spec = tool_spec(name).ok_or_else(|| anyhow::anyhow!("Unknown tool: {}", name))?;
    let (arguments, warnings) = normalize_arguments(name, spec, params["arguments"].clone())?;

    // Tools that read the index without the engine bring it up to date here.
    let mut warnings = warnings;
    if let Some(opts) = update
        && !ENGINE_UPDATED_TOOLS.contains(&name)
        && !NO_UPDATE_TOOLS.contains(&name)
        && let Updated::Skipped(reason) = update_if_stale(&CacheManager::new(root), &opts)
            .map_err(|e| anyhow::anyhow!("{}", mcp_facing_message(&e)))?
    {
        warnings.push(format!("Index not updated: {reason}"));
    }

    // Merged tools run the handler of the form they name; removed names still run.
    let handler = handler_for(name, &arguments)?;
    if let Some((_, instead)) = LEGACY_TOOLS.iter().find(|(old, _)| *old == name) {
        warnings.push(format!("`{name}` is deprecated; use {instead}"));
    }

    let mut data = with_corruption_recovery(name, root, || {
        dispatch_tool(handler, &arguments, root, update)
    })?;
    if FRESHNESS_FIELD_TOOLS.contains(&name) {
        add_freshness_fields(&mut data, root);
    }

    Ok(finish_tool_result(data, warnings))
}

/// Every `dispatch_tool` arm (listed tools and the ones merged into them).
const HANDLERS: &[&str] = &[
    "search_code",
    "search_regex",
    "list_locations",
    "count_occurrences",
    "find_references",
    "search_ast",
    "get_dependencies",
    "get_dependents",
    "get_transitive_deps",
    "find_hotspots",
    "find_circular",
    "find_unused",
    "find_islands",
    "analyze_summary",
    "gather_context",
    "index_project",
    "check_index_status",
];

/// The `dispatch_tool` arm a call runs: `analyze` by its `kind`, `get_dependencies`
/// by `reverse` / `depth`, everything else by its own name.
fn handler_for(name: &str, arguments: &Value) -> Result<&'static str> {
    Ok(match name {
        "analyze" => match arguments["kind"].as_str() {
            Some("summary") => "analyze_summary",
            Some("hotspots") => "find_hotspots",
            Some("circular") => "find_circular",
            Some("unused") => "find_unused",
            Some("islands") => "find_islands",
            other => {
                return Err(invalid_params(format!(
                    "analyze: kind must be one of summary, hotspots, circular, unused, islands (got {})",
                    other.map_or_else(|| "nothing".to_string(), |k| format!("\"{k}\""))
                )));
            }
        },
        "get_dependencies" => {
            let reverse = arguments["reverse"].as_bool().unwrap_or(false);
            match (reverse, arguments.get("depth").filter(|d| !d.is_null())) {
                (true, Some(_)) => {
                    return Err(invalid_params(
                        "get_dependencies: reverse and depth cannot be combined".to_string(),
                    ));
                }
                (true, None) => "get_dependents",
                (false, Some(_)) => "get_transitive_deps",
                (false, None) => "get_dependencies",
            }
        }
        other => HANDLERS
            .iter()
            .copied()
            .find(|h| *h == other)
            .ok_or_else(|| anyhow::anyhow!("Unknown tool: {}", other))?,
    })
}

/// Run one tool. Every arm returns the tool's *data* (a JSON object, or a
/// plain `Value::String` for prose tools such as `gather_context`);
/// `handle_call_tool` wraps it into the MCP `content` envelope.
fn dispatch_tool(
    name: &str,
    arguments: &Value,
    root: &Path,
    update: Option<UpdateOptions>,
) -> Result<Value> {
    match name {
        "list_locations" => {
            // Location discovery tool (minimal token usage)
            let pattern = arguments["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing pattern"))?
                .to_string();

            let lang = arguments["lang"].as_str().map(|s| s.to_string());
            // Substring mode. Default false = whole-identifier match (see `contains` in the schema).
            let contains = arguments["contains"].as_bool().unwrap_or(false);
            let ignore_case = arguments["ignore_case"].as_bool().unwrap_or(false);
            let include_locks = arguments["include_locks"].as_bool().unwrap_or(false);
            let include_generated = arguments["include_generated"].as_bool().unwrap_or(false);
            let file = arguments["file"].as_str().map(|s| s.to_string());
            let glob_patterns = arguments["glob"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let exclude_patterns = arguments["exclude"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let force = arguments["force"].as_bool().unwrap_or(false);
            let dependencies = arguments["dependencies"].as_bool().unwrap_or(false);
            let with_preview = arguments["preview"].as_bool().unwrap_or(false);

            let language = parse_language(lang);

            let filter = QueryFilter {
                language,
                kind: None,
                use_ast: false,
                use_regex: false,
                limit: None, // The tool contract is "one per match, no limit"
                symbols_mode: false,
                expand: false,
                file_pattern: file,
                exact: false,
                use_contains: contains,
                ignore_case,
                include_locks,
                include_generated,
                timeout_secs: 30,
                glob_patterns,
                exclude_patterns,
                // NOT paths_only. That mode collapses each file to its first match, so
                // the flat_map below yielded one entry per FILE while the tool
                // description promised one per MATCH (a 20-match pattern returned 8
                // entries). The response still serialises only {path, line}, so this
                // stays the cheapest tool despite returning every match.
                paths_only: false,
                offset: None,
                force,
                suppress_output: true, // MCP always returns JSON
                include_dependencies: dependencies,
                ..Default::default()
            };

            let cache = CacheManager::new(root);
            let engine = engine_in(cache, update);
            let response = engine.search_with_metadata(&pattern, filter.clone())?;

            // Extract locations (path + line) for each match
            let locations: Vec<serde_json::Value> = response
                .results
                .iter()
                .flat_map(|file_group| {
                    file_group.matches.iter().map(move |m| {
                        let mut loc = json!({
                            "path": file_group.path.clone(),
                            "line": m.span.start_line
                        });
                        // The matching line, trimmed and cut short: enough to tell a
                        // definition from a call without a second search (agents that
                        // could not see the line re-ran the search with grep).
                        if with_preview {
                            let line = m.preview.lines().next().unwrap_or("").trim();
                            let mut short: String =
                                line.chars().take(LOCATION_PREVIEW_CHARS).collect();
                            if line.chars().count() > LOCATION_PREVIEW_CHARS {
                                short.push('…');
                            }
                            loc["preview"] = json!(short);
                        }
                        loc
                    })
                })
                .collect();

            // Return compact response (just locations + count)
            let mut compact_response = json!({
                "status": response.status,
                "total_locations": locations.len(),
                "locations": locations
            });
            annotate_literal_result(&mut compact_response, &response);

            Ok(compact_response)
        }
        "count_occurrences" => {
            // Quick stats tool (minimal token usage)
            let pattern = arguments["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing pattern"))?
                .to_string();

            let lang = arguments["lang"].as_str().map(|s| s.to_string());
            // Substring mode. Default false = whole-identifier match (see `contains` in the schema).
            let contains = arguments["contains"].as_bool().unwrap_or(false);
            let ignore_case = arguments["ignore_case"].as_bool().unwrap_or(false);
            let include_locks = arguments["include_locks"].as_bool().unwrap_or(false);
            let include_generated = arguments["include_generated"].as_bool().unwrap_or(false);
            let kind = arguments["kind"].as_str().map(|s| s.to_string());
            let symbols = arguments["symbols"].as_bool();
            let file = arguments["file"].as_str().map(|s| s.to_string());
            let glob_patterns = arguments["glob"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let exclude_patterns = arguments["exclude"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let force = arguments["force"].as_bool().unwrap_or(false);
            let dependencies = arguments["dependencies"].as_bool().unwrap_or(false);

            let language = parse_language(lang);
            let parsed_kind = parse_symbol_kind(kind);
            let symbols_mode = symbols.unwrap_or(false) || parsed_kind.is_some();

            let filter = QueryFilter {
                language,
                kind: parsed_kind,
                use_ast: false,
                use_regex: false,
                limit: None, // No limit for counting
                count_only: true,
                symbols_mode,
                expand: false,
                file_pattern: file,
                exact: false,
                use_contains: contains,
                ignore_case,
                include_locks,
                include_generated,
                timeout_secs: 30,
                glob_patterns,
                exclude_patterns,
                paths_only: false, // Need to count all occurrences
                offset: None,
                force,
                suppress_output: true, // MCP always returns JSON
                include_dependencies: dependencies,
                ..Default::default()
            };

            let cache = CacheManager::new(root);
            let engine = engine_in(cache, update);
            let response = engine.search_with_metadata(&pattern, filter.clone())?;

            // Count unique files
            use std::collections::HashSet;
            let unique_files: HashSet<String> =
                response.results.iter().map(|fg| fg.path.clone()).collect();

            // Return minimal stats
            let mut stats = json!({
                "status": response.status,
                "pattern": pattern,
                "total": exact_total_or_count(&response),
                "files": response.file_count.unwrap_or(unique_files.len())
            });
            annotate_literal_result(&mut stats, &response);

            Ok(stats)
        }
        "search_code" => {
            let pattern = arguments["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing pattern"))?
                .to_string();

            let lang = arguments["lang"].as_str().map(|s| s.to_string());
            // Substring mode. Default false = whole-identifier match (see `contains` in the schema).
            let contains = arguments["contains"].as_bool().unwrap_or(false);
            let ignore_case = arguments["ignore_case"].as_bool().unwrap_or(false);
            let include_locks = arguments["include_locks"].as_bool().unwrap_or(false);
            let include_generated = arguments["include_generated"].as_bool().unwrap_or(false);
            let kind = arguments["kind"].as_str().map(|s| s.to_string());
            let symbols = arguments["symbols"].as_bool();
            let exact = arguments["exact"].as_bool();
            let file = arguments["file"].as_str().map(|s| s.to_string());
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let expand = arguments["expand"].as_bool();
            let glob_patterns: Vec<String> = arguments["glob"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let exclude_patterns = arguments["exclude"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let paths_only = arguments["paths"].as_bool().unwrap_or(false);
            let force = arguments["force"].as_bool().unwrap_or(false);
            let dependencies = arguments["dependencies"].as_bool().unwrap_or(false);
            let preview_length = arguments["preview_length"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(DEFAULT_MCP_PREVIEW_LENGTH);

            let language = parse_language(lang.clone());

            // Build warning for unsupported language + dependencies combination (REF-171)
            let deps_lang_warning: Option<String> =
                if dependencies && matches!(language, Some(l) if l != Language::Rust) {
                    Some(format!(
                        "Warning: dependencies is currently only supported for Rust files. \
                         No dependency data will be included for {} files.",
                        lang.as_deref().unwrap_or("non-Rust")
                    ))
                } else {
                    None
                };

            let parsed_kind = parse_symbol_kind(kind);
            let symbols_mode = symbols.unwrap_or(false) || parsed_kind.is_some();

            let offset = arguments["offset"].as_u64().map(|n| n as usize);

            // Smart limit handling:
            // 1. If --paths is set and user didn't specify limit: no limit (None)
            // 2. If user specified limit: use that value, capped at 500
            // 3. Otherwise: use the agent-oriented default (REF-191) so find-all
            //    tasks come back in one call instead of paginating.
            let final_limit = if paths_only && limit.is_none() {
                None // --paths without explicit limit means no limit
            } else if let Some(user_limit) = limit {
                Some(user_limit.min(500)) // Use user-specified limit, capped at 500
            } else {
                Some(DEFAULT_MCP_RESULT_LIMIT)
            };

            let mode = arguments["mode"].as_str().unwrap_or("list");

            // Count mode: run query but return only the total match count.
            // Skips preview truncation and full result serialization for speed.
            if mode == "count" {
                let count_filter = QueryFilter {
                    language,
                    kind: parsed_kind,
                    use_ast: false,
                    use_regex: false,
                    limit: None, // count everything
                    count_only: true,
                    symbols_mode,
                    expand: false,
                    file_pattern: file,
                    exact: exact.unwrap_or(false),
                    use_contains: contains,
                    ignore_case,
                    include_locks,
                    include_generated,
                    timeout_secs: 30,
                    glob_patterns,
                    exclude_patterns,
                    paths_only: false,
                    offset: None,
                    force,
                    suppress_output: true,
                    include_dependencies: false,
                    ..Default::default()
                };
                let cache = CacheManager::new(root);
                let engine = engine_in(cache, update);
                let response = engine.search_with_metadata(&pattern, count_filter.clone())?;
                let mut result =
                    json!({"count": exact_total_or_count(&response), "pattern": pattern});
                if let Some(files) = response.file_count {
                    result["files"] = json!(files);
                }
                annotate_literal_result(&mut result, &response);
                return Ok(result);
            }

            let filter = QueryFilter {
                language,
                kind: parsed_kind,
                use_ast: false,
                use_regex: false,
                limit: final_limit,
                symbols_mode,
                expand: expand.unwrap_or(false),
                file_pattern: file,
                exact: exact.unwrap_or(false),
                use_contains: contains,
                ignore_case,
                include_locks,
                include_generated,
                timeout_secs: 30, // Default 30 second timeout for MCP queries
                glob_patterns: glob_patterns.clone(),
                exclude_patterns,
                paths_only,
                offset,
                force,
                suppress_output: true, // MCP always returns JSON
                include_dependencies: dependencies,
                collect_timings: timing_enabled(),
                ..Default::default()
            };

            let cache = CacheManager::new(root);
            let engine = engine_in(cache, update);
            let mut response = engine.search_with_metadata(&pattern, filter.clone())?;

            if paths_only {
                return Ok(paths_only_result(&response));
            }

            // Apply preview truncation for token efficiency
            for file_group in response.results.iter_mut() {
                for m in file_group.matches.iter_mut() {
                    m.preview = crate::cli::truncate_preview(&m.preview, preview_length);
                }
            }

            // Calculate result count for AI instruction
            let result_count: usize = response.results.iter().map(|fg| fg.matches.len()).sum();

            // Generate AI instruction (MCP always uses AI mode)
            response.ai_instruction = crate::query::generate_ai_instruction(
                result_count,
                response.pagination.best_total(),
                response.pagination.has_more,
                symbols_mode,
                paths_only,
                false, // use_ast
                false, // use_regex
                language.is_some(),
                !glob_patterns.is_empty(),
                exact.unwrap_or(false),
            );

            // Prepend language limitation warning to AI instruction (REF-171)
            if let Some(warn) = deps_lang_warning {
                response.ai_instruction = Some(match response.ai_instruction.take() {
                    Some(existing) => format!("{warn}\n\n{existing}"),
                    None => warn,
                });
            }

            // Extract pagination scalars before consuming response (REF-185)
            let has_more = response.pagination.has_more;
            let total_count = response.pagination.total;
            let total_is_exact = response.pagination.total_is_exact;
            let approx_total = response.pagination.approx_total;
            let engine_warnings = response.warnings.clone();
            let engine_hint = response.hint.clone();
            let engine_reason = response.excluded_reason;
            let engine_excluded = response.excluded_by_default;

            let mut response_val = serde_json::to_value(response)?;
            if let serde_json::Value::Object(ref mut map) = response_val {
                map.insert("has_more".to_string(), json!(has_more));
                map.insert("total_count".to_string(), json!(total_count));
                map.insert("total_is_exact".to_string(), json!(total_is_exact));
                if let Some(approx) = approx_total {
                    map.insert("approx_total".to_string(), json!(approx));
                }
                map.insert("returned_count".to_string(), json!(result_count));
            }

            // REF-209: emit the token-efficient columnar shape by default; the
            // env toggle restores the legacy results[] array for compatibility.
            if columnar_enabled() {
                response_val = to_columnar(response_val);
            }

            // Applied after the columnar reshape so the hint survives both shapes.
            annotate_literal_fields(
                &mut response_val,
                &engine_warnings,
                engine_hint.as_deref(),
                engine_reason,
                engine_excluded,
            );

            Ok(response_val)
        }
        "search_regex" => {
            let pattern = arguments["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing pattern"))?
                .to_string();

            let lang = arguments["lang"].as_str().map(|s| s.to_string());
            let file = arguments["file"].as_str().map(|s| s.to_string());
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let glob_patterns: Vec<String> = arguments["glob"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let exclude_patterns = arguments["exclude"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let paths_only = arguments["paths"].as_bool().unwrap_or(false);
            let ignore_case = arguments["ignore_case"].as_bool().unwrap_or(false);
            let include_locks = arguments["include_locks"].as_bool().unwrap_or(false);
            let include_generated = arguments["include_generated"].as_bool().unwrap_or(false);
            let force = arguments["force"].as_bool().unwrap_or(false);
            let dependencies = arguments["dependencies"].as_bool().unwrap_or(false);

            let language = parse_language(lang);
            let offset = arguments["offset"].as_u64().map(|n| n as usize);

            // Smart limit handling (same as search_code)
            let final_limit = if paths_only && limit.is_none() {
                None // --paths without explicit limit means no limit
            } else if let Some(user_limit) = limit {
                Some(user_limit.min(500)) // Use user-specified limit, capped at 500
            } else {
                Some(DEFAULT_MCP_RESULT_LIMIT) // REF-191: one-call default
            };

            let mode = arguments["mode"].as_str().unwrap_or("list");

            // Count mode: return only the total match count, no match bodies.
            if mode == "count" {
                let count_filter = QueryFilter {
                    language,
                    kind: None,
                    use_ast: false,
                    use_regex: true,
                    limit: None, // count everything
                    count_only: true,
                    symbols_mode: false,
                    expand: false,
                    file_pattern: file,
                    exact: false,
                    use_contains: false,
                    ignore_case,
                    include_locks,
                    include_generated,
                    timeout_secs: 30,
                    glob_patterns,
                    exclude_patterns,
                    paths_only: false,
                    offset: None,
                    force,
                    suppress_output: true,
                    include_dependencies: false,
                    ..Default::default()
                };
                let cache = CacheManager::new(root);
                let engine = engine_in(cache, update);
                let response = engine.search_with_metadata(&pattern, count_filter)?;
                let mut result =
                    json!({"count": exact_total_or_count(&response), "pattern": pattern});
                if let Some(files) = response.file_count {
                    result["files"] = json!(files);
                }
                annotate_literal_result(&mut result, &response);
                return Ok(result);
            }

            let filter = QueryFilter {
                language,
                kind: None,
                use_ast: false,
                use_regex: true,
                limit: final_limit,
                symbols_mode: false,
                expand: false,
                file_pattern: file,
                exact: false,
                use_contains: false, // Regex mode uses substring matching via use_regex flag
                ignore_case,
                include_locks,
                include_generated,
                timeout_secs: 30, // Default 30 second timeout for MCP queries
                glob_patterns: glob_patterns.clone(),
                exclude_patterns,
                paths_only,
                offset,
                force,
                suppress_output: true, // MCP always returns JSON
                include_dependencies: dependencies,
                collect_timings: timing_enabled(),
                ..Default::default()
            };

            let cache = CacheManager::new(root);
            let engine = engine_in(cache, update);
            let mut response = engine.search_with_metadata(&pattern, filter)?;

            if paths_only {
                return Ok(paths_only_result(&response));
            }

            // Apply preview truncation for token efficiency
            for file_group in response.results.iter_mut() {
                for m in file_group.matches.iter_mut() {
                    m.preview =
                        crate::cli::truncate_preview(&m.preview, DEFAULT_MCP_PREVIEW_LENGTH);
                }
            }

            // Calculate result count for AI instruction
            let result_count: usize = response.results.iter().map(|fg| fg.matches.len()).sum();

            // Generate AI instruction (MCP always uses AI mode)
            response.ai_instruction = crate::query::generate_ai_instruction(
                result_count,
                response.pagination.best_total(),
                response.pagination.has_more,
                false, // symbols_mode
                paths_only,
                false, // use_ast
                true,  // use_regex
                language.is_some(),
                !glob_patterns.is_empty(),
                false, // exact
            );

            // Extract pagination scalars before consuming response (REF-185)
            let has_more = response.pagination.has_more;
            let total_count = response.pagination.total;
            let total_is_exact = response.pagination.total_is_exact;
            let approx_total = response.pagination.approx_total;
            let mut response_val = serde_json::to_value(response)?;
            if let serde_json::Value::Object(ref mut map) = response_val {
                map.insert("has_more".to_string(), json!(has_more));
                map.insert("total_count".to_string(), json!(total_count));
                map.insert("total_is_exact".to_string(), json!(total_is_exact));
                if let Some(approx) = approx_total {
                    map.insert("approx_total".to_string(), json!(approx));
                }
                map.insert("returned_count".to_string(), json!(result_count));
            }

            // REF-209: emit the token-efficient columnar shape by default; the
            // env toggle restores the legacy results[] array for compatibility.
            if columnar_enabled() {
                response_val = to_columnar(response_val);
            }

            Ok(response_val)
        }
        "search_ast" => {
            // AST pattern (Tree-sitter S-expression)
            let ast_pattern = arguments["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing pattern (AST S-expression)"))?
                .to_string();

            let lang_str = arguments["lang"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing lang (required for AST queries)"))?
                .to_string();

            let file = arguments["file"].as_str().map(|s| s.to_string());
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let glob_patterns: Vec<String> = arguments["glob"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let exclude_patterns: Vec<String> = arguments["exclude"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let paths_only = arguments["paths"].as_bool().unwrap_or(false);
            let force = arguments["force"].as_bool().unwrap_or(false);
            let dependencies = arguments["dependencies"].as_bool().unwrap_or(false);

            let language = parse_language(Some(lang_str.clone())).ok_or_else(|| {
                anyhow::anyhow!("Invalid or unsupported language for AST queries")
            })?;

            // Reject the text tier explicitly rather than letting it fall through to
            // the grammar loader's generic error. There is no grammar, by design.
            if language.is_text() || language.is_excluded_by_default() {
                anyhow::bail!(
                    "lang \"{}\" is the plain-text tier (docs, config, templates, lock and \
                     generated files). These files are trigram-indexed only and have no AST. \
                     Use search_code or search_regex on them instead.",
                    lang_str
                );
            }

            // Warn if glob patterns are not provided (performance issue)
            if glob_patterns.is_empty() && exclude_patterns.is_empty() {
                log::warn!(
                    "⚠️  AST query without glob patterns will scan the ENTIRE codebase. This may take 2-10+ seconds."
                );
                log::warn!(
                    "    Strongly recommend using glob patterns, e.g., glob=['src/**/*.rs']"
                );
            }

            let offset = arguments["offset"].as_u64().map(|n| n as usize);

            // Smart limit handling (same as search_code)
            let final_limit = if paths_only && limit.is_none() {
                None // --paths without explicit limit means no limit
            } else if let Some(user_limit) = limit {
                Some(user_limit) // Use user-specified limit
            } else {
                Some(100) // Default: limit to 100 results for token efficiency
            };

            let filter = QueryFilter {
                language: Some(language),
                kind: None,
                use_ast: true,
                use_regex: false,
                limit: final_limit,
                symbols_mode: false,
                expand: false,
                file_pattern: file,
                exact: false,
                use_contains: false,
                timeout_secs: 60, // Longer timeout for AST queries (they're slow)
                glob_patterns,
                exclude_patterns,
                paths_only,
                offset,
                force,
                suppress_output: true, // MCP always returns JSON
                include_dependencies: dependencies,
                ..Default::default()
            };

            let cache = CacheManager::new(root);
            let engine = engine_in(cache, update);

            // Use the new search_ast_all_files method (no trigram filtering)
            let mut results = engine.search_ast_all_files(&ast_pattern, filter)?;

            // Apply preview truncation for token efficiency
            for result in &mut results {
                result.preview =
                    crate::cli::truncate_preview(&result.preview, DEFAULT_MCP_PREVIEW_LENGTH);
            }

            Ok(serde_json::to_value(&results)?)
        }
        "index_project" => {
            let force = arguments["force"].as_bool().unwrap_or(false);
            let lang_filters: Vec<Language> = arguments["languages"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .filter_map(|s| parse_language(Some(s.to_string())))
                        .collect()
                })
                .unwrap_or_default();

            let stats = rebuild_index(root, force, lang_filters)?;

            Ok(serde_json::to_value(&stats)?)
        }
        "get_dependencies" => {
            let path = arguments["path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing path"))?
                .to_string();

            let cache = CacheManager::new(root);
            let deps_index = DependencyIndex::new(cache);

            // Fuzzy path matching
            let file_id = deps_index
                .get_file_id_by_path(&path)?
                .ok_or_else(|| anyhow::anyhow!("File '{}' not found in index", path))?;

            let dependencies = deps_index.get_dependencies_info(file_id)?;

            Ok(serde_json::to_value(&dependencies)?)
        }
        "get_dependents" => {
            let path = arguments["path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing path"))?
                .to_string();

            let cache = CacheManager::new(root);
            let deps_index = DependencyIndex::new(cache);

            // Fuzzy path matching
            let file_id = deps_index
                .get_file_id_by_path(&path)?
                .ok_or_else(|| anyhow::anyhow!("File '{}' not found in index", path))?;

            let dependents = deps_index.get_dependents(file_id)?;
            let paths = deps_index.get_file_paths(&dependents)?;

            // Convert to array of paths
            let path_list: Vec<String> = dependents
                .iter()
                .filter_map(|id| paths.get(id).cloned())
                .collect();

            Ok(serde_json::to_value(&path_list)?)
        }
        "get_transitive_deps" => {
            let path = arguments["path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing path"))?
                .to_string();

            let depth = arguments["depth"].as_u64().map(|n| n as usize).unwrap_or(3); // Default depth of 3

            let cache = CacheManager::new(root);
            let deps_index = DependencyIndex::new(cache);

            // Fuzzy path matching
            let file_id = deps_index
                .get_file_id_by_path(&path)?
                .ok_or_else(|| anyhow::anyhow!("File '{}' not found in index", path))?;

            let transitive = deps_index.get_transitive_deps(file_id, depth)?;

            // Get paths for all file IDs
            let file_ids: Vec<i64> = transitive.keys().copied().collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            // Build result with path → depth mapping
            let result: Vec<serde_json::Value> = transitive
                .iter()
                .filter_map(|(id, depth)| {
                    paths.get(id).map(|path| {
                        json!({
                            "path": path,
                            "depth": depth
                        })
                    })
                })
                .collect();

            Ok(serde_json::to_value(&result)?)
        }
        "find_hotspots" => {
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let offset = arguments["offset"].as_u64().map(|n| n as usize);
            let min_dependents = arguments["min_dependents"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(2);
            let sort = arguments["sort"].as_str().map(|s| s.to_string());

            let cache = CacheManager::new(root);
            let deps_index = DependencyIndex::new(cache);

            // Get all hotspots first (without limit) to track total count
            let mut all_hotspots = deps_index.find_hotspots(None, min_dependents)?;

            // Apply sorting (default: descending - most imports first)
            let sort_order = sort.as_deref().unwrap_or("desc");
            match sort_order {
                "asc" => {
                    // Ascending: least imports first
                    all_hotspots.sort_by_key(|a| a.1);
                }
                "desc" => {
                    // Descending: most imports first (default)
                    all_hotspots.sort_by_key(|a| std::cmp::Reverse(a.1));
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Invalid sort order '{}'. Supported: asc, desc",
                        sort_order
                    ));
                }
            }

            let total_count = all_hotspots.len();

            // Apply offset pagination
            let offset_val = offset.unwrap_or(0);
            let mut hotspots: Vec<_> = all_hotspots.into_iter().skip(offset_val).collect();

            // Apply limit (default 200)
            let limit_val = limit.unwrap_or(200);
            hotspots.truncate(limit_val);

            let count = hotspots.len();
            let has_more = offset_val + count < total_count;

            // Get paths for all file IDs
            let file_ids: Vec<i64> = hotspots.iter().map(|(id, _)| *id).collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            // Build result with path + import_count (no file_id)
            let results: Vec<serde_json::Value> = hotspots
                .iter()
                .filter_map(|(id, import_count)| {
                    paths.get(id).map(|path| {
                        json!({
                            "path": path,
                            "import_count": import_count,
                        })
                    })
                })
                .collect();

            let response = json!({
                "pagination": {
                    "total": total_count,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit_val,
                    "has_more": has_more,
                },
                "results": results,
            });

            Ok(response)
        }
        "find_circular" => {
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let offset = arguments["offset"].as_u64().map(|n| n as usize);
            let sort = arguments["sort"].as_str().map(|s| s.to_string());

            let cache = CacheManager::new(root);
            let deps_index = DependencyIndex::new(cache);

            let mut all_cycles = deps_index.detect_circular_dependencies()?;

            // Apply sorting (default: descending - longest cycles first)
            let sort_order = sort.as_deref().unwrap_or("desc");
            match sort_order {
                "asc" => {
                    // Ascending: shortest cycles first
                    all_cycles.sort_by_key(|cycle| cycle.len());
                }
                "desc" => {
                    // Descending: longest cycles first (default)
                    all_cycles.sort_by_key(|cycle| std::cmp::Reverse(cycle.len()));
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Invalid sort order '{}'. Supported: asc, desc",
                        sort_order
                    ));
                }
            }

            let total_count = all_cycles.len();

            // Apply offset pagination
            let offset_val = offset.unwrap_or(0);
            let mut cycles: Vec<_> = all_cycles.into_iter().skip(offset_val).collect();

            // Apply limit (default 200)
            let limit_val = limit.unwrap_or(200);
            cycles.truncate(limit_val);

            let count = cycles.len();
            let has_more = offset_val + count < total_count;

            // Convert cycles to paths (without file_ids)
            let file_ids: Vec<i64> = cycles.iter().flat_map(|c| c.iter()).copied().collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            let results: Vec<serde_json::Value> = cycles
                .iter()
                .map(|cycle| {
                    let cycle_paths: Vec<_> = cycle
                        .iter()
                        .filter_map(|id| paths.get(id).cloned())
                        .collect();
                    json!({
                        "paths": cycle_paths,
                    })
                })
                .collect();

            let response = json!({
                "pagination": {
                    "total": total_count,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit_val,
                    "has_more": has_more,
                },
                "results": results,
            });

            Ok(response)
        }
        "find_unused" => {
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let offset = arguments["offset"].as_u64().map(|n| n as usize);

            let cache = CacheManager::new(root);
            let deps_index = DependencyIndex::new(cache);

            let all_unused = deps_index.find_unused_files()?;
            let total_count = all_unused.len();

            // Apply offset pagination
            let offset_val = offset.unwrap_or(0);
            let mut unused: Vec<_> = all_unused.into_iter().skip(offset_val).collect();

            // Apply limit (default 200)
            let limit_val = limit.unwrap_or(200);
            unused.truncate(limit_val);

            let count = unused.len();
            let has_more = offset_val + count < total_count;

            // Get paths for all unused file IDs
            let paths = deps_index.get_file_paths(&unused)?;

            // Build result (flat array of path strings)
            let results: Vec<String> = unused
                .iter()
                .filter_map(|id| paths.get(id).cloned())
                .collect();

            let response = json!({
                "pagination": {
                    "total": total_count,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit_val,
                    "has_more": has_more,
                },
                "results": results,
            });

            Ok(response)
        }
        "find_islands" => {
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let offset = arguments["offset"].as_u64().map(|n| n as usize);
            let min_island_size = arguments["min_island_size"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(2);
            let max_island_size = arguments["max_island_size"].as_u64().map(|n| n as usize);
            let sort = arguments["sort"].as_str().map(|s| s.to_string());

            let cache = CacheManager::new(root);
            let deps_index = DependencyIndex::new(cache);

            let all_islands = deps_index.find_islands()?;
            let total_components = all_islands.len();

            // Get total file count for percentage calculation
            let total_files = deps_index.get_cache().stats()?.total_files;

            // Calculate max_island_size default: min of 500 or 50% of total files
            let max_size = max_island_size.unwrap_or_else(|| {
                let fifty_percent = (total_files as f64 * 0.5) as usize;
                fifty_percent.min(500)
            });

            // Filter islands by size
            let mut islands: Vec<_> = all_islands
                .into_iter()
                .filter(|island| {
                    let size = island.len();
                    size >= min_island_size && size <= max_size
                })
                .collect();

            // Apply sorting (default: descending - largest islands first)
            let sort_order = sort.as_deref().unwrap_or("desc");
            match sort_order {
                "asc" => {
                    // Ascending: smallest islands first
                    islands.sort_by_key(|island| island.len());
                }
                "desc" => {
                    // Descending: largest islands first (default)
                    islands.sort_by_key(|island| std::cmp::Reverse(island.len()));
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Invalid sort order '{}'. Supported: asc, desc",
                        sort_order
                    ));
                }
            }

            let _filtered_count = total_components - islands.len();
            let total_after_filter = islands.len();

            // Apply offset pagination
            let offset_val = offset.unwrap_or(0);
            if offset_val > 0 && offset_val < islands.len() {
                islands = islands.into_iter().skip(offset_val).collect();
            } else if offset_val >= islands.len() {
                islands.clear();
            }

            // Apply limit (default 200)
            let limit_val = limit.unwrap_or(200);
            islands.truncate(limit_val);

            let count = islands.len();
            let has_more = offset_val + count < total_after_filter;

            // Get all file IDs from all islands
            let file_ids: Vec<i64> = islands
                .iter()
                .flat_map(|island| island.iter())
                .copied()
                .collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            // Build result (array of islands with paths, no file_ids)
            let results: Vec<serde_json::Value> = islands
                .iter()
                .enumerate()
                .map(|(idx, island)| {
                    let island_paths: Vec<_> = island
                        .iter()
                        .filter_map(|id| paths.get(id).cloned())
                        .collect();
                    json!({
                        "island_id": idx + 1,
                        "size": island.len(),
                        "paths": island_paths,
                    })
                })
                .collect();

            let response = json!({
                "pagination": {
                    "total": total_after_filter,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit_val,
                    "has_more": has_more,
                },
                "results": results,
            });

            Ok(response)
        }
        "analyze_summary" => {
            let min_dependents = arguments["min_dependents"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(2);

            let cache = CacheManager::new(root);
            let deps_index = DependencyIndex::new(cache);

            let cycles = deps_index.detect_circular_dependencies()?;
            let hotspots = deps_index.find_hotspots(None, min_dependents)?;
            let unused = deps_index.find_unused_files()?;
            let all_islands = deps_index.find_islands()?;

            let summary = json!({
                "circular_dependencies": cycles.len(),
                "hotspots": hotspots.len(),
                "unused_files": unused.len(),
                "islands": all_islands.len(),
                "min_dependents": min_dependents,
            });

            Ok(summary)
        }
        "gather_context" => {
            // Parse optional parameters
            let structure = arguments["structure"].as_bool().unwrap_or(false);
            let file_types = arguments["file_types"].as_bool().unwrap_or(false);
            let project_type = arguments["project_type"].as_bool().unwrap_or(false);
            let framework = arguments["framework"].as_bool().unwrap_or(false);
            let entry_points = arguments["entry_points"].as_bool().unwrap_or(false);
            let test_layout = arguments["test_layout"].as_bool().unwrap_or(false);
            let config_files = arguments["config_files"].as_bool().unwrap_or(false);
            let depth = arguments["depth"].as_u64().map(|n| n as usize).unwrap_or(2);
            let path = arguments["path"].as_str().map(|s| s.to_string());

            // Build context options
            let mut opts = crate::context::ContextOptions {
                structure,
                path,
                file_types,
                project_type,
                framework,
                entry_points,
                test_layout,
                config_files,
                depth,
                json: false, // MCP always returns text format
            };

            // If no context flags specified, return minimal orientation context only.
            // Requesting all types by default floods agent context windows (2000-5000 tokens).
            let no_flags_set = opts.is_empty();
            if no_flags_set {
                opts.project_type = true;
                opts.entry_points = true;
            }

            let cache = CacheManager::new(root);
            let context = crate::context::generate_context(&cache, &opts)?;

            let hint = if no_flags_set {
                "\n\n---\nHint: this is the minimal orientation view. Pass any combination of these flags for more detail: structure, file_types, framework, test_layout, config_files."
            } else {
                ""
            };

            // gather_context returns human-readable prose, not JSON. Running it
            // through make_tool_result would JSON-encode (quote + escape) the text
            // and change content[text]; instead keep content[text] as the raw
            // string (REF-215: content[text] only, no structuredContent).
            let context_text = format!("{}{}", context, hint);
            Ok(Value::String(context_text))
        }
        "check_index_status" => {
            let cache = CacheManager::new(root);

            if !cache.exists() {
                let result = json!({
                    "status": "missing",
                    "action_required": "index_project"
                });
                return Ok(result);
            }

            let engine = engine_in(cache, update);
            // This is the explicit probe. An agent that asks whether the index is
            // current must never be answered from the freshness memo.
            let report = engine.fresh_index_report()?;

            let status_str = match report.status {
                IndexStatus::Fresh => "fresh",
                IndexStatus::Stale => "stale",
            };

            let mut result = if let Some(w) = report.warning {
                let mut obj = json!({
                    "status": status_str,
                    "can_trust_results": report.can_trust_results,
                    "reason": w.reason,
                    // Name the MCP tool, not the CLI. An agent cannot run `rfx index`.
                    "action_required": w.action_required
                });
                for (key, list) in [
                    ("files_modified", w.files_modified),
                    ("files_added", w.files_added),
                    ("files_deleted", w.files_deleted),
                ] {
                    if let Some(paths) = list {
                        obj[key] = json!(paths);
                    }
                }
                if let Some(n) = w.changed_count {
                    obj["changed_count"] = json!(n);
                }
                if w.truncated {
                    obj["truncated"] = json!(true);
                }
                obj
            } else {
                json!({ "status": status_str, "can_trust_results": report.can_trust_results })
            };
            // Branch and commit context for humans. Present even when fresh: since
            // 2.0.0 the indexed commit differing from HEAD is not staleness.
            if let Some(details) = report.details {
                result["details"] = serde_json::to_value(details)?;
            }

            Ok(result)
        }
        "find_references" => {
            let pattern = arguments["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing pattern"))?
                .to_string();

            let lang = arguments["lang"].as_str().map(|s| s.to_string());
            // Substring mode. Default false = whole-identifier match (see `contains` in the schema).
            let contains = arguments["contains"].as_bool().unwrap_or(false);
            let ignore_case = arguments["ignore_case"].as_bool().unwrap_or(false);
            let kind = arguments["kind"].as_str().map(|s| s.to_string());
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let offset = arguments["offset"].as_u64().map(|n| n as usize);
            let glob_patterns: Vec<String> = arguments["glob"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let exclude_patterns: Vec<String> = arguments["exclude"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let force = arguments["force"].as_bool().unwrap_or(false);
            let include_strings = arguments["include_strings"].as_bool().unwrap_or(false);

            let language = parse_language(lang);
            let parsed_kind = parse_symbol_kind(kind);

            let mode = arguments["mode"].as_str().unwrap_or("list");

            // Count mode: skip the definition lookup and just count textual references.
            if mode == "count" {
                let count_filter = QueryFilter {
                    language,
                    kind: None,
                    use_ast: false,
                    use_regex: false,
                    limit: None, // count everything
                    // The count-only fast path returns totals and NO rows. The
                    // string/comment filter below needs rows, so it may only run
                    // when that filter is off; otherwise every count was 0.
                    count_only: include_strings,
                    symbols_mode: false,
                    expand: false,
                    file_pattern: None,
                    exact: false,
                    use_contains: contains,
                    ignore_case,
                    // Code only. The reference search is a plain trigram scan, and
                    // `is_in_string_or_comment` returns false for the text tier (no line
                    // filter exists for markdown), so a mention in a changelog would
                    // otherwise be reported as a call site.
                    exclude_text: true,
                    timeout_secs: 30,
                    glob_patterns,
                    exclude_patterns,
                    paths_only: false,
                    offset: None,
                    force,
                    suppress_output: true,
                    include_dependencies: false,
                    ..Default::default()
                };
                let cache = CacheManager::new(root);
                let engine = engine_in(cache, update);
                let response = engine.search_with_metadata(&pattern, count_filter.clone())?;

                // `include_strings` must mean the same thing in count mode as in list
                // mode. It previously returned `pagination.total`, which is the raw
                // engine total, so include_strings:true and :false gave the SAME number
                // even when many hits were in string literals and doc comments.
                let count = if include_strings {
                    exact_total_or_count(&response)
                } else {
                    let pat = pattern.as_str();
                    response
                        .results
                        .iter()
                        .flat_map(|fg| {
                            fg.matches.iter().filter(move |m| {
                                !is_in_string_or_comment(fg.language, &m.preview, pat)
                            })
                        })
                        .count()
                };

                let mut result = json!({"count": count, "pattern": pattern});
                annotate_literal_result(&mut result, &response);
                return Ok(result);
            }

            // Search 1: Find symbol definition (symbols_mode=true, small cap)
            let def_filter = QueryFilter {
                language,
                kind: parsed_kind,
                use_ast: false,
                use_regex: false,
                limit: Some(5),
                symbols_mode: true,
                expand: false,
                file_pattern: None,
                exact: false,
                use_contains: contains,
                ignore_case,
                // Code only. The reference search is a plain trigram scan, and
                // `is_in_string_or_comment` returns false for the text tier (no line
                // filter exists for markdown), so a mention in a changelog would
                // otherwise be reported as a call site.
                exclude_text: true,
                timeout_secs: 30,
                glob_patterns: glob_patterns.clone(),
                exclude_patterns: exclude_patterns.clone(),
                paths_only: false,
                offset: None,
                force,
                suppress_output: true,
                include_dependencies: false,
                ..Default::default()
            };

            let cache = CacheManager::new(root);
            let engine = engine_in(cache, update);
            let def_response = engine.search_with_metadata(&pattern, def_filter)?;

            // Extract first definition as a compact object (reuse MatchResult's Serialize impl)
            let definition: Option<serde_json::Value> =
                def_response.results.first().and_then(|fg| {
                    fg.matches.first().map(|m| {
                        let mut def_obj = serde_json::to_value(m).unwrap_or(json!({}));
                        if let serde_json::Value::Object(ref mut map) = def_obj {
                            map.insert("path".to_string(), json!(fg.path.clone()));
                            // Truncate preview if present
                            if let Some(preview) = map.get("preview").and_then(|v| v.as_str()) {
                                let truncated = crate::cli::truncate_preview(
                                    preview,
                                    DEFAULT_MCP_PREVIEW_LENGTH,
                                );
                                map.insert("preview".to_string(), json!(truncated));
                            }
                        }
                        def_obj
                    })
                });

            // Search 2: Find all textual references (symbols_mode=false)
            let ref_filter = QueryFilter {
                language,
                kind: None,
                use_ast: false,
                use_regex: false,
                // REF-191: default to the one-call page size so "find all callers"
                // returns the full set instead of paginating at 50.
                limit: limit.map(|l| l.min(500)).or(Some(DEFAULT_MCP_RESULT_LIMIT)),
                symbols_mode: false,
                expand: false,
                file_pattern: None,
                exact: false,
                use_contains: contains,
                ignore_case,
                // Code only. The reference search is a plain trigram scan, and
                // `is_in_string_or_comment` returns false for the text tier (no line
                // filter exists for markdown), so a mention in a changelog would
                // otherwise be reported as a call site.
                exclude_text: true,
                timeout_secs: 30,
                glob_patterns,
                exclude_patterns,
                paths_only: false,
                offset,
                force,
                suppress_output: true,
                include_dependencies: false,
                // `total_references` is documented as the full count of call sites,
                // so this search verifies every candidate even though it pages.
                require_exact_total: true,
                ..Default::default()
            };

            let ref_response = engine.search_with_metadata(&pattern, ref_filter)?;

            // Flatten references to compact {path, line, preview} array,
            // excluding matches inside string literals or comments (unless include_strings).
            let references: Vec<serde_json::Value> = ref_response.results.iter()
                .flat_map(|fg| {
                    fg.matches.iter()
                        .filter(|m| {
                            include_strings
                                || !is_in_string_or_comment(fg.language, &m.preview, &pattern)
                        })
                        .map(move |m| {
                            json!({
                                "path": fg.path,
                                "line": m.span.start_line,
                                "preview": crate::cli::truncate_preview(&m.preview, DEFAULT_MCP_PREVIEW_LENGTH)
                            })
                        })
                })
                .collect();

            // Count consistency. These were three different numbers reported under
            // names that all read like "total", which looked like an off-by-one:
            // pagination.total 25 / total_references 24 / returned_count 24.
            //
            // One meaning each, and the gap is now named instead of implied:
            //   pagination.total  — raw engine total, BEFORE string/comment filtering.
            //                       This is the space `offset` indexes into, so it must
            //                       stay pre-filter or pagination breaks.
            //   returned_count    — references actually in this page, after filtering.
            //   filtered_out      — how many this page dropped. The missing number.
            //   total_references  — kept as an alias of pagination.total for callers
            //                       that already read it.
            let returned_count = references.len();
            let page_matches: usize = ref_response.results.iter().map(|fg| fg.matches.len()).sum();
            let filtered_out = page_matches.saturating_sub(returned_count);
            let total_references = ref_response.pagination.total;
            let has_more = ref_response.pagination.has_more;
            let engine_warnings = ref_response.warnings.clone();
            let engine_hint = ref_response.hint.clone();
            let engine_reason = ref_response.excluded_reason;
            let engine_excluded = ref_response.excluded_by_default;

            let mut response = json!({
                "status": ref_response.status,
                "definition": definition,
                "references": references,
                "total_references": total_references,
                "total_count": total_references,
                "returned_count": returned_count,
                "filtered_out": filtered_out,
                "has_more": has_more,
                "pagination": ref_response.pagination,
            });
            annotate_literal_fields(
                &mut response,
                &engine_warnings,
                engine_hint.as_deref(),
                engine_reason,
                engine_excluded,
            );

            Ok(response)
        }
        _ => Err(anyhow::anyhow!("Unknown tool: {}", name)),
    }
}

/// Process a single JSON-RPC request
fn process_request(
    request: JsonRpcRequest,
    enable_structural: bool,
    root: &Path,
    update: Option<UpdateOptions>,
) -> JsonRpcResponse {
    log::debug!("MCP request: method={}", request.method);

    let result = match request.method.as_str() {
        "initialize" => handle_initialize(request.params),
        "tools/list" => handle_list_tools(request.params, enable_structural),
        "tools/call" => handle_call_tool(request.params, root, update),
        _ => Err(anyhow::anyhow!("Unknown method: {}", request.method)),
    };

    match result {
        Ok(value) => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: Some(value),
            error: None,
        },
        Err(e) => {
            log::error!("MCP error: {}", e);
            let msg = e.to_string();
            // REF-67: map to the correct JSON-RPC error code instead of always using -32603.
            let (code, kind, message) =
                if let Some(re) = e.downcast_ref::<crate::errors::ReflexError>() {
                    let code = match re {
                        crate::errors::ReflexError::QuerySyntaxError(_)
                        | crate::errors::ReflexError::InvalidParams(_) => -32602, // Invalid params
                        _ => -32603, // Internal error
                    };
                    // Agents cannot run the CLI: point them at index_project instead.
                    (code, re.kind().to_string(), mcp_facing_message(&e))
                } else if msg.starts_with("Unknown method:") {
                    (-32601, "MethodNotFound".to_string(), msg)
                } else if msg.starts_with("Missing")
                    || msg.starts_with("Unknown tool:")
                    || msg.starts_with("Invalid or unsupported")
                {
                    (-32602, "InvalidParams".to_string(), msg)
                } else {
                    (-32603, "InternalError".to_string(), msg)
                };
            JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(JsonRpcError {
                    code,
                    message: message.clone(),
                    data: Some(json!({ "kind": kind })),
                }),
            }
        }
    }
}

/// Handle a JSON-RPC Notification. Notifications never receive a response.
fn handle_notification(method: &str, _params: Option<Value>) {
    match method {
        "notifications/initialized" | "notifications/cancelled" => {
            log::debug!("MCP notification: {}", method);
        }
        other => {
            log::debug!("MCP unknown notification: {}", other);
        }
    }
}

/// Run the MCP server on stdio.
/// `rfx mcp`: every tool that reads the index updates a stale one first, unless
/// `no_update`.
pub fn run_mcp_server(no_update: bool) -> Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mcp_config = load_mcp_config();
    run_mcp_server_io_impl(
        stdin.lock(),
        stdout.lock(),
        mcp_config.enable_structural_tools,
        Path::new("."),
        (!no_update).then(UpdateOptions::server),
    )
}

/// Run the MCP server reading JSON-RPC messages from `reader` and writing
/// responses to `writer`. Exposed at crate-level for integration tests.
pub fn run_mcp_server_io<R: BufRead, W: Write>(reader: R, writer: W) -> Result<()> {
    let mcp_config = load_mcp_config();
    run_mcp_server_io_impl(
        reader,
        writer,
        mcp_config.enable_structural_tools,
        Path::new("."),
        None,
    )
}

/// Like [`run_mcp_server_io`] but every tool operates on `root` instead of the
/// process working directory. Exposed for integration tests so they can point
/// the server at a temp workspace without `set_current_dir` (process-global,
/// unsafe with parallel tests).
pub fn run_mcp_server_io_in<R: BufRead, W: Write>(
    root: &Path,
    reader: R,
    writer: W,
    enable_structural: bool,
) -> Result<()> {
    run_mcp_server_io_impl(reader, writer, enable_structural, root, None)
}

/// Like [`run_mcp_server_io_in`], with the automatic update `rfx mcp` runs
/// (`update`; use [`UpdateOptions::library`] in tests: no process is started).
pub fn run_mcp_server_io_with<R: BufRead, W: Write>(
    root: &Path,
    reader: R,
    writer: W,
    enable_structural: bool,
    update: Option<UpdateOptions>,
) -> Result<()> {
    run_mcp_server_io_impl(reader, writer, enable_structural, root, update)
}

/// Inner server loop. Accepts `enable_structural` so tests can drive the flag
/// without touching the filesystem, and `root` so tools resolve `.reflex/`
/// relative to a chosen workspace.
fn run_mcp_server_io_impl<R: BufRead, W: Write>(
    reader: R,
    mut writer: W,
    enable_structural: bool,
    root: &Path,
    update: Option<UpdateOptions>,
) -> Result<()> {
    log::info!("Starting Reflex MCP server on stdio");

    // REF-212: unconditional stderr diagnostic (NOT gated behind RUST_LOG) so the
    // resolved runtime flags are always captured in Claude Code's mcp-logs. This
    // is the verification hook for efficacy trials: it proves which behaviour the
    // running binary actually honoured and pins the exact build it came from.
    eprintln!(
        "{}",
        startup_flags_line(columnar_enabled(), enable_structural)
    );

    for line in reader.lines() {
        let line = line?;

        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        log::debug!("MCP input: {}", line);

        // Parse JSON-RPC message
        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(req) => req,
            Err(e) => {
                log::error!("Failed to parse JSON-RPC request: {}", e);
                // REF-61: send -32700 Parse error instead of silently dropping.
                // Per JSON-RPC 2.0, use null id when the id cannot be determined.
                let error_response = JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: Some(Value::Null),
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("Parse error: {}", e),
                        data: None,
                    }),
                };
                let response_json = serde_json::to_string(&error_response)?;
                writeln!(writer, "{}", response_json)?;
                writer.flush()?;
                continue;
            }
        };

        // Notifications (no `id`) must not receive a response per JSON-RPC 2.0.
        if request.id.is_none() {
            handle_notification(&request.method, request.params);
            continue;
        }

        // Process request and write response
        let response = process_request(request, enable_structural, root, update);
        let response_json = serde_json::to_string(&response)?;
        writeln!(writer, "{}", response_json)?;
        writer.flush()?;

        log::debug!("MCP output: {}", response_json);
    }

    log::info!("Reflex MCP server stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn call_server(input: &str) -> String {
        call_server_with_structural(input, true)
    }

    fn call_server_with_structural(input: &str, enable_structural: bool) -> String {
        let reader = Cursor::new(input.as_bytes());
        let mut output = Vec::new();
        run_mcp_server_io_impl(reader, &mut output, enable_structural, Path::new("."), None)
            .unwrap();
        String::from_utf8(output).unwrap()
    }

    fn parse_first_response(raw: &str) -> serde_json::Value {
        let line = raw.lines().next().expect("no response line");
        serde_json::from_str(line).expect("invalid JSON response")
    }

    // REF-61: malformed JSON must return -32700 Parse error (not silence)
    #[test]
    fn test_parse_error_returns_32700() {
        let raw = call_server("not-valid-json\n");
        let resp = parse_first_response(&raw);
        assert_eq!(resp["jsonrpc"], "2.0");
        assert_eq!(resp["error"]["code"], -32700);
        // Per JSON-RPC 2.0, id must be null when id cannot be determined
        assert!(resp["id"].is_null(), "id must be null for parse errors");
    }

    // REF-67: unknown method returns -32601 (Method not found)
    #[test]
    fn test_unknown_method_returns_32601() {
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"no_such_method","params":null}"#;
        let raw = call_server(&format!("{}\n", req));
        let resp = parse_first_response(&raw);
        assert_eq!(resp["error"]["code"], -32601);
    }

    // REF-67: missing required param returns -32602 (Invalid params), not -32603
    #[test]
    fn test_missing_param_returns_32602() {
        // search_code requires "pattern" — omit it
        let req = r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"search_code","arguments":{}}}"#;
        let raw = call_server(&format!("{}\n", req));
        let resp = parse_first_response(&raw);
        assert_eq!(resp["error"]["code"], -32602);
    }

    // Notification (no id) must produce no response
    #[test]
    fn test_notification_produces_no_response() {
        let notif = r#"{"jsonrpc":"2.0","method":"notifications/initialized","params":null}"#;
        let raw = call_server(&format!("{}\n", notif));
        assert!(
            raw.trim().is_empty(),
            "notification must not get a response"
        );
    }

    // REF-197 introduced the `instructions` field; 1.7.0 reversed its Stage-1
    // "no Claude-Code-isms" policy after field data showed 14 of 16 sessions
    // opened with a wrong argument name because deferred schemas were never
    // loaded. The text must now carry the ToolSearch hint and exact call shapes.
    #[test]
    fn test_initialize_includes_instructions() {
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}"#;
        let raw = call_server(&format!("{}\n", req));
        let resp = parse_first_response(&raw);

        let instructions = resp["result"]["instructions"]
            .as_str()
            .expect("instructions must be a string");
        assert!(!instructions.is_empty(), "instructions must be non-empty");
        assert!(
            instructions.contains("index_project"),
            "instructions must mention the index_project recovery step"
        );
        assert!(
            instructions.to_lowercase().contains("prefer"),
            "instructions must carry the prefer-Reflex directive"
        );
        assert!(
            instructions.to_lowercase().contains("grep"),
            "instructions must name the native tools Reflex replaces (grep/glob/ripgrep)"
        );
        assert!(
            instructions.contains(
                "ToolSearch(\"select:mcp__reflex__search_code,mcp__reflex__search_regex,mcp__reflex__find_references\")"
            ),
            "instructions must tell deferred-schema harnesses how to load the tools"
        );

        // Pre-existing handshake fields must be unchanged.
        assert_eq!(resp["result"]["protocolVersion"], "2025-11-25");
        assert_eq!(resp["result"]["serverInfo"]["name"], "reflex");
    }

    #[test]
    fn test_instructions_have_pattern_call_shape() {
        assert!(
            !MCP_INSTRUCTIONS.contains("do NOT call ToolSearch"),
            "the false 'pre-loaded' claim must never come back"
        );
        assert!(
            !MCP_INSTRUCTIONS.contains("pre-loaded"),
            "the false 'pre-loaded' claim must never come back"
        );
        assert!(
            MCP_INSTRUCTIONS.contains("search_code {\"pattern\":"),
            "must show an exact search_code call shape with the real key"
        );
        assert!(MCP_INSTRUCTIONS.contains("search_regex {\"pattern\":"));
        assert!(MCP_INSTRUCTIONS.contains("find_references {\"pattern\":"));
    }

    /// The coverage text must say what is true: ripgrep's defaults, which skip
    /// dot-directories. "Every tracked file" made an agent trust a false zero.
    #[test]
    fn test_coverage_text_matches_ripgrep_defaults() {
        // The coverage rule is stated once, in the instructions (2026-09-30); no tool
        // description may contradict it.
        assert!(MCP_INSTRUCTIONS.contains("dot-director"));
        let tools = handle_list_tools(None, true).unwrap();
        let mut texts: Vec<String> = vec![MCP_INSTRUCTIONS.to_string()];
        for t in tools["tools"].as_array().unwrap() {
            texts.push(t["description"].as_str().unwrap().to_string());
        }
        for text in &texts {
            for bad in [
                "every tracked file",
                "every non-binary tracked file",
                "every file git tracks",
            ] {
                assert!(
                    !text.contains(bad),
                    "{bad:?} in {}",
                    &text[..text.len().min(120)]
                );
            }
        }
    }

    #[test]
    fn test_instructions_name_canonical_arg_names() {
        for needle in [
            "\"pattern\"",
            "\"limit\"",
            "\"file\"",
            "\"glob\"",
            "\"query\"",
            "\"symbol\"",
            "\"max_results\"",
            "\"path\"",
            "literal text index",
        ] {
            assert!(
                MCP_INSTRUCTIONS.contains(needle),
                "instructions must mention {needle}"
            );
        }
    }

    #[test]
    fn test_instructions_size_budget() {
        assert!(
            // 2400 (was 1700): on 2026-09-30 the matching, coverage and freshness rules
            // moved here from the tool descriptions. The instructions are sent once per
            // session; each description is carried on every turn.
            MCP_INSTRUCTIONS.len() <= 2400,
            "instructions are paid for on every session; keep them under 2400 chars (got {})",
            MCP_INSTRUCTIONS.len()
        );
    }

    // ---- 1.7.0 argument validation -------------------------------------------

    fn call_tool(name: &str, args: &str) -> serde_json::Value {
        let req = format!(
            r#"{{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{{"name":"{}","arguments":{}}}}}"#,
            name, args
        );
        let raw = call_server(&format!("{}\n", req));
        parse_first_response(&raw)
    }

    fn spec(name: &str) -> &'static ToolSpec {
        tool_spec(name).expect("tool has a spec")
    }

    #[test]
    fn test_every_listed_tool_has_spec() {
        for structural in [true, false] {
            let listing = handle_list_tools(None, structural).unwrap();
            for tool in listing["tools"].as_array().unwrap() {
                let name = tool["name"].as_str().unwrap();
                let s = tool_spec(name).unwrap_or_else(|| panic!("no spec for {name}"));
                for req in &s.required {
                    assert!(
                        s.valid.contains(req),
                        "{name}: required {req} not in properties"
                    );
                }
            }
        }
        assert!(tool_spec("no_such_tool").is_none());
    }

    #[test]
    fn test_alias_query_rewritten_to_pattern_with_warning() {
        let (args, warnings) = normalize_arguments(
            "search_code",
            spec("search_code"),
            json!({"query": "fn main"}),
        )
        .unwrap();
        assert_eq!(args["pattern"], "fn main");
        assert!(args.get("query").is_none());
        assert_eq!(
            warnings,
            vec!["argument \"query\" is deprecated; use \"pattern\""]
        );
    }

    #[test]
    fn test_alias_symbol_on_find_references() {
        let (args, warnings) = normalize_arguments(
            "find_references",
            spec("find_references"),
            json!({"symbol": "start_webauthn_registration"}),
        )
        .unwrap();
        assert_eq!(args["pattern"], "start_webauthn_registration");
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn test_alias_max_results_to_limit_and_path_to_file() {
        let (args, warnings) = normalize_arguments(
            "search_code",
            spec("search_code"),
            json!({"pattern": "x", "max_results": "40", "path": "src/protocol/scim"}),
        )
        .unwrap();
        assert_eq!(args["limit"], 40, "alias then numeric-string coercion");
        assert_eq!(args["file"], "src/protocol/scim");
        assert!(args.get("max_results").is_none());
        assert!(args.get("path").is_none());
        assert_eq!(warnings.len(), 2);
    }

    #[test]
    fn test_alias_ignored_when_canonical_present() {
        let (args, warnings) = normalize_arguments(
            "search_code",
            spec("search_code"),
            json!({"pattern": "real", "query": "ignored"}),
        )
        .unwrap();
        assert_eq!(args["pattern"], "real");
        assert!(warnings[0].contains("ignored \"query\""));
    }

    #[test]
    fn test_path_is_real_key_on_get_dependencies_no_warning() {
        let (args, warnings) = normalize_arguments(
            "get_dependencies",
            spec("get_dependencies"),
            json!({"path": "src/main.rs"}),
        )
        .unwrap();
        assert_eq!(args["path"], "src/main.rs");
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_unknown_argument_returns_32602_with_suggestion() {
        let resp = call_tool("search_code", r#"{"pattern":"x","max_resultz":5}"#);
        assert_eq!(resp["error"]["code"], -32602);
        assert_eq!(resp["error"]["data"]["kind"], "InvalidParams");
        let msg = resp["error"]["message"].as_str().unwrap();
        assert!(
            msg.starts_with("Unknown argument \"max_resultz\" for search_code"),
            "{msg}"
        );
        assert!(msg.contains("did you mean \"limit\""), "{msg}");
        assert!(
            msg.contains("Received: [\"max_resultz\",\"pattern\"]"),
            "{msg}"
        );
        assert!(msg.contains("Valid: [\"pattern\""), "{msg}");
    }

    #[test]
    fn test_unknown_argument_without_close_match_has_no_hint() {
        let resp = call_tool("search_code", r#"{"pattern":"x","zzzzzzzzzz":1}"#);
        assert_eq!(resp["error"]["code"], -32602);
        let msg = resp["error"]["message"].as_str().unwrap();
        assert!(!msg.contains("did you mean"), "{msg}");
    }

    #[test]
    fn test_missing_pattern_lists_received_and_valid() {
        let resp = call_tool("search_code", r#"{"limit": 5}"#);
        assert_eq!(resp["error"]["code"], -32602);
        let msg = resp["error"]["message"].as_str().unwrap();
        assert!(
            msg.starts_with("Missing required argument \"pattern\" for search_code"),
            "{msg}"
        );
        assert!(msg.contains("Received: [\"limit\"]"), "{msg}");
        assert!(msg.contains("Valid: ["), "{msg}");
    }

    #[test]
    fn test_numeric_string_limit_coerced_to_40() {
        let (args, _) = normalize_arguments(
            "search_code",
            spec("search_code"),
            json!({"pattern": "x", "limit": "40", "offset": " 3 "}),
        )
        .unwrap();
        assert_eq!(args["limit"].as_u64(), Some(40));
        assert_eq!(args["offset"].as_u64(), Some(3));
    }

    #[test]
    fn test_negative_and_non_numeric_limit_rejected() {
        for bad in [json!(-1), json!("abc"), json!(1.5), json!(true)] {
            let err = normalize_arguments(
                "search_code",
                spec("search_code"),
                json!({"pattern": "x", "limit": bad}),
            )
            .expect_err("must reject");
            let msg = err.to_string();
            assert!(
                msg.starts_with("Invalid value for \"limit\": expected a non-negative integer"),
                "{msg}"
            );
            let re = err.downcast_ref::<crate::errors::ReflexError>().unwrap();
            assert_eq!(re.kind(), "InvalidParams");
        }
    }

    #[test]
    fn test_bool_string_coerced() {
        let (args, _) = normalize_arguments(
            "search_code",
            spec("search_code"),
            json!({"pattern": "x", "paths": "true", "exact": "False"}),
        )
        .unwrap();
        assert_eq!(args["paths"], true);
        assert_eq!(args["exact"], false);
        let err = normalize_arguments(
            "search_code",
            spec("search_code"),
            json!({"pattern": "x", "paths": "yes"}),
        )
        .expect_err("must reject");
        assert!(err.to_string().contains("expected a boolean"));
    }

    #[test]
    fn test_pattern_non_string_typed_error() {
        let resp = call_tool("search_code", r#"{"pattern": 42}"#);
        assert_eq!(resp["error"]["code"], -32602);
        let msg = resp["error"]["message"].as_str().unwrap();
        assert_eq!(
            msg,
            "Invalid value for \"pattern\": expected a string, got 42"
        );
    }

    #[test]
    fn test_glob_string_promoted_to_array() {
        let (args, _) = normalize_arguments(
            "search_code",
            spec("search_code"),
            json!({"pattern": "x", "glob": "src/**/*.rs"}),
        )
        .unwrap();
        assert_eq!(args["glob"], json!(["src/**/*.rs"]));
    }

    #[test]
    fn test_non_object_arguments_rejected() {
        let resp = call_tool("search_code", r#""just a string""#);
        assert_eq!(resp["error"]["code"], -32602);
        let (args, _) = normalize_arguments(
            "check_index_status",
            spec("check_index_status"),
            Value::Null,
        )
        .unwrap();
        assert_eq!(args, json!({}));
    }

    #[test]
    fn test_warnings_survive_columnar_and_prose() {
        let data = to_columnar(sample_search_response());
        let out = finish_tool_result(data, vec!["w1".to_string()]);
        let text: Value =
            serde_json::from_str(out["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(text["warnings"], json!(["w1"]));
        assert!(text["rows"].is_array());

        let out = finish_tool_result(Value::String("prose".into()), vec!["w1".to_string()]);
        assert_eq!(out["content"][0]["text"], "prose\n\nwarnings: w1");
        let out = finish_tool_result(json!({"a": 1}), vec![]);
        assert_eq!(out["content"][0]["text"], r#"{"a":1}"#);
    }

    #[test]
    fn test_index_not_found_message_names_index_project() {
        let e: anyhow::Error = crate::errors::ReflexError::IndexNotFound.into();
        let msg = mcp_facing_message(&e);
        assert!(msg.contains("index_project"), "{msg}");
        assert!(!msg.contains("rfx index"), "{msg}");
        let e: anyhow::Error =
            crate::errors::ReflexError::CacheCorrupted("content.bin is too small".into()).into();
        let msg = mcp_facing_message(&e);
        assert!(msg.contains("content.bin is too small"), "{msg}");
        assert!(msg.contains("index_project"), "{msg}");
        assert!(!msg.contains("rfx "), "{msg}");
    }

    #[test]
    fn test_levenshtein_and_nearest_key() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        let valid: Vec<String> = ["pattern", "limit", "offset"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(nearest_key("limt", &valid), Some("limit"));
        assert_eq!(nearest_key("zzzzzzzz", &valid), None);
    }

    // REF-215: make_tool_result must emit ONLY the spec-guaranteed content[text]
    // JSON string — no structuredContent key. Dropped per the REF-196 board
    // decision so no client can consume both fields and double-count tokens.
    #[test]
    fn test_make_tool_result_content_text_only() {
        let data = json!({"status": "fresh", "count": 3});
        let result = super::make_tool_result(data.clone());

        // No structuredContent key anywhere in the result.
        assert!(
            result.get("structuredContent").is_none(),
            "make_tool_result must not emit structuredContent (REF-215): {result}"
        );

        // content[text] is the data serialized as a JSON string and round-trips
        // back to the original object.
        assert_eq!(result["content"][0]["type"], "text");
        let text = result["content"][0]["text"]
            .as_str()
            .expect("content[0].text must be a string");
        let roundtrip: serde_json::Value =
            serde_json::from_str(text).expect("content[text] must be valid JSON");
        assert_eq!(roundtrip, data);

        // The result object carries exactly the `content` key and nothing else,
        // so the tool-result shape is strictly content[text]-only.
        assert_eq!(
            result.as_object().map(|o| o.len()),
            Some(1),
            "result must contain only the `content` key: {result}"
        );
    }

    // REF-209: columnar output is the shipped default unless a caller opts out.
    #[test]
    fn test_columnar_enabled_default_on() {
        // Relies on REFLEX_MCP_COLUMNAR being unset in the harness environment.
        assert!(super::columnar_enabled());
    }

    // REF-212: the startup diagnostic must faithfully report every resolved flag
    // plus build provenance, so a stale binary or an un-honoured env toggle is
    // visible in Claude Code's mcp-logs for any trial.
    #[test]
    fn test_startup_flags_line_reports_resolved_flags() {
        // REF-215: only columnar + structural_tools remain (structuredContent /
        // sc_stage2 were removed along with the env vars that drove them).
        let line = super::startup_flags_line(false, true);
        assert!(line.starts_with("reflex-mcp startup:"), "line: {line}");
        assert!(line.contains("columnar=off"), "line: {line}");
        assert!(line.contains("structural_tools=on"), "line: {line}");
        // Build provenance present so an out-of-date rfx is identifiable.
        assert!(line.contains("build="), "line: {line}");
        // The removed flags must not reappear in the diagnostic.
        assert!(!line.contains("structuredContent"), "line: {line}");
        assert!(!line.contains("sc_stage2"), "line: {line}");

        // Inverting every flag flips exactly the on/off tokens.
        let off = super::startup_flags_line(true, false);
        assert!(off.contains("columnar=on"), "line: {off}");
        assert!(off.contains("structural_tools=off"), "line: {off}");
    }

    /// A representative two-file list-mode search response (post-flattening),
    /// mixing a symbol match and a plain-text match.
    fn sample_search_response() -> serde_json::Value {
        json!({
            "status": "fresh",
            "pagination": {"total": 2, "count": 2, "offset": 0, "limit": 200, "has_more": false},
            "results": [
                {
                    "path": "src/mcp.rs",
                    "language": "rust",
                    "matches": [
                        {"kind": "Function", "symbol": "make_tool_result",
                         "span": {"start_line": 955, "end_line": 957}, "preview": "fn make_tool_result"}
                    ]
                },
                {
                    "path": "src/query.rs",
                    "language": "rust",
                    "matches": [
                        {"span": {"start_line": 10, "end_line": 10}, "preview": "let x = 1;"},
                        {"span": {"start_line": 20, "end_line": 20}, "preview": "let y = 2;"}
                    ]
                }
            ],
            "total_count": 2,
            "returned_count": 2,
            "has_more": false
        })
    }

    // REF-209: `results` array is replaced by columns/rows; one row per match.
    #[test]
    fn test_to_columnar_reshapes_results() {
        let out = super::to_columnar(sample_search_response());

        // `results` is gone, replaced by `columns` + `rows`.
        assert!(out.get("results").is_none(), "results must be removed");
        let columns = out["columns"].as_array().expect("columns array");
        let rows = out["rows"].as_array().expect("rows array");

        // The five base columns always lead; kind/symbol appended because one
        // match carries them. No context columns (none present).
        let col_names: Vec<&str> = columns.iter().filter_map(|c| c.as_str()).collect();
        assert_eq!(
            col_names,
            vec![
                "path",
                "language",
                "start_line",
                "end_line",
                "preview",
                "kind",
                "symbol"
            ]
        );

        // One row per match across all files (1 + 2 = 3).
        assert_eq!(rows.len(), 3);

        // Symbol row is fully populated and positionally aligned to columns.
        assert_eq!(
            rows[0],
            json!([
                "src/mcp.rs",
                "rust",
                955,
                957,
                "fn make_tool_result",
                "Function",
                "make_tool_result"
            ])
        );
        // Plain-text rows carry null in the trailing optional columns.
        assert_eq!(
            rows[1],
            json!(["src/query.rs", "rust", 10, 10, "let x = 1;", null, null])
        );
        assert_eq!(
            rows[2],
            json!(["src/query.rs", "rust", 20, 20, "let y = 2;", null, null])
        );
    }

    // REF-209: top-level metadata (status/pagination/scalars) survives the reshape.
    #[test]
    fn test_to_columnar_preserves_metadata() {
        let out = super::to_columnar(sample_search_response());
        assert_eq!(out["status"], "fresh");
        assert_eq!(out["total_count"], 2);
        assert_eq!(out["returned_count"], 2);
        assert_eq!(out["has_more"], false);
        assert_eq!(out["pagination"]["total"], 2);
        assert_eq!(out["pagination"]["limit"], 200);
    }

    // REF-209: a pure full-text result (no kind/symbol/context) stays at the five
    // base columns — no all-null padding claws back the token saving.
    #[test]
    fn test_to_columnar_omits_absent_optional_columns() {
        let data = json!({
            "status": "fresh",
            "pagination": {"total": 1, "count": 1, "offset": 0, "limit": 200, "has_more": false},
            "results": [{
                "path": "a.rs", "language": "rust",
                "matches": [{"span": {"start_line": 1, "end_line": 1}, "preview": "struct X"}]
            }],
            "total_count": 1, "returned_count": 1, "has_more": false
        });
        let out = super::to_columnar(data);
        let col_names: Vec<&str> = out["columns"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|c| c.as_str())
            .collect();
        assert_eq!(
            col_names,
            vec!["path", "language", "start_line", "end_line", "preview"]
        );
        assert_eq!(out["rows"][0], json!(["a.rs", "rust", 1, 1, "struct X"]));
    }

    // REF-209: context columns appear only when a match carries context lines.
    #[test]
    fn test_to_columnar_includes_context_columns_when_present() {
        let data = json!({
            "status": "fresh",
            "pagination": {"total": 1, "count": 1, "offset": 0, "limit": 200, "has_more": false},
            "results": [{
                "path": "a.rs", "language": "rust",
                "matches": [{
                    "span": {"start_line": 5, "end_line": 5}, "preview": "hit",
                    "context_before": ["above"], "context_after": ["below"]
                }]
            }],
            "total_count": 1, "returned_count": 1, "has_more": false
        });
        let out = super::to_columnar(data);
        let col_names: Vec<&str> = out["columns"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|c| c.as_str())
            .collect();
        assert_eq!(
            col_names,
            vec![
                "path",
                "language",
                "start_line",
                "end_line",
                "preview",
                "context_before",
                "context_after"
            ]
        );
        assert_eq!(
            out["rows"][0],
            json!(["a.rs", "rust", 5, 5, "hit", ["above"], ["below"]])
        );
    }

    // REF-209: count-mode / non-results shapes pass through untouched, so
    // `to_columnar` is safe to call on any success value.
    #[test]
    fn test_to_columnar_passthrough_non_results() {
        let count = json!({"count": 7, "pattern": "foo"});
        assert_eq!(super::to_columnar(count.clone()), count);

        let scalar = json!("not an object");
        assert_eq!(super::to_columnar(scalar.clone()), scalar);
    }

    // REF-200: tool schemas must advertise the correct default limit (200, raised from 50 in REF-191) and max cap (500)
    #[test]
    fn test_tool_schema_limit_defaults() {
        let req = r#"{"jsonrpc":"2.0","id":5,"method":"tools/list","params":null}"#;
        let raw = call_server(&format!("{}\n", req));
        let resp = parse_first_response(&raw);
        let tools = resp["result"]["tools"].as_array().expect("tools array");

        let find_tool = |name: &str| {
            tools
                .iter()
                .find(|t| t["name"] == name)
                .unwrap_or_else(|| panic!("tool '{}' not found", name))
        };

        for tool_name in &["search_code", "search_regex", "find_references"] {
            let tool = find_tool(tool_name);
            let limit_desc = tool["inputSchema"]["properties"]["limit"]["description"]
                .as_str()
                .unwrap_or_else(|| panic!("{}: missing limit description", tool_name));
            assert!(
                limit_desc.contains("200"),
                "{}: limit description should mention default 200, got: {}",
                tool_name,
                limit_desc
            );
            assert!(
                limit_desc.contains("500"),
                "{}: limit description should mention max 500, got: {}",
                tool_name,
                limit_desc
            );
        }
    }

    // REF-189: structural tools absent when enable_structural_tools = false
    #[test]
    fn test_structural_tools_gated_by_flag() {
        // Since 2026-09-30 the structural analyses are one tool, `analyze`.
        const STRUCTURAL: &[&str] = &["analyze"];
        const ALWAYS_ON: &[&str] = &["search_code", "list_locations", "get_dependencies"];

        let req = r#"{"jsonrpc":"2.0","id":10,"method":"tools/list","params":null}"#;

        // Default (structural enabled): all 5 structural tools present
        let raw_on = call_server_with_structural(&format!("{}\n", req), true);
        let resp_on = parse_first_response(&raw_on);
        let tools_on = resp_on["result"]["tools"].as_array().expect("tools array");
        let names_on: Vec<&str> = tools_on.iter().filter_map(|t| t["name"].as_str()).collect();
        for name in STRUCTURAL {
            assert!(
                names_on.contains(name),
                "structural tool '{}' should appear when flag=true",
                name
            );
        }

        // Disabled: structural tools absent, day-to-day tools still present
        let raw_off = call_server_with_structural(&format!("{}\n", req), false);
        let resp_off = parse_first_response(&raw_off);
        let tools_off = resp_off["result"]["tools"].as_array().expect("tools array");
        let names_off: Vec<&str> = tools_off
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        for name in STRUCTURAL {
            assert!(
                !names_off.contains(name),
                "structural tool '{}' must be absent when flag=false",
                name
            );
        }
        for name in ALWAYS_ON {
            assert!(
                names_off.contains(name),
                "always-on tool '{}' must remain when flag=false",
                name
            );
        }
    }

    // REF-186: find_references should filter string/comment matches by default
    #[test]
    fn test_is_in_string_or_comment_filters_comment() {
        // Pattern inside a Rust single-line comment should be filtered
        let line = "let x = 5; // extract_symbols here";
        assert!(
            super::is_in_string_or_comment(crate::models::Language::Rust, line, "extract_symbols"),
            "pattern in comment should be classified as non-code"
        );
    }

    #[test]
    fn test_is_in_string_or_comment_filters_string_literal() {
        // Pattern inside a string literal should be filtered
        let line = r#"let s = "extract_symbols";"#;
        assert!(
            super::is_in_string_or_comment(crate::models::Language::Rust, line, "extract_symbols"),
            "pattern in string literal should be classified as non-code"
        );
    }

    #[test]
    fn test_is_in_string_or_comment_keeps_real_code() {
        // Pattern in real code should NOT be filtered
        let line = "fn extract_symbols(source: &str) -> Vec<SearchResult> {";
        assert!(
            !super::is_in_string_or_comment(crate::models::Language::Rust, line, "extract_symbols"),
            "real function name should not be classified as non-code"
        );
    }

    #[test]
    fn test_is_in_string_or_comment_mixed_line_keeps_match() {
        // When a line has the pattern both in a string AND in real code, the match
        // should be kept (conservative: real code occurrence wins)
        let line = r#"let _s = "extract_symbols"; extract_symbols(data);"#;
        assert!(
            !super::is_in_string_or_comment(crate::models::Language::Rust, line, "extract_symbols"),
            "when pattern appears in code on the same line, match should be kept"
        );
    }

    #[test]
    fn test_is_in_string_or_comment_unknown_language_keeps_match() {
        // Unknown language has no filter — always keep the match (conservative)
        let line = "extract_symbols in some unknown syntax";
        assert!(
            !super::is_in_string_or_comment(
                crate::models::Language::Unknown,
                line,
                "extract_symbols"
            ),
            "unknown language should never filter matches"
        );
    }

    #[test]
    fn test_find_references_schema_has_include_strings() {
        let tools_json =
            call_server(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":null}"#);
        let resp = parse_first_response(&tools_json);
        let tools = resp["result"]["tools"].as_array().expect("tools array");
        let find_refs = tools
            .iter()
            .find(|t| t["name"] == "find_references")
            .expect("find_references tool");
        let props = &find_refs["inputSchema"]["properties"];
        assert!(
            !props["include_strings"].is_null(),
            "find_references inputSchema must expose include_strings parameter"
        );
    }

    // REF-187: search_code, search_regex, and find_references must expose mode parameter
    #[test]
    fn test_count_mode_schema_exposed_on_search_tools() {
        let raw = call_server(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":null}"#);
        let resp = parse_first_response(&raw);
        let tools = resp["result"]["tools"].as_array().expect("tools array");

        for tool_name in &["search_code", "search_regex", "find_references"] {
            let tool = tools
                .iter()
                .find(|t| t["name"] == *tool_name)
                .unwrap_or_else(|| panic!("tool '{}' not found", tool_name));
            let props = &tool["inputSchema"]["properties"];
            assert!(
                !props["mode"].is_null(),
                "'{}' inputSchema must expose 'mode' parameter (REF-187)",
                tool_name
            );
            let enum_vals = props["mode"]["enum"]
                .as_array()
                .unwrap_or_else(|| panic!("'{}' mode must have enum values", tool_name));
            let vals: Vec<&str> = enum_vals.iter().filter_map(|v| v.as_str()).collect();
            assert!(
                vals.contains(&"count") && vals.contains(&"list"),
                "'{}' mode enum must contain 'count' and 'list', got {:?}",
                tool_name,
                vals
            );
        }
    }

    // REF-187: count mode must return {count, pattern} without match bodies
    // This test verifies the handler shape via a missing-index path (schema-level only,
    // since integration tests with a real index live in tests/).
    #[test]
    fn test_count_mode_missing_required_param_still_returns_32602() {
        // Ensure count mode parsing doesn't interfere with required param validation.
        let req = r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"search_code","arguments":{"mode":"count"}}}"#;
        let raw = call_server(&format!("{}\n", req));
        let resp = parse_first_response(&raw);
        // Missing pattern → must still return InvalidParams (-32602), not a crash
        assert_eq!(
            resp["error"]["code"], -32602,
            "count mode must not bypass required-param validation"
        );
    }
}
