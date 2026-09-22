//! MCP (Model Context Protocol) server implementation
//!
//! This module implements the MCP protocol directly over stdio using JSON-RPC 2.0.
//! It exposes Reflex's code search capabilities as MCP tools for AI coding assistants.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};
use std::path::Path;

use crate::cache::CacheManager;
use crate::dependency::DependencyIndex;
use crate::indexer::Indexer;
use crate::line_filter;
use crate::models::{IndexConfig, IndexStatus, Language, SymbolKind};
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

/// Bracket characters that make a whole-token literal search structurally unmatchable.
///
/// Whole-token matching wraps the pattern as `\b<pattern>\b`. A pattern ending in `)`
/// or `>` can never satisfy the trailing `\b`, because the next character is almost
/// never a word character. So `unwrap()`, `#[derive(` and `-> Result<` all returned a
/// silent `0` in the 1.7.0 field test, against ripgrep counts of 1221, 1141 and 2139.
const REGEX_ONLY_CHARS: &[char] = &['(', ')', '[', ']', '{', '}', '<', '>'];

/// A literal pattern rewritten so it cannot silently return zero.
struct LiteralPattern {
    /// What to actually search for (regex-escaped when `use_regex` is set).
    effective: String,
    /// Whether the query must run down the regex path.
    use_regex: bool,
    /// Explanation for the caller when the pattern was rewritten.
    warning: Option<String>,
}

/// Make a literal pattern searchable, rather than letting it return a confident zero.
///
/// A pattern containing brackets cannot match under whole-token rules, so it is
/// escaped and routed to the regex path, which is substring-based. The caller is told
/// in `warnings` — the rewrite is never silent.
///
/// `contains` mode already does substring matching, so it needs no rewrite.
fn prepare_literal_pattern(pattern: &str, contains: bool) -> LiteralPattern {
    if contains || !pattern.contains(REGEX_ONLY_CHARS) {
        return LiteralPattern {
            effective: pattern.to_string(),
            use_regex: false,
            warning: None,
        };
    }

    LiteralPattern {
        effective: regex::escape(pattern),
        use_regex: true,
        warning: Some(format!(
            "Pattern {:?} contains brackets, which a whole-identifier search can never \
             match. Searched it as an escaped regex instead (substring semantics). For \
             explicit control use search_regex, or pass contains:true.",
            pattern
        )),
    }
}

/// Attach the literal-search safety nets to a response object.
///
/// Two things go on, both aimed at the same failure: an agent reading a `0` and
/// concluding "no callers".
///
/// * `warnings` — set when the pattern was rewritten (brackets → escaped regex).
/// * `hint` — set when the result is empty but substring matches exist.
///
/// `pattern` is the caller's ORIGINAL pattern, not the rewritten one, so the message
/// names what they actually asked for.
///
/// `precomputed` is the engine's `substring_hint_count`: the number of candidate
/// lines that held the pattern only as a substring, counted during the search
/// itself. When the caller has it, no second search runs.
fn annotate_literal_result(
    response: &mut Value,
    root: &Path,
    pattern: &str,
    filter: &QueryFilter,
    prepared: &LiteralPattern,
    precomputed: Option<usize>,
) {
    let Some(obj) = response.as_object_mut() else {
        return;
    };

    if let Some(w) = &prepared.warning {
        obj.insert("warnings".to_string(), json!([w]));
    }

    // Only a whole-identifier search can be "explained"; a rewrite or an explicit
    // contains:true search already has substring semantics.
    if prepared.use_regex || filter.use_contains {
        return;
    }

    if !result_is_empty(obj) {
        return;
    }

    let hint = match precomputed {
        Some(count) => (count > 0).then(|| substring_hint_text(count, pattern)),
        None => zero_result_hint(root, pattern, filter),
    };
    if let Some(hint) = hint {
        obj.insert("hint".to_string(), json!(hint));
    }
}

fn substring_hint_text(count: usize, pattern: &str) -> String {
    format!(
        "0 whole-identifier matches; {} substring matches — pass contains:true to see them. \
         Reflex matches whole identifiers by default, so {:?} does not match longer names \
         that merely contain it.",
        count, pattern
    )
}

/// Whether a tool response carries no matches, across the several response shapes.
fn result_is_empty(obj: &serde_json::Map<String, Value>) -> bool {
    for key in ["total", "count", "total_locations", "total_count"] {
        if let Some(n) = obj.get(key).and_then(|v| v.as_u64()) {
            return n == 0;
        }
    }
    for key in ["rows", "results", "locations", "references"] {
        if let Some(a) = obj.get(key).and_then(|v| v.as_array()) {
            return a.is_empty();
        }
    }
    false
}

/// Explain a zero-result whole-token search by counting substring matches.
///
/// This is the single line that would have prevented every wrong conclusion in the
/// 1.7.0 field test: an agent that sees `0` for `verify_csrf` concludes "no callers"
/// and acts on it, when 89 lines contain `verify_csrf_form_field`.
///
/// Runs only when the search already returned nothing, so it costs nothing on the
/// common path. A failure here is swallowed — a missing hint must never fail a query.
fn zero_result_hint(root: &Path, pattern: &str, filter: &QueryFilter) -> Option<String> {
    let probe = QueryFilter {
        use_contains: true,
        limit: None,
        offset: None,
        paths_only: false,
        suppress_output: true,
        // A hint is a courtesy, never worth a slow response. A pathological substring
        // (say a two-character pattern on a huge repo) times out and yields no hint
        // rather than holding the answer the caller already has.
        timeout_secs: 5,
        ..filter.clone()
    };

    let engine = QueryEngine::new(CacheManager::new(root));
    let count = engine
        .search_with_metadata(pattern, probe)
        .ok()?
        .pagination
        .total;

    if count == 0 {
        return None;
    }

    Some(substring_hint_text(count, pattern))
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

Rules: the required argument is always "pattern", never "query", "symbol", or "text". The result cap is "limit", never "max_results". The path filter is "file" (substring) or "glob" (array), never "path". search_code is a literal text index: a natural-language query like "hot tier promotion" matches nothing, so search for identifiers or code fragments.

Use find_references for a definition plus every call site without string/comment noise. Use get_dependents for what imports a file.

On an "Index not found" or "corrupted" error, call index_project, then retry the failed tool. Only fall back to Grep/Glob after index_project has been called and the tool still fails."#;

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

/// Handle tools/list request
///
/// When `enable_structural` is false, the five structural-analysis-only tools
/// (`find_circular`, `find_islands`, `find_unused`, `analyze_summary`,
/// `get_transitive_deps`) are omitted from the response. Controlled by the
/// `[mcp] enable_structural_tools` flag in `~/.reflex/config.toml`.
fn handle_list_tools(_params: Option<Value>, enable_structural: bool) -> Result<Value> {
    // Names of tools gated behind enable_structural_tools config flag.
    // These are structural-analysis tools rarely needed for day-to-day code search.
    const STRUCTURAL_TOOLS: &[&str] = &[
        "find_circular",
        "find_islands",
        "find_unused",
        "analyze_summary",
        "get_transitive_deps",
    ];

    let all_tools = json!({
        "tools": [
            {
                "name": "list_locations",
                "description": "Cheapest way to find every place a pattern occurs. Prefer this over Glob-based path hunting and over Grep when you only need file + line numbers (no previews). Returns an array of `{path, line}` objects — one per match, no limit. MATCHING: matches WHOLE IDENTIFIERS by default — \"verify_csrf\" does NOT match \"verify_csrf_form_field\". Pass `contains: true` for substring matching. A 0 result carries a `hint` naming the substring count. \n\nUse this for: enumerating locations before deciding which files to Read; counting affected sites; listing all hits of a pattern without paying for previews. Supports `lang`, `file`, `glob`, `exclude` filters. \n\nExample: `pattern: \"CourtCase\"` → `[{\"path\": \"app/Models/CourtCase.php\", \"line\": 15}, {\"path\": \"app/Http/Controllers/CourtController.php\", \"line\": 42}]`. COVERAGE: code (rust, typescript, javascript, go, java, php, kotlin, python, c, c++, c#, ruby, vue, svelte, zig) AND docs/config (md, mdx, txt, yaml, yml, toml, json, proto, html, sh, bash, ini, cfg, sql, graphql). Use `lang: \"text\"` for docs and config only. Lock files are never indexed. On \"Index not found\" error, call `index_project`, then retry. \n\nFRESHNESS: every response carries `status` and `can_trust_results`. `status: \"stale\"` with `can_trust_results: false` means the index does not yet include your uncommitted edits — the accompanying `warning` names the changed paths. Results are still real matches; they may be incomplete, and a deleted file can still produce hits at its old lines. Call `index_project` and retry when completeness matters (find-all-callers, impact analysis, rename planning). This is normal after editing and is not an error.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern (text to find)"
                        },
                        "contains": {
                            "type": "boolean",
                            "description": "Substring matching, like `grep -F`. DEFAULT IS FALSE, which matches WHOLE IDENTIFIERS ONLY: pattern \"verify_csrf\" does NOT match \"verify_csrf_form_field\", and \"jwks_rps\" does NOT match \"jwks_rps_limit\". Pass true to find a pattern anywhere inside a longer identifier. If a search returns 0, check the response `hint` — it reports how many substring matches exist."
                        },
                        "lang": {
                            "type": "string",
                            "description": "Filter by language: rust, typescript, javascript, go, java, php, kotlin, python, c, cpp, csharp, ruby, vue, svelte, zig — or \"text\" for the docs/config tier (md, yaml, toml, json, proto, html, sh, sql, graphql)."
                        },
                        "file": {
                            "type": "string",
                            "description": "Filter by file path substring (e.g., 'Controllers')"
                        },
                        "glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Include files matching patterns (e.g., ['app/**/*.php'])"
                        },
                        "exclude": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude files matching patterns (e.g., ['vendor/**', 'tests/**'])"
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force execution of potentially expensive queries (bypasses broad query detection)"
                        },
                        "dependencies": {
                            "type": "boolean",
                            "description": "Include dependency information (imports) in results. Only extracts static imports."
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "count_occurrences",
                "description": "Count-only statistics for a pattern. Prefer this over piping `grep -c` / `wc -l` / `rg --count` — returns total occurrences and file count in one call without loading any content. MATCHING: matches WHOLE IDENTIFIERS by default — \"verify_csrf\" does NOT match \"verify_csrf_form_field\". Pass `contains: true` for substring matching. A 0 result carries a `hint` naming the substring count. COVERAGE: code (rust, typescript, javascript, go, java, php, kotlin, python, c, c++, c#, ruby, vue, svelte, zig) AND docs/config (md, mdx, txt, yaml, yml, toml, json, proto, html, sh, bash, ini, cfg, sql, graphql). Use `lang: \"text\"` for docs and config only. Lock files are never indexed. \n\nUse this for: \"how many times is X used?\"; impact checks before refactoring; validating search scope. Returns `{total, files, pattern}`. Supports all filters (`lang`, `file`, `glob`, `exclude`, `symbols`, `kind`). \n\nExample: `{\"total\": 87, \"files\": 12, \"pattern\": \"CourtCase\"}`. On \"Index not found\" error, call `index_project`, then retry. \n\nFRESHNESS: every response carries `status` and `can_trust_results`. `status: \"stale\"` with `can_trust_results: false` means the index does not yet include your uncommitted edits — the accompanying `warning` names the changed paths. Results are still real matches; they may be incomplete, and a deleted file can still produce hits at its old lines. Call `index_project` and retry when completeness matters (find-all-callers, impact analysis, rename planning). This is normal after editing and is not an error.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern (text to find)"
                        },
                        "contains": {
                            "type": "boolean",
                            "description": "Substring matching, like `grep -F`. DEFAULT IS FALSE, which matches WHOLE IDENTIFIERS ONLY: pattern \"verify_csrf\" does NOT match \"verify_csrf_form_field\", and \"jwks_rps\" does NOT match \"jwks_rps_limit\". Pass true to find a pattern anywhere inside a longer identifier. If a search returns 0, check the response `hint` — it reports how many substring matches exist."
                        },
                        "lang": {
                            "type": "string",
                            "description": "Filter by language"
                        },
                        "symbols": {
                            "type": "boolean",
                            "description": "Count symbol definitions only (not usages)"
                        },
                        "kind": {
                            "type": "string",
                            "description": "Filter by symbol kind (function, class, etc.)"
                        },
                        "file": {
                            "type": "string",
                            "description": "Filter by file path substring"
                        },
                        "glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Include files matching patterns"
                        },
                        "exclude": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude files matching patterns"
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force execution of potentially expensive queries (bypasses broad query detection)"
                        },
                        "dependencies": {
                            "type": "boolean",
                            "description": "Include dependency information (imports) in results. Only extracts static imports."
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "search_code",
                "description": "Default code search across the codebase. Prefer this over Grep / Glob for any pattern made of letters, digits, underscores, or hyphens — one call returns every occurrence with file paths, line numbers, and code previews. MATCHING: three modes. DEFAULT matches WHOLE IDENTIFIERS only — \"verify_csrf\" does NOT match \"verify_csrf_form_field\". `contains: true` matches substrings, like `grep -F`. `search_regex` matches regular expressions. A pattern with brackets (`()`, `[]`, `<>`) is escaped and run as a regex automatically, and says so in `warnings`. COVERAGE: code (rust, typescript, javascript, go, java, php, kotlin, python, c, c++, c#, ruby, vue, svelte, zig) AND docs/config (md, mdx, txt, yaml, yml, toml, json, proto, html, sh, bash, ini, cfg, sql, graphql). Use `lang: \"text\"` for docs and config only. Lock files are never indexed. Use this for: finding where a pattern occurs; listing all usages of a function/class/variable; finding a symbol's definition (with `symbols: true`); getting line numbers + previews in a single call. \n\nModes: full-text by default (definitions + usages); `symbols: true` returns definitions only; `mode: \"count\"` returns just `{count, pattern}` to check cardinality before paginating. For an explicit regular expression (`.*+?|^$`, character classes, alternation), use `search_regex`. \n\nResult shape is columnar: `{columns, rows}` — each row aligns positionally to `columns` (path, language, start_line, end_line, preview; then kind/symbol/context when present). Set env `REFLEX_MCP_COLUMNAR=0` for the legacy `results[]` shape. \n\nPagination: if `response.pagination.has_more` is true, fetch the next page with the `offset` parameter. On \"Index not found\" error, call `index_project`, then retry. \n\nFRESHNESS: every response carries `status` and `can_trust_results`. `status: \"stale\"` with `can_trust_results: false` means the index does not yet include your uncommitted edits — the accompanying `warning` names the changed paths. Results are still real matches; they may be incomplete, and a deleted file can still produce hits at its old lines. Call `index_project` and retry when completeness matters (find-all-callers, impact analysis, rename planning). This is normal after editing and is not an error.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern (text to find)"
                        },
                        "contains": {
                            "type": "boolean",
                            "description": "Substring matching, like `grep -F`. DEFAULT IS FALSE, which matches WHOLE IDENTIFIERS ONLY: pattern \"verify_csrf\" does NOT match \"verify_csrf_form_field\", and \"jwks_rps\" does NOT match \"jwks_rps_limit\". Pass true to find a pattern anywhere inside a longer identifier. If a search returns 0, check the response `hint` — it reports how many substring matches exist."
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["list", "count"],
                            "description": "Response mode: \"list\" (default) returns full match results; \"count\" returns only {count, pattern} — faster, skips match body serialization."
                        },
                        "lang": {
                            "type": "string",
                            "description": "Filter by language: rust, typescript, javascript, go, java, php, kotlin, python, c, cpp, csharp, ruby, vue, svelte, zig — or \"text\" for the docs/config tier (md, yaml, toml, json, proto, html, sh, sql, graphql)."
                        },
                        "kind": {
                            "type": "string",
                            "description": "Filter by symbol kind (function, class, struct, etc.)"
                        },
                        "symbols": {
                            "type": "boolean",
                            "description": "Symbol-only search (definitions, not usage)"
                        },
                        "exact": {
                            "type": "boolean",
                            "description": "Case-sensitive exact-identifier match. NOTE: substring matching is already OFF by default — use `contains: true` to turn it ON, not this flag."
                        },
                        "file": {
                            "type": "string",
                            "description": "Filter by file path (substring)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results per page (default: 200, max: 500). The 200-result default covers most find-all tasks in a single call. IMPORTANT: If response.has_more is true, you MUST fetch more pages using offset parameter."
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N results). ALWAYS paginate when has_more=true. Example: First call offset=0, second call offset=100, third offset=200, etc."
                        },
                        "expand": {
                            "type": "boolean",
                            "description": "Show full symbol body (not just signature)"
                        },
                        "glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Include files matching glob patterns (e.g., 'src/**/*.rs')"
                        },
                        "exclude": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude files matching glob patterns (e.g., 'target/**')"
                        },
                        "paths": {
                            "type": "boolean",
                            "description": "Return only unique file paths (not full results)"
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force execution of potentially expensive queries (bypasses broad query detection)"
                        },
                        "dependencies": {
                            "type": "boolean",
                            "description": "Include dependency information (imports) in results. **IMPORTANT:** Currently only supported for Rust files — passing this with any other language (typescript, python, go, etc.) will produce no dependency data. Only extracts static imports (string literals); dynamic imports are filtered. See CLAUDE.md for details."
                        },
                        "preview_length": {
                            "type": "integer",
                            "description": "Maximum characters per preview line (default: 180). Use a smaller value (e.g. 60) for wide-result scans where short previews are sufficient."
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "search_regex",
                "description": "Regex code search across the whole codebase. Prefer this over `rg` / `grep -E` / `grep -P` for pattern matching across files — one call returns every match with file paths, line numbers, and previews. \n\nUse this for patterns with special characters or regex operators: `->with\\(`, `::new\\(`, `fn (get|set)_\\w+`, `\\[(derive|test)\\]`, `\\bAuth\\w*Controller\\b`, alternation `a|b`, anchors `^$`, wildcards `.*`. Escaping: must escape `( ) [ ] { } . * + ? \\\\ | ^ $`; no escaping needed for `-> :: - _ / = < >`; in JSON use double backslashes (`\\\\(`, `\\\\[`). \n\nFor simple alphanumeric patterns use `search_code` instead — it is faster and avoids escaping overhead. For symbol definitions use `search_code` with `symbols: true`. \n\n`mode: \"count\"` returns `{count, pattern}` only. List-mode result shape is columnar: `{columns, rows}` — each row aligns positionally to `columns` (path, language, start_line, end_line, preview; then kind/symbol/context when present). Set env `REFLEX_MCP_COLUMNAR=0` for the legacy `results[]` shape. Pagination: if `response.pagination.has_more` is true, fetch the next page with `offset`. On \"Index not found\" error, call `index_project`, then retry. \n\nFRESHNESS: every response carries `status` and `can_trust_results`. `status: \"stale\"` with `can_trust_results: false` means the index does not yet include your uncommitted edits — the accompanying `warning` names the changed paths. Results are still real matches; they may be incomplete, and a deleted file can still produce hits at its old lines. Call `index_project` and retry when completeness matters (find-all-callers, impact analysis, rename planning). This is normal after editing and is not an error.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["list", "count"],
                            "description": "Response mode: \"list\" (default) returns full match results; \"count\" returns only {count, pattern} — faster, skips match body serialization."
                        },
                        "lang": {
                            "type": "string",
                            "description": "Filter by language"
                        },
                        "file": {
                            "type": "string",
                            "description": "Filter by file path"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 200, max: 500). Use with offset for pagination."
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N results after sorting)"
                        },
                        "glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Include files matching glob patterns"
                        },
                        "exclude": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude files matching glob patterns"
                        },
                        "paths": {
                            "type": "boolean",
                            "description": "Return only unique file paths"
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force execution of potentially expensive queries (bypasses broad query detection)"
                        },
                        "dependencies": {
                            "type": "boolean",
                            "description": "Include dependency information (imports) in results. Only extracts static imports."
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "search_ast",
                "description": "Structure-aware search using Tree-sitter AST patterns (S-expressions). ⚠️ SLOW: bypasses trigram optimization and scans the ENTIRE codebase (500ms-10s+). In 95% of cases, prefer `search_code` with `symbols: true` instead (10-100x faster). \n\nUse this only when you must match code structure rather than text: \"all async functions containing a `match` expression\", \"every class with a `serialize` method\", etc. You MUST pass `glob` to limit scope — without it, every file in the codebase is parsed. \n\nExample patterns — Rust: `(function_item) @fn`; Python: `(function_definition) @fn`; TypeScript: `(class_declaration) @class`. Refer to Tree-sitter grammar docs for each language. On \"Index not found\" error, call `index_project`, then retry. \n\nFRESHNESS: every response carries `status` and `can_trust_results`. `status: \"stale\"` with `can_trust_results: false` means the index does not yet include your uncommitted edits — the accompanying `warning` names the changed paths. Results are still real matches; they may be incomplete, and a deleted file can still produce hits at its old lines. Call `index_project` and retry when completeness matters (find-all-callers, impact analysis, rename planning). This is normal after editing and is not an error.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "AST pattern (Tree-sitter S-expression, e.g., '(function_item) @fn')"
                        },
                        "lang": {
                            "type": "string",
                            "description": "Language (REQUIRED: rust, typescript, javascript, python, go, java, c, cpp, csharp, php, ruby, kotlin, zig)"
                        },
                        "glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Include files matching glob patterns (STRONGLY RECOMMENDED to limit scope, e.g., ['src/**/*.rs'])"
                        },
                        "exclude": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude files matching glob patterns (e.g., ['target/**', 'node_modules/**'])"
                        },
                        "file": {
                            "type": "string",
                            "description": "Filter by file path (substring)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (use with offset for pagination)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N results after sorting)"
                        },
                        "paths": {
                            "type": "boolean",
                            "description": "Return only unique file paths"
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force execution of potentially expensive queries (bypasses broad query detection)"
                        },
                        "dependencies": {
                            "type": "boolean",
                            "description": "Include dependency information (imports) in results. Only extracts static imports."
                        }
                    },
                    "required": ["pattern", "lang"]
                }
            },
            {
                "name": "index_project",
                "description": "Rebuild or update the code search index. Call this whenever any Reflex search tool returns an \"Index not found\" or \"stale\" error — the retry will then succeed. Also call after large git operations (checkout, merge, rebase, pull), user file edits, or when results seem stale or missing. \n\nIncremental by default (only changed files re-indexed). Pass `force: true` for a full rebuild when the index appears corrupted.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "force": {
                            "type": "boolean",
                            "description": "Force full rebuild (ignore incremental)"
                        },
                        "languages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Languages to include (empty = all)"
                        }
                    }
                }
            },
            {
                "name": "get_dependencies",
                "description": "List every import (dependency) of a single file. Prefer this over grep-ing for `import` / `use` / `require` statements — Reflex answers from its pre-built import index, which grep cannot replicate without scanning every file. Returns one object per import with path, line, type (internal/external/stdlib), and optional symbols. \n\nUse this for: understanding file dependencies, analyzing import structure, finding what a file depends on. Path matching is fuzzy — exact paths, fragments, or bare filenames all work. Only static imports (string literals) are extracted; dynamic imports are filtered by design. On \"Index not found\" error, call `index_project`, then retry. \n\nFRESHNESS: every response carries `status` and `can_trust_results`. `status: \"stale\"` with `can_trust_results: false` means the index does not yet include your uncommitted edits — the accompanying `warning` names the changed paths. Results are still real matches; they may be incomplete, and a deleted file can still produce hits at its old lines. Call `index_project` and retry when completeness matters (find-all-callers, impact analysis, rename planning). This is normal after editing and is not an error.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path (supports fuzzy matching: 'Controllers/FooController.php' or just 'FooController.php')"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "get_dependents",
                "description": "Reverse dependency lookup — find every file that imports a given file. Prefer this over grep-based find-callers: Reflex answers from its pre-built reverse-import index in one call, which grep cannot replicate without scanning every file. Returns the list of importing file paths. \n\nUse this for: impact analysis before changing a module; finding consumers of a library; detecting file importance. Path matching is fuzzy — exact paths, fragments, or bare filenames all work. Only static imports (string literals) are considered; dynamic imports are filtered by design. On \"Index not found\" error, call `index_project`, then retry. \n\nFRESHNESS: every response carries `status` and `can_trust_results`. `status: \"stale\"` with `can_trust_results: false` means the index does not yet include your uncommitted edits — the accompanying `warning` names the changed paths. Results are still real matches; they may be incomplete, and a deleted file can still produce hits at its old lines. Call `index_project` and retry when completeness matters (find-all-callers, impact analysis, rename planning). This is normal after editing and is not an error.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path (supports fuzzy matching: 'models/User.php' or just 'User.php')"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "get_transitive_deps",
                "description": "Walk the transitive dependency tree of a file up to `depth` levels (default 3). Prefer this over hand-rolling recursive grep across imports — Reflex traverses the static import graph directly, returning a map of file → depth. \n\nUse this for: understanding the full dependency chain, analyzing deep coupling, planning refactoring blast radius. Example: `depth=2` finds file → deps → deps of deps. Only static imports (string literals) are followed; dynamic imports are filtered by design. On \"Index not found\" error, call `index_project`, then retry. \n\nFRESHNESS: every response carries `status` and `can_trust_results`. `status: \"stale\"` with `can_trust_results: false` means the index does not yet include your uncommitted edits — the accompanying `warning` names the changed paths. Results are still real matches; they may be incomplete, and a deleted file can still produce hits at its old lines. Call `index_project` and retry when completeness matters (find-all-callers, impact analysis, rename planning). This is normal after editing and is not an error.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path (supports fuzzy matching)"
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Maximum depth to traverse (default: 3, max recommended: 5)"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "find_hotspots",
                "description": "Rank files by how many other files import them (dependency hotspots). Prefer this over any grep-based \"most-imported file\" heuristic — Reflex answers from its pre-built dependency index in one call; grep cannot answer this without scanning every file. \n\nUse this for: finding critical-path files; identifying refactoring blast radius; ranking modules by coupling; architecture review. Returns `{pagination, results: [{path, import_count}]}` sorted by import count (desc by default; use `sort` to change). Default page size 200; if `pagination.has_more` is true, fetch the next page with `offset`. Only static imports are counted. On \"Index not found\" error, call `index_project`, then retry. \n\nFRESHNESS: every response carries `status` and `can_trust_results`. `status: \"stale\"` with `can_trust_results: false` means the index does not yet include your uncommitted edits — the accompanying `warning` names the changed paths. Results are still real matches; they may be incomplete, and a deleted file can still produce hits at its old lines. Call `index_project` and retry when completeness matters (find-all-callers, impact analysis, rename planning). This is normal after editing and is not an error. \n\nExample: `{\"results\": [{\"path\": \"src/models.rs\", \"import_count\": 27}]}`",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of hotspots per page (default: 200)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N results). Use with limit for pagination."
                        },
                        "min_dependents": {
                            "type": "integer",
                            "description": "Minimum number of dependents to include (default: 2)"
                        },
                        "sort": {
                            "type": "string",
                            "description": "Sort order: 'asc' (least imports first) or 'desc' (most imports first, default)"
                        }
                    }
                }
            },
            {
                "name": "find_circular",
                "description": "Detect circular dependencies (cycles A → B → C → A) in the static import graph. Prefer this over manually grepping for import chains — Reflex does the cycle detection directly. Returns `{pagination, results: [{paths: [\"a.rs\", \"b.rs\", \"a.rs\"]}]}`, sorted with longest cycles first by default. Default page size 200; if `pagination.has_more` is true, fetch the next page with `offset`. Only static imports are considered. On \"Index not found\" error, call `index_project`, then retry. \n\nFRESHNESS: every response carries `status` and `can_trust_results`. `status: \"stale\"` with `can_trust_results: false` means the index does not yet include your uncommitted edits — the accompanying `warning` names the changed paths. Results are still real matches; they may be incomplete, and a deleted file can still produce hits at its old lines. Call `index_project` and retry when completeness matters (find-all-callers, impact analysis, rename planning). This is normal after editing and is not an error.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of cycles per page (default: 200)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N cycles). Use with limit for pagination."
                        },
                        "sort": {
                            "type": "string",
                            "description": "Sort order: 'asc' (shortest cycles first) or 'desc' (longest cycles first, default)"
                        }
                    }
                }
            },
            {
                "name": "find_unused",
                "description": "List files that no other file imports — orphan candidates for deletion. Prefer this over manual Glob + Grep cross-referencing — Reflex answers from the static import graph in one call. Returns `{pagination, results: [\"src/unused.rs\", \"tests/old.rs\", ...]}`. Default page size 200; if `pagination.has_more` is true, fetch the next page with `offset`. Note: entry points (`main.rs`, `index.ts`) appear as unused by design — do not delete them. Only static imports are considered. On \"Index not found\" error, call `index_project`, then retry. \n\nFRESHNESS: every response carries `status` and `can_trust_results`. `status: \"stale\"` with `can_trust_results: false` means the index does not yet include your uncommitted edits — the accompanying `warning` names the changed paths. Results are still real matches; they may be incomplete, and a deleted file can still produce hits at its old lines. Call `index_project` and retry when completeness matters (find-all-callers, impact analysis, rename planning). This is normal after editing and is not an error.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of unused files per page (default: 200)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N files). Use with limit for pagination."
                        }
                    }
                }
            },
            {
                "name": "find_islands",
                "description": "Find disconnected components (islands) in the static import graph — groups of files that have no imports crossing group boundaries. Prefer this over manual Glob + Grep cluster analysis — Reflex computes the connected components directly. Returns `{pagination, results: [{island_id, size, paths: [...]}]}` sorted with largest islands first by default. Default page size 200; if `pagination.has_more` is true, fetch the next page with `offset`. Use `min_island_size` and `max_island_size` to filter by component size (default: 2–500 files, or 50% of total). Only static imports are considered. On \"Index not found\" error, call `index_project`, then retry. \n\nFRESHNESS: every response carries `status` and `can_trust_results`. `status: \"stale\"` with `can_trust_results: false` means the index does not yet include your uncommitted edits — the accompanying `warning` names the changed paths. Results are still real matches; they may be incomplete, and a deleted file can still produce hits at its old lines. Call `index_project` and retry when completeness matters (find-all-callers, impact analysis, rename planning). This is normal after editing and is not an error.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of islands per page (default: 200)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N islands). Use with limit for pagination."
                        },
                        "min_island_size": {
                            "type": "integer",
                            "description": "Minimum files in an island to include (default: 2)"
                        },
                        "max_island_size": {
                            "type": "integer",
                            "description": "Maximum files in an island to include (default: 500 or 50% of total files)"
                        },
                        "sort": {
                            "type": "string",
                            "description": "Sort order: 'asc' (smallest islands first) or 'desc' (largest islands first, default)"
                        }
                    }
                }
            },
            {
                "name": "analyze_summary",
                "description": "One-call overview of codebase dependency health. Prefer this over running `find_circular` + `find_hotspots` + `find_unused` + `find_islands` individually — returns aggregate counts so the agent can decide which specific analysis to drill into. Returns `{circular_dependencies, hotspots, unused_files, islands, min_dependents}`. Only static imports are considered. On \"Index not found\" error, call `index_project`, then retry. \n\nFRESHNESS: every response carries `status` and `can_trust_results`. `status: \"stale\"` with `can_trust_results: false` means the index does not yet include your uncommitted edits — the accompanying `warning` names the changed paths. Results are still real matches; they may be incomplete, and a deleted file can still produce hits at its old lines. Call `index_project` and retry when completeness matters (find-all-callers, impact analysis, rename planning). This is normal after editing and is not an error. \n\nExample: `{\"circular_dependencies\": 17, \"hotspots\": 10, \"unused_files\": 82, \"islands\": 81, \"min_dependents\": 2}`",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "min_dependents": {
                            "type": "integer",
                            "description": "Minimum number of dependents for hotspots (default: 2)"
                        }
                    }
                }
            },
            {
                "name": "find_references",
                "description": "Atomic symbol definition + every usage in one call. Prefer this over the two-step Grep-based find-all-callers pattern (search, then filter to call sites by eye) and over chaining `search_code(symbols=true) + search_code()` — `find_references` returns both the definition and all call sites in a single call, complete with no follow-up searches needed. \n\nUse this for: \"find all callers of X\" (the most common agent refactoring task); impact analysis before changing a function or class; rename planning; dead-code detection before deleting a function. \n\nBy default, matches inside string literals and comments are excluded (so test fixtures and doc comments don't drown out real call sites); pass `include_strings: true` to restore all occurrences. CODE FILES ONLY: the docs/config tier (md, yaml, json, toml, html, sh, proto) is never searched here, because a name mentioned in a changelog is not a call site — use search_code with `lang: \"text\"` for those. Returns `{definition, references, total_references, returned_count, filtered_out, pagination, status}`. `pagination.total` and `total_references` are the RAW totals before string/comment filtering (that is the space `offset` indexes into); `returned_count` is what this page actually returns after filtering, and `filtered_out` is the difference — they are not expected to be equal. `definition` is the first symbol definition (`{path, line, kind, symbol, span, preview}`) or null, and `references` is a flat array of `{path, line, preview}` covering every textual occurrence including the definition site itself. Pagination applies to `references` only; if `pagination.has_more` is true, fetch the next page with `offset`. On \"Index not found\" error, call `index_project`, then retry. \n\nFRESHNESS: every response carries `status` and `can_trust_results`. `status: \"stale\"` with `can_trust_results: false` means the index does not yet include your uncommitted edits — the accompanying `warning` names the changed paths. Results are still real matches; they may be incomplete, and a deleted file can still produce hits at its old lines. Call `index_project` and retry when completeness matters (find-all-callers, impact analysis, rename planning). This is normal after editing and is not an error.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Symbol name or text pattern to find references for (e.g., 'CacheManager', 'extract_symbols')"
                        },
                        "contains": {
                            "type": "boolean",
                            "description": "Substring matching, like `grep -F`. DEFAULT IS FALSE, which matches WHOLE IDENTIFIERS ONLY: pattern \"verify_csrf\" does NOT match \"verify_csrf_form_field\", and \"jwks_rps\" does NOT match \"jwks_rps_limit\". Pass true to find a pattern anywhere inside a longer identifier. If a search returns 0, check the response `hint` — it reports how many substring matches exist."
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["list", "count"],
                            "description": "Response mode: \"list\" (default) returns full results with definition + references; \"count\" returns only {count, pattern} — faster, skips match body serialization."
                        },
                        "kind": {
                            "type": "string",
                            "description": "Filter definition lookup by symbol kind (function, class, struct, trait, etc.)"
                        },
                        "lang": {
                            "type": "string",
                            "description": "Filter by language (rust, typescript, python, go, etc.)"
                        },
                        "glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Include files matching glob patterns (e.g., ['src/**/*.rs'])"
                        },
                        "exclude": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude files matching glob patterns (e.g., ['target/**', 'tests/**'])"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max references per page (default: 200, max: 500). The 200-result default covers most find-all tasks in a single call. Pagination applies to references only."
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset for references (skip first N). Use with limit."
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force execution of potentially expensive queries (bypasses broad query detection)"
                        },
                        "include_strings": {
                            "type": "boolean",
                            "description": "Include matches inside string literals and comments (default: false). By default these are excluded to focus on real call sites."
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "gather_context",
                "description": "One-shot codebase orientation: structure, file types, project type, frameworks, entry points, test layout, config files. Prefer this over Glob-based recon at session start — Reflex returns a single consolidated overview instead of multiple glob calls. By default (no parameters) all context types are gathered; pass individual flags (`structure`, `framework`, `entry_points`, etc.) for a focused slice. Use `depth` to control tree depth (default 2) and `path` to focus on a subdirectory. \n\nUse this for: getting oriented in an unfamiliar codebase; locating entry points; confirming which frameworks/languages are in use. For finding where a specific symbol/pattern lives, use `search_code` or `find_references` instead.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "structure": {
                            "type": "boolean",
                            "description": "Show directory structure"
                        },
                        "file_types": {
                            "type": "boolean",
                            "description": "Show file type distribution"
                        },
                        "project_type": {
                            "type": "boolean",
                            "description": "Detect project type (CLI/library/webapp/monorepo)"
                        },
                        "framework": {
                            "type": "boolean",
                            "description": "Detect frameworks and conventions"
                        },
                        "entry_points": {
                            "type": "boolean",
                            "description": "Show entry point files"
                        },
                        "test_layout": {
                            "type": "boolean",
                            "description": "Show test organization pattern"
                        },
                        "config_files": {
                            "type": "boolean",
                            "description": "List important configuration files"
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Tree depth for structure (default: 2)"
                        },
                        "path": {
                            "type": "string",
                            "description": "Focus on specific directory path"
                        }
                    }
                }
            },
            {
                "name": "check_index_status",
                "description": "Check whether the Reflex search index is fresh, stale, or missing — without running any search. Call this at session start, after any git operation (checkout, merge, rebase, pull), and after editing files. \n\nReturns `{status: \"fresh\" | \"stale\" | \"missing\", can_trust_results, reason, action_required, files_modified?, files_added?, files_deleted?, changed_count?}`. The three file lists name the actual paths (capped at 100 each; `truncated` is set if cut short). `action_required` is the tool to call — `index_project`. \n\nStale covers three cases: an unindexed branch, HEAD moved off the indexed commit, and UNCOMMITTED WORKING-TREE CHANGES — edited, newly created, or deleted files. A deleted file is the serious one: it still produces hits at its old lines until you reindex. \n\n`can_trust_results` is narrower than `status`: it goes false only when the changes could actually affect an answer. A stale index with `can_trust_results: true` means results are still sound, they just may not include your newest edits. \n\nLIMITATION: outside a git repository, working-tree changes are not detected and status is always reported fresh. \n\nExample fresh: `{\"status\": \"fresh\", \"can_trust_results\": true}`. Example stale: `{\"status\": \"stale\", \"can_trust_results\": false, \"reason\": \"Working tree has uncommitted changes since indexing (2 modified, 1 added)\", \"action_required\": \"index_project\", \"files_modified\": [\"src/storage/mod.rs\", \"src/lib.rs\"], \"files_added\": [\"src/storage/zz_probe.rs\"], \"changed_count\": 3}`",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    });

    if enable_structural {
        return Ok(all_tools);
    }

    // Filter out structural-analysis tools when disabled in config
    let tools: Vec<Value> = all_tools["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| !STRUCTURAL_TOOLS.contains(&t["name"].as_str().unwrap_or("")))
        .cloned()
        .collect();
    Ok(json!({ "tools": tools }))
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
    "expand",
    "paths",
    "force",
    "dependencies",
    "include_strings",
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
        let listing = handle_list_tools(None, true).expect("static tool list is valid JSON");
        for tool in listing["tools"].as_array().into_iter().flatten() {
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
    if force {
        log::info!("Force rebuild requested, clearing existing cache");
        cache.clear()?;
    }
    let config = IndexConfig {
        languages,
        ..Default::default()
    };
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

fn handle_call_tool(params: Option<Value>, root: &Path) -> Result<Value> {
    let params = params.ok_or_else(|| anyhow::anyhow!("Missing params for tools/call"))?;

    let name = params["name"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing tool name"))?;

    let spec = tool_spec(name).ok_or_else(|| anyhow::anyhow!("Unknown tool: {}", name))?;
    let (arguments, warnings) = normalize_arguments(name, spec, params["arguments"].clone())?;

    let data = with_corruption_recovery(name, root, || dispatch_tool(name, &arguments, root))?;

    Ok(finish_tool_result(data, warnings))
}

/// Run one tool. Every arm returns the tool's *data* (a JSON object, or a
/// plain `Value::String` for prose tools such as `gather_context`);
/// `handle_call_tool` wraps it into the MCP `content` envelope.
fn dispatch_tool(name: &str, arguments: &Value, root: &Path) -> Result<Value> {
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
            // Brackets can never match under whole-identifier rules, so route them to
            // the regex path rather than returning a confident zero.
            let prepared = prepare_literal_pattern(&pattern, contains);
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

            let filter = QueryFilter {
                language,
                kind: None,
                use_ast: false,
                use_regex: prepared.use_regex,
                limit: None, // The tool contract is "one per match, no limit"
                symbols_mode: false,
                expand: false,
                file_pattern: file,
                exact: false,
                use_contains: contains,
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
            let engine = QueryEngine::new(cache);
            let response = engine.search_with_metadata(&prepared.effective, filter.clone())?;

            // Extract locations (path + line) for each match
            let locations: Vec<serde_json::Value> = response
                .results
                .iter()
                .flat_map(|file_group| {
                    file_group.matches.iter().map(move |m| {
                        json!({
                            "path": file_group.path.clone(),
                            "line": m.span.start_line
                        })
                    })
                })
                .collect();

            // Return compact response (just locations + count)
            let mut compact_response = json!({
                "status": response.status,
                "total_locations": locations.len(),
                "locations": locations
            });
            annotate_literal_result(
                &mut compact_response,
                root,
                &pattern,
                &filter,
                &prepared,
                response.substring_hint_count,
            );

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
            // Brackets can never match under whole-identifier rules, so route them to
            // the regex path rather than returning a confident zero.
            let prepared = prepare_literal_pattern(&pattern, contains);
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
                use_regex: prepared.use_regex,
                limit: None, // No limit for counting
                symbols_mode,
                expand: false,
                file_pattern: file,
                exact: false,
                use_contains: contains,
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
            let engine = QueryEngine::new(cache);
            let response = engine.search_with_metadata(&prepared.effective, filter.clone())?;

            // Count unique files
            use std::collections::HashSet;
            let unique_files: HashSet<String> =
                response.results.iter().map(|fg| fg.path.clone()).collect();

            // Return minimal stats
            let mut stats = json!({
                "status": response.status,
                "pattern": pattern,
                "total": response.pagination.total,
                "files": unique_files.len()
            });
            annotate_literal_result(
                &mut stats,
                root,
                &pattern,
                &filter,
                &prepared,
                response.substring_hint_count,
            );

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
            // Brackets can never match under whole-identifier rules, so route them to
            // the regex path rather than returning a confident zero.
            let prepared = prepare_literal_pattern(&pattern, contains);
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
                    use_regex: prepared.use_regex,
                    limit: None, // count everything
                    symbols_mode,
                    expand: false,
                    file_pattern: file,
                    exact: exact.unwrap_or(false),
                    use_contains: contains,
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
                let engine = QueryEngine::new(cache);
                let response =
                    engine.search_with_metadata(&prepared.effective, count_filter.clone())?;
                let mut result = json!({"count": response.pagination.total, "pattern": pattern});
                annotate_literal_result(
                    &mut result,
                    root,
                    &pattern,
                    &count_filter,
                    &prepared,
                    response.substring_hint_count,
                );
                return Ok(result);
            }

            let filter = QueryFilter {
                language,
                kind: parsed_kind,
                use_ast: false,
                use_regex: prepared.use_regex,
                limit: final_limit,
                symbols_mode,
                expand: expand.unwrap_or(false),
                file_pattern: file,
                exact: exact.unwrap_or(false),
                use_contains: contains,
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
            let engine = QueryEngine::new(cache);
            let mut response = engine.search_with_metadata(&prepared.effective, filter.clone())?;

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
                response.pagination.total,
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
            let substring_hint_count = response.substring_hint_count;

            let mut response_val = serde_json::to_value(response)?;
            if let serde_json::Value::Object(ref mut map) = response_val {
                map.insert("has_more".to_string(), json!(has_more));
                map.insert("total_count".to_string(), json!(total_count));
                map.insert("returned_count".to_string(), json!(result_count));
            }

            // REF-209: emit the token-efficient columnar shape by default; the
            // env toggle restores the legacy results[] array for compatibility.
            if columnar_enabled() {
                response_val = to_columnar(response_val);
            }

            // Applied after the columnar reshape so the hint survives both shapes.
            annotate_literal_result(
                &mut response_val,
                root,
                &pattern,
                &filter,
                &prepared,
                substring_hint_count,
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
                    symbols_mode: false,
                    expand: false,
                    file_pattern: file,
                    exact: false,
                    use_contains: false,
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
                let engine = QueryEngine::new(cache);
                let response = engine.search_with_metadata(&pattern, count_filter)?;
                let result = json!({"count": response.pagination.total, "pattern": pattern});
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
                timeout_secs: 30,    // Default 30 second timeout for MCP queries
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
            let engine = QueryEngine::new(cache);
            let mut response = engine.search_with_metadata(&pattern, filter)?;

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
                response.pagination.total,
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
            let mut response_val = serde_json::to_value(response)?;
            if let serde_json::Value::Object(ref mut map) = response_val {
                map.insert("has_more".to_string(), json!(has_more));
                map.insert("total_count".to_string(), json!(total_count));
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
            if language.is_text() {
                anyhow::bail!(
                    "lang \"{}\" is the plain-text tier (markdown, YAML, JSON, TOML, HTML, \
                     shell, proto). These files are trigram-indexed only and have no AST. \
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
            let engine = QueryEngine::new(cache);

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

            let engine = QueryEngine::new(cache);
            // This is the explicit probe. An agent that asks whether the index is
            // current must never be answered from the freshness memo.
            let (status, can_trust, warning) = engine.fresh_index_status()?;

            let status_str = match status {
                IndexStatus::Fresh => "fresh",
                IndexStatus::Stale => "stale",
            };

            let result = if let Some(w) = warning {
                let mut obj = json!({
                    "status": status_str,
                    "can_trust_results": can_trust,
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
                json!({ "status": status_str, "can_trust_results": can_trust })
            };

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
            // Brackets can never match under whole-identifier rules, so route them to
            // the regex path rather than returning a confident zero.
            let prepared = prepare_literal_pattern(&pattern, contains);
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
                    use_regex: prepared.use_regex,
                    limit: None, // count everything
                    symbols_mode: false,
                    expand: false,
                    file_pattern: None,
                    exact: false,
                    use_contains: contains,
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
                let engine = QueryEngine::new(cache);
                let response =
                    engine.search_with_metadata(&prepared.effective, count_filter.clone())?;

                // `include_strings` must mean the same thing in count mode as in list
                // mode. It previously returned `pagination.total`, which is the raw
                // engine total, so include_strings:true and :false gave the SAME number
                // even when many hits were in string literals and doc comments.
                let count = if include_strings {
                    response.pagination.total
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
                annotate_literal_result(
                    &mut result,
                    root,
                    &pattern,
                    &count_filter,
                    &prepared,
                    response.substring_hint_count,
                );
                return Ok(result);
            }

            // Search 1: Find symbol definition (symbols_mode=true, small cap)
            let def_filter = QueryFilter {
                language,
                kind: parsed_kind,
                use_ast: false,
                use_regex: prepared.use_regex,
                limit: Some(5),
                symbols_mode: true,
                expand: false,
                file_pattern: None,
                exact: false,
                use_contains: contains,
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
            let engine = QueryEngine::new(cache);
            let def_response = engine.search_with_metadata(&prepared.effective, def_filter)?;

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
                use_regex: prepared.use_regex,
                // REF-191: default to the one-call page size so "find all callers"
                // returns the full set instead of paginating at 50.
                limit: limit.map(|l| l.min(500)).or(Some(DEFAULT_MCP_RESULT_LIMIT)),
                symbols_mode: false,
                expand: false,
                file_pattern: None,
                exact: false,
                use_contains: contains,
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
                ..Default::default()
            };

            let ref_filter_for_hint = ref_filter.clone();
            let ref_response = engine.search_with_metadata(&prepared.effective, ref_filter)?;

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
            let substring_hint_count = ref_response.substring_hint_count;

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
            annotate_literal_result(
                &mut response,
                root,
                &pattern,
                &ref_filter_for_hint,
                &prepared,
                substring_hint_count,
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
) -> JsonRpcResponse {
    log::debug!("MCP request: method={}", request.method);

    let result = match request.method.as_str() {
        "initialize" => handle_initialize(request.params),
        "tools/list" => handle_list_tools(request.params, enable_structural),
        "tools/call" => handle_call_tool(request.params, root),
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
pub fn run_mcp_server() -> Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    run_mcp_server_io(stdin.lock(), stdout.lock())
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
    run_mcp_server_io_impl(reader, writer, enable_structural, root)
}

/// Inner server loop. Accepts `enable_structural` so tests can drive the flag
/// without touching the filesystem, and `root` so tools resolve `.reflex/`
/// relative to a chosen workspace.
fn run_mcp_server_io_impl<R: BufRead, W: Write>(
    reader: R,
    mut writer: W,
    enable_structural: bool,
    root: &Path,
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
        let response = process_request(request, enable_structural, root);
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
        run_mcp_server_io_impl(reader, &mut output, enable_structural, Path::new(".")).unwrap();
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
            MCP_INSTRUCTIONS.len() <= 1600,
            "instructions are paid for on every session; keep them under 1600 chars (got {})",
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
        const STRUCTURAL: &[&str] = &[
            "find_circular",
            "find_islands",
            "find_unused",
            "analyze_summary",
            "get_transitive_deps",
        ];
        const ALWAYS_ON: &[&str] = &["search_code", "find_hotspots", "get_dependencies"];

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
