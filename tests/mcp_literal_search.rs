//! Literal-search semantics over MCP.
//!
//! Regression tests for the 1.7.0 field report, where whole-identifier matching was
//! undocumented and had no MCP switch. Measured against ripgrep on a real repo:
//!
//! | pattern          | rg -F | Reflex 1.7.0 |
//! |------------------|-------|--------------|
//! | `verify_csrf`    |    89 |            0 |
//! | `jwks_rps`       |    22 |            0 |
//! | `unwrap()`       |  1221 |            0 |
//! | `-> Result<`     |  2139 |            0 |
//!
//! Every one of those zeros carried `can_trust_results: true`, so the agent concluded
//! "no callers" and acted on it.

use reflex::mcp::run_mcp_server_io_in;
use reflex::{CacheManager, IndexConfig, Indexer};
use serde_json::{Value, json};
use std::io::Cursor;
use std::path::Path;
use tempfile::TempDir;

/// A workspace whose only `verify_csrf`-ish identifier is a LONGER one.
fn workspace() -> TempDir {
    let temp = TempDir::new().unwrap();
    std::fs::create_dir_all(temp.path().join("src")).unwrap();
    std::fs::write(
        temp.path().join("src/auth.rs"),
        // `verify_csrf` never appears as a whole identifier here.
        "pub fn verify_csrf_form_field(token: &str) -> bool {\n\
         \x20   let parsed = token.parse::<u32>().unwrap();\n\
         \x20   parsed > 0\n\
         }\n\
         \n\
         pub fn call_site() -> bool {\n\
         \x20   verify_csrf_form_field(\"abc\")\n\
         }\n",
    )
    .unwrap();

    let cache = CacheManager::new(temp.path());
    Indexer::new(cache, IndexConfig::default())
        .index(temp.path(), false)
        .unwrap();
    temp
}

/// Call one MCP tool and return its parsed result payload.
fn call_tool(root: &Path, tool: &str, args: Value) -> Value {
    let req = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": { "name": tool, "arguments": args }
    });

    let mut out: Vec<u8> = Vec::new();
    run_mcp_server_io_in(
        root,
        Cursor::new(format!("{req}\n").into_bytes()),
        &mut out,
        false,
    )
    .expect("server should not error");

    let line = String::from_utf8(out).unwrap();
    let line = line.lines().find(|l| !l.is_empty()).expect("a response");
    let v: Value = serde_json::from_str(line).unwrap();

    assert!(
        v.get("error").is_none(),
        "{tool} returned an error: {}",
        v["error"]
    );

    // Tool payloads come back as JSON inside content[0].text.
    let text = v["result"]["content"][0]["text"]
        .as_str()
        .expect("content[0].text");
    serde_json::from_str(text).unwrap_or_else(|_| json!({ "raw": text }))
}

#[test]
fn whole_identifier_is_the_default_and_contains_opts_out() {
    let temp = workspace();
    let root = temp.path();

    // Default: whole identifier. `verify_csrf` is only ever a prefix here.
    let default = call_tool(root, "count_occurrences", json!({"pattern": "verify_csrf"}));
    assert_eq!(default["total"], 0, "default must be whole-identifier only");

    // contains: true finds it inside the longer name (definition + call site).
    let substring = call_tool(
        root,
        "count_occurrences",
        json!({"pattern": "verify_csrf", "contains": true}),
    );
    assert!(
        substring["total"].as_u64().unwrap() >= 2,
        "contains:true should find the substring matches, got {}",
        substring["total"]
    );
}

#[test]
fn a_zero_result_carries_a_hint_naming_the_substring_count() {
    let temp = workspace();
    let root = temp.path();

    let res = call_tool(root, "search_code", json!({"pattern": "verify_csrf"}));
    let hint = res["hint"]
        .as_str()
        .unwrap_or_else(|| panic!("expected a hint on a zero result, got: {res}"));

    assert!(
        hint.contains("contains:true"),
        "the hint must name the fix: {hint}"
    );
    assert!(
        hint.contains("substring match"),
        "the hint must name the substring count: {hint}"
    );
}

#[test]
fn a_hit_carries_no_hint() {
    let temp = workspace();
    let root = temp.path();

    let res = call_tool(
        root,
        "search_code",
        json!({"pattern": "verify_csrf_form_field"}),
    );
    assert!(
        res.get("hint").is_none(),
        "a non-empty result must not be annotated: {res}"
    );
}

#[test]
fn bracket_patterns_never_return_a_silent_zero() {
    let temp = workspace();
    let root = temp.path();

    for pattern in ["unwrap()", "parse::<u32>()"] {
        let res = call_tool(
            root,
            "count_occurrences",
            json!({"pattern": pattern, "contains": true}),
        );
        assert!(
            res["total"].as_u64().unwrap() > 0,
            "{pattern:?} with contains:true should match, got {res}"
        );
    }

    // And without contains: the bracket pattern is escaped onto the regex path,
    // with the rewrite reported in `warnings` rather than applied silently.
    let res = call_tool(root, "count_occurrences", json!({"pattern": "unwrap()"}));
    assert!(
        res["total"].as_u64().unwrap() > 0,
        "a bracket pattern must never return a silent 0: {res}"
    );
    let warnings = res["warnings"]
        .as_array()
        .unwrap_or_else(|| panic!("expected warnings explaining the rewrite: {res}"));
    assert!(
        warnings[0].as_str().unwrap().contains("bracket"),
        "the warning must explain the rewrite: {warnings:?}"
    );
}

#[test]
fn list_locations_returns_one_entry_per_match_not_per_file() {
    let temp = workspace();
    let root = temp.path();

    // `verify_csrf_form_field` occurs twice, both in the same file.
    let res = call_tool(
        root,
        "list_locations",
        json!({"pattern": "verify_csrf_form_field"}),
    );

    let locations = res["locations"].as_array().unwrap();
    assert_eq!(
        locations.len(),
        2,
        "expected one entry per MATCH (2), not per file (1): {res}"
    );
    assert_eq!(res["total_locations"], 2);

    // Both entries are in the same file, on different lines.
    let lines: Vec<u64> = locations
        .iter()
        .map(|l| l["line"].as_u64().unwrap())
        .collect();
    assert_ne!(lines[0], lines[1], "the two matches are on distinct lines");
}

#[test]
fn find_references_reports_consistent_counts() {
    let temp = workspace();
    let root = temp.path();

    let res = call_tool(
        root,
        "find_references",
        json!({"pattern": "verify_csrf_form_field"}),
    );

    let total = res["total_references"].as_u64().unwrap();
    let returned = res["returned_count"].as_u64().unwrap();
    let filtered = res["filtered_out"].as_u64().unwrap();
    let paginated = res["pagination"]["total"].as_u64().unwrap();

    assert_eq!(
        total, paginated,
        "total_references must agree with pagination.total"
    );
    assert_eq!(
        res["references"].as_array().unwrap().len() as u64,
        returned,
        "returned_count must equal the references actually returned"
    );
    // The gap is named, not implied. On one page, the three numbers reconcile exactly.
    assert!(!res["has_more"].as_bool().unwrap());
    assert_eq!(
        returned + filtered,
        total,
        "returned_count + filtered_out must reconcile with the raw total"
    );
}

#[test]
fn include_strings_changes_the_count_in_count_mode() {
    let temp = TempDir::new().unwrap();
    std::fs::create_dir_all(temp.path().join("src")).unwrap();
    // One real call site, two occurrences inside a string and a comment.
    std::fs::write(
        temp.path().join("src/lib.rs"),
        "// marker_token is described here\n\
         pub fn marker_token() {}\n\
         pub fn user() {\n\
         \x20   marker_token();\n\
         \x20   let s = \"marker_token\";\n\
         \x20   let _ = s;\n\
         }\n",
    )
    .unwrap();
    let cache = CacheManager::new(temp.path());
    Indexer::new(cache, IndexConfig::default())
        .index(temp.path(), false)
        .unwrap();

    let with = call_tool(
        temp.path(),
        "find_references",
        json!({"pattern": "marker_token", "mode": "count", "include_strings": true}),
    );
    let without = call_tool(
        temp.path(),
        "find_references",
        json!({"pattern": "marker_token", "mode": "count", "include_strings": false}),
    );

    assert!(
        with["count"].as_u64().unwrap() > without["count"].as_u64().unwrap(),
        "include_strings must change the count in count mode: \
         with={} without={}",
        with["count"],
        without["count"]
    );
}

#[test]
fn contains_is_accepted_by_the_argument_validator() {
    let temp = workspace();
    // An unknown argument is rejected; `contains` must not be one.
    let res = call_tool(
        temp.path(),
        "search_code",
        json!({"pattern": "verify_csrf_form_field", "contains": false}),
    );
    assert!(
        res.get("raw").is_none(),
        "contains should be a recognised argument, got: {res}"
    );
}
