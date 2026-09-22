//! Early termination for list-mode searches must never change WHAT a page
//! contains — only how much work it takes to produce it.
//!
//! A search with a `limit` stops verifying candidates once `offset + limit`
//! results exist. Because candidate files are verified in path order and lines
//! within a file ascend, the page is identical to the same slice of a full run.
//! Count mode, no-limit searches and `find_references` still verify everything
//! and keep an exact total.

mod test_helpers;

use reflex::mcp::run_mcp_server_io_in;
use reflex::query::{QueryEngine, QueryFilter};
use reflex::{CacheManager, SearchResult};
use serde_json::{Value, json};
use std::io::Cursor;
use std::path::Path;
use test_helpers::setup_corpus;

/// Patterns with plenty of hits in `tests/corpus`, plus a few narrow ones.
const PATTERNS: &[(&str, bool)] = &[
    ("main", false),
    ("return", false),
    ("self", false),
    ("import", false),
    ("class", false),
    ("test", false),
    ("Controller", false),
    ("zzqx_no_such_token", false),
    (r"fn \w+\(", true),
    (r"(get|set)_\w+", true),
    (r"class \w+", true),
    (r"import .*", true),
];

fn engine() -> QueryEngine {
    QueryEngine::new(CacheManager::new(setup_corpus()))
}

fn key(r: &SearchResult) -> (String, usize, String) {
    (r.path.clone(), r.span.start_line, r.preview.clone())
}

fn full_run(pattern: &str, base: &QueryFilter) -> Vec<SearchResult> {
    let mut filter = base.clone();
    filter.limit = None;
    filter.offset = None;
    engine()
        .search_with_metadata(pattern, filter)
        .expect("full run")
        .results
        .into_iter()
        .flat_map(|fg| {
            let path = fg.path.clone();
            let lang = fg.language;
            fg.matches.into_iter().map(move |m| SearchResult {
                path: path.clone(),
                lang,
                kind: m.kind,
                symbol: m.symbol,
                span: m.span,
                preview: m.preview,
                dependencies: None,
            })
        })
        .collect()
}

fn page(
    pattern: &str,
    base: &QueryFilter,
    offset: usize,
    limit: usize,
) -> (Vec<SearchResult>, reflex::models::PaginationInfo) {
    let mut filter = base.clone();
    filter.limit = Some(limit);
    filter.offset = Some(offset);
    let response = engine()
        .search_with_metadata(pattern, filter)
        .expect("page run");
    let flat: Vec<SearchResult> = response
        .results
        .iter()
        .flat_map(|fg| {
            fg.matches.iter().map(move |m| SearchResult {
                path: fg.path.clone(),
                lang: fg.language,
                kind: m.kind.clone(),
                symbol: m.symbol.clone(),
                span: m.span.clone(),
                preview: m.preview.clone(),
                dependencies: None,
            })
        })
        .collect();
    (flat, response.pagination)
}

fn base_filter(use_regex: bool) -> QueryFilter {
    QueryFilter {
        use_regex,
        suppress_output: true,
        ..Default::default()
    }
}

fn assert_pages_match(pattern: &str, base: &QueryFilter) {
    let full = full_run(pattern, base);
    let full_keys: Vec<_> = full.iter().map(key).collect();

    for &(offset, limit) in &[
        (0usize, 1usize),
        (0, 5),
        (3, 4),
        (0, 100),
        (7, 1),
        (10_000, 5),
    ] {
        let (got, pagination) = page(pattern, base, offset, limit);
        let want: Vec<_> = full_keys.iter().skip(offset).take(limit).cloned().collect();
        let got_keys: Vec<_> = got.iter().map(key).collect();
        assert_eq!(
            got_keys, want,
            "pattern {pattern:?} offset {offset} limit {limit}: page differs from full run"
        );

        // The reported total is exact, or an honest lower bound with an upper bound.
        if pagination.total_is_exact {
            assert_eq!(pagination.total, full.len(), "{pattern:?} exact total");
            assert_eq!(
                pagination.has_more,
                full.len() > offset + got.len(),
                "{pattern:?} has_more (exact)"
            );
            assert!(pagination.approx_total.is_none());
        } else {
            assert!(
                pagination.total >= offset + got.len(),
                "{pattern:?}: lower bound {} < page end {}",
                pagination.total,
                offset + got.len()
            );
            assert!(
                pagination.total <= full.len(),
                "{pattern:?}: lower bound above truth"
            );
            assert!(
                pagination.has_more,
                "{pattern:?}: inexact total must set has_more"
            );
            let approx = pagination.approx_total.expect("upper bound when inexact");
            assert!(
                approx >= full.len(),
                "{pattern:?}: approx {approx} < true {}",
                full.len()
            );
        }
    }
}

#[test]
fn every_page_equals_the_same_slice_of_a_full_run() {
    for &(pattern, use_regex) in PATTERNS {
        assert_pages_match(pattern, &base_filter(use_regex));
    }
}

#[test]
fn pages_match_under_language_glob_file_and_text_filters() {
    let variants: Vec<QueryFilter> = vec![
        QueryFilter {
            language: Some(reflex::Language::Rust),
            ..base_filter(false)
        },
        QueryFilter {
            glob_patterns: vec!["**/*.py".to_string()],
            ..base_filter(false)
        },
        QueryFilter {
            exclude_patterns: vec!["**/rust/**".to_string(), "**/*.md".to_string()],
            ..base_filter(false)
        },
        QueryFilter {
            file_pattern: Some("edge_cases".to_string()),
            ..base_filter(false)
        },
        QueryFilter {
            exclude_text: true,
            ..base_filter(false)
        },
        QueryFilter {
            paths_only: true,
            ..base_filter(false)
        },
        QueryFilter {
            use_contains: true,
            ..base_filter(false)
        },
        QueryFilter {
            language: Some(reflex::Language::Python),
            paths_only: true,
            ..base_filter(true)
        },
    ];
    for base in &variants {
        for pattern in ["main", "return", "self", "test", r"def \w+"] {
            if base.use_regex != pattern.contains('\\') {
                continue;
            }
            assert_pages_match(pattern, base);
        }
    }
}

#[test]
fn count_mode_and_no_limit_totals_are_exact() {
    for &(pattern, use_regex) in PATTERNS {
        let mut filter = base_filter(use_regex);
        filter.limit = None;
        let r = engine().search_with_metadata(pattern, filter).unwrap();
        assert!(r.pagination.total_is_exact, "{pattern:?}");
        assert!(r.pagination.approx_total.is_none(), "{pattern:?}");
        let flat: usize = r.results.iter().map(|fg| fg.matches.len()).sum();
        assert_eq!(r.pagination.total, flat, "{pattern:?}");
        assert!(!r.pagination.has_more, "{pattern:?}");
    }
}

#[test]
fn require_exact_total_verifies_everything_even_with_a_limit() {
    for &(pattern, use_regex) in PATTERNS {
        let full = full_run(pattern, &base_filter(use_regex));
        let mut filter = base_filter(use_regex);
        filter.limit = Some(1);
        filter.require_exact_total = true;
        let r = engine().search_with_metadata(pattern, filter).unwrap();
        assert!(r.pagination.total_is_exact, "{pattern:?}");
        assert_eq!(r.pagination.total, full.len(), "{pattern:?}");
        assert_eq!(r.pagination.has_more, full.len() > 1, "{pattern:?}");
    }
}

#[test]
fn symbol_mode_keeps_exact_totals() {
    let mut filter = base_filter(false);
    filter.symbols_mode = true;
    filter.limit = Some(2);
    let r = engine().search_with_metadata("main", filter).unwrap();
    assert!(r.pagination.total_is_exact);
}

// ---- MCP-level contract -------------------------------------------------------

fn call_tool(root: &Path, tool: &str, args: Value) -> Value {
    let req = json!({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": { "name": tool, "arguments": args }
    });
    let mut out: Vec<u8> = Vec::new();
    run_mcp_server_io_in(
        root,
        Cursor::new(format!("{req}\n").into_bytes()),
        &mut out,
        false,
    )
    .unwrap();
    let text = String::from_utf8(out).unwrap();
    let line = text.lines().find(|l| !l.is_empty()).unwrap();
    let v: Value = serde_json::from_str(line).unwrap();
    assert!(v.get("error").is_none(), "{tool} errored: {}", v["error"]);
    let payload = v["result"]["content"][0]["text"].as_str().unwrap();
    serde_json::from_str(payload).unwrap()
}

#[test]
fn mcp_count_mode_matches_list_locations_and_a_full_page() {
    let root = setup_corpus();
    for pattern in ["main", "return", "self", "Controller"] {
        let count = call_tool(
            root,
            "search_code",
            json!({"pattern": pattern, "mode": "count"}),
        );
        let locations = call_tool(root, "list_locations", json!({"pattern": pattern}));
        let page = call_tool(root, "search_code", json!({"pattern": pattern, "limit": 1}));

        let n = count["count"].as_u64().unwrap();
        assert_eq!(
            locations["total_locations"].as_u64().unwrap(),
            n,
            "{pattern}: list_locations"
        );
        assert_eq!(
            locations["locations"].as_array().unwrap().len() as u64,
            n,
            "{pattern}"
        );

        // The one-result page reports exactness honestly.
        assert_eq!(
            page["returned_count"].as_u64().unwrap(),
            n.min(1),
            "{pattern}"
        );
        if page["total_is_exact"].as_bool().unwrap() {
            assert_eq!(
                page["total_count"].as_u64().unwrap(),
                n,
                "{pattern}: exact total"
            );
        } else {
            assert!(
                page["total_count"].as_u64().unwrap() <= n,
                "{pattern}: lower bound"
            );
            assert!(
                page["approx_total"].as_u64().unwrap() >= n,
                "{pattern}: upper bound"
            );
            assert_eq!(page["has_more"], true, "{pattern}");
        }
    }
}

#[test]
fn mcp_find_references_total_is_exact_and_matches_a_full_run() {
    let root = setup_corpus();
    let full = call_tool(
        root,
        "find_references",
        json!({"pattern": "main", "limit": 500, "include_strings": true}),
    );
    let paged = call_tool(
        root,
        "find_references",
        json!({"pattern": "main", "limit": 1, "include_strings": true}),
    );
    assert_eq!(paged["total_references"], full["total_references"]);
    assert_eq!(paged["pagination"]["total_is_exact"], true);
    assert_eq!(paged["returned_count"], 1);
}
