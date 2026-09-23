//! Case-insensitive search: `(?i)` regexes and `ignore_case` literals must use the
//! trigram index, agree with the regex crate line for line, and say so.
//!
//! Before 1.8.0 the literal extractor discarded every literal the moment it saw an
//! `i` flag, so `(?i)kubernetes` scanned every line of every file (387 ms on the
//! Kubernetes checkout against ripgrep's 166 ms) and printed a "has no literals"
//! warning for a pattern with a 10-character literal.

use reflex::CacheManager;
use reflex::mcp::run_mcp_server_io_in;
use reflex::models::IndexPath;
use reflex::query::{QueryEngine, QueryFilter};
use serde_json::{Value, json};
use std::io::Cursor;
use std::path::Path;
use std::process::{Command, Output};
use tempfile::TempDir;

/// Every casing of one identifier, the two Unicode folds the regex crate applies
/// to ASCII `k` and `s`, a non-ASCII literal, and some filler.
const FIXTURE: &str = "pub struct RealmId(u32);\n\
    let realmId = 1;\n\
    const REALMID: u32 = 2;\n\
    let realm_id = 3;\n\
    let kelvin = 4;\n\
    let \u{212A}elvin = 5;\n\
    let status = 6;\n\
    let \u{017F}tatus = 7;\n\
    let straße = 8;\n\
    let STRASSE = 9;\n\
    let user_id = 10;\n\
    let other = 11;\n\
    fn get_realm() {}\n\
    fn GET_REALM() {}\n";

fn fixture() -> TempDir {
    let temp = TempDir::new().unwrap();
    std::fs::create_dir_all(temp.path().join("src")).unwrap();
    std::fs::write(temp.path().join("src/lib.rs"), FIXTURE).unwrap();
    temp
}

fn rfx(root: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rfx"))
        .current_dir(root)
        .args(args)
        .output()
        .expect("spawn rfx")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn indexed_fixture() -> TempDir {
    let temp = fixture();
    let out = rfx(temp.path(), &["index", "--quiet"]);
    assert!(out.status.success(), "rfx index failed: {}", stderr(&out));
    temp
}

/// Lines of the fixture a regex matches — the same engine ripgrep uses.
fn oracle(pattern: &str) -> usize {
    let re = regex::Regex::new(pattern).unwrap();
    FIXTURE.lines().filter(|l| re.is_match(l)).count()
}

fn engine(root: &Path) -> QueryEngine {
    QueryEngine::new(CacheManager::new(root))
}

fn filter() -> QueryFilter {
    QueryFilter {
        limit: None,
        suppress_output: true,
        collect_timings: true,
        ..Default::default()
    }
}

fn count(response: &reflex::models::QueryResponse) -> usize {
    response.results.iter().map(|fg| fg.matches.len()).sum()
}

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
    assert!(v.get("error").is_none(), "{tool} errored: {}", v["error"]);
    let text = v["result"]["content"][0]["text"].as_str().unwrap();
    serde_json::from_str(text).unwrap_or_else(|_| json!({ "raw": text }))
}

// ==================== Engine ====================

/// Every `(?i)` shape counts exactly what the regex crate counts, and a pattern
/// with a foldable literal reports the trigram path with no warning.
#[test]
fn case_insensitive_regex_matches_the_regex_crate_and_uses_the_index() {
    let temp = indexed_fixture();
    let engine = engine(temp.path());

    let trigram_shapes = [
        "(?i)realmid",
        "(?i)realm_id",
        "(?i)kelvin",   // Kelvin-sign fold
        "(?i)status",   // long-s fold
        "(?i:Realm)Id", // scoped flag
        "(?i)REALM(?-i)Id",
        "(?i)get_realm",
        "(?im)^let realm",
    ];
    for pattern in trigram_shapes {
        let response = engine
            .search_with_metadata(
                pattern,
                QueryFilter {
                    use_regex: true,
                    ..filter()
                },
            )
            .unwrap();
        assert_eq!(count(&response), oracle(pattern), "{pattern}");
        assert!(oracle(pattern) > 0, "{pattern}: fixture must exercise it");
        let timings = response.timings.as_ref().expect("timings requested");
        assert_eq!(timings.index_path, IndexPath::Trigram, "{pattern}");
        assert!(
            response.warnings.is_empty(),
            "{pattern}: no scan warning, got {:?}",
            response.warnings
        );
    }

    // The Kelvin-sign and long-s lines really are in the oracle's count.
    assert_eq!(oracle("(?i)kelvin"), 2);
    assert_eq!(oracle("(?i)status"), 2);
}

/// A `(?i)` pattern with no usable literal still scans, still counts correctly,
/// and now says so in `warnings` rather than only on stderr.
#[test]
fn case_insensitive_regex_without_a_literal_scans_and_warns() {
    let temp = indexed_fixture();
    let engine = engine(temp.path());

    for (pattern, reason) in [
        // `\w+_id` has the literal `_id`; this one has no 3-byte literal at all.
        (r"(?i)\w+_?id", "no literals"),
        ("(?i)straße", "non-ASCII case-insensitive literal"),
    ] {
        let response = engine
            .search_with_metadata(
                pattern,
                QueryFilter {
                    use_regex: true,
                    ..filter()
                },
            )
            .unwrap();
        assert_eq!(count(&response), oracle(pattern), "{pattern}");
        assert!(oracle(pattern) > 0, "{pattern}");
        let timings = response.timings.as_ref().unwrap();
        assert_eq!(timings.index_path, IndexPath::Scan, "{pattern}");
        assert_eq!(
            response.warnings.len(),
            1,
            "{pattern}: {:?}",
            response.warnings
        );
        assert!(
            response.warnings[0].contains(reason),
            "{pattern}: {}",
            response.warnings[0]
        );
    }
}

/// `ignore_case` without a regex: whole-identifier and substring semantics are
/// kept, the query runs on the index, and results still say `text_match`.
#[test]
fn ignore_case_literal_search() {
    let temp = indexed_fixture();
    let engine = engine(temp.path());

    // Whole identifier: `realmid` is RealmId, realmId, REALMID — not realm_id.
    let response = engine
        .search_with_metadata(
            "realmid",
            QueryFilter {
                ignore_case: true,
                ..filter()
            },
        )
        .unwrap();
    assert_eq!(count(&response), 3);
    assert_eq!(count(&response), oracle(r"(?i)\brealmid\b"));
    assert_eq!(
        response.timings.as_ref().unwrap().index_path,
        IndexPath::Trigram
    );
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);
    for fg in &response.results {
        for m in &fg.matches {
            assert_eq!(
                m.kind,
                reflex::models::SymbolKind::Unknown("text_match".into()),
                "{m:?}"
            );
        }
    }

    // Substring: `realm` is inside all four spellings plus get_realm / GET_REALM.
    let response = engine
        .search_with_metadata(
            "realm",
            QueryFilter {
                ignore_case: true,
                use_contains: true,
                ..filter()
            },
        )
        .unwrap();
    assert_eq!(count(&response), oracle("(?i)realm"));
    assert_eq!(count(&response), 6);

    // Whole identifier `realm` matches nothing (underscore is a word character);
    // the substring hint is not produced under ignore_case — documented.
    let response = engine
        .search_with_metadata(
            "realm",
            QueryFilter {
                ignore_case: true,
                ..filter()
            },
        )
        .unwrap();
    assert_eq!(count(&response), 0);
    assert!(response.hint.is_none());

    // Case-sensitive stays case-sensitive.
    let response = engine.search_with_metadata("realmid", filter()).unwrap();
    assert_eq!(count(&response), 0);

    // Brackets + ignore_case: bracket rewrite first (substring + warning), then fold.
    let response = engine
        .search_with_metadata(
            "realmid(u32)",
            QueryFilter {
                ignore_case: true,
                ..filter()
            },
        )
        .unwrap();
    assert_eq!(count(&response), 1);
    assert_eq!(response.warnings.len(), 1);
    assert!(response.warnings[0].contains("bracket"));
}

/// A two-character `ignore_case` literal scans, as the case-sensitive path does,
/// and does not warn about "no literals".
#[test]
fn short_ignore_case_literal_scans_silently() {
    let temp = indexed_fixture();
    let response = engine(temp.path())
        .search_with_metadata(
            "id",
            QueryFilter {
                ignore_case: true,
                ..filter()
            },
        )
        .unwrap();
    assert_eq!(count(&response), oracle(r"(?i)\bid\b"));
    assert_eq!(
        response.timings.as_ref().unwrap().index_path,
        IndexPath::Scan
    );
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);
}

/// The broad-query guard judges the literal, not the regex the engine built
/// from it: `-i fn` on a large index is refused like `fn` is.
#[test]
fn broad_query_guard_sees_through_the_ignore_case_rewrite() {
    let temp = indexed_fixture();
    let guarded = QueryFilter {
        ignore_case: true,
        test_large_index_threshold: Some(0),
        test_short_pattern_threshold: Some(3),
        ..filter()
    };
    let err = engine(temp.path())
        .search_with_metadata("id", guarded.clone())
        .expect_err("short ignore_case literal must hit the guard");
    assert!(err.to_string().contains("too broad"), "{err}");

    // A real regex is still exempt (it has its own literal extraction).
    let ok = engine(temp.path())
        .search_with_metadata(
            "(?i)id",
            QueryFilter {
                use_regex: true,
                ignore_case: false,
                ..guarded
            },
        )
        .unwrap();
    assert_eq!(count(&ok), oracle("(?i)id"));
}

// ==================== CLI and MCP ====================

/// The CLI and the MCP handler agree, the CLI prints no scan warning for a
/// foldable literal, and `--timing --json` names the index path.
#[test]
fn cli_and_mcp_agree_on_case_insensitive_counts() {
    let temp = indexed_fixture();
    let root = temp.path();

    // ignore_case literal: `-i`.
    let out = rfx(root, &["query", "realmid", "-i", "--count", "--json"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let cli: Value = serde_json::from_str(stdout(&out).trim()).unwrap();
    assert_eq!(cli["count"], json!(3), "{cli}");
    let mcp = call_tool(
        root,
        "search_code",
        json!({ "pattern": "realmid", "ignore_case": true, "mode": "count" }),
    );
    assert_eq!(mcp["count"], json!(3), "{mcp}");
    assert!(mcp.get("warnings").is_none(), "{mcp}");

    // Long flag and contains.
    let out = rfx(
        root,
        &[
            "query",
            "realm",
            "--ignore-case",
            "--contains",
            "--count",
            "--json",
        ],
    );
    let cli: Value = serde_json::from_str(stdout(&out).trim()).unwrap();
    assert_eq!(cli["count"], json!(6));
    let mcp = call_tool(
        root,
        "count_occurrences",
        json!({ "pattern": "realm", "ignore_case": true, "contains": true }),
    );
    assert_eq!(mcp["total"], json!(6), "{mcp}");
    let mcp = call_tool(
        root,
        "list_locations",
        json!({ "pattern": "realm", "ignore_case": true, "contains": true }),
    );
    assert_eq!(mcp["total_locations"], json!(6), "{mcp}");

    // `(?i)` regex: no "no literals" warning anywhere, index path reported.
    let out = rfx(root, &["query", "(?i)realmid", "--regex", "--count"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        !stderr(&out).contains("no literals"),
        "stderr must not warn: {}",
        stderr(&out)
    );
    assert!(stdout(&out).contains("3"), "{}", stdout(&out));

    let out = rfx(
        root,
        &["query", "(?i)realmid", "--regex", "--timing", "--json"],
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let v: Value = serde_json::from_str(stdout(&out).trim()).unwrap();
    assert_eq!(v["timings"]["index_path"], json!("trigram"), "{v}");
    assert!(v.get("warnings").is_none(), "{v}");

    let mcp = call_tool(
        root,
        "search_regex",
        json!({ "pattern": "(?i)realmid", "mode": "count" }),
    );
    assert_eq!(mcp["count"], json!(3), "{mcp}");
    assert!(mcp.get("warnings").is_none(), "{mcp}");

    // `ignore_case` on search_regex prepends `(?i)`.
    let mcp = call_tool(
        root,
        "search_regex",
        json!({ "pattern": "realm_?id", "ignore_case": true, "mode": "count" }),
    );
    assert_eq!(mcp["count"], json!(4), "{mcp}");

    // `\w+_id` carries the literal `_id`, so even it uses the index now.
    let out = rfx(
        root,
        &["query", r"(?i)\w+_id", "--regex", "--timing", "--json"],
    );
    let v: Value = serde_json::from_str(stdout(&out).trim()).unwrap();
    assert_eq!(v["timings"]["index_path"], json!("trigram"), "{v}");
    assert_eq!(
        v["pagination"]["total"],
        json!(oracle(r"(?i)\w+_id")),
        "{v}"
    );

    // A literal-free `(?i)` regex still warns, on stderr and in `warnings`.
    let out = rfx(root, &["query", r"(?i)\w+_?id", "--regex", "--count"]);
    assert!(stderr(&out).contains("no literals"), "{}", stderr(&out));
    let mcp = call_tool(
        root,
        "search_regex",
        json!({ "pattern": r"(?i)\w+_?id", "mode": "count" }),
    );
    assert_eq!(mcp["count"], json!(oracle(r"(?i)\w+_?id")), "{mcp}");
    assert!(
        mcp["warnings"][0].as_str().unwrap().contains("no literals"),
        "{mcp}"
    );
}

/// `paths: true` returns paths and a count, not preview rows.
#[test]
fn mcp_paths_mode_is_compact() {
    let temp = indexed_fixture();
    let mcp = call_tool(
        temp.path(),
        "search_code",
        json!({ "pattern": "realm", "ignore_case": true, "contains": true, "paths": true }),
    );
    assert_eq!(mcp["paths"], json!(["src/lib.rs"]), "{mcp}");
    assert_eq!(mcp["total_files"], json!(1), "{mcp}");
    assert_eq!(mcp["status"], json!("fresh"), "{mcp}");
    assert_eq!(mcp["can_trust_results"], json!(true), "{mcp}");
    for absent in ["rows", "columns", "results", "pagination", "has_more"] {
        assert!(mcp.get(absent).is_none(), "{absent} must be absent: {mcp}");
    }

    let mcp = call_tool(
        temp.path(),
        "search_regex",
        json!({ "pattern": "(?i)realm", "paths": true }),
    );
    assert_eq!(mcp["paths"], json!(["src/lib.rs"]), "{mcp}");
    assert!(mcp.get("rows").is_none(), "{mcp}");

    // Bracket rewrite warnings survive the compact shape.
    let mcp = call_tool(
        temp.path(),
        "search_code",
        json!({ "pattern": "RealmId(u32)", "paths": true }),
    );
    assert_eq!(mcp["total_files"], json!(1), "{mcp}");
    assert!(
        mcp["warnings"][0].as_str().unwrap().contains("bracket"),
        "{mcp}"
    );
}
