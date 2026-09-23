//! `rfx query` must never return a silent zero for a bracket pattern.
//!
//! The 1.7.2 bracket rewrite (`unwrap()` → escaped regex, reported in `warnings`)
//! lived only in the MCP layer. The perf-round field test found the CLI still
//! returning `{"count":0,"timing_ms":19}` for every one of these, with no warning and
//! `total_is_exact: true`:
//!
//! | pattern          | `rfx query … --count` | MCP `search_code mode:count` | ripgrep |
//! |------------------|-----------------------|------------------------------|---------|
//! | `unwrap()`       |                     0 |               1252 + warning |    1252 |
//! | `#[derive(`      |                     0 |               1144 + warning |    1144 |
//! | `RealmId::nil()` |                     0 |                 10 + warning |      10 |
//! | `-> Result<`     |                     0 |               2149 + warning |    2149 |
//!
//! The rewrite now lives in `QueryEngine::search_with_metadata`, so these tests drive
//! the real binary end to end and then check the CLI and the MCP handler agree on
//! every pattern.

use reflex::mcp::run_mcp_server_io_in;
use serde_json::{Value, json};
use std::io::Cursor;
use std::path::Path;
use std::process::{Command, Output};
use tempfile::TempDir;

/// A workspace with one hit for each bracket shape from the field test, one plain
/// identifier, and one identifier that only ever appears as a prefix.
fn fixture() -> TempDir {
    let temp = TempDir::new().unwrap();
    std::fs::create_dir_all(temp.path().join("src")).unwrap();
    std::fs::write(
        temp.path().join("src/lib.rs"),
        "#[derive(Debug, Clone)]\n\
         pub struct RealmId(u32);\n\
         \n\
         impl RealmId {\n\
         \x20   pub fn nil() -> Self {\n\
         \x20       RealmId(0)\n\
         \x20   }\n\
         }\n\
         \n\
         pub fn parse(s: &str) -> Result<RealmId, String> {\n\
         \x20   let n = s.parse::<u32>().unwrap();\n\
         \x20   Ok(RealmId(n))\n\
         }\n\
         \n\
         pub fn default_realm() -> RealmId {\n\
         \x20   RealmId::nil()\n\
         }\n\
         \n\
         pub fn verify_csrf_form_field(token: &str) -> bool {\n\
         \x20   token.parse::<u32>().unwrap() > 0\n\
         }\n",
    )
    .unwrap();
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

/// Index the fixture through the binary, exactly as a user would.
fn indexed_fixture() -> TempDir {
    let temp = fixture();
    let out = rfx(temp.path(), &["index", "--quiet"]);
    assert!(out.status.success(), "rfx index failed: {}", stderr(&out));
    temp
}

/// `rfx query <pattern> --count --json`, parsed. Patterns that begin with `-` go
/// after `--`, because clap would otherwise read them as flags.
fn count_json(root: &Path, pattern: &str, extra: &[&str]) -> Value {
    let mut args = vec!["query"];
    args.extend_from_slice(extra);
    args.extend_from_slice(&["--count", "--json", "--", pattern]);
    let out = rfx(root, &args);
    assert!(out.status.success(), "rfx query failed: {}", stderr(&out));
    serde_json::from_str(stdout(&out).trim()).unwrap_or_else(|e| {
        panic!(
            "count --json must print one JSON object: {e}\n{}",
            stdout(&out)
        )
    })
}

/// Call one MCP tool in-process and return its parsed payload.
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

const BRACKET_PATTERNS: &[&str] = &["unwrap()", "#[derive(", "RealmId::nil()", "-> Result<"];

#[test]
fn count_json_reports_bracket_matches_with_a_warning() {
    let temp = indexed_fixture();
    for p in BRACKET_PATTERNS {
        let v = count_json(temp.path(), p, &[]);
        let count = v["count"].as_u64().unwrap_or_else(|| panic!("{p}: {v}"));
        assert!(count > 0, "{p}: silent zero: {v}");
        let w = v["warnings"]
            .as_array()
            .unwrap_or_else(|| panic!("{p}: count --json must carry warnings: {v}"));
        assert!(
            w[0].as_str().unwrap().contains("bracket"),
            "{p}: warning must explain the rewrite: {w:?}"
        );
    }
}

#[test]
fn list_json_carries_the_warnings_field() {
    let temp = indexed_fixture();
    let out = rfx(temp.path(), &["query", "--json", "unwrap()"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let v: Value = serde_json::from_str(stdout(&out).trim()).unwrap();
    assert!(
        v["pagination"]["total"].as_u64().unwrap() > 0,
        "list mode must find the hits: {v}"
    );
    assert!(
        v["warnings"][0].as_str().unwrap().contains("bracket"),
        "list --json must carry warnings: {v}"
    );
}

#[test]
fn plain_mode_finds_results_and_warns_on_stderr() {
    let temp = indexed_fixture();
    let out = rfx(temp.path(), &["query", "unwrap()"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let so = stdout(&out);
    assert!(so.contains("Found"), "plain mode must list hits:\n{so}");
    assert!(!so.contains("No results found"), "silent zero:\n{so}");
    assert!(
        stderr(&out).contains("bracket"),
        "plain mode must warn on stderr:\n{}",
        stderr(&out)
    );

    // `--count` without `--json` too: the summary line, plus the warning on stderr.
    let out = rfx(temp.path(), &["query", "--count", "unwrap()"]);
    assert!(stdout(&out).contains("Found 2 results"), "{}", stdout(&out));
    assert!(stderr(&out).contains("bracket"), "{}", stderr(&out));
}

#[test]
fn paths_json_puts_the_warning_on_stderr() {
    let temp = indexed_fixture();
    let out = rfx(temp.path(), &["query", "--paths", "--json", "unwrap()"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let v: Value = serde_json::from_str(stdout(&out).trim()).unwrap();
    assert_eq!(v.as_array().map(Vec::len), Some(1), "one file: {v}");
    assert!(stderr(&out).contains("bracket"), "{}", stderr(&out));
}

#[test]
fn a_plain_identifier_carries_no_warning() {
    let temp = indexed_fixture();
    let v = count_json(temp.path(), "RealmId", &[]);
    assert!(v["count"].as_u64().unwrap() > 0, "{v}");
    assert!(v.get("warnings").is_none(), "no rewrite, no warning: {v}");
    assert!(v.get("hint").is_none(), "hits, no hint: {v}");
}

#[test]
fn contains_and_regex_modes_are_not_rewritten() {
    let temp = indexed_fixture();
    let v = count_json(temp.path(), "unwrap()", &["--contains"]);
    assert!(v["count"].as_u64().unwrap() > 0, "{v}");
    assert!(
        v.get("warnings").is_none(),
        "--contains already substring: {v}"
    );

    let v = count_json(temp.path(), r"unwrap\(\)", &["--regex"]);
    assert!(v["count"].as_u64().unwrap() > 0, "{v}");
    assert!(v.get("warnings").is_none(), "--regex is explicit: {v}");
}

#[test]
fn a_zero_result_prints_the_substring_hint() {
    let temp = indexed_fixture();

    // `verify_csrf` only ever appears as a prefix of `verify_csrf_form_field`.
    let v = count_json(temp.path(), "verify_csrf", &[]);
    assert_eq!(v["count"], 0, "{v}");
    let hint = v["hint"]
        .as_str()
        .unwrap_or_else(|| panic!("count --json must carry the hint: {v}"));
    assert!(hint.contains("substring match"), "{hint}");
    assert!(hint.contains("--contains"), "{hint}");

    let out = rfx(temp.path(), &["query", "verify_csrf"]);
    assert!(
        stdout(&out).contains("No results found"),
        "{}",
        stdout(&out)
    );
    assert!(
        stderr(&out).contains("substring match"),
        "plain mode prints the hint on stderr:\n{}",
        stderr(&out)
    );
}

/// A pattern beginning with `-` is a flag to clap. `--pattern` is the explicit form;
/// `--` still works.
#[test]
fn a_leading_dash_pattern_works_via_pattern_flag_and_double_dash() {
    let temp = indexed_fixture();
    let out = rfx(
        temp.path(),
        &["query", "--pattern", "-> Result<", "--count", "--json"],
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let flag: Value = serde_json::from_str(stdout(&out).trim()).unwrap();
    assert_eq!(flag["count"], 1, "{flag}");

    let dd = count_json(temp.path(), "-> Result<", &[]);
    assert_eq!(flag["count"], dd["count"], "--pattern and -- must agree");

    // Both forms at once is an error, not a silent choice.
    let out = rfx(temp.path(), &["query", "--pattern", "x", "--", "y"]);
    assert!(
        !out.status.success(),
        "conflicting patterns must be rejected"
    );
}

/// The CLI and the MCP handler run the same engine; they must never disagree.
#[test]
fn cli_and_mcp_counts_agree_on_every_pattern() {
    let temp = indexed_fixture();
    let root = temp.path();

    let cases: Vec<(&str, bool)> = BRACKET_PATTERNS
        .iter()
        .map(|p| (*p, false))
        .chain([("RealmId", false), ("nil", false), ("verify_csrf", true)])
        .collect();

    for (pattern, contains) in cases {
        let extra: &[&str] = if contains { &["--contains"] } else { &[] };
        let cli = count_json(root, pattern, extra);
        let mcp = call_tool(
            root,
            "search_code",
            json!({"pattern": pattern, "mode": "count", "contains": contains}),
        );
        assert_eq!(
            cli["count"], mcp["count"],
            "{pattern} (contains={contains}): CLI {cli} vs MCP {mcp}"
        );
        assert_eq!(
            cli.get("warnings").is_some(),
            mcp.get("warnings").is_some(),
            "{pattern}: warning presence must match: CLI {cli} vs MCP {mcp}"
        );
        assert_eq!(
            cli.get("hint").is_some(),
            mcp.get("hint").is_some(),
            "{pattern}: hint presence must match: CLI {cli} vs MCP {mcp}"
        );
    }
}
