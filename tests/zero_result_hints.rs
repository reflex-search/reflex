//! A zero result names ONE cause, chosen from the filter in a fixed order.
//!
//! The 1.8.0 field test on Hearth: `count_occurrences {pattern:"runs-on",
//! file:".github/"}` answered 0 with "6 candidate file(s) were lock or generated
//! files" — a repo-wide count that ignored the `file` filter. The true cause was a
//! hidden path. An agent added `include_locks:true`, got 0 again, and concluded the
//! thing did not exist.

use reflex::mcp::run_mcp_server_io_in;
use reflex::models::IndexConfig;
use reflex::{CacheManager, Indexer};
use serde_json::{Value, json};
use std::fs;
use std::io::Cursor;
use std::path::Path;
use tempfile::TempDir;

fn fixture() -> TempDir {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    let w = |rel: &str, body: &str| {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, body).unwrap();
    };
    w(
        ".github/workflows/ci.yml",
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n",
    );
    // A lock file that also holds the hidden-path token, so rule 1 must beat rule 3.
    w(
        "Cargo.lock",
        "[[package]]\nname = \"runs-on\"\nchecksum = \"abc\"\n",
    );
    w(
        "vendor/Cargo.lock",
        "[[package]]\nname = \"x\"\nchecksum = \"def\"\n",
    );
    w("src/lib.rs", "pub fn verify_csrf_form_field() {}\n");
    w("tests/certs_jwks.rs", "fn realm() { RealmId::nil(); }\n");
    let mut png = vec![0x89u8, b'P', b'N', b'G', 0, 0];
    png.extend_from_slice(b"RealmId");
    fs::write(root.join("image.png"), png).unwrap();
    temp
}

fn index(root: &Path, config: IndexConfig) {
    Indexer::new(CacheManager::new(root), config)
        .index(root, false)
        .unwrap();
}

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
    let v: Value = serde_json::from_str(text.lines().find(|l| !l.is_empty()).unwrap()).unwrap();
    assert!(v.get("error").is_none(), "{tool} errored: {}", v["error"]);
    let payload = v["result"]["content"][0]["text"].as_str().unwrap();
    serde_json::from_str(payload).unwrap_or_else(|_| json!({ "raw": payload }))
}

fn count(root: &Path, args: Value) -> Value {
    call_tool(root, "count_occurrences", args)
}

#[test]
fn a_hidden_path_filter_is_the_cause_even_when_a_lock_file_holds_the_token() {
    let temp = fixture();
    index(temp.path(), IndexConfig::default());

    let r = count(
        temp.path(),
        json!({ "pattern": "runs-on", "contains": true, "file": ".github/" }),
    );
    assert_eq!(r["total"], 0, "{r}");
    assert_eq!(r["excluded_reason"], "hidden", "{r}");
    let hint = r["hint"].as_str().unwrap();
    assert!(hint.contains("grep"), "{hint}");
    assert!(hint.contains(".github"), "{hint}");
    assert!(!hint.contains("lock"), "must not blame lock files: {hint}");
    assert!(r.get("excluded_by_default").is_none(), "{r}");

    // A glob that names the hidden directory is the same cause.
    let r = count(
        temp.path(),
        json!({ "pattern": "runs-on", "contains": true, "glob": ["**/.github/**"] }),
    );
    assert_eq!(r["excluded_reason"], "hidden", "{r}");
}

#[test]
fn a_lock_file_filter_gets_the_scoped_lock_count() {
    let temp = fixture();
    index(temp.path(), IndexConfig::default());

    let r = count(
        temp.path(),
        json!({ "pattern": "checksum", "contains": true, "file": "Cargo.lock" }),
    );
    assert_eq!(r["total"], 0, "{r}");
    assert_eq!(r["excluded_reason"], "lock_or_generated", "{r}");
    assert_eq!(
        r["excluded_by_default"], 2,
        "both lock files under the filter: {r}"
    );
    assert!(r["hint"].as_str().unwrap().contains("include_locks"), "{r}");

    let r = count(
        temp.path(),
        json!({ "pattern": "checksum", "contains": true, "file": "vendor/" }),
    );
    assert_eq!(r["excluded_by_default"], 1, "scoped to vendor/: {r}");

    let r = count(
        temp.path(),
        json!({ "pattern": "checksum", "contains": true, "file": "Cargo.lock", "include_locks": true }),
    );
    assert_eq!(r["total"], 2, "{r}");
    assert!(r.get("excluded_reason").is_none(), "{r}");
    assert!(r.get("hint").is_none(), "{r}");
}

#[test]
fn a_deleted_and_reindexed_file_is_not_indexed_not_lock() {
    let temp = fixture();
    let root = temp.path();
    index(root, IndexConfig::default());
    fs::remove_file(root.join("tests/certs_jwks.rs")).unwrap();
    let idx = call_tool(root, "index_project", json!({}));
    assert_eq!(idx["deleted_files"], 1, "{idx}");

    let r = call_tool(
        root,
        "search_code",
        json!({ "pattern": "RealmId", "file": "tests/certs_jwks.rs", "mode": "count" }),
    );
    assert_eq!(r["count"], 0, "{r}");
    assert_eq!(r["excluded_reason"], "not_indexed", "{r}");
    let hint = r["hint"].as_str().unwrap();
    assert!(hint.contains("not on disk"), "{hint}");
    assert!(!hint.contains("lock"), "{hint}");
    assert!(
        r.get("warning").is_none(),
        "no ghost warning after a reindex: {r}"
    );
}

#[test]
fn a_binary_file_on_disk_is_not_indexed_with_the_reason() {
    let temp = fixture();
    index(temp.path(), IndexConfig::default());
    let r = count(
        temp.path(),
        json!({ "pattern": "RealmId", "file": "image.png" }),
    );
    assert_eq!(r["excluded_reason"], "not_indexed", "{r}");
    assert!(r["hint"].as_str().unwrap().contains("binary"), "{r}");
}

#[test]
fn a_whole_identifier_zero_keeps_the_substring_hint() {
    let temp = fixture();
    index(temp.path(), IndexConfig::default());
    let r = count(temp.path(), json!({ "pattern": "verify_csrf" }));
    assert_eq!(r["total"], 0, "{r}");
    assert_eq!(r["excluded_reason"], "whole_identifier", "{r}");
    assert!(r["hint"].as_str().unwrap().contains("contains:true"), "{r}");
}

#[test]
fn a_plain_miss_has_no_reason_and_no_hint() {
    let temp = fixture();
    index(temp.path(), IndexConfig::default());
    let r = count(temp.path(), json!({ "pattern": "zz_nothing_here" }));
    assert_eq!(r["total"], 0, "{r}");
    assert!(r.get("excluded_reason").is_none(), "{r}");
    assert!(r.get("hint").is_none(), "{r}");
}

#[test]
fn with_hidden_indexed_the_dot_directory_query_hits() {
    let temp = fixture();
    index(
        temp.path(),
        IndexConfig {
            hidden: true,
            ..Default::default()
        },
    );
    let r = count(
        temp.path(),
        json!({ "pattern": "runs-on", "contains": true, "file": ".github/" }),
    );
    assert_eq!(r["total"], 1, "{r}");
    assert!(r.get("excluded_reason").is_none(), "{r}");
}
