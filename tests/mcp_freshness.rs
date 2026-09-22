//! The four staleness probes from the 1.7.0 field report, run verbatim over MCP.
//!
//! | change                                  | 1.7.0 said | search result          |
//! |-----------------------------------------|------------|------------------------|
//! | HEAD commit changed under the index     | stale      | auto-refreshed, correct|
//! | new untracked `src/storage/zz_probe.rs` | FRESH      | 0 hits                 |
//! | tracked `src/storage/mod.rs` edited     | FRESH      | 0 hits                 |
//! | indexed file deleted                    | FRESH      | ghost hit at old line  |
//!
//! Each wrong answer carried `can_trust_results: true`, so an agent concluded
//! "no usages" and acted on it.

use reflex::mcp::run_mcp_server_io_in;
use reflex::{CacheManager, IndexConfig, Indexer};
use serde_json::{Value, json};
use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

fn git(root: &Path, args: &[&str]) {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// An indexed git repo mirroring the field-test layout.
fn indexed_repo() -> TempDir {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    git(root, &["init", "-q"]);
    git(root, &["config", "user.email", "t@example.com"]);
    git(root, &["config", "user.name", "T"]);

    fs::create_dir_all(root.join("src/storage")).unwrap();
    fs::write(
        root.join("src/storage/mod.rs"),
        "pub fn storage_entry() -> u32 { 1 }\n",
    )
    .unwrap();
    fs::write(root.join("src/lib.rs"), "pub mod storage;\n").unwrap();

    git(root, &["add", "-A"]);
    git(root, &["commit", "-qm", "initial"]);

    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .unwrap();
    temp
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
    let line = text.lines().find(|l| !l.is_empty()).unwrap();
    let v: Value = serde_json::from_str(line).unwrap();
    assert!(v.get("error").is_none(), "{tool} errored: {}", v["error"]);
    let payload = v["result"]["content"][0]["text"].as_str().unwrap();
    serde_json::from_str(payload).unwrap_or_else(|_| json!({ "raw": payload }))
}

fn status(root: &Path) -> Value {
    call_tool(root, "check_index_status", json!({}))
}

fn search(root: &Path, pattern: &str) -> Value {
    call_tool(root, "search_code", json!({ "pattern": pattern }))
}

/// Every path named anywhere in a status response.
fn listed(status: &Value, key: &str) -> Vec<String> {
    status[key]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn probe_0_a_clean_tree_is_fresh_and_trusted() {
    let temp = indexed_repo();
    let s = status(temp.path());
    assert_eq!(s["status"], "fresh", "{s}");
    assert_eq!(s["can_trust_results"], true, "{s}");

    let r = search(temp.path(), "storage_entry");
    assert_eq!(r["status"], "fresh", "{r}");
    assert_eq!(r["can_trust_results"], true, "{r}");
}

#[test]
fn probe_1_head_moving_is_stale_and_names_both_hashes() {
    let temp = indexed_repo();
    let root = temp.path();

    fs::write(root.join("src/storage/mod.rs"), "pub fn moved() {}\n").unwrap();
    git(root, &["add", "-A"]);
    git(root, &["commit", "-qm", "second"]);

    let s = status(root);
    assert_eq!(s["status"], "stale", "{s}");
    assert_eq!(s["can_trust_results"], false, "{s}");
    assert!(
        s["reason"].as_str().unwrap().contains("Commit changed"),
        "{s}"
    );
}

#[test]
fn probe_2_a_new_untracked_file_is_stale_and_named() {
    let temp = indexed_repo();
    let root = temp.path();

    // The field-test probe, verbatim.
    fs::write(
        root.join("src/storage/zz_probe.rs"),
        "pub fn zz_probe_token() {}\n",
    )
    .unwrap();

    let s = status(root);
    assert_eq!(s["status"], "stale", "1.7.0 said fresh here: {s}");
    assert_eq!(s["can_trust_results"], false, "{s}");
    assert_eq!(
        listed(&s, "files_added"),
        vec!["src/storage/zz_probe.rs"],
        "the new file must be named: {s}"
    );

    // The search still returns 0 — the file genuinely is not indexed — but it can no
    // longer claim the answer is trustworthy.
    let r = search(root, "zz_probe_token");
    assert_eq!(r["status"], "stale", "{r}");
    assert_eq!(
        r["can_trust_results"], false,
        "a 0 from a stale index must never be trusted: {r}"
    );
}

#[test]
fn probe_3_an_edited_tracked_file_is_stale_and_named() {
    let temp = indexed_repo();
    let root = temp.path();

    fs::write(
        root.join("src/storage/mod.rs"),
        "pub fn storage_entry() -> u32 { 1 }\npub fn edited_token() {}\n",
    )
    .unwrap();

    let s = status(root);
    assert_eq!(s["status"], "stale", "1.7.0 said fresh here: {s}");
    assert_eq!(s["can_trust_results"], false, "{s}");
    assert_eq!(
        listed(&s, "files_modified"),
        vec!["src/storage/mod.rs"],
        "the edited file must be named: {s}"
    );

    let r = search(root, "edited_token");
    assert_eq!(r["can_trust_results"], false, "{r}");
}

#[test]
fn probe_4_a_deleted_file_is_stale_and_its_ghost_hit_is_not_trusted() {
    let temp = indexed_repo();
    let root = temp.path();

    fs::remove_file(root.join("src/storage/mod.rs")).unwrap();

    let s = status(root);
    assert_eq!(s["status"], "stale", "1.7.0 said fresh here: {s}");
    assert_eq!(s["can_trust_results"], false, "{s}");
    assert_eq!(
        listed(&s, "files_deleted"),
        vec!["src/storage/mod.rs"],
        "the deleted file must be named: {s}"
    );

    // The ghost hit still exists until a reindex — but it is now labelled untrusted
    // instead of being served as fact.
    let r = search(root, "storage_entry");
    assert_eq!(
        r["can_trust_results"], false,
        "a ghost hit must never be trusted: {r}"
    );

    // And a reindex clears it.
    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .unwrap();
    let r = search(root, "storage_entry");
    assert_eq!(r["total_count"], 0, "the ghost must be gone: {r}");
}

#[test]
fn the_advice_names_the_mcp_tool_never_the_cli() {
    let temp = indexed_repo();
    let root = temp.path();
    fs::write(root.join("src/lib.rs"), "pub mod storage;\n// edit\n").unwrap();

    let s = status(root);
    assert_eq!(s["action_required"], "index_project", "{s}");
    assert!(
        !s.to_string().contains("rfx index"),
        "an agent cannot run the CLI: {s}"
    );

    let r = search(root, "storage_entry");
    assert!(
        !r.to_string().contains("rfx index"),
        "search responses must not advise the CLI either: {r}"
    );
}

#[test]
fn the_change_counts_and_reason_describe_the_whole_tree() {
    let temp = indexed_repo();
    let root = temp.path();

    fs::write(root.join("src/lib.rs"), "pub mod storage;\n// edited\n").unwrap();
    fs::write(root.join("src/added.rs"), "fn a() {}\n").unwrap();
    fs::remove_file(root.join("src/storage/mod.rs")).unwrap();

    let s = status(root);
    assert_eq!(s["changed_count"], 3, "{s}");
    let reason = s["reason"].as_str().unwrap();
    for expected in ["1 modified", "1 added", "1 deleted"] {
        assert!(
            reason.contains(expected),
            "reason missing {expected:?}: {s}"
        );
    }
}

#[test]
fn a_reindex_returns_the_tree_to_fresh() {
    let temp = indexed_repo();
    let root = temp.path();

    fs::write(root.join("src/added.rs"), "pub fn added_token() {}\n").unwrap();
    assert_eq!(status(root)["status"], "stale");

    // Committing is what makes the tree clean; indexing makes the content current.
    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .unwrap();
    git(root, &["add", "-A"]);
    git(root, &["commit", "-qm", "third"]);
    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .unwrap();

    let s = status(root);
    assert_eq!(s["status"], "fresh", "{s}");
    assert_eq!(s["can_trust_results"], true, "{s}");

    let r = search(root, "added_token");
    assert_eq!(r["total_count"], 1, "{r}");
    assert_eq!(r["can_trust_results"], true, "{r}");
}
