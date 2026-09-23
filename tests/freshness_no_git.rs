//! Freshness outside a git repository.
//!
//! Until 1.8.0 a directory without `.git` was always reported `fresh`, because
//! `git status` was the only source of changed paths. The content fingerprint
//! (`size`, `mtime_ns`, blake3 per file) makes a tree walk cheap enough to answer
//! the same four questions — edited, added, deleted, reindexed — without git.

use reflex::mcp::run_mcp_server_io_in;
use reflex::{CacheManager, IndexConfig, Indexer};
use serde_json::{Value, json};
use std::fs;
use std::io::Cursor;
use std::path::Path;
use tempfile::TempDir;

fn indexed_dir() -> TempDir {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/a.rs"), "pub fn alpha_token() {}\n").unwrap();
    fs::write(root.join("src/b.rs"), "pub fn beta_token() {}\n").unwrap();
    fs::write(root.join("README.md"), "# readme\n").unwrap();
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

fn index(root: &Path) {
    let r = call_tool(root, "index_project", json!({}));
    assert!(r.get("error").is_none(), "{r}");
}

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
fn a_clean_directory_is_fresh_and_says_it_walked() {
    let temp = indexed_dir();
    let s = status(temp.path());
    assert_eq!(s["status"], "fresh", "{s}");
    assert_eq!(s["can_trust_results"], true, "{s}");
    assert_eq!(s["details"]["checked_by"], "walk", "{s}");
    assert!(s["details"].get("current_commit").is_none(), "{s}");
}

#[test]
fn an_edit_is_stale_and_named_then_indexing_makes_it_fresh() {
    let temp = indexed_dir();
    let root = temp.path();
    fs::write(
        root.join("src/a.rs"),
        "pub fn alpha_token() {}\npub fn gamma_token() {}\n",
    )
    .unwrap();

    let s = status(root);
    assert_eq!(s["status"], "stale", "1.7.2 said fresh here: {s}");
    assert_eq!(s["can_trust_results"], false, "{s}");
    assert_eq!(listed(&s, "files_modified"), vec!["src/a.rs"], "{s}");
    let r = search(root, "gamma_token");
    assert_eq!(r["total_count"], 0, "{r}");
    assert_eq!(r["can_trust_results"], false, "a 0 from a stale index: {r}");

    index(root);
    let s = status(root);
    assert_eq!(s["status"], "fresh", "{s}");
    let r = search(root, "gamma_token");
    assert_eq!(r["total_count"], 1, "{r}");
    assert_eq!(r["can_trust_results"], true, "{r}");
}

#[test]
fn a_new_file_is_added_then_fresh_after_indexing() {
    let temp = indexed_dir();
    let root = temp.path();
    fs::write(root.join("src/c.rs"), "pub fn delta_token() {}\n").unwrap();

    let s = status(root);
    assert_eq!(s["status"], "stale", "{s}");
    assert_eq!(listed(&s, "files_added"), vec!["src/c.rs"], "{s}");

    index(root);
    assert_eq!(status(root)["status"], "fresh");
    assert_eq!(search(root, "delta_token")["total_count"], 1);
}

#[test]
fn a_deleted_file_is_deleted_then_fresh_after_indexing() {
    let temp = indexed_dir();
    let root = temp.path();
    fs::remove_file(root.join("src/b.rs")).unwrap();

    let s = status(root);
    assert_eq!(s["status"], "stale", "{s}");
    assert_eq!(listed(&s, "files_deleted"), vec!["src/b.rs"], "{s}");
    assert_eq!(search(root, "beta_token")["can_trust_results"], false);

    index(root);
    assert_eq!(status(root)["status"], "fresh");
    assert_eq!(search(root, "beta_token")["total_count"], 0);
}

#[test]
fn rewriting_identical_bytes_is_fresh() {
    let temp = indexed_dir();
    let root = temp.path();
    let path = root.join("src/a.rs");
    let bytes = fs::read(&path).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(20));
    fs::write(&path, &bytes).unwrap();

    let s = status(root);
    assert_eq!(s["status"], "fresh", "same bytes, new mtime: {s}");
}

/// Edit, index, then put the old bytes back: the index holds the edit.
#[test]
fn reverting_an_indexed_edit_is_stale() {
    let temp = indexed_dir();
    let root = temp.path();
    let path = root.join("src/a.rs");
    let original = fs::read(&path).unwrap();
    fs::write(
        &path,
        "pub fn alpha_token() {}\npub fn epsilon_token() {}\n",
    )
    .unwrap();
    index(root);
    assert_eq!(status(root)["status"], "fresh");

    fs::write(&path, &original).unwrap();
    let s = status(root);
    assert_eq!(s["status"], "stale", "{s}");
    assert_eq!(listed(&s, "files_modified"), vec!["src/a.rs"], "{s}");
}

/// A binary file is never in the index, so its arrival cannot make the index
/// stale; a lock file IS indexed (and left out of searches by default), so it can.
#[test]
fn a_new_binary_is_not_staleness_but_a_new_lock_file_is() {
    let temp = indexed_dir();
    let root = temp.path();
    fs::write(root.join("image.png"), b"\x89PNG\r\n\x1a\n\0\0").unwrap();
    let s = status(root);
    assert_eq!(s["status"], "fresh", "{s}");

    fs::write(root.join("Cargo.lock"), "[[package]]\nname = \"x\"\n").unwrap();
    let s = status(root);
    assert_eq!(s["status"], "stale", "{s}");
    assert_eq!(listed(&s, "files_added"), vec!["Cargo.lock"], "{s}");
    index(root);
    assert_eq!(status(root)["status"], "fresh");
}
