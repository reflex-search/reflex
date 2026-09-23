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
fn probe_1_head_moving_with_changed_content_is_stale_and_names_the_file() {
    let temp = indexed_repo();
    let root = temp.path();

    fs::write(root.join("src/storage/mod.rs"), "pub fn moved() {}\n").unwrap();
    git(root, &["add", "-A"]);
    git(root, &["commit", "-qm", "second"]);

    // The tree is clean to git; the commit diff is what names the file.
    let s = status(root);
    assert_eq!(s["status"], "stale", "{s}");
    assert_eq!(s["can_trust_results"], false, "{s}");
    assert_eq!(
        listed(&s, "files_modified"),
        vec!["src/storage/mod.rs"],
        "{s}"
    );
    assert_eq!(s["details"]["checked_by"], "git", "{s}");
    assert_ne!(
        s["details"]["indexed_commit"], s["details"]["current_commit"],
        "{s}"
    );
}

fn index(root: &Path) -> Value {
    let r = call_tool(root, "index_project", json!({}));
    assert!(r.get("error").is_none(), "{r}");
    r
}

/// The 2.0.0 field-test sequence: an agent edits, reindexes and searches, and
/// never commits. Before, only `git commit` could return the tree to `fresh`.
#[test]
fn edit_then_index_project_is_fresh_without_commit() {
    let temp = indexed_repo();
    let root = temp.path();

    fs::write(
        root.join("src/storage/mod.rs"),
        "pub fn storage_entry() -> u32 { 1 }\npub fn dirty_token() {}\n",
    )
    .unwrap();
    assert_eq!(status(root)["status"], "stale");

    index(root);

    let s = status(root);
    assert_eq!(s["status"], "fresh", "reindexed content is current: {s}");
    assert_eq!(s["can_trust_results"], true, "{s}");
    assert_eq!(s["details"]["checked_by"], "git", "{s}");

    let r = search(root, "dirty_token");
    assert_eq!(r["total_count"], 1, "{r}");
    assert_eq!(r["status"], "fresh", "{r}");
    assert_eq!(r["can_trust_results"], true, "{r}");
}

#[test]
fn add_untracked_then_index_is_fresh() {
    let temp = indexed_repo();
    let root = temp.path();

    fs::write(root.join("src/added.rs"), "pub fn added_token() {}\n").unwrap();
    let s = status(root);
    assert_eq!(listed(&s, "files_added"), vec!["src/added.rs"], "{s}");

    index(root);
    let s = status(root);
    assert_eq!(s["status"], "fresh", "{s}");
    assert_eq!(search(root, "added_token")["total_count"], 1);
}

#[test]
fn delete_then_index_is_fresh_and_the_file_is_gone() {
    let temp = indexed_repo();
    let root = temp.path();

    fs::remove_file(root.join("src/storage/mod.rs")).unwrap();
    let s = status(root);
    assert_eq!(
        listed(&s, "files_deleted"),
        vec!["src/storage/mod.rs"],
        "{s}"
    );

    let r = index(root);
    assert_eq!(r["total_files"], 1, "{r}");
    let s = status(root);
    assert_eq!(s["status"], "fresh", "{s}");
    assert_eq!(search(root, "storage_entry")["total_count"], 0);
}

/// An edit indexed and then reverted is clean to git, but the index holds the
/// edited bytes. Only the dirty-at-index record can catch this.
#[test]
fn revert_after_indexing_the_edit_is_stale() {
    let temp = indexed_repo();
    let root = temp.path();

    fs::write(
        root.join("src/storage/mod.rs"),
        "pub fn storage_entry() -> u32 { 1 }\npub fn reverted_token() {}\n",
    )
    .unwrap();
    index(root);
    assert_eq!(status(root)["status"], "fresh");

    git(root, &["checkout", "--", "src/storage/mod.rs"]);

    let s = status(root);
    assert_eq!(s["status"], "stale", "content changed back: {s}");
    assert_eq!(
        listed(&s, "files_modified"),
        vec!["src/storage/mod.rs"],
        "{s}"
    );
    assert_eq!(search(root, "reverted_token")["can_trust_results"], false);

    index(root);
    assert_eq!(status(root)["status"], "fresh");
    assert_eq!(search(root, "reverted_token")["total_count"], 0);
}

/// Committing already-indexed content moves HEAD without changing a byte.
#[test]
fn commit_of_indexed_content_stays_fresh() {
    let temp = indexed_repo();
    let root = temp.path();

    fs::write(root.join("src/added.rs"), "pub fn committed_token() {}\n").unwrap();
    index(root);
    git(root, &["add", "-A"]);
    git(root, &["commit", "-qm", "already indexed"]);

    let s = status(root);
    assert_eq!(s["status"], "fresh", "{s}");
    assert_eq!(s["can_trust_results"], true, "{s}");
    assert_ne!(
        s["details"]["indexed_commit"], s["details"]["current_commit"],
        "the commits differ and that is fine: {s}"
    );
}

#[test]
fn touch_with_identical_content_is_fresh() {
    let temp = indexed_repo();
    let root = temp.path();
    let path = root.join("src/lib.rs");
    let bytes = fs::read(&path).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(20));
    fs::write(&path, &bytes).unwrap();

    let s = status(root);
    assert_eq!(s["status"], "fresh", "same bytes, new mtime: {s}");
}

#[test]
fn a_new_branch_on_the_same_tree_is_fresh() {
    let temp = indexed_repo();
    let root = temp.path();
    git(root, &["checkout", "-qb", "feature"]);

    let s = status(root);
    assert_eq!(s["status"], "fresh", "{s}");
    assert_eq!(s["details"]["current_branch"], "feature", "{s}");
    assert_ne!(s["details"]["indexed_branch"], "feature", "{s}");
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

/// The verdict is memoised for `REFLEX_FRESHNESS_TTL_MS` (default 1 s) so a burst
/// of searches spawns git once. An index write in the same process must drop that
/// memo: before this test, `index_project` left a stale "modified" verdict in the
/// cache for up to the TTL, and the first search after re-indexing was reported
/// untrustworthy even though the index was current.
#[test]
fn a_search_right_after_reindex_is_fresh_within_the_memo_ttl() {
    let temp = indexed_repo();
    let root = temp.path();

    // Prime the memo with a stale verdict: a search sees the edited file.
    fs::write(
        root.join("src/storage/mod.rs"),
        "pub fn edited_token() {}\n",
    )
    .unwrap();
    let r = search(root, "storage_entry");
    assert_eq!(r["status"], "stale", "{r}");

    // Commit so the tree is clean, then re-index through the MCP tool.
    git(root, &["add", "-A"]);
    git(root, &["commit", "-qm", "edit"]);
    let idx = call_tool(root, "index_project", json!({}));
    assert!(idx.get("error").is_none(), "{idx}");

    // Well inside the TTL: the search must not be answered from the old memo.
    let r = search(root, "edited_token");
    assert_eq!(r["status"], "fresh", "{r}");
    assert_eq!(r["can_trust_results"], true, "{r}");
    assert_eq!(r["total_count"], 1, "{r}");
}

/// The stale reason "deleted files still produce hits at their old lines" is right
/// before a reindex and must never appear after one.
#[test]
fn a_reindex_after_a_delete_carries_no_ghost_warning() {
    let temp = indexed_repo();
    let root = temp.path();
    fs::remove_file(root.join("src/storage/mod.rs")).unwrap();

    let s = status(root);
    assert!(
        s["reason"]
            .as_str()
            .unwrap()
            .contains("deleted files still produce hits"),
        "{s}"
    );

    index(root);
    let s = status(root);
    assert_eq!(s["status"], "fresh", "{s}");
    assert!(s.get("reason").is_none(), "{s}");

    let r = search(root, "storage_entry");
    assert_eq!(r["total_count"], 0, "{r}");
    assert!(r.get("warning").is_none(), "{r}");
    assert!(
        !r.to_string().contains("deleted files still produce hits"),
        "{r}"
    );
}
