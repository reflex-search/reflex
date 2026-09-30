//! The freshness check sees changes to the files that decide WHICH files are
//! indexed, and does not report files no index run would hold.
//!
//! Before auto-update (2026-09-29) three holes let the check disagree with a fresh
//! build, and an automatic update would have acted on each one forever:
//!   1. an edit to `.gitignore` or `.reflex/config.toml` was never stale (neither is
//!      an indexable file), so newly excluded files stayed searchable;
//!   2. a tracked file that `.gitignore` ignores was reported `added` by every check
//!      (git lists it, the walker skips it), on a fresh build too;
//!   3. compaction deleted the rows of missing files but left them in the stores,
//!      so the check could not see the deletion while search still returned it.

use reflex::cache::CacheManager;
use reflex::indexer::Indexer;
use reflex::models::{IndexConfig, IndexStatus, IndexStatusReport};
use reflex::query::{QueryEngine, QueryFilter};
use std::fs;
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

fn repo() -> TempDir {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    git(root, &["init", "-q"]);
    git(root, &["config", "user.email", "t@example.com"]);
    git(root, &["config", "user.name", "T"]);
    fs::create_dir_all(root.join("src/gen")).unwrap();
    fs::write(root.join("src/lib.rs"), "pub fn rules_lib_token() {}\n").unwrap();
    fs::write(root.join("src/gen/out.rs"), "pub fn rules_gen_token() {}\n").unwrap();
    fs::write(root.join(".gitignore"), ".reflex/\n").unwrap();
    git(root, &["add", "-A"]);
    git(root, &["commit", "-qm", "initial"]);
    temp
}

fn index(root: &Path) {
    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .unwrap();
}

fn report(root: &Path) -> IndexStatusReport {
    QueryEngine::new(CacheManager::new(root))
        .fresh_index_report()
        .unwrap()
}

fn modified(r: &IndexStatusReport) -> Vec<String> {
    r.warning
        .as_ref()
        .and_then(|w| w.files_modified.clone())
        .unwrap_or_default()
}

fn added(r: &IndexStatusReport) -> Vec<String> {
    r.warning
        .as_ref()
        .and_then(|w| w.files_added.clone())
        .unwrap_or_default()
}

fn hits(root: &Path, pattern: &str) -> usize {
    QueryEngine::new(CacheManager::new(root))
        .search_with_metadata(
            pattern,
            QueryFilter {
                suppress_output: true,
                ..Default::default()
            },
        )
        .unwrap()
        .results
        .len()
}

#[test]
fn a_gitignore_edit_is_stale_until_the_next_index_run() {
    let temp = repo();
    let root = temp.path();
    index(root);
    assert_eq!(report(root).status, IndexStatus::Fresh);

    fs::write(root.join(".gitignore"), ".reflex/\nsrc/gen/\n").unwrap();
    let r = report(root);
    assert_eq!(r.status, IndexStatus::Stale, "{r:?}");
    assert!(!r.can_trust_results);
    assert_eq!(modified(&r), vec![".gitignore".to_string()]);

    index(root);
    assert_eq!(report(root).status, IndexStatus::Fresh);
    assert_eq!(
        hits(root, "rules_gen_token"),
        0,
        "the ignored file left the index"
    );
}

#[test]
fn a_nested_gitignore_edit_is_stale() {
    let temp = repo();
    let root = temp.path();
    index(root);

    fs::write(root.join("src/.gitignore"), "gen/\n").unwrap();
    let r = report(root);
    assert_eq!(r.status, IndexStatus::Stale, "{r:?}");
    assert_eq!(modified(&r), vec!["src/.gitignore".to_string()]);

    index(root);
    assert_eq!(
        report(root).status,
        IndexStatus::Fresh,
        "recorded while dirty"
    );
    assert_eq!(hits(root, "rules_gen_token"), 0);
}

#[test]
fn a_config_edit_is_stale_until_the_next_index_run() {
    let temp = repo();
    let root = temp.path();
    index(root);

    let config = root.join(".reflex/config.toml");
    let mut body = fs::read_to_string(&config).unwrap();
    body.push_str("\n# a comment is still an edit\n");
    fs::write(&config, body).unwrap();
    let r = report(root);
    assert_eq!(r.status, IndexStatus::Stale, "{r:?}");
    assert_eq!(modified(&r), vec![".reflex/config.toml".to_string()]);

    index(root);
    assert_eq!(report(root).status, IndexStatus::Fresh);
}

#[test]
fn a_rule_edit_without_git_is_stale() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    fs::write(root.join("a.rs"), "pub fn no_git_rules_token() {}\n").unwrap();
    index(root);
    assert_eq!(report(root).status, IndexStatus::Fresh);

    fs::write(root.join(".ignore"), "a.rs\n").unwrap();
    let r = report(root);
    assert_eq!(r.status, IndexStatus::Stale, "{r:?}");
    assert_eq!(modified(&r), vec![".ignore".to_string()]);
    index(root);
    assert_eq!(report(root).status, IndexStatus::Fresh);
}

#[test]
fn a_tracked_file_the_gitignore_ignores_is_not_added() {
    let temp = repo();
    let root = temp.path();
    // Tracked first, ignored afterwards: git keeps listing it, the walker skips it.
    fs::write(root.join(".gitignore"), ".reflex/\nsrc/gen/\n").unwrap();
    git(root, &["add", "-A"]);
    git(root, &["commit", "-qm", "ignore gen"]);
    index(root);
    assert_eq!(hits(root, "rules_gen_token"), 0);

    // An edit to the ignored file: git lists it, no index run would hold it.
    fs::write(
        root.join("src/gen/out.rs"),
        "pub fn rules_gen_token_v2() {}\n",
    )
    .unwrap();
    let r = report(root);
    assert_eq!(r.status, IndexStatus::Fresh, "{r:?}");
    assert!(added(&r).is_empty());
}

#[test]
fn compaction_leaves_a_deletion_visible() {
    let temp = repo();
    let root = temp.path();
    index(root);

    fs::remove_file(root.join("src/gen/out.rs")).unwrap();
    CacheManager::new(root).compact().unwrap();
    let r = report(root);
    assert_eq!(r.status, IndexStatus::Stale, "{r:?}");
    assert_eq!(
        r.warning.and_then(|w| w.files_deleted).unwrap_or_default(),
        vec!["src/gen/out.rs".to_string()]
    );
}
