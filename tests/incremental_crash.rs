//! Crash safety of index publishes. A child process (this test binary, re-run)
//! dies at each named write point of a delta update, a library update and a full
//! build. The cache must then open without error and answer as either the old or
//! the new tree, never report `fresh` for content it does not hold, and the next
//! `Indexer::index` must bring it to a fresh build's answers.

use reflex::models::IndexStatus;
use reflex::query::{QueryEngine, QueryFilter};
use reflex::{CacheManager, IndexConfig, Indexer};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

/// Test-harness variables of the child run (read by this test binary only).
const POINT: &str = "REFLEX_CRASH_TEST_POINT";
const ROOT: &str = "REFLEX_CRASH_TEST_ROOT";
const MODE: &str = "REFLEX_CRASH_TEST_MODE";

/// The paths `change` touches (for `update_paths`).
fn changed_paths() -> Vec<PathBuf> {
    ["src/b.rs", "src/c.rs", "src/new.rs"]
        .iter()
        .map(PathBuf::from)
        .collect()
}

fn write(root: &Path, rel: &str, body: &str) {
    let p = root.join(rel);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, body).unwrap();
}

fn workspace() -> TempDir {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    for (name, body) in [
        ("a", "pub fn alpha_token() { common_token(); }\n"),
        ("b", "pub fn beta_old_token() { common_token(); }\n"),
        ("c", "pub fn gamma_token() { common_token(); }\n"),
    ] {
        write(root, &format!("src/{name}.rs"), body);
    }
    write(root, "src/lib.rs", "mod a;\nmod b;\nmod c;\n");
    write(root, "Cargo.toml", "[package]\nname = \"demo\"\n");
    temp
}

/// Undo `change`: the tree is the original again.
fn revert(root: &Path) {
    write(
        root,
        "src/b.rs",
        "pub fn beta_old_token() { common_token(); }\n",
    );
    write(
        root,
        "src/c.rs",
        "pub fn gamma_token() { common_token(); }\n",
    );
    fs::remove_file(root.join("src/new.rs")).unwrap();
}

/// Modify b, delete c, add new.
fn change(root: &Path) {
    write(
        root,
        "src/b.rs",
        "pub fn beta_new_token() { common_token(); }\n",
    );
    fs::remove_file(root.join("src/c.rs")).unwrap();
    write(
        root,
        "src/new.rs",
        "pub fn added_token() { common_token(); }\n",
    );
}

/// `mode`: "full" merges every change into a new base (a full build); "fold"
/// folds every update's recent segment into a new delta; otherwise updates keep
/// the recent segment.
fn indexer_for(root: &Path, mode: &str) -> Indexer {
    let mut ix = Indexer::new(CacheManager::new(root), IndexConfig::default());
    if mode == "full" {
        ix.set_merge_limits(0, 0);
    } else {
        ix.set_merge_limits(usize::MAX, u64::MAX);
    }
    if mode == "fold" {
        ix.set_recent_limits(0, 0);
    }
    ix
}

fn indexer(root: &Path, merge: bool) -> Indexer {
    indexer_for(root, if merge { "full" } else { "index" })
}

/// The child: set the abort point and run the update; the process dies there.
#[test]
fn crash_child() {
    let (Ok(point), Ok(root), Ok(mode)) = (
        std::env::var(POINT),
        std::env::var(ROOT),
        std::env::var(MODE),
    ) else {
        return; // not a child run
    };
    let root = PathBuf::from(root);
    let mut ix = indexer_for(&root, &mode);
    ix.set_abort_point(&point);
    match mode.as_str() {
        "update" => {
            ix.update_paths(&root, &changed_paths()).unwrap();
        }
        _ => {
            ix.index(&root, false).unwrap();
        }
    }
    // The point was not reached.
    std::process::exit(3);
}

/// Answers: the paths each token is found in, and the status.
fn answers(root: &Path) -> (Vec<(String, Vec<String>)>, IndexStatus) {
    let engine = QueryEngine::new(CacheManager::new(root));
    let mut status = IndexStatus::Fresh;
    let mut out = Vec::new();
    for token in [
        "alpha_token",
        "beta_old_token",
        "beta_new_token",
        "gamma_token",
        "added_token",
        "common_token",
    ] {
        let filter = QueryFilter {
            suppress_output: true,
            ..Default::default()
        };
        let response = engine
            .search_with_metadata(token, filter)
            .unwrap_or_else(|e| panic!("{token}: {e:#}"));
        status = response.status.clone();
        let value = serde_json::to_value(&response.results).unwrap();
        let mut paths: Vec<String> = value
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g["path"].as_str().unwrap_or_default().to_string())
            .collect();
        paths.sort();
        out.push((token.to_string(), paths));
    }
    (out, status)
}

type Answers = Vec<(String, Vec<String>)>;

/// The answers of a fresh build of the tree before and after `change`.
fn expected() -> (Answers, Answers) {
    let temp = workspace();
    let root = temp.path();
    indexer(root, false).index(root, false).unwrap();
    let old = answers(root).0;
    change(root);
    fs::remove_dir_all(root.join(".reflex")).unwrap();
    indexer(root, false).index(root, false).unwrap();
    let new = answers(root).0;
    (old, new)
}

/// `reverted`: the tree goes back to the original between the crash and the
/// recovery run (so a `meta.db` one generation behind the stores would call the
/// reverted files unchanged while the stores hold the crashed run's bytes).
fn crash_at(mode: &str, point: &str, reverted: bool, old: &Answers, new: &Answers) {
    let temp = workspace();
    let root = temp.path();
    indexer(root, false).index(root, false).unwrap();
    if mode != "full" {
        // A live delta before the crash, so tombstones and the old delta are in play.
        write(
            root,
            "src/a.rs",
            "pub fn alpha_token() { common_token(); }\n// v2\n",
        );
        indexer(root, false).index(root, false).unwrap();
    }
    change(root);

    let status = Command::new(std::env::current_exe().unwrap())
        .args(["crash_child", "--exact", "--nocapture", "--test-threads=1"])
        .env(POINT, point)
        .env(ROOT, root)
        .env(MODE, mode)
        .status()
        .expect("run the child");
    assert_eq!(
        status.code(),
        Some(reflex::indexer::ABORT_EXIT_CODE),
        "{mode}/{point}: the child did not stop at the point ({status})"
    );

    let (got, status) = answers(root);
    assert!(
        got == *old || got == *new,
        "{mode}/{point}: answers are neither the old nor the new tree's: {got:?}"
    );
    if got == *old {
        assert_ne!(
            status,
            IndexStatus::Fresh,
            "{mode}/{point}: the old snapshot is reported fresh"
        );
    }

    // The next run recovers.
    if reverted {
        revert(root);
    }
    indexer(root, false).index(root, false).unwrap();
    let (got, status) = answers(root);
    let expected = if reverted { old } else { new };
    assert_eq!(
        got, *expected,
        "{mode}/{point} (reverted: {reverted}): after recovery"
    );
    assert_eq!(
        status,
        IndexStatus::Fresh,
        "{mode}/{point} (reverted: {reverted}): after recovery"
    );
}

#[test]
fn a_crash_at_any_write_point_leaves_the_old_or_the_new_snapshot() {
    let (old, new) = expected();
    for reverted in [false, true] {
        for mode in ["index", "update", "fold"] {
            for point in ["delta-files", "unlinked", "manifest", "meta"] {
                crash_at(mode, point, reverted, &old, &new);
            }
        }
        for point in ["base-files", "base-manifest", "base-meta"] {
            crash_at("full", point, reverted, &old, &new);
        }
    }
}
