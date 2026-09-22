//! Deleting a file must remove it from the index.
//!
//! In the 1.7.0 field test, deleting an indexed file left `check_index_status` saying
//! "fresh", a search returning a ghost hit at the old line, and the next incremental
//! `index_project` still reporting `total_files: 1027`. Only a later full pass cleared
//! it.
//!
//! Three separate holes, all covered here:
//!   1. `meta.db` was never pruned by `rfx index` — only by `compact()`, which is
//!      throttled to once a day and was skipped entirely for the MCP command.
//!   2. The incremental fast path noticed ADDED paths but never DELETED ones, so a
//!      delete-one-add-one left the file count unchanged and skipped the rebuild.
//!   3. `stats().total_files` counts `file_branches`, so it inherited (1).

use std::fs;

use reflex::cache::CacheManager;
use reflex::indexer::Indexer;
use reflex::models::IndexConfig;
use reflex::query::{QueryEngine, QueryFilter};
use tempfile::TempDir;

fn workspace(n: usize) -> TempDir {
    let temp = TempDir::new().unwrap();
    for i in 0..n {
        fs::write(
            temp.path().join(format!("m{i}.rs")),
            format!("pub fn unique_token_{i}() -> u32 {{ {i} }}\n"),
        )
        .unwrap();
    }
    temp
}

fn index(root: &std::path::Path) -> reflex::models::IndexStats {
    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .unwrap()
}

fn count(root: &std::path::Path, pattern: &str) -> usize {
    let engine = QueryEngine::new(CacheManager::new(root));
    engine
        .search_with_metadata(
            pattern,
            QueryFilter {
                limit: None,
                suppress_output: true,
                ..Default::default()
            },
        )
        .unwrap()
        .pagination
        .total
}

#[test]
fn a_deleted_file_leaves_no_ghost_hit_and_total_files_drops() {
    let temp = workspace(5);
    let root = temp.path();

    index(root);
    assert_eq!(count(root, "unique_token_3"), 1);

    fs::remove_file(root.join("m3.rs")).unwrap();
    let stats = index(root);

    assert_eq!(
        stats.total_files, 4,
        "total_files must reflect the deletion, not the pre-deletion count"
    );
    assert_eq!(
        count(root, "unique_token_3"),
        0,
        "the deleted file must not produce a ghost hit"
    );
    // The survivors are untouched.
    assert_eq!(count(root, "unique_token_4"), 1);
}

#[test]
fn delete_one_add_one_is_noticed_even_though_the_count_is_unchanged() {
    let temp = workspace(5);
    let root = temp.path();

    index(root);
    assert_eq!(count(root, "unique_token_1"), 1);

    // The hole: the file COUNT is identical, and every surviving hash matches, so
    // the incremental fast path used to skip the rebuild entirely.
    fs::remove_file(root.join("m1.rs")).unwrap();
    fs::write(
        root.join("brand_new.rs"),
        "pub fn freshly_added_token() {}\n",
    )
    .unwrap();

    let stats = index(root);
    assert_eq!(stats.total_files, 5, "one out, one in");

    assert_eq!(
        count(root, "unique_token_1"),
        0,
        "the deleted file must be gone"
    );
    assert_eq!(
        count(root, "freshly_added_token"),
        1,
        "the added file must be indexed"
    );
}

#[test]
fn the_branch_file_list_omits_a_deleted_path() {
    let temp = workspace(3);
    let root = temp.path();
    index(root);

    let cache = CacheManager::new(root);
    let branch = reflex::git::get_current_branch(root).unwrap_or_else(|_| "_default".to_string());

    let before = cache.get_branch_files(&branch).unwrap();
    assert!(before.keys().any(|p| p.ends_with("m2.rs")));

    fs::remove_file(root.join("m2.rs")).unwrap();
    index(root);

    let after = cache.get_branch_files(&branch).unwrap();
    assert!(
        !after.keys().any(|p| p.ends_with("m2.rs")),
        "get_branch_files still lists the deleted path: {:?}",
        after.keys().collect::<Vec<_>>()
    );
    assert_eq!(after.len(), 2);
}

#[test]
fn deleting_every_file_empties_the_index() {
    let temp = workspace(3);
    let root = temp.path();
    index(root);

    for i in 0..3 {
        fs::remove_file(root.join(format!("m{i}.rs"))).unwrap();
    }
    // Leave one file so the walker has something to find.
    fs::write(root.join("last.rs"), "pub fn only_survivor() {}\n").unwrap();

    let stats = index(root);
    assert_eq!(stats.total_files, 1);
    assert_eq!(count(root, "unique_token_0"), 0);
    assert_eq!(count(root, "only_survivor"), 1);
}
