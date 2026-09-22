//! `git status --porcelain` parsing for freshness detection.
//!
//! Before 1.7.2, freshness compared `git rev-parse HEAD` to the indexed commit and
//! then sampled the mtimes of the FIRST TEN indexed files. So an edit to any other
//! file, every untracked file (absent from the indexed list entirely), and every
//! deletion (`fs::metadata` fails, skipped silently) all reported `fresh`.

use reflex::git::{WorktreeChanges, get_worktree_changes};
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
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A repo with one committed file and a `.gitignore`.
fn repo() -> TempDir {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    git(root, &["init", "-q"]);
    git(root, &["config", "user.email", "t@example.com"]);
    git(root, &["config", "user.name", "T"]);
    fs::write(root.join(".gitignore"), "ignored/\n*.log\n").unwrap();
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/lib.rs"), "pub fn existing() {}\n").unwrap();
    git(root, &["add", "-A"]);
    git(root, &["commit", "-qm", "initial"]);
    temp
}

/// Keep only paths Reflex would index.
fn indexable(p: &str) -> bool {
    reflex::indexer::Indexer::is_indexable_path(Path::new(p))
}

fn changes(root: &Path) -> WorktreeChanges {
    get_worktree_changes(root, indexable).unwrap()
}

#[test]
fn a_clean_tree_reports_nothing() {
    let temp = repo();
    let c = changes(temp.path());
    assert!(c.is_empty(), "clean tree reported changes: {c:?}");
    assert_eq!(c.total(), 0);
}

#[test]
fn an_untracked_file_is_reported_as_added() {
    let temp = repo();
    fs::write(
        temp.path().join("src/zz_probe.rs"),
        "pub fn probe_token() {}\n",
    )
    .unwrap();

    let c = changes(temp.path());
    assert_eq!(c.added, vec!["src/zz_probe.rs"], "{c:?}");
    assert!(c.modified.is_empty());
    assert!(c.deleted.is_empty());
}

#[test]
fn an_untracked_file_in_a_new_nested_directory_is_listed_individually() {
    let temp = repo();
    fs::create_dir_all(temp.path().join("src/storage/deep")).unwrap();
    fs::write(temp.path().join("src/storage/deep/new.rs"), "fn x() {}\n").unwrap();

    // Without --untracked-files=all, git reports only "src/storage/" here.
    let c = changes(temp.path());
    assert_eq!(c.added, vec!["src/storage/deep/new.rs"], "{c:?}");
}

#[test]
fn an_edited_tracked_file_is_reported_as_modified() {
    let temp = repo();
    fs::write(
        temp.path().join("src/lib.rs"),
        "pub fn existing() {}\npub fn newly_added_fn() {}\n",
    )
    .unwrap();

    let c = changes(temp.path());
    assert_eq!(c.modified, vec!["src/lib.rs"], "{c:?}");
}

#[test]
fn a_deleted_file_is_reported_as_deleted() {
    let temp = repo();
    fs::remove_file(temp.path().join("src/lib.rs")).unwrap();

    let c = changes(temp.path());
    assert_eq!(c.deleted, vec!["src/lib.rs"], "{c:?}");
}

#[test]
fn a_staged_new_file_counts_as_added() {
    let temp = repo();
    fs::write(temp.path().join("src/staged.rs"), "fn s() {}\n").unwrap();
    git(temp.path(), &["add", "src/staged.rs"]);

    let c = changes(temp.path());
    assert_eq!(c.added, vec!["src/staged.rs"], "{c:?}");
}

#[test]
fn a_rename_decomposes_into_a_delete_and_an_add() {
    let temp = repo();
    git(temp.path(), &["mv", "src/lib.rs", "src/renamed.rs"]);

    // --no-renames is what makes this shape appear, and it is exactly what an index
    // must do: drop the old path, index the new one.
    let c = changes(temp.path());
    assert_eq!(c.deleted, vec!["src/lib.rs"], "{c:?}");
    assert_eq!(c.added, vec!["src/renamed.rs"], "{c:?}");
}

#[test]
fn paths_with_spaces_and_non_ascii_survive_intact() {
    let temp = repo();
    // Porcelain v1 C-quotes these; `-z` is what stops them arriving as
    // "\"src/a file.rs\"" or with \303\251 escapes.
    fs::write(temp.path().join("src/a file.rs"), "fn a() {}\n").unwrap();
    fs::write(temp.path().join("src/café_données.rs"), "fn b() {}\n").unwrap();

    let c = changes(temp.path());
    let mut added = c.added.clone();
    added.sort();
    assert_eq!(
        added,
        vec!["src/a file.rs", "src/café_données.rs"],
        "paths must arrive unquoted and unescaped: {c:?}"
    );
}

#[test]
fn gitignored_files_are_not_changes() {
    let temp = repo();
    fs::create_dir_all(temp.path().join("ignored")).unwrap();
    fs::write(temp.path().join("ignored/thing.rs"), "fn i() {}\n").unwrap();

    let c = changes(temp.path());
    assert!(
        c.is_empty(),
        "a .gitignored file is never indexed, so it cannot make the index stale: {c:?}"
    );
}

#[test]
fn files_reflex_does_not_index_are_not_changes() {
    let temp = repo();
    // Editing a README or a lockfile must not mark the index permanently stale.
    fs::write(temp.path().join("README.md"), "# docs\n").unwrap();
    fs::write(temp.path().join("notes.txt"), "hello\n").unwrap();
    fs::write(temp.path().join("data.json"), "{}\n").unwrap();

    let c = changes(temp.path());
    assert!(
        c.is_empty(),
        "non-indexed file types must not count as staleness: {c:?}"
    );
}

#[test]
fn the_lists_are_capped_but_the_counts_are_not() {
    let temp = repo();
    for i in 0..150 {
        fs::write(temp.path().join(format!("src/f{i}.rs")), "fn f() {}\n").unwrap();
    }

    let c = changes(temp.path());
    assert_eq!(c.added_count, 150, "the count must be the true total");
    assert_eq!(c.added.len(), 100, "the list must be capped");
    assert!(c.truncated, "truncation must be advertised");
}

#[test]
fn a_mixed_tree_buckets_every_change_correctly() {
    let temp = repo();
    fs::write(temp.path().join("src/other.rs"), "fn o() {}\n").unwrap();
    git(temp.path(), &["add", "-A"]);
    git(temp.path(), &["commit", "-qm", "second"]);

    fs::write(temp.path().join("src/lib.rs"), "pub fn changed() {}\n").unwrap();
    fs::remove_file(temp.path().join("src/other.rs")).unwrap();
    fs::write(temp.path().join("src/brand_new.rs"), "fn n() {}\n").unwrap();

    let c = changes(temp.path());
    assert_eq!(c.modified, vec!["src/lib.rs"], "{c:?}");
    assert_eq!(c.deleted, vec!["src/other.rs"], "{c:?}");
    assert_eq!(c.added, vec!["src/brand_new.rs"], "{c:?}");
    assert_eq!(c.total(), 3);
}
