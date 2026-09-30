//! Automatic update (`reflex::auto_update`): a stale index is brought up to date
//! before a command reads it, and every answer then equals a fresh build of the
//! same tree.
//!
//! The fidelity sequence below is the gate for removing the MCP text that sends
//! agents to `check_index_status` / `index_project` (user decision 2026-09-29:
//! 95–100 % of answers equal a fresh build; the target is 100 %).

#![cfg(unix)]

use reflex::auto_update::{UpdateOptions, Updated, update_if_stale};
use reflex::cache::CacheManager;
use reflex::indexer::Indexer;
use reflex::models::IndexStatus;
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

fn write(root: &Path, rel: &str, body: &str) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, body).unwrap();
}

/// A git repository with a few modules, a doc and an ignore file.
fn repo() -> TempDir {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    git(root, &["init", "-q", "-b", "main"]);
    git(root, &["config", "user.email", "t@example.com"]);
    git(root, &["config", "user.name", "T"]);
    write(root, ".gitignore", ".reflex/\ntarget/\n");
    for i in 0..12 {
        write(
            root,
            &format!("src/m{i}.rs"),
            &format!(
                "use crate::m{};\npub fn token_m{i}() -> u32 {{ shared_token() + {i} }}\n",
                (i + 1) % 12
            ),
        );
    }
    write(root, "src/lib.rs", "pub fn shared_token() -> u32 { 0 }\n");
    write(
        root,
        "docs/guide.md",
        "The shared_token is documented here.\n",
    );
    git(root, &["add", "-A"]);
    git(root, &["commit", "-qm", "initial"]);
    temp
}

fn opts() -> UpdateOptions {
    UpdateOptions::library()
}

/// An update as the next command would run it. The freshness verdict is
/// memoised for 1 s per process (`REFLEX_FRESHNESS_TTL_MS`); these tests change
/// files faster than an agent's turn, so each command starts past the window.
fn update(root: &Path) -> Updated {
    reflex::query::invalidate_caches(root);
    update_if_stale(&CacheManager::new(root), &opts()).unwrap()
}

fn status(root: &Path) -> IndexStatus {
    QueryEngine::new(CacheManager::new(root))
        .fresh_index_report()
        .unwrap()
        .status
}

const PATTERNS: &[&str] = &[
    "shared_token",
    "token_m0",
    "token_m3",
    "token_m11",
    "token_new",
    "renamed_token",
    "branch_token",
    "u32",
];

/// Every (pattern, path, line, preview) the index answers for [`PATTERNS`].
fn answers(root: &Path) -> Vec<(String, String, usize, String)> {
    let engine = QueryEngine::new(CacheManager::new(root));
    let mut out = Vec::new();
    for pattern in PATTERNS {
        let response = engine
            .search_with_metadata(
                pattern,
                QueryFilter {
                    suppress_output: true,
                    use_contains: true,
                    ..Default::default()
                },
            )
            .unwrap();
        for file in response.results {
            for m in file.matches {
                out.push((
                    pattern.to_string(),
                    file.path.clone(),
                    m.span.start_line,
                    m.preview.clone(),
                ));
            }
        }
    }
    out.sort();
    out
}

/// The answers of a fresh build of a copy of `root` (with its `.git` and its
/// `.reflex/config.toml`, nothing else of `.reflex/`).
fn fresh_answers(root: &Path) -> Vec<(String, String, usize, String)> {
    let copy = TempDir::new().unwrap();
    let status = Command::new("cp")
        .arg("-a")
        .arg(format!("{}/.", root.display()))
        .arg(copy.path())
        .status()
        .unwrap();
    assert!(status.success());
    let cache = copy.path().join(".reflex");
    let config = fs::read(cache.join("config.toml")).ok();
    fs::remove_dir_all(&cache).unwrap();
    if let Some(config) = config {
        fs::create_dir_all(&cache).unwrap();
        fs::write(cache.join("config.toml"), config).unwrap();
    }
    let mgr = CacheManager::new(copy.path());
    let config = mgr.effective_index_config(&[]).unwrap();
    Indexer::new(mgr, config).index(copy.path(), false).unwrap();
    answers(copy.path())
}

#[test]
fn no_index_is_built() {
    let temp = repo();
    assert_eq!(update(temp.path()), Updated::Built);
    assert_eq!(status(temp.path()), IndexStatus::Fresh);
    assert_eq!(update(temp.path()), Updated::Nothing);
}

#[test]
fn an_edit_updates_the_named_path() {
    let temp = repo();
    let root = temp.path();
    update(root);
    write(root, "src/m3.rs", "pub fn token_new() {}\n");
    assert_eq!(update(root), Updated::Paths(1));
    assert_eq!(status(root), IndexStatus::Fresh);
    assert_eq!(answers(root), fresh_answers(root));
}

#[test]
fn many_changes_run_a_full_index() {
    let temp = repo();
    let root = temp.path();
    update(root);
    for i in 0..120 {
        write(root, &format!("gen/f{i}.rs"), "pub fn token_new() {}\n");
    }
    assert!(matches!(update(root), Updated::Index { changed: 120 }));
    assert_eq!(status(root), IndexStatus::Fresh);
}

#[test]
fn a_gitignore_edit_runs_a_full_index() {
    let temp = repo();
    let root = temp.path();
    update(root);
    write(root, ".gitignore", ".reflex/\ntarget/\ndocs/\n");
    assert!(matches!(update(root), Updated::Index { .. }));
    assert_eq!(status(root), IndexStatus::Fresh);
    assert_eq!(answers(root), fresh_answers(root));
}

#[test]
fn another_versions_cache_is_left_alone_by_a_server() {
    let temp = repo();
    let root = temp.path();
    update(root);
    let conn = reflex::cache::open_meta_db(root.join(".reflex/meta.db")).unwrap();
    for (k, v) in [
        ("schema_hash", "deadbeefdeadbeef"),
        ("writer_version", "0.0.1"),
        ("writer_git_sha", "abc1234def5678"),
    ] {
        conn.execute(
            "INSERT OR REPLACE INTO statistics (key, value, updated_at) VALUES (?, ?, 0)",
            [k, v],
        )
        .unwrap();
    }
    drop(conn);
    reflex::query::invalidate_caches(root);
    assert!(matches!(update(root), Updated::Skipped(_)));

    let cli = UpdateOptions {
        spawn_symbol_pass: false,
        self_heal_version: true,
    };
    assert_eq!(
        update_if_stale(&CacheManager::new(root), &cli).unwrap(),
        Updated::Built
    );
    assert_eq!(status(root), IndexStatus::Fresh);
}

#[test]
fn a_read_only_cache_is_skipped_not_an_error() {
    use std::os::unix::fs::PermissionsExt;
    let temp = repo();
    let root = temp.path();
    update(root);
    write(root, "src/m1.rs", "pub fn token_new() {}\n");
    let cache = root.join(".reflex");
    fs::set_permissions(&cache, fs::Permissions::from_mode(0o555)).unwrap();
    // Root ignores permissions; nothing to test then.
    let writable = fs::write(cache.join("probe"), b"").is_ok();
    let result = update(root);
    fs::set_permissions(&cache, fs::Permissions::from_mode(0o755)).unwrap();
    if !writable {
        assert!(matches!(result, Updated::Skipped(_)), "{result:?}");
        assert_eq!(status(root), IndexStatus::Stale);
    }
}

#[test]
fn concurrent_callers_update_once() {
    let temp = repo();
    let root = temp.path().to_path_buf();
    update(&root);
    write(&root, "src/m2.rs", "pub fn token_new() {}\n");
    let results: Vec<Updated> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..4).map(|_| s.spawn(|| update(&root))).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let wrote = results.iter().filter(|u| u.wrote()).count();
    assert_eq!(wrote, 1, "{results:?}");
    assert_eq!(status(&root), IndexStatus::Fresh);
}

/// The agent-style sequence: after each change, update, then compare every
/// answer with a fresh build of the same tree.
#[test]
fn every_answer_equals_a_fresh_build() {
    let temp = repo();
    let root = temp.path();
    update(root);

    type Step = (&'static str, fn(&Path));
    let steps: Vec<Step> = vec![
        ("edit", |r| {
            write(r, "src/m0.rs", "pub fn token_m0_v2() {}\n")
        }),
        ("add", |r| write(r, "src/new.rs", "pub fn token_new() {}\n")),
        ("delete", |r| fs::remove_file(r.join("src/m5.rs")).unwrap()),
        ("rename", |r| {
            fs::rename(r.join("src/m6.rs"), r.join("src/renamed.rs")).unwrap();
            write(r, "src/renamed.rs", "pub fn renamed_token() {}\n");
        }),
        ("add a doc", |r| {
            write(r, "docs/more.md", "shared_token again\n")
        }),
        ("edit twice", |r| {
            write(r, "src/m7.rs", "pub fn token_new() {}\n");
            write(r, "src/m7.rs", "pub fn token_new() { shared_token(); }\n");
        }),
        ("revert an indexed edit", |r| {
            git(r, &["checkout", "--", "src/m7.rs"])
        }),
        ("commit", |r| {
            git(r, &["add", "-A"]);
            git(r, &["commit", "-qm", "work"]);
        }),
        ("switch to a new branch and change it", |r| {
            git(r, &["checkout", "-q", "-b", "side"]);
            write(r, "src/side.rs", "pub fn branch_token() {}\n");
            fs::remove_file(r.join("src/m8.rs")).unwrap();
            git(r, &["add", "-A"]);
            git(r, &["commit", "-qm", "side"]);
        }),
        ("switch back", |r| git(r, &["checkout", "-q", "main"])),
        ("ignore a directory", |r| {
            write(r, ".gitignore", ".reflex/\ntarget/\ndocs/\n")
        }),
        ("exclude by config", |r| {
            let path = r.join(".reflex/config.toml");
            let body = fs::read_to_string(&path).unwrap();
            fs::write(
                path,
                body.replace(
                    "[index.exclude]\npatterns = []",
                    "[index.exclude]\npatterns = [\"src/m9.rs\"]",
                ),
            )
            .unwrap();
        }),
        ("unignore", |r| {
            write(r, ".gitignore", ".reflex/\ntarget/\n")
        }),
        ("delete a directory", |r| {
            fs::remove_dir_all(r.join("docs")).unwrap()
        }),
        ("many files", |r| {
            for i in 0..150 {
                write(r, &format!("bulk/f{i}.rs"), "pub fn token_new() {}\n");
            }
        }),
    ];

    let mut equal = 0;
    let mut report = Vec::new();
    for (name, step) in &steps {
        step(root);
        let updated = update(root);
        let fresh = status(root) == IndexStatus::Fresh;
        let same = answers(root) == fresh_answers(root);
        if fresh && same {
            equal += 1;
        }
        report.push(format!("{name:<40} {updated:?} fresh={fresh} same={same}"));
    }
    println!("{}", report.join("\n"));
    println!("{equal}/{} answers equal a fresh build", steps.len());
    assert_eq!(equal, steps.len(), "\n{}", report.join("\n"));
}
