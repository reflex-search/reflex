//! meta.db connections must all carry Reflex's pragmas.
//!
//! Regression guard for the 1.7.1 field report, where `BEGIN IMMEDIATE` returned a raw
//! `database is locked` the instant the background symbol indexer held a write, because
//! no connection ever set `busy_timeout`. See `reflex::cache::open_meta_db`.

use reflex::cache::{CacheManager, open_meta_db};
use tempfile::TempDir;

/// Every `meta.db` connection must go through `open_meta_db`.
///
/// `Connection::open` is allowed only inside the helper itself, and for the separate
/// pulse snapshot database (`src/pulse/snapshot.rs`), which is not `meta.db`.
#[test]
fn meta_db_connections_all_use_the_helper() {
    let allowed: &[(&str, usize)] = &[
        ("src/cache.rs", 1),          // inside open_meta_db
        ("src/pulse/snapshot.rs", 3), // the snapshot database, not meta.db
    ];

    let mut offenders = Vec::new();
    for entry in walk_rust_sources("src") {
        let text = std::fs::read_to_string(&entry).unwrap();
        let count = text.matches("Connection::open(").count();
        if count == 0 {
            continue;
        }
        let rel = entry.to_string_lossy().replace('\\', "/");
        let budget = allowed
            .iter()
            .find(|(p, _)| rel.ends_with(p))
            .map(|(_, n)| *n)
            .unwrap_or(0);
        if count > budget {
            offenders.push(format!("{rel}: {count} occurrence(s), budget {budget}"));
        }
    }

    assert!(
        offenders.is_empty(),
        "Use reflex::cache::open_meta_db instead of Connection::open so the \
         busy_timeout / WAL / foreign_keys pragmas are applied:\n  {}",
        offenders.join("\n  ")
    );
}

#[test]
fn open_meta_db_sets_the_pragmas() {
    let temp = TempDir::new().unwrap();
    let cache = CacheManager::new(temp.path());
    cache.init().unwrap();

    let conn = open_meta_db(cache.path().join("meta.db")).unwrap();

    let journal: String = conn
        .query_row("PRAGMA journal_mode", [], |r| r.get(0))
        .unwrap();
    assert_eq!(journal.to_lowercase(), "wal", "journal_mode should be WAL");

    let foreign_keys: i64 = conn
        .query_row("PRAGMA foreign_keys", [], |r| r.get(0))
        .unwrap();
    assert_eq!(foreign_keys, 1, "foreign_keys should be ON");

    // busy_timeout is not readable as a pragma on every SQLite build, so assert the
    // observable behaviour instead: a competing writer must be waited on, not rejected.
    let busy: i64 = conn
        .query_row("PRAGMA busy_timeout", [], |r| r.get(0))
        .unwrap_or(-1);
    assert!(
        busy >= 5000 || busy == -1,
        "busy_timeout should be at least 5000ms, got {busy}"
    );
}

/// A second writer must WAIT for the first, not fail instantly with "database is locked".
#[test]
fn busy_timeout_makes_a_competing_writer_wait() {
    let temp = TempDir::new().unwrap();
    let cache = CacheManager::new(temp.path());
    cache.init().unwrap();
    let db = cache.path().join("meta.db");

    let holder = open_meta_db(&db).unwrap();
    holder.execute_batch("BEGIN IMMEDIATE").unwrap();

    let other = open_meta_db(&db).unwrap();
    let start = std::time::Instant::now();
    let err = other.execute_batch("BEGIN IMMEDIATE").unwrap_err();
    let waited = start.elapsed();

    // It still fails (the holder never releases), but only after honouring the timeout.
    // Without busy_timeout this returns in microseconds.
    assert!(
        waited >= std::time::Duration::from_secs(4),
        "second writer gave up after {waited:?}; busy_timeout is not being applied ({err})"
    );
}

fn walk_rust_sources(dir: &str) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(dir)];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    out
}
