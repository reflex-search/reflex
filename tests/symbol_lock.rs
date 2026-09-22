//! An indexer must never surface SQLite's lock error for a running symbol pass.
//!
//! In the 1.7.0 field test, the first `index_project` left a detached
//! `rfx index-symbols-internal` holding `meta.db`. For the next ~4 minutes every
//! `index_project` and `rfx index` failed with:
//!
//!     Failed to begin meta.db schema transaction
//!     Failed to begin meta.db schema transaction: database is locked: Error code 5
//!
//! The cause is two separate locks: the symbol pass takes `.reflex/indexing.lock`,
//! never the workspace `.reflex/index.lock` that `Indexer::index` acquires. So the
//! indexer passed the lock gate and then hit `BEGIN IMMEDIATE`.

use std::fs;
use std::path::Path;

use reflex::background_indexer::BackgroundIndexer;
use reflex::cache::CacheManager;
use reflex::errors::ReflexError;
use reflex::indexer::Indexer;
use reflex::models::IndexConfig;
use tempfile::TempDir;

fn workspace() -> TempDir {
    let temp = TempDir::new().unwrap();
    fs::write(temp.path().join("a.rs"), "fn alpha() {}\n").unwrap();
    fs::write(temp.path().join("b.rs"), "fn beta() {}\n").unwrap();
    temp
}

fn kind(e: &anyhow::Error) -> &'static str {
    e.downcast_ref::<ReflexError>()
        .map(|re| re.kind())
        .unwrap_or("not a ReflexError")
}

/// Write an `indexing.lock` naming a pid, in the 1.7.2 JSON form.
fn write_lock(cache_dir: &Path, pid: u32) {
    fs::create_dir_all(cache_dir).unwrap();
    fs::write(
        cache_dir.join("indexing.lock"),
        format!(
            r#"{{"pid":{pid},"started_at":"{}"}}"#,
            chrono::Utc::now().to_rfc3339()
        ),
    )
    .unwrap();
}

#[test]
fn a_live_symbol_pass_yields_a_typed_error_not_a_sqlite_one() {
    let temp = workspace();
    let cache_dir = temp.path().join(".reflex");

    // Record THIS process as the holder. A process holding its own lock is alive by
    // definition, so the indexer must treat it as a running pass — but this process
    // never yields, so it exercises the timeout path exactly as a wedged pass would.
    write_lock(&cache_dir, std::process::id());

    let err = Indexer::new(CacheManager::new(temp.path()), IndexConfig::default())
        .index(temp.path(), false)
        .expect_err("a live symbol pass must block the indexer");

    assert_eq!(kind(&err), "SymbolIndexingInProgress", "got: {err:#}");

    let msg = format!("{err:#}");
    // The whole point: SQLite's wording must never reach a caller.
    for forbidden in ["database is locked", "Error code 5", "schema transaction"] {
        assert!(
            !msg.contains(forbidden),
            "must not surface {forbidden:?}: {msg}"
        );
    }
    assert!(
        msg.contains("symbol indexing in progress"),
        "must name what is happening: {msg}"
    );
    assert!(
        msg.contains(&format!("pid {}", std::process::id())),
        "must name the process holding the database: {msg}"
    );
}

#[test]
fn a_dead_pid_lock_is_reaped_and_indexing_proceeds() {
    let temp = workspace();
    let cache_dir = temp.path().join(".reflex");

    // PID 1 exists but is init, not a symbol pass. A pid that never existed would
    // work too; this also exercises the cmdline check that defeats pid reuse.
    write_lock(&cache_dir, 1);
    assert!(
        !BackgroundIndexer::is_running(&cache_dir),
        "a lock whose pid is not a symbol pass must not count as running"
    );
    assert!(
        !cache_dir.join("indexing.lock").exists(),
        "the stale lock must be removed, not merely ignored"
    );

    let stats = Indexer::new(CacheManager::new(temp.path()), IndexConfig::default())
        .index(temp.path(), false)
        .expect("indexing must proceed once the stale lock is reaped");
    assert!(stats.total_files >= 2);
}

#[test]
fn no_lock_means_no_wait() {
    let temp = workspace();
    let cache_dir = temp.path().join(".reflex");
    fs::create_dir_all(&cache_dir).unwrap();

    assert!(!BackgroundIndexer::is_running(&cache_dir));
    assert!(BackgroundIndexer::lock_holder(&cache_dir).is_none());

    let start = std::time::Instant::now();
    Indexer::new(CacheManager::new(temp.path()), IndexConfig::default())
        .index(temp.path(), false)
        .unwrap();
    assert!(
        start.elapsed() < std::time::Duration::from_secs(5),
        "an unlocked workspace must not pay the yield timeout"
    );
}

#[test]
fn the_legacy_bare_pid_lock_format_is_still_read() {
    let temp = TempDir::new().unwrap();
    let cache_dir = temp.path().join(".reflex");
    fs::create_dir_all(&cache_dir).unwrap();

    // 1.7.1 and earlier wrote just the pid. An upgrade must not orphan that lock.
    fs::write(cache_dir.join("indexing.lock"), "1\n").unwrap();

    // pid 1 is not a symbol pass, so it is reaped — which proves it was PARSED.
    // An unparsed lock would have left pid 0 and been reaped for a different reason,
    // so also assert the file is gone rather than the parse result directly.
    assert!(!BackgroundIndexer::is_running(&cache_dir));
    assert!(!cache_dir.join("indexing.lock").exists());
}

#[test]
fn the_cancel_sentinel_round_trips() {
    let temp = TempDir::new().unwrap();
    let cache_dir = temp.path().join(".reflex");
    fs::create_dir_all(&cache_dir).unwrap();

    assert!(!BackgroundIndexer::cancel_requested(&cache_dir));
    BackgroundIndexer::request_cancel(&cache_dir).unwrap();
    assert!(BackgroundIndexer::cancel_requested(&cache_dir));
    BackgroundIndexer::clear_cancel(&cache_dir);
    assert!(!BackgroundIndexer::cancel_requested(&cache_dir));
}

#[test]
fn indexing_clears_a_leftover_cancel_sentinel() {
    let temp = workspace();
    let cache_dir = temp.path().join(".reflex");
    fs::create_dir_all(&cache_dir).unwrap();

    // An indexer that died between requesting a cancel and clearing it would
    // otherwise stop the next symbol pass before it began.
    BackgroundIndexer::request_cancel(&cache_dir).unwrap();

    Indexer::new(CacheManager::new(temp.path()), IndexConfig::default())
        .index(temp.path(), false)
        .unwrap();

    assert!(
        !BackgroundIndexer::cancel_requested(&cache_dir),
        "a completed index must not leave a cancel request behind"
    );
}

#[test]
fn the_in_progress_message_matches_the_documented_shape() {
    let e: anyhow::Error = ReflexError::SymbolIndexingInProgress {
        pid: 12345,
        started_at: "14:22:07".to_string(),
        processed: 1000,
        total: 1027,
    }
    .into();

    assert_eq!(
        format!("{e}"),
        "symbol indexing in progress (pid 12345, started 14:22:07, 1000/1027 files)"
    );
}
