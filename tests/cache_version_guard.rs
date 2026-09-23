//! A cache written by another Reflex build: writers refuse, readers degrade.
//!
//! The field report had three `rfx mcp` servers (two v1.6.0, one v1.7.0) from three
//! Claude Code sessions sharing one `.reflex/`, and v1.6.0 had already produced
//! `content.bin is too small`.
//!
//! The cause was the inverse of what it should have been. `validate()` runs on EVERY
//! search and bailed on a schema-hash mismatch; that became `CacheCorrupted`; and the
//! MCP layer answers corruption by FORCE-REBUILDING. So each version saw a mismatch,
//! each rebuilt, and they streamed into content.bin concurrently.
//!
//! Now: readers are let through and told the results are not trustworthy; writers
//! refuse and name the owner.

use reflex::cache::{CacheManager, open_meta_db};
use reflex::errors::ReflexError;
use reflex::indexer::Indexer;
use reflex::models::IndexConfig;
use reflex::query::{QueryEngine, QueryFilter};
use std::fs;
use tempfile::TempDir;

fn indexed() -> TempDir {
    let temp = TempDir::new().unwrap();
    fs::write(
        temp.path().join("a.rs"),
        "pub fn version_guard_token() {}\n",
    )
    .unwrap();
    Indexer::new(CacheManager::new(temp.path()), IndexConfig::default())
        .index(temp.path(), false)
        .unwrap();
    temp
}

/// Rewrite the stamps so the cache looks like another build's.
fn stamp_foreign(cache: &CacheManager, version: &str) {
    let conn = open_meta_db(cache.path().join("meta.db")).unwrap();
    let now = chrono::Utc::now().timestamp().to_string();
    for (k, v) in [
        ("schema_hash", "deadbeefdeadbeef"),
        ("writer_version", version),
        ("writer_git_sha", "abc1234def5678"),
    ] {
        conn.execute(
            "INSERT OR REPLACE INTO statistics (key, value, updated_at) VALUES (?, ?, ?)",
            [k, v, &now],
        )
        .unwrap();
    }
}

fn kind(e: &anyhow::Error) -> &'static str {
    e.downcast_ref::<ReflexError>()
        .map(|re| re.kind())
        .unwrap_or("not a ReflexError")
}

#[test]
fn a_writer_refuses_and_names_both_versions() {
    let temp = indexed();
    let cache = CacheManager::new(temp.path());
    stamp_foreign(&cache, "1.6.0");

    let err = Indexer::new(CacheManager::new(temp.path()), IndexConfig::default())
        .index(temp.path(), false)
        .expect_err("a foreign cache must not be written");

    assert_eq!(kind(&err), "CacheVersionMismatch", "got: {err:#}");
    let msg = format!("{err:#}");
    assert!(msg.contains("1.6.0"), "must name the owner: {msg}");
    assert!(
        msg.contains(env!("CARGO_PKG_VERSION")),
        "must name this binary: {msg}"
    );
    assert!(msg.contains("abc1234"), "must name the owner's sha: {msg}");
}

#[test]
fn a_reader_still_gets_results_but_is_told_not_to_trust_them() {
    let temp = indexed();
    let cache = CacheManager::new(temp.path());
    stamp_foreign(&cache, "1.6.0");

    let engine = QueryEngine::new(CacheManager::new(temp.path()));
    let response = engine
        .search_with_metadata(
            "version_guard_token",
            QueryFilter {
                suppress_output: true,
                ..Default::default()
            },
        )
        .expect("a version mismatch must NOT be an error for a reader");

    assert_eq!(
        response.pagination.total,
        Some(1),
        "the reader must still see results"
    );
    assert!(
        !response.can_trust_results,
        "but must not be told to trust them"
    );

    let warning = response.warning.expect("a warning naming the owner");
    assert!(warning.reason.contains("1.6.0"), "{}", warning.reason);
    assert_eq!(
        warning.action_required, "index_project",
        "the advice must name the MCP tool"
    );
}

#[test]
fn a_version_mismatch_is_not_classified_as_corruption() {
    let temp = indexed();
    let cache = CacheManager::new(temp.path());
    stamp_foreign(&cache, "1.6.0");

    // This is the load-bearing assertion. If a mismatch surfaced as CacheCorrupted,
    // the MCP layer would force-rebuild around the refusal and reinstate the exact
    // stampede that produced `content.bin is too small`.
    assert!(
        CacheManager::new(temp.path()).validate().is_ok(),
        "validate() must not treat a version mismatch as corruption"
    );
}

#[test]
fn force_takes_ownership() {
    let temp = indexed();
    let cache = CacheManager::new(temp.path());
    stamp_foreign(&cache, "1.6.0");

    // `force` clears the cache first, which is what taking ownership means.
    cache.clear().unwrap();
    let stats = Indexer::new(CacheManager::new(temp.path()), IndexConfig::default())
        .index(temp.path(), false)
        .expect("a forced rebuild must succeed");
    assert_eq!(stats.total_files, 1);

    let owner = CacheManager::new(temp.path()).cache_owner();
    assert_eq!(
        owner.map(|(v, _)| v),
        Some(env!("CARGO_PKG_VERSION").to_string()),
        "the rebuild must re-stamp ownership"
    );
}

#[test]
fn the_env_escape_hatch_allows_a_write() {
    let temp = indexed();
    stamp_foreign(&CacheManager::new(temp.path()), "1.6.0");

    // Developers flip CACHE_SCHEMA_HASH on every branch switch, so there must be a
    // way past the refusal that does not discard the cache.
    let cache = CacheManager::new(temp.path());
    assert!(cache.assert_writable(false).is_err());
    assert!(
        cache.assert_writable(true).is_ok(),
        "force must always be permitted"
    );
}

#[test]
fn a_matching_cache_is_writable_and_a_missing_one_is_too() {
    let temp = indexed();
    assert!(
        CacheManager::new(temp.path())
            .assert_writable(false)
            .is_ok()
    );

    let empty = TempDir::new().unwrap();
    assert!(
        CacheManager::new(empty.path())
            .assert_writable(false)
            .is_ok(),
        "a workspace with no cache yet must be writable"
    );
}

#[test]
fn a_pre_1_7_2_cache_without_stamps_is_adopted_not_refused() {
    let temp = indexed();
    let cache = CacheManager::new(temp.path());

    // Simulate a real pre-1.7.2 cache: schema_hash present (that existed), but no
    // ownership stamps (those are new). Every existing user's cache looks like this,
    // so refusing it would break every upgrade.
    let conn = open_meta_db(cache.path().join("meta.db")).unwrap();
    conn.execute(
        "DELETE FROM statistics WHERE key IN ('writer_version', 'writer_git_sha')",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT OR REPLACE INTO statistics (key, value, updated_at) VALUES ('schema_hash', 'oldhash', 0)",
        [],
    )
    .unwrap();
    drop(conn);

    assert!(
        cache.assert_writable(false).is_ok(),
        "an unstamped cache is an upgrade, not a conflict"
    );
}
