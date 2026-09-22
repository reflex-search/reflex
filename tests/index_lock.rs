//! Workspace index lock (1.7.0): two indexers must never write the same
//! `.reflex/` at once.

use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use reflex::atomic_write::IndexLock;
use reflex::cache::CacheManager;
use reflex::errors::ReflexError;
use reflex::indexer::Indexer;
use reflex::models::IndexConfig;
use tempfile::TempDir;

const RFX: &str = env!("CARGO_BIN_EXE_rfx");

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

#[test]
fn second_indexer_gets_index_locked_error() {
    let temp = workspace();
    let root = temp.path();
    let cache_dir = root.join(".reflex");

    let held = IndexLock::try_acquire(&cache_dir).unwrap().unwrap();

    let err = Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .expect_err("must not index while the lock is held");
    assert_eq!(kind(&err), "IndexLocked", "{err}");
    assert!(err.to_string().contains("index.lock"), "{err}");

    drop(held);
    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .expect("lock released on drop");
    assert!(CacheManager::new(root).validate().is_ok());
}

#[test]
fn lock_wait_secs_waits_for_release() {
    let temp = workspace();
    let root = temp.path();
    let cache_dir = root.join(".reflex");

    let held = IndexLock::try_acquire(&cache_dir).unwrap().unwrap();
    let releaser = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(400));
        drop(held);
    });

    let config = IndexConfig {
        lock_wait_secs: 10,
        ..Default::default()
    };
    let started = Instant::now();
    Indexer::new(CacheManager::new(root), config)
        .index(root, false)
        .expect("must acquire once released");
    assert!(
        started.elapsed() >= Duration::from_millis(300),
        "must actually have waited for the release"
    );
    releaser.join().unwrap();
}

#[test]
fn force_clear_refuses_while_locked() {
    let temp = workspace();
    let root = temp.path();
    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .unwrap();

    let _held = IndexLock::try_acquire(&root.join(".reflex"))
        .unwrap()
        .unwrap();
    let err = CacheManager::new(root)
        .clear()
        .expect_err("clear must not delete a cache another indexer is writing");
    assert_eq!(kind(&err), "IndexLocked", "{err}");
    assert!(CacheManager::new(root).exists(), "nothing may be deleted");
}

#[test]
fn cross_process_cli_waits_for_in_process_lock() {
    // The `rfx index` CLI waits up to 30 s for the lock. Hold it here, spawn
    // the CLI, confirm it is still waiting after a moment, release, and the
    // CLI must then finish successfully.
    let temp = workspace();
    let root: &Path = temp.path();
    let cache_dir = root.join(".reflex");

    let held = IndexLock::try_acquire(&cache_dir).unwrap().unwrap();

    let mut child = Command::new(RFX)
        .arg("index")
        .arg(root)
        .arg("--quiet")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();

    std::thread::sleep(Duration::from_millis(1500));
    assert!(
        child.try_wait().unwrap().is_none(),
        "CLI must block on the lock instead of racing or failing"
    );

    drop(held);
    let status = child.wait().unwrap();
    assert!(
        status.success(),
        "CLI must succeed once the lock is released"
    );
    assert!(CacheManager::new(root).validate().is_ok());
}
