//! Crash safety of the index writer (1.7.0).
//!
//! `content.bin` and `trigrams.bin` are written to `<name>.tmp`, synced, then
//! renamed into place. Killing the indexer at a random moment must therefore
//! never leave a short binary behind: after the kill the cache is either
//! valid, absent, or incomplete (a file missing) — never "too small".

use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use reflex::cache::CacheManager;
use reflex::indexer::Indexer;
use reflex::models::IndexConfig;
use tempfile::TempDir;

const RFX: &str = env!("CARGO_BIN_EXE_rfx");

/// A workspace large enough that a debug-build `rfx index` runs for a while.
fn big_workspace(files: usize) -> TempDir {
    let temp = TempDir::new().unwrap();
    let src = temp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    for i in 0..files {
        let body = format!(
            "// file {i}\npub fn func_{i}(x: u32) -> u32 {{\n    let marker_{i} = x + {i};\n    marker_{i} * 2\n}}\n\n#[cfg(test)]\nmod tests_{i} {{\n    #[test]\n    fn t_{i}() {{ assert_eq!(super::func_{i}(1), {}); }}\n}}\n",
            (1 + i) * 2
        );
        fs::write(src.join(format!("m{i}.rs")), body).unwrap();
    }
    temp
}

fn rfx_index(root: &Path) -> std::process::Child {
    Command::new(RFX)
        .arg("index")
        .arg(root)
        .arg("--quiet")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn rfx index")
}

/// Cheap pseudo-random in [lo, hi) from the clock; no rand dev-dep needed.
fn jitter_ms(lo: u64, hi: u64, salt: u64) -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as u64;
    lo + (nanos ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15)) % (hi - lo)
}

fn assert_never_short(root: &Path, iteration: usize) {
    match CacheManager::new(root).validate() {
        Ok(()) => {}
        Err(e) => {
            let msg = e.to_string();
            assert!(
                !msg.contains("too small") && !msg.contains("wrong magic"),
                "iteration {iteration}: killed indexer left a short/garbage index: {msg}"
            );
        }
    }
    // No matter the state, no *.tmp file may masquerade as an index.
    for name in ["content.bin", "trigrams.bin"] {
        let final_path = root.join(".reflex").join(name);
        if final_path.exists() {
            let bytes = fs::read(&final_path).unwrap();
            assert!(
                bytes.len() >= 4,
                "iteration {iteration}: {name} exists but is {} bytes",
                bytes.len()
            );
        }
    }
}

#[test]
fn killed_indexer_never_leaves_truncated_index() {
    let temp = big_workspace(2000);
    let root = temp.path();
    let iterations = if cfg!(windows) { 3 } else { 5 };

    for i in 0..iterations {
        let mut child = rfx_index(root);
        std::thread::sleep(Duration::from_millis(jitter_ms(20, 400, i as u64)));
        let _ = child.kill();
        let _ = child.wait();
        assert_never_short(root, i);
    }

    // A full run afterwards must produce a valid, queryable index.
    //
    // Capture stderr here: the spawned runs above deliberately discard it, but when
    // THIS one fails the reason is the whole point. A bare `must succeed` told a CI
    // reader nothing.
    let out = Command::new(RFX)
        .arg("index")
        .arg(root)
        .arg("--quiet")
        .output()
        .expect("spawn rfx index");
    assert!(
        out.status.success(),
        "final rfx index must succeed (exit {:?})\nstderr:\n{}\nstdout:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout),
    );
    assert!(
        CacheManager::new(root).validate().is_ok(),
        "final index must validate"
    );
    let engine = reflex::query::QueryEngine::new(CacheManager::new(root));
    let results = engine
        .search("marker_1999", reflex::query::QueryFilter::default())
        .unwrap();
    assert!(!results.is_empty(), "rebuilt index must answer queries");
    assert!(
        !root.join(".reflex/content.bin.tmp").exists(),
        "no temp file may remain after a clean run"
    );
}

#[test]
fn stale_tmp_removed_on_index_start() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    fs::write(root.join("a.rs"), "fn alpha() {}\n").unwrap();
    let reflex_dir = root.join(".reflex");
    fs::create_dir_all(&reflex_dir).unwrap();
    fs::write(reflex_dir.join("content.bin.tmp"), b"half-written").unwrap();
    fs::write(reflex_dir.join("trigrams.bin.tmp"), b"half-written").unwrap();

    let cache = CacheManager::new(root);
    Indexer::new(cache, IndexConfig::default())
        .index(root, false)
        .unwrap();

    assert!(!reflex_dir.join("content.bin.tmp").exists());
    assert!(!reflex_dir.join("trigrams.bin.tmp").exists());
    assert!(CacheManager::new(root).validate().is_ok());
}

#[test]
fn reindex_replaces_binaries_atomically_and_keeps_old_readable() {
    // A reader holding the previous content.bin open must keep a complete
    // file while a re-index lands (rename swaps the inode; no truncate).
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    fs::write(root.join("a.rs"), "fn alpha() {}\n").unwrap();
    let cache = CacheManager::new(root);
    Indexer::new(cache, IndexConfig::default())
        .index(root, false)
        .unwrap();

    // Only the Unix assertion below reads this, but it must be captured BEFORE the
    // reindex lands. Bound under cfg so Windows does not see an unused variable —
    // CI runs clippy with -D warnings.
    #[cfg(unix)]
    let before = fs::read(root.join(".reflex/content.bin")).unwrap();
    let held = fs::File::open(root.join(".reflex/content.bin")).unwrap();

    fs::write(root.join("b.rs"), "fn beta() {}\n").unwrap();
    let started = Instant::now();
    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .unwrap();
    assert!(started.elapsed() < Duration::from_secs(60));

    // On Unix the held descriptor still sees the full old bytes.
    #[cfg(unix)]
    {
        use std::io::Read;
        let mut still = Vec::new();
        (&held).read_to_end(&mut still).unwrap();
        assert_eq!(still, before, "old inode must stay intact for open readers");
    }
    drop(held);
    assert!(CacheManager::new(root).validate().is_ok());
}

/// A partial-batch directory left by an indexer that died between two trigram
/// batches (2.0.0) is removed on the next start, like a stale `.tmp`.
#[test]
fn stale_partial_batch_dir_removed_on_index_start() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    fs::write(root.join("a.rs"), "fn alpha() {}\n").unwrap();
    let reflex_dir = root.join(".reflex");
    let partials = reflex_dir.join("trigram_temp");
    fs::create_dir_all(&partials).unwrap();
    fs::write(partials.join("partial_0.bin"), b"half-written").unwrap();

    let cache = CacheManager::new(root);
    Indexer::new(cache, IndexConfig::default())
        .index(root, false)
        .unwrap();

    assert!(!partials.exists(), "trigram_temp must be removed");
    assert!(CacheManager::new(root).validate().is_ok());
}

/// Killing a multi-batch build (tiny batches force on-disk partials) never
/// leaves a short binary or a foreign-key violation behind.
#[test]
fn killed_multi_batch_indexer_never_leaves_truncated_index() {
    let temp = big_workspace(300);
    let root = temp.path();
    let reflex_dir = root.join(".reflex");

    for iteration in 0..4 {
        let mut child = Command::new(RFX)
            .arg("index")
            .arg(root)
            .arg("--quiet")
            .env("REFLEX_INDEX_BATCH_FILES", "40")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn rfx index");
        std::thread::sleep(Duration::from_millis(jitter_ms(30, 400, iteration as u64)));
        let _ = child.kill();
        let _ = child.wait();
        assert_never_short(root, iteration);
    }

    // A clean run afterwards leaves no partials behind and a valid cache.
    let cache = CacheManager::new(root);
    let mut indexer = Indexer::new(cache, IndexConfig::default());
    indexer.set_batch_limits(40, u64::MAX);
    indexer.index(root, false).unwrap();
    assert!(!reflex_dir.join("trigram_temp").exists());
    assert!(CacheManager::new(root).validate().is_ok());
    let conn = reflex::cache::open_meta_db(reflex_dir.join("meta.db")).unwrap();
    let violations: i64 = conn
        .query_row("SELECT COUNT(*) FROM pragma_foreign_key_check", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(violations, 0, "foreign_key_check reported violations");
}
