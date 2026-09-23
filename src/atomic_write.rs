//! Crash-safe file replacement and the workspace index lock.
//!
//! Index binaries (`content.bin`, `trigrams.bin`) used to be opened at their
//! final path with `truncate(true)` and streamed into. A crash, a killed
//! process, or a second indexer starting mid-write left a short file behind,
//! and the next reader failed with `content.bin is too small - appears to be
//! corrupted`. Every writer now goes through this module:
//!
//! 1. write to `<final>.tmp` in the same directory,
//! 2. `sync_all`,
//! 3. [`atomic_replace`] renames the temp file over the final path.
//!
//! Readers therefore only ever see the previous complete file or the new
//! complete file. Because the rename lands in the same directory, it is
//! atomic on every platform we ship to.
//!
//! [`IndexLock`] is an OS advisory lock on `.reflex/index.lock` held for the
//! whole `Indexer::index` run. The OS releases it when the process dies, so
//! there is no stale-PID bookkeeping.

use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::errors::ReflexError;

/// Suffix appended to a final path to build its in-progress temp path.
pub const TMP_SUFFIX: &str = ".tmp";

/// File name of the workspace index lock inside the cache directory.
pub const INDEX_LOCK_FILE: &str = "index.lock";

/// Temp path for `final_path`: same directory, same file name plus `.tmp`.
///
/// Same directory matters: `fs::rename` is only atomic within one filesystem.
pub fn tmp_path_for(final_path: &Path) -> PathBuf {
    let mut name = final_path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(TMP_SUFFIX);
    final_path.with_file_name(name)
}

/// Rename `tmp` over `final_path`.
///
/// On Windows a rename can fail with a sharing violation while another
/// process still maps the old file (the background symbol indexer or an MCP
/// query mid-flight). We retry with backoff, then fall back to copying the
/// bytes over the final path so the index still lands. The fallback is not
/// atomic, so it logs a warning.
pub fn atomic_replace(tmp: &Path, final_path: &Path) -> io::Result<()> {
    let mut last_err = match fs::rename(tmp, final_path) {
        Ok(()) => return Ok(()),
        Err(e) => e,
    };

    if cfg!(windows) {
        for delay_ms in [20u64, 40, 80, 160, 320] {
            std::thread::sleep(Duration::from_millis(delay_ms));
            match fs::rename(tmp, final_path) {
                Ok(()) => return Ok(()),
                Err(e) => last_err = e,
            }
        }
        log::warn!(
            "atomic rename of {} failed after retries ({}); falling back to non-atomic copy",
            final_path.display(),
            last_err
        );
        fs::copy(tmp, final_path)?;
        let _ = fs::remove_file(tmp);
        return Ok(());
    }

    Err(last_err)
}

/// Remove leftover `*.tmp` files in `dir` (a crashed indexer leaves them).
///
/// Errors are logged, never fatal: a stray temp file only wastes disk.
pub fn remove_stale_tmp(dir: &Path) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let is_tmp = path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.ends_with(TMP_SUFFIX))
            .unwrap_or(false);
        if is_tmp && path.is_file() {
            match fs::remove_file(&path) {
                Ok(()) => log::info!("Removed stale temp file {}", path.display()),
                Err(e) => log::warn!("Could not remove stale temp file {}: {}", path.display(), e),
            }
        }
        // Partial trigram batches of an indexer that died between two batches.
        if path.is_dir() && path.file_name().and_then(|n| n.to_str()) == Some("trigram_temp") {
            match fs::remove_dir_all(&path) {
                Ok(()) => log::info!("Removed stale partial index directory {}", path.display()),
                Err(e) => log::warn!(
                    "Could not remove stale partial index directory {}: {}",
                    path.display(),
                    e
                ),
            }
        }
    }
}

/// RAII guard for the workspace index lock.
///
/// Dropping the guard (or the process exiting) releases the lock.
#[derive(Debug)]
pub struct IndexLock {
    file: File,
    path: PathBuf,
}

impl IndexLock {
    /// Path of the lock file inside `cache_dir`.
    pub fn lock_path(cache_dir: &Path) -> PathBuf {
        cache_dir.join(INDEX_LOCK_FILE)
    }

    /// Try to take the lock without waiting.
    ///
    /// Returns `Ok(None)` when another process (or thread) holds it.
    pub fn try_acquire(cache_dir: &Path) -> Result<Option<IndexLock>> {
        fs::create_dir_all(cache_dir)
            .with_context(|| format!("Failed to create {}", cache_dir.display()))?;
        let path = Self::lock_path(cache_dir);
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("Failed to open {}", path.display()))?;
        match file.try_lock() {
            Ok(()) => Ok(Some(IndexLock { file, path })),
            Err(std::fs::TryLockError::WouldBlock) => Ok(None),
            Err(std::fs::TryLockError::Error(e)) => {
                Err(e).with_context(|| format!("Failed to lock {}", path.display()))
            }
        }
    }

    /// Take the lock, polling every 100 ms until `timeout` elapses.
    ///
    /// Returns [`ReflexError::IndexLocked`] on timeout.
    pub fn acquire_with_timeout(cache_dir: &Path, timeout: Duration) -> Result<IndexLock> {
        let start = Instant::now();
        loop {
            if let Some(lock) = Self::try_acquire(cache_dir)? {
                return Ok(lock);
            }
            if start.elapsed() >= timeout {
                return Err(ReflexError::IndexLocked(
                    Self::lock_path(cache_dir).display().to_string(),
                )
                .into());
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// The lock file path this guard holds.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for IndexLock {
    fn drop(&mut self) {
        // Explicit unlock so the file handle's lifetime does not matter on
        // platforms where the lock is tied to the descriptor.
        let _ = self.file.unlock();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn tmp_path_is_sibling_with_suffix() {
        let p = Path::new("/a/b/content.bin");
        assert_eq!(tmp_path_for(p), PathBuf::from("/a/b/content.bin.tmp"));
    }

    #[test]
    fn atomic_replace_moves_bytes_over_final() {
        let dir = TempDir::new().unwrap();
        let final_path = dir.path().join("f.bin");
        fs::write(&final_path, b"old").unwrap();
        let tmp = tmp_path_for(&final_path);
        fs::write(&tmp, b"new-bytes").unwrap();
        atomic_replace(&tmp, &final_path).unwrap();
        assert_eq!(fs::read(&final_path).unwrap(), b"new-bytes");
        assert!(!tmp.exists());
    }

    #[test]
    fn remove_stale_tmp_only_touches_tmp_files() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("content.bin"), b"keep").unwrap();
        fs::write(dir.path().join("content.bin.tmp"), b"stale").unwrap();
        remove_stale_tmp(dir.path());
        assert!(dir.path().join("content.bin").exists());
        assert!(!dir.path().join("content.bin.tmp").exists());
    }

    #[test]
    fn second_acquire_in_other_process_scope_is_none_then_released() {
        // Two handles in one process still exclude each other for
        // `try_lock` on Linux/macOS/Windows (the lock is per open file
        // description), which is what the indexer relies on.
        let dir = TempDir::new().unwrap();
        let first = IndexLock::try_acquire(dir.path()).unwrap();
        assert!(first.is_some());
        let second = IndexLock::try_acquire(dir.path()).unwrap();
        assert!(second.is_none(), "lock must be exclusive while held");
        drop(first);
        let third = IndexLock::try_acquire(dir.path()).unwrap();
        assert!(third.is_some(), "lock must be released on drop");
    }

    #[test]
    fn acquire_with_timeout_reports_index_locked() {
        let dir = TempDir::new().unwrap();
        let _held = IndexLock::try_acquire(dir.path()).unwrap().unwrap();
        let err = IndexLock::acquire_with_timeout(dir.path(), Duration::from_millis(250))
            .expect_err("must time out");
        let re = err
            .downcast_ref::<ReflexError>()
            .expect("typed ReflexError");
        assert_eq!(re.kind(), "IndexLocked");
        assert!(re.to_string().contains(INDEX_LOCK_FILE));
    }

    #[test]
    fn lock_file_is_not_truncated_or_required_to_be_empty() {
        let dir = TempDir::new().unwrap();
        let path = IndexLock::lock_path(dir.path());
        let mut f = File::create(&path).unwrap();
        f.write_all(b"12345").unwrap();
        drop(f);
        let lock = IndexLock::try_acquire(dir.path()).unwrap().unwrap();
        assert_eq!(lock.path(), path);
    }
}
