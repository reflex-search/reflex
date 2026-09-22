//! Background symbol indexer for transparent caching
//!
//! This module provides background processing to parse symbols from all indexed
//! files and populate the symbol cache. It runs as a separate process spawned by
//! `rfx index`, allowing users to continue working while symbols are being indexed.

use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::cache::CacheManager;
use crate::content_store::ContentReader;
use crate::parsers::ParserFactory;
use crate::symbol_cache::SymbolCache;

/// Lock file name to prevent concurrent indexing
const LOCK_FILE: &str = "indexing.lock";

/// Maximum age of a lock file before it's considered stale.
///
/// Lowered from 1 hour to 15 minutes in 1.7.2, because liveness is now decided by
/// checking whether the recorded pid is actually alive. Age is only the last-resort
/// fallback for platforms where that check is unavailable.
const LOCK_MAX_AGE: std::time::Duration = std::time::Duration::from_secs(900);

/// Status file name for progress tracking
const STATUS_FILE: &str = "indexing.status";

/// Sentinel file asking a running symbol pass to stop at the next batch.
///
/// A symbol pass over a large repo runs for minutes. Making `rfx index` wait that
/// long is the complaint this fixes, so instead the indexer asks the pass to yield.
/// `rfx index` re-spawns it when it finishes, so no work is lost.
const CANCEL_FILE: &str = "indexing.cancel";

/// Who holds `indexing.lock`.
///
/// 1.7.1 and earlier wrote a bare pid. This is read back as JSON, falling back to
/// the bare-integer form so an upgrade does not orphan an in-flight lock.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockHolder {
    pub pid: u32,
    /// RFC3339 start time, absent for a legacy bare-pid lock.
    #[serde(default)]
    pub started_at: Option<String>,
}

impl LockHolder {
    /// Start time as `HH:MM:SS` for human- and agent-facing messages.
    pub fn started_clock(&self) -> String {
        self.started_at
            .as_deref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| {
                dt.with_timezone(&chrono::Local)
                    .format("%H:%M:%S")
                    .to_string()
            })
            .unwrap_or_else(|| "unknown".to_string())
    }
}

/// Whether a pid belongs to a live `rfx index-symbols-internal` process.
///
/// Checked before honouring a lock, because the previous mtime-only rule kept a
/// crashed pass's lock for a full hour. No new dependency:
///
/// * Linux — `/proc/<pid>` exists AND its cmdline names the subcommand. The cmdline
///   check also defeats pid reuse, which a bare `kill(pid, 0)` cannot.
/// * macOS — one `ps -o command= -p <pid>`, only on this path.
/// * elsewhere — unknown, so fall back to the age rule rather than reap a live pass.
fn pid_is_live_symbol_pass(pid: u32) -> Option<bool> {
    // We hold our own lock, so we are alive by definition — no need to inspect argv.
    // This also covers `rfx index` running the pass in-process rather than detached.
    if pid == std::process::id() {
        return Some(true);
    }

    #[cfg(target_os = "linux")]
    {
        let proc_dir = std::path::PathBuf::from(format!("/proc/{}", pid));
        if !proc_dir.exists() {
            return Some(false);
        }
        // NUL-separated argv. A recycled pid running something else is not our pass.
        match std::fs::read(proc_dir.join("cmdline")) {
            Ok(raw) => Some(String::from_utf8_lossy(&raw).contains("index-symbols-internal")),
            // The process exists but we cannot read its argv (different user).
            // Treat it as live: reaping a live pass is far worse than keeping a
            // stale lock, which the age rule will clear anyway.
            Err(_) => Some(true),
        }
    }

    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("ps")
            .args(["-o", "command=", "-p", &pid.to_string()])
            .output()
            .ok()?;
        if !out.status.success() {
            return Some(false);
        }
        Some(String::from_utf8_lossy(&out.stdout).contains("index-symbols-internal"))
    }

    // Windows and everything else: liveness is not determinable without either a
    // new dependency or an untested `tasklist` shell-out. `None` means "fall back to
    // the age rule", so a crashed pass holds its lock for at most LOCK_MAX_AGE
    // (15 minutes) instead of being reaped at once. Degraded, not broken: `rfx index`
    // still asks the pass to yield and reports `SymbolIndexingInProgress` with the
    // pid, rather than a raw SQLite error.
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = pid;
        None
    }
}

/// Whether this platform can tell a live symbol pass from a dead one.
///
/// Exposed so tests assert the behaviour the platform actually guarantees, rather
/// than the behaviour Linux happens to have.
pub const fn pid_liveness_supported() -> bool {
    cfg!(any(target_os = "linux", target_os = "macos"))
}

/// Indexing progress status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingStatus {
    /// Current state of the indexer
    pub state: IndexerState,
    /// Total files to process
    pub total_files: usize,
    /// Files processed so far
    pub processed_files: usize,
    /// Files that had symbols cached
    pub cached_files: usize,
    /// Files that were newly parsed
    pub parsed_files: usize,
    /// Files that failed to parse
    pub failed_files: usize,
    /// Start time (ISO 8601)
    pub started_at: String,
    /// Last update time (ISO 8601)
    pub updated_at: String,
    /// Completion time (ISO 8601, None if not finished)
    pub completed_at: Option<String>,
    /// Error message if failed
    pub error: Option<String>,
    /// PID of the process doing the work.
    ///
    /// Previously this lived only in `indexing.lock`, so a caller reading the status
    /// could not name the process it was waiting for.
    #[serde(default)]
    pub pid: u32,
    /// Which stage of the pass is running: `filtering`, `parsing`, `writing`,
    /// `cleanup`. In 1.7.1 a pass that had finished its files still held the database
    /// for minutes inside `cleanup_stale()`, and looked identical to a hang.
    #[serde(default)]
    pub phase: String,
    /// File currently being parsed, when known.
    #[serde(default)]
    pub current_file: Option<String>,
}

/// Indexer state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum IndexerState {
    /// Indexer is currently running
    Running,
    /// Indexer completed successfully
    Completed,
    /// Indexer failed with error
    Failed,
    /// Indexer yielded because an `rfx index` asked for the database.
    ///
    /// Not a failure: the remaining files are re-queued when `rfx index` re-spawns
    /// the pass after it finishes.
    Cancelled,
}

/// Check if a lock file is stale based on its modification time
///
/// A lock file is considered stale if its mtime is older than `LOCK_MAX_AGE`.
/// This allows recovery from crashed indexer processes that didn't clean up
/// their lock file (SIGKILL, OOM, power loss, etc.).
fn is_lock_stale(lock_path: &Path) -> bool {
    let metadata = match std::fs::metadata(lock_path) {
        Ok(m) => m,
        Err(_) => return false, // Can't read => not stale, let caller handle
    };
    let modified = match metadata.modified() {
        Ok(t) => t,
        Err(_) => return false,
    };
    match modified.elapsed() {
        Ok(age) => age > LOCK_MAX_AGE,
        Err(_) => false, // Clock skew — don't remove
    }
}

/// Background symbol indexer
pub struct BackgroundIndexer {
    workspace_path: PathBuf,
    cache_path: PathBuf,
    status: IndexingStatus,
    batch_size: usize,
}

impl BackgroundIndexer {
    /// Create a new background indexer
    ///
    /// # Arguments
    /// * `workspace_path` - Path to the workspace root (e.g., ".")
    pub fn new(workspace_path: &Path) -> Result<Self> {
        let now = chrono::Utc::now().to_rfc3339();

        // Create CacheManager to get the cache directory path
        let cache_mgr = CacheManager::new(workspace_path);
        let cache_path = cache_mgr.path().to_path_buf();

        Ok(Self {
            workspace_path: workspace_path.to_path_buf(),
            cache_path,
            status: IndexingStatus {
                state: IndexerState::Running,
                total_files: 0,
                processed_files: 0,
                cached_files: 0,
                parsed_files: 0,
                failed_files: 0,
                started_at: now.clone(),
                updated_at: now,
                completed_at: None,
                error: None,
                pid: std::process::id(),
                phase: "starting".to_string(),
                current_file: None,
            },
            // 128, not 500: a chunk is the unit of both progress reporting and
            // cancellation, so a wide chunk means a status frozen for minutes and a
            // slow yield to a waiting `rfx index`.
            batch_size: 128,
        })
    }

    /// Who currently holds `indexing.lock`, if anyone.
    ///
    /// Returns `None` when there is no lock, or when the lock is stale and has been
    /// reaped. A lock is stale when its recorded pid is not a live
    /// `rfx index-symbols-internal`, or (where liveness is unknowable) when its mtime
    /// is older than `LOCK_MAX_AGE`.
    pub fn lock_holder(cache_dir: &Path) -> Option<LockHolder> {
        let lock_path = cache_dir.join(LOCK_FILE);
        let raw = std::fs::read_to_string(&lock_path).ok()?;

        // JSON since 1.7.2; bare pid before that.
        let holder: LockHolder = serde_json::from_str(raw.trim()).unwrap_or_else(|_| LockHolder {
            pid: raw.trim().parse().unwrap_or(0),
            started_at: None,
        });

        let stale = match pid_is_live_symbol_pass(holder.pid) {
            Some(true) => false,
            Some(false) => {
                log::warn!(
                    "Reaping indexing lock: pid {} is not a live symbol pass",
                    holder.pid
                );
                true
            }
            // Liveness unknown on this platform — fall back to the age rule.
            None => {
                let old = is_lock_stale(&lock_path);
                if old {
                    log::warn!(
                        "Removing stale indexing lock file (older than {:?})",
                        LOCK_MAX_AGE
                    );
                }
                old
            }
        };

        if stale {
            let _ = std::fs::remove_file(&lock_path);
            return None;
        }

        Some(holder)
    }

    /// Check if an indexing process is already running.
    pub fn is_running(cache_dir: &Path) -> bool {
        Self::lock_holder(cache_dir).is_some()
    }

    /// Ask a running symbol pass to stop at its next batch.
    ///
    /// Cooperative, not a kill: the pass finishes the batch it is on, writes its
    /// status and releases the lock, so `meta.db` is never left mid-write.
    pub fn request_cancel(cache_dir: &Path) -> std::io::Result<()> {
        std::fs::write(cache_dir.join(CANCEL_FILE), b"")
    }

    /// Whether a cancel has been requested.
    pub fn cancel_requested(cache_dir: &Path) -> bool {
        cache_dir.join(CANCEL_FILE).exists()
    }

    /// Clear any outstanding cancel request.
    pub fn clear_cancel(cache_dir: &Path) {
        let _ = std::fs::remove_file(cache_dir.join(CANCEL_FILE));
    }

    /// Get the current indexing status (if available)
    pub fn get_status(cache_dir: &Path) -> Result<Option<IndexingStatus>> {
        let status_path = cache_dir.join(STATUS_FILE);

        if !status_path.exists() {
            return Ok(None);
        }

        let status_json =
            std::fs::read_to_string(&status_path).context("Failed to read indexing status")?;

        let status: IndexingStatus =
            serde_json::from_str(&status_json).context("Failed to parse indexing status")?;

        Ok(Some(status))
    }

    /// Acquire lock file (returns error if already locked)
    ///
    /// If a stale lock file is detected, it is removed before acquiring.
    /// This provides defense-in-depth alongside the `is_running()` check.
    fn acquire_lock(&self) -> Result<File> {
        let lock_path = self.cache_path.join(LOCK_FILE);

        // Same liveness rule as `is_running`, so the two can never disagree about
        // who holds the lock. `lock_holder` reaps a dead holder as a side effect.
        if Self::lock_holder(&self.cache_path).is_some() {
            anyhow::bail!("Indexing already in progress (lock file exists)");
        }

        let mut lock_file = File::create(&lock_path).context("Failed to create lock file")?;

        let pid = std::process::id();
        let holder = LockHolder {
            pid,
            started_at: Some(self.status.started_at.clone()),
        };
        // JSON so a waiting indexer can report "pid N, started HH:MM:SS" instead of a
        // bare number. Readers accept the old bare-pid form too.
        writeln!(lock_file, "{}", serde_json::to_string(&holder)?)?;
        lock_file.flush()?;

        log::debug!("Acquired indexing lock (PID: {})", pid);
        Ok(lock_file)
    }

    /// Release lock file
    fn release_lock(&self) -> Result<()> {
        let lock_path = self.cache_path.join(LOCK_FILE);

        if lock_path.exists() {
            std::fs::remove_file(&lock_path).context("Failed to remove lock file")?;
            log::debug!("Released indexing lock");
        }

        Ok(())
    }

    /// Write current status to status file
    fn write_status(&mut self) -> Result<()> {
        self.status.updated_at = chrono::Utc::now().to_rfc3339();

        let status_path = self.cache_path.join(STATUS_FILE);
        let status_json =
            serde_json::to_string_pretty(&self.status).context("Failed to serialize status")?;

        std::fs::write(&status_path, status_json).context("Failed to write status file")?;

        Ok(())
    }

    /// Run the background indexer
    ///
    /// This processes all indexed files, parsing symbols and caching them.
    /// Progress is written to `.reflex/indexing.status` and can be monitored.
    pub fn run(&mut self) -> Result<()> {
        let start_time = Instant::now();

        // Clear any cancel left over from a previous run. An indexer that died
        // between requesting a cancel and clearing it would otherwise stop this pass
        // before it began.
        Self::clear_cancel(&self.cache_path);

        // Acquire lock (fails if already running)
        let _lock_file = self
            .acquire_lock()
            .context("Failed to acquire indexing lock")?;

        // Ensure lock is released even on panic
        let cache_path = self.cache_path.clone();
        let _guard = scopeguard::guard((), move |_| {
            let _ = std::fs::remove_file(cache_path.join(LOCK_FILE));
        });

        // Run indexing
        let result = self.run_internal();

        // Update status based on result
        match result {
            // A cancelled pass also returns Ok — it stopped cleanly, it did not fail.
            // Don't relabel it Completed, or a caller cannot tell a finished index
            // from one that yielded with files still to parse.
            Ok(()) if self.status.state == IndexerState::Cancelled => {
                self.status.completed_at = Some(chrono::Utc::now().to_rfc3339());
                log::info!(
                    "Symbol indexing cancelled after {} of {} files in {:.2}s",
                    self.status.processed_files,
                    self.status.total_files,
                    start_time.elapsed().as_secs_f64()
                );
            }
            Ok(()) => {
                self.status.state = IndexerState::Completed;
                self.status.phase = "done".to_string();
                self.status.completed_at = Some(chrono::Utc::now().to_rfc3339());
                log::info!(
                    "Symbol indexing completed: {} files processed ({} cached, {} parsed, {} failed) in {:.2}s",
                    self.status.processed_files,
                    self.status.cached_files,
                    self.status.parsed_files,
                    self.status.failed_files,
                    start_time.elapsed().as_secs_f64()
                );
            }
            Err(ref e) => {
                self.status.state = IndexerState::Failed;
                self.status.error = Some(format!("{:#}", e));
                self.status.completed_at = Some(chrono::Utc::now().to_rfc3339());
                log::error!("Symbol indexing failed: {:#}", e);
            }
        }

        // Write final status
        self.write_status()?;

        // Release lock
        self.release_lock()?;

        result
    }

    /// Internal indexing implementation with parallel processing
    fn run_internal(&mut self) -> Result<()> {
        log::info!("Starting background symbol indexing");

        // Calculate thread pool size (25-30% of available CPUs)
        let num_cpus = num_cpus::get();
        let num_threads = ((num_cpus as f32 * 0.275).ceil() as usize).max(1);

        log::info!(
            "Using {} threads for background indexing ({} CPUs available, ~27.5% utilization)",
            num_threads,
            num_cpus
        );

        // Create custom thread pool with limited threads
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .context("Failed to create thread pool")?;

        // Open cache manager and symbol cache
        let cache_mgr = CacheManager::new(&self.workspace_path);
        let symbol_cache =
            SymbolCache::open(&self.cache_path).context("Failed to open symbol cache")?;

        // Load content reader to iterate through all indexed files
        let content_path = self.cache_path.join("content.bin");

        // If content.bin doesn't exist, index is empty - nothing to do
        if !content_path.exists() {
            log::info!("No content.bin found - index is empty, nothing to process");
            self.status.total_files = 0;
            self.status.processed_files = 0;
            self.write_status()?;
            return Ok(());
        }

        let content_reader =
            ContentReader::open(&content_path).context("Failed to open content.bin")?;

        // Get file hashes across all branches (background indexer processes all files)
        let file_hashes = cache_mgr
            .load_all_hashes()
            .context("Failed to load file hashes")?;

        let total_files = content_reader.file_count();
        self.status.total_files = total_files;
        log::info!("Found {} indexed files to process", total_files);
        log::debug!(
            "Loaded {} file hashes from file_branches table",
            file_hashes.len()
        );

        // DEFENSIVE CHECK: If file_hashes is empty but we have files, this indicates a problem
        if file_hashes.is_empty() && total_files > 0 {
            log::error!(
                "CRITICAL: No file hashes found in file_branches table, but {} files exist in content.bin!",
                total_files
            );
            log::error!("This likely means:");
            log::error!("  1. The main indexer failed to populate file_branches table");
            log::error!("  2. WAL checkpoint didn't flush data before background indexer started");
            log::error!("  3. Database transaction was rolled back");

            // Try to diagnose by checking database directly
            log::error!("Attempting diagnostic query to check file_branches table...");
            anyhow::bail!(
                "No file hashes available - cannot index symbols. \
                 This is a database synchronization issue. \
                 Try running 'rfx index' again or clearing the cache with 'rfx clear'."
            );
        }

        // Write initial status
        self.write_status()?;

        // Shared state for status tracking
        let status_mutex = Arc::new(Mutex::new((0usize, 0usize, 0usize))); // (cached, parsed, failed)

        // Process files in batches
        let batch_size = self.batch_size;
        let mut processed = 0;

        // Iterate through all files in content.bin
        let file_ids: Vec<u32> = (0..total_files as u32).collect();

        // DIAGNOSTIC: Log sample paths to debug hash lookup failures
        if !file_ids.is_empty() && !file_hashes.is_empty() {
            // Log first 3 paths from content.bin
            log::debug!("=== Path Comparison Diagnostic ===");
            for sample_id in file_ids.iter().take(3) {
                if let Some(path) = content_reader.get_file_path(*sample_id) {
                    log::debug!(
                        "  content.bin path[{}]: '{}'",
                        sample_id,
                        path.to_string_lossy()
                    );
                }
            }
            // Log first 3 keys from file_hashes HashMap
            let sample_keys: Vec<_> = file_hashes.keys().take(3).collect();
            for key in sample_keys {
                log::debug!("  file_hashes key: '{}'", key);
            }
            log::debug!("=================================");
        }

        for chunk in file_ids.chunks(batch_size) {
            // Yield the database if an `rfx index` is waiting. Cooperative, so the
            // batch just written stays consistent and the lock is released cleanly.
            if Self::cancel_requested(&self.cache_path) {
                log::info!(
                    "Symbol indexing cancelled at {}/{} files (an indexer asked for the database)",
                    processed,
                    total_files
                );
                self.status.state = IndexerState::Cancelled;
                self.status.phase = "cancelled".to_string();
                self.write_status()?;
                return Ok(());
            }

            let chunk_start = Instant::now();
            self.status.phase = "filtering".to_string();

            // Build list of files to parse (with cache check)
            let files_to_parse: Vec<_> = chunk
                .iter()
                .filter_map(|&file_id| {
                    let path = content_reader.get_file_path(file_id)?;
                    let mut path_str = path.to_string_lossy().to_string();

                    // NORMALIZE: Strip "./" prefix to match database paths
                    // content.bin stores paths like "./src/main.rs"
                    // but database stores paths like "src/main.rs"
                    if path_str.starts_with("./") {
                        path_str = path_str[2..].to_string();
                    }

                    let file_hash = file_hashes.get(&path_str)?;

                    // Check if already cached
                    if symbol_cache
                        .get(&path_str, file_hash)
                        .ok()
                        .flatten()
                        .is_some()
                    {
                        // Update cached count
                        let mut status = status_mutex.lock().unwrap();
                        status.0 += 1;
                        None
                    } else {
                        Some((file_id, path_str, file_hash.clone()))
                    }
                })
                .collect();

            // Parse files in parallel using custom thread pool
            let parsed_results: Vec<_> = thread_pool.install(|| {
                files_to_parse
                    .par_iter()
                    .map(|(file_id, path_str, file_hash)| {
                        match self.parse_symbols(&content_reader, *file_id, path_str) {
                            Ok(symbols) => {
                                // Update parsed count
                                let mut status = status_mutex.lock().unwrap();
                                status.1 += 1;
                                Some((path_str.clone(), file_hash.clone(), symbols))
                            }
                            Err(e) => {
                                log::warn!("Failed to parse symbols from {}: {}", path_str, e);
                                // Update failed count
                                let mut status = status_mutex.lock().unwrap();
                                status.2 += 1;
                                None
                            }
                        }
                    })
                    .flatten()
                    .collect()
            });

            // Write batch to cache (sequential - SQLite limitation)
            let parse_done = Instant::now();

            // Announce the write BEFORE it starts. This is the step that blocks on a
            // meta.db write lock, so a status frozen in "writing" says where the time
            // is going instead of looking like a hang.
            self.status.phase = "writing".to_string();
            let _ = self.write_status();

            if !parsed_results.is_empty()
                && let Err(e) = symbol_cache.batch_set(&parsed_results)
            {
                log::error!("Failed to write symbol batch: {}", e);
                let mut status = status_mutex.lock().unwrap();
                status.2 += parsed_results.len();
            }

            // Update status counters
            processed += chunk.len();
            {
                let status = status_mutex.lock().unwrap();
                self.status.cached_files = status.0;
                self.status.parsed_files = status.1;
                self.status.failed_files = status.2;
                self.status.processed_files = processed;
            }

            // Unconditionally, once per chunk. The old `processed % 500 < batch_size`
            // guard was a no-op (batch_size was itself 500, so it was always true),
            // and the real staleness came from the chunk being 500 files wide.
            self.status.phase = "parsing".to_string();
            self.status.current_file = None;
            if let Err(e) = self.write_status() {
                log::warn!("Failed to write status: {}", e);
            }

            let total_ms = chunk_start.elapsed().as_millis();
            log::info!(
                "Symbol batch {}/{}: {}ms total ({}ms parse, {}ms write), {} files parsed",
                processed,
                total_files,
                total_ms,
                parse_done.duration_since(chunk_start).as_millis(),
                parse_done.elapsed().as_millis(),
                parsed_results.len()
            );
        }

        // Final status update
        self.status.processed_files = total_files;
        self.write_status()?;

        // Cleanup stale entries.
        //
        // This runs AFTER the final status write, so in 1.7.1 a pass spending minutes
        // here showed 1027/1027 and a frozen `updated_at` — indistinguishable from a
        // hang. Name the phase and time it.
        self.status.phase = "cleanup".to_string();
        let _ = self.write_status();
        let cleanup_start = Instant::now();

        let removed = symbol_cache
            .cleanup_stale()
            .context("Failed to cleanup stale symbols")?;

        let cleanup_ms = cleanup_start.elapsed().as_millis();
        if cleanup_ms > 1000 {
            log::warn!(
                "cleanup_stale took {}ms for {} removed entries \u{2014} check the index on symbols(file_id)",
                cleanup_ms,
                removed
            );
        }
        if removed > 0 {
            log::info!(
                "Cleaned up {} stale symbol entries in {}ms",
                removed,
                cleanup_ms
            );
        }

        Ok(())
    }

    /// Parse symbols from a file using content.bin
    fn parse_symbols(
        &self,
        content_reader: &ContentReader,
        file_id: u32,
        path: &str,
    ) -> Result<Vec<crate::models::SearchResult>> {
        // Read file contents from content.bin (memory-mapped, zero-copy)
        let source = content_reader
            .get_file_content(file_id)
            .with_context(|| format!("Failed to read file from content.bin: {}", path))?;

        // Detect language from file extension
        let extension = std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        let language = crate::models::Language::from_extension(extension);

        // Parse with appropriate parser
        let symbols = ParserFactory::parse(path, source, language)
            .with_context(|| format!("Failed to parse symbols from: {}", path))?;

        Ok(symbols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CacheManager;
    use tempfile::TempDir;

    #[test]
    fn test_indexer_lock() {
        let temp = TempDir::new().unwrap();
        let cache_mgr = CacheManager::new(temp.path());
        cache_mgr.init().unwrap();

        assert!(!BackgroundIndexer::is_running(cache_mgr.path()));

        let indexer = BackgroundIndexer::new(temp.path()).unwrap();
        let _lock = indexer.acquire_lock().unwrap();

        assert!(BackgroundIndexer::is_running(cache_mgr.path()));

        indexer.release_lock().unwrap();
        assert!(!BackgroundIndexer::is_running(cache_mgr.path()));
    }

    #[test]
    fn test_indexer_lock_prevents_concurrent() {
        let temp = TempDir::new().unwrap();
        let cache_mgr = CacheManager::new(temp.path());
        cache_mgr.init().unwrap();

        let indexer1 = BackgroundIndexer::new(temp.path()).unwrap();
        let _lock1 = indexer1.acquire_lock().unwrap();

        let indexer2 = BackgroundIndexer::new(temp.path()).unwrap();
        let result = indexer2.acquire_lock();

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("already in progress")
        );
    }

    #[test]
    fn test_indexer_status_write() {
        let temp = TempDir::new().unwrap();
        let cache_mgr = CacheManager::new(temp.path());
        cache_mgr.init().unwrap();

        let mut indexer = BackgroundIndexer::new(temp.path()).unwrap();
        indexer.status.total_files = 100;
        indexer.status.processed_files = 50;

        indexer.write_status().unwrap();

        let status = BackgroundIndexer::get_status(cache_mgr.path()).unwrap();
        assert!(status.is_some());

        let status = status.unwrap();
        assert_eq!(status.total_files, 100);
        assert_eq!(status.processed_files, 50);
        assert_eq!(status.state, IndexerState::Running);
    }

    #[test]
    fn test_indexer_status_read_nonexistent() {
        let temp = TempDir::new().unwrap();
        let cache_mgr = CacheManager::new(temp.path());
        cache_mgr.init().unwrap();

        let status = BackgroundIndexer::get_status(cache_mgr.path()).unwrap();
        assert!(status.is_none());
    }

    #[test]
    fn test_indexer_run_empty_index() {
        let temp = TempDir::new().unwrap();
        let cache_mgr = CacheManager::new(temp.path());
        cache_mgr.init().unwrap();

        let mut indexer = BackgroundIndexer::new(temp.path()).unwrap();
        let result = indexer.run();

        assert!(result.is_ok());
        assert_eq!(indexer.status.state, IndexerState::Completed);
        assert_eq!(indexer.status.processed_files, 0);
        assert_eq!(indexer.status.total_files, 0);
    }

    #[test]
    fn test_stale_lock_detection() {
        use filetime::{FileTime, set_file_mtime};

        let temp = TempDir::new().unwrap();
        let cache_mgr = CacheManager::new(temp.path());
        cache_mgr.init().unwrap();

        let lock_path = cache_mgr.path().join(LOCK_FILE);

        // Fresh lock file should not be considered stale
        std::fs::write(&lock_path, "12345").unwrap();
        assert!(!is_lock_stale(&lock_path), "fresh lock should not be stale");

        // Backdate mtime to 2 hours ago (exceeds LOCK_MAX_AGE of 1 hour)
        let two_hours_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 3600);
        set_file_mtime(&lock_path, FileTime::from_system_time(two_hours_ago)).unwrap();

        assert!(is_lock_stale(&lock_path), "2-hour-old lock should be stale");

        // Nonexistent lock file should not be reported as stale
        std::fs::remove_file(&lock_path).unwrap();
        assert!(
            !is_lock_stale(&lock_path),
            "missing lock should not be stale"
        );
    }

    #[test]
    fn test_is_running_cleans_stale_lock() {
        use filetime::{FileTime, set_file_mtime};

        let temp = TempDir::new().unwrap();
        let cache_mgr = CacheManager::new(temp.path());
        cache_mgr.init().unwrap();

        let lock_path = cache_mgr.path().join(LOCK_FILE);

        // Create a stale lock file (backdated 2 hours)
        std::fs::write(&lock_path, "99999999").unwrap();
        let two_hours_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 3600);
        set_file_mtime(&lock_path, FileTime::from_system_time(two_hours_ago)).unwrap();

        assert!(
            lock_path.exists(),
            "lock file should exist before is_running()"
        );

        // is_running() should detect staleness, remove the lock, and return false
        assert!(!BackgroundIndexer::is_running(cache_mgr.path()));
        assert!(
            !lock_path.exists(),
            "stale lock file should be removed by is_running()"
        );
    }

    #[test]
    fn test_acquire_lock_cleans_stale_lock() {
        use filetime::{FileTime, set_file_mtime};

        let temp = TempDir::new().unwrap();
        let cache_mgr = CacheManager::new(temp.path());
        cache_mgr.init().unwrap();

        let lock_path = cache_mgr.path().join(LOCK_FILE);

        // Create a stale lock file
        std::fs::write(&lock_path, "99999999").unwrap();
        let two_hours_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 3600);
        set_file_mtime(&lock_path, FileTime::from_system_time(two_hours_ago)).unwrap();

        // acquire_lock() should succeed by treating the stale lock as removable
        let indexer = BackgroundIndexer::new(temp.path()).unwrap();
        let _lock = indexer
            .acquire_lock()
            .expect("stale lock should not block acquire_lock");

        assert!(lock_path.exists(), "new lock file should be created");
    }
}
