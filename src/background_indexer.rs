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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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

/// How long a pass may go without updating `indexing.status` before it is presumed
/// dead, on platforms where pid liveness cannot be checked.
///
/// A healthy pass writes its status once per 128-file chunk — seconds apart. 60s is
/// generous enough to survive a very slow batch while turning a crash into an
/// immediate recovery rather than a 15-minute wait.
const HEARTBEAT_MAX_AGE: std::time::Duration = std::time::Duration::from_secs(60);

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
    /// Files parsed successfully but NOT persisted, because the batch write failed.
    ///
    /// Distinct from `failed_files`, which counts files that failed to PARSE. Merging
    /// the two made a single SQLite write error read as 27 broken files.
    #[serde(default)]
    pub write_failed_files: usize,
    /// Files skipped because they look minified, so symbol extraction was declined.
    ///
    /// These are still fully text-searchable. Counted so an empty `--symbols` result
    /// for a bundle is visible rather than mysterious.
    #[serde(default)]
    pub skipped_minified: usize,
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
    is_lock_stale_by(lock_path, LOCK_MAX_AGE)
}

/// As [`is_lock_stale`], with an explicit age limit.
fn is_lock_stale_by(lock_path: &Path, max_age: std::time::Duration) -> bool {
    let metadata = match std::fs::metadata(lock_path) {
        Ok(m) => m,
        Err(_) => return false, // Can't read => not stale, let caller handle
    };
    let modified = match metadata.modified() {
        Ok(t) => t,
        Err(_) => return false,
    };
    match modified.elapsed() {
        Ok(age) => age > max_age,
        Err(_) => false, // Clock skew — don't remove
    }
}

/// What happened when a file was offered to the symbol parser.
///
/// A skipped file must be distinguishable from a parsed file that had no symbols —
/// conflating the two is the same class of error that made `failed_files` unreadable.
enum ParseOutcome {
    Parsed(Vec<crate::models::SearchResult>),
    /// Declined: the file looks minified. Still fully text-searchable.
    SkippedMinified,
}

/// Background symbol indexer
pub struct BackgroundIndexer {
    workspace_path: PathBuf,
    cache_path: PathBuf,
    status: IndexingStatus,
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
                write_failed_files: 0,
                skipped_minified: 0,
            },
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
            // Liveness unknown on this platform (Windows). Use the pass's own
            // HEARTBEAT instead of the one-hour-ish age rule.
            //
            // A running pass rewrites `indexing.status` every chunk, so a frozen
            // `updated_at` means it died. Without this, a killed `rfx index` left a
            // lock that Windows could not attribute, the next run treated it as live,
            // and indexing was blocked until LOCK_MAX_AGE — turning a crash into a
            // 15-minute outage. Caught by `killed_indexer_never_leaves_truncated_index`
            // on the Windows runner.
            None => {
                if is_lock_stale(&lock_path) {
                    // Absolute ceiling, independent of any heartbeat: nothing should
                    // hold this lock for a quarter of an hour.
                    log::warn!("Removing indexing lock older than {:?}", LOCK_MAX_AGE);
                    true
                } else if !Self::heartbeat_is_fresh(cache_dir)
                    && is_lock_stale_by(&lock_path, HEARTBEAT_MAX_AGE)
                {
                    // The lock has existed longer than a heartbeat interval and the
                    // status has not moved: the pass died.
                    log::warn!(
                        "Removing indexing lock: no heartbeat within {:?} and liveness \
                         is not determinable on this platform",
                        HEARTBEAT_MAX_AGE
                    );
                    true
                } else {
                    false
                }
            }
        };

        if stale {
            let _ = std::fs::remove_file(&lock_path);
            return None;
        }

        Some(holder)
    }

    /// Whether `indexing.status` was written recently enough to imply a live pass.
    ///
    /// The pass rewrites its status once per chunk (128 files), so on any healthy run
    /// `updated_at` moves every few seconds. A crashed pass leaves it frozen.
    ///
    /// Conservative: an unreadable, unparseable or already-finished status counts as
    /// NO heartbeat, so the caller falls back to the file-age check rather than
    /// honouring a lock nothing is behind.
    pub fn heartbeat_is_fresh(cache_dir: &Path) -> bool {
        let Ok(Some(status)) = Self::get_status(cache_dir) else {
            return false;
        };
        if status.state != IndexerState::Running {
            return false;
        }
        let Ok(updated) = chrono::DateTime::parse_from_rfc3339(&status.updated_at) else {
            return false;
        };
        let age = chrono::Utc::now().signed_duration_since(updated.with_timezone(&chrono::Utc));
        // Negative age means clock skew; treat it as fresh rather than reaping a pass
        // that may well be alive.
        age < chrono::Duration::from_std(HEARTBEAT_MAX_AGE).unwrap_or(chrono::Duration::zero())
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

    /// Internal indexing implementation: a streaming pipeline.
    ///
    /// Workers parse files from `content.bin` and hand encoded blobs to ONE writer
    /// thread that owns the database connection and commits in large batches, so
    /// parsing never waits on SQLite and SQLite never waits on parsing. Until
    /// 1.8.1 the pass ran 128-file `par_iter` chunks separated by a serial write,
    /// opened a connection per file to ask whether it was cached, and recompiled
    /// every tree-sitter query per file: 45 s on a 27k-file tree, 90% of it
    /// avoidable.
    fn run_internal(&mut self) -> Result<()> {
        log::info!("Starting background symbol indexing");

        let cache_mgr = CacheManager::new(&self.workspace_path);
        let config = cache_mgr.load_index_config().unwrap_or_default();
        let num_threads = crate::models::resolve_symbol_thread_count(config.symbol_threads);
        log::info!(
            "Using {} threads for background indexing ({} CPUs available)",
            num_threads,
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        );
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .context("Failed to create thread pool")?;

        // Schema (and the format-version guard) first, then one connection for
        // the whole pass.
        SymbolCache::open(&self.cache_path).context("Failed to open symbol cache")?;
        let mut conn = crate::cache::open_meta_db(self.cache_path.join("meta.db"))
            .context("Failed to open meta.db for the symbol pass")?;
        // WAL + NORMAL is crash-safe for a cache that is rebuilt from content.bin.
        conn.execute_batch("PRAGMA synchronous=NORMAL")
            .context("Failed to set synchronous=NORMAL")?;

        let content_path = self.cache_path.join("content.bin");
        if !content_path.exists() {
            log::info!("No content.bin found - index is empty, nothing to process");
            self.status.total_files = 0;
            self.status.processed_files = 0;
            self.write_status()?;
            return Ok(());
        }
        let content_reader =
            ContentReader::open(&content_path).context("Failed to open content.bin")?;

        // `path → (file_id, hash)` for every indexed file and the set of cached
        // keys: two queries, instead of two queries plus a connection per file.
        let file_rows = cache_mgr
            .load_all_file_rows()
            .context("Failed to load file rows")?;
        let cached_keys =
            SymbolCache::load_cached_keys_on(&conn).context("Failed to load cached symbol keys")?;

        let total_files = content_reader.file_count();
        self.status.total_files = total_files;
        log::info!("Found {} indexed files to process", total_files);

        if file_rows.is_empty() && total_files > 0 {
            log::error!(
                "CRITICAL: No file hashes found in file_branches table, but {} files exist in content.bin!",
                total_files
            );
            log::error!("This likely means:");
            log::error!("  1. The main indexer failed to populate file_branches table");
            log::error!("  2. WAL checkpoint didn't flush data before background indexer started");
            log::error!("  3. Database transaction was rolled back");
            anyhow::bail!(
                "No file hashes available - cannot index symbols. \
                 This is a database synchronization issue. \
                 Try running 'rfx index' again or clearing the cache with 'rfx clear'."
            );
        }

        // Partition the tree once: cached, no parser, no hash, or to parse.
        let mut work: Vec<WorkItem> = Vec::new();
        let mut cached = 0usize;
        let mut no_parser = 0usize;
        let mut no_hash = 0usize;
        for content_id in 0..total_files as u32 {
            let Some(path) = content_reader.get_file_path(content_id) else {
                no_hash += 1;
                continue;
            };
            let path_str = path.to_string_lossy();
            // content.bin may store "./src/main.rs"; the database stores "src/main.rs".
            let path_str = path_str.strip_prefix("./").unwrap_or(&path_str).to_string();
            let Some((db_id, hash)) = file_rows.get(&path_str) else {
                no_hash += 1;
                continue;
            };
            if cached_keys.contains(&(*db_id, hash.clone())) {
                cached += 1;
                continue;
            }
            let language = crate::models::Language::from_path(std::path::Path::new(&path_str));
            if !ParserFactory::has_symbol_parser(language) {
                no_parser += 1;
                continue;
            }
            work.push(WorkItem {
                content_id,
                db_id: *db_id,
                path: path_str,
                hash: hash.clone(),
            });
        }
        drop(cached_keys);
        drop(file_rows);
        log::info!(
            "Symbol pass: {} to parse, {} cached, {} without a symbol parser, {} not in the database",
            work.len(),
            cached,
            no_parser,
            no_hash
        );

        // Files that need no parsing count as processed from the start.
        let base_processed = cached + no_parser + no_hash;
        self.status.cached_files = cached;
        self.status.processed_files = base_processed;
        self.status.phase = "parsing".to_string();
        self.write_status()?;

        let counters = PassCounters::default();
        let cancel = AtomicBool::new(false);
        let (tx, rx) = std::sync::mpsc::sync_channel::<ParsedFile>(256);

        let mut writer_status = self.status.clone();
        let cache_path = self.cache_path.clone();
        let counters_ref = &counters;
        let cancel_ref = &cancel;
        let content_ref = &content_reader;
        let this = &*self;

        let writer_result = std::thread::scope(|scope| {
            let writer = scope.spawn(|| {
                Self::writer_loop(
                    rx,
                    &mut conn,
                    &cache_path,
                    &mut writer_status,
                    base_processed,
                    counters_ref,
                    cancel_ref,
                )
            });

            thread_pool.install(|| {
                work.par_iter().for_each(|item| {
                    if cancel_ref.load(Ordering::Relaxed) {
                        return;
                    }
                    let parse_start = Instant::now();
                    let outcome = this.parse_symbols(content_ref, item.content_id, &item.path);
                    counters_ref
                        .parse_ns
                        .fetch_add(parse_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    match outcome {
                        Ok(ParseOutcome::Parsed(symbols)) => {
                            let encode_start = Instant::now();
                            let encoded = crate::symbol_cache::encode_symbols(&symbols);
                            counters_ref.encode_ns.fetch_add(
                                encode_start.elapsed().as_nanos() as u64,
                                Ordering::Relaxed,
                            );
                            counters_ref
                                .symbols
                                .fetch_add(symbols.len(), Ordering::Relaxed);
                            match encoded {
                                Ok(blob) => {
                                    counters_ref.parsed.fetch_add(1, Ordering::Relaxed);
                                    // The writer only goes away after the pool
                                    // finishes, so a send error means it died.
                                    if tx
                                        .send(ParsedFile {
                                            db_id: item.db_id,
                                            hash: item.hash.clone(),
                                            path: item.path.clone(),
                                            blob,
                                        })
                                        .is_err()
                                    {
                                        cancel_ref.store(true, Ordering::Relaxed);
                                    }
                                }
                                Err(e) => {
                                    log::warn!("Failed to encode symbols for {}: {}", item.path, e);
                                    counters_ref.failed.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                        Ok(ParseOutcome::SkippedMinified) => {
                            counters_ref
                                .skipped_minified
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => {
                            log::warn!("Failed to parse symbols from {}: {}", item.path, e);
                            counters_ref.failed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });
            });
            drop(tx);

            writer
                .join()
                .unwrap_or_else(|_| Err(anyhow::anyhow!("symbol writer thread panicked")))
        });

        // Fold the writer's view (counts, error text) back into ours.
        self.status = writer_status;
        writer_result?;
        log::info!(
            "Symbol pass CPU: parse {} ms, encode {} ms across {} threads; {} symbols, {} blob bytes",
            counters.parse_ns.load(Ordering::Relaxed) / 1_000_000,
            counters.encode_ns.load(Ordering::Relaxed) / 1_000_000,
            num_threads,
            counters.symbols.load(Ordering::Relaxed),
            counters.blob_bytes.load(Ordering::Relaxed)
        );

        let cancelled =
            cancel.load(Ordering::Relaxed) && counters.processed() + base_processed < total_files;
        if cancelled {
            log::info!(
                "Symbol indexing cancelled at {}/{} files (an indexer asked for the database)",
                self.status.processed_files,
                total_files
            );
            self.status.state = IndexerState::Cancelled;
            self.status.phase = "cancelled".to_string();
            self.write_status()?;
            return Ok(());
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

        let removed =
            SymbolCache::cleanup_stale_on(&conn).context("Failed to cleanup stale symbols")?;

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

    /// The single writer: drains parsed files, commits them in large batches,
    /// keeps the status file current and relays a cancel request to the workers.
    #[allow(clippy::too_many_arguments)]
    fn writer_loop(
        rx: std::sync::mpsc::Receiver<ParsedFile>,
        conn: &mut rusqlite::Connection,
        cache_path: &Path,
        status: &mut IndexingStatus,
        base_processed: usize,
        counters: &PassCounters,
        cancel: &AtomicBool,
    ) -> Result<()> {
        use std::sync::mpsc::RecvTimeoutError;

        const BATCH_FILES: usize = 1024;
        const BATCH_BYTES: usize = 16 * 1024 * 1024;
        const POLL: std::time::Duration = std::time::Duration::from_millis(200);
        const STATUS_EVERY: std::time::Duration = std::time::Duration::from_secs(1);

        let mut batch: Vec<ParsedFile> = Vec::new();
        let mut batch_bytes = 0usize;
        let mut last_status = Instant::now();

        let refresh = |status: &mut IndexingStatus, phase: &str| {
            status.parsed_files = counters.parsed.load(Ordering::Relaxed);
            status.failed_files = counters.failed.load(Ordering::Relaxed);
            status.skipped_minified = counters.skipped_minified.load(Ordering::Relaxed);
            status.write_failed_files = counters.write_failed.load(Ordering::Relaxed);
            status.processed_files = base_processed + counters.processed();
            status.phase = phase.to_string();
            status.current_file = None;
        };

        loop {
            let mut done = false;
            match rx.recv_timeout(POLL) {
                Ok(item) => {
                    batch_bytes += item.blob.len();
                    counters
                        .blob_bytes
                        .fetch_add(item.blob.len(), Ordering::Relaxed);
                    batch.push(item);
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => done = true,
            }

            if done || batch.len() >= BATCH_FILES || batch_bytes >= BATCH_BYTES {
                if !batch.is_empty() {
                    refresh(status, "writing");
                    Self::write_status_file(cache_path, status);
                    Self::commit_batch(conn, &mut batch, counters, status);
                    batch_bytes = 0;
                    refresh(status, "parsing");
                    Self::write_status_file(cache_path, status);
                    last_status = Instant::now();
                }
                if done {
                    break;
                }
            }

            // Relay a cancel request (an `rfx index` waiting for the database) to
            // the workers; the batch in hand is still committed on the way out.
            if !cancel.load(Ordering::Relaxed) && Self::cancel_requested(cache_path) {
                log::info!("Cancel requested; finishing in-flight files and stopping");
                cancel.store(true, Ordering::Relaxed);
            }

            if last_status.elapsed() >= STATUS_EVERY {
                refresh(status, "parsing");
                Self::write_status_file(cache_path, status);
                last_status = Instant::now();
            }
        }

        refresh(status, "parsing");
        Ok(())
    }

    /// Commit `batch` in one transaction; on failure wait briefly and retry once,
    /// then count the batch as not persisted (never as a parse failure).
    fn commit_batch(
        conn: &mut rusqlite::Connection,
        batch: &mut Vec<ParsedFile>,
        counters: &PassCounters,
        status: &mut IndexingStatus,
    ) {
        let started = Instant::now();
        let n = batch.len();
        let mut result = Self::try_commit(conn, batch);
        if let Err(e) = &result {
            log::warn!(
                "Symbol batch of {} files failed to commit ({}); retrying once",
                n,
                e
            );
            std::thread::sleep(std::time::Duration::from_millis(250));
            result = Self::try_commit(conn, batch);
        }
        match result {
            Ok(()) => {
                counters.persisted.fetch_add(n, Ordering::Relaxed);
                log::info!(
                    "Symbol batch committed: {} files, {} ms (parsed so far {})",
                    n,
                    started.elapsed().as_millis(),
                    counters.parsed.load(Ordering::Relaxed)
                );
            }
            Err(e) => {
                // A WRITE failure is not a PARSE failure: the files were parsed;
                // they are just not in the cache. Count them apart and name one.
                counters.write_failed.fetch_add(n, Ordering::Relaxed);
                let first = batch.first().map(|f| f.path.as_str()).unwrap_or("unknown");
                status.error = Some(format!(
                    "{} file(s) parsed but not persisted (first: {}): {:#}",
                    n, first, e
                ));
                log::error!(
                    "Failed to write symbol batch of {} file(s) after {} ms, first {}: {:#}",
                    n,
                    started.elapsed().as_millis(),
                    first,
                    e
                );
            }
        }
        batch.clear();
    }

    fn try_commit(conn: &mut rusqlite::Connection, batch: &[ParsedFile]) -> Result<()> {
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .context("begin")?;
        let now = chrono::Utc::now().timestamp().to_string();
        {
            let mut stmt = tx.prepare_cached(SymbolCache::INSERT_SYMBOLS_SQL)?;
            for item in batch {
                stmt.execute(rusqlite::params![item.db_id, item.hash, item.blob, now])?;
            }
        }
        tx.commit().context("commit")?;
        Ok(())
    }

    /// Write `status` to `indexing.status` (best effort; a failure is logged).
    fn write_status_file(cache_path: &Path, status: &mut IndexingStatus) {
        status.updated_at = chrono::Utc::now().to_rfc3339();
        let status_path = cache_path.join(STATUS_FILE);
        match serde_json::to_string_pretty(status) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&status_path, json) {
                    log::warn!("Failed to write status: {}", e);
                }
            }
            Err(e) => log::warn!("Failed to serialize status: {}", e),
        }
    }

    /// Parse symbols from a file using content.bin
    fn parse_symbols(
        &self,
        content_reader: &ContentReader,
        file_id: u32,
        path: &str,
    ) -> Result<ParseOutcome> {
        // Read file contents from content.bin (memory-mapped, zero-copy)
        let source = content_reader
            .get_file_content(file_id)
            .with_context(|| format!("Failed to read file from content.bin: {}", path))?;

        // Detect language from file extension
        let language = crate::models::Language::from_path(std::path::Path::new(path));

        // Ask before parsing, so a declined file can be COUNTED rather than looking
        // like a parser that found nothing. `ParserFactory::parse` checks this too and
        // remains the universal guard for the query and wiki call sites; the scan
        // early-exits on a normal file, so asking twice is close to free.
        if crate::parsers::is_minified(source) {
            return Ok(ParseOutcome::SkippedMinified);
        }

        let symbols = ParserFactory::parse(path, source, language)
            .with_context(|| format!("Failed to parse symbols from: {}", path))?;

        Ok(ParseOutcome::Parsed(symbols))
    }
}

/// A file the pass must parse.
struct WorkItem {
    /// Index into content.bin.
    content_id: u32,
    /// `files.id` in meta.db.
    db_id: i64,
    path: String,
    hash: String,
}

/// A parsed file on its way to the writer.
struct ParsedFile {
    db_id: i64,
    hash: String,
    path: String,
    blob: Vec<u8>,
}

/// Counts shared between workers and the writer.
#[derive(Default)]
struct PassCounters {
    parsed: AtomicUsize,
    failed: AtomicUsize,
    skipped_minified: AtomicUsize,
    persisted: AtomicUsize,
    write_failed: AtomicUsize,
    /// Diagnostics: worker time in parse/extract and in encode, symbols and bytes.
    parse_ns: std::sync::atomic::AtomicU64,
    encode_ns: std::sync::atomic::AtomicU64,
    symbols: AtomicUsize,
    blob_bytes: AtomicUsize,
}

impl PassCounters {
    /// Files the workers have finished with, whatever the outcome.
    fn processed(&self) -> usize {
        self.parsed.load(Ordering::Relaxed)
            + self.failed.load(Ordering::Relaxed)
            + self.skipped_minified.load(Ordering::Relaxed)
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

    fn workspace(files: usize) -> TempDir {
        let temp = TempDir::new().unwrap();
        let src = temp.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        for i in 0..files {
            std::fs::write(
                src.join(format!("m{i}.rs")),
                format!(
                    "pub fn func_{i}(x: u32) -> u32 {{\n    let local_{i} = x + {i};\n    local_{i}\n}}\n\npub struct S{i} {{\n    pub a: u32,\n}}\n"
                ),
            )
            .unwrap();
        }
        std::fs::write(temp.path().join("README.md"), "# no symbols here\n").unwrap();
        std::fs::write(temp.path().join("notes.txt"), "plain text\n").unwrap();
        temp
    }

    fn index_workspace(root: &Path) {
        let cache = CacheManager::new(root);
        crate::indexer::Indexer::new(cache, crate::models::IndexConfig::default())
            .index(root, false)
            .unwrap();
    }

    fn symbol_rows(root: &Path) -> i64 {
        let conn = crate::cache::open_meta_db(root.join(".reflex").join("meta.db")).unwrap();
        conn.query_row("SELECT COUNT(*) FROM symbols", [], |r| r.get(0))
            .unwrap()
    }

    #[test]
    fn full_run_parses_every_code_file_then_reports_them_cached() {
        let temp = workspace(40);
        index_workspace(temp.path());

        let mut first = BackgroundIndexer::new(temp.path()).unwrap();
        first.run().unwrap();
        assert_eq!(first.status.state, IndexerState::Completed);
        assert_eq!(
            first.status.total_files, 42,
            "40 .rs + README.md + notes.txt"
        );
        assert_eq!(first.status.processed_files, 42);
        assert_eq!(first.status.parsed_files, 40);
        assert_eq!(first.status.cached_files, 0);
        assert_eq!(first.status.failed_files, 0);
        assert_eq!(first.status.write_failed_files, 0);
        assert!(first.status.error.is_none(), "{:?}", first.status.error);
        // Text-tier files are skipped, not stored as empty rows.
        assert_eq!(symbol_rows(temp.path()), 40);

        let mut second = BackgroundIndexer::new(temp.path()).unwrap();
        second.run().unwrap();
        assert_eq!(second.status.state, IndexerState::Completed);
        assert_eq!(second.status.cached_files, 40);
        assert_eq!(second.status.parsed_files, 0);
        assert_eq!(second.status.processed_files, 42);

        // The rows decode to the symbols the parser produces.
        let symbol_cache = SymbolCache::open(temp.path().join(".reflex").as_path()).unwrap();
        let conn = crate::cache::open_meta_db(temp.path().join(".reflex").join("meta.db")).unwrap();
        let hash: String = conn
            .query_row(
                "SELECT fb.hash FROM file_branches fb JOIN files f ON f.id = fb.file_id WHERE f.path = 'src/m7.rs'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let cached = symbol_cache.get("src/m7.rs", &hash).unwrap().unwrap();
        let names: Vec<_> = cached.iter().filter_map(|s| s.symbol.as_deref()).collect();
        assert!(names.contains(&"func_7"), "{names:?}");
        assert!(names.contains(&"S7"), "{names:?}");
        assert!(cached.iter().all(|s| s.path == "src/m7.rs"));
    }

    #[test]
    fn cancel_request_stops_the_pass_and_keeps_what_was_parsed() {
        let temp = workspace(1500);
        index_workspace(temp.path());
        let cache_path = temp.path().join(".reflex");

        let canceller = {
            let cache_path = cache_path.clone();
            std::thread::spawn(move || {
                // Wait for the pass to start parsing, then ask it to stop.
                for _ in 0..500 {
                    if let Ok(Some(st)) = BackgroundIndexer::get_status(&cache_path)
                        && st.phase == "parsing"
                    {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                BackgroundIndexer::request_cancel(&cache_path).unwrap();
            })
        };

        let mut indexer = BackgroundIndexer::new(temp.path()).unwrap();
        indexer.run().unwrap();
        canceller.join().unwrap();

        let st = &indexer.status;
        assert!(
            matches!(st.state, IndexerState::Cancelled | IndexerState::Completed),
            "{:?}",
            st.state
        );
        // Everything parsed before the stop was committed, nothing was lost.
        assert_eq!(symbol_rows(temp.path()), st.parsed_files as i64);
        assert_eq!(st.write_failed_files, 0);
        if st.state == IndexerState::Cancelled {
            assert!(st.processed_files < st.total_files, "{st:?}");
            assert_eq!(st.phase, "cancelled");
        }
        assert!(!cache_path.join(LOCK_FILE).exists());

        // A later run finishes the remainder.
        BackgroundIndexer::clear_cancel(&cache_path);
        let mut again = BackgroundIndexer::new(temp.path()).unwrap();
        again.run().unwrap();
        assert_eq!(again.status.state, IndexerState::Completed);
        assert_eq!(symbol_rows(temp.path()), 1500);
        assert_eq!(again.status.cached_files + again.status.parsed_files, 1500);
    }

    /// A batch the writer cannot commit is retried once, then counted apart from
    /// parse failures and named in `error`. Holds the database for the whole run
    /// (two busy timeouts of 5 s), so this test takes ~11 s.
    #[test]
    fn write_failure_is_counted_apart_from_parse_failures() {
        let temp = workspace(5);
        index_workspace(temp.path());
        let db = temp.path().join(".reflex").join("meta.db");

        // Let the pass read its file rows first, then take the write lock.
        let blocker = crate::cache::open_meta_db(&db).unwrap();
        let mut indexer = BackgroundIndexer::new(temp.path()).unwrap();
        blocker.execute_batch("BEGIN EXCLUSIVE").unwrap();
        // With the lock held from the start, the pass cannot even read. Instead,
        // block only the writer: release, start the pass in a thread, re-lock once
        // it is parsing.
        blocker.execute_batch("COMMIT").unwrap();

        let cache_path = temp.path().join(".reflex");
        let blocker_thread = {
            let db = db.clone();
            let cache_path = cache_path.clone();
            std::thread::spawn(move || {
                for _ in 0..500 {
                    if let Ok(Some(st)) = BackgroundIndexer::get_status(&cache_path)
                        && st.phase == "parsing"
                    {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                let conn = crate::cache::open_meta_db(&db).unwrap();
                conn.execute_batch("BEGIN EXCLUSIVE").unwrap();
                // Hold it longer than two busy timeouts plus the retry pause.
                std::thread::sleep(std::time::Duration::from_millis(11_500));
                conn.execute_batch("COMMIT").unwrap();
            })
        };
        let result = indexer.run();
        blocker_thread.join().unwrap();

        assert!(result.is_ok(), "{result:?}");
        let st = &indexer.status;
        assert_eq!(st.state, IndexerState::Completed);
        assert_eq!(st.parsed_files, 5, "parsing succeeded");
        assert_eq!(st.failed_files, 0, "a write failure is not a parse failure");
        assert_eq!(st.write_failed_files, 5, "{st:?}");
        let err = st.error.as_deref().unwrap_or("");
        assert!(err.contains("5 file(s) parsed but not persisted"), "{err}");
        assert!(err.contains("src/m"), "{err}");
    }
}
