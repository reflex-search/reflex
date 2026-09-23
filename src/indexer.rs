//! Indexing engine for parsing source code
//!
//! The indexer scans the project directory, parses source files using Tree-sitter,
//! and builds the symbol/token cache for fast querying.

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::cache::CacheManager;
use crate::content_store::{ContentReader, ContentWriter};
use crate::models::{Dependency, ImportType, IndexConfig, IndexMode, IndexStats, Language};
#[cfg(unix)]
use crate::output;
use crate::parsers::c::CDependencyExtractor;
use crate::parsers::cpp::CppDependencyExtractor;
use crate::parsers::csharp::CSharpDependencyExtractor;
use crate::parsers::go::GoDependencyExtractor;
use crate::parsers::java::JavaDependencyExtractor;
use crate::parsers::kotlin::KotlinDependencyExtractor;
use crate::parsers::php::PhpDependencyExtractor;
use crate::parsers::python::PythonDependencyExtractor;
use crate::parsers::ruby::RubyDependencyExtractor;
use crate::parsers::rust::RustDependencyExtractor;
use crate::parsers::svelte::SvelteDependencyExtractor;
use crate::parsers::typescript::TypeScriptDependencyExtractor;
use crate::parsers::vue::VueDependencyExtractor;
use crate::parsers::zig::ZigDependencyExtractor;
use crate::parsers::{DependencyExtractor, ExportInfo, ImportInfo};
use crate::trigram_build::{TrigramIndexBuilder, TrigramRun};

/// Progress callback type: (current_file_count, total_file_count, status_message)
/// Uses Arc to allow cloning for multi-threaded progress updates
pub type ProgressCallback = Arc<dyn Fn(usize, usize, String) + Send + Sync>;

/// Result of processing a single file (used for parallel processing)
struct FileProcessingResult {
    path_str: String,
    hash: String,
    content: String,
    language: Language,
    line_count: usize,
    /// On-disk size and mtime, taken BEFORE the read so a write that lands
    /// between the two is caught by the next status check, not hidden by it.
    size: u64,
    mtime_ns: i64,
    dependencies: Vec<ImportInfo>,
    exports: Vec<ExportInfo>,
    /// The file's trigram postings, extracted in the pool (no file id yet).
    trigram_run: TrigramRun,
}

/// Find the nearest tsconfig.json for a given source file
///
/// Walks up the directory tree from the source file to find the nearest tsconfig directory.
/// Returns a reference to the PathAliasMap if found.
fn find_nearest_tsconfig<'a>(
    file_path: &str,
    root: &Path,
    tsconfigs: &'a HashMap<PathBuf, crate::parsers::tsconfig::PathAliasMap>,
) -> Option<&'a crate::parsers::tsconfig::PathAliasMap> {
    // Convert file_path to absolute path (relative to root)
    let abs_file_path = if Path::new(file_path).is_absolute() {
        PathBuf::from(file_path)
    } else {
        root.join(file_path)
    };

    // Start from the file's directory and walk up
    let mut current_dir = abs_file_path.parent()?;

    loop {
        // Check if we have a tsconfig for this directory
        if let Some(alias_map) = tsconfigs.get(current_dir) {
            return Some(alias_map);
        }

        // Move up one directory
        current_dir = current_dir.parent()?;

        // Stop if we've reached the root
        if current_dir == root || !current_dir.starts_with(root) {
            break;
        }
    }

    None
}

/// Manages the indexing process
pub struct Indexer {
    cache: CacheManager,
    config: IndexConfig,
    /// `(max_files, max_bytes)` per batch; `None` = defaults / env overrides.
    batch_limits: Option<(usize, u64)>,
}

/// Default cap on files per batch.
pub const BATCH_MAX_FILES: usize = 5000;
/// Default cap on bytes of text per batch. Bounds the per-batch memory of the
/// read pool (file contents) and of the trigram build (postings), whatever the
/// file count: a 9,000-file tree of large files no longer builds one giant
/// in-memory index.
pub const BATCH_MAX_BYTES: u64 = 48 << 20;

/// Cut `sizes` (in file order) into consecutive batches of at most `max_files`
/// files and, past the first file of a batch, at most `max_bytes` bytes.
pub fn plan_batches(
    sizes: &[u64],
    max_files: usize,
    max_bytes: u64,
) -> Vec<std::ops::Range<usize>> {
    let max_files = max_files.max(1);
    let mut batches = Vec::new();
    let mut start = 0usize;
    let mut bytes = 0u64;
    for (i, &size) in sizes.iter().enumerate() {
        let count = i - start;
        if count > 0 && (count >= max_files || bytes.saturating_add(size) > max_bytes) {
            batches.push(start..i);
            start = i;
            bytes = 0;
        }
        bytes = bytes.saturating_add(size);
    }
    if start < sizes.len() {
        batches.push(start..sizes.len());
    }
    batches
}

/// The `[index] include.patterns` / `exclude.patterns` policy, compiled once.
///
/// Built on `ignore::overrides::Override`, so the patterns follow gitignore rules
/// exactly as the walker applies them: a pattern containing `/` is anchored at the
/// workspace root, a bare name matches at any depth, `*` does not cross `/`.
/// Includes are whitelist globs, excludes are `!`-prefixed. With only includes,
/// non-matching *files* are dropped but directories are still walked, so
/// `include = ["src/**/*.rs"]` works without listing `src/`.
///
/// The same policy must answer "would Reflex index this path?" everywhere: the
/// walker, the working-tree freshness check and the watcher. If they disagree, an
/// edit to an excluded file marks the index permanently stale.
#[derive(Clone, Debug)]
pub struct PathPolicy {
    overrides: Option<ignore::overrides::Override>,
    /// `[index] text_tier`.
    text_tier: bool,
    /// `[index] languages`; empty = every supported language.
    languages: Vec<Language>,
    /// `[index] mode`.
    mode: IndexMode,
    /// `[index] hidden`.
    hidden: bool,
}

impl Default for PathPolicy {
    fn default() -> Self {
        Self {
            overrides: None,
            text_tier: true,
            languages: Vec::new(),
            mode: IndexMode::Tracked,
            hidden: false,
        }
    }
}

impl PathPolicy {
    /// Compile the policy from an index config. No patterns → a policy that admits
    /// every path the language rules allow. An invalid pattern is logged and skipped.
    pub fn from_config(root: &Path, config: &IndexConfig) -> Self {
        let mut policy = Self {
            overrides: None,
            text_tier: config.text_tier,
            languages: config.languages.clone(),
            mode: config.mode,
            hidden: config.hidden,
        };
        if config.include_patterns.is_empty() && config.exclude_patterns.is_empty() {
            return policy;
        }
        let mut builder = ignore::overrides::OverrideBuilder::new(root);
        for pat in &config.include_patterns {
            if let Err(e) = builder.add(pat) {
                log::warn!("Invalid [index] include pattern '{}': {}", pat, e);
            }
        }
        for pat in &config.exclude_patterns {
            let negated = format!("!{}", pat.trim_start_matches('!'));
            if let Err(e) = builder.add(&negated) {
                log::warn!("Invalid [index] exclude pattern '{}': {}", pat, e);
            }
        }
        match builder.build() {
            Ok(ov) => policy.overrides = Some(ov),
            Err(e) => log::warn!("Failed to build [index] include/exclude policy: {}", e),
        }
        policy
    }

    /// The language a file at `path` would be indexed as, or `None` when the
    /// policy would not index it (excluded by pattern, tier off, parser not
    /// wanted). Path only: no filesystem access, so it is the same answer for the
    /// walker, the freshness check and the watcher.
    pub fn classify(&self, path: &Path) -> Option<Language> {
        if !self.admits(path, false) {
            return None;
        }
        let lang = Language::from_path(path);
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        match lang {
            // Indexed only in tracked mode, and excluded from searches by default.
            Language::Lock | Language::Generated => {
                (self.mode == IndexMode::Tracked).then_some(lang)
            }
            // The plain-text tier: docs, config, templates and, in tracked mode,
            // every other non-binary file. Deliberately NOT subject to `languages`.
            // That option means "which PARSERS do I care about"; a user with
            // languages = ["rust"] would otherwise lose the text tier silently,
            // which is the very bug this tier exists to fix. `text_tier = false`
            // is the way to turn it off.
            Language::Text => {
                if !self.text_tier {
                    return None;
                }
                match self.mode {
                    IndexMode::Tracked => Some(lang),
                    IndexMode::Allowlist => crate::models::is_text_tier_file(name).then_some(lang),
                }
            }
            Language::Unknown => None,
            // Code without a working grammar (Swift): in tracked mode it is still a
            // text file an agent greps, so it is indexed; symbol queries skip it.
            code if !code.is_supported() => {
                (self.mode == IndexMode::Tracked && self.text_tier).then_some(code)
            }
            code => {
                if !self.languages.is_empty() && !self.languages.contains(&code) {
                    log::debug!(
                        "Skipping {} ({:?} not in configured languages)",
                        path.display(),
                        code
                    );
                    return None;
                }
                Some(code)
            }
        }
    }

    /// Whether a workspace-RELATIVE path is under a directory the walker would
    /// descend into. With `hidden = false` (the default, like ripgrep) no dot
    /// segment is; with `hidden = true` only `.git/` and `.reflex/` are skipped.
    ///
    /// Relative paths only: an absolute path's own ancestors (`/tmp/.cache/…`) are
    /// none of the walker's business.
    pub fn hidden_ok(&self, rel: &Path) -> bool {
        rel.components()
            .filter_map(|c| c.as_os_str().to_str())
            .all(|seg| {
                if seg == "." || seg == ".." {
                    return true;
                }
                if seg == ".git" || seg == crate::cache::CACHE_DIR {
                    return false;
                }
                self.hidden || !is_hidden_segment(seg)
            })
    }

    /// `[index] hidden`.
    pub fn hidden(&self) -> bool {
        self.hidden
    }

    /// Whether the policy admits this path. Directories are always admitted (the
    /// walker descends; files decide), unless an exclude names them.
    pub fn admits(&self, path: &Path, is_dir: bool) -> bool {
        match &self.overrides {
            None => true,
            Some(ov) => !ov.matched(path, is_dir).is_ignore(),
        }
    }

    /// The walker-side view of the policy.
    pub fn overrides(&self) -> Option<&ignore::overrides::Override> {
        self.overrides.as_ref()
    }
}

/// What one directory walk found.
#[derive(Debug, Default)]
struct Discovered {
    files: Vec<PathBuf>,
    /// On-disk size of each entry of `files` (0 when unknown); drives batching.
    sizes: Vec<u64>,
    skipped_too_large: usize,
    skipped_bytes_too_large: u64,
    skipped_binary: usize,
}

/// A path segment the walker treats as hidden: a dot-name other than `.` / `..`.
/// Shared by the walker policy, the freshness check and the zero-result hint.
pub fn is_hidden_segment(seg: &str) -> bool {
    seg.len() > 1 && seg.starts_with('.') && seg != ".."
}

/// ripgrep's binary rule: a NUL byte anywhere in the file.
///
/// Not "in the first 8 KB": a protobuf blob on the Kubernetes checkout
/// (`swagger.pb`, 4109 word matches) carries its first NUL past that point, and
/// ripgrep still skips it. The indexer holds the whole file in memory when it
/// asks, so the full scan is free; `memchr` makes it a few GB/s.
pub fn is_binary(bytes: &[u8]) -> bool {
    memchr::memchr(0, bytes).is_some()
}

/// [`is_binary`] on the file at `path`. A file that cannot be read is not called
/// binary here; the read that follows reports the error. Callers apply this only
/// to non-code files under `max_file_size`.
pub fn looks_binary(path: &Path) -> bool {
    match std::fs::read(path) {
        Ok(bytes) => is_binary(&bytes),
        Err(_) => false,
    }
}

/// Drops the shared query handles for a workspace when an index run ends,
/// on every exit path including errors and panics.
struct InvalidateOnDrop(std::path::PathBuf);

impl Drop for InvalidateOnDrop {
    fn drop(&mut self) {
        crate::query::invalidate_caches(&self.0);
    }
}

impl Indexer {
    /// Create a new indexer with the given cache manager and config
    pub fn new(cache: CacheManager, config: IndexConfig) -> Self {
        Self {
            cache,
            config,
            batch_limits: None,
        }
    }

    /// Override the per-batch limits (files, bytes). For tests that need to
    /// exercise multi-batch builds on small trees; production reads
    /// `REFLEX_INDEX_BATCH_FILES` / `REFLEX_INDEX_BATCH_BYTES` or the defaults.
    #[doc(hidden)]
    pub fn set_batch_limits(&mut self, max_files: usize, max_bytes: u64) {
        self.batch_limits = Some((max_files, max_bytes));
    }

    fn batch_limits(&self) -> (usize, u64) {
        if let Some(limits) = self.batch_limits {
            return limits;
        }
        let files = std::env::var("REFLEX_INDEX_BATCH_FILES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(BATCH_MAX_FILES);
        let bytes = std::env::var("REFLEX_INDEX_BATCH_BYTES")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(BATCH_MAX_BYTES);
        (files, bytes)
    }

    /// The `[index] include/exclude` policy for a workspace root.
    pub fn path_policy(&self, root: &Path) -> PathPolicy {
        PathPolicy::from_config(root, &self.config)
    }

    /// Build or update the index for the given root directory
    pub fn index(&self, root: impl AsRef<Path>, show_progress: bool) -> Result<IndexStats> {
        self.index_with_callback(root, show_progress, None)
    }

    /// How long to wait for a running symbol pass to yield the database.
    ///
    /// One batch is 128 files, so a pass normally yields in well under a second.
    /// 10s covers a slow batch without making the caller wait out a whole pass.
    const SYMBOL_PASS_YIELD_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

    /// Wait for the detached background symbol pass to release `meta.db`.
    ///
    /// The pass (`rfx index-symbols-internal`) runs for minutes on a large repo and
    /// holds a SQLite write lock the whole time, but takes its own `indexing.lock`
    /// rather than the workspace `index.lock`. In 1.7.0 that meant every
    /// `index_project` and `rfx index` during the window failed with a raw
    /// `Failed to begin meta.db schema transaction: database is locked`.
    ///
    /// Asks the pass to stop at its next batch, then waits briefly. If it does not
    /// yield, returns [`ReflexError::SymbolIndexingInProgress`], which names the pid
    /// and the progress — never SQLite's error text.
    fn yield_to_symbol_pass(&self, cache_dir: &Path) -> Result<()> {
        use crate::background_indexer::BackgroundIndexer;

        let Some(holder) = BackgroundIndexer::lock_holder(cache_dir) else {
            // Nothing running (or the lock was stale and has just been reaped).
            BackgroundIndexer::clear_cancel(cache_dir);
            return Ok(());
        };

        log::info!(
            "Symbol indexing is running (pid {}); asking it to yield",
            holder.pid
        );
        let _ = BackgroundIndexer::request_cancel(cache_dir);

        let deadline = std::time::Instant::now() + Self::SYMBOL_PASS_YIELD_TIMEOUT;
        while std::time::Instant::now() < deadline {
            if !BackgroundIndexer::is_running(cache_dir) {
                BackgroundIndexer::clear_cancel(cache_dir);
                log::info!("Symbol indexing yielded; continuing");
                return Ok(());
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        // It did not yield. Report what it is doing, then leave it alone.
        BackgroundIndexer::clear_cancel(cache_dir);
        let status = BackgroundIndexer::get_status(cache_dir).ok().flatten();
        Err(crate::errors::ReflexError::SymbolIndexingInProgress {
            pid: holder.pid,
            started_at: holder.started_clock(),
            processed: status.as_ref().map(|s| s.processed_files).unwrap_or(0),
            total: status.as_ref().map(|s| s.total_files).unwrap_or(0),
        }
        .into())
    }

    /// Build or update the index with progress callback support
    pub fn index_with_callback(
        &self,
        root: impl AsRef<Path>,
        show_progress: bool,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<IndexStats> {
        let root = root.as_ref();
        log::info!("Indexing directory: {:?}", root);
        // Files modified at or after this instant record an unknown mtime, so a
        // write racing the read is caught by hash on the next status check.
        let run_start = std::time::SystemTime::now();

        // Exclusive workspace lock for the whole run. Two indexers streaming
        // into the same content.bin/trigrams.bin is how a reader ends up with a
        // short file. The OS drops the lock if this process dies.
        let cache_dir = self.cache.path().to_path_buf();
        let _index_lock = crate::atomic_write::IndexLock::acquire_with_timeout(
            &cache_dir,
            std::time::Duration::from_secs(self.config.lock_wait_secs),
        )?;
        // A previous indexer that died mid-write leaves `*.tmp` behind.
        crate::atomic_write::remove_stale_tmp(&cache_dir);

        // Any open handle on the stores about to be rewritten is dropped now, and
        // again on every exit path below, so a query in this process never reads a
        // mix of old and new files and never keeps a memo of a stale verdict.
        crate::query::invalidate_caches(root);
        let _invalidate_on_exit = InvalidateOnDrop(root.to_path_buf());

        // The detached symbol pass holds meta.db but NOT this lock, so acquiring
        // `index.lock` above proves nothing about SQLite. Yield to it here, before
        // `cache.init()` runs `BEGIN IMMEDIATE`, so a waiting agent sees progress
        // instead of `database is locked: Error code 5`.
        self.yield_to_symbol_pass(&cache_dir)?;

        // Get git state (if in git repo)
        let git_state = crate::git::get_git_state_optional(root)?;
        let branch = git_state
            .as_ref()
            .map(|s| s.branch.clone())
            .unwrap_or_else(|| "_default".to_string());

        if let Some(ref state) = git_state {
            log::info!(
                "Git state: branch='{}', commit='{}', dirty={}",
                state.branch,
                state.commit,
                state.dirty
            );
        } else {
            log::info!("Not a git repository, using default branch");
        }

        // Configure thread pool for parallel processing.
        // 0 = auto (80% of available cores, up to 32 — the query pool's rule).
        // The pool reads, hashes, extracts imports and trigrams; the only serial
        // work left is streaming file bytes into content.bin. Peak memory is set
        // by the batch byte budget, not by the thread count.
        let num_threads = crate::models::resolve_thread_count(self.config.parallel_threads, 32);

        log::info!(
            "Using {} threads for parallel indexing (out of {} available)",
            num_threads,
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        );

        // Refuse to write a cache another Reflex build owns.
        //
        // A `force` rebuild clears `.reflex/` first, so there is no meta.db left to
        // conflict with and this passes — taking ownership is exactly what force
        // means. Checked before `init()`, which is itself a write.
        self.cache.assert_writable(false)?;

        // Ensure cache is initialized
        self.cache.init()?;

        // Drop rows for files that no longer exist on disk, before anything reads
        // them. `batch_update_files_and_branch` prunes too, but only on a path that
        // actually rebuilds; this covers the skip path, where every surviving hash
        // matches and nothing else would notice the deletion. One `exists()` per row.
        //
        // Note the flag: this prune SHRINKS `existing_hashes` below, which would
        // otherwise make the incremental check see a matching file count and skip
        // the rebuild — leaving the deleted file in content.bin as a ghost hit. A
        // deletion always requires the binary stores to be rewritten.
        let deleted_file_count = match self.cache.identify_deleted_files() {
            Ok(gone) if !gone.is_empty() => {
                log::info!("Removing {} deleted files from meta.db", gone.len());
                if let Err(e) = self.cache.delete_files_from_db(&gone) {
                    log::warn!("Failed to prune deleted files: {}", e);
                }
                gone.len()
            }
            Ok(_) => 0,
            Err(e) => {
                log::warn!("Could not identify deleted files: {}", e);
                0
            }
        };
        let had_deletions = deleted_file_count > 0;

        // Check available disk space after cache is initialized
        self.check_disk_space(root)?;

        // Load existing hashes for incremental indexing (for current branch)
        let existing_hashes = self.cache.load_hashes_for_branch(&branch)?;
        log::debug!(
            "Loaded {} existing file hashes for branch '{}'",
            existing_hashes.len(),
            branch
        );

        // Step 1: Walk directory tree and collect files
        let phase_start = Instant::now();
        let Discovered {
            files,
            sizes,
            skipped_too_large,
            skipped_bytes_too_large,
            skipped_binary,
        } = self.discover_files(root)?;
        let total_files = files.len();
        log::info!(
            "Discovered {} files to index ({} skipped: too large, {} binary) in {} ms",
            total_files,
            skipped_too_large,
            skipped_binary,
            phase_start.elapsed().as_millis()
        );

        // Step 1.4: Parse tsconfig.json files for TypeScript/Vue path alias resolution
        // Must be done before parallel processing so it's available during dependency extraction
        let tsconfigs = crate::parsers::tsconfig::parse_all_tsconfigs(root).unwrap_or_else(|e| {
            log::warn!("Failed to parse tsconfig.json files: {}", e);
            HashMap::new()
        });
        if !tsconfigs.is_empty() {
            log::info!("Found {} tsconfig.json files", tsconfigs.len());
            for (config_dir, alias_map) in &tsconfigs {
                log::debug!(
                    "  {} (base_url: {:?}, {} aliases)",
                    config_dir.display(),
                    alias_map.base_url,
                    alias_map.aliases.len()
                );
            }
        }

        // Step 1.5: Quick incremental check - are all files unchanged?
        // If yes, skip expensive rebuild entirely and return cached stats
        if !existing_hashes.is_empty() && total_files == existing_hashes.len() {
            // Same number of files - check if any changed by comparing hashes.
            // A deletion pruned above already means the binary stores are stale.
            let mut any_changed = had_deletions;
            // Every path we saw on disk this pass. Needed for the deletion check
            // below, which the hash loop alone cannot make.
            let mut current_paths = std::collections::HashSet::<String>::with_capacity(files.len());
            // (path, size, mtime_ns) of every file proven unchanged, so the stored
            // fingerprint can follow a `touch` or a reverted edit without a rebuild.
            let mut stat_rows: Vec<(String, u64, i64)> = Vec::with_capacity(files.len());

            for file_path in &files {
                // Normalize path to be relative to root (handles both ./ prefix and absolute paths).
                // Always use forward slashes so the on-disk index is deterministic across OSes
                // and downstream string lookups (file_pattern filters, dependency resolvers)
                // work regardless of the host separator.
                let path_str = file_path.to_string_lossy().to_string();
                let normalized_path = if let Ok(rel_path) = file_path.strip_prefix(root) {
                    // Convert absolute path to relative
                    rel_path.to_string_lossy().replace('\\', "/")
                } else {
                    // Already relative, just strip ./ prefix
                    path_str.trim_start_matches("./").replace('\\', "/")
                };

                current_paths.insert(normalized_path.clone());

                // Check if file exists in cache
                if let Some(existing_hash) = existing_hashes.get(&normalized_path) {
                    // Stat before the read, for the same reason as the main pass.
                    let stat = std::fs::metadata(file_path)
                        .map(|md| (md.len(), crate::cache::recorded_mtime_ns(&md, run_start)))
                        .unwrap_or((0, 0));
                    // Read and hash file to check if changed. Raw bytes, exactly as
                    // the main pass and the freshness check hash them.
                    match std::fs::read(file_path) {
                        Ok(bytes) => {
                            let current_hash = self.hash_content(&bytes);
                            if &current_hash != existing_hash {
                                any_changed = true;
                                log::debug!("File changed: {}", path_str);
                                break; // Early exit - we know we need to rebuild
                            }
                            stat_rows.push((normalized_path.clone(), stat.0, stat.1));
                        }
                        Err(_) => {
                            any_changed = true;
                            break;
                        }
                    }
                } else {
                    // File not in cache - something changed
                    any_changed = true;
                    break;
                }
            }

            // The loop above catches an ADDED path (not in existing_hashes) but never
            // a cached path with no file on disk. Delete one file and add another and
            // the count is unchanged, every surviving hash matches, and the rebuild is
            // skipped — leaving the deleted file searchable as a ghost hit. Close it
            // with a set difference, which costs nothing extra: the paths are already
            // collected and no file is re-read.
            if !any_changed
                && let Some(gone) = existing_hashes
                    .keys()
                    .find(|indexed| !current_paths.contains(*indexed))
            {
                log::debug!("File deleted since last index: {}", gone);
                any_changed = true;
            }

            if !any_changed {
                let content_path = self.cache.path().join("content.bin");
                let trigrams_path = self.cache.path().join("trigrams.bin");

                // Check if schema hash matches - if not, we need a full rebuild
                // even though file contents haven't changed (binary format may differ)
                let schema_ok = self.cache.check_schema_hash().unwrap_or(false);

                if schema_ok && content_path.exists() && trigrams_path.exists() {
                    // Validate trigrams.bin magic bytes before skipping a rebuild.
                    // A disk-full mid-write leaves a file that still "exists" but is corrupt;
                    // checking only existence would cause us to skip a needed rebuild.
                    // This mirrors the magic-byte check in CacheManager::validate().
                    let trigrams_ok = {
                        use std::io::Read;
                        std::fs::File::open(&trigrams_path)
                            .and_then(|mut f| {
                                let mut h = [0u8; 4];
                                f.read_exact(&mut h).map(|_| h)
                            })
                            .map(|h| &h == b"RFTG")
                            .unwrap_or(false)
                    };
                    if !trigrams_ok {
                        log::warn!(
                            "trigrams.bin corrupted or too small despite hashes matching - forcing rebuild"
                        );
                    } else if let Ok(reader) = ContentReader::open(&content_path) {
                        if reader.file_count() > 0 {
                            log::info!("No files changed - skipping index rebuild");

                            // The CONTENT is current, but the recorded commit may not
                            // be: committing already-indexed files moves HEAD without
                            // changing a single hash. Skipping the metadata update
                            // left `commit_sha` behind forever, so freshness reported
                            // `stale` on a perfectly current index until some
                            // unrelated edit happened to force a rebuild.
                            if let Some(ref state) = git_state
                                && let Err(e) = self.cache.update_branch_metadata(
                                    &branch,
                                    Some(state.commit.as_str()),
                                    existing_hashes.len(),
                                    state.dirty,
                                )
                            {
                                log::warn!("Failed to refresh branch metadata: {}", e);
                            }

                            // Same for the per-file fingerprint: the bytes are
                            // current, but a `touch` or a reverted edit has moved
                            // the mtime, and git's view of which paths are dirty
                            // may have changed too.
                            let dirty_paths = git_state
                                .as_ref()
                                .map(|s| s.dirty_paths.clone())
                                .unwrap_or_default();
                            if let Err(e) =
                                self.cache.refresh_fingerprints(&stat_rows, &dirty_paths)
                            {
                                log::warn!("Failed to refresh file fingerprints: {}", e);
                            }

                            let mut stats = self.cache.stats()?;
                            stats.unchanged_files = total_files;
                            stats.deleted_files = deleted_file_count;
                            stats.skipped_too_large = skipped_too_large;
                            stats.skipped_bytes_too_large = skipped_bytes_too_large;
                            stats.skipped_binary = skipped_binary;
                            return Ok(stats);
                        }
                        log::warn!(
                            "content.bin has no files despite hashes matching - forcing rebuild"
                        );
                    } else {
                        log::warn!("content.bin invalid despite hashes matching - forcing rebuild");
                    }
                } else if !schema_ok {
                    log::info!("Schema hash changed - forcing full rebuild");
                } else {
                    log::warn!("Binary index files missing - forcing rebuild");
                }
            }
        } else if total_files != existing_hashes.len() {
            log::info!(
                "File count changed ({} -> {}) - full reindex required",
                existing_hashes.len(),
                total_files
            );
        }

        // Step 2: Build trigram index + content store
        let mut new_hashes = HashMap::new();
        let mut files_indexed = 0;
        let mut new_file_count = 0usize;
        let mut modified_file_count = 0usize;
        let mut unchanged_file_count = 0usize;
        let mut file_metadata: Vec<crate::cache::FileRow> = Vec::new(); // For batch SQLite update
        let dirty_paths: std::collections::HashSet<String> = git_state
            .as_ref()
            .map(|s| s.dirty_paths.clone())
            .unwrap_or_default();
        let mut all_dependencies: Vec<(String, Vec<ImportInfo>)> = Vec::new(); // For batch dependency insertion
        let mut all_exports: Vec<(String, Vec<ExportInfo>)> = Vec::new(); // For batch export insertion

        // Initialize trigram builder and content store. The builder spills each
        // batch to `<cache>/trigram_temp/` only when there is more than one batch;
        // a single batch stays in memory and the directory is never created.
        let mut trigram_builder = TrigramIndexBuilder::new(self.cache.path().join("trigram_temp"));
        let mut content_writer = ContentWriter::new();

        // Initialize content writer to start streaming writes immediately
        let content_path = self.cache.path().join("content.bin");
        content_writer
            .init(content_path.clone())
            .context("Failed to initialize content writer")?;

        // Create progress bar (only if requested via --progress flag)
        let pb = if show_progress {
            let pb = ProgressBar::new(total_files as u64);
            pb.set_draw_target(ProgressDrawTarget::stderr());
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files ({percent}%) {msg}")
                    .unwrap()
                    .progress_chars("=>-")
            );
            // Force updates every 100ms to ensure progress is visible
            pb.enable_steady_tick(std::time::Duration::from_millis(100));
            pb
        } else {
            ProgressBar::hidden()
        };

        // Atomic counter for thread-safe progress updates
        let progress_counter = Arc::new(AtomicU64::new(0));
        // Shared status message for progress callback
        let progress_status = Arc::new(Mutex::new("Indexing files...".to_string()));

        let batch_phase_start = Instant::now();
        let mut pool_ms = 0u128;
        let mut flush_ms = 0u128;

        // Spawn a background thread to update progress bar and call callback during parallel processing
        let counter_for_thread = Arc::clone(&progress_counter);
        let status_for_thread = Arc::clone(&progress_status);
        let pb_clone = pb.clone();
        let callback_for_thread = progress_callback.clone();
        let total_files_for_thread = total_files;
        let progress_thread = if show_progress || callback_for_thread.is_some() {
            Some(std::thread::spawn(move || {
                loop {
                    let count = counter_for_thread.load(Ordering::Relaxed);
                    pb_clone.set_position(count);

                    // Call progress callback if provided
                    if let Some(ref callback) = callback_for_thread {
                        let status = status_for_thread.lock().unwrap().clone();
                        callback(count as usize, total_files_for_thread, status);
                    }

                    if count >= total_files_for_thread as u64 {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            }))
        } else {
            None
        };

        // Build a custom thread pool with limited threads
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .context("Failed to create thread pool")?;

        // Process files in batches to bound memory: a batch holds at most
        // `max_files` files and about `max_bytes` bytes of text, so the pool's
        // in-flight contents and the trigram build's postings stay bounded on
        // any tree.
        let (max_files, max_bytes) = self.batch_limits();
        let batches = plan_batches(&sizes, max_files, max_bytes);
        let num_batches = batches.len();
        let spill_to_disk = num_batches > 1;
        log::info!(
            "Processing {} files in {} batches (<= {} files, ~{} MB each)",
            total_files,
            num_batches,
            max_files,
            max_bytes >> 20
        );

        for (batch_idx, batch_range) in batches.into_iter().enumerate() {
            let batch_files = &files[batch_range];
            log::info!(
                "Processing batch {}/{} ({} files)",
                batch_idx + 1,
                num_batches,
                batch_files.len()
            );
            let pool_start = Instant::now();

            // Process files in parallel using rayon with custom thread pool.
            // `map_init` gives each worker one reusable trigram sort buffer.
            let counter_clone = Arc::clone(&progress_counter);
            let results: Vec<Option<FileProcessingResult>> = pool.install(|| {
                batch_files
                    .par_iter()
                    .map_init(Vec::<u64>::new, |trigram_scratch, file_path| {
                // Normalize path to be relative to root (handles both ./ prefix and absolute paths).
                // Always emit forward slashes so the persisted path is deterministic across OSes.
                let path_str = file_path.to_string_lossy().to_string();
                let normalized_path = if let Ok(rel_path) = file_path.strip_prefix(root) {
                    // Convert absolute path to relative
                    rel_path.to_string_lossy().replace('\\', "/")
                } else {
                    // Already relative, just strip ./ prefix
                    path_str.trim_start_matches("./").replace('\\', "/")
                };

                // Stat BEFORE the read. If the file changes between the two, the
                // recorded (size, mtime) is older than the bytes, so the next status
                // check re-hashes it rather than trusting a stat that matches.
                let (size, mtime_ns) = std::fs::metadata(file_path)
                    .map(|md| (md.len(), crate::cache::recorded_mtime_ns(&md, run_start)))
                    .unwrap_or((0, 0));

                // Read file content once (used for hashing, trigrams, and parsing).
                // The hash is of the RAW bytes, so the freshness check can hash a
                // file on disk and compare. Invalid UTF-8 (a Latin-1 `.po`, an old
                // doc) is decoded lossily rather than dropped: ripgrep searches
                // those bytes, and an agent expects the same.
                let bytes = match std::fs::read(file_path) {
                    Ok(b) => b,
                    Err(e) => {
                        log::warn!("Failed to read {}: {}", path_str, e);
                        // Update progress
                        counter_clone.fetch_add(1, Ordering::Relaxed);
                        return None;
                    }
                };
                let hash = self.hash_content(&bytes);
                let content = match String::from_utf8(bytes) {
                    Ok(s) => s,
                    Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
                };

                // Detect language
                let language = Language::from_path(file_path);

                // Count lines in the file
                let line_count = content.lines().count();

                // Trigram postings, sorted, without a file id (assigned serially below).
                let trigram_run = crate::trigram_build::extract_trigram_run(&content, trigram_scratch);

                // Extract dependencies and exports for supported languages
                let mut parsed_exports: Vec<ExportInfo> = Vec::new();
                let dependencies = match language {
                    Language::Rust => {
                        match RustDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::Python => {
                        match PythonDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::TypeScript | Language::JavaScript => {
                        // Find nearest tsconfig for path alias resolution. One parse
                        // yields both the imports and the re-exports.
                        let alias_map = find_nearest_tsconfig(&path_str, root, &tsconfigs);
                        match TypeScriptDependencyExtractor::extract_dependencies_and_exports(&content, alias_map) {
                            Ok((deps, exports)) => {
                                parsed_exports = exports;
                                deps
                            }
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::Go => {
                        match GoDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::Java => {
                        match JavaDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::C => {
                        match CDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::Cpp => {
                        match CppDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::CSharp => {
                        match CSharpDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::PHP => {
                        match PhpDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::Ruby => {
                        match RubyDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::Kotlin => {
                        match KotlinDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::Zig => {
                        match ZigDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::Vue => {
                        // Find nearest tsconfig for path alias resolution. One parse
                        // per script block yields both the imports and the re-exports.
                        let alias_map = find_nearest_tsconfig(&path_str, root, &tsconfigs);
                        match VueDependencyExtractor::extract_dependencies_and_exports(&content, alias_map) {
                            Ok((deps, exports)) => {
                                parsed_exports = exports;
                                deps
                            }
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::Svelte => {
                        match SvelteDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    // Other languages not yet implemented
                    _ => Vec::new(),
                };

                // Exports (barrel re-export tracking) came out of the same parse
                // as the dependencies above; only TypeScript/JavaScript/Vue have them.
                let exports = parsed_exports;

                // Update progress atomically
                counter_clone.fetch_add(1, Ordering::Relaxed);

                Some(FileProcessingResult {
                    path_str: normalized_path.to_string(),
                    hash,
                    content,
                    language,
                    line_count,
                    size,
                    mtime_ns,
                    dependencies,
                    exports,
                    trigram_run,
                })
                })
                .collect()
            });
            pool_ms += pool_start.elapsed().as_millis();

            // Process batch results immediately (streaming approach to minimize memory)
            for result in results.into_iter().flatten() {
                // Use the normalized (forward-slash, relative) path everywhere so
                // the trigram index and content store agree with what the database
                // and downstream filters expect, regardless of host separator.
                let normalized_pathbuf = PathBuf::from(&result.path_str);

                // Register the file with the trigram builder (assigns file_id in
                // discovery order) and hand it the postings extracted in the pool.
                let _file_id =
                    trigram_builder.add_file(normalized_pathbuf.clone(), result.trigram_run);

                // Add to content store
                content_writer.add_file(normalized_pathbuf, &result.content);

                files_indexed += 1;

                // Track new / modified / unchanged for the incremental summary
                match existing_hashes.get(&result.path_str) {
                    None => new_file_count += 1,
                    Some(old_hash) if old_hash != &result.hash => modified_file_count += 1,
                    _ => unchanged_file_count += 1,
                }

                // Prepare file metadata for batch database update
                file_metadata.push(crate::cache::FileRow {
                    path: result.path_str.clone(),
                    hash: result.hash.clone(),
                    language: format!("{:?}", result.language),
                    line_count: result.line_count,
                    size: result.size,
                    mtime_ns: result.mtime_ns,
                    dirty: dirty_paths.contains(&result.path_str),
                });

                // Collect dependencies for batch insertion (if any)
                if !result.dependencies.is_empty() {
                    all_dependencies.push((result.path_str.clone(), result.dependencies));
                }

                // Collect exports for batch insertion (if any)
                if !result.exports.is_empty() {
                    all_exports.push((result.path_str.clone(), result.exports));
                }

                new_hashes.insert(result.path_str, result.hash);
            }

            // Build this batch's posting lists (sharded, parallel) into a partial.
            let flush_msg = format!(
                "Building trigram batch {}/{}...",
                batch_idx + 1,
                num_batches
            );
            if show_progress {
                pb.set_message(flush_msg.clone());
            }
            *progress_status.lock().unwrap() = flush_msg;
            let flush_start = Instant::now();
            trigram_builder
                .flush_batch(&pool, spill_to_disk)
                .context("Failed to build trigram batch")?;
            flush_ms += flush_start.elapsed().as_millis();
        }
        log::info!(
            "phase read+extract: {} ms in pool, {} ms building trigram batches, {} ms total",
            pool_ms,
            flush_ms,
            batch_phase_start.elapsed().as_millis()
        );

        // Wait for progress thread to finish
        if let Some(thread) = progress_thread {
            let _ = thread.join();
        }

        // Update progress bar to final count
        if show_progress {
            let final_count = progress_counter.load(Ordering::Relaxed);
            pb.set_position(final_count);
        }

        // Update progress bar message for post-processing
        *progress_status.lock().unwrap() = "Writing file metadata to database...".to_string();
        if show_progress {
            pb.set_message("Writing file metadata to database...".to_string());
        }

        let files_tx_start = Instant::now();
        // Batch write file metadata AND branch hashes in a SINGLE atomic transaction
        // This ensures that if files are inserted, their hashes are guaranteed to be inserted too
        if !file_metadata.is_empty() {
            // Record files for this branch (for branch-aware indexing)
            *progress_status.lock().unwrap() = "Recording branch files...".to_string();
            if show_progress {
                pb.set_message("Recording branch files...".to_string());
            }

            // Use atomic method that combines both operations
            self.cache
                .batch_update_files_and_branch(
                    &file_metadata,
                    &branch,
                    git_state.as_ref().map(|s| s.commit.as_str()),
                )
                .context("Failed to batch update files and branch hashes")?;

            log::info!(
                "Wrote metadata and hashes for {} files to database",
                file_metadata.len()
            );
        }

        // Update branch metadata
        self.cache.update_branch_metadata(
            &branch,
            git_state.as_ref().map(|s| s.commit.as_str()),
            file_metadata.len(),
            git_state.as_ref().map(|s| s.dirty).unwrap_or(false),
        )?;

        // Force WAL checkpoint to ensure background processes see all committed data
        // This is critical when spawning background symbol indexer immediately after
        self.cache
            .checkpoint_wal()
            .context("Failed to checkpoint WAL")?;
        log::debug!("WAL checkpoint completed - database is fully synced");

        log::info!(
            "phase files+branch transaction: {} ms",
            files_tx_start.elapsed().as_millis()
        );
        let deps_start = Instant::now();

        // Steps 2.5 and 2.6 share one connection, one in-memory path resolver and
        // one transaction. The resolver is built AFTER the files transaction above
        // committed, because `INSERT OR REPLACE` hands every re-indexed file a new
        // id and imports may target files that did not change.
        let mut dep_conn = crate::cache::open_meta_db(self.cache.path().join("meta.db"))
            .context("Failed to open meta.db for dependency recording")?;
        let resolver = crate::dependency::PathResolver::from_conn(&dep_conn)
            .context("Failed to load file paths for dependency resolution")?;
        let mut dep_writer = crate::dependency::DependencyWriter::begin(&mut dep_conn)?;

        // Step 2.5: Insert dependencies (after files are inserted and have IDs)
        if !all_dependencies.is_empty() {
            *progress_status.lock().unwrap() = "Extracting dependencies...".to_string();
            if show_progress {
                pb.set_message("Extracting dependencies...".to_string());
            }

            // Find and parse all go.mod files for Go projects (monorepo support)
            let go_modules = crate::parsers::go::parse_all_go_modules(root).unwrap_or_else(|e| {
                log::warn!("Failed to parse go.mod files: {}", e);
                Vec::new()
            });
            if !go_modules.is_empty() {
                log::info!("Found {} Go modules", go_modules.len());
                for module in &go_modules {
                    log::debug!("  {} (project: {})", module.name, module.project_root);
                }
            }

            // Find and parse all pom.xml/build.gradle files for Java projects (monorepo support)
            let java_projects =
                crate::parsers::java::parse_all_java_projects(root).unwrap_or_else(|e| {
                    log::warn!("Failed to parse Java project configs: {}", e);
                    Vec::new()
                });
            if !java_projects.is_empty() {
                log::info!("Found {} Java projects", java_projects.len());
                for project in &java_projects {
                    log::debug!(
                        "  {} (project: {})",
                        project.package_name,
                        project.project_root
                    );
                }
            }

            // Find and parse all Python package configs for Python projects (monorepo support)
            let python_packages = crate::parsers::python::parse_all_python_packages(root)
                .unwrap_or_else(|e| {
                    log::warn!("Failed to parse Python package configs: {}", e);
                    Vec::new()
                });
            if !python_packages.is_empty() {
                log::info!("Found {} Python packages", python_packages.len());
                for package in &python_packages {
                    log::debug!("  {} (project: {})", package.name, package.project_root);
                }
            }

            // Find and parse *.gemspec files for Ruby projects (monorepo support)
            let ruby_projects =
                crate::parsers::ruby::parse_all_ruby_projects(root).unwrap_or_else(|e| {
                    log::warn!("Failed to parse Ruby project configs: {}", e);
                    Vec::new()
                });
            if !ruby_projects.is_empty() {
                log::info!("Found {} Ruby projects", ruby_projects.len());
                for project in &ruby_projects {
                    log::debug!("  {} (project: {})", project.gem_name, project.project_root);
                }
            }

            // Find and parse all Cargo.toml files for Rust workspace support
            let rust_crates =
                crate::parsers::rust::parse_all_rust_crates(root).unwrap_or_else(|e| {
                    log::warn!("Failed to parse Cargo.toml files: {}", e);
                    Vec::new()
                });
            if !rust_crates.is_empty() {
                log::info!("Found {} Rust workspace crates", rust_crates.len());
                for krate in &rust_crates {
                    log::debug!("  {} (root: {})", krate.name, krate.root_path.display());
                }
            }

            // Note: Kotlin projects use the same java_projects above (same build systems: Maven/Gradle)

            // Find and parse all composer.json files for PHP projects (monorepo support)
            let php_psr4_mappings = crate::parsers::php::parse_all_composer_psr4(root)
                .unwrap_or_else(|e| {
                    log::warn!("Failed to parse composer.json files: {}", e);
                    Vec::new()
                });
            if !php_psr4_mappings.is_empty() {
                log::info!(
                    "Found {} PSR-4 mappings from composer.json files",
                    php_psr4_mappings.len()
                );
                for mapping in &php_psr4_mappings {
                    log::debug!(
                        "  {} => {} (project: {})",
                        mapping.namespace_prefix,
                        mapping.directory,
                        mapping.project_root
                    );
                }
            }

            // `tsconfigs` (parsed once in Step 1.4) is reused here for alias resolution.
            let mut total_deps_inserted = 0;

            // Process each file's dependencies
            for (file_path, import_infos) in all_dependencies {
                // Get file ID from database
                let file_id = match resolver.get_file_id_by_path(&file_path)? {
                    Some(id) => id,
                    None => {
                        log::warn!(
                            "File not found in database (skipping dependencies): {}",
                            file_path
                        );
                        continue;
                    }
                };

                // Reclassify and filter dependencies
                let mut resolved_deps = Vec::new();

                for mut import_info in import_infos {
                    // Reclassify Go imports using module names (if Go project)
                    if file_path.ends_with(".go") {
                        // Check if the import matches any Go module
                        let mut reclassified = false;
                        for module in &go_modules {
                            import_info.import_type = crate::parsers::go::reclassify_go_import(
                                &import_info.imported_path,
                                Some(&module.name),
                            );
                            // If it's internal, we've found the right module
                            if matches!(import_info.import_type, ImportType::Internal) {
                                reclassified = true;
                                break;
                            }
                        }
                        // If no module matched, use base classification
                        if !reclassified {
                            import_info.import_type = crate::parsers::go::reclassify_go_import(
                                &import_info.imported_path,
                                None,
                            );
                        }
                    }

                    // Reclassify Java imports using package names (if Java project)
                    if file_path.ends_with(".java") {
                        // Check if the import matches any Java project
                        let mut reclassified = false;
                        for project in &java_projects {
                            import_info.import_type = crate::parsers::java::reclassify_java_import(
                                &import_info.imported_path,
                                Some(&project.package_name),
                            );
                            // If it's internal, we've found the right project
                            if matches!(import_info.import_type, ImportType::Internal) {
                                reclassified = true;
                                break;
                            }
                        }
                        // If no project matched, use base classification
                        if !reclassified {
                            import_info.import_type = crate::parsers::java::reclassify_java_import(
                                &import_info.imported_path,
                                None,
                            );
                        }
                    }

                    // Reclassify Python imports using package names (if Python project)
                    if file_path.ends_with(".py") {
                        // Check if the import matches any Python package
                        let mut reclassified = false;
                        for package in &python_packages {
                            import_info.import_type =
                                crate::parsers::python::reclassify_python_import(
                                    &import_info.imported_path,
                                    Some(&package.name),
                                );
                            // If it's internal, we've found the right package
                            if matches!(import_info.import_type, ImportType::Internal) {
                                reclassified = true;
                                break;
                            }
                        }
                        // If no package matched, use base classification
                        if !reclassified {
                            import_info.import_type =
                                crate::parsers::python::reclassify_python_import(
                                    &import_info.imported_path,
                                    None,
                                );
                        }
                    }

                    // Reclassify Ruby imports using gem names (if Ruby project)
                    if file_path.ends_with(".rb")
                        || file_path.ends_with(".rake")
                        || file_path.ends_with(".gemspec")
                    {
                        // Check if the import matches any Ruby project
                        let mut reclassified = false;
                        for project in &ruby_projects {
                            let gem_names = vec![project.gem_name.clone()];
                            import_info.import_type = crate::parsers::ruby::reclassify_ruby_import(
                                &import_info.imported_path,
                                &gem_names,
                            );
                            // If it's internal, we've found the right project
                            if matches!(import_info.import_type, ImportType::Internal) {
                                reclassified = true;
                                break;
                            }
                        }
                        // If no project matched, use base classification (will be External or Stdlib)
                        if !reclassified {
                            import_info.import_type = crate::parsers::ruby::reclassify_ruby_import(
                                &import_info.imported_path,
                                &[],
                            );
                        }
                    }

                    // Reclassify Kotlin imports using package names (if Kotlin project)
                    if file_path.ends_with(".kt") || file_path.ends_with(".kts") {
                        // Check if the import matches any Java/Kotlin project (same build systems)
                        let mut reclassified = false;
                        for project in &java_projects {
                            import_info.import_type =
                                crate::parsers::kotlin::reclassify_kotlin_import(
                                    &import_info.imported_path,
                                    Some(&project.package_name),
                                );
                            // If it's internal, we've found the right project
                            if matches!(import_info.import_type, ImportType::Internal) {
                                reclassified = true;
                                break;
                            }
                        }
                        // If no project matched, use base classification
                        if !reclassified {
                            import_info.import_type =
                                crate::parsers::kotlin::reclassify_kotlin_import(
                                    &import_info.imported_path,
                                    None,
                                );
                        }
                    }

                    // Reclassify Rust imports using workspace crates
                    if file_path.ends_with(".rs") && !rust_crates.is_empty() {
                        let new_type = crate::parsers::rust::reclassify_rust_import(
                            &import_info.imported_path,
                            &rust_crates,
                        );
                        if matches!(new_type, ImportType::Internal) {
                            import_info.import_type = new_type;
                        }
                    }

                    // External and Stdlib imports: store with resolved_file_id = None.
                    // Graph-analysis queries all filter WHERE resolved_file_id IS NOT NULL,
                    // so storing these here only affects the `rfx deps` display path (REF-78).
                    if matches!(
                        import_info.import_type,
                        ImportType::External | ImportType::Stdlib
                    ) {
                        resolved_deps.push(Dependency {
                            file_id,
                            imported_path: import_info.imported_path.clone(),
                            resolved_file_id: None,
                            import_type: import_info.import_type.clone(),
                            line_number: import_info.line_number,
                            imported_symbols: import_info.imported_symbols.clone(),
                        });
                        continue;
                    }

                    // Resolve PHP dependencies using PSR-4 (deterministic)
                    let resolved_file_id = if file_path.ends_with(".php")
                        && !php_psr4_mappings.is_empty()
                    {
                        // Use PSR-4 to resolve namespace to file path
                        if let Some(resolved_path) =
                            crate::parsers::php::resolve_php_namespace_to_path(
                                &import_info.imported_path,
                                &php_psr4_mappings,
                            )
                        {
                            // Look up file ID in database using exact match
                            match resolver.get_file_id_by_path(&resolved_path) {
                                Ok(Some(id)) => {
                                    log::trace!(
                                        "Resolved PHP dependency: {} -> {} (file_id={})",
                                        import_info.imported_path,
                                        resolved_path,
                                        id
                                    );
                                    Some(id)
                                }
                                Ok(None) => {
                                    log::trace!(
                                        "PHP dependency resolved to path but file not in index: {} -> {}",
                                        import_info.imported_path,
                                        resolved_path
                                    );
                                    None
                                }
                                Err(e) => {
                                    log::debug!(
                                        "Skipping PHP dependency resolution for '{}': {}",
                                        resolved_path,
                                        e
                                    );
                                    None
                                }
                            }
                        } else {
                            log::trace!(
                                "Could not resolve PHP namespace using PSR-4: {}",
                                import_info.imported_path
                            );
                            None
                        }
                    } else if file_path.ends_with(".py") && !python_packages.is_empty() {
                        // Resolve Python dependencies using package mappings
                        if let Some(resolved_path) =
                            crate::parsers::python::resolve_python_import_to_path(
                                &import_info.imported_path,
                                &python_packages,
                                Some(&file_path),
                            )
                        {
                            // Look up file ID in database using exact match
                            match resolver.get_file_id_by_path(&resolved_path) {
                                Ok(Some(id)) => {
                                    log::trace!(
                                        "Resolved Python dependency: {} -> {} (file_id={})",
                                        import_info.imported_path,
                                        resolved_path,
                                        id
                                    );
                                    Some(id)
                                }
                                Ok(None) => {
                                    log::trace!(
                                        "Python dependency resolved to path but file not in index: {} -> {}",
                                        import_info.imported_path,
                                        resolved_path
                                    );
                                    None
                                }
                                Err(e) => {
                                    log::debug!(
                                        "Skipping Python dependency resolution for '{}': {}",
                                        resolved_path,
                                        e
                                    );
                                    None
                                }
                            }
                        } else {
                            log::trace!(
                                "Could not resolve Python import: {}",
                                import_info.imported_path
                            );
                            None
                        }
                    } else if file_path.ends_with(".go") && !go_modules.is_empty() {
                        // Resolve Go dependencies using module mappings
                        if let Some(resolved_path) = crate::parsers::go::resolve_go_import_to_path(
                            &import_info.imported_path,
                            &go_modules,
                            Some(&file_path),
                        ) {
                            // Look up file ID in database using exact match
                            match resolver.get_file_id_by_path(&resolved_path) {
                                Ok(Some(id)) => {
                                    log::trace!(
                                        "Resolved Go dependency: {} -> {} (file_id={})",
                                        import_info.imported_path,
                                        resolved_path,
                                        id
                                    );
                                    Some(id)
                                }
                                Ok(None) => {
                                    log::trace!(
                                        "Go dependency resolved to path but file not in index: {} -> {}",
                                        import_info.imported_path,
                                        resolved_path
                                    );
                                    None
                                }
                                Err(e) => {
                                    log::debug!(
                                        "Skipping Go dependency resolution for '{}': {}",
                                        resolved_path,
                                        e
                                    );
                                    None
                                }
                            }
                        } else {
                            log::trace!(
                                "Could not resolve Go import: {}",
                                import_info.imported_path
                            );
                            None
                        }
                    } else if file_path.ends_with(".ts")
                        || file_path.ends_with(".tsx")
                        || file_path.ends_with(".js")
                        || file_path.ends_with(".jsx")
                        || file_path.ends_with(".mts")
                        || file_path.ends_with(".cts")
                        || file_path.ends_with(".mjs")
                        || file_path.ends_with(".cjs")
                    {
                        // Resolve TypeScript/JavaScript dependencies (relative imports and path aliases)
                        let alias_map = find_nearest_tsconfig(&file_path, root, &tsconfigs);
                        if let Some(candidates_str) =
                            crate::parsers::typescript::resolve_ts_import_to_path(
                                &import_info.imported_path,
                                Some(&file_path),
                                alias_map,
                            )
                        {
                            // Parse pipe-delimited candidates (e.g., "path.tsx|path.ts|path.jsx|path.js")
                            let candidates: Vec<&str> = candidates_str.split('|').collect();

                            // Try each candidate in order until we find one in the database
                            let mut resolved_id = None;
                            for candidate_path in candidates {
                                // Normalize path to be relative to project root
                                // Convert absolute paths to relative (without requiring file to exist)
                                let normalized_candidate = if let Ok(rel_path) =
                                    std::path::Path::new(candidate_path).strip_prefix(root)
                                {
                                    rel_path.to_string_lossy().replace('\\', "/")
                                } else {
                                    // Not an absolute path or not under root - use as-is
                                    // (still normalize separators so DB lookups match).
                                    candidate_path.replace('\\', "/")
                                };

                                log::debug!(
                                    "Looking up TS/JS candidate: '{}' (from '{}')",
                                    normalized_candidate,
                                    candidate_path
                                );
                                match resolver.get_file_id_by_path(&normalized_candidate) {
                                    Ok(Some(id)) => {
                                        log::debug!(
                                            "Resolved TS/JS dependency: {} -> {} (file_id={})",
                                            import_info.imported_path,
                                            normalized_candidate,
                                            id
                                        );
                                        resolved_id = Some(id);
                                        break; // Found a match, stop trying
                                    }
                                    Ok(None) => {
                                        log::trace!(
                                            "TS/JS candidate not in index: {}",
                                            candidate_path
                                        );
                                    }
                                    Err(e) => {
                                        log::debug!(
                                            "Skipping TS/JS dependency resolution for '{}': {}",
                                            normalized_candidate,
                                            e
                                        );
                                    }
                                }
                            }

                            if resolved_id.is_none() {
                                log::trace!(
                                    "TS/JS dependency: no matching file found in database for any candidate: {}",
                                    candidates_str
                                );
                            }

                            resolved_id
                        } else {
                            log::trace!(
                                "Could not resolve TS/JS import (non-relative or external): {}",
                                import_info.imported_path
                            );
                            None
                        }
                    } else if file_path.ends_with(".rs") {
                        // Resolve Rust dependencies (crate::, super::, self::, mod declarations)
                        // Falls back to workspace resolution for cross-crate imports
                        let resolved_path_opt = crate::parsers::rust::resolve_rust_use_to_path(
                            &import_info.imported_path,
                            Some(&file_path),
                            Some(root.to_str().unwrap_or("")),
                        )
                        .or_else(|| {
                            crate::parsers::rust::resolve_rust_workspace_path(
                                &import_info.imported_path,
                                &rust_crates,
                            )
                        });

                        if let Some(resolved_path) = resolved_path_opt {
                            // Look up file ID in database using exact match
                            match resolver.get_file_id_by_path(&resolved_path) {
                                Ok(Some(id)) => {
                                    log::trace!(
                                        "Resolved Rust dependency: {} -> {} (file_id={})",
                                        import_info.imported_path,
                                        resolved_path,
                                        id
                                    );
                                    Some(id)
                                }
                                Ok(None) => {
                                    log::trace!(
                                        "Rust dependency resolved to path but file not in index: {} -> {}",
                                        import_info.imported_path,
                                        resolved_path
                                    );
                                    None
                                }
                                Err(e) => {
                                    log::debug!(
                                        "Skipping Rust dependency resolution for '{}': {}",
                                        resolved_path,
                                        e
                                    );
                                    None
                                }
                            }
                        } else {
                            log::trace!(
                                "Could not resolve Rust import (external or stdlib): {}",
                                import_info.imported_path
                            );
                            None
                        }
                    } else if file_path.ends_with(".java") && !java_projects.is_empty() {
                        // Resolve Java dependencies using project mappings
                        if let Some(resolved_path) =
                            crate::parsers::java::resolve_java_import_to_path(
                                &import_info.imported_path,
                                &java_projects,
                                Some(&file_path),
                            )
                        {
                            // Look up file ID in database using exact match
                            match resolver.get_file_id_by_path(&resolved_path) {
                                Ok(Some(id)) => {
                                    log::trace!(
                                        "Resolved Java dependency: {} -> {} (file_id={})",
                                        import_info.imported_path,
                                        resolved_path,
                                        id
                                    );
                                    Some(id)
                                }
                                Ok(None) => {
                                    log::trace!(
                                        "Java dependency resolved to path but file not in index: {} -> {}",
                                        import_info.imported_path,
                                        resolved_path
                                    );
                                    None
                                }
                                Err(e) => {
                                    log::debug!(
                                        "Skipping Java dependency resolution for '{}': {}",
                                        resolved_path,
                                        e
                                    );
                                    None
                                }
                            }
                        } else {
                            log::trace!(
                                "Could not resolve Java import: {}",
                                import_info.imported_path
                            );
                            None
                        }
                    } else if (file_path.ends_with(".kt") || file_path.ends_with(".kts"))
                        && !java_projects.is_empty()
                    {
                        // Resolve Kotlin dependencies using project mappings (same build systems as Java)
                        if let Some(resolved_path) =
                            crate::parsers::java::resolve_kotlin_import_to_path(
                                &import_info.imported_path,
                                &java_projects,
                                Some(&file_path),
                            )
                        {
                            // Look up file ID in database using exact match
                            match resolver.get_file_id_by_path(&resolved_path) {
                                Ok(Some(id)) => {
                                    log::trace!(
                                        "Resolved Kotlin dependency: {} -> {} (file_id={})",
                                        import_info.imported_path,
                                        resolved_path,
                                        id
                                    );
                                    Some(id)
                                }
                                Ok(None) => {
                                    log::trace!(
                                        "Kotlin dependency resolved to path but file not in index: {} -> {}",
                                        import_info.imported_path,
                                        resolved_path
                                    );
                                    None
                                }
                                Err(e) => {
                                    log::debug!(
                                        "Skipping Kotlin dependency resolution for '{}': {}",
                                        resolved_path,
                                        e
                                    );
                                    None
                                }
                            }
                        } else {
                            log::trace!(
                                "Could not resolve Kotlin import: {}",
                                import_info.imported_path
                            );
                            None
                        }
                    } else if (file_path.ends_with(".rb")
                        || file_path.ends_with(".rake")
                        || file_path.ends_with(".gemspec"))
                        && !ruby_projects.is_empty()
                    {
                        // Resolve Ruby dependencies using project mappings
                        if let Some(resolved_path) =
                            crate::parsers::ruby::resolve_ruby_require_to_path(
                                &import_info.imported_path,
                                &ruby_projects,
                                Some(&file_path),
                            )
                        {
                            // Look up file ID in database using exact match
                            match resolver.get_file_id_by_path(&resolved_path) {
                                Ok(Some(id)) => {
                                    log::trace!(
                                        "Resolved Ruby dependency: {} -> {} (file_id={})",
                                        import_info.imported_path,
                                        resolved_path,
                                        id
                                    );
                                    Some(id)
                                }
                                Ok(None) => {
                                    log::trace!(
                                        "Ruby dependency resolved to path but file not in index: {} -> {}",
                                        import_info.imported_path,
                                        resolved_path
                                    );
                                    None
                                }
                                Err(e) => {
                                    log::debug!(
                                        "Skipping Ruby dependency resolution for '{}': {}",
                                        resolved_path,
                                        e
                                    );
                                    None
                                }
                            }
                        } else {
                            log::trace!(
                                "Could not resolve Ruby require: {}",
                                import_info.imported_path
                            );
                            None
                        }
                    } else if file_path.ends_with(".c") || file_path.ends_with(".h") {
                        // Resolve C dependencies (relative #include paths)
                        if let Some(resolved_path) = crate::parsers::c::resolve_c_include_to_path(
                            &import_info.imported_path,
                            Some(&file_path),
                        ) {
                            // Look up file ID in database using exact match
                            match resolver.get_file_id_by_path(&resolved_path) {
                                Ok(Some(id)) => {
                                    log::trace!(
                                        "Resolved C dependency: {} -> {} (file_id={})",
                                        import_info.imported_path,
                                        resolved_path,
                                        id
                                    );
                                    Some(id)
                                }
                                Ok(None) => {
                                    log::trace!(
                                        "C dependency resolved to path but file not in index: {} -> {}",
                                        import_info.imported_path,
                                        resolved_path
                                    );
                                    None
                                }
                                Err(e) => {
                                    log::debug!(
                                        "Skipping C dependency resolution for '{}': {}",
                                        resolved_path,
                                        e
                                    );
                                    None
                                }
                            }
                        } else {
                            log::trace!(
                                "Could not resolve C include (system header): {}",
                                import_info.imported_path
                            );
                            None
                        }
                    } else if file_path.ends_with(".cpp")
                        || file_path.ends_with(".cc")
                        || file_path.ends_with(".cxx")
                        || file_path.ends_with(".hpp")
                        || file_path.ends_with(".hxx")
                        || file_path.ends_with(".h++")
                        || file_path.ends_with(".C")
                        || file_path.ends_with(".H")
                    {
                        // Resolve C++ dependencies (relative #include paths)
                        if let Some(resolved_path) =
                            crate::parsers::cpp::resolve_cpp_include_to_path(
                                &import_info.imported_path,
                                Some(&file_path),
                            )
                        {
                            // Look up file ID in database using exact match
                            match resolver.get_file_id_by_path(&resolved_path) {
                                Ok(Some(id)) => {
                                    log::trace!(
                                        "Resolved C++ dependency: {} -> {} (file_id={})",
                                        import_info.imported_path,
                                        resolved_path,
                                        id
                                    );
                                    Some(id)
                                }
                                Ok(None) => {
                                    log::trace!(
                                        "C++ dependency resolved to path but file not in index: {} -> {}",
                                        import_info.imported_path,
                                        resolved_path
                                    );
                                    None
                                }
                                Err(e) => {
                                    log::debug!(
                                        "Skipping C++ dependency resolution for '{}': {}",
                                        resolved_path,
                                        e
                                    );
                                    None
                                }
                            }
                        } else {
                            log::trace!(
                                "Could not resolve C++ include (system header): {}",
                                import_info.imported_path
                            );
                            None
                        }
                    } else if file_path.ends_with(".cs") {
                        // Resolve C# dependencies (using namespace-to-path mapping)
                        if let Some(resolved_path) =
                            crate::parsers::csharp::resolve_csharp_using_to_path(
                                &import_info.imported_path,
                                Some(&file_path),
                            )
                        {
                            // Look up file ID in database using exact match
                            match resolver.get_file_id_by_path(&resolved_path) {
                                Ok(Some(id)) => {
                                    log::trace!(
                                        "Resolved C# dependency: {} -> {} (file_id={})",
                                        import_info.imported_path,
                                        resolved_path,
                                        id
                                    );
                                    Some(id)
                                }
                                Ok(None) => {
                                    log::trace!(
                                        "C# dependency resolved to path but file not in index: {} -> {}",
                                        import_info.imported_path,
                                        resolved_path
                                    );
                                    None
                                }
                                Err(e) => {
                                    log::debug!(
                                        "Skipping C# dependency resolution for '{}': {}",
                                        resolved_path,
                                        e
                                    );
                                    None
                                }
                            }
                        } else {
                            log::trace!(
                                "Could not resolve C# using directive: {}",
                                import_info.imported_path
                            );
                            None
                        }
                    } else if file_path.ends_with(".zig") {
                        // Resolve Zig dependencies (relative @import paths)
                        if let Some(resolved_path) = crate::parsers::zig::resolve_zig_import_to_path(
                            &import_info.imported_path,
                            Some(&file_path),
                        ) {
                            // Look up file ID in database using exact match
                            match resolver.get_file_id_by_path(&resolved_path) {
                                Ok(Some(id)) => {
                                    log::trace!(
                                        "Resolved Zig dependency: {} -> {} (file_id={})",
                                        import_info.imported_path,
                                        resolved_path,
                                        id
                                    );
                                    Some(id)
                                }
                                Ok(None) => {
                                    log::trace!(
                                        "Zig dependency resolved to path but file not in index: {} -> {}",
                                        import_info.imported_path,
                                        resolved_path
                                    );
                                    None
                                }
                                Err(e) => {
                                    log::debug!(
                                        "Skipping Zig dependency resolution for '{}': {}",
                                        resolved_path,
                                        e
                                    );
                                    None
                                }
                            }
                        } else {
                            log::trace!(
                                "Could not resolve Zig import (external or stdlib): {}",
                                import_info.imported_path
                            );
                            None
                        }
                    } else if file_path.ends_with(".vue") || file_path.ends_with(".svelte") {
                        // Resolve Vue/Svelte dependencies (use TypeScript/JavaScript resolver for imports in <script> blocks)
                        let alias_map = find_nearest_tsconfig(&file_path, root, &tsconfigs);
                        if let Some(candidates_str) =
                            crate::parsers::typescript::resolve_ts_import_to_path(
                                &import_info.imported_path,
                                Some(&file_path),
                                alias_map,
                            )
                        {
                            // Parse pipe-delimited candidates (e.g., "path.tsx|path.ts|path.jsx|path.js")
                            let candidates: Vec<&str> = candidates_str.split('|').collect();

                            // Try each candidate in order until we find one in the database
                            let mut resolved_id = None;
                            for candidate_path in candidates {
                                // Normalize path to be relative to project root
                                // Convert absolute paths to relative (without requiring file to exist)
                                let normalized_candidate = if let Ok(rel_path) =
                                    std::path::Path::new(candidate_path).strip_prefix(root)
                                {
                                    rel_path.to_string_lossy().replace('\\', "/")
                                } else {
                                    // Not an absolute path or not under root - use as-is
                                    // (still normalize separators so DB lookups match).
                                    candidate_path.replace('\\', "/")
                                };

                                match resolver.get_file_id_by_path(&normalized_candidate) {
                                    Ok(Some(id)) => {
                                        log::trace!(
                                            "Resolved Vue/Svelte dependency: {} -> {} (file_id={})",
                                            import_info.imported_path,
                                            candidate_path,
                                            id
                                        );
                                        resolved_id = Some(id);
                                        break; // Found a match, stop trying
                                    }
                                    Ok(None) => {
                                        log::trace!(
                                            "Vue/Svelte candidate not in index: {}",
                                            candidate_path
                                        );
                                    }
                                    Err(e) => {
                                        log::debug!(
                                            "Skipping Vue/Svelte dependency resolution for '{}': {}",
                                            normalized_candidate,
                                            e
                                        );
                                    }
                                }
                            }

                            if resolved_id.is_none() {
                                log::trace!(
                                    "Vue/Svelte dependency: no matching file found in database for any candidate: {}",
                                    candidates_str
                                );
                            }

                            resolved_id
                        } else {
                            log::trace!(
                                "Could not resolve Vue/Svelte import (non-relative or external): {}",
                                import_info.imported_path
                            );
                            None
                        }
                    } else {
                        None
                    };

                    // resolved_file_id will be populated using deterministic language-specific resolution
                    // All language resolvers have been implemented!
                    resolved_deps.push(Dependency {
                        file_id,
                        imported_path: import_info.imported_path.clone(),
                        resolved_file_id,
                        import_type: import_info.import_type,
                        line_number: import_info.line_number,
                        imported_symbols: import_info.imported_symbols.clone(),
                    });
                }

                // Clear existing dependencies for this file, then insert the new
                // rows, inside the shared transaction.
                dep_writer.replace_dependencies(file_id, &resolved_deps)?;
                total_deps_inserted += resolved_deps.len();
            }

            log::info!("Extracted {} dependencies", total_deps_inserted);
        }

        // Step 2.6: Insert exports (after files are inserted and have IDs)
        if !all_exports.is_empty() {
            *progress_status.lock().unwrap() = "Extracting exports...".to_string();
            if show_progress {
                pb.set_message("Extracting exports...".to_string());
            }

            // `tsconfigs` (parsed once in Step 1.4) is reused here for alias resolution.
            let mut total_exports_inserted = 0;

            // Process each file's exports
            for (file_path, export_infos) in all_exports {
                // Get file ID from database
                let file_id = match resolver.get_file_id_by_path(&file_path)? {
                    Some(id) => id,
                    None => {
                        log::warn!(
                            "File not found in database (skipping exports): {}",
                            file_path
                        );
                        continue;
                    }
                };

                // Resolve export source paths and insert
                for export_info in export_infos {
                    // Resolve export source path (same logic as imports)
                    let resolved_source_id = if file_path.ends_with(".ts")
                        || file_path.ends_with(".tsx")
                        || file_path.ends_with(".js")
                        || file_path.ends_with(".jsx")
                        || file_path.ends_with(".mts")
                        || file_path.ends_with(".cts")
                        || file_path.ends_with(".mjs")
                        || file_path.ends_with(".cjs")
                        || file_path.ends_with(".vue")
                    {
                        // Resolve TypeScript/JavaScript/Vue export paths (relative imports and path aliases)
                        let alias_map = find_nearest_tsconfig(&file_path, root, &tsconfigs);
                        if let Some(candidates_str) =
                            crate::parsers::typescript::resolve_ts_import_to_path(
                                &export_info.source_path,
                                Some(&file_path),
                                alias_map,
                            )
                        {
                            // Parse pipe-delimited candidates (e.g., "path.tsx|path.ts|path.jsx|path.js|path.vue")
                            let candidates: Vec<&str> = candidates_str.split('|').collect();

                            // Try each candidate in order until we find one in the database
                            let mut resolved_id = None;
                            for candidate_path in candidates {
                                // Normalize path to be relative to project root
                                let normalized_candidate = if let Ok(rel_path) =
                                    std::path::Path::new(candidate_path).strip_prefix(root)
                                {
                                    rel_path.to_string_lossy().to_string()
                                } else {
                                    candidate_path.to_string()
                                };

                                match resolver.get_file_id_by_path(&normalized_candidate) {
                                    Ok(Some(id)) => {
                                        log::trace!(
                                            "Resolved export source: {} -> {} (file_id={})",
                                            export_info.source_path,
                                            normalized_candidate,
                                            id
                                        );
                                        resolved_id = Some(id);
                                        break; // Found a match, stop trying
                                    }
                                    Ok(None) => {
                                        log::trace!(
                                            "Export source candidate not in index: {}",
                                            candidate_path
                                        );
                                    }
                                    Err(e) => {
                                        log::debug!(
                                            "Skipping export source resolution for '{}': {}",
                                            normalized_candidate,
                                            e
                                        );
                                    }
                                }
                            }

                            if resolved_id.is_none() {
                                log::trace!(
                                    "Export source: no matching file found in database for any candidate: {}",
                                    candidates_str
                                );
                            }

                            resolved_id
                        } else {
                            log::trace!(
                                "Could not resolve export source (non-relative or external): {}",
                                export_info.source_path
                            );
                            None
                        }
                    } else {
                        None
                    };

                    // Insert export into database
                    dep_writer.insert_export(
                        file_id,
                        export_info.exported_symbol.as_deref(),
                        &export_info.source_path,
                        resolved_source_id,
                        export_info.line_number,
                    )?;

                    total_exports_inserted += 1;
                }
            }

            log::info!("Extracted {} exports", total_exports_inserted);
        }

        // One commit for every dependency and export row of this run.
        let (deps_written, exports_written) = dep_writer.commit()?;
        drop(dep_conn);
        if deps_written + exports_written > 0 {
            self.cache
                .checkpoint_wal()
                .context("Failed to checkpoint WAL after dependency recording")?;
        }
        log::info!(
            "phase dependencies+exports: {} rows, {} ms",
            deps_written + exports_written,
            deps_start.elapsed().as_millis()
        );

        log::info!("Indexed {} files", files_indexed);

        // Step 3: Write trigram index.
        // Crash-safe write: `TrigramIndex::write` streams into `trigrams.bin.tmp`,
        // syncs, then renames over `trigrams.bin` (see `atomic_write`). A crash
        // mid-write leaves the previous index untouched; the fast-path check
        // above still validates magic bytes as a second line of defence.
        *progress_status.lock().unwrap() = "Writing trigram index...".to_string();
        if show_progress {
            pb.set_message("Writing trigram index...".to_string());
        }
        let trigrams_path = self.cache.path().join("trigrams.bin");
        let write_start = Instant::now();
        trigram_builder
            .write(&pool, &trigrams_path)
            .context("Failed to write trigram index")?;
        log::info!(
            "phase trigram write: {} trigrams, {} files, {} ms",
            trigram_builder.trigram_count(),
            trigram_builder.file_count(),
            write_start.elapsed().as_millis()
        );

        // Step 4: Finalize content store (already been writing incrementally)
        *progress_status.lock().unwrap() = "Finalizing content store...".to_string();
        if show_progress {
            pb.set_message("Finalizing content store...".to_string());
        }
        content_writer
            .finalize_if_needed()
            .context("Failed to finalize content store")?;
        log::info!(
            "Wrote {} files ({} bytes) to content.bin",
            content_writer.file_count(),
            content_writer.content_size()
        );

        // Step 5: Update SQLite statistics from database totals (branch-aware)
        *progress_status.lock().unwrap() = "Updating statistics...".to_string();
        if show_progress {
            pb.set_message("Updating statistics...".to_string());
        }
        // Update stats for current branch only
        self.cache.update_stats(&branch)?;

        // Update schema hash to mark cache as compatible with current binary
        self.cache.update_schema_hash()?;

        pb.finish_with_message("Indexing complete");

        // Return stats with incremental breakdown
        let mut stats = self.cache.stats()?;
        stats.new_files = new_file_count;
        stats.modified_files = modified_file_count;
        stats.deleted_files = deleted_file_count;
        stats.unchanged_files = unchanged_file_count;
        stats.skipped_too_large = skipped_too_large;
        stats.skipped_bytes_too_large = skipped_bytes_too_large;
        stats.skipped_binary = skipped_binary;
        log::info!(
            "Indexing complete: {} files (new={}, modified={}, unchanged={})",
            stats.total_files,
            new_file_count,
            modified_file_count,
            unchanged_file_count
        );

        Ok(stats)
    }

    /// Discover all indexable files in the directory tree.
    ///
    /// Returns `(files, skipped_too_large_count, skipped_too_large_bytes)`.
    fn discover_files(&self, root: &Path) -> Result<Discovered> {
        let mut out = Discovered::default();

        let policy = self.path_policy(root);
        let walker = Self::walk_builder(root, &self.config, &policy).build();

        for entry in walker {
            let entry = entry?;
            let path = entry.path();

            // Only process files (not directories)
            if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                continue;
            }

            // Check extension / language eligibility first (cheap)
            let Some(lang) = policy.classify(path) else {
                continue;
            };

            // Check file size separately so we can report skipped counts
            let mut size = 0u64;
            if let Ok(metadata) = std::fs::metadata(path) {
                size = metadata.len();
                if size > self.config.max_file_size as u64 {
                    log::debug!("Skipping {} (too large: {} bytes)", path.display(), size);
                    out.skipped_too_large += 1;
                    out.skipped_bytes_too_large += size;
                    continue;
                }
            }

            // A code extension is trusted to be text. Anything else in the tracked
            // tier (`image.png`, `OWNERS`, `data.bin`) is sniffed: ripgrep's rule, a
            // NUL byte anywhere means binary, and a binary file is never in the
            // index. Only the long tail pays the read (from the page cache, since
            // the main pass reads it again a moment later).
            if !lang.is_code() && looks_binary(path) {
                log::debug!("Skipping {} (binary)", path.display());
                out.skipped_binary += 1;
                continue;
            }

            out.files.push(path.to_path_buf());
            out.sizes.push(size);
        }

        Ok(out)
    }

    /// The directory walker every tree pass shares: the indexer, and the freshness
    /// check outside git (which has no `git status` to name candidates and must
    /// walk). One builder so the two can never disagree about what is in the tree.
    pub fn walk_builder(root: &Path, config: &IndexConfig, policy: &PathPolicy) -> WalkBuilder {
        // WalkBuilder from ignore crate automatically respects:
        // - .gitignore (when in a git repo)
        // - .ignore files
        // - Hidden files (can be configured)
        let mut builder = WalkBuilder::new(root);
        builder
            .follow_links(config.follow_symlinks)
            .hidden(!policy.hidden())
            .git_ignore(true) // Explicitly enable gitignore support (enabled by default, but be explicit)
            .git_global(false) // Don't use global gitignore
            .git_exclude(false); // Don't use .git/info/exclude
        if policy.hidden() {
            // Dot-directories are walked, but never the repository's own and never
            // Reflex's own cache: indexing `.reflex/content.bin` into content.bin
            // is a loop, and `.git/objects` is binary noise by the thousand.
            builder.filter_entry(|e| {
                let name = e.file_name();
                name != ".git" && name != crate::cache::CACHE_DIR
            });
        }
        // `[index] include.patterns` / `exclude.patterns`, gitignore semantics.
        if let Some(ov) = policy.overrides() {
            builder.overrides(ov.clone());
        }
        builder
    }

    /// Whether a path is one Reflex would index, judged by extension alone.
    ///
    /// Public so freshness checking can ask the same question the walker asks. If the
    /// two ever disagree, editing a file Reflex does not index (a README, anything
    /// under `target/`) would mark the index permanently stale — a cure worse than
    /// the disease.
    ///
    /// Judges extension only: no filesystem access, no size check, no `.gitignore`
    /// (git's own output is already filtered by that). Cheap enough to call per path
    /// in a `git status` listing.
    pub fn is_indexable_path(path: &Path) -> bool {
        Self::is_indexable_path_with(path, None)
    }

    /// [`Self::is_indexable_path`] under an `[index]` policy.
    ///
    /// The freshness check must apply the same policy as the walker: a file the
    /// config excludes is not indexed, so editing it must not report staleness.
    pub fn is_indexable_path_with(path: &Path, policy: Option<&PathPolicy>) -> bool {
        let default_policy;
        let policy = match policy {
            Some(p) => p,
            None => {
                default_policy = PathPolicy::default();
                &default_policy
            }
        };
        // Mirror the walker's hidden rule. Without this, Reflex's OWN
        // `.reflex/config.toml` counts as an indexable change the moment the text
        // tier claims `.toml`, and the index reports itself permanently stale.
        policy.hidden_ok(path) && policy.classify(path).is_some()
    }

    /// Check if a file's language/extension is eligible for indexing (without size check).
    fn should_index_lang(&self, path: &Path) -> bool {
        PathPolicy::from_config(Path::new("."), &self.config)
            .classify(path)
            .is_some()
    }

    /// Check if a file should be indexed based on config (language + size).
    #[allow(dead_code)]
    fn should_index(&self, path: &Path) -> bool {
        if !self.should_index_lang(path) {
            return false;
        }

        // Check file size limits
        if let Ok(metadata) = std::fs::metadata(path)
            && metadata.len() > self.config.max_file_size as u64
        {
            log::debug!(
                "Skipping {} (too large: {} bytes)",
                path.display(),
                metadata.len()
            );
            return false;
        }

        // `[index] include/exclude` patterns are applied by the walker
        // (`discover_files`) and by `is_indexable_path_with`; this per-file check
        // has no root to anchor them against.
        true
    }

    /// Compute blake3 hash from file contents for change detection
    fn hash_content(&self, content: &[u8]) -> String {
        let hash = blake3::hash(content);
        hash.to_hex().to_string()
    }

    /// Check available disk space before indexing
    ///
    /// Ensures there's enough free space to create the index. Warns if disk space is low.
    /// This prevents partial index writes and confusing error messages.
    #[cfg_attr(not(unix), allow(unused_variables))]
    fn check_disk_space(&self, root: &Path) -> Result<()> {
        // Get available space on the filesystem containing the cache directory
        let cache_path = self.cache.path();

        // Use statvfs on Unix systems
        #[cfg(unix)]
        {
            // On Linux, we can use statvfs to get available space
            // For now, we'll use a simple heuristic: warn if we can't write a test file
            let test_file = cache_path.join(".space_check");
            match std::fs::write(&test_file, b"test") {
                Ok(_) => {
                    let _ = std::fs::remove_file(&test_file);

                    // Try to estimate available space using df command
                    if let Ok(output) = std::process::Command::new("df")
                        .arg("-k")
                        .arg(cache_path.parent().unwrap_or(root))
                        .output()
                        && let Ok(df_output) = String::from_utf8(output.stdout)
                    {
                        // Parse df output to get available KB
                        if let Some(line) = df_output.lines().nth(1) {
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            if parts.len() >= 4
                                && let Ok(available_kb) = parts[3].parse::<u64>()
                            {
                                let available_mb = available_kb / 1024;

                                // Warn if less than 100MB available
                                if available_mb < 100 {
                                    log::warn!(
                                        "Low disk space: only {}MB available. Indexing may fail.",
                                        available_mb
                                    );
                                    output::warn(&format!(
                                        "Low disk space ({}MB available). Consider freeing up space.",
                                        available_mb
                                    ));
                                } else {
                                    log::debug!("Available disk space: {}MB", available_mb);
                                }
                            }
                        }
                    }

                    Ok(())
                }
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                    anyhow::bail!(
                        "Permission denied writing to cache directory: {}. Check file permissions.",
                        cache_path.display()
                    )
                }
                Err(e) => {
                    // If we can't write, it might be a disk space issue
                    log::warn!(
                        "Failed to write test file (possible disk space issue): {}",
                        e
                    );
                    Err(e).context(
                        "Failed to verify disk space - indexing may fail due to insufficient space",
                    )
                }
            }
        }

        #[cfg(not(unix))]
        {
            // On Windows, try to write a test file
            let test_file = cache_path.join(".space_check");
            match std::fs::write(&test_file, b"test") {
                Ok(_) => {
                    let _ = std::fs::remove_file(&test_file);
                    Ok(())
                }
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                    anyhow::bail!(
                        "Permission denied writing to cache directory: {}. Check file permissions.",
                        cache_path.display()
                    )
                }
                Err(e) => {
                    log::warn!(
                        "Failed to write test file (possible disk space issue): {}",
                        e
                    );
                    Err(e).context(
                        "Failed to verify disk space - indexing may fail due to insufficient space",
                    )
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_indexer_creation() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        assert!(indexer.cache.path().ends_with(".reflex"));
    }

    #[test]
    fn test_hash_content() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        let content1 = b"hello world";
        let content2 = b"hello world";
        let content3 = b"different content";

        let hash1 = indexer.hash_content(content1);
        let hash2 = indexer.hash_content(content2);
        let hash3 = indexer.hash_content(content3);

        // Same content should produce same hash
        assert_eq!(hash1, hash2);

        // Different content should produce different hash
        assert_ne!(hash1, hash3);

        // Hash should be hex string
        assert_eq!(hash1.len(), 64); // blake3 hash is 32 bytes = 64 hex chars
    }

    #[test]
    fn test_should_index_rust_file() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        // Create a small Rust file
        let rust_file = temp.path().join("test.rs");
        fs::write(&rust_file, "fn main() {}").unwrap();

        assert!(indexer.should_index(&rust_file));
    }

    #[test]
    fn test_should_index_unsupported_extension() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        // Tracked mode (the default) takes every file by path; whether a file is
        // binary is decided from its bytes in `discover_files`, not here.
        let unsupported_file = temp.path().join("test.xyz");
        fs::write(&unsupported_file, "mystery format").unwrap();
        assert!(indexer.should_index(&unsupported_file));

        // Allowlist mode keeps the pre-2.0.0 rule.
        let allowlist = Indexer::new(
            CacheManager::new(temp.path()),
            IndexConfig {
                mode: IndexMode::Allowlist,
                ..Default::default()
            },
        );
        assert!(!allowlist.should_index(&unsupported_file));
        let binary_file = temp.path().join("logo.png");
        fs::write(&binary_file, "not really a png").unwrap();
        assert!(!allowlist.should_index(&binary_file));
    }

    #[test]
    fn test_binary_sniff() {
        assert!(!is_binary(b"plain text\n"));
        assert!(is_binary(b"\x89PNG\r\n\x1a\n\0\0"));
        // Anywhere, not just the first 8 KB: ripgrep skips such a file too.
        let mut late = vec![b'a'; 64 * 1024];
        late.push(0);
        assert!(is_binary(&late));
    }

    #[test]
    fn test_should_index_text_tier() {
        let temp = TempDir::new().unwrap();
        let indexer = Indexer::new(CacheManager::new(temp.path()), IndexConfig::default());

        for name in [
            "notes.md",
            "config.yaml",
            "data.json",
            "api.proto",
            "run.sh",
        ] {
            let path = temp.path().join(name);
            fs::write(&path, "realm_marker").unwrap();
            assert!(indexer.should_index(&path), "{name} should be indexed");
        }

        // Lock files are indexed in tracked mode (and left out of searches unless
        // asked for); allowlist mode never indexes them.
        let lock = temp.path().join("package-lock.json");
        fs::write(&lock, "{}").unwrap();
        assert!(indexer.should_index(&lock));
        let allowlist = Indexer::new(
            CacheManager::new(temp.path()),
            IndexConfig {
                mode: IndexMode::Allowlist,
                ..Default::default()
            },
        );
        assert!(!allowlist.should_index(&lock));
    }

    #[test]
    fn test_text_tier_can_be_disabled() {
        let temp = TempDir::new().unwrap();
        let indexer = Indexer::new(
            CacheManager::new(temp.path()),
            IndexConfig {
                text_tier: false,
                ..Default::default()
            },
        );

        let md = temp.path().join("notes.md");
        fs::write(&md, "text").unwrap();
        assert!(!indexer.should_index(&md));

        let rs = temp.path().join("main.rs");
        fs::write(&rs, "fn main() {}").unwrap();
        assert!(indexer.should_index(&rs), "code is unaffected");
    }

    #[test]
    fn test_should_index_no_extension() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        // Tracked mode: every extensionless text file. Allowlist mode: only the
        // names on the list (`Makefile`), not `README`.
        let makefile = temp.path().join("Makefile");
        fs::write(&makefile, "all:\n\techo hello").unwrap();
        assert!(indexer.should_index(&makefile));

        let readme = temp.path().join("README");
        fs::write(&readme, "hello").unwrap();
        assert!(indexer.should_index(&readme));

        let allowlist = Indexer::new(
            CacheManager::new(temp.path()),
            IndexConfig {
                mode: IndexMode::Allowlist,
                ..Default::default()
            },
        );
        assert!(allowlist.should_index(&makefile));
        assert!(!allowlist.should_index(&readme));
    }

    #[test]
    fn test_should_index_size_limit() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        // Config with 100 byte size limit
        let config = IndexConfig {
            max_file_size: 100,
            ..Default::default()
        };

        let indexer = Indexer::new(cache, config);

        // Create small file (should be indexed)
        let small_file = temp.path().join("small.rs");
        fs::write(&small_file, "fn main() {}").unwrap();
        assert!(indexer.should_index(&small_file));

        // Create large file (should be skipped)
        let large_file = temp.path().join("large.rs");
        let large_content = "a".repeat(150);
        fs::write(&large_file, large_content).unwrap();
        assert!(!indexer.should_index(&large_file));
    }

    #[test]
    fn test_discover_files_empty_dir() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        let files = indexer.discover_files(temp.path()).unwrap().files;
        assert_eq!(files.len(), 0);
    }

    #[test]
    fn test_discover_files_single_file() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        // Create a Rust file
        let rust_file = temp.path().join("main.rs");
        fs::write(&rust_file, "fn main() {}").unwrap();

        let files = indexer.discover_files(temp.path()).unwrap().files;
        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("main.rs"));
    }

    #[test]
    fn test_discover_files_multiple_languages() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        // Create files of different languages
        fs::write(temp.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(temp.path().join("script.py"), "print('hello')").unwrap();
        fs::write(temp.path().join("app.js"), "console.log('hi')").unwrap();
        // Since 1.7.2 markdown IS indexed, in the plain-text tier.
        fs::write(temp.path().join("README.md"), "# Project").unwrap();
        // Since 2.0.0 (tracked mode) every non-binary file is indexed, whatever
        // its extension; a binary one is sniffed out.
        fs::write(temp.path().join("mystery.xyz"), "?").unwrap();
        fs::write(temp.path().join("blob.bin"), b"\0\x01\x02").unwrap();

        let found = indexer.discover_files(temp.path()).unwrap();
        assert_eq!(found.files.len(), 5, "3 code files, the markdown, the .xyz");
        assert_eq!(found.skipped_binary, 1);
    }

    #[test]
    fn test_discover_files_subdirectories() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        // Create nested directory structure
        let src_dir = temp.path().join("src");
        fs::create_dir(&src_dir).unwrap();
        fs::write(src_dir.join("main.rs"), "fn main() {}").unwrap();
        fs::write(src_dir.join("lib.rs"), "pub mod test {}").unwrap();

        let tests_dir = temp.path().join("tests");
        fs::create_dir(&tests_dir).unwrap();
        fs::write(tests_dir.join("test.rs"), "#[test] fn test() {}").unwrap();

        let files = indexer.discover_files(temp.path()).unwrap().files;
        assert_eq!(files.len(), 3);
    }

    #[test]
    fn test_discover_files_respects_gitignore() {
        let temp = TempDir::new().unwrap();

        // Initialize git repo (required for .gitignore to work with WalkBuilder)
        std::process::Command::new("git")
            .arg("init")
            .current_dir(temp.path())
            .output()
            .expect("Failed to initialize git repo");

        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        // Create .gitignore - use "ignored/" pattern to ignore the directory
        // Note: WalkBuilder respects .gitignore ONLY in git repositories
        fs::write(temp.path().join(".gitignore"), "ignored/\n").unwrap();

        // Create files
        fs::write(temp.path().join("included.rs"), "fn main() {}").unwrap();
        fs::write(temp.path().join("also_included.py"), "print('hi')").unwrap();

        let ignored_dir = temp.path().join("ignored");
        fs::create_dir(&ignored_dir).unwrap();
        fs::write(ignored_dir.join("excluded.rs"), "fn test() {}").unwrap();

        let files = indexer.discover_files(temp.path()).unwrap().files;

        // Verify the expected files are found
        assert!(
            files.iter().any(|f| f.ends_with("included.rs")),
            "Should find included.rs"
        );
        assert!(
            files.iter().any(|f| f.ends_with("also_included.py")),
            "Should find also_included.py"
        );

        // Verify excluded.rs in ignored/ directory is NOT found
        // This is the key test - gitignore should filter it out
        assert!(
            !files.iter().any(|f| {
                let path_str = f.to_string_lossy();
                path_str.contains("ignored") && f.ends_with("excluded.rs")
            }),
            "Should NOT find excluded.rs in ignored/ directory (gitignore pattern)"
        );

        // Should find exactly 2 files (included.rs and also_included.py)
        // .gitignore file itself has no supported language extension, so it won't be indexed
        assert_eq!(
            files.len(),
            2,
            "Should find exactly 2 files (not including .gitignore or ignored/excluded.rs)"
        );
    }

    #[test]
    fn test_index_empty_directory() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        let stats = indexer.index(temp.path(), false).unwrap();

        assert_eq!(stats.total_files, 0);
    }

    #[test]
    fn test_index_single_rust_file() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        // Create a Rust file
        fs::write(
            project_root.join("main.rs"),
            "fn main() { println!(\"Hello\"); }",
        )
        .unwrap();

        let stats = indexer.index(&project_root, false).unwrap();

        assert_eq!(stats.total_files, 1);
        assert!(stats.files_by_language.contains_key("Rust"));
    }

    #[test]
    fn test_index_multiple_files() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        // Create multiple files
        fs::write(project_root.join("main.rs"), "fn main() {}").unwrap();
        fs::write(project_root.join("lib.rs"), "pub fn test() {}").unwrap();
        fs::write(project_root.join("script.py"), "def main(): pass").unwrap();

        let stats = indexer.index(&project_root, false).unwrap();

        assert_eq!(stats.total_files, 3);
        assert_eq!(stats.files_by_language.get("Rust"), Some(&2));
        assert_eq!(stats.files_by_language.get("Python"), Some(&1));
    }

    #[test]
    fn test_index_creates_trigram_index() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        fs::write(project_root.join("main.rs"), "fn main() {}").unwrap();

        indexer.index(&project_root, false).unwrap();

        // Verify trigrams.bin was created
        let trigrams_path = project_root.join(".reflex/trigrams.bin");
        assert!(trigrams_path.exists());
    }

    #[test]
    fn test_index_creates_content_store() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        fs::write(project_root.join("main.rs"), "fn main() {}").unwrap();

        indexer.index(&project_root, false).unwrap();

        // Verify content.bin was created
        let content_path = project_root.join(".reflex/content.bin");
        assert!(content_path.exists());
    }

    #[test]
    fn test_index_incremental_no_changes() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        fs::write(project_root.join("main.rs"), "fn main() {}").unwrap();

        // First index
        let stats1 = indexer.index(&project_root, false).unwrap();
        assert_eq!(stats1.total_files, 1);

        // Second index without changes
        let stats2 = indexer.index(&project_root, false).unwrap();
        assert_eq!(stats2.total_files, 1);
    }

    #[test]
    fn test_index_incremental_with_changes() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        let main_path = project_root.join("main.rs");
        fs::write(&main_path, "fn main() {}").unwrap();

        // First index
        indexer.index(&project_root, false).unwrap();

        // Modify file
        fs::write(&main_path, "fn main() { println!(\"changed\"); }").unwrap();

        // Second index should detect change
        let stats = indexer.index(&project_root, false).unwrap();
        assert_eq!(stats.total_files, 1);
    }

    #[test]
    fn test_index_incremental_new_file() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        fs::write(project_root.join("main.rs"), "fn main() {}").unwrap();

        // First index
        let stats1 = indexer.index(&project_root, false).unwrap();
        assert_eq!(stats1.total_files, 1);

        // Add new file
        fs::write(project_root.join("lib.rs"), "pub fn test() {}").unwrap();

        // Second index should include new file
        let stats2 = indexer.index(&project_root, false).unwrap();
        assert_eq!(stats2.total_files, 2);
    }

    #[test]
    fn test_index_parallel_threads_config() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);

        // Test with explicit thread count
        let config = IndexConfig {
            parallel_threads: 2,
            ..Default::default()
        };

        let indexer = Indexer::new(cache, config);

        fs::write(project_root.join("main.rs"), "fn main() {}").unwrap();

        let stats = indexer.index(&project_root, false).unwrap();
        assert_eq!(stats.total_files, 1);
    }

    #[test]
    fn test_index_parallel_threads_auto() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);

        // Test with auto thread count (0 = auto)
        let config = IndexConfig {
            parallel_threads: 0,
            ..Default::default()
        };

        let indexer = Indexer::new(cache, config);

        fs::write(project_root.join("main.rs"), "fn main() {}").unwrap();

        let stats = indexer.index(&project_root, false).unwrap();
        assert_eq!(stats.total_files, 1);
    }

    #[test]
    fn test_index_respects_size_limit() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);

        // Very small size limit
        let config = IndexConfig {
            max_file_size: 50,
            ..Default::default()
        };

        let indexer = Indexer::new(cache, config);

        // Small file (should be indexed)
        fs::write(project_root.join("small.rs"), "fn a() {}").unwrap();

        // Large file (should be skipped)
        let large_content = "fn main() {}\n".repeat(10);
        fs::write(project_root.join("large.rs"), large_content).unwrap();

        let stats = indexer.index(&project_root, false).unwrap();

        // Only small file should be indexed
        assert_eq!(stats.total_files, 1);
    }

    #[test]
    fn test_index_mixed_languages() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        // Create files in multiple languages
        fs::write(project_root.join("main.rs"), "fn main() {}").unwrap();
        fs::write(project_root.join("test.py"), "def test(): pass").unwrap();
        fs::write(project_root.join("app.js"), "function main() {}").unwrap();
        fs::write(project_root.join("lib.go"), "func main() {}").unwrap();

        let stats = indexer.index(&project_root, false).unwrap();

        assert_eq!(stats.total_files, 4);
        assert!(stats.files_by_language.contains_key("Rust"));
        assert!(stats.files_by_language.contains_key("Python"));
        assert!(stats.files_by_language.contains_key("JavaScript"));
        assert!(stats.files_by_language.contains_key("Go"));
    }

    #[test]
    fn test_index_updates_cache_stats() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        fs::write(project_root.join("main.rs"), "fn main() {}").unwrap();

        indexer.index(&project_root, false).unwrap();

        // Verify cache stats were updated
        let cache = CacheManager::new(&project_root);
        let stats = cache.stats().unwrap();

        assert_eq!(stats.total_files, 1);
        assert!(stats.index_size_bytes > 0);
    }
}
