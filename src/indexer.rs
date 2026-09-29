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
use crate::content_store::ContentWriter;
use crate::models::{IndexConfig, IndexMode, IndexStats, Language};
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
    hash: String,
    content: String,
    language: Language,
    line_count: usize,
    /// On-disk size and mtime, taken BEFORE the read so a write that lands
    /// between the two is caught by the next status check, not hidden by it.
    size: u64,
    mtime_ns: i64,
    /// Imports and re-exports, when they were extracted this run.
    imports: Option<(Vec<ImportInfo>, Vec<ExportInfo>)>,
    /// The file's trigram postings, extracted in the pool (no file id yet).
    trigram_run: TrigramRun,
}

/// `statistics` key of the resolver config digest the dependency rows were
/// resolved with (see [`crate::dependency_resolve::ResolverConfigs::digest`]).
const RESOLVER_DIGEST_KEY: &str = "resolver_config_digest";

/// `statistics` key of the manifest generation meta.db's rows go with. A manifest
/// of another generation means a run stopped between the two commits.
pub const INDEX_GENERATION_KEY: &str = "index_generation";

/// File in `.reflex/` whose mtime is an index run's start (see `run_marker`).
const RUN_MARKER: &str = ".index-run";

/// What a run found for one discovered file, compared with its `files` row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileStatus {
    /// Same size and mtime as stored: not read.
    Unchanged,
    /// Different stat, same bytes (a `touch`, or an edit reverted).
    Touched,
    /// Different bytes.
    Modified,
    /// No row.
    Added,
}

/// One file the build read, in walk order.
struct IndexedFile {
    /// Position in the discovery lists.
    index: usize,
    hash: String,
    language: Language,
    line_count: usize,
    size: u64,
    mtime_ns: i64,
    /// Imports and re-exports, when they were extracted this run.
    imports: Option<(Vec<ImportInfo>, Vec<ExportInfo>)>,
}

/// Inputs of `Indexer::try_delta_update`.
struct DeltaUpdate<'a> {
    root: &'a Path,
    files: &'a [PathBuf],
    rels: &'a [String],
    metas: &'a [Option<std::fs::Metadata>],
    status: &'a [FileStatus],
    stored: &'a HashMap<String, crate::meta_update::StoredFile>,
    existing_hashes: &'a HashMap<String, String>,
    dirty_paths: &'a std::collections::HashSet<String>,
    branch: &'a str,
    commit: Option<&'a str>,
    git_dirty: bool,
    resolver_configs: &'a crate::dependency_resolve::ResolverConfigs,
    run_start: std::time::SystemTime,
    generation: u64,
    pool: &'a rayon::ThreadPool,
    deleted_file_count: usize,
    /// (too large, bytes too large, binary)
    skipped: (usize, u64, usize),
}

/// What `Indexer::process_file` reads from the run.
struct ProcessCtx<'a> {
    root: &'a Path,
    files: &'a [PathBuf],
    rels: &'a [String],
    stored: &'a HashMap<String, crate::meta_update::StoredFile>,
    run_start: std::time::SystemTime,
    full_deps: bool,
    tsconfigs: &'a HashMap<PathBuf, crate::parsers::tsconfig::PathAliasMap>,
}

/// Inputs of `Indexer::refresh_unchanged`.
struct RefreshUnchanged<'a> {
    rels: &'a [String],
    metas: &'a [Option<std::fs::Metadata>],
    status: &'a [FileStatus],
    stored: &'a HashMap<String, crate::meta_update::StoredFile>,
    existing_hashes: &'a HashMap<String, String>,
    dirty_paths: &'a std::collections::HashSet<String>,
    branch: &'a str,
    commit: Option<&'a str>,
    /// `Some(dirty)` inside git.
    git_dirty: Option<bool>,
    run_start: std::time::SystemTime,
    /// (too large, bytes too large, binary)
    skipped: (usize, u64, usize),
}

/// Inputs of `Indexer::write_meta`.
struct MetaWrite<'a> {
    root: &'a Path,
    rels: &'a [String],
    /// Discovery indices of the files in the new snapshot, in walk order.
    present: &'a [usize],
    /// Files read this run, in walk order (a subset of `present`).
    written: &'a [IndexedFile],
    metas: &'a [Option<std::fs::Metadata>],
    run_start: std::time::SystemTime,
    stored: &'a HashMap<String, crate::meta_update::StoredFile>,
    status: &'a [FileStatus],
    dirty_paths: &'a std::collections::HashSet<String>,
    branch: &'a str,
    commit: Option<&'a str>,
    resolver_configs: &'a crate::dependency_resolve::ResolverConfigs,
    full_deps: bool,
    /// Generation of the manifest these rows go with.
    generation: u64,
}

/// (new, modified, unchanged) of `(path, hash)` pairs against a branch's hashes.
fn breakdown<'a>(
    files: impl Iterator<Item = (&'a str, &'a str)>,
    existing_hashes: &HashMap<String, String>,
) -> (usize, usize, usize) {
    let (mut new, mut modified, mut unchanged) = (0, 0, 0);
    for (path, hash) in files {
        match existing_hashes.get(path) {
            None => new += 1,
            Some(old) if old != hash => modified += 1,
            _ => unchanged += 1,
        }
    }
    (new, modified, unchanged)
}

/// Resolve again every stored import (other than External/Stdlib) and every
/// export of the files not in `skip`, updating the rows whose target changed.
/// Returns how many rows changed.
fn reresolve(
    tx: &rusqlite::Connection,
    ctx: &crate::dependency_resolve::ResolverContext<'_>,
    resolver: &crate::dependency::PathResolver,
    skip: &std::collections::HashSet<i64>,
) -> Result<usize> {
    let mut changed: Vec<(i64, Option<i64>)> = Vec::new();
    {
        let mut stmt = tx.prepare(
            "SELECT d.id, d.file_id, f.path, d.imported_path, d.resolved_file_id
             FROM file_dependencies d JOIN files f ON f.id = d.file_id
             WHERE d.import_type NOT IN ('external', 'stdlib')",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, Option<i64>>(4)?,
            ))
        })?;
        for row in rows {
            let (id, file_id, path, imported_path, old) = row?;
            if skip.contains(&file_id) {
                continue;
            }
            // Resolution reads only the imported path (the stored type is already
            // the reclassified one, and External/Stdlib rows are skipped above).
            let import = ImportInfo {
                imported_path,
                import_type: crate::models::ImportType::Internal,
                line_number: 0,
                imported_symbols: None,
            };
            let new = ctx.resolve_import(&path, &import, resolver);
            if new != old {
                changed.push((id, new));
            }
        }
    }
    let mut stmt = tx.prepare("UPDATE file_dependencies SET resolved_file_id = ? WHERE id = ?")?;
    for (id, new) in &changed {
        stmt.execute(rusqlite::params![new, id])?;
    }
    let mut total = changed.len();

    let mut changed: Vec<(i64, Option<i64>)> = Vec::new();
    {
        let mut stmt = tx.prepare(
            "SELECT e.id, e.file_id, f.path, e.source_path, e.resolved_source_id
             FROM file_exports e JOIN files f ON f.id = e.file_id",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, Option<i64>>(4)?,
            ))
        })?;
        for row in rows {
            let (id, file_id, path, source_path, old) = row?;
            if skip.contains(&file_id) {
                continue;
            }
            let export = ExportInfo {
                exported_symbol: None,
                source_path,
                line_number: 0,
            };
            let new = ctx.resolve_export(&path, &export, resolver);
            if new != old {
                changed.push((id, new));
            }
        }
    }
    let mut stmt = tx.prepare("UPDATE file_exports SET resolved_source_id = ? WHERE id = ?")?;
    for (id, new) in &changed {
        stmt.execute(rusqlite::params![new, id])?;
    }
    total += changed.len();
    Ok(total)
}

/// Imports and re-exports of one file, by language. `path_str` is the path as
/// walked (for the nearest tsconfig).
fn extract_imports(
    language: Language,
    content: &str,
    path_str: &str,
    root: &Path,
    tsconfigs: &HashMap<PathBuf, crate::parsers::tsconfig::PathAliasMap>,
) -> (Vec<ImportInfo>, Vec<ExportInfo>) {
    // Extract dependencies and exports for supported languages
    let mut parsed_exports: Vec<ExportInfo> = Vec::new();
    let dependencies = match language {
        Language::Rust => match RustDependencyExtractor::extract_dependencies(content) {
            Ok(deps) => deps,
            Err(e) => {
                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                Vec::new()
            }
        },
        Language::Python => match PythonDependencyExtractor::extract_dependencies(content) {
            Ok(deps) => deps,
            Err(e) => {
                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                Vec::new()
            }
        },
        Language::TypeScript | Language::JavaScript => {
            // Find nearest tsconfig for path alias resolution. One parse
            // yields both the imports and the re-exports.
            let alias_map =
                crate::dependency_resolve::find_nearest_tsconfig(path_str, root, tsconfigs);
            match TypeScriptDependencyExtractor::extract_dependencies_and_exports(
                content, alias_map,
            ) {
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
        Language::Go => match GoDependencyExtractor::extract_dependencies(content) {
            Ok(deps) => deps,
            Err(e) => {
                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                Vec::new()
            }
        },
        Language::Java => match JavaDependencyExtractor::extract_dependencies(content) {
            Ok(deps) => deps,
            Err(e) => {
                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                Vec::new()
            }
        },
        Language::C => match CDependencyExtractor::extract_dependencies(content) {
            Ok(deps) => deps,
            Err(e) => {
                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                Vec::new()
            }
        },
        Language::Cpp => match CppDependencyExtractor::extract_dependencies(content) {
            Ok(deps) => deps,
            Err(e) => {
                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                Vec::new()
            }
        },
        Language::CSharp => match CSharpDependencyExtractor::extract_dependencies(content) {
            Ok(deps) => deps,
            Err(e) => {
                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                Vec::new()
            }
        },
        Language::PHP => match PhpDependencyExtractor::extract_dependencies(content) {
            Ok(deps) => deps,
            Err(e) => {
                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                Vec::new()
            }
        },
        Language::Ruby => match RubyDependencyExtractor::extract_dependencies(content) {
            Ok(deps) => deps,
            Err(e) => {
                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                Vec::new()
            }
        },
        Language::Kotlin => match KotlinDependencyExtractor::extract_dependencies(content) {
            Ok(deps) => deps,
            Err(e) => {
                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                Vec::new()
            }
        },
        Language::Zig => match ZigDependencyExtractor::extract_dependencies(content) {
            Ok(deps) => deps,
            Err(e) => {
                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                Vec::new()
            }
        },
        Language::Vue => {
            // Find nearest tsconfig for path alias resolution. One parse
            // per script block yields both the imports and the re-exports.
            let alias_map =
                crate::dependency_resolve::find_nearest_tsconfig(path_str, root, tsconfigs);
            match VueDependencyExtractor::extract_dependencies_and_exports(content, alias_map) {
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
        Language::Svelte => match SvelteDependencyExtractor::extract_dependencies(content) {
            Ok(deps) => deps,
            Err(e) => {
                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                Vec::new()
            }
        },
        // Other languages not yet implemented
        _ => Vec::new(),
    };

    // Exports (barrel re-export tracking) came out of the same parse as the
    // dependencies above; only TypeScript/JavaScript/Vue have them.
    (dependencies, parsed_exports)
}

/// Manages the indexing process
pub struct Indexer {
    cache: CacheManager,
    config: IndexConfig,
    /// `(max_files, max_bytes)` per batch; `None` = defaults / env overrides.
    batch_limits: Option<(usize, u64)>,
    /// `(max_files, max_bytes)` of the delta before it is merged into a new base;
    /// `None` = [`DELTA_MAX_FILES`] and [`DELTA_MAX_CORPUS_PERCENT`] of the corpus.
    merge_limits: Option<(usize, u64)>,
}

/// Files the delta may hold before an update merges it into a new base.
pub const DELTA_MAX_FILES: usize = 2000;
/// Text bytes the delta may hold, as a percentage of the live corpus, before an
/// update merges it into a new base.
pub const DELTA_MAX_CORPUS_PERCENT: u64 = 5;

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
    /// Each entry of `files` relative to the root, with forward slashes: the path
    /// the stores and meta.db use.
    rels: Vec<String>,
    /// On-disk size of each entry of `files` (0 when unknown); drives batching.
    sizes: Vec<u64>,
    /// The `stat` of each entry of `files`, taken during the walk.
    metas: Vec<Option<std::fs::Metadata>>,
    skipped_too_large: usize,
    skipped_bytes_too_large: u64,
    skipped_binary: usize,
}

/// `file_path` relative to `root` with forward slashes (a path that is not under
/// `root` loses a leading `./`): the path the stores and meta.db use, the same
/// on every OS.
fn normalize_rel(root: &Path, file_path: &Path) -> String {
    match file_path.strip_prefix(root) {
        Ok(rel) => rel.to_string_lossy().replace('\\', "/"),
        Err(_) => file_path
            .to_string_lossy()
            .trim_start_matches("./")
            .replace('\\', "/"),
    }
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
            merge_limits: None,
        }
    }

    /// Override the per-batch limits (files, bytes). For tests that need to
    /// exercise multi-batch builds on small trees; production reads
    /// `REFLEX_INDEX_BATCH_FILES` / `REFLEX_INDEX_BATCH_BYTES` or the defaults.
    #[doc(hidden)]
    pub fn set_batch_limits(&mut self, max_files: usize, max_bytes: u64) {
        self.batch_limits = Some((max_files, max_bytes));
    }

    /// Override the delta's merge limits (files, text bytes). For tests: a small
    /// tree would otherwise merge on every update under the 5 % rule.
    #[doc(hidden)]
    pub fn set_merge_limits(&mut self, max_files: usize, max_bytes: u64) {
        self.merge_limits = Some((max_files, max_bytes));
    }

    fn merge_limits(&self, live_corpus_bytes: u64) -> (usize, u64) {
        self.merge_limits.unwrap_or((
            DELTA_MAX_FILES,
            live_corpus_bytes * DELTA_MAX_CORPUS_PERCENT / 100,
        ))
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

        // Whether the cache was last completed by a binary with this cache schema.
        // Read BEFORE `init()`: a cache from other code must be rebuilt in full, and
        // a new cache has nothing to reuse either way.
        let schema_ok = self.cache.check_schema_hash().unwrap_or(false);
        let extraction_ok = self.cache.check_extraction_hash().unwrap_or(false);

        // Ensure cache is initialized
        self.cache.init()?;

        // Extraction code (parsers, resolution) changed since this cache was last
        // written: stored symbols are from other code, and every file's imports
        // are extracted again below.
        if !schema_ok || !extraction_ok {
            match self.cache.clear_symbol_cache() {
                Ok(n) if n > 0 => {
                    log::info!("Cleared {} symbol cache rows written by other code", n)
                }
                Ok(_) => {}
                Err(e) => log::warn!("Failed to clear the symbol cache: {}", e),
            }
        }

        // Files modified at or after this run began record an unknown mtime (0), so
        // a write racing the read is caught by hash on the next check. The threshold
        // is the mtime of a file written now, on the same (coarse) clock that stamps
        // the files: a write in the same tick as a read could otherwise carry an
        // mtime just below a process-clock `run_start` and be trusted forever.
        let run_start = self.run_marker(&cache_dir);

        // Check available disk space after cache is initialized
        self.check_disk_space(root)?;

        // What meta.db holds: one row per path of the last indexed tree.
        let meta_path = cache_dir.join(crate::cache::META_DB);
        let (stored, stored_digest, meta_generation) = {
            let conn = crate::cache::open_meta_db(&meta_path)?;
            (
                crate::meta_update::load_stored_files(&conn)?,
                crate::meta_update::get_statistic(&conn, RESOLVER_DIGEST_KEY)?,
                crate::meta_update::get_statistic(&conn, INDEX_GENERATION_KEY)?
                    .and_then(|g| g.parse::<u64>().ok()),
            )
        };
        // The snapshot currently published, and the generation the next publish takes.
        let prev_manifest = crate::snapshot::read_manifest(&cache_dir).ok().flatten();
        let generation = prev_manifest
            .as_ref()
            .map(|m| m.generation)
            .max(meta_generation)
            .unwrap_or(0)
            + 1;

        // Step 1: walk the tree. In parallel, ask git for the branch and the dirty
        // paths, and find the resolver configs (each is its own walk or subprocess).
        let phase_start = Instant::now();
        let (git_state, resolver_configs, discovered) = std::thread::scope(|scope| {
            let git = scope.spawn(|| crate::git::get_git_state_optional(root));
            let configs =
                scope.spawn(|| crate::dependency_resolve::ResolverConfigs::discover(root));
            let discovered = self.discover_files(root, &stored);
            (git.join(), configs.join(), discovered)
        });
        let git_state = git_state.map_err(|_| anyhow::anyhow!("git state thread panicked"))??;
        let resolver_configs =
            resolver_configs.map_err(|_| anyhow::anyhow!("resolver config walk panicked"))?;
        let Discovered {
            files,
            rels,
            sizes,
            metas,
            skipped_too_large,
            skipped_bytes_too_large,
            skipped_binary,
        } = discovered?;
        let total_files = files.len();
        log::info!(
            "Discovered {} files to index ({} skipped: too large, {} binary) in {} ms",
            total_files,
            skipped_too_large,
            skipped_binary,
            phase_start.elapsed().as_millis()
        );

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
        let commit = git_state.as_ref().map(|s| s.commit.clone());
        let dirty_paths: std::collections::HashSet<String> = git_state
            .as_ref()
            .map(|s| s.dirty_paths.clone())
            .unwrap_or_default();

        // The branch's own hashes: the basis of the "new / modified / unchanged"
        // breakdown, as before stable ids.
        let existing_hashes = self.cache.load_hashes_for_branch(&branch)?;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .context("Failed to create thread pool")?;

        // Step 2: what changed. A file whose (size, mtime) match its row is
        // unchanged without being read; the others are read and hashed.
        let classify_start = Instant::now();
        let to_hash: Vec<usize> = (0..total_files)
            .filter(|&i| match stored.get(&rels[i]) {
                Some(row) => !metas[i].as_ref().is_some_and(|md| row.stat_matches(md)),
                None => false,
            })
            .collect();
        let hashed: Vec<(usize, Option<String>)> = pool.install(|| {
            to_hash
                .par_iter()
                .map(|&i| {
                    (
                        i,
                        std::fs::read(&files[i]).ok().map(|b| self.hash_content(&b)),
                    )
                })
                .collect()
        });
        let mut status: Vec<FileStatus> = rels
            .iter()
            .map(|rel| {
                if stored.contains_key(rel) {
                    FileStatus::Unchanged
                } else {
                    FileStatus::Added
                }
            })
            .collect();
        for (i, hash) in hashed {
            status[i] = match hash {
                Some(h) if h == stored[&rels[i]].hash => FileStatus::Touched,
                _ => FileStatus::Modified,
            };
        }
        let discovered_set: std::collections::HashSet<&str> =
            rels.iter().map(String::as_str).collect();
        let gone: Vec<&String> = stored
            .keys()
            .filter(|p| !discovered_set.contains(p.as_str()))
            .collect();
        // Counted as "deleted" only when the file is gone from disk (a file that is
        // merely no longer indexable was never counted), as before.
        let deleted_file_count = gone.iter().filter(|p| !root.join(p).exists()).count();
        let added = status.iter().filter(|s| **s == FileStatus::Added).count();
        let modified = status
            .iter()
            .filter(|s| **s == FileStatus::Modified)
            .count();
        log::info!(
            "phase classify: {} added, {} modified, {} touched, {} gone; {} files hashed in {} ms",
            added,
            modified,
            status.iter().filter(|s| **s == FileStatus::Touched).count(),
            gone.len(),
            to_hash.len(),
            classify_start.elapsed().as_millis()
        );

        // Every dependency row is rewritten when the extraction code, the schema or
        // any resolver config (go.mod, tsconfig.json, ...) changed.
        let full_deps = !schema_ok
            || !extraction_ok
            || stored_digest.as_deref() != Some(resolver_configs.digest.as_str());
        let content_changed = added > 0 || modified > 0 || !gone.is_empty();
        let stores_ok = schema_ok && self.stores_intact(stored.len(), meta_generation);

        if !content_changed && !full_deps && stores_ok {
            log::info!("No files changed - skipping index rebuild");
            return self.refresh_unchanged(RefreshUnchanged {
                rels: &rels,
                metas: &metas,
                status: &status,
                stored: &stored,
                existing_hashes: &existing_hashes,
                dirty_paths: &dirty_paths,
                branch: &branch,
                commit: commit.as_deref(),
                git_dirty: git_state.as_ref().map(|s| s.dirty),
                run_start,
                skipped: (skipped_too_large, skipped_bytes_too_large, skipped_binary),
            });
        }
        if content_changed && !full_deps && stores_ok {
            let update = DeltaUpdate {
                root,
                files: &files,
                rels: &rels,
                metas: &metas,
                status: &status,
                stored: &stored,
                existing_hashes: &existing_hashes,
                dirty_paths: &dirty_paths,
                branch: &branch,
                commit: commit.as_deref(),
                git_dirty: git_state.as_ref().map(|s| s.dirty).unwrap_or(false),
                resolver_configs: &resolver_configs,
                run_start,
                generation,
                pool: &pool,
                deleted_file_count,
                skipped: (skipped_too_large, skipped_bytes_too_large, skipped_binary),
            };
            match self.try_delta_update(update)? {
                Some(stats) => return Ok(stats),
                None => log::info!("The delta would pass its limits - building a new base"),
            }
        }
        if full_deps {
            log::info!(
                "Resolver configs, extraction code or schema changed - re-extracting every file's imports"
            );
        }

        // Step 3: Build trigram index + content store from every file (the stores
        // are rewritten in full). Imports are extracted only where they may have
        // changed.
        let mut files_indexed = 0;
        let mut indexed: Vec<IndexedFile> = Vec::with_capacity(total_files);

        // Initialize trigram builder and content store. The builder spills each
        // batch to `<cache>/trigram_temp/` only when there is more than one batch;
        // a single batch stays in memory and the directory is never created.
        let mut trigram_builder = TrigramIndexBuilder::new(self.cache.path().join("trigram_temp"));
        let mut content_writer = ContentWriter::new();

        // Initialize content writer to start streaming writes immediately
        let (content_name, trigrams_name, plan_name) = crate::snapshot::base_file_names(generation);
        let content_path = cache_dir.join(&content_name);
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
            let batch_start = batch_range.start;
            let batch_len = batch_range.len();
            log::info!(
                "Processing batch {}/{} ({} files)",
                batch_idx + 1,
                num_batches,
                batch_len
            );
            let pool_start = Instant::now();

            // Process files in parallel using rayon with custom thread pool.
            // `map_init` gives each worker one reusable trigram sort buffer.
            let counter_clone = Arc::clone(&progress_counter);
            let tsconfigs = &resolver_configs.tsconfigs;
            let ctx = ProcessCtx {
                root,
                files: &files,
                rels: &rels,
                stored: &stored,
                run_start,
                full_deps,
                tsconfigs,
            };
            let results: Vec<Option<FileProcessingResult>> = pool.install(|| {
                batch_range
                    .into_par_iter()
                    .map_init(Vec::<u64>::new, |trigram_scratch, i| {
                        let result = self.process_file(&ctx, i, trigram_scratch);
                        counter_clone.fetch_add(1, Ordering::Relaxed);
                        result
                    })
                    .collect()
            });
            pool_ms += pool_start.elapsed().as_millis();

            // Process batch results immediately (streaming approach to minimize memory)
            for (offset, result) in results.into_iter().enumerate() {
                let Some(result) = result else { continue };
                let i = batch_start + offset;
                // The normalized (forward-slash, relative) path everywhere, so the
                // trigram index and content store agree with the database and the
                // downstream filters, regardless of host separator.
                let normalized_pathbuf = PathBuf::from(&rels[i]);

                // Register the file with the trigram builder (assigns file_id in
                // discovery order) and hand it the postings extracted in the pool.
                let _file_id =
                    trigram_builder.add_file(normalized_pathbuf.clone(), result.trigram_run);

                // Add to content store
                content_writer.add_file(normalized_pathbuf, &result.content);

                files_indexed += 1;
                indexed.push(IndexedFile {
                    index: i,
                    hash: result.hash,
                    language: result.language,
                    line_count: result.line_count,
                    size: result.size,
                    mtime_ns: result.mtime_ns,
                    imports: result.imports,
                });
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

        // Breakdown against the branch's own hashes (unchanged since stable ids).
        let (new_file_count, modified_file_count, unchanged_file_count) = breakdown(
            indexed
                .iter()
                .map(|f| (rels[f.index].as_str(), f.hash.as_str())),
            &existing_hashes,
        );

        log::info!("Indexed {} files", files_indexed);

        // Step 4: Write trigram index.
        // Crash-safe write: the builder streams into `trigrams.<g>.bin.tmp`, syncs,
        // then renames it into place (see `atomic_write`). Nothing names the new
        // files until the manifest below does, so a crash leaves the previous
        // snapshot as it was.
        *progress_status.lock().unwrap() = "Writing trigram index...".to_string();
        if show_progress {
            pb.set_message("Writing trigram index...".to_string());
        }
        let trigrams_path = cache_dir.join(&trigrams_name);
        let write_start = Instant::now();
        trigram_builder
            .write_with_plan(&pool, &trigrams_path, Some(&cache_dir.join(&plan_name)))
            .context("Failed to write trigram index")?;
        log::info!(
            "phase trigram write: {} trigrams, {} files, {} ms",
            trigram_builder.trigram_count(),
            trigram_builder.file_count(),
            write_start.elapsed().as_millis()
        );

        // Step 5: Finalize content store (already been writing incrementally)
        *progress_status.lock().unwrap() = "Finalizing content store...".to_string();
        if show_progress {
            pb.set_message("Finalizing content store...".to_string());
        }
        content_writer
            .finalize_if_needed()
            .context("Failed to finalize content store")?;
        log::info!(
            "Wrote {} files ({} bytes) to {}",
            content_writer.file_count(),
            content_writer.content_size(),
            content_name
        );

        // Step 6: Publish. The manifest rename is the commit point of the stores;
        // meta.db follows in one transaction that records the same generation.
        // A crash between the two leaves a manifest ahead of meta.db: the changed
        // files read as stale (their rows are older than the stores), and the next
        // run sees the generations differ and rebuilds.
        let manifest = crate::snapshot::Manifest::for_base(
            generation,
            crate::snapshot::SegmentFiles {
                content: content_name.clone(),
                trigrams: trigrams_name.clone(),
                plan: Some(plan_name.clone()),
                files: content_writer.file_count() as u64,
                content_bytes: std::fs::metadata(&content_path)?.len(),
                trigrams_bytes: std::fs::metadata(&trigrams_path)?.len(),
            },
            trigram_builder.trigram_count() as u64,
            content_writer.content_size() as u64,
        );
        crate::snapshot::write_manifest(&cache_dir, &manifest)?;
        crate::snapshot::link_fixed_names(&cache_dir, &manifest);

        let meta_start = Instant::now();
        let present: Vec<usize> = indexed.iter().map(|f| f.index).collect();
        self.write_meta(MetaWrite {
            root,
            rels: &rels,
            present: &present,
            written: &indexed,
            metas: &metas,
            run_start,
            stored: &stored,
            status: &status,
            dirty_paths: &dirty_paths,
            branch: &branch,
            commit: commit.as_deref(),
            resolver_configs: &resolver_configs,
            full_deps,
            generation,
        })?;

        // Update branch metadata
        self.cache.update_branch_metadata(
            &branch,
            commit.as_deref(),
            indexed.len(),
            git_state.as_ref().map(|s| s.dirty).unwrap_or(false),
        )?;

        // Force WAL checkpoint to ensure background processes see all committed data
        // This is critical when spawning background symbol indexer immediately after
        self.cache
            .checkpoint_wal()
            .context("Failed to checkpoint WAL")?;
        log::info!(
            "phase meta.db (files, branches, dependencies, exports): {} ms",
            meta_start.elapsed().as_millis()
        );

        // Readers in this process reopen; files no manifest names any more go.
        crate::query::invalidate_caches(root);
        crate::snapshot::remove_unreferenced(&cache_dir, &manifest, prev_manifest.as_ref());

        // Step 7: Update SQLite statistics from database totals (branch-aware)
        *progress_status.lock().unwrap() = "Updating statistics...".to_string();
        if show_progress {
            pb.set_message("Updating statistics...".to_string());
        }
        // Update stats for current branch only
        self.cache.update_stats(&branch)?;

        // Update schema hash to mark cache as compatible with current binary
        self.cache.update_schema_hash()?;
        self.cache.update_extraction_hash()?;

        pb.finish_with_message("Indexing complete");

        // Return stats with incremental breakdown
        let mut stats = self.cache.stats_on_branch(Some(branch))?;
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

    /// Apply this run's changes as a delta on the published base: the added and
    /// modified files are read, the delta is rebuilt from them and the unchanged
    /// files of the previous delta, and the base files they supersede (or that
    /// are gone) are tombstoned. `Ok(None)` when the delta would pass its limits:
    /// the caller then builds a new base (a merge).
    fn try_delta_update(&self, u: DeltaUpdate<'_>) -> Result<Option<IndexStats>> {
        use crate::snapshot::{IndexSnapshot, Manifest, SegmentFiles};
        let cache_dir = self.cache.path().to_path_buf();
        let snapshot = IndexSnapshot::open(&cache_dir)?;
        let Some(prev) = snapshot.manifest().cloned() else {
            return Ok(None);
        };
        let base_len = snapshot.base_len();
        let delta_start = Instant::now();

        // Live base files and the previous delta's files, by path.
        let mut base_ids: HashMap<&str, u32> = HashMap::new();
        for id in 0..base_len {
            if let Some(p) = snapshot.get_file_path(id).and_then(|p| p.to_str()) {
                base_ids.insert(p, id);
            }
        }
        let mut old_delta: HashMap<&str, u32> = HashMap::new();
        for id in base_len..snapshot.id_bound() {
            if let Some(p) = snapshot.get_file_path(id).and_then(|p| p.to_str()) {
                old_delta.insert(p, id);
            }
        }

        // Read the added and modified files.
        let changed: Vec<usize> = (0..u.rels.len())
            .filter(|&i| matches!(u.status[i], FileStatus::Added | FileStatus::Modified))
            .collect();
        let ctx = ProcessCtx {
            root: u.root,
            files: u.files,
            rels: u.rels,
            stored: u.stored,
            run_start: u.run_start,
            full_deps: false,
            tsconfigs: &u.resolver_configs.tsconfigs,
        };
        let read: Vec<(usize, Option<FileProcessingResult>)> = u.pool.install(|| {
            changed
                .par_iter()
                .map_init(Vec::<u64>::new, |scratch, &i| {
                    (i, self.process_file(&ctx, i, scratch))
                })
                .collect()
        });
        let mut fresh: HashMap<usize, FileProcessingResult> = HashMap::new();
        let mut unreadable: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for (i, result) in read {
            match result {
                Some(r) => {
                    fresh.insert(i, r);
                }
                None => {
                    unreadable.insert(i);
                }
            }
        }

        // The files of the new snapshot, in walk order (an unreadable file is gone).
        let present: Vec<usize> = (0..u.rels.len())
            .filter(|i| !unreadable.contains(i))
            .collect();
        let present_paths: HashMap<&str, usize> =
            present.iter().map(|&i| (u.rels[i].as_str(), i)).collect();

        // The new delta: read files, and the previous delta's files that are
        // still present and unchanged, in walk order.
        enum Source {
            Fresh(usize),
            Old(u32),
        }
        let mut entries: Vec<(usize, Source)> = Vec::new();
        for &i in &present {
            if fresh.contains_key(&i) {
                entries.push((i, Source::Fresh(i)));
            } else if let Some(&id) = old_delta.get(u.rels[i].as_str()) {
                entries.push((i, Source::Old(id)));
            }
        }
        let entry_len = |src: &Source| -> Result<u64> {
            Ok(match src {
                Source::Fresh(i) => fresh[i].content.len() as u64,
                Source::Old(id) => snapshot.get_file_content(*id)?.len() as u64,
            })
        };
        let mut delta_bytes = 0u64;
        for (_, src) in &entries {
            delta_bytes += entry_len(src)?;
        }
        let (max_files, max_bytes) = self.merge_limits(prev.live_corpus_bytes);
        if entries.len() > max_files || delta_bytes > max_bytes {
            log::info!(
                "Delta of {} files / {} bytes passes its limits ({} files / {} bytes)",
                entries.len(),
                delta_bytes,
                max_files,
                max_bytes
            );
            return Ok(None);
        }

        // Base files superseded by a read file, or gone from the tree.
        let mut new_dead: Vec<u32> = base_ids
            .iter()
            .filter(|(path, _)| match present_paths.get(*path) {
                Some(i) => fresh.contains_key(i),
                None => true,
            })
            .map(|(_, &id)| id)
            .collect();
        new_dead.sort_unstable();
        let mut tombstones: Vec<u32> = prev.tombstones.clone();
        tombstones.extend(&new_dead);
        tombstones.sort_unstable();
        tombstones.dedup();

        // Their planning sizes, added to the previous tombstones'.
        let mut tomb: std::collections::BTreeMap<crate::trigram::Trigram, u64> = snapshot
            .tomb()
            .map(|t| t.entries().map(|(t, n)| (t, n as u64)).collect())
            .unwrap_or_default();
        let dead_sizes: Vec<Vec<(crate::trigram::Trigram, u32)>> = u.pool.install(|| {
            new_dead
                .par_iter()
                .map_init(Vec::<u64>::new, |scratch, &id| {
                    let content = snapshot.get_file_content(id).unwrap_or("");
                    let run = crate::trigram_build::extract_trigram_run(content, scratch);
                    crate::trigram_build::run_plan_sizes(&run)
                })
                .collect()
        });
        for sizes in dead_sizes {
            for (t, n) in sizes {
                *tomb.entry(t).or_default() += n as u64;
            }
        }

        // Write the delta stores (generation files: nothing names them yet).
        let (delta_content, delta_trigrams, delta_plan, tomb_name) =
            crate::snapshot::delta_file_names(u.generation);
        let delta = if entries.is_empty() {
            None
        } else {
            let runs: Vec<Option<TrigramRun>> = u.pool.install(|| {
                entries
                    .par_iter()
                    .map_init(Vec::<u64>::new, |scratch, (_, src)| match src {
                        Source::Fresh(_) => None,
                        Source::Old(id) => Some(crate::trigram_build::extract_trigram_run(
                            snapshot.get_file_content(*id).unwrap_or(""),
                            scratch,
                        )),
                    })
                    .collect()
            });
            let mut builder = TrigramIndexBuilder::new(cache_dir.join("trigram_temp"));
            let mut writer = ContentWriter::new();
            let content_path = cache_dir.join(&delta_content);
            writer
                .init(content_path.clone())
                .context("Failed to initialize the delta content store")?;
            for ((i, src), run) in entries.iter().zip(runs) {
                let path = PathBuf::from(&u.rels[*i]);
                match src {
                    Source::Fresh(k) => {
                        let f = fresh.get_mut(k).expect("read file");
                        let run = std::mem::take(&mut f.trigram_run);
                        builder.add_file(path.clone(), run);
                        writer.add_file(path, &f.content);
                    }
                    Source::Old(id) => {
                        builder.add_file(path.clone(), run.expect("old delta run"));
                        writer.add_file(path, snapshot.get_file_content(*id)?);
                    }
                }
            }
            let trigrams_path = cache_dir.join(&delta_trigrams);
            builder
                .write_with_plan(u.pool, &trigrams_path, Some(&cache_dir.join(&delta_plan)))
                .context("Failed to write the delta trigram index")?;
            writer
                .finalize_if_needed()
                .context("Failed to finalize the delta content store")?;
            Some(SegmentFiles {
                content: delta_content.clone(),
                trigrams: delta_trigrams.clone(),
                plan: Some(delta_plan.clone()),
                files: writer.file_count() as u64,
                content_bytes: std::fs::metadata(&content_path)?.len(),
                trigrams_bytes: std::fs::metadata(&trigrams_path)?.len(),
            })
        };
        let tomb_entries: Vec<(crate::trigram::Trigram, u32)> = tomb
            .iter()
            .map(|(&t, &n)| (t, n.min(u32::MAX as u64) as u32))
            .collect();
        let tomb_file = if tombstones.is_empty() {
            None
        } else {
            crate::snapshot::write_tomb_file(&cache_dir.join(&tomb_name), &tomb_entries)?;
            Some(tomb_name)
        };

        // What a full build of this tree would record: its trigrams and its text.
        let base_plan = |t: crate::trigram::Trigram| -> u64 {
            snapshot
                .base()
                .trigrams
                .list_part(t)
                .map_or(0, |(_, plan)| plan)
        };
        let dead_in_base: std::collections::HashSet<crate::trigram::Trigram> = tomb
            .iter()
            .filter(|(t, n)| base_plan(**t) <= **n)
            .map(|(t, _)| *t)
            .collect();
        let mut revived = 0u64;
        if delta.is_some() {
            let index = crate::trigram::TrigramIndex::load(cache_dir.join(&delta_trigrams))?;
            for t in index.trigrams() {
                let base_live = base_plan(t) > 0 && !dead_in_base.contains(&t);
                if !base_live {
                    revived += 1;
                }
            }
        }
        let live_trigrams =
            snapshot.base().trigrams.trigram_count() as u64 - dead_in_base.len() as u64 + revived;
        let dead_set: std::collections::HashSet<u32> = tombstones.iter().copied().collect();
        // Lengths come from the entry table: reading the content would page in (and
        // UTF-8 check) the whole base on every update.
        let mut live_corpus = delta_bytes;
        for id in 0..base_len {
            if !dead_set.contains(&id) {
                live_corpus += snapshot.base().content.file_len(id).unwrap_or(0);
            }
        }

        // Publish: fixed names first (an older binary must not read the base alone
        // once it is incomplete), then the manifest, then meta.db.
        let manifest = Manifest::new(
            u.generation,
            prev.base.clone(),
            delta,
            tombstones,
            tomb_file,
            live_trigrams,
            live_corpus,
        );
        if !manifest.base_only() {
            crate::snapshot::unlink_fixed_names(&cache_dir);
        }
        crate::snapshot::write_manifest(&cache_dir, &manifest)?;
        crate::snapshot::link_fixed_names(&cache_dir, &manifest);
        log::info!(
            "phase delta: {} files ({} read), {} tombstones, {} ms",
            manifest.delta.as_ref().map_or(0, |d| d.files),
            fresh.len(),
            manifest.tombstones.len(),
            delta_start.elapsed().as_millis()
        );

        let written: Vec<IndexedFile> = present
            .iter()
            .filter_map(|i| {
                fresh.remove(i).map(|r| IndexedFile {
                    index: *i,
                    hash: r.hash,
                    language: r.language,
                    line_count: r.line_count,
                    size: r.size,
                    mtime_ns: r.mtime_ns,
                    imports: r.imports,
                })
            })
            .collect();
        let written_hash: HashMap<usize, &str> =
            written.iter().map(|f| (f.index, f.hash.as_str())).collect();
        let hashes: Vec<(&str, &str)> = present
            .iter()
            .map(|&i| {
                let rel = u.rels[i].as_str();
                let hash = written_hash
                    .get(&i)
                    .copied()
                    .or_else(|| u.stored.get(rel).map(|s| s.hash.as_str()))
                    .unwrap_or("");
                (rel, hash)
            })
            .collect();
        let (new_files, modified_files, unchanged_files) =
            breakdown(hashes.into_iter(), u.existing_hashes);

        let meta_start = Instant::now();
        self.write_meta(MetaWrite {
            root: u.root,
            rels: u.rels,
            present: &present,
            written: &written,
            metas: u.metas,
            run_start: u.run_start,
            stored: u.stored,
            status: u.status,
            dirty_paths: u.dirty_paths,
            branch: u.branch,
            commit: u.commit,
            resolver_configs: u.resolver_configs,
            full_deps: false,
            generation: u.generation,
        })?;
        self.cache
            .update_branch_metadata(u.branch, u.commit, present.len(), u.git_dirty)?;
        self.cache
            .checkpoint_wal()
            .context("Failed to checkpoint WAL")?;
        log::info!(
            "phase meta.db (files, branches, dependencies, exports): {} ms",
            meta_start.elapsed().as_millis()
        );

        crate::query::invalidate_caches(u.root);
        drop(snapshot);
        crate::snapshot::remove_unreferenced(&cache_dir, &manifest, Some(&prev));

        self.cache.update_stats(u.branch)?;
        self.cache.update_schema_hash()?;
        self.cache.update_extraction_hash()?;

        let mut stats = self.cache.stats_on_branch(Some(u.branch.to_string()))?;
        stats.new_files = new_files;
        stats.modified_files = modified_files;
        stats.deleted_files = u.deleted_file_count;
        stats.unchanged_files = unchanged_files;
        stats.skipped_too_large = u.skipped.0;
        stats.skipped_bytes_too_large = u.skipped.1;
        stats.skipped_binary = u.skipped.2;
        Ok(Some(stats))
    }

    /// Read one discovered file: stat, bytes, hash, text, trigram run, and imports
    /// when they may have changed. `None` when it cannot be read.
    fn process_file(
        &self,
        ctx: &ProcessCtx<'_>,
        i: usize,
        trigram_scratch: &mut Vec<u64>,
    ) -> Option<FileProcessingResult> {
        let file_path = &ctx.files[i];
        let path_str = file_path.to_string_lossy().to_string();

        // Stat BEFORE the read. If the file changes between the two, the recorded
        // (size, mtime) is older than the bytes, so the next check re-hashes it
        // rather than trusting a stat that matches.
        let (size, mtime_ns) = std::fs::metadata(file_path)
            .map(|md| {
                (
                    md.len(),
                    crate::cache::recorded_mtime_ns(&md, ctx.run_start),
                )
            })
            .unwrap_or((0, 0));

        // Read file content once (used for hashing, trigrams, and parsing). The
        // hash is of the RAW bytes, so the freshness check can hash a file on disk
        // and compare. Invalid UTF-8 (a Latin-1 `.po`, an old doc) is decoded
        // lossily rather than dropped: ripgrep searches those bytes, and an agent
        // expects the same.
        let bytes = match std::fs::read(file_path) {
            Ok(b) => b,
            Err(e) => {
                log::warn!("Failed to read {}: {}", path_str, e);
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

        // Trigram postings, sorted, without a file id (assigned serially).
        let trigram_run = crate::trigram_build::extract_trigram_run(&content, trigram_scratch);

        // Imports and re-exports: for every file when they are all rewritten, else
        // for files whose bytes changed (including a file that changed again since
        // it was classified).
        let unchanged = ctx
            .stored
            .get(&ctx.rels[i])
            .is_some_and(|row| row.hash == hash);
        let imports = (ctx.full_deps || !unchanged)
            .then(|| extract_imports(language, &content, &path_str, ctx.root, ctx.tsconfigs));

        Some(FileProcessingResult {
            hash,
            content,
            language,
            line_count,
            size,
            mtime_ns,
            imports,
            trigram_run,
        })
    }

    /// Write the marker whose mtime is this run's race threshold; returns it (or the
    /// process clock when the marker cannot be written or read).
    fn run_marker(&self, cache_dir: &Path) -> std::time::SystemTime {
        let marker = cache_dir.join(RUN_MARKER);
        std::fs::write(&marker, b"")
            .and_then(|_| std::fs::metadata(&marker))
            .and_then(|md| md.modified())
            .unwrap_or_else(|_| std::time::SystemTime::now())
    }

    /// Whether the published snapshot is complete, holds `expected` files (the rows
    /// of the last indexed tree) and is the one meta.db's rows go with.
    fn stores_intact(&self, expected: usize, meta_generation: Option<u64>) -> bool {
        match crate::snapshot::IndexSnapshot::open(self.cache.path()) {
            Ok(snapshot) => {
                if snapshot.generation().is_none() {
                    log::info!("Index written before the manifest - rebuilding");
                    false
                } else if snapshot.generation() != meta_generation {
                    log::warn!(
                        "Manifest generation {:?} but meta.db rows are generation {:?} (a run stopped between the two) - rebuilding",
                        snapshot.generation(),
                        meta_generation
                    );
                    false
                } else if snapshot.live_file_count() != expected {
                    log::warn!(
                        "Stores hold {} files but meta.db lists {} - rebuilding",
                        snapshot.live_file_count(),
                        expected
                    );
                    false
                } else {
                    true
                }
            }
            Err(e) => {
                log::warn!("Index stores unusable ({:#}) - rebuilding", e);
                false
            }
        }
    }

    /// Nothing to write to the stores: bring meta.db up to date (fingerprints of
    /// touched files, dirty flags, walk positions, this branch's rows, the branch
    /// metadata and the statistics timestamp) and report.
    fn refresh_unchanged(&self, r: RefreshUnchanged<'_>) -> Result<IndexStats> {
        let now = chrono::Utc::now().timestamp();
        let seqs = crate::meta_update::plan_walk_seq(
            &r.rels
                .iter()
                .map(|rel| r.stored.get(rel).map(|s| s.walk_seq))
                .collect::<Vec<_>>(),
        );
        let mut walk = Vec::new();
        let mut flips = Vec::new();
        let mut touched = Vec::new();
        for (i, rel) in r.rels.iter().enumerate() {
            let row = &r.stored[rel];
            let dirty = r.dirty_paths.contains(rel);
            if seqs[i] != row.walk_seq {
                walk.push((row.id, seqs[i]));
            }
            if r.status[i] == FileStatus::Touched {
                let (size, mtime) = r.metas[i]
                    .as_ref()
                    .map(|md| (md.len(), crate::cache::recorded_mtime_ns(md, r.run_start)))
                    .unwrap_or((0, 0));
                touched.push((row.id, size, mtime, dirty));
            } else if dirty != row.dirty {
                flips.push((row.id, dirty));
            }
        }

        let mut conn = crate::cache::open_meta_db(self.cache.path().join(crate::cache::META_DB))?;
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .context("Failed to begin meta.db transaction")?;
        crate::meta_update::set_walk_seqs(&tx, &walk)?;
        crate::meta_update::refresh_stats(&tx, &touched)?;
        crate::meta_update::set_dirty_flags(&tx, &flips)?;
        let branch_id = self
            .cache
            .get_or_create_branch_id(&tx, r.branch, r.commit)?;
        crate::meta_update::sync_branch_rows(&tx, branch_id, now)?;
        tx.commit()?;

        // The CONTENT is current, but the recorded commit may not be: committing
        // already-indexed files moves HEAD without changing a single hash.
        // Skipping the metadata update left `commit_sha` behind forever, so
        // freshness reported `stale` on a perfectly current index until some
        // unrelated edit happened to force a rebuild.
        if let Some(git_dirty) = r.git_dirty
            && let Err(e) =
                self.cache
                    .update_branch_metadata(r.branch, r.commit, r.rels.len(), git_dirty)
        {
            log::warn!("Failed to refresh branch metadata: {}", e);
        }
        self.cache.update_stats(r.branch)?;

        let (new_files, modified_files, unchanged_files) = breakdown(
            r.rels
                .iter()
                .map(|rel| (rel.as_str(), r.stored[rel].hash.as_str())),
            r.existing_hashes,
        );
        let mut stats = self.cache.stats_on_branch(Some(r.branch.to_string()))?;
        stats.new_files = new_files;
        stats.modified_files = modified_files;
        stats.unchanged_files = unchanged_files;
        stats.deleted_files = 0;
        stats.skipped_too_large = r.skipped.0;
        stats.skipped_bytes_too_large = r.skipped.1;
        stats.skipped_binary = r.skipped.2;
        Ok(stats)
    }

    /// One meta.db transaction for a run that rewrote the stores: the rows of
    /// files whose bytes, fingerprint, dirty flag or walk position changed, the
    /// deletions, this branch's rows, and the dependency and export rows (every
    /// file's when `full_deps`, else the changed files', plus re-resolution of the
    /// rest when files were added or removed).
    fn write_meta(&self, w: MetaWrite<'_>) -> Result<()> {
        let now = chrono::Utc::now().timestamp();
        let seqs = crate::meta_update::plan_walk_seq(
            &w.present
                .iter()
                .map(|&i| w.stored.get(&w.rels[i]).map(|s| s.walk_seq))
                .collect::<Vec<_>>(),
        );
        let written: HashMap<usize, &IndexedFile> =
            w.written.iter().map(|f| (f.index, f)).collect();

        let mut rows: Vec<crate::cache::FileRow> = Vec::new();
        let mut row_files: Vec<usize> = Vec::new(); // discovery index of each row
        let mut walk = Vec::new();
        let mut flips = Vec::new();
        let mut touched = Vec::new();
        for (k, &i) in w.present.iter().enumerate() {
            let rel = &w.rels[i];
            let dirty = w.dirty_paths.contains(rel);
            let stored = w.stored.get(rel);
            match written.get(&i) {
                Some(f)
                    if !(w.status[i] == FileStatus::Unchanged
                        && stored.is_some_and(|row| row.hash == f.hash)) =>
                {
                    rows.push(crate::cache::FileRow {
                        path: rel.clone(),
                        hash: f.hash.clone(),
                        language: format!("{:?}", f.language),
                        line_count: f.line_count,
                        size: f.size,
                        mtime_ns: f.mtime_ns,
                        dirty,
                        walk_seq: seqs[k],
                    });
                    row_files.push(i);
                }
                _ => {
                    // Not read this run (or read with the bytes and fingerprint
                    // stored): only the walk position, the fingerprint of a touched
                    // file and the dirty flag can move.
                    let Some(row) = stored else { continue };
                    if seqs[k] != row.walk_seq {
                        walk.push((row.id, seqs[k]));
                    }
                    if w.status[i] == FileStatus::Touched && !written.contains_key(&i) {
                        let (size, mtime) = w.metas[i]
                            .as_ref()
                            .map(|md| (md.len(), crate::cache::recorded_mtime_ns(md, w.run_start)))
                            .unwrap_or((0, 0));
                        touched.push((row.id, size, mtime, dirty));
                    } else if dirty != row.dirty {
                        flips.push((row.id, dirty));
                    }
                }
            }
        }
        let present: std::collections::HashSet<&str> =
            w.present.iter().map(|&i| w.rels[i].as_str()).collect();
        let deleted: Vec<i64> = w
            .stored
            .iter()
            .filter(|(p, _)| !present.contains(p.as_str()))
            .map(|(_, row)| row.id)
            .collect();
        let paths_changed = !deleted.is_empty()
            || w.present
                .iter()
                .any(|&i| !w.stored.contains_key(&w.rels[i]));

        let mut conn = crate::cache::open_meta_db(self.cache.path().join(crate::cache::META_DB))?;
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .context("Failed to begin meta.db transaction")?;

        crate::meta_update::delete_files(&tx, &deleted)?;
        let new_ids = crate::meta_update::upsert_files(&tx, &rows, now)?;
        crate::meta_update::set_walk_seqs(&tx, &walk)?;
        crate::meta_update::refresh_stats(&tx, &touched)?;
        crate::meta_update::set_dirty_flags(&tx, &flips)?;
        let branch_id = self
            .cache
            .get_or_create_branch_id(&tx, w.branch, w.commit)?;
        crate::meta_update::sync_branch_rows(&tx, branch_id, now)?;
        log::info!(
            "meta.db: {} rows written, {} deleted, {} moved, {} touched, {} dirty flags",
            rows.len(),
            deleted.len(),
            walk.len(),
            touched.len(),
            flips.len()
        );

        // id of every file read this run, by discovery index.
        let mut ids: HashMap<usize, i64> = w
            .written
            .iter()
            .filter_map(|f| w.stored.get(&w.rels[f.index]).map(|s| (f.index, s.id)))
            .collect();
        for (i, id) in row_files.iter().zip(new_ids) {
            ids.insert(*i, id);
        }

        let resolver = crate::dependency::PathResolver::from_conn(&tx)
            .context("Failed to load file paths for dependency resolution")?;
        let ctx = crate::dependency_resolve::ResolverContext::new(w.root, w.resolver_configs);
        let mut writer = crate::dependency::DependencyWriter::new(&tx);
        if w.full_deps {
            writer.clear_all()?;
        }
        // In walk order: the row order a full build produces.
        let mut resolved_here: std::collections::HashSet<i64> = std::collections::HashSet::new();
        for f in w.written {
            let Some((imports, exports)) = &f.imports else {
                continue;
            };
            let rel = &w.rels[f.index];
            let file_id = ids[&f.index];
            resolved_here.insert(file_id);
            let deps = ctx.resolve_file_imports(file_id, rel, imports.clone(), &resolver);
            writer.replace_dependencies(file_id, &deps)?;
            if !w.full_deps {
                writer.clear_exports(file_id)?;
            }
            for export in exports {
                let resolved = ctx.resolve_export(rel, export, &resolver);
                writer.insert_export(
                    file_id,
                    export.exported_symbol.as_deref(),
                    &export.source_path,
                    resolved,
                    export.line_number,
                )?;
            }
        }
        let (deps_written, exports_written) = writer.counts();

        // An added or removed path can change how an unchanged file's imports
        // resolve (suffix matches, ambiguity, the first of several candidates).
        let mut reresolved = 0usize;
        if !w.full_deps && paths_changed {
            reresolved = reresolve(&tx, &ctx, &resolver, &resolved_here)?;
        }

        crate::meta_update::set_statistic(
            &tx,
            RESOLVER_DIGEST_KEY,
            &w.resolver_configs.digest,
            now,
        )?;
        crate::meta_update::set_statistic(
            &tx,
            INDEX_GENERATION_KEY,
            &w.generation.to_string(),
            now,
        )?;
        tx.commit()?;
        log::info!(
            "dependencies: {} rows, {} exports written; {} rows re-resolved",
            deps_written,
            exports_written,
            reresolved
        );
        Ok(())
    }

    /// Discover all indexable files in the directory tree.
    ///
    /// `stored` is meta.db's view of the last indexed tree: a non-code file whose
    /// stat matches its row was sniffed for NUL bytes when it was indexed and is
    /// not read again.
    fn discover_files(
        &self,
        root: &Path,
        stored: &HashMap<String, crate::meta_update::StoredFile>,
    ) -> Result<Discovered> {
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
            let metadata = std::fs::metadata(path).ok();
            if let Some(metadata) = &metadata {
                size = metadata.len();
                if size > self.config.max_file_size as u64 {
                    log::debug!("Skipping {} (too large: {} bytes)", path.display(), size);
                    out.skipped_too_large += 1;
                    out.skipped_bytes_too_large += size;
                    continue;
                }
            }

            let rel = normalize_rel(root, path);

            // A code extension is trusted to be text. Anything else in the tracked
            // tier (`image.png`, `OWNERS`, `data.bin`) is sniffed: ripgrep's rule, a
            // NUL byte anywhere means binary, and a binary file is never in the
            // index. Only the long tail pays the read (from the page cache, since
            // the main pass reads it again a moment later), and not a file that is
            // unchanged since it was indexed.
            let unchanged = || {
                stored
                    .get(&rel)
                    .zip(metadata.as_ref())
                    .is_some_and(|(row, md)| row.stat_matches(md))
            };
            if !lang.is_code() && !unchanged() && looks_binary(path) {
                log::debug!("Skipping {} (binary)", path.display());
                out.skipped_binary += 1;
                continue;
            }

            out.files.push(path.to_path_buf());
            out.rels.push(rel);
            out.sizes.push(size);
            out.metas.push(metadata);
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

        let files = indexer
            .discover_files(temp.path(), &HashMap::new())
            .unwrap()
            .files;
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

        let files = indexer
            .discover_files(temp.path(), &HashMap::new())
            .unwrap()
            .files;
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

        let found = indexer
            .discover_files(temp.path(), &HashMap::new())
            .unwrap();
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

        let files = indexer
            .discover_files(temp.path(), &HashMap::new())
            .unwrap()
            .files;
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

        let files = indexer
            .discover_files(temp.path(), &HashMap::new())
            .unwrap()
            .files;

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
