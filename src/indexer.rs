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

use crate::cache::{CacheManager, WALK_SEQ_GAP};
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
    /// Imports, re-exports and package memberships, when they were extracted
    /// this run.
    imports: Option<Extracted>,
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
    /// Imports, re-exports and package memberships, when they were extracted
    /// this run.
    imports: Option<Extracted>,
}

/// One added or modified file a delta update reads.
struct Rewrite {
    rel: String,
    /// Size from the walk's stat (the merge-limit check before any read).
    size: u64,
    walk_seq: i64,
}

/// The dirty flags a change set records: the rewritten paths that are dirty, the
/// touched rows `(id, size, mtime_ns, dirty)`, the rows whose flag flips, and the
/// branch's flag.
struct DirtyFlags {
    rewrites: std::collections::HashSet<String>,
    touched: Vec<(i64, u64, i64, bool)>,
    flips: Vec<(i64, bool)>,
    git_dirty: bool,
}

/// A delta update's change set (see `Indexer::publish_delta`): `rfx index` builds
/// it from a full walk, `update_paths` from the named paths. Rows and store
/// entries it does not name stay as they are.
struct DeltaChanges<'a> {
    root: &'a Path,
    /// Added and modified files, in walk order.
    rewrites: Vec<Rewrite>,
    /// Rows of files gone from the index: `(path, id)`.
    deleted: Vec<(String, i64)>,
    /// Rows whose walk position moved: `(id, walk_seq)`.
    walk_moves: Vec<(i64, i64)>,
    /// The dirty flags, asked for only when meta.db is written (a library update's
    /// `git status` runs while the stores are written).
    dirty: Box<dyn FnOnce() -> DirtyFlags + 'a>,
    /// The rows of the rewritten paths that have one.
    stored: HashMap<String, crate::meta_update::StoredFile>,
    /// Re-sync every branch row (the branch may not be the one last synced);
    /// otherwise only the rewritten rows are set.
    sync_all_branch_rows: bool,
    /// Return the statistics `rfx index` prints (per-language counts: a scan of
    /// every row); otherwise the counts and sizes only.
    full_stats: bool,
    /// Files in the index after the update (unreadable rewrites not subtracted).
    live_files: usize,
    /// (new, modified, unchanged) of the files not rewritten, against the branch's
    /// own hashes.
    breakdown_rest: (usize, usize, usize),
    /// The branch's own hashes of the rewritten paths.
    branch_hashes: HashMap<String, String>,
    branch: &'a str,
    commit: Option<&'a str>,
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
    rels: &'a [String],
    stored: &'a HashMap<String, crate::meta_update::StoredFile>,
    run_start: std::time::SystemTime,
    full_deps: bool,
    tsconfigs: &'a HashMap<PathBuf, crate::parsers::tsconfig::PathAliasMap>,
}

/// Inputs of `Indexer::refresh_unchanged`.
struct RefreshUnchanged<'a> {
    rels: &'a [String],
    metas: &'a [Option<crate::cache::FileStat>],
    status: &'a [FileStatus],
    /// Each walked path's row (every one has a row: nothing was added).
    rows: &'a [Option<&'a crate::meta_update::StoredFile>],
    existing_hashes: &'a BranchHashes<'a>,
    dirty_paths: &'a std::collections::HashSet<String>,
    branch: &'a str,
    /// The branch's rows already name every file (the last run synced them).
    in_sync: bool,
    commit: Option<&'a str>,
    /// `Some(dirty)` inside git.
    git_dirty: Option<bool>,
    run_start: std::time::SystemTime,
    /// (too large, bytes too large, binary)
    skipped: (usize, u64, usize),
}

/// Inputs of `Indexer::refresh_named`.
struct RefreshNamed<'a> {
    touched: &'a [(i64, u64, i64, bool)],
    flips: &'a [(i64, bool)],
    walk_moves: &'a [(i64, i64)],
    branch: &'a str,
    commit: Option<&'a str>,
    git_dirty: bool,
    live_files: usize,
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
    metas: &'a [Option<crate::cache::FileStat>],
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
/// The branch's own hash of each path: its `file_branches` rows, or, for the
/// branch the last run synced, the `files` rows themselves (the same values,
/// without loading them).
enum BranchHashes<'a> {
    Synced(&'a HashMap<String, crate::meta_update::StoredFile>),
    Loaded(HashMap<String, String>),
}

impl BranchHashes<'_> {
    fn get(&self, path: &str) -> Option<&str> {
        match self {
            Self::Synced(stored) => stored.get(path).map(|row| row.hash.as_str()),
            Self::Loaded(hashes) => hashes.get(path).map(String::as_str),
        }
    }
}

fn breakdown<'a>(
    files: impl Iterator<Item = (&'a str, &'a str)>,
    existing_hashes: &BranchHashes<'_>,
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

/// Set `files.vendored` from `rules` for the rows in `ids`, or for every row
/// (`None`: the rules may have changed). Returns how many rows changed.
fn refresh_vendored(
    tx: &rusqlite::Connection,
    rules: &crate::vendor::VendorRules,
    ids: Option<&[i64]>,
) -> Result<usize> {
    let mut changed: Vec<(i64, bool)> = Vec::new();
    {
        let mut visit = |id: i64, path: &str, old: bool| {
            let new = rules.is_vendored(path, crate::models::Language::from_path(Path::new(path)));
            if new != old {
                changed.push((id, new));
            }
        };
        match ids {
            None => {
                let mut stmt = tx.prepare("SELECT id, path, vendored FROM files")?;
                let rows = stmt.query_map([], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, bool>(2)?,
                    ))
                })?;
                for row in rows {
                    let (id, path, old) = row?;
                    visit(id, &path, old);
                }
            }
            Some(ids) => {
                let mut stmt = tx.prepare("SELECT path, vendored FROM files WHERE id = ?")?;
                for &id in ids {
                    let (path, old): (String, bool) =
                        stmt.query_row([id], |r| Ok((r.get(0)?, r.get(1)?)))?;
                    visit(id, &path, old);
                }
            }
        }
    }
    let mut stmt = tx.prepare("UPDATE files SET vendored = ? WHERE id = ?")?;
    for (id, vendored) in &changed {
        stmt.execute(rusqlite::params![vendored, id])?;
    }
    Ok(changed.len())
}

/// Resolve again every stored import (other than External/Stdlib) and every
/// export of the files not in `skip`, updating the rows whose target changed.
/// Package-keyed rows are skipped: their key depends on the import and the
/// module config only, and which files a package holds is `package_members`.
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
             WHERE d.import_type NOT IN ('external', 'stdlib')
               AND d.resolved_package IS NULL",
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

/// What import extraction yields for one file: its imports, its re-exports, and
/// the `(package key, member)` rows its content declares (`package_members`).
type Extracted = (Vec<ImportInfo>, Vec<ExportInfo>, Vec<(String, String)>);

/// Imports, re-exports and declared package memberships of one file, by
/// language. `path_str` is the path as walked (for the nearest tsconfig).
fn extract_imports(
    language: Language,
    content: &str,
    path_str: &str,
    root: &Path,
    tsconfigs: &HashMap<PathBuf, crate::parsers::tsconfig::PathAliasMap>,
) -> Extracted {
    // Extract dependencies and exports for supported languages
    let mut parsed_exports: Vec<ExportInfo> = Vec::new();
    let mut declared_namespaces: Vec<(String, String)> = Vec::new();
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
        Language::CSharp => {
            match CSharpDependencyExtractor::extract_dependencies_and_members(content) {
                Ok((deps, members)) => {
                    declared_namespaces = members;
                    deps
                }
                Err(e) => {
                    log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                    Vec::new()
                }
            }
        }
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
    let members = match language {
        Language::Java | Language::Kotlin => {
            crate::parsers::java::jvm_package_members(path_str, content)
        }
        Language::CSharp => declared_namespaces,
        _ => Vec::new(),
    };
    (dependencies, parsed_exports, members)
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
    /// `(max_files, max_bytes)` of the recent segment before it is folded into the
    /// delta; `None` = [`RECENT_MAX_FILES`] and the merge byte limit over
    /// [`RECENT_BYTES_DIVISOR`].
    recent_limits: Option<(usize, u64)>,
    /// Crash tests: the write point at which the process aborts.
    abort_at: Option<String>,
    /// The rule-file record the current `index` run saw, when it differs from
    /// the stored one: written after the run succeeds.
    rules_pending: std::sync::Mutex<Option<String>>,
}

/// The path resolver of this process's last delta publish: `(meta.db, publish id,
/// resolver)`. The next publish over that manifest patches it with its added and
/// deleted paths instead of loading every path; any other publish (another process,
/// a full build, a cleared cache) has another publish id, and the next one loads
/// again.
static RESOLVER_CACHE: Mutex<Option<(PathBuf, u64, crate::dependency::PathResolver)>> =
    Mutex::new(None);

/// Files the recent segment may hold before an update folds it into the delta.
pub const RECENT_MAX_FILES: usize = 256;
/// The recent segment's text limit: the delta's (merge) limit divided by this.
pub const RECENT_BYTES_DIVISOR: u64 = 16;

/// Exit code of a process stopped by `Indexer::set_abort_point`.
#[doc(hidden)]
pub const ABORT_EXIT_CODE: i32 = 86;

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

/// A walk before the binary check (see `Indexer::walk_candidates`).
#[derive(Debug, Default)]
struct Walked {
    found: Discovered,
    /// Per entry of `found`: whether its type is not code (so it is sniffed).
    non_code: Vec<bool>,
}

/// What one directory walk found.
#[derive(Debug, Default)]
struct Discovered {
    /// Each file relative to the root, with forward slashes: the path the stores
    /// and meta.db use. The walker's own path is `root.join(rel)`.
    rels: Vec<String>,
    /// On-disk size of each entry of `files` (0 when unknown); drives batching.
    sizes: Vec<u64>,
    /// The `stat` of each entry of `files`, taken during the walk.
    metas: Vec<Option<crate::cache::FileStat>>,
    skipped_too_large: usize,
    skipped_bytes_too_large: u64,
    skipped_binary: usize,
}

/// The paths a library update names, relative to the root (`/`-separated): the
/// walk goes down through their ancestors and into them, and nowhere else.
#[derive(Debug, Default)]
struct Targets {
    /// Each named path (a file or a directory).
    exact: std::collections::HashSet<String>,
    /// Every proper ancestor directory of a named path.
    ancestors: std::collections::HashSet<String>,
}

impl Targets {
    fn new(paths: &[String]) -> Self {
        let mut out = Self::default();
        for p in paths {
            let mut end = p.len();
            while let Some(slash) = p[..end].rfind('/') {
                out.ancestors.insert(p[..slash].to_string());
                end = slash;
            }
            out.exact.insert(p.clone());
        }
        out
    }

    /// Whether `rel` is a named path or lies under one.
    fn covers(&self, rel: &str) -> bool {
        if self.exact.contains(rel) {
            return true;
        }
        let mut end = rel.len();
        while let Some(slash) = rel[..end].rfind('/') {
            if self.exact.contains(&rel[..slash]) {
                return true;
            }
            end = slash;
        }
        false
    }

    /// Whether the walk enters `rel`: the root, an ancestor of a named path, or a
    /// path a named path covers.
    fn admits(&self, rel: &str) -> bool {
        rel.is_empty() || self.ancestors.contains(rel) || self.covers(rel)
    }
}

/// Walk positions for the files a library update rewrites (`rewrites`, in walk
/// order; each with its row when it has one), found by probing `files` in
/// `walk_seq` order with the walk-order comparator. A rewritten file keeps its
/// position while its neighbours still bracket it; a new or moved one gets a
/// value between the rows that now bracket it. Rows in `skip` (deleted) are left
/// out. `None` when a comparison cannot be made or a gap has no room: the caller
/// then runs a full index, which renumbers.
fn place_rewrites(
    conn: &rusqlite::Connection,
    walk: &mut crate::walk_order::WalkOrder<'_>,
    rewrites: &[(&str, Option<&crate::meta_update::StoredFile>)],
    skip: &std::collections::HashSet<i64>,
) -> Result<Option<HashMap<String, i64>>> {
    use crate::meta_update::row_near_seq;
    use std::cmp::Ordering::{Greater, Less};
    let mut skip = skip.clone();
    let mut kept: HashMap<&str, i64> = HashMap::new();
    // Rows whose neighbours no longer bracket them are placed anew; a row moved
    // out can change another's neighbours, so check again until nothing moves.
    let mut moved: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for _ in 0..4 {
        kept.clear();
        let mut changed = false;
        for (path, row) in rewrites {
            let Some(row) = row else { continue };
            if moved.contains(path) {
                continue;
            }
            let before = row_near_seq(conn, row.walk_seq, true, &skip)?;
            let after = row_near_seq(conn, row.walk_seq + 1, false, &skip)?;
            let fits = before
                .as_ref()
                .is_none_or(|(_, p, _)| walk.cmp(p, path) == Some(Less))
                && after
                    .as_ref()
                    .is_none_or(|(_, p, _)| walk.cmp(path, p) == Some(Less));
            if fits {
                kept.insert(path, row.walk_seq);
            } else {
                moved.insert(path);
                skip.insert(row.id);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let bounds: (Option<i64>, Option<i64>) =
        conn.query_row("SELECT MIN(walk_seq), MAX(walk_seq) FROM files", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })?;
    // Each file to place, with the positions of the rows that bracket it.
    let mut gaps: Vec<(&str, Option<i64>, Option<i64>)> = Vec::new();
    for (path, row) in rewrites {
        if row.is_some() && !moved.contains(path) {
            continue;
        }
        let (before, after) = match bounds {
            (Some(min), Some(max)) => {
                // The smallest position whose first row comes after `path`.
                let (mut lo, mut hi) = (min, max + 1);
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    let after_path = match row_near_seq(conn, mid, false, &skip)? {
                        None => true,
                        Some((_, q, _)) => match walk.cmp(&q, path) {
                            Some(Greater) => true,
                            Some(Less) => false,
                            _ => return Ok(None),
                        },
                    };
                    if after_path {
                        hi = mid;
                    } else {
                        lo = mid + 1;
                    }
                }
                (
                    row_near_seq(conn, lo, true, &skip)?.map(|r| r.2),
                    row_near_seq(conn, lo, false, &skip)?.map(|r| r.2),
                )
            }
            _ => (None, None),
        };
        gaps.push((path, before, after));
    }
    let mut out: HashMap<String, i64> = kept
        .into_iter()
        .map(|(p, seq)| (p.to_string(), seq))
        .collect();
    // Files that share a gap are spread over it, in walk order.
    let mut i = 0;
    while i < gaps.len() {
        let (_, before, after) = gaps[i];
        let mut j = i;
        while j < gaps.len() && (gaps[j].1, gaps[j].2) == (before, after) {
            j += 1;
        }
        let k = (j - i) as i64;
        for (n, (path, _, _)) in gaps[i..j].iter().enumerate() {
            let n = n as i64;
            let seq = match (before, after) {
                (Some(a), Some(b)) => {
                    let step = (b - a) / (k + 1);
                    if step < 1 {
                        return Ok(None);
                    }
                    a.checked_add(step * (n + 1))
                }
                (Some(a), None) => a.checked_add(WALK_SEQ_GAP * (n + 1)),
                (None, Some(b)) => b.checked_sub(WALK_SEQ_GAP * (k - n)),
                (None, None) => Some(WALK_SEQ_GAP * n),
            };
            let Some(seq) = seq else { return Ok(None) };
            out.insert(path.to_string(), seq);
        }
        i = j;
    }
    Ok(Some(out))
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

/// Phase timings of one run, logged as one line.
struct Laps {
    last: Instant,
    line: String,
}

impl Laps {
    fn new() -> Self {
        Self {
            last: Instant::now(),
            line: String::new(),
        }
    }

    fn lap(&mut self, name: &str) {
        use std::fmt::Write;
        let _ = write!(
            self.line,
            " {}={:.1}",
            name,
            self.last.elapsed().as_secs_f64() * 1000.0
        );
        self.last = Instant::now();
    }
}

/// What `Indexer::update_paths` does with one named path.
enum Named {
    /// Nothing indexed can depend on it (outside the tree, `.git/`, `.reflex/`).
    Skip,
    /// It needs a full run: an ignore file, the project config, a resolver config,
    /// the root itself, or a path that leaves the root.
    Full,
    /// A path under the root, relative and `/`-separated.
    Path(String),
}

fn classify_named(root: &Path, abs_root: &Path, path: &Path) -> Named {
    let rel = if path.is_absolute() {
        match path
            .strip_prefix(root)
            .or_else(|_| path.strip_prefix(abs_root))
        {
            Ok(rel) => rel,
            Err(_) => return Named::Skip,
        }
    } else {
        path
    };
    let mut parts: Vec<String> = Vec::new();
    for component in rel.components() {
        match component {
            std::path::Component::Normal(s) => parts.push(s.to_string_lossy().into_owned()),
            std::path::Component::CurDir => {}
            _ => return Named::Full,
        }
    }
    let Some(name) = parts.last() else {
        return Named::Full;
    };
    if parts[0] == ".git" {
        return Named::Skip;
    }
    if parts[0] == crate::cache::CACHE_DIR {
        return if parts.len() == 2 && parts[1] == "config.toml" {
            Named::Full
        } else {
            Named::Skip
        };
    }
    if is_ignore_file_name(name) || crate::dependency_resolve::is_resolver_config_name(name) {
        return Named::Full;
    }
    Named::Path(parts.join("/"))
}

/// `statistics` key: the rule files an index run saw, as a JSON map from path
/// (relative to the root) to the blake3 of its bytes (`""` when absent). The rule
/// files decide WHICH files are indexed, so a change to one makes the index stale
/// although no indexed file changed. Older binaries ignore the key.
pub(crate) const RULE_FILES_KEY: &str = "rule_files";

/// Rule files compared on every freshness check: Reflex's config and the root's
/// ignore files. Nested ignore files are compared when git lists them.
pub(crate) const FIXED_RULE_FILES: [&str; 4] =
    [".reflex/config.toml", ".gitignore", ".ignore", ".rgignore"];

/// Whether `name` is an ignore file the walker reads in every directory.
pub(crate) fn is_ignore_file_name(name: &str) -> bool {
    matches!(name, ".gitignore" | ".ignore" | ".rgignore")
}

/// Whether the path `rel` (relative, `/`-separated) is a rule file.
pub(crate) fn is_rule_file(rel: &str) -> bool {
    rel == FIXED_RULE_FILES[0] || is_ignore_file_name(rel.rsplit('/').next().unwrap_or(rel))
}

/// The blake3 of the rule file at `rel`, or `""` when it cannot be read.
pub(crate) fn rule_file_hash(root: &Path, rel: &str) -> String {
    match std::fs::read(root.join(rel)) {
        Ok(bytes) => blake3::hash(&bytes).to_hex().to_string(),
        Err(_) => String::new(),
    }
}

/// The rule-file record of an index run over `root`: the fixed files, and every
/// ignore file in `dirty` (the paths `git status` listed).
fn rule_files_record<'a>(root: &Path, dirty: impl IntoIterator<Item = &'a String>) -> String {
    let mut record: std::collections::BTreeMap<String, String> = FIXED_RULE_FILES
        .iter()
        .map(|rel| (rel.to_string(), rule_file_hash(root, rel)))
        .collect();
    for rel in dirty {
        if is_rule_file(rel) {
            record.insert(rel.clone(), rule_file_hash(root, rel));
        }
    }
    serde_json::to_string(&record).unwrap_or_default()
}

/// Whether a directory holds a file named like a resolver config at any depth (no
/// ignore rules: a superset of what the config walk would find).
fn holds_resolver_config(dir: &Path) -> bool {
    ignore::WalkBuilder::new(dir)
        .standard_filters(false)
        .build()
        .flatten()
        .any(|e| {
            e.file_name()
                .to_str()
                .is_some_and(crate::dependency_resolve::is_resolver_config_name)
        })
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
            recent_limits: None,
            abort_at: None,
            rules_pending: std::sync::Mutex::new(None),
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

    /// Abort the process at a named write point, to test crash safety. A delta
    /// publish has `delta-files` (stores written), `unlinked` (fixed names
    /// removed), `manifest` (published) and `meta` (meta.db committed); a full
    /// build has `base-files`, `base-manifest` and `base-meta`.
    #[doc(hidden)]
    pub fn set_abort_point(&mut self, point: &str) {
        self.abort_at = Some(point.to_string());
    }

    fn abort_point(&self, point: &str) {
        if self.abort_at.as_deref() == Some(point) {
            // No destructor, no later write: what a killed process leaves.
            std::process::exit(ABORT_EXIT_CODE);
        }
    }

    /// Override the recent segment's limits (files, text bytes) before it is
    /// folded into the delta. For tests.
    #[doc(hidden)]
    pub fn set_recent_limits(&mut self, max_files: usize, max_bytes: u64) {
        self.recent_limits = Some((max_files, max_bytes));
    }

    fn recent_limits(&self, max_files: usize, max_bytes: u64) -> (usize, u64) {
        self.recent_limits.unwrap_or((
            RECENT_MAX_FILES.min(max_files),
            max_bytes / RECENT_BYTES_DIVISOR,
        ))
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

    /// Bring the index up to date with changes to `paths` alone, without walking
    /// the tree: for a caller that already knows what changed (a file watcher, an
    /// editor). `paths` are files or directories, absolute or relative to `root`,
    /// that were added, modified or deleted; a rename names both paths.
    ///
    /// When every changed path is named, the index then holds what
    /// [`Indexer::index`] would build. The run falls back to [`Indexer::index`]
    /// when the change cannot be applied alone: an ignore file, `.reflex/config.toml`
    /// or a resolver config (`go.mod`, `tsconfig.json`, ...) is involved, the branch
    /// changed, the delta would pass its merge limits, or the index is not one an
    /// update can start from.
    pub fn update_paths(&self, root: impl AsRef<Path>, paths: &[PathBuf]) -> Result<IndexStats> {
        let root = root.as_ref();
        match self.try_update_paths(root, paths)? {
            Some(stats) => Ok(stats),
            None => {
                log::info!("update_paths: the change needs a full index run");
                self.index(root, false)
            }
        }
    }

    /// [`Self::update_paths`] without the fallback: `None` when a full run is
    /// needed (nothing was written). For tests that check which path ran.
    #[doc(hidden)]
    pub fn try_update_paths(&self, root: &Path, paths: &[PathBuf]) -> Result<Option<IndexStats>> {
        let started = Instant::now();
        let mut laps = Laps::new();
        let cache_dir = self.cache.path().to_path_buf();
        if !cache_dir.join(crate::cache::META_DB).exists() {
            return Ok(None);
        }
        let abs_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
        let mut named: Vec<String> = Vec::new();
        for path in paths {
            match classify_named(root, &abs_root, path) {
                Named::Skip => {}
                Named::Full => return Ok(None),
                Named::Path(rel) => named.push(rel),
            }
        }
        named.sort();
        named.dedup();

        // `git status` of the named paths runs while the rest is prepared.
        let in_git = crate::git::is_git_repo(root);
        let spawn_status = |specs: Vec<String>| {
            let root = root.to_path_buf();
            std::thread::spawn(move || {
                if specs.len() > 1000 {
                    crate::git::changed_paths(&root)
                } else {
                    let refs: Vec<&str> = specs.iter().map(String::as_str).collect();
                    crate::git::changed_paths_in(&root, &refs)
                }
            })
        };
        let mut status_jobs = Vec::new();
        if in_git && !named.is_empty() {
            status_jobs.push(spawn_status(named.clone()));
        }

        // The start of a run, as in `index_with_callback`.
        let _index_lock = crate::atomic_write::IndexLock::acquire_with_timeout(
            &cache_dir,
            std::time::Duration::from_secs(self.config.lock_wait_secs),
        )?;
        crate::atomic_write::remove_stale_tmp(&cache_dir);
        crate::query::invalidate_caches(root);
        let _invalidate_on_exit = InvalidateOnDrop(root.to_path_buf());
        self.yield_to_symbol_pass(&cache_dir)?;
        laps.lap("lock");
        self.cache.assert_writable(false)?;
        if !self.cache.check_schema_hash().unwrap_or(false)
            || !self.cache.check_extraction_hash().unwrap_or(false)
        {
            return Ok(None);
        }
        let run_start = self.run_marker(&cache_dir);
        laps.lap("checks");

        let conn = crate::cache::open_meta_db(cache_dir.join(crate::cache::META_DB))?;
        let file_count = crate::meta_update::count_files(&conn)?;
        let stored_digest = crate::meta_update::get_statistic(&conn, RESOLVER_DIGEST_KEY)?;
        let meta_generation = crate::meta_update::get_statistic(&conn, INDEX_GENERATION_KEY)?
            .and_then(|g| g.parse::<u64>().ok());
        let synced_branch =
            crate::meta_update::get_statistic(&conn, crate::meta_update::SYNCED_BRANCH_KEY)?;
        let (Some(generation), Ok(indexed)) =
            (meta_generation, CacheManager::latest_branch_info_on(&conn))
        else {
            return Ok(None);
        };
        if !self.stores_intact(file_count, meta_generation) {
            return Ok(None);
        }
        let generation = generation + 1;
        laps.lap("snapshot");

        // The resolver configs the last walk found, unchanged since.
        let Some(resolver_configs) = crate::dependency_resolve::ResolverConfigs::from_saved_list(
            root,
            &cache_dir,
            &self.config.vendored_patterns,
        ) else {
            return Ok(None);
        };
        if stored_digest.as_deref() != Some(resolver_configs.digest.as_str()) {
            return Ok(None);
        }
        laps.lap("configs");

        // Git: still the branch last indexed; a HEAD that moved adds what it changed.
        let (branch, commit) = if in_git {
            let Ok((commit, branch)) = crate::git::head_commit_and_branch(root) else {
                return Ok(None);
            };
            (branch, Some(commit))
        } else {
            ("_default".to_string(), None)
        };
        // Only the branch whose rows the last run synced (its rows are then
        // updated one by one).
        if synced_branch.as_deref() != Some(branch.as_str()) || branch != indexed.branch {
            return Ok(None);
        }
        if let Some(commit) = &commit
            && *commit != indexed.commit_sha
        {
            let Ok(moved) = crate::git::diff_names(root, &indexed.commit_sha, commit) else {
                return Ok(None);
            };
            let mut extra: Vec<String> = Vec::new();
            for path in moved {
                match classify_named(root, &abs_root, Path::new(&path)) {
                    Named::Skip => {}
                    Named::Full => return Ok(None),
                    Named::Path(rel) => {
                        if named.binary_search(&rel).is_err() {
                            extra.push(rel);
                        }
                    }
                }
            }
            if !extra.is_empty() {
                status_jobs.push(spawn_status(extra.clone()));
                named.extend(extra);
                named.sort();
                named.dedup();
            }
        }
        if resolver_configs.lists_any_under(root, &named)
            || named.iter().any(|rel| {
                let dir = root.join(rel);
                dir.is_dir() && holds_resolver_config(&dir)
            })
        {
            return Ok(None);
        }
        let targets = Arc::new(Targets::new(&named));
        laps.lap("git");

        // The rows at and under the named paths, and those paths as a full walk
        // would see them now.
        let stored = crate::meta_update::load_rows_under(&conn, &named)?;
        let Discovered {
            rels: found_rels,
            metas: found_metas,
            skipped_too_large,
            skipped_bytes_too_large,
            skipped_binary,
            ..
        } = self.discover_files_in(root, &stored, Some(Arc::clone(&targets)))?;
        let found: HashMap<&str, usize> = found_rels
            .iter()
            .enumerate()
            .map(|(i, rel)| (rel.as_str(), i))
            .collect();
        laps.lap("walk");

        // What changed, by the rules of a full run.
        let num_threads = crate::models::resolve_thread_count(self.config.parallel_threads, 32);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .context("Failed to create thread pool")?;
        let to_hash: Vec<usize> = (0..found_rels.len())
            .filter(|&i| {
                stored.get(&found_rels[i]).is_some_and(|row| {
                    !found_metas[i]
                        .as_ref()
                        .is_some_and(|md| row.stat_matches(md))
                })
            })
            .collect();
        let hashed: Vec<(usize, Option<String>)> = pool.install(|| {
            to_hash
                .par_iter()
                .map(|&i| {
                    let hash = std::fs::read(root.join(&found_rels[i]))
                        .ok()
                        .map(|b| self.hash_content(&b));
                    (i, hash)
                })
                .collect()
        });
        let mut found_status: Vec<FileStatus> = found_rels
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
            found_status[i] = match hash {
                Some(h) if h == stored[&found_rels[i]].hash => FileStatus::Touched,
                _ => FileStatus::Modified,
            };
        }
        let rewritten =
            |i: usize| matches!(found_status[i], FileStatus::Added | FileStatus::Modified);
        // Indexed paths at or under a named path that the walk no longer yields.
        let deleted: Vec<&str> = {
            let mut d: Vec<&str> = stored
                .keys()
                .map(String::as_str)
                .filter(|p| !found.contains_key(p))
                .collect();
            d.sort_unstable();
            d
        };
        laps.lap("hash");

        // Walk positions of the rewritten files.
        let to_place: Vec<(&str, Option<&crate::meta_update::StoredFile>)> = (0..found_rels.len())
            .filter(|&i| rewritten(i))
            .map(|i| (found_rels[i].as_str(), stored.get(&found_rels[i])))
            .collect();
        let deleted_ids: std::collections::HashSet<i64> =
            deleted.iter().map(|p| stored[*p].id).collect();
        let mut walk_order = crate::walk_order::WalkOrder::new(root);
        let Some(seq_of) = place_rewrites(&conn, &mut walk_order, &to_place, &deleted_ids)? else {
            return Ok(None);
        };
        drop(conn);
        let added = found_status
            .iter()
            .filter(|s| **s == FileStatus::Added)
            .count();
        let live_files = file_count - deleted.len() + added;
        laps.lap("order");

        let mut rewrites = Vec::new();
        let mut rewrite_rows = HashMap::new();
        for (i, rel) in found_rels.iter().enumerate() {
            if rewritten(i) {
                rewrites.push(Rewrite {
                    rel: rel.clone(),
                    size: found_metas[i].as_ref().map_or(0, |md| md.size()),
                    walk_seq: seq_of[rel],
                });
                if let Some(row) = stored.get(rel) {
                    rewrite_rows.insert(rel.clone(), row.clone());
                }
            }
        }
        // Dirty flags of the named paths as `git status` lists them now (every other
        // row keeps the flag it has), joined only when meta.db is written. If git
        // fails, every named path counts as dirty: a superset is safe for freshness.
        let dirty_job = || -> DirtyFlags {
            let mut now_dirty: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut ok = true;
            for job in status_jobs {
                match job.join() {
                    Ok(Ok(paths)) => now_dirty.extend(paths),
                    _ => ok = false,
                }
            }
            let is_dirty = |rel: &str| !ok || now_dirty.contains(rel);
            let mut flags = DirtyFlags {
                rewrites: std::collections::HashSet::new(),
                touched: Vec::new(),
                flips: Vec::new(),
                git_dirty: in_git && (indexed.is_dirty || !ok || !now_dirty.is_empty()),
            };
            for (i, rel) in found_rels.iter().enumerate() {
                let dirty = is_dirty(rel);
                if rewritten(i) {
                    if dirty {
                        flags.rewrites.insert(rel.clone());
                    }
                } else if let Some(row) = stored.get(rel) {
                    if found_status[i] == FileStatus::Touched {
                        let (size, mtime) = found_metas[i]
                            .as_ref()
                            .map(|md| (md.size(), md.recorded_mtime_ns(run_start)))
                            .unwrap_or((0, 0));
                        flags.touched.push((row.id, size, mtime, dirty));
                    } else if dirty != row.dirty {
                        flags.flips.push((row.id, dirty));
                    }
                }
            }
            flags
        };
        // The branch's own hashes of the rewritten paths; every other file's row
        // matches its hash (the last run on this branch synced them all).
        let branch_hashes: HashMap<String, String> = {
            let conn = crate::cache::open_meta_db(cache_dir.join(crate::cache::META_DB))?;
            let mut stmt = conn.prepare(
                "SELECT fb.hash FROM file_branches fb
                 JOIN files f ON f.id = fb.file_id
                 JOIN branches b ON b.id = fb.branch_id
                 WHERE b.name = ? AND f.path = ?",
            )?;
            let mut out = HashMap::new();
            for r in &rewrites {
                let hash: Option<String> = rusqlite::OptionalExtension::optional(
                    stmt.query_row(rusqlite::params![branch, r.rel], |row| row.get(0)),
                )?;
                if let Some(hash) = hash {
                    out.insert(r.rel.clone(), hash);
                }
            }
            out
        };
        let skipped = (skipped_too_large, skipped_bytes_too_large, skipped_binary);
        let rewritten_count = rewrites.len();
        laps.lap("change_set");

        let result = if deleted.is_empty() && rewrites.is_empty() {
            let flags = dirty_job();
            self.refresh_named(RefreshNamed {
                touched: &flags.touched,
                flips: &flags.flips,
                walk_moves: &[],
                branch: &branch,
                commit: commit.as_deref(),
                git_dirty: flags.git_dirty,
                live_files,
                skipped,
            })
            .map(Some)
        } else {
            let deleted_file_count = deleted.iter().filter(|p| !root.join(p).exists()).count();
            self.publish_delta(DeltaChanges {
                root,
                rewrites,
                deleted: deleted
                    .iter()
                    .map(|p| (p.to_string(), stored[*p].id))
                    .collect(),
                walk_moves: Vec::new(),
                dirty: Box::new(dirty_job),
                stored: rewrite_rows,
                sync_all_branch_rows: false,
                full_stats: false,
                live_files,
                breakdown_rest: (0, 0, live_files - rewritten_count),
                branch_hashes,
                branch: &branch,
                commit: commit.as_deref(),
                resolver_configs: &resolver_configs,
                run_start,
                generation,
                pool: &pool,
                deleted_file_count,
                skipped,
            })
        };
        laps.lap("publish");
        log::info!(
            "update_paths: {} named paths, {} rewritten, {} deleted in {} ms;{}",
            named.len(),
            rewritten_count,
            deleted.len(),
            started.elapsed().as_millis(),
            laps.line
        );
        result
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
        // A pass that has nothing left to do exits within milliseconds: poll fast
        // first, then every 100 ms.
        let mut pause = std::time::Duration::from_millis(5);
        while std::time::Instant::now() < deadline {
            if !BackgroundIndexer::is_running(cache_dir) {
                BackgroundIndexer::clear_cancel(cache_dir);
                log::info!("Symbol indexing yielded; continuing");
                return Ok(());
            }
            std::thread::sleep(pause);
            pause = (pause * 2).min(std::time::Duration::from_millis(100));
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
        let stats = self.index_run(root, show_progress, progress_callback)?;
        // The rule files this run indexed under, once its index is published. A
        // crash before this write leaves the old record: the next check reports
        // the rules as changed and the next run writes it.
        let pending = self
            .rules_pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(record) = pending {
            let conn = crate::cache::open_meta_db(self.cache.path().join(crate::cache::META_DB))?;
            crate::meta_update::set_statistic(
                &conn,
                RULE_FILES_KEY,
                &record,
                chrono::Utc::now().timestamp(),
            )?;
            crate::query::invalidate_caches(root);
        }
        Ok(stats)
    }

    fn index_run(
        &self,
        root: &Path,
        show_progress: bool,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<IndexStats> {
        log::info!("Indexing directory: {:?}", root);
        let mut laps = Laps::new();

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
        laps.lap("lock");

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

        // Ensure cache is initialized. A cache this binary completed a run on has
        // every table: only `config.toml` may need recreating.
        if schema_ok && extraction_ok {
            self.cache.ensure_config()?;
        } else {
            self.cache.init()?;
        }

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

        laps.lap("init");

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .context("Failed to create thread pool")?;

        // Step 1: walk the tree. In parallel, load what meta.db holds (one row per
        // path of the last indexed tree), ask git for the branch and the dirty
        // paths, and find the resolver configs (each is its own walk or subprocess).
        let meta_path = cache_dir.join(crate::cache::META_DB);
        let phase_start = Instant::now();
        let (git_state, resolver_configs, meta_read, walked) = std::thread::scope(|scope| {
            let git = scope.spawn(|| {
                let t = Instant::now();
                let r = crate::git::get_git_state_optional(root);
                (r, t.elapsed().as_millis())
            });
            let configs = scope.spawn(|| {
                let t = Instant::now();
                let r = crate::dependency_resolve::ResolverConfigs::discover(
                    root,
                    &self.config.vendored_patterns,
                );
                (r, t.elapsed().as_millis())
            });
            // Available disk space (a `df` subprocess), now that the cache exists.
            let disk = scope.spawn(|| self.check_disk_space(root));
            let meta_read = scope.spawn(|| -> Result<_> {
                let t = Instant::now();
                let conn = crate::cache::open_meta_db(&meta_path)?;
                let rows = crate::meta_update::load_stored_files(&conn)?;
                log::debug!("meta.db rows read in {} ms", t.elapsed().as_millis());
                Ok((
                    rows,
                    crate::meta_update::get_statistic(&conn, RESOLVER_DIGEST_KEY)?,
                    crate::meta_update::get_statistic(&conn, INDEX_GENERATION_KEY)?
                        .and_then(|g| g.parse::<u64>().ok()),
                    crate::meta_update::get_statistic(
                        &conn,
                        crate::meta_update::SYNCED_BRANCH_KEY,
                    )?,
                    crate::meta_update::get_statistic(&conn, RULE_FILES_KEY)?,
                ))
            });
            let t = Instant::now();
            let walked = self.walk_candidates(root, None);
            let walk_ms = t.elapsed().as_millis();
            let (git, configs) = (git.join(), configs.join());
            log::info!(
                "discovery phase: walk {} ms, git {} ms, resolver configs {} ms",
                walk_ms,
                git.as_ref().map_or(0, |g| g.1),
                configs.as_ref().map_or(0, |c| c.1)
            );
            let disk = disk
                .join()
                .unwrap_or_else(|_| Err(anyhow::anyhow!("disk space check panicked")));
            (
                git.map(|g| g.0),
                configs.map(|c| c.0),
                meta_read.join(),
                walked.and_then(|w| disk.map(|()| w)),
            )
        });
        let (stored, stored_digest, meta_generation, synced_branch, stored_rules) =
            meta_read.map_err(|_| anyhow::anyhow!("meta.db read thread panicked"))??;
        // A non-code file is text only when it holds no NUL byte; one whose stat
        // matches its row was checked when it was indexed.
        let discovered = walked.map(|w| Self::drop_binaries(root, w, &stored, Some(&pool)));
        // The snapshot currently published, and the generation the next publish takes.
        let prev_manifest = crate::snapshot::read_manifest(&cache_dir).ok().flatten();
        let generation = prev_manifest
            .as_ref()
            .map(|m| m.generation)
            .max(meta_generation)
            .unwrap_or(0)
            + 1;
        let git_state = git_state.map_err(|_| anyhow::anyhow!("git state thread panicked"))??;
        let resolver_configs =
            resolver_configs.map_err(|_| anyhow::anyhow!("resolver config walk panicked"))?;
        // For `update_paths`, which parses the same configs without walking.
        if let Err(e) = resolver_configs.save_list(root, &cache_dir) {
            log::warn!("Failed to save the resolver config list: {:#}", e);
        }
        let Discovered {
            rels,
            sizes,
            metas,
            skipped_too_large,
            skipped_bytes_too_large,
            skipped_binary,
        } = discovered?;
        let total_files = rels.len();
        laps.lap("discover");
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
        let rules = rule_files_record(root, &dirty_paths);
        *self.rules_pending.lock().unwrap_or_else(|e| e.into_inner()) =
            (stored_rules.as_deref() != Some(rules.as_str())).then_some(rules);

        // The branch's own hashes: the basis of the "new / modified / unchanged"
        // breakdown, as before stable ids.
        let in_sync = synced_branch.as_deref() == Some(branch.as_str());
        let existing_hashes = if in_sync {
            BranchHashes::Synced(&stored)
        } else {
            BranchHashes::Loaded(self.cache.load_hashes_for_branch(&branch)?)
        };
        laps.lap("branch_hashes");

        // Step 2: what changed. A file whose (size, mtime) match its row is
        // unchanged without being read; the others are read and hashed.
        let classify_start = Instant::now();
        // Each walked path's row, looked up once.
        let rows_of: Vec<Option<&crate::meta_update::StoredFile>> =
            rels.iter().map(|rel| stored.get(rel)).collect();
        let to_hash: Vec<usize> = (0..total_files)
            .filter(|&i| {
                rows_of[i]
                    .is_some_and(|row| !metas[i].as_ref().is_some_and(|md| row.stat_matches(md)))
            })
            .collect();
        let hashed: Vec<(usize, Option<String>)> = pool.install(|| {
            to_hash
                .par_iter()
                .map(|&i| {
                    (
                        i,
                        std::fs::read(root.join(&rels[i]))
                            .ok()
                            .map(|b| self.hash_content(&b)),
                    )
                })
                .collect()
        });
        let mut status: Vec<FileStatus> = rows_of
            .iter()
            .map(|row| {
                if row.is_some() {
                    FileStatus::Unchanged
                } else {
                    FileStatus::Added
                }
            })
            .collect();
        for (i, hash) in hashed {
            status[i] = match (hash, rows_of[i]) {
                (Some(h), Some(row)) if h == row.hash => FileStatus::Touched,
                _ => FileStatus::Modified,
            };
        }
        // Every stored path was walked (the usual case) when as many walked paths
        // have a row as there are rows: nothing is gone.
        let walked_with_row = status.iter().filter(|s| **s != FileStatus::Added).count();
        let gone: Vec<&String> = if walked_with_row == stored.len() {
            Vec::new()
        } else {
            let discovered_set: std::collections::HashSet<&str> =
                rels.iter().map(String::as_str).collect();
            stored
                .keys()
                .filter(|p| !discovered_set.contains(p.as_str()))
                .collect()
        };
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
        laps.lap("classify");
        let stores_ok = schema_ok && self.stores_intact(stored.len(), meta_generation);
        laps.lap("snapshot");

        if !content_changed && !full_deps && stores_ok {
            log::info!("No files changed - skipping index rebuild");
            log::info!("index run phases:{}", laps.line);
            return self.refresh_unchanged(RefreshUnchanged {
                rels: &rels,
                metas: &metas,
                status: &status,
                rows: &rows_of,
                existing_hashes: &existing_hashes,
                dirty_paths: &dirty_paths,
                branch: &branch,
                in_sync,
                commit: commit.as_deref(),
                git_dirty: git_state.as_ref().map(|s| s.dirty),
                run_start,
                skipped: (skipped_too_large, skipped_bytes_too_large, skipped_binary),
            });
        }
        if content_changed && !full_deps && stores_ok {
            // The change set from the whole walk: walk positions planned over every
            // file, dirty flags from `git status`.
            let seqs = crate::meta_update::plan_walk_seq(
                &rels
                    .iter()
                    .map(|rel| stored.get(rel).map(|s| s.walk_seq))
                    .collect::<Vec<_>>(),
            );
            let mut rewrites = Vec::new();
            let mut dirty_rewrites = std::collections::HashSet::new();
            let mut rewrite_rows = HashMap::new();
            let mut branch_hashes = HashMap::new();
            let (mut touched, mut walk_moves, mut flips) = (Vec::new(), Vec::new(), Vec::new());
            let mut rest: Vec<(&str, &str)> = Vec::new();
            for i in 0..total_files {
                let rel = &rels[i];
                let dirty = dirty_paths.contains(rel);
                match (status[i], stored.get(rel)) {
                    (FileStatus::Added | FileStatus::Modified, row) => {
                        rewrites.push(Rewrite {
                            rel: rel.clone(),
                            size: sizes[i],
                            walk_seq: seqs[i],
                        });
                        if dirty {
                            dirty_rewrites.insert(rel.clone());
                        }
                        if let Some(row) = row {
                            rewrite_rows.insert(rel.clone(), row.clone());
                        }
                        if let Some(h) = existing_hashes.get(rel) {
                            branch_hashes.insert(rel.clone(), h.to_string());
                        }
                    }
                    (_, Some(row)) => {
                        if seqs[i] != row.walk_seq {
                            walk_moves.push((row.id, seqs[i]));
                        }
                        if status[i] == FileStatus::Touched {
                            let (size, mtime) = metas[i]
                                .as_ref()
                                .map(|md| (md.size(), md.recorded_mtime_ns(run_start)))
                                .unwrap_or((0, 0));
                            touched.push((row.id, size, mtime, dirty));
                        } else if dirty != row.dirty {
                            flips.push((row.id, dirty));
                        }
                        rest.push((rel.as_str(), row.hash.as_str()));
                    }
                    (_, None) => unreachable!("an unchanged file has a row"),
                }
            }
            let git_dirty = git_state.as_ref().map(|s| s.dirty).unwrap_or(false);
            let changes = DeltaChanges {
                root,
                rewrites,
                deleted: gone.iter().map(|p| ((*p).clone(), stored[*p].id)).collect(),
                walk_moves,
                dirty: Box::new(move || DirtyFlags {
                    rewrites: dirty_rewrites,
                    touched,
                    flips,
                    git_dirty,
                }),
                stored: rewrite_rows,
                sync_all_branch_rows: !in_sync,
                full_stats: true,
                live_files: total_files,
                breakdown_rest: breakdown(rest.into_iter(), &existing_hashes),
                branch_hashes,
                branch: &branch,
                commit: commit.as_deref(),
                resolver_configs: &resolver_configs,
                run_start,
                generation,
                pool: &pool,
                deleted_file_count,
                skipped: (skipped_too_large, skipped_bytes_too_large, skipped_binary),
            };
            match self.publish_delta(changes)? {
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

        // A merge: when the published stores match meta.db, a file whose bytes
        // are unchanged takes its text from them instead of the disk. Same text,
        // same order, same builder: the new base is byte-identical to a build
        // that reads every file.
        let merge_source: Option<(crate::snapshot::IndexSnapshot, HashMap<String, u32>)> =
            if stores_ok {
                crate::snapshot::IndexSnapshot::open(&cache_dir)
                    .ok()
                    .map(|snapshot| {
                        let ids = snapshot
                            .live_ids()
                            .filter_map(|id| {
                                Some((snapshot.get_file_path(id)?.to_str()?.to_string(), id))
                            })
                            .collect();
                        (snapshot, ids)
                    })
            } else {
                None
            };
        let from_snapshot = std::sync::atomic::AtomicUsize::new(0);

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
                        let reused = match &merge_source {
                            Some((snapshot, ids))
                                if matches!(
                                    status[i],
                                    FileStatus::Unchanged | FileStatus::Touched
                                ) =>
                            {
                                ids.get(&rels[i]).and_then(|&id| {
                                    self.reuse_file(
                                        &ctx,
                                        snapshot,
                                        id,
                                        i,
                                        &metas[i],
                                        trigram_scratch,
                                    )
                                })
                            }
                            _ => None,
                        };
                        let result = match reused {
                            Some(r) => {
                                from_snapshot.fetch_add(1, Ordering::Relaxed);
                                Some(r)
                            }
                            None => self.process_file(&ctx, i, trigram_scratch),
                        };
                        counter_clone.fetch_add(1, Ordering::Relaxed);
                        result
                    })
                    .collect()
            });
            pool_ms += pool_start.elapsed().as_millis();
            // A merge copied this batch's text out of the old stores: let their pages
            // go, or the whole old base ends up resident by the last batch.
            if let Some((snapshot, _)) = &merge_source {
                snapshot.release_pages();
            }

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
            "phase read+extract: {} ms in pool, {} ms building trigram batches, {} ms total; {} files from the published stores",
            pool_ms,
            flush_ms,
            batch_phase_start.elapsed().as_millis(),
            from_snapshot.load(Ordering::Relaxed)
        );
        drop(merge_source);

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
        self.abort_point("base-files");
        crate::snapshot::write_manifest(&cache_dir, &manifest)?;
        crate::snapshot::link_fixed_names(&cache_dir, &manifest);
        self.abort_point("base-manifest");

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
        self.abort_point("base-meta");

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

        // Return stats with incremental breakdown (the branch's rows were synced
        // above: count from `files` alone).
        let mut stats = self.cache.stats_synced()?;
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

    /// Apply a change set as a delta on the published base: the rewritten files
    /// are read, the delta is rebuilt from them and the previous delta's files the
    /// change set leaves alone, and the base files they supersede (or that are
    /// gone) are tombstoned. `meta.db` changes only the rows the change set names,
    /// in one transaction. `Ok(None)` when the delta would pass its limits (nothing
    /// is written): the caller then builds a new base (a merge).
    fn publish_delta(&self, c: DeltaChanges<'_>) -> Result<Option<IndexStats>> {
        use crate::snapshot::{IndexSnapshot, Manifest, SegmentFiles};
        use rusqlite::OptionalExtension;
        let cache_dir = self.cache.path().to_path_buf();
        let snapshot = IndexSnapshot::open(&cache_dir)?;
        let Some(prev) = snapshot.manifest().cloned() else {
            return Ok(None);
        };
        let base_len = snapshot.base_len();
        let delta_start = Instant::now();
        let mut laps = Laps::new();

        let rels: Vec<String> = c.rewrites.iter().map(|r| r.rel.clone()).collect();
        let delta_len = snapshot.delta_len();
        // Ids below this are base and delta files, tombstoned in place; the recent
        // segment above them is rebuilt.
        let stable_end = base_len + delta_len;

        // Store entries that go: every rewritten and every deleted path.
        let gone: std::collections::HashSet<String> = rels
            .iter()
            .cloned()
            .chain(c.deleted.iter().map(|(p, _)| p.clone()))
            .collect();
        let path_of = |id: u32| -> Option<&str> { snapshot.get_file_path(id)?.to_str() };
        let (mut new_dead_base, mut new_dead_delta) = (Vec::new(), Vec::new());
        for id in 0..stable_end {
            if path_of(id).is_some_and(|p| gone.contains(p)) {
                if id < base_len {
                    new_dead_base.push(id);
                } else {
                    new_dead_delta.push(id);
                }
            }
        }
        let kept_in = |range: std::ops::Range<u32>| -> Vec<(String, u32)> {
            range
                .filter_map(|id| {
                    let p = path_of(id)?;
                    (!gone.contains(p)).then(|| (p.to_string(), id))
                })
                .collect()
        };
        let delta_kept = kept_in(base_len..stable_end);
        let recent_kept = kept_in(stable_end..snapshot.id_bound());

        // The limits, from stat sizes and entry-table lengths, before anything is
        // read (a merge reads the files itself).
        let len_of = |kept: &[(String, u32)]| -> u64 {
            kept.iter()
                .map(|(_, id)| snapshot.file_len(*id).unwrap_or(0))
                .sum()
        };
        let rewrite_bytes: u64 = c.rewrites.iter().map(|r| r.size).sum();
        let (delta_kept_bytes, recent_kept_bytes) = (len_of(&delta_kept), len_of(&recent_kept));
        let (max_files, max_bytes) = self.merge_limits(prev.live_corpus_bytes);
        let all_files = c.rewrites.len() + delta_kept.len() + recent_kept.len();
        let all_bytes = rewrite_bytes + delta_kept_bytes + recent_kept_bytes;
        if all_files > max_files || all_bytes > max_bytes {
            log::info!(
                "Delta of {} files / ~{} bytes passes its limits ({} files / {} bytes)",
                all_files,
                all_bytes,
                max_files,
                max_bytes
            );
            return Ok(None);
        }
        // The recent segment is rebuilt by every update; past its own limits it is
        // folded, with the delta's live files, into a new delta.
        let (recent_max_files, recent_max_bytes) = self.recent_limits(max_files, max_bytes);
        let fold = c.rewrites.len() + recent_kept.len() > recent_max_files
            || rewrite_bytes + recent_kept_bytes > recent_max_bytes;

        // Read the rewritten files.
        let ctx = ProcessCtx {
            root: c.root,
            rels: &rels,
            stored: &c.stored,
            run_start: c.run_start,
            full_deps: false,
            tsconfigs: &c.resolver_configs.tsconfigs,
        };
        let mut read: Vec<Option<FileProcessingResult>> = c.pool.install(|| {
            (0..rels.len())
                .into_par_iter()
                .map_init(Vec::<u64>::new, |scratch, i| {
                    self.process_file(&ctx, i, scratch)
                })
                .collect()
        });
        laps.lap("read");

        // An unreadable file is gone, as in a full run: its row goes too.
        let mut deleted = c.deleted;
        let mut live_files = c.live_files;
        for (k, r) in read.iter().enumerate() {
            if r.is_none() {
                live_files -= 1;
                if let Some(row) = c.stored.get(&rels[k]) {
                    deleted.push((rels[k].clone(), row.id));
                }
            }
        }

        // The new segment: the read files and the kept ones, in walk order.
        enum Source {
            Fresh(usize),
            Old(u32),
        }
        let kept: Vec<(String, u32)> = if fold {
            delta_kept.into_iter().chain(recent_kept).collect()
        } else {
            recent_kept
        };
        let kept_seqs: HashMap<String, i64> = if kept.is_empty() {
            HashMap::new()
        } else {
            let conn = crate::cache::open_meta_db(cache_dir.join(crate::cache::META_DB))?;
            let mut stmt = conn.prepare("SELECT walk_seq FROM files WHERE path = ?")?;
            let mut out = HashMap::new();
            for (p, _) in &kept {
                if let Some(seq) = stmt.query_row([p], |r| r.get::<_, i64>(0)).optional()? {
                    out.insert(p.clone(), seq);
                }
            }
            out
        };
        let mut entries: Vec<(i64, String, Source)> = Vec::new();
        for (k, r) in read.iter().enumerate() {
            if r.is_some() {
                entries.push((c.rewrites[k].walk_seq, rels[k].clone(), Source::Fresh(k)));
            }
        }
        for (p, id) in kept {
            let seq = kept_seqs.get(&p).copied().unwrap_or(i64::MAX);
            entries.push((seq, p, Source::Old(id)));
        }
        entries.sort_by_key(|(seq, _, _)| *seq);
        let mut segment_bytes = 0u64;
        for (_, _, src) in &entries {
            segment_bytes += match src {
                Source::Fresh(k) => read[*k].as_ref().map_or(0, |f| f.content.len() as u64),
                Source::Old(id) => snapshot.file_len(*id).unwrap_or(0),
            };
        }
        let (stable_files, stable_bytes) = if fold {
            (0, 0)
        } else {
            (
                snapshot.delta_len() as usize
                    - new_dead_delta.len()
                    - (0..delta_len)
                        .filter(|&k| snapshot.is_dead(base_len + k))
                        .count(),
                delta_kept_bytes,
            )
        };
        if entries.len() + stable_files > max_files || segment_bytes + stable_bytes > max_bytes {
            log::info!(
                "Delta of {} files / {} bytes passes its limits after reading",
                entries.len() + stable_files,
                segment_bytes + stable_bytes
            );
            return Ok(None);
        }

        // Tombstones, and the planning sizes of the postings they remove.
        let mut tombstones: Vec<u32> = if fold {
            // A new delta: only base ids stay tombstoned.
            prev.tombstones
                .iter()
                .copied()
                .filter(|&id| id < base_len)
                .chain(new_dead_base.iter().copied())
                .collect()
        } else {
            prev.tombstones
                .iter()
                .copied()
                .chain(new_dead_base.iter().copied())
                .chain(new_dead_delta.iter().copied())
                .collect()
        };
        tombstones.sort_unstable();
        tombstones.dedup();
        let sizes_of = |ids: &[u32]| -> Vec<Vec<(crate::trigram::Trigram, u32)>> {
            c.pool.install(|| {
                ids.par_iter()
                    .map_init(Vec::<u64>::new, |scratch, &id| {
                        let content = snapshot.get_file_content(id).unwrap_or("");
                        let run = crate::trigram_build::extract_trigram_run(content, scratch);
                        crate::trigram_build::run_plan_sizes(&run)
                    })
                    .collect()
            })
        };
        let carried = |plan: Option<&crate::snapshot::TombPlan>| {
            plan.map(|t| t.entries().map(|(t, n)| (t, n as u64)).collect())
                .unwrap_or_default()
        };
        type TombMap = std::collections::BTreeMap<crate::trigram::Trigram, u64>;
        let mut touched_trigrams: Vec<crate::trigram::Trigram> = Vec::new();
        let mut tomb: TombMap = carried(snapshot.tomb());
        for sizes in sizes_of(&new_dead_base) {
            for (t, n) in sizes {
                *tomb.entry(t).or_default() += n as u64;
                touched_trigrams.push(t);
            }
        }
        let mut tomb_delta: TombMap = if fold {
            TombMap::new()
        } else {
            carried(snapshot.tomb_delta())
        };
        for sizes in sizes_of(&new_dead_delta) {
            for (t, n) in sizes {
                if !fold {
                    *tomb_delta.entry(t).or_default() += n as u64;
                }
                touched_trigrams.push(t);
            }
        }
        laps.lap("tombstones");

        // Write the new segment (generation files: nothing names them yet).
        let (delta_content, delta_trigrams, delta_plan, tomb_name) =
            crate::snapshot::delta_file_names(c.generation);
        let (recent_content, recent_trigrams, recent_plan, tomb_delta_name) =
            crate::snapshot::recent_file_names(c.generation);
        let (seg_content, seg_trigrams, seg_plan) = if fold {
            (delta_content, delta_trigrams, delta_plan)
        } else {
            (recent_content, recent_trigrams, recent_plan)
        };
        let segment = if entries.is_empty() {
            None
        } else {
            let runs: Vec<Option<TrigramRun>> = c.pool.install(|| {
                entries
                    .par_iter()
                    .map_init(Vec::<u64>::new, |scratch, (_, _, src)| match src {
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
            let content_path = cache_dir.join(&seg_content);
            writer
                .init(content_path.clone())
                .context("Failed to initialize the delta content store")?;
            for ((_, rel, src), run) in entries.iter().zip(runs) {
                let path = PathBuf::from(rel);
                match src {
                    Source::Fresh(k) => {
                        let f = read[*k].as_mut().expect("read file");
                        let run = std::mem::take(&mut f.trigram_run);
                        builder.add_file(path.clone(), run);
                        writer.add_file(path, &f.content);
                    }
                    Source::Old(id) => {
                        builder.add_file(path.clone(), run.expect("kept file run"));
                        writer.add_file(path, snapshot.get_file_content(*id)?);
                    }
                }
            }
            let trigrams_path = cache_dir.join(&seg_trigrams);
            builder
                .write_with_plan(c.pool, &trigrams_path, Some(&cache_dir.join(&seg_plan)))
                .context("Failed to write the delta trigram index")?;
            writer
                .finalize_if_needed()
                .context("Failed to finalize the delta content store")?;
            Some(SegmentFiles {
                content: seg_content.clone(),
                trigrams: seg_trigrams.clone(),
                plan: Some(seg_plan.clone()),
                files: writer.file_count() as u64,
                content_bytes: std::fs::metadata(&content_path)?.len(),
                trigrams_bytes: std::fs::metadata(&trigrams_path)?.len(),
            })
        };
        let write_tomb = |map: &TombMap, name: String| -> Result<String> {
            let entries: Vec<(crate::trigram::Trigram, u32)> = map
                .iter()
                .map(|(&t, &n)| (t, n.min(u32::MAX as u64) as u32))
                .collect();
            crate::snapshot::write_tomb_file(&cache_dir.join(&name), &entries)?;
            Ok(name)
        };
        let tomb_file = if tombstones.iter().any(|&id| id < base_len) {
            Some(write_tomb(&tomb, tomb_name)?)
        } else {
            None
        };
        let tomb_delta_file = if tombstones.iter().any(|&id| id >= base_len) {
            Some(write_tomb(&tomb_delta, tomb_delta_name)?)
        } else {
            None
        };
        let (delta_files, recent_files) = if fold {
            (segment, None)
        } else {
            (prev.delta.clone(), segment)
        };
        laps.lap("write");

        // The live trigram count, kept up to date from the trigrams this update
        // touches: those of tombstoned files, of the replaced and the new segment.
        let segment_index =
            |files: &Option<SegmentFiles>| -> Result<Option<crate::trigram::TrigramIndex>> {
                let Some(f) = files else { return Ok(None) };
                let mut index = crate::trigram::TrigramIndex::load(cache_dir.join(&f.trigrams))?;
                if let Some(plan) = &f.plan {
                    index.attach_plan(cache_dir.join(plan))?;
                }
                Ok(Some(index))
            };
        let new_segment = segment_index(if fold { &delta_files } else { &recent_files })?;
        if let Some(recent) = snapshot.recent() {
            touched_trigrams.extend(recent.trigrams.trigrams());
        }
        if fold && let Some(delta) = snapshot.delta() {
            touched_trigrams.extend(delta.trigrams.trigrams());
        }
        if let Some(index) = &new_segment {
            touched_trigrams.extend(index.trigrams());
        }
        touched_trigrams.sort_unstable();
        touched_trigrams.dedup();
        let plan_in = |index: Option<&crate::trigram::TrigramIndex>, t| -> i64 {
            index
                .and_then(|i| i.list_part(t))
                .map_or(0, |(_, plan)| plan as i64)
        };
        let old_delta = snapshot.delta().map(|d| &d.trigrams);
        let (new_delta, new_recent) = if fold {
            (new_segment.as_ref(), None)
        } else {
            (old_delta, new_segment.as_ref())
        };
        let mut live_trigrams = prev.live_trigrams as i64;
        for &t in &touched_trigrams {
            let was = snapshot.live_plan(t) > 0;
            let now = plan_in(Some(&snapshot.base().trigrams), t)
                + plan_in(new_delta, t)
                + plan_in(new_recent, t)
                - tomb.get(&t).copied().unwrap_or(0) as i64
                - tomb_delta.get(&t).copied().unwrap_or(0) as i64
                > 0;
            live_trigrams += now as i64 - was as i64;
        }
        let dead_set: std::collections::HashSet<u32> = tombstones.iter().copied().collect();
        // Lengths come from the entry table: reading the content would page in (and
        // UTF-8 check) the whole base on every update.
        let mut live_corpus = segment_bytes;
        let stable_bound = if fold { base_len } else { stable_end };
        for id in 0..stable_bound {
            if !dead_set.contains(&id) {
                live_corpus += snapshot.file_len(id).unwrap_or(0);
            }
        }
        laps.lap("live");

        // Publish: fixed names first (an older binary must not read the base alone
        // once it is incomplete), then the manifest, then meta.db.
        let manifest = Manifest {
            format: 0,
            generation: c.generation,
            base: prev.base.clone(),
            delta: delta_files,
            recent: recent_files,
            tombstones,
            tomb: tomb_file,
            tomb_delta: tomb_delta_file,
            live_trigrams: live_trigrams.max(0) as u64,
            publish_id: 0,
            live_corpus_bytes: live_corpus,
            checksum: String::new(),
        }
        .sealed();
        self.abort_point("delta-files");
        if !manifest.base_only() {
            crate::snapshot::unlink_fixed_names(&cache_dir);
        }
        self.abort_point("unlinked");
        crate::snapshot::write_manifest(&cache_dir, &manifest)?;
        crate::snapshot::link_fixed_names(&cache_dir, &manifest);
        self.abort_point("manifest");
        log::info!(
            "phase delta: {} delta + {} recent files ({} read{}), {} tombstones, {} ms",
            manifest.delta.as_ref().map_or(0, |d| d.files),
            manifest.recent.as_ref().map_or(0, |d| d.files),
            read.iter().filter(|r| r.is_some()).count(),
            if fold {
                ", folded into a new delta"
            } else {
                ""
            },
            manifest.tombstones.len(),
            delta_start.elapsed().as_millis()
        );
        laps.lap("manifest");

        // meta.db: the named rows, their dependencies, the statistics and the
        // branch row, in one transaction.
        let meta_start = Instant::now();
        let now = chrono::Utc::now().timestamp();
        let flags = (c.dirty)();
        laps.lap("dirty");
        let mut rows: Vec<crate::cache::FileRow> = Vec::new();
        let mut row_k: Vec<usize> = Vec::new();
        for (k, r) in read.iter().enumerate() {
            if let Some(f) = r {
                rows.push(crate::cache::FileRow {
                    path: rels[k].clone(),
                    hash: f.hash.clone(),
                    language: format!("{:?}", f.language),
                    line_count: f.line_count,
                    size: f.size,
                    mtime_ns: f.mtime_ns,
                    dirty: flags.rewrites.contains(&rels[k]),
                    walk_seq: c.rewrites[k].walk_seq,
                });
                row_k.push(k);
            }
        }
        let paths_changed =
            !deleted.is_empty() || rows.iter().any(|r| !c.stored.contains_key(&r.path));
        let mut conn = crate::cache::open_meta_db(cache_dir.join(crate::cache::META_DB))?;
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .context("Failed to begin meta.db transaction")?;
        let deleted_ids: Vec<i64> = deleted.iter().map(|(_, id)| *id).collect();
        crate::meta_update::delete_files(&tx, &deleted_ids)?;
        let ids = crate::meta_update::upsert_files(&tx, &rows, now)?;
        crate::meta_update::set_walk_seqs(&tx, &c.walk_moves)?;
        crate::meta_update::refresh_stats(&tx, &flags.touched)?;
        crate::meta_update::set_dirty_flags(&tx, &flags.flips)?;
        let branch_id = self
            .cache
            .get_or_create_branch_id(&tx, c.branch, c.commit)?;
        if c.sync_all_branch_rows {
            crate::meta_update::sync_branch_rows(&tx, branch_id, c.branch, now)?;
        } else {
            let branch_rows: Vec<(i64, &str)> = ids
                .iter()
                .zip(&rows)
                .map(|(id, row)| (*id, row.hash.as_str()))
                .collect();
            crate::meta_update::set_branch_rows(&tx, branch_id, &branch_rows, now)?;
        }
        log::info!(
            "meta.db: {} rows written, {} deleted, {} moved, {} touched, {} dirty flags",
            rows.len(),
            deleted_ids.len(),
            c.walk_moves.len(),
            flags.touched.len(),
            flags.flips.len()
        );
        laps.lap("rows");

        // The resolver this process kept from the previous publish, patched with this
        // one's paths, or every path loaded again.
        let meta_path = cache_dir.join(crate::cache::META_DB);
        let cached = RESOLVER_CACHE
            .lock()
            .ok()
            .and_then(|mut slot| slot.take())
            .filter(|(path, publish, _)| {
                *path == meta_path && prev.publish_id != 0 && *publish == prev.publish_id
            })
            .map(|(_, _, resolver)| resolver);
        let resolver = match cached {
            Some(mut resolver) => {
                for (path, _) in &deleted {
                    resolver.remove(path);
                }
                for (row, id) in rows.iter().zip(&ids) {
                    resolver.insert(*id, &row.path);
                }
                resolver
            }
            None => crate::dependency::PathResolver::from_conn(&tx)
                .context("Failed to load file paths for dependency resolution")?,
        };
        laps.lap("resolver");
        let ctx = crate::dependency_resolve::ResolverContext::new(c.root, c.resolver_configs);
        let mut writer = crate::dependency::DependencyWriter::new(&tx);
        let mut resolved_here: std::collections::HashSet<i64> = std::collections::HashSet::new();
        for (&k, &file_id) in row_k.iter().zip(&ids) {
            let Some((imports, exports, declared)) =
                read[k].as_ref().and_then(|f| f.imports.as_ref())
            else {
                continue;
            };
            let rel = &rels[k];
            resolved_here.insert(file_id);
            let deps = ctx.resolve_file_imports(file_id, rel, imports.clone(), &resolver);
            writer.replace_dependencies(file_id, &deps)?;
            writer.replace_members(
                file_id,
                &crate::dependency_resolve::package_members(rel, declared),
            )?;
            writer.clear_exports(file_id)?;
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
        let reresolved = if paths_changed {
            reresolve(&tx, &ctx, &resolver, &resolved_here)?
        } else {
            0
        };
        refresh_vendored(&tx, &c.resolver_configs.vendor, Some(&ids))?;
        laps.lap("deps");

        crate::meta_update::set_statistic(
            &tx,
            RESOLVER_DIGEST_KEY,
            &c.resolver_configs.digest,
            now,
        )?;
        crate::meta_update::set_statistic(
            &tx,
            INDEX_GENERATION_KEY,
            &c.generation.to_string(),
            now,
        )?;
        CacheManager::update_branch_metadata_on(
            &tx,
            c.branch,
            c.commit,
            live_files,
            flags.git_dirty,
        )?;
        // Every file has its branch row now: the count is the file count.
        CacheManager::set_total_files_on(&tx, live_files, now)?;
        CacheManager::update_schema_hash_on(&tx)?;
        CacheManager::update_extraction_hash_on(&tx)?;
        laps.lap("stamps");
        tx.commit()?;
        self.abort_point("meta");
        if let Ok(mut slot) = RESOLVER_CACHE.lock() {
            *slot = Some((meta_path, manifest.publish_id, resolver));
        }
        laps.lap("commit");
        self.cache
            .checkpoint_wal()
            .context("Failed to checkpoint WAL")?;
        log::info!(
            "dependencies: {} rows, {} exports written; {} rows re-resolved",
            deps_written,
            exports_written,
            reresolved
        );
        log::info!(
            "phase meta.db (files, branches, dependencies, exports): {} ms",
            meta_start.elapsed().as_millis()
        );

        crate::query::invalidate_caches(c.root);
        drop(snapshot);
        crate::snapshot::remove_unreferenced(&cache_dir, &manifest, Some(&prev));
        laps.lap("cleanup");

        let (mut new_files, mut modified_files, mut unchanged_files) = c.breakdown_rest;
        for (k, r) in read.iter().enumerate() {
            if let Some(f) = r {
                match c.branch_hashes.get(&rels[k]) {
                    None => new_files += 1,
                    Some(old) if *old != f.hash => modified_files += 1,
                    _ => unchanged_files += 1,
                }
            }
        }
        let mut stats = if c.full_stats {
            // The branch's rows name every file now: count from `files` alone.
            self.cache.stats_synced()?
        } else {
            self.light_stats(live_files, now)
        };
        laps.lap("stats");
        log::info!("delta update phases:{}", laps.line);
        stats.new_files = new_files;
        stats.modified_files = modified_files;
        stats.deleted_files = c.deleted_file_count;
        stats.unchanged_files = unchanged_files;
        stats.skipped_too_large = c.skipped.0;
        stats.skipped_bytes_too_large = c.skipped.1;
        stats.skipped_binary = c.skipped.2;
        Ok(Some(stats))
    }

    /// [`Self::process_file`] for a file whose bytes are the ones the published
    /// stores hold (`id` there): the text comes from the stores, the hash from its
    /// row, and a touched file's stat from the walk. `None` when the stores cannot
    /// give it (the caller reads the file).
    fn reuse_file(
        &self,
        ctx: &ProcessCtx<'_>,
        snapshot: &crate::snapshot::IndexSnapshot,
        id: u32,
        i: usize,
        meta: &Option<crate::cache::FileStat>,
        trigram_scratch: &mut Vec<u64>,
    ) -> Option<FileProcessingResult> {
        let row = ctx.stored.get(&ctx.rels[i])?;
        let content = snapshot.get_file_content(id).ok()?.to_string();
        let file_path = &ctx.root.join(&ctx.rels[i]);
        let (size, mtime_ns) = match meta {
            Some(md) if !row.stat_matches(md) => (md.size(), md.recorded_mtime_ns(ctx.run_start)),
            _ => (row.size, row.mtime_ns),
        };
        let language = Language::from_path(file_path);
        let line_count = content.lines().count();
        let trigram_run = crate::trigram_build::extract_trigram_run(&content, trigram_scratch);
        let imports = ctx.full_deps.then(|| {
            let path_str = file_path.to_string_lossy().to_string();
            extract_imports(language, &content, &path_str, ctx.root, ctx.tsconfigs)
        });
        Some(FileProcessingResult {
            hash: row.hash.clone(),
            content,
            language,
            line_count,
            size,
            mtime_ns,
            imports,
            trigram_run,
        })
    }

    /// Read one discovered file: stat, bytes, hash, text, trigram run, and imports
    /// when they may have changed. `None` when it cannot be read.
    fn process_file(
        &self,
        ctx: &ProcessCtx<'_>,
        i: usize,
        trigram_scratch: &mut Vec<u64>,
    ) -> Option<FileProcessingResult> {
        let file_path = &ctx.root.join(&ctx.rels[i]);
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
        // A manifest, headers and sizes: not the stores themselves (a run that
        // changes nothing must not page in their path tables). Whatever opens them
        // next validates them in full.
        match crate::snapshot::check_published(self.cache.path()) {
            Ok(None) => {
                log::info!("Index written before the manifest - rebuilding");
                false
            }
            Ok(Some(manifest)) => {
                if Some(manifest.generation) != meta_generation {
                    log::warn!(
                        "Manifest generation {} but meta.db rows are generation {:?} (a run stopped between the two) - rebuilding",
                        manifest.generation,
                        meta_generation
                    );
                    false
                } else if manifest.live_files() as usize != expected {
                    log::warn!(
                        "Stores hold {} files but meta.db lists {} - rebuilding",
                        manifest.live_files(),
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
    /// The statistics of a library update: counts and sizes, without the
    /// per-language maps (they scan every row).
    fn light_stats(&self, total_files: usize, now: i64) -> IndexStats {
        let (index_size_bytes, trigram_index_bytes, corpus_bytes) = self.cache.store_sizes();
        IndexStats {
            total_files,
            index_size_bytes,
            last_updated: chrono::DateTime::from_timestamp(now, 0)
                .unwrap_or_else(chrono::Utc::now)
                .to_rfc3339(),
            corpus_bytes,
            trigram_index_bytes,
            ..Default::default()
        }
    }

    /// A library update that changed no content: the named rows' stats, flags and
    /// positions, the branch row and the statistics (what `refresh_unchanged`
    /// does for a whole walk).
    fn refresh_named(&self, r: RefreshNamed<'_>) -> Result<IndexStats> {
        let mut conn = crate::cache::open_meta_db(self.cache.path().join(crate::cache::META_DB))?;
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .context("Failed to begin meta.db transaction")?;
        crate::meta_update::set_walk_seqs(&tx, r.walk_moves)?;
        crate::meta_update::refresh_stats(&tx, r.touched)?;
        crate::meta_update::set_dirty_flags(&tx, r.flips)?;
        if r.commit.is_some() {
            CacheManager::update_branch_metadata_on(
                &tx,
                r.branch,
                r.commit,
                r.live_files,
                r.git_dirty,
            )?;
        }
        CacheManager::update_stats_on(&tx, r.branch)?;
        tx.commit()?;
        let mut stats = self.light_stats(r.live_files, chrono::Utc::now().timestamp());
        stats.new_files = 0;
        stats.modified_files = 0;
        stats.deleted_files = 0;
        stats.unchanged_files = r.live_files;
        stats.skipped_too_large = r.skipped.0;
        stats.skipped_bytes_too_large = r.skipped.1;
        stats.skipped_binary = r.skipped.2;
        Ok(stats)
    }

    fn refresh_unchanged(&self, r: RefreshUnchanged<'_>) -> Result<IndexStats> {
        let mut laps = Laps::new();
        let now = chrono::Utc::now().timestamp();
        let rows: Vec<&crate::meta_update::StoredFile> = r
            .rows
            .iter()
            .map(|row| row.expect("nothing was added: every file has a row"))
            .collect();
        let seqs = crate::meta_update::plan_walk_seq(
            &rows
                .iter()
                .map(|row| Some(row.walk_seq))
                .collect::<Vec<_>>(),
        );
        let mut walk = Vec::new();
        let mut flips = Vec::new();
        let mut touched = Vec::new();
        let no_dirty = r.dirty_paths.is_empty();
        for (i, rel) in r.rels.iter().enumerate() {
            let row = rows[i];
            let dirty = !no_dirty && r.dirty_paths.contains(rel);
            if seqs[i] != row.walk_seq {
                walk.push((row.id, seqs[i]));
            }
            if r.status[i] == FileStatus::Touched {
                let (size, mtime) = r.metas[i]
                    .as_ref()
                    .map(|md| (md.size(), md.recorded_mtime_ns(r.run_start)))
                    .unwrap_or((0, 0));
                touched.push((row.id, size, mtime, dirty));
            } else if dirty != row.dirty {
                flips.push((row.id, dirty));
            }
        }

        laps.lap("plan");
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
        laps.lap("rows");
        if !r.in_sync {
            crate::meta_update::sync_branch_rows(&tx, branch_id, r.branch, now)?;
        }
        laps.lap("branch_rows");

        // The CONTENT is current, but the recorded commit may not be: committing
        // already-indexed files moves HEAD without changing a single hash.
        // Skipping the metadata update left `commit_sha` behind forever, so
        // freshness reported `stale` on a perfectly current index until some
        // unrelated edit happened to force a rebuild.
        if let Some(git_dirty) = r.git_dirty {
            CacheManager::update_branch_metadata_on(
                &tx,
                r.branch,
                r.commit,
                r.rels.len(),
                git_dirty,
            )?;
        }
        // Every file has its branch row now: the count is the file count.
        CacheManager::set_total_files_on(&tx, r.rels.len(), now)?;
        tx.commit()?;
        laps.lap("commit");

        // On the synced branch every file's branch hash is its row's: all unchanged.
        let (new_files, modified_files, unchanged_files) = if r.in_sync {
            (0, 0, rows.len())
        } else {
            breakdown(
                r.rels
                    .iter()
                    .zip(&rows)
                    .map(|(rel, row)| (rel.as_str(), row.hash.as_str())),
                r.existing_hashes,
            )
        };
        // Nothing changed: the statistics are the rows' (what `stats_synced` would
        // count from `files`), without a query.
        let mut files_by_language: HashMap<String, usize> = HashMap::new();
        let mut lines_by_language: HashMap<String, usize> = HashMap::new();
        for row in &rows {
            match files_by_language.get_mut(&row.language) {
                Some(n) => *n += 1,
                None => {
                    files_by_language.insert(row.language.clone(), 1);
                }
            }
            match lines_by_language.get_mut(&row.language) {
                Some(n) => *n += row.line_count,
                None => {
                    lines_by_language.insert(row.language.clone(), row.line_count);
                }
            }
        }
        let (index_size_bytes, trigram_index_bytes, corpus_bytes) = self.cache.store_sizes();
        let mut stats = IndexStats {
            total_files: r.rels.len(),
            index_size_bytes,
            last_updated: chrono::DateTime::from_timestamp(now, 0)
                .unwrap_or_else(chrono::Utc::now)
                .to_rfc3339(),
            files_by_language,
            lines_by_language,
            corpus_bytes,
            trigram_index_bytes,
            ..Default::default()
        };
        laps.lap("stats");
        log::info!("refresh phases:{}", laps.line);
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
        let mut laps = Laps::new();
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
                            .map(|md| (md.size(), md.recorded_mtime_ns(w.run_start)))
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

        laps.lap("plan");
        let mut conn = crate::cache::open_meta_db(self.cache.path().join(crate::cache::META_DB))?;
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .context("Failed to begin meta.db transaction")?;

        crate::meta_update::delete_files(&tx, &deleted)?;
        let new_ids = crate::meta_update::upsert_files(&tx, &rows, now)?;
        crate::meta_update::set_walk_seqs(&tx, &walk)?;
        crate::meta_update::refresh_stats(&tx, &touched)?;
        crate::meta_update::set_dirty_flags(&tx, &flips)?;
        laps.lap("rows");
        let branch_id = self
            .cache
            .get_or_create_branch_id(&tx, w.branch, w.commit)?;
        crate::meta_update::sync_branch_rows(&tx, branch_id, w.branch, now)?;
        laps.lap("branch_rows");
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
        laps.lap("resolver");
        let ctx = crate::dependency_resolve::ResolverContext::new(w.root, w.resolver_configs);
        let mut writer = crate::dependency::DependencyWriter::new(&tx);
        if w.full_deps {
            writer.clear_all()?;
        }
        // In walk order: the row order a full build produces.
        let mut resolved_here: std::collections::HashSet<i64> = std::collections::HashSet::new();
        for f in w.written {
            let Some((imports, exports, declared)) = &f.imports else {
                continue;
            };
            let rel = &w.rels[f.index];
            let file_id = ids[&f.index];
            resolved_here.insert(file_id);
            let deps = ctx.resolve_file_imports(file_id, rel, imports.clone(), &resolver);
            writer.replace_dependencies(file_id, &deps)?;
            writer.replace_members(
                file_id,
                &crate::dependency_resolve::package_members(rel, declared),
            )?;
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
        laps.lap("deps");

        // An added or removed path can change how an unchanged file's imports
        // resolve (suffix matches, ambiguity, the first of several candidates).
        let mut reresolved = 0usize;
        if !w.full_deps && paths_changed {
            reresolved = reresolve(&tx, &ctx, &resolver, &resolved_here)?;
        }
        let written_ids: Vec<i64> = ids.values().copied().collect();
        refresh_vendored(
            &tx,
            &w.resolver_configs.vendor,
            (!w.full_deps).then_some(written_ids.as_slice()),
        )?;

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
        laps.lap("reresolve");
        tx.commit()?;
        laps.lap("commit");
        log::info!("meta.db write phases:{}", laps.line);
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
    #[cfg(test)]
    fn discover_files(
        &self,
        root: &Path,
        stored: &HashMap<String, crate::meta_update::StoredFile>,
    ) -> Result<Discovered> {
        self.discover_files_in(root, stored, None)
    }

    /// [`Self::discover_files`] limited to `targets` (all files when `None`): the
    /// same walker and rules, entering only the named paths and their ancestors, so
    /// every ignore file on the way applies exactly as in a full walk.
    fn discover_files_in(
        &self,
        root: &Path,
        stored: &HashMap<String, crate::meta_update::StoredFile>,
        targets: Option<Arc<Targets>>,
    ) -> Result<Discovered> {
        let walked = self.walk_candidates(root, targets)?;
        Ok(Self::drop_binaries(root, walked, stored, None))
    }

    /// The walk half of discovery: every file the walker and the path policy admit,
    /// under the size limit, with its stat, in walk order. Non-code files are only
    /// marked: [`Self::drop_binaries`] decides, once the stored rows are known.
    fn walk_candidates(&self, root: &Path, targets: Option<Arc<Targets>>) -> Result<Walked> {
        let mut out = Walked::default();

        let policy = self.path_policy(root);
        let walker = Self::targeted_walk_builder(root, &self.config, &policy, targets).build();

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
                    out.found.skipped_too_large += 1;
                    out.found.skipped_bytes_too_large += size;
                    continue;
                }
            }

            out.found.rels.push(normalize_rel(root, path));
            out.found.sizes.push(size);
            out.found
                .metas
                .push(metadata.as_ref().map(crate::cache::FileStat::of));
            out.non_code.push(!lang.is_code());
        }

        Ok(out)
    }

    /// The binary half of discovery. A code extension is trusted to be text.
    /// Anything else in the tracked tier (`image.png`, `OWNERS`, `data.bin`) is
    /// sniffed: ripgrep's rule, a NUL byte anywhere means binary, and a binary file
    /// is never in the index. Only the long tail pays the read (from the page
    /// cache, since the main pass reads it again a moment later), and not a file
    /// that is unchanged since it was indexed. Order is kept.
    fn drop_binaries(
        root: &Path,
        walked: Walked,
        stored: &HashMap<String, crate::meta_update::StoredFile>,
        pool: Option<&rayon::ThreadPool>,
    ) -> Discovered {
        let Walked {
            found: mut out,
            non_code,
        } = walked;
        let to_sniff: Vec<usize> = (0..out.rels.len())
            .filter(|&i| {
                non_code[i]
                    && !stored
                        .get(&out.rels[i])
                        .zip(out.metas[i].as_ref())
                        .is_some_and(|(row, md)| row.stat_matches(md))
            })
            .collect();
        let sniff = |i: &usize| looks_binary(&root.join(&out.rels[*i]));
        let binary: Vec<bool> = match pool {
            Some(pool) => pool.install(|| to_sniff.par_iter().map(sniff).collect()),
            None => to_sniff.iter().map(sniff).collect(),
        };
        let dropped: std::collections::HashSet<usize> = to_sniff
            .iter()
            .zip(binary)
            .filter_map(|(&i, b)| b.then_some(i))
            .collect();
        if dropped.is_empty() {
            return out;
        }
        for &i in &dropped {
            log::debug!("Skipping {} (binary)", out.rels[i]);
        }
        out.skipped_binary += dropped.len();
        fn keep<T>(v: &mut Vec<T>, dropped: &std::collections::HashSet<usize>) {
            let mut i = 0;
            v.retain(|_| {
                let k = !dropped.contains(&i);
                i += 1;
                k
            });
        }
        keep(&mut out.rels, &dropped);
        keep(&mut out.sizes, &dropped);
        keep(&mut out.metas, &dropped);
        out
    }

    /// The directory walker every tree pass shares: the indexer, and the freshness
    /// check outside git (which has no `git status` to name candidates and must
    /// walk). One builder so the two can never disagree about what is in the tree.
    /// [`Indexer::walk_builder`], entering only `targets` and their ancestors when
    /// given.
    fn targeted_walk_builder(
        root: &Path,
        config: &IndexConfig,
        policy: &PathPolicy,
        targets: Option<Arc<Targets>>,
    ) -> WalkBuilder {
        let mut builder = Self::walk_builder(root, config, policy);
        if let Some(targets) = targets {
            // Replaces the walker's own entry filter, so repeat it.
            let hidden = policy.hidden();
            let root_buf = root.to_path_buf();
            builder.filter_entry(move |e| {
                if hidden {
                    let name = e.file_name();
                    if name == ".git" || name == crate::cache::CACHE_DIR {
                        return false;
                    }
                }
                let rel = normalize_rel(&root_buf, e.path());
                targets.admits(&rel)
            });
        }
        builder
    }

    /// Which of the files `rels` (relative, `/`-separated) the walk reaches: the
    /// ignore files and the include/exclude patterns applied exactly as an index
    /// run applies them. Walks only the ancestors of `rels`.
    pub(crate) fn walked_among(
        root: &Path,
        config: &IndexConfig,
        policy: &PathPolicy,
        rels: &[String],
    ) -> std::collections::HashSet<String> {
        let targets = Arc::new(Targets::new(rels));
        let walker =
            Self::targeted_walk_builder(root, config, policy, Some(Arc::clone(&targets))).build();
        walker
            .flatten()
            .filter(|e| e.file_type().is_some_and(|t| t.is_file()))
            .map(|e| normalize_rel(root, e.path()))
            .filter(|rel| targets.exact.contains(rel))
            .collect()
    }

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
            .rels
            .iter()
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>();
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
            .rels
            .iter()
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>();
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
        assert_eq!(found.rels.len(), 5, "3 code files, the markdown, the .xyz");
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
            .rels
            .iter()
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>();
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
            .rels
            .iter()
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>();

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
