//! Query engine for searching indexed code
//!
//! The query engine loads the memory-mapped cache and executes
//! deterministic searches based on lexical, structural, or symbol patterns.

pub mod filter;
pub mod open_index;
pub mod result;

pub use filter::{LiteralPattern, QueryFilter, prepare_literal_pattern, substring_hint_text};

use anyhow::{Context, Result};
use regex::Regex;

use crate::cache::CacheManager;
use crate::models::{
    IndexPath, IndexStatus, IndexWarning, Language, QueryResponse, SearchResult, Span, SymbolKind,
};
use crate::output;
use crate::parsers::ParserFactory;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use open_index::OpenIndex;

/// Drop every process-wide memo for `workspace_root`: the shared open-index
/// handle and the freshness snapshot. Called around index writes so no query in
/// this process keeps serving from replaced files or a stale verdict.
pub fn invalidate_caches(workspace_root: &std::path::Path) {
    open_index::invalidate(&workspace_root.join(crate::cache::CACHE_DIR));
    status_cache::invalidate(workspace_root);
}

/// What `search_internal` hands back: the page plus the bookkeeping the public
/// wrappers turn into pagination, hints and timings.
struct Internal {
    results: Vec<SearchResult>,
    /// Matches before offset/limit. Exact when `total_is_exact`; otherwise the
    /// number verified before the page filled, which the public wrappers must
    /// NOT report as a total.
    total_count: usize,
    /// False when verification stopped early (list mode with a limit).
    total_is_exact: bool,
    /// Sample-based estimate of the total when it is not exact.
    approx_total: Option<usize>,
    /// Candidate lines that contained the pattern only as a substring
    /// (whole-identifier searches only), for the zero-result hint.
    substring_only: Option<usize>,
    /// Time spent in trigram lookup and intersection.
    candidates_us: u64,
    /// Trigram index or full scan.
    index_path: IndexPath,
    /// Engine warnings from the candidate pass.
    warnings: Vec<String>,
}

/// Bookkeeping from a candidate pass (trigram or regex).
#[derive(Debug, Clone)]
struct CandidateStats {
    candidates_us: u64,
    substring_only: Option<usize>,
    /// Every candidate was verified.
    exhausted: bool,
    /// Sample-based estimate of the total, only when not `exhausted`.
    approx_total: Option<usize>,
    /// Trigram index or full scan.
    index_path: IndexPath,
    /// What the engine wants the caller to know about the candidate pass (a
    /// regex that had to scan), for `QueryResponse.warnings`.
    warnings: Vec<String>,
}

impl Default for CandidateStats {
    fn default() -> Self {
        Self {
            candidates_us: 0,
            substring_only: None,
            exhausted: true,
            approx_total: None,
            index_path: IndexPath::Trigram,
            warnings: Vec::new(),
        }
    }
}

/// Which lines of a candidate file to verify.
enum LineSet {
    /// Only these 1-based lines (from the trigram intersection), ascending.
    Only(Vec<u32>),
    /// Every line: the pattern gave the index nothing to narrow on.
    All,
}

/// Per-file filters, applied INSIDE the verification loop so that early
/// termination counts only files the caller will actually receive.
struct FileFilter {
    language: Option<Language>,
    include: Option<globset::GlobSet>,
    exclude: Option<globset::GlobSet>,
    file_pattern: Option<String>,
    exclude_text: bool,
}

impl FileFilter {
    fn from_filter(filter: &QueryFilter) -> Self {
        Self {
            language: filter.language,
            include: result::build_glob_set(&filter.glob_patterns, "glob"),
            exclude: result::build_glob_set(&filter.exclude_patterns, "exclude"),
            file_pattern: filter.file_pattern.clone(),
            exclude_text: filter.exclude_text,
        }
    }

    fn accept(&self, path: &str, lang: Language) -> bool {
        if let Some(want) = self.language
            && lang != want
        {
            return false;
        }
        if self.exclude_text && lang.is_text() {
            return false;
        }
        if let Some(set) = &self.include
            && !set.is_match(path)
        {
            return false;
        }
        if let Some(set) = &self.exclude
            && set.is_match(path)
        {
            return false;
        }
        if let Some(needle) = &self.file_pattern
            && !path.contains(needle.as_str())
        {
            return false;
        }
        true
    }
}

/// What a streaming verification pass produced.
struct VerifyOutcome {
    /// In `(path, line)` order.
    results: Vec<SearchResult>,
    /// Every candidate file was verified.
    exhausted: bool,
    /// Estimate of the full match count, only when not `exhausted`: the page's
    /// verified matches plus the sampled hit rate over the remaining candidates.
    /// `None` when a remaining file had `LineSet::All` (no line count to scale).
    estimated_total: Option<usize>,
    /// Whole-identifier searches only: candidate lines holding the pattern as a
    /// substring but not as a whole identifier.
    substring_only: usize,
}

/// The per-line predicate for one query, built once.
///
/// The whole-identifier check used to compile `\b…\b` for every candidate line;
/// on a common word that compile cost more than the match itself.
enum LineMatcher {
    /// Whole-identifier match (`\b…\b`), the default. Keeps the literal so the
    /// zero-result hint can count substring-only lines.
    WordBoundary(Regex, String),
    /// Substring match (`contains: true`), also the fallback when the
    /// word-boundary regex cannot be built.
    Contains(String),
    /// User-supplied regex, with the `kind` label its matches carry: a regex the
    /// engine built from an `ignore_case` literal still reports `text_match`.
    Regex(Regex, &'static str),
}

impl LineMatcher {
    fn new(pattern: &str, filter: &QueryFilter) -> Result<Self> {
        if filter.use_regex {
            let label = if filter.ignore_case && filter.rewritten_from.is_some() {
                "text_match"
            } else {
                "regex_match"
            };
            return match Regex::new(pattern) {
                Ok(re) => Ok(Self::Regex(re, label)),
                Err(e) => {
                    log::error!("Invalid regex pattern '{}': {}", pattern, e);
                    anyhow::bail!("Invalid regex pattern '{}': {}", pattern, e);
                }
            };
        }
        if filter.use_contains {
            return Ok(Self::Contains(pattern.to_string()));
        }
        match Regex::new(&format!(r"\b{}\b", regex::escape(pattern))) {
            Ok(re) => Ok(Self::WordBoundary(re, pattern.to_string())),
            Err(_) => {
                log::debug!(
                    "Word boundary regex failed for pattern '{}', falling back to substring",
                    pattern
                );
                Ok(Self::Contains(pattern.to_string()))
            }
        }
    }

    fn is_word_boundary(&self) -> bool {
        matches!(self, Self::WordBoundary(..))
    }

    #[inline]
    fn is_match(&self, line: &str) -> bool {
        match self {
            Self::WordBoundary(re, _) | Self::Regex(re, _) => re.is_match(line),
            Self::Contains(p) => line.contains(p.as_str()),
        }
    }

    /// Byte offset of the first match, to window the preview on it.
    #[inline]
    fn find(&self, line: &str) -> Option<usize> {
        match self {
            Self::WordBoundary(re, _) | Self::Regex(re, _) => re.find(line).map(|m| m.start()),
            Self::Contains(p) => line.find(p.as_str()),
        }
    }

    /// The literal text of a whole-identifier or substring matcher (regex: none).
    fn literal(&self) -> Option<&str> {
        match self {
            Self::WordBoundary(_, p) | Self::Contains(p) => Some(p.as_str()),
            Self::Regex(..) => None,
        }
    }

    /// The `kind` label a text match carries in results.
    fn kind_label(&self) -> &'static str {
        match self {
            Self::Regex(_, label) => label,
            _ => "text_match",
        }
    }
}

/// Verify candidate files in path order, in parallel, stopping once `budget`
/// results exist.
///
/// Files are processed in rounds of growing chunks (16, 32, … 1024). Inside a
/// round rayon's indexed `map().collect()` keeps file order, and lines within a
/// file are ascending, so the concatenation is already in `(path, line)` order:
/// the first page is complete and correctly ordered the moment the budget is met,
/// without verifying the rest. With `budget: None` every file is verified.
/// Files sampled to estimate the total after a page fills early. Spread evenly over
/// the remaining candidates (path order), so one directory of dense hits does not
/// set the rate for the whole repo.
pub const ESTIMATE_SAMPLE_FILES: usize = 32;
/// Remaining candidate lines at or below which the search just finishes instead of
/// sampling: verifying 128 lines costs less than reasoning about them.
pub const ESTIMATE_FINISH_LINES: usize = 128;
/// Candidate lines verified per sampled file, so a single minified bundle with a
/// thousand candidate lines cannot dominate the sample.
pub const ESTIMATE_PER_FILE_LINES: usize = 16;

fn verify_files_streaming(
    open: &OpenIndex,
    files: Vec<(u32, LineSet)>,
    matcher: &LineMatcher,
    file_filter: &FileFilter,
    paths_only: bool,
    budget: Option<usize>,
) -> VerifyOutcome {
    use rayon::prelude::*;

    let content = &open.content;

    // Resolve path + language once per file, drop files the filters reject, and
    // sort by the path string the results will carry.
    let mut accepted: Vec<(u32, String, Language, LineSet)> = files
        .into_iter()
        .filter_map(|(file_id, lines)| {
            let path = content.get_file_path(file_id)?;
            let lang = Language::from_path(path);
            let path_str = path.to_string_lossy().into_owned();
            file_filter
                .accept(&path_str, lang)
                .then_some((file_id, path_str, lang, lines))
        })
        .collect();
    accepted.sort_by(|a, b| a.1.cmp(&b.1));

    let count_substring_only = matcher.is_word_boundary();
    let substring_only = AtomicUsize::new(0);
    let kind_label = matcher.kind_label();

    let verify_one = |file_id: u32, path: &str, lang: Language, lines: &LineSet| {
        let Ok(text) = content.get_file_content(file_id) else {
            return Vec::new();
        };
        let all_lines: Vec<&str> = text.lines().collect();
        let mut out = Vec::new();

        let mut check = |line_no: usize, line: &str| -> bool {
            let Some(offset) = matcher.find(line) else {
                if count_substring_only
                    && let Some(p) = matcher.literal()
                    && line.contains(p)
                {
                    substring_only.fetch_add(1, Ordering::Relaxed);
                }
                return false;
            };
            out.push(SearchResult {
                path: path.to_string(),
                lang,
                kind: SymbolKind::Unknown(kind_label.to_string()),
                symbol: None,
                span: Span {
                    start_line: line_no,
                    end_line: line_no,
                },
                // Bounded, and WINDOWED on the match: on a minified bundle this
                // line is the whole 1.45 MB file, and the first 512 bytes of it
                // would tell the caller nothing.
                preview: crate::parsers::preview::line_preview(line, offset),
                dependencies: None,
            });
            true
        };

        match lines {
            LineSet::Only(nos) => {
                let mut last = 0u32;
                for &line_no in nos {
                    if line_no == last {
                        continue;
                    }
                    last = line_no;
                    let idx = line_no as usize;
                    if idx == 0 || idx > all_lines.len() {
                        log::debug!(
                            "Line {} out of bounds (file has {} lines)",
                            line_no,
                            all_lines.len()
                        );
                        continue;
                    }
                    if check(idx, all_lines[idx - 1]) && paths_only {
                        break;
                    }
                }
            }
            LineSet::All => {
                for (i, line) in all_lines.iter().enumerate() {
                    if check(i + 1, line) && paths_only {
                        break;
                    }
                }
            }
        }
        out
    };

    let mut results: Vec<SearchResult> = Vec::new();
    let mut next = 0usize;
    let mut chunk = 16usize;
    let mut exhausted = true;

    while next < accepted.len() {
        if let Some(b) = budget
            && results.len() >= b
        {
            exhausted = false;
            break;
        }
        let end = (next + chunk).min(accepted.len());
        let slice = &accepted[next..end];
        let round: Vec<Vec<SearchResult>> = open.pool().install(|| {
            slice
                .par_iter()
                .map(|(file_id, path, lang, lines)| verify_one(*file_id, path, *lang, lines))
                .collect()
        });
        for r in round {
            results.extend(r);
        }
        next = end;
        chunk = (chunk * 2).min(1024);
    }

    // The page is full but candidates remain. Either finish (cheap) or estimate the
    // total from a spread sample, so the caller gets a number that is close rather
    // than a posting-list upper bound that ran 2x high in the field.
    let mut estimated_total = None;
    if !exhausted {
        let remaining = &accepted[next..];
        // One "unit" is what a result counts: a line normally, a file in paths mode
        // (`verify_one` stops at the first hit per file there).
        let unit = |lines: &LineSet| -> Option<usize> {
            match lines {
                _ if paths_only => Some(1),
                LineSet::Only(v) => Some(v.len()),
                LineSet::All => None,
            }
        };
        let remaining_units = remaining
            .iter()
            .try_fold(0usize, |acc, (_, _, _, l)| unit(l).map(|u| acc + u));

        let finish = remaining.len() <= ESTIMATE_SAMPLE_FILES
            || remaining_units.is_some_and(|u| u <= ESTIMATE_FINISH_LINES);
        if finish {
            let round: Vec<Vec<SearchResult>> = open.pool().install(|| {
                remaining
                    .par_iter()
                    .map(|(file_id, path, lang, lines)| verify_one(*file_id, path, *lang, lines))
                    .collect()
            });
            for r in round {
                results.extend(r);
            }
            exhausted = true;
        } else if let Some(units) = remaining_units {
            let stride = remaining.len() / ESTIMATE_SAMPLE_FILES;
            let picks: Vec<&(u32, String, Language, LineSet)> = (0..ESTIMATE_SAMPLE_FILES)
                .map(|i| &remaining[i * stride])
                .collect();
            let (hits, sampled_units) = open.pool().install(|| {
                picks
                    .par_iter()
                    .map(|(file_id, path, lang, lines)| {
                        let trimmed = match lines {
                            LineSet::Only(v) if !paths_only => {
                                LineSet::Only(v[..v.len().min(ESTIMATE_PER_FILE_LINES)].to_vec())
                            }
                            LineSet::Only(v) => LineSet::Only(v.clone()),
                            LineSet::All => LineSet::All,
                        };
                        let u = unit(&trimmed).unwrap_or(0);
                        // Sample hits are counted, never appended: the page must stay
                        // a contiguous prefix in path order.
                        (verify_one(*file_id, path, *lang, &trimmed).len(), u)
                    })
                    .reduce(|| (0usize, 0usize), |a, b| (a.0 + b.0, a.1 + b.1))
            });
            let rate = hits as f64 / sampled_units.max(1) as f64;
            estimated_total = Some(results.len() + (rate * units as f64).round() as usize);
        }
    }

    VerifyOutcome {
        results,
        exhausted,
        estimated_total,
        substring_only: substring_only.into_inner(),
    }
}

/// Manages query execution against the index
pub struct QueryEngine {
    cache: CacheManager,
    /// The open index for this engine's lifetime, resolved on first use so every
    /// phase of one query reads the same files.
    open: OnceLock<Arc<OpenIndex>>,
}

impl QueryEngine {
    /// Create a new query engine with the given cache manager
    pub fn new(cache: CacheManager) -> Self {
        Self {
            cache,
            open: OnceLock::new(),
        }
    }

    /// The shared open-index handle (see [`open_index`]).
    ///
    /// Typed `CacheCorrupted` on a short or garbled store, so the MCP layer can
    /// rebuild once and retry.
    fn open_index(&self) -> Result<Arc<OpenIndex>> {
        if let Some(open) = self.open.get() {
            return Ok(Arc::clone(open));
        }
        let open = open_index::get_or_open(&self.cache)?;
        let _ = self.open.set(Arc::clone(&open));
        Ok(open)
    }

    /// Load dependencies for search results if requested (legacy - per result)
    /// Deprecated: Use group_and_load_dependencies for file-level grouping
    fn load_dependencies(&self, results: &mut [SearchResult], include_deps: bool) -> Result<()> {
        if !include_deps || results.is_empty() {
            return Ok(());
        }

        log::debug!("Loading dependencies for {} results", results.len());

        // Create dependency index
        // Note: We need to pass the workspace root, not the cache directory
        // The cache path is .reflex/, so its parent is the workspace root (.)
        let workspace_root = self
            .cache
            .path()
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Cache path has no parent"))?;
        let cache_for_deps = CacheManager::new(workspace_root);
        let dep_index = crate::dependency::DependencyIndex::new(cache_for_deps);

        // Load dependencies for each result
        for result in results {
            // Normalize path: strip leading "./" if present
            let normalized_path = result.path.strip_prefix("./").unwrap_or(&result.path);

            // Get file_id from database by path
            match self.cache.get_file_id(normalized_path) {
                Ok(Some(file_id)) => {
                    log::debug!("Found file_id={} for path={}", file_id, result.path);
                    // Get dependencies for this file
                    match dep_index.get_dependencies_info(file_id) {
                        Ok(dep_infos) => {
                            log::debug!(
                                "Loaded {} dependencies for file_id={}",
                                dep_infos.len(),
                                file_id
                            );
                            if !dep_infos.is_empty() {
                                result.dependencies = Some(dep_infos);
                            }
                        }
                        Err(e) => {
                            log::warn!("Failed to get dependencies for file_id={}: {}", file_id, e);
                        }
                    }
                }
                Ok(None) => {
                    log::warn!("No file_id found for path: {}", result.path);
                }
                Err(e) => {
                    log::warn!("Failed to get file_id for path {}: {}", result.path, e);
                }
            }
        }

        Ok(())
    }

    /// Group search results by file and load dependencies at file level
    /// Returns file-grouped results with dependencies populated once per file
    fn group_and_load_dependencies(
        &self,
        results: Vec<SearchResult>,
        include_deps: bool,
        context_lines: usize,
    ) -> Result<Vec<crate::models::FileGroupedResult>> {
        use crate::models::{FileGroupedResult, MatchResult};
        use std::collections::HashMap;

        if results.is_empty() {
            return Ok(Vec::new());
        }

        // Group results by file path (preserving language from first match)
        let mut grouped: HashMap<String, Vec<SearchResult>> = HashMap::new();
        for result in results {
            grouped.entry(result.path.clone()).or_default().push(result);
        }

        // Create dependency index if needed
        let dep_index = if include_deps {
            let workspace_root = self
                .cache
                .path()
                .parent()
                .ok_or_else(|| anyhow::anyhow!("Cache path has no parent"))?;
            let cache_for_deps = CacheManager::new(workspace_root);
            Some(crate::dependency::DependencyIndex::new(cache_for_deps))
        } else {
            None
        };

        // The shared index handle, for extracting context lines
        let open_opt = self.open_index().ok();
        let content_reader_opt = open_opt.as_deref().map(|o| &o.content);

        // Convert to FileGroupedResult and load dependencies
        let mut file_results: Vec<FileGroupedResult> = grouped
            .into_iter()
            .map(|(path, file_matches)| {
                // Capture language from first match (all matches in a file share the same language)
                let language = file_matches.first().map(|r| r.lang).unwrap_or_default();

                // Load dependencies for this file (once per file, not per result)
                let dependencies = if let Some(dep_idx) = &dep_index {
                    let normalized_path = path.strip_prefix("./").unwrap_or(&path);
                    match self.cache.get_file_id(normalized_path) {
                        Ok(Some(file_id)) => match dep_idx.get_dependencies_info(file_id) {
                            Ok(dep_infos) if !dep_infos.is_empty() => {
                                log::debug!(
                                    "Loaded {} dependencies for file: {}",
                                    dep_infos.len(),
                                    path
                                );
                                Some(dep_infos)
                            }
                            Ok(_) => None,
                            Err(e) => {
                                log::warn!("Failed to get dependencies for {}: {}", path, e);
                                None
                            }
                        },
                        Ok(None) => {
                            log::warn!("No file_id found for path: {}", path);
                            None
                        }
                        Err(e) => {
                            log::warn!("Failed to get file_id for path {}: {}", path, e);
                            None
                        }
                    }
                } else {
                    None
                };

                // Array file id (not the database id) for context extraction
                let file_id_for_context = open_opt.as_deref().and_then(|o| o.file_id_for(&path));
                log::debug!(
                    "Context extraction: file={}, file_id={:?}, content_reader={}",
                    path,
                    file_id_for_context,
                    content_reader_opt.is_some()
                );

                // Convert SearchResults to MatchResults (strip path and dependencies) and extract context
                let matches: Vec<MatchResult> = file_matches
                    .into_iter()
                    .map(|r| {
                        // Extract context lines if requested (0 = disabled)
                        let (context_before, context_after) = if context_lines > 0 {
                            if let (Some(reader), Some(fid)) =
                                (&content_reader_opt, file_id_for_context)
                            {
                                let result = reader
                                    .get_context_by_line(fid, r.span.start_line, context_lines)
                                    .unwrap_or_else(|e| {
                                        log::warn!(
                                            "Failed to extract context for {}:{}: {}",
                                            path,
                                            r.span.start_line,
                                            e
                                        );
                                        (vec![], vec![])
                                    });
                                log::debug!(
                                    "Extracted context for {}:{} - before: {}, after: {}",
                                    path,
                                    r.span.start_line,
                                    result.0.len(),
                                    result.1.len()
                                );
                                result
                            } else {
                                if content_reader_opt.is_none() {
                                    log::debug!(
                                        "No ContentReader available for context extraction"
                                    );
                                }
                                if file_id_for_context.is_none() {
                                    log::debug!("No file_id found for {}", path);
                                }
                                (vec![], vec![])
                            }
                        } else {
                            (vec![], vec![])
                        };

                        MatchResult {
                            kind: r.kind,
                            symbol: r.symbol,
                            span: r.span,
                            preview: r.preview,
                            context_before,
                            context_after,
                        }
                    })
                    .collect();

                FileGroupedResult {
                    path,
                    language,
                    dependencies,
                    matches,
                }
            })
            .collect();

        // Sort by path for deterministic output
        file_results.sort_by(|a, b| a.path.cmp(&b.path));

        Ok(file_results)
    }

    /// Execute a query and return matching results with index metadata
    ///
    /// This is the preferred method for programmatic/JSON output as it includes
    /// index freshness information that AI agents can use to decide whether to re-index.
    pub fn search_with_metadata(
        &self,
        pattern: &str,
        filter: QueryFilter,
    ) -> Result<QueryResponse> {
        log::info!(
            "Executing query with metadata: pattern='{}', filter={:?}",
            pattern,
            filter
        );

        let started = std::time::Instant::now();

        // Ensure cache exists
        if !self.cache.exists() {
            return Err(crate::errors::ReflexError::IndexNotFound.into());
        }

        // Open (or reuse) the index. A short or garbled store surfaces as a typed
        // `CacheCorrupted`, so the MCP layer can auto-rebuild once and retry. This
        // replaces the per-query `validate()`, whose `PRAGMA quick_check` walked the
        // whole database on every call.
        self.open_index()?;
        let open_us = started.elapsed().as_micros() as u64;

        // A whole-identifier pattern containing brackets can never match; run it as an
        // escaped regex and say so. This lives here, not in the surfaces, so the CLI,
        // MCP and HTTP paths cannot disagree about it (1.7.2 fixed only MCP).
        let prepared = prepare_literal_pattern(pattern, &filter);
        let mut filter = filter;
        if prepared.use_regex && !filter.use_regex {
            log::debug!(
                "Pattern {:?} rewritten to regex {:?}",
                pattern,
                prepared.effective
            );
            filter.rewritten_from = Some(pattern.to_string());
        }
        filter.use_regex = prepared.use_regex;

        // Execute the search first, so freshness can be judged against the files this
        // answer actually came from.
        let Internal {
            results,
            total_count: total,
            total_is_exact,
            approx_total,
            substring_only,
            candidates_us,
            index_path,
            warnings: engine_warnings,
        } = self.search_internal(&prepared.effective, filter.clone())?;
        let search_done = started.elapsed();

        // Get index status and warning (without printing warnings to stderr).
        //
        // Scoped to the result paths: the index being behind is reported honestly as
        // `stale` either way, but `can_trust_results` only goes false when a changed
        // file could have affected THIS answer. Without that, every search in an
        // ordinary edit-then-search loop would be flagged untrustworthy, and an agent
        // told to treat that as fatal could not use Reflex at all.
        let scope: Vec<String> = results.iter().map(|r| r.path.clone()).collect();
        let (status, can_trust_results, warning) = self.index_status_for(Some(&scope))?;
        let status_done = started.elapsed();

        // Build pagination metadata
        use crate::models::PaginationInfo;
        let pagination = PaginationInfo {
            // Never the verified-so-far number: that is not a total.
            total: total_is_exact.then_some(total),
            count: results.len(),
            offset: filter.offset.unwrap_or(0),
            limit: filter.limit,
            has_more: !total_is_exact || total > filter.offset.unwrap_or(0) + results.len(),
            total_is_exact,
            approx_total,
        };

        // Always use grouped format (group results by file)
        // Dependencies are loaded only when include_dependencies is true
        let grouped_results = self.group_and_load_dependencies(
            results,
            filter.include_dependencies,
            filter.context_lines,
        )?;

        let total_elapsed = started.elapsed();
        let timings = crate::models::QueryTimings {
            index_path,
            open_us,
            candidates_us,
            verify_us: (search_done.as_micros() as u64).saturating_sub(open_us + candidates_us),
            status_us: (status_done - search_done).as_micros() as u64,
            group_us: (total_elapsed - status_done).as_micros() as u64,
            total_us: total_elapsed.as_micros() as u64,
        };
        log::debug!("Query timings for '{}': {:?}", pattern, timings);

        // Only a whole-identifier search can be "explained"; a rewrite or an explicit
        // contains search already has substring semantics.
        let substring_hint_count = if total == 0 { substring_only } else { None };
        let hint = match substring_hint_count {
            Some(n) if n > 0 && prepared.warning.is_none() && !filter.use_contains => {
                Some(substring_hint_text(n, pattern))
            }
            _ => None,
        };

        Ok(QueryResponse {
            ai_instruction: None, // AI instruction is generated by CLI/MCP layer, not here
            status,
            can_trust_results,
            warning,
            pagination,
            results: grouped_results,
            substring_hint_count,
            warnings: prepared
                .warning
                .into_iter()
                .chain(engine_warnings)
                .collect(),
            hint,
            timings: filter.collect_timings.then_some(timings),
        })
    }

    /// Execute a query and return matching results (legacy method)
    ///
    /// This method prints warnings to stderr and returns just the results.
    /// For programmatic use, prefer `search_with_metadata()`.
    pub fn search(&self, pattern: &str, filter: QueryFilter) -> Result<Vec<SearchResult>> {
        log::info!(
            "Executing query: pattern='{}', filter={:?}",
            pattern,
            filter
        );

        // Ensure cache exists
        if !self.cache.exists() {
            return Err(crate::errors::ReflexError::IndexNotFound.into());
        }

        // Open (or reuse) the index; corruption surfaces as a typed error.
        self.open_index()?;

        // Show non-blocking warnings about branch state and staleness
        self.check_index_freshness(&filter)?;

        // Same bracket rewrite as `search_with_metadata`; this surface has nowhere to
        // report it, but it must not return a different answer.
        let prepared = prepare_literal_pattern(pattern, &filter);
        let mut filter = filter;
        if prepared.use_regex && !filter.use_regex {
            filter.rewritten_from = Some(pattern.to_string());
        }
        filter.use_regex = prepared.use_regex;

        // Execute the search (discard total count - legacy method doesn't use it)
        let mut results = self
            .search_internal(&prepared.effective, filter.clone())?
            .results;

        // Load dependencies if requested
        self.load_dependencies(&mut results, filter.include_dependencies)?;

        Ok(results)
    }

    /// Internal search implementation (used by both search methods)
    /// Returns (results, total_count) where total_count is the count before offset/limit
    fn search_internal(&self, pattern: &str, filter: QueryFilter) -> Result<Internal> {
        use std::time::{Duration, Instant};

        // Start timeout timer if configured
        let start_time = Instant::now();
        let timeout = if filter.timeout_secs > 0 {
            Some(Duration::from_secs(filter.timeout_secs))
        } else {
            None
        };

        // KEYWORD DETECTION (early): Check if this is a keyword query that should scan ALL files
        // When a user searches for a language keyword (like "class", "function") with --symbols or --kind,
        // we interpret it as "list all symbols of that type" and should scan ALL files,
        // not just the first 100 candidates from trigram search.
        //
        // Requirements for keyword query mode:
        // 1. Symbol mode active (--symbols or --kind)
        // 2. Pattern matches a keyword in ANY supported language
        //
        // Note: --lang is optional. If specified, language filtering happens naturally in Phase 2/3.
        // Empty pattern in symbol mode means "list all symbols of the requested kind" —
        // treat it like a keyword query so we scan all files instead of failing the
        // broad-query guard or returning zero trigram matches.
        let is_keyword_query = if filter.symbols_mode || filter.kind.is_some() {
            pattern.is_empty() || ParserFactory::get_all_keywords().contains(&pattern)
        } else {
            false
        };

        // KEYWORD-TO-KIND MAPPING: If user searches for a keyword without --kind, infer the kind
        // Example: "class" → SymbolKind::Class, "function" → SymbolKind::Function
        // This ensures keyword queries return only the relevant symbol type
        let mut filter = filter.clone(); // Clone so we can modify it
        if is_keyword_query
            && filter.kind.is_none()
            && let Some(inferred_kind) = Self::keyword_to_kind(pattern)
        {
            log::info!(
                "Keyword '{}' mapped to kind {:?} (auto-inferred)",
                pattern,
                inferred_kind
            );
            filter.kind = Some(inferred_kind);
        }

        // EARLY BROAD QUERY DETECTION (Index Size Check)
        // This check happens BEFORE the expensive trigram search to prevent hangs on large indexes
        // For very large codebases (like Linux kernel with 62K files), even valid 3-char trigrams
        // like "get" can take 10-30+ seconds to search. This early check prevents that hang.
        //
        // Criteria for early blocking:
        // 1. Large index (> 20,000 files) AND
        // 2. Short pattern (< 4 chars) AND
        // 3. Not using regex (regex has its own trigram extraction) — a regex the
        //    engine built from a literal (`ignore_case`, brackets) is judged as
        //    that literal AND
        // 4. Not a keyword query (keywords are intentionally broad) AND
        // 5. Not forced by --force flag
        let guarded_literal = filter.rewritten_from.as_deref();
        if !filter.force && (!filter.use_regex || guarded_literal.is_some()) && !is_keyword_query {
            // Index-wide file count from the open handle. `cache.stats()` here used
            // to open SQLite and spawn git on every query.
            let total_files = self.open_index()?.file_count();
            let pattern = guarded_literal.unwrap_or(pattern);
            let pattern_len = pattern.chars().count();

            // Thresholds for early blocking:
            // - Large index: 20,000+ files (approximately where performance degrades significantly)
            // - Short pattern: < 4 chars (3-char trigrams are borderline, < 4 catches edge cases)
            // Test overrides allow reducing thresholds for integration tests without creating 20K+ files
            let large_index_threshold = filter.test_large_index_threshold.unwrap_or(20_000);
            let short_pattern_threshold = filter.test_short_pattern_threshold.unwrap_or(4);

            if total_files > large_index_threshold && pattern_len < short_pattern_threshold {
                anyhow::bail!(
                    "Query too broad - would be expensive to execute on this large index\n\
                     \n\
                     This index contains {} files, and pattern '{}' ({} characters) is too short for efficient searching.\n\
                     On large codebases, short patterns can take 10-30+ seconds to complete.\n\
                     \n\
                     This query could:\n\
                     • Hang for an extended period before returning results\n\
                     • Return thousands of results\n\
                     • Flood LLM context windows with excessive data\n\
                     • Fail entirely\n\
                     \n\
                     Suggestions to narrow the query:\n\
                     • Use a longer, more specific pattern (4+ characters recommended for large indexes)\n\
                     • Add a language filter: --lang <language>\n\
                     • Add a file filter: --glob <pattern> or --file <path>\n\
                     • Use --force to bypass this check if you really need all results\n\
                     \n\
                     To force execution anyway:\n\
                     rfx query \"{}\" --force",
                    total_files,
                    pattern,
                    pattern_len,
                    pattern
                );
            }
        }

        // Early termination applies to plain list-mode text/regex searches: the
        // page is complete once `offset + limit` results exist, in order, and the
        // total is reported as inexact. Symbol and AST searches dedup and filter
        // after enrichment, so they verify everything; count and no-limit callers
        // (`mode:"count"`, `list_locations`, `--count`) pass no limit and also do.
        let budget = if filter.symbols_mode
            || filter.kind.is_some()
            || filter.use_ast
            || is_keyword_query
            || filter.require_exact_total
        {
            None
        } else {
            filter
                .limit
                .map(|limit| filter.offset.unwrap_or(0).saturating_add(limit))
        };

        // PHASE 1: Get initial candidates (choose search strategy)
        let mut substring_only = None;
        let mut candidates_us = 0u64;
        let mut total_is_exact = true;
        let mut approx_total = None;
        let index_path;
        let engine_warnings;
        let mut results = if is_keyword_query {
            // A keyword query bypasses the index on purpose.
            index_path = IndexPath::Scan;
            engine_warnings = Vec::new();
            // KEYWORD QUERY MODE: Scan all files (or files of target language if --lang specified)
            // This ensures we find ALL classes/functions/etc, not just those in the first 100 trigram matches
            if let Some(lang) = filter.language {
                log::info!(
                    "Keyword query detected for '{}' - scanning all {:?} files (bypassing trigram search)",
                    pattern,
                    lang
                );
            } else {
                log::info!(
                    "Keyword query detected for '{}' - scanning all files (bypassing trigram search)",
                    pattern
                );
            }
            self.get_all_language_files(&filter)?
        } else {
            let (candidates, stats) = if filter.use_regex {
                // Regex pattern search with trigram optimization
                self.get_regex_candidates(pattern, &filter, timeout.as_ref(), &start_time, budget)?
            } else {
                // Standard trigram-based full-text search
                self.get_trigram_candidates(pattern, &filter, budget)?
            };
            substring_only = stats.substring_only;
            candidates_us = stats.candidates_us;
            total_is_exact = stats.exhausted;
            approx_total = (!stats.exhausted).then_some(stats.approx_total).flatten();
            index_path = stats.index_path;
            engine_warnings = stats.warnings;
            candidates
        };

        // EARLY LANGUAGE FILTER: Apply language filtering BEFORE broad query check
        // This ensures we only parse files matching the language filter in Phase 2
        // Critical for non-keyword queries to work correctly with accurate candidate counts
        //
        // Skip for keyword queries - those candidates are already pre-filtered by language
        if !is_keyword_query && let Some(lang) = filter.language {
            let before_count = results.len();
            results.retain(|r| r.lang == lang);
            log::debug!(
                "Language filter ({:?}): reduced {} candidates to {} candidates",
                lang,
                before_count,
                results.len()
            );
        }

        // EARLY GLOB PATTERN FILTER: Apply glob/exclude filtering BEFORE broad query check
        // This ensures candidate count reflects actual files that will be parsed
        // Critical for queries like: rfx query "index" --symbols --glob "src/**/*.rs"
        if !filter.glob_patterns.is_empty() || !filter.exclude_patterns.is_empty() {
            // Build include matcher (if patterns specified)
            let include_matcher = result::build_glob_set(&filter.glob_patterns, "glob");

            // Build exclude matcher (if patterns specified)
            let exclude_matcher = result::build_glob_set(&filter.exclude_patterns, "exclude");

            // Apply filters
            let before_count = results.len();
            results.retain(|r| {
                // If include patterns specified, path must match at least one
                let included = if let Some(ref matcher) = include_matcher {
                    matcher.is_match(&r.path)
                } else {
                    true // No include patterns = include all
                };

                // If exclude patterns specified, path must NOT match any
                let excluded = if let Some(ref matcher) = exclude_matcher {
                    matcher.is_match(&r.path)
                } else {
                    false // No exclude patterns = exclude none
                };

                included && !excluded
            });
            log::debug!(
                "Glob filter: reduced {} candidates to {} candidates",
                before_count,
                results.len()
            );
        }

        // Check timeout after Phase 1
        if let Some(timeout_duration) = timeout
            && start_time.elapsed() > timeout_duration
        {
            anyhow::bail!(
                "Query timeout exceeded ({} seconds).\n\
                     \n\
                     The query took too long to complete. Try one of these approaches:\n\
                     • Use a more specific search pattern (longer patterns = faster search)\n\
                     • Add a language filter with --lang to narrow the search space\n\
                     • Add a file filter with --file to search specific directories\n\
                     • Increase the timeout with --timeout <seconds>\n\
                     \n\
                     Example: rfx query \"{}\" --lang rust --timeout 60",
                filter.timeout_secs,
                pattern
            );
        }

        // BROAD QUERY DETECTION: Check if query is too expensive BEFORE parsing
        // This protects LLM users from accidentally running expensive queries that flood context windows
        if !filter.force {
            let candidate_count = results.len();
            let pattern_len = pattern.chars().count();

            // Condition 1: Pattern too short (< 3 chars can't use trigram optimization efficiently)
            // Exception: Allow short keyword queries (e.g., "fn", "if") since they scan all language files
            let is_short_pattern = pattern_len < 3 && !filter.use_regex && !is_keyword_query;

            // Condition 2: AST query without glob restriction on large codebases
            // Allow on small codebases (< 100 files) but require glob for larger ones
            let is_broad_ast =
                filter.use_ast && filter.glob_patterns.is_empty() && candidate_count >= 100;

            // Condition 3: Query-type-aware threshold for symbol/AST parsing
            // Different thresholds based on actual performance characteristics:
            // - AST without glob: 100 files (allow small codebases, block large ones)
            // - AST with glob: 10,000 files (~5 seconds max)
            // - Keyword queries: 20,000 files (~3 seconds max) - scan all files of language
            // - Trigram-filtered symbols: 50,000 files (~5 seconds max) - very fast due to trigram filtering
            let threshold = if filter.use_ast && filter.glob_patterns.is_empty() {
                100 // AST without glob - allow small codebases
            } else if filter.use_ast {
                10_000 // AST with glob restriction
            } else if is_keyword_query {
                20_000 // Keyword queries (e.g., "class", "function")
            } else {
                50_000 // Trigram-filtered symbol queries
            };

            let has_many_candidates = candidate_count > threshold
                && (filter.symbols_mode || filter.kind.is_some() || filter.use_ast);

            if is_short_pattern || has_many_candidates || is_broad_ast {
                let reason = if is_short_pattern {
                    format!(
                        "Pattern '{}' is too short ({} characters). Short patterns bypass trigram optimization and require scanning many files.",
                        pattern, pattern_len
                    )
                } else if is_broad_ast {
                    format!(
                        "AST query without --glob restriction will scan the entire codebase ({} files). AST queries are SLOW (500ms-10s+).",
                        candidate_count
                    )
                } else if is_keyword_query {
                    format!(
                        "Keyword query '{}' matched {} files. This query scans all files of the target language, which will take significant time and produce excessive results.",
                        pattern, candidate_count
                    )
                } else {
                    format!(
                        "Query matched {} files. Parsing this many files with --symbols or --kind will take significant time and produce excessive results.",
                        candidate_count
                    )
                };

                let suggestions = if is_short_pattern {
                    vec![
                        "• Use a longer, more specific pattern (3+ characters recommended)",
                        "• Add a language filter: --lang <language>",
                        "• Add a file path filter: --file <path> or --glob <pattern>",
                        "• Use --force to bypass this check if you really need all results",
                    ]
                } else if is_broad_ast {
                    vec![
                        "• Add --glob to restrict AST query to specific files: --glob 'src/**/*.rs'",
                        "• Use --symbols instead (10-100x faster in 95% of cases)",
                        "• Use --force to bypass this check if you need a full codebase scan",
                    ]
                } else if is_keyword_query {
                    vec![
                        "• Add a language filter to reduce files scanned: --lang <language>",
                        "• Add glob patterns to search specific directories: --glob 'src/**/*.rs'",
                        "• Add --kind to filter to specific symbol types: --kind function",
                        "• Use a more specific pattern instead of a keyword",
                        "• Use --force to bypass this check if you need all results",
                    ]
                } else {
                    vec![
                        "• Add a language filter to reduce candidate set: --lang <language>",
                        "• Add glob patterns to search specific directories: --glob 'src/**/*.rs'",
                        "• Use a more specific search pattern",
                        "• Use --force to bypass this check if you need all results",
                    ]
                };

                // Build the command snippet showing current flags
                let mut cmd_flags = String::new();
                if filter.symbols_mode {
                    cmd_flags.push_str("--symbols ");
                }
                if let Some(ref lang) = filter.language {
                    cmd_flags.push_str(&format!("--lang {:?} ", lang));
                }
                if let Some(ref kind) = filter.kind {
                    cmd_flags.push_str(&format!("--kind {:?} ", kind));
                }
                if filter.use_ast {
                    cmd_flags.push_str("--ast ");
                }

                anyhow::bail!(
                    "Query too broad - would be expensive to execute\n\
                     \n\
                     {}\n\
                     \n\
                     This query could:\n\
                     • Hang for an extended period before returning results\n\
                     • Return thousands of results\n\
                     • Flood LLM context windows with excessive data\n\
                     • Fail entirely\n\
                     \n\
                     Suggestions to narrow the query:\n\
                     {}\n\
                     \n\
                     To force execution anyway:\n\
                     rfx query \"{}\" --force {}",
                    reason,
                    suggestions.join("\n             "),
                    pattern,
                    cmd_flags
                );
            }
        }

        // DETERMINISTIC SORTING: Sort candidates early for deterministic results
        // This ensures results are always returned in the same order
        if filter.symbols_mode || filter.kind.is_some() || filter.use_ast {
            results.sort_by(|a, b| {
                a.path
                    .cmp(&b.path)
                    .then_with(|| a.span.start_line.cmp(&b.span.start_line))
            });

            // Warn if many candidates need parsing (helps users refine queries)
            let candidate_count = results.len();
            if candidate_count > 1000 && !filter.suppress_output {
                output::warn(&format!(
                    "Pattern '{}' matched {} files - parsing may take some time. Consider using --file, --glob, or a more specific pattern to narrow the search.",
                    pattern, candidate_count
                ));
            } else if candidate_count > 100 {
                log::info!(
                    "Parsing {} candidate files for symbol extraction",
                    candidate_count
                );
            }
        }

        // PHASE 2: Enrich with symbol information or AST pattern matching (if needed)
        if filter.use_ast {
            // AST pattern matching: Execute Tree-sitter query on candidate files
            results = self.enrich_with_ast(results, pattern, filter.language)?;
        } else if filter.symbols_mode || filter.kind.is_some() {
            // Symbol enrichment: Parse candidate files and extract symbol definitions
            results = self.enrich_with_symbols(results, pattern, &filter, is_keyword_query)?;
        }

        // PHASE 3: Apply post-enrichment filters
        // Note: Language and glob filters are applied in Phase 1 (before broad query check)
        // Only kind, file_pattern, and exact filters are applied here

        // Deduplicate symbols: the same source location can be emitted as both
        // Function and Method by some parsers.  Keep the first hit for each
        // (path, start_line, symbol_name) triple so --kind function doesn't
        // return the same definition twice.
        if filter.symbols_mode || filter.kind.is_some() {
            let mut seen = std::collections::HashSet::<(String, usize, Option<String>)>::new();
            results.retain(|r| seen.insert((r.path.clone(), r.span.start_line, r.symbol.clone())));
        }

        // Apply kind filter (only relevant for symbol searches)
        // Special case: --kind function also includes methods (methods are functions in classes)
        if let Some(ref kind) = filter.kind {
            results.retain(|r| {
                if matches!(kind, SymbolKind::Function) {
                    // When searching for functions, also include methods
                    matches!(r.kind, SymbolKind::Function | SymbolKind::Method)
                } else {
                    r.kind == *kind
                }
            });
        }

        // Apply file path filter (substring match)
        if let Some(ref file_pattern) = filter.file_pattern {
            results.retain(|r| r.path.contains(file_pattern));
        }

        // Apply exact name filter (only for symbol searches)
        if filter.exact && filter.symbols_mode {
            results.retain(|r| r.symbol.as_deref() == Some(pattern));
        }

        // Expand symbol bodies if requested
        // Works for both symbol-mode and regex searches (if regex matched a symbol definition)
        if filter.expand {
            // Fetch full symbol bodies from the shared content store
            if let Ok(open) = self.open_index() {
                let content_reader = &open.content;
                for result in &mut results {
                    // Only expand if the result has a meaningful span (not just a single line)
                    if result.span.start_line < result.span.end_line {
                        // Find the file_id for this result's path
                        if let Some(file_id) = open.file_id_for(&result.path) {
                            // Fetch the full span content
                            if let Ok(content) = content_reader.get_file_content(file_id) {
                                let lines: Vec<&str> = content.lines().collect();
                                let start_idx = result.span.start_line.saturating_sub(1);
                                let end_idx = result.span.end_line.min(lines.len());

                                if start_idx < end_idx {
                                    let full_body = lines[start_idx..end_idx].join("\n");
                                    // Expand shows more than a preview, but not an
                                    // unbounded amount: on a minified file this is
                                    // the entire bundle.
                                    result.preview =
                                        crate::parsers::preview::expand_preview(&full_body);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Step 4: Deduplicate by path if paths-only mode
        if filter.paths_only {
            use std::collections::HashSet;
            let mut seen_paths = HashSet::new();
            results.retain(|r| seen_paths.insert(r.path.clone()));
        }

        // Drop text-tier results when the caller asked for code only. Applied BEFORE
        // the total is captured, so pagination counts what the caller will actually
        // receive rather than what was found and then discarded.
        if filter.exclude_text {
            results.retain(|r| !r.lang.is_text());
        }

        // Step 5: Sort results deterministically (by path, then line number)
        results.sort_by(|a, b| {
            a.path
                .cmp(&b.path)
                .then_with(|| a.span.start_line.cmp(&b.span.start_line))
        });

        // Capture total count AFTER all filtering but BEFORE pagination (offset/limit)
        // This is the total number of results the user can paginate through
        let total_count = results.len();

        // Step 5.5: Apply offset (pagination)
        if let Some(offset) = filter.offset {
            if offset < results.len() {
                results = results.into_iter().skip(offset).collect();
            } else {
                // Offset beyond results - return empty
                results.clear();
            }
        }

        // Step 6: Apply limit
        if let Some(limit) = filter.limit {
            results.truncate(limit);
        }

        log::info!(
            "Query returned {} results (total before pagination: {})",
            results.len(),
            total_count
        );

        Ok(Internal {
            results,
            total_count,
            total_is_exact,
            approx_total,
            substring_only,
            candidates_us,
            index_path,
            warnings: engine_warnings,
        })
    }

    /// Search for symbols by exact name match
    pub fn find_symbol(&self, name: &str) -> Result<Vec<SearchResult>> {
        let filter = QueryFilter {
            symbols_mode: true,
            ..Default::default()
        };
        self.search(name, filter)
    }

    /// Search using a Tree-sitter AST pattern
    pub fn search_ast(&self, pattern: &str, lang: Option<Language>) -> Result<Vec<SearchResult>> {
        let filter = QueryFilter {
            language: lang,
            use_ast: true,
            ..Default::default()
        };

        self.search(pattern, filter)
    }

    /// Execute AST query on all indexed files (no trigram filtering)
    ///
    /// WARNING: This method scans the entire codebase (500ms-2s+).
    /// In 95% of cases, use --symbols instead which is 10-100x faster.
    ///
    /// # Algorithm
    /// 1. Get all indexed files for the specified language
    /// 2. Apply glob/exclude filters to reduce file set
    /// 3. Load file contents for all matching files
    /// 4. Execute AST query pattern using Tree-sitter
    /// 5. Apply remaining filters and return results
    ///
    /// # Performance
    /// - Parses entire codebase (not just trigram candidates)
    /// - Expected: 500ms-2s for medium codebases, 2-10s for large codebases
    /// - Use --glob to limit scope for better performance
    ///
    /// # Requirements
    /// - Language must be specified (AST queries are language-specific)
    /// - AST pattern must be valid S-expression syntax
    pub fn search_ast_all_files(
        &self,
        ast_pattern: &str,
        filter: QueryFilter,
    ) -> Result<Vec<SearchResult>> {
        log::info!(
            "Executing AST query on all files: pattern='{}', filter={:?}",
            ast_pattern,
            filter
        );

        // Require language for AST queries
        let lang = filter.language.ok_or_else(|| anyhow::anyhow!(
            "Language must be specified for AST pattern matching. Use --lang to specify the language.\n\
             \n\
             Example: rfx query \"(function_definition) @fn\" --ast --lang python"
        ))?;

        // Ensure cache exists
        if !self.cache.exists() {
            return Err(crate::errors::ReflexError::IndexNotFound.into());
        }

        // Show non-blocking warnings about branch state and staleness
        self.check_index_freshness(&filter)?;

        // The shared content store
        let open = self.open_index()?;
        let content_reader = &open.content;

        // Build glob matchers ONCE before file iteration (performance optimization)
        let include_matcher = result::build_glob_set(&filter.glob_patterns, "glob");

        let exclude_matcher = result::build_glob_set(&filter.exclude_patterns, "exclude");

        // Get all files matching the language and glob filters
        let mut candidates: Vec<SearchResult> = Vec::new();

        for file_id in 0..content_reader.file_count() {
            let file_path = match content_reader.get_file_path(file_id as u32) {
                Some(p) => p,
                None => continue,
            };

            // Detect language from file extension
            let detected_lang = Language::from_path(file_path);

            // Filter by language
            if detected_lang != lang {
                continue;
            }

            let file_path_str = file_path.to_string_lossy().to_string();

            // Apply glob/exclude filters BEFORE loading content (performance optimization)
            let included = include_matcher
                .as_ref()
                .is_none_or(|m| m.is_match(&file_path_str));
            let excluded = exclude_matcher
                .as_ref()
                .is_some_and(|m| m.is_match(&file_path_str));

            if !included || excluded {
                continue;
            }

            // Create a dummy candidate for this file (AST query will replace it)
            candidates.push(SearchResult {
                path: file_path_str,
                lang: detected_lang,
                span: Span {
                    start_line: 1,
                    end_line: 1,
                },
                symbol: None,
                kind: SymbolKind::Unknown("ast_query".to_string()),
                preview: String::new(),
                dependencies: None,
            });
        }

        log::info!(
            "AST query scanning {} files for language {:?}",
            candidates.len(),
            lang
        );

        // BROAD QUERY DETECTION: Block large AST queries without glob restriction
        // Allow small codebases (<100 files) but require --glob for larger ones
        if !filter.force && filter.glob_patterns.is_empty() && candidates.len() >= 100 {
            anyhow::bail!(
                "Query too broad - would be expensive to execute\n\
                 \n\
                 AST query without --glob restriction will scan the ENTIRE codebase ({} files). AST queries are SLOW (500ms-10s+).\n\
                 \n\
                 This query could:\n\
                 • Hang for an extended period before returning results\n\
                 • Return thousands of results\n\
                 • Flood LLM context windows with excessive data\n\
                 • Fail entirely\n\
                 \n\
                 Suggestions to narrow the query:\n\
                 • Add --glob to restrict AST query to specific files: --glob 'src/**/*.rs'\n\
                 • Use --symbols instead (10-100x faster in 95% of cases)\n\
                 • Use --force to bypass this check if you need a full codebase scan\n\
                 \n\
                 To force execution anyway:\n\
                 rfx query \"{}\" --force --ast --lang {:?}",
                candidates.len(),
                ast_pattern,
                lang
            );
        }

        if candidates.is_empty() {
            if !filter.suppress_output {
                output::warn(&format!(
                    "No files found for language {:?}. Check your language filter or glob patterns.",
                    lang
                ));
            }
            return Ok(Vec::new());
        }

        // Execute the AST query on all candidate files
        // This will load file contents and parse them with tree-sitter
        let mut results = self.enrich_with_ast(candidates, ast_pattern, filter.language)?;

        log::debug!("AST query found {} matches before filtering", results.len());

        // Apply remaining filters (same as search_internal Phase 3)

        // Apply kind filter
        if let Some(ref kind) = filter.kind {
            results.retain(|r| {
                if matches!(kind, SymbolKind::Function) {
                    matches!(r.kind, SymbolKind::Function | SymbolKind::Method)
                } else {
                    r.kind == *kind
                }
            });
        }

        // Note: exact filter doesn't make sense for AST queries (pattern is S-expression, not symbol name)

        // Expand symbol bodies if requested
        if filter.expand
            && let Ok(open) = self.open_index()
        {
            let content_reader = &open.content;
            {
                for result in &mut results {
                    if result.span.start_line < result.span.end_line
                        && let Some(file_id) = open.file_id_for(&result.path)
                        && let Ok(content) = content_reader.get_file_content(file_id)
                    {
                        let lines: Vec<&str> = content.lines().collect();
                        let start_idx = result.span.start_line.saturating_sub(1);
                        let end_idx = result.span.end_line.min(lines.len());

                        if start_idx < end_idx {
                            let full_body = lines[start_idx..end_idx].join("\n");
                            // Bounded: on a minified file this body is the whole bundle.
                            result.preview = crate::parsers::preview::expand_preview(&full_body);
                        }
                    }
                }
            }
        }

        // Deduplicate by path if paths-only mode
        if filter.paths_only {
            use std::collections::HashSet;
            let mut seen_paths = HashSet::new();
            results.retain(|r| seen_paths.insert(r.path.clone()));
        }

        // Drop text-tier results when the caller asked for code only. Applied BEFORE
        // the total is captured, so pagination counts what the caller will actually
        // receive rather than what was found and then discarded.
        if filter.exclude_text {
            results.retain(|r| !r.lang.is_text());
        }

        // Sort results deterministically
        results.sort_by(|a, b| {
            a.path
                .cmp(&b.path)
                .then_with(|| a.span.start_line.cmp(&b.span.start_line))
        });

        // Apply offset (pagination)
        if let Some(offset) = filter.offset {
            if offset < results.len() {
                results = results.into_iter().skip(offset).collect();
            } else {
                results.clear();
            }
        }

        // Apply limit
        if let Some(limit) = filter.limit {
            results.truncate(limit);
        }

        log::info!("AST query returned {} results", results.len());

        // Load dependencies if requested
        self.load_dependencies(&mut results, filter.include_dependencies)?;

        Ok(results)
    }

    /// Search using AST pattern with separate text pattern for trigram filtering
    ///
    /// This allows efficient AST queries by:
    /// 1. Using text_pattern for Phase 1 trigram filtering (narrows to candidate files)
    /// 2. Using ast_pattern for Phase 2 AST matching (structure-aware filtering)
    ///
    /// # Example
    /// ```ignore
    /// // Find async functions: trigram search for "fn ", AST match for function_item
    /// engine.search_ast_with_text_filter("fn ", "(function_item (async))", filter)?;
    /// ```
    pub fn search_ast_with_text_filter(
        &self,
        text_pattern: &str,
        ast_pattern: &str,
        filter: QueryFilter,
    ) -> Result<Vec<SearchResult>> {
        log::info!(
            "Executing AST query with text filter: text='{}', ast='{}', filter={:?}",
            text_pattern,
            ast_pattern,
            filter
        );

        // Ensure cache exists
        if !self.cache.exists() {
            return Err(crate::errors::ReflexError::IndexNotFound.into());
        }

        // Show non-blocking warnings about branch state and staleness
        self.check_index_freshness(&filter)?;

        // Start timeout timer if configured
        use std::time::{Duration, Instant};
        let start_time = Instant::now();
        let timeout = if filter.timeout_secs > 0 {
            Some(Duration::from_secs(filter.timeout_secs))
        } else {
            None
        };

        // PHASE 1: Get initial candidates using text pattern (trigram search)
        let candidates = if filter.use_regex {
            self.get_regex_candidates(text_pattern, &filter, timeout.as_ref(), &start_time, None)?
                .0
        } else {
            self.get_trigram_candidates(text_pattern, &filter, None)?.0
        };

        log::debug!("Phase 1 found {} candidate locations", candidates.len());

        // PHASE 2: Execute AST query on candidates
        let mut results = self.enrich_with_ast(candidates, ast_pattern, filter.language)?;

        log::debug!("Phase 2 AST matching found {} results", results.len());

        // PHASE 3: Apply filters
        if let Some(lang) = filter.language {
            results.retain(|r| r.lang == lang);
        }

        if let Some(ref kind) = filter.kind {
            results.retain(|r| {
                if matches!(kind, SymbolKind::Function) {
                    matches!(r.kind, SymbolKind::Function | SymbolKind::Method)
                } else {
                    r.kind == *kind
                }
            });
        }

        if let Some(ref file_pattern) = filter.file_pattern {
            results.retain(|r| r.path.contains(file_pattern));
        }

        // Apply glob pattern filters (same logic as in search_internal)
        if !filter.glob_patterns.is_empty() || !filter.exclude_patterns.is_empty() {
            let include_matcher = result::build_glob_set(&filter.glob_patterns, "glob");

            let exclude_matcher = result::build_glob_set(&filter.exclude_patterns, "exclude");

            results.retain(|r| {
                let included = include_matcher.as_ref().is_none_or(|m| m.is_match(&r.path));
                let excluded = exclude_matcher
                    .as_ref()
                    .is_some_and(|m| m.is_match(&r.path));
                included && !excluded
            });
        }

        if filter.exact && filter.symbols_mode {
            results.retain(|r| r.symbol.as_deref() == Some(text_pattern));
        }

        // Expand symbol bodies if requested
        if filter.expand
            && let Ok(open) = self.open_index()
        {
            let content_reader = &open.content;
            {
                for result in &mut results {
                    if result.span.start_line < result.span.end_line
                        && let Some(file_id) = open.file_id_for(&result.path)
                        && let Ok(content) = content_reader.get_file_content(file_id)
                    {
                        let lines: Vec<&str> = content.lines().collect();
                        let start_idx = result.span.start_line.saturating_sub(1);
                        let end_idx = result.span.end_line.min(lines.len());

                        if start_idx < end_idx {
                            let full_body = lines[start_idx..end_idx].join("\n");
                            // Bounded: on a minified file this body is the whole bundle.
                            result.preview = crate::parsers::preview::expand_preview(&full_body);
                        }
                    }
                }
            }
        }

        // Drop text-tier results when the caller asked for code only. Applied BEFORE
        // the total is captured, so pagination counts what the caller will actually
        // receive rather than what was found and then discarded.
        if filter.exclude_text {
            results.retain(|r| !r.lang.is_text());
        }

        // Sort results deterministically
        results.sort_by(|a, b| {
            a.path
                .cmp(&b.path)
                .then_with(|| a.span.start_line.cmp(&b.span.start_line))
        });

        // Apply offset (pagination)
        if let Some(offset) = filter.offset {
            if offset < results.len() {
                results = results.into_iter().skip(offset).collect();
            } else {
                results.clear();
            }
        }

        // Apply limit
        if let Some(limit) = filter.limit {
            results.truncate(limit);
        }

        log::info!("AST query returned {} results", results.len());

        Ok(results)
    }

    /// List all symbols of a specific kind
    pub fn list_by_kind(&self, kind: SymbolKind) -> Result<Vec<SearchResult>> {
        let filter = QueryFilter {
            kind: Some(kind),
            symbols_mode: true,
            ..Default::default()
        };

        self.search("*", filter)
    }

    /// Enrich text match candidates with symbol information by parsing files
    ///
    /// Takes a list of text match candidates and extracts symbol information at those locations.
    ///
    /// # Algorithm
    /// 1. Group candidates by file_id for efficient processing
    /// 2. Parse each file with tree-sitter to extract ALL symbols
    /// 3. Filter symbols based on matching strategy:
    ///    - If use_regex=true: Extract symbols whose line spans overlap with candidate locations
    ///    - If use_contains=true: Filter symbols by substring match on symbol name
    ///    - Default: Filter symbols by exact name match
    /// 4. Return filtered symbol results
    ///
    /// # Performance
    /// Only parses files that have text matches, so typically 10-100 files
    /// instead of the entire codebase (62K+ files).
    ///
    /// # Optimizations
    /// 1. Language filtering: Skips files with unsupported languages (no parsers)
    /// 2. Parallel processing: Uses Rayon to parse files concurrently across CPU cores
    fn enrich_with_symbols(
        &self,
        candidates: Vec<SearchResult>,
        pattern: &str,
        filter: &QueryFilter,
        keyword_query: bool,
    ) -> Result<Vec<SearchResult>> {
        use rayon::prelude::*;
        use std::collections::{HashMap, HashSet};

        // The shared index handle (content store + file-id map + meta.db connection)
        let open = self.open_index()?;
        let content_reader = &open.content;

        // Group candidates by file, filtering out unsupported languages
        let mut files_by_path: HashMap<String, Vec<SearchResult>> = HashMap::new();
        let mut skipped_unsupported = 0;

        for candidate in candidates {
            // Skip files with unsupported languages (no parser available)
            if !candidate.lang.is_supported() {
                skipped_unsupported += 1;
                continue;
            }

            files_by_path
                .entry(candidate.path.clone())
                .or_default()
                .push(candidate);
        }

        let total_files = files_by_path.len();
        log::debug!(
            "Processing {} candidate files for symbol enrichment (skipped {} unsupported language files)",
            total_files,
            skipped_unsupported
        );

        // Warn if pattern is very broad (may take time to parse all files)
        if total_files > 1000 && !filter.suppress_output {
            output::warn(&format!(
                "Pattern '{}' matched {} files. This may take some time to parse. Consider using a more specific pattern or adding --lang/--file filters to narrow the search.",
                pattern, total_files
            ));
        }

        // Parse on the shared query pool, sized from `[performance] parallel_threads`
        let pool = open.pool();

        // PHASE 2a: pre-filter — skip files where EVERY occurrence of the pattern is
        // inside a comment or a string literal, so tree-sitter never parses them.
        //
        // Only the candidate lines are examined: the trigram pass already verified
        // that those are the lines holding the pattern, so scanning the whole file
        // (as this did before 1.8.0, serially) found nothing more. A definition
        // line is always a candidate line, so a file that defines the symbol is
        // never skipped. Two cases keep the old outcome exactly:
        // * keyword queries carry dummy line-1 candidates from
        //   `get_all_language_files`, so there is nothing to examine;
        // * a regex pattern's source text never occurs literally, so the old
        //   `line.find(pattern)` skipped nothing.
        let files_to_skip: HashSet<String> = if keyword_query || filter.use_regex {
            HashSet::new()
        } else {
            pool.install(|| {
                files_by_path
                    .par_iter()
                    .filter_map(|(file_path, cands)| {
                        let lang = Language::from_path(std::path::Path::new(file_path));
                        let line_filter = crate::line_filter::get_filter(lang)?;
                        let file_id = open.file_id_for(file_path)?;
                        let content = content_reader.get_file_content(file_id).ok()?;
                        let all_lines: Vec<&str> = content.lines().collect();

                        let mut wanted: Vec<usize> =
                            cands.iter().map(|c| c.span.start_line).collect();
                        wanted.sort_unstable();
                        wanted.dedup();

                        let mut saw_occurrence = false;
                        for line_no in wanted {
                            let Some(line) = line_no.checked_sub(1).and_then(|i| all_lines.get(i))
                            else {
                                continue;
                            };
                            let mut search_start = 0;
                            while let Some(pos) = line[search_start..].find(pattern) {
                                saw_occurrence = true;
                                let at = search_start + pos;
                                if !line_filter.is_in_comment(line, at)
                                    && !line_filter.is_in_string(line, at)
                                {
                                    // In code: this file must be parsed.
                                    return None;
                                }
                                search_start = at + pattern.len();
                            }
                        }
                        // Only skip when there WAS an occurrence and none was in code.
                        saw_occurrence.then(|| {
                            log::debug!(
                                "Pre-filter: Skipping {} (all matches in comments/strings)",
                                file_path
                            );
                            file_path.clone()
                        })
                    })
                    .collect()
            })
        };

        let files_to_process: Vec<String> = files_by_path
            .keys()
            .filter(|p| !files_to_skip.contains(p.as_str()))
            .cloned()
            .collect();

        log::debug!(
            "Pre-filter: Skipped {} files where all matches are in comments/strings (parsing {} files)",
            files_to_skip.len(),
            files_to_process.len()
        );

        // Symbol cache lookup, on the handle's shared connection.
        //
        // The branch comes from `.git/HEAD` (no subprocess); the hashes come from a
        // query restricted to the candidate paths (not the whole branch).
        let root = self.cache.workspace_root();
        let branch = crate::git::read_head_branch(&root).unwrap_or_else(|| "_default".to_string());
        let mut conn = open.meta_conn()?;
        let rows =
            crate::cache::CacheManager::branch_file_rows_on(&conn, &branch, &files_to_process)
                .context("Failed to load file hashes")?;
        log::debug!(
            "Loaded {} file rows for branch '{}' for symbol cache lookups",
            rows.len(),
            branch
        );

        let file_lookup_tuples: Vec<(i64, String, String)> = files_to_process
            .iter()
            .filter_map(|path| {
                let (id, hash) = rows.get(path)?;
                Some((*id, hash.clone(), path.clone()))
            })
            .collect();

        let batch_results = crate::symbol_cache::SymbolCache::batch_get_with_kind_on(
            &conn,
            &file_lookup_tuples,
            filter.kind.clone(),
        )
        .context("Failed to batch read symbol cache")?;

        let id_to_path: HashMap<i64, &str> = rows
            .iter()
            .map(|(path, (id, _))| (*id, path.as_str()))
            .collect();

        let mut cached_symbols: HashMap<String, Vec<SearchResult>> = HashMap::new();
        for (file_id, symbols) in batch_results {
            if let Some(file_path) = id_to_path.get(&file_id) {
                cached_symbols.insert(file_path.to_string(), symbols);
            }
        }

        // Everything else — no row on this branch, or no (current-hash) cache entry.
        let files_needing_parse: Vec<String> = files_to_process
            .iter()
            .filter(|p| !cached_symbols.contains_key(p.as_str()))
            .cloned()
            .collect();

        log::debug!(
            "Symbol cache: {} hits, {} need parsing",
            cached_symbols.len(),
            files_needing_parse.len()
        );

        // Parse cache misses in parallel; cache writes are collected and committed in
        // ONE transaction afterwards, instead of one connection + INSERT per file
        // from inside the pool.
        /// Symbols parsed from one file, plus its `(file_id, hash)` cache key when known.
        type ParsedFile = (Vec<SearchResult>, Option<(i64, String)>);
        let parsed: Vec<ParsedFile> = pool.install(|| {
            files_needing_parse
                .par_iter()
                .map(|file_path| {
                    let Some(file_id) = open.file_id_for(file_path) else {
                        log::warn!("Could not find file_id for path: {}", file_path);
                        return (Vec::new(), None);
                    };
                    let content = match content_reader.get_file_content(file_id) {
                        Ok(c) => c,
                        Err(e) => {
                            log::warn!("Failed to read file {}: {}", file_path, e);
                            return (Vec::new(), None);
                        }
                    };
                    let lang = Language::from_path(std::path::Path::new(file_path));
                    let symbols = match ParserFactory::parse(file_path, content, lang) {
                        Ok(symbols) => {
                            log::debug!("Parsed {} symbols from {}", symbols.len(), file_path);
                            symbols
                        }
                        Err(e) => {
                            log::debug!("Failed to parse {}: {}", file_path, e);
                            Vec::new()
                        }
                    };
                    let key = rows.get(file_path).map(|(id, hash)| (*id, hash.clone()));
                    (symbols, key)
                })
                .collect()
        });

        let mut parsed_symbols: Vec<SearchResult> = Vec::new();
        let mut to_cache: Vec<(i64, String, Vec<SearchResult>)> = Vec::new();
        for (symbols, key) in parsed {
            if let Some((id, hash)) = key {
                to_cache.push((id, hash, symbols.clone()));
            }
            parsed_symbols.extend(symbols);
        }
        // Best-effort: a failed cache write must never fail the query.
        if let Err(e) = crate::symbol_cache::SymbolCache::batch_set_by_id_on(&mut conn, &to_cache) {
            log::debug!(
                "Failed to cache symbols for {} files: {}",
                to_cache.len(),
                e
            );
        }
        drop(conn);

        // Combine cached and parsed symbols
        let mut all_symbols: Vec<SearchResult> = Vec::new();

        // Add all cached symbols
        for symbols in cached_symbols.values() {
            all_symbols.extend_from_slice(symbols);
        }

        // Add all parsed symbols
        all_symbols.extend(parsed_symbols);

        // KEYWORD DETECTION: Check if pattern is a language keyword (e.g., "class", "function")
        // If it matches a keyword AND symbols_mode is true, interpret as "list all symbols of that type"
        // rather than looking for a symbol literally named "class" or "function"
        //
        // IMPORTANT: Only check keywords for languages that will pass Phase 3 filtering.
        // If a language filter is specified, only check that language's keywords.
        // Otherwise, check all languages present in the symbol results.
        let is_keyword_query = {
            // Determine which language to check keywords for
            let lang_to_check = if let Some(lang) = filter.language {
                // Language filter specified - check that language only
                // This ensures keyword detection aligns with Phase 3 language filtering
                vec![lang]
            } else {
                // No language filter - check all languages that appear in the actual symbols
                // (not candidates, but the parsed symbols that made it through)
                // This handles mixed-language codebases correctly
                let mut langs: Vec<Language> =
                    all_symbols.iter().map(|s| s.lang).collect::<Vec<_>>();
                langs.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b))); // Deterministic ordering
                langs.dedup(); // Remove duplicates after sorting
                langs
            };

            // Check if pattern matches a keyword in any of the relevant languages
            lang_to_check
                .iter()
                .any(|lang| ParserFactory::get_keywords(*lang).contains(&pattern))
        };

        // If pattern is a keyword (like "class" or "function"), skip name-based filtering
        // and return all symbols (kind filtering happens in Phase 3)
        let filtered: Vec<SearchResult> = if is_keyword_query {
            log::info!(
                "Pattern '{}' is a language keyword - listing all symbols (kind filtering will be applied in Phase 3)",
                pattern
            );
            all_symbols
        } else if filter.use_regex {
            // For regex queries, candidates already matched content via regex in Phase 1.
            // Extract symbols whose line spans overlap with the candidate locations.
            // This ensures symbols are found at the locations where the regex matched.

            // Build a map of (file_path, line_no) from candidates
            use std::collections::{HashMap, HashSet};
            let mut candidate_lines: HashMap<String, HashSet<usize>> = HashMap::new();
            for candidate in &files_by_path {
                for cand in candidate.1 {
                    candidate_lines
                        .entry(candidate.0.clone())
                        .or_default()
                        .insert(cand.span.start_line);
                }
            }

            // Filter symbols whose spans overlap with candidate lines
            all_symbols
                .into_iter()
                .filter(|sym| {
                    if let Some(lines) = candidate_lines.get(&sym.path) {
                        // Check if symbol's line span overlaps with any candidate line
                        for line in sym.span.start_line..=sym.span.end_line {
                            if lines.contains(&line) {
                                return true;
                            }
                        }
                    }
                    false
                })
                .collect()
        } else if filter.use_contains {
            // Substring match (opt-in with --contains)
            all_symbols
                .into_iter()
                .filter(|sym| sym.symbol.as_deref().is_some_and(|s| s.contains(pattern)))
                .collect()
        } else {
            // Exact match (default)
            all_symbols
                .into_iter()
                .filter(|sym| sym.symbol.as_deref() == Some(pattern))
                .collect()
        };

        log::info!(
            "Symbol enrichment found {} matches for pattern '{}'",
            filtered.len(),
            pattern
        );

        Ok(filtered)
    }

    /// Enrich text match candidates with AST pattern matching
    ///
    /// Takes a list of text match candidates and executes a Tree-sitter AST query
    /// on the candidate files, returning only matches that satisfy the AST pattern.
    ///
    /// # Algorithm
    /// 1. Extract unique file paths from candidates
    /// 2. Load file contents for each candidate file
    /// 3. Execute AST query pattern using Tree-sitter
    /// 4. Return AST matches
    ///
    /// # Performance
    /// Only parses files that have text matches, so typically 10-100 files
    /// instead of the entire codebase (62K+ files).
    ///
    /// # Requirements
    /// - Language must be specified (AST queries are language-specific)
    /// - AST pattern must be valid S-expression syntax
    fn enrich_with_ast(
        &self,
        candidates: Vec<SearchResult>,
        ast_pattern: &str,
        language: Option<Language>,
    ) -> Result<Vec<SearchResult>> {
        // Require language for AST queries
        let lang = language.ok_or_else(|| anyhow::anyhow!(
            "Language must be specified for AST pattern matching. Use --lang to specify the language."
        ))?;

        // The shared index handle (content store + file-id map)
        let open = self.open_index()?;
        let content_reader = &open.content;

        // Collect unique file paths from candidates and load their contents
        use std::collections::HashMap;
        let mut file_contents: HashMap<String, String> = HashMap::new();

        for candidate in &candidates {
            if file_contents.contains_key(&candidate.path) {
                continue;
            }

            // Find file_id for this path
            let file_id = match open.file_id_for(&candidate.path) {
                Some(id) => id,
                None => {
                    log::warn!("Could not find file_id for path: {}", candidate.path);
                    continue;
                }
            };

            // Load file content
            let content = match content_reader.get_file_content(file_id) {
                Ok(c) => c,
                Err(e) => {
                    log::warn!("Failed to read file {}: {}", candidate.path, e);
                    continue;
                }
            };

            file_contents.insert(candidate.path.clone(), content.to_string());
        }

        log::debug!(
            "Executing AST query on {} candidate files with language {:?}",
            file_contents.len(),
            lang
        );

        // Execute AST query using the ast_query module
        let results =
            crate::ast_query::execute_ast_query(candidates, ast_pattern, lang, &file_contents)?;

        log::info!(
            "AST query found {} matches for pattern '{}'",
            results.len(),
            ast_pattern
        );

        Ok(results)
    }

    /// Map keyword patterns to SymbolKind for auto-inference
    ///
    /// When users search for keywords like "class" or "function" with --symbols,
    /// automatically infer the kind filter to return only symbols of that type.
    ///
    /// This makes keyword queries more intuitive: searching for "class" returns
    /// only classes, not all symbols.
    fn keyword_to_kind(keyword: &str) -> Option<SymbolKind> {
        filter::keyword_to_kind(keyword)
    }

    /// Get all files matching the language filter (for keyword queries)
    ///
    /// This method bypasses trigram search and returns ALL files of the specified language.
    /// Used for keyword queries like "list all classes" where we need complete coverage,
    /// not just the first 100 candidates from a trigram search.
    ///
    /// Similar to `search_ast_all_files()` but works for symbol queries instead of AST queries.
    fn get_all_language_files(&self, filter: &QueryFilter) -> Result<Vec<SearchResult>> {
        // Language filter is optional - if not specified, scan all files
        // If specified, only scan files of that language

        // The shared content store
        let open = self.open_index()?;
        let content_reader = &open.content;

        // Build glob matchers if specified (for filtering)
        let include_matcher = result::build_glob_set(&filter.glob_patterns, "glob");

        let exclude_matcher = result::build_glob_set(&filter.exclude_patterns, "exclude");

        // Scan all files and filter by language + glob patterns
        let mut candidates: Vec<SearchResult> = Vec::new();

        for file_id in 0..content_reader.file_count() {
            let file_path = match content_reader.get_file_path(file_id as u32) {
                Some(p) => p,
                None => continue,
            };

            // Detect language from file extension
            let detected_lang = Language::from_path(file_path);

            // Filter by language (if specified)
            if let Some(lang) = filter.language
                && detected_lang != lang
            {
                continue;
            }

            let file_path_str = file_path.to_string_lossy().to_string();

            // Apply glob/exclude filters
            let included = include_matcher
                .as_ref()
                .is_none_or(|m| m.is_match(&file_path_str));
            let excluded = exclude_matcher
                .as_ref()
                .is_some_and(|m| m.is_match(&file_path_str));

            if !included || excluded {
                continue;
            }

            // Apply file path filter if specified
            if let Some(ref file_pattern) = filter.file_pattern
                && !file_path_str.contains(file_pattern)
            {
                continue;
            }

            // Create a dummy candidate for this file
            // Phase 2 (symbol enrichment) will parse it and extract actual symbols
            candidates.push(SearchResult {
                path: file_path_str,
                lang: detected_lang,
                span: Span {
                    start_line: 1,
                    end_line: 1,
                },
                symbol: None,
                kind: SymbolKind::Unknown("keyword_query".to_string()),
                preview: String::new(),
                dependencies: None,
            });
        }

        if let Some(lang) = filter.language {
            log::info!(
                "Keyword query will scan {} {:?} files for symbol extraction",
                candidates.len(),
                lang
            );
        } else {
            log::info!(
                "Keyword query will scan {} files (all languages) for symbol extraction",
                candidates.len()
            );
        }

        Ok(candidates)
    }

    /// Get candidate results using trigram-based full-text search
    ///
    /// The trigram intersection yields exact candidate `(file, line)` pairs; only
    /// those lines are verified, in path order, in parallel, and — in list mode —
    /// only until `budget` results exist (see [`verify_files_streaming`]).
    fn get_trigram_candidates(
        &self,
        pattern: &str,
        filter: &QueryFilter,
        budget: Option<usize>,
    ) -> Result<(Vec<SearchResult>, CandidateStats)> {
        let open = self.open_index()?;
        let trigram_index = &open.trigrams;

        // Patterns shorter than 3 chars have no trigrams, so the trigram index always
        // returns empty.  Fall back to a linear scan of the content store so that
        // --force (which bypasses the broad-query guard) still produces real results.
        if pattern.chars().count() < 3 {
            log::info!(
                "Pattern '{}' is shorter than 3 chars — trigram index cannot be used, \
                 falling back to linear scan",
                pattern
            );
            let results = self.linear_scan_candidates(pattern, filter, &open)?;
            let stats = CandidateStats {
                index_path: IndexPath::Scan,
                ..CandidateStats::default()
            };
            return Ok((results, stats));
        }

        // Search using trigrams
        let candidates_started = std::time::Instant::now();
        let candidates = trigram_index.search_candidates(pattern);
        let candidates_us = candidates_started.elapsed().as_micros() as u64;
        log::debug!(
            "Found {} candidate locations from trigram search in {} us",
            candidates.len(),
            candidates_us
        );

        // Group candidate lines by file. The intersection is sorted by
        // (file_id, line_no), so each file's lines arrive ascending.
        let mut files: Vec<(u32, LineSet)> = Vec::new();
        for loc in candidates {
            match files.last_mut() {
                Some((id, LineSet::Only(lines))) if *id == loc.file_id => lines.push(loc.line_no),
                _ => files.push((loc.file_id, LineSet::Only(vec![loc.line_no]))),
            }
        }
        log::debug!("Scanning {} files with trigram matches", files.len());

        // One matcher for the whole query (see `LineMatcher`).
        let matcher = LineMatcher::new(pattern, filter)?;
        let file_filter = FileFilter::from_filter(filter);

        let outcome = verify_files_streaming(
            &open,
            files,
            &matcher,
            &file_filter,
            filter.paths_only,
            budget,
        );

        let stats = CandidateStats {
            candidates_us,
            substring_only: matcher.is_word_boundary().then_some(outcome.substring_only),
            exhausted: outcome.exhausted,
            approx_total: outcome.estimated_total,
            index_path: IndexPath::Trigram,
            warnings: Vec::new(),
        };
        Ok((outcome.results, stats))
    }

    /// Linear scan fallback for patterns shorter than 3 characters.
    ///
    /// The trigram index requires 3-char n-grams; patterns like "fn" or "i" yield
    /// zero trigrams and therefore zero results.  This method scans every file in
    /// the content store directly using the same matching logic (word-boundary,
    /// contains, or regex) so short-pattern queries always return real results.
    fn linear_scan_candidates(
        &self,
        pattern: &str,
        filter: &QueryFilter,
        open: &OpenIndex,
    ) -> Result<Vec<SearchResult>> {
        use rayon::prelude::*;

        let content_reader = &open.content;
        let pattern_owned = pattern.to_string();
        let file_count = content_reader.file_count();
        let matcher = LineMatcher::new(pattern, filter)?;

        let results: Vec<SearchResult> = open.pool().install(|| {
            (0..file_count as u32)
                .collect::<Vec<_>>()
                .par_iter()
                .flat_map(|&file_id| {
                    let file_path = match content_reader.get_file_path(file_id) {
                        Some(p) => p.to_path_buf(),
                        None => return Vec::new(),
                    };
                    let content = match content_reader.get_file_content(file_id) {
                        Ok(c) => c,
                        Err(_) => return Vec::new(),
                    };

                    let file_path_str = file_path.to_string_lossy().to_string();
                    let lang = Language::from_path(&file_path);

                    let mut seen_lines = std::collections::HashSet::new();
                    let mut file_results = Vec::new();

                    for (line_idx, line) in content.lines().enumerate() {
                        let line_no = line_idx + 1;
                        if seen_lines.contains(&line_no) {
                            continue;
                        }

                        if !matcher.is_match(line) {
                            continue;
                        }

                        seen_lines.insert(line_no);
                        file_results.push(SearchResult {
                            path: file_path_str.clone(),
                            lang,
                            kind: SymbolKind::Unknown("text_match".to_string()),
                            symbol: None,
                            span: Span {
                                start_line: line_no,
                                end_line: line_no,
                            },
                            // Bounded, and WINDOWED on the match: on a minified bundle
                            // this line is the whole 1.45 MB file, and the first 512
                            // bytes of it would tell the caller nothing.
                            preview: crate::parsers::preview::line_preview(
                                line,
                                line.find(pattern_owned.as_str()).unwrap_or(0),
                            ),
                            dependencies: None,
                        });
                    }

                    file_results
                })
                .collect()
        });

        log::info!(
            "Linear scan (short pattern '{}') found {} results across {} files",
            pattern,
            results.len(),
            file_count
        );
        Ok(results)
    }

    /// Get candidate results using regex patterns with trigram optimization
    ///
    /// # Algorithm
    ///
    /// 1. Extract literal sequences (≥3 chars) from the regex pattern
    /// 2. For each literal, take the exact candidate `(file, line)` pairs from the
    ///    trigram index and UNION them
    /// 3. Verify the regex on those lines only, in path order, in parallel, with the
    ///    same early termination as literal search
    /// 4. With no usable literal (none ≥3 chars, or a case-insensitive flag), verify
    ///    every line of every file
    ///
    /// # Why candidate lines are sufficient
    ///
    /// Matching is per line, and every literal the extractor emits must appear
    /// verbatim in any match (the extractor drops the atom before a `?`, `*` or
    /// `{0,n}`). A matching line therefore contains at least one emitted literal
    /// in full, and the union over literals of their exact candidate lines is a
    /// superset of the matching lines. This is the same assumption the literal
    /// path makes; a capped posting list loses a file in both paths alike.
    fn get_regex_candidates(
        &self,
        pattern: &str,
        filter: &QueryFilter,
        timeout: Option<&std::time::Duration>,
        start_time: &std::time::Instant,
        budget: Option<usize>,
    ) -> Result<(Vec<SearchResult>, CandidateStats)> {
        // Step 1: Compile the regex (the filter carries the `kind` label)
        let matcher = LineMatcher::new(pattern, filter)
            .with_context(|| format!("Invalid regex pattern: {}", pattern))?;

        // Check timeout before expensive operations
        if let Some(timeout_duration) = timeout
            && start_time.elapsed() > *timeout_duration
        {
            anyhow::bail!(
                "Query timeout exceeded ({} seconds) during regex compilation",
                timeout_duration.as_secs()
            );
        }

        // Step 2: Extract literals from the regex. A literal under `(?i)` is looked
        // up under every ASCII case variant; a non-ASCII one cannot be folded.
        use crate::regex_trigrams::extract_literals;
        let literals = extract_literals(pattern);
        let unfoldable = literals
            .iter()
            .any(|l| l.case_insensitive && !l.text.is_ascii());

        let open = self.open_index()?;
        let candidates_started = std::time::Instant::now();

        let mut warnings = Vec::new();
        let index_path;
        let files: Vec<(u32, LineSet)> = if literals.is_empty() || unfoldable {
            // No usable literal - fall back to a full scan of every line.
            index_path = IndexPath::Scan;
            // A literal shorter than 3 chars scans silently on the case-sensitive
            // path too (`get_trigram_candidates`); `-i fn` must not warn either.
            let short_literal = filter
                .rewritten_from
                .as_deref()
                .is_some_and(|p| p.chars().count() < 3);
            if !short_literal {
                let text = if unfoldable {
                    format!(
                        "Regex pattern '{}' has a non-ASCII case-insensitive literal, which the trigram index cannot fold; falling back to full content scan. This may be slow on large codebases.",
                        pattern
                    )
                } else {
                    format!(
                        "Regex pattern '{}' has no literals (≥3 chars), falling back to full content scan. This may be slow on large codebases. Consider using patterns with literal text.",
                        pattern
                    )
                };
                if !filter.suppress_output {
                    output::warn(&text);
                }
                warnings.push(text);
            }
            (0..open.content.file_count() as u32)
                .map(|id| (id, LineSet::All))
                .collect()
        } else {
            index_path = IndexPath::Trigram;
            log::debug!(
                "Using {} literals to narrow regex search candidates",
                literals.len()
            );

            // Union of each literal's exact candidate lines (alternation-safe).
            // Each source is sorted by (file, line) and key-unique, so a single
            // source needs no sort; several are merged with one sort + dedup. A
            // per-file BTreeMap here cost ~8 ms per 100k locations.
            let mut locations: Vec<crate::trigram::FileLocation> = Vec::new();
            let mut sources = 0usize;
            // Under Unicode case folding `k` and `s` also match the Kelvin sign
            // and the long s, which live on lines the ASCII fold cannot reach.
            let mut need_exotic = false;
            for literal in &literals {
                let found = if literal.case_insensitive {
                    if literal
                        .text
                        .bytes()
                        .any(|b| matches!(b.to_ascii_lowercase(), b'k' | b's'))
                    {
                        need_exotic = true;
                    }
                    open.trigrams
                        .search_candidates_fold(literal.text.as_bytes())
                } else {
                    open.trigrams.search_candidates(&literal.text)
                };
                log::debug!(
                    "Literal '{}' (ci={}) found on {} candidate lines",
                    literal.text,
                    literal.case_insensitive,
                    found.len()
                );
                locations.extend(found);
                sources += 1;
            }
            if need_exotic {
                locations.extend(open.trigrams.exotic_fold_lines());
                sources += 1;
            }
            if sources > 1 {
                locations.sort_unstable();
                locations.dedup();
            }
            // Group by file; lines arrive ascending within each file.
            let mut files: Vec<(u32, LineSet)> = Vec::new();
            for loc in locations {
                match files.last_mut() {
                    Some((id, LineSet::Only(lines))) if *id == loc.file_id => {
                        lines.push(loc.line_no)
                    }
                    _ => files.push((loc.file_id, LineSet::Only(vec![loc.line_no]))),
                }
            }
            files
        };
        let candidates_us = candidates_started.elapsed().as_micros() as u64;
        log::debug!(
            "Regex candidates: {} files in {} us",
            files.len(),
            candidates_us
        );

        let file_filter = FileFilter::from_filter(filter);
        let outcome = verify_files_streaming(
            &open,
            files,
            &matcher,
            &file_filter,
            filter.paths_only,
            budget,
        );

        log::info!(
            "Regex search found {} matches for pattern '{}'",
            outcome.results.len(),
            pattern
        );
        let stats = CandidateStats {
            candidates_us,
            substring_only: None,
            exhausted: outcome.exhausted,
            approx_total: outcome.estimated_total,
            index_path,
            warnings,
        };
        Ok((outcome.results, stats))
    }

    /// Get index status for programmatic use (doesn't print warnings)
    ///
    /// Returns (status, can_trust_results, warning) tuple for JSON output.
    /// This is optimized for AI agents to detect staleness and auto-reindex.
    pub fn get_index_status(&self) -> Result<(IndexStatus, bool, Option<IndexWarning>)> {
        self.index_status_for(None)
    }

    /// Index status read from the filesystem, never from the freshness memo.
    ///
    /// For `check_index_status`: an agent asking whether the index is current is
    /// exactly the caller that must not be told what was true a second ago.
    pub fn fresh_index_status(&self) -> Result<(IndexStatus, bool, Option<IndexWarning>)> {
        status_cache::invalidate(&self.cache.workspace_root());
        self.index_status_for(None)
    }

    /// Index status, with `scope` naming the files a caller's answer came from.
    ///
    /// `scope` does NOT soften `can_trust_results`. A stale index always yields
    /// `can_trust_results: false`, because the changes that matter most are the ones
    /// scoping cannot see: a newly created file produces no results to intersect
    /// with, and an empty result set has no scope at all — which is exactly the case
    /// where an agent concludes "no callers" and acts on it.
    ///
    /// Scope is used only to describe the impact, via
    /// [`IndexWarning::reason`], so a caller can tell "your results may be missing a
    /// file you just edited" from "a file in these results has changed".
    pub fn index_status_for(
        &self,
        scope: Option<&[String]>,
    ) -> Result<(IndexStatus, bool, Option<IndexWarning>)> {
        // Everything that costs a subprocess or a database open is computed once per
        // TTL in `status_cache`; only the scope-dependent wording is built here.
        let snapshot = status_cache::snapshot(&self.cache)?;
        let (details, changes) = match snapshot.as_ref() {
            status_cache::Snapshot::Decided(verdict) => return Ok(verdict.clone()),
            status_cache::Snapshot::Worktree { details, changes } => (details.clone(), changes),
        };

        if changes.is_empty() {
            return Ok((IndexStatus::Fresh, true, None));
        }

        // Whether a changed file is among the ones this answer came from. Used only
        // to sharpen the message — never to upgrade trust.
        let hits_results = match scope {
            None | Some([]) => false,
            Some(paths) => changes.any(|changed| {
                paths
                    .iter()
                    .any(|p| p == changed || p.ends_with(changed) || changed.ends_with(p.as_str()))
            }),
        };

        let reason = {
            let mut parts = Vec::new();
            if changes.modified_count > 0 {
                parts.push(format!("{} modified", changes.modified_count));
            }
            if changes.added_count > 0 {
                parts.push(format!("{} added", changes.added_count));
            }
            if changes.deleted_count > 0 {
                parts.push(format!("{} deleted", changes.deleted_count));
            }
            let impact = if hits_results {
                " — including a file these results came from"
            } else if changes.deleted_count > 0 {
                " — deleted files still produce hits at their old lines"
            } else {
                " — these results may not reflect them"
            };
            format!(
                "Working tree has uncommitted changes since indexing ({}){}",
                parts.join(", "),
                impact
            )
        };

        let some_if_any = |v: Vec<String>| if v.is_empty() { None } else { Some(v) };
        let warning = IndexWarning {
            reason,
            action_required: "index_project".to_string(),
            files_modified: some_if_any(changes.modified.clone()),
            files_added: some_if_any(changes.added.clone()),
            files_deleted: some_if_any(changes.deleted.clone()),
            changed_count: Some(
                changes.modified_count + changes.added_count + changes.deleted_count,
            ),
            truncated: changes.truncated,
            details: Some(details),
        };

        // Stale means untrusted, with no exception. A search served from an index
        // that does not know about the caller's own edits cannot promise completeness,
        // and a silently-confident wrong answer is the failure this release fixes.
        Ok((IndexStatus::Stale, false, Some(warning)))
    }

    /// Check index freshness and show non-blocking warnings
    ///
    /// This performs lightweight checks to warn users if their index might be stale:
    /// 1. Branch mismatch: indexed different branch
    /// 2. Commit changed: HEAD moved since indexing
    /// 3. File changes: quick mtime check on sample of files (if available)
    fn check_index_freshness(&self, filter: &QueryFilter) -> Result<()> {
        let root = self.cache.workspace_root();

        // Check git state if in a git repo
        if crate::git::is_git_repo(&root) {
            if !crate::git::is_git_available() {
                static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
                if !filter.suppress_output {
                    WARNED.get_or_init(|| {
                        output::warn("⚠️  git binary not found in PATH; index freshness checks disabled for this session.");
                    });
                }
                return Ok(());
            }
            if let Ok(current_branch) = crate::git::get_current_branch(&root) {
                // Check if we're on a different branch than what was indexed
                if !self.cache.branch_exists(&current_branch).unwrap_or(false) {
                    if !filter.suppress_output {
                        output::warn(&format!(
                            "⚠️  WARNING: Index not found for branch '{}'. Run 'rfx index' to index this branch.",
                            current_branch
                        ));
                    }
                    return Ok(());
                }

                // Branch exists - check if commit changed
                if let (Ok(current_commit), Ok(branch_info)) = (
                    crate::git::get_current_commit(&root),
                    self.cache.get_branch_info(&current_branch),
                ) {
                    if branch_info.commit_sha != current_commit {
                        if !filter.suppress_output {
                            output::warn(&format!(
                                "⚠️  WARNING: Index may be stale (commit changed: {} → {}). Consider running 'rfx index'.",
                                &branch_info.commit_sha[..7],
                                &current_commit[..7]
                            ));
                        }
                        return Ok(());
                    }

                    // If commits match, do a quick file freshness check
                    // Sample up to 10 files to check for modifications (cheap mtime check)
                    if let Ok(branch_files) = self.cache.get_branch_files(&current_branch) {
                        let mut checked = 0;
                        let mut changed = 0;
                        const SAMPLE_SIZE: usize = 10;

                        for (path, _indexed_hash) in branch_files.iter().take(SAMPLE_SIZE) {
                            checked += 1;
                            let file_path = std::path::Path::new(path);

                            // Check if file exists and has been modified (mtime/size heuristic)
                            if let Ok(metadata) = std::fs::metadata(file_path)
                                && let Ok(modified) = metadata.modified()
                            {
                                let indexed_time = branch_info.last_indexed;
                                let file_time = modified
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs()
                                    as i64;

                                // If file modified after indexing, it might be stale
                                if file_time > indexed_time {
                                    // File modified after indexing - likely stale
                                    // Note: We skip hash verification for performance (mtime check is sufficient)
                                    // This may cause false positives if files were touched without changes,
                                    // but the warning is non-blocking and vastly better than slow queries
                                    changed += 1;
                                }
                            }
                        }

                        if changed > 0 && !filter.suppress_output {
                            output::warn(&format!(
                                "⚠️  WARNING: {} of {} sampled files changed since indexing. Consider running 'rfx index'.",
                                changed, checked
                            ));
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Generate AI instruction based on query results
///
/// Provides context-aware guidance to AI agents on how to handle search results.
/// Uses priority-based logic to determine the most relevant instruction.
#[allow(clippy::too_many_arguments)]
pub fn generate_ai_instruction(
    result_count: usize,
    total_count: usize,
    has_more: bool,
    symbols_mode: bool,
    paths_only: bool,
    use_ast: bool,
    use_regex: bool,
    language_filter: bool,
    glob_filter: bool,
    exact_mode: bool,
) -> Option<String> {
    // Priority 1: No results
    if result_count == 0 {
        return Some(
            "No results found. Consider these alternatives: 1) Check pattern spelling, 2) Remove --kind or --lang filters to broaden search, 3) Try partial match or related term, 4) Use search_regex tool for pattern matching with special characters or complex patterns."
            .to_string()
        );
    }

    // Priority 2: Query too broad (500+ results)
    if total_count >= 500 {
        return Some(format!(
            "Query too broad: {} results found. STOP. Do not list results. Refine search automatically by adding filters: kind parameter (Function/Struct/Class), lang parameter (rust/python/etc), or glob parameter (['src/**/*.rs']). Call search_code again with appropriate filters.",
            total_count
        ));
    }

    // Priority 3: Paginated results
    //
    // REF-191: for autonomous find-all tasks this instruction must NOT tell the
    // agent to stop and ask the user — there is no user in an agent loop, and a
    // partial answer to "find every occurrence" is wrong. Instruct decisive
    // continuation: fetch the remaining page(s) via offset, or probe the total
    // cheaply with mode="count" first.
    if has_more {
        return Some(format!(
            "Showing {} of {} results — {} more available. This is a partial answer. To finish a find-all task, call again with offset={} (raise limit up to 500 to get the rest in one call), or use mode=\"count\" first if you only need the total.",
            result_count,
            total_count,
            total_count.saturating_sub(result_count),
            result_count
        ));
    }

    // Priority 4: Single precise result (symbols mode)
    if result_count == 1 && symbols_mode {
        return Some(
            "Found 1 precise result. Respond concisely: '[symbol] at [path]:[line]'.".to_string(),
        );
    }

    // Priority 5: Few precise results (symbols mode)
    if (2..=10).contains(&result_count) && symbols_mode {
        return Some(format!(
            "Found {} precise results (definitions only, not usages). List locations concisely: '[symbol] at [path]:[line]' for each result.",
            result_count
        ));
    }

    // Priority 6: Many results (101-500)
    if (101..500).contains(&total_count) {
        return Some(format!(
            "Found {} results - this is broad. Suggest refining search with: kind parameter (Function/Struct/Class/etc), lang parameter (rust/python/etc), or glob parameter to narrow file scope.",
            total_count
        ));
    }

    // Priority 7: Full-text mode with many results (suggest symbols mode)
    if result_count >= 100 && !symbols_mode {
        return Some(format!(
            "Found {} results in full-text search mode (includes definitions AND all usages). Consider using symbols=true parameter to filter to definitions only. This typically reduces results by 80-90%.",
            result_count
        ));
    }

    // Priority 8: Paths-only mode
    if paths_only {
        return Some(format!(
            "Found {} unique files (paths-only mode - no code content included). Next step: Use Read tool on specific files that look relevant based on their paths.",
            result_count
        ));
    }

    // Priority 9: AST query results
    if use_ast {
        return Some(format!(
            "Found {} results using AST pattern matching. These are structure-based matches using Tree-sitter patterns, not text search.",
            result_count
        ));
    }

    // Priority 10: Regex with many results
    if use_regex && result_count >= 100 {
        return Some(format!(
            "Found {} results using regex pattern matching. Regex matches are expansive. Consider using exact text search or symbols mode for more precise results.",
            result_count
        ));
    }

    // Priority 11: Language filter with few results
    if language_filter && result_count <= 5 {
        return Some(format!(
            "Found {} results with language filter active. Results are limited to this language only. Remove lang parameter if you want to search all languages.",
            result_count
        ));
    }

    // Priority 12: Glob filter with few results
    if glob_filter && result_count <= 10 {
        return Some(format!(
            "Found {} results with glob filter active. Results are limited to matching paths. Remove glob parameter to search entire codebase.",
            result_count
        ));
    }

    // Priority 13: Exact mode with few results
    if exact_mode && result_count <= 5 {
        return Some(format!(
            "Found {} results in exact match mode. Only exact symbol name matches are included. Remove exact parameter to allow substring matching.",
            result_count
        ));
    }

    // Normal case (11-100 results, no special conditions) - no instruction
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::Indexer;
    use crate::models::IndexConfig;
    use std::fs;
    use tempfile::TempDir;

    // ==================== Basic Tests ====================

    #[test]
    fn test_query_engine_creation() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let engine = QueryEngine::new(cache);

        assert!(engine.cache.path().ends_with(".reflex"));
    }

    #[test]
    fn test_filter_modes() {
        // Test that symbols_mode works as expected
        let filter_fulltext = QueryFilter::default();
        assert!(!filter_fulltext.symbols_mode);

        let filter_symbols = QueryFilter {
            symbols_mode: true,
            ..Default::default()
        };
        assert!(filter_symbols.symbols_mode);

        // Test that kind implies symbols_mode (handled in CLI layer)
        let filter_with_kind = QueryFilter {
            kind: Some(SymbolKind::Function),
            symbols_mode: true,
            ..Default::default()
        };
        assert!(filter_with_kind.symbols_mode);
    }

    // ==================== Search Mode Tests ====================

    #[test]
    fn test_fulltext_search() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();

        // Create test files
        fs::write(
            project.join("main.rs"),
            "fn main() {\n    println!(\"hello\");\n}",
        )
        .unwrap();
        fs::write(project.join("lib.rs"), "pub fn hello() {}").unwrap();

        // Index the project
        let cache = CacheManager::new(&project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(&project, false).unwrap();

        // Search for "hello"
        let cache = CacheManager::new(&project);
        let engine = QueryEngine::new(cache);
        let filter = QueryFilter::default(); // full-text mode
        let results = engine.search("hello", filter).unwrap();

        // Should find both occurrences (println and function name)
        assert!(results.len() >= 2);
        assert!(results.iter().any(|r| r.path.contains("main.rs")));
        assert!(results.iter().any(|r| r.path.contains("lib.rs")));
    }

    #[test]
    fn test_symbol_search() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();

        // Create test file with function definition and call
        fs::write(
            project.join("main.rs"),
            "fn greet() {}\nfn main() {\n    greet();\n}",
        )
        .unwrap();

        // Index
        let cache = CacheManager::new(&project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(&project, false).unwrap();

        let cache = CacheManager::new(&project);

        // Symbol search (definitions only)
        let engine = QueryEngine::new(cache);
        let filter = QueryFilter {
            symbols_mode: true,
            ..Default::default()
        };
        let results = engine.search("greet", filter).unwrap();

        // Should find only the definition, not the call
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.kind == SymbolKind::Function));
    }

    #[test]
    fn test_regex_search() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();

        fs::write(
            project.join("main.rs"),
            "fn test1() {}\nfn test2() {}\nfn other() {}",
        )
        .unwrap();

        let cache = CacheManager::new(&project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(&project, false).unwrap();

        let cache = CacheManager::new(&project);

        let engine = QueryEngine::new(cache);
        let filter = QueryFilter {
            use_regex: true,
            ..Default::default()
        };
        let results = engine.search(r"fn test\d", filter).unwrap();

        // Should match test1 and test2 but not other
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.preview.contains("test")));
    }

    // ==================== Filter Tests ====================

    #[test]
    fn test_language_filter() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();

        fs::write(project.join("main.rs"), "fn main() {}").unwrap();
        fs::write(project.join("main.js"), "function main() {}").unwrap();

        let cache = CacheManager::new(&project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(&project, false).unwrap();

        let cache = CacheManager::new(&project);

        let engine = QueryEngine::new(cache);

        // Filter to Rust only
        let filter = QueryFilter {
            language: Some(Language::Rust),
            ..Default::default()
        };
        let results = engine.search("main", filter).unwrap();

        assert!(results.iter().all(|r| r.lang == Language::Rust));
        assert!(results.iter().all(|r| r.path.ends_with(".rs")));
    }

    #[test]
    fn test_kind_filter() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();

        fs::write(
            project.join("main.rs"),
            "struct Point {}\nfn main() {}\nimpl Point { fn new() {} }",
        )
        .unwrap();

        let cache = CacheManager::new(&project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(&project, false).unwrap();

        let cache = CacheManager::new(&project);

        let engine = QueryEngine::new(cache);

        // Filter to functions only (includes methods)
        let filter = QueryFilter {
            symbols_mode: true,
            kind: Some(SymbolKind::Function),
            use_contains: true, // "mai" is substring of "main"
            ..Default::default()
        };
        // Search for "mai" which should match "main" (tri gram pattern will def be in index)
        let results = engine.search("mai", filter).unwrap();

        // Should find main function
        assert!(!results.is_empty(), "Should find at least one result");
        assert!(
            results.iter().any(|r| r.symbol.as_deref() == Some("main")),
            "Should find 'main' function"
        );
    }

    #[test]
    fn test_file_pattern_filter() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir_all(project.join("src")).unwrap();
        fs::create_dir_all(project.join("tests")).unwrap();

        fs::write(project.join("src/lib.rs"), "fn foo() {}").unwrap();
        fs::write(project.join("tests/test.rs"), "fn foo() {}").unwrap();

        let cache = CacheManager::new(&project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(&project, false).unwrap();

        let cache = CacheManager::new(&project);

        let engine = QueryEngine::new(cache);

        // Filter to src/ only
        let filter = QueryFilter {
            file_pattern: Some("src/".to_string()),
            ..Default::default()
        };
        let results = engine.search("foo", filter).unwrap();

        assert!(results.iter().all(|r| r.path.contains("src/")));
        assert!(!results.iter().any(|r| r.path.contains("tests/")));
    }

    #[test]
    fn test_limit_filter() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();

        // Create file with many matches
        let content = (0..20)
            .map(|i| format!("fn test{}() {{}}", i))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(project.join("main.rs"), content).unwrap();

        let cache = CacheManager::new(&project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(&project, false).unwrap();

        let cache = CacheManager::new(&project);

        let engine = QueryEngine::new(cache);

        // Limit to 5 results
        let filter = QueryFilter {
            limit: Some(5),
            use_contains: true, // "test" is substring of "test0", "test1", etc.
            ..Default::default()
        };
        let results = engine.search("test", filter).unwrap();

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_exact_match_filter() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();

        fs::write(
            project.join("main.rs"),
            "fn test() {}\nfn test_helper() {}\nfn other_test() {}",
        )
        .unwrap();

        let cache = CacheManager::new(&project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(&project, false).unwrap();

        let cache = CacheManager::new(&project);

        let engine = QueryEngine::new(cache);

        // Exact match for "test"
        let filter = QueryFilter {
            symbols_mode: true,
            exact: true,
            ..Default::default()
        };
        let results = engine.search("test", filter).unwrap();

        // Should only match exactly "test", not "test_helper" or "other_test"
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].symbol.as_deref(), Some("test"));
    }

    // ==================== Expand Mode Tests ====================

    #[test]
    fn test_expand_mode() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();

        fs::write(
            project.join("main.rs"),
            "fn greet() {\n    println!(\"Hello\");\n    println!(\"World\");\n}",
        )
        .unwrap();

        let cache = CacheManager::new(&project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(&project, false).unwrap();

        let cache = CacheManager::new(&project);

        let engine = QueryEngine::new(cache);

        // Search with expand mode
        let filter = QueryFilter {
            symbols_mode: true,
            expand: true,
            ..Default::default()
        };
        let results = engine.search("greet", filter).unwrap();

        // Should have full function body in preview
        assert!(!results.is_empty());
        let result = &results[0];
        assert!(result.preview.contains("println"));
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_search_empty_index() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();

        let cache = CacheManager::new(&project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(&project, false).unwrap();

        let cache = CacheManager::new(&project);

        let engine = QueryEngine::new(cache);
        let filter = QueryFilter::default();
        let results = engine.search("nonexistent", filter).unwrap();

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_search_no_index() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();

        let cache = CacheManager::new(&project);
        let engine = QueryEngine::new(cache);
        let filter = QueryFilter::default();

        // Should fail when index doesn't exist
        assert!(engine.search("test", filter).is_err());
    }

    #[test]
    fn test_search_special_characters() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();

        fs::write(project.join("main.rs"), "let x = 42;\nlet y = x + 1;").unwrap();

        let cache = CacheManager::new(&project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(&project, false).unwrap();

        let cache = CacheManager::new(&project);

        let engine = QueryEngine::new(cache);
        let filter = QueryFilter::default();

        // Search for special characters
        let results = engine.search("x + ", filter).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_search_unicode() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();

        fs::write(project.join("main.rs"), "// 你好世界\nfn main() {}").unwrap();

        let cache = CacheManager::new(&project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(&project, false).unwrap();

        let cache = CacheManager::new(&project);

        let engine = QueryEngine::new(cache);
        let filter = QueryFilter {
            use_contains: true, // Unicode word boundaries may not work as expected
            force: true,        // Bypass broad query detection for 2-char Unicode pattern
            ..Default::default()
        };

        // Search for unicode characters
        let results = engine.search("你好", filter).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_case_sensitive_search() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();

        fs::write(project.join("main.rs"), "fn Test() {}\nfn test() {}").unwrap();

        let cache = CacheManager::new(&project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(&project, false).unwrap();

        let cache = CacheManager::new(&project);

        let engine = QueryEngine::new(cache);
        let filter = QueryFilter::default();

        // Search is case-sensitive
        let results = engine.search("Test", filter).unwrap();
        assert!(results.iter().any(|r| r.preview.contains("Test()")));
    }

    // ==================== Determinism Tests ====================

    #[test]
    fn test_results_sorted_deterministically() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();

        fs::write(project.join("a.rs"), "fn test() {}").unwrap();
        fs::write(project.join("z.rs"), "fn test() {}").unwrap();
        fs::write(project.join("m.rs"), "fn test() {}\nfn test2() {}").unwrap();

        let cache = CacheManager::new(&project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(&project, false).unwrap();

        let cache = CacheManager::new(&project);

        let engine = QueryEngine::new(cache);
        let filter = QueryFilter::default();

        // Run search multiple times
        let results1 = engine.search("test", filter.clone()).unwrap();
        let results2 = engine.search("test", filter.clone()).unwrap();
        let results3 = engine.search("test", filter).unwrap();

        // Results should be identical and sorted by path then line
        assert_eq!(results1.len(), results2.len());
        assert_eq!(results1.len(), results3.len());

        for i in 0..results1.len() {
            assert_eq!(results1[i].path, results2[i].path);
            assert_eq!(results1[i].path, results3[i].path);
            assert_eq!(results1[i].span.start_line, results2[i].span.start_line);
            assert_eq!(results1[i].span.start_line, results3[i].span.start_line);
        }

        // Verify sorting (path ascending, then line ascending)
        for i in 0..results1.len().saturating_sub(1) {
            let curr = &results1[i];
            let next = &results1[i + 1];
            assert!(
                curr.path < next.path
                    || (curr.path == next.path && curr.span.start_line <= next.span.start_line)
            );
        }
    }

    // ==================== Combined Filter Tests ====================

    #[test]
    fn test_multiple_filters_combined() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir_all(project.join("src")).unwrap();

        fs::write(project.join("src/main.rs"), "fn test() {}\nstruct Test {}").unwrap();
        fs::write(project.join("src/lib.rs"), "fn test() {}").unwrap();
        fs::write(project.join("test.js"), "function test() {}").unwrap();

        let cache = CacheManager::new(&project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(&project, false).unwrap();

        let cache = CacheManager::new(&project);

        let engine = QueryEngine::new(cache);

        // Combine language, kind, and file pattern filters
        let filter = QueryFilter {
            language: Some(Language::Rust),
            kind: Some(SymbolKind::Function),
            file_pattern: Some("src/main".to_string()),
            symbols_mode: true,
            ..Default::default()
        };
        let results = engine.search("test", filter).unwrap();

        // Should only find the function in src/main.rs
        assert_eq!(results.len(), 1);
        assert!(results[0].path.contains("src/main.rs"));
        assert_eq!(results[0].kind, SymbolKind::Function);
    }

    // ==================== Helper Method Tests ====================

    #[test]
    fn test_find_symbol_helper() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();

        fs::write(project.join("main.rs"), "fn greet() {}").unwrap();

        let cache = CacheManager::new(&project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(&project, false).unwrap();

        let cache = CacheManager::new(&project);

        let engine = QueryEngine::new(cache);
        let results = engine.find_symbol("greet").unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].kind, SymbolKind::Function);
    }

    #[test]
    fn test_list_by_kind_helper() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();

        fs::write(
            project.join("main.rs"),
            "struct Point {}\nfn test() {}\nstruct Line {}",
        )
        .unwrap();

        let cache = CacheManager::new(&project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(&project, false).unwrap();

        let cache = CacheManager::new(&project);

        let engine = QueryEngine::new(cache);

        // Search for structs that contain "oin" (Point contains it, Line doesn't)
        let filter = QueryFilter {
            kind: Some(SymbolKind::Struct),
            symbols_mode: true,
            use_contains: true, // "oin" is substring of "Point"
            ..Default::default()
        };
        let results = engine.search("oin", filter).unwrap();

        // Should find Point struct
        assert!(!results.is_empty(), "Should find at least Point struct");
        assert!(results.iter().all(|r| r.kind == SymbolKind::Struct));
        assert!(results.iter().any(|r| r.symbol.as_deref() == Some("Point")));
    }

    // ==================== Metadata Tests ====================

    #[test]
    fn test_search_with_metadata() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();

        fs::write(project.join("main.rs"), "fn test() {}").unwrap();

        let cache = CacheManager::new(&project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(&project, false).unwrap();

        let cache = CacheManager::new(&project);

        let engine = QueryEngine::new(cache);
        let filter = QueryFilter::default();
        let response = engine.search_with_metadata("test", filter).unwrap();

        // Check metadata is present (status might be stale if run inside git repo)
        assert!(!response.results.is_empty());
        // Note: can_trust_results may be false if running in a git repo without branch index
    }

    // ==================== Multi-language Tests ====================

    #[test]
    fn test_search_across_languages() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();

        fs::write(project.join("main.rs"), "fn greet() {}").unwrap();
        fs::write(project.join("main.ts"), "function greet() {}").unwrap();
        fs::write(project.join("main.py"), "def greet(): pass").unwrap();

        let cache = CacheManager::new(&project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(&project, false).unwrap();

        let cache = CacheManager::new(&project);

        let engine = QueryEngine::new(cache);
        let filter = QueryFilter::default();
        let results = engine.search("greet", filter).unwrap();

        // Should find greet in all three languages
        assert!(results.len() >= 3);
        assert!(results.iter().any(|r| r.lang == Language::Rust));
        assert!(results.iter().any(|r| r.lang == Language::TypeScript));
        assert!(results.iter().any(|r| r.lang == Language::Python));
    }
}

/// Process-local memo for the freshness verdict's expensive inputs.
///
/// `search_with_metadata` runs the freshness check on EVERY query, and
/// `find_references` calls it two or three times per MCP request. One snapshot
/// costs one `meta.db` connection, two `git` spawns and one `git status
/// --porcelain`; before this module the same query ran three connections and
/// three spawns, and the memo covered only the last of them. Results are memoised
/// briefly, keyed by workspace root.
///
/// What this trades away: an edit landing less than the TTL before a search can be
/// reported fresh. Agent tool round-trips are seconds apart, and `check_index_status`
/// — the explicit probe an agent uses when it cares — always bypasses the cache.
/// Every index write in this process invalidates the memo.
///
/// `REFLEX_FRESHNESS_TTL_MS` overrides the window; `0` disables caching entirely.
mod status_cache {
    use crate::cache::CacheManager;
    use crate::git::WorktreeChanges;
    use crate::models::{IndexStatus, IndexWarning, IndexWarningDetails};
    use anyhow::Result;
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::{Duration, Instant};

    const DEFAULT_TTL_MS: u64 = 1_000;

    pub type Verdict = (IndexStatus, bool, Option<IndexWarning>);

    pub enum Snapshot {
        /// The verdict does not depend on which files an answer came from.
        Decided(Verdict),
        /// The working tree has changed; the wording depends on the caller's scope.
        Worktree {
            details: IndexWarningDetails,
            changes: WorktreeChanges,
        },
    }

    struct Entry {
        computed_at: Instant,
        snapshot: Arc<Snapshot>,
    }

    fn ttl() -> Duration {
        static TTL: OnceLock<Duration> = OnceLock::new();
        *TTL.get_or_init(|| {
            let ms = std::env::var("REFLEX_FRESHNESS_TTL_MS")
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or(DEFAULT_TTL_MS);
            Duration::from_millis(ms)
        })
    }

    fn store() -> &'static Mutex<HashMap<PathBuf, Entry>> {
        static STORE: OnceLock<Mutex<HashMap<PathBuf, Entry>>> = OnceLock::new();
        STORE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn key(root: &Path) -> PathBuf {
        root.canonicalize().unwrap_or_else(|_| root.to_path_buf())
    }

    /// Only paths Reflex would index can make the index stale. Editing a README or
    /// anything under `target/` must not mark it permanently behind.
    fn indexable_with(path: &str, policy: &crate::indexer::PathPolicy) -> bool {
        crate::indexer::Indexer::is_indexable_path_with(Path::new(path), Some(policy))
    }

    fn fresh() -> Verdict {
        (IndexStatus::Fresh, true, None)
    }

    /// The memoised snapshot for the workspace `cache` belongs to.
    pub fn snapshot(cache: &CacheManager) -> Result<Arc<Snapshot>> {
        let root = cache.workspace_root();
        let key = key(&root);
        let ttl = ttl();

        if !ttl.is_zero()
            && let Ok(map) = store().lock()
            && let Some(entry) = map.get(&key)
            && entry.computed_at.elapsed() < ttl
        {
            return Ok(Arc::clone(&entry.snapshot));
        }

        let snapshot = Arc::new(compute(cache, &root)?);

        if !ttl.is_zero()
            && let Ok(mut map) = store().lock()
        {
            map.insert(
                key,
                Entry {
                    computed_at: Instant::now(),
                    snapshot: Arc::clone(&snapshot),
                },
            );
        }

        Ok(snapshot)
    }

    fn compute(cache: &CacheManager, root: &Path) -> Result<Snapshot> {
        let is_git = crate::git::is_git_repo(root) && crate::git::is_git_available();
        let current_branch = if is_git {
            crate::git::get_current_branch(root).ok()
        } else {
            None
        };

        // One connection for the schema hash, the branch row and its metadata.
        let reads = match cache.status_reads(current_branch.as_deref()) {
            Ok(r) => r,
            Err(e) => {
                log::debug!("Could not read index status from meta.db: {}", e);
                return Ok(Snapshot::Decided(fresh()));
            }
        };

        // A cache written by a different Reflex build. `validate()` no longer bails on
        // this (that turned a version skew into a rebuild stampede — see cache.rs), so
        // reads still work; they are simply not to be trusted, and the message names
        // who owns the cache so the user can pick a side.
        if !reads.schema_ok {
            let reason = match reads.owner {
                Some((v, sha)) if v != env!("CARGO_PKG_VERSION") => {
                    let sha = sha
                        .map(|s| format!(" (sha {})", &s[..s.len().min(7)]))
                        .unwrap_or_default();
                    format!(
                        "This .reflex/ was written by reflex {}{}; this binary is reflex {}. \
                         Results come from a cache format this version does not fully \
                         understand.",
                        v,
                        sha,
                        env!("CARGO_PKG_VERSION")
                    )
                }
                // Same version, or unstamped: the cache format changed under it. A
                // reindex brings it up to date; results until then may be partial.
                _ => "The index was built with a different cache format and needs \
                      rebuilding. Results may be incomplete until then."
                    .to_string(),
            };

            let warning = IndexWarning::new(reason, "index_project");
            return Ok(Snapshot::Decided((
                IndexStatus::Stale,
                false,
                Some(warning),
            )));
        }

        // Outside git there is no cheap way to find changes, and walking the tree
        // on every query costs more than the staleness it would detect. Documented
        // as a known limitation in the tool descriptions.
        let Some(current_branch) = current_branch else {
            return Ok(Snapshot::Decided(fresh()));
        };

        // 1. A branch we have never indexed: nothing here is trustworthy.
        if !reads.branch_indexed {
            let warning = IndexWarning::new(
                format!("Branch '{}' has not been indexed", current_branch),
                "index_project",
            )
            .with_details(IndexWarningDetails {
                current_branch: Some(current_branch),
                indexed_branch: None,
                current_commit: None,
                indexed_commit: None,
            });
            return Ok(Snapshot::Decided((
                IndexStatus::Stale,
                false,
                Some(warning),
            )));
        }

        let (Ok(current_commit), Some(branch_info)) =
            (crate::git::get_current_commit(root), reads.branch_info)
        else {
            return Ok(Snapshot::Decided(fresh()));
        };

        let details = IndexWarningDetails {
            current_branch: Some(current_branch.clone()),
            indexed_branch: Some(branch_info.branch.clone()),
            current_commit: Some(current_commit.clone()),
            indexed_commit: Some(branch_info.commit_sha.clone()),
        };

        // 2. HEAD moved. Potentially every file differs, so nothing is trustworthy.
        if branch_info.commit_sha != current_commit {
            let short = |s: &str| s.chars().take(7).collect::<String>();
            let warning = IndexWarning::new(
                format!(
                    "Commit changed from {} to {}",
                    short(&branch_info.commit_sha),
                    short(&current_commit)
                ),
                "index_project",
            )
            .with_details(details);
            return Ok(Snapshot::Decided((
                IndexStatus::Stale,
                false,
                Some(warning),
            )));
        }

        // 3. The working tree. This is the case 1.7.1 missed entirely: it sampled the
        // mtimes of the first TEN indexed files, which never included an untracked
        // file (absent from the list) or a deleted one (metadata() fails, skipped
        // silently). Edit-then-search is the primary agent workflow, and every one of
        // those searches was served stale and labelled fresh.
        // Under the workspace's `[index] include/exclude` policy, so an edit to an
        // excluded file cannot mark the index stale.
        let policy = cache
            .load_index_config()
            .map(|cfg| crate::indexer::PathPolicy::from_config(root, &cfg))
            .unwrap_or_default();
        let indexable = |p: &str| indexable_with(p, &policy);
        let changes = match crate::git::get_worktree_changes(root, indexable) {
            Ok(c) => c,
            // git unavailable mid-session, or a broken repo. Don't claim staleness we
            // cannot demonstrate.
            Err(e) => {
                log::debug!("Could not read working tree state: {}", e);
                return Ok(Snapshot::Decided(fresh()));
            }
        };

        Ok(Snapshot::Worktree { details, changes })
    }

    /// Drop any memo for `root`, so the next read is fresh.
    ///
    /// Used by `check_index_status`, the explicit probe: an agent that asks whether
    /// the index is current must never be answered from a cache. Also called around
    /// every index write.
    pub fn invalidate(root: &Path) {
        let key = key(root);
        if let Ok(mut map) = store().lock() {
            map.remove(&key);
        }
    }
}
