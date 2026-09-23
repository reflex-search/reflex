//! Core data models for Reflex
//!
//! These structures represent the normalized, deterministic output format
//! that Reflex provides to AI agents and other programmatic consumers.

use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};

/// Represents a source code location span (line range only)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Span {
    /// Starting line number (1-indexed)
    pub start_line: usize,
    /// Ending line number (1-indexed)
    pub end_line: usize,
}

impl Span {
    pub fn new(start_line: usize, start_col: usize, end_line: usize, end_col: usize) -> Self {
        // Ignore col parameters for backwards compatibility
        let _ = (start_col, end_col);
        Self {
            start_line,
            end_line,
        }
    }
}

/// Type of symbol found in code
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, EnumString, Display)]
#[strum(serialize_all = "PascalCase")]
pub enum SymbolKind {
    Function,
    Class,
    Struct,
    Enum,
    Interface,
    Trait,
    Constant,
    Variable,
    Method,
    Module,
    Namespace,
    Type,
    Macro,
    Property,
    Event,
    Import,
    Export,
    Attribute,
    /// Catch-all for symbol kinds not yet explicitly supported.
    /// This ensures no data loss when encountering new tree-sitter node types.
    /// The string contains the original kind name from the parser.
    #[strum(default)]
    Unknown(String),
}

/// Programming language identifier
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    #[default]
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Vue,
    Svelte,
    Go,
    Java,
    PHP,
    C,
    Cpp,
    CSharp,
    Ruby,
    Kotlin,
    Swift,
    Zig,
    /// Plain-text tier: documentation, config and templates.
    ///
    /// Trigram-indexed only — no tree-sitter grammar, no symbol extraction, no
    /// import extraction. Added because agents do not partition searches by file
    /// type: a config key lives in the YAML, the Rust struct AND the spec paragraph,
    /// and Reflex used to return the struct and a confident 0 for the rest.
    ///
    /// Serialises as `"text"` (the enum is `rename_all = "lowercase"`).
    Text,
    /// Lock files (`Cargo.lock`, `package-lock.json`, `*.lock`, …).
    ///
    /// Indexed since 1.8.0 in `[index] mode = "tracked"`, but excluded from every
    /// search by default: 100k lines of pinned versions are noise for almost every
    /// query and the one real question ("which lockfile pins serde 1.0.190?") is
    /// asked with `include_locks: true` or `lang: "lock"`.
    Lock,
    /// Generated files judged by name: `*.pb.go`, `*_generated.*`, `*.generated.*`,
    /// `*.min.js`, `*.min.css`, `*.map`. Indexed, excluded by default,
    /// `include_generated: true` or `lang: "generated"` to widen.
    Generated,
    /// Never produced by [`Language::from_path`] since 1.8.0 (every path classifies
    /// as code, `Text`, `Lock` or `Generated`); kept as the miss value of
    /// [`Language::from_extension`] and as the forward-compatible sentinel for
    /// readers of the `language` field.
    Unknown,
}

/// Whether `file_name` is a dependency lock file.
///
/// Judged by full name: `package-lock.json` matches a text extension, and the
/// name is the only thing that tells it apart from `settings.json`.
pub fn is_lock_file(file_name: &str) -> bool {
    if TEXT_FILENAME_EXCLUSIONS
        .iter()
        .any(|n| n.eq_ignore_ascii_case(file_name))
    {
        return true;
    }
    // `*-lock.json` and friends, beyond the names listed above.
    file_name.ends_with("-lock.json") || file_name.ends_with(".lock")
}

/// Whether `file_name` looks like a generated file, judged by name only.
///
/// Content markers (`@generated` in the file head) are not read: the query engine
/// derives a file's language from its path, so a content-based verdict at index
/// time could not be honoured at query time.
pub fn is_generated_name(file_name: &str) -> bool {
    let lower = file_name.to_ascii_lowercase();
    lower.ends_with(".pb.go")
        || lower.ends_with(".min.js")
        || lower.ends_with(".min.css")
        || lower.ends_with(".map")
        || lower.contains("_generated.")
        || lower.contains(".generated.")
}

/// Extensions in the plain-text tier.
///
/// A fixed allowlist, not "everything unrecognised": an index that swallowed every
/// binary blob and generated artefact in a repo would be slower and less useful.
const TEXT_EXTENSIONS: &[&str] = &[
    "md", "mdx", "txt", "yaml", "yml", "toml", "json", "proto", "html", "htm", "sh", "bash", "ini",
    "cfg", "sql", "graphql", "bru",
];

/// Extensionless files in the plain-text tier, matched by exact name.
///
/// Agents grep these as readily as any `.md`; a `Makefile` target or a `Dockerfile`
/// `COPY` line is a legitimate search hit. `Dockerfile.<variant>` is handled as a
/// prefix in [`is_text_tier_file`].
const TEXT_FILENAMES: &[&str] = &["Makefile", "makefile", "Dockerfile", "Justfile", "justfile"];

/// Lock files, by exact name. See [`is_lock_file`] for the suffix rules.
const TEXT_FILENAME_EXCLUSIONS: &[&str] = &[
    "package-lock.json",
    "npm-shrinkwrap.json",
    "composer.lock",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "poetry.lock",
    "Gemfile.lock",
    "go.sum",
    "flake.lock",
    "uv.lock",
    "deno.lock",
    "bun.lock",
];

/// Whether a file belongs in the ALLOWLIST text tier, judged by its full name.
///
/// This is the pre-1.8.0 rule, kept for `[index] mode = "allowlist"`. In the
/// default `tracked` mode every non-binary, non-ignored file is text unless it is
/// code, a lock file or a generated file.
pub fn is_text_tier_file(file_name: &str) -> bool {
    if is_lock_file(file_name) {
        return false;
    }
    if TEXT_FILENAMES.contains(&file_name) || file_name.starts_with("Dockerfile.") {
        return true;
    }
    match file_name.rsplit_once('.') {
        Some((_, ext)) => TEXT_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()),
        None => false,
    }
}

impl Language {
    /// Classify a file by its path.
    ///
    /// This is the one classifier the indexer, watcher and query engine share, so a
    /// file is either indexed, watched and searchable, or none of the three. Path
    /// only: whether the file is binary is decided from its bytes by the indexer,
    /// and a binary file is simply never in the index.
    ///
    /// * A lock file name wins (`Cargo.lock`, `package-lock.json`) → `Lock`.
    /// * A generated name wins next (`x.pb.go`, `app.min.js`) → `Generated`.
    /// * A recognised code extension → that language (`main.rs`, `app.mjs`).
    /// * Everything else → `Text`: `README`, `OWNERS`, `foo.po`, `a.css`,
    ///   `Makefile`, `.githooks/pre-commit`. Whether such a file is INDEXED is the
    ///   `[index] mode` policy's decision (`PathPolicy::classify`).
    pub fn from_path(path: &std::path::Path) -> Self {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if is_lock_file(name) {
            return Language::Lock;
        }
        if is_generated_name(name) {
            return Language::Generated;
        }
        let by_ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(Self::from_extension)
            .unwrap_or(Language::Unknown);
        match by_ext {
            Language::Text | Language::Unknown => Language::Text,
            code => code,
        }
    }

    pub fn from_extension(ext: &str) -> Self {
        match ext {
            "rs" => Language::Rust,
            "py" => Language::Python,
            "js" | "mjs" | "cjs" | "jsx" => Language::JavaScript,
            "ts" | "mts" | "cts" | "tsx" => Language::TypeScript,
            "vue" => Language::Vue,
            "svelte" => Language::Svelte,
            "go" => Language::Go,
            "java" => Language::Java,
            "php" => Language::PHP,
            "c" | "h" => Language::C,
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "C" | "H" => Language::Cpp,
            "cs" => Language::CSharp,
            "rb" | "rake" | "gemspec" => Language::Ruby,
            "kt" | "kts" => Language::Kotlin,
            "swift" => Language::Swift,
            "zig" => Language::Zig,
            // The text tier. Note this maps by EXTENSION only; `is_text_tier_file`
            // additionally excludes lock files by name, and the indexer uses that.
            ext if TEXT_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()) => Language::Text,
            _ => Language::Unknown,
        }
    }

    /// Parse a language from a human-friendly name (CLI/API input)
    ///
    /// Accepts lowercase names and common aliases.
    /// Returns None for unrecognized names.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "rust" | "rs" => Some(Language::Rust),
            "python" | "py" => Some(Language::Python),
            "javascript" | "js" => Some(Language::JavaScript),
            "typescript" | "ts" => Some(Language::TypeScript),
            "vue" => Some(Language::Vue),
            "svelte" => Some(Language::Svelte),
            "go" => Some(Language::Go),
            "java" => Some(Language::Java),
            "php" => Some(Language::PHP),
            "c" => Some(Language::C),
            "cpp" | "c++" => Some(Language::Cpp),
            "csharp" | "cs" | "c#" => Some(Language::CSharp),
            "ruby" | "rb" => Some(Language::Ruby),
            "kotlin" | "kt" => Some(Language::Kotlin),
            "zig" => Some(Language::Zig),
            "text" | "txt" | "plaintext" | "plain" => Some(Language::Text),
            "lock" | "lockfile" | "lockfiles" => Some(Language::Lock),
            "generated" | "gen" => Some(Language::Generated),
            _ => None,
        }
    }

    /// Human-readable list of all supported language names (for error messages)
    pub fn supported_names_help() -> &'static str {
        "rust (rs), python (py), javascript (js), typescript (ts), vue, svelte, \
         go, java, php, c, cpp (c++), csharp (cs, c#), ruby (rb), kotlin (kt), zig, \
         text (every other non-binary file: docs, config, templates, extensionless), \
         lock (lock files, excluded by default), generated (*.pb.go, *.min.js, *.map, \
         *_generated.*, excluded by default)"
    }

    /// Check if this language has a parser implementation
    ///
    /// Returns true only for languages with working Tree-sitter parsers.
    /// This determines which files will be indexed by Reflex.
    pub fn is_supported(&self) -> bool {
        match self {
            Language::Rust => true,
            Language::TypeScript => true,
            Language::JavaScript => true,
            Language::Vue => true,
            Language::Svelte => true,
            Language::Python => true,
            Language::Go => true,
            Language::Java => true,
            Language::PHP => true,
            Language::C => true,
            Language::Cpp => true,
            Language::CSharp => true,
            Language::Ruby => true,
            Language::Kotlin => true,
            Language::Swift => false, // Temporarily disabled - parser queries out of date with tree-sitter-swift 0.7.x grammar
            Language::Zig => true,
            // No tree-sitter grammar, by design.
            Language::Text => false,
            Language::Lock => false,
            Language::Generated => false,
            Language::Unknown => false,
        }
    }

    /// Whether this is the plain-text tier.
    pub fn is_text(&self) -> bool {
        matches!(self, Language::Text)
    }

    /// Whether files of this language are indexed but left out of every search
    /// unless asked for (`include_locks` / `include_generated`, or `lang`).
    pub fn is_excluded_by_default(&self) -> bool {
        matches!(self, Language::Lock | Language::Generated)
    }

    /// Whether this is a code language, supported or not (Swift is code without a
    /// working grammar). Code files are assumed to be text; every other file is
    /// sniffed for a NUL byte before it is indexed.
    pub fn is_code(&self) -> bool {
        !matches!(
            self,
            Language::Text | Language::Lock | Language::Generated | Language::Unknown
        )
    }

    /// Whether files of this language can be in the index at all.
    ///
    /// Distinct from [`Self::is_supported`], which means "has a tree-sitter parser".
    /// The text tier is indexed but never parsed, so symbol search, AST queries and
    /// dependency analysis skip it while full-text search covers it. Whether a given
    /// path IS indexed also depends on `[index] mode` (`PathPolicy::classify`).
    pub fn is_indexable(&self) -> bool {
        !matches!(self, Language::Unknown)
    }
}

/// Which files the indexer takes, beyond the code languages.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum IndexMode {
    /// Every file not ignored by `.gitignore` / `.ignore` / `[index] exclude`,
    /// unless a NUL byte anywhere in it says it is binary. Lock and generated
    /// files are indexed and excluded from searches by default. This is ripgrep's
    /// rule, and what an agent that greps expects.
    #[default]
    Tracked,
    /// The pre-1.8.0 rule: code by extension plus the fixed docs/config extension
    /// list (`is_text_tier_file`). Lock and generated files are not indexed. For
    /// trees where the long tail of data files is not worth the index size.
    Allowlist,
}

impl IndexMode {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "tracked" => Some(Self::Tracked),
            "allowlist" => Some(Self::Allowlist),
            _ => None,
        }
    }
}

/// Type of import/dependency
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ImportType {
    /// Internal project file
    Internal,
    /// External library/package
    External,
    /// Standard library
    Stdlib,
    /// Rust `mod foo;` declaration (parent→child ownership, not a usage edge)
    #[serde(rename = "mod_decl")]
    ModDecl,
}

/// Dependency information for API output (simplified, path-based)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyInfo {
    /// Import path as written in source (or resolved path for internal deps)
    pub path: String,
    /// Line number where import appears (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
    /// Imported symbols (for selective imports like `from x import a, b`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbols: Option<Vec<String>>,
}

/// Full dependency record (internal representation with file IDs)
#[derive(Debug, Clone)]
pub struct Dependency {
    /// Source file ID
    pub file_id: i64,
    /// Import path as written in source code
    pub imported_path: String,
    /// Resolved file ID (None if external or stdlib)
    pub resolved_file_id: Option<i64>,
    /// Import type classification
    pub import_type: ImportType,
    /// Line number where import appears
    pub line_number: usize,
    /// Imported symbols (for selective imports)
    pub imported_symbols: Option<Vec<String>>,
}

/// A lightweight, stable reference to a code symbol for API responses
///
/// Prefer this over `(String, SymbolKind, Span)` tuples — tuples serialize as
/// positional JSON arrays, making any field addition a breaking change.
/// Named fields here are additive-safe: new optional fields can be added without
/// shifting positions or bumping the version.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SymbolRef {
    /// Symbol name (e.g., function name, class name)
    pub name: String,
    /// Symbol kind (function, class, struct, etc.)
    pub kind: SymbolKind,
    /// Location span in source file
    pub span: Span,
}

/// Helper function to skip serializing "Unknown" symbol kinds
fn is_unknown_kind(kind: &SymbolKind) -> bool {
    matches!(kind, SymbolKind::Unknown(_))
}

/// A search result representing a symbol or code location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Absolute or relative path to the file
    pub path: String,
    /// Detected programming language (internal use only, not serialized to save tokens)
    #[serde(skip)]
    pub lang: Language,
    /// Type of symbol found (only included for symbol searches, not text matches)
    #[serde(skip_serializing_if = "is_unknown_kind")]
    pub kind: SymbolKind,
    /// Symbol name (e.g., function name, class name)
    /// None for text/regex matches where symbol name cannot be accurately determined
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
    /// Location span in the source file
    pub span: Span,
    /// Code preview (few lines around the match)
    pub preview: String,
    /// File dependencies (only populated when --dependencies flag is used)
    /// DEPRECATED: Use FileGroupedResult.dependencies instead for file-level grouping
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dependencies: Option<Vec<DependencyInfo>>,
}

/// An individual match within a file (no path or dependencies)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    /// Type of symbol found (only included for symbol searches, not text matches)
    #[serde(skip_serializing_if = "is_unknown_kind")]
    pub kind: SymbolKind,
    /// Symbol name (e.g., function name, class name)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
    /// Location span in the source file
    pub span: Span,
    /// Code preview (few lines around the match)
    pub preview: String,
    /// Lines of code before the match (for context)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub context_before: Vec<String>,
    /// Lines of code after the match (for context)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub context_after: Vec<String>,
}

/// File-level grouped results with dependencies at file level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileGroupedResult {
    /// Absolute or relative path to the file
    pub path: String,
    /// Detected programming language of this file (e.g. "rust", "python", "unknown")
    pub language: Language,
    /// File dependencies (only populated when --dependencies flag is used)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dependencies: Option<Vec<DependencyInfo>>,
    /// Individual matches within this file
    pub matches: Vec<MatchResult>,
}

impl SearchResult {
    pub fn new(
        path: String,
        lang: Language,
        kind: SymbolKind,
        symbol: Option<String>,
        span: Span,
        scope: Option<String>,
        preview: String,
    ) -> Self {
        // Ignore scope parameter for backwards compatibility
        let _ = scope;
        Self {
            path,
            lang,
            kind,
            symbol,
            span,
            preview,
            dependencies: None,
        }
    }
}

/// Configuration for indexing behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Languages to include (empty = all supported)
    pub languages: Vec<Language>,
    /// Glob patterns to include
    pub include_patterns: Vec<String>,
    /// Glob patterns to exclude
    pub exclude_patterns: Vec<String>,
    /// Follow symbolic links
    pub follow_symlinks: bool,
    /// Maximum file size to index (bytes)
    pub max_file_size: usize,
    /// Number of threads for parallel indexing (0 = auto, 80% of available cores)
    pub parallel_threads: usize,
    /// Threads for the background symbol pass (`rfx index-symbols-internal`).
    /// `0` = auto: half the cores, at most 32. See [`resolve_symbol_thread_count`].
    #[serde(default)]
    pub symbol_threads: usize,
    /// Query timeout in seconds (0 = no timeout)
    pub query_timeout_secs: u64,
    /// Maximum entries per trigram posting list (0 = unlimited).
    /// High-frequency trigrams are truncated at this threshold to bound query latency.
    pub max_posting_list_entries: usize,
    /// How long `Indexer::index` waits for `.reflex/index.lock` when another
    /// indexer holds it (seconds). 0 = fail immediately with `IndexLocked`.
    /// The `rfx index` CLI waits; MCP, watcher and HTTP callers fail fast.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub lock_wait_secs: u64,
    /// Index documentation, config and template files alongside code.
    ///
    /// On by default. Covers md, mdx, txt, yaml, yml, toml, json, proto, html, htm,
    /// sh, bash, ini, cfg, sql and graphql, trigram-indexed only — no symbols, no
    /// AST, no dependency analysis. Lock files are always excluded.
    ///
    /// Set `[index] text_tier = false` for a repo with large generated JSON or
    /// vendored documentation where the index growth is not worth it.
    ///
    /// `#[serde(default = ...)]` so a config file written before 1.7.2 still parses
    /// and gets the new default.
    #[serde(default = "default_true")]
    pub text_tier: bool,
    /// Which non-code files the text tier takes: every non-binary, non-hidden, non-ignored file
    /// (`tracked`, the default since 1.8.0) or the fixed extension list
    /// (`allowlist`, the pre-1.8.0 rule).
    #[serde(default)]
    pub mode: IndexMode,
    /// Walk dot-directories and dotfiles too (`.githooks/pre-commit`, `.env.example`).
    /// Off by default, like ripgrep without `--hidden`. `.git/` and `.reflex/` are
    /// never walked.
    #[serde(default)]
    pub hidden: bool,
}

/// Serde default for boolean options that are on unless explicitly disabled.
fn default_true() -> bool {
    true
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            languages: vec![],
            include_patterns: vec![],
            exclude_patterns: vec![],
            follow_symlinks: false,
            max_file_size: 10 * 1024 * 1024,   // 10 MB
            parallel_threads: 0,               // 0 = auto (80% of available cores)
            symbol_threads: 0,                 // 0 = auto (50% of available cores)
            query_timeout_secs: 30,            // 30 seconds default timeout
            max_posting_list_entries: 500_000, // cap at 500k to bound query latency
            text_tier: true,                   // docs and config are searchable by default
            mode: IndexMode::Tracked, // ripgrep defaults: not ignored, not hidden, not binary
            hidden: false,            // dot-directories skipped, like ripgrep
            lock_wait_secs: 0,        // fail fast when another indexer runs
        }
    }
}

fn is_zero(v: &usize) -> bool {
    *v == 0
}
fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}

/// Resolve `[performance] symbol_threads` (the background symbol pass) to a
/// concrete thread count.
///
/// `REFLEX_SYMBOL_THREADS` overrides for benchmarking, then `configured` if
/// non-zero, else half the cores (1..=32). The pass is detached from `rfx index`
/// and runs while the user may be querying, so it takes half the machine rather
/// than the indexer's 80%; before 1.8.1 it took 27.5%.
pub fn resolve_symbol_thread_count(configured: usize) -> usize {
    if let Some(n) = std::env::var("REFLEX_SYMBOL_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return n;
    }
    if configured != 0 {
        return configured.max(1);
    }
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    ((available as f64 * 0.5).ceil() as usize).clamp(1, 32)
}

/// Resolve `[performance] parallel_threads` to a concrete thread count.
///
/// `0` means automatic: 80% of the available cores, at least 1, at most
/// `auto_cap`. A non-zero value is used as given. The indexer passes a cap of 8
/// (write-side cache contention); query-time verification passes a higher cap.
pub fn resolve_thread_count(configured: usize, auto_cap: usize) -> usize {
    if configured != 0 {
        return configured.max(1);
    }
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    ((available as f64 * 0.8).ceil() as usize).clamp(1, auto_cap.max(1))
}

/// How a query found its candidate lines.
///
/// `trigram` is the normal case: the pattern's literals were looked up in the
/// inverted index and only the lines they name were verified. `scan` means every
/// line of every file was verified, which happens for a pattern shorter than
/// 3 chars, a regex with no literal of 3+ chars (`\w+_id`), a non-ASCII literal
/// under `(?i)`, or a keyword symbol query. Before 1.8.0 every `(?i)` regex
/// scanned; since 1.8.0 its literals are looked up under all case variants.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum IndexPath {
    /// Candidates came from the trigram index.
    #[default]
    Trigram,
    /// Every line was verified.
    Scan,
}

/// Per-phase wall-clock timings for one query, in microseconds.
///
/// Present in a [`QueryResponse`] only when the caller asked for it
/// (`rfx query --timing`, or `REFLEX_MCP_TIMING=1` for the MCP server).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct QueryTimings {
    /// Whether the trigram index or a full scan produced the candidates.
    #[serde(default)]
    pub index_path: IndexPath,
    /// Opening (or reusing) the index handle.
    pub open_us: u64,
    /// Trigram lookup and posting-list intersection.
    pub candidates_us: u64,
    /// Verifying candidate lines against the pattern (and any enrichment).
    pub verify_us: u64,
    /// Time the query waited for the freshness check after the search finished.
    /// The check runs on its own thread alongside the search, so this is usually
    /// near zero; `status_compute_us` is what the check itself cost.
    pub status_us: u64,
    /// The freshness check's own duration (git spawns, or a tree walk outside git).
    #[serde(default)]
    pub status_compute_us: u64,
    /// Grouping by file, context lines, dependencies.
    pub group_us: u64,
    /// Whole query as seen by the engine.
    pub total_us: u64,
}

/// Statistics about the index
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IndexStats {
    /// Total files indexed
    pub total_files: usize,
    /// Index size on disk (bytes)
    pub index_size_bytes: u64,
    /// Last update timestamp
    pub last_updated: String,
    /// File count breakdown by language
    pub files_by_language: std::collections::HashMap<String, usize>,
    /// Line count breakdown by language
    pub lines_by_language: std::collections::HashMap<String, usize>,
    /// New files added since last index run (0 if not an incremental run)
    #[serde(default, skip_serializing_if = "is_zero")]
    pub new_files: usize,
    /// Modified files re-indexed since last run (0 if not an incremental run)
    #[serde(default, skip_serializing_if = "is_zero")]
    pub modified_files: usize,
    /// Unchanged files (same hash as last run, still re-indexed due to other changes)
    #[serde(default, skip_serializing_if = "is_zero")]
    pub unchanged_files: usize,
    /// Files dropped from the index because they no longer exist on disk
    #[serde(default, skip_serializing_if = "is_zero")]
    pub deleted_files: usize,
    /// Files skipped because they exceeded max_file_size
    #[serde(default, skip_serializing_if = "is_zero")]
    pub skipped_too_large: usize,
    /// Total bytes of files skipped due to max_file_size
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub skipped_bytes_too_large: u64,
    /// Files skipped because a NUL byte says they are binary (ripgrep's rule)
    #[serde(default, skip_serializing_if = "is_zero")]
    pub skipped_binary: usize,
    /// Raw bytes of indexed source held in content.bin (0 if unknown)
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub corpus_bytes: u64,
    /// Size of trigrams.bin on disk (0 if absent)
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub trigram_index_bytes: u64,
}

/// Information about an indexed file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedFile {
    /// File path
    pub path: String,
    /// Detected language
    pub language: String,
    /// Last indexed timestamp
    pub last_indexed: String,
}

/// Index status for query responses
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IndexStatus {
    /// Index is fresh and up-to-date
    Fresh,
    /// Index is stale (any issue: branch not indexed, commit changed, files modified)
    Stale,
}

/// Warning details when index is stale
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexWarning {
    /// Human-readable reason why index is stale
    pub reason: String,
    /// Command to run to fix the issue
    pub action_required: String,
    /// Tracked files edited since the index was built.
    ///
    /// BREAKING in 1.7.2: this was a `u32` count. It is now the paths themselves,
    /// because a count told an agent something was wrong without telling it what, so
    /// the only safe reaction was to distrust the whole result. The list is capped;
    /// `truncated` says when.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files_modified: Option<Vec<String>>,
    /// Files present on disk but absent from the index (new or untracked).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files_added: Option<Vec<String>>,
    /// Files in the index but no longer on disk. These produce ghost hits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files_deleted: Option<Vec<String>>,
    /// Total changed paths, which may exceed the lengths of the lists above.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub changed_count: Option<usize>,
    /// Whether the lists were cut short. Set on a fresh checkout or a huge rebase.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
    /// Additional context (git branch info, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<IndexWarningDetails>,
}

impl IndexWarning {
    /// A warning with only a reason and an action, no file lists.
    pub fn new(reason: impl Into<String>, action_required: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            action_required: action_required.into(),
            files_modified: None,
            files_added: None,
            files_deleted: None,
            changed_count: None,
            truncated: false,
            details: None,
        }
    }

    /// Attach git branch/commit context.
    pub fn with_details(mut self, details: IndexWarningDetails) -> Self {
        self.details = Some(details);
        self
    }
}

/// Detailed information about index staleness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexWarningDetails {
    /// Current branch (if in git repo)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_branch: Option<String>,
    /// Indexed branch (if in git repo)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indexed_branch: Option<String>,
    /// Current commit SHA (if in git repo)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_commit: Option<String>,
    /// Indexed commit SHA (if in git repo)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indexed_commit: Option<String>,
    /// When the index was last written (Unix seconds).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indexed_at: Option<i64>,
    /// How the working tree was compared to the index: `"git"` (candidates from
    /// `git status`, confirmed by fingerprint) or `"walk"` (every file stat'ed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked_by: Option<String>,
}

/// The full answer to `check_index_status`.
///
/// `details` is present even when the index is fresh, so a human can see that the
/// indexed commit differs from HEAD without that difference being called staleness:
/// since 1.8.0 freshness is judged by file content, not by commit.
#[derive(Debug, Clone)]
pub struct IndexStatusReport {
    pub status: IndexStatus,
    pub can_trust_results: bool,
    pub warning: Option<IndexWarning>,
    pub details: Option<IndexWarningDetails>,
}

/// Pagination information for query results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginationInfo {
    /// Total number of results before offset/limit — **`null` whenever
    /// `total_is_exact` is false**. A list-mode search with a `limit` stops
    /// verifying once the page is full, and the number verified by then is not a
    /// total; reporting it as one made agents stop paginating early (1.8.0 field
    /// test: `total: 851` for a term with 18,752 matches). Use `approx_total` for an
    /// estimate, or a count-mode / no-limit search for the exact number.
    #[serde(default)]
    pub total: Option<usize>,
    /// Number of results in this response (after offset/limit)
    pub count: usize,
    /// Offset used (starting position)
    pub offset: usize,
    /// Limit used (max results per page)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    /// Whether there are more results after this page
    pub has_more: bool,
    /// `true` when `total` counts every match. `false` when a list-mode search
    /// stopped verifying once the page was full: `total` is then `null` and
    /// `approx_total` carries an estimate. Count mode, no-limit searches,
    /// symbol/AST searches and `require_exact_total` callers are always exact.
    #[serde(default = "default_true")]
    pub total_is_exact: bool,
    /// Estimated total when `total_is_exact` is false. After the page filled, a
    /// spread sample of the remaining candidate lines was verified and the
    /// measured hit rate scaled over the rest (see
    /// `query::ESTIMATE_SAMPLE_FILES`). Typically within ±30% on the synthetic
    /// corpus; wider when hits cluster in a few files. Absent when no estimate was
    /// possible (a regex with no literal, where every line is a candidate).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approx_total: Option<usize>,
}

impl PaginationInfo {
    /// The total, only when it is a real one.
    pub fn exact_total(&self) -> Option<usize> {
        if self.total_is_exact {
            self.total
        } else {
            None
        }
    }

    /// The best number available for a threshold or a log line: the exact total,
    /// else the estimate, else the end of this page. Never show it as a total.
    pub fn best_total(&self) -> usize {
        self.exact_total()
            .or(self.approx_total)
            .unwrap_or(self.offset + self.count)
    }
}

/// Query response with results and index status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    /// AI-optimized instruction for how to handle these results
    /// Only present when --ai flag is used or in MCP mode
    /// Provides guidance to AI agents on response format and next actions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ai_instruction: Option<String>,
    /// Status of the index (fresh or stale)
    pub status: IndexStatus,
    /// Whether the results can be trusted
    pub can_trust_results: bool,
    /// Warning information (only present if stale)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<IndexWarning>,
    /// Pagination information
    pub pagination: PaginationInfo,
    /// File-grouped search results
    /// Results are always grouped by file path, with dependencies populated when --dependencies flag is used
    pub results: Vec<FileGroupedResult>,
    /// For a whole-identifier search that found nothing: how many candidate lines
    /// contain the pattern as a substring. Lets the caller explain a zero without
    /// running a second search. Absent whenever there were results, or when the
    /// search was not a whole-identifier one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub substring_hint_count: Option<usize>,
    /// Things the engine did to the query that the caller should know about.
    ///
    /// Today this is one message at most: a whole-identifier pattern containing
    /// brackets was escaped and run as a regex (see
    /// `query::prepare_literal_pattern`). Every surface — CLI, MCP, HTTP — carries it,
    /// so a rewrite is never silent on any of them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
    /// For a whole-identifier search that found nothing while substring matches
    /// exist: a ready-to-show sentence naming the count and the switch that shows
    /// them. Absent whenever there were results, or when the search was not a
    /// whole-identifier one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
    /// Machine-readable cause behind `hint`: `hidden` (the filter names a
    /// dot-directory), `not_indexed` (the `file` filter names a path the index does
    /// not hold), `lock_or_generated`, `whole_identifier`. Absent when no rule
    /// applies, so a harness can branch without parsing prose.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub excluded_reason: Option<crate::query::ExcludedReason>,
    /// For a search that found nothing: how many candidate files were lock or
    /// generated files, which every search leaves out unless `include_locks` /
    /// `include_generated` (or `lang`) asks for them. The `hint` says so too.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub excluded_by_default: Option<usize>,
    /// Count-only searches: the number of files with at least one match. `results`
    /// is empty in that mode, so this is the only place the file count lives.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_count: Option<usize>,
    /// Per-phase timings, only when requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timings: Option<QueryTimings>,
}

/// Report from cache compaction operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionReport {
    /// Number of files removed
    pub files_removed: usize,
    /// Space saved in bytes
    pub space_saved_bytes: u64,
    /// Duration in milliseconds
    pub duration_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_ref_json_shape() {
        let sym = SymbolRef {
            name: "my_function".to_string(),
            kind: SymbolKind::Function,
            span: Span {
                start_line: 10,
                end_line: 20,
            },
        };
        let json = serde_json::to_value(&sym).unwrap();
        assert_eq!(json["name"], "my_function");
        assert_eq!(json["kind"], "Function");
        assert_eq!(json["span"]["start_line"], 10);
        assert_eq!(json["span"]["end_line"], 20);
        assert!(json.as_array().is_none());
    }

    #[test]
    fn test_symbol_ref_roundtrip() {
        let original = SymbolRef {
            name: "MyStruct".to_string(),
            kind: SymbolKind::Struct,
            span: Span {
                start_line: 1,
                end_line: 5,
            },
        };
        let json = serde_json::to_string(&original).unwrap();
        let decoded: SymbolRef = serde_json::from_str(&json).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_symbol_ref_exact_json() {
        let sym = SymbolRef {
            name: "Foo".to_string(),
            kind: SymbolKind::Class,
            span: Span {
                start_line: 3,
                end_line: 7,
            },
        };
        let json = serde_json::to_string(&sym).unwrap();
        assert_eq!(
            json,
            r#"{"name":"Foo","kind":"Class","span":{"start_line":3,"end_line":7}}"#
        );
    }
}
