//! Core data models for Reflex
//!
//! These structures represent the normalized, deterministic output format
//! that Reflex provides to AI agents and other programmatic consumers.

use serde::{Deserialize, Serialize};
use strum::{EnumString, Display};

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
    Unknown,
}

impl Language {
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
            _ => None,
        }
    }

    /// Human-readable list of all supported language names (for error messages)
    pub fn supported_names_help() -> &'static str {
        "rust (rs), python (py), javascript (js), typescript (ts), vue, svelte, \
         go, java, php, c, cpp (c++), csharp (cs, c#), ruby (rb), kotlin (kt), zig"
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
            Language::Swift => false,  // Temporarily disabled - parser queries out of date with tree-sitter-swift 0.7.x grammar
            Language::Zig => true,
            Language::Unknown => false,
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
}

/// Dependency information for API output (simplified, path-based)
/// Note: Only internal dependencies are indexed (external/stdlib filtered during indexing)
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
    /// Query timeout in seconds (0 = no timeout)
    pub query_timeout_secs: u64,
    /// Maximum entries per trigram posting list (0 = unlimited).
    /// High-frequency trigrams are truncated at this threshold to bound query latency.
    pub max_posting_list_entries: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            languages: vec![],
            include_patterns: vec![],
            exclude_patterns: vec![],
            follow_symlinks: false,
            max_file_size: 10 * 1024 * 1024, // 10 MB
            parallel_threads: 0, // 0 = auto (80% of available cores)
            query_timeout_secs: 30, // 30 seconds default timeout
            max_posting_list_entries: 500_000, // cap at 500k to bound query latency
        }
    }
}

/// Statistics about the index
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Additional context (git branch info, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<IndexWarningDetails>,
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
}

/// Pagination information for query results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginationInfo {
    /// Total number of results (before offset/limit applied)
    pub total: usize,
    /// Number of results in this response (after offset/limit)
    pub count: usize,
    /// Offset used (starting position)
    pub offset: usize,
    /// Limit used (max results per page)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    /// Whether there are more results after this page
    pub has_more: bool,
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
            span: Span { start_line: 10, end_line: 20 },
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
            span: Span { start_line: 1, end_line: 5 },
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
            span: Span { start_line: 3, end_line: 7 },
        };
        let json = serde_json::to_string(&sym).unwrap();
        assert_eq!(
            json,
            r#"{"name":"Foo","kind":"Class","span":{"start_line":3,"end_line":7}}"#
        );
    }
}
