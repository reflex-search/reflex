//! Query filter types and stateless filtering helpers

use regex::Regex;

use crate::models::SymbolKind;

/// Query filter options
#[derive(Debug, Clone)]
pub struct QueryFilter {
    /// Language filter (None = all languages)
    pub language: Option<crate::models::Language>,
    /// Symbol kind filter (None = all kinds)
    pub kind: Option<SymbolKind>,
    /// Use AST pattern matching (vs lexical search)
    pub use_ast: bool,
    /// Use regex pattern matching
    pub use_regex: bool,
    /// Maximum number of results
    pub limit: Option<usize>,
    /// Search symbol definitions only (vs full-text)
    pub symbols_mode: bool,
    /// Show full symbol body (from span.start_line to span.end_line)
    pub expand: bool,
    /// File path filter (substring match)
    pub file_pattern: Option<String>,
    /// Exact symbol name match (no substring matching)
    pub exact: bool,
    /// Use substring matching instead of word-boundary matching (opt-in, expansive)
    pub use_contains: bool,
    /// Query timeout in seconds (0 = no timeout)
    pub timeout_secs: u64,
    /// Glob patterns to include (empty = all files)
    pub glob_patterns: Vec<String>,
    /// Glob patterns to exclude (applied after includes)
    pub exclude_patterns: Vec<String>,
    /// Return only unique file paths (deduplicated)
    pub paths_only: bool,
    /// Pagination offset (skip first N results after sorting)
    pub offset: Option<usize>,
    /// Force execution of potentially expensive queries (bypass broad query detection)
    pub force: bool,
    /// Suppress warning/info output (for --json mode to ensure pure JSON output)
    pub suppress_output: bool,
    /// Include dependency information in results
    pub include_dependencies: bool,
    /// Number of context lines to show before and after each match (default: 0 = disabled)
    pub context_lines: usize,
    /// Test-only: Override large index threshold (None = use default of 20,000)
    #[doc(hidden)]
    pub test_large_index_threshold: Option<usize>,
    /// Test-only: Override short pattern threshold (None = use default of 4)
    #[doc(hidden)]
    pub test_short_pattern_threshold: Option<usize>,
}

impl Default for QueryFilter {
    fn default() -> Self {
        Self {
            language: None,
            kind: None,
            use_ast: false,
            use_regex: false,
            limit: Some(100),  // Default: limit to 100 results for token efficiency
            symbols_mode: false,
            expand: false,
            file_pattern: None,
            exact: false,
            use_contains: false,  // Default: word-boundary matching
            timeout_secs: 30, // 30 seconds default timeout
            glob_patterns: Vec::new(),
            exclude_patterns: Vec::new(),
            paths_only: false,
            offset: None,
            force: false,  // Default: enable broad query detection
            suppress_output: false,  // Default: show warnings/info
            include_dependencies: false,  // Default: don't load dependencies for performance
            context_lines: 0,  // Default: no context lines shown
            test_large_index_threshold: None,  // Default: use production threshold (20,000)
            test_short_pattern_threshold: None,  // Default: use production threshold (4)
        }
    }
}

/// Map a language keyword to its corresponding SymbolKind.
///
/// When users search for keywords like "class" or "function" with --symbols,
/// automatically infer the kind filter to return only symbols of that type.
pub fn keyword_to_kind(keyword: &str) -> Option<SymbolKind> {
    match keyword {
        "class" => Some(SymbolKind::Class),
        "struct" => Some(SymbolKind::Struct),
        "enum" => Some(SymbolKind::Enum),
        "interface" => Some(SymbolKind::Interface),
        "trait" => Some(SymbolKind::Trait),
        "type" => Some(SymbolKind::Type),
        "record" => Some(SymbolKind::Struct),  // C# record types
        "function" | "fn" | "def" | "func" => Some(SymbolKind::Function),
        "const" | "static" => Some(SymbolKind::Constant),
        "var" | "let" => Some(SymbolKind::Variable),
        "mod" | "module" | "namespace" => Some(SymbolKind::Module),
        "impl" | "async" => None,
        _ => None,
    }
}

/// Check if pattern appears at word boundaries in a line.
///
/// Used for default (restrictive) matching to find complete identifiers
/// rather than substrings.
pub fn has_word_boundary_match(line: &str, pattern: &str) -> bool {
    let escaped_pattern = regex::escape(pattern);
    let pattern_with_boundaries = format!(r"\b{}\b", escaped_pattern);

    if let Ok(re) = Regex::new(&pattern_with_boundaries) {
        re.is_match(line)
    } else {
        log::debug!("Word boundary regex failed for pattern '{}', falling back to substring", pattern);
        line.contains(pattern)
    }
}
