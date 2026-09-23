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
    /// Match letters regardless of case (`rg -i`). Applies to whole-identifier,
    /// substring and regex line searches alike; the engine runs the query as a
    /// `(?i)` regex whose literals are looked up under every case variant.
    pub ignore_case: bool,
    /// Set by the engine when it rewrote a literal pattern into a regex (brackets,
    /// `ignore_case`): the original literal. Drives the broad-query guard (which
    /// must judge the literal's length, not the regex's) and the result `kind`.
    #[doc(hidden)]
    pub rewritten_from: Option<String>,
    /// Drop plain-text-tier results (docs, config, templates) from this query.
    ///
    /// The tier is included by default, because an agent searching for a config key
    /// wants the YAML as well as the struct. It is excluded for questions that are
    /// only meaningful about code — `find_references` in particular, whose reference
    /// search is a plain trigram scan and would otherwise return the name in a
    /// changelog entry as a "call site".
    pub exclude_text: bool,
    /// Search lock files too (`Cargo.lock`, `package-lock.json`, …). They are
    /// indexed but left out of every search by default; `lang: "lock"` selects
    /// them alone.
    pub include_locks: bool,
    /// Search generated files too (`*.pb.go`, `*.min.js`, `*.map`, …). Same
    /// default as lock files; `lang: "generated"` selects them alone.
    pub include_generated: bool,
    /// Count matches without materialising them: no previews, no grouping, no
    /// result rows. `total_count` / `pagination.total` carry the exact line count
    /// and `QueryResponse.file_count` the files with a match. Honoured only by the
    /// full-text line searches (trigram and regex) with no `limit`; symbol and AST
    /// searches ignore it. Set by `--count`, `mode: "count"` and `count_occurrences`.
    pub count_only: bool,
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
    /// Record per-phase timings into `QueryResponse.timings`.
    pub collect_timings: bool,
    /// Verify every candidate even with a `limit`, so `pagination.total` is exact.
    /// Set by callers whose contract promises an exact total for a page
    /// (`find_references`). Without it, a list-mode search stops once the page is
    /// full and reports `total_is_exact: false`.
    pub require_exact_total: bool,
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
            limit: Some(100), // Default: limit to 100 results for token efficiency
            symbols_mode: false,
            expand: false,
            file_pattern: None,
            exact: false,
            use_contains: false, // Default: word-boundary matching
            ignore_case: false,  // Default: case-sensitive
            rewritten_from: None,
            exclude_text: false,      // Default: docs and config are searched too
            include_locks: false,     // Default: lock files stay out
            include_generated: false, // Default: generated files stay out
            count_only: false,        // Default: materialise results
            timeout_secs: 30,         // 30 seconds default timeout
            glob_patterns: Vec::new(),
            exclude_patterns: Vec::new(),
            paths_only: false,
            offset: None,
            force: false,                       // Default: enable broad query detection
            suppress_output: false,             // Default: show warnings/info
            include_dependencies: false,        // Default: don't load dependencies for performance
            context_lines: 0,                   // Default: no context lines shown
            collect_timings: false,             // Default: no per-phase timings
            require_exact_total: false,         // Default: list mode may stop early
            test_large_index_threshold: None,   // Default: use production threshold (20,000)
            test_short_pattern_threshold: None, // Default: use production threshold (4)
        }
    }
}

impl QueryFilter {
    /// Whether this query is a count-only full-text LINE search: the one shape
    /// the verifier can answer with two numbers instead of result rows.
    pub fn count_only_line_search(&self) -> bool {
        self.count_only
            && self.limit.is_none()
            && !self.symbols_mode
            && !self.use_ast
            && self.kind.is_none()
    }
}

/// Whether a file of language `lang` belongs in this query's results.
///
/// An explicit `want` selects exactly that language (so `lang: "lock"` is how a
/// caller asks for lock files alone). Otherwise the text tier is in unless
/// `exclude_text`, and the lock / generated tiers are out unless asked for.
pub fn tier_admits(
    lang: crate::models::Language,
    want: Option<crate::models::Language>,
    exclude_text: bool,
    include_locks: bool,
    include_generated: bool,
) -> bool {
    use crate::models::Language;
    if let Some(want) = want {
        return lang == want;
    }
    match lang {
        Language::Text => !exclude_text,
        Language::Lock => include_locks,
        Language::Generated => include_generated,
        _ => true,
    }
}

/// Whether `lang` was left out only because nothing asked for it: the count of
/// such candidate files explains a zero result.
pub fn excluded_by_default(
    lang: crate::models::Language,
    want: Option<crate::models::Language>,
    include_locks: bool,
    include_generated: bool,
) -> bool {
    use crate::models::Language;
    want.is_none()
        && match lang {
            Language::Lock => !include_locks,
            Language::Generated => !include_generated,
            _ => false,
        }
}

/// Explain a zero result when the only candidate files were excluded by default.
pub fn excluded_by_default_hint_text(files: usize) -> String {
    format!(
        "{} candidate file(s) were lock or generated files, which every search leaves \
         out by default — pass include_locks:true / include_generated:true \
         (--include-locks / --include-generated on the CLI), or lang:\"lock\" / \
         lang:\"generated\", to search them.",
        files
    )
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
        "record" => Some(SymbolKind::Struct), // C# record types
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
        log::debug!(
            "Word boundary regex failed for pattern '{}', falling back to substring",
            pattern
        );
        line.contains(pattern)
    }
}

/// Bracket characters that make a whole-token literal search structurally unmatchable.
///
/// Whole-token matching wraps the pattern as `\b<pattern>\b`. A pattern ending in `)`
/// or `>` can never satisfy the trailing `\b`, because the next character is almost
/// never a word character. So `unwrap()`, `#[derive(` and `-> Result<` all returned a
/// silent `0` in the 1.7.0 field test, against ripgrep counts of 1221, 1141 and 2139.
/// The 1.7.2 fix lived only in the MCP layer; the CLI and HTTP surfaces kept
/// returning the confident zero until 1.8.0, when the rewrite moved here.
pub const REGEX_ONLY_CHARS: &[char] = &['(', ')', '[', ']', '{', '}', '<', '>'];

/// A literal pattern rewritten so it cannot silently return zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiteralPattern {
    /// What to actually search for (regex-escaped when `use_regex` is set).
    pub effective: String,
    /// Whether the query must run down the regex path.
    pub use_regex: bool,
    /// Explanation for the caller when the pattern was rewritten.
    pub warning: Option<String>,
}

/// Make a literal pattern searchable, rather than letting it return a confident zero.
///
/// A pattern containing brackets cannot match under whole-token rules, so it is
/// escaped and routed to the regex path, which is substring-based. The caller is told
/// in `QueryResponse::warnings` — the rewrite is never silent.
///
/// `ignore_case` is also a rewrite: the query becomes a `(?i)` regex (the caller's
/// own regex verbatim; a substring search escaped; a whole-identifier search
/// escaped and wrapped in `\b…\b`, the same wrap the literal matcher applies).
/// The regex path then looks the literals up under every case variant, so
/// `ignore_case` costs about what the case-sensitive query costs. This rewrite
/// carries no warning. The zero-result substring hint is not produced under
/// `ignore_case`, because the regex path does not count substring-only lines.
///
/// No rewrite happens when the query is not a line search at all: symbol names
/// never contain brackets, and an AST pattern is an S-expression whose parens are
/// the syntax. A regex or substring query is never bracket-rewritten.
pub fn prepare_literal_pattern(pattern: &str, filter: &QueryFilter) -> LiteralPattern {
    let not_a_line_search = filter.use_ast || filter.symbols_mode || filter.kind.is_some();
    let not_a_literal_line_search = filter.use_regex || filter.use_contains || not_a_line_search;

    let mut out = if not_a_literal_line_search || !pattern.contains(REGEX_ONLY_CHARS) {
        LiteralPattern {
            effective: pattern.to_string(),
            use_regex: filter.use_regex,
            warning: None,
        }
    } else {
        LiteralPattern {
            effective: regex::escape(pattern),
            use_regex: true,
            warning: Some(format!(
                "Pattern {:?} contains brackets, which a whole-identifier search can never \
                 match. Searched it as an escaped regex instead (substring semantics). For \
                 explicit control use a regex search (search_regex / --regex) or substring \
                 mode (contains:true / --contains).",
                pattern
            )),
        }
    };

    if filter.ignore_case && !not_a_line_search {
        let body = if filter.use_regex {
            pattern.to_string()
        } else if filter.use_contains || out.use_regex {
            regex::escape(pattern)
        } else {
            format!(r"\b{}\b", regex::escape(pattern))
        };
        out.effective = format!("(?i){}", body);
        out.use_regex = true;
    }

    out
}

/// Explain a zero-result whole-identifier search by naming the substring count.
///
/// This is the single line that would have prevented every wrong conclusion in the
/// 1.7.0 field test: an agent that sees `0` for `verify_csrf` concludes "no callers"
/// and acts on it, when 89 lines contain `verify_csrf_form_field`.
pub fn substring_hint_text(count: usize, pattern: &str) -> String {
    format!(
        "0 whole-identifier matches; {} substring matches — pass contains:true \
         (--contains on the CLI) to see them. Reflex matches whole identifiers by \
         default, so {:?} does not match longer names that merely contain it.",
        count, pattern
    )
}

#[cfg(test)]
mod literal_pattern_tests {
    use super::*;

    fn plain() -> QueryFilter {
        QueryFilter::default()
    }

    #[test]
    fn bracket_patterns_are_escaped_and_routed_to_regex() {
        for p in [
            "unwrap()",
            "#[derive(",
            "-> Result<",
            "RealmId::nil()",
            "vec![]",
        ] {
            let got = prepare_literal_pattern(p, &plain());
            assert!(got.use_regex, "{p} must run as a regex");
            assert_eq!(got.effective, regex::escape(p));
            let w = got.warning.expect("rewrite must be reported");
            assert!(w.contains("bracket"), "{w}");
            assert!(w.contains(p), "warning names the original pattern: {w}");
        }
    }

    #[test]
    fn plain_identifiers_are_untouched() {
        let got = prepare_literal_pattern("verify_csrf", &plain());
        assert_eq!(
            got,
            LiteralPattern {
                effective: "verify_csrf".into(),
                use_regex: false,
                warning: None
            }
        );
    }

    #[test]
    fn ignore_case_becomes_a_case_insensitive_regex() {
        let ic = QueryFilter {
            ignore_case: true,
            ..plain()
        };
        // Whole identifier: escaped and word-bounded, no warning.
        let got = prepare_literal_pattern("realm.id", &ic);
        assert_eq!(got.effective, r"(?i)\brealm\.id\b");
        assert!(got.use_regex);
        assert!(got.warning.is_none());

        // Substring: escaped only.
        let got = prepare_literal_pattern(
            "realm.id",
            &QueryFilter {
                use_contains: true,
                ..ic.clone()
            },
        );
        assert_eq!(got.effective, r"(?i)realm\.id");
        assert!(got.warning.is_none());

        // The caller's own regex is kept verbatim.
        let got = prepare_literal_pattern(
            r"realm_?id\b",
            &QueryFilter {
                use_regex: true,
                ..ic.clone()
            },
        );
        assert_eq!(got.effective, r"(?i)realm_?id\b");

        // Brackets: the bracket rewrite (substring + warning) happens first.
        let got = prepare_literal_pattern("unwrap()", &ic);
        assert_eq!(got.effective, r"(?i)unwrap\(\)");
        assert!(got.warning.is_some());

        // Not a line search: untouched.
        for f in [
            QueryFilter {
                symbols_mode: true,
                ..ic.clone()
            },
            QueryFilter {
                kind: Some(SymbolKind::Function),
                ..ic.clone()
            },
            QueryFilter {
                use_ast: true,
                ..ic.clone()
            },
        ] {
            let got = prepare_literal_pattern("RealmId", &f);
            assert_eq!(got.effective, "RealmId");
            assert!(!got.use_regex);
        }
    }

    #[test]
    fn non_literal_modes_are_never_rewritten() {
        let cases: Vec<(&str, QueryFilter)> = vec![
            (
                "contains",
                QueryFilter {
                    use_contains: true,
                    ..plain()
                },
            ),
            (
                "regex",
                QueryFilter {
                    use_regex: true,
                    ..plain()
                },
            ),
            (
                "symbols",
                QueryFilter {
                    symbols_mode: true,
                    ..plain()
                },
            ),
            (
                "kind",
                QueryFilter {
                    kind: Some(SymbolKind::Function),
                    ..plain()
                },
            ),
            (
                "ast",
                QueryFilter {
                    use_ast: true,
                    ..plain()
                },
            ),
        ];
        for (name, f) in cases {
            let got = prepare_literal_pattern("unwrap()", &f);
            assert_eq!(
                got.effective, "unwrap()",
                "{name}: pattern must be verbatim"
            );
            assert_eq!(
                got.use_regex, f.use_regex,
                "{name}: regex flag passes through"
            );
            assert!(got.warning.is_none(), "{name}: no warning");
        }
    }

    #[test]
    fn hint_names_the_count_and_both_switches() {
        let h = substring_hint_text(89, "verify_csrf");
        assert!(h.contains("89 substring match"));
        assert!(h.contains("contains:true"));
        assert!(h.contains("--contains"));
        assert!(h.contains("verify_csrf"));
    }
}
