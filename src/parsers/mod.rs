//! Tree-sitter parsers for extracting symbols from source code
//!
//! This module provides language-specific parsers that extract symbols
//! (functions, classes, structs, etc.) from source code using Tree-sitter.
//!
//! Each language has its own submodule with a `parse` function that takes
//! source code and returns a vector of symbols.

pub mod c;
pub mod cpp;
pub mod csharp;
pub mod go;
pub mod java;
pub mod kotlin;
pub mod php;
pub mod preview;
pub mod python;
pub mod ruby;
pub mod rust;
pub mod svelte;
pub mod tsconfig;
pub mod typescript;
pub mod vue;
// pub mod swift;  // Temporarily disabled - tree-sitter-swift 0.7.2 grammar node types diverged from this parser's queries
pub mod zig;

use crate::models::{Language, SearchResult};
use anyhow::{Result, anyhow};

/// Parser factory that selects the appropriate parser based on language
pub struct ParserFactory;

/// Extracted import/dependency information (before file ID resolution)
#[derive(Debug, Clone)]
pub struct ImportInfo {
    /// Import path as written in source code
    pub imported_path: String,
    /// Type classification hint (internal/external/stdlib)
    pub import_type: crate::models::ImportType,
    /// Line number where import appears
    pub line_number: usize,
    /// Imported symbols (for selective imports like `from x import a, b`)
    pub imported_symbols: Option<Vec<String>>,
}

/// Extracted export/re-export information (for barrel export tracking)
#[derive(Debug, Clone)]
pub struct ExportInfo {
    /// Symbol being exported (None for wildcard `export * from`)
    pub exported_symbol: Option<String>,
    /// Source path where the symbol is re-exported from
    pub source_path: String,
    /// Line number where export appears
    pub line_number: usize,
}

/// Trait for extracting dependencies from source code
///
/// Each language parser can implement this trait to extract import/include
/// statements from source files.
pub trait DependencyExtractor {
    /// Extract all imports/dependencies from source code
    ///
    /// Returns a list of ImportInfo records (before file ID resolution).
    /// The indexer will resolve these to file IDs and store in the database.
    ///
    /// # Arguments
    ///
    /// * `source` - Source code content
    ///
    /// # Returns
    ///
    /// Vector of ImportInfo records, or an error if parsing fails
    fn extract_dependencies(source: &str) -> Result<Vec<ImportInfo>>;
}

impl ParserFactory {
    /// Get the tree-sitter grammar for a language
    ///
    /// This is the single source of truth for tree-sitter language grammars.
    /// Used by both symbol parsers and AST query matching.
    ///
    /// Returns an error for:
    /// - Vue/Svelte (use line-based parsing instead of tree-sitter)
    /// - Swift (parser queries are out of date with tree-sitter-swift 0.7.x grammar)
    /// - Unknown languages
    pub fn get_language_grammar(language: Language) -> Result<tree_sitter::Language> {
        match language {
            Language::Rust => Ok(tree_sitter_rust::LANGUAGE.into()),
            Language::Python => Ok(tree_sitter_python::LANGUAGE.into()),
            Language::TypeScript => Ok(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
            Language::JavaScript => Ok(tree_sitter_typescript::LANGUAGE_TSX.into()),
            Language::Go => Ok(tree_sitter_go::LANGUAGE.into()),
            Language::Java => Ok(tree_sitter_java::LANGUAGE.into()),
            Language::C => Ok(tree_sitter_c::LANGUAGE.into()),
            Language::Cpp => Ok(tree_sitter_cpp::LANGUAGE.into()),
            Language::CSharp => Ok(tree_sitter_c_sharp::LANGUAGE.into()),
            Language::PHP => Ok(tree_sitter_php::LANGUAGE_PHP.into()),
            Language::Ruby => Ok(tree_sitter_ruby::LANGUAGE.into()),
            Language::Kotlin => Ok(tree_sitter_kotlin_ng::LANGUAGE.into()),
            Language::Zig => Ok(tree_sitter_zig::LANGUAGE.into()),
            Language::Swift => Err(anyhow!(
                "Swift support temporarily disabled (parser queries out of date with tree-sitter-swift 0.7.x grammar)"
            )),
            Language::Vue => Err(anyhow!(
                "Vue uses line-based parsing, not tree-sitter (tree-sitter-vue incompatible with tree-sitter 0.24+)"
            )),
            Language::Svelte => Err(anyhow!(
                "Svelte uses line-based parsing, not tree-sitter (tree-sitter-svelte incompatible with tree-sitter 0.24+)"
            )),
            Language::Text | Language::Lock | Language::Generated => Err(anyhow!(
                "The text tier (docs, config, templates, lock and generated files) is \
                 trigram-indexed only and has no grammar. Use full-text or regex \
                 search on these files, not --symbols or --ast."
            )),
            Language::Unknown => Err(anyhow!("Unknown language")),
        }
    }

    /// Get language keywords that should trigger "list all symbols" behavior
    ///
    /// When a user searches for a keyword (like "class", "function") with --symbols,
    /// we interpret it as "list all symbols of that type" rather than looking for
    /// a symbol literally named "class" or "function".
    ///
    /// Returns an empty slice for languages without common keywords or unsupported languages.
    pub fn get_keywords(language: Language) -> &'static [&'static str] {
        match language {
            Language::Rust => &[
                "fn", "struct", "enum", "trait", "impl", "mod", "const", "static", "type", "macro",
            ],
            Language::PHP => &["class", "function", "trait", "interface", "enum"],
            Language::Python => &["class", "def", "async"],
            Language::TypeScript | Language::JavaScript => &[
                "class",
                "function",
                "interface",
                "type",
                "enum",
                "const",
                "let",
                "var",
            ],
            Language::Go => &["func", "struct", "interface", "type", "const", "var"],
            Language::Java => &["class", "interface", "enum", "@interface"],
            Language::C => &["struct", "enum", "union", "typedef"],
            Language::Cpp => &[
                "class",
                "struct",
                "enum",
                "union",
                "typedef",
                "namespace",
                "template",
            ],
            Language::CSharp => &[
                "class",
                "struct",
                "interface",
                "enum",
                "delegate",
                "record",
                "namespace",
            ],
            Language::Ruby => &["class", "module", "def"],
            Language::Kotlin => &["class", "fun", "interface", "object", "enum", "annotation"],
            Language::Zig => &["fn", "struct", "enum", "const", "var", "type"],
            Language::Swift => &["class", "struct", "enum", "protocol", "func", "var", "let"],
            Language::Vue | Language::Svelte => &["function", "const", "let", "var"],
            // No symbols, so no keyword shortcuts.
            Language::Text | Language::Lock | Language::Generated => &[],
            Language::Unknown => &[],
        }
    }

    /// Get all keywords across all supported languages
    ///
    /// Returns a deduplicated union of keywords from all languages.
    /// Used for keyword detection when --lang is not specified.
    ///
    /// When a user searches for a keyword with --symbols or --kind,
    /// we enable keyword mode regardless of language filter.
    pub fn get_all_keywords() -> &'static [&'static str] {
        &[
            // Functions
            "fn",
            "function",
            "def",
            "func",
            // Classes and types
            "class",
            "struct",
            "enum",
            "interface",
            "trait",
            "type",
            "record",
            // Modules and namespaces
            "mod",
            "module",
            "namespace",
            // Variables and constants
            "const",
            "static",
            "let",
            "var",
            // Other constructs
            "impl",
            "async",
            "object",
            "annotation",
            "protocol",
            "union",
            "typedef",
            "delegate",
            "template",
            // Java annotations
            "@interface",
        ]
    }

    /// Parse a file and extract symbols based on its language
    ///
    /// Minified files are skipped — see [`is_minified`]. They stay fully
    /// text-searchable via trigrams; only SYMBOL extraction is declined.
    pub fn parse(path: &str, source: &str, language: Language) -> Result<Vec<SearchResult>> {
        if is_minified(source) {
            // Not a correctness guard — `preview::PREVIEW_MAX_BYTES` already makes the
            // memory safe. This is a cost guard: parsing a 1.45 MB single line spends
            // minutes to produce ~13,843 one-letter symbol names nobody will search
            // for. Logged at info, and naming the fallback, so a user who wonders why
            // `--symbols` is empty for this file is not left guessing.
            log::info!(
                "Skipping symbol extraction for {} — looks minified ({} bytes over {} lines). \
                 The file is still searchable with full-text and regex queries.",
                path,
                source.len(),
                count_lines(source)
            );
            return Ok(Vec::new());
        }

        match language {
            Language::Rust => rust::parse(path, source),
            Language::TypeScript => typescript::parse(path, source, language),
            Language::JavaScript => typescript::parse(path, source, language),
            Language::Vue => vue::parse(path, source),
            Language::Svelte => svelte::parse(path, source),
            Language::Python => python::parse(path, source),
            Language::Go => go::parse(path, source),
            Language::Java => java::parse(path, source),
            Language::PHP => php::parse(path, source),
            Language::C => c::parse(path, source),
            Language::Cpp => cpp::parse(path, source),
            Language::CSharp => csharp::parse(path, source),
            Language::Ruby => ruby::parse(path, source),
            Language::Kotlin => kotlin::parse(path, source),
            Language::Swift => {
                log::warn!(
                    "Swift support temporarily disabled (parser queries out of date with tree-sitter-swift 0.7.x grammar): {}",
                    path
                );
                Ok(vec![])
            }
            Language::Zig => zig::parse(path, source),
            // debug!, not warn!: the text tier is indexed deliberately, and a repo
            // with thousands of markdown files would otherwise flood the log.
            Language::Text | Language::Lock | Language::Generated => {
                log::debug!("No symbol extraction for text-tier file: {}", path);
                Ok(vec![])
            }
            Language::Unknown => {
                log::warn!("Unknown language for file: {}", path);
                Ok(vec![])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_factory() {
        // Simple test to ensure module compiles
        let _factory = ParserFactory;
    }
}

/// Average bytes per line above which a file is treated as minified.
///
/// Measured on a real 500k-LoC monorepo:
///
/// | category                                   | avg bytes/line   |
/// |--------------------------------------------|------------------|
/// | real source, including generated protobuf   | 32 – 42          |
/// | minified bundles                            | 2,869 – 726,376  |
///
/// 1024 sits 24x above the worst real file and 2.8x below the least-extreme bundle,
/// and still catches all four offenders. Deliberately NOT 512 (the preview cap): CJK
/// source at 3 bytes/char with long comment lines can reach ~600 bytes/line, and
/// silently skipping it would be a worse bug than the one this fixes.
pub const MAX_BYTES_PER_LINE: usize = 1024;

/// Files below this size are never treated as minified.
///
/// A short hand-written file with one long line (a wide data literal, a long URL in a
/// comment) must never lose its symbols. The preview cap already bounds its cost.
pub const MINIFIED_SIZE_FLOOR: usize = 16 * 1024;

/// Count newlines. Plain byte scan — no UTF-8 bookkeeping needed to find `\n`.
fn count_lines(source: &str) -> usize {
    memchr::memchr_iter(b'\n', source.as_bytes()).count()
}

/// Whether a file looks machine-generated and minified.
///
/// Average bytes per line, not MAX line length: a legitimate file may carry one very
/// long line (a base64 descriptor in generated protobuf, say) while being ordinary
/// code everywhere else, and a max-line rule would throw away all its real symbols.
/// An average is robust to a single outlier.
pub fn is_minified(source: &str) -> bool {
    if source.len() < MINIFIED_SIZE_FLOOR {
        return false;
    }
    // Early exit: as soon as enough newlines are seen the answer is "no", so an
    // ordinary file never pays for a full scan.
    let enough = source.len() / MAX_BYTES_PER_LINE;
    let mut seen = 0usize;
    for _ in memchr::memchr_iter(b'\n', source.as_bytes()) {
        seen += 1;
        if seen > enough {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod minified_tests {
    use super::*;

    #[test]
    fn real_source_is_never_minified() {
        // 42 bytes/line, the worst real file measured.
        let src = "pub fn some_function_name(a: u32) -> u32 {\n".repeat(2000);
        assert!(!is_minified(&src));
    }

    #[test]
    fn a_one_line_bundle_is_minified() {
        assert!(is_minified(&"function f(a,b){return a+b}".repeat(60_000)));
    }

    #[test]
    fn a_small_file_is_never_minified_however_long_its_line() {
        // Below the floor: a wide data literal must keep its symbols.
        assert!(!is_minified(&"x".repeat(MINIFIED_SIZE_FLOOR - 1)));
    }

    #[test]
    fn one_long_line_among_normal_ones_is_not_minified() {
        // The `rbac_pb.ts` shape: a 12 KB base64 descriptor in 2000 ordinary lines.
        let src = format!("{}\n{}", "z".repeat(12_282), "const a = 1;\n".repeat(2000));
        assert!(
            !is_minified(&src),
            "generated protobuf must keep its symbols"
        );
    }

    #[test]
    fn cjk_source_is_not_minified() {
        // 3 bytes/char; ~600 bytes/line would trip a 512 threshold.
        let line = format!("// {}\n", "日".repeat(200));
        assert!(!is_minified(&line.repeat(200)));
    }

    #[test]
    fn a_minified_file_yields_no_symbols_but_does_not_error() {
        let src = "function f(a,b){return a+b}".repeat(60_000);
        let out = ParserFactory::parse("bundle.js", &src, Language::JavaScript).unwrap();
        assert!(out.is_empty(), "got {} symbols", out.len());
    }

    #[test]
    fn a_normal_file_still_yields_symbols() {
        let src = "export function realOne(a: number) {\n  return a;\n}\n".repeat(50);
        let out = ParserFactory::parse("real.ts", &src, Language::TypeScript).unwrap();
        assert!(!out.is_empty(), "normal source must still parse");
    }
}

/// A tree-sitter [`Query`](tree_sitter::Query) compiled once per process.
///
/// Dependency extraction runs on every file of every index pass, and until 1.8.1
/// each call recompiled its (constant) query. `Query` is `Send + Sync`, so one
/// compiled copy in a `static` cell serves every thread of the indexing pool. A
/// compile failure is stored too and reported on every call, exactly as the
/// per-call `Query::new` did.
///
/// ```ignore
/// static QUERY: CachedQuery = CachedQuery::new();
/// let query = cached_query(&QUERY, tree_sitter_c::LANGUAGE, QUERY_SRC)
///     .context("Failed to create C include query")?;
/// ```
pub type CachedQuery = std::sync::OnceLock<std::result::Result<tree_sitter::Query, String>>;

/// Compile `source` for `language` on the first call; return the cached query after.
pub fn cached_query<'a>(
    cell: &'a CachedQuery,
    language: impl Into<tree_sitter::Language>,
    source: &str,
) -> Result<&'a tree_sitter::Query> {
    match cell.get_or_init(|| {
        tree_sitter::Query::new(&language.into(), source).map_err(|e| e.to_string())
    }) {
        Ok(query) => Ok(query),
        Err(e) => Err(anyhow!("{e}")),
    }
}
