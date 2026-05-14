//! Tree-sitter parsers for extracting symbols from source code
//!
//! This module provides language-specific parsers that extract symbols
//! (functions, classes, structs, etc.) from source code using Tree-sitter.
//!
//! Each language has its own submodule with a `parse` function that takes
//! source code and returns a vector of symbols.

pub mod rust;
pub mod typescript;
pub mod tsconfig;
pub mod vue;
pub mod svelte;
pub mod php;
pub mod python;
pub mod go;
pub mod java;
pub mod c;
pub mod cpp;
pub mod csharp;
pub mod ruby;
pub mod kotlin;
// pub mod swift;  // Temporarily disabled - tree-sitter-swift 0.7.2 grammar node types diverged from this parser's queries
pub mod zig;

use anyhow::{anyhow, Result};
use crate::models::{Language, SearchResult};

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
            Language::Rust => &["fn", "struct", "enum", "trait", "impl", "mod", "const", "static", "type", "macro"],
            Language::PHP => &["class", "function", "trait", "interface", "enum"],
            Language::Python => &["class", "def", "async"],
            Language::TypeScript | Language::JavaScript => &["class", "function", "interface", "type", "enum", "const", "let", "var"],
            Language::Go => &["func", "struct", "interface", "type", "const", "var"],
            Language::Java => &["class", "interface", "enum", "@interface"],
            Language::C => &["struct", "enum", "union", "typedef"],
            Language::Cpp => &["class", "struct", "enum", "union", "typedef", "namespace", "template"],
            Language::CSharp => &["class", "struct", "interface", "enum", "delegate", "record", "namespace"],
            Language::Ruby => &["class", "module", "def"],
            Language::Kotlin => &["class", "fun", "interface", "object", "enum", "annotation"],
            Language::Zig => &["fn", "struct", "enum", "const", "var", "type"],
            Language::Swift => &["class", "struct", "enum", "protocol", "func", "var", "let"],
            Language::Vue | Language::Svelte => &["function", "const", "let", "var"],
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
            "fn", "function", "def", "func",
            // Classes and types
            "class", "struct", "enum", "interface", "trait", "type", "record",
            // Modules and namespaces
            "mod", "module", "namespace",
            // Variables and constants
            "const", "static", "let", "var",
            // Other constructs
            "impl", "async", "object", "annotation", "protocol",
            "union", "typedef", "delegate", "template",
            // Java annotations
            "@interface",
        ]
    }

    /// Parse a file and extract symbols based on its language
    pub fn parse(
        path: &str,
        source: &str,
        language: Language,
    ) -> Result<Vec<SearchResult>> {
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
                log::warn!("Swift support temporarily disabled (parser queries out of date with tree-sitter-swift 0.7.x grammar): {}", path);
                Ok(vec![])
            }
            Language::Zig => zig::parse(path, source),
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
