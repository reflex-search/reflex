//! AST pattern matching using Tree-sitter queries
//!
//! This module provides structure-aware code search by executing Tree-sitter
//! query patterns (S-expressions) against candidate files.
//!
//! ## Architecture
//!
//! AST queries are executed as Phase 2 enrichment in the query pipeline:
//! 1. Phase 1: Trigram/regex search narrows to 10-100 candidate files
//! 2. Phase 2: Parse candidates + filter by AST pattern (THIS MODULE)
//! 3. Phase 3: Apply remaining filters (language, kind, etc.)
//!
//! ## Performance
//!
//! - Requires trigram pre-filtering (no AST-only queries allowed)
//! - Parses only candidate files (10-100 files, not entire codebase)
//! - Expected query time: 50-200ms depending on pattern complexity
//!
//! ## Example Usage
//!
//! ```rust
//! use reflex::ast_query::execute_ast_query;
//! use reflex::{Language, SearchResult, Span, SymbolKind};
//! use std::collections::HashMap;
//!
//! # fn example() -> anyhow::Result<()> {
//! // Prepare candidates from trigram search
//! let candidates = vec![SearchResult {
//!     path: "test.rs".to_string(),
//!     lang: Language::Rust,
//!     span: Span { start_line: 1, end_line: 1 },
//!     symbol: None,
//!     kind: SymbolKind::Unknown("text_match".to_string()),
//!     preview: String::new(),
//!     dependencies: None,
//! }];
//!
//! // File contents map
//! let mut file_contents = HashMap::new();
//! file_contents.insert("test.rs".to_string(), "async fn fetch() {}".to_string());
//!
//! // Find all async functions using AST query
//! let pattern = "(function_item (async)) @fn";
//! let results = execute_ast_query(candidates, pattern, Language::Rust, &file_contents)?;
//! # Ok(())
//! # }
//! ```

use crate::models::{Language, SearchResult, Span, SymbolKind};
use anyhow::{Context, Result, anyhow};
use std::collections::HashMap;
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};

/// Execute an AST query pattern against candidate files
///
/// Takes a list of candidate files (from Phase 1 trigram/regex search),
/// parses them with Tree-sitter, and filters by the AST pattern.
///
/// # Arguments
///
/// * `candidates` - Files to search (pre-filtered by trigrams)
/// * `ast_pattern` - Tree-sitter query S-expression
/// * `language` - Language for Tree-sitter grammar selection
/// * `file_contents` - Map of file paths to their contents
///
/// # Returns
///
/// Filtered list of search results matching the AST pattern
///
/// # Errors
///
/// Returns error if:
/// - Language not supported for AST queries
/// - AST pattern syntax is invalid
/// - Tree-sitter parsing fails
pub fn execute_ast_query(
    candidates: Vec<SearchResult>,
    ast_pattern: &str,
    language: Language,
    file_contents: &HashMap<String, String>,
) -> Result<Vec<SearchResult>> {
    // Get Tree-sitter grammar for the language
    let mut parser = Parser::new();
    let ts_language = get_tree_sitter_language(language)?;
    parser
        .set_language(&ts_language)
        .context("Failed to set Tree-sitter language")?;

    // Compile the AST query pattern
    let query = Query::new(&ts_language, ast_pattern)
        .map_err(|e| anyhow!("Invalid AST query pattern: {}", e))?;

    // Group candidates by file for efficient parsing
    let mut files_to_parse: HashMap<String, Vec<SearchResult>> = HashMap::new();
    for candidate in candidates {
        files_to_parse
            .entry(candidate.path.clone())
            .or_default()
            .push(candidate);
    }

    let mut matched_results = Vec::new();

    // Parse each file and execute query
    for (file_path, _candidates_in_file) in files_to_parse {
        // Get file content
        let content = match file_contents.get(&file_path) {
            Some(c) => c,
            None => {
                log::warn!(
                    "File content not found for {}: available keys are {:?}",
                    file_path,
                    file_contents.keys().collect::<Vec<_>>()
                );
                continue;
            }
        };

        // Parse file with Tree-sitter
        let tree = match parser.parse(content, None) {
            Some(t) => t,
            None => {
                log::warn!("Failed to parse file: {}", file_path);
                continue;
            }
        };

        // Execute query on the AST
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&query, tree.root_node(), content.as_bytes());

        // Process each match from the cursor
        while let Some(m) = matches.next() {
            // Skip matches without captures - captures are required to extract nodes
            if m.captures.is_empty() {
                log::warn!(
                    "Query pattern '{}' matched but has no captures - use '(node) @name' syntax",
                    ast_pattern
                );
                continue;
            }

            for capture in m.captures {
                let node = capture.node;
                let start_byte = node.start_byte();
                let end_byte = node.end_byte();
                let start_pos = node.start_position();
                let end_pos = node.end_position();

                // Extract matched text
                let matched_text = &content[start_byte..end_byte];

                // Try to determine symbol name and kind
                let (symbol_name, symbol_kind) = extract_symbol_info(&node, content);

                // Detect language from file extension
                let ext = std::path::Path::new(&file_path)
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                let detected_lang = Language::from_extension(ext);

                matched_results.push(SearchResult {
                    path: file_path.clone(),
                    lang: detected_lang,
                    span: Span {
                        start_line: start_pos.row + 1, // Tree-sitter uses 0-indexed lines
                        end_line: end_pos.row + 1,
                    },
                    symbol: symbol_name,
                    kind: symbol_kind
                        .unwrap_or_else(|| SymbolKind::Unknown("ast_match".to_string())),
                    preview: matched_text.to_string(),
                    dependencies: None,
                });
            }
        }
    }

    Ok(matched_results)
}

/// Get Tree-sitter language grammar for a given language
///
/// Delegates to ParserFactory::get_language_grammar() for centralized grammar loading.
/// All languages with tree-sitter grammars are supported automatically.
fn get_tree_sitter_language(lang: Language) -> Result<tree_sitter::Language> {
    crate::parsers::ParserFactory::get_language_grammar(lang)
        .with_context(|| format!("Language {:?} not supported for AST queries", lang))
}

/// Extract symbol name and kind from a Tree-sitter node
///
/// This is a best-effort function that tries to determine what kind of
/// symbol a matched node represents (function, struct, etc.) and its name.
fn extract_symbol_info(
    node: &tree_sitter::Node,
    source: &str,
) -> (Option<String>, Option<SymbolKind>) {
    // Try to identify the node kind and extract symbol name
    let kind = node.kind();

    match kind {
        // Rust-specific nodes
        "function_item" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = source[name_node.start_byte()..name_node.end_byte()].to_string();
                return (Some(name), Some(SymbolKind::Function));
            }
        }
        "struct_item" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = source[name_node.start_byte()..name_node.end_byte()].to_string();
                return (Some(name), Some(SymbolKind::Struct));
            }
        }
        "enum_item" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = source[name_node.start_byte()..name_node.end_byte()].to_string();
                return (Some(name), Some(SymbolKind::Enum));
            }
        }
        "trait_item" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = source[name_node.start_byte()..name_node.end_byte()].to_string();
                return (Some(name), Some(SymbolKind::Trait));
            }
        }
        "impl_item" => {
            if let Some(type_node) = node.child_by_field_name("type") {
                let name = source[type_node.start_byte()..type_node.end_byte()].to_string();
                return (Some(name), Some(SymbolKind::Unknown("impl".to_string())));
            }
        }

        // TypeScript/JavaScript-specific nodes
        "function_declaration" | "function" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = source[name_node.start_byte()..name_node.end_byte()].to_string();
                return (Some(name), Some(SymbolKind::Function));
            }
        }
        "class_declaration" | "class" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = source[name_node.start_byte()..name_node.end_byte()].to_string();
                return (Some(name), Some(SymbolKind::Class));
            }
        }
        "interface_declaration" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = source[name_node.start_byte()..name_node.end_byte()].to_string();
                return (Some(name), Some(SymbolKind::Interface));
            }
        }

        // Python-specific nodes
        "class_definition" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = source[name_node.start_byte()..name_node.end_byte()].to_string();
                return (Some(name), Some(SymbolKind::Class));
            }
        }

        // PHP-specific nodes (function_definition shared with Python above)
        "function_definition" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = source[name_node.start_byte()..name_node.end_byte()].to_string();
                return (Some(name), Some(SymbolKind::Function));
            }
        }
        // Note: class_declaration handled above for TS/JS, class_definition for Python
        // PHP trait_declaration uses same node type as Rust trait_item handled above
        "enum_declaration" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = source[name_node.start_byte()..name_node.end_byte()].to_string();
                return (Some(name), Some(SymbolKind::Enum));
            }
        }

        _ => {}
    }

    // If we couldn't extract specific info, return None
    (None, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Span;

    #[test]
    fn test_rust_function_query() {
        let content = r#"
fn main() {
    println!("Hello");
}

async fn fetch_data() {
    // async function
}

fn sync_helper() {
    // sync function
}
"#;

        let mut file_contents = HashMap::new();
        file_contents.insert("test.rs".to_string(), content.to_string());

        // Create dummy candidates (trigram pre-filtering would have found these)
        let candidates = vec![SearchResult {
            path: "test.rs".to_string(),
            lang: Language::Rust,
            span: Span {
                start_line: 1,
                end_line: 1,
            },
            symbol: None,
            kind: SymbolKind::Unknown("text_match".to_string()),
            preview: String::new(),
            dependencies: None,
        }];

        // Query for all functions - using capture syntax @fn
        let ast_pattern = "(function_item) @fn";
        let results = execute_ast_query(candidates, ast_pattern, Language::Rust, &file_contents)
            .expect("AST query failed");

        // Should find all three functions
        assert_eq!(results.len(), 3);
        assert!(results.iter().any(|r| r.symbol.as_deref() == Some("main")));
        assert!(
            results
                .iter()
                .any(|r| r.symbol.as_deref() == Some("fetch_data"))
        );
        assert!(
            results
                .iter()
                .any(|r| r.symbol.as_deref() == Some("sync_helper"))
        );
    }

    #[test]
    fn test_rust_struct_query() {
        let content = r#"
struct User {
    name: String,
}

struct Config {
    debug: bool,
}
"#;

        let mut file_contents = HashMap::new();
        file_contents.insert("test.rs".to_string(), content.to_string());

        let candidates = vec![SearchResult {
            path: "test.rs".to_string(),
            lang: Language::Rust,
            span: Span {
                start_line: 1,
                end_line: 1,
            },
            symbol: None,
            kind: SymbolKind::Unknown("text_match".to_string()),
            preview: String::new(),
            dependencies: None,
        }];

        // Query for all structs - using capture syntax @struct
        let ast_pattern = "(struct_item) @struct";
        let results = execute_ast_query(candidates, ast_pattern, Language::Rust, &file_contents)
            .expect("AST query failed");

        // Should find both structs
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|r| r.symbol == Some("User".to_string())));
        assert!(
            results
                .iter()
                .any(|r| r.symbol == Some("Config".to_string()))
        );
    }

    #[test]
    fn test_invalid_ast_pattern() {
        let mut file_contents = HashMap::new();
        file_contents.insert("test.rs".to_string(), "fn test() {}".to_string());

        let candidates = vec![SearchResult {
            path: "test.rs".to_string(),
            lang: Language::Rust,
            span: Span {
                start_line: 1,
                end_line: 1,
            },
            symbol: None,
            kind: SymbolKind::Unknown("text_match".to_string()),
            preview: String::new(),
            dependencies: None,
        }];

        // Invalid S-expression syntax (missing closing paren)
        let ast_pattern = "(function_item @fn";
        let result = execute_ast_query(candidates, ast_pattern, Language::Rust, &file_contents);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid AST query pattern")
        );
    }

    #[test]
    fn test_unsupported_language() {
        let mut file_contents = HashMap::new();
        file_contents.insert(
            "test.vue".to_string(),
            "<script>export default {}</script>".to_string(),
        );

        let candidates = vec![SearchResult {
            path: "test.vue".to_string(),
            lang: Language::Vue,
            span: Span {
                start_line: 1,
                end_line: 1,
            },
            symbol: None,
            kind: SymbolKind::Unknown("text_match".to_string()),
            preview: String::new(),
            dependencies: None,
        }];

        // Vue uses line-based parsing, not tree-sitter, so AST queries should fail
        let ast_pattern = "(function_declaration) @fn";
        let result = execute_ast_query(candidates, ast_pattern, Language::Vue, &file_contents);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not supported for AST queries")
        );
    }

    #[test]
    fn test_python_function_query() {
        let content = r#"
def hello():
    print("Hello")

async def fetch_data():
    return await get_data()

def process(x):
    return x * 2
"#;

        let mut file_contents = HashMap::new();
        file_contents.insert("test.py".to_string(), content.to_string());

        let candidates = vec![SearchResult {
            path: "test.py".to_string(),
            lang: Language::Python,
            span: Span {
                start_line: 1,
                end_line: 1,
            },
            symbol: None,
            kind: SymbolKind::Unknown("text_match".to_string()),
            preview: String::new(),
            dependencies: None,
        }];

        // Query for all Python functions
        let ast_pattern = "(function_definition) @fn";
        let results = execute_ast_query(candidates, ast_pattern, Language::Python, &file_contents)
            .expect("AST query failed");

        // Should find all three functions
        assert_eq!(results.len(), 3);
        assert!(results.iter().any(|r| r.preview.contains("def hello")));
        assert!(
            results
                .iter()
                .any(|r| r.preview.contains("async def fetch_data"))
        );
        assert!(results.iter().any(|r| r.preview.contains("def process")));
    }
}
