//! Rust language parser using Tree-sitter
//!
//! Extracts symbols from Rust source code:
//! - Functions (fn)
//! - Structs
//! - Enums
//! - Traits
//! - Impl blocks
//! - Constants
//! - Static variables
//! - Local variables (let bindings)
//! - Modules
//! - Type aliases
//! - Macros (macro_rules! definitions)

use anyhow::{Context, Result};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};
use crate::models::{Language, SearchResult, Span, SymbolKind};
use crate::parsers::{DependencyExtractor, ImportInfo};

/// Parse Rust source code and extract symbols
pub fn parse(path: &str, source: &str) -> Result<Vec<SearchResult>> {
    let mut parser = Parser::new();
    let language = tree_sitter_rust::LANGUAGE;

    parser
        .set_language(&language.into())
        .context("Failed to set Rust language")?;

    let tree = parser
        .parse(source, None)
        .context("Failed to parse Rust source")?;

    let root_node = tree.root_node();

    let mut symbols = Vec::new();

    // Extract different types of symbols using Tree-sitter queries
    symbols.extend(extract_functions(source, &root_node)?);
    symbols.extend(extract_structs(source, &root_node)?);
    symbols.extend(extract_enums(source, &root_node)?);
    symbols.extend(extract_traits(source, &root_node)?);
    symbols.extend(extract_impls(source, &root_node)?);
    symbols.extend(extract_constants(source, &root_node)?);
    symbols.extend(extract_statics(source, &root_node)?);
    symbols.extend(extract_local_variables(source, &root_node)?);
    symbols.extend(extract_modules(source, &root_node)?);
    symbols.extend(extract_type_aliases(source, &root_node)?);
    symbols.extend(extract_macros(source, &root_node)?);
    symbols.extend(extract_attributes(source, &root_node)?);


    // Add file path to all symbols
    for symbol in &mut symbols {
        symbol.path = path.to_string();
        symbol.lang = Language::Rust;
    }

    Ok(symbols)
}

/// Extract function definitions
fn extract_functions(source: &str, root: &tree_sitter::Node) -> Result<Vec<SearchResult>> {
    let language = tree_sitter_rust::LANGUAGE;
    let query_str = r#"
        (function_item
            name: (identifier) @name) @function
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create function query")?;

    extract_symbols(source, root, &query, SymbolKind::Function, None)
}

/// Extract struct definitions
fn extract_structs(source: &str, root: &tree_sitter::Node) -> Result<Vec<SearchResult>> {
    let language = tree_sitter_rust::LANGUAGE;
    let query_str = r#"
        (struct_item
            name: (type_identifier) @name) @struct
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create struct query")?;

    extract_symbols(source, root, &query, SymbolKind::Struct, None)
}

/// Extract enum definitions
fn extract_enums(source: &str, root: &tree_sitter::Node) -> Result<Vec<SearchResult>> {
    let language = tree_sitter_rust::LANGUAGE;
    let query_str = r#"
        (enum_item
            name: (type_identifier) @name) @enum
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create enum query")?;

    extract_symbols(source, root, &query, SymbolKind::Enum, None)
}

/// Extract trait definitions
fn extract_traits(source: &str, root: &tree_sitter::Node) -> Result<Vec<SearchResult>> {
    let language = tree_sitter_rust::LANGUAGE;
    let query_str = r#"
        (trait_item
            name: (type_identifier) @name) @trait
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create trait query")?;

    extract_symbols(source, root, &query, SymbolKind::Trait, None)
}

/// Extract impl blocks
fn extract_impls(source: &str, root: &tree_sitter::Node) -> Result<Vec<SearchResult>> {
    let language = tree_sitter_rust::LANGUAGE;

    // Extract methods from impl blocks
    let query_str = r#"
        (impl_item
            type: (type_identifier) @impl_name
            body: (declaration_list
                (function_item
                    name: (identifier) @method_name))) @impl
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create impl query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut impl_name = None;
        let mut method_name = None;
        let mut method_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            match capture_name {
                "impl_name" => {
                    impl_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                }
                "method_name" => {
                    method_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                    // Find the parent function_item node
                    let mut current = capture.node;
                    while let Some(parent) = current.parent() {
                        if parent.kind() == "function_item" {
                            method_node = Some(parent);
                            break;
                        }
                        current = parent;
                    }
                }
                _ => {}
            }
        }

        if let (Some(impl_name), Some(method_name), Some(node)) = (impl_name, method_name, method_node) {
            let scope = format!("impl {}", impl_name);
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(), // Path will be filled in later
                Language::Rust,
                SymbolKind::Method,
                Some(method_name),
                span,
                Some(scope),
                preview,
            ));
        }
    }

    Ok(symbols)
}

/// Extract constants
fn extract_constants(source: &str, root: &tree_sitter::Node) -> Result<Vec<SearchResult>> {
    let language = tree_sitter_rust::LANGUAGE;
    let query_str = r#"
        (const_item
            name: (identifier) @name) @const
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create const query")?;

    extract_symbols(source, root, &query, SymbolKind::Constant, None)
}

/// Extract static variables
fn extract_statics(source: &str, root: &tree_sitter::Node) -> Result<Vec<SearchResult>> {
    let language = tree_sitter_rust::LANGUAGE;
    let query_str = r#"
        (static_item
            name: (identifier) @name) @static
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create static query")?;

    extract_symbols(source, root, &query, SymbolKind::Variable, None)
}

/// Extract local variable bindings (let statements)
fn extract_local_variables(source: &str, root: &tree_sitter::Node) -> Result<Vec<SearchResult>> {
    let language = tree_sitter_rust::LANGUAGE;
    let query_str = r#"
        (let_declaration
            pattern: (identifier) @name) @let
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create let declaration query")?;

    extract_symbols(source, root, &query, SymbolKind::Variable, None)
}

/// Extract module declarations
fn extract_modules(source: &str, root: &tree_sitter::Node) -> Result<Vec<SearchResult>> {
    let language = tree_sitter_rust::LANGUAGE;
    let query_str = r#"
        (mod_item
            name: (identifier) @name) @module
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create module query")?;

    extract_symbols(source, root, &query, SymbolKind::Module, None)
}

/// Extract type aliases
fn extract_type_aliases(source: &str, root: &tree_sitter::Node) -> Result<Vec<SearchResult>> {
    let language = tree_sitter_rust::LANGUAGE;
    let query_str = r#"
        (type_item
            name: (type_identifier) @name) @type
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create type query")?;

    extract_symbols(source, root, &query, SymbolKind::Type, None)
}

/// Extract macro definitions (macro_rules!)
fn extract_macros(source: &str, root: &tree_sitter::Node) -> Result<Vec<SearchResult>> {
    let language = tree_sitter_rust::LANGUAGE;
    let query_str = r#"
        (macro_definition
            name: (identifier) @name) @macro
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create macro query")?;

    extract_symbols(source, root, &query, SymbolKind::Macro, None)
}

/// Extract attributes: BOTH definitions and uses
/// Definitions: #[proc_macro_attribute] pub fn route(...)
/// Uses: #[test] fn my_test(), #[derive(Debug)] struct Foo
fn extract_attributes(source: &str, root: &tree_sitter::Node) -> Result<Vec<SearchResult>> {
    let language = tree_sitter_rust::LANGUAGE;
    let mut symbols = Vec::new();

    // Part 1: Extract attribute DEFINITIONS (proc macro attributes)
    let func_query_str = r#"
        (function_item
            name: (identifier) @name) @function
    "#;

    let func_query = Query::new(&language.into(), func_query_str)
        .context("Failed to create function query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&func_query, *root, source.as_bytes());

    while let Some(match_) = matches.next() {
        let mut name = None;
        let mut func_node = None;

        for capture in match_.captures {
            let capture_name: &str = &func_query.capture_names()[capture.index as usize];
            match capture_name {
                "name" => {
                    name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                }
                "function" => {
                    func_node = Some(capture.node);
                }
                _ => {}
            }
        }

        // Check if this function has #[proc_macro_attribute] attribute
        if let (Some(name), Some(func_node)) = (name, func_node) {
            let mut has_proc_macro_attr = false;

            if let Some(parent) = func_node.parent() {
                let mut func_index = None;
                for i in 0..parent.child_count() {
                    if let Some(child) = parent.child(i) {
                        if child.id() == func_node.id() {
                            func_index = Some(i);
                            break;
                        }
                    }
                }

                if let Some(func_idx) = func_index {
                    for i in (0..func_idx).rev() {
                        if let Some(child) = parent.child(i) {
                            if child.kind() == "attribute_item" {
                                let attr_text = child.utf8_text(source.as_bytes()).unwrap_or("");
                                if attr_text.contains("proc_macro_attribute") {
                                    has_proc_macro_attr = true;
                                }
                            } else if !child.kind().contains("comment") && child.kind() != "line_comment" {
                                break;
                            }
                        }
                    }
                }
            }

            if has_proc_macro_attr {
                let span = node_to_span(&func_node);
                let preview = extract_preview(source, &span);

                symbols.push(SearchResult::new(
                    String::new(),
                    Language::Rust,
                    SymbolKind::Attribute,
                    Some(name),
                    span,
                    None,
                    preview,
                ));
            }
        }
    }

    // Part 2: Extract attribute USES (#[test], #[derive(...)], etc.)
    let attr_query_str = r#"
        (attribute_item
            (attribute
                (identifier) @attr_name)) @attr
    "#;

    let attr_query = Query::new(&language.into(), attr_query_str)
        .context("Failed to create attribute use query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&attr_query, *root, source.as_bytes());

    while let Some(match_) = matches.next() {
        let mut attr_name = None;
        let mut attr_node = None;

        for capture in match_.captures {
            let capture_name: &str = &attr_query.capture_names()[capture.index as usize];
            match capture_name {
                "attr_name" => {
                    attr_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                }
                "attr" => {
                    attr_node = Some(capture.node);
                }
                _ => {}
            }
        }

        if let (Some(name), Some(node)) = (attr_name, attr_node) {
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::Rust,
                SymbolKind::Attribute,
                Some(name),
                span,
                None,
                preview,
            ));
        }
    }

    Ok(symbols)
}

/// Generic symbol extraction helper
fn extract_symbols(
    source: &str,
    root: &tree_sitter::Node,
    query: &Query,
    kind: SymbolKind,
    scope: Option<String>,
) -> Result<Vec<SearchResult>> {
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        // Find the name capture and the full node
        let mut name = None;
        let mut full_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            if capture_name == "name" {
                name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
            } else {
                // Assume any other capture is the full node
                full_node = Some(capture.node);
            }
        }

        if let (Some(name), Some(node)) = (name, full_node) {
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(), // Path will be filled in later
                Language::Rust,
                kind.clone(),
                Some(name),
                span,
                scope.clone(),
                preview,
            ));
        }
    }

    Ok(symbols)
}

/// Convert a Tree-sitter node to a Span
fn node_to_span(node: &tree_sitter::Node) -> Span {
    let start = node.start_position();
    let end = node.end_position();

    Span::new(
        start.row + 1,  // Convert 0-indexed to 1-indexed
        start.column,
        end.row + 1,
        end.column,
    )
}

/// Extract a preview (5-7 lines) around the symbol
fn extract_preview(source: &str, span: &Span) -> String {
    let lines: Vec<&str> = source.lines().collect();

    // Extract 7 lines: the start line and 6 following lines
    // This provides enough context for AI agents to understand the code
    let start_idx = (span.start_line - 1) as usize; // Convert back to 0-indexed
    let end_idx = (start_idx + 7).min(lines.len());

    lines[start_idx..end_idx].join("\n")
}

/// Rust dependency extractor implementation
pub struct RustDependencyExtractor;

impl DependencyExtractor for RustDependencyExtractor {
    fn extract_dependencies(source: &str) -> Result<Vec<ImportInfo>> {
        let mut parser = Parser::new();
        let language = tree_sitter_rust::LANGUAGE;

        parser
            .set_language(&language.into())
            .context("Failed to set Rust language")?;

        let tree = parser
            .parse(source, None)
            .context("Failed to parse Rust source")?;

        let root_node = tree.root_node();

        let mut imports = Vec::new();

        // Extract use declarations
        imports.extend(extract_use_declarations(source, &root_node)?);

        // Extract mod items (module declarations)
        imports.extend(extract_mod_items(source, &root_node)?);

        // Extract extern crate declarations
        imports.extend(extract_extern_crates(source, &root_node)?);

        Ok(imports)
    }
}

/// Extract use declarations (use std::collections::HashMap)
fn extract_use_declarations(source: &str, root: &tree_sitter::Node) -> Result<Vec<ImportInfo>> {
    let language = tree_sitter_rust::LANGUAGE;
    let query_str = r#"
        (use_declaration) @use
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create use declaration query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut imports = Vec::new();

    while let Some(match_) = matches.next() {
        for capture in match_.captures {
            let node = capture.node;
            let text = node.utf8_text(source.as_bytes()).unwrap_or("");
            let line_number = node.start_position().row + 1;

            // Parse the use declaration text
            let path_info = parse_rust_use_declaration(text);

            for (path, symbols) in path_info {
                let import_type = classify_rust_import(&path);

                imports.push(ImportInfo {
                    imported_path: path,
                    import_type,
                    line_number,
                    imported_symbols: symbols,
                });
            }
        }
    }

    Ok(imports)
}

/// Extract mod items (mod parser;)
fn extract_mod_items(source: &str, root: &tree_sitter::Node) -> Result<Vec<ImportInfo>> {
    let language = tree_sitter_rust::LANGUAGE;
    let query_str = r#"
        (mod_item
            name: (identifier) @name) @mod
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create mod item query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut imports = Vec::new();

    while let Some(match_) = matches.next() {
        let mut name = None;
        let mut mod_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            match capture_name {
                "name" => {
                    name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                }
                "mod" => {
                    mod_node = Some(capture.node);
                }
                _ => {}
            }
        }

        if let (Some(name), Some(node)) = (name, mod_node) {
            // Check if this is an external module declaration (no body)
            let has_body = node.child_by_field_name("body").is_some();

            if !has_body {
                // This is an external module reference (mod parser;)
                let line_number = node.start_position().row + 1;

                imports.push(ImportInfo {
                    imported_path: name,
                    import_type: crate::models::ImportType::Internal,
                    line_number,
                    imported_symbols: None,
                });
            }
        }
    }

    Ok(imports)
}

/// Extract extern crate declarations (extern crate serde;)
fn extract_extern_crates(source: &str, root: &tree_sitter::Node) -> Result<Vec<ImportInfo>> {
    let language = tree_sitter_rust::LANGUAGE;
    let query_str = r#"
        (extern_crate_declaration
            name: (identifier) @name) @extern
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create extern crate query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut imports = Vec::new();

    while let Some(match_) = matches.next() {
        let mut name = None;
        let mut extern_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            match capture_name {
                "name" => {
                    name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                }
                "extern" => {
                    extern_node = Some(capture.node);
                }
                _ => {}
            }
        }

        if let (Some(name), Some(node)) = (name, extern_node) {
            let line_number = node.start_position().row + 1;
            let import_type = classify_rust_import(&name);

            imports.push(ImportInfo {
                imported_path: name,
                import_type,
                line_number,
                imported_symbols: None,
            });
        }
    }

    Ok(imports)
}

/// Classify a Rust import path as Internal, External, or Stdlib
fn classify_rust_import(path: &str) -> crate::models::ImportType {
    use crate::models::ImportType;

    if path.starts_with("std::") || path.starts_with("core::") || path.starts_with("alloc::") {
        ImportType::Stdlib
    } else if path.starts_with("crate::") || path.starts_with("super::") || path.starts_with("self::") {
        ImportType::Internal
    } else {
        // External crate
        ImportType::External
    }
}

/// Parse a Rust use declaration and extract path(s) and symbols
///
/// Handles:
/// - Simple: use std::collections::HashMap;
/// - With symbols: use std::collections::{HashMap, HashSet};
/// - Nested: use std::{io, fs};
/// - With aliases: use std::io::Result as IoResult;
/// - Glob: use std::collections::*;
fn parse_rust_use_declaration(text: &str) -> Vec<(String, Option<Vec<String>>)> {
    // Remove visibility modifiers and keywords
    let text = text.trim()
        .strip_prefix("pub(crate)").unwrap_or(text)
        .trim()
        .strip_prefix("pub(super)").unwrap_or(text)
        .trim()
        .strip_prefix("pub").unwrap_or(text)
        .trim()
        .strip_prefix("use").unwrap_or(text)
        .trim()
        .strip_suffix(";").unwrap_or(text)
        .trim();

    // Handle different patterns
    if text.contains('{') {
        // Has braces - extract base path and symbols
        if let Some(idx) = text.find('{') {
            let base_path = text[..idx].trim_end_matches("::").to_string();

            if let Some(end) = text.find('}') {
                let symbols_str = &text[idx + 1..end];
                let symbols: Vec<String> = symbols_str
                    .split(',')
                    .map(|s| {
                        // Handle aliases like "HashMap as Map" - extract the imported name
                        let trimmed = s.trim();
                        if let Some(as_idx) = trimmed.find(" as ") {
                            trimmed[..as_idx].trim().to_string()
                        } else {
                            trimmed.to_string()
                        }
                    })
                    .filter(|s| !s.is_empty() && s != "*")
                    .collect();

                if !symbols.is_empty() {
                    return vec![(base_path, Some(symbols))];
                }
            }
        }
    }

    // Simple path (possibly with alias)
    let path = if let Some(as_idx) = text.find(" as ") {
        text[..as_idx].trim().to_string()
    } else {
        text.to_string()
    };

    vec![(path, None)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_function() {
        let source = r#"
            fn hello_world() {
                println!("Hello, world!");
            }
        "#;

        let symbols = parse("test.rs", source).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("hello_world"));
        assert!(matches!(symbols[0].kind, SymbolKind::Function));
    }

    #[test]
    fn test_parse_struct() {
        let source = r#"
            struct User {
                name: String,
                age: u32,
            }
        "#;

        let symbols = parse("test.rs", source).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("User"));
        assert!(matches!(symbols[0].kind, SymbolKind::Struct));
    }

    #[test]
    fn test_parse_impl() {
        let source = r#"
            struct User {
                name: String,
            }

            impl User {
                fn new(name: String) -> Self {
                    User { name }
                }

                fn get_name(&self) -> &str {
                    &self.name
                }
            }
        "#;

        let symbols = parse("test.rs", source).unwrap();

        // Should find: struct User, method new, method get_name
        assert!(symbols.len() >= 3);

        let method_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method))
            .collect();

        assert_eq!(method_symbols.len(), 2);
        assert!(method_symbols.iter().any(|s| s.symbol.as_deref() == Some("new")));
        assert!(method_symbols.iter().any(|s| s.symbol.as_deref() == Some("get_name")));

        // Note: scope field was removed from SearchResult for token optimization
        // Methods are identified by SymbolKind::Method
    }

    #[test]
    fn test_parse_enum() {
        let source = r#"
            enum Status {
                Active,
                Inactive,
            }
        "#;

        let symbols = parse("test.rs", source).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("Status"));
        assert!(matches!(symbols[0].kind, SymbolKind::Enum));
    }

    #[test]
    fn test_parse_trait() {
        let source = r#"
            trait Drawable {
                fn draw(&self);
            }
        "#;

        let symbols = parse("test.rs", source).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("Drawable"));
        assert!(matches!(symbols[0].kind, SymbolKind::Trait));
    }

    #[test]
    fn test_parse_multiple_symbols() {
        let source = r#"
            const MAX_SIZE: usize = 100;

            struct Config {
                size: usize,
            }

            fn create_config() -> Config {
                Config { size: MAX_SIZE }
            }
        "#;

        let symbols = parse("test.rs", source).unwrap();

        // Should find: const, struct, function
        assert_eq!(symbols.len(), 3);

        let kinds: Vec<&SymbolKind> = symbols.iter().map(|s| &s.kind).collect();
        assert!(kinds.contains(&&SymbolKind::Constant));
        assert!(kinds.contains(&&SymbolKind::Struct));
        assert!(kinds.contains(&&SymbolKind::Function));
    }

    #[test]
    fn test_local_variables_included() {
        let source = r#"
            fn calculate(input: i32) -> i32 {
                let local_var = input * 2;
                let result = local_var + 10;
                result
            }

            struct Calculator;

            impl Calculator {
                fn compute(&self, value: i32) -> i32 {
                    let temp = value * 3;
                    let mut final_value = temp + 5;
                    final_value += 1;
                    final_value
                }
            }
        "#;

        let symbols = parse("test.rs", source).unwrap();

        // Filter to just variables
        let variables: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        // Check that local variables are captured
        assert!(variables.iter().any(|v| v.symbol.as_deref() == Some("local_var")));
        assert!(variables.iter().any(|v| v.symbol.as_deref() == Some("result")));
        assert!(variables.iter().any(|v| v.symbol.as_deref() == Some("temp")));
        assert!(variables.iter().any(|v| v.symbol.as_deref() == Some("final_value")));

        // Note: scope field was removed from SearchResult for token optimization
    }

    #[test]
    fn test_static_variables() {
        let source = r#"
            static GLOBAL_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            static mut MUTABLE_GLOBAL: i32 = 0;

            const MAX_SIZE: usize = 100;

            fn increment() {
                GLOBAL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        "#;

        let symbols = parse("test.rs", source).unwrap();

        // Filter to statics and constants
        let statics: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        let constants: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Constant))
            .collect();

        // Check that static variables are captured
        assert!(statics.iter().any(|v| v.symbol.as_deref() == Some("GLOBAL_COUNTER")));
        assert!(statics.iter().any(|v| v.symbol.as_deref() == Some("MUTABLE_GLOBAL")));

        // Check that constants are still separate
        assert!(constants.iter().any(|c| c.symbol.as_deref() == Some("MAX_SIZE")));
    }

    #[test]
    fn test_macros() {
        let source = r#"
            macro_rules! say_hello {
                () => {
                    println!("Hello!");
                };
            }

            macro_rules! vec_of_strings {
                ($($x:expr),*) => {
                    vec![$($x.to_string()),*]
                };
            }

            fn main() {
                say_hello!();
            }
        "#;

        let symbols = parse("test.rs", source).unwrap();

        // Filter to macros
        let macros: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Macro))
            .collect();

        // Check that macros are captured
        assert!(macros.iter().any(|m| m.symbol.as_deref() == Some("say_hello")));
        assert!(macros.iter().any(|m| m.symbol.as_deref() == Some("vec_of_strings")));
        assert_eq!(macros.len(), 2);
    }

    #[test]
    fn test_attribute_proc_macros() {
        let source = r#"
            use proc_macro::TokenStream;

            #[proc_macro_attribute]
            pub fn test(attr: TokenStream, item: TokenStream) -> TokenStream {
                item
            }

            #[proc_macro_attribute]
            pub fn route(attr: TokenStream, item: TokenStream) -> TokenStream {
                item
            }

            // Regular function - should NOT be captured
            pub fn helper() {}
        "#;

        let symbols = parse("test.rs", source).unwrap();

        // Filter to attributes
        let attributes: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Attribute))
            .collect();

        // Check that attribute proc macro DEFINITIONS are captured
        assert!(attributes.iter().any(|a| a.symbol.as_deref() == Some("test")));
        assert!(attributes.iter().any(|a| a.symbol.as_deref() == Some("route")));

        // Verify helper function is NOT captured as attribute
        assert!(!attributes.iter().any(|a| a.symbol.as_deref() == Some("helper")));

        // Should find 2 proc macro definitions + 2 attribute uses (#[proc_macro_attribute])
        assert_eq!(attributes.len(), 4);
    }

    #[test]
    fn test_attribute_uses() {
        let source = r#"
            #[test]
            fn test_something() {
                assert_eq!(1, 1);
            }

            #[test]
            #[should_panic]
            fn test_panic() {
                panic!("expected");
            }

            #[derive(Debug, Clone)]
            struct MyStruct {
                field: i32
            }

            #[cfg(test)]
            mod tests {
                #[test]
                fn nested_test() {}
            }
        "#;

        let symbols = parse("test.rs", source).unwrap();

        // Filter to attributes
        let attributes: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Attribute))
            .collect();

        // Check that attribute USES are captured
        assert!(attributes.iter().any(|a| a.symbol.as_deref() == Some("test")));
        assert!(attributes.iter().any(|a| a.symbol.as_deref() == Some("should_panic")));
        assert!(attributes.iter().any(|a| a.symbol.as_deref() == Some("derive")));
        assert!(attributes.iter().any(|a| a.symbol.as_deref() == Some("cfg")));

        // Should find: test (3x), should_panic (1x), derive (1x), cfg (1x) = 6 total
        assert_eq!(attributes.len(), 6);
    }

    #[test]
    fn test_extract_dependencies_use_declarations() {
        let source = r#"
            use std::collections::HashMap;
            use crate::models::{Language, SearchResult};
            use super::utils;
            use anyhow::Result;
        "#;

        let deps = RustDependencyExtractor::extract_dependencies(source).unwrap();

        // Should find 4 imports
        assert_eq!(deps.len(), 4);

        // Check std import
        let std_import = deps.iter().find(|d| d.imported_path == "std::collections::HashMap").unwrap();
        assert!(matches!(std_import.import_type, crate::models::ImportType::Stdlib));

        // Check crate import with symbols
        let crate_import = deps.iter().find(|d| d.imported_path == "crate::models").unwrap();
        assert!(matches!(crate_import.import_type, crate::models::ImportType::Internal));
        assert!(crate_import.imported_symbols.is_some());
        let symbols = crate_import.imported_symbols.as_ref().unwrap();
        assert_eq!(symbols.len(), 2);
        assert!(symbols.contains(&"Language".to_string()));
        assert!(symbols.contains(&"SearchResult".to_string()));

        // Check super import
        let super_import = deps.iter().find(|d| d.imported_path == "super::utils").unwrap();
        assert!(matches!(super_import.import_type, crate::models::ImportType::Internal));

        // Check external import
        let external_import = deps.iter().find(|d| d.imported_path == "anyhow::Result").unwrap();
        assert!(matches!(external_import.import_type, crate::models::ImportType::External));
    }

    #[test]
    fn test_extract_dependencies_mod_declarations() {
        let source = r#"
            mod parser;
            mod utils;

            mod inline {
                fn test() {}
            }
        "#;

        let deps = RustDependencyExtractor::extract_dependencies(source).unwrap();

        // Should find 2 external mod declarations (not the inline one)
        assert_eq!(deps.len(), 2);
        assert!(deps.iter().any(|d| d.imported_path == "parser"));
        assert!(deps.iter().any(|d| d.imported_path == "utils"));
        assert!(deps.iter().all(|d| matches!(d.import_type, crate::models::ImportType::Internal)));
    }

    #[test]
    fn test_extract_dependencies_extern_crate() {
        let source = r#"
            extern crate serde;
            extern crate serde_json;
        "#;

        let deps = RustDependencyExtractor::extract_dependencies(source).unwrap();

        // Should find 2 extern crate declarations
        assert_eq!(deps.len(), 2);
        assert!(deps.iter().any(|d| d.imported_path == "serde"));
        assert!(deps.iter().any(|d| d.imported_path == "serde_json"));
        assert!(deps.iter().all(|d| matches!(d.import_type, crate::models::ImportType::External)));
    }

    #[test]
    fn test_parse_use_with_aliases() {
        let source = r#"
            use std::io::Result as IoResult;
            use std::collections::{HashMap as Map, HashSet};
        "#;

        let deps = RustDependencyExtractor::extract_dependencies(source).unwrap();

        // Check alias handling - should extract the original name
        let io_import = deps.iter().find(|d| d.imported_path == "std::io::Result").unwrap();
        assert!(matches!(io_import.import_type, crate::models::ImportType::Stdlib));

        let collections_import = deps.iter().find(|d| d.imported_path == "std::collections").unwrap();
        let symbols = collections_import.imported_symbols.as_ref().unwrap();
        assert_eq!(symbols.len(), 2);
        assert!(symbols.contains(&"HashMap".to_string()));
        assert!(symbols.contains(&"HashSet".to_string()));
    }

    #[test]
    fn test_classify_rust_imports() {
        use crate::models::ImportType;

        // Stdlib
        assert!(matches!(classify_rust_import("std::collections::HashMap"), ImportType::Stdlib));
        assert!(matches!(classify_rust_import("core::ptr"), ImportType::Stdlib));
        assert!(matches!(classify_rust_import("alloc::vec::Vec"), ImportType::Stdlib));

        // Internal
        assert!(matches!(classify_rust_import("crate::models::Language"), ImportType::Internal));
        assert!(matches!(classify_rust_import("super::utils"), ImportType::Internal));
        assert!(matches!(classify_rust_import("self::helper"), ImportType::Internal));

        // External
        assert!(matches!(classify_rust_import("serde::Serialize"), ImportType::External));
        assert!(matches!(classify_rust_import("anyhow::Result"), ImportType::External));
        assert!(matches!(classify_rust_import("tokio::runtime"), ImportType::External));
    }
}

// ============================================================================
// Path Resolution
// ============================================================================

/// Find the crate root (directory containing Cargo.toml) by walking up from a given path
fn find_crate_root(start_path: &str) -> Option<String> {
    let path = std::path::Path::new(start_path);
    let mut current = path.parent()?;

    // Walk up until we find Cargo.toml
    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            return Some(current.to_string_lossy().to_string());
        }

        // For test paths that don't exist, assume standard Rust structure:
        // If we find "/src" in the path, the parent of "src" is likely the crate root
        if current.ends_with("src") {
            if let Some(parent) = current.parent() {
                return Some(parent.to_string_lossy().to_string());
            }
        }

        // Move up to parent directory
        current = match current.parent() {
            Some(p) if p.as_os_str().is_empty() => return None,
            Some(p) => p,
            None => return None,
        };
    }
}

/// Resolve a Rust use statement to a file path
///
/// Handles:
/// - `crate::` imports: `crate::models::Language` → `src/models.rs` or `src/models/mod.rs`
/// - `super::` imports: relative to parent module
/// - `self::` imports: relative to current module
/// - `mod parser;`: look for `parser.rs` or `parser/mod.rs`
///
/// Does NOT handle:
/// - External crate imports (would require parsing Cargo.toml dependencies)
/// - Stdlib imports (std::, core::, alloc::)
pub fn resolve_rust_use_to_path(
    import_path: &str,
    current_file_path: Option<&str>,
    _project_root: Option<&str>,
) -> Option<String> {
    // Only handle internal imports (crate::, super::, self::, or bare module names)
    if !import_path.starts_with("crate::")
        && !import_path.starts_with("super::")
        && !import_path.starts_with("self::") {
        // Check if it's a simple module name (no :: separator at all)
        if import_path.contains("::") {
            return None; // External or stdlib import
        }
        // Fall through for simple module names like "parser"
    }

    let current_file = current_file_path?;
    let current_path = std::path::Path::new(current_file);

    // Find the crate root
    let crate_root = find_crate_root(current_file)?;
    let crate_root_path = std::path::Path::new(&crate_root);

    if import_path.starts_with("crate::") {
        // Resolve from crate root (typically src/)
        let module_path = import_path.strip_prefix("crate::").unwrap();
        let parts: Vec<&str> = module_path.split("::").collect();

        // Try src/ first (standard Rust project structure)
        let src_root = crate_root_path.join("src");
        resolve_rust_module_path(&src_root, &parts)
    } else if import_path.starts_with("super::") {
        // Resolve relative to parent module
        let module_path = import_path.strip_prefix("super::").unwrap();
        let parts: Vec<&str> = module_path.split("::").collect();

        // Get parent directory (go up one level)
        let current_dir = if current_path.file_name().unwrap() == "mod.rs" {
            // If current file is mod.rs, go up two levels
            current_path.parent()?.parent()?
        } else {
            // Otherwise, go up one level
            current_path.parent()?
        };

        resolve_rust_module_path(current_dir, &parts)
    } else if import_path.starts_with("self::") {
        // Resolve relative to current module
        let module_path = import_path.strip_prefix("self::").unwrap();
        let parts: Vec<&str> = module_path.split("::").collect();

        // Get current module directory
        let current_dir = if current_path.file_name().unwrap() == "mod.rs" {
            // If current file is mod.rs, use parent directory
            current_path.parent()?
        } else {
            // Otherwise, use current directory
            current_path.parent()?
        };

        resolve_rust_module_path(current_dir, &parts)
    } else {
        // Simple module name (e.g., "parser" in "mod parser;")
        // Look for parser.rs or parser/mod.rs in the current directory
        let current_dir = current_path.parent()?;
        let module_file = current_dir.join(format!("{}.rs", import_path));
        let module_dir = current_dir.join(import_path).join("mod.rs");

        if module_file.exists() {
            Some(module_file.to_string_lossy().to_string())
        } else if module_dir.exists() {
            Some(module_dir.to_string_lossy().to_string())
        } else {
            // Return the most likely candidate even if it doesn't exist
            // The indexer will check if the file is actually in the index
            Some(module_file.to_string_lossy().to_string())
        }
    }
}

/// Resolve a Rust module path (list of components) to a file path
///
/// Examples:
/// - `["models"]` → `models.rs` or `models/mod.rs`
/// - `["models", "language"]` → `models/language.rs` or `models/language/mod.rs`
fn resolve_rust_module_path(base_dir: &std::path::Path, parts: &[&str]) -> Option<String> {
    if parts.is_empty() {
        return None;
    }

    // Build the path incrementally
    let mut current_path = base_dir.to_path_buf();

    for (i, part) in parts.iter().enumerate() {
        if i == parts.len() - 1 {
            // Last component - try both .rs file and mod.rs
            let file_path = current_path.join(format!("{}.rs", part));
            let mod_path = current_path.join(part).join("mod.rs");

            log::trace!("Checking Rust module path: {}", file_path.display());
            log::trace!("Checking Rust module path: {}", mod_path.display());

            // Return the first candidate (indexer will validate it exists)
            if file_path.exists() {
                return Some(file_path.to_string_lossy().to_string());
            } else if mod_path.exists() {
                return Some(mod_path.to_string_lossy().to_string());
            } else {
                // Return most likely candidate even if it doesn't exist
                return Some(file_path.to_string_lossy().to_string());
            }
        } else {
            // Intermediate component - must be a directory
            current_path = current_path.join(part);
        }
    }

    None
}

#[cfg(test)]
mod path_resolution_tests {
    use super::*;

    #[test]
    fn test_resolve_crate_import() {
        // crate::models::Language
        let result = resolve_rust_use_to_path(
            "crate::models",
            Some("/home/user/project/src/main.rs"),
            Some("/home/user/project"),
        );

        assert!(result.is_some());
        let path = result.unwrap();
        // Should resolve to src/models.rs or src/models/mod.rs
        assert!(path.contains("models.rs") || path.contains("models/mod.rs"));
    }

    #[test]
    fn test_resolve_super_import() {
        // super::utils from src/commands/index.rs
        let result = resolve_rust_use_to_path(
            "super::utils",
            Some("/home/user/project/src/commands/index.rs"),
            Some("/home/user/project"),
        );

        assert!(result.is_some());
        let path = result.unwrap();
        // Should resolve to src/utils.rs
        assert!(path.contains("src") && path.contains("utils.rs"));
    }

    #[test]
    fn test_resolve_self_import() {
        // self::helper from src/models/mod.rs
        let result = resolve_rust_use_to_path(
            "self::helper",
            Some("/home/user/project/src/models/mod.rs"),
            Some("/home/user/project"),
        );

        assert!(result.is_some());
        let path = result.unwrap();
        // Should resolve to src/models/helper.rs
        assert!(path.contains("models") && path.contains("helper.rs"));
    }

    #[test]
    fn test_resolve_mod_declaration() {
        // mod parser; from src/main.rs
        let result = resolve_rust_use_to_path(
            "parser",
            Some("/home/user/project/src/main.rs"),
            Some("/home/user/project"),
        );

        assert!(result.is_some());
        let path = result.unwrap();
        // Should resolve to src/parser.rs
        assert!(path.contains("parser.rs"));
    }

    #[test]
    fn test_resolve_nested_crate_import() {
        // crate::models::language::Language
        let result = resolve_rust_use_to_path(
            "crate::models::language",
            Some("/home/user/project/src/main.rs"),
            Some("/home/user/project"),
        );

        assert!(result.is_some());
        let path = result.unwrap();
        // Should resolve to src/models/language.rs or src/models/language/mod.rs
        assert!(path.contains("models") && (path.contains("language.rs") || path.contains("language/mod.rs")));
    }

    #[test]
    fn test_external_import_not_supported() {
        // anyhow::Result (external crate)
        let result = resolve_rust_use_to_path(
            "anyhow::Result",
            Some("/home/user/project/src/main.rs"),
            Some("/home/user/project"),
        );

        // Should return None for external imports
        assert!(result.is_none());
    }

    #[test]
    fn test_stdlib_import_not_supported() {
        // std::collections::HashMap (stdlib)
        let result = resolve_rust_use_to_path(
            "std::collections::HashMap",
            Some("/home/user/project/src/main.rs"),
            Some("/home/user/project"),
        );

        // Should return None for stdlib imports
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_without_current_file() {
        let result = resolve_rust_use_to_path(
            "crate::models",
            None,
            Some("/home/user/project"),
        );

        // Should return None if no current file provided
        assert!(result.is_none());
    }
}

// ============================================================================
// WORKSPACE SUPPORT (NEW) - 20260112
// ============================================================================

#[derive(Debug, Clone)]
pub struct RustCrate {
    pub name: String,
    pub root_path: std::path::PathBuf,
}

/// Find all Rust crates in the workspace by looking for Cargo.toml files
pub fn parse_all_rust_crates(root: &std::path::Path) -> anyhow::Result<Vec<RustCrate>> {
    let mut crates = Vec::new();
    // Scan the entire directory tree respecting gitignore
    let walker = ignore::WalkBuilder::new(root)
        .git_ignore(true)
        .build();

    for entry in walker {
        let entry = entry?;
        // Found a Cargo.toml
        if entry.file_name() == "Cargo.toml" {
            // Read it to extract the package name
            // Simple string parsing to avoid adding new dependencies like 'toml'
            let content = std::fs::read_to_string(entry.path())?;
            if let Some(name) = extract_crate_name(&content) {
                // The directory containing Cargo.toml is the crate root
                if let Some(crate_root) = entry.path().parent() {
                    crates.push(RustCrate {
                        name,
                        root_path: crate_root.to_path_buf(),
                    });
                }
            }
        }
    }
    Ok(crates)
}

/// Robust extraction of name="..." from [package] section
fn extract_crate_name(content: &str) -> Option<String> {
    let mut in_package_section = false;
    for line in content.lines() {
        let line = line.trim();
        if line == "[package]" {
            in_package_section = true;
            continue;
        }
        if in_package_section {
            if line.starts_with('[') {
                break; // End of package section
            }
            if line.starts_with("name") {
                // Parse: name = "ts_shared"
                if let Some(val) = line.split('=').nth(1) {
                    return Some(val.trim().trim_matches('"').trim_matches('\'').to_string());
                }
            }
        }
    }
    None
}

/// Reclassify an import based on known workspace crates
pub fn reclassify_rust_import(
    path: &str,
    crates: &[RustCrate],
) -> crate::models::ImportType {
    // If the import matches a known crate name, it is Internal
    for krate in crates {
        // Match "ts_shared" or "ts_shared::submodule"
        if path == krate.name || path.starts_with(&format!("{}::", krate.name)) {
            return crate::models::ImportType::Internal;
        }
    }
    // Fallback to default classification (std, external, etc.)
    classify_rust_import(path)
}

/// Resolve a workspace import to a file path
/// e.g., "ts_shared::config" -> "/path/to/ts_shared/src/config.rs"
pub fn resolve_rust_workspace_path(
    import_path: &str,
    crates: &[RustCrate],
) -> Option<String> {
    // 1. Find which crate matches the start of the import
    for krate in crates {
        if import_path == krate.name || import_path.starts_with(&format!("{}::", krate.name)) {
            // 2. Strip crate name to get relative module path
            // "ts_shared::config" -> "config"
            let relative_module = if import_path == krate.name {
                "" 
            } else {
                import_path.strip_prefix(&format!("{}::", krate.name)).unwrap_or("")
            };
            
            // 3. Assume standard "src/" layout
            let src_root = krate.root_path.join("src");
            
            // 4. Try to resolve the path
            if relative_module.is_empty() {
                // "use ts_shared;" -> likely lib.rs
                let lib = src_root.join("lib.rs");
                if lib.exists() { return Some(lib.to_string_lossy().to_string()); }
                let main = src_root.join("main.rs");
                if main.exists() { return Some(main.to_string_lossy().to_string()); }
            } else {
                // "use ts_shared::config;"
                let parts: Vec<&str> = relative_module.split("::").collect();
                
                // Attempt 1: The path points directly to a module/file
                if let Some(path) = resolve_rust_module_path(&src_root, &parts) {
                    return Some(path);
                }
                
                // Attempt 2: The path points to an ITEM (struct/func) inside a file
                // Try popping the last component ("config::Struct") -> resolve "config"
                if parts.len() > 1 {
                    if let Some(path) = resolve_rust_module_path(&src_root, &parts[..parts.len()-1]) {
                        return Some(path);
                    }
                }
            }
        }
    }
    None
}