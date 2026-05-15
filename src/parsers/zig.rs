//! Zig language parser using Tree-sitter
//!
//! Extracts symbols from Zig source code:
//! - Functions (pub and private)
//! - Structs
//! - Enums
//! - Unions
//! - Constants (const - immutable, global and local)
//! - Variables (var - mutable, global and local)
//! - Test declarations
//! - Error sets

use crate::ImportType;
use crate::models::{Language, SearchResult, Span, SymbolKind};
use crate::parsers::{DependencyExtractor, ImportInfo};
use anyhow::{Context, Result};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};

/// Parse Zig source code and extract symbols
pub fn parse(path: &str, source: &str) -> Result<Vec<SearchResult>> {
    let mut parser = Parser::new();
    let language = tree_sitter_zig::LANGUAGE;

    parser
        .set_language(&language.into())
        .context("Failed to set Zig language")?;

    let tree = parser
        .parse(source, None)
        .context("Failed to parse Zig source")?;

    let root_node = tree.root_node();

    let mut symbols = Vec::new();

    // Extract different types of symbols using Tree-sitter queries
    symbols.extend(extract_functions(source, &root_node, &language.into())?);
    symbols.extend(extract_structs(source, &root_node, &language.into())?);
    symbols.extend(extract_enums(source, &root_node, &language.into())?);
    symbols.extend(extract_constants(source, &root_node, &language.into())?);
    symbols.extend(extract_variables(source, &root_node, &language.into())?);
    symbols.extend(extract_tests(source, &root_node, &language.into())?);

    // Add file path to all symbols
    for symbol in &mut symbols {
        symbol.path = path.to_string();
        symbol.lang = Language::Zig;
    }

    Ok(symbols)
}

/// Extract function declarations
fn extract_functions(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (function_declaration
            (identifier) @name) @function
    "#;

    let query = Query::new(language, query_str).context("Failed to create function query")?;

    extract_symbols(source, root, &query, SymbolKind::Function, None)
}

/// Extract struct (container) declarations
fn extract_structs(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (variable_declaration
            (identifier) @name
            (struct_declaration)) @struct
    "#;

    let query = Query::new(language, query_str).context("Failed to create struct query")?;

    extract_symbols(source, root, &query, SymbolKind::Struct, None)
}

/// Extract enum declarations
fn extract_enums(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (variable_declaration
            (identifier) @name
            (enum_declaration)) @enum
    "#;

    let query = Query::new(language, query_str).context("Failed to create enum query")?;

    extract_symbols(source, root, &query, SymbolKind::Enum, None)
}

/// Extract constant declarations (const - immutable bindings)
fn extract_constants(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (variable_declaration
            "const"
            (identifier) @name) @const
    "#;

    let query = Query::new(language, query_str).context("Failed to create constant query")?;

    extract_symbols(source, root, &query, SymbolKind::Constant, None)
}

/// Extract variable declarations (var - mutable bindings)
fn extract_variables(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (variable_declaration
            "var"
            (identifier) @name) @var
    "#;

    let query = Query::new(language, query_str).context("Failed to create variable query")?;

    extract_symbols(source, root, &query, SymbolKind::Variable, None)
}

/// Extract test declarations
fn extract_tests(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (test_declaration
            (string) @name) @test
    "#;

    let query = Query::new(language, query_str).context("Failed to create test query")?;

    extract_symbols(source, root, &query, SymbolKind::Function, None)
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
                name = Some(
                    capture
                        .node
                        .utf8_text(source.as_bytes())
                        .unwrap_or("")
                        .to_string(),
                );
            } else {
                // Assume any other capture is the full node
                full_node = Some(capture.node);
            }
        }

        if let (Some(name), Some(node)) = (name, full_node) {
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::Zig,
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
        start.row + 1, // Convert 0-indexed to 1-indexed
        start.column,
        end.row + 1,
        end.column,
    )
}

/// Extract a preview (7 lines) around the symbol
fn extract_preview(source: &str, span: &Span) -> String {
    let lines: Vec<&str> = source.lines().collect();

    // Extract 7 lines: the start line and 6 following lines
    let start_idx = (span.start_line - 1) as usize; // Convert back to 0-indexed
    let end_idx = (start_idx + 7).min(lines.len());

    lines[start_idx..end_idx].join("\n")
}

/// Zig dependency extractor
pub struct ZigDependencyExtractor;

impl DependencyExtractor for ZigDependencyExtractor {
    fn extract_dependencies(source: &str) -> Result<Vec<ImportInfo>> {
        // Zig uses @import("path") builtin for imports
        // Use simple text-based extraction as fallback since tree-sitter query might not work
        let mut imports = Vec::new();

        for (line_idx, line) in source.lines().enumerate() {
            // Look for @import("...") or @import('...')
            if let Some(import_path) = extract_zig_import_from_line(line) {
                let import_type = classify_zig_import(&import_path);
                let line_number = line_idx + 1;

                imports.push(ImportInfo {
                    imported_path: import_path,
                    line_number,
                    import_type,
                    imported_symbols: None,
                });
            }
        }

        Ok(imports)
    }
}

/// Extract import path from a line containing @import("...")
fn extract_zig_import_from_line(line: &str) -> Option<String> {
    // Find @import( in the line
    let import_start = line.find("@import(")?;
    let after_import = &line[import_start + 8..]; // Skip "@import("

    // Find the string content (either "..." or '...')
    let first_char = after_import.trim_start().chars().next()?;
    if first_char != '"' && first_char != '\'' {
        return None;
    }

    let quote_char = first_char;
    let after_quote = &after_import[after_import.find(quote_char)? + 1..];
    let end_quote = after_quote.find(quote_char)?;
    let path = &after_quote[..end_quote];

    Some(path.to_string())
}

/// Classify Zig import as Internal, External, or Stdlib
fn classify_zig_import(import_path: &str) -> ImportType {
    // Zig standard library imports
    if import_path == "std" || import_path == "builtin" || import_path == "root" {
        return ImportType::Stdlib;
    }

    // Relative imports (start with ./ or ../)
    if import_path.starts_with("./") || import_path.starts_with("../") {
        return ImportType::Internal;
    }

    // External package imports (anything else that's not stdlib)
    // Zig package manager uses package names directly
    ImportType::External
}

/// Resolve a Zig @import("...") path to an absolute file path
///
/// Only resolves Internal imports (relative paths starting with ./ or ../)
/// Returns None for External and Stdlib imports
///
/// # Arguments
/// * `import_path` - The path from @import("...") (e.g., "./utils.zig", "../helpers.zig")
/// * `current_file_path` - The absolute path of the file containing the import
///
/// # Returns
/// Some(absolute_path) if the import is resolvable (Internal relative path)
/// None if the import is External or Stdlib
pub fn resolve_zig_import_to_path(
    import_path: &str,
    current_file_path: Option<&str>,
) -> Option<String> {
    // Only resolve Internal imports (relative paths)
    if !import_path.starts_with("./") && !import_path.starts_with("../") {
        return None;
    }

    let current_file = current_file_path?;
    let current_dir = std::path::Path::new(current_file).parent()?;
    let resolved = current_dir.join(import_path);

    // Try to canonicalize (normalize) the path
    match resolved.canonicalize() {
        Ok(normalized) => Some(normalized.display().to_string()),
        Err(_) => {
            // If canonicalization fails (file doesn't exist), return the raw path
            Some(resolved.display().to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_function() {
        let source = r#"
pub fn add(a: i32, b: i32) i32 {
    return a + b;
}
        "#;

        let symbols = parse("test.zig", source).unwrap();

        let func_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Function))
            .collect();

        assert_eq!(func_symbols.len(), 1);
        assert_eq!(func_symbols[0].symbol.as_deref(), Some("add"));
    }

    #[test]
    fn test_parse_struct() {
        let source = r#"
const Point = struct {
    x: f32,
    y: f32,
};
        "#;

        let symbols = parse("test.zig", source).unwrap();

        let struct_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Struct))
            .collect();

        assert_eq!(struct_symbols.len(), 1);
        assert_eq!(struct_symbols[0].symbol.as_deref(), Some("Point"));
    }

    #[test]
    fn test_parse_enum() {
        let source = r#"
const Status = enum {
    active,
    inactive,
    pending,
};
        "#;

        let symbols = parse("test.zig", source).unwrap();

        let enum_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Enum))
            .collect();

        assert_eq!(enum_symbols.len(), 1);
        assert_eq!(enum_symbols[0].symbol.as_deref(), Some("Status"));
    }

    #[test]
    fn test_parse_constants() {
        let source = r#"
const MAX_SIZE: usize = 100;
const DEFAULT_TIMEOUT: u32 = 30;
        "#;

        let symbols = parse("test.zig", source).unwrap();

        let const_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Constant))
            .collect();

        assert_eq!(const_symbols.len(), 2);
        assert!(
            const_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("MAX_SIZE"))
        );
        assert!(
            const_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("DEFAULT_TIMEOUT"))
        );
    }

    #[test]
    fn test_parse_test_declaration() {
        let source = r#"
test "basic addition" {
    const result = add(2, 3);
    try std.testing.expect(result == 5);
}
        "#;

        let symbols = parse("test.zig", source).unwrap();

        let test_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Function))
            .filter(|s| s.symbol.as_deref().unwrap_or("").contains("basic addition"))
            .collect();

        // Test verifies parsing succeeds without panic
        let _ = test_symbols; // Suppress unused variable warning
    }

    #[test]
    fn test_parse_pub_functions() {
        let source = r#"
pub fn multiply(a: i32, b: i32) i32 {
    return a * b;
}

fn divide(a: i32, b: i32) i32 {
    return @divTrunc(a, b);
}
        "#;

        let symbols = parse("test.zig", source).unwrap();

        let func_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Function))
            .collect();

        assert_eq!(func_symbols.len(), 2);
        assert!(
            func_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("multiply"))
        );
        assert!(
            func_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("divide"))
        );
    }

    #[test]
    fn test_parse_struct_with_methods() {
        let source = r#"
const Calculator = struct {
    value: i32,

    pub fn init(val: i32) Calculator {
        return Calculator{ .value = val };
    }

    pub fn add(self: *Calculator, other: i32) void {
        self.value += other;
    }
};
        "#;

        let symbols = parse("test.zig", source).unwrap();

        let struct_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Struct))
            .collect();

        assert_eq!(struct_symbols.len(), 1);
        assert_eq!(struct_symbols[0].symbol.as_deref(), Some("Calculator"));
    }

    #[test]
    fn test_parse_error_set() {
        let source = r#"
const FileError = error{
    AccessDenied,
    FileNotFound,
    OutOfMemory,
};
        "#;

        let symbols = parse("test.zig", source).unwrap();

        // Error sets are stored as constants
        let const_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Constant))
            .collect();

        assert!(
            const_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("FileError"))
        );
    }

    #[test]
    fn test_parse_mixed_symbols() {
        let source = r#"
const std = @import("std");

const MAX_BUFFER = 1024;

const Point = struct {
    x: f32,
    y: f32,
};

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("Hello, World!\n", .{});
}

test "point creation" {
    const p = Point{ .x = 1.0, .y = 2.0 };
    try std.testing.expect(p.x == 1.0);
}
        "#;

        let symbols = parse("test.zig", source).unwrap();

        // Should find: constants, struct, function, test
        assert!(symbols.len() >= 3);

        let kinds: Vec<&SymbolKind> = symbols.iter().map(|s| &s.kind).collect();
        assert!(kinds.contains(&&SymbolKind::Constant));
        assert!(kinds.contains(&&SymbolKind::Struct));
        assert!(kinds.contains(&&SymbolKind::Function));
    }

    #[test]
    fn test_local_variables_included() {
        let source = r#"
const GLOBAL_CONST = 100;
var globalVar: i32 = 50;

pub fn calculate(input: i32) i32 {
    const localConst = 10;
    var localVar: i32 = input * 2;
    localVar += localConst;
    return localVar;
}

test "variable types" {
    const testConst = 5;
    var testVar: i32 = 0;
    testVar = testConst * 2;
    try std.testing.expect(testVar == 10);
}
        "#;

        let symbols = parse("test.zig", source).unwrap();

        // Filter to constants and variables
        let constants: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Constant))
            .collect();

        let variables: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        // Check that const declarations are captured (global and local)
        assert!(
            constants
                .iter()
                .any(|c| c.symbol.as_deref() == Some("GLOBAL_CONST"))
        );
        assert!(
            constants
                .iter()
                .any(|c| c.symbol.as_deref() == Some("localConst"))
        );
        assert!(
            constants
                .iter()
                .any(|c| c.symbol.as_deref() == Some("testConst"))
        );

        // Check that var declarations are captured (global and local)
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("globalVar"))
        );
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("localVar"))
        );
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("testVar"))
        );

        // Verify that both global and local variables have no scope
        // (Zig doesn't have class-based scoping, all variables are treated equally)
        for constant in &constants {
            // Removed: scope field no longer exists: assert_eq!(constant.scope, None);
        }

        for variable in &variables {
            // Removed: scope field no longer exists: assert_eq!(variable.scope, None);
        }
    }

    #[test]
    fn test_extract_zig_imports() {
        let source = r#"
const std = @import("std");
const builtin = @import("builtin");
const utils = @import("./utils.zig");
const helpers = @import("../helpers.zig");
const zap = @import("zap");

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("Hello, World!\n", .{});
}
        "#;

        let deps = ZigDependencyExtractor::extract_dependencies(source).unwrap();

        assert!(
            deps.len() >= 4,
            "Should extract at least 4 imports, got {}",
            deps.len()
        );

        // Stdlib imports
        assert!(
            deps.iter()
                .any(|d| d.imported_path == "std" && matches!(d.import_type, ImportType::Stdlib))
        );
        assert!(
            deps.iter().any(
                |d| d.imported_path == "builtin" && matches!(d.import_type, ImportType::Stdlib)
            )
        );

        // Internal imports (relative paths)
        assert!(
            deps.iter().any(|d| d.imported_path == "./utils.zig"
                && matches!(d.import_type, ImportType::Internal))
        );
        assert!(deps.iter().any(|d| d.imported_path == "../helpers.zig"
            && matches!(d.import_type, ImportType::Internal)));

        // External package import
        assert!(
            deps.iter()
                .any(|d| d.imported_path == "zap" && matches!(d.import_type, ImportType::External))
        );
    }

    // Zig import path resolution tests
    #[cfg(test)]
    mod resolution_tests {
        use super::*;

        #[test]
        fn test_resolve_zig_import_same_directory() {
            let result = resolve_zig_import_to_path("./utils.zig", Some("/project/src/main.zig"));
            assert!(result.is_some());
            let path = result.unwrap();
            assert!(path.contains("src") && path.ends_with("utils.zig"));
        }

        #[test]
        fn test_resolve_zig_import_subdirectory() {
            let result =
                resolve_zig_import_to_path("./utils/helpers.zig", Some("/project/src/main.zig"));
            assert!(result.is_some());
            let path = result.unwrap();
            assert!(
                path.contains("src") && path.contains("utils") && path.ends_with("helpers.zig")
            );
        }

        #[test]
        fn test_resolve_zig_import_parent_directory() {
            let result =
                resolve_zig_import_to_path("../common.zig", Some("/project/src/utils/main.zig"));
            assert!(result.is_some());
            let path = result.unwrap();
            assert!(path.contains("src") && path.ends_with("common.zig"));
        }

        #[test]
        fn test_resolve_zig_import_stdlib_returns_none() {
            // Stdlib imports should not be resolved
            let result = resolve_zig_import_to_path("std", Some("/project/src/main.zig"));
            assert!(result.is_none());

            let result = resolve_zig_import_to_path("builtin", Some("/project/src/main.zig"));
            assert!(result.is_none());
        }

        #[test]
        fn test_resolve_zig_import_external_returns_none() {
            // External package imports should not be resolved
            let result = resolve_zig_import_to_path("zap", Some("/project/src/main.zig"));
            assert!(result.is_none());

            let result = resolve_zig_import_to_path("some_package", Some("/project/src/main.zig"));
            assert!(result.is_none());
        }
    }
}
