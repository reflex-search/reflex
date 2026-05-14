//! C# language parser using Tree-sitter
//!
//! Extracts symbols from C# source code:
//! - Classes (regular, abstract, sealed, partial, static)
//! - Interfaces
//! - Structs
//! - Enums
//! - Delegates
//! - Records (C# 9+)
//! - Methods (with class scope, visibility)
//! - Properties (class/struct/record members)
//! - Events (class/struct/interface members)
//! - Indexers (this[] accessor)
//! - Local variables (inside methods)
//! - Namespaces
//! - Constructors

use anyhow::{Context, Result};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};
use crate::models::{Language, SearchResult, Span, SymbolKind};

/// Parse C# source code and extract symbols
pub fn parse(path: &str, source: &str) -> Result<Vec<SearchResult>> {
    let mut parser = Parser::new();
    let language = tree_sitter_c_sharp::LANGUAGE;

    parser
        .set_language(&language.into())
        .context("Failed to set C# language")?;

    let tree = parser
        .parse(source, None)
        .context("Failed to parse C# source")?;

    let root_node = tree.root_node();

    let mut symbols = Vec::new();

    // Extract different types of symbols using Tree-sitter queries
    symbols.extend(extract_namespaces(source, &root_node, &language.into())?);
    symbols.extend(extract_classes(source, &root_node, &language.into())?);
    symbols.extend(extract_interfaces(source, &root_node, &language.into())?);
    symbols.extend(extract_structs(source, &root_node, &language.into())?);
    symbols.extend(extract_enums(source, &root_node, &language.into())?);
    symbols.extend(extract_records(source, &root_node, &language.into())?);
    symbols.extend(extract_delegates(source, &root_node, &language.into())?);
    symbols.extend(extract_attributes(source, &root_node, &language.into())?);
    symbols.extend(extract_methods(source, &root_node, &language.into())?);
    symbols.extend(extract_properties(source, &root_node, &language.into())?);
    symbols.extend(extract_events(source, &root_node, &language.into())?);
    symbols.extend(extract_indexers(source, &root_node, &language.into())?);
    symbols.extend(extract_local_variables(source, &root_node, &language.into())?);

    // Add file path to all symbols
    for symbol in &mut symbols {
        symbol.path = path.to_string();
        symbol.lang = Language::CSharp;
    }

    Ok(symbols)
}

/// Extract namespace declarations
fn extract_namespaces(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (namespace_declaration
            name: (_) @name) @namespace

        (file_scoped_namespace_declaration
            name: (_) @name) @namespace
    "#;

    let query = Query::new(language, query_str)
        .context("Failed to create namespace query")?;

    extract_symbols(source, root, &query, SymbolKind::Namespace, None)
}

/// Extract class declarations
fn extract_classes(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_declaration
            name: (identifier) @name) @class
    "#;

    let query = Query::new(language, query_str)
        .context("Failed to create class query")?;

    extract_symbols(source, root, &query, SymbolKind::Class, None)
}

/// Extract interface declarations
fn extract_interfaces(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (interface_declaration
            name: (identifier) @name) @interface
    "#;

    let query = Query::new(language, query_str)
        .context("Failed to create interface query")?;

    extract_symbols(source, root, &query, SymbolKind::Interface, None)
}

/// Extract struct declarations
fn extract_structs(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (struct_declaration
            name: (identifier) @name) @struct
    "#;

    let query = Query::new(language, query_str)
        .context("Failed to create struct query")?;

    extract_symbols(source, root, &query, SymbolKind::Struct, None)
}

/// Extract enum declarations
fn extract_enums(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (enum_declaration
            name: (identifier) @name) @enum
    "#;

    let query = Query::new(language, query_str)
        .context("Failed to create enum query")?;

    extract_symbols(source, root, &query, SymbolKind::Enum, None)
}

/// Extract record declarations (C# 9+)
fn extract_records(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (record_declaration
            name: (identifier) @name) @record
    "#;

    let query = Query::new(language, query_str)
        .context("Failed to create record query")?;

    extract_symbols(source, root, &query, SymbolKind::Type, None)
}

/// Extract delegate declarations
fn extract_delegates(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (delegate_declaration
            name: (identifier) @name) @delegate
    "#;

    let query = Query::new(language, query_str)
        .context("Failed to create delegate query")?;

    extract_symbols(source, root, &query, SymbolKind::Type, None)
}

/// Extract attributes: BOTH definitions and uses
/// Definitions: class TestAttribute : Attribute { ... }
/// Uses: [Test] public void TestMethod() { ... }
fn extract_attributes(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let mut symbols = Vec::new();

    // Part 1: Extract attribute class DEFINITIONS
    let def_query_str = r#"
        (class_declaration
            name: (identifier) @name) @class
    "#;

    let def_query = Query::new(language, def_query_str)
        .context("Failed to create attribute definition query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&def_query, *root, source.as_bytes());

    while let Some(match_) = matches.next() {
        let mut name = None;
        let mut full_node = None;

        for capture in match_.captures {
            let capture_name: &str = &def_query.capture_names()[capture.index as usize];
            match capture_name {
                "name" => {
                    name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                }
                "class" => {
                    full_node = Some(capture.node);
                }
                _ => {}
            }
        }

        // Extract classes that end with "Attribute" suffix (C# naming convention)
        // or that have Attribute in their base_list
        if let (Some(name), Some(node)) = (name, full_node) {
            let mut is_attribute = name.ends_with("Attribute");

            // Also check if the class inherits from Attribute
            if !is_attribute {
                // Check base_list for inheritance
                for i in 0..node.child_count() {
                    if let Some(child) = node.child(i as u32) {
                        if child.kind() == "base_list" {
                            let base_text = child.utf8_text(source.as_bytes()).unwrap_or("");
                            if base_text.contains("Attribute") {
                                is_attribute = true;
                                break;
                            }
                        }
                    }
                }
            }

            if is_attribute {
                let span = node_to_span(&node);
                let preview = extract_preview(source, &span);

                symbols.push(SearchResult::new(
                    String::new(),
                    Language::CSharp,
                    SymbolKind::Attribute,
                    Some(name),
                    span,
                    None,
                    preview,
                ));
            }
        }
    }

    // Part 2: Extract attribute USES ([Test], [Obsolete], etc.)
    let use_query_str = r#"
        (attribute_list
            (attribute
                name: (_) @name)) @attr
    "#;

    let use_query = Query::new(language, use_query_str)
        .context("Failed to create attribute use query")?;

    symbols.extend(extract_symbols(source, root, &use_query, SymbolKind::Attribute, None)?);

    Ok(symbols)
}

/// Extract method declarations from classes, structs, and interfaces
fn extract_methods(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_declaration
            name: (identifier) @class_name
            body: (declaration_list
                (method_declaration
                    name: (identifier) @method_name))) @class

        (struct_declaration
            name: (identifier) @struct_name
            body: (declaration_list
                (method_declaration
                    name: (identifier) @method_name))) @struct

        (interface_declaration
            name: (identifier) @interface_name
            body: (declaration_list
                (method_declaration
                    name: (identifier) @method_name))) @interface

        (record_declaration
            name: (identifier) @record_name
            body: (declaration_list
                (method_declaration
                    name: (identifier) @method_name))) @record
    "#;

    let query = Query::new(language, query_str)
        .context("Failed to create method query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut scope_name = None;
        let mut scope_type = None;
        let mut method_name = None;
        let mut method_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            match capture_name {
                "class_name" => {
                    scope_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                    scope_type = Some("class");
                }
                "struct_name" => {
                    scope_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                    scope_type = Some("struct");
                }
                "interface_name" => {
                    scope_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                    scope_type = Some("interface");
                }
                "record_name" => {
                    scope_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                    scope_type = Some("record");
                }
                "method_name" => {
                    method_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                    // Find the parent method_declaration node
                    let mut current = capture.node;
                    while let Some(parent) = current.parent() {
                        if parent.kind() == "method_declaration" {
                            method_node = Some(parent);
                            break;
                        }
                        current = parent;
                    }
                }
                _ => {}
            }
        }

        if let (Some(scope_name), Some(scope_type), Some(method_name), Some(node)) =
            (scope_name, scope_type, method_name, method_node) {
            let scope = format!("{} {}", scope_type, scope_name);
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::CSharp,
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

/// Extract property declarations
fn extract_properties(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_declaration
            name: (identifier) @class_name
            body: (declaration_list
                (property_declaration
                    name: (identifier) @property_name))) @class

        (struct_declaration
            name: (identifier) @struct_name
            body: (declaration_list
                (property_declaration
                    name: (identifier) @property_name))) @struct

        (interface_declaration
            name: (identifier) @interface_name
            body: (declaration_list
                (property_declaration
                    name: (identifier) @property_name))) @interface

        (record_declaration
            name: (identifier) @record_name
            body: (declaration_list
                (property_declaration
                    name: (identifier) @property_name))) @record
    "#;

    let query = Query::new(language, query_str)
        .context("Failed to create property query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut scope_name = None;
        let mut scope_type = None;
        let mut property_name = None;
        let mut property_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            match capture_name {
                "class_name" => {
                    scope_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                    scope_type = Some("class");
                }
                "struct_name" => {
                    scope_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                    scope_type = Some("struct");
                }
                "interface_name" => {
                    scope_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                    scope_type = Some("interface");
                }
                "record_name" => {
                    scope_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                    scope_type = Some("record");
                }
                "property_name" => {
                    property_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                    // Find the parent property_declaration node
                    let mut current = capture.node;
                    while let Some(parent) = current.parent() {
                        if parent.kind() == "property_declaration" {
                            property_node = Some(parent);
                            break;
                        }
                        current = parent;
                    }
                }
                _ => {}
            }
        }

        if let (Some(scope_name), Some(scope_type), Some(property_name), Some(node)) =
            (scope_name, scope_type, property_name, property_node) {
            let scope = format!("{} {}", scope_type, scope_name);
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::CSharp,
                SymbolKind::Variable,
                Some(property_name),
                span,
                Some(scope),
                preview,
            ));
        }
    }

    Ok(symbols)
}

/// Extract event declarations from classes, structs, and interfaces
fn extract_events(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_declaration
            name: (identifier) @class_name
            body: (declaration_list
                (event_field_declaration
                    (variable_declaration
                        (variable_declarator
                            (identifier) @event_name))))) @class

        (struct_declaration
            name: (identifier) @struct_name
            body: (declaration_list
                (event_field_declaration
                    (variable_declaration
                        (variable_declarator
                            (identifier) @event_name))))) @struct

        (interface_declaration
            name: (identifier) @interface_name
            body: (declaration_list
                (event_field_declaration
                    (variable_declaration
                        (variable_declarator
                            (identifier) @event_name))))) @interface
    "#;

    let query = Query::new(language, query_str)
        .context("Failed to create event query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut scope_name = None;
        let mut scope_type = None;
        let mut event_name = None;
        let mut event_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            match capture_name {
                "class_name" => {
                    scope_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                    scope_type = Some("class");
                }
                "struct_name" => {
                    scope_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                    scope_type = Some("struct");
                }
                "interface_name" => {
                    scope_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                    scope_type = Some("interface");
                }
                "event_name" => {
                    event_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                    // Find the parent event_field_declaration node
                    let mut current = capture.node;
                    while let Some(parent) = current.parent() {
                        if parent.kind() == "event_field_declaration" {
                            event_node = Some(parent);
                            break;
                        }
                        current = parent;
                    }
                }
                _ => {}
            }
        }

        if let (Some(scope_name), Some(scope_type), Some(event_name), Some(node)) =
            (scope_name, scope_type, event_name, event_node) {
            let scope = format!("{} {}", scope_type, scope_name);
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::CSharp,
                SymbolKind::Event,
                Some(event_name),
                span,
                Some(scope),
                preview,
            ));
        }
    }

    Ok(symbols)
}

/// Extract indexer declarations from classes and structs
fn extract_indexers(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_declaration
            name: (identifier) @class_name
            body: (declaration_list
                (indexer_declaration) @indexer_name)) @class

        (struct_declaration
            name: (identifier) @struct_name
            body: (declaration_list
                (indexer_declaration) @indexer_name)) @struct
    "#;

    let query = Query::new(language, query_str)
        .context("Failed to create indexer query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut scope_name = None;
        let mut scope_type = None;
        let mut indexer_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            match capture_name {
                "class_name" => {
                    scope_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                    scope_type = Some("class");
                }
                "struct_name" => {
                    scope_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                    scope_type = Some("struct");
                }
                "indexer_name" => {
                    indexer_node = Some(capture.node);
                }
                _ => {}
            }
        }

        if let (Some(scope_name), Some(scope_type), Some(node)) =
            (scope_name, scope_type, indexer_node) {
            let scope = format!("{} {}", scope_type, scope_name);
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            // Use "this[]" as the indexer name (C# convention)
            symbols.push(SearchResult::new(
                String::new(),
                Language::CSharp,
                SymbolKind::Property,
                Some("this[]".to_string()),
                span,
                Some(scope),
                preview,
            ));
        }
    }

    Ok(symbols)
}

/// Extract local variable declarations inside methods
fn extract_local_variables(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (local_declaration_statement
            (variable_declaration
                (variable_declarator
                    (identifier) @name))) @var
    "#;

    let query = Query::new(language, query_str)
        .context("Failed to create local variable query")?;

    extract_symbols(source, root, &query, SymbolKind::Variable, None)
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
                String::new(),
                Language::CSharp,
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

/// Extract a preview (7 lines) around the symbol
fn extract_preview(source: &str, span: &Span) -> String {
    let lines: Vec<&str> = source.lines().collect();

    // Extract 7 lines: the start line and 6 following lines
    let start_idx = (span.start_line - 1) as usize; // Convert back to 0-indexed
    let end_idx = (start_idx + 7).min(lines.len());

    lines[start_idx..end_idx].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_class() {
        let source = r#"
public class User
{
    private string name;
    private int age;
}
        "#;

        let symbols = parse("test.cs", source).unwrap();

        let class_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        assert_eq!(class_symbols.len(), 1);
        assert_eq!(class_symbols[0].symbol.as_deref(), Some("User"));
    }

    #[test]
    fn test_parse_namespace() {
        let source = r#"
namespace MyApp.Models
{
    public class User { }
}
        "#;

        let symbols = parse("test.cs", source).unwrap();

        let namespace_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Namespace))
            .collect();

        assert!(namespace_symbols.len() >= 1);
    }

    #[test]
    fn test_parse_file_scoped_namespace() {
        let source = r#"
namespace MyApp.Models;

public class User { }
        "#;

        let symbols = parse("test.cs", source).unwrap();

        let namespace_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Namespace))
            .collect();

        assert!(namespace_symbols.len() >= 1);
    }

    #[test]
    fn test_parse_interface() {
        let source = r#"
public interface IRepository
{
    void Save();
    void Delete();
}
        "#;

        let symbols = parse("test.cs", source).unwrap();

        let interface_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Interface))
            .collect();

        assert_eq!(interface_symbols.len(), 1);
        assert_eq!(interface_symbols[0].symbol.as_deref(), Some("IRepository"));
    }

    #[test]
    fn test_parse_struct() {
        let source = r#"
public struct Point
{
    public int X;
    public int Y;
}
        "#;

        let symbols = parse("test.cs", source).unwrap();

        let struct_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Struct))
            .collect();

        assert_eq!(struct_symbols.len(), 1);
        assert_eq!(struct_symbols[0].symbol.as_deref(), Some("Point"));
    }

    #[test]
    fn test_parse_enum() {
        let source = r#"
public enum Status
{
    Active,
    Inactive,
    Pending
}
        "#;

        let symbols = parse("test.cs", source).unwrap();

        let enum_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Enum))
            .collect();

        assert_eq!(enum_symbols.len(), 1);
        assert_eq!(enum_symbols[0].symbol.as_deref(), Some("Status"));
    }

    #[test]
    fn test_parse_record() {
        let source = r#"
public record Person(string FirstName, string LastName);
        "#;

        let symbols = parse("test.cs", source).unwrap();

        let record_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Type))
            .filter(|s| s.symbol.as_deref() == Some("Person"))
            .collect();

        assert_eq!(record_symbols.len(), 1);
    }

    #[test]
    fn test_parse_methods() {
        let source = r#"
public class Calculator
{
    public int Add(int a, int b)
    {
        return a + b;
    }

    public int Subtract(int a, int b)
    {
        return a - b;
    }
}
        "#;

        let symbols = parse("test.cs", source).unwrap();

        let method_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method))
            .collect();

        assert_eq!(method_symbols.len(), 2);
        assert!(method_symbols.iter().any(|s| s.symbol.as_deref() == Some("Add")));
        assert!(method_symbols.iter().any(|s| s.symbol.as_deref() == Some("Subtract")));

        // Check scope
        for method in method_symbols {
            // Removed: scope field no longer exists: assert_eq!(method.scope.as_ref().unwrap(), "class Calculator");
        }
    }

    #[test]
    fn test_parse_properties() {
        let source = r#"
public class User
{
    public string Name { get; set; }
    public int Age { get; set; }
    public string Email { get; init; }
}
        "#;

        let symbols = parse("test.cs", source).unwrap();

        let property_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        assert_eq!(property_symbols.len(), 3);
        assert!(property_symbols.iter().any(|s| s.symbol.as_deref() == Some("Name")));
        assert!(property_symbols.iter().any(|s| s.symbol.as_deref() == Some("Age")));
        assert!(property_symbols.iter().any(|s| s.symbol.as_deref() == Some("Email")));
    }

    #[test]
    fn test_parse_delegate() {
        let source = r#"
public delegate void EventHandler(object sender, EventArgs e);
        "#;

        let symbols = parse("test.cs", source).unwrap();

        let delegate_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Type))
            .filter(|s| s.symbol.as_deref() == Some("EventHandler"))
            .collect();

        assert_eq!(delegate_symbols.len(), 1);
    }

    #[test]
    fn test_parse_mixed_symbols() {
        let source = r#"
namespace MyApp
{
    public interface IService
    {
        void Execute();
    }

    public class Service : IService
    {
        public void Execute()
        {
            // Implementation
        }
    }

    public enum Priority
    {
        Low, Medium, High
    }
}
        "#;

        let symbols = parse("test.cs", source).unwrap();

        // Should find: namespace, interface, class, enum, method
        assert!(symbols.len() >= 5);

        let kinds: Vec<&SymbolKind> = symbols.iter().map(|s| &s.kind).collect();
        assert!(kinds.contains(&&SymbolKind::Namespace));
        assert!(kinds.contains(&&SymbolKind::Interface));
        assert!(kinds.contains(&&SymbolKind::Class));
        assert!(kinds.contains(&&SymbolKind::Enum));
        assert!(kinds.contains(&&SymbolKind::Method));
    }

    #[test]
    fn test_local_variables_included() {
        let source = r#"
public class Calculator
{
    public int Multiplier { get; set; } = 2;

    public int Compute(int input)
    {
        int localVar = input * Multiplier;
        var result = localVar + 10;
        return result;
    }
}

public class Helper
{
    public static string Format()
    {
        string message = "Hello";
        var count = 5;
        return message;
    }
}
        "#;

        let symbols = parse("test.cs", source).unwrap();

        // Filter to just variables
        let variables: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        // Check that local variables are captured
        assert!(variables.iter().any(|v| v.symbol.as_deref() == Some("localVar")));
        assert!(variables.iter().any(|v| v.symbol.as_deref() == Some("result")));
        assert!(variables.iter().any(|v| v.symbol.as_deref() == Some("message")));
        assert!(variables.iter().any(|v| v.symbol.as_deref() == Some("count")));

        // Check that class property is also captured
        assert!(variables.iter().any(|v| v.symbol.as_deref() == Some("Multiplier")));

        // Verify that local variables have no scope
        let local_vars: Vec<_> = variables.iter()
            .filter(|v| v.symbol.as_deref() == Some("localVar")
                     || v.symbol.as_deref() == Some("result")
                     || v.symbol.as_deref() == Some("message")
                     || v.symbol.as_deref() == Some("count"))
            .collect();

        for var in local_vars {
            // Removed: scope field no longer exists: assert_eq!(var.scope, None);
        }

        // Verify that class property has scope
        let property = variables.iter()
            .find(|v| v.symbol.as_deref() == Some("Multiplier"))
            .unwrap();
        // Removed: scope field no longer exists: assert_eq!(property.scope.as_ref().unwrap(), "class Calculator");
    }

    #[test]
    fn test_parse_events() {
        let source = r#"
public class Button
{
    public event EventHandler Click;
    public event Action Hover;
}

public interface INotifier
{
    event EventHandler<string> Notify;
}
        "#;

        let symbols = parse("test.cs", source).unwrap();

        let event_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Event))
            .collect();

        assert_eq!(event_symbols.len(), 3);
        assert!(event_symbols.iter().any(|s| s.symbol.as_deref() == Some("Click")));
        assert!(event_symbols.iter().any(|s| s.symbol.as_deref() == Some("Hover")));
        assert!(event_symbols.iter().any(|s| s.symbol.as_deref() == Some("Notify")));

        // Check scope
        let click_event = event_symbols.iter()
            .find(|s| s.symbol.as_deref() == Some("Click"))
            .unwrap();
        // Removed: scope field no longer exists: assert_eq!(click_event.scope.as_ref().unwrap(), "class Button");

        let notify_event = event_symbols.iter()
            .find(|s| s.symbol.as_deref() == Some("Notify"))
            .unwrap();
        // Removed: scope field no longer exists: assert_eq!(notify_event.scope.as_ref().unwrap(), "interface INotifier");
    }

    #[test]
    fn test_parse_indexers() {
        let source = r#"
public class StringCollection
{
    private string[] items = new string[100];

    public string this[int index]
    {
        get { return items[index]; }
        set { items[index] = value; }
    }
}

public struct Matrix
{
    public int this[int row, int col]
    {
        get { return 0; }
        set { }
    }
}
        "#;

        let symbols = parse("test.cs", source).unwrap();

        let indexer_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Property))
            .filter(|s| s.symbol.as_deref() == Some("this[]"))
            .collect();

        assert_eq!(indexer_symbols.len(), 2);

        // Note: scope field was removed from SearchResult for token optimization
        // Indexers are identified by SymbolKind::Property with symbol name "this[]"
    }

    #[test]
    fn test_parse_attribute_class() {
        let source = r#"
using System;

// Attribute with naming convention (ends with "Attribute")
public class TestAttribute : Attribute
{
    public string Name { get; set; }
    public TestAttribute(string name) { Name = name; }
}

// Attribute without suffix but inherits from Attribute
public class Obsolete : Attribute
{
    public string Message { get; set; }
}

// Not an attribute (regular class without "Attribute" suffix)
public class RegularClass
{
    public void DoSomething() { }
}

// Attribute with only suffix (no explicit inheritance)
public class CustomAttribute
{
    public int Value { get; set; }
}
        "#;

        let symbols = parse("test.cs", source).unwrap();

        let attribute_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Attribute))
            .collect();

        // Should find TestAttribute, Obsolete, and CustomAttribute
        assert_eq!(attribute_symbols.len(), 3);
        assert!(attribute_symbols.iter().any(|s| s.symbol.as_deref() == Some("TestAttribute")));
        assert!(attribute_symbols.iter().any(|s| s.symbol.as_deref() == Some("Obsolete")));
        assert!(attribute_symbols.iter().any(|s| s.symbol.as_deref() == Some("CustomAttribute")));

        // Should NOT find RegularClass
        assert!(!attribute_symbols.iter().any(|s| s.symbol.as_deref() == Some("RegularClass")));
    }

    #[test]
    fn test_parse_attribute_uses() {
        let source = r#"
using System;

public class TestAttribute : Attribute { }
public class ObsoleteAttribute : Attribute { }

[Test]
public class TestClass
{
    [Test]
    public void TestMethod1()
    {
        // Test code
    }

    [Test]
    [Obsolete]
    public void TestMethod2()
    {
        // Another test
    }
}

[Obsolete]
public class LegacyClass
{
    [Test]
    public void OldTest()
    {
        // Legacy test
    }
}
        "#;

        let symbols = parse("test.cs", source).unwrap();

        let attribute_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Attribute))
            .collect();

        // Should find attribute class definitions (TestAttribute, ObsoleteAttribute)
        // AND attribute uses (Test appears 4 times, Obsolete appears 2 times)
        // Total expected: 2 definitions + 6 uses = 8
        assert!(attribute_symbols.len() >= 6);

        // Count specific attribute uses
        let test_count = attribute_symbols.iter()
            .filter(|s| {
                let symbol = s.symbol.as_deref().unwrap_or("");
                symbol == "Test" || symbol == "TestAttribute"
            })
            .count();

        let obsolete_count = attribute_symbols.iter()
            .filter(|s| {
                let symbol = s.symbol.as_deref().unwrap_or("");
                symbol == "Obsolete" || symbol == "ObsoleteAttribute"
            })
            .count();

        // Should find Test/TestAttribute at least 4 times (1 definition + 4 uses)
        assert!(test_count >= 4);

        // Should find Obsolete/ObsoleteAttribute at least 3 times (1 definition + 2 uses)
        assert!(obsolete_count >= 3);
    }

    #[test]
    fn test_extract_csharp_usings() {
        let source = r#"
            using System;
            using System.Collections.Generic;
            using System.Linq;
            using Microsoft.AspNetCore.Mvc;

            namespace MyApp.Controllers
            {
                public class HomeController : Controller
                {
                    public IActionResult Index()
                    {
                        return View();
                    }
                }
            }
        "#;

        use crate::parsers::DependencyExtractor;

        let deps = CSharpDependencyExtractor::extract_dependencies(source).unwrap();

        assert_eq!(deps.len(), 4, "Should extract 4 using directives");
        assert!(deps.iter().any(|d| d.imported_path == "System"));
        assert!(deps.iter().any(|d| d.imported_path == "System.Collections.Generic"));
        assert!(deps.iter().any(|d| d.imported_path == "System.Linq"));
        assert!(deps.iter().any(|d| d.imported_path == "Microsoft.AspNetCore.Mvc"));

        for dep in &deps {
            assert!(matches!(dep.import_type, ImportType::Stdlib),
                    "System and Microsoft namespaces should be classified as Stdlib");
        }
    }
}

// ============================================================================
// Dependency Extraction
// ============================================================================

use crate::models::ImportType;
use crate::parsers::{DependencyExtractor, ImportInfo};

/// C# dependency extractor
pub struct CSharpDependencyExtractor;

impl DependencyExtractor for CSharpDependencyExtractor {
    fn extract_dependencies(source: &str) -> Result<Vec<ImportInfo>> {
        let mut parser = Parser::new();
        let language = tree_sitter_c_sharp::LANGUAGE;

        parser
            .set_language(&language.into())
            .context("Failed to set C# language")?;

        let tree = parser
            .parse(source, None)
            .context("Failed to parse C# source")?;

        let root_node = tree.root_node();

        let mut imports = Vec::new();

        // Extract using directives
        imports.extend(extract_csharp_usings(source, &root_node)?);

        Ok(imports)
    }
}

/// Extract C# using directives
fn extract_csharp_usings(
    source: &str,
    root: &tree_sitter::Node,
) -> Result<Vec<ImportInfo>> {
    let language = tree_sitter_c_sharp::LANGUAGE;

    let query_str = r#"
        (using_directive
            [
                (qualified_name) @using_path
                (identifier) @using_path
            ])
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create C# using query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut imports = Vec::new();

    while let Some(match_) = matches.next() {
        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            if capture_name == "using_path" {
                let path = capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
                let import_type = classify_csharp_using(&path);
                let line_number = capture.node.start_position().row + 1;

                imports.push(ImportInfo {
                    imported_path: path,
                    import_type,
                    line_number,
                    imported_symbols: None, // C# imports entire namespace
                });
            }
        }
    }

    Ok(imports)
}

/// Classify a C# using directive as internal, external, or stdlib
fn classify_csharp_using(using_path: &str) -> ImportType {
    // C# standard library namespaces (Microsoft-provided)
    const CSHARP_STDLIB_NAMESPACES: &[&str] = &[
        // Core system namespaces
        "System", "System.Collections", "System.Collections.Generic", "System.Collections.Concurrent",
        "System.Collections.Immutable", "System.Collections.ObjectModel", "System.Collections.Specialized",
        "System.ComponentModel", "System.ComponentModel.DataAnnotations",
        "System.Configuration", "System.Data", "System.Data.Common", "System.Data.SqlClient",
        "System.Diagnostics", "System.Diagnostics.CodeAnalysis", "System.Diagnostics.Contracts",
        "System.Drawing", "System.Globalization", "System.IO", "System.IO.Compression",
        "System.IO.Pipes", "System.Linq", "System.Linq.Expressions", "System.Net",
        "System.Net.Http", "System.Net.Mail", "System.Net.Sockets", "System.Numerics",
        "System.Reflection", "System.Reflection.Emit", "System.Resources", "System.Runtime",
        "System.Runtime.CompilerServices", "System.Runtime.InteropServices", "System.Runtime.Serialization",
        "System.Security", "System.Security.Cryptography", "System.Security.Principal",
        "System.Text", "System.Text.Json", "System.Text.RegularExpressions", "System.Threading",
        "System.Threading.Tasks", "System.Timers", "System.Xml", "System.Xml.Linq",
        "System.Xml.Serialization",

        // ASP.NET namespaces
        "Microsoft.AspNetCore", "Microsoft.AspNetCore.Builder", "Microsoft.AspNetCore.Hosting",
        "Microsoft.AspNetCore.Http", "Microsoft.AspNetCore.Mvc", "Microsoft.AspNetCore.Routing",
        "Microsoft.AspNetCore.Authentication", "Microsoft.AspNetCore.Authorization",

        // Entity Framework
        "Microsoft.EntityFrameworkCore", "Microsoft.EntityFrameworkCore.Design",
        "Microsoft.EntityFrameworkCore.Migrations",

        // Extensions
        "Microsoft.Extensions.Configuration", "Microsoft.Extensions.DependencyInjection",
        "Microsoft.Extensions.Hosting", "Microsoft.Extensions.Logging", "Microsoft.Extensions.Options",

        // Windows-specific
        "Microsoft.Win32", "System.Windows", "System.Windows.Forms", "System.Windows.Controls",
        "System.Windows.Data", "System.Windows.Input", "System.Windows.Media",

        // WPF
        "System.Xaml",

        // Older frameworks
        "System.Web", "System.Web.Mvc", "System.Web.Http",
    ];

    // Check if it's a standard library namespace or starts with one
    for stdlib_ns in CSHARP_STDLIB_NAMESPACES {
        if using_path == *stdlib_ns || using_path.starts_with(&format!("{}.", stdlib_ns)) {
            return ImportType::Stdlib;
        }
    }

    // Internal: anything that doesn't match stdlib patterns
    // In C#, project namespaces are typically named after the project/company
    // External packages usually have their own top-level namespace (e.g., Newtonsoft.Json)
    // We'll classify non-System/Microsoft namespaces as internal by default
    // (This means third-party packages will initially appear as internal,
    // but will be filtered out at indexing time if they're not in the project)
    ImportType::Internal
}

// ============================================================================
// Path Resolution
// ============================================================================

/// Resolve a C# using directive to a file path
///
/// # Arguments
/// * `using_path` - The namespace from the using directive (e.g., "MyApp.Models.User")
/// * `current_file_path` - Path to the file containing the using directive (unused for C#)
///
/// # Returns
/// * `Some(path)` if the namespace can be resolved to a file
/// * `None` if resolution fails
///
/// # Notes
/// C# namespace-to-file resolution follows these common patterns:
/// - `MyApp.Models.User` → `MyApp/Models/User.cs`
/// - `MyApp.Services.UserService` → `MyApp/Services/UserService.cs`
///
/// This resolver tries to convert namespace paths to file paths based on
/// C# naming conventions where namespace structure matches directory structure.
pub fn resolve_csharp_using_to_path(
    using_path: &str,
    _current_file_path: Option<&str>,
) -> Option<String> {
    // C# namespaces typically map to directory structure
    // Example: MyApp.Models.User → MyApp/Models/User.cs

    // Convert namespace separators to path separators
    let path_without_extension = using_path.replace('.', "/");

    // Try with .cs extension (most common)
    Some(format!("{}.cs", path_without_extension))
}

// ============================================================================
// Tests for Path Resolution
// ============================================================================

#[cfg(test)]
mod resolution_tests {
    use super::*;

    #[test]
    fn test_resolve_csharp_using_simple_namespace() {
        let result = resolve_csharp_using_to_path(
            "MyApp.Models.User",
            None,
        );

        assert_eq!(result, Some("MyApp/Models/User.cs".to_string()));
    }

    #[test]
    fn test_resolve_csharp_using_services() {
        let result = resolve_csharp_using_to_path(
            "MyApp.Services.UserService",
            None,
        );

        assert_eq!(result, Some("MyApp/Services/UserService.cs".to_string()));
    }

    #[test]
    fn test_resolve_csharp_using_single_level() {
        let result = resolve_csharp_using_to_path(
            "MyApp",
            None,
        );

        assert_eq!(result, Some("MyApp.cs".to_string()));
    }

    #[test]
    fn test_resolve_csharp_using_deep_namespace() {
        let result = resolve_csharp_using_to_path(
            "MyApp.Core.Domain.Models.User",
            None,
        );

        assert_eq!(result, Some("MyApp/Core/Domain/Models/User.cs".to_string()));
    }
}
