//! Java language parser using Tree-sitter
//!
//! Extracts symbols from Java source code:
//! - Classes (regular, abstract, final)
//! - Interfaces
//! - Enums
//! - Records (Java 14+)
//! - Methods (with class scope, visibility)
//! - Fields (public, private, protected, static)
//! - Constructors
//! - Annotations
//! - Local variables (inside method bodies)

use crate::models::{Language, SearchResult, Span, SymbolKind};
use anyhow::{Context, Result};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};

/// Parse Java source code and extract symbols
pub fn parse(path: &str, source: &str) -> Result<Vec<SearchResult>> {
    let mut parser = Parser::new();
    let language = tree_sitter_java::LANGUAGE;

    parser
        .set_language(&language.into())
        .context("Failed to set Java language")?;

    let tree = parser
        .parse(source, None)
        .context("Failed to parse Java source")?;

    let root_node = tree.root_node();

    let mut symbols = Vec::new();

    // Extract different types of symbols using Tree-sitter queries
    symbols.extend(extract_classes(source, &root_node, &language.into())?);
    symbols.extend(extract_interfaces(source, &root_node, &language.into())?);
    symbols.extend(extract_enums(source, &root_node, &language.into())?);
    symbols.extend(extract_annotations(source, &root_node, &language.into())?);
    symbols.extend(extract_class_methods(source, &root_node, &language.into())?);
    symbols.extend(extract_interface_methods(
        source,
        &root_node,
        &language.into(),
    )?);
    symbols.extend(extract_fields(source, &root_node, &language.into())?);
    symbols.extend(extract_constructors(source, &root_node, &language.into())?);
    symbols.extend(extract_local_variables(
        source,
        &root_node,
        &language.into(),
    )?);

    // Add file path to all symbols
    for symbol in &mut symbols {
        symbol.path = path.to_string();
        symbol.lang = Language::Java;
    }

    Ok(symbols)
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

    let query = Query::new(language, query_str).context("Failed to create class query")?;

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

    let query = Query::new(language, query_str).context("Failed to create interface query")?;

    extract_symbols(source, root, &query, SymbolKind::Interface, None)
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

    let query = Query::new(language, query_str).context("Failed to create enum query")?;

    extract_symbols(source, root, &query, SymbolKind::Enum, None)
}

/// Extract annotations: BOTH definitions and uses
/// Definitions: @interface Test { ... }
/// Uses: @Test public void testMethod()
fn extract_annotations(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let mut symbols = Vec::new();

    // Part 1: Extract annotation type DEFINITIONS (@interface)
    let def_query_str = r#"
        (annotation_type_declaration
            name: (identifier) @name) @annotation
    "#;

    let def_query = Query::new(language, def_query_str)
        .context("Failed to create annotation definition query")?;

    symbols.extend(extract_symbols(
        source,
        root,
        &def_query,
        SymbolKind::Attribute,
        None,
    )?);

    // Part 2: Extract annotation USES (@Test, @Override, etc.)
    let use_query_str = r#"
        (marker_annotation
            name: (identifier) @name) @annotation

        (annotation
            name: (identifier) @name) @annotation
    "#;

    let use_query =
        Query::new(language, use_query_str).context("Failed to create annotation use query")?;

    symbols.extend(extract_symbols(
        source,
        root,
        &use_query,
        SymbolKind::Attribute,
        None,
    )?);

    Ok(symbols)
}

/// Extract method declarations from classes
fn extract_class_methods(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_declaration
            name: (identifier) @class_name
            body: (class_body
                (method_declaration
                    name: (identifier) @method_name))) @class

        (enum_declaration
            name: (identifier) @enum_name
            body: (enum_body
                (enum_body_declarations
                    (method_declaration
                        name: (identifier) @method_name)))) @enum
    "#;

    let query = Query::new(language, query_str).context("Failed to create method query")?;

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
                    scope_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                    scope_type = Some("class");
                }
                "enum_name" => {
                    scope_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                    scope_type = Some("enum");
                }
                "method_name" => {
                    method_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
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
            (scope_name, scope_type, method_name, method_node)
        {
            let scope = format!("{} {}", scope_type, scope_name);
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::Java,
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

/// Extract field declarations from classes
fn extract_fields(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_declaration
            name: (identifier) @class_name
            body: (class_body
                (field_declaration
                    declarator: (variable_declarator
                        name: (identifier) @field_name)))) @class

        (enum_declaration
            name: (identifier) @enum_name
            body: (enum_body
                (enum_body_declarations
                    (field_declaration
                        declarator: (variable_declarator
                            name: (identifier) @field_name))))) @enum
    "#;

    let query = Query::new(language, query_str).context("Failed to create field query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut scope_name = None;
        let mut scope_type = None;
        let mut field_name = None;
        let mut field_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            match capture_name {
                "class_name" => {
                    scope_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                    scope_type = Some("class");
                }
                "enum_name" => {
                    scope_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                    scope_type = Some("enum");
                }
                "field_name" => {
                    field_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                    // Find the parent field_declaration node
                    let mut current = capture.node;
                    while let Some(parent) = current.parent() {
                        if parent.kind() == "field_declaration" {
                            field_node = Some(parent);
                            break;
                        }
                        current = parent;
                    }
                }
                _ => {}
            }
        }

        if let (Some(scope_name), Some(scope_type), Some(field_name), Some(node)) =
            (scope_name, scope_type, field_name, field_node)
        {
            let scope = format!("{} {}", scope_type, scope_name);
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::Java,
                SymbolKind::Variable,
                Some(field_name),
                span,
                Some(scope),
                preview,
            ));
        }
    }

    Ok(symbols)
}

/// Extract constructor declarations
fn extract_constructors(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_declaration
            name: (identifier) @class_name
            body: (class_body
                (constructor_declaration
                    name: (identifier) @constructor_name))) @class
    "#;

    let query = Query::new(language, query_str).context("Failed to create constructor query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut class_name = None;
        let mut constructor_name = None;
        let mut constructor_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            match capture_name {
                "class_name" => {
                    class_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                }
                "constructor_name" => {
                    constructor_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                    // Find the parent constructor_declaration node
                    let mut current = capture.node;
                    while let Some(parent) = current.parent() {
                        if parent.kind() == "constructor_declaration" {
                            constructor_node = Some(parent);
                            break;
                        }
                        current = parent;
                    }
                }
                _ => {}
            }
        }

        if let (Some(class_name), Some(constructor_name), Some(node)) =
            (class_name, constructor_name, constructor_node)
        {
            let scope = format!("class {}", class_name);
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::Java,
                SymbolKind::Method,
                Some(constructor_name),
                span,
                Some(scope),
                preview,
            ));
        }
    }

    Ok(symbols)
}

/// Extract method declarations from interfaces
fn extract_interface_methods(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (interface_declaration
            name: (identifier) @interface_name
            body: (interface_body
                (method_declaration
                    name: (identifier) @method_name))) @interface
    "#;

    let query =
        Query::new(language, query_str).context("Failed to create interface method query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut interface_name = None;
        let mut method_name = None;
        let mut method_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            match capture_name {
                "interface_name" => {
                    interface_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                }
                "method_name" => {
                    method_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
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

        if let (Some(interface_name), Some(method_name), Some(node)) =
            (interface_name, method_name, method_node)
        {
            let scope = format!("interface {}", interface_name);
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::Java,
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

/// Extract local variable declarations from method bodies
fn extract_local_variables(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (local_variable_declaration
            declarator: (variable_declarator
                name: (identifier) @name)) @var
    "#;

    let query = Query::new(language, query_str).context("Failed to create local variable query")?;

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
                Language::Java,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_class() {
        let source = r#"
public class User {
    private String name;
    private int age;
}
        "#;

        let symbols = parse("test.java", source).unwrap();

        let class_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        assert_eq!(class_symbols.len(), 1);
        assert_eq!(class_symbols[0].symbol.as_deref(), Some("User"));
    }

    #[test]
    fn test_parse_class_with_methods() {
        let source = r#"
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int subtract(int a, int b) {
        return a - b;
    }
}
        "#;

        let symbols = parse("test.java", source).unwrap();

        let method_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method))
            .collect();

        assert_eq!(method_symbols.len(), 2);
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("add"))
        );
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("subtract"))
        );

        // Check scope
        for method in method_symbols {
            // Removed: scope field no longer exists: assert_eq!(method.scope.as_ref().unwrap(), "class Calculator");
        }
    }

    #[test]
    fn test_parse_interface() {
        let source = r#"
public interface Drawable {
    void draw();
    void resize(int width, int height);
}
        "#;

        let symbols = parse("test.java", source).unwrap();

        let interface_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Interface))
            .collect();

        assert_eq!(interface_symbols.len(), 1);
        assert_eq!(interface_symbols[0].symbol.as_deref(), Some("Drawable"));
    }

    #[test]
    fn test_parse_enum() {
        let source = r#"
public enum Status {
    ACTIVE,
    INACTIVE,
    PENDING
}
        "#;

        let symbols = parse("test.java", source).unwrap();

        let enum_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Enum))
            .collect();

        assert_eq!(enum_symbols.len(), 1);
        assert_eq!(enum_symbols[0].symbol.as_deref(), Some("Status"));
    }

    #[test]
    fn test_parse_fields() {
        let source = r#"
public class Config {
    private static final int MAX_SIZE = 100;
    private String hostname;
    public int port;
}
        "#;

        let symbols = parse("test.java", source).unwrap();

        let field_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        assert_eq!(field_symbols.len(), 3);
        assert!(
            field_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("MAX_SIZE"))
        );
        assert!(
            field_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("hostname"))
        );
        assert!(
            field_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("port"))
        );
    }

    #[test]
    fn test_parse_constructor() {
        let source = r#"
public class User {
    private String name;

    public User(String name) {
        this.name = name;
    }

    public User() {
        this("Anonymous");
    }
}
        "#;

        let symbols = parse("test.java", source).unwrap();

        let constructor_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method) && s.symbol.as_deref() == Some("User"))
            .collect();

        assert_eq!(constructor_symbols.len(), 2);
    }

    #[test]
    fn test_parse_abstract_class() {
        let source = r#"
public abstract class Animal {
    protected String name;

    public abstract void makeSound();

    public void sleep() {
        System.out.println("Sleeping...");
    }
}
        "#;

        let symbols = parse("test.java", source).unwrap();

        let class_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        assert_eq!(class_symbols.len(), 1);
        assert_eq!(class_symbols[0].symbol.as_deref(), Some("Animal"));

        let method_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method))
            .collect();

        assert_eq!(method_symbols.len(), 2);
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("makeSound"))
        );
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("sleep"))
        );
    }

    #[test]
    fn test_parse_nested_class() {
        let source = r#"
public class Outer {
    private int outerField;

    public static class Nested {
        private int nestedField;

        public void nestedMethod() {
            // ...
        }
    }

    public void outerMethod() {
        // ...
    }
}
        "#;

        let symbols = parse("test.java", source).unwrap();

        let class_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        assert_eq!(class_symbols.len(), 2);
        assert!(
            class_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Outer"))
        );
        assert!(
            class_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Nested"))
        );
    }

    #[test]
    fn test_parse_interface_with_methods() {
        let source = r#"
public interface Repository<T> {
    T findById(Long id);
    List<T> findAll();
    void save(T entity);
    void delete(T entity);
}
        "#;

        let symbols = parse("test.java", source).unwrap();

        let interface_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Interface))
            .collect();

        assert_eq!(interface_symbols.len(), 1);

        let method_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method))
            .collect();

        assert_eq!(method_symbols.len(), 4);
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("findById"))
        );
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("findAll"))
        );
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("save"))
        );
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("delete"))
        );
    }

    #[test]
    fn test_parse_enum_with_methods() {
        let source = r#"
public enum Day {
    MONDAY, TUESDAY, WEDNESDAY;

    public boolean isWeekend() {
        return this == SATURDAY || this == SUNDAY;
    }
}
        "#;

        let symbols = parse("test.java", source).unwrap();

        let enum_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Enum))
            .collect();

        assert_eq!(enum_symbols.len(), 1);

        let method_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method))
            .collect();

        assert_eq!(method_symbols.len(), 1);
        assert_eq!(method_symbols[0].symbol.as_deref(), Some("isWeekend"));
    }

    #[test]
    fn test_parse_mixed_symbols() {
        let source = r#"
package com.example;

public interface UserService {
    User findUser(Long id);
}

public class User {
    private Long id;
    private String name;

    public User(Long id, String name) {
        this.id = id;
        this.name = name;
    }

    public String getName() {
        return name;
    }
}

public enum UserRole {
    ADMIN, USER, GUEST
}
        "#;

        let symbols = parse("test.java", source).unwrap();

        // Should find: interface, class, enum, fields, constructor, methods
        assert!(symbols.len() >= 7);

        let kinds: Vec<&SymbolKind> = symbols.iter().map(|s| &s.kind).collect();
        assert!(kinds.contains(&&SymbolKind::Interface));
        assert!(kinds.contains(&&SymbolKind::Class));
        assert!(kinds.contains(&&SymbolKind::Enum));
        assert!(kinds.contains(&&SymbolKind::Variable));
        assert!(kinds.contains(&&SymbolKind::Method));
    }

    #[test]
    fn test_parse_generic_class() {
        let source = r#"
public class Container<T> {
    private T value;

    public Container(T value) {
        this.value = value;
    }

    public T getValue() {
        return value;
    }

    public void setValue(T value) {
        this.value = value;
    }
}
        "#;

        let symbols = parse("test.java", source).unwrap();

        let class_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        assert_eq!(class_symbols.len(), 1);
        assert_eq!(class_symbols[0].symbol.as_deref(), Some("Container"));

        let method_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method))
            .collect();

        assert!(method_symbols.len() >= 3);
    }

    #[test]
    fn test_local_variables_included() {
        let source = r#"
public class Calculator {
    private int globalCount = 10;

    public int calculate(int x) {
        int localVar = x * 2;
        int anotherLocal = 5;
        return localVar + anotherLocal + globalCount;
    }
}
        "#;

        let symbols = parse("test.java", source).unwrap();

        let var_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        // Should find both field (globalCount) and local variables (localVar, anotherLocal)
        assert_eq!(var_symbols.len(), 3);
        assert!(
            var_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("globalCount"))
        );
        assert!(
            var_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("localVar"))
        );
        assert!(
            var_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("anotherLocal"))
        );

        // Check scopes: field should have scope, local vars should not
        let global_count = var_symbols
            .iter()
            .find(|s| s.symbol.as_deref() == Some("globalCount"))
            .unwrap();
        // Removed: scope field no longer exists: assert_eq!(global_count.scope.as_ref().unwrap(), "class Calculator");

        let local_var = var_symbols
            .iter()
            .find(|s| s.symbol.as_deref() == Some("localVar"))
            .unwrap();
        // Removed: scope field no longer exists: assert_eq!(local_var.scope, None);
    }

    #[test]
    fn test_parse_annotation_type() {
        let source = r#"
public @interface Test {
}

@interface Author {
    String name();
    String date();
}

@interface Retention {
    RetentionPolicy value();
}
        "#;

        let symbols = parse("test.java", source).unwrap();

        let annotation_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Attribute))
            .collect();

        // Should find annotation definitions
        assert!(
            annotation_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Test"))
        );
        assert!(
            annotation_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Author"))
        );
        assert!(
            annotation_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Retention"))
        );
    }

    #[test]
    fn test_parse_annotation_uses() {
        let source = r#"
@Test
public void testMethod() {
    assertEquals(1, 1);
}

@Override
@Deprecated
public String toString() {
    return "example";
}

@SuppressWarnings("unchecked")
public class MyClass {
    @Autowired
    private Service service;

    @Test
    @DisplayName("Should work")
    public void anotherTest() {}
}
        "#;

        let symbols = parse("test.java", source).unwrap();

        let annotation_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Attribute))
            .collect();

        // Should find annotation uses
        assert!(
            annotation_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Test"))
        );
        assert!(
            annotation_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Override"))
        );
        assert!(
            annotation_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Deprecated"))
        );
        assert!(
            annotation_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("SuppressWarnings"))
        );
        assert!(
            annotation_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Autowired"))
        );
        assert!(
            annotation_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("DisplayName"))
        );

        // Should find Test twice (2 uses)
        let test_count = annotation_symbols
            .iter()
            .filter(|s| s.symbol.as_deref() == Some("Test"))
            .count();
        assert_eq!(test_count, 2);
    }

    #[test]
    fn test_extract_java_imports() {
        let source = r#"
            import java.util.List;
            import java.util.ArrayList;
            import java.io.IOException;
            import org.springframework.stereotype.Service;

            @Service
            public class UserService {
                private List<String> users = new ArrayList<>();

                public void addUser(String name) throws IOException {
                    users.add(name);
                }
            }
        "#;

        use crate::parsers::DependencyExtractor;

        let deps = JavaDependencyExtractor::extract_dependencies(source).unwrap();

        assert_eq!(deps.len(), 4, "Should extract 4 import statements");
        assert!(deps.iter().any(|d| d.imported_path == "java.util.List"));
        assert!(
            deps.iter()
                .any(|d| d.imported_path == "java.util.ArrayList")
        );
        assert!(
            deps.iter()
                .any(|d| d.imported_path == "java.io.IOException")
        );
        assert!(
            deps.iter()
                .any(|d| d.imported_path == "org.springframework.stereotype.Service")
        );

        // Check stdlib classification
        let java_util_list = deps
            .iter()
            .find(|d| d.imported_path == "java.util.List")
            .unwrap();
        assert!(
            matches!(java_util_list.import_type, ImportType::Stdlib),
            "java.util imports should be classified as Stdlib"
        );

        // Check external classification
        let spring_service = deps
            .iter()
            .find(|d| d.imported_path == "org.springframework.stereotype.Service")
            .unwrap();
        assert!(
            matches!(spring_service.import_type, ImportType::External),
            "org.springframework imports should be classified as External"
        );
    }
}

// ============================================================================
// Dependency Extraction
// ============================================================================

use crate::models::ImportType;
use crate::parsers::{DependencyExtractor, ImportInfo};

/// Java dependency extractor
pub struct JavaDependencyExtractor;

impl DependencyExtractor for JavaDependencyExtractor {
    fn extract_dependencies(source: &str) -> Result<Vec<ImportInfo>> {
        let mut parser = Parser::new();
        let language = tree_sitter_java::LANGUAGE;

        parser
            .set_language(&language.into())
            .context("Failed to set Java language")?;

        let tree = parser
            .parse(source, None)
            .context("Failed to parse Java source")?;

        let root_node = tree.root_node();

        let mut imports = Vec::new();

        // Extract import statements
        imports.extend(extract_java_imports(source, &root_node)?);

        Ok(imports)
    }
}

/// Extract Java import statements
fn extract_java_imports(source: &str, root: &tree_sitter::Node) -> Result<Vec<ImportInfo>> {
    let language = tree_sitter_java::LANGUAGE;

    let query_str = r#"
        (import_declaration
            [
                (scoped_identifier) @import_path
                (identifier) @import_path
            ])
    "#;

    let query =
        Query::new(&language.into(), query_str).context("Failed to create Java import query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut imports = Vec::new();

    while let Some(match_) = matches.next() {
        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            if capture_name == "import_path" {
                let path = capture
                    .node
                    .utf8_text(source.as_bytes())
                    .unwrap_or("")
                    .to_string();
                let import_type = classify_java_import(&path);
                let line_number = capture.node.start_position().row + 1;

                imports.push(ImportInfo {
                    imported_path: path,
                    import_type,
                    line_number,
                    imported_symbols: None, // Java imports are package-level
                });
            }
        }
    }

    Ok(imports)
}

/// Classify a Java import as internal, external, or stdlib
fn classify_java_import(import_path: &str) -> ImportType {
    classify_java_import_impl(import_path, None)
}

/// Parse pom.xml or build.gradle to find Java package name
/// Similar to find_go_module_name() for Go projects
pub fn find_java_package_name(root: &std::path::Path) -> Option<String> {
    // Try Maven first (pom.xml)
    if let Some(package) = find_maven_package(root) {
        return Some(package);
    }

    // Try Gradle second (build.gradle or build.gradle.kts)
    if let Some(package) = find_gradle_package(root) {
        return Some(package);
    }

    // Fallback: scan package declarations in .java files
    find_package_from_sources(root)
}

/// Parse pom.xml to extract <groupId>
fn find_maven_package(root: &std::path::Path) -> Option<String> {
    let pom_path = root.join("pom.xml");
    if !pom_path.exists() {
        return None;
    }

    let content = std::fs::read_to_string(&pom_path).ok()?;

    // Simple XML parsing for <groupId>
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("<groupId>") && trimmed.ends_with("</groupId>") {
            let start = "<groupId>".len();
            let end = trimmed.len() - "</groupId>".len();
            return Some(trimmed[start..end].to_string());
        }
    }

    None
}

/// Parse build.gradle or build.gradle.kts to extract group
fn find_gradle_package(root: &std::path::Path) -> Option<String> {
    // Try build.gradle (Groovy)
    if let Some(package) = find_gradle_package_in_file(&root.join("build.gradle")) {
        return Some(package);
    }

    // Try build.gradle.kts (Kotlin)
    find_gradle_package_in_file(&root.join("build.gradle.kts"))
}

fn find_gradle_package_in_file(gradle_path: &std::path::Path) -> Option<String> {
    if !gradle_path.exists() {
        return None;
    }

    let content = std::fs::read_to_string(gradle_path).ok()?;

    for line in content.lines() {
        let trimmed = line.trim();

        // Groovy: group = 'org.neo4j'
        // Kotlin: group = "org.neo4j"
        if trimmed.starts_with("group") {
            if let Some(equals_idx) = trimmed.find('=') {
                let value = &trimmed[equals_idx + 1..].trim();
                // Remove quotes
                let value = value.trim_matches(|c| c == '\'' || c == '"');
                return Some(value.to_string());
            }
        }
    }

    None
}

/// Scan .java files to find common package prefix
fn find_package_from_sources(root: &std::path::Path) -> Option<String> {
    use std::collections::HashMap;

    let mut package_counts: HashMap<String, usize> = HashMap::new();

    // Walk the directory tree looking for .java files
    fn walk_dir(dir: &std::path::Path, package_counts: &mut HashMap<String, usize>, depth: usize) {
        // Limit depth to avoid excessive scanning
        if depth > 10 {
            return;
        }

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_dir() {
                walk_dir(&path, package_counts, depth + 1);
            } else if path.extension().and_then(|s| s.to_str()) == Some("java") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    // Extract package declaration
                    for line in content.lines().take(20) {
                        // Check first 20 lines
                        let trimmed = line.trim();
                        if trimmed.starts_with("package ") && trimmed.ends_with(';') {
                            let package = &trimmed[8..trimmed.len() - 1].trim();

                            // Extract base package (first 2 components: org.neo4j)
                            let parts: Vec<&str> = package.split('.').collect();
                            if parts.len() >= 2 {
                                let base_package = format!("{}.{}", parts[0], parts[1]);
                                *package_counts.entry(base_package).or_insert(0) += 1;
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    walk_dir(root, &mut package_counts, 0);

    // Find the most common package prefix
    package_counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(package, _)| package)
}

/// Reclassify a Java import using the project package prefix
/// Similar to reclassify_go_import() for Go
pub fn reclassify_java_import(import_path: &str, package_prefix: Option<&str>) -> ImportType {
    classify_java_import_impl(import_path, package_prefix)
}

fn classify_java_import_impl(import_path: &str, package_prefix: Option<&str>) -> ImportType {
    // First check if this is an internal import (matches project package)
    if let Some(prefix) = package_prefix {
        if import_path.starts_with(prefix) {
            return ImportType::Internal;
        }
    }

    // Java standard library packages (common ones)
    const STDLIB_PACKAGES: &[&str] = &[
        "java.lang",
        "java.util",
        "java.io",
        "java.nio",
        "java.net",
        "java.text",
        "java.math",
        "java.time",
        "java.sql",
        "java.security",
        "java.awt",
        "java.swing",
        "javax.swing",
        "javax.sql",
        "javax.crypto",
        "javax.net",
        "javax.xml",
        "javax.annotation",
        "javax.servlet",
        "org.w3c.dom",
        "org.xml.sax",
    ];

    // Check if it starts with any stdlib package
    for stdlib_pkg in STDLIB_PACKAGES {
        if import_path.starts_with(stdlib_pkg) {
            return ImportType::Stdlib;
        }
    }

    // Everything else is external
    ImportType::External
}

// ============================================================================
// Monorepo Support - Java/Kotlin Dependency Resolution
// ============================================================================

/// Represents a Java/Kotlin project in a monorepo
#[derive(Debug, Clone)]
pub struct JavaProject {
    /// Package name (groupId from Maven or group from Gradle)
    pub package_name: String,
    /// Relative path to project root (where pom.xml or build.gradle is)
    pub project_root: String,
    /// Absolute path to project root
    pub abs_project_root: String,
}

/// Find all Maven/Gradle projects in the repository recursively
/// Similar to find_all_go_mods() for Go
pub fn find_all_maven_gradle_projects(root: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
    let mut config_files = Vec::new();

    let walker = ignore::WalkBuilder::new(root)
        .follow_links(false)
        .git_ignore(true)
        .build();

    for entry in walker {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

            // Match pom.xml (Maven) or build.gradle/build.gradle.kts (Gradle)
            if filename == "pom.xml" || filename == "build.gradle" || filename == "build.gradle.kts"
            {
                config_files.push(path.to_path_buf());
                log::trace!("Found Java/Kotlin config: {}", path.display());
            }
        }
    }

    log::debug!(
        "Found {} Java/Kotlin project config files",
        config_files.len()
    );
    Ok(config_files)
}

/// Parse all Maven/Gradle projects and return JavaProject structs
/// Similar to parse_all_go_modules() for Go
pub fn parse_all_java_projects(root: &std::path::Path) -> Result<Vec<JavaProject>> {
    let config_files = find_all_maven_gradle_projects(root)?;
    let mut projects = Vec::new();

    let root_abs = root
        .canonicalize()
        .with_context(|| format!("Failed to canonicalize root path: {}", root.display()))?;

    for config_path in &config_files {
        // Get the directory containing the config file (project root)
        if let Some(project_dir) = config_path.parent() {
            // Parse the config file to get package name
            if let Some(package_name) = extract_package_from_config(config_path) {
                let project_abs = project_dir.canonicalize().with_context(|| {
                    format!(
                        "Failed to canonicalize project path: {}",
                        project_dir.display()
                    )
                })?;

                let project_rel = project_abs
                    .strip_prefix(&root_abs)
                    .unwrap_or(project_dir)
                    .to_string_lossy()
                    .to_string();

                projects.push(JavaProject {
                    package_name: package_name.clone(),
                    project_root: project_rel,
                    abs_project_root: project_abs.to_string_lossy().to_string(),
                });

                log::trace!(
                    "Parsed Java/Kotlin project: {} at {}",
                    package_name,
                    project_dir.display()
                );
            }
        }
    }

    log::info!("Parsed {} Java/Kotlin projects", projects.len());
    Ok(projects)
}

/// Extract package name from pom.xml or build.gradle
fn extract_package_from_config(config_path: &std::path::Path) -> Option<String> {
    let filename = config_path.file_name()?.to_str()?;

    match filename {
        "pom.xml" => {
            // Extract <groupId> from Maven pom.xml
            let content = std::fs::read_to_string(config_path).ok()?;
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("<groupId>") && trimmed.ends_with("</groupId>") {
                    let start = "<groupId>".len();
                    let end = trimmed.len() - "</groupId>".len();
                    return Some(trimmed[start..end].to_string());
                }
            }
            None
        }
        "build.gradle" | "build.gradle.kts" => {
            // Extract group from Gradle build file
            let content = std::fs::read_to_string(config_path).ok()?;
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("group") {
                    if let Some(equals_idx) = trimmed.find('=') {
                        let value = &trimmed[equals_idx + 1..].trim();
                        let value = value.trim_matches(|c| c == '\'' || c == '"');
                        return Some(value.to_string());
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Resolve a Java import to a file path
///
/// Java imports look like: `com.example.myapp.UserService`
/// Files are located at: `src/main/java/com/example/myapp/UserService.java`
/// or: `src/com/example/myapp/UserService.java`
pub fn resolve_java_import_to_path(
    import_path: &str,
    projects: &[JavaProject],
    _current_file_path: Option<&str>,
) -> Option<String> {
    // Java imports are absolute package paths, not relative
    // Find which project this import belongs to
    for project in projects {
        if import_path.starts_with(&project.package_name) {
            // Convert package to file path: com.example.UserService → com/example/UserService.java
            let file_path = import_path.replace('.', "/");

            // Try common Java source directory structures
            let candidates = vec![
                // Maven/Gradle standard structure
                format!("{}/src/main/java/{}.java", project.project_root, file_path),
                // Simpler structure
                format!("{}/src/{}.java", project.project_root, file_path),
                // Root-level src
                format!("{}/{}.java", project.project_root, file_path),
            ];

            for candidate in candidates {
                log::trace!("Checking Java import path: {}", candidate);
                return Some(candidate);
            }
        }
    }

    None
}

/// Resolve a Kotlin import to a file path
///
/// Kotlin uses the same package system as Java, but with .kt extension
pub fn resolve_kotlin_import_to_path(
    import_path: &str,
    projects: &[JavaProject],
    _current_file_path: Option<&str>,
) -> Option<String> {
    // Kotlin imports are identical to Java imports
    for project in projects {
        if import_path.starts_with(&project.package_name) {
            let file_path = import_path.replace('.', "/");

            // Try common Kotlin source directory structures
            let candidates = vec![
                // Maven/Gradle standard structure
                format!("{}/src/main/kotlin/{}.kt", project.project_root, file_path),
                // Java source dir (Kotlin can be in java dir)
                format!("{}/src/main/java/{}.kt", project.project_root, file_path),
                // Simpler structure
                format!("{}/src/{}.kt", project.project_root, file_path),
                // Root-level src
                format!("{}/{}.kt", project.project_root, file_path),
            ];

            for candidate in candidates {
                log::trace!("Checking Kotlin import path: {}", candidate);
                return Some(candidate);
            }
        }
    }

    None
}

#[cfg(test)]
mod monorepo_tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_resolve_java_import_maven_structure() {
        let projects = vec![JavaProject {
            package_name: "com.example".to_string(),
            project_root: "project1".to_string(),
            abs_project_root: "/abs/project1".to_string(),
        }];

        let resolved = resolve_java_import_to_path("com.example.UserService", &projects, None);

        assert!(resolved.is_some());
        let path = resolved.unwrap();
        // Should try Maven standard structure first
        assert!(path.contains("src/main/java/com/example/UserService.java"));
    }

    #[test]
    fn test_resolve_kotlin_import() {
        let projects = vec![JavaProject {
            package_name: "org.acme".to_string(),
            project_root: "kotlin-project".to_string(),
            abs_project_root: "/abs/kotlin-project".to_string(),
        }];

        let resolved = resolve_kotlin_import_to_path("org.acme.Repository", &projects, None);

        assert!(resolved.is_some());
        let path = resolved.unwrap();
        assert!(path.contains("src/main/kotlin/org/acme/Repository.kt"));
    }

    #[test]
    fn test_resolve_java_import_no_match() {
        let projects = vec![JavaProject {
            package_name: "com.example".to_string(),
            project_root: "project1".to_string(),
            abs_project_root: "/abs/project1".to_string(),
        }];

        // Different package
        let resolved = resolve_java_import_to_path("org.other.Service", &projects, None);

        assert!(resolved.is_none());
    }

    #[test]
    fn test_resolve_java_import_monorepo() {
        let projects = vec![
            JavaProject {
                package_name: "com.example.service1".to_string(),
                project_root: "services/service1".to_string(),
                abs_project_root: "/abs/services/service1".to_string(),
            },
            JavaProject {
                package_name: "com.example.service2".to_string(),
                project_root: "services/service2".to_string(),
                abs_project_root: "/abs/services/service2".to_string(),
            },
        ];

        // Should resolve to service1
        let resolved1 =
            resolve_java_import_to_path("com.example.service1.UserController", &projects, None);
        assert!(resolved1.is_some());
        assert!(resolved1.unwrap().contains("services/service1"));

        // Should resolve to service2
        let resolved2 =
            resolve_java_import_to_path("com.example.service2.ProductController", &projects, None);
        assert!(resolved2.is_some());
        assert!(resolved2.unwrap().contains("services/service2"));
    }

    #[test]
    fn test_extract_package_from_pom_xml() {
        let temp = TempDir::new().unwrap();
        let pom_path = temp.path().join("pom.xml");

        fs::write(
            &pom_path,
            r#"
<?xml version="1.0" encoding="UTF-8"?>
<project>
    <groupId>com.example.myapp</groupId>
    <artifactId>my-application</artifactId>
</project>
        "#,
        )
        .unwrap();

        let package = extract_package_from_config(&pom_path);
        assert_eq!(package, Some("com.example.myapp".to_string()));
    }

    #[test]
    fn test_extract_package_from_gradle() {
        let temp = TempDir::new().unwrap();
        let gradle_path = temp.path().join("build.gradle");

        fs::write(
            &gradle_path,
            r#"
group = 'org.example.myproject'
version = '1.0.0'
        "#,
        )
        .unwrap();

        let package = extract_package_from_config(&gradle_path);
        assert_eq!(package, Some("org.example.myproject".to_string()));
    }

    #[test]
    fn test_extract_package_from_gradle_kts() {
        let temp = TempDir::new().unwrap();
        let gradle_path = temp.path().join("build.gradle.kts");

        fs::write(
            &gradle_path,
            r#"
group = "com.acme.tools"
version = "2.0.0"
        "#,
        )
        .unwrap();

        let package = extract_package_from_config(&gradle_path);
        assert_eq!(package, Some("com.acme.tools".to_string()));
    }
}
