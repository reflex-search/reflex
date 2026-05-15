//! PHP language parser using Tree-sitter
//!
//! Extracts symbols from PHP source code:
//! - Functions
//! - Classes (regular, abstract, final)
//! - Interfaces
//! - Traits
//! - Methods (with class/trait scope)
//! - Properties (public, protected, private)
//! - Local variables ($var inside functions)
//! - Constants (class and global)
//! - Namespaces
//! - Enums (PHP 8.1+)

use crate::models::{Language, SearchResult, Span, SymbolKind};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};

/// Parse PHP source code and extract symbols
pub fn parse(path: &str, source: &str) -> Result<Vec<SearchResult>> {
    let mut parser = Parser::new();
    let language = tree_sitter_php::LANGUAGE_PHP;

    parser
        .set_language(&language.into())
        .context("Failed to set PHP language")?;

    let tree = parser
        .parse(source, None)
        .context("Failed to parse PHP source")?;

    let root_node = tree.root_node();

    let mut symbols = Vec::new();

    // Extract different types of symbols using Tree-sitter queries
    symbols.extend(extract_functions(source, &root_node, &language.into())?);
    symbols.extend(extract_classes(source, &root_node, &language.into())?);
    symbols.extend(extract_interfaces(source, &root_node, &language.into())?);
    symbols.extend(extract_traits(source, &root_node, &language.into())?);
    symbols.extend(extract_attributes(source, &root_node, &language.into())?);
    symbols.extend(extract_methods(source, &root_node, &language.into())?);
    symbols.extend(extract_properties(source, &root_node, &language.into())?);
    symbols.extend(extract_local_variables(
        source,
        &root_node,
        &language.into(),
    )?);
    symbols.extend(extract_constants(source, &root_node, &language.into())?);
    symbols.extend(extract_namespaces(source, &root_node, &language.into())?);
    symbols.extend(extract_enums(source, &root_node, &language.into())?);

    // Add file path to all symbols
    for symbol in &mut symbols {
        symbol.path = path.to_string();
        symbol.lang = Language::PHP;
    }

    Ok(symbols)
}

/// Extract function definitions
fn extract_functions(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (function_definition
            name: (name) @name) @function
    "#;

    let query = Query::new(language, query_str).context("Failed to create function query")?;

    extract_symbols(source, root, &query, SymbolKind::Function, None)
}

/// Extract class declarations (including abstract and final classes)
fn extract_classes(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_declaration
            name: (name) @name) @class
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
            name: (name) @name) @interface
    "#;

    let query = Query::new(language, query_str).context("Failed to create interface query")?;

    extract_symbols(source, root, &query, SymbolKind::Interface, None)
}

/// Extract trait declarations
fn extract_traits(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (trait_declaration
            name: (name) @name) @trait
    "#;

    let query = Query::new(language, query_str).context("Failed to create trait query")?;

    extract_symbols(source, root, &query, SymbolKind::Trait, None)
}

/// Extract attributes: BOTH definitions and uses
/// Definitions: #[Attribute] class Route { ... }
/// Uses: #[Route("/api/users")] class UserController { ... }
fn extract_attributes(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let mut symbols = Vec::new();

    // Part 1: Extract attribute class DEFINITIONS (#[Attribute] class X)
    let def_query_str = r#"
        (class_declaration
            (attribute_list)
            name: (name) @name) @attribute_class
    "#;

    let def_query = Query::new(language, def_query_str)
        .context("Failed to create attribute definition query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&def_query, *root, source.as_bytes());

    while let Some(match_) = matches.next() {
        let mut name = None;
        let mut class_node = None;

        for capture in match_.captures {
            let capture_name: &str = &def_query.capture_names()[capture.index as usize];
            match capture_name {
                "name" => {
                    name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                }
                "attribute_class" => {
                    class_node = Some(capture.node);
                }
                _ => {}
            }
        }

        // Check if this class has #[Attribute] specifically
        if let (Some(name), Some(node)) = (name, class_node) {
            let class_text = node.utf8_text(source.as_bytes()).unwrap_or("");

            // Check if the class has #[Attribute] attribute
            if class_text.contains("#[Attribute") {
                let span = node_to_span(&node);
                let preview = extract_preview(source, &span);

                symbols.push(SearchResult::new(
                    String::new(),
                    Language::PHP,
                    SymbolKind::Attribute,
                    Some(name),
                    span,
                    None,
                    preview,
                ));
            }
        }
    }

    // Part 2: Extract attribute USES (#[Route(...)] on classes/methods)
    let use_query_str = r#"
        (attribute_list
            (attribute_group
                (attribute
                    (name) @name))) @attr
    "#;

    let use_query =
        Query::new(language, use_query_str).context("Failed to create attribute use query")?;

    symbols.extend(extract_symbols(
        source,
        root,
        &use_query,
        SymbolKind::Attribute,
        None,
    )?);

    Ok(symbols)
}

/// Extract method definitions from classes, traits, and interfaces
fn extract_methods(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_declaration
            name: (name) @class_name
            body: (declaration_list
                (method_declaration
                    name: (name) @method_name))) @class

        (trait_declaration
            name: (name) @trait_name
            body: (declaration_list
                (method_declaration
                    name: (name) @method_name))) @trait

        (interface_declaration
            name: (name) @interface_name
            body: (declaration_list
                (method_declaration
                    name: (name) @method_name))) @interface
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
                "trait_name" => {
                    scope_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                    scope_type = Some("trait");
                }
                "interface_name" => {
                    scope_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                    scope_type = Some("interface");
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
                Language::PHP,
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

/// Extract property declarations from classes and traits
fn extract_properties(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_declaration
            name: (name) @class_name
            body: (declaration_list
                (property_declaration
                    (property_element
                        (variable_name
                            (name) @prop_name))))) @class

        (trait_declaration
            name: (name) @trait_name
            body: (declaration_list
                (property_declaration
                    (property_element
                        (variable_name
                            (name) @prop_name))))) @trait
    "#;

    let query = Query::new(language, query_str).context("Failed to create property query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut scope_name = None;
        let mut scope_type = None;
        let mut prop_name = None;
        let mut prop_node = None;

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
                "trait_name" => {
                    scope_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                    scope_type = Some("trait");
                }
                "prop_name" => {
                    prop_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                    // Find the parent property_declaration node
                    let mut current = capture.node;
                    while let Some(parent) = current.parent() {
                        if parent.kind() == "property_declaration" {
                            prop_node = Some(parent);
                            break;
                        }
                        current = parent;
                    }
                }
                _ => {}
            }
        }

        if let (Some(scope_name), Some(scope_type), Some(prop_name), Some(node)) =
            (scope_name, scope_type, prop_name, prop_node)
        {
            let scope = format!("{} {}", scope_type, scope_name);
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::PHP,
                SymbolKind::Variable,
                Some(prop_name),
                span,
                Some(scope),
                preview,
            ));
        }
    }

    Ok(symbols)
}

/// Extract local variable assignments inside functions
fn extract_local_variables(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (assignment_expression
            left: (variable_name
                (name) @name)) @assignment
    "#;

    let query = Query::new(language, query_str).context("Failed to create local variable query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut name = None;
        let mut assignment_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            match capture_name {
                "name" => {
                    name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                }
                "assignment" => {
                    assignment_node = Some(capture.node);
                }
                _ => {}
            }
        }

        // Accept all variable assignments (global, local in functions, local in methods)
        // Note: Property declarations are handled separately by extract_properties()
        // and use different syntax (property_declaration), so they won't match this query
        if let (Some(name), Some(node)) = (name, assignment_node) {
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::PHP,
                SymbolKind::Variable,
                Some(name),
                span,
                None, // No scope for local variables or global variables
                preview,
            ));
        }
    }

    Ok(symbols)
}

/// Extract constant declarations (class constants and global constants)
fn extract_constants(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (const_declaration
            (const_element
                (name) @name)) @const
    "#;

    let query = Query::new(language, query_str).context("Failed to create constant query")?;

    extract_symbols(source, root, &query, SymbolKind::Constant, None)
}

/// Extract namespace definitions
fn extract_namespaces(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (namespace_definition
            name: (namespace_name) @name) @namespace
    "#;

    let query = Query::new(language, query_str).context("Failed to create namespace query")?;

    extract_symbols(source, root, &query, SymbolKind::Namespace, None)
}

/// Extract enum declarations (PHP 8.1+)
fn extract_enums(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (enum_declaration
            name: (name) @name) @enum
    "#;

    let query = Query::new(language, query_str).context("Failed to create enum query")?;

    extract_symbols(source, root, &query, SymbolKind::Enum, None)
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

        match (name, full_node) {
            (Some(name), Some(node)) => {
                let span = node_to_span(&node);
                let preview = extract_preview(source, &span);

                symbols.push(SearchResult::new(
                    String::new(),
                    Language::PHP,
                    kind.clone(),
                    Some(name),
                    span,
                    scope.clone(),
                    preview,
                ));
            }
            (None, Some(node)) => {
                log::warn!(
                    "PHP parser: Failed to extract name from {:?} capture at line {}",
                    kind,
                    node.start_position().row + 1
                );
            }
            (Some(_), None) => {
                log::warn!("PHP parser: Failed to extract node for {:?} symbol", kind);
            }
            (None, None) => {
                log::warn!(
                    "PHP parser: Failed to extract both name and node for {:?} symbol",
                    kind
                );
            }
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
    fn test_parse_function() {
        let source = r#"
            <?php
            function greet($name) {
                return "Hello, $name!";
            }
        "#;

        let symbols = parse("test.php", source).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("greet"));
        assert!(matches!(symbols[0].kind, SymbolKind::Function));
    }

    #[test]
    fn test_parse_class() {
        let source = r#"
            <?php
            class User {
                private $name;
                private $email;

                public function __construct($name, $email) {
                    $this->name = $name;
                    $this->email = $email;
                }
            }
        "#;

        let symbols = parse("test.php", source).unwrap();

        // Should find class
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
            <?php
            class Calculator {
                public function add($a, $b) {
                    return $a + $b;
                }

                public function subtract($a, $b) {
                    return $a - $b;
                }
            }
        "#;

        let symbols = parse("test.php", source).unwrap();

        // Should find class + 2 methods
        assert!(symbols.len() >= 3);

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
            <?php
            interface Drawable {
                public function draw();
            }
        "#;

        let symbols = parse("test.php", source).unwrap();

        let interface_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Interface))
            .collect();

        assert_eq!(interface_symbols.len(), 1);
        assert_eq!(interface_symbols[0].symbol.as_deref(), Some("Drawable"));
    }

    #[test]
    fn test_parse_trait() {
        let source = r#"
            <?php
            trait Loggable {
                public function log($message) {
                    echo $message;
                }
            }
        "#;

        let symbols = parse("test.php", source).unwrap();

        let trait_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Trait))
            .collect();

        assert_eq!(trait_symbols.len(), 1);
        assert_eq!(trait_symbols[0].symbol.as_deref(), Some("Loggable"));
    }

    #[test]
    fn test_parse_namespace() {
        let source = r#"
            <?php
            namespace App\Controllers;

            class HomeController {
                public function index() {
                    return 'Home';
                }
            }
        "#;

        let symbols = parse("test.php", source).unwrap();

        let namespace_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Namespace))
            .collect();

        assert_eq!(namespace_symbols.len(), 1);
        assert_eq!(
            namespace_symbols[0].symbol.as_deref(),
            Some("App\\Controllers")
        );
    }

    #[test]
    fn test_parse_constants() {
        let source = r#"
            <?php
            const MAX_SIZE = 100;
            const DEFAULT_NAME = 'Anonymous';
        "#;

        let symbols = parse("test.php", source).unwrap();

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
                .any(|s| s.symbol.as_deref() == Some("DEFAULT_NAME"))
        );
    }

    #[test]
    fn test_parse_properties() {
        let source = r#"
            <?php
            class Config {
                private $debug = false;
                public $timeout = 30;
                protected $secret;
            }
        "#;

        let symbols = parse("test.php", source).unwrap();

        let prop_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        assert_eq!(prop_symbols.len(), 3);
        assert!(
            prop_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("debug"))
        );
        assert!(
            prop_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("timeout"))
        );
        assert!(
            prop_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("secret"))
        );
    }

    #[test]
    fn test_parse_enum() {
        let source = r#"
            <?php
            enum Status {
                case Active;
                case Inactive;
                case Pending;
            }
        "#;

        let symbols = parse("test.php", source).unwrap();

        let enum_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Enum))
            .collect();

        assert_eq!(enum_symbols.len(), 1);
        assert_eq!(enum_symbols[0].symbol.as_deref(), Some("Status"));
    }

    #[test]
    fn test_parse_mixed_symbols() {
        let source = r#"
            <?php
            namespace App\Models;

            interface UserInterface {
                public function getName();
            }

            trait Timestampable {
                private $createdAt;

                public function getCreatedAt() {
                    return $this->createdAt;
                }
            }

            class User implements UserInterface {
                use Timestampable;

                private $name;
                const DEFAULT_ROLE = 'user';

                public function __construct($name) {
                    $this->name = $name;
                }

                public function getName() {
                    return $this->name;
                }
            }

            function createUser($name) {
                return new User($name);
            }
        "#;

        let symbols = parse("test.php", source).unwrap();

        // Should find: namespace, interface, trait, class, methods, properties, const, function
        assert!(symbols.len() >= 8);

        let kinds: Vec<&SymbolKind> = symbols.iter().map(|s| &s.kind).collect();
        assert!(kinds.contains(&&SymbolKind::Namespace));
        assert!(kinds.contains(&&SymbolKind::Interface));
        assert!(kinds.contains(&&SymbolKind::Trait));
        assert!(kinds.contains(&&SymbolKind::Class));
        assert!(kinds.contains(&&SymbolKind::Method));
        assert!(kinds.contains(&&SymbolKind::Variable));
        assert!(kinds.contains(&&SymbolKind::Constant));
        assert!(kinds.contains(&&SymbolKind::Function));
    }

    #[test]
    fn test_local_variables_included() {
        let source = r#"
            <?php
            $global_count = 100;

            function calculate() {
                $local_count = 50;
                $result = $local_count + 10;
                return $result;
            }

            class Math {
                private $value = 5;

                public function compute() {
                    $temp = $this->value * 2;
                    return $temp;
                }
            }
        "#;

        let symbols = parse("test.php", source).unwrap();

        // Filter to just variables (both global assignment, local vars, and class properties)
        let variables: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        // Should find: global_count (global), value (property), local_count, result, temp
        assert_eq!(variables.len(), 5);

        // Check that local variables inside functions are captured
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("local_count"))
        );
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("result"))
        );
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("temp"))
        );

        // Check that global assignment is captured
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("global_count"))
        );

        // Check that class property is captured
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("value"))
        );

        // Verify that local variables have no scope
        let local_vars: Vec<_> = variables
            .iter()
            .filter(|v| {
                v.symbol.as_deref() == Some("local_count")
                    || v.symbol.as_deref() == Some("result")
                    || v.symbol.as_deref() == Some("temp")
            })
            .collect();

        for var in local_vars {
            // Removed: scope field no longer exists: assert_eq!(var.scope, None);
        }

        // Verify that class property has scope
        let property = variables
            .iter()
            .find(|v| v.symbol.as_deref() == Some("value"))
            .unwrap();
        // Removed: scope field no longer exists: assert_eq!(property.scope.as_ref().unwrap(), "class Math");
    }

    #[test]
    fn test_parse_attribute_class() {
        let source = r#"
            <?php
            #[Attribute]
            class Route {
                public function __construct(
                    public string $path,
                    public array $methods = []
                ) {}
            }

            #[Attribute(Attribute::TARGET_METHOD)]
            class Deprecated {
                public string $message;
            }
        "#;

        let symbols = parse("test.php", source).unwrap();

        let attribute_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Attribute))
            .collect();

        // Should find Route and Deprecated attribute classes
        assert!(attribute_symbols.len() >= 2);
        assert!(
            attribute_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Route"))
        );
        assert!(
            attribute_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Deprecated"))
        );
    }

    #[test]
    fn test_parse_attribute_uses() {
        let source = r#"
            <?php
            #[Attribute]
            class Route {
                public function __construct(public string $path) {}
            }

            #[Attribute]
            class Deprecated {}

            #[Route("/api/users")]
            class UserController {
                #[Route("/list")]
                public function list() {
                    return [];
                }

                #[Route("/get/{id}")]
                #[Deprecated]
                public function get($id) {
                    return null;
                }
            }

            #[Route("/api/posts")]
            class PostController {
                #[Route("/all")]
                public function all() {
                    return [];
                }
            }
        "#;

        let symbols = parse("test.php", source).unwrap();

        let attribute_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Attribute))
            .collect();

        // Should find attribute class definitions (Route, Deprecated)
        // AND attribute uses (Route appears 5 times, Deprecated appears 1 time)
        // Total expected: 2 definitions + 6 uses = 8
        assert!(attribute_symbols.len() >= 6);

        // Count specific attribute uses
        let route_count = attribute_symbols
            .iter()
            .filter(|s| s.symbol.as_deref() == Some("Route"))
            .count();

        let deprecated_count = attribute_symbols
            .iter()
            .filter(|s| s.symbol.as_deref() == Some("Deprecated"))
            .count();

        // Should find Route at least 5 times (1 definition + 5 uses)
        assert!(route_count >= 5);

        // Should find Deprecated at least 2 times (1 definition + 1 use)
        assert!(deprecated_count >= 2);
    }

    #[test]
    fn test_parse_class_implementing_multiple_interfaces() {
        let source = r#"
            <?php
            interface Interface1 {
                public function method1();
            }

            interface Interface2 {
                public function method2();
            }

            class SimpleClass {
                public $value;
            }

            // Class implementing multiple interfaces
            class MultiInterfaceClass implements Interface1, Interface2 {
                public function method1() {
                    return true;
                }

                public function method2() {
                    return false;
                }
            }

            /**
             * Complex edge case: Class with large docblock, extends base class, implements multiple interfaces
             *
             * @property string $name
             * @property string $email
             * @property-read int $id
             * @property-read string $created_at
             * @property-read Collection|Role[] $roles
             * @property-read Collection|Permission[] $permissions
             * @property-read Workflow $workflow
             * @property-read Collection|NotificationSetting[] $notificationSettings
             * @property-read Collection|Watch[] $watches
             *
             **/
            class ComplexClass extends SimpleClass implements Interface1, Interface2 {
                private $data;

                public function method1() {
                    return $this->data;
                }

                public function method2() {
                    return !$this->data;
                }
            }
        "#;

        let symbols = parse("test.php", source).unwrap();

        let class_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        // Should find all 3 classes:
        // 1. SimpleClass
        // 2. MultiInterfaceClass (implements 2 interfaces)
        // 3. ComplexClass (extends + implements 2 interfaces + large docblock)
        assert_eq!(class_symbols.len(), 3, "Should find exactly 3 classes");

        assert!(
            class_symbols
                .iter()
                .any(|c| c.symbol.as_deref() == Some("SimpleClass")),
            "Should find SimpleClass"
        );
        assert!(
            class_symbols
                .iter()
                .any(|c| c.symbol.as_deref() == Some("MultiInterfaceClass")),
            "Should find MultiInterfaceClass implementing multiple interfaces"
        );
        assert!(
            class_symbols
                .iter()
                .any(|c| c.symbol.as_deref() == Some("ComplexClass")),
            "Should find ComplexClass with large docblock, extends, and implements multiple interfaces"
        );
    }

    #[test]
    fn test_extract_php_use_dependencies() {
        let source = r#"
            <?php

            use Illuminate\Database\Migrations\Migration;
            use Illuminate\Database\Schema\Blueprint;
            use Illuminate\Support\Facades\Schema;

            return new class extends Migration
            {
                public function up(): void
                {
                    Schema::create('test', function (Blueprint $table) {
                        $table->id();
                    });
                }
            };
        "#;

        let deps = PhpDependencyExtractor::extract_dependencies(source).unwrap();

        // Should find 3 use statements
        assert_eq!(deps.len(), 3, "Should extract 3 use statements");

        // Check specific imports
        assert!(deps.iter().any(|d| d.imported_path.contains("Migration")));
        assert!(deps.iter().any(|d| d.imported_path.contains("Blueprint")));
        assert!(deps.iter().any(|d| d.imported_path.contains("Schema")));

        // All should be Internal (Laravel framework classes)
        for dep in &deps {
            assert!(
                matches!(dep.import_type, ImportType::Internal),
                "Laravel classes should be classified as Internal"
            );
        }
    }

    #[test]
    fn test_dynamic_requires_filtered() {
        let source = r#"
            <?php
            use App\Models\User;
            use App\Services\Auth;
            require 'config.php';
            require_once 'helpers.php';

            // Dynamic requires - should be filtered out
            require $variable;
            require CONSTANT . '/file.php';
            require_once $path;
            include dirname(__FILE__) . '/dynamic.php';
        "#;

        let deps = PhpDependencyExtractor::extract_dependencies(source).unwrap();

        // Should only find static use statements and require with string literals
        // Variable and expression-based requires are filtered (not (string) nodes)
        assert_eq!(deps.len(), 4, "Should extract 4 static imports only");

        assert!(deps.iter().any(|d| d.imported_path.contains("User")));
        assert!(deps.iter().any(|d| d.imported_path.contains("Auth")));
        assert!(deps.iter().any(|d| d.imported_path == "config.php"));
        assert!(deps.iter().any(|d| d.imported_path == "helpers.php"));

        // Verify dynamic requires are NOT captured
        assert!(!deps.iter().any(|d| d.imported_path.contains("variable")));
        assert!(!deps.iter().any(|d| d.imported_path.contains("CONSTANT")));
        assert!(!deps.iter().any(|d| d.imported_path.contains("dirname")));
    }
}

// ============================================================================
// Dependency Extraction
// ============================================================================

use crate::models::ImportType;
use crate::parsers::{DependencyExtractor, ImportInfo};

/// PHP dependency extractor
pub struct PhpDependencyExtractor;

impl DependencyExtractor for PhpDependencyExtractor {
    fn extract_dependencies(source: &str) -> Result<Vec<ImportInfo>> {
        let mut parser = Parser::new();
        let language = tree_sitter_php::LANGUAGE_PHP;

        parser
            .set_language(&language.into())
            .context("Failed to set PHP language")?;

        let tree = parser
            .parse(source, None)
            .context("Failed to parse PHP source")?;

        let root_node = tree.root_node();

        let mut imports = Vec::new();

        // Extract use declarations
        imports.extend(extract_php_uses(source, &root_node)?);

        // Extract require/include statements
        imports.extend(extract_php_requires(source, &root_node)?);

        Ok(imports)
    }
}

/// Extract PHP `use` declarations
fn extract_php_uses(source: &str, root: &tree_sitter::Node) -> Result<Vec<ImportInfo>> {
    let language = tree_sitter_php::LANGUAGE_PHP;

    let query_str = r#"
        (namespace_use_clause
            [
                (name) @use_path
                (qualified_name) @use_path
            ])
    "#;

    let query =
        Query::new(&language.into(), query_str).context("Failed to create PHP use query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut imports = Vec::new();

    while let Some(match_) = matches.next() {
        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            if capture_name == "use_path" {
                let path = capture
                    .node
                    .utf8_text(source.as_bytes())
                    .unwrap_or("")
                    .to_string();
                let import_type = classify_php_use(&path);
                let line_number = capture.node.start_position().row + 1;

                imports.push(ImportInfo {
                    imported_path: path,
                    import_type,
                    line_number,
                    imported_symbols: None, // PHP imports entire namespace/class
                });
            }
        }
    }

    Ok(imports)
}

/// Extract PHP `require`, `require_once`, `include`, `include_once` statements
fn extract_php_requires(source: &str, root: &tree_sitter::Node) -> Result<Vec<ImportInfo>> {
    let language = tree_sitter_php::LANGUAGE_PHP;

    // Match require/include with both string and expression
    let query_str = r#"
        (expression_statement
            (require_expression
                (string) @require_path)) @require

        (expression_statement
            (require_once_expression
                (string) @require_path)) @require

        (expression_statement
            (include_expression
                (string) @require_path)) @require

        (expression_statement
            (include_once_expression
                (string) @require_path)) @require
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create PHP require/include query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut imports = Vec::new();

    while let Some(match_) = matches.next() {
        let mut require_path = None;
        let mut require_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            match capture_name {
                "require_path" => {
                    let raw_path = capture.node.utf8_text(source.as_bytes()).unwrap_or("");
                    // Remove quotes from path
                    require_path =
                        Some(raw_path.trim_matches(|c| c == '"' || c == '\'').to_string());
                }
                "require" => {
                    require_node = Some(capture.node);
                }
                _ => {}
            }
        }

        if let (Some(path), Some(node)) = (require_path, require_node) {
            // For require/include, we consider them as internal file dependencies
            let line_number = node.start_position().row + 1;

            imports.push(ImportInfo {
                imported_path: path,
                import_type: ImportType::Internal, // File includes are always internal
                line_number,
                imported_symbols: None, // Includes don't specify symbols
            });
        }
    }

    Ok(imports)
}

/// Classify a PHP `use` declaration as internal, external, or stdlib
fn classify_php_use(use_path: &str) -> ImportType {
    // PHP standard library extensions/classes (built-in PHP namespaces)
    const PHP_STDLIB_NAMESPACES: &[&str] = &[
        // PSR standards (PHP standard interfaces)
        "Psr\\",
        "Psr\\Http",
        "Psr\\Log",
        "Psr\\Cache",
        "Psr\\Container",
        // PHP built-in classes/interfaces
        "Exception",
        "Error",
        "DateTime",
        "DateTimeImmutable",
        "DateTimeInterface",
        "DateInterval",
        "DatePeriod",
        "PDO",
        "PDOStatement",
        "Closure",
        "Generator",
        "ArrayIterator",
        "IteratorAggregate",
        "Traversable",
        "Iterator",
        "Countable",
        "Serializable",
        "JsonSerializable",
        // SPL (Standard PHP Library)
        "SplFileInfo",
        "SplFileObject",
        "SplDoublyLinkedList",
        "SplQueue",
        "SplStack",
        "SplHeap",
        "SplMinHeap",
        "SplMaxHeap",
        "SplPriorityQueue",
        "SplFixedArray",
        "SplObjectStorage",
        // PHP XML classes
        "SimpleXMLElement",
        "DOMDocument",
        "DOMElement",
        "DOMNode",
        "XMLReader",
        "XMLWriter",
    ];

    // Common vendor packages (third-party dependencies from composer)
    const PHP_VENDOR_NAMESPACES: &[&str] = &[
        // Symfony framework
        "Symfony\\",
        // Popular packages
        "Spatie\\",
        "Stancl\\",
        "Doctrine\\",
        "Monolog\\",
        "PHPUnit\\",
        "Carbon\\",
        "GuzzleHttp\\",
        "Composer\\",
        "Predis\\",
        "League\\",
        "Ramsey\\",
        "Webmozart\\",
        "Brick\\",
        "Mockery\\",
        "Faker\\",
        "PhpParser\\",
        "PHPStan\\",
        "Psalm\\",
        "Pest\\",
        "Filament\\",
        "Livewire\\",
        "Inertia\\",
        "Socialite\\",
        "Sanctum\\",
        "Passport\\",
        "Horizon\\",
        "Telescope\\",
        "Forge\\",
        "Vapor\\",
        "Cashier\\",
        "Nova\\",
        "Spark\\",
        "Jetstream\\",
        "Fortify\\",
        "Breeze\\",
        "Vonage\\",
        "Twilio\\",
        "Stripe\\",
        "Pusher\\",
        "Algolia\\",
        "Aws\\",
        "Google\\",
        "Microsoft\\",
        "Facebook\\",
        "Twitter\\",
        "Sentry\\",
        "Bugsnag\\",
        "Rollbar\\",
        "NewRelic\\",
        "Datadog\\",
        "Elasticsearch\\",
        "Redis\\",
        "Memcached\\",
        "MongoDB\\",
        "PhpOffice\\",
        "Dompdf\\",
        "TCPDF\\",
        "Mpdf\\",
        "Intervention\\",
        "Barryvdh\\",
        "Maatwebsite\\",
        "Rap2hpoutre\\",
        "Yajra\\",
    ];

    // Check if it's a standard library class
    for stdlib_ns in PHP_STDLIB_NAMESPACES {
        if use_path == *stdlib_ns || use_path.starts_with(stdlib_ns) {
            return ImportType::Stdlib;
        }
    }

    // Check if it's a vendor/third-party package
    for vendor_ns in PHP_VENDOR_NAMESPACES {
        if use_path.starts_with(vendor_ns) {
            return ImportType::External;
        }
    }

    // Internal: project namespaces
    ImportType::Internal
}

// ============================================================================
// PSR-4 Autoloading Support (composer.json parser)
// ============================================================================

/// PSR-4 autoload mapping (namespace prefix → directory path)
#[derive(Debug, Clone)]
pub struct Psr4Mapping {
    pub namespace_prefix: String, // e.g., "App\\"
    pub directory: String,        // e.g., "app/"
    pub project_root: String,     // e.g., "services/php/rcm-backend/" (relative to index root)
}

/// Parse composer.json and extract PSR-4 autoload mappings
///
/// Returns a vector of PSR-4 mappings sorted by namespace length (longest first)
/// to ensure more specific namespaces are matched before general ones.
///
/// # Arguments
///
/// * `project_root` - Root directory of the project (where composer.json is located)
pub fn parse_composer_psr4(project_root: &Path) -> Result<Vec<Psr4Mapping>> {
    let composer_path = project_root.join("composer.json");

    // If composer.json doesn't exist, return empty mappings
    if !composer_path.exists() {
        log::debug!("No composer.json found at {:?}", composer_path);
        return Ok(Vec::new());
    }

    let content =
        std::fs::read_to_string(&composer_path).context("Failed to read composer.json")?;

    let json: serde_json::Value =
        serde_json::from_str(&content).context("Failed to parse composer.json")?;

    let mut mappings = Vec::new();

    // Extract PSR-4 mappings from autoload section
    if let Some(autoload) = json.get("autoload") {
        if let Some(psr4) = autoload.get("psr-4") {
            if let Some(psr4_obj) = psr4.as_object() {
                for (namespace, path) in psr4_obj {
                    // path can be a string or array of strings
                    let directories = match path {
                        serde_json::Value::String(s) => vec![s.clone()],
                        serde_json::Value::Array(arr) => arr
                            .iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect(),
                        _ => continue,
                    };

                    for dir in directories {
                        mappings.push(Psr4Mapping {
                            namespace_prefix: namespace.clone(),
                            directory: dir,
                            project_root: String::new(), // Empty for single-project use
                        });
                    }
                }
            }
        }
    }

    // Sort by namespace length (longest first) for correct matching
    // Example: "App\\Http\\" should match before "App\\"
    mappings.sort_by(|a, b| b.namespace_prefix.len().cmp(&a.namespace_prefix.len()));

    log::debug!(
        "Loaded {} PSR-4 mappings from composer.json",
        mappings.len()
    );
    for mapping in &mappings {
        log::trace!("  {} => {}", mapping.namespace_prefix, mapping.directory);
    }

    Ok(mappings)
}

/// Find all composer.json files in a directory tree (excluding vendor directories)
///
/// # Arguments
///
/// * `index_root` - Root directory of the indexed codebase
///
/// # Returns
///
/// Vector of absolute paths to composer.json files (excluding vendor/)
pub fn find_all_composer_json(index_root: &Path) -> Result<Vec<PathBuf>> {
    use ignore::WalkBuilder;

    let mut composer_files = Vec::new();

    let walker = WalkBuilder::new(index_root)
        .follow_links(false)
        .git_ignore(true)
        .build();

    for entry in walker {
        let entry = entry?;
        let path = entry.path();

        // Only process files named composer.json
        if !path.is_file() || path.file_name() != Some(std::ffi::OsStr::new("composer.json")) {
            continue;
        }

        // Skip vendor directories (composer packages)
        if path.components().any(|c| c.as_os_str() == "vendor") {
            log::trace!("Skipping vendor composer.json: {:?}", path);
            continue;
        }

        composer_files.push(path.to_path_buf());
    }

    log::debug!("Found {} project composer.json files", composer_files.len());
    Ok(composer_files)
}

/// Parse all composer.json files in a monorepo and extract PSR-4 mappings
///
/// # Arguments
///
/// * `index_root` - Root directory of the indexed codebase (e.g., monorepo root)
///
/// # Returns
///
/// Vector of PSR-4 mappings with project_root relative to index_root
pub fn parse_all_composer_psr4(index_root: &Path) -> Result<Vec<Psr4Mapping>> {
    let composer_files = find_all_composer_json(index_root)?;

    if composer_files.is_empty() {
        log::debug!("No composer.json files found in {:?}", index_root);
        return Ok(Vec::new());
    }

    let mut all_mappings = Vec::new();
    let composer_count = composer_files.len(); // Save count before moving

    for composer_path in composer_files {
        let project_root = composer_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("composer.json has no parent directory"))?;

        // Get project root relative to index root
        let relative_project_root = project_root
            .strip_prefix(index_root)
            .unwrap_or(project_root)
            .to_string_lossy()
            .to_string();

        log::debug!("Parsing composer.json at {:?}", composer_path);

        // Parse this composer.json
        let mappings = parse_composer_psr4(project_root)?;

        // Add project_root to each mapping
        for mut mapping in mappings {
            mapping.project_root = relative_project_root.clone();
            all_mappings.push(mapping);
        }
    }

    // Sort by namespace length (longest first) for correct matching
    all_mappings.sort_by(|a, b| b.namespace_prefix.len().cmp(&a.namespace_prefix.len()));

    log::info!(
        "Loaded {} total PSR-4 mappings from {} projects",
        all_mappings.len(),
        composer_count
    );

    Ok(all_mappings)
}

/// Resolve a PHP namespace to a file path using PSR-4 autoload rules
///
/// # Arguments
///
/// * `namespace` - Full namespace (e.g., "App\\Http\\Controllers\\UserController")
/// * `psr4_mappings` - PSR-4 mappings from composer.json
///
/// # Returns
///
/// Relative file path (e.g., "app/Http/Controllers/UserController.php") or None if not resolvable
///
/// # PSR-4 Resolution Rules
///
/// 1. Find the longest matching namespace prefix
/// 2. Strip the prefix from the namespace
/// 3. Convert remaining namespace to path (replace \\ with /)
/// 4. Append to the mapped directory
/// 5. Add .php extension
///
/// # Examples
///
/// ```
/// // PSR-4 mapping: "App\\" => "app/"
/// // Input: "App\\Http\\Controllers\\UserController"
/// // Output: "app/Http/Controllers/UserController.php"
/// ```
pub fn resolve_php_namespace_to_path(
    namespace: &str,
    psr4_mappings: &[Psr4Mapping],
) -> Option<String> {
    // Find the longest matching PSR-4 prefix
    for mapping in psr4_mappings {
        if namespace.starts_with(&mapping.namespace_prefix) {
            // Strip the namespace prefix
            let relative_namespace = &namespace[mapping.namespace_prefix.len()..];

            // Convert namespace to path (replace \\ with /)
            let relative_path = relative_namespace.replace('\\', "/");

            // Combine directory + relative_path + .php
            let file_path = if relative_path.is_empty() {
                // Namespace exactly matches prefix (e.g., "App\\") → "app/.php" (invalid)
                // This shouldn't happen for valid class imports
                return None;
            } else {
                // Build full path: project_root + directory + relative_path + .php
                let base_path = if mapping.project_root.is_empty() {
                    // Single-project mode: just directory + file
                    format!("{}{}.php", mapping.directory, relative_path)
                } else {
                    // Monorepo mode: project_root + directory + file
                    format!(
                        "{}/{}{}.php",
                        mapping.project_root, mapping.directory, relative_path
                    )
                };

                // Normalize path separators (replace // with /)
                base_path.replace("//", "/")
            };

            log::trace!("Resolved namespace '{}' to path '{}'", namespace, file_path);
            return Some(file_path);
        }
    }

    // No matching PSR-4 prefix found
    log::trace!("No PSR-4 mapping found for namespace '{}'", namespace);
    None
}
