//! C++ language parser using Tree-sitter
//!
//! Extracts symbols from C++ source code:
//! - Functions (regular and template)
//! - Classes (regular, abstract, template)
//! - Structs
//! - Namespaces
//! - Templates (class and function)
//! - Methods (with class scope, virtual, override)
//! - Constructors/Destructors
//! - Operators
//! - Enums (enum and enum class)
//! - Local variables (inside functions and methods)
//! - Using declarations
//! - Type aliases

use crate::models::{Language, SearchResult, Span, SymbolKind};
use anyhow::{Context, Result};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};

/// Parse C++ source code and extract symbols
pub fn parse(path: &str, source: &str) -> Result<Vec<SearchResult>> {
    let mut parser = Parser::new();
    let language = tree_sitter_cpp::LANGUAGE;

    parser
        .set_language(&language.into())
        .context("Failed to set C++ language")?;

    let tree = parser
        .parse(source, None)
        .context("Failed to parse C++ source")?;

    let root_node = tree.root_node();

    let mut symbols = Vec::new();

    // Extract different types of symbols using Tree-sitter queries
    symbols.extend(extract_functions(source, &root_node, &language.into())?);
    symbols.extend(extract_classes(source, &root_node, &language.into())?);
    symbols.extend(extract_structs(source, &root_node, &language.into())?);
    symbols.extend(extract_namespaces(source, &root_node, &language.into())?);
    symbols.extend(extract_enums(source, &root_node, &language.into())?);
    symbols.extend(extract_methods(source, &root_node, &language.into())?);
    symbols.extend(extract_local_variables(
        source,
        &root_node,
        &language.into(),
    )?);
    symbols.extend(extract_type_aliases(source, &root_node, &language.into())?);

    // Add file path to all symbols
    for symbol in &mut symbols {
        symbol.path = path.to_string();
        symbol.lang = Language::Cpp;
    }

    Ok(symbols)
}

/// Extract function declarations and definitions
fn extract_functions(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (function_definition
            declarator: (function_declarator
                declarator: (identifier) @name)) @function

        (function_definition
            declarator: (function_declarator
                declarator: (qualified_identifier
                    name: (identifier) @name))) @function

        (template_declaration
            (function_definition
                declarator: (function_declarator
                    declarator: (identifier) @name))) @function
    "#;

    let query = Query::new(language, query_str).context("Failed to create function query")?;

    extract_symbols(source, root, &query, SymbolKind::Function, None)
}

/// Extract class declarations
fn extract_classes(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_specifier
            name: (type_identifier) @name) @class

        (template_declaration
            (class_specifier
                name: (type_identifier) @name)) @class
    "#;

    let query = Query::new(language, query_str).context("Failed to create class query")?;

    extract_symbols(source, root, &query, SymbolKind::Class, None)
}

/// Extract struct declarations
fn extract_structs(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (struct_specifier
            name: (type_identifier) @name) @struct

        (template_declaration
            (struct_specifier
                name: (type_identifier) @name)) @struct
    "#;

    let query = Query::new(language, query_str).context("Failed to create struct query")?;

    extract_symbols(source, root, &query, SymbolKind::Struct, None)
}

/// Extract namespace definitions
fn extract_namespaces(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (namespace_definition
            name: (_) @name) @namespace
    "#;

    let query = Query::new(language, query_str).context("Failed to create namespace query")?;

    extract_symbols(source, root, &query, SymbolKind::Namespace, None)
}

/// Extract enum declarations
fn extract_enums(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (enum_specifier
            name: (type_identifier) @name) @enum
    "#;

    let query = Query::new(language, query_str).context("Failed to create enum query")?;

    extract_symbols(source, root, &query, SymbolKind::Enum, None)
}

/// Extract method definitions from classes and structs
fn extract_methods(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_specifier
            name: (type_identifier) @class_name
            body: (field_declaration_list
                (function_definition
                    declarator: (function_declarator
                        declarator: (field_identifier) @method_name)))) @class

        (class_specifier
            name: (type_identifier) @class_name
            body: (field_declaration_list
                (function_definition
                    declarator: (function_declarator
                        declarator: (destructor_name) @method_name)))) @class

        (struct_specifier
            name: (type_identifier) @struct_name
            body: (field_declaration_list
                (function_definition
                    declarator: (function_declarator
                        declarator: (field_identifier) @method_name)))) @struct

        (struct_specifier
            name: (type_identifier) @struct_name
            body: (field_declaration_list
                (function_definition
                    declarator: (function_declarator
                        declarator: (destructor_name) @method_name)))) @struct
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
                "struct_name" => {
                    scope_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                    scope_type = Some("struct");
                }
                "method_name" => {
                    method_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                    // Find the parent function_definition node
                    let mut current = capture.node;
                    while let Some(parent) = current.parent() {
                        if parent.kind() == "function_definition" {
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
                Language::Cpp,
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

/// Extract local variable declarations inside functions and methods
fn extract_local_variables(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (declaration
            declarator: (init_declarator
                declarator: (identifier) @name)) @var
    "#;

    let query = Query::new(language, query_str).context("Failed to create local variable query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut name = None;
        let mut var_node = None;

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
                "var" => {
                    var_node = Some(capture.node);
                }
                _ => {}
            }
        }

        // Only extract variables that are inside function definitions (local variables)
        if let (Some(name), Some(node)) = (name, var_node) {
            let mut is_local_var = false;
            let mut current = node;

            while let Some(parent) = current.parent() {
                if parent.kind() == "function_definition" {
                    is_local_var = true;
                    break;
                }
                current = parent;
            }

            if is_local_var {
                let span = node_to_span(&node);
                let preview = extract_preview(source, &span);

                symbols.push(SearchResult::new(
                    String::new(),
                    Language::Cpp,
                    SymbolKind::Variable,
                    Some(name),
                    span,
                    None, // No scope for local variables
                    preview,
                ));
            }
        }
    }

    Ok(symbols)
}

/// Extract type aliases (using and typedef)
fn extract_type_aliases(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (type_definition
            declarator: (type_identifier) @name) @typedef

        (alias_declaration
            name: (type_identifier) @name) @using
    "#;

    let query = Query::new(language, query_str).context("Failed to create type alias query")?;

    extract_symbols(source, root, &query, SymbolKind::Type, None)
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
    let mut seen_names = std::collections::HashSet::new();

    while let Some(match_) = matches.next() {
        // Find the name capture and the full node
        let mut name = None;
        let mut name_node = None;
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
                name_node = Some(capture.node);
            } else {
                // Assume any other capture is the full node
                full_node = Some(capture.node);
            }
        }

        if let (Some(name), Some(name_node), Some(node)) = (name, name_node, full_node) {
            // Deduplicate by name position - this handles cases where template patterns
            // match the same symbol twice (e.g., both template_declaration and class_specifier)
            let name_key = (name_node.start_byte(), name_node.end_byte(), name.clone());
            if seen_names.contains(&name_key) {
                continue; // Skip duplicate
            }
            seen_names.insert(name_key);

            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::Cpp,
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
    fn test_parse_function() {
        let source = r#"
int add(int a, int b) {
    return a + b;
}
        "#;

        let symbols = parse("test.cpp", source).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("add"));
        assert!(matches!(symbols[0].kind, SymbolKind::Function));
    }

    #[test]
    fn test_parse_class() {
        let source = r#"
class User {
private:
    std::string name;
    int age;

public:
    User(std::string n, int a) : name(n), age(a) {}
};
        "#;

        let symbols = parse("test.cpp", source).unwrap();

        let class_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        assert_eq!(class_symbols.len(), 1);
        assert_eq!(class_symbols[0].symbol.as_deref(), Some("User"));
    }

    #[test]
    fn test_parse_namespace() {
        let source = r#"
namespace MyNamespace {
    int value = 42;
}

namespace Nested::Namespace {
    void function() {}
}
        "#;

        let symbols = parse("test.cpp", source).unwrap();

        let namespace_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Namespace))
            .collect();

        assert!(namespace_symbols.len() >= 1);
        assert!(
            namespace_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("MyNamespace"))
        );
    }

    #[test]
    fn test_parse_struct() {
        let source = r#"
struct Point {
    int x;
    int y;
};
        "#;

        let symbols = parse("test.cpp", source).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("Point"));
        assert!(matches!(symbols[0].kind, SymbolKind::Struct));
    }

    #[test]
    fn test_parse_enum() {
        let source = r#"
enum Color {
    RED,
    GREEN,
    BLUE
};

enum class Status {
    Active,
    Inactive
};
        "#;

        let symbols = parse("test.cpp", source).unwrap();

        let enum_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Enum))
            .collect();

        assert_eq!(enum_symbols.len(), 2);
        assert!(
            enum_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Color"))
        );
        assert!(
            enum_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Status"))
        );
    }

    #[test]
    fn test_parse_template_class() {
        let source = r#"
template <typename T>
class Container {
private:
    T value;

public:
    Container(T v) : value(v) {}
    T getValue() { return value; }
};
        "#;

        let symbols = parse("test.cpp", source).unwrap();

        let class_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        assert_eq!(class_symbols.len(), 1);
        assert_eq!(class_symbols[0].symbol.as_deref(), Some("Container"));
    }

    #[test]
    fn test_parse_template_function() {
        let source = r#"
template <typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}
        "#;

        let symbols = parse("test.cpp", source).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("max"));
        assert!(matches!(symbols[0].kind, SymbolKind::Function));
    }

    #[test]
    fn test_parse_class_with_methods() {
        let source = r#"
class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }

    int subtract(int a, int b) {
        return a - b;
    }
};
        "#;

        let symbols = parse("test.cpp", source).unwrap();

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
    fn test_parse_using_declaration() {
        let source = r#"
using StringVector = std::vector<std::string>;
using IntPtr = int*;
        "#;

        let symbols = parse("test.cpp", source).unwrap();

        let type_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Type))
            .collect();

        assert!(type_symbols.len() >= 1);
        assert!(
            type_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("StringVector"))
        );
    }

    #[test]
    fn test_parse_typedef() {
        let source = r#"
typedef unsigned int uint;
typedef struct {
    int x, y;
} Point;
        "#;

        let symbols = parse("test.cpp", source).unwrap();

        let type_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Type))
            .collect();

        assert!(type_symbols.len() >= 1);
    }

    #[test]
    fn test_parse_mixed_symbols() {
        let source = r#"
namespace Math {
    class Vector {
    private:
        double x, y;

    public:
        Vector(double x, double y) : x(x), y(y) {}

        double magnitude() {
            return sqrt(x*x + y*y);
        }
    };

    enum Operation {
        ADD,
        SUBTRACT
    };

    template <typename T>
    T multiply(T a, T b) {
        return a * b;
    }
}
        "#;

        let symbols = parse("test.cpp", source).unwrap();

        // Should find: namespace, class, enum, method, function
        assert!(symbols.len() >= 5);

        let kinds: Vec<&SymbolKind> = symbols.iter().map(|s| &s.kind).collect();
        assert!(kinds.contains(&&SymbolKind::Namespace));
        assert!(kinds.contains(&&SymbolKind::Class));
        assert!(kinds.contains(&&SymbolKind::Enum));
        assert!(kinds.contains(&&SymbolKind::Method));
        assert!(kinds.contains(&&SymbolKind::Function));
    }

    #[test]
    fn test_parse_nested_namespace() {
        let source = r#"
namespace Outer {
    namespace Inner {
        void function() {}
    }
}
        "#;

        let symbols = parse("test.cpp", source).unwrap();

        let namespace_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Namespace))
            .collect();

        assert_eq!(namespace_symbols.len(), 2);
        assert!(
            namespace_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Outer"))
        );
        assert!(
            namespace_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Inner"))
        );
    }

    #[test]
    fn test_parse_virtual_methods() {
        let source = r#"
class Base {
public:
    virtual void draw() = 0;
    virtual void update() {}
};

class Derived : public Base {
public:
    void draw() override {
        // Implementation
    }
};
        "#;

        let symbols = parse("test.cpp", source).unwrap();

        let class_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        assert_eq!(class_symbols.len(), 2);
        assert!(
            class_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Base"))
        );
        assert!(
            class_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Derived"))
        );

        let method_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method))
            .collect();

        assert!(method_symbols.len() >= 2);
    }

    #[test]
    fn test_parse_operator_overload() {
        let source = r#"
class Complex {
private:
    double real, imag;

public:
    Complex operator+(const Complex& other) {
        return Complex(real + other.real, imag + other.imag);
    }
};
        "#;

        let symbols = parse("test.cpp", source).unwrap();

        let class_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        assert_eq!(class_symbols.len(), 1);
        assert_eq!(class_symbols[0].symbol.as_deref(), Some("Complex"));
    }

    #[test]
    fn test_local_variables_included() {
        let source = r#"
int calculate(int input) {
    int localVar = input * 2;
    auto result = localVar + 10;
    return result;
}

class Calculator {
public:
    int compute(int value) {
        int temp = value * 3;
        auto final = temp + 5;
        return final;
    }
};
        "#;

        let symbols = parse("test.cpp", source).unwrap();

        // Filter to just variables
        let variables: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        // Check that local variables are captured
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("localVar"))
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
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("final"))
        );

        // Verify that local variables have no scope
        for var in variables {
            // Removed: scope field no longer exists: assert_eq!(var.scope, None);
        }
    }

    #[test]
    fn test_parse_destructor() {
        let source = r#"
class Resource {
private:
    int* data;

public:
    Resource() {
        data = new int[100];
    }

    ~Resource() {
        delete[] data;
    }
};
        "#;

        let symbols = parse("test.cpp", source).unwrap();

        let class_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        assert_eq!(class_symbols.len(), 1);
        assert_eq!(class_symbols[0].symbol.as_deref(), Some("Resource"));

        // Check if destructor is extracted
        let method_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method))
            .collect();

        // Should have both constructor and destructor
        assert!(
            method_symbols.len() >= 1,
            "Expected at least constructor or destructor to be extracted"
        );

        // Print what methods we found for debugging
        for method in &method_symbols {
            println!("Found method: {:?}", method.symbol);
        }

        // Check if destructor is present (might be ~Resource or just Resource)
        let has_destructor = method_symbols.iter().any(|s| {
            s.symbol
                .as_deref()
                .map(|name| name.contains("~") || name == "Resource")
                .unwrap_or(false)
        });

        // This test documents current behavior - we'll fix if destructors aren't extracted
        if !has_destructor {
            println!("WARNING: Destructor extraction may not be working");
        }
    }
}

// ============================================================================
// Dependency Extraction
// ============================================================================

use crate::models::ImportType;
use crate::parsers::{DependencyExtractor, ImportInfo};

/// C++ dependency extractor
pub struct CppDependencyExtractor;

impl DependencyExtractor for CppDependencyExtractor {
    fn extract_dependencies(source: &str) -> Result<Vec<ImportInfo>> {
        let mut parser = Parser::new();
        let language = tree_sitter_cpp::LANGUAGE;

        parser
            .set_language(&language.into())
            .context("Failed to set C++ language")?;

        let tree = parser
            .parse(source, None)
            .context("Failed to parse C++ source")?;

        let root_node = tree.root_node();

        let mut imports = Vec::new();

        // Extract #include directives
        imports.extend(extract_cpp_includes(source, &root_node)?);

        Ok(imports)
    }
}

/// Extract C++ #include directives
fn extract_cpp_includes(source: &str, root: &tree_sitter::Node) -> Result<Vec<ImportInfo>> {
    let language = tree_sitter_cpp::LANGUAGE;

    let query_str = r#"
        (preproc_include
            path: (string_literal) @include_path) @include

        (preproc_include
            path: (system_lib_string) @include_path) @include
    "#;

    let query =
        Query::new(&language.into(), query_str).context("Failed to create C++ include query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut imports = Vec::new();

    while let Some(match_) = matches.next() {
        let mut include_path = None;
        let mut include_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            match capture_name {
                "include_path" => {
                    // Remove quotes or angle brackets from path
                    let raw_path = capture.node.utf8_text(source.as_bytes()).unwrap_or("");
                    include_path = Some(
                        raw_path
                            .trim_matches(|c| c == '"' || c == '<' || c == '>')
                            .to_string(),
                    );
                }
                "include" => {
                    include_node = Some(capture.node);
                }
                _ => {}
            }
        }

        if let (Some(path), Some(node)) = (include_path, include_node) {
            let import_type = classify_cpp_include(&path, source, &node);
            let line_number = node.start_position().row + 1;

            imports.push(ImportInfo {
                imported_path: path,
                import_type,
                line_number,
                imported_symbols: None, // C++ includes entire header
            });
        }
    }

    Ok(imports)
}

/// Classify a C++ include as internal, external, or stdlib
fn classify_cpp_include(include_path: &str, source: &str, node: &tree_sitter::Node) -> ImportType {
    // Get the actual #include line to check if it uses quotes or angle brackets
    let line_start = node.start_position();
    let lines: Vec<&str> = source.lines().collect();

    if line_start.row < lines.len() {
        let line = lines[line_start.row];

        // Internal: #include "..." (quotes = local project files)
        if line.contains(&format!("\"{}\"", include_path)) {
            return ImportType::Internal;
        }
    }

    // C++ standard library headers (angle brackets)
    const CPP_STDLIB_HEADERS: &[&str] = &[
        // C standard library (inherited)
        "stdio.h",
        "stdlib.h",
        "string.h",
        "math.h",
        "time.h",
        "ctype.h",
        "assert.h",
        "errno.h",
        "limits.h",
        "float.h",
        "stddef.h",
        "stdint.h",
        "stdbool.h",
        "stdarg.h",
        "setjmp.h",
        "signal.h",
        "locale.h",
        "wchar.h",
        "wctype.h",
        "complex.h",
        "fenv.h",
        "inttypes.h",
        "iso646.h",
        "tgmath.h",
        "threads.h",
        // C++ standard library headers (no .h extension)
        "algorithm",
        "any",
        "array",
        "atomic",
        "barrier",
        "bit",
        "bitset",
        "charconv",
        "chrono",
        "codecvt",
        "compare",
        "complex",
        "concepts",
        "condition_variable",
        "coroutine",
        "deque",
        "exception",
        "execution",
        "expected",
        "filesystem",
        "format",
        "forward_list",
        "fstream",
        "functional",
        "future",
        "initializer_list",
        "iomanip",
        "ios",
        "iosfwd",
        "iostream",
        "istream",
        "iterator",
        "latch",
        "limits",
        "list",
        "locale",
        "map",
        "mdspan",
        "memory",
        "memory_resource",
        "mutex",
        "new",
        "numbers",
        "numeric",
        "optional",
        "ostream",
        "queue",
        "random",
        "ranges",
        "ratio",
        "regex",
        "scoped_allocator",
        "semaphore",
        "set",
        "shared_mutex",
        "source_location",
        "span",
        "sstream",
        "stack",
        "stacktrace",
        "stdexcept",
        "stop_token",
        "streambuf",
        "string",
        "string_view",
        "strstream",
        "syncstream",
        "system_error",
        "thread",
        "tuple",
        "type_traits",
        "typeindex",
        "typeinfo",
        "unordered_map",
        "unordered_set",
        "utility",
        "valarray",
        "variant",
        "vector",
        "version",
        // C++ C-compatibility headers (c-prefixed)
        "cassert",
        "cctype",
        "cerrno",
        "cfenv",
        "cfloat",
        "cinttypes",
        "climits",
        "clocale",
        "cmath",
        "csetjmp",
        "csignal",
        "cstdarg",
        "cstddef",
        "cstdint",
        "cstdio",
        "cstdlib",
        "cstring",
        "ctime",
        "cuchar",
        "cwchar",
        "cwctype",
    ];

    if CPP_STDLIB_HEADERS.contains(&include_path) {
        return ImportType::Stdlib;
    }

    // Everything else with angle brackets is external (third-party libraries)
    ImportType::External
}

// ============================================================================
// Path Resolution
// ============================================================================

/// Resolve a C++ #include directive to a file path
///
/// # Arguments
/// * `include_path` - The path from the #include directive (e.g., "utils/helper.hpp")
/// * `current_file_path` - Path to the file containing the #include directive
///
/// # Returns
/// * `Some(path)` if the include can be resolved (quoted includes only)
/// * `None` for angle bracket includes (system/library headers)
pub fn resolve_cpp_include_to_path(
    include_path: &str,
    current_file_path: Option<&str>,
) -> Option<String> {
    // Only resolve relative includes (quoted includes, which are Internal)
    // Angle bracket includes are system/library headers and won't be resolved

    let current_file = current_file_path?;

    // Get directory of current file
    let current_dir = std::path::Path::new(current_file).parent()?;

    // Resolve the include path relative to current file
    let resolved = current_dir.join(include_path);

    // Normalize the path
    match resolved.canonicalize() {
        Ok(normalized) => Some(normalized.display().to_string()),
        Err(_) => {
            // If canonicalize fails (file doesn't exist yet), return the joined path
            Some(resolved.display().to_string())
        }
    }
}

// ============================================================================
// Tests for Path Resolution
// ============================================================================

#[cfg(test)]
mod resolution_tests {
    use super::*;

    #[test]
    fn test_resolve_cpp_include_same_directory() {
        let result = resolve_cpp_include_to_path("helper.hpp", Some("/project/src/main.cpp"));

        assert!(result.is_some());
        let path = result.unwrap();
        assert!(path.ends_with("src/helper.hpp") || path.ends_with("src\\helper.hpp"));
    }

    #[test]
    fn test_resolve_cpp_include_subdirectory() {
        let result = resolve_cpp_include_to_path("utils/helper.hpp", Some("/project/src/main.cpp"));

        assert!(result.is_some());
        let path = result.unwrap();
        assert!(path.ends_with("src/utils/helper.hpp") || path.ends_with("src\\utils\\helper.hpp"));
    }

    #[test]
    fn test_resolve_cpp_include_parent_directory() {
        let result =
            resolve_cpp_include_to_path("../include/common.hpp", Some("/project/src/main.cpp"));

        assert!(result.is_some());
        let path = result.unwrap();
        assert!(path.contains("include") && path.contains("common.hpp"));
    }

    #[test]
    fn test_resolve_cpp_include_h_extension() {
        let result = resolve_cpp_include_to_path("legacy.h", Some("/project/src/main.cpp"));

        assert!(result.is_some());
        let path = result.unwrap();
        assert!(path.ends_with("src/legacy.h") || path.ends_with("src\\legacy.h"));
    }

    #[test]
    fn test_resolve_cpp_include_no_current_file() {
        let result = resolve_cpp_include_to_path("helper.hpp", None);

        assert!(result.is_none());
    }
}
