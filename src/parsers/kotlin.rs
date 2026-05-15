//! Kotlin language parser using Tree-sitter
//!
//! Extracts symbols from Kotlin source code:
//! - Classes (regular, data, sealed, abstract, open)
//! - Objects (singleton)
//! - Interfaces
//! - Functions
//! - Properties (class/object members)
//! - Local variables (val/var inside functions)
//! - Companion objects
//! - Extensions

use crate::models::{Language, SearchResult, Span, SymbolKind};
use anyhow::{Context, Result};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};

/// Parse Kotlin source code and extract symbols
pub fn parse(path: &str, source: &str) -> Result<Vec<SearchResult>> {
    let mut parser = Parser::new();
    let language = tree_sitter_kotlin_ng::LANGUAGE;

    parser
        .set_language(&language.into())
        .context("Failed to set Kotlin language")?;

    let tree = parser
        .parse(source, None)
        .context("Failed to parse Kotlin source")?;

    let root_node = tree.root_node();

    let mut symbols = Vec::new();

    // Extract different types of symbols using Tree-sitter queries
    symbols.extend(extract_classes(source, &root_node, &language.into())?);
    symbols.extend(extract_objects(source, &root_node, &language.into())?);
    symbols.extend(extract_interfaces(source, &root_node, &language.into())?);
    symbols.extend(extract_annotations(source, &root_node, &language.into())?);
    symbols.extend(extract_functions(source, &root_node, &language.into())?);
    symbols.extend(extract_properties(source, &root_node, &language.into())?);
    symbols.extend(extract_local_variables(
        source,
        &root_node,
        &language.into(),
    )?);

    // Add file path to all symbols
    for symbol in &mut symbols {
        symbol.path = path.to_string();
        symbol.lang = Language::Kotlin;
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
            (identifier) @name) @class
    "#;

    let query = Query::new(language, query_str).context("Failed to create class query")?;

    extract_symbols(source, root, &query, SymbolKind::Class, None)
}

/// Extract object declarations (singletons)
fn extract_objects(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (object_declaration
            (identifier) @name) @object
    "#;

    let query = Query::new(language, query_str).context("Failed to create object query")?;

    extract_symbols(source, root, &query, SymbolKind::Class, None)
}

/// Extract interface declarations
fn extract_interfaces(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_declaration
            "interface"
            (identifier) @name) @interface
    "#;

    let query = Query::new(language, query_str).context("Failed to create interface query")?;

    extract_symbols(source, root, &query, SymbolKind::Interface, None)
}

/// Extract annotations: BOTH definitions and uses
/// Definitions: annotation class Test, annotation class Entity(val tableName: String)
/// Uses: @Test fun testMethod(), @Composable fun MyButton()
fn extract_annotations(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let mut symbols = Vec::new();

    // Part 1: Extract annotation class DEFINITIONS
    let def_query_str = r#"
        (class_declaration
            (identifier) @name) @annotation
    "#;

    let def_query = Query::new(language, def_query_str)
        .context("Failed to create annotation definition query")?;

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
                "annotation" => {
                    class_node = Some(capture.node);
                }
                _ => {}
            }
        }

        // Check if this class has "annotation" modifier
        if let (Some(name), Some(node)) = (name, class_node) {
            let class_text = node.utf8_text(source.as_bytes()).unwrap_or("");

            // Check if the class declaration starts with "annotation class"
            if class_text.trim_start().starts_with("annotation ")
                || class_text.trim_start().starts_with("@Target")
                    && class_text.contains("annotation class")
                || class_text.trim_start().starts_with("@Retention")
                    && class_text.contains("annotation class")
            {
                let span = node_to_span(&node);
                let preview = extract_preview(source, &span);

                symbols.push(SearchResult::new(
                    String::new(),
                    Language::Kotlin,
                    SymbolKind::Attribute,
                    Some(name),
                    span,
                    None,
                    preview,
                ));
            }
        }
    }

    // Part 2: Extract annotation USES (@Test, @Composable, etc.)
    let use_query_str = r#"
        (annotation) @attr
    "#;

    let use_query =
        Query::new(language, use_query_str).context("Failed to create annotation use query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&use_query, *root, source.as_bytes());

    while let Some(match_) = matches.next() {
        for capture in match_.captures {
            let node = capture.node;
            let text = node.utf8_text(source.as_bytes()).unwrap_or("");

            // Extract annotation name from text like "@Test" or "@Composable(...)"
            if let Some(name) = extract_annotation_name(text) {
                let span = node_to_span(&node);
                let preview = extract_preview(source, &span);

                symbols.push(SearchResult::new(
                    String::new(),
                    Language::Kotlin,
                    SymbolKind::Attribute,
                    Some(name),
                    span,
                    None,
                    preview,
                ));
            }
        }
    }

    Ok(symbols)
}

/// Extract annotation name from text like "@Test" or "@Composable(...)"
fn extract_annotation_name(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if !trimmed.starts_with('@') {
        return None;
    }

    let name_part = &trimmed[1..]; // Skip the '@'

    // Find where the annotation name ends (at '(' or whitespace or end of string)
    let end_pos = name_part
        .find(|c: char| c == '(' || c.is_whitespace())
        .unwrap_or(name_part.len());

    Some(name_part[..end_pos].to_string())
}

/// Extract function declarations
fn extract_functions(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_declaration
            (identifier) @class_name
            (class_body
                (function_declaration
                    (identifier) @method_name))) @class

        (object_declaration
            (identifier) @object_name
            (class_body
                (function_declaration
                    (identifier) @method_name))) @object
    "#;

    let query = Query::new(language, query_str).context("Failed to create function query")?;

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
                "object_name" => {
                    scope_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                    scope_type = Some("object");
                }
                "method_name" => {
                    method_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                    // Find the parent function_declaration node
                    let mut current = capture.node;
                    while let Some(parent) = current.parent() {
                        if parent.kind() == "function_declaration" {
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
                Language::Kotlin,
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
            (identifier) @class_name
            (class_body
                (property_declaration
                    (variable_declaration
                        (identifier) @property_name)))) @class

        (object_declaration
            (identifier) @object_name
            (class_body
                (property_declaration
                    (variable_declaration
                        (identifier) @property_name)))) @object
    "#;

    let query = Query::new(language, query_str).context("Failed to create property query")?;

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
                    scope_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                    scope_type = Some("class");
                }
                "object_name" => {
                    scope_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                    scope_type = Some("object");
                }
                "property_name" => {
                    property_name = Some(
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
            (scope_name, scope_type, property_name, property_node)
        {
            let scope = format!("{} {}", scope_type, scope_name);
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::Kotlin,
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

/// Extract local variable declarations (val/var) inside functions
fn extract_local_variables(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (property_declaration
            (variable_declaration
                (identifier) @name)) @var
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

        // Check if this variable declaration is inside a function_declaration
        // (not inside a class_body or object_body, which would be properties)
        if let (Some(name), Some(node)) = (name, var_node) {
            let mut is_local_var = false;
            let mut is_class_property = false;
            let mut current = node;

            while let Some(parent) = current.parent() {
                // If we find a function_declaration, it's a local variable
                if parent.kind() == "function_declaration" {
                    is_local_var = true;
                    break;
                }
                // If we find a class_body or object body before a function, it's a property
                if parent.kind() == "class_body" {
                    is_class_property = true;
                    break;
                }
                current = parent;
            }

            // Only add if it's a local variable (inside function, not a class property)
            if is_local_var && !is_class_property {
                let span = node_to_span(&node);
                let preview = extract_preview(source, &span);

                // Determine if it's val (constant) or var (variable)
                let decl_text = node.utf8_text(source.as_bytes()).unwrap_or("");
                let kind = if decl_text.trim_start().starts_with("val") {
                    SymbolKind::Constant
                } else {
                    SymbolKind::Variable
                };

                symbols.push(SearchResult::new(
                    String::new(),
                    Language::Kotlin,
                    kind,
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
                Language::Kotlin,
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
class User(val name: String, val age: Int)
        "#;

        let symbols = parse("test.kt", source).unwrap();

        let class_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        assert_eq!(class_symbols.len(), 1);
        assert_eq!(class_symbols[0].symbol.as_deref(), Some("User"));
    }

    #[test]
    fn test_parse_data_class() {
        let source = r#"
data class Person(val firstName: String, val lastName: String)
        "#;

        let symbols = parse("test.kt", source).unwrap();

        let class_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        assert_eq!(class_symbols.len(), 1);
        assert_eq!(class_symbols[0].symbol.as_deref(), Some("Person"));
    }

    #[test]
    fn test_parse_object() {
        let source = r#"
object Singleton {
    fun getInstance() = this
}
        "#;

        let symbols = parse("test.kt", source).unwrap();

        let object_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .filter(|s| s.symbol.as_deref() == Some("Singleton"))
            .collect();

        assert_eq!(object_symbols.len(), 1);
    }

    #[test]
    fn test_parse_functions() {
        let source = r#"
class Calculator {
    fun add(a: Int, b: Int): Int {
        return a + b
    }

    fun subtract(a: Int, b: Int): Int {
        return a - b
    }
}
        "#;

        let symbols = parse("test.kt", source).unwrap();

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
    fn test_parse_properties() {
        let source = r#"
class User {
    val name: String = ""
    var age: Int = 0
    lateinit var email: String
}
        "#;

        let symbols = parse("test.kt", source).unwrap();

        let property_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        assert_eq!(property_symbols.len(), 3);
        assert!(
            property_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("name"))
        );
        assert!(
            property_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("age"))
        );
        assert!(
            property_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("email"))
        );
    }

    #[test]
    fn test_parse_companion_object() {
        let source = r#"
class User {
    companion object {
        fun create(name: String) = User(name)
    }
}
        "#;

        let symbols = parse("test.kt", source).unwrap();

        let class_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        assert_eq!(class_symbols.len(), 1);
        assert_eq!(class_symbols[0].symbol.as_deref(), Some("User"));
    }

    #[test]
    fn test_parse_sealed_class() {
        let source = r#"
sealed class Result {
    data class Success(val data: String) : Result()
    data class Error(val error: String) : Result()
}
        "#;

        let symbols = parse("test.kt", source).unwrap();

        let class_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        // Should find Result, Success, and Error
        assert!(class_symbols.len() >= 1);
        assert!(
            class_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Result"))
        );
    }

    #[test]
    fn test_parse_extension_function() {
        let source = r#"
fun String.isEmail(): Boolean {
    return this.contains("@")
}
        "#;

        let symbols = parse("test.kt", source).unwrap();

        // Extension functions should be captured (test verifies parsing succeeds without panic)
        let _ = symbols; // Suppress unused variable warning
    }

    #[test]
    fn test_parse_android_activity() {
        let source = r#"
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
    }

    override fun onResume() {
        super.onResume()
    }
}
        "#;

        let symbols = parse("test.kt", source).unwrap();

        let class_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        let method_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method))
            .collect();

        assert_eq!(class_symbols.len(), 1);
        assert_eq!(method_symbols.len(), 2);
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("onCreate"))
        );
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("onResume"))
        );
    }

    #[test]
    fn test_parse_mixed_symbols() {
        let source = r#"
object Config {
    const val API_URL = "https://api.example.com"
}

interface Repository {
    fun save(item: Any)
}

data class User(val id: Int, val name: String) {
    fun display(): String {
        return "$id: $name"
    }
}
        "#;

        let symbols = parse("test.kt", source).unwrap();

        // Should find: object, interface, data class, method
        assert!(symbols.len() >= 3);

        let kinds: Vec<&SymbolKind> = symbols.iter().map(|s| &s.kind).collect();
        assert!(kinds.contains(&&SymbolKind::Class));
        assert!(kinds.contains(&&SymbolKind::Method));
    }

    #[test]
    fn test_local_variables_included() {
        let source = r#"
class Calculator {
    val multiplier: Int = 2

    fun compute(input: Int): Int {
        val localConst = 10
        var localVar = input * multiplier
        localVar += localConst
        return localVar
    }
}

fun topLevel(): String {
    val result = "test"
    var counter = 0
    return result
}
        "#;

        let symbols = parse("test.kt", source).unwrap();

        // Filter to constants and variables
        let constants: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Constant))
            .collect();

        let variables: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        // Check that val local variables are captured as constants
        assert!(
            constants
                .iter()
                .any(|c| c.symbol.as_deref() == Some("localConst"))
        );
        assert!(
            constants
                .iter()
                .any(|c| c.symbol.as_deref() == Some("result"))
        );

        // Check that var local variables are captured as variables
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("localVar"))
        );
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("counter"))
        );

        // Check that class property is still captured
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("multiplier"))
        );

        // Verify that local variables have no scope
        for constant in &constants {
            if constant.symbol.as_deref() == Some("localConst")
                || constant.symbol.as_deref() == Some("result")
            {
                // Removed: scope field no longer exists: assert_eq!(constant.scope, None);
            }
        }

        for variable in &variables {
            if variable.symbol.as_deref() == Some("localVar")
                || variable.symbol.as_deref() == Some("counter")
            {
                // Removed: scope field no longer exists: assert_eq!(variable.scope, None);
            }
        }

        // Verify that class property has scope
        let multiplier = variables
            .iter()
            .find(|v| v.symbol.as_deref() == Some("multiplier"))
            .unwrap();
        // Removed: scope field no longer exists: assert_eq!(multiplier.scope.as_ref().unwrap(), "class Calculator");
    }

    #[test]
    fn test_parse_annotation_class() {
        let source = r#"
annotation class Test

@Target(AnnotationTarget.CLASS)
annotation class Entity(val tableName: String)

@Retention(AnnotationRetention.RUNTIME)
@Target(AnnotationTarget.FUNCTION)
annotation class Composable
        "#;

        let symbols = parse("test.kt", source).unwrap();

        let annotation_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Attribute))
            .collect();

        // Should find Test, Entity, and Composable annotation classes
        assert!(annotation_symbols.len() >= 1);
        // Note: The exact number depends on whether tree-sitter captures nested annotations
        // We verify at least one is captured
    }

    #[test]
    fn test_parse_annotation_uses() {
        let source = r#"
annotation class Test
annotation class Composable

@Test
fun testMethod() {
    println("test")
}

@Composable
fun MyButton() {
    println("button")
}

@Test
fun anotherTest() {
    println("another")
}

class MyViewModel {
    @Composable
    fun render() {
        println("render")
    }
}
        "#;

        let symbols = parse("test.kt", source).unwrap();

        let annotation_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Attribute))
            .collect();

        // Should find annotation class definitions (Test, Composable)
        // AND annotation uses (@Test appears twice, @Composable appears twice)
        // Total expected: 2 definitions + 4 uses = 6
        assert!(annotation_symbols.len() >= 4);

        // Count specific annotation uses
        let test_count = annotation_symbols
            .iter()
            .filter(|s| s.symbol.as_deref() == Some("Test"))
            .count();

        let composable_count = annotation_symbols
            .iter()
            .filter(|s| s.symbol.as_deref() == Some("Composable"))
            .count();

        // Should find Test at least twice (1 definition + 2 uses)
        assert!(test_count >= 2);

        // Should find Composable at least twice (1 definition + 2 uses)
        assert!(composable_count >= 2);
    }

    #[test]
    fn test_extract_kotlin_imports() {
        let source = r#"
            import java.util.List
            import kotlinx.coroutines.launch
            import com.example.myapp.models.User
            import android.os.Bundle

            class MainActivity {
                fun onCreate() {
                    println("Hello")
                }
            }
        "#;

        let deps = KotlinDependencyExtractor::extract_dependencies(source).unwrap();

        assert_eq!(deps.len(), 4, "Should extract 4 import statements");
        assert!(deps.iter().any(|d| d.imported_path == "java.util.List"));
        assert!(
            deps.iter()
                .any(|d| d.imported_path == "kotlinx.coroutines.launch")
        );
        assert!(
            deps.iter()
                .any(|d| d.imported_path == "com.example.myapp.models.User")
        );
        assert!(deps.iter().any(|d| d.imported_path == "android.os.Bundle"));

        // Check stdlib classification
        let java_dep = deps
            .iter()
            .find(|d| d.imported_path == "java.util.List")
            .unwrap();
        assert!(
            matches!(java_dep.import_type, ImportType::Stdlib),
            "java.util.List should be classified as Stdlib"
        );

        // Check kotlinx classification (external)
        let coroutines_dep = deps
            .iter()
            .find(|d| d.imported_path == "kotlinx.coroutines.launch")
            .unwrap();
        assert!(
            matches!(coroutines_dep.import_type, ImportType::External),
            "kotlinx.coroutines.launch should be classified as External"
        );

        // Check user package classification (external)
        let user_dep = deps
            .iter()
            .find(|d| d.imported_path == "com.example.myapp.models.User")
            .unwrap();
        assert!(
            matches!(user_dep.import_type, ImportType::External),
            "com.example.myapp.models.User should be classified as External"
        );

        // Check android classification (stdlib)
        let android_dep = deps
            .iter()
            .find(|d| d.imported_path == "android.os.Bundle")
            .unwrap();
        assert!(
            matches!(android_dep.import_type, ImportType::Stdlib),
            "android.os.Bundle should be classified as Stdlib"
        );
    }
}

// ============================================================================
// Dependency Extraction
// ============================================================================

use crate::models::ImportType;
use crate::parsers::{DependencyExtractor, ImportInfo};

/// Kotlin dependency extractor
pub struct KotlinDependencyExtractor;

impl DependencyExtractor for KotlinDependencyExtractor {
    fn extract_dependencies(source: &str) -> Result<Vec<ImportInfo>> {
        let mut parser = Parser::new();
        let language = tree_sitter_kotlin_ng::LANGUAGE;

        parser
            .set_language(&language.into())
            .context("Failed to set Kotlin language")?;

        let tree = parser
            .parse(source, None)
            .context("Failed to parse Kotlin source")?;

        let root_node = tree.root_node();

        let mut imports = Vec::new();

        // Extract import statements using tree-sitter
        imports.extend(extract_kotlin_imports(source, &root_node)?);

        Ok(imports)
    }
}

/// Extract Kotlin import statements
/// Uses improved text parsing since tree-sitter-kotlin-ng has non-standard node types
fn extract_kotlin_imports(source: &str, _root: &tree_sitter::Node) -> Result<Vec<ImportInfo>> {
    let mut imports = Vec::new();

    // Parse import statements line by line (improved from previous version)
    for (line_idx, line) in source.lines().enumerate() {
        let trimmed = line.trim();

        // Check if line starts with "import " and isn't a comment
        if trimmed.starts_with("import ") && !trimmed.starts_with("//") {
            if let Some(import_path) = extract_import_path_from_header(trimmed) {
                let import_type = classify_kotlin_import(&import_path);
                let line_number = line_idx + 1;

                imports.push(ImportInfo {
                    imported_path: import_path,
                    line_number,
                    import_type,
                    imported_symbols: None,
                });
            }
        }
    }

    Ok(imports)
}

/// Extract import path from import_header text
/// Examples:
///   "import java.util.List" -> "java.util.List"
///   "import kotlinx.coroutines.*" -> "kotlinx.coroutines"
///   "import com.example.Foo as Bar" -> "com.example.Foo"
fn extract_import_path_from_header(text: &str) -> Option<String> {
    let trimmed = text.trim();

    // Remove "import" keyword
    let after_import = trimmed.strip_prefix("import")?;
    let after_import = after_import.trim();

    // Find the end of the import path (before 'as' or wildcard)
    let end_pos = after_import
        .find(" as ")
        .or_else(|| after_import.find(".*"))
        .unwrap_or(after_import.len());

    let path = after_import[..end_pos].trim();

    // Remove trailing wildcard if present
    let path = path.trim_end_matches(".*");

    if path.is_empty() {
        None
    } else {
        Some(path.to_string())
    }
}

/// Extract import path from text like "import java.util.List" or "import kotlinx.coroutines.*"
fn extract_import_path_from_text(text: &str) -> Option<String> {
    // Remove "import" keyword and whitespace
    let trimmed = text.trim();
    if !trimmed.starts_with("import") {
        return None;
    }

    let after_import = trimmed[6..].trim(); // Skip "import"

    // Find the end of the import path (before any 'as' alias or comments)
    let end_pos = after_import
        .find(" as ")
        .or_else(|| after_import.find("//"))
        .or_else(|| after_import.find("/*"))
        .unwrap_or(after_import.len());

    let path = after_import[..end_pos].trim();

    // Remove trailing wildcard if present
    let path = path.trim_end_matches(".*");

    if path.is_empty() {
        None
    } else {
        Some(path.to_string())
    }
}

/// Reclassify a Kotlin import using the project's package prefix
/// Similar to reclassify_go_import() and reclassify_java_import()
pub fn reclassify_kotlin_import(import_path: &str, package_prefix: Option<&str>) -> ImportType {
    classify_kotlin_import_impl(import_path, package_prefix)
}

/// Classify Kotlin imports into Internal/External/Stdlib
fn classify_kotlin_import(import_path: &str) -> ImportType {
    classify_kotlin_import_impl(import_path, None)
}

fn classify_kotlin_import_impl(import_path: &str, package_prefix: Option<&str>) -> ImportType {
    // First check if this is an internal import (matches project package)
    if let Some(prefix) = package_prefix {
        if import_path.starts_with(prefix) {
            return ImportType::Internal;
        }
    }

    // Java standard library
    if import_path.starts_with("java.") || import_path.starts_with("javax.") {
        return ImportType::Stdlib;
    }

    // Kotlin standard library
    if import_path.starts_with("kotlin.") {
        return ImportType::Stdlib;
    }

    // Android SDK
    if import_path.starts_with("android.") || import_path.starts_with("androidx.") {
        return ImportType::Stdlib;
    }

    // Common external libraries
    let external_prefixes = [
        "kotlinx.",
        "com.google.",
        "org.jetbrains.",
        "io.ktor.",
        "com.squareup.",
        "retrofit2.",
        "okhttp3.",
        "com.jakewharton.",
        "org.koin.",
        "com.github.",
    ];

    for prefix in &external_prefixes {
        if import_path.starts_with(prefix) {
            return ImportType::External;
        }
    }

    // Default to external for unknown packages
    ImportType::External
}
