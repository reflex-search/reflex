//! C language parser using Tree-sitter
//!
//! Extracts symbols from C source code:
//! - Functions (declarations and definitions)
//! - Structs
//! - Enums
//! - Unions
//! - Typedefs
//! - Variables (global, local, static, extern)
//! - Macros (#define for function-like and constant macros)

use crate::models::{Language, SearchResult, Span, SymbolKind};
use anyhow::{Context, Result};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};

/// Parse C source code and extract symbols
pub fn parse(path: &str, source: &str) -> Result<Vec<SearchResult>> {
    let mut parser = Parser::new();
    let language = tree_sitter_c::LANGUAGE;

    parser
        .set_language(&language.into())
        .context("Failed to set C language")?;

    let tree = parser
        .parse(source, None)
        .context("Failed to parse C source")?;

    let root_node = tree.root_node();

    let mut symbols = Vec::new();

    // Extract different types of symbols using Tree-sitter queries
    symbols.extend(extract_functions(source, &root_node, &language.into())?);
    symbols.extend(extract_structs(source, &root_node, &language.into())?);
    symbols.extend(extract_enums(source, &root_node, &language.into())?);
    symbols.extend(extract_unions(source, &root_node, &language.into())?);
    symbols.extend(extract_typedefs(source, &root_node, &language.into())?);
    symbols.extend(extract_variables(source, &root_node, &language.into())?);
    symbols.extend(extract_macros(source, &root_node, &language.into())?);

    // Add file path to all symbols
    for symbol in &mut symbols {
        symbol.path = path.to_string();
        symbol.lang = Language::C;
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
            declarator: (pointer_declarator
                declarator: (function_declarator
                    declarator: (identifier) @name))) @function
    "#;

    let query = Query::new(language, query_str).context("Failed to create function query")?;

    extract_symbols(source, root, &query, SymbolKind::Function, None)
}

/// Extract struct definitions
fn extract_structs(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (struct_specifier
            name: (type_identifier) @name) @struct
    "#;

    let query = Query::new(language, query_str).context("Failed to create struct query")?;

    extract_symbols(source, root, &query, SymbolKind::Struct, None)
}

/// Extract enum definitions
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

/// Extract union definitions
fn extract_unions(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (union_specifier
            name: (type_identifier) @name) @union
    "#;

    let query = Query::new(language, query_str).context("Failed to create union query")?;

    extract_symbols(source, root, &query, SymbolKind::Type, None)
}

/// Extract typedef declarations
fn extract_typedefs(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (type_definition
            declarator: (type_identifier) @name) @typedef
    "#;

    let query = Query::new(language, query_str).context("Failed to create typedef query")?;

    extract_symbols(source, root, &query, SymbolKind::Type, None)
}

/// Extract variable declarations (global and local)
fn extract_variables(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (declaration
            declarator: (init_declarator
                declarator: (identifier) @name)) @var

        (declaration
            declarator: (identifier) @name) @var
    "#;

    let query = Query::new(language, query_str).context("Failed to create variable query")?;

    // Extract all variable declarations (global and local)
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut name = None;
        let mut var_node = None;

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
            } else if capture_name == "var" {
                var_node = Some(capture.node);
            }
        }

        if let (Some(name), Some(node)) = (name, var_node) {
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::C,
                SymbolKind::Variable,
                Some(name),
                span,
                None,
                preview,
            ));
        }
    }

    Ok(symbols)
}

/// Extract macro definitions (#define)
fn extract_macros(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (preproc_def
            name: (identifier) @name) @macro

        (preproc_function_def
            name: (identifier) @name) @macro
    "#;

    let query = Query::new(language, query_str).context("Failed to create macro query")?;

    extract_symbols(source, root, &query, SymbolKind::Macro, None)
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
                Language::C,
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

        let symbols = parse("test.c", source).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("add"));
        assert!(matches!(symbols[0].kind, SymbolKind::Function));
    }

    #[test]
    fn test_parse_struct() {
        let source = r#"
struct User {
    char name[50];
    int age;
};
        "#;

        let symbols = parse("test.c", source).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("User"));
        assert!(matches!(symbols[0].kind, SymbolKind::Struct));
    }

    #[test]
    fn test_parse_enum() {
        let source = r#"
enum Status {
    STATUS_ACTIVE,
    STATUS_INACTIVE,
    STATUS_PENDING
};
        "#;

        let symbols = parse("test.c", source).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("Status"));
        assert!(matches!(symbols[0].kind, SymbolKind::Enum));
    }

    #[test]
    fn test_parse_typedef() {
        let source = r#"
typedef struct {
    int x;
    int y;
} Point;

typedef int UserID;
        "#;

        let symbols = parse("test.c", source).unwrap();

        let typedef_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Type))
            .collect();

        assert!(typedef_symbols.len() >= 1);
        assert!(
            typedef_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Point"))
        );
    }

    #[test]
    fn test_parse_union() {
        let source = r#"
union Data {
    int i;
    float f;
    char str[20];
};
        "#;

        let symbols = parse("test.c", source).unwrap();

        let union_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Type))
            .collect();

        assert_eq!(union_symbols.len(), 1);
        assert_eq!(union_symbols[0].symbol.as_deref(), Some("Data"));
    }

    #[test]
    fn test_parse_global_variables() {
        let source = r#"
int global_counter = 0;
static int internal_state;
extern int external_value;
        "#;

        let symbols = parse("test.c", source).unwrap();

        let var_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        assert_eq!(var_symbols.len(), 3);
        assert!(
            var_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("global_counter"))
        );
        assert!(
            var_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("internal_state"))
        );
        assert!(
            var_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("external_value"))
        );
    }

    #[test]
    fn test_parse_pointer_function() {
        let source = r#"
int* create_array(int size) {
    return malloc(size * sizeof(int));
}
        "#;

        let symbols = parse("test.c", source).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("create_array"));
        assert!(matches!(symbols[0].kind, SymbolKind::Function));
    }

    #[test]
    fn test_parse_mixed_symbols() {
        let source = r#"
#include <stdio.h>

#define MAX_SIZE 100

typedef struct {
    char name[50];
    int age;
} Person;

enum Color {
    RED,
    GREEN,
    BLUE
};

int global_count = 0;

int increment(void) {
    return ++global_count;
}

struct Node {
    int data;
    struct Node* next;
};
        "#;

        let symbols = parse("test.c", source).unwrap();

        // Should find: macro, typedef, enum, variable, function, struct
        assert!(symbols.len() >= 6);

        let kinds: Vec<&SymbolKind> = symbols.iter().map(|s| &s.kind).collect();
        assert!(kinds.contains(&&SymbolKind::Macro));
        assert!(kinds.contains(&&SymbolKind::Type));
        assert!(kinds.contains(&&SymbolKind::Enum));
        assert!(kinds.contains(&&SymbolKind::Variable));
        assert!(kinds.contains(&&SymbolKind::Function));
        assert!(kinds.contains(&&SymbolKind::Struct));

        // Verify the macro symbol is found
        let macro_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Macro))
            .collect();
        assert_eq!(macro_symbols.len(), 1);
        assert_eq!(macro_symbols[0].symbol.as_deref(), Some("MAX_SIZE"));
    }

    #[test]
    fn test_parse_struct_with_typedef() {
        let source = r#"
typedef struct Node {
    int value;
    struct Node* next;
} Node;
        "#;

        let symbols = parse("test.c", source).unwrap();

        // Should find both the struct and the typedef
        assert!(symbols.len() >= 1);
        assert!(symbols.iter().any(|s| s.symbol.as_deref() == Some("Node")));
    }

    #[test]
    fn test_local_variables_included() {
        let source = r#"
int global_var = 10;

int calculate(int x) {
    int local_var = x * 2;
    return local_var;
}
        "#;

        let symbols = parse("test.c", source).unwrap();

        let var_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        // Should find both global_var and local_var
        assert_eq!(var_symbols.len(), 2);
        assert!(
            var_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("global_var"))
        );
        assert!(
            var_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("local_var"))
        );
    }

    #[test]
    fn test_parse_macros() {
        let source = r#"
#define MAX_SIZE 100
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DEBUG_PRINT(x) printf("Debug: %s\n", x)

int main() {
    return 0;
}
        "#;

        let symbols = parse("test.c", source).unwrap();

        let macro_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Macro))
            .collect();

        // Should find all three macros
        assert_eq!(macro_symbols.len(), 3);
        assert!(
            macro_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("MAX_SIZE"))
        );
        assert!(
            macro_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("MIN"))
        );
        assert!(
            macro_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("DEBUG_PRINT"))
        );
    }
}

// ============================================================================
// Dependency Extraction
// ============================================================================

use crate::models::ImportType;
use crate::parsers::{DependencyExtractor, ImportInfo};

/// C dependency extractor
pub struct CDependencyExtractor;

impl DependencyExtractor for CDependencyExtractor {
    fn extract_dependencies(source: &str) -> Result<Vec<ImportInfo>> {
        let mut parser = Parser::new();
        let language = tree_sitter_c::LANGUAGE;

        parser
            .set_language(&language.into())
            .context("Failed to set C language")?;

        let tree = parser
            .parse(source, None)
            .context("Failed to parse C source")?;

        let root_node = tree.root_node();

        let mut imports = Vec::new();

        // Extract #include directives
        imports.extend(extract_c_includes(source, &root_node)?);

        Ok(imports)
    }
}

/// Extract C #include directives
fn extract_c_includes(source: &str, root: &tree_sitter::Node) -> Result<Vec<ImportInfo>> {
    let language = tree_sitter_c::LANGUAGE;

    let query_str = r#"
        (preproc_include
            path: (string_literal) @include_path) @include

        (preproc_include
            path: (system_lib_string) @include_path) @include
    "#;

    let query =
        Query::new(&language.into(), query_str).context("Failed to create C include query")?;

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
            let import_type = classify_c_include(&path, source, &node);
            let line_number = node.start_position().row + 1;

            imports.push(ImportInfo {
                imported_path: path,
                import_type,
                line_number,
                imported_symbols: None, // C includes entire header
            });
        }
    }

    Ok(imports)
}

/// Classify a C include as internal, external, or stdlib
fn classify_c_include(include_path: &str, source: &str, node: &tree_sitter::Node) -> ImportType {
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

    // C standard library headers (angle brackets)
    const STDLIB_HEADERS: &[&str] = &[
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
    ];

    if STDLIB_HEADERS.contains(&include_path) {
        return ImportType::Stdlib;
    }

    // Everything else with angle brackets is external (third-party libraries)
    ImportType::External
}

// ============================================================================
// Path Resolution
// ============================================================================

/// Resolve a C #include directive to a file path
///
/// # Arguments
/// * `include_path` - The path from the #include directive (e.g., "utils/helper.h")
/// * `current_file_path` - Path to the file containing the #include directive
///
/// # Returns
/// * `Some(path)` if the include can be resolved (quoted includes only)
/// * `None` for angle bracket includes (system/library headers)
pub fn resolve_c_include_to_path(
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
    fn test_resolve_c_include_same_directory() {
        let result = resolve_c_include_to_path("helper.h", Some("/project/src/main.c"));

        assert!(result.is_some());
        let path = result.unwrap();
        assert!(path.ends_with("src/helper.h") || path.ends_with("src\\helper.h"));
    }

    #[test]
    fn test_resolve_c_include_subdirectory() {
        let result = resolve_c_include_to_path("utils/helper.h", Some("/project/src/main.c"));

        assert!(result.is_some());
        let path = result.unwrap();
        assert!(path.ends_with("src/utils/helper.h") || path.ends_with("src\\utils\\helper.h"));
    }

    #[test]
    fn test_resolve_c_include_parent_directory() {
        let result = resolve_c_include_to_path("../include/common.h", Some("/project/src/main.c"));

        assert!(result.is_some());
        let path = result.unwrap();
        assert!(path.contains("include") && path.contains("common.h"));
    }

    #[test]
    fn test_resolve_c_include_no_current_file() {
        let result = resolve_c_include_to_path("helper.h", None);

        assert!(result.is_none());
    }
}

#[cfg(test)]
mod dependency_extraction_tests {
    use super::*;

    #[test]
    fn test_extract_basic_includes() {
        let source = r#"
            #include <stdio.h>
            #include <stdlib.h>
            #include "utils.h"
            #include "math/vector.h"
        "#;

        let deps = CDependencyExtractor::extract_dependencies(source).unwrap();

        assert_eq!(deps.len(), 4, "Should extract 4 include statements");
        assert!(deps.iter().any(|d| d.imported_path == "stdio.h"));
        assert!(deps.iter().any(|d| d.imported_path == "stdlib.h"));
        assert!(deps.iter().any(|d| d.imported_path == "utils.h"));
        assert!(deps.iter().any(|d| d.imported_path == "math/vector.h"));
    }

    #[test]
    fn test_macro_includes_filtered() {
        let source = r#"
            #include <stdio.h>
            #include "config.h"

            // Macro-based includes - should be filtered out
            #define HEADER_NAME "dynamic.h"
            #include HEADER_NAME

            #define STRINGIFY(x) #x
            #include STRINGIFY(runtime_header.h)

            // Conditional includes with macros
            #ifdef USE_FEATURE_X
            #define FEATURE_HEADER <feature_x.h>
            #include FEATURE_HEADER
            #endif
        "#;

        let deps = CDependencyExtractor::extract_dependencies(source).unwrap();

        // Should only find static includes (stdio.h, config.h)
        // Macro-based includes are filtered (not string_literal or system_lib_string nodes)
        assert_eq!(deps.len(), 2, "Should extract 2 static includes only");

        assert!(deps.iter().any(|d| d.imported_path == "stdio.h"));
        assert!(deps.iter().any(|d| d.imported_path == "config.h"));

        // Verify macro-based includes are NOT captured
        assert!(!deps.iter().any(|d| d.imported_path.contains("HEADER_NAME")));
        assert!(!deps.iter().any(|d| d.imported_path.contains("dynamic.h")));
        assert!(
            !deps
                .iter()
                .any(|d| d.imported_path.contains("runtime_header"))
        );
        assert!(
            !deps
                .iter()
                .any(|d| d.imported_path.contains("FEATURE_HEADER"))
        );
    }

    #[test]
    fn test_include_classification() {
        let source = r#"
            #include <stdio.h>
            #include "utils.h"
            #include <mylib/api.h>
        "#;

        let deps = CDependencyExtractor::extract_dependencies(source).unwrap();

        // Check stdlib classification
        let stdio_dep = deps.iter().find(|d| d.imported_path == "stdio.h").unwrap();
        assert!(
            matches!(stdio_dep.import_type, ImportType::Stdlib),
            "stdio.h should be classified as Stdlib"
        );

        // Check internal classification (quoted includes)
        let utils_dep = deps.iter().find(|d| d.imported_path == "utils.h").unwrap();
        assert!(
            matches!(utils_dep.import_type, ImportType::Internal),
            "quoted include should be classified as Internal"
        );

        // Check external classification (non-stdlib angle bracket includes)
        let mylib_dep = deps
            .iter()
            .find(|d| d.imported_path == "mylib/api.h")
            .unwrap();
        assert!(
            matches!(mylib_dep.import_type, ImportType::External),
            "non-stdlib angle bracket include should be classified as External"
        );
    }
}
