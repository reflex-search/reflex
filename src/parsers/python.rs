//! Python language parser using Tree-sitter
//!
//! Extracts symbols from Python source code:
//! - Functions (def, async def)
//! - Classes (regular, abstract)
//! - Methods (regular, async, static, class methods, properties via @property)
//! - Decorators (tracked in scope)
//! - Lambda expressions assigned to variables
//! - Local variables (inside functions)
//! - Global variables (module-level non-uppercase variables)
//! - Constants (module-level uppercase variables)
//! - Imports/Exports

use crate::models::{Language, SearchResult, Span, SymbolKind};
use anyhow::{Context, Result};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};

/// Parse Python source code and extract symbols
pub fn parse(path: &str, source: &str) -> Result<Vec<SearchResult>> {
    let mut parser = Parser::new();
    let language = tree_sitter_python::LANGUAGE;

    parser
        .set_language(&language.into())
        .context("Failed to set Python language")?;

    let tree = parser
        .parse(source, None)
        .context("Failed to parse Python source")?;

    let root_node = tree.root_node();

    let mut symbols = Vec::new();

    // Extract different types of symbols using Tree-sitter queries
    symbols.extend(extract_functions(source, &root_node, &language.into())?);
    symbols.extend(extract_classes(source, &root_node, &language.into())?);
    symbols.extend(extract_methods(source, &root_node, &language.into())?);
    symbols.extend(extract_constants(source, &root_node, &language.into())?);
    symbols.extend(extract_global_variables(
        source,
        &root_node,
        &language.into(),
    )?);
    symbols.extend(extract_local_variables(
        source,
        &root_node,
        &language.into(),
    )?);
    symbols.extend(extract_lambdas(source, &root_node, &language.into())?);

    // Add file path to all symbols
    for symbol in &mut symbols {
        symbol.path = path.to_string();
        symbol.lang = Language::Python;
    }

    Ok(symbols)
}

/// Extract function definitions (including async functions)
fn extract_functions(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (function_definition
            name: (identifier) @name) @function
    "#;

    let query = Query::new(language, query_str).context("Failed to create function query")?;

    extract_symbols(source, root, &query, SymbolKind::Function, None)
}

/// Extract class definitions
fn extract_classes(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_definition
            name: (identifier) @name) @class
    "#;

    let query = Query::new(language, query_str).context("Failed to create class query")?;

    extract_symbols(source, root, &query, SymbolKind::Class, None)
}

/// Extract method definitions from classes
fn extract_methods(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_definition
            name: (identifier) @class_name
            body: (block
                (function_definition
                    name: (identifier) @method_name))) @class

        (class_definition
            name: (identifier) @class_name
            body: (block
                (decorated_definition
                    (function_definition
                        name: (identifier) @method_name)))) @class
    "#;

    let query = Query::new(language, query_str).context("Failed to create method query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut class_name = None;
        let mut method_name = None;
        let mut method_node = None;

        for capture in match_.captures {
            let capture_name: &str = query.capture_names()[capture.index as usize];
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

        if let (Some(class_name), Some(method_name), Some(node)) =
            (class_name, method_name, method_node)
        {
            let scope = format!("class {}", class_name);
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::Python,
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

/// Extract module-level constants (uppercase variable assignments)
fn extract_constants(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (module
            (expression_statement
                (assignment
                    left: (identifier) @name))) @const
    "#;

    let query = Query::new(language, query_str).context("Failed to create constant query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut name = None;
        let mut const_node = None;

        for capture in match_.captures {
            let capture_name: &str = query.capture_names()[capture.index as usize];
            if capture_name == "name" {
                let name_text = capture.node.utf8_text(source.as_bytes()).unwrap_or("");
                // Only include if it's all uppercase (Python constant convention)
                if name_text
                    .chars()
                    .all(|c| c.is_uppercase() || c == '_' || c.is_numeric())
                {
                    name = Some(name_text.to_string());
                    // Get the assignment node
                    let mut current = capture.node;
                    while let Some(parent) = current.parent() {
                        if parent.kind() == "assignment" {
                            const_node = Some(parent);
                            break;
                        }
                        current = parent;
                    }
                }
            }
        }

        if let (Some(name), Some(node)) = (name, const_node) {
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::Python,
                SymbolKind::Constant,
                Some(name),
                span,
                None,
                preview,
            ));
        }
    }

    Ok(symbols)
}

/// Extract module-level global variables (non-uppercase variable assignments)
fn extract_global_variables(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (module
            (expression_statement
                (assignment
                    left: (identifier) @name))) @var
    "#;

    let query =
        Query::new(language, query_str).context("Failed to create global variable query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut name = None;
        let mut var_node = None;

        for capture in match_.captures {
            let capture_name: &str = query.capture_names()[capture.index as usize];
            if capture_name == "name" {
                let name_text = capture.node.utf8_text(source.as_bytes()).unwrap_or("");
                // Only include if it's NOT all uppercase (constants are handled separately)
                if !name_text
                    .chars()
                    .all(|c| c.is_uppercase() || c == '_' || c.is_numeric())
                {
                    name = Some(name_text.to_string());
                    // Get the assignment node
                    let mut current = capture.node;
                    while let Some(parent) = current.parent() {
                        if parent.kind() == "assignment" {
                            var_node = Some(parent);
                            break;
                        }
                        current = parent;
                    }
                }
            }
        }

        if let (Some(name), Some(node)) = (name, var_node) {
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::Python,
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

/// Extract local variable assignments inside functions
fn extract_local_variables(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (assignment
            left: (identifier) @name) @assignment
    "#;

    let query = Query::new(language, query_str).context("Failed to create local variable query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut name = None;
        let mut assignment_node = None;

        for capture in match_.captures {
            let capture_name: &str = query.capture_names()[capture.index as usize];
            match capture_name {
                "name" => {
                    let name_text = capture.node.utf8_text(source.as_bytes()).unwrap_or("");
                    // Skip uppercase constants (handled by extract_constants)
                    if !name_text
                        .chars()
                        .all(|c| c.is_uppercase() || c == '_' || c.is_numeric())
                    {
                        name = Some(name_text.to_string());
                    }
                }
                "assignment" => {
                    assignment_node = Some(capture.node);
                }
                _ => {}
            }
        }

        // Check if this assignment is inside a function definition
        if let (Some(name), Some(node)) = (name, assignment_node) {
            let mut is_in_function = false;
            let mut current = node;

            while let Some(parent) = current.parent() {
                if parent.kind() == "function_definition" {
                    is_in_function = true;
                    break;
                }
                // Stop if we hit module level
                if parent.kind() == "module" {
                    break;
                }
                current = parent;
            }

            if is_in_function {
                let span = node_to_span(&node);
                let preview = extract_preview(source, &span);

                symbols.push(SearchResult::new(
                    String::new(),
                    Language::Python,
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

/// Extract lambda expressions assigned to variables
fn extract_lambdas(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (assignment
            left: (identifier) @name
            right: (lambda)) @lambda
    "#;

    let query = Query::new(language, query_str).context("Failed to create lambda query")?;

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
            let capture_name: &str = query.capture_names()[capture.index as usize];
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
                Language::Python,
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
    let start_idx = span.start_line - 1; // Convert back to 0-indexed
    let end_idx = (start_idx + 7).min(lines.len());

    lines[start_idx..end_idx].join("\n")
}

// ============================================================================
// Dependency Extraction
// ============================================================================

use crate::models::ImportType;
use crate::parsers::{DependencyExtractor, ImportInfo};

/// Python dependency extractor
pub struct PythonDependencyExtractor;

impl DependencyExtractor for PythonDependencyExtractor {
    fn extract_dependencies(source: &str) -> Result<Vec<ImportInfo>> {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;

        parser
            .set_language(&language.into())
            .context("Failed to set Python language")?;

        let tree = parser
            .parse(source, None)
            .context("Failed to parse Python source")?;

        let root_node = tree.root_node();

        let mut imports = Vec::new();

        // Extract import statements (import os, sys)
        imports.extend(extract_import_statements(source, &root_node)?);

        // Extract from-import statements (from os import path)
        imports.extend(extract_from_imports(source, &root_node)?);

        Ok(imports)
    }
}

/// Extract regular import statements: import os, import sys
fn extract_import_statements(source: &str, root: &tree_sitter::Node) -> Result<Vec<ImportInfo>> {
    let language = tree_sitter_python::LANGUAGE;

    let query_str = r#"
        (import_statement
            name: (dotted_name) @import_path) @import
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create import statement query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut imports = Vec::new();

    while let Some(match_) = matches.next() {
        let mut import_path = None;
        let mut import_node = None;

        for capture in match_.captures {
            let capture_name: &str = query.capture_names()[capture.index as usize];
            match capture_name {
                "import_path" => {
                    import_path = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                }
                "import" => {
                    import_node = Some(capture.node);
                }
                _ => {}
            }
        }

        if let (Some(path), Some(node)) = (import_path, import_node) {
            let import_type = classify_python_import(&path);
            let line_number = node.start_position().row + 1;

            imports.push(ImportInfo {
                imported_path: path,
                import_type,
                line_number,
                imported_symbols: None,
            });
        }
    }

    Ok(imports)
}

/// Extract from-import statements: from os import path, from . import module
fn extract_from_imports(source: &str, root: &tree_sitter::Node) -> Result<Vec<ImportInfo>> {
    let language = tree_sitter_python::LANGUAGE;

    let query_str = r#"
        (import_from_statement
            module_name: (dotted_name) @module_path) @import

        (import_from_statement
            module_name: (relative_import) @module_path) @import
    "#;

    let query =
        Query::new(&language.into(), query_str).context("Failed to create from-import query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut imports = Vec::new();

    while let Some(match_) = matches.next() {
        let mut module_path = None;
        let mut import_node = None;

        for capture in match_.captures {
            let capture_name: &str = query.capture_names()[capture.index as usize];
            match capture_name {
                "module_path" => {
                    module_path = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                }
                "import" => {
                    import_node = Some(capture.node);
                }
                _ => {}
            }
        }

        if let (Some(path), Some(node)) = (module_path, import_node) {
            let import_type = classify_python_import(&path);
            let line_number = node.start_position().row + 1;

            // Extract imported symbols if present
            let imported_symbols = extract_imported_symbols(source, &node);

            imports.push(ImportInfo {
                imported_path: path,
                import_type,
                line_number,
                imported_symbols,
            });
        }
    }

    Ok(imports)
}

/// Extract the list of imported symbols from a from-import statement
fn extract_imported_symbols(source: &str, import_node: &tree_sitter::Node) -> Option<Vec<String>> {
    let mut symbols = Vec::new();

    // Walk children to find aliased_import or dotted_name nodes
    let mut cursor = import_node.walk();
    for child in import_node.children(&mut cursor) {
        match child.kind() {
            "aliased_import" | "dotted_name" => {
                // Get the first identifier
                let mut child_cursor = child.walk();
                for grandchild in child.children(&mut child_cursor) {
                    if (grandchild.kind() == "identifier" || grandchild.kind() == "dotted_name")
                        && let Ok(text) = grandchild.utf8_text(source.as_bytes())
                    {
                        symbols.push(text.to_string());
                        break; // Only get the first one for aliased imports
                    }
                }
            }
            _ => {}
        }
    }

    if symbols.is_empty() {
        None
    } else {
        Some(symbols)
    }
}

/// Find the Python package name from pyproject.toml, setup.py, or setup.cfg
/// This is used to determine which imports are internal vs external
pub fn find_python_package_name(root: &std::path::Path) -> Option<String> {
    // Try pyproject.toml first (modern standard)
    if let Some(name) = find_pyproject_package(root) {
        return Some(name);
    }

    // Try setup.py second
    if let Some(name) = find_setup_py_package(root) {
        return Some(name);
    }

    // Try setup.cfg third
    if let Some(name) = find_setup_cfg_package(root) {
        return Some(name);
    }

    None
}

/// Parse pyproject.toml to extract package name
fn find_pyproject_package(root: &std::path::Path) -> Option<String> {
    let pyproject_path = root.join("pyproject.toml");
    let content = std::fs::read_to_string(pyproject_path).ok()?;

    // Look for [project] section and name field
    // Example: name = "Django"
    let mut in_project_section = false;

    for line in content.lines() {
        let trimmed = line.trim();

        // Detect [project] section
        if trimmed == "[project]" {
            in_project_section = true;
            continue;
        }

        // Stop if we hit another section
        if trimmed.starts_with('[') && trimmed != "[project]" {
            in_project_section = false;
            continue;
        }

        // Parse name field if we're in [project] section
        if in_project_section
            && trimmed.starts_with("name")
            && trimmed.contains('=')
            && let Some(equals_pos) = trimmed.find('=')
        {
            let after_equals = trimmed[equals_pos + 1..].trim();

            // Handle both "name" and 'name'
            for quote in ['"', '\''] {
                if let Some(start) = after_equals.find(quote)
                    && let Some(end) = after_equals[start + 1..].find(quote)
                {
                    let name = &after_equals[start + 1..start + 1 + end];
                    // Convert to lowercase for matching (Django → django)
                    return Some(name.to_lowercase());
                }
            }
        }
    }

    None
}

/// Parse setup.py to extract package name
fn find_setup_py_package(root: &std::path::Path) -> Option<String> {
    let setup_path = root.join("setup.py");
    let content = std::fs::read_to_string(setup_path).ok()?;

    // Look for: setup(name="package_name", ...) or setup(name='package_name', ...)
    // Simple regex-like parsing
    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.contains("name") && trimmed.contains('=') {
            // Extract quoted value after name=
            if let Some(name_pos) = trimmed.find("name") {
                let after_name = &trimmed[name_pos + 4..]; // Skip "name"

                if let Some(equals_pos) = after_name.find('=') {
                    let after_equals = after_name[equals_pos + 1..].trim();

                    // Handle both "name" and 'name'
                    for quote in ['"', '\''] {
                        if let Some(start) = after_equals.find(quote)
                            && let Some(end) = after_equals[start + 1..].find(quote)
                        {
                            let name = &after_equals[start + 1..start + 1 + end];
                            return Some(name.to_lowercase());
                        }
                    }
                }
            }
        }
    }

    None
}

/// Parse setup.cfg to extract package name
fn find_setup_cfg_package(root: &std::path::Path) -> Option<String> {
    let setup_cfg_path = root.join("setup.cfg");
    let content = std::fs::read_to_string(setup_cfg_path).ok()?;

    // Look for [metadata] section and name field
    let mut in_metadata_section = false;

    for line in content.lines() {
        let trimmed = line.trim();

        // Detect [metadata] section
        if trimmed == "[metadata]" {
            in_metadata_section = true;
            continue;
        }

        // Stop if we hit another section
        if trimmed.starts_with('[') && trimmed != "[metadata]" {
            in_metadata_section = false;
            continue;
        }

        // Parse name field if we're in [metadata] section
        if in_metadata_section
            && trimmed.starts_with("name")
            && trimmed.contains('=')
            && let Some(equals_pos) = trimmed.find('=')
        {
            let name = trimmed[equals_pos + 1..].trim();
            return Some(name.to_lowercase());
        }
    }

    None
}

/// Reclassify a Python import using the project's package name
/// Similar to reclassify_go_import() and reclassify_java_import()
pub fn reclassify_python_import(import_path: &str, package_prefix: Option<&str>) -> ImportType {
    // First check if this is an internal import (matches project package)
    if let Some(prefix) = package_prefix {
        // Extract first component: "django.conf.settings" → "django"
        let first_component = import_path.split('.').next().unwrap_or(import_path);

        if first_component == prefix {
            return ImportType::Internal;
        }
    }

    // Then check if it's relative (always internal)
    if import_path.starts_with('.') {
        return ImportType::Internal;
    }

    // Check stdlib
    if is_python_stdlib(import_path) {
        return ImportType::Stdlib;
    }

    // Default to external
    ImportType::External
}

/// Check if a Python import path is from the standard library
fn is_python_stdlib(path: &str) -> bool {
    const STDLIB_MODULES: &[&str] = &[
        "os",
        "sys",
        "io",
        "re",
        "json",
        "csv",
        "xml",
        "html",
        "http",
        "urllib",
        "collections",
        "itertools",
        "functools",
        "operator",
        "pathlib",
        "glob",
        "tempfile",
        "shutil",
        "pickle",
        "shelve",
        "sqlite3",
        "zlib",
        "gzip",
        "time",
        "datetime",
        "calendar",
        "logging",
        "argparse",
        "configparser",
        "typing",
        "dataclasses",
        "enum",
        "abc",
        "contextlib",
        "weakref",
        "threading",
        "multiprocessing",
        "subprocess",
        "queue",
        "asyncio",
        "socket",
        "email",
        "base64",
        "hashlib",
        "hmac",
        "secrets",
        "uuid",
        "math",
        "random",
        "statistics",
        "decimal",
        "fractions",
        "unittest",
        "doctest",
        "pdb",
        "trace",
        "timeit",
    ];

    // Extract first component of the path
    let first_component = path.split('.').next().unwrap_or("");

    STDLIB_MODULES.contains(&first_component)
}

/// Classify a Python import as internal, external, or stdlib
fn classify_python_import(import_path: &str) -> ImportType {
    // Relative imports (. or ..)
    if import_path.starts_with('.') {
        return ImportType::Internal;
    }

    // Python standard library (common modules)
    const STDLIB_MODULES: &[&str] = &[
        "os",
        "sys",
        "io",
        "re",
        "json",
        "csv",
        "xml",
        "html",
        "http",
        "urllib",
        "collections",
        "itertools",
        "functools",
        "operator",
        "pathlib",
        "glob",
        "tempfile",
        "shutil",
        "pickle",
        "shelve",
        "sqlite3",
        "zlib",
        "gzip",
        "time",
        "datetime",
        "calendar",
        "logging",
        "argparse",
        "configparser",
        "typing",
        "dataclasses",
        "enum",
        "abc",
        "contextlib",
        "weakref",
        "threading",
        "multiprocessing",
        "subprocess",
        "queue",
        "asyncio",
        "socket",
        "email",
        "base64",
        "hashlib",
        "hmac",
        "secrets",
        "uuid",
        "math",
        "random",
        "statistics",
        "decimal",
        "fractions",
        "unittest",
        "doctest",
        "pdb",
        "trace",
        "timeit",
    ];

    // Extract first component of the path
    let first_component = import_path.split('.').next().unwrap_or("");

    if STDLIB_MODULES.contains(&first_component) {
        ImportType::Stdlib
    } else {
        // Everything else is external (third-party packages)
        ImportType::External
    }
}

// ============================================================================
// Monorepo Support & Path Resolution
// ============================================================================

/// Represents a Python package configuration with its location
#[derive(Debug, Clone)]
pub struct PythonPackage {
    /// Package name (e.g., "django", "myapp")
    pub name: String,
    /// Project root relative to index root (e.g., "packages/backend")
    pub project_root: String,
    /// Absolute path to project root
    pub abs_project_root: std::path::PathBuf,
}

/// Recursively find all Python configuration files (pyproject.toml, setup.py, setup.cfg)
/// in the repository, respecting .gitignore
pub fn find_all_python_configs(index_root: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
    use ignore::WalkBuilder;

    let mut config_files = Vec::new();

    let walker = WalkBuilder::new(index_root)
        .follow_links(false)
        .git_ignore(true)
        .build();

    for entry in walker {
        let entry = entry?;
        let path = entry.path();

        if !path.is_file() {
            continue;
        }

        let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // Look for Python config files
        if filename == "pyproject.toml" || filename == "setup.py" || filename == "setup.cfg" {
            // Skip virtual environments and build directories. Normalize
            // separators so these filters fire on Windows (`\venv\`) too.
            let path_str = path.to_string_lossy().replace('\\', "/");
            if path_str.contains("/venv/")
                || path_str.contains("/.venv/")
                || path_str.contains("/site-packages/")
                || path_str.contains("/dist/")
                || path_str.contains("/build/")
                || path_str.contains("/__pycache__/")
            {
                log::trace!(
                    "Skipping Python config in vendor/build directory: {:?}",
                    path
                );
                continue;
            }

            config_files.push(path.to_path_buf());
        }
    }

    log::debug!("Found {} Python config files", config_files.len());
    Ok(config_files)
}

/// Parse all Python packages in a monorepo and track their project roots
pub fn parse_all_python_packages(index_root: &std::path::Path) -> Result<Vec<PythonPackage>> {
    let config_files = find_all_python_configs(index_root)?;

    if config_files.is_empty() {
        log::debug!("No Python config files found in {:?}", index_root);
        return Ok(Vec::new());
    }

    let mut packages = Vec::new();
    let config_count = config_files.len();

    for config_path in &config_files {
        let project_root = config_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Config file has no parent directory"))?;

        // Try to extract package name from this config
        if let Some(package_name) = find_python_package_name(project_root) {
            // Normalize to forward slashes so `starts_with("services/")`
            // style assertions and downstream import resolution behave the
            // same on every OS.
            let relative_project_root = project_root
                .strip_prefix(index_root)
                .unwrap_or(project_root)
                .to_string_lossy()
                .replace('\\', "/");

            log::debug!(
                "Found Python package '{}' at {:?}",
                package_name,
                relative_project_root
            );

            packages.push(PythonPackage {
                name: package_name,
                project_root: relative_project_root,
                abs_project_root: project_root.to_path_buf(),
            });
        }
    }

    log::info!(
        "Loaded {} Python packages from {} config files",
        packages.len(),
        config_count
    );

    Ok(packages)
}

/// Resolve a Python import to a file path
///
/// Handles:
/// - Absolute imports: `from myapp.models import User` → `myapp/models.py` or `myapp/models/__init__.py`
/// - Relative imports: `from .models import User` (requires current_file_path)
/// - Package imports: `import myapp.utils` → `myapp/utils.py` or `myapp/utils/__init__.py`
pub fn resolve_python_import_to_path(
    import_path: &str,
    packages: &[PythonPackage],
    current_file_path: Option<&str>,
) -> Option<String> {
    // Handle relative imports (. or ..)
    if import_path.starts_with('.') {
        return resolve_relative_python_import(import_path, current_file_path);
    }

    // Handle absolute imports using package mappings
    // Extract first component: "django.conf.settings" → "django"
    let first_component = import_path.split('.').next()?;

    // Find matching package
    for package in packages {
        if package.name == first_component {
            // Convert import path to file path
            // "django.conf.settings" → "django/conf/settings.py"
            let module_path = import_path.replace('.', "/");

            // Try both .py file and __init__.py in package
            let candidates = vec![
                format!("{}/{}.py", package.project_root, module_path),
                format!("{}/{}/__init__.py", package.project_root, module_path),
            ];

            if let Some(candidate) = candidates.into_iter().next() {
                log::trace!("Checking Python module path: {}", candidate);
                return Some(candidate);
            }
        }
    }

    None
}

/// Resolve relative Python imports (. or ..)
/// Requires the current file path to determine the relative location
fn resolve_relative_python_import(
    import_path: &str,
    current_file_path: Option<&str>,
) -> Option<String> {
    let current_file = current_file_path?;

    // Count leading dots to determine how many levels to go up
    let dots = import_path.chars().take_while(|&c| c == '.').count();
    if dots == 0 {
        return None;
    }

    // Get the directory of the current file
    let current_dir = std::path::Path::new(current_file).parent()?;

    // Go up (dots - 1) levels (one dot means current directory)
    let mut target_dir = current_dir.to_path_buf();
    for _ in 1..dots {
        target_dir = target_dir.parent()?.to_path_buf();
    }

    // Get the module path after the dots
    let module_path = import_path.trim_start_matches('.');

    if module_path.is_empty() {
        // Just "from ." means import from current package's __init__.py
        return Some(format!("{}/__init__.py", target_dir.to_string_lossy()));
    }

    // Convert dots to slashes: "models.user" → "models/user"
    let file_path = module_path.replace('.', "/");

    // Try both .py file and __init__.py in package
    let candidates = vec![
        format!("{}/{}.py", target_dir.to_string_lossy(), file_path),
        format!("{}/{}/__init__.py", target_dir.to_string_lossy(), file_path),
    ];

    if let Some(candidate) = candidates.into_iter().next() {
        log::trace!("Checking relative Python import: {}", candidate);
        return Some(candidate);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_function() {
        let source = r#"
def hello_world():
    print("Hello, world!")
    return True
        "#;

        let symbols = parse("test.py", source).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("hello_world"));
        assert!(matches!(symbols[0].kind, SymbolKind::Function));
    }

    #[test]
    fn test_parse_async_function() {
        let source = r#"
async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
        "#;

        let symbols = parse("test.py", source).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("fetch_data"));
        assert!(matches!(symbols[0].kind, SymbolKind::Function));
    }

    #[test]
    fn test_parse_class() {
        let source = r#"
class User:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        "#;

        let symbols = parse("test.py", source).unwrap();

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
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    @staticmethod
    def multiply(a, b):
        return a * b
        "#;

        let symbols = parse("test.py", source).unwrap();

        let method_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method))
            .collect();

        assert_eq!(method_symbols.len(), 3);
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
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("multiply"))
        );

        // Check scope
        for _method in method_symbols {
            // Removed: scope field no longer exists: assert_eq!(method.scope.as_ref().unwrap(), "class Calculator");
        }
    }

    #[test]
    fn test_parse_async_method() {
        let source = r#"
class DataFetcher:
    async def get_user(self, user_id):
        return await fetch(f"/users/{user_id}")

    async def get_all_users(self):
        return await fetch("/users")
        "#;

        let symbols = parse("test.py", source).unwrap();

        let method_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method))
            .collect();

        assert_eq!(method_symbols.len(), 2);
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("get_user"))
        );
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("get_all_users"))
        );
    }

    #[test]
    fn test_parse_constants() {
        let source = r#"
MAX_SIZE = 100
DEFAULT_TIMEOUT = 30
API_URL = "https://api.example.com"
        "#;

        let symbols = parse("test.py", source).unwrap();

        let const_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Constant))
            .collect();

        assert_eq!(const_symbols.len(), 3);
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
        assert!(
            const_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("API_URL"))
        );
    }

    #[test]
    fn test_parse_lambda() {
        let source = r#"
square = lambda x: x * x
add = lambda a, b: a + b
        "#;

        let symbols = parse("test.py", source).unwrap();

        let lambda_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Function))
            .collect();

        assert!(lambda_symbols.len() >= 2);
        assert!(
            lambda_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("square"))
        );
        assert!(
            lambda_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("add"))
        );
    }

    #[test]
    fn test_parse_decorated_method() {
        let source = r#"
class WebService:
    @property
    def url(self):
        return self._url

    @classmethod
    def from_config(cls, config):
        return cls(config['url'])

    @staticmethod
    def validate_url(url):
        return url.startswith('http')
        "#;

        let symbols = parse("test.py", source).unwrap();

        let method_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method))
            .collect();

        assert_eq!(method_symbols.len(), 3);
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("url"))
        );
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("from_config"))
        );
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("validate_url"))
        );
    }

    #[test]
    fn test_parse_mixed_symbols() {
        let source = r#"
API_KEY = "secret123"
MAX_RETRIES = 3

class APIClient:
    def __init__(self, api_key):
        self.api_key = api_key

    async def request(self, endpoint):
        return await self._fetch(endpoint)

    @staticmethod
    def build_url(endpoint):
        return f"https://api.example.com/{endpoint}"

def create_client():
    return APIClient(API_KEY)

process = lambda data: data.strip().lower()
        "#;

        let symbols = parse("test.py", source).unwrap();

        // Should find: 2 constants, 1 class, 3 methods, 1 function, 1 lambda
        assert!(symbols.len() >= 8);

        let kinds: Vec<&SymbolKind> = symbols.iter().map(|s| &s.kind).collect();
        assert!(kinds.contains(&&SymbolKind::Constant));
        assert!(kinds.contains(&&SymbolKind::Class));
        assert!(kinds.contains(&&SymbolKind::Method));
        assert!(kinds.contains(&&SymbolKind::Function));
    }

    #[test]
    fn test_parse_nested_class() {
        let source = r#"
class Outer:
    class Inner:
        def inner_method(self):
            pass

    def outer_method(self):
        pass
        "#;

        let symbols = parse("test.py", source).unwrap();

        let class_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        // Should find both Outer and Inner classes
        assert_eq!(class_symbols.len(), 2);
        assert!(
            class_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Outer"))
        );
        assert!(
            class_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Inner"))
        );
    }

    #[test]
    fn test_local_variables_included() {
        let source = r#"
def calculate(input):
    local_var = input * 2
    result = local_var + 10
    return result

class Calculator:
    def compute(self, value):
        temp = value * 3
        final = temp + 5
        return final
        "#;

        let symbols = parse("test.py", source).unwrap();

        // Filter to just variables
        let variables: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        // Check that local variables are captured
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("local_var"))
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
        for _var in variables {
            // Removed: scope field no longer exists: assert_eq!(var.scope, None);
        }
    }

    #[test]
    fn test_global_variables() {
        let source = r#"
# Global constants (uppercase)
MAX_SIZE = 100
DEFAULT_TIMEOUT = 30

# Global variables (non-uppercase)
database_url = "postgresql://localhost/mydb"
config = {"debug": True}
current_user = None

def get_config():
    return config
        "#;

        let symbols = parse("test.py", source).unwrap();

        // Filter to constants and variables
        let constants: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Constant))
            .collect();

        let variables: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        // Check that constants are captured (uppercase)
        assert!(
            constants
                .iter()
                .any(|c| c.symbol.as_deref() == Some("MAX_SIZE"))
        );
        assert!(
            constants
                .iter()
                .any(|c| c.symbol.as_deref() == Some("DEFAULT_TIMEOUT"))
        );

        // Check that global variables are captured (non-uppercase)
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("database_url"))
        );
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("config"))
        );
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("current_user"))
        );

        // Verify no scope for both
        for _constant in constants {
            // Removed: scope field no longer exists: assert_eq!(constant.scope, None);
        }
        for _var in variables {
            // Removed: scope field no longer exists: assert_eq!(var.scope, None);
        }
    }

    #[test]
    fn test_find_all_python_configs() {
        use std::fs;
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create multiple Python projects
        let project1 = root.join("backend");
        fs::create_dir_all(&project1).unwrap();
        fs::write(
            project1.join("pyproject.toml"),
            "[project]\nname = \"backend\"",
        )
        .unwrap();

        let project2 = root.join("frontend/api");
        fs::create_dir_all(&project2).unwrap();
        fs::write(project2.join("setup.py"), "setup(name='api')").unwrap();

        // Create venv directory that should be skipped
        let venv = root.join("venv");
        fs::create_dir_all(&venv).unwrap();
        fs::write(venv.join("setup.py"), "setup(name='should_skip')").unwrap();

        let configs = find_all_python_configs(root).unwrap();

        // Should find 2 configs (skipping venv)
        assert_eq!(configs.len(), 2);
        assert!(
            configs
                .iter()
                .any(|p| p.ends_with("backend/pyproject.toml"))
        );
        assert!(configs.iter().any(|p| p.ends_with("frontend/api/setup.py")));
    }

    #[test]
    fn test_parse_all_python_packages() {
        use std::fs;
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create multiple Python projects with different config types
        let project1 = root.join("services/auth");
        fs::create_dir_all(&project1).unwrap();
        fs::write(
            project1.join("pyproject.toml"),
            "[project]\nname = \"auth-service\"\n",
        )
        .unwrap();

        let project2 = root.join("services/api");
        fs::create_dir_all(&project2).unwrap();
        fs::write(project2.join("setup.py"), "setup(name=\"api-service\")").unwrap();

        let packages = parse_all_python_packages(root).unwrap();

        // Should find 2 packages
        assert_eq!(packages.len(), 2);

        // Check package names (normalized to lowercase)
        let names: Vec<_> = packages.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"auth-service"));
        assert!(names.contains(&"api-service"));

        // Check project roots
        for package in &packages {
            assert!(package.project_root.starts_with("services/"));
            assert!(package.abs_project_root.ends_with(&package.project_root));
        }
    }

    #[test]
    fn test_resolve_python_import_absolute() {
        use std::fs;
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create a Python package structure
        let myapp = root.join("myapp");
        fs::create_dir_all(myapp.join("models")).unwrap();
        fs::write(
            myapp.join("pyproject.toml"),
            "[project]\nname = \"myapp\"\n",
        )
        .unwrap();

        let packages = parse_all_python_packages(root).unwrap();
        assert_eq!(packages.len(), 1);

        // Test absolute import resolution
        // "myapp.models.user" → "myapp/models/user.py"
        let resolved = resolve_python_import_to_path("myapp.models.user", &packages, None);

        assert!(resolved.is_some());
        let path = resolved.unwrap();
        assert!(
            path.contains("myapp/models/user.py") || path.contains("myapp/models/user/__init__.py")
        );
    }

    #[test]
    fn test_resolve_python_import_relative() {
        // Test relative imports: from .models import User
        let current_file = "myapp/views/admin.py";

        // Test single dot (current package)
        let resolved = resolve_python_import_to_path(
            ".models",
            &[], // Empty packages array - relative imports don't need it
            Some(current_file),
        );

        assert!(resolved.is_some());
        let path = resolved.unwrap();
        // from .models → myapp/views/models.py or myapp/views/models/__init__.py
        assert!(path.contains("myapp/views/models"));

        // Test double dot (parent package)
        let resolved = resolve_python_import_to_path("..utils", &[], Some(current_file));

        assert!(resolved.is_some());
        let path = resolved.unwrap();
        // from ..utils → myapp/utils.py or myapp/utils/__init__.py
        assert!(path.contains("myapp/utils"));
    }

    #[test]
    fn test_resolve_python_import_relative_with_module() {
        // Test relative imports with module path: from ..models.user import User
        let current_file = "myapp/views/dashboard/index.py";

        let resolved = resolve_python_import_to_path("..models.user", &[], Some(current_file));

        assert!(resolved.is_some());
        let path = resolved.unwrap();
        // from ..models.user → myapp/views/models/user.py
        assert!(path.contains("models/user"));
    }

    #[test]
    fn test_resolve_python_import_not_found() {
        use std::fs;
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        let root = temp.path();

        let myapp = root.join("myapp");
        fs::create_dir_all(&myapp).unwrap();
        fs::write(
            myapp.join("pyproject.toml"),
            "[project]\nname = \"myapp\"\n",
        )
        .unwrap();

        let packages = parse_all_python_packages(root).unwrap();

        // Try to resolve an import for a different package
        let resolved = resolve_python_import_to_path("other_package.module", &packages, None);

        // Should return None for packages not in the monorepo
        assert!(resolved.is_none());
    }

    #[test]
    fn test_dynamic_imports_filtered() {
        let source = r#"
import os
import sys
from json import loads
from .models import User

# Dynamic imports - should be filtered out
import importlib
mod = importlib.import_module("some_module")
pkg = __import__("package")
exec("import dynamic")
        "#;

        let deps = PythonDependencyExtractor::extract_dependencies(source).unwrap();

        // Should only find static imports (os, sys, json, .models, importlib)
        // importlib.import_module(), __import__(), and exec() are NOT import statements
        assert_eq!(deps.len(), 5, "Should extract 5 static imports only");

        assert!(deps.iter().any(|d| d.imported_path == "os"));
        assert!(deps.iter().any(|d| d.imported_path == "sys"));
        assert!(deps.iter().any(|d| d.imported_path == "json"));
        assert!(deps.iter().any(|d| d.imported_path == ".models"));
        assert!(deps.iter().any(|d| d.imported_path == "importlib"));

        // Verify dynamic imports are NOT captured
        assert!(!deps.iter().any(|d| d.imported_path.contains("some_module")));
        assert!(
            !deps
                .iter()
                .any(|d| d.imported_path.contains("package") && d.imported_path != "json")
        );
        assert!(!deps.iter().any(|d| d.imported_path.contains("dynamic")));
    }
}
