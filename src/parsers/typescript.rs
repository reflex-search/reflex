//! TypeScript/JavaScript language parser using Tree-sitter
//!
//! Extracts symbols from TypeScript and JavaScript source code:
//! - Functions (regular, arrow, async, generator)
//! - Classes (regular, abstract)
//! - Interfaces
//! - Type aliases
//! - Enums
//! - Variables and constants (const, let, var - all scopes)
//! - Methods (with class scope)
//! - Modules/Namespaces
//!
//! This parser handles both TypeScript (.ts, .tsx) and JavaScript (.js, .jsx)
//! files using the tree-sitter-typescript grammar.

use anyhow::{Context, Result};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};
use crate::models::{Language, SearchResult, Span, SymbolKind};

/// Parse TypeScript/JavaScript source code and extract symbols
pub fn parse(path: &str, source: &str, language: Language) -> Result<Vec<SearchResult>> {
    let mut parser = Parser::new();

    // tree-sitter-typescript provides both TypeScript and TSX grammars
    // For JavaScript, we use the TypeScript grammar (it's a superset)
    let ts_language_fn = match language {
        Language::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT,
        Language::JavaScript => tree_sitter_typescript::LANGUAGE_TSX, // TSX handles both JS and JSX
        _ => return Err(anyhow::anyhow!("Unsupported language: {:?}", language)),
    };

    // Convert LanguageFn to Language
    let ts_language: tree_sitter::Language = ts_language_fn.into();

    parser
        .set_language(&ts_language)
        .context("Failed to set TypeScript/JavaScript language")?;

    let tree = parser
        .parse(source, None)
        .context("Failed to parse TypeScript/JavaScript source")?;

    let root_node = tree.root_node();

    let mut symbols = Vec::new();

    // Extract different types of symbols using Tree-sitter queries
    symbols.extend(extract_functions(source, &root_node, &ts_language)?);
    symbols.extend(extract_arrow_functions(source, &root_node, &ts_language)?);
    symbols.extend(extract_classes(source, &root_node, &ts_language)?);
    symbols.extend(extract_interfaces(source, &root_node, &ts_language)?);
    symbols.extend(extract_type_aliases(source, &root_node, &ts_language)?);
    symbols.extend(extract_enums(source, &root_node, &ts_language)?);
    symbols.extend(extract_variables(source, &root_node, &ts_language)?);
    symbols.extend(extract_methods(source, &root_node, &ts_language)?);

    // Add file path and language to all symbols
    for symbol in &mut symbols {
        symbol.path = path.to_string();
        symbol.lang = language.clone();
    }

    Ok(symbols)
}

/// Extract regular function declarations (including async and generator)
fn extract_functions(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (function_declaration
            name: (identifier) @name) @function

        (generator_function_declaration
            name: (identifier) @name) @function
    "#;

    let query = Query::new(language, query_str)
        .context("Failed to create function query")?;

    extract_symbols(source, root, &query, SymbolKind::Function, None)
}

/// Extract arrow functions assigned to variables/constants
fn extract_arrow_functions(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (lexical_declaration
            (variable_declarator
                name: (identifier) @name
                value: (arrow_function))) @arrow_fn

        (variable_declaration
            (variable_declarator
                name: (identifier) @name
                value: (arrow_function))) @arrow_fn
    "#;

    let query = Query::new(language, query_str)
        .context("Failed to create arrow function query")?;

    extract_symbols(source, root, &query, SymbolKind::Function, None)
}

/// Extract class declarations (including abstract classes)
fn extract_classes(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_declaration
            name: (type_identifier) @name) @class

        (abstract_class_declaration
            name: (type_identifier) @name) @class
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
            name: (type_identifier) @name) @interface
    "#;

    let query = Query::new(language, query_str)
        .context("Failed to create interface query")?;

    extract_symbols(source, root, &query, SymbolKind::Interface, None)
}

/// Extract type alias declarations
fn extract_type_aliases(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (type_alias_declaration
            name: (type_identifier) @name) @type
    "#;

    let query = Query::new(language, query_str)
        .context("Failed to create type alias query")?;

    extract_symbols(source, root, &query, SymbolKind::Type, None)
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

/// Extract variable and constant declarations (const, let, var - all scopes)
fn extract_variables(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    // Extract const/let (lexical_declaration) and var (variable_declaration)
    // Skip arrow functions as they're already handled by extract_arrow_functions
    let query_str = r#"
        (lexical_declaration
            (variable_declarator
                name: (identifier) @name)) @decl

        (variable_declaration
            (variable_declarator
                name: (identifier) @name)) @decl
    "#;

    let query = Query::new(language, query_str)
        .context("Failed to create variable query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut name = None;
        let mut declarator_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            if capture_name == "name" {
                name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                // Get the variable_declarator node
                if let Some(parent) = capture.node.parent() {
                    if parent.kind() == "variable_declarator" {
                        declarator_node = Some(parent);
                    }
                }
            }
        }

        if let (Some(name), Some(declarator)) = (name, declarator_node) {
            // Check if the value is an arrow function (skip those)
            let mut is_arrow_function = false;
            for i in 0..declarator.child_count() {
                if let Some(child) = declarator.child(i as u32) {
                    if child.kind() == "arrow_function" {
                        is_arrow_function = true;
                        break;
                    }
                }
            }

            // Only add if it's NOT an arrow function
            if !is_arrow_function {
                if let Some(decl_node) = declarator.parent() {
                    let span = node_to_span(&decl_node);
                    let preview = extract_preview(source, &span);

                    // Determine if it's a constant (const) or variable (let/var)
                    let decl_text = decl_node.utf8_text(source.as_bytes()).unwrap_or("");
                    let kind = if decl_text.trim_start().starts_with("const") {
                        SymbolKind::Constant
                    } else {
                        SymbolKind::Variable
                    };

                    symbols.push(SearchResult::new(
                        String::new(),
                        Language::TypeScript,
                        kind,
                        Some(name),
                        span,
                        None,
                        preview,
                    ));
                }
            }
        }
    }

    Ok(symbols)
}

/// Extract method definitions from classes
fn extract_methods(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_declaration
            name: (type_identifier) @class_name
            body: (class_body
                (method_definition
                    name: (_) @method_name))) @class

        (abstract_class_declaration
            name: (type_identifier) @class_name
            body: (class_body
                (method_definition
                    name: (_) @method_name))) @class
    "#;

    let query = Query::new(language, query_str)
        .context("Failed to create method query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut class_name = None;
        let mut method_name = None;
        let mut method_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            match capture_name {
                "class_name" => {
                    class_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                }
                "method_name" => {
                    method_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or("").to_string());
                    // Find the parent method_definition node
                    let mut current = capture.node;
                    while let Some(parent) = current.parent() {
                        if parent.kind() == "method_definition" {
                            method_node = Some(parent);
                            break;
                        }
                        current = parent;
                    }
                }
                _ => {}
            }
        }

        if let (Some(class_name), Some(method_name), Some(node)) = (class_name, method_name, method_node) {
            let scope = format!("class {}", class_name);
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::TypeScript,
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
                Language::TypeScript,
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
    fn test_parse_function() {
        let source = r#"
            function greet(name: string): string {
                return `Hello, ${name}!`;
            }
        "#;

        let symbols = parse("test.ts", source, Language::TypeScript).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("greet"));
        assert!(matches!(symbols[0].kind, SymbolKind::Function));
    }

    #[test]
    fn test_parse_arrow_function() {
        let source = r#"
            const add = (a: number, b: number): number => {
                return a + b;
            };
        "#;

        let symbols = parse("test.ts", source, Language::TypeScript).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("add"));
        assert!(matches!(symbols[0].kind, SymbolKind::Function));
    }

    #[test]
    fn test_parse_async_function() {
        let source = r#"
            async function fetchData(url: string): Promise<Response> {
                return await fetch(url);
            }
        "#;

        let symbols = parse("test.ts", source, Language::TypeScript).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("fetchData"));
        assert!(matches!(symbols[0].kind, SymbolKind::Function));
    }

    #[test]
    fn test_parse_class() {
        let source = r#"
            class User {
                name: string;
                age: number;

                constructor(name: string, age: number) {
                    this.name = name;
                    this.age = age;
                }
            }
        "#;

        let symbols = parse("test.ts", source, Language::TypeScript).unwrap();

        // Should find class
        let class_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        assert_eq!(class_symbols.len(), 1);
        assert_eq!(class_symbols[0].symbol.as_deref(), Some("User"));
    }

    #[test]
    fn test_parse_class_with_methods() {
        let source = r#"
            class Calculator {
                add(a: number, b: number): number {
                    return a + b;
                }

                subtract(a: number, b: number): number {
                    return a - b;
                }
            }
        "#;

        let symbols = parse("test.ts", source, Language::TypeScript).unwrap();

        // Should find class + 2 methods
        assert!(symbols.len() >= 3);

        let method_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method))
            .collect();

        assert_eq!(method_symbols.len(), 2);
        assert!(method_symbols.iter().any(|s| s.symbol.as_deref() == Some("add")));
        assert!(method_symbols.iter().any(|s| s.symbol.as_deref() == Some("subtract")));

        // Check scope
        for method in method_symbols {
            // Removed: scope field no longer exists: assert_eq!(method.scope.as_ref().unwrap(), "class Calculator");
        }
    }

    #[test]
    fn test_parse_interface() {
        let source = r#"
            interface User {
                name: string;
                age: number;
                email?: string;
            }
        "#;

        let symbols = parse("test.ts", source, Language::TypeScript).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("User"));
        assert!(matches!(symbols[0].kind, SymbolKind::Interface));
    }

    #[test]
    fn test_parse_type_alias() {
        let source = r#"
            type UserId = string | number;
            type UserRole = 'admin' | 'user' | 'guest';
        "#;

        let symbols = parse("test.ts", source, Language::TypeScript).unwrap();
        assert_eq!(symbols.len(), 2);

        let type_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Type))
            .collect();

        assert_eq!(type_symbols.len(), 2);
        assert!(type_symbols.iter().any(|s| s.symbol.as_deref() == Some("UserId")));
        assert!(type_symbols.iter().any(|s| s.symbol.as_deref() == Some("UserRole")));
    }

    #[test]
    fn test_parse_enum() {
        let source = r#"
            enum Status {
                Active,
                Inactive,
                Pending
            }
        "#;

        let symbols = parse("test.ts", source, Language::TypeScript).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("Status"));
        assert!(matches!(symbols[0].kind, SymbolKind::Enum));
    }

    #[test]
    fn test_parse_const() {
        let source = r#"
            const MAX_SIZE = 100;
            const DEFAULT_USER = {
                name: "Anonymous",
                age: 0
            };
        "#;

        let symbols = parse("test.ts", source, Language::TypeScript).unwrap();
        assert_eq!(symbols.len(), 2);

        let const_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Constant))
            .collect();

        assert_eq!(const_symbols.len(), 2);
        assert!(const_symbols.iter().any(|s| s.symbol.as_deref() == Some("MAX_SIZE")));
        assert!(const_symbols.iter().any(|s| s.symbol.as_deref() == Some("DEFAULT_USER")));
    }

    #[test]
    fn test_parse_react_component() {
        let source = r#"
            import React, { useState } from 'react';

            interface ButtonProps {
                label: string;
                onClick: () => void;
            }

            const Button: React.FC<ButtonProps> = ({ label, onClick }) => {
                return (
                    <button onClick={onClick}>
                        {label}
                    </button>
                );
            };

            function useCounter(initial: number) {
                const [count, setCount] = React.useState(initial);
                return { count, setCount };
            }

            export default Button;
        "#;

        let symbols = parse("Button.tsx", source, Language::TypeScript).unwrap();

        // Should find interface, Button component (arrow fn), useCounter hook (function)
        assert!(symbols.iter().any(|s| s.symbol.as_deref() == Some("ButtonProps") && matches!(s.kind, SymbolKind::Interface)));
        assert!(symbols.iter().any(|s| s.symbol.as_deref() == Some("Button") && matches!(s.kind, SymbolKind::Function)));
        assert!(symbols.iter().any(|s| s.symbol.as_deref() == Some("useCounter") && matches!(s.kind, SymbolKind::Function)));
    }

    #[test]
    fn test_parse_mixed_symbols() {
        let source = r#"
            interface Config {
                debug: boolean;
            }

            type ConfigKey = keyof Config;

            const DEFAULT_CONFIG: Config = {
                debug: false
            };

            class ConfigManager {
                private config: Config;

                constructor(config: Config) {
                    this.config = config;
                }

                getConfig(): Config {
                    return this.config;
                }
            }

            function loadConfig(): Config {
                return DEFAULT_CONFIG;
            }
        "#;

        let symbols = parse("test.ts", source, Language::TypeScript).unwrap();

        // Should find: interface, type, const, class, method, function
        assert!(symbols.len() >= 6);

        let kinds: Vec<&SymbolKind> = symbols.iter().map(|s| &s.kind).collect();
        assert!(kinds.contains(&&SymbolKind::Interface));
        assert!(kinds.contains(&&SymbolKind::Type));
        assert!(kinds.contains(&&SymbolKind::Constant));
        assert!(kinds.contains(&&SymbolKind::Class));
        assert!(kinds.contains(&&SymbolKind::Method));
        assert!(kinds.contains(&&SymbolKind::Function));
    }

    #[test]
    fn test_parse_async_class_methods() {
        let source = r#"
            export class CentralUsersModule {
                async getAllUsers(params) {
                    return await this.call('get', `/users`, params)
                }

                async getUser(userId) {
                    return await this.call('get', `/users/${userId}`)
                }

                deleteUser(userId) {
                    return this.call('delete', `/user/${userId}`)
                }
            }
        "#;

        let symbols = parse("test.ts", source, Language::TypeScript).unwrap();

        // Debug: Print all symbols
        println!("\nAll symbols found:");
        for symbol in &symbols {
            println!("  {:?} - {}", symbol.kind, symbol.symbol.as_deref().unwrap_or(""));
        }

        // Should find: class + 3 methods (2 async, 1 regular)
        let class_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();
        assert_eq!(class_symbols.len(), 1);
        assert_eq!(class_symbols[0].symbol.as_deref(), Some("CentralUsersModule"));

        let method_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method))
            .collect();

        // All three should be detected as methods, not variables
        assert_eq!(method_symbols.len(), 3, "Expected 3 methods, found {}", method_symbols.len());
        assert!(method_symbols.iter().any(|s| s.symbol.as_deref() == Some("getAllUsers")));
        assert!(method_symbols.iter().any(|s| s.symbol.as_deref() == Some("getUser")));
        assert!(method_symbols.iter().any(|s| s.symbol.as_deref() == Some("deleteUser")));

        // Verify no async methods are misclassified as variables
        let variable_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Constant) || matches!(s.kind, SymbolKind::Variable))
            .collect();
        assert_eq!(variable_symbols.len(), 0, "Async methods should not be classified as variables");

        // Check scope
        for method in method_symbols {
            // Removed: scope field no longer exists: assert_eq!(method.scope.as_ref().unwrap(), "class CentralUsersModule");
        }
    }

    #[test]
    fn test_parse_user_exact_code() {
        // User's exact code with TypeScript types
        let source = r#"
export class CentralUsersModule extends HttpFactory<WatchHookMap, WatchEvents> {
  protected $events = {
    //
  }

  async checkAuthenticated() {
    return await this.call('get', '/check')
  }

  async getUser(userId: CentralUser['id']) {
    return await this.call<CentralUser>('get', `/users/${userId}`)
  }

  async getAllUsers(params?: PaginatedParams & SortableParams & SearchableParams) {
    return await this.call<CentralUser[]>('get', `/users`, params)
  }

  async deleteUser(userId: CentralUser['id']) {
    return await this.call<void>('delete', `/user/${userId}`)
  }
}
        "#;

        let symbols = parse("test.ts", source, Language::TypeScript).unwrap();

        // Debug: Print all symbols
        println!("\nAll symbols found in user code:");
        for symbol in &symbols {
            println!("  {:?} - {}", symbol.kind, symbol.symbol.as_deref().unwrap_or(""));
        }

        // Verify getAllUsers is a Method, not a Variable
        let get_all_users_symbols: Vec<_> = symbols.iter()
            .filter(|s| s.symbol.as_deref() == Some("getAllUsers"))
            .collect();

        assert_eq!(get_all_users_symbols.len(), 1, "Should find exactly one getAllUsers");
        assert!(
            matches!(get_all_users_symbols[0].kind, SymbolKind::Method),
            "getAllUsers should be a Method, not {:?}",
            get_all_users_symbols[0].kind
        );
    }

    #[test]
    fn test_local_variables_included() {
        let source = r#"
            const GLOBAL_CONSTANT = 100;
            let globalLet = 50;
            var globalVar = 25;

            function calculate(x: number): number {
                const localConst = x * 2;
                let localLet = 5;
                var localVar = 10;
                return localConst + localLet + localVar;
            }
        "#;

        let symbols = parse("test.ts", source, Language::TypeScript).unwrap();

        let var_symbols: Vec<_> = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable) || matches!(s.kind, SymbolKind::Constant))
            .collect();

        // Should find all: 3 global + 3 local = 6 variables
        assert_eq!(var_symbols.len(), 6);

        // Check globals
        assert!(var_symbols.iter().any(|s| s.symbol.as_deref() == Some("GLOBAL_CONSTANT")));
        assert!(var_symbols.iter().any(|s| s.symbol.as_deref() == Some("globalLet")));
        assert!(var_symbols.iter().any(|s| s.symbol.as_deref() == Some("globalVar")));

        // Check locals
        assert!(var_symbols.iter().any(|s| s.symbol.as_deref() == Some("localConst")));
        assert!(var_symbols.iter().any(|s| s.symbol.as_deref() == Some("localLet")));
        assert!(var_symbols.iter().any(|s| s.symbol.as_deref() == Some("localVar")));

        // Verify const vs variable classification
        let global_const = var_symbols.iter().find(|s| s.symbol.as_deref() == Some("GLOBAL_CONSTANT")).unwrap();
        assert!(matches!(global_const.kind, SymbolKind::Constant));

        let global_let = var_symbols.iter().find(|s| s.symbol.as_deref() == Some("globalLet")).unwrap();
        assert!(matches!(global_let.kind, SymbolKind::Variable));
    }
}

// ============================================================================
// Dependency Extraction
// ============================================================================

use crate::models::ImportType;
use crate::parsers::{DependencyExtractor, ImportInfo};

/// TypeScript/JavaScript dependency extractor
pub struct TypeScriptDependencyExtractor;

impl DependencyExtractor for TypeScriptDependencyExtractor {
    fn extract_dependencies(source: &str) -> Result<Vec<ImportInfo>> {
        // Delegate to the version without alias map for compatibility
        Self::extract_dependencies_with_alias_map(source, None)
    }
}

impl TypeScriptDependencyExtractor {
    /// Extract dependencies with optional tsconfig alias map support
    ///
    /// This version properly classifies path alias imports (like @packages/*, ~/*) as Internal
    /// when they match configured aliases from tsconfig.json.
    pub fn extract_dependencies_with_alias_map(
        source: &str,
        alias_map: Option<&crate::parsers::tsconfig::PathAliasMap>,
    ) -> Result<Vec<ImportInfo>> {
        let mut parser = Parser::new();
        let language = tree_sitter_typescript::LANGUAGE_TSX; // Use TSX for JS/TS compatibility

        parser
            .set_language(&language.into())
            .context("Failed to set TypeScript/JavaScript language")?;

        let tree = parser
            .parse(source, None)
            .context("Failed to parse TypeScript/JavaScript source")?;

        let root_node = tree.root_node();

        let mut imports = Vec::new();

        // Extract ES6 import statements
        imports.extend(extract_import_declarations(source, &root_node, alias_map)?);

        // Extract require() statements
        imports.extend(extract_require_statements(source, &root_node, alias_map)?);

        Ok(imports)
    }
}

/// Extract ES6 import declarations: import { foo } from 'module'
fn extract_import_declarations(
    source: &str,
    root: &tree_sitter::Node,
    alias_map: Option<&crate::parsers::tsconfig::PathAliasMap>,
) -> Result<Vec<ImportInfo>> {
    let language = tree_sitter_typescript::LANGUAGE_TSX;

    let query_str = r#"
        (import_statement
            source: (string) @import_path) @import
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create import declaration query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut imports = Vec::new();

    while let Some(match_) = matches.next() {
        let mut import_path = None;
        let mut import_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            match capture_name {
                "import_path" => {
                    // Remove quotes from string literal
                    let raw_path = capture.node.utf8_text(source.as_bytes()).unwrap_or("");
                    import_path = Some(raw_path.trim_matches(|c| c == '"' || c == '\'' || c == '`').to_string());
                }
                "import" => {
                    import_node = Some(capture.node);
                }
                _ => {}
            }
        }

        if let (Some(path), Some(node)) = (import_path, import_node) {
            let import_type = classify_js_import(&path, alias_map);
            let line_number = node.start_position().row + 1;

            // Extract imported symbols
            let imported_symbols = extract_imported_symbols_js(source, &node);

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

/// Extract require() statements: const foo = require('module')
fn extract_require_statements(
    source: &str,
    root: &tree_sitter::Node,
    alias_map: Option<&crate::parsers::tsconfig::PathAliasMap>,
) -> Result<Vec<ImportInfo>> {
    let language = tree_sitter_typescript::LANGUAGE_TSX;

    let query_str = r#"
        (call_expression
            function: (identifier) @func_name
            arguments: (arguments (string) @require_path)) @require_call
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create require query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut imports = Vec::new();

    while let Some(match_) = matches.next() {
        let mut func_name = None;
        let mut require_path = None;
        let mut require_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            match capture_name {
                "func_name" => {
                    func_name = Some(capture.node.utf8_text(source.as_bytes()).unwrap_or(""));
                }
                "require_path" => {
                    // Remove quotes from string literal
                    let raw_path = capture.node.utf8_text(source.as_bytes()).unwrap_or("");
                    require_path = Some(raw_path.trim_matches(|c| c == '"' || c == '\'' || c == '`').to_string());
                }
                "require_call" => {
                    require_node = Some(capture.node);
                }
                _ => {}
            }
        }

        // Only process if it's actually a require() call
        if func_name == Some("require") {
            if let (Some(path), Some(node)) = (require_path, require_node) {
                let import_type = classify_js_import(&path, alias_map);
                let line_number = node.start_position().row + 1;

                imports.push(ImportInfo {
                    imported_path: path,
                    import_type,
                    line_number,
                    imported_symbols: None, // require doesn't have selective imports
                });
            }
        }
    }

    Ok(imports)
}

/// Extract the list of imported symbols from an import statement
fn extract_imported_symbols_js(source: &str, import_node: &tree_sitter::Node) -> Option<Vec<String>> {
    let mut symbols = Vec::new();

    // Walk children to find import_clause nodes
    let mut cursor = import_node.walk();
    for child in import_node.children(&mut cursor) {
        if child.kind() == "import_clause" {
            // Look for named_imports or namespace_import
            let mut clause_cursor = child.walk();
            for grandchild in child.children(&mut clause_cursor) {
                match grandchild.kind() {
                    "named_imports" => {
                        // Extract individual import specifiers
                        let mut specifier_cursor = grandchild.walk();
                        for specifier in grandchild.children(&mut specifier_cursor) {
                            if specifier.kind() == "import_specifier" {
                                // Get the name (could be aliased)
                                if let Ok(text) = specifier.utf8_text(source.as_bytes()) {
                                    // Parse "foo as bar" or just "foo"
                                    let name = text.split_whitespace().next().unwrap_or(text);
                                    symbols.push(name.to_string());
                                }
                            }
                        }
                    }
                    "identifier" => {
                        // Default import: import Foo from 'module'
                        if let Ok(text) = grandchild.utf8_text(source.as_bytes()) {
                            symbols.push(text.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    if symbols.is_empty() {
        None
    } else {
        Some(symbols)
    }
}

/// Classify a JavaScript/TypeScript import as internal, external, or stdlib
///
/// # Arguments
///
/// * `import_path` - The import path string
/// * `alias_map` - Optional tsconfig path alias mappings
///
/// Path alias imports (like `@packages/*` from tsconfig.json) are classified as Internal
/// when they match a configured alias pattern.
fn classify_js_import(import_path: &str, alias_map: Option<&crate::parsers::tsconfig::PathAliasMap>) -> ImportType {
    // Relative imports (./ or ../)
    if import_path.starts_with("./") || import_path.starts_with("../") {
        log::trace!("classify_js_import: '{}' => Internal (relative)", import_path);
        return ImportType::Internal;
    }

    // Absolute imports starting with /
    if import_path.starts_with("/") {
        log::trace!("classify_js_import: '{}' => Internal (absolute)", import_path);
        return ImportType::Internal;
    }

    // Check if import matches a configured path alias (e.g., @packages/*, ~/*)
    if let Some(map) = alias_map {
        log::trace!("classify_js_import: checking '{}' against {} aliases", import_path, map.aliases.len());
        for alias_pattern in map.aliases.keys() {
            // Check for wildcard patterns like "@packages/*"
            if alias_pattern.ends_with("/*") {
                let alias_prefix = alias_pattern.trim_end_matches("/*");
                if import_path.starts_with(alias_prefix) {
                    log::info!("classify_js_import: '{}' => Internal (matches alias pattern '{}')", import_path, alias_pattern);
                    return ImportType::Internal;
                }
            } else {
                // Exact match
                if import_path == alias_pattern {
                    log::info!("classify_js_import: '{}' => Internal (exact match alias '{}')", import_path, alias_pattern);
                    return ImportType::Internal;
                }
            }
        }
        log::trace!("classify_js_import: '{}' did not match any of {} alias patterns", import_path, map.aliases.len());
    } else {
        log::trace!("classify_js_import: no alias map provided for '{}'", import_path);
    }

    // Node.js built-in modules (stdlib)
    const STDLIB_MODULES: &[&str] = &[
        "fs", "path", "os", "crypto", "util", "events", "stream", "buffer",
        "http", "https", "net", "tls", "url", "querystring", "dns",
        "child_process", "cluster", "worker_threads", "readline",
        "zlib", "assert", "console", "module", "process", "timers",
        "vm", "string_decoder", "dgram", "v8", "perf_hooks",
        // Node.js prefixed imports (node:fs, etc.)
        "node:fs", "node:path", "node:os", "node:crypto", "node:util", "node:events",
        "node:stream", "node:buffer", "node:http", "node:https", "node:net",
    ];

    // Check if it's a stdlib module
    if STDLIB_MODULES.contains(&import_path) {
        log::trace!("classify_js_import: '{}' => Stdlib", import_path);
        return ImportType::Stdlib;
    }

    // Everything else is external (third-party packages from npm)
    log::info!("classify_js_import: '{}' => External (not alias, relative, absolute, or stdlib)", import_path);
    ImportType::External
}

// ============================================================================
// Export Extraction (for barrel export tracking)
// ============================================================================

use crate::parsers::ExportInfo;

impl TypeScriptDependencyExtractor {
    /// Extract export/re-export statements for barrel export tracking
    ///
    /// Extracts:
    /// - `export * from './module'` (wildcard re-exports)
    /// - `export { Named } from './module'` (named re-exports)
    /// - `export { default as Name } from './module'` (default re-exports)
    ///
    /// Returns ExportInfo records with the exported symbol name (None for wildcard)
    /// and the source path. The indexer will resolve these to file IDs.
    pub fn extract_export_declarations(
        source: &str,
        _alias_map: Option<&crate::parsers::tsconfig::PathAliasMap>,
    ) -> Result<Vec<ExportInfo>> {
        let mut parser = Parser::new();
        let language = tree_sitter_typescript::LANGUAGE_TSX;

        parser
            .set_language(&language.into())
            .context("Failed to set TypeScript/JavaScript language")?;

        let tree = parser
            .parse(source, None)
            .context("Failed to parse TypeScript/JavaScript source for export extraction")?;

        let root_node = tree.root_node();

        let mut exports = Vec::new();

        // Extract ES6 export statements with source paths
        exports.extend(extract_export_from_statements(source, &root_node)?);

        Ok(exports)
    }
}

/// Extract export statements that re-export from other modules
///
/// Handles:
/// - export * from './module'
/// - export { Named } from './module'
/// - export { Named as Alias } from './module'
/// - export { default as Name } from './module.vue'
fn extract_export_from_statements(
    source: &str,
    root: &tree_sitter::Node,
) -> Result<Vec<ExportInfo>> {
    let language = tree_sitter_typescript::LANGUAGE_TSX;

    // Query for export statements with a source clause
    let query_str = r#"
        (export_statement
            source: (string) @source_path) @export
    "#;

    let query = Query::new(&language.into(), query_str)
        .context("Failed to create export statement query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut exports = Vec::new();

    while let Some(match_) = matches.next() {
        let mut source_path = None;
        let mut export_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            match capture_name {
                "source_path" => {
                    // Remove quotes from string literal
                    let raw_path = capture.node.utf8_text(source.as_bytes()).unwrap_or("");
                    source_path = Some(raw_path.trim_matches(|c| c == '"' || c == '\'' || c == '`').to_string());
                }
                "export" => {
                    export_node = Some(capture.node);
                }
                _ => {}
            }
        }

        if let (Some(path), Some(node)) = (source_path, export_node) {
            let line_number = node.start_position().row + 1;

            // Extract exported symbols from this export statement
            let exported_symbols = extract_exported_symbols(source, &node)?;

            // If no specific symbols extracted (export *), create one entry with None
            if exported_symbols.is_empty() {
                exports.push(ExportInfo {
                    exported_symbol: None, // Wildcard export
                    source_path: path,
                    line_number,
                });
            } else {
                // Create one entry per exported symbol
                for symbol in exported_symbols {
                    exports.push(ExportInfo {
                        exported_symbol: Some(symbol),
                        source_path: path.clone(),
                        line_number,
                    });
                }
            }
        }
    }

    Ok(exports)
}

/// Extract the list of symbols being exported from an export statement
///
/// For `export { A, B as C } from './module'`, returns ["A", "B"]
/// For `export * from './module'`, returns empty vec (handled by caller)
fn extract_exported_symbols(source: &str, export_node: &tree_sitter::Node) -> Result<Vec<String>> {
    let mut symbols = Vec::new();

    // Walk children to find export_clause
    let mut cursor = export_node.walk();
    for child in export_node.children(&mut cursor) {
        if child.kind() == "export_clause" {
            // Extract individual export specifiers
            let mut specifier_cursor = child.walk();
            for specifier in child.children(&mut specifier_cursor) {
                if specifier.kind() == "export_specifier" {
                    // Get the exported name (before "as" if aliased)
                    // For `export { foo as bar }`, we want "foo" (the original name)
                    if let Ok(text) = specifier.utf8_text(source.as_bytes()) {
                        // Parse "foo as bar" or just "foo"
                        let name = text.split_whitespace().next().unwrap_or(text);
                        symbols.push(name.to_string());
                    }
                }
            }
        }
    }

    Ok(symbols)
}

// ============================================================================
// Path Resolution
// ============================================================================

/// Resolve a TypeScript/JavaScript import to a file path
///
/// Handles:
/// - Path aliases: `@packages/ui/store` → `../../packages/ui/store.ts` (requires tsconfig.json)
/// - Relative imports: `./components/Button` → `components/Button.tsx` or `components/Button/index.tsx`
/// - Parent directory imports: `../../utils/helper` → `../../utils/helper.ts`
/// - Index files: `./components` → `components/index.ts`
///
/// Does NOT handle:
/// - Node modules (external dependencies)
pub fn resolve_ts_import_to_path(
    import_path: &str,
    current_file_path: Option<&str>,
    alias_map: Option<&crate::parsers::tsconfig::PathAliasMap>,
) -> Option<String> {
    log::debug!("resolve_ts_import_to_path: import_path={}, current_file={:?}, has_alias_map={}",
               import_path, current_file_path, alias_map.is_some());

    // Try path alias resolution first (if alias map is provided)
    if let Some(map) = alias_map {
        log::debug!("  Trying alias resolution with {} aliases (config_dir: {:?}, base_url: {:?})",
                   map.aliases.len(), map.config_dir, map.base_url);
        if let Some(resolved_alias) = map.resolve_alias(import_path) {
            log::debug!("  Alias matched! {} => {}", import_path, resolved_alias);
            // Alias matched! Now resolve relative to the tsconfig directory
            let resolved_path = map.resolve_relative_to_config(&resolved_alias);
            let path_str = resolved_path.to_string_lossy().to_string();
            log::debug!("  After resolve_relative_to_config: {}", path_str);

            // Check if resolved path has an extension
            let has_extension = path_str.ends_with(".vue")
                || path_str.ends_with(".svelte")
                || path_str.ends_with(".ts")
                || path_str.ends_with(".tsx")
                || path_str.ends_with(".js")
                || path_str.ends_with(".jsx")
                || path_str.ends_with(".mjs")
                || path_str.ends_with(".cjs");

            if has_extension {
                log::trace!("Resolved alias {} => {}", import_path, path_str);
                return Some(path_str);
            }

            // No extension - generate candidates
            let extensions = vec![
                ".tsx", ".ts", ".jsx", ".js", ".mjs", ".cjs",
                "/index.tsx", "/index.ts", "/index.jsx", "/index.js",
            ];

            let candidates: Vec<String> = extensions
                .iter()
                .map(|ext| format!("{}{}", path_str, ext))
                .collect();

            log::trace!("Resolved alias {} => {} (candidates: {})",
                       import_path, path_str, candidates.join(" | "));
            return Some(candidates.join("|"));
        }
    }

    // Fall back to relative import resolution
    // Only handle relative imports
    if !import_path.starts_with("./") && !import_path.starts_with("../") {
        return None;
    }

    let current_file = current_file_path?;

    // Get the directory of the current file
    let current_dir = std::path::Path::new(current_file).parent()?;

    // Resolve the import path relative to current directory
    let resolved = current_dir.join(import_path);

    // Normalize the path (resolve .. and . components)
    // Use components() to properly handle . and .. without requiring filesystem access
    let normalized_path = std::path::Path::new(&resolved)
        .components()
        .fold(std::path::PathBuf::new(), |mut acc, component| {
            match component {
                std::path::Component::CurDir => acc, // Skip .
                std::path::Component::ParentDir => {
                    acc.pop(); // Go up one level for ..
                    acc
                }
                _ => {
                    acc.push(component);
                    acc
                }
            }
        });

    let normalized = normalized_path.to_string_lossy().to_string();

    // Check if the import already has a known extension
    // Vue/Svelte files are imported with their extension: import Foo from './Foo.vue'
    let has_extension = normalized.ends_with(".vue")
        || normalized.ends_with(".svelte")
        || normalized.ends_with(".ts")
        || normalized.ends_with(".tsx")
        || normalized.ends_with(".js")
        || normalized.ends_with(".jsx")
        || normalized.ends_with(".mjs")
        || normalized.ends_with(".cjs");

    if has_extension {
        // Import already has an extension - just return it normalized
        log::trace!("TS/JS import with extension: {}", normalized);
        return Some(normalized);
    }

    // No extension - try multiple file extensions in order of preference
    // TypeScript: .ts, .tsx, .d.ts
    // JavaScript: .js, .jsx, .mjs, .cjs
    // Also try index files if the import is a directory
    //
    // NOTE: We return a list of candidates separated by "|" delimiter
    // The indexer will try each one in order until it finds a match in the database
    let extensions = vec![
        ".tsx", ".ts", ".jsx", ".js", ".mjs", ".cjs",
        "/index.tsx", "/index.ts", "/index.jsx", "/index.js",
    ];

    let candidates: Vec<String> = extensions
        .iter()
        .map(|ext| format!("{}{}", normalized, ext))
        .collect();

    log::trace!("TS/JS import candidates (no extension): {}", candidates.join(" | "));

    // Return all candidates as a pipe-delimited string
    // Format: "path.tsx|path.ts|path.jsx|path.js|..."
    Some(candidates.join("|"))
}

#[cfg(test)]
mod path_resolution_tests {
    use super::*;

    #[test]
    fn test_resolve_relative_import_same_directory() {
        // import { Button } from './Button'
        let result = resolve_ts_import_to_path(
            "./Button",
            Some("src/components/App.tsx"),
            None,
        );

        assert!(result.is_some());
        let candidates = result.unwrap();
        // Should contain pipe-delimited candidates with .tsx first, then .ts, etc.
        assert!(candidates.contains("Button.tsx"));
        assert!(candidates.contains("Button.ts"));
        // First candidate should be .tsx
        assert!(candidates.starts_with("src/components/Button.tsx") || candidates.contains("/Button.tsx|"));
    }

    #[test]
    fn test_resolve_relative_import_parent_directory() {
        // import { helper} from '../utils/helper'
        let result = resolve_ts_import_to_path(
            "../utils/helper",
            Some("src/components/Button.tsx"),
            None,
        );

        assert!(result.is_some());
        let path = result.unwrap();
        assert!(path.contains("utils/helper"));
    }

    #[test]
    fn test_resolve_relative_import_multiple_parents() {
        // import { config } from '../../config/app'
        let result = resolve_ts_import_to_path(
            "../../config/app",
            Some("src/components/ui/Button.tsx"),
            None,
        );

        assert!(result.is_some());
        let path = result.unwrap();
        assert!(path.contains("config/app"));
    }

    #[test]
    fn test_resolve_index_file() {
        // import { components } from './components' (should try ./components/index.tsx)
        let result = resolve_ts_import_to_path(
            "./components",
            Some("src/App.tsx"),
            None,
        );

        assert!(result.is_some());
        // The function returns the first candidate, which is .tsx
        // In reality, the indexer would try each candidate
        assert!(result.unwrap().contains("components"));
    }

    #[test]
    fn test_absolute_import_not_supported_without_alias_map() {
        // import { Button } from '@components/Button' (requires tsconfig.json)
        let result = resolve_ts_import_to_path(
            "@components/Button",
            Some("src/App.tsx"),
            None,
        );

        // Should return None for absolute imports when no alias map provided
        assert!(result.is_none());
    }

    #[test]
    fn test_node_modules_import_not_supported() {
        // import { React } from 'react'
        let result = resolve_ts_import_to_path(
            "react",
            Some("src/App.tsx"),
            None,
        );

        // Should return None for node_modules imports
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_without_current_file() {
        let result = resolve_ts_import_to_path(
            "./Button",
            None,
            None,
        );

        // Should return None if no current file provided
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_nested_directory_structure() {
        // import { api } from './api/client'
        let result = resolve_ts_import_to_path(
            "./api/client",
            Some("src/services/http.ts"),
            None,
        );

        assert!(result.is_some());
        let path = result.unwrap();
        // Should resolve to src/services/api/client with an extension
        assert!(path.contains("api/client"));
    }
}

#[cfg(test)]
mod dependency_extraction_tests {
    use super::*;

    #[test]
    fn test_extract_basic_imports() {
        let source = r#"
            import { Button } from './components/Button';
            import React from 'react';
            import fs from 'fs';
            import '../styles.css';
        "#;

        let deps = TypeScriptDependencyExtractor::extract_dependencies(source).unwrap();

        assert_eq!(deps.len(), 4, "Should extract 4 import statements");
        assert!(deps.iter().any(|d| d.imported_path == "./components/Button"));
        assert!(deps.iter().any(|d| d.imported_path == "react"));
        assert!(deps.iter().any(|d| d.imported_path == "fs"));
        assert!(deps.iter().any(|d| d.imported_path == "../styles.css"));
    }

    #[test]
    fn test_dynamic_imports_filtered() {
        let source = r#"
            import { Button } from './components/Button';
            import React from 'react';
            const fs = require('fs');

            // Dynamic imports - should be filtered out
            const moduleName = './dynamic-module';
            import(moduleName);
            import(`./templates/${template}`);
            require(variable);
            require(CONFIG_PATH + '/settings.js');
        "#;

        let deps = TypeScriptDependencyExtractor::extract_dependencies(source).unwrap();

        // Should only find static imports (Button, React, fs)
        // Variable and template literal imports are filtered (not (string) nodes)
        assert_eq!(deps.len(), 3, "Should extract 3 static imports only");

        assert!(deps.iter().any(|d| d.imported_path == "./components/Button"));
        assert!(deps.iter().any(|d| d.imported_path == "react"));
        assert!(deps.iter().any(|d| d.imported_path == "fs"));

        // Verify dynamic imports are NOT captured
        assert!(!deps.iter().any(|d| d.imported_path.contains("moduleName")));
        assert!(!deps.iter().any(|d| d.imported_path.contains("template")));
        assert!(!deps.iter().any(|d| d.imported_path.contains("variable")));
        assert!(!deps.iter().any(|d| d.imported_path.contains("CONFIG_PATH")));
    }

    #[test]
    fn test_require_with_template_literals_filtered() {
        let source = r#"
            const path = require('path');
            const utils = require('./utils');

            // Dynamic requires with template literals - should be filtered out
            const config = require(`./config/${env}.json`);
            const plugin = require(`${PLUGIN_DIR}/loader`);
        "#;

        let deps = TypeScriptDependencyExtractor::extract_dependencies(source).unwrap();

        // Should only find static requires (path, ./utils)
        // Template literal requires are filtered (template_string nodes, not string nodes)
        assert_eq!(deps.len(), 2, "Should extract 2 static requires only");

        assert!(deps.iter().any(|d| d.imported_path == "path"));
        assert!(deps.iter().any(|d| d.imported_path == "./utils"));

        // Verify dynamic requires are NOT captured
        assert!(!deps.iter().any(|d| d.imported_path.contains("env")));
        assert!(!deps.iter().any(|d| d.imported_path.contains("PLUGIN_DIR")));
    }
}
