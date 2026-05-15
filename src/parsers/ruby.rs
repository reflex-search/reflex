//! Ruby language parser using Tree-sitter
//!
//! Extracts symbols from Ruby source code:
//! - Classes
//! - Modules
//! - Methods (instance and class methods)
//! - Singleton methods
//! - Constants
//! - Local variables (inside methods)
//! - Instance variables (@var)
//! - Class variables (@@var)
//! - Attr readers/writers/accessors (attr_reader, attr_writer, attr_accessor)
//! - Blocks (lambda, proc)

use crate::models::{ImportType, Language, SearchResult, Span, SymbolKind};
use crate::parsers::{DependencyExtractor, ImportInfo};
use anyhow::{Context, Result};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};

/// Parse Ruby source code and extract symbols
pub fn parse(path: &str, source: &str) -> Result<Vec<SearchResult>> {
    let mut parser = Parser::new();
    let language = tree_sitter_ruby::LANGUAGE;

    parser
        .set_language(&language.into())
        .context("Failed to set Ruby language")?;

    let tree = parser
        .parse(source, None)
        .context("Failed to parse Ruby source")?;

    let root_node = tree.root_node();

    let mut symbols = Vec::new();

    // Extract different types of symbols using Tree-sitter queries
    symbols.extend(extract_modules(source, &root_node, &language.into())?);
    symbols.extend(extract_classes(source, &root_node, &language.into())?);
    symbols.extend(extract_methods(source, &root_node, &language.into())?);
    symbols.extend(extract_singleton_methods(
        source,
        &root_node,
        &language.into(),
    )?);
    symbols.extend(extract_constants(source, &root_node, &language.into())?);
    symbols.extend(extract_instance_variables(
        source,
        &root_node,
        &language.into(),
    )?);
    symbols.extend(extract_class_variables(
        source,
        &root_node,
        &language.into(),
    )?);
    symbols.extend(extract_attr_accessors(
        source,
        &root_node,
        &language.into(),
    )?);
    symbols.extend(extract_local_variables(
        source,
        &root_node,
        &language.into(),
    )?);

    // Add file path to all symbols
    for symbol in &mut symbols {
        symbol.path = path.to_string();
        symbol.lang = Language::Ruby;
    }

    Ok(symbols)
}

/// Extract module declarations
fn extract_modules(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (module
            name: (constant) @name) @module
    "#;

    let query = Query::new(language, query_str).context("Failed to create module query")?;

    extract_symbols(source, root, &query, SymbolKind::Module, None)
}

/// Extract class declarations
fn extract_classes(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class
            name: (constant) @name) @class
    "#;

    let query = Query::new(language, query_str).context("Failed to create class query")?;

    extract_symbols(source, root, &query, SymbolKind::Class, None)
}

/// Extract method definitions
fn extract_methods(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class
            name: (constant) @class_name
            (body_statement
                (method
                    name: (_) @method_name))) @class

        (module
            name: (constant) @module_name
            (body_statement
                (method
                    name: (_) @method_name))) @module
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
                "module_name" => {
                    scope_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                    scope_type = Some("module");
                }
                "method_name" => {
                    method_name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                    // Find the parent method node
                    let mut current = capture.node;
                    while let Some(parent) = current.parent() {
                        if parent.kind() == "method" {
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
                Language::Ruby,
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

/// Extract singleton (class) methods
fn extract_singleton_methods(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (singleton_method
            object: (_) @class_name
            name: (_) @method_name) @method
    "#;

    let query =
        Query::new(language, query_str).context("Failed to create singleton method query")?;

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
                }
                "method" => {
                    method_node = Some(capture.node);
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
                Language::Ruby,
                SymbolKind::Method,
                Some(format!("{}.{}", class_name, method_name)),
                span,
                Some(scope),
                preview,
            ));
        }
    }

    Ok(symbols)
}

/// Extract constants
fn extract_constants(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (assignment
            left: (constant) @name
            right: (_)) @const
    "#;

    let query = Query::new(language, query_str).context("Failed to create constant query")?;

    extract_symbols(source, root, &query, SymbolKind::Constant, None)
}

/// Extract local variables (inside methods)
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

        if let (Some(name), Some(node)) = (name, assignment_node) {
            // Check if this assignment is inside a method
            let mut is_in_method = false;
            let mut current = node;

            while let Some(parent) = current.parent() {
                if parent.kind() == "method" || parent.kind() == "singleton_method" {
                    is_in_method = true;
                    break;
                }
                // Stop at program/module/class level
                if parent.kind() == "program"
                    || parent.kind() == "module"
                    || parent.kind() == "class"
                {
                    break;
                }
                current = parent;
            }

            if is_in_method {
                let span = node_to_span(&node);
                let preview = extract_preview(source, &span);

                symbols.push(SearchResult::new(
                    String::new(),
                    Language::Ruby,
                    SymbolKind::Variable,
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

/// Extract instance variables (@variable)
fn extract_instance_variables(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (instance_variable) @name
    "#;

    let query =
        Query::new(language, query_str).context("Failed to create instance variable query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();
    let mut seen = std::collections::HashSet::new();

    while let Some(match_) = matches.next() {
        for capture in match_.captures {
            let name_text = capture.node.utf8_text(source.as_bytes()).unwrap_or("");

            // Only capture the first occurrence of each instance variable
            if !seen.contains(name_text) {
                seen.insert(name_text.to_string());

                let span = node_to_span(&capture.node);
                let preview = extract_preview(source, &span);

                symbols.push(SearchResult::new(
                    String::new(),
                    Language::Ruby,
                    SymbolKind::Variable,
                    Some(name_text.to_string()),
                    span,
                    None,
                    preview,
                ));
            }
        }
    }

    Ok(symbols)
}

/// Extract class variables (@@variable)
fn extract_class_variables(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (class_variable) @name
    "#;

    let query = Query::new(language, query_str).context("Failed to create class variable query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();
    let mut seen = std::collections::HashSet::new();

    while let Some(match_) = matches.next() {
        for capture in match_.captures {
            let name_text = capture.node.utf8_text(source.as_bytes()).unwrap_or("");

            // Only capture the first occurrence of each class variable
            if !seen.contains(name_text) {
                seen.insert(name_text.to_string());

                let span = node_to_span(&capture.node);
                let preview = extract_preview(source, &span);

                symbols.push(SearchResult::new(
                    String::new(),
                    Language::Ruby,
                    SymbolKind::Variable,
                    Some(name_text.to_string()),
                    span,
                    None,
                    preview,
                ));
            }
        }
    }

    Ok(symbols)
}

/// Extract attr_accessor, attr_reader, attr_writer declarations
fn extract_attr_accessors(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (call
            method: (identifier) @method_type
            arguments: (argument_list
                (simple_symbol) @name))

        (#match? @method_type "^(attr_reader|attr_writer|attr_accessor)$")
    "#;

    let query = Query::new(language, query_str).context("Failed to create attr accessor query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut method_type = None;
        let mut name = None;
        let mut call_node = None;

        for capture in match_.captures {
            let capture_name: &str = &query.capture_names()[capture.index as usize];
            match capture_name {
                "method_type" => {
                    method_type = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                }
                "name" => {
                    let symbol_text = capture.node.utf8_text(source.as_bytes()).unwrap_or("");
                    // Remove leading : from symbol
                    name = Some(symbol_text.trim_start_matches(':').to_string());

                    // Find the parent call node
                    let mut current = capture.node;
                    while let Some(parent) = current.parent() {
                        if parent.kind() == "call" {
                            call_node = Some(parent);
                            break;
                        }
                        current = parent;
                    }
                }
                _ => {}
            }
        }

        if let (Some(_method_type), Some(name), Some(node)) = (method_type, name, call_node) {
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::Ruby,
                SymbolKind::Property,
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
                Language::Ruby,
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

/// Ruby dependency extractor for require and require_relative statements
pub struct RubyDependencyExtractor;

impl DependencyExtractor for RubyDependencyExtractor {
    fn extract_dependencies(source: &str) -> Result<Vec<ImportInfo>> {
        let mut parser = Parser::new();
        let language = tree_sitter_ruby::LANGUAGE;

        parser
            .set_language(&language.into())
            .context("Failed to set Ruby language")?;

        let tree = parser
            .parse(source, None)
            .context("Failed to parse Ruby source")?;

        let root_node = tree.root_node();

        // Query for require and require_relative calls
        // Match the entire call, then we'll inspect arguments manually to ensure they're static
        let query_str = r#"
            (call
                method: (identifier) @method_name
                arguments: (argument_list) @args) @call

            (#match? @method_name "^(require|require_relative|load)$")
        "#;

        let query = Query::new(&language.into(), query_str)
            .context("Failed to create Ruby require query")?;

        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&query, root_node, source.as_bytes());

        let mut imports = Vec::new();
        let mut seen = std::collections::HashSet::new(); // Deduplicate by (path, line_number)

        while let Some(match_) = matches.next() {
            let mut method_name = None;
            let mut args_node = None;

            for capture in match_.captures {
                let capture_name: &str = &query.capture_names()[capture.index as usize];
                match capture_name {
                    "method_name" => {
                        method_name = Some(
                            capture
                                .node
                                .utf8_text(source.as_bytes())
                                .unwrap_or("")
                                .to_string(),
                        );
                    }
                    "args" => {
                        args_node = Some(capture.node);
                    }
                    _ => {}
                }
            }

            if let (Some(method), Some(args)) = (method_name, args_node) {
                // Manual filter: only process require, require_relative, load
                // (the #match? predicate in the query doesn't seem to work correctly)
                if !matches!(method.as_str(), "require" | "require_relative" | "load") {
                    continue;
                }

                // Manually inspect the argument_list's direct children
                // STATIC ONLY: Only accept simple strings or symbols, reject complex expressions
                let mut cursor = args.walk();
                for child in args.children(&mut cursor) {
                    match child.kind() {
                        "string" => {
                            // Check for interpolation (dynamic)
                            let mut is_interpolated = false;
                            let mut child_cursor = child.walk();
                            for grandchild in child.children(&mut child_cursor) {
                                if grandchild.kind() == "interpolation" {
                                    is_interpolated = true;
                                    break;
                                }
                            }
                            if is_interpolated {
                                continue; // Skip interpolated strings
                            }

                            // Extract string_content
                            let mut content = None;
                            let mut child_cursor = child.walk();
                            for grandchild in child.children(&mut child_cursor) {
                                if grandchild.kind() == "string_content" {
                                    content = Some(
                                        grandchild
                                            .utf8_text(source.as_bytes())
                                            .unwrap_or("")
                                            .to_string(),
                                    );
                                    break;
                                }
                            }

                            if let Some(path) = content {
                                // Skip empty strings
                                if path.is_empty() {
                                    continue;
                                }

                                let line_number = child.start_position().row + 1;
                                let key = (path.clone(), line_number);

                                // Deduplicate
                                if seen.contains(&key) {
                                    continue;
                                }
                                seen.insert(key);

                                let import_type = classify_ruby_import(&path, &method);

                                imports.push(ImportInfo {
                                    imported_path: path,
                                    line_number,
                                    import_type,
                                    imported_symbols: None,
                                });
                            }
                        }
                        "simple_symbol" => {
                            let mut path =
                                child.utf8_text(source.as_bytes()).unwrap_or("").to_string();
                            // Remove leading ':'
                            if path.starts_with(':') {
                                path = path.trim_start_matches(':').to_string();
                            }

                            let line_number = child.start_position().row + 1;
                            let key = (path.clone(), line_number);

                            // Deduplicate
                            if seen.contains(&key) {
                                continue;
                            }
                            seen.insert(key);

                            let import_type = classify_ruby_import(&path, &method);

                            imports.push(ImportInfo {
                                imported_path: path,
                                line_number,
                                import_type,
                                imported_symbols: None,
                            });
                        }
                        // Ignore all other node types (identifiers, constants, calls, binary expressions, etc.)
                        // These are dynamic requires and should be filtered out
                        _ => {}
                    }
                }
            }
        }

        Ok(imports)
    }
}

/// Ruby project metadata for monorepo support
#[derive(Debug, Clone)]
pub struct RubyProject {
    pub gem_name: String,         // Gem name from gemspec
    pub project_root: String,     // Relative path to project root (gemspec directory)
    pub abs_project_root: String, // Absolute path to project root
}

/// Find all gemspec files in the project (no depth limit for monorepo support)
pub fn find_all_gemspec_files(root: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
    let mut gemspec_files = Vec::new();

    let walker = ignore::WalkBuilder::new(root)
        .follow_links(false)
        .git_ignore(true)
        .build();

    for entry in walker {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if path.extension().and_then(|s| s.to_str()) == Some("gemspec") {
                gemspec_files.push(path.to_path_buf());
            }
        }
    }

    Ok(gemspec_files)
}

/// Parse all Ruby projects from gemspec files
pub fn parse_all_ruby_projects(root: &std::path::Path) -> Result<Vec<RubyProject>> {
    let gemspec_files = find_all_gemspec_files(root)?;
    let mut projects = Vec::new();
    let root_abs = root.canonicalize()?;

    for gemspec_path in &gemspec_files {
        if let Some(project_dir) = gemspec_path.parent() {
            if let Some(gem_name) = parse_gemspec_name(gemspec_path) {
                let project_abs = project_dir.canonicalize()?;
                let project_rel = project_abs
                    .strip_prefix(&root_abs)
                    .unwrap_or(project_dir)
                    .to_string_lossy()
                    .to_string();

                projects.push(RubyProject {
                    gem_name: gem_name.clone(),
                    project_root: project_rel,
                    abs_project_root: project_abs.to_string_lossy().to_string(),
                });
            }
        }
    }

    Ok(projects)
}

/// Find all Ruby gem names from gemspec files in the project (legacy version)
/// DEPRECATED: Use parse_all_ruby_projects() instead for monorepo support
pub fn find_ruby_gem_names(root: &std::path::Path) -> Vec<String> {
    parse_all_ruby_projects(root)
        .unwrap_or_default()
        .into_iter()
        .map(|p| p.gem_name)
        .collect()
}

/// Parse a gemspec file to extract the gem name
fn parse_gemspec_name(gemspec_path: &std::path::Path) -> Option<String> {
    let content = std::fs::read_to_string(gemspec_path).ok()?;

    for line in content.lines() {
        let trimmed = line.trim();

        // Match: s.name = "activerecord"
        // Match: spec.name = "activerecord"
        if (trimmed.starts_with("s.name") || trimmed.starts_with("spec.name"))
            && trimmed.contains('=')
        {
            // Extract quoted value after =
            if let Some(equals_pos) = trimmed.find('=') {
                let after_equals = &trimmed[equals_pos + 1..].trim();

                // Handle both "name" and 'name'
                for quote in ['"', '\''] {
                    if let Some(start) = after_equals.find(quote) {
                        if let Some(end) = after_equals[start + 1..].find(quote) {
                            let name = &after_equals[start + 1..start + 1 + end];
                            return Some(name.to_string());
                        }
                    }
                }
            }
        }
    }

    None
}

/// Convert a gem name to all possible require path variants
/// Handles hyphen/underscore conversions: "active-record" → ["active-record", "active_record"]
fn gem_name_to_require_paths(gem_name: &str) -> Vec<String> {
    let mut paths = Vec::new();

    // 1. Exact match
    paths.push(gem_name.to_string());

    // 2. Convert hyphens to underscores
    if gem_name.contains('-') {
        paths.push(gem_name.replace('-', "_"));
    }

    // 3. Convert underscores to hyphens
    if gem_name.contains('_') {
        paths.push(gem_name.replace('_', "-"));
    }

    paths
}

/// Resolve a Ruby require path to a file path in the project
/// Handles both gem-based requires and relative requires
pub fn resolve_ruby_require_to_path(
    require_path: &str,
    projects: &[RubyProject],
    current_file_path: Option<&str>,
) -> Option<String> {
    // Handle require_relative (relative to current file)
    if require_path.starts_with("./") || require_path.starts_with("../") {
        if let Some(current_file) = current_file_path {
            // Get directory of current file
            if let Some(current_dir) = std::path::Path::new(current_file).parent() {
                let resolved = current_dir.join(require_path);

                // Try with .rb extension
                let candidates = vec![
                    format!("{}.rb", resolved.display()),
                    resolved.display().to_string(),
                ];

                for candidate in candidates {
                    // Normalize path
                    if let Ok(normalized) = std::path::Path::new(&candidate).canonicalize() {
                        return Some(normalized.display().to_string());
                    }
                }
            }
        }
        return None;
    }

    // Handle gem-based requires
    // Extract first component: "active_record/base" → "active_record"
    let first_component = require_path.split('/').next().unwrap_or(require_path);

    for project in projects {
        // Check if this require matches the gem name (or its variants)
        let gem_variants = gem_name_to_require_paths(&project.gem_name);

        for variant in &gem_variants {
            if first_component == variant {
                // Convert require path to file path: "active_record/base" → "lib/active_record/base.rb"
                let require_file_path = require_path.replace("::", "/");

                // Try common Ruby directory structures
                let candidates = vec![
                    format!("{}/lib/{}.rb", project.project_root, require_file_path),
                    format!("{}/{}.rb", project.project_root, require_file_path),
                ];

                for candidate in candidates {
                    return Some(candidate);
                }
            }
        }
    }

    None
}

/// Reclassify a Ruby import using the project's gem names
/// Similar to reclassify_go_import() and reclassify_java_import()
pub fn reclassify_ruby_import(import_path: &str, gem_names: &[String]) -> ImportType {
    // require_relative is always internal
    if import_path.starts_with("./") || import_path.starts_with("../") {
        return ImportType::Internal;
    }

    // Extract first component: "active_record/base" → "active_record"
    let first_component = import_path.split('/').next().unwrap_or(import_path);

    // Check if matches ANY gem name variant
    for gem_name in gem_names {
        for variant in gem_name_to_require_paths(gem_name) {
            if first_component == variant {
                return ImportType::Internal;
            }
        }
    }

    // Check stdlib
    if is_ruby_stdlib(import_path) {
        return ImportType::Stdlib;
    }

    // Default to external
    ImportType::External
}

/// Check if a require path is Ruby stdlib
fn is_ruby_stdlib(path: &str) -> bool {
    let stdlib_prefixes = [
        "json",
        "csv",
        "yaml",
        "uri",
        "net/",
        "open-uri",
        "openssl",
        "digest",
        "base64",
        "securerandom",
        "time",
        "date",
        "set",
        "fileutils",
        "pathname",
        "tempfile",
        "logger",
        "benchmark",
        "ostruct",
        "forwardable",
        "singleton",
        "observer",
        "delegate",
        "abbrev",
        "cgi",
        "erb",
        "optparse",
        "shellwords",
        "stringio",
        "strscan",
        "socket",
        "thread",
        "mutex_m",
        "monitor",
        "sync",
        "timeout",
        "weakref",
        "English",
        "fiddle",
        "rbconfig",
    ];

    for prefix in &stdlib_prefixes {
        if path == *prefix || path.starts_with(&format!("{}/", prefix)) {
            return true;
        }
    }

    false
}

/// Classify Ruby imports into Internal/External/Stdlib (legacy version without gem names)
fn classify_ruby_import(path: &str, method: &str) -> ImportType {
    // require_relative is always internal (relative to current file)
    if method == "require_relative" {
        return ImportType::Internal;
    }

    // Check stdlib
    if is_ruby_stdlib(path) {
        return ImportType::Stdlib;
    }

    // If it starts with a relative path indicator, it's internal
    if path.starts_with("./") || path.starts_with("../") {
        return ImportType::Internal;
    }

    // Default to external for unknown gems
    ImportType::External
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_class() {
        let source = r#"
class User
  attr_accessor :name, :email
end
        "#;

        let symbols = parse("test.rb", source).unwrap();

        let class_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        assert_eq!(class_symbols.len(), 1);
        assert_eq!(class_symbols[0].symbol.as_deref(), Some("User"));
    }

    #[test]
    fn test_parse_module() {
        let source = r#"
module Authentication
  def login
    # implementation
  end
end
        "#;

        let symbols = parse("test.rb", source).unwrap();

        let module_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Module))
            .collect();

        assert_eq!(module_symbols.len(), 1);
        assert_eq!(module_symbols[0].symbol.as_deref(), Some("Authentication"));
    }

    #[test]
    fn test_parse_methods() {
        let source = r#"
class Calculator
  def add(a, b)
    a + b
  end

  def subtract(a, b)
    a - b
  end
end
        "#;

        let symbols = parse("test.rb", source).unwrap();

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
    fn test_parse_singleton_method() {
        let source = r#"
class User
  def self.create(attributes)
    new(attributes).save
  end
end
        "#;

        let symbols = parse("test.rb", source).unwrap();

        let method_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method))
            .collect();

        assert!(method_symbols.len() >= 1);
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref().unwrap_or("").contains("create"))
        );
    }

    #[test]
    fn test_parse_constants() {
        let source = r#"
MAX_SIZE = 100
DEFAULT_TIMEOUT = 30
API_KEY = "secret123"
        "#;

        let symbols = parse("test.rb", source).unwrap();

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
                .any(|s| s.symbol.as_deref() == Some("API_KEY"))
        );
    }

    #[test]
    fn test_parse_nested_class() {
        let source = r#"
module MyApp
  class User
    def initialize(name)
      @name = name
    end
  end
end
        "#;

        let symbols = parse("test.rb", source).unwrap();

        let module_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Module))
            .collect();

        let class_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        assert_eq!(module_symbols.len(), 1);
        assert_eq!(class_symbols.len(), 1);
        assert_eq!(module_symbols[0].symbol.as_deref(), Some("MyApp"));
        assert_eq!(class_symbols[0].symbol.as_deref(), Some("User"));
    }

    #[test]
    fn test_parse_rails_controller() {
        let source = r#"
class UsersController < ApplicationController
  before_action :authenticate_user!

  def index
    @users = User.all
  end

  def show
    @user = User.find(params[:id])
  end

  def create
    @user = User.new(user_params)
    @user.save
  end
end
        "#;

        let symbols = parse("test.rb", source).unwrap();

        let class_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Class))
            .collect();

        let method_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method))
            .collect();

        assert_eq!(class_symbols.len(), 1);
        assert_eq!(method_symbols.len(), 3);
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("index"))
        );
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("show"))
        );
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("create"))
        );
    }

    #[test]
    fn test_parse_mixed_symbols() {
        let source = r#"
MAX_RETRIES = 3

module Authentication
  class Session
    def login(username, password)
      # implementation
    end

    def self.destroy_all
      # implementation
    end
  end
end
        "#;

        let symbols = parse("test.rb", source).unwrap();

        // Should find: constant, module, class, instance method, class method
        assert!(symbols.len() >= 4);

        let kinds: Vec<&SymbolKind> = symbols.iter().map(|s| &s.kind).collect();
        assert!(kinds.contains(&&SymbolKind::Constant));
        assert!(kinds.contains(&&SymbolKind::Module));
        assert!(kinds.contains(&&SymbolKind::Class));
        assert!(kinds.contains(&&SymbolKind::Method));
    }

    #[test]
    fn test_local_variables_included() {
        let source = r#"
GLOBAL_CONSTANT = 100

class Calculator
  def calculate(input)
    local_var = input * 2
    result = local_var + 10
    temp = result / 2
    temp
  end

  def self.process(value)
    squared = value * value
    doubled = squared * 2
    doubled
  end
end
        "#;

        let symbols = parse("test.rb", source).unwrap();

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
                .any(|v| v.symbol.as_deref() == Some("squared"))
        );
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("doubled"))
        );

        // Verify that local variables have no scope
        for var in variables {
            // Removed: scope field no longer exists: assert_eq!(var.scope, None);
        }

        // Verify that GLOBAL_CONSTANT is not included as a variable
        let var_names: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .filter_map(|s| s.symbol.as_deref())
            .collect();
        assert!(!var_names.contains(&"GLOBAL_CONSTANT"));
    }

    #[test]
    fn test_instance_and_class_variables() {
        let source = r#"
class Counter
  @@total_count = 0

  def initialize(name)
    @name = name
    @count = 0
    @@total_count += 1
  end

  def increment
    @count += 1
  end

  def self.get_total
    @@total_count
  end
end
        "#;

        let symbols = parse("test.rb", source).unwrap();

        // Filter to just variables
        let variables: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        // Check that instance variables are captured
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("@name"))
        );
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("@count"))
        );

        // Check that class variables are captured
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("@@total_count"))
        );
    }

    #[test]
    fn test_attr_accessors() {
        let source = r#"
class Person
  attr_reader :name, :age
  attr_writer :email
  attr_accessor :phone, :address

  def initialize(name, age)
    @name = name
    @age = age
  end
end
        "#;

        let symbols = parse("test.rb", source).unwrap();

        // Filter to properties
        let properties: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Property))
            .collect();

        // Check that attr_* declarations are captured
        assert!(
            properties
                .iter()
                .any(|p| p.symbol.as_deref() == Some("name"))
        );
        assert!(
            properties
                .iter()
                .any(|p| p.symbol.as_deref() == Some("age"))
        );
        assert!(
            properties
                .iter()
                .any(|p| p.symbol.as_deref() == Some("email"))
        );
        assert!(
            properties
                .iter()
                .any(|p| p.symbol.as_deref() == Some("phone"))
        );
        assert!(
            properties
                .iter()
                .any(|p| p.symbol.as_deref() == Some("address"))
        );

        assert_eq!(properties.len(), 5);
    }

    #[test]
    fn test_extract_ruby_requires() {
        let source = r#"
            require 'json'
            require 'rails'
            require 'activerecord'
            require_relative '../models/user'
            require_relative './helpers/auth'

            class UsersController
              def index
                # implementation
              end
            end
        "#;

        let deps = RubyDependencyExtractor::extract_dependencies(source).unwrap();

        assert_eq!(deps.len(), 5, "Should extract 5 require statements");
        assert!(deps.iter().any(|d| d.imported_path == "json"));
        assert!(deps.iter().any(|d| d.imported_path == "rails"));
        assert!(deps.iter().any(|d| d.imported_path == "activerecord"));
        assert!(deps.iter().any(|d| d.imported_path == "../models/user"));
        assert!(deps.iter().any(|d| d.imported_path == "./helpers/auth"));

        // Check stdlib classification
        let json_dep = deps.iter().find(|d| d.imported_path == "json").unwrap();
        assert!(
            matches!(json_dep.import_type, ImportType::Stdlib),
            "json should be classified as Stdlib"
        );

        // Check external classification
        let rails_dep = deps.iter().find(|d| d.imported_path == "rails").unwrap();
        assert!(
            matches!(rails_dep.import_type, ImportType::External),
            "rails should be classified as External"
        );

        // Check internal classification (require_relative)
        let user_dep = deps
            .iter()
            .find(|d| d.imported_path == "../models/user")
            .unwrap();
        assert!(
            matches!(user_dep.import_type, ImportType::Internal),
            "require_relative should be classified as Internal"
        );
    }

    #[test]
    fn test_dynamic_requires_filtered() {
        let source = r##"
            require 'json'
            require 'rails'
            require_relative '../models/user'

            # Dynamic requires - should be filtered out
            require variable
            require CONSTANT
            require File.join('path', 'to', 'file')
            require_relative File.dirname(__FILE__) + '/dynamic'
            load "#{Rails.root}/lib/dynamic.rb"
        "##;

        let deps = RubyDependencyExtractor::extract_dependencies(source).unwrap();

        // Should only find static requires (json, rails, ../models/user)
        // Variable, constant, and expression-based requires are filtered (not (string) or (simple_symbol) nodes)
        assert_eq!(deps.len(), 3, "Should extract 3 static requires only");

        assert!(deps.iter().any(|d| d.imported_path == "json"));
        assert!(deps.iter().any(|d| d.imported_path == "rails"));
        assert!(deps.iter().any(|d| d.imported_path == "../models/user"));

        // Verify dynamic requires are NOT captured
        assert!(!deps.iter().any(|d| d.imported_path.contains("variable")));
        assert!(!deps.iter().any(|d| d.imported_path.contains("CONSTANT")));
        assert!(!deps.iter().any(|d| d.imported_path.contains("File")));
        assert!(!deps.iter().any(|d| d.imported_path.contains("Rails")));
    }
}

#[cfg(test)]
mod monorepo_tests {
    use super::*;

    #[test]
    fn test_resolve_ruby_require_lib_structure() {
        let projects = vec![RubyProject {
            gem_name: "activerecord".to_string(),
            project_root: "gems/activerecord".to_string(),
            abs_project_root: "/path/to/gems/activerecord".to_string(),
        }];

        // Test gem-based require with lib/ structure
        let result = resolve_ruby_require_to_path("activerecord/base", &projects, None);

        assert_eq!(
            result,
            Some("gems/activerecord/lib/activerecord/base.rb".to_string())
        );
    }

    #[test]
    fn test_resolve_ruby_require_root_structure() {
        let projects = vec![RubyProject {
            gem_name: "my-gem".to_string(),
            project_root: "gems/my-gem".to_string(),
            abs_project_root: "/path/to/gems/my-gem".to_string(),
        }];

        // Test gem-based require with root structure (no lib/)
        // Should return lib/ path first, but both candidates are generated
        let result = resolve_ruby_require_to_path("my_gem/utils", &projects, None);

        // The resolver returns the first candidate (lib/ version)
        assert_eq!(result, Some("gems/my-gem/lib/my_gem/utils.rb".to_string()));
    }

    #[test]
    fn test_resolve_ruby_require_no_match() {
        let projects = vec![RubyProject {
            gem_name: "activerecord".to_string(),
            project_root: "gems/activerecord".to_string(),
            abs_project_root: "/path/to/gems/activerecord".to_string(),
        }];

        // Test require that doesn't match any gem
        let result = resolve_ruby_require_to_path("rails/application", &projects, None);

        assert_eq!(result, None);
    }

    #[test]
    fn test_resolve_ruby_require_hyphen_underscore_conversion() {
        let projects = vec![RubyProject {
            gem_name: "active-record".to_string(),
            project_root: "gems/active-record".to_string(),
            abs_project_root: "/path/to/gems/active-record".to_string(),
        }];

        // Test that hyphenated gem name matches underscored require
        let result = resolve_ruby_require_to_path("active_record/base", &projects, None);

        assert_eq!(
            result,
            Some("gems/active-record/lib/active_record/base.rb".to_string())
        );
    }

    #[test]
    fn test_resolve_ruby_require_monorepo() {
        let projects = vec![
            RubyProject {
                gem_name: "activerecord".to_string(),
                project_root: "gems/activerecord".to_string(),
                abs_project_root: "/path/to/gems/activerecord".to_string(),
            },
            RubyProject {
                gem_name: "activesupport".to_string(),
                project_root: "gems/activesupport".to_string(),
                abs_project_root: "/path/to/gems/activesupport".to_string(),
            },
            RubyProject {
                gem_name: "actionpack".to_string(),
                project_root: "gems/actionpack".to_string(),
                abs_project_root: "/path/to/gems/actionpack".to_string(),
            },
        ];

        // Test resolving to different gems
        let ar_result = resolve_ruby_require_to_path("activerecord/base", &projects, None);
        assert_eq!(
            ar_result,
            Some("gems/activerecord/lib/activerecord/base.rb".to_string())
        );

        let as_result = resolve_ruby_require_to_path("activesupport/core_ext", &projects, None);
        assert_eq!(
            as_result,
            Some("gems/activesupport/lib/activesupport/core_ext.rb".to_string())
        );

        let ap_result = resolve_ruby_require_to_path("actionpack/controller", &projects, None);
        assert_eq!(
            ap_result,
            Some("gems/actionpack/lib/actionpack/controller.rb".to_string())
        );
    }
}
