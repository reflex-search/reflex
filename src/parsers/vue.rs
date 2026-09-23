//! Vue Single File Component (SFC) parser
//!
//! Extracts symbols from Vue components:
//! - Component exports (default export from script)
//! - Functions and methods
//! - Composables (useX functions)
//! - Variables and constants (const, let, var at all scopes)
//! - Script setup declarations
//!
//! Vue SFCs contain multiple sections: template, script, and style.
//! This parser focuses on extracting symbols from the script sections.
//!
//! Note: This parser uses regex-based extraction for script blocks since
//! tree-sitter-vue is not compatible with tree-sitter 0.24+.

use crate::models::{Language, SearchResult, Span, SymbolKind};
use crate::parsers::typescript::TypeScriptDependencyExtractor;
use crate::parsers::{DependencyExtractor, ImportInfo};
use anyhow::{Context, Result};
use tree_sitter::Parser;

const SYMQ_0: &str = r#"
        (function_declaration
            name: (identifier) @name) @function
    "#;
const SYMQ_1: &str = r#"
        (lexical_declaration
            (variable_declarator
                name: (identifier) @name
                value: (arrow_function))) @arrow_fn

        (variable_declaration
            (variable_declarator
                name: (identifier) @name
                value: (arrow_function))) @arrow_fn
    "#;
const SYMQ_2: &str = r#"
        (lexical_declaration
            (variable_declarator
                name: (identifier) @name)) @decl

        (variable_declaration
            (variable_declarator
                name: (identifier) @name)) @decl
    "#;

/// Every symbol query of this module, run as ONE query per file (see
/// `crate::parsers::LanguageQueries`).
static SYMBOL_QUERIES: crate::parsers::LanguageQueries =
    crate::parsers::LanguageQueries::new(&[SYMQ_0, SYMQ_1, SYMQ_2]);

/// Parse Vue SFC and extract symbols
pub fn parse(path: &str, source: &str) -> Result<Vec<SearchResult>> {
    let mut symbols = Vec::new();

    // Extract script blocks using regex (more robust than outdated tree-sitter-vue)
    let script_blocks = extract_script_blocks(source)?;

    // Parse each script block with the TypeScript parser
    for (script_source, script_offset) in script_blocks {
        let script_symbols = parse_script_block(path, &script_source, script_offset)?;
        symbols.extend(script_symbols);
    }

    Ok(symbols)
}

/// Extract script blocks from Vue SFC using regex
/// Returns (source_code, line_offset) for each script block
fn extract_script_blocks(source: &str) -> Result<Vec<(String, usize)>> {
    let mut script_blocks = Vec::new();

    // Find all <script> blocks (handles <script>, <script setup>, <script lang="ts">, etc.)
    let lines: Vec<&str> = source.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];

        // Check if this line starts a script tag
        if line.trim_start().starts_with("<script") {
            // Find the end of the opening tag
            let mut tag_line = i;
            let mut tag_end_found = false;

            while tag_line < lines.len() {
                if lines[tag_line].contains('>') {
                    tag_end_found = true;
                    break;
                }
                tag_line += 1;
            }

            if !tag_end_found {
                i += 1;
                continue;
            }

            // Find the closing </script> tag
            let mut close_line = tag_line + 1;
            let mut close_found = false;

            while close_line < lines.len() {
                if lines[close_line].trim_start().starts_with("</script>") {
                    close_found = true;
                    break;
                }
                close_line += 1;
            }

            if close_found {
                // Extract the script content (lines between opening and closing tags)
                let script_start = tag_line + 1;
                let script_end = close_line;

                if script_start < script_end {
                    let script_content = lines[script_start..script_end].join("\n");
                    script_blocks.push((script_content, script_start));
                }

                i = close_line + 1;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    Ok(script_blocks)
}

/// Parse a script block using TypeScript parser
fn parse_script_block(
    path: &str,
    script_source: &str,
    line_offset: usize,
) -> Result<Vec<SearchResult>> {
    let mut parser = Parser::new();

    // Use TSX parser to handle both TypeScript and JavaScript
    let ts_language: tree_sitter::Language = tree_sitter_typescript::LANGUAGE_TSX.into();

    parser
        .set_language(&ts_language)
        .context("Failed to set TypeScript language for script block")?;

    let tree = parser
        .parse(script_source, None)
        .context("Failed to parse script block")?;

    let root_node = tree.root_node();
    let table = SYMBOL_QUERIES.run(&ts_language, &root_node, script_source)?;

    let mut symbols = Vec::new();

    // Extract symbols from the script block
    symbols.extend(extract_functions(script_source, &table, line_offset)?);
    symbols.extend(extract_arrow_functions(script_source, &table, line_offset)?);
    symbols.extend(extract_variables(script_source, &table, line_offset)?);

    // Add file path and language to all symbols
    for symbol in &mut symbols {
        symbol.path = path.to_string();
        symbol.lang = Language::Vue;
    }

    Ok(symbols)
}

/// Extract regular function declarations
fn extract_functions(
    source: &str,
    table: &crate::parsers::MatchTable<'_>,
    line_offset: usize,
) -> Result<Vec<SearchResult>> {
    extract_symbols(source, table, 0, SymbolKind::Function, None, line_offset)
}

/// Extract arrow functions
fn extract_arrow_functions(
    source: &str,
    table: &crate::parsers::MatchTable<'_>,
    line_offset: usize,
) -> Result<Vec<SearchResult>> {
    extract_symbols(source, table, 1, SymbolKind::Function, None, line_offset)
}

/// Extract variable and constant declarations (const, let, var at all scopes)
fn extract_variables(
    source: &str,
    table: &crate::parsers::MatchTable<'_>,
    line_offset: usize,
) -> Result<Vec<SearchResult>> {
    let query = table.query();
    let matches = table.sub(2);

    let mut symbols = Vec::new();

    for match_ in matches {
        let mut name = None;
        let mut declarator_node = None;
        let mut decl_node = None;

        for capture in &match_.captures {
            let capture_name: &str = query.capture_names()[capture.index as usize];
            match capture_name {
                "name" => {
                    name = Some(
                        capture
                            .node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string(),
                    );
                    if let Some(parent) = capture.node.parent()
                        && parent.kind() == "variable_declarator"
                    {
                        declarator_node = Some(parent);
                    }
                }
                "decl" => {
                    decl_node = Some(capture.node);
                }
                _ => {}
            }
        }

        if let (Some(name), Some(declarator), Some(decl)) = (name, declarator_node, decl_node) {
            // Check if this is an arrow function (skip those, handled separately)
            let mut is_arrow_function = false;
            for i in 0..declarator.child_count() {
                if let Some(child) = declarator.child(i as u32)
                    && child.kind() == "arrow_function"
                {
                    is_arrow_function = true;
                    break;
                }
            }

            if !is_arrow_function {
                // Determine the kind based on the keyword (const vs let/var)
                let decl_text = decl.utf8_text(source.as_bytes()).unwrap_or("");
                let kind = if decl_text.trim_start().starts_with("const") {
                    SymbolKind::Constant
                } else {
                    SymbolKind::Variable
                };

                let span = node_to_span(&decl, line_offset);
                let preview = extract_preview(source, &decl);

                symbols.push(SearchResult::new(
                    String::new(),
                    Language::Vue,
                    kind,
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

/// Generic symbol extraction helper
fn extract_symbols(
    source: &str,
    table: &crate::parsers::MatchTable<'_>,
    sub: usize,
    kind: SymbolKind,
    scope: Option<String>,
    line_offset: usize,
) -> Result<Vec<SearchResult>> {
    let query = table.query();
    let matches = table.sub(sub);

    let mut symbols = Vec::new();

    for match_ in matches {
        let mut name = None;
        let mut full_node = None;

        for capture in &match_.captures {
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
                full_node = Some(capture.node);
            }
        }

        if let (Some(name), Some(node)) = (name, full_node) {
            let span = node_to_span(&node, line_offset);
            let preview = extract_preview(source, &node);

            symbols.push(SearchResult::new(
                String::new(),
                Language::Vue,
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

/// Convert a Tree-sitter node to a Span with line offset
fn node_to_span(node: &tree_sitter::Node, line_offset: usize) -> Span {
    let start = node.start_position();
    let end = node.end_position();

    Span::new(
        start.row + 1 + line_offset,
        start.column,
        end.row + 1 + line_offset,
        end.column,
    )
}

/// Extract a preview (7 lines) around the symbol
fn extract_preview(source: &str, node: &tree_sitter::Node) -> String {
    // Starts at the node, not at byte 0: see `crate::parsers::preview`.
    crate::parsers::preview::extract_preview_for_node(source, node)
}

/// Vue dependency extractor
pub struct VueDependencyExtractor;

impl DependencyExtractor for VueDependencyExtractor {
    fn extract_dependencies(source: &str) -> Result<Vec<ImportInfo>> {
        // Delegate to the version without alias map for compatibility
        Self::extract_dependencies_with_alias_map(source, None)
    }
}

impl VueDependencyExtractor {
    /// Extract dependencies with optional tsconfig alias map support
    ///
    /// This version properly classifies path alias imports (like @packages/*, ~/*) as Internal
    /// when they match configured aliases from tsconfig.json.
    pub fn extract_dependencies_with_alias_map(
        source: &str,
        alias_map: Option<&crate::parsers::tsconfig::PathAliasMap>,
    ) -> Result<Vec<ImportInfo>> {
        // Extract script blocks from Vue SFC
        let script_blocks = extract_script_blocks(source)?;

        let mut all_imports = Vec::new();

        // Extract dependencies from each script block
        for (script_source, line_offset) in script_blocks {
            // Use TypeScript dependency extractor for the script content with alias map
            match TypeScriptDependencyExtractor::extract_dependencies_with_alias_map(
                &script_source,
                alias_map,
            ) {
                Ok(mut imports) => {
                    // Adjust line numbers to account for the script block offset in the Vue file
                    for import in &mut imports {
                        import.line_number += line_offset;
                    }
                    all_imports.extend(imports);
                }
                Err(e) => {
                    log::warn!(
                        "Failed to extract dependencies from Vue script block: {}",
                        e
                    );
                }
            }
        }

        Ok(all_imports)
    }

    /// Imports and re-exports of every `<script>` block, each block parsed once.
    ///
    /// Same rows as `extract_dependencies_with_alias_map` followed by
    /// `extract_export_declarations`.
    pub fn extract_dependencies_and_exports(
        source: &str,
        alias_map: Option<&crate::parsers::tsconfig::PathAliasMap>,
    ) -> Result<(Vec<ImportInfo>, Vec<crate::parsers::ExportInfo>)> {
        let script_blocks = extract_script_blocks(source)?;

        let mut all_imports = Vec::new();
        let mut all_exports = Vec::new();

        for (script_source, line_offset) in script_blocks {
            match TypeScriptDependencyExtractor::extract_dependencies_and_exports(
                &script_source,
                alias_map,
            ) {
                Ok((mut imports, mut exports)) => {
                    // Adjust line numbers to account for the script block offset in the Vue file
                    for import in &mut imports {
                        import.line_number += line_offset;
                    }
                    for export in &mut exports {
                        export.line_number += line_offset;
                    }
                    all_imports.extend(imports);
                    all_exports.extend(exports);
                }
                Err(e) => {
                    log::warn!(
                        "Failed to extract dependencies from Vue script block: {}",
                        e
                    );
                }
            }
        }

        Ok((all_imports, all_exports))
    }

    /// Extract export/re-export statements for barrel export tracking
    ///
    /// Extracts exports from script blocks in Vue SFCs.
    pub fn extract_export_declarations(
        source: &str,
        alias_map: Option<&crate::parsers::tsconfig::PathAliasMap>,
    ) -> Result<Vec<crate::parsers::ExportInfo>> {
        // Extract script blocks from Vue SFC
        let script_blocks = extract_script_blocks(source)?;

        let mut all_exports = Vec::new();

        // Extract exports from each script block
        for (script_source, line_offset) in script_blocks {
            // Use TypeScript export extractor for the script content
            match TypeScriptDependencyExtractor::extract_export_declarations(
                &script_source,
                alias_map,
            ) {
                Ok(mut exports) => {
                    // Adjust line numbers to account for the script block offset in the Vue file
                    for export in &mut exports {
                        export.line_number += line_offset;
                    }
                    all_exports.extend(exports);
                }
                Err(e) => {
                    log::warn!("Failed to extract exports from Vue script block: {}", e);
                }
            }
        }

        Ok(all_exports)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_vue_sfc_with_script() {
        let source = r#"
<template>
  <div>{{ message }}</div>
</template>

<script>
const message = 'Hello Vue!'

function greet() {
  console.log(message)
}
</script>

<style scoped>
div {
  color: blue;
}
</style>
"#;

        let symbols = parse("test.vue", source).unwrap();
        // Should extract message constant and greet function
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("message"))
        );
        assert!(symbols.iter().any(|s| s.symbol.as_deref() == Some("greet")));
    }

    #[test]
    fn test_parse_vue_sfc_with_script_setup() {
        let source = r#"
<template>
  <div>{{ count }}</div>
</template>

<script setup>
import { ref } from 'vue'

const count = ref(0)
const increment = () => {
  count.value++
}
</script>
"#;

        let symbols = parse("test.vue", source).unwrap();
        // Should extract count and increment
        assert!(symbols.iter().any(|s| s.symbol.as_deref() == Some("count")));
        assert!(
            symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("increment"))
        );
    }

    #[test]
    fn test_parse_vue_sfc_with_typescript() {
        let source = r#"
<template>
  <div>{{ message }}</div>
</template>

<script lang="ts">
interface User {
  name: string;
  age: number;
}

const user: User = {
  name: 'Alice',
  age: 30
}
</script>
"#;

        let symbols = parse("test.vue", source).unwrap();
        // Should extract user constant
        assert!(symbols.iter().any(|s| s.symbol.as_deref() == Some("user")));
    }

    #[test]
    fn test_local_variables_included() {
        let source = r#"
<template>
  <div>{{ result }}</div>
</template>

<script setup>
const API_KEY = 'secret123'

function calculate(input) {
  let localVar = input * 2
  var result = localVar + 10
  const temp = result / 2
  return temp
}

function process(value) {
  let squared = value * value
  var doubled = squared * 2
  return doubled
}
</script>
"#;

        let symbols = parse("test.vue", source).unwrap();

        // Filter to variables and constants
        let variables: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        let constants: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Constant))
            .collect();

        // Check that local variables (let/var) are captured
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
                .any(|v| v.symbol.as_deref() == Some("squared"))
        );
        assert!(
            variables
                .iter()
                .any(|v| v.symbol.as_deref() == Some("doubled"))
        );

        // Check that const declarations are captured as constants
        assert!(
            constants
                .iter()
                .any(|c| c.symbol.as_deref() == Some("API_KEY"))
        );
        assert!(
            constants
                .iter()
                .any(|c| c.symbol.as_deref() == Some("temp"))
        );

        // Verify that all have no scope
        for _var in variables {
            // Removed: scope field no longer exists: assert_eq!(var.scope, None);
        }
        for _constant in constants {
            // Removed: scope field no longer exists: assert_eq!(constant.scope, None);
        }
    }

    #[test]
    fn test_extract_vue_imports() {
        let source = r#"
<template>
  <div>{{ count }}</div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'
import MyComponent from './MyComponent.vue'
import { helper } from '../utils/helpers'

const count = ref(0)
const router = useRouter()
</script>

<style scoped>
div { color: blue; }
</style>
"#;

        let deps = VueDependencyExtractor::extract_dependencies(source).unwrap();

        assert!(
            deps.len() >= 5,
            "Should extract at least 5 imports, got {}",
            deps.len()
        );

        // Check for specific imports
        assert!(deps.iter().any(|d| d.imported_path == "vue"));
        assert!(deps.iter().any(|d| d.imported_path == "vue-router"));
        assert!(deps.iter().any(|d| d.imported_path == "axios"));
        assert!(deps.iter().any(|d| d.imported_path == "./MyComponent.vue"));
        assert!(deps.iter().any(|d| d.imported_path == "../utils/helpers"));

        // Verify line numbers are adjusted for script block offset
        // Script block starts around line 6, so imports should have line numbers >= 7
        for dep in &deps {
            assert!(
                dep.line_number >= 7,
                "Import line number should be >= 7, got {}",
                dep.line_number
            );
        }
    }
}
