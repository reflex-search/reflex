//! Go language parser using Tree-sitter
//!
//! Extracts symbols from Go source code:
//! - Functions (func)
//! - Types (struct, interface)
//! - Methods (with receiver type)
//! - Constants (const declarations and blocks)
//! - Variables (var declarations and short declarations with :=)
//! - Packages/Imports

use crate::models::{Language, SearchResult, Span, SymbolKind};
use anyhow::{Context, Result};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};

/// Parse Go source code and extract symbols
pub fn parse(path: &str, source: &str) -> Result<Vec<SearchResult>> {
    let mut parser = Parser::new();
    let language = tree_sitter_go::LANGUAGE;

    parser
        .set_language(&language.into())
        .context("Failed to set Go language")?;

    let tree = parser
        .parse(source, None)
        .context("Failed to parse Go source")?;

    let root_node = tree.root_node();

    let mut symbols = Vec::new();

    // Extract different types of symbols using Tree-sitter queries
    symbols.extend(extract_functions(source, &root_node, &language.into())?);
    symbols.extend(extract_types(source, &root_node, &language.into())?);
    symbols.extend(extract_interfaces(source, &root_node, &language.into())?);
    symbols.extend(extract_methods(source, &root_node, &language.into())?);
    symbols.extend(extract_constants(source, &root_node, &language.into())?);
    symbols.extend(extract_variables(source, &root_node, &language.into())?);

    // Add file path to all symbols
    for symbol in &mut symbols {
        symbol.path = path.to_string();
        symbol.lang = Language::Go;
    }

    Ok(symbols)
}

/// Extract function declarations
fn extract_functions(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (function_declaration
            name: (identifier) @name) @function
    "#;

    let query = Query::new(language, query_str).context("Failed to create function query")?;

    extract_symbols(source, root, &query, SymbolKind::Function, None)
}

/// Extract type declarations (structs)
fn extract_types(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (type_declaration
            (type_spec
                name: (type_identifier) @name
                type: (struct_type))) @struct
    "#;

    let query = Query::new(language, query_str).context("Failed to create struct query")?;

    extract_symbols(source, root, &query, SymbolKind::Struct, None)
}

/// Extract interface declarations
fn extract_interfaces(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (type_declaration
            (type_spec
                name: (type_identifier) @name
                type: (interface_type))) @interface
    "#;

    let query = Query::new(language, query_str).context("Failed to create interface query")?;

    extract_symbols(source, root, &query, SymbolKind::Interface, None)
}

/// Extract method declarations (functions with receivers)
fn extract_methods(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (method_declaration
            receiver: (parameter_list
                (parameter_declaration
                    type: [(type_identifier) (pointer_type (type_identifier))] @receiver_type))
            name: (field_identifier) @method_name) @method
    "#;

    let query = Query::new(language, query_str).context("Failed to create method query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut receiver_type = None;
        let mut method_name = None;
        let mut method_node = None;

        for capture in match_.captures {
            let capture_name: &str = query.capture_names()[capture.index as usize];
            match capture_name {
                "receiver_type" => {
                    receiver_type = Some(
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

        if let (Some(receiver_type), Some(method_name), Some(node)) =
            (receiver_type, method_name, method_node)
        {
            // Clean up receiver type (remove * if pointer)
            let clean_receiver = receiver_type.trim_start_matches('*');
            let scope = format!("type {}", clean_receiver);
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::Go,
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

/// Extract constant declarations
fn extract_constants(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    let query_str = r#"
        (const_declaration
            (const_spec
                name: (identifier) @name)) @const
    "#;

    let query = Query::new(language, query_str).context("Failed to create const query")?;

    extract_symbols(source, root, &query, SymbolKind::Constant, None)
}

/// Extract variable declarations (var and := short declarations)
fn extract_variables(
    source: &str,
    root: &tree_sitter::Node,
    language: &tree_sitter::Language,
) -> Result<Vec<SearchResult>> {
    // Match both var_spec and short_var_declaration to capture all variable declarations
    let query_str = r#"
        (var_spec
            name: (identifier) @name) @var

        (short_var_declaration
            left: (expression_list (identifier) @name)) @short_var
    "#;

    let query = Query::new(language, query_str).context("Failed to create var query")?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source.as_bytes());

    let mut symbols = Vec::new();

    while let Some(match_) = matches.next() {
        let mut name = None;
        let mut decl_node = None;

        for capture in match_.captures {
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
                }
                "var" | "short_var" => {
                    decl_node = Some(capture.node);
                }
                _ => {}
            }
        }

        if let (Some(name), Some(node)) = (name, decl_node) {
            let span = node_to_span(&node);
            let preview = extract_preview(source, &span);

            symbols.push(SearchResult::new(
                String::new(),
                Language::Go,
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
                Language::Go,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_function() {
        let source = r#"
package main

func helloWorld() string {
    return "Hello, world!"
}
        "#;

        let symbols = parse("test.go", source).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("helloWorld"));
        assert!(matches!(symbols[0].kind, SymbolKind::Function));
    }

    #[test]
    fn test_parse_struct() {
        let source = r#"
package main

type User struct {
    Name string
    Age  int
}
        "#;

        let symbols = parse("test.go", source).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("User"));
        assert!(matches!(symbols[0].kind, SymbolKind::Struct));
    }

    #[test]
    fn test_parse_interface() {
        let source = r#"
package main

type Reader interface {
    Read(p []byte) (n int, err error)
}
        "#;

        let symbols = parse("test.go", source).unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].symbol.as_deref(), Some("Reader"));
        assert!(matches!(symbols[0].kind, SymbolKind::Interface));
    }

    #[test]
    fn test_parse_method() {
        let source = r#"
package main

type User struct {
    Name string
}

func (u *User) GetName() string {
    return u.Name
}

func (u User) SetName(name string) {
    u.Name = name
}
        "#;

        let symbols = parse("test.go", source).unwrap();

        let method_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method))
            .collect();

        assert_eq!(method_symbols.len(), 2);
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("GetName"))
        );
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("SetName"))
        );

        // Check scope
        for _method in method_symbols {
            // Removed: scope field no longer exists: assert_eq!(method.scope.as_ref().unwrap(), "type User");
        }
    }

    #[test]
    fn test_parse_constants() {
        let source = r#"
package main

const MaxSize = 100
const DefaultTimeout = 30

const (
    StatusActive   = 1
    StatusInactive = 2
)
        "#;

        let symbols = parse("test.go", source).unwrap();

        let const_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Constant))
            .collect();

        assert_eq!(const_symbols.len(), 4);
        assert!(
            const_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("MaxSize"))
        );
        assert!(
            const_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("DefaultTimeout"))
        );
        assert!(
            const_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("StatusActive"))
        );
        assert!(
            const_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("StatusInactive"))
        );
    }

    #[test]
    fn test_parse_variables() {
        let source = r#"
package main

var GlobalConfig Config
var (
    Logger  *log.Logger
    Version = "1.0.0"
)
        "#;

        let symbols = parse("test.go", source).unwrap();

        let var_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        assert_eq!(var_symbols.len(), 3);
        assert!(
            var_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("GlobalConfig"))
        );
        assert!(
            var_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Logger"))
        );
        assert!(
            var_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Version"))
        );
    }

    #[test]
    fn test_parse_mixed_symbols() {
        let source = r#"
package main

const DefaultPort = 8080

type Server struct {
    Port int
}

type Handler interface {
    Handle(req *Request) error
}

func (s *Server) Start() error {
    return nil
}

func NewServer(port int) *Server {
    return &Server{Port: port}
}

var globalServer *Server
        "#;

        let symbols = parse("test.go", source).unwrap();

        // Should find: const, struct, interface, method, function, var
        assert!(symbols.len() >= 6);

        let kinds: Vec<&SymbolKind> = symbols.iter().map(|s| &s.kind).collect();
        assert!(kinds.contains(&&SymbolKind::Constant));
        assert!(kinds.contains(&&SymbolKind::Struct));
        assert!(kinds.contains(&&SymbolKind::Interface));
        assert!(kinds.contains(&&SymbolKind::Method));
        assert!(kinds.contains(&&SymbolKind::Function));
        assert!(kinds.contains(&&SymbolKind::Variable));
    }

    #[test]
    fn test_parse_multiple_methods() {
        let source = r#"
package main

type Calculator struct{}

func (c *Calculator) Add(a, b int) int {
    return a + b
}

func (c *Calculator) Subtract(a, b int) int {
    return a - b
}

func (c *Calculator) Multiply(a, b int) int {
    return a * b
}
        "#;

        let symbols = parse("test.go", source).unwrap();

        let method_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Method))
            .collect();

        assert_eq!(method_symbols.len(), 3);
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Add"))
        );
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Subtract"))
        );
        assert!(
            method_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Multiply"))
        );
    }

    #[test]
    fn test_parse_type_alias() {
        let source = r#"
package main

type UserID string
type Age int

type Config struct {
    Host string
    Port int
}
        "#;

        let symbols = parse("test.go", source).unwrap();

        // Should find the Config struct (type aliases UserID and Age are type_spec but not struct_type)
        let struct_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Struct))
            .collect();

        assert_eq!(struct_symbols.len(), 1);
        assert_eq!(struct_symbols[0].symbol.as_deref(), Some("Config"));
    }

    #[test]
    fn test_parse_embedded_interface() {
        let source = r#"
package main

type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}

type ReadWriter interface {
    Reader
    Writer
}
        "#;

        let symbols = parse("test.go", source).unwrap();

        let interface_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Interface))
            .collect();

        assert_eq!(interface_symbols.len(), 3);
        assert!(
            interface_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Reader"))
        );
        assert!(
            interface_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("Writer"))
        );
        assert!(
            interface_symbols
                .iter()
                .any(|s| s.symbol.as_deref() == Some("ReadWriter"))
        );
    }

    #[test]
    fn test_local_variables_included() {
        let source = r#"
package main

var globalCount int = 10

func calculate(x int) int {
    localVar := x * 2
    var anotherLocal int = 5
    return localVar + anotherLocal
}
        "#;

        let symbols = parse("test.go", source).unwrap();

        let var_symbols: Vec<_> = symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Variable))
            .collect();

        // Should find globalCount, localVar (short declaration), and anotherLocal (var declaration)
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
    }

    #[test]
    fn test_extract_go_imports() {
        let source = r#"package main

import (
	"fmt"
	"encoding/json"
	"github.com/gin-gonic/gin"
	"myproject/internal/models"
)

func main() {
	fmt.Println("Hello")
}
"#;

        let deps = GoDependencyExtractor::extract_dependencies(source).unwrap();

        assert_eq!(deps.len(), 4, "Should extract 4 import statements");
        assert!(deps.iter().any(|d| d.imported_path == "fmt"));
        assert!(deps.iter().any(|d| d.imported_path == "encoding/json"));
        assert!(
            deps.iter()
                .any(|d| d.imported_path == "github.com/gin-gonic/gin")
        );
        assert!(
            deps.iter()
                .any(|d| d.imported_path == "myproject/internal/models")
        );

        // Check stdlib classification
        let fmt_dep = deps.iter().find(|d| d.imported_path == "fmt").unwrap();
        assert!(
            matches!(fmt_dep.import_type, ImportType::Stdlib),
            "fmt should be classified as Stdlib"
        );

        let json_dep = deps
            .iter()
            .find(|d| d.imported_path == "encoding/json")
            .unwrap();
        assert!(
            matches!(json_dep.import_type, ImportType::Stdlib),
            "encoding/json should be classified as Stdlib"
        );

        // Check external classification
        let gin_dep = deps
            .iter()
            .find(|d| d.imported_path == "github.com/gin-gonic/gin")
            .unwrap();
        assert!(
            matches!(gin_dep.import_type, ImportType::External),
            "github.com/gin-gonic/gin should be classified as External"
        );

        // Check myproject classification (ambiguous but should be External)
        let models_dep = deps
            .iter()
            .find(|d| d.imported_path == "myproject/internal/models")
            .unwrap();
        assert!(
            matches!(models_dep.import_type, ImportType::External),
            "myproject/internal/models should be classified as External"
        );
    }

    #[test]
    fn test_extract_go_imports_with_comments() {
        // Real-world Go code from Kubernetes with inline comments
        let source = r#"package main

import (
	"os"
	_ "time/tzdata" // for timeZone support in CronJob

	"k8s.io/component-base/cli"
	_ "k8s.io/component-base/logs/json/register"          // for JSON log format registration
	_ "k8s.io/component-base/metrics/prometheus/clientgo" // load all the prometheus client-go plugins
)

func main() {
	os.Exit(0)
}
"#;

        let deps = GoDependencyExtractor::extract_dependencies(source).unwrap();

        println!("Extracted {} dependencies:", deps.len());
        for dep in &deps {
            println!("  - {} (line {})", dep.imported_path, dep.line_number);
        }

        // Should extract all imports, even those with _ alias and comments
        assert!(
            deps.len() >= 4,
            "Should extract at least 4 imports, got {}",
            deps.len()
        );
        assert!(deps.iter().any(|d| d.imported_path == "os"));
        assert!(deps.iter().any(|d| d.imported_path == "time/tzdata"));
        assert!(
            deps.iter()
                .any(|d| d.imported_path == "k8s.io/component-base/cli")
        );
    }

    #[test]
    fn test_find_all_go_mods() {
        use std::fs;
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create multiple Go modules
        let service1 = root.join("services/auth");
        fs::create_dir_all(&service1).unwrap();
        fs::write(
            service1.join("go.mod"),
            "module github.com/myorg/auth\n\ngo 1.21\n",
        )
        .unwrap();

        let service2 = root.join("services/api");
        fs::create_dir_all(&service2).unwrap();
        fs::write(
            service2.join("go.mod"),
            "module github.com/myorg/api\n\ngo 1.21\n",
        )
        .unwrap();

        // Create vendor directory that should be skipped
        let vendor = root.join("vendor");
        fs::create_dir_all(&vendor).unwrap();
        fs::write(vendor.join("go.mod"), "module github.com/external/lib\n").unwrap();

        let mods = find_all_go_mods(root).unwrap();

        // Should find 2 modules (skipping vendor)
        assert_eq!(mods.len(), 2);
        assert!(mods.iter().any(|p| p.ends_with("services/auth/go.mod")));
        assert!(mods.iter().any(|p| p.ends_with("services/api/go.mod")));
    }

    #[test]
    fn test_parse_all_go_modules() {
        use std::fs;
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create multiple Go modules
        let service1 = root.join("services/auth");
        fs::create_dir_all(&service1).unwrap();
        fs::write(
            service1.join("go.mod"),
            "module github.com/myorg/auth\n\ngo 1.21\n",
        )
        .unwrap();

        let service2 = root.join("cmd/api");
        fs::create_dir_all(&service2).unwrap();
        fs::write(
            service2.join("go.mod"),
            "module github.com/myorg/api\n\ngo 1.21\n",
        )
        .unwrap();

        let modules = parse_all_go_modules(root).unwrap();

        // Should find 2 modules
        assert_eq!(modules.len(), 2);

        // Check module names
        let names: Vec<_> = modules.iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"github.com/myorg/auth"));
        assert!(names.contains(&"github.com/myorg/api"));

        // Check project roots
        for module in &modules {
            assert!(
                module.project_root.starts_with("services/")
                    || module.project_root.starts_with("cmd/")
            );
            assert!(module.abs_project_root.ends_with(&module.project_root));
        }
    }

    #[test]
    fn test_resolve_go_import() {
        use std::fs;
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // Create a Go module structure
        let myapp = root.join("myapp");
        fs::create_dir_all(myapp.join("pkg/models")).unwrap();
        fs::write(
            myapp.join("go.mod"),
            "module github.com/myorg/myapp\n\ngo 1.21\n",
        )
        .unwrap();

        let modules = parse_all_go_modules(root).unwrap();
        assert_eq!(modules.len(), 1);

        // Test sub-package import resolution
        // "github.com/myorg/myapp/pkg/models" → "myapp/pkg/models.go" or "myapp/pkg/models/models.go"
        let resolved =
            resolve_go_import_to_path("github.com/myorg/myapp/pkg/models", &modules, None);

        assert!(resolved.is_some());
        let path = resolved.unwrap();
        assert!(path.contains("myapp/pkg/models"));
        assert!(path.ends_with(".go"));
    }

    #[test]
    fn test_resolve_go_import_module_root() {
        use std::fs;
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        let root = temp.path();

        let myapp = root.join("cmd/server");
        fs::create_dir_all(&myapp).unwrap();
        fs::write(
            myapp.join("go.mod"),
            "module github.com/myorg/server\n\ngo 1.21\n",
        )
        .unwrap();

        let modules = parse_all_go_modules(root).unwrap();

        // Test module root import (no sub-package)
        let resolved = resolve_go_import_to_path("github.com/myorg/server", &modules, None);

        assert!(resolved.is_some());
        let path = resolved.unwrap();
        // Should try main.go or server.go
        assert!(path.contains("cmd/server"));
        assert!(path.ends_with(".go"));
    }

    #[test]
    fn test_resolve_go_import_not_found() {
        use std::fs;
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        let root = temp.path();

        let myapp = root.join("myapp");
        fs::create_dir_all(&myapp).unwrap();
        fs::write(
            myapp.join("go.mod"),
            "module github.com/myorg/myapp\n\ngo 1.21\n",
        )
        .unwrap();

        let modules = parse_all_go_modules(root).unwrap();

        // Try to resolve an import for a different module
        let resolved = resolve_go_import_to_path("github.com/other/package", &modules, None);

        // Should return None for modules not in the monorepo
        assert!(resolved.is_none());
    }

    #[test]
    fn test_resolve_go_import_relative() {
        let modules = vec![];

        // Relative imports are not supported yet
        let resolved =
            resolve_go_import_to_path("./utils", &modules, Some("myapp/pkg/api/handler.go"));

        assert!(resolved.is_none());
    }

    #[test]
    fn test_resolve_go_import_root_module_no_leading_slash() {
        // When go.mod is at the repo root, project_root is "" and paths must not
        // start with "/" (which would cause ambiguous fuzzy matches in the DB).
        use std::fs;
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        let root = temp.path();

        // go.mod at repo root → project_root = ""
        fs::write(root.join("go.mod"), "module k8s.io/kubernetes\n\ngo 1.21\n").unwrap();

        let modules = parse_all_go_modules(root).unwrap();
        assert_eq!(modules.len(), 1);
        assert_eq!(modules[0].project_root, "");

        // Sub-package import
        let resolved =
            resolve_go_import_to_path("k8s.io/kubernetes/test/internal/metric", &modules, None);
        assert!(resolved.is_some());
        let path = resolved.unwrap();
        assert!(
            !path.starts_with('/'),
            "path must not start with '/': {}",
            path
        );
        assert!(path.ends_with(".go"));
        assert!(path.contains("test/internal/metric"));

        // Module-root import
        let resolved = resolve_go_import_to_path("k8s.io/kubernetes", &modules, None);
        assert!(resolved.is_some());
        let path = resolved.unwrap();
        assert!(
            !path.starts_with('/'),
            "path must not start with '/': {}",
            path
        );
        assert!(path.ends_with(".go"));
    }
}

// ============================================================================
// Dependency Extraction
// ============================================================================

use crate::models::ImportType;
use crate::parsers::{DependencyExtractor, ImportInfo};

/// Go dependency extractor
pub struct GoDependencyExtractor;

impl DependencyExtractor for GoDependencyExtractor {
    fn extract_dependencies(source: &str) -> Result<Vec<ImportInfo>> {
        let mut parser = Parser::new();
        let language = tree_sitter_go::LANGUAGE;

        parser
            .set_language(&language.into())
            .context("Failed to set Go language")?;

        let tree = parser
            .parse(source, None)
            .context("Failed to parse Go source")?;

        let root_node = tree.root_node();

        let mut imports = Vec::new();

        // Extract import statements
        imports.extend(extract_go_imports(source, &root_node)?);

        Ok(imports)
    }
}

/// Extract Go import statements
fn extract_go_imports(source: &str, root: &tree_sitter::Node) -> Result<Vec<ImportInfo>> {
    let language = tree_sitter_go::LANGUAGE;

    // Go imports can be single or in groups
    let query_str = r#"
        (import_declaration
            (import_spec
                path: (interpreted_string_literal) @import_path)) @import

        (import_declaration
            (import_spec_list
                (import_spec
                    path: (interpreted_string_literal) @import_path))) @import
    "#;

    let query =
        Query::new(&language.into(), query_str).context("Failed to create Go import query")?;

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
                    // Remove quotes from string literal
                    let raw_path = capture.node.utf8_text(source.as_bytes()).unwrap_or("");
                    import_path = Some(raw_path.trim_matches('"').to_string());
                }
                "import" => {
                    import_node = Some(capture.node);
                }
                _ => {}
            }
        }

        if let (Some(path), Some(node)) = (import_path, import_node) {
            let import_type = classify_go_import(&path);
            let line_number = node.start_position().row + 1;

            imports.push(ImportInfo {
                imported_path: path,
                import_type,
                line_number,
                imported_symbols: None, // Go imports entire packages, not selective symbols
            });
        }
    }

    Ok(imports)
}

/// Find and parse go.mod to extract module name
/// Returns None if go.mod not found or module name can't be parsed
pub fn find_go_module_name(root: &std::path::Path) -> Option<String> {
    // Look for go.mod in root directory
    let go_mod_path = root.join("go.mod");
    if !go_mod_path.exists() {
        return None;
    }

    // Read go.mod and extract module name
    let content = std::fs::read_to_string(&go_mod_path).ok()?;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("module ") {
            // Extract module name: "module k8s.io/kubernetes" -> "k8s.io/kubernetes"
            let module_name = trimmed.strip_prefix("module ").unwrap_or("").trim();
            return Some(module_name.to_string());
        }
    }

    None
}

/// Reclassify a Go import based on the module prefix
/// This should be called by the indexer after extraction to correctly identify internal imports
pub fn reclassify_go_import(import_path: &str, module_prefix: Option<&str>) -> ImportType {
    classify_go_import_impl(import_path, module_prefix)
}

/// Classify a Go import as internal, external, or stdlib
fn classify_go_import(import_path: &str) -> ImportType {
    classify_go_import_impl(import_path, None)
}

/// Internal implementation of Go import classification
fn classify_go_import_impl(import_path: &str, module_prefix: Option<&str>) -> ImportType {
    // If we have a module prefix, check if import starts with it → Internal
    if let Some(prefix) = module_prefix {
        if import_path.starts_with(prefix) {
            return ImportType::Internal;
        }
        // Also check for multi-module repos - imports starting with k8s.io/* for Kubernetes
        // Extract the domain portion and check if it matches
        if let Some(import_domain) = import_path.split('/').next()
            && let Some(module_domain) = prefix.split('/').next()
        {
            // If domains match (e.g., both start with k8s.io), consider it internal
            if import_domain == module_domain && module_domain.contains('.') {
                return ImportType::Internal;
            }
        }
    }
    // Relative imports (./ or ../) - rare in Go but possible
    if import_path.starts_with("./") || import_path.starts_with("../") {
        return ImportType::Internal;
    }

    // Internal imports often start with company domain or project path
    // Check for common patterns like github.com/your-org/project
    // For now, we'll consider anything that looks like a full URL path as external
    // and short stdlib-like paths as stdlib

    // Go standard library modules (common ones)
    const STDLIB_MODULES: &[&str] = &[
        "fmt",
        "io",
        "os",
        "path",
        "strings",
        "bytes",
        "bufio",
        "errors",
        "context",
        "sync",
        "time",
        "encoding/json",
        "encoding/xml",
        "encoding/csv",
        "net/http",
        "net/url",
        "net",
        "crypto",
        "crypto/tls",
        "crypto/sha256",
        "database/sql",
        "log",
        "math",
        "regexp",
        "strconv",
        "sort",
        "reflect",
        "runtime",
        "testing",
        "flag",
        "filepath",
        "unicode",
        "html",
        "text/template",
    ];

    // Check if it's a stdlib module
    if STDLIB_MODULES.contains(&import_path) {
        return ImportType::Stdlib;
    }

    // If it contains a domain (has dots and slashes), it's external
    if import_path.contains('/') && import_path.split('/').next().unwrap_or("").contains('.') {
        return ImportType::External;
    }

    // Short paths without domains are likely stdlib
    if !import_path.contains('/') || import_path.split('/').count() <= 2 {
        return ImportType::Stdlib;
    }

    // Everything else is external
    ImportType::External
}

// ============================================================================
// Monorepo Support & Path Resolution
// ============================================================================

/// Represents a Go module with its location
#[derive(Debug, Clone)]
pub struct GoModule {
    /// Module name (e.g., "k8s.io/kubernetes", "github.com/myorg/myproject")
    pub name: String,
    /// Project root relative to index root (e.g., "services/api")
    pub project_root: String,
    /// Absolute path to project root
    pub abs_project_root: std::path::PathBuf,
}

/// Recursively find all go.mod files in the repository, respecting .gitignore
pub fn find_all_go_mods(index_root: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
    use ignore::WalkBuilder;

    let mut go_mod_files = Vec::new();

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

        // Look for go.mod files
        if filename == "go.mod" {
            // Skip vendor directories. Normalize separators so the check
            // works on Windows (`\vendor\`) as well as Unix (`/vendor/`).
            let path_str = path.to_string_lossy().replace('\\', "/");
            if path_str.contains("/vendor/") {
                log::trace!("Skipping go.mod in vendor directory: {:?}", path);
                continue;
            }

            go_mod_files.push(path.to_path_buf());
        }
    }

    log::debug!("Found {} go.mod files", go_mod_files.len());
    Ok(go_mod_files)
}

/// Parse all Go modules in a monorepo and track their project roots
pub fn parse_all_go_modules(index_root: &std::path::Path) -> Result<Vec<GoModule>> {
    let go_mod_files = find_all_go_mods(index_root)?;

    if go_mod_files.is_empty() {
        log::debug!("No go.mod files found in {:?}", index_root);
        return Ok(Vec::new());
    }

    let mut modules = Vec::new();
    let mod_count = go_mod_files.len();

    for go_mod_path in &go_mod_files {
        let project_root = go_mod_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("go.mod has no parent directory"))?;

        // Read and parse go.mod to extract module name
        if let Ok(content) = std::fs::read_to_string(go_mod_path) {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("module ") {
                    let module_name = trimmed
                        .strip_prefix("module ")
                        .unwrap_or("")
                        .trim()
                        .to_string();

                    // Normalize to forward slashes so import resolution and
                    // assertions on project_root behave the same on every OS.
                    let relative_project_root = project_root
                        .strip_prefix(index_root)
                        .unwrap_or(project_root)
                        .to_string_lossy()
                        .replace('\\', "/");

                    log::debug!(
                        "Found Go module '{}' at {:?}",
                        module_name,
                        relative_project_root
                    );

                    modules.push(GoModule {
                        name: module_name,
                        project_root: relative_project_root,
                        abs_project_root: project_root.to_path_buf(),
                    });
                    break;
                }
            }
        }
    }

    log::info!(
        "Loaded {} Go modules from {} go.mod files",
        modules.len(),
        mod_count
    );

    Ok(modules)
}

/// Resolve a Go import to a file path
///
/// Handles:
/// - Internal imports: `mymodule/pkg/utils` → `pkg/utils.go` or `pkg/utils/utils.go`
/// - Sub-packages: `mymodule/internal/models` → `internal/models/models.go`
/// - Relative imports: `./utils` (rare in Go but possible)
pub fn resolve_go_import_to_path(
    import_path: &str,
    modules: &[GoModule],
    _current_file_path: Option<&str>,
) -> Option<String> {
    // Handle relative imports (rare in Go)
    if import_path.starts_with("./") || import_path.starts_with("../") {
        // Go relative imports are rare and complex - skip for now
        return None;
    }

    // Find matching module
    for module in modules {
        if import_path.starts_with(&module.name) {
            // Strip module name to get sub-package path
            // "k8s.io/kubernetes/pkg/api" with module "k8s.io/kubernetes" → "pkg/api"
            let sub_path = import_path
                .strip_prefix(&module.name)
                .unwrap_or(import_path)
                .trim_start_matches('/');

            if sub_path.is_empty() {
                // Importing the module root - could be multiple files
                // Try common patterns
                let basename = module.name.split('/').next_back().unwrap_or("main");
                let candidates = if module.project_root.is_empty() {
                    vec!["main.go".to_string(), format!("{}.go", basename)]
                } else {
                    vec![
                        format!("{}/main.go", module.project_root),
                        format!("{}/{}.go", module.project_root, basename),
                    ]
                };

                if let Some(candidate) = candidates.into_iter().next() {
                    log::trace!("Checking Go module root: {}", candidate);
                    return Some(candidate);
                }
            } else {
                // Sub-package import
                // Try both single file and package directory patterns
                let package_name = sub_path.split('/').next_back().unwrap_or(sub_path);
                let candidates = if module.project_root.is_empty() {
                    vec![
                        format!("{}.go", sub_path),
                        format!("{}/{}.go", sub_path, package_name),
                    ]
                } else {
                    vec![
                        format!("{}/{}.go", module.project_root, sub_path),
                        format!("{}/{}/{}.go", module.project_root, sub_path, package_name),
                    ]
                };

                if let Some(candidate) = candidates.into_iter().next() {
                    log::trace!("Checking Go package path: {}", candidate);
                    return Some(candidate);
                }
            }
        }
    }

    None
}
