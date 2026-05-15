//! MCP (Model Context Protocol) server implementation
//!
//! This module implements the MCP protocol directly over stdio using JSON-RPC 2.0.
//! It exposes Reflex's code search capabilities as MCP tools for AI coding assistants.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use crate::cache::CacheManager;
use crate::dependency::DependencyIndex;
use crate::indexer::Indexer;
use crate::models::{IndexConfig, IndexStatus, Language, SymbolKind};
use crate::query::{QueryEngine, QueryFilter};

/// Default preview truncation length for MCP responses (characters).
/// Raised from 100 so typical Rust/TS/Python function signatures fit without truncation.
const DEFAULT_MCP_PREVIEW_LENGTH: usize = 180;

/// JSON-RPC 2.0 request
#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

/// JSON-RPC 2.0 response
#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    // Per JSON-RPC 2.0, Notifications must omit `id` entirely (not set it to null).
    // We never construct a JsonRpcResponse for a Notification, but skip_serializing_if
    // is a defensive guard against ever emitting `"id": null`, which strict clients
    // (e.g. Claude Code's Zod validators) reject.
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error
#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

/// Parse language string to Language enum
fn parse_language(lang: Option<String>) -> Option<Language> {
    lang.as_deref().and_then(|s| match s.to_lowercase().as_str() {
        "rust" | "rs" => Some(Language::Rust),
        "javascript" | "js" => Some(Language::JavaScript),
        "typescript" | "ts" => Some(Language::TypeScript),
        "vue" => Some(Language::Vue),
        "svelte" => Some(Language::Svelte),
        "php" => Some(Language::PHP),
        "python" | "py" => Some(Language::Python),
        "go" => Some(Language::Go),
        "java" => Some(Language::Java),
        "c" => Some(Language::C),
        "cpp" | "c++" => Some(Language::Cpp),
        _ => None,
    })
}

/// Parse symbol kind string to SymbolKind enum
fn parse_symbol_kind(kind: Option<String>) -> Option<SymbolKind> {
    kind.as_deref().and_then(|s| {
        let capitalized = {
            let mut chars = s.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first
                    .to_uppercase()
                    .chain(chars.flat_map(|c| c.to_lowercase()))
                    .collect(),
            }
        };

        capitalized
            .parse::<SymbolKind>()
            .ok()
            .or_else(|| Some(SymbolKind::Unknown(s.to_string())))
    })
}

/// Handle initialize request
fn handle_initialize(_params: Option<Value>) -> Result<Value> {
    Ok(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "reflex",
            "version": env!("CARGO_PKG_VERSION")
        }
    }))
}

/// Handle tools/list request
fn handle_list_tools(_params: Option<Value>) -> Result<Value> {
    Ok(json!({
        "tools": [
            {
                "name": "list_locations",
                "description": "Fast location discovery with minimal token usage.\n\n**Purpose:** Find where a pattern occurs (file + line) without loading previews or detailed context.\n\n**Returns:** Array of {path, line} objects - one per match location.\n\n**Use this when:**\n- Starting exploration (\"where is X used?\")\n- Counting affected locations\n- Building a list for targeted Read operations\n- You need locations only, not code content\n\n**Workflow:**\n1. Use list_locations to discover (cheap, returns locations only)\n2. Use Read tool or search_code on specific files if you need content (targeted)\n\n**Supports:** lang, file, glob, exclude filters\n**No limit:** Returns ALL matching locations\n\n**Error Handling:** If you receive an error message containing \"Index not found\" or \"stale\", immediately call the index_project tool, wait for it to complete, then retry this operation.\n\n**Example:** Pattern \"CourtCase\" → [{\"path\": \"app/Models/CourtCase.php\", \"line\": 15}, {\"path\": \"app/Http/Controllers/CourtController.php\", \"line\": 42}]",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern (text to find)"
                        },
                        "lang": {
                            "type": "string",
                            "description": "Filter by language (php, rust, typescript, python, etc.)"
                        },
                        "file": {
                            "type": "string",
                            "description": "Filter by file path substring (e.g., 'Controllers')"
                        },
                        "glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Include files matching patterns (e.g., ['app/**/*.php'])"
                        },
                        "exclude": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude files matching patterns (e.g., ['vendor/**', 'tests/**'])"
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force execution of potentially expensive queries (bypasses broad query detection)"
                        },
                        "dependencies": {
                            "type": "boolean",
                            "description": "Include dependency information (imports) in results. Only extracts static imports."
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "count_occurrences",
                "description": "Quick statistics - count how many times a pattern occurs.\n\n**Purpose:** Get total occurrence count and file count without loading any content.\n\n**Use this when:**\n- You need quick stats (\"how many times is X used?\")\n- Checking impact before refactoring\n- Validating search scope\n\n**Returns:** {total: count, files: count, pattern: string}\n\n**Supports:** All filters (lang, file, glob, exclude, symbols)\n\n**Error Handling:** If you receive an error message containing \"Index not found\" or \"stale\", immediately call the index_project tool, wait for it to complete, then retry this operation.\n\n**Example output:** {\"total\": 87, \"files\": 12, \"pattern\": \"CourtCase\"}",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern (text to find)"
                        },
                        "lang": {
                            "type": "string",
                            "description": "Filter by language"
                        },
                        "symbols": {
                            "type": "boolean",
                            "description": "Count symbol definitions only (not usages)"
                        },
                        "kind": {
                            "type": "string",
                            "description": "Filter by symbol kind (function, class, etc.)"
                        },
                        "file": {
                            "type": "string",
                            "description": "Filter by file path substring"
                        },
                        "glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Include files matching patterns"
                        },
                        "exclude": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude files matching patterns"
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force execution of potentially expensive queries (bypasses broad query detection)"
                        },
                        "dependencies": {
                            "type": "boolean",
                            "description": "Include dependency information (imports) in results. Only extracts static imports."
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "search_code",
                "description": "Full-text or symbol-only code search with detailed results.\n\n**When to use search_regex instead:**\n- Patterns with special characters: -> :: () [] {} . * + ? \\\\ | ^ $\n- Complex pattern matching: wildcards, alternation, anchors\n- Examples: '->with(', '::new', 'function*', '[derive]', 'fn (get|set)_.*'\n\n**Search modes:**\n- Full-text (default): Finds ALL occurrences - definitions + usages\n- Symbol-only (symbols=true): Finds ONLY definitions where symbols are declared\n\n**Use this for:**\n- Simple text patterns (alphanumeric, underscores, hyphens)\n- Detailed analysis with line numbers and code previews\n- Symbol definition searches\n\n**Pagination:** Check response.pagination.has_more. If true, use offset parameter to fetch next page.\n\n**Error Handling:** If you receive an error message containing \"Index not found\" or \"stale\", immediately call the index_project tool, wait for it to complete, then retry this operation.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern (text to find)"
                        },
                        "lang": {
                            "type": "string",
                            "description": "Filter by language (rust, typescript, python, etc.)"
                        },
                        "kind": {
                            "type": "string",
                            "description": "Filter by symbol kind (function, class, struct, etc.)"
                        },
                        "symbols": {
                            "type": "boolean",
                            "description": "Symbol-only search (definitions, not usage)"
                        },
                        "exact": {
                            "type": "boolean",
                            "description": "Exact match (no substring matching)"
                        },
                        "file": {
                            "type": "string",
                            "description": "Filter by file path (substring)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results per page (default: 100). IMPORTANT: If response.pagination.has_more is true, you MUST fetch more pages using offset parameter."
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N results). ALWAYS paginate when has_more=true. Example: First call offset=0, second call offset=100, third offset=200, etc."
                        },
                        "expand": {
                            "type": "boolean",
                            "description": "Show full symbol body (not just signature)"
                        },
                        "glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Include files matching glob patterns (e.g., 'src/**/*.rs')"
                        },
                        "exclude": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude files matching glob patterns (e.g., 'target/**')"
                        },
                        "paths": {
                            "type": "boolean",
                            "description": "Return only unique file paths (not full results)"
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force execution of potentially expensive queries (bypasses broad query detection)"
                        },
                        "dependencies": {
                            "type": "boolean",
                            "description": "Include dependency information (imports) in results. **IMPORTANT:** Only extracts static imports (string literals). Dynamic imports (variables, template literals, expressions) are automatically filtered. See CLAUDE.md for details."
                        },
                        "preview_length": {
                            "type": "integer",
                            "description": "Maximum characters per preview line (default: 180). Use a smaller value (e.g. 60) for wide-result scans where short previews are sufficient."
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "search_regex",
                "description": "Regex-based code search for complex pattern matching (e.g., 'fn (get|set)_\\\\w+').\n\n**Use for:**\n- Patterns with special characters: -> :: () [] {} . * + ? \\\\ | ^ $\n- Pattern matching: wildcards (.*), alternation (a|b), anchors (^$)\n- Complex searches: case-insensitive variants, word boundaries\n\n**Common examples:**\n- Method calls: '->with\\\\(', '->map\\\\(', '::new\\\\('\n- Operators: '->', '::', '||', '&&'\n- Functions: 'fn (get|set)_\\\\\\\\w+' (getter/setter functions)\n- Attributes: '\\\\\\\\[(derive|test)\\\\\\\\]' (Rust attributes)\n\n**Escaping rules:**\n- Must escape: ( ) [ ] { } . * + ? \\\\ | ^ $\n- No escaping needed: -> :: - _ / = < >\n- Use double backslash in JSON: \\\\\\\\( \\\\\\\\) \\\\\\\\[ \\\\\\\\]\n\n**Error Handling:** If you receive an error message containing \"Index not found\" or \"stale\", immediately call the index_project tool, wait for it to complete, then retry this operation.\n\n**Don't use for:**\n- Simple text searches (use search_code instead - faster)\n- Symbol definitions (use search_code with symbols=true instead)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern"
                        },
                        "lang": {
                            "type": "string",
                            "description": "Filter by language"
                        },
                        "file": {
                            "type": "string",
                            "description": "Filter by file path"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (use with offset for pagination)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N results after sorting)"
                        },
                        "glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Include files matching glob patterns"
                        },
                        "exclude": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude files matching glob patterns"
                        },
                        "paths": {
                            "type": "boolean",
                            "description": "Return only unique file paths"
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force execution of potentially expensive queries (bypasses broad query detection)"
                        },
                        "dependencies": {
                            "type": "boolean",
                            "description": "Include dependency information (imports) in results. Only extracts static imports."
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "search_ast",
                "description": "⚠️ ADVANCED USERS ONLY - DO NOT USE UNLESS ABSOLUTELY NECESSARY ⚠️\n\nStructure-aware code search using Tree-sitter AST patterns (S-expressions).\n\n**PERFORMANCE WARNING:** AST queries bypass trigram optimization and scan the ENTIRE codebase (500ms-10s+).\n\n**WHEN TO USE (RARE):**\n- You need to match code structure, not just text (e.g., \"all async functions with try/catch blocks\")\n- --symbols search is insufficient (e.g., need to match specific AST node types)\n- You have a very specific structural pattern that cannot be expressed as text\n\n**IN 95% OF CASES, USE search_code with symbols=true INSTEAD** (10-100x faster).\n\n**REQUIRED:** You MUST use glob patterns to limit scope (e.g., glob=['src/**/*.rs']) to avoid scanning thousands of files.\n\n**Token efficiency:** Previews are auto-truncated to ~100 chars. Use limit parameter to control result count.\n\n**Error Handling:** If you receive an error message containing \"Index not found\" or \"stale\", immediately call the index_project tool, wait for it to complete, then retry this operation.\n\n**Example AST patterns:**\n- Rust: '(function_item) @fn' (all functions)\n- Python: '(function_definition) @fn' (all functions)\n- TypeScript: '(class_declaration) @class' (all classes)\n\nRefer to Tree-sitter documentation for each language's grammar.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "AST pattern (Tree-sitter S-expression, e.g., '(function_item) @fn')"
                        },
                        "lang": {
                            "type": "string",
                            "description": "Language (REQUIRED: rust, typescript, javascript, python, go, java, c, cpp, csharp, php, ruby, kotlin, zig)"
                        },
                        "glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Include files matching glob patterns (STRONGLY RECOMMENDED to limit scope, e.g., ['src/**/*.rs'])"
                        },
                        "exclude": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude files matching glob patterns (e.g., ['target/**', 'node_modules/**'])"
                        },
                        "file": {
                            "type": "string",
                            "description": "Filter by file path (substring)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (use with offset for pagination)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N results after sorting)"
                        },
                        "paths": {
                            "type": "boolean",
                            "description": "Return only unique file paths"
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force execution of potentially expensive queries (bypasses broad query detection)"
                        },
                        "dependencies": {
                            "type": "boolean",
                            "description": "Include dependency information (imports) in results. Only extracts static imports."
                        }
                    },
                    "required": ["pattern", "lang"]
                }
            },
            {
                "name": "index_project",
                "description": "Rebuild or update the code search index. Run this when:\n\n- After code changes (user edits, git operations, file creation/deletion)\n- Search results seem stale or missing new files\n- Empty/error results (may indicate missing/corrupt index)\n\n**Modes:**\n- Incremental (default): Only re-indexes changed files (fast)\n- Full rebuild (force=true): Re-indexes everything (use if index seems corrupted)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "force": {
                            "type": "boolean",
                            "description": "Force full rebuild (ignore incremental)"
                        },
                        "languages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Languages to include (empty = all)"
                        }
                    }
                }
            },
            {
                "name": "get_dependencies",
                "description": "Get all dependencies (imports) of a specific file.\n\n**Purpose:** Analyze what modules/files a given file imports.\n\n**Returns:** Array of dependency objects with import path, line number, type (internal/external/stdlib), and optional symbols.\n\n**Use this when:**\n- Understanding file dependencies\n- Analyzing import structure\n- Finding what a file depends on\n\n**IMPORTANT:** Only extracts **static imports** (string literals). Dynamic imports (variables, template literals, expressions) are automatically filtered by tree-sitter query design. See CLAUDE.md section \"Dependency/Import Extraction\" for details.\n\n**Note:** Path matching is fuzzy - supports exact paths, fragments, or just filenames.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path (supports fuzzy matching: 'Controllers/FooController.php' or just 'FooController.php')"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "get_dependents",
                "description": "Get all files that depend on (import) a specific file.\n\n**Purpose:** Find what other files import this file (reverse dependency lookup).\n\n**Returns:** Array of file paths that import the specified file.\n\n**Use this when:**\n- Understanding impact of changes\n- Finding usages of a module\n- Analyzing file importance\n\n**IMPORTANT:** Only considers **static imports** (string literals). Dynamic imports are filtered. See CLAUDE.md section \"Dependency/Import Extraction\" for details.\n\n**Note:** Path matching is fuzzy - supports exact paths, fragments, or just filenames.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path (supports fuzzy matching: 'models/User.php' or just 'User.php')"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "get_transitive_deps",
                "description": "Get transitive dependencies of a file up to a specified depth.\n\n**Purpose:** Find not just direct dependencies, but dependencies of dependencies (the full dependency tree).\n\n**Returns:** Object mapping file IDs to their depth in the dependency tree.\n\n**Use this when:**\n- Understanding full dependency chain\n- Analyzing deep coupling\n- Planning refactoring impact\n\n**IMPORTANT:** Only follows **static imports** (string literals). Dynamic imports are filtered. See CLAUDE.md section \"Dependency/Import Extraction\" for details.\n\n**Example:** depth=2 finds: file → deps → deps of deps",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path (supports fuzzy matching)"
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Maximum depth to traverse (default: 3, max recommended: 5)"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "find_hotspots",
                "description": "Find the most-imported files in the codebase (dependency hotspots).\n\n**Purpose:** Identify files that many other files depend on.\n\n**Pagination:** Default limit of 200 results per page. Check response.pagination.has_more to fetch more pages.\n\n**Sorting:** Default order is descending (most imports first). Use sort parameter to change.\n\n**Returns:** Object with pagination metadata and array of {path, import_count} objects sorted by import count.\n\n**Use this when:**\n- Finding critical files\n- Identifying potential bottlenecks\n- Understanding architecture\n- Planning refactoring priorities\n\n**IMPORTANT:** Only counts **static imports** (string literals). Dynamic imports are filtered. See CLAUDE.md section \"Dependency/Import Extraction\" for details.\n\n**Example output:** {\"pagination\": {...}, \"results\": [{\"path\": \"src/models.rs\", \"import_count\": 27}]}",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of hotspots per page (default: 200)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N results). Use with limit for pagination."
                        },
                        "min_dependents": {
                            "type": "integer",
                            "description": "Minimum number of dependents to include (default: 2)"
                        },
                        "sort": {
                            "type": "string",
                            "description": "Sort order: 'asc' (least imports first) or 'desc' (most imports first, default)"
                        }
                    }
                }
            },
            {
                "name": "find_circular",
                "description": "Detect circular dependencies in the codebase.\n\n**Purpose:** Find dependency cycles (A → B → C → A).\n\n**Pagination:** Default limit of 200 results per page. Check response.pagination.has_more to fetch more pages.\n\n**Sorting:** Default order is descending (longest cycles first). Use sort parameter to change.\n\n**Returns:** Object with pagination metadata and array of cycles, where each cycle is an array of file paths forming the circular path.\n\n**Use this when:**\n- Debugging circular dependency issues\n- Improving code architecture\n- Validating refactoring\n\n**IMPORTANT:** Only detects cycles in **static imports** (string literals). Dynamic imports are filtered. See CLAUDE.md section \"Dependency/Import Extraction\" for details.\n\n**Note:** Circular dependencies can cause compilation issues and indicate architectural problems.\n\n**Example output:** {\"pagination\": {...}, \"results\": [{\"paths\": [\"a.rs\", \"b.rs\", \"a.rs\"]}]}",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of cycles per page (default: 200)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N cycles). Use with limit for pagination."
                        },
                        "sort": {
                            "type": "string",
                            "description": "Sort order: 'asc' (shortest cycles first) or 'desc' (longest cycles first, default)"
                        }
                    }
                }
            },
            {
                "name": "find_unused",
                "description": "Find unused files that no other files import.\n\n**Purpose:** Identify orphaned files that could be safely removed.\n\n**Pagination:** Default limit of 200 results per page. Check response.pagination.has_more to fetch more pages.\n\n**Returns:** Object with pagination metadata and flat array of file path strings (no wrapping objects).\n\n**Use this when:**\n- Cleaning up dead code\n- Reducing codebase size\n- Identifying test-only or entry-point files\n\n**Note:** Entry points (main.rs, index.ts) will appear as unused but should not be deleted.\n\n**Example output:** {\"pagination\": {...}, \"results\": [\"src/unused.rs\", \"tests/old.rs\"]}",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of unused files per page (default: 200)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N files). Use with limit for pagination."
                        }
                    }
                }
            },
            {
                "name": "find_islands",
                "description": "Find disconnected components (islands) in the dependency graph.\n\n**Purpose:** Identify groups of files that are isolated from the rest of the codebase (no dependencies between groups).\n\n**Pagination:** Default limit of 200 results per page. Check response.pagination.has_more to fetch more pages.\n\n**Sorting:** Default order is descending (largest islands first). Use sort parameter to change.\n\n**Returns:** Object with pagination metadata and array of islands, where each island contains multiple file paths that depend on each other.\n\n**Use this when:**\n- Identifying isolated subsystems\n- Understanding codebase modularity\n- Finding potential code splitting opportunities\n- Detecting disconnected feature modules\n\n**IMPORTANT:** Only considers **static imports** (string literals). Dynamic imports are filtered. See CLAUDE.md section \"Dependency/Import Extraction\" for details.\n\n**Size filtering:** Use min_island_size and max_island_size to filter by component size. Default: 2-500 files (or 50% of total files).\n\n**Example output:** {\"pagination\": {...}, \"results\": [{\"island_id\": 1, \"size\": 5, \"paths\": [\"a.rs\", \"b.rs\", \"c.rs\", \"d.rs\", \"e.rs\"]}]}",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of islands per page (default: 200)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N islands). Use with limit for pagination."
                        },
                        "min_island_size": {
                            "type": "integer",
                            "description": "Minimum files in an island to include (default: 2)"
                        },
                        "max_island_size": {
                            "type": "integer",
                            "description": "Maximum files in an island to include (default: 500 or 50% of total files)"
                        },
                        "sort": {
                            "type": "string",
                            "description": "Sort order: 'asc' (smallest islands first) or 'desc' (largest islands first, default)"
                        }
                    }
                }
            },
            {
                "name": "analyze_summary",
                "description": "Get a summary of all dependency analyses.\n\n**Purpose:** Quick overview of codebase dependency health.\n\n**Returns:** Object with counts: {circular_dependencies, hotspots, unused_files, islands, min_dependents}\n\n**Use this when:**\n- Getting a quick health check of the codebase\n- Understanding overall dependency structure\n- Deciding which specific analysis to run next\n\n**IMPORTANT:** Only considers **static imports** (string literals). Dynamic imports are filtered. See CLAUDE.md section \"Dependency/Import Extraction\" for details.\n\n**Example output:** {\"circular_dependencies\": 17, \"hotspots\": 10, \"unused_files\": 82, \"islands\": 81, \"min_dependents\": 2}",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "min_dependents": {
                            "type": "integer",
                            "description": "Minimum number of dependents for hotspots (default: 2)"
                        }
                    }
                }
            },
            {
                "name": "find_references",
                "description": "Find a symbol's definition AND all usage sites in a single call.\n\n**Purpose:** Eliminates the two-step pattern of search_code(symbols=true) + search_code(). Returns both the definition and all usages atomically.\n\n**Returns:** {definition, references, total_references, pagination, status}\n\n**Use this when:**\n- \"Find all callers of X\" — the most common agent refactoring pattern\n- Code review: understand impact before changing a function or class\n- Rename planning: see every site that needs updating\n- Dead code detection: confirm nothing calls a function before removing it\n\n**definition:** First matching symbol definition {path, line, kind, symbol, span, preview}, or null if no symbol definition exists for the pattern.\n\n**references:** Flat array of {path, line, preview} — all textual occurrences including the definition site itself.\n\n**Pagination applies to references only.** Use limit and offset. Check pagination.has_more for more pages.\n\n**Error Handling:** If you receive an error message containing \"Index not found\" or \"stale\", immediately call the index_project tool, wait for it to complete, then retry this operation.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Symbol name or text pattern to find references for (e.g., 'CacheManager', 'extract_symbols')"
                        },
                        "kind": {
                            "type": "string",
                            "description": "Filter definition lookup by symbol kind (function, class, struct, trait, etc.)"
                        },
                        "lang": {
                            "type": "string",
                            "description": "Filter by language (rust, typescript, python, go, etc.)"
                        },
                        "glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Include files matching glob patterns (e.g., ['src/**/*.rs'])"
                        },
                        "exclude": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude files matching glob patterns (e.g., ['target/**', 'tests/**'])"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max references per page (default: 100). Pagination applies to references only."
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset for references (skip first N). Use with limit."
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force execution of potentially expensive queries (bypasses broad query detection)"
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "gather_context",
                "description": "Collects comprehensive codebase information.\n\n**Parameters:**\n- `structure` (bool): Show directory tree\n- `file_types` (bool): Show file type distribution\n- `project_type` (bool): Detect project type (CLI/library/webapp)\n- `framework` (bool): Detect frameworks (React, Django, etc.)\n- `entry_points` (bool): Find main/index files\n- `test_layout` (bool): Show test organization\n- `config_files` (bool): List configuration files\n- `depth` (int): Tree depth for structure (default: 2)\n- `path` (string, optional): Focus on specific directory\n\n**When to use:**\n- Understanding project structure and organization\n- Finding which frameworks/languages are used\n- Locating entry points and test layouts\n- Getting file statistics and distribution\n\n**When NOT to use:**\n- Finding conceptual/architectural information (use search_documentation)\n- Understanding high-level how things work (use search_documentation)\n\n**Note:** By default (no parameters), all context types are gathered.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "structure": {
                            "type": "boolean",
                            "description": "Show directory structure"
                        },
                        "file_types": {
                            "type": "boolean",
                            "description": "Show file type distribution"
                        },
                        "project_type": {
                            "type": "boolean",
                            "description": "Detect project type (CLI/library/webapp/monorepo)"
                        },
                        "framework": {
                            "type": "boolean",
                            "description": "Detect frameworks and conventions"
                        },
                        "entry_points": {
                            "type": "boolean",
                            "description": "Show entry point files"
                        },
                        "test_layout": {
                            "type": "boolean",
                            "description": "Show test organization pattern"
                        },
                        "config_files": {
                            "type": "boolean",
                            "description": "List important configuration files"
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Tree depth for structure (default: 2)"
                        },
                        "path": {
                            "type": "string",
                            "description": "Focus on specific directory path"
                        }
                    }
                }
            },
            {
                "name": "check_index_status",
                "description": "Check whether the Reflex search index is fresh, stale, or missing — without running any search.\n\n**CALL THIS FIRST** at the start of every session and before any significant search task. If the index is stale or missing, call index_project before searching.\n\n**Returns:**\n- `status`: `\"fresh\"` | `\"stale\"` | `\"missing\"`\n- `reason`: why the index is stale (branch not indexed, commit changed, files modified)\n- `action_required`: command to fix the issue (always `rfx index` when stale)\n- `files_modified`: number of recently modified files detected (only present for mtime-based staleness)\n\n**When to call:**\n- At the start of every agent session\n- Before any bulk search or refactoring task\n- After a git operation (checkout, merge, rebase, pull)\n\n**Example fresh response:** `{\"status\": \"fresh\"}`\n**Example stale response:** `{\"status\": \"stale\", \"reason\": \"Commit changed from abc1234 to def5678\", \"action_required\": \"rfx index\"}`",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    }))
}

/// Handle tools/call request
fn handle_call_tool(params: Option<Value>) -> Result<Value> {
    let params = params.ok_or_else(|| anyhow::anyhow!("Missing params for tools/call"))?;

    let name = params["name"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing tool name"))?;

    let arguments = params["arguments"].clone();

    match name {
        "list_locations" => {
            // Location discovery tool (minimal token usage)
            let pattern = arguments["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing pattern"))?
                .to_string();

            let lang = arguments["lang"].as_str().map(|s| s.to_string());
            let file = arguments["file"].as_str().map(|s| s.to_string());
            let glob_patterns = arguments["glob"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();
            let exclude_patterns = arguments["exclude"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();
            let force = arguments["force"].as_bool().unwrap_or(false);
            let dependencies = arguments["dependencies"].as_bool().unwrap_or(false);

            let language = parse_language(lang);

            let filter = QueryFilter {
                language,
                kind: None,
                use_ast: false,
                use_regex: false,
                limit: None,  // No limit for paths-only mode
                symbols_mode: false,
                expand: false,
                file_pattern: file,
                exact: false,
                use_contains: false,
                timeout_secs: 30,
                glob_patterns,
                exclude_patterns,
                paths_only: true,  // KEY: Enable paths-only mode
                offset: None,
                force,
                suppress_output: true,  // MCP always returns JSON
                include_dependencies: dependencies,
                ..Default::default()
            };

            let cache = CacheManager::new(".");
            let engine = QueryEngine::new(cache);
            let response = engine.search_with_metadata(&pattern, filter)?;

            // Extract locations (path + line) for each match
            let locations: Vec<serde_json::Value> = response.results.iter()
                .flat_map(|file_group| {
                    file_group.matches.iter().map(move |m| {
                        json!({
                            "path": file_group.path.clone(),
                            "line": m.span.start_line
                        })
                    })
                })
                .collect();

            // Return compact response (just locations + count)
            let compact_response = json!({
                "status": response.status,
                "total_locations": locations.len(),
                "locations": locations
            });

            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&compact_response)?
                }]
            }))
        }
        "count_occurrences" => {
            // Quick stats tool (minimal token usage)
            let pattern = arguments["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing pattern"))?
                .to_string();

            let lang = arguments["lang"].as_str().map(|s| s.to_string());
            let kind = arguments["kind"].as_str().map(|s| s.to_string());
            let symbols = arguments["symbols"].as_bool();
            let file = arguments["file"].as_str().map(|s| s.to_string());
            let glob_patterns = arguments["glob"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();
            let exclude_patterns = arguments["exclude"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();
            let force = arguments["force"].as_bool().unwrap_or(false);
            let dependencies = arguments["dependencies"].as_bool().unwrap_or(false);

            let language = parse_language(lang);
            let parsed_kind = parse_symbol_kind(kind);
            let symbols_mode = symbols.unwrap_or(false) || parsed_kind.is_some();

            let filter = QueryFilter {
                language,
                kind: parsed_kind,
                use_ast: false,
                use_regex: false,
                limit: None,  // No limit for counting
                symbols_mode,
                expand: false,
                file_pattern: file,
                exact: false,
                use_contains: false,
                timeout_secs: 30,
                glob_patterns,
                exclude_patterns,
                paths_only: false,  // Need to count all occurrences
                offset: None,
                force,
                suppress_output: true,  // MCP always returns JSON
                include_dependencies: dependencies,
                ..Default::default()
            };

            let cache = CacheManager::new(".");
            let engine = QueryEngine::new(cache);
            let response = engine.search_with_metadata(&pattern, filter)?;

            // Count unique files
            use std::collections::HashSet;
            let unique_files: HashSet<String> = response.results.iter().map(|fg| fg.path.clone()).collect();

            // Return minimal stats
            let stats = json!({
                "status": response.status,
                "pattern": pattern,
                "total": response.pagination.total,
                "files": unique_files.len()
            });

            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&stats)?
                }]
            }))
        }
        "search_code" => {
            let pattern = arguments["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing pattern"))?
                .to_string();

            let lang = arguments["lang"].as_str().map(|s| s.to_string());
            let kind = arguments["kind"].as_str().map(|s| s.to_string());
            let symbols = arguments["symbols"].as_bool();
            let exact = arguments["exact"].as_bool();
            let file = arguments["file"].as_str().map(|s| s.to_string());
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let expand = arguments["expand"].as_bool();
            let glob_patterns: Vec<String> = arguments["glob"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();
            let exclude_patterns = arguments["exclude"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();
            let paths_only = arguments["paths"].as_bool().unwrap_or(false);
            let force = arguments["force"].as_bool().unwrap_or(false);
            let dependencies = arguments["dependencies"].as_bool().unwrap_or(false);
            let preview_length = arguments["preview_length"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(DEFAULT_MCP_PREVIEW_LENGTH);

            let language = parse_language(lang);
            let parsed_kind = parse_symbol_kind(kind);
            let symbols_mode = symbols.unwrap_or(false) || parsed_kind.is_some();

            let offset = arguments["offset"].as_u64().map(|n| n as usize);

            // Smart limit handling:
            // 1. If --paths is set and user didn't specify limit: no limit (None)
            // 2. If user specified limit: use that value
            // 3. Otherwise: use default limit of 100
            let final_limit = if paths_only && limit.is_none() {
                None  // --paths without explicit limit means no limit
            } else if let Some(user_limit) = limit {
                Some(user_limit)  // Use user-specified limit
            } else {
                Some(100)  // Default: limit to 100 results for token efficiency
            };

            let filter = QueryFilter {
                language,
                kind: parsed_kind,
                use_ast: false,
                use_regex: false,
                limit: final_limit,
                symbols_mode,
                expand: expand.unwrap_or(false),
                file_pattern: file,
                exact: exact.unwrap_or(false),
                use_contains: false, // Default to word-boundary matching for MCP
                timeout_secs: 30, // Default 30 second timeout for MCP queries
                glob_patterns: glob_patterns.clone(),
                exclude_patterns,
                paths_only,
                offset,
                force,
                suppress_output: true,  // MCP always returns JSON
                include_dependencies: dependencies,
                ..Default::default()
            };

            let cache = CacheManager::new(".");
            let engine = QueryEngine::new(cache);
            let mut response = engine.search_with_metadata(&pattern, filter)?;

            // Apply preview truncation for token efficiency
            for file_group in response.results.iter_mut() {
                for m in file_group.matches.iter_mut() {
                    m.preview = crate::cli::truncate_preview(&m.preview, preview_length);
                }
            }

            // Calculate result count for AI instruction
            let result_count: usize = response.results.iter().map(|fg| fg.matches.len()).sum();

            // Generate AI instruction (MCP always uses AI mode)
            response.ai_instruction = crate::query::generate_ai_instruction(
                result_count,
                response.pagination.total,
                response.pagination.has_more,
                symbols_mode,
                paths_only,
                false,  // use_ast
                false,  // use_regex
                language.is_some(),
                !glob_patterns.is_empty(),
                exact.unwrap_or(false),
            );

            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&response)?
                }]
            }))
        }
        "search_regex" => {
            let pattern = arguments["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing pattern"))?
                .to_string();

            let lang = arguments["lang"].as_str().map(|s| s.to_string());
            let file = arguments["file"].as_str().map(|s| s.to_string());
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let glob_patterns: Vec<String> = arguments["glob"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();
            let exclude_patterns = arguments["exclude"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();
            let paths_only = arguments["paths"].as_bool().unwrap_or(false);
            let force = arguments["force"].as_bool().unwrap_or(false);
            let dependencies = arguments["dependencies"].as_bool().unwrap_or(false);

            let language = parse_language(lang);
            let offset = arguments["offset"].as_u64().map(|n| n as usize);

            // Smart limit handling (same as search_code)
            let final_limit = if paths_only && limit.is_none() {
                None  // --paths without explicit limit means no limit
            } else if let Some(user_limit) = limit {
                Some(user_limit)  // Use user-specified limit
            } else {
                Some(100)  // Default: limit to 100 results for token efficiency
            };

            let filter = QueryFilter {
                language,
                kind: None,
                use_ast: false,
                use_regex: true,
                limit: final_limit,
                symbols_mode: false,
                expand: false,
                file_pattern: file,
                exact: false,
                use_contains: false, // Regex mode uses substring matching via use_regex flag
                timeout_secs: 30, // Default 30 second timeout for MCP queries
                glob_patterns: glob_patterns.clone(),
                exclude_patterns,
                paths_only,
                offset,
                force,
                suppress_output: true,  // MCP always returns JSON
                include_dependencies: dependencies,
                ..Default::default()
            };

            let cache = CacheManager::new(".");
            let engine = QueryEngine::new(cache);
            let mut response = engine.search_with_metadata(&pattern, filter)?;

            // Apply preview truncation for token efficiency
            for file_group in response.results.iter_mut() {
                for m in file_group.matches.iter_mut() {
                    m.preview = crate::cli::truncate_preview(&m.preview, DEFAULT_MCP_PREVIEW_LENGTH);
                }
            }

            // Calculate result count for AI instruction
            let result_count: usize = response.results.iter().map(|fg| fg.matches.len()).sum();

            // Generate AI instruction (MCP always uses AI mode)
            response.ai_instruction = crate::query::generate_ai_instruction(
                result_count,
                response.pagination.total,
                response.pagination.has_more,
                false,  // symbols_mode
                paths_only,
                false,  // use_ast
                true,   // use_regex
                language.is_some(),
                !glob_patterns.is_empty(),
                false,  // exact
            );

            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&response)?
                }]
            }))
        }
        "search_ast" => {
            // AST pattern (Tree-sitter S-expression)
            let ast_pattern = arguments["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing pattern (AST S-expression)"))?
                .to_string();

            let lang_str = arguments["lang"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing lang (required for AST queries)"))?
                .to_string();

            let file = arguments["file"].as_str().map(|s| s.to_string());
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let glob_patterns: Vec<String> = arguments["glob"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();
            let exclude_patterns: Vec<String> = arguments["exclude"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();
            let paths_only = arguments["paths"].as_bool().unwrap_or(false);
            let force = arguments["force"].as_bool().unwrap_or(false);
            let dependencies = arguments["dependencies"].as_bool().unwrap_or(false);

            let language = parse_language(Some(lang_str))
                .ok_or_else(|| anyhow::anyhow!("Invalid or unsupported language for AST queries"))?;

            // Warn if glob patterns are not provided (performance issue)
            if glob_patterns.is_empty() && exclude_patterns.is_empty() {
                log::warn!("⚠️  AST query without glob patterns will scan the ENTIRE codebase. This may take 2-10+ seconds.");
                log::warn!("    Strongly recommend using glob patterns, e.g., glob=['src/**/*.rs']");
            }

            let offset = arguments["offset"].as_u64().map(|n| n as usize);

            // Smart limit handling (same as search_code)
            let final_limit = if paths_only && limit.is_none() {
                None  // --paths without explicit limit means no limit
            } else if let Some(user_limit) = limit {
                Some(user_limit)  // Use user-specified limit
            } else {
                Some(100)  // Default: limit to 100 results for token efficiency
            };

            let filter = QueryFilter {
                language: Some(language),
                kind: None,
                use_ast: true,
                use_regex: false,
                limit: final_limit,
                symbols_mode: false,
                expand: false,
                file_pattern: file,
                exact: false,
                use_contains: false,
                timeout_secs: 60, // Longer timeout for AST queries (they're slow)
                glob_patterns,
                exclude_patterns,
                paths_only,
                offset,
                force,
                suppress_output: true,  // MCP always returns JSON
                include_dependencies: dependencies,
                ..Default::default()
            };

            let cache = CacheManager::new(".");
            let engine = QueryEngine::new(cache);

            // Use the new search_ast_all_files method (no trigram filtering)
            let mut results = engine.search_ast_all_files(&ast_pattern, filter)?;

            // Apply preview truncation for token efficiency
            for result in &mut results {
                result.preview = crate::cli::truncate_preview(&result.preview, DEFAULT_MCP_PREVIEW_LENGTH);
            }

            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&results)?
                }]
            }))
        }
        "index_project" => {
            let force = arguments["force"].as_bool();
            let languages = arguments["languages"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect::<Vec<_>>()
                });

            let cache = CacheManager::new(".");

            if force.unwrap_or(false) {
                log::info!("Force rebuild requested, clearing existing cache");
                cache.clear()?;
            }

            let lang_filters: Vec<Language> = languages
                .unwrap_or_default()
                .iter()
                .filter_map(|s| parse_language(Some(s.clone())))
                .collect();

            let config = IndexConfig {
                languages: lang_filters,
                ..Default::default()
            };

            let indexer = Indexer::new(cache, config);
            let path = PathBuf::from(".");
            let stats = indexer.index(&path, false)?;

            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&stats)?
                }]
            }))
        }
        "get_dependencies" => {
            let path = arguments["path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing path"))?
                .to_string();

            let cache = CacheManager::new(".");
            let deps_index = DependencyIndex::new(cache);

            // Fuzzy path matching
            let file_id = deps_index.get_file_id_by_path(&path)?
                .ok_or_else(|| anyhow::anyhow!("File '{}' not found in index", path))?;

            let dependencies = deps_index.get_dependencies_info(file_id)?;

            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&dependencies)?
                }]
            }))
        }
        "get_dependents" => {
            let path = arguments["path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing path"))?
                .to_string();

            let cache = CacheManager::new(".");
            let deps_index = DependencyIndex::new(cache);

            // Fuzzy path matching
            let file_id = deps_index.get_file_id_by_path(&path)?
                .ok_or_else(|| anyhow::anyhow!("File '{}' not found in index", path))?;

            let dependents = deps_index.get_dependents(file_id)?;
            let paths = deps_index.get_file_paths(&dependents)?;

            // Convert to array of paths
            let path_list: Vec<String> = dependents.iter()
                .filter_map(|id| paths.get(id).cloned())
                .collect();

            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&path_list)?
                }]
            }))
        }
        "get_transitive_deps" => {
            let path = arguments["path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing path"))?
                .to_string();

            let depth = arguments["depth"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(3);  // Default depth of 3

            let cache = CacheManager::new(".");
            let deps_index = DependencyIndex::new(cache);

            // Fuzzy path matching
            let file_id = deps_index.get_file_id_by_path(&path)?
                .ok_or_else(|| anyhow::anyhow!("File '{}' not found in index", path))?;

            let transitive = deps_index.get_transitive_deps(file_id, depth)?;

            // Get paths for all file IDs
            let file_ids: Vec<i64> = transitive.keys().copied().collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            // Build result with path → depth mapping
            let result: Vec<serde_json::Value> = transitive.iter()
                .filter_map(|(id, depth)| {
                    paths.get(id).map(|path| json!({
                        "path": path,
                        "depth": depth
                    }))
                })
                .collect();

            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&result)?
                }]
            }))
        }
        "find_hotspots" => {
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let offset = arguments["offset"].as_u64().map(|n| n as usize);
            let min_dependents = arguments["min_dependents"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(2);
            let sort = arguments["sort"].as_str().map(|s| s.to_string());

            let cache = CacheManager::new(".");
            let deps_index = DependencyIndex::new(cache);

            // Get all hotspots first (without limit) to track total count
            let mut all_hotspots = deps_index.find_hotspots(None, min_dependents)?;

            // Apply sorting (default: descending - most imports first)
            let sort_order = sort.as_deref().unwrap_or("desc");
            match sort_order {
                "asc" => {
                    // Ascending: least imports first
                    all_hotspots.sort_by(|a, b| a.1.cmp(&b.1));
                }
                "desc" => {
                    // Descending: most imports first (default)
                    all_hotspots.sort_by(|a, b| b.1.cmp(&a.1));
                }
                _ => {
                    return Err(anyhow::anyhow!("Invalid sort order '{}'. Supported: asc, desc", sort_order));
                }
            }

            let total_count = all_hotspots.len();

            // Apply offset pagination
            let offset_val = offset.unwrap_or(0);
            let mut hotspots: Vec<_> = all_hotspots.into_iter().skip(offset_val).collect();

            // Apply limit (default 200)
            let limit_val = limit.unwrap_or(200);
            hotspots.truncate(limit_val);

            let count = hotspots.len();
            let has_more = offset_val + count < total_count;

            // Get paths for all file IDs
            let file_ids: Vec<i64> = hotspots.iter().map(|(id, _)| *id).collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            // Build result with path + import_count (no file_id)
            let results: Vec<serde_json::Value> = hotspots.iter()
                .filter_map(|(id, import_count)| {
                    paths.get(id).map(|path| json!({
                        "path": path,
                        "import_count": import_count,
                    }))
                })
                .collect();

            let response = json!({
                "pagination": {
                    "total": total_count,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit_val,
                    "has_more": has_more,
                },
                "results": results,
            });

            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&response)?
                }]
            }))
        }
        "find_circular" => {
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let offset = arguments["offset"].as_u64().map(|n| n as usize);
            let sort = arguments["sort"].as_str().map(|s| s.to_string());

            let cache = CacheManager::new(".");
            let deps_index = DependencyIndex::new(cache);

            let mut all_cycles = deps_index.detect_circular_dependencies()?;

            // Apply sorting (default: descending - longest cycles first)
            let sort_order = sort.as_deref().unwrap_or("desc");
            match sort_order {
                "asc" => {
                    // Ascending: shortest cycles first
                    all_cycles.sort_by_key(|cycle| cycle.len());
                }
                "desc" => {
                    // Descending: longest cycles first (default)
                    all_cycles.sort_by_key(|cycle| std::cmp::Reverse(cycle.len()));
                }
                _ => {
                    return Err(anyhow::anyhow!("Invalid sort order '{}'. Supported: asc, desc", sort_order));
                }
            }

            let total_count = all_cycles.len();

            // Apply offset pagination
            let offset_val = offset.unwrap_or(0);
            let mut cycles: Vec<_> = all_cycles.into_iter().skip(offset_val).collect();

            // Apply limit (default 200)
            let limit_val = limit.unwrap_or(200);
            cycles.truncate(limit_val);

            let count = cycles.len();
            let has_more = offset_val + count < total_count;

            // Convert cycles to paths (without file_ids)
            let file_ids: Vec<i64> = cycles.iter().flat_map(|c| c.iter()).copied().collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            let results: Vec<serde_json::Value> = cycles.iter()
                .map(|cycle| {
                    let cycle_paths: Vec<_> = cycle.iter()
                        .filter_map(|id| paths.get(id).cloned())
                        .collect();
                    json!({
                        "paths": cycle_paths,
                    })
                })
                .collect();

            let response = json!({
                "pagination": {
                    "total": total_count,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit_val,
                    "has_more": has_more,
                },
                "results": results,
            });

            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&response)?
                }]
            }))
        }
        "find_unused" => {
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let offset = arguments["offset"].as_u64().map(|n| n as usize);

            let cache = CacheManager::new(".");
            let deps_index = DependencyIndex::new(cache);

            let all_unused = deps_index.find_unused_files()?;
            let total_count = all_unused.len();

            // Apply offset pagination
            let offset_val = offset.unwrap_or(0);
            let mut unused: Vec<_> = all_unused.into_iter().skip(offset_val).collect();

            // Apply limit (default 200)
            let limit_val = limit.unwrap_or(200);
            unused.truncate(limit_val);

            let count = unused.len();
            let has_more = offset_val + count < total_count;

            // Get paths for all unused file IDs
            let paths = deps_index.get_file_paths(&unused)?;

            // Build result (flat array of path strings)
            let results: Vec<String> = unused.iter()
                .filter_map(|id| paths.get(id).cloned())
                .collect();

            let response = json!({
                "pagination": {
                    "total": total_count,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit_val,
                    "has_more": has_more,
                },
                "results": results,
            });

            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&response)?
                }]
            }))
        }
        "find_islands" => {
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let offset = arguments["offset"].as_u64().map(|n| n as usize);
            let min_island_size = arguments["min_island_size"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(2);
            let max_island_size = arguments["max_island_size"]
                .as_u64()
                .map(|n| n as usize);
            let sort = arguments["sort"].as_str().map(|s| s.to_string());

            let cache = CacheManager::new(".");
            let deps_index = DependencyIndex::new(cache);

            let all_islands = deps_index.find_islands()?;
            let total_components = all_islands.len();

            // Get total file count for percentage calculation
            let total_files = deps_index.get_cache().stats()?.total_files as usize;

            // Calculate max_island_size default: min of 500 or 50% of total files
            let max_size = max_island_size.unwrap_or_else(|| {
                let fifty_percent = (total_files as f64 * 0.5) as usize;
                fifty_percent.min(500)
            });

            // Filter islands by size
            let mut islands: Vec<_> = all_islands.into_iter()
                .filter(|island| {
                    let size = island.len();
                    size >= min_island_size && size <= max_size
                })
                .collect();

            // Apply sorting (default: descending - largest islands first)
            let sort_order = sort.as_deref().unwrap_or("desc");
            match sort_order {
                "asc" => {
                    // Ascending: smallest islands first
                    islands.sort_by_key(|island| island.len());
                }
                "desc" => {
                    // Descending: largest islands first (default)
                    islands.sort_by_key(|island| std::cmp::Reverse(island.len()));
                }
                _ => {
                    return Err(anyhow::anyhow!("Invalid sort order '{}'. Supported: asc, desc", sort_order));
                }
            }

            let _filtered_count = total_components - islands.len();
            let total_after_filter = islands.len();

            // Apply offset pagination
            let offset_val = offset.unwrap_or(0);
            if offset_val > 0 && offset_val < islands.len() {
                islands = islands.into_iter().skip(offset_val).collect();
            } else if offset_val >= islands.len() {
                islands.clear();
            }

            // Apply limit (default 200)
            let limit_val = limit.unwrap_or(200);
            islands.truncate(limit_val);

            let count = islands.len();
            let has_more = offset_val + count < total_after_filter;

            // Get all file IDs from all islands
            let file_ids: Vec<i64> = islands.iter().flat_map(|island| island.iter()).copied().collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            // Build result (array of islands with paths, no file_ids)
            let results: Vec<serde_json::Value> = islands.iter()
                .enumerate()
                .map(|(idx, island)| {
                    let island_paths: Vec<_> = island.iter()
                        .filter_map(|id| paths.get(id).cloned())
                        .collect();
                    json!({
                        "island_id": idx + 1,
                        "size": island.len(),
                        "paths": island_paths,
                    })
                })
                .collect();

            let response = json!({
                "pagination": {
                    "total": total_after_filter,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit_val,
                    "has_more": has_more,
                },
                "results": results,
            });

            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&response)?
                }]
            }))
        }
        "analyze_summary" => {
            let min_dependents = arguments["min_dependents"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(2);

            let cache = CacheManager::new(".");
            let deps_index = DependencyIndex::new(cache);

            let cycles = deps_index.detect_circular_dependencies()?;
            let hotspots = deps_index.find_hotspots(None, min_dependents)?;
            let unused = deps_index.find_unused_files()?;
            let all_islands = deps_index.find_islands()?;

            let summary = json!({
                "circular_dependencies": cycles.len(),
                "hotspots": hotspots.len(),
                "unused_files": unused.len(),
                "islands": all_islands.len(),
                "min_dependents": min_dependents,
            });

            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&summary)?
                }]
            }))
        }
        "gather_context" => {
            // Parse optional parameters
            let structure = arguments["structure"].as_bool().unwrap_or(false);
            let file_types = arguments["file_types"].as_bool().unwrap_or(false);
            let project_type = arguments["project_type"].as_bool().unwrap_or(false);
            let framework = arguments["framework"].as_bool().unwrap_or(false);
            let entry_points = arguments["entry_points"].as_bool().unwrap_or(false);
            let test_layout = arguments["test_layout"].as_bool().unwrap_or(false);
            let config_files = arguments["config_files"].as_bool().unwrap_or(false);
            let depth = arguments["depth"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(2);
            let path = arguments["path"].as_str().map(|s| s.to_string());

            // Build context options
            let mut opts = crate::context::ContextOptions {
                structure,
                path,
                file_types,
                project_type,
                framework,
                entry_points,
                test_layout,
                config_files,
                depth,
                json: false,  // MCP always returns text format
            };

            // If no context flags specified, return minimal orientation context only.
            // Requesting all types by default floods agent context windows (2000-5000 tokens).
            let no_flags_set = opts.is_empty();
            if no_flags_set {
                opts.project_type = true;
                opts.entry_points = true;
            }

            let cache = CacheManager::new(".");
            let context = crate::context::generate_context(&cache, &opts)?;

            let hint = if no_flags_set {
                "\n\n---\nHint: this is the minimal orientation view. Pass any combination of these flags for more detail: structure, file_types, framework, test_layout, config_files."
            } else {
                ""
            };

            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": format!("{}{}", context, hint)
                }]
            }))
        }
        "check_index_status" => {
            let cache = CacheManager::new(".");

            if !cache.exists() {
                let result = json!({
                    "status": "missing",
                    "action_required": "rfx index"
                });
                return Ok(json!({
                    "content": [{
                        "type": "text",
                        "text": serde_json::to_string(&result)?
                    }]
                }));
            }

            let engine = QueryEngine::new(cache);
            let (status, _can_trust, warning) = engine.get_index_status()?;

            let status_str = match status {
                IndexStatus::Fresh => "fresh",
                IndexStatus::Stale => "stale",
            };

            let result = if let Some(w) = warning {
                let mut obj = json!({
                    "status": status_str,
                    "reason": w.reason,
                    "action_required": w.action_required
                });
                if let Some(fm) = w.files_modified {
                    obj["files_modified"] = json!(fm);
                }
                obj
            } else {
                json!({ "status": status_str })
            };

            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&result)?
                }]
            }))
        }
        "find_references" => {
            let pattern = arguments["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing pattern"))?
                .to_string();

            let lang = arguments["lang"].as_str().map(|s| s.to_string());
            let kind = arguments["kind"].as_str().map(|s| s.to_string());
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let offset = arguments["offset"].as_u64().map(|n| n as usize);
            let glob_patterns: Vec<String> = arguments["glob"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();
            let exclude_patterns: Vec<String> = arguments["exclude"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();
            let force = arguments["force"].as_bool().unwrap_or(false);

            let language = parse_language(lang);
            let parsed_kind = parse_symbol_kind(kind);

            // Search 1: Find symbol definition (symbols_mode=true, small cap)
            let def_filter = QueryFilter {
                language,
                kind: parsed_kind,
                use_ast: false,
                use_regex: false,
                limit: Some(5),
                symbols_mode: true,
                expand: false,
                file_pattern: None,
                exact: false,
                use_contains: false,
                timeout_secs: 30,
                glob_patterns: glob_patterns.clone(),
                exclude_patterns: exclude_patterns.clone(),
                paths_only: false,
                offset: None,
                force,
                suppress_output: true,
                include_dependencies: false,
                ..Default::default()
            };

            let cache = CacheManager::new(".");
            let engine = QueryEngine::new(cache);
            let def_response = engine.search_with_metadata(&pattern, def_filter)?;

            // Extract first definition as a compact object (reuse MatchResult's Serialize impl)
            let definition: Option<serde_json::Value> = def_response.results.first().and_then(|fg| {
                fg.matches.first().map(|m| {
                    let mut def_obj = serde_json::to_value(m).unwrap_or(json!({}));
                    if let serde_json::Value::Object(ref mut map) = def_obj {
                        map.insert("path".to_string(), json!(fg.path.clone()));
                        // Truncate preview if present
                        if let Some(preview) = map.get("preview").and_then(|v| v.as_str()) {
                            let truncated = crate::cli::truncate_preview(preview, DEFAULT_MCP_PREVIEW_LENGTH);
                            map.insert("preview".to_string(), json!(truncated));
                        }
                    }
                    def_obj
                })
            });

            // Search 2: Find all textual references (symbols_mode=false)
            let ref_filter = QueryFilter {
                language,
                kind: None,
                use_ast: false,
                use_regex: false,
                limit: limit.or(Some(100)),
                symbols_mode: false,
                expand: false,
                file_pattern: None,
                exact: false,
                use_contains: false,
                timeout_secs: 30,
                glob_patterns,
                exclude_patterns,
                paths_only: false,
                offset,
                force,
                suppress_output: true,
                include_dependencies: false,
                ..Default::default()
            };

            let ref_response = engine.search_with_metadata(&pattern, ref_filter)?;

            // Flatten references to compact {path, line, preview} array
            let references: Vec<serde_json::Value> = ref_response.results.iter()
                .flat_map(|fg| {
                    fg.matches.iter().map(move |m| {
                        json!({
                            "path": fg.path,
                            "line": m.span.start_line,
                            "preview": crate::cli::truncate_preview(&m.preview, DEFAULT_MCP_PREVIEW_LENGTH)
                        })
                    })
                })
                .collect();

            let total_references = ref_response.pagination.total;

            let response = json!({
                "status": ref_response.status,
                "definition": definition,
                "references": references,
                "total_references": total_references,
                "pagination": ref_response.pagination,
            });

            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&response)?
                }]
            }))
        }
        _ => Err(anyhow::anyhow!("Unknown tool: {}", name)),
    }
}

/// Process a single JSON-RPC request
fn process_request(request: JsonRpcRequest) -> JsonRpcResponse {
    log::debug!("MCP request: method={}", request.method);

    let result = match request.method.as_str() {
        "initialize" => handle_initialize(request.params),
        "tools/list" => handle_list_tools(request.params),
        "tools/call" => handle_call_tool(request.params),
        _ => Err(anyhow::anyhow!("Unknown method: {}", request.method)),
    };

    match result {
        Ok(value) => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: Some(value),
            error: None,
        },
        Err(e) => {
            log::error!("MCP error: {}", e);
            let msg = e.to_string();
            // REF-67: map to the correct JSON-RPC error code instead of always using -32603.
            let (code, kind, message) = if let Some(re) = e.downcast_ref::<crate::errors::ReflexError>() {
                let code = match re {
                    crate::errors::ReflexError::QuerySyntaxError(_) => -32602, // Invalid params
                    _ => -32603, // Internal error
                };
                (code, re.kind().to_string(), re.to_string())
            } else if msg.starts_with("Unknown method:") {
                (-32601, "MethodNotFound".to_string(), msg)
            } else if msg.starts_with("Missing")
                || msg.starts_with("Unknown tool:")
                || msg.starts_with("Invalid or unsupported")
            {
                (-32602, "InvalidParams".to_string(), msg)
            } else {
                (-32603, "InternalError".to_string(), msg)
            };
            JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(JsonRpcError {
                    code,
                    message: message.clone(),
                    data: Some(json!({ "kind": kind })),
                }),
            }
        }
    }
}

/// Handle a JSON-RPC Notification. Notifications never receive a response.
fn handle_notification(method: &str, _params: Option<Value>) {
    match method {
        "notifications/initialized" | "notifications/cancelled" => {
            log::debug!("MCP notification: {}", method);
        }
        other => {
            log::debug!("MCP unknown notification: {}", other);
        }
    }
}

/// Run the MCP server on stdio.
pub fn run_mcp_server() -> Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    run_mcp_server_io(stdin.lock(), stdout.lock())
}

/// Run the MCP server reading JSON-RPC messages from `reader` and writing
/// responses to `writer`. Exposed at crate-level for integration tests.
pub fn run_mcp_server_io<R: BufRead, W: Write>(reader: R, mut writer: W) -> Result<()> {
    log::info!("Starting Reflex MCP server on stdio");

    for line in reader.lines() {
        let line = line?;

        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        log::debug!("MCP input: {}", line);

        // Parse JSON-RPC message
        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(req) => req,
            Err(e) => {
                log::error!("Failed to parse JSON-RPC request: {}", e);
                // REF-61: send -32700 Parse error instead of silently dropping.
                // Per JSON-RPC 2.0, use null id when the id cannot be determined.
                let error_response = JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: Some(Value::Null),
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("Parse error: {}", e),
                        data: None,
                    }),
                };
                let response_json = serde_json::to_string(&error_response)?;
                writeln!(writer, "{}", response_json)?;
                writer.flush()?;
                continue;
            }
        };

        // Notifications (no `id`) must not receive a response per JSON-RPC 2.0.
        if request.id.is_none() {
            handle_notification(&request.method, request.params);
            continue;
        }

        // Process request and write response
        let response = process_request(request);
        let response_json = serde_json::to_string(&response)?;
        writeln!(writer, "{}", response_json)?;
        writer.flush()?;

        log::debug!("MCP output: {}", response_json);
    }

    log::info!("Reflex MCP server stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn call_server(input: &str) -> String {
        let reader = Cursor::new(input.as_bytes());
        let mut output = Vec::new();
        run_mcp_server_io(reader, &mut output).unwrap();
        String::from_utf8(output).unwrap()
    }

    fn parse_first_response(raw: &str) -> serde_json::Value {
        let line = raw.lines().next().expect("no response line");
        serde_json::from_str(line).expect("invalid JSON response")
    }

    // REF-61: malformed JSON must return -32700 Parse error (not silence)
    #[test]
    fn test_parse_error_returns_32700() {
        let raw = call_server("not-valid-json\n");
        let resp = parse_first_response(&raw);
        assert_eq!(resp["jsonrpc"], "2.0");
        assert_eq!(resp["error"]["code"], -32700);
        // Per JSON-RPC 2.0, id must be null when id cannot be determined
        assert!(resp["id"].is_null(), "id must be null for parse errors");
    }

    // REF-67: unknown method returns -32601 (Method not found)
    #[test]
    fn test_unknown_method_returns_32601() {
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"no_such_method","params":null}"#;
        let raw = call_server(&format!("{}\n", req));
        let resp = parse_first_response(&raw);
        assert_eq!(resp["error"]["code"], -32601);
    }

    // REF-67: missing required param returns -32602 (Invalid params), not -32603
    #[test]
    fn test_missing_param_returns_32602() {
        // search_code requires "pattern" — omit it
        let req = r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"search_code","arguments":{}}}"#;
        let raw = call_server(&format!("{}\n", req));
        let resp = parse_first_response(&raw);
        assert_eq!(resp["error"]["code"], -32602);
    }

    // Notification (no id) must produce no response
    #[test]
    fn test_notification_produces_no_response() {
        let notif = r#"{"jsonrpc":"2.0","method":"notifications/initialized","params":null}"#;
        let raw = call_server(&format!("{}\n", notif));
        assert!(raw.trim().is_empty(), "notification must not get a response");
    }
}
