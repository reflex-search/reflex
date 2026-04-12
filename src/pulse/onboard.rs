//! Onboard Guide: "Getting Started" page for developer onboarding
//!
//! Identifies entry points (main files, CLI handlers, API routes),
//! suggests a reading order via dependency topology, and provides
//! structural context for LLM narration.

use anyhow::{Context, Result};
use rusqlite::Connection;
use rusqlite::OptionalExtension;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

use crate::cache::CacheManager;
use crate::models::{SearchResult, SymbolKind};

/// Kind of entry point detected
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntryPointKind {
    CliBinary,
    HttpServer,
    Library,
    Script,
    TestRunner,
}

impl std::fmt::Display for EntryPointKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EntryPointKind::CliBinary => write!(f, "CLI Binary"),
            EntryPointKind::HttpServer => write!(f, "HTTP Server"),
            EntryPointKind::Library => write!(f, "Library"),
            EntryPointKind::Script => write!(f, "Script"),
            EntryPointKind::TestRunner => write!(f, "Test Runner"),
        }
    }
}

/// A detected entry point in the codebase
#[derive(Debug, Clone)]
pub struct EntryPoint {
    pub path: String,
    pub kind: EntryPointKind,
    pub key_symbols: Vec<String>,
}

/// A layer in the reading order (BFS from entry points)
#[derive(Debug, Clone)]
pub struct ReadingLayer {
    pub depth: usize,
    pub label: String,
    pub files: Vec<String>,
}

/// Complete reading order computed via BFS from entry points
#[derive(Debug, Clone)]
pub struct ReadingOrder {
    pub layers: Vec<ReadingLayer>,
}

/// Full onboard data
#[derive(Debug, Clone)]
pub struct OnboardData {
    pub entry_points: Vec<EntryPoint>,
    pub reading_order: ReadingOrder,
    pub project_stats: ProjectStats,
    pub narration: Option<String>,
}

/// Quick stats for the onboard page
#[derive(Debug, Clone)]
pub struct ProjectStats {
    pub total_files: usize,
    pub total_lines: usize,
    pub languages: Vec<(String, usize)>,
    pub module_count: usize,
}

/// Detect entry points by matching well-known file patterns and names
pub fn detect_entry_points(cache: &CacheManager) -> Result<Vec<EntryPoint>> {
    let db_path = cache.path().join("meta.db");
    let conn = Connection::open(&db_path)
        .context("Failed to open meta.db")?;

    // Get all file paths
    let mut stmt = conn.prepare("SELECT path FROM files ORDER BY path")?;
    let paths: Vec<String> = stmt.query_map([], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    let mut entry_points = Vec::new();
    let mut seen_paths = HashSet::new();

    for path in &paths {
        let filename = Path::new(path).file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("");
        let lower = filename.to_lowercase();

        // CLI binary entry points
        if matches!(filename, "main.rs" | "main.go" | "main.py" | "main.c" | "main.cpp" | "main.zig")
            || (filename == "cli.rs" || filename == "cli.ts" || filename == "cli.py" || filename == "cli.js")
        {
            if seen_paths.insert(path.clone()) {
                let kind = EntryPointKind::CliBinary;
                let symbols = extract_key_symbols_for_entry(&conn, path);
                entry_points.push(EntryPoint { path: path.clone(), kind, key_symbols: symbols });
            }
            continue;
        }

        // HTTP server entry points
        if matches!(filename, "server.rs" | "server.ts" | "server.js" | "server.py" | "server.go"
            | "app.rs" | "app.ts" | "app.js" | "app.py" | "app.go"
            | "routes.rs" | "routes.ts" | "routes.js" | "routes.py")
        {
            if seen_paths.insert(path.clone()) {
                let symbols = extract_key_symbols_for_entry(&conn, path);
                entry_points.push(EntryPoint { path: path.clone(), kind: EntryPointKind::HttpServer, key_symbols: symbols });
            }
            continue;
        }

        // Library entry points
        if matches!(filename, "lib.rs" | "mod.rs" | "index.ts" | "index.js" | "__init__.py" | "mod.go") {
            // Only include top-level or shallow lib/index files, not deeply nested ones
            let depth = path.matches('/').count();
            if depth <= 2 && seen_paths.insert(path.clone()) {
                let symbols = extract_key_symbols_for_entry(&conn, path);
                entry_points.push(EntryPoint { path: path.clone(), kind: EntryPointKind::Library, key_symbols: symbols });
            }
            continue;
        }

        // Script entry points (package.json scripts, Makefile, etc.)
        if matches!(filename, "Makefile" | "Rakefile" | "Taskfile.yml" | "justfile") {
            if seen_paths.insert(path.clone()) {
                entry_points.push(EntryPoint { path: path.clone(), kind: EntryPointKind::Script, key_symbols: vec![] });
            }
            continue;
        }

        // Test runners
        if matches!(lower.as_str(), "conftest.py" | "jest.config.js" | "jest.config.ts"
            | "vitest.config.ts" | "vitest.config.js" | "pytest.ini" | "setup.cfg")
            && path.matches('/').count() <= 1
        {
            if seen_paths.insert(path.clone()) {
                entry_points.push(EntryPoint { path: path.clone(), kind: EntryPointKind::TestRunner, key_symbols: vec![] });
            }
        }
    }

    // Sort: CLI first, then HTTP, then Library, then others
    entry_points.sort_by_key(|ep| match ep.kind {
        EntryPointKind::CliBinary => 0,
        EntryPointKind::HttpServer => 1,
        EntryPointKind::Library => 2,
        EntryPointKind::Script => 3,
        EntryPointKind::TestRunner => 4,
    });

    Ok(entry_points)
}

/// Extract key symbol names for an entry point file from the symbol cache.
///
/// Queries the `symbols` table which stores all symbols for a file as a
/// serialized JSON blob (`symbols_json` column containing `Vec<SearchResult>`).
fn extract_key_symbols_for_entry(conn: &Connection, path: &str) -> Vec<String> {
    // Get file_id
    let file_id: Option<i64> = conn.query_row(
        "SELECT id FROM files WHERE path = ?1",
        [path],
        |row| row.get(0),
    ).ok();

    let Some(file_id) = file_id else { return vec![] };

    // Query the symbols table for this file's serialized symbols
    let symbols_json: Option<String> = conn.query_row(
        "SELECT symbols_json FROM symbols WHERE file_id = ?1",
        [file_id],
        |row| row.get(0),
    ).optional().ok().flatten();

    let Some(json) = symbols_json else { return vec![] };

    // Deserialize and filter to key symbol kinds
    let symbols: Vec<SearchResult> = match serde_json::from_str(&json) {
        Ok(s) => s,
        Err(_) => return vec![],
    };

    symbols.iter()
        .filter(|sr| matches!(sr.kind,
            SymbolKind::Function | SymbolKind::Struct | SymbolKind::Class
            | SymbolKind::Trait | SymbolKind::Interface
        ))
        .filter_map(|sr| sr.symbol.clone())
        .take(8)
        .collect()
}

/// Compute reading order via BFS from entry points through the dependency graph
pub fn compute_reading_order(cache: &CacheManager, entry_points: &[EntryPoint]) -> Result<ReadingOrder> {
    let db_path = cache.path().join("meta.db");
    let conn = Connection::open(&db_path)?;

    // Build adjacency list: file_id -> [dependent file_ids]
    // We traverse in the direction entry_point -> its dependencies
    let mut deps: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut path_to_id: HashMap<String, i64> = HashMap::new();
    let mut id_to_path: HashMap<i64, String> = HashMap::new();

    // Load file id mappings
    let mut stmt = conn.prepare("SELECT id, path FROM files")?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
    })?;
    for row in rows.flatten() {
        path_to_id.insert(row.1.clone(), row.0);
        id_to_path.insert(row.0, row.1);
    }

    // Load dependency edges (file -> its dependency)
    let mut stmt = conn.prepare(
        "SELECT file_id, resolved_file_id FROM file_dependencies WHERE resolved_file_id IS NOT NULL"
    )?;
    let edges = stmt.query_map([], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
    })?;
    for edge in edges.flatten() {
        deps.entry(edge.0).or_default().push(edge.1);
    }

    // BFS from entry points
    let mut visited: HashSet<i64> = HashSet::new();
    let mut queue: VecDeque<(i64, usize)> = VecDeque::new();
    let mut layers_map: HashMap<usize, Vec<String>> = HashMap::new();

    for ep in entry_points {
        if let Some(&file_id) = path_to_id.get(&ep.path) {
            if visited.insert(file_id) {
                queue.push_back((file_id, 0));
            }
        }
    }

    while let Some((file_id, depth)) = queue.pop_front() {
        if depth > 5 { continue; } // Cap depth to keep reading order manageable

        if let Some(path) = id_to_path.get(&file_id) {
            layers_map.entry(depth).or_default().push(path.clone());
        }

        if let Some(dep_ids) = deps.get(&file_id) {
            for &dep_id in dep_ids {
                if visited.insert(dep_id) {
                    queue.push_back((dep_id, depth + 1));
                }
            }
        }
    }

    let layer_labels = [
        "Entry Points",
        "Direct Dependencies",
        "Core Infrastructure",
        "Supporting Modules",
        "Deep Dependencies",
        "Periphery",
    ];

    let mut layers: Vec<ReadingLayer> = Vec::new();
    for depth in 0..=5 {
        if let Some(files) = layers_map.get(&depth) {
            if !files.is_empty() {
                layers.push(ReadingLayer {
                    depth,
                    label: layer_labels.get(depth).unwrap_or(&"Other").to_string(),
                    files: files.clone(),
                });
            }
        }
    }

    Ok(ReadingOrder { layers })
}

/// Gather project stats for the onboard page
pub fn gather_project_stats(cache: &CacheManager, module_count: usize) -> Result<ProjectStats> {
    let db_path = cache.path().join("meta.db");
    let conn = Connection::open(&db_path)?;

    let total_files: usize = conn.query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))?;
    let total_lines: usize = conn.query_row("SELECT COALESCE(SUM(line_count), 0) FROM files", [], |r| r.get(0))?;

    let mut stmt = conn.prepare(
        "SELECT COALESCE(language, 'other'), COUNT(*) FROM files GROUP BY language ORDER BY COUNT(*) DESC LIMIT 10"
    )?;
    let languages: Vec<(String, usize)> = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
    })?.filter_map(|r| r.ok()).collect();

    Ok(ProjectStats {
        total_files,
        total_lines,
        languages,
        module_count,
    })
}

/// Generate the full onboard data (structural phase)
pub fn generate_onboard_structural(cache: &CacheManager, module_count: usize) -> Result<OnboardData> {
    let entry_points = detect_entry_points(cache)?;
    let reading_order = compute_reading_order(cache, &entry_points)?;
    let project_stats = gather_project_stats(cache, module_count)?;

    Ok(OnboardData {
        entry_points,
        reading_order,
        project_stats,
        narration: None,
    })
}

/// Build structural context string for LLM narration
pub fn build_onboard_context(data: &OnboardData) -> String {
    let mut ctx = String::new();

    ctx.push_str(&format!(
        "Project size: {} files, {} lines across {} modules\n\n",
        data.project_stats.total_files,
        data.project_stats.total_lines,
        data.project_stats.module_count,
    ));

    // Languages
    ctx.push_str("Languages:\n");
    for (lang, count) in &data.project_stats.languages {
        ctx.push_str(&format!("- {}: {} files\n", lang, count));
    }
    ctx.push('\n');

    // Entry points
    ctx.push_str("Entry points:\n");
    for ep in &data.entry_points {
        ctx.push_str(&format!("- {} ({})", ep.path, ep.kind));
        if !ep.key_symbols.is_empty() {
            ctx.push_str(&format!(" — key symbols: {}", ep.key_symbols.join(", ")));
        }
        ctx.push('\n');
    }
    ctx.push('\n');

    // Reading order
    ctx.push_str("Suggested reading order (BFS from entry points through dependencies):\n");
    for layer in &data.reading_order.layers {
        ctx.push_str(&format!("Layer {} — {} ({} files):\n", layer.depth, layer.label, layer.files.len()));
        for file in layer.files.iter().take(15) {
            ctx.push_str(&format!("  - {}\n", file));
        }
        if layer.files.len() > 15 {
            ctx.push_str(&format!("  ... and {} more\n", layer.files.len() - 15));
        }
    }

    ctx
}

/// Render onboard data as markdown (structural content)
pub fn render_onboard_markdown(data: &OnboardData) -> String {
    let mut md = String::new();

    // Narration (if available)
    if let Some(ref narration) = data.narration {
        md.push_str(narration);
        md.push_str("\n\n");
    }

    // Quick stats
    md.push_str("## At a Glance\n\n");
    md.push_str(&format!(
        "| Metric | Value |\n|---|---|\n| Files | {} |\n| Lines | {} |\n| Modules | {} |\n| Languages | {} |\n\n",
        data.project_stats.total_files,
        data.project_stats.total_lines,
        data.project_stats.module_count,
        data.project_stats.languages.len(),
    ));

    // Entry points table
    md.push_str("## Entry Points\n\n");
    md.push_str("These are the starting files — where execution begins or where the public API is exposed.\n\n");
    md.push_str("| File | Kind | Key Symbols |\n|---|---|---|\n");
    for ep in &data.entry_points {
        let symbols = if ep.key_symbols.is_empty() {
            "—".to_string()
        } else {
            ep.key_symbols.iter().map(|s| format!("`{}`", s)).collect::<Vec<_>>().join(", ")
        };
        md.push_str(&format!("| `{}` | {} | {} |\n", ep.path, ep.kind, symbols));
    }
    md.push('\n');

    // Reading order as Mermaid flowchart
    if !data.reading_order.layers.is_empty() {
        md.push_str("## Reading Order\n\n");
        md.push_str("Start at the top and work your way down. Each layer depends on the one below it.\n\n");

        md.push_str("{% mermaid() %}\nflowchart TD\n");
        for layer in &data.reading_order.layers {
            let node_id = format!("L{}", layer.depth);
            let file_list: String = layer.files.iter().take(6)
                .map(|f| {
                    // Extract just the filename for readability
                    Path::new(f).file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(f)
                })
                .collect::<Vec<_>>()
                .join(", ");
            let suffix = if layer.files.len() > 6 {
                format!(" +{} more", layer.files.len() - 6)
            } else {
                String::new()
            };
            md.push_str(&format!(
                "    {}[\"{}: {}{}\"]\n",
                node_id, layer.label, file_list, suffix
            ));
        }

        // Connect layers top-to-bottom
        for i in 0..data.reading_order.layers.len().saturating_sub(1) {
            md.push_str(&format!("    L{} --> L{}\n", i, i + 1));
        }

        // Styling
        md.push_str("    style L0 fill:#a78bfa,color:#0d0d0d,stroke:#a78bfa\n");
        md.push_str("{% end %}\n\n");

        // Detailed file lists per layer
        for layer in &data.reading_order.layers {
            md.push_str(&format!("### Layer {}: {}\n\n", layer.depth, layer.label));
            for file in &layer.files {
                md.push_str(&format!("- `{}`\n", file));
            }
            md.push('\n');
        }
    }

    md
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_point_kind_display() {
        assert_eq!(format!("{}", EntryPointKind::CliBinary), "CLI Binary");
        assert_eq!(format!("{}", EntryPointKind::HttpServer), "HTTP Server");
        assert_eq!(format!("{}", EntryPointKind::Library), "Library");
    }

    #[test]
    fn test_render_onboard_markdown_empty() {
        let data = OnboardData {
            entry_points: vec![],
            reading_order: ReadingOrder { layers: vec![] },
            project_stats: ProjectStats {
                total_files: 100,
                total_lines: 5000,
                languages: vec![("Rust".to_string(), 80), ("Python".to_string(), 20)],
                module_count: 5,
            },
            narration: None,
        };
        let md = render_onboard_markdown(&data);
        assert!(md.contains("## At a Glance"));
        assert!(md.contains("100"));
        assert!(md.contains("5000"));
    }

    #[test]
    fn test_build_onboard_context() {
        let data = OnboardData {
            entry_points: vec![EntryPoint {
                path: "src/main.rs".to_string(),
                kind: EntryPointKind::CliBinary,
                key_symbols: vec!["main".to_string()],
            }],
            reading_order: ReadingOrder {
                layers: vec![ReadingLayer {
                    depth: 0,
                    label: "Entry Points".to_string(),
                    files: vec!["src/main.rs".to_string()],
                }],
            },
            project_stats: ProjectStats {
                total_files: 50,
                total_lines: 3000,
                languages: vec![("Rust".to_string(), 50)],
                module_count: 3,
            },
            narration: None,
        };
        let ctx = build_onboard_context(&data);
        assert!(ctx.contains("src/main.rs"));
        assert!(ctx.contains("CLI Binary"));
        assert!(ctx.contains("Entry Points"));
    }
}
