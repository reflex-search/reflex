//! Architecture map generation
//!
//! Produces dependency diagrams in mermaid or d2 format.
//! Uses detect_modules() for consistent sub-module resolution across all Pulse surfaces.

use anyhow::Result;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::cache::CacheManager;
use crate::dependency::DependencyIndex;

use super::wiki;

/// Zoom level for the architecture map
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MapZoom {
    /// Whole-repo view: modules as nodes
    Repo,
    /// Single module view: files within module as nodes
    Module(String),
}

/// Output format for the map
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MapFormat {
    Mermaid,
    D2,
}

impl std::str::FromStr for MapFormat {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "mermaid" => Ok(MapFormat::Mermaid),
            "d2" => Ok(MapFormat::D2),
            _ => anyhow::bail!("Unknown map format: {}. Supported: mermaid, d2", s),
        }
    }
}

/// Generate an architecture map
pub fn generate_map(
    cache: &CacheManager,
    zoom: &MapZoom,
    format: MapFormat,
) -> Result<String> {
    match zoom {
        MapZoom::Repo => generate_repo_map(cache, format),
        MapZoom::Module(module) => generate_module_map(cache, module, format),
    }
}

fn generate_repo_map(cache: &CacheManager, format: MapFormat) -> Result<String> {
    let db_path = cache.path().join("meta.db");
    let conn = Connection::open(&db_path)?;

    // Use detect_modules() for consistent sub-module resolution
    let modules = wiki::detect_modules(cache, &wiki::ModuleDiscoveryConfig::default())?;

    // Build module info for node labels
    let module_info: Vec<(String, usize)> = modules.iter()
        .map(|m| (m.path.clone(), m.file_count))
        .collect();

    // Get all file-level dependency edges
    let mut stmt = conn.prepare(
        "SELECT f1.path, f2.path
         FROM file_dependencies fd
         JOIN files f1 ON fd.file_id = f1.id
         JOIN files f2 ON fd.resolved_file_id = f2.id
         WHERE fd.resolved_file_id IS NOT NULL"
    )?;

    let file_edges: Vec<(String, String)> = stmt.query_map([], |row| {
        Ok((row.get(0)?, row.get(1)?))
    })?.collect::<Result<Vec<_>, _>>()?;

    // Aggregate file-level edges to module-level edges
    let mut module_edges: HashMap<(String, String), usize> = HashMap::new();
    for (src_file, tgt_file) in &file_edges {
        let src_module = find_owning_module(src_file, &modules);
        let tgt_module = find_owning_module(tgt_file, &modules);

        if src_module != tgt_module {
            *module_edges.entry((src_module, tgt_module)).or_insert(0) += 1;
        }
    }

    let mut edges: Vec<(String, String, usize)> = module_edges.into_iter()
        .map(|((s, t), c)| (s, t, c))
        .collect();
    edges.sort_by(|a, b| b.2.cmp(&a.2));

    // Get hotspots for highlighting
    let deps_index = DependencyIndex::new(cache.clone());
    let hotspots = deps_index.find_hotspots(Some(10), 5).unwrap_or_default();
    let hotspot_modules: HashSet<String> = hotspots.iter()
        .filter_map(|(id, _)| {
            deps_index.get_file_paths(&[*id]).ok()
                .and_then(|paths| paths.get(id).cloned())
                .map(|p| find_owning_module(&p, &modules))
        })
        .collect();

    match format {
        MapFormat::Mermaid => render_mermaid_repo(&module_info, &edges, &hotspot_modules),
        MapFormat::D2 => render_d2_repo(&module_info, &edges, &hotspot_modules),
    }
}

/// Find the most-specific module that owns a given file path
fn find_owning_module(file_path: &str, modules: &[wiki::ModuleDefinition]) -> String {
    let mut best_match = String::new();
    let mut best_len = 0;

    for module in modules {
        let prefix = format!("{}/", module.path);
        if file_path.starts_with(&prefix) && module.path.len() > best_len {
            best_match = module.path.clone();
            best_len = module.path.len();
        }
    }

    if best_match.is_empty() {
        file_path.split('/').next().unwrap_or("root").to_string()
    } else {
        best_match
    }
}

fn generate_module_map(cache: &CacheManager, module_path: &str, format: MapFormat) -> Result<String> {
    let db_path = cache.path().join("meta.db");
    let conn = Connection::open(&db_path)?;
    let pattern = format!("{}/%", module_path);

    // Get files in this module
    let mut stmt = conn.prepare(
        "SELECT id, path FROM files WHERE path LIKE ?1 ORDER BY path"
    )?;
    let files: Vec<(i64, String)> = stmt.query_map([&pattern], |row| {
        Ok((row.get(0)?, row.get(1)?))
    })?.collect::<Result<Vec<_>, _>>()?;

    // Get intra-module edges
    let mut stmt = conn.prepare(
        "SELECT f1.path, f2.path
         FROM file_dependencies fd
         JOIN files f1 ON fd.file_id = f1.id
         JOIN files f2 ON fd.resolved_file_id = f2.id
         WHERE f1.path LIKE ?1 AND f2.path LIKE ?1
           AND fd.resolved_file_id IS NOT NULL"
    )?;
    let edges: Vec<(String, String)> = stmt.query_map([&pattern], |row| {
        Ok((row.get(0)?, row.get(1)?))
    })?.collect::<Result<Vec<_>, _>>()?;

    match format {
        MapFormat::Mermaid => render_mermaid_module(module_path, &files, &edges),
        MapFormat::D2 => render_d2_module(module_path, &files, &edges),
    }
}

/// Create a Mermaid-safe node ID with a prefix to avoid reserved word collisions.
/// Mermaid v11 can choke on IDs that match internal keywords or contain certain patterns.
fn sanitize_id(s: &str) -> String {
    format!("m_{}", s.replace(['/', '.', '-', ' '], "_"))
}

fn render_mermaid_repo(
    modules: &[(String, usize)],
    edges: &[(String, String, usize)],
    hotspot_modules: &HashSet<String>,
) -> Result<String> {
    let mut out = String::from("graph LR\n");

    // Only emit modules that participate in at least one edge
    let connected: HashSet<&str> = edges.iter()
        .flat_map(|(s, t, _)| [s.as_str(), t.as_str()])
        .collect();

    for (module, count) in modules {
        if !connected.contains(module.as_str()) {
            continue;
        }
        let id = sanitize_id(module);
        out.push_str(&format!("  {}[\"{}/ ({} files)\"]\n", id, module, count));
    }

    out.push('\n');

    // Track thick edges for linkStyle directives
    let mut thick_edge_indices: Vec<usize> = Vec::new();
    for (i, (src, tgt, count)) in edges.iter().enumerate() {
        let src_id = sanitize_id(src);
        let tgt_id = sanitize_id(tgt);
        out.push_str(&format!("  {} -->|{}| {}\n", src_id, count, tgt_id));
        if *count > 5 {
            thick_edge_indices.push(i);
        }
    }

    // Apply thick stroke to high-count edges via linkStyle
    for idx in &thick_edge_indices {
        out.push_str(&format!("  linkStyle {} stroke-width:3px,stroke:#a78bfa\n", idx));
    }

    // High-contrast styling for dark theme
    out.push_str("\n  classDef default fill:#1a1a2e,stroke:#a78bfa,color:#e0e0e0\n");
    out.push_str("  classDef hotspot fill:#2a1030,stroke:#f472b6,color:#f472b6\n");
    if !hotspot_modules.is_empty() {
        for module in hotspot_modules {
            if !connected.contains(module.as_str()) {
                continue;
            }
            let id = sanitize_id(module);
            out.push_str(&format!("  class {} hotspot\n", id));
        }
    }

    // Clickable nodes → wiki pages (only connected modules)
    for (module, _) in modules {
        if !connected.contains(module.as_str()) {
            continue;
        }
        let id = sanitize_id(module);
        let slug = module.replace('/', "-");
        out.push_str(&format!("  click {} \"/wiki/{}/\"\n", id, slug));
    }

    Ok(out)
}

/// Generate a layered (top-to-bottom) architecture diagram with Tier 1 subgraphs containing Tier 2 children
pub fn generate_layered_map(
    cache: &CacheManager,
    format: MapFormat,
) -> Result<String> {
    let db_path = cache.path().join("meta.db");
    let conn = Connection::open(&db_path)?;
    let modules = wiki::detect_modules(cache, &wiki::ModuleDiscoveryConfig::default())?;

    let module_info: Vec<(String, usize, u8)> = modules.iter()
        .map(|m| (m.path.clone(), m.file_count, m.tier))
        .collect();

    // Get module-level edges
    let mut stmt = conn.prepare(
        "SELECT f1.path, f2.path
         FROM file_dependencies fd
         JOIN files f1 ON fd.file_id = f1.id
         JOIN files f2 ON fd.resolved_file_id = f2.id
         WHERE fd.resolved_file_id IS NOT NULL"
    )?;
    let file_edges: Vec<(String, String)> = stmt.query_map([], |row| {
        Ok((row.get(0)?, row.get(1)?))
    })?.collect::<Result<Vec<_>, _>>()?;

    let mut module_edges: HashMap<(String, String), usize> = HashMap::new();
    for (src_file, tgt_file) in &file_edges {
        let src_module = find_owning_module(src_file, &modules);
        let tgt_module = find_owning_module(tgt_file, &modules);
        if src_module != tgt_module {
            *module_edges.entry((src_module, tgt_module)).or_insert(0) += 1;
        }
    }

    let mut edges: Vec<(String, String, usize)> = module_edges.into_iter()
        .map(|((s, t), c)| (s, t, c))
        .collect();
    edges.sort_by(|a, b| b.2.cmp(&a.2));

    let deps_index = DependencyIndex::new(cache.clone());
    let hotspots = deps_index.find_hotspots(Some(10), 5).unwrap_or_default();
    let hotspot_modules: HashSet<String> = hotspots.iter()
        .filter_map(|(id, _)| {
            deps_index.get_file_paths(&[*id]).ok()
                .and_then(|paths| paths.get(id).cloned())
                .map(|p| find_owning_module(&p, &modules))
        })
        .collect();

    match format {
        MapFormat::Mermaid => render_mermaid_layered(&module_info, &edges, &hotspot_modules),
        MapFormat::D2 => render_d2_repo(
            &module_info.iter().map(|(p, c, _)| (p.clone(), *c)).collect::<Vec<_>>(),
            &edges,
            &hotspot_modules,
        ),
    }
}

fn render_mermaid_layered(
    modules: &[(String, usize, u8)],
    edges: &[(String, String, usize)],
    hotspot_modules: &HashSet<String>,
) -> Result<String> {
    let mut out = String::from("flowchart TB\n");

    // Only emit modules that participate in at least one edge
    let connected: HashSet<&str> = edges.iter()
        .flat_map(|(s, t, _)| [s.as_str(), t.as_str()])
        .collect();

    // Group Tier 2 modules under their Tier 1 parent
    let tier1: Vec<&(String, usize, u8)> = modules.iter().filter(|m| m.2 == 1).collect();
    let tier2: Vec<&(String, usize, u8)> = modules.iter().filter(|m| m.2 == 2).collect();

    // Build proxy map: Tier 1 modules that become subgraphs get an inner proxy node.
    // Mermaid v11 cannot target subgraph IDs with edges, classDef, or click handlers,
    // so we create a real node inside the subgraph to receive those interactions.
    let mut proxy_map: HashMap<String, String> = HashMap::new();

    for t1 in &tier1 {
        if !connected.contains(t1.0.as_str()) {
            continue;
        }
        let t1_id = sanitize_id(&t1.0);
        let children: Vec<&&(String, usize, u8)> = tier2.iter()
            .filter(|t2| t2.0.starts_with(&format!("{}/", t1.0)) && connected.contains(t2.0.as_str()))
            .collect();

        if children.is_empty() {
            // Standalone Tier 1 node (no subgraph needed)
            out.push_str(&format!("  {}[\"{}/ ({} files)\"]\n", t1_id, t1.0, t1.1));
        } else {
            // Subgraph with proxy node for edges/styling/clicks
            let proxy_id = format!("{}_self", t1_id);
            proxy_map.insert(t1.0.clone(), proxy_id.clone());

            out.push_str(&format!("  subgraph {} [\"{}/ \"]\n", t1_id, t1.0));
            out.push_str(&format!("    {}[\"{}/ ({} files)\"]\n", proxy_id, t1.0, t1.1));
            for child in &children {
                let child_id = sanitize_id(&child.0);
                let short = child.0.strip_prefix(&format!("{}/", t1.0)).unwrap_or(&child.0);
                out.push_str(&format!("    {}[\"{}/ ({} files)\"]\n", child_id, short, child.1));
            }
            out.push_str("  end\n");
        }
    }

    // Orphan Tier 2 modules (no matching Tier 1 parent)
    for t2 in &tier2 {
        if !connected.contains(t2.0.as_str()) {
            continue;
        }
        let has_parent = tier1.iter().any(|t1| t2.0.starts_with(&format!("{}/", t1.0)));
        if !has_parent {
            let id = sanitize_id(&t2.0);
            out.push_str(&format!("  {}[\"{}/ ({} files)\"]\n", id, t2.0, t2.1));
        }
    }

    out.push('\n');

    // Track thick edges for linkStyle directives
    // Resolve edge endpoints through proxy_map so edges target proxy nodes, not subgraphs
    let mut thick_edge_indices: Vec<usize> = Vec::new();
    for (i, (src, tgt, count)) in edges.iter().enumerate() {
        let src_id = proxy_map.get(src)
            .cloned()
            .unwrap_or_else(|| sanitize_id(src));
        let tgt_id = proxy_map.get(tgt)
            .cloned()
            .unwrap_or_else(|| sanitize_id(tgt));
        out.push_str(&format!("  {} -->|{}| {}\n", src_id, count, tgt_id));
        if *count > 5 {
            thick_edge_indices.push(i);
        }
    }

    // Apply thick stroke to high-count edges via linkStyle
    for idx in &thick_edge_indices {
        out.push_str(&format!("  linkStyle {} stroke-width:3px,stroke:#a78bfa\n", idx));
    }

    // Styling — apply classDef to proxy nodes, not subgraph containers
    out.push_str("\n  classDef default fill:#1a1a2e,stroke:#a78bfa,color:#e0e0e0\n");
    out.push_str("  classDef hotspot fill:#2a1030,stroke:#f472b6,color:#f472b6\n");
    for module in hotspot_modules {
        if !connected.contains(module.as_str()) {
            continue;
        }
        let id = proxy_map.get(module)
            .cloned()
            .unwrap_or_else(|| sanitize_id(module));
        out.push_str(&format!("  class {} hotspot\n", id));
    }

    // Clickable nodes — apply click to proxy nodes, not subgraph containers
    for (module, _, _) in modules {
        if !connected.contains(module.as_str()) {
            continue;
        }
        let id = proxy_map.get(module)
            .cloned()
            .unwrap_or_else(|| sanitize_id(module));
        let slug = module.replace('/', "-");
        out.push_str(&format!("  click {} \"/wiki/{}/\"\n", id, slug));
    }

    Ok(out)
}

fn render_d2_repo(
    modules: &[(String, usize)],
    edges: &[(String, String, usize)],
    hotspot_modules: &HashSet<String>,
) -> Result<String> {
    let mut out = String::new();

    for (module, count) in modules {
        let id = sanitize_id(module);
        out.push_str(&format!("{}: \"{}/ ({} files)\"\n", id, module, count));
        if hotspot_modules.contains(module) {
            out.push_str(&format!("{}.style.fill: \"#ff6b6b\"\n", id));
        }
    }

    out.push('\n');

    for (src, tgt, count) in edges {
        let src_id = sanitize_id(src);
        let tgt_id = sanitize_id(tgt);
        out.push_str(&format!("{} -> {}: {}\n", src_id, tgt_id, count));
    }

    Ok(out)
}

fn render_mermaid_module(
    module_path: &str,
    files: &[(i64, String)],
    edges: &[(String, String)],
) -> Result<String> {
    let mut out = format!("graph LR\n  subgraph {}\n", module_path);

    for (_, path) in files {
        let id = sanitize_id(path);
        let short_name = path.rsplit('/').next().unwrap_or(path);
        out.push_str(&format!("    {}[\"{}\"]\n", id, short_name));
    }

    for (src, tgt) in edges {
        let src_id = sanitize_id(src);
        let tgt_id = sanitize_id(tgt);
        out.push_str(&format!("    {} --> {}\n", src_id, tgt_id));
    }

    out.push_str("  end\n");

    Ok(out)
}

fn render_d2_module(
    module_path: &str,
    files: &[(i64, String)],
    edges: &[(String, String)],
) -> Result<String> {
    let mut out = format!("{}: {{\n", sanitize_id(module_path));

    for (_, path) in files {
        let id = sanitize_id(path);
        let short_name = path.rsplit('/').next().unwrap_or(path);
        out.push_str(&format!("  {}: \"{}\"\n", id, short_name));
    }

    for (src, tgt) in edges {
        let src_id = sanitize_id(src);
        let tgt_id = sanitize_id(tgt);
        out.push_str(&format!("  {} -> {}\n", src_id, tgt_id));
    }

    out.push_str("}\n");

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_id() {
        assert_eq!(sanitize_id("src/parsers"), "m_src_parsers");
        assert_eq!(sanitize_id("my-module.rs"), "m_my_module_rs");
    }

    #[test]
    fn test_mermaid_repo_output() {
        let modules = vec![("src".to_string(), 50), ("tests".to_string(), 10)];
        let edges = vec![("src".to_string(), "tests".to_string(), 3)];
        let hotspots = HashSet::new();

        let result = render_mermaid_repo(&modules, &edges, &hotspots).unwrap();
        assert!(result.contains("graph LR"));
        assert!(result.contains("src"));
        assert!(result.contains("tests"));
        assert!(result.contains("-->"));
    }

    #[test]
    fn test_d2_repo_output() {
        let modules = vec![("src".to_string(), 50)];
        let edges = vec![];
        let hotspots = HashSet::from(["src".to_string()]);

        let result = render_d2_repo(&modules, &edges, &hotspots).unwrap();
        assert!(result.contains("src:"));
        assert!(result.contains("#ff6b6b"));
    }

    #[test]
    fn test_mermaid_repo_filters_orphans() {
        let modules = vec![
            ("src".to_string(), 50),
            ("tests".to_string(), 10),
            ("docs".to_string(), 5),       // orphan — no edges
            ("scripts".to_string(), 2),    // orphan — no edges
        ];
        let edges = vec![("src".to_string(), "tests".to_string(), 3)];
        let hotspots = HashSet::from(["docs".to_string()]);

        let result = render_mermaid_repo(&modules, &edges, &hotspots).unwrap();

        // Connected modules are present
        assert!(result.contains("m_src["), "connected module 'src' should be in output");
        assert!(result.contains("m_tests["), "connected module 'tests' should be in output");

        // Orphan modules are excluded
        assert!(!result.contains("m_docs"), "orphan 'docs' should not be in output");
        assert!(!result.contains("m_scripts"), "orphan 'scripts' should not be in output");

        // Hotspot styling for orphan should not appear
        assert!(!result.contains("class m_docs hotspot"), "orphan hotspot should not be styled");

        // Click handlers for orphans should not appear
        assert!(!result.contains("click m_docs"), "orphan should not have click handler");
        assert!(!result.contains("click m_scripts"), "orphan should not have click handler");
    }

    #[test]
    fn test_mermaid_layered_proxy_nodes() {
        let modules = vec![
            ("src".to_string(), 80, 1u8),
            ("src/parsers".to_string(), 15, 2u8),
            ("tests".to_string(), 10, 1u8),
        ];
        let edges = vec![
            ("src/parsers".to_string(), "src".to_string(), 16),
            ("src".to_string(), "tests".to_string(), 3),
        ];
        let hotspots = HashSet::from(["src".to_string()]);

        let result = render_mermaid_layered(&modules, &edges, &hotspots).unwrap();

        // Subgraph for src should exist (it has children)
        assert!(result.contains("subgraph m_src ["), "Tier 1 with children should be a subgraph");

        // Proxy node inside the subgraph
        assert!(result.contains("m_src_self["), "subgraph should contain proxy node");

        // Edges should target proxy node, not subgraph ID
        assert!(result.contains("m_src_self"), "edges should reference proxy node");
        assert!(!result.contains(" -->|16| m_src\n"), "edges should NOT target bare subgraph ID");

        // classDef should target proxy node
        assert!(result.contains("class m_src_self hotspot"), "hotspot class should target proxy node");

        // click should target proxy node
        assert!(result.contains("click m_src_self"), "click handler should target proxy node");

        // tests is standalone Tier 1 (no children), should be a regular node
        assert!(result.contains("m_tests["), "standalone Tier 1 should be a regular node");
        assert!(!result.contains("subgraph m_tests"), "standalone Tier 1 should not be a subgraph");
    }

    #[test]
    fn test_find_owning_module() {
        let modules = vec![
            wiki::ModuleDefinition {
                path: "src".to_string(),
                tier: 1,
                file_count: 80,
                total_lines: 50000,
                languages: vec!["Rust".to_string()],
            },
            wiki::ModuleDefinition {
                path: "src/parsers".to_string(),
                tier: 2,
                file_count: 15,
                total_lines: 8000,
                languages: vec!["Rust".to_string()],
            },
        ];

        assert_eq!(find_owning_module("src/parsers/rust.rs", &modules), "src/parsers");
        assert_eq!(find_owning_module("src/main.rs", &modules), "src");
        assert_eq!(find_owning_module("tests/integration.rs", &modules), "tests");
    }
}
