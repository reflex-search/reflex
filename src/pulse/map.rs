//! Architecture map generation
//!
//! Produces dependency diagrams in mermaid or d2 format.
//! Supports repo-level (modules as nodes) and module-level (files as nodes) zoom.

use anyhow::Result;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::cache::CacheManager;
use crate::dependency::DependencyIndex;

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

    // Get module-to-module edges by aggregating file-level deps
    let mut stmt = conn.prepare(
        "SELECT
            CASE WHEN INSTR(f1.path, '/') > 0
                 THEN SUBSTR(f1.path, 1, INSTR(f1.path, '/') - 1)
                 ELSE '.'
            END AS source_module,
            CASE WHEN INSTR(f2.path, '/') > 0
                 THEN SUBSTR(f2.path, 1, INSTR(f2.path, '/') - 1)
                 ELSE '.'
            END AS target_module,
            COUNT(*) AS edge_count
         FROM file_dependencies fd
         JOIN files f1 ON fd.file_id = f1.id
         JOIN files f2 ON fd.resolved_file_id = f2.id
         WHERE fd.resolved_file_id IS NOT NULL
         GROUP BY source_module, target_module
         HAVING source_module != target_module
         ORDER BY edge_count DESC"
    )?;

    let edges: Vec<(String, String, usize)> = stmt.query_map([], |row| {
        Ok((row.get(0)?, row.get(1)?, row.get::<_, i64>(2)? as usize))
    })?.collect::<Result<Vec<_>, _>>()?;

    // Get module file counts for node labels
    let mut stmt = conn.prepare(
        "SELECT
            CASE WHEN INSTR(path, '/') > 0
                 THEN SUBSTR(path, 1, INSTR(path, '/') - 1)
                 ELSE '.'
            END AS module,
            COUNT(*) AS file_count
         FROM files
         GROUP BY module
         ORDER BY file_count DESC"
    )?;
    let modules: Vec<(String, usize)> = stmt.query_map([], |row| {
        Ok((row.get(0)?, row.get::<_, i64>(1)? as usize))
    })?.collect::<Result<Vec<_>, _>>()?;

    // Get hotspots for highlighting
    let deps_index = DependencyIndex::new(cache.clone());
    let hotspots = deps_index.find_hotspots(Some(10), 5).unwrap_or_default();
    let hotspot_modules: HashSet<String> = hotspots.iter()
        .filter_map(|(id, _)| {
            deps_index.get_file_paths(&[*id]).ok()
                .and_then(|paths| paths.get(id).cloned())
                .and_then(|p| p.split('/').next().map(|s| s.to_string()))
        })
        .collect();

    match format {
        MapFormat::Mermaid => render_mermaid_repo(&modules, &edges, &hotspot_modules),
        MapFormat::D2 => render_d2_repo(&modules, &edges, &hotspot_modules),
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

    let _file_ids: HashSet<i64> = files.iter().map(|(id, _)| *id).collect();

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

fn sanitize_id(s: &str) -> String {
    s.replace(['/', '.', '-', ' '], "_")
}

fn render_mermaid_repo(
    modules: &[(String, usize)],
    edges: &[(String, String, usize)],
    hotspot_modules: &HashSet<String>,
) -> Result<String> {
    let mut out = String::from("graph LR\n");

    for (module, count) in modules {
        let id = sanitize_id(module);
        out.push_str(&format!("  {}[\"{}/ ({} files)\"]\n", id, module, count));
    }

    out.push('\n');

    for (src, tgt, count) in edges {
        let src_id = sanitize_id(src);
        let tgt_id = sanitize_id(tgt);
        if *count > 5 {
            out.push_str(&format!("  {} ==>|{}| {}\n", src_id, count, tgt_id));
        } else {
            out.push_str(&format!("  {} -->|{}| {}\n", src_id, count, tgt_id));
        }
    }

    if !hotspot_modules.is_empty() {
        out.push_str("\n  classDef hotspot fill:#ff6b6b,stroke:#c0392b\n");
        for module in hotspot_modules {
            let id = sanitize_id(module);
            out.push_str(&format!("  class {} hotspot\n", id));
        }
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
        assert_eq!(sanitize_id("src/parsers"), "src_parsers");
        assert_eq!(sanitize_id("my-module.rs"), "my_module_rs");
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
}
