//! Wiki generation: per-module documentation pages
//!
//! Generates a living wiki page for each detected module (directory) in the codebase.
//! Pages include structural sections (dependencies, dependents, key symbols, metrics)
//! and optional LLM-generated summaries.

use anyhow::{Context, Result};
use rayon::prelude::*;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::cache::CacheManager;
use crate::dependency::DependencyIndex;
use crate::models::{Language, SymbolKind};
use crate::parsers::ParserFactory;
use crate::query::{QueryEngine, QueryFilter};
use crate::semantic::context::CodebaseContext;
use crate::semantic::providers::LlmProvider;

use super::llm_cache::LlmCache;
use super::narrate;

/// A detected module in the codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDefinition {
    /// Module path (e.g., "src", "tests", "src/parsers")
    pub path: String,
    /// Module tier: 1 = top-level, 2 = depth-2/3
    pub tier: u8,
    /// Number of files in this module
    pub file_count: usize,
    /// Total line count
    pub total_lines: usize,
    /// Languages present in this module
    pub languages: Vec<String>,
}

/// A generated wiki page for a module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WikiPage {
    pub module_path: String,
    pub title: String,
    pub sections: WikiSections,
}

/// Structural sections of a wiki page (all built without LLM)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WikiSections {
    pub summary: Option<String>,
    pub structure: String,
    pub dependencies: String,
    pub dependents: String,
    pub dependency_diagram: Option<String>,
    pub circular_deps: Option<String>,
    pub key_symbols: String,
    pub metrics: String,
    pub recent_changes: Option<String>,
}

/// Configuration for module discovery depth and filtering
#[derive(Debug, Clone)]
pub struct ModuleDiscoveryConfig {
    /// Max tier level (1 = top-level only, 2 = include sub-modules)
    pub max_depth: u8,
    /// Minimum file count for a module to be included
    pub min_files: usize,
}

impl Default for ModuleDiscoveryConfig {
    fn default() -> Self {
        Self { max_depth: 2, min_files: 1 }
    }
}

/// Detect modules in the codebase using CodebaseContext
///
/// Returns both top-level directories (Tier 1) and their immediate
/// sub-directories with 3+ files (Tier 2). This produces granular modules
/// like `src/parsers`, `src/semantic`, `src/pulse` instead of just `src`.
///
/// Use `config` to control discovery depth and minimum file filtering.
pub fn detect_modules(cache: &CacheManager, config: &ModuleDiscoveryConfig) -> Result<Vec<ModuleDefinition>> {
    let context = CodebaseContext::extract(cache)
        .context("Failed to extract codebase context")?;

    let db_path = cache.path().join("meta.db");
    let conn = Connection::open(&db_path)?;

    let mut modules = Vec::new();

    // Tier 1: top-level directories
    for dir in &context.top_level_dirs {
        let dir_path = dir.trim_end_matches('/');
        if let Some(module) = build_module_def(&conn, dir_path, 1)? {
            if module.file_count >= config.min_files {
                modules.push(module);
            }
        }
    }

    // Tier 2: discover sub-modules under each Tier 1 module
    if config.max_depth >= 2 {
        let tier1_paths: Vec<String> = modules.iter().map(|m| m.path.clone()).collect();
        for parent in &tier1_paths {
            let sub_modules = discover_sub_modules(&conn, parent)?;
            for sub_path in sub_modules {
                // Skip exact duplicates
                if modules.iter().any(|m| m.path == sub_path) {
                    continue;
                }
                if let Some(module) = build_module_def(&conn, &sub_path, 2)? {
                    if module.file_count >= config.min_files {
                        modules.push(module);
                    }
                }
            }
        }

        // Also include common_paths that aren't covered by an exact match
        for path in &context.common_paths {
            let path_str = path.trim_end_matches('/');
            if modules.iter().any(|m| m.path == path_str) {
                continue;
            }
            if let Some(module) = build_module_def(&conn, path_str, 2)? {
                if module.file_count >= config.min_files {
                    modules.push(module);
                }
            }
        }
    }

    // Sort by path for deterministic output
    modules.sort_by(|a, b| a.path.cmp(&b.path));

    Ok(modules)
}

/// Discover immediate child directories under a parent module that have 3+ files.
///
/// Queries meta.db for files under `parent_path/` and groups them by their
/// immediate subdirectory. Returns paths like `src/parsers`, `src/semantic`.
fn discover_sub_modules(conn: &Connection, parent_path: &str) -> Result<Vec<String>> {
    let pattern = format!("{}/%", parent_path);
    let prefix_len = parent_path.len() + 1; // +1 for the '/'

    let mut stmt = conn.prepare(
        "SELECT
            SUBSTR(path, 1, ?2 + INSTR(SUBSTR(path, ?2 + 1), '/') - 1) AS sub_dir,
            COUNT(*) AS file_count
         FROM files
         WHERE path LIKE ?1
           AND INSTR(SUBSTR(path, ?2 + 1), '/') > 0
         GROUP BY sub_dir
         HAVING file_count >= 3
         ORDER BY file_count DESC"
    )?;

    let rows: Vec<String> = stmt.query_map(
        rusqlite::params![pattern, prefix_len],
        |row| row.get(0),
    )?.filter_map(|r| r.ok()).collect();

    Ok(rows)
}

/// Generate a wiki page for a single module
pub fn generate_wiki_page(
    cache: &CacheManager,
    module: &ModuleDefinition,
    all_modules: &[ModuleDefinition],
    diff: Option<&super::diff::SnapshotDiff>,
    no_llm: bool,
    provider: Option<&dyn LlmProvider>,
    llm_cache: Option<&LlmCache>,
    snapshot_id: &str,
) -> Result<WikiPage> {
    let db_path = cache.path().join("meta.db");
    let conn = Connection::open(&db_path)?;
    let deps_index = DependencyIndex::new(cache.clone());
    let query_engine = QueryEngine::new(cache.clone());

    // Find child modules of this module
    let prefix = format!("{}/", module.path);
    let child_modules: Vec<&ModuleDefinition> = all_modules.iter()
        .filter(|m| m.path.starts_with(&prefix) && m.path != module.path)
        .collect();

    // Build structural sections
    let structure = build_structure_section(&conn, &module.path, &child_modules)?;
    let dependencies = build_dependencies_section(&conn, &module.path, all_modules)?;
    let dependents = build_dependents_section(&conn, &deps_index, &module.path, all_modules)?;
    let dependency_diagram = build_dependency_diagram(&conn, &module.path, all_modules);
    let circular_deps = build_circular_deps_section(&deps_index, &module.path);
    let key_symbols = build_key_symbols_section(&conn, &module.path, &query_engine);
    let metrics = build_metrics_section(module, &conn)?;
    let recent_changes = diff.map(|d| build_recent_changes(d, &module.path));

    // Generate LLM summary when provider is available
    let summary = if !no_llm {
        if let (Some(provider), Some(llm_cache)) = (provider, llm_cache) {
            // Build combined structural context for the summary
            let mut context = String::new();
            context.push_str(&format!("Module: {}\n\n", module.path));
            context.push_str(&format!("## Structure\n{}\n\n", structure));
            context.push_str(&format!("## Dependencies\n{}\n\n", dependencies));
            context.push_str(&format!("## Dependents\n{}\n\n", dependents));
            context.push_str(&format!("## Key Symbols\n{}\n\n", key_symbols));
            context.push_str(&format!("## Metrics\n{}\n", metrics));

            narrate::narrate_section(
                provider,
                narrate::wiki_system_prompt(),
                &context,
                llm_cache,
                snapshot_id,
                &module.path,
            )
        } else {
            None
        }
    } else {
        None
    };

    Ok(WikiPage {
        module_path: module.path.clone(),
        title: format!("{}/", module.path),
        sections: WikiSections {
            summary,
            structure,
            dependencies,
            dependents,
            dependency_diagram,
            circular_deps,
            key_symbols,
            metrics,
            recent_changes,
        },
    })
}

/// Generate wiki pages for all detected modules
///
/// `provider` and `llm_cache` are created by the caller (site.rs or CLI handler).
pub fn generate_all_pages(
    cache: &CacheManager,
    diff: Option<&super::diff::SnapshotDiff>,
    no_llm: bool,
    snapshot_id: &str,
    provider: Option<&dyn LlmProvider>,
    llm_cache: Option<&LlmCache>,
    discovery_config: &ModuleDiscoveryConfig,
) -> Result<Vec<WikiPage>> {
    let modules = detect_modules(cache, discovery_config)?;
    let mut pages = Vec::new();

    if provider.is_some() {
        eprintln!("Generating wiki summaries...");
    }

    for module in &modules {
        match generate_wiki_page(
            cache,
            module,
            &modules,
            diff,
            no_llm,
            provider,
            llm_cache,
            snapshot_id,
        ) {
            Ok(page) => pages.push(page),
            Err(e) => {
                log::warn!("Failed to generate wiki page for {}: {}", module.path, e);
            }
        }
    }

    Ok(pages)
}

/// A wiki page with pre-built narration context for batch LLM dispatch
pub struct WikiPageWithContext {
    pub page: WikiPage,
    /// Combined structural context string for LLM narration (None if too brief)
    pub narration_context: Option<String>,
}

/// Generate all wiki pages structurally (no LLM), using rayon for parallelism.
///
/// Each module's structural sections are built concurrently. Returns pages
/// with `summary: None` and pre-built narration contexts for later batch dispatch.
pub fn generate_all_pages_structural(
    cache: &CacheManager,
    diff: Option<&super::diff::SnapshotDiff>,
    discovery_config: &ModuleDiscoveryConfig,
) -> Result<Vec<WikiPageWithContext>> {
    let modules = detect_modules(cache, discovery_config)?;

    // Use rayon par_iter for concurrent structural builds.
    // Each task opens its own DB connection and QueryEngine (safe for parallel use).
    let results: Vec<_> = modules.par_iter().map(|module| {
        let db_path = cache.path().join("meta.db");
        let conn = match Connection::open(&db_path) {
            Ok(c) => c,
            Err(e) => return Err(anyhow::anyhow!("Failed to open meta.db for {}: {}", module.path, e)),
        };
        let deps_index = DependencyIndex::new(cache.clone());
        let query_engine = QueryEngine::new(cache.clone());

        let prefix = format!("{}/", module.path);
        let child_modules: Vec<&ModuleDefinition> = modules.iter()
            .filter(|m| m.path.starts_with(&prefix) && m.path != module.path)
            .collect();

        let structure = build_structure_section(&conn, &module.path, &child_modules)?;
        let dependencies = build_dependencies_section(&conn, &module.path, &modules)?;
        let dependents = build_dependents_section(&conn, &deps_index, &module.path, &modules)?;
        let dependency_diagram = build_dependency_diagram(&conn, &module.path, &modules);
        let circular_deps = build_circular_deps_section(&deps_index, &module.path);
        let key_symbols = build_key_symbols_section(&conn, &module.path, &query_engine);
        let metrics = build_metrics_section(module, &conn)?;
        let recent_changes = diff.map(|d| build_recent_changes(d, &module.path));

        // Build narration context string
        let mut context = String::new();
        context.push_str(&format!("Module: {}\n\n", module.path));
        context.push_str(&format!("## Structure\n{}\n\n", structure));
        context.push_str(&format!("## Dependencies\n{}\n\n", dependencies));
        context.push_str(&format!("## Dependents\n{}\n\n", dependents));
        context.push_str(&format!("## Key Symbols\n{}\n\n", key_symbols));
        context.push_str(&format!("## Metrics\n{}\n", metrics));

        let narration_context = Some(context);

        Ok(WikiPageWithContext {
            page: WikiPage {
                module_path: module.path.clone(),
                title: format!("{}/", module.path),
                sections: WikiSections {
                    summary: None,
                    structure,
                    dependencies,
                    dependents,
                    dependency_diagram,
                    circular_deps,
                    key_symbols,
                    metrics,
                    recent_changes,
                },
            },
            narration_context,
        })
    }).collect();

    // Collect results, logging failures
    let mut pages = Vec::new();
    for result in results {
        match result {
            Ok(page) => pages.push(page),
            Err(e) => log::warn!("Failed to generate wiki page: {}", e),
        }
    }

    // Sort by module path for deterministic output
    pages.sort_by(|a, b| a.page.module_path.cmp(&b.page.module_path));

    Ok(pages)
}

/// Render wiki pages as (filename, markdown) pairs
pub fn render_wiki_markdown(pages: &[WikiPage]) -> Vec<(String, String)> {
    pages.iter().map(|page| {
        let filename = page.module_path.replace('/', "_") + ".md";
        let mut md = String::new();

        md.push_str(&format!("# {}\n\n", page.title));

        if let Some(summary) = &page.sections.summary {
            md.push_str(summary);
            md.push_str("\n\n");
        }

        md.push_str("## Structure\n\n");
        md.push_str(&page.sections.structure);
        md.push_str("\n\n");

        if let Some(diagram) = &page.sections.dependency_diagram {
            md.push_str("## Dependency Diagram\n\n");
            md.push_str("```mermaid\n");
            md.push_str(diagram);
            md.push_str("```\n\n");
        }

        md.push_str("## Dependencies\n\n");
        md.push_str(&page.sections.dependencies);
        md.push_str("\n\n");

        md.push_str("## Dependents\n\n");
        md.push_str(&page.sections.dependents);
        md.push_str("\n\n");

        if let Some(circular) = &page.sections.circular_deps {
            md.push_str("## Circular Dependencies\n\n");
            md.push_str(circular);
            md.push_str("\n\n");
        }

        md.push_str("## Key Symbols\n\n");
        md.push_str(&page.sections.key_symbols);
        md.push_str("\n\n");

        md.push_str("## Metrics\n\n");
        md.push_str(&page.sections.metrics);
        md.push_str("\n\n");

        if let Some(changes) = &page.sections.recent_changes {
            md.push_str("## Recent Changes\n\n");
            md.push_str(changes);
            md.push_str("\n\n");
        }

        (filename, md)
    }).collect()
}

// --- Private helpers ---

/// Build a focused mermaid dependency diagram for a single module.
/// Shows the module as center node with direct deps and dependents.
fn build_dependency_diagram(
    conn: &Connection,
    module_path: &str,
    all_modules: &[ModuleDefinition],
) -> Option<String> {
    let pattern = format!("{}/%", module_path);

    // Collect outgoing deps (module_path → target_module)
    let mut outgoing: HashMap<String, usize> = HashMap::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT f2.path FROM file_dependencies fd
         JOIN files f1 ON fd.file_id = f1.id
         JOIN files f2 ON fd.resolved_file_id = f2.id
         WHERE f1.path LIKE ?1 AND f2.path NOT LIKE ?1"
    ) {
        if let Ok(rows) = stmt.query_map([&pattern], |row| row.get::<_, String>(0)) {
            for dep_file in rows.flatten() {
                let target = find_owning_module(&dep_file, all_modules);
                *outgoing.entry(target).or_insert(0) += 1;
            }
        }
    }

    // Collect incoming deps (source_module → module_path)
    let mut incoming: HashMap<String, usize> = HashMap::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT f1.path FROM file_dependencies fd
         JOIN files f1 ON fd.file_id = f1.id
         JOIN files f2 ON fd.resolved_file_id = f2.id
         WHERE f2.path LIKE ?1 AND f1.path NOT LIKE ?1"
    ) {
        if let Ok(rows) = stmt.query_map([&pattern], |row| row.get::<_, String>(0)) {
            for dep_file in rows.flatten() {
                let source = find_owning_module(&dep_file, all_modules);
                *incoming.entry(source).or_insert(0) += 1;
            }
        }
    }

    if outgoing.is_empty() && incoming.is_empty() {
        return None;
    }

    let mut diagram = String::new();
    diagram.push_str("graph LR\n");

    // Sanitize node IDs with m_ prefix to avoid Mermaid reserved word collisions
    let sanitize = |s: &str| -> String {
        format!("m_{}", s.replace(['/', '.', '-', ' '], "_"))
    };

    let center_id = sanitize(module_path);
    diagram.push_str(&format!("    {}[\"<b>{}/</b>\"]\n", center_id, module_path));
    diagram.push_str(&format!("    style {} fill:#7aa2f7,color:#1a1b26,stroke:#7aa2f7\n", center_id));

    // Track all nodes for clickable links
    let mut all_node_paths: Vec<String> = vec![module_path.to_string()];

    // Outgoing edges (this module depends on)
    let mut out_sorted: Vec<_> = outgoing.into_iter().collect();
    out_sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (target, count) in out_sorted.iter().take(8) {
        let target_id = sanitize(target);
        diagram.push_str(&format!("    {}[\"{}/\"]\n", target_id, target));
        diagram.push_str(&format!("    {} -->|{}| {}\n", center_id, count, target_id));
        all_node_paths.push(target.clone());
    }

    // Incoming edges (modules that depend on this)
    let mut in_sorted: Vec<_> = incoming.into_iter().collect();
    in_sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (source, count) in in_sorted.iter().take(8) {
        let source_id = sanitize(source);
        // Avoid re-declaring if already declared as outgoing target
        if !out_sorted.iter().any(|(t, _)| t == source) {
            diagram.push_str(&format!("    {}[\"{}/\"]\n", source_id, source));
        }
        diagram.push_str(&format!("    {} -->|{}| {}\n", source_id, count, center_id));
        if !all_node_paths.contains(source) {
            all_node_paths.push(source.clone());
        }
    }

    // High-contrast styling
    diagram.push_str("    classDef default fill:#2d3250,stroke:#7aa2f7,color:#c0caf5\n");

    // Clickable nodes → wiki pages
    for node_path in &all_node_paths {
        let node_id = sanitize(node_path);
        let slug = node_path.replace('/', "-");
        diagram.push_str(&format!("    click {} \"/wiki/{}/\"\n", node_id, slug));
    }

    Some(diagram)
}

/// Build a circular dependencies section for a module.
/// Detects cycles that include files within this module's path.
fn build_circular_deps_section(deps_index: &DependencyIndex, module_path: &str) -> Option<String> {
    let cycles = match deps_index.detect_circular_dependencies() {
        Ok(c) => c,
        Err(_) => return None,
    };

    if cycles.is_empty() {
        return None;
    }

    // Collect all file IDs involved in cycles
    let all_ids: Vec<i64> = cycles.iter().flatten().copied().collect();
    let path_map = match deps_index.get_file_paths(&all_ids) {
        Ok(m) => m,
        Err(_) => return None,
    };

    let prefix = format!("{}/", module_path);

    // Filter cycles that involve at least one file in this module
    let mut relevant_cycles: Vec<Vec<String>> = Vec::new();
    for cycle in &cycles {
        let paths: Vec<String> = cycle.iter()
            .filter_map(|id| path_map.get(id).cloned())
            .collect();

        if paths.iter().any(|p| p.starts_with(&prefix)) {
            relevant_cycles.push(paths);
        }
    }

    if relevant_cycles.is_empty() {
        return None;
    }

    let mut content = String::new();
    content.push_str(&format!(
        "**{} circular {}** involving this module:\n\n",
        relevant_cycles.len(),
        if relevant_cycles.len() == 1 { "dependency" } else { "dependencies" }
    ));

    for (i, cycle) in relevant_cycles.iter().take(10).enumerate() {
        let short_paths: Vec<String> = cycle.iter()
            .map(|p| p.rsplit('/').next().unwrap_or(p).to_string())
            .collect();
        content.push_str(&format!("{}. {}\n", i + 1, short_paths.join(" → ")));
    }

    if relevant_cycles.len() > 10 {
        content.push_str(&format!("\n... and {} more. Run `rfx analyze --circular` for full list.\n", relevant_cycles.len() - 10));
    }

    Some(content)
}

fn build_module_def(conn: &Connection, path: &str, tier: u8) -> Result<Option<ModuleDefinition>> {
    let pattern = format!("{}/%", path);

    let file_count: usize = conn.query_row(
        "SELECT COUNT(*) FROM files WHERE path LIKE ?1 OR path = ?2",
        rusqlite::params![&pattern, path],
        |row| row.get(0),
    )?;

    if file_count == 0 {
        return Ok(None);
    }

    let total_lines: usize = conn.query_row(
        "SELECT COALESCE(SUM(line_count), 0) FROM files WHERE path LIKE ?1 OR path = ?2",
        rusqlite::params![&pattern, path],
        |row| row.get(0),
    )?;

    let mut stmt = conn.prepare(
        "SELECT DISTINCT language FROM files WHERE (path LIKE ?1 OR path = ?2) AND language IS NOT NULL"
    )?;
    let languages: Vec<String> = stmt.query_map(rusqlite::params![&pattern, path], |row| row.get(0))?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Some(ModuleDefinition {
        path: path.to_string(),
        tier,
        file_count,
        total_lines,
        languages,
    }))
}

fn build_structure_section(
    conn: &Connection,
    module_path: &str,
    child_modules: &[&ModuleDefinition],
) -> Result<String> {
    let pattern = format!("{}/%", module_path);

    let mut content = String::new();

    // Show sub-modules if this module has children — linked to their wiki pages
    if !child_modules.is_empty() {
        content.push_str("### Sub-modules\n\n");
        for child in child_modules {
            let short_name = child.path.strip_prefix(module_path)
                .unwrap_or(&child.path)
                .trim_start_matches('/');
            let child_slug = child.path.replace('/', "-");
            content.push_str(&format!(
                "- [**{}/**](/wiki/{}/) — {} files, {} lines ({})\n",
                short_name,
                child_slug,
                child.file_count,
                child.total_lines,
                child.languages.join(", "),
            ));
        }
        content.push('\n');
    }

    // Group files by immediate subdirectory with line counts
    let prefix_len = module_path.len() + 1;
    let mut stmt = conn.prepare(
        "SELECT path, language, COALESCE(line_count, 0) FROM files
         WHERE path LIKE ?1
         ORDER BY line_count DESC"
    )?;

    let files: Vec<(String, Option<String>, i64)> = stmt.query_map([&pattern], |row| {
        Ok((row.get(0)?, row.get(1)?, row.get(2)?))
    })?.collect::<Result<Vec<_>, _>>()?;

    // Group by immediate subdirectory
    let mut by_subdir: HashMap<String, (usize, i64)> = HashMap::new(); // subdir -> (file_count, total_lines)
    let mut direct_files: Vec<(String, i64)> = Vec::new();

    for (path, _, lines) in &files {
        let rel = &path[prefix_len.min(path.len())..];
        if let Some(slash_pos) = rel.find('/') {
            let subdir = &rel[..slash_pos];
            let entry = by_subdir.entry(subdir.to_string()).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += lines;
        } else {
            direct_files.push((path.clone(), *lines));
        }
    }

    // Language distribution
    let mut by_lang: HashMap<String, usize> = HashMap::new();
    for (_, lang, _) in &files {
        let lang = lang.as_deref().unwrap_or("other");
        *by_lang.entry(lang.to_string()).or_insert(0) += 1;
    }

    content.push_str("| Language | Files |\n|---|---|\n");
    let mut lang_counts: Vec<_> = by_lang.into_iter().collect();
    lang_counts.sort_by(|a, b| b.1.cmp(&a.1));
    for (lang, count) in &lang_counts {
        content.push_str(&format!("| {} | {} |\n", lang, count));
    }

    // Subdirectory breakdown
    if !by_subdir.is_empty() {
        let mut subdirs: Vec<_> = by_subdir.into_iter().collect();
        subdirs.sort_by(|a, b| b.1.1.cmp(&a.1.1)); // sort by lines desc

        content.push_str("\n### Directories\n\n");
        content.push_str("| Directory | Files | Lines |\n|---|---|---|\n");
        for (subdir, (count, lines)) in subdirs.iter().take(20) {
            content.push_str(&format!("| {}/ | {} | {} |\n", subdir, count, lines));
        }
    }

    // Top 10 largest files, with expandable overflow
    content.push_str("\n### Largest Files\n\n");
    let all_sorted: Vec<_> = files.iter()
        .map(|(path, _, lines)| (path.as_str(), *lines))
        .collect();
    for (path, lines) in all_sorted.iter().take(10) {
        let short = path.strip_prefix(&format!("{}/", module_path)).unwrap_or(path);
        content.push_str(&format!("- `{}` ({} lines)\n", short, lines));
    }

    let total = files.len();
    if total > 10 {
        content.push_str(&format!(
            "\n<details><summary><strong>Show {} more files</strong></summary>\n\n",
            total - 10
        ));
        for (path, lines) in all_sorted.iter().skip(10) {
            let short = path.strip_prefix(&format!("{}/", module_path)).unwrap_or(path);
            content.push_str(&format!("- `{}` ({} lines)\n", short, lines));
        }
        content.push_str("\n</details>\n");
    }

    Ok(content)
}

fn build_dependencies_section(
    conn: &Connection,
    module_path: &str,
    all_modules: &[ModuleDefinition],
) -> Result<String> {
    let pattern = format!("{}/%", module_path);
    let mut stmt = conn.prepare(
        "SELECT DISTINCT f2.path
         FROM file_dependencies fd
         JOIN files f1 ON fd.file_id = f1.id
         JOIN files f2 ON fd.resolved_file_id = f2.id
         WHERE f1.path LIKE ?1 AND f2.path NOT LIKE ?1
         ORDER BY f2.path"
    )?;

    let deps: Vec<String> = stmt.query_map([&pattern], |row| row.get(0))?
        .collect::<Result<Vec<_>, _>>()?;

    if deps.is_empty() {
        return Ok("No outgoing dependencies detected.".to_string());
    }

    // Group deps by target module
    let mut by_module: HashMap<String, Vec<String>> = HashMap::new();
    for dep in &deps {
        let target_module = find_owning_module(dep, all_modules);
        by_module.entry(target_module).or_default().push(dep.clone());
    }

    let mut groups: Vec<_> = by_module.into_iter().collect();
    groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    let total_files = deps.len();
    let total_modules = groups.len();

    let mut content = format!(
        "Depends on **{} files** across **{} modules**.\n\n",
        total_files, total_modules
    );

    for (module, files) in &groups {
        let module_slug = module.replace('/', "-");
        content.push_str(&format!("**[{}/](@/wiki/{}.md)** ({} files):\n", module, module_slug, files.len()));
        for f in files.iter().take(5) {
            let short = f.rsplit('/').next().unwrap_or(f);
            content.push_str(&format!("- `{}`\n", short));
        }
        if files.len() > 5 {
            content.push_str(&format!("- ... and {} more\n", files.len() - 5));
        }
        content.push('\n');
    }

    Ok(content)
}

fn build_dependents_section(
    conn: &Connection,
    _deps_index: &DependencyIndex,
    module_path: &str,
    all_modules: &[ModuleDefinition],
) -> Result<String> {
    let pattern = format!("{}/%", module_path);
    let mut stmt = conn.prepare(
        "SELECT DISTINCT f1.path
         FROM file_dependencies fd
         JOIN files f1 ON fd.file_id = f1.id
         JOIN files f2 ON fd.resolved_file_id = f2.id
         WHERE f2.path LIKE ?1 AND f1.path NOT LIKE ?1
         ORDER BY f1.path"
    )?;

    let dependents: Vec<String> = stmt.query_map([&pattern], |row| row.get(0))?
        .collect::<Result<Vec<_>, _>>()?;

    if dependents.is_empty() {
        return Ok("No incoming dependencies detected.".to_string());
    }

    // Group by source module
    let mut by_module: HashMap<String, Vec<String>> = HashMap::new();
    for dep in &dependents {
        let source_module = find_owning_module(dep, all_modules);
        by_module.entry(source_module).or_default().push(dep.clone());
    }

    let mut groups: Vec<_> = by_module.into_iter().collect();
    groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    let total_files = dependents.len();
    let total_modules = groups.len();

    let mut content = format!(
        "Used by **{} files** across **{} modules**.\n\n",
        total_files, total_modules
    );

    for (module, files) in &groups {
        let module_slug = module.replace('/', "-");
        content.push_str(&format!("**[{}/](@/wiki/{}.md)** ({} files):\n", module, module_slug, files.len()));
        for f in files.iter().take(5) {
            let short = f.rsplit('/').next().unwrap_or(f);
            content.push_str(&format!("- `{}`\n", short));
        }
        if files.len() > 5 {
            content.push_str(&format!("- ... and {} more\n", files.len() - 5));
        }
        content.push('\n');
    }

    Ok(content)
}

/// Language keywords and common variable names that are noise in "Key Symbols" rankings.
/// These appear in thousands of files and tell users nothing about the module.
const SYMBOL_BLOCKLIST: &[&str] = &[
    // Multi-language keywords
    "return", "this", "self", "super", "new", "null", "true", "false", "none",
    "class", "function", "var", "let", "const", "static", "public", "private",
    "protected", "abstract", "virtual", "override", "final", "async", "await",
    "import", "export", "module", "package", "namespace", "use", "from", "as",
    "if", "else", "for", "while", "do", "switch", "case", "default", "break",
    "continue", "try", "catch", "throw", "throws", "finally", "yield",
    "void", "int", "bool", "string", "float", "double", "char", "byte",
    "struct", "enum", "trait", "impl", "interface", "type", "where",
    // Common generic variable names
    "data", "value", "name", "key", "item", "items", "list", "result",
    "error", "err", "msg", "args", "opts", "params", "config", "options",
    "index", "count", "size", "length", "path", "file", "line", "text",
    "input", "output", "request", "response", "context", "state", "props",
    "init", "main", "run", "get", "set", "add", "delete", "update", "create",
    "test", "setup", "describe", "expect",
];

/// Symbol kinds considered high-value for "Key definitions" rankings.
/// These represent meaningful domain abstractions, not individual variables.
const PRIORITY_SYMBOL_KINDS: &[&str] = &[
    "Function", "Struct", "Class", "Trait", "Interface",
    "Enum", "Macro", "Type", "Constant",
];

/// Extract a doc comment preceding (or following, for Python) a symbol definition.
///
/// Walks backwards from `start_line` to collect contiguous comment lines, skipping
/// attributes/decorators. For Python, walks forward to find triple-quoted docstrings.
/// Returns the cleaned comment text with syntax prefixes stripped, or None.
fn extract_doc_comment(source: &str, start_line: usize, language: &Language) -> Option<String> {
    let lines: Vec<&str> = source.lines().collect();
    if start_line == 0 || start_line > lines.len() {
        return None;
    }

    // Python: walk forward from the definition line to find a docstring
    if matches!(language, Language::Python) {
        // Look at lines after the def/class line for a triple-quoted docstring
        let search_start = start_line; // start_line is 1-indexed, so index = start_line - 1 is the def line
        for i in search_start..lines.len().min(search_start + 3) {
            let trimmed = lines[i].trim();
            if trimmed.is_empty() {
                continue;
            }
            // Check for triple-quoted docstring opening
            if trimmed.starts_with("\"\"\"") || trimmed.starts_with("'''") {
                let quote = &trimmed[..3];
                // Single-line docstring: """text"""
                if trimmed.len() > 6 && trimmed.ends_with(quote) {
                    let inner = trimmed[3..trimmed.len() - 3].trim();
                    if !inner.is_empty() {
                        return Some(inner.to_string());
                    }
                }
                // Multi-line docstring
                let mut doc_lines = Vec::new();
                let first_content = trimmed[3..].trim();
                if !first_content.is_empty() {
                    doc_lines.push(first_content.to_string());
                }
                for j in (i + 1)..lines.len() {
                    let line = lines[j].trim();
                    if line.contains(quote) {
                        let before_close = line.trim_end_matches(quote).trim();
                        if !before_close.is_empty() {
                            doc_lines.push(before_close.to_string());
                        }
                        break;
                    }
                    doc_lines.push(line.to_string());
                }
                let result = doc_lines.join("\n").trim().to_string();
                if !result.is_empty() {
                    return Some(result);
                }
            }
            break; // Non-empty, non-docstring line — no docstring
        }
        return None;
    }

    // All other languages: walk backwards from the line before the symbol
    let mut idx = start_line.saturating_sub(2); // Convert to 0-indexed, then go one line up
    let mut comment_lines: Vec<String> = Vec::new();

    // Skip attributes/decorators walking backwards
    loop {
        if idx >= lines.len() {
            break;
        }
        let trimmed = lines[idx].trim();
        // Rust attributes: #[...] or #![...]
        if trimmed.starts_with("#[") || trimmed.starts_with("#![") {
            if idx == 0 { return None; }
            idx -= 1;
            continue;
        }
        // Java/Kotlin/Python-style decorators: @Something
        if trimmed.starts_with('@') && trimmed.len() > 1 && trimmed[1..].starts_with(|c: char| c.is_alphabetic()) {
            if idx == 0 { return None; }
            idx -= 1;
            continue;
        }
        // PHP attributes: #[Attribute]
        if trimmed.starts_with("#[") {
            if idx == 0 { return None; }
            idx -= 1;
            continue;
        }
        break;
    }

    // Determine comment style based on language
    match language {
        Language::Rust => {
            // Rust: /// or //! line comments, or /** */ block comments
            // Check for block comment ending on this line first
            if idx < lines.len() && lines[idx].trim().ends_with("*/") {
                return extract_block_comment(&lines, idx, "/**");
            }
            // Line comments: /// or //!
            while idx < lines.len() {
                let trimmed = lines[idx].trim();
                if trimmed.starts_with("///") {
                    let content = trimmed.trim_start_matches('/').trim();
                    comment_lines.push(content.to_string());
                } else if trimmed.starts_with("//!") {
                    let content = trimmed[3..].trim().to_string();
                    comment_lines.push(content);
                } else {
                    break;
                }
                if idx == 0 { break; }
                idx -= 1;
            }
        }
        Language::Go => {
            // Go: // comment lines before func
            while idx < lines.len() {
                let trimmed = lines[idx].trim();
                if trimmed.starts_with("//") {
                    let content = trimmed[2..].trim().to_string();
                    comment_lines.push(content);
                } else {
                    break;
                }
                if idx == 0 { break; }
                idx -= 1;
            }
        }
        Language::Ruby => {
            // Ruby: # comment lines
            while idx < lines.len() {
                let trimmed = lines[idx].trim();
                if trimmed.starts_with('#') && !trimmed.starts_with("#!") {
                    let content = trimmed[1..].trim().to_string();
                    comment_lines.push(content);
                } else {
                    break;
                }
                if idx == 0 { break; }
                idx -= 1;
            }
        }
        _ => {
            // JS/TS/Java/Kotlin/PHP/C#/C/C++/Zig: /** */ block or /// line comments
            if idx < lines.len() {
                let trimmed = lines[idx].trim();
                if trimmed.ends_with("*/") {
                    return extract_block_comment(&lines, idx, "/**");
                }
                // /// line comments (TypeScript, C#, etc.)
                if trimmed.starts_with("///") || trimmed.starts_with("//") {
                    while idx < lines.len() {
                        let t = lines[idx].trim();
                        if t.starts_with("///") {
                            comment_lines.push(t.trim_start_matches('/').trim().to_string());
                        } else if t.starts_with("//") && !t.starts_with("///") {
                            comment_lines.push(t[2..].trim().to_string());
                        } else {
                            break;
                        }
                        if idx == 0 { break; }
                        idx -= 1;
                    }
                }
            }
        }
    }

    if comment_lines.is_empty() {
        return None;
    }

    // Reverse because we collected bottom-up
    comment_lines.reverse();
    let result = comment_lines.join("\n").trim().to_string();
    if result.is_empty() { None } else { Some(result) }
}

/// Extract a block comment (/** ... */) by walking backwards from the closing line.
fn extract_block_comment(lines: &[&str], end_idx: usize, open_marker: &str) -> Option<String> {
    let mut doc_lines: Vec<String> = Vec::new();
    let mut idx = end_idx;

    loop {
        let trimmed = lines[idx].trim();

        // Check if this line contains the opening marker
        if trimmed.starts_with(open_marker) || trimmed.starts_with("/*") {
            // Single-line block comment: /** text */
            let content = trimmed
                .trim_start_matches(open_marker)
                .trim_start_matches("/*")
                .trim_end_matches("*/")
                .trim_end_matches('*')
                .trim();
            if !content.is_empty() {
                doc_lines.push(content.to_string());
            }
            break;
        }

        // Middle or end line of block comment
        let content = trimmed
            .trim_end_matches("*/")
            .trim_start_matches('*')
            .trim();
        if !content.is_empty() {
            doc_lines.push(content.to_string());
        }

        if idx == 0 { break; }
        idx -= 1;
    }

    doc_lines.reverse();
    let result = doc_lines.join("\n").trim().to_string();
    if result.is_empty() { None } else { Some(result) }
}

/// HTML-escape text to prevent doc comments from being interpreted as markup.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
     .replace('<', "&lt;")
     .replace('>', "&gt;")
}

/// Render a single "By Kind" entry as a pure HTML `<li>` element.
/// Single-line docs are appended inline; multi-line docs use a `<details>` element.
fn render_by_kind_entry(content: &mut String, name: &str, short_path: &str, doc: Option<&str>) {
    match doc {
        Some(d) if d.lines().count() > 1 => {
            let first_line = html_escape(d.lines().next().unwrap_or(""));
            let body: String = d.lines()
                .map(|line| format!("<p>{}</p>", html_escape(line)))
                .collect::<Vec<_>>()
                .join("\n");
            content.push_str(&format!(
                "<li><code>{}</code> ({})\n<details><summary>{}</summary>\n<div class=\"doc-comment\">\n{}\n</div>\n</details>\n</li>\n",
                html_escape(name), html_escape(short_path), first_line, body
            ));
        }
        Some(d) => {
            content.push_str(&format!(
                "<li><code>{}</code> ({}) — <span class=\"doc-comment-inline\">{}</span></li>\n",
                html_escape(name), html_escape(short_path), html_escape(d)
            ));
        }
        None => {
            content.push_str(&format!(
                "<li><code>{}</code> ({})</li>\n",
                html_escape(name), html_escape(short_path)
            ));
        }
    }
}

fn build_key_symbols_section(conn: &Connection, module_path: &str, query_engine: &QueryEngine) -> String {
    let pattern = format!("{}/%", module_path);
    let mut stmt = match conn.prepare(
        "SELECT path, language FROM files
         WHERE path LIKE ?1 AND language IS NOT NULL
         ORDER BY COALESCE(line_count, 0) DESC
         LIMIT 20"
    ) {
        Ok(s) => s,
        Err(_) => return "No symbols extracted.".to_string(),
    };

    let files: Vec<(String, String)> = match stmt.query_map([&pattern], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }) {
        Ok(rows) => rows.filter_map(|r| r.ok()).collect(),
        Err(_) => return "No symbols extracted.".to_string(),
    };

    if files.is_empty() {
        return "No files in this module.".to_string();
    }

    // Parse each file and collect symbols
    // kind -> [(name, path, size, doc_comment)]
    let mut by_kind: HashMap<String, Vec<(String, String, usize, Option<String>)>> = HashMap::new();
    let mut total_symbols = 0usize;

    for (path, lang_str) in &files {
        let language = match Language::from_name(lang_str) {
            Some(l) => l,
            None => continue,
        };

        // Read source from disk
        let source = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let symbols = match ParserFactory::parse(path, &source, language) {
            Ok(s) => s,
            Err(_) => continue,
        };

        for sym in symbols {
            if let Some(name) = &sym.symbol {
                // Skip imports, exports, and unknown kinds
                match &sym.kind {
                    SymbolKind::Import | SymbolKind::Export | SymbolKind::Variable | SymbolKind::Unknown(_) => continue,
                    _ => {}
                }

                let kind_name = format!("{}", sym.kind);
                let size = sym.span.end_line.saturating_sub(sym.span.start_line) + 1;
                let doc_comment = extract_doc_comment(&source, sym.span.start_line, &language);
                by_kind
                    .entry(kind_name)
                    .or_default()
                    .push((name.clone(), path.clone(), size, doc_comment));
                total_symbols += 1;
            }
        }
    }

    if total_symbols == 0 {
        return "No symbols extracted.".to_string();
    }

    let mut content = String::new();

    // Build doc_comments lookup: symbol name -> doc comment
    let mut doc_comments: HashMap<String, String> = HashMap::new();
    for entries in by_kind.values() {
        for (name, _path, _size, doc) in entries {
            if let Some(d) = doc {
                doc_comments.entry(name.clone()).or_insert_with(|| d.clone());
            }
        }
    }

    // --- Top symbols by codebase importance (above the fold) ---
    // Deduplicate symbol names, preferring priority kinds
    let mut unique_symbols: HashMap<String, (String, String)> = HashMap::new(); // name -> (kind, path)
    // First pass: insert priority-kind symbols
    for (kind_str, entries) in &by_kind {
        if PRIORITY_SYMBOL_KINDS.contains(&kind_str.as_str()) {
            for (name, path, _size, _doc) in entries {
                unique_symbols.entry(name.clone()).or_insert_with(|| (kind_str.clone(), path.clone()));
            }
        }
    }
    // Second pass: fill in remaining kinds (won't overwrite priority entries)
    for (kind_str, entries) in &by_kind {
        if !PRIORITY_SYMBOL_KINDS.contains(&kind_str.as_str()) {
            for (name, path, _size, _doc) in entries {
                unique_symbols.entry(name.clone()).or_insert_with(|| (kind_str.clone(), path.clone()));
            }
        }
    }

    // Count references for priority-kind symbols only via the trigram index.
    // Filter out blocklisted keywords and short names, cap at 15 candidates.
    let mut candidates: Vec<(String, String, String, usize)> = Vec::new(); // (name, kind, path, span_size)
    for (name, (kind, path)) in &unique_symbols {
        // Only query priority-kind symbols (functions, structs, traits, etc.)
        if !PRIORITY_SYMBOL_KINDS.contains(&kind.as_str()) {
            continue;
        }
        // Skip short names (< 4 chars) — they're too generic
        if name.len() < 4 {
            continue;
        }
        // Skip blocklisted keywords and common variable names
        if SYMBOL_BLOCKLIST.contains(&name.to_lowercase().as_str()) {
            continue;
        }
        // Skip names that start with $ (PHP variables like $data, $type)
        if name.starts_with('$') {
            let stripped = &name[1..];
            if stripped.len() < 4 || SYMBOL_BLOCKLIST.contains(&stripped.to_lowercase().as_str()) {
                continue;
            }
        }

        // Look up span size for this symbol (larger definitions are more important)
        let span_size = by_kind.get(kind)
            .and_then(|entries| entries.iter().find(|(n, _, _, _)| n == name))
            .map(|(_, _, size, _)| *size)
            .unwrap_or(1);

        candidates.push((name.clone(), kind.clone(), path.clone(), span_size));
    }

    // Sort by span size desc, cap at 15 before querying
    candidates.sort_by(|a, b| b.3.cmp(&a.3).then_with(|| a.0.cmp(&b.0)));
    candidates.truncate(15);

    // Query reference counts and file paths for the capped candidates
    let mut ranked: Vec<(String, String, String, usize)> = Vec::new(); // (name, kind, path, ref_count)
    let mut ref_files: HashMap<String, Vec<String>> = HashMap::new(); // symbol name -> referencing file short names
    for (name, kind, path, _span_size) in &candidates {
        let filter = QueryFilter {
            paths_only: true,
            force: true,
            suppress_output: true,
            limit: None,
            ..Default::default()
        };
        let def_short = path.rsplit('/').next().unwrap_or(path);
        match query_engine.search_with_metadata(name, filter) {
            Ok(response) => {
                let ref_count = response.results.len();
                // Collect unique short filenames, excluding the definition file
                let mut files: Vec<String> = response.results.iter()
                    .map(|r| r.path.rsplit('/').next().unwrap_or(&r.path).to_string())
                    .filter(|f| f != def_short)
                    .collect();
                files.sort();
                files.dedup();
                ref_files.insert(name.clone(), files);
                ranked.push((name.clone(), kind.clone(), path.clone(), ref_count));
            }
            Err(_) => {
                ranked.push((name.clone(), kind.clone(), path.clone(), 0));
            }
        }
    }

    // Sort by reference count desc
    ranked.sort_by(|a, b| b.3.cmp(&a.3).then_with(|| a.0.cmp(&b.0)));

    if !ranked.is_empty() {
        content.push_str("<p><strong>Key definitions:</strong></p>\n<ul>\n");
        for (name, kind, path, ref_count) in ranked.iter().take(5) {
            let short = path.rsplit('/').next().unwrap_or(path);
            content.push_str("<li>\n");
            content.push_str(&format!(
                "<p><code>{}</code> ({}) in {} — referenced in {} {}</p>\n",
                html_escape(name), html_escape(kind), html_escape(short), ref_count,
                if *ref_count == 1 { "file" } else { "files" }
            ));

            // Add doc comment if available
            if let Some(doc) = doc_comments.get(name.as_str()) {
                let first_line = html_escape(doc.lines().next().unwrap_or(""));
                let is_multiline = doc.lines().count() > 1;
                if is_multiline {
                    let body: String = doc.lines()
                        .map(|line| format!("<p>{}</p>", html_escape(line)))
                        .collect::<Vec<_>>()
                        .join("\n");
                    content.push_str(&format!(
                        "<details><summary>{}</summary>\n<div class=\"doc-comment\">\n{}\n</div>\n</details>\n",
                        first_line, body
                    ));
                } else {
                    content.push_str(&format!(
                        "<details><summary>{}</summary></details>\n",
                        first_line
                    ));
                }
            }

            // Add reference file list (top 5 + overflow)
            if let Some(files) = ref_files.get(name.as_str()) {
                if !files.is_empty() {
                    let show: Vec<&str> = files.iter().take(5).map(|s| s.as_str()).collect();
                    let mut ref_line = format!("<ul><li class=\"ref-list\">Referenced by: {}", show.join(", "));
                    if files.len() > 5 {
                        ref_line.push_str(&format!(" +{} more", files.len() - 5));
                    }
                    ref_line.push_str("</li></ul>\n");
                    content.push_str(&ref_line);
                }
            }

            content.push_str("</li>\n");
        }
        content.push_str("</ul>\n\n");
    }

    // --- By Kind view (collapsible, showing ALL symbols) ---
    let display_order = [
        "Function", "Struct", "Class", "Trait", "Interface",
        "Enum", "Method", "Constant", "Type", "Macro",
        "Variable", "Module", "Namespace", "Property", "Attribute",
    ];

    for kind in &display_order {
        let kind_str = kind.to_string();
        if let Some(entries) = by_kind.get_mut(&kind_str) {
            entries.sort_by(|a, b| b.2.cmp(&a.2));
            let count = entries.len();
            content.push_str(&format!("<details><summary><strong>{}</strong> ({})</summary>\n<ul>\n", kind, count));
            for (name, path, _size, doc) in entries.iter() {
                let short = path.rsplit('/').next().unwrap_or(path);
                render_by_kind_entry(&mut content, name, short, doc.as_deref());
            }
            content.push_str("</ul>\n</details>\n\n");
        }
    }

    // Handle any kinds not in display_order
    for (kind, entries) in &mut by_kind {
        if display_order.contains(&kind.as_str()) {
            continue;
        }
        entries.sort_by(|a, b| b.2.cmp(&a.2));
        let count = entries.len();
        content.push_str(&format!("<details><summary><strong>{}</strong> ({})</summary>\n<ul>\n", kind, count));
        for (name, path, _size, doc) in entries.iter() {
            let short = path.rsplit('/').next().unwrap_or(path);
            render_by_kind_entry(&mut content, name, short, doc.as_deref());
        }
        content.push_str("</ul>\n</details>\n\n");
    }

    if content.is_empty() {
        "No symbols extracted.".to_string()
    } else {
        content
    }
}

fn build_metrics_section(module: &ModuleDefinition, conn: &Connection) -> Result<String> {
    let pattern = format!("{}/%", module.path);

    // Average lines per file
    let avg_lines = if module.file_count > 0 {
        module.total_lines / module.file_count
    } else {
        0
    };

    // Outgoing dependency count
    let outgoing: usize = conn.query_row(
        "SELECT COUNT(DISTINCT fd.resolved_file_id)
         FROM file_dependencies fd
         JOIN files f1 ON fd.file_id = f1.id
         JOIN files f2 ON fd.resolved_file_id = f2.id
         WHERE f1.path LIKE ?1 AND f2.path NOT LIKE ?1",
        [&pattern],
        |row| row.get(0),
    ).unwrap_or(0);

    // Incoming dependency count
    let incoming: usize = conn.query_row(
        "SELECT COUNT(DISTINCT fd.file_id)
         FROM file_dependencies fd
         JOIN files f1 ON fd.file_id = f1.id
         JOIN files f2 ON fd.resolved_file_id = f2.id
         WHERE f2.path LIKE ?1 AND f1.path NOT LIKE ?1",
        [&pattern],
        |row| row.get(0),
    ).unwrap_or(0);

    Ok(format!(
        "| Metric | Value |\n|---|---|\n\
         | Files | {} |\n\
         | Total lines | {} |\n\
         | Avg lines/file | {} |\n\
         | Languages | {} |\n\
         | Outgoing deps | {} |\n\
         | Incoming deps | {} |\n\
         | Tier | {} |",
        module.file_count,
        module.total_lines,
        avg_lines,
        module.languages.join(", "),
        outgoing,
        incoming,
        module.tier,
    ))
}

/// Find the most-specific module that owns a given file path
fn find_owning_module(file_path: &str, modules: &[ModuleDefinition]) -> String {
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
        // Fall back to top-level directory
        file_path.split('/').next().unwrap_or("root").to_string()
    } else {
        best_match
    }
}

fn build_recent_changes(diff: &super::diff::SnapshotDiff, module_path: &str) -> String {
    let prefix = format!("{}/", module_path);
    let mut content = String::new();

    let added: Vec<_> = diff.files_added.iter()
        .filter(|f| f.path.starts_with(&prefix))
        .collect();
    let removed: Vec<_> = diff.files_removed.iter()
        .filter(|f| f.path.starts_with(&prefix))
        .collect();
    let modified: Vec<_> = diff.files_modified.iter()
        .filter(|f| f.path.starts_with(&prefix))
        .collect();

    if added.is_empty() && removed.is_empty() && modified.is_empty() {
        return "No changes in this module since last snapshot.".to_string();
    }

    if !added.is_empty() {
        content.push_str(&format!("**Added** ({}):\n", added.len()));
        for f in added.iter().take(10) {
            content.push_str(&format!("- `{}`\n", f.path));
        }
    }
    if !removed.is_empty() {
        content.push_str(&format!("**Removed** ({}):\n", removed.len()));
        for f in removed.iter().take(10) {
            content.push_str(&format!("- `{}`\n", f.path));
        }
    }
    if !modified.is_empty() {
        content.push_str(&format!("**Modified** ({}):\n", modified.len()));
        for f in modified.iter().take(10) {
            let delta = f.new_line_count as i64 - f.old_line_count as i64;
            content.push_str(&format!("- `{}` ({:+} lines)\n", f.path, delta));
        }
    }

    content
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_definition_serialization() {
        let module = ModuleDefinition {
            path: "src".to_string(),
            tier: 1,
            file_count: 50,
            total_lines: 5000,
            languages: vec!["Rust".to_string()],
        };
        let json = serde_json::to_string(&module).unwrap();
        assert!(json.contains("src"));
    }

    #[test]
    fn test_render_wiki_page() {
        let page = WikiPage {
            module_path: "src".to_string(),
            title: "src/".to_string(),
            sections: WikiSections {
                summary: None,
                structure: "test structure".to_string(),
                dependencies: "test deps".to_string(),
                dependents: "test dependents".to_string(),
                dependency_diagram: None,
                circular_deps: None,
                key_symbols: "test symbols".to_string(),
                metrics: "test metrics".to_string(),
                recent_changes: None,
            },
        };
        let rendered = render_wiki_markdown(&[page]);
        assert_eq!(rendered.len(), 1);
        assert_eq!(rendered[0].0, "src.md");
        assert!(rendered[0].1.contains("# src/"));
    }
}
