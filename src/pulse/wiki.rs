//! Wiki generation: per-module documentation pages
//!
//! Generates a living wiki page for each detected module (directory) in the codebase.
//! Pages include structural sections (dependencies, dependents, key symbols, metrics)
//! and optional LLM-generated summaries.

use anyhow::{Context, Result};
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
    pub key_symbols: String,
    pub metrics: String,
    pub recent_changes: Option<String>,
}

/// Detect modules in the codebase using CodebaseContext
///
/// Returns both top-level directories (Tier 1) and their immediate
/// sub-directories with 3+ files (Tier 2). This produces granular modules
/// like `src/parsers`, `src/semantic`, `src/pulse` instead of just `src`.
pub fn detect_modules(cache: &CacheManager) -> Result<Vec<ModuleDefinition>> {
    let context = CodebaseContext::extract(cache)
        .context("Failed to extract codebase context")?;

    let db_path = cache.path().join("meta.db");
    let conn = Connection::open(&db_path)?;

    let mut modules = Vec::new();

    // Tier 1: top-level directories
    for dir in &context.top_level_dirs {
        let dir_path = dir.trim_end_matches('/');
        if let Some(module) = build_module_def(&conn, dir_path, 1)? {
            modules.push(module);
        }
    }

    // Tier 2: discover sub-modules under each Tier 1 module
    let tier1_paths: Vec<String> = modules.iter().map(|m| m.path.clone()).collect();
    for parent in &tier1_paths {
        let sub_modules = discover_sub_modules(&conn, parent)?;
        for sub_path in sub_modules {
            // Skip exact duplicates
            if modules.iter().any(|m| m.path == sub_path) {
                continue;
            }
            if let Some(module) = build_module_def(&conn, &sub_path, 2)? {
                modules.push(module);
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
            modules.push(module);
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
) -> Result<Vec<WikiPage>> {
    let modules = detect_modules(cache)?;
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

    // Sanitize node IDs for mermaid (replace / with _)
    let center_id = module_path.replace('/', "_");
    diagram.push_str(&format!("    {}[\"<b>{}/</b>\"]\n", center_id, module_path));
    diagram.push_str(&format!("    style {} fill:#7aa2f7,color:#1a1b26,stroke:#7aa2f7\n", center_id));

    // Track all nodes for clickable links
    let mut all_node_paths: Vec<String> = vec![module_path.to_string()];

    // Outgoing edges (this module depends on)
    let mut out_sorted: Vec<_> = outgoing.into_iter().collect();
    out_sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (target, count) in out_sorted.iter().take(8) {
        let target_id = target.replace('/', "_");
        diagram.push_str(&format!("    {}[\"{}/\"]\n", target_id, target));
        diagram.push_str(&format!("    {} -->|{}| {}\n", center_id, count, target_id));
        all_node_paths.push(target.clone());
    }

    // Incoming edges (modules that depend on this)
    let mut in_sorted: Vec<_> = incoming.into_iter().collect();
    in_sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (source, count) in in_sorted.iter().take(8) {
        let source_id = source.replace('/', "_");
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
        let node_id = node_path.replace('/', "_");
        let slug = node_path.replace('/', "-");
        diagram.push_str(&format!("    click {} \"/wiki/{}/\"\n", node_id, slug));
    }

    Some(diagram)
}

fn build_module_def(conn: &Connection, path: &str, tier: u8) -> Result<Option<ModuleDefinition>> {
    let pattern = format!("{}/%", path);

    let file_count: usize = conn.query_row(
        "SELECT COUNT(*) FROM files WHERE path LIKE ?1 OR path LIKE ?2",
        rusqlite::params![format!("{}/%", path), format!("{}", path)],
        |row| row.get(0),
    )?;

    if file_count == 0 {
        return Ok(None);
    }

    let total_lines: usize = conn.query_row(
        "SELECT COALESCE(SUM(line_count), 0) FROM files WHERE path LIKE ?1",
        [&pattern],
        |row| row.get(0),
    )?;

    let mut stmt = conn.prepare(
        "SELECT DISTINCT language FROM files WHERE path LIKE ?1 AND language IS NOT NULL"
    )?;
    let languages: Vec<String> = stmt.query_map([&pattern], |row| row.get(0))?
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
                "- [**{}//**](/wiki/{}/) — {} files, {} lines ({})\n",
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
    let mut by_kind: HashMap<String, Vec<(String, String, usize)>> = HashMap::new(); // kind -> [(name, path, size)]
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
                    SymbolKind::Import | SymbolKind::Export | SymbolKind::Unknown(_) => continue,
                    _ => {}
                }

                let kind_name = format!("{}", sym.kind);
                let size = sym.span.end_line.saturating_sub(sym.span.start_line) + 1;
                by_kind
                    .entry(kind_name)
                    .or_default()
                    .push((name.clone(), path.clone(), size));
                total_symbols += 1;
            }
        }
    }

    if total_symbols == 0 {
        return "No symbols extracted.".to_string();
    }

    let mut content = String::new();

    // --- Top symbols above the fold ---
    // Rank by reference count: how many files in the codebase use each symbol
    // Deduplicate symbol names (same name may appear in multiple files within a module)
    let mut unique_symbols: HashMap<String, (String, String)> = HashMap::new(); // name -> (kind, path)
    for (kind_str, entries) in &by_kind {
        for (name, path, _size) in entries {
            unique_symbols.entry(name.clone()).or_insert_with(|| (kind_str.clone(), path.clone()));
        }
    }

    // Count references for each unique symbol via the trigram index
    let mut ranked: Vec<(String, String, String, usize)> = Vec::new(); // (name, kind, path, ref_count)
    for (name, (kind, path)) in &unique_symbols {
        // Trigram index requires 3+ character patterns
        if name.len() < 3 {
            continue;
        }
        let filter = QueryFilter {
            paths_only: true,
            force: true,
            suppress_output: true,
            limit: None,
            ..Default::default()
        };
        let ref_count = match query_engine.search_with_metadata(name, filter) {
            Ok(response) => response.results.len(),
            Err(_) => 0,
        };
        ranked.push((name.clone(), kind.clone(), path.clone(), ref_count));
    }
    ranked.sort_by(|a, b| b.3.cmp(&a.3).then_with(|| a.0.cmp(&b.0)));

    if !ranked.is_empty() {
        content.push_str("**Key definitions:**\n");
        for (name, kind, path, ref_count) in ranked.iter().take(5) {
            let short = path.rsplit('/').next().unwrap_or(path);
            content.push_str(&format!(
                "- `{}` ({}) in {} — referenced in {} {}\n",
                name, kind, short, ref_count,
                if *ref_count == 1 { "file" } else { "files" }
            ));
        }
        content.push('\n');
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
            content.push_str(&format!("<details><summary><strong>{}</strong> ({})</summary>\n\n", kind, count));
            for (name, path, _) in entries.iter() {
                let short = path.rsplit('/').next().unwrap_or(path);
                content.push_str(&format!("- `{}` ({})\n", name, short));
            }
            content.push_str("\n</details>\n\n");
        }
    }

    // Handle any kinds not in display_order
    for (kind, entries) in &mut by_kind {
        if display_order.contains(&kind.as_str()) {
            continue;
        }
        entries.sort_by(|a, b| b.2.cmp(&a.2));
        let count = entries.len();
        content.push_str(&format!("<details><summary><strong>{}</strong> ({})</summary>\n\n", kind, count));
        for (name, path, _) in entries.iter() {
            let short = path.rsplit('/').next().unwrap_or(path);
            content.push_str(&format!("- `{}` ({})\n", name, short));
        }
        content.push_str("\n</details>\n\n");
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
