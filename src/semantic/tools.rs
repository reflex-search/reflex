//! Tool execution system for agentic context gathering
//!
//! This module handles execution of tool calls from the LLM including:
//! - Running `rfx context` commands
//! - Executing exploratory queries
//! - Running codebase analysis (hotspots, unused files, etc.)

use crate::cache::CacheManager;
use crate::dependency::DependencyIndex;
use crate::query::QueryEngine;
use anyhow::{Context as AnyhowContext, Result};

use super::executor::parse_command;
use super::schema_agentic::{AnalysisType, ContextGatheringParams, ToolCall};

/// Result of executing a tool call
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// Description of what this tool did
    pub description: String,

    /// The output/result from the tool
    pub output: String,

    /// Whether the tool execution was successful
    pub success: bool,
}

/// Execute a single tool call
pub async fn execute_tool(tool: &ToolCall, cache: &CacheManager) -> Result<ToolResult> {
    match tool {
        ToolCall::GatherContext { params } => execute_gather_context(params, cache),
        ToolCall::ExploreCodebase {
            description,
            command,
        } => execute_explore_codebase(description, command, cache).await,
        ToolCall::AnalyzeStructure { analysis_type } => {
            execute_analyze_structure(*analysis_type, cache)
        }
        ToolCall::SearchDocumentation { query, files } => {
            execute_search_documentation(query, files.as_deref(), cache)
        }
        ToolCall::GetStatistics => execute_get_statistics(cache),
        ToolCall::GetDependencies { file_path, reverse } => {
            execute_get_dependencies(file_path, *reverse, cache)
        }
        ToolCall::GetAnalysisSummary { min_dependents } => {
            execute_get_analysis_summary(*min_dependents, cache)
        }
        ToolCall::FindIslands { min_size, max_size } => {
            execute_find_islands(*min_size, *max_size, cache)
        }
    }
}

/// Execute context gathering tool
fn execute_gather_context(
    params: &ContextGatheringParams,
    cache: &CacheManager,
) -> Result<ToolResult> {
    log::info!("Executing gather_context tool");

    // Build context options from params
    let mut opts = crate::context::ContextOptions {
        structure: params.structure,
        path: params.path.clone(),
        file_types: params.file_types,
        project_type: params.project_type,
        framework: params.framework,
        entry_points: params.entry_points,
        test_layout: params.test_layout,
        config_files: params.config_files,
        depth: params.depth,
        json: false, // Always use text format for LLM consumption
    };

    // If no specific flags, enable all context types by default
    if opts.is_empty() {
        opts.structure = true;
        opts.file_types = true;
        opts.project_type = true;
        opts.framework = true;
        opts.entry_points = true;
        opts.test_layout = true;
        opts.config_files = true;
    }

    // Generate context
    let output = crate::context::generate_context(cache, &opts)
        .context("Failed to generate codebase context")?;

    // Build description of what was gathered
    let mut parts = Vec::new();
    if opts.structure {
        parts.push("structure");
    }
    if opts.file_types {
        parts.push("file types");
    }
    if opts.project_type {
        parts.push("project type");
    }
    if opts.framework {
        parts.push("frameworks");
    }
    if opts.entry_points {
        parts.push("entry points");
    }
    if opts.test_layout {
        parts.push("test layout");
    }
    if opts.config_files {
        parts.push("config files");
    }

    let description = if parts.is_empty() {
        "Gathered general codebase context".to_string()
    } else {
        format!("Gathered codebase context: {}", parts.join(", "))
    };

    log::debug!("Context gathering successful: {} chars", output.len());

    Ok(ToolResult {
        description,
        output,
        success: true,
    })
}

/// Execute exploratory codebase query
async fn execute_explore_codebase(
    description: &str,
    command: &str,
    cache: &CacheManager,
) -> Result<ToolResult> {
    log::info!("Executing explore_codebase tool: {}", description);

    // Parse the command
    let parsed = parse_command(command)
        .with_context(|| format!("Failed to parse exploration command: {}", command))?;

    // Convert to QueryFilter
    let filter = parsed.to_query_filter()?;

    // Create query engine
    let engine = QueryEngine::new(CacheManager::new(cache.workspace_root()));

    // Execute query
    let response = engine
        .search_with_metadata(&parsed.pattern, filter)
        .with_context(|| format!("Failed to execute exploration query: {}", command))?;

    // Format results for LLM consumption
    let output = format_exploration_results(&response, &parsed.pattern);

    log::debug!(
        "Exploration query found {} file groups",
        response.results.len()
    );

    Ok(ToolResult {
        description: format!("Explored: {}", description),
        output,
        success: true,
    })
}

/// Execute structure analysis (hotspots, unused files, etc.)
fn execute_analyze_structure(
    analysis_type: AnalysisType,
    cache: &CacheManager,
) -> Result<ToolResult> {
    log::info!("Executing analyze_structure tool: {:?}", analysis_type);

    // Create dependency index
    let deps_index = DependencyIndex::new(CacheManager::new(cache.workspace_root()));

    let output = match analysis_type {
        AnalysisType::Hotspots => {
            // Get hotspots (returns file IDs and counts)
            let hotspot_ids = deps_index.find_hotspots(Some(10), 2)?; // top 10, min 2 dependents

            // Convert file IDs to paths
            let file_ids: Vec<i64> = hotspot_ids.iter().map(|(id, _)| *id).collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            // Convert to (String, usize) format
            let hotspots: Vec<(String, usize)> = hotspot_ids
                .iter()
                .filter_map(|(id, count)| paths.get(id).map(|path| (path.clone(), *count)))
                .collect();

            format_hotspots(&hotspots)
        }
        AnalysisType::Unused => {
            // Get unused files (returns file IDs)
            let unused_ids = deps_index.find_unused_files()?;

            // Convert file IDs to paths
            let paths = deps_index.get_file_paths(&unused_ids)?;
            let unused: Vec<String> = unused_ids
                .iter()
                .filter_map(|id| paths.get(id).cloned())
                .collect();

            format_unused_files(&unused)
        }
        AnalysisType::Circular => {
            // Get circular dependencies (returns vectors of file IDs)
            let circular_ids = deps_index.detect_circular_dependencies()?;

            // Collect all unique file IDs
            let all_ids: Vec<i64> = circular_ids
                .iter()
                .flat_map(|cycle| cycle.iter())
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

            // Convert all IDs to paths
            let paths = deps_index.get_file_paths(&all_ids)?;

            // Convert cycles to path cycles
            let circular: Vec<Vec<String>> = circular_ids
                .iter()
                .map(|cycle| {
                    cycle
                        .iter()
                        .filter_map(|id| paths.get(id).cloned())
                        .collect()
                })
                .collect();

            format_circular_deps(&circular)
        }
    };

    let description = match analysis_type {
        AnalysisType::Hotspots => "Analyzed dependency hotspots (most-imported files)",
        AnalysisType::Unused => "Analyzed unused files (no importers)",
        AnalysisType::Circular => "Analyzed circular dependencies",
    };

    log::debug!("Analysis complete: {} chars", output.len());

    Ok(ToolResult {
        description: description.to_string(),
        output,
        success: true,
    })
}

/// Execute documentation search tool
fn execute_search_documentation(
    query: &str,
    files: Option<&[String]>,
    cache: &CacheManager,
) -> Result<ToolResult> {
    log::info!("Executing search_documentation tool: query='{}'", query);

    let workspace_root = cache.workspace_root();

    // Default documentation files to search
    let default_files = vec!["CLAUDE.md".to_string(), "README.md".to_string()];
    let search_files = files.unwrap_or(&default_files);

    let mut found_sections = Vec::new();
    let mut searched_files = Vec::new();

    // Search specified documentation files
    for file in search_files {
        let file_path = workspace_root.join(file);

        if !file_path.exists() {
            log::debug!("Documentation file does not exist: {}", file);
            continue;
        }

        searched_files.push(file.clone());

        match std::fs::read_to_string(&file_path) {
            Ok(content) => {
                // Search for query keywords in the content
                if let Some(sections) = search_documentation_content(&content, query, file) {
                    found_sections.push(sections);
                }
            }
            Err(e) => {
                log::warn!("Failed to read documentation file {}: {}", file, e);
            }
        }
    }

    // Also search .context/ directory for markdown files
    let context_dir = workspace_root.join(".context");
    if context_dir.exists() && context_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&context_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("md") {
                    if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                        if let Ok(content) = std::fs::read_to_string(&path) {
                            if let Some(sections) = search_documentation_content(
                                &content,
                                query,
                                &format!(".context/{}", file_name),
                            ) {
                                found_sections.push(sections);
                                searched_files.push(format!(".context/{}", file_name));
                            }
                        }
                    }
                }
            }
        }
    }

    // Format output
    let output = if found_sections.is_empty() {
        format!(
            "No relevant documentation found for query '{}' in files: {}\n\nTry:\n- Using different keywords\n- Searching the codebase directly with explore_codebase",
            query,
            searched_files.join(", ")
        )
    } else {
        format!(
            "Found documentation for '{}' in {} file(s):\n\n{}",
            query,
            found_sections.len(),
            found_sections.join("\n\n---\n\n")
        )
    };

    log::debug!(
        "Documentation search found {} sections",
        found_sections.len()
    );

    Ok(ToolResult {
        description: format!("Searched documentation for: {}", query),
        output,
        success: !found_sections.is_empty(),
    })
}

/// Search documentation content for query and extract relevant sections
fn search_documentation_content(content: &str, query: &str, file_name: &str) -> Option<String> {
    // Tokenize query into keywords (filter out common stop words)
    let stop_words = [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "should", "could", "may", "might", "can", "what", "how",
        "where", "when", "why", "which", "who",
    ];
    let keywords: Vec<String> = query
        .to_lowercase()
        .split_whitespace()
        .filter(|word| !stop_words.contains(word) && word.len() > 2)
        .map(|s| s.to_string())
        .collect();

    if keywords.is_empty() {
        return None;
    }

    let lines: Vec<&str> = content.lines().collect();
    let mut relevant_sections = Vec::new();
    let mut current_section = String::new();
    let mut current_section_title = String::new();
    let mut in_relevant_section = false;
    let mut relevance_score = 0;

    for line in lines.iter() {
        let line_lower = line.to_lowercase();

        // Check if this is a heading
        if line.starts_with('#') {
            // Save previous section if it was relevant
            if in_relevant_section && relevance_score >= 2 {
                // Need at least 2 keyword matches
                relevant_sections.push(format!(
                    "## {} ({})\n\n{}",
                    current_section_title,
                    file_name,
                    current_section.trim()
                ));
            }

            // Start new section
            current_section.clear();
            current_section_title = line.trim_start_matches('#').trim().to_string();
            relevance_score = 0;
            in_relevant_section = false;

            // Check if heading contains any query keywords
            let heading_lower = line_lower.clone();
            for keyword in &keywords {
                if heading_lower.contains(keyword) {
                    in_relevant_section = true;
                    relevance_score += 10;
                }
            }
        }

        // Check if content contains any query keywords
        let mut line_matches = 0;
        for keyword in &keywords {
            if line_lower.contains(keyword) {
                in_relevant_section = true;
                line_matches += 1;
            }
        }
        relevance_score += line_matches;

        // Add line to current section (with some context)
        if in_relevant_section || relevance_score > 0 {
            current_section.push_str(line);
            current_section.push('\n');

            // Limit section size to prevent massive outputs
            if current_section.lines().count() > 150 {
                break;
            }
        }
    }

    // Save last section if relevant
    if in_relevant_section && relevance_score >= 2 {
        // Need at least 2 keyword matches
        relevant_sections.push(format!(
            "## {} ({})\n\n{}",
            current_section_title,
            file_name,
            current_section.trim()
        ));
    }

    if relevant_sections.is_empty() {
        None
    } else {
        // Sort sections by relevance (most matches first) and limit to top 3
        Some(
            relevant_sections
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n\n"),
        )
    }
}

/// Format exploration query results for LLM
fn format_exploration_results(response: &crate::models::QueryResponse, pattern: &str) -> String {
    if response.results.is_empty() {
        return format!("No results found for pattern: {}", pattern);
    }

    let mut output = Vec::new();
    output.push(format!(
        "Found {} total matches across {} files for pattern '{}':\n",
        response.pagination.total,
        response.results.len(),
        pattern
    ));

    // Show first 5 file groups
    for (idx, file_group) in response.results.iter().take(5).enumerate() {
        output.push(format!("\n{}. {}", idx + 1, file_group.path));

        // Show first 3 matches per file
        for match_result in file_group.matches.iter().take(3) {
            // Show context before the match
            for (idx, line) in match_result.context_before.iter().enumerate() {
                let line_num = match_result
                    .span
                    .start_line
                    .saturating_sub(match_result.context_before.len() - idx);
                output.push(format!("   Line {}: {}", line_num, line.trim()));
            }

            // Show the match line itself
            output.push(format!(
                "   Line {}: {}",
                match_result.span.start_line,
                match_result.preview.lines().next().unwrap_or("").trim()
            ));

            // Show context after the match
            for (idx, line) in match_result.context_after.iter().enumerate() {
                let line_num = match_result.span.start_line + idx + 1;
                output.push(format!("   Line {}: {}", line_num, line.trim()));
            }
        }

        if file_group.matches.len() > 3 {
            output.push(format!(
                "   ... and {} more matches",
                file_group.matches.len() - 3
            ));
        }
    }

    if response.results.len() > 5 {
        output.push(format!(
            "\n... and {} more files",
            response.results.len() - 5
        ));
    }

    output.join("\n")
}

/// Format hotspot analysis results
fn format_hotspots(hotspots: &[(String, usize)]) -> String {
    if hotspots.is_empty() {
        return "No dependency hotspots found.".to_string();
    }

    let mut output = Vec::new();
    output.push(format!(
        "Top {} most-imported files:\n",
        hotspots.len().min(10)
    ));

    for (idx, (path, count)) in hotspots.iter().take(10).enumerate() {
        output.push(format!("{}. {} ({} importers)", idx + 1, path, count));
    }

    if hotspots.len() > 10 {
        output.push(format!("\n... and {} more hotspots", hotspots.len() - 10));
    }

    output.join("\n")
}

/// Format unused files analysis results
fn format_unused_files(unused: &[String]) -> String {
    if unused.is_empty() {
        return "No unused files found (all files are imported by others).".to_string();
    }

    let mut output = Vec::new();
    output.push(format!(
        "Found {} unused files (no importers):\n",
        unused.len()
    ));

    for (idx, path) in unused.iter().take(15).enumerate() {
        output.push(format!("{}. {}", idx + 1, path));
    }

    if unused.len() > 15 {
        output.push(format!("\n... and {} more unused files", unused.len() - 15));
    }

    output.join("\n")
}

/// Format circular dependency analysis results
fn format_circular_deps(circular: &[Vec<String>]) -> String {
    if circular.is_empty() {
        return "No circular dependencies found.".to_string();
    }

    let mut output = Vec::new();
    output.push(format!(
        "Found {} circular dependency chains:\n",
        circular.len()
    ));

    for (idx, cycle) in circular.iter().take(5).enumerate() {
        output.push(format!("\n{}. Cycle ({} files):", idx + 1, cycle.len()));
        output.push(format!("   {}", cycle.join(" → ")));
    }

    if circular.len() > 5 {
        output.push(format!(
            "\n... and {} more circular dependencies",
            circular.len() - 5
        ));
    }

    output.join("\n")
}

/// Execute get statistics tool
fn execute_get_statistics(cache: &CacheManager) -> Result<ToolResult> {
    log::info!("Executing get_statistics tool");

    // Get index statistics
    let stats = cache.stats().context("Failed to get cache statistics")?;

    // Format output
    let output = format_statistics(&stats);

    log::debug!("Statistics retrieved successfully");

    Ok(ToolResult {
        description: "Retrieved index statistics".to_string(),
        output,
        success: true,
    })
}

/// Execute get dependencies tool
fn execute_get_dependencies(
    file_path: &str,
    reverse: bool,
    cache: &CacheManager,
) -> Result<ToolResult> {
    log::info!(
        "Executing get_dependencies tool: file={}, reverse={}",
        file_path,
        reverse
    );

    // Create dependency index
    let deps_index = DependencyIndex::new(CacheManager::new(cache.workspace_root()));

    // Get file ID by path (supports fuzzy matching)
    let file_id = deps_index
        .get_file_id_by_path(file_path)
        .context(format!("Failed to find file: {}", file_path))?
        .ok_or_else(|| anyhow::anyhow!("File not found: {}", file_path))?;

    let output = if reverse {
        // Get files that depend on this file (reverse dependencies)
        let dependent_ids = deps_index
            .get_dependents(file_id)
            .context("Failed to get reverse dependencies")?;

        // Convert file IDs to paths
        let paths = deps_index.get_file_paths(&dependent_ids)?;
        let dependents: Vec<String> = dependent_ids
            .iter()
            .filter_map(|id| paths.get(id).cloned())
            .collect();

        format_reverse_dependencies(file_path, &dependents)
    } else {
        // Get dependencies of this file
        let deps = deps_index
            .get_dependencies_info(file_id)
            .context("Failed to get dependencies")?;

        format_dependencies(file_path, &deps)
    };

    let description = if reverse {
        format!("Found reverse dependencies for: {}", file_path)
    } else {
        format!("Found dependencies for: {}", file_path)
    };

    log::debug!("Dependencies retrieved successfully");

    Ok(ToolResult {
        description,
        output,
        success: true,
    })
}

/// Execute get analysis summary tool
fn execute_get_analysis_summary(min_dependents: usize, cache: &CacheManager) -> Result<ToolResult> {
    log::info!(
        "Executing get_analysis_summary tool: min_dependents={}",
        min_dependents
    );

    // Create dependency index
    let deps_index = DependencyIndex::new(CacheManager::new(cache.workspace_root()));

    // Get hotspots
    let hotspot_ids = deps_index.find_hotspots(Some(10), min_dependents)?;
    let hotspot_count = hotspot_ids.len();

    // Get unused files count
    let unused_ids = deps_index.find_unused_files()?;
    let unused_count = unused_ids.len();

    // Get circular dependencies count
    let circular_ids = deps_index.detect_circular_dependencies()?;
    let circular_count = circular_ids.len();

    // Format summary
    let output =
        format_analysis_summary(hotspot_count, unused_count, circular_count, min_dependents);

    log::debug!("Analysis summary retrieved successfully");

    Ok(ToolResult {
        description: "Retrieved dependency analysis summary".to_string(),
        output,
        success: true,
    })
}

/// Execute find islands tool
fn execute_find_islands(
    min_size: usize,
    max_size: usize,
    cache: &CacheManager,
) -> Result<ToolResult> {
    log::info!(
        "Executing find_islands tool: min_size={}, max_size={}",
        min_size,
        max_size
    );

    // Create dependency index
    let deps_index = DependencyIndex::new(CacheManager::new(cache.workspace_root()));

    // Get all islands
    let all_islands = deps_index.find_islands()?;

    // Filter by size
    let filtered_islands: Vec<Vec<i64>> = all_islands
        .into_iter()
        .filter(|island| island.len() >= min_size && island.len() <= max_size)
        .collect();

    // Convert file IDs to paths
    let all_ids: Vec<i64> = filtered_islands
        .iter()
        .flat_map(|island| island.iter())
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let paths = deps_index.get_file_paths(&all_ids)?;

    let islands_with_paths: Vec<Vec<String>> = filtered_islands
        .iter()
        .map(|island| {
            island
                .iter()
                .filter_map(|id| paths.get(id).cloned())
                .collect()
        })
        .collect();

    // Format output
    let output = format_islands(&islands_with_paths, min_size, max_size);

    log::debug!(
        "Islands retrieved successfully: {} islands found",
        islands_with_paths.len()
    );

    Ok(ToolResult {
        description: format!("Found {} disconnected components", islands_with_paths.len()),
        output,
        success: true,
    })
}

/// Format statistics output
fn format_statistics(stats: &crate::models::IndexStats) -> String {
    let mut output = Vec::new();

    output.push(format!("# Index Statistics\n"));
    output.push(format!("Total files: {}", stats.total_files));
    output.push(format!(
        "Index size: {:.2} MB\n",
        stats.index_size_bytes as f64 / 1_048_576.0
    ));

    // Files by language
    if !stats.files_by_language.is_empty() {
        output.push("## Files by Language\n".to_string());
        let mut lang_counts: Vec<_> = stats.files_by_language.iter().collect();
        lang_counts.sort_by(|a, b| b.1.cmp(a.1)); // Sort by count descending

        for (lang, count) in lang_counts.iter().take(10) {
            let percentage = (**count as f64 / stats.total_files as f64) * 100.0;
            output.push(format!("- {}: {} files ({:.1}%)", lang, count, percentage));
        }

        if lang_counts.len() > 10 {
            output.push(format!("... and {} more languages", lang_counts.len() - 10));
        }
    }

    // Lines by language
    if !stats.lines_by_language.is_empty() {
        output.push("\n## Lines of Code by Language\n".to_string());
        let mut line_counts: Vec<_> = stats.lines_by_language.iter().collect();
        line_counts.sort_by(|a, b| b.1.cmp(a.1)); // Sort by count descending

        let total_lines: usize = stats.lines_by_language.values().sum();

        for (lang, count) in line_counts.iter().take(10) {
            let percentage = (**count as f64 / total_lines as f64) * 100.0;
            let formatted_count = count
                .to_string()
                .as_str()
                .chars()
                .rev()
                .enumerate()
                .map(|(i, c)| {
                    if i != 0 && i % 3 == 0 {
                        format!(",{}", c)
                    } else {
                        c.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<String>();
            output.push(format!(
                "- {}: {} lines ({:.1}%)",
                lang, formatted_count, percentage
            ));
        }

        if line_counts.len() > 10 {
            output.push(format!("... and {} more languages", line_counts.len() - 10));
        }
    }

    output.push(format!("\nLast updated: {}", stats.last_updated));

    output.join("\n")
}

/// Format dependencies output
fn format_dependencies(file_path: &str, deps: &[crate::models::DependencyInfo]) -> String {
    if deps.is_empty() {
        return format!("File '{}' has no dependencies.", file_path);
    }

    let mut output = Vec::new();
    output.push(format!("# Dependencies of '{}'\n", file_path));
    output.push(format!("Found {} dependencies:\n", deps.len()));

    for (idx, dep) in deps.iter().take(20).enumerate() {
        let line_info = dep
            .line
            .map(|l| format!(" (line {})", l))
            .unwrap_or_default();
        output.push(format!("{}. {}{}", idx + 1, dep.path, line_info));

        // Show imported symbols if available
        if let Some(symbols) = &dep.symbols {
            if !symbols.is_empty() {
                output.push(format!("   Symbols: {}", symbols.join(", ")));
            }
        }
    }

    if deps.len() > 20 {
        output.push(format!("\n... and {} more dependencies", deps.len() - 20));
    }

    output.join("\n")
}

/// Format reverse dependencies output
fn format_reverse_dependencies(file_path: &str, dependents: &[String]) -> String {
    if dependents.is_empty() {
        return format!("No files depend on '{}'.", file_path);
    }

    let mut output = Vec::new();
    output.push(format!("# Files that import '{}'\n", file_path));
    output.push(format!("Found {} files:\n", dependents.len()));

    for (idx, path) in dependents.iter().take(20).enumerate() {
        output.push(format!("{}. {}", idx + 1, path));
    }

    if dependents.len() > 20 {
        output.push(format!("\n... and {} more files", dependents.len() - 20));
    }

    output.join("\n")
}

/// Format analysis summary output
fn format_analysis_summary(
    hotspot_count: usize,
    unused_count: usize,
    circular_count: usize,
    min_dependents: usize,
) -> String {
    let mut output = Vec::new();

    output.push("# Dependency Analysis Summary\n".to_string());
    output.push(format!(
        "Hotspots (files with {}+ importers): {}",
        min_dependents, hotspot_count
    ));
    output.push(format!("Unused files (no importers): {}", unused_count));
    output.push(format!("Circular dependency chains: {}", circular_count));

    if hotspot_count > 0 {
        output.push(
            "\n**Hotspots** indicate central/important files that many other files depend on."
                .to_string(),
        );
    }

    if unused_count > 0 {
        output.push(
            "\n**Unused files** may be dead code or entry points (like main.rs, index.ts)."
                .to_string(),
        );
    }

    if circular_count > 0 {
        output.push("\n**Circular dependencies** can cause compilation issues and indicate architectural problems.".to_string());
    }

    output.join("\n")
}

/// Format islands output
fn format_islands(islands: &[Vec<String>], min_size: usize, max_size: usize) -> String {
    if islands.is_empty() {
        return format!(
            "No disconnected components found (size {}-{}).",
            min_size, max_size
        );
    }

    let mut output = Vec::new();
    output.push(format!("# Disconnected Components (Islands)\n"));
    output.push(format!(
        "Found {} islands (size {}-{}):\n",
        islands.len(),
        min_size,
        max_size
    ));

    for (idx, island) in islands.iter().take(5).enumerate() {
        output.push(format!(
            "\n{}. Island with {} files:",
            idx + 1,
            island.len()
        ));

        for (file_idx, file) in island.iter().take(10).enumerate() {
            output.push(format!("   {}. {}", file_idx + 1, file));
        }

        if island.len() > 10 {
            output.push(format!("   ... and {} more files", island.len() - 10));
        }
    }

    if islands.len() > 5 {
        output.push(format!("\n... and {} more islands", islands.len() - 5));
    }

    output.push("\n**Islands** are groups of files that depend on each other but have no dependencies outside the group.".to_string());
    output.push("This can indicate isolated subsystems or potential dead code.".to_string());

    output.join("\n")
}

/// Format all tool results into a single context string for the next LLM call
pub fn format_tool_results(results: &[ToolResult]) -> String {
    if results.is_empty() {
        return String::new();
    }

    let mut output = Vec::new();
    output.push("## Tool Execution Results\n".to_string());

    for (idx, result) in results.iter().enumerate() {
        output.push(format!("\n### Tool {} - {}", idx + 1, result.description));
        output.push(String::new());
        output.push(result.output.clone());
        output.push(String::new());
    }

    output.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_tool_results_empty() {
        let results = vec![];
        let output = format_tool_results(&results);
        assert!(output.is_empty());
    }

    #[test]
    fn test_format_tool_results_single() {
        let results = vec![ToolResult {
            description: "Test tool".to_string(),
            output: "Test output".to_string(),
            success: true,
        }];

        let output = format_tool_results(&results);
        assert!(output.contains("Tool Execution Results"));
        assert!(output.contains("Test tool"));
        assert!(output.contains("Test output"));
    }

    #[test]
    fn test_format_hotspots() {
        let hotspots = vec![
            ("src/main.rs".to_string(), 10),
            ("src/lib.rs".to_string(), 5),
        ];

        let output = format_hotspots(&hotspots);
        assert!(output.contains("most-imported files"));
        assert!(output.contains("src/main.rs"));
        assert!(output.contains("10 importers"));
    }

    #[test]
    fn test_format_unused_files() {
        let unused = vec!["src/old.rs".to_string(), "tests/legacy.rs".to_string()];

        let output = format_unused_files(&unused);
        assert!(output.contains("unused files"));
        assert!(output.contains("src/old.rs"));
    }

    #[test]
    fn test_format_circular_deps() {
        let circular = vec![vec![
            "a.rs".to_string(),
            "b.rs".to_string(),
            "a.rs".to_string(),
        ]];

        let output = format_circular_deps(&circular);
        assert!(output.contains("circular dependency"));
        assert!(output.contains("a.rs → b.rs → a.rs"));
    }
}
