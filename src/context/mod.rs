//! Codebase context generation for AI prompts
//!
//! This module provides structural and organizational context about the project
//! to help LLMs understand project layout and organization.

pub mod detection;
pub mod structure;

use crate::cache::CacheManager;
use anyhow::Result;

/// Context generation options
#[derive(Debug, Clone)]
pub struct ContextOptions {
    /// Show directory structure
    pub structure: bool,

    /// Focus on specific directory path
    pub path: Option<String>,

    /// Show file type distribution
    pub file_types: bool,

    /// Detect project type (CLI/library/webapp/monorepo)
    pub project_type: bool,

    /// Detect frameworks and conventions
    pub framework: bool,

    /// Show entry point files
    pub entry_points: bool,

    /// Show test organization pattern
    pub test_layout: bool,

    /// List important configuration files
    pub config_files: bool,

    /// Tree depth for --structure (default: 1)
    pub depth: usize,

    /// Output as JSON
    pub json: bool,
}

impl Default for ContextOptions {
    fn default() -> Self {
        Self {
            structure: true,
            path: None,
            file_types: true,
            project_type: true,
            framework: true,
            entry_points: true,
            test_layout: true,
            config_files: true,
            depth: 1,
            json: false,
        }
    }
}

impl ContextOptions {
    /// Check if no context types are explicitly enabled
    ///
    /// When true, we should enable all context types (default behavior)
    pub fn is_empty(&self) -> bool {
        !self.structure
            && !self.file_types
            && !self.project_type
            && !self.framework
            && !self.entry_points
            && !self.test_layout
            && !self.config_files
    }
}

/// Generate codebase context based on options
///
/// Returns formatted context string (human-readable or JSON)
pub fn generate_context(cache: &CacheManager, opts: &ContextOptions) -> Result<String> {
    let workspace_root = cache.workspace_root();
    let target_path = opts
        .path
        .as_ref()
        .map(|p| workspace_root.join(p))
        .unwrap_or_else(|| workspace_root.clone());

    // Validate target path exists
    if !target_path.exists() {
        anyhow::bail!(
            "Path '{}' does not exist in workspace",
            opts.path.as_deref().unwrap_or(".")
        );
    }

    // Apply defaults if no flags specified
    let mut effective_opts = opts.clone();
    if effective_opts.is_empty() {
        // Enable all context types by default
        effective_opts.structure = true;
        effective_opts.file_types = true;
        effective_opts.project_type = true;
        effective_opts.framework = true;
        effective_opts.entry_points = true;
        effective_opts.test_layout = true;
        effective_opts.config_files = true;
    }

    if opts.json {
        generate_json_context(cache, &effective_opts, &target_path)
    } else {
        generate_text_context(cache, &effective_opts, &target_path)
    }
}

/// Generate human-readable context
fn generate_text_context(
    cache: &CacheManager,
    opts: &ContextOptions,
    target_path: &std::path::Path,
) -> Result<String> {
    let mut sections = Vec::new();

    // Header
    let path_display = target_path
        .strip_prefix(cache.workspace_root())
        .unwrap_or(target_path)
        .display();
    sections.push(format!("# Project Context: {}\n", path_display));

    // Project type detection
    if opts.project_type {
        if let Ok(project_info) = detection::detect_project_type(cache, target_path) {
            sections.push(format!("## Project Type\n{}\n", project_info));
        }
    }

    // Entry points
    if opts.entry_points {
        if let Ok(entry_points) = detection::find_entry_points(target_path) {
            if !entry_points.is_empty() {
                sections.push(format!("## Entry Points\n{}\n", entry_points.join("\n")));
            }
        }
    }

    // Directory structure
    if opts.structure {
        if let Ok(tree) = structure::generate_tree(target_path, opts.depth) {
            sections.push(format!("## Directory Structure\n{}\n", tree));
        }
    }

    // File type distribution
    if opts.file_types {
        if let Ok(distribution) = detection::get_file_distribution(cache) {
            sections.push(format!("## File Distribution\n{}\n", distribution));
        }
    }

    // Test layout
    if opts.test_layout {
        if let Ok(test_info) = detection::detect_test_layout(target_path) {
            sections.push(format!("## Test Organization\n{}\n", test_info));
        }
    }

    // Framework detection
    if opts.framework {
        if let Ok(frameworks) = detection::detect_frameworks(target_path) {
            if !frameworks.is_empty() {
                sections.push(format!("## Framework Detection\n{}\n", frameworks));
            }
        }
    }

    // Configuration files
    if opts.config_files {
        if let Ok(configs) = detection::find_config_files(target_path) {
            if !configs.is_empty() {
                sections.push(format!("## Configuration Files\n{}\n", configs));
            }
        }
    }

    Ok(sections.join("\n"))
}

/// Generate JSON context
fn generate_json_context(
    cache: &CacheManager,
    opts: &ContextOptions,
    target_path: &std::path::Path,
) -> Result<String> {
    use serde_json::json;

    let mut context = json!({});

    if opts.project_type {
        if let Ok(project_type) = detection::detect_project_type_json(cache, target_path) {
            context["project_type"] = project_type;
        }
    }

    if opts.entry_points {
        if let Ok(entry_points) = detection::find_entry_points_json(target_path) {
            context["entry_points"] = entry_points;
        }
    }

    if opts.structure {
        if let Ok(tree) = structure::generate_tree_json(target_path, opts.depth) {
            context["structure"] = tree;
        }
    }

    if opts.file_types {
        if let Ok(distribution) = detection::get_file_distribution_json(cache) {
            context["file_distribution"] = distribution;
        }
    }

    if opts.test_layout {
        if let Ok(test_layout) = detection::detect_test_layout_json(target_path) {
            context["test_layout"] = test_layout;
        }
    }

    if opts.framework {
        if let Ok(frameworks) = detection::detect_frameworks_json(target_path) {
            context["frameworks"] = frameworks;
        }
    }

    if opts.config_files {
        if let Ok(configs) = detection::find_config_files_json(target_path) {
            context["config_files"] = configs;
        }
    }

    serde_json::to_string_pretty(&context).map_err(Into::into)
}
