//! Codebase context extraction for semantic query generation
//!
//! This module extracts rich context about the indexed codebase to help LLMs
//! generate better search queries. Context includes language distribution,
//! directory structure, monorepo detection, and more.

use crate::cache::CacheManager;
use anyhow::{Context as AnyhowContext, Result};
use rusqlite::Connection;
use std::collections::HashMap;
use std::path::Path;

/// Comprehensive codebase context for LLM prompt injection
#[derive(Debug, Clone)]
pub struct CodebaseContext {
    /// Total number of indexed files
    pub total_files: usize,

    /// Language distribution with counts and percentages
    pub languages: Vec<LanguageInfo>,

    /// Top-level directories (first path segment)
    pub top_level_dirs: Vec<String>,

    /// Common path patterns (depth 2-3) for framework-aware suggestions
    pub common_paths: Vec<String>,

    /// Whether this appears to be a monorepo
    pub is_monorepo: bool,

    /// Number of detected projects in monorepo (if applicable)
    pub project_count: Option<usize>,

    /// Dominant language (if any language is >60% of files)
    pub dominant_language: Option<LanguageInfo>,
}

/// Language information with count and percentage
#[derive(Debug, Clone)]
pub struct LanguageInfo {
    pub name: String,
    pub file_count: usize,
    pub percentage: f64,
}

impl CodebaseContext {
    /// Extract comprehensive context from cache
    pub fn extract(cache: &CacheManager) -> Result<Self> {
        let db_path = cache.path().join("meta.db");
        let conn =
            Connection::open(&db_path).context("Failed to open database for context extraction")?;

        // Get total file count
        let total_files: usize = conn
            .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))
            .unwrap_or(0);

        // Extract language distribution
        let languages = extract_language_distribution(&conn, total_files)?;

        // Find dominant language (>60% of files)
        let dominant_language = languages
            .iter()
            .find(|lang| lang.percentage > 60.0)
            .cloned();

        // Extract file paths for directory analysis
        let file_paths = extract_file_paths(&conn)?;

        // Analyze directory structure
        let top_level_dirs = extract_top_level_dirs(&file_paths);
        let common_paths = extract_common_paths(&file_paths, 2, 10); // depth 2-3, top 10

        // Detect monorepo
        let (is_monorepo, project_count) = detect_monorepo(&file_paths);

        Ok(Self {
            total_files,
            languages,
            top_level_dirs,
            common_paths,
            is_monorepo,
            project_count,
            dominant_language,
        })
    }

    /// Format context as a human-readable string for LLM prompt injection
    pub fn to_prompt_string(&self) -> String {
        let mut parts = Vec::new();

        // Language distribution (Tier 1)
        if !self.languages.is_empty() {
            let lang_summary: Vec<String> = self
                .languages
                .iter()
                .map(|lang| {
                    format!(
                        "{} ({} files, {:.0}%)",
                        lang.name, lang.file_count, lang.percentage
                    )
                })
                .collect();
            parts.push(format!("**Languages:** {}", lang_summary.join(", ")));
        }

        // File scale indicator (Tier 1)
        let scale_hint = if self.total_files < 100 {
            "small codebase - broad queries work well"
        } else if self.total_files < 1000 {
            "medium codebase - moderate specificity recommended"
        } else {
            "large codebase - use specific filters for best results"
        };
        parts.push(format!(
            "**Total files:** {} ({})",
            self.total_files, scale_hint
        ));

        // Top-level directories (Tier 1)
        if !self.top_level_dirs.is_empty() {
            parts.push(format!(
                "**Top-level directories:** {}",
                self.top_level_dirs.join(", ")
            ));
        }

        // Dominant language (Tier 2)
        if let Some(ref dominant) = self.dominant_language {
            parts.push(format!(
                "**Primary language:** {} ({:.0}% of codebase)",
                dominant.name, dominant.percentage
            ));
        }

        // Common paths (Tier 2)
        if !self.common_paths.is_empty() {
            let paths_str = self
                .common_paths
                .iter()
                .take(8) // Limit to 8 most common
                .map(|p| p.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            parts.push(format!("**Common paths:** {}", paths_str));
        }

        // Monorepo info (Tier 2)
        if self.is_monorepo {
            if let Some(count) = self.project_count {
                parts.push(format!("**Monorepo:** Yes ({} projects detected - use --file to target specific projects)", count));
            } else {
                parts
                    .push("**Monorepo:** Yes (use --file to target specific projects)".to_string());
            }
        }

        parts.join("\n")
    }
}

/// Extract language distribution with counts and percentages
fn extract_language_distribution(
    conn: &Connection,
    total_files: usize,
) -> Result<Vec<LanguageInfo>> {
    let mut stmt = conn.prepare(
        "SELECT language, COUNT(*) as count
         FROM files
         WHERE language IS NOT NULL
         GROUP BY language
         ORDER BY count DESC",
    )?;

    let languages = stmt
        .query_map([], |row| {
            let name: String = row.get(0)?;
            let file_count: usize = row.get(1)?;
            let percentage = if total_files > 0 {
                (file_count as f64 / total_files as f64) * 100.0
            } else {
                0.0
            };

            Ok(LanguageInfo {
                name,
                file_count,
                percentage,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(languages)
}

/// Extract all file paths from database
fn extract_file_paths(conn: &Connection) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT path FROM files")?;
    let paths = stmt
        .query_map([], |row| row.get(0))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(paths)
}

/// Extract top-level directories (first path segment)
fn extract_top_level_dirs(paths: &[String]) -> Vec<String> {
    let mut dir_counts: HashMap<String, usize> = HashMap::new();

    for path in paths {
        if let Some(first_segment) = path.split('/').next() {
            if !first_segment.is_empty() && !first_segment.starts_with('.') {
                *dir_counts.entry(first_segment.to_string()).or_insert(0) += 1;
            }
        }
    }

    // Return top directories sorted by count (descending)
    let mut dirs: Vec<(String, usize)> = dir_counts.into_iter().collect();
    dirs.sort_by(|a, b| b.1.cmp(&a.1));

    // Return top 10 directories with trailing slash
    dirs.into_iter()
        .take(10)
        .map(|(dir, _)| format!("{}/", dir))
        .collect()
}

/// Extract common path patterns at specified depth
fn extract_common_paths(paths: &[String], min_depth: usize, max_results: usize) -> Vec<String> {
    let mut path_counts: HashMap<String, usize> = HashMap::new();

    for path in paths {
        let segments: Vec<&str> = path.split('/').collect();

        // Extract paths at depth 2 and 3
        for depth in min_depth..=3 {
            if segments.len() > depth {
                let partial_path = segments[..=depth].join("/");
                // Skip if it's just a filename (no directory structure)
                if !partial_path.contains('/') {
                    continue;
                }
                // Skip hidden directories and common noise
                if partial_path.contains("/.")
                    || partial_path.contains("/node_modules")
                    || partial_path.contains("/vendor")
                    || partial_path.contains("/target")
                {
                    continue;
                }
                *path_counts.entry(partial_path).or_insert(0) += 1;
            }
        }
    }

    // Filter to paths that appear at least 3 times (signal vs noise)
    let min_count = 3;
    let mut common_paths: Vec<(String, usize)> = path_counts
        .into_iter()
        .filter(|(_, count)| *count >= min_count)
        .collect();

    // Sort by count descending
    common_paths.sort_by(|a, b| b.1.cmp(&a.1));

    // Return top paths with trailing slash
    common_paths
        .into_iter()
        .take(max_results)
        .map(|(path, _)| format!("{}/", path))
        .collect()
}

/// Detect if this is a monorepo by counting package manager files
fn detect_monorepo(paths: &[String]) -> (bool, Option<usize>) {
    let package_files = [
        "package.json",
        "Cargo.toml",
        "go.mod",
        "composer.json",
        "pom.xml",
        "build.gradle",
        "Gemfile",
    ];

    let mut project_count = 0;

    for path in paths {
        let path_lower = path.to_lowercase();
        for pkg_file in &package_files {
            if path_lower.ends_with(pkg_file) {
                // Skip root-level package files (not indicative of monorepo)
                // Only count if in subdirectory (e.g., packages/foo/package.json)
                if Path::new(path).components().count() > 2 {
                    project_count += 1;
                    break; // Don't double-count same project
                }
            }
        }
    }

    let is_monorepo = project_count >= 2;
    let project_count_opt = if is_monorepo {
        Some(project_count)
    } else {
        None
    };

    (is_monorepo, project_count_opt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_top_level_dirs() {
        let paths = vec![
            "src/main.rs".to_string(),
            "src/lib.rs".to_string(),
            "app/models/user.rb".to_string(),
            "app/controllers/home.rb".to_string(),
            "tests/test.rs".to_string(),
        ];

        let dirs = extract_top_level_dirs(&paths);
        assert_eq!(dirs.len(), 3);
        assert!(dirs.contains(&"src/".to_string()));
        assert!(dirs.contains(&"app/".to_string()));
        assert!(dirs.contains(&"tests/".to_string()));
    }

    #[test]
    fn test_extract_common_paths() {
        let paths = vec![
            "app/models/user.rb".to_string(),
            "app/models/post.rb".to_string(),
            "app/models/comment.rb".to_string(),
            "app/models/article.rb".to_string(),
            "app/controllers/home.rb".to_string(),
            "app/controllers/posts.rb".to_string(),
            "app/controllers/articles.rb".to_string(),
            "app/controllers/users.rb".to_string(),
            "src/main.rs".to_string(),
        ];

        let common = extract_common_paths(&paths, 1, 10);
        assert!(common.contains(&"app/models/".to_string()));
        assert!(common.contains(&"app/controllers/".to_string()));
    }

    #[test]
    fn test_detect_monorepo() {
        let monorepo_paths = vec![
            "packages/web/package.json".to_string(),
            "packages/api/package.json".to_string(),
            "packages/shared/package.json".to_string(),
        ];

        let (is_monorepo, count) = detect_monorepo(&monorepo_paths);
        assert!(is_monorepo);
        assert_eq!(count, Some(3));

        let single_project = vec!["package.json".to_string(), "src/main.ts".to_string()];

        let (is_mono, _) = detect_monorepo(&single_project);
        assert!(!is_mono);
    }

    #[test]
    fn test_language_percentage() {
        let lang = LanguageInfo {
            name: "Rust".to_string(),
            file_count: 64,
            percentage: 64.0,
        };

        assert_eq!(lang.percentage, 64.0);
    }
}
