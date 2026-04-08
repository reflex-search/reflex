//! Digest generation: periodic narrative of structural changes
//!
//! Produces a structured report from a snapshot diff, optionally narrated by an LLM.
//! When no baseline exists, generates a bootstrap report of current state.

use anyhow::Result;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::cache::CacheManager;
use crate::semantic::providers::LlmProvider;

use super::diff::SnapshotDiff;
use super::llm_cache::LlmCache;
use super::narrate;
use super::snapshot::SnapshotInfo;

/// A complete digest report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Digest {
    pub title: String,
    pub sections: Vec<DigestSection>,
    pub is_bootstrap: bool,
    pub narration_mode: NarrationMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NarrationMode {
    Structural,
    Narrated,
}

/// A section of the digest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigestSection {
    pub heading: String,
    pub structural_content: String,
    pub narrative: Option<String>,
    pub evidence: Vec<String>,
}

/// Generate a digest from a diff (or bootstrap from current snapshot)
///
/// When `diff` is None, produces a bootstrap report from the current snapshot.
/// When `no_llm` is true, all sections are structural-only.
/// `provider` and `llm_cache` are created by the caller (site.rs or CLI handler).
pub fn generate_digest(
    diff: Option<&SnapshotDiff>,
    current_snapshot: &SnapshotInfo,
    cache: Option<&CacheManager>,
    no_llm: bool,
    provider: Option<&dyn LlmProvider>,
    llm_cache: Option<&LlmCache>,
) -> Result<Digest> {
    let mut sections = Vec::new();

    match diff {
        Some(diff) => {
            // Overview section
            sections.push(build_overview_section(diff));

            // File changes section
            if diff.summary.files_added > 0 || diff.summary.files_removed > 0 || diff.summary.files_modified > 0 {
                sections.push(build_file_changes_section(diff));
            }

            // Dependency changes
            if diff.summary.edges_added > 0 || diff.summary.edges_removed > 0 {
                sections.push(build_dependency_section(diff));
            }

            // Hotspot changes
            if !diff.hotspot_changes.is_empty() {
                sections.push(build_hotspot_section(diff));
            }

            // Cycle changes
            if !diff.new_cycles.is_empty() || !diff.resolved_cycles.is_empty() {
                sections.push(build_cycle_section(diff));
            }

            // Module changes
            if !diff.module_changes.is_empty() {
                sections.push(build_module_section(diff));
            }

            // Threshold alerts
            if !diff.threshold_alerts.is_empty() {
                sections.push(build_alerts_section(diff));
            }
        }
        None => {
            // Bootstrap mode: report on current state with enriched structural data
            let mut content = format!(
                "First snapshot — no comparison baseline available.\n\n\
                 | Metric | Value |\n|---|---|\n\
                 | Files | {} |\n\
                 | Total lines | {} |\n\
                 | Dependency edges | {} |\n\
                 | Branch | {} |\n\
                 | Commit | {} |",
                current_snapshot.file_count,
                current_snapshot.total_lines,
                current_snapshot.edge_count,
                current_snapshot.git_branch.as_deref().unwrap_or("unknown"),
                current_snapshot.git_commit.as_deref().map(|c| &c[..8.min(c.len())]).unwrap_or("unknown"),
            );

            // Enrich with data from cache if available
            if let Some(cache) = cache {
                let db_path = cache.path().join("meta.db");
                if let Ok(conn) = Connection::open(&db_path) {
                    // Module-level summary using detect_modules
                    if let Ok(modules) = super::wiki::detect_modules(cache) {
                        if !modules.is_empty() {
                            let tier1: Vec<_> = modules.iter().filter(|m| m.tier == 1).collect();
                            let tier2: Vec<_> = modules.iter().filter(|m| m.tier == 2).collect();

                            content.push_str(&format!(
                                "\n\n### Module Summary\n\n\
                                 **{} top-level modules**, **{} sub-modules** detected.\n\n",
                                tier1.len(), tier2.len()
                            ));

                            content.push_str("| Module | Files | Lines | Languages |\n|---|---|---|---|\n");
                            for m in &modules {
                                let indent = if m.tier == 2 { "  " } else { "" };
                                content.push_str(&format!(
                                    "| {}{} | {} | {} | {} |\n",
                                    indent, m.path, m.file_count, m.total_lines,
                                    m.languages.join(", ")
                                ));
                            }
                        }
                    }

                    // Language distribution
                    if let Ok(lang_dist) = build_language_distribution(&conn) {
                        if !lang_dist.is_empty() {
                            content.push_str("\n### Language Distribution\n\n");
                            content.push_str("| Language | Files | Lines |\n|---|---|---|\n");
                            for (lang, files, lines) in &lang_dist {
                                content.push_str(&format!("| {} | {} | {} |\n", lang, files, lines));
                            }
                        }
                    }

                    // Dependency health: cycles and hotspots
                    if let Ok(hotspots) = build_hotspots_overview(&conn) {
                        if !hotspots.is_empty() {
                            content.push_str("\n### Dependency Hotspots\n\n");
                            content.push_str("Files with most incoming dependencies:\n\n");
                            for (path, count) in &hotspots {
                                content.push_str(&format!("- `{}` ({} dependents)\n", path, count));
                            }
                        }
                    }

                    // Check for circular dependencies
                    if let Ok(cycle_count) = build_cycle_count(&conn) {
                        if cycle_count > 0 {
                            content.push_str(&format!(
                                "\n### Dependency Health\n\n\
                                 **{} circular dependencies** detected. Run `rfx analyze --circular` for details.\n",
                                cycle_count
                            ));
                        } else {
                            content.push_str("\n### Dependency Health\n\nNo circular dependencies detected.\n");
                        }
                    }

                    // Largest files
                    if let Ok(largest) = build_largest_files(&conn) {
                        if !largest.is_empty() {
                            content.push_str("\n### Largest Files\n\n");
                            for (path, lines) in &largest {
                                content.push_str(&format!("- `{}` ({} lines)\n", path, lines));
                            }
                        }
                    }
                }
            }

            sections.push(DigestSection {
                heading: "Codebase Overview".to_string(),
                structural_content: content,
                narrative: None,
                evidence: vec![],
            });
        }
    }

    // Wire LLM narration when provider is available
    if !no_llm {
        if let (Some(provider), Some(llm_cache)) = (provider, llm_cache) {
            eprintln!("Generating digest narration...");
            let snapshot_id = &current_snapshot.id;

            for section in &mut sections {
                section.narrative = narrate::narrate_section(
                    provider,
                    narrate::digest_system_prompt(),
                    &section.structural_content,
                    llm_cache,
                    snapshot_id,
                    &section.heading,
                );
            }
        }
    }

    let has_any_narrative = sections.iter().any(|s| s.narrative.is_some());

    let title = if diff.is_some() {
        "Structural Change Digest".to_string()
    } else {
        "Codebase Snapshot Report".to_string()
    };

    Ok(Digest {
        title,
        sections,
        is_bootstrap: diff.is_none(),
        narration_mode: if no_llm || !has_any_narrative {
            NarrationMode::Structural
        } else {
            NarrationMode::Narrated
        },
    })
}

/// Render a digest as markdown
pub fn render_markdown(digest: &Digest) -> String {
    let mut md = String::new();

    if digest.is_bootstrap {
        md.push_str("*First snapshot — no comparison baseline available.*\n\n");
    }

    match digest.narration_mode {
        NarrationMode::Structural => {
            md.push_str("*Structural-only mode (no LLM narration)*\n\n");
        }
        NarrationMode::Narrated => {}
    }

    for section in &digest.sections {
        md.push_str(&format!("## {}\n\n", section.heading));

        if let Some(narrative) = &section.narrative {
            md.push_str(narrative);
            md.push_str("\n\n");
        }

        md.push_str(&section.structural_content);
        md.push_str("\n\n");

        if !section.evidence.is_empty() {
            md.push_str("**Evidence:**\n");
            for item in &section.evidence {
                md.push_str(&format!("- {}\n", item));
            }
            md.push('\n');
        }
    }

    md
}

fn build_overview_section(diff: &SnapshotDiff) -> DigestSection {
    let content = format!(
        "| Metric | Count |\n|---|---|\n\
         | Files added | {} |\n\
         | Files removed | {} |\n\
         | Files modified | {} |\n\
         | Edges added | {} |\n\
         | Edges removed | {} |\n\
         | Net line change | {:+} |",
        diff.summary.files_added,
        diff.summary.files_removed,
        diff.summary.files_modified,
        diff.summary.edges_added,
        diff.summary.edges_removed,
        diff.summary.net_line_change,
    );

    DigestSection {
        heading: "Overview".to_string(),
        structural_content: content,
        narrative: None,
        evidence: vec![],
    }
}

fn build_file_changes_section(diff: &SnapshotDiff) -> DigestSection {
    let mut content = String::new();

    if !diff.files_added.is_empty() {
        content.push_str("### Added\n");
        for f in diff.files_added.iter().take(20) {
            content.push_str(&format!("- `{}` ({}, {} lines)\n",
                f.path,
                f.language.as_deref().unwrap_or("unknown"),
                f.line_count
            ));
        }
        if diff.files_added.len() > 20 {
            content.push_str(&format!("- ... and {} more\n", diff.files_added.len() - 20));
        }
    }

    if !diff.files_removed.is_empty() {
        content.push_str("\n### Removed\n");
        for f in diff.files_removed.iter().take(20) {
            content.push_str(&format!("- `{}` ({} lines)\n", f.path, f.line_count));
        }
        if diff.files_removed.len() > 20 {
            content.push_str(&format!("- ... and {} more\n", diff.files_removed.len() - 20));
        }
    }

    if !diff.files_modified.is_empty() {
        content.push_str("\n### Modified\n");
        for f in diff.files_modified.iter().take(20) {
            let delta = f.new_line_count as i64 - f.old_line_count as i64;
            content.push_str(&format!("- `{}` ({:+} lines)\n", f.path, delta));
        }
        if diff.files_modified.len() > 20 {
            content.push_str(&format!("- ... and {} more\n", diff.files_modified.len() - 20));
        }
    }

    DigestSection {
        heading: "File Changes".to_string(),
        structural_content: content,
        narrative: None,
        evidence: vec![],
    }
}

fn build_dependency_section(diff: &SnapshotDiff) -> DigestSection {
    let mut content = String::new();

    if !diff.edges_added.is_empty() {
        content.push_str("### New Dependencies\n");
        for e in diff.edges_added.iter().take(15) {
            content.push_str(&format!("- `{}` → `{}` ({})\n",
                e.source_path, e.target_path, e.import_type
            ));
        }
        if diff.edges_added.len() > 15 {
            content.push_str(&format!("- ... and {} more\n", diff.edges_added.len() - 15));
        }
    }

    if !diff.edges_removed.is_empty() {
        content.push_str("\n### Removed Dependencies\n");
        for e in diff.edges_removed.iter().take(15) {
            content.push_str(&format!("- `{}` → `{}`\n", e.source_path, e.target_path));
        }
        if diff.edges_removed.len() > 15 {
            content.push_str(&format!("- ... and {} more\n", diff.edges_removed.len() - 15));
        }
    }

    DigestSection {
        heading: "Dependency Changes".to_string(),
        structural_content: content,
        narrative: None,
        evidence: vec![],
    }
}

fn build_hotspot_section(diff: &SnapshotDiff) -> DigestSection {
    let mut content = String::new();
    for h in diff.hotspot_changes.iter().take(10) {
        let direction = if h.new_fan_in > h.old_fan_in { "↑" } else { "↓" };
        content.push_str(&format!("- `{}`: {} → {} fan-in {}\n",
            h.path, h.old_fan_in, h.new_fan_in, direction
        ));
    }

    DigestSection {
        heading: "Hotspot Changes".to_string(),
        structural_content: content,
        narrative: None,
        evidence: vec![],
    }
}

fn build_cycle_section(diff: &SnapshotDiff) -> DigestSection {
    let mut content = String::new();

    if !diff.new_cycles.is_empty() {
        content.push_str("### New Circular Dependencies\n");
        for cycle in diff.new_cycles.iter().take(5) {
            content.push_str(&format!("- {}\n", cycle.join(" → ")));
        }
    }

    if !diff.resolved_cycles.is_empty() {
        content.push_str("\n### Resolved Circular Dependencies\n");
        for cycle in diff.resolved_cycles.iter().take(5) {
            content.push_str(&format!("- {}\n", cycle.join(" → ")));
        }
    }

    DigestSection {
        heading: "Circular Dependency Changes".to_string(),
        structural_content: content,
        narrative: None,
        evidence: vec![],
    }
}

fn build_module_section(diff: &SnapshotDiff) -> DigestSection {
    let mut content = String::new();

    for m in diff.module_changes.iter().take(20) {
        let fc_old = m.old_file_count.map(|c| c.to_string()).unwrap_or_else(|| "new".to_string());
        let fc_new = m.new_file_count.map(|c| c.to_string()).unwrap_or_else(|| "removed".to_string());
        let tl_old = m.old_total_lines.map(|c| c.to_string()).unwrap_or_else(|| "-".to_string());
        let tl_new = m.new_total_lines.map(|c| c.to_string()).unwrap_or_else(|| "-".to_string());
        content.push_str(&format!("- `{}`: {} → {} files, {} → {} lines\n",
            m.module_path, fc_old, fc_new, tl_old, tl_new
        ));
    }

    DigestSection {
        heading: "Module Activity".to_string(),
        structural_content: content,
        narrative: None,
        evidence: vec![],
    }
}

/// Query language distribution from meta.db: (language, file_count, line_count)
fn build_language_distribution(conn: &Connection) -> Result<Vec<(String, usize, usize)>> {
    let mut stmt = conn.prepare(
        "SELECT COALESCE(language, 'other'), COUNT(*), COALESCE(SUM(line_count), 0)
         FROM files
         GROUP BY language
         ORDER BY COUNT(*) DESC
         LIMIT 15"
    )?;

    let rows: Vec<(String, usize, usize)> = stmt.query_map([], |row| {
        Ok((row.get(0)?, row.get(1)?, row.get(2)?))
    })?.filter_map(|r| r.ok()).collect();

    Ok(rows)
}

/// Query most-imported files (dependency hotspots)
fn build_hotspots_overview(conn: &Connection) -> Result<Vec<(String, usize)>> {
    // Check if file_dependencies table exists
    let table_exists: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='file_dependencies'",
        [],
        |row| row.get(0),
    )?;

    if !table_exists {
        return Ok(vec![]);
    }

    let mut stmt = conn.prepare(
        "SELECT f.path, COUNT(DISTINCT fd.file_id) as dep_count
         FROM file_dependencies fd
         JOIN files f ON fd.resolved_file_id = f.id
         GROUP BY fd.resolved_file_id
         ORDER BY dep_count DESC
         LIMIT 5"
    )?;

    let rows: Vec<(String, usize)> = stmt.query_map([], |row| {
        Ok((row.get(0)?, row.get(1)?))
    })?.filter_map(|r| r.ok()).collect();

    Ok(rows)
}

/// Count circular dependencies (approximation via self-referencing paths)
fn build_cycle_count(conn: &Connection) -> Result<usize> {
    // Check if file_dependencies table exists
    let table_exists: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='file_dependencies'",
        [],
        |row| row.get(0),
    )?;

    if !table_exists {
        return Ok(0);
    }

    // Count files that have mutual dependencies (A→B and B→A)
    let count: usize = conn.query_row(
        "SELECT COUNT(*) FROM (
            SELECT DISTINCT fd1.file_id
            FROM file_dependencies fd1
            JOIN file_dependencies fd2
              ON fd1.file_id = fd2.resolved_file_id
              AND fd1.resolved_file_id = fd2.file_id
            WHERE fd1.resolved_file_id IS NOT NULL
              AND fd2.resolved_file_id IS NOT NULL
         )",
        [],
        |row| row.get(0),
    )?;

    Ok(count)
}

/// Get the largest files in the codebase
fn build_largest_files(conn: &Connection) -> Result<Vec<(String, usize)>> {
    let mut stmt = conn.prepare(
        "SELECT path, COALESCE(line_count, 0) as lc
         FROM files
         ORDER BY lc DESC
         LIMIT 10"
    )?;

    let rows: Vec<(String, usize)> = stmt.query_map([], |row| {
        Ok((row.get(0)?, row.get(1)?))
    })?.filter_map(|r| r.ok()).collect();

    Ok(rows)
}

fn build_alerts_section(diff: &SnapshotDiff) -> DigestSection {
    let mut content = String::new();

    for alert in &diff.threshold_alerts {
        let icon = match alert.severity {
            super::diff::AlertSeverity::Critical => "🔴",
            super::diff::AlertSeverity::Warning => "🟡",
        };
        let path_info = alert.path.as_deref().map(|p| format!(" (`{}`)", p)).unwrap_or_default();
        content.push_str(&format!("{} **{}**: {}{}\n", icon, alert.category, alert.message, path_info));
    }

    DigestSection {
        heading: "Threshold Alerts".to_string(),
        structural_content: content,
        narrative: None,
        evidence: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_digest() {
        let snapshot = SnapshotInfo {
            id: "test".to_string(),
            path: std::path::PathBuf::from("/tmp/test.db"),
            timestamp: "2026-04-07".to_string(),
            git_branch: Some("main".to_string()),
            git_commit: Some("abc12345".to_string()),
            reflex_version: "1.0.5".to_string(),
            file_count: 100,
            total_lines: 10000,
            edge_count: 50,
            size_bytes: 1024,
        };

        let digest = generate_digest(None, &snapshot, None, true, None, None).unwrap();
        assert!(digest.is_bootstrap);
        assert_eq!(digest.sections.len(), 1);
        let md = render_markdown(&digest);
        assert!(md.contains("First snapshot"));
        assert!(md.contains("100"));
    }
}
