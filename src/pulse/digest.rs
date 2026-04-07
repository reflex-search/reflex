//! Digest generation: periodic narrative of structural changes
//!
//! Produces a structured report from a snapshot diff, optionally narrated by an LLM.
//! When no baseline exists, generates a bootstrap report of current state.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::diff::SnapshotDiff;
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
pub fn generate_digest(
    diff: Option<&SnapshotDiff>,
    current_snapshot: &SnapshotInfo,
    no_llm: bool,
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
            // Bootstrap mode: report on current state
            sections.push(DigestSection {
                heading: "Codebase Overview".to_string(),
                structural_content: format!(
                    "First snapshot — no comparison baseline available.\n\n\
                     - **Files**: {}\n\
                     - **Total lines**: {}\n\
                     - **Dependency edges**: {}\n\
                     - **Branch**: {}\n\
                     - **Commit**: {}",
                    current_snapshot.file_count,
                    current_snapshot.total_lines,
                    current_snapshot.edge_count,
                    current_snapshot.git_branch.as_deref().unwrap_or("unknown"),
                    current_snapshot.git_commit.as_deref().map(|c| &c[..8.min(c.len())]).unwrap_or("unknown"),
                ),
                narrative: None,
                evidence: vec![],
            });
        }
    }

    let title = if diff.is_some() {
        "Structural Change Digest".to_string()
    } else {
        "Codebase Snapshot Report".to_string()
    };

    Ok(Digest {
        title,
        sections,
        is_bootstrap: diff.is_none(),
        narration_mode: if no_llm { NarrationMode::Structural } else { NarrationMode::Narrated },
    })
}

/// Render a digest as markdown
pub fn render_markdown(digest: &Digest) -> String {
    let mut md = String::new();

    md.push_str(&format!("# {}\n\n", digest.title));

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

        let digest = generate_digest(None, &snapshot, true).unwrap();
        assert!(digest.is_bootstrap);
        assert_eq!(digest.sections.len(), 1);
        let md = render_markdown(&digest);
        assert!(md.contains("First snapshot"));
        assert!(md.contains("100"));
    }
}
