//! Changelog generation: product-level summary of recent development activity
//!
//! Extracts recent git commits, groups them via LLM into high-level changelog
//! entries, and renders clean prose without commit hashes or file paths.
//! Falls back to a structural commit list when no LLM is available.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

use super::git_intel;

/// A complete changelog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Changelog {
    pub entries: Vec<ChangelogEntry>,
    pub raw_commits: Vec<ChangelogCommit>,
    pub branch: String,
    pub narrated: bool,
}

/// A single high-level changelog entry (rendered output)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangelogEntry {
    pub title: String,
    pub description: String,
}

/// A raw commit — used as LLM input context, NOT rendered directly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangelogCommit {
    pub hash: String,
    pub author: String,
    pub timestamp: i64,
    pub date: String,
    pub subject: String,
    pub files_changed: Vec<String>,
}

/// Extract recent commits and branch name from git.
///
/// Returns `(commits, branch_name)`.
pub fn extract_changelog_commits(root: &Path, count: usize) -> Result<(Vec<ChangelogCommit>, String)> {
    // Get branch name
    let branch_output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .context("Failed to run git rev-parse")?;

    let branch = if branch_output.status.success() {
        String::from_utf8_lossy(&branch_output.stdout).trim().to_string()
    } else {
        "unknown".to_string()
    };

    // Get commits with file lists
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "log",
            &format!("-{}", count),
            "--format=%H|%an|%at|%s",
            "--name-only",
        ])
        .output()
        .context("Failed to run git log")?;

    if !output.status.success() {
        return Ok((vec![], branch));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut commits: Vec<ChangelogCommit> = Vec::new();

    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Try to parse as a commit header line (has | separators)
        let parts: Vec<&str> = trimmed.splitn(4, '|').collect();
        if parts.len() == 4 && parts[0].len() == 40 {
            let timestamp: i64 = parts[2].parse().unwrap_or(0);
            commits.push(ChangelogCommit {
                hash: parts[0].to_string(),
                author: parts[1].to_string(),
                timestamp,
                date: git_intel::epoch_to_date_string(timestamp),
                subject: parts[3].to_string(),
                files_changed: Vec::new(),
            });
        } else if let Some(last) = commits.last_mut() {
            // It's a file path belonging to the most recent commit
            last.files_changed.push(trimmed.to_string());
        }
    }

    Ok((commits, branch))
}

/// Top-level extraction: get commits + build a Changelog struct (structural, no narration).
pub fn extract_changelog(root: &Path, count: usize) -> Result<Changelog> {
    let (commits, branch) = extract_changelog_commits(root, count)?;
    let entries = generate_structural_entries(&commits);

    Ok(Changelog {
        entries,
        raw_commits: commits,
        branch,
        narrated: false,
    })
}

/// Build the context string sent to the LLM (the user never sees this directly).
pub fn build_changelog_context(commits: &[ChangelogCommit], branch: &str) -> String {
    let mut ctx = String::new();
    ctx.push_str(&format!("Branch: {}\n", branch));
    ctx.push_str(&format!("{} most recent commits:\n\n", commits.len()));

    for (i, commit) in commits.iter().enumerate() {
        ctx.push_str(&format!(
            "{}. \"{}\" ({}, {})\n",
            i + 1,
            commit.subject,
            commit.author,
            commit.date,
        ));

        if !commit.files_changed.is_empty() {
            // Group file paths by top-level directory for brevity
            let areas: Vec<&str> = commit
                .files_changed
                .iter()
                .filter_map(|f| {
                    let parts: Vec<&str> = f.splitn(3, '/').collect();
                    if parts.len() >= 2 {
                        Some(parts[..2].join("/").leak() as &str)
                    } else {
                        Some(f.as_str())
                    }
                })
                .collect();

            // Deduplicate
            let mut unique_areas: Vec<&str> = areas.clone();
            unique_areas.sort();
            unique_areas.dedup();
            let display: Vec<&str> = unique_areas.into_iter().take(5).collect();

            ctx.push_str(&format!("   Areas: {}\n", display.join(", ")));
        }
        ctx.push('\n');
    }

    ctx
}

/// Parse the LLM's JSON response into changelog entries.
///
/// Expected format: `{ "entries": [{ "title": "...", "description": "..." }] }`
/// Falls back to structural entries on parse failure.
pub fn parse_changelog_response(response: &str, commits: &[ChangelogCommit]) -> Vec<ChangelogEntry> {
    // Strip markdown fences if present
    let cleaned = response
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    #[derive(Deserialize)]
    struct LlmResponse {
        entries: Vec<LlmEntry>,
    }

    #[derive(Deserialize)]
    struct LlmEntry {
        title: String,
        description: String,
    }

    match serde_json::from_str::<LlmResponse>(cleaned) {
        Ok(parsed) => parsed
            .entries
            .into_iter()
            .map(|e| ChangelogEntry {
                title: e.title,
                description: e.description,
            })
            .collect(),
        Err(e) => {
            log::warn!("Failed to parse changelog LLM response: {}", e);
            generate_structural_entries(commits)
        }
    }
}

/// Generate simple entries from raw commits (no-LLM fallback).
///
/// One entry per commit: title = subject, description = date + author.
pub fn generate_structural_entries(commits: &[ChangelogCommit]) -> Vec<ChangelogEntry> {
    commits
        .iter()
        .map(|c| ChangelogEntry {
            title: c.subject.clone(),
            description: format!("{} — {}", c.author, c.date),
        })
        .collect()
}

/// Format a "YYYY-MM-DD" date into a short human-readable form.
///
/// - Same year as `reference_year`: "Apr 8"
/// - Different year: "Apr 8, 2025"
fn format_date_short(date_str: &str, reference_year: i32) -> String {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];

    let parts: Vec<&str> = date_str.split('-').collect();
    if parts.len() != 3 {
        return date_str.to_string();
    }

    let year: i32 = parts[0].parse().unwrap_or(0);
    let month: usize = parts[1].parse().unwrap_or(1);
    let day: u32 = parts[2].parse().unwrap_or(1);

    let month_name = MONTHS.get(month.wrapping_sub(1)).unwrap_or(&"???");

    if year == reference_year {
        format!("{} {}", month_name, day)
    } else {
        format!("{} {}, {}", month_name, day, year)
    }
}

/// Compute the date range string from raw commits.
///
/// Returns `None` if commits is empty. Collapses to a single date when
/// earliest == latest. Uses "Mon DD" for same-year, "Mon DD, YYYY" otherwise.
fn format_commit_date_range(commits: &[ChangelogCommit]) -> Option<String> {
    if commits.is_empty() {
        return None;
    }

    let dates: Vec<&str> = commits.iter().map(|c| c.date.as_str()).collect();
    let earliest = dates.iter().min().unwrap();
    let latest = dates.iter().max().unwrap();

    let ref_year: i32 = latest
        .split('-')
        .next()
        .and_then(|y| y.parse().ok())
        .unwrap_or(0);

    if earliest == latest {
        Some(format_date_short(earliest, ref_year))
    } else {
        let start = format_date_short(earliest, ref_year);
        let end = format_date_short(latest, ref_year);
        Some(format!("{} – {}", start, end))
    }
}

/// Render a changelog as markdown.
pub fn render_markdown(changelog: &Changelog) -> String {
    let mut md = String::new();
    let date_range = format_commit_date_range(&changelog.raw_commits);

    if changelog.narrated {
        // Date range subtitle before entries
        if let Some(range) = &date_range {
            md.push_str(&format!("*{}*\n\n", range));
        }

        // LLM-narrated: clean prose entries
        for entry in &changelog.entries {
            md.push_str(&format!("## {}\n\n", entry.title));
            md.push_str(&entry.description);
            md.push_str("\n\n");
        }
    } else {
        // Structural: compact commit list with date range
        let date_suffix = date_range
            .as_ref()
            .map(|r| format!(" · {}", r))
            .unwrap_or_default();

        md.push_str(&format!(
            "*Recent activity on `{}`{} ({} commits)*\n\n",
            changelog.branch,
            date_suffix,
            changelog.raw_commits.len(),
        ));

        for entry in &changelog.entries {
            md.push_str(&format!("- **{}** — {}\n", entry.title, entry.description));
        }
        md.push('\n');
    }

    md
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_structural_entries() {
        let commits = vec![
            ChangelogCommit {
                hash: "abc123".into(),
                author: "Alice".into(),
                timestamp: 1712880000,
                date: "2024-04-12".into(),
                subject: "Added search feature".into(),
                files_changed: vec!["src/search.rs".into()],
            },
            ChangelogCommit {
                hash: "def456".into(),
                author: "Bob".into(),
                timestamp: 1712793600,
                date: "2024-04-11".into(),
                subject: "Fixed login bug".into(),
                files_changed: vec!["src/auth.rs".into()],
            },
        ];

        let entries = generate_structural_entries(&commits);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].title, "Added search feature");
        assert!(entries[0].description.contains("Alice"));
        assert!(entries[0].description.contains("2024-04-12"));
    }

    #[test]
    fn test_parse_changelog_response_valid() {
        let json = r#"{"entries": [{"title": "New search", "description": "Added full-text search."}]}"#;
        let entries = parse_changelog_response(json, &[]);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].title, "New search");
    }

    #[test]
    fn test_parse_changelog_response_with_fences() {
        let json = "```json\n{\"entries\": [{\"title\": \"Test\", \"description\": \"Desc\"}]}\n```";
        let entries = parse_changelog_response(json, &[]);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].title, "Test");
    }

    #[test]
    fn test_parse_changelog_response_invalid_fallback() {
        let commits = vec![ChangelogCommit {
            hash: "abc".into(),
            author: "Alice".into(),
            timestamp: 0,
            date: "2024-01-01".into(),
            subject: "Fallback commit".into(),
            files_changed: vec![],
        }];
        let entries = parse_changelog_response("not valid json", &commits);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].title, "Fallback commit");
    }

    #[test]
    fn test_render_markdown_structural() {
        let changelog = Changelog {
            entries: vec![ChangelogEntry {
                title: "Added search".into(),
                description: "Alice — 2024-04-12".into(),
            }],
            raw_commits: vec![ChangelogCommit {
                hash: "abc".into(),
                author: "Alice".into(),
                timestamp: 0,
                date: "2024-04-12".into(),
                subject: "Added search".into(),
                files_changed: vec![],
            }],
            branch: "main".into(),
            narrated: false,
        };

        let md = render_markdown(&changelog);
        assert!(md.contains("Recent activity on `main`"));
        assert!(md.contains("Apr 12"));
        assert!(md.contains("**Added search**"));
    }

    #[test]
    fn test_render_markdown_structural_date_range() {
        let changelog = Changelog {
            entries: vec![
                ChangelogEntry {
                    title: "Added search".into(),
                    description: "Alice — 2024-04-12".into(),
                },
                ChangelogEntry {
                    title: "Fixed bug".into(),
                    description: "Bob — 2024-04-08".into(),
                },
            ],
            raw_commits: vec![
                ChangelogCommit {
                    hash: "abc".into(),
                    author: "Alice".into(),
                    timestamp: 0,
                    date: "2024-04-12".into(),
                    subject: "Added search".into(),
                    files_changed: vec![],
                },
                ChangelogCommit {
                    hash: "def".into(),
                    author: "Bob".into(),
                    timestamp: 0,
                    date: "2024-04-08".into(),
                    subject: "Fixed bug".into(),
                    files_changed: vec![],
                },
            ],
            branch: "main".into(),
            narrated: false,
        };

        let md = render_markdown(&changelog);
        assert!(md.contains("Apr 8 – Apr 12"), "Should contain date range, got: {}", md);
    }

    #[test]
    fn test_render_markdown_narrated() {
        let changelog = Changelog {
            entries: vec![ChangelogEntry {
                title: "Full-text search added".into(),
                description: "The documentation site now includes search.".into(),
            }],
            raw_commits: vec![ChangelogCommit {
                hash: "abc".into(),
                author: "Alice".into(),
                timestamp: 0,
                date: "2024-04-10".into(),
                subject: "Added search".into(),
                files_changed: vec![],
            }],
            branch: "main".into(),
            narrated: true,
        };

        let md = render_markdown(&changelog);
        assert!(md.contains("*Apr 10*"), "Should contain date subtitle, got: {}", md);
        assert!(md.contains("## Full-text search added"));
        assert!(md.contains("documentation site"));
    }

    #[test]
    fn test_format_date_short_same_year() {
        assert_eq!(format_date_short("2026-04-08", 2026), "Apr 8");
        assert_eq!(format_date_short("2026-12-25", 2026), "Dec 25");
        assert_eq!(format_date_short("2026-01-01", 2026), "Jan 1");
    }

    #[test]
    fn test_format_date_short_different_year() {
        assert_eq!(format_date_short("2025-04-08", 2026), "Apr 8, 2025");
    }

    #[test]
    fn test_format_commit_date_range_empty() {
        assert_eq!(format_commit_date_range(&[]), None);
    }

    #[test]
    fn test_format_commit_date_range_single() {
        let commits = vec![ChangelogCommit {
            hash: "abc".into(),
            author: "Alice".into(),
            timestamp: 0,
            date: "2026-04-10".into(),
            subject: "Test".into(),
            files_changed: vec![],
        }];
        assert_eq!(format_commit_date_range(&commits), Some("Apr 10".to_string()));
    }

    #[test]
    fn test_format_commit_date_range_span() {
        let commits = vec![
            ChangelogCommit {
                hash: "a".into(),
                author: "A".into(),
                timestamp: 0,
                date: "2026-04-08".into(),
                subject: "First".into(),
                files_changed: vec![],
            },
            ChangelogCommit {
                hash: "b".into(),
                author: "B".into(),
                timestamp: 0,
                date: "2026-04-13".into(),
                subject: "Last".into(),
                files_changed: vec![],
            },
        ];
        assert_eq!(
            format_commit_date_range(&commits),
            Some("Apr 8 – Apr 13".to_string())
        );
    }

    #[test]
    fn test_format_commit_date_range_cross_year() {
        let commits = vec![
            ChangelogCommit {
                hash: "a".into(),
                author: "A".into(),
                timestamp: 0,
                date: "2025-12-28".into(),
                subject: "First".into(),
                files_changed: vec![],
            },
            ChangelogCommit {
                hash: "b".into(),
                author: "B".into(),
                timestamp: 0,
                date: "2026-01-03".into(),
                subject: "Last".into(),
                files_changed: vec![],
            },
        ];
        assert_eq!(
            format_commit_date_range(&commits),
            Some("Dec 28, 2025 – Jan 3".to_string())
        );
    }

    #[test]
    fn test_build_changelog_context() {
        let commits = vec![ChangelogCommit {
            hash: "abc123".repeat(7)[..40].to_string(),
            author: "Alice".into(),
            timestamp: 1712880000,
            date: "2024-04-12".into(),
            subject: "Added search".into(),
            files_changed: vec!["src/pulse/search.rs".into(), "src/pulse/site.rs".into()],
        }];

        let ctx = build_changelog_context(&commits, "feature/search");
        assert!(ctx.contains("Branch: feature/search"));
        assert!(ctx.contains("\"Added search\""));
        assert!(ctx.contains("Areas:"));
    }
}
