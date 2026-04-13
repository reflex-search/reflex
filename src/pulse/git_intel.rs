//! Git Intelligence: Timeline page with development activity history
//!
//! Extracts git log data to show recent activity, contributor patterns,
//! file churn, and weekly summaries.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

/// Complete git intelligence data
#[derive(Debug, Clone)]
pub struct GitIntel {
    pub commits: Vec<CommitInfo>,
    pub contributors: Vec<Contributor>,
    pub churn: Vec<FileChurn>,
    pub weekly_summaries: Vec<WeekSummary>,
    pub module_activity: Vec<ModuleActivity>,
    pub narration: Option<String>,
}

/// A single commit
#[derive(Debug, Clone)]
pub struct CommitInfo {
    pub hash: String,
    pub author: String,
    pub email: String,
    pub timestamp: i64,
    pub subject: String,
}

/// Contributor stats
#[derive(Debug, Clone)]
pub struct Contributor {
    pub name: String,
    pub email: String,
    pub commit_count: usize,
}

/// File churn: how often a file changes
#[derive(Debug, Clone)]
pub struct FileChurn {
    pub path: String,
    pub change_count: usize,
    pub primary_author: String,
}

/// Weekly activity summary
#[derive(Debug, Clone)]
pub struct WeekSummary {
    pub week_start: String,
    pub commit_count: usize,
    pub files_changed: usize,
    pub contributors: Vec<String>,
    pub top_modules: Vec<String>,
}

/// Per-module activity
#[derive(Debug, Clone)]
pub struct ModuleActivity {
    pub module_path: String,
    pub commit_count: usize,
    pub files_changed: usize,
    pub primary_contributor: String,
}

/// Extract git log data for the last 6 months
pub fn extract_git_intel(root: impl AsRef<Path>) -> Result<GitIntel> {
    let root = root.as_ref();

    // Check if this is a git repo
    if !root.join(".git").exists() {
        return Ok(GitIntel {
            commits: vec![],
            contributors: vec![],
            churn: vec![],
            weekly_summaries: vec![],
            module_activity: vec![],
            narration: None,
        });
    }

    let commits = extract_commits(root)?;
    let contributors = compute_contributors(&commits);
    let churn = extract_file_churn(root)?;
    let weekly_summaries = compute_weekly_summaries(root, &commits)?;
    let module_activity = compute_module_activity(&churn);

    Ok(GitIntel {
        commits,
        contributors,
        churn,
        weekly_summaries,
        module_activity,
        narration: None,
    })
}

/// Parse git log into commits
fn extract_commits(root: &Path) -> Result<Vec<CommitInfo>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["log", "--format=%H|%an|%ae|%at|%s", "--since=6 months ago"])
        .output()
        .context("Failed to run git log")?;

    if !output.status.success() {
        return Ok(vec![]);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let commits: Vec<CommitInfo> = stdout.lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.splitn(5, '|').collect();
            if parts.len() < 5 { return None; }
            Some(CommitInfo {
                hash: parts[0].to_string(),
                author: parts[1].to_string(),
                email: parts[2].to_string(),
                timestamp: parts[3].parse().unwrap_or(0),
                subject: parts[4].to_string(),
            })
        })
        .collect();

    Ok(commits)
}

/// Compute contributor stats from commits (deduplicated by name)
fn compute_contributors(commits: &[CommitInfo]) -> Vec<Contributor> {
    let mut by_name: HashMap<String, (String, usize)> = HashMap::new();
    for commit in commits {
        let entry = by_name.entry(commit.author.clone())
            .or_insert_with(|| (commit.email.clone(), 0));
        entry.1 += 1;
    }

    let mut contributors: Vec<Contributor> = by_name.into_iter()
        .map(|(name, (email, count))| Contributor {
            name,
            email,
            commit_count: count,
        })
        .collect();

    contributors.sort_by(|a, b| b.commit_count.cmp(&a.commit_count));
    contributors
}

/// Extract file change frequency using git log --name-only
fn extract_file_churn(root: &Path) -> Result<Vec<FileChurn>> {
    // Get file change counts with author info
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["log", "--format=%an", "--name-only", "--since=6 months ago"])
        .output()
        .context("Failed to run git log --name-only")?;

    if !output.status.success() {
        return Ok(vec![]);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut file_counts: HashMap<String, usize> = HashMap::new();
    let mut file_authors: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let mut current_author = String::new();

    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Lines without path separators and not starting with spaces are author names
        // (from the %an format), file paths contain / or .
        if !trimmed.contains('/') && !trimmed.contains('.') && !trimmed.starts_with(' ') {
            current_author = trimmed.to_string();
        } else if !current_author.is_empty() {
            *file_counts.entry(trimmed.to_string()).or_default() += 1;
            *file_authors.entry(trimmed.to_string())
                .or_default()
                .entry(current_author.clone())
                .or_default() += 1;
        }
    }

    let mut churn: Vec<FileChurn> = file_counts.into_iter()
        .map(|(path, count)| {
            let primary = file_authors.get(&path)
                .and_then(|authors| authors.iter().max_by_key(|(_, c)| *c))
                .map(|(name, _)| name.clone())
                .unwrap_or_default();
            FileChurn {
                path,
                change_count: count,
                primary_author: primary,
            }
        })
        .collect();

    churn.sort_by(|a, b| b.change_count.cmp(&a.change_count));
    churn.truncate(50); // Top 50 most-changed files

    Ok(churn)
}

/// Compute weekly summaries from commits and file changes
fn compute_weekly_summaries(root: &Path, commits: &[CommitInfo]) -> Result<Vec<WeekSummary>> {
    if commits.is_empty() {
        return Ok(vec![]);
    }

    // Group commits by ISO week
    let mut weeks: HashMap<String, (usize, Vec<String>, HashMap<String, usize>)> = HashMap::new();

    for commit in commits {
        // Convert timestamp to week start date (Monday)
        let ts = commit.timestamp;
        // Simple week computation: round down to nearest Monday
        let days_since_epoch = ts / 86400;
        // 1970-01-01 was a Thursday (day 4), so Monday of that week = day -3
        let week_day = ((days_since_epoch + 3) % 7) as i64; // 0=Monday
        let monday = days_since_epoch - week_day;
        let week_key = format!("{}", monday); // Use epoch-day of Monday as key

        let entry = weeks.entry(week_key).or_insert_with(|| (0, vec![], HashMap::new()));
        entry.0 += 1;
        if !entry.1.contains(&commit.author) {
            entry.1.push(commit.author.clone());
        }
    }

    // Get file changes per week (using git log with date ranges)
    // Instead of running git for each week, use the commit data we already have
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["log", "--format=%at", "--name-only", "--since=6 months ago"])
        .output();

    let mut week_files: HashMap<String, HashMap<String, bool>> = HashMap::new();

    if let Ok(output) = output {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut current_ts: i64 = 0;
            for line in stdout.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() { continue; }
                if let Ok(ts) = trimmed.parse::<i64>() {
                    current_ts = ts;
                } else if current_ts > 0 {
                    let days = current_ts / 86400;
                    let week_day = ((days + 3) % 7) as i64;
                    let monday = days - week_day;
                    let week_key = format!("{}", monday);
                    week_files.entry(week_key)
                        .or_default()
                        .insert(trimmed.to_string(), true);
                }
            }
        }
    }

    // Build summaries
    let mut summaries: Vec<WeekSummary> = weeks.into_iter()
        .map(|(week_key, (count, contributors, _))| {
            let files_changed = week_files.get(&week_key).map(|f| f.len()).unwrap_or(0);

            // Compute top modules from changed files
            let mut module_counts: HashMap<String, usize> = HashMap::new();
            if let Some(files) = week_files.get(&week_key) {
                for file in files.keys() {
                    if let Some(module) = file.split('/').next() {
                        *module_counts.entry(module.to_string()).or_default() += 1;
                    }
                }
            }
            let mut top_modules: Vec<(String, usize)> = module_counts.into_iter().collect();
            top_modules.sort_by(|a, b| b.1.cmp(&a.1));
            let top_modules: Vec<String> = top_modules.into_iter().take(3).map(|(m, _)| m).collect();

            // Convert monday epoch-days back to date string
            let monday_days: i64 = week_key.parse().unwrap_or(0);
            let monday_ts = monday_days * 86400;
            let week_start = epoch_to_date_string(monday_ts);

            WeekSummary {
                week_start,
                commit_count: count,
                files_changed,
                contributors,
                top_modules,
            }
        })
        .collect();

    summaries.sort_by(|a, b| b.week_start.cmp(&a.week_start));
    summaries.truncate(12); // Last 12 weeks

    Ok(summaries)
}

/// Compute per-module activity from file churn
fn compute_module_activity(churn: &[FileChurn]) -> Vec<ModuleActivity> {
    let mut by_module: HashMap<String, (usize, usize, HashMap<String, usize>)> = HashMap::new();

    for file in churn {
        let module = file.path.split('/').next().unwrap_or("root").to_string();
        let entry = by_module.entry(module).or_default();
        entry.0 += file.change_count;
        entry.1 += 1;
        *entry.2.entry(file.primary_author.clone()).or_default() += file.change_count;
    }

    let mut activity: Vec<ModuleActivity> = by_module.into_iter()
        .map(|(module, (commits, files, authors))| {
            let primary = authors.into_iter()
                .max_by_key(|(_, c)| *c)
                .map(|(name, _)| name)
                .unwrap_or_default();
            ModuleActivity {
                module_path: module,
                commit_count: commits,
                files_changed: files,
                primary_contributor: primary,
            }
        })
        .collect();

    activity.sort_by(|a, b| b.commit_count.cmp(&a.commit_count));
    activity
}

/// Convert epoch seconds to YYYY-MM-DD string
pub fn epoch_to_date_string(epoch_secs: i64) -> String {
    // Simple date calculation without external deps
    let days = epoch_secs / 86400;
    let (year, month, day) = days_to_ymd(days);
    format!("{:04}-{:02}-{:02}", year, month, day)
}

/// Convert days since epoch to (year, month, day)
pub fn days_to_ymd(days: i64) -> (i64, u32, u32) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Build structural context for LLM narration
pub fn build_timeline_context(data: &GitIntel) -> String {
    let mut ctx = String::new();

    ctx.push_str(&format!("Total commits (last 6 months): {}\n", data.commits.len()));
    ctx.push_str(&format!("Contributors: {}\n\n", data.contributors.len()));

    // Top contributors
    ctx.push_str("Top contributors:\n");
    for c in data.contributors.iter().take(10) {
        ctx.push_str(&format!("- {} ({} commits)\n", c.name, c.commit_count));
    }
    ctx.push('\n');

    // Hottest files
    ctx.push_str("Most-changed files:\n");
    for f in data.churn.iter().take(15) {
        ctx.push_str(&format!("- {} ({} changes, primarily by {})\n", f.path, f.change_count, f.primary_author));
    }
    ctx.push('\n');

    // Module activity
    ctx.push_str("Module activity:\n");
    for m in data.module_activity.iter().take(10) {
        ctx.push_str(&format!("- {} ({} changes across {} files, led by {})\n",
            m.module_path, m.commit_count, m.files_changed, m.primary_contributor));
    }
    ctx.push('\n');

    // Recent weeks
    ctx.push_str("Recent weekly activity:\n");
    for w in data.weekly_summaries.iter().take(4) {
        ctx.push_str(&format!("- Week of {}: {} commits, {} files changed by {} contributors\n",
            w.week_start, w.commit_count, w.files_changed, w.contributors.len()));
        if !w.top_modules.is_empty() {
            ctx.push_str(&format!("  Most active: {}\n", w.top_modules.join(", ")));
        }
    }

    ctx
}

/// Render timeline data as markdown
pub fn render_timeline_markdown(data: &GitIntel) -> String {
    let mut md = String::new();

    if data.commits.is_empty() {
        md.push_str("*No git history available.*\n");
        return md;
    }

    // Narration
    if let Some(ref narration) = data.narration {
        md.push_str(narration);
        md.push_str("\n\n");
    }

    // Activity chart (simple text-based bar chart using mermaid)
    if !data.weekly_summaries.is_empty() {
        md.push_str("## Weekly Activity\n\n");
        md.push_str("{% mermaid() %}\nxychart-beta\n");
        md.push_str("    title \"Commits per Week\"\n");
        md.push_str("    x-axis [");
        let weeks: Vec<&WeekSummary> = data.weekly_summaries.iter().rev().collect();
        let labels: Vec<String> = weeks.iter()
            .map(|w| {
                // Just use MM-DD for compact labels
                if w.week_start.len() >= 10 {
                    format!("\"{}\"", &w.week_start[5..10])
                } else {
                    format!("\"{}\"", w.week_start)
                }
            })
            .collect();
        md.push_str(&labels.join(", "));
        md.push_str("]\n");
        md.push_str("    y-axis \"Commits\"\n");
        md.push_str("    bar [");
        let counts: Vec<String> = weeks.iter().map(|w| w.commit_count.to_string()).collect();
        md.push_str(&counts.join(", "));
        md.push_str("]\n");
        md.push_str("{% end %}\n\n");
    }

    // Contributors table
    if !data.contributors.is_empty() {
        md.push_str("## Contributors\n\n");
        md.push_str("| Author | Commits |\n|---|---|\n");
        for c in data.contributors.iter().take(15) {
            md.push_str(&format!("| {} | {} |\n", c.name, c.commit_count));
        }
        md.push('\n');
    }

    // Hot files
    if !data.churn.is_empty() {
        md.push_str("## Most-Changed Files\n\n");
        md.push_str("| File | Changes | Primary Author |\n|---|---|---|\n");
        for f in data.churn.iter().take(20) {
            md.push_str(&format!("| `{}` | {} | {} |\n", f.path, f.change_count, f.primary_author));
        }
        md.push('\n');
    }

    // Module activity
    if !data.module_activity.is_empty() {
        md.push_str("## Module Activity\n\n");
        md.push_str("| Module | Changes | Files | Primary Contributor |\n|---|---|---|---|\n");
        for m in &data.module_activity {
            md.push_str(&format!("| `{}` | {} | {} | {} |\n",
                m.module_path, m.commit_count, m.files_changed, m.primary_contributor));
        }
        md.push('\n');
    }

    md
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epoch_to_date_string() {
        // 2024-01-01 00:00:00 UTC = 1704067200
        assert_eq!(epoch_to_date_string(1704067200), "2024-01-01");
    }

    #[test]
    fn test_days_to_ymd() {
        let (y, m, d) = days_to_ymd(0); // 1970-01-01
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn test_compute_contributors() {
        let commits = vec![
            CommitInfo { hash: "a".into(), author: "Alice".into(), email: "a@x.com".into(), timestamp: 1, subject: "test".into() },
            CommitInfo { hash: "b".into(), author: "Alice".into(), email: "a@x.com".into(), timestamp: 2, subject: "test2".into() },
            CommitInfo { hash: "c".into(), author: "Bob".into(), email: "b@x.com".into(), timestamp: 3, subject: "test3".into() },
        ];
        let contributors = compute_contributors(&commits);
        assert_eq!(contributors.len(), 2);
        assert_eq!(contributors[0].name, "Alice");
        assert_eq!(contributors[0].commit_count, 2);
    }

    #[test]
    fn test_render_empty_timeline() {
        let data = GitIntel {
            commits: vec![],
            contributors: vec![],
            churn: vec![],
            weekly_summaries: vec![],
            module_activity: vec![],
            narration: None,
        };
        let md = render_timeline_markdown(&data);
        assert!(md.contains("No git history"));
    }
}
