//! Git repository utilities for branch tracking
//!
//! This module provides helper functions for interacting with git repositories
//! to track branch state, detect uncommitted changes, and capture git metadata
//! for branch-aware indexing.

use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

/// Git repository state
#[derive(Debug, Clone)]
pub struct GitState {
    /// Current branch name (e.g., "main", "feature-x")
    pub branch: String,
    /// Current commit SHA (full 40-character hash)
    pub commit: String,
    /// Whether there are uncommitted changes (modified/added/deleted files)
    pub dirty: bool,
}

/// Check if the current directory is inside a git repository
pub fn is_git_repo(root: impl AsRef<Path>) -> bool {
    root.as_ref().join(".git").exists()
}

/// Check whether the `git` binary is available on PATH.
///
/// Probes `git --version` once per process and caches the result.
/// Returns `false` only when the OS reports the binary doesn't exist
/// (`io::ErrorKind::NotFound`). A `git` that spawns but exits non-zero
/// still counts as "available" so legitimate git errors propagate normally.
pub fn is_git_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| match Command::new("git").arg("--version").output() {
        Ok(_) => true,
        Err(e) => e.kind() != std::io::ErrorKind::NotFound,
    })
}

/// Get the current git branch name
///
/// Returns the branch name (e.g., "main", "feature-x") or "HEAD" if in detached HEAD state.
pub fn get_current_branch(root: impl AsRef<Path>) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root.as_ref())
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .context("Failed to execute git rev-parse")?;

    if !output.status.success() {
        anyhow::bail!(
            "git rev-parse failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let branch = String::from_utf8(output.stdout)
        .context("Invalid UTF-8 in branch name")?
        .trim()
        .to_string();

    Ok(branch)
}

/// Get the current commit SHA
///
/// Returns the full 40-character commit hash for HEAD.
pub fn get_current_commit(root: impl AsRef<Path>) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root.as_ref())
        .args(["rev-parse", "HEAD"])
        .output()
        .context("Failed to execute git rev-parse HEAD")?;

    if !output.status.success() {
        anyhow::bail!(
            "git rev-parse HEAD failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let commit = String::from_utf8(output.stdout)
        .context("Invalid UTF-8 in commit SHA")?
        .trim()
        .to_string();

    Ok(commit)
}

/// Check if there are uncommitted changes in the working tree
///
/// Returns true if there are any modified, added, or deleted files.
/// Uses `git status --porcelain` which is designed for scripting.
pub fn has_uncommitted_changes(root: impl AsRef<Path>) -> Result<bool> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root.as_ref())
        .args(["status", "--porcelain"])
        .output()
        .context("Failed to execute git status")?;

    if !output.status.success() {
        anyhow::bail!(
            "git status failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // If output is empty, working tree is clean
    // If output has any content, there are uncommitted changes
    let has_changes = !output.stdout.is_empty();

    Ok(has_changes)
}

/// Files that differ between the working tree and the last commit.
///
/// Lists are capped (see [`MAX_REPORTED_PATHS`]) so a fresh checkout cannot produce a
/// megabyte of JSON in an MCP response; `truncated` says when that happened and the
/// `*_count` fields carry the real totals.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorktreeChanges {
    /// Tracked files with edits.
    pub modified: Vec<String>,
    /// Untracked files, and files staged as new.
    pub added: Vec<String>,
    /// Files removed from disk.
    pub deleted: Vec<String>,
    pub modified_count: usize,
    pub added_count: usize,
    pub deleted_count: usize,
    /// Whether any list was cut short.
    pub truncated: bool,
}

impl WorktreeChanges {
    /// Whether the working tree differs from the commit the index was built at.
    pub fn is_empty(&self) -> bool {
        self.modified_count == 0 && self.added_count == 0 && self.deleted_count == 0
    }

    /// Total changed paths, counting every category.
    pub fn total(&self) -> usize {
        self.modified_count + self.added_count + self.deleted_count
    }

    /// Whether any changed path is one the caller cares about.
    pub fn any<F: Fn(&str) -> bool>(&self, pred: F) -> bool {
        self.modified
            .iter()
            .chain(&self.added)
            .chain(&self.deleted)
            .any(|p| pred(p))
    }
}

/// How many paths per category a [`WorktreeChanges`] will name.
const MAX_REPORTED_PATHS: usize = 100;

/// List working-tree changes against HEAD.
///
/// This is what makes freshness honest. Before 1.7.2 the check compared
/// `git rev-parse HEAD` to the indexed commit and then sampled the mtimes of the
/// first TEN indexed files — so an edit to any other file, any untracked file, and
/// any deletion all reported `fresh` with `can_trust_results: true`. That is the
/// primary agent workflow: edit, then search, before committing.
///
/// Flags, each load-bearing:
/// * `-z` — porcelain v1 C-quotes paths containing spaces, non-ASCII or backslashes.
///   `-z` emits raw NUL-separated paths and never quotes, so no unescaping is needed.
/// * `--untracked-files=all` — list new files individually rather than just their
///   directory, otherwise a new file inside an existing directory is invisible.
/// * `--no-renames` — decompose `R old -> new` into a delete plus an add, which is
///   exactly what an index must do with a rename, and removes all rename parsing.
/// * `--ignored=no` — `.gitignore`d paths are not indexed, so they cannot make the
///   index stale.
///
/// `keep` filters paths down to those the indexer would actually index. Without it,
/// editing `README.md` or anything under `target/` would mark the index permanently
/// stale, and the cure would be worse than the disease.
pub fn get_worktree_changes<F>(root: impl AsRef<Path>, keep: F) -> Result<WorktreeChanges>
where
    F: Fn(&str) -> bool,
{
    let output = Command::new("git")
        .arg("-C")
        .arg(root.as_ref())
        .args([
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--no-renames",
            "--ignored=no",
        ])
        .output()
        .context("Failed to execute git status")?;

    if !output.status.success() {
        anyhow::bail!(
            "git status failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let mut changes = WorktreeChanges::default();

    // Records are NUL-terminated: "XY <path>\0". With --no-renames there is never a
    // second path in a record, so a plain split is safe.
    for record in output.stdout.split(|b| *b == 0) {
        if record.len() < 4 {
            continue;
        }
        let text = String::from_utf8_lossy(record);
        let (status, path) = text.split_at(3);
        let path = path.trim();
        if path.is_empty() || !keep(path) {
            continue;
        }

        let mut bytes = status.bytes();
        let x = bytes.next().unwrap_or(b' ');
        let y = bytes.next().unwrap_or(b' ');

        // A delete in either column wins: the indexed row must go regardless of
        // whatever else the file did on the way there.
        let (bucket, count) = if x == b'D' || y == b'D' {
            (&mut changes.deleted, &mut changes.deleted_count)
        } else if x == b'?' || x == b'A' {
            (&mut changes.added, &mut changes.added_count)
        } else {
            (&mut changes.modified, &mut changes.modified_count)
        };

        *count += 1;
        if bucket.len() < MAX_REPORTED_PATHS {
            bucket.push(path.to_string());
        } else {
            changes.truncated = true;
        }
    }

    Ok(changes)
}

/// Get complete git state for the current repository
///
/// This is a convenience function that captures branch, commit, and dirty state
/// in one call, which is more efficient than calling each function separately.
pub fn get_git_state(root: impl AsRef<Path>) -> Result<GitState> {
    let root = root.as_ref();

    if !is_git_repo(root) {
        anyhow::bail!("Not a git repository");
    }

    let branch = get_current_branch(root)?;
    let commit = get_current_commit(root)?;
    let dirty = has_uncommitted_changes(root)?;

    Ok(GitState {
        branch,
        commit,
        dirty,
    })
}

/// Get git state, or return None if not in a git repository
///
/// This is useful for indexing non-git projects where we fall back to a default branch.
pub fn get_git_state_optional(root: impl AsRef<Path>) -> Result<Option<GitState>> {
    if !is_git_repo(&root) {
        return Ok(None);
    }

    match get_git_state(root) {
        Ok(state) => Ok(Some(state)),
        Err(e) => {
            log::warn!("Failed to get git state: {}", e);
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_git_repo() {
        // This test project should be a git repo
        assert!(is_git_repo("."));

        // /tmp should not be a git repo
        assert!(!is_git_repo("/tmp"));
    }

    #[test]
    fn test_get_current_branch() {
        // Should return a branch name (or HEAD if detached)
        let branch = get_current_branch(".").unwrap();
        assert!(!branch.is_empty());
        log::info!("Current branch: {}", branch);
    }

    #[test]
    fn test_get_current_commit() {
        // Should return a 40-character SHA
        let commit = get_current_commit(".").unwrap();
        assert_eq!(commit.len(), 40);
        assert!(commit.chars().all(|c| c.is_ascii_hexdigit()));
        log::info!("Current commit: {}", commit);
    }

    #[test]
    fn test_has_uncommitted_changes() {
        // Can't predict if there are changes, but function should not error
        let has_changes = has_uncommitted_changes(".").unwrap();
        log::info!("Has uncommitted changes: {}", has_changes);
    }

    #[test]
    fn test_get_git_state() {
        let state = get_git_state(".").unwrap();
        assert!(!state.branch.is_empty());
        assert_eq!(state.commit.len(), 40);
        log::info!("Git state: {:?}", state);
    }
}
