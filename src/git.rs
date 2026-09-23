//! Git repository utilities for branch tracking
//!
//! This module provides helper functions for interacting with git repositories
//! to track branch state, detect uncommitted changes, and capture git metadata
//! for branch-aware indexing.

use anyhow::{Context, Result};
use std::collections::HashSet;
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
    /// Every path `git status` lists (modified, added, deleted, untracked),
    /// uncapped. The indexer records these as `dirty_at_index`.
    pub dirty_paths: HashSet<String>,
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

/// The current branch, read from `.git/HEAD` without spawning `git`.
///
/// `git rev-parse --abbrev-ref HEAD` costs a process spawn (2–5 ms) and ran on
/// every `--symbols` query. The answer is one small file: `ref: refs/heads/<name>`
/// on a branch, or a bare SHA when detached (`HEAD`, matching `--abbrev-ref`).
/// Worktrees and submodules keep a `.git` *file* holding `gitdir: <path>`, which is
/// followed one level. `None` when there is no readable HEAD, in which case the
/// caller falls back to what the indexer recorded (`_default`).
pub fn read_head_branch(root: impl AsRef<Path>) -> Option<String> {
    let mut git_dir = root.as_ref().join(".git");
    if git_dir.is_file() {
        let text = std::fs::read_to_string(&git_dir).ok()?;
        let target = text.strip_prefix("gitdir:")?.trim();
        let target = Path::new(target);
        git_dir = if target.is_absolute() {
            target.to_path_buf()
        } else {
            root.as_ref().join(target)
        };
    }
    let head = std::fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let head = head.trim();
    if let Some(reference) = head.strip_prefix("ref:") {
        let reference = reference.trim();
        let name = reference.strip_prefix("refs/heads/").unwrap_or(reference);
        return Some(name.to_string());
    }
    if !head.is_empty() && head.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Some("HEAD".to_string());
    }
    None
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
pub const MAX_REPORTED_PATHS: usize = 100;

impl WorktreeChanges {
    /// Count a path in a category, keeping the named list under the cap.
    pub fn push_modified(&mut self, path: &str) {
        self.modified_count += 1;
        if self.modified.len() < MAX_REPORTED_PATHS {
            self.modified.push(path.to_string());
        } else {
            self.truncated = true;
        }
    }

    pub fn push_added(&mut self, path: &str) {
        self.added_count += 1;
        if self.added.len() < MAX_REPORTED_PATHS {
            self.added.push(path.to_string());
        } else {
            self.truncated = true;
        }
    }

    pub fn push_deleted(&mut self, path: &str) {
        self.deleted_count += 1;
        if self.deleted.len() < MAX_REPORTED_PATHS {
            self.deleted.push(path.to_string());
        } else {
            self.truncated = true;
        }
    }

    /// Sort every list so the response is deterministic whatever order the
    /// candidates arrived in.
    pub fn sort(&mut self) {
        self.modified.sort();
        self.added.sort();
        self.deleted.sort();
    }
}

/// One `git status --porcelain=v1 -z` record: the two status columns and the path.
fn porcelain_records(root: &Path) -> Result<Vec<(u8, u8, String)>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
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

    let mut records = Vec::new();
    // Records are NUL-terminated: "XY <path>\0". With --no-renames there is never a
    // second path in a record, so a plain split is safe.
    for record in output.stdout.split(|b| *b == 0) {
        if record.len() < 4 {
            continue;
        }
        let text = String::from_utf8_lossy(record);
        let (status, path) = text.split_at(3);
        let path = path.trim();
        if path.is_empty() {
            continue;
        }
        let mut bytes = status.bytes();
        let x = bytes.next().unwrap_or(b' ');
        let y = bytes.next().unwrap_or(b' ');
        records.push((x, y, path.to_string()));
    }
    Ok(records)
}

/// Every path `git status` lists, uncapped and unfiltered.
///
/// The content-based freshness check uses this only as a CANDIDATE set: each path
/// is then confirmed against the indexed fingerprint, so a file git calls modified
/// whose bytes the index already holds is not stale.
pub fn changed_paths(root: impl AsRef<Path>) -> Result<HashSet<String>> {
    Ok(porcelain_records(root.as_ref())?
        .into_iter()
        .map(|(_, _, p)| p)
        .collect())
}

/// Paths that differ between two commits (`git diff --name-only`).
///
/// When HEAD has moved since indexing, these are the tracked paths whose content
/// may differ from what was indexed; each is then confirmed by fingerprint, so a
/// commit of already-indexed content does not make the index stale.
pub fn diff_names(root: impl AsRef<Path>, from: &str, to: &str) -> Result<Vec<String>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root.as_ref())
        .args(["diff", "--name-only", "-z", "--no-renames", from, to])
        .output()
        .context("Failed to execute git diff")?;
    if !output.status.success() {
        anyhow::bail!(
            "git diff {}..{} failed: {}",
            from,
            to,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(output
        .stdout
        .split(|b| *b == 0)
        .filter(|p| !p.is_empty())
        .map(|p| String::from_utf8_lossy(p).into_owned())
        .collect())
}

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
    let mut changes = WorktreeChanges::default();

    for (x, y, path) in porcelain_records(root.as_ref())? {
        if !keep(&path) {
            continue;
        }
        // A delete in either column wins: the indexed row must go regardless of
        // whatever else the file did on the way there.
        if x == b'D' || y == b'D' {
            changes.push_deleted(&path);
        } else if x == b'?' || x == b'A' {
            changes.push_added(&path);
        } else {
            changes.push_modified(&path);
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
    let dirty_paths = changed_paths(root)?;

    Ok(GitState {
        branch,
        commit,
        dirty: !dirty_paths.is_empty(),
        dirty_paths,
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
