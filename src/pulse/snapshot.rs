//! Snapshot engine: point-in-time capture of structural index state
//!
//! Snapshots capture the structural facts needed for diffing: file metadata,
//! dependency edges, and aggregate metrics. Each snapshot is a standalone SQLite
//! database under `.reflex/snapshots/`.

use anyhow::{Context, Result};
use chrono::{Datelike, Local};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::cache::CacheManager;
use crate::git;

/// Information about a snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotInfo {
    /// Unique identifier (timestamp-based filename without extension)
    pub id: String,
    /// Full path to the snapshot database
    pub path: PathBuf,
    /// When the snapshot was created
    pub timestamp: String,
    /// Git branch at snapshot time (if available)
    pub git_branch: Option<String>,
    /// Git commit SHA at snapshot time (if available)
    pub git_commit: Option<String>,
    /// Reflex version that created the snapshot
    pub reflex_version: String,
    /// Number of files in the snapshot
    pub file_count: usize,
    /// Total line count across all files
    pub total_lines: usize,
    /// Number of dependency edges
    pub edge_count: usize,
    /// Database file size in bytes
    pub size_bytes: u64,
}

/// Report from garbage collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcReport {
    pub snapshots_before: usize,
    pub snapshots_after: usize,
    pub removed: usize,
    pub corrupted_removed: usize,
    pub space_freed_bytes: u64,
}

/// Get the snapshots directory for a cache
pub fn get_snapshots_dir(cache: &CacheManager) -> PathBuf {
    cache.path().join("snapshots")
}

/// Create a snapshot of the current index state
///
/// Copies structural data from meta.db into a standalone snapshot database.
/// Returns info about the created snapshot.
pub fn create_snapshot(cache: &CacheManager) -> Result<SnapshotInfo> {
    let snapshots_dir = get_snapshots_dir(cache);
    std::fs::create_dir_all(&snapshots_dir)
        .context("Failed to create snapshots directory")?;

    let now = Local::now();
    let timestamp = now.format("%Y%m%d_%H%M%S").to_string();
    let snapshot_path = snapshots_dir.join(format!("{}.db", timestamp));

    // Check meta.db exists
    let meta_db_path = cache.path().join("meta.db");
    if !meta_db_path.exists() {
        anyhow::bail!("No index found. Run `rfx index` first.");
    }

    // Create snapshot database
    let conn = Connection::open(&snapshot_path)
        .context("Failed to create snapshot database")?;

    // Enable WAL mode for better concurrent access
    conn.execute_batch("PRAGMA journal_mode=WAL;")?;

    // Create snapshot schema
    conn.execute_batch(
        "CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            language TEXT,
            line_count INTEGER DEFAULT 0
        );

        CREATE TABLE dependency_edges (
            source_file_id INTEGER NOT NULL,
            target_file_id INTEGER NOT NULL,
            import_type TEXT NOT NULL
        );

        CREATE TABLE metrics (
            module_path TEXT PRIMARY KEY,
            file_count INTEGER NOT NULL,
            total_lines INTEGER NOT NULL
        );

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        -- Compatibility view so DependencyIndex methods work transparently
        CREATE VIEW file_dependencies AS
        SELECT source_file_id AS file_id,
               '' AS imported_path,
               target_file_id AS resolved_file_id,
               import_type,
               0 AS line_number,
               NULL AS imported_symbols
        FROM dependency_edges;

        -- Empty exports view for schema compatibility
        CREATE VIEW file_exports AS
        SELECT 0 AS id, 0 AS file_id, NULL AS exported_symbol,
               '' AS source_path, NULL AS resolved_source_id, 0 AS line_number
        WHERE 0;

        CREATE INDEX idx_dep_edges_source ON dependency_edges(source_file_id);
        CREATE INDEX idx_dep_edges_target ON dependency_edges(target_file_id);
        CREATE INDEX idx_files_path ON files(path);"
    )?;

    // Attach meta.db and copy data
    conn.execute(
        "ATTACH DATABASE ?1 AS source",
        [meta_db_path.to_str().unwrap()],
    )?;

    // Copy files
    conn.execute(
        "INSERT INTO files (id, path, language, line_count)
         SELECT id, path, language, line_count FROM source.files",
        [],
    )?;

    // Copy dependency edges (projected from file_dependencies)
    conn.execute(
        "INSERT INTO dependency_edges (source_file_id, target_file_id, import_type)
         SELECT file_id, resolved_file_id, import_type
         FROM source.file_dependencies
         WHERE resolved_file_id IS NOT NULL",
        [],
    )?;

    // Compute module metrics by aggregating files by top-level directory
    conn.execute(
        "INSERT INTO metrics (module_path, file_count, total_lines)
         SELECT
             CASE
                 WHEN INSTR(path, '/') > 0 THEN SUBSTR(path, 1, INSTR(path, '/') - 1)
                 ELSE '.'
             END AS module_path,
             COUNT(*) AS file_count,
             COALESCE(SUM(line_count), 0) AS total_lines
         FROM files
         GROUP BY module_path",
        [],
    )?;

    // Detach source
    conn.execute("DETACH DATABASE source", [])?;

    // Write metadata
    let git_state = git::get_git_state_optional(".")
        .unwrap_or(None);

    let metadata = vec![
        ("timestamp", now.to_rfc3339()),
        ("reflex_version", env!("CARGO_PKG_VERSION").to_string()),
        ("schema_version", "1".to_string()),
    ];

    for (key, value) in &metadata {
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            rusqlite::params![key, value],
        )?;
    }

    if let Some(ref state) = git_state {
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('git_branch', ?1)",
            [&state.branch],
        )?;
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('git_commit', ?1)",
            [&state.commit],
        )?;
    }

    // Gather stats for the returned info
    let file_count: usize = conn.query_row(
        "SELECT COUNT(*) FROM files", [], |row| row.get(0)
    )?;
    let total_lines: usize = conn.query_row(
        "SELECT COALESCE(SUM(line_count), 0) FROM files", [], |row| row.get(0)
    )?;
    let edge_count: usize = conn.query_row(
        "SELECT COUNT(*) FROM dependency_edges", [], |row| row.get(0)
    )?;

    // Close connection before getting file size
    drop(conn);

    let size_bytes = std::fs::metadata(&snapshot_path)
        .map(|m| m.len())
        .unwrap_or(0);

    Ok(SnapshotInfo {
        id: timestamp.clone(),
        path: snapshot_path,
        timestamp: now.to_rfc3339(),
        git_branch: git_state.as_ref().map(|s| s.branch.clone()),
        git_commit: git_state.as_ref().map(|s| s.commit.clone()),
        reflex_version: env!("CARGO_PKG_VERSION").to_string(),
        file_count,
        total_lines,
        edge_count,
        size_bytes,
    })
}

/// List all available snapshots, sorted by timestamp (newest first)
pub fn list_snapshots(cache: &CacheManager) -> Result<Vec<SnapshotInfo>> {
    let snapshots_dir = get_snapshots_dir(cache);
    if !snapshots_dir.exists() {
        return Ok(Vec::new());
    }

    let mut snapshots = Vec::new();

    for entry in std::fs::read_dir(&snapshots_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().is_some_and(|ext| ext == "db") {
            match read_snapshot_info(&path) {
                Ok(info) => snapshots.push(info),
                Err(e) => {
                    log::warn!("Skipping corrupted snapshot {:?}: {}", path, e);
                }
            }
        }
    }

    // Sort newest first
    snapshots.sort_by(|a, b| b.id.cmp(&a.id));

    Ok(snapshots)
}

/// Get info about a specific snapshot by ID
pub fn get_snapshot(cache: &CacheManager, id: &str) -> Result<SnapshotInfo> {
    let snapshot_path = get_snapshots_dir(cache).join(format!("{}.db", id));
    if !snapshot_path.exists() {
        anyhow::bail!("Snapshot '{}' not found", id);
    }
    read_snapshot_info(&snapshot_path)
}

/// Delete a specific snapshot
pub fn delete_snapshot(cache: &CacheManager, id: &str) -> Result<()> {
    let snapshot_path = get_snapshots_dir(cache).join(format!("{}.db", id));
    if snapshot_path.exists() {
        std::fs::remove_file(&snapshot_path)
            .context("Failed to delete snapshot")?;
    }
    Ok(())
}

/// Validate a snapshot database integrity
pub fn validate_snapshot(path: &Path) -> Result<bool> {
    let conn = Connection::open(path)?;
    let result: String = conn.query_row(
        "PRAGMA integrity_check", [], |row| row.get(0)
    )?;
    Ok(result == "ok")
}

/// Run garbage collection based on retention policy
///
/// Keeps the most recent snapshots according to the retention tiers:
/// - `daily`: Keep N most recent daily snapshots
/// - `weekly`: Keep N most recent weekly snapshots (one per week)
/// - `monthly`: Keep N most recent monthly snapshots (one per month)
pub fn run_gc(
    cache: &CacheManager,
    config: &super::config::RetentionConfig,
) -> Result<GcReport> {
    let snapshots = list_snapshots(cache)?;
    let snapshots_before = snapshots.len();
    let mut to_keep: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut corrupted_removed = 0u64;

    // First pass: remove corrupted snapshots
    for snapshot in &snapshots {
        match validate_snapshot(&snapshot.path) {
            Ok(true) => {}
            _ => {
                log::warn!("Removing corrupted snapshot: {}", snapshot.id);
                let _ = std::fs::remove_file(&snapshot.path);
                corrupted_removed += 1;
                continue;
            }
        }
    }

    // Collect valid snapshots with parsed timestamps
    let valid_snapshots: Vec<&SnapshotInfo> = snapshots.iter()
        .filter(|s| validate_snapshot(&s.path).unwrap_or(false))
        .collect();

    // Already sorted newest first from list_snapshots
    // Keep daily tier: the N most recent
    for snapshot in valid_snapshots.iter().take(config.daily) {
        to_keep.insert(snapshot.id.clone());
    }

    // Keep weekly tier: one per calendar week, most recent N weeks
    let mut weeks_seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut weekly_kept = 0;
    for snapshot in &valid_snapshots {
        if weekly_kept >= config.weekly { break; }
        let week_key = snapshot_to_week_key(&snapshot.id);
        if weeks_seen.insert(week_key) {
            to_keep.insert(snapshot.id.clone());
            weekly_kept += 1;
        }
    }

    // Keep monthly tier: one per calendar month, most recent N months
    let mut months_seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut monthly_kept = 0;
    for snapshot in &valid_snapshots {
        if monthly_kept >= config.monthly { break; }
        let month_key = snapshot_to_month_key(&snapshot.id);
        if months_seen.insert(month_key) {
            to_keep.insert(snapshot.id.clone());
            monthly_kept += 1;
        }
    }

    // Remove snapshots not in any keep tier
    let mut space_freed: u64 = 0;
    let mut removed = 0usize;

    for snapshot in &valid_snapshots {
        if !to_keep.contains(&snapshot.id) {
            space_freed += snapshot.size_bytes;
            let _ = std::fs::remove_file(&snapshot.path);
            removed += 1;
        }
    }

    Ok(GcReport {
        snapshots_before,
        snapshots_after: snapshots_before - removed - corrupted_removed as usize,
        removed: removed + corrupted_removed as usize,
        corrupted_removed: corrupted_removed as usize,
        space_freed_bytes: space_freed,
    })
}

/// Read snapshot info from a database file
fn read_snapshot_info(path: &Path) -> Result<SnapshotInfo> {
    let conn = Connection::open(path)
        .context("Failed to open snapshot database")?;

    let id = path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Read metadata
    let get_meta = |key: &str| -> Option<String> {
        conn.query_row(
            "SELECT value FROM metadata WHERE key = ?1",
            [key],
            |row| row.get(0),
        ).ok()
    };

    let timestamp = get_meta("timestamp").unwrap_or_else(|| id.clone());
    let git_branch = get_meta("git_branch");
    let git_commit = get_meta("git_commit");
    let reflex_version = get_meta("reflex_version").unwrap_or_else(|| "unknown".to_string());

    // Gather stats
    let file_count: usize = conn.query_row(
        "SELECT COUNT(*) FROM files", [], |row| row.get(0)
    ).unwrap_or(0);
    let total_lines: usize = conn.query_row(
        "SELECT COALESCE(SUM(line_count), 0) FROM files", [], |row| row.get(0)
    ).unwrap_or(0);
    let edge_count: usize = conn.query_row(
        "SELECT COUNT(*) FROM dependency_edges", [], |row| row.get(0)
    ).unwrap_or(0);

    let size_bytes = std::fs::metadata(path)
        .map(|m| m.len())
        .unwrap_or(0);

    Ok(SnapshotInfo {
        id,
        path: path.to_path_buf(),
        timestamp,
        git_branch,
        git_commit,
        reflex_version,
        file_count,
        total_lines,
        edge_count,
        size_bytes,
    })
}

/// Extract week key from snapshot ID (YYYYMMDD_HHMMSS -> YYYY-WXX)
fn snapshot_to_week_key(id: &str) -> String {
    // Parse YYYYMMDD from ID
    if id.len() >= 8 {
        let year: i32 = id[0..4].parse().unwrap_or(2000);
        let month: u32 = id[4..6].parse().unwrap_or(1);
        let day: u32 = id[6..8].parse().unwrap_or(1);

        if let Some(date) = chrono::NaiveDate::from_ymd_opt(year, month, day) {
            return format!("{}-W{:02}", year, date.iso_week().week());
        }
    }
    id.to_string()
}

/// Extract month key from snapshot ID (YYYYMMDD_HHMMSS -> YYYY-MM)
fn snapshot_to_month_key(id: &str) -> String {
    if id.len() >= 6 {
        format!("{}-{}", &id[0..4], &id[4..6])
    } else {
        id.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_week_key() {
        assert_eq!(snapshot_to_week_key("20260407_120000"), "2026-W15");
    }

    #[test]
    fn test_month_key() {
        assert_eq!(snapshot_to_month_key("20260407_120000"), "2026-04");
        assert_eq!(snapshot_to_month_key("20261231_235959"), "2026-12");
    }

    #[test]
    fn test_snapshot_info_serialization() {
        let info = SnapshotInfo {
            id: "20260407_120000".to_string(),
            path: PathBuf::from("/tmp/test.db"),
            timestamp: "2026-04-07T12:00:00+00:00".to_string(),
            git_branch: Some("main".to_string()),
            git_commit: Some("abc123".to_string()),
            reflex_version: "1.0.5".to_string(),
            file_count: 100,
            total_lines: 10000,
            edge_count: 50,
            size_bytes: 1024,
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("20260407_120000"));
    }
}
