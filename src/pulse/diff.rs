//! Diff engine: cross-snapshot structural comparison
//!
//! Takes two snapshot databases and produces a deterministic delta of all
//! structural changes: files added/removed/modified, dependency edges
//! added/removed, hotspot shifts, cycle changes, and threshold alerts.

use anyhow::{Context, Result};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

use crate::dependency::DependencyIndex;

/// A file that was added or removed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDelta {
    pub path: String,
    pub language: Option<String>,
    pub line_count: usize,
}

/// A file that was modified between snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileModDelta {
    pub path: String,
    pub language: Option<String>,
    pub old_line_count: usize,
    pub new_line_count: usize,
}

/// A dependency edge change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeDelta {
    pub source_path: String,
    pub target_path: String,
    pub import_type: String,
}

/// A hotspot that changed fan-in
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotspotDelta {
    pub path: String,
    pub old_fan_in: usize,
    pub new_fan_in: usize,
}

/// Changes in disconnected components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IslandDelta {
    pub old_count: usize,
    pub new_count: usize,
}

/// Module-level metrics change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleMetricsDelta {
    pub module_path: String,
    pub old_file_count: Option<usize>,
    pub new_file_count: Option<usize>,
    pub old_total_lines: Option<usize>,
    pub new_total_lines: Option<usize>,
}

/// A threshold alert triggered by a metric crossing a boundary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdAlert {
    pub severity: AlertSeverity,
    pub category: String,
    pub message: String,
    pub path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Warning,
    Critical,
}

/// Complete diff between two snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotDiff {
    pub baseline_id: String,
    pub current_id: String,
    pub baseline_timestamp: String,
    pub current_timestamp: String,

    // File-level changes
    pub files_added: Vec<FileDelta>,
    pub files_removed: Vec<FileDelta>,
    pub files_modified: Vec<FileModDelta>,

    // Dependency graph changes
    pub edges_added: Vec<EdgeDelta>,
    pub edges_removed: Vec<EdgeDelta>,

    // Structural analysis deltas
    pub hotspot_changes: Vec<HotspotDelta>,
    pub new_cycles: Vec<Vec<String>>,
    pub resolved_cycles: Vec<Vec<String>>,
    pub island_changes: IslandDelta,

    // Module metrics
    pub module_changes: Vec<ModuleMetricsDelta>,

    // Threshold alerts
    pub threshold_alerts: Vec<ThresholdAlert>,

    // Summary stats
    pub summary: DiffSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffSummary {
    pub files_added: usize,
    pub files_removed: usize,
    pub files_modified: usize,
    pub edges_added: usize,
    pub edges_removed: usize,
    pub net_line_change: i64,
}

/// Compute the diff between two snapshot databases
pub fn compute_diff(
    baseline_path: &Path,
    current_path: &Path,
    thresholds: &super::config::ThresholdConfig,
) -> Result<SnapshotDiff> {
    // Open in-memory connection and attach both snapshots
    let conn = Connection::open_in_memory()
        .context("Failed to open in-memory database")?;

    conn.execute(
        "ATTACH DATABASE ?1 AS baseline",
        [baseline_path.to_str().unwrap()],
    )?;
    conn.execute(
        "ATTACH DATABASE ?1 AS current",
        [current_path.to_str().unwrap()],
    )?;

    // Read metadata
    let baseline_id = read_meta(&conn, "baseline", "timestamp")?;
    let current_id = read_meta(&conn, "current", "timestamp")?;

    // File diffs
    let files_added = query_file_deltas(&conn,
        "SELECT c.path, c.language, COALESCE(c.line_count, 0)
         FROM current.files c
         LEFT JOIN baseline.files b ON c.path = b.path
         WHERE b.path IS NULL
         ORDER BY c.path"
    )?;

    let files_removed = query_file_deltas(&conn,
        "SELECT b.path, b.language, COALESCE(b.line_count, 0)
         FROM baseline.files b
         LEFT JOIN current.files c ON b.path = c.path
         WHERE c.path IS NULL
         ORDER BY b.path"
    )?;

    let files_modified = query_file_mod_deltas(&conn,
        "SELECT b.path, b.language, COALESCE(b.line_count, 0), COALESCE(c.line_count, 0)
         FROM baseline.files b
         JOIN current.files c ON b.path = c.path
         WHERE b.line_count != c.line_count OR b.language != c.language
         ORDER BY b.path"
    )?;

    // Edge diffs (compare by path, not by file ID)
    let edges_added = query_edge_deltas(&conn,
        "SELECT sf.path, tf.path, ce.import_type
         FROM current.dependency_edges ce
         JOIN current.files sf ON ce.source_file_id = sf.id
         JOIN current.files tf ON ce.target_file_id = tf.id
         WHERE NOT EXISTS (
             SELECT 1 FROM baseline.dependency_edges be
             JOIN baseline.files bsf ON be.source_file_id = bsf.id
             JOIN baseline.files btf ON be.target_file_id = btf.id
             WHERE bsf.path = sf.path AND btf.path = tf.path
         )
         ORDER BY sf.path, tf.path"
    )?;

    let edges_removed = query_edge_deltas(&conn,
        "SELECT sf.path, tf.path, be.import_type
         FROM baseline.dependency_edges be
         JOIN baseline.files sf ON be.source_file_id = sf.id
         JOIN baseline.files tf ON be.target_file_id = tf.id
         WHERE NOT EXISTS (
             SELECT 1 FROM current.dependency_edges ce
             JOIN current.files csf ON ce.source_file_id = csf.id
             JOIN current.files ctf ON ce.target_file_id = ctf.id
             WHERE csf.path = sf.path AND ctf.path = tf.path
         )
         ORDER BY sf.path, tf.path"
    )?;

    // Module metric diffs
    let module_changes = query_module_deltas(&conn)?;

    // Hotspot analysis via DependencyIndex on each snapshot
    let baseline_deps = DependencyIndex::from_db_path(baseline_path);
    let current_deps = DependencyIndex::from_db_path(current_path);

    let baseline_hotspots = baseline_deps.find_hotspots(None, 1).unwrap_or_default();
    let current_hotspots = current_deps.find_hotspots(None, 1).unwrap_or_default();

    let hotspot_changes = compute_hotspot_changes(
        &baseline_deps, &current_deps,
        &baseline_hotspots, &current_hotspots,
    );

    // Cycle analysis
    let baseline_cycles = baseline_deps.detect_circular_dependencies().unwrap_or_default();
    let current_cycles = current_deps.detect_circular_dependencies().unwrap_or_default();

    let (new_cycles, resolved_cycles) = compute_cycle_changes(
        &baseline_deps, &current_deps,
        &baseline_cycles, &current_cycles,
    );

    // Island analysis
    let baseline_islands = baseline_deps.find_islands().unwrap_or_default();
    let current_islands = current_deps.find_islands().unwrap_or_default();

    let island_changes = IslandDelta {
        old_count: baseline_islands.len(),
        new_count: current_islands.len(),
    };

    // Compute net line change
    let net_line_change: i64 = files_added.iter().map(|f| f.line_count as i64).sum::<i64>()
        - files_removed.iter().map(|f| f.line_count as i64).sum::<i64>()
        + files_modified.iter().map(|f| f.new_line_count as i64 - f.old_line_count as i64).sum::<i64>();

    // Threshold alerts
    let threshold_alerts = compute_threshold_alerts(
        thresholds,
        &current_hotspots,
        &current_deps,
        &current_cycles,
        &module_changes,
        &files_modified,
    );

    let summary = DiffSummary {
        files_added: files_added.len(),
        files_removed: files_removed.len(),
        files_modified: files_modified.len(),
        edges_added: edges_added.len(),
        edges_removed: edges_removed.len(),
        net_line_change,
    };

    Ok(SnapshotDiff {
        baseline_id: baseline_id.clone(),
        current_id: current_id.clone(),
        baseline_timestamp: baseline_id,
        current_timestamp: current_id,
        files_added,
        files_removed,
        files_modified,
        edges_added,
        edges_removed,
        hotspot_changes,
        new_cycles,
        resolved_cycles,
        island_changes,
        module_changes,
        threshold_alerts,
        summary,
    })
}

fn read_meta(conn: &Connection, db: &str, key: &str) -> Result<String> {
    let sql = format!("SELECT value FROM {}.metadata WHERE key = ?1", db);
    conn.query_row(&sql, [key], |row| row.get(0))
        .unwrap_or_else(|_| "unknown".to_string())
        .pipe(Ok)
}

// Helper trait for pipe syntax
trait Pipe: Sized {
    fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T {
        f(self)
    }
}
impl<T> Pipe for T {}

fn query_file_deltas(conn: &Connection, sql: &str) -> Result<Vec<FileDelta>> {
    let mut stmt = conn.prepare(sql)?;
    let results = stmt.query_map([], |row| {
        Ok(FileDelta {
            path: row.get(0)?,
            language: row.get(1)?,
            line_count: row.get::<_, i64>(2)? as usize,
        })
    })?
    .collect::<Result<Vec<_>, _>>()?;
    Ok(results)
}

fn query_file_mod_deltas(conn: &Connection, sql: &str) -> Result<Vec<FileModDelta>> {
    let mut stmt = conn.prepare(sql)?;
    let results = stmt.query_map([], |row| {
        Ok(FileModDelta {
            path: row.get(0)?,
            language: row.get(1)?,
            old_line_count: row.get::<_, i64>(2)? as usize,
            new_line_count: row.get::<_, i64>(3)? as usize,
        })
    })?
    .collect::<Result<Vec<_>, _>>()?;
    Ok(results)
}

fn query_edge_deltas(conn: &Connection, sql: &str) -> Result<Vec<EdgeDelta>> {
    let mut stmt = conn.prepare(sql)?;
    let results = stmt.query_map([], |row| {
        Ok(EdgeDelta {
            source_path: row.get(0)?,
            target_path: row.get(1)?,
            import_type: row.get(2)?,
        })
    })?
    .collect::<Result<Vec<_>, _>>()?;
    Ok(results)
}

fn query_module_deltas(conn: &Connection) -> Result<Vec<ModuleMetricsDelta>> {
    // SQLite doesn't support FULL OUTER JOIN directly, simulate with UNION
    let sql = "SELECT module_path, old_file_count, new_file_count, old_total_lines, new_total_lines FROM (
        SELECT
            COALESCE(b.module_path, c.module_path) AS module_path,
            b.file_count AS old_file_count, c.file_count AS new_file_count,
            b.total_lines AS old_total_lines, c.total_lines AS new_total_lines
        FROM baseline.metrics b
        LEFT JOIN current.metrics c ON b.module_path = c.module_path
        UNION ALL
        SELECT
            c.module_path,
            NULL AS old_file_count, c.file_count AS new_file_count,
            NULL AS old_total_lines, c.total_lines AS new_total_lines
        FROM current.metrics c
        LEFT JOIN baseline.metrics b ON c.module_path = b.module_path
        WHERE b.module_path IS NULL
    )
    WHERE old_file_count IS NULL OR new_file_count IS NULL
       OR old_file_count != new_file_count OR old_total_lines != new_total_lines
    ORDER BY module_path";

    let mut stmt = conn.prepare(sql)?;
    let results = stmt.query_map([], |row| {
        Ok(ModuleMetricsDelta {
            module_path: row.get(0)?,
            old_file_count: row.get::<_, Option<i64>>(1)?.map(|v| v as usize),
            new_file_count: row.get::<_, Option<i64>>(2)?.map(|v| v as usize),
            old_total_lines: row.get::<_, Option<i64>>(3)?.map(|v| v as usize),
            new_total_lines: row.get::<_, Option<i64>>(4)?.map(|v| v as usize),
        })
    })?
    .collect::<Result<Vec<_>, _>>()?;
    Ok(results)
}

fn compute_hotspot_changes(
    baseline_deps: &DependencyIndex,
    current_deps: &DependencyIndex,
    baseline_hotspots: &[(i64, usize)],
    current_hotspots: &[(i64, usize)],
) -> Vec<HotspotDelta> {
    let mut changes = Vec::new();

    // Build path-based maps for comparison
    let baseline_map: std::collections::HashMap<String, usize> = baseline_hotspots.iter()
        .filter_map(|(id, count)| {
            baseline_deps.get_file_paths(&[*id]).ok()
                .and_then(|paths| paths.get(id).map(|p| (p.clone(), *count)))
        })
        .collect();

    let current_map: std::collections::HashMap<String, usize> = current_hotspots.iter()
        .filter_map(|(id, count)| {
            current_deps.get_file_paths(&[*id]).ok()
                .and_then(|paths| paths.get(id).map(|p| (p.clone(), *count)))
        })
        .collect();

    // Find changes
    for (path, &new_count) in &current_map {
        let old_count = baseline_map.get(path).copied().unwrap_or(0);
        if old_count != new_count {
            changes.push(HotspotDelta {
                path: path.clone(),
                old_fan_in: old_count,
                new_fan_in: new_count,
            });
        }
    }

    // Sort by fan-in change magnitude
    changes.sort_by(|a, b| {
        let a_delta = (a.new_fan_in as i64 - a.old_fan_in as i64).unsigned_abs();
        let b_delta = (b.new_fan_in as i64 - b.old_fan_in as i64).unsigned_abs();
        b_delta.cmp(&a_delta)
    });

    changes
}

fn compute_cycle_changes(
    baseline_deps: &DependencyIndex,
    current_deps: &DependencyIndex,
    baseline_cycles: &[Vec<i64>],
    current_cycles: &[Vec<i64>],
) -> (Vec<Vec<String>>, Vec<Vec<String>>) {
    // Convert cycles to path-based representation for comparison
    let to_path_cycle = |deps: &DependencyIndex, cycle: &[i64]| -> Option<Vec<String>> {
        let paths = deps.get_file_paths(cycle).ok()?;
        let path_cycle: Vec<String> = cycle.iter()
            .filter_map(|id| paths.get(id).cloned())
            .collect();
        if path_cycle.len() == cycle.len() { Some(path_cycle) } else { None }
    };

    let baseline_set: HashSet<Vec<String>> = baseline_cycles.iter()
        .filter_map(|c| to_path_cycle(baseline_deps, c))
        .map(|mut c| { c.sort(); c })
        .collect();

    let current_set: HashSet<Vec<String>> = current_cycles.iter()
        .filter_map(|c| to_path_cycle(current_deps, c))
        .map(|mut c| { c.sort(); c })
        .collect();

    let new_cycles: Vec<Vec<String>> = current_set.difference(&baseline_set).cloned().collect();
    let resolved_cycles: Vec<Vec<String>> = baseline_set.difference(&current_set).cloned().collect();

    (new_cycles, resolved_cycles)
}

fn compute_threshold_alerts(
    thresholds: &super::config::ThresholdConfig,
    current_hotspots: &[(i64, usize)],
    current_deps: &DependencyIndex,
    current_cycles: &[Vec<i64>],
    module_changes: &[ModuleMetricsDelta],
    files_modified: &[FileModDelta],
) -> Vec<ThresholdAlert> {
    let mut alerts = Vec::new();

    // Fan-in alerts
    for &(file_id, count) in current_hotspots {
        if count >= thresholds.fan_in_critical {
            let path = current_deps.get_file_paths(&[file_id]).ok()
                .and_then(|paths| paths.get(&file_id).cloned());
            alerts.push(ThresholdAlert {
                severity: AlertSeverity::Critical,
                category: "fan_in".to_string(),
                message: format!("Critical fan-in: {} imports ({} threshold)", count, thresholds.fan_in_critical),
                path,
            });
        } else if count >= thresholds.fan_in_warning {
            let path = current_deps.get_file_paths(&[file_id]).ok()
                .and_then(|paths| paths.get(&file_id).cloned());
            alerts.push(ThresholdAlert {
                severity: AlertSeverity::Warning,
                category: "fan_in".to_string(),
                message: format!("High fan-in: {} imports ({} threshold)", count, thresholds.fan_in_warning),
                path,
            });
        }
    }

    // Cycle length alerts
    for cycle in current_cycles {
        if cycle.len() >= thresholds.cycle_length {
            alerts.push(ThresholdAlert {
                severity: AlertSeverity::Warning,
                category: "circular_dependency".to_string(),
                message: format!("Circular dependency chain of length {}", cycle.len()),
                path: None,
            });
        }
    }

    // Module size alerts
    for change in module_changes {
        if let Some(count) = change.new_file_count {
            if count >= thresholds.module_file_count {
                alerts.push(ThresholdAlert {
                    severity: AlertSeverity::Warning,
                    category: "module_size".to_string(),
                    message: format!("Module has {} files (threshold: {})", count, thresholds.module_file_count),
                    path: Some(change.module_path.clone()),
                });
            }
        }
    }

    // Line count growth alerts
    for file in files_modified {
        if file.old_line_count > 0 {
            let growth = file.new_line_count as f64 / file.old_line_count as f64;
            if growth >= thresholds.line_count_growth {
                alerts.push(ThresholdAlert {
                    severity: AlertSeverity::Warning,
                    category: "line_growth".to_string(),
                    message: format!(
                        "Line count grew {:.1}x ({} -> {})",
                        growth, file.old_line_count, file.new_line_count
                    ),
                    path: Some(file.path.clone()),
                });
            }
        }
    }

    alerts
}
