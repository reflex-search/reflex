use crate::cache::CacheManager;
use anyhow::Result;

pub(super) fn handle_snapshot_create() -> Result<()> {
    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }

    let info = crate::pulse::snapshot::create_snapshot(&cache)?;
    eprintln!("Snapshot created: {}", info.id);
    eprintln!(
        "  Files: {}, Lines: {}, Edges: {}",
        info.file_count, info.total_lines, info.edge_count
    );
    if let Some(branch) = &info.git_branch {
        eprintln!("  Branch: {}", branch);
    }

    // Run background GC
    let pulse_config = crate::pulse::config::load_pulse_config(cache.path())?;
    let gc_report = crate::pulse::snapshot::run_gc(&cache, &pulse_config.retention)?;
    if gc_report.removed > 0 {
        eprintln!("  GC: removed {} old snapshot(s)", gc_report.removed);
    }

    Ok(())
}

pub(super) fn handle_snapshot_list(json: bool, pretty: bool) -> Result<()> {
    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }

    let snapshots = crate::pulse::snapshot::list_snapshots(&cache)?;

    if json || pretty {
        let output = if pretty {
            serde_json::to_string_pretty(&snapshots)?
        } else {
            serde_json::to_string(&snapshots)?
        };
        println!("{}", output);
    } else {
        if snapshots.is_empty() {
            eprintln!("No snapshots found. Run `rfx snapshot` to create one.");
            return Ok(());
        }
        println!(
            "{:<20} {:>6} {:>8} {:>6}  {}",
            "ID", "Files", "Lines", "Edges", "Branch"
        );
        println!("{}", "-".repeat(60));
        for s in &snapshots {
            println!(
                "{:<20} {:>6} {:>8} {:>6}  {}",
                s.id,
                s.file_count,
                s.total_lines,
                s.edge_count,
                s.git_branch.as_deref().unwrap_or("-")
            );
        }
        eprintln!("\n{} snapshot(s)", snapshots.len());
    }

    Ok(())
}

pub(super) fn handle_snapshot_diff(
    baseline: Option<String>,
    current: Option<String>,
    json: bool,
    pretty: bool,
) -> Result<()> {
    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }

    let snapshots = crate::pulse::snapshot::list_snapshots(&cache)?;
    let pulse_config = crate::pulse::config::load_pulse_config(cache.path())?;

    let current_snapshot = match &current {
        Some(id) => snapshots
            .iter()
            .find(|s| s.id == *id)
            .ok_or_else(|| anyhow::anyhow!("Snapshot '{}' not found", id))?,
        None => snapshots
            .first()
            .ok_or_else(|| anyhow::anyhow!("No snapshots found. Run `rfx snapshot` first."))?,
    };

    let baseline_snapshot = match &baseline {
        Some(id) => snapshots
            .iter()
            .find(|s| s.id == *id)
            .ok_or_else(|| anyhow::anyhow!("Snapshot '{}' not found", id))?,
        None => snapshots.get(1).ok_or_else(|| {
            anyhow::anyhow!(
                "Need at least 2 snapshots to diff. Run `rfx snapshot` again after making changes."
            )
        })?,
    };

    let diff = crate::pulse::diff::compute_diff(
        &baseline_snapshot.path,
        &current_snapshot.path,
        &pulse_config.thresholds,
    )?;

    if json || pretty {
        let output = if pretty {
            serde_json::to_string_pretty(&diff)?
        } else {
            serde_json::to_string(&diff)?
        };
        println!("{}", output);
    } else {
        let s = &diff.summary;
        println!("Diff: {} → {}", diff.baseline_id, diff.current_id);
        println!(
            "  Files: +{} -{} ~{}",
            s.files_added, s.files_removed, s.files_modified
        );
        println!("  Edges: +{} -{}", s.edges_added, s.edges_removed);
        if !diff.threshold_alerts.is_empty() {
            println!("  Alerts: {}", diff.threshold_alerts.len());
            for alert in &diff.threshold_alerts {
                println!("    [{:?}] {}", alert.severity, alert.message);
            }
        }
    }

    Ok(())
}

pub(super) fn handle_snapshot_gc(json: bool) -> Result<()> {
    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }

    let pulse_config = crate::pulse::config::load_pulse_config(cache.path())?;
    let report = crate::pulse::snapshot::run_gc(&cache, &pulse_config.retention)?;

    if json {
        println!("{}", serde_json::to_string(&report)?);
    } else {
        println!(
            "GC complete: before {}, after {}, removed {}",
            report.snapshots_before, report.snapshots_after, report.removed
        );
    }

    Ok(())
}
