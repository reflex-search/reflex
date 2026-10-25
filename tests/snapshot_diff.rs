//! `rfx snapshot diff` reads each snapshot's import graph (hotspots, cycles, islands).
//!
//! Through 2.2.0 the snapshot schema had no `walk_seq`, `vendored`, `package_members`
//! or `import_edges`, so every graph query failed, and `compute_diff` turned each
//! failure into an empty list: no hotspot changes, no new cycles, 0 islands.

use reflex::pulse::config::ThresholdConfig;
use reflex::pulse::diff::compute_diff;
use reflex::pulse::snapshot::create_snapshot;
use reflex::{CacheManager, IndexConfig, Indexer};
use rusqlite::Connection;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

fn write(root: &Path, rel: &str, text: &str) {
    std::fs::write(root.join(rel), text).unwrap();
}

fn index(root: &Path) {
    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .unwrap();
}

/// Snapshot the index, and move the snapshot aside so the next one (named by the
/// second it was taken in) cannot collide with it.
fn snapshot(root: &Path, name: &str) -> PathBuf {
    let info = create_snapshot(&CacheManager::new(root)).unwrap();
    let to = info.path.with_file_name(format!("{name}.db"));
    std::fs::rename(&info.path, &to).unwrap();
    to
}

/// Baseline: `hub.js` has 2 importers, `c.js` imports `d.js`, `lone.js` imports
/// nothing. Current: a third importer of `hub.js`, and `d.js` imports `c.js` back.
fn baseline_and_current() -> (TempDir, PathBuf, PathBuf) {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    write(root, "hub.js", "export function helper() { return 1; }\n");
    write(root, "a.js", "import { helper } from './hub';\n");
    write(root, "b.js", "import { helper } from './hub';\n");
    write(root, "c.js", "import { x } from './d';\n");
    write(root, "d.js", "export const x = 1;\n");
    write(root, "lone.js", "export const y = 2;\n");
    index(root);
    let baseline = snapshot(root, "baseline");

    write(root, "e.js", "import { helper } from './hub';\n");
    write(
        root,
        "d.js",
        "import { c } from './c';\nexport const x = 1;\n",
    );
    index(root);
    let current = snapshot(root, "current");
    (temp, baseline, current)
}

#[test]
fn diff_reports_hotspots_cycles_and_islands() {
    let (_temp, baseline, current) = baseline_and_current();
    let diff = compute_diff(&baseline, &current, &ThresholdConfig::default()).unwrap();

    let hub = diff
        .hotspot_changes
        .iter()
        .find(|h| h.path == "hub.js")
        .unwrap_or_else(|| panic!("hub.js fan-in changed: {:?}", diff.hotspot_changes));
    assert_eq!((hub.old_fan_in, hub.new_fan_in), (2, 3));

    assert_eq!(
        diff.new_cycles.len(),
        1,
        "c.js <-> d.js: {:?}",
        diff.new_cycles
    );
    assert!(
        diff.resolved_cycles.is_empty(),
        "{:?}",
        diff.resolved_cycles
    );

    // {hub, a, b}, {c, d}, {lone}; then e.js joins the hub's island.
    assert_eq!(diff.island_changes.old_count, 3);
    assert_eq!(diff.island_changes.new_count, 3);
}

/// A snapshot written by 2.2.0 or older (schema version 1) is upgraded in place.
#[test]
fn diff_upgrades_a_version_1_snapshot() {
    let (_temp, baseline, current) = baseline_and_current();
    let conn = Connection::open(&baseline).unwrap();
    conn.execute_batch(
        "DROP VIEW import_edges;
         DROP TABLE package_members;
         ALTER TABLE files DROP COLUMN walk_seq;
         ALTER TABLE files DROP COLUMN vendored;
         ALTER TABLE dependency_edges DROP COLUMN dep_id;
         ALTER TABLE dependency_edges DROP COLUMN package;
         UPDATE metadata SET value = '1' WHERE key = 'schema_version';",
    )
    .unwrap();
    drop(conn);

    let diff = compute_diff(&baseline, &current, &ThresholdConfig::default()).unwrap();
    let hub = diff
        .hotspot_changes
        .iter()
        .find(|h| h.path == "hub.js")
        .unwrap_or_else(|| panic!("hub.js fan-in changed: {:?}", diff.hotspot_changes));
    assert_eq!((hub.old_fan_in, hub.new_fan_in), (2, 3));
    assert_eq!(diff.new_cycles.len(), 1, "{:?}", diff.new_cycles);
    assert_eq!(diff.island_changes.old_count, 3);

    let version: String = Connection::open(&baseline)
        .unwrap()
        .query_row(
            "SELECT value FROM metadata WHERE key = 'schema_version'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(version, "2");
}
