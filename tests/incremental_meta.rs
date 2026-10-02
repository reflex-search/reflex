//! Stage 0 of incremental indexing, the change-detection half: an index run reads
//! only files whose stat changed, rewrites only their rows and dependencies, and
//! still leaves `meta.db` answering exactly as a fresh build of the same tree.

use reflex::cache::open_meta_db;
use reflex::{CacheManager, IndexConfig, Indexer};
use std::fs;
use std::path::Path;
use tempfile::TempDir;

fn index(root: &Path) -> reflex::IndexStats {
    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .expect("index")
}

fn meta(root: &Path) -> rusqlite::Connection {
    open_meta_db(root.join(".reflex").join("meta.db")).expect("open meta.db")
}

/// Every dependency and export row, joined to paths (as in dependency_equivalence).
fn dump(root: &Path) -> String {
    let conn = meta(root);
    let mut out = String::new();
    let mut stmt = conn
        .prepare(
            "SELECT f.path, fd.imported_path, r.path, fd.import_type, fd.line_number, fd.imported_symbols
             FROM file_dependencies fd
             JOIN files f ON f.id = fd.file_id
             LEFT JOIN files r ON r.id = fd.resolved_file_id
             ORDER BY f.path, fd.line_number, fd.imported_path, fd.import_type",
        )
        .unwrap();
    let rows = stmt
        .query_map([], |row| {
            Ok(format!(
                "{} | {} | {} | {} | {} | {}",
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?
                    .unwrap_or_else(|| "-".into()),
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, Option<String>>(5)?
                    .unwrap_or_else(|| "-".into()),
            ))
        })
        .unwrap();
    for r in rows {
        out.push_str(&r.unwrap());
        out.push('\n');
    }
    let mut stmt = conn
        .prepare(
            "SELECT f.path, e.exported_symbol, e.source_path, r.path, e.line_number
             FROM file_exports e
             JOIN files f ON f.id = e.file_id
             LEFT JOIN files r ON r.id = e.resolved_source_id
             ORDER BY f.path, e.line_number, e.source_path",
        )
        .unwrap();
    let rows = stmt
        .query_map([], |row| {
            Ok(format!(
                "export {} | {} | {} | {} | {}",
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?
                    .unwrap_or_else(|| "*".into()),
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?
                    .unwrap_or_else(|| "-".into()),
                row.get::<_, i64>(4)?,
            ))
        })
        .unwrap();
    for r in rows {
        out.push_str(&r.unwrap());
        out.push('\n');
    }
    out
}

/// The dump of the index as it stands, and of a fresh build of the same directory.
fn against_fresh(root: &Path) -> (String, String) {
    let updated = dump(root);
    reflex::query::invalidate_caches(root); // Windows: release the shared handle first
    fs::rename(root.join(".reflex"), root.join(".reflex-updated")).unwrap();
    index(root);
    let fresh = dump(root);
    reflex::query::invalidate_caches(root); // Windows: release the shared handle first
    fs::remove_dir_all(root.join(".reflex")).unwrap();
    fs::rename(root.join(".reflex-updated"), root.join(".reflex")).unwrap();
    (updated, fresh)
}

fn write(root: &Path, rel: &str, body: &str) {
    let p = root.join(rel);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, body).unwrap();
}

#[cfg(unix)]
#[test]
fn unchanged_files_are_not_read() {
    use std::os::unix::fs::PermissionsExt;
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    write(root, "a.rs", "pub fn alpha() {}\n");
    write(root, "b.rs", "pub fn beta() {}\n");
    index(root);

    // Unreadable, but its size and mtime are as indexed (chmod moves ctime only).
    let a = root.join("a.rs");
    fs::set_permissions(&a, fs::Permissions::from_mode(0o000)).unwrap();
    if fs::read(&a).is_ok() {
        return; // running as root: permissions do not stop the read
    }
    let stats = index(root);
    fs::set_permissions(&a, fs::Permissions::from_mode(0o644)).unwrap();

    assert_eq!(stats.total_files, 2, "a.rs was kept without being read");
    assert_eq!(stats.unchanged_files, 2);
}

#[test]
fn a_one_file_edit_rewrites_only_that_files_dependency_rows() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    write(root, "Cargo.toml", "[package]\nname = \"demo\"\n");
    write(root, "src/lib.rs", "mod a;\nmod b;\n");
    write(
        root,
        "src/a.rs",
        "use crate::b::beta;\npub fn alpha() { beta(); }\n",
    );
    write(root, "src/b.rs", "use std::fmt;\npub fn beta() {}\n");
    index(root);
    let row_ids = |root: &Path, path: &str| -> Vec<i64> {
        let conn = meta(root);
        let mut stmt = conn
            .prepare(
                "SELECT d.id FROM file_dependencies d JOIN files f ON f.id = d.file_id \
                 WHERE f.path = ? ORDER BY d.id",
            )
            .unwrap();
        stmt.query_map([path], |r| r.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    };
    let a_before = row_ids(root, "src/a.rs");
    let b_before = row_ids(root, "src/b.rs");
    assert!(!a_before.is_empty() && !b_before.is_empty());

    write(root, "src/b.rs", "use std::io;\npub fn beta() {}\n");
    index(root);

    assert_eq!(
        row_ids(root, "src/a.rs"),
        a_before,
        "unchanged file's rows untouched"
    );
    assert_ne!(
        row_ids(root, "src/b.rs"),
        b_before,
        "edited file's rows rewritten"
    );
    let (updated, fresh) = against_fresh(root);
    assert_eq!(updated, fresh);
}

#[test]
fn adding_and_removing_a_target_re_resolves_its_importers() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    write(root, "Cargo.toml", "[package]\nname = \"demo\"\n");
    write(root, "src/lib.rs", "mod a;\n");
    // `crate::b` does not exist yet: unresolved.
    write(
        root,
        "src/a.rs",
        "use crate::b::beta;\npub fn alpha() { beta(); }\n",
    );
    write(
        root,
        "web/app.ts",
        "import { u } from './util';\nexport * from './util';\n",
    );
    index(root);

    write(root, "src/b.rs", "pub fn beta() {}\n");
    write(root, "web/util.ts", "export const u = 1;\n");
    index(root);
    let (updated, fresh) = against_fresh(root);
    assert_eq!(updated, fresh);
    assert!(
        updated.contains("src/a.rs | crate::b::beta | src/b.rs"),
        "{updated}"
    );
    assert!(
        updated.contains("export web/app.ts | * | ./util | web/util.ts"),
        "{updated}"
    );

    fs::remove_file(root.join("src/b.rs")).unwrap();
    fs::remove_file(root.join("web/util.ts")).unwrap();
    index(root);
    let (updated, fresh) = against_fresh(root);
    assert_eq!(updated, fresh);
}

#[test]
fn a_resolver_config_change_re_resolves_every_file() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    write(
        root,
        "tsconfig.json",
        r#"{"compilerOptions":{"baseUrl":".","paths":{"@lib/*":["lib/*"]}}}"#,
    );
    write(root, "lib/x.ts", "export const x = 1;\n");
    write(root, "other/x.ts", "export const x = 2;\n");
    write(root, "app.ts", "import { x } from '@lib/x';\n");
    write(root, "go.mod", "module example.com/one\n");
    write(root, "pkg/p.go", "package pkg\n");
    write(
        root,
        "main.go",
        "package main\nimport \"example.com/two/pkg\"\n",
    );
    index(root);

    // The alias now points elsewhere and the Go module is renamed: app.ts and
    // main.go are unchanged, but their rows are not.
    write(
        root,
        "tsconfig.json",
        r#"{"compilerOptions":{"baseUrl":".","paths":{"@lib/*":["other/*"]}}}"#,
    );
    write(root, "go.mod", "module example.com/two\n");
    index(root);
    let (updated, fresh) = against_fresh(root);
    assert_eq!(updated, fresh);
    assert!(
        updated.contains("app.ts | @lib/x | other/x.ts"),
        "{updated}"
    );
}

#[test]
fn a_touch_refreshes_the_fingerprint_without_a_rebuild() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    write(root, "a.rs", "pub fn alpha() {}\n");
    index(root);
    let mtime = |root: &Path| -> i64 {
        meta(root)
            .query_row("SELECT mtime_ns FROM files WHERE path = 'a.rs'", [], |r| {
                r.get(0)
            })
            .unwrap()
    };
    let before = mtime(root);
    let later = filetime::FileTime::from_unix_time(2_000_000_000, 0);
    filetime::set_file_mtime(root.join("a.rs"), later).unwrap();

    let stats = index(root);
    assert_eq!(stats.unchanged_files, 1);
    assert_eq!(stats.modified_files, 0);
    assert_ne!(mtime(root), before, "the stored mtime follows the touch");
}
