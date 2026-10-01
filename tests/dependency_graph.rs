//! Import-graph accuracy: what `analyze` and `get_dependencies` answer from.
//!
//! On 2.1.0 most internal imports of package-based languages did not resolve
//! (Go 0.3 % on Kubernetes), and every text, lock and generated file was an island
//! and an unused file, so `analyze` reported 27k islands. These tests pin the fixes.

use reflex::dependency::DependencyIndex;
use reflex::{CacheManager, IndexConfig, Indexer};
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

fn index(root: &Path) {
    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .expect("index");
}

fn write(root: &Path, rel: &str, body: &str) {
    let p = root.join(rel);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, body).unwrap();
}

fn deps(root: &Path) -> DependencyIndex {
    DependencyIndex::new(CacheManager::new(root))
}

fn paths(d: &DependencyIndex, ids: &[i64]) -> HashSet<String> {
    d.get_file_paths(ids).unwrap().into_values().collect()
}

#[test]
fn text_lock_generated_never_islands_or_unused() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "Cargo.toml",
        "[package]\nname = \"ws\"\nversion = \"0.1.0\"\n",
    );
    write(r, "src/lib.rs", "mod util;\npub fn f() {}\n");
    write(r, "src/util.rs", "pub fn g() {}\n");
    write(r, "README.md", "# ws\n");
    write(r, "Cargo.lock", "version = 3\n");
    write(r, "gen/x.pb.go", "package gen\n");
    index(r);

    let d = deps(r);
    let island_files: HashSet<String> = d
        .find_islands()
        .unwrap()
        .iter()
        .flat_map(|i| paths(&d, i))
        .collect();
    let unused = paths(&d, &d.find_unused_files().unwrap());
    for f in ["README.md", "Cargo.lock", "Cargo.toml", "gen/x.pb.go"] {
        assert!(
            !island_files.contains(f),
            "{f} is an island: {island_files:?}"
        );
        assert!(!unused.contains(f), "{f} is unused: {unused:?}");
    }
    assert!(island_files.contains("src/util.rs"), "{island_files:?}");
}

/// 120 Go files importing a package of their own module that does not exist:
/// internal, and unresolvable whatever the resolver does.
fn go_workspace_with_missing_package(importers: usize) -> TempDir {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(r, "go.mod", "module example.com/m\n\ngo 1.22\n");
    for i in 0..importers {
        write(
            r,
            &format!("pkg/p{i}/p.go"),
            &format!("package p{i}\n\nimport \"example.com/m/pkg/missing\"\n\nvar _ = missing.X\n"),
        );
    }
    t
}

#[test]
fn resolution_by_language_reports_go_unresolved() {
    let t = go_workspace_with_missing_package(120);
    index(t.path());
    let d = deps(t.path());

    let go = d
        .internal_resolution_by_language()
        .unwrap()
        .into_iter()
        .find(|l| l.language == "Go")
        .expect("a Go row");
    assert_eq!((go.internal, go.resolved), (120, 0));

    let warnings = d.low_resolution_warnings().unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(
        warnings[0].starts_with("Go: 0 of 120 internal imports (0.0%) resolve"),
        "{}",
        warnings[0]
    );
}

#[test]
fn few_unresolved_imports_do_not_warn() {
    let t = go_workspace_with_missing_package(20);
    index(t.path());
    assert!(deps(t.path()).low_resolution_warnings().unwrap().is_empty());
}

fn rfx(root: &Path, args: &[&str]) -> (String, String) {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_rfx"))
        .args(args)
        .current_dir(root)
        .output()
        .expect("run rfx");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    (
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
    )
}

#[test]
fn cli_analyze_and_deps_warn_on_stderr() {
    let t = go_workspace_with_missing_package(120);
    index(t.path());

    let (stdout, stderr) = rfx(t.path(), &["analyze", "--json", "--no-update"]);
    let summary: serde_json::Value = serde_json::from_str(&stdout).expect("stdout is JSON");
    assert!(
        summary["warnings"][0]
            .as_str()
            .unwrap()
            .starts_with("Go: 0 of 120"),
        "{summary}"
    );
    assert!(stderr.contains("Warning: Go: 0 of 120"), "{stderr}");

    let (stdout, stderr) = rfx(t.path(), &["deps", "pkg/p0/p.go", "--json", "--no-update"]);
    serde_json::from_str::<serde_json::Value>(&stdout).expect("stdout is JSON");
    assert!(stderr.contains("Warning: Go: 0 of 120"), "{stderr}");
}

#[test]
fn hotspots_count_distinct_importers() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "Cargo.toml",
        "[package]\nname = \"ws\"\nversion = \"0.1.0\"\n",
    );
    write(
        r,
        "src/lib.rs",
        "mod util;\nuse crate::util::a;\nuse crate::util::b;\npub fn f() { a(); b(); }\n",
    );
    write(r, "src/util.rs", "pub fn a() {}\npub fn b() {}\n");
    index(r);

    let d = deps(r);
    let util = d.get_file_id_by_path("src/util.rs").unwrap().unwrap();
    let hot = d.find_hotspots(None, 1).unwrap();
    assert_eq!(hot, vec![(util, 1)], "one importer, three rows");
}

/// A 2.1.0 cache has `file_dependencies` without the package columns, and
/// `CREATE TABLE IF NOT EXISTS` does not add them: the next index must.
#[test]
fn index_upgrades_a_cache_without_package_columns() {
    let t = go_workspace_with_missing_package(2);
    let r = t.path();
    index(r);
    {
        let conn = reflex::cache::open_meta_db(r.join(".reflex/meta.db")).unwrap();
        conn.execute_batch(
            "DROP VIEW import_edges;
             DROP INDEX idx_deps_package;
             ALTER TABLE file_dependencies DROP COLUMN resolved_package;
             ALTER TABLE file_dependencies DROP COLUMN resolved_member;
             DROP TABLE package_members;
             UPDATE statistics SET value = 'from-2.1.0' WHERE key = 'schema_hash';",
        )
        .unwrap();
    }
    index(r);
    assert!(deps(r).find_islands().is_ok());
}

/// `(importer, imported)` paths of every graph edge, sorted.
fn edges(root: &Path) -> Vec<(String, String)> {
    let conn = reflex::cache::open_meta_db(root.join(".reflex/meta.db")).unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT DISTINCT s.path, t.path FROM import_edges e
             JOIN files s ON s.id = e.src JOIN files t ON t.id = e.dst
             ORDER BY 1, 2",
        )
        .unwrap();
    stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

fn edge(a: &str, b: &str) -> (String, String) {
    (a.to_string(), b.to_string())
}

/// A module with a library package (two files and a test) and a `package main`.
fn go_module(r: &Path) {
    write(r, "go.mod", "module example.com/m\n\ngo 1.22\n");
    write(r, "pkg/b/b1.go", "package b\n\nfunc One() {}\n");
    write(
        r,
        "pkg/b/b2.go",
        "package b\n\nimport \"fmt\"\n\nfunc Two() { fmt.Println() }\n",
    );
    write(
        r,
        "pkg/b/b_test.go",
        "package b\n\nimport \"testing\"\n\nfunc TestOne(t *testing.T) {}\n",
    );
    write(
        r,
        "cmd/app/main.go",
        "package main\n\nimport (\n\t\"example.com/m/pkg/b\"\n\t\"k8s.io/klog/v2\"\n)\n\nfunc main() { b.One(); klog.Info() }\n",
    );
    write(r, "cmd/app/flags.go", "package main\n\nvar verbose bool\n");
}

#[test]
fn go_package_import_links_every_non_test_file() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    go_module(r);
    index(r);

    assert_eq!(
        edges(r),
        vec![
            edge("cmd/app/main.go", "pkg/b/b1.go"),
            edge("cmd/app/main.go", "pkg/b/b2.go"),
        ]
    );
    let d = deps(r);
    let main = d.get_file_id_by_path("cmd/app/main.go").unwrap().unwrap();
    let info = d.get_dependencies_info(main).unwrap();
    let b = info
        .iter()
        .find(|i| i.path == "example.com/m/pkg/b")
        .expect("{info:?}");
    assert_eq!(
        b.resolved_paths.as_deref(),
        Some(&["pkg/b/b1.go".to_string(), "pkg/b/b2.go".to_string()][..])
    );
    let go = d
        .internal_resolution_by_language()
        .unwrap()
        .into_iter()
        .find(|l| l.language == "Go")
        .unwrap();
    // `k8s.io/klog/v2` is in no module of the workspace: External, not unresolved.
    assert_eq!((go.internal, go.resolved), (1, 1));
}

#[test]
fn package_main_siblings_are_one_unit() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    go_module(r);
    index(r);

    let d = deps(r);
    let unused = paths(&d, &d.find_unused_files().unwrap());
    assert!(unused.is_empty(), "{unused:?}");
    let islands: Vec<HashSet<String>> = d
        .find_islands()
        .unwrap()
        .iter()
        .map(|i| paths(&d, i))
        .collect();
    let app = islands
        .iter()
        .find(|i| i.contains("cmd/app/main.go"))
        .unwrap();
    assert!(app.contains("cmd/app/flags.go"), "{islands:?}");
    assert!(app.contains("pkg/b/b2.go"), "{islands:?}");
}

#[test]
fn incremental_add_and_delete_go_file_in_package_matches_full_build() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    go_module(r);
    index(r);

    let indexer = Indexer::new(CacheManager::new(r), IndexConfig::default());
    write(r, "pkg/b/b3.go", "package b\n\nfunc Three() {}\n");
    indexer
        .update_paths(r, &[r.join("pkg/b/b3.go")])
        .expect("update_paths");
    fs::remove_file(r.join("pkg/b/b1.go")).unwrap();
    index(r);
    let incremental = edges(r);

    let fresh = TempDir::new().unwrap();
    go_module(fresh.path());
    fs::remove_file(fresh.path().join("pkg/b/b1.go")).unwrap();
    write(
        fresh.path(),
        "pkg/b/b3.go",
        "package b\n\nfunc Three() {}\n",
    );
    index(fresh.path());

    assert_eq!(incremental, edges(fresh.path()));
    assert!(
        incremental.contains(&edge("cmd/app/main.go", "pkg/b/b3.go")),
        "{incremental:?}"
    );
}
