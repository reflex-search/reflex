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

/// Two Maven modules sharing one groupId (every neo4j module is `org.neo4j`), a
/// package split across them, a wildcard import, and Kotlin importing Java and a
/// top-level function from a file not named after it.
fn jvm_project(r: &Path) {
    let pom =
        "<project>\n  <groupId>org.acme</groupId>\n  <artifactId>x</artifactId>\n</project>\n";
    write(r, "pom.xml", pom);
    write(r, "core/pom.xml", pom);
    write(r, "app/pom.xml", pom);
    write(
        r,
        "core/src/main/java/org/acme/util/Strings.java",
        "package org.acme.util;\n\npublic class Strings {}\n",
    );
    write(
        r,
        "app/src/main/java/org/acme/util/Numbers.java",
        "package org.acme.util;\n\npublic final class Numbers {}\n\nclass Helper {}\n",
    );
    write(
        r,
        "app/src/main/java/org/acme/app/Main.java",
        "package org.acme.app;\n\nimport org.acme.util.Strings;\nimport org.acme.util.Helper;\nimport static org.acme.util.Numbers.parse;\nimport java.util.List;\n\npublic class Main {}\n",
    );
    write(
        r,
        "app/src/main/java/org/acme/app/All.java",
        "package org.acme.app;\n\nimport org.acme.util.*;\n\nclass All {}\n",
    );
    write(
        r,
        "app/src/main/kotlin/org/acme/app/Tool.kt",
        "package org.acme.app\n\nimport org.acme.util.Strings\nimport org.acme.app.helpers.greet\n\nclass Tool\n",
    );
    write(
        r,
        "app/src/main/kotlin/org/acme/app/helpers/Misc.kt",
        "@file:JvmName(\"Misc\")\npackage org.acme.app.helpers\n\nfun greet() {}\n\ninternal fun <T> List<T>.second(): T = this[1]\n",
    );
}

#[test]
fn jvm_imports_resolve_by_declared_package() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    jvm_project(r);
    index(r);

    let main = "app/src/main/java/org/acme/app/Main.java";
    let all = "app/src/main/java/org/acme/app/All.java";
    let tool = "app/src/main/kotlin/org/acme/app/Tool.kt";
    let strings = "core/src/main/java/org/acme/util/Strings.java";
    let numbers = "app/src/main/java/org/acme/util/Numbers.java";
    let misc = "app/src/main/kotlin/org/acme/app/helpers/Misc.kt";
    assert_eq!(
        edges(r),
        vec![
            edge(all, numbers),
            edge(all, strings),
            edge(main, numbers),
            edge(main, strings),
            edge(tool, misc),
            edge(tool, strings),
        ]
    );
    let d = deps(r);
    for lang in d.internal_resolution_by_language().unwrap() {
        assert_eq!(lang.internal, lang.resolved, "{lang:?}");
    }
}

#[test]
fn changing_package_line_moves_edges_incrementally() {
    let moved = "package org.acme.other;\n\npublic final class Numbers {}\n\nclass Helper {}\n";
    let numbers = "app/src/main/java/org/acme/util/Numbers.java";

    let t = TempDir::new().unwrap();
    let r = t.path();
    jvm_project(r);
    index(r);
    write(r, numbers, moved);
    Indexer::new(CacheManager::new(r), IndexConfig::default())
        .update_paths(r, &[r.join(numbers)])
        .expect("update_paths");
    let incremental = edges(r);

    let fresh = TempDir::new().unwrap();
    jvm_project(fresh.path());
    write(fresh.path(), numbers, moved);
    index(fresh.path());

    assert_eq!(incremental, edges(fresh.path()));
    assert!(
        !incremental.iter().any(|(_, to)| to == numbers),
        "{incremental:?}"
    );
}

#[test]
fn csharp_using_links_every_declaring_file() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "src/Core/Strings.cs",
        "namespace Acme.Util\n{\n    public class Strings {}\n}\n",
    );
    write(
        r,
        "src/Core/Numbers.cs",
        "namespace Acme.Util;\n\npublic class Numbers {}\n",
    );
    write(
        r,
        "src/Core/Nested.cs",
        "namespace Acme\n{\n    namespace Util.Extra\n    {\n        class X {}\n    }\n}\n",
    );
    let mut program = String::from("using System;\nusing Acme.Util;\n");
    // A NuGet package: "internal" to the classifier, declared by no file here
    for i in 0..120 {
        program.push_str(&format!("using Newtonsoft.Json.N{i};\n"));
    }
    program.push_str("namespace Acme.App { class Program {} }\n");
    write(r, "src/App/Program.cs", &program);
    index(r);

    assert_eq!(
        edges(r),
        vec![
            edge("src/App/Program.cs", "src/Core/Numbers.cs"),
            edge("src/App/Program.cs", "src/Core/Strings.cs"),
        ]
    );
    let d = deps(r);
    let cs = d
        .internal_resolution_by_language()
        .unwrap()
        .into_iter()
        .find(|l| l.language == "CSharp")
        .unwrap();
    assert_eq!(
        (cs.internal, cs.resolved),
        (1, 1),
        "NuGet usings are not counted"
    );
    assert!(d.low_resolution_warnings().unwrap().is_empty());
}

#[test]
fn csharp_namespaces_using_each_other_are_not_a_cycle() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(r, "A.cs", "using Beta;\nnamespace Alpha { class A {} }\n");
    write(r, "B.cs", "using Alpha;\nnamespace Beta { class B {} }\n");
    index(r);
    assert_eq!(edges(r).len(), 2);
    assert!(deps(r).detect_circular_dependencies().unwrap().is_empty());
}

#[test]
fn python_package_imports_resolve_to_init_py() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(r, "pyproject.toml", "[project]\nname = \"acme\"\n");
    write(r, "acme/__init__.py", "");
    write(r, "acme/db/__init__.py", "from .models import Model\n");
    write(r, "acme/db/models.py", "class Model: pass\n");
    write(
        r,
        "acme/views.py",
        "from acme.db import models\nimport acme.db.models\nfrom . import db\n",
    );
    index(r);

    assert_eq!(
        edges(r),
        vec![
            edge("acme/db/__init__.py", "acme/db/models.py"),
            edge("acme/views.py", "acme/__init__.py"),
            edge("acme/views.py", "acme/db/__init__.py"),
            edge("acme/views.py", "acme/db/models.py"),
        ]
    );
}

/// A file-resolved language (no package keys) is counted and warns too.
#[test]
fn file_resolved_language_with_missing_targets_warns() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    let mut body = String::new();
    for i in 0..120 {
        body.push_str(&format!("import {{ x{i} }} from './missing{i}';\n"));
    }
    write(r, "web/a.ts", &body);
    index(r);
    let warnings = deps(r).low_resolution_warnings().unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(
        warnings[0].starts_with("TypeScript: 0 of 120"),
        "{}",
        warnings[0]
    );
}

/// A module that commits `vendor/` (as Kubernetes does): `main.go` imports a
/// vendored package, which imports another vendored package.
fn go_vendored_module(r: &Path, modules_txt: bool) {
    write(r, "go.mod", "module example.com/m\n\ngo 1.22\n");
    write(
        r,
        "main.go",
        "package main\n\nimport \"golang.org/x/sys/unix\"\n\nfunc main() { unix.Mmap() }\n",
    );
    write(
        r,
        "vendor/golang.org/x/sys/unix/mmap.go",
        "package unix\n\nimport \"golang.org/x/sys/internal/unsafeheader\"\n\nfunc Mmap() { unsafeheader.Use() }\n",
    );
    write(
        r,
        "vendor/golang.org/x/sys/unix/zerrors.go",
        "package unix\n\nconst EINVAL = 22\n",
    );
    write(
        r,
        "vendor/golang.org/x/sys/internal/unsafeheader/h.go",
        "package unsafeheader\n\nfunc Use() {}\n",
    );
    if modules_txt {
        write(
            r,
            "vendor/modules.txt",
            "# golang.org/x/sys v0.20.0\n## explicit; go 1.18\ngolang.org/x/sys/unix\n",
        );
    }
}

fn island_paths(d: &DependencyIndex) -> HashSet<String> {
    d.find_islands()
        .unwrap()
        .iter()
        .flat_map(|i| paths(d, i))
        .collect()
}

/// A repo that commits `vendor/` gets the graph a repo that gitignores it gets:
/// vendored files are searchable, but no island, unused file or edge.
#[test]
fn go_vendor_is_not_in_graph() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    go_vendored_module(r, true);
    index(r);

    let d = deps(r);
    assert_eq!(d.vendored_file_count().unwrap(), 3);
    let islands = island_paths(&d);
    assert!(
        islands.iter().all(|p| !p.starts_with("vendor/")),
        "{islands:?}"
    );
    let unused = paths(&d, &d.find_unused_files().unwrap());
    assert!(
        unused.iter().all(|p| !p.starts_with("vendor/")),
        "{unused:?}"
    );
    assert!(edges(r).is_empty(), "{:?}", edges(r));

    let main = d.get_file_id_by_path("main.go").unwrap().unwrap();
    let rows = d.get_dependencies(main).unwrap();
    let unix = rows
        .iter()
        .find(|i| i.imported_path == "golang.org/x/sys/unix")
        .expect("{rows:?}");
    assert_eq!(unix.import_type, reflex::models::ImportType::External);

    let (stdout, _) = rfx(r, &["analyze", "--json", "--no-update"]);
    let summary: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(summary["vendored_files"], 3, "{summary}");

    // Still searchable.
    let (stdout, _) = rfx(r, &["query", "EINVAL", "--json", "--no-update"]);
    assert!(
        stdout.contains("vendor/golang.org/x/sys/unix/zerrors.go"),
        "{stdout}"
    );
}

/// Without `modules.txt`, Go does not read `vendor/`: it is project code.
#[test]
fn go_vendor_without_modules_txt_stays_project_code() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    go_vendored_module(r, false);
    index(r);
    let d = deps(r);
    assert_eq!(d.vendored_file_count().unwrap(), 0);
    assert!(
        island_paths(&d).contains("vendor/golang.org/x/sys/unix/zerrors.go"),
        "{:?}",
        island_paths(&d)
    );
}

/// Kubernetes `third_party/forked/` is the module's own code, imported by its
/// module path: Go has no name rule, so it stays in the graph.
#[test]
fn go_third_party_forked_stays_in_graph() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    go_vendored_module(r, true);
    write(
        r,
        "cmd/tool/main.go",
        "package main\n\nimport \"example.com/m/third_party/forked/junit\"\n\nfunc main() { junit.Write() }\n",
    );
    write(
        r,
        "third_party/forked/junit/junit.go",
        "package junit\n\nfunc Write() {}\n",
    );
    index(r);
    assert_eq!(
        edges(r),
        vec![edge(
            "cmd/tool/main.go",
            "third_party/forked/junit/junit.go"
        )]
    );
    assert_eq!(deps(r).vendored_file_count().unwrap(), 3);
}

/// `[index.vendored] patterns`: a pattern adds a directory, `!` takes one out.
#[test]
fn vendored_override_adds_and_negates() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    go_vendored_module(r, true);
    write(r, "libs/acme/acme.go", "package acme\n\nfunc A() {}\n");
    write(
        r,
        ".reflex/config.toml",
        "[index.vendored]\npatterns = [\"libs/acme/\", \"!vendor/golang.org/x/sys/internal/\"]\n",
    );
    rfx(r, &["index"]);
    let d = deps(r);
    let islands = island_paths(&d);
    assert!(!islands.contains("libs/acme/acme.go"), "{islands:?}");
    assert!(
        islands.contains("vendor/golang.org/x/sys/internal/unsafeheader/h.go"),
        "{islands:?}"
    );
    // unix (2 files) and acme are vendored; unsafeheader is not.
    assert_eq!(d.vendored_file_count().unwrap(), 3);
}

#[test]
fn deps_on_vendored_file_warns() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    go_vendored_module(r, true);
    index(r);
    let file = "vendor/golang.org/x/sys/unix/mmap.go";
    let warnings = deps(r).graph_warnings_for(file).unwrap();
    assert_eq!(
        warnings,
        vec![format!(
            "{file} is vendored; vendored files are not in the import graph"
        )]
    );
    let (stdout, stderr) = rfx(r, &["deps", file, "--json", "--no-update"]);
    serde_json::from_str::<serde_json::Value>(&stdout).expect("stdout is JSON");
    assert!(stderr.contains("Warning: vendor/golang.org"), "{stderr}");
    assert!(deps(r).graph_warnings_for("main.go").unwrap().is_empty());
}

/// A cache from before `files.vendored` gets the column on the next index.
#[test]
fn index_upgrades_a_cache_without_vendored_column() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    go_vendored_module(r, true);
    index(r);
    {
        let conn = reflex::cache::open_meta_db(r.join(".reflex/meta.db")).unwrap();
        conn.execute_batch(
            "DROP VIEW import_edges;
             ALTER TABLE files DROP COLUMN vendored;
             UPDATE statistics SET value = 'old' WHERE key = 'schema_hash';",
        )
        .unwrap();
    }
    index(r);
    assert_eq!(deps(r).vendored_file_count().unwrap(), 3);
}

#[test]
fn incremental_new_file_under_vendor_is_flagged() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    go_vendored_module(r, true);
    index(r);
    write(
        r,
        "vendor/golang.org/x/sys/unix/zsys.go",
        "package unix\n\nconst X = 1\n",
    );
    Indexer::new(CacheManager::new(r), IndexConfig::default())
        .update_paths(r, &[r.join("vendor/golang.org/x/sys/unix/zsys.go")])
        .expect("update_paths");
    let d = deps(r);
    assert_eq!(d.vendored_file_count().unwrap(), 4);
    assert!(edges(r).is_empty(), "{:?}", edges(r));
}

#[test]
fn adding_modules_txt_flags_vendor_on_next_index() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    go_vendored_module(r, false);
    index(r);
    assert_eq!(deps(r).vendored_file_count().unwrap(), 0);
    go_vendored_module(r, true);
    index(r);
    assert_eq!(deps(r).vendored_file_count().unwrap(), 3);
    fs::remove_file(r.join("vendor/modules.txt")).unwrap();
    index(r);
    assert_eq!(deps(r).vendored_file_count().unwrap(), 0);
}

/// Vendored paths among the island and unused-file answers (none expected).
fn vendored_in_answers(d: &DependencyIndex, prefix: &str) -> Vec<String> {
    let mut out: Vec<String> = island_paths(d)
        .into_iter()
        .chain(paths(d, &d.find_unused_files().unwrap()))
        .filter(|p| p.starts_with(prefix))
        .collect();
    out.sort();
    out
}

fn import_type_of(d: &DependencyIndex, file: &str, import: &str) -> reflex::models::ImportType {
    let id = d.get_file_id_by_path(file).unwrap().unwrap();
    d.get_dependencies(id)
        .unwrap()
        .into_iter()
        .find(|r| r.imported_path == import)
        .unwrap_or_else(|| panic!("{file} has no import {import}"))
        .import_type
}

#[test]
fn composer_vendor_is_not_in_graph() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "composer.json",
        r#"{"autoload": {"psr-4": {"App\\": "src/"}}}"#,
    );
    write(
        r,
        "src/Http/Controller.php",
        "<?php\nnamespace App\\Http;\nuse App\\Models\\User;\nuse Acme\\Widgets\\Widget;\nclass Controller {}\n",
    );
    write(
        r,
        "src/Models/User.php",
        "<?php\nnamespace App\\Models;\nclass User {}\n",
    );
    write(r, "vendor/composer/installed.json", "{\"packages\": []}\n");
    write(
        r,
        "vendor/autoload.php",
        "<?php\nrequire __DIR__ . '/composer/autoload_real.php';\n",
    );
    write(
        r,
        "vendor/acme/widgets/composer.json",
        r#"{"autoload": {"psr-4": {"Acme\\Widgets\\": "src/"}}}"#,
    );
    write(
        r,
        "vendor/acme/widgets/src/Widget.php",
        "<?php\nnamespace Acme\\Widgets;\nuse Acme\\Widgets\\Base;\nclass Widget extends Base {}\n",
    );
    write(
        r,
        "vendor/acme/widgets/src/Base.php",
        "<?php\nnamespace Acme\\Widgets;\nclass Base {}\n",
    );
    index(r);

    let d = deps(r);
    assert_eq!(d.vendored_file_count().unwrap(), 3);
    assert!(vendored_in_answers(&d, "vendor/").is_empty());
    assert_eq!(
        edges(r),
        vec![edge("src/Http/Controller.php", "src/Models/User.php")]
    );
}

/// `cargo vendor` output: a crate directory with `.cargo-checksum.json`. Its
/// `Cargo.toml` is not a workspace crate, so `use serde::…` stays External.
#[test]
fn cargo_vendor_crate_is_not_in_graph_and_not_internal() {
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
        "mod util;\nuse serde::de::Deserialize;\npub fn f() { util::g() }\n",
    );
    write(r, "src/util.rs", "pub fn g() {}\n");
    write(
        r,
        "vendor/serde/Cargo.toml",
        "[package]\nname = \"serde\"\nversion = \"1.0.200\"\n",
    );
    write(
        r,
        "vendor/serde/.cargo-checksum.json",
        "{\"files\":{},\"package\":\"abc\"}\n",
    );
    write(r, "vendor/serde/src/lib.rs", "pub mod de;\n");
    write(r, "vendor/serde/src/de.rs", "pub trait Deserialize {}\n");
    index(r);

    let d = deps(r);
    assert_eq!(
        import_type_of(&d, "src/lib.rs", "serde::de::Deserialize"),
        reflex::models::ImportType::External
    );
    assert_eq!(d.vendored_file_count().unwrap(), 2);
    assert!(vendored_in_answers(&d, "vendor/").is_empty());
    assert_eq!(edges(r), vec![edge("src/lib.rs", "src/util.rs")]);
}

/// `bundle install --path vendor/bundle`: installed gems and their gemspecs are
/// not the project's, so `require 'rack'` stays External.
#[test]
fn installed_gems_are_not_in_graph_and_their_gemspecs_not_projects() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "app.gemspec",
        "Gem::Specification.new do |s|\n  s.name = 'app'\nend\n",
    );
    write(r, "lib/app.rb", "require 'rack'\nrequire 'app/web'\n");
    write(r, "lib/app/web.rb", "module App; end\n");
    let gems = "vendor/bundle/ruby/3.3.0";
    write(
        r,
        &format!("{gems}/specifications/rack-3.0.0.gemspec"),
        "Gem::Specification.new do |s|\n  s.name = 'rack'\nend\n",
    );
    write(
        r,
        &format!("{gems}/gems/rack-3.0.0/rack.gemspec"),
        "Gem::Specification.new do |s|\n  s.name = 'rack'\nend\n",
    );
    write(
        r,
        &format!("{gems}/gems/rack-3.0.0/lib/rack.rb"),
        "require 'rack/builder'\n",
    );
    write(
        r,
        &format!("{gems}/gems/rack-3.0.0/lib/rack/builder.rb"),
        "module Rack; end\n",
    );
    index(r);

    let d = deps(r);
    assert_eq!(
        import_type_of(&d, "lib/app.rb", "rack"),
        reflex::models::ImportType::External
    );
    assert!(vendored_in_answers(&d, "vendor/").is_empty());
    assert_eq!(edges(r), vec![edge("lib/app.rb", "lib/app/web.rb")]);
}

#[test]
fn node_modules_are_not_in_graph() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "src/a.ts",
        "import map from 'lodash/map';\nimport { b } from './b';\nexport const a = b + map;\n",
    );
    write(r, "src/b.ts", "export const b = 1;\n");
    write(
        r,
        "node_modules/lodash/map.js",
        "const base = require('./_base');\nmodule.exports = base;\n",
    );
    write(r, "node_modules/lodash/_base.js", "module.exports = 1;\n");
    write(
        r,
        "node_modules/lodash/tsconfig.json",
        r#"{"compilerOptions": {"paths": {"./b": ["./_base.js"]}}}"#,
    );
    index(r);

    let d = deps(r);
    assert_eq!(d.vendored_file_count().unwrap(), 2);
    assert!(vendored_in_answers(&d, "node_modules/").is_empty());
    assert_eq!(edges(r), vec![edge("src/a.ts", "src/b.ts")]);
}

/// A virtualenv (`pyvenv.cfg`), `site-packages`, and pip-style `_vendor`.
#[test]
fn python_venv_site_packages_and_vendor_are_not_in_graph() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(r, "pyproject.toml", "[project]\nname = \"app\"\n");
    write(r, "app/__init__.py", "");
    write(
        r,
        "app/main.py",
        "import requests\nimport app.util\nfrom app._vendor import six\n",
    );
    write(r, "app/util.py", "X = 1\n");
    write(r, "app/_vendor/__init__.py", "");
    write(r, "app/_vendor/six.py", "PY3 = True\n");
    write(r, "venv/pyvenv.cfg", "home = /usr/bin\n");
    write(r, "venv/bin/activate_this.py", "import os\n");
    let site = "venv/lib/python3.12/site-packages";
    write(
        r,
        &format!("{site}/requests/__init__.py"),
        "from . import api\n",
    );
    write(r, &format!("{site}/requests/api.py"), "def get(): pass\n");
    index(r);

    let d = deps(r);
    assert_eq!(d.vendored_file_count().unwrap(), 5);
    let leaked: Vec<String> = vendored_in_answers(&d, "venv/")
        .into_iter()
        .chain(vendored_in_answers(&d, "app/_vendor/"))
        .collect();
    assert!(leaked.is_empty(), "{leaked:?}");
    assert_eq!(edges(r), vec![edge("app/main.py", "app/util.py")]);
}

#[test]
fn zig_path_dependency_is_not_in_graph() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "build.zig.zon",
        ".{\n    .name = .app,\n    .dependencies = .{\n        .zlib = .{ .path = \"deps/zlib\" },\n    },\n}\n",
    );
    write(
        r,
        "src/main.zig",
        "const zlib = @import(\"zlib\");\nconst util = @import(\"./util.zig\");\n",
    );
    write(r, "src/util.zig", "pub const x = 1;\n");
    write(
        r,
        "deps/zlib/src/root.zig",
        "const inflate = @import(\"./inflate.zig\");\n",
    );
    write(r, "deps/zlib/src/inflate.zig", "pub const y = 2;\n");
    index(r);

    let d = deps(r);
    assert_eq!(d.vendored_file_count().unwrap(), 2);
    assert!(vendored_in_answers(&d, "deps/").is_empty());
    assert_eq!(edges(r), vec![edge("src/main.zig", "src/util.zig")]);
}

#[test]
fn c_third_party_is_not_in_graph() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "src/main.c",
        "#include \"util.h\"\n#include \"../third_party/zlib/zlib.h\"\nint main(void) { return 0; }\n",
    );
    write(r, "src/util.h", "int util(void);\n");
    write(r, "third_party/zlib/zlib.h", "int inflate(void);\n");
    write(
        r,
        "third_party/zlib/inflate.c",
        "#include \"zlib.h\"\nint inflate(void) { return 0; }\n",
    );
    write(
        r,
        "src/native/external/brotli/decode.c",
        "int decode(void) { return 0; }\n",
    );
    index(r);

    let d = deps(r);
    assert_eq!(d.vendored_file_count().unwrap(), 3);
    let leaked: Vec<String> = vendored_in_answers(&d, "third_party/")
        .into_iter()
        .chain(vendored_in_answers(&d, "src/native/external/"))
        .collect();
    assert!(leaked.is_empty(), "{leaked:?}");
    assert_eq!(edges(r), vec![edge("src/main.c", "src/util.h")]);
}

/// Java keeps to `third_party` names: a package called `external` is project code.
#[test]
fn java_package_named_external_stays_in_graph() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "pom.xml",
        "<project>\n  <groupId>org.acme</groupId>\n  <artifactId>x</artifactId>\n</project>\n",
    );
    let src = "src/main/java/org/acme";
    write(
        r,
        &format!("{src}/App.java"),
        "package org.acme;\n\nimport org.acme.external.Client;\nimport com.google.common.collect.ImmutableList;\n\npublic class App {}\n",
    );
    write(
        r,
        &format!("{src}/external/Client.java"),
        "package org.acme.external;\n\npublic class Client {}\n",
    );
    write(
        r,
        "third_party/guava/com/google/common/collect/ImmutableList.java",
        "package com.google.common.collect;\n\npublic class ImmutableList {}\n",
    );
    index(r);

    let d = deps(r);
    assert_eq!(d.vendored_file_count().unwrap(), 1);
    assert!(vendored_in_answers(&d, "third_party/").is_empty());
    assert_eq!(
        edges(r),
        vec![edge(
            &format!("{src}/App.java"),
            &format!("{src}/external/Client.java")
        )]
    );
}

/// C# links a using to every file declaring the namespace, vendored source too.
#[test]
fn csharp_third_party_source_is_not_in_graph() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "src/Program.cs",
        "using Newtonsoft.Json;\nusing Acme.Core;\nnamespace Acme { class Program {} }\n",
    );
    write(
        r,
        "src/Core/Util.cs",
        "namespace Acme.Core { class Util {} }\n",
    );
    write(
        r,
        "third_party/Newtonsoft.Json/JsonConvert.cs",
        "namespace Newtonsoft.Json { class JsonConvert {} }\n",
    );
    index(r);

    let d = deps(r);
    assert_eq!(d.vendored_file_count().unwrap(), 1);
    assert!(vendored_in_answers(&d, "third_party/").is_empty());
    assert_eq!(edges(r), vec![edge("src/Program.cs", "src/Core/Util.cs")]);
}

/// Relative includes resolve the same whatever the process's working directory:
/// the test runs in the repository, not in the indexed root.
#[test]
fn relative_paths_resolve_from_any_working_directory() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "src/main.c",
        "#include \"../include/api.h\"\nint main(void) { return 0; }\n",
    );
    write(r, "include/api.h", "int api(void);\n");
    write(r, "lib/x.cpp", "#include \"./x.hpp\"\n");
    write(r, "lib/x.hpp", "int x();\n");
    write(
        r,
        "zig/main.zig",
        "const u = @import(\"../zig/util.zig\");\n",
    );
    write(r, "zig/util.zig", "pub const x = 1;\n");
    index(r);
    assert_eq!(
        edges(r),
        vec![
            edge("lib/x.cpp", "lib/x.hpp"),
            edge("src/main.c", "include/api.h"),
            edge("zig/main.zig", "zig/util.zig"),
        ]
    );
}

/// A path lookup that falls back to a suffix matches whole path segments:
/// `a.h` is `include/a.h`, never `lib/xa.h`.
#[test]
fn path_lookup_matches_whole_segments() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(r, "include/a.h", "int a(void);\n");
    write(r, "lib/xa.h", "int xa(void);\n");
    index(r);
    let d = deps(r);
    let a = d.get_file_id_by_path("include/a.h").unwrap();
    assert_eq!(d.get_file_id_by_path("a.h").unwrap(), a);
    let conn = reflex::cache::open_meta_db(r.join(".reflex/meta.db")).unwrap();
    let resolver = reflex::dependency::PathResolver::from_conn(&conn).unwrap();
    assert_eq!(resolver.get_file_id_by_path("a.h").unwrap(), a);
    // Windows code names files in any case: the whole path still matches
    assert_eq!(resolver.get_file_id_by_path("Include/A.h").unwrap(), a);
    assert_eq!(d.get_file_id_by_path("Include/A.h").unwrap(), a);
}

/// Two workspace crates, each with `src/lib.rs`: a cross-crate `use` reaches the
/// other crate's file, not an ambiguous `src/lib.rs`.
#[test]
fn rust_workspace_crate_import_reaches_its_crate() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "Cargo.toml",
        "[workspace]\nmembers = [\"crates/a\", \"crates/b\"]\n",
    );
    write(
        r,
        "crates/a/Cargo.toml",
        "[package]\nname = \"a\"\nversion = \"0.1.0\"\n",
    );
    write(
        r,
        "crates/a/src/lib.rs",
        "use b::thing;\npub fn f() { thing() }\n",
    );
    write(
        r,
        "crates/b/Cargo.toml",
        "[package]\nname = \"b\"\nversion = \"0.1.0\"\n",
    );
    write(r, "crates/b/src/lib.rs", "mod inner;\npub fn thing() {}\n");
    write(r, "crates/b/src/inner.rs", "pub fn i() {}\n");
    index(r);
    assert_eq!(
        edges(r),
        vec![
            edge("crates/a/src/lib.rs", "crates/b/src/lib.rs"),
            edge("crates/b/src/lib.rs", "crates/b/src/inner.rs"),
        ]
    );
}

/// A crate named `b-core` is imported as `b_core`.
#[test]
fn rust_hyphenated_crate_is_imported_with_underscores() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "Cargo.toml",
        "[workspace]\nmembers = [\"a\", \"b-core\"]\n",
    );
    write(
        r,
        "a/Cargo.toml",
        "[package]\nname = \"a\"\nversion = \"0.1.0\"\n",
    );
    write(r, "a/src/lib.rs", "use b_core::codec::Decoder;\n");
    write(
        r,
        "b-core/Cargo.toml",
        "[package]\nname = \"b-core\"\nversion = \"0.1.0\"\n",
    );
    write(r, "b-core/src/lib.rs", "pub mod codec;\n");
    write(r, "b-core/src/codec.rs", "pub trait Decoder {}\n");
    index(r);
    assert!(
        edges(r).contains(&edge("a/src/lib.rs", "b-core/src/codec.rs")),
        "{:?}",
        edges(r)
    );
}

/// Zig imports any `*.zig` path relative to the importer, `./` or not.
#[test]
fn zig_file_import_without_dot_slash_is_internal() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "src/main.zig",
        "const util = @import(\"util.zig\");\nconst x = @import(\"lsm/tree.zig\");\nconst std = @import(\"std\");\n",
    );
    write(r, "src/util.zig", "pub const a = 1;\n");
    write(
        r,
        "src/lsm/tree.zig",
        "const util = @import(\"../util.zig\");\n",
    );
    index(r);
    assert_eq!(
        edges(r),
        vec![
            edge("src/lsm/tree.zig", "src/util.zig"),
            edge("src/main.zig", "src/lsm/tree.zig"),
            edge("src/main.zig", "src/util.zig"),
        ]
    );
    let d = deps(r);
    assert_eq!(
        import_type_of(&d, "src/main.zig", "util.zig"),
        reflex::models::ImportType::Internal
    );
}

/// Named modules a `build.zig` defines: `b.addModule("stdx", …)`, and
/// `b.createModule(…)` given a name by `.addImport("vsr", vsr_module)`.
#[test]
fn zig_named_build_modules_resolve_to_their_root_file() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "build.zig",
        r#"const std = @import("std");

pub fn build(b: *std.Build) void {
    const stdx_module = b.addModule("stdx", .{ .root_source_file = b.path("src/stdx/stdx.zig") });
    const vsr_module = b.createModule(.{
        .root_source_file = b.path("src/vsr.zig"),
    });
    vsr_module.addImport("stdx", stdx_module);
    const exe = b.addExecutable(.{
        .name = "app",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
        }),
    });
    exe.root_module.addImport("vsr", options.vsr_module);
    exe.root_module.addImport("zap", b.dependency("zap", .{}).module("zap"));
}
"#,
    );
    write(
        r,
        "src/main.zig",
        "const vsr = @import(\"vsr\");\nconst stdx = @import(\"stdx\");\nconst zap = @import(\"zap\");\n",
    );
    write(r, "src/vsr.zig", "const stdx = @import(\"stdx\");\n");
    write(r, "src/stdx/stdx.zig", "pub const x = 1;\n");
    index(r);
    assert_eq!(
        edges(r),
        vec![
            edge("src/main.zig", "src/stdx/stdx.zig"),
            edge("src/main.zig", "src/vsr.zig"),
            edge("src/vsr.zig", "src/stdx/stdx.zig"),
        ]
    );
    let d = deps(r);
    assert_eq!(
        import_type_of(&d, "src/main.zig", "zap"),
        reflex::models::ImportType::External
    );
}

/// `from pkg import name` reaches `pkg/__init__.py` and, when `name` is a
/// submodule, `pkg/name.py` or `pkg/name/__init__.py` (`from django.db import
/// models`). A submodule added later is reached without re-extracting.
#[test]
fn python_from_import_reaches_submodules() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(r, "pyproject.toml", "[project]\nname = \"app\"\n");
    write(r, "app/__init__.py", "Base = object\n");
    write(r, "app/util.py", "X = 1\n");
    write(r, "app/db/__init__.py", "");
    write(
        r,
        "app/main.py",
        "from app import util, Base, db\nfrom app import later as l\n",
    );
    index(r);
    assert_eq!(
        edges(r),
        vec![
            edge("app/main.py", "app/__init__.py"),
            edge("app/main.py", "app/db/__init__.py"),
            edge("app/main.py", "app/util.py"),
        ]
    );
    let d = deps(r);
    let main = d.get_file_id_by_path("app/main.py").unwrap().unwrap();
    let info = d.get_dependencies_info(main).unwrap();
    assert_eq!(
        info[0].resolved_paths.as_deref(),
        Some(
            &[
                "app/__init__.py".to_string(),
                "app/db/__init__.py".to_string(),
                "app/util.py".to_string()
            ][..]
        ),
        "{info:?}"
    );

    write(r, "app/later.py", "Y = 2\n");
    Indexer::new(CacheManager::new(r), IndexConfig::default())
        .update_paths(r, &[r.join("app/later.py")])
        .expect("update_paths");
    assert!(
        edges(r).contains(&edge("app/main.py", "app/later.py")),
        "{:?}",
        edges(r)
    );
}

/// `from django import forms` is `django/__init__.py`: a candidate is an exact
/// path, never a file that merely ends in `django.py`.
#[test]
fn python_package_import_never_matches_a_suffix() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(r, "pyproject.toml", "[project]\nname = \"django\"\n");
    write(r, "django/__init__.py", "");
    write(r, "django/forms.py", "");
    write(r, "django/template/backends/django.py", "");
    write(r, "django/conf/settings.py", "from django import forms\n");
    index(r);
    let got: Vec<(String, String)> = edges(r)
        .into_iter()
        .filter(|(s, _)| s == "django/conf/settings.py")
        .collect();
    assert_eq!(
        got,
        vec![
            edge("django/conf/settings.py", "django/__init__.py"),
            edge("django/conf/settings.py", "django/forms.py"),
        ]
    );
}

/// Ruby searches every gem's `lib/`: in the rails monorepo `require
/// "rails/command"` is `railties/lib/rails/command.rb`, not in the `rails` gem.
#[test]
fn ruby_require_searches_every_gem_lib() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    let spec = |name: &str| format!("Gem::Specification.new do |s|\n  s.name = '{name}'\nend\n");
    write(r, "rails.gemspec", &spec("rails"));
    write(r, "railties/railties.gemspec", &spec("railties"));
    write(r, "railties/lib/rails/command.rb", "module Rails; end\n");
    write(
        r,
        "railties/lib/rails/app.rb",
        "require \"rails/command\"\n",
    );
    index(r);
    assert_eq!(
        edges(r),
        vec![edge(
            "railties/lib/rails/app.rb",
            "railties/lib/rails/command.rb"
        )]
    );
}

/// `require "active_support/..."` reaches the gem `activesupport`, whose lib/
/// provides `active_support`; `require_relative "helper"` is a sibling file.
#[test]
fn ruby_require_of_a_name_a_gem_lib_provides_is_internal() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    let spec = |name: &str| format!("Gem::Specification.new do |s|\n  s.name = '{name}'\nend\n");
    write(
        r,
        "activesupport/activesupport.gemspec",
        &spec("activesupport"),
    );
    write(
        r,
        "activesupport/lib/active_support/core_ext.rb",
        "module ActiveSupport; end\n",
    );
    write(r, "actionpack/actionpack.gemspec", &spec("actionpack"));
    write(
        r,
        "actionpack/lib/action_dispatch.rb",
        "require \"active_support/core_ext\"\nrequire_relative \"helper\"\n",
    );
    write(r, "actionpack/lib/helper.rb", "module Helper; end\n");
    index(r);
    assert_eq!(
        edges(r),
        vec![
            edge(
                "actionpack/lib/action_dispatch.rb",
                "actionpack/lib/helper.rb"
            ),
            edge(
                "actionpack/lib/action_dispatch.rb",
                "activesupport/lib/active_support/core_ext.rb"
            ),
        ]
    );
}

/// composer.json decides what a PHP `use` is: a project PSR-4 prefix
/// (`autoload` or `autoload-dev`, a directory list too) is Internal, anything
/// else External. An alias is not an import; a group use carries its prefix.
#[test]
fn php_uses_follow_composer_autoload() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "composer.json",
        r#"{"autoload": {"psr-4": {"App\\": "app", "Lib\\": ["src/", "lib/"]}},
            "autoload-dev": {"psr-4": {"Tests\\": "tests/"}}}"#,
    );
    write(
        r,
        "tests/Feature/UserTest.php",
        "<?php\nnamespace Tests\\Feature;\nuse App\\Models\\User;\nuse App\\Models\\{Post, Comment as C};\nuse Lib\\Helper as H;\nuse Tests\\TestCase;\nuse Illuminate\\Support\\Str;\nrequire 'bootstrap.php';\nclass UserTest extends TestCase {}\n",
    );
    write(
        r,
        "tests/TestCase.php",
        "<?php\nnamespace Tests;\nclass TestCase {}\n",
    );
    write(r, "tests/Feature/bootstrap.php", "<?php\n");
    write(
        r,
        "app/Models/User.php",
        "<?php\nnamespace App\\Models;\nclass User {}\n",
    );
    write(
        r,
        "app/Models/Post.php",
        "<?php\nnamespace App\\Models;\nclass Post {}\n",
    );
    write(
        r,
        "app/Models/Comment.php",
        "<?php\nnamespace App\\Models;\nclass Comment {}\n",
    );
    write(
        r,
        "lib/Helper.php",
        "<?php\nnamespace Lib;\nclass Helper {}\n",
    );
    index(r);

    let from = "tests/Feature/UserTest.php";
    assert_eq!(
        edges(r),
        vec![
            edge(from, "app/Models/Comment.php"),
            edge(from, "app/Models/Post.php"),
            edge(from, "app/Models/User.php"),
            edge(from, "lib/Helper.php"),
            edge(from, "tests/Feature/bootstrap.php"),
            edge(from, "tests/TestCase.php"),
        ]
    );
    let d = deps(r);
    let id = d.get_file_id_by_path(from).unwrap().unwrap();
    let rows: Vec<(String, reflex::models::ImportType)> = d
        .get_dependencies(id)
        .unwrap()
        .into_iter()
        .map(|r| (r.imported_path, r.import_type))
        .collect();
    assert!(
        rows.iter()
            .all(|(p, _)| p != "C" && p != "H" && p != "Post"),
        "{rows:?}"
    );
    assert!(
        rows.contains(&(
            "Illuminate\\Support\\Str".to_string(),
            reflex::models::ImportType::External
        )),
        "{rows:?}"
    );
    assert!(d.low_resolution_warnings().unwrap().is_empty());
}

/// `using static A.B.C` and `using X = A.B.C` name a type (or a namespace) `C`
/// in `A.B`: they reach the files declaring it. The alias `X` is not an import.
#[test]
fn csharp_using_static_and_alias_reach_the_named_type() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "src/Program.cs",
        "using static Acme.Util.Math;\nusing Json = Acme.Serialization;\nusing L = Acme.Util.Log<int>;\nglobal using Acme.Core;\nnamespace App { class Program {} }\n",
    );
    write(
        r,
        "src/Util/Math.cs",
        "namespace Acme.Util { public static class Math {} }\n",
    );
    write(
        r,
        "src/Util/Other.cs",
        "namespace Acme.Util { class Other {} }\n",
    );
    write(
        r,
        "src/Util/Log.cs",
        "namespace Acme.Util;\npublic class Log<T> {}\n",
    );
    write(
        r,
        "src/Serialization/Json.cs",
        "namespace Acme.Serialization { class JsonWriter {} }\n",
    );
    write(
        r,
        "src/Core/Core.cs",
        "namespace Acme.Core { class C {} }\n",
    );
    index(r);
    assert_eq!(
        edges(r),
        vec![
            edge("src/Program.cs", "src/Core/Core.cs"),
            edge("src/Program.cs", "src/Serialization/Json.cs"),
            edge("src/Program.cs", "src/Util/Log.cs"),
            edge("src/Program.cs", "src/Util/Math.cs"),
        ]
    );
    let d = deps(r);
    let id = d.get_file_id_by_path("src/Program.cs").unwrap().unwrap();
    let paths: Vec<String> = d
        .get_dependencies(id)
        .unwrap()
        .into_iter()
        .map(|r| r.imported_path)
        .collect();
    assert!(!paths.iter().any(|p| p == "Json" || p == "L"), "{paths:?}");
}

/// Usings and namespaces inside `#if` blocks count.
#[test]
fn csharp_usings_and_namespaces_inside_preprocessor_blocks() {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "src/A.cs",
        "#if NET8_0\nusing Acme.Core;\n#endif\nnamespace App { class A {} }\n",
    );
    write(
        r,
        "src/Core.cs",
        "#if DEBUG\nnamespace Acme.Core { class C {} }\n#endif\n",
    );
    index(r);
    assert_eq!(edges(r), vec![edge("src/A.cs", "src/Core.cs")]);
}
