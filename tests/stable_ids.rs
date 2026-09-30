//! Stage 0 of incremental indexing: `meta.db` keeps one row per path with a stable
//! id across reindexing, so rows that hang off a file (symbols, other branches'
//! hashes, dependencies of unchanged files) survive, and every output that used to
//! list files in id order still lists them in the order a fresh build would.

use reflex::background_indexer::BackgroundIndexer;
use reflex::cache::open_meta_db;
use reflex::{CacheManager, IndexConfig, Indexer};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

fn index(root: &Path) {
    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .expect("index");
}

fn meta(root: &Path) -> rusqlite::Connection {
    open_meta_db(root.join(".reflex").join("meta.db")).expect("open meta.db")
}

fn ids(root: &Path) -> BTreeMap<String, i64> {
    let conn = meta(root);
    let mut stmt = conn.prepare("SELECT path, id FROM files").unwrap();
    stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

fn count(root: &Path, sql: &str) -> i64 {
    meta(root).query_row(sql, [], |r| r.get(0)).unwrap()
}

fn stat(root: &Path, key: &str) -> Option<String> {
    use rusqlite::OptionalExtension;
    meta(root)
        .query_row("SELECT value FROM statistics WHERE key = ?", [key], |r| {
            r.get(0)
        })
        .optional()
        .unwrap()
}

fn set_stat(root: &Path, key: &str, value: &str) {
    meta(root)
        .execute(
            "INSERT OR REPLACE INTO statistics (key, value, updated_at) VALUES (?, ?, 0)",
            [key, value],
        )
        .unwrap();
}

fn workspace() -> TempDir {
    let temp = TempDir::new().unwrap();
    let r = temp.path();
    fs::create_dir_all(r.join("src")).unwrap();
    fs::write(r.join("src/a.rs"), "pub fn alpha() {}\n").unwrap();
    fs::write(r.join("src/b.rs"), "pub fn beta() { crate::a::alpha(); }\n").unwrap();
    fs::write(r.join("src/c.rs"), "pub fn gamma() {}\n").unwrap();
    temp
}

fn run_symbol_pass(root: &Path) -> reflex::background_indexer::IndexingStatus {
    let mut pass = BackgroundIndexer::new(root).unwrap();
    pass.run().unwrap();
    BackgroundIndexer::get_status(&root.join(".reflex"))
        .unwrap()
        .expect("status written")
}

#[test]
fn ids_are_stable_across_reindexing() {
    let temp = workspace();
    let root = temp.path();
    index(root);
    let before = ids(root);

    fs::write(root.join("src/b.rs"), "pub fn beta() { /* edited */ }\n").unwrap();
    fs::remove_file(root.join("src/c.rs")).unwrap();
    fs::write(root.join("src/d.rs"), "pub fn delta() {}\n").unwrap();
    index(root);
    let after = ids(root);

    assert_eq!(after["src/a.rs"], before["src/a.rs"]);
    assert_eq!(
        after["src/b.rs"], before["src/b.rs"],
        "an edited file keeps its id"
    );
    assert!(
        !after.contains_key("src/c.rs"),
        "a deleted file loses its row"
    );
    assert!(
        after["src/d.rs"] > *before.values().max().unwrap(),
        "an added file gets a new id"
    );
}

#[test]
fn symbol_cache_survives_a_one_file_edit() {
    let temp = workspace();
    let root = temp.path();
    index(root);
    let first = run_symbol_pass(root);
    assert_eq!(first.parsed_files, 3);
    let rows = count(root, "SELECT COUNT(*) FROM symbols");
    assert!(rows >= 3, "every file has a symbol row: {rows}");

    fs::write(root.join("src/b.rs"), "pub fn beta2() {}\n").unwrap();
    index(root);
    assert!(
        count(root, "SELECT COUNT(*) FROM symbols") >= rows,
        "a reindex no longer wipes the symbol cache"
    );

    let second = run_symbol_pass(root);
    assert_eq!(
        second.parsed_files, 1,
        "only the edited file is parsed again"
    );
    assert_eq!(second.cached_files, 2);
    // The old version's row is gone once the pass cleans up: no branch holds it.
    assert_eq!(count(root, "SELECT COUNT(*) FROM symbols"), 3);
}

fn git(root: &Path, args: &[&str]) {
    let ok = Command::new("git")
        .args(args)
        .current_dir(root)
        .env("GIT_AUTHOR_NAME", "t")
        .env("GIT_AUTHOR_EMAIL", "t@example.com")
        .env("GIT_COMMITTER_NAME", "t")
        .env("GIT_COMMITTER_EMAIL", "t@example.com")
        .output()
        .expect("git")
        .status
        .success();
    assert!(ok, "git {:?}", args);
}

#[test]
fn other_branches_rows_survive_a_reindex() {
    let temp = workspace();
    let root = temp.path();
    git(root, &["init", "-q", "-b", "main"]);
    git(root, &["add", "."]);
    git(root, &["commit", "-q", "-m", "init"]);
    index(root);

    git(root, &["checkout", "-q", "-b", "feature"]);
    fs::write(root.join("src/b.rs"), "pub fn beta() { /* feature */ }\n").unwrap();
    index(root);

    let per_branch = |name: &str| {
        count(
            root,
            &format!(
                "SELECT COUNT(*) FROM file_branches fb JOIN branches b ON b.id = fb.branch_id \
                 WHERE b.name = '{name}'"
            ),
        )
    };
    assert_eq!(per_branch("feature"), 3);
    assert_eq!(per_branch("main"), 3, "main's rows were kept");
    // main still records the version of b.rs it indexed.
    let main_b: String = meta(root)
        .query_row(
            "SELECT fb.hash FROM file_branches fb JOIN branches b ON b.id = fb.branch_id \
             JOIN files f ON f.id = fb.file_id WHERE b.name = 'main' AND f.path = 'src/b.rs'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let files_b: String = meta(root)
        .query_row("SELECT hash FROM files WHERE path = 'src/b.rs'", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_ne!(main_b, files_b);
}

#[test]
fn a_schema_change_forces_a_full_rebuild() {
    use std::os::unix::fs::MetadataExt;
    let temp = workspace();
    let root = temp.path();
    index(root);
    let content = root.join(".reflex/content.bin");
    let inode = fs::metadata(&content).unwrap().ino();

    // Nothing changed on disk; only the cache's schema stamp is foreign.
    set_stat(root, "schema_hash", "0000000000000000");
    index(root);

    assert_ne!(
        fs::metadata(&content).unwrap().ino(),
        inode,
        "content.bin was rewritten"
    );
    assert_ne!(
        stat(root, "schema_hash").as_deref(),
        Some("0000000000000000")
    );
}

#[test]
fn an_index_run_keeps_the_last_compaction_time() {
    let temp = workspace();
    let root = temp.path();
    index(root);
    set_stat(root, "last_compaction", "1234567890");
    index(root);
    assert_eq!(stat(root, "last_compaction").as_deref(), Some("1234567890"));
}

#[test]
fn an_extraction_change_clears_the_symbol_cache() {
    let temp = workspace();
    let root = temp.path();
    index(root);
    run_symbol_pass(root);
    assert!(count(root, "SELECT COUNT(*) FROM symbols") > 0);

    set_stat(root, "extraction_hash", "0000000000000000");
    fs::write(root.join("src/c.rs"), "pub fn gamma2() {}\n").unwrap();
    index(root);
    assert_eq!(count(root, "SELECT COUNT(*) FROM symbols"), 0);
    assert_ne!(
        stat(root, "extraction_hash").as_deref(),
        Some("0000000000000000")
    );
}

#[test]
fn a_file_that_loses_its_exports_loses_their_rows() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    fs::write(
        root.join("index.ts"),
        "export * from './util';\nexport { a } from './a';\n",
    )
    .unwrap();
    fs::write(root.join("util.ts"), "export const u = 1;\n").unwrap();
    fs::write(root.join("a.ts"), "export const a = 1;\n").unwrap();
    index(root);
    let exports = |root: &Path| {
        count(
            root,
            "SELECT COUNT(*) FROM file_exports e JOIN files f ON f.id = e.file_id \
             WHERE f.path = 'index.ts'",
        )
    };
    assert_eq!(exports(root), 2);

    fs::write(root.join("index.ts"), "const nothing = 0;\n").unwrap();
    index(root);
    assert_eq!(exports(root), 0);
}

/// Every id-ordered dependency output, rendered with paths.
fn graph_outputs(root: &Path) -> Vec<String> {
    use reflex::dependency::DependencyIndex;
    let deps = DependencyIndex::new(CacheManager::new(root));
    let path = |id: i64| {
        deps.get_file_paths(&[id])
            .unwrap()
            .remove(&id)
            .unwrap_or_default()
    };
    let target = deps.get_file_id_by_path("src/core.rs").unwrap().unwrap();
    let mut out = vec![format!(
        "dependents: {:?}",
        deps.get_dependents(target)
            .unwrap()
            .into_iter()
            .map(path)
            .collect::<Vec<_>>()
    )];
    out.push(format!(
        "unused: {:?}",
        deps.find_unused_files()
            .unwrap()
            .into_iter()
            .map(path)
            .collect::<Vec<_>>()
    ));
    out.push(format!(
        "hotspots: {:?}",
        deps.find_hotspots(None, 1)
            .unwrap()
            .into_iter()
            .map(|(id, n)| (path(id), n))
            .collect::<Vec<_>>()
    ));
    out.push(format!(
        "islands: {:?}",
        deps.find_islands()
            .unwrap()
            .into_iter()
            .map(|i| i.into_iter().map(path).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    ));
    out.push(format!(
        "cycles: {:?}",
        deps.detect_circular_dependencies()
            .unwrap()
            .into_iter()
            .map(|c| c.into_iter().map(path).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    ));
    out
}

/// The outputs of the index as it stands, and of a fresh build of the same
/// directory (same readdir order, so the same walk order).
fn against_fresh(root: &Path) -> (Vec<String>, Vec<String>) {
    let updated = graph_outputs(root);
    fs::rename(root.join(".reflex"), root.join(".reflex-updated")).unwrap();
    index(root);
    let fresh = graph_outputs(root);
    fs::remove_dir_all(root.join(".reflex")).unwrap();
    fs::rename(root.join(".reflex-updated"), root.join(".reflex")).unwrap();
    (updated, fresh)
}

#[test]
fn id_ordered_outputs_match_a_fresh_build_after_updates() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("Cargo.toml"), "[package]\nname = \"demo\"\n").unwrap();
    fs::write(root.join("src/core.rs"), "pub fn core() {}\n").unwrap();
    for name in ["m1", "m2", "m3"] {
        fs::write(
            root.join(format!("src/{name}.rs")),
            format!("use crate::core::core;\npub fn {name}() {{ core(); }}\n"),
        )
        .unwrap();
    }
    fs::write(
        root.join("src/lib.rs"),
        "mod core;\nmod m1;\nmod m2;\nmod m3;\n",
    )
    .unwrap();
    index(root);

    // Edits that move rows: an importer is rewritten (its rows get new row ids),
    // a new importer arrives (new id), one goes away, and a cycle appears.
    fs::write(
        root.join("src/m1.rs"),
        "use crate::m2::m2;\nuse crate::core::core;\npub fn m1() { core(); m2(); }\n",
    )
    .unwrap();
    fs::write(
        root.join("src/m2.rs"),
        "use crate::m1::m1;\nuse crate::core::core;\npub fn m2() { core(); m1(); }\n",
    )
    .unwrap();
    fs::write(
        root.join("src/a0.rs"),
        "use crate::core::core;\npub fn a0() { core(); }\n",
    )
    .unwrap();
    fs::remove_file(root.join("src/m3.rs")).unwrap();
    fs::write(
        root.join("src/lib.rs"),
        "mod core;\nmod a0;\nmod m1;\nmod m2;\n",
    )
    .unwrap();
    index(root);

    let (updated, fresh) = against_fresh(root);
    assert_eq!(updated, fresh);
}

#[test]
fn id_ordered_outputs_follow_walk_seq_not_id() {
    use reflex::dependency::DependencyIndex;
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("Cargo.toml"), "[package]\nname = \"demo\"\n").unwrap();
    fs::write(root.join("src/core.rs"), "pub fn core() {}\n").unwrap();
    fs::write(root.join("src/util.rs"), "pub fn util() {}\n").unwrap();
    for name in ["m1", "m2", "m3"] {
        fs::write(
            root.join(format!("src/{name}.rs")),
            format!(
                "use crate::core::core;\nuse crate::util::util;\npub fn {name}() {{ core(); util(); }}\n"
            ),
        )
        .unwrap();
    }
    index(root);

    let deps = DependencyIndex::new(CacheManager::new(root));
    let paths = |ids: Vec<i64>| {
        let map = deps.get_file_paths(&ids).unwrap();
        ids.iter().map(|id| map[id].clone()).collect::<Vec<_>>()
    };
    let core = deps.get_file_id_by_path("src/core.rs").unwrap().unwrap();
    let dependents = paths(deps.get_dependents(core).unwrap());
    let unused = paths(deps.find_unused_files().unwrap());
    let hot = paths(
        deps.find_hotspots(None, 1)
            .unwrap()
            .into_iter()
            .map(|h| h.0)
            .collect(),
    );

    // Reverse the walk order in place: ids stay, positions flip.
    meta(root)
        .execute("UPDATE files SET walk_seq = -walk_seq", [])
        .unwrap();

    let rev = |mut v: Vec<String>| {
        v.reverse();
        v
    };
    assert_eq!(paths(deps.get_dependents(core).unwrap()), rev(dependents));
    assert_eq!(paths(deps.find_unused_files().unwrap()), rev(unused));
    // core and util tie at 3 importers each: the tie follows walk order.
    assert_eq!(
        paths(
            deps.find_hotspots(None, 1)
                .unwrap()
                .into_iter()
                .map(|h| h.0)
                .collect()
        ),
        rev(hot)
    );
}
