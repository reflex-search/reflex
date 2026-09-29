//! Stage 1 of incremental indexing: after a small change, `rfx index` publishes a
//! delta segment (added and modified files) and tombstones over the base instead of
//! rewriting both stores. Every reader must then answer exactly as a fresh build of
//! the same tree does, and a binary without delta support must stop, not serve the
//! base alone.

use reflex::query::{QueryEngine, QueryFilter};
use reflex::snapshot::{IndexSnapshot, Manifest, read_manifest};
use reflex::{CacheManager, IndexConfig, Indexer};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

/// Index `root` with merge limits large enough that a small tree keeps its delta.
fn index(root: &Path) {
    index_with_limits(root, usize::MAX, u64::MAX);
}

fn index_with_limits(root: &Path, files: usize, bytes: u64) {
    let mut indexer = Indexer::new(CacheManager::new(root), IndexConfig::default());
    indexer.set_merge_limits(files, bytes);
    indexer.index(root, false).expect("index");
}

fn manifest(root: &Path) -> Manifest {
    read_manifest(&root.join(".reflex"))
        .expect("read manifest")
        .expect("a manifest")
}

fn write(root: &Path, rel: &str, body: &str) {
    let p = root.join(rel);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, body).unwrap();
}

/// A tree of Rust, Python, TypeScript and Markdown files that share tokens, so a
/// tombstone removes postings from lists other files still hold.
fn workspace() -> TempDir {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    for i in 0..60 {
        write(
            root,
            &format!("src/m{i:02}.rs"),
            &format!(
                "pub fn handler_{i}(input: &str) -> Result<usize, String> {{\n    \
                 let shared_token = input.len() + {i};\n    Ok(shared_token)\n}}\n\n\
                 pub struct Widget{i} {{ pub realm_id: u32 }}\n"
            ),
        );
    }
    for i in 0..30 {
        write(
            root,
            &format!("py/p{i:02}.py"),
            &format!("def compute_{i}(shared_token):\n    return shared_token * {i}\n"),
        );
        write(
            root,
            &format!("web/w{i:02}.ts"),
            &format!("export function render{i}(RealmId: number) {{ return RealmId + {i}; }}\n"),
        );
    }
    write(
        root,
        "README.md",
        "# Demo\n\nThe shared_token flows through handler_1.\n",
    );
    write(root, "Cargo.lock", "[[package]]\nname = \"shared_token\"\n");
    temp
}

fn filter() -> QueryFilter {
    QueryFilter {
        suppress_output: true,
        ..Default::default()
    }
}

/// The answers to a fixed set of queries, as JSON (timings are not part of an answer).
fn battery(root: &Path) -> Vec<(String, Value)> {
    let engine = QueryEngine::new(CacheManager::new(root));
    let cases: Vec<(&str, QueryFilter)> = vec![
        ("shared_token", filter()),
        ("handler_1", filter()),
        (
            "handler_",
            QueryFilter {
                use_contains: true,
                ..filter()
            },
        ),
        (
            "realmid",
            QueryFilter {
                ignore_case: true,
                ..filter()
            },
        ),
        (
            "realm",
            QueryFilter {
                ignore_case: true,
                use_contains: true,
                ..filter()
            },
        ),
        (
            r"fn handler_\d+",
            QueryFilter {
                use_regex: true,
                ..filter()
            },
        ),
        (
            r"\w+_?id",
            QueryFilter {
                use_regex: true,
                ..filter()
            },
        ),
        (
            "Widget",
            QueryFilter {
                symbols_mode: true,
                use_contains: true,
                ..filter()
            },
        ),
        (
            "shared_token",
            QueryFilter {
                count_only: true,
                ..filter()
            },
        ),
        (
            "shared_token",
            QueryFilter {
                paths_only: true,
                ..filter()
            },
        ),
        (
            "shared_token",
            QueryFilter {
                limit: Some(5),
                offset: Some(5),
                ..filter()
            },
        ),
        (
            "shared_token",
            QueryFilter {
                glob_patterns: vec!["py/**".into()],
                ..filter()
            },
        ),
        (
            "shared_token",
            QueryFilter {
                include_locks: true,
                ..filter()
            },
        ),
        ("zz_no_such_token", filter()),
    ];
    cases
        .into_iter()
        .map(|(pattern, f)| {
            let label = format!("{pattern} {f:?}");
            let response = engine
                .search_with_metadata(pattern, f)
                .unwrap_or_else(|e| panic!("{label}: {e:#}"));
            let mut value = serde_json::to_value(&response).unwrap();
            if let Some(obj) = value.as_object_mut() {
                obj.remove("timings");
            }
            (label, value)
        })
        .collect()
}

/// Live paths, live file count and live trigram count of the published snapshot.
fn shape(root: &Path) -> (BTreeSet<String>, usize, usize) {
    let snapshot = IndexSnapshot::open(&root.join(".reflex")).expect("open snapshot");
    let paths = snapshot
        .live_ids()
        .map(|id| {
            snapshot
                .get_file_path(id)
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    (paths, snapshot.live_file_count(), snapshot.trigram_count())
}

/// Assert the index as it stands answers as a fresh build of the same directory.
fn assert_matches_fresh(root: &Path, step: &str) {
    let updated = battery(root);
    let updated_shape = shape(root);
    fs::rename(root.join(".reflex"), root.join(".reflex-updated")).unwrap();
    index(root);
    let fresh = battery(root);
    let fresh_shape = shape(root);
    fs::remove_dir_all(root.join(".reflex")).unwrap();
    fs::rename(root.join(".reflex-updated"), root.join(".reflex")).unwrap();

    assert_eq!(updated_shape, fresh_shape, "{step}: snapshot shape");
    for ((label, u), (_, f)) in updated.iter().zip(&fresh) {
        assert_eq!(u, f, "{step}: {label}");
    }
}

#[test]
fn an_edit_publishes_a_delta_and_hides_the_fixed_names() {
    let temp = workspace();
    let root = temp.path();
    index(root);
    let cache = root.join(".reflex");
    let base = manifest(root);
    assert!(base.base_only());
    assert!(cache.join("content.bin").exists());
    assert!(cache.join("trigrams.bin").exists());

    write(
        root,
        "src/m03.rs",
        "pub fn handler_3() { let shared_token = 1; }\n",
    );
    index(root);
    let m = manifest(root);
    assert_eq!(m.generation, base.generation + 1);
    assert_eq!(
        m.base.content, base.base.content,
        "the base is not rewritten"
    );
    assert_eq!(m.delta.as_ref().map(|d| d.files), Some(1));
    assert_eq!(m.tombstones.len(), 1);
    // A binary that reads only the fixed names would serve the base without the
    // delta: they must be gone while the delta is live.
    assert!(!cache.join("content.bin").exists());
    assert!(!cache.join("trigrams.bin").exists());
    assert_matches_fresh(root, "edit");
}

#[test]
fn every_kind_of_change_matches_a_fresh_build() {
    let temp = workspace();
    let root = temp.path();
    index(root);
    let original = fs::read_to_string(root.join("src/m10.rs")).unwrap();

    write(
        root,
        "src/m10.rs",
        "pub fn handler_10() -> u8 { 0 }\n// realm_id\n",
    );
    index(root);
    assert_matches_fresh(root, "modify");

    write(
        root,
        "src/zz_new.rs",
        "pub fn handler_new(shared_token: u8) {}\n",
    );
    index(root);
    assert_matches_fresh(root, "add");

    fs::remove_file(root.join("py/p05.py")).unwrap();
    index(root);
    assert_matches_fresh(root, "delete");

    fs::rename(root.join("web/w07.ts"), root.join("web/w07_renamed.ts")).unwrap();
    index(root);
    assert_matches_fresh(root, "rename");

    // A file added in one delta and edited in the next: the old delta copy goes.
    write(root, "src/zz_new.rs", "pub fn handler_new2() {}\n");
    index(root);
    assert_matches_fresh(root, "edit a delta file");

    // Removing the only holder of a token leaves no list for it.
    fs::remove_file(root.join("src/zz_new.rs")).unwrap();
    index(root);
    assert_matches_fresh(root, "delete a delta file");

    // Byte-identical revert: the file stays in the delta, answers stay equal.
    write(root, "src/m10.rs", &original);
    index(root);
    assert_matches_fresh(root, "revert");
    assert!(manifest(root).delta.is_some());
}

#[test]
fn the_merge_limit_folds_the_delta_into_a_new_base() {
    let temp = workspace();
    let root = temp.path();
    index(root);
    write(root, "src/m01.rs", "pub fn handler_1() {}\n");
    index(root);
    assert!(manifest(root).delta.is_some());

    // One file allowed; two changed: merge into a new base.
    write(root, "src/m02.rs", "pub fn handler_2() {}\n");
    index_with_limits(root, 1, u64::MAX);
    let m = manifest(root);
    assert!(m.base_only(), "{m:?}");
    let cache = root.join(".reflex");
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let ino = |name: &str| fs::metadata(cache.join(name)).unwrap().ino();
        assert_eq!(ino("content.bin"), ino(&m.base.content));
        assert_eq!(ino("trigrams.bin"), ino(&m.base.trigrams));
    }
    assert_matches_fresh(root, "merge");
}

#[test]
fn only_the_current_and_previous_generations_stay_on_disk() {
    let temp = workspace();
    let root = temp.path();
    index(root);
    for i in 0..5 {
        write(
            root,
            &format!("src/m{i:02}.rs"),
            &format!("pub fn edited_{i}() {{}}\n"),
        );
        index(root);
    }
    // End with a merge, so a whole base and a delta are superseded at once.
    write(root, "src/m09.rs", "pub fn edited_9() {}\n");
    let previous = manifest(root);
    index_with_limits(root, 0, 0);
    let current = manifest(root);
    assert!(current.base_only());

    let cache = root.join(".reflex");
    let keep: BTreeSet<&str> = current
        .files()
        .into_iter()
        .chain(previous.files())
        .collect();
    for name in &keep {
        assert!(cache.join(name).exists(), "{name} is named by a manifest");
    }
    for entry in fs::read_dir(&cache).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        if reflex::snapshot::is_generation_file(&name) {
            assert!(
                keep.contains(name.as_str()),
                "{name} is named by no manifest"
            );
        }
    }
}

#[test]
fn nothing_changed_publishes_nothing() {
    let temp = workspace();
    let root = temp.path();
    index(root);
    write(root, "src/m04.rs", "pub fn handler_4() {}\n");
    index(root);
    let before = manifest(root);
    index(root);
    let after = manifest(root);
    assert_eq!(before.generation, after.generation);
}
