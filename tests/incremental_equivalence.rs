//! Property test of incremental indexing: after any sequence of changes, each
//! followed by `Indexer::index` or `Indexer::update_paths`, the index answers
//! exactly as a fresh build of the same directory does. Compared after every
//! step: a query battery (results, totals, freshness), the dependency and export
//! rows, the dependency analyses, the walk order, and the snapshot's shape.
//!
//! `update_paths` is given the paths a perfect file watcher would report: every
//! path whose bytes or presence changed (a directory rename names the two
//! directories instead).

use reflex::cache::open_meta_db;
use reflex::dependency::DependencyIndex;
use reflex::query::{QueryEngine, QueryFilter};
use reflex::snapshot::IndexSnapshot;
use reflex::{CacheManager, IndexConfig, Indexer};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        (!items.is_empty()).then(|| &items[self.below(items.len())])
    }
}

fn git(root: &Path, args: &[&str]) {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
        .args([
            "-c",
            "commit.gpgsign=false",
            "-c",
            "init.defaultBranch=main",
        ])
        .args(args)
        .output()
        .expect("run git");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn write(root: &Path, rel: &str, body: &str) {
    let p = root.join(rel);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, body).unwrap();
}

/// A multi-language git tree whose files import each other.
fn seed_tree(root: &Path) {
    write(root, "Cargo.toml", "[package]\nname = \"demo\"\n");
    let mods: String = (0..6).map(|i| format!("pub mod m{i};\n")).collect();
    write(root, "src/lib.rs", &mods);
    for i in 0..6 {
        let j = (i + 1) % 6;
        write(
            root,
            &format!("src/m{i}.rs"),
            &format!(
                "use crate::m{j}::func_{j};\npub fn func_{i}() {{ func_{j}(); }} // shared_token {i}\n"
            ),
        );
    }
    write(root, "py/pyproject.toml", "[project]\nname = \"pkg\"\n");
    write(root, "py/pkg/__init__.py", "");
    for i in 0..5 {
        let j = (i + 2) % 5;
        write(
            root,
            &format!("py/pkg/a{i}.py"),
            &format!(
                "from pkg.a{j} import pyfn_{j}\n\ndef pyfn_{i}(shared_token):\n    return pyfn_{j}(shared_token)\n"
            ),
        );
    }
    write(
        root,
        "web/tsconfig.json",
        r#"{"compilerOptions":{"baseUrl":".","paths":{"@lib/*":["lib/*"]}}}"#,
    );
    for i in 0..5 {
        write(
            root,
            &format!("web/lib/u{i}.ts"),
            &format!("export const u{i} = {i}; // shared_token\n"),
        );
        let k = (i + 1) % 5;
        write(
            root,
            &format!("web/app{i}.ts"),
            &format!(
                "import {{ u{i} }} from '@lib/u{i}';\nimport {{ x{k} }} from './app{k}';\nexport const x{i} = u{i} + x{k};\n"
            ),
        );
    }
    write(root, "go/go.mod", "module example.com/demo\n");
    for i in 0..4 {
        let j = (i + 1) % 4;
        write(
            root,
            &format!("go/p{i}/p.go"),
            &format!(
                "package p{i}\n\nimport \"example.com/demo/p{j}\"\n\nfunc F() {{ p{j}.F() }} // shared_token\n"
            ),
        );
    }
    write(
        root,
        "docs/guide.md",
        "# Guide\n\nshared_token appears in f1 and g2.\n",
    );
    write(root, "Cargo.lock", "[[package]]\nname = \"shared_token\"\n");
    write(root, ".gitignore", "ignored/\n");
    git(root, &["init", "-q"]);
    fs::write(root.join(".git/info/exclude"), ".reflex*\n").unwrap();
    git(root, &["add", "-A"]);
    git(root, &["commit", "-q", "-m", "seed"]);
}

/// Every file under `root` (not `.git`, not the caches) with its bytes.
fn tree_state(root: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().to_string_lossy().into_owned();
            if dir == root && (name == ".git" || name.starts_with(".reflex")) {
                continue;
            }
            let path = entry.path();
            if entry.file_type().unwrap().is_dir() {
                walk(root, &path, out);
            } else {
                let rel = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                out.insert(rel, fs::read(&path).unwrap());
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

/// Paths a perfect watcher reports between two tree states.
fn changed_paths(
    before: &BTreeMap<String, Vec<u8>>,
    after: &BTreeMap<String, Vec<u8>>,
) -> Vec<PathBuf> {
    let keys: BTreeSet<&String> = before.keys().chain(after.keys()).collect();
    keys.into_iter()
        .filter(|k| before.get(*k) != after.get(*k))
        .map(PathBuf::from)
        .collect()
}

fn code_files(root: &Path) -> Vec<String> {
    tree_state(root)
        .into_keys()
        .filter(|p| {
            [".rs", ".py", ".ts", ".go", ".md"]
                .iter()
                .any(|e| p.ends_with(e))
                && !p.starts_with("ignored/")
        })
        .collect()
}

/// Apply one random change; returns the paths to name when they are not simply
/// the changed files (a directory rename names the directories).
fn mutate(root: &Path, rng: &mut Rng, step: usize) -> Option<Vec<PathBuf>> {
    let files = code_files(root);
    match rng.below(14) {
        0..=3 => {
            if let Some(f) = rng.pick(&files) {
                let mut body = fs::read_to_string(root.join(f)).unwrap_or_default();
                body.push_str(&format!("// edit {step} token_{}\n", rng.below(5)));
                fs::write(root.join(f), body).unwrap();
            }
        }
        4 => {
            let (rel, body) = match rng.below(4) {
                0 => (
                    format!("src/n{step}.rs"),
                    format!(
                        "use crate::m1::func_1;\npub fn func_n{step}() {{ func_1(); }} // shared_token\n"
                    ),
                ),
                1 => (
                    format!("py/pkg/n{step}.py"),
                    "from pkg.a0 import pyfn_0\n\ndef pyfn_new(): return pyfn_0(1)\n".to_string(),
                ),
                2 => (
                    format!("web/lib/n{step}.ts"),
                    format!("import {{ u1 }} from './u1';\nexport const n{step} = u1;\n"),
                ),
                _ => (
                    format!("go/q{step}/q.go"),
                    "package q\n\nimport \"example.com/demo/p0\"\n\nfunc G() { p0.F() }\n"
                        .to_string(),
                ),
            };
            write(root, &rel, &body);
        }
        5 => {
            if let Some(f) = rng.pick(&files) {
                fs::remove_file(root.join(f)).unwrap();
            }
        }
        6 => {
            if let Some(f) = rng.pick(&files) {
                let (stem, ext) = f.rsplit_once('.').unwrap();
                fs::rename(root.join(f), root.join(format!("{stem}_r{step}.{ext}"))).unwrap();
            }
        }
        7 => {
            // Directory rename: the two directories are named, not the files.
            let dirs = ["web/lib", "py/pkg", "go/p1", "go/p2", "docs"];
            let old = dirs[rng.below(dirs.len())];
            let old = if root.join(old).is_dir() {
                old.to_string()
            } else {
                return None;
            };
            let new = format!("{old}_d{step}");
            fs::rename(root.join(&old), root.join(&new)).unwrap();
            return Some(vec![PathBuf::from(old), PathBuf::from(new)]);
        }
        8 => {
            // Atomic save: write a temporary, rename it over the file.
            if let Some(f) = rng.pick(&files) {
                let mut body = fs::read_to_string(root.join(f)).unwrap_or_default();
                body.insert_str(0, "// saved\n");
                let tmp = root.join(format!("{f}.tmp{step}"));
                fs::write(&tmp, body).unwrap();
                fs::rename(&tmp, root.join(f)).unwrap();
            }
        }
        9 => match rng.below(3) {
            0 => write(
                root,
                "go/go.mod",
                &format!("module example.com/demo{}\n", rng.below(2)),
            ),
            1 => {
                let target = ["lib/*", "other/*"][rng.below(2)];
                write(
                    root,
                    "web/tsconfig.json",
                    &format!(
                        r#"{{"compilerOptions":{{"baseUrl":".","paths":{{"@lib/*":["{target}"]}}}}}}"#
                    ),
                );
            }
            _ => write(
                root,
                "Cargo.toml",
                &format!("[package]\nname = \"demo{}\"\n", rng.below(2)),
            ),
        },
        10 => {
            let patterns = [
                "ignored/\n",
                "ignored/\ndocs/\n",
                "ignored/\n*.md\n",
                "ignored/\nweb/lib/u1.ts\n",
            ];
            write(root, ".gitignore", patterns[rng.below(patterns.len())]);
        }
        11 => {
            // Files no walk indexes: ignored, hidden, binary.
            match rng.below(3) {
                0 => write(
                    root,
                    &format!("ignored/x{step}.rs"),
                    "fn ignored_token() {}\n",
                ),
                1 => write(
                    root,
                    &format!(".hidden/h{step}.rs"),
                    "fn hidden_token() {}\n",
                ),
                _ => {
                    fs::create_dir_all(root.join("docs")).unwrap();
                    fs::write(root.join(format!("docs/blob{step}.dat")), b"bin\0ary").unwrap()
                }
            }
        }
        12 => {
            // A text file turns binary.
            if let Some(f) = rng.pick(&files) {
                fs::write(root.join(f), b"now\0binary").unwrap();
            }
        }
        _ => {
            // Commit everything, or switch to a new branch with an edit and back.
            git(root, &["add", "-A"]);
            git(
                root,
                &[
                    "commit",
                    "-q",
                    "--allow-empty",
                    "-m",
                    &format!("step {step}"),
                ],
            );
            if rng.below(2) == 0 {
                let current = String::from_utf8(
                    Command::new("git")
                        .arg("-C")
                        .arg(root)
                        .args(["rev-parse", "--abbrev-ref", "HEAD"])
                        .output()
                        .unwrap()
                        .stdout,
                )
                .unwrap()
                .trim()
                .to_string();
                git(root, &["checkout", "-q", "-b", &format!("b{step}")]);
                if rng.below(2) == 0 {
                    // Stay on the new branch after one more commit.
                    if let Some(f) = rng.pick(&code_files(root)) {
                        let mut body = fs::read_to_string(root.join(f)).unwrap_or_default();
                        body.push_str("// on a branch\n");
                        fs::write(root.join(f), body).unwrap();
                    }
                    git(
                        root,
                        &["commit", "-q", "-a", "--allow-empty", "-m", "branch"],
                    );
                } else {
                    git(root, &["checkout", "-q", &current]);
                }
            }
        }
    }
    None
}

/// Limits of a run: the merge limit, and the recent segment's limit before it
/// folds into the delta (`None`: the defaults under an unlimited merge).
#[derive(Clone, Copy)]
struct Limits {
    merge: Option<(usize, u64)>,
    recent: Option<(usize, u64)>,
}

fn indexer(root: &Path, limits: Limits) -> Indexer {
    let mut indexer = Indexer::new(CacheManager::new(root), IndexConfig::default());
    let (files, bytes) = limits.merge.unwrap_or((usize::MAX, u64::MAX));
    indexer.set_merge_limits(files, bytes);
    if let Some((files, bytes)) = limits.recent {
        indexer.set_recent_limits(files, bytes);
    }
    indexer
}

fn filter() -> QueryFilter {
    QueryFilter {
        suppress_output: true,
        ..Default::default()
    }
}

/// Drop the fields that record when, not what: timings and timestamps.
fn scrub(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.remove("timings");
            map.remove("indexed_at");
            map.values_mut().for_each(scrub);
        }
        Value::Array(items) => items.iter_mut().for_each(scrub),
        _ => {}
    }
}

fn battery(root: &Path) -> Vec<(String, Value)> {
    let engine = QueryEngine::new(CacheManager::new(root));
    let cases: Vec<(&str, QueryFilter)> = vec![
        ("shared_token", filter()),
        (
            "shared_token",
            QueryFilter {
                use_contains: true,
                ..filter()
            },
        ),
        (
            "SHARED_TOKEN",
            QueryFilter {
                ignore_case: true,
                ..filter()
            },
        ),
        (
            "token",
            QueryFilter {
                ignore_case: true,
                use_contains: true,
                ..filter()
            },
        ),
        (
            r"f\d+\(\)",
            QueryFilter {
                use_regex: true,
                ..filter()
            },
        ),
        (
            r"\w+_\d",
            QueryFilter {
                use_regex: true,
                ..filter()
            },
        ),
        (
            "func_1",
            QueryFilter {
                symbols_mode: true,
                ..filter()
            },
        ),
        (
            "pyfn",
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
                limit: Some(3),
                offset: Some(2),
                ..filter()
            },
        ),
        (
            "token",
            QueryFilter {
                include_locks: true,
                use_contains: true,
                ..filter()
            },
        ),
        (
            "import",
            QueryFilter {
                glob_patterns: vec!["web/**".into()],
                ..filter()
            },
        ),
        ("ignored_token", filter()),
        ("hidden_token", filter()),
    ];
    cases
        .into_iter()
        .map(|(pattern, f)| {
            let label = format!("{pattern} {f:?}");
            let response = engine
                .search_with_metadata(pattern, f)
                .unwrap_or_else(|e| panic!("{label}: {e:#}"));
            let mut value = serde_json::to_value(&response).unwrap();
            scrub(&mut value);
            (label, value)
        })
        .collect()
}

/// Dependency and export rows joined to paths, the walk order, and the analyses.
fn structure(root: &Path) -> Vec<String> {
    let conn = open_meta_db(root.join(".reflex").join("meta.db")).unwrap();
    let rows = |sql: &str| -> Vec<String> {
        let mut stmt = conn.prepare(sql).unwrap();
        let n = stmt.column_count();
        stmt.query_map([], |r| {
            Ok((0..n)
                .map(|i| match r.get_ref(i).unwrap() {
                    rusqlite::types::ValueRef::Null => "-".to_string(),
                    rusqlite::types::ValueRef::Integer(v) => v.to_string(),
                    rusqlite::types::ValueRef::Text(t) => String::from_utf8_lossy(t).into_owned(),
                    other => format!("{other:?}"),
                })
                .collect::<Vec<_>>()
                .join(" | "))
        })
        .unwrap()
        .map(|r| r.unwrap())
        .collect()
    };
    let mut out = vec!["# walk order".to_string()];
    out.extend(rows(
        "SELECT path, language, line_count, hash FROM files ORDER BY walk_seq",
    ));
    out.push("# dependencies".into());
    out.extend(rows(
        "SELECT f.path, fd.imported_path, r.path, fd.import_type, fd.line_number, fd.imported_symbols
         FROM file_dependencies fd JOIN files f ON f.id = fd.file_id
         LEFT JOIN files r ON r.id = fd.resolved_file_id
         ORDER BY f.path, fd.line_number, fd.imported_path, fd.import_type",
    ));
    out.push("# exports".into());
    out.extend(rows(
        "SELECT f.path, e.exported_symbol, e.source_path, r.path, e.line_number
         FROM file_exports e JOIN files f ON f.id = e.file_id
         LEFT JOIN files r ON r.id = e.resolved_source_id
         ORDER BY f.path, e.line_number, e.source_path",
    ));

    let deps = DependencyIndex::new(CacheManager::new(root));
    let ids: Vec<(i64, String)> = {
        let mut stmt = conn
            .prepare("SELECT id, path FROM files ORDER BY walk_seq")
            .unwrap();
        stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    };
    let path_of: BTreeMap<i64, String> = ids.iter().cloned().collect();
    let names = |v: &[i64]| -> String {
        v.iter()
            .map(|id| path_of[id].as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    out.push("# dependents".into());
    for (id, path) in &ids {
        out.push(format!(
            "{path} <- {}",
            names(&deps.get_dependents(*id).unwrap())
        ));
    }
    out.push("# cycles".into());
    for cycle in deps.detect_circular_dependencies().unwrap() {
        out.push(names(&cycle));
    }
    out.push("# hotspots".into());
    for (id, n) in deps.find_hotspots(None, 1).unwrap() {
        out.push(format!("{} {n}", path_of[&id]));
    }
    out.push("# unused".into());
    out.push(names(&deps.find_unused_files().unwrap()));
    out.push("# islands".into());
    for island in deps.find_islands().unwrap() {
        out.push(names(&island));
    }
    out
}

fn shape(root: &Path) -> (Vec<String>, usize, usize) {
    let snapshot = IndexSnapshot::open(&root.join(".reflex")).expect("open snapshot");
    let mut paths: Vec<String> = snapshot
        .live_ids()
        .map(|id| {
            snapshot
                .get_file_path(id)
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    paths.sort();
    (paths, snapshot.live_file_count(), snapshot.trigram_count())
}

fn assert_matches_fresh(root: &Path, context: &str) {
    let updated = (battery(root), structure(root), shape(root));
    let aside = root.join(".reflex-updated");
    reflex::query::invalidate_caches(root); // Windows: release the shared handle first
    fs::rename(root.join(".reflex"), &aside).unwrap();
    indexer(
        root,
        Limits {
            merge: None,
            recent: None,
        },
    )
    .index(root, false)
    .expect("fresh index");
    let fresh = (battery(root), structure(root), shape(root));
    reflex::query::invalidate_caches(root); // Windows: release the shared handle first
    fs::remove_dir_all(root.join(".reflex")).unwrap();
    fs::rename(&aside, root.join(".reflex")).unwrap();

    assert_eq!(updated.2, fresh.2, "{context}: snapshot shape");
    if updated.1 != fresh.1 {
        let diff: Vec<String> = updated
            .1
            .iter()
            .filter(|l| !fresh.1.contains(l))
            .map(|l| format!("- {l}"))
            .chain(
                fresh
                    .1
                    .iter()
                    .filter(|l| !updated.1.contains(l))
                    .map(|l| format!("+ {l}")),
            )
            .collect();
        panic!(
            "{context}: structure differs (- updated, + fresh):\n{}",
            diff.join("\n")
        );
    }
    for ((label, u), (_, f)) in updated.0.iter().zip(&fresh.0) {
        assert_eq!(u, f, "{context}: {label}");
    }
}

/// Returns (update_paths steps applied directly, update_paths steps that needed a
/// full run).
fn run_seed(seed: u64, steps: usize) -> (usize, usize) {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    seed_tree(root);
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5);
    // Most seeds keep the delta; some merge often, some fold the recent segment
    // into the delta often.
    let limits = Limits {
        merge: (seed % 4 == 3).then_some((2usize, u64::MAX)),
        recent: (seed % 4 == 1).then_some((2usize, u64::MAX)),
    };
    indexer(root, limits).index(root, false).unwrap();
    let mut log: Vec<String> = Vec::new();
    let (mut direct, mut full) = (0, 0);
    for step in 0..steps {
        let before = tree_state(root);
        let named = mutate(root, &mut rng, step);
        let after = tree_state(root);
        let named = named.unwrap_or_else(|| changed_paths(&before, &after));
        let use_update = rng.below(3) != 0;
        log.push(format!(
            "step {step}: {} {:?}",
            if use_update { "update_paths" } else { "index" },
            named
        ));
        let ix = indexer(root, limits);
        if use_update {
            match ix.try_update_paths(root, &named).expect("update_paths") {
                Some(_) => direct += 1,
                None => {
                    full += 1;
                    log.last_mut().unwrap().push_str(" (full run)");
                    ix.index(root, false).expect("index");
                }
            }
        } else {
            ix.index(root, false).expect("index");
        }
        assert_matches_fresh(root, &format!("seed {seed}\n{}", log.join("\n")));
    }
    (direct, full)
}

#[test]
fn updates_match_a_fresh_build() {
    let (mut direct, mut full) = (0, 0);
    for seed in 0..4 {
        let (d, f) = run_seed(seed, 18);
        direct += d;
        full += f;
    }
    eprintln!("update_paths: {direct} direct, {full} full runs");
    assert!(direct > full, "most updates must not need a full run");
}

/// The long version: `cargo test --release --test incremental_equivalence -- --ignored`.
#[test]
#[ignore]
fn updates_match_a_fresh_build_many_seeds() {
    let (mut direct, mut full) = (0, 0);
    for seed in 100..140 {
        let (d, f) = run_seed(seed, 30);
        direct += d;
        full += f;
    }
    eprintln!("update_paths: {direct} direct, {full} full runs");
}
