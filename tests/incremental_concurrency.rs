//! Readers during publishes. A writer moves a tree through several states, each
//! published by `Indexer::index` or `Indexer::update_paths` (some as a merge into
//! a new base), while reader threads and a reader child process query in a loop.
//! Every answer must be the answer of one of the states (a snapshot, never a
//! mix), and no query may fail.

use reflex::query::{QueryEngine, QueryFilter};
use reflex::{CacheManager, IndexConfig, Indexer};
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tempfile::TempDir;

/// Test-harness variable of the reader child (read by this test binary only).
const READER_ROOT: &str = "REFLEX_CONCURRENCY_TEST_ROOT";
const STOP_FILE: &str = "stop-readers";

const TOKENS: [&str; 5] = [
    "alpha_token",
    "beta_token",
    "gamma_token",
    "moved_token",
    "common_token",
];

fn write(root: &Path, rel: &str, body: &str) {
    let p = root.join(rel);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, body).unwrap();
}

fn seed(root: &Path) {
    for i in 0..40 {
        write(
            root,
            &format!("src/f{i:02}.rs"),
            &format!("pub fn f{i}() {{ common_token(); }}\n"),
        );
    }
    write(root, "src/a.rs", "fn alpha_token() {}\n");
    write(root, "src/b.rs", "fn beta_token() { moved_token(); }\n");
}

/// Change `step` (1-based); returns the paths it touches.
fn apply(root: &Path, step: usize) -> Vec<PathBuf> {
    let p = |s: &str| PathBuf::from(s);
    match step {
        1 => {
            write(root, "src/b.rs", "fn beta_token() {}\n");
            write(root, "src/c.rs", "fn gamma_token() { moved_token(); }\n");
            vec![p("src/b.rs"), p("src/c.rs")]
        }
        2 => {
            fs::remove_file(root.join("src/a.rs")).unwrap();
            vec![p("src/a.rs")]
        }
        3 => {
            fs::rename(root.join("src/c.rs"), root.join("src/c2.rs")).unwrap();
            vec![p("src/c.rs"), p("src/c2.rs")]
        }
        4 => {
            write(root, "src/a.rs", "fn alpha_token() { moved_token(); }\n");
            write(root, "src/f07.rs", "pub fn f7() {}\n");
            vec![p("src/a.rs"), p("src/f07.rs")]
        }
        _ => {
            write(
                root,
                "src/f07.rs",
                "pub fn f7() { common_token(); gamma_token(); }\n",
            );
            fs::remove_file(root.join("src/b.rs")).unwrap();
            vec![p("src/f07.rs"), p("src/b.rs")]
        }
    }
}
const STEPS: usize = 5;

fn answer(engine: &QueryEngine, token: &str) -> anyhow::Result<Vec<String>> {
    let filter = QueryFilter {
        suppress_output: true,
        ..Default::default()
    };
    let response = engine.search_with_metadata(token, filter)?;
    let mut paths: Vec<String> = response.results.iter().map(|g| g.path.clone()).collect();
    paths.sort();
    Ok(paths)
}

/// For each token, its answer in every state of the tree.
fn expected() -> HashMap<String, BTreeSet<Vec<String>>> {
    let mut out: HashMap<String, BTreeSet<Vec<String>>> = HashMap::new();
    for state in 0..=STEPS {
        let temp = TempDir::new().unwrap();
        let root = temp.path();
        seed(root);
        for step in 1..=state {
            apply(root, step);
        }
        Indexer::new(CacheManager::new(root), IndexConfig::default())
            .index(root, false)
            .unwrap();
        let engine = QueryEngine::new(CacheManager::new(root));
        for token in TOKENS {
            out.entry(token.to_string())
                .or_default()
                .insert(answer(&engine, token).unwrap());
        }
    }
    out
}

/// The reader child: query until the stop file appears; one line per answer.
#[test]
fn reader_child() {
    let Ok(root) = std::env::var(READER_ROOT) else {
        return; // not a child run
    };
    let root = PathBuf::from(root);
    let engine = QueryEngine::new(CacheManager::new(&root));
    while !root.join(".reflex-test").join(STOP_FILE).exists() {
        for token in TOKENS {
            match answer(&engine, token) {
                Ok(paths) => println!("{token}\t{}", paths.join(",")),
                Err(e) => println!("ERROR\t{token}: {e:#}"),
            }
        }
    }
}

#[test]
fn readers_see_whole_snapshots_while_updates_publish() {
    let expected = expected();
    let temp = TempDir::new().unwrap();
    let root = temp.path().to_path_buf();
    seed(&root);
    Indexer::new(CacheManager::new(&root), IndexConfig::default())
        .index(&root, false)
        .unwrap();
    fs::create_dir_all(root.join(".reflex-test")).unwrap();

    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["reader_child", "--exact", "--nocapture", "--test-threads=1"])
        .env(READER_ROOT, &root)
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn the reader child");
    let child_out = child.stdout.take().unwrap();
    let child_lines = std::thread::spawn(move || {
        BufReader::new(child_out)
            .lines()
            .map_while(Result::ok)
            .collect::<Vec<String>>()
    });

    let done = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..3)
        .map(|_| {
            let root = root.clone();
            let done = Arc::clone(&done);
            let expected = expected.clone();
            std::thread::spawn(move || {
                let mut checked = 0usize;
                while !done.load(Ordering::Relaxed) {
                    // A fresh engine each round, as a new request would have.
                    let engine = QueryEngine::new(CacheManager::new(&root));
                    for token in TOKENS {
                        let got =
                            answer(&engine, token).unwrap_or_else(|e| panic!("{token}: {e:#}"));
                        assert!(
                            expected[token].contains(&got),
                            "{token}: {got:?} is no state's answer"
                        );
                        checked += 1;
                    }
                }
                checked
            })
        })
        .collect();

    // The writer: every state twice over (the second pass reverts to the seed
    // through the same steps in a fresh tree state), alternating the two entry
    // points; every third publish merges into a new base.
    for round in 0..3 {
        if round > 0 {
            // Back to the seed: remove everything the steps added, then reseed.
            for rel in ["src/a.rs", "src/b.rs", "src/c.rs", "src/c2.rs"] {
                let _ = fs::remove_file(root.join(rel));
            }
            seed(&root);
            Indexer::new(CacheManager::new(&root), IndexConfig::default())
                .index(&root, false)
                .unwrap();
        }
        for step in 1..=STEPS {
            let named = apply(&root, step);
            let mut ix = Indexer::new(CacheManager::new(&root), IndexConfig::default());
            if (round * STEPS + step).is_multiple_of(3) {
                ix.set_merge_limits(0, 0);
            } else {
                ix.set_merge_limits(usize::MAX, u64::MAX);
            }
            if step.is_multiple_of(2) {
                ix.update_paths(&root, &named).unwrap();
            } else {
                ix.index(&root, false).unwrap();
            }
        }
    }

    done.store(true, Ordering::Relaxed);
    let checked: usize = readers.into_iter().map(|r| r.join().unwrap()).sum();
    fs::write(root.join(".reflex-test").join(STOP_FILE), "").unwrap();
    assert!(child.wait().unwrap().success(), "reader child failed");
    let lines = child_lines.join().unwrap();

    let mut child_checked = 0usize;
    for line in &lines {
        let Some((token, paths)) = line.split_once('\t') else {
            continue; // test harness output
        };
        assert_ne!(token, "ERROR", "reader child: {paths}");
        if !TOKENS.contains(&token) {
            continue;
        }
        let got: Vec<String> = if paths.is_empty() {
            Vec::new()
        } else {
            paths.split(',').map(String::from).collect()
        };
        assert!(
            expected[token].contains(&got),
            "reader child: {token}: {got:?} is no state's answer"
        );
        child_checked += 1;
    }
    assert!(
        checked > 0 && child_checked > 0,
        "{checked} / {child_checked}"
    );
    eprintln!("checked {checked} answers in threads, {child_checked} in the child");
}
