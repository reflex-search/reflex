//! Stage 2 measurements on an indexed git checkout whose cache is a fresh base:
//!
//! 1. query cost (candidate lookup + verification + grouping, not the freshness
//!    check) on the base alone, then with a delta of `n` edited files and as many
//!    tombstones: the tombstone filter and the delta's lists;
//! 2. a 1-file `update_paths` while that delta is live: the whole delta is
//!    rebuilt on every update.
//!
//!   rm -rf <tree>/.reflex && rfx index   (in <tree>)
//!   cargo run --release --example delta_threshold_timing -- <tree> <n> [rounds]
//!
//! The `n` files are the first tracked `.go` / `.rs` files in sorted order; every
//! edit is restored with `git checkout` at the end.

use reflex::query::{QueryEngine, QueryFilter};
use reflex::snapshot::read_manifest;
use reflex::{CacheManager, IndexConfig, Indexer};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

fn shapes() -> Vec<(&'static str, QueryFilter)> {
    let base = QueryFilter {
        suppress_output: true,
        collect_timings: true,
        ..Default::default()
    };
    vec![
        ("zz_absent_token_q", base.clone()),
        ("NewController", base.clone()),
        (
            "return",
            QueryFilter {
                limit: Some(100),
                ..base.clone()
            },
        ),
        (
            r"func (Get|Set)\w+",
            QueryFilter {
                use_regex: true,
                ..base.clone()
            },
        ),
        (
            "podspec",
            QueryFilter {
                ignore_case: true,
                ..base.clone()
            },
        ),
        (
            "Informer",
            QueryFilter {
                use_contains: true,
                ..base.clone()
            },
        ),
        (
            "NewController",
            QueryFilter {
                symbols_mode: true,
                ..base
            },
        ),
    ]
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Median index time (candidates + verify + group, ms) of each shape; with
/// `verbose`, each phase's median too.
fn measure(root: &Path, rounds: usize, verbose: bool) -> Vec<f64> {
    let engine = QueryEngine::new(CacheManager::new(root));
    shapes()
        .into_iter()
        .map(|(pattern, filter)| {
            let runs: Vec<(f64, f64, f64)> = (0..rounds + 1)
                .map(|_| {
                    let r = engine
                        .search_with_metadata(pattern, filter.clone())
                        .expect("query");
                    let t = r.timings.expect("timings");
                    (
                        t.candidates_us as f64 / 1000.0,
                        t.verify_us as f64 / 1000.0,
                        t.group_us as f64 / 1000.0,
                    )
                })
                .skip(1) // warm-up
                .collect();
            if verbose {
                println!(
                    "  {pattern:<28} candidates {:.2}  verify {:.2}  group {:.2}",
                    median(runs.iter().map(|r| r.0).collect()),
                    median(runs.iter().map(|r| r.1).collect()),
                    median(runs.iter().map(|r| r.2).collect())
                );
            }
            median(runs.iter().map(|r| r.0 + r.1 + r.2).collect())
        })
        .collect()
}

fn append(root: &Path, rel: &Path, line: &str) {
    let path = root.join(rel);
    let mut body = std::fs::read(&path).unwrap();
    body.extend_from_slice(line.as_bytes());
    std::fs::write(path, body).unwrap();
}

fn main() {
    env_logger::init();
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("tree"));
    let n: usize = args.next().expect("n").parse().expect("n");
    let rounds: usize = args.next().map_or(7, |r| r.parse().expect("rounds"));
    let load = || {
        std::fs::read_to_string("/proc/loadavg")
            .unwrap_or_default()
            .split(' ')
            .next()
            .unwrap_or("?")
            .to_string()
    };
    let manifest = read_manifest(&root.join(".reflex"))
        .unwrap()
        .expect("a manifest");
    assert!(manifest.base_only(), "index the tree afresh first");

    let listed = Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["ls-files", "*.go", "*.rs"])
        .output()
        .expect("git ls-files");
    let mut files: Vec<PathBuf> = String::from_utf8_lossy(&listed.stdout)
        .lines()
        .map(PathBuf::from)
        .collect();
    files.sort();
    assert!(files.len() > n + 1, "the tree has {} files", files.len());
    let edited = &files[..n];
    let extra = &files[n];

    println!("# load at start {}", load());
    println!("phases on the base:");
    let base = measure(&root, rounds, true);

    for rel in edited {
        append(&root, rel, "// delta probe\n");
    }
    let indexer = Indexer::new(CacheManager::new(&root), IndexConfig::default());
    let t = Instant::now();
    let direct = indexer
        .try_update_paths(&root, edited)
        .expect("update")
        .is_some();
    println!(
        "delta of {n} files: published in {:.0} ms (direct={direct})",
        t.elapsed().as_secs_f64() * 1000.0
    );
    let m = read_manifest(&root.join(".reflex")).unwrap().unwrap();
    println!(
        "  delta files {}, tombstones {}",
        m.delta.as_ref().map_or(0, |d| d.files),
        m.tombstones.len()
    );
    println!("phases with the delta:");
    let with_delta = measure(&root, rounds, true);

    // The same edited tree as a fresh base (moved aside and back): separates what
    // the delta costs from what the edit itself changed.
    let fresh_same_tree = if std::env::args().any(|a| a == "--fresh-too") {
        let cache = root.join(".reflex");
        let aside = root.join(".reflex-delta");
        std::fs::rename(&cache, &aside).unwrap();
        Indexer::new(CacheManager::new(&root), IndexConfig::default())
            .index(&root, false)
            .expect("fresh index");
        println!("phases on a fresh base of the edited tree:");
        let fresh = measure(&root, rounds, true);
        std::fs::remove_dir_all(&cache).unwrap();
        std::fs::rename(&aside, &cache).unwrap();
        reflex::query::invalidate_caches(&root);
        Some(fresh)
    } else {
        None
    };

    // A 1-file update while the delta is live (the delta is rebuilt whole).
    let mut updates = Vec::new();
    for i in 0..rounds {
        append(&root, extra, &format!("// extra {i}\n"));
        let t = Instant::now();
        indexer
            .update_paths(&root, std::slice::from_ref(extra))
            .expect("update");
        updates.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    println!(
        "1-file update with the delta live: median {:.1} ms (load {})",
        median(updates),
        load()
    );

    println!("shape                         base ms   delta ms   change");
    for (((pattern, filter), b), d) in shapes().iter().zip(&base).zip(&with_delta) {
        let label = format!(
            "{}{}",
            pattern,
            if filter.symbols_mode {
                " (symbols)"
            } else if filter.ignore_case {
                " (-i)"
            } else if filter.use_regex {
                " (regex)"
            } else if filter.use_contains {
                " (contains)"
            } else {
                ""
            }
        );
        println!(
            "{label:<30} {b:>8.2} {d:>10.2} {:>+8.1}%",
            (d - b) / b * 100.0
        );
    }
    if let Some(fresh) = &fresh_same_tree {
        println!("shape                    fresh edited tree ms   delta ms   change");
        for (((pattern, _), f), d) in shapes().iter().zip(fresh).zip(&with_delta) {
            println!(
                "{pattern:<30} {f:>8.2} {d:>10.2} {:>+8.1}%",
                (d - f) / f * 100.0
            );
        }
    }
    let sum_b: f64 = base.iter().sum();
    let sum_d: f64 = with_delta.iter().sum();
    println!(
        "{:<30} {sum_b:>8.2} {sum_d:>10.2} {:>+8.1}%",
        "all shapes",
        (sum_d - sum_b) / sum_b * 100.0
    );
    println!("# load at end {}", load());

    // Restore the tree and the index.
    let mut restore: Vec<PathBuf> = edited.to_vec();
    restore.push(extra.clone());
    let status = Command::new("git")
        .arg("-C")
        .arg(&root)
        .arg("checkout")
        .arg("--")
        .args(&restore)
        .status()
        .unwrap();
    assert!(status.success());
    indexer.update_paths(&root, &restore).expect("restore");
}
