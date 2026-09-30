//! Time the library update path (`Indexer::update_paths`) on an indexed tree:
//! a 1-file edit until it is searchable, then an add and a delete.
//!
//!   cargo run --release --example update_paths_timing -- <indexed tree> <file> [runs]
//!
//! `<file>` is relative to the tree. Every edit is reverted (and the revert
//! applied) before the next run. `RUST_LOG=info` prints the phase lines.
//! Used by `benches/incremental/perf.sh`.

use reflex::query::{QueryEngine, QueryFilter};
use reflex::{CacheManager, IndexConfig, Indexer};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// `(VmRSS, VmHWM)` of this process in MiB.
fn memory() -> (f64, f64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let kb = |key: &str| {
        status
            .lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0)
            / 1024.0
    };
    (kb("VmRSS:"), kb("VmHWM:"))
}

fn search_count(root: &Path, pattern: &str) -> usize {
    let engine = QueryEngine::new(CacheManager::new(root));
    let filter = QueryFilter {
        suppress_output: true,
        count_only: true,
        ..Default::default()
    };
    engine
        .search_with_metadata(pattern, filter)
        .map(|r| r.pagination.total.unwrap_or(0))
        .unwrap_or(0)
}

fn update(root: &Path, paths: &[PathBuf]) -> (f64, bool) {
    let indexer = Indexer::new(CacheManager::new(root), IndexConfig::default());
    let t = Instant::now();
    let direct = indexer
        .try_update_paths(root, paths)
        .expect("update_paths")
        .is_some();
    if !direct {
        indexer.index(root, false).expect("index");
    }
    (ms(t), direct)
}

fn main() {
    env_logger::init();
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("tree"));
    let file = PathBuf::from(args.next().expect("file relative to the tree"));
    let runs: usize = args.next().map_or(3, |r| r.parse().expect("runs"));
    let original = std::fs::read(root.join(&file)).expect("read the file");
    let load = || std::fs::read_to_string("/proc/loadavg").unwrap_or_default();

    let (rss, _) = memory();
    println!("memory before the first update: rss {rss:.1} MiB");
    for run in 0..runs {
        let token = format!("zz_update_probe_{run}_{}", std::process::id());
        let mut body = original.clone();
        body.extend_from_slice(format!("// {token}\n").as_bytes());
        std::fs::write(root.join(&file), &body).unwrap();
        let load_at_start = load();
        let t = Instant::now();
        let (update_ms, direct) = update(&root, std::slice::from_ref(&file));
        let found = search_count(&root, &token);
        let (rss, hwm) = memory();
        println!(
            "edit1     update={update_ms:7.1} ms  searchable={:7.1} ms  direct={direct}  found={found}  load={}  rss={rss:.1} MiB  peak={hwm:.1} MiB",
            ms(t),
            load_at_start.split(' ').next().unwrap_or("?")
        );
        assert_eq!(found, 1, "the edit is searchable");

        std::fs::write(root.join(&file), &original).unwrap();
        update(&root, std::slice::from_ref(&file));
    }

    // Add and delete a file next to the edited one (they re-resolve importers).
    let added = file.with_file_name(format!("zz_update_probe_{}.txt", std::process::id()));
    std::fs::write(root.join(&added), "zz_update_probe_added\n").unwrap();
    let (add_ms, direct) = update(&root, std::slice::from_ref(&added));
    println!("add       update={add_ms:7.1} ms  direct={direct}");
    std::fs::remove_file(root.join(&added)).unwrap();
    let (del_ms, direct) = update(&root, std::slice::from_ref(&added));
    println!("delete    update={del_ms:7.1} ms  direct={direct}");
}
