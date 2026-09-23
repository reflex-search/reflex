//! The parallel trigram builder (1.8.1) must produce the same bytes whatever the
//! batch boundaries: `trigrams.bin` and `content.bin` from a single in-memory
//! batch and from many on-disk partials are identical, and so are query results.

use reflex::cache::CacheManager;
use reflex::indexer::{Indexer, plan_batches};
use reflex::models::IndexConfig;
use reflex::{QueryEngine, QueryFilter};
use std::fs;
use std::path::Path;
use tempfile::TempDir;

fn workspace(files: usize) -> TempDir {
    let temp = TempDir::new().unwrap();
    let src = temp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    for i in 0..files {
        let body = format!(
            "// file {i}\npub fn func_{i}(x: u32) -> u32 {{\n    let marker_{i} = x + {i};\n    marker_{i} * 2\n}}\n\n#[cfg(test)]\nmod tests_{i} {{\n    #[test]\n    fn t_{i}() {{ assert_eq!(super::func_{i}(1), {}); }}\n}}\n",
            (1 + i) * 2
        );
        fs::write(src.join(format!("m{i}.rs")), body).unwrap();
    }
    fs::write(
        temp.path().join("README.md"),
        "# demo\n\nfunc_1 is documented here.\n",
    )
    .unwrap();
    fs::write(temp.path().join("empty.txt"), "").unwrap();
    fs::write(temp.path().join("short.txt"), "ab").unwrap();
    temp
}

fn index_with(root: &Path, limits: Option<(usize, u64)>) {
    let cache = CacheManager::new(root);
    let mut indexer = Indexer::new(cache, IndexConfig::default());
    if let Some((files, bytes)) = limits {
        indexer.set_batch_limits(files, bytes);
    }
    indexer.index(root, false).unwrap();
}

fn results(root: &Path, pattern: &str) -> Vec<(String, usize)> {
    let engine = QueryEngine::new(CacheManager::new(root));
    engine
        .search(pattern, QueryFilter::default())
        .unwrap()
        .into_iter()
        .map(|r| (r.path, r.span.start_line))
        .collect()
}

#[test]
fn many_small_batches_produce_identical_index_files() {
    let temp = workspace(300);
    let root = temp.path();

    index_with(root, None);
    let reflex_dir = root.join(".reflex");
    let one_trigrams = fs::read(reflex_dir.join("trigrams.bin")).unwrap();
    let one_content = fs::read(reflex_dir.join("content.bin")).unwrap();
    let one_results: Vec<_> = [
        "func_1",
        "marker_",
        "assert_eq",
        "documented",
        "nothing_here",
    ]
    .iter()
    .map(|p| results(root, p))
    .collect();

    fs::remove_dir_all(&reflex_dir).unwrap();
    index_with(root, Some((50, u64::MAX)));
    assert!(
        !reflex_dir.join("trigram_temp").exists(),
        "partials must be cleaned up"
    );
    assert!(!reflex_dir.join("trigrams.bin.tmp").exists());
    assert_eq!(
        fs::read(reflex_dir.join("trigrams.bin")).unwrap(),
        one_trigrams
    );
    assert_eq!(
        fs::read(reflex_dir.join("content.bin")).unwrap(),
        one_content
    );

    fs::remove_dir_all(&reflex_dir).unwrap();
    index_with(root, Some((usize::MAX, 2_000)));
    assert_eq!(
        fs::read(reflex_dir.join("trigrams.bin")).unwrap(),
        one_trigrams
    );
    assert_eq!(
        fs::read(reflex_dir.join("content.bin")).unwrap(),
        one_content
    );

    let many_results: Vec<_> = [
        "func_1",
        "marker_",
        "assert_eq",
        "documented",
        "nothing_here",
    ]
    .iter()
    .map(|p| results(root, p))
    .collect();
    assert_eq!(many_results, one_results);
    assert!(!one_results[0].is_empty());
    assert!(one_results[4].is_empty());
}

#[test]
fn plan_batches_respects_both_limits() {
    assert_eq!(
        plan_batches(&[], 5, 100),
        Vec::<std::ops::Range<usize>>::new()
    );
    assert_eq!(plan_batches(&[1, 1, 1], 2, 100), vec![0..2, 2..3]);
    assert_eq!(
        plan_batches(&[60, 60, 60], 100, 100),
        vec![0..1, 1..2, 2..3]
    );
    // A single file larger than the byte budget still gets its own batch.
    assert_eq!(plan_batches(&[500, 1, 1], 100, 100), vec![0..1, 1..3]);
    assert_eq!(
        plan_batches(&[10; 10], 4, 35),
        vec![0..3, 3..6, 6..9, 9..10]
    );
}
