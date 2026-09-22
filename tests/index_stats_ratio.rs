//! `IndexStats::corpus_bytes` / `trigram_index_bytes` — the inputs to the
//! `Index/corpus ratio` line printed by `rfx index`.
//!
//! `corpus_bytes` is read from the content.bin header (`index_offset - 32`),
//! `trigram_index_bytes` is the on-disk size of trigrams.bin. Both are
//! omitted from JSON when zero so older consumers see no new keys for an
//! uninitialised cache.

use reflex::{CacheManager, IndexConfig, IndexStats, Indexer};
use std::fs;
use tempfile::TempDir;

fn project_with_files() -> TempDir {
    let dir = TempDir::new().unwrap();
    for i in 0..5 {
        let body: String = (0..200)
            .map(|n| format!("pub fn func_{i}_{n}(value: u64) -> u64 {{ value + {n} }}\n"))
            .collect();
        fs::write(dir.path().join(format!("file_{i}.rs")), body).unwrap();
    }
    dir
}

#[test]
fn indexing_reports_corpus_and_trigram_bytes() {
    let project = project_with_files();
    let source_bytes: u64 = (0..5)
        .map(|i| {
            fs::metadata(project.path().join(format!("file_{i}.rs")))
                .unwrap()
                .len()
        })
        .sum();

    let cache = CacheManager::new(project.path());
    let stats = Indexer::new(cache, IndexConfig::default())
        .index(project.path(), false)
        .expect("index");

    assert_eq!(stats.total_files, 5);
    assert_eq!(
        stats.corpus_bytes, source_bytes,
        "corpus_bytes must equal the raw bytes of the indexed files"
    );

    let trigrams_len = fs::metadata(project.path().join(".reflex/trigrams.bin"))
        .unwrap()
        .len();
    assert_eq!(stats.trigram_index_bytes, trigrams_len);
    assert!(stats.trigram_index_bytes > 0);
    assert!(
        stats.index_size_bytes >= stats.trigram_index_bytes + stats.corpus_bytes,
        "cache size covers at least trigrams.bin and content.bin"
    );

    // Highly repetitive source: the per-line, per-file V4 format must stay
    // well under 2x the corpus even on a tiny project where the directory
    // and paths sections are not amortised.
    let ratio = stats.trigram_index_bytes as f64 / stats.corpus_bytes as f64;
    assert!(ratio < 2.0, "index/corpus ratio {ratio:.2} too high");

    // A fresh CacheManager reads the same numbers back from disk
    let again = CacheManager::new(project.path()).stats().unwrap();
    assert_eq!(again.corpus_bytes, stats.corpus_bytes);
    assert_eq!(again.trigram_index_bytes, stats.trigram_index_bytes);
}

#[test]
fn stats_on_uninitialised_cache_report_zero_ratio_fields() {
    let dir = TempDir::new().unwrap();
    let stats = CacheManager::new(dir.path()).stats().unwrap();
    assert_eq!(stats.corpus_bytes, 0);
    assert_eq!(stats.trigram_index_bytes, 0);
}

#[test]
fn ratio_fields_serialize_only_when_nonzero() {
    let empty = IndexStats::default();
    let json = serde_json::to_value(&empty).unwrap();
    assert!(json.get("corpus_bytes").is_none());
    assert!(json.get("trigram_index_bytes").is_none());

    let filled = IndexStats {
        corpus_bytes: 32_350_466,
        trigram_index_bytes: 30_281_814,
        ..Default::default()
    };
    let json = serde_json::to_value(&filled).unwrap();
    assert_eq!(json["corpus_bytes"], 32_350_466u64);
    assert_eq!(json["trigram_index_bytes"], 30_281_814u64);

    // Pre-1.8 JSON without the fields still deserializes (defaults to 0)
    let legacy = r#"{"total_files":1,"index_size_bytes":10,"last_updated":"x","files_by_language":{},"lines_by_language":{}}"#;
    let parsed: IndexStats = serde_json::from_str(legacy).unwrap();
    assert_eq!(parsed.corpus_bytes, 0);
    assert_eq!(parsed.trigram_index_bytes, 0);
}
