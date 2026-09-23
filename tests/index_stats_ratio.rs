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

/// Tracked mode adds the long tail of small text files (docs, config,
/// extensionless names, lock files). The two stores must stay within what the
/// V4 format promises on a fixture shaped like that tail.
#[test]
fn tracked_mode_long_tail_keeps_the_index_within_the_format_bound() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let mut corpus = 0u64;
    let mut write = |rel: &str, body: String| {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        corpus += body.len() as u64;
        fs::write(p, body).unwrap();
    };
    // Code: the bulk.
    for i in 0..20 {
        let body: String = (0..120)
            .map(|n| {
                format!("pub fn handler_{i}_{n}(req: Request) -> Response {{ route(req, {n}) }}\n")
            })
            .collect();
        write(&format!("src/mod_{i}.rs"), body);
    }
    // The long tail: many small files of many shapes, ≥ 200 KB in total.
    for i in 0..60 {
        write(
            &format!("docs/page_{i}.md"),
            format!("# Page {i}\n\nSee handler_{i}_0 and the config key realm_{i}.\n").repeat(12),
        );
        write(
            &format!("config/svc_{i}.yaml"),
            format!("service: svc_{i}\nrealm: realm_{i}\nreplicas: {i}\n").repeat(10),
        );
        write(
            &format!("owners/OWNERS_{i}"),
            format!("approvers:\n  - owner_{i}\nreviewers:\n  - reviewer_{i}\n"),
        );
        write(
            &format!("i18n/msg_{i}.po"),
            format!("msgid \"greeting_{i}\"\nmsgstr \"hello {i}\"\n").repeat(8),
        );
        write(
            &format!("web/style_{i}.css"),
            format!(".realm-{i} {{ color: #00{i:02x}00; margin: {i}px }}\n").repeat(6),
        );
        write(
            &format!("data/rows_{i}.jsonl"),
            format!("{{\"id\": {i}, \"realm\": \"realm_{i}\", \"ok\": true}}\n").repeat(10),
        );
    }
    write("Cargo.lock", (0..300).map(|i| format!("[[package]]\nname = \"crate_{i}\"\nversion = \"1.0.{i}\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\n\n")).collect());
    assert!(
        corpus >= 200 * 1024,
        "fixture must be at least 200 KB, is {corpus}"
    );

    let stats = Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .expect("index");
    assert_eq!(stats.corpus_bytes, corpus);
    assert_eq!(stats.files_by_language.get("Lock"), Some(&1));

    // What the V4 format can promise. A posting is one per distinct trigram per
    // LINE, so its cost tracks distinct-trigram density: a code line repeats its
    // indentation and identifiers (Reflex's own source: 1.4x in trigrams.bin), a
    // prose line is nearly all distinct trigrams (Reflex's docs alone: 2.5x). That
    // is why a text-heavy tree (Kubernetes, 2.1x total) sits above a PHP-heavy one
    // (Hearth, 1.3x). This mixed fixture is ~40% text by bytes, far more than any
    // real repo, and lands near 1.25x trigrams / 1.05x content; the gate leaves
    // headroom for the format, not for a regression.
    let content_len = fs::metadata(root.join(".reflex/content.bin"))
        .unwrap()
        .len();
    let trigram_ratio = stats.trigram_index_bytes as f64 / corpus as f64;
    let content_ratio = content_len as f64 / corpus as f64;
    assert!(
        trigram_ratio <= 1.6,
        "trigrams.bin is {trigram_ratio:.2}x the corpus (limit 1.6x): trigrams {} corpus {} files {}",
        stats.trigram_index_bytes,
        corpus,
        stats.total_files
    );
    assert!(
        content_ratio <= 1.15,
        "content.bin is {content_ratio:.2}x the corpus (limit 1.15x): content {} corpus {} files {}",
        content_len,
        corpus,
        stats.total_files
    );
}
