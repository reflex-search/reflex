//! Test Helper Functions for Corpus-Based Testing
//!
//! This module provides utilities for testing Reflex against the test corpus.
#![allow(dead_code)] // helper functions used selectively across test files

use reflex::{
    CacheManager, IndexConfig, Indexer, QueryEngine, QueryFilter, SearchResult, SymbolKind,
};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static CORPUS_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Initialize and index the test corpus once
/// Returns the path to the indexed corpus
pub fn setup_corpus() -> &'static Path {
    CORPUS_PATH
        .get_or_init(|| {
            let corpus = PathBuf::from("tests/corpus");

            // Index the corpus
            let cache = CacheManager::new(&corpus);
            let indexer = Indexer::new(cache, IndexConfig::default());
            indexer
                .index(&corpus, false)
                .expect("Failed to index corpus");

            corpus
        })
        .as_path()
}

/// Create a query engine for the corpus
pub fn query_engine() -> QueryEngine {
    let corpus = setup_corpus();
    let cache = CacheManager::new(corpus);
    QueryEngine::new(cache)
}

/// Execute a query on the corpus
pub fn query_corpus(pattern: &str, filter: QueryFilter) -> Vec<SearchResult> {
    query_engine()
        .search(pattern, filter)
        .expect("Query failed")
}

/// Assert that a symbol with the given name and kind was found
pub fn assert_symbol_found(results: &[SearchResult], name: &str, kind: SymbolKind) {
    assert!(
        results
            .iter()
            .any(|r| { r.symbol.as_deref() == Some(name) && r.kind == kind }),
        "Expected to find symbol '{}' of kind {:?}, but it was not in results",
        name,
        kind
    );
}

/// Assert that results contain a file matching the path pattern
pub fn assert_file_match(results: &[SearchResult], path_contains: &str) {
    assert!(
        results.iter().any(|r| r.path.contains(path_contains)),
        "Expected to find result in file containing '{}', but no match found",
        path_contains
    );
}

/// Assert exact result count
pub fn assert_result_count(results: &[SearchResult], expected: usize) {
    assert_eq!(
        results.len(),
        expected,
        "Expected {} results, but got {}",
        expected,
        results.len()
    );
}

/// Assert result count is at least the given value
pub fn assert_result_count_at_least(results: &[SearchResult], min: usize) {
    assert!(
        results.len() >= min,
        "Expected at least {} results, but got {}",
        min,
        results.len()
    );
}

/// Assert result count is at most the given value
pub fn assert_result_count_at_most(results: &[SearchResult], max: usize) {
    assert!(
        results.len() <= max,
        "Expected at most {} results, but got {}",
        max,
        results.len()
    );
}

/// Assert all results are of the specified kind
pub fn assert_all_kind(results: &[SearchResult], kind: SymbolKind) {
    for result in results {
        assert_eq!(
            result.kind, kind,
            "Expected all results to be {:?}, but found {:?}",
            kind, result.kind
        );
    }
}

/// Assert all results are from files with the given language
pub fn assert_all_language(results: &[SearchResult], lang: reflex::Language) {
    for result in results {
        assert_eq!(
            result.lang, lang,
            "Expected all results from {:?}, but found {:?}",
            lang, result.lang
        );
    }
}

/// Assert results are sorted deterministically (by path, then line)
pub fn assert_sorted(results: &[SearchResult]) {
    for i in 0..results.len().saturating_sub(1) {
        let curr = &results[i];
        let next = &results[i + 1];

        assert!(
            curr.path < next.path
                || (curr.path == next.path && curr.span.start_line <= next.span.start_line),
            "Results are not sorted correctly at index {}",
            i
        );
    }
}

/// Assert that the preview contains the pattern
pub fn assert_preview_contains(results: &[SearchResult], pattern: &str) {
    assert!(
        results.iter().any(|r| r.preview.contains(pattern)),
        "Expected at least one preview to contain '{}'",
        pattern
    );
}

/// Count results by kind
pub fn count_by_kind(results: &[SearchResult], kind: SymbolKind) -> usize {
    results.iter().filter(|r| r.kind == kind).count()
}

/// Count results by file pattern
pub fn count_by_file_pattern(results: &[SearchResult], pattern: &str) -> usize {
    results.iter().filter(|r| r.path.contains(pattern)).count()
}

/// Get all unique file paths from results
pub fn unique_files(results: &[SearchResult]) -> Vec<String> {
    use std::collections::HashSet;
    let mut files: HashSet<String> = results.iter().map(|r| r.path.clone()).collect();
    let mut vec: Vec<String> = files.drain().collect();
    vec.sort();
    vec
}

/// Assert no duplicates (same file and line)
pub fn assert_no_duplicates(results: &[SearchResult]) {
    use std::collections::HashSet;
    let mut seen = HashSet::new();

    for result in results {
        let key = (result.path.clone(), result.span.start_line);
        assert!(
            seen.insert(key.clone()),
            "Duplicate result found: {:?}",
            key
        );
    }
}

// ==================== Glob/Exclude/Paths Helper Functions ====================

/// Assert all results match at least one of the glob patterns
pub fn assert_all_match_glob(results: &[SearchResult], patterns: &[String]) {
    use globset::{Glob, GlobSetBuilder};

    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        builder.add(Glob::new(pattern).unwrap());
    }
    let matcher = builder.build().unwrap();

    for result in results {
        assert!(
            matcher.is_match(&result.path),
            "Result path '{}' does not match any glob pattern: {:?}",
            result.path,
            patterns
        );
    }
}

/// Assert no results match any of the exclude patterns
pub fn assert_none_match_exclude(results: &[SearchResult], patterns: &[String]) {
    use globset::{Glob, GlobSetBuilder};

    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        builder.add(Glob::new(pattern).unwrap());
    }
    let matcher = builder.build().unwrap();

    for result in results {
        assert!(
            !matcher.is_match(&result.path),
            "Result path '{}' matches excluded pattern: {:?}",
            result.path,
            patterns
        );
    }
}

/// Assert all results are from paths containing the given substring
pub fn assert_all_paths_contain(results: &[SearchResult], substring: &str) {
    for result in results {
        assert!(
            result.path.contains(substring),
            "Expected path to contain '{}', but got '{}'",
            substring,
            result.path
        );
    }
}

/// Assert no results are from paths containing the given substring
pub fn assert_no_paths_contain(results: &[SearchResult], substring: &str) {
    for result in results {
        assert!(
            !result.path.contains(substring),
            "Expected path to not contain '{}', but got '{}'",
            substring,
            result.path
        );
    }
}

/// Assert all paths are unique (for paths-only mode)
pub fn assert_all_paths_unique(results: &[SearchResult]) {
    let files = unique_files(results);
    assert_eq!(
        results.len(),
        files.len(),
        "Expected all paths to be unique, but found {} results with only {} unique paths",
        results.len(),
        files.len()
    );
}

/// Assert results contain paths from a specific directory
pub fn assert_has_paths_from_dir(results: &[SearchResult], dir: &str) {
    assert!(
        results.iter().any(|r| r.path.contains(dir)),
        "Expected at least one result from directory '{}', but found none",
        dir
    );
}

/// Assert results do not contain paths from a specific directory
pub fn assert_no_paths_from_dir(results: &[SearchResult], dir: &str) {
    assert!(
        results.iter().all(|r| !r.path.contains(dir)),
        "Expected no results from directory '{}', but found some",
        dir
    );
}

/// Assert all paths match a specific file extension
pub fn assert_all_paths_extension(results: &[SearchResult], extension: &str) {
    let ext = if extension.starts_with('.') {
        extension.to_string()
    } else {
        format!(".{}", extension)
    };

    for result in results {
        assert!(
            result.path.ends_with(&ext),
            "Expected path to end with '{}', but got '{}'",
            ext,
            result.path
        );
    }
}

/// Count results from a specific directory
pub fn count_from_dir(results: &[SearchResult], dir: &str) -> usize {
    results.iter().filter(|r| r.path.contains(dir)).count()
}

/// Count unique paths in results
pub fn count_unique_paths(results: &[SearchResult]) -> usize {
    unique_files(results).len()
}

// ==================== Synthetic Latency Corpus ====================

/// Deterministic synthetic Rust corpus for the latency harness
/// (`tests/latency_budget.rs`).
///
/// The corpus reproduces the query shapes of a field test on a ~30 MiB,
/// ~2000-file workspace without checking a large fixture into git:
///
/// | planted token      | where                                        |
/// |--------------------|----------------------------------------------|
/// | `config`           | ~1 in 10 body lines (common *word*)          |
/// | `ident_7`          | rank-8 identifier of a Zipf(s=1) vocabulary  |
/// | `rare_marker_q7`   | exactly [`RARE_MARKER_LINES`] lines          |
/// | `zzqx_absent`      | never                                        |
/// | `fn get_<x>()` / `fn set_<x>()` | exactly one of each per file    |
///
/// Identifier slots are sampled from 20 000 `ident_N` tokens by binary search
/// on a precomputed Zipf CDF, so `ident_N` has rank `N + 1`. Rust keywords
/// (`fn`, `let`, `mut`, `pub`, `struct`, `if`, …) enter the token stream through
/// the line templates rather than the sampled slots, so the output stays
/// syntactically shaped like Rust and tree-sitter parses it without error
/// cascades.
///
/// Generation is cached: a `.generated-<seed>` marker at the root short-circuits
/// a rerun, and [`indexed`] additionally skips indexing when `.reflex/meta.db`
/// exists. Delete the directory to force a rebuild.
pub mod synthetic_corpus {
    use reflex::{CacheManager, IndexConfig, Indexer};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    /// Seed used by the latency harness.
    pub const DEFAULT_SEED: u64 = 7;
    /// File count used by the latency harness.
    pub const DEFAULT_FILES: usize = 2000;
    /// Byte target used by the latency harness (30 MiB).
    pub const DEFAULT_BYTES: usize = 30 * 1024 * 1024;

    /// Common word planted on ~1 in 10 body lines.
    pub const COMMON_WORD: &str = "config";
    /// Common identifier (rank 8 of the Zipf vocabulary).
    pub const COMMON_IDENT: &str = "ident_7";
    /// Rare identifier planted on exactly [`RARE_MARKER_LINES`] lines.
    pub const RARE_MARKER: &str = "rare_marker_q7";
    /// Number of lines carrying [`RARE_MARKER`].
    pub const RARE_MARKER_LINES: usize = 3;
    /// A function defined exactly once in the corpus (`pub fn get_<file_no>()` is
    /// planted per file), for the symbol-lookup and find-references shapes.
    pub const RARE_FN: &str = "get_1234";
    /// Token that never appears in the corpus.
    pub const ABSENT: &str = "zzqx_absent";
    /// Regex that hits every file (two lines per file).
    pub const GETSET_REGEX: &str = r"fn (get|set)_\w+";
    /// Case-insensitive regex over [`RARE_MARKER`]: its literal must still reach
    /// the trigram index (looked up under every case variant), not a full scan.
    pub const CI_REGEX: &str = "(?i)RARE_MARKER_Q7";

    const VOCAB_SIZE: usize = 20_000;
    const CONFIG_LINE_PROBABILITY: f64 = 0.10;

    /// xorshift64 — deterministic, dependency-free.
    struct XorShift64(u64);

    impl XorShift64 {
        fn new(seed: u64) -> Self {
            // xorshift has a fixed point at 0; a splitmix step avoids it
            // and decorrelates small seeds.
            let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            Self((z ^ (z >> 31)) | 1)
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        /// Uniform in `[0, 1)`.
        fn next_f64(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Zipf(s = 1.0) vocabulary: `tokens[i]` has probability ∝ 1 / (i + 1).
    struct Vocab {
        tokens: Vec<String>,
        cdf: Vec<f64>,
    }

    impl Vocab {
        fn new() -> Self {
            let tokens: Vec<String> = (0..VOCAB_SIZE).map(|i| format!("ident_{i}")).collect();
            let weights: Vec<f64> = (1..=VOCAB_SIZE).map(|r| 1.0 / r as f64).collect();
            let norm: f64 = weights.iter().sum();
            let mut acc = 0.0;
            let cdf = weights
                .iter()
                .map(|w| {
                    acc += w / norm;
                    acc
                })
                .collect();
            Self { tokens, cdf }
        }

        fn sample(&self, rng: &mut XorShift64) -> &str {
            let u = rng.next_f64();
            let idx = self
                .cdf
                .partition_point(|&c| c < u)
                .min(self.tokens.len() - 1);
            &self.tokens[idx]
        }
    }

    /// One Rust-shaped body line of 3–12 tokens. `config_line` forces the
    /// common-word plant onto this line.
    fn body_line(v: &Vocab, rng: &mut XorShift64, config_line: bool) -> String {
        let t = |rng: &mut XorShift64| v.sample(rng).to_string();
        if config_line {
            return match rng.below(4) {
                0 => format!("    let config = {}({}, {});", t(rng), t(rng), t(rng)),
                1 => format!("    {}.config = {} + {};", t(rng), t(rng), t(rng)),
                2 => format!("    // config {} {} {}", t(rng), t(rng), t(rng)),
                _ => format!(
                    "    let {} = config.{}({}, {}, {});",
                    t(rng),
                    t(rng),
                    t(rng),
                    t(rng),
                    t(rng)
                ),
            };
        }
        match rng.below(8) {
            0 => format!("    let {} = {};", t(rng), t(rng)),
            1 => format!("    let mut {} = {} + {};", t(rng), t(rng), t(rng)),
            2 => format!(
                "    if {} > {} {{ {} = {}; }}",
                t(rng),
                t(rng),
                t(rng),
                t(rng)
            ),
            3 => format!(
                "    {} = {}({}, {}, {});",
                t(rng),
                t(rng),
                t(rng),
                t(rng),
                t(rng)
            ),
            4 => format!("    // {} {} {} {}", t(rng), t(rng), t(rng), t(rng)),
            5 => format!(
                "    let {} = {}.{}({}).{}({}, {});",
                t(rng),
                t(rng),
                t(rng),
                t(rng),
                t(rng),
                t(rng),
                t(rng)
            ),
            6 => format!(
                "    while {} < {} {{ {} += {}; {} = {}; }}",
                t(rng),
                t(rng),
                t(rng),
                t(rng),
                t(rng),
                t(rng)
            ),
            _ => format!(
                "    match {} {{ 0 => {}, _ => {} }};",
                t(rng),
                t(rng),
                t(rng)
            ),
        }
    }

    /// Generate one file of roughly `target_len` bytes. `rare_lines` is how many
    /// [`RARE_MARKER`] lines to plant in this file.
    fn file_content(
        v: &Vocab,
        rng: &mut XorShift64,
        file_no: usize,
        target_len: usize,
        rare_lines: usize,
    ) -> String {
        let mut s = String::with_capacity(target_len + 512);
        s.push_str(&format!("//! synthetic corpus file {file_no}\n\n"));
        s.push_str(&format!(
            "pub struct S{file_no} {{\n    pub {}: u64,\n    pub {}: String,\n}}\n\n",
            v.sample(rng),
            v.sample(rng)
        ));
        s.push_str(&format!(
            "pub fn get_{file_no}() -> u64 {{\n    {file_no}\n}}\n\n"
        ));
        s.push_str(&format!(
            "pub fn set_{file_no}(v: u64) {{\n    let _ = v;\n}}\n\n"
        ));

        let mut planted = 0;
        let mut fn_no = 0;
        while s.len() < target_len {
            s.push_str(&format!(
                "pub fn body_{fn_no}({}: u64, {}: u64) -> u64 {{\n",
                v.sample(rng),
                v.sample(rng)
            ));
            let lines = 5 + rng.below(16);
            for _ in 0..lines {
                let config_line = rng.next_f64() < CONFIG_LINE_PROBABILITY;
                s.push_str(&body_line(v, rng, config_line));
                s.push('\n');
            }
            if planted < rare_lines {
                s.push_str(&format!("    let {RARE_MARKER} = {};\n", v.sample(rng)));
                planted += 1;
            }
            s.push_str(&format!("    {}\n}}\n\n", v.sample(rng)));
            fn_no += 1;
        }
        s
    }

    /// Generate the corpus under `root` (skipped when `<root>/.generated-<seed>`
    /// exists). Files land at `src/mod_<i>/file_<j>.rs`, 50 per module.
    pub fn generate(root: &Path, seed: u64, files: usize, target_bytes: usize) {
        let marker = root.join(format!(".generated-{seed}"));
        if marker.exists() {
            return;
        }
        // No marker: any partial tree (or a stale index of one) is untrustworthy.
        if root.exists() {
            fs::remove_dir_all(root).expect("remove partial corpus");
        }

        let vocab = Vocab::new();
        let mut rng = XorShift64::new(seed);
        let per_file = target_bytes / files.max(1);
        let per_mod = 50;

        // Spread the rare marker over three distinct files.
        let rare_files: Vec<usize> = (1..=RARE_MARKER_LINES)
            .map(|k| k * files / (RARE_MARKER_LINES + 1))
            .collect();

        for file_no in 0..files {
            let dir = root.join(format!("src/mod_{}", file_no / per_mod));
            fs::create_dir_all(&dir).expect("create corpus dir");
            let rare = rare_files.iter().filter(|&&f| f == file_no).count();
            let content = file_content(&vocab, &mut rng, file_no, per_file, rare);
            fs::write(dir.join(format!("file_{}.rs", file_no % per_mod)), content)
                .expect("write corpus file");
        }

        fs::write(
            &marker,
            format!("seed={seed} files={files} bytes={target_bytes}\n"),
        )
        .expect("write corpus marker");
    }

    /// Root of the cached corpus for `seed`: `<CARGO_TARGET_DIR|target>/latency_corpus/<seed>`.
    pub fn root_for(seed: u64) -> PathBuf {
        let target = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
        PathBuf::from(target)
            .join("latency_corpus")
            .join(seed.to_string())
    }

    /// Generate (if needed) and index (if `.reflex/meta.db` is absent) the
    /// default-sized corpus for `seed`, returning its root. Serialised across
    /// threads so parallel tests in one binary share a single build.
    pub fn indexed(seed: u64) -> PathBuf {
        static BUILD: Mutex<()> = Mutex::new(());
        let _guard = BUILD.lock().unwrap_or_else(|e| e.into_inner());

        let root = root_for(seed);
        generate(&root, seed, DEFAULT_FILES, DEFAULT_BYTES);

        // Always run the indexer: it is a no-op-fast incremental pass when the
        // cached index is current, and a full rebuild when a cached index was
        // written by a binary with a different cache schema (otherwise every
        // query would pay the in-memory format fallback and the harness would
        // measure that instead of the on-disk path).
        let cache = CacheManager::new(&root);
        if !cache.check_schema_hash().unwrap_or(false) {
            let _ = cache.clear();
        }
        Indexer::new(CacheManager::new(&root), IndexConfig::default())
            .index(&root, false)
            .expect("index synthetic corpus");
        root
    }
}
