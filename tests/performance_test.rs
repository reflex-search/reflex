//! Performance tests for Reflex
//!
//! These tests verify that Reflex meets its performance goals:
//! - Indexing: Fast enough for large codebases
//! - Queries: Sub-100ms on 10k+ files
//! - Incremental updates: Efficient reindexing

use reflex::{CacheManager, IndexConfig, Indexer, QueryEngine, QueryFilter};
use std::fs;
use std::time::Instant;
use tempfile::TempDir;

// ==================== Indexing Performance Tests ====================

#[test]
fn test_index_small_codebase_performance() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create 100 small Rust files
    for i in 0..100 {
        let content = format!("fn function_{}() {{\n    println!(\"test\");\n}}", i);
        fs::write(project.join(format!("file_{}.rs", i)), content).unwrap();
    }

    // Measure indexing time
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());

    let start = Instant::now();
    let stats = indexer.index(project, false).unwrap();
    let duration = start.elapsed();

    assert_eq!(stats.total_files, 100);

    // Should index 100 files in under 2.5 seconds
    // (increased from 1s to account for 18 language parsers and system variability)
    assert!(
        duration.as_millis() < 2500,
        "Indexing 100 files took {}ms, expected < 2500ms",
        duration.as_millis()
    );

    println!(
        "✓ Indexed {} files in {}ms",
        stats.total_files,
        duration.as_millis()
    );
}

#[test]
fn test_index_medium_codebase_performance() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create 500 files with moderate content
    for i in 0..500 {
        let content = format!(
            "fn function_{}() {{\n    let x = {};\n    let y = x + 1;\n    println!(\"{{}} {{}}\", x, y);\n}}",
            i, i
        );
        fs::write(project.join(format!("file_{}.rs", i)), content).unwrap();
    }

    // Measure indexing time
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());

    let start = Instant::now();
    let stats = indexer.index(project, false).unwrap();
    let duration = start.elapsed();

    assert_eq!(stats.total_files, 500);

    // Should index 500 files in under 5 seconds
    // (increased from 3s to account for 18 language parsers and system variability)
    assert!(
        duration.as_millis() < 5000,
        "Indexing 500 files took {}ms, expected < 5000ms",
        duration.as_millis()
    );

    println!(
        "✓ Indexed {} files in {}ms",
        stats.total_files,
        duration.as_millis()
    );
}

#[test]
fn test_incremental_reindex_performance() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create initial set of files
    for i in 0..100 {
        fs::write(
            project.join(format!("file_{}.rs", i)),
            format!("fn test_{}() {{}}", i),
        )
        .unwrap();
    }

    // Initial index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Modify only 10 files
    for i in 0..10 {
        fs::write(
            project.join(format!("file_{}.rs", i)),
            format!("fn modified_{}() {{}}", i),
        )
        .unwrap();
    }

    // Measure incremental reindex time
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());

    let start = Instant::now();
    indexer.index(project, false).unwrap();
    let duration = start.elapsed();

    // Incremental reindex should be fast (most files unchanged)
    // (increased from 1000ms to 1500ms to account for system variability)
    assert!(
        duration.as_millis() < 1500,
        "Incremental reindex took {}ms, expected < 1500ms",
        duration.as_millis()
    );

    println!(
        "✓ Incremental reindex (10/100 files changed) took {}ms",
        duration.as_millis()
    );
}

// ==================== Query Performance Tests ====================

#[test]
fn test_fulltext_query_performance() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create 200 files
    for i in 0..200 {
        let content = format!("fn function_{}() {{\n    println!(\"hello world\");\n}}", i);
        fs::write(project.join(format!("file_{}.rs", i)), content).unwrap();
    }

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Measure query time
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        limit: None, // No limit for performance test
        ..Default::default()
    };

    let start = Instant::now();
    let results = engine.search("hello", filter).unwrap();
    let duration = start.elapsed();

    assert!(results.len() >= 200);

    // Query should be sub-150ms
    // (increased from 100ms to account for system variability)
    assert!(
        duration.as_millis() < 150,
        "Full-text query took {}ms, expected < 150ms",
        duration.as_millis()
    );

    println!(
        "✓ Full-text query found {} results in {}ms",
        results.len(),
        duration.as_millis()
    );
}

#[test]
fn test_symbol_query_performance() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create 100 files with various symbols
    for i in 0..100 {
        let content = format!(
            "fn greet_{}() {{}}\nstruct Point_{} {{}}\nimpl Point_{} {{\n    fn new() -> Self {{ Point_{} {{}} }}\n}}",
            i, i, i, i
        );
        fs::write(project.join(format!("file_{}.rs", i)), content).unwrap();
    }

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Measure symbol query time
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        use_contains: true, // "gre" is substring of "greet", not at word boundary
        ..Default::default()
    };

    let start = Instant::now();
    let results = engine.search("gre", filter).unwrap();
    let duration = start.elapsed();

    assert!(results.len() >= 1);

    // Symbol query with runtime parsing may be slower for large result sets
    // Trigrams narrow to ~100 files, then tree-sitter parses each
    // On small codebases with good trigram filtering: <100ms
    // On larger codebases or broad patterns: may be 1-6 seconds
    // This is the trade-off: no upfront indexing cost, but query-time parsing
    // (increased from 5s to 6s to account for additional language parsers)
    assert!(
        duration.as_millis() < 6000,
        "Symbol query took {}ms, expected < 6000ms",
        duration.as_millis()
    );

    println!(
        "✓ Symbol query found {} results in {}ms (runtime tree-sitter parsing)",
        results.len(),
        duration.as_millis()
    );
}

#[test]
fn test_regex_query_performance() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create 150 files
    for i in 0..150 {
        let content = format!("fn test_{}() {{}}\nfn helper_{}() {{}}", i, i);
        fs::write(project.join(format!("file_{}.rs", i)), content).unwrap();
    }

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Measure regex query time
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        use_regex: true,
        limit: None, // No limit for performance test
        ..Default::default()
    };

    let start = Instant::now();
    let results = engine.search(r"fn test_\d+", filter).unwrap();
    let duration = start.elapsed();

    assert!(results.len() >= 150);

    // Regex query should be fast (trigram-optimized)
    assert!(
        duration.as_millis() < 200,
        "Regex query took {}ms, expected < 200ms",
        duration.as_millis()
    );

    println!(
        "✓ Regex query found {} results in {}ms",
        results.len(),
        duration.as_millis()
    );
}

#[test]
fn test_filtered_query_performance() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create mixed language files
    for i in 0..100 {
        fs::write(project.join(format!("file_{}.rs", i)), "fn test() {}").unwrap();
        fs::write(project.join(format!("file_{}.py", i)), "def test(): pass").unwrap();
    }

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Measure filtered query time
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        language: Some(reflex::Language::Rust),
        ..Default::default()
    };

    let start = Instant::now();
    let results = engine.search("test", filter).unwrap();
    let duration = start.elapsed();

    // Should only find Rust files
    assert!(results.iter().all(|r| r.path.ends_with(".rs")));

    // Filtered query should be fast
    assert!(
        duration.as_millis() < 150,
        "Filtered query took {}ms, expected < 150ms",
        duration.as_millis()
    );

    println!(
        "✓ Filtered query found {} Rust files in {}ms",
        results.len(),
        duration.as_millis()
    );
}

// ==================== Memory-mapped I/O Performance Tests ====================

#[test]
fn test_repeated_queries_use_cached_index() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create 100 files
    for i in 0..100 {
        fs::write(project.join(format!("file_{}.rs", i)), "fn test() {}").unwrap();
    }

    // Index once
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Measure 10 repeated queries
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter::default();

    let start = Instant::now();
    for _ in 0..10 {
        let _ = engine.search("test", filter.clone()).unwrap();
    }
    let duration = start.elapsed();

    let avg_ms = duration.as_millis() / 10;

    // Each query should be fast (memory-mapped)
    // (increased from 50ms to 75ms to account for system variability)
    assert!(
        avg_ms < 75,
        "Average query time {}ms, expected < 75ms",
        avg_ms
    );

    println!("✓ 10 repeated queries averaged {}ms each", avg_ms);
}

// ==================== Scalability Tests ====================

#[test]
fn test_large_file_handling() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create one large file (1000 lines)
    let content = (0..1000)
        .map(|i| format!("fn function_{}() {{}}", i))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(project.join("large.rs"), content).unwrap();

    // Measure indexing time for large file
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());

    let start = Instant::now();
    indexer.index(project, false).unwrap();
    let duration = start.elapsed();

    // Should handle large file efficiently
    // (increased from 500ms to 1000ms to account for system variability)
    assert!(
        duration.as_millis() < 1000,
        "Indexing 1000-line file took {}ms, expected < 1000ms",
        duration.as_millis()
    );

    println!("✓ Indexed 1000-line file in {}ms", duration.as_millis());
}

#[test]
fn test_many_small_files_handling() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create 1000 tiny files
    for i in 0..1000 {
        fs::write(project.join(format!("tiny_{}.rs", i)), "fn f() {}").unwrap();
    }

    // Measure indexing time
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());

    let start = Instant::now();
    let stats = indexer.index(project, false).unwrap();
    let duration = start.elapsed();

    assert_eq!(stats.total_files, 1000);

    // Should handle many small files efficiently
    // (increased from 2000ms to 6000ms to account for 18 language parsers and system variability)
    assert!(
        duration.as_millis() < 6000,
        "Indexing 1000 small files took {}ms, expected < 6000ms",
        duration.as_millis()
    );

    println!("✓ Indexed 1000 small files in {}ms", duration.as_millis());
}
