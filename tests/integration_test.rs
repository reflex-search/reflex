//! Integration tests for Reflex

use reflex::{CacheManager, IndexConfig, Indexer, QueryEngine, QueryFilter, SymbolKind};
use std::fs;
use tempfile::TempDir;

// ==================== Basic Workflow Tests ====================

#[test]
fn test_full_workflow() {
    // Create a temporary directory for testing
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Create a sample source file
    let sample_code = r#"
fn main() {
    println!("Hello, Reflex!");
}

fn greet(name: &str) {
    println!("Hello, {}!", name);
}
"#;
    std::fs::write(temp_path.join("test.rs"), sample_code).unwrap();

    // Initialize cache and indexer
    let cache = CacheManager::new(temp_path);
    let config = IndexConfig::default();
    let indexer = Indexer::new(cache, config);

    // Index the directory
    let stats = indexer.index(temp_path, false).unwrap();
    assert_eq!(stats.total_files, 1); // One test.rs file
    // Note: total_symbols is always 0 now (runtime symbol detection, not indexed)

    // Query the index (create new cache instance for query engine)
    let cache = CacheManager::new(temp_path);
    let engine = QueryEngine::new(cache);
    let results = engine.find_symbol("main").unwrap();
    assert_eq!(results.len(), 1); // Should find the main function
    assert_eq!(results[0].symbol.as_deref(), Some("main"));
}

#[test]
fn test_cache_initialization() {
    let temp_dir = TempDir::new().unwrap();
    let cache = CacheManager::new(temp_dir.path());

    assert!(!cache.exists());
    cache.init().unwrap();
    assert!(cache.path().exists());
}

#[test]
fn test_cache_clear() {
    let temp_dir = TempDir::new().unwrap();
    let cache = CacheManager::new(temp_dir.path());

    cache.init().unwrap();
    assert!(cache.path().exists());

    cache.clear().unwrap();
    assert!(!cache.path().exists());
}

// ==================== End-to-End Workflow Tests ====================

#[test]
fn test_index_and_fulltext_search_workflow() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create multiple source files
    fs::write(project.join("main.rs"), "fn main() {\n    println!(\"hello world\");\n}").unwrap();
    fs::write(project.join("lib.rs"), "pub fn hello() -> String {\n    \"hello\".to_string()\n}").unwrap();
    fs::write(project.join("utils.rs"), "// hello helper\nfn helper() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    let stats = indexer.index(project, false).unwrap();
    assert_eq!(stats.total_files, 3);

    // Full-text search for "hello"
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter::default();
    let results = engine.search("hello", filter).unwrap();

    // Should find all 3 occurrences (println, function name, comment)
    assert!(results.len() >= 3);
    assert!(results.iter().any(|r| r.path.contains("main.rs")));
    assert!(results.iter().any(|r| r.path.contains("lib.rs")));
    assert!(results.iter().any(|r| r.path.contains("utils.rs")));
}

#[test]
fn test_index_and_symbol_search_workflow() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create files with symbols
    fs::write(
        project.join("main.rs"),
        "fn greet() {}\nfn main() {\n    greet();\n}"
    ).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Symbol search for "greet"
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        ..Default::default()
    };
    let results = engine.search("greet", filter).unwrap();

    // Should find definition, not call site
    assert!(results.len() >= 1);
    assert!(results.iter().all(|r| r.kind == SymbolKind::Function));
}

#[test]
fn test_index_and_regex_search_workflow() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::write(
        project.join("main.rs"),
        "fn test1() {}\nfn test2() {}\nfn other() {}"
    ).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Regex search
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        use_regex: true,
        ..Default::default()
    };
    let results = engine.search(r"fn test\d", filter).unwrap();

    // Should match test1 and test2 but not other
    assert_eq!(results.len(), 2);
}

#[test]
fn test_incremental_indexing_workflow() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Initial index with one file
    fs::write(project.join("main.rs"), "fn main() {}").unwrap();

    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    let stats1 = indexer.index(project, false).unwrap();
    assert_eq!(stats1.total_files, 1);

    // Add another file and reindex
    fs::write(project.join("lib.rs"), "pub fn test() {}").unwrap();

    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    let stats2 = indexer.index(project, false).unwrap();
    assert_eq!(stats2.total_files, 2);

    // Verify both files are searchable (search for "main" which appears in both)
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        use_contains: true,  // "mai" is substring of "main", not at word boundary
        ..Default::default()
    };
    let results = engine.search("mai", filter).unwrap(); // "mai" is a trigram in "main"
    assert!(results.len() >= 1, "Should find at least main.rs");
}

#[test]
fn test_modify_file_and_reindex_workflow() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();
    let main_path = project.join("main.rs");

    // Initial index
    fs::write(&main_path, "fn old_function() {}").unwrap();

    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Verify we can find old function
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        use_contains: true,  // "old" in "old_function" is not at word boundary
        ..Default::default()
    };
    let results = engine.search("old", filter.clone()).unwrap();
    assert!(results.len() >= 1);

    // Modify file
    fs::write(&main_path, "fn new_function() {}").unwrap();

    // Reindex
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Verify we can find new function
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let results = engine.search("new", filter).unwrap();
    assert!(results.len() >= 1);
    assert!(results.iter().any(|r| r.symbol.as_ref().map_or(false, |s| s.contains("new"))));
}

// ==================== Multi-language Workflow Tests ====================

#[test]
fn test_multi_language_indexing_and_search() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create files in multiple languages
    fs::write(project.join("main.rs"), "fn greet() {}").unwrap();
    fs::write(project.join("app.ts"), "function greet() {}").unwrap();
    fs::write(project.join("script.py"), "def greet(): pass").unwrap();
    fs::write(project.join("main.js"), "function greet() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    let stats = indexer.index(project, false).unwrap();
    assert_eq!(stats.total_files, 4);

    // Search across all languages
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter::default();
    let results = engine.search("greet", filter).unwrap();
    assert!(results.len() >= 4);
}

#[test]
fn test_language_filtered_search_workflow() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::write(project.join("main.rs"), "fn test() {}").unwrap();
    fs::write(project.join("test.py"), "def test(): pass").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search with language filter
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        language: Some(reflex::Language::Rust),
        ..Default::default()
    };
    let results = engine.search("test", filter).unwrap();

    // Should only find Rust file
    assert!(results.iter().all(|r| r.path.ends_with(".rs")));
}

// ==================== Complex Query Workflow Tests ====================

#[test]
fn test_combined_filters_workflow() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create nested directory structure
    fs::create_dir_all(project.join("src")).unwrap();
    fs::create_dir_all(project.join("tests")).unwrap();

    fs::write(
        project.join("src/lib.rs"),
        "struct Point {}\nfn point_new() {}"
    ).unwrap();
    fs::write(
        project.join("tests/test.rs"),
        "fn test_point() {}"
    ).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search with multiple filters: Rust + function + src/ directory
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        language: Some(reflex::Language::Rust),
        kind: Some(SymbolKind::Function),
        file_pattern: Some("src/".to_string()),
        symbols_mode: true,
        use_contains: true,  // "poi" in "point_new" is not at word boundary
        ..Default::default()
    };
    let results = engine.search("poi", filter).unwrap();

    // Should only find point_new in src/lib.rs
    assert!(results.len() >= 1);
    assert!(results.iter().all(|r| r.path.contains("src/")));
    assert!(results.iter().all(|r| r.kind == SymbolKind::Function));
}

#[test]
fn test_limit_and_sorting_workflow() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create files that will produce many matches
    let content = (0..20).map(|i| format!("fn test{}() {{}}", i)).collect::<Vec<_>>().join("\n");
    fs::write(project.join("many.rs"), content).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search with limit
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        limit: Some(5),
        use_contains: true,  // "test" in "test0", "test1" is not at word boundary
        ..Default::default()
    };
    let results = engine.search("test", filter).unwrap();

    // Should limit to 5 results
    assert_eq!(results.len(), 5);

    // Results should be sorted deterministically
    for i in 0..results.len().saturating_sub(1) {
        assert!(results[i].span.start_line <= results[i + 1].span.start_line);
    }
}

// ==================== Error Handling Workflow Tests ====================

#[test]
fn test_query_without_index_fails() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Don't create index, just try to query
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter::default();
    let result = engine.search("test", filter);

    // Should fail with error
    assert!(result.is_err());
}

#[test]
fn test_index_empty_directory_succeeds() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Index empty directory
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    let stats = indexer.index(project, false).unwrap();

    assert_eq!(stats.total_files, 0);
    // Note: total_symbols is always 0 (runtime symbol detection)
}

#[test]
fn test_search_empty_index_returns_no_results() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Index empty directory
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter::default();
    let results = engine.search("anything", filter).unwrap();

    assert_eq!(results.len(), 0);
}

// ==================== Cache Persistence Workflow Tests ====================

#[test]
fn test_cache_persists_across_sessions() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::write(project.join("main.rs"), "fn test() {}").unwrap();

    // Session 1: Index
    {
        let cache = CacheManager::new(project);
        let indexer = Indexer::new(cache, IndexConfig::default());
        indexer.index(project, false).unwrap();
    }

    // Session 2: Query (new cache instance)
    {
        let cache = CacheManager::new(project);
        let engine = QueryEngine::new(cache);
        let filter = QueryFilter::default();
        let results = engine.search("test", filter).unwrap();
        assert!(results.len() >= 1);
    }
}

#[test]
fn test_clear_and_rebuild_workflow() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::write(project.join("main.rs"), "fn test() {}").unwrap();

    // Initial index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Clear cache
    let cache = CacheManager::new(project);
    cache.clear().unwrap();

    // Rebuild
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    let stats = indexer.index(project, false).unwrap();
    assert_eq!(stats.total_files, 1);

    // Verify search still works
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter::default();
    let results = engine.search("test", filter).unwrap();
    assert!(results.len() >= 1);
}

// ==================== Glob Pattern Tests ====================

#[test]
fn test_glob_single_pattern() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create files in different directories
    fs::create_dir_all(project.join("src")).unwrap();
    fs::create_dir_all(project.join("tests")).unwrap();

    fs::write(project.join("src/main.rs"), "fn extract_pattern() {}").unwrap();
    fs::write(project.join("tests/test.rs"), "fn extract_pattern() {}").unwrap();
    fs::write(project.join("other.rs"), "fn extract_pattern() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search with glob pattern matching only src/
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        glob_patterns: vec!["**/src/**/*.rs".to_string()],
        ..Default::default()
    };
    let results = engine.search("extract_pattern", filter).unwrap();

    // Should only find results in src/
    assert!(results.len() >= 1);
    assert!(results.iter().all(|r| r.path.contains("src/")));
    assert!(results.iter().any(|r| r.path.contains("main.rs")));
    assert!(!results.iter().any(|r| r.path.contains("tests/")));
    assert!(!results.iter().any(|r| r.path.contains("other.rs")));
}

#[test]
fn test_glob_multiple_patterns() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::create_dir_all(project.join("src")).unwrap();
    fs::create_dir_all(project.join("examples")).unwrap();
    fs::create_dir_all(project.join("build")).unwrap();

    fs::write(project.join("src/lib.rs"), "TODO: implement").unwrap();
    fs::write(project.join("examples/demo.rs"), "TODO: add example").unwrap();
    fs::write(project.join("build/gen.rs"), "TODO: generated").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search with multiple glob patterns
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        glob_patterns: vec!["**/src/**".to_string(), "**/examples/**".to_string()],
        ..Default::default()
    };
    let results = engine.search("TODO", filter).unwrap();

    // Should find results in src/ and examples/ but not build/
    assert_eq!(results.len(), 2);
    assert!(results.iter().any(|r| r.path.contains("src/lib.rs")));
    assert!(results.iter().any(|r| r.path.contains("examples/demo.rs")));
    assert!(!results.iter().any(|r| r.path.contains("build/")));
}

#[test]
fn test_glob_wildcard_patterns() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::write(project.join("test_one.rs"), "fn test() {}").unwrap();
    fs::write(project.join("test_two.rs"), "fn test() {}").unwrap();
    fs::write(project.join("other.rs"), "fn test() {}").unwrap();
    fs::write(project.join("main.py"), "def test(): pass").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search with wildcard pattern
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        glob_patterns: vec!["**/test_*.rs".to_string()],
        ..Default::default()
    };
    let results = engine.search("test", filter).unwrap();

    // Should only match test_*.rs files
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.path.contains("test_") && r.path.ends_with(".rs")));
    assert!(!results.iter().any(|r| r.path.contains("other.rs")));
    assert!(!results.iter().any(|r| r.path.contains("main.py")));
}

#[test]
fn test_glob_specific_extension() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::write(project.join("main.rs"), "fn pattern() {}").unwrap();
    fs::write(project.join("app.ts"), "function pattern() {}").unwrap();
    fs::write(project.join("script.py"), "def pattern(): pass").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search for only .rs files
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        glob_patterns: vec!["**/*.rs".to_string()],
        ..Default::default()
    };
    let results = engine.search("pattern", filter).unwrap();

    // Should only find Rust files
    assert_eq!(results.len(), 1);
    assert!(results[0].path.ends_with(".rs"));
}

#[test]
fn test_glob_specific_directory() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::create_dir_all(project.join("src/parsers")).unwrap();
    fs::create_dir_all(project.join("src/utils")).unwrap();

    fs::write(project.join("src/parsers/rust.rs"), "TODO: parse").unwrap();
    fs::write(project.join("src/utils/helpers.rs"), "TODO: help").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search only in parsers directory
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        glob_patterns: vec!["**/src/parsers/**".to_string()],
        ..Default::default()
    };
    let results = engine.search("TODO", filter).unwrap();

    // Should only find results in parsers/
    assert_eq!(results.len(), 1);
    assert!(results[0].path.contains("parsers/rust.rs"));
    assert!(!results.iter().any(|r| r.path.contains("utils/")));
}

// ==================== Exclude Pattern Tests ====================

#[test]
fn test_exclude_single_pattern() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::create_dir_all(project.join("src")).unwrap();
    fs::create_dir_all(project.join("build")).unwrap();

    fs::write(project.join("src/main.rs"), "fn extract_pattern() {}").unwrap();
    fs::write(project.join("build/generated.rs"), "fn extract_pattern() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search with exclude pattern
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        exclude_patterns: vec!["**/build/**".to_string()],
        ..Default::default()
    };
    let results = engine.search("extract_pattern", filter).unwrap();

    // Should find results in src/ but not build/
    assert_eq!(results.len(), 1);
    assert!(results[0].path.contains("src/main.rs"));
    assert!(!results.iter().any(|r| r.path.contains("build/")));
}

#[test]
fn test_exclude_multiple_patterns() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::create_dir_all(project.join("src")).unwrap();
    fs::create_dir_all(project.join("build")).unwrap();
    fs::create_dir_all(project.join("target")).unwrap();

    fs::write(project.join("src/main.rs"), "TODO: implement").unwrap();
    fs::write(project.join("build/gen.rs"), "TODO: generated").unwrap();
    fs::write(project.join("target/debug.rs"), "TODO: debug").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search with multiple exclude patterns
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        exclude_patterns: vec!["**/build/**".to_string(), "**/target/**".to_string()],
        ..Default::default()
    };
    let results = engine.search("TODO", filter).unwrap();

    // Should only find results in src/
    assert_eq!(results.len(), 1);
    assert!(results[0].path.contains("src/main.rs"));
}

#[test]
fn test_exclude_generated_files() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::write(project.join("main.rs"), "fn pattern() {}").unwrap();
    fs::write(project.join("generated.rs"), "fn pattern() {}").unwrap();
    fs::write(project.join("codegen.rs"), "fn pattern() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Exclude generated files
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        exclude_patterns: vec!["**/generated.rs".to_string(), "**/codegen.rs".to_string()],
        ..Default::default()
    };
    let results = engine.search("pattern", filter).unwrap();

    // Should only find main.rs
    assert_eq!(results.len(), 1);
    assert!(results[0].path.contains("main.rs"));
}

#[test]
fn test_exclude_specific_directories() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::create_dir_all(project.join("src")).unwrap();
    fs::create_dir_all(project.join("tests")).unwrap();
    fs::create_dir_all(project.join("examples")).unwrap();

    fs::write(project.join("src/lib.rs"), "TODO").unwrap();
    fs::write(project.join("tests/test.rs"), "TODO").unwrap();
    fs::write(project.join("examples/demo.rs"), "TODO").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Exclude tests and examples
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        exclude_patterns: vec!["**/tests/**".to_string(), "**/examples/**".to_string()],
        ..Default::default()
    };
    let results = engine.search("TODO", filter).unwrap();

    // Should only find src/
    assert_eq!(results.len(), 1);
    assert!(results[0].path.contains("src/lib.rs"));
}

// ==================== Paths-Only Mode Tests ====================

#[test]
fn test_paths_only_deduplication() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create a file with multiple occurrences of the pattern
    let content = r#"
fn extract_pattern() {}
fn test_extract() {}
struct Pattern {}
fn another_extract_pattern() {}
"#;
    fs::write(project.join("main.rs"), content).unwrap();
    fs::write(project.join("other.rs"), "fn extract_pattern() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search with paths_only
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        paths_only: true,
        use_contains: true,  // "extract" in "extract_pattern" is not at word boundary
        ..Default::default()
    };
    let results = engine.search("extract", filter).unwrap();

    // Should return only 2 unique paths (main.rs and other.rs)
    assert_eq!(results.len(), 2);

    // Verify paths are unique
    let mut paths: Vec<_> = results.iter().map(|r| &r.path).collect();
    paths.sort();
    paths.dedup();
    assert_eq!(paths.len(), 2);
}

#[test]
fn test_paths_only_with_language_filter() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::write(project.join("main.rs"), "fn pattern() {}").unwrap();
    fs::write(project.join("app.ts"), "function pattern() {}").unwrap();
    fs::write(project.join("lib.rs"), "fn pattern() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search with paths_only + language filter
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        paths_only: true,
        language: Some(reflex::Language::Rust),
        ..Default::default()
    };
    let results = engine.search("pattern", filter).unwrap();

    // Should only return Rust file paths
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.path.ends_with(".rs")));
}

#[test]
fn test_paths_only_single_match_per_file() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // File with many matches
    let content = (0..50).map(|i| format!("fn test{}() {{}}", i)).collect::<Vec<_>>().join("\n");
    fs::write(project.join("many.rs"), content).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search with paths_only
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        paths_only: true,
        use_contains: true,  // "test" in "test0", "test1" is not at word boundary (followed by digit)
        ..Default::default()
    };
    let results = engine.search("test", filter).unwrap();

    // Should return only 1 path despite 50 matches
    assert_eq!(results.len(), 1);
    assert!(results[0].path.contains("many.rs"));
}

#[test]
fn test_paths_only_across_multiple_files() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::create_dir_all(project.join("src")).unwrap();
    fs::write(project.join("src/a.rs"), "TODO TODO TODO").unwrap();
    fs::write(project.join("src/b.rs"), "TODO").unwrap();
    fs::write(project.join("src/c.rs"), "TODO TODO").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search with paths_only
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        paths_only: true,
        ..Default::default()
    };
    let results = engine.search("TODO", filter).unwrap();

    // Should return 3 unique file paths
    assert_eq!(results.len(), 3);
}

// ==================== Combined Filter Tests ====================

#[test]
fn test_glob_and_exclude_together() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::create_dir_all(project.join("src/parsers")).unwrap();
    fs::create_dir_all(project.join("src/utils")).unwrap();

    fs::write(project.join("src/parsers/rust.rs"), "TODO: parse").unwrap();
    fs::write(project.join("src/parsers/generated.rs"), "TODO: generated").unwrap();
    fs::write(project.join("src/utils/helpers.rs"), "TODO: help").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search with glob (include src/) and exclude (exclude generated files)
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        glob_patterns: vec!["**/src/**".to_string()],
        exclude_patterns: vec!["**/generated.rs".to_string()],
        ..Default::default()
    };
    let results = engine.search("TODO", filter).unwrap();

    // Should find rust.rs and helpers.rs but not generated.rs
    assert_eq!(results.len(), 2);
    assert!(results.iter().any(|r| r.path.contains("rust.rs")));
    assert!(results.iter().any(|r| r.path.contains("helpers.rs")));
    assert!(!results.iter().any(|r| r.path.contains("generated.rs")));
}

#[test]
fn test_glob_exclude_and_language() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::create_dir_all(project.join("src")).unwrap();
    fs::create_dir_all(project.join("build")).unwrap();

    fs::write(project.join("src/main.rs"), "fn pattern() {}").unwrap();
    fs::write(project.join("src/app.ts"), "function pattern() {}").unwrap();
    fs::write(project.join("build/gen.rs"), "fn pattern() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search with glob + exclude + language
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        glob_patterns: vec!["**/src/**".to_string()],
        exclude_patterns: vec!["**/*.ts".to_string()],
        language: Some(reflex::Language::Rust),
        ..Default::default()
    };
    let results = engine.search("pattern", filter).unwrap();

    // Should only find src/main.rs
    assert_eq!(results.len(), 1);
    assert!(results[0].path.contains("src/main.rs"));
}

#[test]
fn test_glob_exclude_and_symbols() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::create_dir_all(project.join("src")).unwrap();
    fs::create_dir_all(project.join("tests")).unwrap();

    let src_content = r#"
fn extract_pattern() {}
fn test() {
    extract_pattern();
}
"#;
    fs::write(project.join("src/lib.rs"), src_content).unwrap();
    fs::write(project.join("tests/test.rs"), "fn extract_pattern() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search with glob + exclude + symbols
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        glob_patterns: vec!["**/src/**".to_string()],
        exclude_patterns: vec!["**/tests/**".to_string()],
        symbols_mode: true,
        use_contains: true,  // "extract" in "extract_pattern" is not at word boundary
        ..Default::default()
    };
    let results = engine.search("extract", filter).unwrap();

    // Should find only the function definition in src/, not the call site
    assert!(results.len() >= 1);
    assert!(results.iter().all(|r| r.path.contains("src/")));
    assert!(results.iter().all(|r| r.kind == SymbolKind::Function));
}

#[test]
fn test_all_filters_together() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::create_dir_all(project.join("src")).unwrap();
    fs::create_dir_all(project.join("tests")).unwrap();
    fs::create_dir_all(project.join("build")).unwrap();

    // Multiple occurrences in src/main.rs
    fs::write(project.join("src/main.rs"), "fn extract_pattern() {}\nfn other_extract() {}").unwrap();
    fs::write(project.join("tests/test.rs"), "fn extract_pattern() {}").unwrap();
    fs::write(project.join("build/gen.rs"), "fn extract_pattern() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search with ALL filters: glob + exclude + language + symbols + paths_only
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        glob_patterns: vec!["**/src/**".to_string()],
        exclude_patterns: vec!["**/tests/**".to_string(), "**/build/**".to_string()],
        language: Some(reflex::Language::Rust),
        symbols_mode: true,
        paths_only: true,
        use_contains: true,  // "extract" in "extract_pattern" is not at word boundary
        ..Default::default()
    };
    let results = engine.search("extract", filter).unwrap();

    // Should return only 1 unique path (src/main.rs) despite multiple matches
    assert_eq!(results.len(), 1);
    assert!(results[0].path.contains("src/main.rs"));
}

#[test]
fn test_glob_exclude_paths_with_limit() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::create_dir_all(project.join("src")).unwrap();

    // Create multiple files
    for i in 0..10 {
        fs::write(project.join(format!("src/file{}.rs", i)), "TODO: implement").unwrap();
    }

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search with glob + paths_only + limit
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        glob_patterns: vec!["**/src/**".to_string()],
        paths_only: true,
        limit: Some(5),
        ..Default::default()
    };
    let results = engine.search("TODO", filter).unwrap();

    // Should limit to 5 unique paths
    assert_eq!(results.len(), 5);
    assert!(results.iter().all(|r| r.path.contains("src/")));
}

// ==================== AST Query Tests ====================

#[test]
fn test_ast_query_basic_rust() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create Rust files with functions
    fs::write(
        project.join("main.rs"),
        "fn hello() {}\nfn world() {}\nstruct Point {}"
    ).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // AST query for all functions
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        language: Some(reflex::Language::Rust),
        use_ast: true,
        ..Default::default()
    };
    let results = engine.search_ast_all_files("(function_item) @fn", filter).unwrap();

    // Should find both functions but not the struct
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.kind == SymbolKind::Function));
    assert!(results.iter().any(|r| r.symbol.as_deref() == Some("hello")));
    assert!(results.iter().any(|r| r.symbol.as_deref() == Some("world")));
}

#[test]
fn test_ast_query_with_glob_pattern() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create nested directory structure
    fs::create_dir_all(project.join("src")).unwrap();
    fs::create_dir_all(project.join("tests")).unwrap();

    fs::write(project.join("src/lib.rs"), "fn src_func() {}").unwrap();
    fs::write(project.join("tests/test.rs"), "fn test_func() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // AST query with glob to limit to src/ only
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        language: Some(reflex::Language::Rust),
        use_ast: true,
        glob_patterns: vec!["**/src/**/*.rs".to_string()],
        ..Default::default()
    };
    let results = engine.search_ast_all_files("(function_item) @fn", filter).unwrap();

    // Should only find function in src/, not tests/
    assert_eq!(results.len(), 1);
    assert!(results[0].path.contains("src/lib.rs"));
    assert_eq!(results[0].symbol.as_deref(), Some("src_func"));
}

#[test]
fn test_ast_query_with_exclude_pattern() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::create_dir_all(project.join("src")).unwrap();
    fs::create_dir_all(project.join("build")).unwrap();

    fs::write(project.join("src/main.rs"), "fn main_func() {}").unwrap();
    fs::write(project.join("build/generated.rs"), "fn gen_func() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // AST query with exclude to skip build/
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        language: Some(reflex::Language::Rust),
        use_ast: true,
        exclude_patterns: vec!["**/build/**".to_string()],
        ..Default::default()
    };
    let results = engine.search_ast_all_files("(function_item) @fn", filter).unwrap();

    // Should only find function in src/
    assert_eq!(results.len(), 1);
    assert!(results[0].path.contains("src/main.rs"));
    assert!(!results.iter().any(|r| r.path.contains("build/")));
}

#[test]
fn test_ast_query_multiple_languages() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::write(project.join("main.rs"), "fn rust_func() {}").unwrap();
    fs::write(project.join("app.py"), "def python_func(): pass").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // AST query for Rust functions
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter_rust = QueryFilter {
        language: Some(reflex::Language::Rust),
        use_ast: true,
        ..Default::default()
    };
    let results_rust = engine.search_ast_all_files("(function_item) @fn", filter_rust).unwrap();
    assert_eq!(results_rust.len(), 1);
    assert_eq!(results_rust[0].symbol.as_deref(), Some("rust_func"));

    // AST query for Python functions
    let filter_python = QueryFilter {
        language: Some(reflex::Language::Python),
        use_ast: true,
        ..Default::default()
    };
    let results_python = engine.search_ast_all_files("(function_definition) @fn", filter_python).unwrap();
    assert_eq!(results_python.len(), 1);
    assert!(results_python[0].preview.contains("python_func"));
}

#[test]
fn test_ast_query_requires_language() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::write(project.join("main.rs"), "fn test() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // AST query without language should fail
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        language: None,  // Missing language!
        use_ast: true,
        ..Default::default()
    };
    let result = engine.search_ast_all_files("(function_item) @fn", filter);

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Language must be specified"));
}

#[test]
fn test_ast_query_structs() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::write(
        project.join("main.rs"),
        "struct Point {}\nstruct Line {}\nfn test() {}"
    ).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // AST query for structs
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        language: Some(reflex::Language::Rust),
        use_ast: true,
        ..Default::default()
    };
    let results = engine.search_ast_all_files("(struct_item) @struct", filter).unwrap();

    // Should find both structs but not the function
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.kind == SymbolKind::Struct));
    assert!(results.iter().any(|r| r.symbol.as_deref() == Some("Point")));
    assert!(results.iter().any(|r| r.symbol.as_deref() == Some("Line")));
}

#[test]
fn test_ast_query_glob_and_exclude_together() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::create_dir_all(project.join("src/parsers")).unwrap();
    fs::create_dir_all(project.join("src/utils")).unwrap();
    fs::create_dir_all(project.join("tests")).unwrap();

    fs::write(project.join("src/parsers/rust.rs"), "fn parse_rust() {}").unwrap();
    fs::write(project.join("src/parsers/generated.rs"), "fn gen_parse() {}").unwrap();
    fs::write(project.join("src/utils/helpers.rs"), "fn helper() {}").unwrap();
    fs::write(project.join("tests/test.rs"), "fn test() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // AST query with glob (src/) and exclude (generated.rs)
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        language: Some(reflex::Language::Rust),
        use_ast: true,
        glob_patterns: vec!["**/src/**/*.rs".to_string()],
        exclude_patterns: vec!["**/generated.rs".to_string()],
        ..Default::default()
    };
    let results = engine.search_ast_all_files("(function_item) @fn", filter).unwrap();

    // Should find parse_rust and helper but not gen_parse or test
    assert_eq!(results.len(), 2);
    assert!(results.iter().any(|r| r.symbol.as_deref() == Some("parse_rust")));
    assert!(results.iter().any(|r| r.symbol.as_deref() == Some("helper")));
    assert!(!results.iter().any(|r| r.symbol.as_deref() == Some("gen_parse")));
    assert!(!results.iter().any(|r| r.symbol.as_deref() == Some("test")));
}

#[test]
fn test_ast_query_python_classes() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::write(
        project.join("app.py"),
        "class User:\n    pass\n\nclass Product:\n    pass\n\ndef helper():\n    pass"
    ).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // AST query for Python classes
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        language: Some(reflex::Language::Python),
        use_ast: true,
        ..Default::default()
    };
    let results = engine.search_ast_all_files("(class_definition) @class", filter).unwrap();

    // Should find both classes but not the function
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.kind == SymbolKind::Class));
    assert!(results.iter().any(|r| r.preview.contains("User")));
    assert!(results.iter().any(|r| r.preview.contains("Product")));
}

#[test]
fn test_ast_query_invalid_pattern() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::write(project.join("main.rs"), "fn test() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // AST query with invalid S-expression (missing closing paren)
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        language: Some(reflex::Language::Rust),
        use_ast: true,
        ..Default::default()
    };
    let result = engine.search_ast_all_files("(function_item @fn", filter);

    // Should fail with parse error
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Invalid AST query pattern") || error_msg.contains("error"));
}

// ==================== Keyword Detection Tests ====================
//
// These tests verify that keyword queries (searching for language keywords like
// "struct", "class", "function") properly trigger keyword mode and return accurate
// symbol counts across multiple languages and edge cases.

#[test]
fn test_keyword_rust_struct() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Copy Rust structs corpus file
    let corpus_content = include_str!("corpus/rust/structs.rs");
    fs::write(project.join("structs.rs"), corpus_content).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search for "struct" keyword with --symbols
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        language: Some(reflex::Language::Rust),
        ..Default::default()
    };
    let results = engine.search("struct", filter).unwrap();

    // Should find 11 structs (per corpus file documentation)
    assert_eq!(results.len(), 11);
    assert!(results.iter().all(|r| r.kind == SymbolKind::Struct));
}

#[test]
fn test_keyword_php_class() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Copy PHP classes corpus file
    let corpus_content = include_str!("corpus/php/classes.php");
    fs::write(project.join("classes.php"), corpus_content).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search for "class" keyword with --symbols
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        language: Some(reflex::Language::PHP),
        ..Default::default()
    };
    let results = engine.search("class", filter).unwrap();

    // Should find 9 classes (Point, Person, Employee, Shape, Rectangle, Utils, Counter, AdminUser, ComplexUser)
    assert_eq!(results.len(), 9);
    assert!(results.iter().all(|r| r.kind == SymbolKind::Class));
}

#[test]
fn test_keyword_typescript_interface() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Copy TypeScript interfaces corpus file
    let corpus_content = include_str!("corpus/typescript/interfaces.ts");
    fs::write(project.join("interfaces.ts"), corpus_content).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search for "interface" keyword with --symbols
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        language: Some(reflex::Language::TypeScript),
        ..Default::default()
    };
    let results = engine.search("interface", filter).unwrap();

    // Should find 11 interfaces (per corpus file documentation)
    assert_eq!(results.len(), 11);
    assert!(results.iter().all(|r| r.kind == SymbolKind::Interface));
}

#[test]
fn test_keyword_without_lang_flag() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create mixed-language files with classes
    let php_content = include_str!("corpus/edge_cases/mixed_languages/php_classes.php");
    let ts_content = include_str!("corpus/edge_cases/mixed_languages/ts_classes.ts");
    let js_content = include_str!("corpus/edge_cases/mixed_languages/js_classes.js");

    fs::write(project.join("test.php"), php_content).unwrap();
    fs::write(project.join("test.ts"), ts_content).unwrap();
    fs::write(project.join("test.js"), js_content).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search for "class" without --lang flag
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        language: None,  // No language filter!
        ..Default::default()
    };
    let results = engine.search("class", filter).unwrap();

    // Should find classes from ALL languages (2 PHP + 2 TS + 2 JS = 6)
    assert_eq!(results.len(), 6);
    assert!(results.iter().all(|r| r.kind == SymbolKind::Class));
}

#[test]
fn test_keyword_with_lang_filter() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create mixed-language files with classes
    let php_content = include_str!("corpus/edge_cases/mixed_languages/php_classes.php");
    let ts_content = include_str!("corpus/edge_cases/mixed_languages/ts_classes.ts");

    fs::write(project.join("test.php"), php_content).unwrap();
    fs::write(project.join("test.ts"), ts_content).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search for "class" WITH --lang php
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        language: Some(reflex::Language::PHP),
        ..Default::default()
    };
    let results = engine.search("class", filter).unwrap();

    // Should find only PHP classes (2)
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.kind == SymbolKind::Class));
    assert!(results.iter().all(|r| r.path.ends_with(".php")));
}

#[test]
fn test_keywords_in_strings_ignored() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Use keywords_in_strings corpus file
    let corpus_content = include_str!("corpus/edge_cases/keywords_in_strings.rs");
    fs::write(project.join("test.rs"), corpus_content).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search for "struct" keyword - should find 0 structs (all in strings/comments)
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        language: Some(reflex::Language::Rust),
        ..Default::default()
    };
    let results = engine.search("struct", filter.clone()).unwrap();

    // Should find 0 structs (keywords in strings/comments don't count)
    assert_eq!(results.len(), 0);

    // Search for "fn" - should find 5 actual functions
    let results_fn = engine.search("fn", filter).unwrap();
    assert_eq!(results_fn.len(), 5);
    assert!(results_fn.iter().all(|r| r.kind == SymbolKind::Function));
}

#[test]
fn test_exact_counts_rust_corpus() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Copy Rust structs corpus file (documented: 11 structs)
    let corpus_content = include_str!("corpus/rust/structs.rs");
    fs::write(project.join("structs.rs"), corpus_content).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Verify exact count
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        language: Some(reflex::Language::Rust),
        ..Default::default()
    };
    let results = engine.search("struct", filter).unwrap();

    // Per corpus file: 11 structs expected
    assert_eq!(results.len(), 11, "Expected 11 structs in rust/structs.rs corpus");
    assert!(results.iter().all(|r| r.kind == SymbolKind::Struct));
}

#[test]
fn test_keyword_plus_glob() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::create_dir_all(project.join("src")).unwrap();
    fs::create_dir_all(project.join("tests")).unwrap();

    // Create files in different directories
    fs::write(project.join("src/lib.rs"), "struct SrcStruct {}").unwrap();
    fs::write(project.join("tests/test.rs"), "struct TestStruct {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search for "struct" with glob filter
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        glob_patterns: vec!["**/src/**".to_string()],
        ..Default::default()
    };
    let results = engine.search("struct", filter).unwrap();

    // Should find only struct in src/
    assert_eq!(results.len(), 1);
    assert!(results[0].path.contains("src/"));
    assert!(results[0].symbol.as_deref() == Some("SrcStruct"));
}

#[test]
fn test_keyword_plus_exclude() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::create_dir_all(project.join("src")).unwrap();
    fs::create_dir_all(project.join("target")).unwrap();

    // Create files
    fs::write(project.join("src/lib.rs"), "struct SrcStruct {}").unwrap();
    fs::write(project.join("target/debug.rs"), "struct TargetStruct {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search for "struct" with exclude filter
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        exclude_patterns: vec!["**/target/**".to_string()],
        ..Default::default()
    };
    let results = engine.search("struct", filter).unwrap();

    // Should exclude target/ directory
    assert_eq!(results.len(), 1);
    assert!(results[0].path.contains("src/"));
    assert!(!results.iter().any(|r| r.path.contains("target/")));
}

// ==================== Keyword Mode Triggering Tests ====================
//
// These tests verify when keyword mode should and should not be triggered

#[test]
fn test_non_keyword_search_normal_mode() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create file with struct and identifier containing "struct"
    fs::write(
        project.join("test.rs"),
        "struct Point {}\nfn my_struct_builder() {}\nlet my_struct = Point {};"
    ).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search for "my_struct" (NOT a keyword)
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        use_contains: true,  // "my_struct" in identifiers
        ..Default::default()
    };
    let results = engine.search("my_struct", filter).unwrap();

    // Should find function and variable, not trigger keyword mode
    // (keyword mode would only return structs)
    assert!(results.len() >= 1);
    assert!(results.iter().any(|r| r.kind == SymbolKind::Function));
}

#[test]
fn test_uppercase_keyword_not_trigger() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create file with uppercase and lowercase structs
    fs::write(
        project.join("test.rs"),
        "struct lowercase {}\nstruct UPPERCASE {}"
    ).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search for "STRUCT" (uppercase keyword - NOT a keyword)
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        ..Default::default()
    };
    let results = engine.search("STRUCT", filter).unwrap();

    // Should do full-text search, not keyword mode
    // Will find the identifier "UPPERCASE" but not trigger "list all structs" behavior
    // In this case, it won't match anything because "STRUCT" is not in the code
    // (only "struct" lowercase and "UPPERCASE" identifier exist)
    assert_eq!(results.len(), 0);
}

#[test]
fn test_explicit_kind_overrides_keyword_inference() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create file with functions and structs
    fs::write(
        project.join("test.rs"),
        "fn greet() {}\nfn process() {}\nstruct Point {}"
    ).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search for "struct" but with explicit --kind Function
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        kind: Some(SymbolKind::Function),  // Explicit kind override
        ..Default::default()
    };
    let results = engine.search("struct", filter).unwrap();

    // Should use explicit kind (Function), not infer from keyword
    // Keyword mode + explicit kind = list all symbols of explicit kind
    // So "struct" keyword + kind=Function = list all functions
    assert_eq!(results.len(), 2);  // Both functions (greet and process)
    assert!(results.iter().all(|r| r.kind == SymbolKind::Function));
}

#[test]
fn test_partial_keyword_match_normal_search() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Use partial keywords corpus
    let corpus_content = include_str!("corpus/edge_cases/keywords_partial.rs");
    fs::write(project.join("test.rs"), corpus_content).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search for "struct_builder" (contains "struct" but not exact keyword)
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        use_contains: true,  // "struct_builder" contains "struct"
        ..Default::default()
    };
    let results = engine.search("struct_builder", filter).unwrap();

    // Should find function named struct_builder, NOT trigger keyword mode
    assert!(results.len() >= 1);
    assert!(results.iter().any(|r| r.kind == SymbolKind::Function));
    assert!(results.iter().any(|r| r.symbol.as_ref().map_or(false, |s| s.contains("struct_builder"))));
}

#[test]
fn test_keyword_exact_match_required() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create file with "fn" in various contexts
    fs::write(
        project.join("test.rs"),
        "fn test() {}\nfn another() {}\nlet fn_pointer = test;"
    ).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search for exact keyword "fn"
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        language: Some(reflex::Language::Rust),
        ..Default::default()
    };
    let results = engine.search("fn", filter).unwrap();

    // Should trigger keyword mode and find all functions
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.kind == SymbolKind::Function));
}

// ==================== Additional Edge Case and Filter Tests ====================

#[test]
fn test_keyword_empty_results() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create file with NO structs (only functions)
    fs::write(
        project.join("test.rs"),
        "fn test() {}\nfn another() {}"
    ).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search for "struct" keyword when no structs exist
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        language: Some(reflex::Language::Rust),
        ..Default::default()
    };
    let results = engine.search("struct", filter).unwrap();

    // Should return 0 results
    assert_eq!(results.len(), 0);
}

#[test]
fn test_multi_language_exact_counts() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Copy multiple language corpus files
    let python_classes = include_str!("corpus/python/classes.py");
    let go_structs = include_str!("corpus/go/structs.go");
    let java_classes = include_str!("corpus/java/classes.java");

    fs::write(project.join("test.py"), python_classes).unwrap();
    fs::write(project.join("test.go"), go_structs).unwrap();
    fs::write(project.join("test.java"), java_classes).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search for "class" across all languages
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        ..Default::default()
    };
    let results = engine.search("class", filter).unwrap();

    // Should find Python (13 including nested/metaclass) + Java (11 including nested/anonymous) = 24 classes
    // (Go has structs, not classes)
    assert_eq!(results.len(), 24);
    assert!(results.iter().all(|r| r.kind == SymbolKind::Class));

    // Verify language distribution
    let python_count = results.iter().filter(|r| r.path.ends_with(".py")).count();
    let java_count = results.iter().filter(|r| r.path.ends_with(".java")).count();
    assert_eq!(python_count, 13);  // Includes nested, metaclass, mixin
    assert_eq!(java_count, 11);    // Includes nested, inner, anonymous
}

#[test]
fn test_keyword_with_paths_only() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create multiple files with structs
    fs::create_dir_all(project.join("src")).unwrap();
    fs::write(project.join("src/models.rs"), "struct User {}\nstruct Post {}").unwrap();
    fs::write(project.join("src/types.rs"), "struct Config {}\nstruct State {}").unwrap();
    fs::write(project.join("src/lib.rs"), "struct App {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search for "struct" with paths_only
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        paths_only: true,
        ..Default::default()
    };
    let results = engine.search("struct", filter).unwrap();

    // Should return 3 unique paths (one per file)
    assert_eq!(results.len(), 3);

    // Verify paths are unique
    let paths: Vec<_> = results.iter().map(|r| &r.path).collect();
    let unique_paths: std::collections::HashSet<_> = paths.iter().collect();
    assert_eq!(unique_paths.len(), 3);
}

#[test]
fn test_keyword_with_limit() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Copy Rust structs corpus (11 structs)
    let corpus_content = include_str!("corpus/rust/structs.rs");
    fs::write(project.join("test.rs"), corpus_content).unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Search for "struct" with limit=5
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        limit: Some(5),
        ..Default::default()
    };
    let results = engine.search("struct", filter).unwrap();

    // Should return only 5 results (limited from 11 total)
    assert_eq!(results.len(), 5);
    assert!(results.iter().all(|r| r.kind == SymbolKind::Struct));
}

#[test]
fn test_keyword_multi_language_function_synonyms() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create files with function definitions in multiple languages
    fs::write(project.join("test.rs"), "fn rust_func() {}").unwrap();
    fs::write(project.join("test.py"), "def python_func(): pass").unwrap();
    fs::write(project.join("test.js"), "function js_func() {}").unwrap();
    fs::write(project.join("test.go"), "func go_func() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Test Rust "fn" keyword
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter_rust = QueryFilter {
        symbols_mode: true,
        language: Some(reflex::Language::Rust),
        ..Default::default()
    };
    let results_rust = engine.search("fn", filter_rust).unwrap();
    assert_eq!(results_rust.len(), 1);
    assert!(results_rust[0].path.ends_with(".rs"));

    // Test Python "def" keyword
    let filter_python = QueryFilter {
        symbols_mode: true,
        language: Some(reflex::Language::Python),
        ..Default::default()
    };
    let results_python = engine.search("def", filter_python).unwrap();
    assert_eq!(results_python.len(), 1);
    assert!(results_python[0].path.ends_with(".py"));

    // Test JavaScript "function" keyword
    let filter_js = QueryFilter {
        symbols_mode: true,
        language: Some(reflex::Language::JavaScript),
        ..Default::default()
    };
    let results_js = engine.search("function", filter_js).unwrap();
    assert_eq!(results_js.len(), 1);
    assert!(results_js[0].path.ends_with(".js"));

    // Test Go "func" keyword
    let filter_go = QueryFilter {
        symbols_mode: true,
        language: Some(reflex::Language::Go),
        ..Default::default()
    };
    let results_go = engine.search("func", filter_go).unwrap();
    assert_eq!(results_go.len(), 1);
    assert!(results_go[0].path.ends_with(".go"));
}

// ==================== Broad Query Detection Tests ====================
//
// These tests verify that broad query detection prevents expensive queries
// and allows bypassing with the --force flag.

#[test]
fn test_broad_query_short_pattern_blocked() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create a file
    fs::write(project.join("main.rs"), "fn test() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Query with short pattern (< 3 chars)
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        force: false,  // Not forcing
        ..Default::default()
    };
    let result = engine.search("ab", filter);

    // Should fail with broad query detection error
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Query too broad"));
    assert!(error_msg.contains("too short"));
    assert!(error_msg.contains("--force"));
}

#[test]
fn test_broad_query_short_pattern_bypass_with_force() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create a file with matching pattern
    fs::write(project.join("main.rs"), "let abc = 42;").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Query with short pattern BUT with force=true
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        force: true,  // Force bypass
        ..Default::default()
    };
    let result = engine.search("ab", filter);

    // Should succeed (bypass the check), even though trigram index can't find < 3 char patterns
    // The query won't error out with "Query too broad", it will just return 0 results
    assert!(result.is_ok());
    // Note: results may be empty because trigram index requires >= 3 chars
    // The important thing is that it didn't fail with a "Query too broad" error
}

#[test]
fn test_broad_query_ast_without_glob_blocked() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create 100+ files to trigger the threshold
    for i in 0..105 {
        fs::write(project.join(format!("test{}.rs", i)), "fn func() {}").unwrap();
    }

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // AST query without glob patterns
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        language: Some(reflex::Language::Rust),
        use_ast: true,
        force: false,  // Not forcing
        glob_patterns: vec![],  // No glob!
        ..Default::default()
    };
    let result = engine.search_ast_all_files("(function_item) @fn", filter);

    // Should fail with broad query detection error
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Query too broad"));
    assert!(error_msg.contains("AST query without --glob"));
    assert!(error_msg.contains("--force"));
}

#[test]
fn test_broad_query_ast_without_glob_bypass_with_force() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create files
    fs::write(project.join("test.rs"), "fn func() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // AST query without glob BUT with force=true
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        language: Some(reflex::Language::Rust),
        use_ast: true,
        force: true,  // Force bypass
        glob_patterns: vec![],  // No glob, but forced
        ..Default::default()
    };
    let result = engine.search_ast_all_files("(function_item) @fn", filter);

    // Should succeed
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_broad_query_ast_with_glob_allowed() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create files in different directories
    fs::create_dir_all(project.join("src")).unwrap();
    fs::create_dir_all(project.join("tests")).unwrap();
    fs::write(project.join("src/main.rs"), "fn main() {}").unwrap();
    fs::write(project.join("tests/test.rs"), "fn test() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // AST query WITH glob patterns (should be allowed)
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        language: Some(reflex::Language::Rust),
        use_ast: true,
        force: false,  // Not forcing, but has glob
        glob_patterns: vec!["**/src/**/*.rs".to_string()],  // Glob provided
        ..Default::default()
    };
    let result = engine.search_ast_all_files("(function_item) @fn", filter);

    // Should succeed (glob pattern restricts scope)
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].path.contains("src/main.rs"));
}

#[test]
fn test_broad_query_long_pattern_allowed() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create file
    fs::write(project.join("main.rs"), "fn extract_symbols() {}").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Query with long pattern (>= 3 chars)
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        force: false,  // Not forcing
        use_contains: true,  // "extract" in "extract_symbols"
        ..Default::default()
    };
    let result = engine.search("extract", filter);

    // Should succeed (pattern is long enough)
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results.len() >= 1);
}

#[test]
fn test_broad_query_regex_short_pattern_allowed() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create file
    fs::write(project.join("main.rs"), "let ab = 42; let abc = 43;").unwrap();

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Regex query with short pattern (regex bypass short pattern check)
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        use_regex: true,
        force: false,  // Not forcing
        ..Default::default()
    };
    let result = engine.search("ab", filter);

    // Should succeed (regex mode bypasses short pattern check)
    assert!(result.is_ok());
}

#[test]
fn test_broad_query_error_message_short_pattern() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::write(project.join("main.rs"), "fn test() {}").unwrap();

    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        force: false,
        ..Default::default()
    };
    let result = engine.search("ab", filter);

    // Verify error message content
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Query too broad"));
    assert!(error_msg.contains("too short"));
    assert!(error_msg.contains("2 characters"));
    assert!(error_msg.contains("Use a longer, more specific pattern"));
    assert!(error_msg.contains("Add a language filter: --lang"));
    assert!(error_msg.contains("Add a file path filter: --file"));
    assert!(error_msg.contains("Use --force to bypass this check"));
    assert!(error_msg.contains("rfx query \"ab\" --force"));
}

#[test]
fn test_broad_query_error_message_ast_without_glob() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create 100+ files to trigger the threshold
    for i in 0..105 {
        fs::write(project.join(format!("test{}.rs", i)), "fn test() {}").unwrap();
    }

    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        language: Some(reflex::Language::Rust),
        use_ast: true,
        force: false,
        glob_patterns: vec![],
        ..Default::default()
    };
    let result = engine.search_ast_all_files("(function_item) @fn", filter);

    // Verify error message content
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Query too broad"));
    assert!(error_msg.contains("AST query without --glob"));
    assert!(error_msg.contains("scan the ENTIRE codebase"));
    assert!(error_msg.contains("Add --glob to restrict AST query"));
    assert!(error_msg.contains("Use --symbols instead"));
    assert!(error_msg.contains("Use --force to bypass this check"));
}

#[test]
fn test_broad_query_with_exclude_still_requires_glob_for_ast() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create 100+ files to trigger the threshold
    for i in 0..105 {
        fs::write(project.join(format!("test{}.rs", i)), "fn func() {}").unwrap();
    }

    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // AST query with exclude but no glob (should still be blocked)
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        language: Some(reflex::Language::Rust),
        use_ast: true,
        force: false,
        glob_patterns: vec![],  // No glob!
        exclude_patterns: vec!["**/target/**".to_string()],  // Has exclude, but not enough
        ..Default::default()
    };
    let result = engine.search_ast_all_files("(function_item) @fn", filter);

    // Should still be blocked (exclude alone is not sufficient)
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Query too broad"));
    assert!(error_msg.contains("AST query without --glob"));
}

#[test]
fn test_broad_query_large_index_short_pattern_early_check() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create 150 files to trigger the test threshold (100)
    // Using test override instead of creating 20K+ files for faster tests
    for i in 0..150 {
        fs::write(project.join(format!("file{}.rs", i)), "fn get_data() {}").unwrap();
    }

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Query with short pattern (3 chars - valid trigram but < 4 char threshold)
    // This should be blocked by the EARLY check (before trigram search)
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        force: false,  // Not forcing
        test_large_index_threshold: Some(100),  // Override for testing (150 > 100)
        ..Default::default()
    };
    let result = engine.search("get", filter);

    // Should fail with early broad query detection error
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Query too broad"));
    assert!(error_msg.contains("large index"));
    assert!(error_msg.contains("150 files"));  // Should mention the file count
    assert!(error_msg.contains("3 characters"));  // Should mention pattern length
    assert!(error_msg.contains("--force"));
}

#[test]
fn test_broad_query_large_index_force_bypass() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create 150 files to trigger the test threshold (100)
    // Using test override instead of creating 20K+ files for faster tests
    for i in 0..150 {
        if i == 0 {
            // First file contains the pattern
            fs::write(project.join(format!("file{}.rs", i)), "fn get_data() {}").unwrap();
        } else {
            // Other files don't contain it
            fs::write(project.join(format!("file{}.rs", i)), "fn process() {}").unwrap();
        }
    }

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Query with short pattern BUT with force=true
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        force: true,  // Force bypass
        use_contains: true,  // Enable substring matching (default is word-boundary)
        test_large_index_threshold: Some(100),  // Override for testing (150 > 100)
        ..Default::default()
    };
    let result = engine.search("get", filter);

    // Should succeed (bypass the early check)
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results.len() >= 1);  // Should find "get" in "get_data"
}

#[test]
fn test_broad_query_small_index_short_pattern_allowed() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create only 100 files (well below LARGE_INDEX_THRESHOLD)
    for i in 0..100 {
        fs::write(project.join(format!("file{}.rs", i)), "fn get_data() {}").unwrap();
    }

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Query with same short pattern (3 chars)
    // Should be ALLOWED because index is small
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        force: false,  // Not forcing
        use_contains: true,  // Enable substring matching (default is word-boundary)
        ..Default::default()
    };
    let result = engine.search("get", filter);

    // Should succeed (small index allows short patterns)
    assert!(result.is_ok());
    let results = result.unwrap();
    assert!(results.len() >= 1);
}

#[test]
fn test_broad_query_language_filter_applied_before_check() {
    // Regression test for: Language filter should be applied BEFORE broad query check
    // Previously, the broad query check used the unfiltered candidate count,
    // causing queries like `rfx query "index" --lang rust --symbols` to fail
    // even when the language-filtered result set was small.

    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create many C files containing "index" (simulate Linux kernel scenario)
    // Using test override: 120 C files (above threshold of 100)
    for i in 0..120 {
        fs::write(
            project.join(format!("kernel{}.c", i)),
            "int index_lookup(void) { return 0; }"
        ).unwrap();
    }

    // Create a smaller number of Rust files containing "index"
    // 80 Rust files (below threshold of 100 when filtered)
    for i in 0..80 {
        fs::write(
            project.join(format!("rust{}.rs", i)),
            "fn index_lookup() -> usize { 0 }"
        ).unwrap();
    }

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Query with --lang rust --symbols
    // The language filter should reduce 200 total files to 80 Rust files
    // BEFORE the broad query check (80 < 100 threshold), so this should succeed
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        language: Some(reflex::Language::Rust),
        use_contains: true,  // Match "index" within "index_lookup"
        force: false,  // Should succeed WITHOUT force
        test_large_index_threshold: Some(100),  // Override for testing (200 total > 100, but 80 Rust < 100)
        ..Default::default()
    };
    let result = engine.search("index", filter);

    // Should succeed - language filter is applied BEFORE broad query check
    assert!(result.is_ok(), "Query should succeed when language filter reduces candidate set below threshold");
    let results = result.unwrap();

    // The key assertion is that the query SUCCEEDED without --force
    // This proves language filter was applied BEFORE broad query check
    // (Without the fix, this would error with "Query too broad - 200 files")
    assert!(results.len() > 0, "Should find at least one symbol matching 'index' in Rust files");
}

#[test]
fn test_broad_query_glob_filter_applied_before_check() {
    // Regression test for: Glob filter should be applied BEFORE broad query check
    // Previously, the broad query check used the unfiltered candidate count,
    // causing queries like `rfx query "index" --symbols --glob "src/**/*.rs"` to fail
    // even when the glob-filtered result set was small.

    let temp = TempDir::new().unwrap();
    let project = temp.path();

    // Create many files in different directories containing "index"
    // Using test override: 120 files in build/ (above threshold of 100)

    // Create files in build/ directory (should be excluded by glob)
    fs::create_dir(project.join("build")).unwrap();
    for i in 0..120 {
        fs::write(
            project.join(format!("build/file{}.rs", i)),
            "fn index_lookup() -> usize { 0 }"
        ).unwrap();
    }

    // Create a smaller number of files in src/ directory (should be included by glob)
    // 80 files (below threshold of 100 when filtered)
    fs::create_dir(project.join("src")).unwrap();
    for i in 0..80 {
        fs::write(
            project.join(format!("src/file{}.rs", i)),
            "fn index_lookup() -> usize { 0 }"
        ).unwrap();
    }

    // Index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    // Query with --glob "src/**/*.rs" --symbols
    // The glob filter should reduce 200 total files to 80 src files
    // BEFORE the broad query check (80 < 100 threshold), so this should succeed
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let filter = QueryFilter {
        symbols_mode: true,
        glob_patterns: vec!["src/**/*.rs".to_string()],
        use_contains: true,  // Match "index" within "index_lookup"
        force: false,  // Should succeed WITHOUT force
        test_large_index_threshold: Some(100),  // Override for testing (200 total > 100, but 80 in src/ < 100)
        ..Default::default()
    };
    let result = engine.search("index", filter);

    // The key assertion is that the query SUCCEEDED without --force
    // This proves glob filter was applied BEFORE broad query check
    // (Without the fix, this would error with "Query too broad - 200 files")
    assert!(result.is_ok(), "Query should succeed when glob filter reduces candidate set below threshold. \
                             Without the fix, this would error: 'Query too broad - would be expensive to execute'");
}

// ==================== Partial-Write Recovery Tests ====================
// Audit finding A1 (REF-6): verify the cache survives a simulated disk-full mid-write.
//
// trigrams.bin and content.bin use a non-atomic write pattern (truncate + stream).
// A disk-full condition can leave either file corrupt. The recovery contract is:
// re-running `rfx index` detects corruption via magic-byte validation and rebuilds
// from scratch. Tests below simulate this with set_len(2) on the binary files.

#[test]
fn test_partial_write_recovery_trigrams_bin() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::write(project.join("hello.rs"), "fn greet() -> &'static str { \"hello\" }").unwrap();
    fs::write(project.join("lib.rs"), "pub fn add(a: i32, b: i32) -> i32 { a + b }").unwrap();

    // Build a valid initial index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    let cache = CacheManager::new(project);
    assert!(cache.validate().is_ok(), "Initial cache should be valid");

    // Simulate disk-full mid-write: truncate trigrams.bin below the 4-byte magic header
    let trigrams_path = project.join(".reflex/trigrams.bin");
    {
        let file = std::fs::OpenOptions::new().write(true).open(&trigrams_path).unwrap();
        file.set_len(2).unwrap();
    }

    // validate() must detect the corruption
    let cache = CacheManager::new(project);
    let result = cache.validate();
    assert!(result.is_err(), "Truncated trigrams.bin must fail validate()");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("trigrams.bin"),
        "Error must name the corrupted file, got: {err_msg}"
    );

    // Recovery: re-index must detect the corrupt file via magic-byte check and rebuild cleanly
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    let result = indexer.index(project, false);
    assert!(
        result.is_ok(),
        "Re-indexing after partial trigrams.bin write must succeed, got: {:?}",
        result.err()
    );

    // Rebuilt cache must pass full validation
    let cache = CacheManager::new(project);
    assert!(cache.validate().is_ok(), "Rebuilt cache must pass validate()");

    // Rebuilt cache must return correct query results
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let results = engine.search("greet", QueryFilter::default()).unwrap();
    assert!(!results.is_empty(), "Should find 'greet' in rebuilt cache");
}

#[test]
fn test_partial_write_recovery_content_bin() {
    let temp = TempDir::new().unwrap();
    let project = temp.path();

    fs::write(project.join("main.rs"), "fn main() { println!(\"world\"); }").unwrap();
    fs::write(project.join("util.rs"), "pub fn helper() {}").unwrap();

    // Build a valid initial index
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(project, false).unwrap();

    let cache = CacheManager::new(project);
    assert!(cache.validate().is_ok(), "Initial cache should be valid");

    // Simulate disk-full mid-write: truncate content.bin below the 4-byte magic header
    let content_path = project.join(".reflex/content.bin");
    {
        let file = std::fs::OpenOptions::new().write(true).open(&content_path).unwrap();
        file.set_len(2).unwrap();
    }

    // validate() must detect the corruption
    let cache = CacheManager::new(project);
    let result = cache.validate();
    assert!(result.is_err(), "Truncated content.bin must fail validate()");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("content.bin"),
        "Error must name the corrupted file, got: {err_msg}"
    );

    // Recovery: re-index must detect the invalid content.bin and rebuild cleanly
    let cache = CacheManager::new(project);
    let indexer = Indexer::new(cache, IndexConfig::default());
    let result = indexer.index(project, false);
    assert!(
        result.is_ok(),
        "Re-indexing after partial content.bin write must succeed, got: {:?}",
        result.err()
    );

    // Rebuilt cache must pass full validation
    let cache = CacheManager::new(project);
    assert!(cache.validate().is_ok(), "Rebuilt cache must pass validate()");

    // Rebuilt cache must return correct query results
    let cache = CacheManager::new(project);
    let engine = QueryEngine::new(cache);
    let results = engine.search("world", QueryFilter::default()).unwrap();
    assert!(!results.is_empty(), "Should find 'world' in rebuilt cache");
}
