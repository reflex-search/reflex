//! Build-time schema hash computation for automatic cache invalidation
//!
//! This build script computes a hash of all cache-critical source files at compile time.
//! If any of these files change (schema modifications, data format changes), the hash
//! will change, triggering automatic cache invalidation on next startup.
//!
//! ## How it works:
//! 1. At build time: Hash all cache-critical files and store as CACHE_SCHEMA_HASH env var
//! 2. At runtime: Compare stored hash in meta.db with current CACHE_SCHEMA_HASH
//! 3. On mismatch: Warn user and suggest `rfx index` to rebuild cache
//!
//! ## Cache-critical files:
//! - src/cache.rs: SQLite schema definitions (files, statistics, config tables)
//! - src/content_store.rs: Binary format for content.bin (magic bytes, offsets)
//! - src/trigram.rs: Inverted index format for trigrams.bin (posting lists)
//! - src/indexer.rs: Data extraction and serialization logic
//! - src/symbol_cache.rs: Symbol storage format
//! - src/models.rs: Core data structures (Span, SymbolKind, SearchResult)
//! - src/dependency.rs: Dependency extraction and storage
//!
//! Changes to these files may break compatibility with existing cache files.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

/// Cache-critical source files that affect binary format compatibility
const CACHE_CRITICAL_FILES: &[&str] = &[
    "src/cache.rs",
    "src/content_store.rs",
    "src/trigram.rs",
    "src/indexer.rs",
    "src/symbol_cache.rs",
    "src/models.rs",
    "src/dependency.rs",
];

fn main() {
    // Compute schema hash from all cache-critical files
    let schema_hash = compute_schema_hash();

    // Export as environment variable for runtime access
    println!("cargo:rustc-env=CACHE_SCHEMA_HASH={}", schema_hash);

    // Tell cargo to rerun this build script if any cache-critical file changes
    for file in CACHE_CRITICAL_FILES {
        println!("cargo:rerun-if-changed={}", file);
    }

    println!("cargo:warning=Cache schema hash: {}", schema_hash);
}

/// Compute a deterministic hash of all cache-critical source files
fn compute_schema_hash() -> String {
    let mut hasher = blake3::Hasher::new();

    // Use BTreeSet to ensure deterministic ordering (sorted by file path)
    let files: BTreeSet<String> = CACHE_CRITICAL_FILES.iter().map(|s| s.to_string()).collect();

    // Hash each file's content in sorted order
    for file_path in &files {
        let path = Path::new(file_path);

        if !path.exists() {
            panic!("Cache-critical file not found: {}", file_path);
        }

        // Read file content
        let content =
            fs::read(path).unwrap_or_else(|e| panic!("Failed to read {}: {}", file_path, e));

        // Hash: file path (for identity) + file content (for changes)
        hasher.update(file_path.as_bytes());
        hasher.update(&content);
    }

    // Return first 16 hex chars (64 bits) for compactness
    // Full blake3 hash is 256 bits, but 64 bits gives ~0% collision probability
    let hash = hasher.finalize();

    // Convert first 8 bytes to hex string manually (no external hex crate needed)
    hash.as_bytes()[..8]
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}
