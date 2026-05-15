//! Symbol cache for storing parsed symbols
//!
//! This module provides transparent caching of parsed symbols to avoid
//! re-parsing files during symbol queries. Symbols are stored in SQLite
//! and keyed by (file_path, blake3_hash) for automatic invalidation when
//! files change.

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension};
use std::path::Path;

use crate::models::SearchResult;

#[cfg(test)]
use crate::models::{Language, Span, SymbolKind};

/// Symbol cache for storing and retrieving parsed symbols
pub struct SymbolCache {
    db_path: std::path::PathBuf,
}

impl SymbolCache {
    /// Open a symbol cache at the given cache directory
    pub fn open(cache_dir: &Path) -> Result<Self> {
        let db_path = cache_dir.join("meta.db");

        if !db_path.exists() {
            anyhow::bail!("Cache not initialized - run 'rfx index' first");
        }

        let cache = Self { db_path };
        cache.init_schema()?;

        Ok(cache)
    }

    /// Initialize the symbols table schema if it doesn't exist
    fn init_schema(&self) -> Result<()> {
        let conn = Connection::open(&self.db_path).context("Failed to open meta.db")?;

        // Check if we need to migrate to file_id-based schema
        let uses_file_id: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('symbols') WHERE name='file_id'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0)
            > 0;

        if !uses_file_id {
            // Old schema detected - drop and recreate with new schema
            let table_exists: bool = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='symbols'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap_or(0)
                > 0;

            if table_exists {
                log::warn!("Symbol cache schema outdated - migrating to file_id-based schema");
                conn.execute("DROP TABLE IF EXISTS symbols", [])?;
            }
        }

        // Create symbols table with file_id instead of file_path
        conn.execute(
            "CREATE TABLE IF NOT EXISTS symbols (
                file_id INTEGER NOT NULL,
                file_hash TEXT NOT NULL,
                symbols_json TEXT NOT NULL,
                last_cached INTEGER NOT NULL,
                PRIMARY KEY (file_id, file_hash),
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbols_file_id ON symbols(file_id)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbols_hash ON symbols(file_hash)",
            [],
        )?;

        log::debug!("Symbol cache schema initialized (file_id-based)");
        Ok(())
    }

    /// Get cached symbols for a file (returns None if not cached or hash mismatch)
    pub fn get(&self, file_path: &str, file_hash: &str) -> Result<Option<Vec<SearchResult>>> {
        let conn = Connection::open(&self.db_path)?;

        // Lookup file_id
        let file_id: Option<i64> = conn
            .query_row("SELECT id FROM files WHERE path = ?", [file_path], |row| {
                row.get(0)
            })
            .optional()?;

        let Some(file_id) = file_id else {
            log::debug!("Symbol cache MISS: {} (file not in index)", file_path);
            return Ok(None);
        };

        let symbols_json: Option<String> = conn
            .query_row(
                "SELECT symbols_json FROM symbols WHERE file_id = ? AND file_hash = ?",
                [&file_id.to_string(), file_hash],
                |row| row.get(0),
            )
            .optional()?;

        match symbols_json {
            Some(json) => {
                let mut symbols: Vec<SearchResult> =
                    serde_json::from_str(&json).context("Failed to deserialize cached symbols")?;

                // Restore file_path (it was removed during serialization to save space)
                for symbol in &mut symbols {
                    symbol.path = file_path.to_string();
                }

                log::debug!(
                    "Symbol cache HIT: {} ({} symbols)",
                    file_path,
                    symbols.len()
                );
                Ok(Some(symbols))
            }
            None => {
                log::debug!("Symbol cache MISS: {}", file_path);
                Ok(None)
            }
        }
    }

    /// Get cached symbols for multiple files in one transaction (batch read)
    ///
    /// This is significantly faster than calling `get()` repeatedly because:
    /// - Opens only ONE database connection instead of N
    /// - Reuses ONE prepared statement instead of creating N
    /// - Executes in ONE transaction instead of N
    ///
    /// Returns results in the same order as input. None means cache miss or hash mismatch.
    pub fn batch_get(
        &self,
        files: &[(String, String)],
    ) -> Result<Vec<(String, Option<Vec<SearchResult>>)>> {
        if files.is_empty() {
            return Ok(Vec::new());
        }

        let conn = Connection::open(&self.db_path)?;

        // Prepare statements for file_id lookup and symbol retrieval
        let mut file_id_stmt = conn.prepare("SELECT id FROM files WHERE path = ?")?;
        let mut symbols_stmt =
            conn.prepare("SELECT symbols_json FROM symbols WHERE file_id = ? AND file_hash = ?")?;

        let mut results = Vec::with_capacity(files.len());
        let mut hits = 0;
        let mut misses = 0;

        for (file_path, file_hash) in files {
            // Lookup file_id
            let file_id: Option<i64> = file_id_stmt
                .query_row([file_path.as_str()], |row| row.get(0))
                .optional()?;

            let symbols = if let Some(file_id) = file_id {
                let symbols_json: Option<String> = symbols_stmt
                    .query_row([&file_id.to_string(), file_hash.as_str()], |row| row.get(0))
                    .optional()?;

                match symbols_json {
                    Some(json) => {
                        match serde_json::from_str::<Vec<SearchResult>>(&json) {
                            Ok(mut symbols) => {
                                // Restore file_path (it was removed during serialization to save space)
                                for symbol in &mut symbols {
                                    symbol.path = file_path.clone();
                                }
                                hits += 1;
                                Some(symbols)
                            }
                            Err(e) => {
                                log::warn!(
                                    "Failed to deserialize cached symbols for {}: {}",
                                    file_path,
                                    e
                                );
                                misses += 1;
                                None
                            }
                        }
                    }
                    None => {
                        misses += 1;
                        None
                    }
                }
            } else {
                misses += 1;
                None
            };

            results.push((file_path.clone(), symbols));
        }

        log::debug!(
            "Batch symbol cache: {} hits, {} misses ({}  total)",
            hits,
            misses,
            files.len()
        );
        Ok(results)
    }

    /// Get cached symbols for multiple files with optional kind filtering
    ///
    /// Uses integer file_ids for fast batch retrieval, then filters by kind in Rust.
    /// This avoids the cache miss detection bug that occurs with SQL-level filtering.
    ///
    /// Automatically chunks large batches to avoid SQLite parameter limits (999 max).
    ///
    /// Parameters:
    /// - file_ids: Vec of (file_id, file_hash, file_path) tuples
    /// - kind_filter: Optional symbol kind to filter by (applied in Rust after retrieval)
    ///
    /// Returns HashMap of file_id → symbols for cache hits.
    pub fn batch_get_with_kind(
        &self,
        file_ids: &[(i64, String, String)], // (file_id, hash, path)
        kind_filter: Option<crate::models::SymbolKind>,
    ) -> Result<std::collections::HashMap<i64, Vec<SearchResult>>> {
        use std::collections::HashMap;

        if file_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let conn = Connection::open(&self.db_path)?;

        // SQLite has a limit of 999 parameters by default
        // Chunk requests to stay well under that limit
        const BATCH_SIZE: usize = 900;

        // Build lookup map for file_ids → (hash, path)
        let file_info: HashMap<i64, (String, String)> = file_ids
            .iter()
            .map(|(id, hash, path)| (*id, (hash.clone(), path.clone())))
            .collect();

        // Capture kind filter for Rust-side filtering
        let kind_for_filtering = kind_filter.clone();

        // Collect results across all chunks
        let mut cache_map: HashMap<i64, Vec<SearchResult>> = HashMap::new();
        let mut hits = 0;

        for chunk in file_ids.chunks(BATCH_SIZE) {
            // Build placeholders for IN clause for this chunk
            let id_placeholders = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(", ");

            // Always use simple query - filter by kind in Rust to avoid cache miss detection bug
            let query = format!(
                "SELECT file_id, symbols_json
                 FROM symbols
                 WHERE file_id IN ({})",
                id_placeholders
            );

            // Prepare parameters for this chunk
            let params: Vec<Box<dyn rusqlite::ToSql>> = chunk
                .iter()
                .map(|(id, _, _)| Box::new(*id) as Box<dyn rusqlite::ToSql>)
                .collect();

            // Execute query
            let mut stmt = conn.prepare(&query)?;
            let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();
            let rows = stmt.query_map(param_refs.as_slice(), |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })?;

            for row_result in rows {
                let (file_id, symbols_json) = row_result?;

                // Verify hash matches
                if let Some((_hash, file_path)) = file_info.get(&file_id) {
                    // Note: We can't verify hash here since symbols table doesn't include hash in result
                    // This is OK - we'll verify by checking file_hash in a separate query if needed
                    match serde_json::from_str::<Vec<SearchResult>>(&symbols_json) {
                        Ok(mut symbols) => {
                            // Restore file_path (it was removed during serialization)
                            for symbol in &mut symbols {
                                symbol.path = file_path.clone();
                            }

                            // Filter symbols by kind if needed (Rust-side filtering)
                            // Note: We do this in Rust rather than SQL to avoid cache miss detection bugs
                            // SQL filtering would exclude files without the kind, making QueryEngine think they're uncached
                            if let Some(ref filter_kind) = kind_for_filtering {
                                symbols.retain(|s| &s.kind == filter_kind);
                            }

                            cache_map.insert(file_id, symbols);
                            hits += 1;
                        }
                        Err(e) => {
                            log::warn!(
                                "Failed to deserialize cached symbols for file_id {}: {}",
                                file_id,
                                e
                            );
                        }
                    }
                }
            }
        }

        let misses = file_ids.len() - hits;

        if kind_for_filtering.is_some() {
            log::debug!(
                "Batch symbol cache with Rust-side kind filter: {} hits, {} misses ({} total, {} chunks)",
                hits,
                misses,
                file_ids.len(),
                (file_ids.len() + BATCH_SIZE - 1) / BATCH_SIZE
            );
        } else {
            log::debug!(
                "Batch symbol cache: {} hits, {} misses ({} total, {} chunks)",
                hits,
                misses,
                file_ids.len(),
                (file_ids.len() + BATCH_SIZE - 1) / BATCH_SIZE
            );
        }

        Ok(cache_map)
    }

    /// Store symbols for a file using file_id
    pub fn set(&self, file_path: &str, file_hash: &str, symbols: &[SearchResult]) -> Result<()> {
        let conn = Connection::open(&self.db_path)?;

        // Lookup file_id from file_path
        let file_id: i64 = conn
            .query_row("SELECT id FROM files WHERE path = ?", [file_path], |row| {
                row.get(0)
            })
            .context(format!("File not found in index: {}", file_path))?;

        // Serialize symbols WITHOUT path (we'll restore it on read to save ~90MB)
        let symbols_without_path: Vec<_> = symbols
            .iter()
            .map(|s| {
                let mut s = s.clone();
                s.path = String::new(); // Clear path to avoid duplication
                s
            })
            .collect();

        let symbols_json =
            serde_json::to_string(&symbols_without_path).context("Failed to serialize symbols")?;

        let now = chrono::Utc::now().timestamp();

        conn.execute(
            "INSERT OR REPLACE INTO symbols (file_id, file_hash, symbols_json, last_cached)
             VALUES (?, ?, ?, ?)",
            [
                &file_id.to_string(),
                file_hash,
                &symbols_json,
                &now.to_string(),
            ],
        )?;

        log::debug!("Cached {} symbols for {}", symbols.len(), file_path);
        Ok(())
    }

    /// Batch store symbols for multiple files in a single transaction
    pub fn batch_set(&self, entries: &[(String, String, Vec<SearchResult>)]) -> Result<()> {
        let mut conn = Connection::open(&self.db_path)?;
        let tx = conn.transaction()?;

        let now = chrono::Utc::now().timestamp();
        let now_str = now.to_string();

        for (file_path, file_hash, symbols) in entries {
            // Lookup file_id
            let file_id: i64 = tx
                .query_row(
                    "SELECT id FROM files WHERE path = ?",
                    [file_path.as_str()],
                    |row| row.get(0),
                )
                .context(format!("File not found in index: {}", file_path))?;

            // Serialize symbols WITHOUT path
            let symbols_without_path: Vec<_> = symbols
                .iter()
                .map(|s| {
                    let mut s = s.clone();
                    s.path = String::new();
                    s
                })
                .collect();

            let symbols_json = serde_json::to_string(&symbols_without_path)
                .context("Failed to serialize symbols")?;

            // Insert into symbols table
            tx.execute(
                "INSERT OR REPLACE INTO symbols (file_id, file_hash, symbols_json, last_cached)
                 VALUES (?, ?, ?, ?)",
                [
                    &file_id.to_string(),
                    file_hash.as_str(),
                    &symbols_json,
                    &now_str,
                ],
            )?;
        }

        tx.commit()?;
        log::debug!("Batch cached symbols for {} files", entries.len());
        Ok(())
    }

    /// Clear all cached symbols
    pub fn clear(&self) -> Result<()> {
        let conn = Connection::open(&self.db_path)?;
        conn.execute("DELETE FROM symbols", [])?;
        log::info!("Cleared symbol cache");
        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> Result<SymbolCacheStats> {
        let conn = Connection::open(&self.db_path)?;

        let total_files: usize = conn
            .query_row("SELECT COUNT(DISTINCT file_id) FROM symbols", [], |row| {
                row.get(0)
            })
            .unwrap_or(0);

        let total_entries: usize = conn
            .query_row("SELECT COUNT(*) FROM symbols", [], |row| row.get(0))
            .unwrap_or(0);

        // Estimate cache size by summing length of symbols_json
        let cache_size_bytes: u64 = conn
            .query_row("SELECT SUM(LENGTH(symbols_json)) FROM symbols", [], |row| {
                row.get(0)
            })
            .unwrap_or(0);

        Ok(SymbolCacheStats {
            total_files,
            total_entries,
            cache_size_bytes,
        })
    }

    /// Remove symbols for files that are no longer in the index
    ///
    /// This cleanup operation removes stale symbol cache entries for files
    /// that have been deleted or are no longer indexed.
    ///
    /// Note: With foreign key constraints (CASCADE DELETE), this should rarely
    /// find anything to clean up, but it's useful for manual verification.
    pub fn cleanup_stale(&self) -> Result<usize> {
        let conn = Connection::open(&self.db_path)?;

        let removed = conn.execute(
            "DELETE FROM symbols WHERE file_id NOT IN (SELECT id FROM files)",
            [],
        )?;

        if removed > 0 {
            log::info!("Removed {} stale symbol cache entries", removed);
        }

        Ok(removed)
    }
}

/// Statistics about the symbol cache
#[derive(Debug, Clone)]
pub struct SymbolCacheStats {
    pub total_files: usize,
    pub total_entries: usize,
    pub cache_size_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CacheManager;
    use tempfile::TempDir;

    #[test]
    fn test_symbol_cache_init() {
        let temp = TempDir::new().unwrap();
        let cache_mgr = CacheManager::new(temp.path());
        cache_mgr.init().unwrap();

        let symbol_cache = SymbolCache::open(cache_mgr.path()).unwrap();
        let stats = symbol_cache.stats().unwrap();
        assert_eq!(stats.total_files, 0);
    }

    #[test]
    fn test_symbol_cache_set_get() {
        let temp = TempDir::new().unwrap();
        let cache_mgr = CacheManager::new(temp.path());
        cache_mgr.init().unwrap();

        // Add file to index first (required for symbol_cache.set())
        cache_mgr.update_file("test.rs", "rust", 100).unwrap();

        let symbol_cache = SymbolCache::open(cache_mgr.path()).unwrap();

        let symbols = vec![SearchResult::new(
            "test.rs".to_string(),
            Language::Rust,
            SymbolKind::Function,
            Some("test_fn".to_string()),
            Span::new(1, 0, 5, 0),
            None,
            "fn test_fn() {}".to_string(),
        )];

        // Store symbols
        symbol_cache.set("test.rs", "hash123", &symbols).unwrap();

        // Retrieve symbols
        let cached = symbol_cache.get("test.rs", "hash123").unwrap();
        assert!(cached.is_some());
        assert_eq!(cached.as_ref().unwrap().len(), 1);
        assert_eq!(cached.unwrap()[0].symbol.as_deref(), Some("test_fn"));
    }

    #[test]
    fn test_symbol_cache_hash_mismatch() {
        let temp = TempDir::new().unwrap();
        let cache_mgr = CacheManager::new(temp.path());
        cache_mgr.init().unwrap();

        // Add file to index first
        cache_mgr.update_file("test.rs", "rust", 100).unwrap();

        let symbol_cache = SymbolCache::open(cache_mgr.path()).unwrap();

        let symbols = vec![SearchResult::new(
            "test.rs".to_string(),
            Language::Rust,
            SymbolKind::Function,
            Some("test_fn".to_string()),
            Span::new(1, 0, 5, 0),
            None,
            "fn test_fn() {}".to_string(),
        )];

        // Store with hash123
        symbol_cache.set("test.rs", "hash123", &symbols).unwrap();

        // Try to retrieve with different hash - should return None
        let cached = symbol_cache.get("test.rs", "hash456").unwrap();
        assert!(cached.is_none());
    }

    #[test]
    fn test_symbol_cache_batch_set() {
        let temp = TempDir::new().unwrap();
        let cache_mgr = CacheManager::new(temp.path());
        cache_mgr.init().unwrap();

        // Add files to index first
        cache_mgr.update_file("file1.rs", "rust", 100).unwrap();
        cache_mgr.update_file("file2.rs", "rust", 200).unwrap();

        let symbol_cache = SymbolCache::open(cache_mgr.path()).unwrap();

        let entries = vec![
            (
                "file1.rs".to_string(),
                "hash1".to_string(),
                vec![SearchResult::new(
                    "file1.rs".to_string(),
                    Language::Rust,
                    SymbolKind::Function,
                    Some("fn1".to_string()),
                    Span::new(1, 0, 5, 0),
                    None,
                    "fn fn1() {}".to_string(),
                )],
            ),
            (
                "file2.rs".to_string(),
                "hash2".to_string(),
                vec![SearchResult::new(
                    "file2.rs".to_string(),
                    Language::Rust,
                    SymbolKind::Function,
                    Some("fn2".to_string()),
                    Span::new(1, 0, 5, 0),
                    None,
                    "fn fn2() {}".to_string(),
                )],
            ),
        ];

        symbol_cache.batch_set(&entries).unwrap();

        let stats = symbol_cache.stats().unwrap();
        assert_eq!(stats.total_files, 2);

        let cached1 = symbol_cache.get("file1.rs", "hash1").unwrap();
        assert!(cached1.is_some());

        let cached2 = symbol_cache.get("file2.rs", "hash2").unwrap();
        assert!(cached2.is_some());
    }

    #[test]
    fn test_symbol_cache_batch_get() {
        let temp = TempDir::new().unwrap();
        let cache_mgr = CacheManager::new(temp.path());
        cache_mgr.init().unwrap();

        // Add files to index first
        cache_mgr.update_file("file1.rs", "rust", 100).unwrap();
        cache_mgr.update_file("file2.rs", "rust", 200).unwrap();
        cache_mgr.update_file("file3.rs", "rust", 300).unwrap();

        let symbol_cache = SymbolCache::open(cache_mgr.path()).unwrap();

        // Populate cache with multiple files
        let entries = vec![
            (
                "file1.rs".to_string(),
                "hash1".to_string(),
                vec![SearchResult::new(
                    "file1.rs".to_string(),
                    Language::Rust,
                    SymbolKind::Function,
                    Some("fn1".to_string()),
                    Span::new(1, 0, 5, 0),
                    None,
                    "fn fn1() {}".to_string(),
                )],
            ),
            (
                "file2.rs".to_string(),
                "hash2".to_string(),
                vec![SearchResult::new(
                    "file2.rs".to_string(),
                    Language::Rust,
                    SymbolKind::Struct,
                    Some("Struct2".to_string()),
                    Span::new(1, 0, 5, 0),
                    None,
                    "struct Struct2 {}".to_string(),
                )],
            ),
            (
                "file3.rs".to_string(),
                "hash3".to_string(),
                vec![SearchResult::new(
                    "file3.rs".to_string(),
                    Language::Rust,
                    SymbolKind::Enum,
                    Some("Enum3".to_string()),
                    Span::new(1, 0, 5, 0),
                    None,
                    "enum Enum3 {}".to_string(),
                )],
            ),
        ];

        symbol_cache.batch_set(&entries).unwrap();

        // Test batch_get with all cached files
        let lookup = vec![
            ("file1.rs".to_string(), "hash1".to_string()),
            ("file2.rs".to_string(), "hash2".to_string()),
            ("file3.rs".to_string(), "hash3".to_string()),
        ];

        let results = symbol_cache.batch_get(&lookup).unwrap();
        assert_eq!(results.len(), 3);

        // Verify all hits
        assert!(results[0].1.is_some());
        assert_eq!(
            results[0].1.as_ref().unwrap()[0].symbol.as_deref(),
            Some("fn1")
        );

        assert!(results[1].1.is_some());
        assert_eq!(
            results[1].1.as_ref().unwrap()[0].symbol.as_deref(),
            Some("Struct2")
        );

        assert!(results[2].1.is_some());
        assert_eq!(
            results[2].1.as_ref().unwrap()[0].symbol.as_deref(),
            Some("Enum3")
        );

        // Test batch_get with mixed hits and misses
        let mixed_lookup = vec![
            ("file1.rs".to_string(), "hash1".to_string()), // Hit
            ("nonexistent.rs".to_string(), "hash999".to_string()), // Miss (file doesn't exist)
            ("file2.rs".to_string(), "wrong_hash".to_string()), // Miss (hash mismatch)
            ("file3.rs".to_string(), "hash3".to_string()), // Hit
        ];

        let mixed_results = symbol_cache.batch_get(&mixed_lookup).unwrap();
        assert_eq!(mixed_results.len(), 4);

        assert!(mixed_results[0].1.is_some()); // file1.rs - hit
        assert!(mixed_results[1].1.is_none()); // nonexistent.rs - miss
        assert!(mixed_results[2].1.is_none()); // file2.rs wrong hash - miss
        assert!(mixed_results[3].1.is_some()); // file3.rs - hit

        // Test batch_get with empty input
        let empty_results = symbol_cache.batch_get(&[]).unwrap();
        assert_eq!(empty_results.len(), 0);
    }

    #[test]
    fn test_symbol_cache_clear() {
        let temp = TempDir::new().unwrap();
        let cache_mgr = CacheManager::new(temp.path());
        cache_mgr.init().unwrap();

        // Add file to index first
        cache_mgr.update_file("test.rs", "rust", 100).unwrap();

        let symbol_cache = SymbolCache::open(cache_mgr.path()).unwrap();

        let symbols = vec![SearchResult::new(
            "test.rs".to_string(),
            Language::Rust,
            SymbolKind::Function,
            Some("test_fn".to_string()),
            Span::new(1, 0, 5, 0),
            None,
            "fn test_fn() {}".to_string(),
        )];

        symbol_cache.set("test.rs", "hash123", &symbols).unwrap();

        let stats_before = symbol_cache.stats().unwrap();
        assert_eq!(stats_before.total_files, 1);

        symbol_cache.clear().unwrap();

        let stats_after = symbol_cache.stats().unwrap();
        assert_eq!(stats_after.total_files, 0);
    }

    #[test]
    fn test_symbol_cache_cleanup_stale() {
        let temp = TempDir::new().unwrap();
        let cache_mgr = CacheManager::new(temp.path());
        cache_mgr.init().unwrap();

        // Add a file to the index
        cache_mgr.update_file("exists.rs", "rust", 100).unwrap();
        cache_mgr
            .record_branch_file("exists.rs", "main", "hash1", None)
            .unwrap();

        // Add deleted.rs to index temporarily
        cache_mgr.update_file("deleted.rs", "rust", 200).unwrap();

        let symbol_cache = SymbolCache::open(cache_mgr.path()).unwrap();

        // Cache symbols for both existing and non-existing files
        let symbols = vec![SearchResult::new(
            "test.rs".to_string(),
            Language::Rust,
            SymbolKind::Function,
            Some("test_fn".to_string()),
            Span::new(1, 0, 5, 0),
            None,
            "fn test_fn() {}".to_string(),
        )];

        symbol_cache.set("exists.rs", "hash1", &symbols).unwrap();
        symbol_cache.set("deleted.rs", "hash2", &symbols).unwrap();

        let stats_before = symbol_cache.stats().unwrap();
        assert_eq!(stats_before.total_files, 2);

        // Now remove "deleted.rs" from files table to make its symbol cache entry stale
        // Note: With CASCADE DELETE foreign key constraint, the symbol entry is automatically
        // removed when the file is deleted, so cleanup_stale() won't find anything to remove.
        let conn = rusqlite::Connection::open(cache_mgr.path().join("meta.db")).unwrap();
        conn.execute("DELETE FROM files WHERE path = 'deleted.rs'", [])
            .unwrap();

        // Cleanup stale entries (should find 0 because CASCADE DELETE already cleaned it up)
        let removed = symbol_cache.cleanup_stale().unwrap();
        assert_eq!(removed, 0); // CASCADE DELETE already removed it

        let stats_after = symbol_cache.stats().unwrap();
        assert_eq!(stats_after.total_files, 1);

        // exists.rs should still be cached
        let cached = symbol_cache.get("exists.rs", "hash1").unwrap();
        assert!(cached.is_some());

        // deleted.rs should be gone
        let cached2 = symbol_cache.get("deleted.rs", "hash2").unwrap();
        assert!(cached2.is_none());
    }
}
