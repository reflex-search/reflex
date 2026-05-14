//! Cache management and memory-mapped I/O
//!
//! The cache module handles the `.reflex/` directory structure:
//! - `meta.db`: Metadata, file hashes, and configuration (SQLite)
//! - `tokens.bin`: Compressed lexical tokens (binary)
//! - `content.bin`: Memory-mapped file contents (binary)
//! - `trigrams.bin`: Trigram inverted index (custom varint+zstd binary, V3 format)
//! - `config.toml`: Index settings (TOML text)

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::models::IndexedFile;

/// Default cache directory name
pub const CACHE_DIR: &str = ".reflex";

/// File names within the cache directory
pub const META_DB: &str = "meta.db";
pub const TOKENS_BIN: &str = "tokens.bin";
pub const HASHES_JSON: &str = "hashes.json";
pub const CONFIG_TOML: &str = "config.toml";

/// Manages the Reflex cache directory
#[derive(Clone)]
pub struct CacheManager {
    cache_path: PathBuf,
}

impl CacheManager {
    /// Create a new cache manager for the given root directory
    pub fn new(root: impl AsRef<Path>) -> Self {
        let cache_path = root.as_ref().join(CACHE_DIR);
        Self { cache_path }
    }

    /// Initialize the cache directory structure if it doesn't exist
    pub fn init(&self) -> Result<()> {
        log::info!("Initializing cache at {:?}", self.cache_path);

        if !self.cache_path.exists() {
            std::fs::create_dir_all(&self.cache_path)?;
        }

        // Create meta.db with schema
        self.init_meta_db()?;

        // Create default config.toml
        self.init_config_toml()?;

        // Note: tokens.bin removed - was never used
        // Note: hashes.json is deprecated - hashes are now stored in meta.db

        log::info!("Cache initialized successfully");
        Ok(())
    }

    /// Initialize meta.db with SQLite schema
    fn init_meta_db(&self) -> Result<()> {
        let db_path = self.cache_path.join(META_DB);

        // Skip if already exists
        if db_path.exists() {
            return Ok(());
        }

        let conn = Connection::open(&db_path)
            .context("Failed to create meta.db")?;

        // Create files table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                last_indexed INTEGER NOT NULL,
                language TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                line_count INTEGER DEFAULT 0
            )",
            [],
        )?;

        conn.execute("CREATE INDEX IF NOT EXISTS idx_files_path ON files(path)", [])?;

        // Create statistics table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS statistics (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            )",
            [],
        )?;

        // Initialize default statistics
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "INSERT OR REPLACE INTO statistics (key, value, updated_at) VALUES (?, ?, ?)",
            ["total_files", "0", &now.to_string()],
        )?;
        conn.execute(
            "INSERT OR REPLACE INTO statistics (key, value, updated_at) VALUES (?, ?, ?)",
            ["cache_version", "1", &now.to_string()],
        )?;

        // Store cache schema hash for automatic invalidation detection
        // This hash is computed at build time from cache-critical source files
        let schema_hash = env!("CACHE_SCHEMA_HASH");
        conn.execute(
            "INSERT OR REPLACE INTO statistics (key, value, updated_at) VALUES (?, ?, ?)",
            ["schema_hash", schema_hash, &now.to_string()],
        )?;

        // Initialize last_compaction timestamp (0 = never compacted)
        conn.execute(
            "INSERT OR REPLACE INTO statistics (key, value, updated_at) VALUES (?, ?, ?)",
            ["last_compaction", "0", &now.to_string()],
        )?;

        // Create config table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )",
            [],
        )?;

        // Create branch tracking tables for git-aware indexing
        conn.execute(
            "CREATE TABLE IF NOT EXISTS file_branches (
                file_id INTEGER NOT NULL,
                branch_id INTEGER NOT NULL,
                hash TEXT NOT NULL,
                last_indexed INTEGER NOT NULL,
                PRIMARY KEY (file_id, branch_id),
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
                FOREIGN KEY (branch_id) REFERENCES branches(id) ON DELETE CASCADE
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_branch_lookup ON file_branches(branch_id, file_id)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hash_lookup ON file_branches(hash)",
            [],
        )?;

        // Create branches metadata table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS branches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                commit_sha TEXT NOT NULL,
                last_indexed INTEGER NOT NULL,
                file_count INTEGER DEFAULT 0,
                is_dirty INTEGER DEFAULT 0
            )",
            [],
        )?;

        // Create file dependencies table for tracking imports/includes
        conn.execute(
            "CREATE TABLE IF NOT EXISTS file_dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                imported_path TEXT NOT NULL,
                resolved_file_id INTEGER,
                import_type TEXT NOT NULL,
                line_number INTEGER NOT NULL,
                imported_symbols TEXT,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
                FOREIGN KEY (resolved_file_id) REFERENCES files(id) ON DELETE SET NULL
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_deps_file ON file_dependencies(file_id)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_deps_resolved ON file_dependencies(resolved_file_id)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_deps_type ON file_dependencies(import_type)",
            [],
        )?;

        // Create file exports table for tracking barrel re-exports
        conn.execute(
            "CREATE TABLE IF NOT EXISTS file_exports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                exported_symbol TEXT,
                source_path TEXT NOT NULL,
                resolved_source_id INTEGER,
                line_number INTEGER NOT NULL,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
                FOREIGN KEY (resolved_source_id) REFERENCES files(id) ON DELETE SET NULL
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_exports_file ON file_exports(file_id)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_exports_resolved ON file_exports(resolved_source_id)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_exports_symbol ON file_exports(exported_symbol)",
            [],
        )?;

        log::debug!("Created meta.db with schema");
        Ok(())
    }

    /// Initialize config.toml with defaults
    fn init_config_toml(&self) -> Result<()> {
        let config_path = self.cache_path.join(CONFIG_TOML);

        if config_path.exists() {
            return Ok(());
        }

        let default_config = r#"[index]
languages = []  # Empty = all supported languages
max_file_size = 10485760  # 10 MB
follow_symlinks = false

[index.include]
patterns = []

[index.exclude]
patterns = []

[search]
default_limit = 100
fuzzy_threshold = 0.8

[performance]
parallel_threads = 0  # 0 = auto (80% of available cores), or set a specific number
compression_level = 3  # zstd level

[semantic]
# Semantic query generation using LLMs
# Translate natural language questions into rfx query commands
provider = "openrouter"  # Options: openai, anthropic, openrouter
# model = "openai/gpt-4o-mini"  # Optional: override provider default model
# auto_execute = false  # Optional: auto-execute queries without confirmation
"#;

        std::fs::write(&config_path, default_config)?;

        log::debug!("Created default config.toml");
        Ok(())
    }

    /// Check if cache exists and is valid
    pub fn exists(&self) -> bool {
        self.cache_path.exists()
            && self.cache_path.join(META_DB).exists()
    }

    /// Validate cache integrity and detect corruption
    ///
    /// Performs basic integrity checks on the cache:
    /// - Verifies all required files exist
    /// - Checks SQLite database can be opened
    /// - Validates binary file headers (trigrams.bin, content.bin)
    ///
    /// Returns Ok(()) if cache is valid, Err with details if corrupted.
    pub fn validate(&self) -> Result<()> {
        let start = std::time::Instant::now();

        // Check if cache directory exists
        if !self.cache_path.exists() {
            anyhow::bail!("Cache directory does not exist: {}", self.cache_path.display());
        }

        // Check meta.db exists and can be opened
        let db_path = self.cache_path.join(META_DB);
        if !db_path.exists() {
            anyhow::bail!("Database file missing: {}", db_path.display());
        }

        // Try to open database
        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db - database may be corrupted")?;

        // Verify schema exists
        let tables: Result<Vec<String>, _> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table'")
            .and_then(|mut stmt| {
                stmt.query_map([], |row| row.get(0))
                    .map(|rows| rows.collect())
            })
            .and_then(|result| result);

        match tables {
            Ok(table_list) => {
                // Check for required tables
                let required_tables = vec!["files", "statistics", "config", "file_branches", "branches", "file_dependencies", "file_exports"];
                for table in &required_tables {
                    if !table_list.iter().any(|t| t == table) {
                        anyhow::bail!("Required table '{}' missing from database schema", table);
                    }
                }
            }
            Err(e) => {
                anyhow::bail!("Failed to read database schema: {}", e);
            }
        }

        // Run SQLite integrity check (fast quick_check)
        // Use quick_check instead of integrity_check for speed (<10ms vs 100ms+)
        let integrity_result: String = conn
            .query_row("PRAGMA quick_check", [], |row| row.get(0))?;

        if integrity_result != "ok" {
            log::warn!("Database integrity check failed: {}", integrity_result);
            anyhow::bail!(
                "Database integrity check failed: {}. Cache may be corrupted. \
                 Run 'rfx index' to rebuild cache.",
                integrity_result
            );
        }

        // Check trigrams.bin if it exists
        let trigrams_path = self.cache_path.join("trigrams.bin");
        if trigrams_path.exists() {
            use std::io::Read;

            match File::open(&trigrams_path) {
                Ok(mut file) => {
                    let mut header = [0u8; 4];
                    match file.read_exact(&mut header) {
                        Ok(_) => {
                            // Check magic bytes
                            if &header != b"RFTG" {
                                log::warn!("trigrams.bin has invalid magic bytes - may be corrupted");
                                anyhow::bail!("trigrams.bin appears to be corrupted (invalid magic bytes)");
                            }
                        }
                        Err(_) => {
                            anyhow::bail!("trigrams.bin is too small - appears to be corrupted");
                        }
                    }
                }
                Err(e) => {
                    anyhow::bail!("Failed to open trigrams.bin: {}", e);
                }
            }
        }

        // Check content.bin if it exists
        let content_path = self.cache_path.join("content.bin");
        if content_path.exists() {
            use std::io::Read;

            match File::open(&content_path) {
                Ok(mut file) => {
                    let mut header = [0u8; 4];
                    match file.read_exact(&mut header) {
                        Ok(_) => {
                            // Check magic bytes
                            if &header != b"RFCT" {
                                log::warn!("content.bin has invalid magic bytes - may be corrupted");
                                anyhow::bail!("content.bin appears to be corrupted (invalid magic bytes)");
                            }
                        }
                        Err(_) => {
                            anyhow::bail!("content.bin is too small - appears to be corrupted");
                        }
                    }
                }
                Err(e) => {
                    anyhow::bail!("Failed to open content.bin: {}", e);
                }
            }
        }

        // Check schema hash for automatic invalidation
        let current_schema_hash = env!("CACHE_SCHEMA_HASH");

        let stored_schema_hash: Option<String> = conn
            .query_row(
                "SELECT value FROM statistics WHERE key = 'schema_hash'",
                [],
                |row| row.get(0),
            )
            .optional()?;

        if let Some(stored_hash) = stored_schema_hash {
            if stored_hash != current_schema_hash {
                log::warn!(
                    "Cache schema hash mismatch! Stored: {}, Current: {}",
                    stored_hash,
                    current_schema_hash
                );
                anyhow::bail!(
                    "Cache schema version mismatch.\n\
                     \n\
                     - Cache was built with version {}\n\
                     - Current binary expects version {}\n\
                     \n\
                     The cache format may be incompatible with this version of Reflex.\n\
                     Please rebuild the index by running:\n\
                     \n\
                       rfx index\n\
                     \n\
                     This usually happens after upgrading Reflex or making code changes.",
                    stored_hash,
                    current_schema_hash
                );
            }
        } else {
            log::debug!("No schema_hash found in cache - this cache was created before automatic invalidation was implemented");
            // Don't fail for backward compatibility with old caches
            // They will get the hash on next rebuild
        }

        let elapsed = start.elapsed();
        log::debug!("Cache validation passed (schema hash: {}, took {:?})", current_schema_hash, elapsed);
        Ok(())
    }

    /// Get the path to the cache directory
    pub fn path(&self) -> &Path {
        &self.cache_path
    }

    /// Get the workspace root directory (parent of .reflex/)
    pub fn workspace_root(&self) -> PathBuf {
        self.cache_path
            .parent()
            .expect(".reflex directory should have a parent")
            .to_path_buf()
    }

    /// Clear the entire cache
    pub fn clear(&self) -> Result<()> {
        log::warn!("Clearing cache at {:?}", self.cache_path);

        if self.cache_path.exists() {
            std::fs::remove_dir_all(&self.cache_path)?;
        }

        Ok(())
    }

    /// Force SQLite WAL (Write-Ahead Log) checkpoint
    ///
    /// Ensures all data written in transactions is flushed to the main database file.
    /// This is critical when spawning background processes that open new connections,
    /// as they need to see the committed data immediately.
    ///
    /// Uses TRUNCATE mode to completely flush and reset the WAL file.
    pub fn checkpoint_wal(&self) -> Result<()> {
        let db_path = self.cache_path.join(META_DB);

        if !db_path.exists() {
            // No database to checkpoint
            return Ok(());
        }

        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db for WAL checkpoint")?;

        // PRAGMA wal_checkpoint(TRUNCATE) forces a full checkpoint and truncates the WAL
        // This ensures background processes see all committed data
        // Note: Returns (busy, log_pages, checkpointed_pages) - use query instead of execute
        conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| {
            let busy: i64 = row.get(0)?;
            let log_pages: i64 = row.get(1)?;
            let checkpointed: i64 = row.get(2)?;
            log::debug!(
                "WAL checkpoint completed: busy={}, log_pages={}, checkpointed_pages={}",
                busy, log_pages, checkpointed
            );
            Ok(())
        }).context("Failed to execute WAL checkpoint")?;

        log::debug!("Executed WAL checkpoint (TRUNCATE) on meta.db");
        Ok(())
    }

    /// Load all file hashes across all branches from SQLite
    ///
    /// Used by background indexer to get hashes for all indexed files.
    /// Returns the most recent hash for each file across all branches.
    pub fn load_all_hashes(&self) -> Result<HashMap<String, String>> {
        let db_path = self.cache_path.join(META_DB);

        if !db_path.exists() {
            return Ok(HashMap::new());
        }

        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db")?;

        // Get all hashes from file_branches, joined with files to get paths
        // If a file appears in multiple branches, we'll get multiple entries
        // (HashMap will keep the last one, which is fine for background indexer)
        let mut stmt = conn.prepare(
            "SELECT f.path, fb.hash
             FROM file_branches fb
             JOIN files f ON fb.file_id = f.id"
        )?;
        let hashes: HashMap<String, String> = stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?
        .collect::<Result<HashMap<_, _>, _>>()?;

        log::debug!("Loaded {} file hashes across all branches from SQLite", hashes.len());
        Ok(hashes)
    }

    /// Load file hashes for a specific branch from SQLite
    ///
    /// Used by indexer and query engine to get hashes for the current branch.
    /// This ensures branch-specific incremental indexing and symbol cache lookups.
    pub fn load_hashes_for_branch(&self, branch: &str) -> Result<HashMap<String, String>> {
        let db_path = self.cache_path.join(META_DB);

        if !db_path.exists() {
            return Ok(HashMap::new());
        }

        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db")?;

        // Get hashes for specific branch only
        let mut stmt = conn.prepare(
            "SELECT f.path, fb.hash
             FROM file_branches fb
             JOIN files f ON fb.file_id = f.id
             JOIN branches b ON fb.branch_id = b.id
             WHERE b.name = ?"
        )?;
        let hashes: HashMap<String, String> = stmt.query_map([branch], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?
        .collect::<Result<HashMap<_, _>, _>>()?;

        log::debug!("Loaded {} file hashes for branch '{}' from SQLite", hashes.len(), branch);
        Ok(hashes)
    }

    /// Save file hashes for incremental indexing
    ///
    /// DEPRECATED: Hashes are now saved via record_branch_file() or batch_record_branch_files().
    /// This method is kept for backward compatibility but does nothing.
    #[deprecated(note = "Hashes are now stored in file_branches table via record_branch_file()")]
    pub fn save_hashes(&self, _hashes: &HashMap<String, String>) -> Result<()> {
        // No-op: hashes are now persisted to SQLite in record_branch_file()
        Ok(())
    }

    /// Update file metadata in the files table
    ///
    /// Note: File content hashes are stored separately in the file_branches table
    /// via record_branch_file() or batch_record_branch_files().
    pub fn update_file(&self, path: &str, language: &str, line_count: usize) -> Result<()> {
        let db_path = self.cache_path.join(META_DB);
        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db for file update")?;

        let now = chrono::Utc::now().timestamp();

        conn.execute(
            "INSERT OR REPLACE INTO files (path, last_indexed, language, line_count)
             VALUES (?, ?, ?, ?)",
            [path, &now.to_string(), language, &line_count.to_string()],
        )?;

        Ok(())
    }

    /// Batch update multiple files in a single transaction for performance
    ///
    /// Note: File content hashes are stored separately in the file_branches table
    /// via batch_update_files_and_branch().
    pub fn batch_update_files(&self, files: &[(String, String, usize)]) -> Result<()> {
        let db_path = self.cache_path.join(META_DB);
        let mut conn = Connection::open(&db_path)
            .context("Failed to open meta.db for batch update")?;

        let now = chrono::Utc::now().timestamp();
        let now_str = now.to_string();

        // Use a transaction for batch inserts
        let tx = conn.transaction()?;

        for (path, language, line_count) in files {
            tx.execute(
                "INSERT OR REPLACE INTO files (path, last_indexed, language, line_count)
                 VALUES (?, ?, ?, ?)",
                [path.as_str(), &now_str, language.as_str(), &line_count.to_string()],
            )?;
        }

        tx.commit()?;
        Ok(())
    }

    /// Batch update files AND record their hashes for a branch in a SINGLE transaction
    ///
    /// This is the recommended method for indexing as it ensures atomicity:
    /// if files are inserted, their branch hashes are guaranteed to be inserted too.
    pub fn batch_update_files_and_branch(
        &self,
        files: &[(String, String, usize)],      // (path, language, line_count)
        branch_files: &[(String, String)],       // (path, hash)
        branch: &str,
        commit_sha: Option<&str>,
    ) -> Result<()> {
        log::info!("batch_update_files_and_branch: Processing {} files for branch '{}'", files.len(), branch);

        let db_path = self.cache_path.join(META_DB);
        let mut conn = Connection::open(&db_path)
            .context("Failed to open meta.db for batch update and branch recording")?;

        let now = chrono::Utc::now().timestamp();
        let now_str = now.to_string();

        // Use a SINGLE transaction for both operations
        let tx = conn.transaction()?;

        // Step 1: Insert/update files table
        for (path, language, line_count) in files {
            tx.execute(
                "INSERT OR REPLACE INTO files (path, last_indexed, language, line_count)
                 VALUES (?, ?, ?, ?)",
                [path.as_str(), &now_str, language.as_str(), &line_count.to_string()],
            )?;
        }
        log::info!("Inserted {} files into files table", files.len());

        // Step 2: Get or create branch_id (within same transaction)
        let branch_id = self.get_or_create_branch_id(&tx, branch, commit_sha)?;
        log::debug!("Got branch_id={} for branch '{}'", branch_id, branch);

        // Step 3: Insert file_branches entries (within same transaction)
        let mut inserted = 0;
        for (path, hash) in branch_files {
            // Lookup file_id from path (will find it because we just inserted above)
            let file_id: i64 = tx.query_row(
                "SELECT id FROM files WHERE path = ?",
                [path.as_str()],
                |row| row.get(0)
            ).context(format!("File not found in index after insert: {}", path))?;

            // Insert into file_branches using INTEGER values (not strings!)
            tx.execute(
                "INSERT OR REPLACE INTO file_branches (file_id, branch_id, hash, last_indexed)
                 VALUES (?, ?, ?, ?)",
                rusqlite::params![file_id, branch_id, hash.as_str(), now],
            )?;
            inserted += 1;
        }
        log::info!("Inserted {} file_branches entries", inserted);

        // Commit the entire transaction atomically
        tx.commit()?;
        log::info!("Transaction committed successfully (files + file_branches)");

        // DIAGNOSTIC: Verify data was actually persisted after commit
        // This helps diagnose WAL synchronization issues where commits succeed but data isn't visible
        let verify_conn = Connection::open(&db_path)
            .context("Failed to open meta.db for verification")?;

        // Count actual files in database
        let actual_file_count: i64 = verify_conn.query_row(
            "SELECT COUNT(*) FROM files WHERE path IN (SELECT path FROM files ORDER BY id DESC LIMIT ?)",
            [files.len()],
            |row| row.get(0)
        ).unwrap_or(0);

        // Count actual file_branches entries for this branch
        let actual_fb_count: i64 = verify_conn.query_row(
            "SELECT COUNT(*) FROM file_branches fb
             JOIN branches b ON fb.branch_id = b.id
             WHERE b.name = ?",
            [branch],
            |row| row.get(0)
        ).unwrap_or(0);

        log::info!(
            "Post-commit verification: {} files in files table (expected {}), {} file_branches entries for '{}' (expected {})",
            actual_file_count,
            files.len(),
            actual_fb_count,
            branch,
            inserted
        );

        // DEFENSIVE: Warn if counts don't match expectations
        if actual_file_count < files.len() as i64 {
            log::warn!(
                "MISMATCH: Expected {} files in database, but only found {}! Data may not have persisted.",
                files.len(),
                actual_file_count
            );
        }
        if actual_fb_count < inserted as i64 {
            log::warn!(
                "MISMATCH: Expected {} file_branches entries for branch '{}', but only found {}! Data may not have persisted.",
                inserted,
                branch,
                actual_fb_count
            );
        }

        Ok(())
    }

    /// Update statistics after indexing by calculating totals from database for a specific branch
    ///
    /// Counts only files indexed for the given branch, not all files across all branches.
    pub fn update_stats(&self, branch: &str) -> Result<()> {
        let db_path = self.cache_path.join(META_DB);
        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db for stats update")?;

        // Count files for specific branch only (branch-aware statistics)
        let total_files: usize = conn.query_row(
            "SELECT COUNT(DISTINCT fb.file_id)
             FROM file_branches fb
             JOIN branches b ON fb.branch_id = b.id
             WHERE b.name = ?",
            [branch],
            |row| row.get(0),
        ).unwrap_or(0);

        let now = chrono::Utc::now().timestamp();

        conn.execute(
            "INSERT OR REPLACE INTO statistics (key, value, updated_at) VALUES (?, ?, ?)",
            ["total_files", &total_files.to_string(), &now.to_string()],
        )?;

        log::debug!("Updated statistics for branch '{}': {} files", branch, total_files);
        Ok(())
    }

    /// Check if the stored schema hash matches the current binary's hash.
    /// Returns Ok(true) if they match, Ok(false) if they don't, Err on DB errors.
    pub fn check_schema_hash(&self) -> Result<bool> {
        let db_path = self.cache_path.join(META_DB);
        if !db_path.exists() {
            return Ok(false);
        }
        let conn = Connection::open(&db_path)?;
        let current = env!("CACHE_SCHEMA_HASH");
        let stored: Option<String> = conn
            .query_row(
                "SELECT value FROM statistics WHERE key = 'schema_hash'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        Ok(stored.as_deref() == Some(current))
    }

    /// Update cache schema hash in statistics table
    ///
    /// This should be called after every index operation to ensure the cache
    /// is marked as compatible with the current binary version.
    pub fn update_schema_hash(&self) -> Result<()> {
        let db_path = self.cache_path.join(META_DB);
        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db for schema hash update")?;

        let schema_hash = env!("CACHE_SCHEMA_HASH");
        let now = chrono::Utc::now().timestamp();

        conn.execute(
            "INSERT OR REPLACE INTO statistics (key, value, updated_at) VALUES (?, ?, ?)",
            ["schema_hash", schema_hash, &now.to_string()],
        )?;

        log::debug!("Updated schema hash to: {}", schema_hash);
        Ok(())
    }

    /// Get list of all indexed files
    pub fn list_files(&self) -> Result<Vec<IndexedFile>> {
        let db_path = self.cache_path.join(META_DB);

        if !db_path.exists() {
            return Ok(Vec::new());
        }

        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db")?;

        let mut stmt = conn.prepare(
            "SELECT path, language, last_indexed FROM files ORDER BY path"
        )?;

        let files = stmt.query_map([], |row| {
            let path: String = row.get(0)?;
            let language: String = row.get(1)?;
            let last_indexed: i64 = row.get(2)?;

            Ok(IndexedFile {
                path,
                language,
                last_indexed: chrono::DateTime::from_timestamp(last_indexed, 0)
                    .unwrap_or_else(chrono::Utc::now)
                    .to_rfc3339(),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

        Ok(files)
    }

    /// Get statistics about the current cache
    ///
    /// Returns statistics for the current git branch if in a git repo,
    /// or global statistics if not in a git repo.
    pub fn stats(&self) -> Result<crate::models::IndexStats> {
        let db_path = self.cache_path.join(META_DB);

        if !db_path.exists() {
            // Cache not initialized
            return Ok(crate::models::IndexStats {
                total_files: 0,
                index_size_bytes: 0,
                last_updated: chrono::Utc::now().to_rfc3339(),
                files_by_language: std::collections::HashMap::new(),
                lines_by_language: std::collections::HashMap::new(),
            });
        }

        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db")?;

        // Determine current branch for branch-aware statistics
        let workspace_root = self.workspace_root();
        let current_branch = if crate::git::is_git_repo(&workspace_root) {
            crate::git::get_git_state(&workspace_root)
                .ok()
                .map(|state| state.branch)
        } else {
            Some("_default".to_string())
        };

        log::debug!("stats(): current_branch = {:?}", current_branch);

        // Read total files (branch-aware)
        let total_files: usize = if let Some(ref branch) = current_branch {
            log::debug!("stats(): Counting files for branch '{}'", branch);

            // Debug: Check all branches
            let branches: Vec<(i64, String, i64)> = conn.prepare(
                "SELECT id, name, file_count FROM branches"
            )
            .and_then(|mut stmt| {
                stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
                    .map(|rows| rows.collect())
            })
            .and_then(|result| result)
            .unwrap_or_default();

            for (id, name, count) in &branches {
                log::debug!("stats(): Branch ID={}, Name='{}', FileCount={}", id, name, count);
            }

            // Debug: Count file_branches per branch
            let fb_counts: Vec<(String, i64)> = conn.prepare(
                "SELECT b.name, COUNT(*) FROM file_branches fb
                 JOIN branches b ON fb.branch_id = b.id
                 GROUP BY b.name"
            )
            .and_then(|mut stmt| {
                stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                    .map(|rows| rows.collect())
            })
            .and_then(|result| result)
            .unwrap_or_default();

            for (name, count) in &fb_counts {
                log::debug!("stats(): file_branches count for branch '{}': {}", name, count);
            }

            // Count files for current branch only
            let count: usize = conn.query_row(
                "SELECT COUNT(DISTINCT fb.file_id)
                 FROM file_branches fb
                 JOIN branches b ON fb.branch_id = b.id
                 WHERE b.name = ?",
                [branch],
                |row| row.get(0),
            ).unwrap_or(0);

            log::debug!("stats(): Query returned total_files = {}", count);
            count
        } else {
            // No branch info - should not happen, but return 0
            log::warn!("stats(): No current_branch detected!");
            0
        };

        // Read last updated timestamp
        let last_updated: String = conn.query_row(
            "SELECT updated_at FROM statistics WHERE key = 'total_files'",
            [],
            |row| {
                let timestamp: i64 = row.get(0)?;
                Ok(chrono::DateTime::from_timestamp(timestamp, 0)
                    .unwrap_or_else(chrono::Utc::now)
                    .to_rfc3339())
            },
        ).unwrap_or_else(|_| chrono::Utc::now().to_rfc3339());

        // Calculate total cache size (all binary files)
        let mut index_size_bytes: u64 = 0;

        for file_name in [META_DB, TOKENS_BIN, CONFIG_TOML, "content.bin", "trigrams.bin"] {
            let file_path = self.cache_path.join(file_name);
            if let Ok(metadata) = std::fs::metadata(&file_path) {
                index_size_bytes += metadata.len();
            }
        }

        // Get file count breakdown by language (branch-aware if possible)
        let mut files_by_language = std::collections::HashMap::new();
        if let Some(ref branch) = current_branch {
            // Query files for current branch only
            let mut stmt = conn.prepare(
                "SELECT f.language, COUNT(DISTINCT f.id)
                 FROM files f
                 JOIN file_branches fb ON f.id = fb.file_id
                 JOIN branches b ON fb.branch_id = b.id
                 WHERE b.name = ?
                 GROUP BY f.language"
            )?;
            let lang_counts = stmt.query_map([branch], |row| {
                let language: String = row.get(0)?;
                let count: i64 = row.get(1)?;
                Ok((language, count as usize))
            })?;

            for result in lang_counts {
                let (language, count) = result?;
                files_by_language.insert(language, count);
            }
        } else {
            // Fallback: query all files
            let mut stmt = conn.prepare("SELECT language, COUNT(*) FROM files GROUP BY language")?;
            let lang_counts = stmt.query_map([], |row| {
                let language: String = row.get(0)?;
                let count: i64 = row.get(1)?;
                Ok((language, count as usize))
            })?;

            for result in lang_counts {
                let (language, count) = result?;
                files_by_language.insert(language, count);
            }
        }

        // Get line count breakdown by language (branch-aware if possible)
        let mut lines_by_language = std::collections::HashMap::new();
        if let Some(ref branch) = current_branch {
            // Query lines for current branch only
            let mut stmt = conn.prepare(
                "SELECT f.language, SUM(f.line_count)
                 FROM files f
                 JOIN file_branches fb ON f.id = fb.file_id
                 JOIN branches b ON fb.branch_id = b.id
                 WHERE b.name = ?
                 GROUP BY f.language"
            )?;
            let line_counts = stmt.query_map([branch], |row| {
                let language: String = row.get(0)?;
                let count: i64 = row.get(1)?;
                Ok((language, count as usize))
            })?;

            for result in line_counts {
                let (language, count) = result?;
                lines_by_language.insert(language, count);
            }
        } else {
            // Fallback: query all files
            let mut stmt = conn.prepare("SELECT language, SUM(line_count) FROM files GROUP BY language")?;
            let line_counts = stmt.query_map([], |row| {
                let language: String = row.get(0)?;
                let count: i64 = row.get(1)?;
                Ok((language, count as usize))
            })?;

            for result in line_counts {
                let (language, count) = result?;
                lines_by_language.insert(language, count);
            }
        }

        Ok(crate::models::IndexStats {
            total_files,
            index_size_bytes,
            last_updated,
            files_by_language,
            lines_by_language,
        })
    }

    // ===== Branch-aware indexing methods =====

    /// Get or create a branch ID by name
    ///
    /// Returns the numeric branch ID, creating a new entry if needed.
    fn get_or_create_branch_id(&self, conn: &Connection, branch_name: &str, commit_sha: Option<&str>) -> Result<i64> {
        // Try to get existing branch
        let existing_id: Option<i64> = conn
            .query_row(
                "SELECT id FROM branches WHERE name = ?",
                [branch_name],
                |row| row.get(0),
            )
            .optional()?;

        if let Some(id) = existing_id {
            return Ok(id);
        }

        // Create new branch entry
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "INSERT INTO branches (name, commit_sha, last_indexed, file_count, is_dirty)
             VALUES (?, ?, ?, 0, 0)",
            [branch_name, commit_sha.unwrap_or("unknown"), &now.to_string()],
        )?;

        // Get the ID we just created
        let id: i64 = conn.last_insert_rowid();
        Ok(id)
    }

    /// Record a file's hash for a specific branch
    pub fn record_branch_file(
        &self,
        path: &str,
        branch: &str,
        hash: &str,
        commit_sha: Option<&str>,
    ) -> Result<()> {
        let db_path = self.cache_path.join(META_DB);
        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db for branch file recording")?;

        // Lookup file_id from path
        let file_id: i64 = conn.query_row(
            "SELECT id FROM files WHERE path = ?",
            [path],
            |row| row.get(0)
        ).context(format!("File not found in index: {}", path))?;

        // Get or create branch_id
        let branch_id = self.get_or_create_branch_id(&conn, branch, commit_sha)?;

        let now = chrono::Utc::now().timestamp();

        // Insert using proper INTEGER types (not strings!)
        conn.execute(
            "INSERT OR REPLACE INTO file_branches (file_id, branch_id, hash, last_indexed)
             VALUES (?, ?, ?, ?)",
            rusqlite::params![file_id, branch_id, hash, now],
        )?;

        Ok(())
    }

    /// Batch record multiple files for a specific branch in a single transaction
    ///
    /// IMPORTANT: Files must already exist in the `files` table before calling this method.
    /// For atomic insertion of both files and branch hashes, use `batch_update_files_and_branch()` instead.
    pub fn batch_record_branch_files(
        &self,
        files: &[(String, String)],  // (path, hash)
        branch: &str,
        commit_sha: Option<&str>,
    ) -> Result<()> {
        log::info!("batch_record_branch_files: Processing {} files for branch '{}'", files.len(), branch);

        let db_path = self.cache_path.join(META_DB);
        let mut conn = Connection::open(&db_path)
            .context("Failed to open meta.db for batch branch recording")?;

        let now = chrono::Utc::now().timestamp();

        // Use a transaction for batch inserts
        let tx = conn.transaction()?;

        // Get or create branch_id (use transaction connection)
        let branch_id = self.get_or_create_branch_id(&tx, branch, commit_sha)?;
        log::debug!("Got branch_id={} for branch '{}'", branch_id, branch);

        let mut inserted = 0;
        for (path, hash) in files {
            // Lookup file_id from path
            log::trace!("Looking up file_id for path: {}", path);
            let file_id: i64 = tx.query_row(
                "SELECT id FROM files WHERE path = ?",
                [path.as_str()],
                |row| row.get(0)
            ).context(format!("File not found in index: {}", path))?;
            log::trace!("Found file_id={} for path: {}", file_id, path);

            // Insert using proper INTEGER types (not strings!)
            tx.execute(
                "INSERT OR REPLACE INTO file_branches (file_id, branch_id, hash, last_indexed)
                 VALUES (?, ?, ?, ?)",
                rusqlite::params![file_id, branch_id, hash.as_str(), now],
            )?;
            inserted += 1;
        }

        log::info!("Inserted {} file_branches entries", inserted);
        tx.commit()?;
        log::info!("Transaction committed successfully");
        Ok(())
    }

    /// Get all files indexed for a specific branch
    ///
    /// Returns a HashMap of path → hash for all files in the branch.
    pub fn get_branch_files(&self, branch: &str) -> Result<HashMap<String, String>> {
        let db_path = self.cache_path.join(META_DB);

        if !db_path.exists() {
            return Ok(HashMap::new());
        }

        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db")?;

        let mut stmt = conn.prepare(
            "SELECT f.path, fb.hash
             FROM file_branches fb
             JOIN files f ON fb.file_id = f.id
             JOIN branches b ON fb.branch_id = b.id
             WHERE b.name = ?"
        )?;
        let files: HashMap<String, String> = stmt
            .query_map([branch], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<HashMap<_, _>, _>>()?;

        log::debug!(
            "Loaded {} files for branch '{}' from file_branches table",
            files.len(),
            branch
        );
        Ok(files)
    }

    /// Check if a branch has any indexed files
    ///
    /// Fast existence check using LIMIT 1 for O(1) performance.
    pub fn branch_exists(&self, branch: &str) -> Result<bool> {
        let db_path = self.cache_path.join(META_DB);

        if !db_path.exists() {
            return Ok(false);
        }

        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db")?;

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*)
                 FROM file_branches fb
                 JOIN branches b ON fb.branch_id = b.id
                 WHERE b.name = ?
                 LIMIT 1",
                [branch],
                |row| row.get(0),
            )
            .unwrap_or(0);

        Ok(count > 0)
    }

    /// Get branch metadata (commit, last_indexed, file_count, dirty status)
    pub fn get_branch_info(&self, branch: &str) -> Result<BranchInfo> {
        let db_path = self.cache_path.join(META_DB);

        if !db_path.exists() {
            anyhow::bail!("Database not initialized");
        }

        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db")?;

        let info = conn.query_row(
            "SELECT commit_sha, last_indexed, file_count, is_dirty FROM branches WHERE name = ?",
            [branch],
            |row| {
                Ok(BranchInfo {
                    branch: branch.to_string(),
                    commit_sha: row.get(0)?,
                    last_indexed: row.get(1)?,
                    file_count: row.get(2)?,
                    is_dirty: row.get::<_, i64>(3)? != 0,
                })
            },
        )?;

        Ok(info)
    }

    /// Update branch metadata after indexing
    ///
    /// Uses UPDATE instead of INSERT OR REPLACE to preserve branch_id and prevent
    /// CASCADE DELETE on file_branches table.
    pub fn update_branch_metadata(
        &self,
        branch: &str,
        commit_sha: Option<&str>,
        file_count: usize,
        is_dirty: bool,
    ) -> Result<()> {
        let db_path = self.cache_path.join(META_DB);
        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db for branch metadata update")?;

        let now = chrono::Utc::now().timestamp();
        let is_dirty_int = if is_dirty { 1 } else { 0 };

        // Try UPDATE first to preserve branch_id (prevents CASCADE DELETE)
        let rows_updated = conn.execute(
            "UPDATE branches
             SET commit_sha = ?, last_indexed = ?, file_count = ?, is_dirty = ?
             WHERE name = ?",
            rusqlite::params![
                commit_sha.unwrap_or("unknown"),
                now,
                file_count,
                is_dirty_int,
                branch
            ],
        )?;

        // If no rows updated (branch doesn't exist yet), INSERT new one
        if rows_updated == 0 {
            conn.execute(
                "INSERT INTO branches (name, commit_sha, last_indexed, file_count, is_dirty)
                 VALUES (?, ?, ?, ?, ?)",
                rusqlite::params![
                    branch,
                    commit_sha.unwrap_or("unknown"),
                    now,
                    file_count,
                    is_dirty_int
                ],
            )?;
        }

        log::debug!(
            "Updated branch metadata for '{}': commit={}, files={}, dirty={}",
            branch,
            commit_sha.unwrap_or("unknown"),
            file_count,
            is_dirty
        );
        Ok(())
    }

    /// Find a file with a specific hash (for symbol reuse optimization)
    ///
    /// Returns the path and branch where this hash was first seen,
    /// enabling reuse of parsed symbols across branches.
    pub fn find_file_with_hash(&self, hash: &str) -> Result<Option<(String, String)>> {
        let db_path = self.cache_path.join(META_DB);

        if !db_path.exists() {
            return Ok(None);
        }

        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db")?;

        let result = conn
            .query_row(
                "SELECT f.path, b.name
                 FROM file_branches fb
                 JOIN files f ON fb.file_id = f.id
                 JOIN branches b ON fb.branch_id = b.id
                 WHERE fb.hash = ?
                 LIMIT 1",
                [hash],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;

        Ok(result)
    }

    /// Get file ID by path
    ///
    /// Returns the integer ID for a file path, or None if not found.
    pub fn get_file_id(&self, path: &str) -> Result<Option<i64>> {
        let db_path = self.cache_path.join(META_DB);

        if !db_path.exists() {
            return Ok(None);
        }

        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db")?;

        let result = conn
            .query_row(
                "SELECT id FROM files WHERE path = ?",
                [path],
                |row| row.get(0),
            )
            .optional()?;

        Ok(result)
    }

    /// Batch get file IDs for multiple paths
    ///
    /// Returns a HashMap of path → file_id for all found paths.
    /// Paths not in the database are omitted from the result.
    ///
    /// Automatically chunks large batches to avoid SQLite parameter limits (999 max).
    pub fn batch_get_file_ids(&self, paths: &[String]) -> Result<HashMap<String, i64>> {
        let db_path = self.cache_path.join(META_DB);

        if !db_path.exists() {
            return Ok(HashMap::new());
        }

        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db")?;

        // SQLite has a limit of 999 parameters by default
        // Chunk requests to stay well under that limit
        const BATCH_SIZE: usize = 900;

        let mut results = HashMap::new();

        for chunk in paths.chunks(BATCH_SIZE) {
            // Build IN clause for this chunk
            let placeholders = chunk.iter()
                .map(|_| "?")
                .collect::<Vec<_>>()
                .join(", ");

            let query = format!("SELECT path, id FROM files WHERE path IN ({})", placeholders);

            let params: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
            let mut stmt = conn.prepare(&query)?;

            let chunk_results = stmt.query_map(rusqlite::params_from_iter(params), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })?
            .collect::<Result<HashMap<_, _>, _>>()?;

            results.extend(chunk_results);
        }

        log::debug!("Batch loaded {} file IDs (out of {} requested, {} chunks)",
                   results.len(), paths.len(), paths.len().div_ceil(BATCH_SIZE));
        Ok(results)
    }

    // ===== Cache compaction methods =====

    /// Check if cache compaction should run
    ///
    /// Returns true if 24+ hours have passed since last compaction (or never compacted).
    /// Compaction threshold: 86400 seconds (24 hours)
    pub fn should_compact(&self) -> Result<bool> {
        let db_path = self.cache_path.join(META_DB);

        if !db_path.exists() {
            // No database means no compaction needed
            return Ok(false);
        }

        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db for compaction check")?;

        // Get last_compaction timestamp (defaults to "0" if not found)
        let last_compaction: i64 = conn
            .query_row(
                "SELECT value FROM statistics WHERE key = 'last_compaction'",
                [],
                |row| {
                    let value: String = row.get(0)?;
                    Ok(value.parse::<i64>().unwrap_or(0))
                },
            )
            .unwrap_or(0);

        // Get current timestamp
        let now = chrono::Utc::now().timestamp();

        // Compaction threshold: 24 hours (86400 seconds)
        const COMPACTION_THRESHOLD_SECS: i64 = 86400;

        let elapsed_secs = now - last_compaction;
        let should_run = elapsed_secs >= COMPACTION_THRESHOLD_SECS;

        log::debug!(
            "Compaction check: last={}, now={}, elapsed={}s, should_compact={}",
            last_compaction,
            now,
            elapsed_secs,
            should_run
        );

        Ok(should_run)
    }

    /// Update last_compaction timestamp in statistics table
    ///
    /// Called after successful compaction to record when it ran.
    pub fn update_compaction_timestamp(&self) -> Result<()> {
        let db_path = self.cache_path.join(META_DB);
        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db for compaction timestamp update")?;

        let now = chrono::Utc::now().timestamp();

        conn.execute(
            "INSERT OR REPLACE INTO statistics (key, value, updated_at) VALUES (?, ?, ?)",
            ["last_compaction", &now.to_string(), &now.to_string()],
        )?;

        log::debug!("Updated last_compaction timestamp to: {}", now);
        Ok(())
    }

    /// Compact the cache by removing deleted files and reclaiming disk space
    ///
    /// This operation:
    /// 1. Identifies files in the database that no longer exist on disk
    /// 2. Deletes those files from all database tables (CASCADE handles related data)
    /// 3. Runs VACUUM to reclaim disk space from deleted rows
    /// 4. Updates the last_compaction timestamp
    ///
    /// Returns a CompactionReport with statistics about the operation.
    /// Safe to run concurrently with queries (uses SQLite transactions).
    pub fn compact(&self) -> Result<crate::models::CompactionReport> {
        let start_time = std::time::Instant::now();
        log::info!("Starting cache compaction...");

        // Get initial cache size
        let size_before = self.calculate_cache_size()?;

        // Step 1: Identify deleted files (in DB but not on filesystem)
        let deleted_files = self.identify_deleted_files()?;
        log::info!("Found {} deleted files to remove from cache", deleted_files.len());

        if deleted_files.is_empty() {
            log::info!("No deleted files to compact - cache is clean");
            // Update timestamp anyway to prevent running compaction too frequently
            self.update_compaction_timestamp()?;

            return Ok(crate::models::CompactionReport {
                files_removed: 0,
                space_saved_bytes: 0,
                duration_ms: start_time.elapsed().as_millis() as u64,
            });
        }

        // Step 2: Delete from database (CASCADE handles file_branches, file_dependencies, file_exports)
        self.delete_files_from_db(&deleted_files)?;
        log::info!("Deleted {} files from database", deleted_files.len());

        // Step 3: Run VACUUM to reclaim disk space
        self.vacuum_database()?;
        log::info!("Completed VACUUM operation");

        // Get final cache size
        let size_after = self.calculate_cache_size()?;
        let space_saved = size_before.saturating_sub(size_after);

        // Step 4: Update last_compaction timestamp
        self.update_compaction_timestamp()?;

        let duration_ms = start_time.elapsed().as_millis() as u64;

        log::info!(
            "Cache compaction completed: {} files removed, {} bytes saved ({:.2} MB), took {}ms",
            deleted_files.len(),
            space_saved,
            space_saved as f64 / 1_048_576.0,
            duration_ms
        );

        Ok(crate::models::CompactionReport {
            files_removed: deleted_files.len(),
            space_saved_bytes: space_saved,
            duration_ms,
        })
    }

    /// Identify files in database that no longer exist on filesystem
    ///
    /// Returns a Vec of file IDs for files that should be removed from the cache.
    fn identify_deleted_files(&self) -> Result<Vec<i64>> {
        let db_path = self.cache_path.join(META_DB);
        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db for deleted file identification")?;

        let workspace_root = self.workspace_root();

        // Query all files from database (id, path)
        let mut stmt = conn.prepare("SELECT id, path FROM files")?;
        let files = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;

        log::debug!("Checking {} files for deletion status", files.len());

        // Check which files no longer exist on disk
        let mut deleted_file_ids = Vec::new();
        for (file_id, file_path) in files {
            let full_path = workspace_root.join(&file_path);
            if !full_path.exists() {
                log::trace!("File no longer exists: {} (id={})", file_path, file_id);
                deleted_file_ids.push(file_id);
            }
        }

        Ok(deleted_file_ids)
    }

    /// Delete files from database by file ID
    ///
    /// Uses a transaction for atomicity. CASCADE delete handles:
    /// - file_branches entries
    /// - file_dependencies entries
    /// - file_exports entries
    fn delete_files_from_db(&self, file_ids: &[i64]) -> Result<()> {
        if file_ids.is_empty() {
            return Ok(());
        }

        let db_path = self.cache_path.join(META_DB);
        let mut conn = Connection::open(&db_path)
            .context("Failed to open meta.db for file deletion")?;

        let tx = conn.transaction()?;

        // Delete files in batches to avoid SQLite parameter limit (999 max)
        const BATCH_SIZE: usize = 900;

        for chunk in file_ids.chunks(BATCH_SIZE) {
            let placeholders = chunk.iter()
                .map(|_| "?")
                .collect::<Vec<_>>()
                .join(", ");

            let delete_query = format!("DELETE FROM files WHERE id IN ({})", placeholders);

            let params: Vec<i64> = chunk.to_vec();
            tx.execute(&delete_query, rusqlite::params_from_iter(params))?;
        }

        tx.commit()?;
        log::debug!("Deleted {} files from database (CASCADE handled related tables)", file_ids.len());
        Ok(())
    }

    /// Run VACUUM on SQLite database to reclaim disk space
    ///
    /// VACUUM rebuilds the database file, removing free pages and compacting the file.
    /// This can take several seconds on large databases but significantly reduces disk usage.
    fn vacuum_database(&self) -> Result<()> {
        let db_path = self.cache_path.join(META_DB);
        let conn = Connection::open(&db_path)
            .context("Failed to open meta.db for VACUUM")?;

        // VACUUM cannot run inside a transaction
        // It rebuilds the entire database file
        conn.execute("VACUUM", [])?;

        log::debug!("VACUUM completed successfully");
        Ok(())
    }

    /// Calculate total cache size in bytes
    ///
    /// Sums up the size of all cache files:
    /// - meta.db (SQLite database)
    /// - trigrams.bin (inverted index)
    /// - content.bin (file contents)
    /// - config.toml (configuration)
    fn calculate_cache_size(&self) -> Result<u64> {
        let mut total_size: u64 = 0;

        for file_name in [META_DB, TOKENS_BIN, CONFIG_TOML, "content.bin", "trigrams.bin"] {
            let file_path = self.cache_path.join(file_name);
            if let Ok(metadata) = std::fs::metadata(&file_path) {
                total_size += metadata.len();
            }
        }

        Ok(total_size)
    }
}

/// Branch metadata information
#[derive(Debug, Clone)]
pub struct BranchInfo {
    pub branch: String,
    pub commit_sha: String,
    pub last_indexed: i64,
    pub file_count: usize,
    pub is_dirty: bool,
}

// TODO: Implement memory-mapped readers for:
// - SymbolReader (reads from symbols.bin)
// - TokenReader (reads from tokens.bin)
// - MetaReader (reads from meta.db)

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_cache_init() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        assert!(!cache.exists());
        cache.init().unwrap();
        assert!(cache.exists());
        assert!(cache.path().exists());

        // Verify all expected files were created
        assert!(cache.path().join(META_DB).exists());
        assert!(cache.path().join(CONFIG_TOML).exists());
    }

    #[test]
    fn test_cache_init_idempotent() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        // Initialize twice - should not error
        cache.init().unwrap();
        cache.init().unwrap();

        assert!(cache.exists());
    }

    #[test]
    fn test_cache_clear() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();
        assert!(cache.exists());

        cache.clear().unwrap();
        assert!(!cache.exists());
    }

    #[test]
    fn test_cache_clear_nonexistent() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        // Clearing non-existent cache should not error
        assert!(!cache.exists());
        cache.clear().unwrap();
        assert!(!cache.exists());
    }

    #[test]
    fn test_load_all_hashes_empty() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();
        let hashes = cache.load_all_hashes().unwrap();
        assert_eq!(hashes.len(), 0);
    }

    #[test]
    fn test_load_all_hashes_before_init() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        // Loading hashes before init should return empty map
        let hashes = cache.load_all_hashes().unwrap();
        assert_eq!(hashes.len(), 0);
    }

    #[test]
    fn test_load_hashes_for_branch_empty() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();
        let hashes = cache.load_hashes_for_branch("main").unwrap();
        assert_eq!(hashes.len(), 0);
    }

    #[test]
    fn test_update_file() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();
        cache.update_file("src/main.rs", "rust", 100).unwrap();

        // Verify file was stored (check via list_files)
        let files = cache.list_files().unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "src/main.rs");
        assert_eq!(files[0].language, "rust");
    }

    #[test]
    fn test_update_file_multiple() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();
        cache.update_file("src/main.rs", "rust", 100).unwrap();
        cache.update_file("src/lib.rs", "rust", 200).unwrap();
        cache.update_file("README.md", "markdown", 50).unwrap();

        // Verify files were stored
        let files = cache.list_files().unwrap();
        assert_eq!(files.len(), 3);
    }

    #[test]
    fn test_update_file_replace() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();
        cache.update_file("src/main.rs", "rust", 100).unwrap();
        cache.update_file("src/main.rs", "rust", 150).unwrap();

        // Second update should replace the first
        let files = cache.list_files().unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "src/main.rs");
    }

    #[test]
    fn test_batch_update_files() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();

        let files = vec![
            ("src/main.rs".to_string(), "rust".to_string(), 100),
            ("src/lib.rs".to_string(), "rust".to_string(), 200),
            ("test.py".to_string(), "python".to_string(), 50),
        ];

        cache.batch_update_files(&files).unwrap();

        // Verify files were stored
        let stored_files = cache.list_files().unwrap();
        assert_eq!(stored_files.len(), 3);
    }

    #[test]
    fn test_update_stats() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();
        cache.update_file("src/main.rs", "rust", 100).unwrap();
        cache.update_file("src/lib.rs", "rust", 200).unwrap();

        // Record files for a test branch
        cache.record_branch_file("src/main.rs", "_default", "hash1", None).unwrap();
        cache.record_branch_file("src/lib.rs", "_default", "hash2", None).unwrap();
        cache.update_stats("_default").unwrap();

        let stats = cache.stats().unwrap();
        assert_eq!(stats.total_files, 2);
    }

    #[test]
    fn test_stats_empty_cache() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();
        let stats = cache.stats().unwrap();

        assert_eq!(stats.total_files, 0);
        assert_eq!(stats.files_by_language.len(), 0);
    }

    #[test]
    fn test_stats_before_init() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        // Stats before init should return zeros
        let stats = cache.stats().unwrap();
        assert_eq!(stats.total_files, 0);
    }

    #[test]
    fn test_stats_by_language() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();
        cache.update_file("main.rs", "Rust", 100).unwrap();
        cache.update_file("lib.rs", "Rust", 200).unwrap();
        cache.update_file("script.py", "Python", 50).unwrap();
        cache.update_file("test.py", "Python", 80).unwrap();

        // Record files for a test branch
        cache.record_branch_file("main.rs", "_default", "hash1", None).unwrap();
        cache.record_branch_file("lib.rs", "_default", "hash2", None).unwrap();
        cache.record_branch_file("script.py", "_default", "hash3", None).unwrap();
        cache.record_branch_file("test.py", "_default", "hash4", None).unwrap();
        cache.update_stats("_default").unwrap();

        let stats = cache.stats().unwrap();
        assert_eq!(stats.files_by_language.get("Rust"), Some(&2));
        assert_eq!(stats.files_by_language.get("Python"), Some(&2));
        assert_eq!(stats.lines_by_language.get("Rust"), Some(&300)); // 100 + 200
        assert_eq!(stats.lines_by_language.get("Python"), Some(&130)); // 50 + 80
    }

    #[test]
    fn test_list_files_empty() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();
        let files = cache.list_files().unwrap();
        assert_eq!(files.len(), 0);
    }

    #[test]
    fn test_list_files() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();
        cache.update_file("src/main.rs", "rust", 100).unwrap();
        cache.update_file("src/lib.rs", "rust", 200).unwrap();

        let files = cache.list_files().unwrap();
        assert_eq!(files.len(), 2);

        // Files should be sorted by path
        assert_eq!(files[0].path, "src/lib.rs");
        assert_eq!(files[1].path, "src/main.rs");

        assert_eq!(files[0].language, "rust");
    }

    #[test]
    fn test_list_files_before_init() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        // Listing files before init should return empty vec
        let files = cache.list_files().unwrap();
        assert_eq!(files.len(), 0);
    }

    #[test]
    fn test_branch_exists() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();

        assert!(!cache.branch_exists("main").unwrap());

        // Add file to index first (required for record_branch_file)
        cache.update_file("src/main.rs", "rust", 100).unwrap();
        cache.record_branch_file("src/main.rs", "main", "hash1", Some("commit123")).unwrap();

        assert!(cache.branch_exists("main").unwrap());
        assert!(!cache.branch_exists("feature-branch").unwrap());
    }

    #[test]
    fn test_record_branch_file() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();
        // Add file to index first (required for record_branch_file)
        cache.update_file("src/main.rs", "rust", 100).unwrap();
        cache.record_branch_file("src/main.rs", "main", "hash1", Some("commit123")).unwrap();

        let files = cache.get_branch_files("main").unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files.get("src/main.rs"), Some(&"hash1".to_string()));
    }

    #[test]
    fn test_get_branch_files_empty() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();
        let files = cache.get_branch_files("nonexistent").unwrap();
        assert_eq!(files.len(), 0);
    }

    #[test]
    fn test_batch_record_branch_files() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();

        // Add files to index first (required for batch_record_branch_files)
        let file_metadata = vec![
            ("src/main.rs".to_string(), "rust".to_string(), 100),
            ("src/lib.rs".to_string(), "rust".to_string(), 200),
            ("README.md".to_string(), "markdown".to_string(), 50),
        ];
        cache.batch_update_files(&file_metadata).unwrap();

        let files = vec![
            ("src/main.rs".to_string(), "hash1".to_string()),
            ("src/lib.rs".to_string(), "hash2".to_string()),
            ("README.md".to_string(), "hash3".to_string()),
        ];

        cache.batch_record_branch_files(&files, "main", Some("commit123")).unwrap();

        let branch_files = cache.get_branch_files("main").unwrap();
        assert_eq!(branch_files.len(), 3);
        assert_eq!(branch_files.get("src/main.rs"), Some(&"hash1".to_string()));
        assert_eq!(branch_files.get("src/lib.rs"), Some(&"hash2".to_string()));
        assert_eq!(branch_files.get("README.md"), Some(&"hash3".to_string()));
    }

    #[test]
    fn test_update_branch_metadata() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();
        cache.update_branch_metadata("main", Some("commit123"), 10, false).unwrap();

        let info = cache.get_branch_info("main").unwrap();
        assert_eq!(info.branch, "main");
        assert_eq!(info.commit_sha, "commit123");
        assert_eq!(info.file_count, 10);
        assert_eq!(info.is_dirty, false);
    }

    #[test]
    fn test_update_branch_metadata_dirty() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();
        cache.update_branch_metadata("feature", Some("commit456"), 5, true).unwrap();

        let info = cache.get_branch_info("feature").unwrap();
        assert_eq!(info.is_dirty, true);
    }

    #[test]
    fn test_find_file_with_hash() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();
        // Add file to index first (required for record_branch_file)
        cache.update_file("src/main.rs", "rust", 100).unwrap();
        cache.record_branch_file("src/main.rs", "main", "unique_hash", Some("commit123")).unwrap();

        let result = cache.find_file_with_hash("unique_hash").unwrap();
        assert!(result.is_some());

        let (path, branch) = result.unwrap();
        assert_eq!(path, "src/main.rs");
        assert_eq!(branch, "main");
    }

    #[test]
    fn test_find_file_with_hash_not_found() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();

        let result = cache.find_file_with_hash("nonexistent_hash").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_config_toml_created() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();

        let config_path = cache.path().join(CONFIG_TOML);
        let config_content = std::fs::read_to_string(&config_path).unwrap();

        // Verify config contains expected sections
        assert!(config_content.contains("[index]"));
        assert!(config_content.contains("[search]"));
        assert!(config_content.contains("[performance]"));
        assert!(config_content.contains("max_file_size"));
    }

    #[test]
    fn test_meta_db_schema() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();

        let db_path = cache.path().join(META_DB);
        let conn = Connection::open(&db_path).unwrap();

        // Verify tables exist
        let tables: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table'").unwrap()
            .query_map([], |row| row.get(0)).unwrap()
            .collect::<Result<Vec<_>, _>>().unwrap();

        assert!(tables.contains(&"files".to_string()));
        assert!(tables.contains(&"statistics".to_string()));
        assert!(tables.contains(&"config".to_string()));
        assert!(tables.contains(&"file_branches".to_string()));
        assert!(tables.contains(&"branches".to_string()));
        assert!(tables.contains(&"file_dependencies".to_string()));
        assert!(tables.contains(&"file_exports".to_string()));
    }

    #[test]
    fn test_concurrent_file_updates() {
        use std::thread;

        let temp = TempDir::new().unwrap();
        let cache_path = temp.path().to_path_buf();

        let cache = CacheManager::new(&cache_path);
        cache.init().unwrap();

        // Spawn multiple threads updating different files
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let path = cache_path.clone();
                thread::spawn(move || {
                    let cache = CacheManager::new(&path);
                    cache
                        .update_file(
                            &format!("file_{}.rs", i),
                            "rust",
                            i * 10,
                        )
                        .unwrap();
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let cache = CacheManager::new(&cache_path);
        let files = cache.list_files().unwrap();
        assert_eq!(files.len(), 10);
    }

    // ===== Corruption Detection Tests =====

    #[test]
    fn test_validate_corrupted_database() {
        use std::io::Write;

        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();

        // Corrupt the database by overwriting it with invalid data
        let db_path = cache.path().join(META_DB);
        let mut file = File::create(&db_path).unwrap();
        file.write_all(b"CORRUPTED DATA").unwrap();

        // Validation should fail due to database corruption
        let result = cache.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        eprintln!("Error message: {}", err_msg);
        assert!(err_msg.contains("corrupted") || err_msg.contains("not a database"));
    }

    #[test]
    fn test_validate_corrupted_trigrams() {
        use std::io::Write;

        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();

        // Create trigrams.bin with invalid magic bytes
        let trigrams_path = cache.path().join("trigrams.bin");
        let mut file = File::create(&trigrams_path).unwrap();
        file.write_all(b"BADM").unwrap(); // Wrong magic bytes (should be "RFTG")

        // Validation should fail due to invalid magic bytes
        let result = cache.validate();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("trigrams.bin") && err.contains("corrupted"));
    }

    #[test]
    fn test_validate_corrupted_content() {
        use std::io::Write;

        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();

        // Create content.bin with invalid magic bytes
        let content_path = cache.path().join("content.bin");
        let mut file = File::create(&content_path).unwrap();
        file.write_all(b"BADM").unwrap(); // Wrong magic bytes (should be "RFCT")

        // Validation should fail due to invalid magic bytes
        let result = cache.validate();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("content.bin") && err.contains("corrupted"));
    }

    #[test]
    fn test_validate_missing_schema_table() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        cache.init().unwrap();

        // Drop a required table to simulate schema corruption
        let db_path = cache.path().join(META_DB);
        let conn = Connection::open(&db_path).unwrap();
        conn.execute("DROP TABLE files", []).unwrap();

        // Validation should fail due to missing required table
        let result = cache.validate();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("files") && err.contains("missing"));
    }
}
