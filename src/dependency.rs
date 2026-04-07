//! Dependency tracking and graph analysis
//!
//! This module provides functionality for tracking file dependencies (imports/includes)
//! and analyzing the dependency graph of a codebase.
//!
//! # Architecture
//!
//! The system uses a "depth-1 storage" approach:
//! - Only direct dependencies are stored in the database
//! - Deeper relationships are computed on-demand via graph traversal
//! - This provides O(n) storage while enabling any-depth queries
//!
//! # Example
//!
//! ```no_run
//! use reflex::dependency::DependencyIndex;
//! use reflex::cache::CacheManager;
//!
//! let cache = CacheManager::new(".");
//! let deps = DependencyIndex::new(cache);
//!
//! // Get direct dependencies of a file
//! let file_deps = deps.get_dependencies(42)?;
//!
//! // Get files that import this file (reverse lookup)
//! let dependents = deps.get_dependents(42)?;
//!
//! // Traverse dependency graph to depth 3
//! let transitive = deps.get_transitive_deps(42, 3)?;
//! # Ok::<(), anyhow::Error>(())
//! ```

use anyhow::{Context, Result};
use rusqlite::Connection;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;

use crate::cache::CacheManager;
use crate::models::{Dependency, DependencyInfo, ImportType};

/// Manages dependency storage and graph operations
pub struct DependencyIndex {
    cache: Option<CacheManager>,
    db_path: PathBuf,
}

impl DependencyIndex {
    /// Create a new dependency index for the given cache
    pub fn new(cache: CacheManager) -> Self {
        let db_path = cache.path().join("meta.db");
        Self { cache: Some(cache), db_path }
    }

    /// Create a dependency index pointing directly at a database file.
    ///
    /// Used by Pulse to run analysis against snapshot databases.
    pub fn from_db_path(db_path: impl Into<PathBuf>) -> Self {
        Self { cache: None, db_path: db_path.into() }
    }

    /// Get a reference to the cache manager.
    ///
    /// Panics if this index was created via `from_db_path()`.
    pub fn get_cache(&self) -> &CacheManager {
        self.cache.as_ref().expect("DependencyIndex created with from_db_path has no CacheManager")
    }

    /// Open a database connection to the backing store.
    fn open_conn(&self) -> Result<Connection> {
        Connection::open(&self.db_path).context("Failed to open database")
    }

    /// Insert a dependency into the database
    ///
    /// # Arguments
    ///
    /// * `file_id` - Source file ID
    /// * `imported_path` - Import path as written in source
    /// * `resolved_file_id` - Resolved target file ID (None if external/stdlib)
    /// * `import_type` - Type of import (internal/external/stdlib)
    /// * `line_number` - Line where import appears
    /// * `imported_symbols` - Optional list of imported symbols
    pub fn insert_dependency(
        &self,
        file_id: i64,
        imported_path: String,
        resolved_file_id: Option<i64>,
        import_type: ImportType,
        line_number: usize,
        imported_symbols: Option<Vec<String>>,
    ) -> Result<()> {
        let conn = self.open_conn()?;

        let import_type_str = match import_type {
            ImportType::Internal => "internal",
            ImportType::External => "external",
            ImportType::Stdlib => "stdlib",
        };

        let symbols_json = imported_symbols
            .as_ref()
            .map(|syms| serde_json::to_string(syms).unwrap_or_else(|_| "[]".to_string()));

        conn.execute(
            "INSERT INTO file_dependencies (file_id, imported_path, resolved_file_id, import_type, line_number, imported_symbols)
             VALUES (?, ?, ?, ?, ?, ?)",
            rusqlite::params![
                file_id,
                imported_path,
                resolved_file_id,
                import_type_str,
                line_number as i64,
                symbols_json,
            ],
        )?;

        Ok(())
    }

    /// Insert an export into the database
    ///
    /// # Arguments
    ///
    /// * `file_id` - Source file ID containing the export statement
    /// * `exported_symbol` - Symbol name being exported (None for wildcard exports)
    /// * `source_path` - Path where the symbol is re-exported from
    /// * `resolved_source_id` - Resolved target file ID (None if unresolved)
    /// * `line_number` - Line where export appears
    pub fn insert_export(
        &self,
        file_id: i64,
        exported_symbol: Option<String>,
        source_path: String,
        resolved_source_id: Option<i64>,
        line_number: usize,
    ) -> Result<()> {
        let conn = self.open_conn()?;

        conn.execute(
            "INSERT INTO file_exports (file_id, exported_symbol, source_path, resolved_source_id, line_number)
             VALUES (?, ?, ?, ?, ?)",
            rusqlite::params![
                file_id,
                exported_symbol,
                source_path,
                resolved_source_id,
                line_number as i64,
            ],
        )?;

        Ok(())
    }

    /// Batch insert multiple dependencies in a single transaction
    ///
    /// More efficient than individual inserts for bulk operations.
    pub fn batch_insert_dependencies(&self, dependencies: &[Dependency]) -> Result<()> {
        if dependencies.is_empty() {
            return Ok(());
        }

        let mut conn = self.open_conn()?;

        let tx = conn.transaction()?;

        for dep in dependencies {
            let import_type_str = match dep.import_type {
                ImportType::Internal => "internal",
                ImportType::External => "external",
                ImportType::Stdlib => "stdlib",
            };

            let symbols_json = dep
                .imported_symbols
                .as_ref()
                .map(|syms| serde_json::to_string(syms).unwrap_or_else(|_| "[]".to_string()));

            tx.execute(
                "INSERT INTO file_dependencies (file_id, imported_path, resolved_file_id, import_type, line_number, imported_symbols)
                 VALUES (?, ?, ?, ?, ?, ?)",
                rusqlite::params![
                    dep.file_id,
                    dep.imported_path,
                    dep.resolved_file_id,
                    import_type_str,
                    dep.line_number as i64,
                    symbols_json,
                ],
            )?;
        }

        tx.commit()?;
        log::debug!("Batch inserted {} dependencies", dependencies.len());
        Ok(())
    }

    /// Get all direct dependencies for a file
    ///
    /// Returns a list of files/modules that this file imports.
    pub fn get_dependencies(&self, file_id: i64) -> Result<Vec<Dependency>> {
        let conn = self.open_conn()?;

        let mut stmt = conn.prepare(
            "SELECT file_id, imported_path, resolved_file_id, import_type, line_number, imported_symbols
             FROM file_dependencies
             WHERE file_id = ?
             ORDER BY line_number",
        )?;

        let deps = stmt
            .query_map([file_id], |row| {
                let import_type_str: String = row.get(3)?;
                let import_type = match import_type_str.as_str() {
                    "internal" => ImportType::Internal,
                    "external" => ImportType::External,
                    "stdlib" => ImportType::Stdlib,
                    _ => ImportType::External,
                };

                let symbols_json: Option<String> = row.get(5)?;
                let imported_symbols = symbols_json.and_then(|json| {
                    serde_json::from_str(&json).ok()
                });

                Ok(Dependency {
                    file_id: row.get(0)?,
                    imported_path: row.get(1)?,
                    resolved_file_id: row.get(2)?,
                    import_type,
                    line_number: row.get::<_, i64>(4)? as usize,
                    imported_symbols,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(deps)
    }

    /// Get all files that depend on this file (reverse lookup)
    ///
    /// Returns a list of file IDs that import this file.
    /// Uses `resolved_file_id` column for instant SQL lookup (sub-10ms).
    pub fn get_dependents(&self, file_id: i64) -> Result<Vec<i64>> {
        let conn = self.open_conn()?;

        // Pure SQL query on resolved_file_id (instant)
        let mut stmt = conn.prepare(
            "SELECT DISTINCT file_id
             FROM file_dependencies
             WHERE resolved_file_id = ?
             ORDER BY file_id"
        )?;

        let dependents: Vec<i64> = stmt
            .query_map([file_id], |row| row.get(0))?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(dependents)
    }

    /// Get dependencies as DependencyInfo (for API output)
    ///
    /// Converts internal Dependency records to simplified DependencyInfo
    /// suitable for JSON output.
    pub fn get_dependencies_info(&self, file_id: i64) -> Result<Vec<DependencyInfo>> {
        let deps = self.get_dependencies(file_id)?;

        let dep_infos = deps
            .into_iter()
            .map(|dep| {
                // Try to get the resolved path (all deps are internal now)
                let path = if let Some(resolved_id) = dep.resolved_file_id {
                    // Try to get the actual file path
                    self.get_file_path(resolved_id).unwrap_or(dep.imported_path)
                } else {
                    dep.imported_path
                };

                DependencyInfo {
                    path,
                    line: Some(dep.line_number),
                    symbols: dep.imported_symbols,
                }
            })
            .collect();

        Ok(dep_infos)
    }

    /// Get transitive dependencies up to a given depth
    ///
    /// Traverses the dependency graph using BFS to find all dependencies
    /// reachable within the specified depth.
    /// Uses `resolved_file_id` column for instant SQL lookup (sub-100ms).
    ///
    /// # Arguments
    ///
    /// * `file_id` - Starting file ID
    /// * `max_depth` - Maximum traversal depth (0 = only direct deps)
    ///
    /// # Returns
    ///
    /// HashMap mapping file_id to depth (distance from start file)
    pub fn get_transitive_deps(&self, file_id: i64, max_depth: usize) -> Result<HashMap<i64, usize>> {
        let mut visited = HashMap::new();
        let mut queue = VecDeque::new();

        // Start with the initial file at depth 0
        queue.push_back((file_id, 0));
        visited.insert(file_id, 0);

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            // Get direct dependencies using resolved_file_id (instant)
            let deps = self.get_dependencies(current_id)?;

            for dep in deps {
                // Use resolved_file_id directly (already populated during indexing)
                if let Some(resolved_id) = dep.resolved_file_id {
                    // Only visit if we haven't seen it or found a shorter path
                    if !visited.contains_key(&resolved_id) {
                        visited.insert(resolved_id, depth + 1);
                        queue.push_back((resolved_id, depth + 1));
                    }
                }
            }
        }

        Ok(visited)
    }

    /// Detect circular dependencies in the entire codebase
    ///
    /// Uses depth-first search to find cycles in the dependency graph.
    /// Uses `resolved_file_id` column for instant SQL lookup (sub-100ms).
    ///
    /// Returns a list of cycle paths, where each cycle is represented as
    /// a vector of file IDs forming the cycle.
    pub fn detect_circular_dependencies(&self) -> Result<Vec<Vec<i64>>> {
        let conn = self.open_conn()?;

        // Build in-memory dependency graph using resolved_file_id (instant)
        let mut graph: HashMap<i64, Vec<i64>> = HashMap::new();

        let mut stmt = conn.prepare(
            "SELECT file_id, resolved_file_id
             FROM file_dependencies
             WHERE resolved_file_id IS NOT NULL"
        )?;

        let dependencies: Vec<(i64, i64)> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        // Build adjacency list directly from resolved IDs
        for (file_id, target_id) in dependencies {
            graph.entry(file_id).or_insert_with(Vec::new).push(target_id);
        }

        // Get all file IDs for traversal
        let all_files = self.get_all_file_ids()?;

        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut path = Vec::new();
        let mut cycles = Vec::new();

        for file_id in all_files {
            if !visited.contains(&file_id) {
                self.dfs_cycle_detect(
                    file_id,
                    &graph,
                    &mut visited,
                    &mut rec_stack,
                    &mut path,
                    &mut cycles,
                )?;
            }
        }

        Ok(cycles)
    }

    /// DFS helper for cycle detection using pre-built graph
    fn dfs_cycle_detect(
        &self,
        file_id: i64,
        graph: &HashMap<i64, Vec<i64>>,
        visited: &mut HashSet<i64>,
        rec_stack: &mut HashSet<i64>,
        path: &mut Vec<i64>,
        cycles: &mut Vec<Vec<i64>>,
    ) -> Result<()> {
        visited.insert(file_id);
        rec_stack.insert(file_id);
        path.push(file_id);

        // Get dependencies from the pre-built graph
        if let Some(dependencies) = graph.get(&file_id) {
            for &target_id in dependencies {
                if !visited.contains(&target_id) {
                    self.dfs_cycle_detect(target_id, graph, visited, rec_stack, path, cycles)?;
                } else if rec_stack.contains(&target_id) {
                    // Found a cycle! Extract it from path
                    if let Some(cycle_start) = path.iter().position(|&id| id == target_id) {
                        let cycle = path[cycle_start..].to_vec();
                        cycles.push(cycle);
                    }
                }
            }
        }

        path.pop();
        rec_stack.remove(&file_id);

        Ok(())
    }

    /// Get file paths for a list of file IDs
    ///
    /// Useful for converting file ID results to human-readable paths.
    pub fn get_file_paths(&self, file_ids: &[i64]) -> Result<HashMap<i64, String>> {
        let conn = self.open_conn()?;

        let mut paths = HashMap::new();

        for &file_id in file_ids {
            if let Ok(path) = conn.query_row(
                "SELECT path FROM files WHERE id = ?",
                [file_id],
                |row| row.get::<_, String>(0),
            ) {
                paths.insert(file_id, path);
            }
        }

        Ok(paths)
    }

    /// Get file path for a single file ID
    fn get_file_path(&self, file_id: i64) -> Result<String> {
        let conn = self.open_conn()?;

        let path = conn.query_row(
            "SELECT path FROM files WHERE id = ?",
            [file_id],
            |row| row.get::<_, String>(0),
        )?;

        Ok(path)
    }

    /// Get all file IDs in the database
    fn get_all_file_ids(&self) -> Result<Vec<i64>> {
        let conn = self.open_conn()?;

        let mut stmt = conn.prepare("SELECT id FROM files")?;
        let file_ids = stmt
            .query_map([], |row| row.get(0))?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(file_ids)
    }

    /// Find hotspots (most imported files)
    ///
    /// Returns a list of (file_id, count) tuples sorted by import count descending.
    ///
    /// Uses `resolved_file_id` column for instant SQL aggregation (sub-100ms).
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of hotspots to return (None = all)
    /// * `min_dependents` - Minimum number of imports required to be a hotspot (default: 2)
    pub fn find_hotspots(&self, limit: Option<usize>, min_dependents: usize) -> Result<Vec<(i64, usize)>> {
        let conn = self.open_conn()?;

        // Pure SQL aggregation on resolved_file_id (instant)
        let mut stmt = conn.prepare(
            "SELECT resolved_file_id, COUNT(*) as count
             FROM file_dependencies
             WHERE resolved_file_id IS NOT NULL
             GROUP BY resolved_file_id
             ORDER BY count DESC"
        )?;

        // Get all hotspots and filter by minimum dependent count
        let mut hotspots: Vec<(i64, usize)> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)? as usize))
            })?
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|(_, count)| *count >= min_dependents)
            .collect();

        // Apply limit if specified
        if let Some(lim) = limit {
            hotspots.truncate(lim);
        }

        Ok(hotspots)
    }

    /// Find unused files (files with no incoming dependencies)
    ///
    /// Files that are never imported are potential candidates for deletion.
    /// Uses `resolved_file_id` column for instant SQL lookup (sub-10ms).
    ///
    /// **Barrel Export Resolution**: This function now follows barrel export chains
    /// to detect files that are indirectly imported via re-exports. For example:
    /// - `WithLabel.vue` exported by `packages/ui/components/index.ts`
    /// - App imports `@packages/ui/components` (resolves to index.ts)
    /// - This function follows the export chain and marks `WithLabel.vue` as used
    pub fn find_unused_files(&self) -> Result<Vec<i64>> {
        let conn = self.open_conn()?;

        // Build set of used files by following barrel export chains
        let mut used_files = HashSet::new();

        // Step 1: Get all files directly referenced in resolved_file_id
        let mut stmt = conn.prepare(
            "SELECT DISTINCT resolved_file_id
             FROM file_dependencies
             WHERE resolved_file_id IS NOT NULL"
        )?;

        let direct_imports: Vec<i64> = stmt
            .query_map([], |row| row.get(0))?
            .collect::<Result<Vec<_>, _>>()?;

        used_files.extend(&direct_imports);

        // Step 2: For each direct import, follow barrel export chains
        for file_id in direct_imports {
            // Resolve through barrel exports to find all indirectly used files
            let barrel_chain = self.resolve_through_barrel_exports(file_id)?;
            used_files.extend(barrel_chain);
        }

        // Step 3: Get all files NOT in the used set
        let mut stmt = conn.prepare("SELECT id FROM files ORDER BY id")?;
        let all_files: Vec<i64> = stmt
            .query_map([], |row| row.get(0))?
            .collect::<Result<Vec<_>, _>>()?;

        let unused: Vec<i64> = all_files
            .into_iter()
            .filter(|id| !used_files.contains(id))
            .collect();

        Ok(unused)
    }

    /// Resolve barrel export chains to find all files transitively exported from a given file
    ///
    /// Given a barrel file (e.g., `index.ts` that re-exports from other files), this function
    /// follows the export chain to find all source files that are transitively exported.
    ///
    /// # Example
    ///
    /// If `packages/ui/components/index.ts` contains:
    /// ```typescript
    /// export { default as WithLabel } from './WithLabel.vue';
    /// export { default as Button } from './Button.vue';
    /// ```
    ///
    /// Then calling this with the file_id of `index.ts` will return the file IDs of
    /// `WithLabel.vue` and `Button.vue`.
    ///
    /// # Arguments
    ///
    /// * `barrel_file_id` - File ID of the barrel file to start from
    ///
    /// # Returns
    ///
    /// Vec of file IDs that are transitively exported (includes the barrel file itself)
    pub fn resolve_through_barrel_exports(&self, barrel_file_id: i64) -> Result<Vec<i64>> {
        let conn = self.open_conn()?;

        let mut resolved_files = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Start with the barrel file itself
        queue.push_back(barrel_file_id);
        visited.insert(barrel_file_id);

        while let Some(current_id) = queue.pop_front() {
            resolved_files.push(current_id);

            // Get all exports from this file
            let mut stmt = conn.prepare(
                "SELECT resolved_source_id
                 FROM file_exports
                 WHERE file_id = ? AND resolved_source_id IS NOT NULL"
            )?;

            let exported_files: Vec<i64> = stmt
                .query_map([current_id], |row| row.get(0))?
                .collect::<Result<Vec<_>, _>>()?;

            // Follow each exported file
            for exported_id in exported_files {
                if !visited.contains(&exported_id) {
                    visited.insert(exported_id);
                    queue.push_back(exported_id);
                }
            }
        }

        Ok(resolved_files)
    }

    /// Find disconnected components (islands) in the dependency graph
    ///
    /// An "island" is a connected component - a group of files that depend on each
    /// other (directly or transitively) but have no dependencies to files outside
    /// the group.
    ///
    /// This is useful for identifying:
    /// - Independent subsystems that could be extracted as separate modules
    /// - Unreachable code clusters that might be dead code
    /// - Microservice boundaries in a monolith
    ///
    /// Returns a list of islands, where each island is a vector of file IDs.
    /// Islands are sorted by size (largest first).
    pub fn find_islands(&self) -> Result<Vec<Vec<i64>>> {
        let conn = self.open_conn()?;

        // Build undirected dependency graph (A imports B => edge A-B and B-A)
        let mut graph: HashMap<i64, Vec<i64>> = HashMap::new();

        let mut stmt = conn.prepare(
            "SELECT file_id, resolved_file_id
             FROM file_dependencies
             WHERE resolved_file_id IS NOT NULL"
        )?;

        let dependencies: Vec<(i64, i64)> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        // Build adjacency list (undirected) directly from resolved IDs
        for (file_id, target_id) in dependencies {
            // Add edge in both directions for undirected graph
            graph.entry(file_id).or_insert_with(Vec::new).push(target_id);
            graph.entry(target_id).or_insert_with(Vec::new).push(file_id);
        }

        // Get all file IDs (including isolated files with no dependencies)
        let all_files = self.get_all_file_ids()?;

        // Ensure all files are in the graph (even if they have no edges)
        for file_id in &all_files {
            graph.entry(*file_id).or_insert_with(Vec::new);
        }

        // Find connected components using DFS
        let mut visited = HashSet::new();
        let mut islands = Vec::new();

        for &file_id in &all_files {
            if !visited.contains(&file_id) {
                let mut island = Vec::new();
                self.dfs_island(&file_id, &graph, &mut visited, &mut island);
                islands.push(island);
            }
        }

        // Sort islands by size (largest first)
        islands.sort_by(|a, b| b.len().cmp(&a.len()));

        log::info!("Found {} islands (connected components)", islands.len());

        Ok(islands)
    }

    /// DFS helper for finding connected components (islands)
    fn dfs_island(
        &self,
        file_id: &i64,
        graph: &HashMap<i64, Vec<i64>>,
        visited: &mut HashSet<i64>,
        island: &mut Vec<i64>,
    ) {
        visited.insert(*file_id);
        island.push(*file_id);

        if let Some(neighbors) = graph.get(file_id) {
            for &neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    self.dfs_island(&neighbor, graph, visited, island);
                }
            }
        }
    }

    /// Build a cache of imported_path → file_id mappings for efficient lookup
    ///
    /// This method queries all unique imported_path values from the database
    /// and resolves each one to a file_id using fuzzy matching. The resulting
    /// cache enables O(1) lookups instead of repeated database queries.
    ///
    /// This is used internally by graph analysis operations (hotspots, circular
    /// dependencies, reverse lookups, etc.) to avoid O(N*M*K) query complexity.
    ///
    /// # Performance
    ///
    /// Building the cache requires O(N*M) queries where:
    /// - N = number of unique imported_path values (~1,000-5,000)
    /// - M = average number of path variants tried per path (~10)
    ///
    /// However, this is done ONCE upfront, enabling O(1) lookups for all
    /// subsequent operations. Without caching, each operation would make
    /// 10,000-100,000+ queries.
    ///
    /// # Returns
    ///
    /// HashMap mapping imported_path to resolved file_id (only includes
    /// successfully resolved paths; external/unresolved paths are omitted)
    fn build_resolution_cache(&self) -> Result<HashMap<String, i64>> {
        let conn = self.open_conn()?;

        // Get all unique imported_path values (single query)
        let mut stmt = conn.prepare(
            "SELECT DISTINCT imported_path FROM file_dependencies"
        )?;

        let imported_paths: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .collect::<Result<Vec<_>, _>>()?;

        let total_paths = imported_paths.len();
        log::info!("Building resolution cache for {} unique imported paths", total_paths);

        // Resolve each imported_path once
        let mut cache = HashMap::new();

        for imported_path in imported_paths {
            if let Ok(Some(file_id)) = self.resolve_imported_path_to_file_id(&imported_path) {
                cache.insert(imported_path, file_id);
            }
        }

        log::info!(
            "Resolution cache built: {} resolved, {} unresolved",
            cache.len(),
            total_paths - cache.len()
        );

        Ok(cache)
    }

    /// Clear all dependencies for a file (used during incremental reindexing)
    pub fn clear_dependencies(&self, file_id: i64) -> Result<()> {
        let conn = self.open_conn()?;

        conn.execute(
            "DELETE FROM file_dependencies WHERE file_id = ?",
            [file_id],
        )?;

        Ok(())
    }

    /// Resolve an imported path to a file ID using fuzzy matching
    ///
    /// This method converts an import path (e.g., namespace, module path) to various
    /// file path variants and tries to find a matching file using fuzzy path matching.
    ///
    /// # Arguments
    ///
    /// * `imported_path` - The import path as stored in the database
    ///   (e.g., "Rcm\\Http\\Controllers\\Controller", "crate::models", etc.)
    ///
    /// # Returns
    ///
    /// `Some(file_id)` if exactly one matching file is found, `None` otherwise
    ///
    /// # Examples
    ///
    /// - `Rcm\\Http\\Controllers\\Controller` → finds `services/php/rcm-backend/app/Http/Controllers/Controller.php`
    /// - `crate::models` → finds `src/models.rs`
    pub fn resolve_imported_path_to_file_id(&self, imported_path: &str) -> Result<Option<i64>> {
        let path_variants = generate_path_variants(imported_path);

        for variant in &path_variants {
            if let Ok(Some(file_id)) = self.get_file_id_by_path(variant) {
                log::trace!("Resolved '{}' → '{}' (file_id: {})", imported_path, variant, file_id);
                return Ok(Some(file_id));
            }
        }

        Ok(None)
    }

    /// Get file ID by path with fuzzy matching support
    ///
    /// Supports various path formats:
    /// - Exact paths: `services/php/app/Http/Controllers/FooController.php`
    /// - Relative paths: `./services/php/app/Http/Controllers/FooController.php`
    /// - Path fragments: `Controllers/FooController.php` or `FooController.php`
    /// - Absolute paths: `/home/user/project/services/php/.../FooController.php`
    ///
    /// Returns None if no matches found.
    /// Returns error if multiple matches found (ambiguous path fragment).
    pub fn get_file_id_by_path(&self, path: &str) -> Result<Option<i64>> {
        let conn = self.open_conn()?;

        // Normalize path: strip ./ prefix, ../ prefix, and convert absolute to relative
        let normalized_path = normalize_path_for_lookup(path);

        // Try exact match first (fast path)
        match conn.query_row(
            "SELECT id FROM files WHERE path = ?",
            [&normalized_path],
            |row| row.get::<_, i64>(0),
        ) {
            Ok(id) => return Ok(Some(id)),
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                // No exact match, try suffix match
            }
            Err(e) => return Err(e.into()),
        }

        // Try suffix match: find all files whose path ends with the normalized_path
        let mut stmt = conn.prepare(
            "SELECT id, path FROM files WHERE path LIKE '%' || ?"
        )?;

        let matches: Vec<(i64, String)> = stmt
            .query_map([&normalized_path], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        match matches.len() {
            0 => Ok(None),
            1 => Ok(Some(matches[0].0)),
            _ => {
                // Multiple matches - return error with suggestions
                let paths: Vec<String> = matches.iter().map(|(_, p)| p.clone()).collect();
                anyhow::bail!(
                    "Ambiguous path '{}' matches multiple files:\n  {}\n\nPlease be more specific.",
                    path,
                    paths.join("\n  ")
                );
            }
        }
    }

    /// Get dependency resolution statistics grouped by language
    ///
    /// Returns statistics showing how many internal dependencies are resolved vs unresolved
    /// for each language in the project.
    ///
    /// # Returns
    ///
    /// A vector of tuples: (language, total_deps, resolved_deps, resolution_rate)
    pub fn get_resolution_stats(&self) -> Result<Vec<(String, usize, usize, f64)>> {
        let conn = self.open_conn()?;

        let mut stmt = conn.prepare(
            "SELECT
                CASE
                    WHEN f.path LIKE '%.py' THEN 'Python'
                    WHEN f.path LIKE '%.go' THEN 'Go'
                    WHEN f.path LIKE '%.ts' THEN 'TypeScript'
                    WHEN f.path LIKE '%.rs' THEN 'Rust'
                    WHEN f.path LIKE '%.js' OR f.path LIKE '%.jsx' THEN 'JavaScript'
                    WHEN f.path LIKE '%.php' THEN 'PHP'
                    WHEN f.path LIKE '%.java' THEN 'Java'
                    WHEN f.path LIKE '%.kt' THEN 'Kotlin'
                    WHEN f.path LIKE '%.rb' THEN 'Ruby'
                    WHEN f.path LIKE '%.c' OR f.path LIKE '%.h' THEN 'C'
                    WHEN f.path LIKE '%.cpp' OR f.path LIKE '%.cc' OR f.path LIKE '%.hpp' THEN 'C++'
                    WHEN f.path LIKE '%.cs' THEN 'C#'
                    WHEN f.path LIKE '%.zig' THEN 'Zig'
                    ELSE 'Other'
                END as language,
                COUNT(*) as total,
                SUM(CASE WHEN d.resolved_file_id IS NOT NULL THEN 1 ELSE 0 END) as resolved
            FROM file_dependencies d
            JOIN files f ON d.file_id = f.id
            WHERE d.import_type = 'internal'
            GROUP BY language
            ORDER BY language",
        )?;

        let mut stats = Vec::new();

        let rows = stmt.query_map([], |row| {
            let language: String = row.get(0)?;
            let total: i64 = row.get(1)?;
            let resolved: i64 = row.get(2)?;
            let rate = if total > 0 {
                (resolved as f64 / total as f64) * 100.0
            } else {
                0.0
            };

            Ok((language, total as usize, resolved as usize, rate))
        })?;

        for row in rows {
            stats.push(row?);
        }

        Ok(stats)
    }

    /// Get all internal dependencies with their resolution status
    ///
    /// Returns detailed information about each internal dependency including source file,
    /// imported path, and whether it was successfully resolved.
    ///
    /// # Returns
    ///
    /// A vector of tuples: (source_file, imported_path, resolved_file_path)
    /// where resolved_file_path is None if the dependency couldn't be resolved.
    pub fn get_all_internal_dependencies(&self) -> Result<Vec<(String, String, Option<String>)>> {
        let conn = self.open_conn()?;

        let mut stmt = conn.prepare(
            "SELECT
                f.path,
                d.imported_path,
                f2.path as resolved_path
            FROM file_dependencies d
            JOIN files f ON d.file_id = f.id
            LEFT JOIN files f2 ON d.resolved_file_id = f2.id
            WHERE d.import_type = 'internal'
            ORDER BY f.path",
        )?;

        let mut deps = Vec::new();

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        })?;

        for row in rows {
            deps.push(row?);
        }

        Ok(deps)
    }

    /// Get total count of dependencies by type (for debugging)
    pub fn get_dependency_count_by_type(&self) -> Result<Vec<(String, usize)>> {
        let conn = self.open_conn()?;

        let mut stmt = conn.prepare(
            "SELECT import_type, COUNT(*) as count
             FROM file_dependencies
             GROUP BY import_type
             ORDER BY import_type",
        )?;

        let mut counts = Vec::new();

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)? as usize,
            ))
        })?;

        for row in rows {
            counts.push(row?);
        }

        Ok(counts)
    }
}

/// Generate path variants for an import path
///
/// Converts a namespace/import path to multiple file path variants for fuzzy matching.
/// Tries progressively shorter paths to handle custom PSR-4 mappings.
///
/// Examples:
/// - `Rcm\\Http\\Controllers\\Controller` →
///   - `Rcm/Http/Controllers/Controller.php`
///   - `Http/Controllers/Controller.php`
///   - `Controllers/Controller.php`
///   - `Controller.php`
fn generate_path_variants(import_path: &str) -> Vec<String> {
    // Convert namespace separators to path separators
    let path = import_path.replace('\\', "/").replace("::", "/");

    // Remove quotes if present (some languages quote import paths)
    let path = path.trim_matches('"').trim_matches('\'');

    // Split into components
    let components: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

    if components.is_empty() {
        return vec![];
    }

    let mut variants = Vec::new();

    // Generate progressively shorter paths
    // E.g., for "Rcm/Http/Controllers/Controller":
    // 1. Rcm/Http/Controllers/Controller.php (full path)
    // 2. Http/Controllers/Controller.php (without first component)
    // 3. Controllers/Controller.php (without first two)
    // 4. Controller.php (just the class name)
    for start_idx in 0..components.len() {
        let suffix = components[start_idx..].join("/");

        // Try with .php extension (most common)
        if !suffix.ends_with(".php") {
            variants.push(format!("{}.php", suffix));
        } else {
            variants.push(suffix.clone());
        }

        // Also try without extension (for languages that don't use extensions in imports)
        if !suffix.contains('.') {
            // Try common extensions
            variants.push(format!("{}.rs", suffix));
            variants.push(format!("{}.ts", suffix));
            variants.push(format!("{}.js", suffix));
            variants.push(format!("{}.py", suffix));
        }
    }

    variants
}

/// Normalize a path for fuzzy lookup
///
/// Strips common prefixes that might differ between query and database:
/// - `./` and `../` prefixes
/// - Absolute paths (converts to relative by taking only the path component)
///
/// Examples:
/// - `./services/foo.php` → `services/foo.php`
/// - `/home/user/project/services/foo.php` → `services/foo.php` (just filename portion)
/// - `GetCaseByBatchNumberController.php` → `GetCaseByBatchNumberController.php`
fn normalize_path_for_lookup(path: &str) -> String {
    // Strip ./ and ../ prefixes
    let mut normalized = path.trim_start_matches("./").to_string();
    if normalized.starts_with("../") {
        normalized = normalized.trim_start_matches("../").to_string();
    }

    // If it's an absolute path, extract the relevant portion
    // This handles cases like `/home/user/Code/project/services/php/...`
    // We want to extract just `services/php/...` part
    if normalized.starts_with('/') || normalized.starts_with('\\') {
        // Common project markers (ordered by priority)
        let markers = ["services", "src", "app", "lib", "packages", "modules"];

        let mut found_marker = false;
        for marker in &markers {
            if let Some(idx) = normalized.find(marker) {
                normalized = normalized[idx..].to_string();
                found_marker = true;
                break;
            }
        }

        // If no marker found, just use the filename
        if !found_marker {
            use std::path::Path;
            let path_obj = Path::new(&normalized);
            if let Some(filename) = path_obj.file_name() {
                normalized = filename.to_string_lossy().to_string();
            }
        }
    }

    normalized
}

/// Resolve a Rust import path to an absolute file path
///
/// This function handles Rust-specific path resolution rules:
/// - `crate::` - Starts from crate root (src/lib.rs or src/main.rs)
/// - `super::` - Goes up one module level
/// - `self::` - Stays in current module
/// - `mod name` - Looks for name.rs or name/mod.rs
/// - External crates - Returns None
///
/// # Arguments
///
/// * `import_path` - The import path as written in source (e.g., "crate::models::Language")
/// * `current_file` - Path to the file containing the import (e.g., "src/query.rs")
/// * `project_root` - Root directory of the project
///
/// # Returns
///
/// `Some(path)` if the import resolves to a project file, `None` if it's external/stdlib
pub fn resolve_rust_import(
    import_path: &str,
    current_file: &str,
    project_root: &std::path::Path,
) -> Option<String> {
    use std::path::{Path, PathBuf};

    // External crates and stdlib - don't resolve
    if !import_path.starts_with("crate::")
        && !import_path.starts_with("super::")
        && !import_path.starts_with("self::")
    {
        return None;
    }

    let current_path = Path::new(current_file);
    let mut resolved_path: Option<PathBuf> = None;

    if import_path.starts_with("crate::") {
        // Start from crate root (src/lib.rs or src/main.rs)
        let crate_root = if project_root.join("src/lib.rs").exists() {
            project_root.join("src")
        } else if project_root.join("src/main.rs").exists() {
            project_root.join("src")
        } else {
            // Fallback to src/ directory
            project_root.join("src")
        };

        let path_parts: Vec<&str> = import_path
            .strip_prefix("crate::")
            .unwrap()
            .split("::")
            .collect();

        resolved_path = resolve_module_path(&crate_root, &path_parts);
    } else if import_path.starts_with("super::") {
        // Go up one directory from current file's parent (the current module's parent)
        if let Some(current_dir) = current_path.parent() {
            if let Some(parent_dir) = current_dir.parent() {
                let path_parts: Vec<&str> = import_path
                    .strip_prefix("super::")
                    .unwrap()
                    .split("::")
                    .collect();

                resolved_path = resolve_module_path(parent_dir, &path_parts);
            }
        }
    } else if import_path.starts_with("self::") {
        // Stay in current directory
        if let Some(current_dir) = current_path.parent() {
            let path_parts: Vec<&str> = import_path
                .strip_prefix("self::")
                .unwrap()
                .split("::")
                .collect();

            resolved_path = resolve_module_path(current_dir, &path_parts);
        }
    }

    // Convert to string and make relative to project root
    resolved_path.and_then(|p| {
        p.strip_prefix(project_root)
            .ok()
            .map(|rel| rel.to_string_lossy().to_string())
    })
}

/// Resolve a module path given a starting directory and path components
///
/// Handles Rust's module system rules:
/// - `foo` → check foo.rs or foo/mod.rs
/// - `foo::bar` → check foo/bar.rs or foo/bar/mod.rs
fn resolve_module_path(start_dir: &std::path::Path, components: &[&str]) -> Option<std::path::PathBuf> {

    if components.is_empty() {
        return None;
    }

    let mut current = start_dir.to_path_buf();

    // For all components except the last, they must be directories
    for &component in &components[..components.len() - 1] {
        // Try as a directory with mod.rs
        let dir_path = current.join(component);
        let mod_file = dir_path.join("mod.rs");

        if mod_file.exists() {
            current = dir_path;
        } else {
            // Component must be a directory for nested paths
            return None;
        }
    }

    // For the last component, try both file.rs and file/mod.rs
    let last_component = components.last().unwrap();

    // Try as a single file
    let file_path = current.join(format!("{}.rs", last_component));
    if file_path.exists() {
        return Some(file_path);
    }

    // Try as a directory with mod.rs
    let dir_path = current.join(last_component);
    let mod_file = dir_path.join("mod.rs");
    if mod_file.exists() {
        return Some(mod_file);
    }

    None
}

/// Resolve a `mod` declaration to a file path
///
/// For `mod parser;`, this checks for:
/// - `parser.rs` (sibling file)
/// - `parser/mod.rs` (directory module)
pub fn resolve_rust_mod_declaration(
    mod_name: &str,
    current_file: &str,
    _project_root: &std::path::Path,
) -> Option<String> {
    use std::path::Path;

    let current_path = Path::new(current_file);
    let current_dir = current_path.parent()?;

    // Try sibling file
    let sibling = current_dir.join(format!("{}.rs", mod_name));
    if sibling.exists() {
        return Some(sibling.to_string_lossy().to_string());
    }

    // Try directory module
    let dir_mod = current_dir.join(mod_name).join("mod.rs");
    if dir_mod.exists() {
        return Some(dir_mod.to_string_lossy().to_string());
    }

    None
}

/// Resolve a PHP import path to a file path
///
/// This function handles PHP-specific namespace-to-file mapping:
/// - Converts backslash-separated namespaces to forward-slash paths
/// - Handles PSR-4 autoloading conventions
/// - Filters out external vendor namespaces (returns None for non-project code)
///
/// # Arguments
///
/// * `import_path` - PHP namespace path (e.g., "App\\Http\\Controllers\\UserController")
/// * `current_file` - Not used for PHP (PHP uses absolute namespaces)
/// * `project_root` - Root directory of the project
///
/// # Returns
///
/// `Some(path)` if the import resolves to a project file, `None` if it's external/stdlib
///
/// # Examples
///
/// - `App\\Http\\Controllers\\FooController` → `app/Http/Controllers/FooController.php`
/// - `App\\Models\\User` → `app/Models/User.php`
/// - `Illuminate\\Database\\Migration` → `None` (external vendor namespace)
pub fn resolve_php_import(
    import_path: &str,
    _current_file: &str,
    project_root: &std::path::Path,
) -> Option<String> {
    // External vendor namespaces (Laravel, Symfony, etc.) - don't resolve
    const VENDOR_NAMESPACES: &[&str] = &[
        "Illuminate\\", "Symfony\\", "Laravel\\", "Psr\\",
        "Doctrine\\", "Monolog\\", "PHPUnit\\", "Carbon\\",
        "GuzzleHttp\\", "Composer\\", "Predis\\", "League\\"
    ];

    // Check if this is a vendor namespace
    for vendor_ns in VENDOR_NAMESPACES {
        if import_path.starts_with(vendor_ns) {
            return None;
        }
    }

    // Convert namespace to file path
    // PHP namespaces use backslashes: App\Http\Controllers\FooController
    // Files use forward slashes: app/Http/Controllers/FooController.php
    let file_path = import_path.replace('\\', "/");

    // Try common PSR-4 mappings (lowercase first component)
    // App\... → app/...
    // Database\... → database/...
    let path_candidates = vec![
        // Try with lowercase first component (PSR-4 standard)
        {
            let parts: Vec<&str> = file_path.split('/').collect();
            if let Some(first) = parts.first() {
                let mut result = vec![first.to_lowercase()];
                result.extend(parts[1..].iter().map(|s| s.to_string()));
                result.join("/") + ".php"
            } else {
                file_path.clone() + ".php"
            }
        },
        // Try exact path (some projects use exact case)
        file_path.clone() + ".php",
        // Try all lowercase (legacy projects)
        file_path.to_lowercase() + ".php",
    ];

    // Check each candidate path
    for candidate in &path_candidates {
        let full_path = project_root.join(candidate);
        if full_path.exists() {
            // Return relative path
            return Some(candidate.clone());
        }
    }

    // If no file found, return None (likely external or not yet created)
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup_test_cache() -> (TempDir, CacheManager) {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        cache.init().unwrap();

        // Add some test files
        cache.update_file("src/main.rs", "rust", 100).unwrap();
        cache.update_file("src/lib.rs", "rust", 50).unwrap();
        cache.update_file("src/utils.rs", "rust", 30).unwrap();

        (temp, cache)
    }

    #[test]
    fn test_insert_and_get_dependencies() {
        let (_temp, cache) = setup_test_cache();
        let deps_index = DependencyIndex::new(cache);

        // Get file IDs
        let main_id = 1i64;
        let lib_id = 2i64;

        // Insert a dependency: main.rs imports lib.rs
        deps_index
            .insert_dependency(
                main_id,
                "crate::lib".to_string(),
                Some(lib_id),
                ImportType::Internal,
                5,
                None,
            )
            .unwrap();

        // Retrieve dependencies
        let deps = deps_index.get_dependencies(main_id).unwrap();
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].imported_path, "crate::lib");
        assert_eq!(deps[0].resolved_file_id, Some(lib_id));
        assert_eq!(deps[0].import_type, ImportType::Internal);
    }

    #[test]
    fn test_reverse_lookup() {
        let (_temp, cache) = setup_test_cache();
        let deps_index = DependencyIndex::new(cache);

        let main_id = 1i64;
        let lib_id = 2i64;
        let utils_id = 3i64;

        // main.rs imports lib.rs
        deps_index
            .insert_dependency(
                main_id,
                "crate::lib".to_string(),
                Some(lib_id),
                ImportType::Internal,
                5,
                None,
            )
            .unwrap();

        // utils.rs also imports lib.rs
        deps_index
            .insert_dependency(
                utils_id,
                "crate::lib".to_string(),
                Some(lib_id),
                ImportType::Internal,
                3,
                None,
            )
            .unwrap();

        // Get files that import lib.rs
        let dependents = deps_index.get_dependents(lib_id).unwrap();
        assert_eq!(dependents.len(), 2);
        assert!(dependents.contains(&main_id));
        assert!(dependents.contains(&utils_id));
    }

    #[test]
    fn test_transitive_dependencies() {
        let (_temp, cache) = setup_test_cache();
        let deps_index = DependencyIndex::new(cache);

        let file1 = 1i64;
        let file2 = 2i64;
        let file3 = 3i64;

        // file1 → file2 → file3
        deps_index
            .insert_dependency(
                file1,
                "file2".to_string(),
                Some(file2),
                ImportType::Internal,
                1,
                None,
            )
            .unwrap();

        deps_index
            .insert_dependency(
                file2,
                "file3".to_string(),
                Some(file3),
                ImportType::Internal,
                1,
                None,
            )
            .unwrap();

        // Get transitive deps at depth 2
        let transitive = deps_index.get_transitive_deps(file1, 2).unwrap();

        // Should include file1 (depth 0), file2 (depth 1), file3 (depth 2)
        assert_eq!(transitive.len(), 3);
        assert_eq!(transitive.get(&file1), Some(&0));
        assert_eq!(transitive.get(&file2), Some(&1));
        assert_eq!(transitive.get(&file3), Some(&2));
    }

    #[test]
    fn test_batch_insert() {
        let (_temp, cache) = setup_test_cache();
        let deps_index = DependencyIndex::new(cache);

        let deps = vec![
            Dependency {
                file_id: 1,
                imported_path: "std::collections".to_string(),
                resolved_file_id: None,
                import_type: ImportType::Stdlib,
                line_number: 1,
                imported_symbols: Some(vec!["HashMap".to_string()]),
            },
            Dependency {
                file_id: 1,
                imported_path: "crate::lib".to_string(),
                resolved_file_id: Some(2),
                import_type: ImportType::Internal,
                line_number: 2,
                imported_symbols: None,
            },
        ];

        deps_index.batch_insert_dependencies(&deps).unwrap();

        let retrieved = deps_index.get_dependencies(1).unwrap();
        assert_eq!(retrieved.len(), 2);
    }

    #[test]
    fn test_clear_dependencies() {
        let (_temp, cache) = setup_test_cache();
        let deps_index = DependencyIndex::new(cache);

        // Insert dependencies
        deps_index
            .insert_dependency(
                1,
                "crate::lib".to_string(),
                Some(2),
                ImportType::Internal,
                1,
                None,
            )
            .unwrap();

        // Verify they exist
        assert_eq!(deps_index.get_dependencies(1).unwrap().len(), 1);

        // Clear them
        deps_index.clear_dependencies(1).unwrap();

        // Verify they're gone
        assert_eq!(deps_index.get_dependencies(1).unwrap().len(), 0);
    }

    #[test]
    fn test_resolve_rust_import_crate() {
        use std::fs;
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        let project_root = temp.path();

        // Create directory structure
        fs::create_dir_all(project_root.join("src")).unwrap();
        fs::write(project_root.join("src/lib.rs"), "").unwrap();
        fs::write(project_root.join("src/models.rs"), "").unwrap();

        // Test crate:: resolution
        let resolved = resolve_rust_import(
            "crate::models",
            "src/query.rs",
            project_root,
        );

        assert_eq!(resolved, Some("src/models.rs".to_string()));
    }

    #[test]
    fn test_resolve_rust_import_super() {
        use std::fs;
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        let project_root = temp.path();

        // Create directory structure: src/parsers/rust.rs needs to import src/models.rs
        fs::create_dir_all(project_root.join("src/parsers")).unwrap();
        fs::write(project_root.join("src/models.rs"), "").unwrap();
        fs::write(project_root.join("src/parsers/rust.rs"), "").unwrap();

        // Test super:: resolution from parsers/rust.rs
        // Use absolute path for current_file
        let current_file = project_root.join("src/parsers/rust.rs");
        let resolved = resolve_rust_import(
            "super::models",
            &current_file.to_string_lossy(),
            project_root,
        );

        assert_eq!(resolved, Some("src/models.rs".to_string()));
    }

    #[test]
    fn test_resolve_rust_import_external() {
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        let project_root = temp.path();

        // External crates should return None
        let resolved = resolve_rust_import(
            "serde::Serialize",
            "src/models.rs",
            project_root,
        );

        assert_eq!(resolved, None);

        // Stdlib should return None
        let resolved = resolve_rust_import(
            "std::collections::HashMap",
            "src/models.rs",
            project_root,
        );

        assert_eq!(resolved, None);
    }

    #[test]
    fn test_resolve_rust_mod_declaration() {
        use std::fs;
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        let project_root = temp.path();

        // Create directory structure
        fs::create_dir_all(project_root.join("src")).unwrap();
        fs::write(project_root.join("src/lib.rs"), "").unwrap();
        fs::write(project_root.join("src/parser.rs"), "").unwrap();

        // Test mod declaration resolution
        let resolved = resolve_rust_mod_declaration(
            "parser",
            &project_root.join("src/lib.rs").to_string_lossy(),
            project_root,
        );

        assert!(resolved.is_some());
        assert!(resolved.unwrap().ends_with("src/parser.rs"));
    }

    #[test]
    fn test_resolve_rust_import_nested() {
        use std::fs;
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        let project_root = temp.path();

        // Create directory structure: src/models/language.rs
        fs::create_dir_all(project_root.join("src/models")).unwrap();
        fs::write(project_root.join("src/models/mod.rs"), "").unwrap();
        fs::write(project_root.join("src/models/language.rs"), "").unwrap();

        // Test nested module resolution
        let resolved = resolve_rust_import(
            "crate::models::language",
            "src/query.rs",
            project_root,
        );

        assert_eq!(resolved, Some("src/models/language.rs".to_string()));
    }
}
