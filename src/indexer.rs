//! Indexing engine for parsing source code
//!
//! The indexer scans the project directory, parses source files using Tree-sitter,
//! and builds the symbol/token cache for fast querying.

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::cache::CacheManager;
use crate::content_store::{ContentReader, ContentWriter};
use crate::dependency::DependencyIndex;
use crate::models::{Dependency, IndexConfig, IndexStats, Language, ImportType};
use crate::output;
use crate::parsers::{DependencyExtractor, ImportInfo, ExportInfo};
use crate::parsers::rust::RustDependencyExtractor;
use crate::parsers::python::PythonDependencyExtractor;
use crate::parsers::typescript::TypeScriptDependencyExtractor;
use crate::parsers::go::GoDependencyExtractor;
use crate::parsers::java::JavaDependencyExtractor;
use crate::parsers::c::CDependencyExtractor;
use crate::parsers::cpp::CppDependencyExtractor;
use crate::parsers::csharp::CSharpDependencyExtractor;
use crate::parsers::php::PhpDependencyExtractor;
use crate::parsers::ruby::RubyDependencyExtractor;
use crate::parsers::kotlin::KotlinDependencyExtractor;
use crate::parsers::zig::ZigDependencyExtractor;
use crate::parsers::vue::VueDependencyExtractor;
use crate::parsers::svelte::SvelteDependencyExtractor;
use crate::trigram::TrigramIndex;

/// Progress callback type: (current_file_count, total_file_count, status_message)
/// Uses Arc to allow cloning for multi-threaded progress updates
pub type ProgressCallback = Arc<dyn Fn(usize, usize, String) + Send + Sync>;

/// Result of processing a single file (used for parallel processing)
struct FileProcessingResult {
    path: PathBuf,
    path_str: String,
    hash: String,
    content: String,
    language: Language,
    line_count: usize,
    dependencies: Vec<ImportInfo>,
    exports: Vec<ExportInfo>,
}

/// Find the nearest tsconfig.json for a given source file
///
/// Walks up the directory tree from the source file to find the nearest tsconfig directory.
/// Returns a reference to the PathAliasMap if found.
fn find_nearest_tsconfig<'a>(
    file_path: &str,
    root: &Path,
    tsconfigs: &'a HashMap<PathBuf, crate::parsers::tsconfig::PathAliasMap>,
) -> Option<&'a crate::parsers::tsconfig::PathAliasMap> {
    // Convert file_path to absolute path (relative to root)
    let abs_file_path = if Path::new(file_path).is_absolute() {
        PathBuf::from(file_path)
    } else {
        root.join(file_path)
    };

    // Start from the file's directory and walk up
    let mut current_dir = abs_file_path.parent()?;

    loop {
        // Check if we have a tsconfig for this directory
        if let Some(alias_map) = tsconfigs.get(current_dir) {
            return Some(alias_map);
        }

        // Move up one directory
        current_dir = current_dir.parent()?;

        // Stop if we've reached the root
        if current_dir == root || !current_dir.starts_with(root) {
            break;
        }
    }

    None
}

/// Manages the indexing process
pub struct Indexer {
    cache: CacheManager,
    config: IndexConfig,
}

impl Indexer {
    /// Create a new indexer with the given cache manager and config
    pub fn new(cache: CacheManager, config: IndexConfig) -> Self {
        Self { cache, config }
    }

    /// Build or update the index for the given root directory
    pub fn index(&self, root: impl AsRef<Path>, show_progress: bool) -> Result<IndexStats> {
        self.index_with_callback(root, show_progress, None)
    }

    /// Build or update the index with progress callback support
    pub fn index_with_callback(
        &self,
        root: impl AsRef<Path>,
        show_progress: bool,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<IndexStats> {
        let root = root.as_ref();
        log::info!("Indexing directory: {:?}", root);

        // Get git state (if in git repo)
        let git_state = crate::git::get_git_state_optional(root)?;
        let branch = git_state
            .as_ref()
            .map(|s| s.branch.clone())
            .unwrap_or_else(|| "_default".to_string());

        if let Some(ref state) = git_state {
            log::info!(
                "Git state: branch='{}', commit='{}', dirty={}",
                state.branch,
                state.commit,
                state.dirty
            );
        } else {
            log::info!("Not a git repository, using default branch");
        }

        // Configure thread pool for parallel processing
        // 0 = auto (use 80% of available cores to avoid locking the system)
        let num_threads = if self.config.parallel_threads == 0 {
            let available_cores = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            // Use 80% of available cores (minimum 1, maximum 8)
            // Cap at 8 to prevent diminishing returns from cache contention on high-core systems
            ((available_cores as f64 * 0.8).ceil() as usize).max(1).min(8)
        } else {
            self.config.parallel_threads
        };

        log::info!("Using {} threads for parallel indexing (out of {} available)",
                   num_threads,
                   std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));

        // Ensure cache is initialized
        self.cache.init()?;

        // Check available disk space after cache is initialized
        self.check_disk_space(root)?;

        // Load existing hashes for incremental indexing (for current branch)
        let existing_hashes = self.cache.load_hashes_for_branch(&branch)?;
        log::debug!("Loaded {} existing file hashes for branch '{}'", existing_hashes.len(), branch);

        // Step 1: Walk directory tree and collect files
        let files = self.discover_files(root)?;
        let total_files = files.len();
        log::info!("Discovered {} files to index", total_files);

        // Step 1.4: Parse tsconfig.json files for TypeScript/Vue path alias resolution
        // Must be done before parallel processing so it's available during dependency extraction
        let tsconfigs = crate::parsers::tsconfig::parse_all_tsconfigs(root)
            .unwrap_or_else(|e| {
                log::warn!("Failed to parse tsconfig.json files: {}", e);
                HashMap::new()
            });
        if !tsconfigs.is_empty() {
            log::info!("Found {} tsconfig.json files", tsconfigs.len());
            for (config_dir, alias_map) in &tsconfigs {
                log::debug!("  {} (base_url: {:?}, {} aliases)",
                           config_dir.display(),
                           alias_map.base_url,
                           alias_map.aliases.len());
            }
        }

        // Step 1.5: Quick incremental check - are all files unchanged?
        // If yes, skip expensive rebuild entirely and return cached stats
        if !existing_hashes.is_empty() && total_files == existing_hashes.len() {
            // Same number of files - check if any changed by comparing hashes
            let mut any_changed = false;

            for file_path in &files {
                // Normalize path to be relative to root (handles both ./ prefix and absolute paths)
                let path_str = file_path.to_string_lossy().to_string();
                let normalized_path = if let Ok(rel_path) = file_path.strip_prefix(root) {
                    // Convert absolute path to relative
                    rel_path.to_string_lossy().to_string()
                } else {
                    // Already relative, just strip ./ prefix
                    path_str.trim_start_matches("./").to_string()
                };

                // Check if file exists in cache
                if let Some(existing_hash) = existing_hashes.get(&normalized_path) {
                    // Read and hash file to check if changed
                    match std::fs::read_to_string(file_path) {
                        Ok(content) => {
                            let current_hash = self.hash_content(content.as_bytes());
                            if &current_hash != existing_hash {
                                any_changed = true;
                                log::debug!("File changed: {}", path_str);
                                break; // Early exit - we know we need to rebuild
                            }
                        }
                        Err(_) => {
                            any_changed = true;
                            break;
                        }
                    }
                } else {
                    // File not in cache - something changed
                    any_changed = true;
                    break;
                }
            }

            if !any_changed {
                // Validate that binary index files actually exist and contain data.
                // A prior interrupted run may have written hashes to the database
                // but failed to finalize content.bin/trigrams.bin.
                let content_path = self.cache.path().join("content.bin");
                let trigrams_path = self.cache.path().join("trigrams.bin");
                if content_path.exists() && trigrams_path.exists() {
                    if let Ok(reader) = ContentReader::open(&content_path) {
                        if reader.file_count() > 0 {
                            log::info!("No files changed - skipping index rebuild");
                            return Ok(self.cache.stats()?);
                        }
                    }
                    log::warn!("content.bin invalid despite hashes matching - forcing rebuild");
                } else {
                    log::warn!("Binary index files missing - forcing rebuild");
                }
            }
        } else if total_files != existing_hashes.len() {
            log::info!("File count changed ({} -> {}) - full reindex required",
                       existing_hashes.len(), total_files);
        }

        // Step 2: Build trigram index + content store
        let mut new_hashes = HashMap::new();
        let mut files_indexed = 0;
        let mut file_metadata: Vec<(String, String, String, usize)> = Vec::new(); // For batch SQLite update
        let mut all_dependencies: Vec<(String, Vec<ImportInfo>)> = Vec::new(); // For batch dependency insertion
        let mut all_exports: Vec<(String, Vec<ExportInfo>)> = Vec::new(); // For batch export insertion

        // Initialize trigram index and content store
        let mut trigram_index = TrigramIndex::new();
        let mut content_writer = ContentWriter::new();

        // Enable batch-flush mode for trigram index if we have lots of files
        if total_files > 10000 {
            let temp_dir = self.cache.path().join("trigram_temp");
            trigram_index.enable_batch_flush(temp_dir)
                .context("Failed to enable batch-flush mode for trigram index")?;
            log::info!("Enabled batch-flush mode for {} files", total_files);
        }

        // Initialize content writer to start streaming writes immediately
        let content_path = self.cache.path().join("content.bin");
        content_writer.init(content_path.clone())
            .context("Failed to initialize content writer")?;

        // Create progress bar (only if requested via --progress flag)
        let pb = if show_progress {
            let pb = ProgressBar::new(total_files as u64);
            pb.set_draw_target(ProgressDrawTarget::stderr());
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files ({percent}%) {msg}")
                    .unwrap()
                    .progress_chars("=>-")
            );
            // Force updates every 100ms to ensure progress is visible
            pb.enable_steady_tick(std::time::Duration::from_millis(100));
            pb
        } else {
            ProgressBar::hidden()
        };

        // Atomic counter for thread-safe progress updates
        let progress_counter = Arc::new(AtomicU64::new(0));
        // Shared status message for progress callback
        let progress_status = Arc::new(Mutex::new("Indexing files...".to_string()));

        let _start_time = Instant::now();

        // Spawn a background thread to update progress bar and call callback during parallel processing
        let counter_for_thread = Arc::clone(&progress_counter);
        let status_for_thread = Arc::clone(&progress_status);
        let pb_clone = pb.clone();
        let callback_for_thread = progress_callback.clone();
        let total_files_for_thread = total_files;
        let progress_thread = if show_progress || callback_for_thread.is_some() {
            Some(std::thread::spawn(move || {
                loop {
                    let count = counter_for_thread.load(Ordering::Relaxed);
                    pb_clone.set_position(count);

                    // Call progress callback if provided
                    if let Some(ref callback) = callback_for_thread {
                        let status = status_for_thread.lock().unwrap().clone();
                        callback(count as usize, total_files_for_thread, status);
                    }

                    if count >= total_files_for_thread as u64 {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            }))
        } else {
            None
        };

        // Build a custom thread pool with limited threads
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .context("Failed to create thread pool")?;

        // Process files in batches to avoid OOM on huge codebases
        // Batch size: process 5000 files at a time to limit memory usage
        const BATCH_SIZE: usize = 5000;
        let num_batches = total_files.div_ceil(BATCH_SIZE);
        log::info!("Processing {} files in {} batches of up to {} files",
                   total_files, num_batches, BATCH_SIZE);

        for (batch_idx, batch_files) in files.chunks(BATCH_SIZE).enumerate() {
            log::info!("Processing batch {}/{} ({} files)",
                       batch_idx + 1, num_batches, batch_files.len());

            // Process files in parallel using rayon with custom thread pool
            let counter_clone = Arc::clone(&progress_counter);
            let results: Vec<Option<FileProcessingResult>> = pool.install(|| {
                batch_files
                    .par_iter()
                    .map(|file_path| {
                // Normalize path to be relative to root (handles both ./ prefix and absolute paths)
                let path_str = file_path.to_string_lossy().to_string();
                let normalized_path = if let Ok(rel_path) = file_path.strip_prefix(root) {
                    // Convert absolute path to relative
                    rel_path.to_string_lossy().to_string()
                } else {
                    // Already relative, just strip ./ prefix
                    path_str.trim_start_matches("./").to_string()
                };

                // Read file content once (used for hashing, trigrams, and parsing)
                let content = match std::fs::read_to_string(&file_path) {
                    Ok(c) => c,
                    Err(e) => {
                        log::warn!("Failed to read {}: {}", path_str, e);
                        // Update progress
                        counter_clone.fetch_add(1, Ordering::Relaxed);
                        return None;
                    }
                };

                // Compute hash from content (no duplicate file read!)
                let hash = self.hash_content(content.as_bytes());

                // Detect language
                let ext = file_path.extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                let language = Language::from_extension(ext);

                // Count lines in the file
                let line_count = content.lines().count();

                // Extract dependencies and exports for supported languages
                let dependencies = match language {
                    Language::Rust => {
                        match RustDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::Python => {
                        match PythonDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::TypeScript | Language::JavaScript => {
                        // Find nearest tsconfig for path alias resolution
                        let alias_map = find_nearest_tsconfig(&path_str, root, &tsconfigs);
                        match TypeScriptDependencyExtractor::extract_dependencies_with_alias_map(&content, alias_map) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::Go => {
                        match GoDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::Java => {
                        match JavaDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::C => {
                        match CDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::Cpp => {
                        match CppDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::CSharp => {
                        match CSharpDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::PHP => {
                        match PhpDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::Ruby => {
                        match RubyDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::Kotlin => {
                        match KotlinDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::Zig => {
                        match ZigDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::Vue => {
                        // Find nearest tsconfig for path alias resolution
                        let alias_map = find_nearest_tsconfig(&path_str, root, &tsconfigs);
                        match VueDependencyExtractor::extract_dependencies_with_alias_map(&content, alias_map) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::Svelte => {
                        match SvelteDependencyExtractor::extract_dependencies(&content) {
                            Ok(deps) => deps,
                            Err(e) => {
                                log::warn!("Failed to extract dependencies from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    // Other languages not yet implemented
                    _ => Vec::new(),
                };

                // Extract exports (for barrel export tracking)
                let exports = match language {
                    Language::TypeScript | Language::JavaScript => {
                        // Find nearest tsconfig for path alias resolution
                        let alias_map = find_nearest_tsconfig(&path_str, root, &tsconfigs);
                        match TypeScriptDependencyExtractor::extract_export_declarations(&content, alias_map) {
                            Ok(exports) => exports,
                            Err(e) => {
                                log::warn!("Failed to extract exports from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    Language::Vue => {
                        // Find nearest tsconfig for path alias resolution
                        let alias_map = find_nearest_tsconfig(&path_str, root, &tsconfigs);
                        match VueDependencyExtractor::extract_export_declarations(&content, alias_map) {
                            Ok(exports) => exports,
                            Err(e) => {
                                log::warn!("Failed to extract exports from {}: {}", path_str, e);
                                Vec::new()
                            }
                        }
                    }
                    // Other languages not yet implemented for export tracking
                    _ => Vec::new(),
                };

                // Update progress atomically
                counter_clone.fetch_add(1, Ordering::Relaxed);

                Some(FileProcessingResult {
                    path: file_path.clone(),
                    path_str: normalized_path.to_string(),
                    hash,
                    content,
                    language,
                    line_count,
                    dependencies,
                    exports,
                })
                })
                .collect()
            });

            // Process batch results immediately (streaming approach to minimize memory)
            for result in results.into_iter().flatten() {
                // Add file to trigram index (get file_id)
                let file_id = trigram_index.add_file(result.path.clone());

                // Index file content directly (avoid accumulating all trigrams)
                trigram_index.index_file(file_id, &result.content);

                // Add to content store
                content_writer.add_file(result.path.clone(), &result.content);

                files_indexed += 1;

                // Prepare file metadata for batch database update
                file_metadata.push((
                    result.path_str.clone(),
                    result.hash.clone(),
                    format!("{:?}", result.language),
                    result.line_count
                ));

                // Collect dependencies for batch insertion (if any)
                if !result.dependencies.is_empty() {
                    all_dependencies.push((result.path_str.clone(), result.dependencies));
                }

                // Collect exports for batch insertion (if any)
                if !result.exports.is_empty() {
                    all_exports.push((result.path_str.clone(), result.exports));
                }

                new_hashes.insert(result.path_str, result.hash);
            }

            // Flush trigram index batch to disk if batch-flush mode is enabled
            if total_files > 10000 {
                let flush_msg = format!("Flushing batch {}/{}...", batch_idx + 1, num_batches);
                if show_progress {
                    pb.set_message(flush_msg.clone());
                }
                *progress_status.lock().unwrap() = flush_msg;
                trigram_index.flush_batch()
                    .context("Failed to flush trigram batch")?;
            }
        }

        // Wait for progress thread to finish
        if let Some(thread) = progress_thread {
            let _ = thread.join();
        }

        // Update progress bar to final count
        if show_progress {
            let final_count = progress_counter.load(Ordering::Relaxed);
            pb.set_position(final_count);
        }

        // Finalize trigram index (sort and deduplicate posting lists)
        *progress_status.lock().unwrap() = "Finalizing trigram index...".to_string();
        if show_progress {
            pb.set_message("Finalizing trigram index...".to_string());
        }
        trigram_index.finalize();

        // Update progress bar message for post-processing
        *progress_status.lock().unwrap() = "Writing file metadata to database...".to_string();
        if show_progress {
            pb.set_message("Writing file metadata to database...".to_string());
        }

        // Batch write file metadata AND branch hashes in a SINGLE atomic transaction
        // This ensures that if files are inserted, their hashes are guaranteed to be inserted too
        if !file_metadata.is_empty() {
            // Prepare files data (path, language, line_count)
            let files_without_hash: Vec<(String, String, usize)> = file_metadata
                .iter()
                .map(|(path, _hash, lang, lines)| (path.clone(), lang.clone(), *lines))
                .collect();

        // Record files for this branch (for branch-aware indexing)
        *progress_status.lock().unwrap() = "Recording branch files...".to_string();
        if show_progress {
            pb.set_message("Recording branch files...".to_string());
        }

            // Prepare branch files data (path, hash)
            let branch_files: Vec<(String, String)> = file_metadata
                .iter()
                .map(|(path, hash, _, _)| (path.clone(), hash.clone()))
                .collect();

            // Use atomic method that combines both operations
            self.cache.batch_update_files_and_branch(
                &files_without_hash,
                &branch_files,
                &branch,
                git_state.as_ref().map(|s| s.commit.as_str()),
            ).context("Failed to batch update files and branch hashes")?;

            log::info!("Wrote metadata and hashes for {} files to database", file_metadata.len());
        }

        // Update branch metadata
        self.cache.update_branch_metadata(
            &branch,
            git_state.as_ref().map(|s| s.commit.as_str()),
            file_metadata.len(),
            git_state.as_ref().map(|s| s.dirty).unwrap_or(false),
        )?;

        // Force WAL checkpoint to ensure background processes see all committed data
        // This is critical when spawning background symbol indexer immediately after
        self.cache.checkpoint_wal()
            .context("Failed to checkpoint WAL")?;
        log::debug!("WAL checkpoint completed - database is fully synced");

        // Step 2.5: Insert dependencies (after files are inserted and have IDs)
        if !all_dependencies.is_empty() {
            *progress_status.lock().unwrap() = "Extracting dependencies...".to_string();
            if show_progress {
                pb.set_message("Extracting dependencies...".to_string());
            }

            // Find and parse all go.mod files for Go projects (monorepo support)
            let go_modules = crate::parsers::go::parse_all_go_modules(root)
                .unwrap_or_else(|e| {
                    log::warn!("Failed to parse go.mod files: {}", e);
                    Vec::new()
                });
            if !go_modules.is_empty() {
                log::info!("Found {} Go modules", go_modules.len());
                for module in &go_modules {
                    log::debug!("  {} (project: {})", module.name, module.project_root);
                }
            }

            // Find and parse all pom.xml/build.gradle files for Java projects (monorepo support)
            let java_projects = crate::parsers::java::parse_all_java_projects(root)
                .unwrap_or_else(|e| {
                    log::warn!("Failed to parse Java project configs: {}", e);
                    Vec::new()
                });
            if !java_projects.is_empty() {
                log::info!("Found {} Java projects", java_projects.len());
                for project in &java_projects {
                    log::debug!("  {} (project: {})", project.package_name, project.project_root);
                }
            }

            // Find and parse all Python package configs for Python projects (monorepo support)
            let python_packages = crate::parsers::python::parse_all_python_packages(root)
                .unwrap_or_else(|e| {
                    log::warn!("Failed to parse Python package configs: {}", e);
                    Vec::new()
                });
            if !python_packages.is_empty() {
                log::info!("Found {} Python packages", python_packages.len());
                for package in &python_packages {
                    log::debug!("  {} (project: {})", package.name, package.project_root);
                }
            }

            // Find and parse *.gemspec files for Ruby projects (monorepo support)
            let ruby_projects = crate::parsers::ruby::parse_all_ruby_projects(root)
                .unwrap_or_else(|e| {
                    log::warn!("Failed to parse Ruby project configs: {}", e);
                    Vec::new()
                });
            if !ruby_projects.is_empty() {
                log::info!("Found {} Ruby projects", ruby_projects.len());
                for project in &ruby_projects {
                    log::debug!("  {} (project: {})", project.gem_name, project.project_root);
                }
            }

            // Note: Kotlin projects use the same java_projects above (same build systems: Maven/Gradle)

            // Find and parse all composer.json files for PHP projects (monorepo support)
            let php_psr4_mappings = crate::parsers::php::parse_all_composer_psr4(root)
                .unwrap_or_else(|e| {
                    log::warn!("Failed to parse composer.json files: {}", e);
                    Vec::new()
                });
            if !php_psr4_mappings.is_empty() {
                log::info!("Found {} PSR-4 mappings from composer.json files", php_psr4_mappings.len());
                for mapping in &php_psr4_mappings {
                    log::debug!("  {} => {} (project: {})", mapping.namespace_prefix, mapping.directory, mapping.project_root);
                }
            }

            // Find and parse all tsconfig.json files for TypeScript/Vue projects (monorepo support)
            let tsconfigs = crate::parsers::tsconfig::parse_all_tsconfigs(root)
                .unwrap_or_else(|e| {
                    log::warn!("Failed to parse tsconfig.json files: {}", e);
                    HashMap::new()
                });
            if !tsconfigs.is_empty() {
                log::info!("Found {} tsconfig.json files", tsconfigs.len());
                for (config_dir, alias_map) in &tsconfigs {
                    log::debug!("  {} (base_url: {:?}, {} aliases)",
                               config_dir.display(),
                               alias_map.base_url,
                               alias_map.aliases.len());
                }
            }

            // Create dependency index to resolve paths and insert dependencies
            let cache_for_deps = CacheManager::new(root);
            let dep_index = DependencyIndex::new(cache_for_deps);

            let mut total_deps_inserted = 0;

            // Process each file's dependencies
            for (file_path, import_infos) in all_dependencies {
                // Get file ID from database
                let file_id = match dep_index.get_file_id_by_path(&file_path)? {
                    Some(id) => id,
                    None => {
                        log::warn!("File not found in database (skipping dependencies): {}", file_path);
                        continue;
                    }
                };

                // Reclassify and filter dependencies
                let mut resolved_deps = Vec::new();

                for mut import_info in import_infos {
                    // Reclassify Go imports using module names (if Go project)
                    if file_path.ends_with(".go") {
                        // Check if the import matches any Go module
                        let mut reclassified = false;
                        for module in &go_modules {
                            import_info.import_type = crate::parsers::go::reclassify_go_import(
                                &import_info.imported_path,
                                Some(&module.name),
                            );
                            // If it's internal, we've found the right module
                            if matches!(import_info.import_type, ImportType::Internal) {
                                reclassified = true;
                                break;
                            }
                        }
                        // If no module matched, use base classification
                        if !reclassified {
                            import_info.import_type = crate::parsers::go::reclassify_go_import(
                                &import_info.imported_path,
                                None,
                            );
                        }
                    }

                    // Reclassify Java imports using package names (if Java project)
                    if file_path.ends_with(".java") {
                        // Check if the import matches any Java project
                        let mut reclassified = false;
                        for project in &java_projects {
                            import_info.import_type = crate::parsers::java::reclassify_java_import(
                                &import_info.imported_path,
                                Some(&project.package_name),
                            );
                            // If it's internal, we've found the right project
                            if matches!(import_info.import_type, ImportType::Internal) {
                                reclassified = true;
                                break;
                            }
                        }
                        // If no project matched, use base classification
                        if !reclassified {
                            import_info.import_type = crate::parsers::java::reclassify_java_import(
                                &import_info.imported_path,
                                None,
                            );
                        }
                    }

                    // Reclassify Python imports using package names (if Python project)
                    if file_path.ends_with(".py") {
                        // Check if the import matches any Python package
                        let mut reclassified = false;
                        for package in &python_packages {
                            import_info.import_type = crate::parsers::python::reclassify_python_import(
                                &import_info.imported_path,
                                Some(&package.name),
                            );
                            // If it's internal, we've found the right package
                            if matches!(import_info.import_type, ImportType::Internal) {
                                reclassified = true;
                                break;
                            }
                        }
                        // If no package matched, use base classification
                        if !reclassified {
                            import_info.import_type = crate::parsers::python::reclassify_python_import(
                                &import_info.imported_path,
                                None,
                            );
                        }
                    }

                    // Reclassify Ruby imports using gem names (if Ruby project)
                    if file_path.ends_with(".rb") || file_path.ends_with(".rake") || file_path.ends_with(".gemspec") {
                        // Check if the import matches any Ruby project
                        let mut reclassified = false;
                        for project in &ruby_projects {
                            let gem_names = vec![project.gem_name.clone()];
                            import_info.import_type = crate::parsers::ruby::reclassify_ruby_import(
                                &import_info.imported_path,
                                &gem_names,
                            );
                            // If it's internal, we've found the right project
                            if matches!(import_info.import_type, ImportType::Internal) {
                                reclassified = true;
                                break;
                            }
                        }
                        // If no project matched, use base classification (will be External or Stdlib)
                        if !reclassified {
                            import_info.import_type = crate::parsers::ruby::reclassify_ruby_import(
                                &import_info.imported_path,
                                &[],
                            );
                        }
                    }

                    // Reclassify Kotlin imports using package names (if Kotlin project)
                    if file_path.ends_with(".kt") || file_path.ends_with(".kts") {
                        // Check if the import matches any Java/Kotlin project (same build systems)
                        let mut reclassified = false;
                        for project in &java_projects {
                            import_info.import_type = crate::parsers::kotlin::reclassify_kotlin_import(
                                &import_info.imported_path,
                                Some(&project.package_name),
                            );
                            // If it's internal, we've found the right project
                            if matches!(import_info.import_type, ImportType::Internal) {
                                reclassified = true;
                                break;
                            }
                        }
                        // If no project matched, use base classification
                        if !reclassified {
                            import_info.import_type = crate::parsers::kotlin::reclassify_kotlin_import(
                                &import_info.imported_path,
                                None,
                            );
                        }
                    }

                    // ONLY insert Internal dependencies - skip External and Stdlib
                    if !matches!(import_info.import_type, ImportType::Internal) {
                        continue;
                    }

                    // Resolve PHP dependencies using PSR-4 (deterministic)
                    let resolved_file_id = if file_path.ends_with(".php") && !php_psr4_mappings.is_empty() {
                        // Use PSR-4 to resolve namespace to file path
                        if let Some(resolved_path) = crate::parsers::php::resolve_php_namespace_to_path(
                            &import_info.imported_path,
                            &php_psr4_mappings,
                        ) {
                            // Look up file ID in database using exact match
                            match dep_index.get_file_id_by_path(&resolved_path)? {
                                Some(id) => {
                                    log::trace!("Resolved PHP dependency: {} -> {} (file_id={})",
                                               import_info.imported_path, resolved_path, id);
                                    Some(id)
                                }
                                None => {
                                    log::trace!("PHP dependency resolved to path but file not in index: {} -> {}",
                                               import_info.imported_path, resolved_path);
                                    None
                                }
                            }
                        } else {
                            log::trace!("Could not resolve PHP namespace using PSR-4: {}",
                                       import_info.imported_path);
                            None
                        }
                    } else if file_path.ends_with(".py") && !python_packages.is_empty() {
                        // Resolve Python dependencies using package mappings
                        if let Some(resolved_path) = crate::parsers::python::resolve_python_import_to_path(
                            &import_info.imported_path,
                            &python_packages,
                            Some(&file_path),
                        ) {
                            // Look up file ID in database using exact match
                            match dep_index.get_file_id_by_path(&resolved_path)? {
                                Some(id) => {
                                    log::trace!("Resolved Python dependency: {} -> {} (file_id={})",
                                               import_info.imported_path, resolved_path, id);
                                    Some(id)
                                }
                                None => {
                                    log::trace!("Python dependency resolved to path but file not in index: {} -> {}",
                                               import_info.imported_path, resolved_path);
                                    None
                                }
                            }
                        } else {
                            log::trace!("Could not resolve Python import: {}", import_info.imported_path);
                            None
                        }
                    } else if file_path.ends_with(".go") && !go_modules.is_empty() {
                        // Resolve Go dependencies using module mappings
                        if let Some(resolved_path) = crate::parsers::go::resolve_go_import_to_path(
                            &import_info.imported_path,
                            &go_modules,
                            Some(&file_path),
                        ) {
                            // Look up file ID in database using exact match
                            match dep_index.get_file_id_by_path(&resolved_path)? {
                                Some(id) => {
                                    log::trace!("Resolved Go dependency: {} -> {} (file_id={})",
                                               import_info.imported_path, resolved_path, id);
                                    Some(id)
                                }
                                None => {
                                    log::trace!("Go dependency resolved to path but file not in index: {} -> {}",
                                               import_info.imported_path, resolved_path);
                                    None
                                }
                            }
                        } else {
                            log::trace!("Could not resolve Go import: {}", import_info.imported_path);
                            None
                        }
                    } else if file_path.ends_with(".ts") || file_path.ends_with(".tsx")
                            || file_path.ends_with(".js") || file_path.ends_with(".jsx")
                            || file_path.ends_with(".mts") || file_path.ends_with(".cts")
                            || file_path.ends_with(".mjs") || file_path.ends_with(".cjs") {
                        // Resolve TypeScript/JavaScript dependencies (relative imports and path aliases)
                        let alias_map = find_nearest_tsconfig(&file_path, root, &tsconfigs);
                        if let Some(candidates_str) = crate::parsers::typescript::resolve_ts_import_to_path(
                            &import_info.imported_path,
                            Some(&file_path),
                            alias_map,
                        ) {
                            // Parse pipe-delimited candidates (e.g., "path.tsx|path.ts|path.jsx|path.js")
                            let candidates: Vec<&str> = candidates_str.split('|').collect();

                            // Try each candidate in order until we find one in the database
                            let mut resolved_id = None;
                            for candidate_path in candidates {
                                // Normalize path to be relative to project root
                                // Convert absolute paths to relative (without requiring file to exist)
                                let normalized_candidate = if let Ok(rel_path) = std::path::Path::new(candidate_path).strip_prefix(root) {
                                    rel_path.to_string_lossy().to_string()
                                } else {
                                    // Not an absolute path or not under root - use as-is
                                    candidate_path.to_string()
                                };

                                log::debug!("Looking up TS/JS candidate: '{}' (from '{}')", normalized_candidate, candidate_path);
                                match dep_index.get_file_id_by_path(&normalized_candidate)? {
                                    Some(id) => {
                                        log::debug!("Resolved TS/JS dependency: {} -> {} (file_id={})",
                                                   import_info.imported_path, normalized_candidate, id);
                                        resolved_id = Some(id);
                                        break; // Found a match, stop trying
                                    }
                                    None => {
                                        log::trace!("TS/JS candidate not in index: {}", candidate_path);
                                    }
                                }
                            }

                            if resolved_id.is_none() {
                                log::trace!("TS/JS dependency: no matching file found in database for any candidate: {}",
                                           candidates_str);
                            }

                            resolved_id
                        } else {
                            log::trace!("Could not resolve TS/JS import (non-relative or external): {}", import_info.imported_path);
                            None
                        }
                    } else if file_path.ends_with(".rs") {
                        // Resolve Rust dependencies (crate::, super::, self::, mod declarations)
                        if let Some(resolved_path) = crate::parsers::rust::resolve_rust_use_to_path(
                            &import_info.imported_path,
                            Some(&file_path),
                            Some(root.to_str().unwrap_or("")),
                        ) {
                            // Look up file ID in database using exact match
                            match dep_index.get_file_id_by_path(&resolved_path)? {
                                Some(id) => {
                                    log::trace!("Resolved Rust dependency: {} -> {} (file_id={})",
                                               import_info.imported_path, resolved_path, id);
                                    Some(id)
                                }
                                None => {
                                    log::trace!("Rust dependency resolved to path but file not in index: {} -> {}",
                                               import_info.imported_path, resolved_path);
                                    None
                                }
                            }
                        } else {
                            log::trace!("Could not resolve Rust import (external or stdlib): {}", import_info.imported_path);
                            None
                        }
                    } else if file_path.ends_with(".java") && !java_projects.is_empty() {
                        // Resolve Java dependencies using project mappings
                        if let Some(resolved_path) = crate::parsers::java::resolve_java_import_to_path(
                            &import_info.imported_path,
                            &java_projects,
                            Some(&file_path),
                        ) {
                            // Look up file ID in database using exact match
                            match dep_index.get_file_id_by_path(&resolved_path)? {
                                Some(id) => {
                                    log::trace!("Resolved Java dependency: {} -> {} (file_id={})",
                                               import_info.imported_path, resolved_path, id);
                                    Some(id)
                                }
                                None => {
                                    log::trace!("Java dependency resolved to path but file not in index: {} -> {}",
                                               import_info.imported_path, resolved_path);
                                    None
                                }
                            }
                        } else {
                            log::trace!("Could not resolve Java import: {}", import_info.imported_path);
                            None
                        }
                    } else if (file_path.ends_with(".kt") || file_path.ends_with(".kts")) && !java_projects.is_empty() {
                        // Resolve Kotlin dependencies using project mappings (same build systems as Java)
                        if let Some(resolved_path) = crate::parsers::java::resolve_kotlin_import_to_path(
                            &import_info.imported_path,
                            &java_projects,
                            Some(&file_path),
                        ) {
                            // Look up file ID in database using exact match
                            match dep_index.get_file_id_by_path(&resolved_path)? {
                                Some(id) => {
                                    log::trace!("Resolved Kotlin dependency: {} -> {} (file_id={})",
                                               import_info.imported_path, resolved_path, id);
                                    Some(id)
                                }
                                None => {
                                    log::trace!("Kotlin dependency resolved to path but file not in index: {} -> {}",
                                               import_info.imported_path, resolved_path);
                                    None
                                }
                            }
                        } else {
                            log::trace!("Could not resolve Kotlin import: {}", import_info.imported_path);
                            None
                        }
                    } else if (file_path.ends_with(".rb") || file_path.ends_with(".rake") || file_path.ends_with(".gemspec")) && !ruby_projects.is_empty() {
                        // Resolve Ruby dependencies using project mappings
                        if let Some(resolved_path) = crate::parsers::ruby::resolve_ruby_require_to_path(
                            &import_info.imported_path,
                            &ruby_projects,
                            Some(&file_path),
                        ) {
                            // Look up file ID in database using exact match
                            match dep_index.get_file_id_by_path(&resolved_path)? {
                                Some(id) => {
                                    log::trace!("Resolved Ruby dependency: {} -> {} (file_id={})",
                                               import_info.imported_path, resolved_path, id);
                                    Some(id)
                                }
                                None => {
                                    log::trace!("Ruby dependency resolved to path but file not in index: {} -> {}",
                                               import_info.imported_path, resolved_path);
                                    None
                                }
                            }
                        } else {
                            log::trace!("Could not resolve Ruby require: {}", import_info.imported_path);
                            None
                        }
                    } else if file_path.ends_with(".c") || file_path.ends_with(".h") {
                        // Resolve C dependencies (relative #include paths)
                        if let Some(resolved_path) = crate::parsers::c::resolve_c_include_to_path(
                            &import_info.imported_path,
                            Some(&file_path),
                        ) {
                            // Look up file ID in database using exact match
                            match dep_index.get_file_id_by_path(&resolved_path)? {
                                Some(id) => {
                                    log::trace!("Resolved C dependency: {} -> {} (file_id={})",
                                               import_info.imported_path, resolved_path, id);
                                    Some(id)
                                }
                                None => {
                                    log::trace!("C dependency resolved to path but file not in index: {} -> {}",
                                               import_info.imported_path, resolved_path);
                                    None
                                }
                            }
                        } else {
                            log::trace!("Could not resolve C include (system header): {}", import_info.imported_path);
                            None
                        }
                    } else if file_path.ends_with(".cpp") || file_path.ends_with(".cc") || file_path.ends_with(".cxx")
                           || file_path.ends_with(".hpp") || file_path.ends_with(".hxx") || file_path.ends_with(".h++")
                           || file_path.ends_with(".C") || file_path.ends_with(".H") {
                        // Resolve C++ dependencies (relative #include paths)
                        if let Some(resolved_path) = crate::parsers::cpp::resolve_cpp_include_to_path(
                            &import_info.imported_path,
                            Some(&file_path),
                        ) {
                            // Look up file ID in database using exact match
                            match dep_index.get_file_id_by_path(&resolved_path)? {
                                Some(id) => {
                                    log::trace!("Resolved C++ dependency: {} -> {} (file_id={})",
                                               import_info.imported_path, resolved_path, id);
                                    Some(id)
                                }
                                None => {
                                    log::trace!("C++ dependency resolved to path but file not in index: {} -> {}",
                                               import_info.imported_path, resolved_path);
                                    None
                                }
                            }
                        } else {
                            log::trace!("Could not resolve C++ include (system header): {}", import_info.imported_path);
                            None
                        }
                    } else if file_path.ends_with(".cs") {
                        // Resolve C# dependencies (using namespace-to-path mapping)
                        if let Some(resolved_path) = crate::parsers::csharp::resolve_csharp_using_to_path(
                            &import_info.imported_path,
                            Some(&file_path),
                        ) {
                            // Look up file ID in database using exact match
                            match dep_index.get_file_id_by_path(&resolved_path)? {
                                Some(id) => {
                                    log::trace!("Resolved C# dependency: {} -> {} (file_id={})",
                                               import_info.imported_path, resolved_path, id);
                                    Some(id)
                                }
                                None => {
                                    log::trace!("C# dependency resolved to path but file not in index: {} -> {}",
                                               import_info.imported_path, resolved_path);
                                    None
                                }
                            }
                        } else {
                            log::trace!("Could not resolve C# using directive: {}", import_info.imported_path);
                            None
                        }
                    } else if file_path.ends_with(".zig") {
                        // Resolve Zig dependencies (relative @import paths)
                        if let Some(resolved_path) = crate::parsers::zig::resolve_zig_import_to_path(
                            &import_info.imported_path,
                            Some(&file_path),
                        ) {
                            // Look up file ID in database using exact match
                            match dep_index.get_file_id_by_path(&resolved_path)? {
                                Some(id) => {
                                    log::trace!("Resolved Zig dependency: {} -> {} (file_id={})",
                                               import_info.imported_path, resolved_path, id);
                                    Some(id)
                                }
                                None => {
                                    log::trace!("Zig dependency resolved to path but file not in index: {} -> {}",
                                               import_info.imported_path, resolved_path);
                                    None
                                }
                            }
                        } else {
                            log::trace!("Could not resolve Zig import (external or stdlib): {}", import_info.imported_path);
                            None
                        }
                    } else if file_path.ends_with(".vue") || file_path.ends_with(".svelte") {
                        // Resolve Vue/Svelte dependencies (use TypeScript/JavaScript resolver for imports in <script> blocks)
                        let alias_map = find_nearest_tsconfig(&file_path, root, &tsconfigs);
                        if let Some(candidates_str) = crate::parsers::typescript::resolve_ts_import_to_path(
                            &import_info.imported_path,
                            Some(&file_path),
                            alias_map,
                        ) {
                            // Parse pipe-delimited candidates (e.g., "path.tsx|path.ts|path.jsx|path.js")
                            let candidates: Vec<&str> = candidates_str.split('|').collect();

                            // Try each candidate in order until we find one in the database
                            let mut resolved_id = None;
                            for candidate_path in candidates {
                                // Normalize path to be relative to project root
                                // Convert absolute paths to relative (without requiring file to exist)
                                let normalized_candidate = if let Ok(rel_path) = std::path::Path::new(candidate_path).strip_prefix(root) {
                                    rel_path.to_string_lossy().to_string()
                                } else {
                                    // Not an absolute path or not under root - use as-is
                                    candidate_path.to_string()
                                };

                                match dep_index.get_file_id_by_path(&normalized_candidate)? {
                                    Some(id) => {
                                        log::trace!("Resolved Vue/Svelte dependency: {} -> {} (file_id={})",
                                                   import_info.imported_path, candidate_path, id);
                                        resolved_id = Some(id);
                                        break; // Found a match, stop trying
                                    }
                                    None => {
                                        log::trace!("Vue/Svelte candidate not in index: {}", candidate_path);
                                    }
                                }
                            }

                            if resolved_id.is_none() {
                                log::trace!("Vue/Svelte dependency: no matching file found in database for any candidate: {}",
                                           candidates_str);
                            }

                            resolved_id
                        } else {
                            log::trace!("Could not resolve Vue/Svelte import (non-relative or external): {}", import_info.imported_path);
                            None
                        }
                    } else {
                        None
                    };

                    // resolved_file_id will be populated using deterministic language-specific resolution
                    // All language resolvers have been implemented!
                    resolved_deps.push(Dependency {
                        file_id,
                        imported_path: import_info.imported_path.clone(),
                        resolved_file_id,
                        import_type: import_info.import_type,
                        line_number: import_info.line_number,
                        imported_symbols: import_info.imported_symbols.clone(),
                    });
                }

                // Clear existing dependencies for this file (incremental reindex)
                dep_index.clear_dependencies(file_id)?;

                // Batch insert dependencies
                if !resolved_deps.is_empty() {
                    dep_index.batch_insert_dependencies(&resolved_deps)?;
                    total_deps_inserted += resolved_deps.len();
                }
            }

            log::info!("Extracted {} dependencies", total_deps_inserted);
        }

        // Step 2.6: Insert exports (after files are inserted and have IDs)
        if !all_exports.is_empty() {
            *progress_status.lock().unwrap() = "Extracting exports...".to_string();
            if show_progress {
                pb.set_message("Extracting exports...".to_string());
            }

            // Reuse the tsconfigs parsed earlier for TypeScript/Vue path alias resolution
            let tsconfigs = crate::parsers::tsconfig::parse_all_tsconfigs(root)
                .unwrap_or_else(|e| {
                    log::warn!("Failed to parse tsconfig.json files: {}", e);
                    HashMap::new()
                });

            // Create dependency index to resolve paths and insert exports
            let cache_for_exports = CacheManager::new(root);
            let dep_index = DependencyIndex::new(cache_for_exports);

            let mut total_exports_inserted = 0;

            // Process each file's exports
            for (file_path, export_infos) in all_exports {
                // Get file ID from database
                let file_id = match dep_index.get_file_id_by_path(&file_path)? {
                    Some(id) => id,
                    None => {
                        log::warn!("File not found in database (skipping exports): {}", file_path);
                        continue;
                    }
                };

                // Resolve export source paths and insert
                for export_info in export_infos {
                    // Resolve export source path (same logic as imports)
                    let resolved_source_id = if file_path.ends_with(".ts") || file_path.ends_with(".tsx")
                            || file_path.ends_with(".js") || file_path.ends_with(".jsx")
                            || file_path.ends_with(".mts") || file_path.ends_with(".cts")
                            || file_path.ends_with(".mjs") || file_path.ends_with(".cjs")
                            || file_path.ends_with(".vue") {
                        // Resolve TypeScript/JavaScript/Vue export paths (relative imports and path aliases)
                        let alias_map = find_nearest_tsconfig(&file_path, root, &tsconfigs);
                        if let Some(candidates_str) = crate::parsers::typescript::resolve_ts_import_to_path(
                            &export_info.source_path,
                            Some(&file_path),
                            alias_map,
                        ) {
                            // Parse pipe-delimited candidates (e.g., "path.tsx|path.ts|path.jsx|path.js|path.vue")
                            let candidates: Vec<&str> = candidates_str.split('|').collect();

                            // Try each candidate in order until we find one in the database
                            let mut resolved_id = None;
                            for candidate_path in candidates {
                                // Normalize path to be relative to project root
                                let normalized_candidate = if let Ok(rel_path) = std::path::Path::new(candidate_path).strip_prefix(root) {
                                    rel_path.to_string_lossy().to_string()
                                } else {
                                    candidate_path.to_string()
                                };

                                match dep_index.get_file_id_by_path(&normalized_candidate)? {
                                    Some(id) => {
                                        log::trace!("Resolved export source: {} -> {} (file_id={})",
                                                   export_info.source_path, normalized_candidate, id);
                                        resolved_id = Some(id);
                                        break; // Found a match, stop trying
                                    }
                                    None => {
                                        log::trace!("Export source candidate not in index: {}", candidate_path);
                                    }
                                }
                            }

                            if resolved_id.is_none() {
                                log::trace!("Export source: no matching file found in database for any candidate: {}",
                                           candidates_str);
                            }

                            resolved_id
                        } else {
                            log::trace!("Could not resolve export source (non-relative or external): {}", export_info.source_path);
                            None
                        }
                    } else {
                        None
                    };

                    // Insert export into database
                    dep_index.insert_export(
                        file_id,
                        export_info.exported_symbol,
                        export_info.source_path,
                        resolved_source_id,
                        export_info.line_number,
                    )?;

                    total_exports_inserted += 1;
                }
            }

            log::info!("Extracted {} exports", total_exports_inserted);
        }

        log::info!("Indexed {} files", files_indexed);

        // Step 3: Write trigram index
        *progress_status.lock().unwrap() = "Writing trigram index...".to_string();
        if show_progress {
            pb.set_message("Writing trigram index...".to_string());
        }
        let trigrams_path = self.cache.path().join("trigrams.bin");
        log::info!("Writing trigram index with {} trigrams to trigrams.bin",
                   trigram_index.trigram_count());

        trigram_index.write(&trigrams_path)
            .context("Failed to write trigram index")?;
        log::info!("Wrote {} files to trigrams.bin", trigram_index.file_count());

        // Step 4: Finalize content store (already been writing incrementally)
        *progress_status.lock().unwrap() = "Finalizing content store...".to_string();
        if show_progress {
            pb.set_message("Finalizing content store...".to_string());
        }
        content_writer.finalize_if_needed()
            .context("Failed to finalize content store")?;
        log::info!("Wrote {} files ({} bytes) to content.bin",
                   content_writer.file_count(), content_writer.content_size());

        // Step 5: Update SQLite statistics from database totals (branch-aware)
        *progress_status.lock().unwrap() = "Updating statistics...".to_string();
        if show_progress {
            pb.set_message("Updating statistics...".to_string());
        }
        // Update stats for current branch only
        self.cache.update_stats(&branch)?;

        // Update schema hash to mark cache as compatible with current binary
        self.cache.update_schema_hash()?;

        pb.finish_with_message("Indexing complete");

        // Return stats
        let stats = self.cache.stats()?;
        log::info!("Indexing complete: {} files",
                   stats.total_files);

        Ok(stats)
    }

    /// Discover all indexable files in the directory tree
    fn discover_files(&self, root: &Path) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        // WalkBuilder from ignore crate automatically respects:
        // - .gitignore (when in a git repo)
        // - .ignore files
        // - Hidden files (can be configured)
        let walker = WalkBuilder::new(root)
            .follow_links(self.config.follow_symlinks)
            .git_ignore(true)  // Explicitly enable gitignore support (enabled by default, but be explicit)
            .git_global(false) // Don't use global gitignore
            .git_exclude(false) // Don't use .git/info/exclude
            .build();

        for entry in walker {
            let entry = entry?;
            let path = entry.path();

            // Only process files (not directories)
            if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                continue;
            }

            // Check if should be indexed
            if self.should_index(path) {
                files.push(path.to_path_buf());
            }
        }

        Ok(files)
    }

    /// Check if a file should be indexed based on config
    fn should_index(&self, path: &Path) -> bool {
        // Check file extension for supported languages
        let ext = match path.extension() {
            Some(ext) => ext.to_string_lossy(),
            None => return false,
        };

        let lang = Language::from_extension(&ext);

        // Only index files for languages with parser implementations
        if !lang.is_supported() {
            if !matches!(lang, Language::Unknown) {
                log::debug!("Skipping {} ({:?} parser not yet implemented)",
                           path.display(), lang);
            }
            return false;
        }

        // Check file size limits
        if let Ok(metadata) = std::fs::metadata(path) {
            if metadata.len() > self.config.max_file_size as u64 {
                log::debug!("Skipping {} (too large: {} bytes)",
                           path.display(), metadata.len());
                return false;
            }
        }

        // TODO: Check include/exclude patterns when glob support is added
        // For now, accept all files with supported language extensions

        true
    }

    /// Compute blake3 hash from file contents for change detection
    fn hash_content(&self, content: &[u8]) -> String {
        let hash = blake3::hash(content);
        hash.to_hex().to_string()
    }

    /// Check available disk space before indexing
    ///
    /// Ensures there's enough free space to create the index. Warns if disk space is low.
    /// This prevents partial index writes and confusing error messages.
    fn check_disk_space(&self, root: &Path) -> Result<()> {
        // Get available space on the filesystem containing the cache directory
        let cache_path = self.cache.path();

        // Use statvfs on Unix systems
        #[cfg(unix)]
        {
            // On Linux, we can use statvfs to get available space
            // For now, we'll use a simple heuristic: warn if we can't write a test file
            let test_file = cache_path.join(".space_check");
            match std::fs::write(&test_file, b"test") {
                Ok(_) => {
                    let _ = std::fs::remove_file(&test_file);

                    // Try to estimate available space using df command
                    if let Ok(output) = std::process::Command::new("df")
                        .arg("-k")
                        .arg(cache_path.parent().unwrap_or(root))
                        .output()
                    {
                        if let Ok(df_output) = String::from_utf8(output.stdout) {
                            // Parse df output to get available KB
                            if let Some(line) = df_output.lines().nth(1) {
                                let parts: Vec<&str> = line.split_whitespace().collect();
                                if parts.len() >= 4 {
                                    if let Ok(available_kb) = parts[3].parse::<u64>() {
                                        let available_mb = available_kb / 1024;

                                        // Warn if less than 100MB available
                                        if available_mb < 100 {
                                            log::warn!("Low disk space: only {}MB available. Indexing may fail.", available_mb);
                                            output::warn(&format!("Low disk space ({}MB available). Consider freeing up space.", available_mb));
                                        } else {
                                            log::debug!("Available disk space: {}MB", available_mb);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Ok(())
                }
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                    anyhow::bail!(
                        "Permission denied writing to cache directory: {}. Check file permissions.",
                        cache_path.display()
                    )
                }
                Err(e) => {
                    // If we can't write, it might be a disk space issue
                    log::warn!("Failed to write test file (possible disk space issue): {}", e);
                    Err(e).context("Failed to verify disk space - indexing may fail due to insufficient space")
                }
            }
        }

        #[cfg(not(unix))]
        {
            // On Windows, try to write a test file
            let test_file = cache_path.join(".space_check");
            match std::fs::write(&test_file, b"test") {
                Ok(_) => {
                    let _ = std::fs::remove_file(&test_file);
                    Ok(())
                }
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                    anyhow::bail!(
                        "Permission denied writing to cache directory: {}. Check file permissions.",
                        cache_path.display()
                    )
                }
                Err(e) => {
                    log::warn!("Failed to write test file (possible disk space issue): {}", e);
                    Err(e).context("Failed to verify disk space - indexing may fail due to insufficient space")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;

    #[test]
    fn test_indexer_creation() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        assert!(indexer.cache.path().ends_with(".reflex"));
    }

    #[test]
    fn test_hash_content() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        let content1 = b"hello world";
        let content2 = b"hello world";
        let content3 = b"different content";

        let hash1 = indexer.hash_content(content1);
        let hash2 = indexer.hash_content(content2);
        let hash3 = indexer.hash_content(content3);

        // Same content should produce same hash
        assert_eq!(hash1, hash2);

        // Different content should produce different hash
        assert_ne!(hash1, hash3);

        // Hash should be hex string
        assert_eq!(hash1.len(), 64); // blake3 hash is 32 bytes = 64 hex chars
    }

    #[test]
    fn test_should_index_rust_file() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        // Create a small Rust file
        let rust_file = temp.path().join("test.rs");
        fs::write(&rust_file, "fn main() {}").unwrap();

        assert!(indexer.should_index(&rust_file));
    }

    #[test]
    fn test_should_index_unsupported_extension() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        let unsupported_file = temp.path().join("test.txt");
        fs::write(&unsupported_file, "plain text").unwrap();

        assert!(!indexer.should_index(&unsupported_file));
    }

    #[test]
    fn test_should_index_no_extension() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        let no_ext_file = temp.path().join("Makefile");
        fs::write(&no_ext_file, "all:\n\techo hello").unwrap();

        assert!(!indexer.should_index(&no_ext_file));
    }

    #[test]
    fn test_should_index_size_limit() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        // Config with 100 byte size limit
        let mut config = IndexConfig::default();
        config.max_file_size = 100;

        let indexer = Indexer::new(cache, config);

        // Create small file (should be indexed)
        let small_file = temp.path().join("small.rs");
        fs::write(&small_file, "fn main() {}").unwrap();
        assert!(indexer.should_index(&small_file));

        // Create large file (should be skipped)
        let large_file = temp.path().join("large.rs");
        let large_content = "a".repeat(150);
        fs::write(&large_file, large_content).unwrap();
        assert!(!indexer.should_index(&large_file));
    }

    #[test]
    fn test_discover_files_empty_dir() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        let files = indexer.discover_files(temp.path()).unwrap();
        assert_eq!(files.len(), 0);
    }

    #[test]
    fn test_discover_files_single_file() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        // Create a Rust file
        let rust_file = temp.path().join("main.rs");
        fs::write(&rust_file, "fn main() {}").unwrap();

        let files = indexer.discover_files(temp.path()).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("main.rs"));
    }

    #[test]
    fn test_discover_files_multiple_languages() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        // Create files of different languages
        fs::write(temp.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(temp.path().join("script.py"), "print('hello')").unwrap();
        fs::write(temp.path().join("app.js"), "console.log('hi')").unwrap();
        fs::write(temp.path().join("README.md"), "# Project").unwrap(); // Should be skipped

        let files = indexer.discover_files(temp.path()).unwrap();
        assert_eq!(files.len(), 3); // Only supported languages
    }

    #[test]
    fn test_discover_files_subdirectories() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        // Create nested directory structure
        let src_dir = temp.path().join("src");
        fs::create_dir(&src_dir).unwrap();
        fs::write(src_dir.join("main.rs"), "fn main() {}").unwrap();
        fs::write(src_dir.join("lib.rs"), "pub mod test {}").unwrap();

        let tests_dir = temp.path().join("tests");
        fs::create_dir(&tests_dir).unwrap();
        fs::write(tests_dir.join("test.rs"), "#[test] fn test() {}").unwrap();

        let files = indexer.discover_files(temp.path()).unwrap();
        assert_eq!(files.len(), 3);
    }

    #[test]
    fn test_discover_files_respects_gitignore() {
        let temp = TempDir::new().unwrap();

        // Initialize git repo (required for .gitignore to work with WalkBuilder)
        std::process::Command::new("git")
            .arg("init")
            .current_dir(temp.path())
            .output()
            .expect("Failed to initialize git repo");

        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        // Create .gitignore - use "ignored/" pattern to ignore the directory
        // Note: WalkBuilder respects .gitignore ONLY in git repositories
        fs::write(temp.path().join(".gitignore"), "ignored/\n").unwrap();

        // Create files
        fs::write(temp.path().join("included.rs"), "fn main() {}").unwrap();
        fs::write(temp.path().join("also_included.py"), "print('hi')").unwrap();

        let ignored_dir = temp.path().join("ignored");
        fs::create_dir(&ignored_dir).unwrap();
        fs::write(ignored_dir.join("excluded.rs"), "fn test() {}").unwrap();

        let files = indexer.discover_files(temp.path()).unwrap();

        // Verify the expected files are found
        assert!(files.iter().any(|f| f.ends_with("included.rs")), "Should find included.rs");
        assert!(files.iter().any(|f| f.ends_with("also_included.py")), "Should find also_included.py");

        // Verify excluded.rs in ignored/ directory is NOT found
        // This is the key test - gitignore should filter it out
        assert!(!files.iter().any(|f| {
            let path_str = f.to_string_lossy();
            path_str.contains("ignored") && f.ends_with("excluded.rs")
        }), "Should NOT find excluded.rs in ignored/ directory (gitignore pattern)");

        // Should find exactly 2 files (included.rs and also_included.py)
        // .gitignore file itself has no supported language extension, so it won't be indexed
        assert_eq!(files.len(), 2, "Should find exactly 2 files (not including .gitignore or ignored/excluded.rs)");
    }

    #[test]
    fn test_index_empty_directory() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        let stats = indexer.index(temp.path(), false).unwrap();

        assert_eq!(stats.total_files, 0);
    }

    #[test]
    fn test_index_single_rust_file() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        // Create a Rust file
        fs::write(
            project_root.join("main.rs"),
            "fn main() { println!(\"Hello\"); }"
        ).unwrap();

        let stats = indexer.index(&project_root, false).unwrap();

        assert_eq!(stats.total_files, 1);
        assert!(stats.files_by_language.get("Rust").is_some());
    }

    #[test]
    fn test_index_multiple_files() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        // Create multiple files
        fs::write(project_root.join("main.rs"), "fn main() {}").unwrap();
        fs::write(project_root.join("lib.rs"), "pub fn test() {}").unwrap();
        fs::write(project_root.join("script.py"), "def main(): pass").unwrap();

        let stats = indexer.index(&project_root, false).unwrap();

        assert_eq!(stats.total_files, 3);
        assert_eq!(stats.files_by_language.get("Rust"), Some(&2));
        assert_eq!(stats.files_by_language.get("Python"), Some(&1));
    }

    #[test]
    fn test_index_creates_trigram_index() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        fs::write(project_root.join("main.rs"), "fn main() {}").unwrap();

        indexer.index(&project_root, false).unwrap();

        // Verify trigrams.bin was created
        let trigrams_path = project_root.join(".reflex/trigrams.bin");
        assert!(trigrams_path.exists());
    }

    #[test]
    fn test_index_creates_content_store() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        fs::write(project_root.join("main.rs"), "fn main() {}").unwrap();

        indexer.index(&project_root, false).unwrap();

        // Verify content.bin was created
        let content_path = project_root.join(".reflex/content.bin");
        assert!(content_path.exists());
    }

    #[test]
    fn test_index_incremental_no_changes() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        fs::write(project_root.join("main.rs"), "fn main() {}").unwrap();

        // First index
        let stats1 = indexer.index(&project_root, false).unwrap();
        assert_eq!(stats1.total_files, 1);

        // Second index without changes
        let stats2 = indexer.index(&project_root, false).unwrap();
        assert_eq!(stats2.total_files, 1);
    }

    #[test]
    fn test_index_incremental_with_changes() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        let main_path = project_root.join("main.rs");
        fs::write(&main_path, "fn main() {}").unwrap();

        // First index
        indexer.index(&project_root, false).unwrap();

        // Modify file
        fs::write(&main_path, "fn main() { println!(\"changed\"); }").unwrap();

        // Second index should detect change
        let stats = indexer.index(&project_root, false).unwrap();
        assert_eq!(stats.total_files, 1);
    }

    #[test]
    fn test_index_incremental_new_file() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        fs::write(project_root.join("main.rs"), "fn main() {}").unwrap();

        // First index
        let stats1 = indexer.index(&project_root, false).unwrap();
        assert_eq!(stats1.total_files, 1);

        // Add new file
        fs::write(project_root.join("lib.rs"), "pub fn test() {}").unwrap();

        // Second index should include new file
        let stats2 = indexer.index(&project_root, false).unwrap();
        assert_eq!(stats2.total_files, 2);
    }

    #[test]
    fn test_index_parallel_threads_config() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);

        // Test with explicit thread count
        let mut config = IndexConfig::default();
        config.parallel_threads = 2;

        let indexer = Indexer::new(cache, config);

        fs::write(project_root.join("main.rs"), "fn main() {}").unwrap();

        let stats = indexer.index(&project_root, false).unwrap();
        assert_eq!(stats.total_files, 1);
    }

    #[test]
    fn test_index_parallel_threads_auto() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);

        // Test with auto thread count (0 = auto)
        let mut config = IndexConfig::default();
        config.parallel_threads = 0;

        let indexer = Indexer::new(cache, config);

        fs::write(project_root.join("main.rs"), "fn main() {}").unwrap();

        let stats = indexer.index(&project_root, false).unwrap();
        assert_eq!(stats.total_files, 1);
    }

    #[test]
    fn test_index_respects_size_limit() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);

        // Very small size limit
        let mut config = IndexConfig::default();
        config.max_file_size = 50;

        let indexer = Indexer::new(cache, config);

        // Small file (should be indexed)
        fs::write(project_root.join("small.rs"), "fn a() {}").unwrap();

        // Large file (should be skipped)
        let large_content = "fn main() {}\n".repeat(10);
        fs::write(project_root.join("large.rs"), large_content).unwrap();

        let stats = indexer.index(&project_root, false).unwrap();

        // Only small file should be indexed
        assert_eq!(stats.total_files, 1);
    }

    #[test]
    fn test_index_mixed_languages() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        // Create files in multiple languages
        fs::write(project_root.join("main.rs"), "fn main() {}").unwrap();
        fs::write(project_root.join("test.py"), "def test(): pass").unwrap();
        fs::write(project_root.join("app.js"), "function main() {}").unwrap();
        fs::write(project_root.join("lib.go"), "func main() {}").unwrap();

        let stats = indexer.index(&project_root, false).unwrap();

        assert_eq!(stats.total_files, 4);
        assert!(stats.files_by_language.contains_key("Rust"));
        assert!(stats.files_by_language.contains_key("Python"));
        assert!(stats.files_by_language.contains_key("JavaScript"));
        assert!(stats.files_by_language.contains_key("Go"));
    }

    #[test]
    fn test_index_updates_cache_stats() {
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("project");
        fs::create_dir(&project_root).unwrap();

        let cache = CacheManager::new(&project_root);
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);

        fs::write(project_root.join("main.rs"), "fn main() {}").unwrap();

        indexer.index(&project_root, false).unwrap();

        // Verify cache stats were updated
        let cache = CacheManager::new(&project_root);
        let stats = cache.stats().unwrap();

        assert_eq!(stats.total_files, 1);
        assert!(stats.index_size_bytes > 0);
    }
}
