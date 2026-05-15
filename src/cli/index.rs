use anyhow::{Context, Result};
use std::path::PathBuf;
use crate::cache::CacheManager;
use crate::indexer::Indexer;
use crate::models::Language;


/// Handle the `index status` subcommand
pub(super) fn handle_index_status() -> Result<()> {
    log::info!("Checking background symbol indexing status");

    let cache = CacheManager::new(".");
    let cache_path = cache.path().to_path_buf();

    match crate::background_indexer::BackgroundIndexer::get_status(&cache_path) {
            Ok(Some(status)) => {
                println!("Background Symbol Indexing Status");
                println!("==================================");
                println!("State:           {:?}", status.state);
                println!("Total files:     {}", status.total_files);
                println!("Processed:       {}", status.processed_files);
                println!("Cached:          {}", status.cached_files);
                println!("Parsed:          {}", status.parsed_files);
                println!("Failed:          {}", status.failed_files);
                println!("Started:         {}", status.started_at);
                println!("Last updated:    {}", status.updated_at);

                if let Some(completed_at) = &status.completed_at {
                    println!("Completed:       {}", completed_at);
                }

                if let Some(error) = &status.error {
                    println!("Error:           {}", error);
                }

                // Show progress percentage if running
                if status.state == crate::background_indexer::IndexerState::Running && status.total_files > 0 {
                    let progress = (status.processed_files as f64 / status.total_files as f64) * 100.0;
                    println!("\nProgress:        {:.1}%", progress);
                }

                Ok(())
            }
            Ok(None) => {
                if !cache.exists() {
                    println!("No index cache found in current directory.");
                    println!("\nRun 'rfx index' to build the code search index first.");
                } else {
                    println!("No background symbol indexing in progress.");
                    println!("\nBackground symbol indexing is idle.");
                    println!("Run 'rfx index' to trigger a fresh index build.");
                }
                Ok(())
            }
            Err(e) => {
                anyhow::bail!("Failed to get indexing status: {}", e);
            }
        }
    }


/// Handle the `index compact` subcommand
pub(super) fn handle_index_compact(json: &bool, pretty: &bool) -> Result<()> {
    log::info!("Running cache compaction");

    let cache = CacheManager::new(".");
    let report = cache.compact()?;

    // Output results in requested format
    if *json {
        let json_str = if *pretty {
            serde_json::to_string_pretty(&report)?
        } else {
            serde_json::to_string(&report)?
        };
        println!("{}", json_str);
    } else {
        let space_str = if report.space_saved_bytes == 0 {
            "< 1 KB".to_string()
        } else {
            super::format_bytes(report.space_saved_bytes)
        };
        println!("Cache Compaction Complete");
        println!("=========================");
        println!("Files removed:    {}", report.files_removed);
        println!("Space saved:      {}", space_str);
        println!("Duration:         {}ms", report.duration_ms);
    }

    Ok(())
}


pub(super) fn handle_index_build(path: &PathBuf, force: &bool, languages: &[String], quiet: &bool) -> Result<()> {
    log::info!("Starting index build");

    let cache = CacheManager::new(path);
    let cache_path = cache.path().to_path_buf();

    if *force {
        log::info!("Force rebuild requested, clearing existing cache");
        cache.clear()?;
        if !quiet {
            println!("Cleared existing cache.");
        }
    }

    // Load base config from .reflex/config.toml (or defaults if file absent)
    let mut config = cache.load_index_config()
        .context("Failed to load .reflex/config.toml")?;

    // CLI --languages overrides the config-file value when explicitly provided (REF-98: error on unknown)
    if !languages.is_empty() {
        let lang_filters: Vec<Language> = languages
            .iter()
            .map(|s| {
                Language::from_name(s).ok_or_else(|| anyhow::anyhow!(
                    "Unknown language: '{}'\n\nSupported languages:\n  {}\n\nExample: rfx index --languages rust,python",
                    s, Language::supported_names_help()
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        config.languages = lang_filters;
    }

    let indexer = Indexer::new(cache, config);
    // Show progress by default, unless quiet mode is enabled
    let show_progress = !quiet;
    let stats = indexer.index(path, show_progress)?;

    // In quiet mode, suppress all output
    if !quiet {
        println!("Indexing complete!");
        println!("  Files indexed: {}", stats.total_files);
        println!("  Cache size: {}", super::format_bytes(stats.index_size_bytes));
        println!("  Last updated: {}", stats.last_updated);

        // Show incremental breakdown if available (REF-100)
        let has_breakdown = stats.new_files > 0 || stats.modified_files > 0 || stats.unchanged_files > 0;
        if has_breakdown {
            println!("  Breakdown:     {} new, {} modified, {} unchanged",
                stats.new_files, stats.modified_files, stats.unchanged_files);
        }

        // Warn about skipped large files (REF-99)
        if stats.skipped_too_large > 0 {
            eprintln!("[warn] {} file(s) skipped: exceeded max_file_size ({} total)",
                stats.skipped_too_large,
                super::format_bytes(stats.skipped_bytes_too_large));
        }

        // Display language breakdown if we have indexed files
        if !stats.files_by_language.is_empty() {
            println!("\nFiles by language:");

            // Sort languages by count (descending) for consistent output
            let mut lang_vec: Vec<_> = stats.files_by_language.iter().collect();
            lang_vec.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));

            // Calculate column widths
            let max_lang_len = lang_vec.iter().map(|(lang, _)| lang.len()).max().unwrap_or(8);
            let lang_width = max_lang_len.max(8); // At least "Language" header width

            // Print table header
            println!("  {:<width$}  Files  Lines", "Language", width = lang_width);
            println!("  {}  -----  -------", "-".repeat(lang_width));

            // Print rows
            for (language, file_count) in lang_vec {
                let line_count = stats.lines_by_language.get(language).copied().unwrap_or(0);
                println!("  {:<width$}  {:5}  {:7}",
                    language, file_count, line_count,
                    width = lang_width);
            }
        }
    }

    // Start background symbol indexing (if not already running)
    if !crate::background_indexer::BackgroundIndexer::is_running(&cache_path) {
        if !quiet {
            println!("\nStarting background symbol indexing...");
            println!("  Symbols will be cached for faster queries");
            println!("  Check status with: rfx index status");
        }

        // Spawn detached background process for symbol indexing
        // Pass the workspace root, not the .reflex directory
        let current_exe = std::env::current_exe()
            .context("Failed to get current executable path")?;

        #[cfg(unix)]
        {
            std::process::Command::new(&current_exe)
                .arg("index-symbols-internal")
                .arg(path)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .context("Failed to spawn background indexing process")?;
        }

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;

            std::process::Command::new(&current_exe)
                .arg("index-symbols-internal")
                .arg(&path)
                .creation_flags(CREATE_NO_WINDOW)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .context("Failed to spawn background indexing process")?;
        }

        log::debug!("Spawned background symbol indexing process");
    } else if !quiet {
        println!("\n⚠️  Background symbol indexing already in progress");
        println!("  Check status with: rfx index status");
    }

    Ok(())
}


/// Handle the internal `index-symbols-internal` command
pub(super) fn handle_index_symbols_internal(cache_dir: PathBuf) -> Result<()> {
    let mut indexer = crate::background_indexer::BackgroundIndexer::new(&cache_dir)?;
    indexer.run()?;
    Ok(())
}
