use crate::cache::CacheManager;
use crate::indexer::Indexer;
use crate::models::IndexConfig;
use anyhow::Result;
use std::path::PathBuf;

/// Handle the `watch` subcommand
pub(super) fn handle_watch(path: PathBuf, debounce_ms: u64, quiet: bool) -> Result<()> {
    log::info!("Starting watch mode for {:?}", path);

    // Validate debounce range (5s - 30s)
    if !(5000..=30000).contains(&debounce_ms) {
        anyhow::bail!(
            "Debounce must be between 5000ms (5s) and 30000ms (30s). Got: {}ms",
            debounce_ms
        );
    }

    if !quiet {
        println!("Starting Reflex watch mode...");
        println!("  Directory: {}", path.display());
        println!("  Debounce: {}ms ({}s)", debounce_ms, debounce_ms / 1000);
        println!("  Press Ctrl+C to stop.\n");
    }

    // Setup cache
    let cache = CacheManager::new(&path);

    // Initial index if cache doesn't exist
    if !cache.exists() {
        if !quiet {
            println!("No index found, running initial index...");
        }
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);
        indexer.index(&path, !quiet)?;
        if !quiet {
            println!("Initial index complete. Now watching for changes...\n");
        }
    }

    // Create indexer for watcher
    let cache = CacheManager::new(&path);
    let config = IndexConfig::default();
    let indexer = Indexer::new(cache, config);

    // Start watcher
    let watch_config = crate::watcher::WatchConfig { debounce_ms, quiet };

    crate::watcher::watch(&path, indexer, watch_config)?;

    Ok(())
}
