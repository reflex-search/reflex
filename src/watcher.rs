//! File system watcher for automatic reindexing
//!
//! The watcher monitors the workspace for file changes and automatically
//! triggers incremental reindexing with configurable debouncing.

use anyhow::{Context, Result};
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::time::{Duration, Instant};

use crate::indexer::Indexer;
use crate::models::Language;
use crate::output;

/// Configuration for file watching
#[derive(Debug, Clone)]
pub struct WatchConfig {
    /// Debounce duration in milliseconds
    /// Waits this long after the last change before triggering reindex
    pub debounce_ms: u64,
    /// Suppress output (only log errors)
    pub quiet: bool,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            debounce_ms: 15000, // 15 seconds
            quiet: false,
        }
    }
}

/// Watch a directory for file changes and auto-reindex
///
/// This function blocks until interrupted (Ctrl+C).
///
/// # Algorithm
///
/// 1. Set up file system watcher using notify crate
/// 2. Collect file change events into a HashSet (deduplicate)
/// 3. Wait for debounce period after last change
/// 4. Trigger incremental reindex (only changed files)
/// 5. Repeat
///
/// # Debouncing
///
/// The debounce timer resets on every file change event. This batches
/// rapid changes (e.g., multi-file refactors, format-on-save) into a
/// single reindex operation.
///
/// Example timeline:
/// ```text
/// t=0s:  File A changed  [timer starts]
/// t=2s:  File B changed  [timer resets]
/// t=5s:  File C changed  [timer resets]
/// t=20s: Timer expires    [reindex A, B, C]
/// ```
pub fn watch(path: &Path, indexer: Indexer, config: WatchConfig) -> Result<()> {
    log::info!(
        "Starting file watcher for {:?} with {}ms debounce",
        path,
        config.debounce_ms
    );

    // Setup channel for receiving file system events
    let (tx, rx) = channel();

    // Create watcher with default config
    let mut watcher = RecommendedWatcher::new(tx, Config::default())
        .context("Failed to create file watcher")?;

    // Start watching the directory recursively
    watcher
        .watch(path, RecursiveMode::Recursive)
        .context("Failed to start watching directory")?;

    if !config.quiet {
        println!("Watching for changes (debounce: {}s)...", config.debounce_ms / 1000);
    }

    // Track pending file changes
    let mut pending_files: HashSet<PathBuf> = HashSet::new();
    // Deleted file paths that need to be removed from the index.
    // Tracked separately because the file no longer exists on disk and
    // should_watch_file() cannot reliably inspect it.
    let mut pending_deletions: HashSet<PathBuf> = HashSet::new();
    let mut last_event_time: Option<Instant> = None;
    let debounce_duration = Duration::from_millis(config.debounce_ms);

    // Event loop
    loop {
        // Try to receive events with 100ms timeout (allows checking debounce timer)
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Ok(event)) => {
                // Process the file system event
                if let Some((changed_path, is_removal)) = process_event_typed(&event) {
                    if is_removal {
                        // File is gone — we can no longer call should_watch_file() on it,
                        // but we must still reindex so the deleted entry is removed.
                        // Accept any path whose extension suggests a code file OR has no
                        // extension at all (e.g. a deleted directory triggers a broad Remove).
                        let ext = changed_path.extension()
                            .and_then(|e| e.to_str())
                            .unwrap_or("");
                        let is_code = ext.is_empty() || crate::models::Language::from_extension(ext).is_supported();
                        if is_code {
                            log::debug!("Detected removal: {:?}", changed_path);
                            pending_deletions.insert(changed_path);
                            last_event_time = Some(Instant::now());
                        }
                    } else if should_watch_file(&changed_path) {
                        log::debug!("Detected change: {:?}", changed_path);
                        pending_files.insert(changed_path);
                        last_event_time = Some(Instant::now());
                    }
                }
            }
            Ok(Err(e)) => {
                log::warn!("Watch error: {}", e);
            }
            Err(RecvTimeoutError::Timeout) => {
                // Check if debounce period has elapsed
                let has_pending = !pending_files.is_empty() || !pending_deletions.is_empty();
                if let Some(last_time) = last_event_time {
                    if has_pending && last_time.elapsed() >= debounce_duration {
                        // Trigger reindex
                        let total_changes = pending_files.len() + pending_deletions.len();
                        if !config.quiet {
                            if pending_deletions.is_empty() {
                                println!(
                                    "\nDetected {} changed file(s), reindexing...",
                                    pending_files.len()
                                );
                            } else {
                                println!(
                                    "\nDetected {} change(s) ({} deleted), reindexing...",
                                    total_changes,
                                    pending_deletions.len()
                                );
                            }
                        }

                        let start = Instant::now();
                        match indexer.index(path, false) {
                            Ok(stats) => {
                                let elapsed = start.elapsed();
                                if !config.quiet {
                                    println!(
                                        "✓ Reindexed {} files in {:.1}ms\n",
                                        stats.total_files,
                                        elapsed.as_secs_f64() * 1000.0
                                    );
                                }
                                log::info!(
                                    "Reindexed {} files in {:?}",
                                    stats.total_files,
                                    elapsed
                                );
                            }
                            Err(e) => {
                                output::error(&format!("✗ Reindex failed: {}", e));
                                log::error!("Reindex failed: {}", e);
                            }
                        }

                        // Clear pending changes
                        pending_files.clear();
                        pending_deletions.clear();
                        last_event_time = None;
                    }
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                log::info!("Watcher channel disconnected, stopping...");
                break;
            }
        }
    }

    if !config.quiet {
        println!("Watcher stopped.");
    }

    Ok(())
}

/// Process a file system event and extract the changed path together with a
/// flag indicating whether this is a deletion (Remove) event.
///
/// Returns `Some((path, is_removal))`, or `None` for events that should be
/// ignored (metadata-only changes, etc.).
fn process_event_typed(event: &Event) -> Option<(PathBuf, bool)> {
    match event.kind {
        EventKind::Remove(_) => event.paths.first().cloned().map(|p| (p, true)),
        EventKind::Create(_) | EventKind::Modify(_) => {
            event.paths.first().cloned().map(|p| (p, false))
        }
        _ => None,
    }
}

/// Process a file system event and extract the changed path
///
/// Returns None if the event should be ignored (e.g., metadata changes, directory events)
fn process_event(event: &Event) -> Option<PathBuf> {
    process_event_typed(event).map(|(p, _)| p)
}

/// Check if a file should trigger a reindex
///
/// Returns true if the file has a supported language extension
fn should_watch_file(path: &Path) -> bool {
    // Skip hidden files and directories
    if let Some(file_name) = path.file_name() {
        if file_name.to_string_lossy().starts_with('.') {
            return false;
        }
    }

    // Skip directories
    if path.is_dir() {
        return false;
    }

    // Check if file extension is supported
    if let Some(ext) = path.extension() {
        let ext_str = ext.to_string_lossy();
        let lang = Language::from_extension(&ext_str);
        return lang.is_supported();
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_should_watch_rust_file() {
        let temp = TempDir::new().unwrap();
        let rust_file = temp.path().join("test.rs");
        fs::write(&rust_file, "fn main() {}").unwrap();

        assert!(should_watch_file(&rust_file));
    }

    #[test]
    fn test_should_not_watch_unsupported_file() {
        let temp = TempDir::new().unwrap();
        let txt_file = temp.path().join("test.txt");
        fs::write(&txt_file, "plain text").unwrap();

        assert!(!should_watch_file(&txt_file));
    }

    #[test]
    fn test_should_not_watch_hidden_file() {
        let temp = TempDir::new().unwrap();
        let hidden_file = temp.path().join(".hidden.rs");
        fs::write(&hidden_file, "fn main() {}").unwrap();

        assert!(!should_watch_file(&hidden_file));
    }

    #[test]
    fn test_should_not_watch_directory() {
        let temp = TempDir::new().unwrap();
        let dir = temp.path().join("src");
        fs::create_dir(&dir).unwrap();

        assert!(!should_watch_file(&dir));
    }

    #[test]
    fn test_watch_config_default() {
        let config = WatchConfig::default();
        assert_eq!(config.debounce_ms, 15000);
        assert!(!config.quiet);
    }

    #[test]
    fn test_process_event_create() {
        let event = Event {
            kind: EventKind::Create(notify::event::CreateKind::File),
            paths: vec![PathBuf::from("/test/file.rs")],
            attrs: Default::default(),
        };

        let path = process_event(&event);
        assert!(path.is_some());
        assert_eq!(path.unwrap(), PathBuf::from("/test/file.rs"));
    }

    #[test]
    fn test_process_event_modify() {
        let event = Event {
            kind: EventKind::Modify(notify::event::ModifyKind::Data(
                notify::event::DataChange::Any,
            )),
            paths: vec![PathBuf::from("/test/file.rs")],
            attrs: Default::default(),
        };

        let path = process_event(&event);
        assert!(path.is_some());
        assert_eq!(path.unwrap(), PathBuf::from("/test/file.rs"));
    }

    #[test]
    fn test_process_event_access_ignored() {
        let event = Event {
            kind: EventKind::Access(notify::event::AccessKind::Read),
            paths: vec![PathBuf::from("/test/file.rs")],
            attrs: Default::default(),
        };

        let path = process_event(&event);
        assert!(path.is_none());
    }
}
