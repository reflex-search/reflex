//! Result assembly utilities: trigram index reconstruction and file-id resolution

use anyhow::{Context, Result};

use crate::content_store::ContentReader;
use crate::trigram::TrigramIndex;

/// Find a file_id by its path string in the content store.
pub fn find_file_id(content_reader: &ContentReader, target_path: &str) -> Option<u32> {
    for file_id in 0..content_reader.file_count() {
        if let Some(path) = content_reader.get_file_path(file_id as u32) {
            if path.to_string_lossy() == target_path {
                return Some(file_id as u32);
            }
        }
    }
    None
}

/// Rebuild a trigram index from content store (fallback when trigrams.bin is missing).
pub fn rebuild_trigram_index(content_reader: &ContentReader) -> Result<TrigramIndex> {
    log::debug!("Rebuilding trigram index from {} files", content_reader.file_count());
    let mut trigram_index = TrigramIndex::new();

    for file_id in 0..content_reader.file_count() {
        let file_path = content_reader.get_file_path(file_id as u32)
            .context("Invalid file_id")?
            .to_path_buf();
        let content = content_reader.get_file_content(file_id as u32)?;

        let idx = trigram_index.add_file(file_path);
        trigram_index.index_file(idx, content);
    }

    trigram_index.finalize();
    log::debug!("Trigram index rebuilt with {} trigrams", trigram_index.trigram_count());

    Ok(trigram_index)
}

/// Normalize a glob pattern to ensure it has a proper path prefix.
///
/// Examples:
/// - "src/**/*.rs" → "./src/**/*.rs"
/// - "./services/**/*.php" → unchanged
/// - "**/foo" → unchanged
pub fn normalize_glob_pattern(pattern: &str) -> String {
    if pattern.starts_with('.') || pattern.starts_with('/') || pattern.starts_with('*') {
        pattern.to_string()
    } else {
        format!("./{}", pattern)
    }
}
