//! LLM response cache for Pulse narration
//!
//! Caches LLM-generated summaries keyed by structural context hash.
//! Same structural inputs → cache hit, regardless of LLM provider or model.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A cached LLM response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResponse {
    pub context_hash: String,
    pub response: String,
    pub timestamp: String,
}

/// LLM response cache manager
pub struct LlmCache {
    cache_dir: PathBuf,
}

impl LlmCache {
    /// Create a new LLM cache at the given directory
    pub fn new(reflex_cache_path: &Path) -> Self {
        Self {
            cache_dir: reflex_cache_path.join("pulse").join("llm-cache"),
        }
    }

    /// Compute a cache key from structural context
    ///
    /// Key: blake3(snapshot_id + module_path + structural_context_hash)
    pub fn compute_key(snapshot_id: &str, module_path: &str, context: &str) -> String {
        let input = format!("{}:{}:{}", snapshot_id, module_path, context);
        blake3::hash(input.as_bytes()).to_hex().to_string()
    }

    /// Look up a cached response
    pub fn get(&self, key: &str) -> Result<Option<CachedResponse>> {
        let path = self.cache_dir.join(format!("{}.json", key));
        if !path.exists() {
            return Ok(None);
        }

        let content = std::fs::read_to_string(&path)
            .context("Failed to read LLM cache entry")?;
        let cached: CachedResponse = serde_json::from_str(&content)
            .context("Failed to parse LLM cache entry")?;
        Ok(Some(cached))
    }

    /// Store a response in the cache
    pub fn put(&self, key: &str, context_hash: &str, response: &str) -> Result<()> {
        std::fs::create_dir_all(&self.cache_dir)
            .context("Failed to create LLM cache directory")?;

        let entry = CachedResponse {
            context_hash: context_hash.to_string(),
            response: response.to_string(),
            timestamp: chrono::Local::now().to_rfc3339(),
        };

        let json = serde_json::to_string_pretty(&entry)?;
        let path = self.cache_dir.join(format!("{}.json", key));
        std::fs::write(&path, json)
            .context("Failed to write LLM cache entry")?;

        Ok(())
    }

    /// Clear all cached responses
    pub fn clear(&self) -> Result<()> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir)
                .context("Failed to clear LLM cache")?;
        }
        Ok(())
    }

    /// Count cached entries
    pub fn count(&self) -> usize {
        if !self.cache_dir.exists() {
            return 0;
        }
        std::fs::read_dir(&self.cache_dir)
            .map(|entries| entries.filter(|e| e.is_ok()).count())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_determinism() {
        let key1 = LlmCache::compute_key("snap1", "src", "context_abc");
        let key2 = LlmCache::compute_key("snap1", "src", "context_abc");
        assert_eq!(key1, key2);

        let key3 = LlmCache::compute_key("snap1", "src", "context_different");
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cache_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let cache = LlmCache::new(dir.path());

        let key = "test_key_123";
        assert!(cache.get(key).unwrap().is_none());
        assert_eq!(cache.count(), 0);

        cache.put(key, "hash123", "This module handles authentication.").unwrap();
        assert_eq!(cache.count(), 1);

        let cached = cache.get(key).unwrap().unwrap();
        assert_eq!(cached.response, "This module handles authentication.");
        assert_eq!(cached.context_hash, "hash123");
    }
}
