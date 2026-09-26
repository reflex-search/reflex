//! Content-addressed cache for the Pulse LLM writing pass.
//!
//! The key is the hash of exactly what would be sent: task kind, prompt version,
//! provider, model, output contract, and the full system and user text. There is no
//! snapshot id and no wall clock in it, so an unchanged request hits the cache after
//! any number of re-indexes, on any machine.
//!
//! Layout:
//! ```text
//! <dir>/v1/ab/abcdef….json   one entry per key; no timestamps, stable bytes
//! <dir>/runs.json            keys referenced by the last K successful runs
//! ```
//! Entries are safe to commit or to keep in a CI cache: any older cache is useful.

use crate::atomic_write::{atomic_replace, tmp_path_for};
use crate::semantic::providers::Usage;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const LAYOUT: &str = "v1";
const KEY_SCHEMA: &str = "pulse-write/1";

/// Everything that decides the model's answer. Serialized to JSON and hashed.
#[derive(Serialize)]
pub struct KeyMaterial<'a> {
    pub task_kind: &'a str,
    pub prompt_version: u32,
    /// `None` when `cache_model_agnostic` is set.
    pub provider: Option<&'a str>,
    pub model: Option<&'a str>,
    pub max_tokens: u32,
    pub output_mode: &'a str,
    /// Canonical JSON of the output schema, if any.
    pub output_schema: Option<&'a str>,
    pub system: &'a str,
    pub user: &'a str,
}

impl KeyMaterial<'_> {
    pub fn key(&self) -> String {
        #[derive(Serialize)]
        struct Versioned<'a, 'b> {
            schema: &'static str,
            #[serde(flatten)]
            material: &'a KeyMaterial<'b>,
        }
        let bytes = serde_json::to_vec(&Versioned {
            schema: KEY_SCHEMA,
            material: self,
        })
        .expect("KeyMaterial serializes");
        blake3::hash(&bytes).to_hex().to_string()
    }
}

/// One cached answer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CacheEntry {
    pub key: String,
    pub task_id: String,
    pub task_kind: String,
    pub model: String,
    #[serde(default)]
    pub usage: Option<Usage>,
    pub text: String,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct RunsFile {
    runs: Vec<RunRecord>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RunRecord {
    keys: Vec<String>,
}

/// The write cache rooted at one directory.
#[derive(Debug, Clone)]
pub struct WriteCache {
    dir: PathBuf,
}

impl WriteCache {
    /// Default location: `<reflex cache>/pulse/write-cache`.
    pub fn default_dir(reflex_cache: &Path) -> PathBuf {
        reflex_cache.join("pulse").join("write-cache")
    }

    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn entry_path(&self, key: &str) -> PathBuf {
        let shard = key.get(..2).unwrap_or("00");
        self.dir
            .join(LAYOUT)
            .join(shard)
            .join(format!("{key}.json"))
    }

    /// Read an entry. A missing or unreadable entry is a miss, never an error.
    pub fn get(&self, key: &str) -> Option<CacheEntry> {
        let bytes = std::fs::read(self.entry_path(key)).ok()?;
        match serde_json::from_slice::<CacheEntry>(&bytes) {
            Ok(entry) if entry.key == key => Some(entry),
            Ok(_) => None,
            Err(e) => {
                log::warn!("ignoring corrupt pulse write-cache entry {key}: {e}");
                None
            }
        }
    }

    /// Write an entry atomically (temp file + rename), so concurrent tasks never conflict.
    pub fn put(&self, entry: &CacheEntry) -> Result<()> {
        let path = self.entry_path(&entry.key);
        let parent = path.parent().expect("entry path has a parent");
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
        let tmp = tmp_path_for(&path);
        std::fs::write(&tmp, serde_json::to_vec_pretty(entry)?)
            .with_context(|| format!("writing {}", tmp.display()))?;
        atomic_replace(&tmp, &path)?;
        Ok(())
    }

    /// Record the keys one successful run used, keep the last `keep_runs` records, and
    /// delete every entry none of them reference. Returns the number of entries deleted.
    pub fn record_run_and_prune(
        &self,
        used: impl IntoIterator<Item = String>,
        keep_runs: usize,
    ) -> Result<usize> {
        let runs_path = self.dir.join("runs.json");
        let mut runs: RunsFile = std::fs::read(&runs_path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        let mut keys: Vec<String> = used.into_iter().collect();
        keys.sort();
        keys.dedup();
        runs.runs.push(RunRecord { keys });
        let keep = keep_runs.max(1);
        if runs.runs.len() > keep {
            let drop = runs.runs.len() - keep;
            runs.runs.drain(..drop);
        }

        std::fs::create_dir_all(&self.dir)?;
        let tmp = tmp_path_for(&runs_path);
        std::fs::write(&tmp, serde_json::to_vec_pretty(&runs)?)?;
        atomic_replace(&tmp, &runs_path)?;

        let live: BTreeSet<&str> = runs
            .runs
            .iter()
            .flat_map(|r| r.keys.iter().map(String::as_str))
            .collect();
        let mut deleted = 0;
        let root = self.dir.join(LAYOUT);
        let Ok(shards) = std::fs::read_dir(&root) else {
            return Ok(0);
        };
        for shard in shards.flatten() {
            let Ok(files) = std::fs::read_dir(shard.path()) else {
                continue;
            };
            for file in files.flatten() {
                let path = file.path();
                let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                    continue;
                };
                let is_entry = path.extension().is_some_and(|e| e == "json");
                if is_entry && !live.contains(stem) && std::fs::remove_file(&path).is_ok() {
                    deleted += 1;
                }
            }
        }
        Ok(deleted)
    }

    /// Number of entries on disk.
    pub fn count(&self) -> usize {
        let Ok(shards) = std::fs::read_dir(self.dir.join(LAYOUT)) else {
            return 0;
        };
        shards
            .flatten()
            .filter_map(|s| std::fs::read_dir(s.path()).ok())
            .map(|files| {
                files
                    .flatten()
                    .filter(|f| f.path().extension().is_some_and(|e| e == "json"))
                    .count()
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn material<'a>(user: &'a str, model: Option<&'a str>) -> KeyMaterial<'a> {
        KeyMaterial {
            task_kind: "modules",
            prompt_version: 1,
            provider: model.map(|_| "anthropic"),
            model,
            max_tokens: 600,
            output_mode: "text",
            output_schema: None,
            system: "system prompt",
            user,
        }
    }

    fn entry(key: &str) -> CacheEntry {
        CacheEntry {
            key: key.to_string(),
            task_id: "module:src".into(),
            task_kind: "modules".into(),
            model: "m".into(),
            usage: None,
            text: "hello".into(),
        }
    }

    #[test]
    fn key_is_pinned() {
        // A change here invalidates every user's cache. Bump KEY_SCHEMA on purpose if so.
        assert_eq!(
            material("ctx", Some("claude-sonnet-5")).key(),
            "bf44ce3c02d91bff03615e401af9e65d5d33cfc9229d206b7f7f341466aa2815"
        );
    }

    #[test]
    fn key_depends_on_request_not_time() {
        let a = material("ctx", Some("m1")).key();
        assert_eq!(a, material("ctx", Some("m1")).key());
        assert_ne!(a, material("ctx2", Some("m1")).key());
        assert_ne!(a, material("ctx", Some("m2")).key());
        assert_ne!(a, material("ctx", None).key());
        let mut v2 = material("ctx", Some("m1"));
        v2.prompt_version = 2;
        assert_ne!(a, v2.key());
    }

    #[test]
    fn put_get_round_trip_and_miss() {
        let dir = TempDir::new().unwrap();
        let cache = WriteCache::new(dir.path());
        let key = material("ctx", Some("m")).key();
        assert!(cache.get(&key).is_none());
        cache.put(&entry(&key)).unwrap();
        assert_eq!(cache.get(&key), Some(entry(&key)));
        assert_eq!(cache.count(), 1);
        assert!(
            cache
                .entry_path(&key)
                .starts_with(dir.path().join("v1").join(&key[..2]))
        );
    }

    #[test]
    fn corrupt_entry_is_a_miss() {
        let dir = TempDir::new().unwrap();
        let cache = WriteCache::new(dir.path());
        let key = material("ctx", Some("m")).key();
        let path = cache.entry_path(&key);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"{not json").unwrap();
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn prune_keeps_last_k_runs() {
        let dir = TempDir::new().unwrap();
        let cache = WriteCache::new(dir.path());
        let keys: Vec<String> = (0..4)
            .map(|i| material(&format!("ctx{i}"), Some("m")).key())
            .collect();
        for k in &keys {
            cache.put(&entry(k)).unwrap();
        }
        // keep_runs = 2. Run A uses {0,1}; run B uses {2}; run C uses {3}.
        // After C, the kept runs are B and C, so 0 and 1 go.
        let (k0, k1, k2, k3) = (&keys[0], &keys[1], &keys[2], &keys[3]);
        assert_eq!(
            cache
                .record_run_and_prune([k0.clone(), k1.clone()], 2)
                .unwrap(),
            2,
            "run A: 2 and 3 are unreferenced"
        );
        cache.put(&entry(k2)).unwrap();
        cache.put(&entry(k3)).unwrap();
        assert_eq!(
            cache.record_run_and_prune([k2.clone()], 2).unwrap(),
            1,
            "run B: 3"
        );
        cache.put(&entry(k3)).unwrap();
        assert_eq!(
            cache.record_run_and_prune([k3.clone()], 2).unwrap(),
            2,
            "run C: 0, 1"
        );
        assert!(cache.get(k0).is_none());
        assert!(cache.get(k1).is_none());
        assert!(cache.get(k2).is_some());
        assert!(cache.get(k3).is_some());
    }
}
