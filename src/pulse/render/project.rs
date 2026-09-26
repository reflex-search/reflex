//! Write files into a staged project only when their bytes change.
//!
//! Unchanged files keep their mtime, which keeps Astro's and Vite's caches warm.
//! `.pulse-files.json` records what the last run wrote, so files of pages that no
//! longer exist are removed (only under managed prefixes; the template is never swept).

use crate::atomic_write::{atomic_replace, tmp_path_for};
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const MANIFEST: &str = ".pulse-files.json";

#[derive(Debug, Default, Clone, Copy)]
pub struct WriteStats {
    pub written: usize,
    pub unchanged: usize,
    pub removed: usize,
}

pub struct ProjectWriter {
    root: PathBuf,
    prev: BTreeMap<String, String>,
    now: BTreeMap<String, String>,
    stats: WriteStats,
}

impl ProjectWriter {
    pub fn open(root: &Path) -> Result<Self> {
        std::fs::create_dir_all(root).with_context(|| format!("creating {}", root.display()))?;
        let prev = std::fs::read(root.join(MANIFEST))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        Ok(Self {
            root: root.to_path_buf(),
            prev,
            now: BTreeMap::new(),
            stats: WriteStats::default(),
        })
    }

    /// Write `rel` unless it already holds exactly `bytes`.
    pub fn put(&mut self, rel: &str, bytes: &[u8]) -> Result<()> {
        let hash = blake3::hash(bytes).to_hex().to_string();
        let path = self.root.join(rel);
        let same = self.prev.get(rel) == Some(&hash) && path.exists();
        if same {
            self.stats.unchanged += 1;
        } else {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let tmp = tmp_path_for(&path);
            std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
            atomic_replace(&tmp, &path)?;
            self.stats.written += 1;
        }
        self.now.insert(rel.to_string(), hash);
        Ok(())
    }

    /// Remove files written last run under `sweep_prefix` but not this run; save the
    /// manifest.
    pub fn finish(mut self, sweep_prefix: &str) -> Result<WriteStats> {
        for rel in self.prev.keys() {
            if rel.starts_with(sweep_prefix)
                && !self.now.contains_key(rel)
                && std::fs::remove_file(self.root.join(rel)).is_ok()
            {
                self.stats.removed += 1;
            }
        }
        // Keep entries this run did not touch outside the sweep prefix (template files
        // are written by another pass with its own writer).
        for (rel, hash) in &self.prev {
            if !rel.starts_with(sweep_prefix) && !self.now.contains_key(rel) {
                self.now.insert(rel.clone(), hash.clone());
            }
        }
        let path = self.root.join(MANIFEST);
        let tmp = tmp_path_for(&path);
        std::fs::write(&tmp, serde_json::to_vec_pretty(&self.now)?)?;
        atomic_replace(&tmp, &path)?;
        Ok(self.stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_run_writes_nothing_and_sweeps_removed_pages() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut w = ProjectWriter::open(dir.path()).unwrap();
        w.put("bundle/pages/a.json", b"A").unwrap();
        w.put("bundle/pages/b.json", b"B").unwrap();
        w.put("pulse.config.json", b"{}").unwrap();
        let s = w.finish("bundle/").unwrap();
        assert_eq!((s.written, s.unchanged, s.removed), (3, 0, 0));

        let mut w = ProjectWriter::open(dir.path()).unwrap();
        w.put("bundle/pages/a.json", b"A").unwrap();
        w.put("pulse.config.json", b"{}").unwrap();
        let s = w.finish("bundle/").unwrap();
        assert_eq!((s.written, s.unchanged, s.removed), (0, 2, 1));
        assert!(!dir.path().join("bundle/pages/b.json").exists());

        let mut w = ProjectWriter::open(dir.path()).unwrap();
        w.put("bundle/pages/a.json", b"A2").unwrap();
        let s = w.finish("bundle/").unwrap();
        assert_eq!((s.written, s.unchanged, s.removed), (1, 0, 0));
        assert!(
            dir.path().join("pulse.config.json").exists(),
            "outside the sweep prefix"
        );
    }
}
