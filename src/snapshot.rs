//! The index snapshot: which store files make up the index, and one reader over them.
//!
//! A snapshot is named by `.reflex/manifest.json`, the single commit point of an
//! index run. The manifest names generation-suffixed files (`content.<g>.bin`,
//! `trigrams.<g>.bin`) that are never modified once written: a new run writes new
//! files, then renames a new manifest into place. A reader that opened one
//! generation keeps reading it; one that opens after the rename sees the next. No
//! reader can pair a `content.bin` from one run with a `trigrams.bin` from another,
//! which the two-rename publish before the manifest allowed.
//!
//! `content.bin` and `trigrams.bin` still exist, as hard links to the current base,
//! for older Reflex binaries (which read only those names) and for tools that look
//! at them. They are removed while a delta holds changes the base lacks, so an
//! older binary stops instead of answering from the base alone.
//!
//! Without a manifest (a cache written before it existed) the fixed names are the
//! snapshot, as before.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::content_store::ContentReader;
use crate::errors::ReflexError;
use crate::trigram::{FileLocation, TrigramIndex};

/// File name of the manifest in the cache directory.
pub const MANIFEST: &str = "manifest.json";
/// The names older binaries read; hard links to the current base.
pub const FIXED_CONTENT: &str = "content.bin";
pub const FIXED_TRIGRAMS: &str = "trigrams.bin";

/// Manifest format this binary writes and reads.
const MANIFEST_FORMAT: u32 = 1;

/// One pair of stores (content + trigrams) named by a manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentFiles {
    /// `content.bin`-format file name, relative to the cache directory.
    pub content: String,
    /// `trigrams.bin`-format file name.
    pub trigrams: String,
    /// Planning sizes of `trigrams` (see `trigram::plan_size`), when written.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<String>,
    /// Files in the segment (both stores hold this many).
    pub files: u64,
    /// Byte sizes of the two files, checked at open.
    pub content_bytes: u64,
    pub trigrams_bytes: u64,
}

/// The commit point of an index run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub format: u32,
    /// Bumped at every publish; every file this manifest names carries it or an
    /// earlier generation's number.
    pub generation: u64,
    pub base: SegmentFiles,
    /// Distinct trigrams with at least one posting in a live file.
    pub live_trigrams: u64,
    /// Bytes of text of the live files (what `content.bin` of a full build of the
    /// same tree holds).
    pub live_corpus_bytes: u64,
    /// blake3 of this manifest serialized with an empty `checksum`.
    #[serde(default)]
    pub checksum: String,
}

impl Manifest {
    /// A manifest for a freshly built base.
    pub fn for_base(
        generation: u64,
        base: SegmentFiles,
        live_trigrams: u64,
        live_corpus_bytes: u64,
    ) -> Self {
        let mut m = Self {
            format: MANIFEST_FORMAT,
            generation,
            base,
            live_trigrams,
            live_corpus_bytes,
            checksum: String::new(),
        };
        m.checksum = m.compute_checksum();
        m
    }

    fn compute_checksum(&self) -> String {
        let mut unsigned = self.clone();
        unsigned.checksum = String::new();
        let body = serde_json::to_vec(&unsigned).expect("manifest serializes");
        blake3::hash(&body).to_hex().to_string()
    }

    /// Every file name this manifest refers to.
    pub fn files(&self) -> Vec<&str> {
        let mut out = vec![self.base.content.as_str(), self.base.trigrams.as_str()];
        out.extend(self.base.plan.as_deref());
        out
    }
}

/// `content.<g>.bin` / `trigrams.<g>.bin` / `trigrams.<g>.plan` of a base built
/// at generation `g`.
pub fn base_file_names(generation: u64) -> (String, String, String) {
    (
        format!("content.{generation}.bin"),
        format!("trigrams.{generation}.bin"),
        format!("trigrams.{generation}.plan"),
    )
}

/// Whether `name` is a generation file this module writes (so cleanup may delete
/// it when no manifest names it).
fn is_generation_file(name: &str) -> bool {
    let parts: Vec<&str> = name.split('.').collect();
    let digits = |g: &str| !g.is_empty() && g.bytes().all(|b| b.is_ascii_digit());
    match parts.as_slice() {
        ["content" | "trigrams", g, "bin"] => digits(g),
        ["trigrams", g, "plan"] => digits(g),
        _ => false,
    }
}

/// Read the manifest; `Ok(None)` when there is none (a cache from before it).
pub fn read_manifest(cache_dir: &Path) -> Result<Option<Manifest>> {
    let path = cache_dir.join(MANIFEST);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("Failed to read {}", path.display())),
    };
    let manifest: Manifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("{} is not a valid manifest", path.display()))?;
    if manifest.format != MANIFEST_FORMAT {
        anyhow::bail!(
            "{} has format {}, this binary reads {}",
            path.display(),
            manifest.format,
            MANIFEST_FORMAT
        );
    }
    if manifest.checksum != manifest.compute_checksum() {
        anyhow::bail!("{} failed its checksum", path.display());
    }
    Ok(Some(manifest))
}

/// Publish `manifest`: tmp + fsync + rename + directory fsync. After this returns,
/// every new reader sees the snapshot it names.
pub fn write_manifest(cache_dir: &Path, manifest: &Manifest) -> Result<()> {
    use std::io::Write;
    let path = cache_dir.join(MANIFEST);
    let tmp = crate::atomic_write::tmp_path_for(&path);
    let body = serde_json::to_vec_pretty(manifest).context("Failed to serialize the manifest")?;
    {
        let mut f = std::fs::File::create(&tmp)
            .with_context(|| format!("Failed to create {}", tmp.display()))?;
        f.write_all(&body)?;
        f.sync_all()?;
    }
    crate::atomic_write::atomic_replace(&tmp, &path)
        .with_context(|| format!("Failed to move {} into place", path.display()))?;
    sync_dir(cache_dir);
    Ok(())
}

/// fsync a directory so a rename in it is durable (best effort; not on Windows).
pub fn sync_dir(dir: &Path) {
    #[cfg(unix)]
    if let Ok(d) = std::fs::File::open(dir) {
        let _ = d.sync_all();
    }
    #[cfg(not(unix))]
    let _ = dir;
}

/// Point `content.bin` / `trigrams.bin` at the manifest's base (hard links), for
/// older binaries. Where hard links are not supported the fixed names are removed
/// instead: an older binary then stops rather than reading a stale base.
pub fn link_fixed_names(cache_dir: &Path, manifest: &Manifest) {
    for (generation_file, fixed) in [
        (&manifest.base.content, FIXED_CONTENT),
        (&manifest.base.trigrams, FIXED_TRIGRAMS),
    ] {
        let target = cache_dir.join(fixed);
        let tmp = crate::atomic_write::tmp_path_for(&target);
        let _ = std::fs::remove_file(&tmp);
        let linked = std::fs::hard_link(cache_dir.join(generation_file), &tmp)
            .and_then(|_| crate::atomic_write::atomic_replace(&tmp, &target));
        if let Err(e) = linked {
            log::debug!(
                "Cannot link {} to {}: {}; removing it",
                fixed,
                generation_file,
                e
            );
            let _ = std::fs::remove_file(&tmp);
            let _ = std::fs::remove_file(&target);
        }
    }
}

/// Delete generation files that neither `current` nor `previous` names. A reader
/// that read the previous manifest a moment ago can still open its files; older
/// ones are gone. Failures (a file still mapped on Windows) are retried next run.
pub fn remove_unreferenced(cache_dir: &Path, current: &Manifest, previous: Option<&Manifest>) {
    let mut keep: std::collections::HashSet<&str> = current.files().into_iter().collect();
    if let Some(prev) = previous {
        keep.extend(prev.files());
    }
    let Ok(entries) = std::fs::read_dir(cache_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if is_generation_file(name)
            && !keep.contains(name)
            && let Err(e) = std::fs::remove_file(entry.path())
        {
            log::debug!("Could not remove {}: {}", name, e);
        }
    }
}

/// One pair of opened stores.
struct Segment {
    content: ContentReader,
    trigrams: TrigramIndex,
}

/// Every reader of the stores goes through this: file ids, content, paths and
/// candidate lookups over the snapshot a manifest (or the legacy fixed names) names.
pub struct IndexSnapshot {
    base: Segment,
    /// The manifest this snapshot was opened from; `None` for a legacy cache.
    manifest: Option<Manifest>,
}

impl std::fmt::Debug for IndexSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexSnapshot")
            .field("files", &self.live_file_count())
            .field("generation", &self.generation())
            .finish()
    }
}

/// Whether an error chain carries an `io::ErrorKind::NotFound`.
fn is_not_found(err: &anyhow::Error) -> bool {
    err.chain().any(|c| {
        c.downcast_ref::<std::io::Error>()
            .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound)
    })
}

impl IndexSnapshot {
    /// Open the snapshot the manifest names (retrying when a concurrent publish
    /// removed a file between reading the manifest and opening it), or the legacy
    /// fixed names when there is no manifest.
    pub fn open(cache_dir: &Path) -> Result<Self> {
        let mut last: Option<anyhow::Error> = None;
        for attempt in 0..5u64 {
            let manifest = read_manifest(cache_dir)
                .map_err(|e| ReflexError::CacheCorrupted(format!("content.bin: {e:#}")))?;
            let result = match manifest {
                None => return Self::open_files(cache_dir, None),
                Some(m) => Self::open_files(cache_dir, Some(m)),
            };
            match result {
                Ok(s) => return Ok(s),
                Err(e) if is_not_found(&e) => {
                    last = Some(e);
                    std::thread::sleep(std::time::Duration::from_millis(10 * (attempt + 1)));
                }
                Err(e) => return Err(e),
            }
        }
        Err(last.expect("at least one attempt"))
    }

    fn open_files(cache_dir: &Path, manifest: Option<Manifest>) -> Result<Self> {
        let (content_name, trigrams_name, plan_name) = match &manifest {
            Some(m) => (
                m.base.content.clone(),
                m.base.trigrams.clone(),
                m.base.plan.clone(),
            ),
            None => (FIXED_CONTENT.to_string(), FIXED_TRIGRAMS.to_string(), None),
        };
        let content_path = cache_dir.join(&content_name);
        let trigrams_path = cache_dir.join(&trigrams_name);

        // A manifest-named file that is missing is a publish race (retried) or a
        // damaged cache; the error keeps the logical name `content.bin`.
        if manifest.is_some() && !content_path.exists() {
            return Err(anyhow::Error::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{} is missing", content_name),
            ))
            .context(ReflexError::CacheCorrupted(format!(
                "content.bin: {} is missing",
                content_name
            ))));
        }
        let content = ContentReader::open(&content_path)
            .map_err(|e| ReflexError::CacheCorrupted(format!("content.bin: {e:#}")))?;

        let mut trigrams = if trigrams_path.exists() {
            match TrigramIndex::load(&trigrams_path) {
                Ok(index) => index,
                // A format from another Reflex version is not corruption (see
                // `ReflexError::CacheVersionMismatch`): serve from an in-memory
                // rebuild, and let the schema-hash check report the index stale
                // so the caller re-indexes.
                Err(e) if e.to_string().contains("Unsupported trigrams.bin version") => {
                    log::warn!("{}; rebuilding trigram index in memory for this process", e);
                    crate::query::result::rebuild_trigram_index(&content)?
                }
                Err(e) => {
                    return Err(ReflexError::CacheCorrupted(format!("trigrams.bin: {e:#}")).into());
                }
            }
        } else if manifest.is_some() {
            return Err(anyhow::Error::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{} is missing", trigrams_name),
            ))
            .context(ReflexError::CacheCorrupted(format!(
                "trigrams.bin: {} is missing",
                trigrams_name
            ))));
        } else {
            log::debug!("trigrams.bin not found, rebuilding from content store");
            crate::query::result::rebuild_trigram_index(&content)?
        };

        // Planning sizes: the early-stop planner reads them instead of on-disk
        // bytes (an index without them plans by bytes, as before).
        if let Some(plan) = &plan_name {
            let plan_path = cache_dir.join(plan);
            if !plan_path.exists() {
                return Err(anyhow::Error::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("{} is missing", plan),
                ))
                .context(ReflexError::CacheCorrupted(format!(
                    "trigrams.bin: {} is missing",
                    plan
                ))));
            }
            trigrams
                .attach_plan(&plan_path)
                .map_err(|e| ReflexError::CacheCorrupted(format!("trigrams.bin: {e:#}")))?;
        }

        if trigrams.file_count() != content.file_count() {
            return Err(ReflexError::CacheCorrupted(format!(
                "trigrams.bin lists {} files but content.bin holds {} (index written by two runs?)",
                trigrams.file_count(),
                content.file_count()
            ))
            .into());
        }
        if let Some(m) = &manifest
            && content.file_count() as u64 != m.base.files
        {
            return Err(ReflexError::CacheCorrupted(format!(
                "content.bin holds {} files but the manifest lists {}",
                content.file_count(),
                m.base.files
            ))
            .into());
        }

        Ok(Self {
            base: Segment { content, trigrams },
            manifest,
        })
    }

    /// The generation of the manifest this snapshot was opened from.
    pub fn generation(&self) -> Option<u64> {
        self.manifest.as_ref().map(|m| m.generation)
    }

    /// The manifest this snapshot was opened from (`None` for a legacy cache).
    pub fn manifest(&self) -> Option<&Manifest> {
        self.manifest.as_ref()
    }

    /// Upper bound of file ids (`0..id_bound()`); some ids below it may be dead.
    pub fn id_bound(&self) -> u32 {
        self.base.content.file_count() as u32
    }

    /// Number of live files (what a fresh build of the same tree holds).
    pub fn live_file_count(&self) -> usize {
        self.base.content.file_count()
    }

    /// Whether `id` names a live file.
    pub fn is_live(&self, id: u32) -> bool {
        id < self.id_bound()
    }

    /// Every live file id, ascending.
    pub fn live_ids(&self) -> impl Iterator<Item = u32> + '_ {
        0..self.id_bound()
    }

    /// Content of a live file.
    pub fn get_file_content(&self, file_id: u32) -> Result<&str> {
        self.base.content.get_file_content(file_id)
    }

    /// Path of a live file (`None` for an id that is out of range or dead).
    pub fn get_file_path(&self, file_id: u32) -> Option<&Path> {
        if !self.is_live(file_id) {
            return None;
        }
        self.base.content.get_file_path(file_id)
    }

    /// Lines around `line_number` of a live file.
    pub fn get_context_by_line(
        &self,
        file_id: u32,
        line_number: usize,
        context_lines: usize,
    ) -> Result<(Vec<String>, Vec<String>)> {
        self.base
            .content
            .get_context_by_line(file_id, line_number, context_lines)
    }

    /// Live id of `path` (a leading `./` is ignored), by linear scan. Callers on a
    /// hot path use `OpenIndex::file_id_for`, which caches the map.
    pub fn get_file_id_by_path(&self, path: &str) -> Option<u32> {
        let normalized = path.strip_prefix("./").unwrap_or(path);
        self.live_ids().find(|&id| {
            self.get_file_path(id)
                .and_then(|p| p.to_str())
                .is_some_and(|p| p.strip_prefix("./").unwrap_or(p) == normalized)
        })
    }

    /// Candidate lines for `pattern` (see [`TrigramIndex::search_candidates`]).
    pub fn search_candidates(&self, pattern: &str) -> Vec<FileLocation> {
        self.base.trigrams.search_candidates(pattern)
    }

    /// Case-insensitive candidate lines (see [`TrigramIndex::search_candidates_fold`]).
    pub fn search_candidates_fold(&self, literal: &[u8]) -> Vec<FileLocation> {
        self.base.trigrams.search_candidates_fold(literal)
    }

    /// Lines with a Kelvin sign or long s (see [`TrigramIndex::exotic_fold_lines`]).
    pub fn exotic_fold_lines(&self) -> Vec<FileLocation> {
        self.base.trigrams.exotic_fold_lines()
    }

    /// Distinct trigrams in the snapshot's live files.
    pub fn trigram_count(&self) -> usize {
        match &self.manifest {
            Some(m) => m.live_trigrams as usize,
            None => self.base.trigrams.trigram_count(),
        }
    }
}

/// What the published snapshot is made of, for sizes and integrity checks. Every
/// file is counted once (the fixed names are links to the base, not extra bytes).
#[derive(Debug, Clone)]
pub struct StoreSummary {
    /// `content.bin`-format files of the snapshot.
    pub content_files: Vec<PathBuf>,
    /// `trigrams.bin`-format files of the snapshot.
    pub trigram_files: Vec<PathBuf>,
    /// Sidecars of the snapshot (planning sizes).
    pub other_files: Vec<PathBuf>,
    /// Text bytes of the live files, when a manifest records it.
    pub live_corpus_bytes: Option<u64>,
    /// Distinct trigrams of the live files, when a manifest records it.
    pub live_trigrams: Option<u64>,
}

impl StoreSummary {
    /// Every store file of the snapshot.
    pub fn files(&self) -> impl Iterator<Item = &PathBuf> {
        self.content_files
            .iter()
            .chain(&self.trigram_files)
            .chain(&self.other_files)
    }
}

/// The snapshot of `cache_dir` as its manifest names it, or the fixed names
/// without one.
pub fn store_summary(cache_dir: &Path) -> StoreSummary {
    match read_manifest(cache_dir) {
        Ok(Some(m)) => StoreSummary {
            content_files: vec![cache_dir.join(&m.base.content)],
            trigram_files: vec![cache_dir.join(&m.base.trigrams)],
            other_files: m.base.plan.iter().map(|p| cache_dir.join(p)).collect(),
            live_corpus_bytes: Some(m.live_corpus_bytes),
            live_trigrams: Some(m.live_trigrams),
        },
        _ => StoreSummary {
            content_files: vec![cache_dir.join(FIXED_CONTENT)],
            trigram_files: vec![cache_dir.join(FIXED_TRIGRAMS)],
            other_files: Vec::new(),
            live_corpus_bytes: None,
            live_trigrams: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_file_names() {
        assert!(is_generation_file("content.12.bin"));
        assert!(is_generation_file("trigrams.3.bin"));
        assert!(is_generation_file("trigrams.3.plan"));
        assert!(!is_generation_file("content.3.plan"));
        assert!(!is_generation_file("content.bin"));
        assert!(!is_generation_file("trigrams.bin"));
        assert!(!is_generation_file("content.x.bin"));
        assert!(!is_generation_file("meta.db"));
    }

    #[test]
    fn manifest_round_trip_and_checksum() {
        let temp = tempfile::TempDir::new().unwrap();
        let (content, trigrams, plan) = base_file_names(7);
        let m = Manifest::for_base(
            7,
            SegmentFiles {
                content,
                trigrams,
                plan: Some(plan),
                files: 3,
                content_bytes: 100,
                trigrams_bytes: 200,
            },
            42,
            90,
        );
        write_manifest(temp.path(), &m).unwrap();
        assert_eq!(read_manifest(temp.path()).unwrap(), Some(m));

        // A tampered manifest fails its checksum.
        let path = temp.path().join(MANIFEST);
        let text = std::fs::read_to_string(&path)
            .unwrap()
            .replace("\"generation\": 7", "\"generation\": 8");
        std::fs::write(&path, text).unwrap();
        assert!(read_manifest(temp.path()).is_err());
    }
}
