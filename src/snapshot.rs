//! The index snapshot: which store files make up the index, and one reader over them.
//!
//! A snapshot is named by `.reflex/manifest.json`, the single commit point of an
//! index run. It is:
//!
//! - a **base**: `content.<g>.bin` + `trigrams.<g>.bin` (+ planning sizes) from the
//!   last full build, file ids `0..N-1`;
//! - optionally a **delta**: the same pair for files added or modified since, with
//!   their own ids `0..k-1`, seen through the snapshot as `N..N+k-1`;
//! - **tombstones**: base ids that are deleted or superseded by the delta.
//!
//! Every file a manifest names is generation-suffixed and never modified once
//! written: a new run writes new files, then renames a new manifest into place. A
//! reader that opened one generation keeps reading it; one that opens after the
//! rename sees the next. No reader can pair files from two runs, which the
//! two-rename publish before the manifest allowed.
//!
//! `content.bin` and `trigrams.bin` exist as hard links to the base only while the
//! base alone is the snapshot (no delta, no tombstones), for older Reflex binaries
//! (which read only those names) and for tools. While a delta holds changes they
//! are absent, so an older binary stops instead of answering from the base alone.
//!
//! Without a manifest (a cache written before it existed) the fixed names are the
//! snapshot, as before.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::content_store::ContentReader;
use crate::errors::ReflexError;
use crate::trigram::{
    EXOTIC_LONG_S_RANGE, FileLocation, ListPart, Trigram, TrigramIndex, TrigramList,
    exotic_singles, fold_windows, pattern_trigrams, plan_fold_intersection, plan_intersection,
    union_lists,
};

/// File name of the manifest in the cache directory.
pub const MANIFEST: &str = "manifest.json";
/// The names older binaries read; hard links to the base while it is the snapshot.
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

impl SegmentFiles {
    fn names(&self) -> Vec<&str> {
        let mut out = vec![self.content.as_str(), self.trigrams.as_str()];
        out.extend(self.plan.as_deref());
        out
    }
}

/// The commit point of an index run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub format: u32,
    /// Bumped at every publish; every file this manifest names carries it or an
    /// earlier generation's number.
    pub generation: u64,
    pub base: SegmentFiles,
    /// Files added or modified since the base was built.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<SegmentFiles>,
    /// Base ids that are deleted or superseded by the delta, ascending.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tombstones: Vec<u32>,
    /// `(trigram, planning size)` of the tombstoned base files' postings: what the
    /// base's planning sizes over-count (see [`TombPlan`]). Present with tombstones.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tomb: Option<String>,
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
        Self::new(
            generation,
            base,
            None,
            Vec::new(),
            None,
            live_trigrams,
            live_corpus_bytes,
        )
    }

    /// A manifest for a base with a delta and tombstones (`tombstones` sorted;
    /// `tomb` names their planning sizes).
    pub fn new(
        generation: u64,
        base: SegmentFiles,
        delta: Option<SegmentFiles>,
        tombstones: Vec<u32>,
        tomb: Option<String>,
        live_trigrams: u64,
        live_corpus_bytes: u64,
    ) -> Self {
        let mut m = Self {
            format: MANIFEST_FORMAT,
            generation,
            base,
            delta,
            tombstones,
            tomb,
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
        let mut out = self.base.names();
        if let Some(delta) = &self.delta {
            out.extend(delta.names());
        }
        out.extend(self.tomb.as_deref());
        out
    }

    /// Whether the base alone is the snapshot (no delta, nothing tombstoned).
    pub fn base_only(&self) -> bool {
        self.delta.is_none() && self.tombstones.is_empty()
    }

    /// Files of the snapshot: what a full build of the same tree holds.
    pub fn live_files(&self) -> u64 {
        self.base.files - self.tombstones.len() as u64 + self.delta.as_ref().map_or(0, |d| d.files)
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

/// The files of a delta built at generation `g`: content, trigrams, planning
/// sizes, tombstoned planning sizes.
pub fn delta_file_names(generation: u64) -> (String, String, String, String) {
    (
        format!("delta.{generation}.content.bin"),
        format!("delta.{generation}.trigrams.bin"),
        format!("delta.{generation}.plan"),
        format!("delta.{generation}.tomb"),
    )
}

/// Whether `name` is a generation file this module writes (so cleanup may delete
/// it when no manifest names it).
pub fn is_generation_file(name: &str) -> bool {
    let parts: Vec<&str> = name.split('.').collect();
    let digits = |g: &str| !g.is_empty() && g.bytes().all(|b| b.is_ascii_digit());
    match parts.as_slice() {
        ["content" | "trigrams", g, "bin"] => digits(g),
        ["trigrams", g, "plan"] => digits(g),
        ["delta", g, "content" | "trigrams", "bin"] => digits(g),
        ["delta", g, "plan" | "tomb"] => digits(g),
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
/// older binaries, when the base alone is the snapshot. Otherwise, or where hard
/// links are not supported, the fixed names are removed: an older binary then
/// stops rather than answering from a base that lacks the delta.
pub fn link_fixed_names(cache_dir: &Path, manifest: &Manifest) {
    if !manifest.base_only() {
        unlink_fixed_names(cache_dir);
        return;
    }
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

/// Remove `content.bin` / `trigrams.bin` (before a manifest with a delta is
/// published, so no older binary reads the base alone once it is incomplete).
pub fn unlink_fixed_names(cache_dir: &Path) {
    for fixed in [FIXED_CONTENT, FIXED_TRIGRAMS] {
        match std::fs::remove_file(cache_dir.join(fixed)) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => log::warn!("Could not remove {}: {}", fixed, e),
        }
    }
    sync_dir(cache_dir);
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

const TOMB_MAGIC: &[u8; 4] = b"RFTB";
const TOMB_VERSION: u32 = 1;
const TOMB_HEADER: usize = 16;

/// `delta.<g>.tomb`: for each trigram with postings in tombstoned base files, the
/// planning size of those postings. The base's planning size minus this, plus the
/// delta's, is the planning size a full build of the same tree has. Header: magic
/// `RFTB`, version u32, entry count u64; then `(trigram u32, size u32)` sorted.
pub struct TombPlan {
    mmap: Option<memmap2::Mmap>,
    count: usize,
}

impl TombPlan {
    fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open {}", path.display()))?;
        let len = file.metadata()?.len() as usize;
        if len == TOMB_HEADER {
            // An empty map cannot be mapped on every platform.
            return Ok(Self {
                mmap: None,
                count: 0,
            });
        }
        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .with_context(|| format!("Failed to mmap {}", path.display()))?
        };
        if mmap.len() < TOMB_HEADER || &mmap[0..4] != TOMB_MAGIC {
            anyhow::bail!("{} is not a tombstone planning file", path.display());
        }
        if u32::from_le_bytes(mmap[4..8].try_into().unwrap()) != TOMB_VERSION {
            anyhow::bail!("{} has an unsupported version", path.display());
        }
        let count = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        if mmap.len() != TOMB_HEADER + count * 8 {
            anyhow::bail!("{} is truncated", path.display());
        }
        Ok(Self {
            mmap: Some(mmap),
            count,
        })
    }

    #[inline]
    fn entry(&self, i: usize) -> (Trigram, u32) {
        let mmap = self.mmap.as_ref().expect("non-empty");
        let off = TOMB_HEADER + i * 8;
        (
            u32::from_le_bytes(mmap[off..off + 4].try_into().unwrap()),
            u32::from_le_bytes(mmap[off + 4..off + 8].try_into().unwrap()),
        )
    }

    /// The tombstoned planning size of `trigram` (0 when none).
    pub fn get(&self, trigram: Trigram) -> u32 {
        let (mut lo, mut hi) = (0usize, self.count);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let (t, size) = self.entry(mid);
            match t.cmp(&trigram) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return size,
            }
        }
        0
    }

    /// Every `(trigram, size)`, ascending.
    pub fn entries(&self) -> impl Iterator<Item = (Trigram, u32)> + '_ {
        (0..self.count).map(|i| self.entry(i))
    }
}

/// Write a tombstone planning file (`entries` sorted by trigram; tmp + fsync + rename).
pub fn write_tomb_file(path: &Path, entries: &[(Trigram, u32)]) -> Result<()> {
    use std::io::Write;
    let tmp = crate::atomic_write::tmp_path_for(path);
    {
        let mut w = std::io::BufWriter::new(
            std::fs::File::create(&tmp)
                .with_context(|| format!("Failed to create {}", tmp.display()))?,
        );
        w.write_all(TOMB_MAGIC)?;
        w.write_all(&TOMB_VERSION.to_le_bytes())?;
        w.write_all(&(entries.len() as u64).to_le_bytes())?;
        for (t, size) in entries {
            w.write_all(&t.to_le_bytes())?;
            w.write_all(&size.to_le_bytes())?;
        }
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    crate::atomic_write::atomic_replace(&tmp, path)
        .with_context(|| format!("Failed to move {} into place", path.display()))
}

/// One pair of opened stores.
pub struct Segment {
    pub content: ContentReader,
    pub trigrams: TrigramIndex,
}

/// Every reader of the stores goes through this: file ids, content, paths and
/// candidate lookups over the snapshot a manifest (or the legacy fixed names) names.
pub struct IndexSnapshot {
    base: Segment,
    delta: Option<Segment>,
    /// Tombstoned base ids, one bit each.
    dead: Vec<u64>,
    dead_count: usize,
    tomb: Option<TombPlan>,
    /// Files in the base: the delta's ids start here.
    base_len: u32,
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

/// A missing manifest-named file: a publish race (retried) or a damaged cache.
/// `logical` is the name an error shows (`content.bin` / `trigrams.bin`).
fn missing(logical: &str, name: &str) -> anyhow::Error {
    anyhow::Error::new(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("{} is missing", name),
    ))
    .context(ReflexError::CacheCorrupted(format!(
        "{}: {} is missing",
        logical, name
    )))
}

/// Check a manifest-named file's size (a truncated or replaced file is corruption).
fn check_size(cache_dir: &Path, logical: &str, name: &str, expected: u64) -> Result<()> {
    match std::fs::metadata(cache_dir.join(name)) {
        Ok(md) if md.len() == expected => Ok(()),
        Ok(md) => Err(ReflexError::CacheCorrupted(format!(
            "{}: {} is {} bytes, the manifest says {}",
            logical,
            name,
            md.len(),
            expected
        ))
        .into()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(missing(logical, name)),
        Err(e) => Err(e).with_context(|| format!("Failed to stat {}", name)),
    }
}

/// Open one manifest-named segment.
fn open_segment(cache_dir: &Path, files: &SegmentFiles) -> Result<Segment> {
    check_size(
        cache_dir,
        "content.bin",
        &files.content,
        files.content_bytes,
    )?;
    check_size(
        cache_dir,
        "trigrams.bin",
        &files.trigrams,
        files.trigrams_bytes,
    )?;
    let content = ContentReader::open(cache_dir.join(&files.content))
        .map_err(|e| ReflexError::CacheCorrupted(format!("content.bin: {e:#}")))?;
    let mut trigrams = TrigramIndex::load(cache_dir.join(&files.trigrams))
        .map_err(|e| ReflexError::CacheCorrupted(format!("trigrams.bin: {e:#}")))?;
    // Planning sizes: the early-stop planner reads them instead of on-disk
    // bytes (an index without them plans by bytes, as before).
    if let Some(plan) = &files.plan {
        let plan_path = cache_dir.join(plan);
        if !plan_path.exists() {
            return Err(missing("trigrams.bin", plan));
        }
        trigrams
            .attach_plan(&plan_path)
            .map_err(|e| ReflexError::CacheCorrupted(format!("trigrams.bin: {e:#}")))?;
    }
    check_counts(&content, &trigrams)?;
    if content.file_count() as u64 != files.files {
        return Err(ReflexError::CacheCorrupted(format!(
            "content.bin holds {} files but the manifest lists {}",
            content.file_count(),
            files.files
        ))
        .into());
    }
    Ok(Segment { content, trigrams })
}

fn check_counts(content: &ContentReader, trigrams: &TrigramIndex) -> Result<()> {
    if trigrams.file_count() != content.file_count() {
        return Err(ReflexError::CacheCorrupted(format!(
            "trigrams.bin lists {} files but content.bin holds {} (index written by two runs?)",
            trigrams.file_count(),
            content.file_count()
        ))
        .into());
    }
    Ok(())
}

impl IndexSnapshot {
    /// Open the snapshot the manifest names (retrying when a concurrent publish
    /// removed a file between reading the manifest and opening it), or the legacy
    /// fixed names when there is no manifest.
    pub fn open(cache_dir: &Path) -> Result<Self> {
        Self::open_reading(cache_dir, || read_manifest(cache_dir))
    }

    /// [`Self::open`] with the manifest read by `read` (each attempt reads it again).
    fn open_reading(
        cache_dir: &Path,
        mut read: impl FnMut() -> Result<Option<Manifest>>,
    ) -> Result<Self> {
        let mut last: Option<anyhow::Error> = None;
        for attempt in 0..5u64 {
            let manifest =
                read().map_err(|e| ReflexError::CacheCorrupted(format!("content.bin: {e:#}")))?;
            let result = match manifest {
                None => return Self::open_legacy(cache_dir),
                Some(m) => Self::open_manifest(cache_dir, m),
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

    fn open_manifest(cache_dir: &Path, manifest: Manifest) -> Result<Self> {
        let base = open_segment(cache_dir, &manifest.base)?;
        let base_len = base.content.file_count() as u32;
        let delta = match &manifest.delta {
            Some(d) => Some(open_segment(cache_dir, d)?),
            None => None,
        };
        let tomb = match &manifest.tomb {
            Some(name) => {
                let tomb_path = cache_dir.join(name);
                if !tomb_path.exists() {
                    return Err(missing("trigrams.bin", name));
                }
                Some(
                    TombPlan::load(&tomb_path)
                        .map_err(|e| ReflexError::CacheCorrupted(format!("trigrams.bin: {e:#}")))?,
                )
            }
            None => None,
        };
        let mut dead = vec![0u64; (base_len as usize).div_ceil(64)];
        for &id in &manifest.tombstones {
            if id >= base_len {
                return Err(ReflexError::CacheCorrupted(format!(
                    "content.bin: tombstone {} is past the base's {} files",
                    id, base_len
                ))
                .into());
            }
            dead[(id / 64) as usize] |= 1u64 << (id % 64);
        }
        Ok(Self {
            base,
            delta,
            dead,
            dead_count: manifest.tombstones.len(),
            tomb,
            base_len,
            manifest: Some(manifest),
        })
    }

    fn open_legacy(cache_dir: &Path) -> Result<Self> {
        let content = ContentReader::open(cache_dir.join(FIXED_CONTENT))
            .map_err(|e| ReflexError::CacheCorrupted(format!("content.bin: {e:#}")))?;
        let trigrams_path = cache_dir.join(FIXED_TRIGRAMS);
        let trigrams = if trigrams_path.exists() {
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
        } else {
            log::debug!("trigrams.bin not found, rebuilding from content store");
            crate::query::result::rebuild_trigram_index(&content)?
        };
        check_counts(&content, &trigrams)?;
        let base_len = content.file_count() as u32;
        Ok(Self {
            base: Segment { content, trigrams },
            delta: None,
            dead: Vec::new(),
            dead_count: 0,
            tomb: None,
            base_len,
            manifest: None,
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

    /// The base segment.
    pub fn base(&self) -> &Segment {
        &self.base
    }

    /// The delta segment, if any.
    pub fn delta(&self) -> Option<&Segment> {
        self.delta.as_ref()
    }

    /// Files in the base (the delta's first id).
    pub fn base_len(&self) -> u32 {
        self.base_len
    }

    /// The tombstoned planning sizes (empty without a delta).
    pub fn tomb(&self) -> Option<&TombPlan> {
        self.tomb.as_ref()
    }

    /// Whether base id `id` is tombstoned.
    #[inline]
    pub fn is_dead(&self, id: u32) -> bool {
        id < self.base_len
            && self
                .dead
                .get((id / 64) as usize)
                .is_some_and(|w| w & (1u64 << (id % 64)) != 0)
    }

    fn delta_len(&self) -> u32 {
        self.delta
            .as_ref()
            .map_or(0, |d| d.content.file_count() as u32)
    }

    /// Upper bound of file ids (`0..id_bound()`); some ids below it may be dead.
    pub fn id_bound(&self) -> u32 {
        self.base_len + self.delta_len()
    }

    /// Number of live files (what a fresh build of the same tree holds).
    pub fn live_file_count(&self) -> usize {
        self.base_len as usize - self.dead_count + self.delta_len() as usize
    }

    /// Whether `id` names a live file.
    pub fn is_live(&self, id: u32) -> bool {
        id < self.id_bound() && !self.is_dead(id)
    }

    /// Every live file id, ascending.
    pub fn live_ids(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.base_len)
            .filter(|&id| !self.is_dead(id))
            .chain(self.base_len..self.id_bound())
    }

    /// The segment holding `id`, and the id within it.
    #[inline]
    fn route(&self, id: u32) -> (&Segment, u32) {
        match &self.delta {
            Some(delta) if id >= self.base_len => (delta, id - self.base_len),
            _ => (&self.base, id),
        }
    }

    /// Content of a live file.
    pub fn get_file_content(&self, file_id: u32) -> Result<&str> {
        let (segment, local) = self.route(file_id);
        segment.content.get_file_content(local)
    }

    /// Path of a live file (`None` for an id that is out of range or dead).
    pub fn get_file_path(&self, file_id: u32) -> Option<&Path> {
        if !self.is_live(file_id) {
            return None;
        }
        let (segment, local) = self.route(file_id);
        segment.content.get_file_path(local)
    }

    /// Lines around `line_number` of a live file.
    pub fn get_context_by_line(
        &self,
        file_id: u32,
        line_number: usize,
        context_lines: usize,
    ) -> Result<(Vec<String>, Vec<String>)> {
        let (segment, local) = self.route(file_id);
        segment
            .content
            .get_context_by_line(local, line_number, context_lines)
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

    /// Whether candidate lookups can go straight to the base index.
    fn base_alone(&self) -> bool {
        self.delta.is_none() && self.dead_count == 0
    }

    /// `trigram`'s posting list over the base (tombstones left out) and the delta,
    /// with the planning size a full build of the same tree has.
    fn list(&self, trigram: Trigram) -> Option<TrigramList<'_>> {
        let mut parts: Vec<ListPart<'_>> = Vec::with_capacity(2);
        let mut plan: i64 = 0;
        if let Some((mut part, base_plan)) = self.base.trigrams.list_part(trigram) {
            if self.dead_count > 0 {
                part.dead = Some(&self.dead);
            }
            parts.push(part);
            plan += base_plan as i64;
        }
        if let Some(tomb) = &self.tomb {
            plan -= tomb.get(trigram) as i64;
        }
        if let Some(delta) = &self.delta
            && let Some((mut part, delta_plan)) = delta.trigrams.list_part(trigram)
        {
            part.id_offset = self.base_len;
            parts.push(part);
            plan += delta_plan as i64;
        }
        // Every live file block adds at least 2 to the planning size, so 0 means
        // every file holding the trigram is tombstoned: a fresh build has no list.
        if parts.is_empty() || plan <= 0 {
            return None;
        }
        Some(TrigramList {
            trigram,
            plan: plan as u64,
            parts,
        })
    }

    /// Candidate lines for `pattern` (see [`TrigramIndex::search_candidates`]).
    pub fn search_candidates(&self, pattern: &str) -> Vec<FileLocation> {
        if self.base_alone() {
            return self.base.trigrams.search_candidates(pattern);
        }
        let lists = pattern_trigrams(pattern)
            .into_iter()
            .map(|t| self.list(t))
            .collect::<Vec<_>>();
        if lists.is_empty() {
            return vec![];
        }
        plan_intersection(lists, true)
    }

    /// Case-insensitive candidate lines (see [`TrigramIndex::search_candidates_fold`]).
    pub fn search_candidates_fold(&self, literal: &[u8]) -> Vec<FileLocation> {
        if self.base_alone() {
            return self.base.trigrams.search_candidates_fold(literal);
        }
        let Some(windows) = fold_windows(literal) else {
            return vec![];
        };
        let lists = windows
            .iter()
            .map(|variants| variants.iter().filter_map(|&t| self.list(t)).collect())
            .collect();
        plan_fold_intersection(lists)
    }

    /// Lines with a Kelvin sign or long s (see [`TrigramIndex::exotic_fold_lines`]).
    pub fn exotic_fold_lines(&self) -> Vec<FileLocation> {
        if self.base_alone() {
            return self.base.trigrams.exotic_fold_lines();
        }
        let (lo, hi) = EXOTIC_LONG_S_RANGE;
        let mut trigrams: Vec<Trigram> = self
            .base
            .trigrams
            .list_parts_in(lo, hi)
            .into_iter()
            .map(|(t, _, _)| t)
            .collect();
        if let Some(delta) = &self.delta {
            trigrams.extend(
                delta
                    .trigrams
                    .list_parts_in(lo, hi)
                    .into_iter()
                    .map(|(t, _, _)| t),
            );
        }
        trigrams.extend(exotic_singles());
        trigrams.sort_unstable();
        trigrams.dedup();
        let lists: Vec<TrigramList> = trigrams.into_iter().filter_map(|t| self.list(t)).collect();
        union_lists(&lists)
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
        Ok(Some(m)) => {
            let mut content_files = vec![cache_dir.join(&m.base.content)];
            let mut trigram_files = vec![cache_dir.join(&m.base.trigrams)];
            let mut other_files: Vec<PathBuf> =
                m.base.plan.iter().map(|p| cache_dir.join(p)).collect();
            if let Some(d) = &m.delta {
                content_files.push(cache_dir.join(&d.content));
                trigram_files.push(cache_dir.join(&d.trigrams));
                other_files.extend(d.plan.iter().map(|p| cache_dir.join(p)));
            }
            other_files.extend(m.tomb.iter().map(|t| cache_dir.join(t)));
            StoreSummary {
                content_files,
                trigram_files,
                other_files,
                live_corpus_bytes: Some(m.live_corpus_bytes),
                live_trigrams: Some(m.live_trigrams),
            }
        }
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
        assert!(is_generation_file("delta.4.content.bin"));
        assert!(is_generation_file("delta.4.trigrams.bin"));
        assert!(is_generation_file("delta.4.plan"));
        assert!(is_generation_file("delta.4.tomb"));
        assert!(!is_generation_file("content.3.plan"));
        assert!(!is_generation_file("content.bin"));
        assert!(!is_generation_file("trigrams.bin"));
        assert!(!is_generation_file("content.x.bin"));
        assert!(!is_generation_file("meta.db"));
        assert!(!is_generation_file("manifest.json"));
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

    #[test]
    fn tomb_file_round_trip() {
        let temp = tempfile::TempDir::new().unwrap();
        let path = temp.path().join("t.tomb");
        write_tomb_file(&path, &[(3, 10), (70_000, 4), (0xE2_84_AA, 7)]).unwrap();
        let tomb = TombPlan::load(&path).unwrap();
        assert_eq!(tomb.get(3), 10);
        assert_eq!(tomb.get(70_000), 4);
        assert_eq!(tomb.get(0xE2_84_AA), 7);
        assert_eq!(tomb.get(4), 0);
        write_tomb_file(&path, &[]).unwrap();
        assert_eq!(TombPlan::load(&path).unwrap().get(3), 0);
    }

    /// A reader that meets a manifest naming a file that is gone (a publish in
    /// between) retries and opens the snapshot the next manifest names; with no
    /// next manifest it reports the cache corrupted.
    #[test]
    fn open_retries_while_a_publish_replaces_the_manifest() {
        let temp = tempfile::TempDir::new().unwrap();
        let root = temp.path();
        write(root, "a.rs", "fn alpha() {}\n");
        index_unlimited(root);
        let cache = root.join(".reflex");
        let good = read_manifest(&cache).unwrap().unwrap();
        let mut files = good.base.clone();
        files.content = "content.999.bin".to_string();
        let broken = Manifest::for_base(
            good.generation + 1,
            files,
            good.live_trigrams,
            good.live_corpus_bytes,
        );

        // The first read sees the manifest a publish is about to replace.
        let mut reads = 0;
        let snapshot = IndexSnapshot::open_reading(&cache, || {
            reads += 1;
            Ok(Some(if reads == 1 {
                broken.clone()
            } else {
                good.clone()
            }))
        })
        .expect("a retry opens the next manifest");
        assert_eq!(reads, 2);
        assert_eq!(snapshot.generation(), Some(good.generation));

        write_manifest(&cache, &broken).unwrap();
        let err = IndexSnapshot::open(&cache).unwrap_err();
        assert!(
            matches!(
                err.downcast_ref::<ReflexError>(),
                Some(ReflexError::CacheCorrupted(_))
            ),
            "{err:#}"
        );
    }

    fn index_unlimited(root: &Path) {
        let mut indexer = crate::Indexer::new(
            crate::CacheManager::new(root),
            crate::models::IndexConfig::default(),
        );
        indexer.set_merge_limits(usize::MAX, u64::MAX);
        indexer.index(root, false).unwrap();
    }

    fn write(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }

    /// Candidate lines of `snapshot` for `pattern`, by path.
    fn candidates(snapshot: &IndexSnapshot, pattern: &str) -> Vec<(PathBuf, u32)> {
        let mut out: Vec<(PathBuf, u32)> = snapshot
            .search_candidates(pattern)
            .into_iter()
            .map(|l| {
                let path = snapshot.get_file_path(l.file_id).expect("a live candidate");
                (path.to_path_buf(), l.line_no)
            })
            .collect();
        out.sort();
        out
    }

    /// Base + delta − tombstones: every trigram has the planning size and the
    /// candidate lines a fresh build of the same tree has, and a trigram whose files
    /// are all tombstoned has no list at all.
    #[test]
    fn a_delta_snapshot_plans_and_answers_as_a_fresh_build() {
        let temp = tempfile::TempDir::new().unwrap();
        let root = temp.path();
        // 200 files, so file-id gaps pass 128 (where planning size and on-disk
        // size part ways).
        for i in 0..200 {
            let rare = if i % 50 == 0 { "rare_marker_qx\n" } else { "" };
            write(
                root,
                &format!("src/f{i:03}.rs"),
                &format!("pub fn common_{i}() {{ shared(); }}\n{rare}// line {i}\n"),
            );
        }
        write(root, "only/gone.rs", "fn vanishing_zqj() {}\n");
        index_unlimited(root);

        write(root, "src/f000.rs", "pub fn common_0() { edited(); }\n");
        write(
            root,
            "src/f150.rs",
            "pub fn common_150() {}\nrare_marker_qx\n",
        );
        write(
            root,
            "src/new_file.rs",
            "fn brand_new_token() { shared(); }\n",
        );
        std::fs::remove_file(root.join("src/f100.rs")).unwrap();
        std::fs::remove_file(root.join("only/gone.rs")).unwrap();
        index_unlimited(root);

        let cache = root.join(".reflex");
        let updated = IndexSnapshot::open(&cache).unwrap();
        assert!(updated.delta().is_some() && !updated.base_alone());
        let aside = root.join(".reflex-updated");
        std::fs::rename(&cache, &aside).unwrap();
        index_unlimited(root);
        let fresh = IndexSnapshot::open(&cache).unwrap();
        assert!(fresh.base_alone());

        let mut all: Vec<Trigram> = updated.base().trigrams.trigrams().collect();
        all.extend(updated.delta().unwrap().trigrams.trigrams());
        all.extend(fresh.base().trigrams.trigrams());
        all.sort_unstable();
        all.dedup();
        let mut live = 0;
        for t in all {
            let u = updated.list(t).map(|l| l.plan);
            let f = fresh.list(t).map(|l| l.plan);
            assert_eq!(u, f, "planning size of trigram {t:#08x}");
            live += u.is_some() as usize;
        }
        assert_eq!(live, fresh.trigram_count());
        assert_eq!(updated.trigram_count(), fresh.trigram_count());
        assert_eq!(updated.live_file_count(), fresh.live_file_count());

        for pattern in [
            "shared",
            "common_1",
            "rare_marker_qx",
            "edited",
            "brand_new_token",
            "vanishing_zqj",
            "line 1",
        ] {
            assert_eq!(
                candidates(&updated, pattern),
                candidates(&fresh, pattern),
                "{pattern}"
            );
        }
        assert!(candidates(&updated, "vanishing_zqj").is_empty());
        assert_eq!(
            updated.search_candidates_fold(b"RARE_MARKER").len(),
            fresh.search_candidates_fold(b"RARE_MARKER").len()
        );
        drop(updated);
        std::fs::remove_dir_all(&aside).unwrap();
    }
}
