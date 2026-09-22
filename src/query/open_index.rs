//! A shared, reusable handle on the open index files.
//!
//! Before this module, every query re-opened `content.bin` and `trigrams.bin`
//! (three to four times per query, once per phase), parsed the whole trigram
//! directory on each open, and ran `PRAGMA quick_check` over `meta.db`. On a
//! 29 MB corpus that put a ~50 ms floor under every call, including a zero-hit
//! search through the resident `rfx mcp` server.
//!
//! [`OpenIndex`] holds both memory maps, a path→file-id map, and a rayon pool
//! sized from `[performance] parallel_threads`. Handles live in a process-wide
//! registry keyed by the canonical cache directory, so the MCP server, the HTTP
//! server, and the CLI all share one open per index. A handle is reused while
//! the on-disk files carry the same fingerprint (device, inode, size, mtime);
//! an indexer run invalidates it explicitly, and an `rfx index` from another
//! process is caught by the fingerprint compare on the next lookup.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result};

use crate::cache::CacheManager;
use crate::content_store::ContentReader;
use crate::errors::ReflexError;
use crate::trigram::TrigramIndex;

/// Identity of one on-disk file, cheap to read (one `stat`).
#[derive(Debug, Clone, PartialEq, Eq)]
struct FileStamp {
    dev: u64,
    ino: u64,
    size: u64,
    mtime: Option<std::time::SystemTime>,
}

impl FileStamp {
    fn of(path: &Path) -> std::io::Result<Self> {
        let md = std::fs::metadata(path)?;
        #[cfg(unix)]
        let (dev, ino) = {
            use std::os::unix::fs::MetadataExt;
            (md.dev(), md.ino())
        };
        #[cfg(not(unix))]
        let (dev, ino) = (0u64, 0u64);
        Ok(Self {
            dev,
            ino,
            size: md.len(),
            mtime: md.modified().ok(),
        })
    }
}

/// Fingerprint of the two binary stores a handle was opened from.
///
/// `trigrams` is `None` when `trigrams.bin` is absent and the index was rebuilt
/// in memory from `content.bin`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Fingerprint {
    content: FileStamp,
    trigrams: Option<FileStamp>,
}

impl Fingerprint {
    fn current(cache_dir: &Path) -> std::io::Result<Self> {
        let content = FileStamp::of(&cache_dir.join("content.bin"))?;
        let trigrams = FileStamp::of(&cache_dir.join("trigrams.bin")).ok();
        Ok(Self { content, trigrams })
    }
}

/// Everything a query needs from the index, opened once.
pub struct OpenIndex {
    cache_dir: PathBuf,
    /// Memory-mapped `content.bin`.
    pub content: ContentReader,
    /// Memory-mapped `trigrams.bin` (or an in-memory rebuild when the file is absent).
    pub trigrams: TrigramIndex,
    /// `file_id → path`, with any leading `./` stripped.
    paths: Vec<String>,
    /// Inverse of [`Self::paths`].
    path_to_id: HashMap<String, u32>,
    /// Query-side thread pool, sized from `[performance] parallel_threads`.
    pool: rayon::ThreadPool,
    /// `IndexConfig::max_posting_list_entries` at open time (0 = unlimited).
    posting_cap: usize,
    fingerprint: Fingerprint,
}

impl std::fmt::Debug for OpenIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenIndex")
            .field("cache_dir", &self.cache_dir)
            .field("files", &self.paths.len())
            .field("threads", &self.pool.current_num_threads())
            .finish()
    }
}

/// Upper bound on the automatic thread count for query-time verification.
///
/// The indexer caps itself at 8 to limit cache contention while it writes; a
/// read-only verification pass scales further, and ripgrep's 5x from 16 threads
/// on the field-test box is the number to match.
const QUERY_AUTO_THREAD_CAP: usize = 32;

impl OpenIndex {
    fn open(cache: &CacheManager, fingerprint: Fingerprint) -> Result<Self> {
        let cache_dir = cache.path().to_path_buf();

        let content = ContentReader::open(cache_dir.join("content.bin"))
            .map_err(|e| ReflexError::CacheCorrupted(format!("content.bin: {e:#}")))?;

        let trigrams_path = cache_dir.join("trigrams.bin");
        let trigrams = if trigrams_path.exists() {
            match TrigramIndex::load(&trigrams_path) {
                Ok(index) => index,
                // A format from another Reflex version is not corruption (see
                // `ReflexError::CacheVersionMismatch`): serve from an in-memory
                // rebuild, and let the schema-hash check report the index stale
                // so the caller re-indexes.
                Err(e) if e.to_string().contains("Unsupported trigrams.bin version") => {
                    log::warn!("{}; rebuilding trigram index in memory for this process", e);
                    super::result::rebuild_trigram_index(&content)?
                }
                Err(e) => {
                    return Err(ReflexError::CacheCorrupted(format!("trigrams.bin: {e:#}")).into());
                }
            }
        } else {
            log::debug!("trigrams.bin not found, rebuilding from content store");
            super::result::rebuild_trigram_index(&content)?
        };

        if trigrams.file_count() != content.file_count() {
            return Err(ReflexError::CacheCorrupted(format!(
                "trigrams.bin lists {} files but content.bin holds {} (index written by two runs?)",
                trigrams.file_count(),
                content.file_count()
            ))
            .into());
        }

        let mut paths = Vec::with_capacity(content.file_count());
        let mut path_to_id = HashMap::with_capacity(content.file_count());
        for id in 0..content.file_count() {
            let raw = content
                .get_file_path(id as u32)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
            let normalized = raw.strip_prefix("./").unwrap_or(&raw).to_string();
            path_to_id.entry(normalized.clone()).or_insert(id as u32);
            paths.push(normalized);
        }

        let config = cache.load_index_config().unwrap_or_else(|e| {
            log::debug!("Using default index config for query pool: {}", e);
            crate::models::IndexConfig::default()
        });
        let threads =
            crate::models::resolve_thread_count(config.parallel_threads, QUERY_AUTO_THREAD_CAP);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("rfx-query-{i}"))
            .build()
            .context("Failed to create query thread pool")?;

        log::debug!(
            "Opened index {}: {} files, {} trigrams, {} query threads",
            cache_dir.display(),
            content.file_count(),
            trigrams.trigram_count(),
            threads
        );

        Ok(Self {
            cache_dir,
            content,
            trigrams,
            paths,
            path_to_id,
            pool,
            posting_cap: config.max_posting_list_entries,
            fingerprint,
        })
    }

    /// Number of files in the index.
    pub fn file_count(&self) -> usize {
        self.paths.len()
    }

    /// Array file id for a path as stored in the index (a leading `./` is ignored).
    pub fn file_id_for(&self, path: &str) -> Option<u32> {
        let normalized = path.strip_prefix("./").unwrap_or(path);
        self.path_to_id.get(normalized).copied()
    }

    /// Path for an array file id, without a leading `./`.
    pub fn path_of(&self, file_id: u32) -> Option<&str> {
        self.paths.get(file_id as usize).map(String::as_str)
    }

    /// The query-side thread pool.
    pub fn pool(&self) -> &rayon::ThreadPool {
        &self.pool
    }

    /// `max_posting_list_entries` the index was configured with (0 = unlimited).
    pub fn posting_cap(&self) -> usize {
        self.posting_cap
    }

    /// The cache directory this handle was opened from.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }
}

fn registry() -> &'static Mutex<HashMap<PathBuf, Arc<OpenIndex>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, Arc<OpenIndex>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn registry_key(cache_dir: &Path) -> PathBuf {
    cache_dir
        .canonicalize()
        .unwrap_or_else(|_| cache_dir.to_path_buf())
}

/// The shared handle for `cache`, opening it if no current one exists.
///
/// Cost on a hit: one lock, two `stat` calls, one `canonicalize`. A handle whose
/// files have been replaced on disk is dropped and reopened.
pub fn get_or_open(cache: &CacheManager) -> Result<Arc<OpenIndex>> {
    let cache_dir = cache.path();
    let key = registry_key(cache_dir);

    let fingerprint = match Fingerprint::current(cache_dir) {
        Ok(fp) => fp,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(ReflexError::CacheCorrupted(format!(
                "content.bin: missing from {}",
                cache_dir.display()
            ))
            .into());
        }
        Err(e) => return Err(e).context("Failed to stat index files"),
    };

    if let Ok(map) = registry().lock()
        && let Some(existing) = map.get(&key)
        && existing.fingerprint == fingerprint
    {
        return Ok(Arc::clone(existing));
    }

    let opened = Arc::new(OpenIndex::open(cache, fingerprint)?);
    if let Ok(mut map) = registry().lock() {
        map.insert(key, Arc::clone(&opened));
    }
    Ok(opened)
}

/// Drop the registry entry for `cache_dir`, so the next lookup reopens.
///
/// Called before and after an index write; also safe to call when nothing is open.
pub fn invalidate(cache_dir: &Path) {
    let key = registry_key(cache_dir);
    if let Ok(mut map) = registry().lock() {
        map.remove(&key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::Indexer;
    use crate::models::IndexConfig;
    use tempfile::TempDir;

    fn indexed_project() -> TempDir {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("a.rs"), "fn alpha() {}\n").unwrap();
        std::fs::write(temp.path().join("b.rs"), "fn beta() { alpha() }\n").unwrap();
        Indexer::new(CacheManager::new(temp.path()), IndexConfig::default())
            .index(temp.path(), false)
            .unwrap();
        temp
    }

    #[test]
    fn second_lookup_returns_same_handle() {
        let temp = indexed_project();
        let cache = CacheManager::new(temp.path());
        let first = get_or_open(&cache).unwrap();
        let second = get_or_open(&cache).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn file_id_lookup_ignores_dot_slash() {
        let temp = indexed_project();
        let open = get_or_open(&CacheManager::new(temp.path())).unwrap();
        let id = open.file_id_for("a.rs").expect("a.rs is indexed");
        assert_eq!(open.file_id_for("./a.rs"), Some(id));
        assert_eq!(open.path_of(id), Some("a.rs"));
        assert_eq!(open.file_id_for("missing.rs"), None);
    }

    #[test]
    fn reindex_yields_new_handle() {
        let temp = indexed_project();
        let cache = CacheManager::new(temp.path());
        let first = get_or_open(&cache).unwrap();
        assert_eq!(first.file_count(), 2);

        std::fs::write(temp.path().join("c.rs"), "fn gamma() {}\n").unwrap();
        Indexer::new(CacheManager::new(temp.path()), IndexConfig::default())
            .index(temp.path(), false)
            .unwrap();

        let second = get_or_open(&cache).unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(second.file_count(), 3);
    }

    #[test]
    fn truncated_content_bin_is_reported_as_corruption() {
        let temp = indexed_project();
        let content = temp.path().join(".reflex/content.bin");
        std::fs::OpenOptions::new()
            .write(true)
            .open(&content)
            .unwrap()
            .set_len(2)
            .unwrap();

        let err = get_or_open(&CacheManager::new(temp.path())).unwrap_err();
        let typed = err
            .downcast_ref::<ReflexError>()
            .expect("typed CacheCorrupted");
        assert!(matches!(typed, ReflexError::CacheCorrupted(_)));
        assert!(err.to_string().contains("content.bin"), "{err}");
    }
}
