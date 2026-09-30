//! Automatic update: bring a stale index up to date before a command reads it.
//!
//! Every command that reads the index (`rfx query`, `deps`, `analyze`, `rfx mcp`,
//! `rfx serve`, …) calls [`update_if_stale`] unless `--no-update` was given. The
//! freshness check decides what to do ([`crate::query::update_plan`]): nothing,
//! [`Indexer::update_paths`] on the paths it listed, or a full [`Indexer::index`]
//! run (which still reads only the files whose stat moved). No index at all is
//! built. See `.context/AUTO_UPDATE_RESEARCH.md`.
//!
//! An update never fails the command: when it cannot run (a read-only `.reflex/`,
//! another version's cache in a server, a symbol pass that does not yield) the
//! result is [`Updated::Skipped`] with the reason, and the command answers from the
//! index it has, with the check's `stale` verdict. Only a missing index whose
//! build fails is an error.

use crate::background_indexer::{BackgroundIndexer, IndexerState};
use crate::cache::CacheManager;
use crate::errors::ReflexError;
use crate::indexer::Indexer;
use crate::models::{IndexConfig, LOCK_WAIT_FOREVER};
use crate::query::UpdatePlan;
use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

/// How the calling front end wants an update run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpdateOptions {
    /// Start the background symbol pass after a build or a full run, and restart
    /// one an update cancelled. Only the `rfx` binary may set this: the pass is
    /// started as `<current executable> index-symbols-internal`.
    pub spawn_symbol_pass: bool,
    /// Rebuild a cache another released version wrote. The CLI does, as
    /// `rfx index` does; servers do not, because two servers of different versions
    /// would rebuild the same cache in turns forever.
    pub self_heal_version: bool,
    /// Say on stderr when a build or a full run starts (they take seconds on a
    /// large tree). A path update is silent. Stdout is never written.
    pub progress: bool,
}

impl UpdateOptions {
    /// A CLI command: rebuilds another version's cache.
    pub fn cli() -> Self {
        Self {
            spawn_symbol_pass: true,
            self_heal_version: true,
            progress: true,
        }
    }

    /// `rfx mcp` / `rfx serve`: never writes another version's cache.
    pub fn server() -> Self {
        Self {
            spawn_symbol_pass: true,
            self_heal_version: false,
            progress: false,
        }
    }

    /// A library caller or a test: no process is started.
    pub fn library() -> Self {
        Self {
            spawn_symbol_pass: false,
            self_heal_version: false,
            progress: false,
        }
    }
}

/// What [`update_if_stale`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Updated {
    /// The index was fresh.
    Nothing,
    /// `update_paths` on this many paths (it may have run a full index itself).
    Paths(usize),
    /// A full `index` run over an existing index; `changed` paths were reported.
    Index { changed: usize },
    /// There was no index (or another version's, rebuilt); it was built.
    Built,
    /// Not updated; the index is answered from as it is. The reason is for
    /// `warnings`.
    Skipped(String),
}

impl Updated {
    /// Whether the index files changed on disk.
    pub fn wrote(&self) -> bool {
        matches!(self, Self::Paths(_) | Self::Index { .. } | Self::Built)
    }
}

/// One update at a time per workspace in this process. Across processes,
/// `index.lock` serialises the runs themselves.
fn root_mutex(root: &Path) -> Arc<Mutex<()>> {
    static MUTEXES: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();
    let key = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let mut map = MUTEXES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    Arc::clone(map.entry(key).or_default())
}

/// The last update in this process that did not make the index fresh, per
/// workspace: the plan and the stat of each path it named. The same plan over
/// the same bytes is not tried again (it would run on every query).
type Attempt = (UpdatePlan, Vec<(PathBuf, Option<(u64, SystemTime)>)>);

fn failed_attempts() -> &'static Mutex<HashMap<PathBuf, Attempt>> {
    static FAILED: OnceLock<Mutex<HashMap<PathBuf, Attempt>>> = OnceLock::new();
    FAILED.get_or_init(|| Mutex::new(HashMap::new()))
}

fn stat_of(root: &Path, plan: &UpdatePlan) -> Attempt {
    let paths: &[PathBuf] = match plan {
        UpdatePlan::Paths(p) | UpdatePlan::Full(p) => p,
        _ => &[],
    };
    let stats = paths
        .iter()
        .map(|p| {
            let md = std::fs::symlink_metadata(root.join(p)).ok();
            (
                p.clone(),
                md.and_then(|m| Some((m.len(), m.modified().ok()?))),
            )
        })
        .collect();
    (plan.clone(), stats)
}

/// Bring the index of the workspace `cache` belongs to up to date, if the
/// freshness check says it is stale. See the module docs.
pub fn update_if_stale(cache: &CacheManager, opts: &UpdateOptions) -> Result<Updated> {
    let root = cache.workspace_root();
    let mutex = root_mutex(&root);
    let _one_at_a_time = mutex.lock().unwrap_or_else(|e| e.into_inner());

    if !cache.exists() {
        build(cache, &root, opts)?;
        return Ok(Updated::Built);
    }

    // Asked after taking the mutex: an update that just finished invalidated the
    // memo, so a waiter sees the fresh index and does nothing.
    let plan = match crate::query::update_plan(cache) {
        Ok(plan) => plan,
        Err(e) => return Ok(Updated::Skipped(format!("freshness check failed: {e}"))),
    };
    let key = root.canonicalize().unwrap_or_else(|_| root.clone());
    match &plan {
        UpdatePlan::Fresh => {
            if let Ok(mut failed) = failed_attempts().lock() {
                failed.remove(&key);
            }
            return Ok(Updated::Nothing);
        }
        UpdatePlan::Unknown(reason) => return Ok(Updated::Skipped(reason.clone())),
        UpdatePlan::Foreign(reason) if !opts.self_heal_version => {
            return Ok(Updated::Skipped(reason.clone()));
        }
        _ => {}
    }

    let attempt = stat_of(&root, &plan);
    if let Ok(failed) = failed_attempts().lock()
        && failed.get(&key) == Some(&attempt)
    {
        return Ok(Updated::Skipped(
            "an automatic update of these files did not make the index fresh; \
             run `rfx index` to rebuild it"
                .to_string(),
        ));
    }

    let result = match &plan {
        UpdatePlan::Foreign(_) => {
            crate::query::invalidate_caches(&root);
            match cache.clear() {
                Ok(()) => build(cache, &root, opts).map(|()| Updated::Built),
                Err(e) => Err(e),
            }
        }
        UpdatePlan::Paths(paths) => config(cache).and_then(|config| {
            with_symbol_pass_retry(cache, || {
                Indexer::new(CacheManager::new(&root), config.clone()).update_paths(&root, paths)
            })
            .map(|_| Updated::Paths(paths.len()))
        }),
        UpdatePlan::Full(_) => config(cache).and_then(|config| {
            if opts.progress {
                eprintln!("Updating the index …");
            }
            with_symbol_pass_retry(cache, || {
                Indexer::new(CacheManager::new(&root), config.clone()).index(&root, false)
            })
            .map(|stats| Updated::Index {
                changed: stats.new_files + stats.modified_files + stats.deleted_files,
            })
        }),
        UpdatePlan::Fresh | UpdatePlan::Unknown(_) => unreachable!("returned above"),
    };

    let updated = match result {
        Ok(updated) => updated,
        Err(e) => return Ok(Updated::Skipped(describe(&e))),
    };

    // Did it work? The same plan over the same bytes is not retried.
    match crate::query::update_plan(cache) {
        Ok(UpdatePlan::Fresh) => {
            if let Ok(mut failed) = failed_attempts().lock() {
                failed.remove(&key);
            }
        }
        Ok(after) if after == plan => {
            if let Ok(mut failed) = failed_attempts().lock() {
                failed.insert(key, attempt);
            }
        }
        _ => {}
    }

    restart_symbol_pass(cache, &root, opts, &updated);
    Ok(updated)
}

/// The config every index run uses, waiting for `index.lock` as long as it takes.
fn config(cache: &CacheManager) -> Result<IndexConfig> {
    let mut config = cache.effective_index_config(&[])?;
    config.lock_wait_secs = LOCK_WAIT_FOREVER;
    Ok(config)
}

fn build(cache: &CacheManager, root: &Path, opts: &UpdateOptions) -> Result<()> {
    let config = config(cache)?;
    if opts.progress {
        eprintln!("Building the index for {} …", root.display());
    }
    with_symbol_pass_retry(cache, || {
        Indexer::new(CacheManager::new(root), config.clone()).index(root, false)
    })?;
    restart_symbol_pass(cache, root, opts, &Updated::Built);
    Ok(())
}

/// Run `f`, again while a symbol pass that would not yield is still making
/// progress (each attempt already waits up to 10 s for it).
fn with_symbol_pass_retry<T>(cache: &CacheManager, mut f: impl FnMut() -> Result<T>) -> Result<T> {
    let processed = || {
        BackgroundIndexer::get_status(cache.path())
            .ok()
            .flatten()
            .map(|s| s.processed_files)
    };
    let mut last = processed();
    loop {
        match f() {
            Err(e)
                if matches!(
                    e.downcast_ref::<ReflexError>(),
                    Some(ReflexError::SymbolIndexingInProgress { .. })
                ) =>
            {
                let now = processed();
                if now > last {
                    last = now;
                    continue;
                }
                return Err(e);
            }
            other => return other,
        }
    }
}

/// Start the symbol pass after a run that needs it: a build or a full run, or
/// any run that cancelled a pass in progress.
fn restart_symbol_pass(cache: &CacheManager, root: &Path, opts: &UpdateOptions, updated: &Updated) {
    if !opts.spawn_symbol_pass || !updated.wrote() || BackgroundIndexer::is_running(cache.path()) {
        return;
    }
    let cancelled = BackgroundIndexer::get_status(cache.path())
        .ok()
        .flatten()
        .is_some_and(|s| s.state == IndexerState::Cancelled);
    if (matches!(updated, Updated::Built | Updated::Index { .. }) || cancelled)
        && let Err(e) = BackgroundIndexer::spawn_detached(root)
    {
        log::warn!("Could not start the symbol pass: {e}");
    }
}

/// The reason an update did not run, for `warnings`.
fn describe(e: &anyhow::Error) -> String {
    format!("automatic index update failed: {e}")
}
