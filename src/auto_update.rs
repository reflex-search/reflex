//! Automatic update: bring a stale index up to date before a command reads it.
//!
//! Every command that reads the index (`rfx query`, `deps`, `analyze`, `rfx mcp`,
//! `rfx serve`, …) calls [`update_if_stale`] unless `--no-update` was given. The
//! freshness check decides what to do ([`crate::query::update_plan`]): nothing,
//! [`Indexer::update_paths`] on the paths it listed, or a full [`Indexer::index`]
//! run (which still reads only the files whose stat moved). No index at all is
//! built. See `.context/AUTO_UPDATE_RESEARCH.md`.
//!
//! A cache another released version wrote is rebuilt for this one when the front
//! end allows it ([`SelfHeal`]): the CLI always, a server only after an upgrade
//! (the cache's writer is an older release).
//!
//! An update never fails the command: when it cannot run (a read-only `.reflex/`,
//! a newer version's cache in a server, a symbol pass that does not yield) the
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
    /// Which caches written by another released version this front end rebuilds
    /// ([`SelfHeal`]).
    pub self_heal_version: SelfHeal,
    /// Say on stderr when a build or a full run starts (they take seconds on a
    /// large tree). A path update is silent. Stdout is never written.
    pub progress: bool,
}

/// Which caches written by ANOTHER released version an update may rebuild.
///
/// The writer refuses another version's cache (`CacheManager::assert_writable`),
/// so an update that cannot take it over answers from it as it is, `stale`, for
/// as long as the other version's cache stays. Taking it over means clearing it
/// and building it again for this version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelfHeal {
    /// Never (library callers and tests).
    Never,
    /// Only a cache an OLDER release wrote: an upgrade (`rfx mcp`, `rfx serve`).
    /// A NEWER one is left alone, so two servers of different versions sharing one
    /// `.reflex/` do not rebuild it in turns forever: the newer one takes it over
    /// once, and the older one then refuses it.
    OlderOnly,
    /// Any other version (CLI commands, as `rfx index` does: a human picked
    /// this binary).
    Any,
}

impl SelfHeal {
    /// Whether a cache `owner` wrote may be rebuilt. `None` (the writer is not
    /// known) is rebuilt only by [`SelfHeal::Any`].
    pub fn allows(self, owner: Option<&str>) -> bool {
        match self {
            Self::Never => false,
            Self::OlderOnly => {
                owner.is_some_and(|v| is_older_release(v, env!("CARGO_PKG_VERSION")))
            }
            Self::Any => true,
        }
    }
}

/// Whether release `owner` is strictly older than release `this`, comparing
/// `MAJOR.MINOR.PATCH`. Anything else (a pre-release or build suffix on equal
/// numbers, a version that does not parse) is not older: refusing is the safe
/// side, and the CLI still rebuilds it.
pub fn is_older_release(owner: &str, this: &str) -> bool {
    fn parse(v: &str) -> Option<(u64, u64, u64)> {
        let core = v.split(['-', '+']).next()?;
        let mut parts = core.split('.');
        let n = (
            parts.next()?.parse().ok()?,
            parts.next()?.parse().ok()?,
            parts.next()?.parse().ok()?,
        );
        parts.next().is_none().then_some(n)
    }
    matches!((parse(owner), parse(this)), (Some(a), Some(b)) if a < b)
}

impl UpdateOptions {
    /// A CLI command: rebuilds any other version's cache.
    pub fn cli() -> Self {
        Self {
            spawn_symbol_pass: true,
            self_heal_version: SelfHeal::Any,
            progress: true,
        }
    }

    /// `rfx mcp` / `rfx serve`: rebuilds an older release's cache, never a newer one.
    pub fn server() -> Self {
        Self {
            spawn_symbol_pass: true,
            self_heal_version: SelfHeal::OlderOnly,
            progress: false,
        }
    }

    /// A library caller or a test: no process is started.
    pub fn library() -> Self {
        Self {
            spawn_symbol_pass: false,
            self_heal_version: SelfHeal::Never,
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
    let planned = match crate::query::plan_update(cache) {
        Ok(planned) => planned,
        Err(e) => return Ok(Updated::Skipped(format!("freshness check failed: {e}"))),
    };
    let plan = planned.plan.clone();
    let key = root.canonicalize().unwrap_or_else(|_| root.clone());
    match &plan {
        UpdatePlan::Fresh => {
            if let Ok(mut failed) = failed_attempts().lock() {
                failed.remove(&key);
            }
            return Ok(Updated::Nothing);
        }
        UpdatePlan::Unknown(reason) => return Ok(Updated::Skipped(reason.clone())),
        _ => {}
    }

    // Another version wrote this cache. A different format plans `Foreign`; the
    // same format still has the other version's stamp, and the writer refuses it
    // (`assert_writable`), so either way the update is a takeover or nothing.
    let other = other_writer(cache);
    let foreign = matches!(plan, UpdatePlan::Foreign(_)) || other.is_some();
    if foreign && !opts.self_heal_version.allows(other.as_deref()) {
        return Ok(Updated::Skipped(refusal(&plan, other.as_deref(), opts)));
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
        _ if foreign => take_over(cache, &root, opts),
        UpdatePlan::Foreign(_) => unreachable!("foreign is set for this plan"),
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

    // Did it work? For a path update, only those paths are compared again; the
    // same plan over the same bytes is not retried.
    if crate::query::settle_update(cache, &planned) {
        if let Ok(mut failed) = failed_attempts().lock() {
            failed.remove(&key);
        }
        restart_symbol_pass(cache, &root, opts, &updated);
        return Ok(updated);
    }
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

/// The version stamped on this cache when it is not this binary's version.
fn other_writer(cache: &CacheManager) -> Option<String> {
    cache
        .cache_owner()
        .map(|(version, _)| version)
        .filter(|v| v != env!("CARGO_PKG_VERSION"))
}

/// Rebuild another version's cache for this one: clear it, then build it.
fn take_over(cache: &CacheManager, root: &Path, opts: &UpdateOptions) -> Result<Updated> {
    crate::query::invalidate_caches(root);
    // Another process of this version (a second server after the same upgrade) may
    // have rebuilt it while this one waited: update that index, do not clear it.
    if other_writer(cache).is_none() && cache.cache_owner().is_some() {
        let config = config(cache)?;
        return with_symbol_pass_retry(cache, || {
            Indexer::new(CacheManager::new(root), config.clone()).index(root, false)
        })
        .map(|stats| Updated::Index {
            changed: stats.new_files + stats.modified_files + stats.deleted_files,
        });
    }
    cache.clear()?;
    build(cache, root, opts)?;
    Ok(Updated::Built)
}

/// Why another version's cache was not rebuilt, for `warnings`: who wrote it and
/// the command that rebuilds it (no MCP tool can, unless `index_project` is listed).
fn refusal(plan: &UpdatePlan, other: Option<&str>, opts: &UpdateOptions) -> String {
    let this = env!("CARGO_PKG_VERSION");
    match other {
        Some(v) if opts.self_heal_version == SelfHeal::OlderOnly => format!(
            "this .reflex/ was written by reflex {v}, which is newer than this binary \
             (reflex {this}), so it is not rebuilt automatically. Upgrade this binary, \
             or run `rfx index` to rebuild the index for reflex {this}."
        ),
        Some(v) => format!(
            "this .reflex/ was written by reflex {v}; this binary is reflex {this}. \
             Run `rfx index` to rebuild it for reflex {this}."
        ),
        None => match plan {
            UpdatePlan::Foreign(reason) => reason.clone(),
            _ => "this .reflex/ was written by another reflex version; run `rfx index` \
                  to rebuild it"
                .to_string(),
        },
    }
}

/// The reason an update did not run, for `warnings`.
fn describe(e: &anyhow::Error) -> String {
    format!("automatic index update failed: {e}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn older_release_compares_major_minor_patch() {
        assert!(is_older_release("2.2.0", "2.3.1"));
        assert!(is_older_release("2.3.0", "2.3.1"));
        assert!(is_older_release("1.9.9", "2.0.0"));
        assert!(is_older_release("9.9.9", "10.0.0"));
        assert!(!is_older_release("2.3.1", "2.3.1"));
        assert!(!is_older_release("2.4.0", "2.3.1"));
        assert!(!is_older_release("10.0.0", "9.9.9"));
    }

    #[test]
    fn unparsable_or_suffixed_versions_are_not_older() {
        assert!(!is_older_release("2.3.1-rc.1", "2.3.1"));
        assert!(!is_older_release("2.3", "2.3.1"));
        assert!(!is_older_release("2.3.1.4", "2.4.0"));
        assert!(!is_older_release("garbage", "2.3.1"));
        assert!(!is_older_release("", "2.3.1"));
        // A suffix on an older core is still older.
        assert!(is_older_release("2.2.0-rc.1", "2.3.1"));
    }

    #[test]
    fn self_heal_rules() {
        assert!(SelfHeal::Any.allows(Some("99.0.0")));
        assert!(SelfHeal::Any.allows(None));
        assert!(SelfHeal::OlderOnly.allows(Some("0.0.1")));
        assert!(!SelfHeal::OlderOnly.allows(Some("99.0.0")));
        assert!(!SelfHeal::OlderOnly.allows(None));
        assert!(!SelfHeal::Never.allows(Some("0.0.1")));
    }
}
