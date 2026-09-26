//! The Pulse LLM writing pass.
//!
//! Pulse builds a complete site without an LLM. This pass only fills narrative slots,
//! and every task goes through one executor ([`run_tasks`]) and one content-addressed
//! cache ([`cache::WriteCache`]). Evidence packs, output contracts and the grounding
//! gate build on top of this layer.

pub mod cache;
pub mod provider;
pub mod run;

pub use cache::WriteCache;
pub use provider::{LlmHandle, ModelIdent};
pub use run::{OutputSpec, RunStats, TaskOutcome, WriteConfig, WriteOutcome, WriteTask, run_tasks};

use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Writing-pass options from the command line. `None` fields fall back to `[pulse.write]`.
#[derive(Debug, Clone, Default)]
pub struct WriteOptions {
    pub mode: LlmMode,
    pub force: ForceScope,
    pub dry_run: bool,
    pub concurrency: Option<usize>,
    pub max_llm_tokens: Option<u64>,
    pub cache_dir: Option<PathBuf>,
    pub no_prune: bool,
    pub model: Option<String>,
}

impl WriteOptions {
    /// The LLM is off.
    pub fn off() -> Self {
        Self {
            mode: LlmMode::Off,
            ..Self::default()
        }
    }
}

/// A ready-to-run writing pass: provider handle, cache and executor settings.
pub struct WriteSession {
    pub llm: LlmHandle,
    pub cache: WriteCache,
    pub cfg: WriteConfig,
}

impl WriteSession {
    /// Merge CLI options over `[pulse.write]` and resolve the provider.
    pub fn open(reflex_cache: &Path, opts: &WriteOptions) -> Self {
        let mut settings = crate::pulse::config::load_pulse_config(reflex_cache)
            .map(|c| c.write)
            .unwrap_or_else(|e| {
                log::warn!("ignoring invalid [pulse] config: {e:#}");
                Default::default()
            });
        if opts.model.is_some() {
            settings.model = opts.model.clone();
        }
        let cache_dir = opts
            .cache_dir
            .clone()
            .or_else(|| settings.cache_dir.as_ref().map(PathBuf::from))
            .unwrap_or_else(|| WriteCache::default_dir(reflex_cache));
        let cfg = WriteConfig {
            mode: opts.mode,
            concurrency: opts.concurrency.unwrap_or(settings.concurrency),
            max_llm_tokens: opts.max_llm_tokens.or(settings.max_llm_tokens),
            force: opts.force.clone(),
            dry_run: opts.dry_run,
            keep_runs: settings.keep_runs,
            prune: !opts.no_prune,
            cache_model_agnostic: settings.cache_model_agnostic,
            ..WriteConfig::default()
        };
        Self {
            llm: provider::resolve(opts.mode, &settings),
            cache: WriteCache::new(cache_dir),
            cfg,
        }
    }

    pub fn run(&self, tasks: Vec<WriteTask>) -> WriteOutcome {
        run_tasks(tasks, &self.llm, &self.cache, &self.cfg)
    }

    /// Run one task and return its text, printing the summary on failure.
    pub fn run_one(&self, task: WriteTask) -> Option<String> {
        let id = task.id.clone();
        let out = self.run(vec![task]);
        if out.degraded.is_some() {
            eprintln!("{}", out.summary());
        }
        out.text(&id).map(str::to_string)
    }
}

/// Whether the writing pass may call a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LlmMode {
    /// Call the provider on cache misses.
    #[default]
    On,
    /// Structural output only.
    Off,
    /// Use cached answers; never call. For CI jobs without secrets (fork PRs).
    CacheOnly,
}

impl FromStr for LlmMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "on" => Ok(Self::On),
            "off" => Ok(Self::Off),
            "cache-only" | "cache_only" | "cached" => Ok(Self::CacheOnly),
            other => Err(format!(
                "invalid --llm value '{other}' (expected on, off or cache-only)"
            )),
        }
    }
}

/// Which tasks `--force-renarrate` re-runs, ignoring the cache.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum ForceScope {
    #[default]
    None,
    All,
    /// Task kinds (`modules`, `overview`, …) or globs over task ids (`module:src/pulse*`).
    Only(Vec<String>),
}

impl ForceScope {
    /// Parse `all` or a comma-separated list. An empty string means none.
    pub fn parse(spec: &str) -> Self {
        let parts: Vec<String> = spec
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect();
        if parts.is_empty() {
            Self::None
        } else if parts.iter().any(|p| p == "all") {
            Self::All
        } else {
            Self::Only(parts)
        }
    }

    pub fn matches(&self, task: &WriteTask) -> bool {
        match self {
            Self::None => false,
            Self::All => true,
            Self::Only(parts) => parts.iter().any(|p| {
                p == task.kind
                    || globset::Glob::new(p)
                        .map(|g| g.compile_matcher().is_match(&task.id))
                        .unwrap_or(false)
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::providers::ProviderErrorKind;
    use crate::semantic::providers::mock::MockLlmProvider;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::TempDir;

    const WORDS: &str = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen";

    fn task(id: &str, kind: &'static str) -> WriteTask {
        WriteTask::text(id, kind, "SYSTEM", format!("{WORDS} {id}"))
    }

    fn fast_cfg() -> WriteConfig {
        WriteConfig {
            base_backoff: Duration::from_millis(1),
            ..WriteConfig::default()
        }
    }

    fn echo() -> Arc<MockLlmProvider> {
        Arc::new(MockLlmProvider::routed(|req| {
            Ok(format!("text for {}", req.tag))
        }))
    }

    fn handle(p: &Arc<MockLlmProvider>) -> LlmHandle {
        LlmHandle::from_provider(p.clone())
    }

    #[test]
    fn llm_mode_parse() {
        assert_eq!("on".parse::<LlmMode>().unwrap(), LlmMode::On);
        assert_eq!("OFF".parse::<LlmMode>().unwrap(), LlmMode::Off);
        assert_eq!("cache-only".parse::<LlmMode>().unwrap(), LlmMode::CacheOnly);
        assert!("maybe".parse::<LlmMode>().is_err());
    }

    #[test]
    fn force_scope_parse_and_match() {
        assert_eq!(ForceScope::parse(""), ForceScope::None);
        assert_eq!(ForceScope::parse("all"), ForceScope::All);
        let s = ForceScope::parse("overview, module:src/pulse*");
        assert!(s.matches(&task("project-overview", "overview")));
        assert!(s.matches(&task("module:src/pulse/write", "modules")));
        assert!(!s.matches(&task("module:src/query", "modules")));
    }

    #[test]
    fn calls_then_hits_the_cache_on_rerun() {
        let dir = TempDir::new().unwrap();
        let cache = WriteCache::new(dir.path());
        let p = echo();
        let tasks = vec![task("a", "modules"), task("b", "modules")];

        let first = run_tasks(tasks.clone(), &handle(&p), &cache, &fast_cfg());
        assert!(first.degraded.is_none(), "{:?}", first.degraded);
        assert_eq!(first.stats.called, 2);
        assert_eq!(first.text("a"), Some("text for a"));
        assert_eq!(p.call_count(), 2);

        let second = run_tasks(tasks, &handle(&p), &cache, &fast_cfg());
        assert_eq!(second.stats.cached, 2);
        assert_eq!(second.stats.called, 0);
        assert_eq!(second.text("b"), Some("text for b"));
        assert_eq!(p.call_count(), 2, "no new calls on a warm cache");
    }

    #[test]
    fn system_and_user_are_sent_separately() {
        let dir = TempDir::new().unwrap();
        let p = echo();
        run_tasks(
            vec![task("a", "modules")],
            &handle(&p),
            &WriteCache::new(dir.path()),
            &fast_cfg(),
        );
        let req = &p.requests()[0];
        assert_eq!(req.system, "SYSTEM");
        assert!(req.user.ends_with(" a"));
        assert_eq!(req.output, "text");
    }

    #[test]
    fn identical_requests_share_one_call() {
        let dir = TempDir::new().unwrap();
        let p = echo();
        let a = WriteTask::text("a", "modules", "S", WORDS);
        let b = WriteTask::text("b", "modules", "S", WORDS);
        let out = run_tasks(
            vec![a, b],
            &handle(&p),
            &WriteCache::new(dir.path()),
            &fast_cfg(),
        );
        assert_eq!(p.call_count(), 1);
        assert_eq!(out.text("a"), out.text("b"));
        assert!(out.text("a").is_some());
    }

    #[test]
    fn thin_input_is_suppressed_without_a_call() {
        let dir = TempDir::new().unwrap();
        let p = echo();
        let thin = WriteTask::text("thin", "modules", "S", "too short");
        let out = run_tasks(
            vec![thin],
            &handle(&p),
            &WriteCache::new(dir.path()),
            &fast_cfg(),
        );
        assert!(matches!(
            out.results["thin"].outcome,
            TaskOutcome::Suppressed(_)
        ));
        assert_eq!(p.call_count(), 0);
        assert!(out.degraded.is_none());
    }

    #[test]
    fn model_change_misses_unless_model_agnostic() {
        let dir = TempDir::new().unwrap();
        let cache = WriteCache::new(dir.path());
        let t = vec![task("a", "modules")];
        run_tasks(t.clone(), &handle(&echo()), &cache, &fast_cfg());

        let other = Arc::new(MockLlmProvider::routed(|_| Ok("new".into())).with_model("other"));
        let out = run_tasks(t.clone(), &handle(&other), &cache, &fast_cfg());
        assert_eq!(out.stats.called, 1, "different model → different key");

        let agnostic = WriteConfig {
            cache_model_agnostic: true,
            ..fast_cfg()
        };
        run_tasks(t.clone(), &handle(&echo()), &cache, &agnostic);
        let again = Arc::new(MockLlmProvider::routed(|_| Ok("x".into())).with_model("third"));
        let out = run_tasks(t, &handle(&again), &cache, &agnostic);
        assert_eq!(out.stats.cached, 1);
        assert_eq!(again.call_count(), 0);
    }

    #[test]
    fn fatal_probe_degrades_the_whole_run_and_ignores_hits() {
        let dir = TempDir::new().unwrap();
        let cache = WriteCache::new(dir.path());
        // Warm one entry.
        run_tasks(
            vec![task("a", "modules")],
            &handle(&echo()),
            &cache,
            &fast_cfg(),
        );

        let p = Arc::new(
            MockLlmProvider::routed(|_| Ok("x".into())).fail_first(1, ProviderErrorKind::Auth),
        );
        let tasks = vec![
            task("a", "modules"),
            task("b", "modules"),
            task("c", "modules"),
        ];
        let out = run_tasks(tasks, &handle(&p), &cache, &fast_cfg());
        assert!(out.degraded.is_some());
        assert_eq!(p.call_count(), 1, "nothing is sent after a fatal probe");
        assert_eq!(
            out.text("a"),
            None,
            "cache hits are not used in a degraded run"
        );
        assert!(!out.has_text());
    }

    #[test]
    fn transient_errors_are_retried() {
        let dir = TempDir::new().unwrap();
        let p = Arc::new(
            MockLlmProvider::routed(|_| Ok("ok".into()))
                .fail_first(2, ProviderErrorKind::RateLimited),
        );
        let out = run_tasks(
            vec![task("a", "modules")],
            &handle(&p),
            &WriteCache::new(dir.path()),
            &fast_cfg(),
        );
        assert_eq!(out.text("a"), Some("ok"));
        assert_eq!(out.stats.retries, 2);
        assert_eq!(p.call_count(), 3);
    }

    #[test]
    fn consecutive_failures_trip_the_breaker() {
        let dir = TempDir::new().unwrap();
        let p = Arc::new(MockLlmProvider::routed(|_| {
            Err(MockLlmProvider::error(ProviderErrorKind::Server))
        }));
        let cfg = WriteConfig {
            concurrency: 1,
            max_attempts: 1,
            breaker_threshold: 3,
            ..fast_cfg()
        };
        let tasks: Vec<_> = (0..10)
            .map(|i| task(&format!("t{i:02}"), "modules"))
            .collect();
        let out = run_tasks(tasks, &handle(&p), &WriteCache::new(dir.path()), &cfg);
        assert!(out.degraded.as_deref().unwrap().contains("consecutive"));
        assert!(
            p.call_count() <= 4,
            "stops soon after the breaker trips: {}",
            p.call_count()
        );
    }

    #[test]
    fn low_success_rate_degrades_but_keeps_successes_cached() {
        let dir = TempDir::new().unwrap();
        let cache = WriteCache::new(dir.path());
        let p = Arc::new(MockLlmProvider::routed(|req| {
            if req.tag == "a" {
                Ok("fine".into())
            } else {
                Err(MockLlmProvider::error(ProviderErrorKind::Server))
            }
        }));
        let cfg = WriteConfig {
            max_attempts: 1,
            ..fast_cfg()
        };
        let tasks = vec![
            task("a", "modules"),
            task("b", "modules"),
            task("c", "modules"),
        ];
        let out = run_tasks(tasks.clone(), &handle(&p), &cache, &cfg);
        assert!(out.degraded.is_some());
        assert_eq!(out.text("a"), None);

        let rerun = run_tasks(vec![task("a", "modules")], &handle(&echo()), &cache, &cfg);
        assert_eq!(rerun.results["a"].outcome, TaskOutcome::Cached);
    }

    #[test]
    fn budget_defers_low_priority_tasks_deterministically() {
        let dir = TempDir::new().unwrap();
        let p = echo();
        let one = task("x", "modules");
        let cost = ((one.system.len() + one.user.len()) as f64 / 3.5).ceil() as u64
            + (one.max_tokens as f64 * 0.6).ceil() as u64;
        let tasks = vec![
            task("low", "modules").with_priority(200),
            task("high", "overview").with_priority(10),
            task("mid", "modules").with_priority(50),
        ];
        let cfg = WriteConfig {
            max_llm_tokens: Some(cost * 2 + 5),
            ..fast_cfg()
        };
        let out = run_tasks(tasks, &handle(&p), &WriteCache::new(dir.path()), &cfg);
        assert!(out.truncated);
        assert_eq!(out.results["low"].outcome, TaskOutcome::Deferred);
        assert_eq!(out.results["high"].outcome, TaskOutcome::Called);
        assert_eq!(out.results["mid"].outcome, TaskOutcome::Called);
        assert_eq!(out.stats.pruned, 0, "a truncated run never prunes");
    }

    #[test]
    fn dry_run_makes_no_calls_and_reports_a_plan() {
        let dir = TempDir::new().unwrap();
        let p = echo();
        let cfg = WriteConfig {
            dry_run: true,
            ..fast_cfg()
        };
        let out = run_tasks(
            vec![task("a", "modules"), task("o", "overview")],
            &handle(&p),
            &WriteCache::new(dir.path()),
            &cfg,
        );
        assert_eq!(p.call_count(), 0);
        let table = out.plan_table();
        assert!(table.contains("modules"), "{table}");
        assert!(table.contains("total"), "{table}");
        assert!(table.contains("mock/mock-model"), "{table}");
        assert_eq!(out.text("a"), None);
    }

    #[test]
    fn cache_only_uses_hits_and_never_calls() {
        let dir = TempDir::new().unwrap();
        let cache = WriteCache::new(dir.path());
        run_tasks(
            vec![task("a", "modules")],
            &handle(&echo()),
            &cache,
            &fast_cfg(),
        );

        let ident_only = LlmHandle {
            ident: handle(&echo()).ident,
            provider: None,
            unavailable: Some("cache-only mode".into()),
        };
        let cfg = WriteConfig {
            mode: LlmMode::CacheOnly,
            ..fast_cfg()
        };
        let out = run_tasks(
            vec![task("a", "modules"), task("b", "modules")],
            &ident_only,
            &cache,
            &cfg,
        );
        assert!(out.degraded.is_none());
        assert_eq!(out.text("a"), Some("text for a"));
        assert_eq!(out.results["b"].outcome, TaskOutcome::NotCached);
    }

    #[test]
    fn forced_scope_recalls_only_that_scope() {
        let dir = TempDir::new().unwrap();
        let cache = WriteCache::new(dir.path());
        let tasks = vec![task("m", "modules"), task("o", "overview")];
        run_tasks(tasks.clone(), &handle(&echo()), &cache, &fast_cfg());

        let p = echo();
        let cfg = WriteConfig {
            force: ForceScope::parse("overview"),
            ..fast_cfg()
        };
        let out = run_tasks(tasks, &handle(&p), &cache, &cfg);
        let tags: Vec<String> = p.requests().into_iter().map(|r| r.tag).collect();
        assert_eq!(tags, vec!["o".to_string()]);
        assert_eq!(out.results["m"].outcome, TaskOutcome::Cached);
    }

    #[test]
    fn successful_run_prunes_stale_entries() {
        let dir = TempDir::new().unwrap();
        let cache = WriteCache::new(dir.path());
        let cfg = WriteConfig {
            keep_runs: 1,
            ..fast_cfg()
        };
        run_tasks(vec![task("old", "modules")], &handle(&echo()), &cache, &cfg);
        assert_eq!(cache.count(), 1);
        let out = run_tasks(vec![task("new", "modules")], &handle(&echo()), &cache, &cfg);
        assert_eq!(out.stats.pruned, 1);
        assert_eq!(cache.count(), 1);
    }

    #[test]
    fn off_mode_skips_everything() {
        let dir = TempDir::new().unwrap();
        let p = echo();
        let cfg = WriteConfig {
            mode: LlmMode::Off,
            ..fast_cfg()
        };
        let out = run_tasks(
            vec![task("a", "modules")],
            &handle(&p),
            &WriteCache::new(dir.path()),
            &cfg,
        );
        assert_eq!(out.results["a"].outcome, TaskOutcome::Skipped);
        assert_eq!(p.call_count(), 0);
        assert!(out.degraded.is_none());
    }
}
