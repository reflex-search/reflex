//! The writing-pass executor: plan, budget, call, cache, degrade.
//!
//! Flow of [`run_tasks`]:
//! 1. **Plan.** Suppress tasks with too little input. Compute each cache key. A hit is
//!    used unless the task is in the `--force-renarrate` scope. Tasks with the same key
//!    share one call.
//! 2. **Budget.** With `max_llm_tokens`, keep calls in `(priority, id)` order until the
//!    estimate reaches the cap; the rest are deferred. The cache fills, so later runs
//!    complete the rest.
//! 3. **Dry run** stops here.
//! 4. **Probe.** The first call runs alone. A fatal error (auth, unknown model, bad
//!    request) degrades the whole run to structural output.
//! 5. **Fan-out** under a semaphore, with retries and backoff for transient errors,
//!    halved concurrency after a 429, and a circuit breaker after consecutive failures.
//! 6. **Verdict.** If fewer than `degrade_threshold` of the eligible tasks succeeded, the
//!    whole run degrades. Successful answers stay cached either way.
//! 7. **Prune** the cache after a complete, successful run.

use super::cache::{CacheEntry, KeyMaterial, WriteCache};
use super::provider::LlmHandle;
use super::{ForceScope, LlmMode};
use crate::semantic::providers::{
    CompletionRequest, LlmProvider, OutputMode, ProviderError, ProviderErrorKind, StopReason, Usage,
};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The output contract of one task.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputSpec {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: serde_json::Value,
    },
}

impl OutputSpec {
    pub fn as_mode(&self) -> OutputMode<'_> {
        match self {
            OutputSpec::Text => OutputMode::Text,
            OutputSpec::JsonObject => OutputMode::JsonObject,
            OutputSpec::JsonSchema { name, schema } => OutputMode::JsonSchema { name, schema },
        }
    }
}

/// One unit of LLM writing.
#[derive(Debug, Clone)]
pub struct WriteTask {
    /// Unique and stable, e.g. `module:src/pulse`. Results are keyed by it.
    pub id: String,
    /// Task kind, used for `--force-renarrate` scopes and the plan table.
    pub kind: &'static str,
    /// Bump when parsing or prompt semantics change without a text change.
    pub prompt_version: u32,
    pub system: String,
    pub user: String,
    pub output: OutputSpec,
    pub max_tokens: u32,
    /// Lower runs first when a budget truncates the plan.
    pub priority: u16,
    /// Suppress the task when `user` has fewer words than this.
    pub min_words: usize,
}

impl WriteTask {
    /// A free-text task with default limits.
    pub fn text(
        id: impl Into<String>,
        kind: &'static str,
        system: impl Into<String>,
        user: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            prompt_version: 1,
            system: system.into(),
            user: user.into(),
            output: OutputSpec::Text,
            max_tokens: 1200,
            priority: 100,
            min_words: 15,
        }
    }

    pub fn with_output(mut self, output: OutputSpec) -> Self {
        self.output = output;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_priority(mut self, priority: u16) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_prompt_version(mut self, v: u32) -> Self {
        self.prompt_version = v;
        self
    }

    fn est_input_tokens(&self) -> u64 {
        // ~3.5 characters per token; the budget headroom absorbs the error.
        ((self.system.len() + self.user.len()) as f64 / 3.5).ceil() as u64
    }

    fn est_output_tokens(&self) -> u64 {
        (self.max_tokens as f64 * 0.6).ceil() as u64
    }
}

/// Executor settings.
#[derive(Debug, Clone)]
pub struct WriteConfig {
    pub mode: LlmMode,
    /// Concurrent calls. `0` means one per job.
    pub concurrency: usize,
    pub max_llm_tokens: Option<u64>,
    pub force: ForceScope,
    pub dry_run: bool,
    /// Minimum share of eligible tasks that must succeed, else the run degrades.
    pub degrade_threshold: f32,
    pub keep_runs: usize,
    pub prune: bool,
    pub cache_model_agnostic: bool,
    /// Attempts per call, including the first.
    pub max_attempts: u32,
    pub base_backoff: Duration,
    /// Consecutive failed calls that stop the run.
    pub breaker_threshold: usize,
}

impl Default for WriteConfig {
    fn default() -> Self {
        Self {
            mode: LlmMode::On,
            concurrency: 4,
            max_llm_tokens: None,
            force: ForceScope::None,
            dry_run: false,
            degrade_threshold: 0.9,
            keep_runs: 3,
            prune: true,
            cache_model_agnostic: false,
            max_attempts: 3,
            base_backoff: Duration::from_secs(1),
            breaker_threshold: 5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TaskOutcome {
    /// Answer read from the cache.
    Cached,
    /// Answer from a provider call this run.
    Called,
    /// Not sent: the input is too thin to write about.
    Suppressed(String),
    /// Not sent: outside the token budget or the run stopped first.
    Deferred,
    /// Cache-only mode and no entry.
    NotCached,
    Failed(String),
    /// The LLM is off.
    Skipped,
}

#[derive(Debug, Clone)]
pub struct TaskResult {
    pub kind: &'static str,
    pub outcome: TaskOutcome,
    pub text: Option<String>,
    pub usage: Option<Usage>,
    est_input: u64,
    est_output: u64,
}

#[derive(Debug, Clone, Default)]
pub struct RunStats {
    pub cached: usize,
    pub called: usize,
    pub suppressed: usize,
    pub deferred: usize,
    pub not_cached: usize,
    pub failed: usize,
    pub skipped: usize,
    pub calls_made: usize,
    pub retries: usize,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    pub pruned: usize,
    pub elapsed: Duration,
}

/// Everything a run produced.
#[derive(Debug, Clone)]
pub struct WriteOutcome {
    pub mode: LlmMode,
    pub results: BTreeMap<String, TaskResult>,
    /// Set when the whole run fell back to structural output.
    pub degraded: Option<String>,
    /// Some calls were deferred by the token budget.
    pub truncated: bool,
    pub dry_run: bool,
    pub model: Option<String>,
    pub stats: RunStats,
}

impl WriteOutcome {
    /// The text for a task, or `None` when the section must render structurally.
    pub fn text(&self, id: &str) -> Option<&str> {
        if self.degraded.is_some() || self.dry_run {
            return None;
        }
        self.results.get(id)?.text.as_deref()
    }

    /// True when any task produced text.
    pub fn has_text(&self) -> bool {
        self.degraded.is_none() && !self.dry_run && self.results.values().any(|r| r.text.is_some())
    }

    /// Per-kind plan: tasks, cached, to-call, suppressed, deferred, estimated tokens.
    pub fn plan_table(&self) -> String {
        #[derive(Default)]
        struct Row {
            tasks: usize,
            cached: usize,
            to_call: usize,
            suppressed: usize,
            deferred: usize,
            est_in: u64,
            est_out: u64,
        }
        let mut rows: BTreeMap<&str, Row> = BTreeMap::new();
        let mut total = Row::default();
        for r in self.results.values() {
            for row in [rows.entry(r.kind).or_default(), &mut total] {
                row.tasks += 1;
                match r.outcome {
                    TaskOutcome::Cached => row.cached += 1,
                    TaskOutcome::Suppressed(_) => row.suppressed += 1,
                    TaskOutcome::Deferred => row.deferred += 1,
                    TaskOutcome::Called | TaskOutcome::Failed(_) => {
                        row.to_call += 1;
                        row.est_in += r.est_input;
                        row.est_out += r.est_output;
                    }
                    TaskOutcome::NotCached | TaskOutcome::Skipped => {}
                }
            }
        }
        let mut out = String::new();
        let _ = writeln!(
            out,
            "{:<14}{:>7}{:>8}{:>9}{:>12}{:>10}{:>10}{:>10}",
            "kind", "tasks", "cached", "to-call", "suppressed", "deferred", "est.in", "est.out"
        );
        let mut line = |name: &str, r: &Row| {
            let _ = writeln!(
                out,
                "{:<14}{:>7}{:>8}{:>9}{:>12}{:>10}{:>10}{:>10}",
                name,
                r.tasks,
                r.cached,
                r.to_call,
                r.suppressed,
                r.deferred,
                human_tokens(r.est_in),
                human_tokens(r.est_out)
            );
        };
        for (kind, r) in &rows {
            line(kind, r);
        }
        line("total", &total);
        if let Some(model) = &self.model {
            let _ = writeln!(out, "model: {model}");
        }
        out
    }

    /// One-line run summary for stderr.
    pub fn summary(&self) -> String {
        let s = &self.stats;
        let mut line = format!(
            "LLM: {} cached, {} called, {} suppressed",
            s.cached, s.called, s.suppressed
        );
        for (n, label) in [
            (s.deferred, "deferred"),
            (s.not_cached, "not cached"),
            (s.failed, "failed"),
        ] {
            if n > 0 {
                let _ = write!(line, ", {n} {label}");
            }
        }
        if s.calls_made > 0 {
            let _ = write!(
                line,
                "; tokens in {} (cached {}) / out {}",
                human_tokens(s.input_tokens),
                human_tokens(s.cached_input_tokens),
                human_tokens(s.output_tokens)
            );
        }
        if s.retries > 0 {
            let _ = write!(line, "; {} retries", s.retries);
        }
        if s.pruned > 0 {
            let _ = write!(line, "; pruned {} cache entries", s.pruned);
        }
        let _ = write!(line, " ({:.1}s)", s.elapsed.as_secs_f64());
        if let Some(reason) = &self.degraded {
            let _ = write!(line, "\nLLM output disabled for this run: {reason}");
        }
        line
    }
}

fn human_tokens(n: u64) -> String {
    if n >= 10_000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

/// Tasks that share one cache key and therefore one call.
struct Job {
    key: String,
    ids: Vec<String>,
    task: WriteTask,
}

struct Shared {
    sem: tokio::sync::Semaphore,
    permits: AtomicUsize,
    consecutive_failures: AtomicUsize,
    stop: Mutex<Option<String>>,
    tokens_used: AtomicU64,
    retries: AtomicUsize,
    calls: AtomicUsize,
}

impl Shared {
    fn stopped(&self) -> bool {
        self.stop.lock().unwrap().is_some()
    }

    fn stop_with(&self, reason: String) {
        let mut stop = self.stop.lock().unwrap();
        if stop.is_none() {
            *stop = Some(reason);
        }
    }

    /// After a 429, halve the number of concurrent calls (floor 1).
    fn halve_concurrency(&self) {
        let current = self.permits.load(Ordering::SeqCst);
        if current > 1 {
            let forgotten = self.sem.forget_permits(current / 2);
            self.permits.fetch_sub(forgotten, Ordering::SeqCst);
            if forgotten > 0 {
                log::info!(
                    "pulse write: rate limited; concurrency {current} → {}",
                    current - forgotten
                );
            }
        }
    }
}

enum CallError {
    Fatal(String),
    Failed(String),
}

type CallResult = Result<(String, Option<Usage>), CallError>;

/// Run the writing pass.
pub fn run_tasks(
    mut tasks: Vec<WriteTask>,
    llm: &LlmHandle,
    cache: &WriteCache,
    cfg: &WriteConfig,
) -> WriteOutcome {
    let start = Instant::now();
    tasks.sort_by(|a, b| a.id.cmp(&b.id));
    tasks.dedup_by(|a, b| a.id == b.id);

    let mut outcome = WriteOutcome {
        mode: cfg.mode,
        results: BTreeMap::new(),
        degraded: None,
        truncated: false,
        dry_run: cfg.dry_run,
        model: llm
            .ident
            .as_ref()
            .map(|i| format!("{}/{}", i.provider, i.model)),
        stats: RunStats::default(),
    };
    let result = |task: &WriteTask, outcome: TaskOutcome| TaskResult {
        kind: task.kind,
        outcome,
        text: None,
        usage: None,
        est_input: task.est_input_tokens(),
        est_output: task.est_output_tokens(),
    };

    if cfg.mode == LlmMode::Off {
        for t in &tasks {
            outcome
                .results
                .insert(t.id.clone(), result(t, TaskOutcome::Skipped));
        }
        return finish(outcome, start);
    }

    let Some(ident) = llm.ident.clone() else {
        let reason = llm
            .unavailable
            .clone()
            .unwrap_or_else(|| "no LLM provider configured".into());
        for t in &tasks {
            outcome
                .results
                .insert(t.id.clone(), result(t, TaskOutcome::NotCached));
        }
        if cfg.mode == LlmMode::On {
            outcome.degraded = Some(reason);
        }
        return finish(outcome, start);
    };

    // ── 1. Plan ─────────────────────────────────────────────────────────────
    let mut jobs: BTreeMap<String, Job> = BTreeMap::new();
    let mut used_keys: Vec<String> = Vec::new();
    for task in &tasks {
        let words = task.user.split_whitespace().count();
        if words < task.min_words {
            outcome.results.insert(
                task.id.clone(),
                result(
                    task,
                    TaskOutcome::Suppressed(format!("{words} words of input")),
                ),
            );
            continue;
        }
        let schema_json = match &task.output {
            OutputSpec::JsonSchema { schema, .. } => Some(schema.to_string()),
            _ => None,
        };
        let agnostic = cfg.cache_model_agnostic;
        let key = KeyMaterial {
            task_kind: task.kind,
            prompt_version: task.prompt_version,
            provider: (!agnostic).then_some(ident.provider.as_str()),
            model: (!agnostic).then_some(ident.model.as_str()),
            max_tokens: task.max_tokens,
            output_mode: task.output.as_mode().label(),
            output_schema: schema_json.as_deref(),
            system: &task.system,
            user: &task.user,
        }
        .key();

        if !cfg.force.matches(task)
            && let Some(hit) = cache.get(&key)
        {
            let mut r = result(task, TaskOutcome::Cached);
            r.text = Some(hit.text);
            outcome.results.insert(task.id.clone(), r);
            used_keys.push(key);
            continue;
        }
        if cfg.mode == LlmMode::CacheOnly {
            outcome
                .results
                .insert(task.id.clone(), result(task, TaskOutcome::NotCached));
            continue;
        }
        outcome
            .results
            .insert(task.id.clone(), result(task, TaskOutcome::Called));
        jobs.entry(key.clone())
            .or_insert_with(|| Job {
                key,
                ids: Vec::new(),
                task: task.clone(),
            })
            .ids
            .push(task.id.clone());
    }

    // ── 2. Budget ───────────────────────────────────────────────────────────
    let mut jobs: Vec<Job> = jobs.into_values().collect();
    jobs.sort_by(|a, b| (a.task.priority, &a.task.id).cmp(&(b.task.priority, &b.task.id)));
    if let Some(cap) = cfg.max_llm_tokens {
        let mut spent = 0u64;
        let mut kept = Vec::with_capacity(jobs.len());
        for job in jobs {
            let cost = job.task.est_input_tokens() + job.task.est_output_tokens();
            if spent + cost <= cap {
                spent += cost;
                kept.push(job);
            } else {
                outcome.truncated = true;
                for id in &job.ids {
                    if let Some(r) = outcome.results.get_mut(id) {
                        r.outcome = TaskOutcome::Deferred;
                    }
                }
            }
        }
        jobs = kept;
    }

    // ── 3. Dry run ──────────────────────────────────────────────────────────
    if cfg.dry_run || jobs.is_empty() {
        return verdict(outcome, used_keys, cache, cfg, start);
    }

    let Some(provider) = llm.provider.clone() else {
        outcome.degraded = Some(
            llm.unavailable
                .clone()
                .unwrap_or_else(|| "LLM provider unavailable".into()),
        );
        for job in &jobs {
            for id in &job.ids {
                if let Some(r) = outcome.results.get_mut(id) {
                    r.outcome = TaskOutcome::Failed("provider unavailable".into());
                }
            }
        }
        return finish(outcome, start);
    };

    // ── 4–5. Probe, then fan out ────────────────────────────────────────────
    let permits = if cfg.concurrency == 0 {
        jobs.len()
    } else {
        cfg.concurrency
    }
    .max(1);
    let shared = Arc::new(Shared {
        sem: tokio::sync::Semaphore::new(permits),
        permits: AtomicUsize::new(permits),
        consecutive_failures: AtomicUsize::new(0),
        stop: Mutex::new(None),
        tokens_used: AtomicU64::new(0),
        retries: AtomicUsize::new(0),
        calls: AtomicUsize::new(0),
    });
    eprintln!(
        "  Writing: {} LLM call(s), {} concurrent ({})",
        jobs.len(),
        permits,
        outcome.model.as_deref().unwrap_or("?")
    );

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            outcome.degraded = Some(format!("could not start async runtime: {e}"));
            return finish(outcome, start);
        }
    };

    let cfg_arc = Arc::new(cfg.clone());
    let cache_arc = Arc::new(cache.clone());
    let finished: Vec<(Job, CallResult)> = runtime.block_on(async {
        let mut done = Vec::with_capacity(jobs.len());
        let mut rest = jobs.into_iter();
        let probe = rest.next().expect("jobs is non-empty");
        let probe_result =
            execute(Arc::clone(&provider), &probe, &cfg_arc, &shared, &cache_arc).await;
        done.push((probe, probe_result));

        let mut set = tokio::task::JoinSet::new();
        for job in rest {
            let provider = Arc::clone(&provider);
            let cfg = Arc::clone(&cfg_arc);
            let shared = Arc::clone(&shared);
            let cache = Arc::clone(&cache_arc);
            set.spawn(async move {
                let permit = shared.sem.acquire().await;
                if permit.is_err() || shared.stopped() {
                    return (job, None);
                }
                if let Some(cap) = cfg.max_llm_tokens
                    && shared.tokens_used.load(Ordering::SeqCst) >= cap
                {
                    return (job, None);
                }
                let r = execute(provider, &job, &cfg, &shared, &cache).await;
                (job, Some(r))
            });
        }
        while let Some(joined) = set.join_next().await {
            match joined {
                Ok((job, Some(r))) => done.push((job, r)),
                Ok((job, None)) => {
                    for id in &job.ids {
                        if let Some(r) = outcome.results.get_mut(id) {
                            r.outcome = TaskOutcome::Deferred;
                        }
                    }
                }
                Err(e) => log::warn!("pulse write task panicked: {e}"),
            }
        }
        done
    });

    outcome.stats.retries = shared.retries.load(Ordering::SeqCst);
    outcome.stats.calls_made = shared.calls.load(Ordering::SeqCst);
    for (job, r) in finished {
        match r {
            Ok((text, usage)) => {
                used_keys.push(job.key.clone());
                for id in &job.ids {
                    if let Some(res) = outcome.results.get_mut(id) {
                        res.outcome = TaskOutcome::Called;
                        res.text = Some(text.clone());
                        res.usage = usage;
                    }
                }
                if let Some(u) = usage {
                    outcome.stats.input_tokens += u.input_tokens as u64;
                    outcome.stats.output_tokens += u.output_tokens as u64;
                    outcome.stats.cached_input_tokens += u.cached_input_tokens as u64;
                }
            }
            Err(CallError::Fatal(msg)) | Err(CallError::Failed(msg)) => {
                for id in &job.ids {
                    if let Some(res) = outcome.results.get_mut(id) {
                        res.outcome = TaskOutcome::Failed(msg.clone());
                    }
                }
            }
        }
    }
    if let Some(reason) = shared.stop.lock().unwrap().take() {
        outcome.degraded = Some(reason);
    }
    verdict(outcome, used_keys, cache, cfg, start)
}

/// Call one job with retries; write the cache on success.
async fn execute(
    provider: Arc<dyn LlmProvider>,
    job: &Job,
    cfg: &WriteConfig,
    shared: &Shared,
    cache: &WriteCache,
) -> CallResult {
    let task = &job.task;
    let started = Instant::now();
    let mut max_tokens = task.max_tokens;
    let mut grown = false;
    let mut attempt = 0u32;
    loop {
        if shared.stopped() {
            return Err(CallError::Failed("run stopped".into()));
        }
        attempt += 1;
        let req = CompletionRequest {
            system: &task.system,
            user: &task.user,
            output: task.output.as_mode(),
            max_tokens,
            temperature: 0.0,
            tag: &task.id,
        };
        shared.calls.fetch_add(1, Ordering::SeqCst);
        match provider.complete_request(&req).await {
            Ok(resp) => {
                if let Some(u) = resp.usage {
                    shared.tokens_used.fetch_add(
                        u.input_tokens as u64 + u.output_tokens as u64,
                        Ordering::SeqCst,
                    );
                }
                // Truncated JSON never parses: retry once with room to finish.
                if resp.stop == StopReason::MaxTokens && task.output != OutputSpec::Text && !grown {
                    grown = true;
                    max_tokens = (max_tokens as f32 * 1.5) as u32;
                    shared.retries.fetch_add(1, Ordering::SeqCst);
                    continue;
                }
                if let StopReason::Other(reason) = &resp.stop
                    && reason == "refusal"
                {
                    return Err(fail(shared, cfg, format!("{}: model refused", task.id)));
                }
                shared.consecutive_failures.store(0, Ordering::SeqCst);
                let entry = CacheEntry {
                    key: job.key.clone(),
                    task_id: task.id.clone(),
                    task_kind: task.kind.to_string(),
                    model: resp.model.clone(),
                    usage: resp.usage,
                    text: resp.text.trim().to_string(),
                };
                if let Err(e) = cache.put(&entry) {
                    log::warn!("pulse write cache: {e:#}");
                }
                eprintln!(
                    "  Writing: {} (done, {:.1}s)",
                    task.id,
                    started.elapsed().as_secs_f64()
                );
                return Ok((entry.text, resp.usage));
            }
            Err(e) => {
                let classified = e.downcast_ref::<ProviderError>();
                let kind = classified.map(|p| p.kind);
                if kind.is_some_and(|k| k.is_fatal()) {
                    let reason = format!("{e}");
                    shared.stop_with(reason.clone());
                    return Err(CallError::Fatal(reason));
                }
                // Unclassified errors come from legacy `complete()` paths: retry them.
                let retryable = kind.is_none_or(|k| k.is_retryable());
                if retryable && attempt < cfg.max_attempts {
                    if kind == Some(ProviderErrorKind::RateLimited) {
                        shared.halve_concurrency();
                    }
                    let delay = classified
                        .and_then(|p| p.retry_after)
                        .unwrap_or_else(|| backoff(cfg.base_backoff, attempt, &task.id));
                    log::debug!(
                        "pulse write: {} attempt {attempt} failed: {e}; retry in {delay:?}",
                        task.id
                    );
                    shared.retries.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(delay).await;
                    continue;
                }
                eprintln!(
                    "  Writing: {} (failed, {:.1}s): {e}",
                    task.id,
                    started.elapsed().as_secs_f64()
                );
                return Err(fail(shared, cfg, format!("{e}")));
            }
        }
    }
}

fn fail(shared: &Shared, cfg: &WriteConfig, msg: String) -> CallError {
    let n = shared.consecutive_failures.fetch_add(1, Ordering::SeqCst) + 1;
    if n >= cfg.breaker_threshold {
        shared.stop_with(format!("{n} consecutive LLM failures; last: {msg}"));
    }
    CallError::Failed(msg)
}

/// `min(30s, base · 2^(attempt-1))` plus up to 25% jitter derived from the task id,
/// so reruns wait the same way.
fn backoff(base: Duration, attempt: u32, id: &str) -> Duration {
    let exp = base.saturating_mul(1u32 << (attempt - 1).min(5));
    let capped = exp.min(Duration::from_secs(30));
    let h = blake3::hash(id.as_bytes()).as_bytes()[0] as u32;
    capped + capped.mul_f64(h as f64 / 255.0 * 0.25)
}

fn verdict(
    mut outcome: WriteOutcome,
    used_keys: Vec<String>,
    cache: &WriteCache,
    cfg: &WriteConfig,
    start: Instant,
) -> WriteOutcome {
    if outcome.dry_run {
        return finish(outcome, start);
    }
    let mut ok = 0usize;
    let mut eligible = 0usize;
    for r in outcome.results.values() {
        match r.outcome {
            TaskOutcome::Cached | TaskOutcome::Called => {
                ok += 1;
                eligible += 1;
            }
            TaskOutcome::Failed(_) => eligible += 1,
            _ => {}
        }
    }
    if outcome.degraded.is_none()
        && cfg.mode == LlmMode::On
        && eligible > 0
        && (ok as f32) / (eligible as f32) < cfg.degrade_threshold
    {
        outcome.degraded = Some(format!(
            "only {ok} of {eligible} LLM sections succeeded (need {:.0}%)",
            cfg.degrade_threshold * 100.0
        ));
    }
    let complete = outcome.degraded.is_none() && !outcome.truncated;
    if complete && cfg.prune && cfg.mode == LlmMode::On && eligible > 0 {
        match cache.record_run_and_prune(used_keys, cfg.keep_runs) {
            Ok(n) => outcome.stats.pruned = n,
            Err(e) => log::warn!("pulse write cache prune failed: {e:#}"),
        }
    }
    finish(outcome, start)
}

fn finish(mut outcome: WriteOutcome, start: Instant) -> WriteOutcome {
    let s = &mut outcome.stats;
    for r in outcome.results.values() {
        match r.outcome {
            TaskOutcome::Cached => s.cached += 1,
            TaskOutcome::Called => s.called += 1,
            TaskOutcome::Suppressed(_) => s.suppressed += 1,
            TaskOutcome::Deferred => s.deferred += 1,
            TaskOutcome::NotCached => s.not_cached += 1,
            TaskOutcome::Failed(_) => s.failed += 1,
            TaskOutcome::Skipped => s.skipped += 1,
        }
    }
    if outcome.dry_run {
        // Nothing was called; "Called" in the plan means "would call".
        s.called = 0;
    }
    s.elapsed = start.elapsed();
    outcome
}
