# Reflex Efficacy Benchmark — Phase 2

A/B runner harness for the Reflex AI coding efficacy study ([REF-176](/REF/issues/REF-176)).

## What this does

Runs **arms × tasks × N trials**, invoking Claude Code headless per the Phase 0 recipe.
Each trial captures a full NDJSON transcript under `results/`. The metrics extractor
then parses every transcript into a single tidy CSV for analysis.

## Quick start

```bash
# From the repo root:

# 1. Build rfx (required for treatment arms B, C, B')
cargo build --release

# 2. (Optional) Build the Reflex index for treatment arms
./target/release/rfx index

# 3. Dry-run to verify commands
python3 benches/efficacy/runner.py --dry-run

# 4. Run the full matrix (5 trials per arm × task, ~20 tasks × 4 arms = 100 trials)
python3 benches/efficacy/runner.py

# 5. Extract metrics to CSV
python3 benches/efficacy/extract_metrics.py

# Results are in benches/efficacy/results/metrics.csv
```

## ⚠️ Running under an automated agent — do NOT exhaust its context

The runner streams every transcript **straight to disk** (`stdout=f`) and holds
nothing large in memory, so `runner.py` itself is context-safe. The failure mode
that stalled Phase 4 ([REF-181](/REF/issues/REF-181)) was operational, not a harness
bug: an **agent ran the trial loop inside its own session** — streaming subprocess
output and re-reading the growing transcript pile turn after turn — until the
model's context window was exhausted. That killed runs mid-write and corrupted a
few NDJSON transcripts.

**Rules for any automated agent driving this harness:**

1. **Launch detached — never run the full matrix synchronously in your context:**
   ```bash
   bash benches/efficacy/run-detached.sh --arms A B C --n 5
   ```
   This `setsid nohup`s the runner so it survives your heartbeat ending; output
   goes to `results/phase4-run.log`, never into your conversation.
2. **Poll only tiny derived signals**, not raw transcripts. Use the watcher
   (auto-commits + posts status when the runner exits):
   ```bash
   bash benches/efficacy/phase4-completion-watcher.sh <runner-pid> <issue-id>
   ```
3. **Finalize disk→disk.** Run the extractor to produce the CSV and inspect only
   the small CSV summary — never `cat`/`Read` the `*.ndjson` transcripts into a turn:
   ```bash
   python3 benches/efficacy/extract_metrics.py --out results/metrics.csv
   ```
4. The extractor is **resilient to individual malformed lines** (a stray non-UTF8
   byte or one bad stream-json line no longer discards a whole trial); any skipped
   lines are counted per-trial in the `parse_warnings` column so degraded trials
   stay visible instead of being silently trusted or silently dropped.

## Arms

| Arm    | MCP     | CLAUDE.md nudge | Grep/Glob | Description |
|--------|---------|-----------------|-----------|-------------|
| **A**  | ❌ None | ✅ active       | ✅ allowed | Control: built-ins only |
| **B**  | ✅ Reflex | ✅ active     | ✅ allowed | Realistic: Reflex MCP + nudge |
| **C**  | ✅ Reflex | ✅ active     | ❌ blocked | Reflex-forced |
| **B'** | ✅ Reflex | ❌ neutralized | ✅ allowed | No-nudge (secondary) |
| **Beager** | ✅ Reflex, `alwaysLoad` | ✅ active | ✅ allowed | Arm B with the schemas loaded at session start |
| **S**  | ❌ None | hook | ✅ allowed | CLI route: the `reflex` plugin (skill + SessionStart hook) by `--plugin-dir`, the built `rfx` first on PATH |
| **L**  | ❌ None | — | ✅ allowed | LSP: Claude Code's `LSP` tool with rust-analyzer (`plugins/lsp-rust`); no Reflex |

> **Arm S:** slash commands stay on (the skill needs them), so Claude Code's ~20
> bundled skills are listed too; deny rules do not hide them. The tree is indexed before
> the trial, as in the MCP arms. Bash calls that run `rfx` count as Reflex calls.
>
> **Arm L:** `plugins/lsp-rust` carries the `lspServers` entry of the official
> `rust-analyzer-lsp` plugin. Needs `rust-analyzer` on PATH and every corpus's crates in
> the Cargo cache: run `cargo fetch` on a copy of each corpus first (not in
> `corpus/`: a `Cargo.lock` written there would change the grep answer keys).
> rust-analyzer starts with each session and indexes the fresh copy; that warm-up is
> wall time, not tokens. Its time per corpus is in `.context/AGENT_EDGE_RESEARCH.md`.

> **Note on arm A:** `--strict-mcp-config` is passed so the project `.mcp.json`
> (which configures Reflex MCP) cannot load — the control arm truly has no MCP.

## Runner options

```
--arms A B C Bprime    Arms to run (default: all)
--tasks T01 T02        Task IDs from tasks.json (default: all)
--n 5                  Replicate trials per arm × task (default: 5)
--model <id>           Claude model (default: claude-haiku-4-5-20251001)
--skip-build           Skip cargo build --release
--baselines-only       Run only the MCP context-tax baseline (empty task)
--overwrite            Re-run trials that already have complete transcripts
--dry-run              Print commands without executing
```

## Transcript structure

Each trial writes to `results/{arm}/{task_id}/trial_{NN}.ndjson`.

The first line is a harness metadata record:
```json
{"type": "harness_metadata", "arm": "B", "task_id": "T01_symbol_usages",
 "trial": 1, "model": "claude-haiku-4-5-20251001", "repo_sha": "abc1234",
 "started_at": "2026-07-02T16:00:00Z"}
```

Subsequent lines are the Claude Code streaming JSON events (system init,
assistant messages with tool_use, user messages with tool_result, and the
terminal `result` event with full token usage).

## Metrics extractor

```bash
python3 benches/efficacy/extract_metrics.py [--results-dir results/] [--out metrics.csv]
```

Output CSV columns (one row per trial):

| Column | Description |
|--------|-------------|
| `arm` | Arm identifier (A/B/C/Bprime) |
| `task_id` | Task from tasks.json |
| `trial` | Trial number (1..N) |
| `model` | Model ID (pinned per run) |
| `repo_sha` | Git SHA of repo at run time |
| `wall_ms` | Wall-clock duration in ms |
| `input_tokens` | Total input tokens |
| `output_tokens` | Total output tokens |
| `cache_read_tokens` | Prompt cache hits |
| `cache_creation_tokens` | Prompt cache writes |
| `total_cost_usd` | Estimated cost |
| `assistant_turns` | Number of model inference turns |
| `total_tool_calls` | All tool invocations |
| `search_tool_calls_builtin` | Grep + Glob calls |
| `search_tool_calls_mcp` | Reflex MCP search calls |
| `read_tool_calls` | Read tool invocations |
| `bytes_read` | Total bytes of file content Read into context |
| `lines_read` | Total lines Read into context |
| `mcp_servers_active` | Pipe-separated active MCP servers from init |
| `grep_glob_blocked` | True if Grep/Glob were blocked (arm C) |
| `reflex_tools_used` | Pipe-separated Reflex MCP tools actually called |
| `success` | True if transcript has a non-error result |
| `stop_reason` | Terminal stop reason |
| `parse_warnings` | Count of malformed NDJSON lines skipped for this trial (>0 ⇒ secondary metrics may be undercounted; token/usage from the result event is still authoritative) |
| `transcript_path` | Path for manual audit |

## Cold vs warm index ledger

The index build cost is measured separately:

```bash
# Measure cold index build time
time ./target/release/rfx index --force

# Warm re-index (incremental, no changes)
time ./target/release/rfx index
```

Record these values alongside the A/B metrics for the confound baseline.

## Statistical analysis + plots (Phase 3)

`analyze.py` consumes the Phase 2 tidy table (`metrics.csv`) and emits a stats
summary (JSON + Markdown) plus per-hypothesis SVG plots. It is **stdlib-only**
(bootstrap CIs and the Wilcoxon signed-rank test are hand-implemented in
`stats.py`; plots are hand-rendered SVG in `plots.py`) and fully deterministic —
the bootstrap uses a fixed seed, so the same inputs always yield the same
numbers and figures.

```bash
# Real run (once Phase 4 has produced results/metrics.csv):
python3 benches/efficacy/analyze.py

# With the optional accuracy/quality table (H2/H3) and cold-index ledger:
python3 benches/efficacy/analyze.py \
    --accuracy results/accuracy.csv \
    --index-ledger results/index_ledger.json

# Offline smoke test (asserts the whole stats + decision pipeline, no data needed):
python3 benches/efficacy/analyze.py --self-test

# End-to-end smoke on the committed Phase-0-grounded sample fixture:
python3 benches/efficacy/make_sample_data.py     # regenerate results/sample/*
python3 benches/efficacy/analyze.py \
    --metrics  results/sample/metrics.csv \
    --accuracy results/sample/accuracy.csv \
    --index-ledger results/sample/index_ledger.json
```

Outputs land in `results/analysis/summary.{json,md}` and `results/plots/*.svg`.

### Pre-registered primary endpoint (fixed in code — do not p-fish)

The pre-registration lives as executable constants at the top of `analyze.py`
(`PRIMARY`, `DECISION`) and mirrors the CEO-locked [REF-176 plan](/REF/issues/REF-176#document-plan).
It is fixed **before** any results are viewed:

- **Endpoint:** median over tasks of the per-task token ratio **B / A**
  (`total_tokens` = input + output + cache_read + cache_creation — cache is
  included on purpose so the MCP context tax is visible), on **find-all-usages**
  tasks, reported **cold + warm**, each with a paired bootstrap 95% CI.
- **Verdict is on the warm condition** (a multi-query workflow amortizes the
  one-time index build); cold is an honesty check, never the headline.

**Decision rule** (`r` = warm median ratio, `[lo, hi]` = 95% CI):

| Verdict | Condition |
|---------|-----------|
| ✅ Reflex better | `r < 0.90` **and** `hi < 1.0` |
| ❌ Reflex worse | `r > 1.10` **and** `lo > 1.0` |
| ➖ No difference | CI straddles `1.0` |
| ❔ Indeterminate | effect excludes parity but misses the ±10% thresholds |

Everything else — arms C and B′, other metrics, H2 precision/recall/hallucination,
H3 success-rate + quality rubric — is **secondary/exploratory** and clearly
labelled as such; it may never be substituted into the primary claim.

### Analysis unit and methodology

- The **task** is the unit of analysis: replicate trials collapse to a per-task
  median, then tasks are the exchangeable units for the ratio, the paired
  bootstrap CI, and the Wilcoxon signed-rank test. With the thin slice's tiny
  task counts the test has limited power — `n`, the exact/approx flag, and the
  CI width are all surfaced so the evidence can be read honestly.
- Ratios use **all trials** (successes and failures): a run that burned tokens
  without reaching an answer is a real cost, and dropping it would be
  success-selection bias. A successful-only sensitivity can be derived from the
  same table.
- **Cold vs warm tokens:** agent token totals exclude the `rfx index` build
  (that runs outside the agent token budget), so cold and warm token ratios are
  equal unless the agent itself calls `index_project`; the cold penalty is a
  wall-time/compute cost tracked in the index ledger above.

### Accuracy/quality table schema (H2/H3, optional)

H2 precision/recall/hallucination and the H3 quality rubric need the agent's
returned locations scored against the oracle ground truth (Phase 4/5 answer
scoring). `analyze.py` reads this as an optional CSV keyed by
`(arm, task_id, trial)`:

| Column | Meaning |
|--------|---------|
| `n_expected` | ground-truth match count (from the oracle) |
| `n_returned` | locations the agent claimed |
| `n_correct` | true positives |
| `precision` / `recall` / `hallucination_rate` | derived from the counts if absent |
| `quality_score` | 0–4 answer-quality rubric (comprehension tasks) |

When the table is absent, H2 degrades to a clearly-labelled "pending answer
scoring" note and H3 still reports task-success rate from the metrics table.

## File layout

```
benches/efficacy/
  runner.py              # Main runner harness
  extract_metrics.py     # NDJSON → CSV extractor (Phase 2)
  analyze.py             # Stats + plots + pre-registered decision rule (Phase 3)
  stats.py               # Stdlib bootstrap CI + Wilcoxon signed-rank
  plots.py               # Stdlib SVG figures
  make_sample_data.py    # Deterministic Phase-0-grounded smoke fixture
  tasks.json             # Task definitions
  configs/
    arm-a-no-mcp.json    # Arm A: empty MCP config
    arm-b-reflex-mcp.json # Arm B/C/B': Reflex MCP template
  results/
    A/ B/ C/ Bprime/     # {arm}/{task_id}/trial_NN.ndjson transcripts
    _baseline/            # MCP context-tax baseline trials
    metrics.csv           # Extracted tidy table (Phase 2)
    sample/               # Committed smoke fixture (metrics + accuracy + ledger)
    analysis/             # analyze.py output: summary.{json,md}
    plots/                # analyze.py output: per-hypothesis SVG figures
    run-manifest-*.json   # Run provenance records
```


```
benches/efficacy/
  runner.py              # Main runner harness
  extract_metrics.py     # NDJSON → CSV extractor
  tasks.json             # Task definitions
  configs/
    arm-a-no-mcp.json    # Arm A: empty MCP config
    arm-b-reflex-mcp.json # Arm B/C/B': Reflex MCP template
  results/
    A/
      T01_symbol_usages/
        trial_01.ndjson
        trial_02.ndjson
        ...
    B/ C/ Bprime/
    _baseline/            # MCP context-tax baseline trials
    metrics.csv           # Extracted tidy table
    run-manifest-*.json   # Run provenance records
```
