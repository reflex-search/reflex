# Efficacy A/B on Reflex 2.0.3 (2026-09-28)

Reflex MCP (arm B) against Claude Code's built-in Grep/Glob (arm A), same tasks, same
model, paired per task. Replaces every earlier efficacy number (REF-192/204/209/217/222/225,
all measured on 1.5.3-dev in 2026-07), which were withdrawn.

## Setup

| | |
| --- | --- |
| Reflex | 2.0.3, build `0ded14d` (`reflex-mcp startup:` line), columnar on, 17 tools |
| Models | **`claude-opus-5-5`** (primary, both arms); `claude-sonnet-5` (second run of section 1) |
| Harness | `benches/efficacy/` at `5f68fff` (Sonnet 5 run: `d0c7115`) |
| Corpora | pinned: reflex `d2935f4` (clone in `corpus/reflex`), ripgrep `4649aa9`, tokio `ab3ff69` — see `repos.md` |
| Isolation | `--setting-sources ""` (no user/project settings, no CLAUDE.md), `--disable-slash-commands`, `--strict-mcp-config`, `--disallowedTools Agent,Task` (Opus runs) |
| Arm A tools | Bash, Edit, Glob, Grep, LS, MultiEdit, Read, TodoWrite, Write |
| Arm B tools | the same, plus the 17 `mcp__reflex__*` tools |

Tool loading: Claude Code defers the Reflex schemas behind ToolSearch even with
`--allowedTools`, so arm B pays one ToolSearch round-trip before its first Reflex call.
This is the default an agent gets today, so it was kept.

Harness defects found and fixed during this run (all committed): the control arm had no
Grep/Glob unless `--allowedTools` named them; the pinned reflex corpus's CLAUDE.md sent the
control arm hunting for `mcp__reflex__*` tools (the REF-190 confound); the reflex corpus was
the live checkout; agents could delegate to a subagent, whose turns and tokens the metrics
do not see (27/128 REF-225 control trials on Sonnet 5 — that partial run was discarded).

## 1. Find-all-usages (REF-222 design) — using Reflex costs one extra round-trip

9 tasks (3 reflex, 3 ripgrep, 3 tokio) × 8 trials × 2 arms = 144 trials per model.
Pre-registered primary endpoint: median over tasks of the per-task median `total_tokens`
ratio B/A, bootstrap 95% CI (`analyze.py`). Cost: $6.11 (Opus 5.5), $6.70 (Sonnet 5).

| Endpoint (B/A) | Opus 5.5 | Sonnet 5 |
| --- | --- | --- |
| **total_tokens (primary)** | **1.663** [1.045, 1.685] → **reflex_worse** | **1.646** [1.063, 2.249] → **reflex_worse** |
| Wilcoxon (9 pairs) | p = 0.004, all 9 tasks > 1 | p = 0.004, all 9 tasks > 1 |
| total_cost_usd | 1.271 [1.048, 1.351] | 2.072 [1.331, 2.538] |
| assistant_turns | 1.5 [1.0, 1.5] (median 2 vs 3) | 1.5 [1.0, 2.0] (median 2 vs 4) |
| total_tool_calls | 2.0 [1.0, 2.0] | 2.0 [1.0, 3.0] |
| output_tokens | 1.131 [1.037, 1.278] | 1.097 [1.001, 1.664] |
| wall_ms | 1.178 [1.026, 1.378] | 1.114 [0.905, 1.42] |
| arm B trials that used Reflex | **37/72** | 67/72 |

**Why — it is the ToolSearch round-trip.** Opus 5.5 split cleanly:

| Arm B trials (Opus 5.5) | n | tokens vs control (median) | median turns |
| --- | --- | --- | --- |
| used Reflex (always after a ToolSearch) | 37 | **1.68×** | 3 |
| did not use Reflex (used Grep) | 35 | 1.04× | 2 |

A single Grep answers these tasks in one call (control: 2 turns in every trial). Using Reflex
adds a ToolSearch call to load the deferred schemas, which costs a whole turn, and each turn
re-reads the cached context. Sonnet 5 also called `check_index_status` in 33/72 trials
(Opus: 0/72), adding a further turn. The payload itself is not the cost.

Per task, Opus 5.5 (medians over 8 trials; precision / recall mean over 8 trials):

| Task | tokens B/A | turns A | turns B | cost A $ | cost B $ | A prec / rec | B prec / rec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reflex-findall-extract_symbols (122 sites) | 1.03 | 2 | 2 | 0.076 | 0.075 | 1.000 / 0.062 | 1.000 / 0.115 |
| ripgrep-findall-sinkmatch | 1.05 | 2 | 2 | 0.037 | 0.039 | 1.000 / 1.000 | 1.000 / 1.0* |
| ripgrep-findall-sinkcontext | 1.05 | 2 | 2 | 0.030 | 0.031 | 1.000 / 1.0* | 1.000 / 1.0* |
| ripgrep-findall-mmapchoice | 1.06 | 2 | 2 | 0.029 | 0.033 | 1.000 / 1.000 | 0.891 / 1.000 |
| reflex-findall-trigramindex | 1.66 | 2 | 3 | 0.030 | 0.040 | 1.000 / 0.958 | 1.000 / 0.976 |
| reflex-findall-symbolcache | 1.67 | 2 | 3 | 0.024 | 0.031 | 1.000 / 0.938 | 1.000 / 1.000 |
| tokio-findall-joinerror | 1.68 | 2 | 3 | 0.038 | 0.048 | 1.000 / 1.0* | 1.000 / 1.0* |
| tokio-findall-notified | 1.68 | 2 | 3 | 0.027 | 0.035 | 1.000 / 1.000 | 1.000 / 1.000 |
| tokio-findall-barrier | 1.69 | 2 | 3 | 0.025 | 0.047 | 1.000 / 1.0* | 1.000 / 1.0* |

\* Recall above 1.0 (1.01–1.02) is a scorer artefact: `score_accuracy.py` counts true positives over claimed strings, and one location written two ways (`src/a.rs:5` and `a.rs:5`) matches the same oracle line twice via suffix matching. Read as 1.0.

Accuracy: with Opus 5.5 both arms are near-perfect on 8 of 9 tasks; neither lists the 122
`extract_symbols` sites (Reflex 0.115, Grep 0.062 recall). With Sonnet 5, Reflex was more
complete on 6 of 9 tasks (e.g. ripgrep `sinkcontext` recall 1.00 vs 0.63, `mmapchoice`
1.00 vs 0.81) — the weaker model gains more from exhaustive results.

Sonnet 5 per-task data: `benches/efficacy/results-2.0.3-sonnet5/` (gitignored), report
`REF-222-report.md` there.

## 2. Comprehension / transitive / cross-module (REF-225 design) — Reflex costs more, finds no more

Opus 5.5 only. 16 pre-registered iteration-forcing tasks × 8 trials × 2 arms = 256 trials,
$50.16. Arm-A validity gate (median `num_turns` ≥ 4): 15/16 pass on this run (only
`reflex-cm-language-dispatch` fails, 3.5). As pre-registered, the analysis uses the 13 tasks
frozen in 2026-07 (`VALID_TASK_IDS` in `ref225-phase2-analysis.py`; excludes
`ripgrep-comp-parallel`, `tokio-comp-task-abort`, `tokio-comp-io-driver`). Paired Wilcoxon,
Holm-Bonferroni over 3 endpoints, bootstrap 95% CI on the median ratio.

| Endpoint | A median | B median | B/A | 95% CI | p (adj.) |
| --- | --- | --- | --- | --- | --- |
| assistant_turns | 6 | 6 | 1.00 | [1.00, 1.20] | 0.0009 |
| total_tool_calls | 5 | 5 | 1.00 | [1.00, 1.25] | 0.0006 |
| **total_tokens** | 133,115 | 167,533 | **1.26** | [1.07, 1.30] | < 0.0001 |
| recall (guardrail) | 1.00 | 1.00 | — | — | pass |

The script labels this "mixed" because the median turn ratio is 1.00. The direction is not
mixed: arm B took more turns in 55 pairs and fewer in 27 (22 ties); mean 6.91 vs 6.33 turns.
Dollar cost over the 13 tasks: $20.83 (B) vs $18.86 (A), 1.10×.

Arm B used Reflex in only **48 of 128** trials (Reflex calls: `search_regex` 46,
`search_code` 14, `find_references` 14). Split by use, on the 13 tasks (observational — the
agent chose when to use Reflex, so the groups may differ in difficulty):

| Arm B trials | n | tokens vs control (median) | extra turns | recall |
| --- | --- | --- | --- | --- |
| used Reflex (always after a ToolSearch) | 41 | 1.55× | +2 | 0.943 |
| did not use Reflex | 63 | 1.07× | 0 | 0.988 |
| control (arm A) | 104 | 1.00× | — | 0.969 |

Per task (medians): the token ratio ranges from 0.91 (`reflex-comp-query-path`, 9 vs 10
turns) to 1.73 (`ripgrep-cm-stats-tracking`); 12 of 13 tasks are above 1.0.

Compared with 2026-07 (1.5.3-dev, Sonnet 4.6: turns 1.50×, tokens 1.59×), the turn penalty is
gone at the median and the token penalty is smaller — but the model changed too, so this
does not isolate the effect of the 1.7–2.0 changes.

## 3. Columnar vs legacy payload (per call, no model)

`benches/efficacy/columnar-payload.py` on the pinned reflex corpus, 10 fixed
`search_code` / `search_regex` calls, `content[0].text` bytes:

- median per-call saving **19.8%** (range 2.1% for a 1-row symbol result to 37.4%);
  **21.0%** pooled over all calls.
- Raw numbers: `benches/efficacy/results/columnar-payload.json`.

Session-level A/B runs cannot see this (turn count dominates `total_tokens`).

## What this says

- **Using** Reflex via MCP costs more than built-in Grep in every setting measured: about 1.7×
  tokens on single-shot find-all-usages (both models) and about 1.55× on comprehension tasks
  (Opus 5.5). With Reflex merely available, the averages are 1.66× (find-all) and 1.26×
  (comprehension), because the agent often ignores it.
- The cost is round-trips, not payload: one ToolSearch call to load the deferred Reflex
  schemas before every first use, plus (Sonnet 5) `check_index_status` calls. Columnar
  results already save ~20% per call, which the round-trips outweigh.
- Accuracy: no gain with Opus 5.5 (near-perfect either way; slightly lower recall on the
  comprehension trials that used Reflex). With Sonnet 5, Reflex answers were more complete on
  6 of 9 find-all tasks.
- Levers the data points at (not yet tried): remove the pre-search round-trips — the
  `check_index_status` description says "Call this at session start", but every
  `search_code` / `search_regex` response already carries `status` and
  `can_trust_results`; and avoid the deferred-schema round-trip (e.g. fewer, broader tools so
  the schemas are cheap to load eagerly).

## Limits

- Two models, three Rust repositories, 9 + 16 tasks, 8 trials per task.
- Arm B pays the ToolSearch round-trip because Claude Code defers MCP schemas; a client
  that loads MCP schemas eagerly would not.
- The "used Reflex / did not" splits are observational, not randomised.
- REF-225 ran on Opus 5.5 only; a Sonnet 5 attempt was discarded (subagent defect).
- The recall scorer can exceed 1.0 when one location is written two ways (see section 1).
