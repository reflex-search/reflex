# Efficacy A/B on Reflex 2.0.3 (2026-09-28)

Reflex MCP (arm B) against Claude Code's built-in Grep/Glob (arm A), same tasks, same
model, paired per task. Replaces every earlier efficacy number (REF-192/204/209/217/222/225,
all measured on 1.5.3-dev in 2026-07), which were withdrawn.

## Setup

| | |
| --- | --- |
| Reflex | 2.0.3, build `0ded14d` (`reflex-mcp startup:` line), columnar on, 17 tools |
| Model | `claude-sonnet-5`, both arms |
| Harness | `benches/efficacy/` at `d0c7115` |
| Corpora | pinned: reflex `d2935f4` (clone in `corpus/reflex`), ripgrep `4649aa9`, tokio `ab3ff69` — see `repos.md` |
| Isolation | `--setting-sources ""` (no user/project settings, no CLAUDE.md), `--disable-slash-commands`, `--strict-mcp-config` |
| Arm A tools | Bash, Edit, Glob, Grep, LS, MultiEdit, Read, TodoWrite, Write |
| Arm B tools | the same, plus the 17 `mcp__reflex__*` tools |

Tool loading: Claude Code defers the Reflex schemas behind ToolSearch even with
`--allowedTools`, so arm B pays one ToolSearch round-trip before its first Reflex call.
This is the default an agent gets today, so it was kept.

Harness changes made for this run (all committed): the control arm had no Grep/Glob at all
unless `--allowedTools` named them; the pinned reflex corpus's CLAUDE.md sent the control arm
hunting for `mcp__reflex__*` tools (the REF-190 confound), fixed by loading no settings
sources; the reflex corpus is the pinned clone, not the live checkout.

## 1. Find-all-usages (REF-222 design) — Reflex costs more, finds more

9 tasks (3 reflex, 3 ripgrep, 3 tokio) × 8 trials × 2 arms = 144 trials, $6.70 in total.
Pre-registered primary endpoint: median over tasks of the per-task median `total_tokens`
ratio B/A, bootstrap 95% CI (`analyze.py`).

| Endpoint | B/A | 95% CI |
| --- | --- | --- |
| **total_tokens (primary)** | **1.646** | [1.063, 2.249] → **reflex_worse** (Wilcoxon p = 0.004, all 9 tasks > 1) |
| total_cost_usd | 2.072 | [1.331, 2.538] |
| assistant_turns | 1.5 | [1.0, 2.0] — median 2 (A) vs 4 (B) |
| total_tool_calls | 2.0 | [1.0, 3.0] |
| output_tokens | 1.097 | [1.001, 1.664] |
| wall_ms | 1.114 | [0.905, 1.42] |

Per task (medians over 8 trials):

| Task | tokens B/A | turns A | turns B | cost A $ | cost B $ |
| --- | --- | --- | --- | --- | --- |
| tokio-findall-notified | 1.03 | 2 | 2 | 0.023 | 0.023 |
| reflex-findall-extract_symbols | 1.06 | 3 | 3 | 0.057 | 0.079 |
| ripgrep-findall-sinkcontext | 1.09 | 3 | 3 | 0.040 | 0.053 |
| tokio-findall-barrier | 1.43 | 3 | 4 | 0.039 | 0.057 |
| ripgrep-findall-mmapchoice | 1.65 | 2 | 3 | 0.024 | 0.049 |
| tokio-findall-joinerror | 2.16 | 2 | 4 | 0.026 | 0.056 |
| reflex-findall-symbolcache | 2.18 | 2 | 4 | 0.023 | 0.055 |
| ripgrep-findall-sinkmatch | 2.25 | 2 | 4 | 0.026 | 0.066 |
| reflex-findall-trigramindex | 2.75 | 2 | 5 | 0.023 | 0.072 |

**Why:** the cost tracks turns. At equal turns the ratio is 1.03–1.09; each extra turn
re-reads the cached context. Arm B used ToolSearch in 67/72 trials (loading the deferred
Reflex schemas) and `check_index_status` in 33/72, before or alongside the search itself.
A single Grep usually answers these tasks in one call.

Accuracy (graded against a ripgrep oracle; precision / recall, mean over 8 trials):

| Task | A | B |
| --- | --- | --- |
| reflex-findall-extract_symbols (122 expected) | 1.000 / 0.095 | 1.000 / 0.189 |
| reflex-findall-symbolcache | 0.846 / 1.000 | 0.981 / 1.021* |
| reflex-findall-trigramindex | 1.000 / 0.964 | 1.000 / 0.893 |
| ripgrep-findall-mmapchoice | 0.805 / 0.809 | 1.000 / 1.000 |
| ripgrep-findall-sinkcontext | 1.000 / 0.631 | 1.000 / 1.000 |
| ripgrep-findall-sinkmatch | 1.000 / 0.750 | 1.000 / 0.763 |
| tokio-findall-barrier | 0.840 / 0.769 | 1.000 / 1.019* |
| tokio-findall-joinerror | 1.000 / 0.890 | 1.000 / 1.000 |
| tokio-findall-notified | 1.000 / 1.000 | 1.000 / 1.000 |

\* Recall above 1.0 is a scorer artefact (a location reported twice counts twice); treat as 1.0.

Reflex is at least as precise on every task and more complete on 6 of 9 (clearly on
ripgrep `sinkcontext`, `mmapchoice`, tokio `barrier`, `joinerror`); it is less complete on
`trigramindex` (0.893 vs 0.964). Neither arm lists all 122 `extract_symbols` sites.

## 2. Comprehension / transitive / cross-module (REF-225 design)

_Pending — running._

## 3. Columnar vs legacy payload (per call, no model)

`benches/efficacy/columnar-payload.py` on the pinned reflex corpus, 10 fixed
`search_code` / `search_regex` calls, `content[0].text` bytes:

- median per-call saving **19.8%** (range 2.1% for a 1-row symbol result to 37.4%);
  **21.0%** pooled over all calls.
- Raw numbers: `benches/efficacy/results/columnar-payload.json`.

Session-level A/B runs cannot see this (turn count dominates `total_tokens`).

## What this says

- On single-shot find-all-usages, Reflex via MCP costs about 1.6× the tokens and 2× the
  dollars of built-in Grep, for more complete and more precise answers. The cost is
  round-trips, not payload size: deferred schema loading (ToolSearch) and freshness checks.
- Levers that follow from the data (not yet tried): make one call enough — e.g. fewer
  pre-search calls (the `check_index_status` description says "Call this at session start",
  but every `search_code` / `search_regex` response already carries `status` and
  `can_trust_results`); fewer deferred round-trips.

## Limits

- One model, three Rust repositories, 9 + 16 tasks.
- Arm B pays the ToolSearch round-trip because Claude Code defers MCP schemas; a client
  that loads MCP schemas eagerly would not.
- Recall scorer double-counts duplicate locations (values above 1.0).
