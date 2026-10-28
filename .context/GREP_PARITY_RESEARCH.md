# Grep parity: where the remaining cost goes (2026-10-25, Reflex 2.2.0)

Goal (user, 2026-09-28): MCP Reflex costs no more tokens than built-in Grep on Grep-like
searches. Earlier rounds: `AUTO_UPDATE_RESEARCH.md` ("Efficacy A/B", "Eager schemas",
"Long sessions", "Slim tool surface"). This file starts from the 2026-09-30 numbers on
the 10-tool surface (Sonnet 5 eager: 1.05× Grep at 12 questions, 1.11× at 50; Opus 5.5
eager: 1.15× at 12, 0.89× at 50).

Method notes:
- **Offline first.** Saved transcripts (`benches/efficacy/results-sessions/*`) are read
  by scripts; no tokens are spent. A reply's weight = characters × (1.25 for the cache
  write + 0.1 × every later API call that re-reads it).
- **Live runs cost plan usage, not money** (Claude Code on the user's Max plan). One
  12-question session is ~1M raw tokens (≈25 API calls × ~40K context, 98 % cache
  reads); a 50-question one ~6M. Ask before any run; prefer small, targeted runs.

## Where the gap is (offline, 2026-09-30 transcripts)

Median weighted tokens per session, eager Reflex arm (`Beager`) vs Grep (`A`):

| run | Grep | Reflex | difference |
| --- | --- | --- | --- |
| Sonnet 5, 12 questions | 116.5K | 122.4K | cache reads +7K: the fixed text |
| Sonnet 5, 50 questions | 645.5K | 706.3K | cache writes +27K (Read + Edit renames), reads +31K |
| Opus 5.5, 12 questions | 75.4K | 94.5K | reads +14K (grep through Bash, plus the fixed text) |
| Opus 5.5, 50 questions | 592.2K | 579.4K | Reflex cheaper |

- **Fixed text:** the first API call carries 34.4K tokens with Reflex vs 29.6K without
  (+4.8K per call ≈ +12K weighted per 12-question session). Instructions 2,367 chars;
  `tools/list` 10,216 chars, of which 7,162 are parameter schemas (`glob` 525, `exclude`
  522, `force` 464, `lang` 456 … over 5–6 tools each). Descriptions are already terse:
  dropping rarely used parameters (`force`, `offset`, `dependencies`) saves ~2K chars,
  ~1 % of a 12-question session. A large cut means listing fewer tools (a product
  decision). The instructions are risky to cut: one rewording took adoption from 37/72
  to 2/72 (`AUTO_UPDATE_RESEARCH.md`).
- **Installed but unused costs 4–10 %.** In the `v203-*` session runs the deferred arm
  made 0 Reflex calls (median) and still cost 1.04× (Opus) / 1.09–1.10× (Sonnet) Grep.
- **Reply weight:** removing the duplicate metadata (backlog §1 D: flat `has_more` /
  `total_count` / `returned_count`, fresh `status`) is 6–8 % of Reflex reply weight,
  under 1 % of a session. Grouping `list_locations` rows by file is 51–64 %.
- **Rename style is a coin flip.** In 12-question edit sessions the agent renamed with
  Read + Edit (expensive) or `sed -i` (cheap): Reflex arm 7/15, 8/15, 9/15 Read + Edit
  across three waves; Grep arm 7/15, 4/15, 2/15. One edit session's cost is decided by
  that choice, which swamps any reply-size effect.

## Grouped `list_locations` (`86cd73a`, `REFLEX_MCP_GROUPED`)

`locations` = `{path: [line, ...]}` (with `preview`: `[line, text]`) instead of
`[{path, line}]`. Arm `Beager_flat` (`REFLEX_MCP_GROUPED=0`) runs the old shape from the
same binary; `session_bench.py` now checks each arm's flags before the first trial.

| run (Sonnet 5, eager) | grouped | flat |
| --- | --- | --- |
| smoke, 6 sessions × 1 (investigate + edit) | 167.6K weighted, 50/51 list answers exact | 165.6K, 51/51 |
| `list_locations` reply weight per session (smoke) | 11.4K chars | 25.0K (−54 %) |
| investigate only, 3 sessions × 5 | **119.5K**, 120/120 exact | 130.2K, 120/120 |
| per corpus (reflex / ripgrep / tokio) | 0.967 / 0.913 / **1.159** of flat | |

- The smoke run's cost was decided by rename style (grouped 3/3 Read + Edit; flat 2/3
  `sed`), not by the shape; see the coin flip above.
- The one wrong answer was a copying error: the reply listed `crates/regex/src/literal.rs`,
  the agent wrote `crates/regex/src/regex/literal.rs:82`. Not seen again in 120 answers.
- **tokio +16 %:** 3 of 5 grouped tokio sessions read 10–11 files each. On a line with
  two matches ("`RngSeed` … `RngSeed`") the agent asked whether to list the line twice,
  ran `search_code` count mode, and read the file to check. Reflex lists a matching
  LINE once (like `grep -n`; `total_locations` and count mode agree), but the
  description said "Every match". Fixed: "Every matching line … a line with two matches
  appears once, like grep -n". Flat sessions had the same wording and 0 such sessions;
  whether the shape makes the doubt likelier is open.
- **Re-run with the new wording** (grouped, tokio investigate × 5, build `86cd73a` + the
  description fix): 40/40 list answers exact; file reads 1 in 5 sessions (was 32); reply
  chars 5,035–7,854 per session (was 5,035–18,960). Measured median 151.5K weighted, but
  that is a **cold prompt cache**: the new description changed `tools/list`, so each of
  the 5 parallel sessions wrote the whole 35,058-token prefix on its first call (earlier
  waves read 27,612 of it from cache and wrote 7,422). With the first call at its warm
  cost (−31.8K per session) the median is ~119.8K, **0.88× flat** (135.8K, other wave).
  Later calls wrote 9.2–10.9K (was 9.3–20.6K grouped, 13.3–19.1K flat).
- **Bench lesson:** compare arms in one wave with the same `tools/list` warmth. A new
  description or schema makes the first wave after it pay ~32K weighted per session
  extra. Check the first call's `cache_creation_input_tokens` before reading a cost.

## Opus 5.5 re-measure, 12-question investigate sessions (2026-10-25, `main` `2526848`)

Grep arm `A` vs eager Reflex `Beager`, one wave, 3 investigate sessions × 5 trials per
arm (`results-sessions/opus-investigate-grouped`). Compared with the same 3 sessions of
`steer-opus` (2026-09-30, 2.1.0 before `list_locations` `preview`):

| | 2026-09-30 | 2026-10-25 |
| --- | --- | --- |
| cost ($) Reflex / Grep | 1.167 | **1.019** |
| weighted tokens Reflex / Grep | 1.234 | **1.097** |
| Reflex calls per Reflex-arm session | 2.5 | 11.3 (`list_locations` 8.9, 4.9 of them with `preview`) |
| Bash `grep` / `rg` per Reflex-arm session | 11.1 | 1.1 |
| accuracy | 1.000 | 1.000 |

- **The drift is gone.** Opus now answers "where does X occur" with `list_locations`,
  adding `preview: true` when it wants the line text, instead of `grep -rnw` through Bash.
- **What is left is the fixed text.** Both arms make 24 API calls. Reflex's tool replies
  are smaller than Grep's (9,406 vs 10,147 chars per session) and Opus writes less
  (5,632 vs 6,482 output tokens). But each call carries 24,668 tokens of context at the
  start against 19,684 (+4,984: instructions + `tools/list`). Over 24 calls that is ~12K
  weighted, more than the whole ~10K weighted gap.
- No cold cache: first-call writes were 3,116 (Grep) and 4,000 (Reflex).
- The Grep arm itself cost more than on 2026-09-30 (105.8K vs 87.1K weighted, same
  tool mix). Compare ratios within one wave only.

## REF-219: file-grouped rows for `search_code` / `search_regex` (offline, 2026-10-27) — dropped

Replayed every `search_code` / `search_regex` list reply of the eager Reflex arms: rows
regrouped as `{path: [[start, end, preview, …], …]}` with `path` and `language` dropped
from the columns (scratch script, not kept).

| run | replies / session | rows / file | saved of these replies | saved of all replies |
| --- | --- | --- | --- | --- |
| opus-investigate-grouped (2026-10-25) | 1.9 | 1.2 | 4.0 % | 0.9 % |
| grouped-investigate-sonnet5 | 0.7 | 1.0 | 3.0 % | 0.2 % |
| grouped-wording-tokio-sonnet5 | 1.6 | 1.0 | 3.0 % | 0.7 % |
| grouped-smoke-sonnet5 | 0.7 | 1.0 | 3.1 % | 0.1 % |
| steer-long-opus (2026-09-30) | 2.4 | 1.0 | 3.4 % | 0.2 % |
| slim-long-sonnet5 (2026-09-30, before grouped `list_locations`) | 19.0 | 3.3 | 18.9 % | 7.8 % |

Since `list_locations` answers "where does X occur", agents call `search_code` about twice
a session, mostly with one row per file: nothing repeats. Replies are a small part of a
session next to the fixed text. Dropped by the user on 2026-10-27; a third reply shape is
not worth under 1 %.

## Fewer listed tools (2026-10-27, branch `feat/fewer-listed-tools`)

User decision: `search_ast`, `gather_context`, `index_project` and `check_index_status`
leave the default `tools/list`; `[mcp] extra_tools` lists them; `rfx mcp --no-update` lists
the two index tools (the only way to refresh then). Claude Code can call only listed tools,
so with `index_project` unlisted, `action_required` and error advice name `rfx index`, and
the instructions' freshness paragraph drops the "never call index_project or
check_index_status after edits" clause (`instructions()`).

| tool | chars in `tools/list` | calls in 508 Reflex-arm benchmark sessions |
| --- | --- | --- |
| `search_ast` | 1,060 | 0 |
| `gather_context` | 563 | 0 |
| `index_project` | 357 | 10 |
| `check_index_status` | 223 | 4 |
| `analyze` (kept) | 859 | 0 |

The benchmark questions are Grep-like, so 0 calls says little about structural work;
`analyze` stays listed (the import graph is a feature). Fixed text: instructions 2,367 →
2,307 chars, `tools/list` 10,286 → 8,079 (6 tools), together 12,653 → 10,386 (−18 %);
at ~2.5 chars per token, ~900 fewer tokens on every API call. Expected: about 2K of the
~10K weighted-token gap in the Opus 12-question sessions above.

### Measured (2026-10-27, Sonnet 5.5, `results-sessions/sonnet55-fewer-tools`)

One wave: 3 investigate sessions × 2 trials × arms A / B (deferred) / Beager, 6 workers.
Binary: branch `feat/fewer-listed-tools` uncommitted on `843ec7c` (startup line
`unlisted=search_ast,gather_context,index_project,check_index_status`). Sonnet 5.5 instead
of Opus (user: plan usage); so no earlier wave is comparable, only the ratios in this one.

| arm | weighted (median) | cost $ | turns | Reflex calls / session | accuracy |
| --- | --- | --- | --- | --- | --- |
| A (Grep) | 97.0K | 0.215 | 24 | 0 | 18/18 sessions 1.000 (all arms) |
| B (deferred) | 105.7K (1.07×) | 0.234 (1.06×) | 25 | 12 in 4 of 6, 0 in 2 | |
| Beager | 108.4K (1.12×) | 0.231 (1.06×) | 24 | 12 | |

- **Fixed text:** first call reads + writes 23,806 tokens (Beager) vs 19,757 (A): +4,049 per
  call, against +4,984 with 10 tools on Opus 5.5 (2026-10-25; another model, so the
  tokenizer may differ). Over 24 calls that is still ~10K weighted, about the whole
  Beager−A gap (11.4K): the fixed text remains the gap.
- **Cache warmth equal:** in every arm 2 of 6 sessions wrote the prefix cold (the first
  round), the other 4 read it.
- **Adoption (deferred):** 4 of 6 B sessions loaded the schemas with one ToolSearch and
  answered all 12 questions with Reflex; 2 used Grep only. No session tried a hidden tool
  (0 "No such tool" replies). n = 6: no change in adoption can be read from it.
- Beager answered every question with `list_locations` (54) or `search_code` (18).

## Not done / next levers

- The fixed text is still the gap (+4K tokens per call, ~1.06× Grep's cost on Sonnet 5.5).
  What is left is the 6 schemas (8,079 chars, mostly the shared search parameters) and
  the instructions (2,307 chars, the adoption-critical text).
- `list_locations` `dependencies: true` is accepted but its imports never reached the
  reply (either shape). Fixed 2026-10-25: a `dependencies` map beside `locations`.
