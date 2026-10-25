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

## Not done / next levers

- Opus drift to grep in 12-question sessions (1.15×): `list_locations` `preview: true`
  shipped in 2.1.0 after the last Opus measurement; not re-measured.
- Fewer listed tools (`search_ast`, `gather_context`, `check_index_status`, …) for the
  fixed text: needs a user decision and a deferred-mode adoption check.
- `list_locations` `dependencies: true` is accepted but its imports never reach the
  reply (either shape); unchanged.
