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

## Where the rest of the fixed text goes (offline, 2026-10-28)

`tools/list` 8,054 chars (compact JSON) + instructions 2,307. Arguments agents passed in
5,914 Reflex calls over the 520 saved Reflex-arm sessions (`results-sessions/*/B*`):

| parameter | uses | chars over the 4 search tools |
| --- | --- | --- |
| `exclude` | 0 | 424 |
| `force` | 0 | 336 |
| `offset` | 0 | 312 |
| `dependencies` | 0 | 228 |
| `preview_length`, `expand` | 0 | 145 |
| `exact` | 2 | 60 |
| `kind`, `lang`, `ignore_case` | 1–4 each | ~900 |
| `glob` | 4,482 | 412 |

Options weighed: hide the unused parameters (−1.2K, low risk: done); fold `search_regex`
into `search_code` as `regex: true` (−1.4K, 163 calls); drop "Exact call shapes" and
"Rules" from the instructions (−550); drop every parameter description (−3.3K: names
such as `file` and `force` are unclear without them; not done). `offset` stays listed:
the `search_code` description pages with it.

**Parity cannot come from cuts alone.** The Reflex arms keep Grep in the tool list, so
Reflex's fixed text is pure extra: 24 calls × ~10K chars ≈ 250K chars per session,
against ~10K chars of replies (Grep's: 8.8K). Equal cost needs fewer turns than Grep, or
questions where Grep needs more turns or larger replies (see "aware" sessions below).

## Hidden parameters (2026-10-28, branch `feat/hidden-params`)

`HIDDEN_PARAMS` (`exclude`, `force`, `dependencies`, `preview_length`, `expand`, `exact`)
are dropped from the four listed search schemas; `tool_specs` still validates against
the full schemas, so they are accepted. `search_ast` and `index_project` keep `force`.
`tools/list` 8,054 → 6,847 chars.

### Measured (Sonnet 5.5, `results-sessions/sonnet55-hidden-params`, same shape as above)

- First-call prefix (eager): 23,805 → 23,318 tokens (−487 per call, ~−11.7K raw per
  24-call session). Grep arm: 19,749–19,761.
- Accuracy 1.000 in every arm; 0 tool errors; eager answered 72/72 with Reflex.
  Deferred: 3 of 6 sessions used Reflex (was 4 of 6; n = 6).
- Cost ratios **cannot be compared with the run before**: cache warmth differed. The Grep
  arm started warm in 6/6 sessions (cache writes 11–22K per session), the eager arm cold
  in 2/6 (32K and 35K written). A cold start adds ~20K tokens at the write price, which is
  the eager arm's 1.27× (deferred: 1.02×). Warm sessions only, eager vs Grep: 1.04–1.14×,
  as in the run before. The saving (~1 % of a session) is below this noise.
- **Harness fix needed:** warm every arm's prefix before its trials (one short call per
  arm), so the first trials of each arm start alike.

## `aware` sessions: questions grep should answer badly (2026-10-28)

The investigate sessions ask grep-shaped questions (every whole-word line, a
definition, a file count), where Grep is at home. `session_bench.py generate` now also
writes `{repo}-session-aware`: 10 "every line where `X` is used in code, the
definition included, not lines where it is only in a comment or a string" and 2 counts
of such lines. Names are picked where ≥ 30 % of grep's `.rs` lines are comments or
strings. Answer key: `benches/efficacy/rust_code.py`, a Rust lexer written apart from
`src/line_filter.rs`. On the 137 names that qualify, grep returns 2,261 lines, 1,113 of
them code, so a Grep agent must read context to drop half its lines; `find_references`
answers in one call.

Building the key found a Reflex bug: `find_references` read each line alone and kept
lines inside multi-line block comments (`/*! … */`, ` * …`) and multi-line strings.
`main` (`843ec7c`): exact on 109 of 137 names, 59 noise lines kept, none missed. Fixed on
`feat/hidden-params` (`line_filter::FileRegions`, a whole-file scan with the same rules,
falling back to the line when the scan does not end in code): 137 of 137 exact, 1–4 ms
per call. Two smaller causes on the way: a comment-only line kept because the pattern
occurred inside a longer code identifier (`async_op_span`), and the line check against
the engine's preview, which centres a long line on the match with `…` at both ends.

### Measured (Sonnet 5.5, `results-sessions/sonnet55-aware`, 3 sessions × 2 trials × 3 arms, warm-up on)

| arm | weighted | cost $ | turns | accuracy | tools |
| --- | --- | --- | --- | --- | --- |
| A (Grep) | 101.3K | 0.216 | 24 | 1.000 | Grep 72 |
| B (deferred) | 105.8K (1.04×) | 0.227 (0.98×) | 24 | 0.986 | Grep 48, list_locations 24 (2 of 6 sessions used Reflex) |
| Beager | 110.8K (1.10×) | 0.235 (1.04×) | 24 | 0.986 | list_locations 57, find_references 16 |

**Not a Grep weakness at this size.** With 5–30 grep lines per name, every line is in
Grep's reply, and the agent drops `//`, `///` and string lines by reading them: Grep
was exact on all 72 questions without a single Read. Reflex agents mostly did the same
with `list_locations preview:true` (57 calls) rather than `find_references` (16). Both
misses (B, Beager) are one count question, `OwnedHandle`, where the agent listed the 7
right lines and wrote "Eight": arithmetic, not search. Warm-up worked: cache writes per
session were alike in all arms (A 22–43K, Beager 19–46K). Where grep should lose:
names with hundreds of lines (the agent must read every line to drop the comments, or
answer a count with comment lines included), and cases inside multi-line comments or
strings, which a line preview cannot show.

Not yet in `aware`: import-graph questions (who imports a file, transitive deps). The
corpus is locked to three Rust repos (`benches/efficacy/repos.md`), so a TypeScript
corpus with an exact relative-import key is out. A prototype Rust resolver
(`use crate::` / `self::` / `super::` paths, `{…}` groups expanded, `as` aliases, the
longest module prefix that is a file) on `tokio/src`, 56 files with 3–25 importers,
compared with `get_dependencies reverse:true`:

| | edges |
| --- | --- |
| prototype key | 417 |
| both | 304 (73 %) |
| key only | 113 |
| Reflex only | 23 (mostly `mod x;` declarations: the parent "imports" the child) |

The 113 Reflex misses, sampled: imports inside `cfg_*! { … }` macro bodies (tree-sitter
does not parse macro bodies); `use crate::Item` / `use super::Item` where the item lives
in `lib.rs` or the parent `mod.rs`; one module of a group (`use crate::sync::{oneshot,
watch}` missed `oneshot.rs`). So on Rust the import graph is not yet a proven
strength: fix these before asking agents graph questions graded against a key. Open
question for the key: does `mod x;` count as an import, and does `use crate::Item`
import `lib.rs`?

### `aware-large` (Sonnet 5.5, `results-sessions/sonnet55-aware-large`, 1 trial × 3 corpora × 3 arms, warm-up on)

The same questions on names with 100–400 grep lines in `.rs` files, ≥ 30 % of them
comments or strings, ≥ 3 in code (`aware_large_session`: 6 lists, 6 counts). Extreme
picks: ripgrep `search` 358 grep lines / 9 in code, tokio `access` 254 / 7. Offline,
`find_references` (binary with the `FileRegions` fix) is exact on all 36 names, list
and count mode.

| session | A weighted / $ | B weighted / $ | Beager weighted / $ |
| --- | --- | --- | --- |
| reflex | 282.4K / 0.694 | 256.7K / 0.657 | 249.5K / 0.619 |
| ripgrep | 230.2K / 0.539 | 277.0K / 0.626 | 256.5K / 0.539 |
| tokio | 141.9K / 0.330 | 151.9K / 0.351 (Grep/Bash only) | 300.2K / 0.698 |
| sum | 654.5K / 1.563 | 685.6K (1.05×) / 1.634 (1.05×) | 806.2K (1.23×) / 1.856 (1.19×) |

Accuracy 1.000 in every arm (A reflex 0.9995). **Grep did not over-fetch:** after one
Grep call per session, A agents switched to Bash `rg -nw NAME | rg -v '^[^:]+:[0-9]+:\s*//'`,
which drops `//` / `///` / `//!` lines before they reach the context (tokio: 23K chars
of tool results for 12 questions). That filter is exact here: in these Rust trees the
noise is almost all whole-line comments; trailing comments, strings and `/* */` blocks
are rare. **Reflex agents did not use the strength:** 5 of 6 Reflex sessions answered
with `list_locations preview:true` (every grep line, comments included, read by the
agent: Beager tokio 182K chars of results, 2.1× A's cost). The one session that called
`find_references` (Beager ripgrep, 13 calls) re-checked each answer with Bash rg.
The instructions route "where does X occur" to `list_locations` first, and
`find_references` reads as a symbol tool ("A symbol's definition and every usage"),
so a plain word like `blocking` never goes there.

Conclusion: on Rust, comment noise is cheap for grep (a one-line pipe), so this is not
a strong grep weakness. The loss was routing; see the next run.

### `aware-large` with routing text (Sonnet 5.5, `results-sessions/sonnet55-aware-large-route`, same shape)

Two text changes: the instructions' last paragraph says "For uses in code only (no
comments or strings), use find_references; it drops those lines exactly, so do not
re-check." (it replaced "find_references for a definition plus every call site without
string/comment noise"); the `find_references` description says any identifier, not only
a symbol, and that a whole-file scan drops comment and string lines, so no re-check is
needed. Instructions stay under the 2,400-char budget.

| session | A weighted / $ | B weighted / $ | Beager weighted / $ |
| --- | --- | --- | --- |
| reflex | 252.9K / 0.606 | 187.2K / 0.404 | 198.6K / 0.456 |
| ripgrep | 231.1K / 0.526 | 153.0K / 0.301 | 136.7K / 0.295 |
| tokio | 128.3K / 0.271 | 130.8K / 0.265 | 140.8K / 0.308 |
| sum | 612.4K / 1.403 | 471.0K (0.77×) / 0.970 (0.69×) | 476.1K (0.78×) / 1.059 (0.75×) |

Accuracy 1.000 in every Reflex session (A reflex 0.9992). Every Reflex session used
`find_references` on all 12 questions (count mode on the 5–6 count questions); one B
session ran 4 Bash checks, none of the Beager sessions re-checked. Tool-result chars per
session fell from 108–182K to 23–75K. The A arm repeated its pipe habit (cost within
10 % of the previous wave). One trial per cell: the direction is clear, the size is not.
### Regression check: `investigate` with routing text (`results-sessions/sonnet55-investigate-route`, 1 trial)

Plain "every occurrence" questions did not move: 0 `find_references` calls in 6 Reflex
sessions (`list_locations` 28, `search_code` / `search_regex` 8), accuracy 1.000 in
every arm. Weighted vs A: B 1.04×, Beager 1.39× raw.

**The warm-up does not warm in real runs.** First-call cache usage, every warm-up run:
a session's first call reads 0 unless an earlier session of the same arm already ran
(trial 02 reads 19,541 for Beager). A and B mostly read 16,848 anyway: that is Claude
Code's own prefix, cached by any recent session; Beager's prefix (Reflex schemas in
the tool block) is not, so every first Beager session paid ~19.5K cache-write tokens
(23.4K written, 0 read).

**Cause (fixed 2026-10-29):** `warm_up` called `claude --print -- PROMPT` (text input);
trials send stream-json on stdin (`--input-format stream-json`). The two send different
prompt prefixes, so a trial never read the warm-up's entry. Clean test with a renamed
MCP server (`mcp__rfxx…__*`, a prefix nobody had cached): text warm-up → trial read 0;
stream-json warm-up → trial read 19,602. `warm_up` now uses the trial's command line
(`session_cmd`, `user_message`) and prints its cache write/read; through `run()` on
the never-run `Beager_allparams` arm: warm-up write 23,747 / read 0, then the trial's
first call read 20,065. Runs before the fix: compare with the cold prefix moved from
the write to the read price (cold = first call read < 16,848 for A/B, < 19,541 for Beager):

| run | B raw → warm | Beager raw → warm |
| --- | --- | --- |
| hidden-params (investigate, 2 trials) | 1.06× → 1.06× | 1.19× → 1.10× |
| investigate-route | 1.04× → 1.04× | 1.39× → **1.12×** |
| aware-large (before routing) | 1.05× → 1.05× | 1.25× → 1.26× (A cold too) |
| aware-large-route | 0.79× → 0.79× | 0.80× → **0.68×** |

So the routing text costs nothing on plain searches (Beager 1.10× → 1.12×, within
noise) and cuts code-only questions to 0.68–0.79× Grep's weighted tokens. Stronger candidates: languages with
block comments and docstrings (Python `"""…"""`, JSDoc `/** … */`), where a line pipe
cannot see that a line is inside a comment; the corpus lock rules them out today.

### Warm re-run (2026-10-29, fixed warm-up, `results-sessions/sonnet55-{aware-large,investigate}-warm`)

Same sessions, same binary, 1 trial per cell. Every warm-up wrote or read the prefix
(Beager: write 23,254 on the first run, read 19,578 on the second), and all 18 first
calls read from the cache (A/B 16,848, Beager 19,578). Sums over 3 sessions per arm:

| sessions | A weighted / $ | B vs A (weighted / $) | Beager vs A (weighted / $) | accuracy A / B / Beager |
| --- | --- | --- | --- | --- |
| aware-large | 575.6K / 1.312 | **0.77× / 0.69×** | **0.72× / 0.64×** | 0.972 / 1.000 / 1.000 |
| investigate | 268.9K / 0.570 | 1.04× / 1.03× | 1.05× / 0.98× | 1.000 / 1.000 / 1.000 |

- `aware-large`: every Reflex question went to `find_references` (B 27 lists + 13
  counts, Beager 19 + 18), none re-checked. A used 39 Bash pipes and 7 Grep calls.
- Grep's miss: `command` counted 56, key 58. Its pipe (`grep -rnw … | grep -Ev
  '^…\s*//'`) kept every non-`//` line, and the agent dropped two lines that mix code
  and a string (`format!("Unknown command: {}", command)`,
  `shell_words::split(command).context("Failed to parse command string")`). Mixed
  lines are what a line filter cannot judge and the whole-file scan can.
- `investigate`: no question moved to `find_references`; eager Reflex at Grep's cost
  (0.98×), 1.05× its weighted tokens. Before the fixed warm-up, the same comparison
  needed the cold-prefix correction (1.12× after it).

### `search_regex` folded into `search_code` (2026-10-30, 2.2.0 + unreleased)

`search_code` takes `regex: true`; `search_regex` is a `LEGACY_TOOLS` name (runs as
`search_code` with `regex: true`, plus a deprecation warning). Sizes, compact JSON,
UTF-8 characters:

| | `main` (89acce5) | `regex: true` |
| --- | --- | --- |
| `tools/list` | 6,959 | 5,938 (−1,021) |
| instructions | 2,341 | 2,377 (+36) |
| fixed text per call | 9,300 | 8,315 (−985, −10.6 %) |

- The 6,847 measured for hidden parameters predates the longer `find_references`
  description of the routing change (+108); `main` is 6,959.
- The estimate was −1.4K. The real cut is smaller: `search_code` gained the `regex`
  parameter and a regex clause in its description, and the instructions name
  `regex:true` in the call shape and the matching rules.
- The instructions keep a regex call shape: agents that never load the schema learn
  the flag there. (The call shapes left on 2026-10-31; see "Instructions trimmed".)
- Risk: an agent passes a regex without `regex: true`. It runs as literal text and
  finds the text itself or nothing. On this repo `impl\s+\w+\s+for` counted 1 (a doc
  line) against 58, with no warning; `fn (get|set)_\w+` counted 6 against 89, with the
  bracket-rewrite warning. Fix: `filter::regex_syntax_warning` puts a warning naming
  `regex: true` on any literal search with regex syntax (class escapes, escaped regex
  characters, `.*`, `.+`). It is a warning, not a rewrite: `foo\.bar` can be literal.

#### Sessions (2026-10-31, Sonnet 5.5, 2.2.0 + unreleased)

`regex` sessions (`session_bench.py`): one per corpus, 12 questions each, every
question names a Rust regex. 8 ask for every matching line (3–80 lines), 4 for the
count (10–400 lines); the answer key is `rg -g '*.rs' -e RX SCOPE`. 13 of the 36
regexes have no brackets, so a literal search of them gets only the new warning.
Offline check first: `search_code` with `regex: true` equals the key on all 36;
without the flag every one returns 0 with the warning. 1 trial per cell, warm-up per
arm. Wave 1 (`results-sessions/sonnet55-regex`): A, B, Beager on the new binary.
Wave 2 (`results-sessions/sonnet55-regex-main`): Beager on `main` (89acce5), where
`search_regex` is listed; its Grep baseline is wave 1's A.

| | A (Grep) | B | Beager, `regex: true` | Beager, `main` |
| --- | --- | --- | --- | --- |
| sessions using Reflex | — | 1 of 3 (1 call) | 1 of 3 | 3 of 3 |
| questions answered by Reflex | 0 of 36 | 0 of 36 | 12 of 36 | 33 of 36 |
| Bash re-checks | 0 | 0 | 1 | 6 |
| cost vs A (median) | 1.00× | 1.00× | **1.02×** | 1.06× |
| weighted vs A (median) | 1.00× | 1.01× | **1.09×** | 1.17× |
| accuracy | 1.000 | 1.000 | 1.000 | 1.000 |

Per session (cost $ / weighted tokens):

| session | A | Beager, `regex: true` | Beager, `main` |
| --- | --- | --- | --- |
| reflex | 0.240 / 111,562 | 0.244 / 122,923 (12 `search_code` regex) | 0.258 / 130,597 (10 `search_regex`, 3 `rg -c … \| awk`) |
| ripgrep | 0.295 / 126,057 | 0.300 / 134,987 (Grep) | 0.313 / 148,995 (11 `search_regex`, 3 `grep -rE … \| wc -l`) |
| tokio | 0.271 / 117,625 | 0.297 / 128,267 (Grep) | 0.250 / 120,780 (12 `search_regex`) |

- Fixed text, measured: the Beager warm-up wrote 22,849 cache tokens against 23,254
  on `main` (−405 tokens per call); A's warm-up wrote 19,643.
- Cost: no worse, maybe better. On the one session where both binaries used Reflex
  throughout (`reflex`), `regex: true` cost 5.7 % less; `main` also re-checked 3
  counts with `rg`.
- Adoption fell: a tool named `search_regex` drew regex questions (3 of 3 sessions); a
  flag on `search_code` does not (1 of 3). The rest went to Grep, which is exact on
  regexes, so cost and accuracy did not move. Accepted: Reflex has no accuracy edge on
  regex questions; the fold saves the fixed text on every call of every session.
- When used, `regex: true` was used right: 8 lists and 4 counts (`mode: "count"`), all
  exact, no re-checks.
- Warning defect, fixed: B called `list_locations` with `impl\s+\w+\s+for\s+\w+`. The
  warning said "Pass regex:true", a key `list_locations` does not take, and the agent
  went to Grep. The text now says "Use search_code with regex:true"; the bracket
  warning names `search_code` too.
- 1 trial per cell: the adoption drop (3 of 3 → 1 of 3 sessions) suggests an effect,
  not a proof. A tool choice is made once per session and kept, so the 36 questions
  are not 36 samples.

### Instructions trimmed (2026-10-31, 2.3.0 + unreleased)

"Exact call shapes" (three example calls) and "Rules" (the canonical argument names,
"literal text index") left the instructions. "Search for code, not natural language."
replaces the last rule. "glob and exclude follow gitignore rules" became "glob follows"
(`exclude` is a hidden parameter). Sizes, UTF-8 characters:

| | 2.3.0 | trimmed |
| --- | --- | --- |
| instructions | 2,377 | 1,838 (−539) |
| fixed text per call | 8,315 | 7,776 (−6.5 %) |

- The rules answered 1.7.0 field data: 14 of 16 sessions opened with `query`, because
  agents called deferred tools without their schema. Today Claude Code loads a deferred
  schema before the first call, and `ARG_ALIASES` rewrites `query` / `symbol` / `text`
  / `search` → `pattern`, `max_results` → `limit`, `path` → `file`, with a warning.
- In the saved sessions (`results-sessions/*/B*`), 0 of 6,526 Reflex calls used one of
  those names. The rules were in place for all of them, so this alone does not show
  they are unneeded; the aliases make a wrong name cost a warning, not a failed call.
- Kept word for word: the Reflex-first paragraph, the ToolSearch line, the
  `list_locations` / `find_references` routing and the retry-once fallback. Each was
  measured; `test_instructions_keep_the_routing_rules` pins them.
- The regex call shape left with the rest. Agents learn `regex:true` from the Matching
  paragraph and the `search_code` description.

Sessions (2026-10-31, Sonnet 5.5, `investigate`, 1 trial per cell, warm-up per arm;
`results-sessions/sonnet55-trim-investigate`). A regression check against the
2026-10-29 warm run (`sonnet55-investigate-warm`, before the `regex` fold and the trim):

| | A (Grep) | B | Beager |
| --- | --- | --- | --- |
| sessions using Reflex | — | 3 of 3 (was 2 of 3) | 3 of 3 (was 3 of 3) |
| Reflex calls | 0 | 34 `list_locations`; 2 Grep (was 24 Reflex, 12 Grep) | 24 `list_locations`, 12 `search_code` (unchanged) |
| cost vs A (median) | 1.00× | 1.02× (was 1.03×) | 1.03× (was 0.98×) |
| weighted vs A (median) | 1.00× | 1.06× (was 1.04×) | 1.08× (was 1.05×) |
| accuracy | 1.000 | 1.000 | 1.000 |

- Fixed text, measured by the warm-up cache writes: eager 22,849 → 22,641, deferred
  20,594 → 20,386 (−208 tokens per call in both). With the `regex` fold: eager
  23,254 → 22,641 (−613) since 2.2.0.
- No wrong argument name, alias warning, tool error or prose pattern in any arm (the
  same as the baseline).
- The ratios moved because Grep moved: A's tokio session cost 98,987 weighted tokens
  against 108,107 on 2026-10-29. Eager's own weighted tokens fell 1.0–1.7 % per
  session, about what −613 tokens per call at the cache-read price predicts.

## Not done / next levers

- The fixed text is still the gap: ~7.8K chars per call after the trim (5,938
  schemas + 1,838 instructions). The cut left on the list is the parameter
  descriptions (−3.3K): names such as `file` and `include_locks` are unclear
  without them.
- `list_locations` `dependencies: true` is accepted but its imports never reached the
  reply (either shape). Fixed 2026-10-25: a `dependencies` map beside `locations`.
