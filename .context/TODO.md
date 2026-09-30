# Reflex TODO

**Last Updated:** 2026-09-28 (Reflex 2.0.3)

> **⚠️ AI Assistants:** Read the "Context Management & AI Workflow" section in `CLAUDE.md`.
> Update this file as you work. Keep it to LIVE work: open tasks, current policy, open
> follow-ups. Finished work belongs in CHANGELOG.md and git history, not here.

> **History:** the pre-2026-09 roadmap (MVP plan, per-module task lists, 2025 status
> summaries, old benchmarks) was removed on 2026-09-28 because it described an
> architecture that no longer exists. Read it with `git show fc8da6b:.context/TODO.md`.

---

## 🔄 Auto-update (2026-09-29) — IN PROGRESS on `feature/auto-update` (not pushed)

Every command that reads the index brings it up to date first; no watcher, no manual
`rfx index`. Plan, code survey, steps and gates: `.context/AUTO_UPDATE_RESEARCH.md`.
Builds on the incremental index branch below. Steps 1–9 and docs done; fidelity 15/15,
golden 0 diffs, `latency_budget` +1.6 %, MCP edit-then-search 138–161 ms (as built:
`AUTO_UPDATE_RESEARCH.md`). Step 9 (user, 2026-09-30): the MCP text no longer sends
agents to `check_index_status` / `index_project`; every JSON-object answer carries
`can_trust_results`. Efficacy re-run (Opus 5.5): 1.675×, unchanged — Opus never made the
status calls. Sonnet 5: 1.603× (was 1.646×); status calls 33 → 0, arm-B median turns
4 → 3, tokens 148k → 101k. `alwaysLoad` (now in the docs' example configs) removes the
ToolSearch turn: cost 1.20× (Sonnet) / 0.98× (Opus) Grep, tokens 1.53× / 1.82× — the
44 KB `tools/list` rides on every turn. Long sessions (`session_bench.py`): eager Reflex
1.10× Grep's cost over 50 questions (1.18–1.36× over 12); per query it equals Grep.
Next: shrink the schemas (backlog §1 B), re-measure with `session_bench.py`.

Known limit: the 1 s verdict memo — in `rfx mcp` / `rfx serve`, an edit made within 1 s
of the previous check can be missed by the next call (`REFLEX_FRESHNESS_TTL_MS`).

Decisions (user, 2026-09-29): update inside the command when the check says stale;
default on for every command, `--no-update` opt-out on all; always wait; no index →
build it (no directory guard); remove the MCP `check_index_status` / `index_project` text
only after the fidelity test shows 95–100 % fresh answers.

---

## ⚡ Incremental index updates (2026-09-29) — DONE on the branch, awaiting review

Branch `feature/incremental-index` (not pushed). Design, what changed from the plan and
why: `.context/INCREMENTAL_INDEX_RESEARCH.md` ("As built"); numbers:
`.context/PERFORMANCE_RESEARCH.md` ("Incremental index round"). Hard rule held: golden
battery identical to 2.0.3 on four corpora, fresh and after scripted updates.

Decisions (with the user, 2026-09-29):
1. The early-stop planner uses an id-free planning size (sidecar), so an updated index
   answers exactly like a fresh build (golden diff vs 2.0.3: 0).
2. While a delta is live, `content.bin` / `trigrams.bin` are absent, so an older binary
   stops (`CacheCorrupted`) instead of serving the base.
3. A schema-hash change forces one full rebuild (the fast-path check was dead).
4. `files.walk_seq` keeps walk order for every db-id-ordered output.
5. `Indexer::update_paths` asks `git status` about the named paths only (a whole-tree
   status costs 0.3–0.7 s on Kubernetes). So `branches.is_dirty` can stay `true` after
   every edit is reverted, until the next `rfx index`; meanwhile `rfx stats` can miss its
   "Uncommitted changes not indexed" line. Search results and freshness stay exact.

Open follow-ups:
- **Nothing-changed gate not fully met**: 0.19 s against a 0.15 s walk (+30 %; +21–30 %
  against the whole discovery phase). Left: the meta.db commit (8 ms, `synchronous=FULL`,
  it writes "Last updated" and the branch row), fixed process costs, classify.
- Decide whether the first update after an `rfx index` should wait for the symbol pass
  that run spawned (~76 ms today; auto-update restarts a cancelled pass).
- `latency_budget`: four shapes read +6 … +12 % over 12 runs (`rare_ident`, `ci_regex`,
  `mcp/regex_getset`; 10–325 µs), ranges overlapping; re-measure on an idle machine.
- An add or a delete through `update_paths` re-resolves every internal import (~90 ms on
  Kubernetes); a reverse index of unresolved/suffix-matched imports would make it local.
- `symbol_cache::ensure_schema` treats a failed `pragma_table_info` read as an old
  schema (`unwrap_or(0)`): a query racing the background symbol pass right after
  `rfx index` can warn "Symbol cache schema outdated", drop the symbol cache and
  rebuild it. Unchanged since 2.0.3; seen once in 7 golden captures (2026-09-29).
- A one-shot `rfx query` peaks 1–2.5 % above 2.0.3 (larger binary, planning-size
  pages), 3–8 % with a large delta live; the long-running `rfx mcp` is level.

---

## 📚 Pulse revamp: grounded, Stripe-grade docs site on Starlight (2026-09-25) — IN PROGRESS

Plan: `~/.claude/plans/i-want-to-revamp-purrfect-stonebraker.md` (copy lands in
`docs/features/PULSE.md` at M7). Branch `feat/pulse-revamp`.

Decisions (with the user, 2026-09-25):
- Two tabs: **Docs** (users: overview, get started, guides, CLI + API reference,
  changelog, glossary) and **Internals** (contributors: architecture, modules, dep map, health).
- Renderer: **Astro Starlight**. Node + npm deps are on-demand downloads (system Node
  reused when the major fits), never inside the rfx binary. `-o` = static HTML only.
  CI avoids `npm install`: prebuilt pruned runtime tarball (release asset, SHA-256 pinned).
- LLM: grounded writer on a complete no-LLM base; sentence-level citations; deterministic gate.
- Keep + rethink Map, Changelog, Glossary. Drop Timeline, Explorer, Onboard, wiki dump.
- Defaults: Reflex's Docs tab leads with the CLI (library opt-in via `[docs] public`);
  LLM cache key includes the model; ships as MINOR with a migration note.
- The JSON page bundle is the model/renderer seam; a native Rust renderer is the
  fallback if the M0a spike says no-go.

| # | Milestone | Status |
| --- | --- | --- |
| M0a | Renderer spike: template scaffold, synthetic bundles 100/1k/5k/20k, runtime tarball size, build time + RSS, go/no-go | completed (2026-09-26): CONDITIONAL GO → user chose Starlight design D (Rust renders all HTML, Astro lays out). `.context/PULSE_RENDERER_SPIKE.md` |
| M0b | Provider layer (`CompletionRequest`, JSON mode, error kinds, usage); content-addressed write cache (no snapshot id); executor (dry-run, budget, whole-run degrade, scoped force, prune); drop `postprocess_narration`; typed changelog slot; concurrency default 4 | completed (2026-09-26): `src/semantic/providers/wire.rs`, `src/pulse/write/{cache,run,provider}.rs`; e2e: re-index → 3/3 cache hits |
| M1 | Docs Model (`src/pulse/model/`), file roles, Linker, FactStore, slugs, `rfx pulse model --json`; port map/modules/changelog | completed (2026-09-26): `src/pulse/{model,extract,build}/`; Reflex: 12 pages, 0 broken links, 101 fixture + 31 test files kept out of modules |
| M2 | Starlight renderer replaces Zola: runtime modules, `pulse-runtime.yml`, template v1, `render/*`, `astro.rs`, `publish.rs`, `serve.rs`; delete Zola path | mostly done (2026-09-26): `render/{html,bundle,project}.rs`, `runtime.rs` (npm ci into `~/.reflex/pulse/runtime/<deps-hash>`), `publish.rs`, `serve.rs`, new `site.rs`; Zola/Pagefind/explorer deleted. Reflex: 438 pages, 5.5 s. **Left:** Node download, Pagefind sharding > 8k pages, page caps (prebuilt runtime done in M7) |
| M3 | Rust API reference: `src/parsers/api/rust.rs`, `api.db`, surface resolver, reference pages, clap CLI adapter, manifest entry points + capabilities | mostly done (2026-09-26): extractor, `api.db`, Rust surface (pub reachability, impl attach, `pub use` inlining), module/type reference pages, intra-doc links, clap CLI pages, `pulse.toml [docs]`. Reflex: 412 Docs pages (32 CLI), 0 broken links, model in 0.25 s. **Left:** manifest entry points + capability facts (moved to M5, where the writer needs them) |
| M4 | TS/JS, Python, Go extractors + surfaces; usage scan; markdown guide ingestion | in progress (2026-09-26): guides done (`build/guides.rs`, `build/links.rs`); language-neutral surface done; Python and Go reference merged 2026-09-26 (`parsers/api/{python,pydoc,go}.rs`, `extract/surface/{python,go}.rs`, `EXTRACTOR_VERSION` 5); TS/JS agent running; usage scan pending |
| M5 | Grounded writer: evidence packs, contracts, gate + claim guards, guides, how-tos, `--explain`/`--strict` | mostly done (2026-09-26): capabilities from imports, `model::evidence`, `build::evidence`, `write::{contract,gate}`, grounded prompt, `--explain`, write report. Reflex: 10/10 sections, 0 dropped. **Left:** concept guides + how-tos, critic, `--strict`, live eval harness |
| M6 | Changelog (release surfaces, API diff, `since`) + Glossary rework | changelog done (2026-09-26): `build/releases.rs` (tags, CHANGELOG sections, API delta via `git cat-file --batch`), release pages + index. **Left:** `since` per symbol, glossary rework |
| M7 | pulse-action, `pulse.yml` migration, eject, critic, eval harness, generic extractor, docs | in progress (2026-09-26): prebuilt runtime download (`runtime.rs`, `runtime.lock.json`, sha256 `DEPS_HASH`, `REFLEX_PULSE_MIRROR`), `rfx pulse runtime key/status/install`, `.github/workflows/pulse-runtime.yml` (5 platforms, smoke = `fixtures/smoke` + `scripts/smoke-check.mjs`), composite action `.github/actions/pulse` + `pulse.yml` migration, wiki/onboard/timeline removed, PULSE.md/README/CHANGELOG rewritten. **Left:** first `pulse-runtime` release, eject, `serve --dev`, critic, eval harness, generic extractor |

Known defects this fixes (found 2026-09-25): LLM cache key includes the snapshot
timestamp (`llm_cache.rs:44`) → CI never hits; `json_mode` never used; `postprocess_narration`
corrupts identifiers in JSON; TUI classified as HttpServer by filename (`onboard.rs`);
`/wiki/` links ignore `--base-url`; Mermaid/D3 from CDN; `tests/corpus` fixtures in nav;
failed Zola build still exits 0.

---

## 🐛 Open bugs

- **Stale index, but no `can_trust_results: false`** — MCP fixed on `feature/auto-update`
  (every JSON-object answer carries it). Left: `check_index_status` with no index returns
  only `{status, action_required}`; `rfx query --count --json` returns `{count,
  timing_ms}`; array answers (`get_dependencies`, `get_dependents`, `search_ast`) and
  `get_transitive_deps` (a path-keyed map) have no place for it.
- **`find_references` silently drops call sites on any line containing a URL.**
  `src/line_filter.rs:83` does `line.find("//")` and treats anything after it as a
  comment. `https://` contains `//`, so:

  ```rust
  let url = "https://example.com/api"; call_the_target();
  //                  ^ first "//" at byte 21      ^ pattern at byte 41
  // 21 <= 41  =>  treated as commented  =>  dropped
  ```

  A silently wrong answer. Affects every language filter using the `//` rule, and is far
  worse on minified JS, where one early URL hides every later match on that line.
  Found 2026-09-22; still present in 2.0.3.
- **MCP `index_project` never spawns the symbol pass** (`rebuild_index`, `src/mcp.rs`
  ~L1287); only `rfx index` does (`src/cli/index.rs`). Symbol queries then parse on demand.
- **Readers can open a mismatched pair between the two renames** (`trigrams.bin`, then
  `content.bin`): a count mismatch gives `CacheCorrupted`, which MCP answers with a forced
  rebuild. A manifest publish point fixes it (incremental design, stage 1).
- **`src/watcher.rs` ignores `.gitignore` and reports only the first path of a rename.**
- **`src/trigram_build.rs` is missing from `build.rs`'s schema-hash list.**
- **`rfx serve` defects** (found 2026-09-28 while rewriting `docs/API.md`; listed there
  under "Known limitations"):
  1. `glob` / `exclude` are unusable: declared as lists, the query-string parser cannot
     fill a list, so any value returns 400 "expected a sequence".
  2. Unknown query parameters are ignored silently, so `include_locks`,
     `include_generated`, `exclude_text` do nothing over HTTP (the lock-file zero `hint`
     still tells callers to pass `include_locks:true`).
  4. `POST /index` knows only 11 language names; `csharp`, `ruby`, `kotlin`, `zig` are
     dropped silently (and an all-unknown list indexes everything).
  5. Invalid regex and too-broad queries return 500 `IoError` instead of 400; the
     too-broad message says `--force` instead of `force=true`.
  6. Missing `q` / bad integer / bad boolean → 400 `text/plain`, not JSON.
  7. `kind=bogus` → zero results with a misleading `contains:true` hint.
  8. `/stats` language keys are capitalized, unlike the lowercase `language` field.
- **Qt Linguist `.ts` XML files are parsed as TypeScript.** On O3DE, one such file took
  19 s of a 21 s symbol pass (issue #39). Needs a content sniff (`<!DOCTYPE TS>`).

---

## 📐 Current policy (decisions still in force)

From the 1.7.2 MCP correctness release (2026-09-22):

1. **Auto-update everywhere (decided 2026-09-29; replaces "honest staleness over
   auto-refresh" from 1.7.2 and the earlier in-process watcher plan).** Every command that
   reads the index updates it first when the freshness check says stale, and always waits
   for the update. No index → build it. Opt-out: `--no-update` on every command. Plan and
   decisions: `.context/AUTO_UPDATE_RESEARCH.md`. Not built yet.
2. **Stale always means `can_trust_results: false`**, including a zero-result search.
   Scope (did a changed file appear in the results?) only sharpens the warning text.
3. **Readers degrade, writers refuse.** Refuse only when a different released version
   owns the cache. Refusing on the schema hash alone breaks every upgrade.
4. **The symbol pass yields rather than holding the lock.** It stops at its next batch
   on a cancel sentinel, so `index_project` is never blocked for the whole pass.
5. **Text tier on by default** (`[index] text_tier = false` opts out), and exempt from
   `[index] languages`.

From 2.0.0 (2026-09-23):

- Dot-directories stay skipped unless `[index] hidden = true`; non-UTF-8 is decoded
  lossily; generated detection is by file name only.
- The dependency resolver's suffix match treats `_`/`%` literally.
- `find_references` count mode returns the count AFTER string/comment filtering;
  with `include_strings: true` it is the raw total.
- New symbol queries go into the language's `SYMBOL_QUERIES`, never a separate
  per-kind `QueryCursor` (one combined query per language per file).

---

## 🔭 Open follow-ups

Indexing and freshness:
- **Incremental index path.** In progress on `feature/incremental-index` (see the
  section at the top): a change now publishes a delta segment instead of rewriting the stores.
- The read pool is the indexing floor (3.4 s on Kubernetes, tree-sitter parsing every
  file for imports). A line-scan `#include`/`import` extractor needs an equivalence test first.
- Lexical `..` resolution instead of `canonicalize()` in `c.rs`/`cpp.rs` (resolves more
  includes; changes `rfx deps` output).
- Found during the incremental work (2026-09-29), not fixed (each changes output or
  needs its own decision):
  - The resolver-config walk ignores `[index]` include/exclude patterns (`PathPolicy`);
    kept for identical dependency rows.
  - `OpenIndex::posting_cap` is unused and the build-side cap is never applied.
  - Import resolution that reads the disk (`.exists()` in `rust.rs`, `canonicalize` in
    `c.rs`/`cpp.rs`/`zig.rs`/`ruby.rs`) can change without any indexed path changing; an
    update only sees it when a named path changes.
- `cleanup_stale` tail: instrumented, not yet measured on a large repo.
- `@generated` content marker (needs language persisted in the index).
- Bytes-per-line minified guard (deferred; V4 per-line postings bound the cost).

Output determinism (found 2026-09-29 by `benches/incremental/golden.sh`; all in 2.0.3):
- `rfx stats --json` and every `IndexStats` JSON print `files_by_language` /
  `lines_by_language` in `HashMap` order.
- `rfx deps <file> --reverse` (text) lists dependents in random order.
- `rfx deps --depth N --json` / `--format table` and MCP `get_transitive_deps` list files
  in random order (`transitive.keys()` of a `HashMap`).
- `rfx pulse map` (mermaid and d2) prints edges of equal weight in random order.
- The golden harness compares these as sets; making them sorted is a separate change.

Query latency:
- First symbol query after a re-index fills the symbol cache (Hearth `RealmId --symbols`:
  4 s cold, ~105 ms warm). Worth a background warm-up.
- The first `--symbols` call in a fresh process on a cache miss is ~60 ms (parser + query
  compilation, once per process).
- Count-mode result building (`verify`+`group` ~30 ms for 52k rows).
- Background symbol pass: once logged `128 file(s) parsed but not persisted: database is
  locked`. The single writer retries once; watch `rfx index status` `error`.

MCP:
- 1.7.0 item E: measure first-call correctness on a fresh Claude Code session with
  deferred tools, in a consumer project.
- Hybrid columnar format (file-grouped rows; REF-219) — optional.

Tests:
- `estimate_is_within_band_on_synthetic_corpus` is `#[ignore]` (builds the 30 MB corpus):
  `cargo test --release --test query_early_termination -- --ignored`.

---

## 🗂️ Backlog (not started)

### 1. Grep parity (priority)

**Goal: token parity or better with built-in Grep on Grep-like searches** (user decision,
2026-09-28). Do not route plain searches to Grep; make Reflex cheap enough for them.
Baseline (`.context/EFFICACY-2.0.3.md`): using Reflex costs 1.55–1.7× Grep's tokens, from
extra turns, not payload. Re-measure every step with `benches/efficacy/run-ref222.sh`
(~$6, 35 min on Opus 5.5); pass = token CI includes 1.0 or sits below it.

- **A. Keep the index fresh automatically.** Planned: `.context/AUTO_UPDATE_RESEARCH.md`
  (section "Auto-update" at the top). Removes both the status-check and the
  `index_project` turns once the fidelity test passes.
- **B. Shrink the tool surface, then load it eagerly.** DONE on `feature/auto-update`
  (2026-09-30): 17 → 10 tools, `tools/list` 44 KB → 10.6 KB, old names callable; measure
  with `session_bench.py` (see `AUTO_UPDATE_RESEARCH.md`). Eager loading is measured
  (2026-09-30, `AUTO_UPDATE_RESEARCH.md` "Eager schemas"): turns = Grep, cost ≤ 1.2×, tokens
  1.5–1.8× from the 44 KB schemas (search_code 7.6 KB, find_references 5.2 KB,
  search_regex 4.9 KB, list_locations 4.8 KB, count_occurrences 4.6 KB). Merge `count_occurrences` into
  `mode: "count"` and `get_dependents` into `get_dependencies`; structural tools off by
  default or one `analyze` tool; trim descriptions (~40 KB of text). Then A/B
  `"alwaysLoad": true` in the MCP config: it removes the ToolSearch turn but loads the
  schemas every turn (~17K tokens for the 17 tools today, measured 2026-09-28).
- **D. Close the gap left at equal turns (1.03–1.09×).** Strip duplicate reply metadata
  (flat `has_more` / `total_count` / `returned_count` beside `pagination`); try
  file-grouped columnar rows (REF-219) so rows do not repeat `path` / `language`.

### 2. Capabilities (as parameters on existing tools — never new tools)

Every new tool adds schema tokens to every session, which works against section 1. Each
item below is a parameter or mode on an existing tool; measure its schema cost with B.

- **Enclosing symbol on every match** — a `scope` column (`fn handle_request`,
  `impl Indexer`) from the symbol cache spans. Aim: remove follow-up `Read` turns in
  comprehension tasks (REF-225 arm B made 46 `search_regex` calls). Adds bytes per row;
  measure with the REF-225 design, not only REF-222.
- **Many patterns in one call** — `patterns: [...]` on `search_code` / `search_regex`,
  one result block per pattern. Aim: fewer turns in multi-search comprehension work.
- **Co-occurrence** — files that contain all of A and B and none of C, optionally within
  N lines, as `search_code` parameters. Posting-list intersection makes it cheap.
- **Changed files only** — `changed_since: <ref>` (MCP) / `--changed`, `--since <ref>`
  (CLI), with the file list from `src/git.rs`. For review: "did my change leave callers?"
- **Callers of callers** — `depth` on `find_references` (1–3), built on the enclosing
  symbol. Aim: transitive "who calls this" in one call.
- Deferred until a test shows the need: file outline (a `search_code` mode with
  `symbols: true` + `file`); test locator (a test-path filter on `find_references`).

### 3. Other

- Advanced dependency path resolution: `package.json` workspaces, Cargo workspace members,
  Python virtualenv paths.
- One query across several indexed repositories (a `repo` column).
- `reflexd` background daemon.
- LSP adapter.

---

## 🏛️ Architecture decisions (short history)

1. **2025-10-31 — full-text, not symbol-only.** Reflex is a trigram-indexed full-text
   search engine. The symbol-only index found 1 of 8 occurrences of a name.
2. **2025-11-03 — no symbols in the main index.** `symbols.bin` was removed; symbols come
   from tree-sitter on the candidate files only.
3. **2025-11-09 → 2.0.0 — background symbol cache.** `rfx index` spawns
   `rfx index-symbols-internal`, which stores zstd symbol blobs in `meta.db`
   (`SYMBOL_FORMAT_VERSION` 3). Symbol queries read the cache first and parse misses on demand.
4. **2.0.0 — V4 trigram format, content-based freshness, tracked-file coverage.**
   See CHANGELOG.md 2.0.0 and `.context/BINARY_FORMAT_RESEARCH.md`.
