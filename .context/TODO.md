# Reflex TODO

**Last Updated:** 2026-09-28 (Reflex 2.0.3)

> **⚠️ AI Assistants:** Read the "Context Management & AI Workflow" section in `CLAUDE.md`.
> Update this file as you work. Keep it to LIVE work: open tasks, current policy, open
> follow-ups. Finished work belongs in CHANGELOG.md and git history, not here.

> **History:** the pre-2026-09 roadmap (MVP plan, per-module task lists, 2025 status
> summaries, old benchmarks) was removed on 2026-09-28 because it described an
> architecture that no longer exists. Read it with `git show fc8da6b:.context/TODO.md`.

---

## 🧹 Stale-claims cleanup + efficacy rerun (2026-09-28) — IN PROGRESS

Branch `docs/remove-stale-claims`. Plan: `~/.claude/plans/warm-crafting-pretzel.md`.
Removes pre-1.7 measurements and outdated behaviour descriptions from CLAUDE.md, README,
docs/ and `.context/`, then reruns the efficacy A/B on 2.0.3 with `claude-sonnet-5`.

| Part | Scope | Status |
| --- | --- | --- |
| C | `.context/` rewrite (this file, research files) | completed |
| B | CLAUDE.md, README.md, docs/ | in progress |
| B4 | Source text: MCP descriptions, cache.rs comments, dead config keys, `--format dot` help | pending |
| D1 | Efficacy harness fixes (pinned corpus, model flag, isolation) | pending |
| D2 | Rerun REF-222-style and REF-225-style A/B + columnar payload script | pending |
| D3 | Publish `.context/EFFICACY-2.0.3.md` + README/CLAUDE.md numbers | pending |

Why: the old A/B numbers (REF-222/192/204/209/225) ran on 1.5.3-dev in 2026-07. The README
"~31% cheaper" came from REF-192 (n=3), not REF-222, whose own cost ratio was 1.169
(Reflex cost more). "Within the ±10% parity band" was false (CI upper bound 1.262).

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
- **`max_posting_list_entries` (500k, `src/models.rs:590`) silently drops files past the
  cap.** Found in the 2026-09-22 latency round.
- **Qt Linguist `.ts` XML files are parsed as TypeScript.** On O3DE, one such file took
  19 s of a 21 s symbol pass (issue #39). Needs a content sniff (`<!DOCTYPE TS>`).

---

## 📐 Current policy (decisions still in force)

From the 1.7.2 MCP correctness release (2026-09-22):

1. **Honest staleness over auto-refresh.** A search on a stale index answers with
   `can_trust_results: false` and `action_required: "index_project"`; it never reindexes
   by itself. The 1.7.2 note said auto-refresh would sit behind `REFLEX_MCP_AUTO_INDEX=1`;
   that variable was never built. The original reason (a full rebuild takes minutes) is
   weaker since 2.0.0 (Kubernetes indexes in ~8 s), but any change still rewrites
   `content.bin`/`trigrams.bin` in full. Revisit when an incremental index path exists.
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
- **Incremental index path.** Any change still rewrites `content.bin`/`trigrams.bin` in
  full. Prerequisite for auto-refresh (policy 1).
- The read pool is the indexing floor (3.4 s on Kubernetes, tree-sitter parsing every
  file for imports). A line-scan `#include`/`import` extractor needs an equivalence test first.
- Lexical `..` resolution instead of `canonicalize()` in `c.rs`/`cpp.rs` (resolves more
  includes; changes `rfx deps` output).
- `batch_update_files_and_branch` SELECTs each id after insert (0.9 s on Kubernetes);
  `RETURNING id` would trim it.
- `cleanup_stale` tail: instrumented, not yet measured on a large repo.
- `@generated` content marker (needs language persisted in the index).
- Bytes-per-line minified guard (deferred; V4 per-line postings bound the cost).

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

- `reflexd` background daemon.
- LSP adapter.
- Branch-aware search: `--since <ref>` / `--changed` (search only files changed vs a ref).
- Advanced dependency path resolution: `package.json` workspaces, Cargo workspace members,
  Python virtualenv paths.
- Query result caching (LRU). Measure first: `rfx mcp` already keeps the index open.

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
