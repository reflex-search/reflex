# Agent edge plan: make Reflex a clear win for coding agents

**Written:** 2026-11-09, on Reflex 2.3.1 (`main` `b05d2e7`). **Status:** planned; decisions
D1–D7 made 2026-11-09; no task started.
**Source:** the strategy review (artifact "Reflex next bets", 2026-11-09) and the files it cites:
`GREP_PARITY_RESEARCH.md`, `AUTO_UPDATE_RESEARCH.md`, `EFFICACY-2.0.3.md`, `TODO.md`.

Check items off as they land (`- [x]`). When a phase is done, move its record to
CHANGELOG.md and delete it here (the `.context/README.md` rule). Measurements go in
`.context/AGENT_EDGE_RESEARCH.md` (created by task 0.6), dated, with the version.

---

## Why

- On plain searches Reflex ties Grep: 1.02–1.06× Grep's cost, accuracy equal
  (Sonnet 5.5 and Opus 5.5, 2026-10-25 … 10-31).
- The rest of that gap is fixed text (~7.8K chars per call). Grep stays listed, so Reflex
  cannot get below Grep on a plain search. The last three trims each saved ~1 %, below
  the noise.
- The one clear win is code-only uses: `aware-large` with routing text, 0.64–0.69× Grep's
  cost, accuracy 1.000 against 0.972 (Sonnet 5.5, 2026-10-29).
- Agents spend most tokens on `Read` and on finding the right lines, not on search.
  Grep cannot help there. Reflex knows each symbol's span, so it can.

**Goal:** on three headline workflows (explain a function, change a signature safely,
find the tests for a change), Reflex beats both built-in Grep and the LSP tool.

**Kill criterion (decided before work starts, see D5):** if Phase 8 shows no clear win on
at least 2 of the 3 headline workflows against both Grep and LSP, archive Reflex as a
fast local search library and stop feature work.

---

## Decisions

All seven decided by the user on 2026-11-09: "go with your recommendations on all of
them". Each item's **Decided** line is binding; the options are kept for context.

- [x] **D1 — Schema budget** — **Decided 2026-11-09: (a) + (c).** Raise the
  `tests/mcp_tool_surface.rs` cap to 7,000 bytes; every new listed parameter must show a
  measured win before it merges; Phases 5–6 share one `relation` enum parameter. (before Phase 3). `tests/mcp_tool_surface.rs:107` caps
  `tools/list` at 6,000 bytes; today it is ~5,938 (62 bytes left). Every new parameter
  costs ~80–200 bytes on every call. Options:
  (a) raise the cap to 7,000 bytes, and each new parameter must show a measured win;
  (b) new capabilities only in the CLI + skill, MCP unchanged;
  (c) one shared `relation` enum parameter instead of several booleans (Phases 5–6).
  Recommendation: (a) + (c).
- [x] **D2 — Primary delivery** — **Decided 2026-11-09: the measurement decides.** After
  task 0.6, compare arm `S` (CLI + skill) with `Beager`; the cheaper arm at equal accuracy
  becomes the primary route. Record the result here. Until then, build both (Phase 1).
  **Result (2026-11-10): CLI + skill + SessionStart hook (the `reflex` plugin) is the
  primary route for Claude Code;** `reflex-mcp` stays for clients without a shell. Arm S
  cost 0.68–0.95× Beager's in all 7 sessions measured, accuracy 1.000 in both (aware:
  S 0.80× Grep's cost, Beager 0.95×; investigate: S 0.98×, Beager 1.07×). The skill alone
  never fired; the hook is required. Numbers: `.context/AGENT_EDGE_RESEARCH.md`.
- [x] **D3 — Pulse** — **Decided 2026-11-09: freeze in this repo.** No new milestones;
  its "Left" items move under a "Frozen" heading in `TODO.md`.
- [x] **D4 — Rename** — **Decided 2026-11-09: dry-run diff only.** `rfx rename` never
  writes source files; agents apply the diff with `git apply`.
- [x] **D5 — Kill criterion numbers** — **Decided 2026-11-09.** "Clear win" =
  ≤ 0.85× the other arm's cost at equal or better accuracy, or an accuracy gain of
  ≥ 0.05 at ≤ 1.0× cost. Reflex must clearly win against both `A` (Grep) and `L` (LSP)
  on at least 2 of the 3 headline workflows; otherwise archive (Phase 8).
- [x] **D6 — Outline home** — **Decided 2026-11-09: (a)** `search_code`
  `mode: "outline"`, `pattern` = file path or fragment. Options were:
  (a) `search_code` `mode: "outline"`, with `pattern` = file path or fragment;
  (b) `get_dependencies` `outline: true` (that tool already takes a `path`);
  (c) a new tool (against current policy).
  (a) adds one enum value, and `pattern` stays required.
- [x] **D7 — Agent-irrelevant surfaces** — **Decided 2026-11-09: freeze now.** No new
  work on `rfx ask`, interactive mode or the `rfx serve` defects. Deprecation is a later
  decision, not part of this plan.
- [x] **D8 — Servers and an older cache** — **Decided 2026-11-09 (user: "yes"): servers
  rebuild a cache written by an OLDER release; they still refuse a NEWER one.** Changes
  policy 3 in `TODO.md`. Found in task 1.6.

---

## Ground rules (every item)

- **No new listed MCP tools.** New capabilities are modes or parameters on the five
  listed tools (`search_code`, `list_locations`, `find_references`, `get_dependencies`,
  `analyze`). Every listed byte is paid on every call.
- **CLI parity.** Every capability is reachable from `rfx` with `--json` in the same
  shape as the MCP answer. The skill route (task 1.3) depends on it.
- **One combined tree-sitter query per language per file.** New captures go into the
  language's `SYMBOL_QUERIES`, never a separate `QueryCursor` (current policy).
- **Additive output.** New fields are `Option` with `skip_serializing_if`, so today's
  answers keep their bytes. `SymbolRef` and `SearchResult` stay additive-safe.
- **Freshness contract holds.** Every new JSON-object answer carries `status` and
  `can_trust_results`, and goes through the auto-update chokepoints
  (`cli::update_before`, `handle_call_tool`).
- **Deterministic.** Same index + same query → same answer. Sort every list.
- **Honest limits.** Call graphs and impact are name-based, not type-resolved. Say so in
  `warnings` when a name has more than one definition (`ambiguous: true`).
- **Live benchmark runs cost plan usage.** Ask the user before each run. Prefer offline
  analysis of saved transcripts.

### Definition of done (every feature)

- [ ] Unit tests, plus an integration test in `tests/`; a fixture per language where the
  feature is language-specific
- [ ] `tests/mcp_tool_surface.rs` passes (or D1 is changed on purpose)
- [ ] `tests/latency_budget.rs` shapes no worse than +10 %
- [ ] CLI `--json` and MCP shapes match and are documented
- [ ] CHANGELOG.md entry; CLAUDE.md section; `docs/mcp-tool-cheatsheet.md`
- [ ] A dated measurement in `AGENT_EDGE_RESEARCH.md`

---

## Phase order

```
0 Measurement ──┬─> 8 Headline benchmark
1 Distribution ─┤
2 Foundation ───┼─> 3 Understand ─┬─> 5 Call graph ─> 6 Tests for X
   2.4 graph ───┼─> 4 Impact ─────┘
                └─> 7 Rename, unused symbols, placeholder patterns
```

Phases 0, 1 and 2 can run in parallel. Phase 4 needs 2.2 and 2.4. Phase 6 needs 5 and 2.4.

---

## Phase 0 — Measurement first

Without these, no feature can show a win. Answer keys must not come from Reflex.

- [ ] **0.1 LSP arm `L`** in `benches/efficacy/runner.py` `ARMS`. Claude Code's `LSP`
  tool with a language server per corpus (rust-analyzer for tokio). Grep stays allowed,
  as in the Reflex arms. Record the server's warm-up time apart from session cost.
  Document the setup in `benches/efficacy/README.md`.
- [ ] **0.2 Skill arm `S`**: no MCP server; the Reflex skill (task 1.3) and Bash `rfx`.
  Needs 1.3 and 1.4.
- [ ] **0.3 `explain` session kind** in `session_bench.py`: "What does `X` do, what does it
  call, and what calls it?" Score the callers and callees lists. Answer key:
  rust-analyzer call hierarchy, checked by hand on a sample.
- [ ] **0.4 `impact` session kind**: the harness applies a scripted signature change
  (add a parameter, remove a function) on a scratch branch and asks "which call sites
  must change?" Answer key: `cargo check` errors. The compiler is the oracle.
- [ ] **0.5 `tests` session kind**: "Which tests exercise `X`?" Answer key: mutation.
  Make `X` panic, run the suite, and the failing tests are the truth. Compute it once
  per question and save it under `benches/efficacy/corpus/`.
- [ ] **0.6 Baseline runs** (ask first): arms `A`, `Beager`, `L` on `explain`, `impact`,
  `tests` and `aware-large`, tokio. Create `.context/AGENT_EDGE_RESEARCH.md` with the
  numbers.
- [ ] **0.7 Large corpus.** Add one repo where `rg` takes seconds (candidates: the Linux
  kernel, Chromium without third_party). Record `rg` and `rfx query` wall time for the
  `investigate` shapes. Add it to `benches/efficacy/repos.md`.

---

## Phase 1 — Distribution and dogfooding

- [x] **1.1 Fix this repo's `.mcp.json`.** Done 2026-11-09 (uncommitted): `.mcp.json`
  runs `rfx mcp`; CONTRIBUTING.md step 6; local `rfx` 2.0.3 → 2.3.1 by
  `cargo install --path . --locked`. Verified: `initialize` → `reflex 2.3.1`, five tools
  listed, `list_locations` → `fresh`, `can_trust_results: true` (after 1.6's manual
  `rfx index`). Was: It runs `./node_modules/.bin/rfx`, which does
  not exist, so the `reflex` server fails to connect when you work on Reflex. Point it
  at `rfx` on PATH; document `cargo install --path . --locked` for dev builds in
  CONTRIBUTING.md (Development Setup, step 6).
- [x] **1.2 Version drift.** Done 2026-11-09 (branch `feat/agent-edge-phase1`):
  `auto_update::startup_index_line`, printed on stderr by `rfx mcp` (after the existing
  `reflex-mcp startup:` line, which already had `version=` and `build=`) and by
  `rfx serve`. States: no index, unstamped, this version, another build of it, older
  (rebuilt), newer (`WARNING`), `--no-update`. Unit test
  `startup_index_line_names_the_writer_and_the_outcome`; checked on real caches. Was: The `rfx` on PATH here is 2.0.3; the repo is 2.3.1. Add a
  line to `rfx mcp` startup logs (stderr) with the binary version and the index's
  writer version, so a stale install is visible.
- [x] **1.3 Claude Code plugin + skill.** Done 2026-11-09 (2.3.1, branch
  `feat/agent-edge-phase1`). The repo root is a marketplace
  (`.claude-plugin/marketplace.json`, name `reflex`) with two plugins, so arm S and the
  MCP arm can be installed apart (D2):
  - `plugins/reflex`: `skills/reflex/SKILL.md` only. The CLI call shapes, the matching
    and coverage rules, and the MCP routing sentences (prefer-first, "not natural
    language", retry-once fallback). `claude plugin details`: ~104 tokens always-on,
    ~1.3k when it fires (the MCP server: ~7.8K chars of instructions + schemas per call).
  - `plugins/reflex-mcp`: `mcpServers.reflex` = `rfx mcp`, with
    `REFLEX_MCP_TOOL_PREFIX=mcp__plugin_reflex-mcp_reflex__`. Claude Code lists the
    server as `plugin:reflex-mcp:reflex` (checked with `claude --plugin-dir … mcp list`),
    so the default ToolSearch line (`select:mcp__reflex__…`) named tools that do not
    exist. The server now rewrites the prefix (`with_tool_prefix`, `src/mcp.rs`).
  - Both say to install `rfx` from npm `reflex-search`; no binary in the plugin.
  - Versions: only in each `plugin.json` (not in the marketplace entries);
    `release.toml` `pre-release-replacements` bumps them; `tests/plugin_manifest.rs`
    checks them against `CARGO_PKG_VERSION`, the prefix, and the skill's routing sentences.
  - `claude plugin validate` passes on all three manifests. Warning only: no icon.
  - Code-only uses: the skill says the CLI does not split code from comments yet
    (1.4 adds `rfx refs`, then the skill routes code-only questions to it).
- [x] **1.4 CLI JSON parity audit.** Done 2026-11-09 (2.3.1, branch
  `feat/agent-edge-phase1`). Added:
  - `rfx refs <name>` = `find_references`, `rfx locations <pattern>` (alias `loc`) =
    `list_locations`. Both run the MCP tool (`mcp::call_tool_data`, the body of
    `tools/call` split out as `call_tool_inner`), so `--json` is the MCP answer value
    for value (`tests/cli_tools.rs` compares them). Text form: `rfx refs` prints
    `definition path:line kind name`, then `path:line:text`; `rfx locations` prints
    `path:4,128,193` per file (`--preview`: `path:line:text`). On this repo,
    `update_if_stale`: `rfx locations` 209 bytes, `rfx query` text 5,019, `--json` 2,624.
  - `status` / `can_trust_results` on `rfx query --count --json` and on every
    `rfx analyze --json` answer (incl. the empty-result ones), as the MCP answers.
  - `rfx deps --reverse --depth N` (N ≠ 1) is refused, as `get_dependencies` refuses
    the pair (the depth was ignored without a word).
  - A read command (`query`, `refs`, `locations`, `stats`, `list-files`, `analyze`,
    `deps`, `context`) run in a directory with no `.reflex/` uses the nearest parent
    index (`.reflex/meta.db`), never past a `.git` (`cli::enclosing_index_root`);
    stderr says so; a relative `deps` path is rebased. `index`, `watch` and the servers
    keep working where they are started.
  - The skill routes "where" to `rfx locations` and code-only to `rfx refs` (the MCP
    routing sentences, pinned in `tests/plugin_manifest.rs`).

  Parity table (MCP tool → CLI; "same" = identical JSON):

  | MCP | CLI | JSON |
  | --- | --- | --- |
  | `search_code` list | `rfx query X --json` | differs: CLI `results[]` grouped by file; MCP columnar `{columns, rows}` |
  | `search_code` `mode:"count"` | `rfx query X --count --json` | same fields + `timing_ms` |
  | `search_code` `paths:true` | `rfx query X --paths --json` | differs: CLI a bare array, no `status`; MCP `{status, can_trust_results, paths, total_files}` |
  | `search_code` `symbols` / `regex` / `kind` | `--symbols` / `--regex` / `--kind` | as list |
  | `list_locations` | `rfx locations X --json` | same |
  | `find_references` | `rfx refs X --json` | same |
  | `get_dependencies` | `rfx deps F --json` | differs: CLI `{import_type, line, path (raw import), symbols}`; MCP `{line, path (resolved file)}` |
  | `get_dependencies` `reverse` | `rfx deps F --reverse --json` | same |
  | `get_dependencies` `depth` | `rfx deps F --depth N --json` | not compared |
  | `analyze` hotspots / circular / unused / islands | `rfx analyze --<kind> --json` | same (CLI islands adds `filter`) |
  | `analyze` summary | `rfx analyze --json` | differs: CLI has more fields (`islands` object, `vendored_files`, `warnings`) |
  | `search_ast` | `rfx query P --ast --json` | not compared |
  | `gather_context` | `rfx context` | prose |
  | `index_project` | `rfx index` | n/a |
  | `check_index_status` | none (`rfx index status` is the symbol pass) | gap |

  Left open (changing them breaks a public CLI JSON shape; decide before changing):
  the `query --json` and `--paths --json` shapes, the forward `deps` shape, a CLI
  `check_index_status`. Also: a subdirectory search covers the whole index (rg would
  search the subtree only); `list_locations` gives no zero-result `hint`
  (`search_code` does).
- [x] **1.6 An upgrade leaves `rfx mcp` stale until a manual `rfx index`.** Fixed
  2026-11-09 (branch `feat/agent-edge-phase1`): `auto_update::SelfHeal`
  (`Never` library / `OlderOnly` servers / `Any` CLI) and `is_older_release`
  (MAJOR.MINOR.PATCH; a suffix on equal numbers or an unparsable version is not older).
  Two paths were broken, not one: the `Foreign` plan (format changed), and the same
  format with another version's stamp, where `assert_writable` refused the
  `update_paths` / `index` run (this one hit the CLI's auto-update too). Both now go
  through `take_over` (clear + build; skips the clear when another process of this
  version already rebuilt it). A fresh older cache is not rebuilt. Refusal text names
  `rfx index`, never "force". Tests: 3 unit (`src/auto_update.rs`), 5 integration
  (`tests/auto_update.rs`, incl. the field case). Found
  2026-11-09 doing 1.1: `.reflex/` written by 2.2.0, binary 2.3.1. Every MCP search
  answered `status: stale`, `can_trust_results: false`, `action_required: null`, warning
  "automatic index update failed: this .reflex/ was written by reflex 2.2.0 (sha
  89acce5); this binary is 2.3.1. Rebuild with force, or run the matching binary."
  Cause: policy 3 in `TODO.md` ("readers degrade, writers refuse"; servers skip another
  version's cache, the CLI rebuilds it). So every user who updates the npm package gets
  untrusted answers forever, and the agent cannot fix it (`index_project` is unlisted;
  "force" names nothing it can call). **Needs decision D8.** Proposal: a server rebuilds
  when the cache's writer is an OLDER release (an upgrade) and refuses only when it is
  NEWER (a downgrade, two binaries fighting). Until then, the warning names `rfx index`
  and `action_required` is set.
- [x] **1.5 Install smoke test.** Done 2026-11-10 (npm `reflex-search` 2.3.1):
  `scripts/install-smoke.sh`, results in `AGENT_EDGE_RESEARCH.md`. On
  `node:24-bookworm`: npm install 5.88 s, first CLI query (builds the index of 451
  files) 1.08 s, first MCP call 1.02 s, no manual step; the plugins install from a
  local marketplace in 2.08 s without a login. Found:
  - the npm `rfx` is a Node wrapper: +~240 ms on every call (query 270–343 ms vs
    26–59 ms native). The skill route pays it per Bash call. The shell installer
    (`reflex-search-installer.sh`, 2.54 s) gives the native binary; the README and the
    skill now name it;
  - npm 12 blocks `postinstall`; the wrapper then downloads the binary at the first
    run (1.83 s). Works, with a warning;
  - Alpine (musl) failed: no musl target in `dist-workspace.toml`. **Done
    (2026-11-10)**: `x86_64-` and `aarch64-unknown-linux-musl` added. A local static
    build runs in `node:24-alpine`; the npm package maps musl to it. The first release
    proves the CI build.
  - Not tested: `claude plugin marketplace add reflex-search/reflex` (needs the merge).

---

## Phase 2 — Foundation

### 2.1 Exact code / comment / string split

Today `find_references` uses a lexer (`line_filter::FileRegions`, `src/mcp.rs:68–93`).
Known gaps: Ruby `%q{}` and heredocs, PHP heredocs, C# `$$"…"` raw strings, Rust
format-string captures, HTML comments in Vue/Svelte templates.

Spec:
- The symbol pass collects comment and string-literal node ranges from the tree it
  already builds, per language (a node-kind table per grammar). Interpolation children
  (`${…}` in template strings, f-string `{…}`) count as code.
- Store them as sorted line/byte intervals in the symbol blob (format v4, see 2.2).
  A cache miss parses the file on demand, as symbols do today.
- `reference_is_non_code` and the line filter in `src/query/mod.rs:2225` read the tree
  intervals when the file has a grammar. The lexer stays as the prefilter and as the
  fallback for files with no grammar (text tier, Swift).

Tasks:
- [ ] Node-kind table for the 15 languages (in `AGENT_EDGE_RESEARCH.md`)
- [ ] Collect intervals in the symbol pass; store them in the v4 blob
- [ ] `find_references` reads tree intervals; lexer fallback kept
- [ ] Equivalence test over every fixture: list each lexer/tree disagreement; turn each
  known gap above into a regression test
- [ ] `aware-large` answers stay 1.000; `latency_budget` within +10 %

### 2.2 Symbol facts: scope, signature, columns

Today a cached symbol has `kind`, `symbol` (name), `span` (lines only; `Span::new` drops
columns) and `preview` (`src/models.rs:451`).

Spec — new optional fields on `SearchResult`, stored in the blob:
- `parent: Option<u32>`: index of the nearest enclosing symbol in the same file's list.
  The query side builds the `scope` string from it, e.g. `impl Indexer › fn update_paths`.
- `signature: Option<String>`: the declaration up to the body (`{` in brace languages,
  `:` in Python), whitespace collapsed, at most 240 chars.
- `name_col: Option<u32>`: 1-based byte column of the name identifier.
- `body_hash: Option<u64>`: hash of the body text (Phase 4 uses it to find renames).

Tasks:
- [ ] Bump `SYMBOL_FORMAT_VERSION` 3 → 4 (`src/symbol_cache.rs:33`); the existing
  version check rebuilds the cache
- [ ] Fill the fields in every extractor (`src/parsers/*.rs`)
- [ ] Per-language fixture tests for `parent` and `signature`
- [ ] Measure blob size and symbol-pass time on Kubernetes (baseline 3.4 s on 8 threads,
  2026-09-23); budget +15 %

### 2.3 Call and relation captures

Spec — two more lists per file in the v4 blob:
- **calls**: `(callee_name, line, col, enclosing symbol index)` for each call site.
  Captures per grammar: Rust `call_expression` / method calls / macro invocations, Python
  `call`, Go `call_expression`, Java `method_invocation`, TS/JS `call_expression` and
  `new_expression`, C# `invocation_expression`, and so on.
- **relations**: `(child type, parent type, kind)` with kind `implements`, `extends` or
  `impl_trait`. Rust `impl Trait for Type`; Java, Kotlin, C#, TS, PHP `extends` /
  `implements`; Python class bases; Ruby `<`; C++ base clauses. Go is left out
  (interfaces are satisfied structurally).

Tasks:
- [ ] Add the captures to each language's `SYMBOL_QUERIES` (no new `QueryCursor`)
- [ ] Store and read both lists; fixture tests per language
- [ ] Measure the symbol pass again (same +15 % budget as 2.2, combined)

### 2.4 Import graph accuracy

Today Reflex finds 304 of 417 tokio edges (73 %); C/C++ includes resolve 62–64 %.

- [ ] Rust: `use` inside `cfg_*!` macro bodies (re-parse the token tree)
- [ ] Rust: items imported from `lib.rs` / a parent `mod.rs` (re-exports)
- [ ] Rust: every module of a `{…}` use group
- [ ] tokio answer key (`GREP_PARITY_RESEARCH.md`) at ≥ 95 %
- [ ] Stretch: C/C++ include paths from `compile_commands.json` and CMake
  `target_include_directories`

---

## Phase 3 — Understand: outline, body, scope

### 3.1 File outline

- MCP (D6 option a): `search_code` `mode: "outline"`; `pattern` is a file path or
  fragment. An ambiguous fragment returns an error that lists the candidates.
- CLI: `rfx outline <path> [--json]`.
- Answer: `{path, columns: ["line", "end_line", "kind", "name", "scope", "signature"],
  rows, has_more, status, can_trust_results}`. Local variables are left out. Cap 500 rows.
- Text-tier files: an error that says the file has no grammar.

Tasks:
- [ ] Mode, CLI command, tests
- [ ] One routing sentence in `MCP_INSTRUCTIONS`: outline before reading a large file;
  pin it in `test_instructions_keep_the_routing_rules`

### 3.2 Symbol body

`search_code` already has a hidden `expand` parameter ("Whole symbol body",
`HIDDEN_PARAMS`), which no agent passed in 5,914 calls. Build on it; do not add a twin.

- MCP: `find_references` `body: true` puts `text` (the definition's exact lines) on each
  definition. Cap 300 lines; past the cap, `truncated: true`. At most 5 definitions.
- CLI: `rfx show <name> [--file <fragment>] [--json]`.

Tasks:
- [ ] Share one body reader between `expand` and `body`
- [ ] `body` on `find_references`; CLI command; tests
- [ ] Routing sentence: "to understand X, call find_references with body:true"

### 3.3 Scope on every match

- `find_references` references and `search_code` rows get a `scope` column (from 2.2)
  when the file has cached symbols. `list_locations` keeps its `{path: [line]}` shape.
- Default on for `find_references`. For `search_code`, measure on against off
  (bytes per row against `Read` turns saved), then decide and record it here.

Tasks:
- [ ] `scope` column in both answers; tests
- [ ] Offline estimate of the bytes from saved transcripts; then one live A/B (ask first)

---

## Phase 4 — Impact of a diff

"What did my change break?" Grep cannot answer it in one step. LSP servers do not answer
it at all.

- MCP: `analyze` `kind: "impact"`, with `since` (a git ref; default `HEAD`, which means
  the uncommitted changes).
- CLI: `rfx impact [--since <ref>] [--json]`.

Algorithm:
1. Changed files: `git::diff_names` (`src/git.rs:346`) plus the worktree changes and
   untracked files.
2. Hunks: `git diff -U0 <since>` line ranges, old and new side.
3. Old symbols: parse `git show <since>:<path>` with the same extractors (changed files
   only; not cached). New symbols: the symbol cache.
4. Pair old and new symbols by `(scope, name, kind)`. Classify each: `removed`,
   `added`, `signature_changed`, `body_changed`. An unpaired removed + added pair with
   the same `body_hash` is `renamed` (marked as a guess).
5. For `removed`, `renamed` and `signature_changed`: code-only references across the
   index (the `find_references` path). Mark each `in_diff: true/false`.
6. Deleted or moved files: their importers from `import_edges`.
7. Phase 6 adds the tests that touch each changed symbol.

Answer: `{since, changed_files, symbols: [{name, kind, path, change, old_signature,
new_signature, references: {path: [line, ...]}, outside_diff}], broken_importers,
warnings, status, can_trust_results}`.

Tasks:
- [ ] Hunk parser and old-side symbol extraction
- [ ] Pairing and classification; tests with fixture repos (rename, remove,
  signature change, body-only change, file move)
- [ ] `analyze kind:"impact"` + CLI; schema bytes checked against D1
- [ ] Score on the `impact` session kind (0.4)

---

## Phase 5 — Call graph

One enum parameter on `find_references` (D1 option c):
`relation: "callers" | "callees" | "implementations"`, with `depth` (1–3) for callers.

- **5.1 callers**: references grouped by enclosing symbol (from 2.2). With `depth` > 1,
  the callers of those callers, as a tree. Node cap 200; a cycle is marked, not followed.
- **5.2 callees**: the call sites inside the definition's span (from 2.3). Each callee
  resolves by name, preferring the same file, then imported files (2.4). Unresolved
  names are listed as `external`.
- **5.3 implementations**: for a trait, interface or class name, the types that
  implement or extend it (relation rows from 2.3).

Tasks:
- [ ] `relation` + `depth` parsing, MCP and CLI (`rfx refs <name> --relation callers --depth 2`)
- [ ] Callers tree; callees with resolution; implementations
- [ ] `ambiguous` flag and `warnings` when a name has several definitions
- [ ] Score on the `explain` session kind (0.3)

---

## Phase 6 — Tests that touch X

- MCP: `find_references` `relation: "tests"`. CLI: `rfx refs <name> --relation tests`.
- A test is a test symbol (rules below) that references X directly, or through callers up
  to depth 2. Test files that import X's file are added with `via: "import"`.
- Test rules per language: test files by `is_test_file_name` (`src/mcp.rs:132`); test
  symbols by Rust `#[test]` / `#[tokio::test]` and `#[cfg(test)] mod`; Python `test_*`
  and `Test*`; Go `func TestX(t *testing.T)`; JS/TS `it(` / `test(` / `describe(`;
  Java and Kotlin `@Test`; C# `[Fact]` / `[Test]` / `[TestMethod]`; Ruby `it` /
  `def test_`; PHP `test*` methods and `@test`.
- Answer: `{tests: [{path, name, line, via}], run_hint}`. `run_hint` is one command per
  ecosystem when it can be built (`cargo test <name>`, `pytest <path>::<name>`,
  `go test ./<pkg> -run '^TestX$'`); otherwise it is left out.
- `analyze kind:"impact"` lists the tests for each changed symbol.

Tasks:
- [ ] Test-symbol rules and fixtures per language
- [ ] `relation: "tests"`; `run_hint`; impact integration
- [ ] Score on the `tests` session kind (0.5)

---

## Phase 7 — Rename plan, unused symbols, placeholder patterns

- [ ] **7.1 Rename plan** (D4). CLI `rfx rename <old> <new> [--glob …] [--include-strings]`
  prints a unified diff: whole-identifier code occurrences replaced at exact columns.
  Hits in strings and comments are listed apart and left unchanged unless
  `--include-strings`. No MCP surface; agents run it through Bash and `git apply` it.
- [ ] **7.2 Unused symbols.** `analyze kind:"unused"` with `symbols: true`: symbols with no
  code reference outside their own span. Left out or flagged: public API of library
  crates (`exported: true`), trait impls, `main`, tests, `#[no_mangle]`, framework
  entry points. Each row has a `confidence`.
- [ ] **7.3 Placeholder patterns — spike first.** Try the `ast-grep-core` crate: check that
  its tree-sitter version fits ours. Patterns such as `$X.unwrap()`: literal fragments →
  trigram candidates → structural match on those files only. Go if a pattern with a
  3-char literal runs in ≤ 1 s on Kubernetes. Home: `search_ast` (hidden tool) when the
  pattern holds `$`. Record the spike in `AGENT_EDGE_RESEARCH.md`.

---

## Phase 8 — Headline benchmark and README

- [ ] Run `explain`, `impact`, `tests` and `aware-large` on tokio and the large corpus
  (0.7); arms `A`, `L`, `Beager`, `S` (ask first)
- [ ] Apply the kill criterion (D5); record the verdict and the date at the top of this file
- [ ] If it passes: README headline with the three workflows and their numbers; update
  `docs/mcp-tool-cheatsheet.md` and the CLAUDE.md MCP table
- [ ] If it fails: write the archive note (what Reflex is good for as a library) and stop

---

## Scope freeze

- [ ] **Grep-parity trimming stops.** Close backlog §1 D in `TODO.md` with a pointer here.
- [ ] **Pulse** (D3, freeze in this repo): no new milestones. Move its "Left" items under
  a "Frozen" heading in `TODO.md`.
- [ ] **`rfx ask`, interactive mode, `rfx serve` defects** (D7, freeze): no new work. The
  `serve` defect list stays in `TODO.md` as known limitations.
