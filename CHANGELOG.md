## [Unreleased]

### Added

- **Every command answers from an up-to-date index; `rfx index` is no longer needed
  after an edit.** `rfx query`, `rfx deps`, `rfx analyze`, `rfx stats`, `rfx context`,
  `rfx list-files`, `rfx ask`, `rfx snapshot`, `rfx pulse`, interactive mode, `rfx mcp`
  tools and `rfx serve` compare the tree with the index (the same check behind
  `status` / `can_trust_results`) and, when it is stale, update it before they answer:
  only the changed files for an edit, a full incremental run for many changes or an
  ignore-file / `.reflex/config.toml` edit. A search runs alongside the check, so a fresh
  index costs nothing extra; a stale one is updated and searched again, and the answer
  reports `status: fresh`. A missing index is built. A command waits for another index
  run rather than answer from a stale index.
- **`--no-update`** (every command, before or after the subcommand; also
  `rfx mcp --no-update`, `rfx serve --no-update`) answers from the index as it is, as
  before, including the "Index not found" error.
- If an update cannot run (a read-only `.reflex/`, a cache another rfx version wrote in
  `rfx mcp` / `rfx serve`), the command still answers, from the index it has, marked
  `stale`, with the reason in `warnings`. The CLI rebuilds another version's cache, as
  `rfx index` does; the servers do not (two servers of different versions would rebuild
  it in turns).
- `--timing` / `REFLEX_MCP_TIMING=1`: `timings.update_us` when an update ran.
- **MCP: agents are no longer told to call `check_index_status` or `index_project`.** The
  server instructions and every tool description now say the index is updated before
  each call; `index_project` is for a forced rebuild. Every JSON-object answer carries
  `status` and `can_trust_results`: `list_locations`, `count_occurrences`,
  `find_references`, `mode: "count"` and the structural tools lacked
  `can_trust_results` (array answers and `get_transitive_deps` still have none).
- `rfx index` remembers `--languages`; later automatic updates index the same languages.
  A plain `rfx index` clears it.

### ⚠️ Breaking (MCP)

- **`rfx mcp` lists 10 tools instead of 17, and their schemas are a quarter of the
  size** (`tools/list` 44 KB → 10.6 KB). Claude Code carries every listed schema on every
  turn; in long sessions that prefix was the whole token gap between Reflex and Grep.
  - `count_occurrences` → `search_code` with `mode: "count"` (now also returns `files`).
  - `get_dependents` → `get_dependencies` with `reverse: true`; `get_transitive_deps` →
    `get_dependencies` with `depth: N`.
  - `find_hotspots`, `find_circular`, `find_unused`, `find_islands`, `analyze_summary` →
    one `analyze` tool with `kind: "hotspots" | "circular" | "unused" | "islands" |
    "summary"`. `[mcp] enable_structural_tools = false` now hides only `analyze`.
  - The eight old names still work but are no longer listed; their answers carry a
    `warnings` entry naming the replacement (except the two that answer bare arrays,
    `get_dependents` and `get_transitive_deps`). Clients that allow-list tools by name
    must add `mcp__reflex__analyze`.
  - The matching, coverage and freshness rules moved from every tool description into
    the server instructions (said once per session).
- **`list_locations` takes `preview: true`**: each location also carries its matching line,
  trimmed and cut to 120 characters. Agents that needed to see the lines used to run the
  search again with grep.

### Fixed

- **`analyze` and `get_dependencies` say when the import graph is incomplete.** When a
  language has 100+ internal imports and under half of them resolve to an indexed file,
  the answer carries a warning such as `Go: 135 of 52472 internal imports (0.3%) resolve
  to indexed files; …` (MCP `analyze`: `warnings[]`; MCP `get_dependencies`: a second
  `content` text item, the array is unchanged; CLI: `Warning:` on stderr, and
  `warnings` in `rfx analyze --json`). On 2.1.0 Go, Java, C# and Kotlin resolved 0.3–7 %
  of internal imports, so islands, unused files and hotspots for them were mostly noise.
- **Go imports resolve to their package.** A Go import names a directory, so it now
  reaches every non-`_test.go` file of that directory (`get_dependencies` lists them as
  `resolved_paths`). Modules match on a path boundary, longest first (`k8s.io/apiserver`
  is no longer read as `k8s.io/api` + `server`), and an import in no module of the
  workspace is External (`k8s.io/klog` beside `k8s.io/kubernetes` was Internal and
  unresolved). Files of one package count as one unit for islands and unused files.
  Kubernetes: 84 of 52,472 internal imports resolved (0.2 %) → 48,997 of 48,997;
  islands 27,303 → 1,368; unused files 27,371 → 4,802; `rfx analyze` 0.0 → 2.2 s.
- **Java and Kotlin imports resolve by the package a file declares.** `import a.b.C`
  reaches the file whose `package a.b` line declares `C` (as a file name or a top-level
  type, function, property, object or type alias), in whichever Maven/Gradle module it
  lives; `a.b.*` reaches the whole package. Kotlin can import Java and the reverse. The
  old resolver bound every import to the first module whose groupId matched and guessed
  one `src/main/java` path. Wildcard imports now keep their `.*` in `imported_path`
  (Kotlin dropped it, Java too). neo4j: 356 of 59,658 internal imports resolved (0.6 %)
  → 58,926 (98.8 %); islands 11,423 → 655; unused files 11,753 → 1,682.
- **C# `using` directives resolve to every file that declares the namespace** (block,
  nested and file-scoped declarations). A using whose root namespace no file declares
  (a NuGet package) is left out of the resolution rate. Whole-namespace edges are not
  used for cycles: a `using` is no evidence that one file uses another, and as file
  edges they made 11,130 cycles of dotnet/runtime. dotnet/runtime: 281 of 23,051
  internal usings resolved (1.2 %) → 18,911 (82.0 %); islands 55,938 → 16,346; unused
  files 57,759 → 14,989; cycles 89 → 82.
- **Python imports resolve to packages and from the repo root.** An import tries
  `a/b.py`, then `a/b/__init__.py` (only the first was ever tried), with no leading `/`
  when the package is at the index root (that path fell to a fuzzy filename match,
  usually ambiguous). Relative imports resolve without a `pyproject.toml` / `setup.py`.
  `from x.y import a, b` no longer lists `x` among the imported symbols. Django: 1,662
  of 8,673 internal imports resolved (19.2 %) → 8,672 (100 %); islands 4,327 → 789;
  unused files 2,919 → 416.
- **Vendored code stays searchable but leaves the import graph.** A repository that
  commits its dependencies now gets the graph a repository that gitignores them gets:
  vendored files are not islands, unused files, hotspots, cycle members or `deps`
  targets, and imports of them stay External. Reflex recognises Go `vendor/` (with
  `vendor/modules.txt`), Composer `vendor/` (`vendor/composer/installed.json`),
  `cargo vendor` crates (`.cargo-checksum.json`), virtualenvs (`pyvenv.cfg`),
  `node_modules`, `bower_components`, `site-packages`, installed gems
  (`ruby/<version>/gems`), `build.zig.zon` path dependencies, and per-language
  directory names (`third_party/` everywhere but Go, PHP, Rust and Ruby; also
  `vendor/`, `external/`, `extern/`, `deps/` for C/C++, `vendor/` for JS/TS, `_vendor/`
  for Python). `[index.vendored] patterns` (gitignore rules) adds a directory, and
  `!dir/` marks one as project code. `analyze` reports `vendored_files`, and `deps` on
  a vendored file says it is not in the graph. A vendored `Cargo.toml` or gemspec no
  longer makes its crate or gem Internal. Kubernetes (4,241 vendored files): islands
  1,368 → 530, unused files 4,802 → 579. tokio after `cargo vendor` (7,982 vendored
  files) answers exactly as tokio without it. A Composer project with `vendor/`
  committed: islands 7,907 → 47, unused files 7,860 → 1.
- **Paths resolve the same from any working directory.** The C, C++, Zig and Ruby
  resolvers called `canonicalize()`, which reads the disk relative to the process's
  working directory: in-process indexing lost every relative include, and under
  `rfx index` the absolute result was cut at the first `src`/`app`/`lib` or reduced
  to its file name. Paths are now folded lexically and looked up relative to the
  root. Candidates are exact paths; only C/C++ fall back to a suffix match, which now
  matches whole path segments (`a.h` is no longer `lib/xa.h`).
- **Rust workspace crates.** A cross-crate `use b::thing` falls back to shorter
  module paths and the crate root (`thing` in `b/src/lib.rs`), and a crate `b-core`
  is imported as `b_core`. tokio: 86.5 % → 90.4 % of 1,593 internal imports,
  islands 110 → 68.
- **Zig sibling imports and build modules.** `@import("tree.zig")` (no `./`) is a
  file import, and a module a `build.zig` defines (`b.addModule("stdx", …)`, or
  `b.createModule` named by `.addImport("vsr", …)`) resolves to its root file.
  tigerbeetle: Zig internal imports 489 → 1,118 (all resolved), islands 107 → 48.
- **Python `from pkg import submodule`** reaches `pkg/submodule.py` as well as
  `pkg/__init__.py` (`from django.db import models`). django: edges 8,684 → 10,145,
  unused files 355 → 330.
- **Ruby requires search every workspace gem's `lib/`** (`require "rails/command"` is
  in railties), a require whose first segment a gem's `lib/` provides is Internal
  (`active_support` is in the gem `activesupport`), and `require_relative "x"` is no
  longer External. rails: internal requires 555 → 2,883 (97.1 % resolved), islands
  3,115 → 1,689, unused files 1,435 → 712.
- **PHP uses follow `composer.json`.** `autoload-dev` and PSR-0 are read, every
  directory of a prefix is tried, and with `composer.json` a `use` is Internal
  exactly when a project autoload prefix holds it (`Illuminate\…` is External).
  `use A\B as C` no longer stores `C`; `use A\{B, C}` stores `A\B` and `A\C`.
  laravel: 57.8 % → 98.9 % resolved, islands 630 → 427.
- **C# `using static A.B.C` and `using X = A.B.C`** reach the files declaring `C` in
  `A.B` (a type or a namespace); the alias `X` is no longer stored as a using, and
  usings and namespaces inside `#if` blocks count. dotnet/runtime: C# 82.0 % →
  88.6 % resolved.
- Hotspots count distinct importers (a file with two imports of one target counted
  twice), and one cycle is no longer reported twice when a file imports a target twice.
- Text, lock and generated files are no longer islands or unused files (every README
  and lock file was both). Entry points now cover `main.go`, `*_test.go`, `testdata/`,
  `Program.cs`, `*Test(s)`/`*IT` Java, Kotlin and C# files, `__main__.py`, `manage.py`,
  `conftest.py`, `setup.py`, `*_test.py`, and test directories at any depth.
- An edit to `.gitignore` (any directory), `.ignore`, `.rgignore` or
  `.reflex/config.toml` now makes the index stale (listed under `files_modified`): it
  changes which files are indexed, and was never reported.
- A tracked file that `.gitignore` ignores is no longer reported as `added` by every
  check (a fresh build was stale forever).
- After switching back to a branch indexed earlier, the check reported `fresh` while the
  index held the other branch's files. It now compares with the commit of the last
  index run.
- Background compaction no longer deletes the rows of missing files: the stores still
  returned them while the check could no longer see the deletion. `rfx index compact`
  reports `files_removed: 0`; index runs remove deleted files.
- `index_project`, `POST /index`, `rfx watch`, interactive mode and `rfx ask` indexed
  with default settings and ignored `.reflex/config.toml`. `index_project` and
  `POST /index` now wait for a concurrent index run instead of failing with
  `IndexLocked`.
- The version-mismatch rebuild in `rfx index` kept `--languages`.
- `rfx query --ast --json` reported `status: fresh` whatever the index state.
- `rfx watch` updates only the files it saw change (`Indexer::update_paths`) and reacts
  to ignore-file and config edits.

### Performance

- **`rfx index` updates the index instead of rebuilding it.** Files whose size and mtime
  match the index are no longer read; a change is published as a small delta over the
  existing stores (`.reflex/manifest.json` names them), and only the changed files' rows,
  dependencies and symbols are rewritten. Past 2000 changed files or 5 % of the indexed
  text the delta is merged into a new base, reusing the stored text of unchanged files.
  Kubernetes (27k files): a 1-file edit re-indexes in 0.3 s (was 8.5 s, ~1 GB), a run
  with nothing changed takes 0.19 s (was 0.9 s), a cold build 7.5 s (was 8.5 s).
  Search results, `rfx deps` / `rfx analyze` output, freshness fields and `rfx index
  --force` are unchanged: an updated index answers exactly as a fresh build of the same
  tree. The background symbol pass now parses only files whose content changed.
- The first `rfx index` after upgrading rebuilds the index once (the cache format
  changed). While a delta holds changes, `content.bin` / `trigrams.bin` are absent, so an
  older rfx stops with "Cache appears to be corrupted" instead of answering from stale
  stores; its `rfx index` rebuilds as usual.
- `.reflex/` holds new files: `manifest.json`, generation-numbered stores
  (`content.<g>.bin`, `trigrams.<g>.bin`, `trigrams.<g>.plan`, `delta.<g>.*`,
  `recent.<g>.*`), `resolver-configs.json` and `.index-run`. `content.bin` /
  `trigrams.bin` are hard links to the base while there is no delta.

### Library

- **`Indexer::update_paths(root, paths)`** brings the index up to date for a known set of
  changed files or directories without walking the tree (a 1-file edit on Kubernetes in
  about 50 ms). It falls back to `Indexer::index` when an ignore file,
  `.reflex/config.toml`, a resolver config (`go.mod`, `tsconfig.json`, …) or the branch
  changed. `rfx watch` and automatic updates use it.
- **`reflex::auto_update::update_if_stale(cache, opts)`** brings a stale index up to date
  (or builds a missing one) and says what it did; `QueryEngine::with_update(opts)` makes
  an engine do it before it answers. `QueryEngine::new` still answers from the index as
  it is. `query::update_plan` exposes the check's plan.

## [2.0.3] - 2026-09-28

### ⚠️ Breaking (Pulse)

- **`rfx pulse generate` builds the site with Astro Starlight instead of Zola, and `-o` now holds the finished static HTML** (was a Zola project with HTML in `-o/public/`). Upload `pulse-site/` as is; for GitHub Pages change `path: pulse-site/public` to `path: pulse-site`. An existing Zola output directory is refused until you pass `--clean`. Building needs **Node 22.12+** on `PATH` (or `REFLEX_PULSE_NODE`); the template's pinned packages (the site runtime) install once per template version into `~/.reflex/pulse/runtime/<id>/`, from a prebuilt checksummed tarball when one is published for the platform, else with `npm ci --ignore-scripts` (override the root with `REFLEX_PULSE_HOME`, or point `REFLEX_PULSE_RUNTIME` at a prepared directory). Node and `node_modules` never ship inside the rfx binary. Zola and Pagefind are no longer downloaded to `~/.reflex/bin`.
- **`rfx pulse wiki`, `rfx pulse onboard` and `rfx pulse timeline` are removed.** The site's module pages (Internals), overview (Docs) and per-release changelog replace them. `rfx pulse map` now draws the site's source modules (tests and fixtures are no longer nodes) and no longer emits `click` links to `/wiki/` pages.
- `--include` is ignored (the site always has its Docs and Internals tabs). `rfx pulse serve` serves the built HTML with a built-in server under the site's base path (no Node, no Zola); `--open` is replaced by `--no-open`, and `--host` is new.

### Pulse

- **The LLM cache now survives re-indexing.** The key was `blake3(snapshot_id + suffix + context)`, and the snapshot id is a timestamp, so every index change re-narrated every section and the CI cache in `pulse.yml` never produced a hit. The new write cache (`.reflex/pulse/write-cache/`) keys each answer by exactly what would be sent (task kind, prompt version, provider, model, output contract, system and user text). Measured on this repo: change one file, re-index (new snapshot), regenerate → 3 of 3 sections cached, 0 calls. Entries hold no timestamps, so the directory is safe to commit or keep in CI; unused entries are pruned after a successful run (the last 3 runs are kept). The old `.reflex/pulse/llm-cache/` is no longer read.
- **`rfx pulse generate` gains run control:** `--llm on|off|cache-only` (`--no-llm` still works; `cache-only` never calls, for CI jobs without secrets), `--dry-run` (per-kind table of tasks, cache hits and estimated tokens, then exit), `--max-llm-tokens` (defers the lowest-priority calls past the cap), `--force-renarrate[=SCOPES]` (e.g. `=overview,modules` or `=module:src/pulse*`; bare = all; the cache is never wiped), `--llm-model`, `--llm-cache-dir`, `--no-prune`. `--concurrency` defaults to 4 (was unlimited). A `[pulse.write]` section in `.reflex/config.toml` sets `provider`, `model`, `cache_dir`, `keep_runs`, `cache_model_agnostic`, `concurrency` and `max_llm_tokens`.
- **Failure handling:** a probe call runs first; an auth, unknown-model or bad-request error makes the whole run structural instead of producing a half-narrated site. Transient errors retry with backoff (honouring `Retry-After`), a 429 halves concurrency, and five consecutive failures stop the run. `rfx pulse glossary` gains `--no-llm`.
- **JSON sections use the provider's structured output.** The changelog and glossary tasks send a strict JSON schema (OpenAI/OpenRouter `json_schema`, Anthropic `output_config.format`, downgrading to `json_object` or prompt-only when an endpoint rejects it). The system prompt goes in the system role (cached by Anthropic). Anthropic requests no longer send `temperature`, which current Claude models reject. The post-processing pass that split camelCase words is gone: it also rewrote identifiers inside JSON answers.
- A changelog whose LLM answer fails to parse no longer reports `narrated: true`.
- **The changelog is one page per release.** Releases come from semver git tags (newest 12) plus Unreleased. Each release page shows its `CHANGELOG.md` section verbatim (the heading must name the version), then **API changes**: public items added, removed or with a changed declaration, found by extracting both versions of every changed source file from git (one `git cat-file --batch` process; each list capped at 100 with true totals), then the commits grouped by conventional type. The index lists every release with its date, commit count and `+added −removed ~changed`. On this repository v2.0.0 reports exactly the breaking changes its notes describe (`FileLocation::byte_offset` removed, `FileLocation::new` and `PaginationInfo::total` changed).
- **LLM prose is grounded and verified.** Each narrative section (home overview, architecture, module summaries) is written only from a numbered evidence pack built from the index: module and file doc comments, documented items, dependency edges, README sections, CLI commands, and **capabilities proved by imports** (`axum` → HTTP server, `ratatui` → terminal UI, `rusqlite` → database; each with its import line, also listed on the module's Internals page). The model returns sentences with citations; a deterministic gate drops any sentence that cites nothing, names code the index does not know or no cited fact names, states a number no cited fact has, or claims a capability no cited fact proves. A section that loses too much falls back to its structural text. Kept prose shows its sources and is labelled as LLM-written; `--explain` prints every dropped sentence and why, and `.reflex/pulse/reports/write-report.json` records them. On this repository: 10 sections, 43 sentences kept, 0 dropped, 13.1k input tokens (the previous module prompts alone were ~330k). This removes the old onboarding's "`src/interactive` provides an HTTP server" class of error: filename guessing is gone.
- **The generated site is a two-tab docs site.** Docs: home (README tagline and introduction, key numbers), CLI and API reference, changelog. Internals: architecture (module graph), dependency map, one page per module. Rust renders every page to sanitised HTML (raw HTML in doc comments shows as text, links keep only safe schemes, code is highlighted for light and dark themes, long `fn` signatures wrap like rustfmt) and Starlight provides the layout, sidebar per tab, table of contents, search (Pagefind) and theme switch. Mermaid diagrams are bundled and load only on pages that have one, with clickable modules; the site makes no requests to other origins. This repository: 438 pages in 5.5 s end to end (Astro 4.6 s). New flags: `--no-build` (write the site project and stop), `--offline`, `--verbose-build`. LLM prose is labelled on the page. See `.context/PULSE_RENDERER_SPIKE.md` for the renderer measurements.
- openai-compatible endpoints join the provider auto-detection fallback when `openai_compatible_base_url` is set.
- **New `rfx pulse model [--json]`**: builds the renderer-independent Docs Model — two tabs (Docs, Internals), pages with stable routes, a fact store holding every number with its provenance, and narrative slots with structural fallbacks. Modules come from **source files only**: every indexed file gets a role (source, test, fixture, example, bench, generated, vendor, build, docs, config, lock), so `tests/corpus/*` and `build.rs` are no longer modules (on this repo: 101 fixture and 31 test files). Links are targets resolved by one linker (no hard-coded `/wiki/` paths), source links are permalinks to the indexed commit, and slugs persist in `.reflex/pulse/slugs.json`. The JSON is deterministic. The site is rendered from this model.
- **The Docs Model has an API and CLI reference.** A new extractor (`parsers::api`) reads each Rust source file's documented surface — signatures, doc comments (summary, `# Errors`/`# Panics`/`# Examples` sections, examples with rustdoc's hidden lines removed, intra-doc links), visibility, fields, variants, trait and `impl` members, deprecation — and caches it by content hash in `.reflex/pulse/api.db` (this repo: 125 files in 172 ms cold, then free). A surface resolver walks each crate from `Cargo.toml` through `mod` declarations, decides what a user can reach (`pub` all the way down, `#[doc(hidden)]` excluded, `pub use` re-exports inlined like rustdoc), and attaches `impl` members to their types. The Docs tab gets one page per public module and per public type, with anchored symbols and working intra-doc links, plus one page per command read from clap's derive attributes (usage, arguments, options with defaults, subcommands). `pulse.toml` at the repository root (committable) or `[pulse.docs]` sets `library = false` for CLI-first projects and `include = ["crate::module"]`. The extraction does not run during `rfx index`. **Python** (packages from `pyproject.toml`; `__all__` when it is a literal list, else the underscore rule; Google, NumPy and Sphinx docstrings to Markdown; `:class:`/`:func:` roles as links) and **Go** (packages of each `go.mod` module; exported names; `internal/` and `package main` excluded; `[Name]` doc links) get the same reference pages; their API changes also appear on release pages.
- **Prebuilt site runtime.** `rfx pulse generate` downloads a pruned `node_modules` tarball for the platform (about 33 MB, linux/darwin x64 and arm64, win32 x64) from the `pulse-runtime-<id>` GitHub prerelease instead of running `npm ci`. The SHA-256 comes from `pulse-template/runtime.lock.json`, compiled into the binary; a mismatch or any download error falls back to `npm ci`. `REFLEX_PULSE_MIRROR` replaces the download base. `<id>` hashes the template's `package.json` and `package-lock.json`, so rfx releases that share packages share one download. The `Pulse runtime` workflow builds, smoke-tests and publishes the tarballs and opens a PR that updates the lock.
- **`rfx pulse runtime key|status|install`**: print the runtime id (a CI cache key), show Node / runtime / prebuilt status, or install ahead of time.
- **GitHub Action** `reflex-search/reflex/.github/actions/pulse`: installs rfx (or uses the one on `PATH`), sets up Node, caches the site runtime and the LLM write cache, runs `rfx index` and `rfx pulse generate`. Reflex's own `pulse.yml` uses it.

Library: `LlmProvider` gains `complete_request(&CompletionRequest) -> CompletionResponse` (system prompt, `OutputMode`, `max_tokens`, usage, stop reason), `model()` and `caps()`; failures carry a classified `ProviderError` (`downcast_ref`). `complete()` and `rfx ask` are unchanged. `pulse::llm_cache` is removed; `pulse::write` replaces it.

## [2.0.0] - 2026-09-23

A major version: the on-disk index formats, the MCP pagination fields, glob anchoring, the freshness contract and the default coverage rule all change (below). Every existing `.reflex/` cache is rebuilt once by `rfx index`.

### ⚠️ Breaking

- `trigrams.bin` V4 and `content.bin` V2 — **re-index required**. A V3 index is served from an in-memory rebuild (slow) until `rfx index` runs; `rfx index` detects the schema change and rebuilds in full. Postings are now one per distinct trigram per line, grouped into per-file blocks, with no byte offsets: 3.9x → 0.9x of corpus size on the synthetic latency corpus (127 MB → 30 MB), 1.4x on the Reflex repo. `TrigramIndex::load` is O(files) — the directory is binary-searched in the mmap instead of being decoded and re-sorted. `FileLocation` (library API) drops `byte_offset`; `FileLocation::new` takes `(file_id, line_no)`.

- A list-mode `search_code` / `search_regex` (and `rfx query` with a limit) stops verifying once the page is full. The page is unchanged, but **`pagination.total` / `total_count` is `null` whenever the new `total_is_exact` is `false`** (`PaginationInfo.total` is `Option<usize>` in the library API). It was briefly the number verified before the page filled, which the field test showed is not a total (`851` for a term with 18,752 matches) and which every existing reader took as one. `approx_total` is a **sampled estimate**: after the page fills, 32 files spread over the remaining candidates (at most 16 candidate lines each) are verified and the hit rate scaled over the rest; when 32 files or 128 lines or fewer remain, the search simply finishes and the total is exact. Synthetic corpus: `config` estimated 52010 (exact 52010), `ident_7` 26580 (exact 26969); expect roughly ±30% when hits cluster in a few files. The field is omitted for a regex with no literal (every line a candidate). `has_more` is `true` whenever the total is not exact. `mode: "count"`, `count_occurrences`, `list_locations`, `find_references` (`total_references`), symbol/AST searches and no-limit searches keep exact totals. The CLI prints `Found 10 results (~1234 total, estimated)` and points at `--count`. `PaginationInfo::exact_total()` / `best_total()` are the two honest readers.

- Glob filters follow **gitignore / ripgrep rules**: `glob` and `exclude` on every MCP search tool, `rfx query --glob` / `--exclude`, `rfx list-files --glob`, and `[index] include.patterns` / `exclude.patterns`. A pattern containing `/` is anchored at the index root, so `src/**/*.rs` no longer matches `vendor/src/x.rs` (field test: 7131 hits against ripgrep's 6770); write `**/src/**/*.rs` for the old behaviour. A bare name (`*.rs`, `Makefile`) still matches at any depth, a trailing `/` names a directory anywhere (`target/`), a leading `./` is dropped, and `*` no longer crosses `/` (`src/*.rs` is "directly in `src/`", as the `--glob` help always claimed). `[index] include/exclude` patterns were parsed but never applied; they now drive the walker, the working-tree freshness check and the watcher, so an existing config that sets them will shrink the index on the next `rfx index`. `query::result::build_glob_set` (library API) is the one compiler.

- **Freshness is judged by file content, not by commit.** `meta.db` gains a per-file fingerprint (`files.size`, `mtime_ns`, `hash`, `dirty_at_index`; the schema hash forces the rebuild that the V4 format already requires). `check_index_status` and the `status` / `can_trust_results` on every response now compare the working tree to what the index holds: a path `git status` lists whose bytes the index already has is NOT stale, and a commit of already-indexed content, or a `git checkout -b` on the same tree, no longer reports stale (`details.indexed_commit` and `details.current_commit` may differ while status is `fresh`). Reverting a file after its edit was indexed IS stale, which `git status` alone could never say. **Outside a git repository the tree is now walked** (was: always `fresh`). The `reason` text changed from `Working tree has uncommitted changes since indexing (…)` to `Files changed since the index was built (…)`; `details` gains `indexed_at` and `checked_by` (`"git"` | `"walk"`) and is present on `check_index_status` even when fresh. `QueryTimings` gains `status_compute_us`; `rfx query --timing` prints `status wait … (compute …)`. Library: `CacheManager::batch_update_files_and_branch` takes `&[FileRow]`; `GitState` gains `dirty_paths`; `QueryEngine::fresh_index_report` / `index_report_for` return `IndexStatusReport`.

- **The text tier is tracked-files-based, not an extension allowlist.** `[index] mode = "tracked"` (the new default) indexes every file `.gitignore` / `.ignore` / `.rgignore` / `[index] exclude` do not exclude and that is not under a dot-directory, unless a NUL byte anywhere in it says it is binary — ripgrep's defaults (`[index] hidden = true` walks dot-directories). Every count gap against ripgrep in the field test was a file outside the old list (`composer.lock`, `OWNERS`, `SECURITY_CONTACTS`, `.po`, `.jsonl`, `.css`, `.githooks/*`, lock files); an allowlist can never be complete. `OWNERS`, `foo.po`, `a.css` and every other unlisted or extensionless name are now `language: "text"`; non-UTF-8 files are decoded lossily instead of dropped; code without a working grammar (Swift) is indexed as text. **The index grows on data-heavy trees**: prose costs ~2.5x its bytes in trigrams.bin against ~1.4x for code, because a posting is one per distinct trigram per line and a prose line is nearly all distinct trigrams — this is also why Kubernetes indexes at 2.1x and PHP-heavy Hearth at 1.3x. `[index] mode = "allowlist"` restores the 1.7.2 rule (code plus the fixed docs/config extension list, no lock files). `Language::from_path` never returns `Unknown` any more; `Language::is_indexable` is true for every tier and `PathPolicy::classify` decides what a given config indexes. Dot-directories stay skipped (`[index] hidden = true` walks them; `.git/` and `.reflex/` never).

### Performance

- **The background symbol pass is ~13x faster and its cache ~9x smaller.** Kubernetes (27,448 indexed files, 15,436 with a symbol parser, 16 cores): `rfx index-symbols-internal` 44.9 s → 3.4 s wall, 131 s → 25 s CPU, `symbols` table 256 MB → 29 MB; every file's symbols are identical before and after (per-file comparison of all 15,436 decoded blobs, plus `tests/symbol_equivalence.rs` snapshots generated on the old code). What changed:
  - **One tree-sitter query per language per file** (`parsers::LanguageQueries` / `MatchTable`): each module's 6–14 per-kind queries are compiled once per process into one combined query and run in a single walk of the tree, with matches bucketed per original query so the output order is unchanged. Before, every query was recompiled for every file and each ran its own full tree walk — 70% of extraction CPU on Go, more than parsing itself. The query path (`--symbols`, `find_references`) parses cache misses with the same code: latency harness `symbol_lookup` 7.1 → 2.9 ms, `find_references` 14.2 → 6.2 ms.
  - **Previews are found from the symbol's byte offset** (`preview::extract_preview_from_byte`, one `memrchr` for the line start) instead of `lines().skip(n)` from byte 0 per symbol, which made extraction quadratic in file size. Output is identical: tree-sitter rows and `str::lines` both count `\n` only.
  - **Streaming single-writer pipeline** (`background_indexer`): workers parse from `content.bin` and encode; one writer thread owns the connection and commits 1024-file batches (~27 commits instead of 215), retrying a failed batch once and counting it as `write_failed_files` — never as a parse failure. The per-file cache check (a new SQLite connection and two queries per file, on the main thread) is one `SELECT` of every cached `(file_id, hash)` up front; the writer needs no per-file id lookup. Cancellation (`rfx index` waiting for the database) is relayed to workers within 200 ms.
  - **Symbol blobs are zstd-compressed** (`symbol_cache::encode_symbols` / `decode_symbols`; raw JSON under 256 bytes; a 4-byte magic distinguishes the two). `SYMBOL_FORMAT_VERSION` 2 → 3, so an existing symbol cache is dropped and rebuilt once, by the pass that is now ~13x faster. Readers (`SymbolCache`, Pulse glossary/onboard) decode through one helper. `SymbolCache::stats().cache_size_bytes` is now the compressed size.
  - **Thread policy**: `[performance] symbol_threads` (`0` = 50% of cores, up to 32; `REFLEX_SYMBOL_THREADS` overrides), was a fixed 27.5%.
  - Files with no symbol parser (text tiers, Swift) are skipped rather than stored as empty rows: `processed_files` still counts them, `SymbolCache::stats().total_files` no longer does.

- **`rfx index` full rebuilds are ~70x faster on large trees, with lower peak memory and byte-identical output.** Kubernetes (27,448 files, 245 MB, 16 cores): 532 s → 7.7 s wall (`trigrams.bin` and `content.bin` byte-for-byte identical, peak RSS 1.37 GB → 1.05 GB). Where the time went, and what changed:
  - **Dependency recording was 95% of the run (504 s).** Every import lookup opened a new SQLite connection (with the WAL and foreign-key pragmas), and every miss ran `SELECT … WHERE path LIKE '%' || ?` — a full scan of the `files` table, ~9 ms each, ~53k times on Kubernetes. Each file also paid an autocommit `DELETE` and a transaction commit (two fsyncs), and every export row its own connection and commit. The indexer now builds one in-memory `PathResolver` from the `files` table (exact match, then a binary-searched unique-suffix match) and writes every dependency and export row through one `DependencyWriter` transaction with prepared statements: 504 s → 1.1 s for the same 97,151 rows. The suffix match is ASCII-case-insensitive like `LIKE`, but `_` and `%` are literal: `foo_bar.h` no longer also matches `fooXbar.h` (which reported a false ambiguity and left the import unresolved), so a few more imports may resolve on trees with such names. `DependencyIndex::get_file_id_by_path` (CLI/MCP readers) is unchanged.
  - **Trigram extraction ran on the main thread** (`HashMap` entry per posting, ~11 MB/s), partials stored 8 bytes per posting, and the merge re-encoded every list and then read the whole data section back into memory to insert the directory. Extraction now runs in the read pool (`trigram_build::extract_trigram_run`), each batch is built per top-byte shard in parallel with no sort and no dedup (`TrigramIndexBuilder`), partials are V4-encoded, and the merge copies bytes (rewriting one file-id delta per list) in a single pass because the trigram count is known up front: the batch loop 22 s → 4.2 s (3.4 s of it reading, hashing and tree-sitter import extraction), the final write 5 s → 0.2 s, and no copy of `trigrams.bin` in RAM. Batches are bounded by bytes as well as files (`REFLEX_INDEX_BATCH_FILES`, default 5000; `REFLEX_INDEX_BATCH_BYTES`, default 48 MiB), so a 9,000-file tree of large files no longer builds one giant in-memory index; a single-batch tree never creates `trigram_temp/`, and a stale `trigram_temp/` from a killed run is removed on the next start.
  - tree-sitter dependency queries are compiled once per process (`parsers::cached_query`) instead of once per file; TypeScript/JavaScript/Vue files are parsed once for imports and re-exports instead of twice; `tsconfig.json` files are parsed once per run instead of three times.
  - The indexing pool's auto thread cap is 32 (80% of cores), the query pool's rule; it was 8, chosen when the trigram build was serial. `[performance] parallel_threads` still overrides.
  - `RUST_LOG=info rfx index` logs per-phase timings (`phase read+extract`, `phase files+branch transaction`, `phase dependencies+exports`, `phase trigram write`).

- **Count mode no longer materialises results.** `--count`, `mode: "count"` and `count_occurrences` set `QueryFilter.count_only`: the verifier returns `(lines, files)` from one parallel pass with no `SearchResult`, no preview, no path clone and no grouping (`QueryResponse.file_count` carries the file count). Hearth `realm --count`: 18733 lines in 6 ms (was ~30 ms); Kubernetes `(?i)kubernetes --count`: 115539 lines, verify 35 ms, 55 ms engine time with a warm freshness memo (was 159 ms through MCP; ripgrep 80 ms). Without a page budget the doubling verify rounds (16, 32, …) are replaced by one round, so no pool drain waits on the slowest file at each boundary. The synthetic `common_word_count` shape over MCP: 22 ms → 7 ms.
- **CLI open is O(1).** `content.bin` V2 stores its file index as a fixed-width table (28 bytes per file: offset, length, path position, path length) followed by the path blob, read in place at `index_offset + file_id * 28`; `trigrams.bin`'s path section is validated but no longer decoded on load (`TrigramIndex::get_file` decodes it on first request); `OpenIndex` builds its `path → id` map only when a symbol or AST query asks, and the rayon pool on first parallel use. Kubernetes (24k files): open 12–13 ms → 0.4 ms; Hearth 0.7 → 0.24 ms. What is left of the CLI zero-hit floor is `git status` itself, which now runs on its own thread beside the search: Hearth 11–13 ms wall, Kubernetes 85 ms, of which 75 ms is `git status` itself — 60 ms of that its untracked-file scan, which `files_added` needs (the MCP server pays it once per `REFLEX_FRESHNESS_TTL_MS`). `--timing` reports `status wait` (what the query waited) and `compute` (what the check cost) separately.

Field test (29 MB / 1875 files / 16 cores, warm cache): ripgrep won 6–10x on plain queries and 45–56x on common words. On the synthetic 30 MB latency corpus, medians before → after: MCP zero-hit call 10.1 → 0.07 ms; common word first page 743 → 2.7 ms; common word count 797 → 25 ms; regex `fn (get|set)_\w+` 484 → 10 ms; common identifier first page 3690 → 2.5 ms. `trigrams.bin` 127 MB → 30 MB on that corpus.

- Posting-list intersection is a linear sorted merge with a streaming decoder (only the smallest list is materialised); it was quadratic (`HashSet` retain plus a linear `find` per candidate). The intersection also stops once the next list is far larger than the surviving candidate set, leaving the exact, parallel line verification to finish the job.
- `rfx mcp` and `rfx serve` keep the index open across calls (memory maps, path→id map, query thread pool) and reopen only when the files change on disk or after `index_project`. The per-query `PRAGMA quick_check`, `cache.stats()` (a SQLite open plus a `git` subprocess for the broad-query guard) and two further `git` spawns are gone from the query path; the freshness verdict is one memoised snapshot per workspace, invalidated by every index write (this also fixes `index_project` leaving a stale verdict in the memo for up to the TTL).
- The whole-identifier matcher is compiled once per query, not once per candidate line.
- Regex search verifies only the candidate lines its literals name, in parallel; it scanned every line of every candidate file on one thread. The literal extractor now drops the atom before `?`, `*` and `{0,n}` (`foobar?` → `fooba`), which also fixes a pre-existing file-level miss.
- The zero-result hint (`N substring matches — pass contains:true`) is counted during the search; the MCP layer no longer runs a second full search for it.
- Query-time verification runs on a pool sized by `[performance] parallel_threads` (previously ignored by the query path).
- `--symbols` / `symbols: true` / `find_references` — the one shape the field test found unchanged (40 ms engine against a 16–25 ms floor). The symbol path opened `meta.db` three to four times per query (each with the WAL and foreign-key pragmas) and re-ran the symbol-cache schema migration every time; it now uses one connection held on the shared index handle, opened on first use. It spawned `git rev-parse` per query; the branch is read from `.git/HEAD`. It loaded every file hash on the branch (a three-way join over the whole index) and then looked up every candidate's id in a second query; one query fetches ids and hashes for the candidate paths only. Its comment/string pre-filter scanned every line of every candidate file on one thread; it now checks only the candidate lines, in parallel. Cache misses were written one connection and one `INSERT` at a time from inside the parse pool; they are committed in one transaction afterwards. Synthetic corpus, `get_1234 --symbols`: 15–16 ms → 7–9 ms engine time, with the plain query on the same pattern at 6–9 ms — the symbol overhead fell from ~7 ms to ~2 ms. The latency harness gains `symbol_lookup` (in-process median 7.1 ms, MCP 6.2 ms; budget 25 ms) and `find_references` (14.2 / 13.0 ms; budget 30 ms).

### Added

- **Lock and generated tiers.** Lock files (`Cargo.lock`, `package-lock.json`, `*-lock.json`, `*.lock`, `yarn.lock`, `pnpm-lock.yaml`, `go.sum`, `flake.lock`, `uv.lock`, `bun.lock`, …) are indexed as `lang: "lock"` and generated files judged by name (`*.pb.go`, `*_generated.*`, `*.generated.*`, `*.min.js`, `*.min.css`, `*.map`) as `lang: "generated"`, and both are **left out of every search unless asked for**: `include_locks` / `include_generated` on `search_code`, `search_regex`, `count_occurrences`, `list_locations` (`--include-locks` / `--include-generated` on the CLI), or `lang: "lock"` / `lang: "generated"` to select them alone. A zero result whose candidates were only such files carries `excluded_by_default: N` and a `hint` naming the switches, so "which lockfile pins serde 1.0.190" is no longer a confident zero with no way in. `rfx index` prints `Text: N files, Lock: N, Generated: N` and the number of binary files it skipped (`IndexStats.skipped_binary`). The `@generated` content marker is not read (language is derived from the path at query time). `tests/tracked_mode.rs` holds the handoff fixture and a ripgrep parity gate (`count_occurrences(contains:true, include_locks:true, include_generated:true)` equals `rg -c -F | sum`; skipped when `rg` is absent).
- `[index] mode = "tracked" | "allowlist"` and `[index] hidden = true | false` in `.reflex/config.toml`.

- **`ignore_case`** — `rfx query -i` / `--ignore-case`, `ignore_case=true` on `rfx serve`, and `ignore_case: true` on `search_code`, `search_regex`, `count_occurrences`, `list_locations` and `find_references`: `rg -i`, and with `contains` `rg -i -F`. A whole-identifier search stays whole-identifier (`realmid` finds `RealmId`, `realmId`, `REALMID`, not `realm_id`). The engine runs it as a `(?i)` regex whose literals are looked up under every case variant, so it costs about what the case-sensitive query costs (`QueryFilter.ignore_case` in the library API; `prepare_literal_pattern` does the rewrite, which carries no warning). Results keep `kind: text_match`. The zero-result substring `hint` is not produced under `ignore_case`.
- `timings.index_path` (`"trigram"` or `"scan"`) in `rfx query --timing --json` and `REFLEX_MCP_TIMING=1` responses: whether the candidates came from the inverted index or every line was verified (`IndexPath` in the library API). `scan` now only means a pattern shorter than 3 chars, a regex with no 3-byte literal, a non-ASCII literal under `(?i)`, or a keyword symbol query.
- MCP `paths: true` on `search_code` / `search_regex` returns `{status, can_trust_results, paths, total_files}` (plus `has_more` when a `limit` cut the list, and `warnings` / `hint` when set) instead of full columnar rows: 53 files cost 7172 bytes in the field test, against 1750 for `rg -l`.
- The latency harness gains `ci_regex` (`(?i)RARE_MARKER_Q7`; budget 40 ms, twice the case-sensitive `rare_ident`).
- `rfx index` prints `Index/corpus ratio: 1.4x (trigrams.bin …, content.bin …)`; `IndexStats` gains `corpus_bytes` and `trigram_index_bytes` (omitted from JSON when zero).
- `rfx query --timing` prints per-phase timings (open, candidates, verify, status, group) to stderr and includes a `timings` object with `--json`; `REFLEX_MCP_TIMING=1` adds the same object to `search_code` / `search_regex` responses.
- `QueryResponse.substring_hint_count`, `PaginationInfo.total_is_exact` / `approx_total`, `QueryFilter.require_exact_total` / `collect_timings` (library API).
- `tests/latency_budget.rs`: a latency harness over a deterministic synthetic 30 MB corpus, measured in-process and through a real `rfx mcp` stdio round-trip; CI enforces budgets with `REFLEX_LATENCY_BUDGET=1`. The CI performance step's filter was also fixed (it ran zero tests).
- `rfx query --pattern <p>` for patterns that begin with `-` (`--pattern '-> Result<'`); `rfx query -- '-> Result<'` still works, and `--help` now says so. The two forms conflict rather than silently picking one.
- Plain-text tier: `.bru` (Bruno API collections), and the extensionless `Makefile`, `Dockerfile` (plus `Dockerfile.<variant>`) and `Justfile`. Every count gap in the field test traced to these. `.mjs` / `.cjs` were always parsed as JavaScript; now documented. `Language::from_path` (library API) is the one classifier the indexer, watcher and query engine share; `is_text_tier_file` admits the new names.
- `QueryResponse.warnings` and `QueryResponse.hint` (library API): what the engine did to the query (a bracket rewrite) and, for a whole-identifier zero with substring matches, a ready-to-show sentence. `query::prepare_literal_pattern` / `substring_hint_text` are public.

### Fixed

- **Zero-result hints named lock/generated files for any exclusion; the coverage text claimed every tracked file.** The lock/generated candidate count was taken repo-wide, ignoring `file` / `glob`, so `count_occurrences {pattern:"runs-on", file:".github/"}` answered 0 with "6 candidate file(s) were lock or generated files" when the cause was a hidden path, and a file deleted and re-indexed got the same wrong hint. A zero result now carries **`excluded_reason`** — `hidden` (the filter names a dot-directory or dotfile: "use grep --hidden, or set `[index] hidden = true`"), `not_indexed` (the `file` filter names a path the index does not hold, with the reason read from disk: deleted / binary / larger than `max_file_size` / ignored by `.gitignore` / added since the last index), `lock_or_generated` (only when candidates UNDER the filter were lock/generated; `excluded_by_default` is now that scoped count), or `whole_identifier` (the substring hint) — chosen in that order, one hint, or none when no rule applies. The MCP `instructions` and every COVERAGE clause now say what is true: ripgrep's defaults, not gitignored, not binary, not under a dot-directory; `rfx index` prints the same and README gains a Coverage section. `index_project` / `rfx index` report **`deleted_files`** (the deletion was already handled; the report omitted it). `QueryResponse.excluded_reason` (library) is `Option<ExcludedReason>`.
- **A dirty working tree could never become `fresh`.** The freshness baseline was the indexed COMMIT, so after `index_project` had indexed exactly the dirty content, `check_index_status` still listed the same paths and every search answered `can_trust_results: false` until a `git commit`. An agent's session is dirty from its first edit to its last, so the flag read `false` for the entire session and protected nothing. Now (see Breaking): edit → `stale` naming the file; `index_project` → `fresh`, no commit needed. The candidate paths come from `git status`, from the paths that were dirty when the index was written, and from `git diff --name-only indexed..HEAD` when HEAD moved; each is confirmed by `(size, mtime_ns)` and, on a mismatch, by blake3, so a `touch` costs one hash and nothing else. Status on Hearth (1875 files, 3 dirty) stays under the 10 ms budget; the freshness check now runs on its own thread alongside the search, so on the CLI it no longer adds its git spawns to the zero-hit floor. `tests/mcp_freshness.rs` covers edit / add / delete / revert / commit / touch / new-branch, and `tests/freshness_no_git.rs` the same without git.
- **`(?i)` regexes scanned every file.** The literal extractor discarded every literal the moment it saw an `i` flag, so `(?i)kubernetes` — a pattern with a 10-character literal — verified all 6.6M lines of the Kubernetes checkout (387 ms against ripgrep's 166 ms) and printed `has no literals (≥3 chars), falling back to full content scan`. It was the one query shape whose time grew with corpus size. A literal under `(?i)`, `(?i:…)` or `(?im)` is now looked up under every ASCII case variant of each of its trigrams (`TrigramIndex::search_candidates_fold`), intersected cheapest-first with the same early stop as a plain query, then verified by the regex as before. Because the regex crate's Unicode simple case folding also lets `k` match U+212A KELVIN SIGN and `s` match U+017F LONG S, a literal containing `k` or `s` also unions the lines carrying those two characters (`exotic_fold_lines`; a few hundred directory probes that miss on any code corpus), so counts stay equal to ripgrep `-i`. A non-ASCII literal under `(?i)` still scans, with a warning that says why. Hearth (29 MB): `rfx query '(?i)realmid' --regex --count` finds 3720 lines (ripgrep `-i`: 3720) with 3762 candidate lines from the index, verify 4 ms against 4 ms for the case-sensitive `RealmId` regex, 23 ms engine time end to end; `realmid -i` is 3620 (ripgrep `-iw`: 3620). Synthetic latency corpus: `ci_regex` median 0.27 ms in-process, 0.35 ms over MCP stdio. Kubernetes (6.6M lines): `(?i)kubernetes --count` finds 114244 lines, unchanged, with the candidate phase at 10 ms against 9 ms for the case-sensitive regex; the old scan verified for 210 ms, the index path verifies for 80 ms. The rest of that query's ~220 ms is materialising and grouping 114k results plus the git freshness check, which count mode does not need and which a later change should skip.
- Regex candidate union: the per-literal `BTreeMap` merge is replaced by one sort of the concatenated lists (skipped for a single literal), which was ~8 ms per 100k locations on every regex query. The "no literals" warning is now also carried in `warnings[]` (JSON, MCP), and is not printed for a two-character `ignore_case` literal, which scans silently like its case-sensitive twin.
- **Regex literal extraction had four false negatives**, each a line the regex matched that the trigram candidates never named: a scoped flag group `(?i:foo)bar` swallowed `foo`; an alternation with a branch shorter than 3 chars, `(abc|de)f`, emitted `abc` as if required, so `def` lines were never verified; an optional group `(abc)?x` emitted `abc`; and `\p{Lu}`, `\x41`, `\u{..}`, `\A`, `\z` pushed their escape letter into the literal (`abc\p{Lu}` searched for `abcp`). A character class body was also a literal (`[abc]def` searched for `abc`), a `?` after another quantifier (`abc+?`) popped a second character, and the `x` (verbose) flag was ignored. `tests/regex_candidate_lines.rs` now brute-forces all of these against the regex crate.
- The broad-query guard skipped any regex, including one the engine had just built from a literal (the bracket rewrite, and now `ignore_case`): `-i fn` on a 60k-file index would have scanned unguarded. The guard now judges the original literal (`QueryFilter.rewritten_from`).
- `search_regex` in `mode: "count"` dropped `warnings` and `hint`; it now reports them like `search_code` does.
- **`rfx query 'unwrap()'` (and `--count`, `--json`, `--paths`, `rfx serve`) returned a silent `0`** for every whole-identifier pattern containing `( ) [ ] { } < >` — `unwrap()`, `#[derive(`, `RealmId::nil()`, `-> Result<` — with no warning and `total_is_exact: true`, the exact "confident zero" the 1.7.2 changelog said was fixed. It was fixed only in the MCP handlers. The rewrite (escape, run as a regex, say so) now lives in `QueryEngine::search_with_metadata`, so every surface gets the same answer: `warnings[]` in JSON (also on the `--count --json` object), `Warning:` on stderr in plain and `--paths --json` modes. The zero-result substring hint (`0 whole-identifier matches; 89 substring matches — pass contains:true (--contains on the CLI)`) is now printed by the CLI too, as `hint` in JSON and `Hint:` on stderr. `--regex`, `--contains`, `--symbols`, `--kind` and `--ast` are never rewritten. `tests/cli_query_bracket.rs` drives the real binary and checks the CLI and MCP counts agree on every pattern.
- Symbols read from the cache carry the file's language. `SearchResult.lang` is `#[serde(skip)]`, so every cached symbol deserialised with the default language (`rust`) and `--symbols --json` reported `"language": "rust"` for a TypeScript class. Hidden in the test suite until now because `git rev-parse` inside `tests/corpus` named the repo's branch while the indexer had recorded `_default`, so the cache never hit there.
- Symbol cache reads verify the content hash. `INSERT OR REPLACE` keys on `(file_id, file_hash)`, so a changed file kept its old-hash row beside the new one, and the batch read (which ignored the hash) could serve a changed file its pre-change symbols.

## [1.7.2] - 2026-09-22

Fixes four defects found by a field test of the `rfx mcp` server against ripgrep on a
480k-line, 1027-file repo. Three of them returned a **wrong answer** with
`status: "fresh"` and `can_trust_results: true` — the worst failure shape for an AI
consumer, which does not retry but concludes "no callers" and acts on it.


### ⚠️ Breaking


- `IndexWarning.files_modified` is now a list of paths (`Vec<String>`), not a count (`u32`). It is joined by `files_added`, `files_deleted`, `changed_count` and `truncated`. A count told a caller something was wrong without saying what, so the only safe reaction was to distrust everything.

- A stale index now always reports `can_trust_results: false`. Staleness includes uncommitted working-tree changes, so this fires in ordinary edit-then-search loops. It means "these results may be incomplete", not "this call failed".

- `check_index_status` and search warnings advise `index_project`, not `rfx index`. An agent cannot run the CLI.

- The `language` field serializes **lowercase** (`"rust"`, not `"Rust"`). This was always the behaviour; `CLAUDE.md` documented it wrongly.


### Added


- `contains: true` on `search_code`, `count_occurrences`, `list_locations` and `find_references` — substring matching, like `grep -F`. Literal search matches whole identifiers by default, which was undocumented and had no MCP switch: `verify_csrf` returned 0 against ripgrep's 89, because every hit was `verify_csrf_form_field`.

- A zero-result search now carries a `hint` naming the substring count: `"0 whole-identifier matches; 89 substring matches — pass contains:true"`.

- Plain-text tier: `md mdx txt yaml yml toml json proto html htm sh bash ini cfg sql graphql` are indexed for full-text search. Trigram-only — no symbols, no AST, no dependency analysis. Select with `--lang text`, exclude with `exclude_text: true`, disable with `[index] text_tier = false`. Lock files are never indexed. Previously `count_occurrences` over `*.md` returned 0 against ripgrep's 3425.

- `check_index_status` reports `files_modified`, `files_added`, `files_deleted` and `changed_count` as paths.

- `cache::open_meta_db`, which applies `busy_timeout=5000`, `journal_mode=WAL` and `foreign_keys=ON` to every meta.db connection. `REFLEX_SQLITE_JOURNAL=delete` opts out of WAL on network filesystems.

- `ReflexError::SymbolIndexingInProgress` and `ReflexError::CacheVersionMismatch`.

- `meta.db` records `writer_version` and `writer_git_sha`, replacing the `cache_version` row that nothing read.

- `REFLEX_FRESHNESS_TTL_MS` (default 1000) bounds the cost of the working-tree check; `REFLEX_ALLOW_SCHEMA_REBUILD=1` bypasses the version guard.


### Fixed


- **Symbol indexing no longer uses gigabytes of memory on minified files.** `rfx index-symbols-internal` reached 34.4 GiB RSS and ran 3m55s on a 1027-file repo. `extract_preview` bounded previews in LINES with no byte limit; a minified bundle is 1.4 MB on ONE line, so every one of its ~13,843 symbols got a copy of the whole file. Previews are now capped at 512 bytes, the 14 duplicated copies are one shared function, and minified files are detected by bytes-per-line and skipped for symbol extraction only — they stay fully text-searchable. Measured on the same corpus: **15.27 GiB → 0.11 GiB, 38s → 8s**.

- The same unbounded-preview bug on the QUERY path: full-text and regex results returned the whole matched line, which at the 200-result default page size meant ~290 MB of previews per search over a minified file. Now windowed on the match, so a hit deep inside a single-line file still shows its own neighbourhood. `--expand` gets a separate 32 KB ceiling.

- **Two panics on non-ASCII minified content.** `truncate_preview` sliced a raw byte index when a line had no whitespace in its first 100 characters — exactly minified code, which is also where non-ASCII i18n tables live — crashing `rfx query` and the MCP server. `semantic/answer.rs` had the same unguarded slice in three places. The old per-parser previews also underflowed on line 0 and on Vue/Svelte script offsets.

- `symbol_cache::batch_set` cloned every result to blank a field that parsers already leave empty and that the read path overwrites anyway, doubling peak memory for no effect.

- Symbol-indexing status no longer reports a write failure as a parse failure. One failed batch write used to mark its whole batch as failed without decrementing the parsed count, so 27 successes plus one SQLite error read as `parsed_files: 27, failed_files: 27`. `write_failed_files` and `skipped_minified` are now separate, and the error names the count and the first file.

- Working-tree changes are now detected. Freshness compared `git rev-parse HEAD` to the indexed commit and then sampled the mtimes of the first **ten** indexed files — so an edit to any other file, every untracked file, and every deletion reported `fresh`.

- Literal patterns containing brackets no longer return a silent 0. `unwrap()` (ripgrep: 1221), `#[derive(` (1141) and `-> Result<` (2139) all returned 0, because whole-identifier matching wraps the pattern as `\b…\b` and a pattern ending in `)` can never satisfy the trailing boundary. Such patterns are now escaped onto the regex path, with the rewrite reported in `warnings`.

- Deleted files no longer produce ghost hits. `meta.db` was pruned only by `compact()`, which is throttled to 24h and was skipped entirely for the MCP command; the incremental fast path noticed added paths but never deleted ones, so a delete-one-add-one left the file count unchanged and skipped the rebuild.

- No MCP or CLI call surfaces a raw SQLite lock error. The detached symbol pass holds `meta.db` but takes its own lock, so `rfx index` passed the workspace lock gate and then hit `BEGIN IMMEDIATE`, failing for ~4 minutes with `database is locked: Error code 5`. The indexer now asks the pass to yield at its next batch, and reports `symbol indexing in progress (pid N, started HH:MM:SS, 1000/1027 files)` if it does not.

- A cache-format mismatch is no longer treated as corruption. `validate()` runs on every search and bailed on a schema-hash mismatch, which the MCP layer answered by force-rebuilding — so several Reflex versions sharing one `.reflex/` each rebuilt it concurrently. That is the origin of `content.bin is too small`. Readers now degrade with a warning; writers refuse only when a different released version owns the cache.

- Committing already-indexed content no longer leaves the index permanently stale. The incremental fast path skipped the rebuild and therefore never refreshed the recorded commit.

- `list_locations` returns one entry per **match**, as documented. It set `paths_only`, which collapses each file to its first match, so a 20-match pattern returned 8 entries.

- `find_references` count fields reconcile. `pagination.total` 25 / `total_references` 24 / `returned_count` 24 looked like an off-by-one; each now has one meaning and the gap is named `filtered_out`.

- `include_strings` applies in `find_references` count mode, where it was a no-op returning the raw engine total.

- Symbol-indexing progress advances. The `processed % 500 < batch_size` guard was a no-op (`batch_size` was itself 500); batches are now 128 and status is written per chunk, with `pid`, `phase` and `current_file`.

- Stale `indexing.lock` files are reaped by pid liveness rather than a one-hour age rule.

- Reflex no longer treats its own `.reflex/` directory as a source of staleness.


## [1.5.2] - 2026-05-16


### Documentation


- Fix rfx pulse digest → rfx pulse changelog in README

- Remove stale hardcoded test count badge from README


### Fixed


- Move performance tests to separate release-mode CI step

- Resolve all clippy warnings and enable -D warnings in CI

- Commit insta snapshots and fix Windows clippy errors (REF-172)

- Normalize path separators to fix 21 Windows-only test failures (REF-173)

- Make path/home overrides reach storage and dirs lookups (REF-173)

## [1.5.1] - 2026-05-15


### Fixed


- Correct release.toml format for cargo-release

- Restore git-cliff pre-release hook in correct format

- Repair CI failures blocking release

## [1.4.0] - 2026-05-15


### Added


- Add check_index_status MCP tool (REF-107)

- Raise preview truncation to 180 chars and add preview_length param


### Documentation


- Add MCP tool selection decision tree cheatsheet

- Reframe README to reflect CLI-first, not AI-only


### Fixed


- Fix JSON output correctness across query, deps, and analyze

- Detect corrupt trigrams.bin via magic-byte check before skipping rebuild

- Correct singular/plural for result count in TUI (REF-66)

## [1.3.5] - 2026-05-14


### Fixed


- Hardlink issue in pipeline

## [1.3.4] - 2026-05-14


### Fixed


- Dumb symlink issue

## [1.3.3] - 2026-05-14


### Fixed


- Release pipeline

## [1.3.2] - 2026-05-14


### Fixed


- Cargo-dist version issue

## [1.3.1] - 2026-05-14


### Fixed


- Some cargo-dist dependency issues

## [1.3.0] - 2026-05-14


### Added


- Introduce SymbolRef for stable JSON symbol output

- Wire dependencies parameter to list_locations, count_occurrences, search_regex, search_ast


### Changed


- Decompose query.rs into src/query/ submodules


### Documentation


- Add rfx serve threat model and language forward-compat note


### Fixed


- Update reflex-search npm package to fix CVE vulnerabilities

- Broken csharp tests resolved

- Indexing and git workspace issue

## [1.1.3] - 2026-04-27


### Fixed


- Json serialization for mcp

## [1.1.0] - 2026-04-13


### Added


- Add OpenRouter provider support with model fetching and sorting options

- Remove Groq provider and update references to OpenRouter in configuration and documentation

- Enhance LLM response handling with validation and JSON extraction improvements

- Enhance module detection and snapshot analysis with improved descriptions and visibility adjustments

- Implement static site generator with wiki, digest, and map

- Enhance Pulse functionality with LLM narration support for digest and wiki generation

- Add dependency diagram to wiki pages


### Fixed


- Update OpenRouter sort strategy from "speed" to "latency" for API compatibility

## [1.0.4] - 2026-04-07


### Changed


- Streamline language parsing and error handling in CLI and indexer


### Documentation


- Add important setup notes for running commands and gitignore configuration

- Add gitcgr code graph badge

## [1.0.3] - 2025-11-21


### Added


- Update mcp docs to force auto-reindexing and hopefully fixed auto packaging pipeline

## [1.0.1] - 2025-11-21


### Added


- Added auto package publishing for new releases

## [1.0.0] - 2025-11-21


### Added


- Rfx query interactive mode

- Made context full by default and added rfx context to mcp tools

## [0.9.2] - 2025-11-21


### Documentation


- Update README for clarity and accuracy in feature descriptions

## [0.9.1] - 2025-11-20


### Added


- Enhance agentic loop to return query confidence alongside responses

- Semantic query building with external LLMs

## [0.9.0] - 2025-11-19


### Added


- Implement semantic query execution and parsing

- Refactor semantic query generation with project-specific configuration and context extraction

- Massively cleaned up claude.md

- Enhance query execution with count mode and update prompt template for new flags

- Enhance configuration documentation and improve logging levels for cache and semantic query handling

- Update AI provider models and enhance OpenAI request handling for GPT-5 compatibility

- Enhance environment variable handling in tests and update Gemini model defaults

- Add context generation module for AI prompts

- Update default tree depth for --structure option to 1

- Implement agentic semantic query builder with multi-phase workflow

- Enhance agentic mode with additional command options and reporting capabilities

- Enhance regex pattern syntax documentation and clarify flag combinations

- Add support for OpenAI GPT-OSS models with enhanced handling and messaging

- Enhance agentic mode with improved response structure and reporting capabilities

- Add conversational answer generation feature and update CLI handling

- Add context extraction for match results in query handling and answer generation

- Increase token limits to 4000 for Anthropic, Gemini, Groq, and OpenAI providers

- Add termimad for markdown rendering and update provider display with recommendations

- Enhance agentic reporter with spinner support for improved progress visualization

- Implement interactive TUI chat mode for `rfx ask` with message history, input handling, and progress updates

- Enhance Groq API error handling with detailed logging and timeout configuration

- Add bottom padding for message display to improve text wrapping in TUI chat mode

- Add mouse event handling for scrolling in TUI chat mode

- Add text wrapping functionality for message display in TUI chat mode

- Add markdown rendering with consistent prefix for message display in TUI chat mode

- Update wrap_with_prefix to use consistent border colors for message display in TUI chat mode

- Add debug mode to output full LLM prompts and retain terminal history

- Implement API key configuration check and enhance documentation search functionality

- Enhance agentic mode to gather and utilize context for improved answer generation

## [0.8.2] - 2025-11-16


### Added


- Enhance documentation for dependency tracking and semantic query building

- Implement automatic cache invalidation using schema hash

- Implement comprehensive cache corruption detection and validation tests

- Implement cache compaction functionality and CLI commands for manual compaction

- Add cache compaction commands and update documentation for indexing status

## [0.8.1] - 2025-11-16


### Added


- Update documentation for dependency analysis commands and enhance Ruby parser for require statements

## [0.8.0] - 2025-11-16


### Added


- Updated python, go, ts and rust dependency resolution to support monorepos

- Added java, kotlin and ruby monorepo support

- Fixed vue and ts dep resolution

- Add size filtering for island detection

- Refactor dependency analysis API with cleaner command separation

- Enhance dependency analysis commands with pagination and sorting options

- Add analyze_summary tool for quick dependency health overview

- Enhance dependency analysis with file-level grouping and improved output formats

- Refactor search result handling to always use grouped format and improve response structure

- Dependency tracking and analysis


### Fixed


- Tightening up js/ts path resolution

- Enhanced ts/js path resolution for dependencies

- Some updates to pagination for rfx analyze

## [0.7.1] - 2025-11-12


### Added


- Added basic functionality for dependency support

- Dependency resolution across more languages


### Fixed


- Updated some more broken language dependency parsers

- Some more dependency resolution bugfixes

- Refined some regex functionality

## [0.6.0] - 2025-11-11


### Added


- Implemented more mouse support and fixed some syntax highlighting

- Add missing CLI filter options to interactive mode

- Add mouse click support for new filter options

- More UI refinement

- Interactive mode


### Fixed


- Various bugfixes with result list

- Fixed remaining known syntax highligting issues in interactive mode

- More bugfixes

- Accurate mouse click detection for all filter badges

- More ui bugfixes

- Fixed filter display bug

- Rendering bugfixes

- Incorrect file counts in interactive mode

- Fixed background indexing issues

## [0.5.2] - 2025-11-10


### Added


- Added AI suggestions

## [0.5.1] - 2025-11-09


### Fixed


- Random bugfixes

- Switched symbol indexing to use content cache rather than filesystem reads for performance boost

## [0.5.0] - 2025-11-09


### Added


- First swipe at interactive mode, work in progress

- Enhancements

- Loading windows

- Added preflight check to prevent runaway and unbounded queries

- Added early globbing


### Fixed


- Fixed result scrolling

- Removed early filtering performance boost that ended up sacrificing accuracy

- Small optimization to lookups

- Large performance improvement and bugfix with --kind filtering

- Some database refactoring

- Early language filtering

- Speeding up background indexing

## [0.4.2] - 2025-11-07


### Added


- Added many tests and edge cases to symbol search


### Fixed


- Reworking some of the symbol search functionality, bugs remain

- Major fixes to symbol search accuracy

## [0.4.0] - 2025-11-07


### Added


- Lots of json output optimizations to reduce token usage

- Added pagination to prevent breaking context window limits


### Fixed


- Moved pagination to before symbol enrichment to boost performance

- Pagination total count bug

## [0.3.2] - 2025-11-06


### Added


- Fixed attribute support with --kind query

## [0.3.0] - 2025-11-06


### Added


- Added language filter and parallelized --symbol extraction

- Implement word-boundary matching as default search behavior

- Refactored AST query functionality, much simpler now


### Fixed


- Added thread cap for indexing and querying

- Added some missing symbol grammar and added universal ast support, not just for select languages

## [0.2.13] - 2025-11-05


### Fixed


- Cache validation was reading entire files causing 18x slowdown

## [0.2.12] - 2025-11-05


### Added


- Added globbing and build optimizations

## [0.2.10] - 2025-11-04


### Fixed


- Prevent cargo-dist from uploading archives to release

## [0.2.9] - 2025-11-04


### Fixed


- Improve release artifacts with friendly names

## [0.2.7] - 2025-11-04


### Fixed


- Correct release workflow

## [0.2.6] - 2025-11-04


### Fixed


- Trying more release fixes

## [0.2.5] - 2025-11-04


### Fixed


- Another release pipeline fix

## [0.2.4] - 2025-11-04


### Fixed


- Update the release pipeline

## [0.2.3] - 2025-11-04


### Fixed


- Bump version

## [0.2.2] - 2025-11-04


### Fixed


- Bumped version v0.2.2

## [0.2.1] - 2025-11-04


### Fixed


- Lots of general bugfixes

- Removed zip archives from releases v0.2.1

## [0.2.0] - 2025-11-04


### Added


- Add cross-platform binary distribution with cargo-dist

## [0.1.2] - 2025-11-03


### Fixed


- Use default GITHUB_TOKEN and correct repository URL

## [0.1.1] - 2025-11-03


### Fixed


- Correct release-plz GitHub Action reference

- Correct release-plz.toml configuration format

- Remove rfx symlink to fix release-plz

## [0.1.0] - 2025-11-03

