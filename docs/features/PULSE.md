# Reflex Pulse — Vision Document

**Status:** Draft
**Last updated:** April 2026

## The problem

Software is being written faster than teams can understand it. AI-assisted development has collapsed the distance between idea and merged code, and the bottleneck has shifted from *writing* software to *knowing what software exists*. The symptoms are familiar to anyone on a high-velocity team: nobody is sure what shipped last week, nobody can confidently describe the current shape of the system, tech debt accumulates invisibly because nobody notices the new patterns drifting away from the old ones, and onboarding a new engineer means pointing at a repo and wishing them luck.

The existing tooling landscape attacks adjacent problems. Code search (Reflex included) helps when you know what to look for. AI code review catches issues at the PR level. Static analysis flags measurable quality metrics. Documentation tools produce artifacts that go stale the moment they're written. None of these provide what teams actually need: a continuously updated, browsable understanding of the codebase that keeps pace with the rate of change.

Pulse is Reflex's answer to that gap.

## The vision

Pulse turns Reflex from a tool you *query* into a tool that *tells you things*. It takes the structural facts Reflex already extracts — symbols, dependencies, hotspots, file churn — and projects them into three browsable surfaces that update as the codebase does: a living wiki, a periodic digest, and an architecture map. All three are grounded in the index, optionally narrated by an LLM, and fully local.

The guiding principle is **structure first, prose only when grounded**. Every claim Pulse surfaces must be traceable to something Reflex can verify from the index. The LLM's job is to narrate structural facts in plain English, never to invent them. This is the discipline that separates useful auto-documentation from the kind that erodes trust the first time someone catches it lying.

## Non-goals

Pulse is not trying to be a replacement for human-written documentation, architecture decision records, or team communication. It is not a code review tool. It is not a project management tool. It does not try to explain *why* decisions were made — only *what* the current state is and *what* has changed. The "why" layer belongs to humans and to tools with access to conversations, tickets, and meetings. Pulse stays in its lane: the code itself.

Pulse is also not trying to be a hosted SaaS. It runs locally against a local index, same as the rest of Reflex. Teams that want shared views generate a static site in CI and deploy it wherever they already deploy things — Pulse itself stays local-first and stateless.

## Definitions

Three concepts the rest of this document depends on:

**Module.** A directory that gets its own wiki page. Tier 1 modules are top-level directories, Tier 2 modules are common paths at depth 2–3, and monorepo subproject roots are promoted to Tier 1. Module detection reuses `CodebaseContext::extract()` from `src/semantic/context.rs`, which internally calls `extract_top_level_dirs()`, `extract_common_paths()`, and `detect_monorepo()`. These helpers are private to `context.rs`, so Pulse consumes their output through the public `CodebaseContext` struct (all fields are `pub`), not by calling them directly. A module is the unit of wiki generation, summary regeneration, and structural analysis — everything Pulse produces is either per-module or cross-module.

**Snapshot.** A point-in-time capture of structural index state. Not a full index copy — just the subset of facts needed for diffing: file metadata, dependency edges, and aggregate metrics. Snapshots are stored as individual SQLite databases under `.reflex/snapshots/`. See [Snapshot format](#snapshot-format) for the schema.

**Diff.** The deterministic delta between two snapshots. Computed entirely with set operations on the snapshot tables (SQL `EXCEPT`, `JOIN`), no LLM involved. The diff is the primitive that powers digests, wiki regeneration triggers, and tech debt detection. See [Diff schema](#diff-schema) for the structure.

## The three surfaces

**The living wiki** is an always-on, browsable view of the current state of the codebase. One page per module or significant directory, regenerated as the underlying code drifts. Each page has a stable structure: what this module is, what it depends on, what depends on it, key symbols and their roles, recent activity, and a plain-English summary. The structural sections are 100% derived from the index. The summary is LLM-generated but grounded in the structural sections — if the LLM wants to say "this module handles request routing," it needs evidence from function names, imports, or string literals in the index to support that claim.

**The digest** is a periodic narrative of what changed. Not a git log — a story. Which modules saw the most activity, which new symbols appeared, which dependencies got added or removed, which hotspots got hotter, which files crossed complexity or fan-in thresholds, which patterns appeared that don't match existing conventions. The digest is produced by diffing two index snapshots and handing the diff to an LLM with enough grounding context to narrate it accurately. Cadence is user-controlled: run it on demand, on a schedule, or as a post-commit hook.

**The architecture map** is a generated diagram of the dependency graph at varying zoom levels. Whole-repo, per-module, per-file. Rendered as mermaid, d2, or SVG. Nodes are clickable and link back into wiki pages. The structural skeleton comes directly from Reflex's existing `analyze` output — hotspots, circular dependencies, islands, disconnected components. The LLM only adds labels and short descriptions, never edges.

## The underlying primitive

All three surfaces are thin layers over the same core capability: **snapshot and diff**. Pulse maintains a history of index snapshots under `.reflex/snapshots/`, each one a serialized capture of the symbol table, dependency graph, file hashes, and analyze output at a point in time. The diff engine takes two snapshot IDs and produces a structured delta — new symbols, removed symbols, edges added or removed, hotspot shifts, threshold crossings. The diff itself is deterministic set operations on the index data, no LLM involved.

Once the diff primitive exists, the three surfaces become straightforward: the digest is "run the diff, narrate it"; the wiki is "for each module, summarize current state and flag drift"; the map is "render the current graph, highlight what the diff touched." The diff engine is where the engineering effort lives. Everything else is presentation.

Snapshots are taken automatically on every `rfx index` run, with an explicit `rfx snapshot` command for forcing one. A configurable retention policy keeps recent snapshots dense and older ones sparse (see [Retention policy](#retention-policy)).

### Snapshot format

**Decision: SQLite.** One `.db` file per snapshot under `.reflex/snapshots/{timestamp}.db`. The schema is a projection of the existing `meta.db` tables in `src/cache.rs`:

| Table | Columns | Source |
|-------|---------|--------|
| `files` | `id`, `path`, `language`, `line_count` | Subset of `meta.db` `files` table |
| `dependency_edges` | `source_file_id`, `target_file_id`, `import_type` | Projected from `file_dependencies` (aliasing `file_id` → `source_file_id`, `resolved_file_id` → `target_file_id`) |
| `metrics` | `module_path`, `file_count`, `symbol_count`, `total_lines` | Computed at snapshot time by aggregating `files` table grouped by module path (no source table in `meta.db`) |
| `metadata` | `key`, `value` | timestamp, git_branch, git_commit_sha, reflex_version, schema_version |

**Why SQLite:** `rusqlite` is already in deps. `ATTACH DATABASE` enables cross-snapshot SQL queries for diffing — no custom binary format needed. The existing `meta.db` schema is the template, so snapshot creation is mostly `INSERT INTO ... SELECT FROM`.

**Size estimate:** A 10k-file codebase produces a ~2–5 MB snapshot. Under the default retention policy (~23 snapshots), total snapshot storage is ~60–120 MB.

### Diff schema

The diff engine produces a `SnapshotDiff` by attaching two snapshot databases and running set-difference queries. This is the primitive all three surfaces consume:

```
SnapshotDiff {
    // File-level changes
    files_added: Vec<FileDelta>,        // New files since baseline
    files_removed: Vec<FileDelta>,      // Deleted files since baseline
    files_modified: Vec<FileModDelta>,  // Files with changed line_count or language

    // Dependency graph changes
    edges_added: Vec<EdgeDelta>,        // New import relationships
    edges_removed: Vec<EdgeDelta>,      // Removed import relationships

    // Structural analysis deltas (reuses DependencyIndex logic from src/dependency.rs)
    hotspot_changes: Vec<HotspotDelta>,     // Files whose fan-in crossed a threshold
    new_cycles: Vec<Vec<String>>,           // Circular deps that appeared
    resolved_cycles: Vec<Vec<String>>,      // Circular deps that disappeared
    island_changes: IslandDelta,            // Disconnected components that appeared/merged

    // Threshold alerts
    threshold_alerts: Vec<ThresholdAlert>,  // Any metric crossing a configured boundary
}

// Default thresholds (configurable via [pulse.thresholds] in config.toml):
//   fan_in_warning: 10    — file imported by >= N others triggers a hotspot alert
//   fan_in_critical: 25   — elevated severity for extreme fan-in
//   cycle_length: 3       — circular dependency chains of length >= N are flagged
//   module_file_count: 50 — modules exceeding N files trigger a complexity alert
//   line_count_growth: 2.0 — files whose line count grew by >= Nx between snapshots
```

**Implementation:** `ATTACH DATABASE 'baseline.db' AS baseline` + `ATTACH DATABASE 'current.db' AS current`, then SQL set operations (`EXCEPT`, `LEFT JOIN ... WHERE NULL`). The analysis deltas (hotspots, cycles, islands) reuse the SQL logic from `DependencyIndex` (`detect_circular_dependencies()`, `find_hotspots()`, `find_unused_files()`, `find_islands()`), but these functions currently hardcode their connection to `meta.db` via `self.cache.path()`. **Prerequisite refactor:** `DependencyIndex` needs a `from_connection(conn: Connection)` constructor (or the analysis functions need to accept `&Connection` as a parameter) so they can run against snapshot databases. This is the primary preparatory change before Phase 1 can begin — see [Preparatory refactors](#preparatory-refactors).

## The tech debt angle

This is where Pulse can do something the existing tools can't. Because snapshots preserve structural history, Pulse can detect patterns that require *time* to see: a new convention appeared in three files this week that doesn't match the existing convention in the other forty (probable AI-introduced drift); a module's dependency count jumped by ten in a single commit (probable abstraction added without integration); a file crossed a fan-in threshold and is becoming a god object; a function signature changed and half its callers weren't updated in the same commit. None of this requires an LLM to detect — it's all graph math on snapshot diffs. The LLM just writes the explanation.

Traditional static analysis misses this because it looks at code at a single point in time. PR review tools miss it because they look at one diff in isolation. Pulse catches it because it has the full history of structural snapshots and can ask questions like "has this pattern existed before, or did it appear this week?" That's a genuinely new capability in the tech debt space, and it falls out almost for free once the snapshot/diff primitive exists.

## Grounding and trust

The single biggest risk with a tool like this is the auto-generated documentation death spiral: the tool writes something wrong, a user catches it, trust collapses, nobody reads it again. Pulse avoids this through a strict discipline about what the LLM is allowed to say.

Structural claims — dependencies, symbol counts, churn, graph properties — come directly from the index with zero LLM involvement and are marked as such in the output. Narrative claims — "this module appears to handle authentication" — are only permitted when the index contains grounding evidence, and that evidence is cited inline in the generated text. If the LLM wants to describe a module's purpose and the index doesn't give it enough to work with, it says nothing rather than guessing. Summaries are marked with a confidence indicator based on how much grounding evidence was available.

This matters more than any other design decision. A Pulse that's 80% useful and never wrong is dramatically more valuable than a Pulse that's 100% fluent and occasionally fabricates. The former gets read; the latter gets ignored.

### LLM grounding protocol

Every LLM invocation in Pulse follows a three-part protocol:

1. **Structural context block.** Before the LLM prompt, Pulse assembles an index-derived fact sheet: symbol names, dependency edges, file counts, churn metrics, threshold crossings. This block is machine-generated with no LLM involvement and serves as the only source of truth the LLM may reference.

2. **Constraint instruction.** The system prompt explicitly instructs the LLM: "You may only narrate facts present in the structural context block. If you cannot ground a claim in the provided data, output `[insufficient evidence]` instead." This is enforced in prompt construction, not post-hoc filtering.

3. **Confidence threshold.** Before invoking the LLM at all, Pulse computes a groundability score: the ratio of semantically-clear symbol names (multi-word identifiers, natural-language function names) to opaque ones (single-letter variables, generated hashes). If the score falls below 0.3, the LLM summary is suppressed entirely for that module, and the output contains only structural data. This avoids wasting LLM calls on modules where narration would be meaningless.

**Implementation:** Reuses the `LlmProvider` trait and `call_with_retry()` (currently `pub(crate)`, accessible from new modules within the crate) from `src/semantic/`. The `extract_json()` helper in `src/semantic/mod.rs` is currently private and needs promotion to `pub(crate)` for Pulse use. The structural context block is assembled from `SnapshotDiff` and per-module metric queries. The `--no-llm` flag bypasses steps 1–3 entirely — see [Structural-only mode](#structural-only-mode).

### Structural-only mode

`--no-llm` is not a degraded experience — it is the recommended default for v1. Structural content (dependency graphs, symbol lists, hotspot rankings, churn metrics, threshold alerts) is the foundation that makes Pulse trustworthy. LLM narration is an enhancement layer that adds plain-English summaries when the grounding protocol permits.

Every Pulse output — wiki pages, digests, maps, static sites — is fully functional without LLM narration. The structural sections render completely; the narrative sections show "[structural-only mode]" placeholders. This ensures teams with strict data policies, limited budgets, or air-gapped environments get the full analytical value of Pulse.

## Static site generation and CI/CD

The highest-leverage use case for Pulse is not an engineer running it on their laptop — it's a CI/CD pipeline generating a fresh documentation site on every merge to main and deploying it somewhere the whole team can browse. This turns Pulse from an individual tool into a team artifact, and it's the mechanism by which ambient codebase awareness actually reaches people who aren't already sold on running dev tools locally.

The `rfx pulse generate` command produces a complete, self-contained, deployable static HTML site in a single invocation. No server required, no runtime dependencies, no database — just a directory of HTML, CSS, JavaScript, and asset files that can be served from any static host (GitHub Pages, Netlify, Vercel, S3, nginx, a CDN, a filesystem). The site includes all three surfaces woven into a single browsable experience: the wiki as the primary navigation, the latest digest as a landing page or dedicated section, and the architecture map as an interactive view accessible from both.

The design constraints for the generated site are deliberately strict:

- **Fully static.** No server-side rendering, no API calls at runtime, no external fonts or scripts pulled from CDNs unless explicitly opted in. Everything needed to render the site is in the output directory. This makes it trivially cacheable, trivially deployable, and works in air-gapped environments.
- **Self-contained assets.** CSS and JavaScript are bundled inline or emitted alongside the HTML. No build step required on the consumer end. If you can serve a directory of files, you can deploy the site.
- **Deterministic output.** Given the same index snapshot and the same LLM responses, `rfx pulse generate` produces byte-identical output. This matters for CI caching, for diffing generated sites across commits, and for reproducibility. LLM non-determinism is handled by caching responses keyed on their grounding inputs, so re-runs against the same snapshot don't re-narrate unchanged modules.
- **Incremental regeneration.** In CI, the common case is "regenerate the site for a codebase that barely changed since the last run." Pulse should detect which wiki pages, digest sections, and map views actually need regeneration based on snapshot diffs, and only re-render those. Unchanged pages get reused from the previous build. This keeps CI runs fast even for large codebases.
- **Configurable output shape.** Teams deploying to GitHub Pages want different asset paths than teams deploying behind a reverse proxy. The generate command accepts a base URL, an output directory, an optional theme, and a config file for site-level settings (title, logo, footer, which surfaces to include).

**Templating requirements** (technology decision deferred):
- Templates must be embeddable in the binary for single-binary distribution — no external template files at runtime.
- No Node.js or external build step for consumers.
- One built-in theme with a clear extension point for custom themes (CSS override file or template directory).
- Mermaid diagrams rendered client-side with a vendored script (no CDN by default, opt-in CDN for smaller bundles).
- **Recommended default:** Tera (used by Zola, well-supported in the Rust ecosystem), but the architecture should abstract the template engine behind a trait so alternatives are possible.

The shape of the command, roughly:
```
rfx pulse generate [OPTIONS]

Options:
  --output <DIR>           Output directory for the generated site (default: ./pulse-site)
  --base-url <URL>         Base URL for asset paths (default: /)
  --title <TITLE>          Site title (default: derived from repo name)
  --include <SURFACES>     Comma-separated: wiki,digest,map (default: all)
  --digest-range <RANGE>   Snapshot range for the digest (default: latest vs previous)
  --theme <THEME>          Visual theme (default: built-in)
  --no-llm                 Generate without LLM narration (structural content only)
  --config <PATH>          Path to a pulse.toml config file
  --clean                  Remove output directory before generating
```

A typical CI integration looks like this:
```yaml
# .github/workflows/docs.yml
- name: Generate Pulse site
  run: |
    rfx index
    rfx snapshot
    rfx pulse generate --output ./site --base-url /my-repo/
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

- name: Deploy to GitHub Pages
  uses: peaceiris/actions-gh-pages@v3
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: ./site
```

The team's documentation site is now regenerated and deployed on every push to main, with zero human involvement and zero drift between code and docs. A new engineer bookmarks the site on their first day and never has to wonder whether what they're reading reflects reality — it was generated from reality ten minutes ago.

The `--no-llm` flag matters more than it looks. It means teams with strict data policies, or teams that simply don't want to pay for LLM calls in CI, can still get the full structural value of Pulse: wiki pages with dependencies, symbols, churn, and hotspots; digests as structured change lists; architecture maps with labeled nodes. The prose layer is optional; the structural layer is always available. This is the fallback discipline from the grounding section made concrete — Pulse degrades gracefully instead of failing when the LLM isn't there.

## Performance targets

| Operation | Target (10k-file codebase) | Notes |
|-----------|---------------------------|-------|
| Snapshot creation | < 1s | SQLite INSERT ... SELECT from meta.db |
| Snapshot diff | < 500ms | ATTACH + set-difference queries |
| Wiki generation (structural only) | < 5s | Template rendering, no LLM |
| Static site generation (structural only) | < 10s | Full site with all surfaces |
| Static site generation (with LLM) | < 60s | Dominated by LLM API latency; cached modules skip LLM calls |
| Digest generation (structural only) | < 2s | Diff + template rendering |

These targets assume a warm filesystem cache and are measured on a single core. Parallel module processing (via `rayon`, already in deps) should improve wiki and site generation linearly with core count.

## Error handling

Three failure modes and how Pulse handles each:

1. **LLM unavailable.** Degrade to structural-only output for the entire generation run, not per-section. This avoids inconsistent output where some modules have narration and others don't. Emit a single warning at the start: "LLM provider unreachable; generating structural-only output." The `--no-llm` flag makes this the intentional path rather than a fallback.

2. **Stale index.** Reuse the existing `IndexStatus` / `IndexWarning` mechanism from `src/models.rs`. When the index is stale (branch mismatch, uncommitted changes, outdated commit), Pulse adds a warning banner to generated output: "Generated from stale index — run `rfx index` for current data." Snapshot creation still proceeds (a stale snapshot is better than no snapshot), but the staleness is recorded in the snapshot's `metadata` table.

3. **Corrupted snapshot.** On load, run `PRAGMA integrity_check` on the snapshot database. If it fails, skip the corrupted snapshot with a warning and fall back to the next available snapshot in the retention chain. `rfx snapshot gc` removes corrupted snapshots automatically.

## Integration with existing Reflex

Pulse is a new subcommand namespace alongside `query`, `ask`, `analyze`, and `context`. It reuses everything: the indexer for snapshot generation, the dependency analyzer for structural facts, the LLM provider config from `rfx ask` for narration, the project context detection from `rfx context` for framing. New state lives under `.reflex/snapshots/` and `.reflex/pulse/` and doesn't interfere with the existing cache. No feature flags, no conditional compilation — one binary, one install, one mental model.

The command surface:

- `rfx snapshot` — take an explicit snapshot of the current index state
- `rfx snapshot diff [A] [B]` — compare two snapshots by timestamp or ID; defaults to latest vs previous; outputs JSON or markdown
- `rfx snapshot list` — list available snapshots with timestamp, git branch, commit SHA, and size
- `rfx snapshot gc` — run the retention policy, remove expired and corrupted snapshots
- `rfx pulse digest` — generate a digest between two snapshots (defaults to latest vs previous), output as markdown or JSON
- `rfx pulse wiki` — regenerate the living wiki pages as markdown
- `rfx pulse map` — export an architecture map as mermaid, d2, or SVG
- `rfx pulse generate` — generate a complete static HTML site with all surfaces woven together, suitable for CI/CD deployment
- `rfx pulse serve` — start a local web server for interactive browsing (convenience wrapper around `generate` plus a file server)
- `rfx pulse watch` — background mode that snapshots and regenerates on file changes

The relationship between the individual surface commands and `generate` is that the individual commands exist for piping, scripting, and partial consumption, while `generate` is the batteries-included path that most teams will actually use. `serve` is essentially `generate` plus a local file server with live reload — useful for previewing what CI will produce.

All commands respect the existing `--json` and `--pretty` flags where sensible, and all outputs are designed to be piped, committed, or published as the user sees fit.

### First-run behavior

Pulse is useful immediately, without requiring history:

- **No prior snapshots:** `rfx pulse digest` produces a bootstrap report — current structural state (module summary, hotspots, dependency graph) with no diff section. The digest header notes "First snapshot — no comparison baseline available."
- **Wiki:** Generates from scratch using current index state. Every module page renders completely; the "recent changes" section shows "No prior snapshot for comparison."
- **Map:** Renders the current dependency graph. No diff highlighting.
- **Static site:** Fully functional on first run. The digest section shows the bootstrap report.

This ensures `rfx pulse generate` produces valuable output on the very first invocation. The diff-based features activate automatically once a second snapshot exists.

## What success looks like

A team using Pulse should be able to: browse an auto-deployed site with accurate, current wiki pages for every significant module; read a periodic digest of structural changes without parsing commit logs; glance at a dependency map and understand system shape; and get proactive warnings when hotspots grow or conventions drift. New engineers onboard by browsing the Pulse site, not by asking teammates what each directory does.

The measure of success is not volume of output — it's how much human-written documentation Pulse makes unnecessary, and how much trust it earns by never fabricating.

## LLM response caching

**Cache key:** `blake3(snapshot_id + module_path + structural_context_hash)`. The structural context hash covers the exact set of facts sent to the LLM — symbol names, dependency edges, metrics. Same structure = cache hit, regardless of LLM provider, model version, or wall-clock time. This means switching from OpenAI to Anthropic doesn't invalidate the cache unless the structural inputs changed.

**Bypass:** `--force-renarrate` flag on any Pulse command to skip the cache and re-invoke the LLM for all modules.

**Storage:** `.reflex/pulse/llm-cache/` directory. Exact file format (one file per module vs. SQLite) deferred to implementation — the key derivation is the important decision.

### Summary regeneration triggers

A wiki page's LLM summary is regenerated when the structural context hash for that module changes. This hash covers: file list, symbol names, dependency edges, and aggregate metrics (file count, line count). Cosmetic changes (whitespace, comments) that don't affect these inputs won't trigger regeneration. This is a direct consequence of the cache key design — no separate "drift threshold" needed.

## Retention policy

**Defaults:** 7 daily snapshots, 4 weekly snapshots, 12 monthly snapshots (~23 snapshots total under steady state).

**Execution:** `rfx snapshot gc` runs the policy. It is also auto-invoked at the end of every `rfx snapshot` command (same pattern as auto-compaction in `cache.rs`). The policy selects the most recent snapshot for each retention bucket and removes the rest.

**Configuration:** `[pulse.retention]` section in `.reflex/config.toml`. Exact schema deferred to implementation, but must support overriding each tier independently (e.g., `daily = 14` to keep two weeks of dailies).

**Corrupted snapshots:** `rfx snapshot gc` runs `PRAGMA integrity_check` on each snapshot and removes any that fail, with a warning.

## Open questions

Questions explicitly deferred to implementation:

- **Theming and customization.** A single built-in theme is sufficient for v1. The extension point (CSS override file or template directory) is specified in [Static site generation](#static-site-generation-and-cicd), but the exact mechanism is an implementation detail.
- **LLM offline / local-model path.** `rfx ask` already supports multiple providers. A local-model provider (Ollama, llama.cpp) is a natural addition but not required for v1 — `--no-llm` covers the immediate need. The `LlmProvider` trait in `src/semantic/providers/mod.rs` is already provider-agnostic.
- **Multi-branch deployments.** Generating sites for multiple branches and letting users switch between them is useful for comparing main to a long-lived feature branch. Out of scope for v1, but the snapshot `metadata` table includes `git_branch`, so the data model supports it when the time comes.

## Reused infrastructure

Pulse is deliberately built on top of existing Reflex modules. This table maps each Pulse capability to the code that already supports it:

| Pulse need | Existing module | Key types / functions |
|---|---|---|
| Module definition (Tier 1/2) | `src/semantic/context.rs` | `CodebaseContext` (pub struct, pub fields) — consumed via `CodebaseContext::extract()`, not the private helpers directly |
| Snapshot schema template | `src/cache.rs` | `files`, `file_dependencies` tables in `init_meta_db()` — snapshot uses column aliases for `dependency_edges` |
| Structural analysis (hotspots, cycles, islands) | `src/dependency.rs` | `DependencyIndex` — requires `from_connection()` refactor (see [Preparatory refactors](#preparatory-refactors)) |
| LLM provider abstraction | `src/semantic/providers/mod.rs` | `LlmProvider` trait, `create_provider()` |
| LLM retry and response validation | `src/semantic/mod.rs` | `call_with_retry()` (pub(crate)), `extract_json()` (needs pub(crate) promotion) |
| Content hashing | `blake3` crate | Already used for incremental indexing in `src/indexer.rs` |
| CLI pattern | `src/cli.rs` | `Command` enum with clap derive macros |
| Index staleness detection | `src/models.rs` | `IndexStatus`, `IndexWarning`, `IndexWarningDetails` |
| Configuration | `.reflex/config.toml` | Existing `[index]`, `[search]`, `[performance]` sections; Pulse adds `[pulse]` |

## Preparatory refactors

Before Phase 1 begins, two changes to existing code are required:

1. **`DependencyIndex` connection injection.** Currently, all analysis methods in `src/dependency.rs` open `meta.db` via `Connection::open(self.cache.path().join("meta.db"))`. Pulse needs to run these same queries against snapshot databases. Add a `DependencyIndex::from_connection(conn: Connection)` constructor that accepts an existing connection, keeping the current `new(cache: CacheManager)` constructor for backward compatibility. The analysis methods (`detect_circular_dependencies`, `find_hotspots`, `find_unused_files`, `find_islands`) then use `self.conn` instead of opening a new connection each time. This is a pure refactor — no behavior change for existing callers.

2. **Visibility promotions.** Promote `extract_json()` in `src/semantic/mod.rs` from `fn` to `pub(crate) fn` so Pulse modules can reuse it for LLM response processing.

These are small, safe changes that should be landed before Pulse implementation begins. They do not affect any existing functionality.

## Implementation order

The temptation is to build all three surfaces at once. Don't. Build in this order:

1. **Snapshot and diff engine.** The primitive everything else depends on. Ship it as `rfx snapshot` and `rfx snapshot diff` with JSON output, no LLM involvement at all. This alone is useful as a standalone feature for anyone who wants to see what structurally changed between two points in time.
2. **Digest.** The smallest surface that uses the diff engine end-to-end. Forces the grounding discipline to get nailed down before it matters. Ship as markdown output first.
3. **Living wiki.** Requires the same grounding discipline as the digest plus a notion of "module" and per-module regeneration logic. Ship as markdown output first.
4. **Static site generator.** The `rfx pulse generate` command that weaves the wiki and digest into a deployable HTML site. This is where the project becomes a team tool rather than an individual one, and it's the step that unlocks CI/CD deployment as the primary use case.
5. **Architecture map.** The flashiest but least essential surface. Builds on everything above and gets woven into the generated site as a final enhancement. Ship as mermaid export first, interactive SVG later.

Each step should be usable and shippable on its own. If Pulse stops at step 1, Reflex users still got a useful feature. If it stops at step 2, they got a genuinely valuable one. If it stops at step 4, teams have a real documentation pipeline. The goal is for every step to be worth the work in isolation, so the project can pause or pivot at any point without leaving dead code behind.