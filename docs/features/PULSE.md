# Reflex Pulse

`rfx pulse generate` turns the Reflex index into a documentation site: plain static
HTML that any static host or CDN can serve. It needs no hand-written docs, no config
and no LLM. An LLM, when configured, only adds prose, and every sentence it writes is
checked against the index before it is published.

```sh
rfx index
rfx pulse generate --no-llm      # ./pulse-site
rfx pulse serve                  # http://127.0.0.1:1111
```

## What the site contains

The site has two tabs.

**Docs** is for people who use the code.

| Section | Source |
| --- | --- |
| Overview | README tagline and introduction, key numbers (files, lines, modules, languages) |
| Guides | Markdown documents in the repository (`docs/**/*.md` and top-level guides) |
| CLI reference | One page per command, read from clap derive attributes: usage, arguments, options with defaults, subcommands |
| API reference | One page per public module and per public type, with anchored members, signatures, doc comments, examples and source links |
| Changelog | One page per release: the `CHANGELOG.md` section, the public API changes, the commits |

**Internals** is for contributors.

| Section | Source |
| --- | --- |
| Architecture | Module graph (Mermaid, click-through), `ARCHITECTURE.md` if present |
| Dependency map | Module edges, cycles, hotspots |
| Modules | One page per source module: files, items, capabilities, depends-on and depended-by |
| Contributing | `CONTRIBUTING.md`, `TESTING.md`, `RELEASE.md` and similar documents |

Search (Pagefind), light and dark themes, and a table of contents per page are built in.
The site makes no requests to other origins.

### API reference languages

| Language | Public surface |
| --- | --- |
| Rust | Each crate from `Cargo.toml`: `pub` all the way from `lib.rs`, `pub use` re-exports inlined, `#[doc(hidden)]` excluded, `impl` members attached to their types |
| Python | Packages from `pyproject.toml`: `__all__` when it is a literal list, else the leading-underscore rule; private modules only through re-exports |
| Go | Packages of each `go.mod` module: exported names; `internal/` and `package main` excluded |

Other languages still appear in the Internals tab (modules, files, dependencies).
Only **source** files produce reference pages. Every indexed file gets a role
(source, test, fixture, example, bench, generated, vendor, build, docs, config, lock),
so test fixtures and build scripts do not appear as modules.

`pulse.toml` at the repository root (or `[pulse.docs]` in `.reflex/config.toml`) limits
the API reference:

```toml
[docs]
library = false                  # CLI-first project: no library reference
include = ["mycrate::client"]    # only these modules and their children
```

### Changelog

Releases come from semver git tags (the newest 12) plus Unreleased. A release page shows:

1. The matching `CHANGELOG.md` section, verbatim. Its heading must name the version.
2. **API changes**: public items added, removed, or with a changed declaration. Both
   versions of every changed source file are read from git; each list is capped at 100
   with the true total shown.
3. The commits, grouped by conventional-commit type.

## LLM writing

Without an LLM the site is complete: every narrative section has a structural text.
With an LLM, three kinds of section get prose: the home overview, the architecture
page and each module summary.

**How prose stays true.** Each section is written from a numbered evidence pack built
from the index: doc comments, documented items, dependency edges, README sections, CLI
commands, and capabilities proved by imports (`axum` means an HTTP server, `ratatui` a
terminal UI, `rusqlite` a database). The model returns sentences, each citing facts from
the pack. A deterministic gate then drops a sentence when:

| Reason | Meaning |
| --- | --- |
| `Uncited` | it cites no fact |
| `UnresolvedIdentifier` | it names code the index does not know |
| `UngroundedIdentifier` | it names real code that no cited fact mentions |
| `NumberMismatch` | it states a number no cited fact has |
| `UnsupportedClaim` | it claims a capability ("HTTP server", "database") no cited fact proves |

A section that keeps less than 60% of its sentences falls back to its structural text.
A section whose evidence is too thin (groundability under 0.2) is not sent at all.
Published prose is labelled as LLM-written and lists its sources.

```sh
rfx pulse generate --dry-run     # tasks, cache hits, estimated tokens; no calls
rfx pulse generate               # write
rfx pulse generate --explain     # also print every dropped sentence and why
```

`.reflex/pulse/reports/write-report.json` records the same information.

### Provider and run control

Pulse uses the provider configured for `rfx ask` (`rfx llm config`, or the
`REFLEX_PROVIDER` / `REFLEX_MODEL` / `REFLEX_AI_API_KEY` environment variables).

| Flag | Effect |
| --- | --- |
| `--llm on\|off\|cache-only` | `off` = `--no-llm`; `cache-only` uses cached answers and never calls (CI without secrets) |
| `--dry-run` | print the task table and exit |
| `--max-llm-tokens N` | defer the lowest-priority calls past the cap; the next run finishes them |
| `--force-renarrate[=SCOPES]` | re-write all sections, or a scope: `overview`, `modules`, `module:src/pulse*` |
| `--llm-model M` | model for this run |
| `--concurrency N` | parallel calls (default 4) |
| `--llm-cache-dir DIR`, `--no-prune` | cache location and pruning |

A probe call runs first. An auth, unknown-model or bad-request error makes the whole
run structural, so a site is never half-narrated. Transient errors retry with backoff
(honouring `Retry-After`), a 429 halves the concurrency, and five failures in a row
stop the run. Under a 90% success rate the whole run is structural too.

`[pulse.write]` in `.reflex/config.toml` sets defaults: `provider`, `model`,
`cache_dir`, `keep_runs`, `cache_model_agnostic`, `concurrency`, `max_llm_tokens`.

### LLM cache

The cache key is the `blake3` hash of exactly what would be sent: task kind, prompt
version, provider, model, `max_tokens`, output contract, and the full system and user
text. It holds no snapshot id and no time, so an unchanged section is never paid for
twice, after any number of re-indexes, on any machine.

- **Storage:** `.reflex/pulse/write-cache/v1/<2-hex>/<key>.json`, one timestamp-free
  file per answer, plus `runs.json`. The directory is safe to commit or to keep in a
  CI cache.
- **Pruning:** after a successful run, entries that none of the last `keep_runs`
  (default 3) runs used are deleted. `--no-prune` keeps them.
- **Model in the key:** switching models re-writes the site.
  `cache_model_agnostic = true` keeps the old text instead.

## Building the site: Node and the site runtime

The layout, search and theme come from [Astro Starlight](https://starlight.astro.build).
Rust renders every page to sanitised HTML; Astro only places it in the layout.

- **Node 22.12 or later** must be on `PATH` (or set `REFLEX_PULSE_NODE`). rfx does not
  ship or download Node.
- **The site runtime** is the template's npm packages. It is installed once per
  template version into `~/.reflex/pulse/runtime/<id>/`:
  1. A prebuilt, checksummed tarball for your platform is downloaded from the
     `pulse-runtime-<id>` GitHub release (about 33 MB).
  2. If none is published, or the download fails, `npm ci --ignore-scripts` installs it.

| Command or variable | Effect |
| --- | --- |
| `rfx pulse runtime status` | Node, runtime directory, prebuilt tarball for this platform |
| `rfx pulse runtime install` | install now (for a CI image) |
| `rfx pulse runtime key` | the runtime id, a cache key |
| `REFLEX_PULSE_HOME` | replace `~/.reflex/pulse` |
| `REFLEX_PULSE_RUNTIME` | use a prepared directory that holds `node_modules` |
| `REFLEX_PULSE_MIRROR` | download tarballs from a mirror instead of GitHub |
| `--offline` | never touch the network; fail if the runtime is missing |
| `--no-build` | write the site project and stop (`.reflex/pulse/site`) |
| `--verbose-build` | show the full `astro build` output |

### Output directory

`-o DIR` (default `pulse-site`) receives only the finished HTML. rfx refuses to write
into the workspace root, your home directory, `/`, or a directory with `.git`. It
replaces an earlier Pulse output (marked by `.pulse-site.json`) and refuses anything
else unless you pass `--clean`.

`--base-url` sets where the site is served: `https://docs.example.com`, or a path such
as `/my-repo/` for a GitHub Pages project site. `rfx pulse serve` serves the output
under that path.

## CI: GitHub Actions

The composite action in this repository installs rfx and Node, caches the site
runtime and the LLM cache, indexes and generates:

```yaml
jobs:
  docs:
    runs-on: ubuntu-24.04
    permissions: { contents: read, pages: write, id-token: write }
    environment: github-pages
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }          # tags and history for the changelog
      - uses: reflex-search/reflex/.github/actions/pulse@main
        with:
          base-url: /my-repo/
          args: --no-llm                  # or set REFLEX_AI_API_KEY and drop this
      - uses: actions/upload-pages-artifact@v3
        with: { path: pulse-site }
      - uses: actions/deploy-pages@v4
```

| Input | Default | Meaning |
| --- | --- | --- |
| `rfx-version` | `latest` | release tag to install, or `none` when `rfx` is on `PATH` |
| `output` | `pulse-site` | output directory |
| `base-url`, `title` | | as on the CLI |
| `args` | | more `rfx pulse generate` flags |
| `index` | `true` | run `rfx index` first |
| `node-version` | `22` | |
| `cache` | `true` | cache the runtime and the LLM cache |

Reflex's own site is built this way (`.github/workflows/pulse.yml`).

## Other commands

| Command | Output |
| --- | --- |
| `rfx pulse model [--json]` | the Docs Model (tabs, pages, facts) without rendering; deterministic, for tests and tools |
| `rfx pulse map [--format mermaid\|d2]` | module dependency diagram on stdout |
| `rfx pulse changelog` | recent commits as a product changelog (Markdown or JSON) |
| `rfx pulse glossary` | project concepts (Markdown or JSON) |
| `rfx snapshot`, `rfx snapshot diff` | structural snapshots of the index and their deltas (files, edges, hotspots, cycles, threshold alerts) |

`rfx pulse wiki`, `rfx pulse onboard` and `rfx pulse timeline` were removed in favour
of the site's module pages, overview and changelog.

## Where things live

| Path | Content |
| --- | --- |
| `.reflex/pulse/api.db` | extracted API surface, keyed by file hash (rebuilt when the extractor changes) |
| `.reflex/pulse/slugs.json` | stable page slugs across runs |
| `.reflex/pulse/write-cache/` | LLM answers |
| `.reflex/pulse/reports/write-report.json` | kept and dropped sentences of the last run |
| `.reflex/pulse/site/` | the staged site project (`--no-build`) |
| `~/.reflex/pulse/runtime/<id>/` | the site runtime (npm packages) |
