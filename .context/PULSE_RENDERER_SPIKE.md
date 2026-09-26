# Pulse renderer spike (M0a): Astro Starlight go/no-go

**Date:** 2026-09-26 · **Branch:** `worktree-agent-a3adeb3ed08a9207a` · **Template:** `pulse-template/`

## Verdict: CONDITIONAL GO

Astro Starlight works as the Pulse renderer. The template builds a correct,
self-contained static site from a staged JSON + Markdown bundle, served under a
subpath, from a 33 MB runtime tarball. Reflex-size sites build well inside budget.

The design in the M0a brief does **not** scale to Kubernetes size: Markdown
fragments rendered by Astro's content layer, one `.md` per symbol doc. At 20k
pages it runs out of the default Node heap. With an 8 GB heap it takes 12 min
locally, which is about 22 min on a 4-vCPU runner, and peaks at 8.2 GB RSS. Those
are past both NO-GO lines (> 10 min, > 6 GB).

It is a GO under these conditions:

1. **Rust renders the prose.** Rust turns Markdown into sanitised HTML
   (pulldown-cmark plus an allowlist sanitiser, using the same rules as
   `remark-pulse.mjs`), highlights the code, and writes the HTML inline in the
   page JSON (variant D below). The `.md` fragment path stays available for small
   sites and for hand-written docs.
2. **Page JSON stays on disk.** The page reads its JSON file at render time. It
   never goes into the Astro content store (`build.pageSource: "fs"`).
3. **Pagefind runs on shards of 8k pages or fewer**, merged in the UI, or with
   code and tables excluded. Pagefind itself peaks at 6.3–6.5 GB on 20k pages.
4. **The runtime ships the plain `pagefind` binary**, not `pagefind_extended`. The
   extended binary alone is 50 MB of the 87 MB tarball.
5. **The page granularity caps below are enforced by `rfx pulse`.**

Under these conditions the local 20k build takes 174–193 s while another workload
kept 80–90 % of the CPU busy, with a Node peak of 1.74 GB. Section 4 has the
estimate for a runner.

## 1. What was built

```
pulse-template/
  package.json / package-lock.json   exact pins: astro 7.3.5, @astrojs/starlight 0.42.4,
                                     @astrojs/markdown-satteri 0.4.2, mermaid 11.17.2,
                                     @fontsource-variable/inter 5.3.0, @fontsource/jetbrains-mono 5.3.0
  astro.config.mjs                   static; reads ./pulse.config.json (title, site, base, tabs,
                                     sidebar, build knobs); passthroughImageService; trailingSlash
                                     'always'; Starlight + pagefind; expressive-code dual themes;
                                     routeMiddleware; Header override; customCss; cacheDir .astro-cache
  src/content.config.ts              docs (docsLoader, empty) · pages (glob **/*.json, stem ids)
                                     · fragments (glob **/*.md, {page,toc,lang,origin,depth_offset})
  src/pages/{docs,internals}/[...slug].astro  -> src/lib/paths.ts (getStaticPaths + cacheKey)
  src/layouts/PulsePage.astro        blocks: heading, prose, signature, symbol-card, dep-graph, table
                                     inside <StarlightPage frontmatter headings>
  src/routeData.ts                   per-tab sidebar (pruned > 800 links) + recomputed prev/next
  src/components/Header.astro        "Docs | Internals" tabs
  src/plugins/remark-pulse.mjs       Sätteri mdast plugin (sanitise, base-prefix, mermaid, lang)
  scripts/gen-fixture.mjs            synthetic bundle (+ --prerender, --plain-signatures, --touch)
  scripts/build-runtime.sh, prune.mjs  pruned zstd runtime tarball (+ --verify)
  scripts/bench.sh                   cold/warm build benchmark against an extracted runtime
  scripts/check-dist.mjs             31 correctness/size checks on a built dist
.github/workflows/pulse-spike.yml    workflow_dispatch: acquisition (download+extract vs
                                     actions/cache vs npm ci) + build benchmarks on 4 OSes (not run)
```

**Build knobs** go in `pulse.config.json` as `build: {...}`:

- `incremental` (default `true`): Astro's `experimental.incrementalBuild`.
- `chunkedStore` (default `false`): `experimental.collectionStorage: 'chunked'`.
- `pageSource` (`"fs"` = variant D).
- `concurrency` (default `1`).

**Benchmark variants:**

| id | prose | signatures | page JSON |
| --- | --- | --- | --- |
| **A** (the brief) | `.md` fragments rendered by Astro (Sätteri + expressive-code) | `<Code>` (expressive-code) | content collection |
| B | inline HTML (pre-rendered, as Rust would) | `<Code>` | content collection |
| C | inline HTML | plain `<pre>` (highlighted outside Astro) | content collection |
| **D** (recommended) | inline HTML | plain `<pre>` | read from disk at render time (`bundle/index.json` + `bundle/pages/**`) |
| E | = D with `incremental: false` | | |

## 2. Runtime tarball

The tarball is `node_modules` only, from `npm ci --ignore-scripts --omit=dev`, pruned,
then `tar | zstd -19`. GNU tar runs with `--sort=name --mtime=@0`, so the output is
reproducible: two builds gave the same 32,910,247 bytes.

| stage | bytes | files |
| --- | ---: | ---: |
| `npm ci` (production) | 346.0 MB | 20,603 |
| pruned (maps, d.ts, docs/tests/demos, READMEs, sharp, @img, @types, .bin, mermaid non-core builds, pagefind_extended) | 142.2 MB | 12,819 |
| **zstd -19 tarball (linux-x64)** | **32.9 MB** | |
| same, keeping `pagefind_extended` | 87.1 MB | |

- Compression takes 17 s with `-T0` on 16 threads. Extraction takes 0.7–0.9 s.
- The largest remaining items, zstd-compressed: @rolldown native 5.7 MB, @esbuild
  3.9 MB, lightningcss 2.6 MB, @astrojs 2.3 MB, @bruits (Sätteri native) 1.2 MB,
  @fontsource 2.5 MB, katex 1.1 MB, @shikijs 1.1 MB, and the plain pagefind binary
  about 5 MB.
- **The tarball is per platform.** Rolldown, lightningcss, Sätteri, esbuild and
  pagefind all ship native binaries that npm picks at install time. That means one
  tarball each for linux-x64, linux-arm64, darwin-arm64 and win32-x64. They can be
  built on each runner, or cross-built with `npm ci --os/--cpu/--libc`. The
  linux-x64 build targets glibc (`*-gnu` bindings), so musl/Alpine needs its own.
- **Resolution from the extracted runtime works with no workaround.** The staged
  project sits at `<runtime>/sites/<key>/`. The build is invoked from `/` as
  `node <runtime>/node_modules/astro/bin/astro.mjs build --root <site>`. Vite,
  Astro, the integrations and `import('mermaid')` all resolve through Node's
  upward lookup into `<runtime>/node_modules`. The site needs no `package.json`
  deps, symlinks or `NODE_PATH`. The verify step builds from `/` with no
  `node_modules` above the scratch directory, and all 31 dist checks pass.

## 3. Measurements

**Machine:** AMD Ryzen 7 7840HS, 8 cores / 16 threads at up to 5.1 GHz, 54 GB RAM,
Node 24.15.0.

**Fixture:** K = 8 symbols per page and a dep-graph Mermaid block on every 20th
page. Pages split into about 5 % docs and 95 % internals, in a
topic → module → page sidebar. In variant A that means 1 + K fragments per page:
173k `.md` files at 20k pages.

**Warm build:** about 1 % of the prose pieces edited, with caches (`.astro-cache`,
`dist`) kept.

**Machine load:** another session's runaway headless-Chrome GPU process and
unrelated cargo builds kept the machine 80–90 % busy from about 01:20. Runs marked
† ran under that load. Unmarked runs ran on a quiet machine. Contention inflated
wall time by about 1.2× (A) to 2× (C).

### Cold / warm builds

| var | pages | cold wall | cold peak RSS | content sync | render (static routes) | pagefind | warm wall | warm RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 100 | 6.5 s | 1.51 GB | 1 s | 0.98 s | 0.19 s | 3.2 s | 1.45 GB |
| A | 400 | 10.7 s | 1.73 GB | 4 s | 3.7 s | 0.7 s | 5.0 s | 2.00 GB |
| A | 1000 | 25.9 s | 2.72 GB | 12 s | 8.9 s | 1.6 s | 8.9 s | 2.47 GB |
| A | 5000 | 156.8 s | 3.81 GB | 73 s | 55 s | 8.8 s | 45.9 s | 4.29 GB |
| A | 20000 | **OOM** at 229 s (default ~4 GB heap, during content sync) | | | | | | |
| A | 20000 (8 GB heap + chunked store) | **725.9 s** | **8.18 GB** | 461 s | 165 s | 69 s | killed at 32 min, see G7 | |
| B | 5000 | 57.9 s | 2.76 GB | 2 s | 42 s | 8.8 s | 24.2 s | 3.22 GB |
| C | 400 | 5.2 s | 1.63 GB | 0 | 2.0 s | 0.7 s | 5.0 s | 1.57 GB |
| C | 1000 | 8.4 s | 1.78 GB | 0 | 3.1 s | 1.4 s | 5.9 s | 1.77 GB |
| C | 5000 | 32.9 s | 2.81 GB | 1 s | 18 s | 8.0 s | 22.7 s | 3.30 GB |
| C | 20000 (partly †) | 312 s | 6.49 GB | 9 s | 236 s | 50 s | 230 s | 6.36 GB |
| A † | 400 / 5000 | 21.0 / 194.7 s | 1.90 / 4.36 GB | 6 / 77 s | 7.7 / 89 s | 1.2 / 13 s | 9.2 / 73.6 s | |
| C † | 5000 | 65.0 s | 2.26 GB | 3 s | 37 s | 12 s | 40.6 s | 2.31 GB |
| D † | 400 | 10.0 s | 1.59 GB | 0 | 4.7 s | 1.1 s | 6.5 s | 1.38 GB |
| D † | 5000 | 69.9 s | 1.52 GB | 0 | 52 s | 12 s | 24.9 s | 1.53 GB |
| **D †** | **20000** | **192.5 s** | 6.58 GB (pagefind; Node 1.74 GB) | 0 | 137 s | 50 s | **87.7 s** | 1.74 GB |
| E † | 20000 | 173.6 s | 6.54 GB (pagefind) | 0 | 117 s | 51 s | 176 s (full re-render) | |

The version-1 template run (smart punctuation on, no incremental build) measured
A: 100 = 4.7 s, 400 = 12.5 s, 1000 = 35.6 s, 5000 = 169 s / 4.49 GB. Its warm
builds were only 1.2–1.9× faster, because without `incrementalBuild` Astro
re-renders every page.

**RSS attribution.** A per-process trace of a cold D build at 20k shows Node
peaking at 1.74 GB and the `pagefind` child at 6.47 GB. GNU time reports the
largest single process. Pagefind run on its own over the same 20k-page dist:

| pagefind run | wall | peak RSS | index |
| --- | ---: | ---: | ---: |
| all 20,000 pages | 45.5 s | 6.32 GB | 103 MB |
| excluding `pre`, `table`, `.pulse-meta` | 33.4 s | 4.07 GB | 95 MB |
| 8,019-page subset (glob) | 18.4 s | 2.63 GB | 43 MB |

### Where the time goes (variant A)

A CPU profile of a cold A build at 1000 pages (24 s of samples):

| share | where |
| ---: | --- |
| 27 % | V8 / GC / idle |
| 14 % | Shiki's Oniguruma WASM |
| 9 % | Sätteri |
| 9 % | expressive-code core |
| 8 % | node core |
| 3 % | zod |
| 3 % | Shiki textmate |
| 2 % | devalue |

Syntax highlighting makes up about 35–40 % of the busy time. Content sync costs
about 2 ms per fragment and is single-threaded. So the per-symbol fragment design
pays that cost 173k times at 20k pages. Moving Markdown and highlighting to Rust
(A → C) removes content sync entirely and cut the cold build by 4.8× at 5k pages.
Moving page JSON out of the content store (C → D) took the Node peak from 6.5 GB
to 1.74 GB at 20k pages.

### Output sizes

| var | pages | dist | avg HTML/page | pagefind index | JS per page (static) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 100 | 14.1 MB | 86.5 KB | 1.04 MB | 17.1 KB |
| A | 400 | 66.3 MB | 146 KB (full sidebar, ~380 links) | 2.19 MB | 17.1 KB |
| A | 1000 | 72.3 MB | 62.2 KB (pruned sidebar) | 4.39 MB | 17.1 KB |
| A | 5000 | 366 MB | 66.9 KB | 18.8 MB | 17.1 KB |
| A | 20000 | 1.53 GB | 72.6 KB | 72.2 MB | 17.1 KB |
| C/D | 5000 / 20000 | 248 MB / 1.06 GB | 43.9 / 47.8 KB | 18.7 / 72.1 MB | 14.6 KB |

**JS per page.** A page ships 12.2 KB of module scripts (the 1.2 KB Mermaid
loader included) and 4.9 KB of inline scripts. Starlight's search then fetches the Pagefind UI
(`ui-core`, 94 KB) on idle. Everything a normal page loads comes to 106 KB raw,
32 KB gzip, 26 KB brotli. Mermaid is not loaded on pages without a diagram. On a
page with a flowchart it adds about 785 KB, and 3.37 MB raw / 0.96 MB gzip across
all 103 Mermaid chunks. Fonts are 25 woff2 files, 408 KB; the browser fetches only
the subsets a page uses.

**Mermaid at 20k pages.** About 1,400 of the 20k pages (sampled) have a diagram; the rest load no
Mermaid code.

## 4. Thresholds

The machine above is faster per core than a 4-vCPU `ubuntu-24.04` runner (AMD EPYC
7763, Zen 3, about 3.5 GHz). The build is dominated by one JavaScript thread, so
local times are scaled by **about 1.8×** for the runner. Pagefind and rolldown
also lose cores on the runner.

| threshold | A (brief) | D (recommended) | pass? |
| --- | --- | --- | --- |
| tarball ≤ 60 MB zstd | 32.9 MB (needs the plain pagefind swap; 87 MB otherwise) | same | ✅ |
| Reflex size (300–500 pages) ≤ 30 s | 10.7 s local → ~19 s on a runner | 5–10 s local → ~9–18 s | ✅ |
| 20k pages ≤ 4 min | 726 s local → ~22 min | 174–193 s under load; ~110–140 s local estimated quiet → ~3.3–4.2 min on a runner | A ❌ · D ⚠️ borderline |
| 20k pages RSS ≤ 4 GB | 8.2 GB, and OOM at the default heap | Node 1.74 GB ✅; pagefind 6.5 GB ❌ unless sharded or trimmed (2.6 GB per 8k-page shard) | A ❌ · D ✅ with sharding |
| 20k / 5k time ratio ≤ 5× | 4.6× (only with an 8 GB heap) | 2.75× (both runs under the same load) | ✅ |
| JS per page ≤ 100 KB | 17 KB static; 106 KB raw / 32 KB gzip including Pagefind UI fetched on idle | 14.6 KB static | ✅ (on the wire) |
| zero external requests | no CDN, jsdelivr, unpkg or googleapis in dist; the browser loads only same-origin files | same | ✅ |
| subpath (`base: /reflex/`) works | tabs, sidebar, prev/next, fragment links and Pagefind URLs all prefixed | same | ✅ |
| NO-GO: tarball > 100 MB | no | no | — |
| NO-GO: 20k > 10 min or > 6 GB | **yes** (12 min, 8.2 GB) | no (Node side); pagefind must be sharded | A NO-GO · D GO |

## 5. Correctness (N=100, base `/reflex/`)

`scripts/check-dist.mjs` runs 31 checks. All of them pass for variant A, built from
the extracted runtime by `build-runtime.sh --verify`, and for variant D. A browser
check with Playwright confirmed the rest.

- **Sidebar per tab.** Docs pages list only `/reflex/docs/…`. Internals pages list
  only `/reflex/internals/…`. The current page is marked, and prev/next follow the
  tab's reading order.
- **Tabs.** The header shows `Docs | Internals`. Hrefs carry the base and a
  trailing slash. `aria-current` is set on the active tab.
- **Fragment links.** 9,745 root-relative links get the `/reflex` prefix, and none
  are missed.
- **Pagefind** indexes all 100 custom `StarlightPage` routes: `data-pagefind-body`
  is present. A search for `GlobMatcher3` returns base-prefixed URLs, and excerpts
  keep the hazard text escaped.
- **Hazards render literally.** This covers `Vec<String>`, `<T>`, `{x}`, `{{y}}`,
  `std::vec::Vec`, `a:b`, `<!-- hidden -->`, `<script>alert(1)</script>`,
  `:directive`, `::leaf{#id}`, `10:30`, `--json` and ASCII quotes.
  - No executable `<script>` contains fragment text; this was checked on the DOM.
  - No `javascript:` URL appears in any `href` or `src`.
  - The Markdown link `[bad](javascript:…)` is unwrapped to its text. The raw
    `<a href=javascript:…>` becomes text.
  - The relative image `![](x.png)` becomes `[image: diagram]` and never reaches
    Astro's image pipeline.
  - In the browser, no dialog opened.
- **No CDN references.** `grep -r "https://cdn|jsdelivr|unpkg|googleapis" dist`
  finds nothing, and the browser's resource list has no cross-origin requests.
- **Mermaid** is loaded only on pages with `.pulse-mermaid`, through dynamic
  `import('mermaid')` with `securityLevel: 'strict'`. The SVG rendered, and the
  diagram re-renders when the theme changes.
- **Theme.** The light/dark switch (`starlight-theme-select`) works, and
  expressive-code emits both themes.
- **TOC.** Fragment headings (Overview, Hazards, Safety) and symbol-card headings
  appear. Symbol-doc headings (`toc: false`) are excluded.

## 6. API gotchas and findings

- **G1. Astro 7 no longer uses remark for Markdown.** The default processor is
  Sätteri, a native Rust parser with JS visitor plugins. `markdown.remarkPlugins`
  does not apply to it. `remark-pulse.mjs` is therefore a Sätteri mdast plugin, a
  plain object of visitors passed as `satteri({ mdastPlugins: [...] })`.
  - `unified()` from `@astrojs/markdown-remark` is still supported, as an optional
    peer, but it is slower.
  - A visitor can return `{ rawHtml }` to emit raw HTML. That is how Mermaid
    fences become `<pre class="pulse-mermaid">`.
- **G2. Smart punctuation is on by default** in Astro's Sätteri setup. It turns
  `--json` into an en dash and `<!-- -->` into `<!– —>`, and curls quotes.
  Doc-comment prose needs `features.smartPunctuation: false`.
- **G3. Starlight enables the `directive` parser feature globally.** Its
  restoration plugin turns unclaimed `:name` directives back into text, which is
  correct. Starlight's own transforms (asides, heading anchor links) run only on
  the `docs` collection path, so fragments get no `¶` anchor links.
- **G4. The Astro entry point** is `node_modules/astro/bin/astro.mjs`, not
  `astro.js`.
- **G5. Astro reads `.d.ts` templates at build time**, for example `@astrojs/mdx`'s
  `content-module-types.d.ts`. Pruning must keep `.d.ts` files under `astro/` and
  `@astrojs/*`.
- **G6. Default cache locations.** `cacheDir` defaults to
  `<root>/node_modules/.astro`, which holds `data-store.json`, and Vite writes
  `<root>/node_modules/.vite`. The template sets `cacheDir: './.astro-cache'` and
  `vite.cacheDir` so the site has one cache directory that rfx can persist, and no
  `node_modules` of its own.
- **G7. Two content-store limits at scale.**
  - At 5k pages (A) `data-store.json` is 156 MB. At 20k pages it is 848 MB
    (measured chunked), which is past V8's roughly 512 MB string limit. So a
    single-file store cannot be written at that size, and
    `experimental.collectionStorage: 'chunked'` is required.
  - In Astro 7.3.5, `ChunkedWriter.getChunkEnd` calls `TextEncoder.encode()` once
    per character. In an 8 s CPU sample, 59 % of the time was spent there. The warm
    20k build was still inside content sync after 32 min.
  - The fix is a one-line byte-length computation; it should go upstream, or into
    the runtime as a patch. Variant D keeps the store tiny (no page JSON, no
    fragments), which sidesteps both problems.
- **G8. The Node heap.** Variant A at 20k pages OOMs at the default heap of about
  4 GB during content sync. Variant D stays under 1.8 GB.
- **G9. `StarlightPage` computes pagination before route middleware runs**, from
  the (empty) global sidebar. A middleware that swaps the sidebar must also
  recompute `pagination`; `routeData.ts` does this. The global `sidebar: []` keeps
  Starlight from building and deep-cloning (`klona`) a sidebar on every page.
- **G10. A full sidebar on every page is O(N²) bytes.** At 400 pages the full
  internals tree (about 380 links) adds about 80 KB to every page: 146 KB average
  against 62 KB when pruned. `routeData.ts` keeps the full tree up to 800 links
  (`sidebarFullTreeMax`). Beyond that, groups off the current path collapse to a
  single link to their first page.
- **G11. Starlight's TOC always starts with an "Overview" (`_top`) entry.** A
  fragment heading named "Overview" shows up twice, so Rust should avoid it.
- **G12. Heading ids collide across fragments.** Each fragment slugs its own
  headings, so `# Errors` appears K times per page. The template enables Sätteri
  `headingAttributes` so Rust can write `# Errors {#sym-foo-errors}`. Rust must
  then escape a trailing `{…}` in real heading text.
- **G13. Incremental builds need `experimental.incrementalBuild`** (Astro 7.2+) and
  a `cacheKey` from `getStaticPaths`. With them, the warm A build at 5k pages drops
  from 90 s to 46 s, and warm builds re-render only the changed pages (A at N=100:
  14 pages rendered, 88 of 102 restored). Without them, every page re-renders on every build.
  - The pages are restored from `.astro-cache/dist`, a full copy of the HTML
    (1.4 GB at 20k), so the cache is as large as `dist`.
  - Pagefind still re-indexes everything on each build: 50–60 s at 20k.
- **G14. Pagefind is the memory ceiling.** It needs about 0.32 MB per page with
  code included, or 6.3 GB at 20k. The ways to bound it: shard the index (the UI
  can `mergeIndex`), mark code and tables `data-pagefind-ignore` (4.07 GB at 20k),
  or both. `PAGEFIND_BINARY_PATH` is honoured, and the resolver falls back from
  `pagefind_extended` to `pagefind`. Dropping `extended` loses CJK word
  segmentation.
- **G15. A missing favicon returns 404.** Starlight's default `/favicon.svg` is a
  404 without `public/favicon.svg`, so the template now ships one.
- **G16. An empty `docs` collection is fine.** It logs warnings for the missing
  directory and for the empty `docs`/`i18n` collections, and the 404 page is still
  generated.
- **G17. Build concurrency is 1 by default** (`build.concurrency`). Rendering is
  CPU-bound on one JS thread, so wall time scales with single-core speed, not core
  count. This spike did not measure raising it.
- **G18. `--ignore-scripts` is safe.** No needed package relies on an install
  script: the native bindings come as optional platform packages. `sharp` (an
  optional Astro dependency) is pruned; with `passthroughImageService` it is never
  loaded.

## 7. Recommended page-granularity caps

- **One page per module or file (or per Go package), never per symbol.** Symbols
  are cards within the module page. For Kubernetes, pages per package (about 3k)
  instead of per file (about 15k with a parser) put even design A inside the
  budget.
- **Symbol cards per page: at most 50.** Paginate larger modules into
  `…/page-2/`. With K = 8, pages average 45–70 KB of HTML, so 50 cards comes to
  about 250 KB.
- **Pages per site:**
  - **Design A** (Astro renders `.md` fragments): at most 2,500 pages and 20,000
    fragments. That is about 70 s locally, about 2 min on a runner, RSS about
    3.2 GB, and a data store of about 80 MB.
  - **Design D** (Rust pre-rendered HTML, page JSON on disk): at most 20,000 pages.
    Pagefind must be sharded at 8,000 pages or fewer per shard, or given
    `--exclude-selectors pre,table`. Past 20k, rfx should warn and coarsen the
    granularity (group pages by directory).
- **Sidebar:** full tree up to 800 links, pruned above that (already implemented).
- **Mermaid:** keep dep-graphs to module and package pages. At 1 in 20 pages the
  cost is nothing for pages without a diagram.

## 8. What M1 should take from this

- **Keep the template contract.** Rust writes:
  - `pulse.config.json`: title, site, base, tabs, the sidebar tree, and `build`
    knobs.
  - `bundle/index.json`: `{id, tab, title, hash}` per page.
  - `bundle/pages/<id>.json`: the block list.

  For large sites, Rust writes prose inline as `html` plus `headings`
  (`doc_html`/`doc_headings`, `signature_html`), sanitised with the rules in
  `remark-pulse.mjs`. Small sites can keep `.md` fragments.
- **Rust owns highlighting.** Reflex already ships tree-sitter grammars, and
  `syntect` is already in the dependency tree. The output should match
  expressive-code's dual-theme CSS variables, or use a small Pulse CSS.
- **CI caching.** rfx should persist `<site>/.astro-cache` (data store, Vite cache,
  incremental manifest, restored pages) together with `dist`, keyed by the runtime
  version.
- **Before M1:** upstream or patch Astro's `getChunkEnd` (G7), and decide on
  sharded Pagefind (G14).
- **Not measured here:**
  - `build.concurrency > 1`.
  - Mermaid rendering cost on low-end devices.
  - The 4-OS runner numbers: `pulse-spike.yml` is written but was not run.
  - Windows path handling in `bench.sh`, which runs through Git Bash with MSYS path
    conversion.

## 9. Reproduce

```bash
cd pulse-template && npm ci
# runtime tarball + build-from-runtime verification (N=100, all checks)
TMPDIR=/some/scratch scripts/build-runtime.sh /some/scratch/rt --verify
# benchmark: variant A / D at N pages (TIME_BIN = GNU time)
TIME_BIN=/usr/bin/time scripts/bench.sh /some/scratch/rt/verify 5000
FIXTURE_ARGS="--prerender --plain-signatures" PULSE_BUILD='{"pageSource":"fs"}' \
  TIME_BIN=/usr/bin/time scripts/bench.sh /some/scratch/rt/verify 20000
node scripts/check-dist.mjs /some/scratch/rt/verify/sites/bench-100
```

`bench.sh` writes `results/bench-<N>.json`, the cold and warm build logs, and the
GNU time output. The site directory must not have a `node_modules` directory above
it, other than the runtime's.
