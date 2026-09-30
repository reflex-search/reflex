# Performance Research & Baselines

> Every section is dated. Only the 2026-09-22/23 sections describe the current (2.0.x)
> index format; earlier numbers predate `trigrams.bin` V4 and the WP1–WP7 rounds.

## Criterion Benchmark Baseline (2026-05-13) — pre-2.0, historical

Measured on the `feature/code-quality-refactor` branch.
Run with: `cargo bench --bench trigram_bench`

### trigram_extraction

Measures raw trigram extraction throughput from a 10 KB synthetic Rust source.

| Variant | Min | Median | Max |
|---------|-----|--------|-----|
| `no_locations_10kb` | 7.619 µs | **7.698 µs** | 7.783 µs |
| `with_locations_10kb` | 15.99 µs | **16.20 µs** | 16.47 µs |

**Takeaway**: Adding location tracking roughly doubles extraction cost (~2×), which is acceptable since it is only triggered during indexing, not queries.

### posting_list_intersection

3-trigram intersection across a synthetic 10 K-file index. Pattern `"abcde"` generates
`abc`→5 000 entries, `bcd`→5 000 entries, `cde`→5 000 entries; expected result ~2 500 files.

| Variant | Min | Median | Max |
|---------|-----|--------|-----|
| `3gram_10k_files` | 2.758 ms | **2.840 ms** | 2.934 ms |

**Takeaway**: Intersection for a 3-trigram pattern against 10 K files costs ~2.8 ms. The HashSet-based
implementation allocates once per list pair; see `Algorithmic complexity` lens. Large posting lists
from high-frequency trigrams (e.g., `" th"`) dominate this cost — use `max_posting_list_entries` to cap.

### index_and_query_roundtrip

Full index build + query on 1 000 synthetic Rust files (10 samples due to I/O expense).

| Variant | Min | Median | Max |
|---------|-----|--------|-----|
| `1k_files` | 776.7 ms | **804.2 ms** | 840.5 ms |

**Takeaway**: Indexing 1 K files costs ~800 ms (dominated by disk I/O and content hashing). This benchmark intentionally deletes the `.reflex/` cache each iteration to simulate cold-index builds.

### symbol_query_tree_sitter

Parses 100 candidate Rust files through tree-sitter to extract symbol definitions.
This simulates the symbol-query hot path when the trigram filter returns 100 candidates.

| Variant | Min | Median | Max |
|---------|-----|--------|-----|
| `100_candidates_rust` | 1.651 s | **1.671 s** | 1.692 s |

**Takeaway**: ~16.7 ms per candidate file for tree-sitter Rust parsing. This confirms the
`Tree-sitter query performance` lens: AST queries are ~1 000× slower than trigram search
(2 µs per trigram extraction vs 16.7 ms per tree-sitter parse). Always require `--glob` with `--ast`.

## Analysis

- **Query hot path** (trigram extraction only): <10 µs for 10 KB, scales linearly with source size.
- **Symbol hot path** (trigram + tree-sitter): trigram gets you to ~10–100 candidates in <1 ms, then tree-sitter adds ~16.7 ms/file overhead.
- **Cold indexing**: ~800 ms for 1 K files → ~50 K files/minute throughput. Parallel rayon indexing makes this viable.
- **Posting list** budget: 2.8 ms for a 10 K-file corpus with dense trigrams. For 100 K-file corpora this would be ~28 ms; use `max_posting_list_entries` to keep query latency under 10 ms.

## Benchmark Design Decisions

- **Bench 1 (extraction)**: Uses `make_rust_source(10_240)` which generates deterministic Rust with realistic identifier density. The "no locations" variant mirrors the query path; "with locations" mirrors the index path.
- **Bench 2 (intersection)**: Synthetic posting lists chosen to produce ~50% overlap, stress-testing the intersection algorithm without needing real files.
- **Bench 3 (roundtrip)**: `sample_size(10)` due to disk I/O; uses `TempDir` with per-iteration cache deletion to guarantee cold starts.
- **Bench 4 (tree-sitter)**: 100 files × ~200 lines each. Realistic: trigram filter would return ~10–100 candidates on a large codebase. Rust grammar chosen as the most mature and commonly benchmarked.

## Latency harness baseline (2026-09-22, HEAD before WP1–WP4)

Measured with `tests/latency_budget.rs` on HEAD `b4b7def` (v1.7.2):

```text
cargo test --release --test latency_budget -- --ignored --nocapture --test-threads=1
```

**Setup**: deterministic synthetic corpus (`test_helpers::synthetic_corpus`, seed 7):
2000 `.rs` files, 34 MB source, 154 MB `.reflex/` index. Machine: AMD Ryzen 7 7840HS,
16 logical cores, Linux. Page cache warm (corpus generated and indexed in an earlier
run; ~15 s for generate + index). N = 11 runs per shape; `first` is the first of the
11, `median`/`p90` over all 11 (nearest-rank p90). All times in ms.

The in-process harness builds a fresh `CacheManager` + `QueryEngine` per call, exactly
as the MCP `search_code` handler does, so the MCP − in-process delta is the stdio
round-trip (JSON-RPC framing, result serialisation, freshness check) and nothing else.

### in_process

| shape | hits | first | median | p90 |
|---|---:|---:|---:|---:|
| zero_hit | 0 | 5.10 | 3.77 | 4.08 |
| rare_ident | 3 | 4.41 | 4.89 | 5.47 |
| common_ident_limit1 | 26969 | 3689.56 | 3689.56 | 4834.53 |
| common_word_limit1 | 52010 | 781.09 | 742.91 | 827.50 |
| common_word_count | 52010 | 901.49 | 796.68 | 942.38 |
| regex_getset | 4000 | 457.89 | 483.88 | 528.47 |

### mcp (real `rfx mcp` child over stdio)

| shape | hits | first | median | p90 |
|---|---:|---:|---:|---:|
| zero_hit | 0 | 10.74 | 10.10 | 11.46 |
| rare_ident | 3 | 7.30 | 6.59 | 7.28 |
| common_ident_limit1 | 26969 | 3902.71 | 3599.60 | 3866.76 |
| common_word_limit1 | 52010 | 669.01 | 672.41 | 732.51 |
| common_word_count | 52010 | 2682.15 | 730.03 | 2326.41 |
| regex_getset | 4000 | 447.65 | 438.57 | 464.05 |

`initialize_ms`: 3.48 · `first_call_ms` (zero_hit, includes cold open): 14.08

Parity: every shape's hit count matched a `\b…\b` line scan of the generated files in
both harnesses (the harness asserts this unconditionally).

### Budgets that would fail today (`REFLEX_LATENCY_BUDGET=1`)

Budgets were **not** tuned; they encode the target, not the baseline. Medians vs budget:

| shape | in-process budget | in-process median | MCP budget | MCP median |
|---|---:|---:|---:|---:|
| zero_hit | 5 | 3.77 ✓ | 20 | 10.10 ✓ |
| rare_ident | 20 | 4.89 ✓ | 35 | 6.59 ✓ |
| common_ident_limit1 | 50 | **3689.56 ✗** | 65 | **3599.60 ✗** |
| common_word_limit1 | 50 | **742.91 ✗** | 65 | **672.41 ✗** |
| common_word_count | 150 | **796.68 ✗** | 165 | **730.03 ✗** |
| regex_getset | 300 | **483.88 ✗** | 315 | **438.57 ✗** |

So the CI budget step is expected to fail until WP1–WP4 land; that is the point.

### Observations

- **`limit: 1` buys nothing.** `common_ident_limit1` (3.7 s) and `common_word_limit1`
  (0.74 s) cost the same as fetching everything: `search_with_metadata` materialises
  and sorts the full result set, then `truncate(limit)` (`src/query/mod.rs`, "Step 6").
  `pagination.total` is computed from the full set, so an early-exit would have to
  give that up or count separately.
- **`ident_7` is 5× slower than `config` despite half the hits.** Every one of the
  20 000 `ident_N` identifiers shares the trigrams `ide`/`den`/`ent`/`nt_`, so the
  posting lists are near-universal and the candidate set is essentially the whole
  corpus; the cost is line-by-line `\b` verification, not the trigram intersection.
  Real codebases have the same shape (`get_`, `set_`, `_id`, `handle`…).
- **The stdio transport is cheap.** MCP − in-process is ≈ 2–6 ms on the small shapes
  and within noise on the large ones; the MCP path is even slightly faster on
  `common_word_limit1` (columnar serialisation of 1 row vs the in-process test's own
  overhead). Cold open of the child is ~14 ms including `initialize`. The process
  boundary is not where the field-test latency comes from.
- **`common_word_count` over MCP has a bimodal tail**: 2 of 11 runs at 2.3–2.7 s
  against a 730 ms median (p90 2326). Not seen in-process in this run; worth
  re-checking after WP1 rather than chasing now.
- **Index is 4.5× the source** (154 MB for 34 MB). `content.bin` is a full copy;
  the rest is `trigrams.bin`. Synthetic identifiers are near-random so this is a
  pessimistic ratio, but it is the one the field test on a 30 MiB tree will see.
- **Both `#[ignore]` tests must run with `--test-threads=1`.** In the first (parallel)
  run the in-process medians were 5–15 % higher because the MCP child competed for
  the same cores. The CI step pins `--test-threads=1`.

## Latency harness after WP1–WP3 (2026-09-22, same box, warm cache, `--test-threads=1`)

Commits: WP0 harness `2a20431`, WP2 open-index handle `b18ae06`, WP1 intersection
`da731e0`, WP3 early termination + parallel line-restricted regex (this commit).
`hits` with a `+` is a lower bound: the search stopped once the page was full.

### in_process

| shape | hits | first | median | p90 | baseline median |
|---|---:|---:|---:|---:|---:|
| zero_hit | 0 | 1.73 | 0.03 | 0.05 | 3.77 |
| rare_ident | 3 | 0.18 | 0.15 | 0.18 | 4.89 |
| common_ident_limit1 | 216+ | 50.43 | 48.23 | 50.43 | 3689.56 |
| common_word_limit1 | 461+ | 2.95 | 2.67 | 2.98 | 742.91 |
| common_word_count | 52010 | 24.96 | 24.51 | 25.03 | 796.68 |
| regex_getset | 4000 | 9.15 | 9.84 | 10.60 | 483.88 |

### mcp (real `rfx mcp` child over stdio) — initialize 2.99 ms, first call 1.62 ms

| shape | hits | first | median | p90 | baseline median |
|---|---:|---:|---:|---:|---:|
| zero_hit | 0 | 0.09 | 0.07 | 0.09 | 10.10 |
| rare_ident | 3 | 0.25 | 0.16 | 0.18 | 6.59 |
| common_ident_limit1 | 216+ | 48.91 | 47.57 | 49.41 | 3599.60 |
| common_word_limit1 | 461+ | 2.82 | 2.82 | 2.94 | 672.41 |
| common_word_count | 52010 | 35.88 | 25.99 | 29.97 | 730.03 |
| regex_getset | 4000 | 11.92 | 10.78 | 11.92 | 438.57 |

Where the remaining time goes (`rfx query ident_7 --limit 1 --timing`):
`open 1.3 ms | candidates 45.2 ms | verify 1.1 ms | status 0.6 ms`. The intersection
streams four ~700k-posting lists (`ide`, `den`, `ent`, `nt_`) that do not narrow the
candidate set beyond what `t_7` already gave (105k candidate lines). Next step: stop
intersecting when the candidate set is small relative to the next list, and let the
(exact) line verification absorb the difference — planned on top of the V4 format.

## Latency harness after WP1–WP4 + intersection stop rule (2026-09-22, final)

V4 index on disk (`trigrams.bin` 30.3 MB for a 32.4 MB corpus, ratio 0.9x; 127 MB before).
`hits` with `+` = lower bound (page filled, verification stopped).

| shape | in-process median | MCP stdio median | baseline in-process | baseline MCP |
|---|---:|---:|---:|---:|
| zero_hit | 0.03 | 0.09 | 3.77 | 10.10 |
| rare_ident | 0.16 | 0.23 | 4.89 | 6.59 |
| common_ident_limit1 | 2.53 | 2.60 | 3689.56 | 3599.60 |
| common_word_limit1 | 3.16 | 3.56 | 742.91 | 672.41 |
| common_word_count | 30.23 | 31.95 | 796.68 | 730.03 |
| regex_getset | 12.13 | 11.71 | 483.88 | 438.57 |

MCP first call (cold open of the V4 index): 2.29 ms. `rfx query ident_7 --limit 1 --timing`:
`open 1.6 | candidates 1.7 | verify 1.9 | status 0.5 ms` — the intersection now stops after
the smallest list (`Intersection stopped early: 105497 candidates, next list 614023 bytes`).
In count mode the remaining time is building 27k–52k result objects (`verify` ~14 ms,
`group` ~13–20 ms on one thread); a paths-only or columnar-direct path would cut that.

Caveats: the synthetic corpus has 2,611 distinct trigrams and is not a git repo (no
`git status` in `status`); on a real repo the memoised status check adds ~10 ms to the
first call in each `REFLEX_FRESHNESS_TTL_MS` window (the CLI pays it on every run).

## Indexing throughput round (2026-09-23)

Baseline and result on a scratch clone of Kubernetes (27,448 indexed files, 245 MB of
text, 582 binary skipped; 16 cores, NVMe; release build; `RUST_LOG=info rfx index --quiet`).

| phase | 2.0.0 (a32b456) | after |
| --- | ---: | ---: |
| discovery | 1 s | 0.82 s |
| batch loop (read/hash/imports in pool + trigram build + flushes) | 22 s | 4.18 s (pool 3.45 s, sharded build 0.64 s) |
| files + branch tx | 1 s | 0.88 s |
| dependencies + exports (97,151 rows) | 504 s | 1.12 s |
| trigram merge/write (289 MB, 328,521 trigrams) | 5 s | 0.21 s |
| total | 532 s | 7.65 s |
| user / sys CPU | 357 s / 88 s | 37 s / 2.9 s |
| peak RSS (zsh `%M`) | 1367 MB | 1045 MB |

Root cause of the 504 s: `DependencyIndex::get_file_id_by_path` opened a connection per
call and, on an exact-match miss, ran `SELECT id, path FROM files WHERE path LIKE '%' || ?`
— a full scan of 27k rows (~9 ms) for each of ~53k unresolved internal imports. On the
Linux kernel the scan is over ~80k rows per miss.

The serial trigram build ran at ~11 MB/s (`HashMap` entry per posting on the main
thread); the merge decoded and re-encoded every list and then `read_to_end` the data
section (the size of `trigrams.bin`) to insert the directory.

Verification: `cmp` of `trigrams.bin` and `content.bin` against the baseline files is
identical for the whole tree; dependency row counts per type are identical
(external 9,580 / internal 53,538 (85 resolved) / stdlib 34,033).

What is left (k8s): 3.4 s in the pool is dominated by tree-sitter parsing of every Go
file for imports; the files transaction's per-file `SELECT id` (0.9 s); the discovery
walk (0.8 s, serial `ignore::Walk`).

## Background symbol pass round (2026-09-23)

Same Kubernetes clone. `rfx index-symbols-internal` after `DELETE FROM symbols`:

| step | wall | user CPU | notes |
| --- | ---: | ---: | --- |
| 2.0.0 | 44.9 s | 131 s | 5 threads; 40.3 s "parse" incl. serial per-file cache check; 256 MB JSON |
| + cached queries + byte-offset previews | 20.1 s | 40 s | still 5 threads, serial check, per-batch writes |
| + streaming writer, 8 threads, zstd, skip text tiers | 6.4–8.3 s | 49–64 s | load-dependent (browser + editor at load ~6); parse 63 s, encode 1.7 s |
| + one combined query per language | **3.4 s** | **25 s** | parse 24 s, encode 1.5 s; blob 28.6 MB |

Bench (`examples/symbol_bench.rs`, deleted after use) over the first 4,000 Go files:
tree-sitter parse 3.25 s vs full parse+extract 10.66 s before the combined query
(extraction 70%), 5.18 s after (extraction 1.73 s, 33%). Per-file: 6 query passes over
the whole tree cost more than parsing it.

Query path effect (latency harness, budgets on): `symbol_lookup` 7.1 → 2.9 ms median,
`find_references` 14.2 → 6.2 ms — cache misses parse with the same combined query.

## Incremental index round (2026-09-29, branch `feature/incremental-index`)

Kubernetes scratch clone (`git clone --no-hardlinks`, 27,448 files, 245 MB), 16 cores.
The machine was never idle (desktop apps keep the 1-minute load at 4–8); every number
has its load average. Base = 2.0.3 (`fff4f5a`, `/scratch/cache/rfx-pre-incremental`),
new = `f30bce6`; runs alternate (`benches/incremental/perf.sh`, 2 rounds × 3 runs each,
22:27–22:33 UTC).

### `rfx index` (A/B, load 5.5–10.8)

| scenario | 2.0.3 | `f30bce6` | gate |
| --- | --- | --- | --- |
| cold (6 runs) | 6.76–8.66 s, median 8.46 s | 7.35–7.80 s, median 7.47 s | ≤ +5 %: yes (−12 %) |
| cold peak RSS | median ~1,089 MB | median ~1,059 MB | not higher: yes |
| nothing changed | 0.83–0.96 s | 0.19–0.24 s | see below |
| 1-file edit | 8.23–9.38 s, ~1.06 GB | 0.28–0.32 s, ~71 MB | < 1.5 s: yes |

The fastest single cold run is 2.0.3's (each round's first run after an idle pause);
by median and mean the new build is faster (one discovery walk feeds the config walk,
git state and the stored rows in parallel; `RETURNING id`; no per-import lookups).

**Nothing changed**, measured precisely (10 runs, load 3.5–4.6, `date +%s%N`): 186–192 ms
against a walk of 142–148 ms (the discovery walk alone) — **+30–34 %**; against the whole
discovery phase the log reports (walk ∥ git ∥ resolver configs, 148–154 ms), +21–30 %.
The gate (discovery + 20 %) is met only against the looser reading. What remains after
the walk: the meta.db commit (8 ms: the branch row and "Last updated" must be written,
and meta.db runs `synchronous=FULL`), classify (4.5 ms), process start and exit (~5 ms),
the post-walk binary check / manifest / config list (~7 ms), statistics and plan (~6 ms).
Removed on the way (each measured): the schema transaction in `init()` (30 → 4 ms), the
branch-hash load (14 ms) and full branch-row sync (12 ms) on the synced branch, two extra
commits, the statistics joins (22 → 3 ms), the stored-row load now parallel with the walk
(20 ms), the `df` check parallel, a 100 ms symbol-pass poll.

### Library path `Indexer::update_paths` (`examples/update_paths_timing.rs`)

| step | load | 1-file edit (warm) | add / delete |
| --- | --- | --- | --- |
| full lists into the `rfx index` code | 10–20 | 260–500 ms | — |
| change set (`publish_delta`) | 23 | 132–145 ms | 186 / 171 ms |
| rows by index lookups, `walk_seq` probes | 9 | 73–103 ms | 170 / 130 ms |
| resolver cache, `git status` overlapped, one dir fsync fewer (`f30bce6`) | 10 | **51–53 ms** | 90–95 ms |

The first update after an `rfx index` waits ~76 ms for the symbol pass that run spawned
(the yield); a watcher calling `update_paths` spawns none. What remains is durability:
the recent segment's fsyncs (~17 ms), the manifest (~6 ms), the meta.db commit (~7 ms).
"Searchable" adds the query (~70–80 ms, most of it the freshness check's `git status`).

### Merge (1,500 edited Go files, 16.9 MB > the 12.3 MB delta limit), load 8–9

3.42 s against a 7.38 s cold build: 25,948 files taken from the published stores,
1,500 read. Peak RSS 1,129 MB (+5 % over that run's cold build: the stores' pages are
mapped while the new base is built).

### Two tiers (`examples/delta_threshold_timing.rs`, 1,000-file delta live)

| | load | 1-file update |
| --- | --- | --- |
| one delta, rebuilt whole | 9–13 | 191–207 ms (write 128–134 ms) |
| recent + delta | 4–5 | 51–85 ms |

**Query cost of a live delta (skip-pointer decision).** Compared with the base before
the edit, some shapes looked 9–33 % slower — but a fresh base of the *same edited
tree* is just as slow: the edit changed them, not the delta. Delta vs a fresh base of
the same tree (15 runs each, load 4.3–5.1): all shapes −0.7 %, single shapes −9 % …
+6 %; the candidate phase, where the tombstone filter runs, +0.01 … +1.2 ms (within its
own run-to-run spread). No skip pointers.

### `latency_budget` (synthetic 2,000-file corpus; 12 alternating runs each, load 4–8)

All 24 runs green with `REFLEX_LATENCY_BUDGET=1`. Sum of the 18 shape medians:
63.18 → 61.31 ms (−2.9 %). Faster: `common_word_limit1` (−22 % / −24 %),
`mcp/common_word_count` (−13 %), `common_ident_limit1` (−5 % / −9 %). Above +5 %:
`in_process/rare_ident` 0.110 → 0.120 ms, `in_process/ci_regex` 0.195 → 0.215 ms,
`mcp/rare_ident` 0.165 → 0.185 ms, `mcp/regex_getset` 5.31 → 5.64 ms. Their per-run
ranges overlap (e.g. `ci_regex` 0.17–0.20 vs 0.18–0.24); `regex_getset` narrows through
the same literals on both binaries (45,679 → 2,000 candidate lines) and its CLI phase
timings overlap. Read as noise; not proven either way.

### Memory (peak RSS, `311b279` against 2.0.3, alternating runs, load 4–23)

| scenario | 2.0.3 | this branch |
| --- | --- | --- |
| cold `rfx index` (8 runs each) | median 1,059 MiB (1,023–1,075) | median 1,030 MiB (984–1,060) |
| `rfx index`, 1-file edit | ~1,034 MiB (a full rebuild) | ~69 MiB |
| `rfx index`, nothing changed | 37.3–37.8 MB | 34.5–34.8 MB |
| change past the merge limit (1,500 files, 3 runs) | 1,031–1,105 MiB (full rebuild) | 936–965 MiB (merge) |
| `rfx mcp` after 3 × `search_code` (absent / `Informer`) | 31.4 / 145.0 MB | 31.4 / 144.3 MB |
| `rfx query`, one shot (3 shapes) | 117–191 MB | +1.0 … +2.5 % (base only), +2.8 … +8.1 % with a 1,000-file delta live |
| process running `update_paths` (150 rounds) | — | levels off at 61–67 MiB |

Found and fixed on the way (both would have failed "peak RSS no higher"): the
no-change run opened the whole snapshot to check it (an 8 MB transient; now a
header check) and kept a full `Metadata` plus an absolute path per walked file
(now 24 bytes and the relative path); the merge left the old base's pages
resident (now `MADV_DONTNEED` after each batch). A one-shot query's +2 MB is fixed
cost: a larger binary (~1 MB resident) and the planning-size file's pages.

## Auto-update round (2026-09-30, branch `feature/auto-update`)

Every command updates a stale index before it answers. Numbers, A/B against `c2e0de2`
(the incremental index), Kubernetes scratch clone, 16 cores, load 16–24: see
`.context/AUTO_UPDATE_RESEARCH.md` ("As built"). Scripts:
`benches/incremental/auto_update.sh <rfx> <label>` and `mcp_edit_latency.py` (prints the
engine's `update_us` / `status_compute_us` per call).

- An edit-then-search costs the check (~60–90 ms, `git status`) + `update_paths` (50–70 ms,
  mostly the recent-tier publish and its fsyncs) + the search. A second check after the
  update cost ~100 ms more until `df8cb1e` settled the verdict from the updated paths.
- `latency_budget`: +1.6 % on the sum of medians, green 4/4 (fresh index: no update runs;
  the engine's handle moved from `OnceLock` to a `Mutex`).
