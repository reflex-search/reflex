# Performance Research & Baselines

## Criterion Benchmark Baseline (2026-05-13)

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

| phase | 1.8.0 (a32b456) | after |
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
| 1.8.0 | 44.9 s | 131 s | 5 threads; 40.3 s "parse" incl. serial per-file cache check; 256 MB JSON |
| + cached queries + byte-offset previews | 20.1 s | 40 s | still 5 threads, serial check, per-batch writes |
| + streaming writer, 8 threads, zstd, skip text tiers | 6.4–8.3 s | 49–64 s | load-dependent (browser + editor at load ~6); parse 63 s, encode 1.7 s |
| + one combined query per language | **3.4 s** | **25 s** | parse 24 s, encode 1.5 s; blob 28.6 MB |

Bench (`examples/symbol_bench.rs`, deleted after use) over the first 4,000 Go files:
tree-sitter parse 3.25 s vs full parse+extract 10.66 s before the combined query
(extraction 70%), 5.18 s after (extraction 1.73 s, 33%). Per-file: 6 query passes over
the whole tree cost more than parsing it.

Query path effect (latency harness, budgets on): `symbol_lookup` 7.1 → 2.9 ms median,
`find_references` 14.2 → 6.2 ms — cache misses parse with the same combined query.
