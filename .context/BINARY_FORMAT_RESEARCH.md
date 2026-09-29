# Binary Format Reference

**Created:** 2025-10-31 · **Rewritten:** 2026-09-28 (Reflex 2.0.3)

The on-disk formats in `.reflex/`. Each section names the source file that owns the
format; when this file and the code disagree, the code wins.

| File | Format | Owner |
| --- | --- | --- |
| `trigrams.bin` | custom varint inverted index, **V4** (§3) | `src/trigram.rs`, `src/trigram_build.rs` |
| `content.bin` | concatenated file contents + fixed-width index, **V2** (§1) | `src/content_store.rs` |
| `meta.db` | SQLite: files + fingerprints, branches, statistics, config, dependencies, exports, symbol cache (§2) | `src/cache.rs`, `src/symbol_cache.rs` |
| `config.toml` | TOML project settings | `src/cache.rs` (template), `src/models.rs` |

Both binary files are memory-mapped (`memmap2`) and written atomically
(`src/atomic_write.rs`: tmp + fsync + rename) under the `.reflex/index.lock` advisory lock.

---

## 1. content.bin (V2, 2.0.0)

```text
Header (32 bytes):
  magic: "RFCT" (4 bytes)
  version: 2 (u32)
  num_files: N (u64)
  index_offset: offset to file index (u64)
  reserved: 8 bytes

File contents: concatenated, in file_id order

File index (at index_offset):
  Entry table, num_files × 28 bytes, addressable by file_id:
    offset: u64    (byte offset of the content, relative to the header end)
    length: u64    (content size in bytes)
    path_pos: u64  (absolute file position of the path bytes)
    path_len: u32
  Path blob: the UTF-8 paths, concatenated, in file_id order
```

V1 stored `(path_len, path, offset, length)` per file, so open had to decode the whole
index (12 ms on a 24k-file checkout). V2 reads an entry at `index_offset + file_id * 28`:
open is O(1). Non-UTF-8 text files are stored after lossy decoding.

## 2. meta.db symbol cache (symbol format v3, 2.0.0)

Table `symbols (file_id, file_hash, symbols_json, last_cached)`, primary key
`(file_id, file_hash)`, `ON DELETE CASCADE` from `files`. Written by the background pass
(`rfx index-symbols-internal`); read first by every symbol query, which parses misses on demand.

`symbols_json` holds `encode_symbols` output:
- blobs of 256 bytes or more: 4-byte magic `FF 'R' 'Z' 01` + zstd (level 3) of the JSON;
- shorter blobs: raw JSON (starts with `[`, so the two encodings never collide).

`SYMBOL_FORMAT_VERSION` history: v2 (1.7.2) bounded previews; v3 (2.0.0) zstd blobs.
A stored version that differs invalidates the cache. Files with no symbol parser (text
tiers, Swift) have no row.

The `files` table also carries the freshness fingerprint (`size`, `mtime_ns`, `hash`,
`dirty_at_index`) used by the content-based freshness check.

## Versioning

- Each binary file starts with magic bytes and a format version.
- A V3 `trigrams.bin` is served from an in-memory rebuild (slow) until `rfx index` runs;
  `rfx index` detects the schema change and rebuilds in full.
- Readers degrade, writers refuse: a cache owned by a different released Reflex version
  is not rewritten by a reader (see `.context/TODO.md`, "Current policy").

---

## 3. trigrams.bin

**Purpose:** trigram → sorted `(file_id, line_no)` postings for candidate narrowing
**Format:** custom; fixed-width header + directory, varint posting lists
**Access:** memory-mapped; directory binary-searched in place, posting lists decoded on demand
**Source:** `src/trigram.rs` (`encode_posting_list`, `PostingCursor`, `TrigramIndex::{write,load,find_entry}`)

#### Structure

```
┌──────────────────────────────────────────────────────────────────────┐
│ Header (32 B)                                                        │
│   "RFTG" | version u32 = 4 | num_trigrams u64 | num_files u64 |      │
│   paths_offset u64                                                   │
├──────────────────────────────────────────────────────────────────────┤
│ Directory: num_trigrams × 16 B, sorted by trigram                    │
│   trigram u32 | data_offset u64 (absolute) | compressed_size u32     │
├──────────────────────────────────────────────────────────────────────┤
│ Data: per trigram, a sequence of FILE BLOCKS                         │
│   varint(file_id − prev_file_id)        first block: delta from 0    │
│   varint(n_lines << 1 | enc)            enc reserved, writer emits 0 │
│   enc=0: n_lines × varint(line − prev_line), prev_line = 0 per block │
├──────────────────────────────────────────────────────────────────────┤
│ Paths @ paths_offset: num_files × { varint(len), utf8 }              │
└──────────────────────────────────────────────────────────────────────┘
```

All integers little-endian. Bytes 8..16 (`num_trigrams`) are read directly by
`rfx stats` (`cli/misc.rs`); keep that offset stable across versions.

#### Design decisions

| Decision | Rationale |
|---|---|
| **Drop `byte_offset` from postings** | Never read at query time (line verification re-scans the line from content.bin). It cost a third varint per posting and, worse, a wrapped ~5-byte delta at every file boundary. |
| **One posting per distinct trigram per line** | The intersection key is `(file_id, line_no)`; per-byte duplicates were discarded at query time after being paid for on disk. Dedup happens in `extract_trigrams_with_locations` with a per-line scratch `Vec` (`sort_unstable` + `dedup`). |
| **Per-file blocks with per-block line deltas** | Lines restart at 0 in each block, so the cross-file "wrap" never happens; a (trigram, file) pair costs 2 small varints of header. |
| **`paths_offset` in the header** | `load` no longer decodes the whole directory to sum `compressed_size`s. Load is O(files) and allocates only the path list. |
| **Directory searched in the mmap** | `find_entry` reads the 4-byte trigram at each probe (log₂ N probes, ≈16 for 44k trigrams) and decodes one 16-byte entry on a hit. No `Vec<DirectoryEntry>`, no re-sort. |
| **`enc` bit reserved** | Leaves room for bitmap blocks (`varint(first_line) varint(nbytes) bitmap`) for dense trigrams without a version bump. Reader errors on `enc=1` ("unsupported block encoding"). Measured benefit today: 3.6 % on the Reflex repo, 21 % on the synthetic corpus — not implemented. |
| **Bounds check at load** | `32 + 16·n ≤ paths_offset ≤ len`, else "layout out of bounds". This is the only guard the in-place directory search relies on. |

#### Measured sizes

| Corpus | corpus bytes | V3 | V4 | ratio |
|---|---|---|---|---|
| synthetic latency corpus (2000 files, 2 611 trigrams) | 32 350 466 | 127 420 820 (3.94x) | 30 281 814 | 0.94x |
| Reflex repo (272 files, 44 246 trigrams) | 3 822 333 | — | 5 313 892 | 1.39x |

V4 byte breakdown on the Reflex repo: directory 13 %, block headers 20 %, line
deltas 67 %. `rfx index` prints `Index/corpus ratio: …` from
`IndexStats::{trigram_index_bytes, corpus_bytes}` (corpus = content.bin
`index_offset − 32`).

#### Writers

- `TrigramIndex::write` (in-memory): encodes every list first, so directory offsets
  and `paths_offset` are known before the header is written; single pass.
- `TrigramIndexBuilder` (`src/trigram_build.rs`, used by `rfx index`): builds each batch
  per trigram shard into V4-encoded partials whose records carry `first_file_id` /
  `last_file_id`, then merges them by byte copy, rewriting only each record's first
  file delta. Preconditions: partials cover disjoint, increasing file-id ranges (not
  checked); the trigram count is known up front. Output is byte-identical to
  `TrigramIndex::write`. Partials are temporary, not a stored format.
- Note: `src/trigram_build.rs` is not in `build.rs`'s schema-hash list (only
  `src/trigram.rs` is).

#### Versioning

`build.rs` folds `src/trigram.rs` into `CACHE_SCHEMA_HASH`; any format edit makes an
existing cache report stale and forces a full rebuild on `rfx index`. `load` rejects
other versions with `Unsupported trigrams.bin version: {v} (expected 4). Please
re-index with 'reflex index'.` — `query/open_index.rs` matches on that text to fall
back to an in-memory rebuild for the process, and treats any other load error as
`CacheCorrupted`.

---

---

## Design history

The 2025-10-31 design used `symbols.bin` (rkyv, zero-copy), `tokens.bin` (zstd) and
`hashes.json`. All three were removed: symbols moved to runtime parsing (2025-11-03) and
then to the zstd blob cache in `meta.db` (2.0.0); per-file hashes live in `meta.db`.
`trigrams.bin` V1–V3 carried a per-posting `byte_offset` and were not grouped by file.
The full original design is in git: `git show fc8da6b:.context/BINARY_FORMAT_RESEARCH.md`.
