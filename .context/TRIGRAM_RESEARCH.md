# Trigram Index Implementation Research

**Date:** 2025-10-31
**Status:** Architecture Design Complete, Ready for Implementation
**Related:** See CLAUDE.md for project vision, TODO.md for task breakdown

---

## Overview

Reflex is implementing **trigram-based full-text code search** modeled after Sourcegraph's Zoekt and Google Code Search.

**Goal:** Enable <100ms queries that find **every occurrence** of patterns across 10k+ files.

---

## What is Trigram Indexing?

### Core Concept

A **trigram** is a sequence of 3 consecutive characters:
- `"extract_symbols"` → trigrams: `["ext", "xtr", "tra", "rac", "act", "ct_", "t_s", "_sy", "sym", "ymb", "mbo", "bol", "ols"]`

### Inverted Index Structure

Build a mapping from each trigram to the locations where it appears:

```
Inverted Index:
"ext" → [(file1, line3), (file2, line15), (file5, line8)]
"xtr" → [(file1, line3), (file5, line8)]
"tra" → [(file1, line3), (file3, line42), (file5, line8)]
...
```

### Search Algorithm

When searching for `"extract_symbols"`:

1. **Extract trigrams** from query: `["ext", "xtr", "tra", ..., "ols"]`
2. **Lookup posting lists**: Get file/line locations for each trigram
3. **Intersect lists**: Find locations that contain ALL trigrams
4. **Verify match**: Check actual content at candidate locations
5. **Return results**: With surrounding context

**Performance:** Reduces search from thousands of files to ~10-100 candidates (100-1000x speedup).

---

## Why Trigrams Work

### Key Properties

1. **Small alphabet**: Only 256³ = 16M possible trigrams; most are rare
2. **Discriminative**: Long strings have unique trigram combinations
3. **Substring-friendly**: Any substring >3 chars has trigrams that must appear
4. **Regex-friendly**: Can extract guaranteed trigrams from many patterns
5. **Fast intersection**: Posting lists are small; intersections are quick

### Example

Search for `"extract_symbols"` (13 trigrams):
- Posting list intersection eliminates 99.9% of files
- Only verify matches in ~10 candidate files
- Total time: <10ms (vs ~100ms full scan)

---

## Implementation Details

### Data Structures

#### 1. Trigram Type
```rust
// Represent trigram as 3 bytes (compact)
pub type Trigram = [u8; 3];

// Or as u32 for faster hashing:
pub type Trigram = u32; // pack 3 bytes into 32-bit int
```

#### 2. File Location
```rust
pub struct FileLocation {
    file_id: u32,      // Index into file list
    line_no: u32,      // Line number (1-indexed)
}
```

One posting per **(trigram, file, line)**. Until V4 (1.8.0) the struct also carried
`byte_offset` and a posting was emitted for every byte position, so a line with
`"aaaa"` held two identical `(file, line)` keys that intersection then threw away.
Nothing on the query path ever read `byte_offset` — line verification re-scans the
line from content.bin — so it was pure index bloat (see "Binary Format" below).

#### 3. Inverted Index
```rust
pub struct TrigramIndex {
    // Map trigram to sorted list of locations
    index: HashMap<Trigram, Vec<FileLocation>>,

    // File ID to file path mapping
    files: Vec<PathBuf>,
}
```

### Binary Format (trigrams.bin) — V4 (1.8.0)

```
Header (32 bytes):
  magic:        "RFTG" (4 bytes)
  version:      4 (u32 LE)
  num_trigrams: N (u64 LE)          -- bytes 8..16, also read by `rfx stats` (cli/misc.rs)
  num_files:    F (u64 LE)
  paths_offset: (u64 LE)            -- absolute offset of the paths section

Directory (N × 16 bytes, sorted by trigram, binary-searched IN the mmap):
  trigram:         u32 LE
  data_offset:     u64 LE           -- absolute
  compressed_size: u32 LE

Data (one posting list per directory entry), a sequence of FILE BLOCKS:
  varint(file_id - prev_file_id)    -- first block: delta from 0
  varint(n_lines << 1 | enc)        -- enc bit RESERVED (writer emits 0; reader errors on 1)
  enc=0: n_lines × varint(line - prev_line), prev_line restarts at 0 in every block

Paths @ paths_offset (F entries):
  varint(len), utf8 bytes
```

Implementation: `src/trigram.rs` — `encode_posting_list` (writer, shared by the
in-memory `write` and the streaming k-way merge), `PostingCursor` (streaming
decoder; `seek` skips the tail of a block by scanning varint continuation bits
when the target file id is larger), `TrigramIndex::find_entry` (directory probe).

**What changed from V3 and why**

| | V3 | V4 |
|---|---|---|
| Posting granularity | every byte position | one per distinct trigram per line |
| Per posting | 3 varints (file Δ, line Δ, byte-offset Δ) | 1 varint (line Δ) inside a file block |
| File boundary cost | line/offset deltas `wrapping_sub` across files → ~5-byte varints | 2 small varints per (trigram, file) block |
| Header | 24 B, no paths offset | 32 B with `paths_offset` |
| `load` | decode whole directory into a `Vec`, re-sort, sum sizes to find paths | header check + bounds check + paths; O(files) |

**Measured (release build, 2026-09-22)**

| Corpus | corpus bytes | V3 trigrams.bin | V4 trigrams.bin | V4 ratio |
|---|---|---|---|---|
| synthetic latency corpus (`synthetic_corpus::indexed(7)`, 2000 `.rs`, 2 611 distinct trigrams) | 32 350 466 | 127 420 820 (3.94x) | 30 281 814 | **0.94x** |
| Reflex repo itself (272 files, 44 246 distinct trigrams) | 3 822 333 | — | 5 313 892 | **1.39x** |

V4 byte breakdown (Reflex repo): directory 13.3 %, block headers 19.9 %, line
deltas 66.6 %, paths 0.1 %. Avg 6.0 lines per block. The synthetic corpus has a tiny
alphabet, so its ratio flatters; the real-code number is the one to watch.

**Reserved `enc=1` (bitmap blocks) — not implemented.** Design if ever needed:
`varint(first_line) varint(nbytes) bitmap[nbytes]` (bit k ⇒ line `first_line+k`
present), chosen by the writer when `n_lines * 8 > max_line - min_line + 1`. Measured
gain on the Reflex repo: 7 789 of 514 563 blocks would qualify, saving ~192 KB
(**3.6 %**, 1.39x → 1.34x); 21 % on the synthetic corpus. Not worth a second decoder
path today. The reader rejects `enc=1` with "unsupported block encoding" rather than
misreading it.

**Compatibility.** `build.rs` hashes `src/trigram.rs` into `CACHE_SCHEMA_HASH`, so a V3
cache reports stale and `rfx index` rebuilds it. `TrigramIndex::load` rejects a V3 file
with "Unsupported trigrams.bin version: 3 (expected 4)…"; `query/open_index.rs` keys on
that message to serve the process from an in-memory rebuild until `rfx index` runs.

### Content Store (content.bin)

```
Header (32 bytes):
  magic: "RFCT" (4 bytes)
  version: 1 (u32)
  num_files: F (u64)
  index_offset: offset to file index (u64)
  reserved: 12 bytes

File Index:
  For each file:
    offset: u64 (byte offset to file content)
    length: u64 (file size in bytes)

File Contents:
  [Concatenated file contents]
```

**Design rationale:** Memory-map content.bin for zero-copy access to file contents.

---

## Trigram Extraction Algorithm

### Basic Extraction

```rust
fn extract_trigrams(text: &str) -> Vec<Trigram> {
    let bytes = text.as_bytes();
    let mut trigrams = Vec::new();

    for i in 0..bytes.len().saturating_sub(2) {
        let trigram = [bytes[i], bytes[i+1], bytes[i+2]];
        trigrams.push(trigram);
    }

    trigrams
}
```

### With Line Tracking (per-line dedup, V4)

```rust
fn extract_trigrams_with_locations(text: &str, file_id: u32) -> Vec<(Trigram, FileLocation)> {
    let bytes = text.as_bytes();
    let mut result = Vec::with_capacity(bytes.len().saturating_sub(2));
    let mut line_trigrams: Vec<Trigram> = Vec::with_capacity(128); // scratch for one line
    let mut line_no = 1;

    for (i, &byte) in bytes.iter().enumerate() {
        if byte == b'\n' {
            // flush: sort_unstable + dedup, emit one (trigram, {file_id, line_no}) each
            flush(&mut line_trigrams, file_id, line_no, &mut result);
            line_no += 1;
        }
        if i + 2 < bytes.len() {
            line_trigrams.push(bytes_to_trigram(&bytes[i..i + 3]));
        }
    }
    flush(&mut line_trigrams, file_id, line_no, &mut result);
    result
}
```

**Decision (V4): deduplicate per line at extraction, not at `finalize`.** A trigram
that occurs k times on one line is one posting. The intersection key was always
`(file_id, line_no)`, so the duplicates never contributed a result; they only
inflated posting lists (16 spaces of indentation alone repeated `"   "` 14 times per
line). Dedup here means `finalize`'s `dedup()` is a no-op and the sort is cheaper.

Line attribution is unchanged from V3: a trigram belongs to the line of its first
byte, except that a trigram *starting* on `\n` belongs to the next line (the `\n` is
consumed before the trigram is pushed). Trigrams spanning a newline are still
indexed, so a query such as `"lo\n"` keeps working.

---

## Query Processing

### Plain Text Query

```rust
fn search_plain_text(query: &str, index: &TrigramIndex) -> Vec<Match> {
    if query.len() < 3 {
        // Fall back to full scan for short queries
        return full_scan(query);
    }

    // Step 1: Extract trigrams from query
    let trigrams = extract_trigrams(query);

    // Step 2: Get posting lists for each trigram
    let mut posting_lists: Vec<&Vec<FileLocation>> = trigrams
        .iter()
        .filter_map(|t| index.index.get(t))
        .collect();

    if posting_lists.is_empty() {
        return vec![];
    }

    // Step 3: Sort by list size (smallest first for efficient intersection)
    posting_lists.sort_by_key(|list| list.len());

    // Step 4: Intersect posting lists
    let candidates = intersect_posting_lists(posting_lists);

    // Step 5: Verify actual matches
    let mut results = Vec::new();
    for loc in candidates {
        if verify_match_at_location(query, loc) {
            results.push(create_match_result(loc));
        }
    }

    results
}
```

### Regex Query

```rust
fn search_regex(pattern: &str, index: &TrigramIndex) -> Vec<Match> {
    // Step 1: Extract guaranteed trigrams from regex
    let trigrams = extract_trigrams_from_regex(pattern);

    if trigrams.is_empty() {
        // Regex has no literals → fall back to full scan
        return regex_full_scan(pattern);
    }

    // Step 2: Use trigrams to narrow candidates
    let candidates = search_by_trigrams(&trigrams, index);

    // Step 3: Verify with actual regex engine
    let regex = Regex::new(pattern).unwrap();
    let mut results = Vec::new();
    for loc in candidates {
        let content = get_file_content(loc.file_id);
        if regex.is_match(content) {
            results.push(create_match_result(loc));
        }
    }

    results
}
```

---

## Posting List Intersection

### Naive Intersection (O(n*m))
```rust
fn intersect_two_lists(a: &[FileLocation], b: &[FileLocation]) -> Vec<FileLocation> {
    a.iter()
        .filter(|loc| b.contains(loc))
        .cloned()
        .collect()
}
```

### Optimized Intersection (O(n+m))
```rust
fn intersect_sorted_lists(a: &[FileLocation], b: &[FileLocation]) -> Vec<FileLocation> {
    let mut result = Vec::new();
    let (mut i, mut j) = (0, 0);

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
        }
    }

    result
}
```

**Optimization:** Store posting lists sorted by (file_id, line_no) for linear-time intersection.

---

## Regex Trigram Extraction

### Algorithm

Extract the longest literal substrings from the regex pattern.

### Examples

| Regex Pattern | Extracted Trigrams | Notes |
|---------------|-------------------|-------|
| `extract_symbols` | `["ext", "xtr", "tra", ...]` | All literal |
| `fn\s+extract` | `["fn ", "ext", "xtr", ...]` | Literals + space |
| `Google.*Search` | `["Goo", "oog", "gle", "Sea", "ear", "rch"]` | Both ends are literals |
| `a(bc)+d` | `["abc", "bcb", "bcd"]` | Repetition generates alternatives |
| `if\|else` | `["if ", "els", "lse"]` | Alternation → multiple options |
| `.*` | `[]` | No literals → full scan |

### Implementation Sketch

```rust
fn extract_trigrams_from_regex(pattern: &str) -> Vec<Trigram> {
    // This is simplified; real implementation needs regex parsing

    // Strategy:
    // 1. Parse regex AST
    // 2. Find all literal sequences
    // 3. Extract trigrams from literals
    // 4. Handle special cases (^, $, \b, etc.)

    // For MVP: extract longest contiguous literal substring
    extract_longest_literal(pattern)
        .and_then(|lit| Some(extract_trigrams(&lit)))
        .unwrap_or_default()
}
```

---

## Performance Characteristics

### Index Size

- **Trigram count**: ~20-30 trigrams per 100 characters of code
- **Posting list size**: Avg 10-100 locations per trigram
- **Total index size (measured, V4)**: ~0.9x–1.4x of source bytes — 0.94x on the
  synthetic latency corpus, 1.39x on the Reflex repo. V3 was 3.9x. `rfx index` prints
  the live number as `Index/corpus ratio: 1.4x (trigrams.bin …, content.bin …)`
  (`IndexStats::{corpus_bytes, trigram_index_bytes}`).
- The original ~20 % estimate assumed file-level postings; Reflex keeps line-level
  postings so intersection yields lines, not files.

### Query Performance

| Query Type | Trigram Count | Candidates | Time |
|------------|---------------|------------|------|
| Long literal (`extract_symbols`) | 13 | ~10 files | <10ms |
| Regex with literals (`fn.*test`) | 3-5 | ~100 files | <20ms |
| Short pattern (`if`) | 0 | All files | ~100ms |
| Wildcard (`.*`) | 0 | All files | ~100ms |

### Space/Time Trade-offs

- **More trigrams indexed** → larger index, faster queries
- **Compressed posting lists** → smaller index, slower decompression
- **Memory-mapped I/O** → zero-copy, fast cold start

**Decision for Reflex:** Uncompressed posting lists for <100ms queries.

---

## References

1. **Russ Cox - Regular Expression Matching with a Trigram Index**
   - https://swtch.com/~rsc/regexp/regexp4.html
   - Describes Google Code Search implementation

2. **Zoekt - Sourcegraph's Code Search Engine**
   - https://github.com/sourcegraph/zoekt
   - Production trigram-based search

3. **PostgreSQL pg_trgm Module**
   - Uses trigrams for full-text search
   - Provides reference implementation

---

## Open Questions & TODOs

- [ ] Should we case-fold trigrams? (e.g., "Ext" → "ext")
  - **Recommendation:** No, keep case-sensitive for deterministic results

- [ ] How to handle Unicode?
  - **Recommendation:** UTF-8 bytes, trigrams can span character boundaries

- [ ] Should posting lists be compressed?
  - **Recommendation:** Not for MVP (optimize later if index too large)

- [ ] How to handle very long posting lists (common trigrams)?
  - **Recommendation:** Skip or truncate lists >10k entries (rare trigrams are more useful)

---

**END OF TRIGRAM_RESEARCH.md**
