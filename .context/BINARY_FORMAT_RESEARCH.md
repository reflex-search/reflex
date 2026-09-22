# Binary Format Research & Design

**Created:** 2025-10-31
**Status:** Design Complete
**Decision:** rkyv for symbols, zstd-compressed tokens, SQLite for metadata

---

## Executive Summary

Reflex uses three distinct storage formats optimized for their specific use cases:

1. **symbols.bin** - Zero-copy memory-mapped symbol table (rkyv)
2. **tokens.bin** - Compressed lexical token index (zstd)
3. **meta.db** - Structured metadata and statistics (SQLite)
4. **hashes.json** - File hash cache for incremental indexing (JSON)
5. **config.toml** - User configuration (TOML)

This hybrid approach balances performance, flexibility, and maintainability.

---

## Research Findings

### Serialization Library Comparison

Based on [rust_serialization_benchmark](https://github.com/djkoloski/rust_serialization_benchmark) and [rkyv performance analysis](https://david.kolo.ski/blog/rkyv-is-faster-than/):

| Library | Serialize Speed | Deserialize Speed | Zero-Copy | Random Access | Schema Evolution |
|---------|----------------|-------------------|-----------|---------------|------------------|
| **rkyv** | ⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐⭐ Instant (0-copy) | ✅ Yes | ✅ Yes | ⚠️ Manual |
| **bincode** | ⭐⭐⭐⭐⭐ Fastest | ⭐⭐⭐ Fast | ❌ No | ❌ No | ⚠️ Manual |
| **postcard** | ⭐⭐⭐⭐ Fast | ⭐⭐⭐ Fast | ❌ No | ❌ No | ⚠️ Manual |
| **serde_json** | ⭐⭐ Slow | ⭐⭐ Slow | ❌ No | ✅ Yes | ✅ Easy |

**Decision:** Use **rkyv** for symbols.bin because:
- Zero-copy deserialization is critical for low query latency
- Memory-mapped files can be directly cast to Rust types
- No deserialization overhead on query path
- Excellent for read-heavy workloads (queries >> indexing)

**Trade-offs:**
- More complex than bincode/serde
- Schema evolution requires manual migration
- Larger file size than bincode (~10-20% overhead)

---

## File Format Specifications

### 1. symbols.bin - Zero-Copy Symbol Table

**Purpose:** Store all extracted symbols for fast querying
**Format:** rkyv-serialized binary with custom header
**Access:** Memory-mapped, zero-copy reads

#### Structure

```
┌─────────────────────────────────────────────────────────┐
│ Header (32 bytes)                                       │
├─────────────────────────────────────────────────────────┤
│ Magic Bytes (4 bytes): 0x52464C58 ("RFLX")            │
│ Format Version (u32): 1                                 │
│ Symbol Count (u64): <total symbols>                     │
│ Index Offset (u64): <byte offset to index>             │
│ Reserved (8 bytes): 0x00...                             │
├─────────────────────────────────────────────────────────┤
│ Symbol Data (rkyv-serialized Vec<Symbol>)              │
│   - SymbolKind (enum, 1 byte + alignment)              │
│   - Name (ArchivedString)                               │
│   - Span (4x usize = 32 bytes on 64-bit)               │
│   - Scope (Option<ArchivedString>)                      │
│   - File ID (u32)                                       │
│   - Preview (ArchivedString)                            │
├─────────────────────────────────────────────────────────┤
│ Symbol Index (HashMap<String, Vec<u64>>)               │
│   - Key: Symbol name                                    │
│   - Value: Byte offsets in Symbol Data section         │
└─────────────────────────────────────────────────────────┘
```

#### Symbol Structure (Rust)

```rust
#[derive(Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
pub struct Symbol {
    pub kind: SymbolKind,
    pub name: String,
    pub span: Span,
    pub scope: Option<String>,
    pub file_id: u32,
    pub preview: String,
}
```

#### Read Path (Zero-Copy)

```rust
use memmap2::Mmap;
use rkyv::archived_root;

let file = File::open(".reflex/symbols.bin")?;
let mmap = unsafe { Mmap::map(&file)? };

// Skip 32-byte header
let data = &mmap[32..];

// Zero-copy access to archived data
let symbols = unsafe { archived_root::<Vec<Symbol>>(data) };

// Use symbols directly without deserialization
for symbol in symbols.iter() {
    println!("{}: {}", symbol.name, symbol.kind);
}
```

**Performance:** ~0.1ms to mmap, 0ns to "deserialize", instant access

---

### 2. tokens.bin - Compressed Lexical Token Index

**Purpose:** Enable full-text/fuzzy search across all code tokens
**Format:** Custom binary with zstd compression
**Access:** Sequential reads, decompressed into memory

#### Structure

```
┌─────────────────────────────────────────────────────────┐
│ Header (32 bytes)                                       │
├─────────────────────────────────────────────────────────┤
│ Magic Bytes (4 bytes): 0x52465452 ("RFTK")            │
│ Format Version (u32): 1                                 │
│ Compression Type (u32): 1 (zstd)                        │
│ Uncompressed Size (u64): <bytes>                        │
│ Token Count (u64): <total tokens>                       │
│ Reserved (8 bytes): 0x00...                             │
├─────────────────────────────────────────────────────────┤
│ Compressed Token Data (zstd-compressed)                 │
│   - N-gram Index (HashMap<String, Vec<TokenRef>>)      │
│   - Token Entries (Vec<Token>)                          │
└─────────────────────────────────────────────────────────┘
```

#### Token Structure

```rust
pub struct Token {
    pub file_id: u32,
    pub position: u32,  // Byte offset in source
    pub text: String,
}

pub struct TokenRef {
    pub file_id: u32,
    pub token_id: u32,
}
```

#### N-gram Index

For fuzzy matching, build 3-grams (trigrams):
- "function" → ["fun", "unc", "nct", "cti", "tio", "ion"]
- "getUserData" → ["get", "etU", "tUs", "Use", "ser", "erD", ...]

**Trade-off:** Larger index, but enables substring matching without regex

**Compression:** zstd level 3 (balance between speed and ratio)
- Expected compression: 60-70% (high redundancy in code tokens)
- Decompression speed: ~500 MB/s (fast enough for <100ms queries)

---

### 3. meta.db - SQLite Metadata Database

**Purpose:** Store file metadata, statistics, and index configuration
**Format:** SQLite 3
**Access:** SQL queries via rusqlite

#### Schema

```sql
-- File metadata
CREATE TABLE files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    hash TEXT NOT NULL,              -- blake3 hash
    last_indexed INTEGER NOT NULL,   -- Unix timestamp
    language TEXT NOT NULL,           -- Language enum as string
    symbol_count INTEGER DEFAULT 0,
    token_count INTEGER DEFAULT 0
);

CREATE INDEX idx_files_path ON files(path);
CREATE INDEX idx_files_hash ON files(hash);

-- Cache statistics
CREATE TABLE statistics (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

-- Example statistics:
-- ('total_files', '12543', 1730390400)
-- ('total_symbols', '456789', 1730390400)
-- ('cache_version', '1', 1730390400)
-- ('last_full_index', '1730390400', 1730390400)

-- Index configuration (overrides default config.toml)
CREATE TABLE config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

#### Why SQLite?

**Pros:**
- ✅ Built-in query language (no custom index code)
- ✅ ACID transactions (safe concurrent access)
- ✅ Schema evolution via migrations (ALTER TABLE)
- ✅ Well-tested, battle-hardened
- ✅ Easy to debug (sqlite3 CLI, GUI tools)
- ✅ Small overhead (~100KB for db engine)

**Cons:**
- ⚠️ Slightly slower than custom binary (acceptable for metadata)
- ⚠️ Requires rusqlite dependency

**Performance:** SQLite is fast enough for metadata (~1-10ms queries)
**Decision:** Ease of use outweighs small perf cost for non-critical path

---

### 4. hashes.json - File Hash Cache

**Purpose:** Track file hashes for incremental indexing
**Format:** JSON (serde_json)
**Access:** Load entire file into HashMap

#### Structure

```json
{
  "src/main.rs": "abc123def456...",
  "src/lib.rs": "789ghi012jkl...",
  "tests/integration_test.rs": "mno345pqr678..."
}
```

**Why JSON instead of binary?**
- ✅ Human-readable (debug-friendly)
- ✅ Git-diffable (can see what changed)
- ✅ Simple implementation (serde_json)
- ✅ Small file size (<1MB for 100k files)
- ⚠️ Slower than binary (acceptable, loaded once per index)

**Alternative considered:** Store in meta.db `files` table
- **Rejected:** Hashes need to be loaded before DB updates
- Keeping separate allows atomic compare-and-swap logic

---

### 5. config.toml - User Configuration

**Purpose:** Store user preferences and index settings
**Format:** TOML (serde)
**Access:** Load at indexing time

#### Structure

```toml
[index]
languages = ["rust", "python", "typescript"]
max_file_size = 10485760  # 10 MB
follow_symlinks = false

[index.include]
patterns = ["src/**", "tests/**"]

[index.exclude]
patterns = ["target/**", "node_modules/**", "*.generated.*"]

[search]
default_limit = 100
fuzzy_threshold = 0.8

[performance]
parallel_threads = 0  # 0 = auto-detect
compression_level = 3  # zstd level
```

**Why TOML?**
- ✅ Human-friendly (comments, clear syntax)
- ✅ Rust ecosystem standard (Cargo.toml)
- ✅ Type-safe deserialization with serde

---

## Memory-Mapping Strategy

### memmap2 Crate

Use `memmap2` for zero-copy file access:

```rust
use memmap2::{Mmap, MmapOptions};

let file = File::open(".reflex/symbols.bin")?;
let mmap = unsafe {
    MmapOptions::new()
        .populate()  // Prefault page tables (reduce later page faults)
        .map(&file)?
};

// Access data without copying
let symbols = parse_symbols(&mmap)?;
```

**Best Practices (from research):**

1. **Use `.populate()`** for large files to prefault page tables
   - Causes read-ahead, reduces page fault latency
   - Trade-off: Slower startup, faster queries

2. **Use `.advise(Advice::Sequential)`** for token reads
   - Hints OS to optimize for sequential access
   - Not needed for random symbol access

3. **Keep mmap alive** for lifetime of query
   - Drop when query completes to free memory

4. **Safety:** `unsafe` is required but safe if:
   - File is not modified while mapped
   - File is not truncated
   - Reflex only writes during indexing, never during queries ✅

---

## Cache Size Estimates

Based on research and benchmarks:

### Symbols.bin

**Per-symbol overhead (rkyv):**
- SymbolKind: 1 byte + 7 padding = 8 bytes
- Name: 8 bytes ptr + len (avg 20 chars)
- Span: 4 × 8 bytes = 32 bytes
- Scope: 8 bytes ptr + len (avg 30 chars, 50% have scope)
- File ID: 4 bytes
- Preview: 8 bytes ptr + len (avg 100 chars)

**Estimate:** ~200 bytes/symbol (includes rkyv overhead)

**Example:** 500k symbols = 100 MB

### Tokens.bin (compressed)

**Uncompressed:** ~50 bytes/token × 5M tokens = 250 MB
**Compressed (zstd):** 250 MB × 0.3 = ~75 MB

### Meta.db

**Estimate:** ~500 bytes/file (metadata + indexes)
**Example:** 100k files = 50 MB

### Hashes.json

**Estimate:** ~100 bytes/file (path + hash)
**Example:** 100k files = 10 MB

### Total Cache Size

For a 100k file, 500k symbol project:
- symbols.bin: 100 MB
- tokens.bin: 75 MB
- meta.db: 50 MB
- hashes.json: 10 MB
- **Total: ~235 MB** (~10% of source code size)

**Meets target:** ✅ <10% of source code size

---

## Schema Versioning & Migration

### Version Header

All binary files include version in header:
- Magic bytes (detect corruption)
- Format version (detect incompatibility)

### Migration Strategy

When format version changes:

1. **Read old format**
2. **Convert to new format**
3. **Write with new version**
4. **Delete old cache** (or keep for rollback)

**Implementation:**

```rust
match version {
    1 => read_v1(data),
    2 => read_v2(data),
    _ => Err(anyhow!("Unsupported cache version: {}", version))
}
```

### Cache Invalidation

Trigger full reindex on:
- Format version mismatch
- Magic bytes corruption
- Reflex version upgrade (major)
- User request (`reflex clear`)

**Simple strategy:** Delete `.reflex/` and rebuild
- Fast enough (target: <1 min for 100k files)
- Avoids complex migration code

---

## Alternative Approaches Considered

### 1. All-SQLite Approach

**Pros:** Single file, ACID, easy queries
**Cons:** No zero-copy reads, slower queries
**Verdict:** ❌ Rejected - can't meet <100ms latency target

### 2. All-Custom Binary

**Pros:** Maximum control, smallest size
**Cons:** Complex, manual indexing, hard to debug
**Verdict:** ❌ Rejected - too much work, hard to maintain

### 3. FlatBuffers / Cap'n Proto

**Pros:** Zero-copy like rkyv, schema evolution
**Cons:** Requires schema definition, less Rusty
**Verdict:** ❌ Rejected - rkyv is more idiomatic in Rust

### 4. Embedded KV Store (RocksDB, sled)

**Pros:** Built-in indexing, compression, transactions
**Cons:** Larger dependencies, no mmap zero-copy
**Verdict:** ❌ Rejected - overkill for append-only workload

---

## Performance Validation Plan

### Benchmarks to Run

1. **Symbol query latency** (target: <10ms)
   - Mmap + lookup in symbol index
   - Should be ~0.1ms mmap + ~0.5ms lookup

2. **Token search latency** (target: <50ms)
   - Decompress tokens.bin
   - Search n-gram index
   - Should be ~10ms decompress + ~20ms search

3. **Full query latency** (target: <100ms)
   - Symbol search + token search + ranking
   - End-to-end user-facing latency

4. **Indexing throughput** (target: >10k files/sec)
   - Parse + serialize symbols
   - Should be limited by Tree-sitter parse speed

### Test Codebases

- **Small:** reflex itself (~10 files, ~2k LOC)
- **Medium:** rust-analyzer (~1k files, ~200k LOC)
- **Large:** Linux kernel (~70k files, ~30M LOC)

---

## Implementation Checklist

- [ ] Add rkyv to Cargo.toml
- [ ] Add rusqlite to Cargo.toml
- [ ] Add memmap2 to Cargo.toml
- [ ] Define Symbol struct with rkyv derives
- [ ] Implement symbols.bin writer
- [ ] Implement symbols.bin reader (mmap)
- [ ] Implement meta.db schema
- [ ] Implement SQLite wrapper (CacheMetadata)
- [ ] Implement tokens.bin writer (with zstd)
- [ ] Implement tokens.bin reader
- [ ] Write unit tests for each format
- [ ] Write integration tests (write → read → verify)
- [ ] Benchmark query latency
- [ ] Document file formats in ARCHITECTURE.md

---

## References

- [rkyv Performance Analysis](https://david.kolo.ski/blog/rkyv-is-faster-than/)
- [Rust Serialization Benchmark](https://github.com/djkoloski/rust_serialization_benchmark)
- [memmap2 Documentation](https://docs.rs/memmap2)
- [rusqlite Documentation](https://docs.rs/rusqlite)
- [SQLite Performance Tuning](https://www.sqlite.org/optoverview.html)
- [zstd Compression Benchmark](https://facebook.github.io/zstd/)

---

**END OF BINARY_FORMAT_RESEARCH.md**
