# Reflex Architecture

**Technical deep-dive into Reflex's design and implementation**

This document provides a comprehensive overview of Reflex's architecture, data formats, algorithms, and extension points for developers contributing to the project.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Data Formats](#data-formats)
4. [Indexing Pipeline](#indexing-pipeline)
5. [Query Pipeline](#query-pipeline)
6. [Runtime Symbol Detection](#runtime-symbol-detection)
7. [Performance Optimizations](#performance-optimizations)
8. [Adding New Languages](#adding-new-languages)
9. [Testing Strategy](#testing-strategy)
10. [Future Architecture](#future-architecture)

---

## System Overview

Reflex is a **trigram-based full-text code search engine** with optional symbol-aware filtering. The architecture prioritizes:

1. **Speed**: Instant queries via trigram indexing + memory-mapped I/O
2. **Completeness**: Find every occurrence (not just definitions)
3. **Simplicity**: No daemon required, per-request invocation
4. **Determinism**: Same query → same results (sorted by file:line)

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Reflex CLI                              │
│  (index | query | stats | clear | list-files | serve)          │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼─────┐          ┌─────▼────┐
    │ Indexer  │          │  Query   │
    │          │          │  Engine  │
    └────┬─────┘          └─────┬────┘
         │                      │
         │                      │
    ┌────▼──────────────────────▼────┐
    │      Cache Manager              │
    │  (.reflex/ directory)           │
    └────┬────────────────────────┬───┘
         │                        │
    ┌────▼────┐              ┌────▼────┐
    │Trigram  │              │Content  │
    │Index    │              │Store    │
    │(mmap)   │              │(mmap)   │
    └─────────┘              └─────────┘
```

---

## Core Components

### 1. Cache Manager (`src/cache.rs`)

**Responsibility**: Manage the `.reflex/` cache directory and metadata.

**Files Managed:**
- `meta.db` - SQLite database for file metadata and statistics
- `trigrams.bin` - Memory-mapped trigram inverted index
- `content.bin` - Memory-mapped full file contents
- `hashes.json` - File hashes for incremental indexing (blake3)
- `config.toml` - User configuration

**Key Operations:**
```rust
pub struct CacheManager {
    cache_path: PathBuf,
    db: Connection,  // SQLite connection
}

impl CacheManager {
    pub fn init(&self) -> Result<()>           // Create cache files
    pub fn clear(&self) -> Result<()>          // Delete cache
    pub fn stats(&self) -> Result<IndexStats>  // Get statistics
    pub fn load_hashes(&self) -> Result<HashMap<PathBuf, String>>
    pub fn save_hashes(&self, hashes: HashMap<PathBuf, String>) -> Result<()>
}
```

**SQLite Schema:**
```sql
-- File metadata
CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    hash TEXT NOT NULL,
    language TEXT,
    size_bytes INTEGER,
    indexed_at INTEGER
);

-- Statistics
CREATE TABLE statistics (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Configuration
CREATE TABLE config (
    key TEXT PRIMARY KEY,
    value TEXT
);
```

### 2. Indexer (`src/indexer.rs`)

**Responsibility**: Build the search index from source files.

**Process:**
1. Walk directory tree (respecting `.gitignore`)
2. Filter files by language and size
3. Compute blake3 hash for each file
4. Skip unchanged files (incremental indexing)
5. Extract trigrams from all files
6. Build inverted index: `trigram → [(file_id, line_no)]`
7. Write `trigrams.bin` and `content.bin`
8. Update `meta.db` and `hashes.json`

**Key Types:**
```rust
pub struct Indexer {
    cache: CacheManager,
    config: IndexConfig,
}

pub struct IndexConfig {
    pub max_file_size_mb: u64,
    pub follow_symlinks: bool,
    pub languages: Vec<Language>,
}

pub struct IndexStats {
    pub total_files: usize,
    pub total_bytes: u64,
    pub duration_ms: u64,
    pub files_changed: usize,
    pub files_unchanged: usize,
}
```

**Incremental Indexing:**
```rust
// Compute hash
let hash = blake3::hash(&content).to_hex().to_string();

// Check if changed
let old_hashes = cache.load_hashes()?;
if old_hashes.get(&path) == Some(&hash) {
    // File unchanged, skip indexing
    continue;
}

// File changed or new, index it
index_file(path, content)?;
new_hashes.insert(path, hash);
```

### 3. Trigram Index (`src/trigram.rs`)

**Responsibility**: Extract trigrams and build inverted index.

**Algorithm:**
```rust
// Extract trigrams from text
pub fn extract_trigrams(text: &str) -> HashSet<[u8; 3]> {
    let bytes = text.as_bytes();
    let mut trigrams = HashSet::new();

    for window in bytes.windows(3) {
        if let [a, b, c] = window {
            trigrams.insert([*a, *b, *c]);
        }
    }

    trigrams
}

// Inverted index structure
pub struct TrigramIndex {
    // trigram -> list of (file_id, line_no) postings
    pub index: HashMap<[u8; 3], Vec<(u32, u32)>>,
    pub files: Vec<String>,  // file_id -> file_path
}
```

**Posting List Intersection:**
```rust
// Find files matching ALL trigrams (AND operation)
pub fn search(&self, trigrams: &[[u8; 3]]) -> Vec<u32> {
    if trigrams.is_empty() {
        return vec![];
    }

    // Start with files from first trigram
    let mut candidates = self.index.get(&trigrams[0])
        .map(|postings| postings.iter().map(|(file_id, _)| *file_id).collect())
        .unwrap_or_default();

    // Intersect with files from remaining trigrams
    for trigram in &trigrams[1..] {
        let files: HashSet<u32> = self.index.get(trigram)
            .map(|postings| postings.iter().map(|(file_id, _)| *file_id).collect())
            .unwrap_or_default();

        candidates.retain(|file_id| files.contains(file_id));
    }

    candidates
}
```

### 4. Content Store (`src/content_store.rs`)

**Responsibility**: Store and retrieve full file contents.

**Binary Format:**
```
┌────────────────────────────────────────┐
│ Header (32 bytes)                      │
│  - Magic: "RFCT" (4 bytes)             │
│  - Version: u32                        │
│  - Num Files: u32                      │
│  - Index Offset: u64                   │
│  - Reserved: 12 bytes                  │
├────────────────────────────────────────┤
│ File Contents (variable)               │
│  - Concatenated file contents          │
├────────────────────────────────────────┤
│ File Index (at Index Offset)           │
│  For each file:                        │
│    - Path length: u32                  │
│    - Path: UTF-8 string                │
│    - Content offset: u64               │
│    - Content length: u64               │
└────────────────────────────────────────┘
```

**Memory-Mapped Access:**
```rust
pub struct ContentReader {
    mmap: Mmap,  // Memory-mapped file
    files: Vec<FileEntry>,
}

pub struct FileEntry {
    pub path: String,
    pub offset: u64,
    pub length: u64,
}

impl ContentReader {
    pub fn get_content(&self, file_id: usize) -> Result<&str> {
        let entry = &self.files[file_id];
        let start = entry.offset as usize;
        let end = start + entry.length as usize;

        std::str::from_utf8(&self.mmap[start..end])
    }
}
```

### 5. Query Engine (`src/query.rs`)

**Responsibility**: Execute search queries and return results.

**Query Modes:**
1. **Full-text search**: Find all occurrences of pattern in any file
2. **Symbol search**: Find symbol definitions only (runtime parsing)
3. **Regex search**: Pattern matching with trigram optimization

**Query Pipeline:**
```rust
pub struct QueryEngine {
    cache: CacheManager,
    trigram_index: TrigramIndex,
    content_reader: ContentReader,
}

pub struct QueryFilter {
    pub symbols_mode: bool,           // Symbol-only search?
    pub use_regex: bool,              // Regex pattern?
    pub exact_match: bool,            // Exact string match?
    pub language: Option<Language>,   // Filter by language
    pub kind: Option<SymbolKind>,     // Filter by symbol kind
    pub file_pattern: Option<String>, // Filter by file path
    pub limit: Option<usize>,         // Limit results
}

impl QueryEngine {
    pub fn search(&self, pattern: &str, filter: QueryFilter) -> Result<Vec<SearchResult>> {
        if filter.use_regex {
            self.search_with_regex(pattern, filter)
        } else if filter.symbols_mode {
            self.search_symbols(pattern, filter)
        } else {
            self.search_fulltext(pattern, filter)
        }
    }
}
```

### 6. Parser Factory (`src/parsers/mod.rs`)

**Responsibility**: Select appropriate tree-sitter parser for each language.

**Supported Languages:**
```rust
pub enum Language {
    Rust,
    TypeScript,
    JavaScript,
    Vue,
    Svelte,
    PHP,
    Python,
    Go,
    Java,
    C,
    Cpp,
}

impl Language {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "rs" => Some(Language::Rust),
            "ts" | "tsx" | "mts" | "cts" => Some(Language::TypeScript),
            "js" | "jsx" | "mjs" | "cjs" => Some(Language::JavaScript),
            "vue" => Some(Language::Vue),
            "svelte" => Some(Language::Svelte),
            "php" => Some(Language::PHP),
            "py" => Some(Language::Python),
            "go" => Some(Language::Go),
            "java" => Some(Language::Java),
            "c" | "h" => Some(Language::C),
            "cpp" | "hpp" | "cxx" | "cc" => Some(Language::Cpp),
            _ => None,
        }
    }
}
```

---

## Data Formats

### trigrams.bin Format

**Purpose**: Persistent trigram inverted index (rkyv serialization)

**Structure:**
```rust
#[derive(Archive, Serialize, Deserialize)]
pub struct TrigramIndexArchive {
    pub index: HashMap<[u8; 3], Vec<(u32, u32)>>,
    pub files: Vec<String>,
}
```

**Serialization (rkyv):**
- Zero-copy deserialization via memory-mapping
- Direct access to archived data without parsing
- ~10x faster than serde deserialization

**File Layout:**
```
┌────────────────────────────────────────┐
│ Header                                 │
│  - Magic: "RFTG" (4 bytes)             │
│  - Version: u32                        │
│  - Num Trigrams: u32                   │
│  - Num Files: u32                      │
│  - Files Offset: u64                   │
├────────────────────────────────────────┤
│ Trigram Postings (rkyv)                │
│  - HashMap<[u8;3], Vec<(u32,u32)>>     │
├────────────────────────────────────────┤
│ File List (rkyv, at Files Offset)      │
│  - Vec<String>                         │
└────────────────────────────────────────┘
```

### hashes.json Format

**Purpose**: Track file hashes for incremental indexing

**Format:**
```json
{
  "src/main.rs": "af1349b9f5f9a1a6110b...",
  "src/lib.rs": "7e2c14b2a4f3c5d8901a...",
  "README.md": "9d3f5b8c1e7a2d4f6b8c..."
}
```

**Hash Algorithm**: blake3 (faster than SHA-256, cryptographically secure)

---

## Indexing Pipeline

### Step-by-Step Process

```
1. Walk Directory Tree
   ├─ Use `ignore` crate
   ├─ Respect .gitignore
   └─ Filter by file extension

2. For Each File:
   ├─ Compute blake3 hash
   ├─ Compare with hashes.json
   └─ Skip if unchanged

3. Extract Trigrams:
   ├─ Read file content
   ├─ Extract all 3-char substrings
   └─ Build trigram set

4. Build Inverted Index:
   ├─ For each trigram:
   │   └─ Append (file_id, line_no) to posting list
   └─ Deduplicate postings

5. Write Cache Files:
   ├─ Serialize trigram index (rkyv)
   ├─ Write content.bin (binary)
   ├─ Update meta.db (SQLite)
   └─ Save hashes.json
```

### Incremental Indexing

**Problem**: Full reindex is expensive on large codebases.

**Solution**: Hash-based change detection
```rust
// Load previous hashes
let old_hashes = cache.load_hashes()?;

for path in files {
    let content = fs::read_to_string(&path)?;
    let hash = blake3::hash(content.as_bytes()).to_hex().to_string();

    // Skip if hash unchanged
    if old_hashes.get(&path) == Some(&hash) {
        unchanged_count += 1;
        continue;
    }

    // File changed or new, index it
    index_file(&path, &content)?;
    new_hashes.insert(path, hash);
}

// Save new hashes
cache.save_hashes(new_hashes)?;
```

**Performance**:
- Full index (1000 files): ~2s
- Incremental (10/1000 changed): ~200ms (10x faster)

---

## Query Pipeline

### Full-Text Search

```
1. Extract Trigrams from Pattern
   └─ "hello" → ["hel", "ell", "llo"]

2. Intersect Posting Lists
   ├─ Get files matching ALL trigrams (AND)
   └─ Result: ~10-100 candidate files

3. Verify Matches
   ├─ Load file content from content.bin
   ├─ Search for exact pattern (substring)
   └─ Extract context (before/after lines)

4. Apply Filters
   ├─ Language filter
   ├─ File path filter
   └─ Limit results

5. Sort Results
   ├─ By file path (lexicographic)
   └─ By line number (ascending)
```

**Time Complexity**:
- Trigram extraction: O(n) where n = pattern length
- Posting list intersection: O(k log k) where k = # of trigrams
- Content verification: O(m) where m = # of candidate files
- Total: **O(n + k log k + m)** ≈ **O(m)** for most queries

### Symbol Search (Runtime Parsing)

```
1. Extract Trigrams from Pattern
   └─ "parse" → ["par", "ars", "rse"]

2. Narrow to Candidate Files
   └─ Trigram search → ~10-100 files

3. Parse Candidates with Tree-Sitter
   ├─ For each candidate file:
   │   ├─ Select parser by language
   │   ├─ Parse AST (tree-sitter)
   │   └─ Extract symbols (functions, classes, etc.)
   └─ Parse time: ~2-5ms per file

4. Filter Symbols
   ├─ Name match (exact/prefix/substring)
   ├─ Symbol kind (function, class, etc.)
   └─ Scope context

5. Return Symbol Definitions
   └─ Not call sites (symbols only)
```

**Why Runtime Parsing?**
- **Old approach**: Index all symbols during build (4125ms to load 3.3M symbols)
- **New approach**: Parse only ~10-100 candidate files at query time (~2-224ms)
- **Result**: 2000x faster on small codebases, 18x faster on Linux kernel

### Regex Search

```
1. Extract Literals from Regex
   ├─ "fn test_\w+" → "fn test_" (literal part)
   └─ Must be ≥3 chars for trigram extraction

2. Extract Trigrams from Literals
   └─ "fn test_" → ["fn ", "n t", " te", "tes", "est", "st_"]

3. Narrow to Candidates
   └─ Union (OR) of trigram matches

4. Apply Regex to Candidates
   ├─ Compile regex pattern
   ├─ Match against file content
   └─ Extract matching lines

5. Fallback to Full Scan
   └─ If no literals ≥3 chars (e.g., "a+", "\d")
```

**Regex Optimization Details** (`src/regex_trigrams.rs`):
- Literal extraction handles: alternation `(a|b)`, quantifiers `a+`, groups `(abc)`
- Case-insensitive flag `(?i)` triggers full scan
- Escape sequences handled: `\n`, `\t`, `\.`, `\w`, `\d`

---

## Runtime Symbol Detection

**Key Innovation**: Parse symbols at query time, not index time.

### Architecture Comparison

| Approach | Index Time | Query Time | Memory | Flexibility |
|----------|------------|------------|--------|-------------|
| **Indexed Symbols** | Slow (parse all files) | 4125ms (load 3.3M symbols) | High | Low (reindex to add symbols) |
| **Runtime Parsing** | Fast (trigrams only) | 2-224ms (parse ~10 files) | Low | High (no reindex needed) |

### Implementation

**During Indexing:**
```rust
// NO tree-sitter parsing during indexing
pub fn index_file(path: &Path, content: &str) -> Result<()> {
    // Extract trigrams only
    let trigrams = extract_trigrams(content);

    // Build inverted index
    for trigram in trigrams {
        index.insert(trigram, (file_id, line_no));
    }

    // NO symbol extraction
}
```

**During Symbol Query:**
```rust
pub fn search_symbols(&self, pattern: &str, filter: QueryFilter) -> Result<Vec<Symbol>> {
    // 1. Narrow to candidates with trigrams
    let candidates = self.trigram_search(pattern)?;  // ~10-100 files

    // 2. Parse only candidate files
    let mut symbols = Vec::new();
    for file_id in candidates {
        let content = self.content_reader.get_content(file_id)?;
        let language = Language::from_path(&file_path)?;

        // Parse with tree-sitter (2-5ms per file)
        let parsed_symbols = parse_symbols(content, language)?;

        // Filter symbols by name/kind
        let matches = parsed_symbols.iter()
            .filter(|s| s.name.contains(pattern))
            .filter(|s| filter.kind.map_or(true, |k| s.kind == k))
            .cloned()
            .collect::<Vec<_>>();

        symbols.extend(matches);
    }

    Ok(symbols)
}
```

**Performance Benefits:**
- **Lazy evaluation**: Only parse files that match trigrams
- **Scalability**: Parse 10 files vs load 3.3M symbols
- **Flexibility**: Add new symbol types without reindexing

---

## Performance Optimizations

### 1. Memory-Mapped I/O

**Problem**: Deserializing large indices is slow.

**Solution**: Memory-map cache files for zero-copy access.

```rust
use memmap2::Mmap;

pub struct TrigramIndex {
    mmap: Mmap,
    index: &'static TrigramIndexArchive,  // Points into mmap
}

impl TrigramIndex {
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Zero-copy deserialization (rkyv)
        let index = unsafe { archived_root::<TrigramIndexArchive>(&mmap) };

        Ok(Self { mmap, index })
    }
}
```

**Benefits:**
- **Instant loading**: No deserialization overhead
- **Memory efficiency**: OS handles paging
- **Shared memory**: Multiple processes share read-only cache

### 2. rkyv Serialization

**Why rkyv over serde?**

| Feature | serde | rkyv |
|---------|-------|------|
| **Deserialization** | Parse entire file | Zero-copy (mmap) |
| **Load time** | 100-500ms | <1ms |
| **Memory usage** | High (copy in memory) | Low (OS manages) |
| **Random access** | No (deserialize all) | Yes (direct access) |

### 3. blake3 Hashing

**Why blake3 over SHA-256?**

| Hash | Speed | Security |
|------|-------|----------|
| **blake3** | ~2 GB/s | Cryptographically secure |
| **SHA-256** | ~300 MB/s | Cryptographically secure |
| **xxHash** | ~10 GB/s | Not cryptographically secure |

**Choice**: blake3 is fast enough for incremental indexing and collision-resistant.

### 4. Trigram Indexing

**Why trigrams?**
- **Coverage**: 3 chars is minimal for useful filtering
- **Performance**: ~100-1000x reduction in search space
- **Flexibility**: Works for any text pattern

**Alternative Approaches:**
- **Suffix arrays**: Slower build, larger memory
- **N-grams (n>3)**: Less coverage, more storage
- **Tokens**: Language-specific, incomplete coverage

---

## Adding New Languages

Reflex makes it easy to add support for new languages.

### Step 1: Add Tree-Sitter Grammar

```toml
# Cargo.toml
[dependencies]
tree-sitter-ruby = "0.23"
```

### Step 2: Extend Language Enum

```rust
// src/parsers/mod.rs
pub enum Language {
    // ... existing languages
    Ruby,
}

impl Language {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            // ... existing extensions
            "rb" => Some(Language::Ruby),
            _ => None,
        }
    }
}
```

### Step 3: Implement Parser

```rust
// src/parsers/ruby.rs
use tree_sitter::{Parser, Node};
use tree_sitter_ruby::language;

pub fn extract_symbols(source: &str) -> Vec<Symbol> {
    let mut parser = Parser::new();
    parser.set_language(language()).unwrap();

    let tree = parser.parse(source, None).unwrap();
    let root = tree.root_node();

    let mut symbols = Vec::new();
    symbols.extend(extract_methods(source, root));
    symbols.extend(extract_classes(source, root));
    // ... other symbol types

    symbols
}

fn extract_methods(source: &str, node: Node) -> Vec<Symbol> {
    let query = r#"
        (method
            name: (identifier) @name) @method
    "#;

    // Use tree-sitter query API
    // Extract method name, span, scope
    // Return Symbol structs
}
```

### Step 4: Add Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_ruby_methods() {
        let source = r#"
            class User
              def greet(name)
                puts "Hello, #{name}"
              end
            end
        "#;

        let symbols = extract_symbols(source);
        assert_eq!(symbols.len(), 2);  // User class + greet method
        assert_eq!(symbols[0].name, "User");
        assert_eq!(symbols[1].name, "greet");
    }
}
```

### Step 5: Update Documentation

```markdown
## Supported Languages

| Language | Extensions | Symbol Extraction |
|----------|------------|-------------------|
| **Ruby** | `.rb` | Classes, modules, methods, constants |
```

---

## Testing Strategy

Reflex has **221 comprehensive tests** across 3 categories:

### 1. Unit Tests (194 tests)

**Location**: Embedded in source files (`#[cfg(test)]` modules)

**Coverage:**
- `src/cache.rs` (29 tests): Init, persistence, stats, clearing
- `src/indexer.rs` (24 tests): Filtering, hashing, incremental updates
- `src/query.rs` (22 tests): Pattern parsing, filtering, ranking
- `src/parsers/*.rs` (85 tests): Symbol extraction for all languages
- `src/trigram.rs` (8 tests): Trigram extraction, intersection
- `src/content_store.rs` (4 tests): Binary format, memory-mapping
- `src/regex_trigrams.rs` (22 tests): Literal extraction, optimization

**Example:**
```rust
#[test]
fn test_incremental_indexing() {
    let temp = TempDir::new().unwrap();

    // Initial index
    fs::write(temp.path().join("main.rs"), "fn test() {}").unwrap();
    let cache = CacheManager::new(temp.path());
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(temp.path(), false).unwrap();

    // Modify file
    fs::write(temp.path().join("main.rs"), "fn test2() {}").unwrap();

    // Reindex (should detect change)
    let cache = CacheManager::new(temp.path());
    let indexer = Indexer::new(cache, IndexConfig::default());
    let stats = indexer.index(temp.path(), false).unwrap();

    assert_eq!(stats.files_changed, 1);
}
```

### 2. Integration Tests (17 tests)

**Location**: `tests/integration_test.rs`

**Scenarios:**
- Full workflow (index → query → verify)
- Multi-language indexing and search
- Incremental indexing correctness
- Error handling (missing index, empty directory)
- Cache persistence across sessions

**Example:**
```rust
#[test]
fn test_multi_language_search() {
    let temp = TempDir::new().unwrap();

    // Create files in multiple languages
    fs::write(temp.path().join("main.rs"), "fn greet() {}").unwrap();
    fs::write(temp.path().join("app.ts"), "function greet() {}").unwrap();
    fs::write(temp.path().join("script.py"), "def greet(): pass").unwrap();

    // Index
    let cache = CacheManager::new(temp.path());
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(temp.path(), false).unwrap();

    // Search across all languages
    let cache = CacheManager::new(temp.path());
    let engine = QueryEngine::new(cache);
    let results = engine.search("greet", QueryFilter::default()).unwrap();

    assert_eq!(results.len(), 3);  // Found in all 3 files
}
```

### 3. Performance Tests (10 tests)

**Location**: `tests/performance_test.rs`

**Benchmarks:**
- Indexing speed (100, 500, 1000 files)
- Query latency (full-text, symbol, regex)
- Incremental reindex efficiency
- Memory-mapped I/O performance
- Scalability (large files, many files)

**Example:**
```rust
#[test]
fn test_query_performance() {
    let temp = TempDir::new().unwrap();

    // Create 200 files
    for i in 0..200 {
        let content = format!("fn function_{}() {{\n    println!(\"hello\");\n}}", i);
        fs::write(temp.path().join(format!("file_{}.rs", i)), content).unwrap();
    }

    // Index
    let cache = CacheManager::new(temp.path());
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(temp.path(), false).unwrap();

    // Measure query time
    let cache = CacheManager::new(temp.path());
    let engine = QueryEngine::new(cache);

    let start = Instant::now();
    let results = engine.search("hello", QueryFilter::default()).unwrap();
    let duration = start.elapsed();

    assert!(results.len() >= 200);
    assert!(duration.as_millis() < 100, "Query took {}ms", duration.as_millis());
}
```

**Performance Targets:**
- Indexing: 100 files in <1s, 500 files in <3s
- Full-text query: <100ms on 200+ files
- Symbol query: <5s with runtime parsing
- Regex query: <200ms with trigram optimization

---

## Future Architecture

### Planned Features

#### 1. HTTP Server

**Use case**: Editor plugins, AI agents, CI/CD tools

**Architecture:**
```
┌─────────────┐
│ HTTP Server │  (axum framework)
│   :7878     │
└──────┬──────┘
       │
   ┌───┴────┐
   │ Router │
   └───┬────┘
       │
   ┌───┴────────────────┐
   │  Query Engine      │  (shared instance)
   │  (long-lived)      │
   └────────────────────┘
```

**Endpoints:**
- `GET /query?q=pattern&lang=rust&limit=10`
- `GET /stats`
- `POST /index`

**Benefits:**
- Eliminate process spawn overhead
- Keep cache in memory (warm queries)
- Long-lived connections for streaming results

#### 2. AST Pattern Matching

**Use case**: Structure-aware code search (find patterns in AST, not just text)

**Example Queries:**
- "Find all functions taking String and returning Result"
- "Find all classes with @deprecated annotation"
- "Find all TODO comments in function bodies"

**Implementation:**
```rust
// Tree-sitter S-expression pattern
let pattern = r#"
    (function_item
        parameters: (parameters (parameter type: (type_identifier) @param_type))
        return_type: (type_identifier) @return_type
        (#eq? @param_type "String")
        (#eq? @return_type "Result"))
"#;

// Execute pattern against candidate files
let matches = ast_search(pattern)?;
```

#### 3. MCP Adapter

**Use case**: Integrate with Claude and other AI assistants via Model Context Protocol

**Architecture:**
```
┌─────────────┐
│ Claude API  │
└──────┬──────┘
       │ (MCP)
┌──────┴──────┐
│ MCP Server  │
└──────┬──────┘
       │
┌──────┴──────┐
│ Query Engine│
└─────────────┘
```

**MCP Tools:**
- `reflex_search(pattern, filters)` - Search codebase
- `reflex_get_symbol(name, kind)` - Get symbol definition
- `reflex_list_files(language)` - List indexed files

---

## Design Principles

Reflex follows these core principles:

1. **Performance First**
   - Instant queries via trigram indexing
   - Memory-mapped I/O for zero-copy access
   - Lazy evaluation (runtime symbol detection)

2. **Completeness Over Precision**
   - Find every occurrence (100% recall)
   - Accept some false positives (trigram candidates)
   - Verify matches in actual content

3. **Simplicity Over Features**
   - No daemon required
   - Single binary, per-request invocation
   - Clean separation of concerns

4. **Determinism**
   - Same query → same results
   - Sorted by file path, then line number
   - No probabilistic ranking

5. **Extensibility**
   - Easy to add new languages (tree-sitter)
   - Clean interfaces between components
   - Comprehensive test coverage

---

## Contributing

When adding features to Reflex:

1. **Maintain Performance**: Keep queries <100ms on medium codebases
2. **Add Tests**: Unit + integration tests for all new code
3. **Document Design**: Update this file for architectural changes
4. **Preserve Determinism**: No randomness in results
5. **Keep It Simple**: Avoid complex dependencies

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## References

**Research Papers:**
- [Trigram Indexing for Code Search](https://swtch.com/~rsc/regexp/regexp4.html) (Russ Cox)
- [Google Code Search](https://swtch.com/~rsc/regexp/regexp4.html)

**Similar Projects:**
- [Zoekt](https://github.com/sourcegraph/zoekt) - Trigram-based code search (Go)
- [Sourcegraph](https://sourcegraph.com/) - Code search for teams
- [ripgrep](https://github.com/BurntSushi/ripgrep) - Fast text search

**Tools Used:**
- [tree-sitter](https://tree-sitter.github.io/) - Incremental parsing
- [rkyv](https://rkyv.org/) - Zero-copy serialization
- [memmap2](https://github.com/RazrFalcon/memmap2-rs) - Memory-mapped I/O

---

**Last Updated**: 2025-11-03
**Author**: Reflex Contributors
