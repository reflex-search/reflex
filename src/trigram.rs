//! Trigram-based inverted index for fast full-text code search
//!
//! This module implements the core trigram indexing algorithm used by Reflex.
//! A trigram is a sequence of 3 consecutive bytes. By building an inverted index
//! mapping trigrams to file locations, we can quickly narrow down search candidates
//! and achieve sub-100ms query times even on large codebases.
//!
//! # Algorithm
//!
//! 1. **Indexing**: Extract all trigrams from each file, store locations
//! 2. **Querying**: Extract trigrams from query, intersect posting lists
//! 3. **Verification**: Check actual matches at candidate locations
//!
//! See `.context/TRIGRAM_RESEARCH.md` for detailed algorithm documentation.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// A trigram is 3 consecutive bytes, packed into a u32 for efficient hashing
pub type Trigram = u32;

// Binary format constants for trigrams.bin
const MAGIC: &[u8; 4] = b"RFTG"; // ReFlex TriGrams
/// V4: per-line postings grouped into per-file blocks, no byte offsets,
/// `paths_offset` in the header so `load` never walks the directory.
const VERSION: u32 = 4;
/// Header: magic(4) + version(4) + num_trigrams(8) + num_files(8) + paths_offset(8) = 32 bytes
const HEADER_SIZE: usize = 32;
/// Directory entry: trigram(4) + data_offset(8) + compressed_size(4)
const DIR_ENTRY_SIZE: usize = 16;
/// Header field offsets (bytes 8..16 are also read by `cli/misc.rs`; keep them stable)
const NUM_TRIGRAMS_OFFSET: usize = 8;
const NUM_FILES_OFFSET: usize = 16;
const PATHS_OFFSET_OFFSET: usize = 24;

/// Write a u32 as a varint (variable-length integer)
/// Uses 1-5 bytes depending on magnitude (smaller numbers = fewer bytes)
fn write_varint(writer: &mut impl Write, mut value: u32) -> std::io::Result<()> {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80; // Set continuation bit
        }
        writer.write_all(&[byte])?;
        if value == 0 {
            break;
        }
    }
    Ok(())
}

/// Read a varint from a byte slice, returns (value, bytes_consumed)
fn read_varint(data: &[u8]) -> Result<(u32, usize)> {
    let mut value: u32 = 0;
    let mut shift = 0;
    let mut pos = 0;

    loop {
        if pos >= data.len() {
            anyhow::bail!("Truncated varint");
        }
        let byte = data[pos];
        pos += 1;

        value |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 32 {
            anyhow::bail!("Varint too large");
        }
    }

    Ok((value, pos))
}

/// Skip one varint without decoding it, returning the bytes consumed
#[inline]
fn skip_varint(data: &[u8]) -> Result<usize> {
    let mut pos = 0;
    loop {
        if pos >= data.len() {
            anyhow::bail!("Truncated varint");
        }
        let byte = data[pos];
        pos += 1;
        if byte & 0x80 == 0 {
            return Ok(pos);
        }
        if pos >= 5 {
            anyhow::bail!("Varint too large");
        }
    }
}

/// Read a little-endian u64 at `off` (caller guarantees bounds)
#[inline]
fn read_u64(data: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(data[off..off + 8].try_into().expect("8 bytes"))
}

/// Read a little-endian u32 at `off` (caller guarantees bounds)
#[inline]
fn read_u32(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(data[off..off + 4].try_into().expect("4 bytes"))
}

/// Encode a sorted, key-unique posting list into V4 file blocks.
///
/// Layout, per distinct `file_id` in `locations`:
///
/// ```text
/// varint(file_id - prev_file_id)   // first block: delta from 0
/// varint(n_lines << 1 | enc)       // enc bit reserved; always 0 here
/// n_lines × varint(line - prev_line)   // prev_line restarts at 0 per block
/// ```
///
/// Line deltas never cross a file boundary, so every block starts with two
/// small varints instead of the ~5-byte wrapped deltas V3 paid per file.
fn encode_posting_list(locations: &[FileLocation], out: &mut Vec<u8>) -> Result<()> {
    let mut prev_file_id = 0u32;
    let mut i = 0;
    while i < locations.len() {
        let file_id = locations[i].file_id;
        let mut j = i + 1;
        while j < locations.len() && locations[j].file_id == file_id {
            j += 1;
        }
        let block = &locations[i..j];

        write_varint(out, file_id.wrapping_sub(prev_file_id))?;
        write_varint(out, (block.len() as u32) << 1)?; // enc = 0

        let mut prev_line = 0u32;
        for loc in block {
            write_varint(out, loc.line_no.wrapping_sub(prev_line))?;
            prev_line = loc.line_no;
        }

        prev_file_id = file_id;
        i = j;
    }
    Ok(())
}

/// Decompress a posting list from memory-mapped data
///
/// Reads a compressed posting list (V4 file blocks) from the given offset
/// and decompresses it into a Vec<FileLocation>.
///
/// # Arguments
/// * `mmap` - Memory-mapped file data
/// * `offset` - Absolute byte offset where compressed data starts
/// * `size` - Number of bytes to read
fn decompress_posting_list(mmap: &[u8], offset: u64, size: u32) -> Result<Vec<FileLocation>> {
    let start = offset as usize;
    let end = start + size as usize;

    if end > mmap.len() {
        anyhow::bail!(
            "Posting list out of bounds: offset={}, size={}, mmap_len={}",
            offset,
            size,
            mmap.len()
        );
    }

    let compressed_data = &mmap[start..end];

    // Rough capacity guess: a typical line entry is one 1-byte varint plus
    // amortised block headers.
    let mut locations = Vec::with_capacity(compressed_data.len());
    let mut cursor = PostingCursor::new(compressed_data)?;
    while let Some(loc) = cursor.current() {
        locations.push(loc);
        cursor.advance()?;
    }

    Ok(locations)
}

/// Streaming decoder over a V4 posting list (see [`encode_posting_list`]).
///
/// Decodes one `FileLocation` per line at a time so a large posting list can
/// be intersected against a small candidate set without materialising it.
/// Block headers are consumed transparently by [`advance`](Self::advance);
/// [`seek`](Self::seek) skips the remainder of a block without decoding line
/// values when the target lies in a later file.
pub(crate) struct PostingCursor<'a> {
    /// Compressed posting list bytes
    data: &'a [u8],
    /// Read position in `data`
    pos: usize,
    /// File id of the block being decoded
    file_id: u32,
    /// Lines left to decode in the current block
    remaining: u32,
    /// Last decoded line in the current block (delta base)
    prev_line: u32,
    /// Current location, `None` once exhausted
    cur: Option<FileLocation>,
}

impl<'a> PostingCursor<'a> {
    /// Create a cursor and decode the first entry (if any)
    pub(crate) fn new(data: &'a [u8]) -> Result<Self> {
        let mut cursor = Self {
            data,
            pos: 0,
            file_id: 0,
            remaining: 0,
            prev_line: 0,
            cur: None,
        };
        cursor.advance()?;
        Ok(cursor)
    }

    /// The entry the cursor is positioned on, or `None` if exhausted
    #[inline]
    pub(crate) fn current(&self) -> Option<FileLocation> {
        self.cur
    }

    /// Consume a block header: `varint(file delta) varint(n_lines << 1 | enc)`
    fn read_block_header(&mut self) -> Result<()> {
        let (file_delta, consumed) = read_varint(&self.data[self.pos..])?;
        self.pos += consumed;
        let (header, consumed) = read_varint(&self.data[self.pos..])?;
        self.pos += consumed;

        if header & 1 != 0 {
            anyhow::bail!("unsupported block encoding (enc=1) in posting list");
        }

        self.file_id = self.file_id.wrapping_add(file_delta);
        self.remaining = header >> 1;
        self.prev_line = 0;
        Ok(())
    }

    /// Decode the next entry, returning it (or `None` at end of list)
    pub(crate) fn advance(&mut self) -> Result<Option<FileLocation>> {
        // A block with n_lines == 0 is never written, but tolerate it
        while self.remaining == 0 {
            if self.pos >= self.data.len() {
                self.cur = None;
                return Ok(None);
            }
            self.read_block_header()?;
        }

        let (line_delta, consumed) = read_varint(&self.data[self.pos..])?;
        self.pos += consumed;
        self.remaining -= 1;

        let line_no = self.prev_line.wrapping_add(line_delta);
        self.prev_line = line_no;
        let loc = FileLocation::new(self.file_id, line_no);
        self.cur = Some(loc);
        Ok(self.cur)
    }

    /// Advance until the current key is `>= target` (or the list is exhausted)
    ///
    /// When the target file id is beyond the current block, the rest of the
    /// block is skipped by scanning varint continuation bits only.
    pub(crate) fn seek(&mut self, target: (u32, u32)) -> Result<Option<FileLocation>> {
        while let Some(loc) = self.cur {
            if key(&loc) >= target {
                break;
            }
            if loc.file_id < target.0 {
                for _ in 0..self.remaining {
                    self.pos += skip_varint(&self.data[self.pos..])?;
                }
                self.remaining = 0;
            }
            self.advance()?;
        }
        Ok(self.cur)
    }
}

/// Intersect a sorted, key-deduplicated candidate set with a streamed posting list.
///
/// For each candidate, seeks the cursor to its key and keeps the candidate when
/// the keys match. Runs in O(|cands| + |list|) without materialising the list.
fn intersect_with_cursor(
    cands: &[FileLocation],
    cur: &mut PostingCursor,
) -> Result<Vec<FileLocation>> {
    let mut out = Vec::new();
    for cand in cands {
        let k = key(cand);
        match cur.seek(k)? {
            Some(loc) if key(&loc) == k => out.push(*cand),
            Some(_) => {}
            None => break,
        }
    }
    Ok(out)
}

/// Mark every candidate present in a streamed posting list.
///
/// Same walk as [`intersect_with_cursor`], but sets `keep[i]` for each hit
/// instead of building a new vector, so several lists (the case variants of
/// one trigram) can be OR-ed into one mask over the same candidate slice.
fn mark_with_cursor(
    cands: &[FileLocation],
    cur: &mut PostingCursor,
    keep: &mut [bool],
) -> Result<()> {
    for (i, cand) in cands.iter().enumerate() {
        if keep[i] {
            continue;
        }
        let k = key(cand);
        match cur.seek(k)? {
            Some(loc) if key(&loc) == k => keep[i] = true,
            Some(_) => {}
            None => break,
        }
    }
    Ok(())
}

/// Location of a trigram occurrence in the codebase: one entry per
/// (file, line) a trigram appears on. Derived `Ord` is the intersection key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FileLocation {
    /// File ID (index into file list)
    pub file_id: u32,
    /// Line number (1-indexed)
    pub line_no: u32,
}

impl FileLocation {
    pub fn new(file_id: u32, line_no: u32) -> Self {
        Self { file_id, line_no }
    }
}

/// Directory entry for lazy-loaded trigram index
///
/// Maps each trigram to its compressed posting list location in the data section.
/// Total size: 16 bytes per entry (4 + 8 + 4)
/// Compressed posting bytes worth decoding per surviving candidate line.
///
/// Measured on the synthetic 30 MB corpus (16 cores): streaming a V4 list costs
/// ~7 ns per byte on one thread; verifying one extra candidate line costs
/// ~0.14 µs of CPU spread over the query pool, i.e. ~10–15 ns of wall time.
/// Past this ratio the next list is not worth reading — `ident_7` spent 21 ms
/// streaming four ~750 KB lists that removed no candidate.
const SKIP_BYTES_PER_CANDIDATE: usize = 2;

#[derive(Debug, Clone)]
struct DirectoryEntry {
    /// The trigram value (for binary search)
    trigram: Trigram,
    /// Absolute byte offset in the file where compressed data starts
    data_offset: u64,
    /// Size of compressed posting list in bytes
    compressed_size: u32,
}

/// Trigram-based inverted index
///
/// Maps each trigram to a sorted list of locations where it appears.
/// Posting lists are kept sorted by (file_id, line_no) for efficient intersection.
/// The index itself is kept sorted by trigram for O(log n) binary search.
///
/// Supports three modes:
/// 1. **In-memory mode** (during indexing): All posting lists in RAM
/// 2. **Batch-flush mode** (large codebases): Periodically flushes partial indices to disk to limit RAM
/// 3. **Lazy-loaded mode** (after loading): Compressed posting lists in mmap, decompressed on-demand
pub struct TrigramIndex {
    /// Inverted index: sorted Vec of (trigram, locations) for binary search
    /// Used in in-memory mode (during indexing)
    index: Vec<(Trigram, Vec<FileLocation>)>,
    /// File ID to file path mapping
    files: Vec<PathBuf>,
    /// Temporary HashMap used during batch indexing (None when finalized)
    temp_index: Option<HashMap<Trigram, Vec<FileLocation>>>,
    /// Memory-mapped index file (for lazy loading)
    mmap: Option<memmap2::Mmap>,
    /// Number of directory entries in the mmap (lazy mode). The directory is
    /// binary-searched in place; it is never decoded into a Vec.
    num_trigrams: usize,
    /// Partial index files created during batch flushing (for k-way merge at finalize)
    partial_indices: Vec<PathBuf>,
    /// Temporary directory for partial indices
    temp_dir: Option<PathBuf>,
    /// Cap on posting list size; 0 = unlimited. Enforced at finalize time.
    /// Bounds query latency for high-frequency trigrams (trigram-density lens).
    max_posting_list_entries: usize,
}

impl TrigramIndex {
    /// Create a new empty trigram index
    pub fn new() -> Self {
        Self {
            index: Vec::new(),
            files: Vec::new(),
            temp_index: Some(HashMap::new()),
            mmap: None,
            num_trigrams: 0,
            partial_indices: Vec::new(),
            temp_dir: None,
            max_posting_list_entries: 0,
        }
    }

    /// Set maximum posting list entries per trigram (0 = unlimited).
    pub fn set_max_posting_list_entries(&mut self, cap: usize) {
        self.max_posting_list_entries = cap;
    }

    /// Enable batch-flush mode for large codebases
    ///
    /// Creates a temporary directory for partial indices that will be merged at finalize().
    /// Call this before indexing to enable memory-efficient indexing for huge codebases.
    pub fn enable_batch_flush(&mut self, temp_dir: PathBuf) -> Result<()> {
        std::fs::create_dir_all(&temp_dir)
            .context("Failed to create temp directory for batch flushing")?;
        self.temp_dir = Some(temp_dir);
        log::info!("Enabled batch-flush mode for trigram index");
        Ok(())
    }

    /// Flush current temp_index to a partial index file
    ///
    /// This clears the in-memory HashMap and writes a sorted partial index to disk.
    /// Called periodically during indexing to limit memory usage.
    pub fn flush_batch(&mut self) -> Result<()> {
        let temp_dir = self.temp_dir.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Batch flush not enabled - call enable_batch_flush() first")
        })?;

        // Take ownership of temp_index to finalize it
        let temp_map = self
            .temp_index
            .take()
            .ok_or_else(|| anyhow::anyhow!("No temp index to flush"))?;

        if temp_map.is_empty() {
            // Nothing to flush, restore empty map
            self.temp_index = Some(HashMap::new());
            return Ok(());
        }

        // Convert HashMap to sorted Vec
        let mut partial_index: Vec<(Trigram, Vec<FileLocation>)> = temp_map.into_iter().collect();

        // Sort and deduplicate posting lists
        for (_, list) in partial_index.iter_mut() {
            list.sort_unstable();
            list.dedup();
        }

        // Sort by trigram
        partial_index.sort_unstable_by_key(|(trigram, _)| *trigram);

        // Write to temp file
        let partial_file = temp_dir.join(format!("partial_{}.bin", self.partial_indices.len()));
        self.write_partial_index(&partial_file, &partial_index)?;

        self.partial_indices.push(partial_file);

        // Create new empty temp_index for next batch
        self.temp_index = Some(HashMap::new());

        log::debug!(
            "Flushed batch {} with {} trigrams to disk",
            self.partial_indices.len(),
            partial_index.len()
        );

        Ok(())
    }

    /// Write a partial index to disk (simplified format for merging)
    fn write_partial_index(
        &self,
        path: &Path,
        index: &[(Trigram, Vec<FileLocation>)],
    ) -> Result<()> {
        use std::io::BufWriter;

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;

        let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file);

        // Write number of trigrams
        writer.write_all(&(index.len() as u64).to_le_bytes())?;

        // Write each (trigram, posting_list)
        for (trigram, locations) in index {
            writer.write_all(&trigram.to_le_bytes())?;
            writer.write_all(&(locations.len() as u32).to_le_bytes())?;

            // Fixed 8 B per posting: file_id u32 LE, line_no u32 LE
            for loc in locations {
                writer.write_all(&loc.file_id.to_le_bytes())?;
                writer.write_all(&loc.line_no.to_le_bytes())?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Add a file to the index and return its file_id
    pub fn add_file(&mut self, path: PathBuf) -> u32 {
        let file_id = self.files.len() as u32;
        self.files.push(path);
        file_id
    }

    /// Get file path for a file_id
    pub fn get_file(&self, file_id: u32) -> Option<&PathBuf> {
        self.files.get(file_id as usize)
    }

    /// Get total number of files
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Get total number of unique trigrams
    pub fn trigram_count(&self) -> usize {
        if self.mmap.is_some() {
            // Lazy-loaded mode
            self.num_trigrams
        } else {
            // In-memory mode
            self.index.len()
        }
    }

    /// Binary-search the mmapped directory for `trigram` (lazy mode only).
    ///
    /// Reads only the 4-byte trigram at each probe; the full 16-byte entry is
    /// decoded once, on a hit. `load` has already checked that the whole
    /// directory lies inside the mmap.
    fn find_entry(&self, trigram: Trigram) -> Option<DirectoryEntry> {
        let mmap = self.mmap.as_ref()?;
        let (mut lo, mut hi) = (0usize, self.num_trigrams);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let off = HEADER_SIZE + mid * DIR_ENTRY_SIZE;
            match read_u32(mmap, off).cmp(&trigram) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    return Some(DirectoryEntry {
                        trigram,
                        data_offset: read_u64(mmap, off + 4),
                        compressed_size: read_u32(mmap, off + 12),
                    });
                }
            }
        }
        None
    }

    /// Index a file's content
    ///
    /// Extracts all trigrams from the content and adds them to the inverted index.
    /// Must call finalize() after indexing all files to prepare for searching.
    pub fn index_file(&mut self, file_id: u32, content: &str) {
        let trigrams = extract_trigrams_with_locations(content, file_id);

        // Use the persistent HashMap for O(1) updates during batch processing
        if let Some(ref mut temp_map) = self.temp_index {
            for (trigram, location) in trigrams {
                temp_map
                    .entry(trigram)
                    .or_insert_with(Vec::new)
                    .push(location);
            }
        } else {
            panic!("Cannot call index_file() after finalize(). Index is read-only.");
        }
    }

    /// Build index from a collection of pre-extracted trigrams (bulk operation)
    ///
    /// This is much more efficient than calling index_file() multiple times,
    /// as it builds the HashMap once instead of rebuilding it for each file.
    pub fn build_from_trigrams(&mut self, trigrams: Vec<(Trigram, FileLocation)>) {
        let mut temp_map: HashMap<Trigram, Vec<FileLocation>> = HashMap::new();

        // Group trigrams into posting lists
        for (trigram, location) in trigrams {
            temp_map.entry(trigram).or_default().push(location);
        }

        // Convert to sorted Vec for binary search
        self.index = temp_map.into_iter().collect();

        // Clear temp_index since we're using the Vec directly
        self.temp_index = None;

        // Finalize immediately (sort and deduplicate)
        self.finalize();
    }

    /// Finalize the index by sorting all posting lists and the index itself
    ///
    /// Must be called after all files are indexed, before querying.
    /// Converts the HashMap to a sorted Vec for fast binary search.
    ///
    /// If batch flushing was enabled, finalization will be deferred until write()
    /// is called, which will perform streaming merge directly to disk.
    pub fn finalize(&mut self) {
        // If we have partial indices from batch flushing, DON'T merge yet
        // We'll do streaming merge in write() or write_with_streaming_merge()
        if !self.partial_indices.is_empty() {
            log::info!(
                "Deferring finalization - will stream merge {} partial indices during write()",
                self.partial_indices.len()
            );

            // Flush final batch if temp_index is not empty
            if let Some(ref temp_map) = self.temp_index
                && !temp_map.is_empty()
            {
                self.flush_batch().expect("Failed to flush final batch");
            }

            // Don't merge yet - write() will handle it
            return;
        }

        // Standard finalization (no batch flushing)
        // Convert HashMap to Vec if we have a temp index
        if let Some(temp_map) = self.temp_index.take() {
            self.index = temp_map.into_iter().collect();
        }

        // Sort, deduplicate, and cap posting lists
        let cap = self.max_posting_list_entries;
        for (trigram, list) in self.index.iter_mut() {
            list.sort_unstable();
            list.dedup(); // Remove duplicates (same trigram appearing multiple times on same line)
            if cap > 0 && list.len() > cap {
                log::warn!(
                    "Trigram 0x{:06X} posting list has {} entries (cap {}); truncating.",
                    trigram,
                    list.len(),
                    cap
                );
                list.truncate(cap);
            }
        }

        // Sort the index by trigram for binary search
        self.index.sort_unstable_by_key(|(trigram, _)| *trigram);
    }

    /// Merge all partial indices directly to trigrams.bin using streaming k-way merge
    ///
    /// This avoids loading the entire index into RAM by:
    /// 1. Opening all partial index files as readers
    /// 2. Performing k-way merge using a priority queue
    /// 3. Writing compressed posting lists directly to disk
    /// 4. Never accumulating more than K posting lists in memory at once
    fn merge_partial_indices_to_file(&mut self, output_path: &Path) -> Result<()> {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;
        use std::io::{BufReader, BufWriter, Read};

        log::info!(
            "Streaming merge of {} partial indices to {:?}",
            self.partial_indices.len(),
            output_path
        );

        // Open all partial indices as buffered readers
        struct PartialIndexReader {
            reader: BufReader<File>,
            current_trigram: Option<Trigram>,
            current_posting_list: Vec<FileLocation>,
            reader_id: usize,
        }

        let mut readers: Vec<PartialIndexReader> = Vec::new();

        for (idx, partial_path) in self.partial_indices.iter().enumerate() {
            let file = File::open(partial_path)
                .with_context(|| format!("Failed to open partial index: {:?}", partial_path))?;
            let mut reader = BufReader::with_capacity(16 * 1024 * 1024, file);

            // Read number of trigrams (we don't need it for streaming merge)
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;

            readers.push(PartialIndexReader {
                reader,
                current_trigram: None,
                current_posting_list: Vec::new(),
                reader_id: idx,
            });
        }

        // Helper to read next trigram from a reader
        fn read_next_trigram(reader: &mut PartialIndexReader) -> Result<bool> {
            // Try to read trigram
            let mut trigram_buf = [0u8; 4];
            match reader.reader.read_exact(&mut trigram_buf) {
                Ok(_) => {
                    let trigram = u32::from_le_bytes(trigram_buf);

                    // Read posting list size
                    let mut len_buf = [0u8; 4];
                    reader.reader.read_exact(&mut len_buf)?;
                    let list_len = u32::from_le_bytes(len_buf) as usize;

                    // Read all locations for this trigram (8 B fixed each)
                    let mut locations = Vec::with_capacity(list_len);
                    for _ in 0..list_len {
                        let mut loc_buf = [0u8; 8];
                        reader.reader.read_exact(&mut loc_buf)?;
                        locations.push(FileLocation::new(
                            read_u32(&loc_buf, 0),
                            read_u32(&loc_buf, 4),
                        ));
                    }

                    reader.current_trigram = Some(trigram);
                    reader.current_posting_list = locations;
                    Ok(true)
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    reader.current_trigram = None;
                    Ok(false)
                }
                Err(e) => Err(e.into()),
            }
        }

        // Initialize: read first trigram from each reader
        for reader in &mut readers {
            read_next_trigram(reader)?;
        }

        // Priority queue entry for k-way merge
        #[derive(Eq, PartialEq)]
        struct HeapEntry {
            trigram: Trigram,
            reader_id: usize,
        }

        impl Ord for HeapEntry {
            fn cmp(&self, other: &Self) -> Ordering {
                // Reverse for min-heap
                other
                    .trigram
                    .cmp(&self.trigram)
                    .then_with(|| other.reader_id.cmp(&self.reader_id))
            }
        }

        impl PartialOrd for HeapEntry {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        // Build initial heap
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
        for reader in &readers {
            if let Some(trigram) = reader.current_trigram {
                heap.push(HeapEntry {
                    trigram,
                    reader_id: reader.reader_id,
                });
            }
        }

        // Open the temp output file for writing. Both passes below (data
        // pass, then directory-insertion rewrite) target the temp path; the
        // final file is only replaced by one atomic rename at the end.
        let tmp_path = crate::atomic_write::tmp_path_for(output_path);
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp_path)
            .with_context(|| format!("Failed to create {}", tmp_path.display()))?;

        let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file);

        // Write placeholder header (we'll update it at the end)
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&0u64.to_le_bytes())?; // num_trigrams (placeholder)
        writer.write_all(&(self.files.len() as u64).to_le_bytes())?; // num_files
        writer.write_all(&0u64.to_le_bytes())?; // paths_offset (placeholder)

        // We'll build the directory as we go
        let mut directory: Vec<DirectoryEntry> = Vec::new();
        let mut num_trigrams = 0u64;

        // K-way merge loop
        let mut current_trigram: Option<Trigram> = None;
        let mut merged_locations: Vec<FileLocation> = Vec::new();

        while let Some(entry) = heap.pop() {
            let reader = &mut readers[entry.reader_id];

            // If this is a new trigram, write the previous one
            if let Some(trigram) = current_trigram.filter(|&t| t != entry.trigram) {
                merged_locations.sort_unstable();
                merged_locations.dedup();

                let cap = self.max_posting_list_entries;
                if cap > 0 && merged_locations.len() > cap {
                    log::warn!(
                        "Trigram 0x{:06X} posting list has {} entries (cap {}); truncating.",
                        trigram,
                        merged_locations.len(),
                        cap
                    );
                    merged_locations.truncate(cap);
                }

                // Compress and write this trigram's posting list
                let data_offset = writer.stream_position()?;
                let compressed_size =
                    self.write_compressed_posting_list(&mut writer, &merged_locations)?;

                directory.push(DirectoryEntry {
                    trigram,
                    data_offset,
                    compressed_size,
                });

                num_trigrams += 1;
                merged_locations.clear();
            }

            // Set current trigram
            current_trigram = Some(entry.trigram);

            // Merge this reader's posting list into accumulated list
            merged_locations.extend_from_slice(&reader.current_posting_list);

            // Advance this reader to next trigram
            if read_next_trigram(reader)?
                && let Some(next_trigram) = reader.current_trigram
            {
                heap.push(HeapEntry {
                    trigram: next_trigram,
                    reader_id: entry.reader_id,
                });
            }
        }

        // Write final trigram
        if let Some(trigram) = current_trigram {
            merged_locations.sort_unstable();
            merged_locations.dedup();

            let cap = self.max_posting_list_entries;
            if cap > 0 && merged_locations.len() > cap {
                log::warn!(
                    "Trigram 0x{:06X} posting list has {} entries (cap {}); truncating.",
                    trigram,
                    merged_locations.len(),
                    cap
                );
                merged_locations.truncate(cap);
            }

            let data_offset = writer.stream_position()?;
            let compressed_size =
                self.write_compressed_posting_list(&mut writer, &merged_locations)?;

            directory.push(DirectoryEntry {
                trigram,
                data_offset,
                compressed_size,
            });

            num_trigrams += 1;
        }

        log::info!(
            "Merged {} trigrams from {} partial indices",
            num_trigrams,
            self.partial_indices.len()
        );

        // Data section length: everything after the header so far. The final
        // layout inserts the directory between header and data, so the paths
        // section lands at header + directory + data.
        let data_len = writer.stream_position()? - HEADER_SIZE as u64;
        let paths_offset = (HEADER_SIZE + directory.len() * DIR_ENTRY_SIZE) as u64 + data_len;

        // Write file paths after data section
        for file_path in &self.files {
            let path_str = file_path.to_string_lossy();
            let path_bytes = path_str.as_bytes();
            write_varint(&mut writer, path_bytes.len() as u32)?;
            writer.write_all(path_bytes)?;
        }

        // Flush before we rewrite the beginning
        writer.flush()?;
        drop(writer);

        // Now we need to insert the directory at the beginning
        // We'll read the data+files we just wrote, then rewrite the file with directory in between
        use std::io::{Seek, SeekFrom};

        // Read data and files sections
        let mut temp_data = Vec::new();
        {
            let mut file = File::open(&tmp_path)?;
            file.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
            file.read_to_end(&mut temp_data)?;
        }

        // Rewrite the temp file with correct structure
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&tmp_path)?;
        let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file);

        // Write header with correct num_trigrams and paths_offset
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&num_trigrams.to_le_bytes())?;
        writer.write_all(&(self.files.len() as u64).to_le_bytes())?;
        writer.write_all(&paths_offset.to_le_bytes())?;

        // Write directory
        for entry in &directory {
            writer.write_all(&entry.trigram.to_le_bytes())?;
            // Adjust data offset to account for directory size
            let adjusted_offset = entry.data_offset + (directory.len() * DIR_ENTRY_SIZE) as u64;
            writer.write_all(&adjusted_offset.to_le_bytes())?;
            writer.write_all(&entry.compressed_size.to_le_bytes())?;
        }

        // Write data and files sections
        writer.write_all(&temp_data)?;

        // Flush and sync, then publish atomically
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        crate::atomic_write::atomic_replace(&tmp_path, output_path)
            .with_context(|| format!("Failed to move {} into place", output_path.display()))?;

        // Clean up partial index files
        for partial_path in &self.partial_indices {
            let _ = std::fs::remove_file(partial_path);
        }
        if let Some(ref temp_dir) = self.temp_dir {
            let _ = std::fs::remove_dir(temp_dir);
        }

        log::info!("Wrote {} trigrams to {:?}", num_trigrams, output_path);

        Ok(())
    }

    /// Write a compressed posting list to the writer and return the compressed size
    fn write_compressed_posting_list(
        &self,
        writer: &mut impl Write,
        locations: &[FileLocation],
    ) -> Result<u32> {
        let mut compressed = Vec::with_capacity(locations.len() + 16);
        encode_posting_list(locations, &mut compressed)?;
        let compressed_size = compressed.len() as u32;
        writer.write_all(&compressed)?;
        Ok(compressed_size)
    }

    /// Search for a plain text pattern
    ///
    /// Returns candidate file locations that could contain the pattern.
    /// Caller must verify actual matches.
    ///
    /// In lazy-loaded mode: Decompresses posting lists on-demand from mmap.
    /// In in-memory mode: Uses pre-loaded posting lists.
    pub fn search(&self, pattern: &str) -> Vec<FileLocation> {
        self.search_impl(pattern, false)
    }

    /// Candidate lines for `pattern`, allowing the intersection to stop early.
    ///
    /// Every posting list of a common trigram (`ide`, `ent`, `nt_` …) is streamed
    /// in full on every query, and on a pattern like `ident_7` the four largest
    /// lists cost ~45 ms without removing a single candidate that the smallest
    /// list (`t_7`) had not already narrowed to. The caller verifies every
    /// candidate line anyway — exactly, and on a thread pool — so once the next
    /// list is much larger than the surviving candidate set, decoding it costs
    /// more than verifying the extra candidates would. This method stops there;
    /// the result is a superset of [`Self::search`] and never misses a match.
    pub fn search_candidates(&self, pattern: &str) -> Vec<FileLocation> {
        self.search_impl(pattern, true)
    }

    fn search_impl(&self, pattern: &str, allow_skip: bool) -> Vec<FileLocation> {
        if pattern.len() < 3 {
            // Pattern too short for trigrams - caller must fall back to full scan
            return vec![];
        }

        let mut trigrams = extract_trigrams(pattern);
        // A repeated trigram (e.g. "aaaa") would otherwise be intersected with itself
        trigrams.sort_unstable();
        trigrams.dedup();
        if trigrams.is_empty() {
            return vec![];
        }

        // Check if we're in lazy-loaded mode or in-memory mode
        if let Some(ref mmap) = self.mmap {
            // Lazy-loaded mode: look up every directory entry first; any miss
            // means the pattern cannot match.
            let mut entries: Vec<DirectoryEntry> = Vec::with_capacity(trigrams.len());
            for trigram in &trigrams {
                match self.find_entry(*trigram) {
                    Some(entry) => entries.push(entry),
                    None => return vec![],
                }
            }

            // Smallest compressed list first: it is the only one fully decoded.
            entries.sort_by_key(|e| e.compressed_size);

            let mut cands = match decompress_posting_list(
                mmap,
                entries[0].data_offset,
                entries[0].compressed_size,
            ) {
                Ok(locations) => locations,
                Err(e) => {
                    log::warn!(
                        "Failed to decompress posting list for trigram {}: {}",
                        entries[0].trigram,
                        e
                    );
                    return vec![];
                }
            };
            cands.dedup_by_key(|l| key(l));

            // Stream the remaining lists against the shrinking candidate set.
            for entry in &entries[1..] {
                if cands.is_empty() {
                    break;
                }
                // Lists are ascending, so once one is too big to be worth
                // decoding, all the remaining ones are too.
                if allow_skip
                    && entry.compressed_size as usize
                        > cands.len().saturating_mul(SKIP_BYTES_PER_CANDIDATE)
                {
                    log::debug!(
                        "Intersection stopped early: {} candidates, next list {} bytes",
                        cands.len(),
                        entry.compressed_size
                    );
                    break;
                }
                let start = entry.data_offset as usize;
                let end = start + entry.compressed_size as usize;
                if end > mmap.len() {
                    log::warn!(
                        "Posting list out of bounds for trigram {}: offset={}, size={}, mmap_len={}",
                        entry.trigram,
                        entry.data_offset,
                        entry.compressed_size,
                        mmap.len()
                    );
                    return vec![];
                }
                let result = PostingCursor::new(&mmap[start..end])
                    .and_then(|mut cursor| intersect_with_cursor(&cands, &mut cursor));
                match result {
                    Ok(next) => cands = next,
                    Err(e) => {
                        log::warn!(
                            "Failed to decompress posting list for trigram {}: {}",
                            entry.trigram,
                            e
                        );
                        return vec![];
                    }
                }
            }

            cands
        } else {
            // In-memory mode: use pre-loaded index
            let mut posting_lists: Vec<&[FileLocation]> = Vec::with_capacity(trigrams.len());
            for trigram in &trigrams {
                match self.index.binary_search_by_key(trigram, |(t, _)| *t) {
                    Ok(idx) => posting_lists.push(&self.index[idx].1),
                    // Trigram missing - pattern cannot match
                    Err(_) => return vec![],
                }
            }

            // Sort by list size (smallest first for efficient intersection)
            posting_lists.sort_by_key(|list| list.len());

            if !allow_skip {
                return intersect_sorted(&posting_lists);
            }
            let mut cands: Vec<FileLocation> = posting_lists[0].to_vec();
            cands.dedup_by_key(|l| key(l));
            for list in &posting_lists[1..] {
                if cands.is_empty() {
                    break;
                }
                // ~1.3 bytes per posting on disk; same rule as the lazy path.
                if list.len().saturating_mul(13) / 10
                    > cands.len().saturating_mul(SKIP_BYTES_PER_CANDIDATE)
                {
                    break;
                }
                cands = intersect_two(&cands, list);
            }
            cands
        }
    }

    /// Case-insensitive candidate lines for an ASCII literal.
    ///
    /// Every 3-byte window of `literal` is looked up under all of its ASCII
    /// case variants (≤ 8 trigrams: each letter byte has two forms) and the
    /// variants' posting lists are OR-ed; windows are then AND-ed, cheapest
    /// first, with the same early stop as [`search_candidates`](Self::search_candidates).
    /// The result is a superset of every line that contains the literal in any
    /// ASCII casing; callers verify with a `(?i)` regex.
    ///
    /// Only ASCII case is folded here. Unicode simple case folding also maps
    /// `k` to U+212A KELVIN SIGN and `s` to U+017F LATIN SMALL LETTER LONG S;
    /// a caller whose literal contains `k` or `s` must union
    /// [`exotic_fold_lines`](Self::exotic_fold_lines) into the result to keep
    /// parity with the regex engine. A non-ASCII literal is not supported:
    /// the caller must fall back to a scan.
    pub fn search_candidates_fold(&self, literal: &[u8]) -> Vec<FileLocation> {
        if literal.len() < 3 || !literal.is_ascii() {
            return vec![];
        }

        // Variant set per window; identical windows are intersected once.
        let mut windows: Vec<Vec<Trigram>> = literal.windows(3).map(case_variants).collect();
        windows.sort_unstable();
        windows.dedup();

        if let Some(ref mmap) = self.mmap {
            // (weight, present variants) per window; a window with no present
            // variant cannot match in any casing.
            let mut ws: Vec<(usize, Vec<DirectoryEntry>)> = Vec::with_capacity(windows.len());
            for variants in &windows {
                let entries: Vec<DirectoryEntry> = variants
                    .iter()
                    .filter_map(|t| self.find_entry(*t))
                    .collect();
                if entries.is_empty() {
                    return vec![];
                }
                let weight = entries.iter().map(|e| e.compressed_size as usize).sum();
                ws.push((weight, entries));
            }
            ws.sort_by_key(|w| w.0);

            // Cheapest window: decode every variant and merge.
            let mut cands: Vec<FileLocation> = Vec::new();
            for entry in &ws[0].1 {
                match decompress_posting_list(mmap, entry.data_offset, entry.compressed_size) {
                    Ok(list) => cands.extend(list),
                    Err(e) => {
                        log::warn!(
                            "Failed to decompress posting list for trigram {}: {}",
                            entry.trigram,
                            e
                        );
                        return vec![];
                    }
                }
            }
            cands.sort_unstable();
            cands.dedup_by_key(|l| key(l));

            for (weight, entries) in &ws[1..] {
                if cands.is_empty() {
                    break;
                }
                if *weight > cands.len().saturating_mul(SKIP_BYTES_PER_CANDIDATE) {
                    log::debug!(
                        "Fold intersection stopped early: {} candidates, next window {} bytes",
                        cands.len(),
                        weight
                    );
                    break;
                }
                let mut keep = vec![false; cands.len()];
                let mut window_ok = true;
                for entry in entries {
                    let start = entry.data_offset as usize;
                    let end = start + entry.compressed_size as usize;
                    let result = if end > mmap.len() {
                        Err(anyhow::anyhow!("posting list out of bounds"))
                    } else {
                        PostingCursor::new(&mmap[start..end])
                            .and_then(|mut cur| mark_with_cursor(&cands, &mut cur, &mut keep))
                    };
                    if let Err(e) = result {
                        // Skipping a window keeps the result a superset.
                        log::warn!(
                            "Failed to read posting list for trigram {}: {}; window skipped",
                            entry.trigram,
                            e
                        );
                        window_ok = false;
                        break;
                    }
                }
                if !window_ok {
                    continue;
                }
                cands = cands
                    .iter()
                    .zip(&keep)
                    .filter(|(_, k)| **k)
                    .map(|(c, _)| *c)
                    .collect();
            }
            cands
        } else {
            let mut ws: Vec<(usize, Vec<&[FileLocation]>)> = Vec::with_capacity(windows.len());
            for variants in &windows {
                let lists: Vec<&[FileLocation]> = variants
                    .iter()
                    .filter_map(|t| {
                        self.index
                            .binary_search_by_key(t, |(x, _)| *x)
                            .ok()
                            .map(|i| self.index[i].1.as_slice())
                    })
                    .collect();
                if lists.is_empty() {
                    return vec![];
                }
                // ~1.3 bytes per posting on disk; same rule as the lazy path.
                let weight = lists.iter().map(|l| l.len()).sum::<usize>() * 13 / 10;
                ws.push((weight, lists));
            }
            ws.sort_by_key(|w| w.0);

            let mut cands: Vec<FileLocation> =
                ws[0].1.iter().flat_map(|l| l.iter().copied()).collect();
            cands.sort_unstable();
            cands.dedup_by_key(|l| key(l));

            for (weight, lists) in &ws[1..] {
                if cands.is_empty() {
                    break;
                }
                if *weight > cands.len().saturating_mul(SKIP_BYTES_PER_CANDIDATE) {
                    break;
                }
                let mut next: Vec<FileLocation> = lists
                    .iter()
                    .flat_map(|l| intersect_two(&cands, l))
                    .collect();
                next.sort_unstable();
                next.dedup_by_key(|l| key(l));
                cands = next;
            }
            cands
        }
    }

    /// Every line that contains U+212A KELVIN SIGN or U+017F LONG S.
    ///
    /// Under Unicode simple case folding a `(?i)` regex lets `k` match the
    /// Kelvin sign and `s` match the long s. Their UTF-8 forms (`E2 84 AA` and
    /// `C5 BF`) never align with the ASCII windows of
    /// [`search_candidates_fold`](Self::search_candidates_fold), so a line whose
    /// only match uses one of them would be missed. The Kelvin sign is its own
    /// trigram; the long s appears in a `C5 BF ?` or `? C5 BF` trigram of any
    /// line long enough to hold a ≥3-byte literal. On code corpora every lookup
    /// misses, so this costs a few hundred directory probes and returns nothing.
    pub fn exotic_fold_lines(&self) -> Vec<FileLocation> {
        const KELVIN: Trigram = 0xE2_84_AA;
        const LONG_S_LO: Trigram = 0xC5_BF_00;
        const LONG_S_HI: Trigram = 0xC5_BF_FF;

        let mut out: Vec<FileLocation> = Vec::new();
        if let Some(ref mmap) = self.mmap {
            let mut entries: Vec<DirectoryEntry> = Vec::new();
            entries.extend(self.find_entry(KELVIN));
            entries.extend(self.dir_range(LONG_S_LO, LONG_S_HI));
            for first in 0u32..=0xFF {
                entries.extend(self.find_entry(first << 16 | 0xC5_BF));
            }
            for entry in entries {
                match decompress_posting_list(mmap, entry.data_offset, entry.compressed_size) {
                    Ok(list) => out.extend(list),
                    Err(e) => log::warn!(
                        "Failed to decompress posting list for trigram {}: {}",
                        entry.trigram,
                        e
                    ),
                }
            }
        } else {
            let lo = self.index.partition_point(|(t, _)| *t < LONG_S_LO);
            let hi = self.index.partition_point(|(t, _)| *t <= LONG_S_HI);
            for (_, list) in &self.index[lo..hi] {
                out.extend(list.iter().copied());
            }
            let mut singles: Vec<Trigram> = (0u32..=0xFF).map(|f| f << 16 | 0xC5_BF).collect();
            singles.push(KELVIN);
            for t in singles {
                if let Ok(i) = self.index.binary_search_by_key(&t, |(x, _)| *x) {
                    out.extend(self.index[i].1.iter().copied());
                }
            }
        }
        out.sort_unstable();
        out.dedup_by_key(|l| key(l));
        out
    }

    /// Directory entries with `lo <= trigram <= hi` (lazy mode only).
    fn dir_range(&self, lo: Trigram, hi: Trigram) -> Vec<DirectoryEntry> {
        let Some(mmap) = self.mmap.as_ref() else {
            return vec![];
        };
        let at = |i: usize| read_u32(mmap, HEADER_SIZE + i * DIR_ENTRY_SIZE);
        let (mut l, mut h) = (0usize, self.num_trigrams);
        while l < h {
            let mid = l + (h - l) / 2;
            if at(mid) < lo {
                l = mid + 1;
            } else {
                h = mid;
            }
        }
        let mut out = Vec::new();
        let mut i = l;
        while i < self.num_trigrams {
            let off = HEADER_SIZE + i * DIR_ENTRY_SIZE;
            let trigram = read_u32(mmap, off);
            if trigram > hi {
                break;
            }
            out.push(DirectoryEntry {
                trigram,
                data_offset: read_u64(mmap, off + 4),
                compressed_size: read_u32(mmap, off + 12),
            });
            i += 1;
        }
        out
    }

    /// Search for a plain text pattern and return the distinct candidate file IDs
    ///
    /// Output is sorted ascending. Same caveats as [`search`](Self::search):
    /// callers must verify actual matches, and short patterns yield nothing.
    pub fn search_files(&self, pattern: &str) -> Vec<u32> {
        let mut files: Vec<u32> = self
            .search(pattern)
            .into_iter()
            .map(|l| l.file_id)
            .collect();
        // `search` output is sorted by (file_id, line_no), so duplicates are adjacent
        files.dedup();
        files
    }

    /// Get posting list for a specific trigram (for debugging)
    pub fn get_posting_list(&self, trigram: Trigram) -> Option<&Vec<FileLocation>> {
        self.index
            .binary_search_by_key(&trigram, |(t, _)| *t)
            .ok()
            .map(|idx| &self.index[idx].1)
    }

    /// Write the trigram index to disk
    ///
    /// Binary format V4 (lazy-loadable with directory + data separation):
    /// - Header (32 bytes): magic, version, num_trigrams, num_files, paths_offset
    /// - Directory Section (16 bytes per trigram, sorted by trigram):
    ///   - trigram: u32 (4 bytes)
    ///   - data_offset: u64 (8 bytes) - absolute offset in file
    ///   - compressed_size: u32 (4 bytes) - size of compressed posting list
    /// - Data Section (variable size):
    ///   - Posting lists as per-file blocks (see [`encode_posting_list`])
    /// - File Paths Section at `paths_offset` (variable size):
    ///   - path_len: varint
    ///   - path_bytes: [u8; path_len]
    pub fn write(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();

        // If we have partial indices from batch flushing, use streaming merge
        if !self.partial_indices.is_empty() {
            log::info!(
                "Using streaming merge to write {} partial indices",
                self.partial_indices.len()
            );
            return self.merge_partial_indices_to_file(path);
        }

        // Standard write path (no batch flushing).
        // Crash-safe: stream into `<path>.tmp`, sync, then rename over `path`.
        // A crash mid-write leaves the previous trigrams.bin intact (or no file
        // at all), never a short file with valid magic bytes.
        let tmp_path = crate::atomic_write::tmp_path_for(path);
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp_path)
            .with_context(|| format!("Failed to create {}", tmp_path.display()))?;

        // Use a large buffer (16MB) for streaming writes
        let mut writer = std::io::BufWriter::with_capacity(16 * 1024 * 1024, file);

        // Step 1: Compress all posting lists and build the directory. Data
        // offsets are known up front because the directory has a fixed size.
        let data_start = (HEADER_SIZE + self.index.len() * DIR_ENTRY_SIZE) as u64;
        let mut current_offset = data_start;
        let mut directory: Vec<DirectoryEntry> = Vec::with_capacity(self.index.len());
        let mut compressed_lists: Vec<Vec<u8>> = Vec::with_capacity(self.index.len());

        for (trigram, locations) in &self.index {
            let mut compressed = Vec::with_capacity(locations.len() + 16);
            encode_posting_list(locations, &mut compressed)?;

            directory.push(DirectoryEntry {
                trigram: *trigram,
                data_offset: current_offset,
                compressed_size: compressed.len() as u32,
            });
            current_offset += compressed.len() as u64;
            compressed_lists.push(compressed);
        }
        let paths_offset = current_offset;

        // Step 2: Write header
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&(self.index.len() as u64).to_le_bytes())?; // num_trigrams
        writer.write_all(&(self.files.len() as u64).to_le_bytes())?; // num_files
        writer.write_all(&paths_offset.to_le_bytes())?; // paths_offset

        // Step 3: Write directory
        for entry in &directory {
            writer.write_all(&entry.trigram.to_le_bytes())?;
            writer.write_all(&entry.data_offset.to_le_bytes())?;
            writer.write_all(&entry.compressed_size.to_le_bytes())?;
        }

        // Step 4: Write data section (compressed posting lists)
        for compressed in &compressed_lists {
            writer.write_all(compressed)?;
        }

        // Step 5: Write file paths
        for file_path in &self.files {
            let path_str = file_path.to_string_lossy();
            let path_bytes = path_str.as_bytes();
            write_varint(&mut writer, path_bytes.len() as u32)?;
            writer.write_all(path_bytes)?;
        }

        // Flush and sync, then publish atomically
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        crate::atomic_write::atomic_replace(&tmp_path, path)
            .with_context(|| format!("Failed to move {} into place", path.display()))?;

        log::info!(
            "Wrote lazy-loadable trigram index: {} trigrams, {} files to {:?}",
            self.index.len(),
            self.files.len(),
            path
        );

        Ok(())
    }

    /// Load trigram index from disk using memory-mapped I/O with lazy loading
    ///
    /// Binary format V4: validates the header, bounds-checks the directory
    /// against `paths_offset`, and decodes only the file paths. The directory
    /// stays in the mmap and is binary-searched in place; posting lists are
    /// decompressed on demand during search. Cost is O(files), not O(trigrams).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        let file =
            File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;

        // Memory-map the file (keep it alive for lazy access)
        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .with_context(|| format!("Failed to mmap {}", path.display()))?
        };

        // Validate magic + version before the size check so an older (shorter
        // header) format still reports the version mismatch callers key on.
        if mmap.len() < 8 {
            anyhow::bail!(
                "trigrams.bin too small (expected at least {} bytes)",
                HEADER_SIZE
            );
        }

        if &mmap[0..4] != MAGIC {
            anyhow::bail!("Invalid trigrams.bin (wrong magic bytes)");
        }

        let version = read_u32(&mmap, 4);
        if version != VERSION {
            anyhow::bail!(
                "Unsupported trigrams.bin version: {} (expected {}). Please re-index with 'reflex index'.",
                version,
                VERSION
            );
        }

        if mmap.len() < HEADER_SIZE {
            anyhow::bail!(
                "trigrams.bin too small (expected at least {} bytes)",
                HEADER_SIZE
            );
        }

        let num_trigrams = read_u64(&mmap, NUM_TRIGRAMS_OFFSET) as usize;
        let num_files = read_u64(&mmap, NUM_FILES_OFFSET) as usize;
        let paths_offset = read_u64(&mmap, PATHS_OFFSET_OFFSET);

        log::debug!(
            "Loading lazy trigram index: {} trigrams, {} files",
            num_trigrams,
            num_files
        );

        // Layout check: header + directory must end at or before the paths
        // section, which must lie inside the file. This is the only guard the
        // in-place directory search relies on.
        let directory_end = num_trigrams
            .checked_mul(DIR_ENTRY_SIZE)
            .and_then(|d| d.checked_add(HEADER_SIZE))
            .ok_or_else(|| anyhow::anyhow!("trigrams.bin directory size overflows"))?
            as u64;
        if directory_end > paths_offset || paths_offset > mmap.len() as u64 {
            anyhow::bail!(
                "trigrams.bin layout out of bounds: directory_end={}, paths_offset={}, len={}",
                directory_end,
                paths_offset,
                mmap.len()
            );
        }

        // Read file paths (varint-encoded lengths)
        let mut pos = paths_offset as usize;
        let mut files = Vec::with_capacity(num_files);
        for _ in 0..num_files {
            let (path_len, consumed) = read_varint(&mmap[pos..])?;
            pos += consumed;
            let path_len = path_len as usize;

            if pos + path_len > mmap.len() {
                anyhow::bail!("Truncated file path at pos={}", pos);
            }

            let path_bytes = &mmap[pos..pos + path_len];
            let path_str = std::str::from_utf8(path_bytes).context("Invalid UTF-8 in file path")?;
            files.push(PathBuf::from(path_str));
            pos += path_len;
        }

        log::info!(
            "Loaded lazy trigram index: {} trigrams, {} files (directory: {} KB, in mmap)",
            num_trigrams,
            num_files,
            num_trigrams * DIR_ENTRY_SIZE / 1024
        );

        Ok(Self {
            index: Vec::new(), // Empty in lazy mode
            files,
            temp_index: None,
            mmap: Some(mmap), // Keep mmap alive for lazy decompression!
            num_trigrams,
            partial_indices: Vec::new(),
            temp_dir: None,
            max_posting_list_entries: 0,
        })
    }
}

impl Default for TrigramIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract all trigrams from text
///
/// Returns a vector of trigrams (without location info).
pub fn extract_trigrams(text: &str) -> Vec<Trigram> {
    let bytes = text.as_bytes();
    let mut trigrams = Vec::new();

    for i in 0..bytes.len().saturating_sub(2) {
        let trigram = bytes_to_trigram(&bytes[i..i + 3]);
        trigrams.push(trigram);
    }

    trigrams
}

/// Extract trigrams with file location information
///
/// Returns one `(trigram, location)` pair per **distinct trigram per line**,
/// for building the inverted index. Emitting every byte position (as V3 did)
/// only produced duplicate `(file, line)` keys that the intersection threw
/// away; deduplicating here is what makes `finalize`'s `dedup()` a no-op.
///
/// A trigram that spans a newline is attributed to the line of its first
/// byte, except that a trigram *starting* on `\n` belongs to the next line
/// (unchanged from V3, so `"lo\n"`-style queries keep matching).
pub fn extract_trigrams_with_locations(text: &str, file_id: u32) -> Vec<(Trigram, FileLocation)> {
    let bytes = text.as_bytes();
    let mut result = Vec::with_capacity(bytes.len().saturating_sub(2));
    // Trigrams of the line being scanned; sorted + deduplicated on flush
    let mut line_trigrams: Vec<Trigram> = Vec::with_capacity(128);
    let mut line_no: u32 = 1;

    fn flush(
        line_trigrams: &mut Vec<Trigram>,
        file_id: u32,
        line_no: u32,
        result: &mut Vec<(Trigram, FileLocation)>,
    ) {
        line_trigrams.sort_unstable();
        line_trigrams.dedup();
        let location = FileLocation::new(file_id, line_no);
        result.extend(line_trigrams.drain(..).map(|t| (t, location)));
    }

    for (i, &byte) in bytes.iter().enumerate() {
        // Track newlines
        if byte == b'\n' {
            flush(&mut line_trigrams, file_id, line_no, &mut result);
            line_no += 1;
        }

        // Extract trigram
        if i + 2 < bytes.len() {
            line_trigrams.push(bytes_to_trigram(&bytes[i..i + 3]));
        }
    }
    flush(&mut line_trigrams, file_id, line_no, &mut result);

    result
}

/// Convert 3 bytes to a trigram (packed u32)
#[inline]
fn bytes_to_trigram(bytes: &[u8]) -> Trigram {
    debug_assert_eq!(bytes.len(), 3);
    (bytes[0] as u32) << 16 | (bytes[1] as u32) << 8 | (bytes[2] as u32)
}

/// All ASCII case variants of a 3-byte window, sorted and deduplicated.
///
/// Each ASCII letter byte contributes two forms, every other byte one, so the
/// result has 1, 2, 4 or 8 trigrams.
fn case_variants(window: &[u8]) -> Vec<Trigram> {
    debug_assert_eq!(window.len(), 3);
    let forms = |b: u8| -> Vec<u8> {
        if b.is_ascii_alphabetic() {
            vec![b.to_ascii_lowercase(), b.to_ascii_uppercase()]
        } else {
            vec![b]
        }
    };
    let mut out = Vec::with_capacity(8);
    for a in forms(window[0]) {
        for b in forms(window[1]) {
            for c in forms(window[2]) {
                out.push(bytes_to_trigram(&[a, b, c]));
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Convert trigram back to bytes (for debugging)
#[allow(dead_code)]
fn trigram_to_bytes(trigram: Trigram) -> [u8; 3] {
    [
        ((trigram >> 16) & 0xFF) as u8,
        ((trigram >> 8) & 0xFF) as u8,
        (trigram & 0xFF) as u8,
    ]
}

/// Intersection key: posting lists are matched on (file_id, line_no)
#[inline]
fn key(l: &FileLocation) -> (u32, u32) {
    (l.file_id, l.line_no)
}

/// Sorted two-pointer intersection of `a` and `b` on `key`.
///
/// `a` is the running candidate set (sorted, deduplicated by key). `b` is a sorted
/// posting list; should it hold several entries per key, all of them are
/// consumed on a match. When `b` is much larger
/// than `a`, the scan of `b` gallops via `partition_point`. Output keeps the
/// `FileLocation` from `a`, so it stays sorted and key-unique.
fn intersect_two(a: &[FileLocation], b: &[FileLocation]) -> Vec<FileLocation> {
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    let gallop = b.len() > 8 * a.len();
    let (mut i, mut j) = (0, 0);

    while i < a.len() && j < b.len() {
        let ka = key(&a[i]);
        match key(&b[j]).cmp(&ka) {
            std::cmp::Ordering::Less => {
                if gallop {
                    j += b[j..].partition_point(|x| key(x) < ka);
                } else {
                    j += 1;
                }
            }
            std::cmp::Ordering::Greater => i += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                // Consume every entry in `b` with this key
                j += 1;
                while j < b.len() && key(&b[j]) == ka {
                    j += 1;
                }
            }
        }
    }

    out
}

/// Intersect sorted posting lists by (file_id, line_no).
///
/// Returns locations where ALL trigrams appear on the SAME line (not just in the
/// same file), keeping the `FileLocation` from the first list. Callers should
/// pass lists smallest-first: the first list is copied and deduplicated by key,
/// then folded against the rest in linear time.
pub(crate) fn intersect_sorted(lists: &[&[FileLocation]]) -> Vec<FileLocation> {
    let Some((first, rest)) = lists.split_first() else {
        return vec![];
    };

    let mut cands = first.to_vec();
    cands.dedup_by_key(|l| key(l));

    for list in rest {
        if cands.is_empty() {
            break;
        }
        cands = intersect_two(&cands, list);
    }

    cands
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `search_candidates` may stop intersecting once the next list is much larger
    /// than the candidate set; it must then return a superset of the exact
    /// intersection, never a subset, in both lazy and in-memory modes.
    #[test]
    fn search_candidates_is_a_superset_of_search() {
        let mut index = TrigramIndex::new();
        // Many lines share the common trigrams of "ident_"; only a few carry "t_7".
        for f in 0..20u32 {
            let id = index.add_file(PathBuf::from(format!("f{f}.rs")));
            let mut text = String::new();
            for i in 0..200 {
                text.push_str(&format!("let ident_{} = ident_{}; // filler\n", i, i + 1));
            }
            text.push_str("let ident_7 = 1;\n");
            index.index_file(id, &text);
        }
        index.finalize();

        let exact = index.search("ident_7");
        let cands = index.search_candidates("ident_7");
        assert!(!exact.is_empty());
        let exact_keys: std::collections::HashSet<_> = exact.iter().map(key).collect();
        let cand_keys: std::collections::HashSet<_> = cands.iter().map(key).collect();
        assert!(
            exact_keys.is_subset(&cand_keys),
            "candidates must cover every match"
        );

        // Same contract through the on-disk (lazy) path.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trigrams.bin");
        index.write(&path).unwrap();
        let lazy = TrigramIndex::load(&path).unwrap();
        let lazy_exact: std::collections::HashSet<_> =
            lazy.search("ident_7").iter().map(key).collect();
        let lazy_cands: std::collections::HashSet<_> =
            lazy.search_candidates("ident_7").iter().map(key).collect();
        assert_eq!(lazy_exact, exact_keys);
        assert!(lazy_exact.is_subset(&lazy_cands));
    }
    /// `search_candidates_fold` plus `exotic_fold_lines` must cover every line a
    /// `(?i)` regex matches, in both modes, including the Kelvin-sign and long-s
    /// folds the regex crate applies to `k` and `s`.
    #[test]
    fn search_candidates_fold_covers_case_insensitive_matches() {
        let mut index = TrigramIndex::new();
        let mut texts: Vec<String> = Vec::new();
        for f in 0..10u32 {
            let id = index.add_file(PathBuf::from(format!("f{f}.rs")));
            let mut text = String::new();
            for i in 0..100 {
                text.push_str(&format!("let ident_{} = other_{}; // filler\n", i, i + 1));
            }
            text.push_str("let RealmId = 1;\n");
            text.push_str("let realmId = 2;\n");
            text.push_str("let REALMID = 3;\n");
            text.push_str("let realm_id = 4;\n");
            text.push_str("let kelvin = 5;\n");
            text.push_str("let \u{212A}elvin = 6;\n");
            text.push_str("let \u{017F}tatus = 7;\n");
            text.push_str("let STATUS = 8;\n");
            index.index_file(id, &text);
            texts.push(text);
        }
        index.finalize();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trigrams.bin");
        index.write(&path).unwrap();
        let lazy = TrigramIndex::load(&path).unwrap();

        for literal in ["realmid", "kelvin", "status", "REALM_ID"] {
            let re = regex::Regex::new(&format!("(?i){}", literal)).unwrap();
            let mut expected = std::collections::HashSet::new();
            for (f, text) in texts.iter().enumerate() {
                for (i, line) in text.lines().enumerate() {
                    if re.is_match(line) {
                        expected.insert((f as u32, i as u32 + 1));
                    }
                }
            }
            assert!(!expected.is_empty(), "{literal}");
            let has_ks = literal
                .bytes()
                .any(|b| matches!(b.to_ascii_lowercase(), b'k' | b's'));

            for (name, idx) in [("memory", &index), ("lazy", &lazy)] {
                let mut cands: std::collections::HashSet<_> = idx
                    .search_candidates_fold(literal.as_bytes())
                    .iter()
                    .map(key)
                    .collect();
                if has_ks {
                    cands.extend(idx.exotic_fold_lines().iter().map(key));
                }
                assert!(
                    expected.is_subset(&cands),
                    "{name}: fold candidates for {literal:?} miss {:?}",
                    expected.difference(&cands).collect::<Vec<_>>()
                );
            }
        }

        // The regex crate really does fold these (Unicode simple case folding),
        // and the ASCII windows alone do not reach them.
        assert!(
            regex::Regex::new("(?i)kelvin")
                .unwrap()
                .is_match("\u{212A}elvin")
        );
        assert!(
            regex::Regex::new("(?i)status")
                .unwrap()
                .is_match("\u{017F}tatus")
        );
        let ascii_only: std::collections::HashSet<_> = lazy
            .search_candidates_fold(b"kelvin")
            .iter()
            .map(key)
            .collect();
        assert!(
            !ascii_only.contains(&(0, 106)),
            "line 106 is the Kelvin-sign line"
        );

        // Exotic lines are exactly the two lines per file that carry the folds.
        assert_eq!(lazy.exotic_fold_lines().len(), 20);
        assert_eq!(index.exotic_fold_lines(), lazy.exotic_fold_lines());

        // A window absent in every casing is a definite miss.
        assert!(lazy.search_candidates_fold(b"zzqx_absent").is_empty());
        // Non-ASCII is not folded here.
        assert!(lazy.search_candidates_fold("straße".as_bytes()).is_empty());
    }

    use tempfile::TempDir;

    #[test]
    fn test_extract_trigrams() {
        let text = "hello";
        let trigrams = extract_trigrams(text);

        // "hello" → "hel", "ell", "llo"
        assert_eq!(trigrams.len(), 3);

        // Verify trigrams are unique
        let expected = vec![
            bytes_to_trigram(b"hel"),
            bytes_to_trigram(b"ell"),
            bytes_to_trigram(b"llo"),
        ];
        assert_eq!(trigrams, expected);
    }

    #[test]
    fn test_extract_trigrams_short() {
        assert_eq!(extract_trigrams("ab").len(), 0);
        assert_eq!(extract_trigrams("abc").len(), 1);
    }

    #[test]
    fn test_bytes_to_trigram() {
        let trigram1 = bytes_to_trigram(b"abc");
        let trigram2 = bytes_to_trigram(b"abc");
        let trigram3 = bytes_to_trigram(b"xyz");

        assert_eq!(trigram1, trigram2);
        assert_ne!(trigram1, trigram3);
    }

    #[test]
    fn test_trigram_roundtrip() {
        let original = b"foo";
        let trigram = bytes_to_trigram(original);
        let recovered = trigram_to_bytes(trigram);
        assert_eq!(original, &recovered);
    }

    #[test]
    fn test_extract_with_locations() {
        let text = "hello\nworld";
        let locs = extract_trigrams_with_locations(text, 0);

        // "hello\nworld" has 9 distinct trigrams:
        // line 1: "hel", "ell", "llo", "lo\n", "o\nw"
        // line 2: "\nwo", "wor", "orl", "rld"   (a trigram starting on '\n' is line 2)
        assert_eq!(locs.len(), 9);
        assert_eq!(locs.iter().filter(|(_, l)| l.line_no == 1).count(), 5);
        assert_eq!(locs.iter().filter(|(_, l)| l.line_no == 2).count(), 4);
        assert!(locs.iter().all(|(_, l)| l.file_id == 0));

        let line_of = |t: &[u8; 3]| {
            locs.iter()
                .find(|(tri, _)| *tri == bytes_to_trigram(t))
                .map(|(_, l)| l.line_no)
        };
        assert_eq!(line_of(b"hel"), Some(1));
        assert_eq!(line_of(b"o\nw"), Some(1));
        assert_eq!(line_of(b"\nwo"), Some(2));
        assert_eq!(line_of(b"wor"), Some(2));
    }

    #[test]
    fn test_extract_emits_one_posting_per_distinct_trigram_per_line() {
        // "aaaa" alone yields "aaa" twice at byte level; per line it must be once.
        let locs = extract_trigrams_with_locations("aaaa\naaaa", 3);
        // line 1: "aaa", "aa\n", "a\na"   line 2: "\naa", "aaa"
        assert_eq!(locs.len(), 5);
        let aaa = bytes_to_trigram(b"aaa");
        let aaa_lines: Vec<u32> = locs
            .iter()
            .filter(|(t, _)| *t == aaa)
            .map(|(_, l)| l.line_no)
            .collect();
        assert_eq!(aaa_lines, vec![1, 2]);

        // Every (trigram, file, line) key is unique and each line's run is sorted
        let mut keys: Vec<(Trigram, u32, u32)> = locs
            .iter()
            .map(|(t, l)| (*t, l.file_id, l.line_no))
            .collect();
        let n = keys.len();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), n, "duplicate (trigram, line) postings emitted");
        assert!(locs.iter().all(|(_, l)| l.file_id == 3));

        // Heavy indentation: 16 spaces → "   " appears 14 times per line, emitted once
        let indented = "                x\n                y\n";
        let sp = bytes_to_trigram(b"   ");
        let spaces = extract_trigrams_with_locations(indented, 0);
        assert_eq!(spaces.iter().filter(|(t, _)| *t == sp).count(), 2);
    }

    #[test]
    fn test_trigram_index_basic() {
        let mut index = TrigramIndex::new();

        let file_id = index.add_file(PathBuf::from("test.txt"));
        index.index_file(file_id, "hello world");
        index.finalize();

        // Search for "hello"
        let results = index.search("hello");
        assert!(!results.is_empty());

        // Search for "world"
        let results = index.search("world");
        assert!(!results.is_empty());

        // Search for "goodbye" (not in text)
        let results = index.search("goodbye");
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_multifile() {
        let mut index = TrigramIndex::new();

        let file1 = index.add_file(PathBuf::from("file1.txt"));
        let file2 = index.add_file(PathBuf::from("file2.txt"));

        index.index_file(file1, "extract_symbols is here");
        index.index_file(file2, "extract_symbols is also here");
        index.finalize();

        let results = index.search("extract_symbols");
        assert_eq!(results.len(), 2); // One result per file

        // Verify we got both files
        let file_ids: Vec<u32> = results.iter().map(|loc| loc.file_id).collect();
        assert!(file_ids.contains(&file1));
        assert!(file_ids.contains(&file2));
    }

    #[test]
    fn test_persistence_write_header() {
        let temp = TempDir::new().unwrap();
        let trigrams_path = temp.path().join("trigrams.bin");

        // Build and write index
        let mut index = TrigramIndex::new();
        let file1 = index.add_file(PathBuf::from("src/main.rs"));
        let file2 = index.add_file(PathBuf::from("src/lib.rs"));

        index.index_file(file1, "fn main() { println!(\"hello\"); }");
        index.index_file(
            file2,
            "pub fn hello() -> String { String::from(\"hello\") }",
        );
        index.finalize();
        let num_trigrams = index.trigram_count() as u64;

        index.write(&trigrams_path).unwrap();

        let bytes = std::fs::read(&trigrams_path).unwrap();
        assert!(bytes.len() > HEADER_SIZE);
        assert_eq!(&bytes[0..4], MAGIC);
        assert_eq!(read_u32(&bytes, 4), VERSION);
        assert_eq!(read_u64(&bytes, NUM_TRIGRAMS_OFFSET), num_trigrams);
        assert_eq!(read_u64(&bytes, NUM_FILES_OFFSET), 2);

        // paths_offset points just past header + directory + data, and the
        // paths section decodes to the files we added.
        let paths_offset = read_u64(&bytes, PATHS_OFFSET_OFFSET) as usize;
        assert!(paths_offset >= HEADER_SIZE + num_trigrams as usize * DIR_ENTRY_SIZE);
        assert!(paths_offset < bytes.len());
        let (len, consumed) = read_varint(&bytes[paths_offset..]).unwrap();
        let first = &bytes[paths_offset + consumed..paths_offset + consumed + len as usize];
        assert_eq!(first, b"src/main.rs");
    }

    fn loc(file_id: u32, line_no: u32) -> FileLocation {
        FileLocation::new(file_id, line_no)
    }

    /// `unwrap_err` without requiring `Debug` on the success type
    fn load_err(path: &Path) -> String {
        match TrigramIndex::load(path) {
            Ok(_) => panic!("load unexpectedly succeeded"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn test_intersect_two_empty() {
        let a = [loc(0, 1), loc(1, 2)];
        assert!(intersect_two(&a, &[]).is_empty());
        assert!(intersect_two(&[], &a).is_empty());
        assert!(intersect_two(&[], &[]).is_empty());
    }

    #[test]
    fn test_intersect_two_disjoint() {
        let a = [loc(0, 1), loc(0, 3), loc(2, 1)];
        let b = [loc(0, 2), loc(1, 1), loc(2, 2)];
        assert!(intersect_two(&a, &b).is_empty());
        assert!(intersect_two(&b, &a).is_empty());
    }

    #[test]
    fn test_intersect_two_duplicate_keys_in_b() {
        // Repeated keys in `b` are all consumed; result keeps `a`'s entry once
        let a = [loc(0, 1), loc(0, 2), loc(1, 1)];
        let b = [
            loc(0, 1),
            loc(0, 1),
            loc(0, 1),
            loc(1, 1),
            loc(1, 1),
            loc(1, 2),
        ];
        assert_eq!(intersect_two(&a, &b), vec![loc(0, 1), loc(1, 1)]);
    }

    #[test]
    fn test_intersect_two_gallop_path() {
        // 1 vs 200 entries forces the gallop branch (b.len() > 8 * a.len())
        let b: Vec<FileLocation> = (0u32..200).map(|i| loc(i / 10, i % 10 + 1)).collect();
        assert_eq!(intersect_two(&[loc(7, 4)], &b), vec![loc(7, 4)]);
        assert!(intersect_two(&[loc(7, 11)], &b).is_empty());
        assert!(intersect_two(&[loc(20, 1)], &b).is_empty());

        // A few shared keys spread through a large `b`
        let a = [loc(0, 1), loc(5, 5), loc(12, 3), loc(19, 10), loc(25, 1)];
        assert_eq!(
            intersect_two(&a, &b),
            vec![loc(0, 1), loc(5, 5), loc(12, 3), loc(19, 10)]
        );
    }

    #[test]
    fn test_intersect_two_all_equal() {
        let a: Vec<FileLocation> = (0u32..50).map(|i| loc(i, 1)).collect();
        let b: Vec<FileLocation> = (0u32..50).map(|i| loc(i, 1)).collect();
        assert_eq!(intersect_two(&a, &b), a);
    }

    /// Minimal LCG so the property test is deterministic without extra crates
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
    }

    fn random_sorted_list(rng: &mut Lcg, n: usize) -> Vec<FileLocation> {
        // Keys from a small range (40 files x 60 lines) to force heavy overlap
        let mut list: Vec<FileLocation> = (0..n)
            .map(|_| loc(rng.next() % 40, rng.next() % 60 + 1))
            .collect();
        list.sort_unstable();
        list.dedup();
        list
    }

    fn reference_intersection(lists: &[&[FileLocation]]) -> Vec<FileLocation> {
        use std::collections::HashSet;
        let mut keys: HashSet<(u32, u32)> = lists[0].iter().map(key).collect();
        for list in &lists[1..] {
            let set: HashSet<(u32, u32)> = list.iter().map(key).collect();
            keys.retain(|k| set.contains(k));
        }
        let mut out: Vec<FileLocation> = keys
            .iter()
            .map(|k| *lists[0].iter().find(|l| key(l) == *k).unwrap())
            .collect();
        out.sort_unstable();
        out
    }

    #[test]
    fn test_intersect_sorted_matches_reference() {
        let mut rng = Lcg(0x5eed);
        for _ in 0..5 {
            let lists: Vec<Vec<FileLocation>> = (0..3)
                .map(|_| random_sorted_list(&mut rng, 2_000))
                .collect();
            let mut refs: Vec<&[FileLocation]> = lists.iter().map(|l| l.as_slice()).collect();
            refs.sort_by_key(|l| l.len());

            let expected = reference_intersection(&refs);
            assert!(!expected.is_empty(), "test data should overlap");
            assert_eq!(intersect_sorted(&refs), expected);
        }
    }

    /// Build a 40-file index whose posting lists differ widely in size
    fn random_word_index(seed: u64) -> TrigramIndex {
        let mut rng = Lcg(seed);
        let mut index = TrigramIndex::new();
        for i in 0..40 {
            index.add_file(PathBuf::from(format!("f{i}.txt")));
        }
        let words = ["realm", "real", "alma", "lmn", "rea", "xyz", "ealm"];
        for file_id in 0..40u32 {
            let mut content = String::new();
            for _ in 0..60 {
                for _ in 0..4 {
                    content.push_str(words[(rng.next() % words.len() as u32) as usize]);
                    content.push(' ');
                }
                content.push('\n');
            }
            index.index_file(file_id, &content);
        }
        index.finalize();
        index
    }

    #[test]
    fn test_lazy_search_matches_in_memory() {
        let mut index = random_word_index(0xbeef);

        let temp = TempDir::new().unwrap();
        let path = temp.path().join("trigrams.bin");
        index.write(&path).unwrap();
        let lazy = TrigramIndex::load(&path).unwrap();

        assert_eq!(lazy.trigram_count(), index.trigram_count());
        assert_eq!(lazy.file_count(), 40);
        assert_eq!(lazy.get_file(39), Some(&PathBuf::from("f39.txt")));
        assert_eq!(lazy.get_file(40), None);

        for pattern in ["realm", "alma", "xyz", "lmn r", "ealm x", "nothing", "ab"] {
            assert_eq!(
                lazy.search(pattern),
                index.search(pattern),
                "pattern {pattern:?}"
            );
            assert_eq!(lazy.search_files(pattern), index.search_files(pattern));
        }

        let files = lazy.search_files("realm");
        assert!(!files.is_empty());
        assert!(files.windows(2).all(|w| w[0] < w[1]), "sorted, distinct");
    }

    #[test]
    fn test_find_entry_zero_copy_directory() {
        let mut index = random_word_index(0xf00d);
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("trigrams.bin");
        index.write(&path).unwrap();
        let lazy = TrigramIndex::load(&path).unwrap();

        // In-memory index has no directory to search
        assert!(index.find_entry(bytes_to_trigram(b"rea")).is_none());

        let rea = bytes_to_trigram(b"rea");
        let entry = lazy.find_entry(rea).expect("rea is indexed");
        assert_eq!(entry.trigram, rea);
        let expected_len = index.get_posting_list(rea).unwrap().len();
        let decoded = decompress_posting_list(
            lazy.mmap.as_ref().unwrap(),
            entry.data_offset,
            entry.compressed_size,
        )
        .unwrap();
        assert_eq!(decoded.len(), expected_len);
        assert_eq!(&decoded, index.get_posting_list(rea).unwrap());

        assert!(lazy.find_entry(bytes_to_trigram(b"zzz")).is_none());
        assert!(lazy.find_entry(0).is_none());
        assert!(lazy.find_entry(u32::MAX).is_none());
    }

    #[test]
    fn test_search_repeated_trigram_returns_line_once() {
        let mut index = TrigramIndex::new();
        let file_id = index.add_file(PathBuf::from("a.txt"));
        index.index_file(file_id, "x\naaaa\ny");
        index.finalize();

        let results = index.search("aaaa");
        assert_eq!(results, vec![loc(file_id, 2)]);

        let temp = TempDir::new().unwrap();
        let path = temp.path().join("trigrams.bin");
        index.write(&path).unwrap();
        let lazy = TrigramIndex::load(&path).unwrap();
        assert_eq!(lazy.search("aaaa"), results);
    }

    #[test]
    fn test_posting_list_block_encoding_roundtrip() {
        // Three file blocks; the middle one has a single line, the last is far away
        let locations = vec![
            loc(0, 1),
            loc(0, 7),
            loc(0, 300),
            loc(3, 2),
            loc(70_000, 1),
            loc(70_000, 2),
        ];
        let mut data = Vec::new();
        encode_posting_list(&locations, &mut data).unwrap();

        // Expected bytes: block(0): fd=0, n=3<<1, lines 1,6,293(2 B)
        //                 block(3): fd=3, n=1<<1, line 2
        //                 block(70000): fd=69997 (3 B), n=2<<1, lines 1,1
        assert_eq!(
            data,
            vec![0, 6, 1, 6, 0xA5, 0x02, 3, 2, 2, 0xED, 0xA2, 0x04, 4, 1, 1]
        );

        assert_eq!(
            decompress_posting_list(&data, 0, data.len() as u32).unwrap(),
            locations
        );

        let mut cursor = PostingCursor::new(&data).unwrap();
        assert_eq!(cursor.current(), Some(loc(0, 1)));
        assert_eq!(cursor.seek((0, 7)).unwrap(), Some(loc(0, 7)));
        // Target in a later file: skips the rest of block 0 and block 3
        assert_eq!(cursor.seek((4, 1)).unwrap(), Some(loc(70_000, 1)));
        assert_eq!(cursor.advance().unwrap(), Some(loc(70_000, 2)));
        assert_eq!(cursor.seek((70_000, 2)).unwrap(), Some(loc(70_000, 2)));
        assert_eq!(cursor.seek((70_001, 0)).unwrap(), None);
        assert_eq!(cursor.current(), None);

        // Seeking within the first block by line only
        let mut cursor = PostingCursor::new(&data).unwrap();
        assert_eq!(cursor.seek((0, 8)).unwrap(), Some(loc(0, 300)));
        assert_eq!(cursor.seek((3, 1)).unwrap(), Some(loc(3, 2)));
    }

    #[test]
    fn test_posting_list_empty_and_malformed() {
        let mut data = Vec::new();
        encode_posting_list(&[], &mut data).unwrap();
        assert!(data.is_empty());
        assert!(decompress_posting_list(&data, 0, 0).unwrap().is_empty());

        let empty = PostingCursor::new(&[]).unwrap();
        assert_eq!(empty.current(), None);

        // Truncated varint surfaces as an error rather than a panic
        assert!(PostingCursor::new(&[0x80]).is_err());
        // Block header present, line varint missing
        assert!(PostingCursor::new(&[0, 2]).is_err());
        // Out-of-bounds slice request
        assert!(decompress_posting_list(&[0, 2, 1], 0, 4).is_err());

        // Reserved enc=1 blocks are rejected, not misread
        let enc1 = [0u8, (1 << 1) | 1, 1];
        let err = match PostingCursor::new(&enc1) {
            Ok(_) => panic!("enc=1 block was accepted"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("unsupported block encoding"), "{err}");
        assert!(decompress_posting_list(&enc1, 0, 3).is_err());

        // A zero-length block is tolerated and skipped
        let with_empty_block = [0u8, 0, 2, 1 << 1, 5];
        assert_eq!(
            decompress_posting_list(&with_empty_block, 0, 5).unwrap(),
            vec![loc(2, 5)]
        );
    }

    #[test]
    fn test_v4_roundtrip_file_block_boundaries() {
        // "needle" sits on several lines of files 0, 2 and 4, on none of 1 and 3,
        // and file 5 has it only on its last line. Every block boundary shape:
        // consecutive files, gaps, single-line block.
        let mut index = TrigramIndex::new();
        for i in 0..6 {
            index.add_file(PathBuf::from(format!("dir/file{i}.rs")));
        }
        let with = "needle here\nnothing\nneedle again\n\nneedle\n";
        let without = "haystack\nonly\n";
        index.index_file(0, with);
        index.index_file(1, without);
        index.index_file(2, with);
        index.index_file(3, without);
        index.index_file(4, with);
        index.index_file(5, "haystack\nneedle");
        index.finalize();

        let expected = vec![
            loc(0, 1),
            loc(0, 3),
            loc(0, 5),
            loc(2, 1),
            loc(2, 3),
            loc(2, 5),
            loc(4, 1),
            loc(4, 3),
            loc(4, 5),
            loc(5, 2),
        ];
        assert_eq!(index.search("needle"), expected);

        let temp = TempDir::new().unwrap();
        let path = temp.path().join("trigrams.bin");
        index.write(&path).unwrap();
        let lazy = TrigramIndex::load(&path).unwrap();

        assert_eq!(lazy.search("needle"), expected);
        assert_eq!(lazy.search_files("needle"), vec![0, 2, 4, 5]);
        assert_eq!(
            lazy.search("haystack"),
            vec![loc(1, 1), loc(3, 1), loc(5, 1)]
        );
        assert_eq!(
            lazy.search("needle again"),
            vec![loc(0, 3), loc(2, 3), loc(4, 3)]
        );
        assert!(lazy.search("needle haystack").is_empty());
        assert_eq!(lazy.file_count(), 6);
        assert_eq!(lazy.get_file(5), Some(&PathBuf::from("dir/file5.rs")));
    }

    #[test]
    fn test_v4_roundtrip_empty_index() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("trigrams.bin");

        let mut index = TrigramIndex::new();
        index.finalize();
        index.write(&path).unwrap();

        let lazy = TrigramIndex::load(&path).unwrap();
        assert_eq!(lazy.trigram_count(), 0);
        assert_eq!(lazy.file_count(), 0);
        assert!(lazy.search("anything").is_empty());

        // Files but no indexable content (every file shorter than a trigram)
        let mut index = TrigramIndex::new();
        let f = index.add_file(PathBuf::from("tiny.txt"));
        index.index_file(f, "ab");
        index.finalize();
        index.write(&path).unwrap();
        let lazy = TrigramIndex::load(&path).unwrap();
        assert_eq!(lazy.trigram_count(), 0);
        assert_eq!(lazy.file_count(), 1);
        assert!(lazy.search("abc").is_empty());
    }

    #[test]
    fn test_batch_flush_streaming_merge_roundtrip() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("trigrams.bin");

        // Reference: same content through the in-memory path
        let reference = random_word_index(0xcafe);

        // Batch path: flush every 10 files so the k-way merge sees 4 partials
        let mut rng = Lcg(0xcafe);
        let mut index = TrigramIndex::new();
        index
            .enable_batch_flush(temp.path().join("partials"))
            .unwrap();
        for i in 0..40 {
            index.add_file(PathBuf::from(format!("f{i}.txt")));
        }
        let words = ["realm", "real", "alma", "lmn", "rea", "xyz", "ealm"];
        for file_id in 0..40u32 {
            let mut content = String::new();
            for _ in 0..60 {
                for _ in 0..4 {
                    content.push_str(words[(rng.next() % words.len() as u32) as usize]);
                    content.push(' ');
                }
                content.push('\n');
            }
            index.index_file(file_id, &content);
            if file_id % 10 == 9 {
                index.flush_batch().unwrap();
            }
        }
        index.finalize();
        index.write(&path).unwrap();

        let lazy = TrigramIndex::load(&path).unwrap();
        assert_eq!(lazy.trigram_count(), reference.trigram_count());
        assert_eq!(lazy.file_count(), 40);
        for pattern in ["realm", "alma", "xyz", "lmn r", "ealm x", "nothing"] {
            assert_eq!(
                lazy.search(pattern),
                reference.search(pattern),
                "pattern {pattern:?}"
            );
        }

        // Header bookkeeping from the two-pass streaming writer
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(read_u32(&bytes, 4), VERSION);
        assert_eq!(
            read_u64(&bytes, NUM_TRIGRAMS_OFFSET) as usize,
            reference.trigram_count()
        );
        let paths_offset = read_u64(&bytes, PATHS_OFFSET_OFFSET) as usize;
        let (len, consumed) = read_varint(&bytes[paths_offset..]).unwrap();
        assert_eq!(
            &bytes[paths_offset + consumed..paths_offset + consumed + len as usize],
            b"f0.txt"
        );
    }

    #[test]
    fn test_load_rejects_v3_version() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("trigrams.bin");

        // A V3 header: magic, version 3, num_trigrams, num_files (24 bytes)
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let err = load_err(&path);
        assert!(
            err.contains("Unsupported trigrams.bin version: 3 (expected 4)"),
            "{err}"
        );
        assert!(err.contains("re-index"), "{err}");

        // Wrong magic is a different error
        std::fs::write(&path, b"NOPE\x04\x00\x00\x00").unwrap();
        let err = load_err(&path);
        assert!(err.contains("wrong magic"), "{err}");
    }

    #[test]
    fn test_load_rejects_truncated_or_inconsistent_file() {
        let mut index = random_word_index(0xd00d);
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("trigrams.bin");
        index.write(&path).unwrap();
        let full = std::fs::read(&path).unwrap();
        assert!(TrigramIndex::load(&path).is_ok());

        // Cut in the middle of the data section: paths_offset now lies past EOF
        std::fs::write(&path, &full[..full.len() / 2]).unwrap();
        let err = load_err(&path);
        assert!(err.contains("out of bounds"), "{err}");

        // Cut inside the header
        std::fs::write(&path, &full[..10]).unwrap();
        let err = load_err(&path);
        assert!(err.contains("too small"), "{err}");

        // Cut inside the paths section
        std::fs::write(&path, &full[..full.len() - 3]).unwrap();
        let err = load_err(&path);
        assert!(err.contains("Truncated"), "{err}");

        // Directory claims more entries than fit before paths_offset
        let mut bad = full.clone();
        bad[NUM_TRIGRAMS_OFFSET..NUM_TRIGRAMS_OFFSET + 8]
            .copy_from_slice(&(u64::MAX / 32).to_le_bytes());
        std::fs::write(&path, &bad).unwrap();
        let err = load_err(&path);
        assert!(err.contains("out of bounds"), "{err}");
    }

    #[test]
    fn test_posting_list_cap_enforced() {
        let cap: usize = 10;
        let content = "aaa \n".repeat(200);
        let mut index = TrigramIndex::new();
        index.set_max_posting_list_entries(cap);
        let file_id = index.add_file(PathBuf::from("dense.txt"));
        index.index_file(file_id, &content);
        index.finalize();
        let aaa = bytes_to_trigram(b"aaa");
        let list = index
            .get_posting_list(aaa)
            .expect("aaa trigram should exist");
        assert!(list.len() <= cap, "cap exceeded: {} > {}", list.len(), cap);
    }

    #[test]
    fn test_posting_list_cap_zero_means_unlimited() {
        let repetitions = 50;
        // One line per repetition: postings are per line, not per byte
        let content = "aaa \n".repeat(repetitions);
        let mut index = TrigramIndex::new();
        index.set_max_posting_list_entries(0);
        let file_id = index.add_file(PathBuf::from("dense.txt"));
        index.index_file(file_id, &content);
        index.finalize();
        let aaa = bytes_to_trigram(b"aaa");
        let list = index
            .get_posting_list(aaa)
            .expect("aaa trigram should exist");
        assert!(
            list.len() >= repetitions,
            "expected >= {} entries, got {}",
            repetitions,
            list.len()
        );
    }
}
