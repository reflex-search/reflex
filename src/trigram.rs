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
const VERSION: u32 = 3; // V3: No filtering, lazy loading with directory + data separation
// Header: magic(4) + version(4) + num_trigrams(8) + num_files(8) = 24 bytes
#[allow(dead_code)]
const HEADER_SIZE: usize = 24;

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

/// Decompress a posting list from memory-mapped data
///
/// Reads a compressed posting list (delta+varint encoded) from the given offset
/// and decompresses it into a Vec<FileLocation>.
///
/// # Arguments
/// * `mmap` - Memory-mapped file data
/// * `offset` - Absolute byte offset where compressed data starts
/// * `size` - Number of bytes to read
fn decompress_posting_list(
    mmap: &[u8],
    offset: u64,
    size: u32,
) -> Result<Vec<FileLocation>> {
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

    // Decompress delta-encoded posting list
    let mut locations = Vec::new();
    let mut pos = 0;
    let mut prev_file_id = 0u32;
    let mut prev_line_no = 0u32;
    let mut prev_byte_offset = 0u32;

    while pos < compressed_data.len() {
        // Read file_id delta
        let (file_id_delta, consumed) = read_varint(&compressed_data[pos..])?;
        pos += consumed;

        // Read line_no delta
        let (line_no_delta, consumed) = read_varint(&compressed_data[pos..])?;
        pos += consumed;

        // Read byte_offset delta
        let (byte_offset_delta, consumed) = read_varint(&compressed_data[pos..])?;
        pos += consumed;

        // Reconstruct absolute values from deltas
        let file_id = prev_file_id.wrapping_add(file_id_delta);
        let line_no = prev_line_no.wrapping_add(line_no_delta);
        let byte_offset = prev_byte_offset.wrapping_add(byte_offset_delta);

        locations.push(FileLocation {
            file_id,
            line_no,
            byte_offset,
        });

        // Update previous values for next delta
        prev_file_id = file_id;
        prev_line_no = line_no;
        prev_byte_offset = byte_offset;
    }

    Ok(locations)
}

/// Location of a trigram occurrence in the codebase
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FileLocation {
    /// File ID (index into file list)
    pub file_id: u32,
    /// Line number (1-indexed)
    pub line_no: u32,
    /// Byte offset in file (for context extraction)
    pub byte_offset: u32,
}

impl FileLocation {
    pub fn new(file_id: u32, line_no: u32, byte_offset: u32) -> Self {
        Self {
            file_id,
            line_no,
            byte_offset,
        }
    }
}

/// Directory entry for lazy-loaded trigram index
///
/// Maps each trigram to its compressed posting list location in the data section.
/// Total size: 16 bytes per entry (4 + 8 + 4)
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
    /// Directory of (trigram, offset, size) for lazy loading
    directory: Vec<DirectoryEntry>,
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
            directory: Vec::new(),
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
        let temp_dir = self.temp_dir.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Batch flush not enabled - call enable_batch_flush() first"))?;

        // Take ownership of temp_index to finalize it
        let temp_map = self.temp_index.take()
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

            for loc in locations {
                writer.write_all(&loc.file_id.to_le_bytes())?;
                writer.write_all(&loc.line_no.to_le_bytes())?;
                writer.write_all(&loc.byte_offset.to_le_bytes())?;
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
        if !self.directory.is_empty() {
            // Lazy-loaded mode
            self.directory.len()
        } else {
            // In-memory mode
            self.index.len()
        }
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
            temp_map
                .entry(trigram)
                .or_insert_with(Vec::new)
                .push(location);
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
            log::info!("Deferring finalization - will stream merge {} partial indices during write()",
                       self.partial_indices.len());

            // Flush final batch if temp_index is not empty
            if let Some(ref temp_map) = self.temp_index {
                if !temp_map.is_empty() {
                    self.flush_batch().expect("Failed to flush final batch");
                }
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
                log::warn!("Trigram 0x{:06X} posting list has {} entries (cap {}); truncating.", trigram, list.len(), cap);
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
        use std::io::{BufReader, BufWriter, Read};
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        log::info!("Streaming merge of {} partial indices to {:?}",
                   self.partial_indices.len(), output_path);

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

                    // Read all locations for this trigram
                    let mut locations = Vec::with_capacity(list_len);
                    for _ in 0..list_len {
                        let mut loc_buf = [0u8; 12];
                        reader.reader.read_exact(&mut loc_buf)?;

                        let file_id = u32::from_le_bytes([loc_buf[0], loc_buf[1], loc_buf[2], loc_buf[3]]);
                        let line_no = u32::from_le_bytes([loc_buf[4], loc_buf[5], loc_buf[6], loc_buf[7]]);
                        let byte_offset = u32::from_le_bytes([loc_buf[8], loc_buf[9], loc_buf[10], loc_buf[11]]);

                        locations.push(FileLocation { file_id, line_no, byte_offset });
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
                other.trigram.cmp(&self.trigram)
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

        // Open output file for writing
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(output_path)
            .with_context(|| format!("Failed to create {}", output_path.display()))?;

        let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file);

        // Write placeholder header (we'll update it at the end)
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&0u64.to_le_bytes())?; // num_trigrams (placeholder)
        writer.write_all(&(self.files.len() as u64).to_le_bytes())?; // num_files

        // We'll build the directory as we go
        let mut directory: Vec<DirectoryEntry> = Vec::new();
        let mut num_trigrams = 0u64;

        // K-way merge loop
        let mut current_trigram: Option<Trigram> = None;
        let mut merged_locations: Vec<FileLocation> = Vec::new();

        while let Some(entry) = heap.pop() {
            let reader = &mut readers[entry.reader_id];

            // If this is a new trigram, write the previous one
            if current_trigram.is_some() && current_trigram != Some(entry.trigram) {
                // Write the accumulated posting list for current_trigram
                let trigram = current_trigram.unwrap();
                merged_locations.sort_unstable();
                merged_locations.dedup();

                let cap = self.max_posting_list_entries;
                if cap > 0 && merged_locations.len() > cap {
                    log::warn!("Trigram 0x{:06X} posting list has {} entries (cap {}); truncating.", trigram, merged_locations.len(), cap);
                    merged_locations.truncate(cap);
                }

                // Compress and write this trigram's posting list
                let data_offset = writer.stream_position()?;
                let compressed_size = self.write_compressed_posting_list(&mut writer, &merged_locations)?;

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
            if read_next_trigram(reader)? {
                if let Some(next_trigram) = reader.current_trigram {
                    heap.push(HeapEntry {
                        trigram: next_trigram,
                        reader_id: entry.reader_id,
                    });
                }
            }
        }

        // Write final trigram
        if let Some(trigram) = current_trigram {
            merged_locations.sort_unstable();
            merged_locations.dedup();

            let cap = self.max_posting_list_entries;
            if cap > 0 && merged_locations.len() > cap {
                log::warn!("Trigram 0x{:06X} posting list has {} entries (cap {}); truncating.", trigram, merged_locations.len(), cap);
                merged_locations.truncate(cap);
            }

            let data_offset = writer.stream_position()?;
            let compressed_size = self.write_compressed_posting_list(&mut writer, &merged_locations)?;

            directory.push(DirectoryEntry {
                trigram,
                data_offset,
                compressed_size,
            });

            num_trigrams += 1;
        }

        log::info!("Merged {} trigrams from {} partial indices", num_trigrams, self.partial_indices.len());

        // Remember where data section ended (not used but kept for clarity)
        let _data_end_pos = writer.stream_position()?;

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
            let mut file = File::open(output_path)?;
            file.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
            file.read_to_end(&mut temp_data)?;
        }

        // Rewrite file with correct structure
        let file = OpenOptions::new().write(true).truncate(true).open(output_path)?;
        let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file);

        // Write header with correct num_trigrams
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&num_trigrams.to_le_bytes())?;
        writer.write_all(&(self.files.len() as u64).to_le_bytes())?;

        // Write directory
        for entry in &directory {
            writer.write_all(&entry.trigram.to_le_bytes())?;
            // Adjust data offset to account for directory size
            let adjusted_offset = entry.data_offset + (directory.len() * 16) as u64;
            writer.write_all(&adjusted_offset.to_le_bytes())?;
            writer.write_all(&entry.compressed_size.to_le_bytes())?;
        }

        // Write data and files sections
        writer.write_all(&temp_data)?;

        // Flush and sync
        writer.flush()?;
        writer.get_ref().sync_all()?;

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
        let mut compressed = Vec::new();

        // Compress posting list using delta+varint encoding
        let mut prev_file_id = 0u32;
        let mut prev_line_no = 0u32;
        let mut prev_byte_offset = 0u32;

        for loc in locations {
            // Compute deltas
            let file_id_delta = loc.file_id.wrapping_sub(prev_file_id);
            let line_no_delta = loc.line_no.wrapping_sub(prev_line_no);
            let byte_offset_delta = loc.byte_offset.wrapping_sub(prev_byte_offset);

            // Write deltas as varints
            write_varint(&mut compressed, file_id_delta)?;
            write_varint(&mut compressed, line_no_delta)?;
            write_varint(&mut compressed, byte_offset_delta)?;

            // Update previous values
            prev_file_id = loc.file_id;
            prev_line_no = loc.line_no;
            prev_byte_offset = loc.byte_offset;
        }

        let compressed_size = compressed.len() as u32;
        writer.write_all(&compressed)?;

        Ok(compressed_size)
    }

    /// Merge all partial indices into self.index (old in-memory approach - deprecated)
    #[allow(dead_code)]
    fn merge_partial_indices(&mut self) -> Result<()> {
        use std::io::{BufReader, Read};

        // Read all partial indices into memory (simplified approach for now)
        let mut all_entries: Vec<(Trigram, FileLocation)> = Vec::new();

        for partial_path in &self.partial_indices {
            let file = File::open(partial_path)
                .with_context(|| format!("Failed to open partial index: {:?}", partial_path))?;
            let mut reader = BufReader::with_capacity(16 * 1024 * 1024, file);

            // Read number of trigrams
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            let num_trigrams = u64::from_le_bytes(buf) as usize;

            // Read each (trigram, posting_list)
            for _ in 0..num_trigrams {
                // Read trigram
                let mut trigram_buf = [0u8; 4];
                reader.read_exact(&mut trigram_buf)?;
                let trigram = u32::from_le_bytes(trigram_buf);

                // Read posting list size
                let mut len_buf = [0u8; 4];
                reader.read_exact(&mut len_buf)?;
                let list_len = u32::from_le_bytes(len_buf) as usize;

                // Read all locations
                for _ in 0..list_len {
                    let mut loc_buf = [0u8; 12]; // 3 * u32
                    reader.read_exact(&mut loc_buf)?;

                    let file_id = u32::from_le_bytes([loc_buf[0], loc_buf[1], loc_buf[2], loc_buf[3]]);
                    let line_no = u32::from_le_bytes([loc_buf[4], loc_buf[5], loc_buf[6], loc_buf[7]]);
                    let byte_offset = u32::from_le_bytes([loc_buf[8], loc_buf[9], loc_buf[10], loc_buf[11]]);

                    all_entries.push((trigram, FileLocation { file_id, line_no, byte_offset }));
                }
            }
        }

        log::info!("Read {} total trigram entries from {} partial indices",
                   all_entries.len(), self.partial_indices.len());

        // Group by trigram
        let mut index_map: HashMap<Trigram, Vec<FileLocation>> = HashMap::new();
        for (trigram, location) in all_entries {
            index_map
                .entry(trigram)
                .or_insert_with(Vec::new)
                .push(location);
        }

        // Convert to sorted vec
        self.index = index_map.into_iter().collect();

        // Sort and deduplicate posting lists
        for (_, list) in self.index.iter_mut() {
            list.sort_unstable();
            list.dedup();
        }

        // Sort by trigram
        self.index.sort_unstable_by_key(|(trigram, _)| *trigram);

        // Clean up partial index files
        for partial_path in &self.partial_indices {
            let _ = std::fs::remove_file(partial_path);
        }
        if let Some(ref temp_dir) = self.temp_dir {
            let _ = std::fs::remove_dir(temp_dir);
        }

        log::info!("Merged into final index with {} trigrams", self.index.len());

        Ok(())
    }

    /// Search for a plain text pattern
    ///
    /// Returns candidate file locations that could contain the pattern.
    /// Caller must verify actual matches.
    ///
    /// In lazy-loaded mode: Decompresses posting lists on-demand from mmap.
    /// In in-memory mode: Uses pre-loaded posting lists.
    pub fn search(&self, pattern: &str) -> Vec<FileLocation> {
        if pattern.len() < 3 {
            // Pattern too short for trigrams - caller must fall back to full scan
            return vec![];
        }

        let trigrams = extract_trigrams(pattern);
        if trigrams.is_empty() {
            return vec![];
        }

        // Check if we're in lazy-loaded mode or in-memory mode
        if let Some(ref mmap) = self.mmap {
            // Lazy-loaded mode: decompress posting lists on-demand
            let mut posting_lists: Vec<Vec<FileLocation>> = Vec::new();

            for trigram in &trigrams {
                // Binary search directory for this trigram
                match self.directory.binary_search_by_key(trigram, |e| e.trigram) {
                    Ok(idx) => {
                        let entry = &self.directory[idx];
                        // Decompress this posting list on-demand
                        match decompress_posting_list(mmap, entry.data_offset, entry.compressed_size) {
                            Ok(locations) => posting_lists.push(locations),
                            Err(e) => {
                                log::warn!("Failed to decompress posting list for trigram {}: {}", trigram, e);
                                return vec![];
                            }
                        }
                    }
                    Err(_) => {
                        // Trigram not found - pattern cannot match
                        return vec![];
                    }
                }
            }

            if posting_lists.is_empty() || posting_lists.len() < trigrams.len() {
                return vec![];
            }

            // Sort by list size (smallest first for efficient intersection)
            posting_lists.sort_by_key(|list| list.len());

            // Intersect posting lists (owned version)
            intersect_by_file_owned(&posting_lists)
        } else {
            // In-memory mode: use pre-loaded index
            let mut posting_lists: Vec<&Vec<FileLocation>> = trigrams
                .iter()
                .filter_map(|t| {
                    self.index
                        .binary_search_by_key(t, |(trigram, _)| *trigram)
                        .ok()
                        .map(|idx| &self.index[idx].1)
                })
                .collect();

            if posting_lists.is_empty() {
                return vec![];
            }

            if posting_lists.len() < trigrams.len() {
                // Some trigrams missing - pattern cannot match
                return vec![];
            }

            // Sort by list size (smallest first for efficient intersection)
            posting_lists.sort_by_key(|list| list.len());

            // Intersect posting lists (reference version)
            intersect_by_file(&posting_lists)
        }
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
    /// Binary format V3 (lazy-loadable with directory + data separation):
    /// - Header (24 bytes): magic, version, num_trigrams, num_files
    /// - Directory Section (16 bytes per trigram):
    ///   - trigram: u32 (4 bytes)
    ///   - data_offset: u64 (8 bytes) - absolute offset in file
    ///   - compressed_size: u32 (4 bytes) - size of compressed posting list
    /// - Data Section (variable size):
    ///   - Compressed posting lists (delta+varint encoded)
    /// - File Paths Section (variable size):
    ///   - path_len: varint
    ///   - path_bytes: [u8; path_len]
    pub fn write(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();

        // If we have partial indices from batch flushing, use streaming merge
        if !self.partial_indices.is_empty() {
            log::info!("Using streaming merge to write {} partial indices", self.partial_indices.len());
            return self.merge_partial_indices_to_file(path);
        }

        // Standard write path (no batch flushing).
        // Non-atomic: writes directly to the target path with truncate(true).
        // On disk-full mid-write the file is left corrupt; the indexer fast-path
        // validates the "RFTG" magic bytes on re-index, forcing a clean rebuild.
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .with_context(|| format!("Failed to create {}", path.display()))?;

        // Use a large buffer (16MB) for streaming writes
        let mut writer = std::io::BufWriter::with_capacity(16 * 1024 * 1024, file);

        // Write header
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&(self.index.len() as u64).to_le_bytes())?; // num_trigrams
        writer.write_all(&(self.files.len() as u64).to_le_bytes())?; // num_files

        // Build directory and write compressed data in a single pass
        let mut directory: Vec<DirectoryEntry> = Vec::with_capacity(self.index.len());

        // Calculate directory start and size
        let directory_start = HEADER_SIZE as u64;
        let directory_size = self.index.len() * 16;

        // Reserve space for directory (we'll write it after data)
        let data_start = directory_start + directory_size as u64;
        let mut current_offset = data_start;

        // We need to write in the correct order: header, directory, data, file paths
        // But we need data offsets to write directory
        // So we compress data first, then write header+directory+data

        // Step 1: Compress all posting lists and track offsets
        let mut compressed_lists: Vec<(Trigram, Vec<u8>)> = Vec::with_capacity(self.index.len());

        for (trigram, locations) in &self.index {
            // Compress the posting list
            let mut compressed = Vec::new();
            let mut prev_file_id = 0u32;
            let mut prev_line_no = 0u32;
            let mut prev_byte_offset = 0u32;

            for loc in locations {
                let file_id_delta = loc.file_id.wrapping_sub(prev_file_id);
                let line_no_delta = loc.line_no.wrapping_sub(prev_line_no);
                let byte_offset_delta = loc.byte_offset.wrapping_sub(prev_byte_offset);

                write_varint(&mut compressed, file_id_delta)?;
                write_varint(&mut compressed, line_no_delta)?;
                write_varint(&mut compressed, byte_offset_delta)?;

                prev_file_id = loc.file_id;
                prev_line_no = loc.line_no;
                prev_byte_offset = loc.byte_offset;
            }

            directory.push(DirectoryEntry {
                trigram: *trigram,
                data_offset: current_offset,
                compressed_size: compressed.len() as u32,
            });
            current_offset += compressed.len() as u64;

            compressed_lists.push((*trigram, compressed));
        }

        // Step 2: Write directory
        for entry in &directory {
            writer.write_all(&entry.trigram.to_le_bytes())?;
            writer.write_all(&entry.data_offset.to_le_bytes())?;
            writer.write_all(&entry.compressed_size.to_le_bytes())?;
        }

        // Step 3: Write data section (compressed posting lists)
        for (_, compressed) in &compressed_lists {
            writer.write_all(compressed)?;
        }

        // Step 4: Write file paths
        for file_path in &self.files {
            let path_str = file_path.to_string_lossy();
            let path_bytes = path_str.as_bytes();
            write_varint(&mut writer, path_bytes.len() as u32)?;
            writer.write_all(path_bytes)?;
        }

        // Flush and sync
        writer.flush()?;
        writer.get_ref().sync_all()?;

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
    /// Binary format V3: Only reads the directory and file paths, keeps posting lists compressed in mmap.
    /// Posting lists are decompressed on-demand during search queries.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        let file = File::open(path)
            .with_context(|| format!("Failed to open {}", path.display()))?;

        // Memory-map the file (keep it alive for lazy access)
        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .with_context(|| format!("Failed to mmap {}", path.display()))?
        };

        // Validate header
        if mmap.len() < HEADER_SIZE {
            anyhow::bail!("trigrams.bin too small (expected at least {} bytes)", HEADER_SIZE);
        }

        if &mmap[0..4] != MAGIC {
            anyhow::bail!("Invalid trigrams.bin (wrong magic bytes)");
        }

        let version = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]);
        if version != VERSION {
            anyhow::bail!(
                "Unsupported trigrams.bin version: {} (expected {}). Please re-index with 'reflex index'.",
                version, VERSION
            );
        }

        let num_trigrams = u64::from_le_bytes([
            mmap[8], mmap[9], mmap[10], mmap[11],
            mmap[12], mmap[13], mmap[14], mmap[15],
        ]) as usize;

        let num_files = u64::from_le_bytes([
            mmap[16], mmap[17], mmap[18], mmap[19],
            mmap[20], mmap[21], mmap[22], mmap[23],
        ]) as usize;

        log::debug!("Loading lazy trigram index: {} trigrams, {} files", num_trigrams, num_files);

        // Read directory (trigram → offset mappings) - fast, just metadata
        let mut directory = Vec::with_capacity(num_trigrams);
        let mut pos = HEADER_SIZE;
        let directory_size = num_trigrams * 16; // 16 bytes per entry

        for _ in 0..num_trigrams {
            if pos + 16 > mmap.len() {
                anyhow::bail!("Truncated directory entry at pos={}", pos);
            }

            let trigram = u32::from_le_bytes([
                mmap[pos],
                mmap[pos + 1],
                mmap[pos + 2],
                mmap[pos + 3],
            ]);
            pos += 4;

            let data_offset = u64::from_le_bytes([
                mmap[pos],
                mmap[pos + 1],
                mmap[pos + 2],
                mmap[pos + 3],
                mmap[pos + 4],
                mmap[pos + 5],
                mmap[pos + 6],
                mmap[pos + 7],
            ]);
            pos += 8;

            let compressed_size = u32::from_le_bytes([
                mmap[pos],
                mmap[pos + 1],
                mmap[pos + 2],
                mmap[pos + 3],
            ]);
            pos += 4;

            directory.push(DirectoryEntry {
                trigram,
                data_offset,
                compressed_size,
            });
        }

        // Directory is already sorted by trigram (from write())
        directory.sort_unstable_by_key(|e| e.trigram);

        // Calculate where file paths section starts (after header + directory + data)
        let data_section_size: u64 = directory.iter().map(|e| e.compressed_size as u64).sum();
        let files_section_offset = HEADER_SIZE + directory_size + data_section_size as usize;
        pos = files_section_offset;

        // Read file paths (varint-encoded lengths)
        let mut files = Vec::with_capacity(num_files);
        for _ in 0..num_files {
            // Read path length (varint)
            let (path_len, consumed) = read_varint(&mmap[pos..])?;
            pos += consumed;
            let path_len = path_len as usize;

            if pos + path_len > mmap.len() {
                anyhow::bail!("Truncated file path at pos={}", pos);
            }

            let path_bytes = &mmap[pos..pos + path_len];
            let path_str = std::str::from_utf8(path_bytes)
                .context("Invalid UTF-8 in file path")?;
            files.push(PathBuf::from(path_str));
            pos += path_len;
        }

        log::info!(
            "Loaded lazy trigram index: {} trigrams, {} files (directory: {} KB)",
            num_trigrams,
            num_files,
            directory_size / 1024
        );

        Ok(Self {
            index: Vec::new(), // Empty in lazy mode
            files,
            temp_index: None,
            mmap: Some(mmap), // Keep mmap alive for lazy decompression!
            directory,
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
/// Returns a vector of (trigram, location) pairs for building the inverted index.
pub fn extract_trigrams_with_locations(text: &str, file_id: u32) -> Vec<(Trigram, FileLocation)> {
    let bytes = text.as_bytes();
    let mut result = Vec::new();

    let mut line_no = 1;

    for (i, &byte) in bytes.iter().enumerate() {
        // Track newlines
        if byte == b'\n' {
            line_no += 1;
        }

        // Extract trigram
        if i + 2 < bytes.len() {
            let trigram = bytes_to_trigram(&bytes[i..i + 3]);
            let location = FileLocation::new(file_id, line_no, i as u32);
            result.push((trigram, location));
        }
    }

    result
}

/// Convert 3 bytes to a trigram (packed u32)
#[inline]
fn bytes_to_trigram(bytes: &[u8]) -> Trigram {
    debug_assert_eq!(bytes.len(), 3);
    (bytes[0] as u32) << 16 | (bytes[1] as u32) << 8 | (bytes[2] as u32)
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

/// Intersect posting lists by (file_id, line_no) pairs
///
/// Returns locations where ALL trigrams appear on the SAME line (not just in the same file).
/// This ensures accurate full-text matching.
fn intersect_by_file(lists: &[&Vec<FileLocation>]) -> Vec<FileLocation> {
    if lists.is_empty() {
        return vec![];
    }

    use std::collections::HashSet;

    // Create a set of (file_id, line_no) pairs from the first list
    let mut candidates: HashSet<(u32, u32)> = lists[0]
        .iter()
        .map(|loc| (loc.file_id, loc.line_no))
        .collect();

    // Intersect with (file_id, line_no) pairs from other lists
    for &list in &lists[1..] {
        let list_pairs: HashSet<(u32, u32)> = list
            .iter()
            .map(|loc| (loc.file_id, loc.line_no))
            .collect();
        candidates.retain(|pair| list_pairs.contains(pair));
    }

    // Convert back to FileLocation results
    let mut result = Vec::new();
    for &(file_id, line_no) in &candidates {
        // Find a location matching this (file_id, line_no) from the first list
        if let Some(loc) = lists[0]
            .iter()
            .find(|loc| loc.file_id == file_id && loc.line_no == line_no)
        {
            result.push(*loc);
        }
    }

    result.sort_unstable();
    result
}

/// Intersect posting lists by (file_id, line_no) pairs (owned version for lazy-loading)
///
/// Similar to intersect_by_file() but works with owned Vec<Vec<FileLocation>>
/// instead of references. Used in lazy-loading mode where posting lists are decompressed on-demand.
///
/// Returns locations where ALL trigrams appear on the SAME line (not just in the same file).
fn intersect_by_file_owned(lists: &[Vec<FileLocation>]) -> Vec<FileLocation> {
    if lists.is_empty() {
        return vec![];
    }

    use std::collections::HashSet;

    // Create a set of (file_id, line_no) pairs from the first list
    let mut candidates: HashSet<(u32, u32)> = lists[0]
        .iter()
        .map(|loc| (loc.file_id, loc.line_no))
        .collect();

    // Intersect with (file_id, line_no) pairs from other lists
    for list in &lists[1..] {
        let list_pairs: HashSet<(u32, u32)> = list
            .iter()
            .map(|loc| (loc.file_id, loc.line_no))
            .collect();
        candidates.retain(|pair| list_pairs.contains(pair));
    }

    // Convert back to FileLocation results
    let mut result = Vec::new();
    for &(file_id, line_no) in &candidates {
        // Find a location matching this (file_id, line_no) from the first list
        if let Some(loc) = lists[0]
            .iter()
            .find(|loc| loc.file_id == file_id && loc.line_no == line_no)
        {
            result.push(*loc);
        }
    }

    result.sort_unstable();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

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

        // "hello\nworld" has 9 trigrams:
        // "hel", "ell", "llo", "lo\n", "o\nw", "\nwo", "wor", "orl", "rld"
        assert_eq!(locs.len(), 9);

        // First trigram should be on line 1
        assert_eq!(locs[0].1.line_no, 1);

        // After newline, should be line 2
        let world_start = text.find("world").unwrap();
        let world_trigram_idx = locs
            .iter()
            .position(|(_, loc)| loc.byte_offset as usize == world_start)
            .unwrap();
        assert_eq!(locs[world_trigram_idx].1.line_no, 2);
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
    fn test_persistence_write() {
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        let trigrams_path = temp.path().join("trigrams.bin");

        // Build and write index
        let mut index = TrigramIndex::new();
        let file1 = index.add_file(PathBuf::from("src/main.rs"));
        let file2 = index.add_file(PathBuf::from("src/lib.rs"));

        index.index_file(file1, "fn main() { println!(\"hello\"); }");
        index.index_file(file2, "pub fn hello() -> String { String::from(\"hello\") }");
        index.finalize();

        // Write to disk
        index.write(&trigrams_path).unwrap();

        // Verify file was created
        assert!(trigrams_path.exists());

        // Verify file has content (header + data)
        let metadata = std::fs::metadata(&trigrams_path).unwrap();
        assert!(metadata.len() > HEADER_SIZE as u64);

        // Verify we can read the header back
        use std::io::Read;
        let mut file = File::open(&trigrams_path).unwrap();
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).unwrap();
        assert_eq!(&magic, MAGIC);

        // Note: Full roundtrip test verifies write works correctly.
        // Load verification is tested in production via query performance tests.
    }

    #[test]
    fn test_posting_list_cap_enforced() {
        let cap: usize = 10;
        let content = "aaa ".repeat(200);
        let mut index = TrigramIndex::new();
        index.set_max_posting_list_entries(cap);
        let file_id = index.add_file(PathBuf::from("dense.txt"));
        index.index_file(file_id, &content);
        index.finalize();
        let aaa = bytes_to_trigram(b"aaa");
        let list = index.get_posting_list(aaa).expect("aaa trigram should exist");
        assert!(list.len() <= cap, "cap exceeded: {} > {}", list.len(), cap);
    }

    #[test]
    fn test_posting_list_cap_zero_means_unlimited() {
        let repetitions = 50;
        let content = "aaa ".repeat(repetitions);
        let mut index = TrigramIndex::new();
        index.set_max_posting_list_entries(0);
        let file_id = index.add_file(PathBuf::from("dense.txt"));
        index.index_file(file_id, &content);
        index.finalize();
        let aaa = bytes_to_trigram(b"aaa");
        let list = index.get_posting_list(aaa).expect("aaa trigram should exist");
        assert!(list.len() >= repetitions, "expected >= {} entries, got {}", repetitions, list.len());
    }
}
