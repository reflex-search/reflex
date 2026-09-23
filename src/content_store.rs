//! Content store for memory-mapped file access
//!
//! This module stores the full contents of all indexed files in a single
//! memory-mapped file. This enables zero-copy access to file contents for:
//! - Verifying trigram matches
//! - Extracting context around matches
//! - Fast content retrieval without disk I/O
//!
//! # Binary Format (content.bin)
//!
//! ```text
//! Header (32 bytes):
//!   magic: "RFCT" (4 bytes)
//!   version: 1 (u32)
//!   num_files: N (u64)
//!   index_offset: offset to file index (u64)
//!   reserved: 8 bytes
//!
//! File Contents (variable):
//!   [Concatenated file contents]
//!
//! File Index (at index_offset), version 2:
//!   Entry table, num_files × 28 bytes, addressable by file_id:
//!     offset: u64   (byte offset of the content, relative to the header end)
//!     length: u64   (content size in bytes)
//!     path_pos: u64 (absolute file position of the path bytes)
//!     path_len: u32
//!   Path blob: the UTF-8 paths, concatenated, in file_id order
//! ```
//!
//! Version 1 stored `(path_len, path, offset, length)` per file, which forced the
//! reader to decode the whole index into a `Vec` on open — O(files) work that
//! was the floor of every CLI query on a 24k-file checkout (12 ms). An entry is
//! now read from the mmap at `index_offset + file_id * 28`, so open is O(1) and
//! a query touches only the entries it verifies.

use anyhow::{Context, Result};
use memmap2::Mmap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"RFCT";
const VERSION: u32 = 2;
/// Bytes per file-index entry: offset u64 + length u64 + path_pos u64 + path_len u32.
const ENTRY_SIZE: usize = 28;
const HEADER_SIZE: usize = 32; // 4 (magic) + 4 (version) + 8 (num_files) + 8 (index_offset) + 8 (reserved)

/// Metadata for a file in the content store
#[derive(Debug, Clone)]
pub struct FileEntry {
    /// File path
    pub path: PathBuf,
    /// Byte offset in content.bin where this file's content starts
    pub offset: u64,
    /// Length of this file's content in bytes
    pub length: u64,
}

/// Writer for building content.bin
///
/// Supports two modes:
/// 1. **Streaming mode** (init() called): Writes file contents to disk incrementally to avoid RAM buildup
/// 2. **In-memory mode** (default): Accumulates content in RAM for backward compatibility with tests
pub struct ContentWriter {
    files: Vec<FileEntry>,
    writer: Option<std::io::BufWriter<File>>,
    current_offset: u64,
    file_path: Option<PathBuf>,
    // In-memory content buffer (only used if streaming mode not enabled)
    content: Vec<u8>,
    // First streaming write failure; surfaced by finalize() instead of panicking
    // inside add_file() (whose signature returns the file id, not a Result).
    write_error: Option<std::io::Error>,
}

impl ContentWriter {
    /// Create a new content writer (in-memory mode by default)
    ///
    /// Call init() to enable streaming mode before adding files.
    pub fn new() -> Self {
        Self {
            files: Vec::new(),
            writer: None,
            current_offset: 0,
            file_path: None,
            content: Vec::new(),
            write_error: None,
        }
    }

    /// Initialize the writer by creating the output file and writing header placeholder
    ///
    /// Crash safety: bytes are streamed into `<path>.tmp`; `finalize()` renames the
    /// temp file over `path` only after the header is complete and synced, so a
    /// reader never sees a short `content.bin`.
    pub fn init(&mut self, path: PathBuf) -> Result<()> {
        let tmp_path = crate::atomic_write::tmp_path_for(&path);
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp_path)
            .with_context(|| format!("Failed to create {}", tmp_path.display()))?;

        // Use a large buffer (16MB) for better write performance
        let mut writer = std::io::BufWriter::with_capacity(16 * 1024 * 1024, file);

        // Write placeholder header (will be overwritten in finalize())
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&0u64.to_le_bytes())?; // num_files (placeholder)
        writer.write_all(&0u64.to_le_bytes())?; // index_offset (placeholder)
        writer.write_all(&[0u8; 8])?; // reserved

        self.writer = Some(writer);
        self.current_offset = 0; // Content starts after header
        self.file_path = Some(path);

        Ok(())
    }

    /// Add a file to the content store
    ///
    /// **Streaming mode** (if init() was called): Writes content to disk immediately.
    /// **In-memory mode** (default): Accumulates content in RAM.
    ///
    /// Returns the file_id (index into files array)
    pub fn add_file(&mut self, path: PathBuf, content: &str) -> u32 {
        let file_id = self.files.len() as u32;
        let content_bytes = content.as_bytes();
        let length = content_bytes.len() as u64;

        if let Some(ref mut w) = self.writer {
            // Streaming mode: write content immediately to disk
            let offset = self.current_offset;
            if let Err(e) = w.write_all(content_bytes)
                && self.write_error.is_none()
            {
                // Keep going so the caller sees one clear error from finalize()
                // instead of a panic mid-index; the temp file is discarded.
                self.write_error = Some(e);
            }
            self.current_offset += length;

            self.files.push(FileEntry {
                path,
                offset,
                length,
            });
        } else {
            // In-memory mode: accumulate in RAM (for backward compatibility)
            let offset = self.content.len() as u64;
            self.content.extend_from_slice(content_bytes);

            self.files.push(FileEntry {
                path,
                offset,
                length,
            });
        }

        file_id
    }

    /// Write the content store to disk
    ///
    /// This is the main entry point for the old API. It initializes the writer (if needed),
    /// and finalizes the file.
    pub fn write(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();

        // Initialize writer if not already done
        if self.writer.is_none() && self.file_path.is_none() {
            // Old API: no files written yet, need to write them now in-memory
            // This is a fallback for tests that don't call init()
            return self.write_legacy(path);
        }

        // New streaming API: already been writing, just finalize
        self.finalize_if_needed()?;

        Ok(())
    }

    /// Legacy write path for in-memory mode (backward compatibility)
    ///
    /// This is only used when write() is called without init() first.
    /// Content is accumulated in RAM and written all at once.
    fn write_legacy(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let tmp_path = crate::atomic_write::tmp_path_for(path);
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp_path)
            .with_context(|| format!("Failed to create {}", tmp_path.display()))?;

        // Use a large buffer (8MB) for better write performance
        let mut writer = std::io::BufWriter::with_capacity(8 * 1024 * 1024, file);

        // Calculate index offset (after header + content)
        let index_offset = HEADER_SIZE as u64 + self.content.len() as u64;

        // Write header
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&(self.files.len() as u64).to_le_bytes())?;
        writer.write_all(&index_offset.to_le_bytes())?;
        writer.write_all(&[0u8; 8])?; // reserved

        // Write all accumulated file contents
        writer.write_all(&self.content)?;

        // Write file index: the fixed-width table, then the path blob.
        write_file_index(&mut writer, &self.files, index_offset)?;

        writer.flush()?;
        writer.get_ref().sync_all()?;
        crate::atomic_write::atomic_replace(&tmp_path, path)
            .with_context(|| format!("Failed to move {} into place", path.display()))?;
        Ok(())
    }

    /// Finalize the content.bin file by writing the file index and updating the header
    fn finalize(&mut self) -> Result<()> {
        let mut writer = self
            .writer
            .take()
            .ok_or_else(|| anyhow::anyhow!("ContentWriter not initialized"))?;
        let final_path = self
            .file_path
            .clone()
            .ok_or_else(|| anyhow::anyhow!("ContentWriter has no output path"))?;
        let tmp_path = crate::atomic_write::tmp_path_for(&final_path);

        if let Some(e) = self.write_error.take() {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(anyhow::Error::new(e).context(format!(
                "Failed to write file content to {}",
                tmp_path.display()
            )));
        }

        // Write file index at current position: the fixed-width table, then the
        // path blob.
        let index_offset = HEADER_SIZE as u64 + self.current_offset;
        write_file_index(&mut writer, &self.files, index_offset)?;

        // Consume BufWriter and get the underlying File
        let mut file = writer
            .into_inner()
            .map_err(|e| anyhow::anyhow!("Failed to flush BufWriter: {}", e.error()))?;

        // Rewind to header and update with correct values
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(0))?;

        // Write correct header
        file.write_all(MAGIC)?;
        file.write_all(&VERSION.to_le_bytes())?;
        file.write_all(&(self.files.len() as u64).to_le_bytes())?;
        file.write_all(&index_offset.to_le_bytes())?;
        file.write_all(&[0u8; 8])?; // reserved

        // Final sync to disk, then publish atomically: readers see either the
        // previous complete content.bin or this one, never a partial file.
        file.sync_all()?;
        drop(file);
        crate::atomic_write::atomic_replace(&tmp_path, &final_path)
            .with_context(|| format!("Failed to move {} into place", final_path.display()))?;

        log::debug!(
            "Finalized content.bin: {} files, {} bytes of content",
            self.files.len(),
            self.current_offset
        );

        Ok(())
    }

    /// Get the number of files
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Get total content size
    pub fn content_size(&self) -> usize {
        if self.writer.is_some() || self.file_path.is_some() {
            // Streaming mode
            self.current_offset as usize
        } else {
            // In-memory mode
            self.content.len()
        }
    }

    /// Finalize content store if it hasn't been finalized yet
    ///
    /// This is safe to call multiple times - subsequent calls are no-ops.
    pub fn finalize_if_needed(&mut self) -> Result<()> {
        if self.writer.is_some() {
            self.finalize()?;
            // Clear writer to mark as finalized
            self.writer = None;
        }
        Ok(())
    }
}

impl Default for ContentWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Write the version-2 file index: `files.len()` fixed-width entries followed by
/// the path blob. `index_offset` is where the table starts in the file.
fn write_file_index<W: Write>(
    writer: &mut W,
    files: &[FileEntry],
    index_offset: u64,
) -> Result<()> {
    let blob_start = index_offset + (files.len() * ENTRY_SIZE) as u64;
    let mut path_pos = blob_start;
    for entry in files {
        let path_len = entry.path.to_string_lossy().len() as u64;
        writer.write_all(&entry.offset.to_le_bytes())?;
        writer.write_all(&entry.length.to_le_bytes())?;
        writer.write_all(&path_pos.to_le_bytes())?;
        writer.write_all(&(path_len as u32).to_le_bytes())?;
        path_pos += path_len;
    }
    for entry in files {
        writer.write_all(entry.path.to_string_lossy().as_bytes())?;
    }
    Ok(())
}

/// Reader for memory-mapped content.bin
///
/// Provides zero-copy access to file contents.
pub struct ContentReader {
    _file: File,
    mmap: Mmap,
    /// From the header; entries are read from the mmap on demand.
    num_files: usize,
    /// Start of the entry table.
    index_offset: usize,
}

/// One file-index entry, read in place from the mmap.
#[derive(Debug, Clone, Copy)]
struct Entry<'a> {
    offset: u64,
    length: u64,
    path: &'a str,
}

impl ContentReader {
    /// Open and memory-map content.bin
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        let file =
            File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;

        let mmap = unsafe {
            Mmap::map(&file).with_context(|| format!("Failed to mmap {}", path.display()))?
        };

        // Validate header
        if mmap.len() < HEADER_SIZE {
            anyhow::bail!(
                "content.bin too small (expected at least {} bytes)",
                HEADER_SIZE
            );
        }

        if &mmap[0..4] != MAGIC {
            anyhow::bail!("Invalid content.bin (wrong magic bytes)");
        }

        let version = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]);
        if version != VERSION {
            anyhow::bail!("Unsupported content.bin version: {}", version);
        }

        let num_files = u64::from_le_bytes([
            mmap[8], mmap[9], mmap[10], mmap[11], mmap[12], mmap[13], mmap[14], mmap[15],
        ]);

        let index_offset = u64::from_le_bytes([
            mmap[16], mmap[17], mmap[18], mmap[19], mmap[20], mmap[21], mmap[22], mmap[23],
        ]) as usize;

        // Bounds of the entry table, then the first and last entry as a sanity
        // check. Nothing is decoded: a query reads the entries it verifies.
        let num_files = num_files as usize;
        let table_end = index_offset.saturating_add(num_files.saturating_mul(ENTRY_SIZE));
        if table_end > mmap.len() {
            anyhow::bail!(
                "Truncated file index (index_offset={}, num_files={}, mmap.len()={})",
                index_offset,
                num_files,
                mmap.len()
            );
        }
        let reader = Self {
            _file: file,
            mmap,
            num_files,
            index_offset,
        };
        if num_files > 0 {
            for id in [0u32, (num_files - 1) as u32] {
                if reader.entry(id).is_none() {
                    anyhow::bail!("Truncated file entry at file {}", id);
                }
            }
        }
        Ok(reader)
    }

    /// The file-index entry for `file_id`, read in place. `None` when the id is
    /// out of range or the entry points outside the file.
    fn entry(&self, file_id: u32) -> Option<Entry<'_>> {
        let id = file_id as usize;
        if id >= self.num_files {
            return None;
        }
        let at = self.index_offset + id * ENTRY_SIZE;
        let b = self.mmap.get(at..at + ENTRY_SIZE)?;
        let u64_at = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let offset = u64_at(0);
        let length = u64_at(8);
        let path_pos = u64_at(16) as usize;
        let path_len = u32::from_le_bytes(b[24..28].try_into().unwrap()) as usize;
        let path = std::str::from_utf8(self.mmap.get(path_pos..path_pos + path_len)?).ok()?;
        Some(Entry {
            offset,
            length,
            path,
        })
    }

    /// Get file content by file_id
    pub fn get_file_content(&self, file_id: u32) -> Result<&str> {
        let entry = self
            .entry(file_id)
            .ok_or_else(|| anyhow::anyhow!("Invalid file_id: {}", file_id))?;

        let start = HEADER_SIZE + entry.offset as usize;
        let end = start + entry.length as usize;

        if end > self.mmap.len() {
            anyhow::bail!("File content out of bounds");
        }

        let bytes = &self.mmap[start..end];
        std::str::from_utf8(bytes).context("Invalid UTF-8 in file content")
    }

    /// Get file path by file_id
    pub fn get_file_path(&self, file_id: u32) -> Option<&Path> {
        self.entry(file_id).map(|e| Path::new(e.path))
    }

    /// Get number of files
    pub fn file_count(&self) -> usize {
        self.num_files
    }

    /// Get file_id (array index) by path
    ///
    /// This looks up a file by its path and returns the array index, which is the
    /// correct file_id to use with get_file_content() and other methods.
    ///
    /// Note: This is different from database file_ids, which are AUTO INCREMENT values.
    pub fn get_file_id_by_path(&self, path: &str) -> Option<u32> {
        // Normalize the input path (strip ./ prefix if present)
        let normalized_input = path.strip_prefix("./").unwrap_or(path);

        (0..self.num_files as u32).find(|&id| {
            self.entry(id).is_some_and(|entry| {
                // Normalize the stored path (strip ./ prefix if present)
                entry.path.strip_prefix("./").unwrap_or(entry.path) == normalized_input
            })
        })
    }

    /// Get content at a specific byte offset
    pub fn get_content_at_offset(
        &self,
        file_id: u32,
        byte_offset: u32,
        length: usize,
    ) -> Result<&str> {
        let entry = self
            .entry(file_id)
            .ok_or_else(|| anyhow::anyhow!("Invalid file_id: {}", file_id))?;

        let start = HEADER_SIZE + entry.offset as usize + byte_offset as usize;
        let end = start + length;

        if end > self.mmap.len() {
            anyhow::bail!("Content out of bounds");
        }

        let bytes = &self.mmap[start..end];
        std::str::from_utf8(bytes).context("Invalid UTF-8 in content")
    }

    /// Get context around a byte offset (for showing match results)
    ///
    /// Returns (lines_before, matching_line, lines_after)
    pub fn get_context(
        &self,
        file_id: u32,
        byte_offset: u32,
        context_lines: usize,
    ) -> Result<(Vec<String>, String, Vec<String>)> {
        let content = self.get_file_content(file_id)?;
        let lines: Vec<&str> = content.lines().collect();

        // Find which line contains this byte offset
        let mut current_offset = 0;
        let mut line_idx = 0;

        for (idx, line) in lines.iter().enumerate() {
            let line_end = current_offset + line.len() + 1; // +1 for newline
            if byte_offset as usize >= current_offset && (byte_offset as usize) < line_end {
                line_idx = idx;
                break;
            }
            current_offset = line_end;
        }

        // Extract context
        let start = line_idx.saturating_sub(context_lines);
        let end = (line_idx + context_lines + 1).min(lines.len());

        let before: Vec<String> = lines[start..line_idx]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let matching = lines
            .get(line_idx)
            .map(|s| s.to_string())
            .unwrap_or_default();

        let after: Vec<String> = lines[line_idx + 1..end]
            .iter()
            .map(|s| s.to_string())
            .collect();

        Ok((before, matching, after))
    }

    /// Get context around a specific line number (1-indexed)
    ///
    /// Returns (lines_before, lines_after)
    pub fn get_context_by_line(
        &self,
        file_id: u32,
        line_number: usize,
        context_lines: usize,
    ) -> Result<(Vec<String>, Vec<String>)> {
        let content = self.get_file_content(file_id)?;
        let lines: Vec<&str> = content.lines().collect();

        // Convert from 1-indexed to 0-indexed
        let line_idx = line_number.saturating_sub(1);

        // Extract context
        let start = line_idx.saturating_sub(context_lines);
        let end = (line_idx + context_lines + 1).min(lines.len());

        let before: Vec<String> = lines[start..line_idx]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let after: Vec<String> = lines[line_idx + 1..end]
            .iter()
            .map(|s| s.to_string())
            .collect();

        Ok((before, after))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_content_writer_basic() {
        let mut writer = ContentWriter::new();

        let file1_id = writer.add_file(PathBuf::from("test1.txt"), "Hello, world!");
        let file2_id = writer.add_file(PathBuf::from("test2.txt"), "Goodbye, world!");

        assert_eq!(file1_id, 0);
        assert_eq!(file2_id, 1);
        assert_eq!(writer.file_count(), 2);
    }

    #[test]
    fn test_content_roundtrip() {
        let temp = TempDir::new().unwrap();
        let content_path = temp.path().join("content.bin");

        // Write
        let mut writer = ContentWriter::new();
        writer.add_file(PathBuf::from("file1.txt"), "First file content");
        writer.add_file(PathBuf::from("file2.txt"), "Second file content");
        writer.write(&content_path).unwrap();

        // Read
        let reader = ContentReader::open(&content_path).unwrap();

        assert_eq!(reader.file_count(), 2);
        assert_eq!(reader.get_file_content(0).unwrap(), "First file content");
        assert_eq!(reader.get_file_content(1).unwrap(), "Second file content");
        assert_eq!(reader.get_file_path(0).unwrap(), Path::new("file1.txt"));
        assert_eq!(reader.get_file_path(1).unwrap(), Path::new("file2.txt"));
    }

    #[test]
    fn test_get_context() {
        let temp = TempDir::new().unwrap();
        let content_path = temp.path().join("content.bin");

        let mut writer = ContentWriter::new();
        writer.add_file(
            PathBuf::from("test.txt"),
            "Line 1\nLine 2\nLine 3 with match\nLine 4\nLine 5",
        );
        writer.write(&content_path).unwrap();

        let reader = ContentReader::open(&content_path).unwrap();

        // Byte offset of "Line 3" (14 = "Line 1\n" + "Line 2\n")
        let (before, matching, after) = reader.get_context(0, 14, 1).unwrap();

        assert_eq!(before.len(), 1);
        assert_eq!(before[0], "Line 2");
        assert_eq!(matching, "Line 3 with match");
        assert_eq!(after.len(), 1);
        assert_eq!(after[0], "Line 4");
    }

    #[test]
    fn test_streaming_roundtrip() {
        let temp = TempDir::new().unwrap();
        let content_path = temp.path().join("content.bin");

        // Use the streaming path: init() -> add_file() -> finalize_if_needed()
        let mut writer = ContentWriter::new();
        writer.init(content_path.clone()).unwrap();
        writer.add_file(PathBuf::from("src/main.rs"), "fn main() {}\n");
        writer.add_file(
            PathBuf::from("src/lib.rs"),
            "pub fn hello() -> &'static str { \"hi\" }\n",
        );
        writer.finalize_if_needed().unwrap();

        // Verify the file can be read back correctly
        let reader = ContentReader::open(&content_path).unwrap();
        assert_eq!(reader.file_count(), 2);
        assert_eq!(reader.get_file_content(0).unwrap(), "fn main() {}\n");
        assert_eq!(
            reader.get_file_content(1).unwrap(),
            "pub fn hello() -> &'static str { \"hi\" }\n"
        );
        assert_eq!(reader.get_file_path(0).unwrap(), Path::new("src/main.rs"));
        assert_eq!(reader.get_file_path(1).unwrap(), Path::new("src/lib.rs"));
    }

    #[test]
    fn test_multiline_file() {
        let temp = TempDir::new().unwrap();
        let content_path = temp.path().join("content.bin");

        let content = "fn main() {\n    println!(\"Hello\");\n}\n";

        let mut writer = ContentWriter::new();
        writer.add_file(PathBuf::from("main.rs"), content);
        writer.write(&content_path).unwrap();

        let reader = ContentReader::open(&content_path).unwrap();
        assert_eq!(reader.get_file_content(0).unwrap(), content);
    }
}
