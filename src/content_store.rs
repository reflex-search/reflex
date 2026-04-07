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
//! File Index (at index_offset):
//!   For each file:
//!     path_len: u32
//!     path: UTF-8 string
//!     offset: u64 (byte offset to file content)
//!     length: u64 (file size in bytes)
//! ```

use anyhow::{Context, Result};
use memmap2::Mmap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"RFCT";
const VERSION: u32 = 1;
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
        }
    }

    /// Initialize the writer by creating the output file and writing header placeholder
    pub fn init(&mut self, path: PathBuf) -> Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .with_context(|| format!("Failed to create {}", path.display()))?;

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
            w.write_all(content_bytes)
                .expect("Failed to write file content to content.bin");
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
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .with_context(|| format!("Failed to create {}", path.display()))?;

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

        // Write file index
        for entry in &self.files {
            let path_str = entry.path.to_string_lossy();
            let path_bytes = path_str.as_bytes();

            writer.write_all(&(path_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(path_bytes)?;
            writer.write_all(&entry.offset.to_le_bytes())?;
            writer.write_all(&entry.length.to_le_bytes())?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Finalize the content.bin file by writing the file index and updating the header
    fn finalize(&mut self) -> Result<()> {
        let mut writer = self.writer.take()
            .ok_or_else(|| anyhow::anyhow!("ContentWriter not initialized"))?;

        // Write file index at current position
        let index_offset = HEADER_SIZE as u64 + self.current_offset;

        for entry in &self.files {
            let path_str = entry.path.to_string_lossy();
            let path_bytes = path_str.as_bytes();

            writer.write_all(&(path_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(path_bytes)?;
            writer.write_all(&entry.offset.to_le_bytes())?;
            writer.write_all(&entry.length.to_le_bytes())?;
        }

        // Consume BufWriter and get the underlying File
        let mut file = writer.into_inner()
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

        // Final sync to disk
        file.sync_all()?;

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

/// Reader for memory-mapped content.bin
///
/// Provides zero-copy access to file contents.
pub struct ContentReader {
    _file: File,
    mmap: Mmap,
    files: Vec<FileEntry>,
}

impl ContentReader {
    /// Open and memory-map content.bin
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        let file = File::open(path)
            .with_context(|| format!("Failed to open {}", path.display()))?;

        let mmap = unsafe {
            Mmap::map(&file)
                .with_context(|| format!("Failed to mmap {}", path.display()))?
        };

        // Validate header
        if mmap.len() < HEADER_SIZE {
            anyhow::bail!("content.bin too small (expected at least {} bytes)", HEADER_SIZE);
        }

        if &mmap[0..4] != MAGIC {
            anyhow::bail!("Invalid content.bin (wrong magic bytes)");
        }

        let version = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]);
        if version != VERSION {
            anyhow::bail!("Unsupported content.bin version: {}", version);
        }

        let num_files = u64::from_le_bytes([
            mmap[8], mmap[9], mmap[10], mmap[11],
            mmap[12], mmap[13], mmap[14], mmap[15],
        ]);

        let index_offset = u64::from_le_bytes([
            mmap[16], mmap[17], mmap[18], mmap[19],
            mmap[20], mmap[21], mmap[22], mmap[23],
        ]) as usize;

        // Read file index
        let mut files = Vec::new();
        let mut pos = index_offset;

        for i in 0..num_files {
            if pos + 4 > mmap.len() {
                anyhow::bail!("Truncated file index at file {} (pos={}, mmap.len()={})", i, pos, mmap.len());
            }

            let path_len = u32::from_le_bytes([
                mmap[pos],
                mmap[pos + 1],
                mmap[pos + 2],
                mmap[pos + 3],
            ]) as usize;
            pos += 4;

            if pos + path_len + 16 > mmap.len() {
                anyhow::bail!("Truncated file entry at file {} (pos={}, path_len={}, need={}, mmap.len()={})",
                    i, pos, path_len, pos + path_len + 16, mmap.len());
            }

            let path_bytes = &mmap[pos..pos + path_len];
            let path_str = std::str::from_utf8(path_bytes)
                .context("Invalid UTF-8 in file path")?;
            let path = PathBuf::from(path_str);
            pos += path_len;

            let offset = u64::from_le_bytes([
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

            let length = u64::from_le_bytes([
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

            files.push(FileEntry {
                path,
                offset,
                length,
            });
        }

        Ok(Self {
            _file: file,
            mmap,
            files,
        })
    }

    /// Get file content by file_id
    pub fn get_file_content(&self, file_id: u32) -> Result<&str> {
        let entry = self.files
            .get(file_id as usize)
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
        self.files.get(file_id as usize).map(|e| e.path.as_path())
    }

    /// Get number of files
    pub fn file_count(&self) -> usize {
        self.files.len()
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

        self.files.iter().position(|entry| {
            // Normalize the stored path (strip ./ prefix if present)
            let stored_path = entry.path.to_string_lossy();
            let normalized_stored = stored_path.strip_prefix("./").unwrap_or(&stored_path);
            normalized_stored == normalized_input
        }).map(|idx| idx as u32)
    }

    /// Get content at a specific byte offset
    pub fn get_content_at_offset(&self, file_id: u32, byte_offset: u32, length: usize) -> Result<&str> {
        let entry = self.files
            .get(file_id as usize)
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
    pub fn get_context(&self, file_id: u32, byte_offset: u32, context_lines: usize) -> Result<(Vec<String>, String, Vec<String>)> {
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

        let matching = lines.get(line_idx)
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
    pub fn get_context_by_line(&self, file_id: u32, line_number: usize, context_lines: usize) -> Result<(Vec<String>, Vec<String>)> {
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
        writer.add_file(PathBuf::from("src/lib.rs"), "pub fn hello() -> &'static str { \"hi\" }\n");
        writer.finalize_if_needed().unwrap();

        // Verify the file can be read back correctly
        let reader = ContentReader::open(&content_path).unwrap();
        assert_eq!(reader.file_count(), 2);
        assert_eq!(reader.get_file_content(0).unwrap(), "fn main() {}\n");
        assert_eq!(reader.get_file_content(1).unwrap(), "pub fn hello() -> &'static str { \"hi\" }\n");
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
