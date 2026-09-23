//! Parallel construction of `trigrams.bin` for `rfx index`.
//!
//! The in-memory [`TrigramIndex`](crate::trigram::TrigramIndex) is the query-side
//! structure and the fallback builder for small trees. This module is the
//! indexer's writer for trees of any size. It produces a file byte-identical to
//! `TrigramIndex::write` while doing almost none of the work on one thread:
//!
//! 1. **Extraction runs in the read pool.** Each file yields a [`TrigramRun`]:
//!    its `(trigram, line)` pairs sorted, with a 256-entry table of where each
//!    top byte starts. No file id is inside — the serial loop assigns ids in
//!    discovery order, exactly as before, and attaches them when the batch is
//!    built.
//! 2. **Batches are built per shard, in parallel, without sorting.** Trigram
//!    space is cut into 256 shards by top byte. A shard walks every run's slice
//!    for that byte in file-id order, so each posting list comes out sorted by
//!    `(file_id, line_no)` with no sort and no dedup (lines were deduplicated at
//!    extraction). Lists are encoded straight into a compact partial.
//! 3. **Partials merge by byte copy.** Batches hand out strictly increasing file
//!    ids, so a trigram's final list is the concatenation of its per-partial
//!    lists, and the V4 encoding of the concatenation is the concatenation of
//!    the encodings except for the first file delta of every later partial. The
//!    merge rewrites that one varint and copies the rest — no decode, no
//!    re-encode — and writes header, directory, data and paths in one pass
//!    because the trigram count is known up front.
//!
//! Until 2.0.0 extraction and the inverted index were built on the main thread
//! with a `HashMap` entry per posting, partials stored 8 bytes per posting, and
//! the merge re-encoded every list and then read the whole data section back
//! into memory to insert the directory. On a Kubernetes-sized tree the batch loop
//! ran at ~11 MB/s and the merge held a copy of `trigrams.bin` in RAM.

use crate::trigram::{
    DIR_ENTRY_SIZE, FileLocation, HEADER_SIZE, MAGIC, PATHS_OFFSET_OFFSET, Trigram, VERSION,
    decode_posting_list, encode_posting_list, scan_line_trigrams, skip_varint, write_varint,
};
use anyhow::{Context, Result};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Number of shards: one per top byte of the 24-bit trigram.
const SHARDS: usize = 256;
/// Trigrams per shard (the low 16 bits).
const SHARD_SPAN: usize = 1 << 16;
/// Shards built between two sequential appends to the partial. Bounds the
/// encoded bytes held in memory to roughly a quarter of the batch.
const SHARDS_PER_ROUND: usize = 64;
/// Magic of a partial file. Partials never outlive the index run that wrote them.
const PARTIAL_MAGIC: &[u8; 4] = b"RFTP";
const PARTIAL_VERSION: u32 = 2;
const PARTIAL_HEADER: usize = 8;
/// Fixed part of a partial record: trigram, n_postings, first_file_id,
/// last_file_id, enc_len.
const RECORD_HEADER: usize = 20;
/// Writer buffer for the final file and for a partial on disk.
const WRITE_BUF: usize = 16 * 1024 * 1024;

/// One file's trigram postings, produced in the read pool.
///
/// `keys[i] = (trigram << 32) | line_no`, ascending, so the run is sorted by
/// `(trigram, line_no)`; `shard_starts[s]..shard_starts[s + 1]` is the slice of
/// keys whose trigram has top byte `s`.
#[derive(Debug, Clone, Default)]
pub struct TrigramRun {
    keys: Vec<u64>,
    shard_starts: Box<[u32]>,
}

impl TrigramRun {
    /// Number of `(trigram, line)` postings in the run.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the run holds no postings.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    #[inline]
    fn shard(&self, s: usize) -> &[u64] {
        if self.shard_starts.is_empty() {
            return &[];
        }
        &self.keys[self.shard_starts[s] as usize..self.shard_starts[s + 1] as usize]
    }
}

#[inline]
fn key_trigram(key: u64) -> Trigram {
    (key >> 32) as Trigram
}

#[inline]
fn key_line(key: u64) -> u32 {
    key as u32
}

/// Extract a file's trigram run.
///
/// `scratch` is reused across calls on the same thread so the sort buffer is
/// allocated once per thread, not once per file. The posting set is exactly
/// what [`extract_trigrams_with_locations`](crate::trigram::extract_trigrams_with_locations)
/// produces: one posting per distinct trigram per line, identical newline
/// attribution.
pub fn extract_trigram_run(text: &str, scratch: &mut Vec<u64>) -> TrigramRun {
    scratch.clear();
    scan_line_trigrams(text, |line_no, trigrams| {
        let line = line_no as u64;
        scratch.extend(trigrams.iter().map(|&t| ((t as u64) << 32) | line));
    });
    scratch.sort_unstable();

    let keys: Vec<u64> = scratch.clone();
    let mut shard_starts = vec![0u32; SHARDS + 1].into_boxed_slice();
    let mut i = 0usize;
    for (s, start) in shard_starts.iter_mut().enumerate().take(SHARDS) {
        while i < keys.len() && ((key_trigram(keys[i]) >> 16) as usize) < s {
            i += 1;
        }
        *start = i as u32;
    }
    shard_starts[SHARDS] = keys.len() as u32;
    TrigramRun { keys, shard_starts }
}

/// Where a flushed batch lives.
enum PartialSource {
    Memory(Vec<u8>),
    File(PathBuf),
}

/// Per-thread scratch for [`build_shard`].
#[derive(Default)]
struct ShardScratch {
    counts: Vec<u32>,
    offsets: Vec<u32>,
    postings: Vec<FileLocation>,
    encoded: Vec<u8>,
}

/// Build one shard's partial records from the batch's runs.
///
/// Returns `(records, seen_bits)`: the encoded records for every trigram of the
/// shard that has postings, in ascending trigram order, and a 1024-word bitset
/// of those trigrams (low 16 bits).
fn build_shard(
    shard: usize,
    runs: &[(u32, TrigramRun)],
    scratch: &mut ShardScratch,
) -> Result<(Vec<u8>, Vec<u64>)> {
    let ShardScratch {
        counts,
        offsets,
        postings,
        encoded,
    } = scratch;
    counts.clear();
    counts.resize(SHARD_SPAN, 0);
    offsets.clear();
    offsets.resize(SHARD_SPAN + 1, 0);

    // Count pass.
    let mut total = 0usize;
    for (_, run) in runs {
        let slice = run.shard(shard);
        total += slice.len();
        for &key in slice {
            counts[(key_trigram(key) & 0xFFFF) as usize] += 1;
        }
    }
    if total == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    // Prefix sums; `offsets[t]` is the write cursor of trigram `t` during scatter.
    let mut acc = 0u32;
    for t in 0..SHARD_SPAN {
        offsets[t] = acc;
        acc += counts[t];
    }
    offsets[SHARD_SPAN] = acc;

    // Scatter pass, in file-id order: every list comes out sorted and unique.
    postings.clear();
    postings.resize(total, FileLocation::new(0, 0));
    for (file_id, run) in runs {
        for &key in run.shard(shard) {
            let t = (key_trigram(key) & 0xFFFF) as usize;
            let at = offsets[t] as usize;
            postings[at] = FileLocation::new(*file_id, key_line(key));
            offsets[t] += 1;
        }
    }

    // Encode pass. After scatter `offsets[t]` is the END of list `t`, and its
    // start is `offsets[t] - counts[t]`.
    let mut records = Vec::with_capacity(total * 2);
    let mut seen = vec![0u64; SHARD_SPAN / 64];
    let base = (shard as Trigram) << 16;
    for t in 0..SHARD_SPAN {
        let n = counts[t] as usize;
        if n == 0 {
            continue;
        }
        let end = offsets[t] as usize;
        let list = &postings[end - n..end];
        encoded.clear();
        encode_posting_list(list, encoded)?;
        let trigram = base | t as Trigram;
        records.extend_from_slice(&trigram.to_le_bytes());
        records.extend_from_slice(&(n as u32).to_le_bytes());
        records.extend_from_slice(&list[0].file_id.to_le_bytes());
        records.extend_from_slice(&list[n - 1].file_id.to_le_bytes());
        records.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
        records.extend_from_slice(encoded);
        seen[t / 64] |= 1u64 << (t % 64);
    }
    Ok((records, seen))
}

/// Sink a partial is appended to.
enum PartialSink {
    Memory(Vec<u8>),
    File(BufWriter<File>, PathBuf),
}

impl PartialSink {
    fn append(&mut self, bytes: &[u8]) -> Result<()> {
        match self {
            PartialSink::Memory(buf) => buf.extend_from_slice(bytes),
            PartialSink::File(w, _) => w.write_all(bytes)?,
        }
        Ok(())
    }

    fn finish(self) -> Result<PartialSource> {
        Ok(match self {
            PartialSink::Memory(buf) => PartialSource::Memory(buf),
            PartialSink::File(mut w, path) => {
                w.flush()?;
                PartialSource::File(path)
            }
        })
    }
}

/// One partial record as the merge sees it.
struct Record {
    trigram: Trigram,
    n_postings: u32,
    first_file_id: u32,
    last_file_id: u32,
    bytes: Vec<u8>,
}

/// Sequential reader over a partial's records.
struct PartialReader {
    input: Box<dyn Read + Send>,
    current: Option<Record>,
}

impl PartialReader {
    fn open(source: &PartialSource, buf_size: usize) -> Result<Self> {
        let mut input: Box<dyn Read + Send> = match source {
            PartialSource::Memory(bytes) => Box::new(Cursor::new(bytes.clone())),
            PartialSource::File(path) => {
                let file = File::open(path)
                    .with_context(|| format!("Failed to open partial {}", path.display()))?;
                Box::new(BufReader::with_capacity(buf_size, file))
            }
        };
        let mut header = [0u8; PARTIAL_HEADER];
        input
            .read_exact(&mut header)
            .context("Truncated partial index header")?;
        if &header[..4] != PARTIAL_MAGIC
            || u32::from_le_bytes(header[4..8].try_into().unwrap()) != PARTIAL_VERSION
        {
            anyhow::bail!("Unrecognised partial index header");
        }
        let mut reader = Self {
            input,
            current: None,
        };
        reader.advance()?;
        Ok(reader)
    }

    /// Read the next record into `current`; `None` at end of input.
    fn advance(&mut self) -> Result<()> {
        let mut head = [0u8; RECORD_HEADER];
        match self.input.read_exact(&mut head) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                self.current = None;
                return Ok(());
            }
            Err(e) => return Err(e).context("Failed to read partial record"),
        }
        let u = |i: usize| u32::from_le_bytes(head[i..i + 4].try_into().unwrap());
        let enc_len = u(16) as usize;
        let mut bytes = match self.current.take() {
            Some(rec) => rec.bytes,
            None => Vec::new(),
        };
        bytes.clear();
        bytes.resize(enc_len, 0);
        self.input
            .read_exact(&mut bytes)
            .context("Truncated partial record")?;
        self.current = Some(Record {
            trigram: u(0),
            n_postings: u(4),
            first_file_id: u(8),
            last_file_id: u(12),
            bytes,
        });
        Ok(())
    }
}

/// Builds `trigrams.bin` from per-file runs, batch by batch.
pub struct TrigramIndexBuilder {
    files: Vec<PathBuf>,
    runs: Vec<(u32, TrigramRun)>,
    partials: Vec<PartialSource>,
    temp_dir: PathBuf,
    /// One bit per trigram written to any partial so far (2 MiB).
    seen: Vec<u64>,
    max_posting_list_entries: usize,
}

impl TrigramIndexBuilder {
    /// A builder whose on-disk partials go under `temp_dir` (created on first use).
    pub fn new(temp_dir: PathBuf) -> Self {
        Self {
            files: Vec::new(),
            runs: Vec::new(),
            partials: Vec::new(),
            temp_dir,
            seen: vec![0u64; (1 << 24) / 64],
            max_posting_list_entries: 0,
        }
    }

    /// Cap on posting list size; 0 = unlimited. Same semantics as
    /// [`TrigramIndex::set_max_posting_list_entries`](crate::trigram::TrigramIndex::set_max_posting_list_entries).
    pub fn set_max_posting_list_entries(&mut self, cap: usize) {
        self.max_posting_list_entries = cap;
    }

    /// Register a file and its run; returns the file id (index in discovery order).
    pub fn add_file(&mut self, path: PathBuf, run: TrigramRun) -> u32 {
        let file_id = self.files.len() as u32;
        self.files.push(path);
        if !run.is_empty() {
            self.runs.push((file_id, run));
        }
        file_id
    }

    /// Files registered so far.
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Distinct trigrams flushed so far (the final count once `write` has run).
    pub fn trigram_count(&self) -> usize {
        self.seen.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Whether any run is waiting to be flushed.
    pub fn has_pending(&self) -> bool {
        !self.runs.is_empty()
    }

    /// Build the current batch into a partial (on disk when `to_disk`, else in
    /// memory) and release the runs.
    pub fn flush_batch(&mut self, pool: &rayon::ThreadPool, to_disk: bool) -> Result<()> {
        if self.runs.is_empty() {
            return Ok(());
        }
        let runs = std::mem::take(&mut self.runs);

        let mut sink = if to_disk {
            std::fs::create_dir_all(&self.temp_dir).with_context(|| {
                format!(
                    "Failed to create temp directory {}",
                    self.temp_dir.display()
                )
            })?;
            let path = self
                .temp_dir
                .join(format!("partial_{}.bin", self.partials.len()));
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&path)
                .with_context(|| format!("Failed to create {}", path.display()))?;
            PartialSink::File(BufWriter::with_capacity(WRITE_BUF, file), path)
        } else {
            PartialSink::Memory(Vec::new())
        };
        let mut header = Vec::with_capacity(PARTIAL_HEADER);
        header.extend_from_slice(PARTIAL_MAGIC);
        header.extend_from_slice(&PARTIAL_VERSION.to_le_bytes());
        sink.append(&header)?;

        for round in (0..SHARDS).step_by(SHARDS_PER_ROUND) {
            let shards: Vec<usize> = (round..round + SHARDS_PER_ROUND).collect();
            let built: Vec<(Vec<u8>, Vec<u64>)> = pool.install(|| {
                shards
                    .par_iter()
                    .map_init(ShardScratch::default, |scratch, &shard| {
                        build_shard(shard, &runs, scratch)
                    })
                    .collect::<Result<Vec<_>>>()
            })?;
            for (shard, (records, seen)) in shards.iter().zip(built) {
                if records.is_empty() {
                    continue;
                }
                sink.append(&records)?;
                let base = shard * (SHARD_SPAN / 64);
                for (i, word) in seen.into_iter().enumerate() {
                    self.seen[base + i] |= word;
                }
            }
        }

        self.partials.push(sink.finish()?);
        log::debug!(
            "Flushed trigram batch {} ({} files) {}",
            self.partials.len(),
            runs.len(),
            if to_disk { "to disk" } else { "in memory" }
        );
        Ok(())
    }

    /// Flush what is pending, merge every partial into `path` (crash-safe:
    /// `<path>.tmp` + fsync + rename), and remove the partials.
    pub fn write(&mut self, pool: &rayon::ThreadPool, path: &Path) -> Result<()> {
        // A batch still in memory spills to disk only if others already did.
        let to_disk = self
            .partials
            .iter()
            .any(|p| matches!(p, PartialSource::File(_)));
        self.flush_batch(pool, to_disk)?;

        let result = self.merge(path);
        self.cleanup_partials();
        result
    }

    fn cleanup_partials(&mut self) {
        for partial in self.partials.drain(..) {
            if let PartialSource::File(p) = partial {
                let _ = std::fs::remove_file(p);
            }
        }
        let _ = std::fs::remove_dir(&self.temp_dir);
    }

    fn merge(&mut self, output_path: &Path) -> Result<()> {
        let num_trigrams = self.trigram_count() as u64;
        let num_files = self.files.len() as u64;
        let data_start = (HEADER_SIZE + num_trigrams as usize * DIR_ENTRY_SIZE) as u64;
        log::info!(
            "Merging {} partial indices ({} trigrams, {} files) into {}",
            self.partials.len(),
            num_trigrams,
            num_files,
            output_path.display()
        );

        let tmp_path = crate::atomic_write::tmp_path_for(output_path);
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp_path)
            .with_context(|| format!("Failed to create {}", tmp_path.display()))?;
        let mut writer = BufWriter::with_capacity(WRITE_BUF, file);

        // Header now, directory later (its size is known, its contents are not).
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&num_trigrams.to_le_bytes())?;
        writer.write_all(&num_files.to_le_bytes())?;
        writer.write_all(&0u64.to_le_bytes())?; // paths_offset, patched below
        writer.seek(SeekFrom::Start(data_start))?;

        // Readers, one per partial, in batch order (= ascending file ids).
        let k = self.partials.len().max(1);
        let buf_size = (64 * 1024 * 1024 / k).clamp(1024 * 1024, WRITE_BUF);
        let mut readers = Vec::with_capacity(self.partials.len());
        for source in &self.partials {
            readers.push(PartialReader::open(source, buf_size)?);
        }

        let mut heap: BinaryHeap<std::cmp::Reverse<(Trigram, usize)>> = BinaryHeap::new();
        for (i, r) in readers.iter().enumerate() {
            if let Some(rec) = &r.current {
                heap.push(std::cmp::Reverse((rec.trigram, i)));
            }
        }

        let cap = self.max_posting_list_entries;
        let mut directory: Vec<(Trigram, u64, u32)> = Vec::with_capacity(num_trigrams as usize);
        let mut offset = data_start;
        let mut same: Vec<usize> = Vec::new();
        let mut capped: Vec<FileLocation> = Vec::new();
        let mut encoded: Vec<u8> = Vec::new();

        while let Some(std::cmp::Reverse((trigram, first))) = heap.pop() {
            same.clear();
            same.push(first);
            while let Some(std::cmp::Reverse((t, i))) = heap.peek().copied() {
                if t != trigram {
                    break;
                }
                heap.pop();
                same.push(i);
            }
            // Heap order is (trigram, reader index), so `same` is ascending.

            let total: u64 = same
                .iter()
                .map(|&i| readers[i].current.as_ref().unwrap().n_postings as u64)
                .sum();

            let size: u32 = if cap > 0 && total > cap as u64 {
                // Rare: decode, truncate, re-encode.
                log::warn!(
                    "Trigram 0x{:06X} posting list has {} entries (cap {}); truncating.",
                    trigram,
                    total,
                    cap
                );
                capped.clear();
                for &i in &same {
                    let rec = readers[i].current.as_ref().unwrap();
                    capped.extend(decode_posting_list(&rec.bytes)?);
                }
                capped.truncate(cap);
                encoded.clear();
                encode_posting_list(&capped, &mut encoded)?;
                writer.write_all(&encoded)?;
                encoded.len() as u32
            } else {
                let mut written = 0usize;
                let mut prev_last: Option<u32> = None;
                for &i in &same {
                    let rec = readers[i].current.as_ref().unwrap();
                    match prev_last {
                        None => {
                            writer.write_all(&rec.bytes)?;
                            written += rec.bytes.len();
                        }
                        Some(prev) => {
                            let skip = skip_varint(&rec.bytes)?;
                            let mut delta = [0u8; 5];
                            let mut cursor = Cursor::new(&mut delta[..]);
                            write_varint(&mut cursor, rec.first_file_id.wrapping_sub(prev))?;
                            let n = cursor.position() as usize;
                            writer.write_all(&delta[..n])?;
                            writer.write_all(&rec.bytes[skip..])?;
                            written += n + rec.bytes.len() - skip;
                        }
                    }
                    prev_last = Some(rec.last_file_id);
                }
                written as u32
            };

            directory.push((trigram, offset, size));
            offset += size as u64;

            for &i in &same {
                readers[i].advance()?;
                if let Some(rec) = &readers[i].current {
                    heap.push(std::cmp::Reverse((rec.trigram, i)));
                }
            }
        }

        if directory.len() as u64 != num_trigrams {
            anyhow::bail!(
                "Trigram count mismatch: bitset says {}, merge produced {}",
                num_trigrams,
                directory.len()
            );
        }

        // Paths.
        let paths_offset = offset;
        for file_path in &self.files {
            let path_bytes = file_path.to_string_lossy();
            let path_bytes = path_bytes.as_bytes();
            write_varint(&mut writer, path_bytes.len() as u32)?;
            writer.write_all(path_bytes)?;
        }

        // Directory and the paths offset, then publish.
        writer.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
        for (trigram, data_offset, size) in &directory {
            writer.write_all(&trigram.to_le_bytes())?;
            writer.write_all(&data_offset.to_le_bytes())?;
            writer.write_all(&size.to_le_bytes())?;
        }
        writer.seek(SeekFrom::Start(PATHS_OFFSET_OFFSET as u64))?;
        writer.write_all(&paths_offset.to_le_bytes())?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        crate::atomic_write::atomic_replace(&tmp_path, output_path)
            .with_context(|| format!("Failed to move {} into place", output_path.display()))?;

        log::info!(
            "Wrote {} trigrams for {} files to {}",
            num_trigrams,
            num_files,
            output_path.display()
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trigram::TrigramIndex;
    use tempfile::TempDir;

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

    fn synthetic_files(seed: u64, n: usize) -> Vec<(PathBuf, String)> {
        let mut rng = Lcg(seed);
        let words = [
            "realm",
            "real",
            "alma",
            "lmn",
            "rea",
            "xyz",
            "ealm",
            "é\u{1F600}",
        ];
        (0..n)
            .map(|i| {
                let mut content = String::new();
                for _ in 0..60 {
                    for _ in 0..4 {
                        content.push_str(words[(rng.next() % words.len() as u32) as usize]);
                        content.push(' ');
                    }
                    content.push('\n');
                }
                if i % 7 == 0 {
                    content.push_str("no trailing newline");
                }
                (PathBuf::from(format!("dir/f{i}.txt")), content)
            })
            .collect()
    }

    fn pool() -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
    }

    fn write_in_memory(files: &[(PathBuf, String)], path: &Path, cap: usize) {
        let mut index = TrigramIndex::new();
        index.set_max_posting_list_entries(cap);
        for (p, content) in files {
            let id = index.add_file(p.clone());
            index.index_file(id, content);
        }
        index.finalize();
        index.write(path).unwrap();
    }

    fn write_with_builder(
        files: &[(PathBuf, String)],
        path: &Path,
        temp_dir: &Path,
        batch: usize,
        to_disk: bool,
        cap: usize,
    ) -> TrigramIndexBuilder {
        let pool = pool();
        let mut builder = TrigramIndexBuilder::new(temp_dir.to_path_buf());
        builder.set_max_posting_list_entries(cap);
        let mut scratch = Vec::new();
        for (i, (p, content)) in files.iter().enumerate() {
            let run = extract_trigram_run(content, &mut scratch);
            builder.add_file(p.clone(), run);
            if (i + 1) % batch == 0 {
                builder.flush_batch(&pool, to_disk).unwrap();
            }
        }
        builder.write(&pool, path).unwrap();
        builder
    }

    #[test]
    fn run_matches_location_extraction() {
        let files = synthetic_files(7, 3);
        let mut scratch = Vec::new();
        for (i, (_, content)) in files.iter().enumerate() {
            let run = extract_trigram_run(content, &mut scratch);
            let mut expected: Vec<(Trigram, u32)> =
                crate::trigram::extract_trigrams_with_locations(content, i as u32)
                    .into_iter()
                    .map(|(t, loc)| (t, loc.line_no))
                    .collect();
            expected.sort_unstable();
            let got: Vec<(Trigram, u32)> = run
                .keys
                .iter()
                .map(|&k| (key_trigram(k), key_line(k)))
                .collect();
            assert_eq!(got, expected);
            // Shard table covers the keys exactly.
            assert_eq!(run.shard_starts[0], 0);
            assert_eq!(run.shard_starts[SHARDS] as usize, run.keys.len());
            for s in 0..SHARDS {
                for &k in run.shard(s) {
                    assert_eq!((key_trigram(k) >> 16) as usize, s);
                }
            }
        }
    }

    #[test]
    fn builder_output_is_byte_identical_to_in_memory_writer() {
        let files = synthetic_files(0xcafe, 40);
        let temp = TempDir::new().unwrap();

        let reference = temp.path().join("ref.bin");
        write_in_memory(&files, &reference, 0);

        let one_memory = temp.path().join("one.bin");
        write_with_builder(&files, &one_memory, &temp.path().join("t1"), 1000, false, 0);

        let disk = temp.path().join("disk.bin");
        let builder = write_with_builder(&files, &disk, &temp.path().join("t2"), 10, true, 0);
        assert!(!temp.path().join("t2").exists(), "partials must be removed");

        let reference_bytes = std::fs::read(&reference).unwrap();
        assert_eq!(std::fs::read(&one_memory).unwrap(), reference_bytes);
        assert_eq!(std::fs::read(&disk).unwrap(), reference_bytes);

        let lazy = TrigramIndex::load(&disk).unwrap();
        assert_eq!(lazy.trigram_count(), builder.trigram_count());
        assert_eq!(lazy.file_count(), 40);
        let mem = {
            let mut index = TrigramIndex::new();
            for (p, content) in &files {
                let id = index.add_file(p.clone());
                index.index_file(id, content);
            }
            index.finalize();
            index
        };
        for pattern in ["realm", "alma", "xyz", "lmn r", "ealm x", "nothing", "é"] {
            assert_eq!(
                lazy.search(pattern),
                mem.search(pattern),
                "pattern {pattern:?}"
            );
        }
    }

    #[test]
    fn builder_honours_posting_cap() {
        let files: Vec<(PathBuf, String)> = (0..6)
            .map(|i| (PathBuf::from(format!("d{i}.txt")), "aaa \n".repeat(50)))
            .collect();
        let temp = TempDir::new().unwrap();
        let reference = temp.path().join("ref.bin");
        write_in_memory(&files, &reference, 10);
        let built = temp.path().join("built.bin");
        write_with_builder(&files, &built, &temp.path().join("t"), 2, true, 10);
        assert_eq!(
            std::fs::read(&built).unwrap(),
            std::fs::read(&reference).unwrap()
        );
    }

    #[test]
    fn empty_and_tiny_inputs() {
        let temp = TempDir::new().unwrap();
        let pool = pool();

        let out = temp.path().join("empty.bin");
        let mut builder = TrigramIndexBuilder::new(temp.path().join("t"));
        builder.write(&pool, &out).unwrap();
        let lazy = TrigramIndex::load(&out).unwrap();
        assert_eq!(lazy.trigram_count(), 0);
        assert_eq!(lazy.file_count(), 0);

        let files = vec![
            (PathBuf::from("a.txt"), "ab".to_string()),
            (PathBuf::from("b.txt"), String::new()),
            (PathBuf::from("c.txt"), "abc".to_string()),
        ];
        let reference = temp.path().join("ref.bin");
        write_in_memory(&files, &reference, 0);
        let built = temp.path().join("built.bin");
        write_with_builder(&files, &built, &temp.path().join("t"), 1, true, 0);
        assert_eq!(
            std::fs::read(&built).unwrap(),
            std::fs::read(&reference).unwrap()
        );
        let lazy = TrigramIndex::load(&built).unwrap();
        assert_eq!(lazy.file_count(), 3);
        assert_eq!(lazy.trigram_count(), 1);
    }
}
