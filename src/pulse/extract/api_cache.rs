//! The API of every source file, extracted once per content hash.
//!
//! `.reflex/pulse/api.db` maps `(path, content hash, extractor version)` to a
//! zstd-compressed [`ApiFile`]. A run extracts only files whose content changed since
//! the last run, in parallel, reading source from `content.bin` (never the working
//! tree), and writes the new rows in one transaction.

use super::{Corpus, FileRole};
use crate::cache::CacheManager;
use crate::content_store::ContentReader;
use crate::parsers::api::{self, ApiFile, EXTRACTOR_VERSION};
use crate::symbol_cache::{decode_json_blob, encode_json_blob};
use anyhow::{Context, Result};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

/// Extracted APIs, keyed by index into `Corpus::files`.
#[derive(Debug, Default)]
pub struct ApiIndex {
    pub files: BTreeMap<usize, ApiFile>,
    pub stats: ApiStats,
}

#[derive(Debug, Default, Clone)]
pub struct ApiStats {
    pub cached: usize,
    pub extracted: usize,
    pub failed: usize,
    pub elapsed: Duration,
}

fn open(cache: &CacheManager) -> Result<rusqlite::Connection> {
    let dir = cache.path().join("pulse");
    std::fs::create_dir_all(&dir)?;
    let conn = crate::cache::open_meta_db(dir.join("api.db"))?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS api_files (
            path TEXT PRIMARY KEY,
            hash TEXT NOT NULL,
            version INTEGER NOT NULL,
            blob BLOB NOT NULL
        )",
    )?;
    Ok(conn)
}

/// Load (and fill) the API cache for every source file with an extractor.
pub fn load(cache: &CacheManager, corpus: &Corpus) -> Result<ApiIndex> {
    let start = Instant::now();
    let wanted: Vec<usize> = corpus
        .with_role(FileRole::Source)
        .filter(|(_, f)| api::has_extractor(f.language))
        .map(|(i, _)| i)
        .collect();
    let mut index = ApiIndex::default();
    if wanted.is_empty() {
        return Ok(index);
    }

    let mut conn = open(cache)?;
    let mut cached: HashMap<String, (String, i64, Vec<u8>)> = HashMap::new();
    {
        let mut stmt = conn.prepare("SELECT path, hash, version, blob FROM api_files")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                (
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, Vec<u8>>(3)?,
                ),
            ))
        })?;
        for row in rows {
            let (path, v) = row?;
            cached.insert(path, v);
        }
    }

    let mut misses = Vec::new();
    for &i in &wanted {
        let f = &corpus.files[i];
        let hit = cached.get(&f.path).and_then(|(hash, version, blob)| {
            (hash == &f.hash && *version == EXTRACTOR_VERSION as i64)
                .then(|| decode_json_blob::<ApiFile>(blob).ok())
                .flatten()
        });
        match hit {
            Some(api) => {
                index.files.insert(i, api);
                index.stats.cached += 1;
            }
            None => misses.push(i),
        }
    }

    if !misses.is_empty() {
        let reader =
            ContentReader::open(cache.path().join("content.bin")).context("opening content.bin")?;
        let ids: HashMap<&str, u32> = (0..reader.file_count() as u32)
            .filter_map(|id| {
                let p = reader.get_file_path(id)?.to_str()?;
                Some((p.strip_prefix("./").unwrap_or(p), id))
            })
            .collect();
        let extracted: Vec<(usize, Option<ApiFile>)> = misses
            .par_iter()
            .map(|&i| {
                let f = &corpus.files[i];
                let api = ids
                    .get(f.path.strip_prefix("./").unwrap_or(&f.path))
                    .and_then(|&id| reader.get_file_content(id).ok())
                    .and_then(|src| api::extract(f.language, src));
                (i, api)
            })
            .collect();

        let tx = conn.transaction()?;
        {
            let mut put = tx.prepare(
                "INSERT OR REPLACE INTO api_files (path, hash, version, blob) VALUES (?1, ?2, ?3, ?4)",
            )?;
            for (i, api) in extracted {
                let f = &corpus.files[i];
                match api {
                    Some(api) => {
                        put.execute(rusqlite::params![
                            f.path,
                            f.hash,
                            EXTRACTOR_VERSION as i64,
                            encode_json_blob(&api)?
                        ])?;
                        index.files.insert(i, api);
                        index.stats.extracted += 1;
                    }
                    None => index.stats.failed += 1,
                }
            }
        }
        // Rows for files no longer indexed.
        let live: std::collections::HashSet<&str> = wanted
            .iter()
            .map(|&i| corpus.files[i].path.as_str())
            .collect();
        for path in cached.keys().filter(|p| !live.contains(p.as_str())) {
            tx.execute("DELETE FROM api_files WHERE path = ?1", [path])?;
        }
        tx.commit()?;
    }
    index.stats.elapsed = start.elapsed();
    log::info!(
        "pulse api: {} cached, {} extracted, {} failed in {:?}",
        index.stats.cached,
        index.stats.extracted,
        index.stats.failed,
        index.stats.elapsed
    );
    Ok(index)
}
