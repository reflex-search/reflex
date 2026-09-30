//! `meta.db` writes of an incremental index run.
//!
//! A run writes only what changed: the rows of added, modified and touched files,
//! the rows of deleted files (by id), the walk positions that moved, the dirty
//! flags that flipped, and this branch's hash rows. Every function here runs on a
//! connection inside a transaction the caller holds, so one run commits once.

use std::collections::HashMap;

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension};

use crate::cache::{FileRow, WALK_SEQ_GAP};

/// What `meta.db` holds about one indexed file.
#[derive(Debug, Clone)]
pub struct StoredFile {
    pub id: i64,
    pub size: u64,
    pub mtime_ns: i64,
    pub hash: String,
    pub walk_seq: i64,
    pub dirty: bool,
    /// For the statistics of a run that changes nothing.
    pub language: String,
    pub line_count: usize,
}

impl StoredFile {
    /// Whether `md` describes the file as it was indexed (same rule as the
    /// freshness check: an unknown mtime is never equal).
    pub fn stat_matches(&self, st: &crate::cache::FileStat) -> bool {
        self.mtime_ns != 0 && self.size == st.size() && self.mtime_ns == st.mtime_ns()
    }
}

/// Every `files` row, by path.
pub fn load_stored_files(conn: &Connection) -> Result<HashMap<String, StoredFile>> {
    let mut stmt = conn.prepare(
        "SELECT path, id, size, mtime_ns, hash, walk_seq, dirty_at_index, language, line_count
             FROM files",
    )?;
    let rows = stmt.query_map([], stored_file)?;
    rows.collect::<Result<HashMap<_, _>, _>>()
        .context("Failed to read files rows")
}

fn stored_file(r: &rusqlite::Row<'_>) -> rusqlite::Result<(String, StoredFile)> {
    Ok((
        r.get::<_, String>(0)?,
        StoredFile {
            id: r.get(1)?,
            size: r.get::<_, i64>(2)? as u64,
            mtime_ns: r.get(3)?,
            hash: r.get(4)?,
            walk_seq: r.get(5)?,
            dirty: r.get::<_, i64>(6)? != 0,
            language: r.get(7)?,
            line_count: r.get::<_, i64>(8)? as usize,
        },
    ))
}

/// The `files` rows at each of `paths` or under it (a directory): what a library
/// update needs, by index lookups.
pub fn load_rows_under(conn: &Connection, paths: &[String]) -> Result<HashMap<String, StoredFile>> {
    const COLUMNS: &str =
        "path, id, size, mtime_ns, hash, walk_seq, dirty_at_index, language, line_count";
    let mut exact = conn.prepare_cached(&format!("SELECT {COLUMNS} FROM files WHERE path = ?"))?;
    // `p/` < every path under `p` < `p0` ('0' follows '/').
    let mut under = conn.prepare_cached(&format!(
        "SELECT {COLUMNS} FROM files WHERE path > ? AND path < ?"
    ))?;
    let mut out = HashMap::new();
    for p in paths {
        for row in exact.query_map([p], stored_file)? {
            let (path, file) = row?;
            out.insert(path, file);
        }
        for row in under.query_map([format!("{p}/"), format!("{p}0")], stored_file)? {
            let (path, file) = row?;
            out.insert(path, file);
        }
    }
    Ok(out)
}

/// Number of `files` rows.
pub fn count_files(conn: &Connection) -> Result<usize> {
    Ok(conn.query_row("SELECT COUNT(*) FROM files", [], |r| r.get::<_, i64>(0))? as usize)
}

/// Rows of `files` in `walk_seq` order, probed by value: the first row at or after
/// `seq` (or, with `before`, the last row before it) whose id is not in `skip`.
pub fn row_near_seq(
    conn: &Connection,
    seq: i64,
    before: bool,
    skip: &std::collections::HashSet<i64>,
) -> Result<Option<(i64, String, i64)>> {
    let sql = if before {
        "SELECT id, path, walk_seq FROM files WHERE walk_seq < ? ORDER BY walk_seq DESC LIMIT 16"
    } else {
        "SELECT id, path, walk_seq FROM files WHERE walk_seq >= ? ORDER BY walk_seq LIMIT 16"
    };
    let mut stmt = conn.prepare_cached(sql)?;
    let mut from = seq;
    loop {
        let rows: Vec<(i64, String, i64)> = stmt
            .query_map([from], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .collect::<Result<_, _>>()?;
        let Some(last) = rows.last().map(|r| r.2) else {
            return Ok(None);
        };
        let full = rows.len() == 16;
        if let Some(row) = rows.into_iter().find(|r| !skip.contains(&r.0)) {
            return Ok(Some(row));
        }
        if !full {
            return Ok(None);
        }
        // Sixteen skipped rows in a row: continue past them.
        from = if before { last } else { last + 1 };
    }
}

/// Upsert `rows` (keeping each path's id); returns their ids, in order.
pub fn upsert_files(conn: &Connection, rows: &[FileRow], now: i64) -> Result<Vec<i64>> {
    let mut stmt = conn.prepare_cached(
        "INSERT INTO files
             (path, last_indexed, language, line_count, size, mtime_ns, hash,
              dirty_at_index, walk_seq)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(path) DO UPDATE SET
             last_indexed = excluded.last_indexed,
             language = excluded.language,
             line_count = excluded.line_count,
             size = excluded.size,
             mtime_ns = excluded.mtime_ns,
             hash = excluded.hash,
             dirty_at_index = excluded.dirty_at_index,
             walk_seq = excluded.walk_seq
         RETURNING id",
    )?;
    let mut ids = Vec::with_capacity(rows.len());
    for row in rows {
        let id: i64 = stmt.query_row(
            rusqlite::params![
                row.path,
                now,
                row.language,
                row.line_count as i64,
                row.size as i64,
                row.mtime_ns,
                row.hash,
                row.dirty as i64,
                row.walk_seq,
            ],
            |r| r.get(0),
        )?;
        ids.push(id);
    }
    Ok(ids)
}

/// `UPDATE files SET walk_seq` for each `(id, walk_seq)`.
pub fn set_walk_seqs(conn: &Connection, updates: &[(i64, i64)]) -> Result<()> {
    let mut stmt = conn.prepare_cached("UPDATE files SET walk_seq = ? WHERE id = ?")?;
    for (id, seq) in updates {
        stmt.execute(rusqlite::params![seq, id])?;
    }
    Ok(())
}

/// `UPDATE files SET dirty_at_index` for each `(id, dirty)`.
pub fn set_dirty_flags(conn: &Connection, updates: &[(i64, bool)]) -> Result<()> {
    let mut stmt = conn.prepare_cached("UPDATE files SET dirty_at_index = ? WHERE id = ?")?;
    for (id, dirty) in updates {
        stmt.execute(rusqlite::params![*dirty as i64, id])?;
    }
    Ok(())
}

/// `UPDATE files SET size, mtime_ns, dirty_at_index` for files whose bytes are
/// unchanged (a `touch`, or an edit reverted byte for byte).
pub fn refresh_stats(conn: &Connection, updates: &[(i64, u64, i64, bool)]) -> Result<()> {
    let mut stmt = conn.prepare_cached(
        "UPDATE files SET size = ?, mtime_ns = ?, dirty_at_index = ? WHERE id = ?",
    )?;
    for (id, size, mtime_ns, dirty) in updates {
        stmt.execute(rusqlite::params![*size as i64, mtime_ns, *dirty as i64, id])?;
    }
    Ok(())
}

/// Delete the rows of `ids`; the cascades remove their branch, dependency,
/// export and symbol rows and null importers' resolved ids.
pub fn delete_files(conn: &Connection, ids: &[i64]) -> Result<()> {
    let mut stmt = conn.prepare_cached("DELETE FROM files WHERE id = ?")?;
    for id in ids {
        stmt.execute([id])?;
    }
    Ok(())
}

/// `statistics` key: the branch whose `file_branches` rows the last completed run
/// synced with `files`. Written in the transaction that syncs them; a run on that
/// branch then needs to touch only the rows it rewrites.
pub const SYNCED_BRANCH_KEY: &str = "synced_branch";

/// Make `branch_id`'s rows name every file with its current hash: inserts the
/// missing ones and updates the stale ones, leaving the rest alone. Records
/// `branch` as the synced branch.
pub fn sync_branch_rows(
    conn: &Connection,
    branch_id: i64,
    branch: &str,
    now: i64,
) -> Result<usize> {
    let changed = conn.execute(
        "INSERT OR REPLACE INTO file_branches (file_id, branch_id, hash, last_indexed)
         SELECT f.id, ?1, f.hash, ?2 FROM files f
         WHERE NOT EXISTS (
             SELECT 1 FROM file_branches fb
             WHERE fb.file_id = f.id AND fb.branch_id = ?1 AND fb.hash = f.hash)",
        rusqlite::params![branch_id, now],
    )?;
    set_statistic(conn, SYNCED_BRANCH_KEY, branch, now)?;
    Ok(changed)
}

/// Point `branch_id`'s rows of the given files at their hashes: `(file id, hash)`.
/// A run on the branch that last synced every row (see [`sync_branch_rows`]) needs
/// only the rows it rewrote.
pub fn set_branch_rows(
    conn: &Connection,
    branch_id: i64,
    rows: &[(i64, &str)],
    now: i64,
) -> Result<()> {
    let mut stmt = conn.prepare_cached(
        "INSERT OR REPLACE INTO file_branches (file_id, branch_id, hash, last_indexed)
         VALUES (?, ?, ?, ?)",
    )?;
    for (id, hash) in rows {
        stmt.execute(rusqlite::params![id, branch_id, hash, now])?;
    }
    Ok(())
}

/// `statistics.value` for `key`, if set.
pub fn get_statistic(conn: &Connection, key: &str) -> Result<Option<String>> {
    Ok(conn
        .query_row("SELECT value FROM statistics WHERE key = ?", [key], |r| {
            r.get(0)
        })
        .optional()?)
}

/// Set `statistics.value` for `key`.
pub fn set_statistic(conn: &Connection, key: &str, value: &str, now: i64) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO statistics (key, value, updated_at) VALUES (?, ?, ?)",
        rusqlite::params![key, value, now],
    )?;
    Ok(())
}

/// The walk position each file should have, given each file's stored position
/// (`None` for a new file) in the current walk order.
///
/// Changes as few positions as possible: the longest run of stored positions that
/// is already increasing is kept, and every other file gets a value between its
/// kept neighbours. When a gap is too narrow, every file is renumbered at
/// [`WALK_SEQ_GAP`] spacing (what a full build writes).
pub fn plan_walk_seq(stored: &[Option<i64>]) -> Vec<i64> {
    let n = stored.len();
    // The usual case: every file has a position and they already increase.
    if stored.iter().all(Option::is_some) && stored.windows(2).all(|w| w[0] < w[1]) {
        return stored.iter().map(|v| v.expect("checked")).collect();
    }
    let renumber = || (0..n).map(|i| i as i64 * WALK_SEQ_GAP).collect::<Vec<_>>();

    // Longest strictly increasing subsequence of the stored values (patience).
    let known: Vec<(usize, i64)> = stored
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.map(|v| (i, v)))
        .collect();
    if known.is_empty() {
        return renumber();
    }
    let mut tails: Vec<usize> = Vec::new(); // index into `known` of each pile's top
    let mut prev: Vec<Option<usize>> = vec![None; known.len()];
    for k in 0..known.len() {
        let v = known[k].1;
        let pos = tails.partition_point(|&t| known[t].1 < v);
        if pos > 0 {
            prev[k] = Some(tails[pos - 1]);
        }
        if pos == tails.len() {
            tails.push(k);
        } else {
            tails[pos] = k;
        }
    }
    let mut keep = vec![false; n];
    let mut cur = tails.last().copied();
    while let Some(k) = cur {
        keep[known[k].0] = true;
        cur = prev[k];
    }

    let mut out: Vec<i64> = vec![0; n];
    let mut i = 0;
    let mut lo: Option<i64> = None; // value of the last kept file before `i`
    while i < n {
        if keep[i] {
            let v = stored[i].expect("kept files have a value");
            out[i] = v;
            lo = Some(v);
            i += 1;
            continue;
        }
        // A run of files to place, up to the next kept file (or the end).
        let start = i;
        while i < n && !keep[i] {
            i += 1;
        }
        let count = (i - start) as i64;
        let hi = (i < n).then(|| stored[i].expect("kept files have a value"));
        let (base, step) = match (lo, hi) {
            (Some(lo), Some(hi)) => {
                let step = (hi - lo) / (count + 1);
                if step < 1 {
                    return renumber();
                }
                (lo, step)
            }
            (Some(lo), None) => (lo, WALK_SEQ_GAP),
            (None, Some(hi)) => match hi.checked_sub((count + 1) * WALK_SEQ_GAP) {
                Some(base) => (base, WALK_SEQ_GAP),
                None => return renumber(),
            },
            (None, None) => return renumber(),
        };
        for (j, slot) in out[start..i].iter_mut().enumerate() {
            match base.checked_add((j as i64 + 1) * step) {
                Some(v) => *slot = v,
                None => return renumber(),
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn increasing(v: &[i64]) -> bool {
        v.windows(2).all(|w| w[0] < w[1])
    }

    #[test]
    fn unchanged_order_keeps_every_value() {
        let stored = [Some(0), Some(10), Some(20)];
        assert_eq!(plan_walk_seq(&stored), vec![0, 10, 20]);
    }

    #[test]
    fn a_new_file_goes_between_its_neighbours() {
        let g = WALK_SEQ_GAP;
        let stored = [Some(0), None, Some(g), Some(2 * g)];
        let out = plan_walk_seq(&stored);
        assert!(increasing(&out));
        assert_eq!(out[0], 0);
        assert_eq!(out[2], g);
        assert_eq!(out[3], 2 * g);
    }

    #[test]
    fn new_files_at_either_end() {
        let g = WALK_SEQ_GAP;
        let stored = [None, None, Some(0), Some(g), None];
        let out = plan_walk_seq(&stored);
        assert!(increasing(&out), "{out:?}");
        assert_eq!(&out[2..4], &[0, g]);
    }

    #[test]
    fn a_moved_file_is_the_only_one_renumbered() {
        let g = WALK_SEQ_GAP;
        // The file at position 3 used to come first.
        let stored = [Some(g), Some(2 * g), Some(3 * g), Some(0), Some(4 * g)];
        let out = plan_walk_seq(&stored);
        assert!(increasing(&out), "{out:?}");
        let changed = out
            .iter()
            .zip(&stored)
            .filter(|(a, b)| Some(**a) != **b)
            .count();
        assert_eq!(changed, 1);
    }

    #[test]
    fn a_full_gap_renumbers() {
        let stored = [Some(0), None, None, Some(2)];
        let out = plan_walk_seq(&stored);
        assert!(increasing(&out));
        assert_eq!(
            out,
            vec![0, WALK_SEQ_GAP, 2 * WALK_SEQ_GAP, 3 * WALK_SEQ_GAP]
        );
    }

    #[test]
    fn random_orders_always_come_out_increasing() {
        let mut state = 12345u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as i64
        };
        for _ in 0..200 {
            let n = (next() % 40) as usize;
            let stored: Vec<Option<i64>> = (0..n)
                .map(|_| {
                    if next() % 4 == 0 {
                        None
                    } else {
                        Some(next() % 1000 * WALK_SEQ_GAP / 7)
                    }
                })
                .collect();
            // Distinct stored values, as in the table.
            let mut seen = std::collections::HashSet::new();
            let stored: Vec<Option<i64>> = stored
                .into_iter()
                .map(|v| v.filter(|x| seen.insert(*x)))
                .collect();
            let out = plan_walk_seq(&stored);
            assert_eq!(out.len(), n);
            assert!(increasing(&out), "{stored:?} -> {out:?}");
        }
    }
}
