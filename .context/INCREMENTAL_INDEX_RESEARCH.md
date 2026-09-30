# Incremental Index Updates — Research and Design

**Date:** 2026-09-29 · **Reflex:** 2.0.3 · **Status:** built on `feature/incremental-index`
(see "As built" below; the sections after it are the design as proposed)

When code and this file disagree, the code wins. File:line references are for 2.0.3
(`fc8da6b` + branch `feature/various-enhancements`).

---

## As built (2026-09-29)

Commits on `feature/incremental-index`: `cc920df`, `6798edd` (Stage 0), `9f342d6`
(manifest + `IndexSnapshot`), `29dfb43` (planning size), `112a82a` (delta +
tombstones), `d548268` (`update_paths`), `2a72054` (concurrency / cross-version
tests), `73eb7ff` (merge from the stores), then the two delta tiers.

What differs from the proposal below, and why:

- **Two delta tiers** (Stage 2 condition met). With one delta, every update rebuilt the
  whole delta: a 1-file `update_paths` with a 1000-file (11 MB) delta took 191–207 ms
  on Kubernetes (load 9–13; 128–134 ms of it writing the delta). Now a small
  **recent** segment is rebuilt by each update and folded into the delta past
  256 files or 1/16 of the delta byte limit: the same update takes 71 ms (load 15).
  Tombstones cover base and delta ids; `.dtomb` holds the dead delta postings'
  planning sizes. The live trigram count is kept from the trigrams an update touches
  (dead files' runs, the replaced and the new segment), not recounted.
- **One change-set publish** (`publish_delta`) for `rfx index` and `update_paths`: it
  writes only the named rows and folds the branch row and the statistics stamps into
  the meta.db transaction. The first `update_paths` (full lists into the `rfx index`
  code) took 260–500 ms; the change set brought it to ~135 ms, and loading only the
  named rows (index lookups; `files(walk_seq)` index for placement) to 73–103 ms at
  load 9.
- **Walk placement** by readdir order of the named paths' ancestors
  (`src/walk_order.rs`) and a binary search over `walk_seq` values, not cached
  listings of whole directories.
- **Resolver configs**: every walk saves the config file list
  (`resolver-configs.json`); `update_paths` re-parses those files (36 on Kubernetes)
  and checks the digest instead of walking.
- **Merge from the stores** (Stage 2): unchanged files' text comes from the published
  stores; the result is byte-identical to a fresh build (test). Kubernetes, 1500 files
  edited: 3.4 s against a 7.4 s cold build (load 8–9).
- **`is_dirty`** (decision 5, with the user): `update_paths` runs `git status` on the
  named paths only; the branch's dirty flag can stay set after a revert until the next
  `rfx index`.
- **Skip pointers**: not built. A live 1000-file delta against a fresh base of the
  same tree: all shapes −0.7 %, single shapes −9 % … +6 %, candidate phase +0.01 … +1.2 ms
  (comparing against the tree *before* the edit made some shapes look 9–33 % slower;
  that was the edit, not the delta).
- **Library path, final** (`f30bce6`): 51–53 ms per 1-file edit at load 10 — the path
  resolver stays in the process (keyed by the manifest's random `publish_id`), and
  `git status` of the named paths overlaps the store writes.
- **Nothing changed**: 0.19 s against a 0.15 s walk (+30 %): the synced branch
  (`statistics.synced_branch`) skips the branch-hash load and the full branch-row sync,
  `init()` skips the schema transaction on a cache this binary completed, the stored
  rows load during the walk, statistics come from the rows. The rest is the meta.db
  commit (8 ms, `synchronous=FULL`) and fixed process costs.

Tried and dropped:

- A reader retry race test with real publishes: the window between reading the
  manifest and opening its files is microseconds; 100k+ reader answers never hit it.
  The retry is covered by a unit test that swaps the manifest read instead.
- Treating an unreadable rewrite as "keep the old copy": a full run drops an
  unreadable file, so the delta path drops it too.

Found and fixed on the way: the schema-hash check was dead (read after `init()`
stamped it); `init()` reset `last_compaction` every run; `rfx stats` ran two
debug-only full scans. Changed because stable ids need it: symbol rows are keyed by the
stored bytes' hash (`files.hash`), not a branch row's.

---

## Why

Today any change makes `rfx index` redo the whole tree. Measured 2026-09-29:

| Tree | Cold index | After a 1-file edit | Nothing changed |
| --- | --- | --- | --- |
| Reflex, 407 files | 0.64 s | 0.66 s | 0.17 s |
| Kubernetes, 27,448 files (load avg 24) | 17.7 s | 27.7 s | 5.75 s |

The Kubernetes one-file log: 11.3 s re-reading and extracting every file, 9.7 s rewriting
every `files` row, 3.5 s re-extracting every dependency, then both stores written in full.
The symbol cache is wiped as well (282 rows → 0 on the Reflex clone), so the background
pass re-parses every file (Kubernetes: 3.4 s wall, 25 s CPU on an idle box).

This has been true since v0.2.0: every tag rebuilds from scratch on any change. Only the
"nothing changed" shortcut exists.

What incremental updates unlock:
- `rfx mcp` can keep its index fresh by itself (TODO Backlog §1 step A), which removes the
  agent's `check_index_status` and `index_project` turns.
- A useful in-process watcher; fast `rfx index` after small edits on large trees.
- A symbol cache that survives reindexing.
- Fresher `rfx serve` and faster Pulse regeneration.

Target: a one-file edit on Kubernetes becomes searchable in **under 100 ms**.

---

## How the index works today (facts)

**Two id spaces, joined only by path.**
- Trigram `file_id` = `content.bin` entry index. One serial loop assigns both
  (`src/indexer.rs` ~1096–1108); each writer returns its own list length
  (`src/trigram_build.rs` ~356–362, `src/content_store.rs` ~129–130). `OpenIndex::open`
  rejects a pair whose counts differ (`src/query/open_index.rs` ~147–154).
- These ids are dense and positional, `0..N-1`, in `ignore::WalkBuilder` order (never
  sorted, `src/indexer.rs` ~2453–2479). Adding or removing a file shifts later ids.
- `meta.db` `files.id` is a separate `INTEGER PRIMARY KEY AUTOINCREMENT` with
  `path UNIQUE` (`src/cache.rs` ~148–161). `file_branches`, `file_dependencies`
  (`file_id`, `resolved_file_id`), `file_exports` and `symbols` reference it with cascade
  or set-null foreign keys (`foreign_keys=ON`, `src/cache.rs` ~79).
- `INSERT OR REPLACE INTO files` (`src/cache.rs` ~1068–1083) gives every file a new id on
  every run, so the cascades delete symbols, other branches' `file_branches` rows and
  exports, and null importers' `resolved_file_id`.
- Path → id maps: `OpenIndex::file_id_for` (lazy, content ids, `open_index.rs` ~224–241);
  `branch_file_rows_on` (db ids, `cache.rs` ~947–985); `PathResolver` (db ids,
  `dependency.rs` ~54–76).

**`trigrams.bin` V4.** Header, a directory of 16-byte entries sorted by trigram, posting
data, then paths (`src/trigram.rs` ~26–37). A posting list is one block per file:
`varint(file_id delta)`, `varint(n_lines<<1|enc)`, then line deltas. No skip pointers:
`PostingCursor::seek` skips blocks by scanning varint bytes. Every decoded
`FileLocation` carries its `file_id`, so a tombstone filter is a cheap post-filter.

**Byte-copy merge** (`src/trigram_build.rs` ~469–631): partials must cover disjoint,
increasing file-id ranges (not checked); only each record's first file delta is rewritten.
Base posting lists are byte-compatible with partial records (same `encode_posting_list`).

**Query path.** `search_with_metadata` runs the search and the freshness snapshot in
parallel (`src/query/mod.rs` ~872–888). Candidates come from `TrigramIndex`
(`search_candidates`, `search_candidates_fold`, regex literal unions), are grouped per
file id, then `verify_files_streaming` reads content by id and sorts by path (~299–360),
so output order does not depend on ids. Readers keyed by content id:
`verify_files_streaming`; full scans over `0..content.file_count()` (~2788, ~2629);
`file_id_for` (symbol/AST paths ~1484, ~1766, ~2091, ~2209, ~2402);
`src/background_indexer.rs` ~584; `src/pulse/extract/mod.rs` ~62, ~208;
`src/pulse/extract/api_cache.rs` ~100; `src/cli/query.rs` ~606.

**Reopen.** `OpenIndex` is cached per canonical cache dir and reopens when the
(dev, inode, size, mtime) of `content.bin` or `trigrams.bin` changes (`open_index.rs`
~28–72, ~281–324). meta.db is not part of that fingerprint.

**Publish order today.** meta.db files transaction → deps transaction →
`trigrams.bin` rename → `content.bin` rename. Between the renames a new reader can open a
mismatched pair and get `CacheCorrupted`.

**Freshness.** Git mode: `git status` candidates + `dirty_at_index` + `git diff
indexed..HEAD`, each confirmed by stat then blake3 (`classify_one`, `mod.rs`
~4200–4247). Walk mode compares every file's fingerprint. `WorktreeChanges` lists are
capped at 100 per category, so they cannot feed an update; the uncapped candidate set
and `classify_one` can.

**Indexer steps** (`Indexer::index`): lock, prune deleted files (cascade), discover,
tsconfigs, fast path (only when counts match; still reads and hashes every file), batch
loop (read, hash, imports, trigram runs), files transaction (REPLACE every row),
dependency configs re-parsed for the whole tree, `replace_dependencies` per file
(DELETE + INSERT), exports insert-only (rely on the cascade), write both stores, stats.
Already per-file: `replace_dependencies`, `clear_dependencies`, `refresh_fingerprints`,
`delete_files_from_db`, symbol rows keyed by (db id, hash).

**Symbol pass** (`src/background_indexer.rs` ~588–646) skips files whose (db id, hash)
is cached. With stable db ids it would parse only changed files. Spawned only by
`rfx index`, not by MCP `index_project`.

**Watcher** (`src/watcher.rs`) collects changed paths but calls a full `Indexer::index`;
it ignores `.gitignore` and reports only a rename's first path; 15 s default debounce.

---

## Design

### Stage 0 — stable metadata (small; ships on its own)

1. **Upsert, never replace.** `INSERT … ON CONFLICT(path) DO UPDATE` for `files`, so db ids
   are stable. Symbols, `file_branches` and exports then survive a reindex.
2. **Detect changes without reading.** Compare (size, mtime_ns) with the stored
   fingerprint first; hash only mismatches (reuse `classify_one`). The fast path stops
   reading every file.
3. **Write only changed rows.** `files`, `file_branches`, fingerprints.
4. **Dependencies per changed file.** Re-extract changed files; delete their exports
   explicitly; re-resolve importers of added and deleted paths; run the whole dependency
   pass only when a resolver config changed (`tsconfig.json`, `go.mod`, `Cargo.toml`,
   `pom.xml`, `composer.json`, Python package configs).
5. **Symbol pass parses only changed files** (follows from stable ids). Spawn it from MCP
   `index_project` too.

Effect: removes the `files` rewrite, the dependency re-extraction and the full symbol
re-parse. The stores are still rebuilt from every file until stage 1.

### Stage 1 — one delta segment (the unlock)

A snapshot is:
- **base**: today's `trigrams.bin` + `content.bin`, ids `0..N-1`;
- **delta**: `trigrams.delta.bin` + `content.delta.bin` for added and modified files, ids
  `N..N+k-1`, built with the existing `TrigramIndexBuilder` / `ContentWriter` from an id
  offset;
- **tombstones**: a bitset of base ids that are deleted or superseded;
- **manifest**: generation, file names, counts and checksums, written by atomic rename —
  the single commit point. `OpenIndex` fingerprints the manifest.

Rules:
- Modified file: tombstone its base id, put the new version in the delta. Deleted:
  tombstone. Added: delta. Rename: delete + add.
- Keep **one** delta, rebuilt whole on every update from all paths changed since the last
  compaction. Cost is proportional to the changed set, and there is no segment stack.
- **One reader interface.** Put base + delta + tombstones behind one `IndexSnapshot`
  that every content-id reader uses: candidates = base (minus tombstones) ∪ delta;
  `file(id)` routes by id; `file_id_for` prefers the delta; iteration skips tombstones.
  This replaces the scattered `content.file_count()` loops listed above.
- **Compaction:** when the delta passes a threshold (for example 2,000 files or 5% of
  base bytes), after an idle period, or on `rfx index --force`, run the existing full
  build in the background under `index.lock`, then publish a manifest with an empty delta.
- **Crash safety:** write generation-suffixed delta files (tmp + fsync + rename), then
  rename the manifest. Delete unreferenced generations on the next write. The manifest
  also fixes today's two-rename race.
- **Freshness:** update fingerprint rows per changed file; invalidate the freshness memo
  at publish.
- **Inputs:** the watcher passes event paths (both paths of a rename), filtered by
  `PathPolicy`; a periodic reconcile uses the uncapped freshness candidate set to catch
  missed events. `rfx index` (CLI) discovers and stat-compares, then takes the delta
  path when the change set is small.
- **Versioning:** the manifest and delta files are a format change: add them to
  `build.rs`'s schema hash (and add `src/trigram_build.rs`, missing today). A reader that
  predates the manifest must not read a base while ignoring a live delta; a version or
  header bump makes it degrade instead ("readers degrade, writers refuse"). Mixed-version
  setups exist (Hearth runs its own `rfx mcp`).

Expected one-file update via the watcher: read and extract one file, rebuild a small delta,
upsert a few rows, rename the manifest — milliseconds.

### Stage 2 — only if needed

Several deltas with tiered merging (LSM style) if one rebuilt delta grows costly; skip
pointers in posting lists; compaction that copies unchanged content and filters postings
instead of re-reading files.

---

## Correctness bar

- **Property test:** random sequences of add / edit / delete / rename (and a branch
  switch). After each step, a fixed battery — literal, whole-word, regex, `(?i)`,
  `--symbols`, `find_references`, count, `paths` — must equal a fresh full index of the
  same tree; the path-joined dependency and export dump (as in
  `tests/dependency_equivalence.rs`) and symbols per path must match too. Ids will
  differ; compare by path.
- **Crash test:** kill between the delta write and the manifest rename → the old snapshot
  is served.
- **Concurrency:** a reader during publish sees the old or the new snapshot, never a mix.
- **Performance:** query overhead of the delta ≤ 5% on `tests/latency_budget.rs`; a
  one-file update < 100 ms on a Kubernetes scratch clone.

## Effort (rough)

- Stage 0: about 3–5 days with tests.
- Stage 1: about 2–3 weeks: the `IndexSnapshot` refactor across the readers, delta build
  and publish, compaction, watcher and MCP integration, property tests.
- Main risks: id routing across the content readers; complete dependency re-resolution;
  branch-switch semantics; mixed-version readers.
