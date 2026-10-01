# Dependency resolution research

How imports become graph edges, why package-based languages were nearly absent from the
graph until 2.1.0, and what replaced the old resolvers. Branch
`fix/package-import-resolution`, 2026-09-30. All numbers are from rfx 2.1.0 (before)
and that branch (after), on scratch copies of the corpora, one machine.

## Trigger

An agent asked Reflex for "islands" in Kubernetes. It got 27,313 islands and 27,431
unused files, and presented 7 multi-file islands as structure. They were accidents of
file names: the Go resolver turned `k8s.io/kubernetes/pkg/kubelet/server` into the one
file `pkg/kubelet/server.go`.

## The measure

The measure is the share of **internal** imports that reach an indexed file. Run it with
a read-only sqlite query over `.reflex/meta.db`, grouped by `files.language`. After the
change, "reach" means a row has `resolved_file_id` set, or its `resolved_package` has a
member other than the importer. `DependencyIndex::internal_resolution_by_language` is
the same query, so `analyze` reports it.

**Search is not affected.** On 2026-09-30, 17 searches over 7 repos were checked:
whole-identifier, `--contains`, `-i` and `--regex` on tokio, django, react, dotnet,
neo4j, laravel and rails. Every `--count` was equal to `rg -c`. `--count` counts lines,
not matches.

## Before → after

Each row is the same corpus copy, indexed fresh by each binary.

| Corpus | Language | Resolved, 2.1.0 | Resolved, branch | Islands | Unused files |
|---|---|---|---|---|---|
| kubernetes | Go | 84 / 52,472 (0.2 %) | 48,997 / 48,997 (100 %) | 27,303 → 1,368 | 27,371 → 4,802 |
| neo4j | Java | 356 / 59,658 (0.6 %) | 58,926 / 59,658 (98.8 %) | 11,423 → 655 | 11,753 → 1,682 |
| dotnet/runtime | C# | 281 / 23,051 (1.2 %) | 18,911 / 23,051 (82.0 %) | 55,938 → 16,346 | 57,759 → 14,989 |
| django | Python | 1,662 / 8,673 (19.2 %) | 8,672 / 8,673 (100 %) | 4,327 → 789 | 2,919 → 416 |
| ktorio | Kotlin | 1 / 15 | 15 / 15 | | |

Notes on the table:

- **Kubernetes Go:** the internal count fell from 52,472 to 48,997. The other 3,475
  imports belong to no module of the repo (`k8s.io/klog`, `k8s.io/utils`), so they are
  now External.
- **File-based languages are unchanged:** TS/JS 91–100 %, Rust 86 %, Zig 100 %, Ruby
  67 %, PHP 58 %, C/C++ 64 %.
- **Django JavaScript** is 13 / 545 (vendored xregexp sources). It now gets the
  low-resolution warning.

Costs, Kubernetes:

| | 2.1.0 | branch |
|---|---|---|
| `rfx index` | 6.0 s | 6.2 s |
| `rfx analyze` | 0.0 s | 2.2 s |
| meta.db | 58 MB | 65 MB |

On dotnet, both binaries took between 37 and 50 s to index. The variation came from the
symbol pass of the previous run still using the CPU, not from either binary.

## Why each resolver failed

Each resolver produced ONE guessed path per import, and `resolve_import` made one
`PathResolver` lookup (`src/dependency_resolve.rs`).

- **Go.** The resolver returned only `<sub>.go`. Its second candidate,
  `<sub>/<pkg>.go`, came after `candidates.into_iter().next()`, so it was never
  tried. Modules matched with `starts_with`, with no `/` boundary. The first module in
  directory order won, so `k8s.io/apiserver` could strip to `server/...` under
  `k8s.io/api`. Reclassify also marked any same-domain import as Internal (all of
  `k8s.io/*`).
- **Java.** The package name was the first `<groupId>` in a `pom.xml`. In neo4j every
  module is `org.neo4j`, so every import bound to the first module in walk order. The
  resolver then guessed `<that module>/src/main/java/<pkg>/<Cls>.java`. Wildcards became
  `a/b.java`.
- **Kotlin.** The same as Java, but with `src/main/kotlin` only.
- **C#.** The resolver turned `A.B` into `A/B.cs`. A `using` names a namespace, which is
  not a file.
- **Python.** The resolver tried only `a/b.py`, never `a/b/__init__.py`. With the package
  at the repo root, the path started with `/`. `normalize_path_for_lookup` then fell
  back to a filename suffix match, which was usually ambiguous. Ambiguous meant `None`.

## Design: package keys, expanded when the graph is read

```
file_dependencies.resolved_package  -- go:<dir> | jvm:<package> | cs:<namespace>
file_dependencies.resolved_member   -- a JVM class or top-level name; NULL = whole package
package_members(package, member, file_id)   -- per-file fact, ON DELETE CASCADE
VIEW import_edges(dep_id, src, dst, import_type, package)
```

- **Membership.** A file's membership is written when the file is extracted, through
  `DependencyWriter::replace_members`.
  - Go: the file's directory, unless it is a `_test.go` file.
  - Java and Kotlin: the `package` line, plus the file stem and every column-0
    top-level name.
  - C#: every namespace the file declares, read in the same parse as the usings.
- **Incremental updates stay correct.** An import's key depends only on its text and the
  module config. A config change already forces a full run through
  `RESOLVER_DIGEST_KEY`. Adding, deleting or moving a member file changes only its own
  `package_members` rows, so `reresolve` skips rows that have a package key. Tests:
  `incremental_add_and_delete_go_file_in_package_matches_full_build` and
  `changing_package_line_moves_edges_incrementally`.
- **Old caches.** `CREATE TABLE IF NOT EXISTS` does not add columns.
  `migrate_dependency_columns` adds them before the new index is created. Without it,
  the first update of a 2.1.0 cache failed: "no such column: resolved_package".
- **The view.** `import_edges` is dropped and recreated on every schema init. A view
  holds no data, and `IF NOT EXISTS` would keep an older definition.

### Rejected

- **One row per target file.** On Kubernetes that is about 420k rows instead of 52k
  (8.6 files per import on average; the largest package has 206 files). `reresolve` can
  only UPDATE one value per row, so the rows would go stale when a file is added to a
  package.
- **A single representative file per package.** The other files of the package would
  show up as unused.
- **A cap on fan-out.** It drops edges and says nothing.
- **Classifying C# usings by the namespaces the repo declares.** Classification runs
  before any other file's namespaces are known. Instead, the rate leaves out usings
  whose root namespace no file declares (NuGet packages).

### Graph rules that came with it

- **Hotspots** count distinct importers (`COUNT(DISTINCT src)`).
- **Go package siblings** are one unit for islands and unused files. A `package main`
  file such as `flags.go` beside `main.go` is not unused.
- **Cycles** leave out whole-namespace C# edges. As file edges they produced 11,130
  cycles on dotnet/runtime; with this rule the count is 82 (2.1.0 reported 89).
- **The cycle and island searches** are iterative, with the same visit order as before.
  A real Go or Java graph is thousands of files deep.
- **Text, lock and generated files** are not graph nodes.
- **The warning** fires for a language with at least 100 internal imports and under
  50 % of them resolved (`LOW_RESOLUTION_*` in `src/dependency.rs`).

## Vendored code (2026-10-01)

**Trigger.** After the package fix, an agent reported 577 Kubernetes islands, 558 of them
in `vendor/`, and called them "normal". Kubernetes commits `vendor/` (5,247 tracked files,
`vendor/modules.txt`). Project code imports vendored packages 7,457 times, all External
and unresolved, so 0 edges reached `vendor/`. Each vendored Go package (a sibling unit)
was an island. A repo that gitignores `vendor/` has none of this: Reflex respects
`.gitignore`, so it only meets vendored code a repo commits.

**Decision (user, 2026-10-01): searchable, not in the graph.** Vendored files stay in the
trigram index (ripgrep searches them). They are not graph nodes, and imports of them stay
External. The result is the graph of the same repo with `vendor/` gitignored.

Rejected:
- **Leave vendored code out of the index.** It breaks ripgrep parity; on Kubernetes that
  is 19 % of the indexed files. `[index] exclude` already does it for a user who wants it.
- **Link imports into vendored files.** It needs a resolver per ecosystem (Go
  `modules.txt`, Composer autoload, Node `package.json`, Cargo, gem lib paths,
  site-packages, Zig, C include paths). The gain is `deps --reverse` on a library file;
  a search for the import string (`"k8s.io/klog/v2"`) answers the same question. Hotspots
  and islands would need vendored nodes hidden anyway.
- **Directory names for every language.** Kubernetes `third_party/forked/` is Go code the
  module imports by its own path. Go, PHP, Rust and Ruby define vendoring with a marker,
  so they get marker rules only; Java and Kotlin get only `third_party` names, because a
  package named `external` or `vendor` is common.

**Rules** (`src/vendor.rs`, in order): `[index.vendored] patterns`; marker roots (Go
`vendor/modules.txt`, `vendor/composer/installed.json`, `.cargo-checksum.json`,
`pyvenv.cfg`, `build.zig.zon` `.path` dependencies); dependency directories
(`node_modules`, `bower_components`, `site-packages`, `dist-packages`,
`ruby/<version>/{gems,specifications,...}`); per-language names (`vendor_dir_names`).
- `files.vendored` is set by `refresh_vendored` (`src/indexer.rs`): every row when
  `full_deps`, else the written rows. Markers are resolver configs; their paths (not
  bytes: `go mod vendor` rewrites `modules.txt`) and the patterns are in the digest.
- `.cargo-checksum.json` is a dotfile the config walk skips: it is found by a stat beside
  each `Cargo.toml`.
- Configs under a marker root or dependency directory are dropped. Before, a vendored
  `Cargo.toml` made its crate Internal (`use serde::…` resolved to a wrong `src/de.rs`),
  and a vendored gemspec made `require 'rack'` Internal.
- `vendored_file_count` counts code files only (`modules.txt` is a vendored text file).

**Measured** (branch before → after, same copies, release builds):

| Corpus | Vendored code files | Islands | Unused files | Notes |
|---|---|---|---|---|
| kubernetes | 4,241 | 1,368 → 530 | 4,802 → 579 | 838 islands were all-vendored; Go 48,997/48,997 still |
| dotnet/runtime | 1,052 | 16,346 → 16,087 | 14,989 → 14,440 | `src/native/external/`; 155 vendored hotspots gone |
| django | 63 | 789 → 726 | 416 → 355 | admin `static/admin/js/vendor/`; the JavaScript warning (13/545) was vendored xregexp and is gone |
| laravel-immutable-model, `vendor/` committed | 7,861 | 7,907 → 47 | 7,860 → 1 | PHP rate 71/12,431 → 71/436 (the rest was vendored code) |
| tokio + `cargo vendor` | 7,982 | 2,054 → 110 | 1,280 → 155 | cycles 387 → 51; every answer identical to plain tokio |
| tokio, react, rails | 0 | unchanged | unchanged | no row changed |

- On every corpus, no project file's unused, island, hotspot or cycle answer changed,
  except in tokio + `cargo vendor`: 5 hotspot counts fell by 1 (edges from vendored
  crates into tokio) and one false cycle went (`signal/mod.rs` ↔ `signal/windows.rs`,
  made by the vendored-crate misclassification; plain tokio never had it).
- `rfx analyze` on Kubernetes: 3.90 s → 4.03 s on a loaded machine (2.2 s on 09-30); the
  view joins `files` twice. meta.db 65 → 69 MB.

## Open (see TODO.md)

- **C#:** 18 % of dotnet's internal usings do not resolve. Causes: `using static A.B.C`
  (a type, keyed as a namespace), aliases (stored as two rows), and `global using`
  (untested).
- **Other file-based languages:** PHP 58 %, Ruby 67 % and C/C++ 64 % were not
  investigated.
- **`analyze` speed:** the summary takes 2.2 s on Kubernetes. Each analysis loads
  `import_edges` on its own.
- **`rfx snapshot diff`:** hotspots and islands have been empty since stable ids. This is
  unrelated to this change, and found during it.
