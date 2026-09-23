//! Dependency-phase equivalence: the rows `rfx index` writes to `file_dependencies`
//! and `file_exports` must not change when the resolution machinery changes.
//!
//! The corpus snapshot was generated on the SQLite-per-lookup implementation
//! (2.0.0) and pins the output of the in-memory `PathResolver` that replaced it.
//! The workspace test covers the relative-import shapes the corpus lacks.

mod test_helpers;

use reflex::{CacheManager, IndexConfig, Indexer};
use std::fs;
use std::path::Path;
use tempfile::TempDir;

/// Every dependency row, joined to paths, in a stable order.
fn dump_dependencies(root: &Path) -> String {
    let db = root.join(".reflex").join("meta.db");
    let conn = reflex::cache::open_meta_db(&db).expect("open meta.db");
    let mut stmt = conn
        .prepare(
            "SELECT f.path, fd.imported_path, r.path, fd.import_type, fd.line_number, fd.imported_symbols
             FROM file_dependencies fd
             JOIN files f ON f.id = fd.file_id
             LEFT JOIN files r ON r.id = fd.resolved_file_id
             ORDER BY f.path, fd.line_number, fd.imported_path, fd.import_type",
        )
        .unwrap();
    let rows = stmt
        .query_map([], |row| {
            Ok(format!(
                "{} | {} | {} | {} | {} | {}",
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?
                    .unwrap_or_else(|| "-".into()),
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, Option<String>>(5)?
                    .unwrap_or_else(|| "-".into()),
            ))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    rows.join("\n")
}

/// Every export row, joined to paths, in a stable order.
fn dump_exports(root: &Path) -> String {
    let db = root.join(".reflex").join("meta.db");
    let conn = reflex::cache::open_meta_db(&db).expect("open meta.db");
    let mut stmt = conn
        .prepare(
            "SELECT f.path, fe.exported_symbol, fe.source_path, r.path, fe.line_number
             FROM file_exports fe
             JOIN files f ON f.id = fe.file_id
             LEFT JOIN files r ON r.id = fe.resolved_source_id
             ORDER BY f.path, fe.line_number, fe.source_path, fe.exported_symbol",
        )
        .unwrap();
    let rows = stmt
        .query_map([], |row| {
            Ok(format!(
                "{} | {} | {} | {} | {}",
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?
                    .unwrap_or_else(|| "*".into()),
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?
                    .unwrap_or_else(|| "-".into()),
                row.get::<_, i64>(4)?,
            ))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    rows.join("\n")
}

#[test]
fn corpus_dependency_rows_are_stable() {
    let corpus = test_helpers::setup_corpus();
    insta::assert_snapshot!("corpus_dependencies", dump_dependencies(corpus));
    insta::assert_snapshot!("corpus_exports", dump_exports(corpus));
}

fn index(root: &Path) {
    let cache = CacheManager::new(root);
    Indexer::new(cache, IndexConfig::default())
        .index(root, false)
        .expect("index");
}

/// Relative imports across languages, an underscore filename, and a duplicate
/// basename in two directories (the ambiguous-suffix case).
fn workspace() -> TempDir {
    let t = TempDir::new().unwrap();
    let r = t.path();
    fs::create_dir_all(r.join("src/util")).unwrap();
    fs::create_dir_all(r.join("c/util")).unwrap();
    fs::create_dir_all(r.join("web/a")).unwrap();
    fs::create_dir_all(r.join("web/b")).unwrap();

    fs::write(
        r.join("Cargo.toml"),
        "[package]\nname = \"ws\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::write(
        r.join("src/lib.rs"),
        "mod util;\nmod a_b;\nuse crate::util::helper;\nuse serde::Serialize;\npub fn f() { helper(); }\n",
    )
    .unwrap();
    fs::write(r.join("src/util.rs"), "pub fn helper() {}\n").unwrap();
    fs::write(r.join("src/a_b.rs"), "pub fn ab() {}\n").unwrap();
    fs::write(r.join("src/aXb.rs"), "pub fn axb() {}\n").unwrap();

    fs::write(
        r.join("c/main.c"),
        "#include <stdio.h>\n#include \"util/h.h\"\n#include \"missing.h\"\nint main(void) { return 0; }\n",
    )
    .unwrap();
    fs::write(r.join("c/util/h.h"), "#define H 1\n").unwrap();

    fs::write(
        r.join("web/a/index.ts"),
        "import { x } from './b';\nimport { y } from '../b/shared';\nimport fs from 'fs';\nexport { x } from './b';\nexport const z = 1;\n",
    )
    .unwrap();
    fs::write(r.join("web/a/b.ts"), "export const x = 1;\n").unwrap();
    fs::write(r.join("web/b/shared.ts"), "export const y = 2;\n").unwrap();
    fs::write(r.join("web/b/b.ts"), "export const dup = 1;\n").unwrap();
    t
}

#[test]
fn workspace_dependency_rows_resolve_relative_imports() {
    let t = workspace();
    index(t.path());
    let deps = dump_dependencies(t.path());
    let exports = dump_exports(t.path());

    // Resolved rows the resolver must produce.
    assert!(
        deps.contains("web/a/index.ts | ./b | web/a/b.ts | internal"),
        "TS relative import not resolved:\n{deps}"
    );
    assert!(
        deps.contains("web/a/index.ts | ../b/shared | web/b/shared.ts | internal"),
        "TS parent-relative import not resolved:\n{deps}"
    );
    assert!(
        deps.contains("src/lib.rs | crate::util::helper | src/util.rs | internal"),
        "Rust crate:: import not resolved:\n{deps}"
    );
    assert!(
        exports.contains("web/a/index.ts | x | ./b | web/a/b.ts"),
        "TS re-export not resolved:\n{exports}"
    );
    // External and stdlib rows are stored unresolved.
    assert!(deps.contains("web/a/index.ts | fs | - | "), "{deps}");
    assert!(deps.contains("c/main.c | stdio.h | - | "), "{deps}");
    // A missing target stays unresolved rather than matching something else.
    assert!(
        deps.contains("c/main.c | missing.h | - | internal"),
        "{deps}"
    );
}

#[test]
fn reindex_after_removing_an_import_drops_its_row() {
    let t = workspace();
    index(t.path());
    let before = dump_dependencies(t.path());
    assert!(
        before.contains("src/lib.rs | crate::util::helper |"),
        "{before}"
    );

    fs::write(
        t.path().join("src/lib.rs"),
        "mod util;\nmod a_b;\nuse serde::Serialize;\npub fn f() { util::helper(); }\n",
    )
    .unwrap();
    index(t.path());
    let after = dump_dependencies(t.path());
    assert!(
        !after.contains("src/lib.rs | crate::util::helper |"),
        "stale dependency row survived a reindex:\n{after}"
    );
    assert!(
        after.contains("src/lib.rs | mod util |") || after.contains("src/lib.rs | util |"),
        "{after}"
    );
}
