//! Minified files must not cost gigabytes.
//!
//! `rfx index-symbols-internal` reached **34.4 GiB RSS** and ran 3m55s on a 1027-file
//! repo, because of ONE file: `vendor/swagger-ui-5.17.14/swagger-ui-bundle.js`,
//! 1,452,753 bytes on a single line.
//!
//! `extract_preview` took 7 LINES with no byte bound. With one line it returned an
//! owned copy of the whole file — once per symbol:
//!
//! ```text
//! 13,843 symbols x 1,452,753 B = 18.7 GiB
//! x2 (batch_set cloned)        = 37.4 GiB
//! observed                     = 34.4 GiB
//! ```
//!
//! The bug is MULTIPLICATIVE, so it reproduces at any scale. These tests shrink the
//! input ~700x and assert on total preview bytes rather than RSS: exactly the
//! quantity that was wrong, exactly computable, and fast.

use reflex::models::{IndexConfig, Language};
use reflex::parsers::ParserFactory;
use reflex::parsers::preview::PREVIEW_MAX_BYTES;
use reflex::query::{QueryEngine, QueryFilter};
use reflex::{CacheManager, Indexer};
use std::fs;
use tempfile::TempDir;

/// ~2 MB of minified JavaScript on ONE line, the shape of a real bundle.
fn minified_bundle(functions: usize) -> String {
    (0..functions)
        .map(|i| format!("function f{i}(a,b){{return a+b}}"))
        .collect::<Vec<_>>()
        .join("")
}

/// The single assertion that captures the 34 GiB.
#[test]
fn no_symbol_of_a_minified_file_carries_a_huge_preview() {
    let src = minified_bundle(20_000);
    assert!(src.len() > 500_000, "fixture too small: {}", src.len());
    assert!(!src.contains('\n'), "fixture must be one line");

    // Parse with the shape guard bypassed, so this tests the PREVIEW cap specifically
    // rather than the skip. Both layers matter; this is the correctness one.
    let symbols = reflex::parsers::typescript::parse("bundle.js", &src, Language::JavaScript)
        .expect("parsing must not error");

    assert!(!symbols.is_empty(), "fixture should yield symbols");

    for s in &symbols {
        assert!(
            s.preview.len() <= PREVIEW_MAX_BYTES,
            "preview of {} bytes for symbol {:?} — the bug is back",
            s.preview.len(),
            s.symbol
        );
    }

    // Before the fix this was symbols.len() x src.len() — hundreds of GB.
    let total: usize = symbols.iter().map(|s| s.preview.len()).sum();
    assert!(
        total <= symbols.len() * PREVIEW_MAX_BYTES,
        "{} symbols totalled {total} preview bytes",
        symbols.len()
    );
}

#[test]
fn the_shape_guard_declines_a_bundle_but_keeps_real_code() {
    let bundle = minified_bundle(20_000);
    assert!(
        ParserFactory::parse("bundle.js", &bundle, Language::JavaScript)
            .unwrap()
            .is_empty(),
        "a minified bundle should yield no symbols"
    );

    let real = "export function realOne(a: number) {\n  return a + 1;\n}\n".repeat(200);
    let symbols = ParserFactory::parse("real.ts", &real, Language::TypeScript).unwrap();
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol.as_deref() == Some("realOne")),
        "real source must still yield its symbols"
    );
}

/// The whole point of skipping symbols rather than skipping the file.
#[test]
fn a_minified_file_stays_full_text_searchable() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();

    fs::write(
        root.join("bundle.js"),
        format!("{}function findMeInTheBundle(){{}}", minified_bundle(8_000)),
    )
    .unwrap();
    fs::write(root.join("real.ts"), "export function realOne() {}\n").unwrap();

    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .unwrap();

    let engine = QueryEngine::new(CacheManager::new(root));
    let filter = || QueryFilter {
        suppress_output: true,
        limit: None,
        ..Default::default()
    };

    // Full-text finds it...
    let hits = engine
        .search("findMeInTheBundle", filter())
        .expect("full-text search must work on a minified file");
    assert!(
        hits.iter().any(|r| r.path.ends_with("bundle.js")),
        "minified files must stay text-searchable: {:?}",
        hits.iter().map(|r| &r.path).collect::<Vec<_>>()
    );

    // ...and its preview is bounded, not the whole 500 KB line.
    for h in &hits {
        assert!(
            h.preview.len() <= 600,
            "full-text preview of {} bytes — query-path cap is missing",
            h.preview.len()
        );
    }

    // ...but symbol search declines it.
    let syms = engine
        .search(
            "findMeInTheBundle",
            QueryFilter {
                symbols_mode: true,
                ..filter()
            },
        )
        .unwrap();
    assert!(
        !syms.iter().any(|r| r.path.ends_with("bundle.js")),
        "symbol search must skip minified files"
    );

    // Real code is unaffected by any of it.
    let real = engine
        .search(
            "realOne",
            QueryFilter {
                symbols_mode: true,
                ..filter()
            },
        )
        .unwrap();
    assert!(!real.is_empty(), "real source must still have symbols");
}

/// A hit deep inside a single-line file must show its own neighbourhood.
#[test]
fn a_full_text_preview_is_windowed_on_the_match() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();

    // Varied filler: a repeated single character collapses to one trigram, which the
    // posting-list cap then truncates, so the file would not be findable at all.
    let filler: String = (0..40_000).map(|i| format!("var v{i}={i};")).collect();
    fs::write(
        root.join("one_line.js"),
        // Separated by non-word characters: the needle must sit on its own identifier
        // boundary, or whole-identifier matching correctly declines it.
        format!("{filler};uniqueNeedleToken();{filler}"),
    )
    .unwrap();

    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .unwrap();

    let hits = QueryEngine::new(CacheManager::new(root))
        .search(
            "uniqueNeedleToken",
            QueryFilter {
                suppress_output: true,
                limit: None,
                ..Default::default()
            },
        )
        .unwrap();

    assert_eq!(hits.len(), 1, "{:?}", hits);
    let preview = &hits[0].preview;
    assert!(
        preview.len() <= 600,
        "preview was {} bytes — head truncation or no cap",
        preview.len()
    );
    assert!(
        preview.contains("uniqueNeedleToken"),
        "the match must be visible in its own preview: {preview:.80}"
    );
}
