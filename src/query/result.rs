//! Result assembly utilities: trigram index reconstruction and file-id resolution

use anyhow::{Context, Result};

use crate::content_store::ContentReader;
use crate::trigram::TrigramIndex;

/// Find a file_id by its path string in the content store.
pub fn find_file_id(content_reader: &ContentReader, target_path: &str) -> Option<u32> {
    for file_id in 0..content_reader.file_count() {
        if let Some(path) = content_reader.get_file_path(file_id as u32)
            && path.to_string_lossy() == target_path
        {
            return Some(file_id as u32);
        }
    }
    None
}

/// Rebuild a trigram index from content store (fallback when trigrams.bin is missing).
pub fn rebuild_trigram_index(content_reader: &ContentReader) -> Result<TrigramIndex> {
    log::debug!(
        "Rebuilding trigram index from {} files",
        content_reader.file_count()
    );
    let mut trigram_index = TrigramIndex::new();

    for file_id in 0..content_reader.file_count() {
        let file_path = content_reader
            .get_file_path(file_id as u32)
            .context("Invalid file_id")?
            .to_path_buf();
        let content = content_reader.get_file_content(file_id as u32)?;

        let idx = trigram_index.add_file(file_path);
        trigram_index.index_file(idx, content);
    }

    trigram_index.finalize();
    log::debug!(
        "Trigram index rebuilt with {} trigrams",
        trigram_index.trigram_count()
    );

    Ok(trigram_index)
}

/// Normalize a glob pattern to gitignore / ripgrep rules against the stored paths.
///
/// Indexed paths are stored **relative and without a `./` prefix** (e.g.
/// `src/parsers/rust.rs`). The rules, which are what every agent's prior from
/// `.gitignore` and `rg -g` expects:
///
/// * A pattern containing a `/` is **anchored at the index root**:
///   `src/**/*.rs` matches `src/mcp.rs` but not `vendor/src/x.rs`.
/// * A bare name matches **at any depth**: `*.rs`, `main.rs`, `Makefile`.
/// * `**/src/**/*.rs` explicitly matches `src/` anywhere.
/// * A trailing `/` names a directory and everything under it: `target/`. As in
///   gitignore, a trailing slash alone does not anchor: `src/` is any `src/`.
/// * A leading `./` or `/` is dropped (both mean "from the root").
///
/// Before 2.0.0 every relative pattern got a `**/` prefix, so `src/**/*.rs` matched
/// 7131 files against ripgrep's 6770 in the field test (`simulation/src/`,
/// `sdks/go/src/`, …). Compile the result with [`build_glob_set`], which also
/// stops `*` from crossing `/`.
///
/// Examples:
/// - "src/**/*.rs" → "src/**/*.rs" (anchored)
/// - "*.rs"        → "**/*.rs"
/// - "main.rs"     → "**/main.rs"
/// - "target/"     → "**/target/**"
/// - "src/"        → "**/src/**"
/// - "./services/**/*.php" → "services/**/*.php"
/// - "**/foo"      → unchanged
pub fn normalize_glob_pattern(pattern: &str) -> String {
    let mut p = pattern.trim();
    loop {
        if let Some(rest) = p.strip_prefix("./") {
            p = rest;
        } else if let Some(rest) = p.strip_prefix('/') {
            p = rest;
        } else {
            break;
        }
    }
    if p.is_empty() {
        return "**".to_string();
    }
    let (body, is_dir) = match p.strip_suffix('/') {
        Some(body) => (body, true),
        None => (p, false),
    };
    let mut out = String::with_capacity(body.len() + 6);
    if !body.contains('/') {
        out.push_str("**/");
    }
    out.push_str(body);
    if is_dir {
        out.push_str("/**");
    }
    out
}

/// Compile glob patterns into one matcher under the rules of
/// [`normalize_glob_pattern`]. `None` when there are no patterns.
///
/// `literal_separator(true)` makes `*` stop at `/`, as in gitignore and ripgrep, so
/// `src/*.rs` means "directly in `src/`" (which the `--glob` help always claimed).
/// An invalid pattern is logged and skipped rather than failing the query.
pub fn build_glob_set(patterns: &[String], what: &str) -> Option<globset::GlobSet> {
    use globset::{GlobBuilder, GlobSetBuilder};

    if patterns.is_empty() {
        return None;
    }
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        let normalized = normalize_glob_pattern(pattern);
        match GlobBuilder::new(&normalized)
            .literal_separator(true)
            .build()
        {
            Ok(glob) => {
                builder.add(glob);
            }
            Err(e) => log::warn!("Invalid {} pattern '{}': {}", what, pattern, e),
        }
    }
    match builder.build() {
        Ok(set) => Some(set),
        Err(e) => {
            log::warn!("Failed to build {} matcher: {}", what, e);
            None
        }
    }
}

#[cfg(test)]
mod normalize_glob_tests {
    use super::{build_glob_set, normalize_glob_pattern};

    fn matches(pattern: &str, path: &str) -> bool {
        build_glob_set(&[pattern.to_string()], "test")
            .unwrap()
            .is_match(path)
    }

    #[test]
    fn slash_patterns_are_anchored_and_bare_names_recurse() {
        assert_eq!(normalize_glob_pattern("src/**/*.rs"), "src/**/*.rs");
        assert_eq!(normalize_glob_pattern("src/**"), "src/**");
        assert_eq!(normalize_glob_pattern("*.rs"), "**/*.rs");
        assert_eq!(normalize_glob_pattern("main.rs"), "**/main.rs");
        assert_eq!(normalize_glob_pattern("Makefile"), "**/Makefile");
    }

    #[test]
    fn prefixes_and_trailing_slashes() {
        assert_eq!(
            normalize_glob_pattern("./services/**/*.php"),
            "services/**/*.php"
        );
        assert_eq!(normalize_glob_pattern("/abs/path/*.rs"), "abs/path/*.rs");
        assert_eq!(normalize_glob_pattern("**/foo"), "**/foo");
        assert_eq!(normalize_glob_pattern("target/"), "**/target/**");
        assert_eq!(normalize_glob_pattern("src/"), "**/src/**");
        assert_eq!(normalize_glob_pattern("/"), "**");
    }

    /// REF-191 regression: the natural `src/**` an LLM writes must match
    /// bare stored paths like `src/parsers/rust.rs`.
    #[test]
    fn src_glob_matches_bare_stored_paths() {
        assert!(matches("src/**", "src/parsers/rust.rs"));
        assert!(matches("src/**", "src/mcp.rs"));
        assert!(matches("src/**/*.rs", "src/parsers/rust.rs"));
        assert!(matches("src/**/*.rs", "src/mcp.rs"));
    }

    /// 2.0.0: anchored, like ripgrep's `-g 'src/**/*.rs'`.
    #[test]
    fn anchored_pattern_rejects_nested_src() {
        assert!(!matches("src/**", "vendor/src/b.rs"));
        assert!(!matches("src/**/*.rs", "sdks/go/src/b.rs"));
        assert!(!matches("src/**/*.rs", "simulation/src/lib.rs"));
    }

    #[test]
    fn double_star_prefix_matches_any_depth() {
        assert!(matches("**/src/**/*.rs", "src/a.rs"));
        assert!(matches("**/src/**/*.rs", "vendor/src/b.rs"));
        assert!(matches("**/src/**/*.rs", "sdks/go/src/deep/c.rs"));
    }

    #[test]
    fn dot_slash_prefix_is_stripped() {
        assert!(matches("./src/**/*.rs", "src/a.rs"));
        assert!(!matches("./src/**/*.rs", "vendor/src/b.rs"));
    }

    #[test]
    fn trailing_slash_means_directory() {
        assert!(matches("target/", "target/debug/x.rs"));
        assert!(matches("target/", "crates/a/target/x.rs"));
        assert!(matches("src/", "src/a/b.rs"));
        // gitignore: a trailing slash alone does not anchor.
        assert!(matches("src/", "vendor/src/b.rs"));
        assert!(!matches("src/", "src_helpers/b.rs"));
    }

    #[test]
    fn star_does_not_cross_separator() {
        assert!(matches("src/*.rs", "src/a.rs"));
        assert!(!matches("src/*.rs", "src/a/b.rs"));
        assert!(matches("app/Models/*.php", "app/Models/User.php"));
        assert!(!matches(
            "app/Models/*.php",
            "app/Models/Traits/HasUuid.php"
        ));
    }

    #[test]
    fn src_glob_does_not_match_unrelated_paths() {
        // Component boundary: `src` must be a whole path component.
        assert!(!matches("src/**", "src_helpers/foo.rs"));
        assert!(!matches("src/**/*.rs", "benches/foo.rs"));
    }

    #[test]
    fn bare_filename_matches_at_any_depth() {
        assert!(matches("main.rs", "src/main.rs"));
        assert!(matches("main.rs", "main.rs"));
        assert!(matches("*.rs", "a/b/c.rs"));
        assert!(matches("Makefile", "sdks/go/Makefile"));
    }
}
