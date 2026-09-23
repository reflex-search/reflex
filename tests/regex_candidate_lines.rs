//! A regex search verifies only the candidate LINES its literals name, not every
//! line of every candidate file. That is sound only when every literal the
//! extractor emits is present verbatim in every match — which the extractor
//! guarantees by dropping the atom before `?`, `*` and `{0,n}`. This suite checks
//! the line-restricted path against a brute-force scan of the same corpus.

mod test_helpers;

use reflex::CacheManager;
use reflex::query::{QueryEngine, QueryFilter};
use regex::Regex;
use std::collections::BTreeSet;
use std::path::Path;
use test_helpers::setup_corpus;

const PATTERNS: &[&str] = &[
    r"foo.*bar",
    r"fn (get|set)_\w+",
    r"fn \w+\(",
    r"->\s*Result<",
    r"class \w+",
    r"impor?t",    // optional last char: literal must be `impo`, not `import`
    r"returns?\b", // same, with a boundary
    r"selfs*",
    r"\.unwrap\(\)",
    r"(?i)controller", // case-insensitive: literal looked up under every case variant
    r"(?i:Foo)bar",    // scoped flag: `foo` folded, `bar` exact
    r"(?i)get_\w+",    // folded literal plus a class
    r"(?i)foo(?-i)Bar", // flag turned off again
    r"(?i)fn (get|set)_", // alternation under the flag
    r"(abc|de)f",      // a branch without a literal: must not drop `def` lines
    r"foobar(baz)?x",  // optional group: its literal is not required
    r"[abc]def",       // class body is not a literal
    r"abc\p{Lu}def",   // unknown escape breaks the sequence
    r"^use ",
    r"pub(lic)? fn",
    r"test_?\w+",
];

fn brute_force(root: &Path, pattern: &str) -> BTreeSet<(String, usize)> {
    let re = Regex::new(pattern).unwrap();
    let mut hits = BTreeSet::new();
    let walker = ignore::WalkBuilder::new(root).hidden(true).build();
    for entry in walker.flatten() {
        let path = entry.path();
        if !path.is_file() || path.components().any(|c| c.as_os_str() == ".reflex") {
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if reflex::Language::from_extension(ext) == reflex::Language::Unknown {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let rel = path
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        for (i, line) in text.lines().enumerate() {
            if re.is_match(line) {
                hits.insert((rel.clone(), i + 1));
            }
        }
    }
    hits
}

fn reflex_hits(root: &Path, pattern: &str) -> BTreeSet<(String, usize)> {
    let engine = QueryEngine::new(CacheManager::new(root));
    let filter = QueryFilter {
        use_regex: true,
        limit: None,
        suppress_output: true,
        ..Default::default()
    };
    engine
        .search_with_metadata(pattern, filter)
        .unwrap()
        .results
        .into_iter()
        .flat_map(|fg| {
            let p = fg.path.trim_start_matches("./").to_string();
            fg.matches
                .into_iter()
                .map(move |m| (p.clone(), m.span.start_line))
        })
        .collect()
}

#[test]
fn line_restricted_regex_equals_a_full_scan() {
    let root = setup_corpus();
    for pattern in PATTERNS {
        let want = brute_force(root, pattern);
        let got = reflex_hits(root, pattern);
        let missing: Vec<_> = want.difference(&got).take(5).collect();
        let extra: Vec<_> = got.difference(&want).take(5).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "pattern {pattern:?}: {} expected, {} found; missing {missing:?}, extra {extra:?}",
            want.len(),
            got.len()
        );
    }
}
