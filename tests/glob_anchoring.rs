//! Glob filters follow gitignore / ripgrep rules.
//!
//! The 1.8.0 field test: `--glob 'src/**/*.rs'` returned 7131 hits where ripgrep's
//! `-g 'src/**/*.rs'` returned 6770, because Reflex prefixed every relative pattern
//! with `**/` and so also matched `simulation/src/`, `sdks/go/src/`, `sdks/php/src/`.
//! Every agent's prior for `glob` comes from gitignore, where a pattern containing
//! a `/` is anchored at the root. The same rules now apply to `--glob`, `--exclude`,
//! the MCP `glob` / `exclude` arguments, and `[index] include/exclude` (which were
//! parsed but never applied before 1.8.0).

use reflex::mcp::run_mcp_server_io_in;
use reflex::query::{QueryEngine, QueryFilter};
use reflex::{CacheManager, IndexConfig, Indexer};
use serde_json::{Value, json};
use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

const NEEDLE: &str = "anchored_needle_fn";

/// `src/a.rs` at the root and `vendor/src/b.rs` nested, both holding the needle.
fn tree() -> TempDir {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    fs::create_dir_all(root.join("src")).unwrap();
    fs::create_dir_all(root.join("vendor/src")).unwrap();
    fs::write(root.join("src/a.rs"), format!("pub fn {NEEDLE}() {{}}\n")).unwrap();
    fs::write(
        root.join("vendor/src/b.rs"),
        format!("pub fn {NEEDLE}() {{}}\n"),
    )
    .unwrap();
    fs::write(root.join("Makefile"), format!("all:\n\techo {NEEDLE}\n")).unwrap();
    temp
}

fn index(root: &Path, config: IndexConfig) {
    Indexer::new(CacheManager::new(root), config)
        .index(root, false)
        .unwrap();
}

fn paths(root: &Path, glob: &[&str], exclude: &[&str]) -> Vec<String> {
    let engine = QueryEngine::new(CacheManager::new(root));
    let filter = QueryFilter {
        glob_patterns: glob.iter().map(|s| s.to_string()).collect(),
        exclude_patterns: exclude.iter().map(|s| s.to_string()).collect(),
        limit: None,
        suppress_output: true,
        ..Default::default()
    };
    let mut out: Vec<String> = engine
        .search_with_metadata(NEEDLE, filter)
        .unwrap()
        .results
        .into_iter()
        .map(|fg| fg.path)
        .collect();
    out.sort();
    out
}

fn call_tool(root: &Path, tool: &str, args: Value) -> Value {
    let req = json!({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": { "name": tool, "arguments": args }
    });
    let mut out: Vec<u8> = Vec::new();
    run_mcp_server_io_in(
        root,
        Cursor::new(format!("{req}\n").into_bytes()),
        &mut out,
        false,
    )
    .unwrap();
    let text = String::from_utf8(out).unwrap();
    let line = text.lines().find(|l| !l.is_empty()).unwrap();
    let v: Value = serde_json::from_str(line).unwrap();
    assert!(v.get("error").is_none(), "{tool} errored: {}", v["error"]);
    serde_json::from_str(v["result"]["content"][0]["text"].as_str().unwrap()).unwrap()
}

// ---- query time ----------------------------------------------------------------

#[test]
fn a_pattern_with_a_slash_is_anchored_at_the_root() {
    let temp = tree();
    index(temp.path(), IndexConfig::default());
    assert_eq!(paths(temp.path(), &["src/**/*.rs"], &[]), ["src/a.rs"]);
    assert_eq!(paths(temp.path(), &["src/**"], &[]), ["src/a.rs"]);
    assert_eq!(paths(temp.path(), &["./src/**/*.rs"], &[]), ["src/a.rs"]);
}

#[test]
fn a_double_star_prefix_matches_src_anywhere() {
    let temp = tree();
    index(temp.path(), IndexConfig::default());
    assert_eq!(
        paths(temp.path(), &["**/src/**/*.rs"], &[]),
        ["src/a.rs", "vendor/src/b.rs"]
    );
}

#[test]
fn a_bare_name_matches_at_any_depth() {
    let temp = tree();
    index(temp.path(), IndexConfig::default());
    assert_eq!(
        paths(temp.path(), &["*.rs"], &[]),
        ["src/a.rs", "vendor/src/b.rs"]
    );
    assert_eq!(paths(temp.path(), &["Makefile"], &[]), ["Makefile"]);
    assert_eq!(paths(temp.path(), &["b.rs"], &[]), ["vendor/src/b.rs"]);
}

#[test]
fn exclude_follows_the_same_rules() {
    let temp = tree();
    index(temp.path(), IndexConfig::default());
    assert_eq!(
        paths(temp.path(), &[], &["src/**"]),
        ["Makefile", "vendor/src/b.rs"]
    );
    assert_eq!(paths(temp.path(), &[], &["**/src/**"]), ["Makefile"]);
    assert_eq!(
        paths(temp.path(), &[], &["vendor/"]),
        ["Makefile", "src/a.rs"]
    );
    assert_eq!(paths(temp.path(), &["*.rs"], &["vendor/**"]), ["src/a.rs"]);
}

#[test]
fn star_does_not_cross_a_separator() {
    let temp = tree();
    index(temp.path(), IndexConfig::default());
    // `vendor/*.rs` is "directly in vendor/", and b.rs is one level deeper.
    assert!(paths(temp.path(), &["vendor/*.rs"], &[]).is_empty());
    assert_eq!(
        paths(temp.path(), &["vendor/*/*.rs"], &[]),
        ["vendor/src/b.rs"]
    );
}

#[test]
fn mcp_glob_and_exclude_are_anchored_too() {
    let temp = tree();
    index(temp.path(), IndexConfig::default());
    let root = temp.path();

    let anchored = call_tool(
        root,
        "search_code",
        json!({"pattern": NEEDLE, "mode": "count", "glob": ["src/**/*.rs"]}),
    );
    assert_eq!(anchored["count"], 1, "{anchored}");

    let anywhere = call_tool(
        root,
        "search_code",
        json!({"pattern": NEEDLE, "mode": "count", "glob": ["**/src/**/*.rs"]}),
    );
    assert_eq!(anywhere["count"], 2, "{anywhere}");

    let excluded = call_tool(
        root,
        "list_locations",
        json!({"pattern": NEEDLE, "exclude": ["src/**"]}),
    );
    let got: Vec<&str> = excluded["locations"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["path"].as_str().unwrap())
        .collect();
    assert_eq!(got, ["Makefile", "vendor/src/b.rs"], "{excluded}");
}

// ---- index time ----------------------------------------------------------------

fn git(root: &Path, args: &[&str]) {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The tree as a committed git repo with the given `[index]` config on disk, so
/// both the walker and the freshness check read the same policy.
fn repo_with_config(index_toml: &str) -> TempDir {
    let temp = tree();
    let root = temp.path();
    git(root, &["init", "-q"]);
    git(root, &["config", "user.email", "t@example.com"]);
    git(root, &["config", "user.name", "T"]);
    fs::write(root.join(".gitignore"), ".reflex/\n").unwrap();
    git(root, &["add", "-A"]);
    git(root, &["commit", "-qm", "initial"]);

    let cache = CacheManager::new(root);
    cache.init().unwrap();
    fs::write(root.join(".reflex/config.toml"), index_toml).unwrap();
    let config = cache.load_index_config().unwrap();
    index(root, config);
    temp
}

#[test]
fn index_exclude_patterns_keep_files_out_and_out_of_the_freshness_check() {
    let temp = repo_with_config("[index]\nexclude.patterns = [\"vendor/**\"]\n");
    let root = temp.path();

    assert_eq!(
        paths(root, &[], &[]),
        ["Makefile", "src/a.rs"],
        "vendor/ must never be indexed"
    );

    // Editing an excluded file is not a change Reflex can see; the index stays
    // fresh. Editing an included one makes it stale.
    let engine = QueryEngine::new(CacheManager::new(root));
    let (status, trusted, _) = engine.get_index_status().unwrap();
    assert!(trusted, "clean tree must be fresh: {status:?}");

    fs::write(root.join("vendor/src/b.rs"), "pub fn changed() {}\n").unwrap();
    reflex::query::invalidate_caches(root);
    let (status, trusted, warning) = engine.get_index_status().unwrap();
    assert!(
        trusted,
        "an edit to an excluded file must not mark the index stale: {status:?} {warning:?}"
    );

    fs::write(root.join("src/a.rs"), "pub fn changed() {}\n").unwrap();
    reflex::query::invalidate_caches(root);
    let (_, trusted, _) = engine.get_index_status().unwrap();
    assert!(
        !trusted,
        "an edit to an indexed file must mark the index stale"
    );
}

#[test]
fn index_include_patterns_are_anchored_whitelists() {
    let temp = repo_with_config("[index]\ninclude.patterns = [\"src/**/*.rs\"]\n");
    assert_eq!(paths(temp.path(), &[], &[]), ["src/a.rs"]);

    let temp = repo_with_config("[index]\ninclude.patterns = [\"**/src/**/*.rs\"]\n");
    assert_eq!(
        paths(temp.path(), &[], &[]),
        ["src/a.rs", "vendor/src/b.rs"]
    );
}
