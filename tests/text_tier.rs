//! Docs, config and templates are searchable — but only by full-text search.
//!
//! Measured on a real repo with 1.7.0, `count_occurrences {"pattern":"realm"}`:
//!
//! | ext   | ripgrep      | Reflex |
//! |-------|--------------|--------|
//! | md    | 3425 / 157   |      0 |
//! | yaml  |  243 /  15   |      0 |
//! | html  |  481 /  68   |      0 |
//! | proto |   76 /   4   |      0 |
//! | sh    |  135 /  17   |      0 |
//! | json  |  126 /   9   |      0 |
//!
//! Agents do not partition searches by file type. A config key lives in the YAML, the
//! Rust struct AND the spec paragraph; Reflex returned the struct and a confident 0
//! for the rest.

use reflex::mcp::run_mcp_server_io_in;
use reflex::models::{IndexConfig, Language, is_text_tier_file};
use reflex::query::{QueryEngine, QueryFilter};
use reflex::{CacheManager, Indexer};
use serde_json::{Value, json};
use std::io::Cursor;
use std::path::Path;
use tempfile::TempDir;

/// A workspace with the same token in code and in docs/config.
fn workspace() -> TempDir {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    std::fs::create_dir_all(root.join("src")).unwrap();

    std::fs::write(
        root.join("src/config.rs"),
        "pub struct Cfg { pub realm_marker: String }\n",
    )
    .unwrap();
    std::fs::write(
        root.join("README.md"),
        "The `realm_marker` is a tenant id.\n",
    )
    .unwrap();
    std::fs::write(root.join("config.yaml"), "realm_marker: default\n").unwrap();
    std::fs::write(root.join("settings.json"), "{\"realm_marker\": \"x\"}\n").unwrap();
    std::fs::write(root.join("app.toml"), "realm_marker = \"x\"\n").unwrap();
    std::fs::write(root.join("api.proto"), "string realm_marker = 1;\n").unwrap();
    std::fs::write(root.join("page.html"), "<b>realm_marker</b>\n").unwrap();
    std::fs::write(root.join("run.sh"), "echo realm_marker\n").unwrap();
    std::fs::write(root.join("notes.txt"), "realm_marker in prose\n").unwrap();
    // Must never be indexed: 100k-line lock files are trigram noise.
    std::fs::write(
        root.join("package-lock.json"),
        "{\"name\": \"realm_marker_in_a_lockfile\"}\n",
    )
    .unwrap();
    temp
}

fn index_with(root: &Path, config: IndexConfig) {
    Indexer::new(CacheManager::new(root), config)
        .index(root, false)
        .unwrap();
}

fn search(root: &Path, pattern: &str, filter: QueryFilter) -> Vec<reflex::models::SearchResult> {
    QueryEngine::new(CacheManager::new(root))
        .search(
            pattern,
            QueryFilter {
                suppress_output: true,
                limit: None,
                ..filter
            },
        )
        .unwrap()
}

fn paths(results: &[reflex::models::SearchResult]) -> Vec<String> {
    let mut v: Vec<String> = results.iter().map(|r| r.path.clone()).collect();
    v.sort();
    v.dedup();
    v
}

#[test]
fn every_text_extension_is_searchable_by_default() {
    let temp = workspace();
    index_with(temp.path(), IndexConfig::default());

    let found = paths(&search(temp.path(), "realm_marker", QueryFilter::default()));

    for expected in [
        "README.md",
        "config.yaml",
        "settings.json",
        "app.toml",
        "api.proto",
        "page.html",
        "run.sh",
        "notes.txt",
        "src/config.rs",
    ] {
        assert!(
            found.iter().any(|p| p.ends_with(expected)),
            "{expected} missing from {found:?}"
        );
    }
}

#[test]
fn lock_files_are_never_indexed() {
    let temp = workspace();
    index_with(temp.path(), IndexConfig::default());

    let found = paths(&search(temp.path(), "realm_marker", QueryFilter::default()));
    assert!(
        !found.iter().any(|p| p.contains("package-lock.json")),
        "a lock file is 100k lines of trigram noise: {found:?}"
    );
}

#[test]
fn lang_text_selects_the_tier_and_nothing_else() {
    let temp = workspace();
    index_with(temp.path(), IndexConfig::default());

    let found = paths(&search(
        temp.path(),
        "realm_marker",
        QueryFilter {
            language: Some(Language::Text),
            ..Default::default()
        },
    ));

    assert!(!found.is_empty());
    assert!(
        found.iter().all(|p| !p.ends_with(".rs")),
        "lang=text must exclude code: {found:?}"
    );
    assert!(found.iter().any(|p| p.ends_with("README.md")));
}

#[test]
fn lang_rust_still_excludes_the_tier() {
    let temp = workspace();
    index_with(temp.path(), IndexConfig::default());

    let found = paths(&search(
        temp.path(),
        "realm_marker",
        QueryFilter {
            language: Some(Language::Rust),
            ..Default::default()
        },
    ));
    assert_eq!(found.len(), 1);
    assert!(found[0].ends_with("src/config.rs"));
}

#[test]
fn exclude_text_removes_the_tier_from_a_mixed_search() {
    let temp = workspace();
    index_with(temp.path(), IndexConfig::default());

    let found = paths(&search(
        temp.path(),
        "realm_marker",
        QueryFilter {
            exclude_text: true,
            ..Default::default()
        },
    ));
    assert_eq!(found, vec!["src/config.rs".to_string()], "{found:?}");
}

#[test]
fn symbol_search_never_returns_text_files() {
    let temp = workspace();
    index_with(temp.path(), IndexConfig::default());

    let results = search(
        temp.path(),
        "realm_marker",
        QueryFilter {
            symbols_mode: true,
            ..Default::default()
        },
    );
    assert!(
        results.iter().all(|r| !r.lang.is_text()),
        "the text tier has no symbols: {:?}",
        paths(&results)
    );
}

#[test]
fn the_tier_can_be_turned_off() {
    let temp = workspace();
    index_with(
        temp.path(),
        IndexConfig {
            text_tier: false,
            ..Default::default()
        },
    );

    let found = paths(&search(temp.path(), "realm_marker", QueryFilter::default()));
    assert_eq!(found, vec!["src/config.rs".to_string()], "{found:?}");
}

#[test]
fn the_tier_survives_a_languages_allowlist() {
    let temp = workspace();
    // A user narrowing to Rust means "which parsers do I care about", not "drop my
    // documentation". Losing the tier here silently would be the original bug again.
    index_with(
        temp.path(),
        IndexConfig {
            languages: vec![Language::Rust],
            ..Default::default()
        },
    );

    let found = paths(&search(temp.path(), "realm_marker", QueryFilter::default()));
    assert!(
        found.iter().any(|p| p.ends_with("README.md")),
        "languages=[rust] must not disable the text tier: {found:?}"
    );
}

#[test]
fn the_extension_and_filename_rules_are_what_they_claim() {
    for yes in [
        "a.md",
        "a.mdx",
        "a.txt",
        "a.yaml",
        "a.yml",
        "a.toml",
        "a.json",
        "a.proto",
        "a.html",
        "a.htm",
        "a.sh",
        "a.bash",
        "a.ini",
        "a.cfg",
        "a.sql",
        "a.graphql",
        "A.MD",
    ] {
        assert!(is_text_tier_file(yes), "{yes} should be in the text tier");
    }
    for no in [
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "Cargo.lock",
        "npm-shrinkwrap-lock.json",
        "main.rs",
        "Dockerfile",
        "Makefile",
        "image.png",
    ] {
        assert!(
            !is_text_tier_file(no),
            "{no} should NOT be in the text tier"
        );
    }
}

// --- MCP surface ---

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
    let v: Value = serde_json::from_str(text.lines().find(|l| !l.is_empty()).unwrap()).unwrap();
    if let Some(e) = v.get("error").filter(|e| !e.is_null()) {
        return json!({ "__error__": e.clone() });
    }
    let payload = v["result"]["content"][0]["text"].as_str().unwrap();
    serde_json::from_str(payload).unwrap_or_else(|_| json!({ "raw": payload }))
}

#[test]
fn count_occurrences_sees_docs_and_config() {
    let temp = workspace();
    index_with(temp.path(), IndexConfig::default());

    for glob in ["**/*.md", "**/*.yaml", "**/*.json", "**/*.proto", "**/*.sh"] {
        let r = call_tool(
            temp.path(),
            "count_occurrences",
            json!({"pattern": "realm_marker", "glob": [glob]}),
        );
        assert!(
            r["total"].as_u64().unwrap_or(0) > 0,
            "{glob} returned 0: {r}"
        );
    }
}

#[test]
fn find_references_still_ignores_the_tier() {
    let temp = workspace();
    index_with(temp.path(), IndexConfig::default());

    let r = call_tool(
        temp.path(),
        "find_references",
        json!({"pattern": "realm_marker"}),
    );
    let refs = r["references"].as_array().unwrap();
    assert!(!refs.is_empty(), "{r}");
    for reference in refs {
        let path = reference["path"].as_str().unwrap();
        assert!(
            path.ends_with(".rs"),
            "a mention in docs is not a call site: {path}"
        );
    }
}

#[test]
fn search_ast_rejects_the_tier_with_a_clear_message() {
    let temp = workspace();
    index_with(temp.path(), IndexConfig::default());

    let r = call_tool(
        temp.path(),
        "search_ast",
        json!({"pattern": "(document) @d", "lang": "text", "glob": ["**/*.md"]}),
    );
    let msg = r.to_string();
    assert!(r.get("__error__").is_some(), "expected an error: {r}");
    assert!(
        msg.contains("trigram-indexed only") && msg.contains("search_code"),
        "the error must say what to do instead: {msg}"
    );
}
