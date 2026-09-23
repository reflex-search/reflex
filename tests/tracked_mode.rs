//! `[index] mode = "tracked"`: every non-binary file git tracks or does not
//! ignore is indexed, the way ripgrep searches it.
//!
//! Every remaining count gap against ripgrep in the 1.8.0 field test was a file
//! outside the old extension allowlist: `composer.lock`, `OWNERS`,
//! `SECURITY_CONTACTS`, `.po`, `.jsonl`, `.css`, `.cjs`, lock files. Agents grep
//! all of them. An allowlist can never be complete; ripgrep's rule can.

use reflex::mcp::run_mcp_server_io_in;
use reflex::models::{IndexConfig, IndexMode, Language};
use reflex::query::{QueryEngine, QueryFilter};
use reflex::{CacheManager, Indexer};
use serde_json::{Value, json};
use std::io::Cursor;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

const TOKEN: &str = "tracked_marker";

/// The handoff fixture: seven text files of shapes the allowlist never covered,
/// a lock file, a PNG, a text file with a NUL byte, and an ignored directory.
fn fixture() -> TempDir {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    let w = |rel: &str, body: &[u8]| {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    };
    w(
        "src/main.rs",
        format!("fn main() {{ {TOKEN}(); }}\n").as_bytes(),
    );
    w("Makefile", format!("all:\n\techo {TOKEN}\n").as_bytes());
    w("Dockerfile", format!("RUN echo {TOKEN}\n").as_bytes());
    w("OWNERS", format!("approvers:\n  - {TOKEN}\n").as_bytes());
    w("foo.bru", format!("meta {{ name: {TOKEN} }}\n").as_bytes());
    // Latin-1: `é` is the single byte 0xE9, invalid UTF-8 on its own.
    let mut po = b"msgid \"caf".to_vec();
    po.push(0xE9);
    po.extend_from_slice(format!(" {TOKEN}\"\n").as_bytes());
    w("foo.po", &po);
    w("a.css", format!(".{TOKEN} {{ color: red }}\n").as_bytes());
    w(
        ".githooks/pre-commit",
        format!("#!/bin/sh\necho {TOKEN}\n").as_bytes(),
    );
    w(
        "Cargo.lock",
        format!("[[package]]\nname = \"{TOKEN}\"\n").as_bytes(),
    );
    w("bundle.min.js", format!("var {TOKEN}=1;").as_bytes());
    let mut png = vec![0x89u8, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n', 0, 0];
    png.extend_from_slice(TOKEN.as_bytes());
    w("image.png", &png);
    w("nul.txt", format!("{TOKEN}\0binary tail").as_bytes());
    w(
        "ignored/notes.txt",
        format!("{TOKEN} in an ignored dir\n").as_bytes(),
    );
    w(".ignore", b"ignored/\n");
    temp
}

fn index_with(root: &Path, config: IndexConfig) -> reflex::IndexStats {
    Indexer::new(CacheManager::new(root), config)
        .index(root, false)
        .unwrap()
}

fn search(root: &Path, filter: QueryFilter) -> Vec<String> {
    let results = QueryEngine::new(CacheManager::new(root))
        .search(
            TOKEN,
            QueryFilter {
                suppress_output: true,
                limit: None,
                ..filter
            },
        )
        .unwrap();
    let mut v: Vec<String> = results.iter().map(|r| r.path.clone()).collect();
    v.sort();
    v.dedup();
    v
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
    let v: Value = serde_json::from_str(text.lines().find(|l| !l.is_empty()).unwrap()).unwrap();
    assert!(v.get("error").is_none(), "{tool} errored: {}", v["error"]);
    let payload = v["result"]["content"][0]["text"].as_str().unwrap();
    serde_json::from_str(payload).unwrap_or_else(|_| json!({ "raw": payload }))
}

#[test]
fn every_non_binary_tracked_file_is_searchable_by_default() {
    let temp = fixture();
    let stats = index_with(temp.path(), IndexConfig::default());
    let found = search(temp.path(), QueryFilter::default());

    for expected in [
        "src/main.rs",
        "Makefile",
        "Dockerfile",
        "OWNERS",
        "foo.bru",
        "foo.po",
        "a.css",
    ] {
        assert!(
            found.contains(&expected.to_string()),
            "{expected} missing: {found:?}"
        );
    }
    for absent in [
        "image.png",
        "nul.txt",
        "ignored/notes.txt",
        ".githooks/pre-commit",
        "Cargo.lock",
        "bundle.min.js",
    ] {
        assert!(
            !found.contains(&absent.to_string()),
            "{absent} present: {found:?}"
        );
    }
    assert_eq!(stats.skipped_binary, 2, "png and the NUL file: {stats:?}");
    assert_eq!(stats.files_by_language.get("Lock"), Some(&1));
    assert_eq!(stats.files_by_language.get("Generated"), Some(&1));
}

#[test]
fn lock_and_generated_files_are_indexed_but_opt_in() {
    let temp = fixture();
    index_with(temp.path(), IndexConfig::default());

    let with_locks = search(
        temp.path(),
        QueryFilter {
            include_locks: true,
            ..Default::default()
        },
    );
    assert!(
        with_locks.contains(&"Cargo.lock".to_string()),
        "{with_locks:?}"
    );
    assert!(
        !with_locks.contains(&"bundle.min.js".to_string()),
        "{with_locks:?}"
    );

    let only_locks = search(
        temp.path(),
        QueryFilter {
            language: Some(Language::Lock),
            ..Default::default()
        },
    );
    assert_eq!(only_locks, vec!["Cargo.lock".to_string()]);

    let only_generated = search(
        temp.path(),
        QueryFilter {
            language: Some(Language::Generated),
            ..Default::default()
        },
    );
    assert_eq!(only_generated, vec!["bundle.min.js".to_string()]);

    let with_generated = search(
        temp.path(),
        QueryFilter {
            include_generated: true,
            ..Default::default()
        },
    );
    assert!(with_generated.contains(&"bundle.min.js".to_string()));
    assert!(!with_generated.contains(&"Cargo.lock".to_string()));
}

#[test]
fn a_zero_whose_only_candidates_were_excluded_says_so() {
    let temp = fixture();
    let root = temp.path();
    std::fs::write(
        root.join("Cargo.lock"),
        "[[package]]\nname = \"serde\"\nversion = \"1.0.190\"\n",
    )
    .unwrap();
    index_with(root, IndexConfig::default());

    let r = call_tool(
        root,
        "search_code",
        json!({ "pattern": "1.0.190", "contains": true }),
    );
    assert_eq!(r["total_count"], 0, "{r}");
    assert_eq!(r["excluded_by_default"], 1, "{r}");
    let hint = r["hint"].as_str().unwrap_or("");
    assert!(hint.contains("include_locks"), "{r}");

    let r = call_tool(
        root,
        "search_code",
        json!({ "pattern": "1.0.190", "contains": true, "include_locks": true }),
    );
    assert_eq!(r["total_count"], 1, "{r}");
    assert!(r.get("excluded_by_default").is_none(), "{r}");

    let r = call_tool(
        root,
        "count_occurrences",
        json!({ "pattern": "1.0.190", "contains": true, "lang": "lock" }),
    );
    assert_eq!(r["total"], 1, "{r}");
}

#[test]
fn hidden_directories_are_opt_in() {
    let temp = fixture();
    index_with(
        temp.path(),
        IndexConfig {
            hidden: true,
            ..Default::default()
        },
    );
    let found = search(temp.path(), QueryFilter::default());
    assert!(
        found.contains(&".githooks/pre-commit".to_string()),
        "{found:?}"
    );
    assert!(
        !found
            .iter()
            .any(|p| p.starts_with(".reflex/") || p.starts_with(".git/")),
        "never Reflex's own cache: {found:?}"
    );
}

#[test]
fn allowlist_mode_restores_the_fixed_extension_list() {
    let temp = fixture();
    index_with(
        temp.path(),
        IndexConfig {
            mode: IndexMode::Allowlist,
            ..Default::default()
        },
    );
    let found = search(
        temp.path(),
        QueryFilter {
            include_locks: true,
            include_generated: true,
            ..Default::default()
        },
    );
    assert_eq!(
        found,
        vec![
            "Dockerfile".to_string(),
            "Makefile".to_string(),
            "foo.bru".to_string(),
            "src/main.rs".to_string(),
        ],
        "{found:?}"
    );
}

#[test]
fn the_classifier_rules_are_what_they_claim() {
    use reflex::models::{is_generated_name, is_lock_file};
    for (name, lang) in [
        ("Cargo.lock", Language::Lock),
        ("package-lock.json", Language::Lock),
        ("go.sum", Language::Lock),
        ("x.pb.go", Language::Generated),
        ("app.min.js", Language::Generated),
        ("app.js.map", Language::Generated),
        ("types_generated.rs", Language::Generated),
        ("main.rs", Language::Rust),
        ("app.cjs", Language::JavaScript),
        ("README", Language::Text),
        ("OWNERS", Language::Text),
        ("a.css", Language::Text),
        ("foo.po", Language::Text),
        ("image.png", Language::Text),
    ] {
        assert_eq!(Language::from_path(Path::new(name)), lang, "{name}");
    }
    assert!(is_lock_file("flake.lock") && !is_lock_file("settings.json"));
    assert!(is_generated_name("X.MIN.CSS") && !is_generated_name("generator.rs"));
}

/// The gate the handoff asks for: `count_occurrences(contains:true)` equals
/// `rg -c -F pattern | sum` on the fixture, lock files included when asked.
/// ripgrep honours `.gitignore` only inside a git repository, so the fixture
/// uses `.ignore`, which both respect everywhere.
#[test]
fn substring_counts_match_ripgrep() {
    let Ok(out) = Command::new("rg").arg("--version").output() else {
        eprintln!("ripgrep not installed; skipping parity gate");
        return;
    };
    if !out.status.success() {
        return;
    }
    let temp = fixture();
    let root = temp.path();
    index_with(root, IndexConfig::default());

    let rg_count = |extra: &[&str]| -> u64 {
        let out = Command::new("rg")
            .current_dir(root)
            .args(["-c", "-F", "--no-messages", TOKEN])
            .args(extra)
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|l| l.rsplit_once(':').and_then(|(_, n)| n.parse::<u64>().ok()))
            .sum()
    };

    // Everything ripgrep sees, lock and generated files included.
    let rg_all = rg_count(&[]);
    let r = call_tool(
        root,
        "count_occurrences",
        json!({ "pattern": TOKEN, "contains": true, "include_locks": true, "include_generated": true }),
    );
    assert_eq!(r["total"], rg_all, "ripgrep {rg_all}: {r}");

    // The default leaves lock and generated files out, exactly those.
    let rg_default = rg_count(&["-g", "!Cargo.lock", "-g", "!*.min.js"]);
    let r = call_tool(
        root,
        "count_occurrences",
        json!({ "pattern": TOKEN, "contains": true }),
    );
    assert_eq!(r["total"], rg_default, "ripgrep {rg_default}: {r}");
    assert!(
        rg_all > rg_default,
        "the fixture must contain excluded files"
    );
}
