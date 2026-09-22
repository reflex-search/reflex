//! MCP-layer recovery from a corrupted on-disk cache (1.7.0).
//!
//! Field failure: `MCP error -32603: Cache appears to be corrupted: content.bin
//! is too small - appears to be corrupted. Run 'rfx index'`. The agent cannot
//! run the CLI. The server must now rebuild once and retry, and when that is
//! impossible, name the `index_project` tool instead of `rfx index`.

use std::fs;
use std::io::Cursor;
use std::path::Path;

use reflex::cache::CacheManager;
use reflex::indexer::Indexer;
use reflex::models::IndexConfig;
use tempfile::TempDir;

fn build_workspace() -> TempDir {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    fs::write(
        root.join("hello.rs"),
        "fn greet() -> &'static str { \"hello\" }\n",
    )
    .unwrap();
    fs::write(
        root.join("lib.rs"),
        "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
    )
    .unwrap();
    fs::write(
        root.join("main.rs"),
        "fn main() { println!(\"{}\", greet()); }\n",
    )
    .unwrap();

    let cache = CacheManager::new(root);
    Indexer::new(cache, IndexConfig::default())
        .index(root, false)
        .unwrap();
    assert!(CacheManager::new(root).validate().is_ok());
    temp
}

fn truncate(path: &Path) {
    let file = fs::OpenOptions::new().write(true).open(path).unwrap();
    file.set_len(2).unwrap();
}

fn call(root: &Path, request: &str) -> serde_json::Value {
    let input = format!("{}\n", request);
    let mut out = Vec::new();
    reflex::mcp::run_mcp_server_io_in(root, Cursor::new(input.into_bytes()), &mut out, false)
        .unwrap();
    let raw = String::from_utf8(out).unwrap();
    let line = raw.lines().next().expect("one response line");
    serde_json::from_str(line).unwrap()
}

const SEARCH_FN: &str = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"search_code","arguments":{"pattern":"fn "}}}"#;

#[test]
fn search_code_auto_rebuilds_after_truncated_content_bin() {
    let temp = build_workspace();
    let root = temp.path();

    truncate(&root.join(".reflex/content.bin"));
    let err = CacheManager::new(root).validate().unwrap_err().to_string();
    assert!(err.contains("content.bin"), "precondition: {err}");

    let resp = call(root, SEARCH_FN);
    assert!(
        resp.get("error").is_none(),
        "search_code must recover, got error: {}",
        resp["error"]
    );
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(!text.contains("too small"), "{text}");
    let body: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(
        body["total_count"].as_u64().unwrap_or(0) >= 3,
        "rebuilt index must answer the query: {text}"
    );

    assert!(
        CacheManager::new(root).validate().is_ok(),
        "cache must be valid after auto-rebuild"
    );
}

#[test]
fn search_code_auto_rebuilds_after_truncated_trigrams_bin() {
    let temp = build_workspace();
    let root = temp.path();

    truncate(&root.join(".reflex/trigrams.bin"));

    let resp = call(root, SEARCH_FN);
    assert!(resp.get("error").is_none(), "got error: {}", resp["error"]);
    assert!(CacheManager::new(root).validate().is_ok());
}

#[test]
fn index_project_is_not_wrapped_in_recovery() {
    // A corrupted cache plus a plain (non-force) index_project call must still
    // succeed via the indexer's own magic-byte fast-path check, and must not
    // loop through the recovery wrapper.
    let temp = build_workspace();
    let root = temp.path();
    truncate(&root.join(".reflex/content.bin"));

    let req = r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"index_project","arguments":{}}}"#;
    let resp = call(root, req);
    assert!(resp.get("error").is_none(), "got error: {}", resp["error"]);
    assert!(CacheManager::new(root).validate().is_ok());
}

#[cfg(unix)]
#[test]
fn rebuild_failure_names_index_project_tool() {
    use std::os::unix::fs::PermissionsExt;

    let temp = build_workspace();
    let root = temp.path();
    let reflex_dir = root.join(".reflex");

    truncate(&reflex_dir.join("content.bin"));
    // Read-only cache dir: the forced rebuild cannot delete or create files.
    fs::set_permissions(&reflex_dir, fs::Permissions::from_mode(0o555)).unwrap();

    let resp = call(root, SEARCH_FN);

    // Restore so TempDir cleanup works whatever the outcome.
    fs::set_permissions(&reflex_dir, fs::Permissions::from_mode(0o755)).unwrap();

    // Running as root bypasses the permission bits; then recovery just works.
    if resp.get("error").is_none() {
        eprintln!("skipping assertion: process can write to a 0o555 directory (root?)");
        return;
    }
    let msg = resp["error"]["message"].as_str().unwrap();
    assert!(
        msg.contains("index_project"),
        "must name the MCP tool: {msg}"
    );
    assert!(
        !msg.contains("rfx index"),
        "must not tell an agent to run the CLI: {msg}"
    );
}

#[test]
fn index_not_found_error_names_index_project_tool() {
    let temp = TempDir::new().unwrap();
    fs::write(temp.path().join("a.rs"), "fn a() {}\n").unwrap();

    let resp = call(temp.path(), SEARCH_FN);
    let msg = resp["error"]["message"].as_str().unwrap();
    assert_eq!(resp["error"]["data"]["kind"], "IndexNotFound");
    assert!(msg.contains("index_project"), "{msg}");
    assert!(!msg.contains("rfx index"), "{msg}");
}
