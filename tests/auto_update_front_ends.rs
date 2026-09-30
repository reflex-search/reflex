//! Every front end answers from an updated index: `rfx query`, `rfx deps`,
//! `rfx mcp` tools. `--no-update` keeps the pre-auto-update behaviour.

#![cfg(unix)]

use reflex::auto_update::UpdateOptions;
use reflex::mcp::run_mcp_server_io_with;
use serde_json::{Value, json};
use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

const RFX: &str = env!("CARGO_BIN_EXE_rfx");

fn git(root: &Path, args: &[&str]) {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .unwrap();
    assert!(out.status.success(), "git {args:?}");
}

fn write(root: &Path, rel: &str, body: &str) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, body).unwrap();
}

fn repo() -> TempDir {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    git(root, &["init", "-q"]);
    git(root, &["config", "user.email", "t@example.com"]);
    git(root, &["config", "user.name", "T"]);
    write(root, ".gitignore", ".reflex/\n");
    write(root, "src/lib.rs", "mod a;\npub fn front_lib_token() {}\n");
    write(root, "src/a.rs", "pub fn front_a_token() {}\n");
    git(root, &["add", "-A"]);
    git(root, &["commit", "-qm", "initial"]);
    temp
}

/// Run `rfx` in `root`; (success, stdout, stderr).
fn rfx(root: &Path, args: &[&str]) -> (bool, String, String) {
    let out = Command::new(RFX)
        .args(args)
        .current_dir(root)
        // No symbol pass left running in the temp dir after the test.
        .env("REFLEX_SYMBOL_THREADS", "1")
        .output()
        .unwrap();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn query_json(root: &Path, extra: &[&str], pattern: &str) -> Value {
    let mut args = vec!["query", pattern, "--json"];
    args.extend_from_slice(extra);
    let (ok, out, err) = rfx(root, &args);
    assert!(ok, "rfx query failed: {err}");
    serde_json::from_str(&out).unwrap_or_else(|e| panic!("{e}: {out}"))
}

fn hits(v: &Value) -> usize {
    v["results"]
        .as_array()
        .map(|files| {
            files
                .iter()
                .map(|f| f["matches"].as_array().map_or(0, Vec::len))
                .sum()
        })
        .unwrap_or(0)
}

#[test]
fn rfx_query_builds_a_missing_index() {
    let temp = repo();
    let v = query_json(temp.path(), &[], "front_a_token");
    assert_eq!(v["status"], "fresh", "{v}");
    assert_eq!(hits(&v), 1);

    let other = repo();
    let (ok, out, err) = rfx(
        other.path(),
        &["query", "front_a_token", "--json", "--no-update"],
    );
    assert!(!ok);
    // `--json` reports the error on stdout.
    assert!(
        out.contains("Index not found") || err.contains("Index not found"),
        "{out}{err}"
    );
}

#[test]
fn rfx_query_sees_an_edit_and_no_update_does_not() {
    let temp = repo();
    let root = temp.path();
    let (ok, _, err) = rfx(root, &["index", "--quiet"]);
    assert!(ok, "{err}");

    write(root, "src/a.rs", "pub fn front_edit_token() {}\n");
    let stale = query_json(root, &["--no-update"], "front_edit_token");
    assert_eq!(stale["status"], "stale", "{stale}");
    assert_eq!(hits(&stale), 0);

    let v = query_json(root, &[], "front_edit_token");
    assert_eq!(v["status"], "fresh", "{v}");
    assert_eq!(v["can_trust_results"], true);
    assert_eq!(hits(&v), 1);

    // `--no-update` may come before the subcommand too (a global flag).
    let (ok, _, err) = rfx(root, &["--no-update", "query", "front_edit_token"]);
    assert!(ok, "{err}");
}

#[test]
fn rfx_query_symbols_sees_an_edit() {
    let temp = repo();
    let root = temp.path();
    rfx(root, &["index", "--quiet"]);
    write(root, "src/a.rs", "pub fn front_symbol_token() {}\n");
    let v = query_json(root, &["--symbols"], "front_symbol_token");
    assert_eq!(hits(&v), 1, "{v}");
}

#[test]
fn rfx_deps_sees_a_new_import() {
    let temp = repo();
    let root = temp.path();
    rfx(root, &["index", "--quiet"]);
    write(root, "src/b.rs", "pub fn front_b_token() {}\n");
    write(
        root,
        "src/lib.rs",
        "mod a;\nmod b;\npub fn front_lib_token() {}\n",
    );
    let (ok, out, err) = rfx(root, &["deps", "src/lib.rs", "--json"]);
    assert!(ok, "{err}");
    assert!(out.contains(r#""path":"b""#), "{out}");
}

#[test]
fn rfx_stats_builds_a_missing_index() {
    let temp = repo();
    let (ok, out, err) = rfx(temp.path(), &["stats", "--json"]);
    assert!(ok, "{err}");
    assert!(out.contains("\"total_files\""), "{out}");
    assert!(err.contains("Building the index"), "{err}");
}

fn mcp(root: &Path, tool: &str, args: Value, update: bool) -> Value {
    let req = json!({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": { "name": tool, "arguments": args }
    });
    let mut out: Vec<u8> = Vec::new();
    run_mcp_server_io_with(
        root,
        Cursor::new(format!("{req}\n").into_bytes()),
        &mut out,
        true,
        update.then(UpdateOptions::library),
    )
    .unwrap();
    let text = String::from_utf8(out).unwrap();
    let v: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    assert!(v.get("error").is_none(), "{tool}: {}", v["error"]);
    let payload = v["result"]["content"][0]["text"].as_str().unwrap();
    serde_json::from_str(payload).unwrap_or_else(|_| json!({ "raw": payload }))
}

#[test]
fn mcp_search_after_an_edit_is_fresh_without_index_project() {
    let temp = repo();
    let root = temp.path();
    // No index at all: the first search builds it.
    let first = mcp(
        root,
        "search_code",
        json!({ "pattern": "front_a_token" }),
        true,
    );
    assert_eq!(first["status"], "fresh", "{first}");

    write(root, "src/a.rs", "pub fn front_mcp_token() {}\n");
    reflex::query::invalidate_caches(root);
    // The probe reports the truth and does not repair.
    let probe = mcp(root, "check_index_status", json!({}), true);
    assert_eq!(probe["status"], "stale", "{probe}");

    let v = mcp(
        root,
        "search_code",
        json!({ "pattern": "front_mcp_token" }),
        true,
    );
    assert_eq!(v["status"], "fresh", "{v}");
    assert_eq!(v["can_trust_results"], true, "{v}");
    assert_eq!(v["rows"].as_array().map_or(0, Vec::len), 1, "{v}");
}

#[test]
fn mcp_dependency_tools_update_first() {
    let temp = repo();
    let root = temp.path();
    mcp(
        root,
        "search_code",
        json!({ "pattern": "front_a_token" }),
        true,
    );
    write(root, "src/b.rs", "pub fn front_b_token() {}\n");
    write(
        root,
        "src/lib.rs",
        "mod a;\nmod b;\npub fn front_lib_token() {}\n",
    );
    reflex::query::invalidate_caches(root);
    let v = mcp(
        root,
        "get_dependencies",
        json!({ "path": "src/lib.rs" }),
        true,
    );
    assert!(v.to_string().contains("src/b.rs"), "{v}");
}

#[test]
fn mcp_without_update_reports_stale() {
    let temp = repo();
    let root = temp.path();
    mcp(
        root,
        "search_code",
        json!({ "pattern": "front_a_token" }),
        true,
    );
    write(root, "src/a.rs", "pub fn front_stale_token() {}\n");
    reflex::query::invalidate_caches(root);
    let v = mcp(
        root,
        "search_code",
        json!({ "pattern": "front_stale_token" }),
        false,
    );
    assert_eq!(v["status"], "stale", "{v}");
}

/// Every front end builds its query engine through the update path. A new
/// `QueryEngine::new` outside these files would answer from a stale index.
#[test]
fn query_engines_are_built_by_the_known_front_ends_only() {
    let allowed = [
        "src/query/mod.rs",       // the engine itself and its tests
        "src/cli/mod.rs",         // `cli::engine`: with the update unless --no-update
        "src/mcp.rs",             // `engine_in`: with the server's update
        "src/cli/serve.rs",       // per request, with the server's update
        "src/interactive/app.rs", // with the update unless --no-update
        "src/semantic/tools.rs",  // `rfx ask`: updated before the command runs
        "src/semantic/executor.rs",
    ];
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut offenders = Vec::new();
    let mut stack = vec![root.join("src")];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let rel = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                let body = fs::read_to_string(&path).unwrap();
                if body.contains("QueryEngine::new(") && !allowed.contains(&rel.as_str()) {
                    offenders.push(rel);
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "QueryEngine::new outside the front ends: {offenders:?}"
    );
}

/// Every object-shaped answer carries `can_trust_results`: agents no longer call
/// `check_index_status`, so the answer itself must say whether it can be trusted.
#[test]
fn mcp_object_answers_carry_can_trust_results() {
    let temp = repo();
    let root = temp.path();
    mcp(
        root,
        "search_code",
        json!({ "pattern": "front_a_token" }),
        true,
    );
    let calls = [
        (
            "search_code",
            json!({ "pattern": "front_a_token", "mode": "count" }),
        ),
        (
            "search_regex",
            json!({ "pattern": "front_\\w+", "mode": "count" }),
        ),
        ("list_locations", json!({ "pattern": "front_a_token" })),
        ("count_occurrences", json!({ "pattern": "front_a_token" })),
        ("find_references", json!({ "pattern": "front_a_token" })),
        ("find_hotspots", json!({})),
        ("find_circular", json!({})),
        ("find_unused", json!({})),
        ("find_islands", json!({})),
        ("analyze_summary", json!({})),
    ];
    for (tool, args) in &calls {
        let v = mcp(root, tool, args.clone(), true);
        assert_eq!(v["can_trust_results"], true, "{tool}: {v}");
    }
    // A stale index and no update: every answer says so.
    write(
        root,
        "src/a.rs",
        "pub fn front_a_token() { let _changed = 1; }\n",
    );
    reflex::query::invalidate_caches(root);
    for (tool, args) in &calls {
        let v = mcp(root, tool, args.clone(), false);
        assert_eq!(v["can_trust_results"], false, "{tool}: {v}");
        assert_eq!(v["status"], "stale", "{tool}: {v}");
    }
}

/// The server no longer sends agents to the status probe or a manual reindex.
#[test]
fn mcp_text_does_not_ask_for_manual_reindexing() {
    let temp = repo();
    let root = temp.path();
    let mut out: Vec<u8> = Vec::new();
    let reqs = [
        json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
    ];
    let input: String = reqs.iter().map(|r| format!("{r}\n")).collect();
    run_mcp_server_io_with(root, Cursor::new(input.into_bytes()), &mut out, true, None).unwrap();
    let text = String::from_utf8(out).unwrap();
    for stale_advice in [
        "Call `index_project` and retry",
        "call `index_project`, then retry",
        "Call this at session start",
        "call index_project, then retry",
    ] {
        assert!(!text.contains(stale_advice), "still says: {stale_advice}");
    }
    assert!(text.contains("The index updates itself before every call"));
}
