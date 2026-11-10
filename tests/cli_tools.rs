//! CLI parity with the MCP tools (task 1.4 of `.context/AGENT_EDGE_PLAN.md`): an agent
//! that runs `rfx` through Bash gets the MCP answers.
//!
//! - `rfx refs` = `find_references`, `rfx locations` = `list_locations`: `--json` is
//!   the MCP answer, value for value; the text form is grep-like.
//! - Every JSON-object answer carries `status` and `can_trust_results`
//!   (`rfx query --count --json`, `rfx analyze --json`).
//! - `rfx` run in a subdirectory answers from the enclosing index; it does not build
//!   a second one there.

use reflex::mcp::run_mcp_server_io_in;
use serde_json::{Value, json};
use std::io::Cursor;
use std::path::Path;
use std::process::{Command, Output};
use tempfile::TempDir;

/// `alpha` is defined once, called twice in code, and named in a comment, a string
/// and the README (text tier).
fn indexed_fixture() -> TempDir {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    std::fs::create_dir_all(root.join(".git")).unwrap();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("src/lib.rs"),
        "pub mod util;\n\
         \n\
         pub fn alpha() -> u32 {\n\
         \x20   1\n\
         }\n\
         \n\
         pub fn beta() -> u32 {\n\
         \x20   // alpha is the base case\n\
         \x20   alpha() + alpha()\n\
         }\n",
    )
    .unwrap();
    std::fs::write(
        root.join("src/util.rs"),
        "pub fn name() -> &'static str {\n\
         \x20   \"alpha\"\n\
         }\n\
         \n\
         pub fn gamma() -> u32 {\n\
         \x20   crate::alpha()\n\
         }\n",
    )
    .unwrap();
    std::fs::write(root.join("README.md"), "Call alpha first.\n").unwrap();
    let out = rfx(root, &["index", "--quiet"]);
    assert!(out.status.success(), "rfx index failed: {}", stderr(&out));
    temp
}

fn rfx(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rfx"))
        .current_dir(dir)
        .args(args)
        .output()
        .expect("spawn rfx")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn json_out(dir: &Path, args: &[&str]) -> Value {
    let out = rfx(dir, args);
    assert!(
        out.status.success(),
        "rfx {args:?} failed: {}",
        stderr(&out)
    );
    serde_json::from_str(stdout(&out).trim()).unwrap_or_else(|e| {
        panic!(
            "rfx {args:?} must print one JSON value: {e}\n{}",
            stdout(&out)
        )
    })
}

/// One MCP tool call, in-process; the parsed `content[0].text`.
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
    .expect("server should not error");
    let line = String::from_utf8(out).unwrap();
    let v: Value = serde_json::from_str(line.lines().find(|l| !l.is_empty()).unwrap()).unwrap();
    assert!(v.get("error").is_none(), "{tool} errored: {}", v["error"]);
    serde_json::from_str(v["result"]["content"][0]["text"].as_str().unwrap()).unwrap()
}

#[test]
fn refs_json_is_the_find_references_answer() {
    let temp = indexed_fixture();
    let root = temp.path();
    for (cli, mcp) in [
        (vec!["refs", "alpha", "--json"], json!({"pattern": "alpha"})),
        (
            vec!["refs", "alpha", "--json", "--include-strings"],
            json!({"pattern": "alpha", "include_strings": true}),
        ),
        (
            vec!["refs", "alpha", "--json", "--count"],
            json!({"pattern": "alpha", "mode": "count"}),
        ),
        (
            vec!["refs", "alpha", "--json", "-n", "1", "-o", "1"],
            json!({"pattern": "alpha", "limit": 1, "offset": 1}),
        ),
    ] {
        assert_eq!(
            json_out(root, &cli),
            call_tool(root, "find_references", mcp),
            "{cli:?}"
        );
    }
}

#[test]
fn locations_json_is_the_list_locations_answer() {
    let temp = indexed_fixture();
    let root = temp.path();
    for (cli, mcp) in [
        (
            vec!["locations", "alpha", "--json"],
            json!({"pattern": "alpha"}),
        ),
        (
            vec!["loc", "alpha", "--json", "--preview", "--glob", "src/**"],
            json!({"pattern": "alpha", "preview": true, "glob": ["src/**"]}),
        ),
        (
            vec!["locations", "alph", "--json", "--contains", "-l", "rust"],
            json!({"pattern": "alph", "contains": true, "lang": "rust"}),
        ),
    ] {
        assert_eq!(
            json_out(root, &cli),
            call_tool(root, "list_locations", mcp),
            "{cli:?}"
        );
    }
}

/// The code-only answer: the comment line (8) and the string line (util.rs 2) are
/// left out, the README is not a code file.
#[test]
fn refs_text_lists_uses_in_code_only() {
    let temp = indexed_fixture();
    let out = rfx(temp.path(), &["refs", "alpha"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(
        stdout(&out),
        "definition src/lib.rs:3 Function alpha\n\
         src/lib.rs:3:pub fn alpha() -> u32 {\n\
         src/lib.rs:9:    alpha() + alpha()\n\
         src/util.rs:6:    crate::alpha()\n"
    );
    assert!(
        stderr(&out).contains("3 references in code; 2 lines only in comments or strings left out"),
        "{}",
        stderr(&out)
    );
    let count = rfx(temp.path(), &["refs", "alpha", "--count"]);
    assert_eq!(stdout(&count).trim(), "3");
}

#[test]
fn locations_text_is_one_line_per_file() {
    let temp = indexed_fixture();
    let out = rfx(temp.path(), &["locations", "alpha"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(
        stdout(&out),
        "README.md:1\nsrc/lib.rs:3,8,9\nsrc/util.rs:2,6\n"
    );
    let preview = rfx(
        temp.path(),
        &["locations", "alpha", "--preview", "--file", "util"],
    );
    assert_eq!(
        stdout(&preview),
        "src/util.rs:2:\"alpha\"\nsrc/util.rs:6:crate::alpha()\n"
    );
}

#[test]
fn json_objects_carry_the_freshness_fields() {
    let temp = indexed_fixture();
    let root = temp.path();
    for args in [
        vec!["query", "alpha", "--count", "--json"],
        vec!["analyze", "--json"],
        vec!["analyze", "--hotspots", "--json"],
        vec!["analyze", "--unused", "--json"],
    ] {
        let v = json_out(root, &args);
        assert_eq!(v["status"], "fresh", "{args:?}: {v}");
        assert_eq!(v["can_trust_results"], true, "{args:?}: {v}");
    }
}

/// An agent `cd`s into `src/`: the answers come from the root index, paths stay
/// relative to it, a relative `deps` path is rebased, and no `src/.reflex/` appears.
#[test]
fn a_subdirectory_uses_the_enclosing_index() {
    let temp = indexed_fixture();
    let src = temp.path().join("src");
    let out = rfx(&src, &["locations", "alpha"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        stdout(&out).contains("src/lib.rs:3,8,9"),
        "{}",
        stdout(&out)
    );
    assert!(
        stderr(&out).contains("using the index at"),
        "{}",
        stderr(&out)
    );
    let deps = json_out(&src, &["deps", "util.rs", "--reverse", "--json"]);
    assert_eq!(deps, json!(["src/lib.rs"]));
    assert!(
        !src.join(".reflex").exists(),
        "a second index was built in src/"
    );
}

/// `--reverse` lists direct importers only; a depth was ignored without a word.
#[test]
fn deps_refuses_reverse_with_depth() {
    let temp = indexed_fixture();
    let out = rfx(
        temp.path(),
        &["deps", "src/util.rs", "--reverse", "--depth", "2"],
    );
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("cannot be combined"),
        "{}",
        stderr(&out)
    );
}
