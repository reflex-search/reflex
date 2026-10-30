//! The MCP tool surface: 6 listed tools with short schemas (10 until 2026-10-27;
//! `search_ast`, `gather_context`, `index_project` and `check_index_status` are
//! listed only on request), and the 8 names removed on 2026-09-30 still callable
//! (unlisted, with a deprecation warning). Claude Code carries every listed schema
//! on every turn, so the size of `tools/list` is guarded here too.

use reflex::auto_update::UpdateOptions;
use reflex::mcp::{run_mcp_server_io_in, run_mcp_server_io_with};
use reflex::{CacheManager, IndexConfig, Indexer};
use serde_json::{Value, json};
use std::fs;
use std::io::Cursor;
use std::path::Path;
use tempfile::TempDir;

fn indexed() -> TempDir {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("src/lib.rs"),
        "mod a;\nmod b;\npub fn surface_token() {}\n",
    )
    .unwrap();
    fs::write(
        root.join("src/a.rs"),
        "use crate::b;\npub fn surface_a() { surface_token(); }\n",
    )
    .unwrap();
    fs::write(
        root.join("src/b.rs"),
        "pub fn surface_b() { surface_token(); }\n",
    )
    .unwrap();
    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .unwrap();
    temp
}

fn rpc(root: &Path, method: &str, params: Value) -> Value {
    let req = json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params});
    let mut out: Vec<u8> = Vec::new();
    run_mcp_server_io_in(
        root,
        Cursor::new(format!("{req}\n").into_bytes()),
        &mut out,
        true,
    )
    .unwrap();
    serde_json::from_str(String::from_utf8(out).unwrap().lines().next().unwrap()).unwrap()
}

fn call(root: &Path, tool: &str, args: Value) -> Value {
    let v = rpc(root, "tools/call", json!({"name": tool, "arguments": args}));
    assert!(v.get("error").is_none(), "{tool}: {}", v["error"]);
    let text = v["result"]["content"][0]["text"].as_str().unwrap();
    serde_json::from_str(text).unwrap_or_else(|_| json!({ "raw": text }))
}

fn warnings(v: &Value) -> String {
    v["warnings"].to_string()
}

/// One request to a server that updates the index first, as `rfx mcp` does.
fn rpc_updating(root: &Path, method: &str, params: Value) -> Value {
    let req = json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params});
    let mut out: Vec<u8> = Vec::new();
    run_mcp_server_io_with(
        root,
        Cursor::new(format!("{req}\n").into_bytes()),
        &mut out,
        true,
        Some(UpdateOptions::library()),
    )
    .unwrap();
    serde_json::from_str(String::from_utf8(out).unwrap().lines().next().unwrap()).unwrap()
}

fn names(tools: &Value) -> Vec<&str> {
    tools
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect()
}

#[test]
fn six_tools_are_listed_in_a_small_schema() {
    let temp = indexed();
    let tools = rpc_updating(temp.path(), "tools/list", json!({}))["result"]["tools"].clone();
    assert_eq!(
        names(&tools),
        [
            "search_code",
            "search_regex",
            "list_locations",
            "find_references",
            "get_dependencies",
            "analyze",
        ]
    );
    // Was 44 KB with 17 tools, 10.3 KB with 10, 8.1 KB before HIDDEN_PARAMS; every byte here is carried on every
    // agent turn.
    let bytes = serde_json::to_string(&tools).unwrap().len();
    assert!(bytes <= 7_000, "tools/list is {bytes} bytes");
}

/// Without the automatic update, the index tools are the only way to refresh the
/// index, so they are listed.
#[test]
fn no_update_lists_the_index_tools() {
    let temp = indexed();
    let tools = rpc(temp.path(), "tools/list", json!({}))["result"]["tools"].clone();
    let listed = names(&tools);
    assert!(listed.contains(&"index_project"), "{listed:?}");
    assert!(listed.contains(&"check_index_status"), "{listed:?}");
    assert!(!listed.contains(&"search_ast"), "{listed:?}");
    assert!(!listed.contains(&"gather_context"), "{listed:?}");
}

/// Unlisted tools still answer a client that calls them by name.
#[test]
fn hidden_tools_still_answer_by_name() {
    let temp = indexed();
    for (tool, args) in [
        (
            "search_ast",
            json!({"pattern": "(function_item) @fn", "lang": "rust", "glob": ["src/**/*.rs"]}),
        ),
        ("gather_context", json!({"structure": true})),
        ("check_index_status", json!({})),
    ] {
        let v = rpc_updating(
            temp.path(),
            "tools/call",
            json!({"name": tool, "arguments": args}),
        );
        assert!(v.get("error").is_none(), "{tool}: {v}");
        assert!(!v.to_string().contains("deprecated"), "{tool}: {v}");
    }
}

/// With `index_project` unlisted, `action_required` names the CLI: Claude Code can
/// call only listed tools.
#[test]
fn action_required_names_the_cli_when_index_project_is_unlisted() {
    let temp = TempDir::new().unwrap();
    let args = json!({"name": "check_index_status", "arguments": {}});
    let v = rpc_updating(temp.path(), "tools/call", args.clone());
    let text = v["result"]["content"][0]["text"].as_str().unwrap();
    let status: Value = serde_json::from_str(text).unwrap();
    assert_eq!(status["status"], "missing", "{status}");
    assert_eq!(status["action_required"], "rfx index", "{status}");

    let v = rpc(temp.path(), "tools/call", args);
    let text = v["result"]["content"][0]["text"].as_str().unwrap();
    let status: Value = serde_json::from_str(text).unwrap();
    assert_eq!(status["action_required"], "index_project", "{status}");
}

#[test]
fn analyze_runs_each_kind() {
    let temp = indexed();
    for kind in ["summary", "hotspots", "circular", "unused", "islands"] {
        let v = call(temp.path(), "analyze", json!({"kind": kind}));
        assert!(v.get("raw").is_none(), "{kind}: {v}");
        assert!(v["can_trust_results"].is_boolean(), "{kind}: {v}");
    }
    let bad = rpc(
        temp.path(),
        "tools/call",
        json!({"name": "analyze", "arguments": {"kind": "bogus"}}),
    );
    assert!(
        bad["error"]["message"]
            .as_str()
            .unwrap()
            .contains("kind must be one of"),
        "{bad}"
    );
}

#[test]
fn get_dependencies_covers_reverse_and_depth() {
    let temp = indexed();
    let forward = call(
        temp.path(),
        "get_dependencies",
        json!({"path": "src/lib.rs"}),
    );
    assert!(forward.to_string().contains("src/a.rs"), "{forward}");
    let reverse = call(
        temp.path(),
        "get_dependencies",
        json!({"path": "src/b.rs", "reverse": true}),
    );
    assert!(reverse.to_string().contains("src/lib.rs"), "{reverse}");
    assert_eq!(
        reverse,
        call(temp.path(), "get_dependents", json!({"path": "src/b.rs"}))
    );
    let deep = call(
        temp.path(),
        "get_dependencies",
        json!({"path": "src/lib.rs", "depth": 2}),
    );
    assert!(deep.is_array(), "{deep}");
    let both = rpc(
        temp.path(),
        "tools/call",
        json!({"name": "get_dependencies", "arguments": {"path": "src/lib.rs", "reverse": true, "depth": 2}}),
    );
    assert!(
        both["error"]["message"]
            .as_str()
            .unwrap()
            .contains("cannot be combined"),
        "{both}"
    );
}

#[test]
fn removed_names_still_work_with_a_deprecation_warning() {
    let temp = indexed();
    let cases = [
        (
            "count_occurrences",
            json!({"pattern": "surface_token"}),
            "search_code with mode",
        ),
        ("find_hotspots", json!({}), "analyze with kind"),
        ("find_circular", json!({}), "analyze with kind"),
        ("find_unused", json!({}), "analyze with kind"),
        ("find_islands", json!({}), "analyze with kind"),
        ("analyze_summary", json!({}), "analyze with kind"),
    ];
    for (name, args, hint) in cases {
        let req = json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": name, "arguments": args}});
        let mut out: Vec<u8> = Vec::new();
        run_mcp_server_io_in(
            temp.path(),
            Cursor::new(format!("{req}\n").into_bytes()),
            &mut out,
            true,
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(!text.contains("\"error\""), "{name}: {text}");
        assert!(text.contains("is deprecated"), "{name}: {text}");
        assert!(text.contains(hint), "{name} should name {hint}: {text}");
    }
    // `get_dependents` and `get_transitive_deps` answer bare arrays: they work, but
    // have no place for the warning.
    let dependents = call(temp.path(), "get_dependents", json!({"path": "src/b.rs"}));
    assert!(dependents.is_array(), "{dependents}");
    let deep = call(
        temp.path(),
        "get_transitive_deps",
        json!({"path": "src/lib.rs"}),
    );
    assert!(deep.is_array(), "{deep}");

    let count = call(
        temp.path(),
        "count_occurrences",
        json!({"pattern": "surface_token"}),
    );
    assert_eq!(count["total"], 3, "{count}");
    assert!(warnings(&count).contains("deprecated"));
}

#[test]
fn count_mode_reports_files() {
    let temp = indexed();
    let v = call(
        temp.path(),
        "search_code",
        json!({"pattern": "surface_token", "mode": "count"}),
    );
    assert_eq!(v["count"], 3, "{v}");
    assert_eq!(v["files"], 3, "{v}");
}

#[test]
fn list_locations_adds_the_line_on_request() {
    let temp = indexed();
    let plain = call(
        temp.path(),
        "list_locations",
        json!({"pattern": "surface_token"}),
    );
    // `{path: [line, ...]}`: a bare line number per match.
    assert!(plain["locations"]["src/lib.rs"][0].is_u64(), "{plain}");

    let v = call(
        temp.path(),
        "list_locations",
        json!({"pattern": "surface_token", "preview": true}),
    );
    // `{path: [[line, text], ...]}`
    let files = v["locations"].as_object().unwrap();
    let total: usize = files.values().map(|l| l.as_array().unwrap().len()).sum();
    assert_eq!(total, 3, "{v}");
    let def = &files["src/lib.rs"][0];
    assert_eq!(def[1], "pub fn surface_token() {}", "{v}");
    // A string "true" is coerced like every other flag.
    let coerced = call(
        temp.path(),
        "list_locations",
        json!({"pattern": "surface_token", "preview": "true"}),
    );
    assert!(
        coerced["locations"]["src/lib.rs"][0][1].is_string(),
        "{coerced}"
    );
}

#[test]
fn list_locations_attaches_imports_on_request() {
    let temp = indexed();
    let plain = call(
        temp.path(),
        "list_locations",
        json!({"pattern": "surface_token"}),
    );
    assert!(plain.get("dependencies").is_none(), "{plain}");

    let v = call(
        temp.path(),
        "list_locations",
        json!({"pattern": "surface_token", "dependencies": true}),
    );
    // `{path: [import, ...]}` beside `locations`, which keeps its shape.
    assert!(v["locations"]["src/lib.rs"][0].is_u64(), "{v}");
    let deps = v["dependencies"].as_object().expect("dependencies map");
    let own = call(
        temp.path(),
        "get_dependencies",
        json!({"path": "src/lib.rs"}),
    );
    assert_eq!(deps["src/lib.rs"], own, "{v}");
    // A matching file with no imports is left out.
    assert!(!deps.contains_key("src/b.rs"), "{v}");
}

/// An unknown `lang` was dropped, so the search ran on every language. It is now an
/// invalid-params error, like the CLI's.
#[test]
fn unknown_lang_is_an_error_not_a_search_of_everything() {
    let temp = indexed();
    for (tool, args) in [
        (
            "search_code",
            json!({"pattern": "surface_token", "lang": "cobol"}),
        ),
        (
            "search_regex",
            json!({"pattern": "surface_token", "lang": "pyhton"}),
        ),
        (
            "list_locations",
            json!({"pattern": "surface_token", "lang": "rustt"}),
        ),
        (
            "find_references",
            json!({"pattern": "surface_token", "lang": "go lang"}),
        ),
        ("index_project", json!({"languages": ["rust", "nope"]})),
    ] {
        let v = rpc(
            temp.path(),
            "tools/call",
            json!({"name": tool, "arguments": args}),
        );
        assert_eq!(v["error"]["code"], -32602, "{tool}: {v}");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("Unknown language"), "{tool}: {msg}");
        assert!(msg.contains("python (py)"), "{tool}: {msg}");
    }
    // Names are matched without case; an empty `lang` is no filter.
    for lang in ["RUST", "rs", ""] {
        let v = call(
            temp.path(),
            "list_locations",
            json!({"pattern": "surface_token", "lang": lang}),
        );
        assert_eq!(v["total_locations"], 3, "{lang:?}: {v}");
    }
}

/// `swift` was not a language name, so `lang: "swift"` was an error (and the CLI's
/// Swift warning never ran). Swift files are indexed as text with no symbol parser:
/// a full-text search finds them, and a symbol search says why it finds nothing.
#[test]
fn swift_is_a_language_and_a_symbol_search_says_it_has_no_parser() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    fs::write(
        root.join("App.swift"),
        "func swiftProbe() {}
let x = swiftProbe()
",
    )
    .unwrap();
    fs::write(root.join("app.py"), "def swiftProbe():\n    pass\n").unwrap();
    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .unwrap();

    let v = call(
        root,
        "list_locations",
        json!({"pattern": "swiftProbe", "lang": "swift"}),
    );
    assert_eq!(v["locations"], json!({"App.swift": [1, 2]}), "{v}");
    assert!(v.get("warnings").is_none(), "{v}");

    let v = call(
        root,
        "search_code",
        json!({"pattern": "swiftProbe", "lang": "Swift", "symbols": true}),
    );
    assert_eq!(v["total_count"], 0, "{v}");
    assert!(warnings(&v).contains("Swift has no symbol parser"), "{v}");

    // The definition search is the symbol search; its note reaches the answer.
    let v = call(
        root,
        "find_references",
        json!({"pattern": "swiftProbe", "lang": "swift"}),
    );
    assert_eq!(v["total_references"], 2, "{v}");
    assert!(v["definition"].is_null(), "{v}");
    assert!(warnings(&v).contains("Swift has no symbol parser"), "{v}");

    // A language with a parser carries no such note.
    let v = call(
        root,
        "search_code",
        json!({"pattern": "swiftProbe", "lang": "python", "symbols": true}),
    );
    assert_eq!(v["total_count"], 1, "{v}");
    assert!(!warnings(&v).contains("no symbol parser"), "{v}");
}

/// Every language `lang` accepts has an import extractor or, in the text tiers and
/// Swift, no imports. The warning said "only Rust" and was false: the imports were in the
/// same reply.
#[test]
fn search_code_returns_imports_for_python_without_a_warning() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    fs::write(
        root.join("main.py"),
        "import os\n\ndef dep_probe():\n    pass\n",
    )
    .unwrap();
    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .unwrap();

    let v = call(
        root,
        "search_code",
        json!({"pattern": "dep_probe", "lang": "python", "dependencies": true}),
    );
    assert!(v.to_string().contains("\"os\""), "{v}");
    assert!(!v.to_string().contains("only supported"), "{v}");
}

/// 120 Go files importing a missing package of their own module: a Go graph with
/// 0 % of its internal imports resolved.
fn unresolved_go() -> TempDir {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    fs::write(root.join("go.mod"), "module example.com/m\n\ngo 1.22\n").unwrap();
    for i in 0..120 {
        let dir = root.join(format!("pkg/p{i}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("p.go"),
            format!("package p{i}\n\nimport \"example.com/m/pkg/missing\"\n\nvar _ = missing.X\n"),
        )
        .unwrap();
    }
    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .unwrap();
    temp
}

#[test]
fn analyze_warns_on_low_resolution() {
    let temp = unresolved_go();
    for kind in ["summary", "islands", "unused"] {
        let v = call(temp.path(), "analyze", json!({"kind": kind}));
        assert!(
            warnings(&v).contains("Go: 0 of 120 internal imports (0.0%) resolve"),
            "{kind}: {v}"
        );
    }
    // A fully resolved graph says nothing.
    let fine = indexed();
    let v = call(fine.path(), "analyze", json!({"kind": "summary"}));
    assert!(v.get("warnings").is_none(), "{v}");
}

#[test]
fn get_dependencies_warning_is_second_content_item() {
    let temp = unresolved_go();
    let v = rpc(
        temp.path(),
        "tools/call",
        json!({"name": "get_dependencies", "arguments": {"path": "pkg/p0/p.go"}}),
    );
    let content = v["result"]["content"].as_array().unwrap();
    // The array answer keeps its shape.
    let first: Value = serde_json::from_str(content[0]["text"].as_str().unwrap()).unwrap();
    assert!(first.is_array(), "{first}");
    assert_eq!(content.len(), 2, "{v}");
    let text = content[1]["text"].as_str().unwrap();
    assert!(text.starts_with("warnings: Go: 0 of 120"), "{text}");
}

#[test]
fn vendored_file_warns_and_summary_counts_vendored() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    for (rel, body) in [
        ("go.mod", "module example.com/m\n\ngo 1.22\n"),
        (
            "main.go",
            "package main\n\nimport \"golang.org/x/sys/unix\"\n\nfunc main() { unix.F() }\n",
        ),
        (
            "vendor/modules.txt",
            "# golang.org/x/sys v0.20.0\ngolang.org/x/sys/unix\n",
        ),
        (
            "vendor/golang.org/x/sys/unix/u.go",
            "package unix\n\nfunc F() {}\n",
        ),
    ] {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, body).unwrap();
    }
    Indexer::new(CacheManager::new(root), IndexConfig::default())
        .index(root, false)
        .unwrap();

    let v = rpc(
        root,
        "tools/call",
        json!({"name": "get_dependencies",
               "arguments": {"path": "vendor/golang.org/x/sys/unix/u.go", "reverse": true}}),
    );
    let content = v["result"]["content"].as_array().unwrap();
    assert_eq!(content.len(), 2, "{v}");
    let text = content[1]["text"].as_str().unwrap();
    assert!(
        text.contains("vendor/golang.org/x/sys/unix/u.go is vendored"),
        "{text}"
    );

    let v = call(root, "analyze", json!({"kind": "summary"}));
    assert_eq!(v["vendored_files"], 1, "{v}");
    assert_eq!(v["islands"], 1, "only main.go: {v}");
}
