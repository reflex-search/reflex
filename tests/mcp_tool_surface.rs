//! The MCP tool surface after the 2026-09-30 merge: 10 listed tools with short
//! schemas, and the 8 removed names still callable (unlisted, with a deprecation
//! warning). Claude Code carries every listed schema on every turn, so the size of
//! `tools/list` is guarded here too.

use reflex::mcp::run_mcp_server_io_in;
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

#[test]
fn ten_tools_are_listed_in_a_small_schema() {
    let temp = indexed();
    let tools = rpc(temp.path(), "tools/list", json!({}))["result"]["tools"].clone();
    let names: Vec<&str> = tools
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "search_code",
            "search_regex",
            "list_locations",
            "find_references",
            "search_ast",
            "get_dependencies",
            "analyze",
            "gather_context",
            "index_project",
            "check_index_status",
        ]
    );
    // Was 44 KB with 17 tools; every byte here is carried on every agent turn.
    let bytes = serde_json::to_string(&tools).unwrap().len();
    assert!(bytes <= 14_000, "tools/list is {bytes} bytes");
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

/// Every language `lang` accepts has an import extractor or, in the text tiers, no
/// imports. The warning said "only Rust" and was false: the imports were in the
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
