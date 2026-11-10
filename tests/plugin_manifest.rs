//! The Claude Code marketplace (`.claude-plugin/marketplace.json`) and its plugins
//! (`plugins/*`). `cargo release` bumps the plugin versions
//! (`pre-release-replacements` in `release.toml`); these tests catch a drift.

use serde_json::Value;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn marketplace_plugins() -> Vec<(String, PathBuf)> {
    let market = json(&root().join(".claude-plugin/marketplace.json"));
    market["plugins"]
        .as_array()
        .expect("plugins[]")
        .iter()
        .map(|p| {
            let name = p["name"].as_str().expect("name").to_string();
            let source = p["source"].as_str().expect("source is a relative path");
            (name, root().join(source))
        })
        .collect()
}

#[test]
fn every_plugin_has_the_crate_version() {
    let plugins = marketplace_plugins();
    assert!(!plugins.is_empty());
    for (name, dir) in plugins {
        let manifest = json(&dir.join(".claude-plugin/plugin.json"));
        assert_eq!(manifest["name"], name.as_str(), "{}", dir.display());
        assert_eq!(
            manifest["version"],
            env!("CARGO_PKG_VERSION"),
            "{name}: plugin.json version must equal Cargo.toml (release.toml bumps it)"
        );
    }
}

/// One source of truth: a version in a marketplace entry would need its own bump.
#[test]
fn marketplace_entries_carry_no_version() {
    let market = json(&root().join(".claude-plugin/marketplace.json"));
    for p in market["plugins"].as_array().unwrap() {
        assert!(p.get("version").is_none(), "{}", p["name"]);
    }
}

/// Claude Code names a plugin server's tools `mcp__plugin_<plugin>_<server>__<tool>`;
/// the server's ToolSearch instruction uses this prefix (`REFLEX_MCP_TOOL_PREFIX`).
#[test]
fn mcp_plugin_passes_its_tool_prefix() {
    let manifest = json(&root().join("plugins/reflex-mcp/.claude-plugin/plugin.json"));
    let plugin = manifest["name"].as_str().unwrap();
    let servers = manifest["mcpServers"].as_object().expect("mcpServers");
    assert_eq!(servers.len(), 1);
    let (server, entry) = servers.iter().next().unwrap();
    assert_eq!(entry["command"], "rfx");
    assert_eq!(entry["args"], serde_json::json!(["mcp"]));
    assert_eq!(
        entry["env"]["REFLEX_MCP_TOOL_PREFIX"],
        format!("mcp__plugin_{plugin}_{server}__").as_str()
    );
}

/// The skill carries the routing sentences the MCP instructions carry
/// (`test_instructions_keep_the_routing_rules` in `src/mcp.rs`).
#[test]
fn skill_keeps_the_routing_rules() {
    let skill = std::fs::read_to_string(root().join("plugins/reflex/skills/reflex/SKILL.md"))
        .expect("SKILL.md");
    assert!(skill.starts_with("---\nname: reflex\ndescription: "));
    // A sentence may wrap: compare with the whitespace collapsed.
    let skill = skill.split_whitespace().collect::<Vec<_>>().join(" ");
    for needle in [
        "prefer `rfx` over Grep, Glob, ripgrep, or shell grep",
        "not natural language",
        "If `rfx` fails, retry it once; only fall back to Grep/Glob after the retry also fails.",
        "npm install -g reflex-search",
        // The MCP routing sentences, with the CLI commands (`list_locations`,
        // `find_references`): measured to matter for adoption and cost.
        "To list where something occurs, use `rfx locations X`",
        "use `rfx refs X`; it drops those lines exactly, so do not re-check",
    ] {
        assert!(skill.contains(needle), "lost: {needle}");
    }
}
