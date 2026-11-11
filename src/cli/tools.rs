//! `rfx refs` and `rfx locations`: the MCP tools `find_references` and
//! `list_locations` on the CLI (task 1.4 of `.context/AGENT_EDGE_PLAN.md`).
//!
//! Both run the MCP tool itself (`mcp::call_tool_data`), so `--json` prints the MCP
//! answer byte for byte: the same matching, the same comment and string filter, the
//! same freshness fields. The text form is grep-like (`path:line:text`), for agents
//! that read the output in a shell.

use anyhow::Result;
use serde_json::{Map, Value, json};

/// Filters `rfx refs` and `rfx locations` share; the names are `rfx query`'s.
#[derive(clap::Args, Debug)]
pub struct ToolFilters {
    /// Text to find. Whole identifiers by default, like `grep -w`
    pub pattern: Option<String>,

    /// Only these files or directories, like grep's path arguments (`src/`, `lib/a.rs`)
    #[arg(value_name = "PATH")]
    pub scope: Vec<String>,

    /// The pattern as a named flag, for patterns that start with `-`. Every
    /// positional argument is then a PATH.
    #[arg(long = "pattern", value_name = "PATTERN", allow_hyphen_values = true)]
    pub pattern_flag: Option<String>,

    /// Match substrings, not whole identifiers (like `grep -F`)
    #[arg(long)]
    pub contains: bool,

    /// Match letters regardless of case (like `rg -i`)
    #[arg(short = 'i', long)]
    pub ignore_case: bool,

    /// Only files of this language (rust, python, typescript, text, …)
    #[arg(short, long)]
    pub lang: Option<String>,

    /// Only paths that contain this substring
    #[arg(short, long)]
    pub file: Option<String>,

    /// Only paths matching this glob (gitignore rules; repeatable)
    #[arg(short, long)]
    pub glob: Vec<String>,

    /// Skip paths matching this glob (gitignore rules; repeatable)
    #[arg(short = 'x', long)]
    pub exclude: Vec<String>,

    /// Also search lock files (Cargo.lock, package-lock.json, …)
    #[arg(long)]
    pub include_locks: bool,

    /// Also search generated files (*.pb.go, *.min.js, …)
    #[arg(long)]
    pub include_generated: bool,

    /// Run a pattern too broad to run by default (under 3 characters)
    #[arg(long)]
    pub force: bool,

    /// Print the MCP tool's JSON answer
    #[arg(long)]
    pub json: bool,

    /// Pretty-print the JSON (with --json)
    #[arg(long)]
    pub pretty: bool,
}

impl ToolFilters {
    /// The tool arguments: only the flags that were given, so the tool's defaults
    /// apply to the rest.
    fn arguments(&self) -> Result<Map<String, Value>> {
        let (pattern, scope) = split_pattern(
            self.pattern.clone(),
            self.pattern_flag.clone(),
            self.scope.clone(),
        );
        let pattern = pattern
            .ok_or_else(|| anyhow::anyhow!("a pattern is required (or --pattern PATTERN)"))?;
        let glob = scope_globs(&scope, &self.glob)?;
        let mut args = Map::new();
        args.insert("pattern".into(), json!(pattern));
        for (key, on) in [
            ("contains", self.contains),
            ("ignore_case", self.ignore_case),
            ("include_locks", self.include_locks),
            ("include_generated", self.include_generated),
            ("force", self.force),
        ] {
            if on {
                args.insert(key.into(), json!(true));
            }
        }
        if let Some(lang) = &self.lang {
            args.insert("lang".into(), json!(lang));
        }
        if let Some(file) = &self.file {
            args.insert("file".into(), json!(file));
        }
        if !glob.is_empty() {
            args.insert("glob".into(), json!(glob));
        }
        if !self.exclude.is_empty() {
            args.insert("exclude".into(), json!(self.exclude));
        }
        Ok(args)
    }
}

/// The pattern and the PATH arguments. With `--pattern`, the first positional
/// argument is a PATH too (`rfx locations --pattern '-> T' src/`).
pub(crate) fn split_pattern(
    positional: Option<String>,
    flag: Option<String>,
    mut scope: Vec<String>,
) -> (Option<String>, Vec<String>) {
    match flag {
        Some(flag) => {
            if let Some(first) = positional {
                scope.insert(0, first);
            }
            (Some(flag), scope)
        }
        None => (positional, scope),
    }
}

/// PATH arguments as globs, the way grep reads them: a directory covers everything
/// under it, a file only itself, `.` (or the index root) the whole index. With
/// `--glob` too, a directory narrows each glob (`src/` + `*.rs` = `src/**/*.rs`).
/// Paths are relative to the index root (the CLI rebases them from a subdirectory).
pub(crate) fn scope_globs(scope: &[String], globs: &[String]) -> Result<Vec<String>> {
    use std::path::{Component, Path};
    if scope.is_empty() {
        return Ok(globs.to_vec());
    }
    let root = std::env::current_dir()?;
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    for raw in scope {
        let path = Path::new(raw);
        let rel = if path.is_absolute() {
            path.strip_prefix(&root)
                .map_err(|_| anyhow::anyhow!("{raw} is outside the index at {}", root.display()))?
        } else {
            path
        };
        let mut parts: Vec<String> = Vec::new();
        for c in rel.components() {
            match c {
                Component::Normal(s) => parts.push(s.to_string_lossy().into_owned()),
                Component::ParentDir if parts.pop().is_some() => {}
                Component::ParentDir => {
                    anyhow::bail!("{raw} is outside the index at {}", root.display())
                }
                _ => {}
            }
        }
        if parts.is_empty() {
            return Ok(globs.to_vec()); // the whole index
        }
        let rel = parts.join("/");
        if root.join(&rel).is_dir() {
            dirs.push(rel);
        } else if root.join(&rel).is_file() || rel.contains(['*', '?', '[']) {
            files.push(rel);
        } else {
            anyhow::bail!("{raw}: no such file or directory");
        }
    }
    let mut out = files;
    for dir in dirs {
        if globs.is_empty() {
            out.push(format!("{dir}/**"));
        }
        for g in globs {
            if g.trim_start_matches('/').contains('/') {
                anyhow::bail!(
                    "--glob {g} contains a '/': it cannot be combined with the path {dir}; \
                     put the directory in the glob instead"
                );
            }
            out.push(format!("{dir}/**/{}", g.trim_start_matches('/')));
        }
    }
    Ok(out)
}

/// Run one MCP tool for the CLI, with the CLI's automatic update.
fn call(name: &str, args: Map<String, Value>, no_update: bool) -> Result<Value> {
    let root = std::env::current_dir()?;
    let update = (!no_update).then(crate::auto_update::UpdateOptions::cli);
    crate::mcp::call_tool_data(name, Value::Object(args), &root, update)
}

fn print_json(data: &Value, pretty: bool) -> Result<()> {
    let text = if pretty {
        serde_json::to_string_pretty(data)?
    } else {
        serde_json::to_string(data)?
    };
    println!("{text}");
    Ok(())
}

/// The first line of a preview, as grep prints a line.
fn first_line(preview: &str) -> &str {
    preview.lines().next().unwrap_or("")
}

/// Freshness, warnings and the zero-result hint, on stderr: stdout stays the answer.
fn report_notes(data: &Value) {
    if data["can_trust_results"] == json!(false) {
        eprintln!(
            "Warning: the index may be stale (status: {}); run `rfx index` and search again",
            data["status"].as_str().unwrap_or("unknown")
        );
    }
    for w in data["warnings"].as_array().into_iter().flatten() {
        if let Some(w) = w.as_str() {
            eprintln!("Warning: {w}");
        }
    }
    if let Some(hint) = data["hint"].as_str() {
        eprintln!("Hint: {hint}");
    }
}

/// `rfx refs` flags beyond the shared filters.
pub(super) struct RefsOptions {
    pub kind: Option<String>,
    pub include_strings: bool,
    pub count: bool,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    /// `path:line` of one Rust function (`find_references` `at`).
    pub at: Option<String>,
}

/// `rfx refs`: the definition and every use in code (`find_references`).
pub(super) fn handle_refs(filters: ToolFilters, opts: RefsOptions, no_update: bool) -> Result<()> {
    let RefsOptions {
        kind,
        include_strings,
        count,
        limit,
        offset,
        at,
    } = opts;
    let mut args = filters.arguments()?;
    if let Some(at) = &at {
        args.insert("at".into(), json!(at));
    }
    if let Some(kind) = kind {
        args.insert("kind".into(), json!(kind));
    }
    if include_strings {
        args.insert("include_strings".into(), json!(true));
    }
    if count {
        args.insert("mode".into(), json!("count"));
    }
    if let Some(n) = limit {
        args.insert("limit".into(), json!(n));
    }
    if let Some(n) = offset {
        args.insert("offset".into(), json!(n));
    }
    let mut data = call("find_references", args, no_update)?;
    if filters.json {
        return print_json(&data, filters.pretty);
    }
    // The several-definitions hint names the MCP `at`; the text form says it with
    // the CLI flag, on stdout, above the uses (`refs_text`).
    let several = data["definitions"].as_array().is_some_and(|d| d.len() > 1);
    let at_hint = several
        && data["hint"]
            .as_str()
            .is_some_and(|h| h.contains("pass at:"));
    if at_hint {
        data.as_object_mut().map(|o| o.remove("hint"));
    }
    report_notes(&data);
    if count {
        println!("{}", data["count"]);
        return Ok(());
    }
    print!("{}", refs_text(&data, at_hint));
    if at.is_some() {
        let n = |k: &str| data[k].as_array().map_or(0, |a| a.len());
        eprintln!(
            "{} uses of this definition; {} possible (could not tell which `{}`); {} lines name other items (left out)",
            n("references"),
            n("possible"),
            data["definition"]["symbol"].as_str().unwrap_or(""),
            data["elsewhere"].as_u64().unwrap_or(0),
        );
        return Ok(());
    }
    // `pagination` counts lines before the comment and string filter (the space
    // `offset` indexes), so the next page starts after both.
    let shown = data["returned_count"].as_u64().unwrap_or(0);
    let filtered = data["filtered_out"].as_u64().unwrap_or(0);
    eprintln!("{shown} references in code; {filtered} lines only in comments or strings left out");
    if data["has_more"] == json!(true) {
        let next = data["pagination"]["offset"].as_u64().unwrap_or(0) + shown + filtered;
        eprintln!("More: --offset {next}");
    }
    Ok(())
}

/// `definition path:line kind name`, then one `path:line:text` per reference. With
/// several functions of that name: each one, and how to keep the uses of one.
fn refs_text(data: &Value, at_hint: bool) -> String {
    let mut out = String::new();
    let def = &data["definition"];
    let defs = data["definitions"]
        .as_array()
        .map_or(&[][..], |a| a.as_slice());
    let symbol = def["symbol"].as_str().unwrap_or("");
    if defs.len() > 1 {
        out.push_str(&format!(
            "{} definitions of {symbol}; the uses below are of the name:\n",
            defs.len()
        ));
        for d in defs {
            out.push_str(&format!(
                "definition {}:{} {} {symbol}\n",
                d["path"].as_str().unwrap_or(""),
                d["line"],
                d["kind"].as_str().unwrap_or(""),
            ));
        }
        if at_hint {
            out.push_str(&format!(
                "For the uses of one: rfx refs {symbol} --at PATH:LINE\n"
            ));
        }
    } else if def.is_object() {
        let line = if def["line"].is_null() {
            &def["span"]["start_line"]
        } else {
            &def["line"]
        };
        let owner = def["impl"]
            .as_str()
            .map(|t| format!(" in impl {t}"))
            .unwrap_or_default();
        out.push_str(&format!(
            "definition {}:{} {} {}{owner}\n",
            def["path"].as_str().unwrap_or(""),
            line,
            def["kind"].as_str().unwrap_or(""),
            def["symbol"].as_str().unwrap_or(""),
        ));
    }
    let row = |out: &mut String, r: &Value| {
        let caller = if r["import"] == json!(true) {
            "  [import]".to_string()
        } else {
            r["in"]
                .as_str()
                .map(|c| format!("  [in {c}]"))
                .unwrap_or_default()
        };
        out.push_str(&format!(
            "{}:{}:{}{caller}\n",
            r["path"].as_str().unwrap_or(""),
            r["line"],
            first_line(r["preview"].as_str().unwrap_or("")).trim_end(),
        ));
    };
    for r in data["references"].as_array().into_iter().flatten() {
        row(&mut out, r);
    }
    let possible = data["possible"]
        .as_array()
        .map_or(&[][..], |a| a.as_slice());
    if !possible.is_empty() {
        out.push_str("possible (the receiver's type is not written; check these):\n");
        for r in possible {
            row(&mut out, r);
        }
    }
    out
}

/// `rfx locations`: every matching line, by file (`list_locations`).
pub(super) fn handle_locations(filters: ToolFilters, preview: bool, no_update: bool) -> Result<()> {
    let mut args = filters.arguments()?;
    if preview {
        args.insert("preview".into(), json!(true));
    }
    let data = call("list_locations", args, no_update)?;
    if filters.json {
        return print_json(&data, filters.pretty);
    }
    report_notes(&data);
    print!("{}", locations_text(&data));
    let files = data["locations"].as_object().map_or(0, |m| m.len());
    eprintln!(
        "{} lines in {files} file{}",
        data["total_locations"].as_u64().unwrap_or(0),
        if files == 1 { "" } else { "s" }
    );
    Ok(())
}

/// Grouped (`path:4,128,193`), or `path:line:text` per line with `--preview`. The
/// flat `[{path, line}]` answer (`REFLEX_MCP_GROUPED=0`) prints `path:line`.
fn locations_text(data: &Value) -> String {
    let mut out = String::new();
    match &data["locations"] {
        Value::Object(files) => {
            for (path, lines) in files {
                let lines = lines.as_array().cloned().unwrap_or_default();
                if lines.iter().all(Value::is_number) {
                    let joined: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
                    out.push_str(&format!("{path}:{}\n", joined.join(",")));
                } else {
                    for pair in lines {
                        out.push_str(&format!(
                            "{path}:{}:{}\n",
                            pair[0],
                            pair[1].as_str().unwrap_or("")
                        ));
                    }
                }
            }
        }
        Value::Array(rows) => {
            for r in rows {
                out.push_str(&format!(
                    "{}:{}\n",
                    r["path"].as_str().unwrap_or(""),
                    r["line"]
                ));
            }
        }
        _ => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locations_text_is_grep_like() {
        let grouped = json!({"locations": {"src/a.rs": [4, 128], "src/b.rs": [7]}});
        assert_eq!(locations_text(&grouped), "src/a.rs:4,128\nsrc/b.rs:7\n");
        let preview = json!({"locations": {"src/a.rs": [[4, "fn a()"], [9, "a();"]]}});
        assert_eq!(
            locations_text(&preview),
            "src/a.rs:4:fn a()\nsrc/a.rs:9:a();\n"
        );
        let flat = json!({"locations": [{"path": "src/a.rs", "line": 4}]});
        assert_eq!(locations_text(&flat), "src/a.rs:4\n");
    }

    #[test]
    fn refs_text_names_the_definition_then_each_use() {
        let data = json!({
            "definition": {"path": "src/a.rs", "span": {"start_line": 3, "end_line": 9},
                           "kind": "Function", "symbol": "alpha", "preview": "fn alpha() {\n}"},
            "references": [{"path": "src/b.rs", "line": 12, "preview": "    alpha();"}]
        });
        assert_eq!(
            refs_text(&data, false),
            "definition src/a.rs:3 Function alpha\nsrc/b.rs:12:    alpha();\n"
        );
        let no_def = json!({"definition": null, "references": []});
        assert_eq!(refs_text(&no_def, false), "");
    }
}
