//! Command-line interfaces declared with clap's derive API.
//!
//! The command tree is read from the extracted API, never by running the program:
//! a `#[derive(Parser)]` struct is the root; `#[command(subcommand)]` fields name a
//! `#[derive(Subcommand)]` enum whose variants are commands; `#[command(flatten)]`
//! merges the fields of an `Args` struct; `#[arg(...)]` fields are options; fields
//! without `#[arg]` are positional. Doc comments are the help text, as in clap.

use crate::parsers::api::{ApiItem, ApiKind, DocComment};
use crate::pulse::extract::surface::{RustApi, RustItem};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct CliCommand {
    /// `["rfx", "pulse", "generate"]`
    pub path: Vec<String>,
    pub about: Option<String>,
    pub doc: Option<DocComment>,
    pub args: Vec<CliArg>,
    pub subcommands: Vec<CliCommand>,
    pub hidden: bool,
    pub file: String,
    pub line: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CliArg {
    pub field: String,
    pub long: Option<String>,
    pub short: Option<char>,
    pub positional: bool,
    pub value_name: Option<String>,
    /// Rust type as written (`Option<PathBuf>`).
    pub ty: String,
    pub default: Option<String>,
    pub flag: bool,
    pub multiple: bool,
    pub required: bool,
    pub hidden: bool,
    pub doc: Option<DocComment>,
}

impl CliArg {
    /// `-o, --output <OUTPUT>` or `<PATH>`.
    pub fn usage(&self) -> String {
        let value = self
            .value_name
            .clone()
            .unwrap_or_else(|| self.field.to_ascii_uppercase());
        if self.positional {
            let v = format!("<{value}>");
            return if self.multiple { format!("{v}...") } else { v };
        }
        let mut parts = Vec::new();
        if let Some(s) = self.short {
            parts.push(format!("-{s}"));
        }
        if let Some(l) = &self.long {
            parts.push(format!("--{l}"));
        }
        let mut u = parts.join(", ");
        if !self.flag {
            u.push_str(&format!(" <{value}>"));
        }
        u
    }
}

impl CliCommand {
    pub fn name(&self) -> &str {
        self.path.last().map(String::as_str).unwrap_or("")
    }

    /// `rfx pulse generate [OPTIONS] <PATH>`, with `<COMMAND>` if it has subcommands.
    pub fn usage(&self) -> String {
        let mut u = self.path.join(" ");
        if self.args.iter().any(|a| !a.positional && !a.hidden) {
            u.push_str(" [OPTIONS]");
        }
        for a in self.args.iter().filter(|a| a.positional && !a.hidden) {
            let s = a.usage();
            if a.required {
                u.push_str(&format!(" {s}"));
            } else {
                u.push_str(&format!(" [{s}]"));
            }
        }
        if self.subcommands.iter().any(|c| !c.hidden) {
            u.push_str(" <COMMAND>");
        }
        u
    }

    /// Every command in the tree, depth-first, this one first.
    pub fn walk(&self) -> Vec<&CliCommand> {
        let mut out = vec![self];
        for c in &self.subcommands {
            out.extend(c.walk());
        }
        out
    }
}

/// `foo_bar` / `FooBar` → `foo-bar` (clap's default rename).
pub fn kebab(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                out.push('-');
            }
            out.push(c.to_ascii_lowercase());
        } else if c == '_' {
            out.push('-');
        } else {
            out.push(c);
        }
    }
    out
}

/// Split the inside of `arg(...)` at top-level commas: `long, default_value = "a,b"`.
fn attr_args(attr: &str, name: &str) -> Option<Vec<(String, Option<String>)>> {
    let inner = attr
        .strip_prefix(name)?
        .trim()
        .strip_prefix('(')?
        .strip_suffix(')')?;
    let mut out = Vec::new();
    let (mut depth, mut in_str, mut cur) = (0i32, false, String::new());
    for c in inner.chars() {
        match c {
            '"' => {
                in_str = !in_str;
                cur.push(c);
            }
            '(' | '[' | '{' if !in_str => {
                depth += 1;
                cur.push(c);
            }
            ')' | ']' | '}' if !in_str => {
                depth -= 1;
                cur.push(c);
            }
            ',' if !in_str && depth == 0 => {
                out.push(std::mem::take(&mut cur));
            }
            _ => cur.push(c),
        }
    }
    out.push(cur);
    Some(
        out.into_iter()
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .map(|p| match p.split_once('=') {
                Some((k, v)) => (
                    k.trim().to_string(),
                    Some(v.trim().trim_matches('"').to_string()),
                ),
                None => (p, None),
            })
            .collect(),
    )
}

fn attr_kv(attrs: &[String], name: &str) -> HashMap<String, Option<String>> {
    attrs
        .iter()
        .filter_map(|a| attr_args(a, name))
        .flatten()
        .collect()
}

fn has_derive(item: &ApiItem, what: &str) -> bool {
    item.attrs.iter().any(|a| {
        a.starts_with("derive(")
            && a.trim_start_matches("derive(")
                .trim_end_matches(')')
                .split(',')
                .any(|d| d.trim().rsplit("::").next() == Some(what))
    })
}

/// `Option<Foo>` / `Box<Foo>` / `crate::x::Foo` → `Foo`.
fn inner_type(ty: &str) -> &str {
    let mut t = ty.trim();
    for wrapper in ["Option<", "Box<", "Vec<"] {
        if let Some(rest) = t.strip_prefix(wrapper) {
            t = rest.strip_suffix('>').unwrap_or(rest);
        }
    }
    let t = t.split('<').next().unwrap_or(t);
    t.rsplit("::").next().unwrap_or(t).trim()
}

fn field_type(field: &ApiItem) -> String {
    field
        .signature
        .split_once(':')
        .map(|(_, t)| t.trim().to_string())
        .unwrap_or_default()
}

struct Types<'a> {
    by_name: HashMap<&'a str, &'a RustItem>,
}

/// Find every clap command tree in the code. Usually one.
pub fn find_commands(api: &RustApi) -> Vec<CliCommand> {
    let mut all: Vec<&RustItem> = Vec::new();
    for k in &api.crates {
        for m in &k.modules {
            all.extend(m.items.iter());
        }
    }
    let types = Types {
        by_name: all
            .iter()
            .filter(|i| i.item.kind.is_type())
            .map(|i| (i.item.name.as_str(), *i))
            .collect(),
    };
    let mut roots: Vec<CliCommand> = Vec::new();
    for it in all
        .iter()
        .filter(|i| i.item.kind == ApiKind::Struct && has_derive(&i.item, "Parser"))
    {
        let cmd = attr_kv(&it.item.attrs, "command");
        let bin = api
            .crates
            .iter()
            .find(|k| !k.is_lib)
            .map(|k| k.name.replace('_', "-"));
        let name = cmd
            .get("name")
            .cloned()
            .flatten()
            .or(bin)
            .unwrap_or_else(|| kebab(&it.item.name));
        if roots.iter().any(|r| r.path[0] == name) {
            continue;
        }
        let mut root = CliCommand {
            path: vec![name],
            about: cmd.get("about").cloned().flatten(),
            doc: it.item.doc.clone(),
            args: Vec::new(),
            subcommands: Vec::new(),
            hidden: false,
            file: it.file.clone(),
            line: it.item.start_line,
        };
        if let Some(Some(long)) = cmd.get("long_about") {
            root.doc = Some(DocComment {
                markdown: long
                    .replace("\\\n", "")
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" "),
                summary: root.about.clone().unwrap_or_default(),
                ..DocComment::default()
            });
        }
        fill_from_fields(&mut root, &it.item.members, &types, 0);
        roots.push(root);
    }
    roots
}

fn fill_from_fields(cmd: &mut CliCommand, fields: &[ApiItem], types: &Types, depth: usize) {
    if depth > 16 {
        return;
    }
    for f in fields.iter().filter(|f| f.kind == ApiKind::Field) {
        let command = attr_kv(&f.attrs, "command");
        let ty = field_type(f);
        if command.contains_key("subcommand") {
            if let Some(e) = types.by_name.get(inner_type(&ty)) {
                cmd.subcommands
                    .extend(subcommands(&cmd.path, e, types, depth + 1));
            }
            continue;
        }
        if command.contains_key("flatten") {
            if let Some(s) = types.by_name.get(inner_type(&ty)) {
                fill_from_fields(cmd, &s.item.members, types, depth + 1);
            }
            continue;
        }
        cmd.args.push(arg(f, &ty));
    }
}

fn arg(f: &ApiItem, ty: &str) -> CliArg {
    let a = attr_kv(&f.attrs, "arg");
    let a2 = attr_kv(&f.attrs, "clap");
    let get = |k: &str| a.get(k).or_else(|| a2.get(k));
    let long = get("long").map(|v| v.clone().unwrap_or_else(|| kebab(&f.name)));
    let short = get("short").map(|v| {
        v.as_deref()
            .and_then(|s| s.trim_matches('\'').chars().next())
            .unwrap_or_else(|| f.name.chars().next().unwrap_or('?'))
    });
    let is_bool = ty == "bool";
    let action = get("action").cloned().flatten().unwrap_or_default();
    let flag = is_bool || action.ends_with("Count") || action.ends_with("SetTrue");
    let optional = ty.starts_with("Option<");
    let multiple = ty.starts_with("Vec<");
    let default = get("default_value")
        .or_else(|| get("default_value_t"))
        .cloned()
        .flatten();
    let positional = long.is_none() && short.is_none();
    CliArg {
        field: f.name.clone(),
        long,
        short,
        positional,
        value_name: get("value_name").cloned().flatten(),
        ty: ty.to_string(),
        default: default.clone(),
        flag,
        multiple,
        required: get("required").is_some_and(|v| v.as_deref() != Some("false"))
            || (positional && !optional && !multiple && default.is_none() && !is_bool),
        hidden: get("hide").is_some_and(|v| v.as_deref() != Some("false")),
        doc: f.doc.clone(),
    }
}

fn subcommands(parent: &[String], e: &RustItem, types: &Types, depth: usize) -> Vec<CliCommand> {
    if e.item.kind != ApiKind::Enum {
        return Vec::new();
    }
    let mut out = Vec::new();
    for v in e.item.members.iter().filter(|m| m.kind == ApiKind::Variant) {
        let cmd = attr_kv(&v.attrs, "command");
        let name = cmd
            .get("name")
            .cloned()
            .flatten()
            .unwrap_or_else(|| kebab(&v.name));
        let mut path = parent.to_vec();
        path.push(name);
        let mut c = CliCommand {
            path,
            about: cmd.get("about").cloned().flatten(),
            doc: v.doc.clone(),
            args: Vec::new(),
            subcommands: Vec::new(),
            hidden: cmd
                .get("hide")
                .is_some_and(|h| h.as_deref() != Some("false")),
            file: e.file.clone(),
            line: v.start_line,
        };
        if !v.members.is_empty() {
            fill_from_fields(&mut c, &v.members, types, depth + 1);
        } else if let Some(inner) = v
            .signature
            .split_once('(')
            .map(|(_, r)| r.trim_end_matches(')'))
            && let Some(t) = types.by_name.get(inner_type(inner))
        {
            // `Variant(Args)` or `Variant(NestedEnum)`.
            match t.item.kind {
                ApiKind::Enum => c.subcommands = subcommands(&c.path, t, types, depth + 1),
                _ => fill_from_fields(&mut c, &t.item.members, types, depth + 1),
            }
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pulse::extract::surface::{RustCrate, RustModule};
    use std::collections::BTreeMap;

    fn api(src: &str) -> RustApi {
        let file = crate::parsers::api::rust::extract(src).unwrap();
        let items = file
            .items
            .into_iter()
            .map(|item| RustItem {
                path: format!("app::{}", item.name),
                item,
                file: "src/cli.rs".into(),
                public: false,
                defined_at: None,
                impl_files: BTreeMap::new(),
            })
            .collect();
        RustApi {
            crates: vec![RustCrate {
                name: "app".into(),
                package: "app".into(),
                manifest: "Cargo.toml".into(),
                is_lib: false,
                root_file: "src/main.rs".into(),
                modules: vec![RustModule {
                    path: "app".into(),
                    name: "app".into(),
                    file: "src/cli.rs".into(),
                    public: false,
                    doc: None,
                    items,
                    children: vec![],
                    parent: None,
                }],
            }],
        }
    }

    const SRC: &str = r#"
/// The app.
#[derive(Parser, Debug)]
#[command(name = "app", about = "Does things")]
pub struct Cli {
    /// Be loud
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Build the index
    ///
    /// Longer text.
    Index {
        /// Where
        #[arg(value_name = "PATH", default_value = ".")]
        path: PathBuf,
        /// Rebuild
        #[arg(short, long)]
        force: bool,
        #[command(flatten)]
        common: Common,
    },
    /// Nested
    Pulse {
        #[command(subcommand)]
        command: PulseCmd,
    },
    #[command(hide = true)]
    Internal,
}

#[derive(Args)]
struct Common {
    /// Output file
    #[arg(short = 'o', long = "out-file", default_value = "a,b")]
    output: Option<String>,
    /// Names
    names: Vec<String>,
}

#[derive(Subcommand)]
enum PulseCmd {
    /// Generate it
    GenerateSite,
}
"#;

    #[test]
    fn command_tree() {
        let roots = find_commands(&api(SRC));
        assert_eq!(roots.len(), 1);
        let root = &roots[0];
        assert_eq!(root.path, vec!["app"]);
        assert_eq!(root.about.as_deref(), Some("Does things"));
        assert_eq!(root.args[0].usage(), "-v, --verbose");
        assert!(root.args[0].flag);
        assert_eq!(root.usage(), "app [OPTIONS] <COMMAND>");

        let names: Vec<String> = root.walk().iter().map(|c| c.path.join(" ")).collect();
        assert_eq!(
            names,
            vec![
                "app",
                "app index",
                "app pulse",
                "app pulse generate-site",
                "app internal"
            ]
        );
        assert!(root.subcommands[2].hidden);

        let index = &root.subcommands[0];
        assert_eq!(index.doc.as_ref().unwrap().summary, "Build the index");
        let usages: Vec<String> = index.args.iter().map(|a| a.usage()).collect();
        assert_eq!(
            usages,
            vec![
                "<PATH>",
                "-f, --force",
                "-o, --out-file <OUTPUT>",
                "<NAMES>..."
            ]
        );
        assert!(!index.args[0].required, "has a default");
        assert_eq!(index.args[0].default.as_deref(), Some("."));
        assert_eq!(index.args[2].default.as_deref(), Some("a,b"));
        assert_eq!(index.usage(), "app index [OPTIONS] [<PATH>] [<NAMES>...]");
    }

    #[test]
    fn helpers() {
        assert_eq!(kebab("GenerateSite"), "generate-site");
        assert_eq!(kebab("no_llm"), "no-llm");
        assert_eq!(inner_type("Option<crate::x::Foo>"), "Foo");
        let kv = attr_args(
            r#"arg(long, default_value = "a,b", num_args = 0..=1)"#,
            "arg",
        )
        .unwrap();
        assert_eq!(kv.len(), 3);
        assert_eq!(kv[1], ("default_value".into(), Some("a,b".into())));
    }
}
