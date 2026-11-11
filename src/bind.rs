//! `find_references` with `at` (`rfx refs X --at path:line`): which uses of a name
//! reach ONE definition, when the code has several items with that name.
//!
//! A name search answers "uses of this name". An agent asked "who calls the
//! `waker_ref` at `util/wake.rs:35`?" needs "uses of this definition": tokio has three
//! `waker_ref` functions. This module sorts each line that names `X` into one of
//! three verdicts, by Rust's name rules read from the syntax tree (tree-sitter):
//!
//! - `This`: the line uses the definition at `at` (a call, a value, an import).
//! - `Other`: it names another item: another definition, a local, a field, a module,
//!   a method of another type, an item of `std`.
//! - `Possible`: the rules cannot tell (a method call on a receiver whose type is not
//!   written down, a glob import). The agent reads these lines.
//!
//! The rules: `use` paths, `Self::` / `Type::` / `module::` qualifiers, the file's own
//! items, locals and parameters, and the receiver's type where the code writes it
//! (`self`, a typed parameter or `let`, a field of `Self`). No type inference, no
//! macro expansion, no trait resolution: a method call on a receiver of unknown type
//! binds to the target only when the target is the only method with that name.
//! Rust only. Measured on compiler answer keys: `.context/AGENT_EDGE_RESEARCH.md`.

use anyhow::{Result, bail};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tree_sitter::{Node, Parser, Tree};

/// What one line that names `X` refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Other,
    Possible,
    This,
}

/// A function (or trait method declaration) named `X`.
#[derive(Clone, Debug, PartialEq)]
pub struct Def {
    pub path: String,
    pub line: usize,
    /// The `impl` block's type, or the trait's name for a method declared in a trait.
    pub impl_type: Option<String>,
    /// In a trait, or in an `impl Trait for Type` block.
    pub in_trait: bool,
    pub has_self: bool,
    /// Not in an `impl` or a trait.
    pub free: bool,
    pub krate: String,
    pub module: Vec<String>,
    first_row: usize,
    last_row: usize,
}

/// One line and what it refers to.
#[derive(Clone, Debug, Serialize)]
pub struct BoundLine {
    pub path: String,
    pub line: usize,
    pub verdict: Verdict,
    /// The function the use is in.
    pub caller: Option<String>,
    /// A `use` line: it names the function but does not call it.
    pub import: bool,
}

pub struct Binding {
    pub target: Def,
    /// Every input line except the target's own definition line, in input order.
    pub lines: Vec<BoundLine>,
}

struct File {
    src: String,
    tree: Tree,
    krate: String,
    module: Vec<String>,
    /// The dependencies of the file's package (`[dependencies]` and the like).
    deps: HashSet<String>,
}

/// Where a type comes from, judged by the file's imports and its package's
/// dependencies.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Origin {
    Std,
    /// A crate that is a dependency and not in this workspace.
    External,
    /// This workspace, or not known.
    Local,
}

/// Parse `path:line`.
pub fn parse_at(at: &str) -> Result<(String, usize)> {
    let Some((path, line)) = at.rsplit_once(':') else {
        bail!("at: expected path:line, got {at:?}");
    };
    let line: usize = line
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("at: expected path:line, got {at:?}"))?;
    let path = path.trim().trim_start_matches("./").to_string();
    Ok((path, line))
}

/// Sort `lines` (code lines that name `name`, relative paths) by what they refer to,
/// for the function `name` defined at `at` (`path:line`, the line of its `fn`).
pub fn bind(root: &Path, at: &str, name: &str, lines: &[(String, usize)]) -> Result<Binding> {
    let (at_path, at_line) = parse_at(at)?;
    if !at_path.ends_with(".rs") {
        bail!("at: only Rust definitions can be bound for now ({at_path})");
    }
    let mut files: HashMap<String, File> = HashMap::new();
    let mut order: Vec<String> = vec![at_path.clone()];
    for (p, _) in lines {
        if !order.contains(p) {
            order.push(p.clone());
        }
    }
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_rust::LANGUAGE.into())?;
    let mut deps_of: HashMap<PathBuf, HashSet<String>> = HashMap::new();
    for p in &order {
        if !p.ends_with(".rs") {
            continue;
        }
        let Ok(src) = std::fs::read_to_string(root.join(p)) else {
            if *p == at_path {
                bail!("at: cannot read {p}");
            }
            continue;
        };
        let Some((src, tree)) = parse_flat(&mut parser, src) else {
            continue;
        };
        let (krate, module, package) = crate_module(root, p);
        let deps = match package {
            Some(dir) => deps_of
                .entry(dir.clone())
                .or_insert_with(|| package_deps(&dir.join("Cargo.toml")))
                .clone(),
            None => HashSet::new(),
        };
        files.insert(
            p.clone(),
            File {
                src,
                tree,
                krate,
                module,
                deps,
            },
        );
    }

    let mut defs = Vec::new();
    for p in &order {
        if let Some(f) = files.get(p) {
            collect_defs(p, f, name, &mut defs);
        }
    }
    let target =
        defs.iter()
            .filter(|d| d.path == at_path && d.line == at_line)
            .chain(defs.iter().filter(|d| {
                d.path == at_path && d.first_row < at_line && at_line <= d.last_row + 1
            }))
            .next()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("at: no function `{name}` is defined at {at}"))?;

    let cx = Cx {
        name,
        target: &target,
        defs: &defs,
        crates: workspace_crates(root),
    };
    let mut out = Vec::new();
    for (p, line) in lines {
        if *p == target.path && *line == target.line {
            continue;
        }
        let (verdict, caller, import) = match files.get(p) {
            Some(f) => cx.line(p, f, *line),
            None => (Verdict::Other, None, false), // not Rust: not a use of a Rust function
        };
        out.push(BoundLine {
            path: p.clone(),
            line: *line,
            verdict,
            caller,
            import,
        });
    }
    Ok(Binding { target, lines: out })
}

/// What plain `rfx refs X` knows of one line, from the Rust syntax tree.
#[derive(Clone, Debug, Default)]
pub struct LineFn {
    /// The function the line's first `name` is in (`[in f]`).
    pub caller: Option<String>,
    /// The line defines a function named `name`: `Some("Method")` in an `impl` or a
    /// trait, else `Some("Function")`. It sees definitions inside item-level macros
    /// (`cfg_rt! { … }`), which the symbol cache misses.
    pub defines: Option<&'static str>,
}

/// [`LineFn`] for each line (default: not Rust, or the file cannot be read).
pub fn enclosing_fns(root: &Path, name: &str, lines: &[(String, usize)]) -> Vec<LineFn> {
    let mut parser = Parser::new();
    if parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .is_err()
    {
        return vec![LineFn::default(); lines.len()];
    }
    let mut parsed: HashMap<String, Option<(String, Tree)>> = HashMap::new();
    lines
        .iter()
        .map(|(p, line)| {
            if !p.ends_with(".rs") {
                return LineFn::default();
            }
            let entry = parsed.entry(p.clone()).or_insert_with(|| {
                let src = std::fs::read_to_string(root.join(p)).ok()?;
                parse_flat(&mut parser, src)
            });
            let Some((src, tree)) = entry.as_ref() else {
                return LineFn::default();
            };
            let start: usize = src.split('\n').take(line - 1).map(|l| l.len() + 1).sum();
            let Some(row) = src.split('\n').nth(line - 1) else {
                return LineFn::default();
            };
            let col = row.find(name).unwrap_or(row.len() - row.trim_start().len());
            let a = start + col;
            let Some(n) = tree.root_node().descendant_for_byte_range(a, a) else {
                return LineFn::default();
            };
            let mut out = LineFn::default();
            let mut cur = Some(n);
            while let Some(c) = cur {
                if matches!(c.kind(), "function_item" | "function_signature_item")
                    && let Some(nm) = c.child_by_field_name("name")
                {
                    if nm.start_byte() == a && &src[nm.start_byte()..nm.end_byte()] == name {
                        let method = ancestor(c, &["impl_item", "trait_item"]).is_some();
                        out.defines = Some(if method { "Method" } else { "Function" });
                    } else if c.kind() == "function_item" {
                        out.caller = Some(src[nm.start_byte()..nm.end_byte()].to_string());
                        break;
                    }
                }
                if let Some(m) = macro_name(c, src) {
                    out.caller = Some(m);
                    break;
                }
                cur = c.parent();
            }
            out
        })
        .collect()
}

/// Parse, with item-level brace macros opened up: `cfg_rt! { fn a() {} }` holds
/// tokens, not items, for tree-sitter, and tokio wraps most of its code in such
/// blocks. `name! {` and its `}` become spaces (same byte offsets and lines), so the
/// items sit in the enclosing module or impl, where the cfg macro puts them.
fn parse_flat(parser: &mut Parser, mut src: String) -> Option<(String, Tree)> {
    for _ in 0..8 {
        let tree = parser.parse(&src, None)?;
        let mut spans = Vec::new();
        let mut stack = vec![tree.root_node()];
        while let Some(n) = stack.pop() {
            let mut c = n.walk();
            for ch in n.named_children(&mut c) {
                if matches!(
                    ch.kind(),
                    "source_file" | "declaration_list" | "mod_item" | "impl_item"
                ) {
                    stack.push(ch);
                } else if ch.kind() == "macro_invocation"
                    && let Some(body) = ch.named_children(&mut ch.walk()).last()
                    && body.kind() == "token_tree"
                    && src.as_bytes().get(body.start_byte()) == Some(&b'{')
                    && src.as_bytes().get(body.end_byte() - 1) == Some(&b'}')
                    && !text(ch, &src).starts_with("macro_rules")
                {
                    spans.push((ch.start_byte(), body.start_byte() + 1, body.end_byte() - 1));
                }
            }
        }
        if spans.is_empty() {
            return Some((src, tree));
        }
        let mut bytes = src.into_bytes();
        for (a, b, close) in spans {
            for x in &mut bytes[a..b] {
                if *x != b'\n' {
                    *x = b' ';
                }
            }
            bytes[close] = b' ';
        }
        src = String::from_utf8(bytes).ok()?;
    }
    let tree = parser.parse(&src, None)?;
    Some((src, tree))
}

// ---------------------------------------------------------------------------
// Crates and modules
// ---------------------------------------------------------------------------

/// The crate name (`grep_printer`) and the module path (`["util", "wake"]`) of a file,
/// from its place under the crate root (the nearest directory with `lib.rs` or
/// `main.rs`, at or under the package's `Cargo.toml`).
fn crate_module(root: &Path, rel: &str) -> (String, Vec<String>, Option<PathBuf>) {
    let file = root.join(rel);
    let mut package = None;
    let mut dir = file.parent();
    while let Some(d) = dir {
        if d.join("Cargo.toml").is_file() {
            package = Some(d.to_path_buf());
            break;
        }
        if d == root {
            break;
        }
        dir = d.parent();
    }
    let krate = package
        .as_ref()
        .and_then(|p| crate_name(&p.join("Cargo.toml")))
        .unwrap_or_default();
    let mut crate_root = None;
    let mut dir = file.parent();
    while let Some(d) = dir {
        if d.join("lib.rs").is_file() || d.join("main.rs").is_file() {
            crate_root = Some(d.to_path_buf());
            break;
        }
        if package.as_deref() == Some(d) || d == root {
            break;
        }
        dir = d.parent();
    }
    let base = crate_root
        .or_else(|| package.as_ref().map(|p| p.join("src")))
        .unwrap_or_else(|| root.to_path_buf());
    let mut module: Vec<String> = file
        .strip_prefix(&base)
        .map(|r| {
            r.components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    if let Some(last) = module.pop() {
        let stem = last.trim_end_matches(".rs").to_string();
        let at_root = module.is_empty() && (stem == "lib" || stem == "main");
        if stem != "mod" && !at_root {
            module.push(stem);
        }
    }
    (krate, module, package)
}

/// The crate names of every package under `root` (`Cargo.toml` files, 4 levels deep;
/// `target/` and dot-directories skipped).
fn workspace_crates(root: &Path) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut stack = vec![(root.to_path_buf(), 0)];
    while let Some((dir, depth)) = stack.pop() {
        if let Some(n) = crate_name(&dir.join("Cargo.toml")) {
            out.insert(n);
        }
        if depth == 4 {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || matches!(name.as_str(), "target" | "node_modules") {
                continue;
            }
            if e.file_type().is_ok_and(|t| t.is_dir()) {
                stack.push((e.path(), depth + 1));
            }
        }
    }
    out
}

/// The dependency names in a `Cargo.toml` (normal, dev, build, per-target).
fn package_deps(cargo_toml: &Path) -> HashSet<String> {
    let mut out = HashSet::new();
    let Some(v) = std::fs::read_to_string(cargo_toml)
        .ok()
        .and_then(|t| t.parse::<toml::Value>().ok())
    else {
        return out;
    };
    let mut tables = vec![&v];
    if let Some(t) = v.get("target").and_then(|t| t.as_table()) {
        tables.extend(t.values());
    }
    for t in tables {
        for key in ["dependencies", "dev-dependencies", "build-dependencies"] {
            if let Some(d) = t.get(key).and_then(|d| d.as_table()) {
                out.extend(d.keys().map(|k| k.replace('-', "_")));
            }
        }
    }
    out
}

fn crate_name(cargo_toml: &Path) -> Option<String> {
    let v: toml::Value = std::fs::read_to_string(cargo_toml).ok()?.parse().ok()?;
    let name = v
        .get("lib")
        .and_then(|l| l.get("name"))
        .or_else(|| v.get("package").and_then(|p| p.get("name")))?
        .as_str()?;
    Some(name.replace('-', "_"))
}

// ---------------------------------------------------------------------------
// Syntax helpers
// ---------------------------------------------------------------------------

fn text<'a>(n: Node, src: &'a str) -> &'a str {
    n.utf8_text(src.as_bytes()).unwrap_or("")
}

fn ancestor<'t>(n: Node<'t>, kinds: &[&str]) -> Option<Node<'t>> {
    let mut cur = n.parent();
    while let Some(c) = cur {
        if kinds.contains(&c.kind()) {
            return Some(c);
        }
        cur = c.parent();
    }
    None
}

/// `Arc<Mutex<Foo<T>>>` → `Foo`; `&mut self::a::Bar` → `Bar`.
fn type_name(t: &str) -> String {
    let mut t = t.trim();
    loop {
        let before = t;
        for p in ["&", "mut ", "dyn ", "impl ", "const ", "*"] {
            t = t.strip_prefix(p).unwrap_or(t).trim_start();
        }
        if t.starts_with('\'') {
            t = t.split_once(' ').map_or("", |(_, r)| r).trim_start();
        }
        if t == before {
            break;
        }
    }
    let (base, generics) = match t.find('<') {
        Some(i) => (&t[..i], Some(&t[i + 1..t.rfind('>').unwrap_or(t.len())])),
        None => (t, None),
    };
    let base = base.trim();
    if let Some(rest) = ["std::", "core::", "alloc::"]
        .iter()
        .find_map(|p| base.strip_prefix(p))
    {
        // `std::time::Instant` is not this crate's `Instant`.
        return format!("std::{}", rest.rsplit("::").next().unwrap_or(rest));
    }
    let base = base.rsplit("::").next().unwrap_or(base).trim();
    const WRAPPERS: [&str; 13] = [
        "Arc",
        "Rc",
        "Box",
        "Pin",
        "RefCell",
        "Cell",
        "Mutex",
        "RwLock",
        "MutexGuard",
        "Ref",
        "RefMut",
        "ManuallyDrop",
        "UnsafeCell",
    ];
    if let (true, Some(g)) = (WRAPPERS.contains(&base), generics) {
        // The first top-level generic argument.
        let mut depth = 0;
        let mut end = g.len();
        for (i, c) in g.char_indices() {
            match c {
                '<' | '(' | '[' => depth += 1,
                '>' | ')' | ']' => depth -= 1,
                ',' if depth == 0 => {
                    end = i;
                    break;
                }
                _ => {}
            }
        }
        return type_name(&g[..end]);
    }
    base.to_string()
}

/// Types whose methods are never this crate's functions.
const STD_TYPES: [&str; 38] = [
    "u8",
    "u16",
    "u32",
    "u64",
    "u128",
    "usize",
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
    "isize",
    "f32",
    "f64",
    "bool",
    "char",
    "str",
    "String",
    "Vec",
    "Option",
    "Result",
    "Duration",
    "HashMap",
    "HashSet",
    "BTreeMap",
    "BTreeSet",
    "VecDeque",
    "Path",
    "PathBuf",
    "OsStr",
    "OsString",
    "AtomicUsize",
    "AtomicU64",
    "AtomicU32",
    "AtomicBool",
    "AtomicIsize",
    "AtomicI64",
    "AtomicPtr",
];

fn collect_defs(path: &str, f: &File, name: &str, out: &mut Vec<Def>) {
    let mut stack = vec![f.tree.root_node()];
    while let Some(n) = stack.pop() {
        let mut c = n.walk();
        for ch in n.named_children(&mut c) {
            stack.push(ch);
        }
        if !matches!(n.kind(), "function_item" | "function_signature_item") {
            continue;
        }
        let Some(nm) = n.child_by_field_name("name") else {
            continue;
        };
        if text(nm, &f.src) != name {
            continue;
        }
        let has_self = n.child_by_field_name("parameters").is_some_and(|ps| {
            let mut c = ps.walk();
            // `&self`, or a typed `self: Arc<Self>` (a `parameter` whose pattern is `self`).
            ps.named_children(&mut c).any(|p| {
                p.kind() == "self_parameter"
                    || (p.kind() == "parameter"
                        && p.child_by_field_name("pattern")
                            .is_some_and(|x| x.kind() == "self"))
            })
        });
        let owner = ancestor(n, &["impl_item", "trait_item", "function_item"]);
        let (impl_type, in_trait) = match owner {
            Some(o) if o.kind() == "impl_item" => (
                o.child_by_field_name("type")
                    .map(|t| type_name(text(t, &f.src))),
                o.child_by_field_name("trait").is_some(),
            ),
            Some(o) if o.kind() == "trait_item" => (
                o.child_by_field_name("name")
                    .map(|t| text(t, &f.src).to_string()),
                true,
            ),
            _ => (None, false),
        };
        out.push(Def {
            path: path.to_string(),
            line: nm.start_position().row + 1,
            free: impl_type.is_none(),
            impl_type,
            in_trait,
            has_self,
            krate: f.krate.clone(),
            module: module_of(f, n),
            first_row: n.start_position().row,
            last_row: n.end_position().row,
        });
    }
    out.sort_by_key(|d| (d.path.clone(), d.line));
    out.dedup();
}

/// The module a node is in: the file's module plus inline `mod name { }` blocks.
fn module_of(f: &File, n: Node) -> Vec<String> {
    let mut inline = Vec::new();
    let mut cur = n.parent();
    while let Some(c) = cur {
        if c.kind() == "mod_item"
            && let Some(nm) = c.child_by_field_name("name")
        {
            inline.push(text(nm, &f.src).to_string());
        }
        cur = c.parent();
    }
    inline.reverse();
    let mut m = f.module.clone();
    m.extend(inline);
    m
}

fn caller(f: &File, n: Node) -> Option<String> {
    let mut cur = n.parent();
    while let Some(c) = cur {
        if let Some(m) = macro_name(c, &f.src) {
            return Some(m);
        }
        if c.kind() == "function_item"
            && let Some(nm) = c.child_by_field_name("name")
            && nm != n
        {
            return Some(text(nm, &f.src).to_string());
        }
        cur = c.parent();
    }
    None
}

/// `macro_rules! name { … }`: `macro_rules! name`. A use in a macro body runs where
/// the macro is invoked, so the caller is not a function here.
fn macro_name(n: Node, src: &str) -> Option<String> {
    if n.kind() != "macro_definition" {
        return None;
    }
    let nm = n.child_by_field_name("name")?;
    Some(format!(
        "macro_rules! {}",
        &src[nm.start_byte()..nm.end_byte()]
    ))
}

fn enclosing_impl_type(f: &File, n: Node) -> Option<String> {
    let i = ancestor(n, &["impl_item", "trait_item"])?;
    if i.kind() == "trait_item" {
        return i
            .child_by_field_name("name")
            .map(|t| text(t, &f.src).to_string());
    }
    i.child_by_field_name("type")
        .map(|t| type_name(text(t, &f.src)))
}

fn is_upper(s: &str) -> bool {
    s.chars().next().is_some_and(|c| c.is_ascii_uppercase())
}

/// `a::<T>::b` → `["a", "b"]`.
fn path_segments(p: &str) -> Vec<String> {
    let mut out = String::new();
    let mut depth = 0;
    for c in p.chars() {
        match c {
            '<' => depth += 1,
            '>' => depth -= 1,
            _ if depth == 0 && !c.is_whitespace() => out.push(c),
            _ => {}
        }
    }
    out.split("::")
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// The rules
// ---------------------------------------------------------------------------

struct Cx<'a> {
    name: &'a str,
    target: &'a Def,
    defs: &'a [Def],
    /// Crate names of this workspace's packages.
    crates: HashSet<String>,
}

impl Cx<'_> {
    /// The verdict of one line: the strongest of its occurrences of the name.
    fn line(&self, path: &str, f: &File, line: usize) -> (Verdict, Option<String>, bool) {
        let Some(row) = f.src.split('\n').nth(line - 1) else {
            return (Verdict::Other, None, false);
        };
        let start: usize = f.src.split('\n').take(line - 1).map(|l| l.len() + 1).sum();
        let mut best: Option<(Verdict, Option<String>, bool)> = None;
        for (col, _) in row.match_indices(self.name) {
            let a = start + col;
            let b = a + self.name.len();
            let word = |i: usize| {
                f.src.as_bytes()[i].is_ascii_alphanumeric() || f.src.as_bytes()[i] == b'_'
            };
            if (a > 0 && word(a - 1)) || (b < f.src.len() && word(b)) {
                continue;
            }
            let Some(n) = f.tree.root_node().descendant_for_byte_range(a, b) else {
                continue;
            };
            if text(n, &f.src) != self.name
                || ancestor(
                    n,
                    &[
                        "line_comment",
                        "block_comment",
                        "string_literal",
                        "raw_string_literal",
                    ],
                )
                .is_some()
            {
                continue;
            }
            let v = self.occurrence(path, f, n);
            if best.as_ref().is_none_or(|(b, _, _)| v > *b) {
                let import = ancestor(n, &["use_declaration"]).is_some();
                best = Some((v, caller(f, n), import));
            }
        }
        best.unwrap_or((Verdict::Other, None, false))
    }

    fn occurrence(&self, path: &str, f: &File, n: Node) -> Verdict {
        let Some(p) = n.parent() else {
            return Verdict::Other;
        };
        if matches!(p.kind(), "function_item" | "function_signature_item")
            && p.child_by_field_name("name") == Some(n)
        {
            return Verdict::Other; // another definition (the target's own line is skipped)
        }
        if ancestor(n, &["use_declaration"]).is_some() {
            return self.import(f, n);
        }
        if n.kind() == "field_identifier" {
            if p.kind() == "field_expression" && is_called(p) {
                return match p.child_by_field_name("value") {
                    Some(r) => self.method(f, r, n),
                    None => Verdict::Possible,
                };
            }
            return Verdict::Other; // a field
        }
        match p.kind() {
            "scoped_identifier" => {
                if p.child_by_field_name("name") != Some(n) {
                    return Verdict::Other; // a module or type segment
                }
                let q = p
                    .child_by_field_name("path")
                    .map_or("", |q| text(q, &f.src));
                self.qualified(f, n, &path_segments(q))
            }
            "macro_invocation" | "mod_item" | "macro_definition" => Verdict::Other,
            "token_tree" => self.in_macro(path, f, n),
            _ if n.kind() != "identifier" => Verdict::Other,
            _ if self.binds_local(f, n) => Verdict::Other,
            _ => self.plain(path, f, n),
        }
    }

    /// A pattern that binds the name, or a use of a local or parameter of that name.
    fn binds_local(&self, f: &File, n: Node) -> bool {
        if ancestor(n, &["closure_parameters", "parameters"]).is_some() {
            return true;
        }
        if let Some(l) = ancestor(n, &["let_declaration"])
            && l.child_by_field_name("pattern")
                .is_some_and(|p| p.start_byte() <= n.start_byte() && n.end_byte() <= p.end_byte())
        {
            return true;
        }
        if n.parent().is_some_and(|p| {
            matches!(
                p.kind(),
                "shorthand_field_initializer" | "match_pattern" | "tuple_pattern"
            )
        }) {
            return true;
        }
        self.is_local(f, n)
    }

    /// A `let` or parameter of this name before `n` in its function.
    fn is_local(&self, f: &File, n: Node) -> bool {
        let Some(func) = ancestor(n, &["function_item", "closure_expression"]) else {
            return false;
        };
        let name = self.name;
        let mut stack = vec![func];
        while let Some(c) = stack.pop() {
            if c.start_byte() >= n.start_byte() {
                continue;
            }
            match c.kind() {
                "let_declaration" if c.end_byte() <= n.start_byte() => {
                    if let Some(p) = c.child_by_field_name("pattern")
                        && pattern_binds(p, &f.src, name)
                    {
                        return true;
                    }
                }
                "parameter" | "closure_parameters" => {
                    let pat = c.child_by_field_name("pattern").unwrap_or(c);
                    if pattern_binds(pat, &f.src, name) {
                        return true;
                    }
                }
                _ => {}
            }
            let mut w = c.walk();
            for ch in c.named_children(&mut w) {
                if ch.kind() != "function_item" {
                    stack.push(ch);
                }
            }
        }
        false
    }

    /// The type written for a local or parameter `var` before `n` (`x: &Foo`,
    /// `let x = Foo::new()`); `""` for a `let` with no written type.
    fn local_type_of(&self, f: &File, n: Node, var: &str) -> Option<String> {
        // `for x in self.items.iter()`: the element type of the field.
        let mut cur = n.parent();
        while let Some(c) = cur {
            if c.kind() == "function_item" {
                break;
            }
            if c.kind() == "for_expression"
                && c.child_by_field_name("pattern")
                    .is_some_and(|p| pattern_binds(p, &f.src, var))
                && c.child_by_field_name("body")
                    .is_some_and(|b| b.start_byte() <= n.start_byte())
            {
                return Some(
                    c.child_by_field_name("value")
                        .and_then(|v| self.iterated_type(f, v, n))
                        .unwrap_or_default(),
                );
            }
            cur = c.parent();
        }
        let func = ancestor(n, &["function_item"])?;
        let mut found = None;
        if let Some(ps) = func.child_by_field_name("parameters") {
            let mut c = ps.walk();
            for p in ps.named_children(&mut c) {
                if p.kind() == "parameter"
                    && p.child_by_field_name("pattern")
                        .is_some_and(|pat| pattern_binds(pat, &f.src, var))
                {
                    found = p
                        .child_by_field_name("type")
                        .map(|t| type_name(text(t, &f.src)));
                }
            }
        }
        let body = func.child_by_field_name("body")?;
        let mut stack = vec![body];
        let mut lets = Vec::new();
        while let Some(c) = stack.pop() {
            if c.start_byte() >= n.start_byte() {
                continue;
            }
            if c.kind() == "let_declaration"
                && c.end_byte() <= n.start_byte()
                && c.child_by_field_name("pattern")
                    .is_some_and(|p| pattern_binds(p, &f.src, var))
            {
                lets.push(c);
            }
            let mut w = c.walk();
            for ch in c.named_children(&mut w) {
                if !matches!(ch.kind(), "function_item" | "closure_expression") {
                    stack.push(ch);
                }
            }
        }
        if let Some(l) = lets.into_iter().max_by_key(|l| l.start_byte()) {
            found = l
                .child_by_field_name("type")
                .map(|t| type_name(text(t, &f.src)))
                .or_else(|| {
                    l.child_by_field_name("value")
                        .and_then(|v| constructed_type(f, v))
                });
            if found.is_none() {
                return Some(String::new()); // a local, type not written
            }
        }
        found
    }

    /// `self.items`, `self.items.iter()`, `self.items.iter_mut().enumerate()`, `&self.items`:
    /// the element type of the field `items`.
    fn iterated_type(&self, f: &File, v: Node, at: Node) -> Option<String> {
        let mut v = v;
        loop {
            match v.kind() {
                "reference_expression" | "parenthesized_expression" => {
                    v = v.named_children(&mut v.walk()).last()?;
                }
                // `self.levels[a..]`: a slice of the field, the same elements.
                "index_expression" => v = v.named_child(0)?,
                "call_expression" => {
                    let func = v.child_by_field_name("function")?;
                    if func.kind() != "field_expression" {
                        return None;
                    }
                    let m = text(func.child_by_field_name("field")?, &f.src);
                    if !matches!(
                        m,
                        "iter" | "iter_mut" | "into_iter" | "enumerate" | "rev" | "skip" | "take"
                    ) {
                        return None;
                    }
                    v = func.child_by_field_name("value")?;
                }
                "field_expression" => {
                    if v.child_by_field_name("value")
                        .is_none_or(|x| x.kind() != "self")
                    {
                        return None;
                    }
                    let field = text(v.child_by_field_name("field")?, &f.src);
                    let owner = enclosing_impl_type(f, at)?;
                    return element_type(&field_type_text(f, &owner, field)?);
                }
                _ => return None,
            }
        }
    }

    /// `self`, a typed local, `self.field` (the field's declared type), `Foo::new()`.
    fn receiver_type(&self, f: &File, r: Node, at: Node) -> Option<String> {
        match r.kind() {
            "self" => enclosing_impl_type(f, at),
            "identifier" => self
                .local_type_of(f, at, text(r, &f.src))
                .filter(|t| !t.is_empty()),
            "field_expression" => {
                let v = r.child_by_field_name("value")?;
                let field = text(r.child_by_field_name("field")?, &f.src);
                if v.kind() != "self" {
                    return None;
                }
                let owner = enclosing_impl_type(f, at)?;
                field_type(f, &owner, field)
            }
            "parenthesized_expression" | "reference_expression" | "unary_expression" => {
                let mut c = r.walk();
                let inner = r.named_children(&mut c).last()?;
                self.receiver_type(f, inner, at)
            }
            "call_expression" => {
                // `b.encoding(..).bom_sniffing(..)` on an external builder `b`: the
                // chain stays external. Any other chain's type is not known.
                let func = r.child_by_field_name("function")?;
                if func.kind() != "field_expression" {
                    return constructed_type(f, r);
                }
                let mut root = func.child_by_field_name("value")?;
                loop {
                    match root.kind() {
                        "call_expression" => {
                            let g = root.child_by_field_name("function")?;
                            if g.kind() != "field_expression" {
                                break;
                            }
                            root = g.child_by_field_name("value")?;
                        }
                        "try_expression" | "await_expression" => {
                            root = root.named_child(0)?;
                        }
                        _ => break,
                    }
                }
                let t = self.receiver_type(f, root, at)?;
                let t = self.resolve(f, &t);
                t.starts_with("ext::").then_some(t)
            }
            "index_expression" => {
                // `self.levels[i]`: the element type of the field's declared type.
                let v = r.named_child(0)?;
                if v.kind() != "field_expression"
                    || v.child_by_field_name("value")
                        .is_none_or(|x| x.kind() != "self")
                {
                    return None;
                }
                let field = text(v.child_by_field_name("field")?, &f.src);
                let owner = enclosing_impl_type(f, at)?;
                element_type(&field_type_text(f, &owner, field)?)
            }
            _ => None,
        }
    }

    /// A type path as `of_type` reads it: `std::T` for a standard type, `ext::T` for
    /// another crate's type, else `T`.
    fn resolve(&self, f: &File, t: &str) -> String {
        if t.starts_with("std::") || t.starts_with("ext::") {
            return t.to_string();
        }
        let segs: Vec<&str> = t.split("::").collect();
        let last = segs.last().copied().unwrap_or(t);
        let origin = if segs.len() > 1 {
            self.origin(f, segs[0])
        } else {
            match imported_root(f, last) {
                Some(r) => self.origin_of_root(f, &r),
                None => Origin::Local,
            }
        };
        match origin {
            Origin::Std => format!("std::{last}"),
            Origin::External => format!("ext::{last}"),
            Origin::Local => last.to_string(),
        }
    }

    /// The origin of a path's first segment: a crate, or a name the file imports.
    fn origin(&self, f: &File, seg: &str) -> Origin {
        match self.origin_of_root(f, seg) {
            Origin::Local => match imported_root(f, seg) {
                Some(r) if r != seg => self.origin_of_root(f, &r),
                _ => Origin::Local,
            },
            o => o,
        }
    }

    /// The origin of a path that starts with `seg` (`std`, `crate`, a crate name).
    fn origin_of_root(&self, f: &File, seg: &str) -> Origin {
        match seg {
            "std" | "core" | "alloc" => Origin::Std,
            "crate" | "self" | "super" | "Self" => Origin::Local,
            _ if seg == f.krate || self.crates.contains(seg) => Origin::Local,
            _ if f.deps.contains(seg) => Origin::External,
            _ => Origin::Local,
        }
    }

    fn method(&self, f: &File, receiver: Node, n: Node) -> Verdict {
        if !self.target.has_self {
            return Verdict::Other; // `.x()` cannot call a free or associated fn
        }
        match self.receiver_type(f, receiver, n) {
            Some(t) => self.of_type(&self.resolve(f, &t)),
            None => self.unknown_receiver(),
        }
    }

    fn unknown_receiver(&self) -> Verdict {
        if !self.target.has_self {
            return Verdict::Other;
        }
        let methods = self.defs.iter().filter(|d| d.has_self).count();
        if methods == 1 && !self.target.in_trait {
            Verdict::This
        } else {
            Verdict::Possible
        }
    }

    /// A method or associated fn reached through the type `t`.
    fn of_type(&self, t: &str) -> Verdict {
        if self.target.impl_type.as_deref() == Some(t) {
            return Verdict::This;
        }
        if STD_TYPES.contains(&t) || t.starts_with("std::") {
            return Verdict::Other;
        }
        if t.starts_with("ext::") {
            // Another crate's type reaches this crate's functions only through a
            // trait implemented here.
            return if self.target.in_trait {
                Verdict::Possible
            } else {
                Verdict::Other
            };
        }
        if self
            .defs
            .iter()
            .any(|d| d.impl_type.as_deref() == Some(t) && d != self.target)
        {
            return Verdict::Other;
        }
        if self.target.free {
            return Verdict::Other;
        }
        // A type with no method of this name here: a trait impl, `Deref`, or an
        // external type.
        Verdict::Possible
    }

    /// `Q::name` with `Q` given as segments.
    fn qualified(&self, f: &File, n: Node, q: &[String]) -> Verdict {
        let Some(last) = q.last() else {
            return self.plain_fallback(f, n);
        };
        if last == "Self" {
            return match enclosing_impl_type(f, n) {
                Some(t) => self.of_type(&t),
                None => Verdict::Possible,
            };
        }
        if matches!(q[0].as_str(), "std" | "core" | "alloc") {
            return Verdict::Other;
        }
        if is_upper(last) {
            return self.of_type(&self.resolve(f, &q.join("::")));
        }
        if !self.target.free {
            return Verdict::Other; // `module::name` reaches a free fn, not a method
        }
        let here = module_of(f, n);
        let mut candidates: Vec<(String, Vec<String>)> = Vec::new();
        match q[0].as_str() {
            "crate" => candidates.push((f.krate.clone(), q[1..].to_vec())),
            "self" => {
                let mut m = here.clone();
                m.extend_from_slice(&q[1..]);
                candidates.push((f.krate.clone(), m));
            }
            "super" => {
                let mut m = here.clone();
                let mut rest = q;
                while rest.first().is_some_and(|s| s == "super") {
                    m.pop();
                    rest = &rest[1..];
                }
                m.extend_from_slice(rest);
                candidates.push((f.krate.clone(), m));
            }
            first => {
                if self.defs.iter().any(|d| d.krate == first) && first != f.krate {
                    candidates.push((first.to_string(), q[1..].to_vec()));
                }
                if first == f.krate {
                    candidates.push((f.krate.clone(), q[1..].to_vec()));
                }
                let mut m = here.clone();
                m.extend_from_slice(q);
                candidates.push((f.krate.clone(), m));
                // A module this file imports, or a crate-relative path.
                candidates.push((f.krate.clone(), q.to_vec()));
            }
        }
        let t = self.target;
        let exact =
            |d: &Def, (k, m): &(String, Vec<String>)| d.free && d.krate == *k && d.module == *m;
        if candidates.iter().any(|c| exact(t, c)) {
            return Verdict::This;
        }
        if self
            .defs
            .iter()
            .any(|d| d != t && candidates.iter().any(|c| exact(d, c)))
        {
            return Verdict::Other;
        }
        // A re-export from an ancestor module (`crate::util::waker_ref` for
        // `util/wake.rs`), when no other definition lives under that module.
        let under = |d: &Def, (k, m): &(String, Vec<String>)| {
            d.free && d.krate == *k && d.module.len() > m.len() && d.module.starts_with(m)
        };
        let reexport: Vec<&Def> = self
            .defs
            .iter()
            .filter(|d| candidates.iter().any(|c| under(d, c)))
            .collect();
        if reexport.len() == 1 && reexport[0] == t {
            return Verdict::This;
        }
        if q.len() == 1 && t.module.last() == Some(&q[0]) {
            return Verdict::This;
        }
        if !reexport.is_empty() && !reexport.contains(&t) {
            return Verdict::Other;
        }
        // `other_crate::name`, a glob re-export, a module imported under another name.
        Verdict::Possible
    }

    /// The name inside `use …;`.
    fn import(&self, f: &File, n: Node) -> Verdict {
        let p = n.parent();
        let mut segs: Vec<String> = Vec::new();
        if let Some(p) = p
            && p.kind() == "scoped_identifier"
        {
            if p.child_by_field_name("name") != Some(n) {
                return Verdict::Other; // `use a::name::b`: a module segment
            }
            if let Some(q) = p.child_by_field_name("path") {
                segs = path_segments(text(q, &f.src));
            }
        }
        let mut cur = n.parent();
        while let Some(c) = cur {
            if c.kind() == "use_declaration" {
                break;
            }
            if c.kind() == "scoped_use_list"
                && let Some(q) = c.child_by_field_name("path")
            {
                let mut s = path_segments(text(q, &f.src));
                s.extend(segs);
                segs = s;
            }
            cur = c.parent();
        }
        if segs.is_empty() {
            return Verdict::Possible;
        }
        self.qualified(f, n, &segs)
    }

    /// A bare `name` (a call or a value): the file's own fn, an import, the same
    /// module.
    fn plain(&self, path: &str, f: &File, n: Node) -> Verdict {
        let here = module_of(f, n);
        let own: Vec<&Def> = self
            .defs
            .iter()
            .filter(|d| d.path == path && d.free && here.starts_with(&d.module))
            .collect();
        if !own.is_empty() {
            return if own.contains(&self.target) {
                Verdict::This
            } else {
                Verdict::Other
            };
        }
        let imports = self.file_imports(f);
        if imports.contains(&Verdict::This) {
            return Verdict::This;
        }
        if !imports.is_empty() && imports.iter().all(|v| *v == Verdict::Other) {
            return Verdict::Other;
        }
        self.plain_fallback(f, n)
    }

    fn plain_fallback(&self, f: &File, n: Node) -> Verdict {
        let t = self.target;
        if !t.free {
            // Inside the target's own impl, a bare name is not the method either
            // (it needs `self.` or `Self::`).
            return Verdict::Other;
        }
        if t.krate == f.krate && here_is(&module_of(f, n), &t.module) {
            return Verdict::This;
        }
        Verdict::Possible
    }

    /// The verdicts of the `use` lines in `f` that import the name.
    fn file_imports(&self, f: &File) -> Vec<Verdict> {
        let mut out = Vec::new();
        let mut stack = vec![f.tree.root_node()];
        while let Some(c) = stack.pop() {
            if c.kind() == "use_declaration" {
                let mut inner = vec![c];
                while let Some(x) = inner.pop() {
                    if x.kind() == "identifier" && text(x, &f.src) == self.name {
                        let v = self.import(f, x);
                        if !(v == Verdict::Other
                            && x.parent().is_some_and(|p| {
                                p.kind() == "scoped_identifier"
                                    && p.child_by_field_name("name") != Some(x)
                            }))
                        {
                            out.push(v);
                        }
                    }
                    let mut w = x.walk();
                    inner.extend(x.named_children(&mut w));
                }
                continue;
            }
            let mut w = c.walk();
            stack.extend(c.named_children(&mut w));
        }
        out
    }

    /// The name inside a macro call's tokens: read the tokens before it.
    fn in_macro(&self, path: &str, f: &File, n: Node) -> Verdict {
        let prev = n.prev_sibling();
        match prev.map(|p| p.kind()) {
            Some(".") => {
                // A call: `.name(…)` or `.name::<T>(…)`. `if self.a.name { … }` reads
                // a field; the `{ … }` is the `if` block.
                let next = n.next_sibling();
                let called = next.is_some_and(|x| {
                    x.kind() == "::"
                        || (x.kind() == "token_tree" && text(x, &f.src).starts_with('('))
                });
                if !called {
                    return Verdict::Other; // `self.name` without a call: a field
                }
                let recv = prev.and_then(|p| p.prev_sibling());
                if recv.is_some_and(|r| text(r, &f.src) == "self") {
                    match enclosing_impl_type(f, n) {
                        Some(t) if self.target.has_self => self.of_type(&t),
                        _ => Verdict::Other,
                    }
                } else if self.target.has_self {
                    self.unknown_receiver()
                } else {
                    Verdict::Other
                }
            }
            Some("::") => {
                let q = prev
                    .and_then(|p| p.prev_sibling())
                    .map(|q| text(q, &f.src).to_string())
                    .unwrap_or_default();
                self.qualified(f, n, &[q])
            }
            _ => {
                if self.is_local(f, n) {
                    Verdict::Other
                } else {
                    self.plain(path, f, n)
                }
            }
        }
    }
}

fn here_is(here: &[String], module: &[String]) -> bool {
    here.starts_with(module)
}

/// `recv.name(…)` or `recv.name::<T>(…)`.
fn is_called(field_expr: Node) -> bool {
    let Some(gp) = field_expr.parent() else {
        return false;
    };
    if gp.kind() == "call_expression" {
        return gp.child_by_field_name("function") == Some(field_expr);
    }
    gp.kind() == "generic_function"
        && gp.parent().is_some_and(|c| {
            c.kind() == "call_expression" && c.child_by_field_name("function") == Some(gp)
        })
}

/// The pattern binds `name` (`x`, `mut x`, `(a, x)`, `Some(x)`).
fn pattern_binds(p: Node, src: &str, name: &str) -> bool {
    if p.kind() == "identifier" {
        return text(p, src) == name;
    }
    let mut c = p.walk();
    let kids: Vec<Node> = p.named_children(&mut c).collect();
    kids.into_iter().any(|k| {
        // `Foo { x: y }` binds y, `Some(x)` binds x, a path `Foo::Bar` binds nothing.
        !matches!(
            k.kind(),
            "scoped_identifier" | "type_identifier" | "field_identifier"
        ) && pattern_binds(k, src, name)
    })
}

/// The first segment of the `use` path that imports `name` into `f`
/// (`use std::time::{Duration, Instant}` → `std` for `Instant`).
fn imported_root(f: &File, name: &str) -> Option<String> {
    let mut stack = vec![f.tree.root_node()];
    while let Some(c) = stack.pop() {
        if c.kind() == "use_declaration" {
            let mut inner = vec![c];
            while let Some(x) = inner.pop() {
                let leaf = x.kind() == "identifier"
                    && text(x, &f.src) == name
                    && x.parent().is_none_or(|p| {
                        p.kind() != "scoped_identifier" || p.child_by_field_name("name") == Some(x)
                    });
                if leaf {
                    // The highest node above the name that is not a `{…}` list: its
                    // leftmost identifier is the path's first segment.
                    let mut top = x;
                    let mut cur = x.parent();
                    while let Some(p) = cur {
                        if p.kind() == "use_declaration" {
                            break;
                        }
                        if p.kind() != "use_list" {
                            top = p;
                        }
                        cur = p.parent();
                    }
                    let mut first = top;
                    while let Some(ch) = first.named_child(0) {
                        first = ch;
                    }
                    return Some(text(first, &f.src).to_string());
                }
                let mut w = x.walk();
                inner.extend(x.named_children(&mut w));
            }
            continue;
        }
        let mut w = c.walk();
        stack.extend(c.named_children(&mut w));
    }
    None
}

/// `Foo::new(…)`, `Foo { … }`, `Foo::default()`: `Foo`.
fn constructed_type(f: &File, v: Node) -> Option<String> {
    match v.kind() {
        "struct_expression" => v
            .child_by_field_name("name")
            .map(|t| type_name(text(t, &f.src))),
        "call_expression" => {
            let func = v.child_by_field_name("function")?;
            if func.kind() != "scoped_identifier" {
                return None;
            }
            let segs = path_segments(text(func, &f.src));
            let (ctor, ty) = (segs.last()?, segs.get(segs.len().checked_sub(2)?)?);
            // The type's whole path (`hir::translate::TranslatorBuilder`): `resolve`
            // reads where it comes from.
            (is_upper(ty) && matches!(ctor.as_str(), "new" | "now" | "default" | "with_capacity"))
                .then(|| segs[..segs.len() - 1].join("::"))
        }
        // `Instant::now() + timeout`: the left operand's type.
        "binary_expression" => {
            let op = v.child_by_field_name("operator").map(|o| text(o, &f.src));
            if !matches!(op, Some("+" | "-")) {
                return None;
            }
            constructed_type(f, v.child_by_field_name("left")?)
        }
        "try_expression" | "await_expression" => {
            let mut c = v.walk();
            let inner = v.named_children(&mut c).next()?;
            constructed_type(f, inner)
        }
        _ => None,
    }
}

/// The declared type of `field` in `struct owner { … }` in the same file.
fn field_type(f: &File, owner: &str, field: &str) -> Option<String> {
    field_type_text(f, owner, field).map(|t| type_name(&t))
}

/// The element of `Box<[T; N]>`, `[T]`, `Vec<T>`, `VecDeque<T>`: `T`.
fn element_type(t: &str) -> Option<String> {
    let mut t = t
        .trim()
        .trim_start_matches('&')
        .trim_start_matches("mut ")
        .trim();
    for w in ["Box<", "Arc<", "Rc<"] {
        if let Some(inner) = t.strip_prefix(w) {
            t = inner.strip_suffix('>')?.trim();
        }
    }
    if let Some(inner) = t.strip_prefix('[') {
        let inner = inner.strip_suffix(']')?;
        let elem = inner.rsplit_once(';').map_or(inner, |(e, _)| e);
        return Some(type_name(elem));
    }
    for w in ["Vec<", "VecDeque<"] {
        if let Some(inner) = t.strip_prefix(w) {
            return Some(type_name(inner.strip_suffix('>')?));
        }
    }
    None
}

fn field_type_text(f: &File, owner: &str, field: &str) -> Option<String> {
    let mut stack = vec![f.tree.root_node()];
    while let Some(c) = stack.pop() {
        if c.kind() == "struct_item"
            && c.child_by_field_name("name")
                .is_some_and(|n| text(n, &f.src) == owner)
        {
            let body = c.child_by_field_name("body")?;
            let mut w = body.walk();
            for fd in body.named_children(&mut w) {
                if fd.kind() == "field_declaration"
                    && fd
                        .child_by_field_name("name")
                        .is_some_and(|n| text(n, &f.src) == field)
                {
                    return fd
                        .child_by_field_name("type")
                        .map(|t| text(t, &f.src).to_string());
                }
            }
            return None;
        }
        let mut w = c.walk();
        stack.extend(c.named_children(&mut w));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (p, s) in files {
            let path = dir.path().join(p);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, s).unwrap();
        }
        dir
    }

    fn lines_naming(root: &Path, files: &[(&str, &str)], name: &str) -> Vec<(String, usize)> {
        let _ = root;
        let mut out = Vec::new();
        for (p, s) in files {
            for (i, l) in s.lines().enumerate() {
                if l.contains(name) && p.ends_with(".rs") {
                    out.push((p.to_string(), i + 1));
                }
            }
        }
        out
    }

    fn verdicts(files: &[(&str, &str)], at: &str, name: &str) -> Vec<(String, Verdict)> {
        let dir = tree(files);
        let lines = lines_naming(dir.path(), files, name);
        bind(dir.path(), at, name, &lines)
            .unwrap()
            .lines
            .into_iter()
            .map(|l| (format!("{}:{}", l.path, l.line), l.verdict))
            .collect()
    }

    const CARGO: (&str, &str) = ("Cargo.toml", "[package]\nname = \"demo\"\n");

    /// Held-out misses (2026-11-11): a type from `std` or a dependency, a builder
    /// chain on a dependency's type, an `if` block in a macro, a typed `self`.
    #[test]
    fn type_origin_and_typed_self() {
        let files = [
            (
                "Cargo.toml",
                "[package]\nname = \"demo\"\n[dependencies]\nbuilder-crate = \"1\"\n",
            ),
            (
                "src/lib.rs",
                "pub mod other;\nuse std::sync::Arc;\npub struct Instant;\npub struct Conf { pub size: bool }\nimpl Instant {\n    pub fn size(self: Arc<Self>) -> bool {\n        true\n    }\n}\npub fn go(i: Arc<Instant>, c: Conf) {\n    i.clone().size();\n    let v = vec![if c.size { 1 } else { 2 }];\n    drop(v);\n}\n",
            ),
            (
                "src/other.rs",
                "use std::time::Instant;\nuse builder_crate::Builder;\npub fn wait(t: u64) {\n    let deadline = Instant::now() + t;\n    deadline.size();\n    let b = Builder::new();\n    b.verbose(true).size(3);\n}\n",
            ),
        ];
        let v = verdicts(&files, "src/lib.rs:6", "size");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|x| x.1);
        assert_eq!(get("src/lib.rs:11"), Some(Verdict::This), "self: Arc<Self>");
        assert_eq!(
            get("src/lib.rs:12"),
            Some(Verdict::Other),
            "if block in vec!"
        );
        assert_eq!(get("src/other.rs:5"), Some(Verdict::Other), "std Instant");
        assert_eq!(
            get("src/other.rs:7"),
            Some(Verdict::Other),
            "dependency chain"
        );
    }

    #[test]
    fn enclosing_fns_names_callers_macros_and_definitions() {
        let root = tree(&[
            CARGO,
            (
                "src/lib.rs",
                "fn add(x: u8) -> u8 { x }\nfn run() {\n    add(1);\n}\nmacro_rules! many {\n    () => { add(2) };\n}\nstruct S;\nimpl S {\n    fn add(&self) {}\n}\n",
            ),
        ]);
        let lines: Vec<(String, usize)> = [1, 3, 6, 10]
            .iter()
            .map(|l| ("src/lib.rs".to_string(), *l))
            .collect();
        let got: Vec<(Option<String>, Option<&str>)> = enclosing_fns(root.path(), "add", &lines)
            .into_iter()
            .map(|f| (f.caller, f.defines))
            .collect();
        assert_eq!(
            got,
            vec![
                (None, Some("Function")),
                (Some("run".to_string()), None),
                (Some("macro_rules! many".to_string()), None),
                (None, Some("Method")),
            ]
        );
    }

    #[test]
    fn free_fn_binds_through_imports_and_qualifiers() {
        let files = [
            CARGO,
            ("src/lib.rs", "mod util;\nmod task;\nmod sched;\n"),
            (
                "src/util.rs",
                "pub(crate) fn waker_ref(x: u32) -> u32 {\n    x\n}\n",
            ),
            (
                "src/task.rs",
                "pub(crate) fn waker_ref(x: u8) -> u8 {\n    x\n}\nfn run() {\n    let waker_ref = waker_ref(1);\n    drop(waker_ref);\n}\n",
            ),
            (
                "src/sched.rs",
                "use crate::util::waker_ref;\nstruct Handle;\nimpl Handle {\n    fn waker_ref(&self) -> u32 {\n        waker_ref(2)\n    }\n}\nfn go(h: &Handle) {\n    h.waker_ref();\n    crate::util::waker_ref(3);\n    crate::task::waker_ref(4);\n}\n",
            ),
        ];
        let v = verdicts(&files, "src/util.rs:1", "waker_ref");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|x| x.1);
        assert_eq!(
            get("src/task.rs:1"),
            Some(Verdict::Other),
            "another definition"
        );
        assert_eq!(
            get("src/task.rs:5"),
            Some(Verdict::Other),
            "task's own waker_ref"
        );
        assert_eq!(get("src/task.rs:6"), Some(Verdict::Other), "a local");
        assert_eq!(get("src/sched.rs:1"), Some(Verdict::This), "the import");
        assert_eq!(get("src/sched.rs:4"), Some(Verdict::Other), "a method");
        assert_eq!(
            get("src/sched.rs:5"),
            Some(Verdict::This),
            "call through the import"
        );
        assert_eq!(
            get("src/sched.rs:9"),
            Some(Verdict::Other),
            "method call on &Handle"
        );
        assert_eq!(get("src/sched.rs:10"), Some(Verdict::This), "crate path");
        assert_eq!(
            get("src/sched.rs:11"),
            Some(Verdict::Other),
            "the other module"
        );
    }

    #[test]
    fn method_binds_by_receiver_type() {
        let files = [
            CARGO,
            (
                "src/lib.rs",
                "pub struct Instant;\nimpl Instant {\n    pub fn checked_add(&self, d: u64) -> Option<Instant> {\n        None\n    }\n}\nstruct Buf { filled: usize }\nimpl Buf {\n    fn grow(&mut self, n: usize) {\n        self.filled.checked_add(n);\n    }\n}\nfn deadline(now: Instant) {\n    now.checked_add(1);\n    Instant::checked_add(&now, 2);\n    let t = Instant::now();\n    t.checked_add(3);\n}\n",
            ),
        ];
        let v = verdicts(&files, "src/lib.rs:3", "checked_add");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|x| x.1);
        assert_eq!(get("src/lib.rs:10"), Some(Verdict::Other), "usize field");
        assert_eq!(get("src/lib.rs:14"), Some(Verdict::This), "typed parameter");
        assert_eq!(get("src/lib.rs:15"), Some(Verdict::This), "Type::name");
        assert_eq!(
            get("src/lib.rs:17"),
            Some(Verdict::This),
            "let t = Instant::now()"
        );
    }

    #[test]
    fn self_call_binds_to_the_enclosing_impl() {
        let files = [
            CARGO,
            (
                "src/lib.rs",
                "mod util {\n    pub fn trim(x: u8) -> u8 { x }\n}\nstruct Std;\nimpl Std {\n    fn trim(&self, x: u8) -> u8 {\n        util::trim(x)\n    }\n    fn write(&self) {\n        self.trim(1);\n    }\n}\n",
            ),
        ];
        let v = verdicts(&files, "src/lib.rs:2", "trim");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|x| x.1);
        assert_eq!(get("src/lib.rs:6"), Some(Verdict::Other));
        assert_eq!(get("src/lib.rs:7"), Some(Verdict::This), "util::trim");
        assert_eq!(
            get("src/lib.rs:10"),
            Some(Verdict::Other),
            "self.trim is Std's"
        );
        let v = verdicts(&files, "src/lib.rs:6", "trim");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|x| x.1);
        assert_eq!(get("src/lib.rs:7"), Some(Verdict::Other));
        assert_eq!(get("src/lib.rs:10"), Some(Verdict::This));
    }

    #[test]
    fn index_receiver_and_macro_field() {
        let files = [
            CARGO,
            (
                "src/lib.rs",
                "struct Level;\nimpl Level {\n    fn next(&self) -> u8 { 0 }\n}\nstruct Wheel { levels: Box<[Level; 6]>, next: u8 }\nimpl Wheel {\n    fn next(&self) -> u8 {\n        self.levels[0].next();\n        assert!(matches!(self.next, 0));\n        for (_, l) in self.levels.iter().enumerate() { l.next(); }\n        for l in &self.levels[1..] { l.next(); }\n        0\n    }\n}\n",
            ),
        ];
        let v = verdicts(&files, "src/lib.rs:3", "next");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|x| x.1);
        assert_eq!(
            get("src/lib.rs:8"),
            Some(Verdict::This),
            "element of Box<[Level; 6]>"
        );
        assert_eq!(
            get("src/lib.rs:10"),
            Some(Verdict::This),
            "for over self.levels.iter()"
        );
        assert_eq!(
            get("src/lib.rs:11"),
            Some(Verdict::This),
            "for over &self.levels[1..]"
        );
        assert_eq!(
            get("src/lib.rs:9"),
            Some(Verdict::Other),
            "a field read in a macro"
        );
    }

    #[test]
    fn cfg_macro_blocks_hold_items() {
        let files = [
            CARGO,
            (
                "src/lib.rs",
                "cfg_rt! {\n    pub(crate) fn spawn_inner(x: u8) -> u8 {\n        x\n    }\n}\nfn run() {\n    spawn_inner(1);\n}\n",
            ),
        ];
        let v = verdicts(&files, "src/lib.rs:2", "spawn_inner");
        assert_eq!(v, vec![("src/lib.rs:7".to_string(), Verdict::This)]);
    }

    #[test]
    fn at_must_name_a_function() {
        let files = [CARGO, ("src/lib.rs", "fn a() {}\n")];
        let dir = tree(&files);
        let err = bind(dir.path(), "src/lib.rs:1", "b", &[]).err().unwrap();
        assert!(err.to_string().contains("no function `b`"), "{err}");
        assert!(parse_at("src/lib.rs").is_err());
    }
}
