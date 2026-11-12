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
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;
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
    /// The line of that function's name (`[in f:LINE]`): the next `--at` when an
    /// agent walks callers up. None for a `macro_rules!` body.
    pub caller_line: Option<usize>,
    /// A `use` line: it names the function but does not call it.
    pub import: bool,
}

pub struct Binding {
    pub target: Def,
    /// Every input line except the target's own definition line, in input order.
    pub lines: Vec<BoundLine>,
}

struct File {
    /// The path relative to the root.
    rel: String,
    src: String,
    tree: Tree,
    krate: String,
    module: Vec<String>,
    /// The crate's root directory (the one with `lib.rs` or `main.rs`).
    base: PathBuf,
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
///
/// `type_defs(T)` lists the lines (`path`, `line`) that define a type named `T`
/// anywhere in the index (`struct`, `enum`, `union`, `type`, `trait`). When two types
/// share the name of the target's `impl` type, a receiver's written type path is
/// resolved to its module (`use` lines, `pub use` re-exports) before it counts.
pub fn bind(
    root: &Path,
    at: &str,
    name: &str,
    lines: &[(String, usize)],
    type_defs: &dyn Fn(&str) -> Vec<(String, usize)>,
) -> Result<Binding> {
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
        let (krate, module, package, base) = crate_module(root, p);
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
                rel: p.clone(),
                src,
                tree,
                krate,
                module,
                base,
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

    let _own = OwnWrapper::set(target.impl_type.as_deref());
    let (returns, ret_decls) = method_returns(&files);
    let mut cx = Cx {
        name,
        target: &target,
        defs: &defs,
        crates: workspace_crates(root),
        returns,
        ret_decls,
        files: &files,
        root,
        loaded: RefCell::new(HashMap::new()),
        uses: RefCell::new(HashMap::new()),
        items: RefCell::new(HashMap::new()),
        ty_homes: Vec::new(),
        target_home: None,
        type_defs,
        aliases: RefCell::new(HashMap::new()),
        homes: RefCell::new(HashMap::new()),
        depth: std::cell::Cell::new(0),
    };
    cx.find_type_homes();
    let mut out = Vec::new();
    for (p, line) in lines {
        if *p == target.path && *line == target.line {
            continue;
        }
        let (verdict, at, import) = match files.get(p) {
            Some(f) => cx.line(p, f, *line),
            None => (Verdict::Other, None, false), // not Rust: not a use of a Rust function
        };
        out.push(BoundLine {
            path: p.clone(),
            line: *line,
            verdict,
            caller: at.as_ref().map(|(c, _)| c.clone()),
            caller_line: at.and_then(|(_, l)| l),
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
    /// The line of that function's name. None for a macro.
    pub caller_line: Option<usize>,
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
                        out.caller_line = Some(nm.start_position().row + 1);
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
            for (x, y) in arm_marks(&bytes, b, close) {
                for c in &mut bytes[x..y] {
                    if *c != b'\n' {
                        *c = b' ';
                    }
                }
            }
        }
        src = String::from_utf8(bytes).ok()?;
    }
    let tree = parser.parse(&src, None)?;
    Some((src, tree))
}

/// A macro body of labelled arms (tokio `cfg_metrics_variant! { stable: { .. },
/// unstable: { .. } }`): the byte ranges of each label with its `{`, and each `}` with
/// its `,`. Blanked, the arms' items parse as the enclosing block's (cfg twins).
/// Empty when the body is not such a list.
fn arm_marks(s: &[u8], from: usize, to: usize) -> Vec<(usize, usize)> {
    let skip_ws = |mut i: usize| {
        while i < to && s[i].is_ascii_whitespace() {
            i += 1;
        }
        i
    };
    let mut out = Vec::new();
    let mut i = from;
    loop {
        i = skip_ws(i);
        if i >= to {
            return out;
        }
        let a = i;
        while i < to && (s[i].is_ascii_alphanumeric() || s[i] == b'_') {
            i += 1;
        }
        let colon = skip_ws(i);
        let open = skip_ws(colon + 1);
        if i == a || colon >= to || s[colon] != b':' || open >= to || s[open] != b'{' {
            return Vec::new();
        }
        let mut depth = 0;
        let mut m = open;
        while m < to {
            match s[m] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            m += 1;
        }
        if m >= to {
            return Vec::new();
        }
        let comma = skip_ws(m + 1);
        let end = if comma < to && s[comma] == b',' {
            comma + 1
        } else {
            m + 1
        };
        out.push((a, open + 1));
        out.push((m, end));
        i = end;
    }
}

// ---------------------------------------------------------------------------
// Crates and modules
// ---------------------------------------------------------------------------

/// The crate name (`grep_printer`) and the module path (`["util", "wake"]`) of a file,
/// from its place under the crate root (the nearest directory with `lib.rs` or
/// `main.rs`, at or under the package's `Cargo.toml`).
fn crate_module(root: &Path, rel: &str) -> (String, Vec<String>, Option<PathBuf>, PathBuf) {
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
    let crate_base = base.clone();
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
    (krate, module, package, crate_base)
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

/// The deepest `receiver_type` nesting: a chain, a field path and the locals they
/// read stay far below it.
const MAX_DEPTH: usize = 64;

/// Types `type_name` sees through to their first generic argument.
const WRAPPERS: [&str; 14] = [
    "NonNull",
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

thread_local! {
    /// The target's own type when it is named like a wrapper (tokio's `loom`
    /// `UnsafeCell`, its async `Mutex`): `type_name` keeps it, not its argument.
    static OWN_WRAPPER: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// `type_name` reads through `base<T>` to `T`.
fn unwraps(base: &str) -> bool {
    WRAPPERS.contains(&base) && OWN_WRAPPER.with(|o| o.borrow().as_deref() != Some(base))
}

/// Clears `OWN_WRAPPER` when a `bind` call returns.
struct OwnWrapper;

impl OwnWrapper {
    fn set(t: Option<&str>) -> Self {
        let t = t.filter(|t| WRAPPERS.contains(t)).map(str::to_string);
        OWN_WRAPPER.with(|o| *o.borrow_mut() = t);
        OwnWrapper
    }
}

impl Drop for OwnWrapper {
    fn drop(&mut self) {
        OWN_WRAPPER.with(|o| *o.borrow_mut() = None);
    }
}

/// The type an `impl` block is for, as written for `type_name`. An inherent impl
/// (`impl<T> UnsafeCell<T>`) is of the crate's own type, so a wrapper's name is kept
/// (tokio's `loom` `UnsafeCell`, its `Mutex`); a trait impl for `Arc<Handle>` is
/// Handle's.
fn impl_written<'s>(imp: Node, ty: Node, src: &'s str) -> &'s str {
    let t = text(ty, src);
    if imp.child_by_field_name("trait").is_some() {
        t
    } else {
        t.split('<').next().unwrap_or(t).trim()
    }
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
    if let (true, Some(g)) = (unwraps(base), generics) {
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
/// Method names of common std types (`Cell`, `Option`, atomics, collections,
/// channels, I/O): a receiver of unknown type calling one may be a std value.
const STD_METHODS: &[&str] = &[
    "and_then",
    "as_mut",
    "as_ptr",
    "as_ref",
    "borrow",
    "borrow_mut",
    "cancel",
    "clear",
    "clone",
    "close",
    "cmp",
    "compare_exchange",
    "contains",
    "contains_key",
    "drain",
    "entry",
    "eq",
    "extend",
    "fetch_add",
    "fetch_and",
    "fetch_or",
    "fetch_sub",
    "fetch_update",
    "filter",
    "find",
    "first",
    "flush",
    "fmt",
    "get",
    "get_mut",
    "get_or_insert_with",
    "hash",
    "insert",
    "into_inner",
    "is_empty",
    "is_none",
    "is_some",
    "iter",
    "iter_mut",
    "join",
    "last",
    "len",
    "load",
    "lock",
    "map",
    "next",
    "poll",
    "pop",
    "pop_back",
    "pop_front",
    "push",
    "push_back",
    "push_front",
    "read",
    "recv",
    "remove",
    "replace",
    "reserve",
    "retain",
    "send",
    "set",
    "shutdown",
    "sort",
    "split",
    "store",
    "swap",
    "take",
    "to_owned",
    "to_string",
    "truncate",
    "try_lock",
    "try_recv",
    "try_send",
    "unwrap",
    "wake",
    "wake_by_ref",
    "with",
    "with_mut",
    "write",
    "write_all",
];

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
                    .map(|t| type_name(impl_written(o, t, &f.src))),
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

/// A caller's name and the line of its name (None for a `macro_rules!` body).
type Caller = (String, Option<usize>);

/// The function (or `macro_rules!` body) `n` is in, with the line of its name.
fn caller(f: &File, n: Node) -> Option<Caller> {
    let mut cur = n.parent();
    while let Some(c) = cur {
        if let Some(m) = macro_name(c, &f.src) {
            return Some((m, None));
        }
        if c.kind() == "function_item"
            && let Some(nm) = c.child_by_field_name("name")
            && nm != n
        {
            return Some((
                text(nm, &f.src).to_string(),
                Some(nm.start_position().row + 1),
            ));
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
        .map(|t| type_name(impl_written(i, t, &f.src)))
}

fn is_upper(s: &str) -> bool {
    s.chars().next().is_some_and(|c| c.is_ascii_uppercase())
}

/// The bounds of the generic `name` of `func`, from its `<..>` list or `where`.
fn generic_bound(g: &File, func: Node, name: &str) -> Option<String> {
    let mut stack = vec![func];
    let mut out = Vec::new();
    while let Some(n) = stack.pop() {
        if matches!(n.kind(), "constrained_type_parameter" | "where_predicate")
            && n.child_by_field_name("left")
                .is_some_and(|l| text(l, &g.src) == name)
            && let Some(b) = n.child_by_field_name("bounds")
        {
            out.push(text(b, &g.src).to_string());
        }
        if n == func || matches!(n.kind(), "type_parameters" | "where_clause") {
            let mut w = n.walk();
            stack.extend(n.named_children(&mut w).filter(|c| c.kind() != "block"));
        }
    }
    (!out.is_empty()).then(|| out.join(" + "))
}

/// The argument types of the `Fn`/`FnMut`/`FnOnce` in a bound text:
/// `FnMut(Snapshot, &T) -> R` → `["Snapshot", "&T"]`.
fn fn_trait_args(bound: &str) -> Option<Vec<String>> {
    let b = bound.as_bytes();
    let mut at = None;
    for (i, _) in bound.match_indices("Fn") {
        if i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_') {
            continue;
        }
        let rest = &bound[i + 2..];
        let rest = rest
            .strip_prefix("Mut")
            .or_else(|| rest.strip_prefix("Once"))
            .unwrap_or(rest);
        if let Some(r) = rest.trim_start().strip_prefix('(') {
            at = Some(bound.len() - r.len());
            break;
        }
    }
    let start = at?;
    let mut depth = 0;
    let mut parts = Vec::new();
    let mut from = start;
    for (j, ch) in bound[start..].char_indices() {
        let j = start + j;
        match ch {
            '<' | '(' | '[' => depth += 1,
            '>' if j > 0 && b[j - 1] == b'-' => {}
            '>' | ']' => depth -= 1,
            ')' if depth == 0 => {
                parts.push(bound[from..j].trim().to_string());
                return Some(parts.into_iter().filter(|p| !p.is_empty()).collect());
            }
            ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(bound[from..j].trim().to_string());
                from = j + 1;
            }
            _ => {}
        }
    }
    None
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
    /// (impl type, method) → the type it returns (`method_returns`); `None` when two
    /// impls disagree or the return type says nothing.
    returns: HashMap<(String, String), Option<String>>,
    /// (impl type, method) → where its return type is written: (file, module, the
    /// type with `&`, `Result` and `Option` removed). For a shared type name.
    ret_decls: RetDecls,
    /// Every parsed file, for a struct declared in another file than the use.
    files: &'a HashMap<String, File>,
    root: &'a Path,
    /// Module files read from disk to follow re-exports, by relative path.
    loaded: RefCell<HashMap<String, Option<Rc<File>>>>,
    /// Each file's `use` items, read once.
    uses: RefCell<HashMap<String, Rc<Vec<UseItem>>>>,
    /// Each file's module-level items (name, module), read once.
    items: RefCell<HashMap<String, Rc<FileItems>>>,
    /// The modules that define a type named like the target's `impl` type, when
    /// there are two or more (else empty: the bare name is enough).
    ty_homes: Vec<Home>,
    /// The module of the target's `impl` type, when `ty_homes` is not empty.
    target_home: Option<Home>,
    /// The lines that define a type of a given name (`bind`'s `type_defs`).
    type_defs: &'a dyn Fn(&str) -> Vec<(String, usize)>,
    /// `type T = X;`: T → X tagged where it is written; None when T is not one alias.
    aliases: RefCell<HashMap<String, Option<String>>>,
    /// Each type name's defining modules (`homes_of`), read once.
    homes: RefCell<HashMap<String, Rc<Vec<Home>>>>,
    /// `receiver_type` nesting: a cycle the rules missed gives "unknown", never a
    /// stack overflow (that aborts `rfx mcp`).
    depth: std::cell::Cell<usize>,
}

impl Cx<'_> {
    /// The verdict of one line: the strongest of its occurrences of the name.
    fn line(&self, path: &str, f: &File, line: usize) -> (Verdict, Option<Caller>, bool) {
        let Some(row) = f.src.split('\n').nth(line - 1) else {
            return (Verdict::Other, None, false);
        };
        let start: usize = f.src.split('\n').take(line - 1).map(|l| l.len() + 1).sum();
        let mut best: Option<(Verdict, Option<Caller>, bool)> = None;
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
        // The innermost closure parameter or `match` / `if let` pattern that binds
        // `var`: (its start, the type).
        let mut closure: Option<(usize, String)> = None;
        let mut cur = n.parent();
        while let Some(c) = cur {
            if c.kind() == "function_item" {
                break;
            }
            if closure.is_none()
                && c.kind() == "closure_expression"
                && let Some(t) = self.closure_param_type(f, c, var)
            {
                closure = Some((c.start_byte(), t));
            }
            if closure.is_none()
                && let Some(t) = self.arm_binding(f, c, n, var)
            {
                closure = Some((c.start_byte(), t)); // shadowed like a closure parameter
            }
            if closure.is_none()
                && c.kind() == "for_expression"
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
                        .and_then(|t| self.written_type(f, t));
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
                // A closure's own `let`s count only inside that closure.
                let holds_n = ch.start_byte() <= n.start_byte() && n.end_byte() <= ch.end_byte();
                if ch.kind() != "function_item" && (ch.kind() != "closure_expression" || holds_n) {
                    stack.push(ch);
                }
            }
        }
        let last_let = lets.into_iter().max_by_key(|l| l.start_byte());
        // A closure parameter shadows the `let`s before the closure.
        if let Some((start, t)) = closure
            && last_let.is_none_or(|l| l.start_byte() < start)
        {
            return Some(t);
        }
        if let Some(l) = last_let {
            found = l
                .child_by_field_name("type")
                .and_then(|t| self.written_type(f, t))
                .or_else(|| {
                    // `at: v`: only earlier `let`s, so `let x = x.f()` does not loop.
                    l.child_by_field_name("value").and_then(|v| {
                        constructed_type(f, v)
                            .map(|p| self.tag_path(f, &module_of(f, v), Some(v.start_byte()), &p))
                            .or_else(|| self.fn_value_type(f, v, l, var))
                            .or_else(|| self.receiver_type(f, v, v))
                    })
                });
            if found.is_none() {
                return Some(String::new()); // a local, type not written
            }
        }
        found
    }

    /// A type node as written (`&Arc<Self>`), tagged; `Self` is the enclosing impl's.
    fn written_type(&self, f: &File, t: Node) -> Option<String> {
        let w = text(t, &f.src);
        if type_name(w) == "Self" {
            return self.impl_tag(f, t);
        }
        Some(self.tag(f, &module_of(f, t), Some(t.start_byte()), w))
    }

    /// The declared type of `field` of the struct `t` (tagged), read where `t` is
    /// defined: its home module when the name is shared, else `f`, else any file.
    fn field_type(&self, f: &File, t: &str, field: &str) -> Option<String> {
        let home = t.split_once(TAG).map(|(_, h)| h.to_string());
        let owner = bare(&type_name(t)).to_string();
        let field_in =
            |g: &File| field_decl(g, &owner, field).map(|(t, m)| self.tag(g, &m, None, &t));
        // A shared type name with a known home: that module's struct.
        if home.as_deref() == Some("?") {
            return None; // two structs of that name, and not known which
        }
        let homed = home.and_then(|h| {
            let mut it = h.split('/');
            let k = it.next()?;
            let m: Vec<String> = it.map(String::from).collect();
            let rel = self.module_file(k, &m)?;
            self.with_file(&rel, field_in).flatten()
        });
        homed
            .or_else(|| field_in(f))
            .or_else(|| self.files.values().find_map(field_in))
    }

    /// The type of `var` bound by a parameter of the closure `c`: a typed parameter
    /// (`|x: Foo|`), or a field of a struct pattern (`|Data { ctx, .. }|`). `Some("")`
    /// when the closure binds it with no type written; None when it does not bind it.
    fn closure_param_type(&self, f: &File, c: Node, var: &str) -> Option<String> {
        let ps = c.child_by_field_name("parameters")?;
        let mut w = ps.walk();
        for (i, p) in ps.named_children(&mut w).enumerate() {
            let (pat, ty) = if p.kind() == "parameter" {
                (
                    p.child_by_field_name("pattern")?,
                    p.child_by_field_name("type"),
                )
            } else {
                (p, None)
            };
            if !pattern_binds(pat, &f.src, var) {
                continue;
            }
            if let Some(t) = ty {
                return Some(self.written_type(f, t).unwrap_or_default());
            }
            if pat.kind() == "struct_pattern" {
                return Some(self.struct_field_bound(f, pat, var).unwrap_or_default());
            }
            // `|mut next| ..` passed to `fn f<F: FnMut(Snapshot)>(&self, f: F)`.
            let plain = pat.kind() == "identifier"
                || (pat.kind() == "mut_pattern"
                    && pat
                        .named_child(pat.named_child_count().saturating_sub(1) as u32)
                        .is_some_and(|x| x.kind() == "identifier"));
            let t = plain.then(|| self.closure_arg_type(f, c, i)).flatten();
            return Some(t.unwrap_or_default());
        }
        None
    }

    /// `c` is a `match` arm, or an `if let` / `while let`, whose pattern binds `var` for
    /// `n`: the bound value's type (`Some(x)` / `Ok(x)` of the scrutinee; a variant's
    /// or tuple struct's field). `Some("")` when bound with no type known.
    fn arm_binding(&self, f: &File, c: Node, n: Node, var: &str) -> Option<String> {
        let inside = |x: Option<Node>| {
            x.is_some_and(|x| x.start_byte() <= n.start_byte() && n.end_byte() <= x.end_byte())
        };
        let (pat, value) = match c.kind() {
            "match_arm" => {
                if !inside(c.child_by_field_name("value")) {
                    return None;
                }
                let scrut = c.parent()?.parent()?.child_by_field_name("value")?;
                (c.child_by_field_name("pattern")?, scrut)
            }
            "if_expression" | "while_expression" => {
                let body = if c.kind() == "if_expression" {
                    "consequence"
                } else {
                    "body"
                };
                if !inside(c.child_by_field_name(body)) {
                    return None;
                }
                let cond = c.child_by_field_name("condition")?;
                let lc = if cond.kind() == "let_condition" {
                    cond
                } else {
                    let mut w = cond.walk();
                    let found = cond.named_children(&mut w).find(|x| {
                        x.kind() == "let_condition"
                            && x.child_by_field_name("pattern")
                                .is_some_and(|p| pattern_binds(p, &f.src, var))
                    });
                    found?
                };
                (
                    lc.child_by_field_name("pattern")?,
                    lc.child_by_field_name("value")?,
                )
            }
            _ => return None,
        };
        if !pattern_binds(pat, &f.src, var) {
            return None;
        }
        // Typed where it is written: inside the arm, `match x { Some(x) => .. }` loops.
        let scrut = self
            .receiver_type(f, value, value)
            .filter(|t| !t.is_empty());
        Some(self.pattern_type(f, pat, var, scrut).unwrap_or_default())
    }

    /// The type `pat` gives `var`, when the whole value has type `whole`.
    fn pattern_type(
        &self,
        f: &File,
        pat: Node,
        var: &str,
        whole: Option<String>,
    ) -> Option<String> {
        match pat.kind() {
            "identifier" => (text(pat, &f.src) == var).then_some(whole).flatten(),
            "match_pattern" => self.pattern_type(f, pat.named_child(0)?, var, whole),
            "mut_pattern" | "ref_pattern" | "reference_pattern" => {
                let last = pat.named_child(pat.named_child_count().checked_sub(1)? as u32)?;
                self.pattern_type(f, last, var, whole)
            }
            "struct_pattern" => self.struct_field_bound(f, pat, var),
            "or_pattern" => {
                let mut w = pat.walk();
                let alt = pat
                    .named_children(&mut w)
                    .find(|p| pattern_binds(*p, &f.src, var))?;
                self.pattern_type(f, alt, var, whole)
            }
            "tuple_struct_pattern" => {
                let ty = pat.child_by_field_name("type")?;
                let mut w = pat.walk();
                let kids: Vec<Node> = pat.named_children(&mut w).filter(|k| *k != ty).collect();
                let j = kids.iter().position(|k| pattern_binds(*k, &f.src, var))?;
                let segs = path_segments(text(ty, &f.src));
                let (v, e) = segs.split_last()?;
                let field = match v.as_str() {
                    "Some" | "Ok" => whole,
                    "Err" | "None" => None,
                    _ if e.is_empty() => self.tuple_field(f, ty, v, None, j),
                    _ => self.tuple_field(f, ty, &e.join("::"), Some(v), j),
                };
                self.pattern_type(f, kids[j], var, field)
            }
            _ => None,
        }
    }

    /// Field `j` of the tuple variant `E::variant`, or of the tuple struct `E` (no
    /// variant), written `e` at `at` in `f`; tagged where the type is defined.
    fn tuple_field(
        &self,
        f: &File,
        at: Node,
        e: &str,
        variant: Option<&String>,
        j: usize,
    ) -> Option<String> {
        let t = if e == "Self" {
            self.impl_tag(f, at)?
        } else {
            self.tag(f, &module_of(f, at), Some(at.start_byte()), e)
        };
        let (name, home) = match t.split_once(TAG) {
            Some((b, h)) => (b.to_string(), Some(h.to_string())),
            None => (bare(&type_name(&t)).to_string(), None),
        };
        if home.as_deref() == Some("?") {
            return None;
        }
        let want = home.and_then(|h| {
            let mut it = h.split('/');
            let k = it.next()?.to_string();
            let m: Vec<String> = it.map(String::from).collect();
            self.module_file(&k, &m)
        });
        let defs: Vec<(String, usize)> = (self.type_defs)(&name)
            .into_iter()
            .filter(|(rel, _)| want.as_ref().is_none_or(|w| w == rel))
            .collect();
        let [(rel, line)] = defs.as_slice() else {
            return None;
        };
        self.with_file(rel, |g| {
            let mut stack = vec![g.tree.root_node()];
            while let Some(x) = stack.pop() {
                if matches!(x.kind(), "enum_item" | "struct_item")
                    && x.start_position().row < *line
                    && *line <= x.end_position().row + 1
                    && x.child_by_field_name("name")
                        .is_some_and(|nm| text(nm, &g.src) == name)
                {
                    let body = x.child_by_field_name("body")?;
                    let fields = match variant {
                        None => body,
                        Some(v) => {
                            let mut w = body.walk();
                            let found = body.named_children(&mut w).find(|ev| {
                                ev.kind() == "enum_variant"
                                    && ev
                                        .child_by_field_name("name")
                                        .is_some_and(|nm| text(nm, &g.src) == v.as_str())
                            });
                            found?.child_by_field_name("body")?
                        }
                    };
                    if fields.kind() != "ordered_field_declaration_list" {
                        return None;
                    }
                    let mut w = fields.walk();
                    let ty = fields.children_by_field_name("type", &mut w).nth(j)?;
                    return Some(self.tag(g, &module_of(g, x), None, text(ty, &g.src)));
                }
                let mut w = x.walk();
                stack.extend(x.named_children(&mut w));
            }
            None
        })
        .flatten()
    }

    /// The type of parameter `i` of the closure `c`, from the function it is passed
    /// to: that argument's `impl FnMut(A, ..)`, or its generic `F: FnMut(A, ..)`.
    /// The function is looked up by name in this file, then in the files read.
    fn closure_arg_type(&self, f: &File, c: Node, i: usize) -> Option<String> {
        let args = c.parent().filter(|a| a.kind() == "arguments")?;
        let call = args.parent().filter(|x| x.kind() == "call_expression")?;
        let mut w = args.walk();
        let k = args.named_children(&mut w).position(|a| a == c)?;
        let func = call.child_by_field_name("function")?;
        let (name, method) = match func.kind() {
            "field_expression" => (text(func.child_by_field_name("field")?, &f.src), true),
            "scoped_identifier" => (text(func.child_by_field_name("name")?, &f.src), false),
            "identifier" => (text(func, &f.src), false),
            _ => return None,
        };
        let from = |g: &File| -> Option<String> {
            let mut found: Option<String> = None;
            let mut stack = vec![g.tree.root_node()];
            while let Some(n) = stack.pop() {
                if n.kind() == "function_item"
                    && n.child_by_field_name("name")
                        .is_some_and(|x| text(x, &g.src) == name)
                {
                    let t = self.fn_closure_param(g, n, k, method, i)?;
                    if found.as_ref().is_some_and(|x| *x != t) {
                        return None; // two functions of that name disagree
                    }
                    found = Some(t);
                    continue;
                }
                let mut w = n.walk();
                stack.extend(n.named_children(&mut w));
            }
            found
        };
        from(f).or_else(|| {
            let mut out: Option<String> = None;
            for g in self.files.values().filter(|g| g.rel != f.rel) {
                if let Some(t) = from(g) {
                    if out.as_ref().is_some_and(|x| *x != t) {
                        return None;
                    }
                    out = Some(t);
                }
            }
            out
        })
    }

    /// For the function item `func`, argument `k` of a call (a method call skips
    /// `self`): when it is a closure type, the type of its parameter `i`, tagged.
    fn fn_closure_param(
        &self,
        g: &File,
        func: Node,
        k: usize,
        method: bool,
        i: usize,
    ) -> Option<String> {
        let ps = func.child_by_field_name("parameters")?;
        let mut w = ps.walk();
        let all: Vec<Node> = ps
            .named_children(&mut w)
            .filter(|p| p.kind() != "attribute_item")
            .collect();
        let has_self = all.first().is_some_and(|p| p.kind() == "self_parameter");
        let idx = if has_self && method { k + 1 } else { k };
        let p = all.get(idx).filter(|p| p.kind() == "parameter")?;
        let ty = text(p.child_by_field_name("type")?, &g.src).trim();
        let bound = if ty.starts_with("impl ") || ty.starts_with("&") || ty.starts_with("Box<") {
            ty.to_string()
        } else {
            generic_bound(g, func, ty)?
        };
        let arg = fn_trait_args(&bound)?.into_iter().nth(i)?;
        if type_name(&arg) == "Self" {
            return self.impl_tag(g, func);
        }
        Some(self.tag(g, &module_of(g, func), None, &arg))
    }

    /// `Data { ctx, .. }` or `Data { ctx: x, .. }` binding `var`: the declared type
    /// of that field of `Data`.
    fn struct_field_bound(&self, f: &File, pat: Node, var: &str) -> Option<String> {
        if pat.kind() != "struct_pattern" {
            return None;
        }
        let ty = pat.child_by_field_name("type")?;
        let owner = self.tag(
            f,
            &module_of(f, ty),
            Some(ty.start_byte()),
            text(ty, &f.src),
        );
        let mut w = pat.walk();
        for fp in pat.named_children(&mut w) {
            if fp.kind() != "field_pattern" {
                continue;
            }
            let name = fp.child_by_field_name("name")?;
            let field = text(name, &f.src);
            match fp.child_by_field_name("pattern") {
                // `ctx: x` (a plain name) binds `x` to the field.
                Some(sub) if sub.kind() == "identifier" && text(sub, &f.src) == var => {
                    return self.field_type(f, &owner, field);
                }
                Some(sub) if pattern_binds(sub, &f.src, var) => {
                    return self.struct_field_bound(f, sub, var);
                }
                None if field == var => return self.field_type(f, &owner, field),
                _ => {}
            }
        }
        None
    }

    /// `let (tx, rx) = channel(16);`: the type a free function of this file returns,
    /// or the tuple element at `var`'s position.
    fn fn_value_type(&self, f: &File, v: Node, l: Node, var: &str) -> Option<String> {
        let mut v = v;
        while matches!(v.kind(), "try_expression" | "await_expression") {
            v = v.named_child(0)?;
        }
        if v.kind() != "call_expression" {
            return None;
        }
        let mut func = v.child_by_field_name("function")?;
        if func.kind() == "generic_function" {
            func = func.child_by_field_name("function")?;
        }
        if !matches!(func.kind(), "identifier" | "scoped_identifier") {
            return None;
        }
        let segs = path_segments(text(func, &f.src));
        let name = segs.last()?.clone();
        let pat = l.child_by_field_name("pattern")?;
        let pick = |g: &File, decl: Node| -> Option<String> {
            let mut ret = text(decl.child_by_field_name("return_type")?, &g.src);
            if pat.kind() == "tuple_pattern" {
                // `_` is not a named node: count every element, not the punctuation.
                let mut c = pat.walk();
                let i = pat
                    .children(&mut c)
                    .filter(|p| !matches!(p.kind(), "(" | ")" | ","))
                    .position(|p| pattern_binds(p, &f.src, var))?;
                ret = tuple_elems(ret)?.get(i).copied()?;
            }
            return_type(ret, "")?;
            Some(self.tag(g, &module_of(g, decl), None, unwrap_ret(ret)))
        };
        if segs.len() == 1
            && let Some(decl) = free_fn(f, &name)
        {
            return pick(f, decl);
        }
        // `queue::local()`, or an imported `local()`: the fn in its module's file.
        let (k, p) = self.absolute(f, &module_of(f, v), &segs, 6, Some(v.start_byte()))?;
        let rel = self.module_file(&k, &p[..p.len().checked_sub(1)?])?;
        self.with_file(&rel, |g| pick(g, free_fn(g, &name)?))
            .flatten()
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
                    return element_type(&field_type_text(f, &owner, field)?)
                        .map(|e| self.tag_bare(e));
                }
                _ => return None,
            }
        }
    }

    /// `self`, a typed local, `self.field` (the field's declared type), `Foo::new()`.
    fn receiver_type(&self, f: &File, r: Node, at: Node) -> Option<String> {
        let d = self.depth.get();
        if d >= MAX_DEPTH {
            return None;
        }
        self.depth.set(d + 1);
        let out = self.receiver_type_at(f, r, at);
        self.depth.set(d);
        out
    }

    fn receiver_type_at(&self, f: &File, r: Node, at: Node) -> Option<String> {
        match r.kind() {
            "self" => self.impl_tag(f, at),
            "identifier" => self
                .local_type_of(f, at, text(r, &f.src))
                .filter(|t| !t.is_empty()),
            "field_expression" => {
                // `self.handle.shared`: each step's declared field type.
                let v = r.child_by_field_name("value")?;
                let field = text(r.child_by_field_name("field")?, &f.src);
                let t = if v.kind() == "self" {
                    self.impl_tag(f, at)?
                } else {
                    self.receiver_type(f, v, at)?
                };
                self.field_type(f, &t, field)
            }
            // `open(..)?`, `fut.await`: the call's type (`Result` is already unwrapped).
            "try_expression" | "await_expression" => self.receiver_type(f, r.named_child(0)?, at),
            "parenthesized_expression" | "reference_expression" | "unary_expression" => {
                let mut c = r.walk();
                let inner = r.named_children(&mut c).last()?;
                self.receiver_type(f, inner, at)
            }
            "call_expression" => {
                let func = r.child_by_field_name("function")?;
                if func.kind() != "field_expression" {
                    // `Foo::new()`, or `Builder::new_current_thread()`: what the
                    // associated fn declares it returns.
                    return constructed_type(f, r)
                        .map(|p| self.tag_path(f, &module_of(f, r), Some(r.start_byte()), &p))
                        .or_else(|| {
                            if func.kind() != "scoped_identifier" {
                                return None;
                            }
                            let segs = path_segments(text(func, &f.src));
                            let (m, ty) = (segs.last()?, segs.get(segs.len().checked_sub(2)?)?);
                            if !is_upper(ty) {
                                return None;
                            }
                            let owner = segs[..segs.len() - 1].join("::");
                            let at = Some(r.start_byte());
                            self.ret(&self.tag_path(f, &module_of(f, r), at, &owner), m)
                        });
                }
                // A chain `root.a(..).b(..)`: the methods, outermost first.
                let mut methods = vec![text(func.child_by_field_name("field")?, &f.src)];
                let mut root = func.child_by_field_name("value")?;
                loop {
                    match root.kind() {
                        "call_expression" => {
                            let g = root.child_by_field_name("function")?;
                            if g.kind() != "field_expression" {
                                break;
                            }
                            methods.push(text(g.child_by_field_name("field")?, &f.src));
                            root = g.child_by_field_name("value")?;
                        }
                        "try_expression" | "await_expression" => {
                            root = root.named_child(0)?;
                        }
                        _ => break,
                    }
                }
                let t = self.receiver_type(f, root, at)?;
                // `b.encoding(..).bom_sniffing(..)` on an external builder `b`: the
                // chain stays external.
                let resolved = self.resolve(f, &t);
                if resolved.starts_with("ext::") {
                    return Some(resolved);
                }
                // This workspace's type: each method's declared return type in turn
                // (`Builder::new_current_thread().enable_all().build()`).
                let mut t = type_name(&t);
                if unwraps(t.trim_start_matches("std::")) {
                    return None; // `Arc::new(x).clone()`: the wrapped type is not known
                }
                for m in methods.iter().rev() {
                    t = self.ret(&t, m)?;
                }
                Some(t)
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
                element_type(&field_type_text(f, &owner, field)?).map(|e| self.tag_bare(e))
            }
            _ => None,
        }
    }

    /// The type `ty.m(..)` returns, from `m`'s declaration in an `impl ty` block.
    /// `unwrap`, `expect` and `clone` keep the type (`Result`/`Option` are already
    /// unwrapped by `return_type`).
    fn ret(&self, ty: &str, m: &str) -> Option<String> {
        if matches!(m, "unwrap" | "expect" | "clone" | "unwrap_or_default") {
            return Some(ty.to_string());
        }
        let b = bare(ty);
        let Some(r) = self.returns.get(&(type_name(b), m.to_string())) else {
            // Not declared on `ty`: a pin projection or a guard (`self.project()`,
            // `x.lock()`) derefs to the same type, in this crate's code. `ty` is
            // already out of its `Result`/`Option`, so `.context(..)?` keeps it too.
            let same = matches!(
                m,
                "project"
                    | "project_ref"
                    | "lock"
                    | "borrow"
                    | "borrow_mut"
                    | "get_mut"
                    | "read"
                    | "write"
                    | "as_ref"
                    | "as_mut"
                    | "as_deref"
                    | "context"
                    | "with_context"
                    | "map_err"
                    | "ok"
                    | "ok_or"
                    | "ok_or_else"
                    | "unwrap_or"
                    | "unwrap_or_else"
                    | "cloned"
                    | "copied"
            );
            return (same && !b.starts_with("std::") && !STD_TYPES.contains(&b))
                .then(|| ty.to_string());
        };
        let r = r.clone()?;
        if r == type_name(b) {
            return Some(ty.to_string()); // `-> Self`: the same type, the same tag
        }
        // A shared type name: resolve the written path where the method declares it.
        if self.target.impl_type.as_deref() == Some(r.as_str())
            && let Some((rel, module, written)) = self.ret_decls.get(&(type_name(b), m.to_string()))
            && let Some(t) = self.with_file(rel, |g| self.tag(g, module, None, written))
        {
            return Some(t);
        }
        Some(self.tag_bare(r))
    }

    /// A type path as `of_type` reads it: `std::T` for a standard type, `ext::T` for
    /// another crate's type, else `T`.
    fn resolve(&self, f: &File, t: &str) -> String {
        if t.starts_with("std::") || t.starts_with("ext::") || t.contains(TAG) {
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
        let t = self.receiver_type(f, receiver, n);
        let v = match &t {
            Some(t) => self.of_type(&self.resolve(f, t)),
            None => self.unknown_receiver(),
        };
        if std::env::var_os("REFLEX_BIND_DEBUG").is_some() {
            let row = n.start_position().row + 1;
            eprintln!("bind: {}:{row} receiver {t:?} -> {v:?}", f.rel);
        }
        v
    }

    fn unknown_receiver(&self) -> Verdict {
        if !self.target.has_self {
            return Verdict::Other;
        }
        let others = self
            .defs
            .iter()
            .filter(|d| d.has_self && !self.is_target(d))
            .count();
        // A receiver of unknown type may be a std type: `x.replace(..)` on a `Cell`.
        if others == 0 && !self.target.in_trait && !STD_METHODS.contains(&self.name) {
            Verdict::This
        } else {
            Verdict::Possible
        }
    }

    /// A method or associated fn reached through the type `t`.
    fn of_type(&self, t: &str) -> Verdict {
        let (t, tag) = match t.split_once(TAG) {
            Some((b, h)) => (b, Some(h)),
            None => (t, None),
        };
        if self.target.impl_type.as_deref() == Some(t) {
            if self.ty_homes.is_empty() {
                return Verdict::This;
            }
            // Two types share the name: the receiver's home decides.
            return match (tag, &self.target_home) {
                (Some(h), Some(th)) if h != "?" && h == home_key(th) => Verdict::This,
                (Some(h), Some(_)) if h != "?" => Verdict::Other,
                _ => Verdict::Possible,
            };
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
            .any(|d| d.impl_type.as_deref() == Some(t) && !self.is_target(d))
        {
            return Verdict::Other;
        }
        if self.target.free {
            return Verdict::Other;
        }
        // `type EntryList = LinkedList<..>`: the aliased type.
        if let Some(x) = self.alias(t) {
            return self.of_type(&x);
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
            return match self.impl_tag(f, n) {
                Some(t) => self.of_type(&t),
                None => Verdict::Possible,
            };
        }
        if matches!(q[0].as_str(), "std" | "core" | "alloc") {
            return Verdict::Other;
        }
        if is_upper(last) {
            let t = self.tag_path(f, &module_of(f, n), Some(n.start_byte()), &q.join("::"));
            return self.of_type(&self.resolve(f, &t));
        }
        if !self.target.free {
            return Verdict::Other; // `module::name` reaches a free fn, not a method
        }
        let here = module_of(f, n);
        let mut candidates: Vec<(String, Vec<String>)> = Vec::new();
        // The path read through this file's `use` lines and the modules' `pub use`
        // re-exports: where the function is defined.
        if let Some((k, mut p)) = self.absolute(f, &here, q, 6, Some(n.start_byte())) {
            p.push(self.name.to_string());
            let (k, p) = self.defined_at(&k, &p, 6);
            if let Some((_, m)) = p.split_last() {
                candidates.push((k, m.to_vec()));
            }
        }
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
            .any(|d| !self.is_target(d) && candidates.iter().any(|c| exact(d, c)))
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
        if !reexport.is_empty() && reexport.iter().all(|d| self.is_target(d)) {
            return Verdict::This;
        }
        if q.len() == 1 && t.module.last() == Some(&q[0]) {
            return Verdict::This;
        }
        if !reexport.is_empty() && !reexport.iter().any(|d| self.is_target(d)) {
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
            return if own.iter().any(|d| self.is_target(d)) {
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
                    match self.impl_tag(f, n) {
                        Some(t) if self.target.has_self => self.of_type(&t),
                        _ => Verdict::Other,
                    }
                } else if self.target.has_self {
                    // `assert_eq!(x.name(), 1)`: a local `x` (not `a.x`) has the
                    // type its `let` or parameter gives it.
                    let local = recv.filter(|r| {
                        r.kind() == "identifier" && r.prev_sibling().is_none_or(|p| p.kind() != ".")
                    });
                    match local.and_then(|r| {
                        self.local_type_of(f, n, text(r, &f.src))
                            .filter(|t| !t.is_empty())
                    }) {
                        Some(t) => self.of_type(&self.resolve(f, &t)),
                        None => self.unknown_receiver(),
                    }
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
    // `Foo { x, .. }` binds x (a shorthand field).
    if matches!(p.kind(), "identifier" | "shorthand_field_identifier") {
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

/// The one free function (not in an `impl` or `trait`) named `name` in `f`.
fn free_fn<'t>(f: &'t File, name: &str) -> Option<Node<'t>> {
    let mut found = Vec::new();
    let mut stack = vec![f.tree.root_node()];
    while let Some(n) = stack.pop() {
        match n.kind() {
            "impl_item" | "trait_item" => continue,
            "function_item" => {
                if n.child_by_field_name("name")
                    .is_some_and(|m| text(m, &f.src) == name)
                {
                    found.push(n);
                }
                continue;
            }
            _ => {}
        }
        let mut c = n.walk();
        stack.extend(n.named_children(&mut c));
    }
    (found.len() == 1).then(|| found[0])
}

/// `(Sender<T>, Receiver<T>)`: its top-level elements.
fn tuple_elems(t: &str) -> Option<Vec<&str>> {
    let inner = t.trim().strip_prefix('(')?.strip_suffix(')')?;
    let (mut depth, mut start, mut out) = (0i32, 0, Vec::new());
    for (i, ch) in inner.char_indices() {
        match ch {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(inner[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    let last = inner[start..].trim();
    if !last.is_empty() {
        out.push(last);
    }
    Some(out)
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
            // `Arc::new(Foo::new())`: a smart pointer derefs to its content's methods.
            if matches!(ty.as_str(), "Arc" | "Rc" | "Box") && matches!(ctor.as_str(), "new" | "pin")
            {
                let args = v.child_by_field_name("arguments")?;
                return constructed_type(f, args.named_child(0)?);
            }
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

/// A file's items: (name, module path).
type FileItems = Vec<(String, Vec<String>)>;

type RetDecls = HashMap<(String, String), (String, Vec<String>, String)>;

/// Every method of every `impl T` block in `files`: (T, name) → the type it returns
/// (`return_type`). Two impls that disagree (generic impls, cfg twins) give `None`.
/// Also where each return type is written (`RetDecls`).
fn method_returns(
    files: &HashMap<String, File>,
) -> (HashMap<(String, String), Option<String>>, RetDecls) {
    let mut out: HashMap<(String, String), Option<String>> = HashMap::new();
    let mut decls = RetDecls::new();
    for f in files.values() {
        let mut stack = vec![f.tree.root_node()];
        while let Some(n) = stack.pop() {
            let mut c = n.walk();
            stack.extend(n.named_children(&mut c));
            if n.kind() != "function_item" {
                continue;
            }
            let Some(imp) = ancestor(n, &["impl_item", "function_item"]) else {
                continue;
            };
            if imp.kind() != "impl_item" {
                continue;
            }
            let (Some(ty), Some(nm)) = (
                imp.child_by_field_name("type"),
                n.child_by_field_name("name"),
            ) else {
                continue;
            };
            let ty = type_name(impl_written(imp, ty, &f.src));
            let written = n
                .child_by_field_name("return_type")
                .map(|r| text(r, &f.src));
            let ret = written.and_then(|r| return_type(r, &ty));
            let key = (ty, text(nm, &f.src).to_string());
            if let Some(w) = written {
                decls.insert(
                    key.clone(),
                    (f.rel.clone(), module_of(f, n), unwrap_ret(w).to_string()),
                );
            }
            match out.get(&key) {
                Some(prev) if *prev != ret => {
                    out.insert(key, None);
                }
                Some(_) => {}
                None => {
                    out.insert(key, ret);
                }
            }
        }
    }
    (out, decls)
}

/// `&mut io::Result<Foo<T>>` → `Foo<T>`: the written type a method's caller holds.
fn unwrap_ret(ret: &str) -> &str {
    let t = ret.trim().trim_start_matches('&').trim_start();
    let t = t.strip_prefix("mut ").unwrap_or(t).trim_start();
    if let Some(i) = t.find('<') {
        let last = t[..i].rsplit("::").next().unwrap_or(&t[..i]).trim();
        if matches!(last, "Result" | "Option")
            && let Some(j) = t.rfind('>')
        {
            return unwrap_ret(first_generic(&t[i + 1..j]));
        }
    }
    t
}

/// A declared return type as a receiver type: `&mut Self` → `impl_ty`,
/// `io::Result<Runtime>` → `Runtime`, `Option<Arc<Handle>>` → `Handle`. `None` for a
/// type that says nothing (`()`, `impl Trait`, a generic `T`).
fn return_type(ret: &str, impl_ty: &str) -> Option<String> {
    let t = ret.trim();
    if let Some(i) = t.find('<') {
        let base = t[..i].trim().trim_start_matches('&').trim();
        let last = base.rsplit("::").next().unwrap_or(base);
        if matches!(last, "Result" | "Option") {
            let inner = &t[i + 1..t.rfind('>')?];
            return return_type(first_generic(inner), impl_ty);
        }
    }
    if t.starts_with("impl ") || t.starts_with('(') {
        return None;
    }
    let n = type_name(t);
    if n == "Self" {
        return Some(impl_ty.to_string());
    }
    // A standard type or a primitive (`of_type`: never this crate's method). A
    // one-letter name is a type parameter.
    (n.starts_with("std::") || STD_TYPES.contains(&n.as_str()) || (n.len() > 1 && is_upper(&n)))
        .then_some(n)
}

/// The first top-level argument of a generic list `A, B<C, D>`.
fn first_generic(g: &str) -> &str {
    let mut depth = 0;
    for (i, c) in g.char_indices() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => return &g[..i],
            _ => {}
        }
    }
    g
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
    field_decl(f, owner, field).map(|(t, _)| t)
}

/// The written type of `field` in `struct owner` in `f` (named or tuple field `0`),
/// and the struct's module.
fn field_decl(f: &File, owner: &str, field: &str) -> Option<(String, Vec<String>)> {
    let mut stack = vec![f.tree.root_node()];
    while let Some(c) = stack.pop() {
        if c.kind() == "struct_item"
            && c.child_by_field_name("name")
                .is_some_and(|n| text(n, &f.src) == owner)
        {
            let body = c.child_by_field_name("body")?;
            let module = module_of(f, c);
            let mut w = body.walk();
            if body.kind() == "ordered_field_declaration_list" {
                // `struct ReadHalf<'a>(&'a UnixStream)`: field `0` is the first type.
                let i: usize = field.parse().ok()?;
                let types: Vec<Node> = body
                    .named_children(&mut w)
                    .filter(|t| !matches!(t.kind(), "visibility_modifier" | "attribute_item"))
                    .collect();
                return types.get(i).map(|t| (text(*t, &f.src).to_string(), module));
            }
            for fd in body.named_children(&mut w) {
                if fd.kind() == "field_declaration"
                    && fd
                        .child_by_field_name("name")
                        .is_some_and(|n| text(n, &f.src) == field)
                {
                    return fd
                        .child_by_field_name("type")
                        .map(|t| (text(t, &f.src).to_string(), module));
                }
            }
            return None;
        }
        let mut w = c.walk();
        stack.extend(c.named_children(&mut w));
    }
    None
}

// ---------------------------------------------------------------------------
// Name resolution: which module a written path names
// ---------------------------------------------------------------------------

/// A crate and a module path: where an item is defined.
type Home = (String, Vec<String>);

/// One name a `use` declaration brings into scope.
#[derive(Clone, Debug)]
struct UseItem {
    /// The name in scope (`c` for `use a::b::c`, `x` for `use a::b as x`); `*` for a glob.
    bound: String,
    /// The path of the item, as written (`["a", "b", "c"]`); a glob's module path.
    full: Vec<String>,
    /// The module the `use` sits in.
    module: Vec<String>,
    /// The block (`{ … }`, byte range) a `use` inside a function sits in: it
    /// applies there only, before the module's names. None at module level.
    scope: Option<(usize, usize)>,
}

/// A tag on a receiver type whose bare name the target's type shares with another
/// type: `T#crate/a/b` (the module that defines it) or `T#?` (not resolved).
const TAG: char = '#';

fn bare(t: &str) -> &str {
    t.split(TAG).next().unwrap_or(t)
}

fn home_key((k, m): &Home) -> String {
    let mut s = k.clone();
    for x in m {
        s.push('/');
        s.push_str(x);
    }
    s
}

/// `Arc<Mutex<a::Foo<T>>>` → `["a", "Foo"]`: the path `type_name` takes the last
/// segment of.
fn type_path(t: &str) -> Vec<String> {
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
    let segs = path_segments(base.trim());
    if let (Some(last), Some(g)) = (segs.last(), generics)
        && unwraps(last)
    {
        return type_path(first_generic(g));
    }
    segs
}

/// Every name the `use` declarations of `f` bring into scope.
fn use_items(f: &File) -> Vec<UseItem> {
    type Scope = Option<(usize, usize)>;
    fn walk(
        f: &File,
        n: Node,
        prefix: &[String],
        module: &[String],
        scope: Scope,
        out: &mut Vec<UseItem>,
    ) {
        let mut push = |bound: String, full: Vec<String>| {
            out.push(UseItem {
                bound,
                full,
                module: module.to_vec(),
                scope,
            })
        };
        match n.kind() {
            "identifier" | "crate" | "super" | "metavariable" => {
                let mut full = prefix.to_vec();
                full.push(text(n, &f.src).to_string());
                push(text(n, &f.src).to_string(), full);
            }
            "self" => {
                // `use a::b::{self}`: the module `b`.
                if let Some(last) = prefix.last() {
                    push(last.clone(), prefix.to_vec());
                }
            }
            "scoped_identifier" => {
                let mut full = prefix.to_vec();
                full.extend(path_segments(text(n, &f.src)));
                if let Some(last) = full.last().cloned() {
                    if last == "self" {
                        full.pop();
                        if let Some(l) = full.last().cloned() {
                            push(l, full);
                        }
                    } else {
                        push(last, full);
                    }
                }
            }
            "use_as_clause" => {
                let (Some(path), Some(alias)) = (
                    n.child_by_field_name("path"),
                    n.child_by_field_name("alias"),
                ) else {
                    return;
                };
                let mut full = prefix.to_vec();
                full.extend(path_segments(text(path, &f.src)));
                push(text(alias, &f.src).to_string(), full);
            }
            "use_wildcard" => {
                let mut full = prefix.to_vec();
                let t = text(n, &f.src).trim_end_matches('*').trim_end_matches("::");
                full.extend(path_segments(t));
                push("*".to_string(), full);
            }
            "use_list" => {
                let mut c = n.walk();
                for ch in n.named_children(&mut c) {
                    walk(f, ch, prefix, module, scope, out);
                }
            }
            "scoped_use_list" => {
                let mut p = prefix.to_vec();
                if let Some(path) = n.child_by_field_name("path") {
                    p.extend(path_segments(text(path, &f.src)));
                }
                if let Some(list) = n.child_by_field_name("list") {
                    walk(f, list, &p, module, scope, out);
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    let mut stack = vec![f.tree.root_node()];
    while let Some(c) = stack.pop() {
        if c.kind() == "use_declaration" {
            if let Some(arg) = c.child_by_field_name("argument") {
                let module = module_of(f, c);
                let scope = ancestor(c, &["block", "mod_item", "source_file"])
                    .filter(|b| b.kind() == "block")
                    .map(|b| (b.start_byte(), b.end_byte()));
                walk(f, arg, &[], &module, scope, &mut out);
            }
            continue;
        }
        let mut w = c.walk();
        stack.extend(c.named_children(&mut w));
    }
    out
}

/// The module-level items of `f` (types, traits, free functions, modules, consts):
/// (name, module).
fn file_items(f: &File) -> Vec<(String, Vec<String>)> {
    let mut out = Vec::new();
    let mut stack = vec![f.tree.root_node()];
    while let Some(c) = stack.pop() {
        let mut w = c.walk();
        stack.extend(c.named_children(&mut w));
        if !matches!(
            c.kind(),
            "struct_item"
                | "enum_item"
                | "union_item"
                | "type_item"
                | "trait_item"
                | "function_item"
                | "mod_item"
                | "const_item"
                | "static_item"
        ) {
            continue;
        }
        if ancestor(c, &["impl_item", "trait_item", "function_item"]).is_some() {
            continue;
        }
        if let Some(nm) = c.child_by_field_name("name") {
            out.push((text(nm, &f.src).to_string(), module_of(f, c)));
        }
    }
    out
}

impl Cx<'_> {
    /// Run `g` on the file at `rel`: a parsed input file, or a module file read from
    /// disk (once) to follow a re-export.
    fn with_file<R>(&self, rel: &str, g: impl FnOnce(&File) -> R) -> Option<R> {
        if let Some(f) = self.files.get(rel) {
            return Some(g(f));
        }
        let cached = self.loaded.borrow().get(rel).cloned();
        let f = match cached {
            Some(f) => f?,
            None => {
                let f = std::fs::read_to_string(self.root.join(rel))
                    .ok()
                    .and_then(|src| {
                        let mut parser = Parser::new();
                        parser
                            .set_language(&tree_sitter_rust::LANGUAGE.into())
                            .ok()?;
                        let (src, tree) = parse_flat(&mut parser, src)?;
                        let (krate, module, _, base) = crate_module(self.root, rel);
                        Some(Rc::new(File {
                            rel: rel.to_string(),
                            src,
                            tree,
                            krate,
                            module,
                            base,
                            deps: HashSet::new(),
                        }))
                    });
                self.loaded.borrow_mut().insert(rel.to_string(), f.clone());
                f?
            }
        };
        Some(g(&f))
    }

    fn uses_of(&self, f: &File) -> Rc<Vec<UseItem>> {
        if let Some(u) = self.uses.borrow().get(&f.rel) {
            return u.clone();
        }
        let u = Rc::new(use_items(f));
        self.uses.borrow_mut().insert(f.rel.clone(), u.clone());
        u
    }

    fn items_of(&self, f: &File) -> Rc<FileItems> {
        if let Some(i) = self.items.borrow().get(&f.rel) {
            return i.clone();
        }
        let i = Rc::new(file_items(f));
        self.items.borrow_mut().insert(f.rel.clone(), i.clone());
        i
    }

    /// The file of module `m` of crate `k` (`a/b.rs`, `a/b/mod.rs`, or the file of
    /// the nearest enclosing module for an inline `mod b { }`), relative to the root.
    /// `type T = X;` defined once in the workspace (and T defined nowhere else): X,
    /// tagged in the alias's module.
    fn alias(&self, t: &str) -> Option<String> {
        if let Some(x) = self.aliases.borrow().get(t) {
            return x.clone();
        }
        self.aliases.borrow_mut().insert(t.to_string(), None); // a cycle stops here
        let defs = (self.type_defs)(t);
        // Every definition must be an alias, and all of them of one type (tokio
        // writes `type WaitList = LinkedList<..>` in two modules).
        let one = |rel: &String, line: &usize| -> Option<String> {
            let line = *line;
            self.with_file(rel, |g| {
                let src_line = g.src.split('\n').nth(line - 1)?;
                let col = src_line.len() - src_line.trim_start().len();
                let at = tree_sitter::Point::new(line - 1, col);
                let row = g.tree.root_node().descendant_for_point_range(at, at)?;
                let mut stack = vec![row];
                let mut n = row;
                while let Some(p) = n.parent() {
                    stack.push(p);
                    n = p;
                }
                let item = stack
                    .into_iter()
                    .find(|n| n.kind() == "type_item" && n.start_position().row == line - 1)?;
                if item.child_by_field_name("name").map(|n| text(n, &g.src)) != Some(t) {
                    return None;
                }
                let v = item.child_by_field_name("type")?;
                Some(self.tag(g, &module_of(g, item), None, text(v, &g.src)))
            })
            .flatten()
        };
        let mut x: Option<String> = None;
        for (rel, line) in &defs {
            match one(rel, line) {
                Some(a) if x.as_ref().is_none_or(|b| *b == a) => x = Some(a),
                _ => {
                    x = None;
                    break;
                }
            }
        }
        self.aliases.borrow_mut().insert(t.to_string(), x.clone());
        x
    }

    fn module_file(&self, k: &str, m: &[String]) -> Option<String> {
        let base = self
            .files
            .values()
            .find(|f| f.krate == k)
            .map(|f| f.base.clone())
            .or_else(|| {
                self.loaded
                    .borrow()
                    .values()
                    .flatten()
                    .find(|f| f.krate == k)
                    .map(|f| f.base.clone())
            })?;
        for end in (0..=m.len()).rev() {
            let part = &m[..end];
            let cands: Vec<PathBuf> = if part.is_empty() {
                vec![base.join("lib.rs"), base.join("main.rs")]
            } else {
                let mut d = base.clone();
                for x in &part[..part.len() - 1] {
                    d.push(x);
                }
                let last = &part[part.len() - 1];
                vec![d.join(format!("{last}.rs")), d.join(last).join("mod.rs")]
            };
            for c in cands {
                if c.is_file() {
                    return c
                        .strip_prefix(self.root)
                        .ok()
                        .map(|r| r.to_string_lossy().replace('\\', "/"));
                }
            }
        }
        None
    }

    /// The absolute path (crate, segments) that `segs`, written in module `here` of
    /// `f`, names: through `crate`/`self`/`super`, crate names, the module's `use`
    /// lines and globs, and its own items. None when the first segment is not found.
    fn absolute(
        &self,
        f: &File,
        here: &[String],
        segs: &[String],
        depth: usize,
        at: Option<usize>,
    ) -> Option<Home> {
        if depth == 0 || segs.is_empty() {
            return None;
        }
        let first = segs[0].as_str();
        let rest = &segs[1..];
        let join = |mut a: Vec<String>, b: &[String]| {
            a.extend_from_slice(b);
            a
        };
        match first {
            "crate" => return Some((f.krate.clone(), rest.to_vec())),
            "self" => return Some((f.krate.clone(), join(here.to_vec(), rest))),
            "super" => {
                let mut m = here.to_vec();
                let mut r = segs;
                while r.first().is_some_and(|s| s == "super") {
                    m.pop()?;
                    r = &r[1..];
                }
                return Some((f.krate.clone(), join(m, r)));
            }
            "std" | "core" | "alloc" => return Some(("std".to_string(), rest.to_vec())),
            "Self" => return None,
            _ => {}
        }
        let uses = self.uses_of(f);
        // The blocks around `at` (innermost first), then the module: a `use` in a
        // function body shadows the module's names.
        let mut scopes: Vec<Option<(usize, usize)>> = uses
            .iter()
            .filter_map(|u| u.scope)
            .filter(|(a, b)| at.is_some_and(|x| *a <= x && x < *b))
            .map(Some)
            .collect();
        scopes.sort_by_key(|s| s.map_or(usize::MAX, |(a, b)| b - a));
        scopes.dedup();
        scopes.push(None);
        for sc in scopes {
            let in_scope = |u: &&UseItem| u.scope == sc && (sc.is_some() || u.module == here);
            for u in uses.iter().filter(in_scope).filter(|u| u.bound == first) {
                if u.full.first().map(String::as_str) == Some(first) && u.full.len() == 1 {
                    // `use name;`: a crate.
                    return Some((first.to_string(), rest.to_vec()));
                }
                let inner = u.scope.map(|(a, _)| a);
                if let Some((k, p)) = self.absolute(f, &u.module, &u.full, depth - 1, inner) {
                    return Some((k, join(p, rest)));
                }
            }
            if sc.is_none() {
                if self
                    .items_of(f)
                    .iter()
                    .any(|(n, m)| n == first && m == here)
                {
                    return Some((f.krate.clone(), join(here.to_vec(), segs)));
                }
                if first == f.krate || self.crates.contains(first) || f.deps.contains(first) {
                    return Some((first.to_string(), rest.to_vec()));
                }
            }
            // A glob (`use super::*` in a test module): the first glob whose module
            // has the name.
            for u in uses.iter().filter(in_scope).filter(|u| u.bound == "*") {
                let inner = u.scope.map(|(a, _)| a);
                let Some((k, g)) = self.absolute(f, &u.module, &u.full, depth - 1, inner) else {
                    continue;
                };
                let mut p = g.clone();
                p.push(first.to_string());
                if self.module_has(&k, &g, first, depth - 1) {
                    return Some((k, join(p, rest)));
                }
            }
        }
        None
    }

    /// Module `m` of crate `k` defines or imports `name`.
    fn module_has(&self, k: &str, m: &[String], name: &str, depth: usize) -> bool {
        if depth == 0 {
            return false;
        }
        let Some(rel) = self.module_file(k, m) else {
            return false;
        };
        self.with_file(&rel, |g| {
            self.items_of(g).iter().any(|(n, gm)| n == name && gm == m)
                || self
                    .uses_of(g)
                    .iter()
                    .any(|u| u.bound == name && u.module == m && u.scope.is_none())
        })
        .unwrap_or(false)
    }

    /// Where the item at the absolute path `(k, p)` is defined: `pub use` re-exports
    /// are followed through module files. A path that cannot be followed is returned
    /// as it is.
    fn defined_at(&self, k: &str, p: &[String], depth: usize) -> Home {
        let unchanged = (k.to_string(), p.to_vec());
        let Some((name, m)) = p.split_last() else {
            return unchanged;
        };
        if depth == 0 || k == "std" {
            return unchanged;
        }
        let Some(rel) = self.module_file(k, m) else {
            return unchanged;
        };
        let next = self.with_file(&rel, |g| {
            if self.items_of(g).iter().any(|(n, gm)| n == name && gm == m) {
                return None; // defined here
            }
            let uses = self.uses_of(g);
            let u = uses
                .iter()
                .find(|u| &u.bound == name && u.module == m && u.scope.is_none())?;
            self.absolute(g, m, &u.full, depth, None)
        });
        match next.flatten() {
            Some((k2, p2)) if (k2.as_str(), p2.as_slice()) != (k, p) => {
                self.defined_at(&k2, &p2, depth - 1)
            }
            _ => unchanged,
        }
    }

    /// The home (among `ty_homes`) of the type written `written` in module `here`
    /// of `f`. None when the path cannot be followed to one home.
    fn type_home(
        &self,
        f: &File,
        here: &[String],
        at: Option<usize>,
        written: &str,
    ) -> Option<Home> {
        let segs = type_path(written);
        let (k, p) = self.absolute(f, here, &segs, 6, at)?;
        let (k, p) = self.defined_at(&k, &p, 6);
        let m = p.split_last()?.1.to_vec();
        let home = (k, m);
        let homes = self.homes_of(&type_name(written));
        if homes.contains(&home) {
            return Some(home);
        }
        // A re-export the files did not show: one home under the module.
        let under: Vec<&Home> = homes
            .iter()
            .filter(|(hk, hm)| *hk == home.0 && hm.starts_with(&home.1))
            .collect();
        (under.len() == 1).then(|| under[0].clone())
    }

    /// Fill `ty_homes` and `target_home` from the index's type definitions.
    fn find_type_homes(&mut self) {
        let Some(t) = self.target.impl_type.clone() else {
            return;
        };
        if self.target.in_trait || t.starts_with("std::") {
            return;
        }
        let homes = self.homes_of(&t);
        if homes.len() < 2 {
            return;
        }
        self.ty_homes = homes.to_vec();
        self.set_target_home();
    }

    /// The modules that define a type named `t` (struct, enum, union, alias, trait),
    /// read once.
    fn homes_of(&self, t: &str) -> Rc<Vec<Home>> {
        if let Some(h) = self.homes.borrow().get(t) {
            return h.clone();
        }
        let mut homes: Vec<Home> = Vec::new();
        for (path, line) in (self.type_defs)(t) {
            let home = self
                .with_file(&path, |g| {
                    let row = line.checked_sub(1)?;
                    let mut stack = vec![g.tree.root_node()];
                    while let Some(c) = stack.pop() {
                        if c.start_position().row > row || c.end_position().row < row {
                            continue;
                        }
                        if matches!(
                            c.kind(),
                            "struct_item" | "enum_item" | "union_item" | "type_item" | "trait_item"
                        ) && c
                            .child_by_field_name("name")
                            .is_some_and(|n| text(n, &g.src) == t && n.start_position().row == row)
                        {
                            return Some((g.krate.clone(), module_of(g, c)));
                        }
                        let mut w = c.walk();
                        stack.extend(c.named_children(&mut w));
                    }
                    None
                })
                .flatten();
            if let Some(h) = home
                && !homes.contains(&h)
            {
                homes.push(h);
            }
        }
        let homes = Rc::new(homes);
        self.homes.borrow_mut().insert(t.to_string(), homes.clone());
        homes
    }

    /// `target_home`: the module of the target's `impl` type.
    fn set_target_home(&mut self) {
        let target = self.target;
        self.target_home = self
            .with_file(&target.path, |f| {
                let row = target.line.checked_sub(1)?;
                let n = f.tree.root_node().descendant_for_point_range(
                    tree_sitter::Point { row, column: 0 },
                    tree_sitter::Point { row, column: 0 },
                )?;
                // The `fn` line: its node, then the `impl` above it.
                let mut cur = Some(n);
                let mut imp = None;
                while let Some(c) = cur {
                    if c.kind() == "impl_item" {
                        imp = Some(c);
                        break;
                    }
                    cur = c.parent();
                }
                let imp = imp.or_else(|| {
                    let mut stack = vec![f.tree.root_node()];
                    while let Some(c) = stack.pop() {
                        if c.kind() == "impl_item"
                            && c.start_position().row <= row
                            && row <= c.end_position().row
                        {
                            return Some(c);
                        }
                        let mut w = c.walk();
                        stack.extend(c.named_children(&mut w));
                    }
                    None
                })?;
                let ty = imp.child_by_field_name("type")?;
                self.type_home(f, &module_of(f, imp), None, impl_written(imp, ty, &f.src))
            })
            .flatten();
        if std::env::var_os("REFLEX_BIND_DEBUG").is_some() {
            eprintln!(
                "bind: homes of {:?}: {:?}; target {:?}",
                self.target.impl_type, self.ty_homes, self.target_home
            );
        }
    }

    /// The receiver type written `written` in `f` at module `here`: its bare name,
    /// tagged with its home when the target's type name is shared.
    fn tag(&self, f: &File, here: &[String], at: Option<usize>, written: &str) -> String {
        let t = type_name(written);
        if !self.needs_home(&t) {
            return t;
        }
        // Another crate's type of that name: `of_type` reads `std::` / `ext::`.
        if let Some((k, _)) = self.absolute(f, here, &type_path(written), 6, at) {
            if k == "std" {
                return format!("std::{t}");
            }
            if k != f.krate && !self.crates.contains(&k) {
                return format!("ext::{t}");
            }
        }
        let out = match self.type_home(f, here, at, written) {
            Some(h) => format!("{t}{TAG}{}", home_key(&h)),
            None => format!("{t}{TAG}?"),
        };
        if std::env::var_os("REFLEX_BIND_DEBUG").is_some() {
            eprintln!("bind: {} {here:?} `{written}` -> {out}", f.rel);
        }
        out
    }

    /// A written type path (`a::Foo`) as `resolve` reads it, tagged when its name is
    /// the target's shared type name.
    fn tag_path(&self, f: &File, here: &[String], at: Option<usize>, path: &str) -> String {
        if !self.needs_home(&type_name(path)) {
            return path.to_string();
        }
        self.tag(f, here, at, path)
    }

    /// A type name that needs its home in a tag: the target's `impl` type when that
    /// name is shared (`ty_homes`), or another type name defined twice or more (its
    /// fields are read by home, `receiver_type`).
    fn needs_home(&self, t: &str) -> bool {
        if self.target.impl_type.as_deref() == Some(t) {
            return !self.ty_homes.is_empty();
        }
        if t.len() < 2 || !is_upper(t) || t.contains("::") || STD_TYPES.contains(&t) || t == "Self"
        {
            return false;
        }
        self.homes_of(t).len() > 1
    }

    /// A bare type name from a declared return type: tagged unknown when the
    /// target's type name is shared.
    fn tag_bare(&self, t: String) -> String {
        if !self.ty_homes.is_empty() && self.target.impl_type.as_deref() == Some(t.as_str()) {
            format!("{t}{TAG}?")
        } else {
            t
        }
    }

    /// The `impl` type of the block around `n`, tagged.
    fn impl_tag(&self, f: &File, n: Node) -> Option<String> {
        let i = ancestor(n, &["impl_item", "trait_item"])?;
        if i.kind() == "trait_item" {
            return i
                .child_by_field_name("name")
                .map(|t| text(t, &f.src).to_string());
        }
        let ty = i.child_by_field_name("type")?;
        let mut w = impl_written(i, ty, &f.src);
        // `impl<T> From<T> for Mutex<T>` with the crate's own `Mutex`: Mutex, not T.
        let base = w.split('<').next().unwrap_or(w).trim();
        if base != w && WRAPPERS.contains(&base) && !(self.type_defs)(base).is_empty() {
            w = base;
        }
        Some(self.tag(f, &module_of(f, i), None, w))
    }

    /// `d` is the target, or a `#[cfg]` twin of it (same file, module and type).
    fn is_target(&self, d: &Def) -> bool {
        let t = self.target;
        d == t
            || (d.path == t.path
                && d.module == t.module
                && d.impl_type == t.impl_type
                && d.free == t.free
                && d.has_self == t.has_self)
    }
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
        bind(dir.path(), at, name, &lines, &|t: &str| {
            type_defs_in(files, t)
        })
        .unwrap()
        .lines
        .into_iter()
        .map(|l| (format!("{}:{}", l.path, l.line), l.verdict))
        .collect()
    }

    const CARGO: (&str, &str) = ("Cargo.toml", "[package]\nname = \"demo\"\n");

    /// What the index's type search gives `bind`: the lines that define a type `t`.
    fn type_defs_in(files: &[(&str, &str)], t: &str) -> Vec<(String, usize)> {
        let re = regex::Regex::new(&format!(
            r"\b(struct|enum|union|type|trait)\s+{}\b",
            regex::escape(t)
        ))
        .unwrap();
        let mut out = Vec::new();
        for (p, s) in files {
            for (i, l) in s.lines().enumerate() {
                if p.ends_with(".rs") && re.is_match(l) {
                    out.push((p.to_string(), i + 1));
                }
            }
        }
        out
    }

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
    fn chains_and_field_paths_bind_by_declared_types() {
        let src = "pub struct Builder;\npub struct Runtime;\npub struct Other;\nimpl Builder {\n    pub fn new_current_thread() -> Builder {\n        Builder\n    }\n    pub fn enable_all(&mut self) -> &mut Self {\n        self\n    }\n    pub fn build(&mut self) -> std::io::Result<Runtime> {\n        Ok(Runtime)\n    }\n}\nimpl Other {\n    pub fn build(&self) -> u8 {\n        1\n    }\n}\nstruct Handle {\n    shared: Shared,\n}\nimpl Handle {\n    fn go(&mut self) {\n        self.shared.builder.build();\n    }\n}\nfn rt() -> Runtime {\n    Builder::new_current_thread().enable_all().build().unwrap()\n}\nfn other(o: Other) {\n    o.build();\n    let s = std::sync::Arc::new(Builder);\n    s.clone().build();\n    let b = Builder::new_current_thread();\n    b.build();\n}\n";
        // `Shared` is declared in another file than its use.
        let files = [
            CARGO,
            ("src/lib.rs", src),
            (
                "src/shared.rs",
                "pub struct Shared {\n    pub builder: crate::Builder,\n}\n",
            ),
        ];
        let line = |needle: &str| src.lines().position(|l| l.contains(needle)).unwrap() + 1;
        let at = format!("src/lib.rs:{}", line("pub fn build(&mut self)"));
        let v = verdicts(&files, &at, "build");
        let get = |needle: &str| {
            let k = format!("src/lib.rs:{}", line(needle));
            v.iter().find(|(p, _)| *p == k).map(|x| x.1)
        };
        assert_eq!(
            get("self.shared.builder"),
            Some(Verdict::This),
            "field of a field"
        );
        assert_eq!(
            get("enable_all().build()"),
            Some(Verdict::This),
            "builder chain"
        );
        assert_eq!(get("o.build()"), Some(Verdict::Other), "another type");
        assert_eq!(
            get("s.clone().build()"),
            Some(Verdict::Possible),
            "Arc of unknown"
        );
        assert_eq!(
            get("b.build()"),
            Some(Verdict::This),
            "let from an associated fn"
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

    /// Two types named `Sender` (tokio's `broadcast` and `watch`, 2026-11-12): a
    /// receiver counts only when its written type path reaches the target's module.
    #[test]
    fn shared_type_names_resolve_by_module() {
        let files = [
            CARGO,
            ("src/lib.rs", "pub mod sync;\n"),
            (
                "src/sync/mod.rs",
                "pub mod broadcast;\npub mod watch;\npub use broadcast::Sender;\n",
            ),
            (
                "src/sync/broadcast.rs",
                "pub struct Sender<T> { t: T }\nimpl<T> Sender<T> {\n    pub fn new(t: T) -> Self { Sender { t } }\n    pub fn count(&self) -> usize { 0 }\n}\n#[cfg(test)]\nmod tests {\n    use super::*;\n    fn t() {\n        let s = Sender::<i32>::new(1);\n        s.count();\n    }\n}\n",
            ),
            (
                "src/sync/watch.rs",
                "pub struct Sender { n: u8 }\nimpl Sender {\n    pub fn count(&self) -> usize { 1 }\n    fn g(&self) -> usize { self.count() }\n}\n",
            ),
            (
                "src/user.rs",
                "use crate::sync::watch;\nuse crate::sync::Sender;\nfn a(w: &watch::Sender, b: &Sender<u8>) {\n    w.count();\n    b.count();\n}\nfn c(x: &crate::sync::broadcast::Sender<u8>) {\n    x.count();\n}\n",
            ),
        ];
        let v = verdicts(&files, "src/sync/broadcast.rs:4", "count");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|(_, x)| *x);
        assert_eq!(
            get("src/sync/broadcast.rs:11"),
            Some(Verdict::This),
            "glob in a test module"
        );
        assert_eq!(
            get("src/sync/watch.rs:4"),
            Some(Verdict::Other),
            "self of the other Sender"
        );
        assert_eq!(get("src/user.rs:4"), Some(Verdict::Other), "watch::Sender");
        assert_eq!(
            get("src/user.rs:5"),
            Some(Verdict::This),
            "a pub use re-export"
        );
        assert_eq!(get("src/user.rs:8"), Some(Verdict::This), "a full path");
    }

    /// `#[cfg(unix)]` and `#[cfg(not(unix))]` copies of one function are one item.
    #[test]
    fn cfg_twins_are_the_target() {
        let files = [
            CARGO,
            ("src/lib.rs", "mod util;\nmod net;\n"),
            (
                "src/util/mod.rs",
                "mod check;\npub(crate) use check::probe;\n",
            ),
            (
                "src/util/check.rs",
                "#[cfg(unix)]\npub(crate) fn probe() {}\n#[cfg(not(unix))]\npub(crate) fn probe() {}\n",
            ),
            (
                "src/net.rs",
                "use crate::util::probe;\nfn f() {\n    probe();\n}\n",
            ),
        ];
        let v = verdicts(&files, "src/util/check.rs:2", "probe");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|(_, x)| *x);
        assert_eq!(get("src/net.rs:3"), Some(Verdict::This));
        assert_eq!(
            get("src/util/check.rs:4"),
            Some(Verdict::Other),
            "the twin's own line"
        );
    }

    /// A `use` in a function body shadows the file's; `Arc::new(T::new())` is a `T`;
    /// a tuple `let` takes its element type from a free function's return type.
    #[test]
    fn block_uses_wrappers_and_tuple_lets() {
        let files = [
            CARGO,
            (
                "src/lib.rs",
                "pub mod a;\npub mod b;\npub mod ch;\nmod t;\n",
            ),
            (
                "src/a.rs",
                "pub struct Sem;\nimpl Sem {\n    pub fn new() -> Self { Sem }\n    pub fn get(&self) {}\n}\n",
            ),
            (
                "src/b.rs",
                "pub struct Sem;\nimpl Sem {\n    pub fn new() -> Self { Sem }\n    pub fn get(&self) {}\n}\n",
            ),
            (
                "src/ch.rs",
                "pub struct Tx;\npub struct Rx;\nimpl Tx {\n    pub fn get(&self) {}\n}\nimpl Rx {\n    pub fn get(&self) {}\n}\npub fn pair() -> (Tx, Rx) { (Tx, Rx) }\n#[cfg(test)]\nmod tests {\n    use super::*;\n    fn t() {\n        let (tx, rx) = pair();\n        tx.get();\n        rx.get();\n    }\n}\n",
            ),
            (
                "src/t.rs",
                "use std::sync::Arc;\nuse crate::b::Sem;\nfn f() {\n    use crate::a::*;\n    let s = Arc::new(Sem::new());\n    s.get();\n}\nfn g() {\n    let s = Arc::new(Sem::new());\n    s.get();\n}\n",
            ),
        ];
        let v = verdicts(&files, "src/a.rs:4", "get");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|(_, x)| *x);
        assert_eq!(
            get("src/t.rs:6"),
            Some(Verdict::This),
            "the block's glob use"
        );
        assert_eq!(get("src/t.rs:10"), Some(Verdict::Other), "the file's use");
        let v = verdicts(&files, "src/ch.rs:4", "get");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|(_, x)| *x);
        assert_eq!(get("src/ch.rs:15"), Some(Verdict::This), "tuple element 0");
        assert_eq!(get("src/ch.rs:16"), Some(Verdict::Other), "tuple element 1");
    }

    /// `T::open(..).context(..)?` is a `T`: `Result` is already unwrapped.
    #[test]
    fn result_adapters_keep_the_type() {
        let files = [
            CARGO,
            ("src/lib.rs", "mod r;\nmod d;\nmod u;\n"),
            (
                "src/r.rs",
                "pub struct R;\nimpl R {\n    pub fn open(p: &str) -> anyhow::Result<Self> { Ok(R) }\n    pub fn get(&self) {}\n}\n",
            ),
            (
                "src/d.rs",
                "pub struct D;\nimpl D {\n    pub fn get(&self) {}\n}\n",
            ),
            (
                "src/u.rs",
                "use anyhow::Context;\nuse crate::r::R;\nfn f() -> anyhow::Result<()> {\n    let r =\n        R::open(\"x\").context(\"y\")?;\n    r.get();\n    Ok(())\n}\n",
            ),
        ];
        let v = verdicts(&files, "src/r.rs:4", "get");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|(_, x)| *x);
        assert_eq!(get("src/u.rs:6"), Some(Verdict::This));
    }

    /// `type L = List<u8>` is a `List`; `let (_, b) = q::pair()` reads `q`'s fn.
    #[test]
    fn aliases_and_tuple_lets_from_other_modules() {
        let files = [
            CARGO,
            ("src/lib.rs", "mod list;\nmod q;\nmod u;\n"),
            (
                "src/list.rs",
                "pub struct List<T> { t: T }\nimpl<T> List<T> {\n    pub fn is_empty(&self) -> bool { true }\n}\n",
            ),
            (
                "src/q.rs",
                "pub struct A;\npub struct B;\nimpl B {\n    pub fn is_empty(&self) -> bool { true }\n}\npub fn pair() -> (A, crate::list::List<u8>) { todo!() }\n",
            ),
            (
                "src/u.rs",
                "use crate::list::List;\nuse crate::q;\ntype Slots = List<u8>;\nstruct W { slot: [Slots; 4] }\nimpl W {\n    fn f(&self) -> bool { self.slot[0].is_empty() }\n}\nfn g() {\n    let (_, b) = q::pair();\n    b.is_empty();\n}\n",
            ),
        ];
        let v = verdicts(&files, "src/list.rs:3", "is_empty");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|(_, x)| *x);
        assert_eq!(get("src/u.rs:6"), Some(Verdict::This), "an alias");
        assert_eq!(
            get("src/u.rs:10"),
            Some(Verdict::This),
            "a tuple from q::pair"
        );
    }

    /// `h.inner` where two `Handle` structs both have `inner`: the receiver's home picks.
    #[test]
    fn fields_of_shared_type_names_follow_the_home() {
        let files = [
            CARGO,
            ("src/lib.rs", "pub mod rt;\npub mod io;\nmod u;\n"),
            (
                "src/rt.rs",
                "pub struct Handle { pub inner: Core }\npub struct Core;\nimpl Core {\n    pub fn driver(&self) {}\n}\n",
            ),
            (
                "src/io.rs",
                "pub struct Handle { pub inner: Other }\npub struct Other;\nimpl Other {\n    pub fn driver(&self) {}\n}\n",
            ),
            (
                "src/u.rs",
                "use crate::rt::Handle;\nfn f(h: &Handle, g: &crate::io::Handle) {\n    h.inner.driver();\n    g.inner.driver();\n}\n",
            ),
        ];
        let v = verdicts(&files, "src/rt.rs:4", "driver");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|(_, x)| *x);
        assert_eq!(get("src/u.rs:3"), Some(Verdict::This));
        assert_eq!(get("src/u.rs:4"), Some(Verdict::Other));
    }

    #[test]
    fn self_in_a_trait_impl_of_another_type() {
        // tokio `impl<T> From<T> for Mutex<T> { Self::new(s) }` with the target
        // `UnsafeCell::new`; `Mutex` is a shared name (loom has one too).
        let files = [
            CARGO,
            ("src/lib.rs", "mod cell;\nmod mutex;\nmod loom;\n"),
            (
                "src/cell.rs",
                "pub struct UnsafeCell<T>(T);\nimpl<T> UnsafeCell<T> {\n    pub const fn new(d: T) -> UnsafeCell<T> {\n        UnsafeCell(d)\n    }\n    pub fn with_mut(&self) {}\n}\n",
            ),
            (
                "src/mutex.rs",
                "use crate::cell::UnsafeCell;\npub struct Mutex<T: ?Sized> {\n    c: UnsafeCell<T>,\n}\nimpl<T: ?Sized> Mutex<T> {\n    pub fn new(t: T) -> Self\n    where\n        T: Sized,\n    {\n        Mutex { c: UnsafeCell::new(t) }\n    }\n}\nimpl<T> From<T> for Mutex<T> {\n    fn from(s: T) -> Self {\n        Self::new(s)\n    }\n}\nimpl<T> Mutex<T> {\n    fn put(&self) {\n        self.c.with_mut();\n    }\n}\n",
            ),
            (
                "src/loom.rs",
                "pub struct Mutex<T>(T);\nimpl<T> Mutex<T> {\n    pub fn new(t: T) -> Self {\n        Mutex(t)\n    }\n}\n",
            ),
        ];
        let v = verdicts(&files, "src/cell.rs:3", "new");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|x| x.1);
        assert_eq!(get("src/mutex.rs:10"), Some(Verdict::This));
        assert_eq!(
            get("src/mutex.rs:15"),
            Some(Verdict::Other),
            "Self is Mutex"
        );
        let v = verdicts(&files, "src/cell.rs:6", "with_mut");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|x| x.1);
        assert_eq!(
            get("src/mutex.rs:20"),
            Some(Verdict::This),
            "c: UnsafeCell<T>"
        );
    }

    #[test]
    fn labelled_macro_arms_keep_the_impl() {
        // tokio metrics/batch.rs: `cfg_metrics_variant! { stable: {..}, unstable: {..} }`
        // inside `impl MetricsBatch` hid the impl from the parse.
        let files = [
            CARGO,
            ("src/lib.rs", "mod batch;\nmod stats;\n"),
            (
                "src/batch.rs",
                "pub struct Batch;\nimpl Batch {\n    cfg_variant! {\n        stable: {\n            fn park(&mut self) {}\n        },\n        unstable: {\n            fn park(&mut self) {}\n        }\n    }\n    pub fn end(&mut self) {}\n}\n",
            ),
            (
                "src/stats.rs",
                "use crate::batch::Batch;\npub struct Stats {\n    batch: Batch,\n}\nimpl Stats {\n    pub fn end(&mut self) {\n        self.batch.end();\n    }\n}\n",
            ),
        ];
        let v = verdicts(&files, "src/batch.rs:11", "end");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|x| x.1);
        assert_eq!(get("src/stats.rs:7"), Some(Verdict::This));
        assert_eq!(get("src/stats.rs:6"), Some(Verdict::Other));
        let v = verdicts(&files, "src/batch.rs:5", "park");
        assert!(v.iter().all(|(_, x)| *x != Verdict::This), "{v:?}");
    }

    #[test]
    fn closure_parameters_carry_types() {
        // tokio task/local.rs: `CURRENT.with(|LocalData { ctx, .. }| ctx.replace(..))`.
        let files = [
            CARGO,
            (
                "src/lib.rs",
                "pub struct Cell;\nimpl Cell {\n    pub fn replace(&self) {}\n}\npub struct Other;\nimpl Other {\n    pub fn replace(&self) {}\n}\nstruct Data {\n    ctx: Cell,\n    n: Other,\n}\nfn with<F: Fn(&Data)>(f: F) {}\nfn a() {\n    with(|Data { ctx, .. }| ctx.replace());\n    with(|Data { n: ctx, .. }| ctx.replace());\n    with(|x: &Cell| x.replace());\n    with(|Data { ctx, .. }| {\n        let ctx = Other {};\n        ctx.replace()\n    });\n}\n",
            ),
        ];
        let v = verdicts(&files, "src/lib.rs:3", "replace");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|x| x.1);
        assert_eq!(get("src/lib.rs:15"), Some(Verdict::This), "field ctx: Cell");
        assert_eq!(get("src/lib.rs:16"), Some(Verdict::Other), "field n: Other");
        assert_eq!(get("src/lib.rs:17"), Some(Verdict::This), "typed parameter");
        assert_eq!(
            get("src/lib.rs:20"),
            Some(Verdict::Other),
            "the let shadows it"
        );
    }

    #[test]
    fn closure_parameters_take_the_callee_bound() {
        // tokio task/state.rs: `self.fetch_update_action(|mut next| { next.ref_dec(); .. })`
        // with `fn fetch_update_action<F, T>(&self, f: F) where F: FnMut(Snapshot) -> ..`.
        let files = [
            CARGO,
            (
                "src/lib.rs",
                "pub struct Cell;\nimpl Cell {\n    pub fn replace(&self) {}\n}\npub struct Other;\nimpl Other {\n    pub fn replace(&self) {}\n}\nstruct Holder;\nimpl Holder {\n    fn fetch<F, T>(&self, f: F) -> T\n    where\n        F: FnMut(Cell) -> (T, Option<Cell>),\n    {\n        todo!()\n    }\n    fn each(&self, f: impl Fn(u8, &Other)) {}\n}\nfn a(h: Holder) {\n    h.fetch(|mut c| (c.replace(), None));\n    h.each(|_, o| o.replace());\n}\n",
            ),
        ];
        let v = verdicts(&files, "src/lib.rs:3", "replace");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|x| x.1);
        assert_eq!(get("src/lib.rs:20"), Some(Verdict::This), "F: FnMut(Cell)");
        assert_eq!(
            get("src/lib.rs:21"),
            Some(Verdict::Other),
            "impl Fn(u8, &Other)"
        );
    }

    #[test]
    fn match_and_if_let_bindings() {
        let files = [
            CARGO,
            (
                "src/lib.rs",
                "pub struct Cell;\nimpl Cell {\n    pub fn fire(&self) {}\n}\npub struct Other;\nimpl Other {\n    pub fn fire(&self) {}\n}\nenum Reaper {\n    Signal(Cell),\n    Pidfd(u8, Other),\n}\nstruct Wheel;\nimpl Wheel {\n    fn poll(&self) -> Option<Cell> {\n        None\n    }\n}\nimpl Reaper {\n    fn go(&self, w: Wheel) {\n        match self {\n            Self::Signal(r) => r.fire(),\n            Reaper::Pidfd(_, r) => r.fire(),\n        }\n        while let Some(e) = w.poll() {\n            e.fire();\n        }\n        if let Some(e) = w.poll() {\n            let e = Other;\n            e.fire();\n        }\n    }\n}\nfn g(w: Wheel) {\n    let e = w.poll();\n    match e {\n        Some(e) => e.fire(),\n        None => {}\n    }\n}\n",
            ),
        ];
        let v = verdicts(&files, "src/lib.rs:3", "fire");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|x| x.1);
        assert_eq!(
            get("src/lib.rs:22"),
            Some(Verdict::This),
            "Self::Signal(Cell)"
        );
        assert_eq!(
            get("src/lib.rs:23"),
            Some(Verdict::Other),
            "Reaper::Pidfd(_, Other)"
        );
        assert_eq!(
            get("src/lib.rs:26"),
            Some(Verdict::This),
            "while let Some(Cell)"
        );
        assert_eq!(
            get("src/lib.rs:30"),
            Some(Verdict::Possible),
            "the let shadows it"
        );
        assert_eq!(
            get("src/lib.rs:37"),
            Some(Verdict::This),
            "match e {{ Some(e) => .. }}"
        );
    }

    #[test]
    fn identical_aliases_in_two_modules() {
        // tokio: `type WaitList = LinkedList<..>` in sync/notify.rs and io/scheduled_io.rs.
        let files = [
            CARGO,
            ("src/lib.rs", "mod list;\nmod a;\nmod b;\n"),
            (
                "src/list.rs",
                "pub struct LinkedList<T>(T);\nimpl<T> LinkedList<T> {\n    pub fn is_empty(&self) -> bool {\n        true\n    }\n}\n",
            ),
            (
                "src/a.rs",
                "use crate::list::LinkedList;\ntype WaitList = LinkedList<u8>;\nfn f(w: &WaitList) -> bool {\n    w.is_empty()\n}\n",
            ),
            (
                "src/b.rs",
                "use crate::list::LinkedList;\ntype WaitList = LinkedList<u8>;\npub struct Q;\nimpl Q {\n    pub fn is_empty(&self) -> bool {\n        false\n    }\n}\n",
            ),
        ];
        let v = verdicts(&files, "src/list.rs:3", "is_empty");
        let get = |k: &str| v.iter().find(|(p, _)| p == k).map(|x| x.1);
        assert_eq!(get("src/a.rs:4"), Some(Verdict::This));
    }

    #[test]
    fn at_must_name_a_function() {
        let files = [CARGO, ("src/lib.rs", "fn a() {}\n")];
        let dir = tree(&files);
        let err = bind(dir.path(), "src/lib.rs:1", "b", &[], &|_: &str| Vec::new())
            .err()
            .unwrap();
        assert!(err.to_string().contains("no function `b`"), "{err}");
        assert!(parse_at("src/lib.rs").is_err());
    }
}
