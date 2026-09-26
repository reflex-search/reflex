//! The API reference: one page per public module, one per public type.
//!
//! ```text
//! Reference
//! └─ my_lib                 module page: docs, submodules, types, functions, constants
//!    ├─ Engine              type page: definition, fields/variants, methods, trait impls
//!    └─ api                 module page
//!       └─ Thing            type page
//! ```
//! Doc comments keep their markdown. Rustdoc's hidden example lines are removed and
//! intra-doc links (`` [`Foo`] ``, `[Self::new]`, `[x](crate::a::B)`) become links to
//! the symbol's page, resolved from the item's scope like rustdoc does.

use super::{PageSpec, SiteBuilder};
use crate::parsers::api::{ApiItem, ApiKind, DocComment, Visibility};
use crate::pulse::extract::surface::{Package, Surface, SurfaceItem};
use crate::pulse::model::ids::slugify;
use crate::pulse::model::{
    AnchorId, Block, Inline, MarkdownOrigin, MarkdownText, NavNode, PageId, PageKind, ParamRow,
    SourceLoc, SymbolBlock, SymbolEntry, SymbolId, TabId, Target,
};
use std::collections::{BTreeMap, HashMap};
use std::sync::LazyLock;

use super::links::SYMBOL_SCHEME;

/// Longest signature kept on one line.
const SIGNATURE_WIDTH: usize = 90;

/// Tidy a whitespace-collapsed signature, and wrap a long `fn` one like rustfmt:
/// one parameter per line, `where` on its own line.
pub fn pretty_signature(sig: &str) -> String {
    let mut s = sig
        .replace("( ", "(")
        .replace(" )", ")")
        .replace(",)", ")")
        .replace("< ", "<")
        .replace(" >", ">");
    while s.contains(", )") {
        s = s.replace(", )", ")");
    }
    if s.chars().count() <= SIGNATURE_WIDTH {
        return s;
    }
    let Some(fn_at) = s.find("fn ") else {
        return s;
    };
    // The parameter list opens at the first `(` outside the generics after the name.
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut open = None;
    for (i, &b) in bytes.iter().enumerate().skip(fn_at) {
        match b {
            b'<' => depth += 1,
            b'>' if i > 0 && bytes[i - 1] != b'-' => depth -= 1,
            b'(' if depth == 0 => {
                open = Some(i);
                break;
            }
            _ => {}
        }
    }
    let Some(open) = open else {
        return s;
    };
    let mut depth = 0i32;
    let mut close = None;
    let mut parts = Vec::new();
    let mut last = open + 1;
    for (i, &b) in bytes.iter().enumerate().skip(open + 1) {
        match b {
            b'(' | b'[' | b'{' | b'<' => depth += 1,
            b'>' if bytes[i - 1] == b'-' => {}
            b')' if depth == 0 => {
                close = Some(i);
                break;
            }
            b')' | b']' | b'}' | b'>' => depth -= 1,
            b',' if depth == 0 => {
                parts.push(s[last..i].trim().to_string());
                last = i + 1;
            }
            _ => {}
        }
    }
    let Some(close) = close else {
        return s;
    };
    let tail = s[last..close].trim();
    if !tail.is_empty() {
        parts.push(tail.to_string());
    }
    let mut out = s[..=open].to_string();
    for p in &parts {
        out.push_str(&format!("\n    {p},"));
    }
    if !parts.is_empty() {
        out.push('\n');
    }
    let rest = &s[close..];
    match rest.split_once(" where ") {
        Some((before, clauses)) => {
            out.push_str(before);
            out.push_str("\nwhere\n");
            for c in clauses.split(", ").filter(|c| !c.is_empty()) {
                out.push_str(&format!("    {},\n", c.trim_end_matches(',')));
            }
            out.truncate(out.trim_end().len());
        }
        None => out.push_str(rest),
    }
    out
}

/// Which parts of the library surface to document.
#[derive(Debug, Clone, Default)]
pub struct ReferenceOptions {
    /// Leave the library reference out (a CLI-first project).
    pub skip_library: bool,
    /// Only modules whose path starts with one of these (`reflex::query`). Empty = all.
    pub include: Vec<String>,
}

fn sym_id(lang: &str, path: &str, kind: ApiKind) -> SymbolId {
    SymbolId(format!("{lang}:{path}#{}", kind.label().replace(' ', "-")))
}

fn anchor(kind: ApiKind, name: &str) -> AnchorId {
    AnchorId(format!(
        "{}.{}",
        slugify(kind.label()),
        name.to_ascii_lowercase()
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect::<String>()
    ))
}

fn module_page(path: &str) -> PageId {
    PageId(format!("docs/ref/mod/{path}"))
}

fn type_page(path: &str) -> PageId {
    PageId(format!("docs/ref/type/{path}"))
}

fn path_slug(path: &str, sep: &str) -> String {
    path.split(sep).map(slugify).collect::<Vec<_>>().join("/")
}

/// Build the library reference and return its nav nodes (one group per crate).
pub fn build(b: &mut SiteBuilder, api: &Surface, opts: &ReferenceOptions) -> Vec<NavNode> {
    if opts.skip_library {
        return Vec::new();
    }
    let mut nav = Vec::new();
    for k in api.libraries() {
        let included = |path: &str| {
            opts.include.is_empty()
                || opts.include.iter().any(|p| {
                    path == p
                        || path.starts_with(&format!("{p}::"))
                        || p.starts_with(&format!("{path}::"))
                })
        };
        let modules: Vec<usize> = (0..k.modules.len())
            .filter(|&i| k.modules[i].public && included(&k.modules[i].path))
            .collect();
        if modules.is_empty() {
            continue;
        }
        let index = SymbolIndex::build(k, &modules, b);
        let mut crate_nav = Vec::new();
        for &mi in &modules {
            let m = &k.modules[mi];
            let types: Vec<&SurfaceItem> = public_items(k, mi)
                .filter(|it| it.item.kind.is_type())
                .collect();
            let mut group = vec![NavNode::page(&module_page(&m.path))];
            for t in &types {
                let blocks = type_blocks(k, t, &index);
                let summary = t.item.doc.as_ref().map(|d| plain(&d.summary));
                b.add_page(PageSpec {
                    id: type_page(&t.path),
                    tab: TabId::Docs,
                    kind: PageKind::ReferenceType {
                        path: t.path.clone(),
                    },
                    title: t.item.name.clone(),
                    description: summary.filter(|s| !s.is_empty()),
                    slug: Some(format!("reference/{}", path_slug(&t.path, k.sep))),
                    badges: badges_for(&t.item),
                    blocks,
                });
                group.push(NavNode::page(&type_page(&t.path)));
            }
            let blocks = module_blocks(k, mi, &modules, &index);
            b.add_page(PageSpec {
                id: module_page(&m.path),
                tab: TabId::Docs,
                kind: PageKind::ReferenceModule {
                    path: m.path.clone(),
                },
                title: m.path.clone(),
                description: m
                    .doc
                    .as_ref()
                    .map(|d| plain(&d.summary))
                    .filter(|s| !s.is_empty()),
                slug: Some(format!("reference/{}", path_slug(&m.path, k.sep))),
                badges: vec!["module".into()],
                blocks,
            });
            if mi == 0 {
                crate_nav.extend(group);
            } else {
                crate_nav.push(NavNode::Group {
                    label: m
                        .path
                        .strip_prefix(&format!("{}{}", k.name, k.sep))
                        .unwrap_or(&m.path)
                        .to_string(),
                    collapsed: true,
                    children: group,
                });
            }
        }
        b.symbols.extend(index.entries);
        nav.push(NavNode::Group {
            label: k.name.clone(),
            collapsed: false,
            children: crate_nav,
        });
    }
    nav
}

fn public_items(k: &Package, mi: usize) -> impl Iterator<Item = &SurfaceItem> {
    k.modules[mi].items.iter().filter(|it| it.public)
}

/// Documented symbols of one crate, for the registry and for intra-doc links.
struct SymbolIndex {
    entries: BTreeMap<SymbolId, SymbolEntry>,
    by_path: HashMap<String, SymbolId>,
    by_name: HashMap<String, Vec<SymbolId>>,
    crate_name: String,
    lang: String,
    sep: &'static str,
}

impl SymbolIndex {
    fn build(k: &Package, modules: &[usize], _b: &SiteBuilder) -> Self {
        let mut idx = Self {
            entries: BTreeMap::new(),
            by_path: HashMap::new(),
            by_name: HashMap::new(),
            crate_name: k.name.clone(),
            lang: k.lang_id(),
            sep: k.sep,
        };
        for &mi in modules {
            let m = &k.modules[mi];
            idx.add(
                &m.path,
                &m.name,
                ApiKind::Module,
                module_page(&m.path),
                None,
            );
            for it in public_items(k, mi) {
                if it.item.kind.is_type() {
                    let page = type_page(&it.path);
                    idx.add(&it.path, &it.item.name, it.item.kind, page.clone(), None);
                    for mem in documented_members(&it.item) {
                        idx.add(
                            &k.join(&it.path, &mem.name),
                            &mem.name,
                            mem.kind,
                            page.clone(),
                            Some(anchor(mem.kind, &mem.name)),
                        );
                    }
                } else {
                    idx.add(
                        &it.path,
                        &it.item.name,
                        it.item.kind,
                        module_page(&m.path),
                        Some(anchor(it.item.kind, &it.item.name)),
                    );
                }
            }
        }
        idx
    }

    fn add(
        &mut self,
        path: &str,
        name: &str,
        kind: ApiKind,
        page: PageId,
        anchor: Option<AnchorId>,
    ) {
        let id = sym_id(&self.lang, path, kind);
        self.by_path
            .entry(path.to_string())
            .or_insert_with(|| id.clone());
        self.by_name
            .entry(name.to_string())
            .or_default()
            .push(id.clone());
        self.entries.insert(
            id,
            SymbolEntry {
                name: name.to_string(),
                path: path.to_string(),
                kind: kind.label().to_string(),
                page,
                anchor,
            },
        );
    }

    /// Resolve an intra-doc link written in `module` (and inside `self_type`, if any),
    /// the way rustdoc does for Rust; other languages use the same scopes.
    fn resolve(&self, link: &str, module: &str, self_type: Option<&str>) -> Option<&SymbolId> {
        let sep = self.sep;
        let link = link.trim_end_matches("()").trim_end_matches('!');
        let j = |a: &str, b: &str| format!("{a}{sep}{b}");
        let try_path = |p: &str| self.by_path.get(p);
        if sep == "::" {
            if let Some(rest) = link.strip_prefix("Self::") {
                return self_type.and_then(|t| try_path(&j(t, rest)));
            }
            if let Some(rest) = link.strip_prefix("crate::") {
                return try_path(&j(&self.crate_name, rest));
            }
            if let Some(rest) = link.strip_prefix("self::") {
                return try_path(&j(module, rest));
            }
            if let Some(rest) = link.strip_prefix("super::") {
                let parent = module.rsplit_once(sep).map(|(p, _)| p)?;
                return try_path(&j(parent, rest));
            }
        }
        if let Some(t) = self_type
            && let Some(id) = try_path(&j(t, link))
        {
            return Some(id);
        }
        try_path(&j(module, link))
            .or_else(|| try_path(link))
            .or_else(|| try_path(&j(&self.crate_name, link)))
            .or_else(|| {
                // A bare name that is unique in the package.
                let name = link.rsplit(sep).next().unwrap_or(link);
                match self.by_name.get(name).map(Vec::as_slice) {
                    Some([only]) => Some(only),
                    _ => None,
                }
            })
    }
}

/// Members documented on a type page: fields, variants, public methods, associated items.
fn documented_members(item: &ApiItem) -> impl Iterator<Item = &ApiItem> {
    documented_members_indexed(item).map(|(_, m)| m)
}

fn documented_members_indexed(item: &ApiItem) -> impl Iterator<Item = (usize, &ApiItem)> {
    item.members.iter().enumerate().filter(|(_, m)| {
        !m.test_only
            && !m.hidden
            && match m.kind {
                ApiKind::Field => {
                    m.visibility == Visibility::Public || m.visibility == Visibility::Inherited
                }
                ApiKind::Variant => true,
                _ => m.trait_impl.is_none() && m.visibility.is_public(),
            }
    })
}

fn plain(md: &str) -> String {
    md.replace(['`', '*'], "").replace("[", "").replace("]", "")
}

fn badges_for(item: &ApiItem) -> Vec<String> {
    let mut v = vec![item.kind.label().to_string()];
    if item.deprecated.is_some() {
        v.push("deprecated".into());
    }
    v
}

static INTRA_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(
        r"\[(`?)([A-Za-z_][\w:]*(?:\(\)|!)?)(`?)\](\(([A-Za-z_][\w]*(?:::[\w]+)*(?:\(\)|!)?)\))?",
    )
    .expect("valid regex")
});

/// Remove rustdoc hidden lines from rust fences, tag bare fences as `rust`, and turn
/// intra-doc links into `pulse-symbol:` links.
fn render_doc(
    doc: &DocComment,
    index: &SymbolIndex,
    module: &str,
    self_type: Option<&str>,
    file: &str,
) -> MarkdownText {
    let mut out = Vec::new();
    let mut fence: Option<bool> = None; // Some(is_rust)
    for line in doc.markdown.lines() {
        let t = line.trim_start();
        let is_fence = t.starts_with("```") || t.starts_with("~~~");
        match (&fence, is_fence) {
            (None, true) => {
                let info = t.trim_start_matches(['`', '~']).trim();
                // Rustdoc fences (bare, `rust`, `no_run`, …) hide `# ` lines; in other
                // languages a bare fence is code in the package's own language.
                let is_rust = index.lang == "rust";
                let rusty = is_rust
                    && (info.is_empty()
                        || info.split(',').all(|a| {
                            a.trim().starts_with("rust")
                                || matches!(
                                    a.trim(),
                                    "ignore" | "no_run" | "should_panic" | "compile_fail"
                                )
                                || a.trim().starts_with("edition")
                        }));
                fence = Some(rusty);
                out.push(if rusty {
                    "```rust".to_string()
                } else if info.is_empty() {
                    format!("```{}", index.lang)
                } else {
                    line.to_string()
                });
            }
            (Some(_), true) => {
                fence = None;
                out.push("```".to_string());
            }
            (Some(true), false) => {
                if t == "#" || t.starts_with("# ") {
                    continue;
                }
                out.push(if t.starts_with("##") {
                    line.replacen("##", "#", 1)
                } else {
                    line.to_string()
                });
            }
            (Some(false), false) => out.push(line.to_string()),
            (None, false) => {
                let rewritten = INTRA_RE.replace_all(line, |c: &regex::Captures| {
                    let whole = c.get(0).unwrap().as_str();
                    let (tick, text, dest) = (&c[1], &c[2], c.get(5).map(|m| m.as_str()));
                    // `[text](https://…)` never matches; a bare `[word]` needs a path shape.
                    let target = dest.unwrap_or(text);
                    if dest.is_none() && tick.is_empty() && !text.contains(index.sep) {
                        return whole.to_string();
                    }
                    match index.resolve(target, module, self_type) {
                        Some(id) => format!("[{tick}{text}{tick}]({SYMBOL_SCHEME}{})", id.0),
                        None if !tick.is_empty() => format!("`{text}`"),
                        None => whole.to_string(),
                    }
                });
                out.push(rewritten.into_owned());
            }
        }
    }
    MarkdownText {
        source: out.join("\n"),
        origin: MarkdownOrigin::Comment,
        from: Some(SourceLoc::lines(file, doc.start_line, doc.end_line)),
    }
}

fn symbol_block(
    item: &ApiItem,
    path: &str,
    file: &str,
    index: &SymbolIndex,
    module: &str,
    self_type: Option<&str>,
) -> Block {
    let sig = item.sig.as_ref();
    let mut badges = Vec::new();
    if let Some(s) = sig {
        for (on, label) in [
            (s.is_const, "const"),
            (s.is_async, "async"),
            (s.is_unsafe, "unsafe"),
        ] {
            if on {
                badges.push(label.to_string());
            }
        }
    }
    let deprecated = item.deprecated.as_ref().map(|d| {
        let mut s = String::from("Deprecated");
        if let Some(v) = &d.since {
            s.push_str(&format!(" since {v}"));
        }
        if let Some(n) = &d.note {
            s.push_str(&format!(": {n}"));
        }
        s
    });
    Block::Symbol {
        symbol: Box::new(SymbolBlock {
            id: sym_id(&index.lang, path, item.kind),
            anchor: anchor(item.kind, &item.name),
            kind: item.kind.label().to_string(),
            name: item.name.clone(),
            lang: index.lang.clone(),
            signature: pretty_signature(&item.signature),
            doc: item
                .doc
                .as_ref()
                .map(|d| render_doc(d, index, module, self_type, file)),
            params: sig
                .map(|s| {
                    s.params
                        .iter()
                        .map(|p| ParamRow {
                            name: p.name.clone(),
                            ty: p.ty.clone(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            returns: sig.and_then(|s| s.returns.clone()),
            deprecated,
            badges,
            source: SourceLoc::lines(file, item.start_line, item.end_line),
        }),
    }
}

fn summary_cell(
    doc: Option<&DocComment>,
    index: &SymbolIndex,
    module: &str,
    file: &str,
) -> Vec<Inline> {
    match doc {
        Some(d) if !d.summary.is_empty() => {
            let one = DocComment {
                markdown: d.summary.clone(),
                ..d.clone()
            };
            let md = render_doc(&one, index, module, None, file).source;
            vec![Inline::text(plain(&md))]
        }
        _ => vec![],
    }
}

fn module_blocks(k: &Package, mi: usize, included: &[usize], index: &SymbolIndex) -> Vec<Block> {
    let m = &k.modules[mi];
    let mut blocks = Vec::new();
    if let Some(d) = &m.doc {
        blocks.push(Block::Markdown {
            markdown: render_doc(d, index, &m.path, None, &m.file),
        });
    }
    let children: Vec<usize> = m
        .children
        .iter()
        .copied()
        .filter(|c| included.contains(c))
        .collect();
    if !children.is_empty() {
        blocks.push(Block::heading(2, "Modules"));
        blocks.push(Block::Table {
            columns: vec!["Module".into(), "Summary".into()],
            rows: children
                .iter()
                .map(|&c| {
                    let cm = &k.modules[c];
                    vec![
                        vec![Inline::code_link(
                            Target::page(module_page(&cm.path)),
                            &cm.name,
                        )],
                        summary_cell(cm.doc.as_ref(), index, &cm.path, &cm.file),
                    ]
                })
                .collect(),
        });
    }

    let items: Vec<&SurfaceItem> = public_items(k, mi).collect();
    let types: Vec<&&SurfaceItem> = items.iter().filter(|it| it.item.kind.is_type()).collect();
    if !types.is_empty() {
        blocks.push(Block::heading(2, "Types"));
        blocks.push(Block::Table {
            columns: vec!["Name".into(), "Kind".into(), "Summary".into()],
            rows: types
                .iter()
                .map(|it| {
                    vec![
                        vec![Inline::code_link(
                            Target::page(type_page(&it.path)),
                            &it.item.name,
                        )],
                        vec![Inline::text(it.item.kind.label())],
                        summary_cell(it.item.doc.as_ref(), index, &m.path, &it.file),
                    ]
                })
                .collect(),
        });
    }
    for (kind, title) in [
        (ApiKind::Function, "Functions"),
        (ApiKind::Macro, "Macros"),
        (ApiKind::Const, "Constants"),
        (ApiKind::Static, "Statics"),
        (ApiKind::TypeAlias, "Type aliases"),
    ] {
        let group: Vec<&&SurfaceItem> = items.iter().filter(|it| it.item.kind == kind).collect();
        if group.is_empty() {
            continue;
        }
        blocks.push(Block::heading(2, title));
        for it in group {
            blocks.push(symbol_block(
                &it.item, &it.path, &it.file, index, &m.path, None,
            ));
        }
    }
    if blocks.is_empty() {
        blocks.push(Block::text("This module has no public items."));
    }
    blocks
}

fn type_blocks(k: &Package, t: &SurfaceItem, index: &SymbolIndex) -> Vec<Block> {
    let module = t.path.rsplit_once(k.sep).map(|(m, _)| m).unwrap_or(&k.name);
    // The page is the type: its definition renders flat, not as a card under its own title.
    let mut blocks = vec![Block::Code {
        lang: k.lang_id(),
        code: pretty_signature(&t.item.signature),
        title: None,
    }];
    if let Some(d) = &t.item.deprecated {
        let mut msg = String::from("Deprecated");
        if let Some(v) = &d.since {
            msg.push_str(&format!(" since {v}"));
        }
        if let Some(n) = &d.note {
            msg.push_str(&format!(": {n}"));
        }
        blocks.push(Block::Callout {
            kind: crate::pulse::model::content::CalloutKind::Caution,
            title: Some("Deprecated".into()),
            body: vec![Block::text(msg)],
        });
    }
    if let Some(doc) = &t.item.doc {
        blocks.push(Block::Markdown {
            markdown: render_doc(doc, index, module, Some(&t.path), &t.file),
        });
    }
    let def = SourceLoc::lines(&t.file, t.item.start_line, t.item.end_line);
    blocks.push(Block::para(vec![
        Inline::text("Defined in "),
        Inline::code_link(Target::Source { loc: def.clone() }, def.label()),
        Inline::text("."),
    ]));
    if let Some(def) = &t.defined_at {
        blocks.push(Block::note(format!("Re-exported here; defined as {def}.")));
    }

    let members: Vec<&ApiItem> = documented_members(&t.item).collect();
    let data: Vec<&&ApiItem> = members
        .iter()
        .filter(|m| matches!(m.kind, ApiKind::Field | ApiKind::Variant))
        .collect();
    if !data.is_empty() {
        let title = if t.item.kind == ApiKind::Enum {
            "Variants"
        } else {
            "Fields"
        };
        blocks.push(Block::heading(2, title));
        for m in data {
            blocks.push(symbol_block(
                m,
                &k.join(&t.path, &m.name),
                &t.file,
                index,
                module,
                Some(&t.path),
            ));
        }
    }

    let methods: Vec<(usize, &ApiItem)> = documented_members_indexed(&t.item)
        .filter(|(_, m)| !matches!(m.kind, ApiKind::Field | ApiKind::Variant))
        .collect();
    if !methods.is_empty() {
        let title = if t.item.kind == ApiKind::Trait {
            "Trait items"
        } else {
            "Methods"
        };
        blocks.push(Block::heading(2, title));
        for (i, m) in methods {
            // Members from `impl` blocks may live in another file.
            let file = t.impl_files.get(&i).map(String::as_str).unwrap_or(&t.file);
            blocks.push(symbol_block(
                m,
                &k.join(&t.path, &m.name),
                file,
                index,
                module,
                Some(&t.path),
            ));
        }
    }

    let mut traits: Vec<&str> = t
        .item
        .members
        .iter()
        .filter_map(|m| m.trait_impl.as_deref())
        .collect();
    traits.sort_unstable();
    traits.dedup();
    if !traits.is_empty() {
        blocks.push(Block::heading(2, "Trait implementations"));
        blocks.push(Block::List {
            ordered: false,
            items: traits
                .iter()
                .map(|tr| {
                    let base = tr.split('<').next().unwrap_or(tr);
                    let code = format!("impl {tr} for {}", t.item.name);
                    match index.resolve(base, module, None) {
                        Some(id) => vec![Inline::code_link(
                            Target::Symbol { symbol: id.clone() },
                            code,
                        )],
                        None => vec![Inline::code(code)],
                    }
                })
                .collect(),
        });
    }
    blocks
}

#[cfg(test)]
mod signature_tests {
    use super::pretty_signature;

    #[test]
    fn short_signatures_are_tidied() {
        assert_eq!(pretty_signature("pub fn f( a: u8, )"), "pub fn f(a: u8)");
        assert_eq!(pretty_signature("pub struct S< T >"), "pub struct S<T>");
    }

    #[test]
    fn long_signatures_wrap_like_rustfmt() {
        let s = "pub fn search_with_metadata( &self, pattern: &str, filter: QueryFilter, ) -> Result<QueryResponse, Error> where F: Fn(u8) -> u8";
        assert_eq!(
            pretty_signature(s),
            "pub fn search_with_metadata(\n    &self,\n    pattern: &str,\n    filter: QueryFilter,\n) -> Result<QueryResponse, Error>\nwhere\n    F: Fn(u8) -> u8,"
        );
        let g = "pub fn run<T: Into<String>, F: Fn(T) -> T>(input: Vec<(T, u8)>, callback: F, extra_argument_name: HashMap<String, u8>) -> T";
        let out = pretty_signature(g);
        assert!(
            out.starts_with(
                "pub fn run<T: Into<String>, F: Fn(T) -> T>(\n    input: Vec<(T, u8)>,\n"
            ),
            "{out}"
        );
        assert!(out.ends_with(") -> T"), "{out}");
    }
}
