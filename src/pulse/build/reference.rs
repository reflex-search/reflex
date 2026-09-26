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
use crate::pulse::extract::surface::{RustApi, RustCrate, RustItem};
use crate::pulse::model::ids::slugify;
use crate::pulse::model::{
    AnchorId, Block, Inline, MarkdownOrigin, MarkdownText, NavNode, PageId, PageKind, ParamRow,
    SourceLoc, SymbolBlock, SymbolEntry, SymbolId, TabId, Target,
};
use std::collections::{BTreeMap, HashMap};
use std::sync::LazyLock;

/// URL scheme for symbol links inside markdown; rewritten to routes once all pages exist.
pub const SYMBOL_SCHEME: &str = "pulse-symbol:";

/// Which parts of the library surface to document.
#[derive(Debug, Clone, Default)]
pub struct ReferenceOptions {
    /// Leave the library reference out (a CLI-first project).
    pub skip_library: bool,
    /// Only modules whose path starts with one of these (`reflex::query`). Empty = all.
    pub include: Vec<String>,
}

fn sym_id(path: &str, kind: ApiKind) -> SymbolId {
    SymbolId(format!("rust:{path}#{}", kind.label().replace(' ', "-")))
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

fn path_slug(path: &str) -> String {
    path.split("::").map(slugify).collect::<Vec<_>>().join("/")
}

/// Build the library reference and return its nav nodes (one group per crate).
pub fn build(b: &mut SiteBuilder, api: &RustApi, opts: &ReferenceOptions) -> Vec<NavNode> {
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
            let types: Vec<&RustItem> = public_items(k, mi)
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
                    slug: Some(format!("reference/{}", path_slug(&t.path))),
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
                slug: Some(format!("reference/{}", path_slug(&m.path))),
                badges: vec!["module".into()],
                blocks,
            });
            if mi == 0 {
                crate_nav.extend(group);
            } else {
                crate_nav.push(NavNode::Group {
                    label: m
                        .path
                        .strip_prefix(&format!("{}::", k.name))
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

fn public_items(k: &RustCrate, mi: usize) -> impl Iterator<Item = &RustItem> {
    k.modules[mi].items.iter().filter(|it| it.public)
}

/// Documented symbols of one crate, for the registry and for intra-doc links.
struct SymbolIndex {
    entries: BTreeMap<SymbolId, SymbolEntry>,
    by_path: HashMap<String, SymbolId>,
    by_name: HashMap<String, Vec<SymbolId>>,
    crate_name: String,
}

impl SymbolIndex {
    fn build(k: &RustCrate, modules: &[usize], _b: &SiteBuilder) -> Self {
        let mut idx = Self {
            entries: BTreeMap::new(),
            by_path: HashMap::new(),
            by_name: HashMap::new(),
            crate_name: k.name.clone(),
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
                            &format!("{}::{}", it.path, mem.name),
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
        let id = sym_id(path, kind);
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

    /// Resolve an intra-doc link written in `module` (and inside `self_type`, if any).
    fn resolve(&self, link: &str, module: &str, self_type: Option<&str>) -> Option<&SymbolId> {
        let link = link.trim_end_matches("()").trim_end_matches('!');
        let try_path = |p: &str| self.by_path.get(p);
        if let Some(rest) = link.strip_prefix("Self::") {
            return self_type.and_then(|t| try_path(&format!("{t}::{rest}")));
        }
        if let Some(rest) = link.strip_prefix("crate::") {
            return try_path(&format!("{}::{rest}", self.crate_name));
        }
        if let Some(rest) = link.strip_prefix("self::") {
            return try_path(&format!("{module}::{rest}"));
        }
        if let Some(rest) = link.strip_prefix("super::") {
            let parent = module.rsplit_once("::").map(|(p, _)| p)?;
            return try_path(&format!("{parent}::{rest}"));
        }
        if let Some(t) = self_type
            && let Some(id) = try_path(&format!("{t}::{link}"))
        {
            return Some(id);
        }
        try_path(&format!("{module}::{link}"))
            .or_else(|| try_path(link))
            .or_else(|| try_path(&format!("{}::{link}", self.crate_name)))
            .or_else(|| {
                // A bare name that is unique in the crate.
                let name = link.rsplit("::").next().unwrap_or(link);
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
                let rusty = info.is_empty()
                    || info.split(',').all(|a| {
                        a.trim().starts_with("rust")
                            || matches!(
                                a.trim(),
                                "ignore" | "no_run" | "should_panic" | "compile_fail"
                            )
                            || a.trim().starts_with("edition")
                    });
                fence = Some(rusty);
                out.push(if rusty {
                    "```rust".to_string()
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
                    if dest.is_none() && tick.is_empty() && !text.contains("::") {
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
            id: sym_id(path, item.kind),
            anchor: anchor(item.kind, &item.name),
            kind: item.kind.label().to_string(),
            name: item.name.clone(),
            lang: "rust".into(),
            signature: item.signature.clone(),
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

fn module_blocks(k: &RustCrate, mi: usize, included: &[usize], index: &SymbolIndex) -> Vec<Block> {
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

    let items: Vec<&RustItem> = public_items(k, mi).collect();
    let types: Vec<&&RustItem> = items.iter().filter(|it| it.item.kind.is_type()).collect();
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
        let group: Vec<&&RustItem> = items.iter().filter(|it| it.item.kind == kind).collect();
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

fn type_blocks(k: &RustCrate, t: &RustItem, index: &SymbolIndex) -> Vec<Block> {
    let module = t.path.rsplit_once("::").map(|(m, _)| m).unwrap_or(&k.name);
    let mut blocks = vec![symbol_block(
        &t.item,
        &t.path,
        &t.file,
        index,
        module,
        Some(&t.path),
    )];
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
                &format!("{}::{}", t.path, m.name),
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
                &format!("{}::{}", t.path, m.name),
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

/// Rewrite `pulse-symbol:` links in every markdown text to base-less routes.
pub fn resolve_markdown_links(site: &mut crate::pulse::model::Site) {
    let routes: HashMap<String, String> = {
        let linker = crate::pulse::model::Linker::new(site);
        site.symbols
            .keys()
            .filter_map(|id| {
                linker
                    .resolve(&Target::Symbol { symbol: id.clone() })
                    .map(|r| (id.0.clone(), r.href))
            })
            .collect()
    };
    static LINK: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"\(pulse-symbol:([^)\s]+)\)").expect("valid regex"));
    let fix = |md: &mut MarkdownText| {
        if !md.source.contains(SYMBOL_SCHEME) {
            return;
        }
        md.source = LINK
            .replace_all(&md.source, |c: &regex::Captures| match routes.get(&c[1]) {
                Some(href) => format!("({href})"),
                None => "()".to_string(),
            })
            .into_owned();
    };
    fn walk(blocks: &mut [Block], fix: &dyn Fn(&mut MarkdownText)) {
        for b in blocks {
            match b {
                Block::Markdown { markdown } => fix(markdown),
                Block::Symbol { symbol } => {
                    if let Some(d) = symbol.doc.as_mut() {
                        fix(d);
                    }
                }
                Block::Callout { body, .. } => walk(body, fix),
                Block::Narrative { fallback, .. } => walk(fallback, fix),
                _ => {}
            }
        }
    }
    for page in site.pages.values_mut() {
        walk(&mut page.blocks, &fix);
    }
}
