//! Rust API extraction by walking the syntax tree.
//!
//! Items are read in source order. Outer doc comments (`///`, `/** */`) and attributes
//! are sibling nodes before an item in tree-sitter-rust, so the walker collects them as
//! it goes and attaches them to the next item. Function bodies are never entered.

use super::{
    ApiFile, ApiItem, ApiKind, Deprecation, ModDecl, Param, ReExport, SigParts, Visibility,
    collapse_ws, doc,
};
use anyhow::{Context, Result};
use std::sync::LazyLock;
use tree_sitter::Node;

/// Longest signature kept, in characters.
const MAX_SIGNATURE: usize = 600;

pub fn extract(source: &str) -> Result<ApiFile> {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .context("loading the Rust grammar")?;
    let tree = parser.parse(source, None).context("parsing Rust")?;
    let root = tree.root_node();

    let mut file = ApiFile {
        module_doc: inner_doc(&root, source),
        ..ApiFile::default()
    };
    let mut w = Walker {
        src: source,
        mod_decls: &mut file.mod_decls,
        reexports: &mut file.reexports,
        top: true,
    };
    file.items = w.items(&root, &Ctx::default());
    Ok(file)
}

#[derive(Clone, Default)]
struct Ctx {
    test_only: bool,
    in_trait: bool,
    /// `(self type, trait)` when walking an impl body.
    impl_of: Option<(String, Option<String>)>,
}

struct Walker<'a, 'b> {
    src: &'a str,
    mod_decls: &'b mut Vec<ModDecl>,
    reexports: &'b mut Vec<ReExport>,
    /// Only file-level `mod`/`use` are recorded.
    top: bool,
}

/// Doc comment lines and attributes seen before the next item.
#[derive(Default)]
struct Pending {
    doc: Vec<String>,
    doc_start: u32,
    doc_end: u32,
    attrs: Vec<String>,
}

fn text<'s>(node: &Node, src: &'s str) -> &'s str {
    node.utf8_text(src.as_bytes()).unwrap_or("")
}

fn line(node: &Node) -> u32 {
    node.start_position().row as u32 + 1
}

fn end_line(node: &Node) -> u32 {
    node.end_position().row as u32 + 1
}

/// The text of an outer doc comment, or `None` for an ordinary comment.
fn doc_text(node: &Node, src: &str, outer: bool) -> Option<String> {
    let marker = if outer {
        "outer_doc_comment_marker"
    } else {
        "inner_doc_comment_marker"
    };
    let mut c = node.walk();
    let has_marker = node.children(&mut c).any(|ch| ch.kind() == marker);
    if !has_marker {
        return None;
    }
    // A bare `///` has a marker but no `doc` child: it is an empty line.
    let Some(body) = node.child_by_field_name("doc") else {
        return Some(String::new());
    };
    let raw = text(&body, src);
    if node.kind() == "block_comment" {
        // `/** a\n * b\n */` → strip the leading ` * ` of continuation lines.
        let lines: Vec<String> = raw
            .lines()
            .map(|l| {
                let t = l.trim_start();
                t.strip_prefix("* ")
                    .or_else(|| t.strip_prefix('*'))
                    .unwrap_or_else(|| l.strip_prefix(' ').unwrap_or(l))
                    .to_string()
            })
            .collect();
        Some(lines.join("\n"))
    } else {
        Some(raw.trim_end_matches(['\n', '\r']).to_string())
    }
}

fn inner_doc(root: &Node, src: &str) -> Option<super::DocComment> {
    let mut lines = Vec::new();
    let (mut start, mut end) = (0, 0);
    let mut c = root.walk();
    for child in root.children(&mut c) {
        match child.kind() {
            "line_comment" | "block_comment" => {
                if let Some(t) = doc_text(&child, src, false) {
                    if lines.is_empty() {
                        start = line(&child);
                    }
                    end = end_line(&child);
                    lines.extend(t.split('\n').map(str::to_string));
                }
            }
            "attribute_item" | "inner_attribute_item" => {}
            _ => break,
        }
    }
    doc::from_lines(&lines, start, end)
}

fn visibility(node: &Node, src: &str, ctx: &Ctx) -> Visibility {
    let mut c = node.walk();
    let vis = node
        .children(&mut c)
        .find(|ch| ch.kind() == "visibility_modifier");
    match vis {
        Some(v) => {
            let t = collapse_ws(text(&v, src));
            if t == "pub" {
                Visibility::Public
            } else {
                let scope = t
                    .trim_start_matches("pub")
                    .trim()
                    .trim_start_matches('(')
                    .trim_end_matches(')')
                    .trim()
                    .to_string();
                Visibility::Restricted(scope)
            }
        }
        None if ctx.in_trait => Visibility::Inherited,
        None if matches!(&ctx.impl_of, Some((_, Some(_)))) => Visibility::Inherited,
        None => Visibility::Private,
    }
}

/// Declaration text from the item start up to `body` (or the whole item).
fn signature(node: &Node, src: &str, cut_at_body: bool) -> String {
    let end = if cut_at_body {
        node.child_by_field_name("body")
            .map(|b| b.start_byte())
            .unwrap_or(node.end_byte())
    } else {
        node.end_byte()
    };
    let raw = &src[node.start_byte()..end];
    let mut s = collapse_ws(
        raw.trim_end()
            .trim_end_matches(|c: char| c == ';' || c == '{' || c == ',' || c.is_whitespace()),
    );
    if s.chars().count() > MAX_SIGNATURE {
        s = s.chars().take(MAX_SIGNATURE).collect::<String>() + " …";
    }
    s
}

static SINCE_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r#"since\s*=\s*"([^"]*)""#).expect("valid regex"));
static NOTE_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r#"note\s*=\s*"([^"]*)""#).expect("valid regex"));
static DOC_ATTR_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r#"^doc\s*=\s*"(.*)"$"#).expect("valid regex"));

fn deprecation(attrs: &[String]) -> Option<Deprecation> {
    let a = attrs.iter().find(|a| {
        a.as_str() == "deprecated" || a.starts_with("deprecated(") || a.starts_with("deprecated =")
    })?;
    let cap = |re: &regex::Regex| re.captures(a).map(|c| c[1].to_string());
    let note = cap(&NOTE_RE).or_else(|| {
        a.strip_prefix("deprecated =")
            .map(|s| s.trim().trim_matches('"').to_string())
    });
    Some(Deprecation {
        since: cap(&SINCE_RE),
        note,
    })
}

fn has_attr(attrs: &[String], want: &str) -> bool {
    attrs.iter().any(|a| {
        let a = a.replace(' ', "");
        a == want || a.starts_with(&format!("{want}("))
    })
}

fn is_cfg_test(attrs: &[String]) -> bool {
    attrs.iter().any(|a| {
        let a = a.replace(' ', "");
        a == "cfg(test)" || a == "test" || a.starts_with("cfg(all(test")
    })
}

fn sig_parts(node: &Node, src: &str) -> SigParts {
    let mut sig = SigParts::default();
    if let Some(tp) = node.child_by_field_name("type_parameters") {
        sig.generics = Some(collapse_ws(text(&tp, src)));
    }
    if let Some(ret) = node.child_by_field_name("return_type") {
        sig.returns = Some(collapse_ws(text(&ret, src)));
    }
    let mut c = node.walk();
    for ch in node.children(&mut c) {
        if ch.kind() == "function_modifiers" {
            let m = text(&ch, src);
            sig.is_async = m.split_whitespace().any(|w| w == "async");
            sig.is_unsafe = m.split_whitespace().any(|w| w == "unsafe");
            sig.is_const = m.split_whitespace().any(|w| w == "const");
        }
    }
    if let Some(params) = node.child_by_field_name("parameters") {
        let mut pc = params.walk();
        for p in params.named_children(&mut pc) {
            match p.kind() {
                "self_parameter" => sig.receiver = Some(collapse_ws(text(&p, src))),
                "parameter" => {
                    let name = p
                        .child_by_field_name("pattern")
                        .map(|n| collapse_ws(text(&n, src)))
                        .unwrap_or_default();
                    let ty = p
                        .child_by_field_name("type")
                        .map(|n| collapse_ws(text(&n, src)))
                        .unwrap_or_default();
                    sig.params.push(Param { name, ty });
                }
                _ => {}
            }
        }
    }
    sig
}

impl Walker<'_, '_> {
    fn items(&mut self, container: &Node, ctx: &Ctx) -> Vec<ApiItem> {
        let mut out = Vec::new();
        let mut pending = Pending::default();
        let mut c = container.walk();
        for child in container.named_children(&mut c) {
            match child.kind() {
                "line_comment" | "block_comment" => {
                    if let Some(t) = doc_text(&child, self.src, true) {
                        if pending.doc.is_empty() {
                            pending.doc_start = line(&child);
                        }
                        pending.doc_end = end_line(&child);
                        pending.doc.extend(t.split('\n').map(str::to_string));
                    }
                }
                "attribute_item" => {
                    if let Some(attr) = child.named_child(0) {
                        let a = collapse_ws(text(&attr, self.src));
                        if let Some(cap) = DOC_ATTR_RE.captures(&a) {
                            if pending.doc.is_empty() {
                                pending.doc_start = line(&child);
                            }
                            pending.doc_end = end_line(&child);
                            pending.doc.push(cap[1].replace("\\\"", "\""));
                        } else {
                            pending.attrs.push(a);
                        }
                    }
                }
                _ => {
                    let p = std::mem::take(&mut pending);
                    self.item(&child, ctx, p, &mut out);
                }
            }
        }
        out
    }

    fn base(&self, node: &Node, kind: ApiKind, name: String, ctx: &Ctx, p: Pending) -> ApiItem {
        let docs = doc::from_lines(&p.doc, p.doc_start, p.doc_end);
        let test_only = ctx.test_only || is_cfg_test(&p.attrs);
        ApiItem {
            name,
            kind,
            visibility: visibility(node, self.src, ctx),
            signature: String::new(),
            sig: None,
            doc: docs,
            deprecated: deprecation(&p.attrs),
            hidden: p.attrs.iter().any(|a| a.replace(' ', "") == "doc(hidden)"),
            test_only,
            start_line: line(node),
            end_line: end_line(node),
            trait_impl: ctx.impl_of.as_ref().and_then(|(_, t)| t.clone()),
            self_type: ctx.impl_of.as_ref().map(|(s, _)| s.clone()),
            members: Vec::new(),
            attrs: p.attrs,
        }
    }

    fn name_of(&self, node: &Node) -> String {
        node.child_by_field_name("name")
            .map(|n| text(&n, self.src).to_string())
            .unwrap_or_default()
    }

    fn item(&mut self, node: &Node, ctx: &Ctx, p: Pending, out: &mut Vec<ApiItem>) {
        let src = self.src;
        match node.kind() {
            "function_item" | "function_signature_item" => {
                let kind = if ctx.in_trait || ctx.impl_of.is_some() {
                    ApiKind::Method
                } else {
                    ApiKind::Function
                };
                let mut it = self.base(node, kind, self.name_of(node), ctx, p);
                it.signature = signature(node, src, true);
                it.sig = Some(sig_parts(node, src));
                if has_attr(&it.attrs, "test") {
                    it.test_only = true;
                }
                out.push(it);
            }
            "struct_item" | "union_item" => {
                let kind = if node.kind() == "struct_item" {
                    ApiKind::Struct
                } else {
                    ApiKind::Union
                };
                let mut it = self.base(node, kind, self.name_of(node), ctx, p);
                let body = node.child_by_field_name("body");
                let named_fields = body.is_some_and(|b| b.kind() == "field_declaration_list");
                it.signature = signature(node, src, named_fields);
                if let Some(b) = body {
                    it.members = self.fields(&b, &it_ctx(ctx, it.test_only));
                }
                if let Some(tp) = node.child_by_field_name("type_parameters") {
                    it.sig = Some(SigParts {
                        generics: Some(collapse_ws(text(&tp, src))),
                        ..SigParts::default()
                    });
                }
                out.push(it);
            }
            "enum_item" => {
                let mut it = self.base(node, ApiKind::Enum, self.name_of(node), ctx, p);
                it.signature = signature(node, src, true);
                if let Some(b) = node.child_by_field_name("body") {
                    it.members = self.variants(&b, &it_ctx(ctx, it.test_only));
                }
                out.push(it);
            }
            "trait_item" => {
                let mut it = self.base(node, ApiKind::Trait, self.name_of(node), ctx, p);
                it.signature = signature(node, src, true);
                if let Some(b) = node.child_by_field_name("body") {
                    let inner = Ctx {
                        in_trait: true,
                        impl_of: None,
                        test_only: it.test_only,
                    };
                    it.members = self.items(&b, &inner);
                }
                out.push(it);
            }
            "impl_item" => {
                let self_type = node
                    .child_by_field_name("type")
                    .map(|t| collapse_ws(text(&t, src)))
                    .unwrap_or_default();
                let trait_name = node
                    .child_by_field_name("trait")
                    .map(|t| collapse_ws(text(&t, src)));
                let inner = Ctx {
                    in_trait: false,
                    impl_of: Some((self_type, trait_name)),
                    test_only: ctx.test_only || is_cfg_test(&p.attrs),
                };
                if let Some(b) = node.child_by_field_name("body") {
                    let top = std::mem::replace(&mut self.top, false);
                    out.extend(self.items(&b, &inner));
                    self.top = top;
                }
            }
            "mod_item" => {
                let name = self.name_of(node);
                let path_attr = p.attrs.iter().find_map(|a| {
                    a.strip_prefix("path =")
                        .map(|v| v.trim().trim_matches('"').to_string())
                });
                let mut it = self.base(node, ApiKind::Module, name.clone(), ctx, p);
                let body = node.child_by_field_name("body");
                if self.top {
                    self.mod_decls.push(ModDecl {
                        name,
                        visibility: it.visibility.clone(),
                        inline: body.is_some(),
                        path_attr,
                        test_only: it.test_only,
                        line: it.start_line,
                    });
                }
                if let Some(b) = body {
                    it.signature = signature(node, src, true);
                    let top = std::mem::replace(&mut self.top, false);
                    it.members = self.items(&b, &it_ctx(ctx, it.test_only));
                    self.top = top;
                    out.push(it);
                }
            }
            "use_declaration" => {
                let vis = visibility(node, src, ctx);
                if self.top
                    && vis != Visibility::Private
                    && let Some(arg) = node.child_by_field_name("argument")
                {
                    let mut found = Vec::new();
                    flatten_use(&arg, src, "", &mut found);
                    for (path, name) in found {
                        self.reexports.push(ReExport {
                            path,
                            name,
                            visibility: vis.clone(),
                            line: line(node),
                        });
                    }
                }
            }
            "const_item" | "static_item" | "type_item" | "associated_type" => {
                let kind = match node.kind() {
                    "const_item" => ApiKind::Const,
                    "static_item" => ApiKind::Static,
                    _ if ctx.in_trait || ctx.impl_of.is_some() => ApiKind::AssociatedType,
                    _ => ApiKind::TypeAlias,
                };
                let mut it = self.base(node, kind, self.name_of(node), ctx, p);
                it.signature = signature(node, src, false);
                out.push(it);
            }
            "macro_definition" => {
                let exported = has_attr(&p.attrs, "macro_export");
                let mut it = self.base(node, ApiKind::Macro, self.name_of(node), ctx, p);
                it.visibility = if exported {
                    Visibility::Public
                } else {
                    Visibility::Private
                };
                it.signature = format!("macro_rules! {}", it.name);
                out.push(it);
            }
            _ => {}
        }
    }

    fn fields(&mut self, body: &Node, ctx: &Ctx) -> Vec<ApiItem> {
        let mut out = Vec::new();
        let mut pending = Pending::default();
        let mut index = 0;
        let mut c = body.walk();
        for child in body.named_children(&mut c) {
            match child.kind() {
                "line_comment" | "block_comment" => {
                    if let Some(t) = doc_text(&child, self.src, true) {
                        if pending.doc.is_empty() {
                            pending.doc_start = line(&child);
                        }
                        pending.doc_end = end_line(&child);
                        pending.doc.extend(t.split('\n').map(str::to_string));
                    }
                }
                "attribute_item" => {
                    if let Some(attr) = child.named_child(0) {
                        pending.attrs.push(collapse_ws(text(&attr, self.src)));
                    }
                }
                "field_declaration" => {
                    let p = std::mem::take(&mut pending);
                    let name = self.name_of(&child);
                    let mut it = self.base(&child, ApiKind::Field, name, ctx, p);
                    it.signature = signature(&child, self.src, false)
                        .trim_end_matches(',')
                        .to_string();
                    out.push(it);
                }
                "visibility_modifier" | "attribute" => {}
                _ if body.kind() == "ordered_field_declaration_list" => {
                    // Tuple struct: `pub struct Id(pub u32, String);` — types only.
                    if child.kind() == "visibility_modifier" {
                        continue;
                    }
                    let p = std::mem::take(&mut pending);
                    let mut it = self.base(&child, ApiKind::Field, index.to_string(), ctx, p);
                    it.visibility = Visibility::Inherited;
                    it.signature = collapse_ws(text(&child, self.src));
                    index += 1;
                    out.push(it);
                }
                _ => pending = Pending::default(),
            }
        }
        out
    }

    fn variants(&mut self, body: &Node, ctx: &Ctx) -> Vec<ApiItem> {
        let mut out = Vec::new();
        let mut pending = Pending::default();
        let mut c = body.walk();
        for child in body.named_children(&mut c) {
            match child.kind() {
                "line_comment" | "block_comment" => {
                    if let Some(t) = doc_text(&child, self.src, true) {
                        if pending.doc.is_empty() {
                            pending.doc_start = line(&child);
                        }
                        pending.doc_end = end_line(&child);
                        pending.doc.extend(t.split('\n').map(str::to_string));
                    }
                }
                "attribute_item" => {
                    if let Some(attr) = child.named_child(0) {
                        pending.attrs.push(collapse_ws(text(&attr, self.src)));
                    }
                }
                "enum_variant" => {
                    let p = std::mem::take(&mut pending);
                    let name = self.name_of(&child);
                    let mut it = self.base(&child, ApiKind::Variant, name, ctx, p);
                    it.visibility = Visibility::Inherited;
                    it.signature = signature(&child, self.src, false)
                        .trim_end_matches(',')
                        .to_string();
                    if let Some(b) = child.child_by_field_name("body")
                        && b.kind() == "field_declaration_list"
                    {
                        let inner = Ctx {
                            in_trait: true, // fields of a variant share its visibility
                            ..ctx.clone()
                        };
                        it.members = self.fields(&b, &inner);
                    }
                    out.push(it);
                }
                _ => pending = Pending::default(),
            }
        }
        out
    }
}

fn it_ctx(ctx: &Ctx, test_only: bool) -> Ctx {
    Ctx {
        test_only,
        in_trait: false,
        impl_of: None,
    }
    .with_parent(ctx)
}

impl Ctx {
    fn with_parent(mut self, parent: &Ctx) -> Self {
        self.test_only |= parent.test_only;
        self
    }
}

/// Flatten a use tree into `(full path, exported name)` pairs.
fn flatten_use(node: &Node, src: &str, prefix: &str, out: &mut Vec<(String, String)>) {
    let join = |p: &str, s: &str| {
        if p.is_empty() {
            s.to_string()
        } else {
            format!("{p}::{s}")
        }
    };
    match node.kind() {
        "identifier" | "scoped_identifier" | "self" | "crate" | "super" => {
            let path = join(prefix, &collapse_ws(text(node, src)));
            let name = path.rsplit("::").next().unwrap_or(&path).to_string();
            out.push((path, name));
        }
        "use_as_clause" => {
            let path = node
                .child_by_field_name("path")
                .map(|p| join(prefix, &collapse_ws(text(&p, src))))
                .unwrap_or_default();
            let alias = node
                .child_by_field_name("alias")
                .map(|a| text(&a, src).to_string())
                .unwrap_or_default();
            out.push((path, alias));
        }
        "use_wildcard" => {
            let base = node
                .named_child(0)
                .map(|p| join(prefix, &collapse_ws(text(&p, src))))
                .unwrap_or_else(|| prefix.to_string());
            out.push((join(&base, "*"), "*".into()));
        }
        "scoped_use_list" => {
            let base = node
                .child_by_field_name("path")
                .map(|p| join(prefix, &collapse_ws(text(&p, src))))
                .unwrap_or_else(|| prefix.to_string());
            if let Some(list) = node.child_by_field_name("list") {
                flatten_use(&list, src, &base, out);
            }
        }
        "use_list" => {
            let mut c = node.walk();
            for ch in node.named_children(&mut c) {
                flatten_use(&ch, src, prefix, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"//! Crate docs.
//! More.

/// Adds numbers.
///
/// # Examples
/// ```
/// # use x::y;
/// assert_eq!(add(1, 2), 3);
/// ```
#[inline]
#[deprecated(since = "1.2", note = "use sum")]
pub async unsafe fn add<T: Copy>(a: T, b: u32) -> Result<u8, E>
where
    T: Clone,
{
    a
}

/** Block doc
 * line two */
pub(crate) struct S<T> {
    /// field doc
    pub x: u32,
    y: T,
}
pub struct Id(pub u32, String);
/// An enum.
#[derive(Debug, Clone)]
pub enum E {
    /// variant
    A,
    B(u32),
    C { z: u8 },
}
impl<T> Tr for S<T> {
    fn m(&mut self) {}
    type Out = u8;
}
impl<T> S<T> {
    /// Make one.
    pub fn new() -> Self { todo!() }
    fn private(&self) {}
}
/// A trait.
pub trait Tr {
    type Out;
    /// Required.
    fn m(&mut self);
}
#[cfg(test)]
mod tests {
    fn t() {}
}
pub mod m;
#[path = "x.rs"]
mod n;
pub use crate::a::{B, C as D};
pub use self::m::*;
use std::fmt;
#[macro_export]
macro_rules! mac { () => {} }
#[doc(hidden)]
pub type Al = u8;
pub static ST: u8 = 1;
pub const CO: u8 = 1;
"#;

    fn find<'a>(items: &'a [ApiItem], name: &str) -> &'a ApiItem {
        items.iter().find(|i| i.name == name).unwrap_or_else(|| {
            panic!(
                "{name} not found in {:?}",
                items.iter().map(|i| &i.name).collect::<Vec<_>>()
            )
        })
    }

    #[test]
    fn functions_signatures_docs_attrs() {
        let f = extract(SAMPLE).unwrap();
        assert_eq!(
            f.module_doc.as_ref().unwrap().markdown,
            "Crate docs.\nMore."
        );
        let add = find(&f.items, "add");
        assert_eq!(add.kind, ApiKind::Function);
        assert_eq!(add.visibility, Visibility::Public);
        assert_eq!(
            add.signature,
            "pub async unsafe fn add<T: Copy>(a: T, b: u32) -> Result<u8, E> where T: Clone"
        );
        let sig = add.sig.as_ref().unwrap();
        assert!(sig.is_async && sig.is_unsafe);
        assert_eq!(sig.generics.as_deref(), Some("<T: Copy>"));
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.params[1].name, "b");
        assert_eq!(sig.params[1].ty, "u32");
        assert_eq!(sig.returns.as_deref(), Some("Result<u8, E>"));
        let d = add.doc.as_ref().unwrap();
        assert_eq!(d.summary, "Adds numbers.");
        assert_eq!(d.examples[0].code, "assert_eq!(add(1, 2), 3);");
        assert_eq!(d.start_line, 4);
        let dep = add.deprecated.as_ref().unwrap();
        assert_eq!(dep.since.as_deref(), Some("1.2"));
        assert_eq!(dep.note.as_deref(), Some("use sum"));
        assert!(add.attrs.contains(&"inline".to_string()));
    }

    #[test]
    fn types_fields_variants() {
        let f = extract(SAMPLE).unwrap();
        let s = find(&f.items, "S");
        assert_eq!(s.visibility, Visibility::Restricted("crate".into()));
        assert_eq!(s.signature, "pub(crate) struct S<T>");
        assert_eq!(s.doc.as_ref().unwrap().markdown, "Block doc\nline two");
        assert_eq!(s.members.len(), 2);
        assert_eq!(s.members[0].name, "x");
        assert_eq!(s.members[0].signature, "pub x: u32");
        assert_eq!(s.members[0].doc.as_ref().unwrap().summary, "field doc");
        assert_eq!(s.members[1].visibility, Visibility::Private);

        let id = find(&f.items, "Id");
        assert_eq!(id.signature, "pub struct Id(pub u32, String)");

        let e = find(&f.items, "E");
        assert!(e.attrs.contains(&"derive(Debug, Clone)".to_string()));
        let names: Vec<&str> = e.members.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["A", "B", "C"]);
        assert_eq!(e.members[0].doc.as_ref().unwrap().summary, "variant");
        assert_eq!(e.members[1].signature, "B(u32)");
        assert_eq!(e.members[2].members[0].name, "z");
    }

    #[test]
    fn impls_and_traits() {
        let f = extract(SAMPLE).unwrap();
        let methods: Vec<&ApiItem> = f
            .items
            .iter()
            .filter(|i| i.kind == ApiKind::Method)
            .collect();
        let new = methods.iter().find(|m| m.name == "new").unwrap();
        assert_eq!(new.self_type.as_deref(), Some("S<T>"));
        assert_eq!(new.trait_impl, None);
        assert_eq!(new.visibility, Visibility::Public);
        let private = methods.iter().find(|m| m.name == "private").unwrap();
        assert_eq!(private.visibility, Visibility::Private);
        let m = methods.iter().find(|m| m.name == "m").unwrap();
        assert_eq!(m.trait_impl.as_deref(), Some("Tr"));
        assert_eq!(m.visibility, Visibility::Inherited);
        assert_eq!(
            m.sig.as_ref().unwrap().receiver.as_deref(),
            Some("&mut self")
        );

        let tr = find(&f.items, "Tr");
        assert_eq!(tr.kind, ApiKind::Trait);
        let req = find(&tr.members, "m");
        assert_eq!(req.kind, ApiKind::Method);
        assert_eq!(req.visibility, Visibility::Inherited);
        assert_eq!(req.signature, "fn m(&mut self)");
        assert_eq!(find(&tr.members, "Out").kind, ApiKind::AssociatedType);
    }

    #[test]
    fn modules_reexports_misc() {
        let f = extract(SAMPLE).unwrap();
        let tests_mod = find(&f.items, "tests");
        assert!(tests_mod.test_only);
        assert!(tests_mod.members[0].test_only);

        let decls: Vec<(&str, bool, Option<&str>)> = f
            .mod_decls
            .iter()
            .map(|d| (d.name.as_str(), d.inline, d.path_attr.as_deref()))
            .collect();
        assert_eq!(
            decls,
            vec![
                ("tests", true, None),
                ("m", false, None),
                ("n", false, Some("x.rs"))
            ]
        );
        assert_eq!(f.mod_decls[1].visibility, Visibility::Public);

        let re: Vec<(&str, &str)> = f
            .reexports
            .iter()
            .map(|r| (r.path.as_str(), r.name.as_str()))
            .collect();
        assert_eq!(
            re,
            vec![
                ("crate::a::B", "B"),
                ("crate::a::C", "D"),
                ("self::m::*", "*")
            ]
        );

        assert_eq!(find(&f.items, "mac").visibility, Visibility::Public);
        let al = find(&f.items, "Al");
        assert!(al.hidden);
        assert_eq!(al.kind, ApiKind::TypeAlias);
        assert_eq!(find(&f.items, "CO").signature, "pub const CO: u8 = 1");
    }

    /// Extract every file under `src/` (run with `--ignored --nocapture`).
    #[test]
    #[ignore]
    fn extract_this_repository() {
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for e in std::fs::read_dir(dir).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push(p);
                }
            }
        }
        let mut files = Vec::new();
        walk(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
            &mut files,
        );
        let start = std::time::Instant::now();
        let (mut items, mut documented, mut public) = (0, 0, 0);
        fn count(items: &[ApiItem], n: &mut usize, d: &mut usize, p: &mut usize) {
            for i in items {
                *n += 1;
                if i.doc.is_some() {
                    *d += 1;
                }
                if i.visibility == Visibility::Public {
                    *p += 1;
                }
                count(&i.members, n, d, p);
            }
        }
        for f in &files {
            let src = std::fs::read_to_string(f).unwrap();
            let api = extract(&src).unwrap();
            count(&api.items, &mut items, &mut documented, &mut public);
        }
        println!(
            "{} files, {items} items ({documented} documented, {public} pub) in {:?}",
            files.len(),
            start.elapsed()
        );
        assert!(items > 1000);
    }
}
