//! Python API extraction by walking the syntax tree.
//!
//! Statements are read in source order; function bodies are never entered. What each
//! statement becomes:
//!
//! | Source | Item |
//! | --- | --- |
//! | `def f(…)` / `async def f(…)` | `Function` (`Method` in a class) |
//! | `class C(Base)` | `Class`; `Enum` when a base is an `…Enum` / `…Flag` |
//! | `@property` / `@cached_property` | `Field` (setters and deleters are folded away) |
//! | `x: int = 0` in a class body, `X = 0` | `Field` (`Variant` in an enum) |
//! | `MAX = 3`, `x: int = 3` at module level | `Const` |
//! | `X: TypeAlias = …`, `type X = …`, `X = NewType(…)`, `X = Union[…]` | `TypeAlias` |
//! | `__all__ = [...]` (also `+=`, `.extend`, `.append`) | [`ApiFile::exports`]; `None` when any part is computed |
//! | `from .x import A as B`, `from pkg import *` | [`ReExport`] (`.x.A` → `B`) |
//!
//! Docstrings (the first statement of a module, class or function; a string right after
//! an assignment; `#:` comments before one) go through [`super::pydoc`]. Decorators are
//! kept in `attrs` without the `@`. `@deprecated(…)` (PEP 702, `typing_extensions`,
//! the `deprecation` package) and `.. deprecated::` set `deprecated`.
//!
//! Visibility: a leading `_` is `Private`; a dunder (`__init__`) is `Inherited`, and
//! dunders are `hidden` except the ones a caller uses directly: `__init__` (documented as
//! the constructor), `__call__`, and the context-manager, iterator, container and
//! awaitable protocols. `__repr__`, `__eq__`, `__hash__` and friends are noise on a
//! reference page. `@typing.overload` stubs are dropped when an implementation follows.
//!
//! Top-level `if`/`try` blocks are walked (first branch only), except
//! `if __name__ == "__main__":`, so version-gated and optional-import definitions count.

use super::collapse_ws;
use super::pydoc::{self, Entry, PyDoc};
use super::{ApiFile, ApiItem, ApiKind, Deprecation, Param, ReExport, SigParts, Visibility};
use crate::models::Language;
use anyhow::{Context, Result};
use regex::Regex;
use std::sync::LazyLock;
use tree_sitter::Node;

/// Longest signature kept, in characters.
const MAX_SIGNATURE: usize = 600;
/// Longest value shown in a constant's or field's signature.
const MAX_VALUE: usize = 60;

/// Dunder methods documented on a class page.
const SHOWN_DUNDERS: &[&str] = &[
    "__init__",
    "__call__",
    "__enter__",
    "__exit__",
    "__aenter__",
    "__aexit__",
    "__iter__",
    "__aiter__",
    "__next__",
    "__anext__",
    "__await__",
    "__getitem__",
    "__setitem__",
    "__delitem__",
    "__contains__",
    "__len__",
];

pub fn extract(source: &str) -> Result<ApiFile> {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&crate::parsers::ParserFactory::get_language_grammar(
            Language::Python,
        )?)
        .context("loading the Python grammar")?;
    let tree = parser.parse(source, None).context("parsing Python")?;
    let root = tree.root_node();

    let mut w = Walker {
        src: source,
        reexports: Vec::new(),
        exports: None,
        exports_dynamic: false,
    };
    let (doc, items) = w.body(&root, Scope::Module);
    Ok(ApiFile {
        module_doc: doc.and_then(|d| d.doc),
        items,
        mod_decls: Vec::new(),
        reexports: w.reexports,
        // A list built from other lists (`__all__ = base.__all__ + [...]`) is unknown:
        // the leading-underscore rule applies instead.
        exports: w.exports.filter(|_| !w.exports_dynamic),
        package: None,
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Scope {
    Module,
    Class { is_enum: bool },
}

struct Walker<'a> {
    src: &'a str,
    reexports: Vec<ReExport>,
    exports: Option<Vec<String>>,
    /// `__all__` has a part that is not a string literal.
    exports_dynamic: bool,
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

fn is_dunder(name: &str) -> bool {
    name.len() > 4 && name.starts_with("__") && name.ends_with("__")
}

fn visibility(name: &str) -> Visibility {
    if is_dunder(name) {
        Visibility::Inherited
    } else if name.starts_with('_') {
        Visibility::Private
    } else {
        Visibility::Public
    }
}

static UPPER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^_*[A-Z][A-Z0-9_]*$").expect("valid regex"));
static CAPWORDS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^_*[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*$").expect("valid regex"));

/// Last dotted segment of a decorator's callee: `functools.cached_property` → `cached_property`.
fn decorator_name(d: &str) -> &str {
    let callee = d.split('(').next().unwrap_or(d).trim();
    callee.rsplit('.').next().unwrap_or(callee)
}

fn has_decorator(attrs: &[String], names: &[&str]) -> bool {
    attrs.iter().any(|a| names.contains(&decorator_name(a)))
}

static DEP_POSITIONAL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"^[\w.]+\(\s*[rRuU]?(?:"([^"]*)"|'([^']*)')"#).expect("valid regex")
});
static DEP_SINCE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"\b(?:version|since|deprecated_in)\s*=\s*["']([^"']*)["']"#).expect("valid regex")
});
static DEP_NOTE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"\b(?:reason|details|message|msg)\s*=\s*["']([^"']*)["']"#).expect("valid regex")
});

/// `@deprecated("msg")`, `@typing_extensions.deprecated(…)`, `@deprecated(deprecated_in=…, details=…)`.
fn deprecation(attrs: &[String]) -> Option<Deprecation> {
    let d = attrs.iter().find(|a| decorator_name(a) == "deprecated")?;
    let cap = |re: &Regex| {
        re.captures(d)
            .and_then(|c| c.get(1).or_else(|| c.get(2)))
            .map(|m| m.as_str().to_string())
    };
    Some(Deprecation {
        since: cap(&DEP_SINCE_RE),
        note: cap(&DEP_POSITIONAL_RE).or_else(|| cap(&DEP_NOTE_RE)),
    })
}

/// `value` collapsed and cut to [`MAX_VALUE`] characters.
fn short_value(v: &str) -> String {
    let v = collapse_ws(v);
    if v.chars().count() > MAX_VALUE {
        v.chars().take(MAX_VALUE - 1).collect::<String>() + "…"
    } else {
        v
    }
}

/// Declaration text from `node` up to its body, without the trailing `:`.
fn signature(node: &Node, src: &str) -> String {
    let end = node
        .child_by_field_name("body")
        .map(|b| b.start_byte())
        .unwrap_or(node.end_byte());
    let raw = &src[node.start_byte()..end];
    let raw = raw.trim_end();
    let raw = raw.strip_suffix(':').unwrap_or(raw);
    let mut s = collapse_ws(raw);
    if s.chars().count() > MAX_SIGNATURE {
        s = s.chars().take(MAX_SIGNATURE).collect::<String>() + " …";
    }
    s
}

/// The docstring of a block: its first statement, when that is a string.
fn docstring(block: &Node, src: &str) -> Option<PyDoc> {
    let mut c = block.walk();
    let first = block
        .named_children(&mut c)
        .find(|n| n.kind() != "comment")?;
    string_statement(&first, src)
}

/// A statement that is a bare string literal, parsed as a docstring.
fn string_statement(stmt: &Node, src: &str) -> Option<PyDoc> {
    if stmt.kind() != "expression_statement" || stmt.named_child_count() != 1 {
        return None;
    }
    let s = stmt.named_child(0)?;
    (s.kind() == "string").then(|| pydoc::parse(text(&s, src), line(&s), end_line(&s)))
}

/// String literals directly inside a list / tuple (and `+` of them). Returns `false`
/// when some part is not a literal (`other.__all__`, a name, a comprehension).
fn string_list(node: &Node, src: &str, out: &mut Vec<String>) -> bool {
    match node.kind() {
        "string" => match pydoc::literal_value(text(node, src)) {
            Some(v) => {
                out.push(v);
                true
            }
            None => false,
        },
        "list" | "tuple" | "binary_operator" | "parenthesized_expression" | "argument_list" => {
            let mut c = node.walk();
            let mut complete = true;
            for ch in node.named_children(&mut c) {
                complete &= string_list(&ch, src, out);
            }
            complete
        }
        "comment" => true,
        _ => false,
    }
}

impl Walker<'_> {
    /// `__all__ = …` (`replace`) or `__all__ += …` / `.extend(…)` / `.append(…)`.
    fn add_exports(&mut self, value: &Node, replace: bool) {
        let mut names = Vec::new();
        let complete = string_list(value, self.src, &mut names);
        if replace {
            self.exports = Some(names);
            self.exports_dynamic = !complete;
        } else {
            self.exports.get_or_insert_default().extend(names);
            self.exports_dynamic |= !complete;
        }
    }

    /// A module or class body: its docstring and items.
    fn body(&mut self, block: &Node, scope: Scope) -> (Option<PyDoc>, Vec<ApiItem>) {
        let doc = docstring(block, self.src);
        let mut out = Vec::new();
        self.statements(block, scope, doc.is_some(), &mut out);
        drop_overloads(&mut out);
        (doc, out)
    }

    fn statements(&mut self, block: &Node, scope: Scope, skip_first: bool, out: &mut Vec<ApiItem>) {
        let src = self.src;
        let mut c = block.walk();
        let children: Vec<Node> = block.named_children(&mut c).collect();
        let mut skipped = !skip_first;
        // `#:` comment lines before the next assignment.
        let mut comment_doc: Vec<(String, u32)> = Vec::new();
        // Index in `out` of an item that a following string literal documents.
        let mut last_assign: Option<usize> = None;
        for child in &children {
            if child.kind() == "comment" {
                match text(child, src).strip_prefix("#:") {
                    Some(t) => comment_doc
                        .push((t.strip_prefix(' ').unwrap_or(t).to_string(), line(child))),
                    None => comment_doc.clear(),
                }
                continue;
            }
            if !skipped {
                skipped = true;
                continue;
            }
            let assign = last_assign.take();
            let comments = std::mem::take(&mut comment_doc);
            match child.kind() {
                "function_definition" => {
                    if let Some(it) = self.function(child, child, Vec::new(), scope) {
                        out.push(it);
                    }
                }
                "class_definition" => out.push(self.class(child, child, Vec::new(), scope)),
                "decorated_definition" => {
                    let mut dc = child.walk();
                    let attrs: Vec<String> = child
                        .named_children(&mut dc)
                        .filter(|n| n.kind() == "decorator")
                        .map(|n| collapse_ws(text(&n, src).trim_start_matches('@').trim()))
                        .collect();
                    match child.child_by_field_name("definition") {
                        Some(d) if d.kind() == "function_definition" => {
                            if let Some(it) = self.function(&d, child, attrs, scope) {
                                out.push(it);
                            }
                        }
                        Some(d) if d.kind() == "class_definition" => {
                            out.push(self.class(&d, child, attrs, scope));
                        }
                        _ => {}
                    }
                }
                "expression_statement" => {
                    let Some(inner) = child.named_child(0) else {
                        continue;
                    };
                    match inner.kind() {
                        "assignment" => {
                            if let Some(mut it) = self.assignment(&inner, child, scope) {
                                if !comments.is_empty() {
                                    let lines: Vec<&str> =
                                        comments.iter().map(|(t, _)| t.as_str()).collect();
                                    let d = pydoc::from_text(
                                        &lines.join("\n"),
                                        comments[0].1,
                                        comments[comments.len() - 1].1,
                                    );
                                    it.doc = d.doc;
                                }
                                out.push(it);
                                last_assign = Some(out.len() - 1);
                            }
                        }
                        "string" => {
                            // An attribute docstring: a string right after an assignment.
                            if let Some(i) = assign
                                && out[i].doc.is_none()
                                && let Some(d) = string_statement(child, src)
                            {
                                out[i].doc = d.doc;
                            }
                        }
                        "augmented_assignment" if scope == Scope::Module => {
                            let left = inner.child_by_field_name("left");
                            if left.is_some_and(|l| text(&l, src) == "__all__")
                                && let Some(r) = inner.child_by_field_name("right")
                            {
                                self.add_exports(&r, false);
                            }
                        }
                        "call" if scope == Scope::Module => {
                            let f = inner
                                .child_by_field_name("function")
                                .map(|f| text(&f, src))
                                .unwrap_or("");
                            if (f == "__all__.extend" || f == "__all__.append")
                                && let Some(args) = inner.child_by_field_name("arguments")
                            {
                                self.add_exports(&args, false);
                            }
                        }
                        _ => {}
                    }
                }
                "type_alias_statement" => {
                    let name = child
                        .child_by_field_name("left")
                        .map(|l| {
                            let t = text(&l, src);
                            t.split(['[', ' ']).next().unwrap_or(t).to_string()
                        })
                        .unwrap_or_default();
                    let mut it = base(child, ApiKind::TypeAlias, name);
                    it.signature = short_signature(text(child, src));
                    out.push(it);
                }
                "import_from_statement" if scope == Scope::Module => self.import_from(child),
                "if_statement" if scope == Scope::Module => {
                    let cond = child
                        .child_by_field_name("condition")
                        .map(|c| text(&c, src))
                        .unwrap_or("");
                    if !cond.contains("__name__")
                        && let Some(b) = child.child_by_field_name("consequence")
                    {
                        self.statements(&b, scope, false, out);
                    }
                }
                "try_statement" if scope == Scope::Module => {
                    if let Some(b) = child.child_by_field_name("body") {
                        self.statements(&b, scope, false, out);
                    }
                }
                _ => {}
            }
        }
    }

    fn import_from(&mut self, node: &Node) {
        let src = self.src;
        let module = node
            .child_by_field_name("module_name")
            .map(|m| collapse_ws(text(&m, src)).replace(' ', ""))
            .unwrap_or_default();
        if module == "__future__" {
            return;
        }
        let join = |name: &str| {
            if module.ends_with('.') {
                format!("{module}{name}")
            } else {
                format!("{module}.{name}")
            }
        };
        let mut c = node.walk();
        let mut push = |path: String, name: String| {
            self.reexports.push(ReExport {
                path,
                visibility: visibility(&name),
                name,
                line: line(node),
            });
        };
        for ch in node.children_by_field_name("name", &mut c) {
            match ch.kind() {
                "dotted_name" => {
                    let n = text(&ch, src).to_string();
                    push(join(&n), n);
                }
                "aliased_import" => {
                    let n = ch
                        .child_by_field_name("name")
                        .map(|n| text(&n, src).to_string())
                        .unwrap_or_default();
                    let alias = ch
                        .child_by_field_name("alias")
                        .map(|a| text(&a, src).to_string())
                        .unwrap_or_else(|| n.clone());
                    push(join(&n), alias);
                }
                _ => {}
            }
        }
        let mut c = node.walk();
        if node
            .named_children(&mut c)
            .any(|n| n.kind() == "wildcard_import")
        {
            push(join("*"), "*".into());
        }
    }

    fn function(
        &mut self,
        def: &Node,
        outer: &Node,
        attrs: Vec<String>,
        scope: Scope,
    ) -> Option<ApiItem> {
        let src = self.src;
        let name = def
            .child_by_field_name("name")
            .map(|n| text(&n, src).to_string())
            .unwrap_or_default();
        // `@x.setter` / `@x.deleter` belong to the property `x`.
        if attrs.iter().any(|a| {
            a.split('(')
                .next()
                .is_some_and(|c| c.ends_with(".setter") || c.ends_with(".deleter"))
        }) {
            return None;
        }
        let in_class = matches!(scope, Scope::Class { .. });
        let is_property =
            in_class && has_decorator(&attrs, &["property", "cached_property", "abstractproperty"]);
        let is_static = has_decorator(&attrs, &["staticmethod"]);
        let pydoc = def
            .child_by_field_name("body")
            .and_then(|b| docstring(&b, src))
            .unwrap_or_default();

        let mut sig = SigParts {
            is_async: def.child(0).is_some_and(|c| c.kind() == "async"),
            ..SigParts::default()
        };
        if let Some(tp) = def.child_by_field_name("type_parameters") {
            sig.generics = Some(collapse_ws(text(&tp, src)));
        }
        if let Some(r) = def.child_by_field_name("return_type") {
            sig.returns = Some(collapse_ws(text(&r, src)));
        }
        if let Some(ps) = def.child_by_field_name("parameters") {
            sig.params = params(&ps, src);
        }
        if in_class && !is_static && !sig.params.is_empty() && !sig.params[0].name.starts_with('*')
        {
            sig.receiver = Some(sig.params.remove(0).name);
        }
        fill_param_types(&mut sig.params, &pydoc.params);

        let kind = if is_property {
            ApiKind::Field
        } else if in_class {
            ApiKind::Method
        } else {
            ApiKind::Function
        };
        let mut it = base(outer, kind, name);
        it.hidden = is_dunder(&it.name) && !SHOWN_DUNDERS.contains(&it.name.as_str());
        it.deprecated = deprecation(&attrs).or(pydoc.deprecated);
        it.doc = pydoc.doc;
        it.attrs = attrs;
        if is_property {
            it.signature = match &sig.returns {
                Some(r) => format!("{}: {r}", it.name),
                None => it.name.clone(),
            };
        } else {
            it.signature = signature(def, src);
            it.sig = Some(sig);
        }
        Some(it)
    }

    fn class(&mut self, def: &Node, outer: &Node, attrs: Vec<String>, _scope: Scope) -> ApiItem {
        let src = self.src;
        let name = def
            .child_by_field_name("name")
            .map(|n| text(&n, src).to_string())
            .unwrap_or_default();
        let bases: Vec<String> = def
            .child_by_field_name("superclasses")
            .map(|s| {
                let mut c = s.walk();
                s.named_children(&mut c)
                    .filter(|n| n.kind() != "keyword_argument")
                    .map(|n| text(&n, src).to_string())
                    .collect()
            })
            .unwrap_or_default();
        let is_enum = bases.iter().any(|b| {
            let last = b.rsplit('.').next().unwrap_or(b);
            last.ends_with("Enum") || last.ends_with("Flag")
        });
        let (doc, mut members) = match def.child_by_field_name("body") {
            Some(b) => self.body(&b, Scope::Class { is_enum }),
            None => (None, Vec::new()),
        };
        let doc = doc.unwrap_or_default();
        fill_field_docs(&mut members, &doc.attributes);

        let kind = if is_enum {
            ApiKind::Enum
        } else {
            ApiKind::Class
        };
        let mut it = base(outer, kind, name);
        it.signature = signature(def, src);
        if let Some(tp) = def.child_by_field_name("type_parameters") {
            it.sig = Some(SigParts {
                generics: Some(collapse_ws(text(&tp, src))),
                ..SigParts::default()
            });
        }
        it.deprecated = deprecation(&attrs).or(doc.deprecated);
        it.doc = doc.doc;
        it.attrs = attrs;
        it.members = members;
        it
    }

    fn assignment(&mut self, node: &Node, stmt: &Node, scope: Scope) -> Option<ApiItem> {
        let src = self.src;
        let left = node.child_by_field_name("left")?;
        if left.kind() != "identifier" {
            return None;
        }
        let name = text(&left, src).to_string();
        let ty = node
            .child_by_field_name("type")
            .map(|t| collapse_ws(text(&t, src)));
        let right = node.child_by_field_name("right");
        if name == "__all__" && scope == Scope::Module {
            if let Some(r) = right {
                self.add_exports(&r, true);
            }
            return None;
        }
        if is_dunder(&name) {
            return None;
        }
        let rhs_call = right
            .filter(|r| r.kind() == "call")
            .and_then(|r| r.child_by_field_name("function"))
            .map(|f| {
                let t = text(&f, src);
                t.rsplit('.').next().unwrap_or(t).to_string()
            });
        let value = right.map(|r| short_value(text(&r, src)));
        let sig = match (&ty, &value) {
            (Some(t), Some(v)) => format!("{name}: {t} = {v}"),
            (Some(t), None) => format!("{name}: {t}"),
            (None, Some(v)) => format!("{name} = {v}"),
            (None, None) => name.clone(),
        };
        let is_alias_annotation = ty
            .as_deref()
            .is_some_and(|t| t == "TypeAlias" || t.ends_with(".TypeAlias"));
        let kind = match scope {
            _ if matches!(
                rhs_call.as_deref(),
                Some("TypeVar" | "ParamSpec" | "TypeVarTuple")
            ) =>
            {
                return None;
            }
            _ if is_alias_annotation || rhs_call.as_deref() == Some("NewType") => {
                ApiKind::TypeAlias
            }
            Scope::Class { is_enum: true } if ty.is_none() => {
                if name.starts_with('_') {
                    return None;
                }
                ApiKind::Variant
            }
            Scope::Class { .. } if ty.is_some() || UPPER_RE.is_match(&name) => ApiKind::Field,
            Scope::Class { .. } => return None,
            Scope::Module if ty.is_some() || UPPER_RE.is_match(&name) => ApiKind::Const,
            Scope::Module
                if CAPWORDS_RE.is_match(&name)
                    && right.is_some_and(|r| r.kind() == "subscript") =>
            {
                ApiKind::TypeAlias
            }
            Scope::Module => return None,
        };
        let mut it = base(stmt, kind, name);
        it.signature = sig;
        if kind == ApiKind::Variant {
            it.visibility = Visibility::Inherited;
        }
        Some(it)
    }
}

fn short_signature(s: &str) -> String {
    let s = collapse_ws(s);
    if s.chars().count() > MAX_SIGNATURE {
        s.chars().take(MAX_SIGNATURE).collect::<String>() + " …"
    } else {
        s
    }
}

fn base(node: &Node, kind: ApiKind, name: String) -> ApiItem {
    ApiItem {
        visibility: visibility(&name),
        name,
        kind,
        signature: String::new(),
        sig: None,
        doc: None,
        attrs: Vec::new(),
        deprecated: None,
        hidden: false,
        test_only: false,
        start_line: line(node),
        end_line: end_line(node),
        trait_impl: None,
        self_type: None,
        members: Vec::new(),
    }
}

/// Parameters as written; `*args` and `**kwargs` keep their stars, and the bare `*` and
/// `/` separators are left out.
fn params(list: &Node, src: &str) -> Vec<Param> {
    let mut out = Vec::new();
    let mut c = list.walk();
    for p in list.named_children(&mut c) {
        let field = |n: &str| p.child_by_field_name(n).map(|x| collapse_ws(text(&x, src)));
        let (name, ty) = match p.kind() {
            "identifier" => (text(&p, src).to_string(), None),
            "typed_parameter" => {
                let mut pc = p.walk();
                let name = p
                    .named_children(&mut pc)
                    .find(|n| n.kind() != "type")
                    .map(|n| text(&n, src).to_string())
                    .unwrap_or_default();
                (name, field("type"))
            }
            "default_parameter" | "typed_default_parameter" => {
                (field("name").unwrap_or_default(), field("type"))
            }
            "list_splat_pattern" | "dictionary_splat_pattern" => (text(&p, src).to_string(), None),
            _ => continue,
        };
        out.push(Param {
            name,
            ty: ty.unwrap_or_default(),
        });
    }
    out
}

/// Types from the docstring for parameters without an annotation.
fn fill_param_types(params: &mut [Param], documented: &[Entry]) {
    for p in params.iter_mut().filter(|p| p.ty.is_empty()) {
        let bare = p.name.trim_start_matches('*');
        let found = documented.iter().find(|e| {
            e.name
                .split(',')
                .any(|n| n.trim().trim_start_matches('*') == bare)
        });
        if let Some(ty) = found.and_then(|e| e.ty.as_deref()) {
            p.ty = ty.to_string();
        }
    }
}

/// Docs from the class docstring's `Attributes` for fields without their own.
fn fill_field_docs(members: &mut [ApiItem], attributes: &[Entry]) {
    for m in members
        .iter_mut()
        .filter(|m| matches!(m.kind, ApiKind::Field | ApiKind::Variant) && m.doc.is_none())
    {
        if let Some(e) = attributes.iter().find(|e| e.name == m.name)
            && !e.desc.is_empty()
        {
            m.doc = pydoc::from_text(&e.desc, m.start_line, m.start_line).doc;
        }
    }
}

/// Drop `@overload` stubs of a name that also has an implementation.
fn drop_overloads(items: &mut Vec<ApiItem>) {
    let is_overload = |it: &ApiItem| has_decorator(&it.attrs, &["overload"]);
    let implemented: Vec<String> = items
        .iter()
        .filter(|it| !is_overload(it))
        .map(|it| it.name.clone())
        .collect();
    let mut seen: Vec<String> = Vec::new();
    items.retain(|it| {
        if !is_overload(it) {
            return true;
        }
        // Stubs only (a `.pyi`-style module): keep the first.
        let keep = !implemented.contains(&it.name) && !seen.contains(&it.name);
        seen.push(it.name.clone());
        keep
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"#!/usr/bin/env python
"""Storage engines.

See :class:`Engine`.
"""
from __future__ import annotations

import os
from typing import TypeAlias, TypeVar, overload
from ._core import Engine, helper as public_helper
from .. import sibling
from pkg.sub import *

__all__ = ["Engine", "connect", "MAX_SIZE"]
__all__ += ["Color"]
__version__ = "1.0"

T = TypeVar("T")

#: Largest payload, in bytes.
MAX_SIZE = 1024
timeout: float = 2.5
"""Default timeout."""
logger = object()
Pair: TypeAlias = tuple[int, int]
Rows = list[tuple[str, int]]
type Grid = list[list[int]]


async def connect(url: str, *, retries: int = 3, **opts) -> "Engine":
    """Open a connection.

    Args:
        url: Where to connect.
        opts (dict): Driver options.
    """
    return Engine(url)


def _private():
    pass


@deprecated("use connect")
def open_db(path):
    """Old entry point.

    :param path: File path.
    :type path: str
    """


@dataclass(frozen=True)
class Config(Base, metaclass=Meta):
    """Settings.

    Attributes:
        name: Display name.
    """

    name: str
    port: int = 8080
    """Listening port."""
    DEFAULT = "x"
    _secret: str = ""
    plain = 3

    def __init__(self, name: str) -> None:
        """Make one."""

    def __repr__(self):
        return ""

    def __enter__(self):
        return self

    @property
    def url(self) -> str:
        """The URL."""
        return ""

    @url.setter
    def url(self, v):
        pass

    @classmethod
    def load(cls, path: str) -> "Config":
        ...

    @staticmethod
    def check(x):
        ...

    @overload
    def get(self, k: int) -> int: ...
    @overload
    def get(self, k: str) -> str: ...
    def get(self, k):
        """Get one."""

    class Inner:
        pass


class Color(Enum):
    """Colors."""

    RED = 1
    """Warm."""
    GREEN = 2

    def describe(self) -> str:
        return ""


if TYPE_CHECKING:
    def typed_only() -> None: ...

if __name__ == "__main__":
    def main(): ...

try:
    from fast import speedup
except ImportError:
    def speedup(): ...
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
    fn module_doc_exports_and_reexports() {
        let f = extract(SAMPLE).unwrap();
        let d = f.module_doc.as_ref().unwrap();
        assert_eq!(d.summary, "Storage engines.");
        assert_eq!(d.links, vec!["Engine"]);
        assert_eq!(
            f.exports.as_deref(),
            Some(&["Engine", "connect", "MAX_SIZE", "Color"].map(String::from)[..])
        );
        let re: Vec<(&str, &str)> = f
            .reexports
            .iter()
            .map(|r| (r.path.as_str(), r.name.as_str()))
            .collect();
        assert_eq!(
            re,
            vec![
                ("typing.TypeAlias", "TypeAlias"),
                ("typing.TypeVar", "TypeVar"),
                ("typing.overload", "overload"),
                ("._core.Engine", "Engine"),
                ("._core.helper", "public_helper"),
                ("..sibling", "sibling"),
                ("pkg.sub.*", "*"),
                ("fast.speedup", "speedup"),
            ]
        );
    }

    #[test]
    fn functions_signatures_and_docs() {
        let f = extract(SAMPLE).unwrap();
        let c = find(&f.items, "connect");
        assert_eq!(c.kind, ApiKind::Function);
        assert_eq!(c.visibility, Visibility::Public);
        assert_eq!(
            c.signature,
            "async def connect(url: str, *, retries: int = 3, **opts) -> \"Engine\""
        );
        let sig = c.sig.as_ref().unwrap();
        assert!(sig.is_async);
        let ps: Vec<(&str, &str)> = sig
            .params
            .iter()
            .map(|p| (p.name.as_str(), p.ty.as_str()))
            .collect();
        assert_eq!(
            ps,
            vec![("url", "str"), ("retries", "int"), ("**opts", "dict")],
            "unannotated **opts takes its type from the docstring"
        );
        assert_eq!(sig.returns.as_deref(), Some("\"Engine\""));
        assert!(
            c.doc
                .as_ref()
                .unwrap()
                .markdown
                .contains("- `url` — Where to connect.")
        );
        assert_eq!(find(&f.items, "_private").visibility, Visibility::Private);

        let old = find(&f.items, "open_db");
        assert_eq!(old.attrs, vec!["deprecated(\"use connect\")"]);
        assert_eq!(
            old.deprecated.as_ref().unwrap().note.as_deref(),
            Some("use connect")
        );
        assert_eq!(old.sig.as_ref().unwrap().params[0].ty, "str");
    }

    #[test]
    fn constants_and_aliases() {
        let f = extract(SAMPLE).unwrap();
        let names: Vec<(&str, ApiKind)> = f
            .items
            .iter()
            .filter(|i| matches!(i.kind, ApiKind::Const | ApiKind::TypeAlias))
            .map(|i| (i.name.as_str(), i.kind))
            .collect();
        assert_eq!(
            names,
            vec![
                ("MAX_SIZE", ApiKind::Const),
                ("timeout", ApiKind::Const),
                ("Pair", ApiKind::TypeAlias),
                ("Rows", ApiKind::TypeAlias),
                ("Grid", ApiKind::TypeAlias),
            ],
            "TypeVars, dunders and plain variables are not API"
        );
        let max = find(&f.items, "MAX_SIZE");
        assert_eq!(max.signature, "MAX_SIZE = 1024");
        assert_eq!(
            max.doc.as_ref().unwrap().summary,
            "Largest payload, in bytes."
        );
        let t = find(&f.items, "timeout");
        assert_eq!(t.signature, "timeout: float = 2.5");
        assert_eq!(t.doc.as_ref().unwrap().summary, "Default timeout.");
        assert_eq!(
            find(&f.items, "Grid").signature,
            "type Grid = list[list[int]]"
        );
    }

    #[test]
    fn classes_dataclasses_properties_and_methods() {
        let f = extract(SAMPLE).unwrap();
        let c = find(&f.items, "Config");
        assert_eq!(c.kind, ApiKind::Class);
        assert_eq!(c.signature, "class Config(Base, metaclass=Meta)");
        assert_eq!(c.attrs, vec!["dataclass(frozen=True)"]);
        assert_eq!(c.start_line, 53, "the item starts at its first decorator");
        let names: Vec<&str> = c.members.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "name",
                "port",
                "DEFAULT",
                "_secret",
                "__init__",
                "__repr__",
                "__enter__",
                "url",
                "load",
                "check",
                "get",
                "Inner"
            ],
            "unannotated lowercase attributes, setters and overload stubs are left out"
        );
        let name = find(&c.members, "name");
        assert_eq!(name.kind, ApiKind::Field);
        assert_eq!(name.signature, "name: str");
        assert_eq!(
            name.doc.as_ref().unwrap().summary,
            "Display name.",
            "from the class docstring's Attributes"
        );
        let port = find(&c.members, "port");
        assert_eq!(port.signature, "port: int = 8080");
        assert_eq!(port.doc.as_ref().unwrap().summary, "Listening port.");
        assert_eq!(find(&c.members, "_secret").visibility, Visibility::Private);

        let init = find(&c.members, "__init__");
        assert_eq!(init.kind, ApiKind::Method);
        assert_eq!(init.visibility, Visibility::Inherited);
        assert!(!init.hidden, "the constructor is documented");
        let isig = init.sig.as_ref().unwrap();
        assert_eq!(isig.receiver.as_deref(), Some("self"));
        assert_eq!(isig.params.len(), 1);
        assert!(find(&c.members, "__repr__").hidden);
        assert!(!find(&c.members, "__enter__").hidden);

        let url = find(&c.members, "url");
        assert_eq!(url.kind, ApiKind::Field);
        assert_eq!(url.signature, "url: str");
        assert_eq!(url.attrs, vec!["property"]);
        assert_eq!(url.doc.as_ref().unwrap().summary, "The URL.");

        let load = find(&c.members, "load");
        assert_eq!(load.attrs, vec!["classmethod"]);
        assert_eq!(load.sig.as_ref().unwrap().receiver.as_deref(), Some("cls"));
        let check = find(&c.members, "check");
        assert_eq!(check.sig.as_ref().unwrap().receiver, None);
        assert_eq!(check.sig.as_ref().unwrap().params[0].name, "x");
        assert_eq!(
            find(&c.members, "get").doc.as_ref().unwrap().summary,
            "Get one."
        );
        assert_eq!(find(&c.members, "Inner").kind, ApiKind::Class);
    }

    #[test]
    fn enums_and_conditional_blocks() {
        let f = extract(SAMPLE).unwrap();
        let color = find(&f.items, "Color");
        assert_eq!(color.kind, ApiKind::Enum);
        let red = find(&color.members, "RED");
        assert_eq!(red.kind, ApiKind::Variant);
        assert_eq!(red.signature, "RED = 1");
        assert_eq!(red.doc.as_ref().unwrap().summary, "Warm.");
        assert_eq!(find(&color.members, "describe").kind, ApiKind::Method);

        assert!(f.items.iter().any(|i| i.name == "typed_only"));
        assert!(!f.items.iter().any(|i| i.name == "main"));
        assert!(
            !f.items.iter().any(|i| i.name == "speedup"),
            "only the try body is walked; the fallback def is in `except`"
        );
    }

    #[test]
    fn numpy_docstring_and_deprecated_directive() {
        let src = r#"
def add(x, y=0):
    """Add.

    Parameters
    ----------
    x : int
        First.
    y : int, optional
        Second.

    .. deprecated:: 2.0
       Use ``plus``.
    """
"#;
        let f = extract(src).unwrap();
        let add = find(&f.items, "add");
        let ps = &add.sig.as_ref().unwrap().params;
        assert_eq!(ps[0].ty, "int");
        assert_eq!(ps[1].ty, "int, optional");
        let dep = add.deprecated.as_ref().unwrap();
        assert_eq!(dep.since.as_deref(), Some("2.0"));
        assert_eq!(dep.note.as_deref(), Some("Use `plus`."));
    }

    #[test]
    fn computed_all_is_unknown() {
        let f = extract("__all__ = base.__all__ + [\"a\"]\n").unwrap();
        assert_eq!(f.exports, None, "partly computed: the name rule applies");
        let f = extract("__all__ = [\n    # public\n    \"a\",\n]\n__all__.append('b')\n").unwrap();
        assert_eq!(f.exports, Some(vec!["a".into(), "b".into()]));
        let f = extract("__all__ = ['a']\n__all__ += helpers.__all__\n").unwrap();
        assert_eq!(f.exports, None);
    }

    #[test]
    fn broken_source_still_extracts() {
        let f = extract("def ok():\n    '''Fine.'''\n\ndef broken(:\n").unwrap();
        assert!(f.items.iter().any(|i| i.name == "ok"));
    }

    /// Extract every `.py` under `$PULSE_PY_REPO` (run with `--ignored --nocapture`).
    #[test]
    #[ignore]
    fn extract_python_repository() {
        let Ok(root) = std::env::var("PULSE_PY_REPO") else {
            return;
        };
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
                let p = e.path();
                if p.is_dir() {
                    if !p
                        .file_name()
                        .is_some_and(|n| n.to_string_lossy().starts_with('.'))
                    {
                        walk(&p, out);
                    }
                } else if p.extension().is_some_and(|x| x == "py") {
                    out.push(p);
                }
            }
        }
        let mut files = Vec::new();
        walk(std::path::Path::new(&root), &mut files);
        let sources: Vec<String> = files
            .iter()
            .filter_map(|f| std::fs::read_to_string(f).ok())
            .collect();
        let parse_start = std::time::Instant::now();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();
        for src in &sources {
            parser.parse(src, None);
        }
        println!("parse only: {:?}", parse_start.elapsed());
        let start = std::time::Instant::now();
        let (mut items, mut documented, mut public, mut examples, mut failed) = (0, 0, 0, 0, 0);
        fn count(items: &[ApiItem], n: &mut usize, d: &mut usize, p: &mut usize, e: &mut usize) {
            for i in items {
                *n += 1;
                if let Some(doc) = &i.doc {
                    *d += 1;
                    *e += doc.examples.len();
                }
                if i.visibility == Visibility::Public {
                    *p += 1;
                }
                count(&i.members, n, d, p, e);
            }
        }
        for src in &sources {
            match extract(src) {
                Ok(api) => count(
                    &api.items,
                    &mut items,
                    &mut documented,
                    &mut public,
                    &mut examples,
                ),
                Err(_) => failed += 1,
            }
        }
        println!(
            "{} files ({failed} failed), {items} items ({documented} documented, {public} public, {examples} examples) in {:?}",
            sources.len(),
            start.elapsed()
        );
    }
}
