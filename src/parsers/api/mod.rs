//! API extraction: the documented surface of a source file.
//!
//! The symbol index ([`crate::parsers`]) answers "where is `X` defined"; it keeps
//! name, kind and span only, because it sits on the query hot path. Docs need more:
//! the signature, the doc comment, visibility, the container (`impl Foo`), fields,
//! variants and methods. This module extracts that from one file's source.
//!
//! Pulse runs it at generation time over source-role files and caches the result by
//! content hash (see `pulse::extract::api_cache`), so `rfx index` pays nothing for it.

pub mod doc;
pub mod rust;

use crate::models::Language;
use serde::{Deserialize, Serialize};

/// Bump when the output of any extractor changes; cached results are then rebuilt.
pub const EXTRACTOR_VERSION: u32 = 2;

/// The documented surface of one file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ApiFile {
    /// Inner doc comment of the file (`//!`), if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub module_doc: Option<DocComment>,
    /// Top-level items, in source order. Members nest inside their parent.
    pub items: Vec<ApiItem>,
    /// `mod foo;` declarations (Rust), in source order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mod_decls: Vec<ModDecl>,
    /// `pub use` re-exports (Rust), in source order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reexports: Vec<ReExport>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ApiKind {
    Module,
    Function,
    Method,
    Struct,
    Enum,
    Variant,
    Field,
    Trait,
    TypeAlias,
    AssociatedType,
    Const,
    Static,
    Macro,
    Union,
}

impl ApiKind {
    /// Types get their own reference page; everything else is listed on its module
    /// or type page.
    pub fn is_type(&self) -> bool {
        matches!(
            self,
            ApiKind::Struct | ApiKind::Enum | ApiKind::Trait | ApiKind::Union
        )
    }

    pub fn label(&self) -> &'static str {
        match self {
            ApiKind::Module => "module",
            ApiKind::Function => "fn",
            ApiKind::Method => "method",
            ApiKind::Struct => "struct",
            ApiKind::Enum => "enum",
            ApiKind::Variant => "variant",
            ApiKind::Field => "field",
            ApiKind::Trait => "trait",
            ApiKind::TypeAlias => "type",
            ApiKind::AssociatedType => "associated type",
            ApiKind::Const => "const",
            ApiKind::Static => "static",
            ApiKind::Macro => "macro",
            ApiKind::Union => "union",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "scope", rename_all = "kebab-case")]
pub enum Visibility {
    /// `pub`, `export`, an uppercase Go name.
    Public,
    /// `pub(crate)`, `pub(super)`, `pub(in path)`.
    Restricted(String),
    /// No modifier where one is required to be public.
    Private,
    /// Visibility follows the container (trait methods, enum variants).
    Inherited,
}

impl Visibility {
    pub fn is_public(&self) -> bool {
        matches!(self, Visibility::Public | Visibility::Inherited)
    }
}

/// One documented item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApiItem {
    pub name: String,
    pub kind: ApiKind,
    pub visibility: Visibility,
    /// Declaration text up to the body, whitespace-collapsed (`pub fn f(x: u32) -> u8`).
    pub signature: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sig: Option<SigParts>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc: Option<DocComment>,
    /// Attributes other than doc comments: `derive(Debug, Clone)`, `cfg(test)`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attrs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deprecated: Option<Deprecation>,
    /// `#[doc(hidden)]`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub hidden: bool,
    /// Inside `#[cfg(test)]`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub test_only: bool,
    /// 1-based lines of the whole item.
    pub start_line: u32,
    pub end_line: u32,
    /// For methods and associated items: `impl Trait for Type`'s trait, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trait_impl: Option<String>,
    /// For impl members: the implementing type as written (`Foo<T>`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub self_type: Option<String>,
    /// Fields, variants, methods, associated items; inline module contents.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub members: Vec<ApiItem>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SigParts {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generics: Option<String>,
    /// `&self`, `&mut self`, `self`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receiver: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<Param>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub returns: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_async: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_unsafe: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_const: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub ty: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Deprecation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub since: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// A doc comment, as markdown, plus the parts the renderer lays out separately.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DocComment {
    /// The comment with markers stripped (`///`, `//!`, `/** */`).
    pub markdown: String,
    /// First paragraph, as plain-ish markdown.
    pub summary: String,
    /// `# Errors`, `# Panics`, `# Safety`, `# Examples`, … keyed by lowercase heading.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sections: Vec<(String, String)>,
    /// Code fences; rustdoc hidden lines (`# `) removed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<CodeExample>,
    /// Intra-doc links as written: `Foo`, `crate::x::Bar`, `Self::new`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<String>,
    /// 1-based lines of the comment.
    pub start_line: u32,
    pub end_line: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeExample {
    pub lang: String,
    pub code: String,
    /// Rustdoc compiles bare and `rust` fences unless marked `ignore`/`no_run`/`text`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub doctest: bool,
}

/// `mod foo;` / `pub mod foo;` / `mod foo { … }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModDecl {
    pub name: String,
    pub visibility: Visibility,
    /// `true` for `mod foo { … }` (its items are in [`ApiItem::members`]).
    pub inline: bool,
    /// `#[path = "…"]`, if present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_attr: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub test_only: bool,
    pub line: u32,
}

/// `pub use a::b::C;` / `pub use a::b::C as D;` / `pub use a::b::*;`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReExport {
    /// Path as written, `::`-separated (`crate::query::QueryEngine`).
    pub path: String,
    /// Name it is exported under (`QueryEngine`, `D`, or `*`).
    pub name: String,
    pub visibility: Visibility,
    pub line: u32,
}

/// Whether [`extract`] handles this language.
pub fn has_extractor(language: Language) -> bool {
    matches!(language, Language::Rust)
}

/// Extract the API of one file, if its language has an extractor.
pub fn extract(language: Language, source: &str) -> Option<ApiFile> {
    match language {
        Language::Rust => rust::extract(source).ok(),
        _ => None,
    }
}

/// Collapse runs of whitespace to one space.
pub(crate) fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            space = true;
        } else {
            if space && !out.is_empty() {
                out.push(' ');
            }
            space = false;
            out.push(c);
        }
    }
    out
}
