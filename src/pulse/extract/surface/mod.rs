//! Public surface: which items a user of the code can reach, and under which path.
//!
//! Per language, a resolver walks from the package entry points (a crate's `lib.rs`,
//! a package's `exports`, …) and marks what is reachable. Everything else in a source
//! file is internal: it appears on Internals pages only.

pub mod python;
pub mod rust;

use crate::parsers::api::{ApiItem, DocComment};
use std::collections::BTreeMap;

/// Every package the corpus documents.
#[derive(Debug, Clone, Default)]
pub struct Surface {
    pub packages: Vec<Package>,
}

#[derive(Debug, Clone)]
/// A unit of distribution: a Rust crate, a Python package, an npm package, a Go module.
pub struct Package {
    /// Name as code refers to it (`reflex`, `my_lib`, `requests`).
    pub name: String,
    pub lang: crate::models::Language,
    /// Path separator in `SurfaceModule::path` / `SurfaceItem::path` (`::`, `.`).
    pub sep: &'static str,
    /// Distribution name (`reflex-search`).
    pub package: String,
    pub manifest: String,
    pub is_lib: bool,
    pub root_file: String,
    /// Tree order; `modules[0]` is the crate root.
    pub modules: Vec<SurfaceModule>,
}

#[derive(Debug, Clone)]
pub struct SurfaceModule {
    /// `demo`, `demo::store`, `demo::store::db`.
    pub path: String,
    pub name: String,
    pub file: String,
    pub public: bool,
    pub doc: Option<DocComment>,
    pub items: Vec<SurfaceItem>,
    pub children: Vec<usize>,
    pub parent: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct SurfaceItem {
    /// For types, `members` also holds methods and associated items from `impl` blocks.
    pub item: ApiItem,
    pub file: String,
    pub public: bool,
    /// Documented path (`demo::store::Db`, or the re-export path).
    pub path: String,
    /// Where it is defined, when `path` is a re-export.
    pub defined_at: Option<String>,
    /// File of each member attached from an `impl` block, by index in `item.members`.
    pub impl_files: BTreeMap<usize, String>,
}

impl Surface {
    /// Library packages: the ones with a public API to document.
    pub fn libraries(&self) -> impl Iterator<Item = &Package> {
        self.packages.iter().filter(|c| c.is_lib)
    }

    pub fn extend(&mut self, other: Surface) {
        self.packages.extend(other.packages);
    }
}

impl Package {
    /// `a`, `b` → `a::b` (or `a.b`).
    pub fn join(&self, a: &str, b: &str) -> String {
        format!("{a}{}{b}", self.sep)
    }

    /// Lowercase language id for symbol ids and code highlighting (`rust`).
    pub fn lang_id(&self) -> String {
        serde_json::to_value(self.lang)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_else(|| "text".into())
    }
}
