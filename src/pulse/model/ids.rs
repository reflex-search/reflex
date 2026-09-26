//! Stable identities and slugs.
//!
//! Identity is semantic ("the module `src/pulse`", "the changelog"), never positional
//! or derived from a file's line numbers, so ids survive edits. A page's URL comes from
//! its id through [`SlugAllocator`], which remembers earlier choices in
//! `.reflex/pulse/slugs.json` so published links stay stable.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

macro_rules! id_type {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            pub fn new(s: impl Into<String>) -> Self {
                Self(s.into())
            }
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.to_string())
            }
        }
    };
}

id_type!(
    /// A page, e.g. `int/module/src/pulse` or `docs/changelog`.
    PageId
);
id_type!(
    /// An anchor inside a page, e.g. `dependencies`.
    AnchorId
);
id_type!(
    /// A module: a directory of source files, e.g. `src/pulse`.
    ModuleId
);
id_type!(
    /// A fact, e.g. `module:src/pulse:files`.
    FactId
);
id_type!(
    /// A symbol, e.g. `rust:reflex_search::query::QueryEngine#struct`. Filled from M3.
    SymbolId
);

/// Lowercase kebab slug of one path segment: `[a-z0-9-]`, no leading/trailing `-`.
pub fn slugify(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    let mut dash = false;
    for c in segment.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            dash = false;
        } else if !dash && !out.is_empty() {
            out.push('-');
            dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        out.push('x');
    }
    out
}

/// A heading's anchor, compatible with GitHub/Starlight heading slugs for ASCII text.
pub fn anchor_for(text: &str) -> AnchorId {
    AnchorId(slugify(text))
}

/// Assigns each page a URL path segment (the part after the tab prefix) and keeps it
/// stable across runs.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SlugAllocator {
    /// Page id → slug, from earlier runs and this one.
    assigned: BTreeMap<PageId, String>,
    #[serde(skip)]
    taken: BTreeSet<String>,
    #[serde(skip)]
    used: BTreeSet<PageId>,
}

impl SlugAllocator {
    /// Load `.reflex/pulse/slugs.json`, or start empty.
    pub fn load(path: &Path) -> Self {
        let mut me: Self = std::fs::read(path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        me.taken = me.assigned.values().cloned().collect();
        me
    }

    /// The slug for `id`, preferring `wanted` (already slugified segments joined by `/`).
    ///
    /// A page keeps the slug it had last run. A new page that collides with another
    /// page's slug gets `-2`, `-3`, … in id order, so output is deterministic.
    pub fn assign(&mut self, id: &PageId, wanted: &str) -> String {
        self.used.insert(id.clone());
        if let Some(existing) = self.assigned.get(id) {
            return existing.clone();
        }
        let mut slug = wanted.to_string();
        let mut n = 2;
        while self.taken.contains(&slug) {
            slug = format!("{wanted}-{n}");
            n += 1;
        }
        self.taken.insert(slug.clone());
        self.assigned.insert(id.clone(), slug.clone());
        slug
    }

    /// Save the slugs of pages used this run. Slugs of pages that disappeared are
    /// dropped, so the name can be reused.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let kept: BTreeMap<&PageId, &String> = self
            .assigned
            .iter()
            .filter(|(id, _)| self.used.contains(*id))
            .collect();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = crate::atomic_write::tmp_path_for(path);
        std::fs::write(
            &tmp,
            serde_json::to_vec_pretty(&serde_json::json!({ "assigned": kept }))?,
        )?;
        crate::atomic_write::atomic_replace(&tmp, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_segments() {
        assert_eq!(slugify("src"), "src");
        assert_eq!(slugify("Query Engine"), "query-engine");
        assert_eq!(slugify("build.rs"), "build-rs");
        assert_eq!(slugify("__init__"), "init");
        assert_eq!(slugify("--"), "x");
        assert_eq!(slugify("v2.0.1"), "v2-0-1");
    }

    #[test]
    fn slugs_are_stable_and_collisions_deterministic() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("slugs.json");
        let mut a = SlugAllocator::load(&path);
        assert_eq!(a.assign(&"p1".into(), "src/a-b"), "src/a-b");
        assert_eq!(a.assign(&"p2".into(), "src/a-b"), "src/a-b-2");
        a.save(&path).unwrap();

        // Next run: p2 keeps its slug even if assigned first.
        let mut b = SlugAllocator::load(&path);
        assert_eq!(b.assign(&"p2".into(), "src/a-b"), "src/a-b-2");
        assert_eq!(b.assign(&"p1".into(), "src/a-b"), "src/a-b");
        assert_eq!(b.assign(&"p3".into(), "src/a-b"), "src/a-b-3");
    }

    #[test]
    fn vanished_pages_free_their_slug() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("slugs.json");
        let mut a = SlugAllocator::load(&path);
        a.assign(&"old".into(), "x");
        a.save(&path).unwrap();
        let b = SlugAllocator::load(&path);
        b.save(&path).unwrap(); // "old" not used this run
        let mut c = SlugAllocator::load(&path);
        assert_eq!(c.assign(&"new".into(), "x"), "x");
    }
}
