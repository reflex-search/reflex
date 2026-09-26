//! Rust public surface.
//!
//! 1. Crates come from `Cargo.toml` files in the index: a library (`[lib] path`, default
//!    `src/lib.rs`) and binaries (`[[bin]]`, `src/main.rs`, `src/bin/*.rs`).
//! 2. From each crate root the module tree follows `mod` declarations: `name.rs`,
//!    `name/mod.rs`, `#[path]`, and inline `mod name { … }`. `#[cfg(test)]` modules are
//!    skipped.
//! 3. A module is public when its parent is public and it is declared `pub`; the root
//!    of a library is public, the root of a binary is not. An item is public when its
//!    module is public, it is `pub`, and it is not `#[doc(hidden)]`.
//! 4. `impl` blocks anywhere in the crate attach their members to the type they name.
//! 5. `pub use` in a public module makes its target public under the re-exported path
//!    (rustdoc's inlining of items from private modules).

use crate::parsers::api::{ApiFile, ApiItem, ApiKind, DocComment, Visibility};
use crate::pulse::extract::api_cache::ApiIndex;
use crate::pulse::extract::{ContentAccess, Corpus};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
pub struct RustApi {
    pub crates: Vec<RustCrate>,
}

#[derive(Debug, Clone)]
pub struct RustCrate {
    /// Crate name as code refers to it (`reflex`, `my_lib`).
    pub name: String,
    /// Package name from `Cargo.toml` (`reflex-search`).
    pub package: String,
    pub manifest: String,
    pub is_lib: bool,
    pub root_file: String,
    /// Tree order; `modules[0]` is the crate root.
    pub modules: Vec<RustModule>,
}

#[derive(Debug, Clone)]
pub struct RustModule {
    /// `demo`, `demo::store`, `demo::store::db`.
    pub path: String,
    pub name: String,
    pub file: String,
    pub public: bool,
    pub doc: Option<DocComment>,
    pub items: Vec<RustItem>,
    pub children: Vec<usize>,
    pub parent: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct RustItem {
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

impl RustApi {
    pub fn resolve(corpus: &Corpus, apis: &ApiIndex, content: &ContentAccess) -> Self {
        let file_api = |path: &str| -> Option<&ApiFile> {
            corpus.index_of(path).and_then(|i| apis.files.get(&i))
        };
        let exists = |p: &str| corpus.index_of(p).is_some();
        let mut crates = Vec::new();
        for f in corpus
            .files
            .iter()
            .filter(|f| f.path == "Cargo.toml" || f.path.ends_with("/Cargo.toml"))
        {
            let Some(text) = content.read(&f.path) else {
                continue;
            };
            for spec in crate_specs(&f.path, text, |p| corpus.index_of(p).is_some()) {
                let mut k = RustCrate {
                    name: spec.name,
                    package: spec.package,
                    manifest: f.path.clone(),
                    is_lib: spec.is_lib,
                    root_file: spec.root.clone(),
                    modules: Vec::new(),
                };
                let mut walker = Walker {
                    crate_: &mut k,
                    file_api: &file_api,
                    exists: &exists,
                    impls: Vec::new(),
                    reexports: Vec::new(),
                    depth: 0,
                };
                let root_name = walker.crate_.name.clone();
                walker.file(&spec.root, root_name.clone(), root_name, spec.is_lib, None);
                let (impls, reexports) = (walker.impls, walker.reexports);
                attach_impls(&mut k, impls);
                apply_reexports(&mut k, reexports);
                crates.push(k);
            }
        }
        crates.sort_by(|a, b| (!a.is_lib, &a.name).cmp(&(!b.is_lib, &b.name)));
        Self { crates }
    }

    pub fn libraries(&self) -> impl Iterator<Item = &RustCrate> {
        self.crates.iter().filter(|c| c.is_lib)
    }
}

struct CrateSpec {
    name: String,
    package: String,
    root: String,
    is_lib: bool,
}

fn join(dir: &str, rel: &str) -> String {
    let mut parts: Vec<&str> = if dir.is_empty() {
        Vec::new()
    } else {
        dir.split('/').collect()
    };
    for seg in rel.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            s => parts.push(s),
        }
    }
    parts.join("/")
}

fn parent_dir(path: &str) -> &str {
    path.rsplit_once('/').map(|(d, _)| d).unwrap_or("")
}

/// Library and binary crates declared by one manifest.
fn crate_specs(manifest: &str, text: &str, exists: impl Fn(&str) -> bool) -> Vec<CrateSpec> {
    let Ok(table) = text.parse::<toml::Table>() else {
        return Vec::new();
    };
    let Some(package) = table
        .get("package")
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str())
    else {
        return Vec::new();
    };
    let dir = parent_dir(manifest);
    let snake = package.replace('-', "_");
    let mut out = Vec::new();

    let lib = table.get("lib");
    let lib_path = lib
        .and_then(|l| l.get("path"))
        .and_then(|p| p.as_str())
        .unwrap_or("src/lib.rs");
    let lib_root = join(dir, lib_path);
    if exists(&lib_root) {
        out.push(CrateSpec {
            name: lib
                .and_then(|l| l.get("name"))
                .and_then(|n| n.as_str())
                .map(|n| n.replace('-', "_"))
                .unwrap_or_else(|| snake.clone()),
            package: package.to_string(),
            root: lib_root,
            is_lib: true,
        });
    }

    let mut bins: Vec<(String, String)> = table
        .get("bin")
        .and_then(|b| b.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|b| {
                    let name = b.get("name")?.as_str()?.to_string();
                    let path = b
                        .get("path")
                        .and_then(|p| p.as_str())
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("src/bin/{name}.rs"));
                    Some((name, join(dir, &path)))
                })
                .collect()
        })
        .unwrap_or_default();
    let main = join(dir, "src/main.rs");
    if exists(&main) && !bins.iter().any(|(_, p)| *p == main) {
        bins.push((package.to_string(), main));
    }
    for (name, root) in bins {
        if exists(&root) {
            out.push(CrateSpec {
                name: name.replace('-', "_"),
                package: package.to_string(),
                root,
                is_lib: false,
            });
        }
    }
    out
}

struct PendingImpl {
    item: ApiItem,
    file: String,
    module: usize,
}

struct PendingReexport {
    module: usize,
    path: String,
    name: String,
}

struct Walker<'a, 'f> {
    crate_: &'a mut RustCrate,
    file_api: &'f dyn Fn(&str) -> Option<&'f ApiFile>,
    exists: &'f dyn Fn(&str) -> bool,
    impls: Vec<PendingImpl>,
    reexports: Vec<PendingReexport>,
    depth: usize,
}

impl Walker<'_, '_> {
    fn push_module(
        &mut self,
        path: String,
        name: String,
        file: &str,
        public: bool,
        doc: Option<DocComment>,
        parent: Option<usize>,
    ) -> usize {
        let idx = self.crate_.modules.len();
        self.crate_.modules.push(RustModule {
            path,
            name,
            file: file.to_string(),
            public,
            doc,
            items: Vec::new(),
            children: Vec::new(),
            parent,
        });
        if let Some(p) = parent {
            self.crate_.modules[p].children.push(idx);
        }
        idx
    }

    fn file(
        &mut self,
        file: &str,
        path: String,
        name: String,
        public: bool,
        parent: Option<usize>,
    ) {
        // Guard against `#[path]` loops.
        if self.depth > 64
            || self
                .crate_
                .modules
                .iter()
                .any(|m| m.file == file && m.path == path)
        {
            return;
        }
        let Some(api) = (self.file_api)(file) else {
            self.push_module(path, name, file, public, None, parent);
            return;
        };
        let idx = self.push_module(
            path.clone(),
            name,
            file,
            public,
            api.module_doc.clone(),
            parent,
        );
        self.items(&api.items, file, idx);
        for r in &api.reexports {
            if r.visibility == Visibility::Public {
                self.reexports.push(PendingReexport {
                    module: idx,
                    path: r.path.clone(),
                    name: r.name.clone(),
                });
            }
        }

        let fname = file.rsplit('/').next().unwrap_or(file);
        let is_mod_rs = matches!(fname, "mod.rs" | "lib.rs" | "main.rs") || parent.is_none();
        let here = parent_dir(file);
        let stem = fname.trim_end_matches(".rs");
        let child_dir = if is_mod_rs {
            here.to_string()
        } else {
            join(here, stem)
        };
        for d in &api.mod_decls {
            if d.inline || d.test_only {
                continue;
            }
            let candidates = match &d.path_attr {
                Some(p) => vec![join(here, p)],
                None => vec![
                    join(&child_dir, &format!("{}.rs", d.name)),
                    join(&child_dir, &format!("{}/mod.rs", d.name)),
                ],
            };
            if let Some(child) = candidates.into_iter().find(|c| (self.exists)(c)) {
                let child_public = public && d.visibility == Visibility::Public;
                self.depth += 1;
                self.file(
                    &child,
                    format!("{path}::{}", d.name),
                    d.name.clone(),
                    child_public,
                    Some(idx),
                );
                self.depth -= 1;
            }
        }
    }

    fn items(&mut self, items: &[ApiItem], file: &str, module: usize) {
        let module_public = self.crate_.modules[module].public;
        let module_path = self.crate_.modules[module].path.clone();
        for it in items {
            if it.test_only {
                continue;
            }
            if it.self_type.is_some() {
                self.impls.push(PendingImpl {
                    item: it.clone(),
                    file: file.to_string(),
                    module,
                });
                continue;
            }
            if it.kind == ApiKind::Module {
                let public = module_public && it.visibility == Visibility::Public && !it.hidden;
                let idx = self.push_module(
                    format!("{module_path}::{}", it.name),
                    it.name.clone(),
                    file,
                    public,
                    it.doc.clone(),
                    Some(module),
                );
                self.items(&it.members, file, idx);
                continue;
            }
            let public = module_public && it.visibility == Visibility::Public && !it.hidden;
            self.crate_.modules[module].items.push(RustItem {
                path: format!("{module_path}::{}", it.name),
                item: it.clone(),
                file: file.to_string(),
                public,
                defined_at: None,
                impl_files: BTreeMap::new(),
            });
        }
    }
}

/// `Foo<T>`, `&'a Foo`, `crate::x::Foo` → `Foo`.
fn base_type_name(t: &str) -> &str {
    let t = t.trim_start_matches('&').trim_start();
    let t = t.strip_prefix("mut ").unwrap_or(t);
    let t = if t.starts_with('\'') {
        t.split_once(' ').map(|(_, r)| r).unwrap_or(t)
    } else {
        t
    };
    let t = t.split('<').next().unwrap_or(t);
    t.rsplit("::").next().unwrap_or(t).trim()
}

fn attach_impls(k: &mut RustCrate, impls: Vec<PendingImpl>) {
    // Type name → (module, item) locations.
    let mut types: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
    for (mi, m) in k.modules.iter().enumerate() {
        for (ii, it) in m.items.iter().enumerate() {
            if it.item.kind.is_type() {
                types
                    .entry(it.item.name.clone())
                    .or_default()
                    .push((mi, ii));
            }
        }
    }
    for p in impls {
        let Some(self_type) = p.item.self_type.as_deref() else {
            continue;
        };
        let Some(cands) = types.get(base_type_name(self_type)) else {
            continue;
        };
        let pick = cands
            .iter()
            .find(|(mi, ii)| k.modules[*mi].items[*ii].file == p.file)
            .or_else(|| cands.iter().find(|(mi, _)| *mi == p.module))
            .or_else(|| (cands.len() == 1).then(|| &cands[0]));
        if let Some(&(mi, ii)) = pick {
            let target = &mut k.modules[mi].items[ii];
            target.impl_files.insert(target.item.members.len(), p.file);
            target.item.members.push(p.item);
        }
    }
}

/// Resolve a `use` path from `module` to a module index plus the final segment.
fn resolve_use(k: &RustCrate, module: usize, path: &str) -> Option<(usize, String)> {
    let segs: Vec<&str> = path.split("::").collect();
    let (last, init) = segs.split_last()?;
    let mut cur = module;
    for (i, seg) in init.iter().enumerate() {
        cur = match *seg {
            "crate" if i == 0 => 0,
            s if i == 0 && s == k.name => 0,
            "self" if i == 0 => module,
            "super" => k.modules[cur].parent?,
            name => *k.modules[cur]
                .children
                .iter()
                .find(|&&c| k.modules[c].name == name)?,
        };
    }
    Some((cur, last.to_string()))
}

fn apply_reexports(k: &mut RustCrate, reexports: Vec<PendingReexport>) {
    for r in reexports {
        if !k.modules[r.module].public {
            continue;
        }
        let Some((target, last)) = resolve_use(k, r.module, &r.path) else {
            continue;
        };
        let at = k.modules[r.module].path.clone();
        let picks: Vec<usize> = k.modules[target]
            .items
            .iter()
            .enumerate()
            .filter(|(_, it)| {
                if last == "*" {
                    it.item.visibility == Visibility::Public
                } else {
                    it.item.name == last
                }
            })
            .map(|(i, _)| i)
            .collect();
        for ii in picks {
            let it = &k.modules[target].items[ii];
            if it.public || it.item.hidden {
                continue;
            }
            // Like rustdoc, document the item where it is re-exported; the definition in
            // the private module stays internal.
            let exported_as = if r.name == "*" {
                it.item.name.clone()
            } else {
                r.name.clone()
            };
            let mut copy = it.clone();
            copy.defined_at = Some(it.path.clone());
            copy.path = format!("{at}::{exported_as}");
            copy.item.name = exported_as;
            copy.public = true;
            k.modules[r.module].items.push(copy);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pulse::extract::api_cache::ApiIndex;

    struct Fixture {
        files: BTreeMap<&'static str, &'static str>,
    }

    impl Fixture {
        fn resolve(&self) -> RustApi {
            let exists = |p: &str| self.files.contains_key(p);
            let apis: BTreeMap<&str, ApiFile> = self
                .files
                .iter()
                .filter(|(p, _)| p.ends_with(".rs"))
                .map(|(p, s)| (*p, crate::parsers::api::rust::extract(s).unwrap()))
                .collect();
            let file_api = |p: &str| apis.get(p);
            let mut crates = Vec::new();
            for spec in crate_specs("Cargo.toml", self.files["Cargo.toml"], exists) {
                let mut k = RustCrate {
                    name: spec.name,
                    package: spec.package,
                    manifest: "Cargo.toml".into(),
                    is_lib: spec.is_lib,
                    root_file: spec.root.clone(),
                    modules: Vec::new(),
                };
                let mut w = Walker {
                    crate_: &mut k,
                    file_api: &file_api,
                    exists: &exists,
                    impls: Vec::new(),
                    reexports: Vec::new(),
                    depth: 0,
                };
                let n = w.crate_.name.clone();
                w.file(&spec.root, n.clone(), n, spec.is_lib, None);
                let (i, r) = (w.impls, w.reexports);
                attach_impls(&mut k, i);
                apply_reexports(&mut k, r);
                crates.push(k);
            }
            let _ = ApiIndex::default();
            RustApi { crates }
        }
    }

    fn fixture() -> Fixture {
        let mut files = BTreeMap::new();
        files.insert(
            "Cargo.toml",
            "[package]\nname = \"my-lib\"\nversion = \"0.1.0\"\n",
        );
        files.insert(
            "src/lib.rs",
            "//! My lib.\npub mod api;\nmod internal;\n#[cfg(test)]\nmod tests;\npub use internal::Engine;\npub mod inline { pub fn f() {} }\n",
        );
        files.insert("src/main.rs", "fn main() {}\n");
        files.insert(
            "src/api/mod.rs",
            "pub mod types;\npub(crate) mod hidden;\n/// Run it.\npub fn run() {}\nfn helper() {}\n",
        );
        files.insert(
            "src/api/types.rs",
            "/// A thing.\npub struct Thing;\nimpl Thing {\n    /// Make.\n    pub fn new() -> Self { Thing }\n}\n#[doc(hidden)]\npub struct Secret;\n",
        );
        files.insert("src/api/hidden.rs", "pub fn nope() {}\n");
        files.insert(
            "src/internal.rs",
            "pub struct Engine;\nimpl Engine { pub fn go(&self) {} }\npub struct NotExported;\n",
        );
        files.insert("src/tests.rs", "pub fn t() {}\n");
        Fixture { files }
    }

    fn item<'a>(k: &'a RustCrate, path: &str) -> &'a RustItem {
        k.modules
            .iter()
            .flat_map(|m| &m.items)
            .find(|i| i.path == path)
            .unwrap_or_else(|| panic!("{path} not found"))
    }

    #[test]
    fn crates_and_module_tree() {
        let api = fixture().resolve();
        assert_eq!(api.crates.len(), 2);
        let lib = &api.crates[0];
        assert!(lib.is_lib);
        assert_eq!(lib.name, "my_lib");
        assert_eq!(lib.package, "my-lib");
        let mods: Vec<(&str, bool)> = lib
            .modules
            .iter()
            .map(|m| (m.path.as_str(), m.public))
            .collect();
        assert_eq!(
            mods,
            vec![
                ("my_lib", true),
                ("my_lib::inline", true),
                ("my_lib::api", true),
                ("my_lib::api::types", true),
                ("my_lib::api::hidden", false),
                ("my_lib::internal", false),
            ]
        );
        assert_eq!(lib.modules[0].doc.as_ref().unwrap().summary, "My lib.");
        assert!(!api.crates[1].is_lib);
        assert!(
            !api.crates[1].modules[0].public,
            "binary roots are not public"
        );
    }

    #[test]
    fn visibility_impls_and_reexports() {
        let api = fixture().resolve();
        let lib = &api.crates[0];
        assert!(item(lib, "my_lib::api::run").public);
        assert!(!item(lib, "my_lib::api::helper").public);
        assert!(!item(lib, "my_lib::api::hidden::nope").public);
        assert!(!item(lib, "my_lib::api::types::Secret").public);

        let thing = item(lib, "my_lib::api::types::Thing");
        assert!(thing.public);
        assert_eq!(thing.item.members[0].name, "new");

        let engine = item(lib, "my_lib::Engine");
        assert!(engine.public, "re-exported from a private module");
        assert!(
            !item(lib, "my_lib::internal::Engine").public,
            "definition stays internal"
        );
        assert_eq!(
            engine.defined_at.as_deref(),
            Some("my_lib::internal::Engine")
        );
        assert_eq!(engine.item.members[0].name, "go");
        assert!(!item(lib, "my_lib::internal::NotExported").public);
    }

    #[test]
    fn helpers() {
        assert_eq!(base_type_name("Foo<T>"), "Foo");
        assert_eq!(base_type_name("&'a mut crate::x::Foo<T>"), "Foo");
        assert_eq!(join("src/api", "../lib.rs"), "src/lib.rs");
        assert_eq!(join("", "src/lib.rs"), "src/lib.rs");
    }
}
