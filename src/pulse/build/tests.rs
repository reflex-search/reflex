//! End-to-end model build over a small indexed fixture repository.

use super::*;
use crate::models::IndexConfig;
use crate::pulse::model::{Block, Inline, PageId, PageKind, TabId};
use crate::{CacheManager, Indexer};
use std::path::Path;
use tempfile::TempDir;

fn write(root: &Path, rel: &str, body: &str) {
    let p = root.join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, body).unwrap();
}

/// `src/api` (3 files) and `src/store` (3 files) import each other: a cycle.
/// Tests, a corpus fixture and `build.rs` must not become modules.
fn fixture() -> TempDir {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "README.md",
        "# Demo\n\n[![ci](b.svg)](c)\n\nDemo is a tiny key-value service. It stores values.\n\n## Usage\n\n```sh\ndemo serve\n```\n",
    );
    write(
        r,
        "Cargo.toml",
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\n",
    );
    write(r, "build.rs", "fn main() {}\n");
    write(
        r,
        "src/main.rs",
        "mod api;\nmod store;\n\nfn main() {\n    api::run();\n}\n",
    );
    write(
        r,
        "src/api/mod.rs",
        "pub mod handlers;\npub mod routes;\n\npub fn run() {}\n",
    );
    write(
        r,
        "src/api/routes.rs",
        "use crate::store::db::Db;\n\npub fn routes(_db: &Db) {}\n",
    );
    write(
        r,
        "src/api/handlers.rs",
        "use crate::store::db::Db;\n\npub fn get(_db: &Db) {}\npub fn put(_db: &Db) {}\n",
    );
    write(r, "src/store/mod.rs", "pub mod cache;\npub mod db;\n");
    write(
        r,
        "src/store/db.rs",
        "pub struct Db;\n\nimpl Db {\n    pub fn open() -> Self { Db }\n}\n",
    );
    write(
        r,
        "src/store/cache.rs",
        "use crate::api::routes::routes;\n\npub fn warm() { let _ = routes; }\n",
    );
    write(r, "tests/it.rs", "#[test]\nfn works() {}\n");
    write(r, "tests/corpus/sample.rs", "fn sample() {}\n");
    Indexer::new(CacheManager::new(r), IndexConfig::default())
        .index(r, false)
        .unwrap();
    t
}

fn opts() -> BuildOptions {
    BuildOptions {
        detect_repo: false,
        ..BuildOptions::new("Demo")
    }
}

#[test]
fn builds_tabs_modules_and_excludes_non_source() {
    let t = fixture();
    let site = build_site(&CacheManager::new(t.path()), &opts()).unwrap();

    let modules: Vec<&str> = site
        .pages
        .values()
        .filter_map(|p| match &p.kind {
            PageKind::Module { module } => Some(module.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(modules, vec!["src", "src/api", "src/store"]);

    assert_eq!(site.tabs.len(), 2);
    assert_eq!(site.tabs[0].id, TabId::Docs);
    assert_eq!(site.pages[&site.tabs[0].landing].route, "/");
    assert_eq!(site.pages[&site.tabs[1].landing].route, "/internals/");
    assert_eq!(
        site.pages[&module_page_id(&"src/api".into())].route,
        "/internals/modules/src/api/"
    );
    assert!(
        site.report.broken_links.is_empty(),
        "{:?}",
        site.report.broken_links
    );
    assert_eq!(site.report.files_by_role.get("fixture"), Some(&1));
    assert_eq!(site.report.files_by_role.get("build"), Some(&1));
}

#[test]
fn facts_are_consistent_and_cycles_found() {
    let t = fixture();
    let site = build_site(&CacheManager::new(t.path()), &opts()).unwrap();
    let get = |id: &str| site.facts.get(&FactId::new(id)).unwrap().value.display();
    assert_eq!(get("site:source_files"), "7");
    assert_eq!(get("site:modules"), "3");
    assert_eq!(get("module:src/api:files"), "3");
    assert_eq!(get("module:src:files_with_submodules"), "7");
    assert_eq!(get("site:module_cycles"), "1");

    let api = &site.pages[&module_page_id(&"src/api".into())];
    let has_cycle_callout = api
        .blocks
        .iter()
        .any(|b| matches!(b, Block::Callout { title: Some(t), .. } if t == "Dependency cycle"));
    assert!(has_cycle_callout, "src/api imports src/store and back");
}

#[test]
fn home_uses_readme_intro_and_has_overview_slot() {
    let t = fixture();
    let site = build_site(&CacheManager::new(t.path()), &opts()).unwrap();
    let home = &site.pages[&site.tabs[0].landing];
    assert_eq!(
        home.description.as_deref(),
        Some("Demo is a tiny key-value service.")
    );
    let Block::Narrative { slot, fallback, .. } = &home.blocks[0] else {
        panic!("home starts with the overview slot");
    };
    assert_eq!(slot, "project-overview");
    let Block::Markdown { markdown } = &fallback[0] else {
        panic!("fallback is the README intro");
    };
    assert_eq!(
        markdown.source,
        "Demo is a tiny key-value service. It stores values."
    );
    assert_eq!(markdown.from.as_ref().unwrap().label(), "README.md:5");

    let slots = site.narrative_slots();
    assert!(slots.contains(&"architecture".to_string()));
    assert!(slots.contains(&"module:src/api".to_string()));
}

#[test]
fn model_json_is_deterministic_and_fills_slots() {
    let t = fixture();
    let cache = CacheManager::new(t.path());
    let a = build_site(&cache, &opts()).unwrap().to_json().unwrap();
    let b = build_site(&cache, &opts()).unwrap().to_json().unwrap();
    assert_eq!(a, b);
    assert!(!a.contains("generated_at"));

    let mut site = build_site(&cache, &opts()).unwrap();
    let filled = site.fill_narratives(|slot| (slot == "architecture").then(|| "Prose.".into()));
    assert_eq!(filled, 1);
}

#[test]
fn slugs_persist_across_runs() {
    let t = fixture();
    let cache = CacheManager::new(t.path());
    let slugs = t.path().join(".reflex/pulse/slugs.json");
    let o = BuildOptions {
        slugs_path: Some(slugs.clone()),
        ..opts()
    };
    let first = build_site(&cache, &o).unwrap();
    assert!(slugs.exists());
    let second = build_site(&cache, &o).unwrap();
    let routes = |s: &Site| {
        s.pages
            .values()
            .map(|p| p.route.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(routes(&first), routes(&second));
}

fn library_fixture() -> TempDir {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "Cargo.toml",
        "[package]\nname = \"kv-lib\"\nversion = \"0.1.0\"\n",
    );
    write(
        r,
        "src/lib.rs",
        "//! A key-value library. Start with [`Store`].\npub mod store;\nmod util;\npub use util::helper;\n",
    );
    write(
        r,
        "src/store.rs",
        "/// An in-memory store.\n///\n/// Create one with [`Store::new`].\n///\n/// ```\n/// # use kv_lib::store::Store;\n/// let s = Store::new();\n/// ```\npub struct Store {\n    /// Number of entries.\n    pub len: usize,\n}\n\nimpl Store {\n    /// Make an empty store. See [`crate::helper`].\n    pub fn new() -> Self { Store { len: 0 } }\n    fn secret(&self) {}\n}\n\nimpl Default for Store {\n    fn default() -> Self { Self::new() }\n}\n\n/// Largest key size.\npub const MAX_KEY: usize = 256;\n",
    );
    write(r, "src/util.rs", "/// Helps.\npub fn helper() {}\n");
    Indexer::new(CacheManager::new(r), IndexConfig::default())
        .index(r, false)
        .unwrap();
    t
}

#[test]
fn library_reference_pages_symbols_and_links() {
    let t = library_fixture();
    let site = build_site(&CacheManager::new(t.path()), &opts()).unwrap();
    assert!(
        site.report.broken_links.is_empty(),
        "{:?}",
        site.report.broken_links
    );

    let root = &site.pages[&PageId::new("docs/ref/mod/kv_lib")];
    assert_eq!(root.route, "/docs/reference/kv-lib/");
    let store = &site.pages[&PageId::new("docs/ref/type/kv_lib::store::Store")];
    assert_eq!(store.route, "/docs/reference/kv-lib/store/store/");

    // The type's definition block: doc with hidden line removed and a resolved link.
    let Block::Symbol { symbol } = &store.blocks[0] else {
        panic!("type page starts with its definition");
    };
    assert_eq!(symbol.signature, "pub struct Store");
    let doc = &symbol.doc.as_ref().unwrap().source;
    assert!(
        doc.contains("[`Store::new`](/docs/reference/kv-lib/store/store/#method.new)"),
        "{doc}"
    );
    assert!(doc.contains("```rust\nlet s = Store::new();\n```"), "{doc}");
    assert!(!doc.contains("# use"), "{doc}");

    // Fields and public methods are documented; private ones are not.
    let names: Vec<&str> = store
        .blocks
        .iter()
        .filter_map(|b| match b {
            Block::Symbol { symbol } => Some(symbol.name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(names, vec!["Store", "len", "new"]);

    // `pub use util::helper` makes a private-module function public at the root.
    let helper = site
        .symbols
        .values()
        .find(|s| s.name == "helper")
        .expect("re-exported helper is documented");
    assert_eq!(helper.path, "kv_lib::helper");

    // Module docs link to the type page.
    let Block::Markdown { markdown } = &root.blocks[0] else {
        panic!("root page starts with the crate docs");
    };
    assert!(
        markdown
            .source
            .contains("[`Store`](/docs/reference/kv-lib/store/store/)"),
        "{}",
        markdown.source
    );

    // Trait impls are listed, not documented as methods.
    let lists_default = store.blocks.iter().any(|b| matches!(b,
        Block::List { items, .. } if items.iter().flatten().any(|i| matches!(i, Inline::Code { code } if code == "impl Default for Store"))));
    assert!(lists_default);
}
