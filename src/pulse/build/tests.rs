//! End-to-end model build over a small indexed fixture repository.

use super::*;
use crate::models::IndexConfig;
use crate::pulse::model::{Block, PageKind, TabId};
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
