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
    write(
        r,
        "docs/usage.md",
        "# Using kv\n\nOpen a [`Store`](../src/store.rs), then read [how to contribute](../CONTRIBUTING.md#setup).\n",
    );
    write(
        r,
        "CONTRIBUTING.md",
        "# Contributing\n\n## Setup\n\nRun the tests.\n",
    );
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

    // The type's definition renders flat: signature, then doc with the hidden line
    // removed and a resolved link.
    let Block::Code { code, .. } = &store.blocks[0] else {
        panic!("type page starts with its signature");
    };
    assert_eq!(code, "pub struct Store");
    let Block::Markdown { markdown } = &store.blocks[1] else {
        panic!("then its docs");
    };
    let doc = &markdown.source;
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
    assert_eq!(names, vec!["len", "new"]);

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

#[test]
fn guides_become_pages_with_resolved_links() {
    let t = library_fixture();
    let site = build_site(&CacheManager::new(t.path()), &opts()).unwrap();
    let usage = &site.pages[&PageId::new("docs/guide/docs/usage.md")];
    assert_eq!(usage.title, "Using kv");
    assert_eq!(usage.route, "/docs/guides/usage/");
    let contributing = &site.pages[&PageId::new("int/guide/CONTRIBUTING.md")];
    assert_eq!(contributing.route, "/internals/contributing/contributing/");
    let Block::Markdown { markdown } = &usage.blocks[0] else {
        panic!("guide body");
    };
    assert!(
        markdown
            .source
            .contains("[how to contribute](/internals/contributing/contributing/#setup)"),
        "{}",
        markdown.source
    );
    // No repository URL: the source link becomes its text.
    assert!(
        markdown.source.contains("Open a `Store`, then"),
        "{}",
        markdown.source
    );
    assert!(
        site.report.broken_links.is_empty(),
        "{:?}",
        site.report.broken_links
    );
}

/// A Go module: two packages, an `internal/` package and a command.
fn go_fixture() -> TempDir {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(r, "go.mod", "module github.com/acme/kv\n\ngo 1.22\n");
    write(
        r,
        "client.go",
        r#"// Package kv is a key-value client.
//
// Create a [Client] with [New] and configure it with [Option] values.
package kv

import "context"

// Client talks to a kv server. Configure it with [Option].
//
// Reads block:
//
//	v, err := c.Get(ctx, "k")
type Client struct {
	// Addr is the server address.
	Addr    string `json:"addr"`
	timeout int
}

// New returns a client for addr.
func New(addr string, opts ...Option) *Client { return &Client{Addr: addr} }

// Get reads key. See [Client.Put] and [context.Context].
func (c *Client) Get(ctx context.Context, key string) (string, error) { return "", nil }

// Put writes key.
func (c *Client) Put(ctx context.Context, key, value string) error { return nil }

func (c *Client) secret() {}
"#,
    );
    write(
        r,
        "option.go",
        "package kv\n\n// Option configures a [Client].\ntype Option func(*Client)\n\n// WithTimeout sets the timeout in seconds.\nfunc WithTimeout(s int) Option { return func(c *Client) { c.timeout = s } }\n\n// Level is a consistency level.\ntype Level int\n\n// Consistency levels.\nconst (\n\tOne Level = iota\n\tQuorum\n)\n\n// String names the level.\nfunc (l Level) String() string { return \"\" }\n",
    );
    write(
        r,
        "codec/codec.go",
        "// Package codec encodes values for a [kv.Client].\npackage codec\n\n// Encode encodes v.\nfunc Encode(v any) []byte { return nil }\n",
    );
    write(
        r,
        "internal/x/x.go",
        "package x\n\n// Hidden is internal.\nfunc Hidden() {}\n",
    );
    write(
        r,
        "cmd/tool/main.go",
        "package main\n\n// Run is a command.\nfunc Run() {}\n\nfunc main() {}\n",
    );
    Indexer::new(CacheManager::new(r), IndexConfig::default())
        .index(r, false)
        .unwrap();
    t
}

fn symbol_names(blocks: &[Block]) -> Vec<&str> {
    blocks
        .iter()
        .filter_map(|b| match b {
            Block::Symbol { symbol } => Some(symbol.name.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn go_reference_pages_symbols_and_links() {
    let t = go_fixture();
    let site = build_site(&CacheManager::new(t.path()), &opts()).unwrap();
    assert!(
        site.report.broken_links.is_empty(),
        "{:?}",
        site.report.broken_links
    );

    let root = &site.pages[&PageId::new("docs/ref/mod/kv")];
    assert_eq!(root.route, "/docs/reference/kv/");
    let Block::Markdown { markdown } = &root.blocks[0] else {
        panic!("package page starts with the package doc");
    };
    let md = &markdown.source;
    assert!(md.starts_with("Import path: `github.com/acme/kv`"), "{md}");
    assert!(
        md.contains("[`Client`](/docs/reference/kv/client/)"),
        "{md}"
    );
    // `Option` has no methods: it is listed on the package page, and links there.
    assert!(
        md.contains("[`Option`](/docs/reference/kv/#type.option)"),
        "{md}"
    );
    assert_eq!(
        symbol_names(&root.blocks),
        vec!["New", "WithTimeout", "One", "Quorum", "Option"]
    );

    let client = &site.pages[&PageId::new("docs/ref/type/kv.Client")];
    assert_eq!(client.route, "/docs/reference/kv/client/");
    let Block::Code { code, lang, .. } = &client.blocks[0] else {
        panic!("type page starts with its signature");
    };
    assert_eq!((code.as_str(), lang.as_str()), ("type Client struct", "go"));
    let Block::Markdown { markdown } = &client.blocks[1] else {
        panic!("then its docs");
    };
    let doc = &markdown.source;
    assert!(
        doc.contains("[`Option`](/docs/reference/kv/#type.option)"),
        "{doc}"
    );
    assert!(
        doc.contains("```go\nv, err := c.Get(ctx, \"k\")\n```"),
        "{doc}"
    );

    // Exported fields and methods (from any file) are documented; unexported are not.
    assert_eq!(symbol_names(&client.blocks), vec!["Addr", "Get", "Put"]);
    let get = client
        .blocks
        .iter()
        .find_map(|b| match b {
            Block::Symbol { symbol } if symbol.name == "Get" => Some(symbol),
            _ => None,
        })
        .unwrap();
    let get_doc = &get.doc.as_ref().unwrap().source;
    assert!(
        get_doc.contains("[`Client.Put`](/docs/reference/kv/client/#method.put)"),
        "{get_doc}"
    );
    assert!(get_doc.contains("`context.Context`"), "{get_doc}");
    assert_eq!(get.params.len(), 2);

    // A named non-struct type with methods gets a page.
    let level = &site.pages[&PageId::new("docs/ref/type/kv.Level")];
    assert_eq!(level.route, "/docs/reference/kv/level/");
    assert_eq!(symbol_names(&level.blocks), vec!["String"]);

    // A sub-package has its own page under the module.
    let codec = &site.pages[&PageId::new("docs/ref/mod/kv/codec")];
    assert_eq!(codec.route, "/docs/reference/kv/codec/");
    let Block::Markdown { markdown } = &codec.blocks[0] else {
        panic!("package doc");
    };
    assert!(
        markdown
            .source
            .starts_with("Import path: `github.com/acme/kv/codec`"),
        "{}",
        markdown.source
    );

    // `internal/` and `package main` directories are not API.
    let ref_pages: Vec<&str> = site
        .pages
        .keys()
        .map(|p| p.0.as_str())
        .filter(|p| p.starts_with("docs/ref/"))
        .collect();
    assert!(
        !ref_pages
            .iter()
            .any(|p| p.contains("internal") || p.contains("tool")),
        "{ref_pages:?}"
    );
    assert!(
        !site
            .symbols
            .values()
            .any(|s| s.name == "Hidden" || s.name == "Run" || s.name == "secret")
    );
    assert!(site.symbols.values().any(|s| s.name == "WithTimeout"));
}

fn python_fixture() -> TempDir {
    let t = TempDir::new().unwrap();
    let r = t.path();
    write(
        r,
        "pyproject.toml",
        "[project]\nname = \"pkg\"\nversion = \"0.1.0\"\n",
    );
    write(
        r,
        "src/pkg/__init__.py",
        r#""""Tools for engines. Start with :class:`Engine`."""
from ._core import Engine
from . import util

__all__ = ["Engine", "connect"]


def connect(url: str) -> Engine:
    """Connect to *url*.

    Returns:
        Engine: A running :class:`Engine`.
    """
    return Engine(url)
"#,
    );
    write(
        r,
        "src/pkg/_core.py",
        r#""""Engine internals."""


class Engine:
    """Runs jobs.

    Use :meth:`Engine.start` to begin; see :func:`pkg.util.slugify`.

    Example:
        >>> e = Engine("x")
        >>> e.start()
    """

    #: The URL.
    url: str

    def __init__(self, url: str) -> None:
        """Create an engine for *url*."""
        self.url = url

    def start(self) -> None:
        """Start it."""

    @classmethod
    def default(cls) -> "Engine":
        """The default engine."""

    def _private(self): ...

    def __repr__(self):
        return "Engine()"
"#,
    );
    write(
        r,
        "src/pkg/util.py",
        r#""""Helpers."""


def slugify(s: str) -> str:
    """Make a slug from *s*. See :class:`~pkg.Engine`."""
    return s
"#,
    );
    write(r, "tests/test_engine.py", "def test_start():\n    pass\n");
    Indexer::new(CacheManager::new(r), IndexConfig::default())
        .index(r, false)
        .unwrap();
    t
}

#[test]
fn python_reference_pages_symbols_and_links() {
    let t = python_fixture();
    let site = build_site(&CacheManager::new(t.path()), &opts()).unwrap();
    assert!(
        site.report.broken_links.is_empty(),
        "{:?}",
        site.report.broken_links
    );

    let root = &site.pages[&PageId::new("docs/ref/mod/pkg")];
    assert_eq!(root.route, "/docs/reference/pkg/");
    assert!(
        site.pages
            .contains_key(&PageId::new("docs/ref/mod/pkg.util"))
    );
    assert!(
        !site
            .pages
            .contains_key(&PageId::new("docs/ref/mod/pkg._core")),
        "private modules have no reference page"
    );
    let engine = &site.pages[&PageId::new("docs/ref/type/pkg.Engine")];
    assert_eq!(engine.route, "/docs/reference/pkg/engine/");
    assert_eq!(engine.badges, vec!["class"]);

    let Block::Code { code, lang, .. } = &engine.blocks[0] else {
        panic!("type page starts with its signature");
    };
    assert_eq!((code.as_str(), lang.as_str()), ("class Engine", "python"));
    let Block::Markdown { markdown } = &engine.blocks[1] else {
        panic!("then its docs");
    };
    let doc = &markdown.source;
    assert!(
        doc.contains("[`Engine.start`](/docs/reference/pkg/engine/#method.start)"),
        "{doc}"
    );
    assert!(
        doc.contains("[`pkg.util.slugify`](/docs/reference/pkg/util/#fn.slugify)"),
        "{doc}"
    );
    assert!(
        doc.contains("```python\n>>> e = Engine(\"x\")\n>>> e.start()\n```"),
        "{doc}"
    );
    let reexported = engine.blocks.iter().any(|b| {
        matches!(b, Block::Callout { body, .. } if body.iter().any(|x| matches!(x,
            Block::Paragraph { content } if content.iter().any(|i| matches!(i,
                Inline::Text { text } if text.contains("defined as pkg._core.Engine"))))))
    });
    assert!(reexported, "{:#?}", engine.blocks);

    // The field, the constructor and public methods; not `_private` or `__repr__`.
    let symbols: Vec<(&str, &str, Vec<&str>)> = engine
        .blocks
        .iter()
        .filter_map(|b| match b {
            Block::Symbol { symbol } => Some((
                symbol.name.as_str(),
                symbol.kind.as_str(),
                symbol.badges.iter().map(String::as_str).collect(),
            )),
            _ => None,
        })
        .collect();
    assert_eq!(
        symbols,
        vec![
            ("url", "field", vec![]),
            ("__init__", "method", vec!["constructor"]),
            ("start", "method", vec![]),
            ("default", "method", vec!["classmethod"]),
        ]
    );

    // Package docs, and a `~`-shortened role in another module, link to the class page.
    let Block::Markdown { markdown } = &root.blocks[0] else {
        panic!("root page starts with the package docs");
    };
    assert!(
        markdown
            .source
            .contains("[`Engine`](/docs/reference/pkg/engine/)"),
        "{}",
        markdown.source
    );
    let util = &site.pages[&PageId::new("docs/ref/mod/pkg.util")];
    let slugify = util
        .blocks
        .iter()
        .find_map(|b| match b {
            Block::Symbol { symbol } if symbol.name == "slugify" => Some(symbol),
            _ => None,
        })
        .expect("slugify is documented");
    assert!(
        slugify
            .doc
            .as_ref()
            .unwrap()
            .source
            .contains("[`Engine`](/docs/reference/pkg/engine/)"),
        "{:?}",
        slugify.doc
    );
    assert_eq!(slugify.signature, "def slugify(s: str) -> str");

    let connect = site
        .symbols
        .values()
        .find(|s| s.name == "connect")
        .expect("connect is documented");
    assert_eq!(connect.path, "pkg.connect");
    assert!(!site.symbols.values().any(|s| s.name == "test_start"));
}
