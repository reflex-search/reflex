//! The Pulse Docs Model: a renderer-independent description of the whole site.
//!
//! ```text
//! Site ─┬─ meta (title, repo permalinks)
//!       ├─ tabs: Docs | Internals ── nav tree of PageIds
//!       ├─ pages: PageId → Page { route, kind, blocks }
//!       ├─ facts: FactId → Fact (every number, with provenance)
//!       └─ report (broken links, excluded files)
//! ```
//!
//! Builders in [`crate::pulse::build`] produce it from the index; the writing pass fills
//! its narrative slots; a renderer turns it into files. The JSON form
//! (`rfx pulse model --json`) is deterministic: no timestamps, sorted maps.

pub mod content;
pub mod facts;
pub mod ids;
pub mod xref;

pub use content::{Block, Card, Inline, MarkdownOrigin, MarkdownText, ParamRow, Stat, SymbolBlock};
pub use facts::{Fact, FactStore, FactValue, Provenance, Subject};
pub use ids::{AnchorId, FactId, ModuleId, PageId, SlugAllocator, SymbolId};
pub use xref::{Linker, RepoInfo, SourceLoc, Target};

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Bump when the JSON shape changes incompatibly.
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Site {
    pub schema_version: u32,
    pub meta: SiteMeta,
    pub tabs: Vec<Tab>,
    pub pages: BTreeMap<PageId, Page>,
    /// Every documented symbol and where it is documented.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub symbols: BTreeMap<SymbolId, SymbolEntry>,
    pub facts: FactStore,
    pub report: BuildReport,
}

/// Where a symbol is documented.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymbolEntry {
    pub name: String,
    /// Language path (`reflex::query::QueryEngine`).
    pub path: String,
    pub kind: String,
    pub page: PageId,
    pub anchor: Option<AnchorId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SiteMeta {
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo: Option<RepoInfo>,
    /// Version of rfx that built the model.
    pub generator: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TabId {
    /// For people who use the code.
    Docs,
    /// For people who work on the code.
    Internals,
}

impl TabId {
    /// Route prefix of the tab's pages.
    pub fn prefix(&self) -> &'static str {
        match self {
            TabId::Docs => "/docs/",
            TabId::Internals => "/internals/",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tab {
    pub id: TabId,
    pub label: String,
    pub landing: PageId,
    pub nav: Vec<NavNode>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum NavNode {
    Page {
        page: PageId,
    },
    Group {
        label: String,
        #[serde(default)]
        collapsed: bool,
        children: Vec<NavNode>,
    },
}

impl NavNode {
    pub fn page(id: &PageId) -> Self {
        NavNode::Page { page: id.clone() }
    }
}

pub(crate) fn collect_nav_pages(nav: &[NavNode], out: &mut Vec<PageId>) {
    for n in nav {
        match n {
            NavNode::Page { page } => out.push(page.clone()),
            NavNode::Group { children, .. } => collect_nav_pages(children, out),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Page {
    pub id: PageId,
    pub tab: TabId,
    pub kind: PageKind,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Base-less route, e.g. `/internals/modules/src-pulse/`. Always ends with `/`.
    pub route: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub badges: Vec<String>,
    pub blocks: Vec<Block>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum PageKind {
    /// Site home (Docs tab landing).
    Landing,
    /// Internals tab landing: how the system is put together.
    Architecture,
    /// One module (a directory of source files).
    Module {
        module: ModuleId,
    },
    /// All modules and their dependency graph.
    DependencyMap,
    Changelog,
    Glossary,
    /// A page built from a repository document.
    Guide {
        source: String,
    },
    /// A library module in the API reference.
    ReferenceModule {
        path: String,
    },
    /// A type (struct, enum, trait) in the API reference.
    ReferenceType {
        path: String,
    },
    /// A command in the CLI reference.
    CliCommand {
        command: String,
    },
}

/// What the builder noticed while building.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BuildReport {
    /// Indexed files per role that produced no reference content (tests, fixtures, …).
    pub files_by_role: BTreeMap<String, usize>,
    pub broken_links: Vec<(PageId, Target)>,
    pub warnings: Vec<String>,
}

impl Site {
    /// Page ids in nav order for one tab, landing first.
    pub fn nav_order(&self, tab: TabId) -> Vec<PageId> {
        let Some(t) = self.tabs.iter().find(|t| t.id == tab) else {
            return Vec::new();
        };
        let mut ids = vec![t.landing.clone()];
        collect_nav_pages(&t.nav, &mut ids);
        ids
    }

    /// Fill narrative slots from the writing pass. `text(slot)` returns the markdown for
    /// a slot or `None` to keep the structural fallback.
    pub fn fill_narratives(&mut self, mut text: impl FnMut(&str) -> Option<String>) -> usize {
        fn fill(blocks: &mut [Block], text: &mut dyn FnMut(&str) -> Option<String>) -> usize {
            let mut n = 0;
            for b in blocks {
                match b {
                    Block::Narrative { slot, text: t, .. } => {
                        if let Some(md) = text(slot) {
                            *t = Some(MarkdownText {
                                source: md,
                                origin: MarkdownOrigin::Llm,
                                from: None,
                            });
                            n += 1;
                        }
                    }
                    Block::Callout { body, .. } => n += fill(body, text),
                    _ => {}
                }
            }
            n
        }
        self.pages
            .values_mut()
            .map(|p| fill(&mut p.blocks, &mut text))
            .sum()
    }

    /// Every narrative slot id on the site, sorted.
    pub fn narrative_slots(&self) -> Vec<String> {
        let mut out = Vec::new();
        for p in self.pages.values() {
            for b in &p.blocks {
                if let Block::Narrative { slot, .. } = b {
                    out.push(slot.clone());
                }
            }
        }
        out.sort();
        out.dedup();
        out
    }

    /// Record every link that does not resolve in `report.broken_links`.
    pub fn check_links(&mut self) {
        let broken = Linker::new(self).broken();
        self.report.broken_links = broken;
    }

    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
}
