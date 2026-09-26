//! Build the Docs Model ([`Site`]) from the index.
//!
//! ```text
//! Corpus (files, roles, edges, README) ─┐
//! ModuleGraph (modules, edges, cycles) ─┼─► facts ─► Internals tab ─► Docs tab ─► Site
//! git history ──────────────────────────┘
//! ```
//! Nothing here calls an LLM. Narrative blocks carry structural fallbacks; the writing
//! pass fills them later by slot id.

pub mod docs_tab;
pub mod internals_tab;
pub mod modules;

use crate::cache::CacheManager;
use crate::pulse::extract::Corpus;
use crate::pulse::model::{
    Block, BuildReport, FactId, FactStore, Inline, NavNode, Page, PageId, PageKind, RepoInfo,
    SCHEMA_VERSION, Site, SiteMeta, SlugAllocator, Subject, Tab, TabId,
};
use anyhow::Result;
use modules::ModuleGraph;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct BuildOptions {
    pub title: String,
    /// 1 = top-level modules only, 2 = also subdirectories with 3+ source files.
    pub max_depth: u8,
    pub min_files: usize,
    /// Recent commits on the changelog page.
    pub changelog_commits: usize,
    /// Where slugs persist between runs; `None` keeps them in memory only.
    pub slugs_path: Option<PathBuf>,
    /// Detect the git remote for source permalinks.
    pub detect_repo: bool,
}

impl BuildOptions {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            max_depth: 2,
            min_files: 1,
            changelog_commits: 50,
            slugs_path: None,
            detect_repo: true,
        }
    }
}

/// Build the model for the index in `cache`.
pub fn build_site(cache: &CacheManager, opts: &BuildOptions) -> Result<Site> {
    let corpus = Corpus::load(cache)?;
    let graph = ModuleGraph::build(&corpus, opts.max_depth, opts.min_files);
    let commits =
        crate::pulse::changelog::extract_changelog_commits(&corpus.root, opts.changelog_commits)
            .map(|(c, _)| c)
            .unwrap_or_default();
    let repo = opts
        .detect_repo
        .then(|| RepoInfo::detect(&corpus.root))
        .flatten();

    let slugs = match &opts.slugs_path {
        Some(p) => SlugAllocator::load(p),
        None => SlugAllocator::default(),
    };
    let mut b = SiteBuilder::new(opts.title.clone(), repo, slugs);
    b.report.files_by_role = corpus.role_counts();
    compute_facts(&mut b.facts, &corpus, &graph);

    internals_tab::build(&mut b, &corpus, &graph);
    docs_tab::build(&mut b, &corpus, &graph, &commits);

    let (site, slugs) = b.finish();
    if let Some(p) = &opts.slugs_path
        && let Err(e) = slugs.save(p)
    {
        log::warn!("could not save {}: {e}", p.display());
    }
    Ok(site)
}

/// Collects pages, facts and nav while the tabs are built.
pub struct SiteBuilder {
    title: String,
    repo: Option<RepoInfo>,
    slugs: SlugAllocator,
    pub facts: FactStore,
    pub pages: BTreeMap<PageId, Page>,
    pub tabs: Vec<Tab>,
    pub report: BuildReport,
}

impl SiteBuilder {
    pub fn new(title: String, repo: Option<RepoInfo>, slugs: SlugAllocator) -> Self {
        Self {
            title,
            repo,
            slugs,
            facts: FactStore::default(),
            pages: BTreeMap::new(),
            tabs: Vec::new(),
            report: BuildReport::default(),
        }
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    pub fn repo(&self) -> Option<&RepoInfo> {
        self.repo.as_ref()
    }

    /// Add a page. `slug` is `None` for a tab landing (`/` or `/internals/`), else
    /// slash-separated, already-slugified segments under the tab prefix.
    pub fn add_page(&mut self, spec: PageSpec) -> PageId {
        let route = match (&spec.slug, spec.tab) {
            (None, TabId::Docs) => "/".to_string(),
            (None, tab) => tab.prefix().to_string(),
            (Some(wanted), tab) => {
                let slug = self.slugs.assign(&spec.id, wanted);
                format!("{}{slug}/", tab.prefix())
            }
        };
        let id = spec.id.clone();
        self.pages.insert(
            id.clone(),
            Page {
                id: spec.id,
                tab: spec.tab,
                kind: spec.kind,
                title: spec.title,
                description: spec.description,
                route,
                badges: spec.badges,
                blocks: spec.blocks,
            },
        );
        id
    }

    pub fn add_tab(&mut self, id: TabId, label: &str, landing: PageId, nav: Vec<NavNode>) {
        self.tabs.push(Tab {
            id,
            label: label.to_string(),
            landing,
            nav,
        });
    }

    /// A fact id, asserting the fact exists (a missing fact is a builder bug).
    pub fn fact(&self, subject: &Subject, key: &str) -> FactId {
        let id = FactStore::id_for(subject, key);
        debug_assert!(self.facts.contains(&id), "missing fact {id}");
        id
    }

    pub fn fact_inline(&self, subject: &Subject, key: &str) -> Inline {
        Inline::fact(self.fact(subject, key))
    }

    fn finish(self) -> (Site, SlugAllocator) {
        let mut tabs = self.tabs;
        tabs.sort_by_key(|t| t.id);
        let mut site = Site {
            schema_version: SCHEMA_VERSION,
            meta: SiteMeta {
                title: self.title,
                description: None,
                repo: self.repo,
                generator: format!("rfx {}", env!("CARGO_PKG_VERSION")),
            },
            tabs,
            pages: self.pages,
            facts: self.facts,
            report: self.report,
        };
        site.check_links();
        (site, self.slugs)
    }
}

/// Arguments for [`SiteBuilder::add_page`].
pub struct PageSpec {
    pub id: PageId,
    pub tab: TabId,
    pub kind: PageKind,
    pub title: String,
    pub description: Option<String>,
    pub slug: Option<String>,
    pub badges: Vec<String>,
    pub blocks: Vec<Block>,
}

/// Every number the site states, computed once.
fn compute_facts(facts: &mut FactStore, corpus: &Corpus, graph: &ModuleGraph) {
    use crate::pulse::extract::FileRole;
    let source: Vec<_> = corpus.with_role(FileRole::Source).collect();
    let q = |s: &str| s.to_string();
    facts.count(
        Subject::Site,
        "source_files",
        source.len() as u64,
        &q("files table, role = source"),
    );
    facts.count(
        Subject::Site,
        "source_lines",
        source.iter().map(|(_, f)| f.lines).sum(),
        &q("sum(line_count), role = source"),
    );
    facts.count(
        Subject::Site,
        "indexed_files",
        corpus.files.len() as u64,
        &q("files table"),
    );
    facts.count(
        Subject::Site,
        "modules",
        graph.modules.len() as u64,
        &q("module detection over source files"),
    );
    let mut langs: BTreeMap<String, usize> = BTreeMap::new();
    for (_, f) in &source {
        *langs
            .entry(crate::pulse::extract::language_name(f.language))
            .or_insert(0) += 1;
    }
    facts.count(
        Subject::Site,
        "languages",
        langs.len() as u64,
        &q("distinct languages, role = source"),
    );
    facts.count(
        Subject::Site,
        "module_cycles",
        graph.cycles().len() as u64,
        &q("strongly connected module groups"),
    );

    for (mi, m) in graph.modules.iter().enumerate() {
        let subject = Subject::Module(m.id.clone());
        let sub_files: usize = graph
            .children(mi)
            .iter()
            .map(|&c| graph.modules[c].files.len())
            .sum();
        facts.count(
            subject.clone(),
            "files",
            m.files.len() as u64,
            &q("source files owned by the module"),
        );
        facts.count(
            subject.clone(),
            "files_with_submodules",
            (m.files.len() + sub_files) as u64,
            &q("source files in the module and its submodules"),
        );
        facts.count(subject.clone(), "lines", m.lines, &q("sum(line_count)"));
        facts.count(
            subject.clone(),
            "dependencies",
            graph.dependencies(mi).len() as u64,
            &q("modules imported from (resolved imports)"),
        );
        facts.count(
            subject,
            "dependents",
            graph.dependents(mi).len() as u64,
            &q("modules importing it (resolved imports)"),
        );
    }
}

/// Page id of a module page.
pub fn module_page_id(module: &crate::pulse::model::ModuleId) -> PageId {
    PageId(format!("int/module/{module}"))
}

/// Slug of a module page under the Internals prefix.
pub fn module_slug(module: &crate::pulse::model::ModuleId) -> String {
    if module.as_str() == "." {
        return "modules/root".into();
    }
    let segs: Vec<String> = module
        .as_str()
        .split('/')
        .map(crate::pulse::model::ids::slugify)
        .collect();
    format!("modules/{}", segs.join("/"))
}

#[cfg(test)]
mod tests;
