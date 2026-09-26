//! `rfx pulse generate`: index → static documentation site.
//!
//! 1. **Model.** [`build::build_site`] turns the index into the Docs Model.
//! 2. **Write.** The LLM pass fills narrative slots (home overview, architecture,
//!    module summaries) from each page's own structural text; every slot keeps its
//!    structural fallback, so `--llm off` still yields a complete site.
//! 3. **Render.** Every page becomes sanitised HTML in a bundle next to the embedded
//!    Starlight template, staged under the shared runtime.
//! 4. **Build and publish.** `astro build`, then the static output replaces `-o`.
//!    `--no-build` stops after staging and prints where the project is.

use super::build::{self, BuildOptions};
use super::model::{Block, Page, PageKind, Site};
use super::render::bundle::{BaseUrl, BundleStats, write_bundle};
use super::write::{LlmMode, WriteOptions, WriteSession, WriteTask};
use super::{narrate, publish, runtime};
use crate::cache::CacheManager;
use anyhow::Result;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct SiteConfig {
    pub output_dir: PathBuf,
    pub base_url: String,
    pub title: String,
    /// LLM writing pass options (mode, force scope, budget, cache).
    pub write: WriteOptions,
    pub clean: bool,
    /// Module discovery depth (1 = top level only, 2 = default).
    pub max_depth: u8,
    pub min_files: usize,
    /// Stage the site project but do not install the runtime or build.
    pub no_build: bool,
    /// Never download or install anything.
    pub offline: bool,
    /// Stream the full `astro build` output.
    pub verbose_build: bool,
}

impl Default for SiteConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("pulse-site"),
            base_url: "/".into(),
            title: "Documentation".into(),
            write: WriteOptions::off(),
            clean: false,
            max_depth: 2,
            min_files: 1,
            no_build: false,
            offline: false,
            verbose_build: false,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SiteReport {
    pub output_dir: String,
    pub pages: usize,
    pub docs_pages: usize,
    pub internals_pages: usize,
    pub broken_links: usize,
    pub narrated_sections: usize,
    pub narration_mode: String,
    /// `built`, `staged` (--no-build) or `dry-run`.
    pub build: String,
    pub project_dir: Option<String>,
    pub build_seconds: f64,
    pub total_seconds: f64,
    #[serde(skip)]
    pub bundle: BundleStats,
}

/// Page text sent as context for a slot, at most this many characters.
const CONTEXT_CHARS: usize = 14_000;

/// Writing tasks for every narrative slot, with each page's text as context.
pub fn narrative_tasks(site: &Site) -> Vec<WriteTask> {
    let mut tasks = Vec::new();
    for page in site.pages.values() {
        for b in &page.blocks {
            let Block::Narrative { slot, .. } = b else {
                continue;
            };
            let ctx = super::model::text::page_text(site, page, CONTEXT_CHARS);
            let task = if slot == narrate::ids::OVERVIEW {
                narrate::overview_task(&overview_context(site, &ctx))
            } else if slot == narrate::ids::ARCHITECTURE {
                narrate::architecture_task(&ctx)
            } else if let Some(module) = slot.strip_prefix("module:") {
                narrate::wiki_task(module, &ctx)
            } else {
                continue;
            };
            tasks.push(task);
        }
    }
    tasks
}

/// The home page is thin on its own; add the architecture page's text.
fn overview_context(site: &Site, home: &str) -> String {
    let arch: Option<&Page> = site
        .pages
        .values()
        .find(|p| matches!(p.kind, PageKind::Architecture));
    match arch {
        Some(a) => format!(
            "{home}\n\n{}",
            super::model::text::page_text(site, a, CONTEXT_CHARS / 2)
        ),
        None => home.to_string(),
    }
}

/// Generate the site.
pub fn generate_site(cache: &CacheManager, config: &SiteConfig) -> Result<SiteReport> {
    let start = Instant::now();
    let workspace = cache
        .path()
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    // 1. Model
    let docs = super::config::load_pulse_config(cache.path())?.docs;
    let opts = BuildOptions {
        max_depth: config.max_depth,
        min_files: config.min_files,
        slugs_path: Some(cache.path().join("pulse").join("slugs.json")),
        reference: build::reference::ReferenceOptions {
            skip_library: !docs.library,
            include: docs.include,
        },
        ..BuildOptions::new(config.title.clone())
    };
    let mut site = build::build_site(cache, &opts)?;
    eprintln!(
        "Model: {} pages, {} symbols, {} broken links ({:.1}s)",
        site.pages.len(),
        site.symbols.len(),
        site.report.broken_links.len(),
        start.elapsed().as_secs_f64()
    );

    // 2. Write
    let mut narrated = 0;
    let mode = config.write.mode;
    if mode != LlmMode::Off {
        let session = WriteSession::open(cache.path(), &config.write);
        match (&session.llm.provider, &session.llm.unavailable) {
            (Some(p), _) => eprintln!("LLM writing enabled ({}/{}).", p.name(), p.model()),
            (None, Some(reason)) => eprintln!("LLM writing: {reason}"),
            _ => {}
        }
        let tasks = narrative_tasks(&site);
        let outcome = session.run(tasks);
        if config.write.dry_run {
            eprintln!("{}", outcome.plan_table());
            return Ok(SiteReport {
                output_dir: config.output_dir.display().to_string(),
                pages: site.pages.len(),
                docs_pages: 0,
                internals_pages: 0,
                broken_links: site.report.broken_links.len(),
                narrated_sections: 0,
                narration_mode: "dry-run".into(),
                build: "dry-run".into(),
                project_dir: None,
                build_seconds: 0.0,
                total_seconds: start.elapsed().as_secs_f64(),
                bundle: BundleStats::default(),
            });
        }
        eprintln!("  {}", outcome.summary());
        narrated = site.fill_narratives(|slot| outcome.text(slot).map(str::to_string));
    }

    // 3. Render + stage
    let base = BaseUrl::parse(&config.base_url);
    let (rt, site_dir) = if config.no_build {
        (None, cache.path().join("pulse").join("site"))
    } else {
        let rt = runtime::ensure(config.offline)?;
        let dir = rt.site_dir(&workspace);
        (Some(rt), dir)
    };
    runtime::stage_template(&site_dir)?;
    let bundle = write_bundle(&site, &site_dir, &base)?;
    eprintln!(
        "Rendered {} pages into {} ({} written, {} unchanged, {} removed)",
        bundle.pages,
        site_dir.display(),
        bundle.written,
        bundle.unchanged,
        bundle.removed
    );

    let count = |tab| site.pages.values().filter(|p| p.tab == tab).count();
    let mut report = SiteReport {
        output_dir: config.output_dir.display().to_string(),
        pages: site.pages.len(),
        docs_pages: count(super::model::TabId::Docs),
        internals_pages: count(super::model::TabId::Internals),
        broken_links: site.report.broken_links.len(),
        narrated_sections: narrated,
        narration_mode: match (mode, narrated) {
            (LlmMode::Off, _) => "disabled".into(),
            (_, 0) => "structural".into(),
            _ => "narrated".into(),
        },
        build: "staged".into(),
        project_dir: Some(site_dir.display().to_string()),
        build_seconds: 0.0,
        total_seconds: 0.0,
        bundle,
    };

    // 4. Build + publish
    if let Some(rt) = rt {
        publish::check_target(&config.output_dir, &workspace, config.clean)?;
        eprintln!(
            "Building the site with Astro (node {}.{}.{})…",
            rt.node.version.0, rt.node.version.1, rt.node.version.2
        );
        let outcome = runtime::astro_build(&rt, &site_dir, config.verbose_build)?;
        report.build_seconds = outcome.elapsed.as_secs_f64();
        let meta = serde_json::json!({
            "base": base.base,
            "site": base.site,
            "generator": site.meta.generator,
            "pages": site.pages.len(),
            "template": runtime::template::TEMPLATE_HASH,
        });
        let files = publish::publish(
            &outcome.dist,
            &config.output_dir,
            &workspace,
            config.clean,
            &meta,
        )?;
        eprintln!("Published {files} files to {}", config.output_dir.display());
        report.build = "built".into();
    }
    report.total_seconds = start.elapsed().as_secs_f64();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::IndexConfig;
    use crate::{CacheManager, Indexer};

    #[test]
    fn slots_become_tasks_with_page_context() {
        let t = tempfile::TempDir::new().unwrap();
        let r = t.path();
        std::fs::create_dir_all(r.join("src/api")).unwrap();
        std::fs::write(r.join("README.md"), "# Demo\n\nDemo stores values.\n").unwrap();
        std::fs::write(r.join("src/main.rs"), "mod api;\nfn main() {}\n").unwrap();
        for f in ["a", "b", "c"] {
            std::fs::write(r.join(format!("src/api/{f}.rs")), "pub fn f() {}\n").unwrap();
        }
        Indexer::new(CacheManager::new(r), IndexConfig::default())
            .index(r, false)
            .unwrap();
        let site = build::build_site(
            &CacheManager::new(r),
            &BuildOptions {
                detect_repo: false,
                ..BuildOptions::new("Demo")
            },
        )
        .unwrap();
        let tasks = narrative_tasks(&site);
        let ids: Vec<&str> = tasks.iter().map(|t| t.id.as_str()).collect();
        assert!(ids.contains(&"project-overview"), "{ids:?}");
        assert!(ids.contains(&"architecture"), "{ids:?}");
        assert!(ids.contains(&"module:src/api"), "{ids:?}");
        let overview = tasks.iter().find(|t| t.id == "project-overview").unwrap();
        assert!(
            overview.user.contains("Demo stores values."),
            "{}",
            overview.user
        );
        assert!(
            overview.user.contains("# Architecture"),
            "includes the architecture page"
        );
    }
}
