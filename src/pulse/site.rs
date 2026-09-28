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
use super::model::Site;
use super::render::bundle::{BaseUrl, BundleStats, write_bundle};
use super::write::{LlmMode, WriteOptions, WriteSession, WriteTask};
use super::write::{contract, gate};
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

/// Packs below this groundability are not sent: they would produce generic prose.
const MIN_GROUNDABILITY: f32 = 0.2;

/// Grounded writing tasks for every narrative slot with enough evidence.
pub fn narrative_tasks(site: &Site) -> Vec<WriteTask> {
    let slots: std::collections::BTreeSet<String> = site.narrative_slots().into_iter().collect();
    site.evidence
        .values()
        .filter(|p| slots.contains(&p.slot))
        .filter(|p| p.groundability() >= MIN_GROUNDABILITY && p.descriptive_count() >= 1)
        .map(narrate::grounded_task)
        .collect()
}

/// Verify each answer and turn the kept sentences into Markdown with sources.
fn grounded_texts(
    site: &Site,
    outcome: &super::write::WriteOutcome,
) -> (
    std::collections::BTreeMap<String, String>,
    Vec<gate::GateReport>,
) {
    let mut known = gate::KnownNames::default();
    for n in &site.known_names {
        known.insert(n);
    }
    let mut texts = std::collections::BTreeMap::new();
    let mut reports = Vec::new();
    for (slot, pack) in &site.evidence {
        let Some(raw) = outcome.text(slot) else {
            continue;
        };
        let min_sentences = if slot.starts_with("module:") { 2 } else { 3 };
        let cfg = gate::GateConfig {
            min_sentences,
            ..gate::GateConfig::default()
        };
        let verified = contract::parse(raw)
            .map_err(|e| gate::GateReport {
                slot: slot.clone(),
                rejected: Some(format!("unparseable answer: {e}")),
                ..Default::default()
            })
            .and_then(|c| gate::verify(&c, pack, &known, &cfg));
        match verified {
            Ok(v) => {
                let mut md: Vec<String> = v
                    .paragraphs
                    .iter()
                    .map(|p| {
                        p.iter()
                            .map(|s| s.text.as_str())
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .collect();
                let mut cited: Vec<usize> = v
                    .paragraphs
                    .iter()
                    .flatten()
                    .flat_map(|s| s.cites.clone())
                    .collect();
                cited.sort_unstable();
                cited.dedup();
                let sources: Vec<String> = cited
                    .iter()
                    .filter_map(|&i| pack.items.get(i)?.source.clone())
                    .map(|loc| {
                        let anchor = match (loc.start, loc.end) {
                            (0, _) => String::new(),
                            (s, e) if e > s => format!("#L{s}-L{e}"),
                            (s, _) => format!("#L{s}"),
                        };
                        format!(
                            "[{}]({}{}{anchor})",
                            loc.label(),
                            super::build::links::SOURCE_SCHEME,
                            loc.path
                        )
                    })
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect();
                if !sources.is_empty() {
                    md.push(format!("Sources: {}.", sources.join(", ")));
                }
                if v.confidence == gate::Confidence::Low {
                    md.push("_Low confidence: parts of this section could not be verified and were removed._".into());
                }
                texts.insert(slot.clone(), md.join("\n\n"));
                reports.push(v.report);
            }
            Err(r) => reports.push(r),
        }
    }
    (texts, reports)
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
        let (texts, reports) = grounded_texts(&site, &outcome);
        let dropped: usize = reports.iter().map(|r| r.dropped.len()).sum();
        let rejected = reports.iter().filter(|r| r.rejected.is_some()).count();
        let kept: usize = reports.iter().map(|r| r.kept).sum();
        eprintln!(
            "  Grounding: {kept} sentences kept, {dropped} dropped, {rejected} section(s) fell back to structure"
        );
        if config.write.explain {
            for r in &reports {
                if let Some(why) = &r.rejected {
                    eprintln!("    [{}] fell back: {why}", r.slot);
                }
                for (sentence, reason) in &r.dropped {
                    eprintln!("    [{}] dropped ({:?}): {sentence}", r.slot, reason);
                }
            }
        }
        let report_dir = cache.path().join("pulse").join("reports");
        if std::fs::create_dir_all(&report_dir).is_ok() {
            let _ = std::fs::write(
                report_dir.join("write-report.json"),
                serde_json::to_vec_pretty(&reports).unwrap_or_default(),
            );
        }
        narrated = site.fill_narratives(|slot| texts.get(slot).cloned());
        super::build::links::resolve_markdown_links(&mut site);
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
    fn slots_become_grounded_tasks() {
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
        // `src/api` has only undocumented functions: still descriptive (items).
        let module = tasks.iter().find(|t| t.id == "module:src/api").unwrap();
        assert!(module.user.contains("pub fn f()"), "{}", module.user);
        let overview = tasks.iter().find(|t| t.id == "project-overview").unwrap();
        assert!(
            overview.user.contains("Demo stores values."),
            "{}",
            overview.user
        );
        assert!(
            overview.user.contains("<facts>\n[F1]"),
            "numbered facts: {}",
            overview.user
        );
        assert!(overview.system.contains("cites 1-3 facts"));
    }
}
