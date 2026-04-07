//! Static site generator
//!
//! Orchestrates wiki, digest, and map into a self-contained HTML site.
//! Uses Tera templates embedded in the binary via `include_str!`.

use anyhow::{Context, Result};
use serde::Serialize;
use std::path::PathBuf;
use tera::Tera;

use crate::cache::CacheManager;
use super::digest;
use super::diff;
use super::map::{self, MapFormat, MapZoom};
use super::snapshot;
use super::wiki;

/// Site generation configuration
#[derive(Debug, Clone)]
pub struct SiteConfig {
    pub output_dir: PathBuf,
    pub base_url: String,
    pub title: String,
    pub surfaces: Vec<Surface>,
    pub no_llm: bool,
    pub clean: bool,
    pub force_renarrate: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Surface {
    Wiki,
    Digest,
    Map,
}

impl Default for SiteConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("pulse-site"),
            base_url: "/".to_string(),
            title: "Pulse".to_string(),
            surfaces: vec![Surface::Wiki, Surface::Digest, Surface::Map],
            no_llm: true,
            clean: false,
            force_renarrate: false,
        }
    }
}

/// Report from site generation
#[derive(Debug, Clone, Serialize)]
pub struct SiteReport {
    pub output_dir: String,
    pub pages_generated: usize,
    pub digest_generated: bool,
    pub map_generated: bool,
}

// Embedded templates
const BASE_TEMPLATE: &str = include_str!("templates/base.html");
const INDEX_TEMPLATE: &str = include_str!("templates/index.html");
const WIKI_PAGE_TEMPLATE: &str = include_str!("templates/wiki_page.html");
const DIGEST_TEMPLATE: &str = include_str!("templates/digest.html");
const MAP_TEMPLATE: &str = include_str!("templates/map.html");
const STYLE_CSS: &str = include_str!("templates/style.css");

/// Generate the complete static site
pub fn generate_site(cache: &CacheManager, config: &SiteConfig) -> Result<SiteReport> {
    // Clean output dir if requested
    if config.clean && config.output_dir.exists() {
        std::fs::remove_dir_all(&config.output_dir)
            .context("Failed to clean output directory")?;
    }

    std::fs::create_dir_all(&config.output_dir)
        .context("Failed to create output directory")?;

    // Set up Tera
    let mut tera = Tera::default();
    tera.add_raw_template("base.html", BASE_TEMPLATE)?;
    tera.add_raw_template("index.html", INDEX_TEMPLATE)?;
    tera.add_raw_template("wiki_page.html", WIKI_PAGE_TEMPLATE)?;
    tera.add_raw_template("digest.html", DIGEST_TEMPLATE)?;
    tera.add_raw_template("map.html", MAP_TEMPLATE)?;

    // Get snapshots for diff
    let snapshots = snapshot::list_snapshots(cache)?;
    let current_snapshot = snapshots.first();
    let baseline_snapshot = snapshots.get(1);

    let snapshot_diff = match (current_snapshot, baseline_snapshot) {
        (Some(current), Some(baseline)) => {
            let pulse_config = super::config::load_pulse_config(cache.path())?;
            diff::compute_diff(&baseline.path, &current.path, &pulse_config.thresholds).ok()
        }
        _ => None,
    };

    let mut pages_generated = 0;
    let mut digest_generated = false;
    let mut map_generated = false;

    // Generate wiki pages
    if config.surfaces.contains(&Surface::Wiki) {
        let wiki_dir = config.output_dir.join("wiki");
        std::fs::create_dir_all(&wiki_dir)?;

        let wiki_pages = wiki::generate_all_pages(cache, snapshot_diff.as_ref(), config.no_llm)?;

        for page in &wiki_pages {
            let mut ctx = tera::Context::new();
            ctx.insert("title", &page.title);
            ctx.insert("base_url", &config.base_url);
            ctx.insert("site_title", &config.title);
            ctx.insert("page", &page);
            ctx.insert("wiki_pages", &wiki_pages);

            let html = tera.render("wiki_page.html", &ctx)?;
            let filename = page.module_path.replace('/', "_") + ".html";
            std::fs::write(wiki_dir.join(&filename), html)?;
            pages_generated += 1;
        }

        // Generate index page with wiki page list
        let mut ctx = tera::Context::new();
        ctx.insert("title", &config.title);
        ctx.insert("base_url", &config.base_url);
        ctx.insert("site_title", &config.title);
        ctx.insert("wiki_pages", &wiki_pages);
        ctx.insert("has_digest", &config.surfaces.contains(&Surface::Digest));
        ctx.insert("has_map", &config.surfaces.contains(&Surface::Map));

        let html = tera.render("index.html", &ctx)?;
        std::fs::write(config.output_dir.join("index.html"), html)?;
    }

    // Generate digest
    if config.surfaces.contains(&Surface::Digest) {
        if let Some(current) = current_snapshot {
            let digest_data = digest::generate_digest(
                snapshot_diff.as_ref(),
                current,
                config.no_llm,
            )?;

            let mut ctx = tera::Context::new();
            ctx.insert("title", "Digest");
            ctx.insert("base_url", &config.base_url);
            ctx.insert("site_title", &config.title);
            ctx.insert("digest", &digest_data);
            ctx.insert("digest_markdown", &digest::render_markdown(&digest_data));

            let html = tera.render("digest.html", &ctx)?;
            std::fs::write(config.output_dir.join("digest.html"), html)?;
            digest_generated = true;
        }
    }

    // Generate map
    if config.surfaces.contains(&Surface::Map) {
        let map_content = map::generate_map(cache, &MapZoom::Repo, MapFormat::Mermaid)?;

        let mut ctx = tera::Context::new();
        ctx.insert("title", "Architecture Map");
        ctx.insert("base_url", &config.base_url);
        ctx.insert("site_title", &config.title);
        ctx.insert("mermaid_content", &map_content);

        let html = tera.render("map.html", &ctx)?;
        std::fs::write(config.output_dir.join("map.html"), html)?;
        map_generated = true;
    }

    // Write CSS
    std::fs::write(config.output_dir.join("style.css"), STYLE_CSS)?;

    Ok(SiteReport {
        output_dir: config.output_dir.display().to_string(),
        pages_generated,
        digest_generated,
        map_generated,
    })
}
