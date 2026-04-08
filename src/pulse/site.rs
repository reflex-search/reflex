//! Static site generator
//!
//! Orchestrates wiki, digest, and map into a Zola project.
//! Generates markdown content with TOML front matter, Tera templates,
//! and a Zola config. Optionally runs `zola build` to produce HTML.

use anyhow::{Context, Result};
use rusqlite::Connection;
use serde::Serialize;
use std::path::{Path, PathBuf};

use crate::cache::CacheManager;
use crate::semantic::providers::LlmProvider;
use super::digest;
use super::diff;
use super::map::{self, MapFormat, MapZoom};
use super::narrate;
use super::snapshot;
use super::wiki;
use super::zola;

/// Truncate a string to at most `max_chars` Unicode characters, appending "..." if truncated.
fn truncate_str(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{}...", truncated)
    } else {
        truncated
    }
}

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
    pub narration_mode: String,
    pub build_success: bool,
}

/// Generate the complete Zola project and optionally build it
pub fn generate_site(cache: &CacheManager, config: &SiteConfig) -> Result<SiteReport> {
    // Auto-snapshot if index has changed since last snapshot
    let pulse_config = super::config::load_pulse_config(cache.path())?;
    let ensure_result = snapshot::ensure_snapshot(cache, &pulse_config.retention)?;
    match &ensure_result {
        snapshot::EnsureSnapshotResult::Created(info) => {
            eprintln!("Auto-snapshot created: {} ({} files)", info.id, info.file_count);
        }
        snapshot::EnsureSnapshotResult::Reused(info) => {
            eprintln!("Using snapshot: {} (index unchanged)", info.id);
        }
    }

    // Clean output dir if requested
    if config.clean && config.output_dir.exists() {
        std::fs::remove_dir_all(&config.output_dir)
            .context("Failed to clean output directory")?;
    }

    // Create Zola project structure
    create_directory_structure(&config.output_dir)?;

    // Write Zola config
    write_zola_config(&config.output_dir, &config.base_url, &config.title)?;

    // Write templates
    write_templates(&config.output_dir)?;

    // Write static assets
    write_static_assets(&config.output_dir)?;

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

    // Clear LLM cache if force-renarrate is set
    if config.force_renarrate && !config.no_llm {
        let llm_cache = super::llm_cache::LlmCache::new(cache.path());
        if let Err(e) = llm_cache.clear() {
            log::warn!("Failed to clear LLM cache: {}", e);
        }
    }

    // Create LLM provider once for all surfaces
    let provider: Option<Box<dyn LlmProvider>> = if !config.no_llm {
        match narrate::create_pulse_provider() {
            Ok(p) => {
                eprintln!("LLM provider ready, narration enabled.");
                Some(p)
            }
            Err(e) => {
                eprintln!("LLM narration unavailable: {}", e);
                None
            }
        }
    } else {
        None
    };

    let llm_cache = provider.as_ref().map(|_| super::llm_cache::LlmCache::new(cache.path()));

    let mut pages_generated = 0;
    let mut digest_generated = false;
    let mut map_generated = false;
    let mut has_narration = false;

    // Collect wiki pages for the home page index
    let mut wiki_page_index: Vec<WikiPageMeta> = Vec::new();

    // Generate wiki pages
    if config.surfaces.contains(&Surface::Wiki) {
        let snapshot_id = snapshots.first().map(|s| s.id.as_str()).unwrap_or("unknown");
        let wiki_pages = wiki::generate_all_pages(
            cache,
            snapshot_diff.as_ref(),
            config.no_llm,
            snapshot_id,
            provider.as_ref().map(|p| p.as_ref()),
            llm_cache.as_ref(),
        )?;

        if wiki_pages.iter().any(|p| p.sections.summary.is_some()) {
            has_narration = true;
        }

        // Write wiki section index
        write_wiki_section_index(&config.output_dir)?;

        // Detect modules for metadata
        let modules = wiki::detect_modules(cache)?;
        let module_map: std::collections::HashMap<&str, &wiki::ModuleDefinition> = modules.iter()
            .map(|m| (m.path.as_str(), m))
            .collect();

        for (i, page) in wiki_pages.iter().enumerate() {
            let module = module_map.get(page.module_path.as_str());
            let slug = page.module_path.replace('/', "-");

            let summary_preview = page.sections.summary.as_deref()
                .map(|s| s.chars().take(200).collect::<String>())
                .unwrap_or_else(|| format!("{} files", module.map(|m| m.file_count).unwrap_or(0)));

            let tier = module.map(|m| m.tier).unwrap_or(1);
            let parent_path = if tier == 2 {
                // Find Tier 1 parent: e.g., "src/parsers" → "src"
                page.module_path.split('/').next().map(|s| s.to_string())
            } else {
                None
            };

            wiki_page_index.push(WikiPageMeta {
                title: page.title.clone(),
                slug: slug.clone(),
                file_count: module.map(|m| m.file_count).unwrap_or(0),
                total_lines: module.map(|m| m.total_lines).unwrap_or(0),
                description: summary_preview,
                tier,
                parent_path,
            });

            write_wiki_page(
                &config.output_dir,
                page,
                module,
                i + 1,
            )?;
            pages_generated += 1;
        }
    }

    // Generate digest
    if config.surfaces.contains(&Surface::Digest) {
        if let Some(current) = current_snapshot {
            let digest_data = digest::generate_digest(
                snapshot_diff.as_ref(),
                current,
                Some(cache),
                config.no_llm,
                provider.as_ref().map(|p| p.as_ref()),
                llm_cache.as_ref(),
            )?;

            if digest_data.sections.iter().any(|s| s.narrative.is_some()) {
                has_narration = true;
            }

            let digest_md = digest::render_markdown(&digest_data);
            write_digest_page(&config.output_dir, &digest_md, &digest_data)?;
            digest_generated = true;
        }
    }

    // Generate map
    let mut architecture_narrative: Option<String> = None;
    if config.surfaces.contains(&Surface::Map) {
        let map_content = map::generate_map(cache, &MapZoom::Repo, MapFormat::Mermaid)?;
        let layered_content = map::generate_layered_map(cache, MapFormat::Mermaid).ok();

        // Build architecture narrative from map data
        if let (Some(provider), Some(llm_cache)) = (provider.as_ref(), llm_cache.as_ref()) {
            let snapshot_id = snapshots.first().map(|s| s.id.as_str()).unwrap_or("unknown");
            let arch_context = build_architecture_context(cache, &wiki_page_index);
            architecture_narrative = narrate::narrate_section(
                provider.as_ref(),
                narrate::architecture_narrative_system_prompt(),
                &arch_context,
                llm_cache,
                snapshot_id,
                "architecture-narrative",
            );
            if architecture_narrative.is_some() {
                has_narration = true;
            }
        }

        write_map_page(
            &config.output_dir,
            &map_content,
            layered_content.as_deref(),
            architecture_narrative.as_deref(),
        )?;
        map_generated = true;
    }

    // Build project overview narrative
    let mut project_overview: Option<String> = None;
    if let (Some(provider), Some(llm_cache)) = (provider.as_ref(), llm_cache.as_ref()) {
        let snapshot_id = snapshots.first().map(|s| s.id.as_str()).unwrap_or("unknown");
        let overview_context = build_project_overview_context(cache, &wiki_page_index);
        project_overview = narrate::narrate_section(
            provider.as_ref(),
            narrate::project_overview_system_prompt(),
            &overview_context,
            llm_cache,
            snapshot_id,
            "project-overview",
        );
        if project_overview.is_some() {
            has_narration = true;
        }
    }

    // Write home page
    write_home_page(
        &config.output_dir,
        &config.title,
        &wiki_page_index,
        digest_generated,
        map_generated,
        project_overview.as_deref(),
    )?;

    // Compute narration mode
    let narration_mode = if config.no_llm {
        "disabled".to_string()
    } else if has_narration {
        "narrated".to_string()
    } else {
        "structural".to_string()
    };

    // Try to build with Zola
    let build_success = try_zola_build(&config.output_dir);

    Ok(SiteReport {
        output_dir: config.output_dir.display().to_string(),
        pages_generated,
        digest_generated,
        map_generated,
        narration_mode,
        build_success,
    })
}

// ── Directory structure ──────────────────────────────────────

fn create_directory_structure(output_dir: &Path) -> Result<()> {
    let dirs = [
        "",
        "content",
        "content/wiki",
        "content/digest",
        "content/map",
        "templates",
        "templates/shortcodes",
        "static",
        "sass",
    ];

    for dir in &dirs {
        std::fs::create_dir_all(output_dir.join(dir))
            .with_context(|| format!("Failed to create directory: {}", dir))?;
    }

    Ok(())
}

// ── Zola config ──────────────────────────────────────────────

fn write_zola_config(output_dir: &Path, base_url: &str, title: &str) -> Result<()> {
    let config = format!(
r#"# Zola configuration — generated by rfx pulse generate
base_url = "{base_url}"
title = "{title}"
description = "Auto-generated codebase documentation"
compile_sass = false
build_search_index = false
generate_feeds = false
minify_html = false

[markdown]
highlight_code = true
highlight_theme = "base16-ocean-dark"
render_emoji = false
external_links_target_blank = true
smart_punctuation = true

[extra]
generated_by = "Reflex Pulse"
"#);

    std::fs::write(output_dir.join("config.toml"), config)
        .context("Failed to write Zola config.toml")
}

// ── Templates ────────────────────────────────────────────────

fn write_templates(output_dir: &Path) -> Result<()> {
    // Base template with hierarchical sidebar, favicon, mobile hamburger
    let base_html = r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{% block title %}{{ config.title }}{% endblock title %}</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><path d='M4 16 Q8 8 12 16 Q16 24 20 16 Q24 8 28 16' fill='none' stroke='%237aa2f7' stroke-width='3' stroke-linecap='round'/></svg>">
    <link rel="stylesheet" href="{{ get_url(path='style.css') }}">
</head>
<body>
    <button class="mobile-menu-toggle" aria-label="Toggle menu" onclick="document.body.classList.toggle('sidebar-open')">
        <span></span><span></span><span></span>
    </button>
    <div class="layout">
        <nav class="sidebar">
            <div class="sidebar-header">
                <a href="{{ get_url(path='/') }}"><h2>{{ config.title }}</h2></a>
            </div>
            <ul class="nav-list">
                <li><a href="{{ get_url(path='/') }}" {% if current_path == "/" %}class="active"{% endif %}>Home</a></li>
                {% set wiki_section = get_section(path="wiki/_index.md") %}
                <li class="nav-section">
                    <a href="{{ get_url(path='wiki') }}" {% if current_path is starting_with("wiki") %}class="active"{% endif %}>Reference</a>
                    <ul class="nav-wiki-tree">
                        {# Build hierarchical sidebar: Tier 1 modules #}
                        {% for page in wiki_section.pages %}
                            {% if page.extra.tier is defined and page.extra.tier == 1 %}
                                {% set parent_name = page.title | trim_end_matches(pat="/") %}
                                {% set children = wiki_section.pages | filter(attribute="extra.parent_path", value=parent_name) %}
                                {% if children | length > 0 %}
                                {# Tier 1 WITH children: collapsible #}
                                <li class="nav-tier1">
                                    <details {% if current_path is starting_with(page.path) %}open{% endif %}>
                                        <summary>
                                            <a href="{{ page.permalink }}" {% if current_path == page.path %}class="active"{% endif %}>{{ page.title }}</a>
                                        </summary>
                                        <ul>
                                            {% for child in children %}
                                            <li><a href="{{ child.permalink }}" {% if current_path == child.path %}class="active"{% endif %}>{{ child.title }}</a></li>
                                            {% endfor %}
                                        </ul>
                                    </details>
                                </li>
                                {% else %}
                                {# Tier 1 WITHOUT children: plain link #}
                                <li class="nav-tier1"><a href="{{ page.permalink }}" {% if current_path == page.path %}class="active"{% endif %}>{{ page.title }}</a></li>
                                {% endif %}
                            {% endif %}
                        {% endfor %}
                        {# Orphan Tier 2 pages (no parent_path set in front matter) #}
                        {% for page in wiki_section.pages %}
                            {% if page.extra.tier is defined and page.extra.tier == 2 and page.extra.parent_path is undefined %}
                            <li><a href="{{ page.permalink }}" {% if current_path == page.path %}class="active"{% endif %}>{{ page.title }}</a></li>
                            {% endif %}
                        {% endfor %}
                    </ul>
                </li>
                <li><a href="{{ get_url(path='digest') }}" {% if current_path is starting_with("digest") %}class="active"{% endif %}>Digest</a></li>
                <li><a href="{{ get_url(path='map') }}" {% if current_path is starting_with("map") %}class="active"{% endif %}>Map</a></li>
            </ul>
        </nav>
        <main class="content">
            {% block content %}{% endblock content %}
        </main>
    </div>
    {% block scripts %}{% endblock scripts %}
</body>
</html>"##;

    // Index (home) template
    let index_html = r#"{% extends "base.html" %}
{% block title %}{{ config.title }}{% endblock title %}
{% block content %}
{{ section.content | safe }}
{% endblock content %}"#;

    // Section template (wiki/, digest/, map/)
    let section_html = r#"{% extends "base.html" %}
{% block title %}{{ section.title }} — {{ config.title }}{% endblock title %}
{% block content %}
<h1>{{ section.title }}</h1>
{{ section.content | safe }}
{% if section.pages %}
<div class="page-list">
    {% for page in section.pages %}
    <div class="page-card">
        <h3><a href="{{ page.permalink }}">{{ page.title }}</a></h3>
        {% if page.description %}
        <p>{{ page.description }}</p>
        {% endif %}
    </div>
    {% endfor %}
</div>
{% endif %}
{% endblock content %}
{% block scripts %}
{% if section.extra.has_mermaid is defined %}
<script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
    mermaid.initialize({
        startOnLoad: true,
        theme: 'base',
        themeVariables: {
            primaryColor: '#2d3250',
            primaryTextColor: '#c0caf5',
            primaryBorderColor: '#7aa2f7',
            lineColor: '#565f89',
            secondaryColor: '#292e42',
            tertiaryColor: '#1a1b26',
            edgeLabelBackground: 'transparent',
            clusterBkg: '#24283b',
            clusterBorder: '#3b4261',
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            fontSize: '14px'
        },
        securityLevel: 'loose'
    });
</script>
{% endif %}
{% endblock scripts %}"#;

    // Page template (individual wiki modules) with breadcrumbs
    let page_html = r#"{% extends "base.html" %}
{% block title %}{{ page.title }} — {{ config.title }}{% endblock title %}
{% block content %}
<nav class="breadcrumbs" aria-label="Breadcrumb">
    <a href="/">Home</a>
    <span class="sep">/</span>
    <a href="/wiki/">Reference</a>
    {% if page.extra.parent_path is defined %}
    <span class="sep">/</span>
    {% set parent_slug = page.extra.parent_path | replace(from="/", to="-") %}
    <a href="/wiki/{{ parent_slug }}/">{{ page.extra.parent_path }}/</a>
    {% endif %}
    <span class="sep">/</span>
    <span class="current">{{ page.title }}</span>
</nav>

<h1>{{ page.title }}</h1>
{% if page.extra.tier %}
<div class="page-meta">
    <span class="badge tier-{{ page.extra.tier }}">Tier {{ page.extra.tier }}</span>
    {% if page.extra.file_count %}
    <span class="badge">{{ page.extra.file_count }} files</span>
    {% endif %}
    {% if page.extra.languages %}
    <span class="badge">{{ page.extra.languages }}</span>
    {% endif %}
</div>
{% endif %}
{{ page.content | safe }}
{% endblock content %}
{% block scripts %}
{% if page.extra.has_mermaid is defined %}
<script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
    mermaid.initialize({
        startOnLoad: true,
        theme: 'base',
        themeVariables: {
            primaryColor: '#2d3250',
            primaryTextColor: '#c0caf5',
            primaryBorderColor: '#7aa2f7',
            lineColor: '#565f89',
            secondaryColor: '#292e42',
            tertiaryColor: '#1a1b26',
            edgeLabelBackground: 'transparent',
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            fontSize: '14px'
        },
        securityLevel: 'loose'
    });
</script>
{% endif %}
{% endblock scripts %}"#;

    // Mermaid shortcode
    let mermaid_shortcode = r#"<pre class="mermaid">
{{ body }}
</pre>"#;

    std::fs::write(output_dir.join("templates/base.html"), base_html)?;
    std::fs::write(output_dir.join("templates/index.html"), index_html)?;
    std::fs::write(output_dir.join("templates/section.html"), section_html)?;
    std::fs::write(output_dir.join("templates/page.html"), page_html)?;
    std::fs::write(output_dir.join("templates/shortcodes/mermaid.html"), mermaid_shortcode)?;

    Ok(())
}

// ── Static assets ────────────────────────────────────────────

fn write_static_assets(output_dir: &Path) -> Result<()> {
    let css = r#":root {
    --bg: #1a1b26;
    --bg-surface: #24283b;
    --bg-hover: #292e42;
    --bg-elevated: #1f2335;
    --fg: #c0caf5;
    --fg-muted: #565f89;
    --fg-accent: #7aa2f7;
    --fg-green: #9ece6a;
    --fg-yellow: #e0af68;
    --fg-red: #f7768e;
    --border: #3b4261;
    --sidebar-width: 270px;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--fg);
    line-height: 1.7;
}

/* ── Layout ──────────────────────────────────── */

.layout {
    display: flex;
    min-height: 100vh;
}

.sidebar {
    width: var(--sidebar-width);
    background: linear-gradient(180deg, var(--bg-surface) 0%, var(--bg-elevated) 100%);
    border-right: 1px solid var(--border);
    padding: 1.5rem 0;
    position: fixed;
    height: 100vh;
    overflow-y: auto;
    z-index: 100;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
}

.sidebar::-webkit-scrollbar { width: 6px; }
.sidebar::-webkit-scrollbar-track { background: transparent; }
.sidebar::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
.sidebar::-webkit-scrollbar-thumb:hover { background: var(--fg-muted); }

.sidebar-header {
    padding: 0 1.25rem 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.25rem;
}

.sidebar-header h2 {
    font-size: 1.1rem;
    color: var(--fg-accent);
    letter-spacing: 0.02em;
}

.sidebar a {
    color: var(--fg);
    text-decoration: none;
    transition: color 0.15s;
}

.sidebar a:hover {
    color: var(--fg-accent);
}

.sidebar a.active {
    color: var(--fg-accent);
    font-weight: 600;
    border-left: 2px solid var(--fg-accent);
    padding-left: 0.5rem;
    margin-left: -0.5rem;
}

/* ── Sidebar nav ─────────────────────────────── */

.nav-list {
    list-style: none;
    padding: 0;
}

.nav-list > li {
    padding: 0.35rem 1.25rem;
}

.nav-list > li > a {
    font-weight: 500;
    font-size: 0.95rem;
}

.nav-section ul {
    list-style: none;
    padding-left: 0.75rem;
    margin-top: 0.25rem;
}

.nav-section ul li {
    padding: 0.15rem 0;
}

.nav-section ul li a {
    font-size: 0.85rem;
    color: var(--fg-muted);
}

.nav-section ul li a:hover {
    color: var(--fg-accent);
}

/* Hierarchical wiki tree */
.nav-wiki-tree {
    list-style: none;
    padding-left: 0;
    margin-top: 0.25rem;
}

.nav-wiki-tree .nav-tier1 {
    margin-bottom: 0.1rem;
}

.nav-wiki-tree .nav-tier1 > a {
    font-size: 0.9rem;
    display: block;
    padding: 0.2rem 0;
}

.nav-wiki-tree details {
    margin-bottom: 0;
}

.nav-wiki-tree details summary {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.2rem 0;
    font-size: 0.9rem;
}

.nav-wiki-tree details summary::before {
    content: "▸";
    flex-shrink: 0;
    font-size: 0.75rem;
}

.nav-wiki-tree details[open] summary::before {
    content: "▾";
}

.nav-wiki-tree details summary a {
    display: inline;
}

.nav-wiki-tree details ul {
    list-style: none;
    padding-left: 1rem;
}

.nav-wiki-tree details ul li {
    padding: 0.1rem 0;
}

.nav-wiki-tree details ul li a {
    font-size: 0.8rem;
    color: var(--fg-muted);
}

.nav-section ul li a {
    display: block;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 190px;
}

/* ── Content area ────────────────────────────── */

.content {
    margin-left: var(--sidebar-width);
    padding: 2rem 3rem;
    max-width: 960px;
    flex: 1;
}

/* ── Typography ──────────────────────────────── */

h1 {
    font-size: 1.8rem;
    margin-bottom: 1rem;
    color: var(--fg);
    letter-spacing: -0.01em;
}

h2 {
    font-size: 1.4rem;
    margin: 2rem 0 0.75rem;
    color: var(--fg-accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
}

h3 {
    font-size: 1.15rem;
    margin: 1.5rem 0 0.5rem;
    color: var(--fg);
}

p { margin-bottom: 0.75rem; }

a { color: var(--fg-accent); text-decoration: none; transition: color 0.15s; }
a:hover { text-decoration: underline; }

code {
    background: var(--bg-surface);
    padding: 0.15em 0.4em;
    border-radius: 3px;
    font-size: 0.9em;
    font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", monospace;
    font-feature-settings: "liga" 1, "calt" 1;
}

pre {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    overflow-x: auto;
    margin: 1rem 0;
}

pre code {
    background: none;
    padding: 0;
}

/* ── Tables ──────────────────────────────────── */

table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.95rem;
}

th, td {
    text-align: left;
    padding: 0.6rem 0.85rem;
    border: 1px solid var(--border);
}

th {
    background: var(--bg-surface);
    font-weight: 600;
    color: var(--fg-accent);
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

tr:nth-child(even) { background: var(--bg-elevated); }
tr:hover { background: var(--bg-hover); }

td:first-child {
    border-left: 2px solid var(--border);
}

/* ── Lists ───────────────────────────────────── */

ul, ol { padding-left: 1.5rem; margin-bottom: 0.75rem; }
li { margin-bottom: 0.3rem; }

/* ── Badges ──────────────────────────────────── */

.page-meta {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}

.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 500;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    color: var(--fg-muted);
}

.tier-1 { color: var(--fg-accent); border-color: var(--fg-accent); background: rgba(122, 162, 247, 0.1); }
.tier-2 { color: var(--fg-green); border-color: var(--fg-green); background: rgba(158, 206, 106, 0.1); }

/* ── Breadcrumbs ─────────────────────────────── */

.breadcrumbs {
    font-size: 0.85rem;
    color: var(--fg-muted);
    margin-bottom: 1rem;
    padding: 0.5rem 0;
}

.breadcrumbs a {
    color: var(--fg-muted);
    transition: color 0.15s;
}

.breadcrumbs a:hover {
    color: var(--fg-accent);
    text-decoration: none;
}

.breadcrumbs .sep {
    margin: 0 0.4rem;
    color: var(--border);
}

.breadcrumbs .current {
    color: var(--fg);
    font-weight: 500;
}

/* ── Cards ───────────────────────────────────── */

.page-card {
    padding: 1rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 0.75rem;
    background: var(--bg-surface);
    transition: background 0.15s, border-color 0.15s;
}

.page-card:hover {
    background: var(--bg-hover);
    border-color: var(--fg-accent);
}

.page-card h3 { margin: 0 0 0.25rem; font-size: 1rem; }
.page-card p { margin: 0; font-size: 0.9rem; color: var(--fg-muted); }

/* ── Mermaid diagrams ────────────────────────── */

.mermaid {
    background: var(--bg-surface);
    padding: 2rem;
    border-radius: 8px;
    text-align: center;
    border: 1px solid var(--border);
    overflow: auto;
    max-height: 80vh;
    cursor: grab;
    position: relative;
}

.mermaid svg {
    max-width: none;
    min-width: 100%;
}

/* Edge labels */
.mermaid .edgeLabel {
    background: var(--bg-surface) !important;
    color: var(--fg) !important;
}

/* ── Module grid ─────────────────────────────── */

.module-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.module-card {
    padding: 1.25rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg-surface);
    transition: border-color 0.2s, background 0.2s, transform 0.2s;
}

.module-card:hover {
    background: var(--bg-hover);
    border-color: var(--fg-accent);
    transform: translateY(-1px);
}

.module-card h3 {
    margin: 0 0 0.25rem;
    font-size: 1.05rem;
}

.module-card h3 a { color: var(--fg); }
.module-card h3 a:hover { color: var(--fg-accent); text-decoration: none; }

.module-stats {
    font-size: 0.8rem;
    color: var(--fg-muted);
    margin-bottom: 0.5rem;
}

.module-card p {
    margin: 0;
    font-size: 0.9rem;
    color: var(--fg-muted);
    line-height: 1.5;
}

.tier-1-card {
    border-left: 3px solid var(--fg-accent);
}

.sub-modules {
    list-style: none;
    padding: 0.5rem 0 0 0;
    margin: 0;
    border-top: 1px solid var(--border);
    margin-top: 0.75rem;
}

.sub-modules li {
    padding: 0.15rem 0;
    font-size: 0.85rem;
}

.sub-modules li a { color: var(--fg-muted); }
.sub-modules li a:hover { color: var(--fg-accent); }

.sub-stats {
    color: var(--fg-muted);
    font-size: 0.75rem;
}

/* ── Collapsible details/summary ─────────────── */

details {
    margin-bottom: 0.5rem;
}

details summary {
    cursor: pointer;
    padding: 0.3rem 0;
    color: var(--fg);
    list-style: none;
    transition: color 0.15s;
}

details summary:hover { color: var(--fg-accent); }

details summary::-webkit-details-marker { display: none; }

details summary::before {
    content: "▸ ";
    color: var(--fg-muted);
    transition: color 0.15s;
}

details[open] summary::before {
    content: "▾ ";
}

details[open] {
    padding-bottom: 0.25rem;
}

/* ── Map-specific: full width for diagrams ──── */

.map-diagram {
    max-width: none;
}

/* ── Diagram view toggle ─────────────────────── */

.diagram-tabs {
    display: flex;
    gap: 0;
    margin-bottom: 0;
    border-bottom: 1px solid var(--border);
}

.diagram-tab {
    padding: 0.5rem 1.25rem;
    cursor: pointer;
    font-size: 0.9rem;
    color: var(--fg-muted);
    border-bottom: 2px solid transparent;
    transition: color 0.15s, border-color 0.15s;
    background: none;
    border-top: none;
    border-left: none;
    border-right: none;
    font-family: inherit;
}

.diagram-tab:hover { color: var(--fg); }
.diagram-tab.active { color: var(--fg-accent); border-bottom-color: var(--fg-accent); }

.diagram-panel { display: none; }
.diagram-panel.active { display: block; }

/* ── Mobile hamburger ────────────────────────── */

.mobile-menu-toggle {
    display: none;
    position: fixed;
    top: 0.75rem;
    left: 0.75rem;
    z-index: 200;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.5rem;
    cursor: pointer;
    width: 36px;
    height: 36px;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 4px;
}

.mobile-menu-toggle span {
    display: block;
    width: 18px;
    height: 2px;
    background: var(--fg);
    border-radius: 1px;
    transition: transform 0.2s, opacity 0.2s;
}

/* ── Responsive ──────────────────────────────── */

@media (max-width: 768px) {
    .mobile-menu-toggle { display: flex; }

    .sidebar {
        transform: translateX(-100%);
        transition: transform 0.25s ease;
    }

    .sidebar-open .sidebar {
        transform: translateX(0);
    }

    .content {
        margin-left: 0;
        padding: 3.5rem 1rem 1rem;
    }

    .module-grid { grid-template-columns: 1fr; }

    .mermaid { padding: 1rem; }
}
"#;

    std::fs::write(output_dir.join("static/style.css"), css)
        .context("Failed to write style.css")
}

// ── Content generation ───────────────────────────────────────

struct WikiPageMeta {
    title: String,
    slug: String,
    file_count: usize,
    total_lines: usize,
    description: String,
    tier: u8,
    parent_path: Option<String>,
}

fn write_home_page(
    output_dir: &Path,
    title: &str,
    wiki_pages: &[WikiPageMeta],
    has_digest: bool,
    has_map: bool,
    project_overview: Option<&str>,
) -> Result<()> {
    let mut content = String::new();

    content.push_str("+++\n");
    content.push_str(&format!("title = \"{}\"\n", title));
    content.push_str("sort_by = \"weight\"\n");
    content.push_str("+++\n\n");

    content.push_str(&format!("# {}\n\n", title));

    // Project Overview (narrated or structural fallback)
    if let Some(overview) = project_overview {
        content.push_str(overview);
        content.push_str("\n\n");
    } else {
        content.push_str("Auto-generated codebase documentation powered by [Reflex](https://github.com/reflex-search/reflex).\n\n");
    }

    // Core Modules (Tier 1) as prominent cards
    let tier1: Vec<&WikiPageMeta> = wiki_pages.iter().filter(|p| p.tier == 1).collect();
    let tier2: Vec<&WikiPageMeta> = wiki_pages.iter().filter(|p| p.tier == 2).collect();

    if !tier1.is_empty() {
        content.push_str("## Core Modules\n\n");
        content.push_str("<div class=\"module-grid\">\n\n");
        for page in &tier1 {
            let desc = truncate_str(&page.description, 150);
            content.push_str(&format!(
                "<div class=\"module-card tier-1-card\">\n\
                 <h3><a href=\"/wiki/{}/\">{}</a></h3>\n\
                 <div class=\"module-stats\">{} files · {} lines</div>\n\
                 <p>{}</p>\n",
                page.slug, page.title, page.file_count, page.total_lines, desc
            ));

            // Nest Tier 2 children under their Tier 1 parent
            let parent_name = page.title.trim_end_matches('/');
            let children: Vec<&&WikiPageMeta> = tier2.iter()
                .filter(|t2| t2.parent_path.as_deref() == Some(parent_name))
                .collect();
            if !children.is_empty() {
                content.push_str("<ul class=\"sub-modules\">\n");
                for child in children {
                    content.push_str(&format!(
                        "<li><a href=\"/wiki/{}/\">{}</a> <span class=\"sub-stats\">({} files)</span></li>\n",
                        child.slug, child.title, child.file_count
                    ));
                }
                content.push_str("</ul>\n");
            }

            content.push_str("</div>\n\n");
        }
        content.push_str("</div>\n\n");
    }

    // Remaining Tier 2 modules that don't have a Tier 1 parent shown above
    let orphan_tier2: Vec<&&WikiPageMeta> = tier2.iter()
        .filter(|t2| {
            let parent = t2.parent_path.as_deref().unwrap_or("");
            !tier1.iter().any(|t1| t1.title.trim_end_matches('/') == parent)
        })
        .collect();
    if !orphan_tier2.is_empty() {
        content.push_str("## Sub-modules\n\n");
        content.push_str("| Module | Files | Lines | Description |\n|---|---|---|---|\n");
        for page in orphan_tier2 {
            let desc = truncate_str(&page.description, 77);
            content.push_str(&format!(
                "| [{}](@/wiki/{}.md) | {} | {} | {} |\n",
                page.title, page.slug, page.file_count, page.total_lines, desc
            ));
        }
        content.push('\n');
    }

    // Navigation
    content.push_str("## Sections\n\n");
    content.push_str("- [Reference](/wiki/) — Per-module documentation\n");
    if has_digest {
        content.push_str("- [Digest](/digest/) — Structural change report\n");
    }
    if has_map {
        content.push_str("- [Map](/map/) — Architecture dependency graph\n");
    }

    // Reading Guide
    content.push_str("\n## Reading Guide\n\n");
    content.push_str("- **Tier 1** modules are top-level directories that define the project's major components.\n");
    content.push_str("- **Tier 2** modules are sub-directories within Tier 1 modules that have 3+ files.\n");
    content.push_str("- **Dependency hotspots** are files imported by many others — changes to these have wide blast radius.\n");
    content.push_str("- **Circular dependencies** indicate tightly coupled modules that may be hard to refactor independently.\n");
    content.push_str("- The **Map** page shows the full module dependency graph. Thick arrows indicate many file-level edges.\n");

    std::fs::write(output_dir.join("content/_index.md"), content)
        .context("Failed to write home page")
}

fn write_wiki_section_index(output_dir: &Path) -> Result<()> {
    let content = r#"+++
title = "Reference"
sort_by = "weight"
template = "section.html"
+++

Per-module documentation pages. Each page covers a detected module's structure,
dependencies, key symbols, and metrics.
"#;

    std::fs::write(output_dir.join("content/wiki/_index.md"), content)
        .context("Failed to write wiki section index")
}

fn write_wiki_page(
    output_dir: &Path,
    page: &wiki::WikiPage,
    module: Option<&&wiki::ModuleDefinition>,
    weight: usize,
) -> Result<()> {
    let slug = page.module_path.replace('/', "-");
    let mut content = String::new();

    // TOML front matter
    content.push_str("+++\n");
    content.push_str(&format!("title = \"{}\"\n", page.title));
    content.push_str(&format!("weight = {}\n", weight));
    if let Some(summary) = &page.sections.summary {
        let desc = truncate_str(summary, 200)
            .replace('\\', "\\\\")
            .replace('"', "'")
            .replace('\n', " ");
        content.push_str(&format!("description = \"{}\"\n", desc));
    }

    let has_mermaid = page.sections.dependency_diagram.is_some();

    content.push_str("\n[extra]\n");
    if let Some(m) = module {
        content.push_str(&format!("tier = {}\n", m.tier));
        content.push_str(&format!("file_count = {}\n", m.file_count));
        content.push_str(&format!("total_lines = {}\n", m.total_lines));
        content.push_str(&format!("languages = \"{}\"\n", m.languages.join(", ")));
        // Parent path for breadcrumb navigation (Tier 2 modules)
        if m.tier == 2 {
            if let Some(parent) = page.module_path.split('/').next() {
                content.push_str(&format!("parent_path = \"{}\"\n", parent));
            }
        }
    }
    if has_mermaid {
        content.push_str("has_mermaid = true\n");
    }
    content.push_str("+++\n\n");

    // Page content
    if let Some(summary) = &page.sections.summary {
        content.push_str(summary);
        content.push_str("\n\n");
    }

    // Dependency diagram (mermaid)
    if let Some(diagram) = &page.sections.dependency_diagram {
        content.push_str("## Dependency Diagram\n\n");
        content.push_str("{% mermaid() %}\n");
        content.push_str(diagram);
        content.push_str("{% end %}\n\n");
    }

    content.push_str("## Structure\n\n");
    content.push_str(&page.sections.structure);
    content.push_str("\n\n");

    content.push_str("## Dependencies\n\n");
    content.push_str(&page.sections.dependencies);
    content.push_str("\n\n");

    content.push_str("## Dependents\n\n");
    content.push_str(&page.sections.dependents);
    content.push_str("\n\n");

    content.push_str("## Key Symbols\n\n");
    content.push_str(&page.sections.key_symbols);
    content.push_str("\n\n");

    content.push_str("## Metrics\n\n");
    content.push_str(&page.sections.metrics);
    content.push_str("\n\n");

    if let Some(changes) = &page.sections.recent_changes {
        content.push_str("## Recent Changes\n\n");
        content.push_str(changes);
        content.push_str("\n\n");
    }

    let filename = format!("{}.md", slug);
    std::fs::write(output_dir.join("content/wiki").join(&filename), content)
        .with_context(|| format!("Failed to write wiki page: {}", filename))
}

fn write_digest_page(
    output_dir: &Path,
    digest_md: &str,
    digest_data: &digest::Digest,
) -> Result<()> {
    // Section index
    let mut index_content = String::new();
    index_content.push_str("+++\n");
    index_content.push_str(&format!("title = \"{}\"\n", digest_data.title));
    index_content.push_str("template = \"section.html\"\n");
    index_content.push_str("+++\n\n");
    index_content.push_str(digest_md);

    std::fs::write(output_dir.join("content/digest/_index.md"), index_content)
        .context("Failed to write digest page")
}

fn write_map_page(
    output_dir: &Path,
    mermaid_content: &str,
    layered_content: Option<&str>,
    narrative: Option<&str>,
) -> Result<()> {
    let mut content = String::new();
    content.push_str("+++\n");
    content.push_str("title = \"Architecture Map\"\n");
    content.push_str("template = \"section.html\"\n");
    content.push_str("\n[extra]\n");
    content.push_str("has_mermaid = true\n");
    content.push_str("+++\n\n");

    // Architecture narrative (LLM-generated or structural fallback)
    if let Some(narrative) = narrative {
        content.push_str(narrative);
        content.push_str("\n\n");
    } else {
        content.push_str("Module-level dependency graph showing how code modules relate to each other.\n\n");
    }

    // Diagram with view toggle (flat vs layered)
    content.push_str("## Dependency Graph\n\n");

    if let Some(layered) = layered_content {
        content.push_str("<div class=\"diagram-tabs\">\n");
        content.push_str("  <button class=\"diagram-tab active\" onclick=\"switchDiagram('flat')\">Flat View</button>\n");
        content.push_str("  <button class=\"diagram-tab\" onclick=\"switchDiagram('layered')\">Layered View</button>\n");
        content.push_str("</div>\n\n");

        content.push_str("<div id=\"diagram-flat\" class=\"diagram-panel active\">\n\n");
        content.push_str("{% mermaid() %}\n");
        content.push_str(mermaid_content);
        content.push_str("{% end %}\n\n");
        content.push_str("</div>\n\n");

        content.push_str("<div id=\"diagram-layered\" class=\"diagram-panel\">\n\n");
        content.push_str("{% mermaid() %}\n");
        content.push_str(layered);
        content.push_str("{% end %}\n\n");
        content.push_str("</div>\n\n");

        content.push_str("<script>\n");
        content.push_str("function switchDiagram(view) {\n");
        content.push_str("  document.querySelectorAll('.diagram-panel').forEach(p => p.classList.remove('active'));\n");
        content.push_str("  document.querySelectorAll('.diagram-tab').forEach(t => t.classList.remove('active'));\n");
        content.push_str("  document.getElementById('diagram-' + view).classList.add('active');\n");
        content.push_str("  event.target.classList.add('active');\n");
        content.push_str("}\n");
        content.push_str("</script>\n\n");
    } else {
        content.push_str("{% mermaid() %}\n");
        content.push_str(mermaid_content);
        content.push_str("{% end %}\n\n");
    }

    // Legend
    content.push_str("## Legend\n\n");
    content.push_str("- **Thick arrows** indicate many file-level dependency edges between modules.\n");
    content.push_str("- **Red-highlighted nodes** are dependency hotspots (imported by many modules).\n");
    content.push_str("- **Arrow labels** show the number of file-level import edges.\n");
    content.push_str("- **Direction** follows the import: A → B means A depends on B.\n");
    content.push_str("- **Click** any module node to navigate to its wiki page.\n");

    std::fs::write(output_dir.join("content/map/_index.md"), content)
        .context("Failed to write map page")
}

// ── Context builders for LLM narration ───────────────────────

/// Build structural context for the project overview LLM prompt
fn build_project_overview_context(cache: &CacheManager, wiki_pages: &[WikiPageMeta]) -> String {
    let mut ctx = String::new();

    // Aggregate stats
    let total_files: usize = wiki_pages.iter().map(|p| p.file_count).sum();
    let total_lines: usize = wiki_pages.iter().map(|p| p.total_lines).sum();
    let tier1_count = wiki_pages.iter().filter(|p| p.tier == 1).count();
    let tier2_count = wiki_pages.iter().filter(|p| p.tier == 2).count();

    ctx.push_str(&format!("Total files: {}\nTotal lines: {}\n", total_files, total_lines));
    ctx.push_str(&format!("Tier 1 modules: {}\nTier 2 modules: {}\n\n", tier1_count, tier2_count));

    // Language distribution from meta.db
    let db_path = cache.path().join("meta.db");
    if let Ok(conn) = Connection::open(&db_path) {
        if let Ok(mut stmt) = conn.prepare(
            "SELECT COALESCE(language, 'other'), COUNT(*) FROM files GROUP BY language ORDER BY COUNT(*) DESC LIMIT 10"
        ) {
            if let Ok(rows) = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
            }) {
                ctx.push_str("Languages:\n");
                for row in rows.flatten() {
                    ctx.push_str(&format!("- {}: {} files\n", row.0, row.1));
                }
                ctx.push('\n');
            }
        }

        // Dependency stats
        if let Ok(edge_count) = conn.query_row::<usize, _, _>(
            "SELECT COUNT(*) FROM file_dependencies WHERE resolved_file_id IS NOT NULL",
            [],
            |row| row.get(0),
        ) {
            ctx.push_str(&format!("Dependency edges: {}\n", edge_count));
        }

        // Hotspot count
        if let Ok(hotspot_count) = conn.query_row::<usize, _, _>(
            "SELECT COUNT(*) FROM (
                SELECT resolved_file_id, COUNT(DISTINCT file_id) as c
                FROM file_dependencies WHERE resolved_file_id IS NOT NULL
                GROUP BY resolved_file_id HAVING c >= 5
            )",
            [],
            |row| row.get(0),
        ) {
            ctx.push_str(&format!("Dependency hotspots (5+ dependents): {}\n", hotspot_count));
        }

        // Cycle count (mutual deps)
        if let Ok(cycle_count) = conn.query_row::<usize, _, _>(
            "SELECT COUNT(*) FROM (
                SELECT DISTINCT fd1.file_id FROM file_dependencies fd1
                JOIN file_dependencies fd2
                  ON fd1.file_id = fd2.resolved_file_id
                  AND fd1.resolved_file_id = fd2.file_id
                WHERE fd1.resolved_file_id IS NOT NULL AND fd2.resolved_file_id IS NOT NULL
            )",
            [],
            |row| row.get(0),
        ) {
            ctx.push_str(&format!("Files in circular dependencies: {}\n", cycle_count));
        }
    }

    ctx.push('\n');

    // Tier 1 modules with descriptions
    ctx.push_str("Core modules:\n");
    for page in wiki_pages.iter().filter(|p| p.tier == 1) {
        let desc = truncate_str(&page.description, 120);
        ctx.push_str(&format!("- {} ({} files, {} lines): {}\n",
            page.title, page.file_count, page.total_lines, desc));
    }

    ctx
}

/// Build structural context for the architecture narrative LLM prompt
fn build_architecture_context(cache: &CacheManager, wiki_pages: &[WikiPageMeta]) -> String {
    let mut ctx = String::new();

    let db_path = cache.path().join("meta.db");
    let conn = match Connection::open(&db_path) {
        Ok(c) => c,
        Err(_) => return "No dependency data available.".to_string(),
    };

    // Module-to-module edges
    ctx.push_str("Module dependency edges:\n");
    let modules: Vec<&WikiPageMeta> = wiki_pages.iter().collect();
    for source in &modules {
        let source_path = source.title.trim_end_matches('/');
        let pattern = format!("{}/%", source_path);
        if let Ok(mut stmt) = conn.prepare(
            "SELECT DISTINCT f2.path
             FROM file_dependencies fd
             JOIN files f1 ON fd.file_id = f1.id
             JOIN files f2 ON fd.resolved_file_id = f2.id
             WHERE f1.path LIKE ?1 AND f2.path NOT LIKE ?1"
        ) {
            if let Ok(dep_files) = stmt.query_map([&pattern], |row| row.get::<_, String>(0)) {
                let dep_files: Vec<String> = dep_files.flatten().collect();
                // Map files to modules
                let mut target_modules = std::collections::HashMap::new();
                for dep_file in &dep_files {
                    for target in &modules {
                        let target_path = target.title.trim_end_matches('/');
                        if dep_file.starts_with(&format!("{}/", target_path)) {
                            *target_modules.entry(target_path.to_string()).or_insert(0usize) += 1;
                        }
                    }
                }
                for (target, count) in &target_modules {
                    ctx.push_str(&format!("- {} → {} ({} file edges)\n", source_path, target, count));
                }
            }
        }
    }

    ctx.push('\n');

    // Top hotspots
    ctx.push_str("Dependency hotspots (most-imported files):\n");
    if let Ok(mut stmt) = conn.prepare(
        "SELECT f.path, COUNT(DISTINCT fd.file_id) as dep_count
         FROM file_dependencies fd
         JOIN files f ON fd.resolved_file_id = f.id
         GROUP BY fd.resolved_file_id
         ORDER BY dep_count DESC
         LIMIT 10"
    ) {
        if let Ok(rows) = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
        }) {
            for row in rows.flatten() {
                ctx.push_str(&format!("- {} ({} dependents)\n", row.0, row.1));
            }
        }
    }

    ctx.push('\n');

    // Circular dependencies
    if let Ok(cycle_count) = conn.query_row::<usize, _, _>(
        "SELECT COUNT(*) FROM (
            SELECT DISTINCT fd1.file_id FROM file_dependencies fd1
            JOIN file_dependencies fd2
              ON fd1.file_id = fd2.resolved_file_id
              AND fd1.resolved_file_id = fd2.file_id
            WHERE fd1.resolved_file_id IS NOT NULL AND fd2.resolved_file_id IS NOT NULL
        )",
        [],
        |row| row.get(0),
    ) {
        ctx.push_str(&format!("Files involved in circular dependencies: {}\n", cycle_count));
    }

    ctx
}

// ── Zola build ───────────────────────────────────────────────

fn try_zola_build(output_dir: &Path) -> bool {
    match zola::ensure_zola() {
        Ok(zola_path) => {
            eprintln!("Building site with Zola...");
            let public_dir = output_dir.join("public");

            let result = std::process::Command::new(&zola_path)
                .current_dir(output_dir)
                .arg("build")
                .arg("--force")
                .arg("--output-dir")
                .arg(&public_dir)
                .output();

            match result {
                Ok(output) if output.status.success() => {
                    // Count HTML files in public/
                    let html_count = count_html_files(&public_dir);
                    eprintln!("Site built at {}/ ({} pages)", public_dir.display(), html_count);
                    true
                }
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    eprintln!("Zola build failed:\n{}", stderr);
                    eprintln!("The Zola project was generated at {}/ — you can build manually with:", output_dir.display());
                    eprintln!("  cd {} && zola build", output_dir.display());
                    false
                }
                Err(e) => {
                    eprintln!("Failed to run Zola: {}", e);
                    false
                }
            }
        }
        Err(e) => {
            eprintln!("Could not download Zola: {}", e);
            eprintln!("The Zola project was generated at {}/ — install Zola and run:", output_dir.display());
            eprintln!("  cd {} && zola build", output_dir.display());
            eprintln!("Install Zola: https://www.getzola.org/documentation/getting-started/installation/");
            false
        }
    }
}

fn count_html_files(dir: &Path) -> usize {
    if !dir.exists() {
        return 0;
    }
    walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|ext| ext == "html").unwrap_or(false))
        .count()
}
