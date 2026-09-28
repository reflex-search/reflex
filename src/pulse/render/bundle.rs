//! Write the page bundle the site template builds from.
//!
//! ```text
//! <site>/pulse.config.json         title, base, tabs, per-tab sidebar trees, build knobs
//! <site>/bundle/index.json         [{id, tab, hash}] — one row per page except the home
//! <site>/bundle/pages/<id>.json    {title, description, template, headings, html}
//! <site>/bundle/pages/index.json   the home page (route `/`)
//! <site>/src/styles/pulse-highlight.css
//! ```
//! Page ids are routes without slashes (`docs/changelog`, `internals`). Files are
//! rewritten only when their bytes change, so Astro's incremental build and the
//! content hashes in `index.json` stay stable.

use super::html::{Heading, Renderer, highlight_css};
use super::project::ProjectWriter;
use crate::pulse::model::{NavNode, PageKind, Site, TabId};
use anyhow::Result;
use rayon::prelude::*;
use serde::Serialize;
use serde_json::json;

/// Where the site is served from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseUrl {
    /// `https://host` when an absolute URL was given.
    pub site: Option<String>,
    /// `/` or `/sub/`.
    pub base: String,
}

impl BaseUrl {
    /// `https://host/sub` → site `https://host`, base `/sub/`; `/sub` → base `/sub/`.
    pub fn parse(s: &str) -> Self {
        let s = s.trim();
        let (site, path) = match s.split_once("://") {
            Some((scheme, rest)) => {
                let (host, path) = rest.split_once('/').unwrap_or((rest, ""));
                (Some(format!("{scheme}://{host}")), path.to_string())
            }
            None => (None, s.to_string()),
        };
        let trimmed = path.trim_matches('/');
        let base = if trimmed.is_empty() {
            "/".to_string()
        } else {
            format!("/{trimmed}/")
        };
        Self { site, base }
    }

    /// The base without its trailing slash (`""` for the root).
    pub fn prefix(&self) -> &str {
        self.base.trim_end_matches('/')
    }
}

#[derive(Serialize)]
struct PageJson<'a> {
    title: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'a str>,
    template: &'a str,
    badges: &'a [String],
    headings: Vec<Heading>,
    html: String,
    mermaid: bool,
}

fn page_id(route: &str) -> String {
    let t = route.trim_matches('/');
    if t.is_empty() {
        "index".into()
    } else {
        t.into()
    }
}

fn tab_key(tab: TabId) -> &'static str {
    match tab {
        TabId::Docs => "docs",
        TabId::Internals => "internals",
    }
}

fn sidebar(site: &Site, nodes: &[NavNode]) -> Vec<serde_json::Value> {
    nodes
        .iter()
        .filter_map(|n| match n {
            NavNode::Page { page } => {
                let p = site.pages.get(page)?;
                Some(json!({ "label": p.title, "slug": p.route.trim_matches('/') }))
            }
            NavNode::Group {
                label,
                collapsed,
                children,
            } => Some(json!({
                "label": label,
                "collapsed": collapsed,
                "items": sidebar(site, children),
            })),
        })
        .collect()
}

/// Summary of one bundle write.
#[derive(Debug, Default, Clone, Copy)]
pub struct BundleStats {
    pub pages: usize,
    pub written: usize,
    pub unchanged: usize,
    pub removed: usize,
}

/// Render every page and write the bundle into `site_dir`.
pub fn write_bundle(
    site: &Site,
    site_dir: &std::path::Path,
    base: &BaseUrl,
) -> Result<BundleStats> {
    let renderer = Renderer::new(site, &base.base);
    let pages: Vec<_> = site.pages.values().collect();
    let rendered: Vec<(String, TabId, Vec<u8>)> = pages
        .par_iter()
        .map(|p| {
            let r = renderer.render_page(p);
            let template = if matches!(p.kind, PageKind::Landing) {
                "splash"
            } else {
                "doc"
            };
            let json = PageJson {
                title: &p.title,
                description: p.description.as_deref(),
                template,
                badges: &p.badges,
                headings: r.headings,
                html: r.html,
                mermaid: r.has_mermaid,
            };
            let bytes = serde_json::to_vec(&json).expect("page JSON serializes");
            (page_id(&p.route), p.tab, bytes)
        })
        .collect();

    let mut w = ProjectWriter::open(site_dir)?;
    let mut index = Vec::new();
    for (id, tab, bytes) in &rendered {
        w.put(&format!("bundle/pages/{id}.json"), bytes)?;
        if id != "index" {
            let hash = blake3::hash(bytes).to_hex()[..16].to_string();
            index.push(json!({ "id": id, "tab": tab_key(*tab), "hash": hash }));
        }
    }
    w.put("bundle/index.json", &serde_json::to_vec(&index)?)?;

    let tabs: Vec<serde_json::Value> = site
        .tabs
        .iter()
        .map(|t| {
            let landing = site.pages.get(&t.landing);
            json!({
                "id": tab_key(t.id),
                "label": t.label,
                "prefix": t.id.prefix(),
                "href": landing.map(|p| p.route.clone()).unwrap_or_else(|| t.id.prefix().to_string()),
            })
        })
        .collect();
    let mut sidebars = serde_json::Map::new();
    for t in &site.tabs {
        let mut nodes = Vec::new();
        if let Some(l) = site.pages.get(&t.landing) {
            let label = if t.id == TabId::Docs {
                "Overview".to_string()
            } else {
                l.title.clone()
            };
            nodes.push(json!({ "label": label, "slug": l.route.trim_matches('/') }));
        }
        nodes.extend(sidebar(site, &t.nav));
        sidebars.insert(tab_key(t.id).into(), serde_json::Value::Array(nodes));
    }
    let config = json!({
        "title": site.meta.title,
        "description": site.meta.description,
        "site": base.site,
        "base": base.base,
        "tabs": tabs,
        "sidebar": sidebars,
        "sidebarFullTreeMax": 800,
        "build": { "pageSource": "fs", "incremental": true },
        "repo": site.meta.repo.as_ref().map(|r| r.web_url.clone()),
        "generator": site.meta.generator,
    });
    // Astro validates types: an absent key is fine, `null` is not.
    let mut config = config;
    if let Some(obj) = config.as_object_mut() {
        obj.retain(|_, v| !v.is_null());
    }
    w.put("pulse.config.json", &serde_json::to_vec_pretty(&config)?)?;
    w.put("src/styles/pulse-highlight.css", highlight_css().as_bytes())?;

    let s = w.finish("bundle/")?;
    Ok(BundleStats {
        pages: rendered.len(),
        written: s.written,
        unchanged: s.unchanged,
        removed: s.removed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_urls() {
        assert_eq!(
            BaseUrl::parse("/"),
            BaseUrl {
                site: None,
                base: "/".into()
            }
        );
        assert_eq!(
            BaseUrl::parse("/reflex"),
            BaseUrl {
                site: None,
                base: "/reflex/".into()
            }
        );
        assert_eq!(
            BaseUrl::parse("https://pulse.rfx-search.dev"),
            BaseUrl {
                site: Some("https://pulse.rfx-search.dev".into()),
                base: "/".into()
            }
        );
        assert_eq!(
            BaseUrl::parse("https://x.github.io/repo/"),
            BaseUrl {
                site: Some("https://x.github.io".into()),
                base: "/repo/".into()
            }
        );
        assert_eq!(BaseUrl::parse("/reflex/").prefix(), "/reflex");
        assert_eq!(BaseUrl::parse("/").prefix(), "");
    }

    #[test]
    fn page_ids() {
        assert_eq!(page_id("/"), "index");
        assert_eq!(page_id("/internals/"), "internals");
        assert_eq!(page_id("/docs/reference/x/"), "docs/reference/x");
    }
}
