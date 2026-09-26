//! Link targets and their central resolution.
//!
//! Content links to a [`Target`]; only the [`Linker`] turns targets into URLs. Routes
//! are base-less (`/internals/modules/src-pulse/`); the renderer prefixes the site's
//! base path, so `--base-url /reflex/` needs no change here.

use super::Site;
use super::content::{Block, Inline};
use super::ids::{AnchorId, PageId, SymbolId};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Target {
    Page { page: PageId },
    Section { page: PageId, anchor: AnchorId },
    Symbol { symbol: SymbolId },
    Source { loc: SourceLoc },
    External { url: String },
}

impl Target {
    pub fn page(id: impl Into<PageId>) -> Self {
        Target::Page { page: id.into() }
    }
}

/// Lines of a file in the indexed tree. `start == 0` means the whole file.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SourceLoc {
    pub path: String,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub start: u32,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub end: u32,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

impl SourceLoc {
    pub fn file(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            start: 0,
            end: 0,
        }
    }

    pub fn lines(path: impl Into<String>, start: u32, end: u32) -> Self {
        Self {
            path: path.into(),
            start,
            end: end.max(start),
        }
    }

    /// `src/a.rs`, `src/a.rs:10` or `src/a.rs:10-20`.
    pub fn label(&self) -> String {
        match (self.start, self.end) {
            (0, _) => self.path.clone(),
            (s, e) if e > s => format!("{}:{s}-{e}", self.path),
            (s, _) => format!("{}:{s}", self.path),
        }
    }
}

/// Where the source lives on the web, for permalinks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepoInfo {
    /// `https://github.com/owner/repo`
    pub web_url: String,
    pub host: RepoHost,
    /// Full commit sha the site was built from.
    pub commit: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RepoHost {
    GitHub,
    GitLab,
    Other,
}

impl RepoInfo {
    /// Detect from `git remote get-url origin` and `HEAD`. `None` outside git or with
    /// a remote we cannot build web links for.
    pub fn detect(workspace: &Path) -> Option<Self> {
        let git = |args: &[&str]| -> Option<String> {
            let out = std::process::Command::new("git")
                .args(args)
                .current_dir(workspace)
                .output()
                .ok()?;
            out.status
                .success()
                .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
                .filter(|s| !s.is_empty())
        };
        let remote = git(&["remote", "get-url", "origin"])?;
        let commit = git(&["rev-parse", "HEAD"])?;
        Self::from_remote(&remote, commit)
    }

    /// Parse a git remote (https or scp-style ssh) into a web URL.
    pub fn from_remote(remote: &str, commit: String) -> Option<Self> {
        let remote = remote.trim();
        let (host, path) = if let Some(rest) = remote.strip_prefix("git@") {
            rest.split_once(':')?
        } else {
            let rest = remote
                .strip_prefix("https://")
                .or_else(|| remote.strip_prefix("http://"))
                .or_else(|| remote.strip_prefix("ssh://git@"))?;
            let rest = rest.rsplit_once('@').map(|(_, r)| r).unwrap_or(rest);
            rest.split_once('/')?
        };
        let host = host.split(':').next()?; // drop an ssh port
        let path = path.trim_end_matches('/').trim_end_matches(".git");
        if path.is_empty() || host.is_empty() {
            return None;
        }
        let kind = match host {
            "github.com" => RepoHost::GitHub,
            h if h == "gitlab.com" || h.starts_with("gitlab.") => RepoHost::GitLab,
            _ => RepoHost::Other,
        };
        Some(Self {
            web_url: format!("https://{host}/{path}"),
            host: kind,
            commit,
        })
    }

    /// Permalink to lines of a file at the site's commit.
    pub fn blob_url(&self, loc: &SourceLoc) -> Option<String> {
        let blob = match self.host {
            RepoHost::GitHub => "blob",
            RepoHost::GitLab => "-/blob",
            RepoHost::Other => return None,
        };
        let mut url = format!("{}/{blob}/{}/{}", self.web_url, self.commit, loc.path);
        match (self.host, loc.start, loc.end) {
            (_, 0, _) => {}
            (RepoHost::GitHub, s, e) if e > s => url.push_str(&format!("#L{s}-L{e}")),
            (RepoHost::GitLab, s, e) if e > s => url.push_str(&format!("#L{s}-{e}")),
            (_, s, _) => url.push_str(&format!("#L{s}")),
        }
        Some(url)
    }
}

/// A resolved link.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    /// Base-less route (`/docs/…/#anchor`) or an absolute external URL.
    pub href: String,
    pub external: bool,
}

/// Resolves [`Target`]s against one site.
pub struct Linker<'a> {
    site: &'a Site,
}

impl<'a> Linker<'a> {
    pub fn new(site: &'a Site) -> Self {
        Self { site }
    }

    pub fn resolve(&self, target: &Target) -> Option<Resolved> {
        let internal = |href: String| Resolved {
            href,
            external: false,
        };
        match target {
            Target::Page { page } => self.site.pages.get(page).map(|p| internal(p.route.clone())),
            Target::Section { page, anchor } => self
                .site
                .pages
                .get(page)
                .map(|p| internal(format!("{}#{anchor}", p.route))),
            Target::Symbol { .. } => None,
            Target::Source { loc } => self
                .site
                .meta
                .repo
                .as_ref()
                .and_then(|r| r.blob_url(loc))
                .map(|href| Resolved {
                    href,
                    external: true,
                }),
            Target::External { url } => Some(Resolved {
                href: url.clone(),
                external: true,
            }),
        }
    }

    /// Every internal link that does not resolve, as `(page, target)`. Source links
    /// without a repo are not broken: they render as plain paths.
    pub fn broken(&self) -> Vec<(PageId, Target)> {
        let mut out = Vec::new();
        for page in self.site.pages.values() {
            let mut targets = Vec::new();
            collect_block_targets(&page.blocks, &mut targets);
            for t in targets {
                let checkable = matches!(
                    t,
                    Target::Page { .. } | Target::Section { .. } | Target::Symbol { .. }
                );
                if checkable && self.resolve(&t).is_none() {
                    out.push((page.id.clone(), t));
                }
            }
        }
        for tab in &self.site.tabs {
            let mut ids = vec![tab.landing.clone()];
            super::collect_nav_pages(&tab.nav, &mut ids);
            for id in ids {
                if !self.site.pages.contains_key(&id) {
                    out.push((tab.landing.clone(), Target::Page { page: id }));
                }
            }
        }
        out.sort();
        out.dedup();
        out
    }
}

fn collect_inline_targets(inlines: &[Inline], out: &mut Vec<Target>) {
    for i in inlines {
        match i {
            Inline::Link { to, content } => {
                out.push(to.clone());
                collect_inline_targets(content, out);
            }
            Inline::Strong { content } | Inline::Emph { content } => {
                collect_inline_targets(content, out)
            }
            Inline::Text { .. } | Inline::Code { .. } | Inline::Fact { .. } => {}
        }
    }
}

pub(super) fn collect_block_targets(blocks: &[Block], out: &mut Vec<Target>) {
    for b in blocks {
        match b {
            Block::Paragraph { content } => collect_inline_targets(content, out),
            Block::List { items, .. } => {
                for item in items {
                    collect_inline_targets(item, out);
                }
            }
            Block::Table { rows, .. } => {
                for cell in rows.iter().flatten() {
                    collect_inline_targets(cell, out);
                }
            }
            Block::Callout { body, .. } => collect_block_targets(body, out),
            Block::Narrative { fallback, .. } => collect_block_targets(fallback, out),
            Block::Diagram { links, .. } => out.extend(links.iter().map(|(_, t)| t.clone())),
            Block::Cards { cards } => out.extend(cards.iter().map(|c| c.to.clone())),
            Block::Heading { .. }
            | Block::Markdown { .. }
            | Block::Code { .. }
            | Block::Stats { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remotes_parse() {
        let c = "abc123".to_string();
        let r =
            RepoInfo::from_remote("git@github.com:reflex-search/reflex.git", c.clone()).unwrap();
        assert_eq!(r.web_url, "https://github.com/reflex-search/reflex");
        assert_eq!(r.host, RepoHost::GitHub);
        let r = RepoInfo::from_remote("https://github.com/o/r", c.clone()).unwrap();
        assert_eq!(r.web_url, "https://github.com/o/r");
        let r = RepoInfo::from_remote("https://token@gitlab.com/g/sub/r.git", c.clone()).unwrap();
        assert_eq!(r.web_url, "https://gitlab.com/g/sub/r");
        assert_eq!(r.host, RepoHost::GitLab);
        let r =
            RepoInfo::from_remote("ssh://git@gitlab.example.com:2222/g/r.git", c.clone()).unwrap();
        assert_eq!(r.web_url, "https://gitlab.example.com/g/r");
        assert!(RepoInfo::from_remote("/local/path", c).is_none());
    }

    #[test]
    fn blob_urls() {
        let gh = RepoInfo::from_remote("git@github.com:o/r.git", "sha".into()).unwrap();
        assert_eq!(
            gh.blob_url(&SourceLoc::lines("src/a.rs", 10, 20)).unwrap(),
            "https://github.com/o/r/blob/sha/src/a.rs#L10-L20"
        );
        assert_eq!(
            gh.blob_url(&SourceLoc::lines("src/a.rs", 7, 7)).unwrap(),
            "https://github.com/o/r/blob/sha/src/a.rs#L7"
        );
        assert_eq!(
            gh.blob_url(&SourceLoc::file("README.md")).unwrap(),
            "https://github.com/o/r/blob/sha/README.md"
        );
        let gl = RepoInfo::from_remote("https://gitlab.com/o/r", "sha".into()).unwrap();
        assert_eq!(
            gl.blob_url(&SourceLoc::lines("a", 1, 3)).unwrap(),
            "https://gitlab.com/o/r/-/blob/sha/a#L1-3"
        );
        let other = RepoInfo::from_remote("https://git.example.com/o/r", "sha".into()).unwrap();
        assert!(other.blob_url(&SourceLoc::file("a")).is_none());
    }

    #[test]
    fn source_loc_labels() {
        assert_eq!(SourceLoc::file("a.rs").label(), "a.rs");
        assert_eq!(SourceLoc::lines("a.rs", 4, 4).label(), "a.rs:4");
        assert_eq!(SourceLoc::lines("a.rs", 4, 9).label(), "a.rs:4-9");
    }
}
