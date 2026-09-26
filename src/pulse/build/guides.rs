//! Guides: the repository's own Markdown documents as site pages.
//!
//! Candidates are Markdown files at the repository root or under `docs/`, `doc/` or
//! `guides/`. `README` feeds the home page and `CHANGELOG` the changelog; agent
//! instruction files (`CLAUDE.md`, `AGENTS.md`, …) and licences are skipped.
//! Contributor documents (contributing, testing, release, architecture, design notes)
//! go to Internals → Contributing; the rest to Docs → Guides.
//!
//! Relative links between documents become links to their pages; links to indexed
//! files become source permalinks (or plain text without a repository URL).

use super::links::{PAGE_SCHEME, SOURCE_SCHEME};
use super::{PageSpec, SiteBuilder};
use crate::pulse::extract::{ContentAccess, Corpus, FileRole};
use crate::pulse::model::ids::slugify;
use crate::pulse::model::{
    Block, MarkdownOrigin, MarkdownText, NavNode, PageId, PageKind, SourceLoc, TabId,
};
use std::collections::HashMap;
use std::sync::LazyLock;

/// A document that becomes a guide page.
#[derive(Debug, Clone)]
pub struct Guide {
    pub path: String,
    pub title: String,
    /// Markdown without front matter and without the title heading.
    pub body: String,
    /// Line of `body`'s first line in the file (1-based).
    pub body_line: u32,
    pub tab: TabId,
    pub order: i64,
    pub page: PageId,
}

const DOC_DIRS: &[&str] = &["docs", "doc", "guides", "documentation"];
const SKIP_NAMES: &[&str] = &[
    "readme",
    "changelog",
    "history",
    "changes",
    "license",
    "licence",
    "copying",
    "notice",
    "claude",
    "agents",
    "gemini",
    "copilot-instructions",
    "reflex",
    "authors",
    "contributors",
];
const INTERNAL_NAMES: &[&str] = &[
    "contributing",
    "release",
    "releasing",
    "testing",
    "architecture",
    "development",
    "hacking",
    "maintaining",
    "maintainers",
    "governance",
    "code_of_conduct",
    "code-of-conduct",
    "design",
    "internals",
];
const INTERNAL_DIRS: &[&str] = &[
    "features",
    "internal",
    "internals",
    "dev",
    "development",
    "contributing",
    "adr",
    "adrs",
    "design",
    "rfc",
    "rfcs",
    "decisions",
];

fn stem(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.rsplit_once('.')
        .map(|(s, _)| s)
        .unwrap_or(name)
        .to_string()
}

/// Whether a Markdown file is a guide candidate, and for which tab.
pub fn classify(path: &str) -> Option<TabId> {
    let lower = path.to_ascii_lowercase();
    let parts: Vec<&str> = lower.split('/').collect();
    let s = stem(&lower);
    let in_docs = parts.len() > 1 && DOC_DIRS.contains(&parts[0]);
    if !(parts.len() == 1 || in_docs) || SKIP_NAMES.contains(&s.as_str()) {
        return None;
    }
    let internal = INTERNAL_NAMES.contains(&s.as_str())
        || parts[..parts.len() - 1]
            .iter()
            .skip(1)
            .any(|d| INTERNAL_DIRS.contains(d));
    Some(if internal {
        TabId::Internals
    } else {
        TabId::Docs
    })
}

/// `ai-agent-integration` → `Ai agent integration`.
fn title_from_stem(s: &str) -> String {
    let words = s.replace(['-', '_'], " ");
    let mut c = words.chars();
    match c.next() {
        Some(f) => f
            .to_uppercase()
            .chain(c.flat_map(|c| c.to_lowercase()))
            .collect(),
        None => s.to_string(),
    }
}

/// Split front matter, the title heading and the body.
fn parse(path: &str, content: &str) -> (String, String, u32, i64) {
    let mut lines: Vec<&str> = content.lines().collect();
    let mut offset = 0u32;
    let mut title = None;
    let mut order = i64::MAX;
    if lines.first() == Some(&"---")
        && let Some(end) = lines.iter().skip(1).position(|l| *l == "---")
    {
        for l in &lines[1..=end] {
            if let Some((k, v)) = l.split_once(':') {
                let v = v.trim().trim_matches(['"', '\'']);
                match k.trim() {
                    "title" => title = Some(v.to_string()),
                    "order" | "sidebar_position" | "weight" | "nav_order" => {
                        order = v.parse().unwrap_or(order)
                    }
                    _ => {}
                }
            }
        }
        offset = end as u32 + 2;
        lines.drain(..end + 2);
    }
    // Leading blank lines, then an optional `# Title`.
    while lines.first().is_some_and(|l| l.trim().is_empty()) {
        lines.remove(0);
        offset += 1;
    }
    if let Some(first) = lines.first()
        && let Some(h) = first.strip_prefix("# ")
    {
        if title.is_none() {
            title = Some(h.trim().to_string());
        }
        lines.remove(0);
        offset += 1;
    }
    let title = title.unwrap_or_else(|| title_from_stem(&stem(path)));
    (title, lines.join("\n"), offset + 1, order)
}

pub fn collect(corpus: &Corpus, content: &ContentAccess) -> Vec<Guide> {
    let mut out = Vec::new();
    for (_, f) in corpus.with_role(FileRole::Docs) {
        let lower = f.path.to_ascii_lowercase();
        if !(lower.ends_with(".md") || lower.ends_with(".mdx") || lower.ends_with(".markdown")) {
            continue;
        }
        let Some(tab) = classify(&f.path) else {
            continue;
        };
        let Some(text) = content.read(&f.path) else {
            continue;
        };
        let (title, body, body_line, order) = parse(&f.path, text);
        if body.trim().is_empty() {
            continue;
        }
        let page = PageId(format!(
            "{}/guide/{}",
            if tab == TabId::Docs { "docs" } else { "int" },
            f.path
        ));
        out.push(Guide {
            path: f.path.clone(),
            title,
            body,
            body_line,
            tab,
            order,
            page,
        });
    }
    out.sort_by(|a, b| (a.tab, a.order, &a.title).cmp(&(b.tab, b.order, &b.title)));
    out
}

/// Resolve `rel` against `dir` (a leading `/` means the repository root).
fn normalize(dir: &str, rel: &str) -> String {
    let (base, rel) = match rel.strip_prefix('/') {
        Some(r) => ("", r),
        None => (dir, rel),
    };
    let mut parts: Vec<&str> = if base.is_empty() {
        Vec::new()
    } else {
        base.split('/').collect()
    };
    for seg in rel.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            s => parts.push(s),
        }
    }
    parts.join("/")
}

static MD_LINK: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"\]\(([^)\s]+)\)").expect("valid regex"));

/// Rewrite relative links: other guides → `pulse-page:`, indexed files → `pulse-source:`.
pub fn rewrite_links(
    md: &str,
    from: &str,
    guides: &HashMap<String, PageId>,
    corpus: &Corpus,
) -> String {
    let dir = from.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
    let mut out = Vec::new();
    let mut fence = false;
    for line in md.lines() {
        let t = line.trim_start();
        if t.starts_with("```") || t.starts_with("~~~") {
            fence = !fence;
        }
        if fence {
            out.push(line.to_string());
            continue;
        }
        let rewritten = MD_LINK.replace_all(line, |c: &regex::Captures| {
            let dest = &c[1];
            let lower = dest.to_ascii_lowercase();
            if lower.starts_with("http:")
                || lower.starts_with("https:")
                || lower.starts_with("mailto:")
                || dest.starts_with('#')
                || lower.contains("://")
            {
                return c[0].to_string();
            }
            let (path, anchor) = match dest.split_once('#') {
                Some((p, a)) => (p, Some(a)),
                None => (dest, None),
            };
            let target = normalize(dir, path);
            let hash = anchor.map(|a| format!("#{a}")).unwrap_or_default();
            if let Some(page) = guides.get(&target) {
                return format!("]({PAGE_SCHEME}{}{hash})", page.0);
            }
            if corpus.index_of(&target).is_some() {
                return format!("]({SOURCE_SCHEME}{target})");
            }
            c[0].to_string()
        });
        out.push(rewritten.into_owned());
    }
    out.join("\n")
}

/// Add guide pages; returns (Docs nav, Internals nav).
pub fn build(
    b: &mut SiteBuilder,
    corpus: &Corpus,
    guides: &[Guide],
) -> (Vec<NavNode>, Vec<NavNode>) {
    let by_path: HashMap<String, PageId> = guides
        .iter()
        .map(|g| (g.path.clone(), g.page.clone()))
        .collect();
    let mut docs = Vec::new();
    let mut internals = Vec::new();
    for g in guides {
        let body = rewrite_links(&g.body, &g.path, &by_path, corpus);
        let end = g.body_line + g.body.lines().count() as u32;
        let first_para = g
            .body
            .split("\n\n")
            .map(str::trim)
            .find(|p| {
                !p.is_empty()
                    && !p.starts_with('#')
                    && !p.starts_with("```")
                    && !p.starts_with('|')
                    && !p.starts_with('>')
            })
            .map(|p| {
                let flat: String = p.split_whitespace().collect::<Vec<_>>().join(" ");
                let plain = flat.replace(['*', '`'], "");
                plain.chars().take(180).collect::<String>()
            });
        let slug_root = if g.tab == TabId::Docs {
            "guides"
        } else {
            "contributing"
        };
        let id = b.add_page(PageSpec {
            id: g.page.clone(),
            tab: g.tab,
            kind: PageKind::Guide {
                source: g.path.clone(),
            },
            title: g.title.clone(),
            description: first_para,
            slug: Some(format!("{slug_root}/{}", slugify(&stem(&g.path)))),
            badges: vec![],
            blocks: vec![Block::Markdown {
                markdown: MarkdownText {
                    source: body,
                    origin: MarkdownOrigin::Doc,
                    from: Some(SourceLoc::lines(&g.path, g.body_line, end)),
                },
            }],
        });
        match g.tab {
            TabId::Docs => docs.push(NavNode::page(&id)),
            TabId::Internals => internals.push(NavNode::page(&id)),
        }
    }
    (docs, internals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification() {
        assert_eq!(classify("docs/API.md"), Some(TabId::Docs));
        assert_eq!(classify("docs/mcp-tool-cheatsheet.md"), Some(TabId::Docs));
        assert_eq!(classify("docs/ARCHITECTURE.md"), Some(TabId::Internals));
        assert_eq!(classify("docs/features/PULSE.md"), Some(TabId::Internals));
        assert_eq!(classify("CONTRIBUTING.md"), Some(TabId::Internals));
        assert_eq!(classify("SECURITY.md"), Some(TabId::Docs));
        for skip in [
            "README.md",
            "CHANGELOG.md",
            "CLAUDE.md",
            "AGENTS.md",
            "LICENSE.md",
            "src/semantic/prompt.md",
            "benches/x/README.md",
            "docs/README.md",
        ] {
            assert_eq!(classify(skip), None, "{skip}");
        }
    }

    #[test]
    fn front_matter_title_and_body() {
        let (t, body, line, order) = parse(
            "docs/a.md",
            "---\ntitle: \"Custom\"\norder: 3\n---\n\n# Ignored\nBody\n",
        );
        assert_eq!(
            (t.as_str(), body.as_str(), line, order),
            ("Custom", "Body", 7, 3)
        );
        let (t, body, line, _) = parse("docs/ai-agent-integration.md", "Intro\n");
        assert_eq!(
            (t.as_str(), body.as_str(), line),
            ("Ai agent integration", "Intro", 1)
        );
        let (t, _, line, _) = parse("x.md", "# Hello World\n\ntext");
        assert_eq!((t.as_str(), line), ("Hello World", 2));
    }

    #[test]
    fn paths_normalize() {
        assert_eq!(normalize("docs", "../README.md"), "README.md");
        assert_eq!(
            normalize("docs/features", "./PULSE.md"),
            "docs/features/PULSE.md"
        );
        assert_eq!(normalize("docs", "/src/main.rs"), "src/main.rs");
    }
}
