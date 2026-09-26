//! The Docs tab: the project for people who use it.
//!
//! - Home (landing): the README introduction, headline numbers, entry cards, and the
//!   narrative slot `project-overview`.
//! - Changelog: recent commits grouped by day, typed by conventional-commit prefix.
//!
//! Getting started, guides and the API reference join this tab in later milestones.

use super::internals_tab::{ARCHITECTURE, DEPENDENCY_MAP};
use super::modules::ModuleGraph;
use super::{PageSpec, SiteBuilder};
use crate::pulse::changelog::ChangelogCommit;
use crate::pulse::extract::{Corpus, DocFile};
use crate::pulse::model::{
    Block, Card, Inline, MarkdownOrigin, MarkdownText, NavNode, PageId, PageKind, SourceLoc, Stat,
    Subject, TabId, Target,
};

pub const HOME: &str = "docs/home";
pub const CHANGELOG: &str = "docs/changelog";

/// Longest README introduction shown on the home page.
const MAX_INTRO_CHARS: usize = 1500;

pub fn build(
    b: &mut SiteBuilder,
    corpus: &Corpus,
    _graph: &ModuleGraph,
    commits: &[ChangelogCommit],
    reference_nav: Vec<NavNode>,
) {
    let changelog = b.add_page(PageSpec {
        id: PageId::new(CHANGELOG),
        tab: TabId::Docs,
        kind: PageKind::Changelog,
        title: "Changelog".into(),
        description: Some("Recent changes, newest first.".into()),
        slug: Some("changelog".into()),
        badges: vec![],
        blocks: changelog_blocks(b, commits),
    });

    let home_blocks = home_blocks(b, corpus);
    let home = b.add_page(PageSpec {
        id: PageId::new(HOME),
        tab: TabId::Docs,
        kind: PageKind::Landing,
        title: b.title().to_string(),
        description: corpus
            .readme
            .as_ref()
            .and_then(|r| readme_intro(&r.content))
            .and_then(|(intro, _)| first_sentence(&intro)),
        slug: None,
        badges: vec![],
        blocks: home_blocks,
    });

    let mut nav = Vec::new();
    if !reference_nav.is_empty() {
        nav.push(NavNode::Group {
            label: "Reference".into(),
            collapsed: false,
            children: reference_nav,
        });
    }
    nav.push(NavNode::page(&changelog));
    b.add_tab(TabId::Docs, "Docs", home, nav);
}

fn home_blocks(b: &SiteBuilder, corpus: &Corpus) -> Vec<Block> {
    let site = Subject::Site;
    let fallback = match corpus.readme.as_ref().and_then(readme_block) {
        Some(block) => vec![block],
        None => vec![Block::para(vec![
            Inline::text(format!("{} has ", b.title())),
            b.fact_inline(&site, "source_files"),
            Inline::text(" source files in "),
            b.fact_inline(&site, "modules"),
            Inline::text(" modules."),
        ])],
    };
    vec![
        Block::Narrative {
            slot: crate::pulse::narrate::ids::OVERVIEW.into(),
            text: None,
            fallback,
        },
        Block::Stats {
            stats: vec![
                Stat {
                    label: "Source files".into(),
                    fact: b.fact(&site, "source_files"),
                },
                Stat {
                    label: "Lines of code".into(),
                    fact: b.fact(&site, "source_lines"),
                },
                Stat {
                    label: "Modules".into(),
                    fact: b.fact(&site, "modules"),
                },
                Stat {
                    label: "Languages".into(),
                    fact: b.fact(&site, "languages"),
                },
            ],
        },
        Block::Cards {
            cards: vec![
                Card {
                    title: "Architecture".into(),
                    to: Target::page(ARCHITECTURE),
                    description: Some("How the modules fit together.".into()),
                },
                Card {
                    title: "Dependency map".into(),
                    to: Target::page(DEPENDENCY_MAP),
                    description: Some("Most-imported files and dependency cycles.".into()),
                },
                Card {
                    title: "Changelog".into(),
                    to: Target::page(CHANGELOG),
                    description: Some("What changed recently.".into()),
                },
            ],
        },
    ]
}

/// The README introduction as a markdown block with its source lines.
fn readme_block(readme: &DocFile) -> Option<Block> {
    let (intro, (start, end)) = readme_intro(&readme.content)?;
    Some(Block::Markdown {
        markdown: MarkdownText {
            source: intro,
            origin: MarkdownOrigin::Doc,
            from: Some(SourceLoc::lines(&readme.path, start, end)),
        },
    })
}

/// The README's introduction: the text after the title and before the next heading,
/// without badge and HTML lines. Returns the text and its 1-based line range.
pub fn readme_intro(content: &str) -> Option<(String, (u32, u32))> {
    let mut out: Vec<(u32, &str)> = Vec::new();
    let mut seen_title = false;
    let mut in_fence = false;
    let mut chars = 0usize;
    for (i, line) in content.lines().enumerate() {
        let n = i as u32 + 1;
        let t = line.trim();
        if t.starts_with("```") || t.starts_with("~~~") {
            in_fence = !in_fence;
        }
        if !in_fence && t.starts_with('#') {
            if !seen_title && t.starts_with("# ") && out.is_empty() {
                seen_title = true;
                continue;
            }
            if out.iter().any(|(_, l)| !l.trim().is_empty()) {
                break;
            }
            seen_title = true;
            continue;
        }
        let noise = t.starts_with("[![")
            || t.starts_with("![")
            || t.starts_with('<')
            || t.starts_with("---")
            || (t.starts_with('[') && t.contains("]:"));
        if noise || (t.is_empty() && out.is_empty()) {
            continue;
        }
        if chars + line.len() > MAX_INTRO_CHARS && t.is_empty() {
            break;
        }
        chars += line.len() + 1;
        out.push((n, line));
    }
    while out.last().is_some_and(|(_, l)| l.trim().is_empty()) {
        out.pop();
    }
    let (first, last) = (out.first()?.0, out.last()?.0);
    let text: Vec<&str> = out.iter().map(|(_, l)| *l).collect();
    Some((text.join("\n"), (first, last)))
}

fn first_sentence(text: &str) -> Option<String> {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let plain = flat.replace("**", "").replace('`', "");
    let end = plain
        .char_indices()
        .find(|&(i, c)| c == '.' && plain[i + 1..].starts_with(' '))
        .map(|(i, _)| i + 1)
        .unwrap_or(plain.len());
    let s = plain[..end].trim();
    (!s.is_empty()).then(|| s.chars().take(200).collect())
}

/// `feat(pulse)!: subject` → ("feat", Some("pulse"), breaking, "subject").
fn conventional(subject: &str) -> Option<(&str, Option<&str>, bool, &str)> {
    let (head, rest) = subject.split_once(": ")?;
    let breaking = head.ends_with('!');
    let head = head.trim_end_matches('!');
    let (kind, scope) = match head.split_once('(') {
        Some((k, s)) => (k, Some(s.trim_end_matches(')'))),
        None => (head, None),
    };
    let known = [
        "feat", "fix", "perf", "refactor", "docs", "test", "tests", "build", "ci", "chore",
        "style", "revert", "bump",
    ];
    known
        .contains(&kind)
        .then_some((kind, scope, breaking, rest))
}

fn changelog_blocks(b: &SiteBuilder, commits: &[ChangelogCommit]) -> Vec<Block> {
    if commits.is_empty() {
        return vec![Block::text("No git history was found for this index.")];
    }
    let mut blocks = Vec::new();
    let mut day = String::new();
    let mut items: Vec<Vec<Inline>> = Vec::new();
    let flush = |blocks: &mut Vec<Block>, day: &str, items: &mut Vec<Vec<Inline>>| {
        if !items.is_empty() {
            blocks.push(Block::heading(2, day));
            blocks.push(Block::List {
                ordered: false,
                items: std::mem::take(items),
            });
        }
    };
    for c in commits {
        let date = c.date.get(..10).unwrap_or(&c.date).to_string();
        if date != day {
            flush(&mut blocks, &day, &mut items);
            day = date;
        }
        let mut item = Vec::new();
        match conventional(&c.subject) {
            Some((kind, scope, breaking, rest)) => {
                let tag = match scope {
                    Some(s) => format!("{kind}({s})"),
                    None => kind.to_string(),
                };
                item.push(Inline::code(tag));
                if breaking {
                    item.push(Inline::text(" "));
                    item.push(Inline::strong("breaking"));
                }
                item.push(Inline::text(format!(" {rest}")));
            }
            None => item.push(Inline::text(c.subject.clone())),
        }
        let short: String = c.hash.chars().take(7).collect();
        item.push(Inline::text(format!(" — {} · ", c.author)));
        item.push(match b.repo() {
            Some(r) => Inline::code_link(
                Target::External {
                    url: format!("{}/commit/{}", r.web_url, c.hash),
                },
                short,
            ),
            None => Inline::code(short),
        });
        items.push(item);
    }
    flush(&mut blocks, &day, &mut items);
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readme_intro_skips_title_and_badges() {
        let md = "# Reflex\n\n[![CI](x)](y)\n<p align=center>logo</p>\n\nLocal-first **code search**.\nFast.\n\n## Install\n\ncargo install";
        let (intro, (s, e)) = readme_intro(md).unwrap();
        assert_eq!(intro, "Local-first **code search**.\nFast.");
        assert_eq!((s, e), (6, 7));
        assert_eq!(first_sentence(&intro).unwrap(), "Local-first code search.");
    }

    #[test]
    fn readme_intro_ignores_hashes_in_code() {
        let md = "# T\n\nIntro line.\n```sh\n# comment\n```\nmore\n## Next";
        let (intro, _) = readme_intro(md).unwrap();
        assert!(intro.contains("# comment"));
        assert!(intro.ends_with("more"));
    }

    #[test]
    fn readme_intro_after_a_leading_section_heading() {
        // `# T` then `## Overview` then text: the text is the introduction.
        let (intro, _) = readme_intro("# Title\n\n## Overview\ntext\n## Next\nmore").unwrap();
        assert_eq!(intro, "text");
        assert!(readme_intro("# Title\n").is_none());
    }

    #[test]
    fn conventional_commits() {
        assert_eq!(
            conventional("feat(pulse)!: new cache"),
            Some(("feat", Some("pulse"), true, "new cache"))
        );
        assert_eq!(conventional("fix: bug"), Some(("fix", None, false, "bug")));
        assert_eq!(conventional("Merge pull request #1"), None);
        assert_eq!(conventional("Note: not a type"), None);
    }
}
