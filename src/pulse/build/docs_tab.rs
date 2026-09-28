//! The Docs tab: the project for people who use it.
//!
//! - Home (landing): the README introduction, headline numbers, entry cards, and the
//!   narrative slot `project-overview`.
//! - Guides and Reference: added by `build::guides`, `build::cli_ref` and `build::reference`.
//! - Changelog: an index of releases and one page per release (`build::releases`):
//!   CHANGELOG.md notes, public API changes, commits grouped by conventional type.

use super::internals_tab::{ARCHITECTURE, DEPENDENCY_MAP};
use super::modules::ModuleGraph;
use super::releases::Release;
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
    releases: &[Release],
    guide_nav: Vec<NavNode>,
    reference_nav: Vec<NavNode>,
) {
    // One page per release, then the index that lists them.
    let mut release_nav = Vec::new();
    for r in releases {
        let id = PageId(format!("docs/release/{}", r.version));
        let blocks = release_blocks(b, r);
        let slug = if r.tag.is_none() {
            "changelog/unreleased".to_string()
        } else {
            format!(
                "changelog/{}",
                crate::pulse::model::ids::slugify(&r.version)
            )
        };
        b.add_page(PageSpec {
            id: id.clone(),
            tab: TabId::Docs,
            kind: PageKind::Changelog,
            title: if r.tag.is_some() {
                format!("v{}", r.version)
            } else {
                "Unreleased".into()
            },
            description: Some(release_summary(r)),
            slug: Some(slug),
            badges: if r.date.is_empty() {
                vec![]
            } else {
                vec![r.date.clone()]
            },
            blocks,
        });
        release_nav.push(NavNode::page(&id));
    }
    let changelog = b.add_page(PageSpec {
        id: PageId::new(CHANGELOG),
        tab: TabId::Docs,
        kind: PageKind::Changelog,
        title: "Changelog".into(),
        description: Some("Releases, newest first.".into()),
        slug: Some("changelog".into()),
        badges: vec![],
        blocks: changelog_index(releases),
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
            .and_then(|(intro, _)| {
                let (lead, _) = split_lead(&intro);
                lead.or_else(|| first_sentence(&intro))
            }),
        slug: None,
        badges: vec![],
        blocks: home_blocks,
    });

    let mut nav = Vec::new();
    if !guide_nav.is_empty() {
        nav.push(NavNode::Group {
            label: "Guides".into(),
            collapsed: false,
            children: guide_nav,
        });
    }
    if !reference_nav.is_empty() {
        nav.push(NavNode::Group {
            label: "Reference".into(),
            collapsed: false,
            children: reference_nav,
        });
    }
    let mut changelog_group = vec![NavNode::page(&changelog)];
    changelog_group.extend(release_nav);
    nav.push(NavNode::Group {
        label: "Changelog".into(),
        collapsed: true,
        children: changelog_group,
    });
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
/// A README that opens with a short bold line (`**Fast local search**`) has a tagline:
/// return it as plain text, and the intro without it.
fn split_lead(intro: &str) -> (Option<String>, &str) {
    let (first, rest) = intro.split_once("\n\n").unwrap_or((intro, ""));
    let t = first.trim();
    let bold = t.starts_with("**")
        && t.ends_with("**")
        && t.len() > 4
        && !t[2..t.len() - 2].contains("**");
    if bold && t.len() < 160 && !rest.trim().is_empty() {
        (
            Some(t.trim_matches('*').trim().to_string()),
            rest.trim_start(),
        )
    } else {
        (None, intro)
    }
}

fn readme_block(readme: &DocFile) -> Option<Block> {
    let (full, (start, end)) = readme_intro(&readme.content)?;
    let (_, body) = split_lead(&full);
    let intro = body.to_string();
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

fn commit_item(b: &SiteBuilder, c: &ChangelogCommit) -> Vec<Inline> {
    let mut item = Vec::new();
    match conventional(&c.subject) {
        Some((kind, scope, breaking, rest)) => {
            if let Some(s) = scope {
                item.push(Inline::code(s.to_string()));
                item.push(Inline::text(" "));
            }
            if breaking {
                item.push(Inline::strong("breaking"));
                item.push(Inline::text(" "));
            }
            let _ = kind;
            item.push(Inline::text(rest.to_string()));
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
    item
}

fn release_summary(r: &Release) -> String {
    let api = r.api.total();
    let mut s = format!(
        "{} commit{}",
        r.commits.len(),
        if r.commits.len() == 1 { "" } else { "s" }
    );
    if api > 0 {
        s.push_str(&format!(
            ", {api} public API change{}",
            if api == 1 { "" } else { "s" }
        ));
    }
    if !r.date.is_empty() {
        s.push_str(&format!(", {}", r.date));
    }
    s.push('.');
    s
}

fn changelog_index(releases: &[Release]) -> Vec<Block> {
    if releases.is_empty() {
        return vec![Block::text("No git history was found for this index.")];
    }
    vec![Block::Table {
        columns: vec![
            "Release".into(),
            "Date".into(),
            "Commits".into(),
            "API changes".into(),
        ],
        rows: releases
            .iter()
            .map(|r| {
                let title = if r.tag.is_some() {
                    format!("v{}", r.version)
                } else {
                    "Unreleased".into()
                };
                let api = format!(
                    "+{} −{} ~{}",
                    r.api.added_total, r.api.removed_total, r.api.changed_total
                );
                vec![
                    vec![Inline::link(
                        Target::page(PageId(format!("docs/release/{}", r.version))),
                        title,
                    )],
                    vec![Inline::text(r.date.clone())],
                    vec![Inline::text(r.commits.len().to_string())],
                    vec![Inline::code(api)],
                ]
            })
            .collect(),
    }]
}

fn release_blocks(b: &SiteBuilder, r: &Release) -> Vec<Block> {
    let mut blocks = Vec::new();
    if let Some((md, start, end)) = &r.notes {
        blocks.push(Block::Markdown {
            markdown: MarkdownText {
                source: md.clone(),
                origin: MarkdownOrigin::Doc,
                from: Some(SourceLoc::lines("CHANGELOG.md", *start, *end)),
            },
        });
    }

    if !r.api.is_empty() {
        blocks.push(Block::heading(2, "API changes"));
        blocks.push(Block::text(
            "Public items whose declaration was added, removed or changed, found by comparing \
             both versions of every changed source file.",
        ));
        let table = |list: &[crate::pulse::build::releases::ApiChange]| Block::Table {
            columns: vec!["Item".into(), "Kind".into(), "Declaration".into()],
            rows: list
                .iter()
                .map(|c| {
                    vec![
                        vec![Inline::code(c.name.clone())],
                        vec![Inline::text(c.kind)],
                        vec![Inline::code(c.signature.clone())],
                    ]
                })
                .collect(),
        };
        if !r.api.added.is_empty() {
            blocks.push(Block::heading(3, "Added"));
            blocks.push(table(&r.api.added));
        }
        if !r.api.removed.is_empty() {
            blocks.push(Block::heading(3, "Removed"));
            blocks.push(table(&r.api.removed));
        }
        if !r.api.changed.is_empty() {
            blocks.push(Block::heading(3, "Changed"));
            blocks.push(Block::Table {
                columns: vec!["Item".into(), "Before".into(), "After".into()],
                rows: r
                    .api
                    .changed
                    .iter()
                    .map(|(c, before)| {
                        vec![
                            vec![Inline::code(c.name.clone())],
                            vec![Inline::code(before.clone())],
                            vec![Inline::code(c.signature.clone())],
                        ]
                    })
                    .collect(),
            });
        }
        if r.api.omitted() > 0 {
            blocks.push(Block::note(format!(
                "{} more API changes are not listed.",
                r.api.omitted()
            )));
        }
    }

    if !r.commits.is_empty() {
        blocks.push(Block::heading(2, "Commits"));
        let groups: [(&str, &[&str]); 5] = [
            ("Features", &["feat"]),
            ("Fixes", &["fix"]),
            ("Performance", &["perf"]),
            ("Documentation", &["docs"]),
            ("Other changes", &[]),
        ];
        fn kind_of(c: &ChangelogCommit) -> &str {
            conventional(&c.subject).map(|(k, ..)| k).unwrap_or("")
        }
        for (title, kinds) in groups {
            let items: Vec<Vec<Inline>> = r
                .commits
                .iter()
                .filter(|c| {
                    let k = kind_of(c);
                    if kinds.is_empty() {
                        !["feat", "fix", "perf", "docs"].contains(&k)
                    } else {
                        kinds.contains(&k)
                    }
                })
                .map(|c| commit_item(b, c))
                .collect();
            if !items.is_empty() {
                blocks.push(Block::heading(3, title));
                blocks.push(Block::List {
                    ordered: false,
                    items,
                });
            }
        }
    }
    if blocks.is_empty() {
        blocks.push(Block::text("No changes recorded."));
    }
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
    fn bold_lead_line_is_a_tagline() {
        let (lead, body) = split_lead("**Fast local search**\n\nReflex is a tool.");
        assert_eq!(lead.as_deref(), Some("Fast local search"));
        assert_eq!(body, "Reflex is a tool.");
        let (lead, body) = split_lead("Reflex is **fast**.\n\nMore.");
        assert!(lead.is_none());
        assert_eq!(body, "Reflex is **fast**.\n\nMore.");
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
