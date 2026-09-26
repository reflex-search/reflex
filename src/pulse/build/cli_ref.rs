//! The command-line reference: one page per command.
//!
//! Each page shows usage, the help text, positional arguments, options with their
//! defaults, and links to subcommands. Hidden commands and options are left out.

use super::{PageSpec, SiteBuilder};
use crate::pulse::extract::cli::{CliArg, CliCommand};
use crate::pulse::model::ids::slugify;
use crate::pulse::model::{
    Block, Inline, MarkdownOrigin, MarkdownText, NavNode, PageId, PageKind, SourceLoc, TabId,
    Target,
};

fn page_id(cmd: &CliCommand) -> PageId {
    PageId(format!("docs/cli/{}", cmd.path.join(" ")))
}

/// Build pages for every command tree; returns nav nodes (one group per tool).
pub fn build(b: &mut SiteBuilder, roots: &[CliCommand]) -> Vec<NavNode> {
    let mut nav = Vec::new();
    for root in roots {
        add(b, root);
        nav.push(NavNode::Group {
            label: format!("{} CLI", root.name()),
            collapsed: false,
            children: nav_for(root, true),
        });
    }
    nav
}

fn visible(cmd: &CliCommand) -> impl Iterator<Item = &CliCommand> {
    cmd.subcommands.iter().filter(|c| !c.hidden)
}

fn nav_for(cmd: &CliCommand, root: bool) -> Vec<NavNode> {
    let mut out = vec![NavNode::page(&page_id(cmd))];
    for c in visible(cmd) {
        if c.subcommands.iter().any(|s| !s.hidden) {
            out.push(NavNode::Group {
                label: c.name().to_string(),
                collapsed: true,
                children: nav_for(c, false),
            });
        } else {
            out.push(NavNode::page(&page_id(c)));
        }
    }
    let _ = root;
    out
}

fn help_markdown(cmd: &CliCommand) -> Option<MarkdownText> {
    let doc = cmd.doc.as_ref()?;
    // Clap help is plain text with hand-indented examples: keep line breaks, and fence
    // indented runs so they render as code instead of collapsing.
    let mut out: Vec<String> = Vec::new();
    let mut in_code = false;
    for line in doc.markdown.lines() {
        let indented = line.starts_with("  ") && !line.trim().is_empty();
        if indented && !in_code {
            out.push("```text".into());
            in_code = true;
        } else if !indented && in_code && !line.trim().is_empty() {
            out.push("```".into());
            in_code = false;
        }
        out.push(if in_code {
            line.trim_start_matches("  ").to_string()
        } else {
            line.to_string()
        });
    }
    if in_code {
        while out.last().is_some_and(|l| l.trim().is_empty()) {
            out.pop();
        }
        out.push("```".into());
    }
    Some(MarkdownText {
        source: out.join("\n"),
        origin: MarkdownOrigin::Comment,
        from: Some(SourceLoc::lines(&cmd.file, doc.start_line, doc.end_line)),
    })
}

fn arg_help(a: &CliArg) -> Vec<Inline> {
    a.doc
        .as_ref()
        .map(|d| vec![Inline::text(d.summary.clone())])
        .unwrap_or_default()
}

fn add(b: &mut SiteBuilder, cmd: &CliCommand) {
    for c in visible(cmd) {
        add(b, c);
    }
    let mut blocks = vec![Block::Code {
        lang: "sh".into(),
        code: cmd.usage(),
        title: Some("Usage".into()),
    }];
    match help_markdown(cmd) {
        Some(md) => blocks.push(Block::Markdown { markdown: md }),
        None => {
            if let Some(about) = &cmd.about {
                blocks.push(Block::text(about.clone()));
            }
        }
    }

    let subs: Vec<&CliCommand> = visible(cmd).collect();
    if !subs.is_empty() {
        blocks.push(Block::heading(2, "Commands"));
        blocks.push(Block::Table {
            columns: vec!["Command".into(), "Description".into()],
            rows: subs
                .iter()
                .map(|c| {
                    let summary = c
                        .about
                        .clone()
                        .or_else(|| c.doc.as_ref().map(|d| d.summary.clone()))
                        .unwrap_or_default();
                    vec![
                        vec![Inline::code_link(Target::page(page_id(c)), c.name())],
                        vec![Inline::text(summary)],
                    ]
                })
                .collect(),
        });
    }

    let positional: Vec<&CliArg> = cmd
        .args
        .iter()
        .filter(|a| a.positional && !a.hidden)
        .collect();
    if !positional.is_empty() {
        blocks.push(Block::heading(2, "Arguments"));
        blocks.push(Block::Table {
            columns: vec!["Argument".into(), "Default".into(), "Description".into()],
            rows: positional
                .iter()
                .map(|a| {
                    vec![
                        vec![Inline::code(a.usage())],
                        a.default.iter().map(|d| Inline::code(d.clone())).collect(),
                        arg_help(a),
                    ]
                })
                .collect(),
        });
    }
    let options: Vec<&CliArg> = cmd
        .args
        .iter()
        .filter(|a| !a.positional && !a.hidden)
        .collect();
    if !options.is_empty() {
        blocks.push(Block::heading(2, "Options"));
        blocks.push(Block::Table {
            columns: vec!["Option".into(), "Default".into(), "Description".into()],
            rows: options
                .iter()
                .map(|a| {
                    vec![
                        vec![Inline::code(a.usage())],
                        a.default.iter().map(|d| Inline::code(d.clone())).collect(),
                        arg_help(a),
                    ]
                })
                .collect(),
        });
    }
    blocks.push(Block::para(vec![
        Inline::text("Defined in "),
        Inline::code_link(
            Target::Source {
                loc: SourceLoc::lines(&cmd.file, cmd.line, cmd.line),
            },
            format!("{}:{}", cmd.file, cmd.line),
        ),
        Inline::text("."),
    ]));

    let description = cmd
        .about
        .clone()
        .or_else(|| cmd.doc.as_ref().map(|d| d.summary.clone()))
        .filter(|s| !s.is_empty());
    let slug = format!(
        "cli/{}",
        cmd.path
            .iter()
            .map(|s| slugify(s))
            .collect::<Vec<_>>()
            .join("/")
    );
    b.add_page(PageSpec {
        id: page_id(cmd),
        tab: TabId::Docs,
        kind: PageKind::CliCommand {
            command: cmd.path.join(" "),
        },
        title: cmd.path.join(" "),
        description,
        slug: Some(slug),
        badges: vec!["command".into()],
        blocks,
    });
}
