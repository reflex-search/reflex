//! Links inside Markdown, resolved once every page exists.
//!
//! Builders write links as `pulse-symbol:<id>`, `pulse-page:<id>[#anchor]` or
//! `pulse-source:<path>`, because routes are only final after all pages are added.
//! [`resolve_markdown_links`] turns them into base-less routes or source permalinks;
//! a source link without a repository URL, or any unresolvable link, becomes its text.

use crate::pulse::model::{Block, Linker, MarkdownText, PageId, Site, SourceLoc, SymbolId, Target};
use std::sync::LazyLock;

pub const SYMBOL_SCHEME: &str = "pulse-symbol:";
pub const PAGE_SCHEME: &str = "pulse-page:";
pub const SOURCE_SCHEME: &str = "pulse-source:";

static LINK: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r"\[((?:[^\[\]]|\[[^\]]*\])*)\]\(pulse-(symbol|page|source):([^)\s]+)\)")
        .expect("valid regex")
});

fn resolve(linker: &Linker, scheme: &str, id: &str, anchor: &str) -> Option<String> {
    let target = match scheme {
        "symbol" => Target::Symbol {
            symbol: SymbolId(id.to_string()),
        },
        "page" => Target::Page {
            page: PageId(id.to_string()),
        },
        _ => Target::Source {
            loc: SourceLoc::file(id),
        },
    };
    let r = linker.resolve(&target)?;
    Some(if r.external {
        r.href
    } else {
        format!("{}{anchor}", r.href)
    })
}

fn fix(linker: &Linker, md: &mut MarkdownText) {
    if !md.source.contains("](pulse-") {
        return;
    }
    md.source = LINK
        .replace_all(&md.source, |c: &regex::Captures| {
            // Symbol ids contain `#` (`…::new#method`); pages and sources carry an anchor.
            let (id, anchor) = match &c[2] {
                "symbol" => (&c[3], ""),
                _ => match c[3].find('#') {
                    Some(i) => (&c[3][..i], &c[3][i..]),
                    None => (&c[3], ""),
                },
            };
            match resolve(linker, &c[2], id, anchor) {
                Some(href) => format!("[{}]({href})", &c[1]),
                None => c[1].to_string(),
            }
        })
        .into_owned();
}

fn walk(blocks: &mut [Block], f: &mut dyn FnMut(&mut MarkdownText)) {
    for b in blocks {
        match b {
            Block::Markdown { markdown } => f(markdown),
            Block::Symbol { symbol } => {
                if let Some(d) = symbol.doc.as_mut() {
                    f(d);
                }
            }
            Block::Callout { body, .. } => walk(body, f),
            Block::Narrative { fallback, text, .. } => {
                if let Some(t) = text.as_mut() {
                    f(t);
                }
                walk(fallback, f)
            }
            _ => {}
        }
    }
}

/// Rewrite every `pulse-*:` link on the site.
pub fn resolve_markdown_links(site: &mut Site) {
    let snapshot = site.clone();
    let linker = Linker::new(&snapshot);
    for page in site.pages.values_mut() {
        walk(&mut page.blocks, &mut |md| fix(&linker, md));
    }
}
