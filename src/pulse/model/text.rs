//! Plain-text view of a page: the structural context the writing pass sends to the LLM.
//!
//! Numbers come from the fact store and links become their text, so the model sees
//! exactly what the page states. Diagrams are skipped; they duplicate the lists.

use super::content::{Block, Inline};
use super::{Page, Site};

/// The page as text, capped at `max_chars` (cut at a line boundary).
pub fn page_text(site: &Site, page: &Page, max_chars: usize) -> String {
    let mut out = format!("# {}\n", page.title);
    if let Some(d) = &page.description {
        out.push_str(d);
        out.push('\n');
    }
    blocks(site, &page.blocks, &mut out);
    if out.len() > max_chars {
        let mut cut = max_chars;
        while !out.is_char_boundary(cut) {
            cut -= 1;
        }
        if let Some(nl) = out[..cut].rfind('\n') {
            cut = nl;
        }
        out.truncate(cut);
        out.push_str("\n[… truncated]");
    }
    out
}

fn inlines(site: &Site, v: &[Inline], out: &mut String) {
    for i in v {
        match i {
            Inline::Text { text } => out.push_str(text),
            Inline::Code { code } => {
                out.push('`');
                out.push_str(code);
                out.push('`');
            }
            Inline::Strong { content }
            | Inline::Emph { content }
            | Inline::Link { content, .. } => inlines(site, content, out),
            Inline::Fact { fact } => match site.facts.get(fact) {
                Some(f) => out.push_str(&f.value.display()),
                None => out.push('?'),
            },
        }
    }
}

fn blocks(site: &Site, bs: &[Block], out: &mut String) {
    for b in bs {
        match b {
            Block::Heading { level, text, .. } => {
                out.push('\n');
                out.push_str(&"#".repeat(*level as usize));
                out.push(' ');
                out.push_str(text);
                out.push('\n');
            }
            Block::Paragraph { content } => {
                inlines(site, content, out);
                out.push('\n');
            }
            Block::Markdown { markdown } => {
                out.push_str(&markdown.source);
                out.push('\n');
            }
            Block::Code { code, .. } => {
                out.push_str(code);
                out.push('\n');
            }
            Block::List { items, .. } => {
                for it in items {
                    out.push_str("- ");
                    inlines(site, it, out);
                    out.push('\n');
                }
            }
            Block::Table { columns, rows } => {
                out.push_str(&columns.join(" | "));
                out.push('\n');
                for r in rows {
                    for (i, cell) in r.iter().enumerate() {
                        if i > 0 {
                            out.push_str(" | ");
                        }
                        inlines(site, cell, out);
                    }
                    out.push('\n');
                }
            }
            Block::Callout { body, .. } => blocks(site, body, out),
            Block::Stats { stats } => {
                for s in stats {
                    let v = site
                        .facts
                        .get(&s.fact)
                        .map(|f| f.value.display())
                        .unwrap_or_default();
                    out.push_str(&format!("{}: {v}\n", s.label));
                }
            }
            Block::Cards { .. } | Block::Diagram { .. } => {}
            Block::Narrative { fallback, .. } => blocks(site, fallback, out),
            Block::Symbol { symbol } => {
                out.push_str(&symbol.signature);
                out.push('\n');
                if let Some(d) = &symbol.doc {
                    let first = d.source.split("\n\n").next().unwrap_or("");
                    out.push_str(first);
                    out.push('\n');
                }
            }
        }
    }
}
