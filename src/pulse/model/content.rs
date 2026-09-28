//! Page content: a small block/inline AST.
//!
//! There are no URLs here. Links point at a [`Target`] and the renderer resolves them
//! through the [`Linker`](super::xref::Linker), so a base-url change or a slug change
//! can never leave a stale link. Numbers are [`Inline::Fact`] references into the
//! site's [`FactStore`](super::facts::FactStore), so every page shows the same count.

use super::facts::FactId;
use super::ids::AnchorId;
use super::xref::{SourceLoc, Target};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Block {
    Heading {
        level: u8,
        text: String,
        anchor: AnchorId,
    },
    Paragraph {
        content: Vec<Inline>,
    },
    /// Markdown written by people (README, doc comments, CHANGELOG) or by the LLM.
    /// The renderer treats it as untrusted: raw HTML is shown as text.
    Markdown {
        markdown: MarkdownText,
    },
    Code {
        lang: String,
        code: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    List {
        ordered: bool,
        items: Vec<Vec<Inline>>,
    },
    Table {
        columns: Vec<String>,
        rows: Vec<Vec<Vec<Inline>>>,
    },
    Callout {
        kind: CalloutKind,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        body: Vec<Block>,
    },
    /// A graph. `links` maps node ids in `source` to link targets; the renderer binds
    /// clicks itself, so the diagram source never carries URLs.
    Diagram {
        kind: DiagramKind,
        source: String,
        links: Vec<(String, Target)>,
        #[serde(skip_serializing_if = "Option::is_none")]
        caption: Option<String>,
    },
    Cards {
        cards: Vec<Card>,
    },
    /// A row of headline numbers.
    Stats {
        stats: Vec<Stat>,
    },
    /// One documented item: signature, docs, parameters, source.
    Symbol {
        symbol: Box<SymbolBlock>,
    },
    /// A slot the writing pass may fill. `fallback` is always complete on its own, so a
    /// site built without an LLM has no holes.
    Narrative {
        slot: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        text: Option<MarkdownText>,
        fallback: Vec<Block>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Inline {
    Text {
        text: String,
    },
    Code {
        code: String,
    },
    Strong {
        content: Vec<Inline>,
    },
    Emph {
        content: Vec<Inline>,
    },
    Link {
        to: Target,
        content: Vec<Inline>,
    },
    /// A number from the fact store, rendered with its provenance.
    Fact {
        fact: FactId,
    },
}

impl Inline {
    pub fn text(s: impl Into<String>) -> Self {
        Inline::Text { text: s.into() }
    }

    pub fn code(s: impl Into<String>) -> Self {
        Inline::Code { code: s.into() }
    }

    pub fn strong(s: impl Into<String>) -> Self {
        Inline::Strong {
            content: vec![Inline::text(s)],
        }
    }

    pub fn link(to: Target, label: impl Into<String>) -> Self {
        Inline::Link {
            to,
            content: vec![Inline::text(label)],
        }
    }

    pub fn code_link(to: Target, code: impl Into<String>) -> Self {
        Inline::Link {
            to,
            content: vec![Inline::code(code)],
        }
    }

    pub fn fact(id: FactId) -> Self {
        Inline::Fact { fact: id }
    }
}

impl Block {
    pub fn heading(level: u8, text: impl Into<String>) -> Self {
        let text = text.into();
        Block::Heading {
            level,
            anchor: super::ids::anchor_for(&text),
            text,
        }
    }

    pub fn para(content: Vec<Inline>) -> Self {
        Block::Paragraph { content }
    }

    pub fn text(s: impl Into<String>) -> Self {
        Block::Paragraph {
            content: vec![Inline::text(s)],
        }
    }

    pub fn note(body: impl Into<String>) -> Self {
        Block::Callout {
            kind: CalloutKind::Note,
            title: None,
            body: vec![Block::text(body)],
        }
    }
}

/// Markdown source plus where it came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarkdownText {
    pub source: String,
    pub origin: MarkdownOrigin,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<SourceLoc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MarkdownOrigin {
    /// A repository document (README, docs/, CHANGELOG).
    Doc,
    /// A doc comment.
    Comment,
    /// A commit message.
    Commit,
    /// Written by the LLM writing pass.
    Llm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CalloutKind {
    Note,
    Tip,
    Caution,
    Danger,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DiagramKind {
    Mermaid,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Card {
    pub title: String,
    pub to: Target,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stat {
    pub label: String,
    pub fact: FactId,
}

/// A documented item as the reference shows it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymbolBlock {
    pub id: super::ids::SymbolId,
    pub anchor: AnchorId,
    /// `fn`, `struct`, `method`, …
    pub kind: String,
    pub name: String,
    /// Code language of the signature (`rust`).
    pub lang: String,
    pub signature: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc: Option<MarkdownText>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<ParamRow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub returns: Option<String>,
    /// Deprecation message ("Deprecated since 1.2: use `sum`").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deprecated: Option<String>,
    /// `async`, `unsafe`, `const`, `impl Display`…
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub badges: Vec<String>,
    pub source: SourceLoc,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParamRow {
    pub name: String,
    pub ty: String,
}
