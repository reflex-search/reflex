//! Evidence packs: the only facts a narrative slot may be written from.
//!
//! Each pack is an ordered list of [`Evidence`] items with stable ids. The prompt shows
//! them as `[F1] …`, the model must cite handles for every sentence, and the gate
//! (`pulse::write::gate`) checks every identifier, number and capability claim against the
//! cited items. Line numbers stay out of the prompt text (so an edit that only moves
//! lines keeps the cache warm) and come back through `source` when citations render.

use crate::pulse::model::SourceLoc;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EvidenceKind {
    /// A number from the fact store (files, lines, dependents).
    Metric,
    /// A module's own documentation.
    ModuleDoc,
    /// A documented item: signature and doc summary.
    Item,
    /// A dependency edge between modules.
    Dependency,
    /// A capability proved by an import.
    Capability,
    /// A README or guide section.
    DocSection,
    /// A CLI command and its help.
    Command,
}

impl EvidenceKind {
    pub fn label(&self) -> &'static str {
        match self {
            EvidenceKind::Metric => "metric",
            EvidenceKind::ModuleDoc => "module-doc",
            EvidenceKind::Item => "item",
            EvidenceKind::Dependency => "dependency",
            EvidenceKind::Capability => "capability",
            EvidenceKind::DocSection => "doc",
            EvidenceKind::Command => "command",
        }
    }

    /// Descriptive evidence says what something does; metrics and edges only say
    /// how big or how connected it is.
    pub fn is_descriptive(&self) -> bool {
        matches!(
            self,
            EvidenceKind::ModuleDoc
                | EvidenceKind::DocSection
                | EvidenceKind::Command
                | EvidenceKind::Capability
        ) || *self == EvidenceKind::Item
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    /// Stable, semantic id (`module:src/pulse:files`, `item:src/pulse/site.rs::generate_site`).
    pub id: String,
    pub kind: EvidenceKind,
    /// What the fact is about, shown to the model (`src/pulse`, `generate_site`).
    pub subject: String,
    /// The fact itself, as the model sees it.
    pub text: String,
    /// Identifiers this fact names (symbols, paths, commands); the gate accepts a code
    /// mention only if a cited fact names it.
    pub names: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceLoc>,
    /// Capability id for [`EvidenceKind::Capability`] facts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability: Option<String>,
}

/// Evidence for one narrative slot.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EvidencePack {
    pub slot: String,
    /// What the section is about (`the module src/pulse`, `the project`).
    pub subject: String,
    pub items: Vec<Evidence>,
}

/// Longest text of one fact in the prompt.
const MAX_FACT_CHARS: usize = 480;

impl EvidencePack {
    pub fn new(slot: impl Into<String>, subject: impl Into<String>) -> Self {
        Self {
            slot: slot.into(),
            subject: subject.into(),
            items: Vec::new(),
        }
    }

    pub fn push(&mut self, mut e: Evidence) {
        if self.items.iter().any(|x| x.id == e.id) {
            return;
        }
        if e.text.chars().count() > MAX_FACT_CHARS {
            e.text = e.text.chars().take(MAX_FACT_CHARS).collect::<String>() + "…";
        }
        self.items.push(e);
    }

    /// Keep at most `max_chars` of fact text: facts are already in priority order.
    pub fn truncate_to(&mut self, max_chars: usize) {
        let mut used = 0;
        self.items.retain(|e| {
            used += e.text.len() + e.subject.len() + 16;
            used <= max_chars
        });
    }

    /// `F1`-style handle of the item at `i`.
    pub fn handle(i: usize) -> String {
        format!("F{}", i + 1)
    }

    pub fn by_handle(&self, h: &str) -> Option<&Evidence> {
        let n: usize = h.trim().trim_start_matches('F').parse().ok()?;
        self.items.get(n.checked_sub(1)?)
    }

    /// The facts block of the prompt.
    pub fn render(&self) -> String {
        let mut out = String::from("<facts>\n");
        for (i, e) in self.items.iter().enumerate() {
            out.push_str(&format!(
                "[{}] {} · {}\n{}\n",
                Self::handle(i),
                e.kind.label(),
                e.subject,
                e.text
            ));
        }
        out.push_str("</facts>");
        out
    }

    /// How much of the pack describes behaviour rather than size (0..=1). Packs with
    /// nothing descriptive produce generic prose, so they are not sent.
    pub fn groundability(&self) -> f32 {
        if self.items.is_empty() {
            return 0.0;
        }
        let descriptive = self
            .items
            .iter()
            .filter(|e| e.kind.is_descriptive())
            .count();
        (descriptive as f32 / (self.items.len().min(10) as f32)).min(1.0)
    }

    pub fn descriptive_count(&self) -> usize {
        self.items
            .iter()
            .filter(|e| e.kind.is_descriptive())
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(id: &str, kind: EvidenceKind, text: &str) -> Evidence {
        Evidence {
            id: id.into(),
            kind,
            subject: "s".into(),
            text: text.into(),
            names: vec![],
            source: None,
            capability: None,
        }
    }

    #[test]
    fn handles_render_and_lookup() {
        let mut p = EvidencePack::new("module:src", "the module src");
        p.push(ev("a", EvidenceKind::Metric, "files=3"));
        p.push(ev("b", EvidenceKind::ModuleDoc, "Parses things."));
        p.push(ev("a", EvidenceKind::Metric, "dup"));
        assert_eq!(p.items.len(), 2);
        let r = p.render();
        assert!(r.contains("[F1] metric · s\nfiles=3"));
        assert!(r.contains("[F2] module-doc · s\nParses things."));
        assert_eq!(p.by_handle("F2").unwrap().id, "b");
        assert!(p.by_handle("F9").is_none());
        assert!(p.by_handle("F0").is_none());
    }

    #[test]
    fn groundability_counts_descriptive_facts() {
        let mut p = EvidencePack::new("x", "x");
        assert_eq!(p.groundability(), 0.0);
        p.push(ev("a", EvidenceKind::Metric, "1"));
        p.push(ev("b", EvidenceKind::Dependency, "2"));
        assert_eq!(p.groundability(), 0.0);
        p.push(ev("c", EvidenceKind::Item, "does"));
        assert!((p.groundability() - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn long_facts_are_cut() {
        let mut p = EvidencePack::new("x", "x");
        p.push(ev("a", EvidenceKind::DocSection, &"x".repeat(2000)));
        assert!(p.items[0].text.chars().count() <= MAX_FACT_CHARS + 1);
    }
}
