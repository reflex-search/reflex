//! The grounding gate: deterministic checks on every LLM sentence.
//!
//! A sentence survives only if:
//! 1. it cites at least one fact of its pack;
//! 2. every code-shaped mention (backticks, `snake_case`, `CamelCase`, paths, flags)
//!    names something the index knows, and a *cited* fact names it too;
//! 3. every number of two or more digits appears in a cited fact;
//! 4. every capability phrase ("HTTP server", "terminal UI", "database", …) is backed
//!    by a cited capability fact, or a cited document that says it.
//!
//! A paragraph that loses more than half its sentences is dropped; a section that
//! keeps less than [`GateConfig::min_kept_ratio`] of its sentences, or fewer than
//! `min_sentences`, is rejected and renders its structural fallback. The gate runs on
//! every render, cache hits included, so a renamed symbol cannot leave stale prose.

use super::contract::Contract;
use crate::pulse::model::evidence::{EvidenceKind, EvidencePack};
use serde::Serialize;
use std::collections::HashSet;
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy)]
pub struct GateConfig {
    pub min_kept_ratio: f32,
    pub min_sentences: usize,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            min_kept_ratio: 0.6,
            min_sentences: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "reason", content = "detail", rename_all = "kebab-case")]
pub enum DropReason {
    Uncited,
    /// A code mention the index does not know: invented.
    UnresolvedIdentifier(String),
    /// A real name the cited facts do not mention: unsupported.
    UngroundedIdentifier(String),
    NumberMismatch(String),
    UnsupportedClaim(String),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct KeptSentence {
    pub text: String,
    /// Indices into the pack.
    pub cites: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Confidence {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct GateReport {
    pub slot: String,
    pub total: usize,
    pub kept: usize,
    pub dropped: Vec<(String, DropReason)>,
    /// Why the whole section fell back, if it did.
    pub rejected: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Verified {
    pub paragraphs: Vec<Vec<KeptSentence>>,
    pub confidence: Confidence,
    pub report: GateReport,
}

/// Every name the index knows: symbol names and paths, module paths, file paths,
/// commands. A code mention outside this set was invented.
#[derive(Debug, Clone, Default)]
pub struct KnownNames {
    names: HashSet<String>,
}

impl KnownNames {
    pub fn insert(&mut self, name: &str) {
        let n = name.trim();
        if n.is_empty() {
            return;
        }
        self.names.insert(n.to_string());
        // `a::b::C` is also known as `C` and `b::C`; `src/x/y.rs` also as `y.rs`.
        for sep in ["::", ".", "/"] {
            let parts: Vec<&str> = n.split(sep).collect();
            for i in 1..parts.len() {
                self.names.insert(parts[i..].join(sep));
            }
        }
    }

    pub fn contains(&self, m: &str) -> bool {
        let m = m.trim_end_matches("()").trim_end_matches('!');
        self.names.contains(m)
            || m.rsplit("::")
                .next()
                .is_some_and(|last| self.names.contains(last))
    }
}

/// Words that look like code but are ordinary names.
const PROSE_WORDS: &[&str] = &[
    "TypeScript",
    "JavaScript",
    "GitHub",
    "GitLab",
    "PostgreSQL",
    "MySQL",
    "SQLite",
    "WebSocket",
    "OpenAI",
    "OpenRouter",
    "LLMs",
    "APIs",
    "JSON",
    "YAML",
    "TOML",
    "HTTP",
    "HTTPS",
    "MCP",
    "CLI",
    "TUI",
    "AST",
    "IDE",
    "macOS",
    "iOS",
    "NodeJS",
    "README",
    "CHANGELOG",
    "e.g",
    "i.e",
    "etc",
];

static BACKTICK: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"`([^`]+)`").expect("valid regex"));
static CODEISH: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(
        r"(?x)
        (--[a-z][a-z0-9-]+)                              # --flag
      | ([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+) # a::b
      | ([a-z0-9_]+_[a-z0-9_]+)                          # snake_case
      | ([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+)              # CamelCase
      | ((?:[\w.-]+/)+[\w.-]+\.[a-z]{1,5})               # path/to/file.ext
    ",
    )
    .expect("valid regex")
});
static NUMBER: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"\b\d[\d,]*(?:\.\d+)?\b").expect("valid regex"));

/// (phrase pattern, capability id) for claims that need evidence.
static CLAIMS: LazyLock<Vec<(regex::Regex, &'static str)>> = LazyLock::new(|| {
    [
        (r"(?i)\b(http|web|rest|api)\s+server\b|\bhttp api\b|\bweb (app|application|service)\b|\bendpoints?\b", "http-server"),
        (r"(?i)\bterminal (ui|user interface)\b|\btui\b|\binteractive terminal\b", "tui"),
        (r"(?i)\bdatabases?\b|\bsqlite\b|\bsql\b", "database"),
        (r"(?i)\bllms?\b|\blanguage models?\b|\bai providers?\b", "llm"),
        (r"(?i)\bfile[- ]watch(er|ing)\b|\bwatches (files|the file ?system)\b", "file-watching"),
        (r"(?i)\bweb (ui|user interface|frontend)\b", "web-ui"),
    ]
    .into_iter()
    .map(|(p, c)| (regex::Regex::new(p).expect("valid regex"), c))
    .collect()
});

fn mentions(text: &str) -> Vec<String> {
    let mut out: Vec<String> = BACKTICK
        .captures_iter(text)
        .map(|c| c[1].trim().to_string())
        .collect();
    let outside = BACKTICK.replace_all(text, " ");
    for c in CODEISH.captures_iter(&outside) {
        let m = c.get(0).unwrap().as_str().trim_end_matches('.');
        if !PROSE_WORDS.contains(&m) {
            out.push(m.to_string());
        }
    }
    out.retain(|m| !m.is_empty() && !PROSE_WORDS.contains(&m.as_str()));
    out
}

fn norm_number(s: &str) -> String {
    s.replace(',', "")
}

fn fact_text(e: &crate::pulse::model::evidence::Evidence) -> String {
    format!("{}\n{}\n{}", e.subject, e.text, e.names.join("\n"))
}

/// A code mention that a cited fact does not name but another fact of the pack does
/// is grounded by that fact: add it to the cites (the sentence then shows it as a
/// source) instead of dropping the sentence. Returns the extra cites.
fn extra_cites(text: &str, cites: &[usize], pack: &EvidencePack) -> Vec<usize> {
    let cited_text: String = cites
        .iter()
        .filter_map(|&i| pack.items.get(i))
        .map(fact_text)
        .collect::<Vec<_>>()
        .join("\n");
    let mut extra = Vec::new();
    for m in mentions(text) {
        if cited_text.contains(&m) {
            continue;
        }
        if let Some(i) = pack
            .items
            .iter()
            .position(|e| e.names.iter().any(|n| n == &m) || e.subject == m)
            && !cites.contains(&i)
            && !extra.contains(&i)
        {
            extra.push(i);
        }
    }
    extra
}

/// Check one sentence against its cited facts.
fn check(
    text: &str,
    cites: &[usize],
    pack: &EvidencePack,
    known: &KnownNames,
) -> Option<DropReason> {
    if cites.is_empty() {
        return Some(DropReason::Uncited);
    }
    let cited: Vec<_> = cites.iter().filter_map(|&i| pack.items.get(i)).collect();
    let cited_text: String = cited
        .iter()
        .map(|e| format!("{}\n{}\n{}", e.subject, e.text, e.names.join("\n")))
        .collect::<Vec<_>>()
        .join("\n");

    for m in mentions(text) {
        // A mention with spaces is prose in backticks (a command line); check its first word.
        let head = m.split_whitespace().next().unwrap_or(&m).to_string();
        if !known.contains(&m) && !known.contains(&head) && !cited_text.contains(&m) {
            return Some(DropReason::UnresolvedIdentifier(m));
        }
        let grounded = cited_text.contains(&m)
            || cited_text.contains(&head)
            || m.rsplit("::")
                .next()
                .is_some_and(|last| cited_text.contains(last))
            || m.rsplit('/')
                .next()
                .is_some_and(|last| cited_text.contains(last));
        if !grounded {
            return Some(DropReason::UngroundedIdentifier(m));
        }
    }

    let cited_numbers: HashSet<String> = NUMBER
        .find_iter(&cited_text)
        .map(|n| norm_number(n.as_str()))
        .collect();
    for n in NUMBER.find_iter(&BACKTICK.replace_all(text, " ")) {
        let v = norm_number(n.as_str());
        if v.len() >= 2 && !cited_numbers.contains(&v) {
            return Some(DropReason::NumberMismatch(n.as_str().to_string()));
        }
    }

    for (re, cap) in CLAIMS.iter() {
        if let Some(m) = re.find(text) {
            let phrase = m.as_str().to_ascii_lowercase();
            let backed = cited.iter().any(|e| {
                e.capability.as_deref() == Some(*cap)
                    || (matches!(
                        e.kind,
                        EvidenceKind::DocSection
                            | EvidenceKind::ModuleDoc
                            | EvidenceKind::Item
                            | EvidenceKind::Command
                    ) && e.text.to_ascii_lowercase().contains(&phrase))
            });
            if !backed {
                return Some(DropReason::UnsupportedClaim(m.as_str().to_string()));
            }
        }
    }
    None
}

/// Split model sentences that hold several real sentences; each piece keeps the cites.
fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_tick = false;
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        cur.push(c);
        if c == '`' {
            in_tick = !in_tick;
        }
        let end = matches!(c, '.' | '!' | '?')
            && !in_tick
            && chars.get(i + 1) == Some(&' ')
            && chars.get(i + 2).is_some_and(|n| n.is_uppercase());
        if end {
            out.push(cur.trim().to_string());
            cur.clear();
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Verify a parsed contract. `Err` means the section falls back to structure.
pub fn verify(
    contract: &Contract,
    pack: &EvidencePack,
    known: &KnownNames,
    cfg: &GateConfig,
) -> Result<Verified, GateReport> {
    let mut report = GateReport {
        slot: pack.slot.clone(),
        ..GateReport::default()
    };
    if contract.status != "ok" {
        report.rejected = Some(format!(
            "insufficient evidence: {}",
            contract.missing.join("; ")
        ));
        return Err(report);
    }
    let mut paragraphs = Vec::new();
    for p in &contract.paragraphs {
        let mut kept = Vec::new();
        let mut total = 0;
        for s in &p.sentences {
            let cites: Vec<usize> = s
                .cite
                .iter()
                .filter_map(|h| {
                    let n: usize = h.trim().trim_start_matches('F').parse().ok()?;
                    (n >= 1 && n <= pack.items.len()).then_some(n - 1)
                })
                .take(3)
                .collect();
            for piece in split_sentences(&s.text) {
                total += 1;
                report.total += 1;
                let mut cites = cites.clone();
                if !cites.is_empty() {
                    cites.extend(extra_cites(&piece, &cites, pack));
                }
                match check(&piece, &cites, pack, known) {
                    None => kept.push(KeptSentence { text: piece, cites }),
                    Some(r) => report.dropped.push((piece, r)),
                }
            }
        }
        if total > 0 && kept.len() * 2 >= total {
            report.kept += kept.len();
            paragraphs.push(kept);
        } else {
            // Drop the whole paragraph; its survivors read badly alone.
            for k in kept {
                report.dropped.push((k.text, DropReason::Uncited));
            }
        }
    }
    let ratio = if report.total == 0 {
        0.0
    } else {
        report.kept as f32 / report.total as f32
    };
    if report.kept < cfg.min_sentences || ratio < cfg.min_kept_ratio {
        report.rejected = Some(format!(
            "kept {} of {} sentences (need {} and {:.0}%)",
            report.kept,
            report.total,
            cfg.min_sentences,
            cfg.min_kept_ratio * 100.0
        ));
        return Err(report);
    }
    let g = pack.groundability();
    let confidence = if ratio >= 0.9 && g >= 0.5 {
        Confidence::High
    } else if ratio < 0.75 || g < 0.3 {
        Confidence::Low
    } else {
        Confidence::Medium
    };
    Ok(Verified {
        paragraphs,
        confidence,
        report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pulse::model::evidence::Evidence;
    use crate::pulse::write::contract::{Paragraph, Sentence};

    fn ev(kind: EvidenceKind, text: &str, names: &[&str], cap: Option<&str>) -> Evidence {
        Evidence {
            id: text.into(),
            kind,
            subject: "src/interactive".into(),
            text: text.into(),
            names: names.iter().map(|s| s.to_string()).collect(),
            source: None,
            capability: cap.map(str::to_string),
        }
    }

    fn pack() -> EvidencePack {
        let mut p = EvidencePack::new("module:src/interactive", "the module src/interactive");
        p.push(ev(EvidenceKind::Metric, "files=12 lines=4,210", &[], None));
        p.push(ev(
            EvidenceKind::Capability,
            "terminal user interface (imports `ratatui` in src/interactive/app.rs:5)",
            &["ratatui"],
            Some("tui"),
        ));
        p.push(ev(
            EvidenceKind::Item,
            "pub struct InteractiveApp — Interactive search session state",
            &["InteractiveApp"],
            None,
        ));
        p
    }

    fn known() -> KnownNames {
        let mut k = KnownNames::default();
        for n in [
            "reflex::interactive::InteractiveApp",
            "src/interactive",
            "src/interactive/app.rs",
            "ratatui",
            "reflex::cache::CacheManager",
        ] {
            k.insert(n);
        }
        k
    }

    fn contract(sents: &[(&str, &[&str])]) -> Contract {
        Contract {
            status: "ok".into(),
            missing: vec![],
            paragraphs: vec![Paragraph {
                sentences: sents
                    .iter()
                    .map(|(t, c)| Sentence {
                        text: t.to_string(),
                        cite: c.iter().map(|s| s.to_string()).collect(),
                    })
                    .collect(),
            }],
        }
    }

    fn reasons(c: &Contract) -> Vec<DropReason> {
        match verify(
            c,
            &pack(),
            &known(),
            &GateConfig {
                min_kept_ratio: 0.0,
                min_sentences: 0,
            },
        ) {
            Ok(v) => v.report.dropped.into_iter().map(|(_, r)| r).collect(),
            Err(r) => r.dropped.into_iter().map(|(_, r)| r).collect(),
        }
    }

    #[test]
    fn grounded_sentences_pass() {
        let c = contract(&[
            (
                "The module renders an interactive terminal UI with `ratatui`.",
                &["F2"],
            ),
            (
                "`InteractiveApp` holds the session state across 12 files.",
                &["F3", "F1"],
            ),
        ]);
        let v = verify(&c, &pack(), &known(), &GateConfig::default()).unwrap();
        assert_eq!(v.report.kept, 2, "{:?}", v.report.dropped);
    }

    #[test]
    fn the_http_server_regression_is_blocked() {
        // The old Pulse said "src/interactive provides an HTTP server": no fact backs it.
        let c = contract(&[("This module provides an HTTP server for search.", &["F3"])]);
        assert_eq!(
            reasons(&c),
            vec![DropReason::UnsupportedClaim("HTTP server".into())]
        );
    }

    #[test]
    fn each_drop_reason() {
        assert_eq!(
            reasons(&contract(&[("It renders things.", &[])])),
            vec![DropReason::Uncited]
        );
        assert_eq!(
            reasons(&contract(&[("It uses `FooBarBaz` internally.", &["F3"])])),
            vec![DropReason::UnresolvedIdentifier("FooBarBaz".into())]
        );
        assert_eq!(
            reasons(&contract(&[(
                "It wraps `CacheManager` for storage.",
                &["F3"]
            )])),
            vec![DropReason::UngroundedIdentifier("CacheManager".into())]
        );
        assert_eq!(
            reasons(&contract(&[("It spans 40 files.", &["F1"])])),
            vec![DropReason::NumberMismatch("40".into())]
        );
        assert_eq!(
            reasons(&contract(&[("It has 4210 lines.", &["F1"])])).len(),
            0,
            "4,210 and 4210 are the same number"
        );
    }

    #[test]
    fn a_name_from_another_fact_of_the_pack_is_cited_automatically() {
        // The sentence cites only the metric, but `InteractiveApp` is named by F3.
        let c = contract(&[(
            "`InteractiveApp` holds the session state across 12 files.",
            &["F1"],
        )]);
        let v = verify(
            &c,
            &pack(),
            &known(),
            &GateConfig {
                min_kept_ratio: 0.0,
                min_sentences: 0,
            },
        )
        .unwrap();
        assert_eq!(v.report.kept, 1, "{:?}", v.report.dropped);
        assert_eq!(v.paragraphs[0][0].cites, vec![0, 2]);
    }

    #[test]
    fn bad_sections_are_rejected() {
        let c = contract(&[
            ("It uses `Nope`.", &["F3"]),
            ("It spans 99 files.", &["F1"]),
        ]);
        let r = verify(&c, &pack(), &known(), &GateConfig::default()).unwrap_err();
        assert!(r.rejected.is_some());
        let insufficient = Contract {
            status: "insufficient_evidence".into(),
            missing: vec!["what the module does".into()],
            paragraphs: vec![],
        };
        let r = verify(&insufficient, &pack(), &known(), &GateConfig::default()).unwrap_err();
        assert!(r.rejected.unwrap().contains("what the module does"));
    }

    #[test]
    fn run_on_sentences_are_split() {
        assert_eq!(
            split_sentences("It parses code. It caches symbols in `a.b`. Done"),
            vec!["It parses code.", "It caches symbols in `a.b`.", "Done"]
        );
    }

    #[test]
    fn mentions_find_code_shapes() {
        let m = mentions(
            "Run `rfx index` with --force; see src/cache.rs and the QueryEngine or query_engine, not TypeScript.",
        );
        assert!(m.contains(&"rfx index".to_string()));
        assert!(m.contains(&"--force".to_string()));
        assert!(m.contains(&"src/cache.rs".to_string()));
        assert!(m.contains(&"QueryEngine".to_string()));
        assert!(m.contains(&"query_engine".to_string()));
        assert!(!m.contains(&"TypeScript".to_string()));
    }
}
