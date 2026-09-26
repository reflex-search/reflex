//! Evidence packs for every narrative slot, built from what the index proves.
//!
//! - `module:<path>`: size facts, the module's own doc comment, file doc comments,
//!   capabilities (with the proving import), heaviest dependency edges, and its best
//!   documented items (public and documented first).
//! - `architecture`: site facts, one fact per module (doc summary or top items), the
//!   heaviest module edges, cycles, capabilities.
//! - `project-overview`: README sections, CLI commands, crate docs, capabilities, site
//!   facts.
//!
//! Selection is deterministic: fixed priority order, then name order, then a character
//! budget per pack.

use super::guides::Guide;
use super::modules::ModuleGraph;
use crate::parsers::api::{ApiItem, ApiKind, Visibility};
use crate::pulse::extract::api_cache::ApiIndex;
use crate::pulse::extract::cli::CliCommand;
use crate::pulse::extract::surface::Surface;
use crate::pulse::extract::{Corpus, DocFile};
use crate::pulse::model::evidence::{Evidence, EvidenceKind, EvidencePack};
use crate::pulse::model::facts::{FactValue, Provenance, Subject};
use crate::pulse::model::{Site, SourceLoc};
use crate::pulse::narrate::ids;
use std::collections::{BTreeMap, BTreeSet};

/// Fact text per pack, in characters (~3.5 k tokens).
const MODULE_BUDGET: usize = 12_000;
const SITE_BUDGET: usize = 16_000;

fn metric(id: &str, subject: &str, text: String, names: Vec<String>) -> Evidence {
    Evidence {
        id: id.into(),
        kind: EvidenceKind::Metric,
        subject: subject.into(),
        text,
        names,
        source: None,
        capability: None,
    }
}

fn count(site: &Site, subject: &Subject, key: &str) -> u64 {
    match site
        .facts
        .get(&crate::pulse::model::FactStore::id_for(subject, key))
        .map(|f| &f.value)
    {
        Some(FactValue::Count(n)) => *n,
        _ => 0,
    }
}

/// Capability evidence for one module, from the fact store.
fn capabilities(site: &Site, module: &str) -> Vec<Evidence> {
    let prefix = format!("module:{module}:capability:");
    site.facts
        .iter()
        .filter(|f| f.id.as_str().starts_with(&prefix))
        .map(|f| {
            let cap = f.id.as_str()[prefix.len()..].to_string();
            let import = match &f.value {
                FactValue::Text(t) => t.split('`').nth(1).unwrap_or_default().to_string(),
                _ => String::new(),
            };
            Evidence {
                id: f.id.to_string(),
                kind: EvidenceKind::Capability,
                subject: module.to_string(),
                text: f.value.display(),
                names: vec![
                    import.split("::").next().unwrap_or(&import).to_string(),
                    import,
                ],
                source: f.provenance.iter().find_map(|p| match p {
                    Provenance::Source { loc } => Some(loc.clone()),
                    _ => None,
                }),
                capability: Some(cap),
            }
        })
        .collect()
}

fn doc_summary(item: &ApiItem) -> String {
    item.doc
        .as_ref()
        .map(|d| d.summary.clone())
        .unwrap_or_default()
}

/// Rank items: public types, public functions, documented, then everything else.
fn rank(item: &ApiItem) -> (u8, u8, u8) {
    let public = matches!(item.visibility, Visibility::Public);
    let kind = match item.kind {
        k if k.is_type() => 0,
        ApiKind::Function => 1,
        ApiKind::Const | ApiKind::Static | ApiKind::TypeAlias | ApiKind::Macro => 2,
        _ => 3,
    };
    (u8::from(!public), kind, u8::from(item.doc.is_none()))
}

fn item_evidence(item: &ApiItem, file: &str) -> Evidence {
    let summary = doc_summary(item);
    let name = match &item.self_type {
        Some(t) => format!("{}::{}", t.split('<').next().unwrap_or(t), item.name),
        None => item.name.clone(),
    };
    let text = if summary.is_empty() {
        item.signature.clone()
    } else {
        format!("{} — {summary}", item.signature)
    };
    let mut names = vec![item.name.clone(), name.clone()];
    names.extend(
        item.members
            .iter()
            .filter(|m| m.visibility.is_public())
            .take(12)
            .map(|m| m.name.clone()),
    );
    Evidence {
        id: format!("item:{file}::{name}"),
        kind: EvidenceKind::Item,
        subject: name,
        text,
        names,
        source: Some(SourceLoc::lines(file, item.start_line, item.end_line)),
        capability: None,
    }
}

fn module_pack(
    site: &Site,
    corpus: &Corpus,
    graph: &ModuleGraph,
    apis: &ApiIndex,
    mi: usize,
) -> EvidencePack {
    let m = &graph.modules[mi];
    let path = m.id.as_str();
    let subject = Subject::Module(m.id.clone());
    let mut p = EvidencePack::new(ids::module(path), format!("the module {}", m.name()));
    let langs: Vec<&str> = m.languages.keys().map(String::as_str).collect();
    p.push(metric(
        &format!("module:{path}:size"),
        m.name(),
        format!(
            "{} holds {} source files ({} lines) in {}; it imports from {} modules and is imported by {} modules.",
            m.name(),
            count(site, &subject, "files"),
            crate::pulse::model::facts::group_thousands(count(site, &subject, "lines")),
            langs.join(", "),
            count(site, &subject, "dependencies"),
            count(site, &subject, "dependents"),
        ),
        vec![path.to_string()],
    ));

    // The module's own doc comment (Rust `//!` in mod.rs / lib.rs / <dir>.rs), then
    // file-level docs of its other files.
    let mut file_docs: Vec<(u8, usize)> = m
        .files
        .iter()
        .filter(|&&f| apis.files.get(&f).is_some_and(|a| a.module_doc.is_some()))
        .map(|&f| {
            let name = corpus.files[f].path.rsplit('/').next().unwrap_or("");
            let primary = matches!(
                name,
                "mod.rs"
                    | "lib.rs"
                    | "main.rs"
                    | "__init__.py"
                    | "index.ts"
                    | "index.js"
                    | "doc.go"
            );
            (u8::from(!primary), f)
        })
        .collect();
    file_docs.sort();
    for (_, f) in file_docs.iter().take(6) {
        let file = &corpus.files[*f].path;
        let doc = apis.files[f].module_doc.as_ref().expect("filtered");
        let para = doc.markdown.split("\n\n").next().unwrap_or("").to_string();
        p.push(Evidence {
            id: format!("doc:{file}"),
            kind: EvidenceKind::ModuleDoc,
            subject: file.clone(),
            text: para.split_whitespace().collect::<Vec<_>>().join(" "),
            names: vec![file.clone(), path.to_string()],
            source: Some(SourceLoc::lines(file, doc.start_line, doc.end_line)),
            capability: None,
        });
    }

    for e in capabilities(site, path) {
        p.push(e);
    }

    for (other, n, dir) in graph
        .dependencies(mi)
        .into_iter()
        .take(5)
        .map(|(o, n)| (o, n, "imports from"))
        .chain(
            graph
                .dependents(mi)
                .into_iter()
                .take(5)
                .map(|(o, n)| (o, n, "is imported by")),
        )
    {
        let o = graph.modules[other].name();
        p.push(Evidence {
            id: format!("dep:{path}:{dir}:{o}"),
            kind: EvidenceKind::Dependency,
            subject: m.name().to_string(),
            text: format!("{} {dir} {o} ({n} imports)", m.name()),
            names: vec![path.to_string(), o.to_string()],
            source: None,
            capability: None,
        });
    }

    let mut items: Vec<(&ApiItem, &str)> = Vec::new();
    for &f in &m.files {
        if let Some(api) = apis.files.get(&f) {
            for it in api
                .items
                .iter()
                .filter(|i| !i.test_only && i.kind != ApiKind::Module)
            {
                items.push((it, corpus.files[f].path.as_str()));
            }
        }
    }
    items.sort_by(|a, b| rank(a.0).cmp(&rank(b.0)).then(a.0.name.cmp(&b.0.name)));
    for (it, file) in items.into_iter().take(16) {
        p.push(item_evidence(it, file));
    }
    p.truncate_to(MODULE_BUDGET);
    p
}

fn readme_sections(readme: &DocFile) -> Vec<Evidence> {
    let mut out = Vec::new();
    let mut title = String::from("Introduction");
    let mut buf: Vec<&str> = Vec::new();
    let mut start = 1u32;
    let mut fence = false;
    let flush =
        |title: &str, buf: &mut Vec<&str>, start: u32, end: u32, out: &mut Vec<Evidence>| {
            let text: String = buf
                .iter()
                .filter(|l| !l.trim_start().starts_with("[![") && !l.trim_start().starts_with('<'))
                .copied()
                .collect::<Vec<_>>()
                .join("\n");
            let text = text.trim();
            if !text.is_empty() {
                out.push(Evidence {
                    id: format!("readme:{}", crate::pulse::model::ids::slugify(title)),
                    kind: EvidenceKind::DocSection,
                    subject: format!("README: {title}"),
                    text: text.to_string(),
                    names: Vec::new(),
                    source: Some(SourceLoc::lines(&readme.path, start, end)),
                    capability: None,
                });
            }
            buf.clear();
        };
    let lines: Vec<&str> = readme.content.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        if t.starts_with("```") {
            fence = !fence;
        }
        if !fence && (t.starts_with("# ") || t.starts_with("## ")) {
            flush(&title, &mut buf, start, i as u32, &mut out);
            title = t.trim_start_matches('#').trim().to_string();
            start = i as u32 + 2;
            continue;
        }
        buf.push(line);
    }
    flush(&title, &mut buf, start, lines.len() as u32, &mut out);
    out
}

fn command_evidence(cmd: &CliCommand) -> Evidence {
    let about = cmd
        .about
        .clone()
        .or_else(|| cmd.doc.as_ref().map(|d| d.summary.clone()))
        .unwrap_or_default();
    let flags: Vec<String> = cmd
        .args
        .iter()
        .filter(|a| !a.hidden)
        .filter_map(|a| a.long.as_ref().map(|l| format!("--{l}")))
        .collect();
    let mut names = vec![cmd.path.join(" ")];
    names.extend(flags.iter().cloned());
    Evidence {
        id: format!("cli:{}", cmd.path.join(" ")),
        kind: EvidenceKind::Command,
        subject: format!("command `{}`", cmd.path.join(" ")),
        text: if flags.is_empty() {
            format!("`{}` — {about}", cmd.path.join(" "))
        } else {
            format!(
                "`{}` — {about} Options: {}",
                cmd.path.join(" "),
                flags.join(", ")
            )
        },
        names,
        source: Some(SourceLoc::lines(&cmd.file, cmd.line, cmd.line)),
        capability: None,
    }
}

fn site_metrics(site: &Site) -> Evidence {
    let s = Subject::Site;
    metric(
        "site:size",
        "the project",
        format!(
            "{} source files, {} lines of code, {} modules, {} languages, {} dependency cycles between modules.",
            count(site, &s, "source_files"),
            crate::pulse::model::facts::group_thousands(count(site, &s, "source_lines")),
            count(site, &s, "modules"),
            count(site, &s, "languages"),
            count(site, &s, "module_cycles"),
        ),
        vec![],
    )
}

fn module_line(
    site: &Site,
    corpus: &Corpus,
    graph: &ModuleGraph,
    apis: &ApiIndex,
    mi: usize,
) -> Evidence {
    let pack = module_pack(site, corpus, graph, apis, mi);
    let m = &graph.modules[mi];
    let doc = pack
        .items
        .iter()
        .find(|e| e.kind == EvidenceKind::ModuleDoc)
        .map(|e| e.text.clone());
    let items: Vec<String> = pack
        .items
        .iter()
        .filter(|e| e.kind == EvidenceKind::Item)
        .take(4)
        .map(|e| e.subject.clone())
        .collect();
    let subject = Subject::Module(m.id.clone());
    let text = format!(
        "{}: {}{} ({} files, used by {} modules){}",
        m.name(),
        doc.as_deref().unwrap_or(""),
        if items.is_empty() {
            String::new()
        } else {
            format!(" Key items: {}.", items.join(", "))
        },
        count(site, &subject, "files"),
        count(site, &subject, "dependents"),
        ""
    );
    let mut names = vec![m.id.to_string()];
    names.extend(items);
    Evidence {
        id: format!("module:{}", m.id),
        kind: if doc.is_some() {
            EvidenceKind::ModuleDoc
        } else {
            EvidenceKind::Item
        },
        subject: m.name().to_string(),
        text,
        names,
        source: pack
            .items
            .iter()
            .find(|e| e.kind == EvidenceKind::ModuleDoc)
            .and_then(|e| e.source.clone()),
        capability: None,
    }
}

#[allow(clippy::too_many_arguments)]
/// Build every pack and the known-name list.
pub fn build(
    site: &Site,
    corpus: &Corpus,
    graph: &ModuleGraph,
    apis: &ApiIndex,
    surface: &Surface,
    commands: &[CliCommand],
    guides: &[Guide],
) -> (BTreeMap<String, EvidencePack>, Vec<String>) {
    let mut packs = BTreeMap::new();
    for mi in 0..graph.modules.len() {
        let p = module_pack(site, corpus, graph, apis, mi);
        packs.insert(p.slot.clone(), p);
    }

    // Architecture
    let mut arch = EvidencePack::new(ids::ARCHITECTURE, "the architecture of the project");
    arch.push(site_metrics(site));
    for mi in 0..graph.modules.len() {
        arch.push(module_line(site, corpus, graph, apis, mi));
    }
    let mut edges: Vec<(usize, usize, usize)> =
        graph.edges.iter().map(|(&(a, b), &n)| (a, b, n)).collect();
    edges.sort_by(|x, y| y.2.cmp(&x.2).then((x.0, x.1).cmp(&(y.0, y.1))));
    for (a, b, n) in edges.into_iter().take(25) {
        let (an, bn) = (graph.modules[a].name(), graph.modules[b].name());
        arch.push(Evidence {
            id: format!("edge:{an}->{bn}"),
            kind: EvidenceKind::Dependency,
            subject: an.to_string(),
            text: format!("{an} imports from {bn} ({n} imports)"),
            names: vec![an.to_string(), bn.to_string()],
            source: None,
            capability: None,
        });
    }
    for c in graph.cycles() {
        let names: Vec<String> = c
            .iter()
            .map(|&i| graph.modules[i].name().to_string())
            .collect();
        arch.push(Evidence {
            id: format!("cycle:{}", names.join(",")),
            kind: EvidenceKind::Dependency,
            subject: "dependency cycle".into(),
            text: format!(
                "These modules form a dependency cycle: {}",
                names.join(", ")
            ),
            names,
            source: None,
            capability: None,
        });
    }
    for m in &graph.modules {
        for e in capabilities(site, m.id.as_str()) {
            arch.push(e);
        }
    }
    arch.truncate_to(SITE_BUDGET);
    packs.insert(arch.slot.clone(), arch);

    // Overview
    let mut ov = EvidencePack::new(ids::OVERVIEW, "the project");
    if let Some(r) = &corpus.readme {
        for e in readme_sections(r).into_iter().take(6) {
            ov.push(e);
        }
    }
    ov.push(site_metrics(site));
    for root in commands {
        ov.push(command_evidence(root));
        for c in root.subcommands.iter().filter(|c| !c.hidden) {
            ov.push(command_evidence(c));
        }
    }
    for k in surface.libraries() {
        if let Some(d) = k.modules.first().and_then(|m| m.doc.as_ref()) {
            ov.push(Evidence {
                id: format!("crate:{}", k.name),
                kind: EvidenceKind::ModuleDoc,
                subject: format!("library {}", k.name),
                text: d.markdown.split("\n\n").next().unwrap_or("").to_string(),
                names: vec![k.name.clone()],
                source: Some(SourceLoc::lines(&k.root_file, d.start_line, d.end_line)),
                capability: None,
            });
        }
    }
    let mut seen = BTreeSet::new();
    for m in &graph.modules {
        for e in capabilities(site, m.id.as_str()) {
            if seen.insert(e.capability.clone()) {
                ov.push(e);
            }
        }
    }
    for g in guides
        .iter()
        .filter(|g| g.tab == crate::pulse::model::TabId::Docs)
        .take(4)
    {
        let para = g
            .body
            .split("\n\n")
            .find(|p| !p.trim().is_empty() && !p.trim_start().starts_with('#'))
            .unwrap_or("");
        ov.push(Evidence {
            id: format!("guide:{}", g.path),
            kind: EvidenceKind::DocSection,
            subject: format!("guide: {}", g.title),
            text: para.split_whitespace().collect::<Vec<_>>().join(" "),
            names: vec![g.path.clone()],
            source: Some(SourceLoc::lines(&g.path, g.body_line, g.body_line)),
            capability: None,
        });
    }
    ov.truncate_to(SITE_BUDGET);
    packs.insert(ov.slot.clone(), ov);

    // Known names
    let mut known: BTreeSet<String> = BTreeSet::new();
    for e in site.symbols.values() {
        known.insert(e.name.clone());
        known.insert(e.path.clone());
    }
    for f in &corpus.files {
        known.insert(f.path.clone());
    }
    for m in &graph.modules {
        known.insert(m.id.to_string());
    }
    for imp in &corpus.imports {
        known.insert(imp.path.clone());
    }
    for root in commands {
        for c in root.walk() {
            known.insert(c.path.join(" "));
            for a in &c.args {
                if let Some(l) = &a.long {
                    known.insert(format!("--{l}"));
                }
            }
        }
    }
    for p in packs.values() {
        for e in &p.items {
            known.extend(e.names.iter().cloned());
        }
    }
    for api in apis.files.values() {
        for it in &api.items {
            known.insert(it.name.clone());
            for m in &it.members {
                known.insert(m.name.clone());
            }
        }
    }
    (packs, known.into_iter().collect())
}
