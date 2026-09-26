//! The Internals tab: how the code is put together, for people who work on it.
//!
//! - Architecture (landing): module graph, module table, narrative slot `architecture`.
//! - Dependency map: most-imported files, cycles, isolated modules.
//! - One page per module: files, what it imports and what imports it, narrative slot
//!   `module:<path>`.

use super::modules::ModuleGraph;
use super::{PageSpec, SiteBuilder, module_page_id, module_slug};
use crate::parsers::api::{ApiItem, ApiKind, Visibility};
use crate::pulse::extract::api_cache::ApiIndex;
use crate::pulse::extract::{Corpus, language_name};
use crate::pulse::model::content::{CalloutKind, DiagramKind};
use crate::pulse::model::{
    Block, Card, Inline, NavNode, PageId, PageKind, SourceLoc, Stat, Subject, TabId, Target,
};

pub const ARCHITECTURE: &str = "int/architecture";
pub const DEPENDENCY_MAP: &str = "int/dependency-map";

/// Edges drawn on the architecture diagram, heaviest first.
const MAX_DIAGRAM_EDGES: usize = 80;
/// Neighbours drawn on a module diagram, per direction.
const MAX_NEIGHBOURS: usize = 8;
/// Rows in a module's file table.
const MAX_FILE_ROWS: usize = 100;

pub fn build(b: &mut SiteBuilder, corpus: &Corpus, graph: &ModuleGraph, apis: &ApiIndex) {
    let cycles = graph.cycles();
    let in_cycle = |mi: usize| cycles.iter().find(|c| c.contains(&mi));

    for mi in 0..graph.modules.len() {
        let mut blocks = module_blocks(b, corpus, graph, mi, in_cycle(mi));
        blocks.extend(item_blocks(corpus, &graph.modules[mi].files, apis));
        let m = &graph.modules[mi];
        let langs: Vec<&str> = m.languages.keys().map(String::as_str).collect();
        b.add_page(PageSpec {
            id: module_page_id(&m.id),
            tab: TabId::Internals,
            kind: PageKind::Module {
                module: m.id.clone(),
            },
            title: m.name().to_string(),
            description: Some(format!(
                "{} source files, {} lines{}",
                m.files.len(),
                crate::pulse::model::facts::group_thousands(m.lines),
                if langs.is_empty() {
                    String::new()
                } else {
                    format!(" · {}", langs.join(", "))
                }
            )),
            slug: Some(module_slug(&m.id)),
            badges: vec![format!("tier {}", m.tier)],
            blocks,
        });
    }

    let arch = architecture_blocks(b, graph);
    let landing = b.add_page(PageSpec {
        id: PageId::new(ARCHITECTURE),
        tab: TabId::Internals,
        kind: PageKind::Architecture,
        title: "Architecture".into(),
        description: Some("Modules and how they depend on each other.".into()),
        slug: None,
        badges: vec![],
        blocks: arch,
    });

    let depmap = dependency_map_blocks(corpus, graph, &cycles);
    let depmap_id = b.add_page(PageSpec {
        id: PageId::new(DEPENDENCY_MAP),
        tab: TabId::Internals,
        kind: PageKind::DependencyMap,
        title: "Dependency map".into(),
        description: Some("Most-imported files, dependency cycles and isolated modules.".into()),
        slug: Some("dependency-map".into()),
        badges: vec![],
        blocks: depmap,
    });

    let mut module_nav = Vec::new();
    for (mi, m) in graph.modules.iter().enumerate() {
        if m.tier != 1 {
            continue;
        }
        let children = graph.children(mi);
        let own = NavNode::page(&module_page_id(&m.id));
        if children.is_empty() {
            module_nav.push(own);
        } else {
            let mut items = vec![own];
            items.extend(
                children
                    .iter()
                    .map(|&c| NavNode::page(&module_page_id(&graph.modules[c].id))),
            );
            module_nav.push(NavNode::Group {
                label: m.name().to_string(),
                collapsed: true,
                children: items,
            });
        }
    }
    let nav = vec![
        NavNode::page(&depmap_id),
        NavNode::Group {
            label: "Modules".into(),
            collapsed: false,
            children: module_nav,
        },
    ];
    b.add_tab(TabId::Internals, "Internals", landing, nav);
}

fn module_link(graph: &ModuleGraph, mi: usize) -> Inline {
    let m = &graph.modules[mi];
    Inline::code_link(Target::page(module_page_id(&m.id)), m.name())
}

fn file_link(path: &str) -> Inline {
    Inline::code_link(
        Target::Source {
            loc: SourceLoc::file(path),
        },
        path,
    )
}

fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

fn module_blocks(
    b: &SiteBuilder,
    corpus: &Corpus,
    graph: &ModuleGraph,
    mi: usize,
    cycle: Option<&Vec<usize>>,
) -> Vec<Block> {
    let m = &graph.modules[mi];
    let subject = Subject::Module(m.id.clone());
    let deps = graph.dependencies(mi);
    let users = graph.dependents(mi);
    let children = graph.children(mi);
    let mut blocks = Vec::new();

    let mut fallback = vec![
        Inline::code(m.name()),
        Inline::text(" holds "),
        b.fact_inline(&subject, "files"),
        Inline::text(" source files ("),
        b.fact_inline(&subject, "lines"),
        Inline::text(" lines)"),
    ];
    if !m.languages.is_empty() {
        let langs: Vec<&str> = m.languages.keys().map(String::as_str).collect();
        fallback.push(Inline::text(format!(" in {}", langs.join(", "))));
    }
    fallback.push(Inline::text(format!(
        ". It imports from {} and is imported by {}.",
        plural(deps.len(), "module", "modules"),
        plural(users.len(), "module", "modules")
    )));
    blocks.push(Block::Narrative {
        slot: crate::pulse::narrate::ids::module(m.id.as_str()),
        text: None,
        fallback: vec![Block::para(fallback)],
    });

    blocks.push(Block::Stats {
        stats: vec![
            Stat {
                label: "Source files".into(),
                fact: b.fact(&subject, "files"),
            },
            Stat {
                label: "Lines".into(),
                fact: b.fact(&subject, "lines"),
            },
            Stat {
                label: "Depends on".into(),
                fact: b.fact(&subject, "dependencies"),
            },
            Stat {
                label: "Used by".into(),
                fact: b.fact(&subject, "dependents"),
            },
        ],
    });

    if let Some(cycle) = cycle {
        let mut body = vec![Inline::text("This module is part of a dependency cycle: ")];
        for (i, &c) in cycle.iter().enumerate() {
            if i > 0 {
                body.push(Inline::text(", "));
            }
            body.push(module_link(graph, c));
        }
        body.push(Inline::text(
            ". Each of these modules imports, directly or indirectly, from each of the others.",
        ));
        blocks.push(Block::Callout {
            kind: CalloutKind::Caution,
            title: Some("Dependency cycle".into()),
            body: vec![Block::para(body)],
        });
    }

    if let Some(diagram) = neighbourhood_diagram(graph, mi, &deps, &users) {
        blocks.push(diagram);
    }

    if !children.is_empty() {
        blocks.push(Block::heading(2, "Submodules"));
        blocks.push(Block::Table {
            columns: vec!["Module".into(), "Files".into(), "Lines".into()],
            rows: children
                .iter()
                .map(|&c| {
                    let cs = Subject::Module(graph.modules[c].id.clone());
                    vec![
                        vec![module_link(graph, c)],
                        vec![b.fact_inline(&cs, "files")],
                        vec![b.fact_inline(&cs, "lines")],
                    ]
                })
                .collect(),
        });
    }

    let edge_list = |list: &[(usize, usize)], empty: &str| -> Block {
        if list.is_empty() {
            return Block::text(empty);
        }
        Block::List {
            ordered: false,
            items: list
                .iter()
                .map(|&(o, n)| {
                    vec![
                        module_link(graph, o),
                        Inline::text(format!(" — {}", plural(n, "import", "imports"))),
                    ]
                })
                .collect(),
        }
    };
    blocks.push(Block::heading(2, "Depends on"));
    blocks.push(edge_list(&deps, "No resolved imports from other modules."));
    blocks.push(Block::heading(2, "Used by"));
    blocks.push(edge_list(&users, "No other module imports from this one."));

    if !m.files.is_empty() {
        blocks.push(Block::heading(2, "Files"));
        let mut files: Vec<usize> = m.files.clone();
        files.sort_by(|&x, &y| {
            corpus.files[y]
                .lines
                .cmp(&corpus.files[x].lines)
                .then_with(|| corpus.files[x].path.cmp(&corpus.files[y].path))
        });
        let shown = files.len().min(MAX_FILE_ROWS);
        blocks.push(Block::Table {
            columns: vec!["File".into(), "Language".into(), "Lines".into()],
            rows: files[..shown]
                .iter()
                .map(|&f| {
                    let file = &corpus.files[f];
                    vec![
                        vec![file_link(&file.path)],
                        vec![Inline::text(language_name(file.language))],
                        vec![Inline::text(crate::pulse::model::facts::group_thousands(
                            file.lines,
                        ))],
                    ]
                })
                .collect(),
        });
        if files.len() > shown {
            blocks.push(Block::note(format!(
                "Showing the {shown} largest of {} files.",
                files.len()
            )));
        }
    }
    blocks
}

fn node_label(s: &str) -> String {
    s.replace(['"', '[', ']', '(', ')'], "'")
}

fn neighbourhood_diagram(
    graph: &ModuleGraph,
    mi: usize,
    deps: &[(usize, usize)],
    users: &[(usize, usize)],
) -> Option<Block> {
    if deps.is_empty() && users.is_empty() {
        return None;
    }
    let mut src = String::from("flowchart LR\n");
    let mut links = Vec::new();
    let mut add_node = |src: &mut String, i: usize| {
        let m = &graph.modules[i];
        src.push_str(&format!("  m{i}[\"{}\"]\n", node_label(m.name())));
        links.push((format!("m{i}"), Target::page(module_page_id(&m.id))));
    };
    add_node(&mut src, mi);
    for &(u, _) in users.iter().take(MAX_NEIGHBOURS) {
        add_node(&mut src, u);
    }
    for &(d, _) in deps.iter().take(MAX_NEIGHBOURS) {
        if !users.iter().take(MAX_NEIGHBOURS).any(|&(u, _)| u == d) {
            add_node(&mut src, d);
        }
    }
    for &(u, n) in users.iter().take(MAX_NEIGHBOURS) {
        src.push_str(&format!("  m{u} -->|{n}| m{mi}\n"));
    }
    for &(d, n) in deps.iter().take(MAX_NEIGHBOURS) {
        src.push_str(&format!("  m{mi} -->|{n}| m{d}\n"));
    }
    src.push_str(&format!("  class m{mi} focus\n"));
    let hidden =
        deps.len().saturating_sub(MAX_NEIGHBOURS) + users.len().saturating_sub(MAX_NEIGHBOURS);
    Some(Block::Diagram {
        kind: DiagramKind::Mermaid,
        source: src,
        links,
        caption: Some(if hidden > 0 {
            format!(
                "Imports in and out of this module (edge labels count imports; {hidden} more not drawn)."
            )
        } else {
            "Imports in and out of this module (edge labels count imports).".into()
        }),
    })
}

fn architecture_blocks(b: &SiteBuilder, graph: &ModuleGraph) -> Vec<Block> {
    let mut blocks = Vec::new();

    // Fallback narrative: which modules everything else leans on.
    let mut hubs: Vec<(usize, usize)> = (0..graph.modules.len())
        .map(|mi| (mi, graph.dependents(mi).len()))
        .filter(|&(_, n)| n > 0)
        .collect();
    hubs.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
    let mut fallback = vec![
        Inline::text("The codebase has "),
        b.fact_inline(&crate::pulse::model::Subject::Site, "modules"),
        Inline::text(" modules and "),
        b.fact_inline(&crate::pulse::model::Subject::Site, "source_files"),
        Inline::text(" source files."),
    ];
    if !hubs.is_empty() {
        fallback.push(Inline::text(" The most-used modules are "));
        for (i, &(mi, n)) in hubs.iter().take(3).enumerate() {
            if i > 0 {
                fallback.push(Inline::text(", "));
            }
            fallback.push(module_link(graph, mi));
            fallback.push(Inline::text(format!(" (used by {n})")));
        }
        fallback.push(Inline::text("."));
    }
    blocks.push(Block::Narrative {
        slot: crate::pulse::narrate::ids::ARCHITECTURE.into(),
        text: None,
        fallback: vec![Block::para(fallback)],
    });

    let site = crate::pulse::model::Subject::Site;
    blocks.push(Block::Stats {
        stats: vec![
            Stat {
                label: "Modules".into(),
                fact: b.fact(&site, "modules"),
            },
            Stat {
                label: "Source files".into(),
                fact: b.fact(&site, "source_files"),
            },
            Stat {
                label: "Lines of code".into(),
                fact: b.fact(&site, "source_lines"),
            },
            Stat {
                label: "Languages".into(),
                fact: b.fact(&site, "languages"),
            },
        ],
    });

    if let Some(d) = architecture_diagram(graph) {
        blocks.push(Block::heading(2, "Module graph"));
        blocks.push(d);
    }

    blocks.push(Block::heading(2, "Modules"));
    blocks.push(Block::Table {
        columns: vec![
            "Module".into(),
            "Files".into(),
            "Lines".into(),
            "Used by".into(),
            "Depends on".into(),
        ],
        rows: graph
            .modules
            .iter()
            .enumerate()
            .map(|(mi, m)| {
                let s = Subject::Module(m.id.clone());
                vec![
                    vec![module_link(graph, mi)],
                    vec![b.fact_inline(&s, "files")],
                    vec![b.fact_inline(&s, "lines")],
                    vec![b.fact_inline(&s, "dependents")],
                    vec![b.fact_inline(&s, "dependencies")],
                ]
            })
            .collect(),
    });

    blocks.push(Block::Cards {
        cards: vec![Card {
            title: "Dependency map".into(),
            to: Target::page(DEPENDENCY_MAP),
            description: Some("Most-imported files, cycles and isolated modules.".into()),
        }],
    });
    blocks
}

fn architecture_diagram(graph: &ModuleGraph) -> Option<Block> {
    if graph.modules.len() < 2 {
        return None;
    }
    let mut edges: Vec<(usize, usize, usize)> =
        graph.edges.iter().map(|(&(a, b), &n)| (a, b, n)).collect();
    edges.sort_by(|x, y| y.2.cmp(&x.2).then((x.0, x.1).cmp(&(y.0, y.1))));
    let total = edges.len();
    edges.truncate(MAX_DIAGRAM_EDGES);

    let mut src = String::from("flowchart LR\n");
    let mut links = Vec::new();
    for (mi, m) in graph.modules.iter().enumerate() {
        if m.tier != 1 {
            continue;
        }
        let children = graph.children(mi);
        if children.is_empty() {
            src.push_str(&format!("  m{mi}[\"{}\"]\n", node_label(m.name())));
        } else {
            src.push_str(&format!("  subgraph g{mi}[\"{}\"]\n", node_label(m.name())));
            src.push_str(&format!("    m{mi}[\"{}\"]\n", node_label(m.name())));
            for c in children {
                src.push_str(&format!(
                    "    m{c}[\"{}\"]\n",
                    node_label(graph.modules[c].name())
                ));
            }
            src.push_str("  end\n");
        }
    }
    for (mi, m) in graph.modules.iter().enumerate() {
        links.push((format!("m{mi}"), Target::page(module_page_id(&m.id))));
    }
    for (a, b, n) in &edges {
        src.push_str(&format!("  m{a} -->|{n}| m{b}\n"));
    }
    Some(Block::Diagram {
        kind: DiagramKind::Mermaid,
        source: src,
        links,
        caption: Some(if total > edges.len() {
            format!(
                "Arrows point from the importing module; labels count imports. The {} heaviest of {total} edges are drawn.",
                edges.len()
            )
        } else {
            "Arrows point from the importing module; labels count imports.".into()
        }),
    })
}

fn dependency_map_blocks(
    corpus: &Corpus,
    graph: &ModuleGraph,
    cycles: &[Vec<usize>],
) -> Vec<Block> {
    let mut blocks = vec![Block::heading(2, "Most-imported files")];
    let hot = graph.file_hotspots(corpus, 15);
    if hot.is_empty() {
        blocks.push(Block::text("No resolved imports between source files."));
    } else {
        blocks.push(Block::Table {
            columns: vec!["File".into(), "Module".into(), "Imported by".into()],
            rows: hot
                .iter()
                .map(|&(f, n)| {
                    let module = graph.owner.get(&f).copied();
                    vec![
                        vec![file_link(&corpus.files[f].path)],
                        module
                            .map(|mi| vec![module_link(graph, mi)])
                            .unwrap_or_default(),
                        vec![Inline::text(plural(n, "file", "files"))],
                    ]
                })
                .collect(),
        });
    }

    blocks.push(Block::heading(2, "Dependency cycles"));
    if cycles.is_empty() {
        blocks.push(Block::text("No dependency cycles between modules."));
    } else {
        blocks.push(Block::List {
            ordered: false,
            items: cycles
                .iter()
                .map(|c| {
                    let mut item = Vec::new();
                    for (i, &mi) in c.iter().enumerate() {
                        if i > 0 {
                            item.push(Inline::text(" ↔ "));
                        }
                        item.push(module_link(graph, mi));
                    }
                    item
                })
                .collect(),
        });
    }

    let isolated: Vec<usize> = (0..graph.modules.len())
        .filter(|&mi| graph.dependencies(mi).is_empty() && graph.dependents(mi).is_empty())
        .collect();
    blocks.push(Block::heading(2, "Isolated modules"));
    if isolated.is_empty() {
        blocks.push(Block::text(
            "Every module imports from or is imported by another module.",
        ));
    } else {
        blocks.push(Block::text(
            "These modules have no resolved imports to or from other modules. \
             They may be entry points, leaf utilities, or code loaded dynamically.",
        ));
        blocks.push(Block::List {
            ordered: false,
            items: isolated
                .iter()
                .map(|&mi| vec![module_link(graph, mi)])
                .collect(),
        });
    }
    blocks
}

/// Rows in a module's item table.
const MAX_ITEM_ROWS: usize = 300;

/// Every item in the module's files as a one-line signature with a source link:
/// the contributor's index, not the reference (that is the Docs tab's job).
fn item_blocks(corpus: &Corpus, files: &[usize], apis: &ApiIndex) -> Vec<Block> {
    fn push(rows: &mut Vec<Vec<Vec<Inline>>>, path: &str, it: &ApiItem, owner: Option<&str>) {
        if it.test_only {
            return;
        }
        let name = match owner {
            Some(o) => format!("{o}::{}", it.name),
            None => it.name.clone(),
        };
        let vis = match &it.visibility {
            Visibility::Public => "pub".to_string(),
            Visibility::Restricted(s) => format!("pub({s})"),
            Visibility::Private | Visibility::Inherited => String::new(),
        };
        rows.push(vec![
            vec![Inline::code_link(
                Target::Source {
                    loc: SourceLoc::lines(path, it.start_line, it.end_line),
                },
                name,
            )],
            vec![Inline::text(it.kind.label())],
            vec![Inline::text(vis)],
            vec![Inline::code(it.signature.clone())],
        ]);
    }
    let mut rows = Vec::new();
    let mut total = 0usize;
    for &f in files {
        let Some(api) = apis.files.get(&f) else {
            continue;
        };
        let path = &corpus.files[f].path;
        for it in &api.items {
            match it.kind {
                ApiKind::Module => {
                    for m in &it.members {
                        total += 1;
                        push(&mut rows, path, m, Some(&it.name));
                    }
                }
                _ => {
                    total += 1;
                    let owner = it
                        .self_type
                        .as_deref()
                        .map(|t| t.split('<').next().unwrap_or(t));
                    push(&mut rows, path, it, owner);
                }
            }
        }
    }
    if rows.is_empty() {
        return Vec::new();
    }
    let shown = rows.len().min(MAX_ITEM_ROWS);
    rows.truncate(shown);
    let mut out = vec![
        Block::heading(2, "Items"),
        Block::Table {
            columns: vec![
                "Item".into(),
                "Kind".into(),
                "Visibility".into(),
                "Signature".into(),
            ],
            rows,
        },
    ];
    if total > shown {
        out.push(Block::note(format!("Showing {shown} of {total} items.")));
    }
    out
}
