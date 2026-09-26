//! Modules: directories of source files, and the dependency graph between them.
//!
//! A module is a top-level directory (tier 1) or one of its immediate subdirectories
//! with at least three source files (tier 2). Deeper files belong to their nearest
//! module. Only [`FileRole::Source`] files count, so test corpora, fixtures and build
//! scripts never become modules. Source files at the repository root form the root
//! module `.`.

use crate::pulse::extract::{Corpus, FileRole};
use crate::pulse::model::ModuleId;
use std::collections::{BTreeMap, BTreeSet};

/// Minimum source files for a subdirectory to be its own module.
pub const SUBMODULE_MIN_FILES: usize = 3;

#[derive(Debug, Clone)]
pub struct Module {
    pub id: ModuleId,
    pub tier: u8,
    pub parent: Option<ModuleId>,
    /// Indices into `Corpus::files`, sorted by path.
    pub files: Vec<usize>,
    pub lines: u64,
    /// Language display name → file count.
    pub languages: BTreeMap<String, usize>,
}

impl Module {
    pub fn is_root(&self) -> bool {
        self.id.as_str() == "."
    }

    /// Display name: the path, or `(root)`.
    pub fn name(&self) -> &str {
        if self.is_root() {
            "(root)"
        } else {
            self.id.as_str()
        }
    }
}

/// Module-level view of the corpus.
#[derive(Debug, Clone)]
pub struct ModuleGraph {
    /// Sorted by id.
    pub modules: Vec<Module>,
    /// File index → owning module index (source files only).
    pub owner: BTreeMap<usize, usize>,
    /// (from, to) module indices → number of file-level imports.
    pub edges: BTreeMap<(usize, usize), usize>,
}

impl ModuleGraph {
    pub fn build(corpus: &Corpus, max_depth: u8, min_files: usize) -> Self {
        let source: Vec<(usize, &str)> = corpus
            .with_role(FileRole::Source)
            .map(|(i, f)| (i, f.path.as_str()))
            .collect();

        // Count source files per tier-2 candidate directory (`a/b` for `a/b/**`).
        let mut sub_counts: BTreeMap<String, usize> = BTreeMap::new();
        if max_depth >= 2 {
            for (_, path) in &source {
                let parts: Vec<&str> = path.split('/').collect();
                if parts.len() >= 3 {
                    *sub_counts
                        .entry(format!("{}/{}", parts[0], parts[1]))
                        .or_default() += 1;
                }
            }
        }

        let mut by_id: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, path) in &source {
            let parts: Vec<&str> = path.split('/').collect();
            let id = match parts.len() {
                1 => ".".to_string(),
                2 => parts[0].to_string(),
                _ => {
                    let sub = format!("{}/{}", parts[0], parts[1]);
                    if sub_counts.get(&sub).copied().unwrap_or(0) >= SUBMODULE_MIN_FILES {
                        sub
                    } else {
                        parts[0].to_string()
                    }
                }
            };
            by_id.entry(id).or_default().push(*i);
        }
        // Every tier-2 module needs its tier-1 parent, even with no direct files.
        let parents: BTreeSet<String> = by_id
            .keys()
            .filter_map(|id| id.split_once('/').map(|(p, _)| p.to_string()))
            .collect();
        for p in parents {
            by_id.entry(p).or_default();
        }

        let mut modules: Vec<Module> = by_id
            .into_iter()
            .map(|(id, mut files)| {
                files.sort_unstable();
                let mut languages = BTreeMap::new();
                let mut lines = 0;
                for &i in &files {
                    let f = &corpus.files[i];
                    lines += f.lines;
                    *languages
                        .entry(crate::pulse::extract::language_name(f.language))
                        .or_insert(0) += 1;
                }
                let (tier, parent) = match id.split_once('/') {
                    Some((p, _)) => (2, Some(ModuleId::new(p))),
                    None => (1, None),
                };
                Module {
                    id: ModuleId::new(id),
                    tier,
                    parent,
                    files,
                    lines,
                    languages,
                }
            })
            .collect();

        // Drop small modules, but never a parent that still has children.
        let total_files = |m: &Module, all: &[Module]| {
            m.files.len()
                + all
                    .iter()
                    .filter(|c| c.parent.as_ref() == Some(&m.id))
                    .map(|c| c.files.len())
                    .sum::<usize>()
        };
        let keep: Vec<bool> = modules
            .iter()
            .map(|m| total_files(m, &modules) >= min_files.max(1))
            .collect();
        let mut i = 0;
        modules.retain(|_| {
            i += 1;
            keep[i - 1]
        });

        let mut owner = BTreeMap::new();
        for (mi, m) in modules.iter().enumerate() {
            for &f in &m.files {
                owner.insert(f, mi);
            }
        }
        let mut edges: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for &(a, b) in &corpus.edges {
            if let (Some(&ma), Some(&mb)) = (owner.get(&a), owner.get(&b))
                && ma != mb
            {
                *edges.entry((ma, mb)).or_insert(0) += 1;
            }
        }
        Self {
            modules,
            owner,
            edges,
        }
    }

    pub fn index_of(&self, id: &ModuleId) -> Option<usize> {
        self.modules.binary_search_by(|m| m.id.cmp(id)).ok()
    }

    pub fn children(&self, mi: usize) -> Vec<usize> {
        let id = &self.modules[mi].id;
        (0..self.modules.len())
            .filter(|&c| self.modules[c].parent.as_ref() == Some(id))
            .collect()
    }

    /// Modules `mi` imports from, heaviest first: (module, import count).
    pub fn dependencies(&self, mi: usize) -> Vec<(usize, usize)> {
        let mut v: Vec<(usize, usize)> = self
            .edges
            .iter()
            .filter(|((a, _), _)| *a == mi)
            .map(|((_, b), n)| (*b, *n))
            .collect();
        v.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
        v
    }

    /// Modules that import from `mi`, heaviest first.
    pub fn dependents(&self, mi: usize) -> Vec<(usize, usize)> {
        let mut v: Vec<(usize, usize)> = self
            .edges
            .iter()
            .filter(|((_, b), _)| *b == mi)
            .map(|((a, _), n)| (*a, *n))
            .collect();
        v.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
        v
    }

    /// Strongly connected components with more than one module (dependency cycles),
    /// each sorted, the list sorted. Tarjan's algorithm.
    pub fn cycles(&self) -> Vec<Vec<usize>> {
        let n = self.modules.len();
        let mut adj = vec![Vec::new(); n];
        for &(a, b) in self.edges.keys() {
            adj[a].push(b);
        }
        struct State {
            index: usize,
            idx: Vec<Option<usize>>,
            low: Vec<usize>,
            on: Vec<bool>,
            stack: Vec<usize>,
            out: Vec<Vec<usize>>,
        }
        fn visit(v: usize, adj: &[Vec<usize>], s: &mut State) {
            s.idx[v] = Some(s.index);
            s.low[v] = s.index;
            s.index += 1;
            s.stack.push(v);
            s.on[v] = true;
            for &w in &adj[v] {
                match s.idx[w] {
                    None => {
                        visit(w, adj, s);
                        s.low[v] = s.low[v].min(s.low[w]);
                    }
                    Some(iw) if s.on[w] => s.low[v] = s.low[v].min(iw),
                    _ => {}
                }
            }
            if Some(s.low[v]) == s.idx[v] {
                let mut comp = Vec::new();
                while let Some(w) = s.stack.pop() {
                    s.on[w] = false;
                    comp.push(w);
                    if w == v {
                        break;
                    }
                }
                if comp.len() > 1 {
                    comp.sort_unstable();
                    s.out.push(comp);
                }
            }
        }
        let mut s = State {
            index: 0,
            idx: vec![None; n],
            low: vec![0; n],
            on: vec![false; n],
            stack: Vec::new(),
            out: Vec::new(),
        };
        for v in 0..n {
            if s.idx[v].is_none() {
                visit(v, &adj, &mut s);
            }
        }
        s.out.sort();
        s.out
    }

    /// File-level fan-in within source files: (file index, dependents), highest first.
    pub fn file_hotspots(&self, corpus: &Corpus, limit: usize) -> Vec<(usize, usize)> {
        let mut fan_in: BTreeMap<usize, usize> = BTreeMap::new();
        for &(a, b) in &corpus.edges {
            if self.owner.contains_key(&a) && self.owner.contains_key(&b) {
                *fan_in.entry(b).or_insert(0) += 1;
            }
        }
        let mut v: Vec<(usize, usize)> = fan_in.into_iter().collect();
        v.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
        v.truncate(limit);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Language;
    use crate::pulse::extract::{FileInfo, roles};
    use std::path::{Path, PathBuf};

    fn corpus(paths: &[&str], edges: &[(usize, usize)]) -> Corpus {
        let mut files: Vec<FileInfo> = paths
            .iter()
            .map(|p| FileInfo {
                path: p.to_string(),
                language: Language::from_path(Path::new(p)),
                lines: 10,
                role: roles::classify(p),
            })
            .collect();
        files.sort_by(|a, b| a.path.cmp(&b.path));
        Corpus {
            root: PathBuf::from("."),
            files,
            edges: edges.to_vec(),
            readme: None,
        }
    }

    #[test]
    fn modules_skip_non_source_and_split_tier_two() {
        let c = corpus(
            &[
                "build.rs",
                "src/lib.rs",
                "src/main.rs",
                "src/pulse/a.rs",
                "src/pulse/b.rs",
                "src/pulse/deep/c.rs",
                "src/tiny/x.rs",
                "tests/corpus/sample.rs",
                "tests/it.rs",
                "main.py",
            ],
            &[],
        );
        let g = ModuleGraph::build(&c, 2, 1);
        let ids: Vec<&str> = g.modules.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec![".", "src", "src/pulse"]);
        let src = &g.modules[g.index_of(&"src".into()).unwrap()];
        // src/tiny has 1 file (< 3), so it stays in `src`.
        assert_eq!(src.files.len(), 3);
        let pulse = &g.modules[g.index_of(&"src/pulse".into()).unwrap()];
        assert_eq!(
            pulse.files.len(),
            3,
            "deep files belong to the tier-2 module"
        );
        assert_eq!(pulse.parent.as_ref().unwrap().as_str(), "src");
        assert_eq!(g.modules[0].name(), "(root)");
    }

    #[test]
    fn depth_one_keeps_top_level_only() {
        let c = corpus(&["src/a/1.rs", "src/a/2.rs", "src/a/3.rs"], &[]);
        let g = ModuleGraph::build(&c, 1, 1);
        let ids: Vec<&str> = g.modules.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["src"]);
    }

    #[test]
    fn edges_cycles_and_hotspots() {
        // Files sorted: a/1.rs(0) a/2.rs(1) b/1.rs(2) b/2.rs(3) c/1.rs(4) c/2.rs(5)
        let c = corpus(
            &["a/1.rs", "a/2.rs", "b/1.rs", "b/2.rs", "c/1.rs", "c/2.rs"],
            &[(0, 2), (1, 2), (2, 0), (4, 2)],
        );
        let g = ModuleGraph::build(&c, 2, 1);
        let (a, b, cc) = (0, 1, 2);
        assert_eq!(g.edges.get(&(a, b)), Some(&2));
        assert_eq!(g.edges.get(&(b, a)), Some(&1));
        assert_eq!(g.dependents(b), vec![(a, 2), (cc, 1)]);
        assert_eq!(g.cycles(), vec![vec![a, b]]);
        assert_eq!(g.file_hotspots(&c, 1), vec![(2, 3)]);
    }
}
