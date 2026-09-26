//! Python public surface.
//!
//! 1. Packages come from manifests in the index (`pyproject.toml`: `[project] name` or
//!    `[tool.poetry] name`, plus the `packages` / `package-dir` settings of setuptools,
//!    Poetry, Hatch, Flit and maturin; else `setup.cfg` / `setup.py`). The import
//!    package is `src/<name>/` (src layout) or `<name>/` (flat), `-` and `.` read as
//!    `_`; a single-module distribution is `<name>.py`. When the name does not match a
//!    directory (`python-dateutil` ships `dateutil`), every top-level package beside the
//!    manifest counts. With no manifest at all, every top-level source directory with
//!    an `__init__.py` (at the root or under `src/`) is a package.
//! 2. Modules are the `.py` files under the package: `pkg/sub/mod.py` is `pkg.sub.mod`,
//!    `__init__.py` is its package. A module is public unless a segment below the
//!    package starts with `_`. Only source-role files count (`test_*.py`, `conftest.py`
//!    and `tests/` are tests).
//! 3. An item is public when its module is, it is not a hidden dunder, and it is listed
//!    in the module's `__all__` or, without one, its name does not start with `_`.
//! 4. `from ._impl import Client` (also `from pkg._impl import *`) documents `Client`
//!    where it is imported, when the importing module exposes the name and the item is
//!    not already public where it is defined, like rustdoc inlining. Chains through
//!    private modules resolve, deepest module first.

use super::{Package, Surface, SurfaceItem, SurfaceModule};
use crate::models::Language;
use crate::parsers::api::{ApiFile, ApiItem};
use crate::pulse::extract::api_cache::ApiIndex;
use crate::pulse::extract::{ContentAccess, Corpus, FileRole};
use std::collections::{BTreeMap, BTreeSet};

impl Surface {
    /// Python packages in the corpus.
    pub fn resolve_python(corpus: &Corpus, apis: &ApiIndex, content: &ContentAccess) -> Self {
        let sources: Vec<&str> = corpus
            .files
            .iter()
            .filter(|f| f.role == FileRole::Source && f.language == Language::Python)
            .map(|f| f.path.as_str())
            .collect();
        if sources.is_empty() {
            return Self::default();
        }
        let manifests: Vec<(&str, &str)> = corpus
            .files
            .iter()
            .filter(|f| {
                matches!(f.role, FileRole::Config | FileRole::Build)
                    && matches!(
                        file_name(&f.path),
                        "pyproject.toml" | "setup.cfg" | "setup.py"
                    )
            })
            .filter_map(|f| content.read(&f.path).map(|t| (f.path.as_str(), t)))
            .collect();
        let file_api = |path: &str| -> Option<&ApiFile> {
            corpus.index_of(path).and_then(|i| apis.files.get(&i))
        };
        let mut packages: Vec<Package> = package_specs(&manifests, &sources)
            .into_iter()
            .map(|spec| build_package(spec, &sources, &file_api))
            .collect();
        packages.sort_by(|a, b| a.name.cmp(&b.name));
        Self { packages }
    }
}

/// A package to document: its import name and where its files are.
#[derive(Debug, Clone, PartialEq)]
struct PackageSpec {
    /// Import name (`my_pkg`).
    name: String,
    /// Distribution name (`my-pkg`).
    dist: String,
    manifest: String,
    /// Package directory (`src/my_pkg`), or the module file of a single-module
    /// distribution (`my_pkg.py`).
    root: String,
}

fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn parent_dir(path: &str) -> &str {
    path.rsplit_once('/').map(|(d, _)| d).unwrap_or("")
}

fn join(dir: &str, rel: &str) -> String {
    let rel = rel.trim_start_matches("./").trim_end_matches('/');
    match (dir.is_empty(), rel.is_empty() || rel == ".") {
        (_, true) => dir.to_string(),
        (true, false) => rel.to_string(),
        (false, false) => format!("{dir}/{rel}"),
    }
}

fn normalize(dist: &str) -> String {
    dist.to_ascii_lowercase().replace(['-', '.'], "_")
}

fn is_identifier(s: &str) -> bool {
    let mut c = s.chars();
    c.next().is_some_and(|f| f.is_alphabetic() || f == '_')
        && c.all(|x| x.is_alphanumeric() || x == '_')
}

/// What one directory's manifests say.
#[derive(Default)]
struct ManifestInfo {
    manifest: String,
    dist: Option<String>,
    /// Directories packages live in, relative to the manifest (`src`).
    bases: Vec<String>,
    /// Package directories, relative to the manifest (`src/pkg`, `lib/pkg`).
    dirs: Vec<String>,
    /// Top-level import names (`pkg`), looked up under `bases`.
    names: Vec<String>,
}

fn read_pyproject(text: &str, info: &mut ManifestInfo) {
    let Ok(t) = text.parse::<toml::Table>() else {
        return;
    };
    let get = |path: &[&str]| -> Option<&toml::Value> {
        let mut v = t.get(path[0])?;
        for k in &path[1..] {
            v = v.get(k)?;
        }
        Some(v)
    };
    let strs = |v: Option<&toml::Value>| -> Vec<String> {
        v.and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };
    info.dist = info.dist.take().or_else(|| {
        get(&["project", "name"])
            .or_else(|| get(&["tool", "poetry", "name"]))
            .and_then(|v| v.as_str())
            .map(str::to_string)
    });
    // setuptools
    if let Some(dirs) = get(&["tool", "setuptools", "package-dir"]).and_then(|v| v.as_table()) {
        for (k, v) in dirs {
            if let Some(d) = v.as_str() {
                if k.is_empty() {
                    info.bases.push(d.to_string());
                } else if !k.contains('.') {
                    info.dirs.push(d.to_string());
                }
            }
        }
    }
    info.bases.extend(strs(get(&[
        "tool",
        "setuptools",
        "packages",
        "find",
        "where",
    ])));
    info.names.extend(
        strs(get(&["tool", "setuptools", "packages"]))
            .into_iter()
            .filter(|p| !p.contains('.')),
    );
    // Poetry: packages = [{ include = "pkg", from = "src" }]
    if let Some(pkgs) = get(&["tool", "poetry", "packages"]).and_then(|v| v.as_array()) {
        for p in pkgs {
            if let Some(inc) = p.get("include").and_then(|v| v.as_str()) {
                let from = p.get("from").and_then(|v| v.as_str()).unwrap_or("");
                info.dirs.push(join(from, inc));
            }
        }
    }
    // Hatch: packages = ["src/pkg"]
    info.dirs.extend(strs(get(&[
        "tool", "hatch", "build", "targets", "wheel", "packages",
    ])));
    // Flit: [tool.flit.module] name = "pkg"
    if let Some(n) = get(&["tool", "flit", "module", "name"]).and_then(|v| v.as_str()) {
        info.names.push(n.to_string());
    }
    // maturin: python-source = "python", module-name = "pkg._native"
    if let Some(src) = get(&["tool", "maturin", "python-source"]).and_then(|v| v.as_str()) {
        info.bases.push(src.to_string());
    }
    if let Some(m) = get(&["tool", "maturin", "module-name"]).and_then(|v| v.as_str()) {
        info.names
            .push(m.split('.').next().unwrap_or(m).to_string());
    }
}

/// `setup.cfg`: `[metadata] name`, `[options] package_dir = =src`.
fn read_setup_cfg(text: &str, info: &mut ManifestInfo) {
    let mut section = String::new();
    let mut key = String::new();
    for raw in text.lines() {
        let line = raw.trim_end();
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') || t.starts_with(';') {
            continue;
        }
        if t.starts_with('[') {
            section = t.trim_matches(['[', ']']).trim().to_string();
            continue;
        }
        let continuation = raw.starts_with([' ', '\t']);
        let (k, v) = if continuation {
            (key.as_str(), t)
        } else {
            match t.split_once(['=', ':']) {
                Some((k, v)) => {
                    key = k.trim().to_string();
                    (key.as_str(), v.trim())
                }
                None => continue,
            }
        };
        match (section.as_str(), k) {
            ("metadata", "name") if !v.is_empty() && info.dist.is_none() => {
                info.dist = Some(v.to_string());
            }
            ("options", "package_dir") => {
                if let Some(d) = v.strip_prefix('=') {
                    info.bases.push(d.trim().to_string());
                }
            }
            _ => {}
        }
    }
}

static SETUP_NAME_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r#"\bname\s*=\s*["']([^"']+)["']"#).expect("valid regex")
});
static SETUP_BASE_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r#"package_dir\s*=\s*\{\s*["']["']\s*:\s*["']([^"']+)["']"#)
        .expect("valid regex")
});

/// `setup.py`: `setup(name="…", package_dir={"": "src"})`, read, never run.
fn read_setup_py(text: &str, info: &mut ManifestInfo) {
    if info.dist.is_none() {
        info.dist = SETUP_NAME_RE.captures(text).map(|c| c[1].to_string());
    }
    if let Some(c) = SETUP_BASE_RE.captures(text) {
        info.bases.push(c[1].to_string());
    }
}

/// Top-level packages under `base`: `base/<x>/…` directories holding source files
/// (an `__init__.py`, or any file for a namespace package).
fn top_level_packages(base: &str, sources: &[&str], need_init: bool) -> BTreeSet<String> {
    let prefix = if base.is_empty() {
        String::new()
    } else {
        format!("{base}/")
    };
    sources
        .iter()
        .filter_map(|s| s.strip_prefix(&prefix))
        .filter_map(|rest| {
            let (top, tail) = rest.split_once('/')?;
            (is_identifier(top) && (!need_init || tail == "__init__.py")).then(|| join(base, top))
        })
        .collect()
}

/// Packages declared by the manifests, or found by layout when there are none.
fn package_specs(manifests: &[(&str, &str)], sources: &[&str]) -> Vec<PackageSpec> {
    let has_dir = |dir: &str| {
        let p = format!("{dir}/");
        sources.iter().any(|s| s.starts_with(&p))
    };
    let has_file = |f: &str| sources.contains(&f);

    let mut by_dir: BTreeMap<&str, ManifestInfo> = BTreeMap::new();
    // pyproject.toml first: it wins over setup.cfg and setup.py for the name.
    let rank = |p: &str| match file_name(p) {
        "pyproject.toml" => 0,
        "setup.cfg" => 1,
        _ => 2,
    };
    let mut ordered: Vec<&(&str, &str)> = manifests.iter().collect();
    ordered.sort_by_key(|(p, _)| (parent_dir(p), rank(p)));
    for (path, text) in ordered {
        let info = by_dir.entry(parent_dir(path)).or_default();
        if info.manifest.is_empty() {
            info.manifest = path.to_string();
        }
        match file_name(path) {
            "pyproject.toml" => read_pyproject(text, info),
            "setup.cfg" => read_setup_cfg(text, info),
            _ => read_setup_py(text, info),
        }
    }

    let mut out: Vec<PackageSpec> = Vec::new();
    let mut claimed: BTreeSet<String> = BTreeSet::new();
    let mut push = |out: &mut Vec<PackageSpec>, root: String, dist: &str, manifest: &str| {
        let name = file_name(&root).trim_end_matches(".py").to_string();
        if is_identifier(&name) && claimed.insert(root.clone()) {
            out.push(PackageSpec {
                name,
                dist: dist.to_string(),
                manifest: manifest.to_string(),
                root,
            });
        }
    };
    for (dir, info) in &by_dir {
        let Some(dist) = info.dist.as_deref() else {
            continue;
        };
        let mut bases: Vec<String> = info.bases.iter().map(|b| join(dir, b)).collect();
        bases.extend([join(dir, "src"), dir.to_string()]);
        let before = out.len();
        for d in &info.dirs {
            let d = join(dir, d);
            if has_dir(&d) {
                push(&mut out, d, dist, &info.manifest);
            }
        }
        let mut names = info.names.clone();
        if out.len() == before && names.is_empty() {
            names.push(normalize(dist));
        }
        for n in &names {
            let found = bases.iter().find_map(|b| {
                let d = join(b, n);
                if has_dir(&d) {
                    Some(d)
                } else {
                    let f = format!("{d}.py");
                    has_file(&f).then_some(f)
                }
            });
            if let Some(root) = found {
                push(&mut out, root, dist, &info.manifest);
            }
        }
        if out.len() == before {
            // `python-dateutil` ships `dateutil`: every top-level package beside the
            // manifest.
            for b in &bases {
                for d in top_level_packages(b, sources, true) {
                    push(&mut out, d, dist, &info.manifest);
                }
                if out.len() > before {
                    break;
                }
            }
        }
    }
    if by_dir.values().all(|i| i.dist.is_none()) {
        for base in ["src", ""] {
            for d in top_level_packages(base, sources, true) {
                let dist = file_name(&d).to_string();
                push(&mut out, d, &dist, "");
            }
        }
    }
    out
}

/// Module tree, items and re-exports of one package.
fn build_package<'f>(
    spec: PackageSpec,
    sources: &[&str],
    file_api: &dyn Fn(&str) -> Option<&'f ApiFile>,
) -> Package {
    // Dotted path → file (`None` for a directory without `__init__.py`).
    let mut files: BTreeMap<String, Option<String>> = BTreeMap::new();
    if spec.root.ends_with(".py") {
        files.insert(spec.name.clone(), Some(spec.root.clone()));
    } else {
        files.insert(spec.name.clone(), None);
        let prefix = format!("{}/", spec.root);
        for s in sources.iter().filter(|s| s.ends_with(".py")) {
            let Some(rel) = s.strip_prefix(&prefix) else {
                continue;
            };
            let mut segs: Vec<&str> = rel.trim_end_matches(".py").split('/').collect();
            if segs.last() == Some(&"__init__") {
                segs.pop();
            }
            if !segs.iter().all(|s| is_identifier(s)) {
                continue;
            }
            let mut path = spec.name.clone();
            for seg in &segs {
                path = format!("{path}.{seg}");
                files.entry(path.clone()).or_insert(None);
            }
            files.insert(path, Some(s.to_string()));
        }
    }

    let mut k = Package {
        name: spec.name.clone(),
        lang: Language::Python,
        sep: ".",
        package: spec.dist,
        manifest: spec.manifest,
        is_lib: true,
        root_file: files
            .get(&spec.name)
            .cloned()
            .flatten()
            .unwrap_or_else(|| spec.root.clone()),
        modules: Vec::new(),
    };
    // Depth-first, children by name: `modules[0]` is the package.
    fn add<'f>(
        k: &mut Package,
        files: &BTreeMap<String, Option<String>>,
        path: &str,
        parent: Option<usize>,
        dir: &str,
        file_api: &dyn Fn(&str) -> Option<&'f ApiFile>,
    ) {
        let file = files.get(path).cloned().flatten();
        let name = path.rsplit('.').next().unwrap_or(path).to_string();
        let public = parent.is_none_or(|p| k.modules[p].public)
            && (parent.is_none() || !name.starts_with('_'));
        let api = file.as_deref().and_then(file_api);
        let idx = k.modules.len();
        let module_path = path.to_string();
        let exports = api.and_then(|a| a.exports.as_ref());
        let items = api
            .map(|a| {
                a.items
                    .iter()
                    .map(|it| SurfaceItem {
                        path: format!("{module_path}.{}", it.name),
                        public: public && exposes(exports, &it.name) && !it.hidden,
                        item: it.clone(),
                        file: file.clone().unwrap_or_default(),
                        defined_at: None,
                        impl_files: BTreeMap::new(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        k.modules.push(SurfaceModule {
            path: module_path,
            name,
            file: file.clone().unwrap_or_else(|| dir.to_string()),
            public,
            doc: api.and_then(|a| a.module_doc.clone()),
            items,
            children: Vec::new(),
            parent,
        });
        if let Some(p) = parent {
            k.modules[p].children.push(idx);
        }
        let prefix = format!("{path}.");
        let children: Vec<String> = files
            .keys()
            .filter(|c| {
                c.strip_prefix(&prefix)
                    .is_some_and(|rest| !rest.contains('.'))
            })
            .cloned()
            .collect();
        for c in children {
            let seg = c.rsplit('.').next().unwrap_or(&c).to_string();
            add(k, files, &c, Some(idx), &format!("{dir}/{seg}"), file_api);
        }
    }
    add(&mut k, &files, &spec.name, None, &spec.root, file_api);
    apply_reexports(&mut k, file_api);
    k
}

/// Whether a module with export list `exports` exposes `name`.
fn exposes(exports: Option<&Vec<String>>, name: &str) -> bool {
    match exports {
        Some(list) => list.iter().any(|e| e == name),
        None => !name.starts_with('_'),
    }
}

/// Module index for an import path written in module `from` (`.x.A` → module of `.x`,
/// `A`). `None` when the module is outside the package.
fn resolve_import(k: &Package, from: usize, path: &str) -> Option<(usize, String)> {
    let level = path.len() - path.trim_start_matches('.').len();
    let rest = &path[level..];
    let (module, name) = match rest.rsplit_once('.') {
        Some((m, n)) => (m, n),
        None => ("", rest),
    };
    let target = if level > 0 {
        let m = &k.modules[from];
        let is_package = m.file.ends_with("__init__.py") || !m.file.ends_with(".py");
        let mut base = if is_package {
            m.path.clone()
        } else {
            m.path.rsplit_once('.').map(|(p, _)| p.to_string())?
        };
        for _ in 1..level {
            base = base.rsplit_once('.').map(|(p, _)| p.to_string())?;
        }
        if module.is_empty() {
            base
        } else {
            format!("{base}.{module}")
        }
    } else {
        module.to_string()
    };
    let idx = k.modules.iter().position(|m| m.path == target)?;
    Some((idx, name.to_string()))
}

/// Document imported items where they are re-exported.
fn apply_reexports<'f>(k: &mut Package, file_api: &dyn Fn(&str) -> Option<&'f ApiFile>) {
    for mi in (0..k.modules.len()).rev() {
        let Some(api) = file_api(&k.modules[mi].file) else {
            continue;
        };
        let exports = api.exports.as_ref();
        for r in &api.reexports {
            let Some((target, name)) = resolve_import(k, mi, &r.path) else {
                continue;
            };
            if target == mi {
                continue;
            }
            let target_exports = file_api(&k.modules[target].file).and_then(|a| a.exports.as_ref());
            let picks: Vec<SurfaceItem> = k.modules[target]
                .items
                .iter()
                .filter(|it| {
                    if name == "*" {
                        exposes(target_exports, &it.item.name)
                    } else {
                        it.item.name == name
                    }
                })
                .cloned()
                .collect();
            for it in picks {
                let exported_as = if r.name == "*" {
                    it.item.name.clone()
                } else {
                    r.name.clone()
                };
                let m = &k.modules[mi];
                let public = m.public && exposes(exports, &exported_as) && !it.item.hidden;
                // Already documented where it is defined, a local definition of the same
                // name, or a name this (public) module does not expose.
                if it.public
                    || m.items.iter().any(|x| x.item.name == exported_as)
                    || (m.public && !public)
                {
                    continue;
                }
                let mut copy = it.clone();
                copy.defined_at = Some(it.defined_at.clone().unwrap_or(it.path.clone()));
                copy.path = format!("{}.{exported_as}", m.path);
                copy.item = ApiItem {
                    name: exported_as,
                    ..it.item
                };
                copy.public = public;
                k.modules[mi].items.push(copy);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        files: BTreeMap<&'static str, &'static str>,
    }

    impl Fixture {
        fn resolve(&self) -> Surface {
            let sources: Vec<&str> = self
                .files
                .keys()
                .copied()
                .filter(|p| {
                    p.ends_with(".py")
                        && crate::pulse::extract::roles::classify(p) == FileRole::Source
                })
                .collect();
            let manifests: Vec<(&str, &str)> = self
                .files
                .iter()
                .filter(|(p, _)| {
                    matches!(file_name(p), "pyproject.toml" | "setup.cfg" | "setup.py")
                })
                .map(|(p, t)| (*p, *t))
                .collect();
            let apis: BTreeMap<&str, ApiFile> = sources
                .iter()
                .map(|p| {
                    (
                        *p,
                        crate::parsers::api::python::extract(self.files[p]).unwrap(),
                    )
                })
                .collect();
            let file_api = |p: &str| apis.get(p);
            let mut packages: Vec<Package> = package_specs(&manifests, &sources)
                .into_iter()
                .map(|s| build_package(s, &sources, &file_api))
                .collect();
            packages.sort_by(|a, b| a.name.cmp(&b.name));
            Surface { packages }
        }
    }

    fn fixture() -> Fixture {
        let mut files = BTreeMap::new();
        files.insert(
            "pyproject.toml",
            "[project]\nname = \"my-pkg\"\nversion = \"0.1.0\"\n",
        );
        files.insert(
            "src/my_pkg/__init__.py",
            "\"\"\"My package.\"\"\"\nfrom ._core import Engine, _hidden\nfrom ._impl import *\nfrom .api import run\n__all__ = [\"Engine\", \"VERSION\", \"Fast\", \"run\"]\nVERSION = \"1\"\nLIMIT = 3\n",
        );
        files.insert(
            "src/my_pkg/_core.py",
            "class Engine:\n    \"\"\"Runs.\"\"\"\n    def go(self): ...\nclass NotExported: ...\ndef _hidden(): ...\n",
        );
        files.insert(
            "src/my_pkg/_impl/__init__.py",
            "from ._fast import Fast\n__all__ = [\"Fast\"]\n",
        );
        files.insert("src/my_pkg/_impl/_fast.py", "class Fast: ...\n");
        files.insert(
            "src/my_pkg/api.py",
            "def run(): ...\ndef _helper(): ...\nclass Client: ...\n",
        );
        files.insert("src/my_pkg/sub/deep.py", "def f(): ...\n");
        files.insert("src/my_pkg/__main__.py", "def main(): ...\n");
        files.insert("src/my_pkg/test_api.py", "def test_x(): ...\n");
        files.insert("tests/test_it.py", "def test_y(): ...\n");
        files.insert("scripts/tool.py", "def tool(): ...\n");
        Fixture { files }
    }

    fn item<'a>(k: &'a Package, path: &str) -> &'a SurfaceItem {
        k.modules
            .iter()
            .flat_map(|m| &m.items)
            .find(|i| i.path == path)
            .unwrap_or_else(|| panic!("{path} not found"))
    }

    #[test]
    fn packages_and_module_tree() {
        let api = fixture().resolve();
        assert_eq!(api.packages.len(), 1);
        let k = &api.packages[0];
        assert_eq!((k.name.as_str(), k.package.as_str()), ("my_pkg", "my-pkg"));
        assert_eq!(k.sep, ".");
        assert_eq!(k.root_file, "src/my_pkg/__init__.py");
        assert_eq!(k.manifest, "pyproject.toml");
        let mods: Vec<(&str, bool)> = k
            .modules
            .iter()
            .map(|m| (m.path.as_str(), m.public))
            .collect();
        assert_eq!(
            mods,
            vec![
                ("my_pkg", true),
                ("my_pkg.__main__", false),
                ("my_pkg._core", false),
                ("my_pkg._impl", false),
                ("my_pkg._impl._fast", false),
                ("my_pkg.api", true),
                ("my_pkg.sub", true),
                ("my_pkg.sub.deep", true),
            ],
            "test files are not modules; `sub` has no __init__.py (namespace package)"
        );
        assert_eq!(k.modules[0].doc.as_ref().unwrap().summary, "My package.");
        assert_eq!(k.modules[6].file, "src/my_pkg/sub");
    }

    #[test]
    fn all_and_reexports() {
        let api = fixture().resolve();
        let k = &api.packages[0];
        assert!(item(k, "my_pkg.VERSION").public, "listed in __all__");
        assert!(!item(k, "my_pkg.LIMIT").public, "not in __all__");
        assert!(
            item(k, "my_pkg.api.run").public,
            "no __all__: public by name"
        );
        assert!(!item(k, "my_pkg.api._helper").public);
        assert!(item(k, "my_pkg.api.Client").public);

        let engine = item(k, "my_pkg.Engine");
        assert!(engine.public, "re-exported from a private module");
        assert_eq!(engine.defined_at.as_deref(), Some("my_pkg._core.Engine"));
        assert_eq!(engine.item.members[0].name, "go");
        assert!(!item(k, "my_pkg._core.Engine").public);

        let fast = item(k, "my_pkg.Fast");
        assert!(fast.public, "star import through a private package");
        assert_eq!(fast.defined_at.as_deref(), Some("my_pkg._impl._fast.Fast"));

        let root = &k.modules[0];
        assert!(
            !root.items.iter().any(|i| i.item.name == "run"),
            "already public in my_pkg.api: not duplicated"
        );
        assert!(!root.items.iter().any(|i| i.item.name == "_hidden"));
        assert!(!root.items.iter().any(|i| i.item.name == "NotExported"));
    }

    #[test]
    fn manifests_and_layouts() {
        let src = [
            "lib/foo/__init__.py",
            "lib/foo/a.py",
            "bar.py",
            "dateutil/__init__.py",
        ];
        let specs = |m: &[(&str, &str)]| -> Vec<(String, String, String)> {
            package_specs(m, &src)
                .into_iter()
                .map(|s| (s.name, s.dist, s.root))
                .collect()
        };
        assert_eq!(
            specs(&[(
                "pyproject.toml",
                "[tool.poetry]\nname = \"foo-lib\"\npackages = [{ include = \"foo\", from = \"lib\" }]\n"
            )]),
            vec![("foo".into(), "foo-lib".into(), "lib/foo".into())]
        );
        assert_eq!(
            specs(&[(
                "pyproject.toml",
                "[project]\nname = \"x\"\n[tool.setuptools]\npackage-dir = { \"\" = \"lib\" }\npackages = [\"foo\", \"foo.a\"]\n"
            )]),
            vec![("foo".into(), "x".into(), "lib/foo".into())]
        );
        assert_eq!(
            specs(&[("setup.py", "setup(name='bar', py_modules=['bar'])")]),
            vec![("bar".into(), "bar".into(), "bar.py".into())],
            "single-module distribution"
        );
        assert_eq!(
            specs(&[("setup.cfg", "[metadata]\nname = python-dateutil\n")]),
            vec![(
                "dateutil".into(),
                "python-dateutil".into(),
                "dateutil".into()
            )],
            "name mismatch: top-level packages beside the manifest"
        );
        assert_eq!(
            specs(&[]),
            vec![("dateutil".into(), "dateutil".into(), "dateutil".into())],
            "no manifest: top-level directories with __init__.py"
        );
    }

    #[test]
    fn relative_imports_resolve() {
        let api = fixture().resolve();
        let k = &api.packages[0];
        let idx = |p: &str| k.modules.iter().position(|m| m.path == p).unwrap();
        let root = idx("my_pkg");
        let deep = idx("my_pkg.sub.deep");
        assert_eq!(
            resolve_import(k, root, "._core.Engine"),
            Some((idx("my_pkg._core"), "Engine".into()))
        );
        assert_eq!(
            resolve_import(k, deep, "..api.run"),
            Some((idx("my_pkg.api"), "run".into()))
        );
        assert_eq!(
            resolve_import(k, deep, ".x"),
            Some((idx("my_pkg.sub"), "x".into()))
        );
        assert_eq!(
            resolve_import(k, deep, "my_pkg.api.*"),
            Some((idx("my_pkg.api"), "*".into()))
        );
        assert_eq!(resolve_import(k, root, "typing.Any"), None);
    }
}
