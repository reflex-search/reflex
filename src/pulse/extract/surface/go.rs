//! Go public surface.
//!
//! 1. Go modules come from `go.mod` files (`module github.com/x/y`). A directory belongs
//!    to the nearest `go.mod` at or above it (nested modules, like Kubernetes'
//!    `staging/src/k8s.io/*`, are their own packages).
//! 2. Each directory under a module root with non-test Go source files is a Go package.
//!    `package main` directories are commands, not API; directories named `internal`
//!    (Go's visibility rule), `testdata` or `vendor`, or starting with `.` or `_` (the
//!    go tool ignores them) are skipped. Files marked `//go:build ignore` do not count.
//! 3. An item is public when its name is exported (uppercase). Methods attach to their
//!    receiver type from any file of the package, so an unexported type's methods are
//!    not documented. When build-tagged files (`x_linux.go`, `x_windows.go`) define the
//!    same name, the untagged or Linux/amd64 one wins, like pkgsite's default view.
//! 4. Testable examples (`ExampleT_M` in `_test.go` files of the package's directory)
//!    are attached to the item they name.

use super::{Package, Surface, SurfaceItem, SurfaceModule};
use crate::models::Language;
use crate::parsers::api::go::{self as goapi, is_exported};
use crate::parsers::api::{ApiFile, ApiItem, ApiKind, CodeExample, DocComment, doc};
use crate::pulse::extract::api_cache::ApiIndex;
use crate::pulse::extract::{ContentAccess, Corpus, FileRole};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

impl Surface {
    /// Go modules in the corpus.
    pub fn resolve_go(corpus: &Corpus, apis: &ApiIndex, content: &ContentAccess) -> Self {
        let mods: Vec<(String, &str)> = corpus
            .files
            .iter()
            .filter(|f| f.path == "go.mod" || f.path.ends_with("/go.mod"))
            .filter_map(|f| Some((f.path.clone(), content.read(&f.path)?)))
            .collect();
        if mods.is_empty() {
            return Self::default();
        }
        let start = std::time::Instant::now();
        let mut files = Vec::new();
        let mut tests = Vec::new();
        for (i, f) in corpus.files.iter().enumerate() {
            if f.language != Language::Go {
                continue;
            }
            if f.path.ends_with("_test.go") {
                tests.push(f.path.clone());
                continue;
            }
            if f.role != FileRole::Source {
                continue;
            }
            if let Some(api) = apis.files.get(&i) {
                let ignored = content.read(&f.path).is_some_and(build_ignored);
                files.push(GoFile {
                    path: f.path.clone(),
                    api,
                    ignored,
                });
            }
        }
        let read = |p: &str| content.read(p);
        let surface = resolve(&mods, &files, &tests, &read);
        log::info!(
            "pulse go surface: {} modules, {} packages in {:?}",
            surface.packages.len(),
            surface
                .packages
                .iter()
                .map(|p| p.modules.len())
                .sum::<usize>(),
            start.elapsed()
        );
        surface
    }
}

/// One non-test Go source file with its extracted API.
struct GoFile<'a> {
    path: String,
    api: &'a ApiFile,
    /// `//go:build ignore`: a generator or example program, not part of the package.
    ignored: bool,
}

fn parent_dir(path: &str) -> &str {
    path.rsplit_once('/').map(|(d, _)| d).unwrap_or("")
}

fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// The module path declared by a `go.mod`.
fn module_path(go_mod: &str) -> Option<String> {
    go_mod.lines().find_map(|l| {
        let l = l.split("//").next().unwrap_or("").trim();
        let rest = l.strip_prefix("module")?;
        if !rest.starts_with([' ', '\t', '"']) {
            return None;
        }
        let m = rest.trim().trim_matches(['"', '`']).to_string();
        (!m.is_empty()).then_some(m)
    })
}

/// Display name of a module: the last path segment, skipping a `/vN` major version
/// (`gopkg.in/yaml.v3` → `yaml`). `.` separates items in documented paths, so it
/// never appears in the name.
fn module_name(path: &str) -> String {
    let is_major =
        |s: &str| s.len() > 1 && s.starts_with('v') && s[1..].chars().all(|c| c.is_ascii_digit());
    let mut segs = path.rsplit('/');
    let last = segs.next().unwrap_or(path);
    let name = match (is_major(last), segs.next()) {
        (true, Some(prev)) => prev,
        _ => match last.rsplit_once('.') {
            Some((base, v)) if is_major(v) => base,
            _ => last,
        },
    };
    dotless(name)
}

/// A path segment with `.` replaced, for documented paths (`.` is the separator).
fn dotless(s: &str) -> String {
    s.replace('.', "-")
}

/// `//go:build ignore` (or `// +build ignore`) before the package clause.
fn build_ignored(src: &str) -> bool {
    for l in src.lines() {
        let t = l.trim();
        if t.starts_with("package ") {
            break;
        }
        let expr = t
            .strip_prefix("//go:build")
            .or_else(|| t.strip_prefix("// +build"));
        if let Some(e) = expr
            && e.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '!'))
                .any(|tok| tok == "ignore")
        {
            return true;
        }
    }
    false
}

/// A directory the go tool never builds as an importable package.
fn skipped_dir(rel: &str) -> bool {
    rel.split('/').any(|s| {
        matches!(s, "internal" | "testdata" | "vendor") || s.starts_with('.') || s.starts_with('_')
    })
}

const GOOS: &[&str] = &[
    "aix",
    "android",
    "darwin",
    "dragonfly",
    "freebsd",
    "hurd",
    "illumos",
    "ios",
    "js",
    "nacl",
    "netbsd",
    "openbsd",
    "plan9",
    "solaris",
    "wasip1",
    "windows",
    "zos",
];
const GOARCH: &[&str] = &[
    "386", "arm", "arm64", "loong64", "mips", "mips64", "mips64le", "mipsle", "ppc64", "ppc64le",
    "riscv64", "s390x", "wasm",
];

/// Which file wins when build-tagged variants define the same name: untagged first,
/// then Linux/amd64, then other platforms.
fn platform_rank(path: &str) -> u8 {
    let stem = file_name(path).trim_end_matches(".go");
    let parts: Vec<&str> = stem.split('_').skip(1).collect();
    let tail: Vec<&str> = parts.iter().rev().take(2).copied().collect();
    if tail.iter().any(|p| GOOS.contains(p) || GOARCH.contains(p)) {
        2
    } else if tail
        .iter()
        .any(|p| matches!(*p, "linux" | "amd64" | "unix"))
    {
        1
    } else {
        0
    }
}

/// One Go package while it is being assembled.
struct Pkg<'a> {
    rel: String,
    name: String,
    files: Vec<&'a GoFile<'a>>,
}

fn resolve<'s>(
    mods: &[(String, &str)],
    files: &[GoFile],
    tests: &[String],
    read: &dyn Fn(&str) -> Option<&'s str>,
) -> Surface {
    // (root dir, go.mod path, module path), deepest root first so the nearest wins.
    let mut roots: Vec<(String, String, String)> = mods
        .iter()
        .filter(|(p, _)| !skipped_dir(parent_dir(p)))
        .filter_map(|(p, text)| Some((parent_dir(p).to_string(), p.clone(), module_path(text)?)))
        .collect();
    roots.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.0.cmp(&b.0)));
    let owner = |dir: &str| -> Option<(usize, String)> {
        roots.iter().enumerate().find_map(|(i, (root, _, _))| {
            if root.is_empty() {
                Some((i, dir.to_string()))
            } else if dir == root {
                Some((i, String::new()))
            } else {
                dir.strip_prefix(root.as_str())
                    .and_then(|r| r.strip_prefix('/'))
                    .map(|r| (i, r.to_string()))
            }
        })
    };

    // (module, dir relative to its root) → files.
    let mut dirs: BTreeMap<(usize, String), Vec<&GoFile>> = BTreeMap::new();
    for f in files.iter().filter(|f| !f.ignored) {
        let Some((m, rel)) = owner(parent_dir(&f.path)) else {
            continue;
        };
        if skipped_dir(&rel) {
            continue;
        }
        dirs.entry((m, rel)).or_default().push(f);
    }
    let mut tests_by_dir: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for t in tests {
        tests_by_dir.entry(parent_dir(t)).or_default().push(t);
    }

    let mut by_module: BTreeMap<usize, Vec<Pkg>> = BTreeMap::new();
    for ((m, rel), fs) in dirs {
        // The package name most files declare (a stray `package documentation` loses).
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for f in &fs {
            if let Some(p) = f.api.package.as_deref() {
                *counts.entry(p).or_default() += 1;
            }
        }
        let Some(name) = counts
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
            .map(|(n, _)| n.to_string())
        else {
            continue;
        };
        if name == "main" {
            continue;
        }
        let files = fs
            .into_iter()
            .filter(|f| f.api.package.as_deref() == Some(name.as_str()))
            .collect();
        by_module
            .entry(m)
            .or_default()
            .push(Pkg { rel, name, files });
    }

    let mut packages = Vec::new();
    for (m, mut pkgs) in by_module {
        let (root_dir, manifest, path) = &roots[m];
        // Tree order: the root package, then directories segment by segment.
        pkgs.sort_by_cached_key(|p| {
            let segs: Vec<String> = p.rel.split('/').map(str::to_string).collect();
            (!p.rel.is_empty(), segs)
        });
        let local: BTreeSet<String> = pkgs.iter().map(|p| p.name.clone()).collect();
        let name = module_name(path);
        let mut k = Package {
            name: name.clone(),
            lang: Language::Go,
            sep: ".",
            package: path.clone(),
            manifest: manifest.clone(),
            is_lib: true,
            root_file: manifest.clone(),
            modules: Vec::new(),
        };
        if pkgs.first().is_none_or(|p| !p.rel.is_empty()) {
            // No package at the module root: a synthetic root lists the packages.
            k.modules.push(SurfaceModule {
                path: name.clone(),
                name: name.clone(),
                file: manifest.clone(),
                public: true,
                doc: Some(synthetic_doc(&format!(
                    "Import path prefix: `{path}`. The module has no package at its root."
                ))),
                items: Vec::new(),
                children: Vec::new(),
                parent: None,
            });
        }
        let mut rel_index: BTreeMap<String, usize> = BTreeMap::new();
        for p in &pkgs {
            let import = if p.rel.is_empty() {
                path.clone()
            } else {
                format!("{path}/{}", p.rel)
            };
            let module_path = if p.rel.is_empty() {
                name.clone()
            } else {
                format!("{name}/{}", dotless(&p.rel))
            };
            let parent = if p.rel.is_empty() {
                None
            } else {
                // The nearest ancestor directory that is a package, else the root.
                let mut up = p.rel.as_str();
                let mut found = Some(0);
                while let Some((d, _)) = up.rsplit_once('/') {
                    if let Some(&i) = rel_index.get(d) {
                        found = Some(i);
                        break;
                    }
                    up = d;
                }
                found
            };
            let idx = k.modules.len();
            let mut m = package_module(p, module_path, &import, parent, &local);
            if let Some(dir_tests) = tests_by_dir.get(join_dir(root_dir, &p.rel).as_str()) {
                attach_examples(&mut m, dir_tests, read);
            }
            k.modules.push(m);
            if let Some(par) = parent {
                k.modules[par].children.push(idx);
            }
            rel_index.insert(p.rel.clone(), idx);
        }
        if let Some(root) = pkgs.first().filter(|p| p.rel.is_empty()) {
            k.root_file = root
                .files
                .first()
                .map(|f| f.path.clone())
                .unwrap_or(k.root_file);
        }
        packages.push(k);
    }
    packages.sort_by(|a, b| (&a.name, &a.package).cmp(&(&b.name, &b.package)));
    Surface { packages }
}

fn join_dir(root: &str, rel: &str) -> String {
    match (root.is_empty(), rel.is_empty()) {
        (true, _) => rel.to_string(),
        (false, true) => root.to_string(),
        (false, false) => format!("{root}/{rel}"),
    }
}

fn synthetic_doc(markdown: &str) -> DocComment {
    DocComment {
        markdown: markdown.to_string(),
        summary: String::new(),
        sections: Vec::new(),
        examples: Vec::new(),
        links: Vec::new(),
        start_line: 1,
        end_line: 1,
    }
}

/// The surface module of one Go package: doc, items, methods attached to their types.
fn package_module(
    p: &Pkg,
    module_path: String,
    import: &str,
    parent: Option<usize>,
    local: &BTreeSet<String>,
) -> SurfaceModule {
    let mut files: Vec<&GoFile> = p.files.clone();
    files.sort_by(|a, b| (platform_rank(&a.path), &a.path).cmp(&(platform_rank(&b.path), &b.path)));

    // The package doc: `doc.go` first, then the first file that has one.
    let doc_file = files
        .iter()
        .filter(|f| f.api.module_doc.is_some())
        .min_by_key(|f| (file_name(&f.path) != "doc.go", f.path.as_str()))
        .copied();
    let import_line = format!("Import path: `{import}`");
    let doc = match doc_file.and_then(|f| f.api.module_doc.clone()) {
        Some(mut d) => {
            d.markdown = format!("{import_line}\n\n{}", d.markdown);
            degrade_links(&mut d, local);
            Some(d)
        }
        None => Some(synthetic_doc(&import_line)),
    };
    let file = doc_file
        .or(files.first().copied())
        .map(|f| f.path.clone())
        .unwrap_or_default();

    let mut items: Vec<SurfaceItem> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut methods: Vec<(&ApiItem, &str)> = Vec::new();
    for f in &files {
        for it in &f.api.items {
            if it.kind == ApiKind::Method && it.self_type.is_some() {
                methods.push((it, &f.path));
                continue;
            }
            if it.name == "_" || !seen.insert(it.name.clone()) {
                continue;
            }
            let mut item = it.clone();
            degrade_item_links(&mut item, local);
            items.push(SurfaceItem {
                path: format!("{module_path}.{}", it.name),
                public: is_exported(&it.name),
                item,
                file: f.path.clone(),
                defined_at: None,
                impl_files: BTreeMap::new(),
            });
        }
    }
    let types: BTreeMap<String, usize> = items
        .iter()
        .enumerate()
        // Any named type takes methods: `type Kind int` has `String()`.
        .filter(|(_, it)| it.item.kind.is_type() || it.item.kind == ApiKind::TypeAlias)
        .map(|(i, it)| (it.item.name.clone(), i))
        .collect();
    for (m, file) in methods {
        let Some(&ti) = m.self_type.as_ref().and_then(|t| types.get(t)) else {
            continue;
        };
        let target = &mut items[ti];
        if target
            .item
            .members
            .iter()
            .any(|x| x.kind == ApiKind::Method && x.name == m.name)
        {
            continue;
        }
        let mut method = m.clone();
        degrade_item_links(&mut method, local);
        target
            .impl_files
            .insert(target.item.members.len(), file.to_string());
        target.item.members.push(method);
    }

    SurfaceModule {
        path: module_path,
        name: p.name.clone(),
        file,
        public: true,
        doc,
        items,
        children: Vec::new(),
        parent,
    }
}

static QUALIFIED_LINK_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"\[`([a-z_][\w]*)\.([\w.]+)`\]").expect("valid regex"));

/// A doc link into a package outside this module (`[io.Reader]`) cannot resolve here;
/// render it as code so it never links to a local item of the same name.
fn degrade_links(d: &mut DocComment, local: &BTreeSet<String>) {
    if !d.markdown.contains("[`") {
        return;
    }
    let mut in_fence = false;
    let mut out = Vec::new();
    for line in d.markdown.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
        }
        if in_fence {
            out.push(line.to_string());
            continue;
        }
        out.push(
            QUALIFIED_LINK_RE
                .replace_all(line, |c: &regex::Captures| {
                    if local.contains(&c[1]) {
                        c[0].to_string()
                    } else {
                        format!("`{}.{}`", &c[1], &c[2])
                    }
                })
                .into_owned(),
        );
    }
    d.markdown = out.join("\n");
    d.summary = QUALIFIED_LINK_RE
        .replace_all(&d.summary, |c: &regex::Captures| {
            if local.contains(&c[1]) {
                c[0].to_string()
            } else {
                format!("`{}.{}`", &c[1], &c[2])
            }
        })
        .into_owned();
    d.links.retain(|l| match l.split_once('.') {
        Some((q, _)) if q.starts_with(|c: char| c.is_lowercase()) => local.contains(q),
        _ => true,
    });
}

fn degrade_item_links(it: &mut ApiItem, local: &BTreeSet<String>) {
    if let Some(d) = &mut it.doc {
        degrade_links(d, local);
    }
    for m in &mut it.members {
        degrade_item_links(m, local);
    }
}

/// Attach `Example…` functions from the directory's test files.
fn attach_examples<'s>(
    m: &mut SurfaceModule,
    tests: &[&str],
    read: &dyn Fn(&str) -> Option<&'s str>,
) {
    for path in tests {
        let Some(src) = read(path) else {
            continue;
        };
        if !src.contains("func Example") {
            continue;
        }
        let Ok(examples) = goapi::examples(src) else {
            continue;
        };
        for ex in examples {
            let (target, method, suffix) = split_example_name(&ex.name);
            let doc = if target.is_empty() {
                m.doc.get_or_insert_with(|| synthetic_doc(""))
            } else {
                let Some(it) = m.items.iter_mut().find(|i| i.item.name == target) else {
                    continue;
                };
                let item = match method {
                    Some(meth) => {
                        match it
                            .item
                            .members
                            .iter_mut()
                            .find(|x| x.kind == ApiKind::Method && x.name == meth)
                        {
                            Some(x) => x,
                            None => continue,
                        }
                    }
                    None => &mut it.item,
                };
                let line = item.start_line;
                item.doc.get_or_insert_with(|| DocComment {
                    start_line: line,
                    end_line: line,
                    ..synthetic_doc("")
                })
            };
            add_example(doc, &ex, suffix);
        }
    }
}

/// `T_M_suffix` → (`T`, Some(`M`), `suffix`); `F_suffix` → (`F`, None, `suffix`).
fn split_example_name(name: &str) -> (&str, Option<&str>, String) {
    let mut parts = name.split('_');
    let target = parts.next().unwrap_or("");
    let rest: Vec<&str> = parts.collect();
    match rest.first() {
        Some(m) if m.starts_with(|c: char| c.is_uppercase()) => {
            (target, Some(m), rest[1..].join("_"))
        }
        _ => (target, None, rest.join("_")),
    }
}

fn add_example(d: &mut DocComment, ex: &goapi::Example, suffix: String) {
    let mut md = d.markdown.trim_end().to_string();
    if !d.sections.iter().any(|(h, _)| h == "examples") {
        if !md.is_empty() {
            md.push_str("\n\n");
        }
        md.push_str("# Examples");
    }
    md.push_str("\n\n");
    if !suffix.is_empty() {
        md.push_str(&format!("Example ({}):\n\n", suffix.replace('_', " ")));
    }
    if let Some(text) = &ex.doc {
        md.push_str(text);
        md.push_str("\n\n");
    }
    md.push_str(&format!("```go\n{}\n```", ex.code));
    d.markdown = md;
    d.sections = doc::sections(&d.markdown);
    d.examples.push(CodeExample {
        lang: "go".into(),
        code: ex.code.clone(),
        doctest: false,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> BTreeMap<&'static str, &'static str> {
        let mut f = BTreeMap::new();
        f.insert("go.mod", "module github.com/acme/widget/v2\n\ngo 1.22\n");
        f.insert(
            "doc.go",
            "// Package widget builds widgets.\n//\n// Start with [New]; see [io.Reader] and [sub.Part].\npackage widget\n",
        );
        f.insert(
            "widget.go",
            "// Other file doc, not preferred.\npackage widget\n\n// Widget is a widget.\ntype Widget struct{ Name string }\n\n// New makes one.\nfunc New() *Widget { return nil }\n\ntype hidden struct{}\n\n// Exported method on an unexported type.\nfunc (h hidden) Visible() {}\n",
        );
        f.insert(
            "widget_linux.go",
            "package widget\n\n// Size on Linux.\nfunc (w *Widget) Size() int { return 1 }\n",
        );
        f.insert(
            "widget_windows.go",
            "package widget\n\n// Size on Windows.\nfunc (w *Widget) Size() int { return 2 }\n",
        );
        f.insert(
            "gen.go",
            "//go:build ignore\n\npackage main\n\nfunc main() {}\n",
        );
        f.insert(
            "widget_test.go",
            "package widget_test\n\nfunc ExampleWidget_Size() {\n\tfmt.Println(widget.New().Size())\n\t// Output: 1\n}\n\nfunc ExampleNew() {\n\t_ = widget.New()\n}\n\nfunc Example() {}\n",
        );
        f.insert(
            "sub/part.go",
            "// Package sub has parts.\npackage sub\n\n// Part is a part.\ntype Part int\n\n// String names it.\nfunc (p Part) String() string { return \"\" }\n",
        );
        f.insert("sub/deeper/d.go", "package deeper\n\nfunc D() {}\n");
        f.insert("internal/secret/s.go", "package secret\n\nfunc S() {}\n");
        f.insert("cmd/tool/main.go", "package main\n\nfunc main() {}\n");
        f.insert("testdata/x.go", "package x\n\nfunc X() {}\n");
        f.insert("_tools/t.go", "package tools\n\nfunc T() {}\n");
        f.insert("nested/go.mod", "module example.com/nested\n");
        f.insert("nested/n.go", "package nested\n\nfunc N() {}\n");
        f
    }

    fn resolve_fixture() -> Surface {
        let f = fixture();
        let apis: BTreeMap<&str, ApiFile> = f
            .iter()
            .filter(|(p, _)| p.ends_with(".go") && !p.ends_with("_test.go"))
            .map(|(p, s)| (*p, goapi::extract(s).unwrap()))
            .collect();
        let files: Vec<GoFile> = apis
            .iter()
            .map(|(p, a)| GoFile {
                path: p.to_string(),
                api: a,
                ignored: build_ignored(f[p]),
            })
            .collect();
        let mods: Vec<(String, &str)> = f
            .iter()
            .filter(|(p, _)| p.ends_with("go.mod"))
            .map(|(p, s)| (p.to_string(), *s))
            .collect();
        let tests: Vec<String> = f
            .keys()
            .filter(|p| p.ends_with("_test.go"))
            .map(|p| p.to_string())
            .collect();
        let read = |p: &str| f.get(p).copied();
        resolve(&mods, &files, &tests, &read)
    }

    fn item<'a>(k: &'a Package, path: &str) -> &'a SurfaceItem {
        k.modules
            .iter()
            .flat_map(|m| &m.items)
            .find(|i| i.path == path)
            .unwrap_or_else(|| panic!("{path} not found"))
    }

    #[test]
    fn modules_packages_and_exclusions() {
        let s = resolve_fixture();
        let names: Vec<(&str, &str)> = s
            .packages
            .iter()
            .map(|p| (p.name.as_str(), p.package.as_str()))
            .collect();
        assert_eq!(
            names,
            vec![
                ("nested", "example.com/nested"),
                ("widget", "github.com/acme/widget/v2")
            ]
        );
        let w = &s.packages[1];
        assert_eq!(w.sep, ".");
        assert_eq!(w.lang, Language::Go);
        let mods: Vec<(&str, &str, Option<usize>)> = w
            .modules
            .iter()
            .map(|m| (m.path.as_str(), m.name.as_str(), m.parent))
            .collect();
        assert_eq!(
            mods,
            vec![
                ("widget", "widget", None),
                ("widget/sub", "sub", Some(0)),
                ("widget/sub/deeper", "deeper", Some(1)),
            ],
            "internal/, cmd/ (package main), testdata/, _tools/ and nested/ are not packages of this module"
        );
        assert_eq!(w.modules[0].children, vec![1]);
        let doc = w.modules[0].doc.as_ref().unwrap();
        assert!(
            doc.markdown.starts_with(
                "Import path: `github.com/acme/widget/v2`\n\nPackage widget builds widgets."
            ),
            "{}",
            doc.markdown
        );
        assert_eq!(w.modules[0].file, "doc.go", "doc.go is preferred");
        assert!(
            w.modules[1]
                .doc
                .as_ref()
                .unwrap()
                .markdown
                .contains("`github.com/acme/widget/v2/sub`")
        );
        // `[io.Reader]` is outside the module: code, not a link. `[sub.Part]` stays.
        assert!(
            doc.markdown.contains("see `io.Reader` and [`sub.Part`]"),
            "{}",
            doc.markdown
        );
        assert_eq!(doc.links, vec!["New", "sub.Part"]);
    }

    #[test]
    fn items_methods_and_platform_variants() {
        let s = resolve_fixture();
        let w = &s.packages[1];
        let widget = item(w, "widget.Widget");
        assert!(widget.public);
        let methods: Vec<&str> = widget
            .item
            .members
            .iter()
            .filter(|m| m.kind == ApiKind::Method)
            .map(|m| m.name.as_str())
            .collect();
        assert_eq!(methods, vec!["Size"], "one Size, from the Linux file");
        let size = widget
            .item
            .members
            .iter()
            .find(|m| m.name == "Size")
            .unwrap();
        assert!(
            size.doc
                .as_ref()
                .unwrap()
                .markdown
                .starts_with("Size on Linux.")
        );
        assert_eq!(
            widget.impl_files.values().next().map(String::as_str),
            Some("widget_linux.go")
        );
        assert!(item(w, "widget.New").public);
        let hidden = item(w, "widget.hidden");
        assert!(!hidden.public);
        assert_eq!(hidden.item.members[0].name, "Visible");

        let part = item(w, "widget/sub.Part");
        assert_eq!(part.item.kind, ApiKind::TypeAlias);
        assert_eq!(part.item.members[0].name, "String");
    }

    #[test]
    fn testable_examples_attach() {
        let s = resolve_fixture();
        let w = &s.packages[1];
        let widget = item(w, "widget.Widget");
        let size = widget
            .item
            .members
            .iter()
            .find(|m| m.name == "Size")
            .unwrap();
        let d = size.doc.as_ref().unwrap();
        assert!(
            d.markdown.contains(
                "# Examples\n\n```go\nfmt.Println(widget.New().Size())\n// Output: 1\n```"
            ),
            "{}",
            d.markdown
        );
        assert_eq!(d.examples.len(), 1);
        assert_eq!(
            item(w, "widget.New")
                .item
                .doc
                .as_ref()
                .unwrap()
                .examples
                .len(),
            1
        );
        assert_eq!(w.modules[0].doc.as_ref().unwrap().examples.len(), 1);
    }

    #[test]
    fn helpers() {
        assert_eq!(module_name("github.com/acme/widget/v2"), "widget");
        assert_eq!(module_name("k8s.io/client-go"), "client-go");
        assert_eq!(module_name("gopkg.in/yaml.v3"), "yaml");
        assert_eq!(module_name("example.com/x.y"), "x-y");
        assert_eq!(
            module_path("// c\nmodule \"example.com/x\" // trailing\n").as_deref(),
            Some("example.com/x")
        );
        assert_eq!(module_path("modules foo\n"), None);
        assert!(build_ignored("// +build ignore\n\npackage main\n"));
        assert!(!build_ignored("//go:build linux\n\npackage x\n// ignore\n"));
        assert_eq!(platform_rank("a/b.go"), 0);
        assert_eq!(platform_rank("a/b_linux.go"), 1);
        assert_eq!(platform_rank("a/b_windows_amd64.go"), 2);
        assert_eq!(split_example_name("T_M_x"), ("T", Some("M"), "x".into()));
        assert_eq!(split_example_name("F_second"), ("F", None, "second".into()));
        assert!(skipped_dir("a/internal/b"));
        assert!(!skipped_dir("a/internals"));
    }

    /// Resolve the Go surface of `$REFLEX_GO_TREE` (default: a Kubernetes checkout),
    /// reading files directly (read-only, no index). Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn resolve_go_tree() {
        use crate::pulse::extract::roles;
        let root = std::env::var("REFLEX_GO_TREE").unwrap_or_else(|_| {
            format!(
                "{}/Code/misc/test/kubernetes",
                std::env::var("HOME").unwrap_or_default()
            )
        });
        let root = std::path::PathBuf::from(root);
        let mut paths = Vec::new();
        let mut stack = vec![root.clone()];
        while let Some(dir) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&dir) else {
                continue;
            };
            for e in rd.flatten() {
                let p = e.path();
                let name = e.file_name().to_string_lossy().to_string();
                let Ok(ft) = e.file_type() else { continue };
                if ft.is_dir() {
                    if !name.starts_with('.') && name != "vendor" && name != "_output" {
                        stack.push(p);
                    }
                } else if ft.is_file() && (name.ends_with(".go") || name == "go.mod") {
                    let rel = p.strip_prefix(&root).unwrap().to_string_lossy().to_string();
                    paths.push(rel);
                }
            }
        }
        paths.sort();
        let sources: BTreeMap<String, String> = paths
            .iter()
            .filter_map(|p| Some((p.clone(), std::fs::read_to_string(root.join(p)).ok()?)))
            .collect();
        let start = std::time::Instant::now();
        let apis: Vec<(String, ApiFile)> = sources
            .iter()
            .filter(|(p, _)| {
                p.ends_with(".go")
                    && !p.ends_with("_test.go")
                    && roles::classify(p) == FileRole::Source
            })
            .filter_map(|(p, s)| Some((p.clone(), goapi::extract(s).ok()?)))
            .collect();
        let extracted = start.elapsed();
        let files: Vec<GoFile> = apis
            .iter()
            .map(|(p, a)| GoFile {
                path: p.clone(),
                api: a,
                ignored: build_ignored(&sources[p]),
            })
            .collect();
        let mods: Vec<(String, &str)> = sources
            .iter()
            .filter(|(p, _)| p.ends_with("go.mod"))
            .map(|(p, s)| (p.clone(), s.as_str()))
            .collect();
        let tests: Vec<String> = sources
            .keys()
            .filter(|p| p.ends_with("_test.go"))
            .cloned()
            .collect();
        let read = |p: &str| sources.get(p).map(String::as_str);
        let t = std::time::Instant::now();
        let s = resolve(&mods, &files, &tests, &read);
        let resolved = t.elapsed();
        let (mut pkgs, mut items, mut public, mut members, mut examples) = (0, 0, 0, 0, 0);
        for k in &s.packages {
            pkgs += k.modules.len();
            for m in &k.modules {
                for it in &m.items {
                    items += 1;
                    if it.public {
                        public += 1;
                        members += it
                            .item
                            .members
                            .iter()
                            .filter(|x| x.visibility.is_public())
                            .count();
                        examples +=
                            it.item.doc.as_ref().map_or(0, |d| {
                                d.examples.iter().filter(|e| e.lang == "go").count()
                            });
                    }
                }
            }
        }
        println!(
            "{} source files extracted in {extracted:?}; {} modules, {pkgs} packages, \
             {items} items ({public} public, {members} public members, {examples} go code blocks) \
             resolved in {resolved:?}",
            files.len(),
            s.packages.len(),
        );
    }
}
