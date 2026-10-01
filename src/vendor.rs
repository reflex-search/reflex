//! Vendored code: third-party code committed into the repository.
//!
//! Vendored files are indexed and searchable (ripgrep searches them too), but they
//! are not nodes of the import graph: a repository that commits `vendor/` gets the
//! graph a repository that gitignores it gets. Imports of vendored code stay
//! External. See `.context/DEPENDENCY_RESOLUTION_RESEARCH.md` ("Vendored code").
//!
//! A file is vendored when, in this order:
//! 1. `[index.vendored] patterns` (gitignore rules) match it: a pattern makes it
//!    vendored, a `!pattern` makes it project code;
//! 2. it is under a root a toolchain marker names (Go `vendor/modules.txt`,
//!    Composer `vendor/composer/installed.json`, a `cargo vendor` crate's
//!    `.cargo-checksum.json`, a virtualenv's `pyvenv.cfg`, a `build.zig.zon` path
//!    dependency);
//! 3. a directory of its path is `node_modules`, `bower_components`,
//!    `site-packages`, `dist-packages` or an installed-gems directory
//!    (`ruby/<version>/gems`);
//! 4. a directory of its path has a conventional third-party name for its language
//!    ([`vendor_dir_names`]). Go, PHP, Rust and Ruby have none: their toolchains
//!    define vendoring, so a Go `third_party/forked/` the module imports by its own
//!    path stays project code.

use std::path::Path;

use ignore::Match;
use ignore::gitignore::{Gitignore, GitignoreBuilder};

use crate::models::Language;

/// Directory names that hold every language's dependencies wherever they appear.
const DEPENDENCY_DIRS: [&str; 4] = [
    "node_modules",
    "bower_components",
    "site-packages",
    "dist-packages",
];

/// Conventional third-party directory names, for the languages whose toolchain
/// has no vendoring marker. Java and Kotlin keep to `third_party` spellings: a
/// package named `external` or `vendor` is project code.
pub fn vendor_dir_names(language: Language) -> &'static [&'static str] {
    const THIRD_PARTY: [&str; 3] = ["third_party", "third-party", "thirdparty"];
    match language {
        Language::C | Language::Cpp => &[
            "third_party",
            "third-party",
            "thirdparty",
            "vendor",
            "external",
            "extern",
            "deps",
        ],
        Language::JavaScript | Language::TypeScript | Language::Vue | Language::Svelte => {
            &["third_party", "third-party", "thirdparty", "vendor"]
        }
        Language::Python => &[
            "third_party",
            "third-party",
            "thirdparty",
            "_vendor",
            "_vendored",
        ],
        Language::Java | Language::Kotlin | Language::CSharp | Language::Zig => &THIRD_PARTY,
        _ => &[],
    }
}

/// The vendored root a marker file names, relative and `/`-terminated, or `None`
/// when `rel` is not a marker. `rel` is relative to the root, `/`-separated.
pub fn marker_root(rel: &str) -> Option<String> {
    let (dir, name) = match rel.rsplit_once('/') {
        Some((dir, name)) => (dir, name),
        None => ("", rel),
    };
    let under = |d: &str| {
        if d.is_empty() {
            String::new()
        } else {
            format!("{d}/")
        }
    };
    match name {
        // Go: `go mod vendor` writes vendor/modules.txt
        "modules.txt" if dir == "vendor" || dir.ends_with("/vendor") => Some(under(dir)),
        // Composer: vendor/composer/installed.json
        "installed.json" => {
            let vendor = dir
                .strip_suffix("composer")?
                .strip_suffix('/')
                .unwrap_or("");
            (vendor == "vendor" || vendor.ends_with("/vendor")).then(|| under(vendor))
        }
        // `cargo vendor`: every crate directory holds a .cargo-checksum.json
        ".cargo-checksum.json" | "pyvenv.cfg" if !dir.is_empty() => Some(under(dir)),
        _ => None,
    }
}

/// The path dependencies of a `build.zig.zon` at `rel` (`.path = "deps/x"`), as
/// `/`-terminated roots relative to the workspace root.
pub fn zig_path_dependencies(rel: &str, source: &str) -> Vec<String> {
    let dir = rel.rsplit_once('/').map_or("", |(d, _)| d);
    let mut roots = Vec::new();
    let mut rest = source;
    while let Some(at) = rest.find(".path") {
        rest = &rest[at + ".path".len()..];
        let Some(value) = rest.trim_start().strip_prefix('=') else {
            continue;
        };
        let Some(value) = value.trim_start().strip_prefix('"') else {
            continue;
        };
        let Some(end) = value.find('"') else { break };
        if let Some(root) = crate::dependency_resolve::fold_path(dir, &value[..end]) {
            roots.push(format!("{root}/"));
        }
    }
    roots
}

/// The directories of a RubyGems install (`bundle install --path vendor/bundle`
/// writes `vendor/bundle/ruby/<version>/{gems,specifications,...}`).
const GEM_INSTALL_DIRS: [&str; 7] = [
    "gems",
    "specifications",
    "extensions",
    "cache",
    "build_info",
    "doc",
    "bundler",
];

/// Whether a directory of `rel` (not its file name) holds every language's
/// dependencies: `node_modules`, `site-packages`, installed gems.
fn in_dependency_dir(dirs: &[&str]) -> bool {
    dirs.iter().any(|d| DEPENDENCY_DIRS.contains(d))
        || dirs.windows(3).any(|w| {
            w[0] == "ruby"
                && w[1].starts_with(|c: char| c.is_ascii_digit())
                && GEM_INSTALL_DIRS.contains(&w[2])
        })
}

/// The rules that decide which files are vendored (see the module docs).
#[derive(Debug, Default, Clone)]
pub struct VendorRules {
    /// Marker roots, relative and `/`-terminated.
    roots: Vec<String>,
    /// `[index.vendored] patterns`.
    overrides: Option<Gitignore>,
}

impl VendorRules {
    /// Rules over the marker `roots` and the user's `patterns` (gitignore rules,
    /// relative to `root`). An invalid pattern is logged and skipped.
    pub fn new(root: &Path, mut roots: Vec<String>, patterns: &[String]) -> Self {
        roots.sort();
        roots.dedup();
        let overrides = (!patterns.is_empty()).then(|| {
            let mut builder = GitignoreBuilder::new(root);
            for pattern in patterns {
                if let Err(e) = builder.add_line(None, pattern) {
                    log::warn!("[index.vendored] pattern '{}' is invalid: {}", pattern, e);
                }
            }
            builder.build().unwrap_or_else(|e| {
                log::warn!("[index.vendored] patterns are invalid: {}", e);
                Gitignore::empty()
            })
        });
        Self { roots, overrides }
    }

    /// Whether `rel` is under a marker root or a dependency directory: the rules
    /// that hold whatever the file's language. Configs under such a directory are
    /// not the workspace's configs.
    pub fn in_vendored_dir(&self, rel: &str) -> bool {
        if self.roots.iter().any(|r| rel.starts_with(r.as_str())) {
            return true;
        }
        let mut dirs: Vec<&str> = rel.split('/').collect();
        dirs.pop();
        in_dependency_dir(&dirs)
    }

    /// Whether the file at `rel` (relative, `/`-separated) is vendored.
    pub fn is_vendored(&self, rel: &str, language: Language) -> bool {
        if let Some(overrides) = &self.overrides {
            match overrides.matched_path_or_any_parents(rel, false) {
                Match::Ignore(_) => return true,
                Match::Whitelist(_) => return false,
                Match::None => {}
            }
        }
        if self.in_vendored_dir(rel) {
            return true;
        }
        let names = vendor_dir_names(language);
        let mut dirs: Vec<&str> = rel.split('/').collect();
        dirs.pop();
        dirs.iter().any(|d| names.contains(d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules(roots: &[&str], patterns: &[&str]) -> VendorRules {
        VendorRules::new(
            Path::new("/ws"),
            roots.iter().map(|r| r.to_string()).collect(),
            &patterns.iter().map(|p| p.to_string()).collect::<Vec<_>>(),
        )
    }

    #[test]
    fn marker_roots() {
        assert_eq!(
            marker_root("vendor/modules.txt").as_deref(),
            Some("vendor/")
        );
        assert_eq!(
            marker_root("svc/vendor/modules.txt").as_deref(),
            Some("svc/vendor/")
        );
        assert_eq!(marker_root("docs/modules.txt"), None);
        assert_eq!(
            marker_root("vendor/composer/installed.json").as_deref(),
            Some("vendor/")
        );
        assert_eq!(marker_root("composer/installed.json"), None);
        assert_eq!(
            marker_root("third_party/rust/vendor/serde/.cargo-checksum.json").as_deref(),
            Some("third_party/rust/vendor/serde/")
        );
        assert_eq!(marker_root("venv/pyvenv.cfg").as_deref(), Some("venv/"));
        assert_eq!(marker_root("pyvenv.cfg"), None);
    }

    #[test]
    fn zig_path_dependencies_are_roots() {
        let zon = r#".{
            .name = .app,
            .paths = .{ "src", "build.zig" },
            .dependencies = .{
                .zlib = .{ .path = "deps/zlib" },
                .net = .{ .url = "https://x", .hash = "1220" },
                .up = .{ .path = "../shared" },
            },
        }"#;
        assert_eq!(
            zig_path_dependencies("app/build.zig.zon", zon),
            vec!["app/deps/zlib/".to_string(), "shared/".to_string()]
        );
    }

    #[test]
    fn each_language_has_its_own_names() {
        let r = rules(&[], &[]);
        assert!(r.is_vendored("src/native/external/zlib/inflate.c", Language::C));
        assert!(r.is_vendored("deps/x/a.h", Language::C));
        assert!(!r.is_vendored("third_party/forked/x/a.go", Language::Go));
        assert!(!r.is_vendored("vendor/x/a.go", Language::Go));
        assert!(!r.is_vendored("src/org/acme/external/A.java", Language::Java));
        assert!(r.is_vendored("third_party/guava/A.java", Language::Java));
        assert!(r.is_vendored("static/js/vendor/jquery.js", Language::JavaScript));
        assert!(r.is_vendored("pip/_vendor/requests/api.py", Language::Python));
        assert!(!r.is_vendored("app/vendor/x.php", Language::PHP));
    }

    #[test]
    fn dependency_dirs_hold_for_every_language() {
        let r = rules(&[], &[]);
        assert!(r.is_vendored("web/node_modules/lodash/fp.js", Language::JavaScript));
        assert!(r.is_vendored("node_modules/addon/src/a.c", Language::C));
        assert!(r.is_vendored(
            "venv/lib/python3.12/site-packages/requests/api.py",
            Language::Python
        ));
        assert!(r.is_vendored(
            "vendor/bundle/ruby/3.3.0/gems/rack-3.0.0/lib/rack.rb",
            Language::Ruby
        ));
        assert!(!r.is_vendored("lib/gems/x.rb", Language::Ruby));
    }

    #[test]
    fn marker_roots_hold_for_every_language() {
        let r = rules(&["vendor/"], &[]);
        assert!(r.is_vendored("vendor/golang.org/x/sys/unix/a.go", Language::Go));
        assert!(!r.is_vendored("vendorx/a.go", Language::Go));
    }

    #[test]
    fn patterns_come_first() {
        let r = rules(&["vendor/"], &["libs/acme/", "!vendor/ours/", "!src/deps/"]);
        assert!(r.is_vendored("libs/acme/a.go", Language::Go));
        assert!(!r.is_vendored("vendor/ours/a.go", Language::Go));
        assert!(r.is_vendored("vendor/theirs/a.go", Language::Go));
        assert!(!r.is_vendored("src/deps/a.c", Language::C));
    }
}
