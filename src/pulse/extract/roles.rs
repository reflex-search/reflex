//! File roles: what part a file plays in the project.
//!
//! Only [`FileRole::Source`] files produce modules and reference content. Tests,
//! fixtures, vendored code and build scripts are indexed and searchable, but a docs
//! site that lists `tests/corpus/*` as modules or `build.rs` as a subsystem is wrong.

use crate::models::Language;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FileRole {
    /// Code that ships.
    Source,
    Test,
    /// Test inputs: corpora, fixtures, snapshots, testdata.
    Fixture,
    Example,
    Bench,
    Generated,
    Vendor,
    /// Build scripts and build output.
    Build,
    /// Prose: README, docs/, *.md.
    Docs,
    /// Other text: config, data, manifests.
    Config,
    Lock,
}

impl FileRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            FileRole::Source => "source",
            FileRole::Test => "test",
            FileRole::Fixture => "fixture",
            FileRole::Example => "example",
            FileRole::Bench => "bench",
            FileRole::Generated => "generated",
            FileRole::Vendor => "vendor",
            FileRole::Build => "build",
            FileRole::Docs => "docs",
            FileRole::Config => "config",
            FileRole::Lock => "lock",
        }
    }
}

const VENDOR_DIRS: &[&str] = &[
    "vendor",
    "third_party",
    "third-party",
    "node_modules",
    "bower_components",
];
const FIXTURE_DIRS: &[&str] = &[
    "fixtures",
    "fixture",
    "__fixtures__",
    "testdata",
    "test-data",
    "test_data",
    "__snapshots__",
    "snapshots",
    "corpus",
    "golden",
];
const TEST_DIRS: &[&str] = &[
    "tests",
    "test",
    "__tests__",
    "spec",
    "specs",
    "e2e",
    "integration_tests",
];
const EXAMPLE_DIRS: &[&str] = &["examples", "example"];
const BENCH_DIRS: &[&str] = &["benches", "bench", "benchmarks", "benchmark"];
const BUILD_DIRS: &[&str] = &["target", "dist", "build", "out", ".next"];
const DOC_EXTS: &[&str] = &["md", "mdx", "markdown", "rst", "adoc", "txt"];

/// Classify a path relative to the index root.
pub fn classify(path: &str) -> FileRole {
    let lang = Language::from_path(Path::new(path));
    match lang {
        Language::Lock => return FileRole::Lock,
        Language::Generated => return FileRole::Generated,
        _ => {}
    }

    let lower = path.to_ascii_lowercase();
    let parts: Vec<&str> = lower.split('/').collect();
    let (dirs, name) = parts.split_at(parts.len() - 1);
    let name = name[0];
    let in_dir = |set: &[&str]| dirs.iter().any(|d| set.contains(d));

    if in_dir(VENDOR_DIRS) {
        return FileRole::Vendor;
    }
    let in_tests = in_dir(TEST_DIRS);
    // A corpus or fixture directory, or data nested under a test directory.
    if in_dir(FIXTURE_DIRS) || (in_tests && dirs.iter().any(|d| *d == "data" || *d == "samples")) {
        return FileRole::Fixture;
    }
    if in_tests || is_test_file_name(name) {
        return FileRole::Test;
    }
    if in_dir(BENCH_DIRS) {
        return FileRole::Bench;
    }
    if in_dir(EXAMPLE_DIRS) {
        return FileRole::Example;
    }
    // `build/`, `dist/`, `out/` are output only at the top of the tree (`src/build/` is
    // code); `target/` is Cargo output at any depth.
    let output_dir = dirs.first().is_some_and(|d| BUILD_DIRS.contains(d)) || in_dir(&["target"]);
    if output_dir || is_build_script(dirs, name) {
        return FileRole::Build;
    }
    if lang.is_code() {
        return FileRole::Source;
    }
    let ext = name.rsplit_once('.').map(|(_, e)| e).unwrap_or("");
    if DOC_EXTS.contains(&ext) || dirs.first() == Some(&"docs") || name.starts_with("readme") {
        FileRole::Docs
    } else {
        FileRole::Config
    }
}

fn is_test_file_name(name: &str) -> bool {
    let stem = name.split('.').next().unwrap_or(name);
    name.ends_with("_test.go")
        || name.ends_with("_test.py")
        || (name.starts_with("test_") && name.ends_with(".py"))
        || name == "conftest.py"
        || name.contains(".test.")
        || name.contains(".spec.")
        || stem.ends_with("test") && name.ends_with(".java")
        || stem.ends_with("tests") && (name.ends_with(".java") || name.ends_with(".cs"))
        || name.ends_with("_spec.rb")
}

fn is_build_script(dirs: &[&str], name: &str) -> bool {
    // Cargo build scripts, Python packaging entry points, JS bundler configs.
    // A `build.rs` inside `src/` is an ordinary module, not a Cargo build script.
    (name == "build.rs" && !dirs.contains(&"src"))
        || (dirs.is_empty() && (name == "setup.py" || name == "noxfile.py" || name == "fabfile.py"))
        || name.starts_with("webpack.config.")
        || name.starts_with("vite.config.")
        || name.starts_with("rollup.config.")
        || name.starts_with("babel.config.")
        || name.starts_with("jest.config.")
        || name.starts_with("vitest.config.")
        || name.starts_with("eslint.config.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles() {
        use FileRole::*;
        let cases = [
            ("src/main.rs", Source),
            ("src/pulse/site.rs", Source),
            ("lib/app.ts", Source),
            ("tests/corpus/rust/sample.rs", Fixture),
            ("tests/fixtures/a.json", Fixture),
            ("pkg/testdata/x.go", Fixture),
            ("tests/integration_test.rs", Test),
            ("src/foo_test.go", Test),
            ("pkg/test_utils.py", Test),
            ("web/app.test.tsx", Test),
            ("src/test/java/FooTest.java", Test),
            ("benches/search.rs", Bench),
            ("examples/basic.rs", Example),
            ("vendor/github.com/x/y.go", Vendor),
            ("web/node_modules/a/index.js", Vendor),
            ("build.rs", Build),
            ("crates/core/build.rs", Build),
            ("src/codegen/build.rs", Source),
            ("dist/bundle.js", Build),
            ("src/pulse/build/mod.rs", Source),
            ("crates/x/target/debug/gen.rs", Build),
            ("vite.config.ts", Build),
            ("README.md", Docs),
            ("docs/ARCHITECTURE.md", Docs),
            ("CHANGELOG.md", Docs),
            ("Cargo.toml", Config),
            (".github/workflows/ci.yml", Config),
            ("Cargo.lock", Lock),
            ("api.pb.go", Generated),
        ];
        for (path, want) in cases {
            assert_eq!(classify(path), want, "{path}");
        }
    }
}
