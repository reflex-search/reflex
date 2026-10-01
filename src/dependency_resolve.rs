//! Import resolution for the dependency pass.
//!
//! Two parts, both taken out of `Indexer::index_with_callback` unchanged:
//!
//! - [`ResolverConfigs`]: the project files that change how imports are classified
//!   and resolved (`go.mod`, Maven/Gradle builds, Python package configs, gemspecs,
//!   `Cargo.toml`, `composer.json`, `tsconfig.json`). They are found by ONE walk
//!   that applies each finder's own rules; the seven finders used to walk the tree
//!   once each.
//! - [`ResolverContext`]: the per-language rules that reclassify an extracted import
//!   and resolve it (or a re-export's source) to a `files.id`.
//!
//! Resolution depends on the whole set of indexed paths (through [`PathResolver`]'s
//! suffix matching) and on these configs, not only on the importing file.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::dependency::PathResolver;
use crate::models::{Dependency, ImportType};
use crate::parsers::go::GoModule;
use crate::parsers::java::JavaProject;
use crate::parsers::php::Psr4Mapping;
use crate::parsers::python::PythonPackage;
use crate::parsers::ruby::RubyProject;
use crate::parsers::rust::RustCrate;
use crate::parsers::tsconfig::PathAliasMap;
use crate::parsers::{ExportInfo, ImportInfo};

/// Find the nearest tsconfig.json for a given source file
///
/// Walks up the directory tree from the source file to find the nearest tsconfig directory.
/// Returns a reference to the PathAliasMap if found.
pub fn find_nearest_tsconfig<'a>(
    file_path: &str,
    root: &Path,
    tsconfigs: &'a HashMap<PathBuf, PathAliasMap>,
) -> Option<&'a PathAliasMap> {
    // Convert file_path to absolute path (relative to root)
    let abs_file_path = if Path::new(file_path).is_absolute() {
        PathBuf::from(file_path)
    } else {
        root.join(file_path)
    };

    // Start from the file's directory and walk up
    let mut current_dir = abs_file_path.parent()?;

    loop {
        // Check if we have a tsconfig for this directory
        if let Some(alias_map) = tsconfigs.get(current_dir) {
            return Some(alias_map);
        }

        // Move up one directory
        current_dir = current_dir.parent()?;

        // Stop if we've reached the root
        if current_dir == root || !current_dir.starts_with(root) {
            break;
        }
    }

    None
}

/// The file under `.reflex/` that lists the config files the last walk found, so
/// an update that does not walk can parse the same configs.
pub const CONFIG_LIST: &str = "resolver-configs.json";

/// Whether a file with this name is a resolver config (one the config walk keeps).
pub fn is_resolver_config_name(name: &str) -> bool {
    matches!(
        name,
        "go.mod"
            | "pom.xml"
            | "build.gradle"
            | "build.gradle.kts"
            | "pyproject.toml"
            | "setup.py"
            | "setup.cfg"
            | "composer.json"
            | "tsconfig.json"
            | "Cargo.toml"
    ) || name.ends_with(".gemspec")
}

/// [`ConfigFiles`] on disk: paths relative to the root, `/`-separated.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize, PartialEq)]
struct ConfigList {
    go_mods: Vec<String>,
    java: Vec<String>,
    python: Vec<String>,
    gemspecs: Vec<String>,
    cargo_tomls: Vec<String>,
    composer: Vec<String>,
    tsconfigs: Vec<String>,
    walk_error: Option<String>,
}

/// The config files one walk found, per kind, in walk order.
#[derive(Debug, Default, Clone)]
struct ConfigFiles {
    go_mods: Vec<PathBuf>,
    java: Vec<PathBuf>,
    python: Vec<PathBuf>,
    gemspecs: Vec<PathBuf>,
    cargo_tomls: Vec<PathBuf>,
    composer: Vec<PathBuf>,
    tsconfigs: Vec<PathBuf>,
    /// The first walk error. Every finder except tsconfig's stopped on its first
    /// error (`entry?`) and returned it, so their kinds fail with it here too.
    walk_error: Option<String>,
}

impl ConfigFiles {
    /// One walk with the settings all seven finders shared (a default `WalkBuilder`
    /// with `follow_links(false)`, `git_ignore(true)`), keeping each finder's own
    /// filter.
    fn walk(root: &Path) -> Self {
        let mut out = Self::default();
        let walker = ignore::WalkBuilder::new(root)
            .follow_links(false)
            .git_ignore(true)
            .build();
        for entry in walker {
            let entry = match entry {
                Ok(entry) => entry,
                Err(e) => {
                    if out.walk_error.is_none() {
                        out.walk_error = Some(e.to_string());
                    }
                    continue;
                }
            };
            let path = entry.path();
            let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let is_gemspec = path.extension().and_then(|s| s.to_str()) == Some("gemspec");
            match filename {
                "go.mod" | "pom.xml" | "build.gradle" | "build.gradle.kts" | "pyproject.toml"
                | "setup.py" | "setup.cfg" | "composer.json" | "tsconfig.json" | "Cargo.toml" => {}
                _ if is_gemspec => {}
                _ => continue,
            }

            // Rust's finder matched the name alone; tsconfig's likewise.
            if filename == "Cargo.toml" {
                out.cargo_tomls.push(path.to_path_buf());
            }
            if filename == "tsconfig.json" {
                out.tsconfigs.push(path.to_path_buf());
            }
            if !path.is_file() {
                continue;
            }
            // Normalize separators so the directory filters work on Windows too.
            let path_str = path.to_string_lossy().replace('\\', "/");
            match filename {
                "go.mod" => {
                    if !path_str.contains("/vendor/") {
                        out.go_mods.push(path.to_path_buf());
                    }
                }
                "pom.xml" | "build.gradle" | "build.gradle.kts" => {
                    out.java.push(path.to_path_buf())
                }
                "pyproject.toml" | "setup.py" | "setup.cfg" => {
                    let skipped = path_str.contains("/venv/")
                        || path_str.contains("/.venv/")
                        || path_str.contains("/site-packages/")
                        || path_str.contains("/dist/")
                        || path_str.contains("/build/")
                        || path_str.contains("/__pycache__/");
                    if !skipped {
                        out.python.push(path.to_path_buf());
                    }
                }
                "composer.json" if !path.components().any(|c| c.as_os_str() == "vendor") => {
                    out.composer.push(path.to_path_buf());
                }
                _ => {}
            }
            if is_gemspec {
                out.gemspecs.push(path.to_path_buf());
            }
        }
        out
    }

    /// A digest of everything the config parsers read: each kind's files in walk
    /// order with their bytes, the Python parser's sibling files, the root
    /// `Cargo.toml` gate and the walk error. Equal digests mean equal configs.
    fn digest(&self, root: &Path) -> String {
        let mut h = blake3::Hasher::new();
        let mut file = |tag: &str, path: &Path| {
            let rel = path.strip_prefix(root).unwrap_or(path);
            h.update(tag.as_bytes());
            h.update(rel.to_string_lossy().as_bytes());
            h.update(b"\0");
            match std::fs::read(path) {
                Ok(bytes) => {
                    h.update(blake3::hash(&bytes).as_bytes());
                }
                Err(_) => {
                    h.update(b"<absent>");
                }
            }
        };
        let kinds: [(&str, &Vec<PathBuf>); 7] = [
            ("go", &self.go_mods),
            ("java", &self.java),
            ("python", &self.python),
            ("gem", &self.gemspecs),
            ("cargo", &self.cargo_tomls),
            ("composer", &self.composer),
            ("tsconfig", &self.tsconfigs),
        ];
        for (tag, paths) in kinds {
            for path in paths {
                file(tag, path);
            }
        }
        // `find_python_package_name` reads all three names in each project root.
        for path in &self.python {
            if let Some(dir) = path.parent() {
                for name in ["pyproject.toml", "setup.py", "setup.cfg"] {
                    file("python-sibling", &dir.join(name));
                }
            }
        }
        h.update(if root.join("Cargo.toml").exists() {
            b"cargo-gate:1"
        } else {
            b"cargo-gate:0"
        });
        h.update(self.walk_error.as_deref().unwrap_or("").as_bytes());
        h.finalize().to_hex().to_string()
    }

    fn to_list(&self, root: &Path) -> ConfigList {
        let rel = |paths: &Vec<PathBuf>| -> Vec<String> {
            paths
                .iter()
                .map(|p| {
                    p.strip_prefix(root)
                        .unwrap_or(p)
                        .to_string_lossy()
                        .replace('\\', "/")
                })
                .collect()
        };
        ConfigList {
            go_mods: rel(&self.go_mods),
            java: rel(&self.java),
            python: rel(&self.python),
            gemspecs: rel(&self.gemspecs),
            cargo_tomls: rel(&self.cargo_tomls),
            composer: rel(&self.composer),
            tsconfigs: rel(&self.tsconfigs),
            walk_error: self.walk_error.clone(),
        }
    }

    fn from_list(root: &Path, list: &ConfigList) -> Self {
        let abs =
            |paths: &Vec<String>| -> Vec<PathBuf> { paths.iter().map(|p| root.join(p)).collect() };
        Self {
            go_mods: abs(&list.go_mods),
            java: abs(&list.java),
            python: abs(&list.python),
            gemspecs: abs(&list.gemspecs),
            cargo_tomls: abs(&list.cargo_tomls),
            composer: abs(&list.composer),
            tsconfigs: abs(&list.tsconfigs),
            walk_error: list.walk_error.clone(),
        }
    }

    fn check_walk(&self) -> anyhow::Result<()> {
        match &self.walk_error {
            Some(e) => Err(anyhow::anyhow!("{}", e)),
            None => Ok(()),
        }
    }
}

/// Every resolver config of a workspace, parsed.
#[derive(Debug, Default)]
pub struct ResolverConfigs {
    pub tsconfigs: HashMap<PathBuf, PathAliasMap>,
    pub go_modules: Vec<GoModule>,
    pub java_projects: Vec<JavaProject>,
    pub python_packages: Vec<PythonPackage>,
    pub ruby_projects: Vec<RubyProject>,
    pub rust_crates: Vec<RustCrate>,
    pub php_psr4: Vec<Psr4Mapping>,
    /// Digest of every input above (see `ConfigFiles::digest`). An index run
    /// whose digest differs from the stored one re-resolves every file.
    pub digest: String,
    /// The config files these were parsed from.
    files: ConfigFiles,
}

impl ResolverConfigs {
    /// Find and parse every resolver config under `root`. A kind that fails to parse
    /// is logged and left empty, as before.
    pub fn discover(root: &Path) -> Self {
        Self::parse(root, ConfigFiles::walk(root))
    }

    /// Parse the config files listed in `cache_dir` by [`Self::save_list`], without
    /// walking. `None` when there is no list. The caller compares the digest with
    /// the stored one: a listed file that changed shows there.
    pub fn from_saved_list(root: &Path, cache_dir: &Path) -> Option<Self> {
        let bytes = std::fs::read(cache_dir.join(CONFIG_LIST)).ok()?;
        let list: ConfigList = serde_json::from_slice(&bytes).ok()?;
        Some(Self::parse(root, ConfigFiles::from_list(root, &list)))
    }

    /// Write the list of config files (atomically) unless it is already there.
    pub fn save_list(&self, root: &Path, cache_dir: &Path) -> anyhow::Result<()> {
        let list = self.files.to_list(root);
        let path = cache_dir.join(CONFIG_LIST);
        if let Ok(bytes) = std::fs::read(&path)
            && serde_json::from_slice::<ConfigList>(&bytes).ok().as_ref() == Some(&list)
        {
            return Ok(());
        }
        let tmp = crate::atomic_write::tmp_path_for(&path);
        std::fs::write(&tmp, serde_json::to_vec_pretty(&list)?)?;
        crate::atomic_write::atomic_replace(&tmp, &path)?;
        Ok(())
    }

    /// Whether any of `rels` (relative, `/`-separated) is a listed config file or
    /// a directory above one.
    pub fn lists_any_under(&self, root: &Path, rels: &[String]) -> bool {
        let list = self.files.to_list(root);
        [
            &list.go_mods,
            &list.java,
            &list.python,
            &list.gemspecs,
            &list.cargo_tomls,
            &list.composer,
            &list.tsconfigs,
        ]
        .into_iter()
        .flatten()
        .any(|p| {
            rels.iter().any(|rel| {
                p == rel
                    || p.strip_prefix(rel.as_str())
                        .is_some_and(|r| r.starts_with('/'))
            })
        })
    }

    fn parse(root: &Path, files: ConfigFiles) -> Self {
        let digest = files.digest(root);

        let tsconfigs = crate::parsers::tsconfig::parse_tsconfigs_from(&files.tsconfigs)
            .unwrap_or_else(|e| {
                log::warn!("Failed to parse tsconfig.json files: {}", e);
                HashMap::new()
            });
        if !tsconfigs.is_empty() {
            log::info!("Found {} tsconfig.json files", tsconfigs.len());
            for (config_dir, alias_map) in &tsconfigs {
                log::debug!(
                    "  {} (base_url: {:?}, {} aliases)",
                    config_dir.display(),
                    alias_map.base_url,
                    alias_map.aliases.len()
                );
            }
        }

        let go_modules = files
            .check_walk()
            .and_then(|_| crate::parsers::go::parse_go_modules_from(root, &files.go_mods))
            .unwrap_or_else(|e| {
                log::warn!("Failed to parse go.mod files: {}", e);
                Vec::new()
            });
        if !go_modules.is_empty() {
            log::info!("Found {} Go modules", go_modules.len());
            for module in &go_modules {
                log::debug!("  {} (project: {})", module.name, module.project_root);
            }
        }

        let java_projects = files
            .check_walk()
            .and_then(|_| crate::parsers::java::parse_java_projects_from(root, &files.java))
            .unwrap_or_else(|e| {
                log::warn!("Failed to parse Java project configs: {}", e);
                Vec::new()
            });
        if !java_projects.is_empty() {
            log::info!("Found {} Java projects", java_projects.len());
            for project in &java_projects {
                log::debug!(
                    "  {} (project: {})",
                    project.package_name,
                    project.project_root
                );
            }
        }

        let python_packages = files
            .check_walk()
            .and_then(|_| crate::parsers::python::parse_python_packages_from(root, &files.python))
            .unwrap_or_else(|e| {
                log::warn!("Failed to parse Python package configs: {}", e);
                Vec::new()
            });
        if !python_packages.is_empty() {
            log::info!("Found {} Python packages", python_packages.len());
            for package in &python_packages {
                log::debug!("  {} (project: {})", package.name, package.project_root);
            }
        }

        let ruby_projects = files
            .check_walk()
            .and_then(|_| crate::parsers::ruby::parse_ruby_projects_from(root, &files.gemspecs))
            .unwrap_or_else(|e| {
                log::warn!("Failed to parse Ruby project configs: {}", e);
                Vec::new()
            });
        if !ruby_projects.is_empty() {
            log::info!("Found {} Ruby projects", ruby_projects.len());
            for project in &ruby_projects {
                log::debug!("  {} (project: {})", project.gem_name, project.project_root);
            }
        }

        // Gate: only a workspace with a root `Cargo.toml` is scanned for crates.
        let rust_crates = if root.join("Cargo.toml").exists() {
            files
                .check_walk()
                .and_then(|_| crate::parsers::rust::parse_rust_crates_from(&files.cargo_tomls))
                .unwrap_or_else(|e| {
                    log::warn!("Failed to parse Cargo.toml files: {}", e);
                    Vec::new()
                })
        } else {
            Vec::new()
        };
        if !rust_crates.is_empty() {
            log::info!("Found {} Rust workspace crates", rust_crates.len());
            for krate in &rust_crates {
                log::debug!("  {} (root: {})", krate.name, krate.root_path.display());
            }
        }

        // Note: Kotlin projects use the same java_projects above (same build systems: Maven/Gradle)

        let php_psr4 = files
            .check_walk()
            .and_then(|_| {
                crate::parsers::php::parse_composer_psr4_from(root, files.composer.clone())
            })
            .unwrap_or_else(|e| {
                log::warn!("Failed to parse composer.json files: {}", e);
                Vec::new()
            });
        if !php_psr4.is_empty() {
            log::info!(
                "Found {} PSR-4 mappings from composer.json files",
                php_psr4.len()
            );
            for mapping in &php_psr4 {
                log::debug!(
                    "  {} => {} (project: {})",
                    mapping.namespace_prefix,
                    mapping.directory,
                    mapping.project_root
                );
            }
        }

        Self {
            tsconfigs,
            go_modules,
            java_projects,
            python_packages,
            ruby_projects,
            rust_crates,
            php_psr4,
            digest,
            files,
        }
    }
}

/// What an import resolved to (see [`ResolverContext::resolve`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolution {
    /// One indexed file (`file_dependencies.resolved_file_id`).
    File(i64),
    /// A package key (`resolved_package`) and optionally one member of it
    /// (`resolved_member`); the graph reaches every file in `package_members`.
    Package {
        key: String,
        member: Option<String>,
    },
    Unresolved,
}

/// The `(package key, member)` rows that make the file at `rel_path` reachable
/// by package imports (`package_members`). Member `""` stands for the whole
/// package.
pub fn package_members(rel_path: &str) -> Vec<(String, String)> {
    if rel_path.ends_with(".go") {
        return crate::parsers::go::go_package_member(rel_path)
            .map(|key| vec![(key, String::new())])
            .unwrap_or_default();
    }
    Vec::new()
}

/// The resolution rules for one workspace, over its parsed configs.
pub struct ResolverContext<'a> {
    pub root: &'a Path,
    pub configs: &'a ResolverConfigs,
}

impl<'a> ResolverContext<'a> {
    pub fn new(root: &'a Path, configs: &'a ResolverConfigs) -> Self {
        Self { root, configs }
    }

    /// The alias map a TS/JS/Vue file is extracted and resolved with.
    pub fn alias_map_for(&self, file_path: &str) -> Option<&'a PathAliasMap> {
        find_nearest_tsconfig(file_path, self.root, &self.configs.tsconfigs)
    }

    /// Every extracted import of one file, reclassified and resolved, as the rows
    /// `replace_dependencies` writes.
    pub fn resolve_file_imports(
        &self,
        file_id: i64,
        file_path: &str,
        import_infos: Vec<ImportInfo>,
        resolver: &PathResolver,
    ) -> Vec<Dependency> {
        let mut resolved_deps = Vec::with_capacity(import_infos.len());
        for mut import_info in import_infos {
            self.reclassify(file_path, &mut import_info);

            // External and Stdlib imports: store with resolved_file_id = None.
            // Graph-analysis queries all filter WHERE resolved_file_id IS NOT NULL,
            // so storing these here only affects the `rfx deps` display path (REF-78).
            if matches!(
                import_info.import_type,
                ImportType::External | ImportType::Stdlib
            ) {
                resolved_deps.push(Dependency {
                    file_id,
                    imported_path: import_info.imported_path.clone(),
                    resolved_file_id: None,
                    resolved_package: None,
                    resolved_member: None,
                    import_type: import_info.import_type.clone(),
                    line_number: import_info.line_number,
                    imported_symbols: import_info.imported_symbols.clone(),
                });
                continue;
            }

            let (resolved_file_id, resolved_package, resolved_member) =
                match self.resolve(file_path, &import_info, resolver) {
                    Resolution::File(id) => (Some(id), None, None),
                    Resolution::Package { key, member } => (None, Some(key), member),
                    Resolution::Unresolved => (None, None, None),
                };
            resolved_deps.push(Dependency {
                file_id,
                imported_path: import_info.imported_path.clone(),
                resolved_file_id,
                resolved_package,
                resolved_member,
                import_type: import_info.import_type,
                line_number: import_info.line_number,
                imported_symbols: import_info.imported_symbols.clone(),
            });
        }
        resolved_deps
    }

    /// Reclassify an extracted import with the workspace's module, package and
    /// crate names.
    pub fn reclassify(&self, file_path: &str, import_info: &mut ImportInfo) {
        let go_modules = &self.configs.go_modules;
        let java_projects = &self.configs.java_projects;
        let python_packages = &self.configs.python_packages;
        let ruby_projects = &self.configs.ruby_projects;
        let rust_crates = &self.configs.rust_crates;

        // Reclassify Go imports using module names (if Go project)
        if file_path.ends_with(".go") {
            // Check if the import matches any Go module
            let mut reclassified = false;
            for module in go_modules {
                import_info.import_type = crate::parsers::go::reclassify_go_import(
                    &import_info.imported_path,
                    Some(&module.name),
                );
                // If it's internal, we've found the right module
                if matches!(import_info.import_type, ImportType::Internal) {
                    reclassified = true;
                    break;
                }
            }
            // If no module matched, use base classification
            if !reclassified {
                import_info.import_type =
                    crate::parsers::go::reclassify_go_import(&import_info.imported_path, None);
            }
        }

        // Reclassify Java imports using package names (if Java project)
        if file_path.ends_with(".java") {
            // Check if the import matches any Java project
            let mut reclassified = false;
            for project in java_projects {
                import_info.import_type = crate::parsers::java::reclassify_java_import(
                    &import_info.imported_path,
                    Some(&project.package_name),
                );
                // If it's internal, we've found the right project
                if matches!(import_info.import_type, ImportType::Internal) {
                    reclassified = true;
                    break;
                }
            }
            // If no project matched, use base classification
            if !reclassified {
                import_info.import_type =
                    crate::parsers::java::reclassify_java_import(&import_info.imported_path, None);
            }
        }

        // Reclassify Python imports using package names (if Python project)
        if file_path.ends_with(".py") {
            // Check if the import matches any Python package
            let mut reclassified = false;
            for package in python_packages {
                import_info.import_type = crate::parsers::python::reclassify_python_import(
                    &import_info.imported_path,
                    Some(&package.name),
                );
                // If it's internal, we've found the right package
                if matches!(import_info.import_type, ImportType::Internal) {
                    reclassified = true;
                    break;
                }
            }
            // If no package matched, use base classification
            if !reclassified {
                import_info.import_type = crate::parsers::python::reclassify_python_import(
                    &import_info.imported_path,
                    None,
                );
            }
        }

        // Reclassify Ruby imports using gem names (if Ruby project)
        if file_path.ends_with(".rb")
            || file_path.ends_with(".rake")
            || file_path.ends_with(".gemspec")
        {
            // Check if the import matches any Ruby project
            let mut reclassified = false;
            for project in ruby_projects {
                let gem_names = vec![project.gem_name.clone()];
                import_info.import_type = crate::parsers::ruby::reclassify_ruby_import(
                    &import_info.imported_path,
                    &gem_names,
                );
                // If it's internal, we've found the right project
                if matches!(import_info.import_type, ImportType::Internal) {
                    reclassified = true;
                    break;
                }
            }
            // If no project matched, use base classification (will be External or Stdlib)
            if !reclassified {
                import_info.import_type =
                    crate::parsers::ruby::reclassify_ruby_import(&import_info.imported_path, &[]);
            }
        }

        // Reclassify Kotlin imports using package names (if Kotlin project)
        if file_path.ends_with(".kt") || file_path.ends_with(".kts") {
            // Check if the import matches any Java/Kotlin project (same build systems)
            let mut reclassified = false;
            for project in java_projects {
                import_info.import_type = crate::parsers::kotlin::reclassify_kotlin_import(
                    &import_info.imported_path,
                    Some(&project.package_name),
                );
                // If it's internal, we've found the right project
                if matches!(import_info.import_type, ImportType::Internal) {
                    reclassified = true;
                    break;
                }
            }
            // If no project matched, use base classification
            if !reclassified {
                import_info.import_type = crate::parsers::kotlin::reclassify_kotlin_import(
                    &import_info.imported_path,
                    None,
                );
            }
        }

        // Reclassify Rust imports using workspace crates
        if file_path.ends_with(".rs") && !rust_crates.is_empty() {
            let new_type = crate::parsers::rust::reclassify_rust_import(
                &import_info.imported_path,
                rust_crates,
            );
            if matches!(new_type, ImportType::Internal) {
                import_info.import_type = new_type;
            }
        }
    }

    /// Where an import that is neither External nor Stdlib leads: one file, or a
    /// package the graph expands to its member files ([`package_members`]).
    pub fn resolve(
        &self,
        file_path: &str,
        import_info: &ImportInfo,
        resolver: &PathResolver,
    ) -> Resolution {
        if file_path.ends_with(".go") {
            // A Go import names a package (a directory), never one file
            return crate::parsers::go::go_package_key(
                &import_info.imported_path,
                &self.configs.go_modules,
            )
            .map_or(Resolution::Unresolved, |key| Resolution::Package {
                key,
                member: None,
            });
        }
        self.resolve_import(file_path, import_info, resolver)
            .map_or(Resolution::Unresolved, Resolution::File)
    }

    /// The `files.id` an import that is neither External nor Stdlib resolves to,
    /// for the languages whose imports name one file (see [`Self::resolve`]).
    pub fn resolve_import(
        &self,
        file_path: &str,
        import_info: &ImportInfo,
        resolver: &PathResolver,
    ) -> Option<i64> {
        let root = self.root;
        let tsconfigs = &self.configs.tsconfigs;
        let java_projects = &self.configs.java_projects;
        let python_packages = &self.configs.python_packages;
        let ruby_projects = &self.configs.ruby_projects;
        let rust_crates = &self.configs.rust_crates;
        let php_psr4_mappings = &self.configs.php_psr4;

        if file_path.ends_with(".php") && !php_psr4_mappings.is_empty() {
            // Use PSR-4 to resolve namespace to file path
            if let Some(resolved_path) = crate::parsers::php::resolve_php_namespace_to_path(
                &import_info.imported_path,
                php_psr4_mappings,
            ) {
                // Look up file ID in database using exact match
                match resolver.get_file_id_by_path(&resolved_path) {
                    Ok(Some(id)) => {
                        log::trace!(
                            "Resolved PHP dependency: {} -> {} (file_id={})",
                            import_info.imported_path,
                            resolved_path,
                            id
                        );
                        Some(id)
                    }
                    Ok(None) => {
                        log::trace!(
                            "PHP dependency resolved to path but file not in index: {} -> {}",
                            import_info.imported_path,
                            resolved_path
                        );
                        None
                    }
                    Err(e) => {
                        log::debug!(
                            "Skipping PHP dependency resolution for '{}': {}",
                            resolved_path,
                            e
                        );
                        None
                    }
                }
            } else {
                log::trace!(
                    "Could not resolve PHP namespace using PSR-4: {}",
                    import_info.imported_path
                );
                None
            }
        } else if file_path.ends_with(".py") && !python_packages.is_empty() {
            // Resolve Python dependencies using package mappings
            if let Some(resolved_path) = crate::parsers::python::resolve_python_import_to_path(
                &import_info.imported_path,
                python_packages,
                Some(file_path),
            ) {
                // Look up file ID in database using exact match
                match resolver.get_file_id_by_path(&resolved_path) {
                    Ok(Some(id)) => {
                        log::trace!(
                            "Resolved Python dependency: {} -> {} (file_id={})",
                            import_info.imported_path,
                            resolved_path,
                            id
                        );
                        Some(id)
                    }
                    Ok(None) => {
                        log::trace!(
                            "Python dependency resolved to path but file not in index: {} -> {}",
                            import_info.imported_path,
                            resolved_path
                        );
                        None
                    }
                    Err(e) => {
                        log::debug!(
                            "Skipping Python dependency resolution for '{}': {}",
                            resolved_path,
                            e
                        );
                        None
                    }
                }
            } else {
                log::trace!(
                    "Could not resolve Python import: {}",
                    import_info.imported_path
                );
                None
            }
        } else if file_path.ends_with(".ts")
            || file_path.ends_with(".tsx")
            || file_path.ends_with(".js")
            || file_path.ends_with(".jsx")
            || file_path.ends_with(".mts")
            || file_path.ends_with(".cts")
            || file_path.ends_with(".mjs")
            || file_path.ends_with(".cjs")
        {
            // Resolve TypeScript/JavaScript dependencies (relative imports and path aliases)
            let alias_map = find_nearest_tsconfig(file_path, root, tsconfigs);
            if let Some(candidates_str) = crate::parsers::typescript::resolve_ts_import_to_path(
                &import_info.imported_path,
                Some(file_path),
                alias_map,
            ) {
                // Parse pipe-delimited candidates (e.g., "path.tsx|path.ts|path.jsx|path.js")
                let candidates: Vec<&str> = candidates_str.split('|').collect();

                // Try each candidate in order until we find one in the database
                let mut resolved_id = None;
                for candidate_path in candidates {
                    // Normalize path to be relative to project root
                    // Convert absolute paths to relative (without requiring file to exist)
                    let normalized_candidate = if let Ok(rel_path) =
                        std::path::Path::new(candidate_path).strip_prefix(root)
                    {
                        rel_path.to_string_lossy().replace('\\', "/")
                    } else {
                        // Not an absolute path or not under root - use as-is
                        // (still normalize separators so DB lookups match).
                        candidate_path.replace('\\', "/")
                    };

                    log::debug!(
                        "Looking up TS/JS candidate: '{}' (from '{}')",
                        normalized_candidate,
                        candidate_path
                    );
                    match resolver.get_file_id_by_path(&normalized_candidate) {
                        Ok(Some(id)) => {
                            log::debug!(
                                "Resolved TS/JS dependency: {} -> {} (file_id={})",
                                import_info.imported_path,
                                normalized_candidate,
                                id
                            );
                            resolved_id = Some(id);
                            break; // Found a match, stop trying
                        }
                        Ok(None) => {
                            log::trace!("TS/JS candidate not in index: {}", candidate_path);
                        }
                        Err(e) => {
                            log::debug!(
                                "Skipping TS/JS dependency resolution for '{}': {}",
                                normalized_candidate,
                                e
                            );
                        }
                    }
                }

                if resolved_id.is_none() {
                    log::trace!(
                        "TS/JS dependency: no matching file found in database for any candidate: {}",
                        candidates_str
                    );
                }

                resolved_id
            } else {
                log::trace!(
                    "Could not resolve TS/JS import (non-relative or external): {}",
                    import_info.imported_path
                );
                None
            }
        } else if file_path.ends_with(".rs") {
            // Resolve Rust dependencies (crate::, super::, self::, mod declarations)
            // Falls back to workspace resolution for cross-crate imports
            let resolved_path_opt = crate::parsers::rust::resolve_rust_use_to_path(
                &import_info.imported_path,
                Some(file_path),
                Some(root.to_str().unwrap_or("")),
            )
            .or_else(|| {
                crate::parsers::rust::resolve_rust_workspace_path(
                    &import_info.imported_path,
                    rust_crates,
                )
            });

            if let Some(resolved_path) = resolved_path_opt {
                // Look up file ID in database using exact match
                match resolver.get_file_id_by_path(&resolved_path) {
                    Ok(Some(id)) => {
                        log::trace!(
                            "Resolved Rust dependency: {} -> {} (file_id={})",
                            import_info.imported_path,
                            resolved_path,
                            id
                        );
                        Some(id)
                    }
                    Ok(None) => {
                        log::trace!(
                            "Rust dependency resolved to path but file not in index: {} -> {}",
                            import_info.imported_path,
                            resolved_path
                        );
                        None
                    }
                    Err(e) => {
                        log::debug!(
                            "Skipping Rust dependency resolution for '{}': {}",
                            resolved_path,
                            e
                        );
                        None
                    }
                }
            } else {
                log::trace!(
                    "Could not resolve Rust import (external or stdlib): {}",
                    import_info.imported_path
                );
                None
            }
        } else if file_path.ends_with(".java") && !java_projects.is_empty() {
            // Resolve Java dependencies using project mappings
            if let Some(resolved_path) = crate::parsers::java::resolve_java_import_to_path(
                &import_info.imported_path,
                java_projects,
                Some(file_path),
            ) {
                // Look up file ID in database using exact match
                match resolver.get_file_id_by_path(&resolved_path) {
                    Ok(Some(id)) => {
                        log::trace!(
                            "Resolved Java dependency: {} -> {} (file_id={})",
                            import_info.imported_path,
                            resolved_path,
                            id
                        );
                        Some(id)
                    }
                    Ok(None) => {
                        log::trace!(
                            "Java dependency resolved to path but file not in index: {} -> {}",
                            import_info.imported_path,
                            resolved_path
                        );
                        None
                    }
                    Err(e) => {
                        log::debug!(
                            "Skipping Java dependency resolution for '{}': {}",
                            resolved_path,
                            e
                        );
                        None
                    }
                }
            } else {
                log::trace!(
                    "Could not resolve Java import: {}",
                    import_info.imported_path
                );
                None
            }
        } else if (file_path.ends_with(".kt") || file_path.ends_with(".kts"))
            && !java_projects.is_empty()
        {
            // Resolve Kotlin dependencies using project mappings (same build systems as Java)
            if let Some(resolved_path) = crate::parsers::java::resolve_kotlin_import_to_path(
                &import_info.imported_path,
                java_projects,
                Some(file_path),
            ) {
                // Look up file ID in database using exact match
                match resolver.get_file_id_by_path(&resolved_path) {
                    Ok(Some(id)) => {
                        log::trace!(
                            "Resolved Kotlin dependency: {} -> {} (file_id={})",
                            import_info.imported_path,
                            resolved_path,
                            id
                        );
                        Some(id)
                    }
                    Ok(None) => {
                        log::trace!(
                            "Kotlin dependency resolved to path but file not in index: {} -> {}",
                            import_info.imported_path,
                            resolved_path
                        );
                        None
                    }
                    Err(e) => {
                        log::debug!(
                            "Skipping Kotlin dependency resolution for '{}': {}",
                            resolved_path,
                            e
                        );
                        None
                    }
                }
            } else {
                log::trace!(
                    "Could not resolve Kotlin import: {}",
                    import_info.imported_path
                );
                None
            }
        } else if (file_path.ends_with(".rb")
            || file_path.ends_with(".rake")
            || file_path.ends_with(".gemspec"))
            && !ruby_projects.is_empty()
        {
            // Resolve Ruby dependencies using project mappings
            if let Some(resolved_path) = crate::parsers::ruby::resolve_ruby_require_to_path(
                &import_info.imported_path,
                ruby_projects,
                Some(file_path),
            ) {
                // Look up file ID in database using exact match
                match resolver.get_file_id_by_path(&resolved_path) {
                    Ok(Some(id)) => {
                        log::trace!(
                            "Resolved Ruby dependency: {} -> {} (file_id={})",
                            import_info.imported_path,
                            resolved_path,
                            id
                        );
                        Some(id)
                    }
                    Ok(None) => {
                        log::trace!(
                            "Ruby dependency resolved to path but file not in index: {} -> {}",
                            import_info.imported_path,
                            resolved_path
                        );
                        None
                    }
                    Err(e) => {
                        log::debug!(
                            "Skipping Ruby dependency resolution for '{}': {}",
                            resolved_path,
                            e
                        );
                        None
                    }
                }
            } else {
                log::trace!(
                    "Could not resolve Ruby require: {}",
                    import_info.imported_path
                );
                None
            }
        } else if file_path.ends_with(".c") || file_path.ends_with(".h") {
            // Resolve C dependencies (relative #include paths)
            if let Some(resolved_path) = crate::parsers::c::resolve_c_include_to_path(
                &import_info.imported_path,
                Some(file_path),
            ) {
                // Look up file ID in database using exact match
                match resolver.get_file_id_by_path(&resolved_path) {
                    Ok(Some(id)) => {
                        log::trace!(
                            "Resolved C dependency: {} -> {} (file_id={})",
                            import_info.imported_path,
                            resolved_path,
                            id
                        );
                        Some(id)
                    }
                    Ok(None) => {
                        log::trace!(
                            "C dependency resolved to path but file not in index: {} -> {}",
                            import_info.imported_path,
                            resolved_path
                        );
                        None
                    }
                    Err(e) => {
                        log::debug!(
                            "Skipping C dependency resolution for '{}': {}",
                            resolved_path,
                            e
                        );
                        None
                    }
                }
            } else {
                log::trace!(
                    "Could not resolve C include (system header): {}",
                    import_info.imported_path
                );
                None
            }
        } else if file_path.ends_with(".cpp")
            || file_path.ends_with(".cc")
            || file_path.ends_with(".cxx")
            || file_path.ends_with(".hpp")
            || file_path.ends_with(".hxx")
            || file_path.ends_with(".h++")
            || file_path.ends_with(".C")
            || file_path.ends_with(".H")
        {
            // Resolve C++ dependencies (relative #include paths)
            if let Some(resolved_path) = crate::parsers::cpp::resolve_cpp_include_to_path(
                &import_info.imported_path,
                Some(file_path),
            ) {
                // Look up file ID in database using exact match
                match resolver.get_file_id_by_path(&resolved_path) {
                    Ok(Some(id)) => {
                        log::trace!(
                            "Resolved C++ dependency: {} -> {} (file_id={})",
                            import_info.imported_path,
                            resolved_path,
                            id
                        );
                        Some(id)
                    }
                    Ok(None) => {
                        log::trace!(
                            "C++ dependency resolved to path but file not in index: {} -> {}",
                            import_info.imported_path,
                            resolved_path
                        );
                        None
                    }
                    Err(e) => {
                        log::debug!(
                            "Skipping C++ dependency resolution for '{}': {}",
                            resolved_path,
                            e
                        );
                        None
                    }
                }
            } else {
                log::trace!(
                    "Could not resolve C++ include (system header): {}",
                    import_info.imported_path
                );
                None
            }
        } else if file_path.ends_with(".cs") {
            // Resolve C# dependencies (using namespace-to-path mapping)
            if let Some(resolved_path) = crate::parsers::csharp::resolve_csharp_using_to_path(
                &import_info.imported_path,
                Some(file_path),
            ) {
                // Look up file ID in database using exact match
                match resolver.get_file_id_by_path(&resolved_path) {
                    Ok(Some(id)) => {
                        log::trace!(
                            "Resolved C# dependency: {} -> {} (file_id={})",
                            import_info.imported_path,
                            resolved_path,
                            id
                        );
                        Some(id)
                    }
                    Ok(None) => {
                        log::trace!(
                            "C# dependency resolved to path but file not in index: {} -> {}",
                            import_info.imported_path,
                            resolved_path
                        );
                        None
                    }
                    Err(e) => {
                        log::debug!(
                            "Skipping C# dependency resolution for '{}': {}",
                            resolved_path,
                            e
                        );
                        None
                    }
                }
            } else {
                log::trace!(
                    "Could not resolve C# using directive: {}",
                    import_info.imported_path
                );
                None
            }
        } else if file_path.ends_with(".zig") {
            // Resolve Zig dependencies (relative @import paths)
            if let Some(resolved_path) = crate::parsers::zig::resolve_zig_import_to_path(
                &import_info.imported_path,
                Some(file_path),
            ) {
                // Look up file ID in database using exact match
                match resolver.get_file_id_by_path(&resolved_path) {
                    Ok(Some(id)) => {
                        log::trace!(
                            "Resolved Zig dependency: {} -> {} (file_id={})",
                            import_info.imported_path,
                            resolved_path,
                            id
                        );
                        Some(id)
                    }
                    Ok(None) => {
                        log::trace!(
                            "Zig dependency resolved to path but file not in index: {} -> {}",
                            import_info.imported_path,
                            resolved_path
                        );
                        None
                    }
                    Err(e) => {
                        log::debug!(
                            "Skipping Zig dependency resolution for '{}': {}",
                            resolved_path,
                            e
                        );
                        None
                    }
                }
            } else {
                log::trace!(
                    "Could not resolve Zig import (external or stdlib): {}",
                    import_info.imported_path
                );
                None
            }
        } else if file_path.ends_with(".vue") || file_path.ends_with(".svelte") {
            // Resolve Vue/Svelte dependencies (use TypeScript/JavaScript resolver for imports in <script> blocks)
            let alias_map = find_nearest_tsconfig(file_path, root, tsconfigs);
            if let Some(candidates_str) = crate::parsers::typescript::resolve_ts_import_to_path(
                &import_info.imported_path,
                Some(file_path),
                alias_map,
            ) {
                // Parse pipe-delimited candidates (e.g., "path.tsx|path.ts|path.jsx|path.js")
                let candidates: Vec<&str> = candidates_str.split('|').collect();

                // Try each candidate in order until we find one in the database
                let mut resolved_id = None;
                for candidate_path in candidates {
                    // Normalize path to be relative to project root
                    // Convert absolute paths to relative (without requiring file to exist)
                    let normalized_candidate = if let Ok(rel_path) =
                        std::path::Path::new(candidate_path).strip_prefix(root)
                    {
                        rel_path.to_string_lossy().replace('\\', "/")
                    } else {
                        // Not an absolute path or not under root - use as-is
                        // (still normalize separators so DB lookups match).
                        candidate_path.replace('\\', "/")
                    };

                    match resolver.get_file_id_by_path(&normalized_candidate) {
                        Ok(Some(id)) => {
                            log::trace!(
                                "Resolved Vue/Svelte dependency: {} -> {} (file_id={})",
                                import_info.imported_path,
                                candidate_path,
                                id
                            );
                            resolved_id = Some(id);
                            break; // Found a match, stop trying
                        }
                        Ok(None) => {
                            log::trace!("Vue/Svelte candidate not in index: {}", candidate_path);
                        }
                        Err(e) => {
                            log::debug!(
                                "Skipping Vue/Svelte dependency resolution for '{}': {}",
                                normalized_candidate,
                                e
                            );
                        }
                    }
                }

                if resolved_id.is_none() {
                    log::trace!(
                        "Vue/Svelte dependency: no matching file found in database for any candidate: {}",
                        candidates_str
                    );
                }

                resolved_id
            } else {
                log::trace!(
                    "Could not resolve Vue/Svelte import (non-relative or external): {}",
                    import_info.imported_path
                );
                None
            }
        } else {
            None
        }
    }

    /// The `files.id` a re-export's source resolves to (TS/JS/Vue only).
    pub fn resolve_export(
        &self,
        file_path: &str,
        export_info: &ExportInfo,
        resolver: &PathResolver,
    ) -> Option<i64> {
        let root = self.root;
        let tsconfigs = &self.configs.tsconfigs;

        if file_path.ends_with(".ts")
            || file_path.ends_with(".tsx")
            || file_path.ends_with(".js")
            || file_path.ends_with(".jsx")
            || file_path.ends_with(".mts")
            || file_path.ends_with(".cts")
            || file_path.ends_with(".mjs")
            || file_path.ends_with(".cjs")
            || file_path.ends_with(".vue")
        {
            // Resolve TypeScript/JavaScript/Vue export paths (relative imports and path aliases)
            let alias_map = find_nearest_tsconfig(file_path, root, tsconfigs);
            if let Some(candidates_str) = crate::parsers::typescript::resolve_ts_import_to_path(
                &export_info.source_path,
                Some(file_path),
                alias_map,
            ) {
                // Parse pipe-delimited candidates (e.g., "path.tsx|path.ts|path.jsx|path.js|path.vue")
                let candidates: Vec<&str> = candidates_str.split('|').collect();

                // Try each candidate in order until we find one in the database
                let mut resolved_id = None;
                for candidate_path in candidates {
                    // Normalize path to be relative to project root
                    let normalized_candidate = if let Ok(rel_path) =
                        std::path::Path::new(candidate_path).strip_prefix(root)
                    {
                        rel_path.to_string_lossy().to_string()
                    } else {
                        candidate_path.to_string()
                    };

                    match resolver.get_file_id_by_path(&normalized_candidate) {
                        Ok(Some(id)) => {
                            log::trace!(
                                "Resolved export source: {} -> {} (file_id={})",
                                export_info.source_path,
                                normalized_candidate,
                                id
                            );
                            resolved_id = Some(id);
                            break; // Found a match, stop trying
                        }
                        Ok(None) => {
                            log::trace!("Export source candidate not in index: {}", candidate_path);
                        }
                        Err(e) => {
                            log::debug!(
                                "Skipping export source resolution for '{}': {}",
                                normalized_candidate,
                                e
                            );
                        }
                    }
                }

                if resolved_id.is_none() {
                    log::trace!(
                        "Export source: no matching file found in database for any candidate: {}",
                        candidates_str
                    );
                }

                resolved_id
            } else {
                log::trace!(
                    "Could not resolve export source (non-relative or external): {}",
                    export_info.source_path
                );
                None
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, body).unwrap();
    }

    /// A tree with every config kind, plus each finder's skip rule.
    fn workspace() -> TempDir {
        let temp = TempDir::new().unwrap();
        let r = temp.path();
        write(r, "Cargo.toml", "[package]\nname = \"root_crate\"\n");
        write(r, "crates/a/Cargo.toml", "[package]\nname = \"crate_a\"\n");
        write(r, "svc/go.mod", "module example.com/svc\n");
        write(r, "svc/vendor/x/go.mod", "module example.com/vendored\n");
        write(
            r,
            "java/pom.xml",
            "<project>\n<groupId>com.example</groupId>\n</project>\n",
        );
        write(r, "kt/build.gradle.kts", "group = \"com.kt\"\n");
        write(r, "py/pyproject.toml", "[project]\nname = \"pkg\"\n");
        write(r, "py/venv/lib/setup.py", "setup(name=\"skipped\")\n");
        write(
            r,
            "rb/thing.gemspec",
            "Gem::Specification.new do |s|\n  s.name = \"thing\"\nend\n",
        );
        write(
            r,
            "php/composer.json",
            r#"{"autoload":{"psr-4":{"App\\":"src/"}}}"#,
        );
        write(
            r,
            "php/vendor/lib/composer.json",
            r#"{"autoload":{"psr-4":{"Lib\\":"src/"}}}"#,
        );
        write(
            r,
            "web/tsconfig.json",
            r#"{"compilerOptions":{"baseUrl":".","paths":{"@/*":["src/*"]}}}"#,
        );
        // `.gitignore` applies only inside a repository.
        fs::create_dir_all(r.join(".git")).unwrap();
        write(r, ".gitignore", "ignored/\n");
        write(r, "ignored/go.mod", "module example.com/ignored\n");
        temp
    }

    #[test]
    fn one_walk_finds_what_the_seven_finders_found() {
        let temp = workspace();
        let root = temp.path();
        let one = ResolverConfigs::discover(root);

        let mut tsconfigs: Vec<_> = crate::parsers::tsconfig::parse_all_tsconfigs(root)
            .unwrap()
            .into_iter()
            .map(|(k, v)| format!("{:?} {:?}", k, v))
            .collect();
        tsconfigs.sort();
        let mut got: Vec<_> = one
            .tsconfigs
            .iter()
            .map(|(k, v)| format!("{:?} {:?}", k, v))
            .collect();
        got.sort();
        assert_eq!(got, tsconfigs);

        let dbg = |v: &dyn std::fmt::Debug| format!("{:?}", v);
        assert_eq!(
            dbg(&one.go_modules),
            dbg(&crate::parsers::go::parse_all_go_modules(root).unwrap())
        );
        assert_eq!(
            dbg(&one.java_projects),
            dbg(&crate::parsers::java::parse_all_java_projects(root).unwrap())
        );
        assert_eq!(
            dbg(&one.python_packages),
            dbg(&crate::parsers::python::parse_all_python_packages(root).unwrap())
        );
        assert_eq!(
            dbg(&one.ruby_projects),
            dbg(&crate::parsers::ruby::parse_all_ruby_projects(root).unwrap())
        );
        assert_eq!(
            dbg(&one.rust_crates),
            dbg(&crate::parsers::rust::parse_all_rust_crates(root).unwrap())
        );
        assert_eq!(
            dbg(&one.php_psr4),
            dbg(&crate::parsers::php::parse_all_composer_psr4(root).unwrap())
        );

        // The skip rules held: vendored and ignored configs are not there.
        assert_eq!(one.go_modules.len(), 1);
        assert_eq!(one.python_packages.len(), 1);
        assert_eq!(one.rust_crates.len(), 2);
    }

    #[test]
    fn rust_crates_need_a_root_cargo_toml() {
        let temp = TempDir::new().unwrap();
        write(
            temp.path(),
            "crates/a/Cargo.toml",
            "[package]\nname = \"crate_a\"\n",
        );
        let one = ResolverConfigs::discover(temp.path());
        assert!(one.rust_crates.is_empty());
    }
}
