//! TypeScript configuration file parser
//!
//! Parses tsconfig.json files to extract path alias mappings from compilerOptions.paths.
//! These mappings are used to resolve non-relative imports in TypeScript/JavaScript/Vue files.
//!
//! Example tsconfig.json:
//! ```json
//! {
//!   "compilerOptions": {
//!     "baseUrl": ".",
//!     "paths": {
//!       "~/*": ["./src/*"],
//!       "@packages/*": ["../../packages/*"]
//!     }
//!   }
//! }
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Path alias mapping from tsconfig.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathAliasMap {
    /// Map of alias pattern to target paths
    /// Example: "@packages/*" => ["../../packages/*"]
    pub aliases: HashMap<String, Vec<String>>,
    /// Base URL for resolving relative paths
    pub base_url: Option<String>,
    /// Directory containing the tsconfig.json file
    pub config_dir: PathBuf,
}

/// TypeScript compiler options (subset)
#[derive(Debug, Deserialize)]
struct CompilerOptions {
    #[serde(rename = "baseUrl")]
    base_url: Option<String>,
    paths: Option<HashMap<String, Vec<String>>>,
}

/// TypeScript configuration file structure (subset)
#[derive(Debug, Deserialize)]
struct TsConfig {
    #[serde(rename = "compilerOptions")]
    compiler_options: Option<CompilerOptions>,
}

impl PathAliasMap {
    /// Parse a tsconfig.json file and extract path aliases
    pub fn from_file(tsconfig_path: impl AsRef<Path>) -> Result<Self> {
        let tsconfig_path = tsconfig_path.as_ref();
        let content = std::fs::read_to_string(tsconfig_path).with_context(|| {
            format!("Failed to read tsconfig.json: {}", tsconfig_path.display())
        })?;

        // Parse JSON5 (tsconfig.json supports comments and trailing commas)
        let config: TsConfig = json5::from_str(&content).with_context(|| {
            format!("Failed to parse tsconfig.json: {}", tsconfig_path.display())
        })?;

        let config_dir = tsconfig_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Invalid tsconfig.json path"))?
            .to_path_buf();

        let compiler_options = config.compiler_options.unwrap_or_else(|| CompilerOptions {
            base_url: None,
            paths: None,
        });

        Ok(Self {
            aliases: compiler_options.paths.unwrap_or_default(),
            base_url: compiler_options.base_url,
            config_dir,
        })
    }

    /// Find the nearest tsconfig.json for a given source file
    ///
    /// Walks up the directory tree from the source file until it finds a tsconfig.json.
    /// Returns None if no tsconfig.json is found.
    pub fn find_nearest_tsconfig(source_file: &Path) -> Option<PathBuf> {
        let mut current_dir = source_file.parent()?;

        loop {
            let tsconfig_path = current_dir.join("tsconfig.json");
            if tsconfig_path.exists() {
                return Some(tsconfig_path);
            }

            // Also check for .nuxt/tsconfig.json (Nuxt-generated)
            let nuxt_tsconfig = current_dir.join(".nuxt/tsconfig.json");
            if nuxt_tsconfig.exists() {
                return Some(nuxt_tsconfig);
            }

            // Move up one directory
            current_dir = current_dir.parent()?;
        }
    }

    /// Resolve an import path using the path alias mappings
    ///
    /// Returns the resolved path if the import matches an alias, None otherwise.
    ///
    /// Example:
    /// - Alias: `@packages/*` => `../../packages/*`
    /// - Import: `@packages/ui/stores/auth`
    /// - Resolves to: `../../packages/ui/stores/auth`
    pub fn resolve_alias(&self, import_path: &str) -> Option<String> {
        log::debug!(
            "  resolve_alias: trying to match '{}' against {} aliases",
            import_path,
            self.aliases.len()
        );

        // Try to match against each alias pattern
        for (alias_pattern, target_paths) in &self.aliases {
            log::trace!(
                "    Checking alias pattern: {} => {:?}",
                alias_pattern,
                target_paths
            );
            // Check if pattern has a wildcard
            if alias_pattern.ends_with("/*") {
                let alias_prefix = alias_pattern.trim_end_matches("/*");

                // Check if import starts with the alias prefix
                if import_path.starts_with(alias_prefix) {
                    // Extract the suffix after the alias prefix
                    // Example: "@packages/ui/stores/auth" with alias "@packages"
                    //          => suffix = "/ui/stores/auth"
                    let suffix = import_path.strip_prefix(alias_prefix).unwrap_or("");

                    // Use the first target path (tsconfig allows multiple, but we'll use the first)
                    if let Some(target_pattern) = target_paths.first() {
                        // Replace wildcard in target with the suffix
                        let resolved = if target_pattern.ends_with("/*") {
                            let target_prefix = target_pattern.trim_end_matches("/*");
                            format!("{}{}", target_prefix, suffix)
                        } else {
                            // No wildcard in target - append suffix with proper path joining
                            // Strip leading '/' from suffix to avoid double slashes
                            // Example: ".." + "/ui/stores/auth" => "../ui/stores/auth"
                            let clean_suffix = suffix.trim_start_matches('/');
                            if clean_suffix.is_empty() {
                                target_pattern.to_string()
                            } else {
                                format!("{}/{}", target_pattern, clean_suffix)
                            }
                        };

                        log::trace!(
                            "Resolved alias {} + {} => {}",
                            alias_pattern,
                            import_path,
                            resolved
                        );
                        return Some(resolved);
                    }
                }
            } else {
                // Exact match (no wildcard)
                if import_path == alias_pattern {
                    if let Some(target) = target_paths.first() {
                        log::trace!("Resolved exact alias {} => {}", alias_pattern, target);
                        return Some(target.clone());
                    }
                }
            }
        }

        None
    }

    /// Resolve a path relative to the tsconfig directory and baseUrl
    pub fn resolve_relative_to_config(&self, path: &str) -> PathBuf {
        let base = if let Some(ref base_url) = self.base_url {
            self.config_dir.join(base_url)
        } else {
            self.config_dir.clone()
        };

        let joined = base.join(path);

        // Normalize the path to resolve .. components without requiring file to exist
        // Example: /home/user/packages/ui/./ui => /home/user/packages/ui
        let normalized = joined
            .components()
            .fold(PathBuf::new(), |mut acc, component| {
                match component {
                    std::path::Component::CurDir => acc, // Skip .
                    std::path::Component::ParentDir => {
                        acc.pop(); // Go up one level for ..
                        acc
                    }
                    _ => {
                        acc.push(component);
                        acc
                    }
                }
            });

        normalized
    }
}

/// Find and parse all tsconfig.json files in a project directory
///
/// Walks the directory tree to discover all tsconfig.json files (including .nuxt/tsconfig.json)
/// and returns a HashMap mapping each tsconfig directory to its PathAliasMap.
///
/// This supports monorepos with multiple tsconfig.json files in different directories.
/// Respects .gitignore rules to skip node_modules and other ignored directories.
pub fn parse_all_tsconfigs(
    root: &Path,
) -> Result<std::collections::HashMap<PathBuf, PathAliasMap>> {
    use ignore::WalkBuilder;
    use std::collections::HashMap;

    log::debug!("Starting tsconfig discovery in {}", root.display());
    let mut tsconfigs = HashMap::new();
    let mut file_count = 0;

    // Walk directory tree respecting .gitignore
    for entry in WalkBuilder::new(root)
        .follow_links(false)
        .build()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();

        // Check if this is a tsconfig.json file
        if path.file_name().and_then(|n| n.to_str()) == Some("tsconfig.json") {
            file_count += 1;
            log::debug!(
                "Found tsconfig.json file #{}: {}",
                file_count,
                path.display()
            );

            // Parse the tsconfig file
            match PathAliasMap::from_file(path) {
                Ok(alias_map) => {
                    // Store using the directory containing the tsconfig as key
                    let config_dir = alias_map.config_dir.clone();
                    log::debug!(
                        "  Parsed successfully: base_url={:?}, {} aliases",
                        alias_map.base_url,
                        alias_map.aliases.len()
                    );
                    tsconfigs.insert(config_dir, alias_map);
                }
                Err(e) => {
                    log::warn!("Failed to parse tsconfig.json at {}: {}", path.display(), e);
                }
            }
        }
    }

    log::debug!(
        "Tsconfig discovery complete: found {} files, parsed {} successfully",
        file_count,
        tsconfigs.len()
    );
    Ok(tsconfigs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_parse_tsconfig_with_paths() {
        let temp = TempDir::new().unwrap();
        let tsconfig_path = temp.path().join("tsconfig.json");

        let tsconfig_content = r#"{
            "compilerOptions": {
                "baseUrl": ".",
                "paths": {
                    "~/*": ["./src/*"],
                    "@packages/*": ["../../packages/*"]
                }
            }
        }"#;

        fs::write(&tsconfig_path, tsconfig_content).unwrap();

        let alias_map = PathAliasMap::from_file(&tsconfig_path).unwrap();

        assert_eq!(alias_map.base_url, Some(".".to_string()));
        assert_eq!(alias_map.aliases.len(), 2);
        assert!(alias_map.aliases.contains_key("~/*"));
        assert!(alias_map.aliases.contains_key("@packages/*"));
    }

    #[test]
    fn test_resolve_wildcard_alias() {
        let temp = TempDir::new().unwrap();
        let alias_map = PathAliasMap {
            aliases: HashMap::from([(
                "@packages/*".to_string(),
                vec!["../../packages/*".to_string()],
            )]),
            base_url: Some(".".to_string()),
            config_dir: temp.path().to_path_buf(),
        };

        // Test wildcard alias resolution
        let resolved = alias_map.resolve_alias("@packages/ui/stores/auth");
        assert_eq!(resolved, Some("../../packages/ui/stores/auth".to_string()));
    }

    #[test]
    fn test_resolve_exact_alias() {
        let temp = TempDir::new().unwrap();
        let alias_map = PathAliasMap {
            aliases: HashMap::from([("~".to_string(), vec!["./src".to_string()])]),
            base_url: None,
            config_dir: temp.path().to_path_buf(),
        };

        // Test exact alias resolution
        let resolved = alias_map.resolve_alias("~");
        assert_eq!(resolved, Some("./src".to_string()));
    }

    #[test]
    fn test_no_match() {
        let temp = TempDir::new().unwrap();
        let alias_map = PathAliasMap {
            aliases: HashMap::from([(
                "@packages/*".to_string(),
                vec!["../../packages/*".to_string()],
            )]),
            base_url: None,
            config_dir: temp.path().to_path_buf(),
        };

        // Import doesn't match any alias
        let resolved = alias_map.resolve_alias("./relative/path");
        assert_eq!(resolved, None);
    }

    #[test]
    fn test_find_nearest_tsconfig() {
        let temp = TempDir::new().unwrap();

        // Create directory structure: temp/src/components/
        let src_dir = temp.path().join("src");
        let components_dir = src_dir.join("components");
        fs::create_dir_all(&components_dir).unwrap();

        // Create tsconfig.json in temp/
        let tsconfig_path = temp.path().join("tsconfig.json");
        fs::write(&tsconfig_path, "{}").unwrap();

        // Create a source file in components/
        let source_file = components_dir.join("Button.tsx");
        fs::write(&source_file, "export const Button = () => {}").unwrap();

        // Should find the tsconfig.json in temp/
        let found = PathAliasMap::find_nearest_tsconfig(&source_file);
        assert_eq!(found, Some(tsconfig_path));
    }

    #[test]
    fn test_resolve_relative_to_config() {
        let temp = TempDir::new().unwrap();
        let alias_map = PathAliasMap {
            aliases: HashMap::new(),
            base_url: Some("src".to_string()),
            config_dir: temp.path().to_path_buf(),
        };

        let resolved = alias_map.resolve_relative_to_config("utils/helper.ts");
        assert_eq!(resolved, temp.path().join("src/utils/helper.ts"));
    }
}
