//! Command parser and query executor for semantic queries

use anyhow::{Context, Result};
use std::collections::HashSet;

use crate::cache::CacheManager;
use crate::models::{FileGroupedResult, Language, SymbolKind};
use crate::query::{QueryEngine, QueryFilter};

use super::schema::QueryCommand;

/// Parse a command string into query parameters
///
/// The command string should be in the format:
/// `query "pattern" [flags...]`
///
/// Example: `query "TODO" --symbols --lang rust`
pub fn parse_command(command: &str) -> Result<ParsedCommand> {
    // Parse the command using shell-words to handle quoted strings
    let parts = shell_words::split(command).context("Failed to parse command string")?;

    if parts.is_empty() {
        anyhow::bail!("Empty command string");
    }

    // First word should be "query"
    if parts[0] != "query" {
        anyhow::bail!("Command must start with 'query', got '{}'", parts[0]);
    }

    if parts.len() < 2 {
        anyhow::bail!("Missing search pattern in query command");
    }

    // Second word is the pattern
    let pattern = parts[1].clone();

    // Parse remaining arguments as flags
    let mut parsed = ParsedCommand {
        pattern,
        symbols: false,
        lang: None,
        kind: None,
        use_ast: false,
        use_regex: false,
        limit: None,
        offset: None,
        expand: false,
        file: None,
        exact: false,
        contains: false,
        glob: Vec::new(),
        exclude: Vec::new(),
        paths: false,
        all: false,
        force: false,
        dependencies: false,
        count: false,
    };

    let mut i = 2;
    while i < parts.len() {
        match parts[i].as_str() {
            "--symbols" | "-s" => {
                parsed.symbols = true;
                i += 1;
            }
            "--lang" | "-l" => {
                if i + 1 >= parts.len() {
                    anyhow::bail!("--lang requires a value");
                }
                parsed.lang = Some(parts[i + 1].clone());
                i += 2;
            }
            "--kind" | "-k" => {
                if i + 1 >= parts.len() {
                    anyhow::bail!("--kind requires a value");
                }
                parsed.kind = Some(parts[i + 1].clone());
                i += 2;
            }
            "--ast" => {
                parsed.use_ast = true;
                i += 1;
            }
            "--regex" | "-r" => {
                parsed.use_regex = true;
                i += 1;
            }
            "--limit" | "-n" => {
                if i + 1 >= parts.len() {
                    anyhow::bail!("--limit requires a value");
                }
                let limit_val: usize = parts[i + 1].parse().context("--limit must be a number")?;
                parsed.limit = Some(limit_val);
                i += 2;
            }
            "--offset" | "-o" => {
                if i + 1 >= parts.len() {
                    anyhow::bail!("--offset requires a value");
                }
                let offset_val: usize =
                    parts[i + 1].parse().context("--offset must be a number")?;
                parsed.offset = Some(offset_val);
                i += 2;
            }
            "--expand" => {
                parsed.expand = true;
                i += 1;
            }
            "--file" | "-f" => {
                if i + 1 >= parts.len() {
                    anyhow::bail!("--file requires a value");
                }
                parsed.file = Some(parts[i + 1].clone());
                i += 2;
            }
            "--exact" => {
                parsed.exact = true;
                i += 1;
            }
            "--contains" => {
                parsed.contains = true;
                i += 1;
            }
            "--glob" | "-g" => {
                if i + 1 >= parts.len() {
                    anyhow::bail!("--glob requires a value");
                }
                parsed.glob.push(parts[i + 1].clone());
                i += 2;
            }
            "--exclude" | "-x" => {
                if i + 1 >= parts.len() {
                    anyhow::bail!("--exclude requires a value");
                }
                parsed.exclude.push(parts[i + 1].clone());
                i += 2;
            }
            "--paths" | "-p" => {
                parsed.paths = true;
                i += 1;
            }
            "--all" | "-a" => {
                parsed.all = true;
                i += 1;
            }
            "--force" => {
                parsed.force = true;
                i += 1;
            }
            "--dependencies" => {
                parsed.dependencies = true;
                i += 1;
            }
            "--count" | "-c" => {
                parsed.count = true;
                i += 1;
            }
            unknown => {
                log::debug!("Ignoring unknown flag: {}", unknown);
                i += 1;
            }
        }
    }

    Ok(parsed)
}

/// Parsed command structure
#[derive(Debug, Clone)]
pub struct ParsedCommand {
    pub pattern: String,
    pub symbols: bool,
    pub lang: Option<String>,
    pub kind: Option<String>,
    pub use_ast: bool,
    pub use_regex: bool,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub expand: bool,
    pub file: Option<String>,
    pub exact: bool,
    pub contains: bool,
    pub glob: Vec<String>,
    pub exclude: Vec<String>,
    pub paths: bool,
    pub all: bool,
    pub force: bool,
    pub dependencies: bool,
    pub count: bool,
}

impl ParsedCommand {
    /// Convert to QueryFilter
    pub fn to_query_filter(&self) -> Result<QueryFilter> {
        // Parse language
        let language = if let Some(lang_str) = &self.lang {
            match lang_str.to_lowercase().as_str() {
                "rust" | "rs" => Some(Language::Rust),
                "python" | "py" => Some(Language::Python),
                "javascript" | "js" => Some(Language::JavaScript),
                "typescript" | "ts" => Some(Language::TypeScript),
                "vue" => Some(Language::Vue),
                "svelte" => Some(Language::Svelte),
                "go" => Some(Language::Go),
                "java" => Some(Language::Java),
                "php" => Some(Language::PHP),
                "c" => Some(Language::C),
                "cpp" | "c++" => Some(Language::Cpp),
                "csharp" | "cs" | "c#" => Some(Language::CSharp),
                "ruby" | "rb" => Some(Language::Ruby),
                "kotlin" | "kt" => Some(Language::Kotlin),
                "swift" => Some(Language::Swift),
                "zig" => Some(Language::Zig),
                _ => anyhow::bail!("Unknown language: {}", lang_str),
            }
        } else {
            None
        };

        // Parse symbol kind
        let kind = if let Some(kind_str) = &self.kind {
            // Capitalize first letter for parsing
            let capitalized = {
                let mut chars = kind_str.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first
                        .to_uppercase()
                        .chain(chars.flat_map(|c| c.to_lowercase()))
                        .collect(),
                }
            };

            let parsed_kind: SymbolKind = capitalized
                .parse()
                .ok()
                .or_else(|| {
                    log::debug!("Treating '{}' as unknown symbol kind", kind_str);
                    Some(SymbolKind::Unknown(kind_str.to_string()))
                })
                .context("Failed to parse symbol kind")?;

            Some(parsed_kind)
        } else {
            None
        };

        // Symbol mode is enabled if --symbols flag OR --kind is specified
        let symbols_mode = self.symbols || self.kind.is_some();

        // Handle --all flag (unlimited results)
        let limit = if self.all { None } else { self.limit };

        Ok(QueryFilter {
            language,
            kind,
            use_ast: self.use_ast,
            use_regex: self.use_regex,
            limit,
            symbols_mode,
            expand: self.expand,
            file_pattern: self.file.clone(),
            exact: self.exact,
            use_contains: self.contains,
            timeout_secs: 30, // Default timeout
            glob_patterns: self.glob.clone(),
            exclude_patterns: self.exclude.clone(),
            paths_only: self.paths,
            offset: self.offset,
            force: self.force,
            suppress_output: true, // Suppress output for programmatic use
            include_dependencies: self.dependencies,
            ..Default::default()
        })
    }
}

/// Execute multiple queries with ordering and merging
///
/// Queries are executed in order based on their `order` field.
/// Results are merged based on the `merge` flag - only queries with `merge: true`
/// contribute to the final result set.
///
/// Results are deduplicated by (file_path, start_line, end_line) to avoid duplicates
/// across multiple queries.
///
/// Returns a tuple of (merged results, total count across all queries, count_only mode).
/// If count_only is true, all queries had --count flag and only the count should be displayed.
pub async fn execute_queries(
    queries: Vec<QueryCommand>,
    cache: &CacheManager,
) -> Result<(Vec<FileGroupedResult>, usize, bool)> {
    if queries.is_empty() {
        return Ok((Vec::new(), 0, false));
    }

    // Sort queries by order field
    let mut sorted_queries = queries.clone();
    sorted_queries.sort_by_key(|q| q.order);

    log::info!("Executing {} queries in order", sorted_queries.len());

    let mut merged_results: Vec<FileGroupedResult> = Vec::new();
    let mut seen_matches: HashSet<(String, usize, usize)> = HashSet::new();
    let mut total_count: usize = 0;
    let mut all_count_only = true;

    // Create a single QueryEngine and reuse it for all queries
    // This avoids redundant cache validation and SQLite connection overhead
    let engine = QueryEngine::new(cache.clone());

    for query_cmd in sorted_queries {
        log::debug!("Executing query {}: {}", query_cmd.order, query_cmd.command);

        // Parse command
        let parsed = parse_command(&query_cmd.command)
            .with_context(|| format!("Failed to parse query command: {}", query_cmd.command))?;

        // Track if this query has --count flag
        if !parsed.count {
            all_count_only = false;
        }

        // Convert to QueryFilter
        let filter = parsed.to_query_filter()?;

        // Execute query (reusing the same engine)
        let response = engine
            .search_with_metadata(&parsed.pattern, filter)
            .with_context(|| format!("Failed to execute query: {}", query_cmd.command))?;

        // Always accumulate total count from all queries
        total_count += response.pagination.total;

        log::debug!(
            "Query {} returned {} file groups, {} total matches (merge={})",
            query_cmd.order,
            response.results.len(),
            response.pagination.total,
            query_cmd.merge
        );

        // If merge is true, add results to merged set (with deduplication)
        if query_cmd.merge {
            for file_group in response.results {
                // Find or create file group in merged results
                let file_path = file_group.path.clone();

                let existing_group = merged_results.iter_mut().find(|g| g.path == file_path);

                if let Some(group) = existing_group {
                    // Add matches to existing group (deduplicate)
                    for match_result in file_group.matches {
                        let key = (
                            file_path.clone(),
                            match_result.span.start_line,
                            match_result.span.end_line,
                        );

                        if !seen_matches.contains(&key) {
                            seen_matches.insert(key);
                            group.matches.push(match_result);
                        }
                    }
                } else {
                    // Create new group
                    for match_result in &file_group.matches {
                        let key = (
                            file_path.clone(),
                            match_result.span.start_line,
                            match_result.span.end_line,
                        );
                        seen_matches.insert(key);
                    }

                    merged_results.push(file_group);
                }
            }
        }
    }

    log::info!(
        "Merged results: {} file groups, {} unique matches, {} total count (count_only={})",
        merged_results.len(),
        seen_matches.len(),
        total_count,
        all_count_only
    );

    Ok((merged_results, total_count, all_count_only))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_query() {
        let cmd = r#"query "TODO""#;
        let parsed = parse_command(cmd).unwrap();

        assert_eq!(parsed.pattern, "TODO");
        assert!(!parsed.symbols);
        assert!(parsed.lang.is_none());
    }

    #[test]
    fn test_parse_query_with_flags() {
        let cmd = r#"query "extract_symbols" --symbols --lang rust"#;
        let parsed = parse_command(cmd).unwrap();

        assert_eq!(parsed.pattern, "extract_symbols");
        assert!(parsed.symbols);
        assert_eq!(parsed.lang, Some("rust".to_string()));
    }

    #[test]
    fn test_parse_query_with_kind() {
        let cmd = r#"query "main" --kind function --lang rust"#;
        let parsed = parse_command(cmd).unwrap();

        assert_eq!(parsed.pattern, "main");
        assert_eq!(parsed.kind, Some("function".to_string()));
        assert_eq!(parsed.lang, Some("rust".to_string()));
    }

    #[test]
    fn test_parse_query_with_glob() {
        let cmd = r#"query "TODO" --glob "src/**/*.rs" --glob "tests/**/*.rs""#;
        let parsed = parse_command(cmd).unwrap();

        assert_eq!(parsed.pattern, "TODO");
        assert_eq!(parsed.glob.len(), 2);
        assert_eq!(parsed.glob[0], "src/**/*.rs");
        assert_eq!(parsed.glob[1], "tests/**/*.rs");
    }

    #[test]
    fn test_parse_query_with_exclude() {
        let cmd = r#"query "config" --exclude "target/**" --exclude "*.gen.rs""#;
        let parsed = parse_command(cmd).unwrap();

        assert_eq!(parsed.pattern, "config");
        assert_eq!(parsed.exclude.len(), 2);
    }

    #[test]
    fn test_parse_invalid_command() {
        let cmd = r#"search "pattern""#;
        let result = parse_command(cmd);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("must start with 'query'")
        );
    }

    #[test]
    fn test_parse_empty_command() {
        let cmd = "";
        let result = parse_command(cmd);
        assert!(result.is_err());
    }

    #[test]
    fn test_to_query_filter() {
        let cmd = r#"query "TODO" --symbols --lang rust --limit 10"#;
        let parsed = parse_command(cmd).unwrap();
        let filter = parsed.to_query_filter().unwrap();

        assert_eq!(filter.language, Some(Language::Rust));
        assert!(filter.symbols_mode);
        assert_eq!(filter.limit, Some(10));
    }

    #[test]
    fn test_to_query_filter_with_kind() {
        let cmd = r#"query "parse" --kind function"#;
        let parsed = parse_command(cmd).unwrap();
        let filter = parsed.to_query_filter().unwrap();

        assert!(filter.symbols_mode); // kind implies symbols mode
        assert!(matches!(filter.kind, Some(SymbolKind::Function)));
    }
}
