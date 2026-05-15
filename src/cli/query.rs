use anyhow::Result;
use std::time::Instant;
use indicatif::{ProgressBar, ProgressStyle};
use owo_colors::OwoColorize;
use crate::cache::CacheManager;
use crate::models::Language;
use crate::query::{QueryEngine, QueryFilter};


/// Smart truncate preview to reduce token usage
/// Truncates at word boundary if possible, adds ellipsis if truncated
pub fn truncate_preview(preview: &str, max_length: usize) -> String {
    if preview.len() <= max_length {
        return preview.to_string();
    }

    // Find a good break point (prefer word boundary)
    let truncate_at = preview.char_indices()
        .take(max_length)
        .filter(|(_, c)| c.is_whitespace())
        .last()
        .map(|(i, _)| i)
        .unwrap_or(max_length.min(preview.len()));

    let mut truncated = preview[..truncate_at].to_string();
    truncated.push('…');
    truncated
}


/// Handle the `query` subcommand
pub(super) fn handle_query(
    pattern: String,
    symbols_flag: bool,
    lang: Option<String>,
    kind_str: Option<String>,
    use_ast: bool,
    use_regex: bool,
    as_json: bool,
    pretty_json: bool,
    ai_mode: bool,
    limit: Option<usize>,
    offset: Option<usize>,
    expand: bool,
    file_pattern: Option<String>,
    exact: bool,
    use_contains: bool,
    count_only: bool,
    timeout_secs: u64,
    plain: bool,
    glob_patterns: Vec<String>,
    exclude_patterns: Vec<String>,
    paths_only: bool,
    no_truncate: bool,
    context_arg: Option<usize>,
    all: bool,
    force: bool,
    include_dependencies: bool,
) -> Result<()> {
    log::info!("Starting query command");

    // AI mode implies JSON output
    let as_json = as_json || ai_mode;

    let cache = CacheManager::new(".");
    let engine = QueryEngine::new(cache);

    // Parse and validate language filter
    let language = if let Some(lang_str) = lang.as_deref() {
        match Language::from_name(lang_str) {
            Some(l) => Some(l),
            None => anyhow::bail!(
                "Unknown language: '{}'\n\nSupported languages:\n  {}\n\nExample: rfx query \"pattern\" --lang rust",
                lang_str, Language::supported_names_help()
            ),
        }
    } else {
        None
    };

    // Parse and validate symbol kind — error on unrecognised values (REF-60)
    let kind = if let Some(s) = kind_str.as_deref() {
        let capitalized = {
            let mut chars = s.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().chain(chars.flat_map(|c| c.to_lowercase())).collect(),
            }
        };
        let parsed = capitalized.parse::<crate::models::SymbolKind>()
            .unwrap_or(crate::models::SymbolKind::Unknown(s.to_string()));
        if let crate::models::SymbolKind::Unknown(_) = &parsed {
            anyhow::bail!(
                "Unknown symbol kind: '{}'\n\nSupported kinds:\n  function, class, struct, enum, interface, trait, \
                constant, variable, method, module, namespace, type, macro, property, event, import, export, attribute\n\n\
                Example: rfx query \"parse\" --kind function",
                s
            );
        }
        Some(parsed)
    } else {
        None
    };

    // Smart behavior: --kind implies --symbols
    let symbols_mode = symbols_flag || kind.is_some();

    // --limit 0 is rejected; use --all for unlimited results
    if limit == Some(0) {
        anyhow::bail!(
            "--limit 0 is not valid. To return all results use --all (or -a).\n\
             To return exactly 0 results is meaningless; omit --limit to use the default of 100."
        );
    }

    // Smart limit handling:
    // 1. If --count is set: no limit (count should always show total)
    // 2. If --all is set: no limit (None)
    // 3. If --paths is set and user didn't specify --limit: no limit (None)
    // 4. If user specified --limit: use that value
    // 5. Otherwise: use default limit of 100
    let final_limit = if count_only {
        None  // --count always shows total count, no pagination
    } else if all {
        None  // --all means no limit
    } else if paths_only && limit.is_none() {
        None  // --paths without explicit --limit means no limit
    } else if let Some(user_limit) = limit {
        Some(user_limit)  // Use user-specified limit
    } else {
        Some(100)  // Default: limit to 100 results for token efficiency
    };

    // Validate AST query requirements
    if use_ast && language.is_none() {
        anyhow::bail!(
            "AST pattern matching requires a language to be specified.\n\
             \n\
             Use --lang to specify the language for tree-sitter parsing.\n\
             \n\
             Supported languages for AST queries:\n\
             • rust, python, go, java, c, c++, c#, php, ruby, kotlin, zig, typescript, javascript\n\
             \n\
             Note: Vue and Svelte use line-based parsing and do not support AST queries.\n\
             \n\
             WARNING: AST queries are SLOW (500ms-2s+). Use --symbols instead for 95% of cases.\n\
             \n\
             Examples:\n\
             • rfx query \"(function_definition) @fn\" --ast --lang python\n\
             • rfx query \"(class_declaration) @class\" --ast --lang typescript --glob \"src/**/*.ts\""
        );
    }

    // VALIDATION: Check for conflicting or problematic flag combinations
    // Only show warnings/errors in non-JSON mode (avoid breaking parsers)
    if !as_json {
        let mut has_errors = false;

        // ERROR: Mutually exclusive pattern matching modes
        if use_regex && use_contains {
            eprintln!("{}", "ERROR: Cannot use --regex and --contains together.".red().bold());
            eprintln!("  {} --regex for pattern matching (alternation, wildcards, etc.)", "•".dimmed());
            eprintln!("  {} --contains for substring matching (expansive search)", "•".dimmed());
            eprintln!("\n  {} Choose one based on your needs:", "Tip:".cyan().bold());
            eprintln!("    {} for OR logic: --regex", "pattern1|pattern2".yellow());
            eprintln!("    {} for substring: --contains", "partial_text".yellow());
            has_errors = true;
        }

        // ERROR: Contradictory matching requirements
        if exact && use_contains {
            eprintln!("{}", "ERROR: Cannot use --exact and --contains together (contradictory).".red().bold());
            eprintln!("  {} --exact requires exact symbol name match", "•".dimmed());
            eprintln!("  {} --contains allows substring matching", "•".dimmed());
            has_errors = true;
        }

        // WARNING: Redundant file filtering
        if file_pattern.is_some() && !glob_patterns.is_empty() {
            eprintln!("{}", "WARNING: Both --file and --glob specified.".yellow().bold());
            eprintln!("  {} --file does substring matching on file paths", "•".dimmed());
            eprintln!("  {} --glob does pattern matching with wildcards", "•".dimmed());
            eprintln!("  {} Both filters will apply (AND condition)", "Note:".dimmed());
            eprintln!("\n  {} Usually you only need one:", "Tip:".cyan().bold());
            eprintln!("    {} for simple matching", "--file User.php".yellow());
            eprintln!("    {} for pattern matching", "--glob src/**/*.php".yellow());
        }

        // INFO: Detect potentially problematic glob patterns
        for pattern in &glob_patterns {
            // Check for literal quotes in pattern
            if (pattern.starts_with('\'') && pattern.ends_with('\'')) ||
               (pattern.starts_with('"') && pattern.ends_with('"')) {
                eprintln!("{}",
                    format!("WARNING: Glob pattern contains quotes: {}", pattern).yellow().bold()
                );
                eprintln!("  {} Shell quotes should not be part of the pattern", "Note:".dimmed());
                eprintln!("  {} --glob src/**/*.rs", "Correct:".green());
                eprintln!("  {} --glob 'src/**/*.rs'", "Wrong:".red().dimmed());
            }

            // Suggest using ** instead of * for recursive matching
            if pattern.contains("*/") && !pattern.contains("**/") {
                eprintln!("{}",
                    format!("INFO: Glob '{}' uses * (matches one directory level)", pattern).cyan()
                );
                eprintln!("  {} Use ** for recursive matching across subdirectories", "Tip:".cyan().bold());
                eprintln!("    {} → matches files in Models/ only", "app/Models/*.php".yellow());
                eprintln!("    {} → matches files in Models/ and subdirs", "app/Models/**/*.php".green());
            }
        }

        if has_errors {
            anyhow::bail!("Invalid flag combination. Fix the errors above and try again.");
        }

        // REF-58: Warn when --file looks like a concrete path that doesn't exist on disk
        if let Some(ref fp) = file_pattern {
            // Only warn when it looks like a literal path (no glob wildcards, has extension or slash)
            let looks_like_path = !fp.contains('*') && !fp.contains('?') &&
                (fp.contains('/') || fp.contains('.'));
            if looks_like_path && !std::path::Path::new(fp).exists() {
                eprintln!("{}", format!("[warn] --file path not found on disk: {}", fp).yellow());
                eprintln!("  Continuing with substring match — results will be empty if no indexed path contains '{}'.", fp);
            }
        }
    }

    // Clamp context lines to a sane max (10) to avoid huge output
    let context_lines = context_arg.map(|n| n.min(10)).unwrap_or(0);

    let filter = QueryFilter {
        language,
        kind,
        use_ast,
        use_regex,
        limit: final_limit,
        symbols_mode,
        expand,
        file_pattern,
        exact,
        use_contains,
        timeout_secs,
        glob_patterns: glob_patterns.clone(),
        exclude_patterns,
        paths_only,
        offset,
        force,
        suppress_output: as_json,  // Suppress warnings in JSON mode
        include_dependencies,
        context_lines,
        ..Default::default()
    };

    // Measure query time
    let start = Instant::now();

    // Execute query and get pagination metadata
    // Handle errors specially for JSON output mode
    let (query_response, mut flat_results, total_results, has_more) = if use_ast {
        // AST query: pattern is the S-expression, scan all files
        match engine.search_ast_all_files(&pattern, filter.clone()) {
            Ok(ast_results) => {
                let count = ast_results.len();
                (None, ast_results, count, false)
            }
            Err(e) => {
                if as_json {
                    // Output error as JSON
                    let error_response = serde_json::json!({
                        "error": e.to_string(),
                        "query_too_broad": e.to_string().contains("Query too broad")
                    });
                    let json_output = if pretty_json {
                        serde_json::to_string_pretty(&error_response)?
                    } else {
                        serde_json::to_string(&error_response)?
                    };
                    println!("{}", json_output);
                    std::process::exit(1);
                } else {
                    return Err(e);
                }
            }
        }
    } else {
        // Use metadata-aware search for all queries (to get pagination info)
        match engine.search_with_metadata(&pattern, filter.clone()) {
            Ok(response) => {
                let total = response.pagination.total;
                let has_more = response.pagination.has_more;

                // Flatten grouped results to SearchResult vec for plain text formatting
                let flat = response.results.iter()
                    .flat_map(|file_group| {
                        file_group.matches.iter().map(move |m| {
                            crate::models::SearchResult {
                                path: file_group.path.clone(),
                                lang: crate::models::Language::Unknown, // Will be set by formatter if needed
                                kind: m.kind.clone(),
                                symbol: m.symbol.clone(),
                                span: m.span.clone(),
                                preview: m.preview.clone(),
                                dependencies: file_group.dependencies.clone(),
                            }
                        })
                    })
                    .collect();

                (Some(response), flat, total, has_more)
            }
            Err(e) => {
                if as_json {
                    // Output error as JSON
                    let error_response = serde_json::json!({
                        "error": e.to_string(),
                        "query_too_broad": e.to_string().contains("Query too broad")
                    });
                    let json_output = if pretty_json {
                        serde_json::to_string_pretty(&error_response)?
                    } else {
                        serde_json::to_string(&error_response)?
                    };
                    println!("{}", json_output);
                    std::process::exit(1);
                } else {
                    return Err(e);
                }
            }
        }
    };

    // Apply preview truncation unless --no-truncate is set
    if !no_truncate {
        const MAX_PREVIEW_LENGTH: usize = 100;
        for result in &mut flat_results {
            result.preview = truncate_preview(&result.preview, MAX_PREVIEW_LENGTH);
        }
    }

    let elapsed = start.elapsed();

    // Format timing string
    let timing_str = if elapsed.as_millis() < 1 {
        format!("{:.1}ms", elapsed.as_secs_f64() * 1000.0)
    } else {
        format!("{}ms", elapsed.as_millis())
    };

    if as_json {
        if count_only {
            // Count-only JSON mode: output simple count object
            let count_response = serde_json::json!({
                "count": total_results,
                "timing_ms": elapsed.as_millis()
            });
            let json_output = if pretty_json {
                serde_json::to_string_pretty(&count_response)?
            } else {
                serde_json::to_string(&count_response)?
            };
            println!("{}", json_output);
        } else if paths_only {
            // Paths-only JSON mode: output deduplicated array of path strings (REF-62)
            let mut seen = std::collections::HashSet::new();
            let unique_paths: Vec<String> = flat_results.iter()
                .filter_map(|r| {
                    if seen.insert(r.path.clone()) { Some(r.path.clone()) } else { None }
                })
                .collect();

            let json_output = if ai_mode {
                // REF-59: wrap with ai_instruction when --ai flag is used
                let ai_instruction = crate::query::generate_ai_instruction(
                    unique_paths.len(),
                    total_results,
                    has_more,
                    symbols_mode,
                    true,
                    use_ast,
                    use_regex,
                    language.is_some(),
                    !glob_patterns.is_empty(),
                    exact,
                );
                let wrapper = serde_json::json!({
                    "ai_instruction": ai_instruction,
                    "count": unique_paths.len(),
                    "results": unique_paths,
                });
                if pretty_json { serde_json::to_string_pretty(&wrapper)? } else { serde_json::to_string(&wrapper)? }
            } else {
                if pretty_json {
                    serde_json::to_string_pretty(&unique_paths)?
                } else {
                    serde_json::to_string(&unique_paths)?
                }
            };
            println!("{}", json_output);
        } else {
            // Get or build QueryResponse for JSON output
            let mut response = if let Some(resp) = query_response {
                // We already have a response from search_with_metadata
                // Apply truncation to the response (the flat_results were already truncated)
                let mut resp = resp;

                // Apply truncation to results
                if !no_truncate {
                    const MAX_PREVIEW_LENGTH: usize = 100;
                    for file_group in resp.results.iter_mut() {
                        for m in file_group.matches.iter_mut() {
                            m.preview = truncate_preview(&m.preview, MAX_PREVIEW_LENGTH);
                        }
                    }
                }

                resp
            } else {
                // For AST queries, build a response with minimal metadata
                // Group flat results by file path
                use crate::models::{PaginationInfo, IndexStatus, FileGroupedResult, MatchResult};
                use std::collections::HashMap;

                let mut grouped: HashMap<String, Vec<crate::models::SearchResult>> = HashMap::new();
                for result in &flat_results {
                    grouped
                        .entry(result.path.clone())
                        .or_default()
                        .push(result.clone());
                }

                // Load ContentReader for extracting context lines
                use crate::content_store::ContentReader;
                let local_cache = CacheManager::new(".");
                let content_path = local_cache.path().join("content.bin");
                let content_reader_opt = ContentReader::open(&content_path).ok();

                let mut file_results: Vec<FileGroupedResult> = grouped
                    .into_iter()
                    .map(|(path, file_matches)| {
                        // Get file_id for context extraction
                        // Note: We use ContentReader's get_file_id_by_path() which returns array indices,
                        // not database file_ids (which are AUTO INCREMENT values)
                        let normalized_path = path.strip_prefix("./").unwrap_or(&path);
                        let file_id_for_context = if let Some(reader) = &content_reader_opt {
                            reader.get_file_id_by_path(normalized_path)
                        } else {
                            None
                        };

                        let language = file_matches.first().map(|r| r.lang).unwrap_or_default();
                        let matches: Vec<MatchResult> = file_matches
                            .into_iter()
                            .map(|r| {
                                // Extract context lines (default: 3 lines before and after)
                                let (context_before, context_after) = if let (Some(reader), Some(fid)) = (&content_reader_opt, file_id_for_context) {
                                    reader.get_context_by_line(fid as u32, r.span.start_line, 3)
                                        .unwrap_or_else(|_| (vec![], vec![]))
                                } else {
                                    (vec![], vec![])
                                };

                                MatchResult {
                                    kind: r.kind,
                                    symbol: r.symbol,
                                    span: r.span,
                                    preview: r.preview,
                                    context_before,
                                    context_after,
                                }
                            })
                            .collect();
                        FileGroupedResult {
                            path,
                            language,
                            dependencies: None,
                            matches,
                        }
                    })
                    .collect();

                // Sort by path for deterministic output
                file_results.sort_by(|a, b| a.path.cmp(&b.path));

                crate::models::QueryResponse {
                    ai_instruction: None,  // Will be populated below if ai_mode is true
                    status: IndexStatus::Fresh,
                    can_trust_results: true,
                    warning: None,
                    pagination: PaginationInfo {
                        total: flat_results.len(),
                        count: flat_results.len(),
                        offset: offset.unwrap_or(0),
                        limit,
                        has_more: false, // AST already applied pagination
                    },
                    results: file_results,
                }
            };

            // Generate AI instruction if in AI mode
            if ai_mode {
                let result_count: usize = response.results.iter().map(|fg| fg.matches.len()).sum();

                response.ai_instruction = crate::query::generate_ai_instruction(
                    result_count,
                    response.pagination.total,
                    response.pagination.has_more,
                    symbols_mode,
                    paths_only,
                    use_ast,
                    use_regex,
                    language.is_some(),
                    !glob_patterns.is_empty(),
                    exact,
                );
            }

            let json_output = if pretty_json {
                serde_json::to_string_pretty(&response)?
            } else {
                serde_json::to_string(&response)?
            };
            println!("{}", json_output);

            let result_count: usize = response.results.iter().map(|fg| fg.matches.len()).sum();
            eprintln!("Found {} result{} in {}", result_count, if result_count == 1 { "" } else { "s" }, timing_str);
        }
    } else {
        // Standard output with formatting
        if count_only {
            let n = flat_results.len();
            println!("Found {} result{} in {}", n, if n == 1 { "" } else { "s" }, timing_str);
            return Ok(());
        }

        if paths_only {
            // Paths-only plain text mode: output one path per line
            if flat_results.is_empty() {
                eprintln!("No results found (searched in {}).", timing_str);
            } else {
                for result in &flat_results {
                    println!("{}", result.path);
                }
                let n = flat_results.len();
                eprintln!("Found {} unique file{} in {}", n, if n == 1 { "" } else { "s" }, timing_str);
            }
        } else {
            // Standard result formatting
            if flat_results.is_empty() {
                println!("No results found (searched in {}).", timing_str);
            } else {
                // Use formatter for pretty output
                let formatter = crate::formatter::OutputFormatter::new(plain);
                formatter.format_results(&flat_results, &pattern)?;

                // Print summary at the bottom with pagination details
                let n = flat_results.len();
                if total_results > n {
                    // Results were paginated - show detailed count
                    println!("\nFound {} result{} ({} total) in {}", n, if n == 1 { "" } else { "s" }, total_results, timing_str);
                    // Show pagination hint if there are more results available
                    if has_more {
                        println!("Use --limit and --offset to paginate");
                    }
                } else {
                    // All results shown - simple count
                    println!("\nFound {} result{} in {}", n, if n == 1 { "" } else { "s" }, timing_str);
                }
            }
        }
    }

    Ok(())
}


/// Handle interactive mode (default when no command is given)
pub(super) fn handle_interactive() -> Result<()> {
    log::info!("Launching interactive mode");
    crate::interactive::run_interactive()
}
