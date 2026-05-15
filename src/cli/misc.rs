use anyhow::{Context, Result};
use crate::cache::CacheManager;


/// Handle the `stats` subcommand
pub(super) fn handle_stats(as_json: bool, pretty_json: bool) -> Result<()> {
    log::info!("Showing index statistics");

    let cache = CacheManager::new(".");

    if !cache.exists() {
        anyhow::bail!(
            "No index found in current directory.\n\
             \n\
             Run 'rfx index' to build the code search index first.\n\
             This will scan all files in the current directory and create a .reflex/ cache.\n\
             \n\
             Example:\n\
             $ rfx index          # Index current directory\n\
             $ rfx stats          # Show index statistics"
        );
    }

    let stats = cache.stats().with_context(|| {
        "Failed to read index statistics.\n\
         The cache may be corrupted. To recover:\n\
         $ rfx index              # Rebuild the index\n\
         $ rfx clear && rfx index # Clear and rebuild from scratch"
    })?;

    // Read trigram count from trigrams.bin header (magic + version + num_trigrams + num_files = 24 bytes)
    let trigram_count = {
        let trigrams_path = cache.path().join("trigrams.bin");
        if trigrams_path.exists() {
            use std::io::Read;
            std::fs::File::open(&trigrams_path)
                .ok()
                .and_then(|mut f| {
                    let mut header = [0u8; 24];
                    f.read_exact(&mut header).ok()?;
                    if &header[..4] == b"RFTG" {
                        Some(u64::from_le_bytes([
                            header[8], header[9], header[10], header[11],
                            header[12], header[13], header[14], header[15],
                        ]))
                    } else {
                        None
                    }
                })
                .unwrap_or(0)
        } else {
            0
        }
    };

    if as_json {
        let json_output = if pretty_json {
            serde_json::to_string_pretty(&stats)?
        } else {
            serde_json::to_string(&stats)?
        };
        println!("{}", json_output);
    } else {
        println!("Reflex Index Statistics");
        println!("=======================");

        // Show git branch info if in git repo, or (None) if not
        let root = std::env::current_dir()?;
        if crate::git::is_git_repo(&root) {
            match crate::git::get_git_state(&root) {
                Ok(git_state) => {
                    let dirty_indicator = if git_state.dirty { " (uncommitted changes)" } else { " (clean)" };
                    println!("Branch:         {}@{}{}",
                             git_state.branch,
                             &git_state.commit[..7],
                             dirty_indicator);

                    // Check if current branch is indexed
                    match cache.get_branch_info(&git_state.branch) {
                        Ok(branch_info) => {
                            if branch_info.commit_sha != git_state.commit {
                                println!("                ⚠️  Index commit mismatch (indexed: {})",
                                         &branch_info.commit_sha[..7]);
                            }
                            if git_state.dirty && !branch_info.is_dirty {
                                println!("                ⚠️  Uncommitted changes not indexed");
                            }
                        }
                        Err(_) => {
                            println!("                ⚠️  Branch not indexed");
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Failed to get git state: {}", e);
                }
            }
        } else {
            // Not a git repository - show (None)
            println!("Branch:         (None)");
        }

        println!("Files indexed:  {}", stats.total_files);
        println!("Index size:     {}", super::format_bytes(stats.index_size_bytes));
        if trigram_count > 0 {
            println!("Trigrams:       {}", trigram_count);
        }
        println!("Last updated:   {}", stats.last_updated);

        // Display language breakdown if we have indexed files
        if !stats.files_by_language.is_empty() {
            println!("\nFiles by language:");

            // Sort languages by count (descending) for consistent output
            let mut lang_vec: Vec<_> = stats.files_by_language.iter().collect();
            lang_vec.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));

            // Calculate column widths
            let max_lang_len = lang_vec.iter().map(|(lang, _)| lang.len()).max().unwrap_or(8);
            let lang_width = max_lang_len.max(8); // At least "Language" header width

            // Print table header
            println!("  {:<width$}  Files  Lines", "Language", width = lang_width);
            println!("  {}  -----  -------", "-".repeat(lang_width));

            // Print rows
            for (language, file_count) in lang_vec {
                let line_count = stats.lines_by_language.get(language).copied().unwrap_or(0);
                println!("  {:<width$}  {:5}  {:7}",
                    language, file_count, line_count,
                    width = lang_width);
            }
        }
    }

    Ok(())
}


/// Handle the `clear` subcommand
pub(super) fn handle_clear(skip_confirm: bool) -> Result<()> {
    let cache = CacheManager::new(".");

    if !cache.exists() {
        println!("No cache to clear.");
        return Ok(());
    }

    if !skip_confirm {
        println!("This will delete the local Reflex cache at: {:?}", cache.path());
        print!("Are you sure? [y/N] ");
        use std::io::{self, Write};
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Cancelled.");
            return Ok(());
        }
    }

    cache.clear()?;
    println!("Cache cleared successfully.");

    Ok(())
}


/// Handle the `list-files` subcommand
pub(super) fn handle_list_files(
    as_json: bool,
    pretty_json: bool,
    lang_filter: Option<String>,
    glob_patterns: Vec<String>,
) -> Result<()> {
    use crate::models::Language;

    let cache = CacheManager::new(".");

    if !cache.exists() {
        anyhow::bail!(
            "No index found in current directory.\n\
             \n\
             Run 'rfx index' to build the code search index first.\n\
             This will scan all files in the current directory and create a .reflex/ cache.\n\
             \n\
             Example:\n\
             $ rfx index            # Index current directory\n\
             $ rfx list-files       # List indexed files"
        );
    }

    // Validate --lang if provided (REF-94)
    let lang_name_filter: Option<String> = if let Some(ref lang_str) = lang_filter {
        if Language::from_name(lang_str).is_none() {
            anyhow::bail!(
                "Unknown language: '{}'\n\nSupported languages:\n  {}\n\nExample: rfx list-files --lang rust",
                lang_str, Language::supported_names_help()
            );
        }
        // Normalise to the canonical lowercase name stored in the DB
        Some(lang_str.to_lowercase())
    } else {
        None
    };

    // Build globset for --glob filtering
    let glob_set = if !glob_patterns.is_empty() {
        let mut builder = globset::GlobSetBuilder::new();
        for pat in &glob_patterns {
            builder.add(globset::Glob::new(pat).with_context(|| format!("Invalid glob pattern: {}", pat))?);
        }
        Some(builder.build().context("Failed to build glob set")?)
    } else {
        None
    };

    let all_files = cache.list_files()?;

    let files: Vec<_> = all_files.into_iter().filter(|f| {
        // Language filter: compare lowercase language name
        if let Some(ref wanted_lang) = lang_name_filter {
            if !f.language.to_lowercase().starts_with(wanted_lang.as_str()) &&
               f.language.to_lowercase() != *wanted_lang {
                return false;
            }
        }
        // Glob filter
        if let Some(ref gs) = glob_set {
            if !gs.is_match(&f.path) {
                return false;
            }
        }
        true
    }).collect();

    if as_json {
        let total = files.len();
        let envelope = serde_json::json!({ "files": files, "total": total });
        let json_output = if pretty_json {
            serde_json::to_string_pretty(&envelope)?
        } else {
            serde_json::to_string(&envelope)?
        };
        println!("{}", json_output);
    } else if files.is_empty() {
        println!("No files indexed yet.");
    } else {
        println!("Indexed Files ({} total):", files.len());
        println!();
        for file in &files {
            println!("  {} ({})", file.path, file.language);
        }
    }

    Ok(())
}


/// Handle the `mcp` subcommand
pub(super) fn handle_mcp() -> Result<()> {
    log::info!("Starting MCP server");
    crate::mcp::run_mcp_server()
}


/// Handle the `context` command
pub(super) fn handle_context(
    structure: bool,
    path: Option<String>,
    file_types: bool,
    project_type: bool,
    framework: bool,
    entry_points: bool,
    test_layout: bool,
    config_files: bool,
    depth: usize,
    json: bool,
) -> Result<()> {
    let cache = CacheManager::new(".");

    if !cache.exists() {
        anyhow::bail!(
            "No index found in current directory.\n\
             \n\
             Run 'rfx index' to build the code search index first.\n\
             \n\
             Example:\n\
             $ rfx index                  # Index current directory\n\
             $ rfx context                # Generate context"
        );
    }

    // Build context options
    let opts = crate::context::ContextOptions {
        structure,
        path,
        file_types,
        project_type,
        framework,
        entry_points,
        test_layout,
        config_files,
        depth,
        json,
    };

    // Generate context
    let context_output = crate::context::generate_context(&cache, &opts)
        .context("Failed to generate codebase context")?;

    // Print output
    println!("{}", context_output);

    Ok(())
}
