use anyhow::{Context, Result};
use owo_colors::OwoColorize;
use std::sync::{Arc, Mutex};
use indicatif::{ProgressBar, ProgressStyle};
use crate::cache::CacheManager;


/// Handle the `ask` command
pub(super) fn handle_ask(
    question: Option<String>,
    auto_execute: bool,
    provider_override: Option<String>,
    as_json: bool,
    pretty_json: bool,
    additional_context: Option<String>,
    configure: bool,
    agentic: bool,
    max_iterations: usize,
    no_eval: bool,
    show_reasoning: bool,
    verbose: bool,
    quiet: bool,
    answer: bool,
    interactive: bool,
    debug: bool,
) -> Result<()> {
    // If --configure flag is set, launch the configuration wizard (deprecated)
    if configure {
        eprintln!("Note: --configure is deprecated, use `rfx llm config` instead");
        log::info!("Launching configuration wizard");
        return crate::semantic::run_configure_wizard();
    }

    // Check if any API key is configured before allowing rfx ask to run
    if !crate::semantic::is_any_api_key_configured() {
        anyhow::bail!(
            "No API key configured.\n\
             \n\
             Please run 'rfx ask --configure' to set up your API provider and key.\n\
             \n\
             Alternatively, you can set an environment variable:\n\
             - OPENAI_API_KEY\n\
             - ANTHROPIC_API_KEY\n\
             - OPENROUTER_API_KEY"
        );
    }

    // If no question provided and not in configure mode, default to interactive mode
    // If --interactive flag is set, launch interactive chat mode (TUI)
    if interactive || question.is_none() {
        log::info!("Launching interactive chat mode");
        let cache = CacheManager::new(".");

        if !cache.exists() {
            anyhow::bail!(
                "No index found in current directory.\n\
                 \n\
                 Run 'rfx index' to build the code search index first.\n\
                 \n\
                 Example:\n\
                 $ rfx index                          # Index current directory\n\
                 $ rfx ask                            # Launch interactive chat"
            );
        }

        return crate::semantic::run_chat_mode(cache, provider_override, None);
    }

    // At this point, question must be Some
    let question = question.unwrap();

    log::info!("Starting ask command");

    let cache = CacheManager::new(".");

    if !cache.exists() {
        anyhow::bail!(
            "No index found in current directory.\n\
             \n\
             Run 'rfx index' to build the code search index first.\n\
             \n\
             Example:\n\
             $ rfx index                          # Index current directory\n\
             $ rfx ask \"Find all TODOs\"          # Ask questions"
        );
    }

    // Create a tokio runtime for async operations
    let runtime = tokio::runtime::Runtime::new()
        .context("Failed to create async runtime")?;

    // Force quiet mode for JSON output (machine-readable, no UI output)
    let quiet = quiet || as_json;

    // Create optional spinner (skip entirely in JSON mode for clean machine-readable output)
    let spinner = if !as_json {
        let s = ProgressBar::new_spinner();
        s.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .unwrap()
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
        );
        s.set_message("Generating queries...".to_string());
        s.enable_steady_tick(std::time::Duration::from_millis(80));
        Some(s)
    } else {
        None
    };

    let (queries, results, total_count, count_only, gathered_context) = if agentic {
        // Agentic mode: multi-step reasoning with context gathering

        // Wrap spinner in Arc<Mutex<>> for sharing with reporter (non-quiet mode)
        let spinner_shared = if !quiet {
            spinner.as_ref().map(|s| Arc::new(Mutex::new(s.clone())))
        } else {
            None
        };

        // Create reporter based on flags
        let reporter: Box<dyn crate::semantic::AgenticReporter> = if quiet {
            Box::new(crate::semantic::QuietReporter)
        } else {
            Box::new(crate::semantic::ConsoleReporter::new(show_reasoning, verbose, debug, spinner_shared))
        };

        // Set initial spinner message and enable ticking
        if let Some(ref s) = spinner {
            s.set_message("Starting agentic mode...".to_string());
            s.enable_steady_tick(std::time::Duration::from_millis(80));
        }

        let agentic_config = crate::semantic::AgenticConfig {
            max_iterations,
            max_tools_per_phase: 5,
            enable_evaluation: !no_eval,
            eval_config: Default::default(),
            provider_override: provider_override.clone(),
            model_override: None,
            show_reasoning,
            verbose,
            debug,
        };

        let agentic_response = runtime.block_on(async {
            crate::semantic::run_agentic_loop(&question, &cache, agentic_config, &*reporter).await
        }).context("Failed to run agentic loop")?;

        // Clear spinner after agentic loop completes
        if let Some(ref s) = spinner {
            s.finish_and_clear();
        }

        // Clear ephemeral output (Phase 5 evaluation) before showing final results
        if !as_json {
            reporter.clear_all();
        }

        log::info!("Agentic loop completed: {} queries generated", agentic_response.queries.len());

        // Destructure AgenticQueryResponse into tuple (preserve gathered_context)
        let count_only_mode = agentic_response.total_count.is_none();
        let count = agentic_response.total_count.unwrap_or(0);
        (agentic_response.queries, agentic_response.results, count, count_only_mode, agentic_response.gathered_context)
    } else {
        // Standard mode: single LLM call + execution
        if let Some(ref s) = spinner {
            s.set_message("Generating queries...".to_string());
            s.enable_steady_tick(std::time::Duration::from_millis(80));
        }

        let semantic_response = runtime.block_on(async {
            crate::semantic::ask_question(&question, &cache, provider_override.clone(), additional_context, debug).await
        }).context("Failed to generate semantic queries")?;

        if let Some(ref s) = spinner {
            s.finish_and_clear();
        }
        log::info!("LLM generated {} queries", semantic_response.queries.len());

        // --execute: show queries and require y/N confirmation before running them
        if auto_execute && !as_json {
            println!("\n{}", "Generated Queries:".bold().cyan());
            println!("{}", "==================".cyan());
            for (idx, query_cmd) in semantic_response.queries.iter().enumerate() {
                println!(
                    "{}. {} {}",
                    (idx + 1).to_string().bright_white().bold(),
                    "rfx".bright_green().bold(),
                    query_cmd.command.bright_white()
                );
            }
            println!();
            eprint!("{}", "⚠  Run these queries against your codebase? [y/N] ".yellow());
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).context("Failed to read confirmation")?;
            if !input.trim().eq_ignore_ascii_case("y") {
                eprintln!("Aborted.");
                return Ok(());
            }
        }

        // Execute queries for standard mode
        let (exec_results, exec_total, exec_count_only) = runtime.block_on(async {
            crate::semantic::execute_queries(semantic_response.queries.clone(), &cache).await
        }).context("Failed to execute queries")?;

        (semantic_response.queries, exec_results, exec_total, exec_count_only, None)
    };

    // Generate conversational answer if --answer flag is set
    let generated_answer = if answer {
        // Show spinner while generating answer
        let answer_spinner = if !as_json {
            let s = ProgressBar::new_spinner();
            s.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.cyan} {msg}")
                    .unwrap()
                    .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
            );
            s.set_message("Generating answer...".to_string());
            s.enable_steady_tick(std::time::Duration::from_millis(80));
            Some(s)
        } else {
            None
        };

        // Initialize provider for answer generation
        let mut config = crate::semantic::config::load_config(cache.path())?;
        if let Some(provider) = &provider_override {
            config.provider = provider.clone();
        }
        let api_key = crate::semantic::config::get_api_key(&config.provider)?;
        let model = if config.model.is_some() {
            config.model.clone()
        } else {
            crate::semantic::config::get_user_model(&config.provider)
        };
        let provider_instance = crate::semantic::providers::create_provider(
            &config.provider,
            api_key,
            model,
            crate::semantic::config::get_provider_options(&config.provider),
            config.timeout_seconds,
        )?;

        // Extract codebase context (always available metadata: languages, file counts, directories)
        let codebase_context_str = crate::semantic::context::CodebaseContext::extract(&cache)
            .ok()
            .map(|ctx| ctx.to_prompt_string());

        // Generate answer (with optional gathered context from agentic mode + codebase context)
        let answer_result = runtime.block_on(async {
            crate::semantic::generate_answer(
                &question,
                &results,
                total_count,
                gathered_context.as_deref(),
                codebase_context_str.as_deref(),
                &*provider_instance,
            ).await
        }).context("Failed to generate answer")?;

        if let Some(s) = answer_spinner {
            s.finish_and_clear();
        }

        Some(answer_result)
    } else {
        None
    };

    // Output in JSON format if requested
    if as_json {
        // Build AgenticQueryResponse for JSON output (includes both queries and results)
        let json_response = crate::semantic::AgenticQueryResponse {
            queries: queries.clone(),
            results: results.clone(),
            total_count: if count_only { None } else { Some(total_count) },
            gathered_context: gathered_context.clone(),
            tools_executed: None, // No tools in non-agentic mode
            answer: generated_answer,
        };

        let json_str = if pretty_json {
            serde_json::to_string_pretty(&json_response)?
        } else {
            serde_json::to_string(&json_response)?
        };
        println!("{}", json_str);
        return Ok(());
    }

    // Display generated queries with color (unless in answer mode)
    if !answer {
        println!("\n{}", "Generated Queries:".bold().cyan());
        println!("{}", "==================".cyan());
        for (idx, query_cmd) in queries.iter().enumerate() {
            println!(
                "{}. {} {} {}",
                (idx + 1).to_string().bright_white().bold(),
                format!("[order: {}, merge: {}]", query_cmd.order, query_cmd.merge).dimmed(),
                "rfx".bright_green().bold(),
                query_cmd.command.bright_white()
            );
        }
        println!();
    }

    // Note: queries already executed in both modes above
    // Agentic mode: executed during run_agentic_loop
    // Standard mode: executed after ask_question

    // Display answer or results
    println!();
    if let Some(answer_text) = generated_answer {
        // Answer mode: show the conversational answer
        println!("{}", "Answer:".bold().green());
        println!("{}", "=======".green());
        println!();

        // Render markdown if it looks like markdown, otherwise print as-is
        termimad::print_text(&answer_text);
        println!();

        // Show summary of results used
        if !results.is_empty() {
            println!(
                "{}",
                format!(
                    "(Based on {} matches across {} files)",
                    total_count,
                    results.len()
                ).dimmed()
            );
        }
    } else {
        // Standard mode: show raw results
        if count_only {
            // Count-only mode: just show the total count (matching direct CLI behavior)
            println!("{} {}", "Found".bright_green().bold(), format!("{} result{}", total_count, if total_count == 1 { "" } else { "s" }).bright_white().bold());
        } else if results.is_empty() {
            println!("{}", "No results found.".yellow());
        } else {
            println!(
                "{} {} {} {} {}",
                "Found".bright_green().bold(),
                total_count.to_string().bright_white().bold(),
                "total results across".dimmed(),
                results.len().to_string().bright_white().bold(),
                "files:".dimmed()
            );
            println!();

            for file_group in &results {
                println!("{}:", file_group.path.bright_cyan().bold());
                for match_result in &file_group.matches {
                    println!(
                        "  {} {}-{}: {}",
                        "Line".dimmed(),
                        match_result.span.start_line.to_string().bright_yellow(),
                        match_result.span.end_line.to_string().bright_yellow(),
                        match_result.preview.lines().next().unwrap_or("")
                    );
                }
                println!();
            }
        }
    }

    Ok(())
}
