//! Agentic loop orchestrator for multi-step query generation
//!
//! This module implements the main agentic workflow:
//! 1. Phase 1: Assess if more context is needed
//! 2. Phase 2: Gather context using tools
//! 3. Phase 3: Generate final queries
//! 4. Phase 4: Execute queries
//! 5. Phase 5: Evaluate results
//! 6. Phase 6: Refine if needed

use anyhow::{Context as AnyhowContext, Result};
use crate::cache::CacheManager;

use super::providers::{LlmProvider, create_provider};
use super::config;
use super::schema::{QueryResponse, AgenticQueryResponse};
use super::schema_agentic::{AgenticResponse, Phase, ToolCall};
use super::tools::{execute_tool, format_tool_results, ToolResult};
use super::evaluator::{evaluate_results, EvaluationConfig};
use super::reporter::AgenticReporter;

/// Configuration for agentic loop
#[derive(Debug, Clone)]
pub struct AgenticConfig {
    /// Maximum iterations for refinement (default: 2)
    pub max_iterations: usize,

    /// Maximum tool calls per gathering phase (default: 5)
    pub max_tools_per_phase: usize,

    /// Enable result evaluation phase
    pub enable_evaluation: bool,

    /// Evaluation configuration
    pub eval_config: EvaluationConfig,

    /// Provider name override
    pub provider_override: Option<String>,

    /// Model override
    pub model_override: Option<String>,

    /// Show LLM reasoning blocks (default: false)
    pub show_reasoning: bool,

    /// Verbose output (show tool results, etc.) (default: false)
    pub verbose: bool,

    /// Debug mode: output full LLM prompts (default: false)
    pub debug: bool,
}

impl Default for AgenticConfig {
    fn default() -> Self {
        Self {
            max_iterations: 2,
            max_tools_per_phase: 5,
            enable_evaluation: true,
            eval_config: EvaluationConfig::default(),
            provider_override: None,
            model_override: None,
            show_reasoning: false,
            verbose: false,
            debug: false,
        }
    }
}

/// Run the full agentic loop
pub async fn run_agentic_loop(
    question: &str,
    cache: &CacheManager,
    config: AgenticConfig,
    reporter: &dyn AgenticReporter,
) -> Result<AgenticQueryResponse> {
    log::info!("Starting agentic loop for question: {}", question);

    // Validate cache before starting - auto-reindex if schema mismatch detected
    if let Err(e) = cache.validate() {
        let error_msg = e.to_string();

        // Check if this is a schema mismatch error
        if error_msg.contains("Cache schema version mismatch") {
            log::warn!("Cache schema mismatch detected, auto-reindexing...");

            // Create progress callback that reports to the reporter
            use std::sync::Arc;
            let progress_callback: crate::indexer::ProgressCallback = Arc::new({
                // Clone reporter reference for the callback closure
                // Note: We can't capture `reporter` directly since it's a trait object,
                // so we'll just log progress and rely on the indexer's built-in progress bar
                move |current: usize, total: usize, message: String| {
                    log::debug!("Reindex progress: [{}/{}] {}", current, total, message);
                }
            });

            // Trigger reindexing
            let workspace_root = cache.workspace_root();
            let index_config = crate::IndexConfig::default();
            let indexer = crate::indexer::Indexer::new(cache.clone(), index_config);

            log::info!("Auto-reindexing cache at {:?}", workspace_root);
            indexer.index_with_callback(&workspace_root, false, Some(progress_callback))?;

            log::info!("Cache reindexing completed successfully");
        } else {
            // Other validation errors should propagate up
            return Err(e);
        }
    }

    // Initialize provider
    let provider = initialize_provider(&config, cache)?;

    // Phase 1: Initial assessment - does the LLM need more context?
    let (needs_context, initial_response) = phase_1_assess(
        question,
        cache,
        &*provider,
        reporter,
        config.debug,
    ).await?;

    // Phase 2: Context gathering (if needed)
    let (gathered_context, tools_executed) = if needs_context {
        phase_2_gather(
            question,
            initial_response,
            cache,
            &*provider,
            &config,
            reporter,
        ).await?
    } else {
        (String::new(), Vec::new())
    };

    // Phase 3: Generate final queries
    let (query_response, query_confidence) = phase_3_generate(
        question,
        &gathered_context,
        cache,
        &*provider,
        reporter,
        config.debug,
    ).await?;

    // Phase 4: Execute queries
    let (results, total_count, count_only) = super::executor::execute_queries(
        query_response.queries.clone(),
        cache,
    ).await?;

    log::info!("Executed queries: {} file groups, {} total matches", results.len(), total_count);

    // Phase 5: Evaluate results (if enabled and not count-only)
    if config.enable_evaluation && !count_only {
        let evaluation = evaluate_results(
            &results,
            total_count,
            question,
            &config.eval_config,
            if !gathered_context.is_empty() { Some(gathered_context.as_str()) } else { None },
            query_response.queries.len(),
            Some(query_confidence),
        );

        log::info!("Evaluation: success={}, score={:.2}", evaluation.success, evaluation.score);

        // Report evaluation
        reporter.report_evaluation(&evaluation);

        // Phase 6: Refinement (if needed and iterations remaining)
        if !evaluation.success && config.max_iterations > 1 {
            log::info!("Results unsatisfactory, attempting refinement");

            return phase_6_refine(
                question,
                &gathered_context,
                &query_response,
                &evaluation,
                cache,
                &*provider,
                &config,
                reporter,
                config.debug,
            ).await;
        }
    }

    // Return enhanced response with both queries and results
    Ok(AgenticQueryResponse {
        queries: query_response.queries,
        results,
        total_count: if count_only { None } else { Some(total_count) },
        gathered_context: if !gathered_context.is_empty() {
            Some(gathered_context)
        } else {
            None
        },
        tools_executed: if !tools_executed.is_empty() {
            Some(tools_executed)
        } else {
            None
        },
        answer: None,  // No answer generation in agentic mode (handled in CLI)
    })
}

/// Phase 1: Assess if more context is needed
async fn phase_1_assess(
    question: &str,
    cache: &CacheManager,
    provider: &dyn LlmProvider,
    reporter: &dyn AgenticReporter,
    debug: bool,
) -> Result<(bool, AgenticResponse)> {
    log::info!("Phase 1: Assessing context needs");

    // Build assessment prompt
    let prompt = super::prompt_agentic::build_assessment_prompt(question, cache)?;

    // Debug mode: output full prompt
    if debug {
        eprintln!("\n{}", "=".repeat(80));
        eprintln!("DEBUG: Full LLM Prompt (Phase 1: Assessment)");
        eprintln!("{}", "=".repeat(80));
        eprintln!("{}", prompt);
        eprintln!("{}\n", "=".repeat(80));
    }

    // Call LLM — validate against AgenticResponse (requires phase + reasoning)
    let json_response = call_with_retry(
        provider, &prompt, 2, super::validate_agentic_response,
    ).await?;

    // Parse response
    let response: AgenticResponse = serde_json::from_str(&json_response)
        .context("Failed to parse LLM assessment response")?;

    // Validate phase
    if response.phase != Phase::Assessment && response.phase != Phase::Final {
        anyhow::bail!("Expected 'assessment' or 'final' phase, got {:?}", response.phase);
    }

    let needs_context = response.needs_context && !response.tool_calls.is_empty();

    log::info!(
        "Assessment complete: needs_context={}, tool_calls={}",
        needs_context,
        response.tool_calls.len()
    );

    // Report assessment
    reporter.report_assessment(&response.reasoning, needs_context, &response.tool_calls);

    Ok((needs_context, response))
}

/// Phase 2: Gather context using tools
async fn phase_2_gather(
    _question: &str,
    initial_response: AgenticResponse,
    cache: &CacheManager,
    _provider: &dyn LlmProvider,
    config: &AgenticConfig,
    reporter: &dyn AgenticReporter,
) -> Result<(String, Vec<String>)> {
    log::info!("Phase 2: Gathering context via tools");

    let mut all_tool_results = Vec::new();
    let mut tool_descriptions = Vec::new();

    // Limit tool calls to prevent excessive execution
    let tool_calls: Vec<ToolCall> = initial_response.tool_calls
        .into_iter()
        .take(config.max_tools_per_phase)
        .collect();

    log::info!("Executing {} tool calls", tool_calls.len());

    // Execute all tool calls
    for (idx, tool) in tool_calls.iter().enumerate() {
        log::debug!("Executing tool {}/{}: {:?}", idx + 1, tool_calls.len(), tool);

        // Get tool description for UI display
        let tool_desc = describe_tool_for_ui(tool);
        tool_descriptions.push(tool_desc);

        // Report tool start
        reporter.report_tool_start(idx + 1, tool);

        match execute_tool(tool, cache).await {
            Ok(result) => {
                log::info!("Tool {} succeeded: {}", idx + 1, result.description);
                reporter.report_tool_complete(idx + 1, &result);
                all_tool_results.push(result);
            }
            Err(e) => {
                log::warn!("Tool {} failed: {}", idx + 1, e);
                // Continue with other tools even if one fails
                let failed_result = ToolResult {
                    description: format!("Tool {} (failed)", idx + 1),
                    output: format!("Error: {}", e),
                    success: false,
                };
                reporter.report_tool_complete(idx + 1, &failed_result);
                all_tool_results.push(failed_result);
            }
        }
    }

    // Format all tool results into context string
    let gathered_context = format_tool_results(&all_tool_results);

    log::info!("Context gathering complete: {} chars", gathered_context.len());

    Ok((gathered_context, tool_descriptions))
}

/// Generate a user-friendly description of a tool call
fn describe_tool_for_ui(tool: &ToolCall) -> String {
    match tool {
        ToolCall::GatherContext { params } => {
            let mut parts = Vec::new();
            if params.structure { parts.push("structure"); }
            if params.file_types { parts.push("file types"); }
            if params.project_type { parts.push("project type"); }
            if params.framework { parts.push("frameworks"); }
            if params.entry_points { parts.push("entry points"); }
            if params.test_layout { parts.push("test layout"); }
            if params.config_files { parts.push("config files"); }

            if parts.is_empty() {
                "gather_context: General codebase context".to_string()
            } else {
                format!("gather_context: {}", parts.join(", "))
            }
        }
        ToolCall::ExploreCodebase { description, .. } => {
            format!("explore_codebase: {}", description)
        }
        ToolCall::AnalyzeStructure { analysis_type } => {
            format!("analyze_structure: {:?}", analysis_type)
        }
        ToolCall::SearchDocumentation { query, files } => {
            if let Some(file_list) = files {
                format!("search_documentation: '{}' in files {:?}", query, file_list)
            } else {
                format!("search_documentation: '{}'", query)
            }
        }
        ToolCall::GetStatistics => {
            "get_statistics: Retrieved file counts and language stats".to_string()
        }
        ToolCall::GetDependencies { file_path, reverse } => {
            if *reverse {
                format!("get_dependencies: What depends on '{}'", file_path)
            } else {
                format!("get_dependencies: Dependencies of '{}'", file_path)
            }
        }
        ToolCall::GetAnalysisSummary { .. } => {
            "get_analysis_summary: Dependency health overview".to_string()
        }
        ToolCall::FindIslands { .. } => {
            "find_islands: Disconnected component analysis".to_string()
        }
    }
}

/// Phase 3: Generate final queries
///
/// Returns (QueryResponse, confidence_score)
async fn phase_3_generate(
    question: &str,
    gathered_context: &str,
    cache: &CacheManager,
    provider: &dyn LlmProvider,
    reporter: &dyn AgenticReporter,
    debug: bool,
) -> Result<(QueryResponse, f32)> {
    log::info!("Phase 3: Generating final queries");

    // Build generation prompt with gathered context
    let prompt = super::prompt_agentic::build_generation_prompt(
        question,
        gathered_context,
        cache,
    )?;

    // Debug mode: output full prompt
    if debug {
        eprintln!("\n{}", "=".repeat(80));
        eprintln!("DEBUG: Full LLM Prompt (Phase 3: Query Generation)");
        eprintln!("{}", "=".repeat(80));
        eprintln!("{}", prompt);
        eprintln!("{}\n", "=".repeat(80));
    }

    // Call LLM — accepts either AgenticResponse or QueryResponse (fallback path)
    let json_response = call_with_retry(
        provider, &prompt, 2, super::validate_agentic_or_query_response,
    ).await?;

    // Parse response - could be AgenticResponse or QueryResponse
    // Try AgenticResponse first (for agentic mode)
    if let Ok(agentic_response) = serde_json::from_str::<AgenticResponse>(&json_response) {
        if agentic_response.phase == Phase::Final {
            let confidence = agentic_response.confidence;

            // Report generation with reasoning
            reporter.report_generation(
                Some(&agentic_response.reasoning),
                agentic_response.queries.len(),
                confidence,
            );

            // Convert to QueryResponse and return with confidence
            return Ok((
                QueryResponse {
                    queries: agentic_response.queries,
                    message: None,
                },
                confidence,
            ));
        }
    }

    // Fallback: try direct QueryResponse
    let query_response: QueryResponse = serde_json::from_str(&json_response)
        .context("Failed to parse LLM query generation response")?;

    log::info!("Generated {} queries", query_response.queries.len());

    // Report generation without reasoning (fallback mode)
    reporter.report_generation(None, query_response.queries.len(), 1.0);

    // Default confidence of 1.0 for fallback mode
    Ok((query_response, 1.0))
}

/// Phase 6: Refine queries based on evaluation
async fn phase_6_refine(
    question: &str,
    gathered_context: &str,
    previous_response: &QueryResponse,
    evaluation: &super::schema_agentic::EvaluationReport,
    cache: &CacheManager,
    provider: &dyn LlmProvider,
    config: &AgenticConfig,
    reporter: &dyn AgenticReporter,
    debug: bool,
) -> Result<AgenticQueryResponse> {
    log::info!("Phase 6: Refining queries based on evaluation");

    // Report refinement start
    reporter.report_refinement_start();

    // Build refinement prompt with evaluation feedback
    let prompt = super::prompt_agentic::build_refinement_prompt(
        question,
        gathered_context,
        previous_response,
        evaluation,
        cache,
    )?;

    // Debug mode: output full prompt
    if debug {
        eprintln!("\n{}", "=".repeat(80));
        eprintln!("DEBUG: Full LLM Prompt (Phase 6: Refinement)");
        eprintln!("{}", "=".repeat(80));
        eprintln!("{}", prompt);
        eprintln!("{}\n", "=".repeat(80));
    }

    // Call LLM for refinement — expects QueryResponse format
    let json_response = call_with_retry(
        provider, &prompt, 2, super::validate_query_response,
    ).await?;

    // Parse refined response
    let refined_response: QueryResponse = serde_json::from_str(&json_response)
        .context("Failed to parse LLM refinement response")?;

    log::info!("Refinement complete: {} refined queries", refined_response.queries.len());

    // Execute refined queries
    let (results, total_count, count_only) = super::executor::execute_queries(
        refined_response.queries.clone(),
        cache,
    ).await?;

    // Evaluate refined results (one final time)
    let refined_evaluation = evaluate_results(
        &results,
        total_count,
        question,
        &config.eval_config,
        if !gathered_context.is_empty() { Some(gathered_context) } else { None },
        refined_response.queries.len(),
        None,  // No confidence available in refinement
    );

    log::info!(
        "Refined evaluation: success={}, score={:.2}",
        refined_evaluation.success,
        refined_evaluation.score
    );

    // Return enhanced response with both queries and results
    Ok(AgenticQueryResponse {
        queries: refined_response.queries,
        results,
        total_count: if count_only { None } else { Some(total_count) },
        gathered_context: if !gathered_context.is_empty() {
            Some(gathered_context.to_string())
        } else {
            None
        },
        tools_executed: None,  // No new tools executed during refinement
        answer: None,  // No answer generation in agentic mode (handled in CLI)
    })
}

/// Initialize LLM provider based on configuration
fn initialize_provider(
    config: &AgenticConfig,
    cache: &CacheManager,
) -> Result<Box<dyn LlmProvider>> {
    // Load semantic config
    let mut semantic_config = config::load_config(cache.path())?;

    // Apply overrides
    if let Some(provider) = &config.provider_override {
        semantic_config.provider = provider.clone();
    }

    // Get API key
    let api_key = config::get_api_key(&semantic_config.provider)?;

    let model = config::resolve_model(&semantic_config, config.model_override.as_deref());

    // Create provider
    create_provider(&semantic_config.provider, api_key, model, config::get_provider_options(&semantic_config.provider), semantic_config.timeout_seconds)
}

/// Call LLM provider with retry logic (from semantic/mod.rs)
async fn call_with_retry(
    provider: &dyn LlmProvider,
    prompt: &str,
    max_retries: usize,
    validator: impl Fn(&str) -> Result<(), String>,
) -> Result<String> {
    super::call_with_retry(provider, prompt, max_retries, validator).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agentic_config_defaults() {
        let config = AgenticConfig::default();
        assert_eq!(config.max_iterations, 2);
        assert_eq!(config.max_tools_per_phase, 5);
        assert!(config.enable_evaluation);
    }

    #[test]
    fn test_agentic_config_custom() {
        let config = AgenticConfig {
            max_iterations: 3,
            max_tools_per_phase: 10,
            enable_evaluation: false,
            ..Default::default()
        };

        assert_eq!(config.max_iterations, 3);
        assert_eq!(config.max_tools_per_phase, 10);
        assert!(!config.enable_evaluation);
    }
}
