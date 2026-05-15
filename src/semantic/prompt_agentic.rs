//! Prompt generation for agentic mode
//!
//! This module builds specialized prompts for each phase of the agentic loop

use crate::cache::CacheManager;
use anyhow::Result;

use super::context::CodebaseContext;
use super::schema::QueryResponse;
use super::schema_agentic::EvaluationReport;

/// Embed agentic prompt template at compile time
const AGENTIC_TEMPLATE: &str = include_str!("prompt_agentic.md");

/// Build prompt for Phase 1: Assessment
///
/// This prompt asks the LLM to assess if it has enough context to answer
/// the user's question, or if it needs to gather more information first.
pub fn build_assessment_prompt(
    question: &str,
    cache: &CacheManager,
) -> Result<String> {
    // Extract basic codebase context
    let context = CodebaseContext::extract(cache).unwrap_or_else(|e| {
        log::warn!("Failed to extract codebase context: {}. Using empty context.", e);
        CodebaseContext {
            total_files: 0,
            languages: vec![],
            top_level_dirs: vec![],
            common_paths: vec![],
            is_monorepo: false,
            project_count: None,
            dominant_language: None,
        }
    });

    let context_str = if context.total_files == 0 {
        "No files indexed yet (empty codebase).".to_string()
    } else {
        context.to_prompt_string()
    };

    // Read project-specific config from REFLEX.md
    let workspace_root = cache.workspace_root();
    let project_config = read_project_config(&workspace_root)
        .unwrap_or_else(|| "No project-specific instructions provided.".to_string());

    // Build assessment prompt
    Ok(format!(
        r#"{template}

## Current Phase: ASSESSMENT

You are in the **assessment phase**. Your task is to determine if you have enough context to generate accurate search queries for the user's question.

## Codebase Context (Currently Available)

{codebase_context}

## Project-Specific Instructions

{project_config}

## User Question

{question}

## Your Task

Analyze the question and determine:

1. **Can you answer the question using ONLY the Codebase Context above?**
   - For counting questions (file counts, language stats): Check if the answer is already in the Codebase Context
   - For simple factual questions: Check if the metadata already provides the answer
   - If yes, return `phase: "final"` with empty queries array `[]`

2. **If not, do you have enough context to generate accurate search queries?**
   - Do you understand the project structure well enough?
   - Do you know which directories/files to target?
   - Can you generate precise search patterns?

3. **What additional context would help?** (if needed)
   - Project structure details (`gather_context`)
   - Exploratory queries to find patterns (`explore_codebase`)
   - Dependency analysis (`analyze_structure`)
   - File counts and statistics (`get_statistics`)

## Response Format

Return a JSON object with:
- `phase`: "assessment" or "final"
- `reasoning`: Your thought process
- `needs_context`: true if you need more info, false if ready to answer
- `tool_calls`: Array of tools to execute (if needs_context=true)
- `queries`: Empty for assessment, or array of queries if phase="final" (CAN BE EMPTY if answer is in context)
- `confidence`: 0.0-1.0 confidence score

**Important:**
- If the answer is already in the Codebase Context, set `phase: "final"` with `queries: []` (empty array)
- If you need to search the codebase, set `phase: "final"` and provide the search queries
- If you need more context first, set `phase: "assessment"` with `needs_context: true` and specify tools

**Schema:**
```json
{schema}
```

**IMPORTANT:** Return ONLY valid JSON.
"#,
        template = AGENTIC_TEMPLATE,
        codebase_context = context_str,
        project_config = project_config,
        question = question,
        schema = super::schema_agentic::AGENTIC_RESPONSE_SCHEMA,
    ))
}

/// Build prompt for Phase 3: Query Generation
///
/// This prompt asks the LLM to generate the final search queries based on
/// the gathered context from tools.
pub fn build_generation_prompt(
    question: &str,
    gathered_context: &str,
    cache: &CacheManager,
) -> Result<String> {
    // Extract basic codebase context
    let context = CodebaseContext::extract(cache).unwrap_or_else(|e| {
        log::warn!("Failed to extract codebase context: {}. Using empty context.", e);
        CodebaseContext {
            total_files: 0,
            languages: vec![],
            top_level_dirs: vec![],
            common_paths: vec![],
            is_monorepo: false,
            project_count: None,
            dominant_language: None,
        }
    });

    let context_str = context.to_prompt_string();

    // Read project config
    let workspace_root = cache.workspace_root();
    let project_config = read_project_config(&workspace_root)
        .unwrap_or_else(|| "No project-specific instructions provided.".to_string());

    // Build context section
    let context_section = if gathered_context.is_empty() {
        String::new()
    } else {
        format!("\n## Gathered Context\n\n{}\n", gathered_context)
    };

    Ok(format!(
        r#"{template}

## Current Phase: FINAL QUERY GENERATION

You have completed context gathering. Now generate the final search queries.

## Codebase Context

{codebase_context}

## Project-Specific Instructions

{project_config}
{context_section}
## User Question

{question}

## Your Task

Determine if you can answer the question from the available context, or if you need to search the codebase:

1. **Check if the answer is already available:**
   - Review the Codebase Context for file counts, language stats, and structural information
   - Review the Gathered Context for tool outputs (documentation, dependency info, etc.)
   - If the answer is there, return empty queries array `[]`

2. **If not, generate precise search queries:**
   - Use the context to create targeted search patterns
   - Focus queries on the most relevant files and patterns

## Response Format

Return a JSON object with:
- `phase`: "final"
- `reasoning`: Your thought process (explain why queries are needed, or why answer is in context)
- `queries`: Array of query commands (CAN BE EMPTY if answer is already in Codebase/Gathered Context)
- `confidence`: 0.0-1.0 confidence score

**Schema:**
```json
{schema}
```

**IMPORTANT:** Return ONLY valid JSON.
"#,
        template = AGENTIC_TEMPLATE,
        codebase_context = context_str,
        project_config = project_config,
        context_section = context_section,
        question = question,
        schema = super::schema_agentic::AGENTIC_RESPONSE_SCHEMA,
    ))
}

/// Build prompt for Phase 6: Query Refinement
///
/// This prompt asks the LLM to refine queries based on evaluation feedback
pub fn build_refinement_prompt(
    question: &str,
    gathered_context: &str,
    previous_response: &QueryResponse,
    evaluation: &EvaluationReport,
    cache: &CacheManager,
) -> Result<String> {
    // Extract basic codebase context
    let context = CodebaseContext::extract(cache).unwrap_or_else(|e| {
        log::warn!("Failed to extract codebase context: {}. Using empty context.", e);
        CodebaseContext {
            total_files: 0,
            languages: vec![],
            top_level_dirs: vec![],
            common_paths: vec![],
            is_monorepo: false,
            project_count: None,
            dominant_language: None,
        }
    });

    let context_str = context.to_prompt_string();

    // Format previous queries
    let previous_queries = previous_response.queries.iter()
        .map(|q| format!("- {}", q.command))
        .collect::<Vec<_>>()
        .join("\n");

    // Format evaluation feedback
    let eval_feedback = super::evaluator::format_evaluation_for_llm(evaluation);

    // Build context section
    let context_section = if gathered_context.is_empty() {
        String::new()
    } else {
        format!("\n## Gathered Context\n\n{}\n", gathered_context)
    };

    Ok(format!(
        r#"{template}

## Current Phase: QUERY REFINEMENT

Your previous queries did not produce satisfactory results. Please refine them based on the evaluation feedback.

## Codebase Context

{codebase_context}
{context_section}
## User Question

{question}

## Previous Queries

{previous_queries}

## Evaluation Feedback

{eval_feedback}

## Your Task

Refine the queries to address the issues identified in the evaluation.
Pay special attention to the suggestions provided.

## Response Format

Return a JSON object with refined queries:

```json
{{
  "queries": [
    {{
      "command": "query \"pattern\" --flags",
      "order": 1,
      "merge": true
    }}
  ]
}}
```

**IMPORTANT:** Return ONLY valid JSON matching the QueryResponse schema.
"#,
        template = AGENTIC_TEMPLATE,
        codebase_context = context_str,
        context_section = context_section,
        question = question,
        previous_queries = previous_queries,
        eval_feedback = eval_feedback,
    ))
}

/// Read project-specific configuration from REFLEX.md
fn read_project_config(workspace_root: &std::path::Path) -> Option<String> {
    let config_path = workspace_root.join("REFLEX.md");

    if !config_path.exists() {
        return None;
    }

    std::fs::read_to_string(&config_path).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_build_assessment_prompt() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        let prompt = build_assessment_prompt("find todos", &cache).unwrap();

        assert!(prompt.contains("ASSESSMENT"));
        assert!(prompt.contains("find todos"));
        assert!(prompt.contains("needs_context"));
    }

    #[test]
    fn test_build_generation_prompt() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());
        let context = "Test context from tools";

        let prompt = build_generation_prompt("find todos", context, &cache).unwrap();

        assert!(prompt.contains("FINAL QUERY GENERATION"));
        assert!(prompt.contains("Test context from tools"));
        assert!(prompt.contains("find todos"));
    }

    #[test]
    fn test_build_refinement_prompt() {
        let temp = TempDir::new().unwrap();
        let cache = CacheManager::new(temp.path());

        let previous = QueryResponse {
            queries: vec![super::super::schema::QueryCommand {
                command: "query \"TODO\"".to_string(),
                order: 1,
                merge: true,
            }],
            message: None,
        };

        let eval = EvaluationReport {
            success: false,
            issues: vec![],
            suggestions: vec!["Try broader pattern".to_string()],
            score: 0.3,
        };

        let prompt = build_refinement_prompt(
            "find todos",
            "",
            &previous,
            &eval,
            &cache,
        ).unwrap();

        assert!(prompt.contains("REFINEMENT"));
        assert!(prompt.contains("previous queries"));
        assert!(prompt.contains("evaluation"));
    }
}
