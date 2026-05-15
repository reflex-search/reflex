//! Answer generation from search results
//!
//! This module provides functionality to synthesize conversational answers
//! from code search results using LLM providers.

use super::providers::LlmProvider;
use crate::models::FileGroupedResult;
use anyhow::Result;

/// Maximum number of matches to include in the prompt (to avoid token limits)
const MAX_MATCHES_IN_PROMPT: usize = 50;

/// Maximum preview length per match (characters)
const MAX_PREVIEW_LENGTH: usize = 200;

/// Generate a conversational answer based on search results
///
/// Takes the user's original question and search results, then calls the LLM
/// to synthesize a natural language answer that references specific files and
/// line numbers from the results.
///
/// # Arguments
///
/// * `question` - The original user question
/// * `results` - Search results grouped by file
/// * `total_count` - Total number of matches found
/// * `gathered_context` - Optional context gathered from tools (documentation, codebase structure)
/// * `codebase_context` - Optional codebase metadata (always available, language distribution, directories)
/// * `provider` - LLM provider to use for answer generation
///
/// # Returns
///
/// A conversational answer string that summarizes the findings
pub async fn generate_answer(
    question: &str,
    results: &[FileGroupedResult],
    total_count: usize,
    gathered_context: Option<&str>,
    codebase_context: Option<&str>,
    provider: &dyn LlmProvider,
) -> Result<String> {
    // Handle empty results - use gathered context if available, then codebase context
    if results.is_empty() {
        // Try gathered context first (from tools like search_documentation, gather_context)
        if let Some(context) = gathered_context {
            if !context.is_empty() {
                // Generate answer from documentation/context alone
                let prompt = build_context_only_prompt(question, context);
                log::debug!(
                    "Generating answer from gathered context ({} chars)",
                    prompt.len()
                );
                let answer = provider.complete(&prompt, false).await?;
                let cleaned = strip_markdown_fences(&answer);
                return Ok(cleaned.to_string());
            }
        }

        // Try codebase context (language distribution, file counts, directories)
        if let Some(context) = codebase_context {
            if !context.is_empty() {
                // Generate answer from codebase metadata alone
                let prompt = build_codebase_context_prompt(question, context);
                log::debug!(
                    "Generating answer from codebase context ({} chars)",
                    prompt.len()
                );
                let answer = provider.complete(&prompt, false).await?;
                let cleaned = strip_markdown_fences(&answer);
                return Ok(cleaned.to_string());
            }
        }

        return Ok(format!("No results found for: {}", question));
    }

    // Build the prompt with search results (and optional gathered context)
    let prompt = build_answer_prompt(question, results, total_count, gathered_context);

    log::debug!("Generating answer with prompt ({} chars)", prompt.len());

    // Call LLM to generate answer (json_mode: false for plain text output)
    let answer = provider.complete(&prompt, false).await?;

    // Clean up the response (remove markdown fences if present)
    let cleaned = strip_markdown_fences(&answer);

    Ok(cleaned.to_string())
}

/// Build the prompt for answer generation (with optional gathered context)
fn build_answer_prompt(
    question: &str,
    results: &[FileGroupedResult],
    total_count: usize,
    gathered_context: Option<&str>,
) -> String {
    let mut prompt = String::new();

    // Instructions
    prompt.push_str("You are analyzing code search results to answer a developer's question.\n\n");
    prompt.push_str("IMPORTANT: Provide ONLY the answer text, without any markdown formatting, code fences, or explanatory prefixes.\n\n");

    prompt.push_str(&format!("Question: {}\n\n", question));

    // Add gathered context if available (documentation, codebase structure)
    if let Some(context) = gathered_context {
        if !context.is_empty() {
            prompt.push_str("Additional Context (from documentation and codebase analysis):\n");
            prompt.push_str(
                "====================================================================\n\n",
            );
            prompt.push_str(context);
            prompt.push_str("\n\n");
        }
    }

    // Add search result summary
    prompt.push_str(&format!(
        "Found {} total matches across {} files.\n\n",
        total_count,
        results.len()
    ));

    prompt.push_str("Code Search Results:\n");
    prompt.push_str("====================\n\n");

    // Format results for the prompt (limit to avoid token overflow)
    let mut match_count = 0;
    for file_group in results {
        if match_count >= MAX_MATCHES_IN_PROMPT {
            prompt.push_str(&format!(
                "\n... and {} more matches not shown\n",
                total_count - match_count
            ));
            break;
        }

        prompt.push_str(&format!("File: {}\n", file_group.path));

        for match_result in &file_group.matches {
            if match_count >= MAX_MATCHES_IN_PROMPT {
                break;
            }

            log::debug!(
                "Formatting match at {}:{} - context_before: {}, context_after: {}",
                file_group.path,
                match_result.span.start_line,
                match_result.context_before.len(),
                match_result.context_after.len()
            );

            // Show context before the match
            for (idx, line) in match_result.context_before.iter().enumerate() {
                let line_num = match_result
                    .span
                    .start_line
                    .saturating_sub(match_result.context_before.len() - idx);
                // Truncate long lines
                let truncated = if line.len() > MAX_PREVIEW_LENGTH {
                    format!("{}...", &line[..MAX_PREVIEW_LENGTH])
                } else {
                    line.clone()
                };
                prompt.push_str(&format!("  Line {}: {}\n", line_num, truncated.trim()));
            }

            // Show the match line itself
            let preview = if match_result.preview.len() > MAX_PREVIEW_LENGTH {
                format!("{}...", &match_result.preview[..MAX_PREVIEW_LENGTH])
            } else {
                match_result.preview.clone()
            };

            prompt.push_str(&format!(
                "  Line {}-{}: {}\n",
                match_result.span.start_line,
                match_result.span.end_line,
                preview.trim()
            ));

            // Show context after the match
            for (idx, line) in match_result.context_after.iter().enumerate() {
                let line_num = match_result.span.start_line + idx + 1;
                // Truncate long lines
                let truncated = if line.len() > MAX_PREVIEW_LENGTH {
                    format!("{}...", &line[..MAX_PREVIEW_LENGTH])
                } else {
                    line.clone()
                };
                prompt.push_str(&format!("  Line {}: {}\n", line_num, truncated.trim()));
            }

            match_count += 1;
        }

        prompt.push_str("\n");
    }

    // Instructions for answer format
    prompt.push_str("\nProvide a conversational answer that:\n");
    prompt.push_str("1. Directly answers the question based on the search results\n");
    prompt.push_str("2. References specific files and line numbers where relevant\n");
    prompt
        .push_str("3. Summarizes patterns or common approaches if multiple results are similar\n");
    prompt.push_str("4. Is concise but informative (typically 2-4 sentences)\n");
    prompt.push_str("5. Only mentions information that appears in the search results above\n\n");

    prompt.push_str("Answer (plain text only, no markdown):\n");

    prompt
}

/// Build prompt for answering from context alone (no code search results)
fn build_context_only_prompt(question: &str, gathered_context: &str) -> String {
    let mut prompt = String::new();

    prompt.push_str(
        "You are answering a developer's question using documentation and codebase context.\n\n",
    );
    prompt.push_str("IMPORTANT: Provide ONLY the answer text, without any markdown formatting, code fences, or explanatory prefixes.\n\n");

    prompt.push_str(&format!("Question: {}\n\n", question));

    prompt.push_str("Available Context (from documentation and codebase analysis):\n");
    prompt.push_str("================================================================\n\n");
    prompt.push_str(gathered_context);
    prompt.push_str("\n\n");

    prompt.push_str("Provide a conversational answer that:\n");
    prompt.push_str("1. Directly answers the question based on the context above\n");
    prompt.push_str("2. References documentation sections or files where relevant\n");
    prompt.push_str("3. Is concise but informative (typically 2-4 sentences)\n");
    prompt.push_str("4. Only mentions information that appears in the context above\n\n");

    prompt.push_str("Answer (plain text only, no markdown):\n");

    prompt
}

/// Build prompt for answering from codebase metadata alone (file counts, languages, directories)
fn build_codebase_context_prompt(question: &str, codebase_context: &str) -> String {
    let mut prompt = String::new();

    prompt.push_str("You are answering a developer's question using codebase metadata.\n\n");
    prompt.push_str("IMPORTANT: Provide ONLY the answer text, without any markdown formatting, code fences, or explanatory prefixes.\n\n");

    prompt.push_str(&format!("Question: {}\n\n", question));

    prompt.push_str("Codebase Metadata:\n");
    prompt.push_str("==================\n\n");
    prompt.push_str(codebase_context);
    prompt.push_str("\n\n");

    prompt.push_str("Provide a conversational answer that:\n");
    prompt.push_str("1. Directly answers the question using the metadata above\n");
    prompt.push_str("2. Uses specific numbers and percentages from the metadata\n");
    prompt.push_str("3. Is concise but informative (typically 1-2 sentences)\n");
    prompt.push_str("4. Only mentions information that appears in the metadata above\n\n");

    prompt.push_str("Answer (plain text only, no markdown):\n");

    prompt
}

/// Strip markdown code fences from LLM response
///
/// Some LLMs add markdown formatting even when instructed not to.
fn strip_markdown_fences(text: &str) -> &str {
    let trimmed = text.trim();

    // Check for markdown code fence pattern
    if trimmed.starts_with("```") && trimmed.ends_with("```") {
        // Remove opening fence (either ```markdown, ```text, or just ```)
        let without_start = if let Some(rest) = trimmed.strip_prefix("```markdown") {
            rest
        } else if let Some(rest) = trimmed.strip_prefix("```text") {
            rest
        } else if let Some(rest) = trimmed.strip_prefix("```") {
            rest
        } else {
            return trimmed;
        };

        // Remove closing fence
        let without_end = without_start.strip_suffix("```").unwrap_or(without_start);

        without_end.trim()
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_markdown_fences() {
        let input = "```\nThis is the answer\n```";
        assert_eq!(strip_markdown_fences(input), "This is the answer");
    }

    #[test]
    fn test_strip_markdown_fences_with_language() {
        let input = "```text\nThis is the answer\n```";
        assert_eq!(strip_markdown_fences(input), "This is the answer");
    }

    #[test]
    fn test_strip_markdown_fences_no_fences() {
        let input = "This is the answer";
        assert_eq!(strip_markdown_fences(input), "This is the answer");
    }

    #[test]
    fn test_build_answer_prompt_empty_results() {
        let results: Vec<FileGroupedResult> = vec![];
        let prompt = build_answer_prompt("Find TODOs", &results, 0, None);

        assert!(prompt.contains("Found 0 total matches"));
        assert!(prompt.contains("Question: Find TODOs"));
    }
}
