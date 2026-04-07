//! Semantic query generation using LLMs

pub mod config;
pub mod configure;
pub mod context;
pub mod executor;
pub mod prompt;
pub mod providers;
pub mod schema;
pub mod answer;

// Agentic mode modules (experimental)
pub mod schema_agentic;
pub mod agentic;
pub mod tools;
pub mod evaluator;
pub mod prompt_agentic;
pub mod reporter;

// Interactive chat mode modules
pub mod chat_session;
pub mod chat_tui;

// Re-export main types for convenience
pub use configure::run_configure_wizard;
pub use executor::{execute_queries, parse_command, ParsedCommand};
pub use schema::{QueryCommand, QueryResponse as SemanticQueryResponse, AgenticQueryResponse};
pub use agentic::{run_agentic_loop, AgenticConfig};
pub use reporter::{AgenticReporter, ConsoleReporter, QuietReporter};
pub use answer::generate_answer;
pub use chat_tui::run_chat_mode;
pub use config::{save_user_provider, is_any_api_key_configured};

use anyhow::{Context, Result};
use crate::cache::CacheManager;

/// Generate query commands from a natural language question
///
/// This is the main entry point for the semantic query feature.
pub async fn ask_question(
    question: &str,
    cache: &CacheManager,
    provider_override: Option<String>,
    additional_context: Option<String>,
    debug: bool,
) -> Result<schema::QueryResponse> {
    // Load config
    let mut config = config::load_config(cache.path())?;

    // Override provider if specified
    if let Some(provider) = provider_override {
        config.provider = provider;
    }

    // Get API key
    let api_key = config::get_api_key(&config.provider)?;

    // Determine which model to use (priority order):
    // 1. Project config model override (config.model from .reflex/config.toml)
    // 2. User-configured model for this provider (~/.reflex/config.toml)
    // 3. Provider default (handled by provider)
    let model = if config.model.is_some() {
        config.model.clone()
    } else {
        config::get_user_model(&config.provider)
    };

    // Create provider
    let provider = providers::create_provider(
        &config.provider,
        api_key,
        model,
        config::get_provider_options(&config.provider),
    )?;

    log::info!("Using provider: {} (model: {})", provider.name(), provider.default_model());

    // Build prompt with language injection
    let prompt = prompt::build_prompt(question, cache, additional_context.as_deref())?;

    log::debug!("Generated prompt ({} chars)", prompt.len());

    // Debug mode: output full prompt
    if debug {
        eprintln!("\n{}", "=".repeat(80));
        eprintln!("DEBUG: Full LLM Prompt (Standard Mode)");
        eprintln!("{}", "=".repeat(80));
        eprintln!("{}", prompt);
        eprintln!("{}\n", "=".repeat(80));
    }

    // Call LLM with retry logic
    let json_response = call_with_retry(&*provider, &prompt, 2, validate_query_response).await?;

    log::debug!("Received response ({} chars)", json_response.len());

    // Parse JSON response
    let response: schema::QueryResponse = serde_json::from_str(&json_response)
        .context("Failed to parse LLM response as JSON. The LLM may have returned invalid JSON.")?;

    // Validate response
    if response.queries.is_empty() {
        anyhow::bail!("LLM returned no queries");
    }

    log::info!("Generated {} quer{}", response.queries.len(), if response.queries.len() == 1 { "y" } else { "ies" });

    Ok(response)
}

/// Extract JSON from an LLM response that may contain extra formatting
///
/// Some LLMs wrap JSON in markdown code fences or embed it in explanatory text.
/// This function extracts the JSON object robustly.
///
/// Handles:
/// - ```json\n{...}\n``` (case-insensitive language tag)
/// - ```\n{...}\n```
/// - JSON embedded in surrounding text (finds first `{` to last matching `}`)
/// - {raw JSON} (no-op, returns as-is)
pub(crate) fn extract_json(text: &str) -> &str {
    let trimmed = text.trim();

    // Check for markdown code fence pattern (case-insensitive language tag)
    if trimmed.starts_with("```") && trimmed.ends_with("```") {
        // Find end of opening fence line
        let after_backticks = &trimmed[3..];
        let content_start = after_backticks.find('\n')
            .map(|i| 3 + i + 1)
            .unwrap_or(3);

        // Remove closing fence
        let content = &trimmed[content_start..trimmed.len() - 3];
        return content.trim();
    }

    // If it doesn't start with `{`, try to find JSON embedded in text
    if !trimmed.starts_with('{') {
        if let Some(start) = trimmed.find('{') {
            // Find the matching closing brace by counting depth
            let bytes = trimmed.as_bytes();
            let mut depth = 0i32;
            let mut last_close = start;
            let mut in_string = false;
            let mut escape_next = false;

            for (i, &b) in bytes[start..].iter().enumerate() {
                if escape_next {
                    escape_next = false;
                    continue;
                }
                match b {
                    b'\\' if in_string => escape_next = true,
                    b'"' => in_string = !in_string,
                    b'{' if !in_string => depth += 1,
                    b'}' if !in_string => {
                        depth -= 1;
                        if depth == 0 {
                            last_close = start + i;
                            break;
                        }
                    }
                    _ => {}
                }
            }

            if depth == 0 && last_close > start {
                return trimmed[start..=last_close].trim();
            }
        }
    }

    trimmed
}

/// Recursively convert string-encoded booleans in a JSON Value tree.
/// Handles LLMs that return `"true"` instead of `true`.
fn coerce_string_values(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::String(s) => {
            match s.as_str() {
                "true" => *value = serde_json::Value::Bool(true),
                "false" => *value = serde_json::Value::Bool(false),
                _ => {}
            }
        }
        serde_json::Value::Object(map) => {
            for v in map.values_mut() {
                coerce_string_values(v);
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr.iter_mut() {
                coerce_string_values(v);
            }
        }
        _ => {}
    }
}

/// Call LLM provider with retry logic
///
/// Retries up to `max_retries` times on:
/// - Network errors
/// - Invalid JSON responses
///
/// Uses exponential backoff between retries.
///
/// The `validator` function checks whether the cleaned response is valid
/// for the caller's expected schema. This allows each call site to validate
/// against the correct type (e.g., `AgenticResponse` vs `QueryResponse`).
///
/// NOTE: Exported for use by agentic module
pub(crate) async fn call_with_retry(
    provider: &dyn providers::LlmProvider,
    prompt: &str,
    max_retries: usize,
    validator: impl Fn(&str) -> Result<(), String>,
) -> Result<String> {
    let mut last_error = None;

    for attempt in 0..=max_retries {
        if attempt > 0 {
            log::warn!("Retrying LLM call (attempt {}/{})", attempt + 1, max_retries + 1);
        }

        match provider.complete(prompt, true).await {  // json_mode: true for query generation
            Ok(response) => {
                // Extract JSON from response (handles markdown fences, embedded text, etc.)
                let cleaned_response = extract_json(&response);

                // Coerce string-encoded booleans ("true" → true, "false" → false)
                // Some LLMs (e.g., DeepSeek via OpenRouter) return "true" instead of true
                let cleaned_response = match serde_json::from_str::<serde_json::Value>(cleaned_response) {
                    Ok(mut value) => {
                        coerce_string_values(&mut value);
                        value.to_string()
                    }
                    Err(_) => cleaned_response.to_string(), // Not valid JSON yet — let validator report the error
                };

                // Validate using caller-specified schema check
                match validator(&cleaned_response) {
                    Ok(()) => {
                        // Valid response - return the cleaned version
                        return Ok(cleaned_response);
                    }
                    Err(e) => {
                        if attempt < max_retries {
                            log::warn!(
                                "Invalid JSON response from LLM, retrying ({}/{}): {}",
                                attempt + 1,
                                max_retries,
                                e
                            );
                            last_error = Some(anyhow::anyhow!(
                                "Invalid JSON format: {}. Response: {}",
                                e,
                                cleaned_response
                            ));

                            // Exponential backoff: 500ms, 1s, 1.5s...
                            let delay_ms = 500 * (attempt as u64 + 1);
                            tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
                            continue;
                        } else {
                            // Final attempt failed
                            last_error = Some(anyhow::anyhow!(
                                "Invalid JSON format after {} attempts: {}. Response: {}",
                                max_retries + 1,
                                e,
                                cleaned_response
                            ));
                        }
                    }
                }
            }
            Err(e) => {
                if attempt < max_retries {
                    log::warn!(
                        "LLM API call failed, retrying ({}/{}): {}",
                        attempt + 1,
                        max_retries,
                        e
                    );

                    // Exponential backoff
                    let delay_ms = 500 * (attempt as u64 + 1);
                    tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
                }
                last_error = Some(e);
            }
        }
    }

    Err(last_error.unwrap())
}

/// Validator: response must parse as `QueryResponse` (requires `queries` field)
pub(crate) fn validate_query_response(json: &str) -> Result<(), String> {
    serde_json::from_str::<schema::QueryResponse>(json)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Validator: response must parse as `AgenticResponse` (requires `phase` + `reasoning`)
pub(crate) fn validate_agentic_response(json: &str) -> Result<(), String> {
    serde_json::from_str::<schema_agentic::AgenticResponse>(json)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Validator: response must parse as either `AgenticResponse` or `QueryResponse`
///
/// Used for Phase 3 (generation) which has a fallback path accepting either format.
pub(crate) fn validate_agentic_or_query_response(json: &str) -> Result<(), String> {
    if serde_json::from_str::<schema_agentic::AgenticResponse>(json).is_ok() {
        return Ok(());
    }
    serde_json::from_str::<schema::QueryResponse>(json)
        .map(|_| ())
        .map_err(|e| format!("Neither AgenticResponse nor QueryResponse: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_with_json_label() {
        let input = r#"```json
{
  "queries": [
    {
      "command": "query \"User\" --symbols --kind class --lang php",
      "order": 1,
      "merge": true
    }
  ]
}
```"#;
        let expected = r#"{
  "queries": [
    {
      "command": "query \"User\" --symbols --kind class --lang php",
      "order": 1,
      "merge": true
    }
  ]
}"#;
        assert_eq!(extract_json(input), expected);
    }

    #[test]
    fn test_extract_json_without_json_label() {
        let input = r#"```
{"queries": []}
```"#;
        let expected = r#"{"queries": []}"#;
        assert_eq!(extract_json(input), expected);
    }

    #[test]
    fn test_extract_json_no_fences() {
        let input = r#"{"queries": []}"#;
        assert_eq!(extract_json(input), input);
    }

    #[test]
    fn test_extract_json_with_whitespace() {
        let input = r#"  ```json
{"queries": []}
```  "#;
        let expected = r#"{"queries": []}"#;
        assert_eq!(extract_json(input), expected);
    }

    #[test]
    fn test_extract_json_case_insensitive_tag() {
        let input = r#"```JSON
{"queries": []}
```"#;
        let expected = r#"{"queries": []}"#;
        assert_eq!(extract_json(input), expected);
    }

    #[test]
    fn test_extract_json_embedded_in_text() {
        let input = r#"Here is the response:
{"phase": "assessment", "reasoning": "need context", "needs_context": true, "tool_calls": []}
Some trailing text here."#;
        let expected = r#"{"phase": "assessment", "reasoning": "need context", "needs_context": true, "tool_calls": []}"#;
        assert_eq!(extract_json(input), expected);
    }

    #[test]
    fn test_extract_json_with_nested_braces_in_strings() {
        let input = r#"Result: {"reasoning": "found {braces} in code", "phase": "final"} done"#;
        let expected = r#"{"reasoning": "found {braces} in code", "phase": "final"}"#;
        assert_eq!(extract_json(input), expected);
    }

    #[test]
    fn test_validate_query_response() {
        assert!(validate_query_response(r#"{"queries": []}"#).is_ok());
        assert!(validate_query_response(r#"{"phase": "assessment"}"#).is_err());
    }

    #[test]
    fn test_validate_agentic_response() {
        let valid = r#"{"phase": "assessment", "reasoning": "test"}"#;
        assert!(validate_agentic_response(valid).is_ok());

        // Missing required `phase` field
        assert!(validate_agentic_response(r#"{"queries": []}"#).is_err());
    }

    #[test]
    fn test_validate_agentic_or_query_response() {
        // Both formats should pass
        assert!(validate_agentic_or_query_response(r#"{"queries": []}"#).is_ok());
        assert!(validate_agentic_or_query_response(
            r#"{"phase": "final", "reasoning": "done"}"#
        ).is_ok());

        // Invalid JSON should fail
        assert!(validate_agentic_or_query_response(r#"{"bad": true}"#).is_err());
    }

    #[test]
    fn test_module_structure() {
        // Just verify the module compiles
        assert!(true);
    }

    #[test]
    fn test_coerce_string_values_booleans() {
        let mut value = serde_json::json!({
            "needs_context": "true",
            "structure": "false",
            "reasoning": "some text"
        });
        coerce_string_values(&mut value);
        assert_eq!(value["needs_context"], serde_json::json!(true));
        assert_eq!(value["structure"], serde_json::json!(false));
        assert_eq!(value["reasoning"], serde_json::json!("some text"));
    }

    #[test]
    fn test_coerce_string_values_nested() {
        let mut value = serde_json::json!({
            "tool_calls": [
                {
                    "expand": "true",
                    "symbols": "false",
                    "query": "search term"
                }
            ],
            "outer": {
                "inner_bool": "true"
            }
        });
        coerce_string_values(&mut value);
        assert_eq!(value["tool_calls"][0]["expand"], serde_json::json!(true));
        assert_eq!(value["tool_calls"][0]["symbols"], serde_json::json!(false));
        assert_eq!(value["tool_calls"][0]["query"], serde_json::json!("search term"));
        assert_eq!(value["outer"]["inner_bool"], serde_json::json!(true));
    }

    #[test]
    fn test_coerce_string_values_preserves_strings() {
        let mut value = serde_json::json!({
            "reasoning": "this is true and false",
            "description": "truly amazing",
            "phase": "assessment",
            "command": "query \"true\" --symbols"
        });
        let original = value.clone();
        coerce_string_values(&mut value);
        assert_eq!(value, original);
    }

    #[test]
    fn test_extract_json_and_coerce_integration() {
        // Simulates a DeepSeek response with string-encoded booleans
        let llm_output = r#"```json
{
  "phase": "assessment",
  "reasoning": "need to find relevant files",
  "needs_context": "true",
  "tool_calls": [
    {
      "tool": "search_code",
      "query": "trigram",
      "symbols": "false",
      "expand": "true"
    }
  ]
}
```"#;

        let extracted = extract_json(llm_output);
        let mut value: serde_json::Value = serde_json::from_str(extracted).unwrap();
        coerce_string_values(&mut value);

        assert_eq!(value["needs_context"], serde_json::json!(true));
        assert_eq!(value["tool_calls"][0]["symbols"], serde_json::json!(false));
        assert_eq!(value["tool_calls"][0]["expand"], serde_json::json!(true));
        // String fields preserved
        assert_eq!(value["phase"], serde_json::json!("assessment"));
        assert_eq!(value["reasoning"], serde_json::json!("need to find relevant files"));
        assert_eq!(value["tool_calls"][0]["query"], serde_json::json!("trigram"));
    }
}
