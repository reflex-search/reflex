//! Prompt template generation with codebase context injection

use crate::cache::CacheManager;
use anyhow::Result;

use super::context::CodebaseContext;

/// Embed prompt_template.md at compile time
const PROMPT_TEMPLATE: &str = include_str!("prompt_template.md");

/// Read project-specific configuration from REFLEX.md
///
/// Looks for REFLEX.md at the workspace root and returns its contents.
/// Returns None if the file doesn't exist or cannot be read.
fn read_project_config(workspace_root: &std::path::Path) -> Option<String> {
    let config_path = workspace_root.join("REFLEX.md");

    if !config_path.exists() {
        log::debug!("No REFLEX.md found at workspace root");
        return None;
    }

    match std::fs::read_to_string(&config_path) {
        Ok(contents) => {
            log::info!(
                "Loaded project configuration from REFLEX.md ({} bytes)",
                contents.len()
            );
            Some(contents)
        }
        Err(e) => {
            log::warn!("Failed to read REFLEX.md: {}", e);
            None
        }
    }
}

/// Build the complete prompt for the LLM
///
/// Extracts comprehensive codebase context and injects it into the prompt template
pub fn build_prompt(
    question: &str,
    cache: &CacheManager,
    additional_context: Option<&str>,
) -> Result<String> {
    // Extract comprehensive codebase context
    let context = CodebaseContext::extract(cache).unwrap_or_else(|e| {
        log::warn!(
            "Failed to extract full codebase context: {}. Using minimal context.",
            e
        );
        // Fallback to minimal context if extraction fails
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

    // Format context as a prompt-friendly string
    let context_str = if context.total_files == 0 {
        "No files indexed yet (empty codebase).".to_string()
    } else {
        context.to_prompt_string()
    };

    // Read project-specific configuration from REFLEX.md
    let workspace_root = cache.workspace_root();
    let project_config = read_project_config(&workspace_root).unwrap_or_else(|| {
        log::debug!("No project-specific configuration found, using defaults");
        "No project-specific instructions provided.".to_string()
    });

    // Format additional context
    let additional_context_str = additional_context
        .map(|ctx| format!("\n## Additional Context\n\n{}\n", ctx))
        .unwrap_or_default();

    // Inject context and project config into template
    let prompt = PROMPT_TEMPLATE
        .replace("{CODEBASE_CONTEXT}", &context_str)
        .replace("{PROJECT_CONFIG}", &project_config)
        .replace("{ADDITIONAL_CONTEXT}", &additional_context_str);

    // Build final prompt with JSON schema
    Ok(format!(
        r#"{prompt}

## Response Format

You MUST respond with valid JSON matching this exact schema:

```json
{schema}
```

## User Question

{question}

**IMPORTANT:** Return ONLY valid JSON. No markdown code blocks, no explanations outside the JSON structure. Just pure JSON.
"#,
        prompt = prompt,
        schema = crate::semantic::schema::RESPONSE_SCHEMA,
        question = question
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_prompt_contains_schema() {
        let temp_dir = TempDir::new().unwrap();
        let cache = CacheManager::new(temp_dir.path());

        let prompt = build_prompt("find todos", &cache, None).unwrap();

        assert!(prompt.contains("Response Format"));
        assert!(prompt.contains("find todos"));
        assert!(prompt.contains("JSON"));
    }

    #[test]
    fn test_prompt_injects_codebase_context() {
        let temp_dir = TempDir::new().unwrap();
        let cache = CacheManager::new(temp_dir.path());
        cache.init().unwrap();

        let prompt = build_prompt("test", &cache, None).unwrap();

        // Should handle empty codebase gracefully (note: includes period)
        assert!(
            prompt.contains("No files indexed yet (empty codebase).")
                || prompt.contains("Languages:")
        );
    }

    #[test]
    fn test_prompt_injects_additional_context() {
        let temp_dir = TempDir::new().unwrap();
        let cache = CacheManager::new(temp_dir.path());

        let additional_context = "## Project Structure\nservices/\n  backend/\n  frontend/";
        let prompt = build_prompt("test", &cache, Some(additional_context)).unwrap();

        assert!(prompt.contains("Additional Context"));
        assert!(prompt.contains("services/"));
        assert!(prompt.contains("backend/"));
    }
}
