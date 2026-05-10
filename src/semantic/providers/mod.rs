//! LLM provider implementations

pub mod openai;
pub mod anthropic;
pub mod openrouter;
pub mod openai_compatible;

use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::HashMap;

/// Trait for LLM providers that generate structured query responses
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Send a prompt and get response
    ///
    /// # Arguments
    ///
    /// * `prompt` - The prompt to send to the LLM
    /// * `json_mode` - Whether to request JSON structured output (true) or plain text (false)
    ///
    /// When `json_mode` is true, the response should be valid JSON matching the QueryResponse schema.
    /// When `json_mode` is false, the response can be plain text (used for answer generation).
    async fn complete(&self, prompt: &str, json_mode: bool) -> Result<String>;

    /// Get provider name (for logging and error messages)
    fn name(&self) -> &str;

    /// Get default model identifier
    fn default_model(&self) -> &str;
}

/// Default model name for a provider, for display when no model is configured.
///
/// Mirrors the constructor defaults baked into each provider's `new()`. Returns
/// `""` for `openai-compatible` because self-hosted endpoints have no
/// predictable default — callers should render `""` as something like
/// `"(not configured)"` and refuse the LLM call until the user configures a
/// model. Returns `""` for unknown providers (callers should not invoke them).
pub fn default_model_for(provider_name: &str) -> &'static str {
    match provider_name.to_lowercase().as_str() {
        "openai" => "gpt-4o-mini",
        "anthropic" => "claude-3-5-haiku-20241022",
        "openrouter" => "anthropic/claude-sonnet-4",
        "openai-compatible" | "openai_compatible" => "",
        _ => "",
    }
}

/// Create a provider instance from name and API key
///
/// The `options` parameter allows passing provider-specific settings.
/// Currently used by OpenRouter for sort strategy (e.g., `{"sort": "price"}`).
/// Other providers ignore this parameter.
pub fn create_provider(
    provider_name: &str,
    api_key: String,
    model: Option<String>,
    options: Option<HashMap<String, String>>,
) -> Result<Box<dyn LlmProvider>> {
    match provider_name.to_lowercase().as_str() {
        "openai" => Ok(Box::new(openai::OpenAiProvider::new(api_key, model)?)),
        "anthropic" => Ok(Box::new(anthropic::AnthropicProvider::new(api_key, model)?)),
        "openrouter" => {
            let sort = options.as_ref().and_then(|o| o.get("sort").cloned());
            Ok(Box::new(openrouter::OpenRouterProvider::new(api_key, model, sort)?))
        }
        "openai-compatible" | "openai_compatible" => {
            let base_url = options
                .as_ref()
                .and_then(|o| o.get("base_url").cloned())
                .context(
                    "openai-compatible provider requires 'base_url' in options \
                     (set credentials.openai_compatible_base_url in ~/.reflex/config.toml \
                     or the OPENAI_COMPATIBLE_BASE_URL env var)",
                )?;
            let model = model.unwrap_or_default();
            let key = if api_key.is_empty() { None } else { Some(api_key) };
            Ok(Box::new(openai_compatible::OpenAiCompatibleProvider::new(
                key, model, base_url,
            )?))
        }
        _ => anyhow::bail!(
            "Unknown provider: {}. Supported: openai, anthropic, openrouter, openai-compatible",
            provider_name
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_provider_openai() {
        let provider = create_provider("openai", "test-key".to_string(), None, None);
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().name(), "openai");
    }

    #[test]
    fn test_default_model_for_known_providers() {
        assert_eq!(default_model_for("openai"), "gpt-4o-mini");
        assert_eq!(default_model_for("OpenAI"), "gpt-4o-mini");
        assert_eq!(default_model_for("anthropic"), "claude-3-5-haiku-20241022");
        assert_eq!(default_model_for("openrouter"), "anthropic/claude-sonnet-4");
        assert_eq!(default_model_for("openai-compatible"), "");
        assert_eq!(default_model_for("openai_compatible"), "");
        assert_eq!(default_model_for("unknown"), "");
    }

    #[test]
    fn test_create_provider_case_insensitive() {
        let provider = create_provider("OpenAI", "test-key".to_string(), None, None);
        assert!(provider.is_ok());
    }

    #[test]
    fn test_create_provider_unknown() {
        let provider = create_provider("unknown", "test-key".to_string(), None, None);
        assert!(provider.is_err());
        if let Err(e) = provider {
            assert!(e.to_string().contains("Unknown provider"));
        }
    }

    #[test]
    fn test_create_provider_openrouter() {
        let provider = create_provider("openrouter", "test-key".to_string(), None, None);
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().name(), "openrouter");
    }

    #[test]
    fn test_create_provider_openrouter_with_sort() {
        let mut opts = HashMap::new();
        opts.insert("sort".to_string(), "speed".to_string());
        let provider = create_provider(
            "openrouter",
            "test-key".to_string(),
            Some("openai/gpt-4o-mini".to_string()),
            Some(opts),
        );
        assert!(provider.is_ok());
    }

    #[test]
    fn test_create_provider_openai_compatible_with_base_url() {
        let mut opts = HashMap::new();
        opts.insert("base_url".to_string(), "http://localhost:1234/v1".to_string());
        let provider = create_provider(
            "openai-compatible",
            "test-key".to_string(),
            Some("qwen2.5-coder".to_string()),
            Some(opts),
        );
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().name(), "openai-compatible");
    }

    #[test]
    fn test_create_provider_openai_compatible_accepts_underscore_alias() {
        let mut opts = HashMap::new();
        opts.insert("base_url".to_string(), "http://localhost:1234/v1".to_string());
        let provider = create_provider(
            "openai_compatible",
            "test-key".to_string(),
            Some("qwen2.5-coder".to_string()),
            Some(opts),
        );
        assert!(provider.is_ok());
    }

    #[test]
    fn test_create_provider_openai_compatible_allows_empty_api_key() {
        let mut opts = HashMap::new();
        opts.insert("base_url".to_string(), "http://localhost:1234/v1".to_string());
        let provider = create_provider(
            "openai-compatible",
            String::new(),
            Some("qwen2.5-coder".to_string()),
            Some(opts),
        );
        assert!(provider.is_ok());
    }

    #[test]
    fn test_create_provider_openai_compatible_requires_base_url() {
        let provider = create_provider(
            "openai-compatible",
            String::new(),
            Some("qwen2.5-coder".to_string()),
            None,
        );
        assert!(provider.is_err());
        if let Err(e) = provider {
            assert!(e.to_string().contains("base_url"));
        }
    }

    #[test]
    fn test_create_provider_openai_compatible_requires_model() {
        let mut opts = HashMap::new();
        opts.insert("base_url".to_string(), "http://localhost:1234/v1".to_string());
        let provider = create_provider(
            "openai-compatible",
            String::new(),
            None,
            Some(opts),
        );
        assert!(provider.is_err());
    }
}
