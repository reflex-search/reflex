//! LLM provider implementations

pub mod anthropic;
#[cfg(test)]
pub mod mock;
pub mod openai;
pub mod openai_compatible;
pub mod openrouter;
pub mod wire;

use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Duration;

/// What shape the model must answer in.
#[derive(Debug, Clone, Copy)]
pub enum OutputMode<'a> {
    /// Free text.
    Text,
    /// Any JSON object (OpenAI `json_object`, Anthropic prompt-only).
    JsonObject,
    /// JSON that matches `schema` (OpenAI strict `json_schema`, Anthropic forced tool use).
    JsonSchema {
        name: &'a str,
        schema: &'a serde_json::Value,
    },
}

impl OutputMode<'_> {
    /// Stable label for cache keys and logs.
    pub fn label(&self) -> &'static str {
        match self {
            OutputMode::Text => "text",
            OutputMode::JsonObject => "json_object",
            OutputMode::JsonSchema { .. } => "json_schema",
        }
    }

    pub fn is_json(&self) -> bool {
        !matches!(self, OutputMode::Text)
    }
}

/// A completion request with a separate system prompt and an explicit output contract.
///
/// Used by Pulse. `rfx ask` still uses [`LlmProvider::complete`].
#[derive(Debug, Clone, Copy)]
pub struct CompletionRequest<'a> {
    pub system: &'a str,
    pub user: &'a str,
    pub output: OutputMode<'a>,
    pub max_tokens: u32,
    pub temperature: f32,
    /// Task id for logs and test routing. Never sent to the provider.
    pub tag: &'a str,
}

/// Token usage as the provider reports it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Input tokens served from the provider's prompt cache, when reported.
    #[serde(default)]
    pub cached_input_tokens: u32,
}

/// Why the model stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    End,
    MaxTokens,
    Other(String),
}

#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub text: String,
    pub usage: Option<Usage>,
    pub stop: StopReason,
    pub model: String,
}

/// Classification of a provider failure. Fatal kinds will not succeed on retry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderErrorKind {
    Auth,
    NotFound,
    BadRequest,
    RateLimited,
    Server,
    Timeout,
    Network,
    Malformed,
}

impl ProviderErrorKind {
    /// Auth, NotFound and BadRequest fail the same way every time.
    pub fn is_fatal(&self) -> bool {
        matches!(self, Self::Auth | Self::NotFound | Self::BadRequest)
    }

    /// Transient failures worth a retry with backoff.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimited | Self::Server | Self::Timeout | Self::Network
        )
    }
}

/// A classified provider error. Returned inside `anyhow::Error`; recover it with
/// `err.downcast_ref::<ProviderError>()`.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{provider} API error ({kind:?}{}): {body}", status.map(|s| format!(", HTTP {s}")).unwrap_or_default())]
pub struct ProviderError {
    pub provider: String,
    pub kind: ProviderErrorKind,
    pub status: Option<u16>,
    pub retry_after: Option<Duration>,
    pub body: String,
}

/// What a provider supports natively.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProviderCaps {
    pub system_role: bool,
    pub json_object: bool,
    pub json_schema: bool,
    pub reports_usage: bool,
}

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

    /// The model this instance actually calls (configured, else default).
    fn model(&self) -> &str {
        self.default_model()
    }

    /// Native capabilities. The default claims none.
    fn caps(&self) -> ProviderCaps {
        ProviderCaps::default()
    }

    /// Send a structured request.
    ///
    /// The default joins system and user into one prompt and calls [`Self::complete`],
    /// so providers without native support still work (no usage, no schema enforcement).
    async fn complete_request(&self, req: &CompletionRequest<'_>) -> Result<CompletionResponse> {
        let prompt = if req.system.is_empty() {
            req.user.to_string()
        } else {
            format!("{}\n\n{}", req.system, req.user)
        };
        let text = self.complete(&prompt, req.output.is_json()).await?;
        Ok(CompletionResponse {
            text,
            usage: None,
            stop: StopReason::End,
            model: self.model().to_string(),
        })
    }
}

/// Default model name for a provider, for display when no model is configured.
pub fn default_model_for(provider_name: &str) -> &'static str {
    match provider_name.to_lowercase().as_str() {
        "openai" => "gpt-4o-mini",
        "anthropic" => "claude-3-5-haiku-20241022",
        "openrouter" => "anthropic/claude-sonnet-4",
        "openai-compatible" | "openai_compatible" => "",
        _ => "",
    }
}

/// Create a provider instance from name, API key, and request timeout.
pub fn create_provider(
    provider_name: &str,
    api_key: String,
    model: Option<String>,
    options: Option<HashMap<String, String>>,
    timeout_secs: u64,
) -> Result<Box<dyn LlmProvider>> {
    match provider_name.to_lowercase().as_str() {
        "openai" => Ok(Box::new(openai::OpenAiProvider::new(
            api_key,
            model,
            timeout_secs,
        )?)),
        "anthropic" => Ok(Box::new(anthropic::AnthropicProvider::new(
            api_key,
            model,
            timeout_secs,
        )?)),
        "openrouter" => {
            let sort = options.as_ref().and_then(|o| o.get("sort").cloned());
            Ok(Box::new(openrouter::OpenRouterProvider::new(
                api_key,
                model,
                sort,
                timeout_secs,
            )?))
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
            let key = if api_key.is_empty() {
                None
            } else {
                Some(api_key)
            };
            Ok(Box::new(openai_compatible::OpenAiCompatibleProvider::new(
                key,
                model,
                base_url,
                timeout_secs,
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
        let provider = create_provider("openai", "test-key".to_string(), None, None, 300);
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
        let provider = create_provider("OpenAI", "test-key".to_string(), None, None, 300);
        assert!(provider.is_ok());
    }

    #[test]
    fn test_create_provider_unknown() {
        let provider = create_provider("unknown", "test-key".to_string(), None, None, 300);
        assert!(provider.is_err());
        if let Err(e) = provider {
            assert!(e.to_string().contains("Unknown provider"));
        }
    }

    #[test]
    fn test_create_provider_openrouter() {
        let provider = create_provider("openrouter", "test-key".to_string(), None, None, 300);
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
            300,
        );
        assert!(provider.is_ok());
    }

    #[test]
    fn test_create_provider_openai_compatible_with_base_url() {
        let mut opts = HashMap::new();
        opts.insert(
            "base_url".to_string(),
            "http://localhost:1234/v1".to_string(),
        );
        let provider = create_provider(
            "openai-compatible",
            "test-key".to_string(),
            Some("qwen2.5-coder".to_string()),
            Some(opts),
            300,
        );
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().name(), "openai-compatible");
    }

    #[test]
    fn test_create_provider_openai_compatible_accepts_underscore_alias() {
        let mut opts = HashMap::new();
        opts.insert(
            "base_url".to_string(),
            "http://localhost:1234/v1".to_string(),
        );
        let provider = create_provider(
            "openai_compatible",
            "test-key".to_string(),
            Some("qwen2.5-coder".to_string()),
            Some(opts),
            300,
        );
        assert!(provider.is_ok());
    }

    #[test]
    fn test_create_provider_openai_compatible_allows_empty_api_key() {
        let mut opts = HashMap::new();
        opts.insert(
            "base_url".to_string(),
            "http://localhost:1234/v1".to_string(),
        );
        let provider = create_provider(
            "openai-compatible",
            String::new(),
            Some("qwen2.5-coder".to_string()),
            Some(opts),
            300,
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
            300,
        );
        assert!(provider.is_err());
        if let Err(e) = provider {
            assert!(e.to_string().contains("base_url"));
        }
    }

    #[test]
    fn test_create_provider_openai_compatible_requires_model() {
        let mut opts = HashMap::new();
        opts.insert(
            "base_url".to_string(),
            "http://localhost:1234/v1".to_string(),
        );
        let provider = create_provider("openai-compatible", String::new(), None, Some(opts), 300);
        assert!(provider.is_err());
    }
}
