//! OpenRouter API provider implementation
//!
//! OpenRouter is an OpenAI-compatible API aggregator that routes requests
//! to 200+ models across providers (Claude, GPT, Gemini, Llama, etc.).
//! It adds a "sort" strategy for provider routing (by price, speed, or throughput).

use super::LlmProvider;
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::json;

/// Model info fetched from OpenRouter API
#[derive(Debug, Clone)]
pub struct OpenRouterModel {
    pub id: String,
    pub name: String,
    pub prompt_price: f64,      // USD per million tokens
    pub completion_price: f64,  // USD per million tokens
    pub context_length: u64,
}

/// Fetch available models from OpenRouter API
pub async fn fetch_models(api_key: &str) -> Result<Vec<OpenRouterModel>> {
    let client = reqwest::Client::new();

    let response = client
        .get("https://openrouter.ai/api/v1/models")
        .header("Authorization", format!("Bearer {}", api_key))
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .context("Failed to fetch models from OpenRouter")?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        anyhow::bail!("OpenRouter API error ({}): {}", status, error_text);
    }

    let data: serde_json::Value = response
        .json()
        .await
        .context("Failed to parse OpenRouter models response")?;

    let models_array = data["data"]
        .as_array()
        .context("No 'data' array in OpenRouter models response")?;

    let mut models: Vec<OpenRouterModel> = models_array
        .iter()
        .filter_map(|m| {
            let id = m["id"].as_str()?;
            let name = m["name"].as_str().unwrap_or(id);

            // Skip models without prompt/completion pricing (image, audio, embedding models)
            let prompt_str = m["pricing"]["prompt"].as_str()?;
            let completion_str = m["pricing"]["completion"].as_str()?;

            let prompt_per_token: f64 = prompt_str.parse().ok()?;
            let completion_per_token: f64 = completion_str.parse().ok()?;

            // Skip free/zero-cost models that are likely non-text or test endpoints
            // Also skip if both are zero (often indicates non-functional endpoints)
            if prompt_per_token < 0.0 || completion_per_token < 0.0 {
                return None;
            }

            let context_length = m["context_length"].as_u64().unwrap_or(0);

            Some(OpenRouterModel {
                id: id.to_string(),
                name: name.to_string(),
                prompt_price: prompt_per_token * 1_000_000.0,
                completion_price: completion_per_token * 1_000_000.0,
                context_length,
            })
        })
        .collect();

    models.sort_by(|a, b| a.id.cmp(&b.id));

    Ok(models)
}

/// OpenRouter provider (OpenAI-compatible API with provider routing)
pub struct OpenRouterProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    sort: String,
}

impl OpenRouterProvider {
    /// Create a new OpenRouter provider
    ///
    /// # Arguments
    /// * `api_key` - OpenRouter API key
    /// * `model` - Optional model override (default: anthropic/claude-sonnet-4)
    /// * `sort` - Optional sort strategy: "price", "speed", or "throughput" (default: "price")
    pub fn new(api_key: String, model: Option<String>, sort: Option<String>) -> Result<Self> {
        // Normalize sort value: map legacy "speed" to the correct API value "latency"
        let sort = sort
            .map(|s| if s == "speed" { "latency".to_string() } else { s })
            .unwrap_or_else(|| "price".to_string());
        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            model: model.unwrap_or_else(|| "anthropic/claude-sonnet-4".to_string()),
            sort,
        })
    }
}

#[async_trait]
impl LlmProvider for OpenRouterProvider {
    async fn complete(&self, prompt: &str, json_mode: bool) -> Result<String> {
        let messages = vec![json!({
            "role": "user",
            "content": prompt
        })];

        let mut request_body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4000,
            "provider": {
                "sort": self.sort,
                "allow_fallbacks": true
            }
        });

        // Add JSON response format if requested
        if json_mode {
            request_body["response_format"] = json!({
                "type": "json_object"
            });
        }

        let response = self
            .client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("HTTP-Referer", "https://github.com/reflex-search/reflex")
            .header("X-Title", "Reflex")
            .json(&request_body)
            .timeout(std::time::Duration::from_secs(60))
            .send()
            .await
            .map_err(|e| {
                log::error!("OpenRouter API request failed: {}", e);
                if e.is_timeout() {
                    log::error!("  Reason: Request timeout (>60s)");
                } else if e.is_connect() {
                    log::error!("  Reason: Connection failed");
                } else if e.is_request() {
                    log::error!("  Reason: Invalid request");
                }
                anyhow::anyhow!("Failed to send request to OpenRouter API: {}", e)
            })?;

        // Check for HTTP errors
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            let error_msg = match status.as_u16() {
                429 => {
                    log::warn!("OpenRouter rate limit exceeded: {}", error_text);
                    "Rate limit exceeded (try again in a few seconds)".to_string()
                }
                503 | 502 | 504 => {
                    log::warn!("OpenRouter service unavailable ({}): {}", status, error_text);
                    format!("OpenRouter service temporarily unavailable ({})", status)
                }
                401 => {
                    log::error!("OpenRouter authentication failed: {}", error_text);
                    "Authentication failed - check API key".to_string()
                }
                _ => {
                    log::error!("OpenRouter API error ({}): {}", status, error_text);
                    format!("API error ({}): {}", status, error_text)
                }
            };

            anyhow::bail!("{}", error_msg);
        }

        let data: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse OpenRouter response as JSON")?;

        // Extract content from response (OpenAI-compatible format)
        let content = data["choices"][0]["message"]["content"]
            .as_str()
            .context("No content in OpenRouter response")?;

        Ok(content.to_string())
    }

    fn name(&self) -> &str {
        "openrouter"
    }

    fn default_model(&self) -> &str {
        "anthropic/claude-sonnet-4"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_with_defaults() {
        let provider = OpenRouterProvider::new("test-key".to_string(), None, None).unwrap();
        assert_eq!(provider.name(), "openrouter");
        assert_eq!(provider.model, "anthropic/claude-sonnet-4");
        assert_eq!(provider.sort, "price");
    }

    #[test]
    fn test_new_with_custom_model_and_sort() {
        let provider = OpenRouterProvider::new(
            "test-key".to_string(),
            Some("openai/gpt-4o-mini".to_string()),
            Some("latency".to_string()),
        )
        .unwrap();
        assert_eq!(provider.model, "openai/gpt-4o-mini");
        assert_eq!(provider.sort, "latency");
    }

    #[test]
    fn test_new_maps_legacy_speed_to_latency() {
        let provider = OpenRouterProvider::new(
            "test-key".to_string(),
            None,
            Some("speed".to_string()),
        )
        .unwrap();
        assert_eq!(provider.sort, "latency");
    }

    #[test]
    fn test_openrouter_model_pricing_conversion() {
        // Simulate what fetch_models does with per-token pricing strings
        let prompt_str = "0.000003";
        let completion_str = "0.000015";

        let prompt_per_token: f64 = prompt_str.parse().unwrap();
        let completion_per_token: f64 = completion_str.parse().unwrap();

        let prompt_per_million = prompt_per_token * 1_000_000.0;
        let completion_per_million = completion_per_token * 1_000_000.0;

        assert!((prompt_per_million - 3.0).abs() < 0.001);
        assert!((completion_per_million - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_openrouter_model_struct() {
        let model = OpenRouterModel {
            id: "anthropic/claude-sonnet-4".to_string(),
            name: "Anthropic: Claude Sonnet 4".to_string(),
            prompt_price: 3.0,
            completion_price: 15.0,
            context_length: 200000,
        };

        assert_eq!(model.id, "anthropic/claude-sonnet-4");
        assert_eq!(model.prompt_price, 3.0);
        assert_eq!(model.completion_price, 15.0);
        assert_eq!(model.context_length, 200000);
    }
}
