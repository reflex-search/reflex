//! Anthropic API provider implementation

use super::LlmProvider;
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::json;
use std::time::Duration;

/// Anthropic provider for Claude models
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider
    pub fn new(api_key: String, model: Option<String>, timeout_secs: u64) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .context("Failed to build reqwest client")?;
        Ok(Self {
            client,
            api_key,
            model: model.unwrap_or_else(|| "claude-3-5-haiku-20241022".to_string()),
        })
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn complete(&self, prompt: &str, _json_mode: bool) -> Result<String> {
        // Anthropic doesn't have a JSON mode - it returns plain text by default
        // The json_mode parameter is ignored
        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&json!({
                "model": self.model,
                "max_tokens": 4000,
                "temperature": 0.1,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }))
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    anyhow::anyhow!(
                        "Anthropic request timed out. Set REFLEX_LLM_TIMEOUT_SECONDS to increase the limit."
                    )
                } else {
                    anyhow::anyhow!("Failed to send request to Anthropic API: {}", e)
                }
            })?;

        // Check for HTTP errors
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("Anthropic API error ({}): {}", status, error_text);
        }

        let data: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse Anthropic response as JSON")?;

        // Extract content from response
        let content = data["content"][0]["text"]
            .as_str()
            .context("No content in Anthropic response")?;

        Ok(content.to_string())
    }

    fn name(&self) -> &str {
        "anthropic"
    }

    fn default_model(&self) -> &str {
        "claude-3-5-haiku-20241022"
    }
}

/// Fetch chat models from Anthropic's `/v1/models` endpoint.
///
/// Unlike OpenAI, Anthropic only ships chat-capable Claude models on this
/// endpoint, so no allow/deny filtering is needed — every returned ID is a
/// valid chat model.
pub async fn fetch_models(api_key: &str) -> Result<Vec<String>> {
    let client = reqwest::Client::new();

    let response = client
        .get("https://api.anthropic.com/v1/models?limit=1000")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .context("Failed to fetch models from Anthropic")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        anyhow::bail!("Anthropic API error ({}): {}", status, body);
    }

    let data: serde_json::Value = response
        .json()
        .await
        .context("Failed to parse Anthropic models response")?;

    if data["has_more"].as_bool() == Some(true) {
        log::warn!(
            "Anthropic /v1/models returned has_more=true with limit=1000; pagination may be needed"
        );
    }

    let arr = data["data"]
        .as_array()
        .context("No 'data' array in Anthropic models response")?;

    let mut ids: Vec<String> = arr
        .iter()
        .filter_map(|m| m["id"].as_str().map(String::from))
        .collect();

    sort_anthropic_models(&mut ids);
    Ok(ids)
}

// Anthropic's API returns models in created_at-descending order, which we
// preserve. The pin only re-orders if the preferred model is present; if it
// isn't, the newest model naturally takes the recommended slot.
fn sort_anthropic_models(ids: &mut Vec<String>) {
    const PREFERRED: &str = "claude-sonnet-4-5";
    if let Some(pos) = ids.iter().position(|id| id == PREFERRED) {
        let pinned = ids.remove(pos);
        ids.insert(0, pinned);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_with_default_model() {
        let provider = AnthropicProvider::new("test-key".to_string(), None, 300).unwrap();
        assert_eq!(provider.name(), "anthropic");
        assert_eq!(provider.model, "claude-3-5-haiku-20241022");
    }

    #[test]
    fn test_new_with_custom_model() {
        let provider = AnthropicProvider::new(
            "test-key".to_string(),
            Some("claude-3-5-sonnet-20241022".to_string()),
            300
        ).unwrap();
        assert_eq!(provider.model, "claude-3-5-sonnet-20241022");
    }

    #[test]
    fn test_sort_pins_preferred_first() {
        let mut ids = vec![
            "claude-opus-4-7".to_string(),
            "claude-sonnet-4-6".to_string(),
            "claude-sonnet-4-5".to_string(),
            "claude-haiku-4-5".to_string(),
        ];
        sort_anthropic_models(&mut ids);
        assert_eq!(ids[0], "claude-sonnet-4-5");
    }

    #[test]
    fn test_sort_preserves_order_when_preferred_absent() {
        let mut ids = vec![
            "claude-opus-4-7".to_string(),
            "claude-sonnet-4-6".to_string(),
            "claude-haiku-4-5".to_string(),
        ];
        let before = ids.clone();
        sort_anthropic_models(&mut ids);
        assert_eq!(ids, before);
    }
}
