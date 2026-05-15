//! OpenAI API provider implementation

use super::LlmProvider;
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::json;
use std::time::Duration;

/// OpenAI provider for GPT models
pub struct OpenAiProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl OpenAiProvider {
    /// Create a new OpenAI provider
    pub fn new(api_key: String, model: Option<String>, timeout_secs: u64) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .context("Failed to build reqwest client")?;
        Ok(Self {
            client,
            api_key,
            model: model.unwrap_or_else(|| "gpt-4o-mini".to_string()),
        })
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    async fn complete(&self, prompt: &str, json_mode: bool) -> Result<String> {
        // GPT-5 models require max_completion_tokens instead of max_tokens
        let is_gpt5 = self.model.starts_with("gpt-5");

        let mut request_body = json!({
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
        });

        // Add JSON response format if requested
        if json_mode {
            request_body["response_format"] = json!({
                "type": "json_object"
            });
        }

        // Add the appropriate token limit parameter
        if is_gpt5 {
            request_body["max_completion_tokens"] = json!(4000);
        } else {
            request_body["max_tokens"] = json!(4000);
        }

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .context("Failed to send request to OpenAI API")?;

        // Check for HTTP errors
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("OpenAI API error ({}): {}", status, error_text);
        }

        let data: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse OpenAI response as JSON")?;

        // Extract content from response
        let content = data["choices"][0]["message"]["content"]
            .as_str()
            .context("No content in OpenAI response")?;

        Ok(content.to_string())
    }

    fn name(&self) -> &str {
        "openai"
    }

    fn default_model(&self) -> &str {
        "gpt-4o-mini"
    }
}

/// Fetch chat-capable models from OpenAI's `/v1/models` endpoint.
///
/// OpenAI's listing endpoint returns every model the key can access (chat,
/// embeddings, TTS, image, moderation, fine-tuned variants, dated snapshots),
/// and provides no category field — so filter by ID heuristics.
pub async fn fetch_models(api_key: &str) -> Result<Vec<String>> {
    let client = reqwest::Client::new();

    let response = client
        .get("https://api.openai.com/v1/models")
        .header("Authorization", format!("Bearer {}", api_key))
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .context("Failed to fetch models from OpenAI")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        anyhow::bail!("OpenAI API error ({}): {}", status, body);
    }

    let data: serde_json::Value = response
        .json()
        .await
        .context("Failed to parse OpenAI models response")?;

    let arr = data["data"]
        .as_array()
        .context("No 'data' array in OpenAI models response")?;

    let mut ids: Vec<String> = arr
        .iter()
        .filter_map(|m| m["id"].as_str().map(String::from))
        .filter(|id| is_chat_model(id))
        .collect();

    sort_with_recommended_first(&mut ids);
    Ok(ids)
}

fn is_chat_model(id: &str) -> bool {
    const ALLOWED_PREFIXES: &[&str] = &["gpt-5", "gpt-4", "o1", "o3", "o4", "chatgpt-"];
    if !ALLOWED_PREFIXES.iter().any(|p| id.starts_with(p)) {
        return false;
    }
    const DENIED_SUBSTRINGS: &[&str] = &[
        "-realtime",
        "-audio",
        "-image",
        "-tts",
        "-search",
        "-embedding",
        "-moderation",
        "-transcribe",
    ];
    !DENIED_SUBSTRINGS.iter().any(|d| id.contains(d))
}

// Pinning preserves the existing "first item is shown as (recommended)" UX in
// the configure wizard. If gpt-5.1 ages out of OpenAI's catalog, the
// reverse-alphabetical sort lets the next-newest model take the slot.
fn sort_with_recommended_first(ids: &mut Vec<String>) {
    ids.sort_by(|a, b| b.cmp(a));
    if let Some(pos) = ids.iter().position(|id| id == "gpt-5.1") {
        let pinned = ids.remove(pos);
        ids.insert(0, pinned);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_with_default_model() {
        let provider = OpenAiProvider::new("test-key".to_string(), None, 300).unwrap();
        assert_eq!(provider.name(), "openai");
        assert_eq!(provider.model, "gpt-4o-mini");
    }

    #[test]
    fn test_new_with_custom_model() {
        let provider =
            OpenAiProvider::new("test-key".to_string(), Some("gpt-4o".to_string()), 300).unwrap();
        assert_eq!(provider.model, "gpt-4o");
    }

    #[test]
    fn test_is_chat_model_keeps_chat_families() {
        for id in [
            "gpt-5.1",
            "gpt-5",
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
            "o1",
            "o1-mini",
            "o3",
            "o3-mini",
            "o4-mini",
            "chatgpt-4o-latest",
        ] {
            assert!(is_chat_model(id), "expected chat model: {}", id);
        }
    }

    #[test]
    fn test_is_chat_model_rejects_non_chat() {
        for id in [
            "text-embedding-3-large",
            "tts-1",
            "whisper-1",
            "dall-e-3",
            "gpt-image-1",
            "omni-moderation-latest",
            "gpt-4o-realtime-preview",
            "gpt-4o-audio-preview",
            "gpt-4o-transcribe",
            "gpt-4o-search-preview",
            "babbage-002",
            "davinci-002",
        ] {
            assert!(!is_chat_model(id), "expected non-chat model: {}", id);
        }
    }

    #[test]
    fn test_sort_pins_gpt_51_first() {
        let mut ids = vec![
            "gpt-4o".to_string(),
            "gpt-5".to_string(),
            "gpt-5.1".to_string(),
            "o3-mini".to_string(),
        ];
        sort_with_recommended_first(&mut ids);
        assert_eq!(ids[0], "gpt-5.1");
    }

    #[test]
    fn test_sort_without_gpt_51_falls_through() {
        let mut ids = vec!["gpt-4o".to_string(), "gpt-5".to_string(), "o3".to_string()];
        sort_with_recommended_first(&mut ids);
        // Reverse-alpha: o3 > gpt-5 > gpt-4o
        assert_eq!(ids[0], "o3");
    }
}
