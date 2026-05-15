//! OpenAI-compatible provider for self-hosted / third-party endpoints
//!
//! Supports any inference server that implements the OpenAI Chat Completions
//! schema, including LMStudio, llama.cpp server, vLLM, Ollama (via /v1 shim),
//! and litellm proxies. The API key is optional since many local servers do
//! not require authentication.
//!
//! Uses the OpenAI structured-outputs `json_schema` response_format (with a
//! permissive schema) so requests work across servers that have dropped
//! support for the legacy `json_object` mode (e.g. LMStudio).

use super::LlmProvider;
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::json;

pub struct OpenAiCompatibleProvider {
    client: reqwest::Client,
    api_key: Option<String>,
    model: String,
    base_url: String,
}

impl OpenAiCompatibleProvider {
    pub fn new(api_key: Option<String>, model: String, base_url: String, timeout_secs: u64) -> Result<Self> {
        if model.trim().is_empty() {
            anyhow::bail!(
                "openai-compatible provider requires a model name (no default available for self-hosted endpoints)"
            );
        }
        if base_url.trim().is_empty() {
            anyhow::bail!("openai-compatible provider requires a non-empty base_url");
        }

        let normalized_base = base_url.trim_end_matches('/').to_string();

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .context("Failed to build reqwest client")?;
        Ok(Self {
            client,
            api_key: api_key.filter(|k| !k.is_empty()),
            model,
            base_url: normalized_base,
        })
    }

    fn build_request_body(&self, prompt: &str, json_mode: bool) -> serde_json::Value {
        let mut body = json!({
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 4000,
        });

        if json_mode {
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "reflex_response",
                    "strict": false,
                    "schema": {
                        "type": "object",
                        "additionalProperties": true
                    }
                }
            });
        }

        body
    }
}

#[async_trait]
impl LlmProvider for OpenAiCompatibleProvider {
    async fn complete(&self, prompt: &str, json_mode: bool) -> Result<String> {
        let request_body = self.build_request_body(prompt, json_mode);

        let url = format!("{}/chat/completions", self.base_url);
        let mut request = self
            .client
            .post(&url)
            .header("Content-Type", "application/json");

        if let Some(api_key) = &self.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = request
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    anyhow::anyhow!(
                        "OpenAI-compatible request to {} timed out. Set REFLEX_LLM_TIMEOUT_SECONDS to increase the limit.",
                        url
                    )
                } else {
                    anyhow::anyhow!("Failed to send request to OpenAI-compatible endpoint at {}: {}", url, e)
                }
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!(
                "OpenAI-compatible API error from {} ({}): {}",
                url,
                status,
                error_text
            );
        }

        let data: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse OpenAI-compatible response as JSON")?;

        let content = data["choices"][0]["message"]["content"]
            .as_str()
            .context("No content in OpenAI-compatible response")?;

        Ok(content.to_string())
    }

    fn name(&self) -> &str {
        "openai-compatible"
    }

    fn default_model(&self) -> &str {
        ""
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_with_key_and_model() {
        let provider = OpenAiCompatibleProvider::new(
            Some("sk-test".to_string()),
            "qwen2.5-coder".to_string(),
            "http://localhost:1234/v1".to_string(),
            30,
        )
        .unwrap();
        assert_eq!(provider.name(), "openai-compatible");
        assert_eq!(provider.model, "qwen2.5-coder");
        assert_eq!(provider.base_url, "http://localhost:1234/v1");
        assert_eq!(provider.api_key.as_deref(), Some("sk-test"));
    }

    #[test]
    fn test_new_without_key_for_local_server() {
        let provider = OpenAiCompatibleProvider::new(
            None,
            "llama-3".to_string(),
            "http://localhost:11434/v1".to_string(),
            30,
        )
        .unwrap();
        assert!(provider.api_key.is_none());
    }

    #[test]
    fn test_new_treats_empty_key_as_none() {
        let provider = OpenAiCompatibleProvider::new(
            Some(String::new()),
            "llama-3".to_string(),
            "http://localhost:11434/v1".to_string(),
            30,
        )
        .unwrap();
        assert!(provider.api_key.is_none());
    }

    #[test]
    fn test_new_trims_trailing_slash_from_base_url() {
        let provider = OpenAiCompatibleProvider::new(
            None,
            "llama-3".to_string(),
            "http://localhost:1234/v1/".to_string(),
            30,
        )
        .unwrap();
        assert_eq!(provider.base_url, "http://localhost:1234/v1");
    }

    #[test]
    fn test_new_rejects_empty_model() {
        let result = OpenAiCompatibleProvider::new(
            None,
            String::new(),
            "http://localhost:1234/v1".to_string(),
            30,
        );
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("model name"));
        }
    }

    #[test]
    fn test_build_request_body_with_json_mode_uses_json_schema() {
        let provider = OpenAiCompatibleProvider::new(
            None,
            "qwen2.5-coder".to_string(),
            "http://localhost:1234/v1".to_string(),
            30
        )
        .unwrap();
        let body = provider.build_request_body("hi", true);
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(body["response_format"]["json_schema"]["name"], "reflex_response");
        assert_eq!(body["response_format"]["json_schema"]["strict"], false);
        assert_eq!(body["response_format"]["json_schema"]["schema"]["type"], "object");
        assert_eq!(
            body["response_format"]["json_schema"]["schema"]["additionalProperties"],
            true
        );
        assert_eq!(body["model"], "qwen2.5-coder");
    }

    #[test]
    fn test_build_request_body_without_json_mode_omits_response_format() {
        let provider = OpenAiCompatibleProvider::new(
            None,
            "qwen2.5-coder".to_string(),
            "http://localhost:1234/v1".to_string(),
            30
        )
        .unwrap();
        let body = provider.build_request_body("hi", false);
        assert!(body.get("response_format").is_none());
    }

    #[test]
    fn test_new_rejects_empty_base_url() {
        let result = OpenAiCompatibleProvider::new(
            None,
            "llama-3".to_string(),
            String::new(),
            30,
        );
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("base_url"));
        }
    }

    #[tokio::test]
    async fn timeout_fires_within_2x_configured_value() {
        use std::time::Instant;
        use tokio::io::AsyncReadExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                if let Ok((mut stream, _)) = listener.accept().await {
                    tokio::spawn(async move {
                        let mut buf = [0u8; 1024];
                        loop { match stream.read(&mut buf).await { Ok(0) | Err(_) => break, Ok(_) => {} } }
                    });
                }
            }
        });

        let provider = OpenAiCompatibleProvider::new(
            None, "test-model".to_string(), format!("http://{}", addr), 1,
        ).unwrap();

        let start = Instant::now();
        let result = provider.complete("hello", false).await;
        let elapsed = start.elapsed();

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("timed out"), "expected timed out in error");
        assert!(elapsed.as_secs() < 2, "timeout too long: {}s", elapsed.as_secs());
    }
}
