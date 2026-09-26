//! Pure request builders, response parsers and error classifiers for
//! [`CompletionRequest`](super::CompletionRequest).
//!
//! Kept free of I/O so the exact wire bodies can be unit-tested.

use super::{
    CompletionRequest, CompletionResponse, OutputMode, ProviderError, ProviderErrorKind,
    StopReason, Usage,
};
use serde_json::{Value, json};
use std::time::Duration;

// ── Errors ──────────────────────────────────────────────────────────────────

/// Classify an HTTP error response.
pub fn classify_status(
    provider: &str,
    status: u16,
    retry_after: Option<&str>,
    body: String,
) -> ProviderError {
    let kind = match status {
        401 | 403 => ProviderErrorKind::Auth,
        404 => ProviderErrorKind::NotFound,
        408 => ProviderErrorKind::Timeout,
        409 | 425 | 429 => ProviderErrorKind::RateLimited,
        500..=599 => ProviderErrorKind::Server,
        _ => ProviderErrorKind::BadRequest,
    };
    ProviderError {
        provider: provider.to_string(),
        kind,
        status: Some(status),
        retry_after: retry_after.and_then(parse_retry_after),
        body: truncate_body(body),
    }
}

/// Classify a transport error (no HTTP status).
pub fn classify_transport(provider: &str, err: &reqwest::Error) -> ProviderError {
    let kind = if err.is_timeout() {
        ProviderErrorKind::Timeout
    } else {
        ProviderErrorKind::Network
    };
    ProviderError {
        provider: provider.to_string(),
        kind,
        status: None,
        retry_after: None,
        body: err.to_string(),
    }
}

/// A 2xx response whose body we could not use.
pub fn malformed(provider: &str, detail: impl Into<String>) -> ProviderError {
    ProviderError {
        provider: provider.to_string(),
        kind: ProviderErrorKind::Malformed,
        status: None,
        retry_after: None,
        body: detail.into(),
    }
}

/// `Retry-After` in delta-seconds. HTTP-date values are ignored.
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    let secs: f64 = value.trim().parse().ok()?;
    (secs.is_finite() && secs >= 0.0).then(|| Duration::from_secs_f64(secs.min(600.0)))
}

fn truncate_body(mut body: String) -> String {
    const MAX: usize = 2000;
    if body.len() > MAX {
        let mut cut = MAX;
        while !body.is_char_boundary(cut) {
            cut -= 1;
        }
        body.truncate(cut);
        body.push('…');
    }
    body
}

/// True when a 400 body says the endpoint does not understand the requested output format,
/// so the caller can downgrade and retry.
pub fn is_output_format_rejection(err: &ProviderError) -> bool {
    if err.kind != ProviderErrorKind::BadRequest {
        return false;
    }
    let b = err.body.to_ascii_lowercase();
    [
        "response_format",
        "json_schema",
        "output_config",
        "structured output",
    ]
    .iter()
    .any(|needle| b.contains(needle))
}

/// Send a JSON body and return the parsed JSON, classifying every failure.
pub async fn post_json(
    provider: &str,
    request: reqwest::RequestBuilder,
    body: &Value,
) -> Result<Value, ProviderError> {
    let response = request
        .json(body)
        .send()
        .await
        .map_err(|e| classify_transport(provider, &e))?;
    let status = response.status();
    if !status.is_success() {
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let text = response.text().await.unwrap_or_default();
        return Err(classify_status(
            provider,
            status.as_u16(),
            retry_after.as_deref(),
            text,
        ));
    }
    response
        .json::<Value>()
        .await
        .map_err(|e| malformed(provider, format!("response body is not JSON: {e}")))
}

// ── OpenAI chat completions (OpenAI, OpenRouter, openai-compatible) ─────────

/// How to express the output contract on an OpenAI-style endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFormat {
    /// `response_format: json_schema` (strict).
    JsonSchema,
    /// `response_format: json_object`.
    JsonObject,
    /// No `response_format`; the prompt alone asks for JSON.
    None,
}

impl ChatFormat {
    /// The format a request asks for, before any downgrade.
    pub fn requested(output: &OutputMode<'_>) -> Self {
        match output {
            OutputMode::Text => ChatFormat::None,
            OutputMode::JsonObject => ChatFormat::JsonObject,
            OutputMode::JsonSchema { .. } => ChatFormat::JsonSchema,
        }
    }

    /// One step weaker.
    pub fn downgrade(self) -> Self {
        match self {
            ChatFormat::JsonSchema => ChatFormat::JsonObject,
            ChatFormat::JsonObject | ChatFormat::None => ChatFormat::None,
        }
    }

    pub fn as_u8(self) -> u8 {
        match self {
            ChatFormat::JsonSchema => 0,
            ChatFormat::JsonObject => 1,
            ChatFormat::None => 2,
        }
    }

    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => ChatFormat::JsonSchema,
            1 => ChatFormat::JsonObject,
            _ => ChatFormat::None,
        }
    }

    /// The weaker of two formats.
    pub fn min(self, other: Self) -> Self {
        Self::from_u8(self.as_u8().max(other.as_u8()))
    }
}

/// OpenAI reasoning models (gpt-5*, o1/o3/o4…) take `max_completion_tokens` and reject
/// a non-default `temperature`.
pub fn is_openai_reasoning_model(model: &str) -> bool {
    let m = model.rsplit('/').next().unwrap_or(model);
    m.starts_with("gpt-5")
        || (m.starts_with('o') && m[1..].starts_with(|c: char| c.is_ascii_digit()))
}

/// Build a chat-completions body.
pub fn chat_body(model: &str, req: &CompletionRequest<'_>, format: ChatFormat) -> Value {
    let mut messages = Vec::with_capacity(2);
    if !req.system.is_empty() {
        messages.push(json!({"role": "system", "content": req.system}));
    }
    messages.push(json!({"role": "user", "content": req.user}));

    let mut body = json!({ "model": model, "messages": messages });
    let reasoning = is_openai_reasoning_model(model);
    if reasoning {
        body["max_completion_tokens"] = json!(req.max_tokens);
    } else {
        body["max_tokens"] = json!(req.max_tokens);
        body["temperature"] = json!(req.temperature);
    }

    match (format, req.output) {
        (ChatFormat::JsonSchema, OutputMode::JsonSchema { name, schema }) => {
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": { "name": name, "strict": true, "schema": schema }
            });
        }
        (ChatFormat::JsonSchema | ChatFormat::JsonObject, o) if o.is_json() => {
            body["response_format"] = json!({"type": "json_object"});
        }
        _ => {}
    }
    body
}

/// Send a chat-completions request, downgrading the output format when the endpoint
/// rejects it. `format_cap` remembers the weakest format that worked, for later calls.
/// `extend` adds provider-specific fields (e.g. OpenRouter routing) to each body.
#[allow(clippy::too_many_arguments)]
pub async fn chat_complete(
    provider: &str,
    client: &reqwest::Client,
    url: &str,
    bearer: Option<&str>,
    headers: &[(&str, &str)],
    model: &str,
    req: &CompletionRequest<'_>,
    format_cap: &std::sync::atomic::AtomicU8,
    extend: &(dyn Fn(&mut Value, ChatFormat) + Sync),
) -> anyhow::Result<CompletionResponse> {
    use std::sync::atomic::Ordering;
    let mut format = ChatFormat::requested(&req.output)
        .min(ChatFormat::from_u8(format_cap.load(Ordering::Relaxed)));
    loop {
        let mut body = chat_body(model, req, format);
        extend(&mut body, format);
        let mut request = client.post(url);
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        if let Some(key) = bearer {
            request = request.header("Authorization", format!("Bearer {key}"));
        }
        match post_json(provider, request, &body).await {
            Ok(data) => return Ok(parse_chat(provider, &data, model)?),
            Err(e) if format != ChatFormat::None && is_output_format_rejection(&e) => {
                let weaker = format.downgrade();
                log::info!("{provider}: {model} rejected {format:?}; retrying with {weaker:?}");
                format_cap.fetch_max(weaker.as_u8(), Ordering::Relaxed);
                format = weaker;
            }
            Err(e) => return Err(e.into()),
        }
    }
}

/// Parse a chat-completions response.
pub fn parse_chat(
    provider: &str,
    data: &Value,
    fallback_model: &str,
) -> Result<CompletionResponse, ProviderError> {
    let choice = &data["choices"][0];
    let text = choice["message"]["content"]
        .as_str()
        .ok_or_else(|| malformed(provider, "no choices[0].message.content"))?
        .to_string();
    let stop = match choice["finish_reason"].as_str() {
        Some("stop") | None => StopReason::End,
        Some("length") => StopReason::MaxTokens,
        Some(other) => StopReason::Other(other.to_string()),
    };
    let usage = data.get("usage").filter(|u| u.is_object()).map(|u| Usage {
        input_tokens: as_u32(&u["prompt_tokens"]),
        output_tokens: as_u32(&u["completion_tokens"]),
        cached_input_tokens: as_u32(&u["prompt_tokens_details"]["cached_tokens"]),
    });
    let model = data["model"].as_str().unwrap_or(fallback_model).to_string();
    Ok(CompletionResponse {
        text,
        usage,
        stop,
        model,
    })
}

// ── Anthropic Messages API ──────────────────────────────────────────────────

/// Build a `/v1/messages` body.
///
/// - The system prompt is one cached block: every Pulse task of one kind shares it.
/// - `temperature` is never sent. Current Claude models reject sampling parameters,
///   and Pulse gets determinism from its cache, not from sampling.
/// - `JsonSchema` uses `output_config.format` when `native_schema` is true. `JsonObject`
///   has no native form; the prompt asks for JSON and the caller extracts it.
pub fn anthropic_body(model: &str, req: &CompletionRequest<'_>, native_schema: bool) -> Value {
    let mut body = json!({
        "model": model,
        "max_tokens": req.max_tokens,
        "messages": [{"role": "user", "content": req.user}],
    });
    if !req.system.is_empty() {
        body["system"] = json!([{
            "type": "text",
            "text": req.system,
            "cache_control": {"type": "ephemeral"}
        }]);
    }
    if native_schema && let OutputMode::JsonSchema { schema, .. } = req.output {
        body["output_config"] = json!({
            "format": { "type": "json_schema", "schema": schema }
        });
    }
    body
}

/// Parse a `/v1/messages` response. Only `text` blocks are read; `thinking` blocks are skipped.
pub fn parse_anthropic(
    provider: &str,
    data: &Value,
    fallback_model: &str,
) -> Result<CompletionResponse, ProviderError> {
    let blocks = data["content"]
        .as_array()
        .ok_or_else(|| malformed(provider, "no content array"))?;
    let text: String = blocks
        .iter()
        .filter(|b| b["type"] == "text")
        .filter_map(|b| b["text"].as_str())
        .collect();
    let stop = match data["stop_reason"].as_str() {
        Some("end_turn") | Some("stop_sequence") | None => StopReason::End,
        Some("max_tokens") => StopReason::MaxTokens,
        Some(other) => StopReason::Other(other.to_string()),
    };
    if text.is_empty() && stop == StopReason::End {
        return Err(malformed(provider, "response has no text block"));
    }
    let u = &data["usage"];
    let usage = u.is_object().then(|| Usage {
        input_tokens: as_u32(&u["input_tokens"])
            + as_u32(&u["cache_read_input_tokens"])
            + as_u32(&u["cache_creation_input_tokens"]),
        output_tokens: as_u32(&u["output_tokens"]),
        cached_input_tokens: as_u32(&u["cache_read_input_tokens"]),
    });
    let model = data["model"].as_str().unwrap_or(fallback_model).to_string();
    Ok(CompletionResponse {
        text,
        usage,
        stop,
        model,
    })
}

fn as_u32(v: &Value) -> u32 {
    v.as_u64()
        .map(|n| n.min(u32::MAX as u64) as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> Value {
        json!({"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false})
    }

    fn req<'a>(output: OutputMode<'a>) -> CompletionRequest<'a> {
        CompletionRequest {
            system: "SYS",
            user: "USER",
            output,
            max_tokens: 900,
            temperature: 0.0,
            tag: "t",
        }
    }

    #[test]
    fn classify_status_kinds() {
        let k = |s| classify_status("p", s, None, String::new()).kind;
        assert_eq!(k(401), ProviderErrorKind::Auth);
        assert_eq!(k(403), ProviderErrorKind::Auth);
        assert_eq!(k(404), ProviderErrorKind::NotFound);
        assert_eq!(k(400), ProviderErrorKind::BadRequest);
        assert_eq!(k(422), ProviderErrorKind::BadRequest);
        assert_eq!(k(408), ProviderErrorKind::Timeout);
        assert_eq!(k(429), ProviderErrorKind::RateLimited);
        assert_eq!(k(500), ProviderErrorKind::Server);
        assert_eq!(k(529), ProviderErrorKind::Server);
        assert!(ProviderErrorKind::Auth.is_fatal());
        assert!(!ProviderErrorKind::RateLimited.is_fatal());
        assert!(ProviderErrorKind::RateLimited.is_retryable());
        assert!(!ProviderErrorKind::Malformed.is_retryable());
    }

    #[test]
    fn retry_after_seconds() {
        assert_eq!(parse_retry_after("3"), Some(Duration::from_secs(3)));
        assert_eq!(
            parse_retry_after(" 1.5 "),
            Some(Duration::from_millis(1500))
        );
        assert_eq!(parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT"), None);
        assert_eq!(parse_retry_after("-1"), None);
        let e = classify_status("p", 429, Some("7"), "slow down".into());
        assert_eq!(e.retry_after, Some(Duration::from_secs(7)));
    }

    #[test]
    fn body_is_truncated_on_char_boundary() {
        let e = classify_status("p", 500, None, "é".repeat(3000));
        assert!(e.body.len() <= 2000 + '…'.len_utf8());
        assert!(e.body.ends_with('…'));
    }

    #[test]
    fn output_format_rejection_detection() {
        let e = classify_status("p", 400, None, "Unknown parameter: response_format".into());
        assert!(is_output_format_rejection(&e));
        let e = classify_status("p", 400, None, "max_tokens too large".into());
        assert!(!is_output_format_rejection(&e));
        let e = classify_status("p", 500, None, "json_schema".into());
        assert!(!is_output_format_rejection(&e));
    }

    #[test]
    fn chat_body_json_schema_strict() {
        let s = schema();
        let b = chat_body(
            "gpt-4o-mini",
            &req(OutputMode::JsonSchema {
                name: "out",
                schema: &s,
            }),
            ChatFormat::JsonSchema,
        );
        assert_eq!(b["messages"][0]["role"], "system");
        assert_eq!(b["messages"][0]["content"], "SYS");
        assert_eq!(b["messages"][1]["content"], "USER");
        assert_eq!(b["max_tokens"], 900);
        assert_eq!(b["temperature"], 0.0);
        assert_eq!(b["response_format"]["type"], "json_schema");
        assert_eq!(b["response_format"]["json_schema"]["strict"], true);
        assert_eq!(b["response_format"]["json_schema"]["name"], "out");
        assert_eq!(b["response_format"]["json_schema"]["schema"], s);
    }

    #[test]
    fn chat_body_downgrades() {
        let s = schema();
        let r = req(OutputMode::JsonSchema {
            name: "out",
            schema: &s,
        });
        let b = chat_body("m", &r, ChatFormat::JsonObject);
        assert_eq!(b["response_format"]["type"], "json_object");
        let b = chat_body("m", &r, ChatFormat::None);
        assert!(b.get("response_format").is_none());
        let b = chat_body("m", &req(OutputMode::Text), ChatFormat::JsonSchema);
        assert!(b.get("response_format").is_none());
        assert_eq!(ChatFormat::JsonSchema.downgrade(), ChatFormat::JsonObject);
        assert_eq!(ChatFormat::JsonObject.downgrade(), ChatFormat::None);
        assert_eq!(
            ChatFormat::JsonSchema.min(ChatFormat::None),
            ChatFormat::None
        );
    }

    #[test]
    fn chat_body_reasoning_models() {
        for m in ["gpt-5", "gpt-5-mini", "o3", "o4-mini", "openai/gpt-5"] {
            assert!(is_openai_reasoning_model(m), "{m}");
            let b = chat_body(m, &req(OutputMode::Text), ChatFormat::None);
            assert_eq!(b["max_completion_tokens"], 900);
            assert!(b.get("max_tokens").is_none());
            assert!(b.get("temperature").is_none());
        }
        for m in [
            "gpt-4o-mini",
            "anthropic/claude-sonnet-4",
            "openrouter/auto",
            "qwen",
        ] {
            assert!(!is_openai_reasoning_model(m), "{m}");
        }
    }

    #[test]
    fn parse_chat_response() {
        let data = json!({
            "model": "gpt-4o-mini-2024",
            "choices": [{"message": {"content": "{\"a\":\"x\"}"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30,
                      "prompt_tokens_details": {"cached_tokens": 64}}
        });
        let r = parse_chat("openai", &data, "fallback").unwrap();
        assert_eq!(r.text, "{\"a\":\"x\"}");
        assert_eq!(r.stop, StopReason::MaxTokens);
        assert_eq!(r.model, "gpt-4o-mini-2024");
        assert_eq!(
            r.usage,
            Some(Usage {
                input_tokens: 120,
                output_tokens: 30,
                cached_input_tokens: 64
            })
        );
        let err = parse_chat("openai", &json!({"choices": []}), "m").unwrap_err();
        assert_eq!(err.kind, ProviderErrorKind::Malformed);
    }

    #[test]
    fn anthropic_body_shape() {
        let s = schema();
        let b = anthropic_body(
            "claude-sonnet-5",
            &req(OutputMode::JsonSchema {
                name: "out",
                schema: &s,
            }),
            true,
        );
        assert_eq!(b["model"], "claude-sonnet-5");
        assert_eq!(b["max_tokens"], 900);
        assert!(b.get("temperature").is_none());
        assert_eq!(b["system"][0]["text"], "SYS");
        assert_eq!(b["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(b["messages"][0]["role"], "user");
        assert_eq!(b["output_config"]["format"]["type"], "json_schema");
        assert_eq!(b["output_config"]["format"]["schema"], s);

        let b = anthropic_body(
            "m",
            &req(OutputMode::JsonSchema {
                name: "out",
                schema: &s,
            }),
            false,
        );
        assert!(b.get("output_config").is_none());
        let b = anthropic_body("m", &req(OutputMode::JsonObject), true);
        assert!(b.get("output_config").is_none());
    }

    #[test]
    fn parse_anthropic_skips_thinking_and_sums_cached_input() {
        let data = json!({
            "model": "claude-sonnet-5",
            "content": [
                {"type": "thinking", "thinking": ""},
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "cache_read_input_tokens": 90,
                      "cache_creation_input_tokens": 0, "output_tokens": 5}
        });
        let r = parse_anthropic("anthropic", &data, "m").unwrap();
        assert_eq!(r.text, "Hello world");
        assert_eq!(r.stop, StopReason::End);
        assert_eq!(r.usage.unwrap().input_tokens, 100);
        assert_eq!(r.usage.unwrap().cached_input_tokens, 90);

        let refusal = json!({"content": [], "stop_reason": "refusal"});
        let r = parse_anthropic("anthropic", &refusal, "m").unwrap();
        assert_eq!(r.stop, StopReason::Other("refusal".into()));

        let empty = json!({"content": [], "stop_reason": "end_turn"});
        assert!(parse_anthropic("anthropic", &empty, "m").is_err());
    }
}
