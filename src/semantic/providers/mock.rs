//! Mock LLM provider for unit tests.
//!
//! Accepts a pre-configured list of response strings and returns them in order,
//! cycling back to the start once exhausted.  Gated behind `#[cfg(test)]` so it
//! never ships in release builds.
//!
//! For [`LlmProvider::complete_request`] it also records every request, can route
//! responses by request (`routed`), inject failures (`fail_first`), and reports
//! fake usage so budget tests have numbers to work with.

use super::{
    CompletionRequest, CompletionResponse, LlmProvider, ProviderError, ProviderErrorKind,
    StopReason, Usage,
};
use anyhow::Result;
use async_trait::async_trait;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// A request as the mock saw it.
#[cfg(test)]
#[derive(Debug, Clone)]
pub struct RecordedRequest {
    pub tag: String,
    pub system: String,
    pub user: String,
    pub output: &'static str,
    pub max_tokens: u32,
}

#[cfg(test)]
type Router = dyn Fn(&CompletionRequest<'_>) -> Result<String> + Send + Sync;

/// A deterministic, zero-I/O LLM provider for unit tests.
#[cfg(test)]
pub struct MockLlmProvider {
    responses: Vec<String>,
    call_count: Arc<AtomicUsize>,
    router: Option<Box<Router>>,
    fail_first: Mutex<(usize, Option<ProviderErrorKind>)>,
    requests: Mutex<Vec<RecordedRequest>>,
    model: String,
}

#[cfg(test)]
impl MockLlmProvider {
    /// Create a mock backed by a sequence of responses. Cycles on exhaustion.
    pub fn new(responses: Vec<impl Into<String>>) -> Self {
        Self {
            responses: responses.into_iter().map(Into::into).collect(),
            call_count: Arc::new(AtomicUsize::new(0)),
            router: None,
            fail_first: Mutex::new((0, None)),
            requests: Mutex::new(Vec::new()),
            model: "mock-model".to_string(),
        }
    }

    /// Convenience constructor for a single-response mock.
    pub fn single(response: impl Into<String>) -> Self {
        Self::new(vec![response])
    }

    /// Answer each structured request with `router(req)`. An `Err` whose root cause is a
    /// [`ProviderError`] is returned as is, so tests can script per-task failures.
    pub fn routed(
        router: impl Fn(&CompletionRequest<'_>) -> Result<String> + Send + Sync + 'static,
    ) -> Self {
        let mut m = Self::new(Vec::<String>::new());
        m.router = Some(Box::new(router));
        m
    }

    /// Fail the first `n` structured calls with `kind`.
    pub fn fail_first(self, n: usize, kind: ProviderErrorKind) -> Self {
        *self.fail_first.lock().unwrap() = (n, Some(kind));
        self
    }

    /// Report a different model id (for cache-key tests).
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    /// Number of times `complete` or `complete_request` has been called.
    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }

    /// Every structured request so far, in call order.
    pub fn requests(&self) -> Vec<RecordedRequest> {
        self.requests.lock().unwrap().clone()
    }

    /// A classified error, as a real provider would return it.
    pub fn error(kind: ProviderErrorKind) -> anyhow::Error {
        ProviderError {
            provider: "mock".into(),
            kind,
            status: None,
            retry_after: None,
            body: format!("injected {kind:?}"),
        }
        .into()
    }
}

#[cfg(test)]
#[async_trait]
impl LlmProvider for MockLlmProvider {
    async fn complete(&self, _prompt: &str, _json_mode: bool) -> Result<String> {
        if self.responses.is_empty() {
            anyhow::bail!("MockLlmProvider: no responses configured");
        }
        let idx = self.call_count.fetch_add(1, Ordering::SeqCst) % self.responses.len();
        Ok(self.responses[idx].clone())
    }

    fn name(&self) -> &str {
        "mock"
    }

    fn default_model(&self) -> &str {
        "mock-model"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn complete_request(&self, req: &CompletionRequest<'_>) -> Result<CompletionResponse> {
        self.requests.lock().unwrap().push(RecordedRequest {
            tag: req.tag.to_string(),
            system: req.system.to_string(),
            user: req.user.to_string(),
            output: req.output.label(),
            max_tokens: req.max_tokens,
        });
        {
            let mut ff = self.fail_first.lock().unwrap();
            if ff.0 > 0 {
                ff.0 -= 1;
                let kind = ff.1.unwrap_or(ProviderErrorKind::Server);
                self.call_count.fetch_add(1, Ordering::SeqCst);
                return Err(Self::error(kind));
            }
        }
        let text = match &self.router {
            Some(router) => {
                self.call_count.fetch_add(1, Ordering::SeqCst);
                router(req)?
            }
            None => self.complete(req.user, req.output.is_json()).await?,
        };
        Ok(CompletionResponse {
            usage: Some(Usage {
                input_tokens: ((req.system.len() + req.user.len()) / 4) as u32,
                output_tokens: (text.len() / 4) as u32,
                cached_input_tokens: 0,
            }),
            text,
            stop: StopReason::End,
            model: self.model.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::chat_session::{ChatSession, MessageRole};
    use crate::semantic::schema_agentic::{AgenticResponse, Phase, ToolCall};

    // ── Test 1: single-turn chat returns the configured response ─────────────

    #[tokio::test]
    async fn test_single_turn_returns_configured_response() {
        let payload =
            r#"{"queries": [{"command": "query \"fn main\"", "order": 1, "merge": true}]}"#;
        let mock = MockLlmProvider::single(payload);

        let response = mock
            .complete("find the main function", false)
            .await
            .unwrap();

        assert!(response.contains("fn main"));
        assert_eq!(mock.call_count(), 1);
    }

    // ── Test 2: multi-turn conversation accumulates history correctly ─────────

    #[tokio::test]
    async fn test_multi_turn_conversation_accumulates_history() {
        let mock = MockLlmProvider::new(vec![
            r#"{"queries": [{"command": "query \"foo\"", "order": 1, "merge": true}]}"#,
            r#"{"queries": [{"command": "query \"bar\"", "order": 1, "merge": true}]}"#,
        ]);

        let mut session = ChatSession::new("mock".to_string(), "mock-model".to_string());

        // Turn 1
        session.add_user_message("Find foo".to_string());
        let r1 = mock.complete("Find foo", false).await.unwrap();
        session.add_answer_message(r1.clone());

        // Turn 2
        session.add_user_message("Now find bar".to_string());
        let r2 = mock.complete("Now find bar", false).await.unwrap();
        session.add_answer_message(r2.clone());

        // History must contain all four messages
        assert_eq!(session.messages().len(), 4);
        assert_eq!(session.messages()[0].role, MessageRole::User);
        assert_eq!(session.messages()[1].role, MessageRole::AssistantAnswer);
        assert_eq!(session.messages()[2].role, MessageRole::User);
        assert_eq!(session.messages()[3].role, MessageRole::AssistantAnswer);

        assert_eq!(mock.call_count(), 2);

        // Response content matches each pre-configured reply
        assert!(r1.contains("foo"));
        assert!(r2.contains("bar"));

        // build_context() should surface both user turns
        let ctx = session.build_context();
        assert!(ctx.contains("Find foo"));
        assert!(ctx.contains("Now find bar"));
    }

    // ── Test 3: agentic tool-call response parsed and dispatched without I/O ──

    #[tokio::test]
    async fn test_agentic_tool_call_parsed_without_network_io() {
        let agentic_json = r#"{
            "phase": "assessment",
            "reasoning": "I need to gather context about the project structure",
            "needs_context": true,
            "tool_calls": [
                {
                    "type": "gather_context",
                    "structure": true,
                    "file_types": true,
                    "project_type": false,
                    "framework": false,
                    "entry_points": false,
                    "test_layout": false,
                    "config_files": false,
                    "depth": 2
                }
            ],
            "queries": [],
            "confidence": 0.0
        }"#;

        let mock = MockLlmProvider::single(agentic_json);

        // complete() never touches the network
        let response = mock
            .complete("What is the project structure?", true)
            .await
            .unwrap();

        // Parse as AgenticResponse — mirrors what the agentic loop does
        let parsed: AgenticResponse = serde_json::from_str(&response).unwrap();

        assert_eq!(parsed.phase, Phase::Assessment);
        assert!(parsed.needs_context);
        assert_eq!(parsed.tool_calls.len(), 1);

        // Verify the tool dispatches to the correct variant
        match &parsed.tool_calls[0] {
            ToolCall::GatherContext { params } => {
                assert!(params.structure);
                assert!(params.file_types);
                assert!(!params.project_type);
            }
            other => panic!("Expected GatherContext, got {:?}", other),
        }

        // Exactly one call — no retries, no network
        assert_eq!(mock.call_count(), 1);
        assert_eq!(mock.name(), "mock");
        assert_eq!(mock.default_model(), "mock-model");
    }

    // ── Test 4: responses cycle when exhausted ────────────────────────────────

    #[tokio::test]
    async fn test_responses_cycle_when_exhausted() {
        let mock = MockLlmProvider::new(vec!["alpha", "beta"]);

        let r1 = mock.complete("q1", false).await.unwrap();
        let r2 = mock.complete("q2", false).await.unwrap();
        let r3 = mock.complete("q3", false).await.unwrap(); // cycles

        assert_eq!(r1, "alpha");
        assert_eq!(r2, "beta");
        assert_eq!(r3, "alpha");
        assert_eq!(mock.call_count(), 3);
    }

    // ── Test 5: empty mock returns an error ───────────────────────────────────

    #[tokio::test]
    async fn test_empty_mock_returns_error() {
        let mock = MockLlmProvider::new(Vec::<String>::new());
        let result = mock.complete("anything", false).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("no responses configured")
        );
    }
}
