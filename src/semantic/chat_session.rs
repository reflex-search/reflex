//! Chat session management for interactive `rfx ask` mode
//!
//! This module manages conversation state, message history, token tracking,
//! and context window management for the TUI chat interface.

use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};

/// Maximum context window sizes by provider (in tokens)
const OPENAI_CONTEXT_WINDOW: usize = 128_000;
const ANTHROPIC_CONTEXT_WINDOW: usize = 200_000;
/// Rough estimate: 4 characters per token (common heuristic)
const CHARS_PER_TOKEN: usize = 4;

/// A single message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Message role
    pub role: MessageRole,

    /// Message content
    pub content: String,

    /// Estimated token count for this message
    pub tokens: usize,

    /// Timestamp when message was created
    pub timestamp: DateTime<Local>,

    /// Optional metadata (queries executed, results found, etc.)
    pub metadata: Option<MessageMetadata>,
}

/// Message role in conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageRole {
    /// User input
    User,

    /// Assistant - Phase 1: Thinking/Assessment
    AssistantThinking,

    /// Assistant - Phase 2: Tool gathering results
    AssistantTools,

    /// Assistant - Phase 3: Generated queries
    AssistantQueries,

    /// Assistant - Phase 4: Execution status
    AssistantExecuting,

    /// Assistant - Phase 5: Final answer
    AssistantAnswer,

    /// System message (for compaction summaries, etc.)
    System,
}

/// Metadata attached to assistant messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMetadata {
    /// Generated queries (for AssistantQueries phase)
    #[serde(default)]
    pub queries: Vec<String>,

    /// Tool calls made (for AssistantTools phase)
    #[serde(default)]
    pub tool_calls: Vec<String>,

    /// Number of results found (for AssistantExecuting phase)
    #[serde(default)]
    pub results_count: usize,

    /// Execution time in milliseconds
    #[serde(default)]
    pub execution_time_ms: Option<u64>,

    /// Whether this needs more context (for AssistantThinking phase)
    #[serde(default)]
    pub needs_context: bool,
}

/// Chat session state
pub struct ChatSession {
    /// Conversation history
    messages: Vec<Message>,

    /// LLM provider name
    provider: String,

    /// Model name
    model: String,

    /// Context window limit for current model
    context_limit: usize,

    /// Total tokens used in conversation
    total_tokens: usize,
}

impl ChatSession {
    /// Create a new chat session
    pub fn new(provider: String, model: String) -> Self {
        let context_limit = Self::get_context_limit(&provider);

        Self {
            messages: Vec::new(),
            provider,
            model,
            context_limit,
            total_tokens: 0,
        }
    }

    /// Add a user message to the conversation
    pub fn add_user_message(&mut self, content: String) {
        let tokens = Self::estimate_tokens(&content);
        let message = Message {
            role: MessageRole::User,
            content,
            tokens,
            timestamp: Local::now(),
            metadata: None,
        };

        self.total_tokens += tokens;
        self.messages.push(message);
    }

    /// Add an assistant message to the conversation (generic)
    pub fn add_assistant_message(&mut self, content: String, role: MessageRole, metadata: Option<MessageMetadata>) {
        let tokens = Self::estimate_tokens(&content);
        let message = Message {
            role,
            content,
            tokens,
            timestamp: Local::now(),
            metadata,
        };

        self.total_tokens += tokens;
        self.messages.push(message);
    }

    /// Add a thinking/assessment message
    pub fn add_thinking_message(&mut self, reasoning: String, needs_context: bool) {
        let metadata = MessageMetadata {
            queries: Vec::new(),
            tool_calls: Vec::new(),
            results_count: 0,
            execution_time_ms: None,
            needs_context,
        };
        self.add_assistant_message(reasoning, MessageRole::AssistantThinking, Some(metadata));
    }

    /// Add a tool gathering message
    pub fn add_tools_message(&mut self, content: String, tool_calls: Vec<String>) {
        let metadata = MessageMetadata {
            queries: Vec::new(),
            tool_calls,
            results_count: 0,
            execution_time_ms: None,
            needs_context: false,
        };
        self.add_assistant_message(content, MessageRole::AssistantTools, Some(metadata));
    }

    /// Add a queries generated message
    pub fn add_queries_message(&mut self, queries: Vec<String>) {
        let content = format!("Generated {} queries", queries.len());
        let metadata = MessageMetadata {
            queries: queries.clone(),
            tool_calls: Vec::new(),
            results_count: 0,
            execution_time_ms: None,
            needs_context: false,
        };
        self.add_assistant_message(content, MessageRole::AssistantQueries, Some(metadata));
    }

    /// Add an execution status message
    pub fn add_execution_message(&mut self, results_count: usize, execution_time_ms: u64) {
        let content = format!("Found {} result{}", results_count, if results_count == 1 { "" } else { "s" });
        let metadata = MessageMetadata {
            queries: Vec::new(),
            tool_calls: Vec::new(),
            results_count,
            execution_time_ms: Some(execution_time_ms),
            needs_context: false,
        };
        self.add_assistant_message(content, MessageRole::AssistantExecuting, Some(metadata));
    }

    /// Add a final answer message
    pub fn add_answer_message(&mut self, answer: String) {
        self.add_assistant_message(answer, MessageRole::AssistantAnswer, None);
    }

    /// Add a system message (e.g., compaction summary)
    pub fn add_system_message(&mut self, content: String) {
        let tokens = Self::estimate_tokens(&content);
        let message = Message {
            role: MessageRole::System,
            content,
            tokens,
            timestamp: Local::now(),
            metadata: None,
        };

        self.total_tokens += tokens;
        self.messages.push(message);
    }

    /// Clear all messages and reset token count
    pub fn clear(&mut self) {
        self.messages.clear();
        self.total_tokens = 0;
    }

    /// Get all messages in the conversation
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Get total token count
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Get context window limit
    pub fn context_limit(&self) -> usize {
        self.context_limit
    }

    /// Get context usage as percentage (0.0 to 1.0)
    pub fn context_usage(&self) -> f32 {
        if self.context_limit == 0 {
            return 0.0;
        }
        (self.total_tokens as f32) / (self.context_limit as f32)
    }

    /// Check if we're approaching context limit (>80%)
    pub fn is_near_limit(&self) -> bool {
        self.context_usage() > 0.8
    }

    /// Check if we should suggest compaction (>90%)
    pub fn should_compact(&self) -> bool {
        self.context_usage() > 0.9
    }

    /// Get provider name
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// Get model name
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Update provider and model (for /model command)
    pub fn update_provider(&mut self, provider: String, model: String) {
        self.provider = provider.clone();
        self.model = model;
        self.context_limit = Self::get_context_limit(&provider);
    }

    /// Build conversation history for LLM prompt
    ///
    /// Returns a formatted string suitable for including in LLM prompts,
    /// containing all messages in chronological order.
    pub fn build_context(&self) -> String {
        let mut context = String::new();

        context.push_str("Previous conversation:\n");
        context.push_str("======================\n\n");

        for msg in &self.messages {
            match msg.role {
                MessageRole::User => {
                    context.push_str(&format!("User: {}\n\n", msg.content));
                }
                MessageRole::AssistantThinking
                | MessageRole::AssistantTools
                | MessageRole::AssistantQueries
                | MessageRole::AssistantExecuting
                | MessageRole::AssistantAnswer => {
                    context.push_str(&format!("Assistant: {}\n\n", msg.content));
                }
                MessageRole::System => {
                    context.push_str(&format!("[System Note: {}]\n\n", msg.content));
                }
            }
        }

        context
    }

    /// Compact old messages by summarizing them
    ///
    /// Keeps the last `keep_recent` messages verbatim and returns the older
    /// messages as a formatted string that can be sent to an LLM for summarization.
    ///
    /// Returns (old_messages_for_summary, kept_messages_count, tokens_to_compact)
    pub fn prepare_compaction(&self, keep_recent: usize) -> (String, usize, usize) {
        if self.messages.len() <= keep_recent {
            return (String::new(), self.messages.len(), 0);
        }

        let split_point = self.messages.len() - keep_recent;
        let old_messages = &self.messages[..split_point];

        let mut summary_text = String::new();
        let mut tokens_to_compact = 0;

        for msg in old_messages {
            tokens_to_compact += msg.tokens;

            match msg.role {
                MessageRole::User => {
                    summary_text.push_str(&format!("User: {}\n\n", msg.content));
                }
                MessageRole::AssistantThinking
                | MessageRole::AssistantTools
                | MessageRole::AssistantQueries
                | MessageRole::AssistantExecuting
                | MessageRole::AssistantAnswer => {
                    summary_text.push_str(&format!("Assistant: {}\n\n", msg.content));
                }
                MessageRole::System => {
                    summary_text.push_str(&format!("[System: {}]\n\n", msg.content));
                }
            }
        }

        (summary_text, old_messages.len(), tokens_to_compact)
    }

    /// Apply compaction by replacing old messages with a summary
    ///
    /// Removes the first `remove_count` messages and replaces them with
    /// a single system message containing the summary.
    pub fn apply_compaction(&mut self, remove_count: usize, summary: String) {
        if remove_count >= self.messages.len() {
            // Safety check: don't remove all messages
            return;
        }

        // Calculate tokens being removed
        let removed_tokens: usize = self.messages[..remove_count]
            .iter()
            .map(|m| m.tokens)
            .sum();

        // Remove old messages
        self.messages.drain(..remove_count);

        // Add summary as system message at the beginning
        let summary_tokens = Self::estimate_tokens(&summary);
        let summary_msg = Message {
            role: MessageRole::System,
            content: format!("Summary of previous conversation: {}", summary),
            tokens: summary_tokens,
            timestamp: Local::now(),
            metadata: None,
        };

        self.messages.insert(0, summary_msg);

        // Update total token count
        self.total_tokens = self.total_tokens - removed_tokens + summary_tokens;
    }

    /// Estimate token count from text (rough heuristic: ~4 chars per token)
    fn estimate_tokens(text: &str) -> usize {
        (text.len() + CHARS_PER_TOKEN - 1) / CHARS_PER_TOKEN
    }

    /// Get context window limit for a provider
    fn get_context_limit(provider: &str) -> usize {
        match provider.to_lowercase().as_str() {
            "openai" => OPENAI_CONTEXT_WINDOW,
            "anthropic" => ANTHROPIC_CONTEXT_WINDOW,
            _ => 32_000, // Conservative default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_session() {
        let session = ChatSession::new("openai".to_string(), "gpt-4o-mini".to_string());
        assert_eq!(session.messages().len(), 0);
        assert_eq!(session.total_tokens(), 0);
        assert_eq!(session.context_limit(), OPENAI_CONTEXT_WINDOW);
    }

    #[test]
    fn test_add_messages() {
        let mut session = ChatSession::new("anthropic".to_string(), "claude-3-5-haiku".to_string());

        session.add_user_message("Hello!".to_string());
        assert_eq!(session.messages().len(), 1);
        assert!(session.total_tokens() > 0);

        session.add_answer_message("Hi there!".to_string());
        assert_eq!(session.messages().len(), 2);
    }

    #[test]
    fn test_clear() {
        let mut session = ChatSession::new("openai".to_string(), "gpt-4o".to_string());
        session.add_user_message("Test".to_string());
        session.add_answer_message("Response".to_string());

        assert_eq!(session.messages().len(), 2);

        session.clear();
        assert_eq!(session.messages().len(), 0);
        assert_eq!(session.total_tokens(), 0);
    }

    #[test]
    fn test_context_usage() {
        let mut session = ChatSession::new("openai".to_string(), "gpt-4o-mini".to_string());
        assert_eq!(session.context_usage(), 0.0);

        // Add a message that's roughly 1/4 of the context window
        let large_text = "a".repeat(OPENAI_CONTEXT_WINDOW * CHARS_PER_TOKEN / 4);
        session.add_user_message(large_text);

        let usage = session.context_usage();
        assert!(usage > 0.2 && usage < 0.3); // Should be around 25%
    }

    #[test]
    fn test_prepare_compaction() {
        let mut session = ChatSession::new("openai".to_string(), "gpt-4o-mini".to_string());

        for i in 0..10 {
            session.add_user_message(format!("Message {}", i));
            session.add_answer_message(format!("Response {}", i));
        }

        let (summary_text, old_count, tokens) = session.prepare_compaction(4);

        assert_eq!(old_count, 16); // 20 messages - 4 kept = 16 old
        assert!(!summary_text.is_empty());
        assert!(tokens > 0);
    }

    #[test]
    fn test_apply_compaction() {
        let mut session = ChatSession::new("anthropic".to_string(), "claude".to_string());

        for i in 0..6 {
            session.add_user_message(format!("Q{}", i));
            session.add_answer_message(format!("A{}", i));
        }

        let initial_count = session.messages().len();
        let initial_tokens = session.total_tokens();

        session.apply_compaction(8, "This is a summary".to_string());

        // Should have: 1 summary + 4 kept messages = 5 total
        assert_eq!(session.messages().len(), 5);
        assert_eq!(session.messages()[0].role, MessageRole::System);

        // Token count should be updated
        assert!(session.total_tokens() < initial_tokens);
    }

    #[test]
    fn test_estimate_tokens() {
        let text = "Hello, world!"; // 13 chars
        let tokens = ChatSession::estimate_tokens(text);
        // Uses ceiling division: (13 + 4 - 1) / 4 = 16 / 4 = 4
        assert_eq!(tokens, (text.len() + CHARS_PER_TOKEN - 1) / CHARS_PER_TOKEN);
        assert_eq!(tokens, 4);
    }
}
