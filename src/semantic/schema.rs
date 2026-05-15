//! Schema definitions for LLM responses

use serde::{Deserialize, Serialize};

/// LLM response containing rfx query commands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    /// Array of rfx commands to execute. Most queries should have 1 command.
    /// Only use multiple commands when absolutely necessary (e.g., cross-language search).
    pub queries: Vec<QueryCommand>,
    /// Optional human-readable message. Set when queries is empty to explain why
    /// (e.g., question is outside rfx scope).
    #[serde(default)]
    pub message: Option<String>,
}

/// Enhanced response for agentic mode containing both queries and results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgenticQueryResponse {
    /// Generated query commands
    pub queries: Vec<QueryCommand>,

    /// Executed search results (file-grouped)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub results: Vec<crate::models::FileGroupedResult>,

    /// Total count of matches across all results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_count: Option<usize>,

    /// Context gathered from tools (documentation, codebase structure, etc.)
    /// Used for answer generation when query results are insufficient
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gathered_context: Option<String>,

    /// Tools that were executed during context gathering (for UI display)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools_executed: Option<Vec<String>>,

    /// Conversational answer synthesized from results (only when --answer is used)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
}

/// A single rfx query command with execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCommand {
    /// The rfx query command WITHOUT the 'rfx' prefix
    ///
    /// Examples:
    /// - `query "TODO"`
    /// - `query "async" --symbols --kind function --lang typescript`
    pub command: String,

    /// Execution order (1-based). Commands execute sequentially by order.
    pub order: i32,

    /// Whether to include this result in final output
    /// - `false`: context-only (used to inform subsequent queries)
    /// - `true`: include in merged results shown to user
    pub merge: bool,
}

/// JSON schema for LLM prompt (OpenAI structured output)
pub const RESPONSE_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "queries": {
      "type": "array",
      "description": "Array of rfx commands to execute. Most queries should have 1 command.",
      "items": {
        "type": "object",
        "properties": {
          "command": {
            "type": "string",
            "description": "The rfx query command WITHOUT the 'rfx' prefix"
          },
          "order": {
            "type": "integer",
            "description": "Execution order (1-based, sequential)"
          },
          "merge": {
            "type": "boolean",
            "description": "Whether to include in final results (false = context-only)"
          }
        },
        "required": ["command", "order", "merge"]
      }
    }
  },
  "required": ["queries"]
}"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_single_query() {
        let json = r#"{
            "queries": [{
                "command": "query \"TODO\"",
                "order": 1,
                "merge": true
            }]
        }"#;

        let response: QueryResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.queries.len(), 1);
        assert_eq!(response.queries[0].command, "query \"TODO\"");
        assert_eq!(response.queries[0].order, 1);
        assert_eq!(response.queries[0].merge, true);
    }

    #[test]
    fn test_deserialize_multiple_queries() {
        let json = r#"{
            "queries": [
                {
                    "command": "query \"User\" --symbols --kind struct --lang rust",
                    "order": 1,
                    "merge": false
                },
                {
                    "command": "query \"User\" --lang rust",
                    "order": 2,
                    "merge": true
                }
            ]
        }"#;

        let response: QueryResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.queries.len(), 2);
        assert_eq!(response.queries[0].order, 1);
        assert_eq!(response.queries[0].merge, false);
        assert_eq!(response.queries[1].order, 2);
        assert_eq!(response.queries[1].merge, true);
    }
}
