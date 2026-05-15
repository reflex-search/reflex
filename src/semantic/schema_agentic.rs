//! Agentic schema definitions for multi-step reasoning and context gathering
//!
//! This module defines the schema for agentic `rfx ask` which allows the LLM to:
//! 1. Assess if it needs more context
//! 2. Gather context using tools (rfx context, exploratory queries)
//! 3. Generate final optimized queries
//! 4. Evaluate results and refine if needed

use serde::{Deserialize, Serialize};

/// Agentic response from LLM with tool calling support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgenticResponse {
    /// Current phase of the agentic loop
    pub phase: Phase,

    /// LLM's reasoning/thought process
    pub reasoning: String,

    /// Whether more context is needed before generating final queries
    #[serde(default)]
    pub needs_context: bool,

    /// Tool calls to gather additional context (only for assessment/gathering phases)
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,

    /// Final query commands (only for final phase)
    #[serde(default)]
    pub queries: Vec<super::schema::QueryCommand>,

    /// Confidence score (0.0-1.0) in the generated queries
    #[serde(default)]
    pub confidence: f32,
}

/// Phase of the agentic loop
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    /// Initial assessment: determine if more context is needed
    Assessment,

    /// Context gathering: execute tool calls to collect information
    Gathering,

    /// Final query generation: produce the search queries
    Final,

    /// Evaluation: assess if results match user intent (internal phase, not from LLM)
    #[serde(skip)]
    Evaluation,

    /// Refinement: regenerate queries based on evaluation (internal phase, not from LLM)
    #[serde(skip)]
    Refinement,
}

/// Tool call for gathering context or exploring codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolCall {
    /// Gather comprehensive codebase context
    GatherContext {
        /// Context gathering parameters
        #[serde(flatten)]
        params: ContextGatheringParams,
    },

    /// Run exploratory queries to understand codebase
    ExploreCodebase {
        /// Description of what this query is exploring
        description: String,

        /// The rfx query command (without 'rfx' prefix)
        command: String,
    },

    /// Analyze codebase structure (hotspots, unused files, etc.)
    AnalyzeStructure {
        /// Type of analysis to run
        analysis_type: AnalysisType,
    },

    /// Search project documentation files
    SearchDocumentation {
        /// Search query/keywords
        query: String,

        /// Optional: specific files to search (defaults to ["CLAUDE.md", "README.md"])
        #[serde(default)]
        files: Option<Vec<String>>,
    },

    /// Get index statistics (file counts, languages, etc.)
    GetStatistics,

    /// Get dependencies of a specific file
    GetDependencies {
        /// File path (supports fuzzy matching)
        file_path: String,

        /// Show reverse dependencies (what depends on this file)
        #[serde(default)]
        reverse: bool,
    },

    /// Get dependency analysis summary
    GetAnalysisSummary {
        /// Minimum dependents for hotspot counting
        #[serde(default = "default_min_dependents")]
        min_dependents: usize,
    },

    /// Find disconnected components (islands) in the dependency graph
    FindIslands {
        /// Minimum island size to include
        #[serde(default = "default_min_island_size")]
        min_size: usize,

        /// Maximum island size to include
        #[serde(default = "default_max_island_size")]
        max_size: usize,
    },
}

/// Parameters for context gathering tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextGatheringParams {
    /// Show directory structure
    #[serde(default)]
    pub structure: bool,

    /// Show file type distribution
    #[serde(default)]
    pub file_types: bool,

    /// Detect project type
    #[serde(default)]
    pub project_type: bool,

    /// Detect frameworks
    #[serde(default)]
    pub framework: bool,

    /// Show entry points
    #[serde(default)]
    pub entry_points: bool,

    /// Show test layout
    #[serde(default)]
    pub test_layout: bool,

    /// List configuration files
    #[serde(default)]
    pub config_files: bool,

    /// Tree depth for structure (default: 2)
    #[serde(default = "default_depth")]
    pub depth: usize,

    /// Focus on specific directory path
    #[serde(default)]
    pub path: Option<String>,
}

fn default_depth() -> usize {
    2
}

fn default_min_dependents() -> usize {
    2
}

fn default_min_island_size() -> usize {
    2
}

fn default_max_island_size() -> usize {
    500
}

impl Default for ContextGatheringParams {
    fn default() -> Self {
        Self {
            structure: false,
            file_types: false,
            project_type: false,
            framework: false,
            entry_points: false,
            test_layout: false,
            config_files: false,
            depth: default_depth(),
            path: None,
        }
    }
}

/// Type of codebase analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnalysisType {
    /// Find most-imported files (dependency hotspots)
    Hotspots,

    /// Find unused files
    Unused,

    /// Find circular dependencies
    Circular,
}

/// Result evaluation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    /// Overall success assessment
    pub success: bool,

    /// Specific issues found with the results
    pub issues: Vec<EvaluationIssue>,

    /// Suggestions for refinement
    pub suggestions: Vec<String>,

    /// Evaluation score (0.0-1.0)
    pub score: f32,
}

/// Specific issue found during result evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationIssue {
    /// Type of issue
    pub issue_type: IssueType,

    /// Description of the issue
    pub description: String,

    /// Severity (0.0-1.0, higher is more severe)
    pub severity: f32,
}

/// Type of evaluation issue
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueType {
    /// No results found (query too specific or wrong pattern)
    EmptyResults,

    /// Too many results (query too broad)
    TooManyResults,

    /// Results in unexpected file types
    WrongFileTypes,

    /// Results in unexpected directories
    WrongLocations,

    /// Pattern doesn't match expected symbol type
    WrongSymbolType,

    /// Language filter seems incorrect
    WrongLanguage,
}

/// JSON schema for agentic LLM prompt
pub const AGENTIC_RESPONSE_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "phase": {
      "type": "string",
      "enum": ["assessment", "gathering", "final"],
      "description": "Current phase: 'assessment' if deciding whether to gather context, 'gathering' if executing tools, 'final' if generating queries"
    },
    "reasoning": {
      "type": "string",
      "description": "Your thought process and reasoning for this response"
    },
    "needs_context": {
      "type": "boolean",
      "description": "Whether you need more context before generating final queries (only relevant in assessment phase)"
    },
    "tool_calls": {
      "type": "array",
      "description": "Array of tools to execute for gathering context (only for assessment/gathering phases)",
      "items": {
        "type": "object",
        "oneOf": [
          {
            "properties": {
              "type": { "const": "gather_context" },
              "structure": { "type": "boolean" },
              "file_types": { "type": "boolean" },
              "project_type": { "type": "boolean" },
              "framework": { "type": "boolean" },
              "entry_points": { "type": "boolean" },
              "test_layout": { "type": "boolean" },
              "config_files": { "type": "boolean" },
              "depth": { "type": "integer" },
              "path": { "type": "string" }
            },
            "required": ["type"]
          },
          {
            "properties": {
              "type": { "const": "explore_codebase" },
              "description": { "type": "string" },
              "command": { "type": "string" }
            },
            "required": ["type", "description", "command"]
          },
          {
            "properties": {
              "type": { "const": "analyze_structure" },
              "analysis_type": { "type": "string", "enum": ["hotspots", "unused", "circular"] }
            },
            "required": ["type", "analysis_type"]
          },
          {
            "properties": {
              "type": { "const": "search_documentation" },
              "query": { "type": "string" },
              "files": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Optional: specific files to search (defaults to [\"CLAUDE.md\", \"README.md\"])"
              }
            },
            "required": ["type", "query"]
          },
          {
            "properties": {
              "type": { "const": "get_statistics" }
            },
            "required": ["type"]
          },
          {
            "properties": {
              "type": { "const": "get_dependencies" },
              "file_path": { "type": "string" },
              "reverse": { "type": "boolean" }
            },
            "required": ["type", "file_path"]
          },
          {
            "properties": {
              "type": { "const": "get_analysis_summary" },
              "min_dependents": { "type": "integer" }
            },
            "required": ["type"]
          },
          {
            "properties": {
              "type": { "const": "find_islands" },
              "min_size": { "type": "integer" },
              "max_size": { "type": "integer" }
            },
            "required": ["type"]
          }
        ]
      }
    },
    "queries": {
      "type": "array",
      "description": "Array of rfx commands to execute (only for final phase)",
      "items": {
        "type": "object",
        "properties": {
          "command": { "type": "string" },
          "order": { "type": "integer" },
          "merge": { "type": "boolean" }
        },
        "required": ["command", "order", "merge"]
      }
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Confidence score (0.0-1.0) in your generated queries"
    }
  },
  "required": ["phase", "reasoning"]
}"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_assessment_phase() {
        let json = r#"{
            "phase": "assessment",
            "reasoning": "I need to understand the project structure",
            "needs_context": true,
            "tool_calls": [{
                "type": "gather_context",
                "structure": true,
                "file_types": true
            }],
            "confidence": 0.0
        }"#;

        let response: AgenticResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.phase, Phase::Assessment);
        assert!(response.needs_context);
        assert_eq!(response.tool_calls.len(), 1);
    }

    #[test]
    fn test_deserialize_final_phase() {
        let json = r#"{
            "phase": "final",
            "reasoning": "Based on the context, I can generate queries",
            "needs_context": false,
            "queries": [{
                "command": "query \"TODO\"",
                "order": 1,
                "merge": true
            }],
            "confidence": 0.85
        }"#;

        let response: AgenticResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.phase, Phase::Final);
        assert!(!response.needs_context);
        assert_eq!(response.queries.len(), 1);
        assert_eq!(response.confidence, 0.85);
    }

    #[test]
    fn test_deserialize_explore_tool() {
        let json = r#"{
            "type": "explore_codebase",
            "description": "Find validation functions",
            "command": "query \"validate\" --symbols --kind function"
        }"#;

        let tool: ToolCall = serde_json::from_str(json).unwrap();
        match tool {
            ToolCall::ExploreCodebase {
                description,
                command,
            } => {
                assert_eq!(description, "Find validation functions");
                assert!(command.contains("validate"));
            }
            _ => panic!("Expected ExploreCodebase variant"),
        }
    }

    #[test]
    fn test_deserialize_analyze_tool() {
        let json = r#"{
            "type": "analyze_structure",
            "analysis_type": "hotspots"
        }"#;

        let tool: ToolCall = serde_json::from_str(json).unwrap();
        match tool {
            ToolCall::AnalyzeStructure { analysis_type } => {
                assert_eq!(analysis_type, AnalysisType::Hotspots);
            }
            _ => panic!("Expected AnalyzeStructure variant"),
        }
    }

    #[test]
    fn test_evaluation_report() {
        let report = EvaluationReport {
            success: false,
            issues: vec![EvaluationIssue {
                issue_type: IssueType::EmptyResults,
                description: "No results found".to_string(),
                severity: 0.9,
            }],
            suggestions: vec!["Try broader search pattern".to_string()],
            score: 0.1,
        };

        assert!(!report.success);
        assert_eq!(report.issues.len(), 1);
        assert!(report.score < 0.5);
    }
}
