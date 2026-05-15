//! Result evaluation for agentic query refinement
//!
//! This module evaluates query results to determine if they match user intent
//! and provides feedback for query refinement if needed.

use crate::models::FileGroupedResult;
use super::schema_agentic::{EvaluationReport, EvaluationIssue, IssueType};

/// Configuration for result evaluation
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Minimum number of results to consider successful (default: 1)
    pub min_results: usize,

    /// Maximum number of results before considering too broad (default: 1000)
    pub max_results: usize,

    /// Enable file type checking
    pub check_file_types: bool,

    /// Enable location checking
    pub check_locations: bool,

    /// Strictness level (0.0-1.0, higher is stricter)
    pub strictness: f32,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            min_results: 1,
            max_results: 1000,
            check_file_types: true,
            check_locations: true,
            strictness: 0.5,
        }
    }
}

/// Evaluate query results and generate a report
///
/// This function checks for common issues like:
/// - Empty results (query too specific)
/// - Too many results (query too broad)
/// - Results in unexpected file types or directories
/// - Potential language or symbol type mismatches
///
/// # Parameters
/// - `gathered_context`: Optional context from tools (if context gathering was used)
/// - `num_queries`: Number of queries generated (0 = direct answer from context)
/// - `confidence`: LLM confidence score (0.0-1.0) in the response
pub fn evaluate_results(
    results: &[FileGroupedResult],
    total_count: usize,
    user_question: &str,
    config: &EvaluationConfig,
    gathered_context: Option<&str>,
    num_queries: usize,
    confidence: Option<f32>,
) -> EvaluationReport {
    let mut issues = Vec::new();
    let mut score = 1.0; // Start at perfect score, deduct for issues

    // Check 1: Empty results - BUT check if this is intentional
    if total_count == 0 {
        // Case 1: No queries generated (LLM answered directly from context/metadata)
        if num_queries == 0 {
            // High confidence direct answer - this is GOOD, not an error
            if confidence.unwrap_or(0.0) >= 0.90 {
                // Perfect score - answer is in gathered context or metadata
                score = 1.0;
                // Don't add any issues - this is the correct behavior
            } else {
                // Lower confidence direct answer - still probably okay
                issues.push(EvaluationIssue {
                    issue_type: IssueType::EmptyResults,
                    description: "No queries generated. Answer provided from available context.".to_string(),
                    severity: 0.2,  // Very low severity - likely intentional
                });
                score -= 0.2;
            }
        }
        // Case 2: Queries were generated but found nothing - actual problem
        else {
            // Reduce severity if context was gathered (more forgiving)
            let severity = if gathered_context.is_some() { 0.6 } else { 0.8 };
            issues.push(EvaluationIssue {
                issue_type: IssueType::EmptyResults,
                description: "No results found. Query may be too specific or pattern may be incorrect.".to_string(),
                severity,
            });
            score -= severity;
        }
    }
    // Check 2: Too many results
    else if total_count > config.max_results {
        let severity = (total_count as f32 / config.max_results as f32 - 1.0).min(0.8);
        issues.push(EvaluationIssue {
            issue_type: IssueType::TooManyResults,
            description: format!(
                "Found {} results (max threshold: {}). Query may be too broad.",
                total_count, config.max_results
            ),
            severity,
        });
        score -= severity;
    }
    // Check 3: Few results (warning, not failure)
    else if total_count < config.min_results {
        let severity = 0.3; // Lower severity for just below threshold
        issues.push(EvaluationIssue {
            issue_type: IssueType::EmptyResults,
            description: format!(
                "Only {} result(s) found. Consider broadening the search.",
                total_count
            ),
            severity,
        });
        score -= severity;
    }

    // Check 4: File type consistency (if enabled)
    if config.check_file_types && !results.is_empty() {
        let file_type_issues = check_file_type_consistency(results, user_question);
        score -= file_type_issues.iter().map(|i| i.severity).sum::<f32>();
        issues.extend(file_type_issues);
    }

    // Check 5: Location consistency (if enabled)
    if config.check_locations && !results.is_empty() {
        let location_issues = check_location_patterns(results, user_question);
        score -= location_issues.iter().map(|i| i.severity).sum::<f32>();
        issues.extend(location_issues);
    }

    // Clamp score to [0.0, 1.0]
    score = score.max(0.0).min(1.0);

    // Determine success based on score and strictness
    let success_threshold = 0.4 + (config.strictness * 0.2);
    let success = score >= success_threshold;

    // Generate refinement suggestions
    let suggestions = generate_suggestions(&issues, results, user_question);

    EvaluationReport {
        success,
        issues,
        suggestions,
        score,
    }
}

/// Check if file types in results match the expected types from the question
fn check_file_type_consistency(
    results: &[FileGroupedResult],
    user_question: &str,
) -> Vec<EvaluationIssue> {
    let mut issues = Vec::new();
    let question_lower = user_question.to_lowercase();

    // Extract file extensions from results
    let mut extensions: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for result in results {
        if let Some(ext) = std::path::Path::new(&result.path)
            .extension()
            .and_then(|e| e.to_str())
        {
            *extensions.entry(ext.to_lowercase()).or_insert(0) += 1;
        }
    }

    // Check for language-specific keywords in question
    let language_hints: Vec<(&str, Vec<&str>)> = vec![
        ("rust", vec!["rs"]),
        ("python", vec!["py"]),
        ("typescript", vec!["ts", "tsx"]),
        ("javascript", vec!["js", "jsx"]),
        ("java", vec!["java"]),
        ("go", vec!["go"]),
        ("c++", vec!["cpp", "cc", "cxx", "hpp", "h"]),
        ("c#", vec!["cs"]),
        ("ruby", vec!["rb"]),
        ("php", vec!["php"]),
    ];

    for (lang, expected_exts) in language_hints {
        if question_lower.contains(lang) {
            // Check if any results match expected extensions
            let has_matching = expected_exts.iter().any(|ext| extensions.contains_key(*ext));

            if !has_matching && !results.is_empty() {
                issues.push(EvaluationIssue {
                    issue_type: IssueType::WrongFileTypes,
                    description: format!(
                        "Question mentions '{}' but results don't contain {} files. Found: {}",
                        lang,
                        expected_exts.join("/"),
                        extensions.keys()
                            .take(5)
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    severity: 0.3,
                });
            }
        }
    }

    issues
}

/// Check if result locations match expected patterns from the question
fn check_location_patterns(
    results: &[FileGroupedResult],
    user_question: &str,
) -> Vec<EvaluationIssue> {
    let mut issues = Vec::new();
    let question_lower = user_question.to_lowercase();

    // Common directory hints in questions
    let dir_hints = vec![
        ("test", vec!["test", "tests", "spec", "__tests__"]),
        ("source", vec!["src", "lib", "app"]),
        ("config", vec!["config", "conf", "settings"]),
        ("util", vec!["util", "utils", "helper", "helpers"]),
        ("api", vec!["api", "endpoint", "route"]),
        ("model", vec!["model", "models", "entity", "entities"]),
    ];

    for (hint, expected_dirs) in dir_hints {
        if question_lower.contains(hint) {
            // Check if results contain expected directories
            let has_matching = results.iter().any(|r| {
                let path_lower = r.path.to_lowercase();
                expected_dirs.iter().any(|dir| path_lower.contains(dir))
            });

            if !has_matching && results.len() > 3 {
                // Only flag if we have several results but none in expected location
                issues.push(EvaluationIssue {
                    issue_type: IssueType::WrongLocations,
                    description: format!(
                        "Question mentions '{}' but results are not in typical directories ({})",
                        hint,
                        expected_dirs.join(", ")
                    ),
                    severity: 0.15, // Very low severity - just a hint
                });
            }
        }
    }

    issues
}

/// Generate refinement suggestions based on issues
fn generate_suggestions(
    issues: &[EvaluationIssue],
    _results: &[FileGroupedResult],
    _user_question: &str,
) -> Vec<String> {
    let mut suggestions = Vec::new();

    for issue in issues {
        match issue.issue_type {
            IssueType::EmptyResults => {
                suggestions.push("Try a broader search pattern (remove --exact, use --contains)".to_string());
                suggestions.push("Remove language or file filters to expand search scope".to_string());
                suggestions.push("Check if the pattern spelling is correct".to_string());
            }
            IssueType::TooManyResults => {
                suggestions.push("Add --symbols flag to find only definitions".to_string());
                suggestions.push("Add --kind filter to narrow by symbol type".to_string());
                suggestions.push("Add --lang or --glob filter to narrow file scope".to_string());
                suggestions.push("Use more specific search pattern".to_string());
            }
            IssueType::WrongFileTypes => {
                suggestions.push("Add --lang filter to search only relevant language files".to_string());
                suggestions.push("Verify the language mentioned in question matches codebase".to_string());
            }
            IssueType::WrongLocations => {
                suggestions.push("Add --file or --glob filter to focus on specific directories".to_string());
            }
            IssueType::WrongSymbolType => {
                suggestions.push("Adjust --kind filter to match expected symbol type".to_string());
                suggestions.push("Remove --symbols flag to find usages instead of definitions".to_string());
            }
            IssueType::WrongLanguage => {
                suggestions.push("Review --lang filter and ensure it matches the codebase".to_string());
            }
        }
    }

    // Deduplicate suggestions
    suggestions.sort();
    suggestions.dedup();

    // Limit to top 5 most relevant suggestions
    suggestions.truncate(5);

    suggestions
}

/// Format evaluation report for LLM consumption
pub fn format_evaluation_for_llm(report: &EvaluationReport) -> String {
    let mut output = Vec::new();

    output.push("## Query Result Evaluation\n".to_string());
    output.push(format!("**Success:** {}", report.success));
    output.push(format!("**Score:** {:.2}/1.0\n", report.score));

    if !report.issues.is_empty() {
        output.push("### Issues Found:\n".to_string());
        for (idx, issue) in report.issues.iter().enumerate() {
            output.push(format!(
                "{}. **{:?}** (severity: {:.2})",
                idx + 1,
                issue.issue_type,
                issue.severity
            ));
            output.push(format!("   {}\n", issue.description));
        }
    }

    if !report.suggestions.is_empty() {
        output.push("\n### Refinement Suggestions:\n".to_string());
        for (idx, suggestion) in report.suggestions.iter().enumerate() {
            output.push(format!("{}. {}", idx + 1, suggestion));
        }
    }

    output.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{MatchResult, Span};

    fn create_test_result(path: &str, line: usize) -> FileGroupedResult {
        FileGroupedResult {
            path: path.to_string(),
            language: crate::models::Language::Unknown,
            dependencies: None,
            matches: vec![MatchResult {
                kind: crate::models::SymbolKind::Unknown("test".to_string()),
                symbol: None,
                span: Span {
                    start_line: line,
                    end_line: line,
                },
                preview: "test preview".to_string(),
                context_before: vec![],
                context_after: vec![],
            }],
        }
    }

    #[test]
    fn test_evaluate_empty_results() {
        let config = EvaluationConfig::default();
        let report = evaluate_results(&[], 0, "find todos", &config, None, 1, Some(0.9));

        assert!(!report.success);
        assert!(!report.issues.is_empty());
        assert_eq!(report.issues[0].issue_type, IssueType::EmptyResults);
        assert!(!report.suggestions.is_empty());
    }

    #[test]
    fn test_evaluate_too_many_results() {
        let config = EvaluationConfig::default();
        let results = vec![create_test_result("test.rs", 1)];
        let report = evaluate_results(&results, 2000, "find all", &config, None, 1, Some(0.9));

        assert!(!report.success);
        assert!(report.issues.iter().any(|i| i.issue_type == IssueType::TooManyResults));
    }

    #[test]
    fn test_evaluate_success() {
        let config = EvaluationConfig::default();
        let results = vec![
            create_test_result("src/main.rs", 10),
            create_test_result("src/lib.rs", 20),
        ];
        let report = evaluate_results(&results, 10, "find functions", &config, None, 1, Some(0.85));

        assert!(report.success);
        assert!(report.score > 0.7);
    }

    #[test]
    fn test_check_file_type_consistency() {
        let results = vec![create_test_result("test.py", 1)];
        let issues = check_file_type_consistency(&results, "Find Rust functions");

        assert!(!issues.is_empty());
        assert_eq!(issues[0].issue_type, IssueType::WrongFileTypes);
    }

    #[test]
    fn test_check_location_patterns() {
        let results = vec![
            create_test_result("src/main.rs", 1),
            create_test_result("src/lib.rs", 2),
            create_test_result("src/utils.rs", 3),
            create_test_result("src/helper.rs", 4),
        ];
        let issues = check_location_patterns(&results, "Find test functions");

        // Should suggest results should be in test directories
        assert!(!issues.is_empty());
        assert_eq!(issues[0].issue_type, IssueType::WrongLocations);
    }

    #[test]
    fn test_generate_suggestions() {
        let issues = vec![
            EvaluationIssue {
                issue_type: IssueType::EmptyResults,
                description: "No results".to_string(),
                severity: 0.9,
            },
        ];

        let suggestions = generate_suggestions(&issues, &[], "test");
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("broader")));
    }

    #[test]
    fn test_evaluate_direct_answer_high_confidence() {
        let config = EvaluationConfig::default();
        let report = evaluate_results(&[], 0, "How many files?", &config, None, 0, Some(0.95));

        assert!(report.success);  // Should pass!
        assert_eq!(report.score, 1.0);  // Perfect score
        assert!(report.issues.is_empty());  // No issues
    }

    #[test]
    fn test_evaluate_direct_answer_low_confidence() {
        let config = EvaluationConfig::default();
        let report = evaluate_results(&[], 0, "How many files?", &config, None, 0, Some(0.75));

        assert!(report.success);  // Should still pass
        assert!(report.score >= 0.7);  // Decent score
        assert_eq!(report.issues.len(), 1);  // One low-severity issue
        assert!(report.issues[0].severity < 0.3);  // Low severity
    }
}
