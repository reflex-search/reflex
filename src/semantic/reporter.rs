//! Progress reporting for agentic loop
//!
//! This module provides transparent "show your work" output for the agentic loop,
//! displaying the LLM's reasoning at each phase similar to Claude Code's thinking blocks.

use indicatif::ProgressBar;
use owo_colors::OwoColorize;
use std::sync::{Arc, Mutex};

use super::schema_agentic::{EvaluationReport, ToolCall};
use super::tools::ToolResult;

/// Trait for reporting agentic loop progress
pub trait AgenticReporter: Send + Sync {
    /// Report assessment phase completion
    fn report_assessment(&self, reasoning: &str, needs_context: bool, tools: &[ToolCall]);

    /// Report start of tool execution
    fn report_tool_start(&self, idx: usize, tool: &ToolCall);

    /// Report tool execution completion
    fn report_tool_complete(&self, idx: usize, result: &ToolResult);

    /// Report query generation completion
    fn report_generation(&self, reasoning: Option<&str>, query_count: usize, confidence: f32);

    /// Report evaluation results
    fn report_evaluation(&self, evaluation: &EvaluationReport);

    /// Report refinement start
    fn report_refinement_start(&self);

    /// Report phase start
    fn report_phase(&self, phase_num: usize, phase_name: &str);

    /// Report reindexing progress (when cache needs to be rebuilt)
    fn report_reindex_progress(&self, current: usize, total: usize, message: String);

    /// Clear all ephemeral output (called before final results are shown)
    fn clear_all(&self);
}

/// Console reporter with colored output and ephemeral thinking
pub struct ConsoleReporter {
    /// Show LLM reasoning blocks
    show_reasoning: bool,

    /// Verbose output (show tool results, etc.)
    verbose: bool,

    /// Debug mode: disable ephemeral clearing to retain all output
    debug: bool,

    /// Number of lines printed by the last phase (for ephemeral clearing)
    lines_printed: Mutex<usize>,

    /// Optional progress spinner to update with phase information
    spinner: Option<Arc<Mutex<ProgressBar>>>,
}

impl ConsoleReporter {
    /// Create a new console reporter
    pub fn new(
        show_reasoning: bool,
        verbose: bool,
        debug: bool,
        spinner: Option<Arc<Mutex<ProgressBar>>>,
    ) -> Self {
        Self {
            show_reasoning,
            verbose,
            debug,
            lines_printed: Mutex::new(0),
            spinner,
        }
    }

    /// Clear the last N lines of output (for ephemeral display)
    fn clear_last_output(&self) {
        // Skip clearing in debug mode to retain all output
        if self.debug {
            return;
        }

        let lines = *self.lines_printed.lock().unwrap();
        if lines > 0 {
            for _ in 0..lines {
                // Move cursor up one line and clear it
                eprint!("\x1b[1A\x1b[2K");
            }
            *self.lines_printed.lock().unwrap() = 0;
        }
    }

    /// Track that N lines were printed
    fn add_lines(&self, count: usize) {
        *self.lines_printed.lock().unwrap() += count;
    }

    /// Count lines in a string
    fn count_lines(text: &str) -> usize {
        if text.is_empty() {
            0
        } else {
            text.lines().count()
        }
    }

    /// Display formatted reasoning block with line prefix (dark gray like Claude Code)
    fn display_reasoning_block(&self, reasoning: &str) {
        let mut line_count = 0;
        for line in reasoning.lines() {
            if line.trim().is_empty() {
                println!();
            } else {
                // Use ANSI bright black (dark gray) for thinking text
                println!("  \x1b[90m{}\x1b[0m", line);
            }
            line_count += 1;
        }
        self.add_lines(line_count);
    }

    /// Describe a tool for display
    fn describe_tool(&self, tool: &ToolCall) -> String {
        match tool {
            ToolCall::GatherContext { params } => {
                let mut parts = Vec::new();
                if params.structure {
                    parts.push("structure");
                }
                if params.file_types {
                    parts.push("file types");
                }
                if params.project_type {
                    parts.push("project type");
                }
                if params.framework {
                    parts.push("frameworks");
                }
                if params.entry_points {
                    parts.push("entry points");
                }
                if params.test_layout {
                    parts.push("test layout");
                }
                if params.config_files {
                    parts.push("config files");
                }

                if parts.is_empty() {
                    "gather_context: General codebase context".to_string()
                } else {
                    format!("gather_context: {}", parts.join(", "))
                }
            }
            ToolCall::ExploreCodebase {
                description,
                command,
            } => {
                format!("explore_codebase: {} ({})", description, command)
            }
            ToolCall::AnalyzeStructure { analysis_type } => {
                format!("analyze_structure: {:?}", analysis_type)
            }
            ToolCall::SearchDocumentation { query, files } => {
                if let Some(file_list) = files {
                    format!("search_documentation: '{}' in files {:?}", query, file_list)
                } else {
                    format!("search_documentation: '{}'", query)
                }
            }
            ToolCall::GetStatistics => "get_statistics: Retrieve index statistics".to_string(),
            ToolCall::GetDependencies { file_path, reverse } => {
                if *reverse {
                    format!("get_dependencies: Reverse deps for '{}'", file_path)
                } else {
                    format!("get_dependencies: Dependencies of '{}'", file_path)
                }
            }
            ToolCall::GetAnalysisSummary { min_dependents } => {
                format!(
                    "get_analysis_summary: Dependency analysis (min_dependents={})",
                    min_dependents
                )
            }
            ToolCall::FindIslands { min_size, max_size } => {
                format!(
                    "find_islands: Disconnected components (size {}-{})",
                    min_size, max_size
                )
            }
        }
    }

    /// Truncate text for preview display
    fn truncate(&self, text: &str, max_len: usize) -> String {
        if text.len() <= max_len {
            return text.to_string();
        }

        let truncated = &text[..max_len];
        format!("{}...", truncated)
    }

    /// Execute a closure with the spinner suspended
    /// This prevents visual conflicts between spinner and printed output
    fn with_suspended_spinner<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        if let Some(ref spinner) = self.spinner {
            if let Ok(spinner_guard) = spinner.lock() {
                return spinner_guard.suspend(f);
            }
        }
        // If no spinner or lock failed, just execute the closure
        f()
    }
}

impl AgenticReporter for ConsoleReporter {
    fn report_phase(&self, phase_num: usize, phase_name: &str) {
        if let Some(ref spinner) = self.spinner {
            // Lock spinner once for suspend, print, and finish
            if let Ok(spinner_guard) = spinner.lock() {
                // Suspend spinner, print output
                spinner_guard.suspend(|| {
                    let line = format!("\n━━━ Phase {}: {} ━━━", phase_num, phase_name);
                    println!("{}", line.bold().cyan());
                    self.add_lines(2); // Newline + phase line
                });
                // Finish and clear the spinner completely to hide it
                // It will automatically reappear when set_message() is called with a non-empty message
                spinner_guard.finish_and_clear();
            }
        } else {
            // No spinner, just print
            let line = format!("\n━━━ Phase {}: {} ━━━", phase_num, phase_name);
            println!("{}", line.bold().cyan());
            self.add_lines(2); // Newline + phase line
        }
    }

    fn report_assessment(&self, reasoning: &str, needs_context: bool, tools: &[ToolCall]) {
        self.report_phase(1, "Assessment");

        self.with_suspended_spinner(|| {
            if self.show_reasoning && !reasoning.is_empty() {
                println!("\n{}", "💭 Reasoning:".dimmed());
                self.add_lines(2); // Newline + header
                self.display_reasoning_block(reasoning);
            }

            println!();
            self.add_lines(1);

            if needs_context && !tools.is_empty() {
                println!(
                    "{} {}",
                    "→".bright_green(),
                    "Needs additional context".bold()
                );
                println!("  {} tool(s) to execute:", tools.len());
                self.add_lines(2);
                for (i, tool) in tools.iter().enumerate() {
                    println!(
                        "  {}. {}",
                        (i + 1).to_string().bright_white(),
                        self.describe_tool(tool).dimmed()
                    );
                    self.add_lines(1);
                }
            } else {
                println!("{} {}", "→".bright_green(), "Has sufficient context".bold());
                println!("  Proceeding directly to query generation");
                self.add_lines(2);
            }
        });
    }

    fn report_tool_start(&self, idx: usize, tool: &ToolCall) {
        if idx == 1 {
            self.report_phase(2, "Context Gathering");
            self.with_suspended_spinner(|| {
                println!();
                self.add_lines(1);
            });
        }

        if self.verbose {
            self.with_suspended_spinner(|| {
                println!(
                    "  {} Executing: {}",
                    "⋯".dimmed(),
                    self.describe_tool(tool).dimmed()
                );
                self.add_lines(1);
            });
        }
    }

    fn report_tool_complete(&self, idx: usize, result: &ToolResult) {
        self.with_suspended_spinner(|| {
            if result.success {
                println!(
                    "  {} {} {}",
                    "✓".bright_green(),
                    format!("[{}]", idx).dimmed(),
                    result.description
                );
                self.add_lines(1);

                if self.verbose && !result.output.is_empty() {
                    // Show truncated output
                    let preview = self.truncate(&result.output, 150);
                    let lines_shown = preview.lines().take(3);
                    for line in lines_shown {
                        println!("    {}", line.dimmed());
                        self.add_lines(1);
                    }
                    if result.output.lines().count() > 3 {
                        println!("    {}", "...".dimmed());
                        self.add_lines(1);
                    }
                }
            } else {
                println!(
                    "  {} {} {} - {}",
                    "✗".bright_red(),
                    format!("[{}]", idx).dimmed(),
                    result.description,
                    "failed".red()
                );
                self.add_lines(1);
            }
        });
    }

    fn report_generation(&self, reasoning: Option<&str>, query_count: usize, confidence: f32) {
        // Clear all previous output (assessment + tools are ephemeral)
        self.clear_last_output();

        self.report_phase(3, "Query Generation");

        self.with_suspended_spinner(|| {
            if self.show_reasoning {
                if let Some(reasoning_text) = reasoning {
                    if !reasoning_text.is_empty() {
                        println!("\n{}", "💭 Reasoning:".dimmed());
                        self.add_lines(2);
                        self.display_reasoning_block(reasoning_text);
                    }
                }
            }

            println!();
            self.add_lines(1);

            let confidence_pct = (confidence * 100.0) as u8;

            print!(
                "{} Generated {} {} (confidence: ",
                "→".bright_green(),
                query_count,
                if query_count == 1 { "query" } else { "queries" }
            );

            if confidence >= 0.8 {
                println!("{}%)", confidence_pct.to_string().bright_green());
            } else if confidence >= 0.6 {
                println!("{}%)", confidence_pct.to_string().yellow());
            } else {
                println!("{}%)", confidence_pct.to_string().bright_red());
            }
            self.add_lines(1);
        });
    }

    fn report_evaluation(&self, evaluation: &EvaluationReport) {
        // Clear query generation output (ephemeral)
        self.clear_last_output();

        self.report_phase(5, "Evaluation");

        self.with_suspended_spinner(|| {
            println!();
            self.add_lines(1);

            if evaluation.success {
                println!(
                    "{} {} (score: {}/1.0)",
                    "✓".bright_green(),
                    "Success".bold().bright_green(),
                    format!("{:.2}", evaluation.score).bright_white()
                );
                self.add_lines(1);

                if self.verbose && !evaluation.issues.is_empty() {
                    println!("\n  Minor issues noted:");
                    self.add_lines(2);
                    for issue in &evaluation.issues {
                        println!(
                            "  - {} (severity: {:.2})",
                            issue.description.dimmed(),
                            issue.severity
                        );
                        self.add_lines(1);
                    }
                }
            } else {
                println!(
                    "{} {} (score: {}/1.0)",
                    "⚠".yellow(),
                    "Results need refinement".bold().yellow(),
                    format!("{:.2}", evaluation.score).bright_white()
                );
                self.add_lines(1);

                if !evaluation.issues.is_empty() {
                    println!("\n  Issues found:");
                    self.add_lines(2);
                    for (idx, issue) in evaluation.issues.iter().enumerate().take(3) {
                        println!(
                            "  {}. {}",
                            (idx + 1).to_string().dimmed(),
                            issue.description
                        );
                        self.add_lines(1);
                    }
                }

                if !evaluation.suggestions.is_empty() {
                    println!("\n  Suggestions:");
                    self.add_lines(2);
                    for (idx, suggestion) in evaluation.suggestions.iter().enumerate().take(3) {
                        println!(
                            "  {}. {}",
                            (idx + 1).to_string().dimmed(),
                            suggestion.dimmed()
                        );
                        self.add_lines(1);
                    }
                }
            }
        });
    }

    fn report_refinement_start(&self) {
        // Clear evaluation output (ephemeral)
        self.clear_last_output();

        self.report_phase(6, "Refinement");

        self.with_suspended_spinner(|| {
            println!();
            println!(
                "{} Refining queries based on evaluation feedback...",
                "→".yellow()
            );
            self.add_lines(2);
        });
    }

    fn report_reindex_progress(&self, current: usize, total: usize, message: String) {
        self.with_suspended_spinner(|| {
            // Update the current line with progress
            if current > 0 {
                // Clear previous progress line
                eprint!("\r\x1b[2K");
            }

            let percentage = if total > 0 {
                (current as f32 / total as f32 * 100.0) as u8
            } else {
                0
            };

            eprint!(
                "  {} Reindexing cache: [{}/{}] {}% - {}",
                "⋯".yellow(),
                current,
                total,
                percentage,
                message.dimmed()
            );

            // Flush to ensure immediate display
            use std::io::Write;
            let _ = std::io::stderr().flush();

            // If we're done, add a newline
            if current >= total {
                eprintln!();
                self.add_lines(1);
            }
        });
    }

    fn clear_all(&self) {
        // Clear all ephemeral output before showing final results
        // (skip in debug mode to retain terminal history)
        self.clear_last_output();
    }
}

/// No-op reporter for quiet mode
pub struct QuietReporter;

impl AgenticReporter for QuietReporter {
    fn report_assessment(&self, _reasoning: &str, _needs_context: bool, _tools: &[ToolCall]) {}
    fn report_tool_start(&self, _idx: usize, _tool: &ToolCall) {}
    fn report_tool_complete(&self, _idx: usize, _result: &ToolResult) {}
    fn report_generation(&self, _reasoning: Option<&str>, _query_count: usize, _confidence: f32) {}
    fn report_evaluation(&self, _evaluation: &EvaluationReport) {}
    fn report_refinement_start(&self) {}
    fn report_phase(&self, _phase_num: usize, _phase_name: &str) {}
    fn report_reindex_progress(&self, _current: usize, _total: usize, _message: String) {}
    fn clear_all(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::schema_agentic::*;

    #[test]
    fn test_console_reporter_creation() {
        let reporter = ConsoleReporter::new(true, false, false, None);
        assert!(reporter.show_reasoning);
        assert!(!reporter.verbose);
        assert!(!reporter.debug);
    }

    #[test]
    fn test_truncate() {
        let reporter = ConsoleReporter::new(false, false, false, None);
        let text = "a".repeat(300);
        let truncated = reporter.truncate(&text, 100);
        assert!(truncated.len() <= 103); // 100 + "..."
    }

    #[test]
    fn test_describe_gather_context_tool() {
        let reporter = ConsoleReporter::new(false, false, false, None);
        let tool = ToolCall::GatherContext {
            params: ContextGatheringParams {
                structure: true,
                file_types: true,
                ..Default::default()
            },
        };

        let desc = reporter.describe_tool(&tool);
        assert!(desc.contains("gather_context"));
        assert!(desc.contains("structure"));
    }
}
