//! Interactive TUI chat mode for `rfx ask`
//!
//! Provides a Claude Code-like interface with:
//! - Fixed stats panel (top)
//! - Scrollable message history (middle)
//! - Fixed input box (bottom)

use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame, Terminal,
};
use std::io;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;
use textwrap;

use crate::cache::CacheManager;

use super::chat_session::{ChatSession, MessageRole};
use super::AgenticConfig;

/// Progress updates from async execution
#[derive(Debug, Clone)]
enum PhaseUpdate {
    /// Phase 0: Triage - deciding whether to search or answer directly
    Triaging,

    /// Fast path: Answering from conversation context
    AnsweringFromContext,

    /// Phase 1: Thinking/Assessment (agentic path)
    Thinking {
        reasoning: String,
        needs_context: bool,
    },
    /// Phase 2: Tool gathering (agentic path)
    Tools {
        content: String,
        tool_calls: Vec<String>,
    },
    /// Phase 3: Query generation (agentic path)
    Queries {
        queries: Vec<String>,
    },
    /// Phase 4: Execution status (agentic path)
    Executing {
        results_count: usize,
        execution_time_ms: u64,
    },
    /// Reindexing cache (schema mismatch detected)
    Reindexing {
        current: usize,
        total: usize,
        message: String,
    },
    /// Phase 5: Final answer (both paths)
    Answer {
        answer: String,
    },
    /// Error occurred
    Error {
        error: String,
    },
    /// Non-fatal notice surfaced to the status bar (e.g. degraded LLM mode
    /// when the LLM call failed but the path can still produce a useful
    /// result via fallback).
    Notice {
        message: String,
    },
    /// Processing complete
    Done,
}

/// Triage decision for question handling
#[derive(Debug, Clone)]
enum TriageDecision {
    /// Can answer directly from conversation context
    DirectAnswer,
    /// Needs to search codebase
    NeedsSearch { reasoning: String },
}

/// Helper function to wrap text with consistent "│ " prefix on each line
fn wrap_with_prefix(content: &str, area_width: u16, border_color: Color) -> Vec<Line<'_>> {
    let mut lines = Vec::new();

    // Calculate usable width: total width - borders (2) - prefix "│ " (2)
    let usable_width = (area_width.saturating_sub(4)) as usize;

    // Ensure we have at least some width to work with
    if usable_width < 10 {
        // Fallback for very narrow terminals - just add prefix without wrapping
        for content_line in content.lines() {
            lines.push(Line::from(vec![
                Span::styled("│ ", Style::default().fg(border_color)),
                Span::styled(content_line, Style::default().fg(Color::White)),
            ]));
        }
        return lines;
    }

    // Wrap each line of the content
    for content_line in content.lines() {
        if content_line.is_empty() {
            // Preserve empty lines
            lines.push(Line::from(Span::styled(
                "│ ",
                Style::default().fg(border_color),
            )));
        } else {
            // Wrap the line to fit the usable width
            let wrapped = textwrap::wrap(content_line, usable_width);
            for wrapped_line in wrapped {
                lines.push(Line::from(vec![
                    Span::styled("│ ", Style::default().fg(border_color)),
                    Span::styled(wrapped_line.to_string(), Style::default().fg(Color::White)),
                ]));
            }
        }
    }

    lines
}

/// Helper function to render markdown with consistent "│ " prefix on each line
fn render_markdown_with_prefix(content: &str, area_width: u16, border_color: Color) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    // Calculate usable width: total width - borders (2) - prefix "│ " (2)
    let usable_width = (area_width.saturating_sub(4)) as usize;

    // Ensure we have at least some width to work with
    if usable_width < 10 {
        // Fallback for very narrow terminals
        for content_line in content.lines() {
            lines.push(Line::from(vec![
                Span::styled("│ ", Style::default().fg(border_color)),
                Span::styled(content_line.to_string(), Style::default().fg(Color::White)),
            ]));
        }
        return lines;
    }

    let mut in_code_block = false;

    for content_line in content.lines() {
        // Check for code block markers
        if content_line.trim().starts_with("```") {
            in_code_block = !in_code_block;
            continue; // Don't render the ``` markers
        }

        if in_code_block {
            // Inside code block - preserve formatting, no markdown processing
            // Separate border color from content color
            lines.push(Line::from(vec![
                Span::styled("│ ", Style::default().fg(border_color)),
                Span::styled(content_line.to_string(), Style::default().fg(Color::Cyan).bg(Color::Black)),
            ]));
            continue;
        }

        if content_line.is_empty() {
            lines.push(Line::from(Span::styled("│ ", Style::default().fg(border_color))));
            continue;
        }

        // Check for headers
        let (header_level, text_after_header) = if content_line.starts_with("### ") {
            (3, &content_line[4..])
        } else if content_line.starts_with("## ") {
            (2, &content_line[3..])
        } else if content_line.starts_with("# ") {
            (1, &content_line[2..])
        } else {
            (0, content_line)
        };

        let wrapped = textwrap::wrap(text_after_header, usable_width);

        for wrapped_line in wrapped {
            let parsed_spans = parse_inline_markdown(&wrapped_line);

            // Build the line with prefix (using border color)
            let mut line_spans = vec![Span::styled("│ ", Style::default().fg(border_color))];

            // Apply header styling if needed
            if header_level > 0 {
                for span in parsed_spans {
                    let mut style = span.style;
                    style = style.add_modifier(Modifier::BOLD);
                    if header_level == 1 {
                        style = style.fg(Color::Yellow);
                    } else if header_level == 2 {
                        style = style.fg(Color::Cyan);
                    }
                    line_spans.push(Span::styled(span.content.to_string(), style));
                }
            } else {
                line_spans.extend(parsed_spans.into_iter().map(|s| Span::styled(s.content.to_string(), s.style)));
            }

            lines.push(Line::from(line_spans));
        }
    }

    lines
}

/// Parse inline markdown elements (bold, italic, code)
fn parse_inline_markdown(text: &str) -> Vec<Span<'static>> {
    let mut result = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Check for **bold**
        if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == '*' {
            if let Some(end) = find_closing_double_star(&chars, i + 2) {
                let content: String = chars[i + 2..end].iter().collect();
                result.push(Span::styled(
                    content,
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ));
                i = end + 2;
                continue;
            }
        }

        // Check for *italic* or _italic_
        if chars[i] == '*' || chars[i] == '_' {
            let marker = chars[i];
            if let Some(end) = find_closing_char(&chars, i + 1, marker) {
                let content: String = chars[i + 1..end].iter().collect();
                result.push(Span::styled(
                    content,
                    Style::default().fg(Color::White).add_modifier(Modifier::ITALIC),
                ));
                i = end + 1;
                continue;
            }
        }

        // Check for `code`
        if chars[i] == '`' {
            if let Some(end) = find_closing_char(&chars, i + 1, '`') {
                let content: String = chars[i + 1..end].iter().collect();
                result.push(Span::styled(
                    content,
                    Style::default().fg(Color::Cyan).bg(Color::Black),
                ));
                i = end + 1;
                continue;
            }
        }

        // Regular text - collect until next markdown character
        let mut plain_text = String::new();
        while i < chars.len() && chars[i] != '*' && chars[i] != '_' && chars[i] != '`' {
            plain_text.push(chars[i]);
            i += 1;
        }

        if !plain_text.is_empty() {
            result.push(Span::styled(plain_text, Style::default().fg(Color::White)));
        }

        // SAFETY: If we're still at a markdown character with no match, treat it as plain text
        // This prevents infinite loops on unmatched *, _, or `
        if i < chars.len() && (chars[i] == '*' || chars[i] == '_' || chars[i] == '`') {
            result.push(Span::styled(chars[i].to_string(), Style::default().fg(Color::White)));
            i += 1;  // CRITICAL: Always advance to prevent infinite loop
        }
    }

    if result.is_empty() {
        result.push(Span::raw(""));
    }

    result
}

/// Find closing ** for bold
fn find_closing_double_star(chars: &[char], start: usize) -> Option<usize> {
    for i in start..chars.len().saturating_sub(1) {
        if chars[i] == '*' && chars[i + 1] == '*' {
            return Some(i);
        }
    }
    None
}

/// Find closing character for italic or code
fn find_closing_char(chars: &[char], start: usize, marker: char) -> Option<usize> {
    for i in start..chars.len() {
        if chars[i] == marker {
            // For *, make sure it's not **
            if marker == '*' && i + 1 < chars.len() && chars[i + 1] == '*' {
                continue; // Skip **, we're looking for single *
            }
            return Some(i);
        }
    }
    None
}

/// Main chat application state
pub struct ChatApp {
    /// Chat session (message history and token tracking)
    session: ChatSession,

    /// Current input buffer
    input: String,

    /// Cursor position in input
    cursor: usize,

    /// Scroll offset for message history (0 = bottom, higher = scroll up)
    scroll_offset: usize,

    /// Whether to quit the application
    should_quit: bool,

    /// Cache manager for executing queries
    cache: CacheManager,

    /// Provider configuration
    provider_name: String,

    /// Optional model override
    model_override: Option<String>,

    /// Status message (ephemeral, e.g., "Compacted 10 messages")
    status_message: Option<String>,

    /// Whether we're currently waiting for LLM response
    waiting: bool,

    /// Progress updates from async execution
    progress_rx: Option<Receiver<PhaseUpdate>>,

    /// Spinner animation frame counter for loading indicator
    spinner_frame: usize,
}

impl ChatApp {
    /// Create a new chat application
    pub fn new(
        cache: CacheManager,
        provider_name: String,
        model_override: Option<String>,
    ) -> Result<Self> {
        // Get actual model name (priority: override > user config > provider default).
        // For self-hosted providers with no built-in default, fall back to
        // a "(not configured)" placeholder for the session label — the
        // failure will surface at send time as a Notice.
        let model = super::config::resolve_model_for(&provider_name, None, model_override.as_deref())
            .or_else(|| {
                let d = super::providers::default_model_for(&provider_name);
                if d.is_empty() {
                    None
                } else {
                    Some(d.to_string())
                }
            })
            .unwrap_or_else(|| "(not configured)".to_string());

        let session = ChatSession::new(provider_name.clone(), model);

        Ok(Self {
            session,
            input: String::new(),
            cursor: 0,
            scroll_offset: 0,
            should_quit: false,
            cache,
            provider_name,
            model_override,
            status_message: None,
            waiting: false,
            progress_rx: None,
            spinner_frame: 0,
        })
    }

    /// Run the chat event loop
    pub fn run(&mut self) -> Result<()> {
        // Setup terminal
        let mut terminal = setup_terminal()?;

        // Show welcome message
        self.session.add_system_message(
            "Welcome to rfx ask interactive mode!\n\
             \n\
             Type your questions naturally and press Enter to send.\n\
             \n\
             Slash commands:\n\
             • /clear - Clear conversation history\n\
             • /compact - Summarize old messages to save tokens\n\
             • /model [provider] [model] - Show or change provider/model\n\
             • /help - Show this help message\n\
             \n\
             Press Ctrl+C to exit.".to_string()
        );

        // Main event loop
        let result = self.event_loop(&mut terminal);

        // Restore terminal
        restore_terminal(terminal)?;

        result
    }

    fn event_loop(&mut self, terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
        loop {
            // Check for progress updates from async execution
            // Use a separate scope to release the borrow before calling handle_progress_update
            let updates: Vec<PhaseUpdate> = if let Some(ref rx) = self.progress_rx {
                let mut updates = Vec::new();
                // Try to receive all pending updates (non-blocking)
                while let Ok(update) = rx.try_recv() {
                    updates.push(update);
                }
                updates
            } else {
                Vec::new()
            };

            // Process updates
            for update in updates {
                self.handle_progress_update(update);
            }

            // Update spinner animation frame
            if self.waiting {
                self.spinner_frame = (self.spinner_frame + 1) % 10;
            }

            // Render UI
            terminal.draw(|f| self.render(f))?;

            // Handle events (with timeout for smooth rendering)
            if event::poll(Duration::from_millis(100))? {
                match event::read()? {
                    Event::Key(key) => self.handle_key(key)?,
                    Event::Mouse(mouse) => self.handle_mouse(mouse),
                    _ => {}
                }
            }

            if self.should_quit {
                break;
            }
        }

        Ok(())
    }

    fn render(&mut self, f: &mut Frame) {
        let size = f.area();

        // Create layout: [Stats (2 lines), Messages (fill), Input (4 lines)]
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),  // Stats panel
                Constraint::Min(10),    // Message history (scrollable)
                Constraint::Length(4),  // Input box
            ])
            .split(size);

        // Render each section
        self.render_stats(f, chunks[0]);
        self.render_messages(f, chunks[1]);
        self.render_input(f, chunks[2]);
    }

    fn render_stats(&self, f: &mut Frame, area: Rect) {
        let usage = self.session.context_usage();
        let percentage = (usage * 100.0) as u32;

        // Color based on usage
        let usage_color = if usage > 0.9 {
            Color::Red
        } else if usage > 0.8 {
            Color::Yellow
        } else {
            Color::Green
        };

        let line1 = Line::from(vec![
            Span::raw("Model: "),
            Span::styled(
                format!("{} ", self.session.model()),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
            Span::raw("│ Provider: "),
            Span::styled(
                format!("{} ", self.session.provider()),
                Style::default().fg(Color::Blue),
            ),
            Span::raw("│ Tokens: "),
            Span::styled(
                format!("{}/{} ", self.session.total_tokens(), self.session.context_limit()),
                Style::default().fg(usage_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("({}%)", percentage),
                Style::default().fg(usage_color),
            ),
        ]);

        // Status message or hint
        let line2_text = if let Some(ref status) = self.status_message {
            status.clone()
        } else if self.waiting {
            "⏳ Waiting for response...".to_string()
        } else if self.session.should_compact() {
            "⚠ Context >90% full! Use /compact to summarize older messages.".to_string()
        } else if self.session.is_near_limit() {
            "⚠ Context >80% full. Consider using /compact soon.".to_string()
        } else {
            "Ready • Type your question or /help for commands".to_string()
        };

        let line2_color = if self.session.should_compact() {
            Color::Red
        } else if self.session.is_near_limit() {
            Color::Yellow
        } else if self.waiting {
            Color::Cyan
        } else {
            Color::Gray
        };

        let line2 = Line::from(Span::styled(line2_text, Style::default().fg(line2_color)));

        let paragraph = Paragraph::new(vec![line1, line2])
            .style(Style::default().bg(Color::Black));

        f.render_widget(paragraph, area);
    }

    fn render_messages(&mut self, f: &mut Frame, area: Rect) {
        let mut lines: Vec<Line> = Vec::new();

        // Render all messages
        for msg in self.session.messages() {
            match msg.role {
                MessageRole::User => {
                    // User message header
                    lines.push(Line::from(""));
                    lines.push(Line::from(Span::styled(
                        "╭─ You ─────────────────────────────────────",
                        Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
                    )));

                    // Message content (with proper wrapping and consistent green border)
                    lines.extend(wrap_with_prefix(&msg.content, area.width, Color::Green));

                    lines.push(Line::from(Span::styled(
                        "╰───────────────────────────────────────────",
                        Style::default().fg(Color::Green),
                    )));
                }
                MessageRole::AssistantThinking => {
                    // Phase 1: Thinking/Assessment
                    lines.push(Line::from(""));
                    lines.push(Line::from(Span::styled(
                        "╭─ Assistant (Thinking) ────────────────────",
                        Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
                    )));

                    // Show needs_context indicator
                    if let Some(ref meta) = msg.metadata {
                        if meta.needs_context {
                            lines.push(Line::from(vec![
                                Span::styled("│ ", Style::default().fg(Color::Magenta)),
                                Span::styled("🔍 Needs context gathering", Style::default().fg(Color::Yellow)),
                            ]));
                        }
                    }

                    // Message content (with proper wrapping and consistent magenta border)
                    lines.extend(wrap_with_prefix(&msg.content, area.width, Color::Magenta));

                    lines.push(Line::from(Span::styled(
                        "╰───────────────────────────────────────────",
                        Style::default().fg(Color::Magenta),
                    )));
                }
                MessageRole::AssistantTools => {
                    // Phase 2: Tool gathering
                    lines.push(Line::from(""));
                    lines.push(Line::from(Span::styled(
                        "╭─ Assistant (Tools) ───────────────────────",
                        Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
                    )));

                    // Show tool calls
                    if let Some(ref meta) = msg.metadata {
                        if !meta.tool_calls.is_empty() {
                            lines.push(Line::from(vec![
                                Span::styled("│ ", Style::default().fg(Color::Blue)),
                                Span::styled(
                                    format!("🔧 {} tool calls made", meta.tool_calls.len()),
                                    Style::default().fg(Color::DarkGray)
                                ),
                            ]));
                        }
                    }

                    // Message content (with proper wrapping and consistent blue border)
                    lines.extend(wrap_with_prefix(&msg.content, area.width, Color::Blue));

                    lines.push(Line::from(Span::styled(
                        "╰───────────────────────────────────────────",
                        Style::default().fg(Color::Blue),
                    )));
                }
                MessageRole::AssistantQueries => {
                    // Phase 3: Generated queries
                    lines.push(Line::from(""));
                    lines.push(Line::from(Span::styled(
                        "╭─ Assistant (Queries) ─────────────────────",
                        Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
                    )));

                    // Show query count
                    if let Some(ref meta) = msg.metadata {
                        if !meta.queries.is_empty() {
                            lines.push(Line::from(vec![
                                Span::styled("│ ", Style::default().fg(Color::Magenta)),
                                Span::styled(
                                    format!("📝 Generated {} queries", meta.queries.len()),
                                    Style::default().fg(Color::DarkGray)
                                ),
                            ]));
                            // Optionally show the queries
                            for (i, query) in meta.queries.iter().enumerate() {
                                lines.push(Line::from(vec![
                                    Span::styled("│ ", Style::default().fg(Color::Magenta)),
                                    Span::styled(
                                        format!("  {}. {}", i + 1, query),
                                        Style::default().fg(Color::DarkGray)
                                    ),
                                ]));
                            }
                        }
                    }

                    // Message content (with proper wrapping and consistent magenta border)
                    lines.extend(wrap_with_prefix(&msg.content, area.width, Color::Magenta));

                    lines.push(Line::from(Span::styled(
                        "╰───────────────────────────────────────────",
                        Style::default().fg(Color::Magenta),
                    )));
                }
                MessageRole::AssistantExecuting => {
                    // Phase 4: Execution status
                    lines.push(Line::from(""));
                    lines.push(Line::from(Span::styled(
                        "╭─ Assistant (Executing) ───────────────────",
                        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                    )));

                    // Show execution stats
                    if let Some(ref meta) = msg.metadata {
                        let time_str = if let Some(ms) = meta.execution_time_ms {
                            format!(" in {}ms", ms)
                        } else {
                            String::new()
                        };
                        lines.push(Line::from(vec![
                            Span::styled("│ ", Style::default().fg(Color::Yellow)),
                            Span::styled(
                                format!("⚡ Found {} result{}{}", meta.results_count, if meta.results_count == 1 { "" } else { "s" }, time_str),
                                Style::default().fg(Color::DarkGray)
                            ),
                        ]));
                    }

                    // Message content (with proper wrapping and consistent yellow border)
                    lines.extend(wrap_with_prefix(&msg.content, area.width, Color::Yellow));

                    lines.push(Line::from(Span::styled(
                        "╰───────────────────────────────────────────",
                        Style::default().fg(Color::Yellow),
                    )));
                }
                MessageRole::AssistantAnswer => {
                    // Phase 5: Final answer
                    lines.push(Line::from(""));
                    lines.push(Line::from(Span::styled(
                        "╭─ Assistant (Answer) ──────────────────────",
                        Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
                    )));

                    // Message content (with markdown rendering and consistent blue border)
                    lines.extend(render_markdown_with_prefix(&msg.content, area.width, Color::Blue));

                    lines.push(Line::from(Span::styled(
                        "╰───────────────────────────────────────────",
                        Style::default().fg(Color::Blue),
                    )));
                }
                MessageRole::System => {
                    // System message (e.g., welcome, compaction summary)
                    lines.push(Line::from(""));
                    lines.push(Line::from(Span::styled(
                        "╭─ System ──────────────────────────────────",
                        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                    )));

                    // Message content (with proper wrapping)
                    lines.extend(wrap_with_prefix(&msg.content, area.width, Color::Yellow));

                    lines.push(Line::from(Span::styled(
                        "╰───────────────────────────────────────────",
                        Style::default().fg(Color::Yellow),
                    )));
                }
            }
        }

        // Show loading indicator if waiting for response
        if self.waiting {
            // Spinner animation characters (braille patterns)
            const SPINNER_CHARS: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let spinner_char = SPINNER_CHARS[self.spinner_frame % SPINNER_CHARS.len()];

            // Get current status message or default
            let status_text = self.status_message.as_ref()
                .map(|s| s.clone())
                .unwrap_or_else(|| "Working...".to_string());

            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "╭─ Processing ──────────────────────────────",
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            )));
            lines.push(Line::from(vec![
                Span::styled("│ ", Style::default().fg(Color::Cyan)),
                Span::styled(
                    spinner_char,
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" {}", status_text),
                    Style::default().fg(Color::White),
                ),
            ]));
            lines.push(Line::from(Span::styled(
                "╰───────────────────────────────────────────",
                Style::default().fg(Color::Cyan),
            )));
        }

        // Always add bottom padding to account for text wrapping
        // (long lines that span multiple terminal rows)
        for _ in 0..8 {
            lines.push(Line::from(""));
        }

        // Calculate scroll position
        // scroll_offset = 0 means show bottom (latest messages)
        // scroll_offset > 0 means scroll up
        let total_lines = lines.len();
        let visible_height = area.height.saturating_sub(2) as usize; // Account for borders

        let scroll = if total_lines <= visible_height {
            0 // No scrolling needed
        } else {
            // Calculate scroll from bottom
            let max_scroll = total_lines.saturating_sub(visible_height);

            // Clamp scroll_offset to valid range to prevent scrolling past top
            self.scroll_offset = self.scroll_offset.min(max_scroll);

            max_scroll.saturating_sub(self.scroll_offset) as u16
        };

        let paragraph = Paragraph::new(lines)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(" Messages ")
                .border_style(Style::default().fg(Color::DarkGray)))
            .scroll((scroll as u16, 0));

        f.render_widget(paragraph, area);
    }

    fn render_input(&self, f: &mut Frame, area: Rect) {
        let input_display = if self.input.is_empty() {
            "Type your question here...".to_string()
        } else {
            self.input.clone()
        };

        let input_style = if self.input.is_empty() {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default().fg(Color::White)
        };

        // Show shortcuts in the border
        let shortcuts = " Enter: Send | Ctrl+C: Quit | Ctrl+L: /clear | Ctrl+K: /compact | Ctrl+U: Clear input ";

        let paragraph = Paragraph::new(input_display)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(vec![
                    Span::raw(" "),
                    Span::styled(">", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                    Span::raw(" Input "),
                ])
                .title_bottom(Line::from(Span::styled(
                    shortcuts,
                    Style::default().fg(Color::DarkGray)
                )))
                .border_style(Style::default().fg(if self.waiting { Color::DarkGray } else { Color::Green })))
            .style(input_style);

        f.render_widget(paragraph, area);

        // Set cursor position if not waiting
        if !self.waiting && !self.input.is_empty() {
            f.set_cursor_position((
                area.x + 1 + (self.cursor as u16),
                area.y + 1,
            ));
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> Result<()> {
        // Global shortcuts
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c') | KeyCode::Char('d') => {
                    // Quit
                    self.should_quit = true;
                    return Ok(());
                }
                KeyCode::Char('l') => {
                    // Clear conversation
                    self.handle_slash_command("/clear")?;
                    return Ok(());
                }
                KeyCode::Char('k') => {
                    // Compact conversation
                    self.handle_slash_command("/compact")?;
                    return Ok(());
                }
                KeyCode::Char('u') => {
                    // Clear input
                    self.input.clear();
                    self.cursor = 0;
                    return Ok(());
                }
                _ => {}
            }
        }

        // Don't accept input while waiting for response
        if self.waiting {
            return Ok(());
        }

        // Handle input
        match key.code {
            KeyCode::Enter => {
                self.handle_enter()?;
            }
            KeyCode::Char(c) => {
                self.input.insert(self.cursor, c);
                self.cursor += 1;
            }
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    self.input.remove(self.cursor - 1);
                    self.cursor -= 1;
                }
            }
            KeyCode::Delete => {
                if self.cursor < self.input.len() {
                    self.input.remove(self.cursor);
                }
            }
            KeyCode::Left => {
                self.cursor = self.cursor.saturating_sub(1);
            }
            KeyCode::Right => {
                if self.cursor < self.input.len() {
                    self.cursor += 1;
                }
            }
            KeyCode::Home => {
                self.cursor = 0;
            }
            KeyCode::End => {
                self.cursor = self.input.len();
            }
            KeyCode::Up => {
                // Scroll messages up
                self.scroll_offset = self.scroll_offset.saturating_add(1);
            }
            KeyCode::Down => {
                // Scroll messages down
                self.scroll_offset = self.scroll_offset.saturating_sub(1);
            }
            KeyCode::PageUp => {
                // Fast scroll up
                self.scroll_offset = self.scroll_offset.saturating_add(10);
            }
            KeyCode::PageDown => {
                // Fast scroll down
                self.scroll_offset = self.scroll_offset.saturating_sub(10);
            }
            _ => {}
        }

        Ok(())
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        // Handle mouse scroll events
        match mouse.kind {
            MouseEventKind::ScrollUp => {
                // Scroll up (show older messages) - increase scroll_offset by 3
                self.scroll_offset = self.scroll_offset.saturating_add(3);
            }
            MouseEventKind::ScrollDown => {
                // Scroll down (show newer messages) - decrease scroll_offset by 3
                self.scroll_offset = self.scroll_offset.saturating_sub(3);
            }
            _ => {}
        }
    }

    fn handle_enter(&mut self) -> Result<()> {
        let input = self.input.trim().to_string();

        if input.is_empty() {
            return Ok(());
        }

        // Check for slash commands
        if input.starts_with('/') {
            return self.handle_slash_command(&input);
        }

        // Add user message to session
        self.session.add_user_message(input.clone());

        // Clear input
        self.input.clear();
        self.cursor = 0;

        // Auto-scroll to bottom to see the new message
        self.scroll_offset = 0;

        // Execute query asynchronously
        // For now, we'll do it synchronously (blocking)
        // TODO: Make this async for better UX
        self.execute_query(&input)?;

        Ok(())
    }

    fn execute_query(&mut self, question: &str) -> Result<()> {
        self.waiting = true;
        self.status_message = Some("Analyzing question...".to_string());

        // Create progress channel
        let (tx, rx) = mpsc::channel();
        self.progress_rx = Some(rx);

        // Clone data needed for background thread
        let question = question.to_string();
        let cache_path = self.cache.path().to_path_buf();
        let provider_name = self.provider_name.clone();
        let model_override = self.model_override.clone();

        // Build conversation history for triage
        let conversation_history = self.session.build_context();

        // Spawn background thread for async work
        std::thread::spawn(move || {
            // Create tokio runtime in background thread
            let runtime = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = tx.send(PhaseUpdate::Error {
                        error: format!("Failed to create async runtime: {}", e),
                    });
                    return;
                }
            };

            runtime.block_on(async {
                execute_query_async(
                    &question,
                    &conversation_history,
                    cache_path,
                    &provider_name,
                    model_override.as_deref(),
                    tx,
                ).await
            });
        });

        Ok(())
    }

    fn handle_progress_update(&mut self, update: PhaseUpdate) {
        match update {
            PhaseUpdate::Triaging => {
                self.status_message = Some("Analyzing question...".to_string());
            }
            PhaseUpdate::AnsweringFromContext => {
                self.status_message = Some("Answering from conversation...".to_string());
            }
            PhaseUpdate::Thinking { reasoning, needs_context } => {
                self.status_message = Some("Thinking...".to_string());
                self.session.add_thinking_message(reasoning, needs_context);
                self.scroll_offset = 0; // Auto-scroll to bottom
            }
            PhaseUpdate::Tools { content, tool_calls } => {
                self.status_message = Some(format!("Gathering context ({} tools)...", tool_calls.len()));
                self.session.add_tools_message(content, tool_calls);
                self.scroll_offset = 0;
            }
            PhaseUpdate::Queries { queries } => {
                self.status_message = Some(format!("Generated {} queries...", queries.len()));
                self.session.add_queries_message(queries);
                self.scroll_offset = 0;
            }
            PhaseUpdate::Executing { results_count, execution_time_ms } => {
                self.status_message = Some(format!("Found {} result{}...", results_count, if results_count == 1 { "" } else { "s" }));
                self.session.add_execution_message(results_count, execution_time_ms);
                self.scroll_offset = 0;
            }
            PhaseUpdate::Reindexing { current, total, message } => {
                let percentage = if total > 0 {
                    (current as f32 / total as f32 * 100.0) as u8
                } else {
                    0
                };
                self.status_message = Some(format!(
                    "Reindexing cache: [{}/{}] {}% - {}",
                    current, total, percentage, message
                ));
                // Don't scroll during reindexing - user should stay where they are
            }
            PhaseUpdate::Answer { answer } => {
                self.status_message = Some("Generating answer...".to_string());
                self.session.add_answer_message(answer);
                self.scroll_offset = 0;
            }
            PhaseUpdate::Error { error } => {
                self.session.add_system_message(format!("Error: {}", error));
                self.waiting = false;
                self.status_message = Some(format!("❌ Error: {}", error));
                self.progress_rx = None;
                self.scroll_offset = 0;
            }
            PhaseUpdate::Notice { message } => {
                self.status_message = Some(message);
            }
            PhaseUpdate::Done => {
                self.waiting = false;
                self.status_message = None;
                self.progress_rx = None;
            }
        }
    }

    fn handle_slash_command(&mut self, command: &str) -> Result<()> {
        let command = command.trim();

        match command {
            "/clear" => {
                self.session.clear();
                self.status_message = Some("✓ Conversation cleared".to_string());
                self.input.clear();
                self.cursor = 0;

                // Add welcome message again
                self.session.add_system_message(
                    "Conversation cleared. Start fresh!".to_string()
                );
            }
            "/compact" => {
                self.handle_compact()?;
                self.input.clear();
                self.cursor = 0;
            }
            "/help" => {
                self.session.add_system_message(
                    "Available slash commands:\n\
                     \n\
                     • /clear - Clear conversation history\n\
                     • /compact - Summarize old messages to save tokens\n\
                     • /model [provider] [model] - Show or change provider/model\n\
                     • /help - Show this help message\n\
                     \n\
                     Keyboard shortcuts:\n\
                     • Enter - Send message\n\
                     • Ctrl+C - Quit\n\
                     • Ctrl+L - Clear conversation\n\
                     • Ctrl+K - Compact conversation\n\
                     • Ctrl+U - Clear input\n\
                     • Up/Down - Scroll messages\n\
                     • PgUp/PgDn - Fast scroll".to_string()
                );
                self.input.clear();
                self.cursor = 0;
            }
            _ if command.starts_with("/model") => {
                self.handle_model_command(command)?;
                self.input.clear();
                self.cursor = 0;
            }
            _ => {
                self.status_message = Some(format!("Unknown command: {}", command));
            }
        }

        Ok(())
    }

    fn handle_compact(&mut self) -> Result<()> {
        // Prepare compaction (keep last 4 messages)
        let (old_messages, removed_count, tokens_saved_potential) = self.session.prepare_compaction(4);

        if old_messages.is_empty() {
            self.status_message = Some("Nothing to compact (less than 4 messages)".to_string());
            return Ok(());
        }

        self.waiting = true;
        self.status_message = Some("Compacting conversation...".to_string());

        // Create tokio runtime for async operations
        let runtime = tokio::runtime::Runtime::new()
            .context("Failed to create async runtime")?;

        // Initialize provider for summarization
        let provider_instance = {
            let mut config = super::config::load_config(self.cache.path())?;
            config.provider = self.provider_name.clone();
            let api_key = super::config::get_api_key(&config.provider)?;
            let model = super::config::resolve_model(&config, self.model_override.as_deref());
            super::providers::create_provider(&config.provider, api_key, model, super::config::get_provider_options(&config.provider), config.timeout_seconds)?
        };

        // Build summarization prompt
        let prompt = format!(
            "Summarize the following conversation history concisely while retaining \
             key technical details, code findings, and decisions made. \
             Provide a 2-3 paragraph summary.\n\n{}",
            old_messages
        );

        // Get summary from LLM
        let summary = runtime.block_on(async {
            provider_instance.complete(&prompt, false).await
        })?;

        // Apply compaction
        self.session.apply_compaction(removed_count, summary.clone());

        self.waiting = false;
        self.status_message = Some(format!(
            "✓ Compacted {} messages (saved ~{} tokens)",
            removed_count,
            tokens_saved_potential
        ));

        Ok(())
    }

    fn handle_model_command(&mut self, command: &str) -> Result<()> {
        let parts: Vec<&str> = command.split_whitespace().collect();

        // /model (no args) - show current
        if parts.len() == 1 {
            self.session.add_system_message(format!(
                "Current configuration:\n\
                 • Provider: {}\n\
                 • Model: {}\n\
                 \n\
                 Available providers: openai, anthropic, openrouter\n\
                 \n\
                 Usage:\n\
                 • /model <provider> - Switch provider (uses configured model or default)\n\
                 • /model <provider> <model> - Switch to specific provider and model",
                self.session.provider(),
                self.session.model()
            ));
            return Ok(());
        }

        // Extract provider and optional model
        let new_provider = parts[1].to_lowercase();
        let new_model_arg = parts.get(2).map(|s| s.to_string());

        // Validate provider
        let valid_providers = ["openai", "anthropic", "openrouter", "openai-compatible"];
        if !valid_providers.contains(&new_provider.as_str()) {
            self.status_message = Some(format!(
                "Invalid provider '{}'. Available: {}",
                new_provider,
                valid_providers.join(", ")
            ));
            return Ok(());
        }

        // Determine model (priority: command arg > user config > provider default)
        let new_model = super::config::resolve_model_for(
            &new_provider,
            None,
            new_model_arg.as_deref(),
        )
        .unwrap_or_else(|| super::providers::default_model_for(&new_provider).to_string());

        // openai-compatible has no built-in default; refuse the switch with a
        // friendly message rather than persisting a blank model name.
        if new_model.is_empty() {
            self.status_message = Some(format!(
                "/model {} requires a model name (e.g. /model {} <model>) — \
                 self-hosted endpoints have no default. Run 'rfx llm config' first.",
                new_provider, new_provider
            ));
            return Ok(());
        }

        // Update session
        self.session.update_provider(new_provider.clone(), new_model.clone());
        self.provider_name = new_provider.clone();
        self.model_override = new_model_arg.clone();

        // Persist to user config
        if let Err(e) = super::save_user_provider(&new_provider, Some(&new_model)) {
            log::warn!("Failed to save provider preference to config: {}", e);
            self.status_message = Some("⚠ Model changed but not saved to config".to_string());
        } else {
            self.status_message = Some(format!(
                "✓ Switched to {} ({})",
                new_provider,
                new_model
            ));
        }

        // Add system message
        self.session.add_system_message(format!(
            "Switched to provider '{}' with model '{}'.\n\
             \n\
             This preference has been saved to ~/.reflex/config.toml.",
            new_provider,
            new_model
        ));

        Ok(())
    }
}

/// Retry an async operation with exponential backoff
async fn retry_with_backoff<F, Fut, T>(
    mut operation: F,
    max_retries: usize,
    operation_name: &str,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut last_error = None;

    for attempt in 0..=max_retries {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                let err_msg = e.to_string();

                // Determine wait time based on error type
                let wait_ms = if err_msg.contains("Rate limit exceeded") || err_msg.contains("429") {
                    // Rate limit: longer wait
                    5000 * (attempt as u64 + 1)
                } else if err_msg.contains("timeout") || err_msg.contains("Timeout") {
                    // Timeout: moderate wait
                    2000 * (attempt as u64 + 1)
                } else {
                    // Other errors: standard exponential backoff
                    1000 * (attempt as u64 + 1)
                };

                if attempt < max_retries {
                    log::warn!(
                        "{} failed (attempt {}/{}): {}. Retrying in {}ms...",
                        operation_name,
                        attempt + 1,
                        max_retries + 1,
                        err_msg,
                        wait_ms
                    );
                    tokio::time::sleep(tokio::time::Duration::from_millis(wait_ms)).await;
                }

                last_error = Some(e);
            }
        }
    }

    Err(last_error.unwrap())
}

/// Triage a question to determine if it needs codebase search
async fn triage_question(
    question: &str,
    conversation_history: &str,
    provider_name: &str,
    model_override: Option<&str>,
    cache_path: &std::path::Path,
) -> Result<TriageDecision> {
    // Create provider for triage
    let provider_instance = {
        let mut config = super::config::load_config(cache_path)?;
        config.provider = provider_name.to_string();
        let api_key = super::config::get_api_key(&config.provider)?;
        let model = super::config::resolve_model(&config, model_override);
        super::providers::create_provider(&config.provider, api_key, model, super::config::get_provider_options(&config.provider), config.timeout_seconds)?
    };

    // Build triage prompt
    let triage_prompt = format!(
        "You are a helpful coding assistant with access to a codebase search engine.\n\
         \n\
         {}\n\
         \n\
         USER'S NEW QUESTION: {}\n\
         \n\
         TASK: Determine if you can answer this question using ONLY the conversation history above, \
         or if you need to search the codebase.\n\
         \n\
         Answer \"direct\" if:\n\
         - It's a follow-up question about something already discussed\n\
         - It's asking for clarification or explanation of prior context\n\
         - It's a general programming question not specific to this codebase\n\
         - Examples: \"What does that mean?\", \"Can you explain X?\", \"Why?\"\n\
         \n\
         Answer \"search\" if:\n\
         - It's asking about code not yet discussed\n\
         - It requires finding specific files, functions, or patterns\n\
         - It's a new topic requiring codebase investigation\n\
         - Examples: \"How is auth implemented?\", \"Find all uses of X\", \"Where is Y defined?\"\n\
         \n\
         Respond with ONLY a single word: either \"direct\" or \"search\"",
        conversation_history,
        question
    );

    // Call LLM for triage
    let response = provider_instance.complete(&triage_prompt, false).await?;
    let decision = response.trim().to_lowercase();

    if decision.contains("direct") {
        Ok(TriageDecision::DirectAnswer)
    } else {
        Ok(TriageDecision::NeedsSearch {
            reasoning: "Question requires codebase search".to_string(),
        })
    }
}

/// Execute query asynchronously and send progress updates
async fn execute_query_async(
    question: &str,
    conversation_history: &str,
    cache_path: std::path::PathBuf,
    provider_name: &str,
    model_override: Option<&str>,
    tx: Sender<PhaseUpdate>,
) {
    // Recreate cache manager from root directory
    // cache_path is .reflex/, so get parent to pass to CacheManager::new
    let root_dir = cache_path.parent().unwrap_or(&cache_path);
    let cache = CacheManager::new(root_dir);

    // Extract codebase context (always available metadata: languages, file counts, directories)
    let codebase_context_str = super::context::CodebaseContext::extract(&cache)
        .ok()
        .map(|ctx| ctx.to_prompt_string());

    // TRIAGE PHASE: Decide if we need to search or can answer directly
    let _ = tx.send(PhaseUpdate::Triaging);

    let decision = match triage_question(
        question,
        conversation_history,
        provider_name,
        model_override,
        &cache_path,
    ).await {
        Ok(decision) => decision,
        Err(e) => {
            let msg = format!("LLM unavailable, falling back to search: {}", e);
            log::warn!("{}", msg);
            let _ = tx.send(PhaseUpdate::Notice { message: msg });
            TriageDecision::NeedsSearch {
                reasoning: "Triage failed, using search as fallback".to_string(),
            }
        }
    };

    match decision {
        TriageDecision::DirectAnswer => {
            // FAST PATH: Answer from conversation context
            let _ = tx.send(PhaseUpdate::AnsweringFromContext);

            // Generate answer using conversation history
            let provider_instance = match (|| -> Result<_> {
                let mut config = super::config::load_config(&cache_path)?;
                config.provider = provider_name.to_string();
                let api_key = super::config::get_api_key(&config.provider)?;
                let model = super::config::resolve_model(&config, model_override);
                super::providers::create_provider(&config.provider, api_key, model, super::config::get_provider_options(&config.provider), config.timeout_seconds)
            })() {
                Ok(provider) => provider,
                Err(e) => {
                    let _ = tx.send(PhaseUpdate::Error {
                        error: format!("Failed to create provider: {}", e),
                    });
                    return;
                }
            };

            let answer_prompt = format!(
                "{}\n\nUSER'S QUESTION: {}\n\n\
                 Answer the question based on the conversation history above. \
                 Be concise and helpful.",
                conversation_history,
                question
            );

            // Retry answer generation with exponential backoff
            let answer_result = retry_with_backoff(
                || async {
                    provider_instance.complete(&answer_prompt, false).await
                },
                2, // max 2 retries
                "Answer generation"
            ).await;

            match answer_result {
                Ok(answer) => {
                    let _ = tx.send(PhaseUpdate::Answer { answer });
                    let _ = tx.send(PhaseUpdate::Done);
                }
                Err(e) => {
                    // Fallback: If direct answer fails after retries, try search instead
                    log::warn!("Direct answer failed after retries, falling back to search: {}", e);
                    let _ = tx.send(PhaseUpdate::Thinking {
                        reasoning: format!(
                            "Direct answer failed ({}), searching codebase as fallback",
                            e
                        ),
                        needs_context: true,
                    });

                    // Run search path (copy of agentic path below)
                    let agentic_config = AgenticConfig {
                        max_iterations: 2,
                        max_tools_per_phase: 5,
                        enable_evaluation: true,
                        eval_config: Default::default(),
                        provider_override: Some(provider_name.to_string()),
                        model_override: model_override.map(|s| s.to_string()),
                        show_reasoning: false,
                        verbose: false,
                        debug: false,
                    };

                    let reporter = Box::new(super::QuietReporter);

                    match super::run_agentic_loop(question, &cache, agentic_config, &*reporter).await {
                        Ok(agentic_response) => {
                            // Send tools phase update if tools were executed
                            if let Some(ref tools) = agentic_response.tools_executed {
                                if !tools.is_empty() {
                                    let content = format!("Gathered context using {} tools", tools.len());
                                    let _ = tx.send(PhaseUpdate::Tools {
                                        content,
                                        tool_calls: tools.clone(),
                                    });
                                }
                            }

                            // Get results count (needed for answer generation)
                            let results_count = agentic_response.total_count.unwrap_or(0);

                            // Send queries phase update only if queries were generated
                            if !agentic_response.queries.is_empty() {
                                let query_strings: Vec<String> = agentic_response.queries
                                    .iter()
                                    .map(|q| q.command.clone())
                                    .collect();

                                let _ = tx.send(PhaseUpdate::Queries { queries: query_strings });

                                let _ = tx.send(PhaseUpdate::Executing {
                                    results_count,
                                    execution_time_ms: 0,
                                });
                            }

                            let provider_instance = match (|| -> Result<_> {
                                let mut config = super::config::load_config(&cache_path)?;
                                config.provider = provider_name.to_string();
                                let api_key = super::config::get_api_key(&config.provider)?;
                                let model = super::config::resolve_model(&config, model_override);
                                super::providers::create_provider(&config.provider, api_key, model, super::config::get_provider_options(&config.provider), config.timeout_seconds)
                            })() {
                                Ok(provider) => provider,
                                Err(e) => {
                                    let _ = tx.send(PhaseUpdate::Error {
                                        error: format!("Failed to create provider for fallback: {}", e),
                                    });
                                    return;
                                }
                            };

                            match super::generate_answer(
                                question,
                                &agentic_response.results,
                                results_count,
                                agentic_response.gathered_context.as_deref(),
                                codebase_context_str.as_deref(),
                                &*provider_instance,
                            ).await {
                                Ok(answer) => {
                                    let _ = tx.send(PhaseUpdate::Answer { answer });
                                    let _ = tx.send(PhaseUpdate::Done);
                                }
                                Err(e) => {
                                    let _ = tx.send(PhaseUpdate::Error {
                                        error: format!("Fallback search failed: {}", e),
                                    });
                                }
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(PhaseUpdate::Error {
                                error: format!("Both direct answer and search failed: {}", e),
                            });
                        }
                    }
                }
            }
        }

        TriageDecision::NeedsSearch { reasoning } => {
            // AGENTIC PATH: Full search pipeline
            let _ = tx.send(PhaseUpdate::Thinking {
                reasoning,
                needs_context: true,
            });

            // Configure agentic mode
            let agentic_config = AgenticConfig {
                max_iterations: 2,
                max_tools_per_phase: 5,
                enable_evaluation: true,
                eval_config: Default::default(),
                provider_override: Some(provider_name.to_string()),
                model_override: model_override.map(|s| s.to_string()),
                show_reasoning: false,
                verbose: false,
                debug: false,
            };

            // Use quiet reporter to suppress console output
            let reporter = Box::new(super::QuietReporter);

            // Run agentic loop
            let agentic_response = match super::run_agentic_loop(
                question,
                &cache,
                agentic_config,
                &*reporter,
            ).await {
                Ok(response) => response,
                Err(e) => {
                    let _ = tx.send(PhaseUpdate::Error {
                        error: format!("Agentic loop failed: {}", e),
                    });
                    return;
                }
            };

            // Send tools phase update if tools were executed
            if let Some(ref tools) = agentic_response.tools_executed {
                if !tools.is_empty() {
                    let content = format!("Gathered context using {} tools", tools.len());
                    let _ = tx.send(PhaseUpdate::Tools {
                        content,
                        tool_calls: tools.clone(),
                    });
                }
            }

            // Get results count (needed for answer generation)
            let results_count = agentic_response.total_count.unwrap_or(0);

            // Send queries phase update only if queries were generated
            if !agentic_response.queries.is_empty() {
                let query_strings: Vec<String> = agentic_response.queries
                    .iter()
                    .map(|q| q.command.clone())
                    .collect();

                let _ = tx.send(PhaseUpdate::Queries {
                    queries: query_strings,
                });

                // Send execution phase update only if queries were executed
                let start_time = std::time::Instant::now();
                let execution_time_ms = start_time.elapsed().as_millis() as u64;

                let _ = tx.send(PhaseUpdate::Executing {
                    results_count,
                    execution_time_ms,
                });
            }

            // Generate answer
            let provider_instance = match (|| -> Result<_> {
                let mut config = super::config::load_config(&cache_path)?;
                config.provider = provider_name.to_string();
                let api_key = super::config::get_api_key(&config.provider)?;
                let model = super::config::resolve_model(&config, model_override);
                super::providers::create_provider(&config.provider, api_key, model, super::config::get_provider_options(&config.provider), config.timeout_seconds)
            })() {
                Ok(provider) => provider,
                Err(e) => {
                    let _ = tx.send(PhaseUpdate::Error {
                        error: format!("Failed to create provider: {}", e),
                    });
                    return;
                }
            };

            let answer = match super::generate_answer(
                question,
                &agentic_response.results,
                results_count,
                agentic_response.gathered_context.as_deref(),
                codebase_context_str.as_deref(),
                &*provider_instance,
            ).await {
                Ok(answer) => answer,
                Err(e) => {
                    let _ = tx.send(PhaseUpdate::Error {
                        error: format!("Failed to generate answer: {}", e),
                    });
                    return;
                }
            };

            // Send answer phase update
            let _ = tx.send(PhaseUpdate::Answer { answer });

            // Send done signal
            let _ = tx.send(PhaseUpdate::Done);
        }
    }
}

/// Setup terminal for TUI mode
fn setup_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(
        stdout,
        crossterm::terminal::EnterAlternateScreen,
        crossterm::event::EnableMouseCapture,
        crossterm::cursor::Show
    )?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;
    Ok(terminal)
}

/// Restore terminal after TUI mode
fn restore_terminal(mut terminal: Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    crossterm::terminal::disable_raw_mode()?;
    crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen,
        crossterm::event::DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    Ok(())
}

/// Run interactive chat mode
pub fn run_chat_mode(
    cache: CacheManager,
    provider: Option<String>,
    model: Option<String>,
) -> Result<()> {
    // Determine provider
    let provider_name = if let Some(p) = provider {
        p
    } else {
        // Load from config
        let config = super::config::load_config(cache.path())?;
        config.provider
    };

    let mut app = ChatApp::new(cache, provider_name, model)?;
    app.run()
}
