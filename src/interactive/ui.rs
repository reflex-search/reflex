use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, Borders, List, ListItem, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
        Wrap,
    },
};

use super::app::{AppMode, FocusState, IndexStatusState, InteractiveApp};

/// Main render function
pub fn render(f: &mut Frame, app: &mut InteractiveApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header (input)
            Constraint::Length(3), // Filters
            Constraint::Min(1),    // Content area
            Constraint::Length(1), // Footer/help bar
        ])
        .split(f.area());

    render_header(f, chunks[0], &*app);
    render_filters(f, chunks[1], &*app);

    match app.mode() {
        AppMode::Help => render_help_screen(f, chunks[2], app),
        AppMode::FilePreview => render_file_preview(f, chunks[2], app),
        AppMode::FilterSelector => {
            render_results_area(f, chunks[2], app);
            // Render filter selector modal on top
            // Clone theme to avoid borrow conflict
            let theme = app.theme().clone();
            if let Some(selector) = app.filter_selector_mut() {
                selector.render(f, chunks[2], &theme);
            }
        }
        AppMode::Indexing | AppMode::Normal => render_results_area(f, chunks[2], app),
    }

    render_footer(f, chunks[3], app);
}

fn render_header(f: &mut Frame, area: Rect, app: &InteractiveApp) {
    let palette = &app.theme().palette;
    let input_focused = matches!(app.focus_state(), FocusState::Input);

    // Build title line with status on the right
    let title_left = if input_focused {
        " Search [TYPING - Press Tab/Enter to navigate] "
    } else {
        " Search [Press Tab to focus, / to type] "
    };

    // Build status indicators for title right
    let main_status = match app.index_status() {
        IndexStatusState::Ready { file_count, .. } => {
            format!("✓ {} files", file_count)
        }
        IndexStatusState::Missing => "⚠ No index".to_string(),
        IndexStatusState::Stale { files_changed, .. } => {
            format!("⚠ {} changed", files_changed)
        }
        IndexStatusState::Indexing { current, total, .. } => {
            format!("⏳ {}/{}", current, total)
        }
    };

    // Get animated spinner for running symbol indexing
    let spinner_frames = ['◐', '◓', '◑', '◒'];
    let frame_idx = (app.effects().frame() / 3) as usize % spinner_frames.len();
    let spinner = spinner_frames[frame_idx];

    // Build symbol indexing status indicator (ALWAYS VISIBLE)
    let symbol_status = match app.index_status() {
        IndexStatusState::Ready { symbol_status, .. }
        | IndexStatusState::Stale { symbol_status, .. }
        | IndexStatusState::Indexing { symbol_status, .. } => match symbol_status {
            super::app::SymbolIndexingState::Running { processed, total } => {
                let percent = if *total > 0 {
                    (*processed as f64 / *total as f64 * 100.0) as u32
                } else {
                    0
                };
                format!(" │ {} {}% [Shift+I: clear]", spinner, percent)
            }
            super::app::SymbolIndexingState::Completed => {
                " │ ✓ symbols [Shift+I: clear]".to_string()
            }
            super::app::SymbolIndexingState::Failed => " │ ⚠ symbols [Shift+I: clear]".to_string(),
            super::app::SymbolIndexingState::NotStarted => {
                " │ ⏳ symbols... [Shift+I: clear]".to_string()
            }
        },
        IndexStatusState::Missing => String::new(),
    };

    let status_indicator = format!("{}{} ", main_status, symbol_status);

    // Calculate spacing to push status to the right
    let available_width = area.width.saturating_sub(2) as usize; // Subtract borders
    let title_len = title_left.chars().count();
    let status_len = status_indicator.chars().count();
    let spaces_needed = available_width.saturating_sub(title_len + status_len);
    let spacing = " ".repeat(spaces_needed);

    // Build complete title line
    let title_spans = vec![
        Span::raw(title_left),
        Span::raw(spacing),
        Span::styled(status_indicator, Style::default().fg(palette.muted)),
    ];

    let input_block = Block::default()
        .borders(Borders::ALL)
        .title_top(Line::from(title_spans))
        .border_style(if input_focused {
            Style::default()
                .fg(palette.accent)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(palette.muted)
        });

    let input_text = app.input().value();
    let input_style = if input_focused {
        Style::default()
            .fg(palette.foreground)
            .bg(Color::Rgb(40, 40, 40)) // Subtle background highlight when focused
    } else {
        Style::default().fg(palette.foreground).bg(Color::Black) // Explicit background to prevent rendering artifacts
    };

    let input_paragraph = Paragraph::new(input_text)
        .block(input_block)
        .style(input_style)
        .wrap(Wrap { trim: false }); // Ensure text doesn't overflow the widget bounds

    f.render_widget(input_paragraph, area);

    // Set cursor position if input is focused
    if input_focused {
        let cursor_x = area.x + 1 + app.input().visual_cursor() as u16;
        let cursor_y = area.y + 1;
        f.set_cursor_position((cursor_x.min(area.right() - 2), cursor_y));
    }
}

fn render_filters(f: &mut Frame, area: Rect, app: &InteractiveApp) {
    // Clone what we need upfront to avoid borrow checker issues
    let filters = app.filters().clone();
    let palette = app.theme().palette.clone();
    let filters_focused = matches!(app.focus_state(), FocusState::Filters);

    let border_style = if filters_focused {
        Style::default()
            .fg(palette.accent)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(palette.muted)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Filters [s: symbols, r: regex, l: lang, k: kind, e: expand, c: contains] ")
        .border_style(border_style);

    // Create clickable filter buttons - ALL VISIBLE ALL THE TIME
    // Track position for accurate mouse click detection
    let mut filter_spans = vec![];
    let mut pos = 0; // Current column position
    let inactive_style = Style::default()
        .fg(palette.muted)
        .bg(Color::Rgb(30, 30, 30));

    // Symbols button
    let symbols_text = " [s] Symbols ";
    let symbols_style = if filters.symbols_mode {
        Style::default()
            .fg(Color::Black)
            .bg(palette.badge_active)
            .add_modifier(Modifier::BOLD)
    } else {
        inactive_style
    };
    let symbols_start = pos;
    filter_spans.push(Span::styled(symbols_text, symbols_style));
    pos += symbols_text.len();
    let symbols_end = pos;
    filter_spans.push(Span::raw("  "));
    pos += 2;

    // Regex button
    let regex_text = " [r] Regex ";
    let regex_style = if filters.regex_mode {
        Style::default()
            .fg(Color::Black)
            .bg(palette.warning)
            .add_modifier(Modifier::BOLD)
    } else {
        inactive_style
    };
    let regex_start = pos;
    filter_spans.push(Span::styled(regex_text, regex_style));
    pos += regex_text.len();
    let regex_end = pos;
    filter_spans.push(Span::raw("  "));
    pos += 2;

    // Language filter (always visible)
    let lang_style = if filters.language.is_some() {
        Style::default()
            .fg(Color::Black)
            .bg(palette.info)
            .add_modifier(Modifier::BOLD)
    } else {
        inactive_style
    };
    let lang_text = if let Some(ref lang) = filters.language {
        format!(" [l] Lang: {} ", lang)
    } else {
        " [l] Lang ".to_string()
    };
    let lang_start = pos;
    filter_spans.push(Span::styled(lang_text.clone(), lang_style));
    pos += lang_text.len();
    let lang_end = pos;
    filter_spans.push(Span::raw("  "));
    pos += 2;

    // Kind filter (always visible)
    let kind_style = if filters.kind.is_some() {
        Style::default()
            .fg(Color::Black)
            .bg(palette.info)
            .add_modifier(Modifier::BOLD)
    } else {
        inactive_style
    };
    let kind_text = if let Some(ref kind) = filters.kind {
        format!(" [k] Kind: {} ", kind)
    } else {
        " [k] Kind ".to_string()
    };
    let kind_start = pos;
    filter_spans.push(Span::styled(kind_text.clone(), kind_style));
    pos += kind_text.len();
    let kind_end = pos;
    filter_spans.push(Span::raw("  "));
    pos += 2;

    // Expand mode (always visible)
    let expand_text = " [e] Expand ";
    let expand_style = if filters.expand {
        Style::default()
            .fg(Color::Black)
            .bg(Color::Rgb(150, 200, 100))
            .add_modifier(Modifier::BOLD)
    } else {
        inactive_style
    };
    let expand_start = pos;
    filter_spans.push(Span::styled(expand_text, expand_style));
    pos += expand_text.len();
    let expand_end = pos;
    filter_spans.push(Span::raw("  "));
    pos += 2;

    // Contains mode (always visible)
    let contains_text = " [c] Contains ";
    let contains_style = if filters.contains {
        Style::default()
            .fg(Color::Black)
            .bg(Color::Rgb(180, 120, 200))
            .add_modifier(Modifier::BOLD)
    } else {
        inactive_style
    };
    let contains_start = pos;
    filter_spans.push(Span::styled(contains_text, contains_style));
    pos += contains_text.len();
    let contains_end = pos;

    // Store badge positions for mouse click detection (using interior mutability)
    let badge_positions = app.filter_badge_positions();
    badge_positions.symbols.set((symbols_start, symbols_end));
    badge_positions.regex.set((regex_start, regex_end));
    badge_positions.language.set((lang_start, lang_end));
    badge_positions.kind.set((kind_start, kind_end));
    badge_positions.expand.set((expand_start, expand_end));
    badge_positions.contains.set((contains_start, contains_end));

    let paragraph = Paragraph::new(Line::from(filter_spans))
        .block(block)
        .style(Style::default().fg(palette.foreground).bg(Color::Black)) // Explicit background to prevent content bleeding through
        .wrap(Wrap { trim: false }); // Ensure badges don't overflow the widget bounds

    f.render_widget(paragraph, area);
}

fn render_results_area(f: &mut Frame, area: Rect, app: &InteractiveApp) {
    let palette = &app.theme().palette;

    // Show animated indexing modal (takes priority over search)
    if app.indexing() {
        // Create a centered modal for the indexing animation
        let modal_width = 50.min(area.width.saturating_sub(4));
        let modal_height = 11; // Increased to fit status message
        let modal_x = (area.width.saturating_sub(modal_width)) / 2;
        let modal_y = (area.height.saturating_sub(modal_height)) / 2;
        let modal_area = Rect::new(
            area.x + modal_x,
            area.y + modal_y,
            modal_width,
            modal_height,
        );

        // Render background (dimmed results area) - fill entire area to prevent artifacts
        let background_paragraph = Paragraph::new("")
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Results ")
                    .border_style(Style::default().fg(palette.muted)),
            )
            .style(Style::default().bg(Color::Black));
        f.render_widget(background_paragraph, area);

        // Animate the spinner character based on frame count
        let spinner_frames = ['◐', '◓', '◑', '◒'];
        let frame_idx = (app.effects().frame() / 3) as usize % spinner_frames.len();
        let spinner = spinner_frames[frame_idx];

        // Get elapsed time
        let elapsed_secs = app.indexing_elapsed_secs().unwrap_or(0);
        let elapsed_text = if elapsed_secs < 60 {
            format!("{}s", elapsed_secs)
        } else {
            format!("{}m {}s", elapsed_secs / 60, elapsed_secs % 60)
        };

        // Get progress info from index status
        let (current, total, percent, status_msg) = match app.index_status() {
            crate::interactive::app::IndexStatusState::Indexing {
                current,
                total,
                status,
                ..
            } => {
                let pct = if *total > 0 {
                    (*current as f64 / *total as f64 * 100.0) as u32
                } else {
                    0
                };
                (*current, *total, pct, status.clone())
            }
            _ => (0, 0, 0, "Indexing...".to_string()),
        };

        // Create animated progress bar
        let bar_width = 32;
        let filled = if total > 0 {
            ((current as f64 / total as f64) * bar_width as f64) as usize
        } else {
            // Indeterminate progress - animated
            let pos = (app.effects().frame() / 2) as usize % bar_width;
            pos.min(bar_width - 4)
        };

        let progress_bar = if total > 0 {
            // Determinate progress bar
            let filled_chars = "█".repeat(filled);
            let empty_chars = "░".repeat(bar_width.saturating_sub(filled));
            format!("{}{}", filled_chars, empty_chars)
        } else {
            // Indeterminate animated progress bar
            let mut chars = vec!['░'; bar_width];
            for i in 0..4 {
                let pos = (filled + i) % bar_width;
                chars[pos] = '█';
            }
            chars.iter().collect()
        };

        // Create status line
        let status_line = if total > 0 {
            format!(
                "{}/{} files ({}%) • {}",
                current, total, percent, elapsed_text
            )
        } else {
            format!("Indexing... • {}", elapsed_text)
        };

        // Create animated loading text with multiple lines
        let loading_lines = vec![
            Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    spinner.to_string(),
                    Style::default()
                        .fg(Color::Rgb(255, 150, 0))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
            ]),
            Line::from(""),
            Line::from(vec![Span::styled(
                "Building index",
                Style::default()
                    .fg(palette.accent)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(""),
            Line::from(vec![Span::styled(
                progress_bar,
                Style::default().fg(palette.info),
            )]),
            Line::from(""),
            Line::from(vec![Span::styled(
                status_line,
                Style::default().fg(palette.muted),
            )]),
            Line::from(""),
            Line::from(vec![Span::styled(
                status_msg,
                Style::default().fg(Color::Rgb(150, 150, 150)),
            )]),
        ];

        // Render modal with animated border
        let modal = Paragraph::new(loading_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(
                        Style::default()
                            .fg(Color::Rgb(255, 150, 0))
                            .add_modifier(Modifier::BOLD),
                    )
                    .title(Line::from(vec![
                        Span::raw(" "),
                        Span::styled("📦", Style::default().fg(Color::Rgb(255, 200, 0))),
                        Span::raw(" Indexing "),
                        Span::styled("📦", Style::default().fg(Color::Rgb(255, 200, 0))),
                        Span::raw(" "),
                    ]))
                    .style(Style::default().bg(Color::Rgb(25, 20, 30))),
            )
            .alignment(Alignment::Center);

        f.render_widget(modal, modal_area);
        return;
    }

    // Show animated loading modal if query is executing
    if app.searching() {
        // Create a centered modal for the loading animation
        let modal_width = 50.min(area.width.saturating_sub(4));
        let modal_height = 9;
        let modal_x = (area.width.saturating_sub(modal_width)) / 2;
        let modal_y = (area.height.saturating_sub(modal_height)) / 2;
        let modal_area = Rect::new(
            area.x + modal_x,
            area.y + modal_y,
            modal_width,
            modal_height,
        );

        // Render background (dimmed results area) - fill entire area to prevent artifacts
        let background_paragraph = Paragraph::new("")
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Results ")
                    .border_style(Style::default().fg(palette.muted)),
            )
            .style(Style::default().bg(Color::Black));
        f.render_widget(background_paragraph, area);

        // Animate the spinner character based on frame count
        let spinner_frames = ['◐', '◓', '◑', '◒'];
        let frame_idx = (app.effects().frame() / 3) as usize % spinner_frames.len();
        let spinner = spinner_frames[frame_idx];

        // Create animated loading text with multiple lines
        let loading_lines = vec![
            Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    spinner.to_string(),
                    Style::default()
                        .fg(Color::Rgb(0, 200, 255))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
            ]),
            Line::from(""),
            Line::from(vec![Span::styled(
                "Searching codebase",
                Style::default()
                    .fg(palette.accent)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(""),
            Line::from(vec![Span::styled(
                "━━━━━━━━━━━━━━━━",
                Style::default().fg(palette.info),
            )]),
            Line::from(""),
            Line::from(vec![Span::styled(
                "Hang tight...",
                Style::default().fg(palette.muted),
            )]),
        ];

        // Render modal with animated border
        let modal = Paragraph::new(loading_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(
                        Style::default()
                            .fg(Color::Rgb(0, 200, 255))
                            .add_modifier(Modifier::BOLD),
                    )
                    .title(Line::from(vec![
                        Span::raw(" "),
                        Span::styled("⚡", Style::default().fg(Color::Rgb(255, 200, 0))),
                        Span::raw(" Loading "),
                        Span::styled("⚡", Style::default().fg(Color::Rgb(255, 200, 0))),
                        Span::raw(" "),
                    ]))
                    .style(Style::default().bg(Color::Rgb(20, 20, 30))),
            )
            .alignment(Alignment::Center);

        f.render_widget(modal, modal_area);
        return;
    }

    // Show error message if present
    if let Some(error) = app.error_message() {
        let error_text = Paragraph::new(error)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Error ")
                    .border_style(Style::default().fg(palette.error)),
            )
            .style(Style::default().fg(palette.error))
            .wrap(Wrap { trim: true });
        f.render_widget(error_text, area);
        return;
    }

    // Show info message if present
    if let Some(info) = app.info_message() {
        let info_text = Paragraph::new(info)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Info ")
                    .border_style(Style::default().fg(palette.info)),
            )
            .style(Style::default().fg(palette.info))
            .wrap(Wrap { trim: true });
        f.render_widget(info_text, area);
        return;
    }

    let results = app.results();

    if results.is_empty() {
        // Show empty state with helpful instructions
        let empty_message = if app.input().value().trim().is_empty() {
            if matches!(app.focus_state(), FocusState::Input) {
                "Start typing to search...\n\nKeyboard shortcuts:\n  j/k or ↓/↑ - Navigate results\n  / - Focus search\n  Esc/Enter - Unfocus search\n  ? - Show help\n  q - Quit"
            } else {
                "Press / to start typing a search query\nPress ? for full help"
            }
        } else {
            "No results found. Try a different query.\n\nTip: Press / to edit your search"
        };

        let empty_text = Paragraph::new(empty_message)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Results ")
                    .border_style(Style::default().fg(palette.muted)),
            )
            .style(Style::default().fg(palette.muted))
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true });

        f.render_widget(empty_text, area);
        return;
    }

    // Render result list with variable-height items
    // Load theme for syntax highlighting
    let theme = app.theme().load_syntect_theme();

    // First pass: calculate how many results fit in visible area
    let visible_lines = area.height.saturating_sub(2) as usize;
    let mut lines_used = 0;
    let mut visible_results_count = 0;
    const MAX_PREVIEW_LINES: usize = 20; // Limit preview to 20 lines max to prevent expand mode breakage

    for result in results.results().iter().skip(results.scroll_offset()) {
        // Calculate lines for this result
        let has_symbol = !matches!(result.kind, crate::models::SymbolKind::Unknown(_))
            && result.symbol.is_some();
        let symbol_lines = if has_symbol { 1 } else { 0 };
        let path_lines = 1;
        // Limit preview lines to prevent expand mode from breaking scroll window
        let preview_lines = result.preview.lines().count().min(MAX_PREVIEW_LINES);
        let total_lines = symbol_lines + path_lines + preview_lines;

        if lines_used + total_lines <= visible_lines {
            lines_used += total_lines;
            visible_results_count += 1;
        } else {
            break;
        }
    }

    // Update the results list visible height for scroll calculations
    // This uses interior mutability (Cell) to update during rendering
    results.set_visible_height(visible_results_count);

    let items: Vec<ListItem> = results
        .visible_results(visible_results_count)
        .iter()
        .enumerate()
        .map(|(idx, result)| {
            let global_idx = idx + results.scroll_offset();
            let is_selected = global_idx == results.selected_index();

            // Make path relative to project root
            let relative_path = std::path::Path::new(&result.path)
                .strip_prefix(app.cwd())
                .ok()
                .and_then(|p| p.to_str())
                .map(|p| format!("./{}", p))
                .unwrap_or_else(|| result.path.clone());

            let mut lines = Vec::new();

            // Check if result has symbol information
            let has_symbol = !matches!(result.kind, crate::models::SymbolKind::Unknown(_))
                && result.symbol.is_some();

            // Add symbol line if present (before file path)
            if has_symbol {
                let symbol_text = format!("[{}] {}", result.kind, result.symbol.as_ref().unwrap());

                if is_selected {
                    lines.push(Line::from(symbol_text));
                } else {
                    lines.push(Line::from(vec![Span::styled(
                        symbol_text,
                        Style::default().fg(Color::Rgb(200, 150, 255)), // Purple/magenta
                    )]));
                }
            }

            // Add file path line
            let file_line_text = format!("{}:{}", relative_path, result.span.start_line);
            if is_selected {
                lines.push(Line::from(file_line_text));
            } else {
                lines.push(Line::from(vec![Span::styled(
                    file_line_text,
                    Style::default().fg(palette.info), // Cyan for file path
                )]));
            }

            // Split preview into lines and apply syntax highlighting
            // Limit to MAX_PREVIEW_LINES to prevent expand mode from breaking scroll window
            let preview_lines_vec: Vec<String> = result
                .preview
                .lines()
                .take(MAX_PREVIEW_LINES)
                .map(|s| s.to_string())
                .collect();

            if is_selected {
                // When selected, use simple formatting (background will be applied)
                for line in preview_lines_vec {
                    lines.push(Line::from(format!("    {}", line)));
                }
            } else {
                // Apply syntax highlighting to preview lines
                let highlighted =
                    super::syntax::highlight_code_lines(&preview_lines_vec, result.lang, &theme);

                for highlighted_line in highlighted {
                    // Add indentation and prepend to highlighted spans
                    let mut line_spans = vec![Span::raw("    ")];
                    line_spans.extend(highlighted_line.spans);
                    lines.push(Line::from(line_spans));
                }
            }

            // Apply selection style if selected
            if is_selected {
                let style = Style::default()
                    .fg(Color::Black)
                    .bg(palette.highlight)
                    .add_modifier(Modifier::BOLD);
                ListItem::new(lines).style(style)
            } else {
                ListItem::new(lines)
            }
        })
        .collect();

    let result_count = results.len();
    let title = format!(" Results ({}) ", result_count);

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(title)
            .border_style(Style::default().fg(palette.accent)),
    );

    f.render_widget(list, area);

    // Render scrollbar if there are more results than visible
    if result_count > visible_results_count {
        let mut scrollbar_state =
            ScrollbarState::new(result_count).position(results.selected_index());

        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(Some("▲"))
            .end_symbol(Some("▼"))
            .track_symbol(Some("│"))
            .thumb_symbol("█")
            .style(Style::default().fg(palette.accent));

        f.render_stateful_widget(
            scrollbar,
            area.inner(ratatui::layout::Margin {
                horizontal: 0,
                vertical: 1,
            }),
            &mut scrollbar_state,
        );
    }
}

fn render_help_screen(f: &mut Frame, area: Rect, app: &InteractiveApp) {
    let palette = &app.theme().palette;

    let help_text = vec![
        "",
        "  Reflex Interactive Mode - Keyboard Shortcuts",
        "  ═══════════════════════════════════════════════",
        "",
        "  Navigation:",
        "    j / ↓         Move to next result",
        "    k / ↑         Move to previous result",
        "    PageDown      Jump 10 results down",
        "    PageUp        Jump 10 results up",
        "    Home / g      Go to first result",
        "    End / G       Go to last result",
        "",
        "  Search:",
        "    /             Focus search input",
        "    Esc           Unfocus input / close help / close selector",
        "    Ctrl+P        Previous query from history",
        "    Ctrl+N        Next query from history",
        "",
        "  Filters:",
        "    s             Toggle symbols-only mode",
        "    r             Toggle regex mode",
        "    l             Select language filter",
        "    k             Select symbol kind filter",
        "    g             Add glob pattern (CLI only for now)",
        "    x             Add exclude pattern (CLI only for now)",
        "    e             Toggle expand mode (full definitions)",
        "    c             Toggle contains mode (substring)",
        "    Ctrl+L        Clear language filter",
        "    Ctrl+K        Clear kind filter",
        "",
        "  Actions:",
        "    o / Enter     Open file in $EDITOR / Expand preview",
        "    i             Trigger reindex",
        "    ?             Toggle this help screen",
        "    q / Ctrl+C    Quit",
        "",
        "  Mouse:",
        "    Click         Select result / Focus input / Toggle filters",
        "    Click status  Trigger reindex (top-right corner)",
        "    Scroll        Navigate results",
        "",
        "  Press '?' to close this help screen",
        "",
    ];

    let help_paragraph = Paragraph::new(help_text.join("\n"))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Help ")
                .border_style(Style::default().fg(palette.accent)),
        )
        .style(Style::default().fg(palette.foreground))
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: false });

    f.render_widget(help_paragraph, area);
}

fn render_file_preview(f: &mut Frame, area: Rect, app: &InteractiveApp) {
    let palette = &app.theme().palette;

    if let Some(preview) = app.preview_content() {
        let visible_height = area.height.saturating_sub(2) as usize;
        let start = preview.scroll_offset();
        let center = preview.center_line();
        let lang = preview.language();

        // Load theme for syntax highlighting
        let theme = app.theme().load_syntect_theme();

        // Get content lines in visible range
        let content_lines = preview.content();
        let end = (start + visible_height).min(content_lines.len());

        // IMPORTANT: Highlight from the beginning of the file up to the visible end
        // This ensures syntect maintains proper state (for multi-line strings, comments, etc.)
        // We'll only render the visible portion, but we need to process all lines up to that point
        let lines_to_highlight: Vec<String> = content_lines[..end].to_vec();
        let all_highlighted =
            super::syntax::highlight_code_lines(&lines_to_highlight, lang, &theme);

        // Extract only the visible portion
        let highlighted_lines: Vec<_> = all_highlighted.into_iter().skip(start).collect();

        let items: Vec<ListItem> = highlighted_lines
            .into_iter()
            .enumerate()
            .map(|(idx, highlighted_line)| {
                let line_number = start + idx + 1;
                let is_center = line_number == center;

                // Build the complete line with line number prefix
                let mut spans = vec![Span::styled(
                    format!("{:4} │ ", line_number),
                    Style::default().fg(palette.muted),
                )];

                // Add highlighted code spans
                spans.extend(highlighted_line.spans);

                let line_content = Line::from(spans);

                // Apply selection style if this is the center line
                if is_center {
                    ListItem::new(line_content).style(
                        Style::default()
                            .bg(palette.highlight)
                            .add_modifier(Modifier::BOLD),
                    )
                } else {
                    ListItem::new(line_content)
                }
            })
            .collect();

        // Make path relative to project root
        let relative_path = preview
            .path()
            .strip_prefix(app.cwd().to_str().unwrap_or(""))
            .unwrap_or(preview.path())
            .trim_start_matches('/');
        let relative_display = if relative_path.is_empty() {
            "./".to_string()
        } else {
            format!("./{}", relative_path)
        };
        let title = format!(" {} (line {}) ", relative_display, center);
        let list = List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(title)
                .border_style(Style::default().fg(palette.accent)),
        );

        f.render_widget(list, area);
    }
}

fn render_footer(f: &mut Frame, area: Rect, app: &InteractiveApp) {
    let palette = &app.theme().palette;

    // Build footer content based on mode
    let footer_spans = match app.mode() {
        AppMode::Help => vec![
            Span::styled("Press ", Style::default().fg(palette.muted)),
            Span::styled(
                "?",
                Style::default()
                    .fg(palette.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" to close help", Style::default().fg(palette.muted)),
        ],
        AppMode::FilterSelector => vec![
            Span::styled(
                "[FILTER SELECTOR] ",
                Style::default()
                    .fg(palette.info)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "↑↓/j/k",
                Style::default()
                    .fg(palette.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" navigate  ", Style::default().fg(palette.muted)),
            Span::styled(
                "Enter",
                Style::default()
                    .fg(palette.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" select  ", Style::default().fg(palette.muted)),
            Span::styled(
                "Esc",
                Style::default()
                    .fg(palette.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" cancel", Style::default().fg(palette.muted)),
        ],
        AppMode::FilePreview => vec![
            Span::styled(
                "[PREVIEW MODE] ",
                Style::default()
                    .fg(palette.info)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("j/k scroll  ", Style::default().fg(palette.muted)),
            Span::styled(
                "Esc",
                Style::default()
                    .fg(palette.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" close  ", Style::default().fg(palette.muted)),
            Span::styled(
                "o",
                Style::default()
                    .fg(palette.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" open in editor", Style::default().fg(palette.muted)),
        ],
        AppMode::Indexing | AppMode::Normal => {
            let mut spans = vec![];

            // Show mode indicator based on focus state
            match app.focus_state() {
                FocusState::Input => {
                    spans.push(Span::styled(
                        "[INPUT MODE] ",
                        Style::default()
                            .fg(palette.accent)
                            .add_modifier(Modifier::BOLD),
                    ));
                }
                FocusState::Filters => {
                    spans.push(Span::styled(
                        "[FILTERS MODE] ",
                        Style::default()
                            .fg(palette.info)
                            .add_modifier(Modifier::BOLD),
                    ));
                }
                FocusState::Results => {
                    spans.push(Span::styled(
                        "[NAVIGATE MODE] ",
                        Style::default()
                            .fg(palette.success)
                            .add_modifier(Modifier::BOLD),
                    ));
                }
            }

            // Show appropriate hint based on terminal capabilities
            let hint = app.capabilities().open_hint();
            spans.push(Span::styled(hint, Style::default().fg(palette.muted)));
            spans.push(Span::raw("  "));
            spans.push(Span::styled("? help", Style::default().fg(palette.muted)));

            spans
        }
    };

    let footer = Paragraph::new(Line::from(footer_spans))
        .style(Style::default().fg(palette.foreground).bg(Color::Black)); // Explicit background to prevent rendering artifacts

    f.render_widget(footer, area);
}
