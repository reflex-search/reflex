use crossterm::event::{KeyCode, MouseEvent, MouseEventKind};
use ratatui::{
    Frame,
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
};

use super::theme::ThemeManager;

/// Filter selector state
#[derive(Debug, Clone, PartialEq)]
pub enum FilterSelectorType {
    Language,
    Kind,
}

/// Filter selector modal for choosing language or symbol kind
pub struct FilterSelector {
    /// Type of selector (language or kind)
    selector_type: FilterSelectorType,
    /// Current selection index
    selected_index: usize,
    /// Available options
    options: Vec<String>,
    /// Last rendered modal area (for mouse detection)
    modal_area: Rect,
}

impl FilterSelector {
    /// Create a new language selector
    pub fn new_language() -> Self {
        let options = vec![
            "rust".to_string(),
            "python".to_string(),
            "javascript".to_string(),
            "typescript".to_string(),
            "vue".to_string(),
            "svelte".to_string(),
            "go".to_string(),
            "java".to_string(),
            "php".to_string(),
            "c".to_string(),
            "cpp".to_string(),
            "csharp".to_string(),
            "ruby".to_string(),
            "kotlin".to_string(),
            "zig".to_string(),
        ];

        Self {
            selector_type: FilterSelectorType::Language,
            selected_index: 0,
            options,
            modal_area: Rect::default(),
        }
    }

    /// Create a new kind selector
    pub fn new_kind() -> Self {
        let options = vec![
            "Function".to_string(),
            "Class".to_string(),
            "Struct".to_string(),
            "Enum".to_string(),
            "Interface".to_string(),
            "Trait".to_string(),
            "Constant".to_string(),
            "Variable".to_string(),
            "Method".to_string(),
            "Module".to_string(),
            "Namespace".to_string(),
            "Type".to_string(),
            "Macro".to_string(),
            "Property".to_string(),
            "Event".to_string(),
            "Import".to_string(),
            "Export".to_string(),
            "Attribute".to_string(),
        ];

        Self {
            selector_type: FilterSelectorType::Kind,
            selected_index: 0,
            options,
            modal_area: Rect::default(),
        }
    }

    /// Move selection up
    pub fn prev(&mut self) {
        if self.selected_index > 0 {
            self.selected_index -= 1;
        } else {
            // Wrap to bottom
            self.selected_index = self.options.len() - 1;
        }
    }

    /// Move selection down
    pub fn next(&mut self) {
        if self.selected_index < self.options.len() - 1 {
            self.selected_index += 1;
        } else {
            // Wrap to top
            self.selected_index = 0;
        }
    }

    /// Get the currently selected option
    pub fn selected(&self) -> Option<String> {
        self.options.get(self.selected_index).cloned()
    }

    /// Handle key input
    /// Returns Some(selection) if Enter was pressed, None otherwise
    pub fn handle_key(&mut self, key: KeyCode) -> Option<String> {
        match key {
            KeyCode::Up | KeyCode::Char('k') => {
                self.prev();
                None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.next();
                None
            }
            KeyCode::Enter => self.selected(),
            _ => None,
        }
    }

    /// Handle mouse input
    /// Returns Some(selection) if an item was clicked, None otherwise
    pub fn handle_mouse(&mut self, event: MouseEvent) -> Option<String> {
        match event.kind {
            MouseEventKind::Down(_) => {
                // Check if click is within modal area
                let col = event.column;
                let row = event.row;

                if col >= self.modal_area.x
                    && col < self.modal_area.x + self.modal_area.width
                    && row >= self.modal_area.y
                    && row < self.modal_area.y + self.modal_area.height
                {
                    // Click is within modal
                    // Calculate which option was clicked (accounting for border and title)
                    let relative_row = row.saturating_sub(self.modal_area.y + 1) as usize; // +1 for top border

                    if relative_row > 0 && relative_row <= self.options.len() {
                        // Clicked on an option
                        let option_index = relative_row - 1; // -1 because first row is title
                        if option_index < self.options.len() {
                            self.selected_index = option_index;
                            return self.selected(); // Double-click behavior: select immediately
                        }
                    }
                }
                None
            }
            MouseEventKind::ScrollDown => {
                self.next();
                None
            }
            MouseEventKind::ScrollUp => {
                self.prev();
                None
            }
            _ => None,
        }
    }

    /// Render the selector modal
    pub fn render(&mut self, f: &mut Frame, area: Rect, theme: &ThemeManager) {
        let palette = &theme.palette;

        // Create centered modal
        let modal_width = 40.min(area.width.saturating_sub(4));
        let modal_height =
            (self.options.len() + 4).min(area.height.saturating_sub(4) as usize) as u16;
        let modal_x = (area.width.saturating_sub(modal_width)) / 2;
        let modal_y = (area.height.saturating_sub(modal_height)) / 2;
        let modal_area = Rect::new(
            area.x + modal_x,
            area.y + modal_y,
            modal_width,
            modal_height,
        );

        // Store modal area for mouse detection
        self.modal_area = modal_area;

        // Render background (dimmed)
        let background = Block::default().style(Style::default().bg(Color::Black));
        f.render_widget(background, area);

        // Create modal title
        let title = match self.selector_type {
            FilterSelectorType::Language => " Select Language ",
            FilterSelectorType::Kind => " Select Symbol Kind ",
        };

        // Create list items
        let items: Vec<ListItem> = self
            .options
            .iter()
            .enumerate()
            .map(|(idx, option)| {
                let is_selected = idx == self.selected_index;
                let content = if is_selected {
                    format!("▶ {}", option)
                } else {
                    format!("  {}", option)
                };

                if is_selected {
                    ListItem::new(content).style(
                        Style::default()
                            .fg(Color::Black)
                            .bg(palette.highlight)
                            .add_modifier(Modifier::BOLD),
                    )
                } else {
                    ListItem::new(content).style(Style::default().fg(palette.foreground))
                }
            })
            .collect();

        // Create list widget
        let list = List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(title)
                .border_style(
                    Style::default()
                        .fg(palette.accent)
                        .add_modifier(Modifier::BOLD),
                )
                .style(Style::default().bg(Color::Rgb(25, 25, 30))),
        );

        // Render the modal
        f.render_widget(list, modal_area);

        // Render footer hint
        let footer_area = Rect::new(
            modal_area.x,
            modal_area.y + modal_area.height,
            modal_area.width,
            1,
        );
        let footer = Paragraph::new(Line::from(vec![
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
        ]))
        .alignment(Alignment::Center);
        f.render_widget(footer, footer_area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_selector_creation() {
        let selector = FilterSelector::new_language();
        assert_eq!(selector.selector_type, FilterSelectorType::Language);
        assert_eq!(selector.selected_index, 0);
        assert!(selector.options.len() > 0);
    }

    #[test]
    fn test_kind_selector_creation() {
        let selector = FilterSelector::new_kind();
        assert_eq!(selector.selector_type, FilterSelectorType::Kind);
        assert_eq!(selector.selected_index, 0);
        assert!(selector.options.len() > 0);
    }

    #[test]
    fn test_navigation() {
        let mut selector = FilterSelector::new_language();
        let initial = selector.selected_index;

        selector.next();
        assert_eq!(selector.selected_index, initial + 1);

        selector.prev();
        assert_eq!(selector.selected_index, initial);
    }

    #[test]
    fn test_wrap_navigation() {
        let mut selector = FilterSelector::new_language();
        let max_idx = selector.options.len() - 1;

        // At top, prev wraps to bottom
        selector.prev();
        assert_eq!(selector.selected_index, max_idx);

        // At bottom, next wraps to top
        selector.next();
        assert_eq!(selector.selected_index, 0);
    }

    #[test]
    fn test_selection() {
        let selector = FilterSelector::new_language();
        let selected = selector.selected();
        assert!(selected.is_some());
        assert_eq!(selected.unwrap(), "rust");
    }
}
