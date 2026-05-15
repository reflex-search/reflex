use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// Input field state for text entry
#[derive(Debug, Clone)]
pub struct InputField {
    /// Current input text
    value: String,
    /// Cursor position (byte index)
    cursor: usize,
}

impl InputField {
    pub fn new() -> Self {
        Self {
            value: String::new(),
            cursor: 0,
        }
    }

    /// Get the current value
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Get the cursor position
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Set the value and move cursor to end
    pub fn set_value(&mut self, value: String) {
        self.cursor = value.len();
        self.value = value;
    }

    /// Set cursor position (clamped to valid range)
    pub fn set_cursor(&mut self, position: usize) {
        self.cursor = position.min(self.value.len());
        // Ensure we're on a character boundary
        while self.cursor > 0 && !self.value.is_char_boundary(self.cursor) {
            self.cursor -= 1;
        }
    }

    /// Clear the input
    pub fn clear(&mut self) {
        self.value.clear();
        self.cursor = 0;
    }

    /// Handle a key event and return whether the input changed
    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Char(c) => {
                self.insert_char(c);
                true
            }
            KeyCode::Backspace => {
                self.backspace();
                true
            }
            KeyCode::Delete => {
                self.delete();
                true
            }
            KeyCode::Left => {
                self.move_cursor_left();
                false
            }
            KeyCode::Right => {
                self.move_cursor_right();
                false
            }
            KeyCode::Home => {
                self.cursor = 0;
                false
            }
            KeyCode::End => {
                self.cursor = self.value.len();
                false
            }
            _ => false,
        }
    }

    fn insert_char(&mut self, c: char) {
        self.value.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    fn backspace(&mut self) {
        if self.cursor > 0 {
            let prev_char_start = self.prev_char_boundary(self.cursor);
            self.value.remove(prev_char_start);
            self.cursor = prev_char_start;
        }
    }

    fn delete(&mut self) {
        if self.cursor < self.value.len() {
            self.value.remove(self.cursor);
        }
    }

    fn move_cursor_left(&mut self) {
        if self.cursor > 0 {
            self.cursor = self.prev_char_boundary(self.cursor);
        }
    }

    fn move_cursor_right(&mut self) {
        if self.cursor < self.value.len() {
            self.cursor = self.next_char_boundary(self.cursor);
        }
    }

    fn prev_char_boundary(&self, pos: usize) -> usize {
        let mut new_pos = pos.saturating_sub(1);
        while new_pos > 0 && !self.value.is_char_boundary(new_pos) {
            new_pos -= 1;
        }
        new_pos
    }

    fn next_char_boundary(&self, pos: usize) -> usize {
        let mut new_pos = pos + 1;
        while new_pos < self.value.len() && !self.value.is_char_boundary(new_pos) {
            new_pos += 1;
        }
        new_pos.min(self.value.len())
    }

    /// Get the visual cursor position (accounting for multi-byte chars)
    pub fn visual_cursor(&self) -> usize {
        self.value[..self.cursor].chars().count()
    }
}

/// Application-wide key command
#[derive(Debug, Clone, PartialEq)]
pub enum KeyCommand {
    // Navigation
    NextResult,
    PrevResult,
    PageDown,
    PageUp,
    First,
    Last,
    ScrollDown,
    ScrollUp,

    // Input focus
    FocusInput,
    UnfocusInput,

    // Filters
    ToggleSymbols,
    ToggleRegex,
    PromptLanguage,
    PromptKind,
    PromptGlob,
    PromptExclude,
    ToggleExpand,
    ToggleContains,
    ClearLanguage,
    ClearKind,

    // Actions
    OpenInEditor,
    Reindex,
    ClearAndReindex,
    ShowHelp,
    Quit,

    // History
    HistoryPrev,
    HistoryNext,

    // Char input (for text input mode)
    CharInput(char),
    Backspace,
    Delete,

    // Ignore
    None,
}

impl KeyCommand {
    /// Parse a key event into a command
    pub fn from_key(key: KeyEvent, input_focused: bool) -> Self {
        // Handle Ctrl+C always
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return Self::Quit;
        }

        // If input is focused, most keys go to the input field
        if input_focused {
            return match key.code {
                KeyCode::Esc => Self::UnfocusInput,
                KeyCode::Enter => Self::UnfocusInput,
                KeyCode::Char(c) if key.modifiers.contains(KeyModifiers::CONTROL) => match c {
                    'p' => Self::HistoryPrev,
                    'n' => Self::HistoryNext,
                    _ => Self::None,
                },
                _ => Self::None, // Let InputField handle it
            };
        }

        // Global shortcuts when input not focused
        match (key.code, key.modifiers) {
            // Navigation (j/k conflict resolved - j for next, arrow keys work)
            (KeyCode::Char('j'), KeyModifiers::NONE) | (KeyCode::Down, _) => Self::NextResult,
            (KeyCode::Up, _) => Self::PrevResult,
            (KeyCode::PageDown, _) => Self::PageDown,
            (KeyCode::PageUp, _) => Self::PageUp,
            (KeyCode::Home, _) => Self::First,
            (KeyCode::Char('g'), KeyModifiers::SHIFT) => Self::First, // G for first (like vim gg)
            (KeyCode::End, _) | (KeyCode::Char('G'), KeyModifiers::SHIFT) => Self::Last,

            // Input focus
            (KeyCode::Char('/'), KeyModifiers::NONE) => Self::FocusInput,
            (KeyCode::Esc, _) => Self::UnfocusInput,

            // Filters - need specific ordering to avoid conflicts
            (KeyCode::Char('l'), KeyModifiers::CONTROL) => Self::ClearLanguage,
            (KeyCode::Char('k'), KeyModifiers::CONTROL) => Self::ClearKind,
            (KeyCode::Char('s'), KeyModifiers::NONE) => Self::ToggleSymbols,
            (KeyCode::Char('r'), KeyModifiers::NONE) => Self::ToggleRegex,
            (KeyCode::Char('l'), KeyModifiers::NONE) => Self::PromptLanguage,
            (KeyCode::Char('k'), KeyModifiers::NONE) => Self::PromptKind,
            (KeyCode::Char('g'), KeyModifiers::NONE) => Self::PromptGlob,
            (KeyCode::Char('x'), KeyModifiers::NONE) => Self::PromptExclude,
            (KeyCode::Char('e'), KeyModifiers::NONE) => Self::ToggleExpand,
            (KeyCode::Char('c'), KeyModifiers::NONE) => Self::ToggleContains,

            // Actions
            (KeyCode::Char('o'), KeyModifiers::NONE) | (KeyCode::Enter, _) => Self::OpenInEditor,
            (KeyCode::Char('i'), KeyModifiers::NONE) => Self::Reindex,
            (KeyCode::Char('I'), KeyModifiers::SHIFT) => Self::ClearAndReindex,
            (KeyCode::Char('?'), KeyModifiers::NONE) => Self::ShowHelp,
            (KeyCode::Char('q'), KeyModifiers::NONE) => Self::Quit,

            // History (Ctrl+P/N)
            (KeyCode::Char('p'), KeyModifiers::CONTROL) => Self::HistoryPrev,
            (KeyCode::Char('n'), KeyModifiers::CONTROL) => Self::HistoryNext,

            _ => Self::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_field_basic() {
        let mut input = InputField::new();
        assert_eq!(input.value(), "");
        assert_eq!(input.cursor(), 0);

        input.set_value("hello".to_string());
        assert_eq!(input.value(), "hello");
        assert_eq!(input.cursor(), 5);

        input.clear();
        assert_eq!(input.value(), "");
        assert_eq!(input.cursor(), 0);
    }

    #[test]
    fn test_input_field_char_insertion() {
        let mut input = InputField::new();

        let key = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE);
        input.handle_key(key);
        assert_eq!(input.value(), "a");

        let key = KeyEvent::new(KeyCode::Char('b'), KeyModifiers::NONE);
        input.handle_key(key);
        assert_eq!(input.value(), "ab");
    }

    #[test]
    fn test_input_field_backspace() {
        let mut input = InputField::new();
        input.set_value("hello".to_string());

        let key = KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE);
        input.handle_key(key);
        assert_eq!(input.value(), "hell");
    }

    #[test]
    fn test_input_field_cursor_movement() {
        let mut input = InputField::new();
        input.set_value("hello".to_string());

        let key = KeyEvent::new(KeyCode::Home, KeyModifiers::NONE);
        input.handle_key(key);
        assert_eq!(input.cursor(), 0);

        let key = KeyEvent::new(KeyCode::End, KeyModifiers::NONE);
        input.handle_key(key);
        assert_eq!(input.cursor(), 5);
    }

    #[test]
    fn test_key_command_parsing() {
        let key = KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE);
        assert_eq!(KeyCommand::from_key(key, false), KeyCommand::NextResult);

        let key = KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE);
        assert_eq!(KeyCommand::from_key(key, false), KeyCommand::Quit);

        let key = KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE);
        assert_eq!(KeyCommand::from_key(key, false), KeyCommand::FocusInput);
    }
}
