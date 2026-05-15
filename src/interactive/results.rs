use crate::models::SearchResult;
use std::cell::Cell;

/// Result list manager with navigation and display state
#[derive(Debug)]
pub struct ResultList {
    /// All search results
    results: Vec<SearchResult>,
    /// Currently selected index
    selected_index: usize,
    /// Scroll offset for displaying results
    scroll_offset: usize,
    /// Maximum number of results to display
    max_results: usize,
    /// Last visible height (for automatic scroll updates)
    /// Uses Cell for interior mutability so it can be updated during rendering
    last_visible_height: Cell<usize>,
}

impl ResultList {
    pub fn new(max_results: usize) -> Self {
        Self {
            results: Vec::new(),
            selected_index: 0,
            scroll_offset: 0,
            max_results,
            last_visible_height: Cell::new(20), // Default estimate
        }
    }

    /// Update the result list with new results
    pub fn set_results(&mut self, results: Vec<SearchResult>) {
        // Limit to max_results
        self.results = results.into_iter().take(self.max_results).collect();
        self.selected_index = 0;
        self.scroll_offset = 0;
    }

    /// Get all results
    pub fn results(&self) -> &[SearchResult] {
        &self.results
    }

    /// Get the currently selected result
    pub fn selected(&self) -> Option<&SearchResult> {
        self.results.get(self.selected_index)
    }

    /// Get the selected index
    pub fn selected_index(&self) -> usize {
        self.selected_index
    }

    /// Get the scroll offset
    pub fn scroll_offset(&self) -> usize {
        self.scroll_offset
    }

    /// Get the number of results
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Check if there are no results
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Move selection to next result
    pub fn next(&mut self) {
        if self.results.is_empty() {
            return;
        }

        if self.selected_index < self.results.len() - 1 {
            self.selected_index += 1;
            self.update_scroll(self.last_visible_height.get());
        }
    }

    /// Move selection to previous result
    pub fn prev(&mut self) {
        if self.selected_index > 0 {
            self.selected_index -= 1;
            self.update_scroll(self.last_visible_height.get());
        }
    }

    /// Jump down by n results
    pub fn jump_down(&mut self, n: usize) {
        if self.results.is_empty() {
            return;
        }

        self.selected_index = (self.selected_index + n).min(self.results.len() - 1);
        self.update_scroll(self.last_visible_height.get());
    }

    /// Jump up by n results
    pub fn jump_up(&mut self, n: usize) {
        self.selected_index = self.selected_index.saturating_sub(n);
        self.update_scroll(self.last_visible_height.get());
    }

    /// Move to first result
    pub fn first(&mut self) {
        self.selected_index = 0;
        self.update_scroll(self.last_visible_height.get());
    }

    /// Move to last result
    pub fn last(&mut self) {
        if !self.results.is_empty() {
            self.selected_index = self.results.len() - 1;
            self.update_scroll(self.last_visible_height.get());
        }
    }

    /// Select a specific index (for mouse clicks)
    pub fn select(&mut self, index: usize) {
        if index < self.results.len() {
            self.selected_index = index;
            self.update_scroll(self.last_visible_height.get());
        }
    }

    /// Update scroll offset to keep selected item visible
    /// Returns true if scroll offset changed
    pub fn update_scroll(&mut self, visible_height: usize) -> bool {
        if self.results.is_empty() || visible_height == 0 {
            return false;
        }

        let old_offset = self.scroll_offset;

        // If selected is above visible area, scroll up
        if self.selected_index < self.scroll_offset {
            self.scroll_offset = self.selected_index;
        }

        // If selected is below visible area, scroll down
        if self.selected_index >= self.scroll_offset + visible_height {
            self.scroll_offset = self.selected_index - visible_height + 1;
        }

        old_offset != self.scroll_offset
    }

    /// Set the visible height (called during rendering)
    /// Uses interior mutability to allow updates during immutable rendering
    pub fn set_visible_height(&self, height: usize) {
        self.last_visible_height.set(height);
    }

    /// Get the visible results for rendering
    pub fn visible_results(&self, height: usize) -> &[SearchResult] {
        let start = self.scroll_offset;
        let end = (start + height).min(self.results.len());
        &self.results[start..end]
    }

    /// Clear all results
    pub fn clear(&mut self) {
        self.results.clear();
        self.selected_index = 0;
        self.scroll_offset = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::SearchResult;

    fn make_result(file: &str, line: usize) -> SearchResult {
        use crate::models::{Language, Span, SymbolKind};
        SearchResult {
            path: file.to_string(),
            lang: Language::Rust,
            kind: SymbolKind::Function,
            symbol: Some("test".to_string()),
            span: Span {
                start_line: line,
                end_line: line,
            },
            preview: "test".to_string(),
            dependencies: None,
        }
    }

    #[test]
    fn test_result_list_creation() {
        let list = ResultList::new(500);
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());
    }

    #[test]
    fn test_set_results() {
        let mut list = ResultList::new(500);
        let results = vec![
            make_result("a.rs", 1),
            make_result("b.rs", 2),
            make_result("c.rs", 3),
        ];

        list.set_results(results);
        assert_eq!(list.len(), 3);
        assert_eq!(list.selected_index(), 0);
    }

    #[test]
    fn test_navigation() {
        let mut list = ResultList::new(500);
        let results = vec![
            make_result("a.rs", 1),
            make_result("b.rs", 2),
            make_result("c.rs", 3),
        ];

        list.set_results(results);

        assert_eq!(list.selected_index(), 0);

        list.next();
        assert_eq!(list.selected_index(), 1);

        list.next();
        assert_eq!(list.selected_index(), 2);

        list.prev();
        assert_eq!(list.selected_index(), 1);

        list.first();
        assert_eq!(list.selected_index(), 0);

        list.last();
        assert_eq!(list.selected_index(), 2);
    }

    #[test]
    fn test_max_results() {
        let mut list = ResultList::new(2);
        let results = vec![
            make_result("a.rs", 1),
            make_result("b.rs", 2),
            make_result("c.rs", 3),
        ];

        list.set_results(results);
        assert_eq!(list.len(), 2); // Should only keep first 2
    }

    #[test]
    fn test_scroll_update() {
        let mut list = ResultList::new(500);
        let results = (0..20).map(|i| make_result("test.rs", i)).collect();
        list.set_results(results);

        // With visible height of 10
        list.select(15);
        let scrolled = list.update_scroll(10);
        assert!(scrolled);
        assert!(list.scroll_offset() > 0);
    }
}
