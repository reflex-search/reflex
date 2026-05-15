use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;

/// Query history manager with persistence
#[derive(Debug, Clone)]
pub struct QueryHistory {
    /// Historical queries (newest first)
    queries: VecDeque<HistoricalQuery>,
    /// Maximum number of queries to store
    max_size: usize,
    /// Current position in history (for Ctrl+P/N navigation)
    cursor: Option<usize>,
}

/// A single historical query entry
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HistoricalQuery {
    /// The search pattern
    pub pattern: String,
    /// When this query was made
    pub timestamp: String,
    /// Active filters
    pub filters: QueryFilters,
}

/// Query filters that can be applied
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct QueryFilters {
    /// Symbols-only mode
    pub symbols_mode: bool,
    /// Regex mode
    pub regex_mode: bool,
    /// Language filter
    pub language: Option<String>,
    /// Symbol kind filter
    pub kind: Option<String>,
    /// Glob patterns (include files matching)
    pub glob_patterns: Vec<String>,
    /// Exclude patterns (exclude files matching)
    pub exclude_patterns: Vec<String>,
    /// Expand mode (show full symbol definitions)
    pub expand: bool,
    /// Contains mode (substring matching)
    pub contains: bool,
}

impl QueryHistory {
    /// Create a new query history
    pub fn new(max_size: usize) -> Self {
        Self {
            queries: VecDeque::new(),
            max_size,
            cursor: None,
        }
    }

    /// Load history from disk
    pub fn load() -> Result<Self> {
        let path = Self::history_path()?;

        if !path.exists() {
            return Ok(Self::new(1000));
        }

        let content = fs::read_to_string(&path)?;
        let queries: Vec<HistoricalQuery> = serde_json::from_str(&content)?;

        Ok(Self {
            queries: queries.into(),
            max_size: 1000,
            cursor: None,
        })
    }

    /// Save history to disk
    pub fn save(&self) -> Result<()> {
        let path = Self::history_path()?;

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let queries: Vec<_> = self.queries.iter().cloned().collect();
        let content = serde_json::to_string_pretty(&queries)?;
        fs::write(&path, content)?;

        Ok(())
    }

    /// Add a new query to history
    pub fn add(&mut self, pattern: String, filters: QueryFilters) {
        // Don't add empty patterns
        if pattern.trim().is_empty() {
            return;
        }

        let query = HistoricalQuery {
            pattern,
            timestamp: chrono::Utc::now().to_rfc3339(),
            filters,
        };

        // Remove duplicate if it exists
        self.queries
            .retain(|q| q.pattern != query.pattern || q.filters != query.filters);

        // Add to front
        self.queries.push_front(query);

        // Trim to max size
        while self.queries.len() > self.max_size {
            self.queries.pop_back();
        }

        // Reset cursor
        self.cursor = None;
    }

    /// Get the previous query in history (Ctrl+P)
    pub fn prev(&mut self) -> Option<&HistoricalQuery> {
        if self.queries.is_empty() {
            return None;
        }

        let new_cursor = match self.cursor {
            None => 0,
            Some(pos) => {
                if pos + 1 < self.queries.len() {
                    pos + 1
                } else {
                    pos
                }
            }
        };

        self.cursor = Some(new_cursor);
        self.queries.get(new_cursor)
    }

    /// Get the next query in history (Ctrl+N)
    pub fn next(&mut self) -> Option<&HistoricalQuery> {
        match self.cursor {
            None => None,
            Some(0) => {
                self.cursor = None;
                None
            }
            Some(pos) => {
                let new_cursor = pos - 1;
                self.cursor = Some(new_cursor);
                self.queries.get(new_cursor)
            }
        }
    }

    /// Reset cursor to start of history
    pub fn reset_cursor(&mut self) {
        self.cursor = None;
    }

    /// Get the number of queries in history
    pub fn len(&self) -> usize {
        self.queries.len()
    }

    /// Check if history is empty
    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }

    /// Get the path to the history file
    fn history_path() -> Result<PathBuf> {
        let home = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;
        Ok(home.join(".reflex").join("interactive_history.json"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_history_creation() {
        let history = QueryHistory::new(100);
        assert_eq!(history.len(), 0);
        assert!(history.is_empty());
    }

    #[test]
    fn test_add_query() {
        let mut history = QueryHistory::new(100);

        history.add("test".to_string(), QueryFilters::default());
        assert_eq!(history.len(), 1);

        history.add("another".to_string(), QueryFilters::default());
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_deduplication() {
        let mut history = QueryHistory::new(100);

        history.add("test".to_string(), QueryFilters::default());
        history.add("test".to_string(), QueryFilters::default());

        // Should only have one entry
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_max_size() {
        let mut history = QueryHistory::new(3);

        history.add("one".to_string(), QueryFilters::default());
        history.add("two".to_string(), QueryFilters::default());
        history.add("three".to_string(), QueryFilters::default());
        history.add("four".to_string(), QueryFilters::default());

        // Should only keep 3 most recent
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn test_navigation() {
        let mut history = QueryHistory::new(100);

        history.add("one".to_string(), QueryFilters::default());
        history.add("two".to_string(), QueryFilters::default());
        history.add("three".to_string(), QueryFilters::default());

        // Navigate backward
        assert_eq!(history.prev().unwrap().pattern, "three");
        assert_eq!(history.prev().unwrap().pattern, "two");
        assert_eq!(history.prev().unwrap().pattern, "one");

        // Navigate forward
        assert_eq!(history.next().unwrap().pattern, "two");
        assert_eq!(history.next().unwrap().pattern, "three");
        assert!(history.next().is_none());
    }
}
