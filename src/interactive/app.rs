use anyhow::Result;
use crossterm::event::{self, Event, KeyEvent, MouseEvent};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use crate::cache::CacheManager;
use crate::indexer::Indexer;
use crate::models::{IndexConfig, SearchResult};
use crate::query::{QueryEngine, QueryFilter};

use super::effects::EffectManager;
use super::history::{QueryFilters, QueryHistory};
use super::input::{InputField, KeyCommand};
use super::mouse::{MouseAction, MouseState};
use super::results::ResultList;
use super::terminal::TerminalCapabilities;
use super::theme::ThemeManager;
use super::ui;

/// Main application state for interactive mode
pub struct InteractiveApp {
    /// Query input field
    input: InputField,
    /// Search results
    results: ResultList,
    /// Query history
    history: QueryHistory,
    /// Current filter state
    filters: QueryFilters,
    /// Query engine
    engine: QueryEngine,
    /// Cache manager
    cache: CacheManager,
    /// Current application mode
    mode: AppMode,
    /// Terminal capabilities
    capabilities: TerminalCapabilities,
    /// Theme manager
    theme: ThemeManager,
    /// Effect manager for animations
    effects: EffectManager,
    /// Mouse state
    mouse: MouseState,
    /// Filter badge positions (for accurate mouse click detection)
    filter_badge_positions: super::mouse::FilterBadgePositions,
    /// Index status
    index_status: IndexStatusState,
    /// Whether to quit
    should_quit: bool,
    /// Current focus state
    focus_state: FocusState,
    /// Current working directory
    cwd: PathBuf,
    /// Error message to display (if any)
    error_message: Option<String>,
    /// Info message to display (if any)
    info_message: Option<String>,
    /// File preview content (when a result is expanded)
    preview_content: Option<FilePreview>,
    /// Whether a search is currently executing
    searching: bool,
    /// Channel receiver for async search results
    search_rx: Option<mpsc::Receiver<Result<crate::models::QueryResponse>>>,
    /// Whether indexing is currently in progress
    indexing: bool,
    /// Channel receiver for async indexing results
    index_rx: Option<mpsc::Receiver<Result<crate::models::IndexStats>>>,
    /// Channel receiver for indexing progress updates (current, total, status)
    index_progress_rx: Option<mpsc::Receiver<(usize, usize, String)>>,
    /// Indexing start time (for elapsed time display)
    indexing_start_time: Option<Instant>,
    /// Time when filters were last changed (for debounced auto-search)
    filter_change_time: Option<Instant>,
    /// Filter debounce duration in milliseconds
    filter_debounce_ms: u64,
    /// Current filter selector (if open)
    filter_selector: Option<super::filter_selector::FilterSelector>,
    /// Time when info message was shown (for auto-dismissal)
    info_message_time: Option<Instant>,
    /// Force full terminal clear on next render (to fix rendering artifacts)
    needs_full_clear: bool,
}

/// File preview state
#[derive(Debug, Clone)]
pub struct FilePreview {
    path: String,
    content: Vec<String>,
    center_line: usize,
    scroll_offset: usize,
    language: crate::models::Language,
}

/// Application mode
#[derive(Debug, Clone, PartialEq)]
pub enum AppMode {
    /// Normal browsing mode
    Normal,
    /// Help screen is showing
    Help,
    /// Indexing in progress
    Indexing,
    /// File preview is showing
    FilePreview,
    /// Filter selector is showing (language or kind)
    FilterSelector,
}

/// Focus state for Tab navigation
#[derive(Debug, Clone, PartialEq)]
pub enum FocusState {
    Input,
    Filters,
    Results,
}

/// Background symbol indexing state
#[derive(Debug, Clone)]
pub enum SymbolIndexingState {
    /// Symbol indexing is running
    Running { processed: usize, total: usize },
    /// Symbol indexing completed successfully
    Completed,
    /// Symbol indexing failed
    Failed,
    /// Symbol indexing not started yet
    NotStarted,
}

/// Index status state
#[derive(Debug, Clone)]
pub enum IndexStatusState {
    /// Index is ready
    Ready {
        file_count: usize,
        last_updated: String,
        symbol_status: SymbolIndexingState,
    },
    /// Index doesn't exist
    Missing,
    /// Index is stale (files changed)
    Stale {
        files_changed: usize,
        symbol_status: SymbolIndexingState,
    },
    /// Currently indexing
    Indexing {
        current: usize,
        total: usize,
        status: String,
        symbol_status: SymbolIndexingState,
    },
}

impl InteractiveApp {
    /// Create a new interactive application
    pub fn new() -> Result<Self> {
        let cwd = std::env::current_dir()?;
        let cache = CacheManager::new(&cwd);
        let cache2 = CacheManager::new(&cwd); // Create second instance for engine
        let engine = QueryEngine::new(cache2);
        let capabilities = TerminalCapabilities::detect();
        let theme = ThemeManager::detect();
        let history = QueryHistory::load().unwrap_or_else(|_| QueryHistory::new(1000));

        // Check index status
        let index_status = if cache.exists() {
            // Get actual file count from cache stats
            match cache.stats() {
                Ok(stats) => {
                    // Check background symbol indexer status
                    let symbol_status =
                        match crate::background_indexer::BackgroundIndexer::get_status(cache.path())
                        {
                            Ok(Some(status)) => match status.state {
                                crate::background_indexer::IndexerState::Running => {
                                    SymbolIndexingState::Running {
                                        processed: status.processed_files,
                                        total: status.total_files,
                                    }
                                }
                                crate::background_indexer::IndexerState::Completed => {
                                    SymbolIndexingState::Completed
                                }
                                crate::background_indexer::IndexerState::Failed => {
                                    SymbolIndexingState::Failed
                                }
                            },
                            _ => SymbolIndexingState::NotStarted,
                        };

                    IndexStatusState::Ready {
                        file_count: stats.total_files,
                        last_updated: stats.last_updated,
                        symbol_status,
                    }
                }
                Err(_) => IndexStatusState::Missing,
            }
        } else {
            IndexStatusState::Missing
        };

        Ok(Self {
            input: InputField::new(),
            results: ResultList::new(500),
            history,
            filters: QueryFilters::default(),
            engine,
            cache,
            mode: AppMode::Normal,
            capabilities,
            theme,
            effects: EffectManager::new(),
            mouse: MouseState::new(),
            filter_badge_positions: super::mouse::FilterBadgePositions::default(),
            index_status,
            should_quit: false,
            focus_state: FocusState::Input, // Start with input focused
            cwd,
            error_message: None,
            info_message: None,
            preview_content: None,
            searching: false,
            search_rx: None,
            indexing: false,
            index_rx: None,
            index_progress_rx: None,
            indexing_start_time: None,
            filter_change_time: None,
            filter_debounce_ms: 500, // 500ms
            filter_selector: None,
            info_message_time: None,
            needs_full_clear: false,
        })
    }

    /// Run the interactive event loop
    pub fn run(&mut self) -> Result<()> {
        // Show help on first launch (if history is empty)
        if self.history.is_empty() {
            self.mode = AppMode::Help;
        }

        // Setup terminal FIRST
        let mut terminal = Self::setup_terminal()?;

        // Check if we need to trigger indexing (will show modal in event loop)
        let needs_index = matches!(self.index_status, IndexStatusState::Missing);

        // Main event loop (will handle deferred indexing with modal)
        let result = self.event_loop(&mut terminal, needs_index);

        // Restore terminal
        Self::restore_terminal(terminal)?;

        // Save history on exit
        if let Err(e) = self.history.save() {
            eprintln!("Warning: Failed to save history: {}", e);
        }

        result
    }

    fn event_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
        needs_index: bool,
    ) -> Result<()> {
        let mut last_frame = Instant::now();
        let frame_duration = Duration::from_millis(16); // ~60 FPS
        let mut need_editor_open: Option<SearchResult> = None;
        let mut first_frame = true;

        while !self.should_quit {
            // Trigger indexing on first frame if needed (UI loads first, then shows modal)
            if first_frame && needs_index {
                self.trigger_index()?;
                first_frame = false;
            }

            // Get terminal size for event handling
            let terminal_size = terminal.size()?;

            // Render UI
            terminal.draw(|f| {
                // Force backend clear if needed (BEFORE rendering to reset ratatui's buffer)
                if self.needs_full_clear {
                    f.render_widget(ratatui::widgets::Clear, f.area());
                    self.needs_full_clear = false;
                }
                ui::render(f, self)
            })?;

            // Handle deferred editor opening (after rendering)
            if let Some(result) = need_editor_open.take() {
                self.open_in_editor_suspended(terminal, &result)?;
            }

            // Handle events (with timeout for smooth rendering)
            if event::poll(Duration::from_millis(50))? {
                match event::read()? {
                    Event::Key(key) => {
                        if let Some(result) = self.handle_key_event_with_editor(key)? {
                            need_editor_open = Some(result);
                        }
                    }
                    Event::Mouse(mouse) => {
                        self.handle_mouse_event(mouse, (terminal_size.width, terminal_size.height))
                    }
                    Event::Resize(_, _) => {
                        // Terminal resized, will redraw on next frame
                    }
                    _ => {}
                }
            }

            // Check for search results
            if let Some(ref rx) = self.search_rx {
                if let Ok(result) = rx.try_recv() {
                    // Search completed
                    match result {
                        Ok(response) => {
                            // Flatten grouped results to SearchResult vec for display
                            let flat_results = response
                                .results
                                .iter()
                                .flat_map(|file_group| {
                                    file_group.matches.iter().map(move |m| {
                                        crate::models::SearchResult {
                                            path: file_group.path.clone(),
                                            lang: crate::models::Language::Unknown,
                                            kind: m.kind.clone(),
                                            symbol: m.symbol.clone(),
                                            span: m.span.clone(),
                                            preview: m.preview.clone(),
                                            dependencies: file_group.dependencies.clone(),
                                        }
                                    })
                                })
                                .collect();

                            self.results.set_results(flat_results);
                            self.error_message = None;

                            // Add to history
                            let pattern = self.input.value().to_string();
                            self.history.add(pattern, self.filters.clone());

                            // Auto-move to results after search
                            // BUT: don't auto-focus if regex/contains filters are active during typing
                            // to prevent rendering corruption
                            let should_auto_focus =
                                !self.results.is_empty() && self.focus_state != FocusState::Input;
                            if should_auto_focus {
                                self.focus_state = FocusState::Results;
                            }
                        }
                        Err(e) => {
                            self.error_message = Some(format!("Search error: {}", e));
                            self.results.clear();
                        }
                    }
                    self.searching = false;
                    self.search_rx = None;
                }
            }

            // Check for debounced filter change (auto-search after 1.5s)
            if let Some(change_time) = self.filter_change_time {
                if change_time.elapsed() >= Duration::from_millis(self.filter_debounce_ms) {
                    // Debounce period elapsed, trigger search if input is not empty
                    if !self.input.value().trim().is_empty() && !self.searching {
                        let _ = self.execute_search();
                    }
                    self.filter_change_time = None;
                }
            }

            // Auto-clear info messages after 3 seconds
            if let Some(info_time) = self.info_message_time {
                if info_time.elapsed() >= Duration::from_secs(3) {
                    self.info_message = None;
                    self.info_message_time = None;
                }
            }

            // Check for indexing progress updates
            if let Some(ref rx) = self.index_progress_rx {
                if let Ok((current, total, status)) = rx.try_recv() {
                    // Update progress state (preserve symbol status)
                    let symbol_status = match &self.index_status {
                        IndexStatusState::Indexing { symbol_status, .. } => symbol_status.clone(),
                        IndexStatusState::Ready { symbol_status, .. } => symbol_status.clone(),
                        IndexStatusState::Stale { symbol_status, .. } => symbol_status.clone(),
                        IndexStatusState::Missing => SymbolIndexingState::NotStarted,
                    };
                    self.index_status = IndexStatusState::Indexing {
                        current,
                        total,
                        status,
                        symbol_status,
                    };
                }
            }

            // Check for indexing results
            if let Some(ref rx) = self.index_rx {
                if let Ok(result) = rx.try_recv() {
                    // Indexing completed
                    match result {
                        Ok(stats) => {
                            // Preserve or check symbol status
                            let symbol_status = match &self.index_status {
                                IndexStatusState::Indexing { symbol_status, .. } => {
                                    symbol_status.clone()
                                }
                                IndexStatusState::Ready { symbol_status, .. } => {
                                    symbol_status.clone()
                                }
                                IndexStatusState::Stale { symbol_status, .. } => {
                                    symbol_status.clone()
                                }
                                IndexStatusState::Missing => SymbolIndexingState::NotStarted,
                            };
                            self.index_status = IndexStatusState::Ready {
                                file_count: stats.total_files,
                                last_updated: "just now".to_string(),
                                symbol_status,
                            };
                            // Don't re-trigger search - keep current results
                        }
                        Err(e) => {
                            self.error_message = Some(format!("Index error: {}", e));
                        }
                    }
                    self.indexing = false;
                    self.indexing_start_time = None;
                    self.index_rx = None;
                    self.index_progress_rx = None;
                }
            }

            // Poll background symbol indexer status (every few frames to reduce overhead)
            if self.effects.frame() % 30 == 0 {
                // Every ~0.5s at 60fps
                log::trace!(
                    "Polling background symbol indexer status (frame {})",
                    self.effects.frame()
                );
                match crate::background_indexer::BackgroundIndexer::get_status(self.cache.path()) {
                    Ok(Some(bg_status)) => {
                        log::debug!(
                            "Background symbol indexer status: {:?} - {}/{} files",
                            bg_status.state,
                            bg_status.processed_files,
                            bg_status.total_files
                        );

                        let new_symbol_status = match bg_status.state {
                            crate::background_indexer::IndexerState::Running => {
                                log::debug!(
                                    "Symbol indexing is RUNNING: {}/{} ({}%)",
                                    bg_status.processed_files,
                                    bg_status.total_files,
                                    if bg_status.total_files > 0 {
                                        (bg_status.processed_files as f64
                                            / bg_status.total_files as f64
                                            * 100.0) as u32
                                    } else {
                                        0
                                    }
                                );
                                SymbolIndexingState::Running {
                                    processed: bg_status.processed_files,
                                    total: bg_status.total_files,
                                }
                            }
                            crate::background_indexer::IndexerState::Completed => {
                                log::debug!("Symbol indexing COMPLETED");
                                SymbolIndexingState::Completed
                            }
                            crate::background_indexer::IndexerState::Failed => {
                                log::warn!("Symbol indexing FAILED: {:?}", bg_status.error);
                                SymbolIndexingState::Failed
                            }
                        };

                        // Update symbol status in current index state
                        self.index_status = match &self.index_status {
                            IndexStatusState::Ready {
                                file_count,
                                last_updated,
                                ..
                            } => IndexStatusState::Ready {
                                file_count: *file_count,
                                last_updated: last_updated.clone(),
                                symbol_status: new_symbol_status,
                            },
                            IndexStatusState::Stale { files_changed, .. } => {
                                IndexStatusState::Stale {
                                    files_changed: *files_changed,
                                    symbol_status: new_symbol_status,
                                }
                            }
                            IndexStatusState::Indexing {
                                current,
                                total,
                                status,
                                ..
                            } => IndexStatusState::Indexing {
                                current: *current,
                                total: *total,
                                status: status.clone(),
                                symbol_status: new_symbol_status,
                            },
                            IndexStatusState::Missing => IndexStatusState::Missing,
                        };
                    }
                    Ok(None) => {
                        log::trace!("No background symbol indexer status file found");
                    }
                    Err(e) => {
                        log::warn!("Failed to read background symbol indexer status: {}", e);
                    }
                }
            }

            // Update effects
            let elapsed = last_frame.elapsed();
            self.effects.update(elapsed);
            last_frame = Instant::now();

            // Frame pacing
            std::thread::sleep(frame_duration.saturating_sub(elapsed));
        }

        Ok(())
    }

    fn handle_key_event_with_editor(&mut self, key: KeyEvent) -> Result<Option<SearchResult>> {
        // Handle filter selector mode first
        if self.mode == AppMode::FilterSelector {
            if let Some(ref mut selector) = self.filter_selector {
                if key.code == crossterm::event::KeyCode::Esc {
                    // Close selector without selection
                    self.mode = AppMode::Normal;
                    self.filter_selector = None;
                    return Ok(None);
                }

                if let Some(selection) = selector.handle_key(key.code) {
                    // We need to know which type of selector this is
                    // Let's check by seeing if selection is a valid language or kind
                    let selection_lower = selection.to_lowercase();
                    let is_language = matches!(
                        selection_lower.as_str(),
                        "rust"
                            | "python"
                            | "javascript"
                            | "typescript"
                            | "vue"
                            | "svelte"
                            | "go"
                            | "java"
                            | "php"
                            | "c"
                            | "cpp"
                            | "csharp"
                            | "ruby"
                            | "kotlin"
                            | "zig"
                    );

                    if is_language {
                        self.filters.language = Some(selection);
                    } else {
                        self.filters.kind = Some(selection);
                    }

                    self.mode = AppMode::Normal;
                    self.filter_selector = None;
                    self.cancel_ongoing_search(); // Cancel old search immediately
                    self.filter_change_time = Some(Instant::now());
                    self.info_message = None;
                }
                return Ok(None);
            }
        }

        // Handle Tab/Shift+Tab for focus cycling
        if key.code == crossterm::event::KeyCode::Tab {
            if key
                .modifiers
                .contains(crossterm::event::KeyModifiers::SHIFT)
            {
                self.focus_prev();
            } else {
                self.focus_next();
            }
            return Ok(None);
        }

        // Handle Escape - close preview or unfocus
        if key.code == crossterm::event::KeyCode::Esc {
            if self.mode == AppMode::FilePreview {
                self.mode = AppMode::Normal;
                self.preview_content = None;
                return Ok(None);
            }
            self.focus_state = FocusState::Results;
            return Ok(None);
        }

        // Handle Enter - different behavior based on focus
        if key.code == crossterm::event::KeyCode::Enter {
            match self.focus_state {
                FocusState::Input => {
                    // Execute search and move to results
                    if !self.input.value().trim().is_empty() {
                        self.execute_search()?;
                        self.focus_state = FocusState::Results;
                    }
                }
                FocusState::Results => {
                    // Expand file preview
                    if let Some(result) = self.results.selected().cloned() {
                        self.show_file_preview(&result)?;
                    }
                }
                _ => {}
            }
            return Ok(None);
        }

        let command = KeyCommand::from_key(key, self.focus_state == FocusState::Input);

        match command {
            KeyCommand::Quit => {
                self.should_quit = true;
                Ok(None)
            }

            KeyCommand::ShowHelp => {
                self.mode = if self.mode == AppMode::Help {
                    AppMode::Normal
                } else {
                    AppMode::Help
                };
                Ok(None)
            }

            KeyCommand::FocusInput => {
                self.focus_state = FocusState::Input;
                Ok(None)
            }

            KeyCommand::UnfocusInput => {
                self.focus_state = FocusState::Results;
                Ok(None)
            }

            KeyCommand::NextResult => {
                if self.mode == AppMode::FilePreview {
                    self.scroll_preview_down();
                } else {
                    self.results.next();
                }
                Ok(None)
            }

            KeyCommand::PrevResult => {
                if self.mode == AppMode::FilePreview {
                    self.scroll_preview_up();
                } else {
                    self.results.prev();
                }
                Ok(None)
            }

            KeyCommand::PageDown => {
                self.results.jump_down(10);
                Ok(None)
            }

            KeyCommand::PageUp => {
                self.results.jump_up(10);
                Ok(None)
            }

            KeyCommand::First => {
                self.results.first();
                Ok(None)
            }

            KeyCommand::Last => {
                self.results.last();
                Ok(None)
            }

            KeyCommand::ToggleSymbols => {
                self.filters.symbols_mode = !self.filters.symbols_mode;
                self.cancel_ongoing_search(); // Cancel old search immediately
                self.filter_change_time = Some(Instant::now());
                self.info_message = None;
                Ok(None)
            }

            KeyCommand::ToggleRegex => {
                self.filters.regex_mode = !self.filters.regex_mode;
                self.cancel_ongoing_search(); // Cancel old search immediately
                self.filter_change_time = Some(Instant::now());
                self.info_message = None;
                self.needs_full_clear = true; // Force full redraw to prevent artifacts
                Ok(None)
            }

            KeyCommand::PromptLanguage => {
                self.filter_selector = Some(super::filter_selector::FilterSelector::new_language());
                self.mode = AppMode::FilterSelector;
                Ok(None)
            }

            KeyCommand::PromptKind => {
                self.filter_selector = Some(super::filter_selector::FilterSelector::new_kind());
                self.mode = AppMode::FilterSelector;
                Ok(None)
            }

            KeyCommand::PromptGlob => {
                // For now, set a simple info message. In future, could add text input modal
                self.info_message =
                    Some("Glob patterns: Use CLI for now (--glob flag)".to_string());
                self.info_message_time = Some(Instant::now());
                Ok(None)
            }

            KeyCommand::PromptExclude => {
                // For now, set a simple info message. In future, could add text input modal
                self.info_message =
                    Some("Exclude patterns: Use CLI for now (--exclude flag)".to_string());
                self.info_message_time = Some(Instant::now());
                Ok(None)
            }

            KeyCommand::ToggleExpand => {
                self.filters.expand = !self.filters.expand;
                self.cancel_ongoing_search(); // Cancel old search immediately
                self.filter_change_time = Some(Instant::now());
                self.info_message = None;
                Ok(None)
            }

            KeyCommand::ToggleContains => {
                self.filters.contains = !self.filters.contains;
                self.cancel_ongoing_search(); // Cancel old search immediately
                self.filter_change_time = Some(Instant::now());
                self.info_message = None;
                self.needs_full_clear = true; // Force full redraw to prevent artifacts
                Ok(None)
            }

            KeyCommand::ClearLanguage => {
                self.filters.language = None;
                self.cancel_ongoing_search(); // Cancel old search immediately
                self.filter_change_time = Some(Instant::now());
                self.info_message = None;
                Ok(None)
            }

            KeyCommand::ClearKind => {
                self.filters.kind = None;
                self.cancel_ongoing_search(); // Cancel old search immediately
                self.filter_change_time = Some(Instant::now());
                self.info_message = None;
                Ok(None)
            }

            KeyCommand::OpenInEditor => {
                // Return the result to open in editor
                Ok(self.results.selected().cloned())
            }

            KeyCommand::Reindex => {
                self.trigger_index()?;
                Ok(None)
            }

            KeyCommand::ClearAndReindex => {
                self.trigger_clear_and_reindex()?;
                Ok(None)
            }

            KeyCommand::HistoryPrev => {
                if let Some(query) = self.history.prev() {
                    self.input.set_value(query.pattern.clone());
                    self.filters = query.filters.clone();
                }
                Ok(None)
            }

            KeyCommand::HistoryNext => {
                if let Some(query) = self.history.next() {
                    self.input.set_value(query.pattern.clone());
                    self.filters = query.filters.clone();
                } else {
                    // At the end of history, clear input
                    self.input.clear();
                    self.results.clear();
                }
                Ok(None)
            }

            KeyCommand::None => {
                // If input is focused, handle the key for text input
                if self.focus_state == FocusState::Input {
                    self.input.handle_key(key);
                }
                Ok(None)
            }

            _ => Ok(None),
        }
    }

    fn focus_next(&mut self) {
        self.focus_state = match self.focus_state {
            FocusState::Input => FocusState::Filters,
            FocusState::Filters => FocusState::Results,
            FocusState::Results => FocusState::Input,
        };
    }

    fn focus_prev(&mut self) {
        self.focus_state = match self.focus_state {
            FocusState::Input => FocusState::Results,
            FocusState::Filters => FocusState::Input,
            FocusState::Results => FocusState::Filters,
        };
    }

    fn show_file_preview(&mut self, result: &SearchResult) -> Result<()> {
        // Read file content
        let content = std::fs::read_to_string(&result.path)?;
        let lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();

        // Detect language from file extension
        let language = std::path::Path::new(&result.path)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| crate::models::Language::from_extension(ext))
            .unwrap_or(crate::models::Language::Unknown);

        self.preview_content = Some(FilePreview {
            path: result.path.clone(),
            content: lines,
            center_line: result.span.start_line,
            scroll_offset: result.span.start_line.saturating_sub(10),
            language,
        });

        self.mode = AppMode::FilePreview;
        Ok(())
    }

    fn scroll_preview_down(&mut self) {
        if let Some(ref mut preview) = self.preview_content {
            if preview.scroll_offset + 20 < preview.content.len() {
                preview.scroll_offset += 1;
            }
        }
    }

    fn scroll_preview_up(&mut self) {
        if let Some(ref mut preview) = self.preview_content {
            preview.scroll_offset = preview.scroll_offset.saturating_sub(1);
        }
    }

    fn handle_mouse_event(&mut self, mouse: MouseEvent, terminal_size: (u16, u16)) {
        // In filter selector mode, pass events to the selector
        if self.mode == AppMode::FilterSelector {
            if let Some(ref mut selector) = self.filter_selector {
                if let Some(selection) = selector.handle_mouse(mouse) {
                    // Selection was made
                    let selection_lower = selection.to_lowercase();
                    let is_language = matches!(
                        selection_lower.as_str(),
                        "rust"
                            | "python"
                            | "javascript"
                            | "typescript"
                            | "vue"
                            | "svelte"
                            | "go"
                            | "java"
                            | "php"
                            | "c"
                            | "cpp"
                            | "csharp"
                            | "ruby"
                            | "kotlin"
                            | "zig"
                    );

                    if is_language {
                        self.filters.language = Some(selection);
                    } else {
                        self.filters.kind = Some(selection);
                    }

                    self.mode = AppMode::Normal;
                    self.filter_selector = None;
                    self.cancel_ongoing_search(); // Cancel old search immediately
                    self.filter_change_time = Some(Instant::now());
                    self.info_message = None;
                }
            }
            return;
        }

        // In preview mode, handle scroll events for file content
        if self.mode == AppMode::FilePreview {
            match mouse.kind {
                crossterm::event::MouseEventKind::ScrollDown => {
                    for _ in 0..3 {
                        self.scroll_preview_down();
                    }
                }
                crossterm::event::MouseEventKind::ScrollUp => {
                    for _ in 0..3 {
                        self.scroll_preview_up();
                    }
                }
                crossterm::event::MouseEventKind::Down(_) => {
                    // Click anywhere to close preview
                    self.mode = AppMode::Normal;
                    self.preview_content = None;
                }
                _ => {}
            }
            return;
        }

        // Calculate UI regions based on terminal size
        // Layout: header(3) + filters(3) + results(rest) + footer(1)
        let input_area = ratatui::layout::Rect::new(0, 0, terminal_size.0, 3);
        let filters_area = ratatui::layout::Rect::new(0, 3, terminal_size.0, 3);
        let result_y = 6;
        let result_height = terminal_size.1.saturating_sub(7); // 6 from top + 1 from bottom
        let result_area = ratatui::layout::Rect::new(0, result_y, terminal_size.0, result_height);

        let action = self.mouse.handle_event(
            mouse,
            input_area,
            filters_area,
            result_area,
            &self.filter_badge_positions,
        );

        match action {
            MouseAction::FocusInput(cursor_pos) => {
                self.focus_state = FocusState::Input;
                // Clamp cursor position to input length
                let max_pos = self.input.value().len();
                let clamped_pos = cursor_pos.min(max_pos);
                self.input.set_cursor(clamped_pos);
            }
            MouseAction::ToggleSymbols => {
                self.filters.symbols_mode = !self.filters.symbols_mode;
                self.cancel_ongoing_search(); // Cancel old search immediately
                self.filter_change_time = Some(Instant::now());
                self.info_message = None;
            }
            MouseAction::ToggleRegex => {
                self.filters.regex_mode = !self.filters.regex_mode;
                self.cancel_ongoing_search(); // Cancel old search immediately
                self.filter_change_time = Some(Instant::now());
                self.info_message = None;
                self.needs_full_clear = true; // Force full redraw to prevent artifacts
            }
            MouseAction::PromptLanguage => {
                self.cancel_ongoing_search(); // Cancel old search immediately
                self.filter_selector = Some(super::filter_selector::FilterSelector::new_language());
                self.mode = AppMode::FilterSelector;
            }
            MouseAction::PromptKind => {
                self.cancel_ongoing_search(); // Cancel old search immediately
                self.filter_selector = Some(super::filter_selector::FilterSelector::new_kind());
                self.mode = AppMode::FilterSelector;
            }
            MouseAction::ToggleExpand => {
                self.filters.expand = !self.filters.expand;
                self.cancel_ongoing_search(); // Cancel old search immediately
                self.filter_change_time = Some(Instant::now());
                self.info_message = None;
            }
            MouseAction::ToggleContains => {
                self.filters.contains = !self.filters.contains;
                self.cancel_ongoing_search(); // Cancel old search immediately
                self.filter_change_time = Some(Instant::now());
                self.info_message = None;
                self.needs_full_clear = true; // Force full redraw to prevent artifacts
            }
            MouseAction::SelectResult(line_index) => {
                // Convert line index to result index (results have variable heights)
                let result_index = self.line_index_to_result_index(line_index);
                self.results.select(result_index);
            }
            MouseAction::DoubleClick(line_index) => {
                // Convert line index to result index (results have variable heights)
                let result_index = self.line_index_to_result_index(line_index);
                self.results.select(result_index);
                if let Some(result) = self.results.selected().cloned() {
                    let _ = self.show_file_preview(&result);
                }
            }
            MouseAction::ScrollDown => {
                self.results.next();
            }
            MouseAction::ScrollUp => {
                self.results.prev();
            }
            MouseAction::TriggerIndex => {
                let _ = self.trigger_index();
            }
            MouseAction::ClearAndReindex => {
                let _ = self.trigger_clear_and_reindex();
            }
            _ => {}
        }
    }

    /// Convert a line index within the results area to a result index
    /// Accounts for variable-height results (symbol line + path line + preview lines)
    fn line_index_to_result_index(&self, line_index: usize) -> usize {
        let mut current_line = 0;
        let scroll_offset = self.results.scroll_offset();

        for (idx, result) in self
            .results
            .results()
            .iter()
            .enumerate()
            .skip(scroll_offset)
        {
            // Calculate lines for this result
            // Must match the MAX_PREVIEW_LINES limit used in ui.rs
            const MAX_PREVIEW_LINES: usize = 20;
            let has_symbol = !matches!(result.kind, crate::models::SymbolKind::Unknown(_))
                && result.symbol.is_some();
            let symbol_lines = if has_symbol { 1 } else { 0 };
            let path_lines = 1;
            let preview_lines = result.preview.lines().count().min(MAX_PREVIEW_LINES);
            let total_lines = symbol_lines + path_lines + preview_lines;

            // Check if click was in this result's range
            if line_index < current_line + total_lines {
                return idx;
            }

            current_line += total_lines;
        }

        // If we get here, click was beyond all results - return last result
        self.results.len().saturating_sub(1)
    }

    /// Cancel any ongoing search immediately
    /// This prevents race conditions when filters change while a search is running
    fn cancel_ongoing_search(&mut self) {
        self.searching = false;
        self.search_rx = None;
    }

    fn execute_search(&mut self) -> Result<()> {
        // Reset history cursor when executing a new search
        self.history.reset_cursor();

        let pattern = self.input.value();
        if pattern.trim().is_empty() {
            self.results.clear();
            self.searching = false;
            return Ok(());
        }

        // Parse language filter
        let language = self.filters.language.as_ref().and_then(|lang_str| {
            match lang_str.to_lowercase().as_str() {
                "rust" | "rs" => Some(crate::models::Language::Rust),
                "python" | "py" => Some(crate::models::Language::Python),
                "javascript" | "js" => Some(crate::models::Language::JavaScript),
                "typescript" | "ts" => Some(crate::models::Language::TypeScript),
                "vue" => Some(crate::models::Language::Vue),
                "svelte" => Some(crate::models::Language::Svelte),
                "go" => Some(crate::models::Language::Go),
                "java" => Some(crate::models::Language::Java),
                "php" => Some(crate::models::Language::PHP),
                "c" => Some(crate::models::Language::C),
                "cpp" | "c++" => Some(crate::models::Language::Cpp),
                "csharp" | "cs" | "c#" => Some(crate::models::Language::CSharp),
                "ruby" | "rb" => Some(crate::models::Language::Ruby),
                "kotlin" | "kt" => Some(crate::models::Language::Kotlin),
                "zig" => Some(crate::models::Language::Zig),
                _ => None,
            }
        });

        // Parse symbol kind filter
        let kind = self
            .filters
            .kind
            .as_ref()
            .and_then(|kind_str| kind_str.parse::<crate::models::SymbolKind>().ok());

        // Build query filter
        let filter = QueryFilter {
            language,
            kind,
            use_ast: false,
            use_regex: self.filters.regex_mode,
            limit: Some(500),
            symbols_mode: self.filters.symbols_mode,
            expand: self.filters.expand,
            file_pattern: None,
            exact: false, // Exact match is the default behavior
            use_contains: self.filters.contains,
            timeout_secs: 10,
            glob_patterns: self.filters.glob_patterns.clone(),
            exclude_patterns: self.filters.exclude_patterns.clone(),
            paths_only: false,
            offset: None,
            force: false,
            suppress_output: false,
            include_dependencies: false, // Interactive mode doesn't support dependencies yet
            ..Default::default()
        };

        // Spawn background thread for search
        let (tx, rx) = mpsc::channel();
        let pattern_owned = pattern.to_string();
        let cache = CacheManager::new(&self.cwd);
        let engine = QueryEngine::new(cache);

        std::thread::spawn(move || {
            let result = engine.search_with_metadata(&pattern_owned, filter);
            tx.send(result).ok();
        });

        self.searching = true;
        self.search_rx = Some(rx);

        Ok(())
    }

    fn trigger_index(&mut self) -> Result<()> {
        // Preserve symbol status when starting new index
        let symbol_status = match &self.index_status {
            IndexStatusState::Indexing { symbol_status, .. } => symbol_status.clone(),
            IndexStatusState::Ready { symbol_status, .. } => symbol_status.clone(),
            IndexStatusState::Stale { symbol_status, .. } => symbol_status.clone(),
            IndexStatusState::Missing => SymbolIndexingState::NotStarted,
        };

        self.index_status = IndexStatusState::Indexing {
            current: 0,
            total: 0,
            status: "Starting...".to_string(),
            symbol_status,
        };

        // Create channels for results and progress
        let (result_tx, result_rx) = mpsc::channel();
        let (progress_tx, progress_rx) = mpsc::channel();
        let cwd = self.cwd.clone();
        let _cache_path = self.cache.path().to_path_buf();

        // Spawn background thread for indexing with progress callback
        std::thread::spawn(move || {
            let config = IndexConfig::default();
            let cache = CacheManager::new(&cwd);
            let indexer = Indexer::new(cache, config);

            // Create progress callback that sends updates through channel
            let callback = Arc::new(move |current: usize, total: usize, status: String| {
                let _ = progress_tx.send((current, total, status));
            });

            let result = indexer.index_with_callback(&cwd, false, Some(callback));

            // If indexing succeeded, spawn background symbol indexer
            if result.is_ok() {
                log::debug!("Main indexing completed, spawning background symbol indexer");

                // Spawn detached background process for symbol indexing
                // (Same approach as CLI: use index-symbols-internal hidden command)
                let current_exe = std::env::current_exe();
                if let Ok(exe_path) = current_exe {
                    #[cfg(unix)]
                    {
                        let _ = std::process::Command::new(&exe_path)
                            .arg("index-symbols-internal")
                            .arg(&cwd)
                            .stdin(std::process::Stdio::null())
                            .stdout(std::process::Stdio::null())
                            .stderr(std::process::Stdio::null())
                            .spawn();
                    }

                    #[cfg(windows)]
                    {
                        use std::os::windows::process::CommandExt;
                        const CREATE_NO_WINDOW: u32 = 0x08000000;

                        let _ = std::process::Command::new(&exe_path)
                            .arg("index-symbols-internal")
                            .arg(&cwd)
                            .creation_flags(CREATE_NO_WINDOW)
                            .stdin(std::process::Stdio::null())
                            .stdout(std::process::Stdio::null())
                            .stderr(std::process::Stdio::null())
                            .spawn();
                    }

                    log::debug!("Background symbol indexing process spawned");
                }
            }

            result_tx.send(result).ok();
        });

        self.indexing = true;
        self.indexing_start_time = Some(Instant::now());
        self.index_rx = Some(result_rx);
        self.index_progress_rx = Some(progress_rx);

        Ok(())
    }

    /// Clear the entire index cache and trigger a full rebuild
    fn trigger_clear_and_reindex(&mut self) -> Result<()> {
        // Get cache directory path
        let cache_dir = self.cache.path();

        // Clear all cache files (but preserve the directory structure)
        // Be careful not to delete the entire directory as it might be recreated immediately
        let files_to_remove = [
            "meta.db",
            "trigrams.bin",
            "content.bin",
            "symbols.db",
            "indexing.lock",
            "indexing.status",
        ];

        for file_name in &files_to_remove {
            let file_path = cache_dir.join(file_name);
            if file_path.exists() {
                if let Err(e) = std::fs::remove_file(&file_path) {
                    log::warn!("Failed to remove {}: {}", file_name, e);
                }
            }
        }

        // Set status to Missing to indicate we need a fresh index
        self.index_status = IndexStatusState::Missing;

        // Trigger fresh index
        self.trigger_index()
    }

    fn open_in_editor_suspended(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
        result: &SearchResult,
    ) -> Result<()> {
        let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vim".to_string());
        let line = result.span.start_line;

        // Build command with line number
        let args = match editor.as_str() {
            "vim" | "nvim" => vec![format!("+{}", line), result.path.clone()],
            "emacs" => vec![format!("+{}:0", line), result.path.clone()],
            "code" | "vscode" => vec!["-g".to_string(), format!("{}:{}", result.path, line)],
            _ => vec![result.path.clone()],
        };

        // Suspend terminal properly
        crossterm::terminal::disable_raw_mode()?;
        crossterm::execute!(
            terminal.backend_mut(),
            crossterm::terminal::LeaveAlternateScreen,
            crossterm::event::DisableMouseCapture,
            crossterm::cursor::Show
        )?;
        terminal.show_cursor()?;

        // Open editor
        let status = std::process::Command::new(&editor).args(&args).status()?;

        // Resume terminal
        crossterm::terminal::enable_raw_mode()?;
        crossterm::execute!(
            terminal.backend_mut(),
            crossterm::terminal::EnterAlternateScreen,
            crossterm::event::EnableMouseCapture
        )?;
        terminal.clear()?;

        if !status.success() {
            self.error_message = Some(format!(
                "Editor exited with error code: {:?}",
                status.code()
            ));
        }

        Ok(())
    }

    fn setup_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
        crossterm::terminal::enable_raw_mode()?;
        let mut stdout = io::stdout();
        crossterm::execute!(
            stdout,
            crossterm::terminal::EnterAlternateScreen,
            crossterm::event::EnableMouseCapture
        )?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        Ok(terminal)
    }

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

    // Getters for UI rendering
    pub fn input(&self) -> &InputField {
        &self.input
    }

    pub fn results(&self) -> &ResultList {
        &self.results
    }

    pub fn results_mut(&mut self) -> &mut ResultList {
        &mut self.results
    }

    pub fn mode(&self) -> &AppMode {
        &self.mode
    }

    pub fn filters(&self) -> &QueryFilters {
        &self.filters
    }

    pub fn capabilities(&self) -> &TerminalCapabilities {
        &self.capabilities
    }

    pub fn theme(&self) -> &ThemeManager {
        &self.theme
    }

    pub fn effects(&self) -> &EffectManager {
        &self.effects
    }

    pub fn index_status(&self) -> &IndexStatusState {
        &self.index_status
    }

    pub fn focus_state(&self) -> &FocusState {
        &self.focus_state
    }

    pub fn error_message(&self) -> Option<&str> {
        self.error_message.as_deref()
    }

    pub fn info_message(&self) -> Option<&str> {
        self.info_message.as_deref()
    }

    pub fn preview_content(&self) -> Option<&FilePreview> {
        self.preview_content.as_ref()
    }

    pub fn searching(&self) -> bool {
        self.searching
    }

    pub fn indexing(&self) -> bool {
        self.indexing
    }

    pub fn indexing_elapsed_secs(&self) -> Option<u64> {
        self.indexing_start_time
            .map(|start| start.elapsed().as_secs())
    }

    pub fn cwd(&self) -> &PathBuf {
        &self.cwd
    }

    pub fn filter_selector(&self) -> Option<&super::filter_selector::FilterSelector> {
        self.filter_selector.as_ref()
    }

    pub fn filter_selector_mut(&mut self) -> Option<&mut super::filter_selector::FilterSelector> {
        self.filter_selector.as_mut()
    }

    pub fn filter_badge_positions(&self) -> &super::mouse::FilterBadgePositions {
        &self.filter_badge_positions
    }
}

impl FilePreview {
    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn content(&self) -> &[String] {
        &self.content
    }

    pub fn visible_lines(&self, height: usize) -> &[String] {
        let start = self.scroll_offset;
        let end = (start + height).min(self.content.len());
        &self.content[start..end]
    }

    pub fn center_line(&self) -> usize {
        self.center_line
    }

    pub fn scroll_offset(&self) -> usize {
        self.scroll_offset
    }

    pub fn language(&self) -> crate::models::Language {
        self.language
    }
}
