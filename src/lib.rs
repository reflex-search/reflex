//! Reflex: Local-first, structure-aware code search engine
//!
//! Reflex is a fast, deterministic code search tool designed specifically
//! for AI coding agents. It provides structured results (symbols, spans,
//! scopes) with sub-100ms latency by maintaining a lightweight, incremental
//! cache in `.reflex/`.
//!
//! # Architecture
//!
//! - **Indexer**: Scans code and builds trigram index; writes to cache
//! - **Query Engine**: Loads cache on demand; executes deterministic searches; parses symbols at runtime
//! - **Cache**: Memory-mapped storage for trigrams, content, and metadata
//!
//! # Example Usage
//!
//! ```no_run
//! use reflex::{cache::CacheManager, indexer::Indexer, models::IndexConfig};
//!
//! // Create and initialize index
//! let cache = CacheManager::new(".");
//! let config = IndexConfig::default();
//! let indexer = Indexer::new(cache, config);
//! let stats = indexer.index(".", false).unwrap();
//!
//! println!("Indexed {} files", stats.total_files);
//! ```

pub mod ast_query;
pub mod background_indexer;
pub mod cache;
pub mod cli;
pub mod context;
pub mod content_store;
pub mod dependency;
pub mod formatter;
pub mod git;
pub mod indexer;
pub mod interactive;
pub mod line_filter;
pub mod mcp;
pub mod models;
pub mod output;
pub mod parsers;
pub mod pulse;
pub mod query;
pub mod regex_trigrams;
pub mod semantic;
pub mod symbol_cache;
pub mod trigram;
pub mod watcher;

// Re-export commonly used types
pub use cache::CacheManager;
pub use indexer::Indexer;
pub use models::{
    Dependency, DependencyInfo, FileGroupedResult, ImportType, IndexConfig, IndexStats, IndexStatus, IndexWarning,
    IndexWarningDetails, IndexedFile, Language, MatchResult, QueryResponse, SearchResult, Span, SymbolKind,
};
pub use query::{QueryEngine, QueryFilter};
pub use watcher::{watch, WatchConfig};
