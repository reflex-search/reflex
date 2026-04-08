//! CLI argument parsing and command handlers

use anyhow::{Context, Result};
use clap::{CommandFactory, Parser, Subcommand};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use indicatif::{ProgressBar, ProgressStyle};
use owo_colors::OwoColorize;

use crate::cache::CacheManager;
use crate::indexer::Indexer;
use crate::models::{IndexConfig, Language};
use crate::output;
use crate::pulse;
use crate::query::{QueryEngine, QueryFilter};

/// Reflex: Local-first, structure-aware code search for AI agents
#[derive(Parser, Debug)]
#[command(
    name = "rfx",
    version,
    about = "A fast, deterministic code search engine built for AI",
    long_about = "Reflex is a local-first, structure-aware code search engine that returns \
                  structured results (symbols, spans, scopes) with sub-100ms latency. \
                  Designed for AI coding agents and automation."
)]
pub struct Cli {
    /// Enable verbose logging (can be repeated for more verbosity)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub enum IndexSubcommand {
    /// Show background symbol indexing status
    Status,

    /// Compact the cache by removing deleted files
    ///
    /// Removes files from the cache that no longer exist on disk and reclaims
    /// disk space using SQLite VACUUM. This operation is also performed automatically
    /// in the background every 24 hours during normal usage.
    ///
    /// Examples:
    ///   rfx index compact                # Show compaction results
    ///   rfx index compact --json         # JSON output
    Compact {
        /// Output format as JSON
        #[arg(long)]
        json: bool,

        /// Pretty-print JSON output (only with --json)
        #[arg(long)]
        pretty: bool,
    },
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Build or update the local code index
    Index {
        /// Directory to index (defaults to current directory)
        #[arg(value_name = "PATH", default_value = ".")]
        path: PathBuf,

        /// Force full rebuild (ignore incremental cache)
        #[arg(short, long)]
        force: bool,

        /// Languages to include (empty = all)
        #[arg(short, long, value_delimiter = ',')]
        languages: Vec<String>,

        /// Suppress all output (no progress bar, no summary)
        #[arg(short, long)]
        quiet: bool,

        /// Subcommand (status, compact)
        #[command(subcommand)]
        command: Option<IndexSubcommand>,
    },

    /// Query the code index
    ///
    /// If no pattern is provided, launches interactive mode (TUI).
    ///
    /// Search modes:
    ///   - Default: Word-boundary matching (precise, finds complete identifiers)
    ///     Example: rfx query "Error" → finds "Error" but not "NetworkError"
    ///     Example: rfx query "test" → finds "test" but not "test_helper"
    ///
    ///   - Symbol search: Word-boundary for text, exact match for symbols
    ///     Example: rfx query "parse" --symbols → finds only "parse" function/class
    ///     Example: rfx query "parse" --kind function → finds only "parse" functions
    ///
    ///   - Substring search: Expansive matching (opt-in with --contains)
    ///     Example: rfx query "mb" --contains → finds "mb", "kmb_dai_ops", "symbol", etc.
    ///
    ///   - Regex search: Pattern-controlled matching (opt-in with --regex)
    ///     Example: rfx query "^mb_.*" --regex → finds "mb_init", "mb_start", etc.
    ///
    /// Interactive mode:
    ///   - Launch with: rfx query
    ///   - Search, filter, and navigate code results in a live TUI
    ///   - Press '?' for help, 'q' to quit
    Query {
        /// Search pattern (omit to launch interactive mode)
        pattern: Option<String>,

        /// Search symbol definitions only (functions, classes, etc.)
        #[arg(short, long)]
        symbols: bool,

        /// Filter by language
        /// Supported: rust, python, javascript, typescript, vue, svelte, go, java, php, c, c++, c#, ruby, kotlin, zig
        #[arg(short, long)]
        lang: Option<String>,

        /// Filter by symbol kind (implies --symbols)
        /// Supported: function, class, struct, enum, interface, trait, constant, variable, method, module, namespace, type, macro, property, event, import, export, attribute
        #[arg(short, long)]
        kind: Option<String>,

        /// Use AST pattern matching (SLOW: 500ms-2s+, scans all files)
        ///
        /// WARNING: AST queries bypass trigram optimization and scan the entire codebase.
        /// In 95% of cases, use --symbols instead which is 10-100x faster.
        ///
        /// When --ast is set, the pattern parameter is interpreted as a Tree-sitter
        /// S-expression query instead of text search.
        ///
        /// RECOMMENDED: Always use --glob to limit scope for better performance.
        ///
        /// Examples:
        ///   Fast (2-50ms):    rfx query "fetch" --symbols --kind function --lang python
        ///   Slow (500ms-2s):  rfx query "(function_definition) @fn" --ast --lang python
        ///   Faster with glob: rfx query "(class_declaration) @class" --ast --lang typescript --glob "src/**/*.ts"
        #[arg(long)]
        ast: bool,

        /// Use regex pattern matching
        ///
        /// Enables standard regex syntax in the search pattern:
        ///   |  for alternation (OR) - NO backslash needed
        ///   .  matches any character
        ///   .*  matches zero or more characters
        ///   ^  anchors to start of line
        ///   $  anchors to end of line
        ///
        /// Examples:
        ///   --regex "belongsTo|hasMany"       Match belongsTo OR hasMany
        ///   --regex "^import.*from"           Lines starting with import...from
        ///   --regex "fn.*test"                Functions containing 'test'
        ///
        /// Note: Cannot be combined with --contains (mutually exclusive)
        #[arg(short = 'r', long)]
        regex: bool,

        /// Output format as JSON
        #[arg(long)]
        json: bool,

        /// Pretty-print JSON output (only with --json)
        /// By default, JSON is minified to reduce token usage
        #[arg(long)]
        pretty: bool,

        /// AI-optimized mode: returns JSON with ai_instruction field
        /// Implies --json (minified by default, use --pretty for formatted output)
        /// Provides context-aware guidance to AI agents on response format and next actions
        #[arg(long)]
        ai: bool,

        /// Maximum number of results
        #[arg(short = 'n', long)]
        limit: Option<usize>,

        /// Pagination offset (skip first N results after sorting)
        /// Use with --limit for pagination: --offset 0 --limit 10, then --offset 10 --limit 10
        #[arg(short = 'o', long)]
        offset: Option<usize>,

        /// Show full symbol definition (entire function/class body)
        /// Only applicable to symbol searches
        #[arg(long)]
        expand: bool,

        /// Filter by file path (supports substring matching)
        /// Example: --file math.rs or --file helpers/
        #[arg(short = 'f', long)]
        file: Option<String>,

        /// Exact symbol name match (no substring matching)
        /// Only applicable to symbol searches
        #[arg(long)]
        exact: bool,

        /// Use substring matching for both text and symbols (expansive search)
        ///
        /// Default behavior uses word-boundary matching for precision:
        ///   "Error" matches "Error" but not "NetworkError"
        ///
        /// With --contains, enables substring matching (expansive):
        ///   "Error" matches "Error", "NetworkError", "error_handler", etc.
        ///
        /// Use cases:
        ///   - Finding partial matches: --contains "partial"
        ///   - When you're unsure of exact names
        ///   - Exploratory searches
        ///
        /// Note: Cannot be combined with --regex or --exact (mutually exclusive)
        #[arg(long)]
        contains: bool,

        /// Only show count and timing, not the actual results
        #[arg(short, long)]
        count: bool,

        /// Query timeout in seconds (0 = no timeout, default: 30)
        #[arg(short = 't', long, default_value = "30")]
        timeout: u64,

        /// Use plain text output (disable colors and syntax highlighting)
        #[arg(long)]
        plain: bool,

        /// Include files matching glob pattern (can be repeated)
        ///
        /// Pattern syntax (NO shell quotes in the pattern itself):
        ///   ** = recursive match (all subdirectories)
        ///   *  = single level match (one directory)
        ///
        /// Examples:
        ///   --glob src/**/*.rs          All .rs files under src/ (recursive)
        ///   --glob app/Models/*.php     PHP files directly in Models/ (not subdirs)
        ///   --glob tests/**/*_test.go   All test files under tests/
        ///
        /// Tip: Use --file for simple substring matching instead:
        ///   --file User.php             Simpler than --glob **/User.php
        #[arg(short = 'g', long)]
        glob: Vec<String>,

        /// Exclude files matching glob pattern (can be repeated)
        ///
        /// Same syntax as --glob (** for recursive, * for single level)
        ///
        /// Examples:
        ///   --exclude target/**         Exclude all files under target/
        ///   --exclude **/*.gen.rs       Exclude generated Rust files
        ///   --exclude node_modules/**   Exclude npm dependencies
        #[arg(short = 'x', long)]
        exclude: Vec<String>,

        /// Return only unique file paths (no line numbers or content)
        /// Compatible with --json to output ["path1", "path2", ...]
        #[arg(short = 'p', long)]
        paths: bool,

        /// Disable smart preview truncation (show full lines)
        /// By default, previews are truncated to ~100 chars to reduce token usage
        #[arg(long)]
        no_truncate: bool,

        /// Return all results (no limit)
        /// Equivalent to --limit 0, convenience flag for getting unlimited results
        #[arg(short = 'a', long)]
        all: bool,

        /// Force execution of potentially expensive queries
        /// Bypasses broad query detection that prevents queries with:
        /// • Short patterns (< 3 characters)
        /// • High candidate counts (> 5,000 files for symbol/AST queries)
        /// • AST queries without --glob restrictions
        #[arg(long)]
        force: bool,

        /// Include dependency information (imports) in results
        /// Currently only available for Rust files
        #[arg(long)]
        dependencies: bool,
    },

    /// Start a local HTTP API server
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "7878")]
        port: u16,

        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },

    /// Show index statistics and cache information
    Stats {
        /// Output format as JSON
        #[arg(long)]
        json: bool,

        /// Pretty-print JSON output (only with --json)
        #[arg(long)]
        pretty: bool,
    },

    /// Clear the local cache
    Clear {
        /// Skip confirmation prompt
        #[arg(short, long)]
        yes: bool,
    },

    /// List all indexed files
    ListFiles {
        /// Output format as JSON
        #[arg(long)]
        json: bool,

        /// Pretty-print JSON output (only with --json)
        #[arg(long)]
        pretty: bool,
    },

    /// Watch for file changes and auto-reindex
    ///
    /// Continuously monitors the workspace for changes and automatically
    /// triggers incremental reindexing. Useful for IDE integrations and
    /// keeping the index always fresh during active development.
    ///
    /// The debounce timer resets on every file change, batching rapid edits
    /// (e.g., multi-file refactors, format-on-save) into a single reindex.
    Watch {
        /// Directory to watch (defaults to current directory)
        #[arg(value_name = "PATH", default_value = ".")]
        path: PathBuf,

        /// Debounce duration in milliseconds (default: 15000 = 15s)
        /// Waits this long after the last change before reindexing
        /// Valid range: 5000-30000 (5-30 seconds)
        #[arg(short, long, default_value = "15000")]
        debounce: u64,

        /// Suppress output (only log errors)
        #[arg(short, long)]
        quiet: bool,
    },

    /// Start MCP server for AI agent integration
    ///
    /// Runs Reflex as a Model Context Protocol (MCP) server using stdio transport.
    /// This command is automatically invoked by MCP clients like Claude Code and
    /// should not be run manually.
    ///
    /// Configuration example for Claude Code (~/.claude/claude_code_config.json):
    /// {
    ///   "mcpServers": {
    ///     "reflex": {
    ///       "type": "stdio",
    ///       "command": "rfx",
    ///       "args": ["mcp"]
    ///     }
    ///   }
    /// }
    Mcp,

    /// Analyze codebase structure and dependencies
    ///
    /// Perform graph-wide dependency analysis to understand code architecture.
    /// By default, shows a summary report with counts. Use specific flags for
    /// detailed results.
    ///
    /// Examples:
    ///   rfx analyze                                # Summary report
    ///   rfx analyze --circular                     # Find cycles
    ///   rfx analyze --hotspots                     # Most-imported files
    ///   rfx analyze --hotspots --min-dependents 5  # Filter by minimum
    ///   rfx analyze --unused                       # Orphaned files
    ///   rfx analyze --islands                      # Disconnected components
    ///   rfx analyze --hotspots --count             # Just show count
    ///   rfx analyze --circular --glob "src/**"     # Limit to src/
    Analyze {
        /// Show circular dependencies
        #[arg(long)]
        circular: bool,

        /// Show most-imported files (hotspots)
        #[arg(long)]
        hotspots: bool,

        /// Minimum number of dependents for hotspots (default: 2)
        #[arg(long, default_value = "2", requires = "hotspots")]
        min_dependents: usize,

        /// Show unused/orphaned files
        #[arg(long)]
        unused: bool,

        /// Show disconnected components (islands)
        #[arg(long)]
        islands: bool,

        /// Minimum island size (default: 2)
        #[arg(long, default_value = "2", requires = "islands")]
        min_island_size: usize,

        /// Maximum island size (default: 500 or 50% of total files)
        #[arg(long, requires = "islands")]
        max_island_size: Option<usize>,

        /// Output format: tree (default), table, dot
        #[arg(short = 'f', long, default_value = "tree")]
        format: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Pretty-print JSON output
        #[arg(long)]
        pretty: bool,

        /// Only show count and timing, not the actual results
        #[arg(short, long)]
        count: bool,

        /// Return all results (no limit)
        /// Equivalent to --limit 0, convenience flag for unlimited results
        #[arg(short = 'a', long)]
        all: bool,

        /// Use plain text output (disable colors and syntax highlighting)
        #[arg(long)]
        plain: bool,

        /// Include files matching glob pattern (can be repeated)
        /// Example: --glob "src/**/*.rs" --glob "tests/**/*.rs"
        #[arg(short = 'g', long)]
        glob: Vec<String>,

        /// Exclude files matching glob pattern (can be repeated)
        /// Example: --exclude "target/**" --exclude "*.gen.rs"
        #[arg(short = 'x', long)]
        exclude: Vec<String>,

        /// Force execution of potentially expensive queries
        /// Bypasses broad query detection
        #[arg(long)]
        force: bool,

        /// Maximum number of results
        #[arg(short = 'n', long)]
        limit: Option<usize>,

        /// Pagination offset
        #[arg(short = 'o', long)]
        offset: Option<usize>,

        /// Sort order for results: asc (ascending) or desc (descending)
        /// Applies to --hotspots (by import_count), --islands (by size), --circular (by cycle length)
        /// Default: desc (most important first)
        #[arg(long)]
        sort: Option<String>,
    },

    /// Analyze dependencies for a specific file
    ///
    /// Show dependencies and dependents for a single file.
    /// For graph-wide analysis, use 'rfx analyze' instead.
    ///
    /// Examples:
    ///   rfx deps src/main.rs                  # Show dependencies
    ///   rfx deps src/config.rs --reverse      # Show dependents
    ///   rfx deps src/api.rs --depth 3         # Transitive deps
    Deps {
        /// File path to analyze
        file: PathBuf,

        /// Show files that depend on this file (reverse lookup)
        #[arg(short, long)]
        reverse: bool,

        /// Traversal depth for transitive dependencies (default: 1)
        #[arg(short, long, default_value = "1")]
        depth: usize,

        /// Output format: tree (default), table, dot
        #[arg(short = 'f', long, default_value = "tree")]
        format: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Pretty-print JSON output
        #[arg(long)]
        pretty: bool,
    },

    /// Ask a natural language question and generate search queries
    ///
    /// Uses an LLM to translate natural language questions into `rfx query` commands.
    /// Requires API key configuration for one of: OpenAI, Anthropic, or OpenRouter.
    ///
    /// If no question is provided, launches interactive chat mode by default.
    ///
    /// Configuration:
    ///   1. Run interactive setup wizard (recommended):
    ///      rfx ask --configure
    ///
    ///   2. OR set API key via environment variable:
    ///      - OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY
    ///
    ///   3. Optional: Configure provider in .reflex/config.toml:
    ///      [semantic]
    ///      provider = "openai"  # or anthropic, openrouter
    ///      model = "gpt-5.1-mini"  # optional, defaults to provider default
    ///
    /// Examples:
    ///   rfx ask --configure                           # Interactive setup wizard
    ///   rfx ask                                       # Launch interactive chat (default)
    ///   rfx ask "Find all TODOs in Rust files"
    ///   rfx ask "Where is the main function defined?" --execute
    ///   rfx ask "Show me error handling code" --provider openrouter
    Ask {
        /// Natural language question
        question: Option<String>,

        /// Execute queries immediately without confirmation
        #[arg(short, long)]
        execute: bool,

        /// Override configured LLM provider (openai, anthropic, openrouter)
        #[arg(short, long)]
        provider: Option<String>,

        /// Output format as JSON
        #[arg(long)]
        json: bool,

        /// Pretty-print JSON output (only with --json)
        #[arg(long)]
        pretty: bool,

        /// Additional context to inject into prompt (e.g., from `rfx context`)
        #[arg(long)]
        additional_context: Option<String>,

        /// Launch interactive configuration wizard to set up AI provider and API key
        #[arg(long)]
        configure: bool,

        /// Enable agentic mode (multi-step reasoning with context gathering)
        #[arg(long)]
        agentic: bool,

        /// Maximum iterations for query refinement in agentic mode (default: 2)
        #[arg(long, default_value = "2")]
        max_iterations: usize,

        /// Skip result evaluation in agentic mode
        #[arg(long)]
        no_eval: bool,

        /// Show LLM reasoning blocks at each phase (agentic mode only)
        #[arg(long)]
        show_reasoning: bool,

        /// Verbose output: show tool results and details (agentic mode only)
        #[arg(long)]
        verbose: bool,

        /// Quiet mode: suppress progress output (agentic mode only)
        #[arg(long)]
        quiet: bool,

        /// Generate a conversational answer based on search results
        #[arg(long)]
        answer: bool,

        /// Launch interactive chat mode (TUI) with conversation history
        #[arg(short = 'i', long)]
        interactive: bool,

        /// Debug mode: output full LLM prompts and retain terminal history
        #[arg(long)]
        debug: bool,
    },

    /// Generate codebase context for AI prompts
    ///
    /// Provides structural and organizational context about the project to help
    /// LLMs understand project layout. Use with `rfx ask --additional-context`.
    ///
    /// By default (no flags), shows all context types. Use individual flags to
    /// select specific context types.
    ///
    /// Examples:
    ///   rfx context                                    # Full context (all types)
    ///   rfx context --path services/backend            # Full context for monorepo subdirectory
    ///   rfx context --framework --entry-points         # Specific context types only
    ///   rfx context --structure --depth 5              # Deep directory tree
    ///
    ///   # Use with semantic queries
    ///   rfx ask "find auth" --additional-context "$(rfx context --framework)"
    Context {
        /// Show directory structure (enabled by default)
        #[arg(long)]
        structure: bool,

        /// Focus on specific directory path
        #[arg(short, long)]
        path: Option<String>,

        /// Show file type distribution (enabled by default)
        #[arg(long)]
        file_types: bool,

        /// Detect project type (CLI/library/webapp/monorepo)
        #[arg(long)]
        project_type: bool,

        /// Detect frameworks and conventions
        #[arg(long)]
        framework: bool,

        /// Show entry point files
        #[arg(long)]
        entry_points: bool,

        /// Show test organization pattern
        #[arg(long)]
        test_layout: bool,

        /// List important configuration files
        #[arg(long)]
        config_files: bool,

        /// Tree depth for --structure (default: 1)
        #[arg(long, default_value = "1")]
        depth: usize,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Internal command: Run background symbol indexing (hidden from help)
    #[command(hide = true)]
    IndexSymbolsInternal {
        /// Cache directory path
        cache_dir: PathBuf,
    },

    /// Take and manage codebase snapshots for structural tracking
    ///
    /// Snapshots capture the structural state of the index (files, dependencies,
    /// metrics) for diffing and historical analysis.
    ///
    /// With no subcommand, creates a new snapshot.
    ///
    /// Examples:
    ///   rfx snapshot               # Create a new snapshot
    ///   rfx snapshot list           # List available snapshots
    ///   rfx snapshot diff           # Diff latest vs previous
    ///   rfx snapshot gc             # Run retention policy
    Snapshot {
        #[command(subcommand)]
        command: Option<SnapshotSubcommand>,
    },

    /// Generate codebase intelligence surfaces (digest, wiki, map, site)
    ///
    /// Pulse turns structural facts from the index into browsable documentation.
    /// The `generate` command creates a Zola project and builds it into a static HTML site.
    ///
    /// Examples:
    ///   rfx pulse digest --no-llm           # Structural-only digest
    ///   rfx pulse wiki --no-llm             # Generate wiki pages
    ///   rfx pulse map                        # Architecture map (mermaid)
    ///   rfx pulse generate --no-llm          # Full static site (Zola)
    Pulse {
        #[command(subcommand)]
        command: PulseSubcommand,
    },

    /// Manage LLM provider configuration (shared by `ask` and `pulse`)
    ///
    /// Examples:
    ///   rfx llm config                       # Launch interactive setup wizard
    ///   rfx llm status                       # Show current LLM configuration
    Llm {
        #[command(subcommand)]
        command: LlmSubcommand,
    },
}

#[derive(Subcommand, Debug)]
pub enum SnapshotSubcommand {
    /// Compare two snapshots
    ///
    /// Defaults to latest vs previous snapshot.
    Diff {
        /// Baseline snapshot ID (defaults to second-most-recent)
        #[arg(long)]
        baseline: Option<String>,

        /// Current snapshot ID (defaults to most recent)
        #[arg(long)]
        current: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Pretty-print JSON output
        #[arg(long)]
        pretty: bool,
    },

    /// List available snapshots
    List {
        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Pretty-print JSON output
        #[arg(long)]
        pretty: bool,
    },

    /// Run snapshot garbage collection
    Gc {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
pub enum PulseSubcommand {
    /// Generate a structural change digest
    Digest {
        /// Baseline snapshot ID
        #[arg(long)]
        baseline: Option<String>,

        /// Current snapshot ID
        #[arg(long)]
        current: Option<String>,

        /// Skip LLM narration (structural content only)
        #[arg(long)]
        no_llm: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Pretty-print JSON output
        #[arg(long)]
        pretty: bool,
    },

    /// Generate living wiki pages
    Wiki {
        /// Skip LLM narration
        #[arg(long)]
        no_llm: bool,

        /// Output directory for markdown files
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Export an architecture map
    Map {
        /// Output format (mermaid, d2)
        #[arg(short, long, default_value = "mermaid")]
        format: String,

        /// Output file (prints to stdout if not set)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Zoom level: repo (default) or module path
        #[arg(short, long)]
        zoom: Option<String>,
    },

    /// Generate a complete static site (Zola project + HTML build)
    ///
    /// Creates a Zola project with markdown content, templates, and CSS,
    /// then downloads Zola and builds it into a static HTML site.
    /// The --base-url maps to Zola's base_url config.
    Generate {
        /// Output directory for the Zola project
        #[arg(short, long, default_value = "pulse-site")]
        output: PathBuf,

        /// Base URL for the site (maps to Zola's base_url)
        #[arg(long, default_value = "/")]
        base_url: String,

        /// Site title
        #[arg(long)]
        title: Option<String>,

        /// Surfaces to include (comma-separated: wiki,digest,map)
        #[arg(long)]
        include: Option<String>,

        /// Skip LLM narration
        #[arg(long)]
        no_llm: bool,

        /// Clean output directory before generating
        #[arg(long)]
        clean: bool,

        /// Force re-narration (ignore LLM cache)
        #[arg(long)]
        force_renarrate: bool,

        /// Maximum concurrent LLM requests (0 = unlimited, default)
        #[arg(long, default_value = "0")]
        concurrency: usize,

        /// Maximum directory depth for module discovery (1=top-level only, 2=default)
        #[arg(long, default_value = "2")]
        depth: u8,

        /// Minimum file count for a module to be included
        #[arg(long, default_value = "1")]
        min_files: usize,
    },

    /// Serve the generated site locally
    ///
    /// Starts a local development server for the Pulse site.
    /// Uses Zola's built-in server with live reload.
    Serve {
        /// Directory containing the generated Zola project
        #[arg(short, long, default_value = "pulse-site")]
        output: PathBuf,

        /// Port to serve on
        #[arg(short, long, default_value = "1111")]
        port: u16,

        /// Open browser automatically
        #[arg(long, default_value = "true")]
        open: bool,
    },
}

#[derive(Subcommand, Debug)]
pub enum LlmSubcommand {
    /// Launch interactive configuration wizard for AI provider and API key
    Config,
    /// Show current LLM configuration status
    Status,
}

/// Try to run background cache compaction if needed
///
/// Checks if 24+ hours have passed since last compaction.
/// If yes, spawns a non-blocking background thread to compact the cache.
/// Main command continues immediately without waiting for compaction.
///
/// Compaction is skipped for commands that don't need it:
/// - Clear (will delete the cache anyway)
/// - Mcp (long-running server process)
/// - Watch (long-running watcher process)
/// - Serve (long-running HTTP server)
fn try_background_compact(cache: &CacheManager, command: &Command) {
    // Skip compaction for certain commands
    match command {
        Command::Clear { .. } => {
            log::debug!("Skipping compaction for Clear command");
            return;
        }
        Command::Mcp => {
            log::debug!("Skipping compaction for Mcp command");
            return;
        }
        Command::Watch { .. } => {
            log::debug!("Skipping compaction for Watch command");
            return;
        }
        Command::Serve { .. } => {
            log::debug!("Skipping compaction for Serve command");
            return;
        }
        _ => {}
    }

    // Check if compaction should run
    let should_compact = match cache.should_compact() {
        Ok(true) => true,
        Ok(false) => {
            log::debug!("Compaction not needed yet (last run <24h ago)");
            return;
        }
        Err(e) => {
            log::warn!("Failed to check compaction status: {}", e);
            return;
        }
    };

    if !should_compact {
        return;
    }

    log::info!("Starting background cache compaction...");

    // Clone cache path for background thread
    let cache_path = cache.path().to_path_buf();

    // Spawn background thread for compaction
    std::thread::spawn(move || {
        let cache = CacheManager::new(cache_path.parent().expect("Cache should have parent directory"));

        match cache.compact() {
            Ok(report) => {
                log::info!(
                    "Background compaction completed: {} files removed, {:.2} MB saved, took {}ms",
                    report.files_removed,
                    report.space_saved_bytes as f64 / 1_048_576.0,
                    report.duration_ms
                );
            }
            Err(e) => {
                log::warn!("Background compaction failed: {}", e);
            }
        }
    });

    log::debug!("Background compaction thread spawned - main command continuing");
}

impl Cli {
    /// Execute the CLI command
    pub fn execute(self) -> Result<()> {
        // Setup logging based on verbosity
        let log_level = match self.verbose {
            0 => "warn",   // Default: only warnings and errors
            1 => "info",   // -v: show info messages
            2 => "debug",  // -vv: show debug messages
            _ => "trace",  // -vvv: show trace messages
        };
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
            .init();

        // Try background compaction (non-blocking) before command execution
        if let Some(ref command) = self.command {
            // Use current directory as default cache location
            let cache = CacheManager::new(".");
            try_background_compact(&cache, command);
        }

        // Execute the subcommand, or show help if no command provided
        match self.command {
            None => {
                // No subcommand: show help
                Cli::command().print_help()?;
                println!();  // Add newline after help
                Ok(())
            }
            Some(Command::Index { path, force, languages, quiet, command }) => {
                match command {
                    None => {
                        // Default: run index build
                        handle_index_build(&path, &force, &languages, &quiet)
                    }
                    Some(IndexSubcommand::Status) => {
                        handle_index_status()
                    }
                    Some(IndexSubcommand::Compact { json, pretty }) => {
                        handle_index_compact(&json, &pretty)
                    }
                }
            }
            Some(Command::Query { pattern, symbols, lang, kind, ast, regex, json, pretty, ai, limit, offset, expand, file, exact, contains, count, timeout, plain, glob, exclude, paths, no_truncate, all, force, dependencies }) => {
                // If no pattern provided, launch interactive mode
                match pattern {
                    None => handle_interactive(),
                    Some(pattern) => handle_query(pattern, symbols, lang, kind, ast, regex, json, pretty, ai, limit, offset, expand, file, exact, contains, count, timeout, plain, glob, exclude, paths, no_truncate, all, force, dependencies)
                }
            }
            Some(Command::Serve { port, host }) => {
                handle_serve(port, host)
            }
            Some(Command::Stats { json, pretty }) => {
                handle_stats(json, pretty)
            }
            Some(Command::Clear { yes }) => {
                handle_clear(yes)
            }
            Some(Command::ListFiles { json, pretty }) => {
                handle_list_files(json, pretty)
            }
            Some(Command::Watch { path, debounce, quiet }) => {
                handle_watch(path, debounce, quiet)
            }
            Some(Command::Mcp) => {
                handle_mcp()
            }
            Some(Command::Analyze { circular, hotspots, min_dependents, unused, islands, min_island_size, max_island_size, format, json, pretty, count, all, plain, glob, exclude, force, limit, offset, sort }) => {
                handle_analyze(circular, hotspots, min_dependents, unused, islands, min_island_size, max_island_size, format, json, pretty, count, all, plain, glob, exclude, force, limit, offset, sort)
            }
            Some(Command::Deps { file, reverse, depth, format, json, pretty }) => {
                handle_deps(file, reverse, depth, format, json, pretty)
            }
            Some(Command::Ask { question, execute, provider, json, pretty, additional_context, configure, agentic, max_iterations, no_eval, show_reasoning, verbose, quiet, answer, interactive, debug }) => {
                handle_ask(question, execute, provider, json, pretty, additional_context, configure, agentic, max_iterations, no_eval, show_reasoning, verbose, quiet, answer, interactive, debug)
            }
            Some(Command::Context { structure, path, file_types, project_type, framework, entry_points, test_layout, config_files, depth, json }) => {
                handle_context(structure, path, file_types, project_type, framework, entry_points, test_layout, config_files, depth, json)
            }
            Some(Command::IndexSymbolsInternal { cache_dir }) => {
                handle_index_symbols_internal(cache_dir)
            }
            Some(Command::Snapshot { command }) => {
                match command {
                    None => handle_snapshot_create(),
                    Some(crate::cli::SnapshotSubcommand::List { json, pretty }) => {
                        handle_snapshot_list(json, pretty)
                    }
                    Some(crate::cli::SnapshotSubcommand::Diff { baseline, current, json, pretty }) => {
                        handle_snapshot_diff(baseline, current, json, pretty)
                    }
                    Some(crate::cli::SnapshotSubcommand::Gc { json }) => {
                        handle_snapshot_gc(json)
                    }
                }
            }
            Some(Command::Pulse { command }) => {
                match command {
                    crate::cli::PulseSubcommand::Digest { baseline, current, no_llm, json, pretty } => {
                        handle_pulse_digest(baseline, current, no_llm, json, pretty)
                    }
                    crate::cli::PulseSubcommand::Wiki { no_llm, output, json } => {
                        handle_pulse_wiki(no_llm, output, json)
                    }
                    crate::cli::PulseSubcommand::Map { format, output, zoom } => {
                        handle_pulse_map(format, output, zoom)
                    }
                    crate::cli::PulseSubcommand::Generate { output, base_url, title, include, no_llm, clean, force_renarrate, concurrency, depth, min_files } => {
                        handle_pulse_generate(output, base_url, title, include, no_llm, clean, force_renarrate, concurrency, depth, min_files)
                    }
                    crate::cli::PulseSubcommand::Serve { output, port, open } => {
                        handle_pulse_serve(output, port, open)
                    }
                }
            }
            Some(Command::Llm { command }) => {
                match command {
                    crate::cli::LlmSubcommand::Config => handle_llm_config(),
                    crate::cli::LlmSubcommand::Status => handle_llm_status(),
                }
            }
        }
    }
}

/// Handle the `index status` subcommand
fn handle_index_status() -> Result<()> {
    log::info!("Checking background symbol indexing status");

    let cache = CacheManager::new(".");
    let cache_path = cache.path().to_path_buf();

    match crate::background_indexer::BackgroundIndexer::get_status(&cache_path) {
            Ok(Some(status)) => {
                println!("Background Symbol Indexing Status");
                println!("==================================");
                println!("State:           {:?}", status.state);
                println!("Total files:     {}", status.total_files);
                println!("Processed:       {}", status.processed_files);
                println!("Cached:          {}", status.cached_files);
                println!("Parsed:          {}", status.parsed_files);
                println!("Failed:          {}", status.failed_files);
                println!("Started:         {}", status.started_at);
                println!("Last updated:    {}", status.updated_at);

                if let Some(completed_at) = &status.completed_at {
                    println!("Completed:       {}", completed_at);
                }

                if let Some(error) = &status.error {
                    println!("Error:           {}", error);
                }

                // Show progress percentage if running
                if status.state == crate::background_indexer::IndexerState::Running && status.total_files > 0 {
                    let progress = (status.processed_files as f64 / status.total_files as f64) * 100.0;
                    println!("\nProgress:        {:.1}%", progress);
                }

                Ok(())
            }
            Ok(None) => {
                println!("No background symbol indexing in progress.");
                println!("\nRun 'rfx index' to start background symbol indexing.");
                Ok(())
            }
            Err(e) => {
                anyhow::bail!("Failed to get indexing status: {}", e);
            }
        }
    }

/// Handle the `index compact` subcommand
fn handle_index_compact(json: &bool, pretty: &bool) -> Result<()> {
    log::info!("Running cache compaction");

    let cache = CacheManager::new(".");
    let report = cache.compact()?;

    // Output results in requested format
    if *json {
        let json_str = if *pretty {
            serde_json::to_string_pretty(&report)?
        } else {
            serde_json::to_string(&report)?
        };
        println!("{}", json_str);
    } else {
        println!("Cache Compaction Complete");
        println!("=========================");
        println!("Files removed:    {}", report.files_removed);
        println!("Space saved:      {:.2} MB", report.space_saved_bytes as f64 / 1_048_576.0);
        println!("Duration:         {}ms", report.duration_ms);
    }

    Ok(())
}

fn handle_index_build(path: &PathBuf, force: &bool, languages: &[String], quiet: &bool) -> Result<()> {
    log::info!("Starting index build");

    let cache = CacheManager::new(path);
    let cache_path = cache.path().to_path_buf();

    if *force {
        log::info!("Force rebuild requested, clearing existing cache");
        cache.clear()?;
    }

    // Parse language filters
    let lang_filters: Vec<Language> = languages
        .iter()
        .filter_map(|s| {
            Language::from_name(s).or_else(|| {
                output::warn(&format!("Unknown language: '{}'. Supported: {}", s, Language::supported_names_help()));
                None
            })
        })
        .collect();

    let config = IndexConfig {
        languages: lang_filters,
        ..Default::default()
    };

    let indexer = Indexer::new(cache, config);
    // Show progress by default, unless quiet mode is enabled
    let show_progress = !quiet;
    let stats = indexer.index(path, show_progress)?;

    // In quiet mode, suppress all output
    if !quiet {
        println!("Indexing complete!");
        println!("  Files indexed: {}", stats.total_files);
        println!("  Cache size: {}", format_bytes(stats.index_size_bytes));
        println!("  Last updated: {}", stats.last_updated);

        // Display language breakdown if we have indexed files
        if !stats.files_by_language.is_empty() {
            println!("\nFiles by language:");

            // Sort languages by count (descending) for consistent output
            let mut lang_vec: Vec<_> = stats.files_by_language.iter().collect();
            lang_vec.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));

            // Calculate column widths
            let max_lang_len = lang_vec.iter().map(|(lang, _)| lang.len()).max().unwrap_or(8);
            let lang_width = max_lang_len.max(8); // At least "Language" header width

            // Print table header
            println!("  {:<width$}  Files  Lines", "Language", width = lang_width);
            println!("  {}  -----  -------", "-".repeat(lang_width));

            // Print rows
            for (language, file_count) in lang_vec {
                let line_count = stats.lines_by_language.get(language).copied().unwrap_or(0);
                println!("  {:<width$}  {:5}  {:7}",
                    language, file_count, line_count,
                    width = lang_width);
            }
        }
    }

    // Start background symbol indexing (if not already running)
    if !crate::background_indexer::BackgroundIndexer::is_running(&cache_path) {
        if !quiet {
            println!("\nStarting background symbol indexing...");
            println!("  Symbols will be cached for faster queries");
            println!("  Check status with: rfx index status");
        }

        // Spawn detached background process for symbol indexing
        // Pass the workspace root, not the .reflex directory
        let current_exe = std::env::current_exe()
            .context("Failed to get current executable path")?;

        #[cfg(unix)]
        {
            std::process::Command::new(&current_exe)
                .arg("index-symbols-internal")
                .arg(path)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .context("Failed to spawn background indexing process")?;
        }

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;

            std::process::Command::new(&current_exe)
                .arg("index-symbols-internal")
                .arg(&path)
                .creation_flags(CREATE_NO_WINDOW)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .context("Failed to spawn background indexing process")?;
        }

        log::debug!("Spawned background symbol indexing process");
    } else if !quiet {
        println!("\n⚠️  Background symbol indexing already in progress");
        println!("  Check status with: rfx index status");
    }

    Ok(())
}

/// Format bytes into human-readable size (KB, MB, GB, etc.)
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Smart truncate preview to reduce token usage
/// Truncates at word boundary if possible, adds ellipsis if truncated
pub fn truncate_preview(preview: &str, max_length: usize) -> String {
    if preview.len() <= max_length {
        return preview.to_string();
    }

    // Find a good break point (prefer word boundary)
    let truncate_at = preview.char_indices()
        .take(max_length)
        .filter(|(_, c)| c.is_whitespace())
        .last()
        .map(|(i, _)| i)
        .unwrap_or(max_length.min(preview.len()));

    let mut truncated = preview[..truncate_at].to_string();
    truncated.push('…');
    truncated
}

/// Handle the `query` subcommand
fn handle_query(
    pattern: String,
    symbols_flag: bool,
    lang: Option<String>,
    kind_str: Option<String>,
    use_ast: bool,
    use_regex: bool,
    as_json: bool,
    pretty_json: bool,
    ai_mode: bool,
    limit: Option<usize>,
    offset: Option<usize>,
    expand: bool,
    file_pattern: Option<String>,
    exact: bool,
    use_contains: bool,
    count_only: bool,
    timeout_secs: u64,
    plain: bool,
    glob_patterns: Vec<String>,
    exclude_patterns: Vec<String>,
    paths_only: bool,
    no_truncate: bool,
    all: bool,
    force: bool,
    include_dependencies: bool,
) -> Result<()> {
    log::info!("Starting query command");

    // AI mode implies JSON output
    let as_json = as_json || ai_mode;

    let cache = CacheManager::new(".");
    let engine = QueryEngine::new(cache);

    // Parse and validate language filter
    let language = if let Some(lang_str) = lang.as_deref() {
        match Language::from_name(lang_str) {
            Some(l) => Some(l),
            None => anyhow::bail!(
                "Unknown language: '{}'\n\nSupported languages:\n  {}\n\nExample: rfx query \"pattern\" --lang rust",
                lang_str, Language::supported_names_help()
            ),
        }
    } else {
        None
    };

    // Parse symbol kind - try exact match first (case-insensitive), then treat as Unknown
    let kind = kind_str.as_deref().and_then(|s| {
        // Try parsing with proper case (PascalCase for SymbolKind)
        let capitalized = {
            let mut chars = s.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().chain(chars.flat_map(|c| c.to_lowercase())).collect(),
            }
        };

        capitalized.parse::<crate::models::SymbolKind>()
            .ok()
            .or_else(|| {
                // If not a known kind, treat as Unknown for flexibility
                log::debug!("Treating '{}' as unknown symbol kind for filtering", s);
                Some(crate::models::SymbolKind::Unknown(s.to_string()))
            })
    });

    // Smart behavior: --kind implies --symbols
    let symbols_mode = symbols_flag || kind.is_some();

    // Smart limit handling:
    // 1. If --count is set: no limit (count should always show total)
    // 2. If --all is set: no limit (None)
    // 3. If --limit 0 is set: no limit (None) - treat 0 as "unlimited"
    // 4. If --paths is set and user didn't specify --limit: no limit (None)
    // 5. If user specified --limit: use that value
    // 6. Otherwise: use default limit of 100
    let final_limit = if count_only {
        None  // --count always shows total count, no pagination
    } else if all {
        None  // --all means no limit
    } else if limit == Some(0) {
        None  // --limit 0 means no limit (unlimited results)
    } else if paths_only && limit.is_none() {
        None  // --paths without explicit --limit means no limit
    } else if let Some(user_limit) = limit {
        Some(user_limit)  // Use user-specified limit
    } else {
        Some(100)  // Default: limit to 100 results for token efficiency
    };

    // Validate AST query requirements
    if use_ast && language.is_none() {
        anyhow::bail!(
            "AST pattern matching requires a language to be specified.\n\
             \n\
             Use --lang to specify the language for tree-sitter parsing.\n\
             \n\
             Supported languages for AST queries:\n\
             • rust, python, go, java, c, c++, c#, php, ruby, kotlin, zig, typescript, javascript\n\
             \n\
             Note: Vue and Svelte use line-based parsing and do not support AST queries.\n\
             \n\
             WARNING: AST queries are SLOW (500ms-2s+). Use --symbols instead for 95% of cases.\n\
             \n\
             Examples:\n\
             • rfx query \"(function_definition) @fn\" --ast --lang python\n\
             • rfx query \"(class_declaration) @class\" --ast --lang typescript --glob \"src/**/*.ts\""
        );
    }

    // VALIDATION: Check for conflicting or problematic flag combinations
    // Only show warnings/errors in non-JSON mode (avoid breaking parsers)
    if !as_json {
        let mut has_errors = false;

        // ERROR: Mutually exclusive pattern matching modes
        if use_regex && use_contains {
            eprintln!("{}", "ERROR: Cannot use --regex and --contains together.".red().bold());
            eprintln!("  {} --regex for pattern matching (alternation, wildcards, etc.)", "•".dimmed());
            eprintln!("  {} --contains for substring matching (expansive search)", "•".dimmed());
            eprintln!("\n  {} Choose one based on your needs:", "Tip:".cyan().bold());
            eprintln!("    {} for OR logic: --regex", "pattern1|pattern2".yellow());
            eprintln!("    {} for substring: --contains", "partial_text".yellow());
            has_errors = true;
        }

        // ERROR: Contradictory matching requirements
        if exact && use_contains {
            eprintln!("{}", "ERROR: Cannot use --exact and --contains together (contradictory).".red().bold());
            eprintln!("  {} --exact requires exact symbol name match", "•".dimmed());
            eprintln!("  {} --contains allows substring matching", "•".dimmed());
            has_errors = true;
        }

        // WARNING: Redundant file filtering
        if file_pattern.is_some() && !glob_patterns.is_empty() {
            eprintln!("{}", "WARNING: Both --file and --glob specified.".yellow().bold());
            eprintln!("  {} --file does substring matching on file paths", "•".dimmed());
            eprintln!("  {} --glob does pattern matching with wildcards", "•".dimmed());
            eprintln!("  {} Both filters will apply (AND condition)", "Note:".dimmed());
            eprintln!("\n  {} Usually you only need one:", "Tip:".cyan().bold());
            eprintln!("    {} for simple matching", "--file User.php".yellow());
            eprintln!("    {} for pattern matching", "--glob src/**/*.php".yellow());
        }

        // INFO: Detect potentially problematic glob patterns
        for pattern in &glob_patterns {
            // Check for literal quotes in pattern
            if (pattern.starts_with('\'') && pattern.ends_with('\'')) ||
               (pattern.starts_with('"') && pattern.ends_with('"')) {
                eprintln!("{}",
                    format!("WARNING: Glob pattern contains quotes: {}", pattern).yellow().bold()
                );
                eprintln!("  {} Shell quotes should not be part of the pattern", "Note:".dimmed());
                eprintln!("  {} --glob src/**/*.rs", "Correct:".green());
                eprintln!("  {} --glob 'src/**/*.rs'", "Wrong:".red().dimmed());
            }

            // Suggest using ** instead of * for recursive matching
            if pattern.contains("*/") && !pattern.contains("**/") {
                eprintln!("{}",
                    format!("INFO: Glob '{}' uses * (matches one directory level)", pattern).cyan()
                );
                eprintln!("  {} Use ** for recursive matching across subdirectories", "Tip:".cyan().bold());
                eprintln!("    {} → matches files in Models/ only", "app/Models/*.php".yellow());
                eprintln!("    {} → matches files in Models/ and subdirs", "app/Models/**/*.php".green());
            }
        }

        if has_errors {
            anyhow::bail!("Invalid flag combination. Fix the errors above and try again.");
        }
    }

    let filter = QueryFilter {
        language,
        kind,
        use_ast,
        use_regex,
        limit: final_limit,
        symbols_mode,
        expand,
        file_pattern,
        exact,
        use_contains,
        timeout_secs,
        glob_patterns: glob_patterns.clone(),
        exclude_patterns,
        paths_only,
        offset,
        force,
        suppress_output: as_json,  // Suppress warnings in JSON mode
        include_dependencies,
        ..Default::default()
    };

    // Measure query time
    let start = Instant::now();

    // Execute query and get pagination metadata
    // Handle errors specially for JSON output mode
    let (query_response, mut flat_results, total_results, has_more) = if use_ast {
        // AST query: pattern is the S-expression, scan all files
        match engine.search_ast_all_files(&pattern, filter.clone()) {
            Ok(ast_results) => {
                let count = ast_results.len();
                (None, ast_results, count, false)
            }
            Err(e) => {
                if as_json {
                    // Output error as JSON
                    let error_response = serde_json::json!({
                        "error": e.to_string(),
                        "query_too_broad": e.to_string().contains("Query too broad")
                    });
                    let json_output = if pretty_json {
                        serde_json::to_string_pretty(&error_response)?
                    } else {
                        serde_json::to_string(&error_response)?
                    };
                    println!("{}", json_output);
                    std::process::exit(1);
                } else {
                    return Err(e);
                }
            }
        }
    } else {
        // Use metadata-aware search for all queries (to get pagination info)
        match engine.search_with_metadata(&pattern, filter.clone()) {
            Ok(response) => {
                let total = response.pagination.total;
                let has_more = response.pagination.has_more;

                // Flatten grouped results to SearchResult vec for plain text formatting
                let flat = response.results.iter()
                    .flat_map(|file_group| {
                        file_group.matches.iter().map(move |m| {
                            crate::models::SearchResult {
                                path: file_group.path.clone(),
                                lang: crate::models::Language::Unknown, // Will be set by formatter if needed
                                kind: m.kind.clone(),
                                symbol: m.symbol.clone(),
                                span: m.span.clone(),
                                preview: m.preview.clone(),
                                dependencies: file_group.dependencies.clone(),
                            }
                        })
                    })
                    .collect();

                (Some(response), flat, total, has_more)
            }
            Err(e) => {
                if as_json {
                    // Output error as JSON
                    let error_response = serde_json::json!({
                        "error": e.to_string(),
                        "query_too_broad": e.to_string().contains("Query too broad")
                    });
                    let json_output = if pretty_json {
                        serde_json::to_string_pretty(&error_response)?
                    } else {
                        serde_json::to_string(&error_response)?
                    };
                    println!("{}", json_output);
                    std::process::exit(1);
                } else {
                    return Err(e);
                }
            }
        }
    };

    // Apply preview truncation unless --no-truncate is set
    if !no_truncate {
        const MAX_PREVIEW_LENGTH: usize = 100;
        for result in &mut flat_results {
            result.preview = truncate_preview(&result.preview, MAX_PREVIEW_LENGTH);
        }
    }

    let elapsed = start.elapsed();

    // Format timing string
    let timing_str = if elapsed.as_millis() < 1 {
        format!("{:.1}ms", elapsed.as_secs_f64() * 1000.0)
    } else {
        format!("{}ms", elapsed.as_millis())
    };

    if as_json {
        if count_only {
            // Count-only JSON mode: output simple count object
            let count_response = serde_json::json!({
                "count": total_results,
                "timing_ms": elapsed.as_millis()
            });
            let json_output = if pretty_json {
                serde_json::to_string_pretty(&count_response)?
            } else {
                serde_json::to_string(&count_response)?
            };
            println!("{}", json_output);
        } else if paths_only {
            // Paths-only JSON mode: output array of {path, line} objects
            let locations: Vec<serde_json::Value> = flat_results.iter()
                .map(|r| serde_json::json!({
                    "path": r.path,
                    "line": r.span.start_line
                }))
                .collect();
            let json_output = if pretty_json {
                serde_json::to_string_pretty(&locations)?
            } else {
                serde_json::to_string(&locations)?
            };
            println!("{}", json_output);
            eprintln!("Found {} unique files in {}", locations.len(), timing_str);
        } else {
            // Get or build QueryResponse for JSON output
            let mut response = if let Some(resp) = query_response {
                // We already have a response from search_with_metadata
                // Apply truncation to the response (the flat_results were already truncated)
                let mut resp = resp;

                // Apply truncation to results
                if !no_truncate {
                    const MAX_PREVIEW_LENGTH: usize = 100;
                    for file_group in resp.results.iter_mut() {
                        for m in file_group.matches.iter_mut() {
                            m.preview = truncate_preview(&m.preview, MAX_PREVIEW_LENGTH);
                        }
                    }
                }

                resp
            } else {
                // For AST queries, build a response with minimal metadata
                // Group flat results by file path
                use crate::models::{PaginationInfo, IndexStatus, FileGroupedResult, MatchResult};
                use std::collections::HashMap;

                let mut grouped: HashMap<String, Vec<crate::models::SearchResult>> = HashMap::new();
                for result in &flat_results {
                    grouped
                        .entry(result.path.clone())
                        .or_default()
                        .push(result.clone());
                }

                // Load ContentReader for extracting context lines
                use crate::content_store::ContentReader;
                let local_cache = CacheManager::new(".");
                let content_path = local_cache.path().join("content.bin");
                let content_reader_opt = ContentReader::open(&content_path).ok();

                let mut file_results: Vec<FileGroupedResult> = grouped
                    .into_iter()
                    .map(|(path, file_matches)| {
                        // Get file_id for context extraction
                        // Note: We use ContentReader's get_file_id_by_path() which returns array indices,
                        // not database file_ids (which are AUTO INCREMENT values)
                        let normalized_path = path.strip_prefix("./").unwrap_or(&path);
                        let file_id_for_context = if let Some(reader) = &content_reader_opt {
                            reader.get_file_id_by_path(normalized_path)
                        } else {
                            None
                        };

                        let matches: Vec<MatchResult> = file_matches
                            .into_iter()
                            .map(|r| {
                                // Extract context lines (default: 3 lines before and after)
                                let (context_before, context_after) = if let (Some(reader), Some(fid)) = (&content_reader_opt, file_id_for_context) {
                                    reader.get_context_by_line(fid as u32, r.span.start_line, 3)
                                        .unwrap_or_else(|_| (vec![], vec![]))
                                } else {
                                    (vec![], vec![])
                                };

                                MatchResult {
                                    kind: r.kind,
                                    symbol: r.symbol,
                                    span: r.span,
                                    preview: r.preview,
                                    context_before,
                                    context_after,
                                }
                            })
                            .collect();
                        FileGroupedResult {
                            path,
                            dependencies: None,
                            matches,
                        }
                    })
                    .collect();

                // Sort by path for deterministic output
                file_results.sort_by(|a, b| a.path.cmp(&b.path));

                crate::models::QueryResponse {
                    ai_instruction: None,  // Will be populated below if ai_mode is true
                    status: IndexStatus::Fresh,
                    can_trust_results: true,
                    warning: None,
                    pagination: PaginationInfo {
                        total: flat_results.len(),
                        count: flat_results.len(),
                        offset: offset.unwrap_or(0),
                        limit,
                        has_more: false, // AST already applied pagination
                    },
                    results: file_results,
                }
            };

            // Generate AI instruction if in AI mode
            if ai_mode {
                let result_count: usize = response.results.iter().map(|fg| fg.matches.len()).sum();

                response.ai_instruction = crate::query::generate_ai_instruction(
                    result_count,
                    response.pagination.total,
                    response.pagination.has_more,
                    symbols_mode,
                    paths_only,
                    use_ast,
                    use_regex,
                    language.is_some(),
                    !glob_patterns.is_empty(),
                    exact,
                );
            }

            let json_output = if pretty_json {
                serde_json::to_string_pretty(&response)?
            } else {
                serde_json::to_string(&response)?
            };
            println!("{}", json_output);

            let result_count: usize = response.results.iter().map(|fg| fg.matches.len()).sum();
            eprintln!("Found {} results in {}", result_count, timing_str);
        }
    } else {
        // Standard output with formatting
        if count_only {
            println!("Found {} results in {}", flat_results.len(), timing_str);
            return Ok(());
        }

        if paths_only {
            // Paths-only plain text mode: output one path per line
            if flat_results.is_empty() {
                eprintln!("No results found (searched in {}).", timing_str);
            } else {
                for result in &flat_results {
                    println!("{}", result.path);
                }
                eprintln!("Found {} unique files in {}", flat_results.len(), timing_str);
            }
        } else {
            // Standard result formatting
            if flat_results.is_empty() {
                println!("No results found (searched in {}).", timing_str);
            } else {
                // Use formatter for pretty output
                let formatter = crate::formatter::OutputFormatter::new(plain);
                formatter.format_results(&flat_results, &pattern)?;

                // Print summary at the bottom with pagination details
                if total_results > flat_results.len() {
                    // Results were paginated - show detailed count
                    println!("\nFound {} results ({} total) in {}", flat_results.len(), total_results, timing_str);
                    // Show pagination hint if there are more results available
                    if has_more {
                        println!("Use --limit and --offset to paginate");
                    }
                } else {
                    // All results shown - simple count
                    println!("\nFound {} results in {}", flat_results.len(), timing_str);
                }
            }
        }
    }

    Ok(())
}

/// Handle the `serve` subcommand
fn handle_serve(port: u16, host: String) -> Result<()> {
    log::info!("Starting HTTP server on {}:{}", host, port);

    println!("Starting Reflex HTTP server...");
    println!("  Address: http://{}:{}", host, port);
    println!("\nEndpoints:");
    println!("  GET  /query?q=<pattern>&lang=<lang>&kind=<kind>&limit=<n>&symbols=true&regex=true&exact=true&contains=true&expand=true&file=<pattern>&timeout=<secs>&glob=<pattern>&exclude=<pattern>&paths=true&dependencies=true");
    println!("  GET  /stats");
    println!("  POST /index");
    println!("\nPress Ctrl+C to stop.");

    // Start the server using tokio runtime
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        run_server(port, host).await
    })
}

/// Run the HTTP server
async fn run_server(port: u16, host: String) -> Result<()> {
    use axum::{
        extract::{Query as AxumQuery, State},
        http::StatusCode,
        response::{IntoResponse, Json},
        routing::{get, post},
        Router,
    };
    use tower_http::cors::{CorsLayer, Any};
    use std::sync::Arc;

    // Server state shared across requests
    #[derive(Clone)]
    struct AppState {
        cache_path: String,
    }

    // Query parameters for GET /query
    #[derive(Debug, serde::Deserialize)]
    struct QueryParams {
        q: String,
        #[serde(default)]
        lang: Option<String>,
        #[serde(default)]
        kind: Option<String>,
        #[serde(default)]
        limit: Option<usize>,
        #[serde(default)]
        offset: Option<usize>,
        #[serde(default)]
        symbols: bool,
        #[serde(default)]
        regex: bool,
        #[serde(default)]
        exact: bool,
        #[serde(default)]
        contains: bool,
        #[serde(default)]
        expand: bool,
        #[serde(default)]
        file: Option<String>,
        #[serde(default = "default_timeout")]
        timeout: u64,
        #[serde(default)]
        glob: Vec<String>,
        #[serde(default)]
        exclude: Vec<String>,
        #[serde(default)]
        paths: bool,
        #[serde(default)]
        force: bool,
        #[serde(default)]
        dependencies: bool,
    }

    // Default timeout for HTTP queries (30 seconds)
    fn default_timeout() -> u64 {
        30
    }

    // Request body for POST /index
    #[derive(Debug, serde::Deserialize)]
    struct IndexRequest {
        #[serde(default)]
        force: bool,
        #[serde(default)]
        languages: Vec<String>,
    }

    // GET /query endpoint
    async fn handle_query_endpoint(
        State(state): State<Arc<AppState>>,
        AxumQuery(params): AxumQuery<QueryParams>,
    ) -> Result<Json<crate::models::QueryResponse>, (StatusCode, String)> {
        log::info!("Query request: pattern={}", params.q);

        let cache = CacheManager::new(&state.cache_path);
        let engine = QueryEngine::new(cache);

        // Parse language filter
        let language = if let Some(lang_str) = params.lang.as_deref() {
            match Language::from_name(lang_str) {
                Some(l) => Some(l),
                None => return Err((
                    StatusCode::BAD_REQUEST,
                    format!("Unknown language '{}'. Supported: {}", lang_str, Language::supported_names_help())
                )),
            }
        } else {
            None
        };

        // Parse symbol kind
        let kind = params.kind.as_deref().and_then(|s| {
            let capitalized = {
                let mut chars = s.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().chain(chars.flat_map(|c| c.to_lowercase())).collect(),
                }
            };

            capitalized.parse::<crate::models::SymbolKind>()
                .ok()
                .or_else(|| {
                    log::debug!("Treating '{}' as unknown symbol kind for filtering", s);
                    Some(crate::models::SymbolKind::Unknown(s.to_string()))
                })
        });

        // Smart behavior: --kind implies --symbols
        let symbols_mode = params.symbols || kind.is_some();

        // Smart limit handling (same as CLI and MCP)
        let final_limit = if params.paths && params.limit.is_none() {
            None  // --paths without explicit limit means no limit
        } else if let Some(user_limit) = params.limit {
            Some(user_limit)  // Use user-specified limit
        } else {
            Some(100)  // Default: limit to 100 results for token efficiency
        };

        let filter = QueryFilter {
            language,
            kind,
            use_ast: false,
            use_regex: params.regex,
            limit: final_limit,
            symbols_mode,
            expand: params.expand,
            file_pattern: params.file,
            exact: params.exact,
            use_contains: params.contains,
            timeout_secs: params.timeout,
            glob_patterns: params.glob,
            exclude_patterns: params.exclude,
            paths_only: params.paths,
            offset: params.offset,
            force: params.force,
            suppress_output: true,  // HTTP API always returns JSON, suppress warnings
            include_dependencies: params.dependencies,
            ..Default::default()
        };

        match engine.search_with_metadata(&params.q, filter) {
            Ok(response) => Ok(Json(response)),
            Err(e) => {
                log::error!("Query error: {}", e);
                Err((StatusCode::INTERNAL_SERVER_ERROR, format!("Query failed: {}", e)))
            }
        }
    }

    // GET /stats endpoint
    async fn handle_stats_endpoint(
        State(state): State<Arc<AppState>>,
    ) -> Result<Json<crate::models::IndexStats>, (StatusCode, String)> {
        log::info!("Stats request");

        let cache = CacheManager::new(&state.cache_path);

        if !cache.exists() {
            return Err((StatusCode::NOT_FOUND, "No index found. Run 'rfx index' first.".to_string()));
        }

        match cache.stats() {
            Ok(stats) => Ok(Json(stats)),
            Err(e) => {
                log::error!("Stats error: {}", e);
                Err((StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to get stats: {}", e)))
            }
        }
    }

    // POST /index endpoint
    async fn handle_index_endpoint(
        State(state): State<Arc<AppState>>,
        Json(req): Json<IndexRequest>,
    ) -> Result<Json<crate::models::IndexStats>, (StatusCode, String)> {
        log::info!("Index request: force={}, languages={:?}", req.force, req.languages);

        let cache = CacheManager::new(&state.cache_path);

        if req.force {
            log::info!("Force rebuild requested, clearing existing cache");
            if let Err(e) = cache.clear() {
                return Err((StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to clear cache: {}", e)));
            }
        }

        // Parse language filters
        let lang_filters: Vec<Language> = req.languages
            .iter()
            .filter_map(|s| match s.to_lowercase().as_str() {
                "rust" | "rs" => Some(Language::Rust),
                "python" | "py" => Some(Language::Python),
                "javascript" | "js" => Some(Language::JavaScript),
                "typescript" | "ts" => Some(Language::TypeScript),
                "vue" => Some(Language::Vue),
                "svelte" => Some(Language::Svelte),
                "go" => Some(Language::Go),
                "java" => Some(Language::Java),
                "php" => Some(Language::PHP),
                "c" => Some(Language::C),
                "cpp" | "c++" => Some(Language::Cpp),
                _ => {
                    log::warn!("Unknown language: {}", s);
                    None
                }
            })
            .collect();

        let config = IndexConfig {
            languages: lang_filters,
            ..Default::default()
        };

        let indexer = Indexer::new(cache, config);
        let path = std::path::PathBuf::from(&state.cache_path);

        match indexer.index(&path, false) {
            Ok(stats) => Ok(Json(stats)),
            Err(e) => {
                log::error!("Index error: {}", e);
                Err((StatusCode::INTERNAL_SERVER_ERROR, format!("Indexing failed: {}", e)))
            }
        }
    }

    // Health check endpoint
    async fn handle_health() -> impl IntoResponse {
        (StatusCode::OK, "Reflex is running")
    }

    // Create shared state
    let state = Arc::new(AppState {
        cache_path: ".".to_string(),
    });

    // Configure CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build the router
    let app = Router::new()
        .route("/query", get(handle_query_endpoint))
        .route("/stats", get(handle_stats_endpoint))
        .route("/index", post(handle_index_endpoint))
        .route("/health", get(handle_health))
        .layer(cors)
        .with_state(state);

    // Bind to the specified address
    let addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&addr).await
        .map_err(|e| anyhow::anyhow!("Failed to bind to {}: {}", addr, e))?;

    log::info!("Server listening on {}", addr);

    // Run the server
    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("Server error: {}", e))?;

    Ok(())
}

/// Handle the `stats` subcommand
fn handle_stats(as_json: bool, pretty_json: bool) -> Result<()> {
    log::info!("Showing index statistics");

    let cache = CacheManager::new(".");

    if !cache.exists() {
        anyhow::bail!(
            "No index found in current directory.\n\
             \n\
             Run 'rfx index' to build the code search index first.\n\
             This will scan all files in the current directory and create a .reflex/ cache.\n\
             \n\
             Example:\n\
             $ rfx index          # Index current directory\n\
             $ rfx stats          # Show index statistics"
        );
    }

    let stats = cache.stats()?;

    if as_json {
        let json_output = if pretty_json {
            serde_json::to_string_pretty(&stats)?
        } else {
            serde_json::to_string(&stats)?
        };
        println!("{}", json_output);
    } else {
        println!("Reflex Index Statistics");
        println!("=======================");

        // Show git branch info if in git repo, or (None) if not
        let root = std::env::current_dir()?;
        if crate::git::is_git_repo(&root) {
            match crate::git::get_git_state(&root) {
                Ok(git_state) => {
                    let dirty_indicator = if git_state.dirty { " (uncommitted changes)" } else { " (clean)" };
                    println!("Branch:         {}@{}{}",
                             git_state.branch,
                             &git_state.commit[..7],
                             dirty_indicator);

                    // Check if current branch is indexed
                    match cache.get_branch_info(&git_state.branch) {
                        Ok(branch_info) => {
                            if branch_info.commit_sha != git_state.commit {
                                println!("                ⚠️  Index commit mismatch (indexed: {})",
                                         &branch_info.commit_sha[..7]);
                            }
                            if git_state.dirty && !branch_info.is_dirty {
                                println!("                ⚠️  Uncommitted changes not indexed");
                            }
                        }
                        Err(_) => {
                            println!("                ⚠️  Branch not indexed");
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Failed to get git state: {}", e);
                }
            }
        } else {
            // Not a git repository - show (None)
            println!("Branch:         (None)");
        }

        println!("Files indexed:  {}", stats.total_files);
        println!("Index size:     {} bytes", stats.index_size_bytes);
        println!("Last updated:   {}", stats.last_updated);

        // Display language breakdown if we have indexed files
        if !stats.files_by_language.is_empty() {
            println!("\nFiles by language:");

            // Sort languages by count (descending) for consistent output
            let mut lang_vec: Vec<_> = stats.files_by_language.iter().collect();
            lang_vec.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));

            // Calculate column widths
            let max_lang_len = lang_vec.iter().map(|(lang, _)| lang.len()).max().unwrap_or(8);
            let lang_width = max_lang_len.max(8); // At least "Language" header width

            // Print table header
            println!("  {:<width$}  Files  Lines", "Language", width = lang_width);
            println!("  {}  -----  -------", "-".repeat(lang_width));

            // Print rows
            for (language, file_count) in lang_vec {
                let line_count = stats.lines_by_language.get(language).copied().unwrap_or(0);
                println!("  {:<width$}  {:5}  {:7}",
                    language, file_count, line_count,
                    width = lang_width);
            }
        }
    }

    Ok(())
}

/// Handle the `clear` subcommand
fn handle_clear(skip_confirm: bool) -> Result<()> {
    let cache = CacheManager::new(".");

    if !cache.exists() {
        println!("No cache to clear.");
        return Ok(());
    }

    if !skip_confirm {
        println!("This will delete the local Reflex cache at: {:?}", cache.path());
        print!("Are you sure? [y/N] ");
        use std::io::{self, Write};
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Cancelled.");
            return Ok(());
        }
    }

    cache.clear()?;
    println!("Cache cleared successfully.");

    Ok(())
}

/// Handle the `list-files` subcommand
fn handle_list_files(as_json: bool, pretty_json: bool) -> Result<()> {
    let cache = CacheManager::new(".");

    if !cache.exists() {
        anyhow::bail!(
            "No index found in current directory.\n\
             \n\
             Run 'rfx index' to build the code search index first.\n\
             This will scan all files in the current directory and create a .reflex/ cache.\n\
             \n\
             Example:\n\
             $ rfx index            # Index current directory\n\
             $ rfx list-files       # List indexed files"
        );
    }

    let files = cache.list_files()?;

    if as_json {
        let json_output = if pretty_json {
            serde_json::to_string_pretty(&files)?
        } else {
            serde_json::to_string(&files)?
        };
        println!("{}", json_output);
    } else if files.is_empty() {
        println!("No files indexed yet.");
    } else {
        println!("Indexed Files ({} total):", files.len());
        println!();
        for file in files {
            println!("  {} ({})",
                     file.path,
                     file.language);
        }
    }

    Ok(())
}

/// Handle the `watch` subcommand
fn handle_watch(path: PathBuf, debounce_ms: u64, quiet: bool) -> Result<()> {
    log::info!("Starting watch mode for {:?}", path);

    // Validate debounce range (5s - 30s)
    if !(5000..=30000).contains(&debounce_ms) {
        anyhow::bail!(
            "Debounce must be between 5000ms (5s) and 30000ms (30s). Got: {}ms",
            debounce_ms
        );
    }

    if !quiet {
        println!("Starting Reflex watch mode...");
        println!("  Directory: {}", path.display());
        println!("  Debounce: {}ms ({}s)", debounce_ms, debounce_ms / 1000);
        println!("  Press Ctrl+C to stop.\n");
    }

    // Setup cache
    let cache = CacheManager::new(&path);

    // Initial index if cache doesn't exist
    if !cache.exists() {
        if !quiet {
            println!("No index found, running initial index...");
        }
        let config = IndexConfig::default();
        let indexer = Indexer::new(cache, config);
        indexer.index(&path, !quiet)?;
        if !quiet {
            println!("Initial index complete. Now watching for changes...\n");
        }
    }

    // Create indexer for watcher
    let cache = CacheManager::new(&path);
    let config = IndexConfig::default();
    let indexer = Indexer::new(cache, config);

    // Start watcher
    let watch_config = crate::watcher::WatchConfig {
        debounce_ms,
        quiet,
    };

    crate::watcher::watch(&path, indexer, watch_config)?;

    Ok(())
}

/// Handle interactive mode (default when no command is given)
fn handle_interactive() -> Result<()> {
    log::info!("Launching interactive mode");
    crate::interactive::run_interactive()
}

/// Handle the `mcp` subcommand
fn handle_mcp() -> Result<()> {
    log::info!("Starting MCP server");
    crate::mcp::run_mcp_server()
}

/// Handle the internal `index-symbols-internal` command
fn handle_index_symbols_internal(cache_dir: PathBuf) -> Result<()> {
    let mut indexer = crate::background_indexer::BackgroundIndexer::new(&cache_dir)?;
    indexer.run()?;
    Ok(())
}

/// Handle the `analyze` subcommand
#[allow(clippy::too_many_arguments)]
fn handle_analyze(
    circular: bool,
    hotspots: bool,
    min_dependents: usize,
    unused: bool,
    islands: bool,
    min_island_size: usize,
    max_island_size: Option<usize>,
    format: String,
    as_json: bool,
    pretty_json: bool,
    count_only: bool,
    all: bool,
    plain: bool,
    _glob_patterns: Vec<String>,
    _exclude_patterns: Vec<String>,
    _force: bool,
    limit: Option<usize>,
    offset: Option<usize>,
    sort: Option<String>,
) -> Result<()> {
    use crate::dependency::DependencyIndex;

    log::info!("Starting analyze command");

    let cache = CacheManager::new(".");

    if !cache.exists() {
        anyhow::bail!(
            "No index found in current directory.\n\
             \n\
             Run 'rfx index' to build the code search index first.\n\
             \n\
             Example:\n\
             $ rfx index             # Index current directory\n\
             $ rfx analyze           # Run dependency analysis"
        );
    }

    let deps_index = DependencyIndex::new(cache);

    // JSON mode overrides format
    let format = if as_json { "json" } else { &format };

    // Smart limit handling for analyze commands (default: 200 per page)
    let final_limit = if all {
        None  // --all means no limit
    } else if let Some(user_limit) = limit {
        Some(user_limit)  // Use user-specified limit
    } else {
        Some(200)  // Default: limit to 200 results per page for token efficiency
    };

    // If no specific flags, show summary
    if !circular && !hotspots && !unused && !islands {
        return handle_analyze_summary(&deps_index, min_dependents, count_only, as_json, pretty_json);
    }

    // Run specific analyses based on flags
    if circular {
        handle_deps_circular(&deps_index, format, pretty_json, final_limit, offset, count_only, plain, sort.clone())?;
    }

    if hotspots {
        handle_deps_hotspots(&deps_index, format, pretty_json, final_limit, offset, min_dependents, count_only, plain, sort.clone())?;
    }

    if unused {
        handle_deps_unused(&deps_index, format, pretty_json, final_limit, offset, count_only, plain)?;
    }

    if islands {
        handle_deps_islands(&deps_index, format, pretty_json, final_limit, offset, min_island_size, max_island_size, count_only, plain, sort.clone())?;
    }

    Ok(())
}

/// Handle analyze summary (default --analyze behavior)
fn handle_analyze_summary(
    deps_index: &crate::dependency::DependencyIndex,
    min_dependents: usize,
    count_only: bool,
    as_json: bool,
    pretty_json: bool,
) -> Result<()> {
    // Gather counts
    let cycles = deps_index.detect_circular_dependencies()?;
    let hotspots = deps_index.find_hotspots(None, min_dependents)?;
    let unused = deps_index.find_unused_files()?;
    let all_islands = deps_index.find_islands()?;

    if as_json {
        // JSON output
        let summary = serde_json::json!({
            "circular_dependencies": cycles.len(),
            "hotspots": hotspots.len(),
            "unused_files": unused.len(),
            "islands": all_islands.len(),
            "min_dependents": min_dependents,
        });

        let json_str = if pretty_json {
            serde_json::to_string_pretty(&summary)?
        } else {
            serde_json::to_string(&summary)?
        };
        println!("{}", json_str);
    } else if count_only {
        // Just show counts without any extra formatting
        println!("{} circular dependencies", cycles.len());
        println!("{} hotspots ({}+ dependents)", hotspots.len(), min_dependents);
        println!("{} unused files", unused.len());
        println!("{} islands", all_islands.len());
    } else {
        // Full summary with headers and suggestions
        println!("Dependency Analysis Summary\n");

        // Circular dependencies
        println!("Circular Dependencies: {} cycle(s)", cycles.len());

        // Hotspots
        println!("Hotspots: {} file(s) with {}+ dependents", hotspots.len(), min_dependents);

        // Unused
        println!("Unused Files: {} file(s)", unused.len());

        // Islands
        println!("Islands: {} disconnected component(s)", all_islands.len());

        println!("\nUse specific flags for detailed results:");
        println!("  rfx analyze --circular");
        println!("  rfx analyze --hotspots");
        println!("  rfx analyze --unused");
        println!("  rfx analyze --islands");
    }

    Ok(())
}

/// Handle the `deps` subcommand
fn handle_deps(
    file: PathBuf,
    reverse: bool,
    depth: usize,
    format: String,
    as_json: bool,
    pretty_json: bool,
) -> Result<()> {
    use crate::dependency::DependencyIndex;

    log::info!("Starting deps command");

    let cache = CacheManager::new(".");

    if !cache.exists() {
        anyhow::bail!(
            "No index found in current directory.\n\
             \n\
             Run 'rfx index' to build the code search index first.\n\
             \n\
             Example:\n\
             $ rfx index          # Index current directory\n\
             $ rfx deps <file>    # Analyze dependencies"
        );
    }

    let deps_index = DependencyIndex::new(cache);

    // JSON mode overrides format
    let format = if as_json { "json" } else { &format };

    // Convert file path to string
    let file_str = file.to_string_lossy().to_string();

    // Get file ID
    let file_id = deps_index.get_file_id_by_path(&file_str)?
        .ok_or_else(|| anyhow::anyhow!("File '{}' not found in index", file_str))?;

    if reverse {
        // Show dependents (who imports this file)
        let dependents = deps_index.get_dependents(file_id)?;
        let paths = deps_index.get_file_paths(&dependents)?;

        match format.as_ref() {
            "json" => {
                let output: Vec<_> = dependents.iter()
                    .filter_map(|id| paths.get(id).map(|path| serde_json::json!({
                        "file_id": id,
                        "path": path,
                    })))
                    .collect();

                let json_str = if pretty_json {
                    serde_json::to_string_pretty(&output)?
                } else {
                    serde_json::to_string(&output)?
                };
                println!("{}", json_str);
                eprintln!("Found {} files that import {}", dependents.len(), file_str);
            }
            "tree" => {
                println!("Files that import {}:", file_str);
                for (id, path) in &paths {
                    if dependents.contains(id) {
                        println!("  └─ {}", path);
                    }
                }
                eprintln!("\nFound {} dependents", dependents.len());
            }
            "table" => {
                println!("ID     Path");
                println!("-----  ----");
                for id in &dependents {
                    if let Some(path) = paths.get(id) {
                        println!("{:<5}  {}", id, path);
                    }
                }
                eprintln!("\nFound {} dependents", dependents.len());
            }
            _ => {
                anyhow::bail!("Unknown format '{}'. Supported: json, tree, table, dot", format);
            }
        }
    } else {
        // Show dependencies (what this file imports)
        if depth == 1 {
            // Direct dependencies only
            let deps = deps_index.get_dependencies(file_id)?;

            match format.as_ref() {
                "json" => {
                    let output: Vec<_> = deps.iter()
                        .map(|dep| serde_json::json!({
                            "imported_path": dep.imported_path,
                            "resolved_file_id": dep.resolved_file_id,
                            "import_type": match dep.import_type {
                                crate::models::ImportType::Internal => "internal",
                                crate::models::ImportType::External => "external",
                                crate::models::ImportType::Stdlib => "stdlib",
                            },
                            "line": dep.line_number,
                            "symbols": dep.imported_symbols,
                        }))
                        .collect();

                    let json_str = if pretty_json {
                        serde_json::to_string_pretty(&output)?
                    } else {
                        serde_json::to_string(&output)?
                    };
                    println!("{}", json_str);
                    eprintln!("Found {} dependencies for {}", deps.len(), file_str);
                }
                "tree" => {
                    println!("Dependencies of {}:", file_str);
                    for dep in &deps {
                        let type_label = match dep.import_type {
                            crate::models::ImportType::Internal => "[internal]",
                            crate::models::ImportType::External => "[external]",
                            crate::models::ImportType::Stdlib => "[stdlib]",
                        };
                        println!("  └─ {} {} (line {})", dep.imported_path, type_label, dep.line_number);
                    }
                    eprintln!("\nFound {} dependencies", deps.len());
                }
                "table" => {
                    println!("Path                          Type       Line");
                    println!("----------------------------  ---------  ----");
                    for dep in &deps {
                        let type_str = match dep.import_type {
                            crate::models::ImportType::Internal => "internal",
                            crate::models::ImportType::External => "external",
                            crate::models::ImportType::Stdlib => "stdlib",
                        };
                        println!("{:<28}  {:<9}  {}", dep.imported_path, type_str, dep.line_number);
                    }
                    eprintln!("\nFound {} dependencies", deps.len());
                }
                _ => {
                    anyhow::bail!("Unknown format '{}'. Supported: json, tree, table, dot", format);
                }
            }
        } else {
            // Transitive dependencies (depth > 1)
            let transitive = deps_index.get_transitive_deps(file_id, depth)?;
            let file_ids: Vec<_> = transitive.keys().copied().collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            match format.as_ref() {
                "json" => {
                    let output: Vec<_> = transitive.iter()
                        .filter_map(|(id, d)| {
                            paths.get(id).map(|path| serde_json::json!({
                                "file_id": id,
                                "path": path,
                                "depth": d,
                            }))
                        })
                        .collect();

                    let json_str = if pretty_json {
                        serde_json::to_string_pretty(&output)?
                    } else {
                        serde_json::to_string(&output)?
                    };
                    println!("{}", json_str);
                    eprintln!("Found {} transitive dependencies (depth {})", transitive.len(), depth);
                }
                "tree" => {
                    println!("Transitive dependencies of {} (depth {}):", file_str, depth);
                    // Group by depth for tree display
                    let mut by_depth: std::collections::HashMap<usize, Vec<i64>> = std::collections::HashMap::new();
                    for (id, d) in &transitive {
                        by_depth.entry(*d).or_insert_with(Vec::new).push(*id);
                    }

                    for depth_level in 0..=depth {
                        if let Some(ids) = by_depth.get(&depth_level) {
                            let indent = "  ".repeat(depth_level);
                            for id in ids {
                                if let Some(path) = paths.get(id) {
                                    if depth_level == 0 {
                                        println!("{}{} (self)", indent, path);
                                    } else {
                                        println!("{}└─ {}", indent, path);
                                    }
                                }
                            }
                        }
                    }
                    eprintln!("\nFound {} transitive dependencies", transitive.len());
                }
                "table" => {
                    println!("Depth  File ID  Path");
                    println!("-----  -------  ----");
                    let mut sorted: Vec<_> = transitive.iter().collect();
                    sorted.sort_by_key(|(_, d)| *d);
                    for (id, d) in sorted {
                        if let Some(path) = paths.get(id) {
                            println!("{:<5}  {:<7}  {}", d, id, path);
                        }
                    }
                    eprintln!("\nFound {} transitive dependencies", transitive.len());
                }
                _ => {
                    anyhow::bail!("Unknown format '{}'. Supported: json, tree, table, dot", format);
                }
            }
        }
    }

    Ok(())
}

/// Handle the `ask` command
fn handle_ask(
    question: Option<String>,
    _auto_execute: bool,
    provider_override: Option<String>,
    as_json: bool,
    pretty_json: bool,
    additional_context: Option<String>,
    configure: bool,
    agentic: bool,
    max_iterations: usize,
    no_eval: bool,
    show_reasoning: bool,
    verbose: bool,
    quiet: bool,
    answer: bool,
    interactive: bool,
    debug: bool,
) -> Result<()> {
    // If --configure flag is set, launch the configuration wizard (deprecated)
    if configure {
        eprintln!("Note: --configure is deprecated, use `rfx llm config` instead");
        log::info!("Launching configuration wizard");
        return crate::semantic::run_configure_wizard();
    }

    // Check if any API key is configured before allowing rfx ask to run
    if !crate::semantic::is_any_api_key_configured() {
        anyhow::bail!(
            "No API key configured.\n\
             \n\
             Please run 'rfx ask --configure' to set up your API provider and key.\n\
             \n\
             Alternatively, you can set an environment variable:\n\
             - OPENAI_API_KEY\n\
             - ANTHROPIC_API_KEY\n\
             - OPENROUTER_API_KEY"
        );
    }

    // If no question provided and not in configure mode, default to interactive mode
    // If --interactive flag is set, launch interactive chat mode (TUI)
    if interactive || question.is_none() {
        log::info!("Launching interactive chat mode");
        let cache = CacheManager::new(".");

        if !cache.exists() {
            anyhow::bail!(
                "No index found in current directory.\n\
                 \n\
                 Run 'rfx index' to build the code search index first.\n\
                 \n\
                 Example:\n\
                 $ rfx index                          # Index current directory\n\
                 $ rfx ask                            # Launch interactive chat"
            );
        }

        return crate::semantic::run_chat_mode(cache, provider_override, None);
    }

    // At this point, question must be Some
    let question = question.unwrap();

    log::info!("Starting ask command");

    let cache = CacheManager::new(".");

    if !cache.exists() {
        anyhow::bail!(
            "No index found in current directory.\n\
             \n\
             Run 'rfx index' to build the code search index first.\n\
             \n\
             Example:\n\
             $ rfx index                          # Index current directory\n\
             $ rfx ask \"Find all TODOs\"          # Ask questions"
        );
    }

    // Create a tokio runtime for async operations
    let runtime = tokio::runtime::Runtime::new()
        .context("Failed to create async runtime")?;

    // Force quiet mode for JSON output (machine-readable, no UI output)
    let quiet = quiet || as_json;

    // Create optional spinner (skip entirely in JSON mode for clean machine-readable output)
    let spinner = if !as_json {
        let s = ProgressBar::new_spinner();
        s.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .unwrap()
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
        );
        s.set_message("Generating queries...".to_string());
        s.enable_steady_tick(std::time::Duration::from_millis(80));
        Some(s)
    } else {
        None
    };

    let (queries, results, total_count, count_only, gathered_context) = if agentic {
        // Agentic mode: multi-step reasoning with context gathering

        // Wrap spinner in Arc<Mutex<>> for sharing with reporter (non-quiet mode)
        let spinner_shared = if !quiet {
            spinner.as_ref().map(|s| Arc::new(Mutex::new(s.clone())))
        } else {
            None
        };

        // Create reporter based on flags
        let reporter: Box<dyn crate::semantic::AgenticReporter> = if quiet {
            Box::new(crate::semantic::QuietReporter)
        } else {
            Box::new(crate::semantic::ConsoleReporter::new(show_reasoning, verbose, debug, spinner_shared))
        };

        // Set initial spinner message and enable ticking
        if let Some(ref s) = spinner {
            s.set_message("Starting agentic mode...".to_string());
            s.enable_steady_tick(std::time::Duration::from_millis(80));
        }

        let agentic_config = crate::semantic::AgenticConfig {
            max_iterations,
            max_tools_per_phase: 5,
            enable_evaluation: !no_eval,
            eval_config: Default::default(),
            provider_override: provider_override.clone(),
            model_override: None,
            show_reasoning,
            verbose,
            debug,
        };

        let agentic_response = runtime.block_on(async {
            crate::semantic::run_agentic_loop(&question, &cache, agentic_config, &*reporter).await
        }).context("Failed to run agentic loop")?;

        // Clear spinner after agentic loop completes
        if let Some(ref s) = spinner {
            s.finish_and_clear();
        }

        // Clear ephemeral output (Phase 5 evaluation) before showing final results
        if !as_json {
            reporter.clear_all();
        }

        log::info!("Agentic loop completed: {} queries generated", agentic_response.queries.len());

        // Destructure AgenticQueryResponse into tuple (preserve gathered_context)
        let count_only_mode = agentic_response.total_count.is_none();
        let count = agentic_response.total_count.unwrap_or(0);
        (agentic_response.queries, agentic_response.results, count, count_only_mode, agentic_response.gathered_context)
    } else {
        // Standard mode: single LLM call + execution
        if let Some(ref s) = spinner {
            s.set_message("Generating queries...".to_string());
            s.enable_steady_tick(std::time::Duration::from_millis(80));
        }

        let semantic_response = runtime.block_on(async {
            crate::semantic::ask_question(&question, &cache, provider_override.clone(), additional_context, debug).await
        }).context("Failed to generate semantic queries")?;

        if let Some(ref s) = spinner {
            s.finish_and_clear();
        }
        log::info!("LLM generated {} queries", semantic_response.queries.len());

        // Execute queries for standard mode
        let (exec_results, exec_total, exec_count_only) = runtime.block_on(async {
            crate::semantic::execute_queries(semantic_response.queries.clone(), &cache).await
        }).context("Failed to execute queries")?;

        (semantic_response.queries, exec_results, exec_total, exec_count_only, None)
    };

    // Generate conversational answer if --answer flag is set
    let generated_answer = if answer {
        // Show spinner while generating answer
        let answer_spinner = if !as_json {
            let s = ProgressBar::new_spinner();
            s.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.cyan} {msg}")
                    .unwrap()
                    .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
            );
            s.set_message("Generating answer...".to_string());
            s.enable_steady_tick(std::time::Duration::from_millis(80));
            Some(s)
        } else {
            None
        };

        // Initialize provider for answer generation
        let mut config = crate::semantic::config::load_config(cache.path())?;
        if let Some(provider) = &provider_override {
            config.provider = provider.clone();
        }
        let api_key = crate::semantic::config::get_api_key(&config.provider)?;
        let model = if config.model.is_some() {
            config.model.clone()
        } else {
            crate::semantic::config::get_user_model(&config.provider)
        };
        let provider_instance = crate::semantic::providers::create_provider(
            &config.provider,
            api_key,
            model,
            crate::semantic::config::get_provider_options(&config.provider),
        )?;

        // Extract codebase context (always available metadata: languages, file counts, directories)
        let codebase_context_str = crate::semantic::context::CodebaseContext::extract(&cache)
            .ok()
            .map(|ctx| ctx.to_prompt_string());

        // Generate answer (with optional gathered context from agentic mode + codebase context)
        let answer_result = runtime.block_on(async {
            crate::semantic::generate_answer(
                &question,
                &results,
                total_count,
                gathered_context.as_deref(),
                codebase_context_str.as_deref(),
                &*provider_instance,
            ).await
        }).context("Failed to generate answer")?;

        if let Some(s) = answer_spinner {
            s.finish_and_clear();
        }

        Some(answer_result)
    } else {
        None
    };

    // Output in JSON format if requested
    if as_json {
        // Build AgenticQueryResponse for JSON output (includes both queries and results)
        let json_response = crate::semantic::AgenticQueryResponse {
            queries: queries.clone(),
            results: results.clone(),
            total_count: if count_only { None } else { Some(total_count) },
            gathered_context: gathered_context.clone(),
            tools_executed: None, // No tools in non-agentic mode
            answer: generated_answer,
        };

        let json_str = if pretty_json {
            serde_json::to_string_pretty(&json_response)?
        } else {
            serde_json::to_string(&json_response)?
        };
        println!("{}", json_str);
        return Ok(());
    }

    // Display generated queries with color (unless in answer mode)
    if !answer {
        println!("\n{}", "Generated Queries:".bold().cyan());
        println!("{}", "==================".cyan());
        for (idx, query_cmd) in queries.iter().enumerate() {
            println!(
                "{}. {} {} {}",
                (idx + 1).to_string().bright_white().bold(),
                format!("[order: {}, merge: {}]", query_cmd.order, query_cmd.merge).dimmed(),
                "rfx".bright_green().bold(),
                query_cmd.command.bright_white()
            );
        }
        println!();
    }

    // Note: queries already executed in both modes above
    // Agentic mode: executed during run_agentic_loop
    // Standard mode: executed after ask_question

    // Display answer or results
    println!();
    if let Some(answer_text) = generated_answer {
        // Answer mode: show the conversational answer
        println!("{}", "Answer:".bold().green());
        println!("{}", "=======".green());
        println!();

        // Render markdown if it looks like markdown, otherwise print as-is
        termimad::print_text(&answer_text);
        println!();

        // Show summary of results used
        if !results.is_empty() {
            println!(
                "{}",
                format!(
                    "(Based on {} matches across {} files)",
                    total_count,
                    results.len()
                ).dimmed()
            );
        }
    } else {
        // Standard mode: show raw results
        if count_only {
            // Count-only mode: just show the total count (matching direct CLI behavior)
            println!("{} {}", "Found".bright_green().bold(), format!("{} results", total_count).bright_white().bold());
        } else if results.is_empty() {
            println!("{}", "No results found.".yellow());
        } else {
            println!(
                "{} {} {} {} {}",
                "Found".bright_green().bold(),
                total_count.to_string().bright_white().bold(),
                "total results across".dimmed(),
                results.len().to_string().bright_white().bold(),
                "files:".dimmed()
            );
            println!();

            for file_group in &results {
                println!("{}:", file_group.path.bright_cyan().bold());
                for match_result in &file_group.matches {
                    println!(
                        "  {} {}-{}: {}",
                        "Line".dimmed(),
                        match_result.span.start_line.to_string().bright_yellow(),
                        match_result.span.end_line.to_string().bright_yellow(),
                        match_result.preview.lines().next().unwrap_or("")
                    );
                }
                println!();
            }
        }
    }

    Ok(())
}

/// Handle the `context` command
fn handle_context(
    structure: bool,
    path: Option<String>,
    file_types: bool,
    project_type: bool,
    framework: bool,
    entry_points: bool,
    test_layout: bool,
    config_files: bool,
    depth: usize,
    json: bool,
) -> Result<()> {
    let cache = CacheManager::new(".");

    if !cache.exists() {
        anyhow::bail!(
            "No index found in current directory.\n\
             \n\
             Run 'rfx index' to build the code search index first.\n\
             \n\
             Example:\n\
             $ rfx index                  # Index current directory\n\
             $ rfx context                # Generate context"
        );
    }

    // Build context options
    let opts = crate::context::ContextOptions {
        structure,
        path,
        file_types,
        project_type,
        framework,
        entry_points,
        test_layout,
        config_files,
        depth,
        json,
    };

    // Generate context
    let context_output = crate::context::generate_context(&cache, &opts)
        .context("Failed to generate codebase context")?;

    // Print output
    println!("{}", context_output);

    Ok(())
}

/// Handle --circular flag (detect cycles)
fn handle_deps_circular(
    deps_index: &crate::dependency::DependencyIndex,
    format: &str,
    pretty_json: bool,
    limit: Option<usize>,
    offset: Option<usize>,
    count_only: bool,
    _plain: bool,
    sort: Option<String>,
) -> Result<()> {
    let mut all_cycles = deps_index.detect_circular_dependencies()?;

    // Apply sorting (default: descending - longest cycles first)
    let sort_order = sort.as_deref().unwrap_or("desc");
    match sort_order {
        "asc" => {
            // Ascending: shortest cycles first
            all_cycles.sort_by_key(|cycle| cycle.len());
        }
        "desc" => {
            // Descending: longest cycles first (default)
            all_cycles.sort_by_key(|cycle| std::cmp::Reverse(cycle.len()));
        }
        _ => {
            anyhow::bail!("Invalid sort order '{}'. Supported: asc, desc", sort_order);
        }
    }

    let total_count = all_cycles.len();

    if count_only {
        println!("Found {} circular dependencies", total_count);
        return Ok(());
    }

    if all_cycles.is_empty() {
        println!("No circular dependencies found.");
        return Ok(());
    }

    // Apply offset pagination
    let offset_val = offset.unwrap_or(0);
    let mut cycles: Vec<_> = all_cycles.into_iter().skip(offset_val).collect();

    // Apply limit
    if let Some(lim) = limit {
        cycles.truncate(lim);
    }

    if cycles.is_empty() {
        println!("No circular dependencies found at offset {}.", offset_val);
        return Ok(());
    }

    let count = cycles.len();
    let has_more = offset_val + count < total_count;

    match format {
        "json" => {
            let file_ids: Vec<i64> = cycles.iter().flat_map(|c| c.iter()).copied().collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            let results: Vec<_> = cycles.iter()
                .map(|cycle| {
                    let cycle_paths: Vec<_> = cycle.iter()
                        .filter_map(|id| paths.get(id).cloned())
                        .collect();
                    serde_json::json!({
                        "paths": cycle_paths,
                    })
                })
                .collect();

            let output = serde_json::json!({
                "pagination": {
                    "total": total_count,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit,
                    "has_more": has_more,
                },
                "results": results,
            });

            let json_str = if pretty_json {
                serde_json::to_string_pretty(&output)?
            } else {
                serde_json::to_string(&output)?
            };
            println!("{}", json_str);
            if total_count > count {
                eprintln!("Found {} circular dependencies ({} total)", count, total_count);
            } else {
                eprintln!("Found {} circular dependencies", count);
            }
        }
        "tree" => {
            println!("Circular Dependencies Found:");
            let file_ids: Vec<i64> = cycles.iter().flat_map(|c| c.iter()).copied().collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            for (idx, cycle) in cycles.iter().enumerate() {
                println!("\nCycle {}:", idx + 1);
                for id in cycle {
                    if let Some(path) = paths.get(id) {
                        println!("  → {}", path);
                    }
                }
                // Show cycle completion
                if let Some(first_id) = cycle.first() {
                    if let Some(path) = paths.get(first_id) {
                        println!("  → {} (cycle completes)", path);
                    }
                }
            }
            if total_count > count {
                eprintln!("\nFound {} cycles ({} total)", count, total_count);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else {
                eprintln!("\nFound {} cycles", count);
            }
        }
        "table" => {
            println!("Cycle  Files in Cycle");
            println!("-----  --------------");
            let file_ids: Vec<i64> = cycles.iter().flat_map(|c| c.iter()).copied().collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            for (idx, cycle) in cycles.iter().enumerate() {
                let cycle_str = cycle.iter()
                    .filter_map(|id| paths.get(id).map(|p| p.as_str()))
                    .collect::<Vec<_>>()
                    .join(" → ");
                println!("{:<5}  {}", idx + 1, cycle_str);
            }
            if total_count > count {
                eprintln!("\nFound {} cycles ({} total)", count, total_count);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else {
                eprintln!("\nFound {} cycles", count);
            }
        }
        _ => {
            anyhow::bail!("Unknown format '{}'. Supported: json, tree, table", format);
        }
    }

    Ok(())
}

/// Handle --hotspots flag (most-imported files)
fn handle_deps_hotspots(
    deps_index: &crate::dependency::DependencyIndex,
    format: &str,
    pretty_json: bool,
    limit: Option<usize>,
    offset: Option<usize>,
    min_dependents: usize,
    count_only: bool,
    _plain: bool,
    sort: Option<String>,
) -> Result<()> {
    // Get all hotspots without limit first to track total count
    let mut all_hotspots = deps_index.find_hotspots(None, min_dependents)?;

    // Apply sorting (default: descending - most imports first)
    let sort_order = sort.as_deref().unwrap_or("desc");
    match sort_order {
        "asc" => {
            // Ascending: least imports first
            all_hotspots.sort_by(|a, b| a.1.cmp(&b.1));
        }
        "desc" => {
            // Descending: most imports first (default)
            all_hotspots.sort_by(|a, b| b.1.cmp(&a.1));
        }
        _ => {
            anyhow::bail!("Invalid sort order '{}'. Supported: asc, desc", sort_order);
        }
    }

    let total_count = all_hotspots.len();

    if count_only {
        println!("Found {} hotspots with {}+ dependents", total_count, min_dependents);
        return Ok(());
    }

    if all_hotspots.is_empty() {
        println!("No hotspots found.");
        return Ok(());
    }

    // Apply offset pagination
    let offset_val = offset.unwrap_or(0);
    let mut hotspots: Vec<_> = all_hotspots.into_iter().skip(offset_val).collect();

    // Apply limit
    if let Some(lim) = limit {
        hotspots.truncate(lim);
    }

    if hotspots.is_empty() {
        println!("No hotspots found at offset {}.", offset_val);
        return Ok(());
    }

    let count = hotspots.len();
    let has_more = offset_val + count < total_count;

    let file_ids: Vec<i64> = hotspots.iter().map(|(id, _)| *id).collect();
    let paths = deps_index.get_file_paths(&file_ids)?;

    match format {
        "json" => {
            let results: Vec<_> = hotspots.iter()
                .filter_map(|(id, import_count)| {
                    paths.get(id).map(|path| serde_json::json!({
                        "path": path,
                        "import_count": import_count,
                    }))
                })
                .collect();

            let output = serde_json::json!({
                "pagination": {
                    "total": total_count,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit,
                    "has_more": has_more,
                },
                "results": results,
            });

            let json_str = if pretty_json {
                serde_json::to_string_pretty(&output)?
            } else {
                serde_json::to_string(&output)?
            };
            println!("{}", json_str);
            if total_count > count {
                eprintln!("Found {} hotspots ({} total)", count, total_count);
            } else {
                eprintln!("Found {} hotspots", count);
            }
        }
        "tree" => {
            println!("Hotspots (Most-Imported Files):");
            for (idx, (id, import_count)) in hotspots.iter().enumerate() {
                if let Some(path) = paths.get(id) {
                    println!("  {}. {} ({} imports)", idx + 1, path, import_count);
                }
            }
            if total_count > count {
                eprintln!("\nFound {} hotspots ({} total)", count, total_count);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else {
                eprintln!("\nFound {} hotspots", count);
            }
        }
        "table" => {
            println!("Rank  Imports  File");
            println!("----  -------  ----");
            for (idx, (id, import_count)) in hotspots.iter().enumerate() {
                if let Some(path) = paths.get(id) {
                    println!("{:<4}  {:<7}  {}", idx + 1, import_count, path);
                }
            }
            if total_count > count {
                eprintln!("\nFound {} hotspots ({} total)", count, total_count);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else {
                eprintln!("\nFound {} hotspots", count);
            }
        }
        _ => {
            anyhow::bail!("Unknown format '{}'. Supported: json, tree, table", format);
        }
    }

    Ok(())
}

/// Handle --unused flag (orphaned files)
fn handle_deps_unused(
    deps_index: &crate::dependency::DependencyIndex,
    format: &str,
    pretty_json: bool,
    limit: Option<usize>,
    offset: Option<usize>,
    count_only: bool,
    _plain: bool,
) -> Result<()> {
    let all_unused = deps_index.find_unused_files()?;
    let total_count = all_unused.len();

    if count_only {
        println!("Found {} unused files", total_count);
        return Ok(());
    }

    if all_unused.is_empty() {
        println!("No unused files found (all files have incoming dependencies).");
        return Ok(());
    }

    // Apply offset pagination
    let offset_val = offset.unwrap_or(0);
    let mut unused: Vec<_> = all_unused.into_iter().skip(offset_val).collect();

    if unused.is_empty() {
        println!("No unused files found at offset {}.", offset_val);
        return Ok(());
    }

    // Apply limit
    if let Some(lim) = limit {
        unused.truncate(lim);
    }

    let count = unused.len();
    let has_more = offset_val + count < total_count;

    let paths = deps_index.get_file_paths(&unused)?;

    match format {
        "json" => {
            // Return flat array of path strings (no "path" key wrapper)
            let results: Vec<String> = unused.iter()
                .filter_map(|id| paths.get(id).cloned())
                .collect();

            let output = serde_json::json!({
                "pagination": {
                    "total": total_count,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit,
                    "has_more": has_more,
                },
                "results": results,
            });

            let json_str = if pretty_json {
                serde_json::to_string_pretty(&output)?
            } else {
                serde_json::to_string(&output)?
            };
            println!("{}", json_str);
            if total_count > count {
                eprintln!("Found {} unused files ({} total)", count, total_count);
            } else {
                eprintln!("Found {} unused files", count);
            }
        }
        "tree" => {
            println!("Unused Files (No Incoming Dependencies):");
            for (idx, id) in unused.iter().enumerate() {
                if let Some(path) = paths.get(id) {
                    println!("  {}. {}", idx + 1, path);
                }
            }
            if total_count > count {
                eprintln!("\nFound {} unused files ({} total)", count, total_count);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else {
                eprintln!("\nFound {} unused files", count);
            }
        }
        "table" => {
            println!("Path");
            println!("----");
            for id in &unused {
                if let Some(path) = paths.get(id) {
                    println!("{}", path);
                }
            }
            if total_count > count {
                eprintln!("\nFound {} unused files ({} total)", count, total_count);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else {
                eprintln!("\nFound {} unused files", count);
            }
        }
        _ => {
            anyhow::bail!("Unknown format '{}'. Supported: json, tree, table", format);
        }
    }

    Ok(())
}

/// Handle --islands flag (disconnected components)
fn handle_deps_islands(
    deps_index: &crate::dependency::DependencyIndex,
    format: &str,
    pretty_json: bool,
    limit: Option<usize>,
    offset: Option<usize>,
    min_island_size: usize,
    max_island_size: Option<usize>,
    count_only: bool,
    _plain: bool,
    sort: Option<String>,
) -> Result<()> {
    let all_islands = deps_index.find_islands()?;
    let total_components = all_islands.len();

    // Get total file count from the cache for percentage calculation
    let cache = deps_index.get_cache();
    let total_files = cache.stats()?.total_files as usize;

    // Calculate max_island_size default: min of 500 or 50% of total files
    let max_size = max_island_size.unwrap_or_else(|| {
        let fifty_percent = (total_files as f64 * 0.5) as usize;
        fifty_percent.min(500)
    });

    // Filter islands by size
    let mut islands: Vec<_> = all_islands.into_iter()
        .filter(|island| {
            let size = island.len();
            size >= min_island_size && size <= max_size
        })
        .collect();

    // Apply sorting (default: descending - largest islands first)
    let sort_order = sort.as_deref().unwrap_or("desc");
    match sort_order {
        "asc" => {
            // Ascending: smallest islands first
            islands.sort_by_key(|island| island.len());
        }
        "desc" => {
            // Descending: largest islands first (default)
            islands.sort_by_key(|island| std::cmp::Reverse(island.len()));
        }
        _ => {
            anyhow::bail!("Invalid sort order '{}'. Supported: asc, desc", sort_order);
        }
    }

    let filtered_count = total_components - islands.len();

    if count_only {
        if filtered_count > 0 {
            println!("Found {} islands (filtered {} of {} total components by size: {}-{})",
                islands.len(), filtered_count, total_components, min_island_size, max_size);
        } else {
            println!("Found {} islands", islands.len());
        }
        return Ok(());
    }

    // Apply offset pagination first
    let offset_val = offset.unwrap_or(0);
    if offset_val > 0 && offset_val < islands.len() {
        islands = islands.into_iter().skip(offset_val).collect();
    } else if offset_val >= islands.len() {
        if filtered_count > 0 {
            println!("No islands found at offset {} (filtered {} of {} total components by size: {}-{}).",
                offset_val, filtered_count, total_components, min_island_size, max_size);
        } else {
            println!("No islands found at offset {}.", offset_val);
        }
        return Ok(());
    }

    // Apply limit to number of islands
    if let Some(lim) = limit {
        islands.truncate(lim);
    }

    if islands.is_empty() {
        if filtered_count > 0 {
            println!("No islands found matching criteria (filtered {} of {} total components by size: {}-{}).",
                filtered_count, total_components, min_island_size, max_size);
        } else {
            println!("No islands found.");
        }
        return Ok(());
    }

    // Get all file IDs from all islands and track pagination
    let count = islands.len();
    let has_more = offset_val + count < total_components - filtered_count;

    let file_ids: Vec<i64> = islands.iter().flat_map(|island| island.iter()).copied().collect();
    let paths = deps_index.get_file_paths(&file_ids)?;

    match format {
        "json" => {
            let results: Vec<_> = islands.iter()
                .enumerate()
                .map(|(idx, island)| {
                    let island_paths: Vec<_> = island.iter()
                        .filter_map(|id| paths.get(id).cloned())
                        .collect();
                    serde_json::json!({
                        "island_id": idx + 1,
                        "size": island.len(),
                        "paths": island_paths,
                    })
                })
                .collect();

            let output = serde_json::json!({
                "pagination": {
                    "total": total_components - filtered_count,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit,
                    "has_more": has_more,
                },
                "results": results,
            });

            let json_str = if pretty_json {
                serde_json::to_string_pretty(&output)?
            } else {
                serde_json::to_string(&output)?
            };
            println!("{}", json_str);
            if filtered_count > 0 {
                eprintln!("Found {} islands (filtered {} of {} total components by size: {}-{})",
                    count, filtered_count, total_components, min_island_size, max_size);
            } else if total_components - filtered_count > count {
                eprintln!("Found {} islands ({} total)", count, total_components - filtered_count);
            } else {
                eprintln!("Found {} islands (disconnected components)", count);
            }
        }
        "tree" => {
            println!("Islands (Disconnected Components):");
            for (idx, island) in islands.iter().enumerate() {
                println!("\nIsland {} ({} files):", idx + 1, island.len());
                for id in island {
                    if let Some(path) = paths.get(id) {
                        println!("  ├─ {}", path);
                    }
                }
            }
            if filtered_count > 0 {
                eprintln!("\nFound {} islands (filtered {} of {} total components by size: {}-{})",
                    count, filtered_count, total_components, min_island_size, max_size);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else if total_components - filtered_count > count {
                eprintln!("\nFound {} islands ({} total)", count, total_components - filtered_count);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else {
                eprintln!("\nFound {} islands", count);
            }
        }
        "table" => {
            println!("Island  Size  Files");
            println!("------  ----  -----");
            for (idx, island) in islands.iter().enumerate() {
                let island_files = island.iter()
                    .filter_map(|id| paths.get(id).map(|p| p.as_str()))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("{:<6}  {:<4}  {}", idx + 1, island.len(), island_files);
            }
            if filtered_count > 0 {
                eprintln!("\nFound {} islands (filtered {} of {} total components by size: {}-{})",
                    count, filtered_count, total_components, min_island_size, max_size);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else if total_components - filtered_count > count {
                eprintln!("\nFound {} islands ({} total)", count, total_components - filtered_count);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else {
                eprintln!("\nFound {} islands", count);
            }
        }
        _ => {
            anyhow::bail!("Unknown format '{}'. Supported: json, tree, table", format);
        }
    }

    Ok(())
}

// ── Snapshot handlers ──────────────────────────────────────────

fn handle_snapshot_create() -> Result<()> {
    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }

    let info = pulse::snapshot::create_snapshot(&cache)?;
    eprintln!("Snapshot created: {}", info.id);
    eprintln!("  Files: {}, Lines: {}, Edges: {}", info.file_count, info.total_lines, info.edge_count);
    if let Some(branch) = &info.git_branch {
        eprintln!("  Branch: {}", branch);
    }

    // Run background GC
    let pulse_config = pulse::config::load_pulse_config(cache.path())?;
    let gc_report = pulse::snapshot::run_gc(&cache, &pulse_config.retention)?;
    if gc_report.removed > 0 {
        eprintln!("  GC: removed {} old snapshot(s)", gc_report.removed);
    }

    Ok(())
}

fn handle_snapshot_list(json: bool, pretty: bool) -> Result<()> {
    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }

    let snapshots = pulse::snapshot::list_snapshots(&cache)?;

    if json || pretty {
        let output = if pretty {
            serde_json::to_string_pretty(&snapshots)?
        } else {
            serde_json::to_string(&snapshots)?
        };
        println!("{}", output);
    } else {
        if snapshots.is_empty() {
            eprintln!("No snapshots found. Run `rfx snapshot` to create one.");
            return Ok(());
        }
        println!("{:<20} {:>6} {:>8} {:>6}  {}", "ID", "Files", "Lines", "Edges", "Branch");
        println!("{}", "-".repeat(60));
        for s in &snapshots {
            println!("{:<20} {:>6} {:>8} {:>6}  {}",
                s.id, s.file_count, s.total_lines, s.edge_count,
                s.git_branch.as_deref().unwrap_or("-"));
        }
        eprintln!("\n{} snapshot(s)", snapshots.len());
    }

    Ok(())
}

fn handle_snapshot_diff(
    baseline: Option<String>,
    current: Option<String>,
    json: bool,
    pretty: bool,
) -> Result<()> {
    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }

    let snapshots = pulse::snapshot::list_snapshots(&cache)?;
    let pulse_config = pulse::config::load_pulse_config(cache.path())?;

    let current_snapshot = match &current {
        Some(id) => snapshots.iter().find(|s| s.id == *id)
            .ok_or_else(|| anyhow::anyhow!("Snapshot '{}' not found", id))?,
        None => snapshots.first()
            .ok_or_else(|| anyhow::anyhow!("No snapshots found. Run `rfx snapshot` first."))?,
    };

    let baseline_snapshot = match &baseline {
        Some(id) => snapshots.iter().find(|s| s.id == *id)
            .ok_or_else(|| anyhow::anyhow!("Snapshot '{}' not found", id))?,
        None => snapshots.get(1)
            .ok_or_else(|| anyhow::anyhow!("Need at least 2 snapshots to diff. Run `rfx snapshot` again after making changes."))?,
    };

    let diff = pulse::diff::compute_diff(
        &baseline_snapshot.path,
        &current_snapshot.path,
        &pulse_config.thresholds,
    )?;

    if json || pretty {
        let output = if pretty {
            serde_json::to_string_pretty(&diff)?
        } else {
            serde_json::to_string(&diff)?
        };
        println!("{}", output);
    } else {
        let s = &diff.summary;
        println!("Diff: {} → {}", diff.baseline_id, diff.current_id);
        println!("  Files: +{} -{} ~{}", s.files_added, s.files_removed, s.files_modified);
        println!("  Edges: +{} -{}", s.edges_added, s.edges_removed);
        if !diff.threshold_alerts.is_empty() {
            println!("  Alerts: {}", diff.threshold_alerts.len());
            for alert in &diff.threshold_alerts {
                println!("    [{:?}] {}", alert.severity, alert.message);
            }
        }
    }

    Ok(())
}

fn handle_snapshot_gc(json: bool) -> Result<()> {
    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }

    let pulse_config = pulse::config::load_pulse_config(cache.path())?;
    let report = pulse::snapshot::run_gc(&cache, &pulse_config.retention)?;

    if json {
        println!("{}", serde_json::to_string(&report)?);
    } else {
        println!("GC complete: before {}, after {}, removed {}", report.snapshots_before, report.snapshots_after, report.removed);
    }

    Ok(())
}

// ── Pulse handlers ─────────────────────────────────────────────

fn handle_pulse_digest(
    baseline: Option<String>,
    current: Option<String>,
    no_llm: bool,
    json: bool,
    pretty: bool,
) -> Result<()> {
    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }

    let pulse_config = pulse::config::load_pulse_config(cache.path())?;

    // Auto-snapshot if index has changed since last snapshot
    let ensure_result = pulse::snapshot::ensure_snapshot(&cache, &pulse_config.retention)?;
    match &ensure_result {
        pulse::snapshot::EnsureSnapshotResult::Created(info) => {
            eprintln!("Auto-snapshot created: {} ({} files)", info.id, info.file_count);
        }
        pulse::snapshot::EnsureSnapshotResult::Reused(info) => {
            eprintln!("Using snapshot: {} (index unchanged)", info.id);
        }
    }

    let snapshots = pulse::snapshot::list_snapshots(&cache)?;

    let current_snapshot = match &current {
        Some(id) => snapshots.iter().find(|s| s.id == *id)
            .ok_or_else(|| anyhow::anyhow!("Snapshot '{}' not found", id))?,
        None => snapshots.first()
            .ok_or_else(|| anyhow::anyhow!("No snapshots found. Run `rfx snapshot` first."))?,
    };

    let snapshot_diff = match &baseline {
        Some(id) => {
            let bl = snapshots.iter().find(|s| s.id == *id)
                .ok_or_else(|| anyhow::anyhow!("Snapshot '{}' not found", id))?;
            Some(pulse::diff::compute_diff(&bl.path, &current_snapshot.path, &pulse_config.thresholds)?)
        }
        None => {
            snapshots.get(1).map(|bl| {
                pulse::diff::compute_diff(&bl.path, &current_snapshot.path, &pulse_config.thresholds)
            }).transpose()?
        }
    };

    // Create provider for standalone digest command
    let (provider, llm_cache) = if !no_llm {
        match pulse::narrate::create_pulse_provider() {
            Ok(p) => {
                eprintln!("LLM provider ready.");
                let c = pulse::llm_cache::LlmCache::new(cache.path());
                (Some(p), Some(c))
            }
            Err(e) => {
                eprintln!("LLM unavailable: {}", e);
                (None, None)
            }
        }
    } else {
        (None, None)
    };

    let digest = pulse::digest::generate_digest(
        snapshot_diff.as_ref(),
        current_snapshot,
        Some(&cache),
        no_llm,
        provider.as_ref().map(|p| p.as_ref()),
        llm_cache.as_ref(),
    )?;

    if json || pretty {
        let output = if pretty {
            serde_json::to_string_pretty(&digest)?
        } else {
            serde_json::to_string(&digest)?
        };
        println!("{}", output);
    } else {
        println!("{}", pulse::digest::render_markdown(&digest));
    }

    Ok(())
}

fn handle_pulse_wiki(no_llm: bool, output: Option<PathBuf>, json: bool) -> Result<()> {
    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }

    let pulse_config = pulse::config::load_pulse_config(cache.path())?;

    // Auto-snapshot if index has changed since last snapshot
    let ensure_result = pulse::snapshot::ensure_snapshot(&cache, &pulse_config.retention)?;
    match &ensure_result {
        pulse::snapshot::EnsureSnapshotResult::Created(info) => {
            eprintln!("Auto-snapshot created: {} ({} files)", info.id, info.file_count);
        }
        pulse::snapshot::EnsureSnapshotResult::Reused(info) => {
            eprintln!("Using snapshot: {} (index unchanged)", info.id);
        }
    }

    let snapshots = pulse::snapshot::list_snapshots(&cache)?;

    let snapshot_diff = if snapshots.len() >= 2 {
        pulse::diff::compute_diff(&snapshots[1].path, &snapshots[0].path, &pulse_config.thresholds).ok()
    } else {
        None
    };

    // Create provider for standalone wiki command
    let (provider, llm_cache) = if !no_llm {
        match pulse::narrate::create_pulse_provider() {
            Ok(p) => {
                eprintln!("LLM provider ready.");
                let c = pulse::llm_cache::LlmCache::new(cache.path());
                (Some(p), Some(c))
            }
            Err(e) => {
                eprintln!("LLM unavailable: {}", e);
                (None, None)
            }
        }
    } else {
        (None, None)
    };

    let snapshot_id = snapshots.first().map(|s| s.id.as_str()).unwrap_or("unknown");
    let pages = pulse::wiki::generate_all_pages(
        &cache,
        snapshot_diff.as_ref(),
        no_llm,
        snapshot_id,
        provider.as_ref().map(|p| p.as_ref()),
        llm_cache.as_ref(),
        &pulse::wiki::ModuleDiscoveryConfig::default(),
    )?;

    if json {
        println!("{}", serde_json::to_string_pretty(&pages)?);
    } else if let Some(out_dir) = output {
        std::fs::create_dir_all(&out_dir)?;
        let rendered = pulse::wiki::render_wiki_markdown(&pages);
        for (filename, content) in &rendered {
            std::fs::write(out_dir.join(filename), content)?;
        }
        eprintln!("Wrote {} wiki pages to {}", rendered.len(), out_dir.display());
    } else {
        let rendered = pulse::wiki::render_wiki_markdown(&pages);
        for (filename, content) in &rendered {
            println!("--- {} ---\n{}\n", filename, content);
        }
    }

    Ok(())
}

fn handle_pulse_map(format: String, output: Option<PathBuf>, zoom: Option<String>) -> Result<()> {
    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }

    let map_format: pulse::map::MapFormat = format.parse()?;
    let map_zoom = match zoom {
        Some(module) => pulse::map::MapZoom::Module(module),
        None => pulse::map::MapZoom::Repo,
    };

    let content = pulse::map::generate_map(&cache, &map_zoom, map_format)?;

    if let Some(out_path) = output {
        std::fs::write(&out_path, &content)?;
        eprintln!("Map written to {}", out_path.display());
    } else {
        println!("{}", content);
    }

    Ok(())
}

fn handle_pulse_generate(
    output: PathBuf,
    base_url: String,
    title: Option<String>,
    include: Option<String>,
    no_llm: bool,
    clean: bool,
    force_renarrate: bool,
    concurrency: usize,
    depth: u8,
    min_files: usize,
) -> Result<()> {
    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }

    let surfaces = match include {
        Some(ref s) => {
            s.split(',')
                .map(|part| match part.trim().to_lowercase().as_str() {
                    "wiki" => Ok(pulse::site::Surface::Wiki),
                    "digest" => Ok(pulse::site::Surface::Digest),
                    "map" => Ok(pulse::site::Surface::Map),
                    other => anyhow::bail!("Unknown surface '{}'. Supported: wiki, digest, map", other),
                })
                .collect::<Result<Vec<_>>>()?
        }
        None => vec![
            pulse::site::Surface::Wiki,
            pulse::site::Surface::Digest,
            pulse::site::Surface::Map,
        ],
    };

    let config = pulse::site::SiteConfig {
        output_dir: output,
        base_url,
        title: title.unwrap_or_else(|| "Pulse".to_string()),
        surfaces,
        no_llm,
        clean,
        force_renarrate,
        concurrency,
        max_depth: depth,
        min_files,
    };

    let report = pulse::site::generate_site(&cache, &config)?;

    eprintln!("Zola project generated in {}/", report.output_dir);
    eprintln!("  Wiki pages: {}", report.pages_generated);
    eprintln!("  Digest: {}", if report.digest_generated { "yes" } else { "no" });
    eprintln!("  Map: {}", if report.map_generated { "yes" } else { "no" });
    eprintln!("  Narration: {}", report.narration_mode);
    if report.build_success {
        eprintln!("  Build: success (HTML in {}/public/)", report.output_dir);
    } else {
        eprintln!("  Build: skipped (run `cd {} && zola build` manually)", report.output_dir);
    }

    Ok(())
}

fn handle_pulse_serve(output: PathBuf, port: u16, open: bool) -> Result<()> {
    // Verify the output dir has a config.toml (i.e., was generated)
    if !output.join("config.toml").exists() {
        anyhow::bail!(
            "No Zola project found at '{}'. Run `rfx pulse generate` first.",
            output.display()
        );
    }

    let zola_path = pulse::zola::ensure_zola()?;

    let url = format!("http://127.0.0.1:{}", port);
    eprintln!("Serving Pulse site at {}", url);
    eprintln!("Press Ctrl+C to stop.\n");

    if open {
        open_browser(&url);
    }

    let status = std::process::Command::new(&zola_path)
        .current_dir(&output)
        .arg("serve")
        .arg("--port")
        .arg(port.to_string())
        .arg("--interface")
        .arg("127.0.0.1")
        .status()
        .context("Failed to start Zola server")?;

    if !status.success() {
        anyhow::bail!("Zola server exited with error");
    }

    Ok(())
}

fn open_browser(url: &str) {
    let result = if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(url).spawn()
    } else if cfg!(target_os = "windows") {
        std::process::Command::new("cmd")
            .args(["/c", "start", url])
            .spawn()
    } else {
        std::process::Command::new("xdg-open").arg(url).spawn()
    };

    if let Err(e) = result {
        eprintln!("Could not open browser: {e}");
    }
}

fn handle_llm_config() -> Result<()> {
    crate::semantic::run_configure_wizard()
}

fn handle_llm_status() -> Result<()> {
    use crate::semantic::config;

    let semantic_config = config::load_config(std::path::Path::new("."))?;
    let provider = &semantic_config.provider;

    let model = if let Some(ref m) = semantic_config.model {
        m.clone()
    } else {
        config::get_user_model(provider)
            .unwrap_or_else(|| "(provider default)".to_string())
    };

    let key_status = match config::get_api_key(provider) {
        Ok(key) => {
            if key.len() > 8 {
                format!("configured ({}...****)", &key[..8])
            } else {
                "configured".to_string()
            }
        }
        Err(_) => "not configured".to_string(),
    };

    println!("Provider: {}", provider);
    println!("Model:    {}", model);
    println!("API key:  {}", key_status);

    Ok(())
}

