//! CLI argument parsing and command router

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use std::path::PathBuf;
use crate::cache::CacheManager;

mod ask;
mod deps;
mod index;
mod llm;
mod misc;
mod pulse;
mod query;
mod serve;
mod snapshot;
mod watch;

pub use self::query::truncate_preview;

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

        /// Number of context lines to show before and after each match (max: 10)
        /// Example: -C 3 shows 3 lines before and after each match
        #[arg(short = 'C', long, value_name = "N")]
        context: Option<usize>,

        /// Return all results (no limit)
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

        /// Filter by language (e.g. rust, python, typescript)
        #[arg(short, long)]
        lang: Option<String>,

        /// Include files matching glob pattern (can be repeated)
        /// Example: --glob "src/**/*.rs"
        #[arg(short = 'g', long)]
        glob: Vec<String>,
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

        /// Override configured LLM provider (openai, anthropic, openrouter, openai-compatible)
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

    /// Generate codebase intelligence surfaces (changelog, wiki, map, site)
    ///
    /// Pulse turns structural facts from the index into browsable documentation.
    /// The `generate` command creates a Zola project and builds it into a static HTML site.
    ///
    /// Examples:
    ///   rfx pulse changelog --no-llm         # Structural-only changelog
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
    /// Generate a product-level changelog from recent commits
    Changelog {
        /// Number of recent commits to include (default: 20)
        #[arg(long, default_value = "20")]
        count: usize,

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

        /// Surfaces to include (comma-separated: wiki,changelog,map,onboard,timeline,glossary,explorer)
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

    /// Generate a developer onboarding guide
    Onboard {
        /// Skip LLM narration
        #[arg(long)]
        no_llm: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Show development timeline from git history
    Timeline {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Generate cross-cutting symbol glossary
    Glossary {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
pub enum LlmSubcommand {
    /// Launch interactive configuration wizard for AI provider and API key
    Config,
    /// Show current LLM configuration status
    Status,
}

/// Format a byte count into a human-readable string (B, KB, MB, GB, TB).
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
    } else if bytes > 0 {
        format!("{} bytes", bytes)
    } else {
        "< 1 KB".to_string()
    }
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
                        index::handle_index_build(&path, &force, &languages, &quiet)
                    }
                    Some(IndexSubcommand::Status) => {
                        index::handle_index_status()
                    }
                    Some(IndexSubcommand::Compact { json, pretty }) => {
                        index::handle_index_compact(&json, &pretty)
                    }
                }
            }
            Some(Command::Query { pattern, symbols, lang, kind, ast, regex, json, pretty, ai, limit, offset, expand, file, exact, contains, count, timeout, plain, glob, exclude, paths, no_truncate, context, all, force, dependencies }) => {
                // If no pattern provided, launch interactive mode (REF-68: require TTY)
                match pattern {
                    None => {
                        use crossterm::tty::IsTty;
                        if !std::io::stdin().is_tty() {
                            eprintln!("error: interactive mode requires a terminal (TTY).");
                            eprintln!("Use 'rfx query <pattern>' for non-interactive search.");
                            std::process::exit(1);
                        }
                        query::handle_interactive()
                    }
                    Some(pattern) => query::handle_query(pattern, symbols, lang, kind, ast, regex, json, pretty, ai, limit, offset, expand, file, exact, contains, count, timeout, plain, glob, exclude, paths, no_truncate, context, all, force, dependencies)
                }
            }
            Some(Command::Serve { port, host }) => {
                serve::handle_serve(port, host)
            }
            Some(Command::Stats { json, pretty }) => {
                misc::handle_stats(json, pretty)
            }
            Some(Command::Clear { yes }) => {
                misc::handle_clear(yes)
            }
            Some(Command::ListFiles { json, pretty, lang, glob }) => {
                misc::handle_list_files(json, pretty, lang, glob)
            }
            Some(Command::Watch { path, debounce, quiet }) => {
                watch::handle_watch(path, debounce, quiet)
            }
            Some(Command::Mcp) => {
                misc::handle_mcp()
            }
            Some(Command::Analyze { circular, hotspots, min_dependents, unused, islands, min_island_size, max_island_size, format, json, pretty, count, all, plain, glob, exclude, force, limit, offset, sort }) => {
                deps::handle_analyze(circular, hotspots, min_dependents, unused, islands, min_island_size, max_island_size, format, json, pretty, count, all, plain, glob, exclude, force, limit, offset, sort)
            }
            Some(Command::Deps { file, reverse, depth, format, json, pretty }) => {
                deps::handle_deps(file, reverse, depth, format, json, pretty)
            }
            Some(Command::Ask { question, execute, provider, json, pretty, additional_context, configure, agentic, max_iterations, no_eval, show_reasoning, verbose, quiet, answer, interactive, debug }) => {
                ask::handle_ask(question, execute, provider, json, pretty, additional_context, configure, agentic, max_iterations, no_eval, show_reasoning, verbose, quiet, answer, interactive, debug)
            }
            Some(Command::Context { structure, path, file_types, project_type, framework, entry_points, test_layout, config_files, depth, json }) => {
                misc::handle_context(structure, path, file_types, project_type, framework, entry_points, test_layout, config_files, depth, json)
            }
            Some(Command::IndexSymbolsInternal { cache_dir }) => {
                index::handle_index_symbols_internal(cache_dir)
            }
            Some(Command::Snapshot { command }) => {
                match command {
                    None => snapshot::handle_snapshot_create(),
                    Some(SnapshotSubcommand::List { json, pretty }) => {
                        snapshot::handle_snapshot_list(json, pretty)
                    }
                    Some(SnapshotSubcommand::Diff { baseline, current, json, pretty }) => {
                        snapshot::handle_snapshot_diff(baseline, current, json, pretty)
                    }
                    Some(SnapshotSubcommand::Gc { json }) => {
                        snapshot::handle_snapshot_gc(json)
                    }
                }
            }
            Some(Command::Pulse { command }) => {
                match command {
                    PulseSubcommand::Changelog { count, no_llm, json, pretty } => {
                        pulse::handle_pulse_changelog(count, no_llm, json, pretty)
                    }
                    PulseSubcommand::Wiki { no_llm, output, json } => {
                        pulse::handle_pulse_wiki(no_llm, output, json)
                    }
                    PulseSubcommand::Map { format, output, zoom } => {
                        pulse::handle_pulse_map(format, output, zoom)
                    }
                    PulseSubcommand::Generate { output, base_url, title, include, no_llm, clean, force_renarrate, concurrency, depth, min_files } => {
                        pulse::handle_pulse_generate(output, base_url, title, include, no_llm, clean, force_renarrate, concurrency, depth, min_files)
                    }
                    PulseSubcommand::Serve { output, port, open } => {
                        pulse::handle_pulse_serve(output, port, open)
                    }
                    PulseSubcommand::Onboard { no_llm, json } => {
                        pulse::handle_pulse_onboard(no_llm, json)
                    }
                    PulseSubcommand::Timeline { json } => {
                        pulse::handle_pulse_timeline(json)
                    }
                    PulseSubcommand::Glossary { json } => {
                        pulse::handle_pulse_glossary(json)
                    }
                }
            }
            Some(Command::Llm { command }) => {
                match command {
                    LlmSubcommand::Config => llm::handle_llm_config(),
                    LlmSubcommand::Status => llm::handle_llm_status(),
                }
            }
        }
    }
}
