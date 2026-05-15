# Changelog

All notable changes to Reflex will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- New `openai-compatible` LLM provider for any endpoint that implements the OpenAI Chat Completions schema, including LMStudio, Ollama, llama.cpp server, vLLM, and litellm proxies. Configure via `rfx llm config` or by setting `[credentials] openai_compatible_base_url` (and optionally `openai_compatible_api_key` / `openai_compatible_model`) in `~/.reflex/config.toml`. The API key is optional for keyless local servers. Closes [#30](https://github.com/reflex-search/reflex/issues/30).

- **MCP server — full 17-tool suite for AI coding agents.** Reflex ships a first-class [Model Context Protocol](https://modelcontextprotocol.io/) server that exposes the complete Reflex search and dependency-analysis engine to AI coding agents over stdio JSON-RPC 2.0. Start it with `rfx mcp`.

  **Search tools:**
  | Tool | Purpose |
  |------|---------|
  | `search_code` | Full-text or symbol-only search with line numbers, code previews, and pagination |
  | `search_regex` | Regex pattern matching with trigram pre-filtering for speed |
  | `search_ast` | Structure-aware Tree-sitter S-expression matching (advanced; use glob to limit scope) |
  | `list_locations` | Fast `{path, line}` location discovery with no content payload — cheapest starting point |
  | `count_occurrences` | Quick `{total, files}` stats without loading any content |
  | `find_references` | Symbol definition + all usage sites in a single atomic call |

  **Dependency graph tools:**
  | Tool | Purpose |
  |------|---------|
  | `get_dependencies` | All static imports for a given file (internal / external / stdlib) |
  | `get_dependents` | Reverse lookup: which files import this one |
  | `get_transitive_deps` | Full dependency tree up to N levels deep |
  | `find_hotspots` | Most-imported files ranked by dependent count |
  | `find_circular` | Circular dependency cycles (A→B→C→A) |
  | `find_unused` | Orphaned files with no importers |
  | `find_islands` | Isolated subsystems in the dependency graph |
  | `analyze_summary` | Codebase health at a glance: counts for circular deps, hotspots, unused files, and islands |

  **Context & index tools:**
  | Tool | Purpose |
  |------|---------|
  | `gather_context` | Project structure, detected frameworks, entry points, and file-type distribution |
  | `check_index_status` | Verify index freshness (`fresh` / `stale` / `missing`) before searching |
  | `index_project` | Build or incrementally update the trigram index; `force: true` for full rebuild |

  **Claude Code integration** — add Reflex to your MCP settings (global: `~/.claude/claude_code_config.json`, or per-project: `.claude/claude_code_config.json`):

  ```json
  {
    "mcpServers": {
      "reflex": {
        "command": "rfx",
        "args": ["mcp"]
      }
    }
  }
  ```

  Then index your project once:

  ```bash
  rfx index
  ```

  Claude Code starts `rfx mcp` automatically and Reflex tools are available in every conversation. See the [Claude Code + Reflex MCP Quickstart](./docs/ai-agent-integration.md) for step-by-step setup, usage examples, and troubleshooting.

### Changed

- LLM model resolution centralized into `config::resolve_model` / `config::resolve_model_for`. **Previously, `~/.reflex/config.toml` `[credentials] {provider}_model` was silently ignored by chat_tui's runtime calls** (compaction, mid-session triage, in-session provider switches) and the provider's hard-coded constructor default was used instead. After this change those user-config values are honored everywhere. **If you set `openrouter_model = "anthropic/claude-opus-4"` (or similar) thinking it had no effect, your interactive sessions will now actually use that model — which may change costs.** Verify your `~/.reflex/config.toml` after upgrading.
- Four near-duplicate model-resolution implementations across `semantic/mod.rs`, `semantic/agentic.rs`, `semantic/chat_tui.rs`, and `pulse/narrate.rs` collapsed into one helper.

### Fixed

- chat_tui no longer silently falls back to the provider's hard-coded default model when `~/.reflex/config.toml` has a model set. Previously this only worked on the initial chat session; runtime calls (compaction, triage, mid-session provider switches) used the constructor default. Invisible for openai/anthropic/openrouter (silent wrong-model); a hard error for the new `openai-compatible` provider (no default model available).
- The `/model` command in interactive chat now accepts `openai-compatible` and no longer panics via `unreachable!()` when switching to a provider without a hard-coded default. It refuses the switch with a friendly status-bar message if no model is configured for the target provider.
- Triage failures in interactive chat now surface in the status bar as a `PhaseUpdate::Notice` event instead of being written only to the log, so silent fallback to keyword-only search is no longer invisible to the user.

## [1.1.3] - 2026-04-27

### Fixed

- MCP stdio server now conforms to JSON-RPC 2.0 and is accepted by strict clients (e.g. Claude Code's Zod validators):
  - Notifications (messages without an `id`, such as `notifications/initialized`) no longer receive a response.
  - Responses never emit `"id": null`; the `id` field is now omitted when absent rather than serialized as null.

## [0.2.3](https://github.com/reflex-search/reflex/compare/v0.2.2...v0.2.3) - 2025-11-04

### Other

- update Cargo.toml dependencies

## [0.1.3](https://github.com/reflex-search/reflex/compare/v0.1.2...v0.1.3) - 2025-11-03

### Other

- Merge pull request #2 from reflex-search/release-plz-2025-11-03T22-53-50Z
- *(ci)* add release status logging to workflow

## [0.1.2](https://github.com/reflex-search/reflex/compare/v0.1.1...v0.1.2) - 2025-11-03

### Fixed

- *(ci)* use default GITHUB_TOKEN and correct repository URL
- *(ci)* remove rfx symlink to fix release-plz
- *(ci)* correct release-plz.toml configuration format
- *(ci)* correct release-plz GitHub Action reference

### Other

- release v0.1.1
- (feat) initial release
- Some test fixes
- Added test file corpus for various languages
- Renamed folder to .bruno
- Added MCP support
- Added a "watch" command for rebuilding index on the fly with debounce
- Added AST query support
- Rebuilt query engine to use a builder pipeline instead of conditional cases and fixed symbol searches when using regex
- Make symbol field optional for regex matches
- Revert "Fix regex queries to return actual symbol names instead of regex matches"
- Fix regex queries to return actual symbol names instead of regex matches
- Removed symbol counts from stats, because they're deprecated
- Added bruno collection and incremental index rebuild
- Added HTTP server
- Updated default index flag to -p and updated docs
- Renamed binary to "rfx"
- Fix tests
- Filled out test suite comprehensively
- Fixed indexing memory ballooning on massive codebases
- Flushing memory between indexing batches to save system memory
- Had an idea that resulted in unbelievably fast lookups - removing the symbol index entirely and doing a 2-stage lookup for symbols
- More performance enhancements, needs more work to handle the largest codebases like linux and chromium
- Massive performance boosts on reindexing and query retrieval for huge codebases
- Fixed Java parser
- Added many missing language parsers
- Updated todo and fixed another regex bug
- Fixed some more regex issues
- Added some better warning handling for stale indexes to json output
- Added --json flag for programmatic parsing
- Update warning messages to use ⚠️ emoji and WARNING prefix
- Replace blocking validation with non-blocking warnings
- Fix branch recording performance issue
- Fixed another bug with regex
- Refining regex
- Got regex working
- Reduced number of indexing CPU threads to 80% of available threads
- Added lines and symbols output for indexer
- Added PHP support
- Updated todo
- Updated --kind function filter to include class methods
- Fixed some ts parsing issues with functions
- Fixed lang parameter to only accept supported languages
- Added react/vue/svelte support and fixed some warnings
- Added Unknown fallback type to SymbolKind to ensure 100% indexing coverage consistently
- Added missing symbol types
- Added file type breakdown in indexing summary
- Ignoring unsupported languages entirely
- Massive performance gains
- Fixed progress bar issues
- Added some perf enhancements, a --count flag and progress bar for the indexer
- Some performance improvements
- Added retrieval time measurement
- update todo
- More filter options and suppressing logs
- Further refining querying. added "kind" "expand" and "file" filters
- Refining query capabilities
- Some fixes
- Fixed indexing truncation bug
- Added query and list-files commands
- Added basic parser functionality
- Some context engineering instructions
- Updated some todos
- Initial commit

## [0.1.1](https://github.com/reflex-search/reflex/compare/v0.1.0...v0.1.1) - 2025-11-03

### Fixed

- *(ci)* remove rfx symlink to fix release-plz
- *(ci)* correct release-plz.toml configuration format
- *(ci)* correct release-plz GitHub Action reference

### Other

- (feat) initial release
- Some test fixes
- Added test file corpus for various languages
- Renamed folder to .bruno
- Added MCP support
- Added a "watch" command for rebuilding index on the fly with debounce
- Added AST query support
- Rebuilt query engine to use a builder pipeline instead of conditional cases and fixed symbol searches when using regex
- Make symbol field optional for regex matches
- Revert "Fix regex queries to return actual symbol names instead of regex matches"
- Fix regex queries to return actual symbol names instead of regex matches
- Removed symbol counts from stats, because they're deprecated
- Added bruno collection and incremental index rebuild
- Added HTTP server
- Updated default index flag to -p and updated docs
- Renamed binary to "rfx"
- Fix tests
- Filled out test suite comprehensively
- Fixed indexing memory ballooning on massive codebases
- Flushing memory between indexing batches to save system memory
- Had an idea that resulted in unbelievably fast lookups - removing the symbol index entirely and doing a 2-stage lookup for symbols
- More performance enhancements, needs more work to handle the largest codebases like linux and chromium
- Massive performance boosts on reindexing and query retrieval for huge codebases
- Fixed Java parser
- Added many missing language parsers
- Updated todo and fixed another regex bug
- Fixed some more regex issues
- Added some better warning handling for stale indexes to json output
- Added --json flag for programmatic parsing
- Update warning messages to use ⚠️ emoji and WARNING prefix
- Replace blocking validation with non-blocking warnings
- Fix branch recording performance issue
- Fixed another bug with regex
- Refining regex
- Got regex working
- Reduced number of indexing CPU threads to 80% of available threads
- Added lines and symbols output for indexer
- Added PHP support
- Updated todo
- Updated --kind function filter to include class methods
- Fixed some ts parsing issues with functions
- Fixed lang parameter to only accept supported languages
- Added react/vue/svelte support and fixed some warnings
- Added Unknown fallback type to SymbolKind to ensure 100% indexing coverage consistently
- Added missing symbol types
- Added file type breakdown in indexing summary
- Ignoring unsupported languages entirely
- Massive performance gains
- Fixed progress bar issues
- Added some perf enhancements, a --count flag and progress bar for the indexer
- Some performance improvements
- Added retrieval time measurement
- update todo
- More filter options and suppressing logs
- Further refining querying. added "kind" "expand" and "file" filters
- Refining query capabilities
- Some fixes
- Fixed indexing truncation bug
- Added query and list-files commands
- Added basic parser functionality
- Some context engineering instructions
- Updated some todos
- Initial commit

### Added
- Query timeout support with `--timeout` flag (default: 30 seconds)
- HTTP API timeout parameter support
- MCP server with 30-second default timeout for all queries

## [1.0.0] - 2025-11-03

Initial release of Reflex - a local-first, structure-aware code search engine for AI coding workflows.

### Added

#### Core Features
- **Trigram-based indexing** for sub-100ms full-text search on large codebases (10k+ files)
- **Runtime symbol detection** using Tree-sitter parsing on candidate files only
- **Complete coverage** - finds every occurrence of patterns, not just symbol definitions
- **Deterministic results** - same query always returns same results (sorted by file:line)
- **Memory-mapped I/O** for instant cache access (zero-copy)
- **Incremental indexing** using blake3 content hashing (only reindex changed files)
- **Regex support** with trigram optimization for fast pattern matching
- **AST pattern matching** using Tree-sitter S-expression queries

#### Language Support
- **Rust** - Functions, structs, enums, traits, impls, modules, methods
- **TypeScript** - Functions, classes, interfaces, types, enums, React components
- **JavaScript** - Functions, classes, constants, methods, React components
- **Vue** - Functions, constants, methods from `<script>` blocks (Composition API support)
- **Svelte** - Functions, variables, reactive declarations (`$:`)
- **PHP** - Functions, classes, interfaces, traits, methods, namespaces, enums (PHP 8.1+)

#### CLI Commands
- `rfx index` - Build or update the local search index
- `rfx query` - Search the codebase with multiple search modes:
  - Full-text search (default - finds all occurrences)
  - Symbol-only search (`--symbols` flag)
  - Regex search (`--regex` flag)
  - AST pattern matching (`--ast` flag)
- `rfx stats` - Display index statistics
- `rfx clear` - Clear the search index
- `rfx list-files` - List all indexed files
- `rfx watch` - Auto-reindex on file changes with configurable debouncing
- `rfx serve` - Start HTTP API server for programmatic access
- `rfx mcp` - Start MCP (Model Context Protocol) server for AI agents

#### Search Features
- **Symbol filtering** by kind (function, class, struct, enum, etc.)
- **Language filtering** to search specific languages only
- **File path filtering** with substring matching
- **Result limiting** with `--limit` flag
- **Exact matching** mode with `--exact` flag
- **Symbol expansion** with `--expand` flag (show full function/class body)
- **JSON output** for AI agents and automation tools
- **Count-only mode** with `--count` flag

#### HTTP API
- **GET /query** - Search the codebase
  - Query params: `q`, `lang`, `kind`, `limit`, `symbols`, `regex`, `exact`, `expand`, `file`
  - Returns: `QueryResponse` JSON with results and index status
- **GET /stats** - Get index statistics
  - Returns: `IndexStats` JSON with file counts, sizes, language breakdowns
- **POST /index** - Trigger reindexing
  - Body: `{"force": boolean, "languages": [string]}`
  - Returns: `IndexStats` JSON after indexing completes
- **GET /health** - Health check endpoint
- **CORS enabled** for browser clients

#### MCP Server
- **search_code** - Full-text or symbol search
- **search_regex** - Regex pattern matching with trigram optimization
- **search_ast** - Structure-aware AST pattern matching
- **index_project** - Trigger reindexing
- **stdio transport** for seamless integration with Claude Code and other MCP clients

#### File Watching
- **Auto-reindex** on file changes
- **Configurable debouncing** (5-30 seconds, default: 15s)
- **Quiet mode** for background operation
- **Respects .gitignore** patterns automatically

#### Index Features
- **Git-aware** - tracks current branch and commit SHA
- **Staleness detection** - warns when index is out of sync with working tree
- **Branch tracking** - separate indices per branch
- **Dirty state tracking** - knows when uncommitted changes exist
- **.gitignore support** - automatically excludes ignored files
- **Configurable max file size** (default: 10 MB)
- **Parallel indexing** using rayon (default: 80% of CPU cores)

#### Cache Structure
- `.reflex/meta.db` - SQLite database for file metadata and statistics
- `.reflex/trigrams.bin` - Memory-mapped inverted index (trigram → file locations)
- `.reflex/content.bin` - Memory-mapped full file contents for context extraction
- `.reflex/config.toml` - Index configuration (auto-generated)

#### Performance
- **Sub-100ms queries** on 10k+ files (trigram indexing)
- **2-3ms queries** on small codebases (50 files)
- **124ms full-text search** on Linux kernel (62K files)
- **224ms symbol search** on Linux kernel (runtime parsing of ~3 candidate files)
- **Incremental indexing**: <1 second for changed files

#### Documentation
- Comprehensive README.md with usage examples and API reference
- Detailed ARCHITECTURE.md with system design and data formats
- CLAUDE.md with project overview and development workflow
- 221 comprehensive tests (unit, integration, performance)
- Rustdoc comments for all public APIs

#### Quality & Testing
- **221 comprehensive tests**
  - 194 unit tests (cache, indexer, query, parsers, core modules)
  - 17 integration tests (workflows, multi-language, error handling)
  - 10 performance tests (indexing speed, query latency, scalability)
- **Zero unsafe code** - all safe Rust
- **Error handling** - comprehensive error messages with context
- **Logging support** - configurable with RUST_LOG environment variable

### Performance

- **Indexing**: 100 files in <1 second, 1,000 files in <2 seconds
- **Query latency**: Sub-100ms on large codebases (10k+ files)
- **Memory usage**: Efficient memory-mapped I/O (zero-copy)
- **Cache size**: Compressed trigram index + full file contents

### Architecture

- **Trigram-based inverted index** inspired by Zoekt/Google Code Search
- **Runtime symbol detection** - parse only candidate files at query time
- **Memory-mapped I/O** for instant cache access
- **Parallel processing** with rayon for multi-core indexing
- **Incremental updates** using blake3 content hashing
- **Zero-copy deserialization** with memory-mapped files

### Technology Stack

- **Language**: Rust (Edition 2024)
- **Parsing**: tree-sitter with language-specific grammars
- **Storage**: rusqlite (metadata), custom binary format (trigrams + content)
- **Hashing**: blake3 for fast content hashing
- **HTTP**: axum web framework with CORS support
- **CLI**: clap for argument parsing
- **Async**: tokio runtime for HTTP server
- **Parallelism**: rayon for multi-threaded indexing

[Unreleased]: https://github.com/reflex-search/reflex/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/reflex-search/reflex/releases/tag/v1.0.0
