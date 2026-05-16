## [1.5.2] - 2026-05-16


### Documentation


- Fix rfx pulse digest → rfx pulse changelog in README

- Remove stale hardcoded test count badge from README


### Fixed


- Move performance tests to separate release-mode CI step

- Resolve all clippy warnings and enable -D warnings in CI

- Commit insta snapshots and fix Windows clippy errors (REF-172)

- Normalize path separators to fix 21 Windows-only test failures (REF-173)

- Make path/home overrides reach storage and dirs lookups (REF-173)

## [1.5.1] - 2026-05-15


### Fixed


- Correct release.toml format for cargo-release

- Restore git-cliff pre-release hook in correct format

- Repair CI failures blocking release

## [1.4.0] - 2026-05-15


### Added


- Add check_index_status MCP tool (REF-107)

- Raise preview truncation to 180 chars and add preview_length param


### Documentation


- Add MCP tool selection decision tree cheatsheet

- Reframe README to reflect CLI-first, not AI-only


### Fixed


- Fix JSON output correctness across query, deps, and analyze

- Detect corrupt trigrams.bin via magic-byte check before skipping rebuild

- Correct singular/plural for result count in TUI (REF-66)

## [1.3.5] - 2026-05-14


### Fixed


- Hardlink issue in pipeline

## [1.3.4] - 2026-05-14


### Fixed


- Dumb symlink issue

## [1.3.3] - 2026-05-14


### Fixed


- Release pipeline

## [1.3.2] - 2026-05-14


### Fixed


- Cargo-dist version issue

## [1.3.1] - 2026-05-14


### Fixed


- Some cargo-dist dependency issues

## [1.3.0] - 2026-05-14


### Added


- Introduce SymbolRef for stable JSON symbol output

- Wire dependencies parameter to list_locations, count_occurrences, search_regex, search_ast


### Changed


- Decompose query.rs into src/query/ submodules


### Documentation


- Add rfx serve threat model and language forward-compat note


### Fixed


- Update reflex-search npm package to fix CVE vulnerabilities

- Broken csharp tests resolved

- Indexing and git workspace issue

## [1.1.3] - 2026-04-27


### Fixed


- Json serialization for mcp

## [1.1.0] - 2026-04-13


### Added


- Add OpenRouter provider support with model fetching and sorting options

- Remove Groq provider and update references to OpenRouter in configuration and documentation

- Enhance LLM response handling with validation and JSON extraction improvements

- Enhance module detection and snapshot analysis with improved descriptions and visibility adjustments

- Implement static site generator with wiki, digest, and map

- Enhance Pulse functionality with LLM narration support for digest and wiki generation

- Add dependency diagram to wiki pages


### Fixed


- Update OpenRouter sort strategy from "speed" to "latency" for API compatibility

## [1.0.4] - 2026-04-07


### Changed


- Streamline language parsing and error handling in CLI and indexer


### Documentation


- Add important setup notes for running commands and gitignore configuration

- Add gitcgr code graph badge

## [1.0.3] - 2025-11-21


### Added


- Update mcp docs to force auto-reindexing and hopefully fixed auto packaging pipeline

## [1.0.1] - 2025-11-21


### Added


- Added auto package publishing for new releases

## [1.0.0] - 2025-11-21


### Added


- Rfx query interactive mode

- Made context full by default and added rfx context to mcp tools

## [0.9.2] - 2025-11-21


### Documentation


- Update README for clarity and accuracy in feature descriptions

## [0.9.1] - 2025-11-20


### Added


- Enhance agentic loop to return query confidence alongside responses

- Semantic query building with external LLMs

## [0.9.0] - 2025-11-19


### Added


- Implement semantic query execution and parsing

- Refactor semantic query generation with project-specific configuration and context extraction

- Massively cleaned up claude.md

- Enhance query execution with count mode and update prompt template for new flags

- Enhance configuration documentation and improve logging levels for cache and semantic query handling

- Update AI provider models and enhance OpenAI request handling for GPT-5 compatibility

- Enhance environment variable handling in tests and update Gemini model defaults

- Add context generation module for AI prompts

- Update default tree depth for --structure option to 1

- Implement agentic semantic query builder with multi-phase workflow

- Enhance agentic mode with additional command options and reporting capabilities

- Enhance regex pattern syntax documentation and clarify flag combinations

- Add support for OpenAI GPT-OSS models with enhanced handling and messaging

- Enhance agentic mode with improved response structure and reporting capabilities

- Add conversational answer generation feature and update CLI handling

- Add context extraction for match results in query handling and answer generation

- Increase token limits to 4000 for Anthropic, Gemini, Groq, and OpenAI providers

- Add termimad for markdown rendering and update provider display with recommendations

- Enhance agentic reporter with spinner support for improved progress visualization

- Implement interactive TUI chat mode for `rfx ask` with message history, input handling, and progress updates

- Enhance Groq API error handling with detailed logging and timeout configuration

- Add bottom padding for message display to improve text wrapping in TUI chat mode

- Add mouse event handling for scrolling in TUI chat mode

- Add text wrapping functionality for message display in TUI chat mode

- Add markdown rendering with consistent prefix for message display in TUI chat mode

- Update wrap_with_prefix to use consistent border colors for message display in TUI chat mode

- Add debug mode to output full LLM prompts and retain terminal history

- Implement API key configuration check and enhance documentation search functionality

- Enhance agentic mode to gather and utilize context for improved answer generation

## [0.8.2] - 2025-11-16


### Added


- Enhance documentation for dependency tracking and semantic query building

- Implement automatic cache invalidation using schema hash

- Implement comprehensive cache corruption detection and validation tests

- Implement cache compaction functionality and CLI commands for manual compaction

- Add cache compaction commands and update documentation for indexing status

## [0.8.1] - 2025-11-16


### Added


- Update documentation for dependency analysis commands and enhance Ruby parser for require statements

## [0.8.0] - 2025-11-16


### Added


- Updated python, go, ts and rust dependency resolution to support monorepos

- Added java, kotlin and ruby monorepo support

- Fixed vue and ts dep resolution

- Add size filtering for island detection

- Refactor dependency analysis API with cleaner command separation

- Enhance dependency analysis commands with pagination and sorting options

- Add analyze_summary tool for quick dependency health overview

- Enhance dependency analysis with file-level grouping and improved output formats

- Refactor search result handling to always use grouped format and improve response structure

- Dependency tracking and analysis


### Fixed


- Tightening up js/ts path resolution

- Enhanced ts/js path resolution for dependencies

- Some updates to pagination for rfx analyze

## [0.7.1] - 2025-11-12


### Added


- Added basic functionality for dependency support

- Dependency resolution across more languages


### Fixed


- Updated some more broken language dependency parsers

- Some more dependency resolution bugfixes

- Refined some regex functionality

## [0.6.0] - 2025-11-11


### Added


- Implemented more mouse support and fixed some syntax highlighting

- Add missing CLI filter options to interactive mode

- Add mouse click support for new filter options

- More UI refinement

- Interactive mode


### Fixed


- Various bugfixes with result list

- Fixed remaining known syntax highligting issues in interactive mode

- More bugfixes

- Accurate mouse click detection for all filter badges

- More ui bugfixes

- Fixed filter display bug

- Rendering bugfixes

- Incorrect file counts in interactive mode

- Fixed background indexing issues

## [0.5.2] - 2025-11-10


### Added


- Added AI suggestions

## [0.5.1] - 2025-11-09


### Fixed


- Random bugfixes

- Switched symbol indexing to use content cache rather than filesystem reads for performance boost

## [0.5.0] - 2025-11-09


### Added


- First swipe at interactive mode, work in progress

- Enhancements

- Loading windows

- Added preflight check to prevent runaway and unbounded queries

- Added early globbing


### Fixed


- Fixed result scrolling

- Removed early filtering performance boost that ended up sacrificing accuracy

- Small optimization to lookups

- Large performance improvement and bugfix with --kind filtering

- Some database refactoring

- Early language filtering

- Speeding up background indexing

## [0.4.2] - 2025-11-07


### Added


- Added many tests and edge cases to symbol search


### Fixed


- Reworking some of the symbol search functionality, bugs remain

- Major fixes to symbol search accuracy

## [0.4.0] - 2025-11-07


### Added


- Lots of json output optimizations to reduce token usage

- Added pagination to prevent breaking context window limits


### Fixed


- Moved pagination to before symbol enrichment to boost performance

- Pagination total count bug

## [0.3.2] - 2025-11-06


### Added


- Fixed attribute support with --kind query

## [0.3.0] - 2025-11-06


### Added


- Added language filter and parallelized --symbol extraction

- Implement word-boundary matching as default search behavior

- Refactored AST query functionality, much simpler now


### Fixed


- Added thread cap for indexing and querying

- Added some missing symbol grammar and added universal ast support, not just for select languages

## [0.2.13] - 2025-11-05


### Fixed


- Cache validation was reading entire files causing 18x slowdown

## [0.2.12] - 2025-11-05


### Added


- Added globbing and build optimizations

## [0.2.10] - 2025-11-04


### Fixed


- Prevent cargo-dist from uploading archives to release

## [0.2.9] - 2025-11-04


### Fixed


- Improve release artifacts with friendly names

## [0.2.7] - 2025-11-04


### Fixed


- Correct release workflow

## [0.2.6] - 2025-11-04


### Fixed


- Trying more release fixes

## [0.2.5] - 2025-11-04


### Fixed


- Another release pipeline fix

## [0.2.4] - 2025-11-04


### Fixed


- Update the release pipeline

## [0.2.3] - 2025-11-04


### Fixed


- Bump version

## [0.2.2] - 2025-11-04


### Fixed


- Bumped version v0.2.2

## [0.2.1] - 2025-11-04


### Fixed


- Lots of general bugfixes

- Removed zip archives from releases v0.2.1

## [0.2.0] - 2025-11-04


### Added


- Add cross-platform binary distribution with cargo-dist

## [0.1.2] - 2025-11-03


### Fixed


- Use default GITHUB_TOKEN and correct repository URL

## [0.1.1] - 2025-11-03


### Fixed


- Correct release-plz GitHub Action reference

- Correct release-plz.toml configuration format

- Remove rfx symlink to fix release-plz

## [0.1.0] - 2025-11-03

