# Contributing to Reflex

Thank you for your interest in contributing to Reflex! We welcome contributions that help make Reflex faster, more accurate, and easier to use.

## Project Philosophy

Reflex is built around three core principles:

1. **Speed**: Extremely fast queries on large codebases through efficient trigram indexing
2. **Accuracy**: Complete coverage with deterministic results (no probabilistic ranking)
3. **Simplicity**: Clean architecture that's easy to understand, extend, and maintain

When contributing, please keep these principles in mind.

---

## Getting Started

### Prerequisites

- **Rust**: Version 1.75 or later (edition 2024)

### Development Setup

1. **Clone the repository:**

```bash
git clone https://github.com/reflex-search/reflex.git
cd reflex
```

2. **Build the project:**

```bash
cargo build --release
```

3. **Run tests:**

```bash
cargo test
```

4. **Run with debug logging:**

```bash
RUST_LOG=debug cargo run -- query "pattern"
```

5. **Install git hooks (one-time setup):**

```bash
# Install lefthook (pick one):
cargo install lefthook   # via Cargo
# brew install lefthook  # via Homebrew (macOS)

# Wire the hooks into .git/hooks/:
lefthook install
```

After this, any commit that includes `.rs` files will automatically run
`cargo fmt --all` and re-stage the formatted files — the commit proceeds
without any extra steps.

6. **Generate documentation:**

```bash
cargo doc --open
```

---

## Project Structure

```
reflex/
├── src/
│   ├── lib.rs              # Library root (public API)
│   ├── main.rs             # CLI entry point
│   ├── cli.rs              # Command-line interface
│   ├── cache.rs            # Cache management (SQLite, file I/O)
│   ├── indexer.rs          # Indexing logic (trigram extraction)
│   ├── query.rs            # Query engine (search execution)
│   ├── trigram.rs          # Trigram algorithm (inverted index)
│   ├── content_store.rs    # Memory-mapped content storage
│   ├── regex_trigrams.rs   # Regex optimization
│   ├── ast_query.rs        # AST pattern matching
│   ├── formatter.rs        # Output formatting
│   ├── watcher.rs          # File watcher (auto-reindex)
│   ├── mcp.rs              # MCP server (AI integration)
│   ├── git.rs              # Git integration
│   ├── models.rs           # Data structures and types
│   └── parsers/            # Language-specific parsers
│       ├── mod.rs          # Parser factory
│       ├── rust.rs         # Rust parser
│       ├── typescript.rs   # TypeScript/JavaScript parser
│       ├── python.rs       # Python parser
│       ├── go.rs           # Go parser
│       ├── java.rs         # Java parser
│       ├── c.rs            # C parser
│       ├── cpp.rs          # C++ parser
│       ├── php.rs          # PHP parser
│       ├── csharp.rs       # C# parser
│       ├── ruby.rs         # Ruby parser
│       ├── kotlin.rs       # Kotlin parser
│       ├── zig.rs          # Zig parser
│       ├── vue.rs          # Vue parser
│       └── svelte.rs       # Svelte parser
├── tests/
│   ├── integration_test.rs     # End-to-end workflow tests
│   └── performance_test.rs     # Performance benchmarks
├── .context/
│   └── TODO.md             # Task tracking and roadmap
├── ARCHITECTURE.md         # System design documentation
├── CLAUDE.md               # Development workflow guide
├── README.md               # User documentation
├── API.md                  # HTTP API reference
└── Cargo.toml              # Rust package configuration
```

---

## Development Workflow

### Making Changes

1. **Create a branch** for your feature or bugfix:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

2. **Make your changes** following our [Code Style](#code-style) guidelines

3. **Write tests** for new functionality (see [Testing](#testing))

4. **Run the test suite:**

```bash
cargo test
```

5. **Run linters:**

```bash
cargo fmt --check  # Verify formatting (lefthook auto-fixes on commit)
cargo clippy -- -D warnings  # Run linter
```

6. **Update documentation** if adding new features or changing behavior

7. **Commit your changes** using [Conventional Commits](#conventional-commits)

8. **Push to your fork and create a Pull Request**

### Pull Request Process

1. **Open a PR** with a clear title and description
2. **Link related issues** (e.g., "Fixes #123")
3. **Wait for CI checks** to pass (tests, linting, formatting)
4. **Respond to review feedback** and make requested changes
5. **Squash commits** if requested before merge

**PR Checklist:**

- [ ] Tests pass locally (`cargo test`)
- [ ] Code is formatted (`cargo fmt` — lefthook handles this automatically on commit)
- [ ] No clippy warnings (`cargo clippy`)
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow Conventional Commits format
- [ ] PR description explains what/why (not just how)

---

## Code Style

### Rust Style Guidelines

Reflex follows **standard Rust conventions**:

- Use `rustfmt` for code formatting (run `cargo fmt` before committing)
- Use `clippy` for linting (address all warnings: `cargo clippy -- -D warnings`)
- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)

### Naming Conventions

- **Modules**: `snake_case` (e.g., `query_engine`, `ast_parser`)
- **Structs/Enums**: `PascalCase` (e.g., `QueryEngine`, `SymbolKind`)
- **Functions/Methods**: `snake_case` (e.g., `extract_symbols`, `parse_file`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `DEFAULT_TIMEOUT`, `MAX_FILE_SIZE`)

### Documentation

- **Public API**: All public items must have rustdoc comments
- **Modules**: Start with `//!` module-level docs
- **Functions**: Use `///` doc comments with examples for complex functions
- **Examples**: Include code examples in doc comments when helpful

```rust
/// Parse source code into an abstract syntax tree.
///
/// # Arguments
///
/// * `source` - Source code string
/// * `language` - Programming language
///
/// # Returns
///
/// Returns a `Tree` on success, or an error if parsing fails.
///
/// # Examples
///
/// ```
/// let tree = parse_tree("fn main() {}", Language::Rust)?;
/// ```
pub fn parse_tree(source: &str, language: Language) -> Result<Tree> {
    // ...
}
```

### Error Handling

- Use `anyhow::Result` for functions that return errors
- Use `anyhow::bail!()` for early returns with error messages
- Add context to errors using `.context()` or `.with_context()`

```rust
use anyhow::{Context, Result};

fn read_file(path: &Path) -> Result<String> {
    std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read file: {}", path.display()))
}
```

---

## Testing

Reflex has **comprehensive tests** across three categories:

### Unit Tests

Located in `#[cfg(test)]` modules within source files:

```bash
# Run all unit tests
cargo test --lib

# Run tests for a specific module
cargo test --lib cache::tests

# Run with output visible
cargo test -- --nocapture
```

**Writing Unit Tests:**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigram_extraction() {
        let text = "hello world";
        let trigrams = extract_trigrams(text);
        assert_eq!(trigrams, vec!["hel", "ell", "llo"]);
    }
}
```

### Integration Tests

Located in `tests/` directory:

```bash
# Run integration tests
cargo test --test integration_test

# Run specific integration test
cargo test --test integration_test test_full_workflow
```

**Integration tests** verify end-to-end workflows (index → query → verify).

### Performance Tests

Located in `tests/performance_test.rs`:

```bash
# Run performance tests
cargo test --test performance_test --release

# Skip slow tests during development
cargo test --test performance_test --release -- --skip large
```

**Performance tests** ensure queries remain under target latencies.

### Test Coverage Goals

- **New features**: Must include tests covering typical and edge cases
- **Bug fixes**: Add a regression test that would have caught the bug
- **Parsers**: Each language parser needs 8-15 tests covering all symbol types
- **Core modules**: Aim for >80% code coverage on critical paths

---

## Conventional Commits

Reflex uses **Conventional Commits** for automatic changelog generation and version bumping.

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Commit Types

- `feat:` - New feature (triggers MINOR version bump)
- `fix:` - Bug fix (triggers PATCH version bump)
- `docs:` - Documentation only changes
- `refactor:` - Code refactoring (no functional changes)
- `perf:` - Performance improvements
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks (dependencies, build, etc.)
- `BREAKING CHANGE:` - Breaking change (triggers MAJOR version bump)

### Examples

```bash
# Feature: Adds timeout support
git commit -m "feat(query): add --timeout flag for query timeout control"

# Bug fix: Fixes crash
git commit -m "fix(indexer): handle empty files without panic"

# Breaking change: Removes deprecated API
git commit -m "feat(api): remove deprecated /search endpoint

BREAKING CHANGE: The /search endpoint has been removed.
Use /query instead."

# Documentation
git commit -m "docs(readme): add examples for AST pattern matching"

# Refactoring
git commit -m "refactor(trigram): simplify posting list intersection"

# Performance
git commit -m "perf(query): optimize symbol lookup with hash map"

# Tests
git commit -m "test(parser): add tests for PHP enum parsing"

# Chore
git commit -m "chore(deps): update tree-sitter to 0.24.1"
```

### Validation

Install `cocogitto` for local commit validation:

```bash
cargo install cocogitto

# Validate commits
cog check

# Create a conventional commit interactively
cog commit
```

---

## Adding New Language Support

Reflex supports 14+ languages through Tree-sitter parsers. Here's how to add a new language:

### 1. Add Tree-sitter Grammar Dependency

Edit `Cargo.toml`:

```toml
[dependencies]
tree-sitter-yourLanguage = "0.23"
```

### 2. Update Language Enum

In `src/models.rs`, add your language to the `Language` enum:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    Rust,
    Python,
    // ... existing languages
    YourLanguage,  // Add here
}
```

### 3. Create Parser Module

Create `src/parsers/your_language.rs`:

```rust
use tree_sitter::{Node, Parser, Query, QueryCursor};
use crate::models::{Symbol, SymbolKind, Span};

/// Extract symbols from YourLanguage source code
pub fn extract_symbols(source: &str) -> Vec<Symbol> {
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_your_language::language())
        .expect("Failed to load YourLanguage grammar");

    let tree = match parser.parse(source, None) {
        Some(tree) => tree,
        None => return Vec::new(),
    };

    let mut symbols = Vec::new();
    let root = tree.root_node();

    // Extract functions
    symbols.extend(extract_functions(source, root));

    // Extract classes
    symbols.extend(extract_classes(source, root));

    // Extract other symbol types...

    symbols
}

fn extract_functions(source: &str, root: Node) -> Vec<Symbol> {
    // Use Tree-sitter query to find function nodes
    let query_str = "(function_declaration name: (identifier) @name)";
    let query = Query::new(&tree_sitter_your_language::language(), query_str)
        .expect("Invalid query");

    let mut cursor = QueryCursor::new();
    let matches = cursor.matches(&query, root, source.as_bytes());

    let mut functions = Vec::new();
    for m in matches {
        for capture in m.captures {
            let node = capture.node;
            let name = node.utf8_text(source.as_bytes()).unwrap_or("");

            functions.push(Symbol {
                name: name.to_string(),
                kind: SymbolKind::Function,
                span: Span {
                    start_line: node.start_position().row + 1,
                    start_col: node.start_position().column + 1,
                    end_line: node.end_position().row + 1,
                    end_col: node.end_position().column + 1,
                },
                scope: None,
            });
        }
    }

    functions
}
```

### 4. Register in Parser Factory

In `src/parsers/mod.rs`, add your parser:

```rust
pub mod your_language;

pub fn parse_file(source: &str, language: Language) -> Vec<Symbol> {
    match language {
        Language::Rust => rust::extract_symbols(source),
        Language::Python => python::extract_symbols(source),
        // ... existing parsers
        Language::YourLanguage => your_language::extract_symbols(source),
        _ => Vec::new(),
    }
}
```

### 5. Update File Extensions

In `src/indexer.rs`, map file extensions to your language:

```rust
fn detect_language(path: &Path) -> Option<Language> {
    match path.extension()?.to_str()? {
        "rs" => Some(Language::Rust),
        "py" => Some(Language::Python),
        // ... existing extensions
        "yourlang" | "yl" => Some(Language::YourLanguage),
        _ => None,
    }
}
```

### 6. Add Tests

Create comprehensive tests in `src/parsers/your_language.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_functions() {
        let source = r#"
            function hello() {
                console.log("Hello");
            }
        "#;

        let symbols = extract_symbols(source);
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].name, "hello");
        assert_eq!(symbols[0].kind, SymbolKind::Function);
    }

    // Add more tests for classes, methods, etc.
}
```

### 7. Update Documentation

- Add your language to `README.md` (Supported Languages table)
- Add your language to `CLAUDE.md` (Supported Languages section)
- Update `API.md` (Supported Languages table)

### Resources

- **Tree-sitter Documentation**: https://tree-sitter.github.io/tree-sitter/
- **Tree-sitter Playground**: https://tree-sitter.github.io/tree-sitter/playground
- **Reflex Architecture Guide**: See [ARCHITECTURE.md](ARCHITECTURE.md#extension-guide)

---

## Debugging Tips

### Enable Debug Logging

```bash
# All debug output
RUST_LOG=debug rfx query "pattern"

# Specific module
RUST_LOG=reflex::query=debug rfx query "pattern"

# Trace level (very verbose)
RUST_LOG=trace rfx query "pattern"
```

### Profile Performance

```bash
# Build with profiling enabled
cargo build --release --features profiling

# Run with profiler (Linux)
perf record --call-graph=dwarf ./target/release/rfx query "pattern"
perf report
```

### Inspect Cache Files

```bash
# View SQLite metadata
sqlite3 .reflex/meta.db ".schema"
sqlite3 .reflex/meta.db "SELECT * FROM files LIMIT 10;"

# View file hashes
cat .reflex/hashes.json | jq '.'
```

### Run Single Test with Output

```bash
cargo test test_name -- --nocapture --test-threads=1
```

---

## Release Process

Reflex uses **automated releases** via [release-plz](https://release-plz.ieni.dev/):

1. **Make changes** and commit using Conventional Commits
2. **Push to `main`** branch
3. **GitHub Action** automatically:
   - Analyzes commits
   - Determines next version
   - Updates `Cargo.toml` and `CHANGELOG.md`
   - Opens release PR
4. **Merge release PR** to create tag and GitHub Release

**Manual releases** are discouraged. See [CLAUDE.md Release Management](CLAUDE.md#release-management) for details.

---

## Getting Help

- **Documentation**: Start with [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md)
- **Issues**: Check existing [GitHub Issues](https://github.com/reflex-search/reflex/issues)
- **Discussions**: Open a [GitHub Discussion](https://github.com/reflex-search/reflex/discussions) for questions
- **Context**: Read [CLAUDE.md](CLAUDE.md) for project philosophy and workflow

---

## Code of Conduct

We expect all contributors to:

- Be respectful and constructive in discussions
- Focus on technical merit and project goals
- Help create a welcoming environment for all skill levels
- Report unacceptable behavior to the maintainers

---

## License

By contributing to Reflex, you agree that your contributions will be licensed under the MIT License.

See [LICENSE](LICENSE) for details.

---

## Recognition

Contributors will be recognized in:

- GitHub contributor list
- Release notes for significant contributions
- CHANGELOG.md for feature additions

Thank you for contributing to Reflex! 🚀
