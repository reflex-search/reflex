# Reflex Makefile
# Run `make help` to see available targets.

CARGO ?= cargo

.DEFAULT_GOAL := help

.PHONY: help build release check install uninstall \
        fmt fmt-check clippy test test-verbose \
        pre-commit ci \
        index serve mcp \
        doc doc-open \
        clean clean-cache clean-all

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "; printf "\nUsage: make \033[36m<target>\033[0m\n\nTargets:\n"} \
		/^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# --- Build ---------------------------------------------------------------

build: ## Debug build
	$(CARGO) build

release: ## Optimized release build (target/release/rfx)
	$(CARGO) build --release

check: ## Fast type-check, no codegen
	$(CARGO) check

install: ## Install rfx to ~/.cargo/bin
	$(CARGO) install --path .

uninstall: ## Remove installed rfx binary
	$(CARGO) uninstall reflex-search

# --- Quality -------------------------------------------------------------

fmt: ## Format code
	$(CARGO) fmt

fmt-check: ## Verify formatting (CI-safe, no writes)
	$(CARGO) fmt -- --check

clippy: ## Lint with warnings as errors
	$(CARGO) clippy --all-targets -- -D warnings

test: ## Run unit + integration tests
	$(CARGO) test

test-verbose: ## Run tests with stdout shown
	$(CARGO) test -- --nocapture

# --- Composite -----------------------------------------------------------

pre-commit: fmt clippy test ## Run fmt + clippy + test before pushing

ci: fmt-check clippy test ## Suite a CI runner should execute

# --- Run -----------------------------------------------------------------

index: ## Build the index cache for the current directory
	$(CARGO) run --release -- index

serve: ## Start HTTP server on port 7878
	$(CARGO) run --release -- serve

mcp: ## Start MCP server
	$(CARGO) run --release -- mcp

# --- Docs ----------------------------------------------------------------

doc: ## Generate API docs (no deps)
	$(CARGO) doc --no-deps

doc-open: ## Generate API docs and open in browser
	$(CARGO) doc --no-deps --open

# --- Maintenance ---------------------------------------------------------

clean: ## Remove target/ build artifacts
	$(CARGO) clean

clean-cache: ## Remove local .reflex/ index cache
	rm -rf .reflex

clean-all: clean clean-cache ## Clean both build artifacts and index cache
