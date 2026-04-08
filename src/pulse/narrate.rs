//! LLM narration helpers for Pulse
//!
//! Provides centralized LLM calling for digest and wiki surfaces.
//! Handles provider setup, caching, content gating, and async bridging.

use anyhow::Result;
use std::path::Path;
use std::sync::Arc;

use crate::semantic::config;
use crate::semantic::providers::{self, LlmProvider};

use super::llm_cache::LlmCache;

/// System prompt for digest section narration
const DIGEST_SYSTEM_PROMPT: &str = "\
You are a technical writer narrating a codebase change report.
You may ONLY describe facts present in the STRUCTURAL CONTEXT below.

Guidelines:
- Lead with the most significant structural change (largest file delta, new module, or dependency shift).
- Call out dependency graph changes: new edges, removed edges, cycle changes.
- Mention any threshold alerts or hotspot shifts with their numbers.
- Write 3-6 concise sentences with specific numbers and file/module names.
- Do NOT speculate about intent or add information not in the context.

STRUCTURAL CONTEXT:
";

/// System prompt for wiki module summary
const WIKI_SYSTEM_PROMPT: &str = "\
You are a technical writer creating a module overview for a codebase wiki.
You may ONLY describe facts present in the STRUCTURAL CONTEXT below.

CRITICAL RULES:
- NEVER start with 'The X module consists of...', 'This module contains...', or any variant.
- Your first sentence MUST state what the module DOES or what PURPOSE it serves — infer this from file names, symbol names, and its dependency position.
- Focus on PURPOSE, RESPONSIBILITIES, and ARCHITECTURAL ROLE — not on listing individual files or classes.
- Describe the module's architectural role: Is it a hub (many dependents)? A leaf (few dependents)? A bridge between subsystems?
- Explain how this module fits into the larger system — what it provides to modules that depend on it, and what it consumes from its own dependencies.
- If the module has high fan-in (many dependents), note that changes to it have wide blast radius.
- If the module has significantly more or fewer files/lines than average for the codebase, note that.
- Note complexity: file count, line count, symbol density.
- Do NOT enumerate specific file names, class names, or function names unless they represent a truly central abstraction that defines the module's identity (e.g., a primary entry point or the single core type). When in doubt, describe WHAT it does rather than naming the file that does it.
- Vary your sentence structure. Do NOT repeat patterns across modules.
- Write 4-8 sentences. Be specific about what the module does and its scale, not about which files it contains.
- Do NOT speculate about design intent or add information not in the context.
- NEVER leave missing spaces between words. Proofread your output.

STRUCTURAL CONTEXT:
";

/// System prompt for project overview narration
const PROJECT_OVERVIEW_SYSTEM_PROMPT: &str = "\
You are a technical writer creating a project overview for auto-generated codebase documentation.
You may ONLY describe facts present in the STRUCTURAL CONTEXT below.

CRITICAL RULES:
- NEVER start with 'This project consists of...' or 'The codebase is...'
- Your first sentence MUST describe what this software DOES — its purpose and primary function. Use evidence from module names and symbol names to infer the specific domain (e.g., 'code search' from TrigramIndex, QueryEngine, ParserFactory).
- Paragraph 1: What it does and how (infer from module names, key symbols, languages used).
- Paragraph 2: Architecture — how the major modules relate. Which modules are central hubs? What are the natural boundaries? Describe the data flow direction — which modules produce data and which consume it.
- Paragraph 3: Scale and notable patterns — file/line counts, language mix, dependency health (cycles, hotspots).
- Write exactly 3-4 paragraphs. Be specific: use module names, file counts, and dependency numbers.
- Do NOT speculate or add information not in the context.
- NEVER leave missing spaces between words. Proofread your output.

STRUCTURAL CONTEXT:
";

/// System prompt for architecture narrative narration
const ARCHITECTURE_NARRATIVE_SYSTEM_PROMPT: &str = "\
You are a technical writer narrating the architecture of a codebase based on its dependency graph.
You may ONLY describe facts present in the STRUCTURAL CONTEXT below.

CRITICAL RULES:
- NEVER start with 'The architecture consists of...' or 'This codebase is organized...'
- Lead with the most connected module and explain WHY it's central (what it provides to others).
- Describe data flow: which modules are producers (depended-on) vs consumers (depend on many).
- Identify if the codebase follows a layered pattern (e.g., parsers → models → query engine → CLI) and describe the information flow between layers.
- Identify natural boundaries: groups of tightly-coupled modules that form subsystems.
- Call out concerning patterns: circular dependencies, extreme fan-in hotspots, isolated modules.
- Note peripheral modules: what sits at the edges and what role they serve.
- Write 3-5 paragraphs. Every claim must reference specific module names and dependency counts.
- Do NOT speculate about design intent or add information not in the context.
- NEVER leave missing spaces between words. Proofread your output.

STRUCTURAL CONTEXT:
";

/// Minimum word count to attempt narration.
/// Sections below this threshold are too brief to produce useful summaries.
const MIN_CONTENT_WORDS: usize = 15;

/// Create an LLM provider using the user's ~/.reflex/config.toml (same config as `rfx ask`)
pub fn create_pulse_provider() -> Result<Box<dyn LlmProvider>> {
    let semantic_config = config::load_config(Path::new("."))?;
    let api_key = config::get_api_key(&semantic_config.provider)?;

    let model = if semantic_config.model.is_some() {
        semantic_config.model.clone()
    } else {
        config::get_user_model(&semantic_config.provider)
    };

    let options = config::get_provider_options(&semantic_config.provider);

    providers::create_provider(&semantic_config.provider, api_key, model, options)
}

/// Narrate a structural context block using LLM.
///
/// Returns `None` if:
/// - Content is too brief (fewer than MIN_CONTENT_WORDS words)
/// - LLM call fails (degrades gracefully, logs warning)
/// - Cache hit returns previously generated narration
///
/// Checks `LlmCache` first; stores response on success.
pub fn narrate_section(
    provider: &dyn LlmProvider,
    system_prompt: &str,
    structural_context: &str,
    cache: &LlmCache,
    snapshot_id: &str,
    cache_key_suffix: &str,
) -> Option<String> {
    // Check minimum content length
    let word_count = structural_context.split_whitespace().count();
    if word_count < MIN_CONTENT_WORDS {
        eprintln!("  Skipping: {} (too brief, {} words)", cache_key_suffix, word_count);
        return None;
    }

    // Check cache
    let cache_key = LlmCache::compute_key(snapshot_id, cache_key_suffix, structural_context);
    match cache.get(&cache_key) {
        Ok(Some(cached)) => {
            log::debug!("LLM cache hit for '{}'", cache_key_suffix);
            eprintln!("  Narrating: {} (cached)", cache_key_suffix);
            return Some(cached.response);
        }
        Ok(None) => {}
        Err(e) => {
            log::warn!("Failed to read LLM cache: {}", e);
        }
    }

    // Build prompt
    let prompt = format!("{}{}", system_prompt, structural_context);

    eprintln!("  Narrating: {}...", cache_key_suffix);

    // Call LLM with retry (sync bridge over async)
    let result = call_llm_sync(provider, &prompt);

    match result {
        Ok(response) => {
            let response = postprocess_narration(&response);

            // Cache the response
            let context_hash = blake3::hash(structural_context.as_bytes()).to_hex().to_string();
            if let Err(e) = cache.put(&cache_key, &context_hash, &response) {
                log::warn!("Failed to write LLM cache: {}", e);
            }

            Some(response)
        }
        Err(e) => {
            log::warn!("LLM narration failed for '{}': {}", cache_key_suffix, e);
            None
        }
    }
}

/// A narration task for batch dispatch
pub struct NarrationTask {
    pub system_prompt: &'static str,
    pub structural_context: String,
    pub snapshot_id: String,
    pub cache_key_suffix: String,
}

/// Result of a narration task
pub struct NarrationResult {
    pub cache_key_suffix: String,
    pub response: Option<String>,
}

/// Narrate multiple sections concurrently using a single tokio runtime.
///
/// Pre-filters cache hits and too-brief content. Remaining tasks are dispatched
/// concurrently with a semaphore bound. Results are returned in order.
pub fn narrate_batch(
    provider: Arc<dyn LlmProvider>,
    tasks: Vec<NarrationTask>,
    cache: &LlmCache,
    concurrency: usize,
) -> Vec<NarrationResult> {
    let total = tasks.len();
    if total == 0 {
        return Vec::new();
    }

    // Pre-filter: resolve cache hits and too-brief content synchronously
    let mut results: Vec<NarrationResult> = Vec::with_capacity(total);
    let mut pending: Vec<(usize, NarrationTask, String)> = Vec::new(); // (result_index, task, cache_key)

    for task in tasks {
        let word_count = task.structural_context.split_whitespace().count();
        if word_count < MIN_CONTENT_WORDS {
            eprintln!("  Skipping: {} (too brief, {} words)", task.cache_key_suffix, word_count);
            results.push(NarrationResult {
                cache_key_suffix: task.cache_key_suffix,
                response: None,
            });
            continue;
        }

        let cache_key = LlmCache::compute_key(
            &task.snapshot_id,
            &task.cache_key_suffix,
            &task.structural_context,
        );
        match cache.get(&cache_key) {
            Ok(Some(cached)) => {
                eprintln!("  Narrating: {} (cached)", task.cache_key_suffix);
                results.push(NarrationResult {
                    cache_key_suffix: task.cache_key_suffix,
                    response: Some(cached.response),
                });
            }
            _ => {
                let idx = results.len();
                results.push(NarrationResult {
                    cache_key_suffix: task.cache_key_suffix.clone(),
                    response: None,
                });
                pending.push((idx, task, cache_key));
            }
        }
    }

    if pending.is_empty() {
        return results;
    }

    let pending_count = pending.len();
    let effective_concurrency = if concurrency == 0 { pending_count } else { concurrency };
    eprintln!(
        "  Dispatching {} LLM calls ({} concurrent)...",
        pending_count, effective_concurrency
    );

    // Clone cache_dir for use inside async tasks
    let cache_dir = cache.cache_dir().to_path_buf();

    // Single tokio runtime for all concurrent LLM calls
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            log::warn!("Failed to create tokio runtime for batch narration: {}", e);
            return results;
        }
    };

    let async_results = rt.block_on(async {
        let semaphore = Arc::new(tokio::sync::Semaphore::new(effective_concurrency));
        let mut join_set = tokio::task::JoinSet::new();

        for (idx, task, cache_key) in pending {
            let provider = Arc::clone(&provider);
            let sem = Arc::clone(&semaphore);
            let cache_dir = cache_dir.clone();

            join_set.spawn(async move {
                let _permit = sem.acquire().await.expect("semaphore closed");
                let start = std::time::Instant::now();
                eprintln!("  Narrating: {}...", task.cache_key_suffix);

                let prompt = format!("{}{}", task.system_prompt, task.structural_context);
                let result = call_llm_async(&*provider, &prompt).await;

                let response = match result {
                    Ok(raw) => {
                        let response = postprocess_narration(&raw);

                        // Write to cache (file-based, unique key per task — no conflicts)
                        let task_cache = LlmCache::from_dir(cache_dir);
                        let context_hash = blake3::hash(task.structural_context.as_bytes())
                            .to_hex()
                            .to_string();
                        if let Err(e) = task_cache.put(&cache_key, &context_hash, &response) {
                            log::warn!("Failed to write LLM cache for '{}': {}", task.cache_key_suffix, e);
                        }

                        eprintln!(
                            "  Narrating: {} (done, {:.1}s)",
                            task.cache_key_suffix,
                            start.elapsed().as_secs_f64()
                        );
                        Some(response)
                    }
                    Err(e) => {
                        log::warn!("LLM narration failed for '{}': {}", task.cache_key_suffix, e);
                        eprintln!(
                            "  Narrating: {} (failed, {:.1}s)",
                            task.cache_key_suffix,
                            start.elapsed().as_secs_f64()
                        );
                        None
                    }
                };

                (idx, task.cache_key_suffix, response)
            });
        }

        let mut async_results = Vec::new();
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(r) => async_results.push(r),
                Err(e) => log::warn!("Narration task panicked: {}", e),
            }
        }
        async_results
    });

    // Distribute results back
    for (idx, cache_key_suffix, response) in async_results {
        results[idx] = NarrationResult {
            cache_key_suffix,
            response,
        };
    }

    results
}

/// Async LLM call with retry logic (native async, no per-call Runtime)
async fn call_llm_async(provider: &dyn LlmProvider, prompt: &str) -> Result<String> {
    let max_retries = 2;
    let mut last_error = None;

    for attempt in 0..=max_retries {
        if attempt > 0 {
            log::debug!("Retrying LLM narration (attempt {}/{})", attempt + 1, max_retries + 1);
            tokio::time::sleep(tokio::time::Duration::from_millis(500 * attempt as u64)).await;
        }

        match provider.complete(prompt, false).await {
            Ok(response) => return Ok(response),
            Err(e) => {
                log::debug!("LLM call attempt {} failed: {}", attempt + 1, e);
                last_error = Some(e);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("LLM call failed")))
}

/// Get the system prompt for digest narration
pub fn digest_system_prompt() -> &'static str {
    DIGEST_SYSTEM_PROMPT
}

/// Get the system prompt for wiki narration
pub fn wiki_system_prompt() -> &'static str {
    WIKI_SYSTEM_PROMPT
}

/// Get the system prompt for project overview narration
pub fn project_overview_system_prompt() -> &'static str {
    PROJECT_OVERVIEW_SYSTEM_PROMPT
}

/// Get the system prompt for architecture narrative narration
pub fn architecture_narrative_system_prompt() -> &'static str {
    ARCHITECTURE_NARRATIVE_SYSTEM_PROMPT
}

/// Known compound words / proper nouns that should NOT be split by camelCase regex.
/// These are common technical terms found in codebases.
const CAMEL_CASE_BLOCKLIST: &[&str] = &[
    "TypeScript", "JavaScript", "CoffeeScript", "ActionScript",
    "PostgreSQL", "MySQL", "MariaDB", "MongoDB", "CouchDB", "GraphQL",
    "GitHub", "GitLab", "BitBucket", "WordPress", "PostCSS",
    "IntelliJ", "WebSocket", "WebAssembly", "DevOps", "DevTools",
    "DataFrame", "NumPy", "PyTorch", "TensorFlow", "FastAPI",
    "NextJS", "NestJS", "NodeJS", "ExpressJS", "AngularJS",
    "iPhone", "iPad", "macOS", "iOS", "FreeBSD", "OpenBSD",
    "CodePen", "CodeSandbox", "JetBrains", "PhpStorm", "AppKit",
    "SwiftUI", "UIKit", "CoreData", "MapReduce",
    "CloudFormation", "CloudFront", "CloudWatch",
    "RedHat", "OpenShift", "OpenStack",
    "SourceMap", "AutoComplete", "IntelliSense",
];

/// Post-process LLM narration output to fix common formatting issues.
fn postprocess_narration(text: &str) -> String {
    let mut result = text.trim().to_string();

    // Fix missing spaces after periods (e.g., "module.The" → "module. The" but not "config.toml")
    // Only insert space when followed by an uppercase letter (sentence boundary)
    let re = regex::Regex::new(r"([a-z])\.([A-Z])").unwrap();
    result = re.replace_all(&result, "$1. $2").to_string();

    // Fix missing spaces between lowercase and uppercase (e.g., "moduledrives" → "module drives")
    // Protect known compound words with placeholders before applying the regex
    let mut placeholders: Vec<(&str, String)> = Vec::new();
    for (i, term) in CAMEL_CASE_BLOCKLIST.iter().enumerate() {
        if result.contains(*term) {
            let placeholder = format!("\x00KEEP{}\x00", i);
            result = result.replace(*term, &placeholder);
            placeholders.push((term, placeholder));
        }
    }

    // Apply camelCase splitting only to non-code segments (outside backticks)
    let re = regex::Regex::new(r"([a-z]{3,})([A-Z][a-z]{2,})").unwrap();
    let parts: Vec<&str> = result.split('`').collect();
    let mut assembled = String::new();
    for (i, part) in parts.iter().enumerate() {
        if i % 2 == 0 {
            // Outside backticks — apply fix
            assembled.push_str(&re.replace_all(part, "$1 $2"));
        } else {
            // Inside backticks — preserve as-is
            assembled.push('`');
            assembled.push_str(part);
            assembled.push('`');
        }
    }
    result = assembled;

    // Restore protected compound words
    for (term, placeholder) in &placeholders {
        result = result.replace(placeholder, term);
    }

    // Fix double spaces
    while result.contains("  ") {
        result = result.replace("  ", " ");
    }

    result
}

/// Synchronous LLM call with retry logic.
/// Uses tokio runtime to bridge async provider calls.
fn call_llm_sync(provider: &dyn LlmProvider, prompt: &str) -> Result<String> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let mut last_error = None;
        let max_retries = 2;

        for attempt in 0..=max_retries {
            if attempt > 0 {
                log::debug!("Retrying LLM narration (attempt {}/{})", attempt + 1, max_retries + 1);
                tokio::time::sleep(tokio::time::Duration::from_millis(500 * attempt as u64)).await;
            }

            match provider.complete(prompt, false).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    log::debug!("LLM call attempt {} failed: {}", attempt + 1, e);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("LLM call failed")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_count_sufficient() {
        // 15+ words should pass the gate
        let text = "src/parsers/rust.rs has 250 lines and contains extract_symbols fn_name and other important functions used for parsing code";
        let count = text.split_whitespace().count();
        assert!(count >= MIN_CONTENT_WORDS, "Word count {} should be >= {}", count, MIN_CONTENT_WORDS);
    }

    #[test]
    fn test_word_count_too_brief() {
        // < 15 words should be rejected
        let text = "No data available yet.";
        let count = text.split_whitespace().count();
        assert!(count < MIN_CONTENT_WORDS, "Word count {} should be < {}", count, MIN_CONTENT_WORDS);
    }

    #[test]
    fn test_word_count_empty() {
        let count = "".split_whitespace().count();
        assert!(count < MIN_CONTENT_WORDS);
    }

    #[test]
    fn test_word_count_wiki_structural() {
        // Typical wiki page with markdown table + file list should pass
        let text = "| Language | Files | Lines |\n| --- | --- | --- |\n| Rust | 45 | 12,500 |\n\n**Files:** src/main.rs src/lib.rs src/query/mod.rs src/parsers/rust.rs";
        let count = text.split_whitespace().count();
        assert!(count >= MIN_CONTENT_WORDS, "Wiki structural word count {} should be >= {}", count, MIN_CONTENT_WORDS);
    }

    #[test]
    fn test_word_count_digest_bootstrap() {
        // Typical digest with structural data should pass
        let text = "Branch: feature/pulse Commit: abc1234 Files: 120 Edges: 340 Modules: src tests build.rs config.toml main.rs lib.rs";
        let count = text.split_whitespace().count();
        assert!(count >= MIN_CONTENT_WORDS, "Digest bootstrap word count {} should be >= {}", count, MIN_CONTENT_WORDS);
    }

    #[test]
    fn test_digest_system_prompt() {
        assert!(digest_system_prompt().contains("STRUCTURAL CONTEXT"));
    }

    #[test]
    fn test_wiki_system_prompt() {
        assert!(wiki_system_prompt().contains("STRUCTURAL CONTEXT"));
    }

    #[test]
    fn test_postprocess_preserves_proper_nouns() {
        let input = "The TypeScript module handles JavaScript compilation.";
        let result = postprocess_narration(input);
        assert!(result.contains("TypeScript"), "Should preserve TypeScript, got: {}", result);
        assert!(result.contains("JavaScript"), "Should preserve JavaScript, got: {}", result);
    }

    #[test]
    fn test_postprocess_splits_run_on_words() {
        // "moduledrives" should become "module drives"
        let input = "The parseModule drives the query engine.";
        let result = postprocess_narration(input);
        assert!(result.contains("parse Module"), "Should split run-on camelCase: {}", result);
    }

    #[test]
    fn test_postprocess_preserves_backtick_code() {
        let input = "Uses `TypeScript` and `parseModule` for processing.";
        let result = postprocess_narration(input);
        assert!(result.contains("`TypeScript`"), "Should preserve code: {}", result);
        assert!(result.contains("`parseModule`"), "Should preserve code: {}", result);
    }

    #[test]
    fn test_postprocess_fixes_missing_sentence_space() {
        let input = "First sentence.Second sentence starts here.";
        let result = postprocess_narration(input);
        assert!(result.contains(". S"), "Should add space after period: {}", result);
    }

    #[test]
    fn test_postprocess_fixes_double_spaces() {
        let input = "Too  many  spaces  here.";
        let result = postprocess_narration(input);
        assert!(!result.contains("  "), "Should remove double spaces: {}", result);
    }
}
