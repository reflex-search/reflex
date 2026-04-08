//! LLM narration helpers for Pulse
//!
//! Provides centralized LLM calling for digest and wiki surfaces.
//! Handles provider setup, caching, content gating, and async bridging.

use anyhow::Result;
use std::path::Path;

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
- Identify key abstractions: main structs/classes, core functions, public API surface.
- Describe the module's architectural role: Is it a hub (many dependents)? A leaf (few dependents)? A bridge between subsystems?
- Explain how this module fits into the larger system — what it provides to modules that depend on it, and what it consumes from its own dependencies.
- If the module has high fan-in (many dependents), note that changes to it have wide blast radius.
- If the module has significantly more or fewer files/lines than average for the codebase, note that.
- Note complexity: file count, line count, symbol density.
- Reference specific file and function names.
- Vary your sentence structure. Do NOT repeat patterns across modules.
- Write 4-8 sentences. Be specific, not generic. Every sentence should contain at least one concrete name or number.
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

/// Post-process LLM narration output to fix common formatting issues.
fn postprocess_narration(text: &str) -> String {
    let mut result = text.trim().to_string();

    // Fix missing spaces after periods (e.g., "module.The" → "module. The" but not "config.toml")
    // Only insert space when followed by an uppercase letter (sentence boundary)
    let re = regex::Regex::new(r"([a-z])\.([A-Z])").unwrap();
    result = re.replace_all(&result, "$1. $2").to_string();

    // Fix missing spaces between lowercase and uppercase (e.g., "moduledrives" → "module drives")
    // Be conservative: only fix when a lowercase letter is directly followed by an uppercase
    // but NOT in common patterns like camelCase identifiers
    // We look for word-like patterns where the case transition happens mid-"word" without any code context
    let re = regex::Regex::new(r"([a-z]{3,})([A-Z][a-z]{2,})").unwrap();
    // Only apply this outside of backtick-quoted code
    let mut fixed = String::new();
    let mut in_code = false;
    for ch in result.chars() {
        if ch == '`' {
            in_code = !in_code;
        }
        fixed.push(ch);
    }
    // Apply the regex only to non-code segments
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
}
