//! Prompts and task builders for the current Pulse surfaces.
//!
//! Each builder returns a [`WriteTask`] for the writing pass
//! ([`crate::pulse::write`]), which owns calling, retries, caching and degradation.
//! The system prompt stays fixed per task kind (so providers can cache it); the
//! structural context goes in the user message under a labelled header.
//!
//! These prompts are superseded by grounded evidence packs in a later milestone.

use super::write::{OutputSpec, WriteTask};
use serde_json::json;

/// Task ids, used to read results back from a [`crate::pulse::write::WriteOutcome`].
pub mod ids {
    pub const CHANGELOG: &str = "changelog";
    pub const ARCHITECTURE: &str = "architecture";
    pub const ONBOARD: &str = "onboard-guide";
    pub const TIMELINE: &str = "timeline-summary";
    pub const GLOSSARY: &str = "glossary";
    pub const OVERVIEW: &str = "project-overview";

    /// Id of the summary task for one wiki module.
    pub fn module(path: &str) -> String {
        format!("module:{path}")
    }
}

/// System prompt for changelog narration
const CHANGELOG_SYSTEM_PROMPT: &str = "\
You are a technical writer creating a product-level changelog from recent development activity.
Your audience is developers and stakeholders who want to understand what changed, why, and what it impacts — NOT the raw commit details.

Guidelines:
- Group related commits into 3-8 high-level changelog entries.
- Each entry needs a clear title (what changed) and a 2-4 sentence description (why it matters, what it impacts).
- Include an approximate date or date range in parentheses after each entry's title, like \"Added search (Apr 10–12)\".
- Write at a product/feature level, not code level. Say \"Added search to documentation\" not \"Integrated pagefind library into site.rs\".
- Focus on user-visible impact and system-level consequences.
- Do NOT include commit hashes, file paths, or diff statistics in your output.
- Do NOT speculate beyond what the commit messages and file changes reveal.

Output VALID JSON:
{
  \"entries\": [
    {
      \"title\": \"Short descriptive title (Apr 10–12)\",
      \"description\": \"2-4 sentences explaining what changed, why, and what it impacts.\"
    }
  ]
}
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
";

/// System prompt for onboard guide narration
const ONBOARD_SYSTEM_PROMPT: &str = "\
You are a technical writer creating a \"Getting Started\" guide for a developer's first day on this codebase.
You may ONLY describe facts present in the STRUCTURAL CONTEXT below.

CRITICAL RULES:
- Write 4-5 paragraphs in plain language that a new team member could follow.
- Paragraph 1: What this project does — its purpose and primary function, in one or two sentences a non-developer could understand.
- Paragraph 2: How the code is organized — the major directories/modules, what each is responsible for.
- Paragraph 3: Where to start reading — which entry points to look at first, and why.
- Paragraph 4: Key patterns and conventions — recurring design patterns, naming conventions, or architectural idioms a newcomer should know.
- Use specific file and module names from the context.
- Do NOT speculate or add information not in the context.
- NEVER leave missing spaces between words. Proofread your output.
";

/// System prompt for timeline narration
const TIMELINE_SYSTEM_PROMPT: &str = "\
You are a technical writer summarizing recent development activity for a codebase.
You may ONLY describe facts present in the STRUCTURAL CONTEXT below.

CRITICAL RULES:
- Lead with the most active area of the codebase and explain what's happening there.
- Identify stable modules (few recent changes) vs evolving modules (many recent changes).
- Flag high-churn files that may warrant attention — files changing very frequently could indicate active development or instability.
- Note contributor patterns — is this a solo project or a team effort? Who owns which areas?
- Write 3-5 concise paragraphs with specific numbers, file names, and module names.
- Do NOT speculate about intent or add information not in the context.
- NEVER leave missing spaces between words. Proofread your output.
";

/// System prompt for product-concept glossary generation.
///
/// The LLM receives structural evidence (module paths, anchor symbol names,
/// scale stats) and returns a single JSON document containing an intro
/// paragraph plus ~10-15 high-level product concepts with plain-language
/// definitions. The response is parsed in `glossary::parse_concepts_response`.
const CONCEPTS_SYSTEM_PROMPT: &str = "\
You are documenting a software product's core vocabulary for a non-technical reader.

From the structural evidence below, identify 10-15 HIGH-LEVEL product concepts that someone needs to understand to know what this product DOES and how it works. Concepts are NOUN PHRASES describing capabilities, data ideas, or workflows — NOT specific class names, function names, or file names.

GOOD concept examples: 'Trigram Index', 'Symbol Cache', 'AST Query', 'Dependency Graph', 'LLM Narration', 'Runtime Symbol Detection'
BAD concept examples: 'SearchResult struct', 'QueryEngine class', 'extract_symbols function'

Rules:
- Each definition must be 1-3 sentences in plain language a product person could understand.
- Do NOT start definitions with 'This is a...', 'Represents a...', 'A struct that...'
- Group concepts into 2-4 categories of your choice (e.g. 'Core Capabilities', 'Data Model', 'Workflows', 'Developer Tools').
- Anchor each concept to 1-3 module paths from the evidence — these become wiki links.
- Write exactly ONE intro paragraph (2-3 sentences) describing what kind of vocabulary this page catalogs for this specific product.

Output VALID JSON MATCHING THIS SCHEMA EXACTLY — no markdown fences, no commentary before or after:
{
  \"intro\": \"...\",
  \"concepts\": [
    {
      \"name\": \"Concept Name\",
      \"category\": \"Category Name\",
      \"definition\": \"1-3 sentence plain-language definition.\",
      \"related_modules\": [\"src/foo\", \"src/bar\"]
    }
  ]
}
";

fn with_header(header: &str, context: &str) -> String {
    format!("{header}\n{context}")
}

/// Summary of one wiki module.
pub fn wiki_task(module_path: &str, context: &str) -> WriteTask {
    WriteTask::text(
        ids::module(module_path),
        "modules",
        WIKI_SYSTEM_PROMPT,
        with_header("STRUCTURAL CONTEXT:", context),
    )
    .with_max_tokens(900)
    .with_priority(60)
}

/// Product-level changelog entries, as JSON.
pub fn changelog_task(context: &str) -> WriteTask {
    let entry = json!({
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"}
        },
        "required": ["title", "description"],
        "additionalProperties": false
    });
    let schema = json!({
        "type": "object",
        "properties": {"entries": {"type": "array", "items": entry}},
        "required": ["entries"],
        "additionalProperties": false
    });
    WriteTask::text(
        ids::CHANGELOG,
        "changelog",
        CHANGELOG_SYSTEM_PROMPT,
        with_header("COMMIT DATA:", context),
    )
    .with_output(OutputSpec::JsonSchema {
        name: "changelog".into(),
        schema,
    })
    .with_max_tokens(1500)
    .with_priority(80)
}

/// Architecture narrative for the map page.
pub fn architecture_task(context: &str) -> WriteTask {
    WriteTask::text(
        ids::ARCHITECTURE,
        "architecture",
        ARCHITECTURE_NARRATIVE_SYSTEM_PROMPT,
        with_header("STRUCTURAL CONTEXT:", context),
    )
    .with_max_tokens(1400)
    .with_priority(20)
}

/// First-day guide for the onboarding page.
pub fn onboard_task(context: &str) -> WriteTask {
    WriteTask::text(
        ids::ONBOARD,
        "guides",
        ONBOARD_SYSTEM_PROMPT,
        with_header("STRUCTURAL CONTEXT:", context),
    )
    .with_max_tokens(1400)
    .with_priority(15)
}

/// Summary of recent development activity.
pub fn timeline_task(context: &str) -> WriteTask {
    WriteTask::text(
        ids::TIMELINE,
        "timeline",
        TIMELINE_SYSTEM_PROMPT,
        with_header("STRUCTURAL CONTEXT:", context),
    )
    .with_max_tokens(1200)
    .with_priority(90)
}

/// Product concepts for the glossary, as JSON.
pub fn concepts_task(context: &str) -> WriteTask {
    let concept = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "category": {"type": "string"},
            "definition": {"type": "string"},
            "related_modules": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["name", "category", "definition", "related_modules"],
        "additionalProperties": false
    });
    let schema = json!({
        "type": "object",
        "properties": {
            "intro": {"type": "string"},
            "concepts": {"type": "array", "items": concept}
        },
        "required": ["intro", "concepts"],
        "additionalProperties": false
    });
    WriteTask::text(
        ids::GLOSSARY,
        "glossary",
        CONCEPTS_SYSTEM_PROMPT,
        with_header("STRUCTURAL EVIDENCE:", context),
    )
    .with_output(OutputSpec::JsonSchema {
        name: "concepts".into(),
        schema,
    })
    .with_max_tokens(2500)
    .with_priority(70)
}

/// Project overview for the home page.
pub fn overview_task(context: &str) -> WriteTask {
    WriteTask::text(
        ids::OVERVIEW,
        "overview",
        PROJECT_OVERVIEW_SYSTEM_PROMPT,
        with_header("STRUCTURAL CONTEXT:", context),
    )
    .with_max_tokens(1200)
    .with_priority(10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_prompts_are_stable_and_headers_move_to_user() {
        let t = wiki_task("src/pulse", "ctx");
        assert!(t.system.contains("STRUCTURAL CONTEXT below"));
        assert!(!t.system.trim_end().ends_with("STRUCTURAL CONTEXT:"));
        assert_eq!(t.user, "STRUCTURAL CONTEXT:\nctx");
        assert_eq!(t.id, "module:src/pulse");
        assert_eq!(t.kind, "modules");

        let c = changelog_task("commits");
        assert!(c.system.contains("Output VALID JSON"));
        assert!(c.user.starts_with("COMMIT DATA:\n"));
    }

    #[test]
    fn json_tasks_declare_strict_schemas() {
        for t in [changelog_task("x"), concepts_task("x")] {
            let OutputSpec::JsonSchema { schema, .. } = &t.output else {
                panic!("{} should use a JSON schema", t.id);
            };
            assert_eq!(schema["additionalProperties"], false);
            assert!(schema["required"].as_array().is_some());
        }
    }

    #[test]
    fn task_ids_are_unique() {
        let tasks = [
            changelog_task("x"),
            architecture_task("x"),
            onboard_task("x"),
            timeline_task("x"),
            concepts_task("x"),
            overview_task("x"),
            wiki_task("src", "x"),
        ];
        let mut ids: Vec<&str> = tasks.iter().map(|t| t.id.as_str()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), tasks.len());
    }
}
