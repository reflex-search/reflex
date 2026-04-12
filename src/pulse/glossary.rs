//! Glossary: product-level vocabulary
//!
//! The glossary surfaces ~10-15 *product concepts* — high-level noun phrases
//! that describe what the software does (capabilities, data ideas, workflows),
//! not specific Rust types or function names.
//!
//! Unlike v2, we do **not** rank symbols from the cache. Instead we collect
//! a compact "structural evidence" bundle (module paths + a handful of anchor
//! symbol names per module) and hand it to the LLM in a single narration
//! task. The LLM is responsible for selecting concepts, writing definitions,
//! grouping into categories, and anchoring each concept to the modules that
//! implement it. We then render the result as a card-based markdown page.
//!
//! In `--no-llm` mode (or if the LLM call fails), the page falls back to a
//! minimal message directing the user to re-run with LLM enabled, plus a
//! module list derived from the same structural evidence.

use anyhow::{Context, Result};
use rusqlite::Connection;
use serde::Deserialize;
use std::collections::HashMap;

use crate::cache::CacheManager;
use crate::models::SearchResult;

/// How many anchor symbol names to pull per module as LLM evidence.
const ANCHOR_SYMBOLS_PER_MODULE: usize = 5;

/// Maximum number of modules to include in the evidence bundle. Keeps the
/// prompt bounded on very wide repositories.
const MAX_MODULES_IN_EVIDENCE: usize = 25;

/// A single product-level concept, as decided and written by the LLM.
#[derive(Debug, Clone)]
pub struct Concept {
    /// Human-readable concept name (e.g. "Trigram Index").
    pub name: String,
    /// 1-3 sentence plain-language definition.
    pub definition: String,
    /// Module paths (e.g. "src/index", "src/query") that the LLM anchored
    /// this concept to. Used to render wiki links in the card footer.
    pub related_modules: Vec<String>,
    /// LLM-assigned category bucket (e.g. "Core Capabilities", "Data Model").
    pub category: Option<String>,
}

/// Full glossary data rendered on the `/glossary/` page.
#[derive(Debug, Clone, Default)]
pub struct GlossaryData {
    pub concepts: Vec<Concept>,
    /// 2-3 sentence LLM-written intro paragraph for the page.
    pub intro: Option<String>,
}

/// Summary of one module for the LLM evidence bundle.
#[derive(Debug, Clone)]
pub struct ModuleEvidence {
    /// Module path (e.g. "src/pulse").
    pub path: String,
    /// Number of files in the module.
    pub file_count: usize,
    /// Top-N anchor symbol names (strings only, no kind or location).
    pub anchor_symbols: Vec<String>,
}

/// Structural evidence handed to the LLM to let it pick product concepts.
#[derive(Debug, Clone, Default)]
pub struct GlossaryEvidence {
    pub total_files: usize,
    pub total_lines: usize,
    pub language_mix: Vec<(String, usize)>,
    pub dependency_edges: usize,
    pub hotspot_files: Vec<String>,
    pub modules: Vec<ModuleEvidence>,
}

/// Raw JSON shape returned by the LLM. Deserialized then lifted into
/// [`GlossaryData`].
#[derive(Debug, Clone, Deserialize)]
pub struct ConceptsResponse {
    #[serde(default)]
    pub intro: Option<String>,
    #[serde(default)]
    pub concepts: Vec<RawConcept>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawConcept {
    pub name: String,
    #[serde(default)]
    pub definition: String,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub related_modules: Vec<String>,
}

impl From<RawConcept> for Concept {
    fn from(raw: RawConcept) -> Self {
        Concept {
            name: raw.name,
            definition: raw.definition,
            category: raw.category,
            related_modules: raw.related_modules,
        }
    }
}

impl From<ConceptsResponse> for GlossaryData {
    fn from(resp: ConceptsResponse) -> Self {
        GlossaryData {
            concepts: resp.concepts.into_iter().map(Into::into).collect(),
            intro: resp.intro,
        }
    }
}

/// Derive a top-two-segment module path from a file path.
///
/// - `src/models.rs` → `src`
/// - `src/pulse/wiki.rs` → `src/pulse`
/// - `src/parsers/rust/mod.rs` → `src/parsers`
fn module_of(file_path: &str) -> String {
    let parts: Vec<&str> = file_path.split('/').collect();
    match parts.len() {
        0 | 1 => String::new(),
        2 => parts[0].to_string(),
        _ => format!("{}/{}", parts[0], parts[1]),
    }
}

/// Convert a module path like `src/pulse` into its wiki slug (`src-pulse`).
fn module_slug(module_path: &str) -> String {
    module_path.replace('/', "-")
}

/// Relative "weight" used only to sort anchor symbols within a module so that
/// type-like names (Struct, Trait, Enum) come before Functions before
/// Variables. This is *not* a filter — every non-Variable kind may contribute
/// anchor names.
fn anchor_priority(kind: &str) -> u8 {
    match kind.to_lowercase().as_str() {
        "struct" | "class" | "trait" | "interface" | "enum" | "type" | "typedef" => 0,
        "function" | "method" | "macro" | "module" => 1,
        "constant" | "property" | "event" | "attribute" | "export" => 2,
        // Variables, imports, and unknowns get the lowest priority; the plan
        // explicitly notes variables clutter the evidence.
        _ => 3,
    }
}

/// Collect the structural evidence bundle that will be handed to the LLM for
/// concept selection. Cheap: a handful of SQL queries plus symbol-name
/// extraction, no tree-sitter parsing.
///
/// Returns `Ok(None)` if the cache exists but has no symbols table (nothing
/// to anchor concepts to).
pub fn collect_glossary_evidence(cache: &CacheManager) -> Result<Option<GlossaryEvidence>> {
    let db_path = cache.path().join("meta.db");
    let conn = Connection::open(&db_path).context("Failed to open meta.db")?;

    let has_symbols: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='symbols'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map(|c| c > 0)
        .unwrap_or(false);

    if !has_symbols {
        return Ok(None);
    }

    let total_files: usize = conn
        .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
        .unwrap_or(0);
    let total_lines: usize = conn
        .query_row("SELECT COALESCE(SUM(line_count), 0) FROM files", [], |r| {
            r.get(0)
        })
        .unwrap_or(0);

    // Language mix (top 10).
    let mut language_mix: Vec<(String, usize)> = Vec::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT COALESCE(language, 'other'), COUNT(*) FROM files \
         GROUP BY language ORDER BY COUNT(*) DESC LIMIT 10",
    ) {
        if let Ok(rows) =
            stmt.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?)))
        {
            language_mix = rows.flatten().collect();
        }
    }

    // Dependency edge count (best-effort; may be 0 if table absent).
    let dependency_edges: usize = conn
        .query_row::<usize, _, _>(
            "SELECT COUNT(*) FROM file_dependencies WHERE resolved_file_id IS NOT NULL",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    // Top hotspot files (most-imported) — good anchor hints for the LLM.
    let mut hotspot_files: Vec<String> = Vec::new();
    if dependency_edges > 0 {
        if let Ok(mut stmt) = conn.prepare(
            "SELECT f.path, COUNT(DISTINCT fd.file_id) as dep_count \
             FROM file_dependencies fd JOIN files f ON fd.resolved_file_id = f.id \
             GROUP BY fd.resolved_file_id ORDER BY dep_count DESC LIMIT 8",
        ) {
            if let Ok(rows) = stmt.query_map([], |row| row.get::<_, String>(0)) {
                hotspot_files = rows.flatten().collect();
            }
        }
    }

    // Walk the symbols table once and bucket symbol names by module path.
    // For each module we keep up to `ANCHOR_SYMBOLS_PER_MODULE` names, sorted
    // by anchor priority (types before functions before constants, etc.).
    let mut stmt = conn.prepare(
        "SELECT s.symbols_json, f.path, f.line_count \
         FROM symbols s JOIN files f ON s.file_id = f.id",
    )?;
    let rows: Vec<(String, String, usize)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, usize>(2).unwrap_or(0),
            ))
        })?
        .filter_map(|r| r.ok())
        .collect();

    #[derive(Default)]
    struct ModuleBucket {
        file_count: usize,
        // (priority, name) — kept in a Vec so we can dedupe then truncate.
        candidates: Vec<(u8, String)>,
    }

    let mut by_module: HashMap<String, ModuleBucket> = HashMap::new();

    for (symbols_json, file_path, _line_count) in rows {
        let module = module_of(&file_path);
        if module.is_empty() {
            continue;
        }
        let bucket = by_module.entry(module.clone()).or_default();
        bucket.file_count += 1;

        let symbols: Vec<SearchResult> = match serde_json::from_str(&symbols_json) {
            Ok(s) => s,
            Err(_) => continue,
        };

        for sr in symbols {
            let Some(name) = sr.symbol else { continue };
            if name.len() < 3 {
                continue;
            }
            let kind_str = sr.kind.to_string();
            // Skip the noisiest kinds outright.
            let kl = kind_str.to_lowercase();
            if kl == "variable" || kl == "import" || kl == "export" || kl == "unknown" {
                continue;
            }
            let priority = anchor_priority(&kind_str);
            bucket.candidates.push((priority, name));
        }
    }

    // Build per-module evidence. Sort candidates by priority then by name for
    // determinism, dedupe, and truncate.
    let mut modules: Vec<ModuleEvidence> = by_module
        .into_iter()
        .map(|(path, mut bucket)| {
            bucket
                .candidates
                .sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            let mut anchors: Vec<String> = Vec::new();
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            for (_, name) in bucket.candidates {
                if seen.insert(name.clone()) {
                    anchors.push(name);
                    if anchors.len() >= ANCHOR_SYMBOLS_PER_MODULE {
                        break;
                    }
                }
            }
            ModuleEvidence {
                path,
                file_count: bucket.file_count,
                anchor_symbols: anchors,
            }
        })
        .collect();

    // Largest modules first, then alphabetical. Cap at MAX_MODULES_IN_EVIDENCE.
    modules.sort_by(|a, b| b.file_count.cmp(&a.file_count).then_with(|| a.path.cmp(&b.path)));
    modules.truncate(MAX_MODULES_IN_EVIDENCE);

    Ok(Some(GlossaryEvidence {
        total_files,
        total_lines,
        language_mix,
        dependency_edges,
        hotspot_files,
        modules,
    }))
}

/// Build the structural evidence block that will be concatenated onto the
/// concepts system prompt. The format is plain text with labeled sections
/// because the LLM parses it as free-form evidence, not structured data.
pub fn build_concepts_context(evidence: &GlossaryEvidence, project_name: &str) -> String {
    let mut ctx = String::new();

    ctx.push_str(&format!("Project: {}\n", project_name));
    ctx.push_str(&format!(
        "Scale: {} files, {} lines, {} modules, {} dependency edges\n",
        evidence.total_files,
        evidence.total_lines,
        evidence.modules.len(),
        evidence.dependency_edges,
    ));

    if !evidence.language_mix.is_empty() {
        let langs: Vec<String> = evidence
            .language_mix
            .iter()
            .map(|(lang, count)| format!("{} ({})", lang, count))
            .collect();
        ctx.push_str(&format!("Languages: {}\n", langs.join(", ")));
    }
    ctx.push('\n');

    ctx.push_str("Top-level modules (with anchor symbol names):\n");
    for m in &evidence.modules {
        if m.anchor_symbols.is_empty() {
            ctx.push_str(&format!("- {} ({} files)\n", m.path, m.file_count));
        } else {
            ctx.push_str(&format!(
                "- {} ({} files) — key symbols: {}\n",
                m.path,
                m.file_count,
                m.anchor_symbols.join(", ")
            ));
        }
    }
    ctx.push('\n');

    if !evidence.hotspot_files.is_empty() {
        ctx.push_str("Dependency hotspots (most-imported files):\n");
        for path in &evidence.hotspot_files {
            ctx.push_str(&format!("- {}\n", path));
        }
        ctx.push('\n');
    }

    ctx
}

/// Parse the LLM's JSON response into a [`ConceptsResponse`].
///
/// Accepts a raw response that may be wrapped in markdown code fences
/// (```json ... ```) — we strip them before feeding to `serde_json` because
/// models occasionally violate the "no code fences" instruction in the
/// system prompt.
pub fn parse_concepts_response(raw: &str) -> Result<ConceptsResponse> {
    let trimmed = raw.trim();

    // Strip ```json ... ``` or ``` ... ``` if the LLM wrapped its output.
    let cleaned: &str = if let Some(rest) = trimmed.strip_prefix("```json") {
        rest.trim_start().trim_end_matches("```").trim()
    } else if let Some(rest) = trimmed.strip_prefix("```") {
        rest.trim_start().trim_end_matches("```").trim()
    } else {
        trimmed
    };

    // If there's leading/trailing prose, try to extract the JSON object by
    // looking for the first '{' and the matching last '}'.
    let slice = if cleaned.starts_with('{') {
        cleaned
    } else if let (Some(start), Some(end)) = (cleaned.find('{'), cleaned.rfind('}')) {
        &cleaned[start..=end]
    } else {
        cleaned
    };

    serde_json::from_str::<ConceptsResponse>(slice)
        .context("Failed to parse concepts JSON response from LLM")
}

/// Render the full glossary page markdown. When concepts are present, this
/// emits the card-based layout; when empty, it emits the "no concepts"
/// fallback message (used in LLM-failure paths).
pub fn render_glossary_markdown(data: &GlossaryData) -> String {
    if data.concepts.is_empty() {
        return "*Concepts are generated by the LLM narration pipeline. \
                Re-run `rfx pulse generate` with LLM enabled to populate this page.*\n"
            .to_string();
    }

    let mut md = String::new();

    if let Some(ref intro) = data.intro {
        md.push_str(intro.trim());
        md.push_str("\n\n");
    }

    // Group concepts by category preserving first-seen order so the page
    // reflects whatever ordering the LLM chose.
    let mut order: Vec<String> = Vec::new();
    let mut grouped: HashMap<String, Vec<&Concept>> = HashMap::new();
    for concept in &data.concepts {
        let cat = concept
            .category
            .clone()
            .unwrap_or_else(|| "Concepts".to_string());
        if !grouped.contains_key(&cat) {
            order.push(cat.clone());
        }
        grouped.entry(cat).or_default().push(concept);
    }

    md.push_str(&format!(
        "**{}** core concepts across {} {}.\n\n",
        data.concepts.len(),
        order.len(),
        if order.len() == 1 { "category" } else { "categories" },
    ));

    for cat in &order {
        md.push_str(&format!("## {}\n\n", cat));
        if let Some(items) = grouped.get(cat) {
            for concept in items {
                md.push_str(&format!("### {}\n\n", concept.name));

                // Blockquoted definition.
                for line in concept.definition.trim().lines() {
                    md.push_str("> ");
                    md.push_str(line);
                    md.push('\n');
                }
                md.push('\n');

                if !concept.related_modules.is_empty() {
                    let links: Vec<String> = concept
                        .related_modules
                        .iter()
                        .map(|m| {
                            format!("[`{}`](/wiki/{}/)", m.trim(), module_slug(m.trim()))
                        })
                        .collect();
                    md.push_str(&format!("*Implemented in {}*\n\n", links.join(", ")));
                }
            }
        }
    }

    md
}

/// Render the `--no-llm` fallback page: a short explanation plus a bullet
/// list of modules from the evidence bundle so the page still shows useful
/// structure even without LLM narration.
pub fn render_glossary_no_llm(evidence: &GlossaryEvidence) -> String {
    let mut md = String::new();
    md.push_str(
        "*Concepts are generated by the LLM narration pipeline. \
         Re-run `rfx pulse generate` with LLM enabled to populate this page.*\n\n",
    );

    if evidence.modules.is_empty() {
        return md;
    }

    md.push_str("**Modules in this codebase:**\n\n");
    for m in &evidence.modules {
        md.push_str(&format!(
            "- [`{}`](/wiki/{}/) ({} files)\n",
            m.path,
            module_slug(&m.path),
            m.file_count
        ));
    }
    md.push('\n');
    md
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CacheManager;
    use tempfile::TempDir;

    fn empty_cache() -> (TempDir, CacheManager) {
        let tmp = TempDir::new().unwrap();
        let cache = CacheManager::new(tmp.path().to_str().unwrap());
        cache.init().unwrap();
        (tmp, cache)
    }

    #[test]
    fn test_module_of() {
        assert_eq!(module_of("src/models.rs"), "src");
        assert_eq!(module_of("src/pulse/wiki.rs"), "src/pulse");
        assert_eq!(module_of("src/parsers/rust/mod.rs"), "src/parsers");
        assert_eq!(module_of("README.md"), "");
    }

    #[test]
    fn test_module_slug() {
        assert_eq!(module_slug("src"), "src");
        assert_eq!(module_slug("src/pulse"), "src-pulse");
        assert_eq!(module_slug("src/parsers/rust"), "src-parsers-rust");
    }

    #[test]
    fn test_anchor_priority_orders_types_first() {
        assert!(anchor_priority("struct") < anchor_priority("function"));
        assert!(anchor_priority("trait") < anchor_priority("constant"));
        assert!(anchor_priority("enum") < anchor_priority("variable"));
    }

    #[test]
    fn test_collect_glossary_evidence_empty_cache() {
        let (_tmp, cache) = empty_cache();
        let result = collect_glossary_evidence(&cache).unwrap();
        // No symbols table → None.
        assert!(result.is_none());
    }

    #[test]
    fn test_build_concepts_context_includes_modules() {
        let evidence = GlossaryEvidence {
            total_files: 120,
            total_lines: 18_500,
            language_mix: vec![("rust".to_string(), 110), ("toml".to_string(), 10)],
            dependency_edges: 340,
            hotspot_files: vec!["src/models.rs".to_string()],
            modules: vec![
                ModuleEvidence {
                    path: "src".to_string(),
                    file_count: 42,
                    anchor_symbols: vec![
                        "Cli".to_string(),
                        "SearchResult".to_string(),
                        "run".to_string(),
                    ],
                },
                ModuleEvidence {
                    path: "src/pulse".to_string(),
                    file_count: 18,
                    anchor_symbols: vec![
                        "generate_site".to_string(),
                        "PulseReport".to_string(),
                    ],
                },
                ModuleEvidence {
                    path: "src/query".to_string(),
                    file_count: 9,
                    anchor_symbols: vec!["QueryEngine".to_string()],
                },
            ],
        };
        let ctx = build_concepts_context(&evidence, "Reflex");

        assert!(ctx.contains("Project: Reflex"));
        assert!(ctx.contains("120 files"));
        assert!(ctx.contains("src (42 files)"));
        assert!(ctx.contains("src/pulse"));
        assert!(ctx.contains("src/query"));
        assert!(ctx.contains("SearchResult"));
        assert!(ctx.contains("QueryEngine"));
        assert!(ctx.contains("Languages: rust (110)"));
        assert!(ctx.contains("Dependency hotspots"));
    }

    #[test]
    fn test_parse_concepts_response_valid_json() {
        let raw = r#"{
            "intro": "Reflex catalogs search primitives and indexing building blocks.",
            "concepts": [
                {
                    "name": "Trigram Index",
                    "category": "Core Capabilities",
                    "definition": "A fast inverted index built from three-character substrings.",
                    "related_modules": ["src/index", "src/query"]
                },
                {
                    "name": "Symbol Cache",
                    "category": "Data Model",
                    "definition": "A persistent store of parsed language symbols keyed by content hash.",
                    "related_modules": ["src/cache"]
                }
            ]
        }"#;

        let parsed = parse_concepts_response(raw).expect("should parse");
        assert_eq!(parsed.concepts.len(), 2);
        assert_eq!(parsed.concepts[0].name, "Trigram Index");
        assert_eq!(parsed.concepts[0].related_modules, vec!["src/index", "src/query"]);
        assert!(parsed.intro.as_ref().unwrap().contains("search primitives"));
    }

    #[test]
    fn test_parse_concepts_response_strips_markdown_fence() {
        let raw = "```json\n{\"intro\":\"x\",\"concepts\":[]}\n```";
        let parsed = parse_concepts_response(raw).expect("should parse");
        assert_eq!(parsed.concepts.len(), 0);
        assert_eq!(parsed.intro.as_deref(), Some("x"));
    }

    #[test]
    fn test_parse_concepts_response_extracts_embedded_json() {
        let raw = "Here is the output you requested:\n\
                   {\"intro\":\"y\",\"concepts\":[{\"name\":\"X\",\"definition\":\"d\"}]}\n\
                   Hope that helps!";
        let parsed = parse_concepts_response(raw).expect("should parse");
        assert_eq!(parsed.concepts.len(), 1);
        assert_eq!(parsed.concepts[0].name, "X");
    }

    #[test]
    fn test_parse_concepts_response_rejects_malformed() {
        let raw = "this is definitely not JSON at all";
        assert!(parse_concepts_response(raw).is_err());
    }

    #[test]
    fn test_render_with_concepts() {
        let data = GlossaryData {
            intro: Some(
                "Reflex catalogs the core pieces of a local code-search engine."
                    .to_string(),
            ),
            concepts: vec![
                Concept {
                    name: "Trigram Index".to_string(),
                    definition: "A fast inverted index built from three-character substrings."
                        .to_string(),
                    category: Some("Core Capabilities".to_string()),
                    related_modules: vec!["src/index".to_string(), "src/query".to_string()],
                },
                Concept {
                    name: "Symbol Cache".to_string(),
                    definition: "A persistent store of parsed language symbols.".to_string(),
                    category: Some("Data Model".to_string()),
                    related_modules: vec!["src/cache".to_string()],
                },
            ],
        };

        let md = render_glossary_markdown(&data);

        // Structural assertions
        assert!(md.contains("Reflex catalogs"));
        assert!(md.contains("## Core Capabilities"));
        assert!(md.contains("## Data Model"));
        assert!(md.contains("### Trigram Index"));
        assert!(md.contains("### Symbol Cache"));
        assert!(md.contains("> A fast inverted index"));
        assert!(md.contains("[`src/index`](/wiki/src-index/)"));
        assert!(md.contains("[`src/query`](/wiki/src-query/)"));
        assert!(md.contains("Implemented in"));

        // Must NOT contain v2 artifacts:
        assert!(!md.contains("```rust"), "no signature code blocks");
        assert!(!md.contains(":1"), "no file:line markers (cheap check)");
        assert!(!md.contains("| Symbol | Kind"), "no flat table");
    }

    #[test]
    fn test_render_no_llm_fallback() {
        let data = GlossaryData::default();
        let md = render_glossary_markdown(&data);
        assert!(md.contains("LLM narration pipeline"));
        assert!(md.contains("rfx pulse generate"));
    }

    #[test]
    fn test_render_no_llm_fallback_with_evidence_lists_modules() {
        let evidence = GlossaryEvidence {
            total_files: 10,
            total_lines: 500,
            language_mix: vec![],
            dependency_edges: 0,
            hotspot_files: vec![],
            modules: vec![
                ModuleEvidence {
                    path: "src".to_string(),
                    file_count: 5,
                    anchor_symbols: vec![],
                },
                ModuleEvidence {
                    path: "src/pulse".to_string(),
                    file_count: 3,
                    anchor_symbols: vec![],
                },
            ],
        };
        let md = render_glossary_no_llm(&evidence);
        assert!(md.contains("LLM narration pipeline"));
        assert!(md.contains("[`src`](/wiki/src/)"));
        assert!(md.contains("[`src/pulse`](/wiki/src-pulse/)"));
        assert!(md.contains("(5 files)"));
    }

    #[test]
    fn test_concepts_response_into_glossary_data() {
        let resp = ConceptsResponse {
            intro: Some("hi".to_string()),
            concepts: vec![RawConcept {
                name: "Concept".to_string(),
                definition: "def".to_string(),
                category: Some("Cat".to_string()),
                related_modules: vec!["src".to_string()],
            }],
        };
        let data: GlossaryData = resp.into();
        assert_eq!(data.concepts.len(), 1);
        assert_eq!(data.concepts[0].name, "Concept");
        assert_eq!(data.intro.as_deref(), Some("hi"));
    }
}
