use crate::cache::CacheManager;
use crate::pulse;
use anyhow::Result;
use std::path::PathBuf;

/// "<Directory> Documentation", from the working directory name.
fn default_title() -> String {
    let name = std::env::current_dir()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "Pulse".to_string());
    let mut chars = name.chars();
    let capitalized = match chars.next() {
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
        None => name,
    };
    format!("{} Documentation", capitalized)
}

/// Writing-pass options for the standalone commands: LLM on, `[pulse.write]` defaults.
fn llm_on() -> pulse::write::WriteOptions {
    pulse::write::WriteOptions::default()
}

pub(super) fn handle_pulse_changelog(
    count: usize,
    no_llm: bool,
    json: bool,
    pretty: bool,
) -> Result<()> {
    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }

    let workspace_root = cache.path().parent().unwrap_or(std::path::Path::new("."));
    let mut changelog = pulse::changelog::extract_changelog(workspace_root, count)?;

    if !no_llm && !changelog.raw_commits.is_empty() {
        let session = pulse::write::WriteSession::open(cache.path(), &llm_on());
        let ctx =
            pulse::changelog::build_changelog_context(&changelog.raw_commits, &changelog.branch);
        if let Some(entries) = session
            .run_one(pulse::narrate::changelog_task(&ctx))
            .as_deref()
            .and_then(pulse::changelog::parse_changelog_response)
        {
            changelog.entries = entries;
            changelog.narrated = true;
        }
    }

    if json || pretty {
        let output = if pretty {
            serde_json::to_string_pretty(&changelog)?
        } else {
            serde_json::to_string(&changelog)?
        };
        println!("{}", output);
    } else {
        println!("{}", pulse::changelog::render_markdown(&changelog));
    }

    Ok(())
}

pub(super) fn handle_pulse_map(
    format: String,
    output: Option<PathBuf>,
    zoom: Option<String>,
) -> Result<()> {
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

pub(super) struct GenerateArgs {
    pub output: PathBuf,
    pub base_url: String,
    pub title: Option<String>,
    pub include: Option<String>,
    pub clean: bool,
    pub write: pulse::write::WriteOptions,
    pub depth: u8,
    pub min_files: usize,
    pub no_build: bool,
    pub offline: bool,
    pub verbose_build: bool,
}

pub(super) fn handle_pulse_generate(args: GenerateArgs) -> Result<()> {
    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }
    if args.include.is_some() {
        eprintln!("Note: --include is ignored; the site always has its Docs and Internals tabs.");
    }
    let config = pulse::site::SiteConfig {
        output_dir: args.output,
        base_url: args.base_url,
        title: args.title.unwrap_or_else(default_title),
        write: args.write,
        clean: args.clean,
        max_depth: args.depth,
        min_files: args.min_files,
        no_build: args.no_build,
        offline: args.offline,
        verbose_build: args.verbose_build,
    };
    let dry_run = config.write.dry_run;
    let report = pulse::site::generate_site(&cache, &config)?;
    if dry_run {
        eprintln!("Dry run: no LLM calls were made and nothing was written.");
        return Ok(());
    }
    eprintln!(
        "Pulse: {} pages ({} Docs, {} Internals), {} narrated section(s) [{}], {} broken link(s)",
        report.pages,
        report.docs_pages,
        report.internals_pages,
        report.narrated_sections,
        report.narration_mode,
        report.broken_links
    );
    match report.build.as_str() {
        "built" => eprintln!(
            "Site built in {:.1}s ({:.1}s total): {}/  — preview with `rfx pulse serve -o {}`",
            report.build_seconds, report.total_seconds, report.output_dir, report.output_dir
        ),
        _ => eprintln!(
            "Site project written to {} (not built). Build it with Node {}.{}+: \
             `npx astro build --root {}` after `npm ci` in pulse-template's dependencies, \
             or rerun without --no-build.",
            report.project_dir.as_deref().unwrap_or("?"),
            pulse::runtime::MIN_NODE.0,
            pulse::runtime::MIN_NODE.1,
            report.project_dir.as_deref().unwrap_or("?")
        ),
    }
    Ok(())
}

pub(super) fn handle_pulse_glossary(no_llm: bool, json: bool) -> Result<()> {
    use crate::pulse::glossary;

    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }

    let evidence = glossary::collect_glossary_evidence(&cache)?;

    // Generate concepts via the LLM when enabled; fall back to structural-only output.
    let data: glossary::GlossaryData = match evidence.as_ref() {
        Some(ev) if !no_llm => {
            let project_name = std::env::current_dir()
                .ok()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                .unwrap_or_else(|| "project".to_string());
            let context = glossary::build_concepts_context(ev, &project_name);
            let session = pulse::write::WriteSession::open(cache.path(), &llm_on());
            session
                .run_one(pulse::narrate::concepts_task(&context))
                .and_then(|raw| glossary::parse_concepts_response(&raw).ok())
                .map(glossary::GlossaryData::from)
                .unwrap_or_default()
        }
        _ => glossary::GlossaryData::default(),
    };

    if json {
        let module_summaries = evidence
            .as_ref()
            .map(|ev| {
                ev.modules
                    .iter()
                    .map(|m| {
                        serde_json::json!({
                            "path": m.path,
                            "file_count": m.file_count,
                            "anchor_symbols": m.anchor_symbols,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "total_concepts": data.concepts.len(),
                "concepts": data.concepts.iter().map(|c| serde_json::json!({
                    "name": c.name,
                    "category": c.category,
                    "definition": c.definition,
                })).collect::<Vec<_>>(),
                "evidence_modules": module_summaries,
            }))?
        );
    } else {
        let md = if data.concepts.is_empty() {
            // No LLM or LLM failed — show structural evidence with hint
            if let Some(ev) = evidence.as_ref() {
                glossary::render_glossary_no_llm(ev)
            } else {
                glossary::render_glossary_markdown(&data)
            }
        } else {
            glossary::render_glossary_markdown(&data)
        };
        println!("{}", md);
    }

    Ok(())
}

pub(super) fn handle_pulse_model(
    json: bool,
    title: Option<String>,
    depth: u8,
    min_files: usize,
) -> Result<()> {
    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }
    let docs = pulse::config::load_pulse_config(cache.path())?.docs;
    let opts = pulse::build::BuildOptions {
        max_depth: depth,
        min_files,
        slugs_path: Some(cache.path().join("pulse").join("slugs.json")),
        reference: pulse::build::reference::ReferenceOptions {
            skip_library: !docs.library,
            include: docs.include,
        },
        ..pulse::build::BuildOptions::new(title.unwrap_or_else(default_title))
    };
    let site = pulse::build::build_site(&cache, &opts)?;
    if json {
        println!("{}", site.to_json()?);
        return Ok(());
    }
    println!("{} ({})", site.meta.title, site.meta.generator);
    for tab in &site.tabs {
        let pages = site.nav_order(tab.id);
        println!("  {}: {} pages", tab.label, pages.len());
        for id in pages {
            if let Some(p) = site.pages.get(&id) {
                println!("    {:<40} {}", p.route, p.title);
            }
        }
    }
    println!("  facts: {}", site.facts.len());
    println!("  narrative slots: {}", site.narrative_slots().len());
    let roles: Vec<String> = site
        .report
        .files_by_role
        .iter()
        .map(|(r, n)| format!("{r} {n}"))
        .collect();
    println!("  indexed files by role: {}", roles.join(", "));
    if site.report.broken_links.is_empty() {
        println!("  broken links: none");
    } else {
        println!("  broken links: {}", site.report.broken_links.len());
        for (page, target) in &site.report.broken_links {
            println!("    {page} -> {target:?}");
        }
    }
    Ok(())
}

pub fn handle_pulse_runtime(command: super::PulseRuntimeCommand) -> Result<()> {
    use super::PulseRuntimeCommand as C;
    use crate::pulse::runtime;
    match command {
        C::Key => println!("{}", runtime::key()),
        C::Status => print!("{}", runtime::status()),
        C::Install => {
            let rt = runtime::ensure(false)?;
            println!("{}", rt.root.display());
        }
    }
    Ok(())
}
