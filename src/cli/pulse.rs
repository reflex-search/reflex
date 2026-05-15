use anyhow::{Context, Result};
use std::path::PathBuf;
use crate::cache::CacheManager;
use crate::pulse;


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
        match pulse::narrate::create_pulse_provider() {
            Ok(provider) => {
                eprintln!("LLM provider ready.");
                let llm_cache = pulse::llm_cache::LlmCache::new(cache.path());

                let pulse_config = pulse::config::load_pulse_config(cache.path())?;
                let ensure_result = pulse::snapshot::ensure_snapshot(&cache, &pulse_config.retention)?;
                let snapshot_id = match &ensure_result {
                    pulse::snapshot::EnsureSnapshotResult::Created(info) => info.id.clone(),
                    pulse::snapshot::EnsureSnapshotResult::Reused(info) => info.id.clone(),
                };

                let ctx = pulse::changelog::build_changelog_context(&changelog.raw_commits, &changelog.branch);
                let response = pulse::narrate::narrate_section(
                    provider.as_ref(),
                    pulse::narrate::changelog_system_prompt(),
                    &ctx,
                    &llm_cache,
                    &snapshot_id,
                    "changelog",
                );

                if let Some(text) = response {
                    changelog.entries = pulse::changelog::parse_changelog_response(&text, &changelog.raw_commits);
                    changelog.narrated = true;
                }
            }
            Err(e) => {
                eprintln!("LLM unavailable: {}", e);
            }
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


pub(super) fn handle_pulse_wiki(no_llm: bool, output: Option<PathBuf>, json: bool) -> Result<()> {
    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }

    let pulse_config = pulse::config::load_pulse_config(cache.path())?;

    // Auto-snapshot if index has changed since last snapshot
    let ensure_result = pulse::snapshot::ensure_snapshot(&cache, &pulse_config.retention)?;
    match &ensure_result {
        pulse::snapshot::EnsureSnapshotResult::Created(info) => {
            eprintln!("Auto-snapshot created: {} ({} files)", info.id, info.file_count);
        }
        pulse::snapshot::EnsureSnapshotResult::Reused(info) => {
            eprintln!("Using snapshot: {} (index unchanged)", info.id);
        }
    }

    let snapshots = pulse::snapshot::list_snapshots(&cache)?;

    let snapshot_diff = if snapshots.len() >= 2 {
        pulse::diff::compute_diff(&snapshots[1].path, &snapshots[0].path, &pulse_config.thresholds).ok()
    } else {
        None
    };

    // Create provider for standalone wiki command
    let (provider, llm_cache) = if !no_llm {
        match pulse::narrate::create_pulse_provider() {
            Ok(p) => {
                eprintln!("LLM provider ready.");
                let c = pulse::llm_cache::LlmCache::new(cache.path());
                (Some(p), Some(c))
            }
            Err(e) => {
                eprintln!("LLM unavailable: {}", e);
                (None, None)
            }
        }
    } else {
        (None, None)
    };

    let snapshot_id = snapshots.first().map(|s| s.id.as_str()).unwrap_or("unknown");
    let pages = pulse::wiki::generate_all_pages(
        &cache,
        snapshot_diff.as_ref(),
        no_llm,
        snapshot_id,
        provider.as_ref().map(|p| p.as_ref()),
        llm_cache.as_ref(),
        &pulse::wiki::ModuleDiscoveryConfig::default(),
    )?;

    if json {
        println!("{}", serde_json::to_string_pretty(&pages)?);
    } else if let Some(out_dir) = output {
        std::fs::create_dir_all(&out_dir)?;
        let rendered = pulse::wiki::render_wiki_markdown(&pages);
        for (filename, content) in &rendered {
            std::fs::write(out_dir.join(filename), content)?;
        }
        eprintln!("Wrote {} wiki pages to {}", rendered.len(), out_dir.display());
    } else {
        let rendered = pulse::wiki::render_wiki_markdown(&pages);
        for (filename, content) in &rendered {
            println!("--- {} ---\n{}\n", filename, content);
        }
    }

    Ok(())
}


pub(super) fn handle_pulse_map(format: String, output: Option<PathBuf>, zoom: Option<String>) -> Result<()> {
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


pub(super) fn handle_pulse_generate(
    output: PathBuf,
    base_url: String,
    title: Option<String>,
    include: Option<String>,
    no_llm: bool,
    clean: bool,
    force_renarrate: bool,
    concurrency: usize,
    depth: u8,
    min_files: usize,
) -> Result<()> {
    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }

    let surfaces = match include {
        Some(ref s) => {
            s.split(',')
                .map(|part| match part.trim().to_lowercase().as_str() {
                    "wiki" => Ok(pulse::site::Surface::Wiki),
                    "changelog" | "digest" => Ok(pulse::site::Surface::Changelog),
                    "map" => Ok(pulse::site::Surface::Map),
                    "onboard" => Ok(pulse::site::Surface::Onboard),
                    "timeline" => Ok(pulse::site::Surface::Timeline),
                    "glossary" => Ok(pulse::site::Surface::Glossary),
                    "explorer" => Ok(pulse::site::Surface::Explorer),
                    other => anyhow::bail!("Unknown surface '{}'. Supported: wiki, changelog, map, onboard, timeline, glossary, explorer", other),
                })
                .collect::<Result<Vec<_>>>()?
        }
        None => vec![
            pulse::site::Surface::Wiki,
            pulse::site::Surface::Changelog,
            pulse::site::Surface::Map,
            pulse::site::Surface::Onboard,
            pulse::site::Surface::Timeline,
            pulse::site::Surface::Glossary,
            pulse::site::Surface::Explorer,
        ],
    };

    let config = pulse::site::SiteConfig {
        output_dir: output,
        base_url,
        title: title.unwrap_or_else(|| {
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
        }),
        surfaces,
        no_llm,
        clean,
        force_renarrate,
        concurrency,
        max_depth: depth,
        min_files,
    };

    let report = pulse::site::generate_site(&cache, &config)?;

    eprintln!("Zola project generated in {}/", report.output_dir);
    eprintln!("  Wiki pages: {}", report.pages_generated);
    eprintln!("  Changelog: {}", if report.changelog_generated { "yes" } else { "no" });
    eprintln!("  Map: {}", if report.map_generated { "yes" } else { "no" });
    eprintln!("  Onboard: {}", if report.onboard_generated { "yes" } else { "no" });
    eprintln!("  Timeline: {}", if report.timeline_generated { "yes" } else { "no" });
    eprintln!("  Glossary: {}", if report.glossary_generated { "yes" } else { "no" });
    eprintln!("  Explorer: {}", if report.explorer_generated { "yes" } else { "no" });
    eprintln!("  Narration: {}", report.narration_mode);
    if report.build_success {
        eprintln!("  Build: success (HTML in {}/public/)", report.output_dir);
    } else {
        eprintln!("  Build: skipped (run `cd {} && zola build` manually)", report.output_dir);
    }

    Ok(())
}


pub(super) fn handle_pulse_serve(output: PathBuf, port: u16, open: bool) -> Result<()> {
    // Verify the output dir has a config.toml (i.e., was generated)
    if !output.join("config.toml").exists() {
        anyhow::bail!(
            "No Zola project found at '{}'. Run `rfx pulse generate` first.",
            output.display()
        );
    }

    let zola_path = pulse::zola::ensure_zola()?;

    let url = format!("http://127.0.0.1:{}", port);
    eprintln!("Serving Pulse site at {}", url);
    eprintln!("Press Ctrl+C to stop.\n");

    if open {
        open_browser(&url);
    }

    let status = std::process::Command::new(&zola_path)
        .current_dir(&output)
        .arg("serve")
        .arg("--port")
        .arg(port.to_string())
        .arg("--interface")
        .arg("127.0.0.1")
        .status()
        .context("Failed to start Zola server")?;

    if !status.success() {
        anyhow::bail!("Zola server exited with error");
    }

    Ok(())
}


fn open_browser(url: &str) {
    let result = if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(url).spawn()
    } else if cfg!(target_os = "windows") {
        std::process::Command::new("cmd")
            .args(["/c", "start", url])
            .spawn()
    } else {
        std::process::Command::new("xdg-open").arg(url).spawn()
    };

    if let Err(e) = result {
        eprintln!("Could not open browser: {e}");
    }
}


pub(super) fn handle_pulse_onboard(no_llm: bool, json: bool) -> Result<()> {
    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }

    let modules = crate::pulse::wiki::detect_modules(&cache, &crate::pulse::wiki::ModuleDiscoveryConfig::default())?;
    let mut data = crate::pulse::onboard::generate_onboard_structural(&cache, modules.len())?;

    if !no_llm {
        if let Ok(provider) = crate::pulse::narrate::create_pulse_provider() {
            let llm_cache = crate::pulse::llm_cache::LlmCache::new(cache.path());
            let ctx = crate::pulse::onboard::build_onboard_context(&data);
            let narration = crate::pulse::narrate::narrate_section(
                &*provider,
                crate::pulse::narrate::onboard_system_prompt(),
                &ctx,
                &llm_cache,
                "standalone",
                "onboard-guide",
            );
            data.narration = narration;
        }
    }

    if json {
        let ctx = crate::pulse::onboard::build_onboard_context(&data);
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({
            "entry_points": data.entry_points.iter().map(|ep| serde_json::json!({
                "path": ep.path,
                "kind": format!("{}", ep.kind),
                "key_symbols": ep.key_symbols,
            })).collect::<Vec<_>>(),
            "reading_order_layers": data.reading_order.layers.len(),
            "context": ctx,
        }))?);
    } else {
        let md = crate::pulse::onboard::render_onboard_markdown(&data);
        println!("{}", md);
    }

    Ok(())
}


pub(super) fn handle_pulse_timeline(json: bool) -> Result<()> {
    let data = crate::pulse::git_intel::extract_git_intel(".")?;

    if json {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({
            "commits": data.commits.len(),
            "contributors": data.contributors.iter().map(|c| serde_json::json!({
                "name": c.name,
                "email": c.email,
                "commit_count": c.commit_count,
            })).collect::<Vec<_>>(),
            "churn_files": data.churn.len(),
            "weekly_summaries": data.weekly_summaries.len(),
        }))?);
    } else {
        let md = crate::pulse::git_intel::render_timeline_markdown(&data);
        println!("{}", md);
    }

    Ok(())
}


pub(super) fn handle_pulse_glossary(json: bool) -> Result<()> {
    use crate::pulse::glossary;

    let cache = CacheManager::new(".");
    if !cache.path().exists() {
        anyhow::bail!("No .reflex cache found. Run `rfx index` first.");
    }

    let evidence = glossary::collect_glossary_evidence(&cache)?;

    // Try to generate concepts via LLM if configured; fall back to structural-only output.
    let data: glossary::GlossaryData = if let Some(ev) = evidence.as_ref() {
        match pulse::narrate::create_pulse_provider() {
            Ok(provider) => {
                let llm_cache = pulse::llm_cache::LlmCache::new(cache.path());
                let project_name = std::env::current_dir()
                    .ok()
                    .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                    .unwrap_or_else(|| "project".to_string());
                let context = glossary::build_concepts_context(ev, &project_name);
                let raw = pulse::narrate::narrate_section(
                    provider.as_ref(),
                    pulse::narrate::concepts_system_prompt(),
                    &context,
                    &llm_cache,
                    "cli-glossary",
                    "glossary",
                );
                if let Some(raw_text) = raw {
                    glossary::parse_concepts_response(&raw_text)
                        .map(glossary::GlossaryData::from)
                        .unwrap_or_default()
                } else {
                    glossary::GlossaryData::default()
                }
            }
            Err(_) => glossary::GlossaryData::default(),
        }
    } else {
        glossary::GlossaryData::default()
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
