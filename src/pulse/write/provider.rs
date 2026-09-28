//! Resolve which LLM the writing pass talks to.

use super::LlmMode;
use crate::pulse::config::WriteSettings;
use crate::semantic::config;
use crate::semantic::providers::{self, LlmProvider};
use std::path::Path;
use std::sync::Arc;

/// Provider and model: the identity that goes into every cache key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIdent {
    pub provider: String,
    pub model: String,
}

/// What the writing pass may use this run.
#[derive(Clone, Default)]
pub struct LlmHandle {
    /// Identity for cache keys. Set whenever a provider is configured, even without a key.
    pub ident: Option<ModelIdent>,
    /// A live provider, when `mode` is `On` and one could be built.
    pub provider: Option<Arc<dyn LlmProvider>>,
    /// Why no live provider exists, for the run summary.
    pub unavailable: Option<String>,
}

impl LlmHandle {
    /// A handle around an existing provider (tests, embedding).
    pub fn from_provider(provider: Arc<dyn LlmProvider>) -> Self {
        Self {
            ident: Some(ModelIdent {
                provider: provider.name().to_string(),
                model: provider.model().to_string(),
            }),
            provider: Some(provider),
            unavailable: None,
        }
    }
}

/// Build the handle for `mode`.
///
/// - `Off`: empty.
/// - `CacheOnly`: identity from config only. No API key is needed, so fork PRs in CI
///   can reuse a cache that a trusted run filled.
/// - `On`: a live provider. If the configured provider has no key, the first provider
///   that has one is used instead (CI often sets only `OPENROUTER_API_KEY`).
pub fn resolve(mode: LlmMode, settings: &WriteSettings) -> LlmHandle {
    match mode {
        LlmMode::Off => LlmHandle::default(),
        LlmMode::CacheOnly => match configured_ident(settings) {
            Ok(ident) => LlmHandle {
                ident: Some(ident),
                provider: None,
                unavailable: Some("cache-only mode".into()),
            },
            Err(e) => LlmHandle {
                unavailable: Some(e.to_string()),
                ..Default::default()
            },
        },
        LlmMode::On => match create_provider(settings) {
            Ok(p) => LlmHandle::from_provider(Arc::from(p)),
            Err(e) => LlmHandle {
                ident: configured_ident(settings).ok(),
                provider: None,
                unavailable: Some(e.to_string()),
            },
        },
    }
}

fn configured_provider(settings: &WriteSettings) -> anyhow::Result<String> {
    if let Some(p) = &settings.provider {
        return Ok(p.clone());
    }
    Ok(config::load_config(Path::new("."))?.provider)
}

/// Model for `provider`. The configured model names apply only to the configured
/// provider; an auto-detected fallback uses its own `[credentials]` model or default.
fn resolve_model(provider: &str, settings: &WriteSettings) -> anyhow::Result<Option<String>> {
    let semantic = config::load_config(Path::new("."))?;
    let configured = settings.provider.as_deref().unwrap_or(&semantic.provider);
    if !provider.eq_ignore_ascii_case(configured) {
        return Ok(config::resolve_model_for(provider, None, None));
    }
    let project_model = configured
        .eq_ignore_ascii_case(&semantic.provider)
        .then_some(semantic.model.as_deref())
        .flatten();
    Ok(config::resolve_model_for(
        provider,
        project_model,
        settings.model.as_deref(),
    ))
}

/// The identity a live provider would report, without building it.
fn configured_ident(settings: &WriteSettings) -> anyhow::Result<ModelIdent> {
    let provider = configured_provider(settings)?;
    let model = resolve_model(&provider, settings)?
        .unwrap_or_else(|| providers::default_model_for(&provider).to_string());
    if model.is_empty() {
        anyhow::bail!("no model configured for provider '{provider}'");
    }
    Ok(ModelIdent {
        provider: canonical_name(&provider),
        model,
    })
}

/// Provider names as `LlmProvider::name` reports them.
fn canonical_name(provider: &str) -> String {
    match provider.to_ascii_lowercase().as_str() {
        "openai_compatible" => "openai-compatible".into(),
        other => other.into(),
    }
}

fn create_provider(settings: &WriteSettings) -> anyhow::Result<Box<dyn LlmProvider>> {
    let semantic = config::load_config(Path::new("."))?;
    let configured = configured_provider(settings)?;

    let (provider, api_key) = match config::get_api_key(&configured) {
        Ok(key) => (configured.clone(), key),
        Err(configured_err) => {
            let mut fallbacks: Vec<&str> = vec!["openrouter", "anthropic", "openai"];
            // Keyless local endpoints count only when a base URL is configured.
            if config::get_provider_options("openai-compatible").is_some() {
                fallbacks.push("openai-compatible");
            }
            let mut found = None;
            for candidate in fallbacks {
                if candidate.eq_ignore_ascii_case(&configured) {
                    continue;
                }
                if let Ok(key) = config::get_api_key(candidate) {
                    eprintln!(
                        "Note: no API key for configured provider '{configured}', using auto-detected '{candidate}'"
                    );
                    found = Some((candidate.to_string(), key));
                    break;
                }
            }
            found.ok_or(configured_err)?
        }
    };

    let model = resolve_model(&provider, settings)?;
    let options = config::get_provider_options(&provider);
    providers::create_provider(&provider, api_key, model, options, semantic.timeout_seconds)
}
