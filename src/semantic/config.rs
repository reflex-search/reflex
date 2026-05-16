//! Configuration for semantic query feature

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};

/// Locate the user's home directory.
///
/// `dirs::home_dir()` queries `SHGetKnownFolderPath(FOLDERID_Profile)` on
/// Windows and therefore ignores `HOME` / `USERPROFILE` env vars. That makes
/// it impossible to redirect to a temp directory in tests. Honour those env
/// vars (and `REFLEX_HOME` for an explicit override) before falling back to
/// the OS-native lookup so test code can point us at a temp directory on
/// every platform.
fn user_home_dir() -> Option<PathBuf> {
    for var in ["REFLEX_HOME", "HOME", "USERPROFILE"] {
        if let Some(val) = env::var_os(var)
            && !val.is_empty()
        {
            return Some(PathBuf::from(val));
        }
    }
    dirs::home_dir()
}

/// Semantic query configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConfig {
    /// Enable semantic query feature
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// LLM provider (openai, anthropic, openrouter)
    #[serde(default = "default_provider")]
    pub provider: String,

    /// Optional model override (uses provider default if None)
    #[serde(default)]
    pub model: Option<String>,

    /// Auto-execute generated commands without confirmation
    #[serde(default)]
    pub auto_execute: bool,

    /// Enable agentic mode (multi-step reasoning with context gathering)
    #[serde(default = "default_agentic_enabled")]
    pub agentic_enabled: bool,

    /// Maximum iterations for query refinement in agentic mode
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,

    /// Maximum tool calls per context gathering phase
    #[serde(default = "default_max_tools")]
    pub max_tools_per_phase: usize,

    /// Enable result evaluation in agentic mode
    #[serde(default = "default_evaluation_enabled")]
    pub evaluation_enabled: bool,

    /// Evaluation strictness (0.0-1.0, higher is stricter)
    #[serde(default = "default_strictness")]
    pub evaluation_strictness: f32,

    /// LLM request timeout in seconds (default: 30)
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,
}

fn default_enabled() -> bool {
    true
}

fn default_provider() -> String {
    "openai".to_string()
}

fn default_agentic_enabled() -> bool {
    false // Disabled by default, opt-in for experimental feature
}

fn default_max_iterations() -> usize {
    2
}

fn default_max_tools() -> usize {
    5
}

fn default_evaluation_enabled() -> bool {
    true
}

fn default_strictness() -> f32 {
    0.5
}

fn default_timeout_seconds() -> u64 {
    30
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: "openai".to_string(),
            model: None,
            auto_execute: false,
            agentic_enabled: false,
            max_iterations: 2,
            max_tools_per_phase: 5,
            evaluation_enabled: true,
            evaluation_strictness: 0.5,
            timeout_seconds: 30,
        }
    }
}

/// Apply environment variable overrides to a semantic config.
///
/// Supports:
/// - `REFLEX_PROVIDER` — overrides the provider (e.g., "openrouter", "anthropic", "openai")
/// - `REFLEX_MODEL` — overrides the model
///
/// This enables CI/headless usage where there's no ~/.reflex/config.toml.
fn apply_env_overrides(mut config: SemanticConfig) -> SemanticConfig {
    if let Ok(provider) = env::var("REFLEX_PROVIDER")
        && !provider.is_empty()
    {
        log::debug!(
            "Overriding provider from REFLEX_PROVIDER env var: {}",
            provider
        );
        config.provider = provider;
    }

    if let Ok(model) = env::var("REFLEX_MODEL")
        && !model.is_empty()
    {
        log::debug!("Overriding model from REFLEX_MODEL env var: {}", model);
        config.model = Some(model);
    }

    if let Ok(val) = env::var("REFLEX_LLM_TIMEOUT_SECONDS") {
        match val.trim().parse::<u64>() {
            Ok(secs) if secs > 0 => {
                log::debug!(
                    "Overriding LLM timeout from REFLEX_LLM_TIMEOUT_SECONDS: {}s",
                    secs
                );
                config.timeout_seconds = secs;
            }
            _ => log::warn!(
                "REFLEX_LLM_TIMEOUT_SECONDS is invalid (must be a positive integer): {}",
                val
            ),
        }
    }

    config
}

/// Load semantic config from ~/.reflex/config.toml
///
/// Semantic configuration is ALWAYS user-level (not project-level).
/// Falls back to defaults if file doesn't exist or [semantic] section is missing.
/// Environment variables `REFLEX_PROVIDER` and `REFLEX_MODEL` override config file values.
///
/// Note: The cache_dir parameter is ignored - kept for API compatibility but will be removed in future.
pub fn load_config(_cache_dir: &Path) -> Result<SemanticConfig> {
    // Semantic config is always in user home directory, not project directory
    let home = match user_home_dir() {
        Some(h) => h,
        None => {
            log::debug!("Could not determine home directory, using defaults");
            return Ok(apply_env_overrides(SemanticConfig::default()));
        }
    };

    let config_path = home.join(".reflex").join("config.toml");

    if !config_path.exists() {
        log::debug!("No ~/.reflex/config.toml found, using default semantic config");
        return Ok(apply_env_overrides(SemanticConfig::default()));
    }

    let config_str =
        std::fs::read_to_string(&config_path).context("Failed to read ~/.reflex/config.toml")?;

    let toml_value: toml::Value =
        toml::from_str(&config_str).context("Failed to parse ~/.reflex/config.toml")?;

    // REF-90: Warn about unknown top-level sections
    let known_sections = ["semantic", "credentials", "index", "search", "performance"];
    if let Some(table) = toml_value.as_table() {
        for key in table.keys() {
            if !known_sections.contains(&key.as_str()) {
                eprintln!(
                    "[warn] ~/.reflex/config.toml: unknown section '[{}]' — ignored",
                    key
                );
            }
        }
    }

    // REF-90: Warn about unknown keys within the [semantic] section
    let known_semantic_keys = ["provider", "model", "auto_execute"];
    if let Some(toml::Value::Table(sem_table)) = toml_value.get("semantic") {
        for key in sem_table.keys() {
            if !known_semantic_keys.contains(&key.as_str()) {
                eprintln!(
                    "[warn] ~/.reflex/config.toml: unknown key '[semantic].{}' — ignored",
                    key
                );
            }
        }
    }

    // Extract [semantic] section
    if let Some(semantic_table) = toml_value.get("semantic") {
        let config: SemanticConfig = semantic_table
            .clone()
            .try_into()
            .context("Failed to parse [semantic] section in ~/.reflex/config.toml")?;
        log::debug!(
            "Loaded semantic config from ~/.reflex/config.toml: provider={}",
            config.provider
        );
        Ok(apply_env_overrides(config))
    } else {
        log::debug!("No [semantic] section in ~/.reflex/config.toml, using defaults");
        Ok(apply_env_overrides(SemanticConfig::default()))
    }
}

/// User configuration structure for ~/.reflex/config.toml
#[derive(Debug, Clone, Serialize, Deserialize)]
struct UserConfig {
    #[serde(default)]
    credentials: Option<Credentials>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Credentials {
    #[serde(default)]
    openai_api_key: Option<String>,
    #[serde(default)]
    anthropic_api_key: Option<String>,
    #[serde(default)]
    openrouter_api_key: Option<String>,
    #[serde(default)]
    openai_compatible_api_key: Option<String>,
    #[serde(default)]
    openai_model: Option<String>,
    #[serde(default)]
    anthropic_model: Option<String>,
    #[serde(default)]
    openrouter_model: Option<String>,
    #[serde(default)]
    openai_compatible_model: Option<String>,
    #[serde(default)]
    openrouter_sort: Option<String>,
    #[serde(default)]
    openai_compatible_base_url: Option<String>,
}

/// Load user configuration from ~/.reflex/config.toml
fn load_user_config() -> Result<Option<UserConfig>> {
    let home = match user_home_dir() {
        Some(h) => h,
        None => {
            log::debug!("Could not determine home directory");
            return Ok(None);
        }
    };

    let config_path = home.join(".reflex").join("config.toml");

    if !config_path.exists() {
        log::debug!("No user config found at ~/.reflex/config.toml");
        return Ok(None);
    }

    let config_str =
        std::fs::read_to_string(&config_path).context("Failed to read ~/.reflex/config.toml")?;

    let config: UserConfig =
        toml::from_str(&config_str).context("Failed to parse ~/.reflex/config.toml")?;

    Ok(Some(config))
}

/// Get API key for a provider
///
/// Checks in priority order:
/// 1. ~/.reflex/config.toml (user config file)
/// 2. REFLEX_AI_API_KEY environment variable (generic, provider-agnostic)
/// 3. {PROVIDER}_API_KEY environment variable (e.g., OPENAI_API_KEY)
/// 4. Error if not found
pub fn get_api_key(provider: &str) -> Result<String> {
    let provider_lc = provider.to_lowercase();
    let is_openai_compatible =
        provider_lc == "openai-compatible" || provider_lc == "openai_compatible";

    // First check user config file
    if let Ok(Some(user_config)) = load_user_config()
        && let Some(credentials) = &user_config.credentials
    {
        // Get the appropriate key based on provider
        let key = match provider_lc.as_str() {
            "openai" => credentials.openai_api_key.as_ref(),
            "anthropic" => credentials.anthropic_api_key.as_ref(),
            "openrouter" => credentials.openrouter_api_key.as_ref(),
            "openai-compatible" | "openai_compatible" => {
                credentials.openai_compatible_api_key.as_ref()
            }
            _ => None,
        };

        if let Some(api_key) = key {
            log::debug!("Using {} API key from ~/.reflex/config.toml", provider);
            return Ok(api_key.clone());
        }
    }

    // Check generic REFLEX_AI_API_KEY env var (provider-agnostic, useful for CI)
    if let Ok(key) = env::var("REFLEX_AI_API_KEY")
        && !key.is_empty()
    {
        log::debug!(
            "Using API key from REFLEX_AI_API_KEY env var for provider '{}'",
            provider
        );
        return Ok(key);
    }

    // Fall back to provider-specific environment variables
    let env_var = match provider_lc.as_str() {
        "openai" => "OPENAI_API_KEY",
        "anthropic" => "ANTHROPIC_API_KEY",
        "openrouter" => "OPENROUTER_API_KEY",
        "openai-compatible" | "openai_compatible" => "OPENAI_COMPATIBLE_API_KEY",
        _ => anyhow::bail!("Unknown provider: {}", provider),
    };

    if let Ok(key) = env::var(env_var) {
        return Ok(key);
    }

    // openai-compatible can run keyless against local servers — return empty
    // string instead of erroring. Caller is responsible for ensuring base_url
    // is configured separately.
    if is_openai_compatible {
        log::debug!(
            "No API key configured for openai-compatible; sending requests without auth header"
        );
        return Ok(String::new());
    }

    Err(anyhow::anyhow!(
        "API key not found for provider '{}'.\n\
         \n\
         Either:\n\
         1. Run 'rfx llm config' to set up your API key interactively\n\
         2. Set REFLEX_AI_API_KEY (works with any provider)\n\
         3. Set the {} environment variable\n\
         \n\
         Example: export REFLEX_AI_API_KEY=sk-...",
        provider,
        env_var
    ))
}

/// Check if any API key is configured for any supported provider
///
/// Checks in priority order:
/// 1. ~/.reflex/config.toml (credentials section)
/// 2. REFLEX_AI_API_KEY environment variable (generic)
/// 3. Provider-specific environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY)
///
/// Returns true if at least one API key is found for any provider.
pub fn is_any_api_key_configured() -> bool {
    // Check user config file first
    if let Ok(Some(user_config)) = load_user_config()
        && let Some(credentials) = &user_config.credentials
    {
        // Check if any provider has an API key in the config file
        if credentials.openai_api_key.is_some()
                || credentials.anthropic_api_key.is_some()
                || credentials.openrouter_api_key.is_some()
                || credentials.openai_compatible_api_key.is_some()
                // openai-compatible can run keyless — a configured base_url
                // counts as "configured" even without an API key.
                || credentials.openai_compatible_base_url.is_some()
        {
            log::debug!("Found provider credential in ~/.reflex/config.toml");
            return true;
        }
    }

    // Check generic REFLEX_AI_API_KEY
    if let Ok(key) = env::var("REFLEX_AI_API_KEY")
        && !key.is_empty()
    {
        log::debug!("Found REFLEX_AI_API_KEY env var");
        return true;
    }

    // Check provider-specific environment variables
    let env_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_COMPATIBLE_API_KEY",
        "OPENAI_COMPATIBLE_BASE_URL",
    ];

    for env_var in &env_vars {
        if env::var(env_var).is_ok() {
            log::debug!("Found {} environment variable", env_var);
            return true;
        }
    }

    log::debug!("No provider credentials found in config or environment variables");
    false
}

/// Get the preferred model for a provider from user config
///
/// Returns None if no model is configured for this provider.
/// The caller should use provider defaults if None is returned.
pub fn get_user_model(provider: &str) -> Option<String> {
    if let Ok(Some(user_config)) = load_user_config()
        && let Some(credentials) = &user_config.credentials
    {
        let model = match provider.to_lowercase().as_str() {
            "openai" => credentials.openai_model.as_ref(),
            "anthropic" => credentials.anthropic_model.as_ref(),
            "openrouter" => credentials.openrouter_model.as_ref(),
            "openai-compatible" | "openai_compatible" => {
                credentials.openai_compatible_model.as_ref()
            }
            _ => None,
        };

        if let Some(model_name) = model {
            log::debug!(
                "Using {} model from ~/.reflex/config.toml: {}",
                provider,
                model_name
            );
            return Some(model_name.clone());
        }
    }

    // Fall back to OPENAI_COMPATIBLE_MODEL env var for the openai-compatible provider
    let provider_lc = provider.to_lowercase();
    if (provider_lc == "openai-compatible" || provider_lc == "openai_compatible")
        && let Ok(model) = env::var("OPENAI_COMPATIBLE_MODEL")
        && !model.is_empty()
    {
        log::debug!(
            "Using openai-compatible model from OPENAI_COMPATIBLE_MODEL env var: {}",
            model
        );
        return Some(model);
    }

    None
}

/// Resolve the effective model for an LLM call.
///
/// Precedence:
///   1. Explicit override (CLI flag, `--model`, `/model` command arg, etc.)
///   2. `[semantic] model` from `~/.reflex/config.toml` (also receives
///      `REFLEX_MODEL` env var via `apply_env_overrides`)
///   3. `[credentials] {provider}_model` via `get_user_model`
///   4. `None` — caller's provider constructor applies its own default
///
/// Returning `None` lets each provider keep its own built-in default
/// (e.g. OpenAI → `gpt-4o-mini`). The openai-compatible provider has no
/// default and will error if `None` is returned, which is the correct
/// behavior for self-hosted endpoints — the fix is to configure a model.
pub fn resolve_model(config: &SemanticConfig, override_model: Option<&str>) -> Option<String> {
    resolve_model_for(&config.provider, config.model.as_deref(), override_model)
}

/// Same as [`resolve_model`] but takes provider/project-model separately.
///
/// Use when the caller has resolved a provider that may not match
/// `semantic_config.provider` — e.g. `pulse/narrate.rs` auto-detects a
/// provider with a working API key when the configured one has none.
pub fn resolve_model_for(
    provider: &str,
    project_model: Option<&str>,
    override_model: Option<&str>,
) -> Option<String> {
    override_model
        .map(String::from)
        .or_else(|| project_model.map(String::from))
        .or_else(|| get_user_model(provider))
}

/// Save user's provider/model preference to ~/.reflex/config.toml
///
/// Updates the [credentials] section with the new model for the specified provider.
/// Creates the config file and directory if they don't exist.
pub fn save_user_provider(provider: &str, model: Option<&str>) -> Result<()> {
    let home = user_home_dir().context("Cannot find home directory")?;
    let config_dir = home.join(".reflex");
    let config_path = config_dir.join("config.toml");

    // Create directory if needed
    std::fs::create_dir_all(&config_dir).context("Failed to create ~/.reflex directory")?;

    // Read existing config or create empty
    let mut config: toml::Value = if config_path.exists() {
        let content = std::fs::read_to_string(&config_path)
            .context("Failed to read ~/.reflex/config.toml")?;
        toml::from_str(&content).context("Failed to parse ~/.reflex/config.toml")?
    } else {
        toml::Value::Table(toml::map::Map::new())
    };

    // Ensure [credentials] section exists
    let credentials = config
        .as_table_mut()
        .context("Config root is not a table")?
        .entry("credentials")
        .or_insert(toml::Value::Table(toml::map::Map::new()))
        .as_table_mut()
        .context("[credentials] is not a table")?;

    // Set model for this provider (if provided)
    if let Some(m) = model {
        let key = format!("{}_model", provider.to_lowercase());
        credentials.insert(key, toml::Value::String(m.to_string()));
        log::info!("Saved {} model: {}", provider, m);
    }

    // Write back to file
    let toml_str = toml::to_string_pretty(&config).context("Failed to serialize config to TOML")?;
    std::fs::write(&config_path, toml_str).context("Failed to write ~/.reflex/config.toml")?;

    Ok(())
}

/// Get provider-specific options from user config
///
/// Returns `Some(HashMap)` for providers that need extra settings (e.g., OpenRouter sort strategy).
/// Returns `None` for providers with no additional options.
pub fn get_provider_options(provider: &str) -> Option<HashMap<String, String>> {
    let provider_lc = provider.to_lowercase();

    match provider_lc.as_str() {
        "openrouter" => {
            if let Ok(Some(user_config)) = load_user_config()
                && let Some(credentials) = &user_config.credentials
                && let Some(sort) = &credentials.openrouter_sort
            {
                let mut opts = HashMap::new();
                opts.insert("sort".to_string(), sort.clone());
                return Some(opts);
            }
            None
        }
        "openai-compatible" | "openai_compatible" => {
            // base_url priority: config file → OPENAI_COMPATIBLE_BASE_URL env var
            let base_url = load_user_config()
                .ok()
                .flatten()
                .and_then(|cfg| cfg.credentials)
                .and_then(|c| c.openai_compatible_base_url)
                .or_else(|| env::var("OPENAI_COMPATIBLE_BASE_URL").ok())
                .filter(|s| !s.is_empty());

            base_url.map(|url| {
                let mut opts = HashMap::new();
                opts.insert("base_url".to_string(), url);
                opts
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};
    use tempfile::TempDir;

    /// Tests in this module manipulate process-wide environment variables
    /// (`HOME`, `OPENAI_API_KEY`, etc.). Cargo runs tests in parallel by
    /// default, which causes races: one test's `env::remove_var("HOME")`
    /// executes mid-flight while another test is reading config from a
    /// `HOME`-rooted path. Acquire this mutex at the start of every test
    /// that touches env state to serialize them. Tests that don't touch
    /// env state can omit it.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Acquire the env-state lock for the duration of a test. Drops on
    /// scope exit, restoring parallelism. Robust to poisoning from a
    /// panicking test (recover instead of propagating).
    fn env_guard() -> MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Point `dirs::home_dir()` at the given path. On Unix that means
    /// `HOME`; on Windows the resolver reads `USERPROFILE` instead, so we
    /// must set the platform-appropriate variable for the override to take
    /// effect.
    fn set_home(path: &std::path::Path) {
        unsafe {
            env::set_var("HOME", path);
            if cfg!(windows) {
                env::set_var("USERPROFILE", path);
            }
        }
    }

    /// Reset the home override applied by [`set_home`].
    fn unset_home() {
        unsafe {
            env::remove_var("HOME");
            if cfg!(windows) {
                env::remove_var("USERPROFILE");
            }
        }
    }

    #[test]
    fn test_default_config() {
        let config = SemanticConfig::default();
        assert!(config.enabled);
        assert_eq!(config.provider, "openai");
        assert_eq!(config.model, None);
        assert!(!config.auto_execute);
    }

    #[test]
    fn test_load_config_no_file() {
        let _g = env_guard();
        let temp = TempDir::new().unwrap();

        // Set HOME to temp directory to avoid loading user's config
        unsafe {
            env::set_var("HOME", temp.path());
        }
        let config = load_config(temp.path()).unwrap();
        unsafe {
            env::remove_var("HOME");
        }

        // Should return defaults
        assert_eq!(config.provider, "openai");
        assert!(config.enabled);
    }

    #[test]
    fn test_load_config_with_semantic_section() {
        let _g = env_guard();
        let temp = TempDir::new().unwrap();
        let reflex_dir = temp.path().join(".reflex");
        std::fs::create_dir_all(&reflex_dir).unwrap();
        let config_path = reflex_dir.join("config.toml");

        std::fs::write(
            &config_path,
            r#"
[semantic]
enabled = true
provider = "anthropic"
model = "claude-3-5-sonnet-20241022"
auto_execute = true
            "#,
        )
        .unwrap();

        // Set HOME to temp directory to load test config
        set_home(temp.path());
        let config = load_config(temp.path()).unwrap();
        unset_home();

        assert!(config.enabled);
        assert_eq!(config.provider, "anthropic");
        assert_eq!(config.model, Some("claude-3-5-sonnet-20241022".to_string()));
        assert!(config.auto_execute);
    }

    #[test]
    fn test_load_config_without_semantic_section() {
        let _g = env_guard();
        let temp = TempDir::new().unwrap();
        let reflex_dir = temp.path().join(".reflex");
        std::fs::create_dir_all(&reflex_dir).unwrap();
        let config_path = reflex_dir.join("config.toml");

        std::fs::write(
            &config_path,
            r#"
[index]
languages = []
            "#,
        )
        .unwrap();

        // Set HOME to temp directory to load test config
        unsafe {
            env::set_var("HOME", temp.path());
        }
        let config = load_config(temp.path()).unwrap();
        unsafe {
            env::remove_var("HOME");
        }

        // Should return defaults
        assert_eq!(config.provider, "openai");
    }

    #[test]
    fn test_get_api_key_env_var() {
        let _g = env_guard();
        let temp = TempDir::new().unwrap();

        // Set HOME to temp directory to avoid loading user's config
        unsafe {
            env::set_var("HOME", temp.path());
            env::set_var("OPENAI_API_KEY", "test-key-123");
        }

        let key = get_api_key("openai").unwrap();
        assert_eq!(key, "test-key-123");

        unsafe {
            env::remove_var("OPENAI_API_KEY");
            env::remove_var("HOME");
        }
    }

    #[test]
    fn test_get_api_key_missing() {
        let _g = env_guard();
        let temp = TempDir::new().unwrap();

        // Set HOME to temp directory to avoid loading user's config
        unsafe {
            env::set_var("HOME", temp.path());
            env::remove_var("OPENROUTER_API_KEY");
            env::remove_var("REFLEX_AI_API_KEY");
        }

        let result = get_api_key("openrouter");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("OPENROUTER_API_KEY")
        );

        unsafe {
            env::remove_var("HOME");
        }
    }

    #[test]
    fn test_get_api_key_unknown_provider() {
        let _g = env_guard();
        let result = get_api_key("unknown");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown provider"));
    }

    #[test]
    fn test_env_override_provider() {
        let _g = env_guard();
        let temp = TempDir::new().unwrap();

        unsafe {
            env::set_var("HOME", temp.path());
            env::set_var("REFLEX_PROVIDER", "openrouter");
        }

        let config = load_config(temp.path()).unwrap();

        unsafe {
            env::remove_var("REFLEX_PROVIDER");
            env::remove_var("HOME");
        }

        assert_eq!(config.provider, "openrouter");
    }

    #[test]
    fn test_env_override_model() {
        let _g = env_guard();
        let temp = TempDir::new().unwrap();

        unsafe {
            env::set_var("HOME", temp.path());
            env::set_var("REFLEX_MODEL", "google/gemini-2.5-flash");
        }

        let config = load_config(temp.path()).unwrap();

        unsafe {
            env::remove_var("REFLEX_MODEL");
            env::remove_var("HOME");
        }

        assert_eq!(config.model, Some("google/gemini-2.5-flash".to_string()));
        // Provider should remain the default since we didn't override it
        assert_eq!(config.provider, "openai");
    }

    #[test]
    fn test_get_api_key_generic_env_var() {
        let _g = env_guard();
        let temp = TempDir::new().unwrap();

        unsafe {
            env::set_var("HOME", temp.path());
            env::remove_var("OPENROUTER_API_KEY");
            env::set_var("REFLEX_AI_API_KEY", "generic-key-456");
        }

        let key = get_api_key("openrouter").unwrap();
        assert_eq!(key, "generic-key-456");

        unsafe {
            env::remove_var("REFLEX_AI_API_KEY");
            env::remove_var("HOME");
        }
    }

    #[test]
    fn test_get_api_key_openai_compatible_returns_empty_when_unset() {
        let _g = env_guard();
        let temp = TempDir::new().unwrap();

        unsafe {
            env::set_var("HOME", temp.path());
            env::remove_var("OPENAI_COMPATIBLE_API_KEY");
            env::remove_var("REFLEX_AI_API_KEY");
        }

        // For openai-compatible, missing key is OK (local servers don't require auth)
        let key = get_api_key("openai-compatible").unwrap();
        assert_eq!(key, "");

        unsafe {
            env::remove_var("HOME");
        }
    }

    #[test]
    fn test_get_provider_options_openai_compatible_from_config() {
        let _g = env_guard();
        let temp = TempDir::new().unwrap();
        let reflex_dir = temp.path().join(".reflex");
        std::fs::create_dir_all(&reflex_dir).unwrap();
        let config_path = reflex_dir.join("config.toml");

        std::fs::write(
            &config_path,
            r#"
[credentials]
openai_compatible_base_url = "http://localhost:1234/v1"
openai_compatible_model = "qwen2.5-coder"
            "#,
        )
        .unwrap();

        unsafe {
            env::remove_var("OPENAI_COMPATIBLE_BASE_URL");
        }
        set_home(temp.path());

        let opts = get_provider_options("openai-compatible");
        let model = get_user_model("openai-compatible");

        unset_home();

        let opts = opts.expect("base_url should be discovered from config");
        assert_eq!(
            opts.get("base_url").map(|s| s.as_str()),
            Some("http://localhost:1234/v1")
        );
        assert_eq!(model, Some("qwen2.5-coder".to_string()));
    }

    #[test]
    fn test_get_provider_options_openai_compatible_from_env() {
        let _g = env_guard();
        let temp = TempDir::new().unwrap();

        unsafe {
            env::set_var("HOME", temp.path());
            env::set_var("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:11434/v1");
        }

        let opts = get_provider_options("openai-compatible");

        unsafe {
            env::remove_var("OPENAI_COMPATIBLE_BASE_URL");
            env::remove_var("HOME");
        }

        let opts = opts.expect("base_url should be discovered from env var");
        assert_eq!(
            opts.get("base_url").map(|s| s.as_str()),
            Some("http://localhost:11434/v1")
        );
    }

    fn config_with(provider: &str, project_model: Option<&str>) -> SemanticConfig {
        SemanticConfig {
            provider: provider.to_string(),
            model: project_model.map(String::from),
            ..SemanticConfig::default()
        }
    }

    #[test]
    fn resolve_model_prefers_override() {
        let config = config_with("openai", Some("gpt-4o"));
        let resolved = resolve_model(&config, Some("gpt-4o-2024-08-06"));
        assert_eq!(resolved.as_deref(), Some("gpt-4o-2024-08-06"));
    }

    #[test]
    fn resolve_model_falls_back_to_project_config() {
        let config = config_with("openai", Some("gpt-4o"));
        let resolved = resolve_model(&config, None);
        assert_eq!(resolved.as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn resolve_model_returns_none_when_unset() {
        let _g = env_guard();
        // No override, no [semantic] model, no [credentials] entry — caller
        // is expected to fall back to the provider's own default.
        let temp = TempDir::new().unwrap();
        unsafe {
            env::set_var("HOME", temp.path());
        }

        let config = config_with("openai", None);
        let resolved = resolve_model(&config, None);

        unsafe {
            env::remove_var("HOME");
        }

        assert_eq!(resolved, None);
    }

    #[test]
    fn resolve_model_for_openai_compatible_reads_user_config() {
        let _g = env_guard();
        // The actual bug repro at the unit level: model lives in
        // ~/.reflex/config.toml [credentials] openai_compatible_model and
        // resolve_model_for must surface it when override + project are None.
        let temp = TempDir::new().unwrap();
        let reflex_dir = temp.path().join(".reflex");
        std::fs::create_dir_all(&reflex_dir).unwrap();
        std::fs::write(
            reflex_dir.join("config.toml"),
            r#"
[credentials]
openai_compatible_model = "gpt-oss:20b-cloud"
            "#,
        )
        .unwrap();

        set_home(temp.path());

        let resolved = resolve_model_for("openai-compatible", None, None);

        unset_home();

        assert_eq!(resolved.as_deref(), Some("gpt-oss:20b-cloud"));
    }

    #[test]
    fn resolve_model_for_override_beats_user_config() {
        let _g = env_guard();
        let temp = TempDir::new().unwrap();
        let reflex_dir = temp.path().join(".reflex");
        std::fs::create_dir_all(&reflex_dir).unwrap();
        std::fs::write(
            reflex_dir.join("config.toml"),
            r#"
[credentials]
openrouter_model = "anthropic/claude-opus-4"
            "#,
        )
        .unwrap();

        unsafe {
            env::set_var("HOME", temp.path());
        }

        let resolved = resolve_model_for("openrouter", None, Some("openai/gpt-4o"));

        unsafe {
            env::remove_var("HOME");
        }

        assert_eq!(resolved.as_deref(), Some("openai/gpt-4o"));
    }
}
