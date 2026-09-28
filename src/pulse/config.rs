//! Pulse configuration types
//!
//! Configuration for snapshot retention, threshold alerts, and generation options.
//! Settings are loaded from the `[pulse]` section of `.reflex/config.toml`.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level Pulse configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PulseConfig {
    #[serde(default)]
    pub retention: RetentionConfig,
    #[serde(default)]
    pub thresholds: ThresholdConfig,
    #[serde(default)]
    pub write: WriteSettings,
    #[serde(default)]
    pub docs: DocsSettings,
}

/// `[pulse.docs]`: what the Docs tab documents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocsSettings {
    /// Document library APIs (default: true). Set false for a CLI-first project.
    #[serde(default = "default_true")]
    pub library: bool,
    /// Only these module paths and their children (`reflex::query`). Empty = all public.
    #[serde(default)]
    pub include: Vec<String>,
}

impl Default for DocsSettings {
    fn default() -> Self {
        Self {
            library: true,
            include: Vec::new(),
        }
    }
}

fn default_true() -> bool {
    true
}

/// `[pulse.write]`: the LLM writing pass.
///
/// CLI flags override these. Provider credentials stay in `~/.reflex/config.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteSettings {
    /// Provider for Pulse only (default: the `[semantic]` provider used by `rfx ask`).
    #[serde(default)]
    pub provider: Option<String>,
    /// Model for Pulse only. Docs usually deserve a stronger model than query generation.
    #[serde(default)]
    pub model: Option<String>,
    /// Cache directory, relative to the workspace root (default: `.reflex/pulse/write-cache`).
    /// Entries hold no timestamps, so the directory is safe to commit or keep in CI caches.
    #[serde(default)]
    pub cache_dir: Option<String>,
    /// Successful runs whose cache entries survive pruning (default: 3).
    #[serde(default = "default_keep_runs")]
    pub keep_runs: usize,
    /// Leave provider and model out of the cache key, so switching models keeps old text.
    #[serde(default)]
    pub cache_model_agnostic: bool,
    /// Concurrent LLM calls (default: 4).
    #[serde(default = "default_write_concurrency")]
    pub concurrency: usize,
    /// Stop planning calls once estimated input+output tokens reach this cap.
    #[serde(default)]
    pub max_llm_tokens: Option<u64>,
}

impl Default for WriteSettings {
    fn default() -> Self {
        Self {
            provider: None,
            model: None,
            cache_dir: None,
            keep_runs: default_keep_runs(),
            cache_model_agnostic: false,
            concurrency: default_write_concurrency(),
            max_llm_tokens: None,
        }
    }
}

fn default_keep_runs() -> usize {
    3
}
fn default_write_concurrency() -> usize {
    4
}

/// Snapshot retention policy
///
/// Controls how many snapshots are kept at each granularity level.
/// Under steady state with defaults: ~23 snapshots total.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionConfig {
    /// Number of daily snapshots to keep (default: 7)
    #[serde(default = "default_daily")]
    pub daily: usize,
    /// Number of weekly snapshots to keep (default: 4)
    #[serde(default = "default_weekly")]
    pub weekly: usize,
    /// Number of monthly snapshots to keep (default: 12)
    #[serde(default = "default_monthly")]
    pub monthly: usize,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            daily: default_daily(),
            weekly: default_weekly(),
            monthly: default_monthly(),
        }
    }
}

fn default_daily() -> usize {
    7
}
fn default_weekly() -> usize {
    4
}
fn default_monthly() -> usize {
    12
}

/// Threshold configuration for structural alerts
///
/// When metrics cross these thresholds, Pulse generates alerts in digests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdConfig {
    /// Fan-in warning threshold (default: 10)
    #[serde(default = "default_fan_in_warning")]
    pub fan_in_warning: usize,
    /// Fan-in critical threshold (default: 25)
    #[serde(default = "default_fan_in_critical")]
    pub fan_in_critical: usize,
    /// Minimum cycle length to flag (default: 3)
    #[serde(default = "default_cycle_length")]
    pub cycle_length: usize,
    /// Module file count warning (default: 50)
    #[serde(default = "default_module_file_count")]
    pub module_file_count: usize,
    /// Line count growth multiplier warning (default: 2.0)
    #[serde(default = "default_line_count_growth")]
    pub line_count_growth: f64,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            fan_in_warning: default_fan_in_warning(),
            fan_in_critical: default_fan_in_critical(),
            cycle_length: default_cycle_length(),
            module_file_count: default_module_file_count(),
            line_count_growth: default_line_count_growth(),
        }
    }
}

fn default_fan_in_warning() -> usize {
    10
}
fn default_fan_in_critical() -> usize {
    25
}
fn default_cycle_length() -> usize {
    3
}
fn default_module_file_count() -> usize {
    50
}
fn default_line_count_growth() -> f64 {
    2.0
}

/// Load Pulse configuration.
///
/// A `pulse.toml` at the workspace root (next to `.reflex/`) wins: it can be committed,
/// while `.reflex/` is usually ignored. It uses the same sections without the `pulse.`
/// prefix (`[docs]`, `[write]`). Otherwise the `[pulse]` section of
/// `.reflex/config.toml` is used, else defaults.
pub fn load_pulse_config(cache_path: &Path) -> Result<PulseConfig> {
    if let Some(root) = cache_path.parent() {
        let committed = root.join("pulse.toml");
        if committed.exists() {
            let content = std::fs::read_to_string(&committed)?;
            return Ok(toml::from_str(&content)?);
        }
    }
    let config_path = cache_path.join("config.toml");

    if !config_path.exists() {
        return Ok(PulseConfig::default());
    }

    let content = std::fs::read_to_string(&config_path)?;
    let table: toml::Value = content.parse()?;

    if let Some(pulse_section) = table.get("pulse") {
        let config: PulseConfig = pulse_section.clone().try_into()?;
        Ok(config)
    } else {
        Ok(PulseConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PulseConfig::default();
        assert_eq!(config.retention.daily, 7);
        assert_eq!(config.retention.weekly, 4);
        assert_eq!(config.retention.monthly, 12);
        assert_eq!(config.thresholds.fan_in_warning, 10);
        assert_eq!(config.thresholds.fan_in_critical, 25);
        assert_eq!(config.thresholds.cycle_length, 3);
        assert_eq!(config.thresholds.module_file_count, 50);
        assert!((config.thresholds.line_count_growth - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_missing_config() {
        let config = load_pulse_config(Path::new("/nonexistent")).unwrap();
        assert_eq!(config.retention.daily, 7);
    }

    #[test]
    fn test_deserialize_partial_config() {
        let toml_str = r#"
            [pulse.retention]
            daily = 14
        "#;
        let table: toml::Value = toml_str.parse().unwrap();
        let pulse_section = table.get("pulse").unwrap();
        let config: PulseConfig = pulse_section.clone().try_into().unwrap();
        assert_eq!(config.retention.daily, 14);
        assert_eq!(config.retention.weekly, 4); // default
        assert_eq!(config.thresholds.fan_in_warning, 10); // default
        assert_eq!(config.write.keep_runs, 3); // default
    }

    #[test]
    fn test_root_pulse_toml_wins() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join(".reflex")).unwrap();
        std::fs::write(
            dir.path().join(".reflex/config.toml"),
            "[pulse.docs]\nlibrary = true\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("pulse.toml"),
            "[docs]\nlibrary = false\ninclude = [\"x::y\"]\n",
        )
        .unwrap();
        let c = load_pulse_config(&dir.path().join(".reflex")).unwrap();
        assert!(!c.docs.library);
        assert_eq!(c.docs.include, vec!["x::y"]);
        assert_eq!(c.retention.daily, 7);
    }

    #[test]
    fn test_deserialize_write_settings() {
        let toml_str = r#"
            [pulse.write]
            model = "claude-sonnet-5"
            cache_dir = "docs/.pulse-cache"
            max_llm_tokens = 200000
        "#;
        let table: toml::Value = toml_str.parse().unwrap();
        let config: PulseConfig = table.get("pulse").unwrap().clone().try_into().unwrap();
        assert_eq!(config.write.model.as_deref(), Some("claude-sonnet-5"));
        assert_eq!(config.write.cache_dir.as_deref(), Some("docs/.pulse-cache"));
        assert_eq!(config.write.max_llm_tokens, Some(200_000));
        assert_eq!(config.write.concurrency, 4);
        assert!(!config.write.cache_model_agnostic);
    }
}
