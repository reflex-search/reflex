//! Pulse configuration types
//!
//! Configuration for snapshot retention, threshold alerts, and generation options.
//! Settings are loaded from the `[pulse]` section of `.reflex/config.toml`.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level Pulse configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PulseConfig {
    #[serde(default)]
    pub retention: RetentionConfig,
    #[serde(default)]
    pub thresholds: ThresholdConfig,
}

impl Default for PulseConfig {
    fn default() -> Self {
        Self {
            retention: RetentionConfig::default(),
            thresholds: ThresholdConfig::default(),
        }
    }
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

fn default_daily() -> usize { 7 }
fn default_weekly() -> usize { 4 }
fn default_monthly() -> usize { 12 }

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

fn default_fan_in_warning() -> usize { 10 }
fn default_fan_in_critical() -> usize { 25 }
fn default_cycle_length() -> usize { 3 }
fn default_module_file_count() -> usize { 50 }
fn default_line_count_growth() -> f64 { 2.0 }

/// Load Pulse configuration from the project's `.reflex/config.toml`
///
/// Falls back to defaults if the `[pulse]` section is missing.
pub fn load_pulse_config(cache_path: &Path) -> Result<PulseConfig> {
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
    }
}
