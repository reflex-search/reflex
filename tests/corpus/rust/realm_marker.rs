//! Shared-token fixture: `realm_marker` appears here AND in tests/corpus/text/.

/// A tenant boundary identifier.
pub struct RealmConfig {
    pub realm_marker: String,
    pub jwks_rps_limit: u32,
}

pub fn read_realm_marker(cfg: &RealmConfig) -> &str {
    &cfg.realm_marker
}
