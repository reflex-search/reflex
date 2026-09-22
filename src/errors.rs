use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReflexError {
    #[error("Index not found. Run 'rfx index' to build the search index.")]
    IndexNotFound,

    #[error("Query syntax error: {0}")]
    QuerySyntaxError(String),

    #[error("I/O error: {0}")]
    IoError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("LLM error: {0}")]
    LlmError(String),

    /// A tool/API call carried an argument set the server cannot accept:
    /// unknown key, wrong type, or a required key that is absent. The payload
    /// is the full human-readable diagnostic (received keys, valid keys,
    /// nearest-match suggestion). Maps to JSON-RPC `-32602` in the MCP layer.
    #[error("{0}")]
    InvalidParams(String),

    /// The on-disk cache failed structural validation (bad magic bytes, short
    /// file, broken SQLite). The payload is the inner finding, e.g.
    /// `content.bin is too small - appears to be corrupted`. The Display text
    /// keeps the historical wording so string-matching callers stay valid.
    #[error(
        "Cache appears to be corrupted: {0}. Run 'rfx clear' followed by 'rfx index' to rebuild."
    )]
    CacheCorrupted(String),

    /// Another process holds the workspace index lock (`.reflex/index.lock`).
    /// The payload names the lock path.
    #[error(
        "Another indexer is already running on this workspace ({0}). Wait for it to finish and retry."
    )]
    IndexLocked(String),

    /// The detached background symbol pass (`rfx index-symbols-internal`) holds
    /// `meta.db`. Raised BEFORE any SQLite call, so the agent sees progress instead
    /// of `database is locked: Error code 5`, which is what 1.7.0 surfaced for the
    /// four minutes a 1027-file pass was running.
    #[error(
        "symbol indexing in progress (pid {pid}, started {started_at}, {processed}/{total} files)"
    )]
    SymbolIndexingInProgress {
        pid: u32,
        /// Wall-clock start time, `HH:MM:SS`.
        started_at: String,
        processed: usize,
        total: usize,
    },

    /// This `.reflex/` was written by a different Reflex build.
    ///
    /// Deliberately NOT `CacheCorrupted`: the MCP layer force-rebuilds on corruption,
    /// so classifying a version mismatch as corruption makes N servers at differing
    /// versions rebuild the same cache concurrently. That stampede is what produced
    /// `content.bin is too small` in the field.
    #[error(
        "this .reflex/ was written by reflex {owner_version}{owner_sha}; this binary is {this_version}. \
         Rebuild with force, or run the matching binary."
    )]
    CacheVersionMismatch {
        owner_version: String,
        /// Pre-formatted as ` (sha abc1234)`, or empty when unknown.
        owner_sha: String,
        this_version: String,
    },
}

impl ReflexError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::IndexNotFound => "IndexNotFound",
            Self::QuerySyntaxError(_) => "QuerySyntaxError",
            Self::IoError(_) => "IoError",
            Self::ParseError(_) => "ParseError",
            Self::LlmError(_) => "LlmError",
            Self::InvalidParams(_) => "InvalidParams",
            Self::CacheCorrupted(_) => "CacheCorrupted",
            Self::IndexLocked(_) => "IndexLocked",
            Self::SymbolIndexingInProgress { .. } => "SymbolIndexingInProgress",
            Self::CacheVersionMismatch { .. } => "CacheVersionMismatch",
        }
    }

    pub fn exit_code(&self) -> i32 {
        match self {
            Self::IndexNotFound => 2,
            Self::QuerySyntaxError(_) => 3,
            Self::IoError(_) => 4,
            Self::ParseError(_) => 5,
            Self::LlmError(_) => 6,
            Self::InvalidParams(_) => 3,
            Self::CacheCorrupted(_) => 2,
            Self::IndexLocked(_) => 7,
            Self::SymbolIndexingInProgress { .. } => 7,
            Self::CacheVersionMismatch { .. } => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exit_codes() {
        assert_eq!(ReflexError::IndexNotFound.exit_code(), 2);
        assert_eq!(ReflexError::QuerySyntaxError("bad".into()).exit_code(), 3);
        assert_eq!(ReflexError::IoError("fail".into()).exit_code(), 4);
        assert_eq!(ReflexError::ParseError("oops".into()).exit_code(), 5);
        assert_eq!(ReflexError::LlmError("timeout".into()).exit_code(), 6);
        assert_eq!(ReflexError::InvalidParams("bad".into()).exit_code(), 3);
        assert_eq!(ReflexError::CacheCorrupted("x".into()).exit_code(), 2);
        assert_eq!(ReflexError::IndexLocked("x".into()).exit_code(), 7);
    }

    #[test]
    fn test_new_variant_kinds_and_display() {
        assert_eq!(
            ReflexError::InvalidParams("x".into()).kind(),
            "InvalidParams"
        );
        assert_eq!(
            ReflexError::CacheCorrupted("x".into()).kind(),
            "CacheCorrupted"
        );
        assert_eq!(ReflexError::IndexLocked("x".into()).kind(), "IndexLocked");
        // Display text of CacheCorrupted must keep the historical wording.
        let msg = ReflexError::CacheCorrupted(
            "content.bin is too small - appears to be corrupted".into(),
        )
        .to_string();
        assert_eq!(
            msg,
            "Cache appears to be corrupted: content.bin is too small - appears to be corrupted. \
             Run 'rfx clear' followed by 'rfx index' to rebuild."
        );
        assert_eq!(
            ReflexError::InvalidParams("Unknown argument \"q\"".into()).to_string(),
            "Unknown argument \"q\""
        );
    }

    #[test]
    fn test_kind_strings() {
        assert_eq!(ReflexError::IndexNotFound.kind(), "IndexNotFound");
        assert_eq!(
            ReflexError::QuerySyntaxError("x".into()).kind(),
            "QuerySyntaxError"
        );
        assert_eq!(ReflexError::IoError("x".into()).kind(), "IoError");
        assert_eq!(ReflexError::ParseError("x".into()).kind(), "ParseError");
        assert_eq!(ReflexError::LlmError("x".into()).kind(), "LlmError");
    }

    #[test]
    fn test_mcp_json_error_shape() {
        let err = ReflexError::IndexNotFound;
        let kind = err.kind();
        let message = err.to_string();
        let json_data = serde_json::json!({ "kind": kind, "message": message });

        assert_eq!(json_data["kind"], "IndexNotFound");
        assert!(json_data["message"].as_str().unwrap().contains("rfx index"));
    }

    #[test]
    fn test_http_json_error_shape() {
        let err = ReflexError::QuerySyntaxError("invalid pattern".into());
        let kind = err.kind();
        let msg = err.to_string();
        let body = serde_json::json!({ "error": { "kind": kind, "message": msg } });

        assert_eq!(body["error"]["kind"], "QuerySyntaxError");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("invalid pattern")
        );
    }

    #[test]
    fn test_anyhow_downcast() {
        let err: anyhow::Error = ReflexError::IndexNotFound.into();
        let downcasted = err.downcast_ref::<ReflexError>().unwrap();
        assert_eq!(downcasted.exit_code(), 2);
        assert_eq!(downcasted.kind(), "IndexNotFound");
    }

    #[test]
    fn test_non_reflex_error_fallback() {
        let err = anyhow::anyhow!("some other error");
        let exit_code = if let Some(re) = err.downcast_ref::<ReflexError>() {
            re.exit_code()
        } else {
            1
        };
        assert_eq!(
            exit_code, 1,
            "Non-ReflexError should fall back to exit code 1"
        );
    }
}
