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
}

impl ReflexError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::IndexNotFound => "IndexNotFound",
            Self::QuerySyntaxError(_) => "QuerySyntaxError",
            Self::IoError(_) => "IoError",
            Self::ParseError(_) => "ParseError",
            Self::LlmError(_) => "LlmError",
        }
    }

    pub fn exit_code(&self) -> i32 {
        match self {
            Self::IndexNotFound => 2,
            Self::QuerySyntaxError(_) => 3,
            Self::IoError(_) => 4,
            Self::ParseError(_) => 5,
            Self::LlmError(_) => 6,
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
