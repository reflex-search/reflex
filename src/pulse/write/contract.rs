//! The output contract of grounded sections: sentences with citations.
//!
//! ```json
//! {"status": "ok", "missing": [],
//!  "paragraphs": [{"sentences": [{"text": "…", "cite": ["F1", "F3"]}]}]}
//! ```
//! `status` is `insufficient_evidence` when the facts do not support a useful section;
//! `missing` then says what would. Every field is required, so providers can enforce
//! the schema strictly.

use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Contract {
    pub status: String,
    #[serde(default)]
    pub missing: Vec<String>,
    #[serde(default)]
    pub paragraphs: Vec<Paragraph>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Paragraph {
    pub sentences: Vec<Sentence>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sentence {
    pub text: String,
    #[serde(default)]
    pub cite: Vec<String>,
}

/// The JSON schema of [`Contract`].
pub fn schema() -> serde_json::Value {
    let sentence = json!({
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "cite": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["text", "cite"],
        "additionalProperties": false
    });
    let paragraph = json!({
        "type": "object",
        "properties": {"sentences": {"type": "array", "items": sentence}},
        "required": ["sentences"],
        "additionalProperties": false
    });
    json!({
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "insufficient_evidence"]},
            "missing": {"type": "array", "items": {"type": "string"}},
            "paragraphs": {"type": "array", "items": paragraph}
        },
        "required": ["status", "missing", "paragraphs"],
        "additionalProperties": false
    })
}

/// Parse a response: plain JSON, fenced JSON, or JSON inside prose.
pub fn parse(text: &str) -> anyhow::Result<Contract> {
    let t = text.trim();
    let body = if let (Some(a), Some(b)) = (t.find('{'), t.rfind('}')) {
        &t[a..=b]
    } else {
        t
    };
    let mut c: Contract = serde_json::from_str(body)?;
    c.status = c.status.trim().to_ascii_lowercase().replace('-', "_");
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_fenced_and_wrapped() {
        let j = r#"{"status":"ok","missing":[],"paragraphs":[{"sentences":[{"text":"A.","cite":["F1"]}]}]}"#;
        assert_eq!(
            parse(j).unwrap().paragraphs[0].sentences[0].cite,
            vec!["F1"]
        );
        assert!(parse(&format!("```json\n{j}\n```")).is_ok());
        assert!(parse(&format!("Here you go:\n{j}\nThanks")).is_ok());
        assert_eq!(
            parse(r#"{"status":"Insufficient-Evidence"}"#)
                .unwrap()
                .status,
            "insufficient_evidence"
        );
        assert!(parse("no json").is_err());
    }

    #[test]
    fn schema_is_strict() {
        let s = schema();
        assert_eq!(s["additionalProperties"], false);
        assert_eq!(s["required"].as_array().unwrap().len(), 3);
    }
}
