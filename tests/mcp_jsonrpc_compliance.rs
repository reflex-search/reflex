//! JSON-RPC 2.0 conformance tests for the MCP stdio server.
//!
//! Regression tests for a bug where the server emitted `"id": null` and
//! responded to Notifications, both of which violate JSON-RPC 2.0 and are
//! rejected by strict clients (e.g. Claude Code's Zod validators).

use reflex::mcp::run_mcp_server_io;
use serde_json::Value;
use std::collections::HashSet;
use std::io::Cursor;

/// Run a sequence of JSON-RPC input lines through the server and return the
/// emitted output lines (one per response).
fn run_exchange(input_lines: &[&str]) -> Vec<String> {
    let input = input_lines.join("\n");
    let reader = Cursor::new(input.into_bytes());
    let mut output: Vec<u8> = Vec::new();

    run_mcp_server_io(reader, &mut output).expect("server should not error");

    String::from_utf8(output)
        .expect("output is valid UTF-8")
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect()
}

/// Assert a response message satisfies the structural rules every emitted
/// JSON-RPC message must follow.
fn assert_response_well_formed(line: &str) {
    let v: Value = serde_json::from_str(line).expect("valid JSON");
    let obj = v.as_object().expect("top-level object");

    // Only allowed top-level keys.
    let allowed: HashSet<&str> = ["jsonrpc", "id", "result", "error"].into_iter().collect();
    for key in obj.keys() {
        assert!(allowed.contains(key.as_str()), "unrecognized key: {}", key);
    }

    // jsonrpc must be exactly "2.0".
    assert_eq!(obj.get("jsonrpc").and_then(|v| v.as_str()), Some("2.0"));

    // id must be present and either a string or a number — never null.
    let id = obj.get("id").expect("response must have id");
    assert!(
        id.is_string() || id.is_number(),
        "id must be string or number, got: {}",
        id
    );

    // Exactly one of result / error must be present.
    let has_result = obj.contains_key("result");
    let has_error = obj.contains_key("error");
    assert!(
        has_result ^ has_error,
        "exactly one of result/error must be set"
    );
}

#[test]
fn initialize_round_trip() {
    let init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}"#;
    let out = run_exchange(&[init]);

    assert_eq!(out.len(), 1, "expected exactly one response");
    assert_response_well_formed(&out[0]);

    let v: Value = serde_json::from_str(&out[0]).unwrap();
    assert_eq!(v["id"], Value::from(1));
    assert_eq!(v["result"]["protocolVersion"], "2024-11-05");
    assert!(v["result"]["serverInfo"]["name"].is_string());
}

#[test]
fn notifications_initialized_is_silent() {
    let notif = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
    let out = run_exchange(&[notif]);

    assert!(
        out.is_empty(),
        "server must not respond to notifications, got: {:?}",
        out
    );
}

#[test]
fn unknown_notification_is_silent() {
    let notif = r#"{"jsonrpc":"2.0","method":"notifications/some_unknown"}"#;
    let out = run_exchange(&[notif]);

    assert!(
        out.is_empty(),
        "server must not respond to any notification, got: {:?}",
        out
    );
}

#[test]
fn unknown_method_on_request_returns_error_with_matching_id() {
    let req = r#"{"jsonrpc":"2.0","id":42,"method":"totally/unknown","params":{}}"#;
    let out = run_exchange(&[req]);

    assert_eq!(out.len(), 1);
    assert_response_well_formed(&out[0]);

    let v: Value = serde_json::from_str(&out[0]).unwrap();
    assert_eq!(v["id"], Value::from(42));
    assert!(v["error"].is_object());
    assert!(v["error"]["message"].is_string());
}

#[test]
fn full_handshake_does_not_emit_id_null() {
    let lines = [
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}"#,
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
    ];
    let out = run_exchange(&lines);

    // Expect exactly two responses: one for initialize (id=1), one for tools/list (id=2).
    assert_eq!(
        out.len(),
        2,
        "expected 2 responses (no response for the notification), got: {:?}",
        out
    );

    for line in &out {
        assert!(
            !line.contains("\"id\":null"),
            "response must never contain id: null — got {}",
            line
        );
        assert_response_well_formed(line);
    }

    let r0: Value = serde_json::from_str(&out[0]).unwrap();
    let r1: Value = serde_json::from_str(&out[1]).unwrap();
    assert_eq!(r0["id"], Value::from(1));
    assert_eq!(r1["id"], Value::from(2));
}

#[test]
fn string_id_is_preserved() {
    let req =
        r#"{"jsonrpc":"2.0","id":"req-abc","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}"#;
    let out = run_exchange(&[req]);

    assert_eq!(out.len(), 1);
    let v: Value = serde_json::from_str(&out[0]).unwrap();
    assert_eq!(v["id"], Value::from("req-abc"));
    assert_response_well_formed(&out[0]);
}
